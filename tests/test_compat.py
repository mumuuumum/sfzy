"""models/compat.py 的验收测试。

这些补丁是给特定模型的远程代码填坑的，逻辑必须锁住。尤其是
`all_tied_weights_keys` **必须是 dict 不是 list** —— 这个类型错误
在第一处调用（`len(...) > 0`）看不出来，只会在 transformers 内部
第二处（`.keys()`）才暴露，而那时已经烧了一轮 Kaggle 时间。
"""

from __future__ import annotations

from transformers import PreTrainedModel

from sfzy.models.compat import (
    ensure_legacy_cache,
    ensure_tp_plan,
    patch_pretrained_config,
    patch_tied_weights_keys,
    patch_tp_plan_for_quantized_load,
    resolve_padding_side,
)


class FakeConfig:
    """模拟 ChatGLM3 的 config：只有 seq_length，没有 max_length / use_cache。"""

    def __init__(self, **kwargs):
        self.seq_length = 8192
        for key, value in kwargs.items():
            setattr(self, key, value)


# ---------------------------------------------------------------- tied weights

def test_tied_weights_keys_必须是dict不是list():
    """必须是 dict —— accelerate 那边会调 .keys()。

    填 [] 能通过 'len(...) > 0' 那处，但会在
    'getattr(model, "...", {}).keys()' 那处报
    'list' object has no attribute 'keys'。
    """
    patch_tied_weights_keys()
    value = PreTrainedModel.all_tied_weights_keys
    assert isinstance(value, dict), "必须是 dict，不能是 list"
    assert value == {}


def test_tied_weights_keys_重复调用不报错():
    patch_tied_weights_keys()
    patch_tied_weights_keys()
    assert isinstance(PreTrainedModel.all_tied_weights_keys, dict)


# ---------------------------------------------------------------- config 回填

def test_回填_generation_属性():
    config = FakeConfig()
    patch_pretrained_config(config)
    assert config.use_cache is True
    assert config.num_beams == 1
    assert config.temperature == 1.0


def test_max_length_取seq_length而不是generation默认值():
    """generation 的默认 max_length 是 20。

    直接用会让 ChatGLM3 的 self.max_sequence_length 变成 20，
    那模型就只能处理 20 个 token —— 而且不报错。
    """
    config = FakeConfig()
    patch_pretrained_config(config)
    assert config.max_length == 8192, "应该取 seq_length，不是 20"


def test_已有max_length时不覆盖():
    config = FakeConfig(max_length=4096)
    patch_pretrained_config(config)
    assert config.max_length == 4096


def test_已有use_cache时不覆盖():
    config = FakeConfig(use_cache=False)
    patch_pretrained_config(config)
    assert config.use_cache is False


def test_原始config没变_只加字段():
    """回填不能破坏模型自己定义的字段。"""
    config = FakeConfig(hidden_size=4096, num_layers=28)
    patch_pretrained_config(config)
    assert config.hidden_size == 4096
    assert config.num_layers == 28


# ---------------------------------------------------------------- 显式补丁

def test_config_patches_显式补丁生效():
    config = FakeConfig()
    patch_pretrained_config(config, {"config_patches": {"custom_field": 42}})
    assert config.custom_field == 42


def test_config_patches_不覆盖已有字段():
    config = FakeConfig(custom_field=1)
    patch_pretrained_config(config, {"config_patches": {"custom_field": 42}})
    assert config.custom_field == 1


def test_config_patches_缺省不报错():
    patch_pretrained_config(FakeConfig())
    patch_pretrained_config(FakeConfig(), {})


# ---------------------------------------------------------------- padding 方向

def test_padding_side_默认左():
    """decoder-only 的批量生成必须左 padding。"""
    assert resolve_padding_side({}) == "left"


def test_padding_side_可被配置覆盖():
    assert resolve_padding_side({"padding_side": "right"}) == "right"


# ------------------------------------------------------- 没有 TP plan 的模型
#
# 这一组对应"单进程能加载、一加 DDP 立刻崩"的那个坑：
# 新版 transformers 的 get_total_byte_count 只在 **torch.distributed 已初始化** 时
# 去读 model.tp_plan，ChatGLM3 没有这个属性 → len(None) 报 TypeError。

class FakeModelWithoutTpPlan:
    """模拟 ChatGLM3：属性存在但是 None（新版的默认值）。"""

    def __init__(self):
        self.all_tied_weights_keys = {}
        self.tp_plan = None


class FakeModelWithReadOnlyTpPlan:
    """最坏情况：tp_plan 是只读 property，实例赋值会抛 AttributeError。"""

    def __init__(self):
        self._tp_plan = None

    @property
    def tp_plan(self):
        return self._tp_plan


def test_ensure_tp_plan_把None换成dict():
    model = FakeModelWithoutTpPlan()
    assert ensure_tp_plan(model) is True
    assert model.tp_plan == {}


def test_ensure_tp_plan_对只读property也能兜住():
    model = FakeModelWithReadOnlyTpPlan()
    assert ensure_tp_plan(model) is True
    assert model.tp_plan == {}


def test_ensure_tp_plan_已有计划时不覆盖():
    class Model:
        tp_plan = {"layers.0.fc1": "colwise"}

    model = Model()
    assert ensure_tp_plan(model) is True
    assert model.tp_plan == {"layers.0.fc1": "colwise"}


def test_补丁让没有tp_plan的模型能过显存预热(monkeypatch):
    """重现新版 transformers 在 DDP 下的那一行，验证补丁真的能救活它。"""
    import transformers.modeling_utils as modeling_utils

    calls = []

    def fake_get_total_byte_count(model, accelerator_device_map, hf_quantizer=None):
        # 就是新版 modeling_utils.py 里那一行（去掉 is_initialized 判断，
        # 直接走 DDP 分支）
        tp_plan = model.tp_plan
        calls.append(tp_plan)
        return {"total": 0} if len(tp_plan) == 0 else {"total": 1}

    monkeypatch.setattr(modeling_utils, "get_total_byte_count",
                        fake_get_total_byte_count, raising=False)

    model = FakeModelWithoutTpPlan()
    # 补丁之前：复现线上那个 TypeError
    try:
        modeling_utils.get_total_byte_count(model, {"w": "cuda:0"}, None)
    except TypeError as exc:
        assert "NoneType" in str(exc)
    else:  # pragma: no cover - 说明复现失败，测试本身有问题
        raise AssertionError("应该复现出 len(None) 的 TypeError")

    assert patch_tp_plan_for_quantized_load() is True
    assert modeling_utils.get_total_byte_count(model, {"w": "cuda:0"}, None) == {"total": 0}
    assert calls[-1] == {}, "补丁应该把 tp_plan 补成空计划"


def test_补丁重复调用不会层层套娃(monkeypatch):
    import transformers.modeling_utils as modeling_utils

    monkeypatch.setattr(modeling_utils, "get_total_byte_count",
                        lambda *a, **k: {}, raising=False)
    assert patch_tp_plan_for_quantized_load() is True
    inner = modeling_utils.get_total_byte_count
    assert patch_tp_plan_for_quantized_load() is False
    assert modeling_utils.get_total_byte_count is inner


def test_老版transformers没有这个函数时补丁是空操作():
    """本机 4.57.6 就没有 get_total_byte_count，patch 必须安静地返回 False。"""
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "get_total_byte_count"):  # pragma: no cover
        import pytest

        pytest.skip("当前 transformers 有该函数，这个用例只在旧版上有意义")
    assert patch_tp_plan_for_quantized_load() is False


# ---------------------------------------------------------------- 梯度检查点
#
# 这三个测试锁住的是一个**花了三轮才定位到**的坑：
# 配置里写了 gradient_checkpointing: true，模型上一个标志都没设，
# 于是显存按"每层完整保存"涨，最后 OOM，而报错栈里全是别的东西。
#
# 根因：ChatGLM3 的 gradient_checkpointing_enable 是空实现 ——
# 它只检查 supports 标志，然后什么都不做。调用它不报错、也不生效。
# 真正的开关在 GLMTransformer（model.transformer.encoder）上。

import torch.nn as nn  # noqa: E402

from sfzy.models.compat import ensure_gradient_checkpointing  # noqa: E402


class _FakeGLMTransformer(nn.Module):
    """ChatGLM3 的 GLMTransformer：开关定义在它自己身上。"""

    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False


class _FakeChatGLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeGLMTransformer()


class _FakeChatGLM(nn.Module):
    """按 ChatGLM3 的真实结构搭：model.transformer.encoder.gradient_checkpointing。

    gradient_checkpointing_enable 是空实现 —— 这正是坑的来源，
    所以这里按真实行为复刻，不"修好"它。
    """

    def __init__(self):
        super().__init__()
        self.transformer = _FakeChatGLMModel()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return None


def test_ChatGLM3的gradient_checkpointing_enable是静默无效的():
    """先把坑本身测出来。

    这条是"反面证据"：如果哪天有人把 loader 里的
    ensure_gradient_checkpointing 删掉、只留 model.gradient_checkpointing_enable()，
    这条会继续通过，而下面那条会失败 —— 正好把问题指出来。
    """
    model = _FakeChatGLM()
    model.gradient_checkpointing_enable()
    flags = [n for n, m in model.named_modules()
             if getattr(m, "gradient_checkpointing", False)]
    assert flags == [], "ChatGLM3 的空实现不该设上任何标志"


def test_ensure_gradient_checkpointing_能补上ChatGLM3的空实现():
    model = _FakeChatGLM()
    assert ensure_gradient_checkpointing(model) is True
    assert model.transformer.encoder.gradient_checkpointing is True


def test_ensure_gradient_checkpointing_对自带开关的自定义模型也有效():
    """既不走 HF 标准接口、也不是 ChatGLM3 层级的模型（比如 GLM-4）。"""

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gradient_checkpointing = False

    class _Custom(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Block() for _ in range(2)])

    model = _Custom()
    assert ensure_gradient_checkpointing(model) is True
    assert all(b.gradient_checkpointing for b in model.layers)
def test_回填架构属性别名():
    """ChatGLM3 的 config 只有 num_layers，而新版 transformers 的
    generate() 在构造 KV cache 时要 num_hidden_layers。

    这个 bug 的症状极有迷惑性：加载 ✓ 前向 ✓ generate() ✗。
    它骗过了整条训练链路（训练只做前向和反向），要等 SFT 跑完、
    开始生成三元组时才炸。所以必须单独测。
    """

    class FakeChatGLMConfig:
        num_layers = 28
        multi_query_group_num = 2
        num_attention_heads = 32

    config = FakeChatGLMConfig()
    assert not hasattr(config, "num_hidden_layers")
    patch_pretrained_config(config)

    assert config.num_hidden_layers == 28
    # 必须优先取 multi_query_group_num（=2）而不是 num_attention_heads（=32），
    # 否则 KV cache 会按 32 个头分配，显存虚高且形状错
    assert config.num_key_value_heads == 2


def test_已有的架构属性不被覆盖():
    """模型自己声明过就不要动 —— 回填只补缺失，不覆盖已有值。"""

    class FakeConfig:
        num_hidden_layers = 12
        num_layers = 28

    config = FakeConfig()
    patch_pretrained_config(config)
    assert config.num_hidden_layers == 12


def test_别名缺失时不硬造属性():
    """别名也不存在时保持缺失，不要凭空造一个 None —— 那会让报错
    从"缺属性"变成更难懂的"类型错误"。"""

    class Bare:
        pass

    config = Bare()
    patch_pretrained_config(config)
    assert not hasattr(config, "num_hidden_layers")
def test_按类名自动判断旧版缓存():
    """类名里带 chatglm 的走旧版缓存路径，其它模型不动。

    ChatGLM3 的远程代码期望自己的 past_key_values 格式，而新版 transformers
    会传一个惰性分配的 DynamicCache —— 它的 get_masks 直接
    `past_key_values[0][0].shape[0]`，撞在 None 上。
    """

    class ChatGLMForConditionalGeneration:
        @classmethod
        def _supports_default_dynamic_cache(cls):
            return True

    class Qwen2ForCausalLM:
        @classmethod
        def _supports_default_dynamic_cache(cls):
            return True

    assert ensure_legacy_cache(ChatGLMForConditionalGeneration()) is True
    assert ChatGLMForConditionalGeneration._supports_default_dynamic_cache() is False

    # 其它模型不能被误伤
    assert ensure_legacy_cache(Qwen2ForCausalLM()) is False
    assert Qwen2ForCausalLM._supports_default_dynamic_cache() is True


def test_旧版缓存可以不按类名强制():
    class Anything:
        @classmethod
        def _supports_default_dynamic_cache(cls):
            return True

    assert ensure_legacy_cache(Anything(), force=True) is True
    assert Anything._supports_default_dynamic_cache() is False


def test_旧版缓存不重复patch():
    """同一个类被多次加载时，第二次应该直接跳过（返回值 False）。"""

    class ChatGLMConfig:
        @classmethod
        def _supports_default_dynamic_cache(cls):
            return True

    instance = ChatGLMConfig()
    assert ensure_legacy_cache(instance) is True
    assert ensure_legacy_cache(instance) is False


def test_显式关闭旧版缓存():
    class ChatGLMForConditionalGeneration:
        @classmethod
        def _supports_default_dynamic_cache(cls):
            return True

    assert ensure_legacy_cache(ChatGLMForConditionalGeneration(), force=False) is False
    assert ChatGLMForConditionalGeneration._supports_default_dynamic_cache() is True
