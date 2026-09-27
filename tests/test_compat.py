"""models/compat.py 的验收测试。

这些补丁是给特定模型的远程代码填坑的，逻辑必须锁住。尤其是
`all_tied_weights_keys` **必须是 dict 不是 list** —— 这个类型错误
在第一处调用（`len(...) > 0`）看不出来，只会在 transformers 内部
第二处（`.keys()`）才暴露，而那时已经烧了一轮 Kaggle 时间。
"""

from __future__ import annotations

from transformers import PreTrainedModel

from sfzy.models.compat import (
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
