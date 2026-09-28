"""针对特定模型远程代码的兼容性补丁。

**这个文件的存在本身就是一条结论**：ChatGLM3 的远程代码停留在 2023 年，
而 transformers 一直在演进。两者之间的裂缝需要有人去填 —— 而填的位置
集中放在这里，不要散落在业务代码中。

为什么要抽出来：
  * 换模型时一眼就能看出哪些是为迁就旧模型而加的，删起来干净；
  * 散在 loader.py 里会慢慢变成"没人知道为什么存在、删了又怕出事"的代码；
  * 每一条都能单独写测试（用假 config 验证回填逻辑，不需要真的加载模型）。

这里集中五件事，对应实际踩过的五个坑：
  1. 新版 transformers 移走的 generation 属性（max_length / use_cache ...）
  2. all_tied_weights_keys 缺失
  3. tokenizer 的 padding 方向
  4. 梯度检查点在 ChatGLM3 上是空实现（见 ensure_gradient_checkpointing）
  5. 没有 TP plan 的模型在新版 transformers 的加载路径里会崩（见
     patch_tp_plan_for_quantized_load）
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from transformers import GenerationConfig, PreTrainedModel

# GenerationConfig 里不该被回填到 model config 的字段。
# 以 "_" 开头的是内部标记，transformers_version 是版本号，回填没意义。
_SKIP_FIELDS = {"transformers_version"}


def patch_tied_weights_keys() -> bool:
    """给 PreTrainedModel 补上 all_tied_weights_keys。返回是否真的打了补丁。

    ChatGLM3 的远程代码只提供了 ``_tied_weights_keys``，而新版 transformers
    的量化路径要读 ``all_tied_weights_keys``，于是加载时直接报：

        AttributeError: 'ChatGLMForConditionalGeneration' object has no
        attribute 'all_tied_weights_keys'

    **类型必须是 dict，不能是 list。** 同一个属性在 transformers 内部两处
    用法要求不同：

        quantizers/base.py       : len(model.all_tied_weights_keys) > 0   (list/dict 都行)
        integrations/accelerate  : getattr(model, "...", {}).keys()       (必须是 dict)

    第二行的 ``getattr`` 默认值写了 ``{}``，这就是 transformers 作者自己
    承认的正确类型。填 ``[]`` 会在第二处报 ``'list' object has no attribute 'keys'``。

    补在基类的类属性上：模型自己在 ``__init__`` 里设过的会被实例属性覆盖，
    所以对其他模型没有影响。
    """
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return False
    PreTrainedModel.all_tied_weights_keys = {}
    return True


def ensure_tp_plan(model: Any) -> bool:
    """确保 ``model.tp_plan`` 是个 dict（空 dict = 不做张量并行）。返回是否成功。

    先试实例、再试它自己那一个类：属性可能是普通字段（实例赋值即可），
    也可能是带 setter 的 property（4.57 / main 都是这样），
    还可能是只读 property —— 那就在子类的类字典里放一个 {} 把它遮掉。
    """
    if getattr(model, "tp_plan", None) is not None:
        return True
    for target in (model, type(model)):
        try:
            setattr(target, "tp_plan", {})
        except Exception:  # noqa: BLE001 —— 只读属性/奇怪元类，继续试下一个
            continue
        if getattr(model, "tp_plan", None) is not None:
            return True
    return False


def patch_tp_plan_for_quantized_load() -> bool:
    """让没有 TP plan 的模型能通过新版 transformers 的"显存预热"。返回是否真的打了补丁。

    报错长这样（**只在 DDP 下必现，单进程跑不出来**）：

        File ".../transformers/modeling_utils.py", line 4706, in get_total_byte_count
            if len(tp_plan) > 0:
        TypeError: object of type 'NoneType' has no len()

    根因是 transformers 内部两段代码对"这个模型有没有张量并行计划"的假设不一致：

        caching_allocator_warmup: 只在 device_map 不为 None 时调用（4-bit 量化必走）
        get_total_byte_count:     tp_plan = model.tp_plan
                                            if torch.distributed.is_initialized() else []

    也就是说 **单进程时它走 `[]` 分支，压根不读 model.tp_plan；一旦进程组初始化了，
    它就去读这个属性**。ChatGLM3 的远程代码没有 TP plan（它的 config 里没有
    base_model_tp_plan），新版 transformers 又不给默认值，于是拿到 None。

    这正好解释了"同一份配置、同一个模型，单进程能加载、一加 DDP 立刻挂"——
    和训练逻辑无关，纯粹是加载路径分叉。

    补丁打在 ``transformers.modeling_utils.get_total_byte_count`` 这个模块级函数上：
    ``caching_allocator_warmup`` 是**运行时按全局名**查它的（不是 import 时就绑定），
    所以替换模块属性就能生效。旧版 transformers 没有这个函数（本机 4.57.6 就没有），
    patch 直接返回 False，零副作用。
    """
    try:
        modeling_utils = importlib.import_module("transformers.modeling_utils")
    except ImportError:  # pragma: no cover - transformers 一定装得上
        return False

    original = getattr(modeling_utils, "get_total_byte_count", None)
    if original is None or getattr(original, "_sfzy_patched", False):
        return False

    def get_total_byte_count(model, accelerator_device_map, hf_quantizer=None):
        ensure_tp_plan(model)
        return original(model, accelerator_device_map, hf_quantizer)

    # 打个标记，重复调用（每条 rank 都会调 load_model）时不会层层套娃
    get_total_byte_count._sfzy_patched = True
    modeling_utils.get_total_byte_count = get_total_byte_count
    return True


def patch_pretrained_config(config: Any, model_cfg: Optional[Dict[str, Any]] = None) -> None:
    """把新版 transformers 移走的 generation 属性回填到 config 上（就地修改）。

    新版 transformers 把 generation 相关的属性从 ``PretrainedConfig`` 搬到了
    ``GenerationConfig``，而 ChatGLM3 的远程代码还在从 config 上读它们：

        config.max_length   →  self.max_sequence_length = config.max_length
        config.use_cache    →  use_cache = ... else self.config.use_cache

    这两次报错是**同一个根因**，说明"缺哪个补哪个"治不好 —— 所以这里
    把 ``GenerationConfig`` 的默认值**全部回填**，不管它的代码还要读哪个
    generation 属性都不会再炸。

    唯一的例外是 ``max_length``：generation 的默认值是 20，直接用会让
    ``self.max_sequence_length`` 变成 20。所以只在它原本缺失时，才用模型
    自己的 ``seq_length`` 覆盖。

    最后还会应用 ``model_cfg["config_patches"]`` 里的显式补丁（字面值），
    作为兜住回填覆盖不到的情况的逃生口。
    """
    had_max_length = hasattr(config, "max_length")

    for field, value in GenerationConfig().to_dict().items():
        if field.startswith("_") or field in _SKIP_FIELDS:
            continue
        if not hasattr(config, field):
            setattr(config, field, value)

    if not had_max_length and hasattr(config, "seq_length"):
        config.max_length = config.seq_length

    for field, value in ((model_cfg or {}).get("config_patches") or {}).items():
        if not hasattr(config, field):
            setattr(config, field, value)

    _patch_arch_aliases(config)


# 架构属性的别名表：左边是新版 transformers 期望的名字，右边是 ChatGLM3
# 实际用的名字（按优先级排列）。
# 注意这些**不是** generation 配置项，所以上面那轮 GenerationConfig 回填
# 覆盖不到它们，必须单独处理。
_ARCH_ALIASES: Dict[str, tuple] = {
    "num_hidden_layers": ("num_layers",),
    "num_key_value_heads": ("multi_query_group_num", "num_attention_heads"),
}


def _patch_arch_aliases(config: Any) -> None:
    """把 ChatGLM3 的架构属性名补成新版 transformers 期望的名字。

    不补会怎样 —— 症状非常有迷惑性：

        加载模型   ✓
        前向一次   ✓
        generate() ✗

        AttributeError: 'ChatGLMConfig' object has no attribute 'num_hidden_layers'

    崩在 `generation/utils.py` 的 `_prepare_cache_for_generation`：新版
    transformers 构造 DynamicCache 时要 `decoder_config.num_hidden_layers`，
    而 ChatGLM3 的 config 只有 `num_layers`。

    **这个 bug 骗过了整条训练链路** —— 训练只做前向和反向，一次都不会碰
    到 generate()。要等跑了几个小时的 SFT、开始生成三元组时才炸。
    所以 check_model.py 现在会真的调一次 generate()。

    只补"缺失 + 别名明确"的属性，不做猜测；拿不准的走
    `model_cfg["config_patches"]` 显式指定。
    """
    for canonical, aliases in _ARCH_ALIASES.items():
        if hasattr(config, canonical):
            continue
        for alias in aliases:
            value = getattr(config, alias, None)
            if value is not None:
                setattr(config, canonical, value)
                break


def ensure_legacy_cache(model: Any, force: Optional[bool] = None) -> bool:
    """让不兼容 DynamicCache 的模型（ChatGLM3）退回**旧版缓存路径**。

    不处理会怎样 —— 训练全绿，生成崩溃：

        modeling_chatglm.py:688, in get_masks
            past_length = past_key_values[0][0].shape[0]
        AttributeError: 'NoneType' object has no attribute 'shape'

    原因是两套缓存约定对不上：
      * 新版 transformers 的 generate() 默认造一个 `DynamicCache` 传给模型，
        而它是**惰性分配**的 —— 初始时每层的 key/value 都是 None；
      * ChatGLM3 的远程代码期望的是它自己的格式（list of (key, value) 元组）。

    ChatGLM3 的 `get_masks` 里其实有 `if past_key_values:` 的判断，但
    `DynamicCache` 实现了 `__len__`（28 层 > 0），所以判断为真，走进了崩溃分支。

    **解法是让 `_supports_default_dynamic_cache()` 返回 False。** 见
    `generation/utils.py` 的 "Quick escape route 3"：这时 generate() 直接
    跳过 cache 的构造，退回旧版路径 —— 由模型自己维护 past_key_values，
    正好是 ChatGLM3 期望的格式。

    **为什么不用 `use_cache=False`。** 那条路每生成一个 token 都要重算整个
    前缀（4096 token 的 prompt + 384 token 输出 → 计算量差几百倍）。
    旧版缓存路径是**正常的增量解码**，只是缓存对象由模型自己管。

    force 为 None 时按类名自动判断（含 chatglm 的走旧版），也可以由
    model_cfg 显式覆盖。
    """
    cls = type(model)
    if force is None:
        force = "chatglm" in cls.__name__.lower()
    if not force or getattr(cls, "_sfzy_legacy_cache_patched", False):
        return False

    # 这是 GenerationMixin 上的 classmethod，覆盖到具体模型类上即可。
    # 打成标记避免重复 patch（同一个类可能被多次加载）。
    cls._supports_default_dynamic_cache = classmethod(lambda _cls: False)
    cls._sfzy_legacy_cache_patched = True
    return True


def resolve_padding_side(model_cfg: Optional[Dict[str, Any]] = None, default: str = "left") -> str:
    """决定 tokenizer 的 padding 方向。

    **decoder-only 模型的批量生成必须用左 padding**：

        右 padding: [t1, t2, t3, PAD, PAD]   ← 最后一位是 PAD，
                                                generate 从 PAD 往后续写，结果错位
        左 padding: [PAD, PAD, t1, t2, t3]   ← 最后一位是真实 token ✓

    另外 ChatGLM3 的 tokenizer 里有一句 ``assert self.padding_side == "left"``，
    设成 right 会直接 AssertionError —— 它的实现只支持左 padding。

    训练不受影响：我们的 collator 是自己构造 input_ids / labels /
    attention_mask 的，不走 tokenizer 的 pad()。
    """
    return (model_cfg or {}).get("padding_side", default)


def ensure_gradient_checkpointing(model: Any) -> bool:
    """确认梯度检查点**真的**开了；没开就手动补上。返回最终是否生效。

    为什么需要这个：``prepare_model_for_kbit_training`` 内部会调
    ``model.gradient_checkpointing_enable()``，而那个方法依赖
    ``model._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=...)``。

    **ChatGLM3 的远程代码把签名写成了 `_set_gradient_checkpointing(self, module, value=False)`**
    —— 参数名和基类对不上，于是基类传进去的 ``enable=True`` 被当成 ``module``，
    ``isinstance(True, GLMTransformer)`` 为假，什么都不做、也不报错。

    结果就是：配置里写了 ``gradient_checkpointing: true``，显存却按**没开**的方式涨，
    最后 OOM，而报错信息里全是 bitsandbytes 的调用栈，看不出是这里的问题。

    实测证据：max_length=8192 / batch=1 在 16GB 的 T4 上仍然 OOM，
    说明激活值是按"每层完整保存"算的。
    """
    # **不要写"先检查有没有模块开着，开着就提前返回"** —— 那正是它上次没生效的原因：
    # 只要有任何模块碰巧带着这个属性为 True，就会提前 return，
    # 而真正需要设的 encoder 反而没被设上。
    #
    # ChatGLM3 的 transformer 主体是 model.transformer.encoder（GLMTransformer），
    # 它在 __init__ 里定义了 self.gradient_checkpointing = False，
    # 而 forward 里会检查它。直接设即可。
    encoder = getattr(getattr(model, "transformer", None), "encoder", None)
    if encoder is not None and hasattr(encoder, "gradient_checkpointing"):
        encoder.gradient_checkpointing = True

    # 其他模型先走 HuggingFace 的标准接口（Qwen 这类原生支持的模型走这条）
    if not any(getattr(m, "gradient_checkpointing", False) for m in model.modules()):
        try:
            model.gradient_checkpointing_enable()
        except Exception:  # noqa: BLE001 - 有些远程代码的实现不规范
            pass

    # 最后一道兜底：凡是**自己带着这个属性**的模块全部置 True。
    #
    # 这一条是为了覆盖"模型实现了自己的检查点逻辑，但既不走 HF 的标准接口、
    # 也不是 ChatGLM3 那个层级"的情况 —— 比如 GLM-4、以及各种从零实现的模型。
    # 只在前面两条都没生效时才做，避免和 HF 的选择性设置打架。
    if not any(getattr(m, "gradient_checkpointing", False) for m in model.modules()):
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True

    return any(getattr(m, "gradient_checkpointing", False) for m in model.modules())


def describe_gradient_checkpointing(model: Any) -> Dict[str, Any]:
    """诊断用：把梯度检查点的实际状态全部摊开。

    加这个是因为我们在这件事上**连续猜错了两轮** —— 每次都是"推理认为应该开了"，
    然后在 Kaggle 上 OOM。与其继续推测，不如让代码把状态直接打印出来。
    """
    on = [n for n, m in model.named_modules() if getattr(m, "gradient_checkpointing", False)]
    encoder = getattr(getattr(model, "transformer", None), "encoder", None)
    return {
        "modules_with_flag": len(on),
        "examples": on[:3],
        "model_training": getattr(model, "training", None),
        "has_transformer": hasattr(model, "transformer"),
        "has_encoder": encoder is not None,
        "encoder_flag": getattr(encoder, "gradient_checkpointing", "（无此属性）")
        if encoder is not None else None,
    }
