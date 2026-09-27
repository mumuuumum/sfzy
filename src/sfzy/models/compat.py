"""针对特定模型远程代码的兼容性补丁。

**这个文件的存在本身就是一条结论**：ChatGLM3 的远程代码停留在 2023 年，
而 transformers 一直在演进。两者之间的裂缝需要有人去填 —— 而填的位置
集中放在这里，不要散落在业务代码中。

为什么要抽出来：
  * 换模型时一眼就能看出哪些是为迁就旧模型而加的，删起来干净；
  * 散在 loader.py 里会慢慢变成"没人知道为什么存在、删了又怕出事"的代码；
  * 每一条都能单独写测试（用假 config 验证回填逻辑，不需要真的加载模型）。

这里集中三件事，对应实际踩过的三个坑：
  1. 新版 transformers 移走的 generation 属性（max_length / use_cache ...）
  2. all_tied_weights_keys 缺失
  3. tokenizer 的 padding 方向
"""

from __future__ import annotations

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
    # 正常路径：基类的递归设置成功了，某个模块上会有 flag
    if any(getattr(module, "gradient_checkpointing", False) for module in model.modules()):
        return True

    # 兜底：ChatGLM3 的 transformer 主体是 model.transformer.encoder
    encoder = getattr(getattr(model, "transformer", None), "encoder", None)
    if encoder is not None and hasattr(encoder, "gradient_checkpointing"):
        encoder.gradient_checkpointing = True
        return True

    return False
