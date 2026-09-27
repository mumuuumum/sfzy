"""模型与 tokenizer 的加载。

============================ 你要实现的文件 ============================

-------------------------------- 两个环境，两套配置 --------------------------------
本地调试（4GB / CPU 或小显存）：
    configs/model_debug_small.yaml   Qwen2.5-0.5B-Instruct，float32，不量化
Kaggle 生产（2×T4 16GB）：
    configs/model_chatglm3_6b.yaml   ChatGLM3-6B，4-bit nf4 量化

同一份代码要同时支持这两种，所以所有差异都从 config 读，不要硬编码。

-------------------------------- 必须记住的一条 --------------------------------
**4-bit 量化（bitsandbytes）在 CPU 上不可用。**
load_in_4bit=True 只有在 CUDA 环境里才能跑。本地跑调试配置时必须是 False。

-------------------------------- 加载顺序会踩的坑 --------------------------------
先加载 tokenizer 再加载模型。因为 ChatGLM3 的 tokenizer 可能需要
trust_remote_code，而模型的 config 也要读同一个参数。
=====================================================================
"""

from __future__ import annotations

from typing import Any, Dict

from sfzy.config import Config
from collections import Counter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from sfzy.models.compat import (
    ensure_gradient_checkpointing,
    patch_pretrained_config,
    patch_tied_weights_keys,
    patch_tp_plan_for_quantized_load,
    resolve_padding_side,
)
from peft import prepare_model_for_kbit_training

import torch


def build_quant_config(model_cfg: Dict[str, Any]) -> Any:
    """根据配置构造 BitsAndBytesConfig（4-bit 量化）。

    要点：
      * 只有 model_cfg['load_in_4bit'] 为 True 时才构造，否则返回 None。
      * nf4 + double quant 是 QLoRA 的标准组合（bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True）。
      * bnb_4bit_compute_dtype 在 T4 上必须用 float16，不能用 bfloat16
        （T4 是 Turing 架构，不支持 bf16）。

    返回的对象直接传给 from_pretrained(quantization_config=...)。
    """
    if not model_cfg.get('load_in_4bit',False):
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get('bnb_4bit_quant_type',"nf4"),
        bnb_4bit_use_double_quant=model_cfg.get('bnb_4bit_use_double_quant',True),
        bnb_4bit_compute_dtype=model_cfg.get('bnb_4bit_compute_dtype',"float16"),
    )
      


def load_tokenizer(model_cfg: Dict[str, Any]) -> Any:
    """加载 tokenizer。

    要点：
      * trust_remote_code 从配置读（ChatGLM3 需要 True）。
      * padding_side 默认 "left"。**decoder-only 模型的批量生成必须用左 padding**：
            右 padding: [t1, t2, t3, PAD, PAD]  ← 最后一位是 PAD，generate 从 PAD 往后续写，错位
            左 padding: [PAD, PAD, t1, t2, t3]  ← 最后一位是真实 token ✓
        （我最初在提示里写反了，说"左 padding 会错位"，实际正好相反。）
        另外 ChatGLM3 的 tokenizer 里有一句 `assert self.padding_side == "left"`，
        设成 right 会直接 AssertionError —— 它只支持左 padding。
      * 确保 pad_token 存在：很多生成式模型的 tokenizer 没有 pad_token，
        通常用 eos_token 顶上，否则 collator 里 padding 会直接报错。
    """
    model_name = model_cfg.get("model_name_or_path")
    if not model_name:
        raise ValueError(
            "model_cfg 中缺少模型名称或路径，"
            "请提供 'model_name_or_path'"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=model_cfg.get("trust_remote_code",False),
        padding_side=resolve_padding_side(model_cfg),
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    model_cfg: Dict[str, Any],
    quant_config: Any = None,
    gradient_checkpointing: bool = True,
    local_rank: int = -1,
) -> Any:
    """加载模型。

    local_rank >= 0 表示处于 DDP 环境（由 torchrun 注入），此时把整模型
    **钉在本进程自己的卡上**。原因是 DDP 和 `device_map="auto"` 的语义相反：

        device_map="auto"  →  模型并行，把模型**摊到多张卡**上（给推理用）
        DDP                →  数据并行，**每张卡一份完整副本**，只跑不同数据

    两者叠加会直接报错（每个进程只拿到一部分层，没法各自前向）。
    """
    model_name = model_cfg.get("model_name_or_path")
    if not model_name:
        raise ValueError(
            "model_cfg 中缺少模型名称或路径，"
            "请提供 'model_name_or_path'"
        )
    
    
    torch_dtype_cfg = model_cfg.get("torch_dtype", "auto")
    if isinstance(torch_dtype_cfg, str):
        if torch_dtype_cfg == "auto":
            torch_dtype = "auto"
        else:
            torch_dtype = getattr(torch, torch_dtype_cfg, None)
            if torch_dtype is None:
                raise ValueError(f"未知的 torch_dtype: {torch_dtype_cfg}")
    else:
        torch_dtype = torch_dtype_cfg

    # 远程代码的兼容性补丁集中放在 models/compat.py，这里只负责调用。
    # 具体踩过哪些坑、为什么这么修，都在那个文件里写了。
    patch_tied_weights_keys()
    # 这个必须在 from_pretrained **之前**：新版 transformers 会在加载权重时
    # 调 caching_allocator_warmup → get_total_byte_count，而它只在
    # torch.distributed 已初始化（也就是 DDP）时才去读 model.tp_plan，
    # ChatGLM3 没有这个属性 → len(None) 直接崩。
    patch_tp_plan_for_quantized_load()

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    patch_pretrained_config(config, model_cfg)


    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
        quantization_config=quant_config,
        device_map={"": local_rank} if local_rank >= 0 else model_cfg.get("device_map", "auto"),
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    
    # ---------- 4. 量化模型必须 prepare_model_for_kbit_training ----------
    # 调用顺序：load_model(quant) -> prepare_model_for_kbit_training -> inject_lora
    #
    # 为什么放在这里而不是 trainer 里？
    #   1. prepare_model_for_kbit_training 本质上是“量化模型的加载后处理”，
    #      它做的是：把 LayerNorm 等非量化层转成 fp32、开启输入嵌入的梯度、
    #      配合 gradient_checkpointing 打补丁。这些都是“模型本身的属性”，
    #      不是训练循环的逻辑。
    #   2. 放在这里能保证调用顺序：一旦用户拿到模型，它已经处于“可接 LoRA”的状态，
    #      不会出现“先 inject_lora 再 prepare”这种顺序错误（那会导致训练报错或梯度异常）。
    #   3. 缺点是纯推理场景也会走这一步。如果只想推理，可以给 load_model 加一个
    #      for_training: bool 参数，或者干脆不传 quant_config 之外的开关。
    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
        if gradient_checkpointing and not ensure_gradient_checkpointing(model):
            import warnings

            warnings.warn(
                "梯度检查点没能启用（配置里写了 true，但模型上没有生效）。"
                "显存会按未开启的方式增长，很可能 OOM。",
                stacklevel=2,
            )
    elif gradient_checkpointing:
        # 非量化分支（bf16 / fp16）不会走 prepare_model_for_kbit_training，
        # 得在这里自己开。
        #
        # **关键：不能只调 model.gradient_checkpointing_enable()。**
        # ChatGLM3 把这个方法覆写成了空壳 —— 它只检查 supports 标志，
        # 然后什么都不做（见 modeling_chatglm.py 的 ChatGLMPreTrainedModel）。
        # 调用它不报错、也不生效，于是显存按"每层完整保存"的方式涨，
        # 最后 OOM，而报错栈里全是别的东西，看不出是这里的问题。
        #
        # 真正的开关在 GLMTransformer（model.transformer.encoder）上，
        # 由 compat.ensure_gradient_checkpointing 直接设。
        try:
            model.gradient_checkpointing_enable()
        except Exception:  # noqa: BLE001 - 有些远程代码的实现不规范
            pass

        if not ensure_gradient_checkpointing(model):
            import warnings

            warnings.warn(
                "梯度检查点没能启用（配置里写了 true，但模型上没有任何模块生效）。"
                "显存会按未开启的方式增长，很可能 OOM。",
                stacklevel=2,
            )

        if hasattr(model, "enable_input_require_grads"):
            # 梯度检查点要求被检查的片段至少有一个输入需要梯度，否则重算出来的
            # 输出没有 grad_fn，反向传播时整条链是断的，报
            # "element 0 of tensors does not require grad and does not have a grad_fn"。
            # LoRA 场景下第一层的输入是 embedding 的输出，默认不需要梯度，
            # 所以必须显式打开这个开关。
            # prepare_model_for_kbit_training 内部会做同样的事，
            # 我们这条非量化分支要自己补上。
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False

    return model
    


def describe_model(model: Any) -> Dict[str, Any]:
    """返回模型的概况，用于日志：参数量、可训练参数量、dtype、设备。

    这个函数在注入 LoRA 前后各调一次，是验证"LoRA 只训练了极少参数"
    最直观的证据（通常可训练参数占比在 1% 以下）。

    返回形如 {"total_params": int, "trainable_params": int,
              "trainable_ratio": float, "dtype": str, "device": str}
    """
    total = trainable = 0
    dtype_counts = Counter()
    devices = set()
    compute_dtype = None

    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
        dtype_counts[str(p.dtype)] += p.numel()
        devices.add(str(p.device))
        # 量化参数是整数类型，is_floating_point() 会跳过它们
        if compute_dtype is None and p.is_floating_point():
            compute_dtype = str(p.dtype)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
        "dtype": compute_dtype or "unknown",
        "dtype_breakdown": dict(dtype_counts),
        "devices": sorted(devices)
    }
