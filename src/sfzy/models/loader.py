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

from typing import Any, Dict, Optional, Tuple

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
    ensure_legacy_cache,
    patch_generation_private_attrs,
    patch_pretrained_config,
    patch_tied_weights_keys,
    patch_tp_plan_for_quantized_load,
    resolve_padding_side,
)
from peft import prepare_model_for_kbit_training

import torch

from sfzy.utils.logging import get_logger

logger = get_logger("loader")


# ---------------------------------------------------------------------------
# 设备放置：4/8-bit 量化模型不能 .to()
# ---------------------------------------------------------------------------
# transformers 的 `PreTrainedModel.to()` 对 bitsandbytes 量化模型直接抛
#     ValueError: `.to()` is not supported for 4/8-bit bitsandbytes models
# 因为量化权重是 int8/uint8 + 分布式 metadata，搬动会破坏它们，正确做法是
# **加载时**用 `device_map` 决定位置。所以"加载后 .to(device)"这条路对量化
# 模型根本走不通，只能在 load_model 里把 device_map 给对。
#
# 踩过的现场：`model_qwen25_7b.yaml` 开着 load_in_4bit，脚本里那句
# `if device_map in (None, "", "none"): model.to(device)` 在 device_map 写成
# null 的配置上会放行 → 4090 上直接抛上面的 ValueError；T4 用 0.5B/bf16 的
# 配置时 device_map 也是 null 但没量化，`.to()` 正常，于是"T4 能跑、4090 崩"。
_NO_DEVICE_MAP = (None, "", "none")


def is_quantized_model(model: Any) -> bool:
    """模型是不是 bitsandbytes 4/8-bit 量化的。三个属性取或，兼容不同版本。"""
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        return True
    if getattr(model, "is_quantized", False):
        return True
    method = getattr(model, "quantization_method", None)
    if method is None:
        return False
    # quantization_method 可能是枚举（QuantizationMethod.BITS_AND_BYTES），
    # 也可能被字符串化。取 name 再比，别依赖 str() 的分隔符写法。
    name = getattr(method, "name", None) or str(method)
    return "BITS_AND_BYTES" in name.upper() or "BITSANDBYTES" in name.upper()


def resolve_device_index(device: Any) -> Optional[int]:
    """把设备串翻译成卡号：'cuda:1' → 1，'cuda' → 0，'cpu'/None/'auto' → None。"""
    if device is None or isinstance(device, int):
        return device if isinstance(device, int) else None
    text = str(device)
    if not text.startswith("cuda"):
        return None
    return int(text.split(":", 1)[1]) if ":" in text else 0


def move_model_to_device(model: Any, device: Any) -> Any:
    """把**非量化**模型搬到 device；量化模型原样返回。

    量化模型的位置已经由加载时的 device_map 定死了，再 `.to()` 会抛
    `ValueError: .to() is not supported for ... bitsandbytes models`。
    所有"手动搬运"的地方都该走这个函数，而不是裸 `model.to(...)`。
    """
    if is_quantized_model(model):
        return model
    return model.to(device)


def resolve_model_device(model: Any, fallback: Any = "cuda:0") -> str:
    """模型实际落在哪个设备上。用来给运行时/tokenizer 对齐设备，避免
    "模型在 cuda:0、输入张量送到 cuda:1"这种静默错位。"""
    try:
        return str(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        return str(fallback)


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
    # ChatGLM3 自带的 stream_generate 跳过 generate()，直接调
    # _get_logits_processor / _get_stopping_criteria，而新版 transformers
    # 这两个方法要读一个只有
    # generate() 才会设上的私有属性 _eos_token_tensor → AttributeError。
    # 补在类上，全局一次即可（幂等）。
    patch_generation_private_attrs()

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

    # ---- use_cache 的处理必须放在**两个分支之外** ----
    #
    # 之前这一行只写在非量化分支里，于是走 4-bit 路径时 use_cache 一直是 True，
    # ChatGLM3 的 GLMTransformer.forward 会打出
    #     "`use_cache=True` is incompatible with gradient checkpointing..."
    # 然后在内部自己改掉。**它自己兜住了，所以不会故障** ——
    # 但也正因为不会故障，这个配置没生效的问题一直没被发现。
    #
    # 换成没有这道自我保护的模型（比如 Qwen 系），use_cache=True 会在训练时
    # 真的建 KV cache，白白吃显存，在 24GB 卡上可能就是 OOM 和不 OOM 的区别。
    if gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False

    # ---- 旧版缓存路径 ----
    #
    # 新版 transformers 的 generate() 默认给模型造一个 DynamicCache 传进去，
    # 而它是**惰性分配**的（初始时每层 key/value 都是 None）。ChatGLM3 的
    # 远程代码期望的是自己的 past_key_values 格式，于是它的 get_masks 里
    # `past_key_values[0][0].shape[0]` 直接炸在 None 上。
    #
    # 这两行让 ChatGLM3 退回**旧版缓存路径**（模型自己维护 past_key_values）。
    # 注意它和 use_cache 是两件事：use_cache 关掉会让每个 token 重算整个
    # 前缀（慢几百倍），而旧版路径是正常的增量解码。
    #
    # 训练永远不会碰到这段 —— 只有 generate() 走缓存逻辑。所以 check_model.py
    # 的第八步（真调一次 generate）是专门为它加的。
    if ensure_legacy_cache(model, force=model_cfg.get("force_legacy_cache")):
        logger.info(
            "已为 %s 启用旧版缓存路径（DynamicCache 与它的 past_key_values 格式不兼容）",
            type(model).__name__,
        )

    return model
    


def _target_device(device: Any) -> str:
    """把 'auto'/None 解析成一个具体设备串。"""
    if device is None or str(device) in ("auto", ""):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device)


def load_inference_model(
    model_cfg: Dict[str, Any],
    device: Any = "auto",
    load_in_4bit: Optional[bool] = None,
    gradient_checkpointing: bool = False,
) -> Tuple[Any, Any, str]:
    """纯推理加载：tokenizer + model，并**正确处理 4/8-bit 的设备放置**。

    返回 `(model, tokenizer, device_str)`。`device_str` 是模型**实际**所在的
    设备，调用方应该拿它建运行时，而不要再用手写的 `--device` —— 量化模型的
    位置由 `device_map` 决定，两者可能不一致，错位后报的是"张量不在同一设备"，
    根因却在这里。

    和 `load_model` 的区别只有一条，但很关键：
    **量化模型不能 `.to()`**，只能用 `device_map` 在加载时放好。所以这里的
    做法是先把 `--device`（或 `auto`）翻译成 `{"": 卡号}` 再加载，加载完
    一个字都不搬。非量化模型也走同一条路，行为一致、少一个分支。

    `load_in_4bit` 为 None 时跟随配置；给 True/False 可以强制开关，
    4090 上跑 7B 时用它关掉量化走 bf16（24GB 装得下）。
    """
    cfg = dict(model_cfg)
    if load_in_4bit is not None:
        cfg["load_in_4bit"] = bool(load_in_4bit)

    target = _target_device(device)
    idx = resolve_device_index(target)
    # 先判设备再构造量化配置：CPU 上构造 BitsAndBytesConfig 本身也会失败
    # （依赖 bitsandbytes），这里要先给出"4-bit 需要 CUDA"这个可读的错误。
    wants_quant = bool(cfg.get("load_in_4bit") or cfg.get("load_in_8bit"))
    if wants_quant and idx is None:
        raise ValueError(
            f"load_in_4bit=True 需要 CUDA 设备，但解析出的设备是 {target!r}。"
            " 要么 --device cuda:N，要么关掉 4-bit（--no-4bit）改走 bf16/fp16。"
        )
    quant = build_quant_config(cfg)

    # 推理固定单卡：不做模型并行分片，位置完全可预测。
    cfg["device_map"] = {"": idx} if idx is not None else None

    tokenizer = load_tokenizer(cfg)
    model = load_model(
        cfg, quant_config=quant, gradient_checkpointing=gradient_checkpointing
    )
    # device_map 已经把模型放好了；CPU 分支再兜一次底（from_pretrained(None) 默认 CPU）。
    if idx is None:
        model = move_model_to_device(model, target)
    return model, tokenizer, resolve_model_device(model, fallback=target)


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
