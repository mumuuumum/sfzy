"""把项目的 LoRA 导出成 **vLLM / PEFT 认识的 adapter 目录**。

为什么需要这一层：本项目训练出来的 LoRA 是自定义格式 ——

    transformer.encoder.layers.0.self_attention.query_key_value.lora_A.weight

而 vLLM 只认 PEFT 的命名：键名必须带 ``base_model.model.`` 前缀，且目录里要有
``adapter_config.json`` + ``adapter_model.safetensors``。差一个前缀，vLLM 的
``parse_fine_tuned_lora_name`` 就找不到层 —— 而且它**不会报错**，会安静地按
"零 LoRA"生成，你只会觉得"挂了 adapter 但效果没变"。

三件事必须做对（都有测试钉住）：

1. **剥掉 DDP 的 ``module.`` 前缀**并补上 ``base_model.model.``；
2. **rank 从 ``lora_A`` 的形状推断**（``(r, in_features)``），而不是信配置 ——
   配置写错时按实际值导出并告警，否则 alpha/r 的 scaling 会静默改变强度；
3. **目标层名必须在白名单里**（ChatGLM3 是 query_key_value / dense /
   dense_h_to_4h / dense_4h_to_h），导出阶段就暴露"导了模型不支持的层"，
   不要等 vLLM 加载到一半才报错。

两个入口：
  * `convert_checkpoint_to_peft`：从磁盘上的 ``step_*.pt`` / ``best.pt`` 导出
    （离线工具用，例如 `scripts/generate_triples_vllm.py`）；
  * `export_lora_from_model`：从**活着的模型**导出（GRPO 每步换策略时用，
    不需要先落一份 checkpoint）。

反向也要能走：`convert_peft_to_checkpoint` 把 PEFT 目录（例如从云端拷回来的
``adapter_model.safetensors``）转回项目自己的 ``.pt``，好让本地的
`--adapter`（`load_checkpoint`）能直接吃。同样是"错了不报错"的地方：前缀没剥
干净时每个键都会被算成"模型里没有"，`load_checkpoint` 会直接报"一个都没载入"。
转回来时会把 ``adapter_config.json`` 的 alpha/target_modules 一并记进 payload，
这样从这份 ``.pt`` 再导出回 PEFT 时不必靠人记得传 alpha（rank 能从形状反推，
alpha 不能，而 alpha/r 直接决定 LoRA 强度）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from sfzy.utils.logging import get_logger

logger = get_logger("lora_export")

# ChatGLM3-6B 的四个叶子线性层（等价于 LLaMA-Factory 的 lora_target: all）。
CHATGLM3_TARGET_MODULES: tuple = ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h")


def extract_lora_state(payload: Any) -> Mapping[str, Any]:
    """从 checkpoint 里取出 LoRA 权重。

    `save_checkpoint` 存的是 ``{"model": {...}, "optimizer": ..., "state": ...}``；
    而 `save_lora` 直接存 ``{键: 张量}``。两种都要认，否则导出的 adapter 会是空的。
    """
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        return payload["model"]
    return payload


def lora_state_to_peft(
    state: Mapping[str, Any],
    out_dir: Union[str, Path],
    *,
    alpha: int,
    dropout: float = 0.0,
    target_modules: Sequence[str] = CHATGLM3_TARGET_MODULES,
    base_model: str = "",
    rank: Optional[int] = None,
    dtype: Any = None,
) -> Path:
    """把一组 LoRA 权重写成 PEFT 目录，返回该目录。

    `rank` 只是"声称的值"：实际 rank 由 ``lora_A`` 的形状决定，不一致时告警并按
    实际值导出（rank 错了 vLLM 会直接加载失败，但 alpha 与它一起错就会静默改变
    强度 —— 所以必须让"实际值"说了算）。
    """
    import torch
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    converted: Dict[str, Any] = {}
    ranks: set = set()
    unknown_modules: List[str] = []

    for key, tensor in state.items():
        if "lora_A" not in key and "lora_B" not in key:
            continue
        name = str(key)
        if name.startswith("module."):
            name = name[len("module."):]
        if not name.startswith("base_model.model."):
            name = "base_model.model." + name
        value = tensor.detach() if hasattr(tensor, "detach") else tensor
        if dtype is not None:
            value = value.to(dtype)
        converted[name] = value.contiguous()
        if name.endswith("lora_A.weight"):
            ranks.add(int(value.shape[0]))
        # ...base_model.model.<路径>.<模块名>.lora_A.weight → 取模块名
        parts = name.split(".")
        module_name = parts[-3] if len(parts) >= 3 else ""
        if module_name not in tuple(target_modules):
            unknown_modules.append(module_name)

    if not converted:
        raise ValueError(
            "这组权重里没有任何 lora_A/lora_B —— 导出会是空 adapter，"
            "而 vLLM 不会报错，它只会安静地按零 LoRA 生成。"
        )
    if len(ranks) != 1:
        raise ValueError(f"LoRA rank 不一致：{sorted(ranks)}")
    actual_rank = ranks.pop()
    if rank is not None and int(rank) != actual_rank:
        logger.warning(
            "配置声称 r=%s，实际 rank=%d —— 按实际值导出。请确认 alpha 也是训练时那一版"
            "（scaling=alpha/r 配错会静默改变 LoRA 强度）",
            rank, actual_rank,
        )
    if unknown_modules:
        logger.warning(
            "导出里出现了 target_modules 之外的层：%s（vLLM 可能加载失败或忽略它们）",
            sorted(set(unknown_modules)),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(out_dir / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": int(actual_rank),
        "lora_alpha": int(alpha),
        "lora_dropout": float(dropout),
        "fan_in_fan_out": False,
        "bias": "none",
        "target_modules": list(target_modules),
        "base_model_name_or_path": str(base_model),
        "init_lora_weights": True,
        "use_rslora": False,
    }
    (out_dir / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "LoRA 已导出：%s（%d 个张量，r=%d，alpha=%d）",
        out_dir, len(converted), actual_rank, int(alpha),
    )
    return out_dir


def convert_checkpoint_to_peft(
    adapter_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    alpha: Optional[int] = None,
    dropout: float = 0.0,
    target_modules: Sequence[str] = CHATGLM3_TARGET_MODULES,
    base_model: str = "",
    rank: Optional[int] = None,
) -> Path:
    """从磁盘上的 ``.pt``（``save_checkpoint`` 或 ``save_lora`` 格式）导出。

    ``alpha`` 省略时用 checkpoint 自带的 `convert_peft_to_checkpoint` 存下的
    ``lora_config``。alpha 是唯一无法从权重形状反推的量（rank 可以），而缩放
    系数 ``alpha/r`` 直接决定 LoRA 强度 —— 宁可在缺信息时报错，也不要默认一个值
    悄悄改掉强度。
    """
    import torch

    adapter_path = Path(adapter_path)
    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    state = extract_lora_state(payload)
    if alpha is None:
        config = payload.get("lora_config") if isinstance(payload, Mapping) else None
        if not isinstance(config, Mapping) or "lora_alpha" not in config:
            raise ValueError(
                f"{adapter_path} 里没有记录 lora_alpha（也不是在命令行里给的）——"
                "alpha/r 配错会静默改变 LoRA 强度，所以这里不替你猜：请显式传 alpha="
            )
        alpha = int(config["lora_alpha"])
        dropout = float(config.get("lora_dropout", dropout))
        target_modules = list(config.get("target_modules", target_modules))
        base_model = base_model or str(config.get("base_model_name_or_path", ""))
    try:
        return lora_state_to_peft(
            state, out_dir, alpha=alpha, dropout=dropout,
            target_modules=target_modules, base_model=base_model, rank=rank,
        )
    except ValueError as exc:
        # 把"哪个文件"补进报错里 —— 批量导出时这条信息很关键
        raise ValueError(f"{adapter_path}：{exc}") from exc


def export_lora_from_model(
    model: Any,
    out_dir: Union[str, Path],
    *,
    alpha: int,
    dropout: float = 0.0,
    target_modules: Sequence[str] = CHATGLM3_TARGET_MODULES,
    base_model: str = "",
    rank: Optional[int] = None,
    keys: Optional[Iterable[str]] = None,
) -> Path:
    """从**活着的模型**导出 LoRA（GRPO 每步换策略用，不必先存 checkpoint）。

    只取名字里含 ``lora_`` 的可训练参数（与 `mark_only_lora_trainable` 的约定一致），
    导出时搬到 CPU 并保持 fp32 —— vLLM 加载时会按它自己的 dtype 转换。
    """
    state = {
        name: param.detach().to("cpu")
        for name, param in model.named_parameters()
        if "lora_" in name and (keys is None or name in keys)
    }
    if not state:
        raise ValueError(
            "模型里没有 LoRA 参数（名字含 lora_）—— 先 inject_lora 再导出。"
        )
    return lora_state_to_peft(
        state, out_dir, alpha=alpha, dropout=dropout,
        target_modules=target_modules, base_model=base_model, rank=rank,
    )


def peft_to_lora_state(tensors: Mapping[str, Any], *, dtype: Any = None) -> Dict[str, Any]:
    """PEFT 命名 → 项目内部命名（`save_checkpoint` / `save_lora` 用的那一套）。

    导出时补上的 ``base_model.model.`` 前缀这里必须原样剥掉。差一个前缀，
    `load_state_dict` 会把每个键都当成"模型里没有的参数"，一个都不载入 ——
    `load_checkpoint` 虽然会把这种情况报出来，但直接改键名的手工转换不会。
    DDP 的 ``module.`` 前缀在 PEFT 目录里不该出现，顺手一起容忍。
    """
    stripped_prefixes = ("base_model.model.", "module.")
    converted: Dict[str, Any] = {}
    for key, value in tensors.items():
        if "lora_A" not in key and "lora_B" not in key:
            continue
        name = str(key)
        while name.startswith(stripped_prefixes):
            name = name[len(next(p for p in stripped_prefixes if name.startswith(p))):]
        value = value.detach() if hasattr(value, "detach") else value
        value = value.to("cpu")
        if dtype is not None:
            value = value.to(dtype)
        converted[name] = value.contiguous()

    if not converted:
        raise ValueError(
            "这组权重里没有任何 lora_A/lora_B —— 转出来会是空 adapter，"
            "而载入时未必报错，只会安静地按零 LoRA 推理。"
        )
    ranks = {
        int(t.shape[0]) for name, t in converted.items() if name.endswith("lora_A.weight")
    }
    if len(ranks) != 1:
        raise ValueError(f"LoRA rank 不一致：{sorted(ranks)}")
    return converted


def convert_peft_to_checkpoint(
    adapter_path: Union[str, Path],
    out_path: Union[str, Path],
    *,
    dtype: Any = None,
) -> Path:
    """PEFT adapter（目录或 ``adapter_model.safetensors``）→ 项目自己的 ``.pt``。

    刻意存成 ``{"model": {...}}`` 而不是裸的 ``{键: 张量}``：项目里读 adapter
    的地方走的都是 `load_checkpoint`，它只在 payload 里找 ``"model"`` 这一层，
    裸字典会被**静默跳过**（一个参数都不载入、还没有报错）。

    adapter 目录里有 ``adapter_config.json`` 时，顺便把它记进 payload 的
    ``lora_config``：rank 能从 ``lora_A`` 的形状反推，**alpha 不能**，而它决定
    缩放系数 alpha/r —— 不记下来的话，这条路走回 PEFT 时只能靠人记得传对。
    """
    import torch
    from safetensors.torch import load_file

    adapter_path = Path(adapter_path)
    source = (
        adapter_path / "adapter_model.safetensors" if adapter_path.is_dir() else adapter_path
    )
    if not source.exists():
        raise FileNotFoundError(f"找不到 adapter 权重：{source}")

    state = peft_to_lora_state(load_file(str(source)), dtype=dtype)
    config = _read_adapter_config(adapter_path)
    payload: Dict[str, Any] = {"model": state, "only_trainable": True}
    if config:
        claimed = config.get("r")
        actual = next(
            int(t.shape[0]) for name, t in state.items() if name.endswith("lora_A.weight")
        )
        if claimed is not None and int(claimed) != actual:
            logger.warning(
                "adapter_config.json 声称 r=%s，实际 rank=%d —— 以实际形状为准",
                claimed, actual,
            )
        payload["lora_config"] = config
        logger.info(
            "已记下 lora_config：r=%d alpha=%s（导回 PEFT 时会直接用它）",
            actual, config.get("lora_alpha"),
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    logger.info("PEFT adapter 已转成 %s（%d 个张量）", out_path, len(state))
    return out_path


def _read_adapter_config(adapter_path: Path) -> Dict[str, Any]:
    """读 PEFT 的 ``adapter_config.json``；读不到（或坏了）就返回空字典。

    这里刻意不抛异常：没有配置文件照样能转（rank 从形状推），只是丢掉了 alpha
    这一项，调用方按"缺 alpha"处理即可。
    """
    path = adapter_path / "adapter_config.json" if adapter_path.is_dir() else None
    if path is None or not path.exists():
        return {}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: BLE001 —— 坏配置不该让转换失败
        logger.warning("adapter_config.json 读不了（%s），按无配置处理", exc)
        return {}
    return config if isinstance(config, dict) else {}
