"""断点续训的存取（工程代码，已实现）。

为什么必须有：Kaggle 单次会话上限 9 小时（batch 运行 12 小时），
6B 模型的 QLoRA 训练跑不完一轮是常态。没有断点续训，每次被掐都要重跑。

设计上的两个选择：
  1. **一个 checkpoint 一个文件**（step_000200.pt），而不是一个目录。
     方便按编号 prune，也方便直接下载下来挂到新环境里继续跑。
     注意：本项目**不做"自动找最新"** —— 续训必须由训练者明确指定文件，
     见 sft/trainer.py 的 resume()。
  2. **默认只存可训练参数**（only_trainable=True）。
     LoRA 场景下可训练参数只有几十 MB，而整个 6B 模型是十几 GB，
     Kaggle 的 /kaggle/working 配额撑不住每次都存全量。
     恢复时用 strict=False，因为只载入了一部分参数。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

_STEP_PATTERN = re.compile(r"step_(\d+)\.pt$")


def save_checkpoint(
    path: Union[str, Path],
    model: Any,
    optimizer: Any = None,
    scaler: Any = None,
    scheduler: Any = None,
    state: Any = None,
    only_trainable: bool = True,
) -> Path:
    """保存训练状态，返回写出的路径。

    参数：
        model        —— 任意 nn.Module
        optimizer/scaler/scheduler —— 可为 None；续训时它们的状态必须一起存，
                        否则优化器的动量、GradScaler 的缩放因子都会丢
        state        —— 任意可序列化对象（通常传 TrainState.to_dict()）
        only_trainable —— True 时只存 requires_grad=True 的参数（LoRA 场景）

    注意：state_dict 要 .detach().cpu() 之后再存，
    不然会把整个 GPU 显存快照一起写进文件，文件巨大且换设备读不了。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    full_state = model.state_dict()
    if only_trainable:
        trainable_names = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
        model_state = {k: v for k, v in full_state.items() if k in trainable_names}
    else:
        model_state = full_state

    payload: Dict[str, Any] = {
        "model": {k: v.detach().cpu() for k, v in model_state.items()},
        "only_trainable": only_trainable,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if state is not None:
        payload["state"] = state.to_dict() if hasattr(state, "to_dict") else state

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Union[str, Path],
    model: Any = None,
    optimizer: Any = None,
    scaler: Any = None,
    scheduler: Any = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """载入训练状态，返回 {"state": ..., "only_trainable": ..., "path": ...}。

    strict 默认 False：因为我们默认只保存了可训练参数，
    强行 strict 会因为缺少冻结层的权重而报错。

    map_location="cpu" 是为了让在 GPU 上存的 checkpoint 能在 CPU 上读
    （本地没有 CUDA 调试时需要），载入后再由模型自己放到对应设备。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)

    if model is not None and "model" in payload:
        missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
        if missing and not strict:
            pass  # 只存了可训练参数时，缺失冻结层权重是预期行为
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])

    return {
        "state": payload.get("state"),
        "only_trainable": payload.get("only_trainable", False),
        "path": str(path),
    }


def prune_checkpoints(ckpt_dir: Union[str, Path], keep_last_n: int = 3) -> int:
    """只保留最近 keep_last_n 个 checkpoint，返回删除的数量。

    配合 configs/sft.yaml 的 keep_last_n_checkpoints 使用。
    不清理的话，一个 200MB 的 adapter 存几十次就顶满 Kaggle 配额。
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir() or keep_last_n < 0:
        return 0

    candidates = []
    for entry in ckpt_dir.iterdir():
        match = _STEP_PATTERN.search(entry.name)
        if match and entry.is_file():
            candidates.append((int(match.group(1)), entry))

    candidates.sort(key=lambda item: item[0])
    to_remove = candidates[:-keep_last_n] if keep_last_n else candidates
    for _, path in to_remove:
        path.unlink()
    return len(to_remove)
