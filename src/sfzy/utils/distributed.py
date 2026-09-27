"""分布式环境的探测与初始化（工程代码，已实现）。

设计原则：**所有进程相关的东西都收在这里。**

SFTTrainer 不自己读环境变量、不自己判断 rank，而是接收注入的
is_main_process。好处有三个：

  * 同一个 trainer 在 CPU 单卡 / GPU 单卡 / DDP 下都能跑；
  * 测试里能干净地构造 trainer，不用 mock 环境变量；
  * "谁来装配"这件事是明确的 —— 装配在 scripts/ 里。

本项目当前**只跑单卡**（Kaggle 单张 T4 足够放下 6B 的 QLoRA），
所以下面的分布式分支是"设计上可插拔、实现上先留好接口"。
真要上多卡时，脚本里包一层 DistributedDataParallel 即可，
trainer 一行都不用改。
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


def init_distributed(backend: Optional[str] = None) -> int:
    """初始化进程组，返回 local_rank；非分布式环境返回 -1。

    识别标准环境变量 RANK / WORLD_SIZE / LOCAL_RANK（torchrun 会自动设置）。
    WORLD_SIZE <= 1 时直接返回 -1，不初始化进程组 —— 这样单卡场景下
    这段代码完全无副作用，不会干扰本地调试和单元测试。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return -1

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """只有主进程该写日志、存 checkpoint、上报实验跟踪。

    多进程同时写同一个文件会互相覆盖，而这些问题在单卡下永远暴露不出来。
    """
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup() -> None:
    """训练结束时销毁进程组。"""
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def pick_device(preferred: str = "auto") -> torch.device:
    """选择训练设备。

    优先级：显式配置 > CUDA 可用 > CPU。

    preferred="auto" 时自动判断；显式写成 "cpu" 就老老实实用 CPU ——
    本地调试常常故意要用 CPU 来排除 GPU 相关的不确定性，
    所以显式配置必须被尊重，不能因为检测到 CUDA 就覆盖掉。
    """
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_environment() -> dict:
    """环境概况，训练开始时打一行日志，方便回溯问题。

    记录 PYTHONHASHSEED 是因为它只能由环境变量在进程启动时设定，
    运行时改了也不生效 —— 复现实验时它是常被忽略的一环。
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "distributed": is_distributed(),
        "rank": get_rank(),
        "world_size": get_world_size(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
    return info
