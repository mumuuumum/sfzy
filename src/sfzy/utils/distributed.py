"""分布式环境的探测与初始化（工程代码，已实现）。

设计原则：**所有进程相关的东西都收在这里。**

SFTTrainer 不自己读环境变量、不自己判断 rank，而是接收注入的
is_main_process。好处有三个：

  * 同一个 trainer 在 CPU 单卡 / GPU 单卡 / DDP 下都能跑；
  * 测试里能干净地构造 trainer，不用 mock 环境变量；
  * "谁来装配"这件事是明确的 —— 装配在 scripts/ 里。

本项目在 Kaggle 上跑 **2×T4 数据并行**（6B 的 QLoRA 一轮要 15 小时，
单卡装不下这 9 小时的会话限制）。设计上坚持"进程相关的东西全收在这里"：
脚本负责装配（clone 环境、建 trainer、包 DDP），trainer 只接收
注入进来的 is_main_process —— 所以同一份 trainer 在 CPU 单卡 / GPU 单卡 /
DDP 下都能跑，`tests/test_ddp_smoke.py` 用 CPU + gloo 也能验 DDP 的正确性。
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

    # **只有 NCCL（GPU）才需要绑定设备。**
    # 早先这里写的是「只要 CUDA 可用就 set_device(local_rank)」，有两个问题：
    #   1. 用 gloo 做 CPU 多进程时，rank=1 会去绑一张不存在的卡而失败；
    #   2. 本机实测 `CUDA_VISIBLE_DEVICES=""` 会让 torch 在
    #      cuda.is_available() 里 native 崩溃（free(): double free），
    #      所以想「强制 CPU」不能靠清空这个环境变量。
    if backend == "nccl" and torch.cuda.is_available():
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


def wrap_model(model: Any, local_rank: int) -> Any:
    """按需把模型包成 DistributedDataParallel。

    抽成函数是为了让**脚本和测试走同一条代码路径** —— 早先这段逻辑
    只写在 train_sft.py 里，结果 DDP 测试覆盖不到它，梯度没同步都测不出来。

    local_rank < 0 表示单进程，原样返回。

    两个容易踩的点：
      * **必须在注入 LoRA 之后包**。先包再注入的话，新加的模块不在
        DDP 的管辖范围内，梯度不会同步，两张卡会各训各的。
      * `device_ids` 要看**模型在哪块设备上**，而不是"CUDA 可不可用"。
        这两个在 DDP 测试里会分叉：机器有 GPU，但为了测 CPU 路径把模型放在
        CPU 上 —— 这时传 device_ids=[rank] 会直接报
            ValueError: device_ids ... only work with GPU modules or CPU modules,
                        but got device_ids [0] ... module parameters {device('cpu')}
        这个错误是 DDP 测试抓出来的，单卡跑永远碰不到。
    """
    if local_rank < 0:
        return model

    from torch.nn.parallel import DistributedDataParallel

    on_cuda = next(model.parameters()).device.type == "cuda"
    device_ids = [local_rank] if on_cuda else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=device_ids[0] if device_ids else None,
        # LoRA 的参数每层都会用到，没有 unused 参数，关掉能省一点开销
        find_unused_parameters=False,
    )


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
