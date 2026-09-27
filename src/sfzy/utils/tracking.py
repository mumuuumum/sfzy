"""实验跟踪的抽象层（工程代码，已实现）。

trainer 只调 `tracker.log(metrics, step)` 和 `tracker.finish()`，
不知道后端是 swanlab、wandb，还是只往 stdout 打印。

两个设计选择：

1. **本地调试默认用 NullTracker**，只打印、不联网、不用登录。
   跑单元测试和调试数据管线时不该被实验跟踪拖住。

2. **swanlab 和 wandb 的 API 是兼容的**（swanlab 刻意对齐了 wandb），
   所以两个后端共用同一段初始化代码。国内直连 swanlab 比 wandb 快很多，
   这是个实际的工程选择，不是随意换个库。

断点续训时要把 `resume_id` 传进来接回原来那次实验，
否则一次训练会在面板上被拆成好几条断掉的曲线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol


class Tracker(Protocol):
    """实验跟踪的最小接口。trainer 只依赖这三个方法。"""

    def log(self, metrics: Dict[str, float], step: int) -> None:
        ...

    def finish(self) -> None:
        ...

    @property
    def run_id(self) -> Optional[str]:
        """当前 run 的 id，写进 checkpoint 供续训时接回去。"""
        ...


class NullTracker:
    """不做任何上报，只保证接口完整。"""

    def log(self, metrics: Dict[str, float], step: int) -> None:  # noqa: D102
        pass

    def finish(self) -> None:  # noqa: D102
        pass

    @property
    def run_id(self) -> Optional[str]:  # noqa: D102
        return None


class SwanlabWandbTracker:
    """swanlab / wandb 的共用实现 —— 两者 API 兼容。"""

    def __init__(
        self,
        backend: str,
        project: str,
        run_name: Optional[str] = None,
        resume_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if backend == "swanlab":
            import swanlab as module
        elif backend == "wandb":
            import wandb as module
        else:
            raise ValueError(f"未知的 tracking 后端: {backend}")

        self._module = module
        # resume_id 非空时用 resume="must"：宁可报错也不要静默新建一条 run，
        # 因为静默新建意味着你的续训曲线和原来的断开了，而你不会立刻发现。
        run = module.init(
            project=project,
            name=run_name,
            id=resume_id,
            resume="must" if resume_id else None,
            config=config or {},
        )
        self._run = run
        self._run_id = resume_id or getattr(run, "id", None)

    def log(self, metrics: Dict[str, float], step: int) -> None:  # noqa: D102
        # 显式传 step，否则各后端自己按调用次数编号，
        # 断点续训后会从头计数，曲线会出现重叠。
        self._module.log(dict(metrics), step=step)

    def finish(self) -> None:  # noqa: D102
        self._module.finish()

    @property
    def run_id(self) -> Optional[str]:  # noqa: D102
        return self._run_id


def build_tracker(
    backend: str = "none",
    project: str = "sfzy",
    run_name: Optional[str] = None,
    resume_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
) -> Tracker:
    """按配置构建 tracker。

    enabled=False（或非主进程）时直接返回 NullTracker ——
    多进程下每个 rank 都去 init 一遍会把面板弄乱。
    """
    if not enabled or backend in ("", "none", None):
        return NullTracker()
    return SwanlabWandbTracker(
        backend=backend,
        project=project,
        run_name=run_name,
        resume_id=resume_id,
        config=config,
    )


def make_run_name(prefix: str = "sfzy", **hyperparams: Any) -> str:
    """把关键超参写进 run 名字，方便在面板上一眼分辨。

    例：sfzy-lora_r8-a16-lr2e-4-ep3
    """
    parts = [prefix]
    for key, value in hyperparams.items():
        if value is None:
            continue
        text = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{key}{text}")
    return "-".join(parts)
