"""跨 vLLM 版本的 EngineArgs 参数兼容。

同一个脚本要在多个 vLLM 版本上跑（本机 0.6.x、云端 0.3x），而 EngineArgs 的
字段会随版本增删。已知的一处：`swap_space` 在 0.6.x 还有，新版的 V1 引擎不再
使用 CPU swap，直接把它从 EngineArgs 里删了 —— 于是 `LLM(swap_space=...)`
报 `TypeError: EngineArgs.__init__() got an unexpected keyword argument`。

这里按**当前安装版本实际支持的字段**过滤传给 `LLM(...)` 的 kwargs，把不认识的
丢掉并回报给调用方（打日志），而不是硬编码"哪个版本有什么"。
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Dict, List, Optional, Set, Tuple


def supported_fields(cls: Any) -> Optional[Set[str]]:
    """`cls` 构造时接受的字段名；判断不出来（有 **kwargs 兜底）时返回 None。"""
    try:
        return {f.name for f in dataclasses.fields(cls)}
    except TypeError:
        pass
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return {name for name in params if name != "self"}


def filter_llm_kwargs(
    kwargs: Dict[str, Any], engine_args_cls: Any = None
) -> Tuple[Dict[str, Any], List[str]]:
    """返回 `(保留的 kwargs, 被丢弃的键)`。

    `engine_args_cls` 不传就自动用当前 vLLM 的 `EngineArgs`；vLLM 没装/取不到
    时原样返回（不擅自丢参数）。
    """
    if engine_args_cls is None:
        try:
            from vllm.engine.arg_utils import EngineArgs  # noqa: PLC0415

            engine_args_cls = EngineArgs
        except Exception:  # noqa: BLE001 - 没装 vLLM 时不该在这里炸
            return dict(kwargs), []

    fields = supported_fields(engine_args_cls)
    if fields is None:
        return dict(kwargs), []
    kept = {k: v for k, v in kwargs.items() if k in fields}
    dropped = [k for k in kwargs if k not in fields]
    return kept, dropped
