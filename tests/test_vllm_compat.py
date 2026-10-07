"""vllm_compat.py 的验收测试（纯标准库，不需要装 vLLM）。"""

from __future__ import annotations

import dataclasses
import sys
import types

from sfzy.utils.vllm_compat import filter_llm_kwargs, supported_fields


@dataclasses.dataclass
class _EngineArgsV06:
    """仿 0.6.x：有 swap_space。"""

    model: str = ""
    swap_space: int = 4
    max_logprobs: int = 20
    enforce_eager: bool = False


@dataclasses.dataclass
class _EngineArgsV31:
    """仿新版 V1：删了 swap_space。"""

    model: str = ""
    max_logprobs: int = 20
    enforce_eager: bool = False


def test_dataclass字段():
    assert supported_fields(_EngineArgsV06) == {
        "model", "swap_space", "max_logprobs", "enforce_eager",
    }


def test_过滤掉不认识的参数():
    kept, dropped = filter_llm_kwargs(
        {"model": "m", "swap_space": 4, "foo": 1}, engine_args_cls=_EngineArgsV06
    )
    assert kept == {"model": "m", "swap_space": 4}
    assert dropped == ["foo"]


def test_新版删掉swap_space_时不报错():
    kept, dropped = filter_llm_kwargs(
        {"model": "m", "swap_space": 4, "max_logprobs": 50},
        engine_args_cls=_EngineArgsV31,
    )
    assert kept == {"model": "m", "max_logprobs": 50}
    assert dropped == ["swap_space"]


def test_有kwargs兜底时原样返回():
    class _Any:
        def __init__(self, **kwargs):
            ...

    kept, dropped = filter_llm_kwargs({"a": 1}, engine_args_cls=_Any)
    assert kept == {"a": 1} and dropped == []


def test_不传cls时自动用vllm的EngineArgs(monkeypatch):
    pkg = types.ModuleType("vllm"); pkg.__path__ = []
    eng = types.ModuleType("vllm.engine"); eng.__path__ = []
    arg_utils = types.ModuleType("vllm.engine.arg_utils")
    arg_utils.EngineArgs = _EngineArgsV31
    monkeypatch.setitem(sys.modules, "vllm", pkg)
    monkeypatch.setitem(sys.modules, "vllm.engine", eng)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)

    kept, dropped = filter_llm_kwargs({"model": "m", "swap_space": 4})
    assert kept == {"model": "m"} and dropped == ["swap_space"]
