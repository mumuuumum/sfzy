"""api_runtime.py 的验收测试（纯标准库，不需要 torch）。

直接按文件路径加载，绕开 `sfzy.judge.__init__` 对 torch 的依赖。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "sfzy_api_runtime", ROOT / "src/sfzy/judge/api_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["sfzy_api_runtime"] = module
    spec.loader.exec_module(module)
    return module


api = _load()


def test_必须给base_url和model():
    with pytest.raises(ValueError, match="base_url"):
        api.APIRuntime("", "m")
    with pytest.raises(ValueError, match="base_url"):
        api.APIRuntime("http://x/v1", "")


def test_缺密钥直接报清楚(monkeypatch):
    monkeypatch.delenv("SFZY_JUDGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SFZY_JUDGE_API_KEY"):
        api.APIRuntime("http://x/v1", "m")


def test_密钥自动去掉首尾空白(monkeypatch):
    monkeypatch.setenv("SFZY_JUDGE_API_KEY", "  sk-abc123456789  \n")
    rt = api.APIRuntime("http://x/v1", "m")
    assert rt.api_key == "sk-abc123456789"


def test_generate_messages_保序并发():
    rt = api.APIRuntime("http://x/v1", "m", api_key="sk-test", concurrency=4)
    seen = []

    def fake_chat(messages, n):
        seen.append(messages[0]["content"])
        return messages[0]["content"].upper()

    rt._chat = fake_chat            # 实例上打桩，不发真实请求
    convs = [[{"role": "user", "content": str(i)}] for i in range(10)]
    assert rt.generate_messages(convs, 16) == [str(i).upper() for i in range(10)]
    assert sorted(seen) == [str(i) for i in range(10)]


def test_generate_batch_把prompt当user():
    rt = api.APIRuntime("http://x/v1", "m", api_key="sk-test", concurrency=2)
    got = {}

    def fake_chat(messages, n):
        got[messages[0]["content"]] = messages
        return "ok"

    rt._chat = fake_chat
    assert rt.generate_batch(["a", "b"], 8) == ["ok", "ok"]
    assert got["a"] == [{"role": "user", "content": "a"}]


def test_不做判定():
    rt = api.APIRuntime("http://x/v1", "m", api_key="sk-test")
    with pytest.raises(NotImplementedError, match="只做抽取"):
        rt.score_digits_batch(["p"])


def test_build_api_runtime_读配置():
    rt = api.build_api_runtime({
        "base_url": "http://x/v1/", "model": "m", "api_key": "sk-test", "concurrency": 6,
        "max_retries": 1, "timeout_s": 5,
    })
    assert rt.base_url == "http://x/v1"      # 末尾斜杠归一化
    assert rt._url == "http://x/v1/chat/completions"
    assert rt.concurrency == 6 and rt.max_retries == 1 and rt.timeout == 5
    with pytest.raises(ValueError, match="extract_api"):
        api.build_api_runtime("not-a-dict")
