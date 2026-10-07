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


def test_用logprobs复刻受限解码():
    rt = api.APIRuntime("http://x/v1", "m", api_key="sk-test", max_logprobs=20)
    dist = [
        {"token": "3", "logprob": -0.1},
        {"token": "4", "logprob": -1.5},
        {"token": "2", "logprob": -2.0},
        {"token": "1", "logprob": -4.0},
        {"token": "0", "logprob": -6.0},
    ]

    def fake_post(messages, *, max_tokens, logprobs=False, top_logprobs=None):
        return {"choices": [{
            "message": {"content": "3"},
            "logprobs": {"content": [{"token": "3", "logprob": -0.1,
                                      "top_logprobs": dist}]},
        }], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}

    rt._post = fake_post
    scores, pmaxs, probs = rt.score_digits_batch(["p"])
    assert scores == [3]
    assert pmaxs[0] > 0.5
    assert len(probs[0]) == 5 and probs[0][3] == max(probs[0])


def test_没有logprobs时退回解析文本():
    rt = api.APIRuntime("http://x/v1", "m", api_key="sk-test")

    def fake_post(messages, *, max_tokens, logprobs=False, top_logprobs=None):
        return {"choices": [{"message": {"content": "4"}}]}

    rt._post = fake_post
    scores, pmaxs, probs = rt.score_digits_batch(["p"])
    assert scores == [4] and pmaxs == [None] and probs == [[]]


def test_ping不带json模式():
    """探活消息里没有 'json'，不能带 response_format=json_object（DeepSeek 会 400）。"""
    rt = api.APIRuntime("http://x/v1", "m", api_key="sk-test")
    seen = {}

    def fake_post(messages, *, max_tokens, logprobs=False, top_logprobs=None,
                  json_mode=True):
        seen["json_mode"] = json_mode
        return {"choices": [{"message": {"content": "pong"}}]}

    rt._post = fake_post
    assert rt.ping() == "pong"
    assert seen["json_mode"] is False


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
