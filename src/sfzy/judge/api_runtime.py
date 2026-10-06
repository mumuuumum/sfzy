"""用 OpenAI 兼容的 Chat Completions API 做**抽取**（方案 A 的 API 那一半）。

============================ 为什么只做抽取 ============================
抽取是最难、最容易出错的一步（长文、要素散落、仲裁阶段陈述 vs 本案答辩），
值得用更强的 API 模型；判定（0~4 受限解码）要跑 `6要素 × 候选数` 次，放在
训练回路里用 API 又慢又贵，而且拿不到 logits、复刻不了 `pmax`。所以这里只实现
`generate_messages` / `generate_batch`，判定仍交给本地 runtime。

接口对齐 `sfzy/judge/runtime.py` 的 TorchRuntime：
  * `render(messages)`        —— 只用于日志/长度估算，不参与请求
  * `generate_messages(convs, max_new_tokens)` —— 会话直发（抽取走这条）
  * `generate_batch(prompts, max_new_tokens)`  —— 兼容旧调用：每条 prompt 当一条 user
  * `score_digits_batch`      —— 显式不支持（抛错），避免有人误把它当裁判

并发：API 是 I/O 密集，用线程池按 `concurrency` 并发；429/5xx/超时按指数退避重试。
只用标准库（urllib），不额外引入依赖。
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence
from urllib import error, request


class APIError(RuntimeError):
    """API 调用最终失败（重试也没救回来）。"""


class APIRuntime:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        api_key_env: str = "SFZY_JUDGE_API_KEY",
        timeout: float = 60.0,
        max_retries: int = 3,
        concurrency: int = 16,
        temperature: float = 0.0,
        response_format_json: bool = True,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if not base_url or not model:
            raise ValueError("APIRuntime 需要 base_url 和 model")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.concurrency = max(1, int(concurrency))
        self.temperature = float(temperature)
        self.response_format_json = bool(response_format_json)
        self.extra_body = dict(extra_body or {})
        self.extra_headers = dict(extra_headers or {})
        self._url = f"{self.base_url}/chat/completions"
        self.stats: Dict[str, int] = {
            "calls": 0, "retries": 0, "failed": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
        }
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 兼容接口
    def render(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join(m.get("content", "") for m in messages)

    def generate_messages(
        self, conversations: Sequence[List[Dict[str, str]]], max_new_tokens: int = 512
    ) -> List[str]:
        """批量会话 → 批量文本（顺序与输入一致，线程池并发）。"""
        convs = list(conversations)
        if not convs:
            return []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda c: self._chat(c, max_new_tokens), convs))

    def generate_batch(
        self, prompts: Sequence[str], max_new_tokens: int = 512
    ) -> List[str]:
        convs = [[{"role": "user", "content": p}] for p in prompts]
        return self.generate_messages(convs, max_new_tokens)

    def score_digits_batch(self, prompts, digits: str = "") -> Any:
        raise NotImplementedError(
            "APIRuntime 只做抽取（方案 A）；判定请用本地 runtime。"
            "要全 API 判定得另做一条支持 logprobs 的打分路径。"
        )

    def count_tokens(self, text: str) -> int:  # 没有本地 tokenizer，估个上限
        return len(text)

    # ---------------------------------------------------------------- HTTP
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _chat(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": int(max_new_tokens),
        }
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                req = request.Request(
                    self._url, data=body, headers=self._headers(), method="POST"
                )
                with request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                usage = data.get("usage") or {}
                with self._lock:
                    self.stats["calls"] += 1
                    self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    self.stats["completion_tokens"] += int(
                        usage.get("completion_tokens") or 0
                    )
                return (data["choices"][0]["message"]["content"] or "").strip()
            except error.HTTPError as exc:  # noqa: PERF203
                last = exc
                retryable = exc.code in (408, 409, 429, 500, 502, 503, 504)
                if not retryable or attempt >= self.max_retries:
                    detail = exc.read()[:300]
                    raise APIError(f"HTTP {exc.code}: {detail!r}") from exc
            except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last = exc
                if attempt >= self.max_retries:
                    raise APIError(f"{type(exc).__name__}: {exc}") from exc
            with self._lock:
                self.stats["retries"] += 1
            time.sleep(min(30.0, 1.5 * (2 ** attempt)) + random.random())
        raise APIError(f"重试 {self.max_retries} 次仍失败：{last}")


def build_api_runtime(cfg: Dict[str, Any]) -> APIRuntime:
    """从 `semantic.extract_api` 这一段配置构造。"""
    if not isinstance(cfg, dict):
        raise ValueError("extract_api 必须是 {base_url, model, ...} 形式的映射")
    return APIRuntime(
        base_url=str(cfg.get("base_url", "")),
        model=str(cfg.get("model", "")),
        api_key=cfg.get("api_key"),
        api_key_env=str(cfg.get("api_key_env", "SFZY_JUDGE_API_KEY")),
        timeout=float(cfg.get("timeout_s", 60)),
        max_retries=int(cfg.get("max_retries", 3)),
        concurrency=int(cfg.get("concurrency", 16)),
        temperature=float(cfg.get("temperature", 0.0)),
        response_format_json=bool(cfg.get("response_format_json", True)),
        extra_body=cfg.get("extra_body"),
        extra_headers=cfg.get("extra_headers"),
    )
