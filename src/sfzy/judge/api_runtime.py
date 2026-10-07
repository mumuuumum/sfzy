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
import logging
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence
from urllib import error, request

logger = logging.getLogger("sfzy.api_runtime")

# 判定分数词表，和 sfzy/judge/runtime.py 的 DIGITS 一致（这里不 import 它，
# 免得把 torch 依赖带进来）。
DIGITS = "01234"


class APIError(RuntimeError):
    """API 调用最终失败（重试也没救回来）。"""


def _mask(secret: str) -> str:
    if len(secret) <= 12:
        return f"***（长度 {len(secret)}）"
    return f"{secret[:6]}…{secret[-4:]}（长度 {len(secret)}）"


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
        max_logprobs: int = 20,
        response_format_json: bool = True,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if not base_url or not model:
            raise ValueError("APIRuntime 需要 base_url 和 model")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        # 一定要 strip：从 notebook/环境变量里粘过来的 key 常带换行或空格，
        # 带着空白发出去就是 401（"Authentication Fails"），但看起来 key 是对的。
        self.api_key = (api_key or os.environ.get(api_key_env) or "").strip()
        if not self.api_key:
            raise ValueError(
                f"抽取 API 没有拿到密钥：环境变量 {api_key_env} 为空/未导出。\n"
                "  排查：① 在**同一个 shell**里 export 后再启动 python；"
                "② 值用单引号包住，别让 $、! 之类被展开；"
                "③ 确认没把占位符（sk-axxxxx）当成真 key。"
            )
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.concurrency = max(1, int(concurrency))
        self.temperature = float(temperature)
        # 判定时向 API 要多少个备选 logprob（0~4 五个数字一定在 top-20 里）
        self.max_logprobs = max(1, int(max_logprobs))
        self.response_format_json = bool(response_format_json)
        self.extra_body = dict(extra_body or {})
        self.extra_headers = dict(extra_headers or {})
        self._url = f"{self.base_url}/chat/completions"
        self.stats: Dict[str, int] = {
            "calls": 0, "retries": 0, "failed": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
        }
        self._lock = threading.Lock()
        logger.info(
            "抽取 API：base_url=%s model=%s key=%s（来自 $%s）",
            self._url, self.model, _mask(self.api_key), api_key_env,
        )

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

    def score_digits_messages(
        self, conversations: Sequence[List[Dict[str, str]]], digits: str = DIGITS
    ) -> Any:
        """判定：一次前向（max_tokens=1）+ logprobs，复刻本地受限解码。

        本地版是在 `{'0','1','2','3','4'}` 五个 token 的 logprob 上 softmax、
        取 argmax；这里用 OpenAI 兼容的 `logprobs=True, top_logprobs=N` 拿到同一
        位置的备选分布，按 **token 文本**（不是 id，跨供应商 id 不一样）匹配数字，
        再 softmax。拿不到 logprobs 时退回解析生成文本里的第一个数字（此时没有
        `pmax`）。
        """
        convs = list(conversations)
        if not convs:
            return [], [], []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(self._score_one, convs))
        return (
            [r[0] for r in results],
            [r[1] for r in results],
            [r[2] for r in results],
        )

    def score_digits_batch(self, prompts, digits: str = DIGITS) -> Any:
        """兼容旧调用：每条 prompt 当一条 user 消息（会丢掉 system 角色）。"""
        convs = [[{"role": "user", "content": p}] for p in prompts]
        return self.score_digits_messages(convs, digits)

    def _score_one(self, messages: List[Dict[str, str]]):
        body = self._post(messages, max_tokens=1, logprobs=True,
                          top_logprobs=self.max_logprobs)
        choice = (body.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()

        logprobs = choice.get("logprobs") or {}
        content_lp = logprobs.get("content") or []
        best_logprob: Dict[str, float] = {}
        if content_lp:
            first = content_lp[0] or {}
            entries = list(first.get("top_logprobs") or [])
            if first.get("token") is not None:
                entries.append({"token": first.get("token"),
                                "logprob": first.get("logprob", float("-inf"))})
            for entry in entries:
                token = str(entry.get("token", "")).strip()
                if token in DIGITS:
                    value = float(entry.get("logprob", float("-inf")))
                    best_logprob[token] = max(best_logprob.get(token, float("-inf")), value)

        if best_logprob:
            logits = [best_logprob.get(d, float("-inf")) for d in DIGITS]
            top = max(logits)
            exps = [math.exp(v - top) if v != float("-inf") else 0.0 for v in logits]
            total = sum(exps) or 1.0
            probs = [e / total for e in exps]
            best = max(range(len(probs)), key=lambda i: probs[i])
            return int(best), float(probs[best]), [round(p, 4) for p in probs]

        match = re.search(r"[0-4]", content)
        if match:
            return int(match.group()), None, []      # 没有 logprobs，拿不到 pmax
        raise APIError(f"判定返回里找不到 0~4 的分数：content={content!r}")

    def count_tokens(self, text: str) -> int:  # 没有本地 tokenizer，估个上限
        return len(text)

    def ping(self) -> str:
        """发一次最小请求，验证 base_url / 模型名 / 鉴权是否可用。

        **不带** `response_format=json_object`：DeepSeek 等供应商要求用 JSON 模式
        时 prompt 里必须出现 "json"，而探活消息没有，会被 400 拒掉（和鉴权无关）。
        """
        return self._chat([{"role": "user", "content": "ping"}], 1, json_mode=False)

    # ---------------------------------------------------------------- HTTP
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _chat(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 512,
        *, json_mode: bool = True,
    ) -> str:
        body = self._post(messages, max_tokens=max_new_tokens, json_mode=json_mode)
        choice = (body.get("choices") or [{}])[0]
        return ((choice.get("message") or {}).get("content") or "").strip()

    def _post(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        json_mode: bool = True,
    ) -> Dict[str, Any]:
        """发一次 chat.completions，返回解析后的 JSON（带重试）。"""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": int(max_tokens),
        }
        if self.response_format_json and json_mode and not logprobs:
            payload["response_format"] = {"type": "json_object"}
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs:
                payload["top_logprobs"] = int(top_logprobs)
        payload.update(self.extra_body)
        json_mode_active = "response_format" in payload

        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
                return data
            except error.HTTPError as exc:  # noqa: PERF203
                last = exc
                detail = exc.read()[:300]
                # 有的供应商（DeepSeek）要求用 json_object 时 prompt 里必须出现
                # "json"；我们的抽取 prompt 有，但换个 prompt/供应商就可能没有。
                # 命中这条就丢掉 response_format 直接重试（普通模式 + 我们的三级
                # JSON 容错解析），别让一个格式开关把整批打分打崩。
                if (
                    exc.code == 400
                    and json_mode_active
                    and b"json_object" in detail
                    and b"response_format" in detail
                ):
                    payload.pop("response_format", None)
                    json_mode_active = False
                    logger.warning(
                        "供应商要求 prompt 含 'json' 才允许 json_object，已退回普通模式重试"
                    )
                    continue
                retryable = exc.code in (408, 409, 429, 500, 502, 503, 504)
                if not retryable or attempt >= self.max_retries:
                    if exc.code in (401, 403):
                        raise APIError(
                            f"HTTP {exc.code}: {detail!r}\n"
                            f"  鉴权失败。base_url={self.base_url}，"
                            f"key={_mask(self.api_key)}（取自 ${self.api_key_env}）。常见原因：\n"
                            "  ① 这个环境变量没在**运行 python 的那个 shell**里导出（或值带换行/被截断）；\n"
                            "  ② key 与 base_url 不是同一家（OpenAI 的 key 不能用于 DeepSeek，反之亦然）；\n"
                            "  ③ key 过期 / 额度被停 / 未实名。\n"
                            "  先跑 `python scripts/score_reward_vllm.py --check-extract-api` 直连验证。"
                        ) from exc
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
    """从 `semantic.extract_api` / `semantic.judge_api` 这一段配置构造。"""
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
        max_logprobs=int(cfg.get("max_logprobs", 20)),
        response_format_json=bool(cfg.get("response_format_json", True)),
        extra_body=cfg.get("extra_body"),
        extra_headers=cfg.get("extra_headers"),
    )
