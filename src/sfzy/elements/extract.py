"""调 API 抽三要素：并发、缓存、容错。

============================ 缓存键为什么这么算 ============================
`sha1(正文 + 模型名 + prompt 指纹)`。三者缺一不可：

  * 正文——不同摘要不能串；
  * 模型名——换模型重抽是另一批结果；
  * prompt 指纹——**改了 prompt 旧结果必须失效**。抽取结果完全由 prompt 决定，
    只按正文缓存会在换 prompt 后拿到旧要素，而分数看起来还挺正常（最坏的一类错）。

人工摘要固定不变 → 离线抽一次、之后全查表；GRPO 候选摘要是新文本 → 现算。
缓存只写**成功**的抽取：失败是被缓存下来最要命，之后每次跑都复现同一个空结果。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sfzy.elements import client as client_mod
from sfzy.elements.client import ChatClient, Completion, diagnose_empty_completion
from sfzy.elements.prompts import PromptTemplate
from sfzy.elements.schema import ELEMENTS, ElementSet, parse_elements

logger = logging.getLogger("sfzy.elements")

#: 校验用：缓存记录里保留正文开头多少字符
HEAD_CHARS = 48


def cache_key(text: str, model: str, prompt_fingerprint: str) -> str:
    payload = f"{text}\x00{model}\x00{prompt_fingerprint}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


@dataclass
class Extraction:
    """一条摘要的抽取结果。"""

    text: str
    elements: ElementSet
    ok: bool
    raw: str = ""
    error: str = ""
    from_cache: bool = False
    model: str = ""
    prompt_fingerprint: str = ""
    #: 该要素是否在原始回复里出现过（用来区分"模型说没有"和"模型没写这个键"）
    reported: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elements": self.elements.to_dict(),
            "ok": self.ok,
            "error": self.error,
            "from_cache": self.from_cache,
            "model": self.model,
            "prompt": self.prompt_fingerprint,
            "reported": list(self.reported),
            "raw": self.raw[:2000],
        }


class ElementCache:
    """抽取结果缓存（JSONL）。`writable=False` 时只读——训练回路里用这个。"""

    def __init__(self, *, writable: bool = True) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}
        self.writable = writable
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str, text: str) -> Optional[Extraction]:
        record = self._entries.get(key)
        if record is None:
            return None
        if record.get("n_chars") != len(text) or record.get("head") != text[:HEAD_CHARS]:
            return None
        elements = ElementSet.from_dict(record.get("elements") or {})
        return Extraction(
            text=text,
            elements=elements,
            ok=bool(record.get("ok", True)),
            error=str(record.get("error", "")),
            from_cache=True,
            model=str(record.get("model", "")),
            prompt_fingerprint=str(record.get("prompt", "")),
            reported=tuple(record.get("reported") or ()),
            raw=str(record.get("raw", "")),
        )

    def put(self, key: str, extraction: Extraction) -> None:
        if not self.writable:
            return
        self._entries[key] = {
            "key": key,
            "n_chars": len(extraction.text),
            "head": extraction.text[:HEAD_CHARS],
            "model": extraction.model,
            "prompt": extraction.prompt_fingerprint,
            "ok": extraction.ok,
            "error": extraction.error,
            "reported": list(extraction.reported),
            "elements": extraction.elements.to_dict(),
            "raw": extraction.raw[:2000],
        }

    # ---------------------------------------------------------------- 落盘
    @classmethod
    def load(cls, path: Optional[Union[str, Path]], *, writable: bool = False) -> "ElementCache":
        cache = cls(writable=writable)
        if not path:
            return cache
        file = Path(path)
        if not file.exists():
            return cache
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("key")
                if key:
                    cache._entries[key] = record
        return cache

    def save(self, path: Union[str, Path]) -> int:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(self._entries.items(), key=lambda kv: kv[0])
        with file.open("w", encoding="utf-8") as handle:
            for _, record in items:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(items)


@dataclass
class ExtractStats:
    requested: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    retried: int = 0
    failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: List[str] = field(default_factory=list)
    #: 整批调用失败的原因（超时/限流/余额不足…），最多留 3 条
    api_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "retried": self.retried,
            "failed": self.failed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "errors": self.errors[:5],
            "api_errors": self.api_errors[:3],
        }


class ElementExtractor:
    """三要素抽取器：同一批文本去重 → 命中缓存 → 其余并发调 API。"""

    def __init__(
        self,
        runtime: Any,
        prompt: PromptTemplate,
        *,
        cache: Optional[ElementCache] = None,
        model: str = "",
        max_tokens: int = 1024,
        parse_retries: int = 1,
        json_mode: bool = True,
    ) -> None:
        self.runtime = runtime
        self.prompt = prompt
        self.cache = cache
        self.model = model or str(getattr(runtime, "model", ""))
        self.max_tokens = int(max_tokens)
        self.parse_retries = max(0, int(parse_retries))
        #: 直接对齐 runtime 的 HTTP 层：需要 `reasoning_content` 与 `finish_reason`
        self.client = ChatClient(runtime, max_tokens=self.max_tokens, json_mode=json_mode)
        self.stats = ExtractStats()

    # ---------------------------------------------------------------- 单条
    def _call(self, texts: Sequence[str]) -> List[Completion]:
        conversations = [self.prompt.render(text) for text in texts]
        completions = self.client.chat(conversations)
        for completion in completions:
            self.stats.prompt_tokens += completion.prompt_tokens
            self.stats.completion_tokens += completion.completion_tokens
        return completions

    def _call_safe(self, texts: Sequence[str]) -> Optional[List[Completion]]:
        """调 API；**整批**失败（超时/限流/余额不足）返回 None，由调用方标失败。

        单条/整批失败都不该让整轮评测崩掉：抽取失败会记成 `ok=False` 并计数，
        绝不会被当成"要素是空的"（那会把 API 故障算成模型质量差异）。
        """

        try:
            return self._call(texts)
        except Exception as exc:                                  # noqa: BLE001
            message = client_mod.explain_api_error(exc)
            logger.error("抽取 API 调用失败：%s", message.splitlines()[0])
            self.stats.api_errors.append(message)
            return None

    def _parse(self, text: str, completion: Optional[Completion]) -> Extraction:
        if completion is not None and completion.error:
            # 单条调用失败：记失败原因，**不当成空要素**
            return Extraction(
                text=text,
                elements=ElementSet(),
                ok=False,
                raw="",
                error=completion.error.splitlines()[0],
                model=self.model,
                prompt_fingerprint=self.prompt.fingerprint(),
            )
        raw = (completion.content if completion else "") or ""
        elements, ok = parse_elements(raw)
        if not ok and completion is not None and completion.reasoning:
            # 推理模型偶尔把 JSON 直接写在思维链末尾：能救就救，但记下来源
            salvaged, salvaged_ok = parse_elements(completion.reasoning)
            if salvaged_ok:
                elements, ok = salvaged, True
                raw = completion.reasoning
        reported: Tuple[str, ...] = ()
        if ok:
            # 模型明确写了哪些键（用于区分"没有这一要素"和"漏输出了键"）
            reported = tuple(name for name in ELEMENTS if name in (raw or ""))
        return Extraction(
            text=text,
            elements=elements,
            ok=ok,
            raw=raw or "",
            error="" if ok else diagnose_empty_completion(completion or Completion(content=raw)),
            model=self.model,
            prompt_fingerprint=self.prompt.fingerprint(),
            reported=reported,
        )

    def extract(self, text: str) -> Extraction:
        return self.extract_batch([text])[0]

    # ---------------------------------------------------------------- 批量
    def extract_batch(self, texts: Sequence[str]) -> List[Extraction]:
        texts = list(texts)
        self.stats.requested += len(texts)
        results: List[Optional[Extraction]] = [None] * len(texts)

        # 1) 去重 + 查缓存
        unique: Dict[str, int] = {}
        pending: List[str] = []
        for index, text in enumerate(texts):
            if not text:
                results[index] = Extraction(text=text, elements=ElementSet(), ok=False,
                                            error="空文本，跳过抽取", model=self.model,
                                            prompt_fingerprint=self.prompt.fingerprint())
                continue
            key = cache_key(text, self.model, self.prompt.fingerprint())
            if text in unique:
                continue
            unique[text] = index
            cached = self.cache.get(key, text) if self.cache is not None else None
            if cached is not None:
                results[index] = cached
                self.stats.cache_hits += 1
            else:
                pending.append(text)

        # 2) 未命中的并发调 API
        fetched: Dict[str, Extraction] = {}
        if pending:
            batch = list(pending)
            raws = self._call_safe(batch)
            self.stats.api_calls += len(batch)
            if raws is None:
                for text in batch:
                    extraction = self._parse(text, None)
                    extraction.error = self.stats.api_errors[-1].splitlines()[0]
                    fetched[text] = extraction
                self.stats.failed += len(batch)
                pending = []
                raws = []
            failures: List[str] = []
            for text, raw in zip(batch, raws):
                extraction = self._parse(text, raw)
                if not extraction.ok:
                    failures.append(text)
                fetched[text] = extraction
            # 3) 解析失败的重试（格式问题重发一次通常就好）
            for attempt in range(self.parse_retries):
                if not failures:
                    break
                self.stats.retried += len(failures)
                logger.warning("解析失败 %d 条，重试第 %d 次", len(failures), attempt + 1)
                retry_targets = list(failures)
                retry_raws = self._call_safe(retry_targets)
                self.stats.api_calls += len(retry_targets)
                if retry_raws is None:
                    break
                still: List[str] = []
                for text, raw in zip(retry_targets, retry_raws):
                    extraction = self._parse(text, raw)
                    if not extraction.ok:
                        still.append(text)
                    else:
                        fetched[text] = extraction
                failures = still
            for text in failures:
                self.stats.failed += 1
                message = fetched[text].error
                self.stats.errors.append(message)
                if message and message not in self.stats.api_errors:
                    self.stats.api_errors.append(message)

        # 4) 回填（重复文本共享同一结果）+ 写缓存
        for text, index in unique.items():
            if results[index] is None:
                results[index] = fetched[text]
            extraction = results[index]
            if extraction is not None and extraction.ok and self.cache is not None:
                self.cache.put(cache_key(text, self.model, self.prompt.fingerprint()), extraction)
            if extraction is None:                       # pragma: no cover - 逻辑兜底
                extraction = Extraction(text=text, elements=ElementSet(), ok=False,
                                        error="未取到结果", model=self.model,
                                        prompt_fingerprint=self.prompt.fingerprint())
                results[index] = extraction
        for index, text in enumerate(texts):
            if results[index] is None:                   # 重复文本：借用同文本的结果
                results[index] = results[unique.get(text, index)]  # type: ignore[assignment]

        missing = [i for i, r in enumerate(results) if r is None]
        if missing:                                      # pragma: no cover - 逻辑兜底
            raise RuntimeError(f"抽取结果与输入没对齐（缺第 {missing[:5]} 条）")
        return list(results)  # type: ignore[arg-type]


def build_extractor(
    api: Any,
    prompt: PromptTemplate,
    *,
    cache: Optional[ElementCache] = None,
    runtime: Any = None,
) -> ElementExtractor:
    """按配置搭一个抽取器（runtime 可注入，测试时不用打网络）。"""

    rt = runtime if runtime is not None else client_mod.build_runtime(api)
    return ElementExtractor(
        rt,
        prompt,
        cache=cache,
        model=str(getattr(api, "model", "")),
        max_tokens=int(getattr(api, "max_tokens", 1024)),
        parse_retries=int(getattr(api, "parse_retries", 1)),
    )


__all__ = [
    "Extraction",
    "ExtractStats",
    "ElementCache",
    "ElementExtractor",
    "build_extractor",
    "cache_key",
]
