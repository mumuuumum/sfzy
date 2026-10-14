"""切分结果的缓存：人工摘要离线预热，GRPO 候选摘要实时算。

============================ 为什么要分两种用法 ============================
切分器本身是纯函数（毫秒级），但两条链路的要求完全不同：

  * **人工摘要（reference）**：训练前就固定了，而且每条会被反复切（算 reward、
    出报告、做对账）。适合一次性离线切完落盘，之后全是查表。
  * **GRPO 候选摘要（rollout 输出）**：每次采样都是新文本，根本没法预热；
    必须在训练循环里实时切。所以这条路径**只读缓存**，绝不写盘——否则
    rollout 会把磁盘写爆，也会把"随机采样文本"永久留在缓存里。

于是这里的 API 是：

    cache = SegmentCache.load(path)     # 离线预热产物
    res = cache.get(text)               # 命中就返回，不命中返回 None
    cache.put(text, result)             # 显式写（离线预热的批量循环里用）
    res = cache.get_or_segment(text)    # 实时路径：命中查表，不命中现算——
                                        # 现算结果**不落盘**（除非 cache.writable）

缓存键是正文的 sha1；值里额外存了字符数和正文开头 48 个字符，用来兜住哈希碰撞
（或"文本被换过"）的情况——口径不对时必须**报不命中**，而不是拿旧切分配新文本。
值里**不存整段正文**：摘要正文本身就有几 MB，重复存一份没必要；`get(text)` 命中时
把调用方传进来的正文绑回结果上，`text_of()` / `marked_text()` 照常可用。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union

from sfzy.segment.segmenter import (
    ELEMENTS,
    Confidence,
    ElementSpan,
    SegmentationResult,
    Segmenter,
    Status,
)

#: 校验用：缓存记录里保留的正文开头长度。
HEAD_CHARS = 48


def summary_key(text: str) -> str:
    """缓存键：正文的 sha1（十六进制）。"""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _status_of(value: str) -> Status:
    return Status(value)


def _confidence_of(value: str) -> Confidence:
    return Confidence(value)


def result_to_record(text: str, result: SegmentationResult) -> Dict[str, object]:
    """把切分结果压成一行可落盘的记录（含偏移量和识别来源，便于复盘）。

    正文只存开头 `HEAD_CHARS` 个字符用于校验，不存全文。
    """

    return {
        "key": summary_key(text),
        "n_chars": len(text),
        "head": text[:HEAD_CHARS],
        "complete": result.complete,
        "order_ok": result.order_ok,
        "spans": {k: result.spans[k].to_dict() for k in ELEMENTS},
    }


def result_from_record(record: Dict[str, object]) -> SegmentationResult:
    """缓存记录 → `SegmentationResult`（正文留空，命中时由 `get()` 绑定）。"""

    spans: Dict[str, ElementSpan] = {}
    raw_spans = record["spans"]
    for key in ELEMENTS:  # type: ignore[assignment]
        raw = raw_spans[key]  # type: ignore[index]
        spans[key] = ElementSpan(
            name=key,
            start=raw["start"],
            end=raw["end"],
            status=_status_of(raw["status"]),
            start_rule=raw.get("start_rule", ""),
            end_rule=raw.get("end_rule", ""),
            confidence=_confidence_of(raw.get("confidence", Confidence.LOW.value)),
            notes=list(raw.get("notes", [])),
        )
    return SegmentationResult(
        summary="",
        spans=spans,
        order_ok=bool(record.get("order_ok", True)),
        notes=["from_cache"],
    )


class SegmentCache:
    """摘要正文 → 切分结果。命中即返回，未命中可选现算。"""

    def __init__(
        self,
        entries: Optional[Dict[str, SegmentationResult]] = None,
        *,
        segmenter: Optional[Segmenter] = None,
        writable: bool = True,
    ) -> None:
        self._entries: Dict[str, SegmentationResult] = entries or {}
        #: 校验信息（字符数 + 正文开头），落盘/读盘都只带这一份，不带正文
        self._records: Dict[str, Dict[str, object]] = {}
        self.segmenter = segmenter or Segmenter()
        #: 实时链路（GRPO rollout）用 writable=False：只读不写，避免把随机
        #: 采样文本堆进内存/磁盘。
        self.writable = writable
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------ 基本操作
    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, text: str) -> bool:
        return summary_key(text) in self._entries

    def get(self, text: str) -> Optional[SegmentationResult]:
        res = self._entries.get(summary_key(text))
        if res is None:
            return None
        # 双重校验：哈希碰撞或文本被改动时要能发现，而不是静默错位。
        record = self._records.get(summary_key(text))
        if record is not None:
            if record["n_chars"] != len(text) or record["head"] != text[:HEAD_CHARS]:
                return None
        elif res.summary != text:
            return None
        # 把调用方的正文绑回结果，`text_of()` / `marked_text()` 才拿得到切片。
        if res.summary != text:
            res.summary = text
        return res

    def put(self, text: str, result: Optional[SegmentationResult] = None) -> SegmentationResult:
        """写入一条（不给 result 就现算）。`writable=False` 时静默跳过写入。"""

        res = result if result is not None else self.segmenter.segment(text)
        if self.writable:
            key = summary_key(text)
            self._entries[key] = res
            self._records[key] = {"n_chars": len(text), "head": text[:HEAD_CHARS]}
        return res

    def get_or_segment(self, text: str) -> SegmentationResult:
        """实时路径入口：命中查表，未命中现算（未命中时不写盘）。"""

        res = self.get(text)
        if res is not None:
            self.hits += 1
            return res
        self.misses += 1
        return self.segmenter.segment(text)

    # ------------------------------------------------ 批量预热
    def warm(self, texts: Iterable[str]) -> int:
        """离线预热：把一批文本切好放进缓存，返回新写入的条数。"""

        added = 0
        for text in texts:
            if summary_key(text) in self._entries:
                continue
            self.put(text)
            added += 1
        return added

    # ------------------------------------------------ 落盘 / 读盘
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        segmenter: Optional[Segmenter] = None,
        writable: bool = False,
    ) -> "SegmentCache":
        """读 JSONL 缓存。文件不存在时返回空缓存（离线预热前的正常状态）。"""

        entries: Dict[str, SegmentationResult] = {}
        records: Dict[str, Dict[str, object]] = {}
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    res = result_from_record(record)
                    entries[record["key"]] = res
                    records[record["key"]] = record
        cache = cls(entries, segmenter=segmenter, writable=writable)
        cache._records = records
        return cache

    def save(self, path: Union[str, Path]) -> int:
        """写 JSONL 缓存（按 key 排序，保证同一份数据两次落盘字节一致）。"""

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(self._entries.items(), key=lambda kv: kv[0])
        with p.open("w", encoding="utf-8") as f:
            for key, res in items:
                record = {"key": key, **self._records.get(key, {})}
                record.setdefault("n_chars", len(res.summary))
                record.setdefault("head", res.summary[:HEAD_CHARS])
                record["complete"] = res.complete
                record["order_ok"] = res.order_ok
                record["spans"] = {k: res.spans[k].to_dict() for k in ELEMENTS}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(items)

    def iter_records(self) -> Iterator[Tuple[str, SegmentationResult]]:
        yield from self._entries.items()


__all__ = [
    "SegmentCache",
    "summary_key",
    "result_to_record",
    "result_from_record",
]
