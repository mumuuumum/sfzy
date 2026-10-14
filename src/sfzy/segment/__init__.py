"""纯规则三要素切分器（原告诉请 / 被告辩称 / 法院事实与说理）。

    from sfzy.segment import Segmenter, SegmentCache

    seg = Segmenter().segment(summary)          # 实时（GRPO 候选摘要）
    cache = SegmentCache.load(path)             # 离线预热（人工摘要）
    cache.get_or_segment(summary)

这个包**只依赖标准库**：它要跑在 reward 这条不能拉起 torch / transformers 的
路径上（和 `sfzy.rl.reward_terms` 同样的约束）。
"""

from sfzy.segment.cache import SegmentCache, summary_key
from sfzy.segment.segmenter import (
    ELEMENT_NAMES,
    ELEMENTS,
    Confidence,
    ElementSpan,
    SegmentationResult,
    Segmenter,
    Sentence,
    Status,
    segment_summary,
    split_sentences,
)

__all__ = [
    "ELEMENTS",
    "ELEMENT_NAMES",
    "Status",
    "Confidence",
    "ElementSpan",
    "SegmentationResult",
    "Segmenter",
    "Sentence",
    "segment_summary",
    "split_sentences",
    "SegmentCache",
    "summary_key",
]
