"""LLM Judge：事实一致性 / 关键要素覆盖率 / 信息必要性（INP）。

    from sfzy.judge import FactConsistencyJudge

    judge = FactConsistencyJudge(
        runtime=..., tasks=("fact_consistency", "element_coverage"),
    )
    results = judge.judge_candidates(原文, [候选1, ..., 候选8], reference=人工摘要)
    signals = [r.signals for r in results]   # {"fact_consistency": …, "element_coverage": …}

不再抽取六要素：每路判定直接吃全文、输出六个要素分（INP 输出逐命题必要性）。
prompt 在 `prompts.py`，权重与聚合在 `schema.py`，判定流程在 `judge.py`。
"""

from sfzy.judge.judge import FactConsistencyJudge, parse_inp, parse_six_scores
from sfzy.judge.schema import (
    DEFAULT_WEIGHTS,
    ELEMENTS,
    ELEMENT_ZH,
    MAX_SCORE,
    MIN_SCORE_ELEMENTS,
    JudgeResult,
    SixElements,
    aggregate,
    aggregate_coverage,
    summarize_scores,
)
from sfzy.judge.scorer import SixElementScorer

__all__ = [
    "FactConsistencyJudge",
    "SixElementScorer",
    "parse_six_scores",
    "parse_inp",
    "SixElements",
    "JudgeResult",
    "ELEMENTS",
    "ELEMENT_ZH",
    "DEFAULT_WEIGHTS",
    "MIN_SCORE_ELEMENTS",
    "MAX_SCORE",
    "aggregate",
    "aggregate_coverage",
    "summarize_scores",
]
