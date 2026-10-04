"""六要素事实一致性 Judge（Qwen 0.5B 级）。

    from sfzy.judge import FactConsistencyJudge

    judge = FactConsistencyJudge("models/Qwen2.5-0.5B-Instruct", device="cuda:1")
    results = judge.judge_candidates(原文, [候选1, ..., 候选8])
    rewards = [r.weighted_reward for r in results]      # ∈ [0, 1]

设计动机（为什么不接着用点式打分、为什么拆六要素）见 `schema.py` 和
`judge.py` 的模块注释。
"""

from sfzy.judge.judge import (
    ExtractionFailure,
    FactConsistencyJudge,
    parse_six_json,
    parse_six_json_debug,
)
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
    empty_field_rule,
    summarize_scores,
)
from sfzy.judge.scorer import SixElementScorer

__all__ = [
    "FactConsistencyJudge",
    "SixElementScorer",
    "ExtractionFailure",
    "parse_six_json",
    "parse_six_json_debug",
    "SixElements",
    "JudgeResult",
    "ELEMENTS",
    "ELEMENT_ZH",
    "DEFAULT_WEIGHTS",
    "MIN_SCORE_ELEMENTS",
    "MAX_SCORE",
    "aggregate",
    "aggregate_coverage",
    "empty_field_rule",
    "summarize_scores",
]
