"""tools/score_fact_judge.py 的纯函数验收（不加载模型）。

这个脚本是"SFT vs GRPO 的事实一致性"这一列的唯一来源，
所以它有两件事必须钉死：

  1. **逐条失败不能变成 0 分** —— 把失败当 0 会让统计系统性偏低，
     而且看起来像"模型变差了"
  2. **结果必须摊回正确的 id** —— 错位不报错，只是把 A 的分算到 B 头上

    python -m pytest tests/test_score_fact_judge.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sfzy.judge.schema import ELEMENTS, MAX_SCORE, aggregate

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "score_fact_judge", ROOT / "tools" / "score_fact_judge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


class _FakeJudge:
    """按"候选里有没有'驳回'"给分，用来验证对齐和失败处理。"""

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.seen: list = []

    def judge_candidates(self, document, candidates, candidate_ids=None):
        self.seen.append((document, candidates, candidate_ids))
        if self.fail_on and document == self.fail_on:
            raise RuntimeError("六要素抽取失败")
        out = []
        for cid, cand in zip(candidate_ids or [], candidates):
            raw = {e: MAX_SCORE for e in ELEMENTS}
            if "驳回" in cand:
                raw["judgment_result"] = 0
            out.append(aggregate(raw, candidate_id=str(cid)))
        return out


def test_逐条打分对齐到正确的_id():
    judge = _FakeJudge()
    rows = [
        {"id": "a", "source": "文书1", "output": "判决被告还款"},
        {"id": "b", "source": "文书2", "output": "判决驳回原告的诉讼请求"},
    ]
    out = TOOL.score_records(judge, rows)
    assert [r[0] for r in out] == ["a", "b"]
    assert out[0][1].weighted_reward == pytest.approx(1.0)
    assert out[1][1].weighted_reward == pytest.approx(0.70)     # 结果写反


def test_单条失败只记_error_不写假分数():
    judge = _FakeJudge(fail_on="坏文书")
    rows = [
        {"id": "ok", "source": "好文书", "output": "判决还款"},
        {"id": "bad", "source": "坏文书", "output": "判决还款"},
    ]
    out = TOOL.score_records(judge, rows)
    assert out[1][1] is None and "RuntimeError" in out[1][2]

    rec = TOOL.result_to_record(*out[1])
    assert rec["fact_reward"] is None and "error" in rec
    # ★ 不能变成 0 分
    assert rec["fact_reward"] != 0.0


def test_失败条目不参与均值():
    """把失败当 0 分会让"模型变差了"这种假结论直接冒出来。"""
    good = TOOL.result_to_record("a", aggregate({e: MAX_SCORE for e in ELEMENTS}), None)
    bad = TOOL.result_to_record("b", None, "APIError: 裁判调用失败")
    stats = TOOL.summarize([good, bad])
    assert stats["n"] == 1.0 and stats["n_failed"] == 1.0
    assert stats["mean_fact_reward"] == pytest.approx(1.0)      # 不是 0.5


def test_汇总复用需求的统计口径():
    rows = [TOOL.result_to_record(str(i), aggregate({e: i % 5 for e in ELEMENTS}), None)
            for i in range(5)]
    stats = TOOL.summarize(rows)
    assert stats["mean_fact_reward"] == pytest.approx(0.5)
    assert "mean_min_element_score" in stats
    assert all(f"mean_{e}_score" in stats for e in ELEMENTS)
    assert sum(stats[f"ratio_score_{lv}"] for lv in range(MAX_SCORE + 1)) == pytest.approx(1.0)


def test_汇总空输入不报错():
    assert TOOL.summarize([]) == {}
