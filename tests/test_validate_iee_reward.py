"""scripts/validate_iee_reward.py 的纯函数验收（不调 API、不需要 torch）。

钉住两件最容易悄悄错的事：

  1. **EE 三维权重必须来自配置文件** —— 不是代码里那份 0.4/0.2/0.4 的默认值。
  2. **组内区分度指标定义** —— 零方差、极差、不同评分数量、成对可区分率。

    python -m pytest tests/test_validate_iee_reward.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "validate_iee_reward", ROOT / "scripts" / "validate_iee_reward.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()

W = {"semantic_redundancy": 0.3, "verbal_redundancy": 0.3, "abstraction_adequacy": 0.4}


def _record(gid, idx, sd, vr, aa, chars=100):
    dims = {"semantic_redundancy": sd, "verbal_redundancy": vr, "abstraction_adequacy": aa}
    rec = {
        "group_id": gid, "candidate_index": idx, "candidate_id": f"{gid}#{idx}",
        "candidate_chars": chars, "reference_chars": 100,
        "len_ratio_to_ref": chars / 100, "len_diff_to_ref": chars - 100,
    }
    for name, value in dims.items():
        rec[f"{name}_raw"] = value
        rec[f"{name}_norm"] = value / 4
    rec["iee"] = TOOL.expression_efficiency_score(dims, W)
    return rec


def test_weights_come_from_config_not_default():
    """回归护栏：权重必须来自 yaml 的 dimension_weights（0.3/0.3/0.4）。"""
    weights, cfg_info, _ = TOOL.load_iee_weights(ROOT / "configs" / "grpo_fact_coverage_inp_t4.yaml")
    assert weights == {
        "semantic_redundancy": 0.3, "verbal_redundancy": 0.3, "abstraction_adequacy": 0.4,
    }
    # 不能等于代码里的默认 0.4/0.2/0.4
    assert weights != {"semantic_redundancy": 0.4, "verbal_redundancy": 0.2, "abstraction_adequacy": 0.4}
    assert cfg_info["weight_sum"] == 1.0


def test_describe_percentiles():
    d = TOOL.describe([0, 1, 2, 3, 4])
    assert d["mean"] == 2.0 and d["median"] == 2.0
    assert d["min"] == 0 and d["max"] == 4
    assert d["p25"] == 1.0 and d["p75"] == 3.0


def test_group_metric():
    m = TOOL.group_metric([3, 3, 3, 3])          # 完全同分
    assert m["std"] == 0.0 and m["range"] == 0.0
    assert m["n_distinct"] == 1 and m["pair_diff_ratio"] == 0.0 and m["max_tie"] == 4
    m2 = TOOL.group_metric([4, 3, 2, 1])         # 四个都不同
    assert m2["n_distinct"] == 4 and m2["pair_diff_ratio"] == 1.0 and m2["max_tie"] == 1
    m3 = TOOL.group_metric([4, 4, 2, 2])         # 两两并列 → 6 对里 4 对不同
    assert round(m3["pair_diff_ratio"], 6) == round(4 / 6, 6) and m3["max_tie"] == 2


def _groups_from(records):
    seen = {}
    for r in records:
        seen.setdefault(r["group_id"], r["candidate_index"])
    return [{"group_id": g, "source": "原文", "reference": "参考",
             "candidates": [f"候选{i}" for i in range(4)]} for g in seen]


def test_analyze_marks_zero_variance_group():
    # g1 全同分（零方差），g2 有区分度
    recs = [_record("g1", i, 3, 3, 3) for i in range(4)]
    recs += [_record("g2", 0, 4, 4, 4, 200), _record("g2", 1, 3, 3, 3, 150),
             _record("g2", 2, 2, 2, 2, 100), _record("g2", 3, 2, 2, 2, 80)]
    summary = TOOL.analyze(recs, _groups_from(recs), W,
                           {"config_path": "x", "judge_api": {}, "dimension_weights": {},
                            "weight_sum": 1.0, "six_max_new_tokens": 512})
    assert summary["group"]["n_groups"] == 2
    assert summary["group"]["iee"]["zero_var_groups"] == 1
    assert summary["group"]["iee"]["nonzero_var_ratio"] == 0.5
    # g2 的 IEE 递减，长度也递减 → 组内应出现负相关
    assert summary["length_bias"]["within_group"]["iee~candidate_chars"]["spearman_r"] is not None
    assert summary["dim_discriminability"]["all_three_indistinguishable_groups"] == 1


def test_select_samples_categories():
    recs = [_record("g1", i, 3, 3, 3) for i in range(4)]              # 完全同分
    recs += [_record("g2", 0, 4, 4, 4, 300), _record("g2", 1, 1, 1, 1, 100),
             _record("g2", 2, 2, 2, 2, 200), _record("g2", 3, 2, 2, 2, 150)]
    samples = TOOL.select_samples(recs, _groups_from(recs), max_per_cat=2)
    cats = {s["category"] for s in samples}
    assert "fully_tied" in cats and "max_spread" in cats
    assert all(len(s["candidates"]) == 4 for s in samples)
