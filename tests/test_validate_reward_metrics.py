"""scripts/validate_reward_metrics.py 的验收测试（纯 CPU，不调 API）。

重点是三件"算错了不报错、只是结论悄悄变错"的事：

  1. **FC 硬门控**：事实一致性六要素里任意一个原始分为 0 → 最终 Reward 必须为 0
     （不是按 FC 加权总分阈值判断）；
  2. **并列 Top-1**：用集合比较，不能取第一个候选；
  3. **排序一致性**：Kendall τ-b（全并列不可定义）与成对翻转/并列变化。

    python -m pytest tests/test_validate_reward_metrics.py -q
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_reward_metrics as V                       # noqa: E402

from sfzy.judge.schema import ELEMENTS                     # noqa: E402
from sfzy.rl.reward_spec import RewardSpec                 # noqa: E402

FC_WEIGHTS = {e: 1.0 / len(ELEMENTS) for e in ELEMENTS}

# 能通过文本门控（够长、字数比合理、没有禁前缀、含裁判结果标记）的样本
CAND = "原被告系借款合同纠纷。" + "本院认为借贷关系合法有效。" * 4 + "判决如下：被告归还原告借款。"
REF = CAND


def _config(fact_has_error: int = 1):
    return {
        "normalize_weights": True,
        "gate": {"fact_has_error": fact_has_error},
        "terms": {
            "fact_consistency": {"enabled": True, "weight": 0.5},
            "element_coverage": {"enabled": True, "weight": 0.5,
                                 "element_weights": dict(FC_WEIGHTS)},
        },
    }


def _ctx(raw_cfg):
    spec = RewardSpec.from_config(raw_cfg)
    spec_ungated = copy.deepcopy(spec)
    spec_ungated.gate.enabled = False
    spec_fc = copy.deepcopy(spec)
    spec_fc.gate.cfg["fact_has_error"] = 1
    return {"spec": spec, "spec_ungated": spec_ungated, "spec_fc": spec_fc,
            "missing_signals": []}


def _signals(fc_reward=0.75, cov_reward=0.8, has_error=0):
    return {"fact_consistency": fc_reward, "element_coverage": cov_reward,
            "fact_has_error": float(has_error)}


# ---------------------------------------------------------------- 事实错误门控

def test_FC判定有事实错误_最终reward必须为0():
    out = V.reward_fields(CAND, REF, _signals(fc_reward=0.0, has_error=1), _ctx(_config(1)))
    assert out["effective_reward"] == 0.0
    assert out["effective_reward_fc"] == 0.0
    assert out["gated"] is True
    assert out["gate_reason"] == "fact_has_error"
    assert out["gate"]["fact_has_error"] == 1
    # 门控前（raw）仍然是有值的加权和 —— 两个口径不能混
    assert out["raw_reward"] is not None and out["raw_reward"] > 0


def test_FC分数低但没有事实错误_不触发门控():
    """规则是"裁判判定存在明确事实错误"，不是"整体分低于某个阈值"。"""
    out = V.reward_fields(CAND, REF, _signals(fc_reward=0.25, has_error=0), _ctx(_config(1)))
    assert out["gated"] is False
    assert out["effective_reward"] == pytest.approx(out["raw_reward"])
    assert out["gate"]["fact_has_error"] == 0


def test_FC门控开关关闭时不拦():
    out = V.reward_fields(CAND, REF, _signals(fc_reward=0.0, has_error=1), _ctx(_config(0)))
    # 配置口径：开关关闭 → 不拦
    assert out["gated"] is False
    assert out["gate"]["fc_gate_enabled"] is False
    assert out["effective_reward"] == pytest.approx(out["raw_reward"])
    # 强制开启口径：拦下
    assert out["effective_reward_fc"] == 0.0
    assert out["gate"]["fc_strict_reason"] == "fact_has_error"


def test_文本门控与事实门控重叠时以文本门控为准():
    """文本门控先判：短摘要先被拦，gate_reason 是文本原因。"""
    out = V.reward_fields("太短", REF, _signals(has_error=1), _ctx(_config(1)))
    assert out["gated"] is True
    assert out["gate"]["text_reason"]  # 文本门控先命中


# ---------------------------------------------------------------- Top-1 并列

def test_top1集合_并列全部纳入():
    sa, sb = V.top1_sets([3, 3, 1], [3, 1, 1])
    assert sa == {0, 1} and sb == {0}
    assert sa != sb and (sa & sb) == {0}


def test_top1集合_全并列也算相同():
    sa, sb = V.top1_sets([2, 2, 2, 2], [1, 1, 1, 1])
    assert sa == {0, 1, 2, 3} == sb


def test_top1_不取第一个候选():
    sa, sb = V.top1_sets([1, 3, 3], [3, 3, 1])
    assert sa == {1, 2} and sb == {0, 1}
    assert (sa & sb) == {1}


# ---------------------------------------------------------------- Kendall τ-b

def test_kendall_tau_b_完全一致与完全相反():
    assert V.kendall_tau_b([4, 3, 2, 1], [4, 3, 2, 1]) == pytest.approx(1.0)
    assert V.kendall_tau_b([4, 3, 2, 1], [1, 2, 3, 4]) == pytest.approx(-1.0)


def test_kendall_tau_b_含并列仍可定义():
    val = V.kendall_tau_b([1, 2, 2, 3], [1, 2, 2, 3])
    assert val == pytest.approx(1.0)


def test_kendall_tau_b_全并列不可定义():
    assert V.kendall_tau_b([2, 2, 2, 2], [1, 2, 3, 4]) is None
    assert V.kendall_tau_b([1, 2, 3, 4], [0, 0, 0, 0]) is None


# ---------------------------------------------------------------- 成对关系

def test_成对关系_完全相反全部翻转():
    st = V.pairwise_relation_stats([4, 3, 2, 1], [1, 2, 3, 4])
    assert st["pairs"] == 6
    assert st["flip"] == 6 and st["keep"] == 0
    assert st["flip_rate"] == pytest.approx(1.0)
    assert st["tie_change"] == 0


def test_成对关系_完全一致全部保持():
    st = V.pairwise_relation_stats([1, 2, 3, 4], [1, 2, 3, 4])
    assert st["keep"] == 6 and st["flip"] == 0
    assert st["keep_rate"] == pytest.approx(1.0)


def test_成对关系_并列变化单独统计():
    st = V.pairwise_relation_stats([1, 2, 3, 4], [1, 1, 3, 4])
    assert st["pairs"] == 6
    assert st["keep"] == 5 and st["flip"] == 0
    assert st["tie_loss"] == 1 and st["tie_gain"] == 0
    assert st["tie_change_rate"] == pytest.approx(1 / 6)


def test_成对关系_两边都并列算保持():
    st = V.pairwise_relation_stats([2, 2, 1], [5, 5, 1])
    assert st["keep"] == 3 and st["both_tied_pairs"] == 1
    assert st["comparable_pairs"] == 2


# ---------------------------------------------------------------- 组级门控 / 门控影响

def _rec(gid, idx, raw, eff, gated=False, metrics_reward=None):
    reward = raw if metrics_reward is None else metrics_reward
    metrics = {
        name: {"reward": reward, "sub_raw": {s: 2 for s in ms.subs}}
        for name, ms in V.METRIC_BY_NAME.items()
    }
    return {
        "group_id": gid, "candidate_index": idx, "candidate_id": f"{gid}#{idx}",
        "metrics": metrics, "raw_reward": raw, "effective_reward": eff,
        "effective_reward_fc": eff, "gated": gated, "gate_reason": "",
    }


def test_组级门控覆盖与前后区分度():
    records = [
        _rec("g0", 0, 0.4, 0.0, True), _rec("g0", 1, 0.2, 0.0, True),
        _rec("g1", 0, 0.4, 0.4), _rec("g1", 1, 0.2, 0.0, True),
        _rec("g2", 0, 0.3, 0.3), _rec("g2", 1, 0.1, 0.1),
    ]
    grouped = V._grouped(records)
    order = ["g0", "g1", "g2"]
    gg = V._gate_group_stats(grouped, order)
    assert gg["all_gated_groups"] == 1
    assert gg["partial_gated_groups"] == 1
    assert gg["never_gated_groups"] == 1
    assert gg["distinguishable_before_tied_after"] == 1
    gi = V._gate_impact(grouped, order)
    assert gi["n_usable_groups"] == 3
    # g0 门控后完全同分 → Top-1 集合从单元素变成全并列，算"变化"
    assert gi["top1_changed_groups"] == 1
    # g0 的全并列 → τ 不可定义
    assert gi["kendall_tau_b"]["undefined_groups"] == 1
    assert gi["kendall_tau_b"]["valid_groups"] == 2


def test_门控前后奖励字段区分度统计():
    records = [_rec("g0", i, r, e) for i, (r, e) in
               enumerate([(0.4, 0.0), (0.3, 0.0), (0.2, 0.0), (0.1, 0.0)])]
    grouped = V._grouped(records)
    raw = V._reward_field_stats(records, grouped, ["g0"], "raw_reward")
    eff = V._reward_field_stats(records, grouped, ["g0"], "effective_reward")
    assert raw["group"]["nonzero_var_ratio"] == 1.0
    assert eff["group"]["nonzero_var_ratio"] == 0.0
    assert eff["global"]["zero_ratio"] == 1.0


# ---------------------------------------------------------------- 重复评分稳定性

def test_重复评分_排序稳定性逐字段统计():
    weights = {m.name: {s: 1.0 / len(m.subs) for s in m.subs} for m in V.METRICS}
    main = [
        _rec("g0", 0, 0.4, 0.4, metrics_reward=0.4),
        _rec("g0", 1, 0.3, 0.3, metrics_reward=0.3),
        _rec("g1", 0, 0.5, 0.5, metrics_reward=0.5),
        _rec("g1", 1, 0.5, 0.5, metrics_reward=0.5),
    ]
    repeat = [
        _rec("g0", 0, 0.1, 0.1, metrics_reward=0.1),
        _rec("g0", 1, 0.2, 0.2, metrics_reward=0.2),
        _rec("g1", 0, 0.5, 0.5, metrics_reward=0.5),
        _rec("g1", 1, 0.5, 0.5, metrics_reward=0.5),
    ]
    out = V.analyze_repeat(main, repeat, weights)
    per_field = out["per_repeat_index"]["0"]["per_field"]
    raw = per_field["raw_reward"]
    assert raw["n_pairs"] == 4
    assert raw["pairwise"]["pairs"] == 2
    assert raw["pairwise"]["flip"] == 1
    assert raw["pairwise"]["flip_rate"] == pytest.approx(0.5)
    assert raw["top1_strict_agreement"] == pytest.approx(0.5)
    assert raw["kendall_tau_b_valid_groups"] == 1
    assert raw["kendall_tau_b_undefined_groups"] == 1
    assert raw["kendall_tau_b_mean"] == pytest.approx(-1.0)
    assert raw["exact_rate"] == pytest.approx(0.5)
    assert raw["mean_abs_diff"] == pytest.approx((0.3 + 0.1 + 0 + 0) / 4)
    for name in weights:
        assert name in per_field
        assert "flip_rate" in per_field[name]["pairwise"]


def test_重复评分_缺失评分的有效样本数():
    weights = {m.name: {s: 1.0 for s in m.subs} for m in V.METRICS}
    main = [_rec("g0", 0, 0.4, 0.4), _rec("g0", 1, 0.2, 0.2)]
    repeat = [_rec("g0", 0, 0.4, 0.4)]          # 只有 1 条重复
    out = V.analyze_repeat(main, repeat, weights)
    raw = out["per_repeat_index"]["0"]["per_field"]["raw_reward"]
    assert raw["n_pairs"] == 1
    assert raw["n_groups"] == 1
    assert raw["n_usable_groups"] == 0
    assert raw["top1_strict_agreement"] is None
