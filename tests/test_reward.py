"""rl/reward.py 的验收测试：门控 + 多 reward 的归一化加权聚合 + 多信号。

奖励项本身怎么算由注册表（reward_terms）和裁判负责；这里只测：
  * 门控的分支
  * 权重归一化 + 逐项求和
  * 多信号怎么进来、缺了怎么办、量纲不对怎么拦

    python -m pytest tests/test_reward.py -q
"""

from __future__ import annotations

import pytest

from sfzy.rl.reward import (
    check_gate,
    compute_reward,
    compute_rewards,
    extract_facts,
    summarize_gate_reasons,
)


# ---------------------------------------------------------------- 事实抽取（只用于 prompt 池筛选）

def test_金额单位归一化():
    assert extract_facts("共计48000元") == extract_facts("共计4.8万元")


def test_金额支持亿元():
    assert extract_facts("标的额2亿元") == extract_facts("标的额200000000元")


def test_日期按书写精度归一化():
    assert "date:2015-07-19" in extract_facts("2015年7月19日书写欠据")
    assert "date:2017-05" in extract_facts("2017年5月起")
    assert "date:2017" in extract_facts("2017年立案")


def test_法条号被排除():
    assert extract_facts("依照《合同法》第1079条之规定") == set()


def test_法条号里的数字不会变成编号():
    facts = extract_facts("依照第1079条判决")
    assert not any(f.startswith("id:") for f in facts)


def test_提取顺序_金额不会被当成编号():
    facts = extract_facts("被告支付48000元")
    assert facts == {"money:48000.00"}
    assert not any(f.startswith("id:") for f in facts)


def test_提取顺序_年份不会被当成编号():
    assert extract_facts("2015年7月19日") == {"date:2015-07-19"}


def test_纯编号能被抽出来():
    facts = extract_facts("（2018）陕1023民初539号")
    assert "id:1023" in facts


def test_空文本返回空集合():
    assert extract_facts("") == set()


# ---------------------------------------------------------------- 门控

REF = "原被告系借款合同纠纷。原告请求判令被告归还借款本金48000元及利息。本院认为借贷关系合法有效。判决如下：被告归还原告48000元。"

# 六要素的内部权重（只能写在配置里；测试里也用同一份）
ELEMENT_WEIGHTS = {
    "case_type": 0.05,
    "plaintiff_claims": 0.15,
    "defendant_defenses": 0.10,
    "court_facts": 0.25,
    "legal_basis": 0.15,
    "judgment_result": 0.30,
}

# 两个 reward 都开，权重 0.3 / 0.7（和为 1，归一化后不变）
SPEC = {"terms": {
    "rouge_l": {"enabled": True, "weight": 0.3},
    "fact_consistency": {
        "enabled": True, "weight": 0.7,
        "element_weights": dict(ELEMENT_WEIGHTS),
    },
}}


def gate_cfg(**over):
    cfg = {"min_chars": 20, "length_ratio_range": [0.5, 1.5],
           "forbidden_prefixes": ["以下是", "摘要："], "require_result_marker": True}
    cfg.update(over)
    return cfg


def test_门控_通过():
    assert check_gate(REF, REF, gate_cfg()) is None


def test_门控_太短():
    assert check_gate("太短", REF, gate_cfg()) == "too_short"


def test_门控_长度超上限():
    assert check_gate(REF * 3, REF, gate_cfg()) == "length_too_long"


def test_门控_长度低于下限():
    long_ref = REF * 4
    assert check_gate(REF[:30], long_ref, gate_cfg()) == "length_too_short"


def test_门控_禁止前缀():
    assert check_gate("以下是摘要：" + REF, REF, gate_cfg()).startswith("prefix:")


def test_门控_缺判决结果标记():
    no_marker = "原被告系借款合同纠纷。" * 10
    assert check_gate(no_marker, no_marker, gate_cfg()) == "no_result_marker"


# ---------------------------------------------------------------- 多 reward 聚合

def test_多个reward按归一化权重求和():
    score, bd = compute_reward(
        REF, REF, spec=SPEC, judge_signals={"fact_consistency": 0.8}
    )
    assert bd.gated is False
    assert bd.values["fact_consistency"] == pytest.approx(0.8)
    assert score == pytest.approx(0.3 * bd.values["rouge_l"] + 0.7 * 0.8)


def test_权重会被归一到一():
    spec = {"terms": {
        "rouge_l": {"weight": 30},
        "fact_consistency": {"weight": 70, "element_weights": dict(ELEMENT_WEIGHTS)},
    }}
    score, bd = compute_reward(
        REF, REF, spec=spec, judge_signals={"fact_consistency": 1.0}
    )
    assert score == pytest.approx(0.3 * bd.values["rouge_l"] + 0.7 * 1.0)


def test_可以只开一个reward():
    spec = {"terms": {
        "rouge_l": {"enabled": True, "weight": 1.0},
        "fact_consistency": {"enabled": False},
    }}
    score, bd = compute_reward(REF, REF, spec=spec)     # 不需要裁判
    assert set(bd.values) == {"rouge_l"}
    assert score == pytest.approx(bd.values["rouge_l"])


def test_可以关掉rouge只留裁判():
    spec = {"terms": {
        "rouge_l": {"enabled": False},
        "fact_consistency": {
            "enabled": True, "weight": 1.0,
            "element_weights": dict(ELEMENT_WEIGHTS),
        },
    }}
    score, bd = compute_reward(
        REF, REF, spec=spec, judge_signals={"fact_consistency": 0.4}
    )
    assert set(bd.values) == {"fact_consistency"}
    assert score == pytest.approx(0.4)


def test_两个裁判项各按自己的权重进入总分():
    spec = {"terms": {
        "fact_consistency": {"enabled": True, "weight": 0.6,
                             "element_weights": dict(ELEMENT_WEIGHTS)},
        "element_coverage": {"enabled": True, "weight": 0.4,
                             "element_weights": dict(ELEMENT_WEIGHTS)},
    }}
    score, bd = compute_reward(
        REF, REF, spec=spec,
        judge_signals={"fact_consistency": 0.5, "element_coverage": 1.0},
    )
    assert set(bd.values) == {"fact_consistency", "element_coverage"}
    assert score == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)


def test_覆盖率信号缺失要报错():
    spec = {"terms": {
        "element_coverage": {"enabled": True, "weight": 1.0,
                             "element_weights": dict(ELEMENT_WEIGHTS)},
    }}
    with pytest.raises(ValueError, match="element_coverage"):
        compute_reward(REF, REF, spec=spec, judge_signals={"fact_consistency": 0.9})


def test_judge项缺信号要报错():
    with pytest.raises(ValueError, match="fact_consistency"):
        compute_reward(REF, REF, spec=SPEC)


def test_judge信号量纲不对要报错():
    with pytest.raises(ValueError, match="不在 \\[0,1\\]"):
        compute_reward(REF, REF, spec=SPEC, judge_signals={"fact_consistency": 80.0})


def test_不传spec直接报错():
    """没有奖励配置就不该算分 —— 权重只能来自配置文件。"""
    with pytest.raises(ValueError, match="rl.reward"):
        compute_reward(REF, REF)


# ---------------------------------------------------------------- 门控是全局开关

def test_被门控时总分0且不碰裁判():
    score, bd = compute_reward("太短", REF, spec=SPEC)   # 不给信号也不该报错
    assert bd.gated is True
    assert bd.gate_reason == "too_short"
    assert score == 0.0
    assert bd.values == {}


def test_门控可以整块关掉():
    spec = {"gate": {"enabled": False}, "terms": {"rouge_l": {"weight": 1.0}}}
    score, bd = compute_reward("太短", REF, spec=spec)
    assert bd.gated is False
    assert "rouge_l" in bd.values


# ---------------------------------------------------------------- 批量与统计

def test_批量打分_带多信号():
    out = compute_rewards(
        [REF, REF], [REF, REF], spec=SPEC,
        judge_signals={"fact_consistency": [0.1, 0.9]},
    )
    assert out[0].values["fact_consistency"] == pytest.approx(0.1)
    assert out[1].total > out[0].total


def test_批量打分_信号个数要对齐():
    with pytest.raises(ValueError, match="不一致"):
        compute_rewards(
            [REF, REF], [REF, REF], spec=SPEC,
            judge_signals={"fact_consistency": [0.1]},
        )


def test_门控原因统计():
    bds = compute_rewards(
        [REF, "太短", REF * 3], [REF] * 3, spec=SPEC,
        judge_signals={"fact_consistency": [0.5, 0.5, 0.5]},
    )
    stats = summarize_gate_reasons(bds)
    assert stats["total"] == 3
    assert stats["gated"] == 2
    assert stats["too_short"] == 1
    assert stats["length_too_long"] == 1


def test_分项明细能导出():
    _, bd = compute_reward(REF, REF, spec={"terms": {"rouge_l": {"weight": 1.0}}})
    detail = bd.to_dict()
    assert detail["reward"] == pytest.approx(bd.total)
    assert "reward_rouge_l" in detail
    assert detail["gated"] == 0.0
