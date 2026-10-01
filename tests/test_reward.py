"""rl/reward.py 的验收测试。

奖励现在只有 `fact_judge` 一种：门控 + 六要素裁判分 + ROUGE-L。
规则事实抽取（`extract_facts`）保留，但**只给 prompt 池筛选**用，
不参与奖励。

    python -m pytest tests/test_reward.py -q
"""

from __future__ import annotations

import pytest

from sfzy.rl.reward import (
    DEFAULT_REWARD_CFG,
    check_gate,
    compute_reward,
    compute_rewards,
    extract_facts,
    summarize_gate_reasons,
)


# ---------------------------------------------------------------- 事实抽取（只用于 prompt 池筛选）

def test_金额单位归一化():
    """48000元 与 4.8万元 必须视为同一个事实。不折算的话信号全是噪声。"""
    assert extract_facts("共计48000元") == extract_facts("共计4.8万元")


def test_金额支持亿元():
    assert extract_facts("标的额2亿元") == extract_facts("标的额200000000元")


def test_日期按书写精度归一化():
    assert "date:2015-07-19" in extract_facts("2015年7月19日书写欠据")
    assert "date:2017-05" in extract_facts("2017年5月起")
    assert "date:2017" in extract_facts("2017年立案")


def test_法条号被排除():
    """项目决定不考虑法条，所以"第1079条"不该产生任何事实。"""
    assert extract_facts("依照《合同法》第1079条之规定") == set()


def test_法条号里的数字不会变成编号():
    """这是提取顺序的意义：法条号先挖掉，1079 就不会落到 ID 分支。"""
    facts = extract_facts("依照第1079条判决")
    assert not any(f.startswith("id:") for f in facts)


def test_提取顺序_金额不会被当成编号():
    facts = extract_facts("被告支付48000元")
    assert facts == {"money:48000.00"}
    assert not any(f.startswith("id:") for f in facts)


def test_提取顺序_年份不会被当成编号():
    facts = extract_facts("2015年7月19日")
    assert facts == {"date:2015-07-19"}


def test_纯编号能被抽出来():
    """案号里的 4 位以上数字（既不是金额也不是日期）算编号。"""
    facts = extract_facts("（2018）陕1023民初539号")
    assert "id:1023" in facts


def test_空文本返回空集合():
    assert extract_facts("") == set()


# ---------------------------------------------------------------- 门控

REF = "原被告系借款合同纠纷。原告请求判令被告归还借款本金48000元及利息。本院认为借贷关系合法有效。判决如下：被告归还原告48000元。"


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


# ---------------------------------------------------------------- 奖励

def test_fact_judge模式_通过门控后按权重组合():
    cfg = {"mode": "fact_judge", "weights": {"rouge_l": 0.3, "fact": 0.7}}
    score, bd = compute_reward(REF, REF, cfg=cfg, semantic=0.8)
    assert bd.gated is False
    assert bd.fact_judge == pytest.approx(0.8)
    assert score == pytest.approx(0.3 * bd.rouge_l + 0.7 * 0.8)


def test_fact_judge模式_不做量纲换算():
    """裁判返回的 weighted_reward 本就是 [0,1]，不能再按 0-100 除一遍。
    配错量纲不会报错，只会让事实项缩水 100 倍 —— 必须钉住。"""
    cfg = {"mode": "fact_judge", "semantic_scale": 100.0}
    _, bd = compute_reward(REF, REF, cfg=cfg, semantic=0.8)
    assert bd.fact_judge == pytest.approx(0.8)          # 不是 0.008


def test_fact_judge模式_裁判分越高奖励越高():
    cfg = {"mode": "fact_judge"}
    s_low, _ = compute_reward(REF, REF, cfg=cfg, semantic=0.0)
    s_high, _ = compute_reward(REF, REF, cfg=cfg, semantic=1.0)
    assert s_high > s_low


def test_fact_judge模式_被门控时总分0():
    cfg = {"mode": "fact_judge"}
    score, bd = compute_reward("太短", REF, cfg=cfg, semantic=1.0)
    assert bd.gated is True
    assert bd.gate_reason == "too_short"
    assert score == 0.0


def test_fact_judge模式_缺裁判分要报错():
    """静默降级只会让你以为在跑 Judge，其实没有。"""
    with pytest.raises(ValueError, match="fact_judge"):
        compute_reward(REF, REF, cfg={"mode": "fact_judge"})


def test_fact_judge模式_接到0到100量纲的裁判要报错():
    """把 0-100 量纲的分数接到 fact_judge 上，事实项会凭空大 100 倍 ——
    而且不报错。护栏必须拦住。"""
    with pytest.raises(ValueError, match="不在 \\[0,1\\]"):
        compute_reward(REF, REF, cfg={"mode": "fact_judge"}, semantic=80.0)


def test_未知的奖励模式直接报错():
    """rouge_only / flat / gated / gated_judge 已删除，写进来必须报错，
    而不是静默跑成别的口径。"""
    with pytest.raises(ValueError, match="fact_judge"):
        compute_reward(REF, REF, cfg={"mode": "gated"})


def test_权重可覆盖():
    cfg = {"mode": "fact_judge", "weights": {"rouge_l": 1.0, "fact": 0.0}}
    score, bd = compute_reward(REF, REF, cfg=cfg, semantic=0.9)
    assert score == pytest.approx(bd.rouge_l)


# ---------------------------------------------------------------- 批量与统计

def test_批量打分():
    out = compute_rewards([REF, "太短"], [REF, REF], semantic_scores=[0.5, 0.5])
    assert len(out) == 2
    assert out[1].gated is True


def test_批量打分_裁判分个数必须对齐():
    with pytest.raises(ValueError, match="裁判分个数"):
        compute_rewards([REF, REF], [REF, REF], semantic_scores=[0.5])


def test_批量打分_带裁判分():
    out = compute_rewards(
        [REF, REF], [REF, REF], cfg={"mode": "fact_judge"},
        semantic_scores=[0.1, 0.9],
    )
    assert out[0].fact_judge == pytest.approx(0.1)
    assert out[1].total > out[0].total


def test_门控原因统计():
    bds = compute_rewards(
        [REF, "太短", REF * 3], [REF] * 3, semantic_scores=[0.5] * 3
    )
    stats = summarize_gate_reasons(bds)
    assert stats["total"] == 3
    assert stats["gated"] == 2
    assert stats["too_short"] == 1
    assert stats["length_too_long"] == 1


def test_默认配置是fact_judge模式():
    assert DEFAULT_REWARD_CFG["mode"] == "fact_judge"
    assert set(DEFAULT_REWARD_CFG["weights"]) == {"rouge_l", "fact"}


def test_分项明细能导出():
    _, bd = compute_reward(REF, REF, semantic=0.6)
    detail = bd.to_dict()
    assert set(detail) == {"reward", "reward_rouge_l", "reward_fact_judge", "gated"}
    assert detail["reward_fact_judge"] == pytest.approx(0.6)
