"""rl/reward.py 的验收测试。

重点是两处最容易写错的地方：
  1. **事实归一化** —— "48000元" 和 "4.8万元" 必须是同一个事实，
     否则覆盖率算出来全是噪声
  2. **提取顺序** —— 法条号、金额、日期都要先挖掉，否则 4 位数字会被
     当成编号重复计入

    python -m pytest tests/test_reward.py -q
"""

from __future__ import annotations

import pytest

from sfzy.rl.reward import (
    DEFAULT_REWARD_CFG,
    RewardBreakdown,
    check_gate,
    compute_reward,
    compute_rewards,
    extract_facts,
    fact_coverage,
    fact_precision,
    length_reward,
    summarize_gate_reasons,
)


# ---------------------------------------------------------------- 事实提取

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


# ---------------------------------------------------------------- 覆盖率

def test_覆盖率_全命中():
    cov, total, missed = fact_coverage("判令被告支付48000元", "判令被告支付48000元")
    assert (cov, total, missed) == (1.0, 1, 0)


def test_覆盖率_单位不同也算命中():
    cov, _, _ = fact_coverage("判令归还4.8万元", "判令归还48000元")
    assert cov == 1.0


def test_覆盖率_部分命中():
    cov, total, missed = fact_coverage(
        "判令支付48000元", "判令支付48000元，于2015年7月19日付清"
    )
    assert total == 2 and missed == 1
    assert cov == pytest.approx(0.5)


def test_覆盖率_一个都没写():
    cov, _, missed = fact_coverage("被告应当归还借款", "判令归还48000元")
    assert cov == 0.0 and missed == 1


def test_覆盖率_参考没有事实时返回0():
    """参考里没有事实时无从判断，返回 0 而不是 1 —— 不给分也不假装满分。"""
    assert fact_coverage("随便什么", "这里没有任何数字")[0] == 0.0


# ---------------------------------------------------------------- 精确率

def test_精确率_事实都在原文里():
    assert fact_precision("判令支付48000元", "经查被告欠48000元") == 1.0


def test_精确率_有编造的数字():
    p = fact_precision("判令支付48000元和99999元", "经查被告欠48000元")
    assert p == pytest.approx(0.5)


def test_精确率_候选没写事实时返回1():
    """没写事实不是"不精确"，该由覆盖率去惩罚，不在这里重复扣分。"""
    assert fact_precision("被告应当归还借款", "经查被告欠48000元") == 1.0


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


# ---------------------------------------------------------------- 总分

def test_gated模式_被门控时总分是0():
    score, bd = compute_reward("太短", REF)
    assert score == 0.0
    assert bd.gated is True
    assert bd.gate_reason == "too_short"


def test_gated模式_通过后按权重组合():
    score, bd = compute_reward(REF, REF)
    assert bd.gated is False
    expected = 0.5 * bd.rouge_l + 0.5 * bd.fact_coverage
    assert score == pytest.approx(expected)


def test_flat模式_不被门控():
    """flat 是对照组：所有项加权求和，可以互相补偿。

    拿同一个输入跑两种模式才说明问题 —— gated 会把它拦下，flat 不会。
    """
    short = "太短"
    score_gated, bd_gated = compute_reward(short, REF)
    _, bd_flat = compute_reward(short, REF, cfg={"mode": "flat"})
    assert bd_gated.gated is True and score_gated == 0.0
    assert bd_flat.gated is False


def test_两种模式的分项相同():
    """A1/A2 的对比必须是干净的单变量：分项计算完全共用。"""
    _, bd_gated = compute_reward(REF, REF)
    _, bd_flat = compute_reward(REF, REF, cfg={"mode": "flat"})
    assert bd_gated.rouge_l == bd_flat.rouge_l
    assert bd_gated.fact_coverage == bd_flat.fact_coverage
    assert bd_gated.length_ratio == bd_flat.length_ratio


def test_漏事实会拉低奖励():
    """这是整个奖励设计的目的：漏掉原文里的金额必须扣分。"""
    # 用小的 min_chars，让测试句子专注在"漏事实"这一件事上，
    # 不要被默认的 60 字门控拦掉（那是另一个测试的事）
    cfg = {"gate": {"min_chars": 10}}
    ref = "判令被告归还原告借款本金48000元，并支付利息。判决如下：被告归还原告48000元。"
    full = "判令被告归还原告借款本金48000元，并支付利息。判决如下：被告归还原告48000元。"
    missing = "判令被告归还原告借款本金，并支付利息。判决如下：被告归还原告借款。"
    s_full, bd_full = compute_reward(full, ref, cfg=cfg)
    s_missing, bd_missing = compute_reward(missing, ref, cfg=cfg)
    assert bd_full.gated is False and bd_missing.gated is False, "不该被门控，测的是奖励值"
    assert bd_full.fact_coverage == 1.0
    assert bd_missing.fact_coverage == 0.0
    assert s_full > s_missing


def test_长度奖励():
    assert length_reward("abcde", "abcde", tolerance=0.5) == 1.0
    assert length_reward("a", "abcde", tolerance=0.5) == 0.0     # 偏离 80%
    assert 0.0 < length_reward("abcd", "abcde", tolerance=0.5) < 1.0


def test_权重可覆盖():
    cfg = {"weights": {"rouge_l": 1.0, "fact_coverage": 0.0, "length": 0.0}}
    score, bd = compute_reward(REF, REF, cfg=cfg)
    assert score == pytest.approx(bd.rouge_l)


# ---------------------------------------------------------------- 批量与统计

def test_批量打分():
    out = compute_rewards([REF, "太短"], [REF, REF])
    assert len(out) == 2
    assert out[1].gated is True


def test_门控原因统计():
    bds = compute_rewards([REF, "太短", REF * 3], [REF] * 3)
    stats = summarize_gate_reasons(bds)
    assert stats["total"] == 3
    assert stats["gated"] == 2
    assert stats["too_short"] == 1
    assert stats["length_too_long"] == 1


def test_默认配置是gated模式():
    assert DEFAULT_REWARD_CFG["mode"] == "gated"
    assert set(DEFAULT_REWARD_CFG["weights"]) == {"rouge_l", "fact_coverage", "length"}
