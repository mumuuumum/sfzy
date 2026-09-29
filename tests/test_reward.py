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
    fact_score,
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


def test_覆盖率_参考没有事实时漏掉数必须为0():
    """参考里没有可验证事实时，**不能**把这批样本算成"漏了事实"。

    实测只有 38.3% 的参考摘要含金额/日期/编号，另外 61.7% 一条都没有。
    如果这些样本的 missed 计数不为 0，check_reward.py 里
    "漏事实 vs 干净"的对比就会被它们稀释，结论直接失真 ——
    而这是统计口径问题，不会报错。
    """
    coverage, n_ref, missed = fact_coverage("随便什么", "这里没有任何数字")
    assert (coverage, n_ref, missed) == (0.0, 0, 0)


# ---------------------------------------------------------------- 精确率

def test_精确率_事实都在原文里():
    assert fact_precision("判令支付48000元", "经查被告欠48000元") == 1.0


def test_精确率_有编造的数字():
    p = fact_precision("判令支付48000元和99999元", "经查被告欠48000元")
    assert p == pytest.approx(0.5)


def test_精确率_候选没写事实时返回1():
    """没写事实不是"不精确"，该由覆盖率去惩罚，不在这里重复扣分。"""
    assert fact_precision("被告应当归还借款", "经查被告欠48000元") == 1.0


# ---------------------------------------------------------------- 对称 F1

def test_事实F1_两边都没写事实时满分():
    """★ 这是换掉 coverage 的全部理由之一。

    参考里没有事实的样本占 38.3%（含事实）之外的那 61.7%。旧口径返回 0，
    而 0 在组内是常数 —— GRPO 组内归一化会把它抵消掉，整组零梯度。
    """
    assert fact_score("被告应当归还借款", "被告应当归还借款") == 1.0


def test_事实F1_参考没写但候选写了要扣分():
    """★ 另一半理由。实测：参考无事实的 827 条里，模型多写 3 个以上数字时
    官方总分从 0.5912 掉到 0.4921，而旧口径对此毫无惩罚。"""
    assert fact_score("判令被告支付48000元", "被告应当归还借款") == 0.0


def test_事实F1_参考写了但候选漏了要扣分():
    assert fact_score("被告应当归还借款", "判令归还48000元") == 0.0


def test_事实F1_全命中():
    assert fact_score("判令被告支付48000元", "判令被告支付48000元") == 1.0


def test_事实F1_单位不同也算命中():
    assert fact_score("判令归还4.8万元", "判令归还48000元") == 1.0


def test_事实F1_部分命中取调和平均():
    # 参考 2 个事实 {48000, 2015-07-19}，候选 1 个 {48000}
    # P = 1/1, R = 1/2 → F1 = 2*1*0.5/1.5 = 2/3
    cand = "判令支付48000元"
    ref = "判令支付48000元，于2015年7月19日付清"
    assert fact_score(cand, ref) == pytest.approx(2 / 3)


def test_事实F1_多写了参考没有的也要扣分():
    """堆数字刷分会被 F1 惩罚：precision 掉下来。"""
    ref = "判令归还48000元"
    cand = "判令归还48000元，另支付99999元和88888元"
    # P=1/3, R=1 → F1 = 2*1/3*1/(1/3+1) = 0.5。写对的那一个盖不住多堆的两个。
    assert fact_score(cand, ref) == pytest.approx(0.5)
    assert fact_score(cand, ref) < fact_score("判令归还48000元", ref)


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
    # 默认权重 0.3 ROUGE / 0.4 事实 / 0.3 裁判（gated 模式不用裁判）
    expected = 0.3 * bd.rouge_l + 0.4 * bd.fact_score
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
    assert bd_gated.fact_score == bd_flat.fact_score
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
    cfg = {"weights": {"rouge_l": 1.0, "fact": 0.0, "length": 0.0}}
    score, bd = compute_reward(REF, REF, cfg=cfg)
    assert score == pytest.approx(bd.rouge_l)


def test_旧配置的fact_coverage键仍被识别():
    """老 yaml 写的是 weights.fact_coverage。改名不该让老配置静默失效 ——
    静默失效的表现是"事实项乘了 0，等于没接"，很难发现。"""
    cfg = {"weights": {"rouge_l": 0.0, "fact_coverage": 1.0}}
    score, bd = compute_reward(REF, REF, cfg=cfg)
    assert score == pytest.approx(bd.fact_score)


# ---------------------------------------------------------------- 四种模式

def test_rouge_only模式_裸官方指标():
    """A1 对照组：不过门控、不看事实，回答"传统 RLVR 能到哪"。"""
    short_score, bd_short = compute_reward("太短", REF, cfg={"mode": "rouge_only"})
    assert bd_short.gated is False                       # 不过门控，这就是对照组的定义
    assert short_score == pytest.approx(bd_short.rouge_l)
    long_score, _ = compute_reward(REF, REF, cfg={"mode": "rouge_only"})
    assert long_score > short_score


def test_gated_judge模式_加入裁判分():
    cfg = {"mode": "gated_judge"}
    s_low, bd_low = compute_reward(REF, REF, cfg=cfg, semantic=20.0)
    s_high, bd_high = compute_reward(REF, REF, cfg=cfg, semantic=90.0)
    assert bd_low.semantic == pytest.approx(0.2)
    assert bd_high.semantic == pytest.approx(0.9)
    assert s_high > s_low
    assert s_high - s_low == pytest.approx(0.3 * 0.7)


def test_gated_judge模式_缺裁判分要报错而不是降级():
    """静默降级成 gated 是最坏的结果：你会以为在跑 A4，其实跑的是 A2。"""
    with pytest.raises(ValueError, match="gated_judge"):
        compute_reward(REF, REF, cfg={"mode": "gated_judge"})


def test_非judge模式也记录裁判分但不计分():
    """同一个 checkpoint 换口径离线重打分要免费。"""
    cfg = {"mode": "gated"}
    s_with, bd_with = compute_reward(REF, REF, cfg=cfg, semantic=90.0)
    s_without, _ = compute_reward(REF, REF, cfg=cfg)
    assert bd_with.semantic == pytest.approx(0.9)
    assert s_with == pytest.approx(s_without)


# ---------------------------------------------------------------- fact_judge 模式

def test_fact_judge模式_事实项来自裁判而不是规则():
    """需求要求事实一致性完全交给 Judge：金额/日期/法条一律不做规则匹配。

    证据：`REF` 里明明有金额（规则口径会抽到事实），但 fact_judge 模式下
    `n_ref_facts == 0` —— 规则提取根本没跑，事实项只认裁判分。
    """
    cfg = {"mode": "fact_judge", "weights": {"rouge_l": 0.3, "fact": 0.7}}
    _, bd = compute_reward(REF, REF, cfg=cfg, semantic=0.8)
    assert bd.n_ref_facts == 0
    assert bd.fact_score == 1.0            # 空 kinds，规则项退化成中性值
    assert bd.fact_judge == pytest.approx(0.8)
    assert bd.total == pytest.approx(0.3 * bd.rouge_l + 0.7 * 0.8)


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
    _, bd = compute_reward("太短", REF, cfg={"mode": "fact_judge"}, semantic=1.0)
    assert bd.gated is True
    assert bd.total == 0.0


def test_fact_judge模式_缺裁判分要报错():
    """和 gated_judge 一样：静默降级只会让你以为在跑 Judge，其实没有。"""
    with pytest.raises(ValueError, match="fact_judge"):
        compute_reward(REF, REF, cfg={"mode": "fact_judge"})


# ---------------------------------------------------------------- 批量与统计

def test_批量打分():
    out = compute_rewards([REF, "太短"], [REF, REF])
    assert len(out) == 2
    assert out[1].gated is True


def test_批量打分_裁判分个数必须对齐():
    with pytest.raises(ValueError, match="裁判分个数"):
        compute_rewards([REF, REF], [REF, REF], semantic_scores=[80.0])


def test_批量打分_带裁判分():
    out = compute_rewards(
        [REF, REF], [REF, REF], cfg={"mode": "gated_judge"},
        semantic_scores=[10.0, 90.0],
    )
    assert out[0].semantic == pytest.approx(0.1)
    assert out[1].total > out[0].total


def test_门控原因统计():
    bds = compute_rewards([REF, "太短", REF * 3], [REF] * 3)
    stats = summarize_gate_reasons(bds)
    assert stats["total"] == 3
    assert stats["gated"] == 2
    assert stats["too_short"] == 1
    assert stats["length_too_long"] == 1


def test_默认配置是gated模式():
    assert DEFAULT_REWARD_CFG["mode"] == "gated"
    assert set(DEFAULT_REWARD_CFG["weights"]) == {"rouge_l", "fact", "semantic", "length"}
    assert DEFAULT_REWARD_CFG["fact_term"] == "f1"


def test_分项明细能导出():
    """训练日志要把分项都记下来 —— 只盯总分判断不出"好在哪"。"""
    _, bd = compute_reward(REF, REF)
    d = bd.to_dict()
    for key in ("reward", "reward_rouge_l", "reward_fact_score", "reward_semantic",
                "fact_precision", "gated"):
        assert key in d
