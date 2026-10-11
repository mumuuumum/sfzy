"""GRPO 监控层的验收测试。

覆盖需求第十四节：分组、组内标准化、零方差/全正、过滤口径、优势 raw/effective、
Gate 前后、六要素与覆盖率统计、EE 维度、相关系数常量组、消融并列、
Ratio/KL/长度归一化、SwanLab step 对齐，以及"监控不改变训练"。

    python -m pytest tests/test_monitor.py -q
"""

from __future__ import annotations

import math

import pytest

from sfzy.rl.monitor import (
    DEFAULT_MONITOR,
    GRPOMonitor,
    MonitorBatch,
    MonitorConfig,
    _group_advantages,
    _mean_group_spearman,
    _percentile,
    _pstd,
    _spearman,
)
from sfzy.judge.schema import EE_DIMENSIONS, ELEMENTS


def make_config(**over) -> MonitorConfig:
    cfg = dict(DEFAULT_MONITOR)
    cfg.update({"log_interval": 1, "reward_log_interval": 1, "relation_log_interval": 1})
    cfg.update(over)
    return MonitorConfig.from_config(cfg)


def make_result(fact_score=4, has_error=0, coverage_raw=None, ee_dims=None):
    """新的 FC 口径：整体分 + 事实错误门控（没有逐要素分）。"""
    from sfzy.judge.schema import JudgeResult

    res = JudgeResult(
        raw_scores={"fact_consistency": fact_score},
        scores={"fact_consistency": fact_score / 4.0},
        weighted_reward=fact_score / 4.0,
        fact_score=fact_score,
        has_factual_error=has_error,
        coverage_raw=dict(coverage_raw) if coverage_raw else {},
        ee_dims=dict(ee_dims) if ee_dims else {},
    )
    return res


def make_batch(**over) -> MonitorBatch:
    over = dict(over)
    n = over.pop("n", 4)
    defaults = dict(
        step=1,
        group_size=2,
        responses=[f"候选{i}" * (i + 1) for i in range(n)],
        references=["参考摘要"] * n,
        final_rewards=[0.5] * n,
        pre_gate_rewards=[0.5] * n,
        gate_flags=[False] * n,
        gate_reasons=[""] * n,
        term_values={"rouge_l": [0.5] * n},
        term_weights={"rouge_l": 1.0},
        raw_signals={},
        judge_results=[],
        advantages_raw=[0.0] * n,
        advantages_effective=[0.0] * n,
        kept_mask=[True] * (n // 2),
        output_lengths=[10 * (i + 1) for i in range(n)],
        token_lengths=[5 * (i + 1) for i in range(n)],
        ratios=[1.0] * n,
        normalization_factors=[math.sqrt(10 * (i + 1)) for i in range(n)],
        length_norm_mode="sqrt",
        clip_frac=0.0,
        policy_loss=0.1,
        total_loss=0.1,
        grad_norm_preclip=0.5,
        grad_norm_postclip=0.5,
        clip_applied=0.0,
        learning_rate=1e-5,
        clip_ratio=0.2,
        fact_gate_enabled=False,
        upper_length_ratio=None,
        max_new_tokens=100,
    )
    defaults.update(over)
    return MonitorBatch(**defaults)


# ---------------------------------------------------------------- 基础统计
def test_population_std_与组内归一化一致():
    assert _pstd([1, 2, 3, 4]) == pytest.approx(math.sqrt(1.25))
    assert _percentile([0, 10], 0.9) == pytest.approx(9.0)


def test_spearman_常量向量返回None而不是0():
    assert _spearman([1, 2, 3], [5, 5, 5]) is None
    assert _spearman([1, 1, 1], [1, 2, 3]) is None
    assert _spearman([1], [2]) is None
    assert _spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_组内优势与手写公式一致():
    values = [1.0, 3.0, 10.0, 10.0]
    adv = _group_advantages(values, 2)
    g1 = [1.0, 3.0]
    mu = 2.0
    sd = _pstd(g1)
    assert adv[0] == pytest.approx((1.0 - mu) / (sd + 1e-4))
    assert adv[1] == pytest.approx((3.0 - mu) / (sd + 1e-4))
    # 第二组全同 → 优势全 0，而不是 nan
    assert adv[2] == 0.0 and adv[3] == 0.0


# ---------------------------------------------------------------- 组内 reward
def test_组内奖励分布_零方差与全正():
    b = make_batch(
        group_size=2,
        final_rewards=[0.0, 0.0, 0.5, 0.5],     # 组1全0；组2全等正
        advantages_raw=[0.0, 0.0, 0.0, 0.0],
        kept_mask=[False, False],
    )
    out = GRPOMonitor(make_config()).observe(b)
    assert out["grpo/all_zero_group_rate"] == pytest.approx(0.5)
    assert out["grpo/all_equal_positive_group_rate"] == pytest.approx(0.5)
    assert out["grpo/zero_std_group_rate"] == pytest.approx(1.0)
    assert out["grpo/near_zero_std_group_rate"] == pytest.approx(1.0)
    assert out["grpo/groups_per_step"] == 2


def test_组内明显差异与微小差异():
    b = make_batch(
        group_size=2,
        final_rewards=[0.1, 0.9, 0.500, 0.501],
        kept_mask=[True, True],
    )
    out = GRPOMonitor(make_config(near_zero_std_eps=1e-6)).observe(b)
    stds = [0.4, 0.0005]
    assert out["grpo/group_reward_std_mean"] == pytest.approx(sum(stds) / 2, abs=1e-6)
    assert out["grpo/group_reward_range_mean"] == pytest.approx((0.8 + 0.001) / 2, abs=1e-6)
    assert out["grpo/near_zero_std_group_rate"] == pytest.approx(0.0)


def test_组过滤前后有效组比例():
    b = make_batch(
        group_size=2,
        final_rewards=[0.1, 0.9, 0.5, 0.5],   # 组1有方差，组2无
        advantages_raw=[0.5, -0.5, 0.0, 0.0],
        kept_mask=[True, False],               # 实际只保留组1
    )
    out = GRPOMonitor(make_config()).observe(b)
    assert out["grpo/kept_group_rate"] == pytest.approx(0.5)
    assert out["grpo/filtered_group_rate"] == pytest.approx(0.5)
    assert out["grpo/effective_group_rate"] == pytest.approx(0.5)
    assert out["grpo/effective_groups_per_step"] == 1


def test_不同组的奖励不互相参与组内标准化():
    a = _group_advantages([1.0, 3.0], 2)
    b = _group_advantages([1.0, 3.0, 100.0, 200.0], 2)[:2]
    assert a == pytest.approx(b), "第一组的结果不应受第二组尺度影响"


# ---------------------------------------------------------------- 优势
def test_优势raw与effective口径不同():
    b = make_batch(
        n=4, group_size=2,
        advantages_raw=[0.5, -0.5, 0.5, -0.5],
        advantages_effective=[0.25, -0.25, 0.0, 0.0],   # 第二组被 mask
    )
    out = GRPOMonitor(make_config()).observe(b)
    assert out["advantage/raw_abs_mean"] == pytest.approx(0.5)
    assert out["advantage/raw_positive_ratio"] == pytest.approx(0.5)
    assert out["advantage/effective_abs_mean"] == pytest.approx(0.125)
    assert out["advantage/effective_zero_ratio"] == pytest.approx(0.5)


# ---------------------------------------------------------------- Gate
def test_gate前后奖励与组内标准差():
    b = make_batch(
        n=4, group_size=2,
        pre_gate_rewards=[0.9, 0.1, 0.8, 0.6],
        final_rewards=[0.0, 0.0, 0.8, 0.6],     # 组1被门控清零
        gate_flags=[True, True, False, False],
        gate_reasons=["fact_has_error", "fact_has_error", "", ""],
    )
    out = GRPOMonitor(make_config()).observe(b)
    assert out["gate/all_gated_group_rate"] == pytest.approx(0.5)
    assert out["gate/group_std_before"] == pytest.approx((0.4 + 0.1) / 2, abs=1e-6)
    assert out["gate/group_std_after"] == pytest.approx(0.1 / 2, abs=1e-6)
    assert out["gate/fact_rate"] == pytest.approx(0.5)
    assert out["gate/other_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------- 六要素 / 覆盖率 / EE
def test_事实一致性整体分与错误率统计():
    r0 = make_result(fact_score=0, has_error=1)     # 判为有事实错误 → 0 分
    r1 = make_result(fact_score=4)                  # 没问题
    b = make_batch(
        n=2, group_size=2,
        raw_signals={"fact_consistency": [r0.weighted_reward, r1.weighted_reward]},
        judge_results=[r0, r1],
    )
    out = GRPOMonitor(make_config(), fact_enabled=True).observe(b)
    assert out["fact/score_mean"] == pytest.approx(0.5)          # (0+4)/2/4
    assert out["fact/error_rate"] == pytest.approx(0.5)
    assert out["fact/score_ratio_0"] == pytest.approx(0.5)
    assert out["fact/score_ratio_4"] == pytest.approx(0.5)


def test_覆盖率逐要素与漏覆盖():
    raw = {e: 4 for e in ELEMENTS}
    raw["court_facts"] = 0
    res = make_result(coverage_raw=raw)
    b = make_batch(
        n=2, group_size=2,
        raw_signals={"element_coverage": [0.5, 0.5]},
        judge_results=[res, res],
    )
    out = GRPOMonitor(make_config(), coverage_enabled=True).observe(b)
    assert out["coverage/mean"] == pytest.approx(0.5)
    assert out["coverage/court_facts/missing_rate"] == pytest.approx(1.0)
    assert out["coverage/court_facts/mean"] == pytest.approx(0.0)
    assert out["coverage/case_type/missing_rate"] == pytest.approx(0.0)


def test_EE三个维度方向与统计():
    dims = {name: 2 for name in EE_DIMENSIONS}
    res = make_result(ee_dims=dims)
    b = make_batch(
        n=2, group_size=2,
        raw_signals={"expression_efficiency": [0.5, 0.5]},
        judge_results=[res, res],
    )
    out = GRPOMonitor(
        make_config(ee_low_threshold=0.5, ee_high_threshold=0.75),
        ee_enabled=True,
    ).observe(b)
    assert out["ee/semantic_redundancy_mean"] == pytest.approx(0.5)
    assert out["ee/verbal_redundancy_mean"] == pytest.approx(0.5)
    assert out["ee/abstraction_adequacy_mean"] == pytest.approx(0.5)
    assert out["ee/low_score_rate"] == pytest.approx(1.0)
    assert out["ee/high_score_rate"] == pytest.approx(0.0)


def test_constant组不写相关系数():
    b = make_batch(
        n=4, group_size=2,
        raw_signals={"element_coverage": [0.5] * 4},
        output_lengths=[10, 20, 30, 40],
    )
    out = GRPOMonitor(make_config()).observe(b)
    # 覆盖率为常量 → 组内相关无法定义 → 不写该键（而不是写 0）
    assert "relation/group_coverage_length_spearman" not in out


# ---------------------------------------------------------------- 分项消融
def test_消融不改变真实奖励且处理并列():
    # 组内两条候选：覆盖率不同、EE 相同。
    b = make_batch(
        n=2, group_size=2,
        final_rewards=[0.4, 0.6],
        pre_gate_rewards=[0.4, 0.6],
        term_values={"element_coverage": [0.2, 0.8], "expression_efficiency": [0.5, 0.5]},
        term_weights={"element_coverage": 0.5, "expression_efficiency": 0.5},
        raw_signals={"element_coverage": [0.2, 0.8], "expression_efficiency": [0.5, 0.5]},
        kept_mask=[True],
    )
    before = list(b.final_rewards)
    out = GRPOMonitor(make_config()).observe(b)
    assert b.final_rewards == before, "消融不得改变真实奖励"
    # 移除 coverage 后两条并列 → 最优集合改变（原来的 top1 只有一条）
    assert out["relation/coverage_top1_flip_rate"] == pytest.approx(1.0)
    # 移除 EE：EE 无区分度，最优仍是覆盖率高的那条 → 不翻转
    assert out["relation/ee_top1_flip_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------- KL / 长度归一化
def test_KL未计算时不写0():
    b = make_batch(kl_estimate=None, kl_penalty=None)
    out = GRPOMonitor(make_config()).observe(b)
    assert "train/ref_kl_estimate" not in out
    assert "train/kl_penalty" not in out
    b2 = make_batch(kl_estimate=0.03, kl_penalty=0.0006)
    out2 = GRPOMonitor(make_config()).observe(b2)
    assert out2["train/ref_kl_estimate"] == pytest.approx(0.03)
    assert out2["train/kl_penalty"] == pytest.approx(0.0006)


def test_长度归一化统计与实际因子一致():
    b = make_batch(n=4, group_size=2)
    out = GRPOMonitor(make_config()).observe(b)
    factors = [math.sqrt(10 * (i + 1)) for i in range(4)]
    assert out["length/normalization_factor_mean"] == pytest.approx(sum(factors) / 4)
    assert out["length/normalization_factor_p90"] == pytest.approx(
        _percentile(factors, 0.9)
    )


# ---------------------------------------------------------------- trainer 集成
def test_监控开关不改变训练结果(tmp_path):
    """开启监控前后：Reward、优势、loss、模型参数必须完全一致（只读派生）。"""
    import torch

    import test_smoke_grpo as h

    def run(enabled: bool):
        torch.manual_seed(0)
        trainer = h.GRPOTrainer(
            model=h.tiny_model(), tokenizer=h.FakeTokenizer(),
            cfg=h.make_cfg(monitor={"enabled": enabled}),
            output_dir=str(tmp_path), scorer=h.FactStubScorer(),
        )
        torch.manual_seed(123)
        metrics = trainer.step(h.make_prompts(2))
        params = [p.detach().clone() for p in trainer.model.parameters()]
        return metrics, params

    m_on, p_on = run(True)
    m_off, p_off = run(False)
    for key in ("loss", "reward_mean", "advantage_abs_mean", "kept_ratio"):
        assert m_on[key] == pytest.approx(m_off[key], abs=1e-6)
    for a, b in zip(p_on, p_off):
        assert torch.allclose(a, b, atol=1e-6), "监控不得改变模型更新结果"


def test_首次更新前ratio接近1(tmp_path):
    """dropout 关闭时，old/current logprob 在同一参数状态计算 → ratio≈1、clip≈0。"""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    import test_smoke_grpo as h

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=h.VOCAB, n_positions=256, n_embd=32, n_layer=2, n_head=2,
        n_inner=64, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.config.pad_token_id = h.FakeTokenizer.pad_token_id
    trainer = h.GRPOTrainer(
        model=model, tokenizer=h.FakeTokenizer(), cfg=h.make_cfg(),
        output_dir=str(tmp_path), scorer=h.FactStubScorer(),
    )
    metrics = trainer.step(h.make_prompts(2))
    assert metrics["train/ratio_mean"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["train/clip_frac"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["train/clip_frac_low"] == pytest.approx(0.0)
    assert metrics["train/clip_frac_high"] == pytest.approx(0.0)


def test_swanlab_step与优化步对齐且只上报监控指标(tmp_path):
    import test_smoke_grpo as h

    class FakeTracker:
        def __init__(self):
            self.logs = []

        def log(self, metrics, step):
            self.logs.append((step, dict(metrics)))

        def finish(self):
            pass

        @property
        def run_id(self):
            return None

    tracker = FakeTracker()
    trainer = h.GRPOTrainer(
        model=h.tiny_model(), tokenizer=h.FakeTokenizer(), cfg=h.make_cfg(),
        output_dir=str(tmp_path), scorer=h.FactStubScorer(), tracker=tracker,
    )
    trainer.train(h.make_prompts(4))
    steps = [s for s, _ in tracker.logs]
    assert steps == [1, 2], "必须按实际 optimizer global_step 记录"
    for _, logged in tracker.logs:
        assert "grpo/group_reward_std_mean" in logged
        assert "reward/final_mean" in logged
        # history 里的扁平键不再重复上报
        assert "reward_mean" not in logged


def test_可选验证集指标只在配置开启时产生(tmp_path):
    import json

    import test_smoke_grpo as h

    val = tmp_path / "val.jsonl"
    records = [
        {"id": f"v{i}", "source": "判决如下：被告支付48000元。",
         "summary": "判令被告支付48000元。"}
        for i in range(3)
    ]
    val.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records),
                   encoding="utf-8")
    cfg = h.make_cfg(monitor={
        "enabled": True,
        "eval": {"enabled": True, "data_file": str(val), "every_n_steps": 1,
                 "max_samples": 3, "baseline": {"rouge_l": 0.4}},
    })
    trainer = h.GRPOTrainer(
        model=h.tiny_model(), tokenizer=h.FakeTokenizer(), cfg=cfg,
        output_dir=str(tmp_path), scorer=h.FactStubScorer(),
    )
    out = trainer._run_eval(1)
    assert "eval/rouge_l" in out and "eval/fact_mean" in out
    assert "eval/delta_rouge_l" in out          # 配置给了基线
    # 没有 coverage 信号 → 不编造 coverage 指标
    assert "eval/coverage_mean" not in out


def test_短run不会因为自适应缩短而每步都算熵():
    """熵要物化完整 (B,L,vocab) 的 fp32 概率，是最吃显存的一项。

    `_interval` 的规则是"短 run 至少记到几次扩展统计"（把间隔缩到 1）——
    对熵来说这会把 2~4 步的冒烟 run 变成"每一步都算熵"，直接把训练推到 OOM
    （实测踩过：27.7GB 训练 + 15.8GB vLLM 子进程，只剩 1.08GB，熵要 1.65GB）。
    所以 entropy_due 刻意不走自适应缩短。
    """
    from sfzy.rl.monitor import GRPOMonitor, MonitorConfig

    cfg = MonitorConfig.from_config({"entropy_log_interval": 20})
    m = GRPOMonitor(cfg)

    m.set_total_steps(2)                      # 冒烟 run：不该算熵
    assert [s for s in (1, 2) if m.entropy_due(s)] == []

    m.set_total_steps(40)                     # 正常 run：仍按 20 触发，行为不变
    assert m.entropy_due(20) and not m.entropy_due(19)
    # 对照组：便宜的指标仍然享受自适应缩短（total=40 时 20 → 8，所以第 8 步就触发）
    assert m.lora_due(8) and not m.lora_due(1)
