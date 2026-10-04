"""GRPOTrainer 的冒烟测试：CPU 上跑通 rollout → 多信号打分 → advantage → 更新。

这是**唯一**把 RL 全链路串起来的测试。奖励项本身怎么算由
tests/test_reward.py / test_reward_spec.py 覆盖；这里测的是**接缝**：
多信号怎么从裁判进到聚合、逐项指标有没有漏、缺信号会不会补、只开规则项时
会不会多余地要求裁判。

规模压到最小：微型 GPT2、假 tokenizer、2 个 prompt × G=2、生成 8 个 token，
门控整块关掉（这里测管线，不测门控）。

    python -m pytest tests/test_smoke_grpo.py -q
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from sfzy.config import load_config
from sfzy.rl.trainer import GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 1024
FACT_SIGNAL = "fact_consistency"
ELEMENT_WEIGHTS = {
    "case_type": 0.05,
    "plaintiff_claims": 0.15,
    "defendant_defenses": 0.10,
    "court_facts": 0.25,
    "legal_basis": 0.15,
    "judgment_result": 0.30,
}


class FakeTokenizer:
    """与其它测试同一套约定：一个字符一个 id，id < 10 当特殊 token。"""

    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"
    GENERATION_MARKER = 999

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[10 + (ord(c) % 200) for c in text])

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids = []
        for message in messages:
            ids.extend(self(message["content"]).input_ids)
        if add_generation_prompt:
            ids.append(self.GENERATION_MARKER)
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(i - 10) for i in token_ids if i >= 10)


def _stub_fact_score(text: str) -> float:
    """长度项 + **内容相关的稳定扰动**，取值 ∈ [0,1]。

    为什么必须带内容：微型随机模型在 eval 模式下为同一个 prompt 采样的 G 条
    长度经常完全相同，只按长度打分会让奖励在组内变成常数、advantage 全 0 ——
    那是夹具退化，不是管线问题。用 `sum(map(ord, ...))` 而不是内置 `hash()`，
    保证跨进程可复现。
    """
    return min(1.0, 0.1 + 0.02 * len(text) + (sum(map(ord, text)) % 7) / 100.0)


class FactStubScorer:
    """给 fact_consistency 信号用的裁判桩。量纲 [0,1]。

    裁判契约是结构化描述（`sfzy.eval.metrics.SignalScorer` 是个 Protocol），
    所以桩**不需要继承任何东西** —— 只要有这两个方法和一个日志名。
    """

    name = "judge_fact_stub"

    def __init__(self) -> None:
        self.calls = 0
        self.seen_items: list = []

    def available_signals(self):
        return {FACT_SIGNAL}

    def score_batch_signals(self, items):
        self.calls += 1
        self.seen_items.extend(items)
        return [{FACT_SIGNAL: _stub_fact_score(i["candidate"])} for i in items]


class FlakyScorer:
    """一半返回 None，模拟提取失败/显存抖动。量纲仍是 [0,1]。"""

    name = "judge_fact_flaky"

    def available_signals(self):
        return {FACT_SIGNAL}

    def score_batch_signals(self, items):
        return [{FACT_SIGNAL: None if i % 2 else 0.7} for i, _ in enumerate(items)]


class TwoSignalScorer:
    """同时产出事实一致性和关键要素覆盖率两路信号。"""

    name = "judge_two"

    def available_signals(self):
        return {"fact_consistency", "element_coverage"}

    def score_batch_signals(self, items):
        return [
            {
                "fact_consistency": _stub_fact_score(i["candidate"]),
                "element_coverage": min(1.0, 0.2 + 0.02 * len(i["candidate"])),
            }
            for i in items
        ]


def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB, n_positions=256, n_embd=32, n_layer=2, n_head=2, n_inner=64
    )
    model = GPT2LMHeadModel(config)
    model.config.pad_token_id = FakeTokenizer.pad_token_id
    return model


def make_prompts(n: int = 2, with_sft: bool = True) -> list[dict]:
    out = []
    for i in range(n):
        rec = {
            "id": f"p{i}",
            "source": "判决如下：被告支付48000元。" * (i + 1),
            "summary": "判令被告支付48000元。",
        }
        if with_sft:
            rec["sft_output"] = "判令被告支付48000元，本案受理费由被告负担。"
        out.append(rec)
    return out


def reward_cfg(**over):
    cfg = {
        "normalize_weights": True,
        "terms": {
            "rouge_l": {"enabled": True, "weight": 0.3},
            "fact_consistency": {
                "enabled": True, "weight": 0.7,
                "element_weights": dict(ELEMENT_WEIGHTS),
            },
        },
        "rouge_mode": "char",
        # 门控整块关掉：生成只有 8 个 token，正常门控会全拦下，测不出东西。
        "gate": {"enabled": False},
    }
    cfg.update(over)
    return cfg


def make_cfg(**rl_over):
    cfg = load_config(ROOT / "configs" / "grpo_fact_judge.yaml")
    cfg["rl"].update({
        "group_size": 2,
        "prompts_per_step": 2,
        "max_new_tokens": 8,
        "max_length": 64,
        "kl_coef": 0.0,
        "length_norm": "sqrt",
        "save_every_n_steps": 1000,
        "log_every_n_steps": 1,
        "learning_rate": 1e-3,
        "temperature": 0.9,
        "reward": reward_cfg(),
    })
    cfg["rl"].update(rl_over)
    return cfg


_UNSET = object()


def make_trainer(tmp_path, scorer=_UNSET, **rl_over):
    if scorer is _UNSET:
        scorer = FactStubScorer()
    return GRPOTrainer(
        model=tiny_model(),
        tokenizer=FakeTokenizer(),
        cfg=make_cfg(**rl_over),
        output_dir=str(tmp_path),
        scorer=scorer,
    )


# ---------------------------------------------------------------- 全链路

def test_一步训练能跑完并返回全部指标(tmp_path):
    trainer = make_trainer(tmp_path)
    metrics = trainer.step(make_prompts())

    for key in ("loss", "reward_mean", "reward_group_std", "gating_rate",
                "term_rouge_l", "term_rouge_l_group_std",
                "term_fact_consistency", "term_fact_consistency_group_std",
                "kept_groups", "total_groups", "kept_ratio", "anchor_filtered",
                "output_len_mean", "clipped_frac", "grad_norm", "judge_missing"):
        assert key in metrics, f"训练日志缺少指标 {key}"
    assert metrics["loss"] == metrics["loss"]          # 不是 nan


def test_一步训练确实更新了参数(tmp_path):
    """梯度真的传到了参数上。裁判桩的分数随内容变化，组内才有真实方差。"""
    trainer = make_trainer(tmp_path, scorer=FactStubScorer())
    before = [p.detach().clone() for p in trainer.model.parameters()]
    metrics = trainer.step(make_prompts())
    assert metrics["advantage_abs_mean"] > 0, "advantage 全是 0，梯度无从谈起"
    assert metrics["reward_std"] > 0, "组内奖励没有方差，GRPO 学不到方向"
    changed = sum(
        1 for a, b in zip(before, trainer.model.parameters())
        if not torch.equal(a, b.detach())
    )
    assert changed > 0, "一步训练后参数一个都没变"


def test_每个reward项都有自己的组内方差(tmp_path):
    """组内方差是 GRPO 的前提。逐项都要有方差，否则那一项等于白接。"""
    trainer = make_trainer(tmp_path)
    metrics = trainer.step(make_prompts())
    assert metrics["reward_std"] > 0
    assert metrics["kept_groups"] > 0
    assert metrics["term_fact_consistency_group_std"] > 0


def test_train_能跑到最后并保存(tmp_path):
    trainer = make_trainer(tmp_path)
    state = trainer.train(make_prompts(4))
    assert state.step == 2
    assert len(state.history) == 2
    assert (tmp_path / "step_000002.pt").exists()


# ---------------------------------------------------------------- train/eval 模式

def test_训练前向必须处在_train_模式(tmp_path, monkeypatch):
    import sfzy.rl.trainer as trainer_mod

    seen: dict = {}
    real = trainer_mod.sequence_logprob

    def spy(model, *args, **kwargs):
        seen.setdefault("training", model.training)
        return real(model, *args, **kwargs)

    monkeypatch.setattr(trainer_mod, "sequence_logprob", spy)

    trainer = make_trainer(tmp_path)
    trainer.model.eval()
    trainer.step(make_prompts())

    assert seen["training"] is True, (
        "带梯度的策略前向必须在 train 模式，否则 ChatGLM3 的梯度检查点不生效"
    )


def test_rollout生成时必须切到_eval_模式(tmp_path, monkeypatch):
    import sfzy.rl.trainer as trainer_mod

    seen: dict = {}
    real = trainer_mod.generate_batch

    def spy(model, *args, **kwargs):
        seen["training"] = model.training
        return real(model, *args, **kwargs)

    monkeypatch.setattr(trainer_mod, "generate_batch", spy)

    trainer = make_trainer(tmp_path)
    trainer.model.train()
    trainer.step(make_prompts())

    assert seen["training"] is False, "生成必须显式 eval，不能靠模型当前模式"


# ---------------------------------------------------------------- 日志列

def test_日志按启用的reward打印列(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="sfzy.grpo"):
        trainer = make_trainer(tmp_path)
        trainer.train(make_prompts(2))

    text = " ".join(r.message for r in caplog.records)
    assert "rouge_l" in text and "fact_consistency" in text
    assert "裁判缺失" in text
    assert "组内σ" in text


# ---------------------------------------------------------------- 多信号接线

def test_裁判信号批量进来(tmp_path):
    scorer = FactStubScorer()
    trainer = make_trainer(tmp_path, scorer=scorer)
    metrics = trainer.step(make_prompts())

    assert scorer.calls == 1, "裁判应该被**批量**调用一次，而不是逐条调用"
    assert len(scorer.seen_items) == trainer.group_size * 2
    assert metrics["term_fact_consistency"] > 0


def test_裁判信号真的改变了奖励():
    """同一个生成结果、同一个配置，只换裁判信号，奖励必须不同。"""
    from sfzy.rl.reward import compute_rewards

    cfg = reward_cfg()
    args = (["同一段文本"], ["判令被告支付48000元"])
    low = compute_rewards(*args, spec=cfg, judge_signals={FACT_SIGNAL: [0.1]})[0].total
    high = compute_rewards(*args, spec=cfg, judge_signals={FACT_SIGNAL: [0.9]})[0].total
    assert high > low


def test_有judge项却没裁判直接报错(tmp_path):
    with pytest.raises(ValueError, match="judge"):
        make_trainer(tmp_path, scorer=None)


def test_裁判产不出需要的信号直接报错(tmp_path):
    class WrongScorer:
        name = "judge_wrong"

        def available_signals(self):
            return {"别的信号"}

        def score_batch_signals(self, items):
            return [{"别的信号": 0.5} for _ in items]

    with pytest.raises(ValueError, match="产不出来"):
        make_trainer(tmp_path, scorer=WrongScorer())


def test_只开规则项时不加载裁判也不需要裁判(tmp_path):
    trainer = make_trainer(tmp_path, scorer=None, reward=reward_cfg(
        terms={"rouge_l": {"enabled": True, "weight": 1.0}},
    ))
    metrics = trainer.step(make_prompts())
    assert "term_rouge_l" in metrics
    assert "term_fact_consistency" not in metrics


def test_两个reward同时训练(tmp_path):
    """事实一致性 + 关键要素覆盖率：两路信号都要进指标、都要有组内方差。"""
    reward = reward_cfg(terms={
        "rouge_l": {"enabled": False, "weight": 0.0},
        "fact_consistency": {"enabled": True, "weight": 0.6,
                             "element_weights": dict(ELEMENT_WEIGHTS)},
        "element_coverage": {"enabled": True, "weight": 0.4,
                             "element_weights": dict(ELEMENT_WEIGHTS)},
    })
    trainer = make_trainer(tmp_path, scorer=TwoSignalScorer(), reward=reward)
    metrics = trainer.step(make_prompts())
    assert "term_fact_consistency" in metrics
    assert "term_element_coverage" in metrics
    assert metrics["judge_missing"] == 0.0


# ---------------------------------------------------------------- 判分缺失

def test_裁判个别失败不中断训练(tmp_path):
    trainer = make_trainer(tmp_path, scorer=FlakyScorer())
    metrics = trainer.step(make_prompts())
    assert metrics["judge_missing"] == pytest.approx(0.5)
    assert metrics["term_fact_consistency"] > 0, "组内另一半的分数应该被用上"
    assert metrics["loss"] == metrics["loss"]


def test_补缺_用组内均值而不是0(tmp_path):
    trainer = make_trainer(tmp_path)
    filled, frac = trainer.fill_semantic_gaps(
        {FACT_SIGNAL: [0.8, None, None, 0.6]}, group_size=2
    )
    assert filled[FACT_SIGNAL] == [0.8, 0.8, 0.6, 0.6]
    assert frac == pytest.approx(0.5)


def test_补缺_整组缺失时退化为0(tmp_path):
    trainer = make_trainer(tmp_path)
    filled, frac = trainer.fill_semantic_gaps({FACT_SIGNAL: [None, None]}, group_size=2)
    assert filled[FACT_SIGNAL] == [0.0, 0.0] and frac == 1.0


# ---------------------------------------------------------------- 基线锚

def test_基线锚_从prompt池算出SFT奖励(tmp_path):
    trainer = make_trainer(tmp_path, anchor={"enabled": True, "slack": 0.05})
    trainer.prepare_baseline(make_prompts(3, with_sft=True))
    assert len(trainer.baseline_rewards) == 3
    assert all(isinstance(v, float) for v in trainer.baseline_rewards.values())


def test_基线锚_没有sft_output时跳过而不是崩(tmp_path):
    trainer = make_trainer(tmp_path, anchor={"enabled": True})
    trainer.prepare_baseline(make_prompts(3, with_sft=False))
    assert trainer.baseline_rewards == {}


def test_基线锚_默认关闭(tmp_path):
    trainer = make_trainer(tmp_path)
    assert trainer.anchor_enabled is False


def test_基线锚_训练时不误伤(tmp_path):
    trainer = make_trainer(tmp_path, anchor={"enabled": True, "slack": 0.05})
    state = trainer.train(make_prompts(2, with_sft=True))
    assert state.step == 1
    assert trainer.state.history[0]["total_groups"] == 2
