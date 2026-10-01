"""GRPOTrainer 的冒烟测试：CPU 上跑通 rollout → 打分 → advantage → 更新。

这是**唯一**把 RL 全链路串起来的测试。在此之前 rollout、裁判接线、基线锚、
组过滤都只有单元测试，各自都对，但接起来是不是对的没人验证过 —— 而 RL 的
bug 恰恰最爱藏在这种接缝上（比如"裁判分算出来了但没传进奖励"）。

奖励只有 fact_judge 一种，所以每个 trainer 都必须挂裁判桩。规模压到最小：
微型 GPT2、假 tokenizer、2 个 prompt × G=2、生成 8 个 token，门控放开
（这里测的是**管线**，不是奖励设计本身，奖励语义由 tests/test_reward.py 覆盖）。

    python -m pytest tests/test_smoke_grpo.py -q
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from sfzy.config import load_config
from sfzy.eval.metrics import SemanticScorer
from sfzy.rl.trainer import GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 1024


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

    为什么必须带内容：微型随机模型在 eval 模式下（rollout 的正确模式）为同一个
    prompt 采样的 G 条**长度经常完全相同**，只按长度打分会让奖励在组内变成常数、
    advantage 全 0 —— 那是夹具退化，不是管线问题。

    用 `sum(map(ord, ...))` 而不是内置 `hash()`：后者受 PYTHONHASHSEED 影响，
    同一份输入跨进程会给出不同的分，测试就不再可复现。
    """
    return min(1.0, 0.1 + 0.02 * len(text) + (sum(map(ord, text)) % 7) / 100.0)


class FactStubScorer(SemanticScorer):
    """六要素裁判桩：分数和候选内容相关，量纲 [0,1]（fact_judge 只认这个量纲）。

    同时记录调用次数 —— 没有这条断言，"裁判被调用但结果被丢掉"这种接缝
    bug 会让测试依然通过。
    """

    name = "judge_fact_stub"

    def __init__(self) -> None:
        self.calls = 0
        self.seen_items: list = []

    def score_batch(self, items):
        self.calls += 1
        self.seen_items.extend(items)
        return [_stub_fact_score(i["candidate"]) for i in items]


class FlakyScorer(SemanticScorer):
    """一半返回 None，模拟提取失败/显存抖动。量纲仍是 [0,1]。"""

    name = "judge_fact_flaky"

    def score_batch(self, items):
        return [None if i % 2 else 0.7 for i, _ in enumerate(items)]


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
            # 基线锚要用：SFT 阶段对同一个 prompt 的输出
            rec["sft_output"] = "判令被告支付48000元，本案受理费由被告负担。"
        out.append(rec)
    return out


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
        "reward": {
            "mode": "fact_judge",
            "weights": {"rouge_l": 0.3, "fact": 0.7},
            "rouge_mode": "char",
            # 门控放开：生成只有 8 个 token，正常门控会把每条都拦下，
            # 那一组奖励全 0、advantage 全 0，测不出任何东西。
            "gate": {
                "min_chars": 0,
                "length_ratio_range": [0.0, 1000.0],
                "forbidden_prefixes": [],
                "require_result_marker": False,
            },
        },
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

    for key in ("loss", "reward_mean", "rouge_l", "fact_judge",
                "fact_judge_group_std", "kept_groups", "total_groups", "gating_rate",
                "output_len_mean", "clipped_frac", "anchor_filtered",
                # swanlab 面板要用的那几项；少一个就在面板上少一根曲线
                "grad_norm", "kept_ratio", "reward_group_std", "judge_missing"):
        assert key in metrics, f"训练日志缺少指标 {key}"
    assert metrics["loss"] == metrics["loss"]          # 不是 nan


def test_一步训练确实更新了参数(tmp_path):
    """梯度真的传到了参数上。

    裁判桩的分数随内容变化，组内才有真实方差；纯规则奖励在微型随机模型上
    常常是常数，advantage 全 0 时 loss 恒等于 0，测试等于没测。
    """
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


def test_fact_judge模式的奖励有方差(tmp_path):
    """组内方差是 GRPO 的前提。奖励恒定 → advantage 全 0 → 白跑一个 epoch。"""
    trainer = make_trainer(tmp_path)
    metrics = trainer.step(make_prompts())
    assert metrics["reward_std"] > 0
    assert metrics["kept_groups"] > 0, "有方差的组不该被过滤掉"
    # 事实项自己的组内方差：均值好看但组内没方差，等于白接一个裁判
    assert metrics["fact_judge_group_std"] > 0


def test_train_能跑到最后并保存(tmp_path):
    trainer = make_trainer(tmp_path)
    state = trainer.train(make_prompts(4))
    assert state.step == 2                       # 4 个 prompt / 每步 2 个
    assert len(state.history) == 2
    assert (tmp_path / "step_000002.pt").exists()


# ---------------------------------------------------------------- train/eval 模式

def test_训练前向必须处在_train_模式(tmp_path, monkeypatch):
    """回归测试：GRPO 此前从不调 `model.train()`。

    HF 的 `from_pretrained` 结尾会把模型设成 eval，而 ChatGLM3 的梯度检查点
    条件是 `if self.gradient_checkpointing and self.training:` —— training=False
    会让整个条件短路，检查点静默失效、激活按「每层完整保存」算，长序列必然
    OOM，且报错栈里全是 bitsandbytes 的调用，看不出根因。
    """
    import sfzy.rl.trainer as trainer_mod

    seen: dict = {}
    real = trainer_mod.sequence_logprob

    def spy(model, *args, **kwargs):
        seen.setdefault("training", model.training)
        return real(model, *args, **kwargs)

    monkeypatch.setattr(trainer_mod, "sequence_logprob", spy)

    trainer = make_trainer(tmp_path)
    trainer.model.eval()                     # from_pretrained 之后就是这个状态
    trainer.step(make_prompts())

    assert seen["training"] is True, (
        "带梯度的策略前向必须在 train 模式，否则 ChatGLM3 的梯度检查点不生效"
    )


def test_rollout生成时必须切到_eval_模式(tmp_path, monkeypatch):
    """生成时必须是 eval。

    ChatGLM3 在 train + 梯度检查点下会把 KV cache 关掉，自回归生成退化成
    每个 token 重算整个前缀 —— rollout 是 GRPO 的瓶颈，这个代价受不了。
    """
    import sfzy.rl.trainer as trainer_mod

    seen: dict = {}
    real = trainer_mod.generate_batch

    def spy(model, *args, **kwargs):
        seen["training"] = model.training
        return real(model, *args, **kwargs)

    monkeypatch.setattr(trainer_mod, "generate_batch", spy)

    trainer = make_trainer(tmp_path)
    trainer.model.train()                    # 故意先置成 train
    trainer.step(make_prompts())

    assert seen["training"] is False, "生成必须显式 eval，不能靠模型当前模式"


# ---------------------------------------------------------------- 日志列

def test_日志打印事实一致性和裁判缺失(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="sfzy.grpo"):
        trainer = make_trainer(tmp_path)
        trainer.train(make_prompts(2))

    text = " ".join(r.message for r in caplog.records)
    assert "事实一致性" in text
    assert "裁判缺失" in text
    assert "组内σ" in text


# ---------------------------------------------------------------- 裁判接线

def test_裁判分进入奖励(tmp_path):
    scorer = FactStubScorer()
    trainer = make_trainer(tmp_path, scorer=scorer)
    metrics = trainer.step(make_prompts())

    assert scorer.calls == 1, "裁判应该被**批量**调用一次，而不是逐条调用"
    assert len(scorer.seen_items) == trainer.group_size * 2
    assert metrics["fact_judge"] > 0, "裁判分算出来了却没进明细"


def test_裁判分真的改变了奖励():
    """同一个生成结果、同一个配置，只换裁判分，奖励必须不同。

    这条防的是"裁判接了但权重是 0"（改了名义没改实际）。
    """
    from sfzy.rl.reward import compute_rewards

    cfg = make_cfg()["rl"]["reward"]
    args = (["同一段文本"], ["判令被告支付48000元"])
    low = compute_rewards(*args, cfg=cfg, semantic_scores=[0.1])[0].total
    high = compute_rewards(*args, cfg=cfg, semantic_scores=[0.9])[0].total
    assert high > low


def test_没裁判直接报错(tmp_path):
    """fact_judge 的唯一事实来源就是裁判，没有裁判必须当场报错。"""
    with pytest.raises(ValueError, match="fact_judge"):
        make_trainer(tmp_path, scorer=None)


def test_裁判个别失败不中断训练(tmp_path):
    """一次超时不该让跑了三小时的 run 在第 137 步崩掉。
    缺失用组内均值补，比例记进日志。"""
    trainer = make_trainer(tmp_path, scorer=FlakyScorer())
    metrics = trainer.step(make_prompts())
    assert metrics["judge_missing"] == pytest.approx(0.5)
    assert metrics["fact_judge"] > 0, "组内另一半的分数应该被用上"
    assert metrics["loss"] == metrics["loss"]


def test_裁判补缺_用组内均值而不是0(tmp_path):
    """填 0 等于"这条很差"，会把优势估计带偏。"""
    trainer = make_trainer(tmp_path)
    filled, frac = trainer.fill_semantic_gaps([0.8, None, None, 0.6], group_size=2)
    assert filled == [0.8, 0.8, 0.6, 0.6]
    assert frac == pytest.approx(0.5)


def test_裁判补缺_整组缺失时退化为0(tmp_path):
    trainer = make_trainer(tmp_path)
    filled, frac = trainer.fill_semantic_gaps([None, None], group_size=2)
    assert filled == [0.0, 0.0] and frac == 1.0


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
    """锚开着也要能正常训练：prompt 池带 sft_output 时不该报错。"""
    trainer = make_trainer(tmp_path, anchor={"enabled": True, "slack": 0.05})
    state = trainer.train(make_prompts(2, with_sft=True))
    assert state.step == 1
    assert trainer.state.history[0]["total_groups"] == 2
