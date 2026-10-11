"""KL 参考策略：共用底座 + 临时换 LoRA 旁路。

背景（为什么需要这个文件）：`kl_coef > 0` 需要一个"参考策略"的对数概率。
本项目的策略与 SFT 策略共用同一个冻结底座、只有 LoRA 不同，所以正确做法是
在载入 SFT 权重之后 `snapshot_lora()` 存一份 LoRA 副本（≈118 MB），算 KL 时
临时换进去前向一次再换回来 —— 不用第二份 6B 模型。

这里锁住四件事：
  1. `use_lora_weights` 换进换出后**参数对象不变**、数值原样恢复
     （AdamW 的动量按参数对象索引，替换对象会静默错位）；
  2. `kl_coef > 0` 时 k3 惩罚真的进了 loss（`metrics["kl"] > 0`），
     且参考权重不会残留在策略里；
  3. 拿不到参考策略时**直接报错**，不静默把 KL 跳过；
  4. 单次更新下重要性比率恒为 1 ⇒ 裁剪不触发（`clipped_frac` 恒为 0）——
     这是"一次 rollout 只更新一次"的必然结果，值得钉住。

    python -m pytest tests/test_kl_reference.py -q
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from sfzy.config import load_config
from sfzy.models.lora import (
    inject_lora,
    lora_snapshot_scale,
    mark_only_lora_trainable,
    snapshot_lora,
    use_lora_weights,
)
from sfzy.rl.trainer import GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 1024


class FakeTokenizer:
    """与 tests/test_smoke_grpo.py 同一套约定：一个字符一个 id。"""

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


def _stub_score(text: str) -> float:
    """内容相关的稳定扰动：组内必须有真实方差，否则 advantage 全 0。"""
    return min(1.0, 0.1 + 0.02 * len(text) + (sum(map(ord, text)) % 7) / 100.0)


class FactStubScorer:
    name = "judge_fact_stub"

    def available_signals(self):
        return {"fact_consistency"}

    def score_batch_signals(self, items):
        return [{"fact_consistency": _stub_score(i["candidate"])} for i in items]


def tiny_model():
    """微型 GPT2。**dropout 全部置 0** —— 见文件末尾那条 ratio 断言。"""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB, n_positions=256, n_embd=32, n_layer=2, n_head=2, n_inner=64,
        resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.config.pad_token_id = FakeTokenizer.pad_token_id
    return model


def lora_model():
    """GPT2 + 手写 LoRA。GPT2 的 c_attn/c_proj 是 Conv1D，能注入的叶子
    nn.Linear 只有 lm_head，够用：只要 LoRA 真的排在计算图里就行。"""
    model = tiny_model()
    replaced = inject_lora(model, target_modules=["lm_head"], r=4, alpha=8, dropout=0.0)
    assert replaced == 1
    mark_only_lora_trainable(model)
    return model


def lora_names(model):
    return [n for n, _ in model.named_parameters() if "lora_" in n]


def perturb(model, delta=0.01, only_b=False, seed=1234):
    """把当前 LoRA 权重挪一下，模拟"策略已经走过几步"。

    **不能用 `param.add_(常数)`。** `lora_B` 注入时是全零，整体加一个常数会得到
    "每一行完全相同"的 B，于是 `B(A·x)` 在词表维上是一个**常数偏移** ——
    log_softmax 对"每个位置整体平移"不敏感，序列 logprob 会严格不变，KL 恒为 0。
    那样测试会"通过"却是假通过（这个坑现场踩过一次）。
    用带种子的随机扰动，保证每个词表项的增量不同。
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name and (not only_b or "lora_B" in name):
                param.add_(torch.randn(param.shape, generator=gen) * delta)


def make_prompts(n=2):
    return [
        {
            "id": f"p{i}",
            "source": "判决如下：被告支付48000元。" * (i + 1),
            "summary": "判令被告支付48000元。",
            "sft_output": "判令被告支付48000元，本案受理费由被告负担。",
        }
        for i in range(n)
    ]


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
            "normalize_weights": True,
            "terms": {
                "rouge_l": {"enabled": True, "weight": 0.3},
                "fact_consistency": {"enabled": True, "weight": 0.7},
            },
            "rouge_mode": "char",
            # 门控整块关掉：生成只有 8 个 token，正常门控会全拦下，测不出东西。
            "gate": {"enabled": False},
        },
    })
    cfg["rl"].update(rl_over)
    return cfg


def make_trainer(tmp_path, model=None, **rl_over):
    return GRPOTrainer(
        model=model if model is not None else tiny_model(),
        tokenizer=FakeTokenizer(),
        cfg=make_cfg(**rl_over),
        output_dir=str(tmp_path),
        scorer=FactStubScorer(),
    )


# ---------------------------------------------------------------- 快照与换权重

def test_快照体检_B全零说明参考等于原始底座():
    model = lora_model()                     # 刚注入时 lora_B 是零初始化
    b_abs, total_abs = lora_snapshot_scale(snapshot_lora(model))
    assert b_abs == 0.0
    assert total_abs > 0.0                   # lora_A 是 kaiming 随机，非零


def test_换权重后参数对象不变且数值原样恢复():
    model = lora_model()
    ref = snapshot_lora(model)
    names = lora_names(model)
    assert names, "模型里应该有 LoRA 参数"
    assert all(not t.requires_grad for t in ref.values())

    perturb(model, delta=0.01)
    moved = {n: p.detach().clone() for n, p in model.named_parameters() if n in ref}
    for n in names:
        assert not torch.allclose(ref[n], moved[n]), "快照被就地改动了（存的是引用？）"

    before_objects = {n: p for n, p in model.named_parameters() if n in ref}
    with use_lora_weights(model, ref):
        for n, p in model.named_parameters():
            if n in ref:
                assert torch.equal(p.detach(), ref[n]), "换进去的不是参考权重"
        inner_objects = {n: p for n, p in model.named_parameters() if n in ref}
    for n in names:
        assert inner_objects[n] is before_objects[n], "参数对象被替换了，优化器状态会错位"
        assert torch.equal(
            dict(model.named_parameters())[n].detach(), moved[n]
        ), "退出上下文后没有恢复原值"


def test_快照键对不上时直接报错():
    model = lora_model()
    ref = snapshot_lora(model)
    bad = {k.replace("lora_A", "lora_X"): v for k, v in ref.items()}
    with pytest.raises(ValueError, match="找不到"):
        with use_lora_weights(model, bad):
            pass


# ---------------------------------------------------------------- 训练循环里的 KL

def test_kl_生效并走_sft_快照(tmp_path):
    model = lora_model()
    ref = snapshot_lora(model)               # “SFT 那一刻”
    perturb(model, delta=0.05, only_b=True)  # 策略已经偏离 SFT

    trainer = GRPOTrainer(
        model=model, tokenizer=FakeTokenizer(),
        cfg=make_cfg(kl_coef=0.05), output_dir=str(tmp_path),
        scorer=FactStubScorer(), reference_lora=ref,
    )
    assert trainer.reference_lora is not None
    metrics = trainer.step(make_prompts())

    assert metrics["kl"] > 0.0, "kl_coef>0 但 kl 恒为 0：参考策略没生效"
    assert metrics["reward_group_std"] > 0, "夹具退化，奖励组内没方差"

    # 参考权重只是临时换进去的，不该留在策略里
    leaked = [
        n for n, p in model.named_parameters()
        if n in ref and torch.equal(p.detach(), ref[n])
    ]
    assert not leaked, f"参考权重泄漏到策略里：{leaked[:2]}"


def test_kl_关闭时不需要参考策略(tmp_path):
    trainer = make_trainer(tmp_path, kl_coef=0.0)
    metrics = trainer.step(make_prompts())
    assert metrics["kl"] == 0.0


def test_kl_拿不到参考策略时直接报错(tmp_path):
    model = lora_model()
    assert not hasattr(model, "disable_adapter"), "前提：手写 LoRA 没有 disable_adapter"
    with pytest.raises(ValueError, match="拿不到参考策略"):
        make_trainer(tmp_path, model=model, kl_coef=0.05)


def test_先快照再resume_参考仍然是_sft(tmp_path):
    """train_grpo.py 的真实顺序：

        载入 SFT adapter → snapshot_lora()（参考） → 构造 trainer → resume(RL checkpoint)

    `resume()` 会用 RL 的中间权重覆盖 LoRA，但**参考必须还是 SFT 那一份**。
    顺序写反了不会报错，只会让 KL 变成"往自己拉"（恒为 0），所以钉住它。
    """
    from sfzy.sft.checkpoint import save_checkpoint

    model = lora_model()
    perturb(model, delta=0.02, only_b=True, seed=11)   # 模拟"载入了 SFT adapter"
    ref = snapshot_lora(model)

    trained = lora_model()                   # 模拟"RL 跑过一段的 LoRA"
    perturb(trained, delta=0.3, only_b=True, seed=22)
    ckpt = tmp_path / "rl_step_000010.pt"
    save_checkpoint(ckpt, model=trained, only_trainable=True)

    trainer = GRPOTrainer(
        model=model, tokenizer=FakeTokenizer(),
        cfg=make_cfg(kl_coef=0.05), output_dir=str(tmp_path),
        scorer=FactStubScorer(), reference_lora=ref,
    )
    trainer.resume(str(ckpt))

    current = {n: p.detach().clone() for n, p in model.named_parameters() if n in ref}
    assert any(
        not torch.allclose(current[n], ref[n]) for n in ref
    ), "resume 没生效，这个测试就失去意义了"

    metrics = trainer.step(make_prompts())
    assert metrics["kl"] > 0, "参考被 resume 覆盖成了当前策略（快照取晚了？）"


# ---------------------------------------------------------------- 裁剪为什么咬不到

def test_单次更新下重要性比率恒为1_裁剪不触发(tmp_path):
    """rollout 之后只有一次 optimizer.step()：old_logprobs 与策略 logprobs 是
    同一参数状态、同一批数据、dropout 全 0 ⇒ ρ ≡ 1 ⇒ clip 的两个分支恒等，
    `clipped_frac` 严格等于 0（μ=1，即 `rl.inner_iterations` 的默认值）。
    μ>1 的对照见 `test_μ大于1时裁剪开始生效`。"""
    trainer = make_trainer(tmp_path, kl_coef=0.0)
    metrics = trainer.step(make_prompts())
    assert metrics["clipped_frac"] == 0.0
    assert metrics["advantage_abs_mean"] > 0  # 有梯度，不是为了“全 0”才恒等


# ---------------------------------------------------------------- μ（内层更新次数）

def test_μ_从配置读取_默认是1(tmp_path):
    assert make_trainer(tmp_path).inner_iterations == 1
    assert make_trainer(tmp_path, inner_iterations=4).inner_iterations == 4
    # YAML 里写浮点也能接受（2.0 → 2），但整数以外的东西必须当场报错
    assert make_trainer(tmp_path, inner_iterations=2.0).inner_iterations == 2
    with pytest.raises(ValueError, match="必须 ≥ 1"):
        make_trainer(tmp_path, inner_iterations=0)
    with pytest.raises(ValueError, match="必须是整数"):
        make_trainer(tmp_path, inner_iterations="四次")


def test_μ_就是每个_rollout_做多少次_optimizer_step(tmp_path):
    trainer = make_trainer(tmp_path, kl_coef=0.0, inner_iterations=4)
    calls = {"n": 0}
    real_step = trainer.optimizer.step

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real_step(*args, **kwargs)

    trainer.optimizer.step = counted
    metrics = trainer.step(make_prompts())
    assert calls["n"] == 4, "一次 rollout 应该做 μ 次梯度更新"
    assert metrics["inner_iterations"] == 4


def test_μ大于1时裁剪开始生效(tmp_path):
    """μ>1 的**唯一**作用就是让第 2 次起 ρ≠1，从而让 clip 真正参与。

    走的是 LoRA 路径（和真实训练同一条），并刻意把学习率调大，好让 μ=4 的 4 步
    漂移足够撞上 ±20% 的裁剪带。两个对照都在这里：

        μ=1 → 无论 lr 多大，最后一轮仍是 0（ρ≡1 是结构性的，不是"漂移恰好小"）；
        μ=4 → 第 1 轮必然 0（与采样同参数），最后一轮 > 0。

    真实训练（lr=1e-5）的触发率会低得多，具体多少看日志里那列
    `μ4 clip 0%→X%` —— 它就是为这个准备的读数。
    """
    def run(mu: int):
        torch.manual_seed(0)            # 两个 trainer 的初始权重逐位相同
        trainer = GRPOTrainer(
            model=lora_model(), tokenizer=FakeTokenizer(),
            cfg=make_cfg(kl_coef=0.0, inner_iterations=mu, learning_rate=5e-3),
            output_dir=str(tmp_path), scorer=FactStubScorer(),
        )
        return trainer.step(make_prompts())

    one, four = run(1), run(4)
    assert one["clipped_frac_last"] == 0.0, "μ=1 时 ρ≡1，裁剪不可能触发"
    assert four["clipped_frac_first"] == 0.0, "第 1 轮与采样同参数，必然 0"
    assert four["clipped_frac_last"] > 0.0, "μ=4 漂移后仍未触发裁剪"
    assert four["advantage_abs_mean"] > 0


def test_μ越大同一批数据把策略推得越远(tmp_path):
    """μ 的语义：同一批 rollout 多做几次更新 ⇒ 这一步结束时策略漂移更大。

    两个 trainer 的初始权重逐位相同（tiny_model 固定了种子），漂移量可直接比。
    """
    def drift(mu):
        model = lora_model()
        before = {
            n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n
        }
        trainer = GRPOTrainer(
            model=model, tokenizer=FakeTokenizer(),
            cfg=make_cfg(kl_coef=0.0, inner_iterations=mu),
            output_dir=str(tmp_path), scorer=FactStubScorer(),
        )
        trainer.step(make_prompts())
        sq = 0.0
        for n, p in model.named_parameters():
            if n in before:
                d = (p.detach() - before[n]).float()
                sq += float((d * d).sum())
        return sq ** 0.5

    d1, d4 = drift(1), drift(4)
    assert d4 > d1 * 1.5, f"μ=4 的漂移 {d4:.4g} 没有明显大于 μ=1 的 {d1:.4g}"
