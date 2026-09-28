"""rl/dpo.py 的 `sequence_logprob` 验收测试。

这个函数是 DPO 和 GRPO **共用**的核心：GRPO 的 old_logprobs、策略 logprobs、
KL 的参考 logprobs 全走它。它此前一条测试都没有 —— 所以 `logsumexp` 传了
一个不存在的 `dtype` 参数、一调就 TypeError，居然一直没被发现
（见下面对 `test_与手工实现一致` 的说明）。

    python -m pytest tests/test_dpo.py -q
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sfzy.rl.dpo import sequence_logprob


def tiny_model(vocab: int = 64):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=vocab, n_positions=32, n_embd=16, n_layer=1, n_head=2, n_inner=32)
    )
    model.eval()
    return model


def manual_logprob(model, input_ids, labels, shift=True):
    """手工实现一遍（本文档用的就是最直白的写法），用来对拍。"""
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    if shift:
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = labels != -100
    safe = labels.masked_fill(~mask, 0)
    return (logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=-1)


def test_与手工实现一致():
    """这条测试的存在本身就是发现 bug 的原因：
    在此之前 reward/grpo 的单元测试全绿，但没有任何测试真正调用过这个函数。
    """
    model = tiny_model()
    input_ids = torch.randint(1, 64, (2, 10))
    labels = input_ids.clone()
    got = sequence_logprob(model, input_ids, labels)
    assert torch.allclose(got, manual_logprob(model, input_ids, labels), atol=1e-4)


def test_必须错开一位():
    """logits[i] 预测的是 token[i+1]。不错开的话算出来的是"复制输入"的置信度，
    数值看着正常但完全错误。"""
    model = tiny_model()
    input_ids = torch.randint(1, 64, (1, 8))
    labels = input_ids.clone()
    shifted = sequence_logprob(model, input_ids, labels, shift=True)
    unshifted = manual_logprob(model, input_ids, labels, shift=False)
    assert not torch.allclose(shifted, unshifted, atol=1e-3)


def test_只累加答案段():
    """labels 里 -100 的位置是 prompt / padding，不该贡献任何 logprob。"""
    model = tiny_model()
    input_ids = torch.randint(1, 64, (1, 8))
    full = input_ids.clone()
    masked = full.clone()
    masked[:, :4] = -100                     # 前 4 个当 prompt
    total = sequence_logprob(model, input_ids, full)
    only_answer = sequence_logprob(model, input_ids, masked)
    assert only_answer.abs().item() < total.abs().item()
    assert torch.allclose(only_answer, manual_logprob(model, input_ids, masked), atol=1e-4)


def test_全掩码时为零():
    model = tiny_model()
    input_ids = torch.randint(1, 64, (1, 6))
    labels = torch.full_like(input_ids, -100)
    assert sequence_logprob(model, input_ids, labels).item() == 0.0


def test_半精度下不崩且量级合理():
    """这就是原来崩溃的地方：logsumexp 没有 dtype 参数。
    同时也确认不手动 upcast 整份 logits 时数值仍然可用。"""
    model = tiny_model().half()
    input_ids = torch.randint(1, 64, (1, 10))
    labels = input_ids.clone()
    got = sequence_logprob(model, input_ids, labels)
    assert torch.isfinite(got).all()
    # 10 个 token、词表 64，逐 token 的 logprob 大致在 -1 ~ -10 之间
    assert -100 < got.item() < 0


def test_批量形状正确():
    model = tiny_model()
    input_ids = torch.randint(1, 64, (4, 9))
    out = sequence_logprob(model, input_ids, input_ids.clone())
    assert out.shape == (4,)
