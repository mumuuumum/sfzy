"""rl/grpo.py 的验收测试。

重点是三处"写错了不报错、只是效果变差"的地方：
  1. 组内全同时必须退化成 0 而不是 nan（std=0 会除零）
  2. 长度归一化要真的生效（不做的话系统性偏袒长输出）
  3. grpo_loss 的裁剪方向要对（取 min 而不是取裁剪值）

    python -m pytest tests/test_grpo.py -q
"""

from __future__ import annotations

import math

import pytest
import torch

from sfzy.rl.grpo import (
    compute_advantages,
    group_advantages,
    group_mask,
    grpo_loss,
    kl_penalty,
    length_normalize,
)


# ---------------------------------------------------------------- 组内归一化

def test_组内归一化_均值为零():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    adv = group_advantages(rewards)
    assert adv.mean().item() == pytest.approx(0.0, abs=1e-5)


def test_组内归一化_符号正确():
    """奖励高于组内均值的是正 advantage，低的为负。"""
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    adv = group_advantages(rewards)[0]
    assert (adv[:2] < 0).all() and (adv[2:] > 0).all()


def test_组内归一化_减均值后除以标准差():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    adv = group_advantages(rewards, eps=0.0)
    std = rewards.std(unbiased=False).item()
    assert adv[0, 3].item() == pytest.approx(1.5 / std, rel=1e-5)


def test_组内全同时退化成零而不是nan():
    """std=0 时除法会炸。加 eps 之后 advantage 全 0 —— 这是正确的：
    这一组没有任何偏好信息，不该产生梯度。"""
    adv = group_advantages(torch.tensor([[2.0, 2.0, 2.0, 2.0]]))
    assert not torch.isnan(adv).any()
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_多个prompt各自归一化():
    rewards = torch.tensor([[1.0, 3.0], [100.0, 300.0]])   # 尺度差 100 倍
    adv = group_advantages(rewards)
    # 归一化后两组应该尺度相近
    assert adv[0].abs().max().item() == pytest.approx(adv[1].abs().max().item(), rel=1e-3)


# ---------------------------------------------------------------- 组掩码

def test_组掩码_过滤无区分度的组():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]])
    mask = group_mask(rewards)
    assert mask.tolist() == [True, False]


def test_组掩码_全部有区分度():
    rewards = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
    assert group_mask(rewards).all()


# ---------------------------------------------------------------- 长度归一化

def test_长度归一化_sqrt():
    adv = length_normalize(torch.tensor([4.0]), torch.tensor([4]), mode="sqrt")
    assert adv.item() == pytest.approx(2.0)


def test_长度归一化_mean():
    adv = length_normalize(torch.tensor([4.0]), torch.tensor([4]), mode="mean")
    assert adv.item() == pytest.approx(1.0)


def test_长度归一化_none():
    adv = length_normalize(torch.tensor([4.0]), torch.tensor([4]), mode="none")
    assert adv.item() == pytest.approx(4.0)


def test_长度归一化_压住长序列的尺度():
    """同样的 advantage，长序列应该被缩得更小。"""
    adv = torch.tensor([1.0, 1.0])
    lengths = torch.tensor([100, 400])
    out = length_normalize(adv, lengths, mode="mean")
    assert out[1] < out[0]


def test_长度归一化_零长度不崩():
    out = length_normalize(torch.tensor([1.0]), torch.tensor([0]), mode="mean")
    assert math.isfinite(out.item())


def test_长度归一化_未知模式报错():
    with pytest.raises(ValueError):
        length_normalize(torch.tensor([1.0]), torch.tensor([1]), mode="乱写")


# ---------------------------------------------------------------- loss

def test_策略未变时_loss等于负平均advantage():
    """logprobs == old_logprobs → ratio 恒为 1 → loss = -mean(A)。"""
    logp = torch.tensor([-1.0, -2.0])
    adv = torch.tensor([1.0, -1.0])
    loss, clipped = grpo_loss(logp, logp.clone(), adv)
    assert loss.item() == pytest.approx(0.0)
    assert clipped.item() == 0.0


def test_策略未变时_正advantage应降低loss():
    logp = torch.tensor([-1.0, -1.0])
    adv = torch.tensor([0.5, 0.5])
    loss, _ = grpo_loss(logp, logp.clone(), adv)
    assert loss.item() == pytest.approx(-0.5)


def test_裁剪生效时被记录():
    """ratio 远超 1+ε 时会被裁，clipped 比例应该反映出来。"""
    old = torch.tensor([-10.0, -10.0])
    new = torch.tensor([0.0, 0.0])          # ratio = exp(10) 巨大
    adv = torch.tensor([1.0, 1.0])
    loss, clipped = grpo_loss(new, old, adv, clip_ratio=0.2)
    assert clipped.item() == pytest.approx(1.0)
    # 裁剪后每项是 1.2，loss = -1.2
    assert loss.item() == pytest.approx(-1.2)


def test_裁剪方向_取min而不是取裁剪值():
    """advantage 为负时，ratio 变小对策略更有利，min 会选未裁剪那一项。"""
    old = torch.tensor([0.0])
    new = torch.tensor([-10.0])             # ratio ≈ 0，远低于下界
    adv = torch.tensor([-1.0])
    loss, clipped = grpo_loss(new, old, adv, clip_ratio=0.2)
    assert clipped.item() == pytest.approx(1.0)
    # 未裁剪项 = 0 * -1 = 0；裁剪项 = 0.8 * -1 = -0.8；min 取 -0.8
    assert loss.item() == pytest.approx(0.8)


def test_组内归一化_对常数平移不变():
    """这是基线减法的真正意义：先给奖励加个常数再归一化，结果不变。

    注意**不是 loss 对平移不变** —— loss 的数值会变。不变的是归一化后的
    advantage，而"组内和恒为 0"正是基线起作用的机制。
    （我一开始把这条写成"loss 平移不变"，测试直接挂了 —— 那个前提是错的。）
    """
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.allclose(group_advantages(r), group_advantages(r + 100.0), atol=1e-5)


# ---------------------------------------------------------------- KL

def test_kl_相同分布时为零():
    lp = torch.tensor([-1.0, -2.0, -3.0])
    assert kl_penalty(lp, lp.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_非负():
    policy = torch.tensor([-1.0, -2.0])
    ref = torch.tensor([-3.0, -1.0])
    assert kl_penalty(policy, ref).item() >= 0.0


def test_kl_系数可缩放():
    policy = torch.tensor([-1.0, -2.0])
    ref = torch.tensor([-3.0, -1.0])
    assert kl_penalty(policy, ref, coef=2.0).item() == pytest.approx(
        2 * kl_penalty(policy, ref).item()
    )


# ---------------------------------------------------------------- 端到端

def test_compute_advantages_端到端():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    lengths = torch.tensor([[100, 100, 100, 100], [100, 100, 100, 100]])
    adv, mask = compute_advantages(rewards, lengths)
    assert mask.tolist() == [True, False]
    assert adv[:4].abs().sum().item() > 0     # 有效组有非零 advantage
    assert adv[4:].abs().sum().item() == 0    # 无效组全 0


def test_compute_advantages_输出展平():
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lengths = torch.tensor([[10, 10], [10, 10]])
    adv, mask = compute_advantages(rewards, lengths)
    assert adv.shape == (4,)
    assert mask.shape == (2,)
