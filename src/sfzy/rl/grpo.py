"""GRPO（Group Relative Policy Optimization）手写实现。

============================ 它和 PPO 差在哪 ============================
PPO 需要一个 critic（价值网络）估计"这个状态值多少"，好处是基线准，
代价是要多训一个和策略同规模的模型 —— 显存翻倍。

GRPO 把基线换成**同一 prompt 下多个采样的平均奖励**：

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

不需要 critic。代价是每个 prompt 要采样 G 条，**用推理时间换显存**。

============================ 为什么它适合摘要任务 ============================
摘要的奖励是可验证的（ROUGE、事实覆盖），同一条 prompt 采样 G 条
各自的奖励有真实方差，组内归一化就能给出有意义的优势估计。

反过来，如果奖励是"人类打分"（噪声大、方差小），组内归一化会把噪声
也放大 —— 这是 GRPO 在不适合的场景里的典型失败模式。

============================ 三个容易写错的地方 ============================
1. **old_logprobs 必须 detach，且来自当前策略**。它是采样那一刻的
   概率，是常数。忘了 detach 会让梯度算错，而且不报错。
   更重要的是：rollout 和 update 之间不能有别的参数更新，
   否则 ratio 偏离 1 很远，裁剪一直生效、梯度几乎无效。
2. **长度归一化不能省**。序列级 logprob 是求和的，长序列的梯度尺度
   天然更大。我们的输出长度 122~1659 字符，差 13 倍。
3. **组内全同时要退化成 0 而不是 nan**。std=0 时除法会炸；
   加 eps 之后 advantage 全 0 —— 这其实是对的：**这一组没提供任何
   偏好信息，不该产生梯度**。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """组内归一化。输入形状 (num_prompts, group_size)，输出同形状。

    A = (r - mean) / (std + eps)

    减均值是方差缩减（不减去基线的话，所有采样被同等鼓励，学不到方向）；
    除以标准差是把不同 prompt 的奖励尺度拉到相近 —— 有的参考摘要长、
    ROUGE 天然高，有的短、天然低，不归一化的话尺度大的 prompt 会主导梯度。

    `eps` 不能省：一组采样恰好奖励全同时 std=0，除法直接出 nan。
    加上之后 advantage 全变 0，这是**正确的退化行为**。
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


def group_mask(
    rewards: torch.Tensor,
    min_std: float = 1e-6,
    baseline: Optional[torch.Tensor] = None,
    slack: float = 0.0,
) -> torch.Tensor:
    """标出"该产生梯度"的组，返回形状 (num_prompts,) 的 bool 张量。

    奖励全同的组（全过门控 / 全被门控）归一化后 advantage 恒为 0，
    **算它们纯属浪费算力**。主动过滤掉，省下的时间可以多跑几组有区分度的。

    这是 RTL-RLVR 那条"n=8 配合组内过滤"的做法。

    ---- baseline：SFT 基线锚（防止整体退化）----
    `baseline[i]` 是第 i 个 prompt 上 **SFT 输出**的奖励，形状 (num_prompts,)。
    给了它之后，只有满足"组内最好的那一条 ≥ 基线 - slack"的组才保留。

    为什么必须是过滤而不是"奖励里减去基线"：

        GRPO 的 advantage 是组内归一化 A_i = (r_i - mean_j r_j) / std_j r_j，
        **给组内所有 r 同加同减一个常数，A_i 完全不变**。

    所以"r_i - 基线"这种写法对梯度毫无影响（有测试钉住这条）。
    要让基线真的起作用，只能走非线性：要么过滤整组（这里），
    要么在奖励里做门控（`reward.py` 的 mode=gated*）。

    `slack` 是容差：slack=0 意味着"只要整组都不如 SFT 就丢掉"，
    在训练早期这会频繁触发。给 0.05 表示"比 SFT 差 5 分以内还算这一组的
    相对排序有意义"。**这一项是实验变量，不是拍脑袋的常数。**

    被过滤的样本在日志里记为 `anchor_filtered`，和 `无区分度` 分开统计 ——
    前者说明策略在退步，后者只是运气不好，两者要做的事完全不同。
    """
    keep = rewards.std(dim=-1, unbiased=False) > min_std
    if baseline is not None:
        best = rewards.max(dim=-1).values
        keep = keep & (best >= baseline.to(rewards.device) - slack)
    return keep


def mask_advantages(
    advantages: torch.Tensor, keep: torch.Tensor, group_size: int
) -> torch.Tensor:
    """把被过滤掉的组的 advantage 置 0，返回展平后的张量。

    置 0 之后这些组对 loss 的贡献是常数 0，梯度自然为 0 —— 等价于不训练它们，
    但**保留在同一个 batch 里**，不用重排张量。
    """
    flat = advantages.reshape(-1)
    mask = keep.repeat_interleave(group_size).to(flat.device, dtype=flat.dtype)
    return flat * mask


def length_normalize(
    advantages: torch.Tensor, lengths: torch.Tensor, mode: str = "sqrt"
) -> torch.Tensor:
    """按序列长度缩放 advantage，抵消"长序列梯度更大"的偏差。

    序列级 logprob 是**求和**的，梯度范数随长度增长。我们的输出长度
    122~1659 字符（差 13 倍），不归一化会系统性地偏袒长输出 ——
    而这个任务本来就容易长度膨胀了。

    mode:
        "sqrt"  按 √长度 缩放（默认，折中）
        "mean"  按长度缩放（完全抵消，但对短序列放得太大）
        "none"  不缩放
    """
    if mode == "none":
        return advantages
    if mode == "mean":
        scale = lengths.clamp(min=1).float()
    elif mode == "sqrt":
        scale = lengths.clamp(min=1).float().sqrt()
    else:
        raise ValueError(f"未知的 length_normalize mode: {mode}")
    return advantages / scale


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (loss, 平均裁剪比例)。

    形状都是 (num_sequences,)：

        ratio    = exp(logprobs - old_logprobs)          ← 重要性比率
        loss     = -mean(min(ratio * A, clip(ratio) * A))

    **为什么要裁剪。** 策略更新步子太大会直接毁掉模型。裁剪把 ratio
    限制在 [1-ε, 1+ε] 内，等价于"一次更新别让某个 token 的概率变化太多"。

    **为什么取 min 而不是取裁剪后的值。** 这是 PPO 的 "pessimistic"
    选择：advantage > 0 时只按裁剪上限给梯度，不会因为 ratio 涨得特别大
    而过度奖励；advantage < 0 时同理。

    第二个返回值是**被裁剪的比例**，用来监控：如果长期接近 100%，
    说明 rollout 和 update 之间策略漂移太大（多半是 old_logprobs 用错了）。
    """
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    per_sequence = torch.min(unclipped, clipped)

    clipped_frac = (unclipped != per_sequence).float().mean()
    return -per_sequence.mean(), clipped_frac


def kl_penalty(
    policy_logprobs: torch.Tensor, reference_logprobs: torch.Tensor, coef: float = 1.0
) -> torch.Tensor:
    """KL(π_θ || π_ref) 的 k3 无偏估计，返回标量。

        k3 = mean( exp(ref - policy) - (ref - policy) - 1 )

    为什么不直接用 (policy - ref) 的均值：那是有偏的，而且当两个分布
    接近时会趋近于 0，起不到约束作用。k3 是非负的无偏估计。

    **KL 系数在我们的场景里要谨慎调。** SFT 本身就有"漏事实"的习惯，
    而 KL 惩罚会**保护这个错误行为** —— 它把策略往 SFT 拉，而 SFT 就是
    要改进的对象。所以 configs/grpo_cloud.yaml 里准备了 0 / 0.01 / 0.05
    三档做对照，不能照抄论文默认值。
    """
    log_ratio = reference_logprobs - policy_logprobs
    return coef * (log_ratio.exp() - log_ratio - 1.0).mean()


def compute_advantages(
    rewards: torch.Tensor,
    lengths: torch.Tensor,
    length_mode: str = "sqrt",
    eps: float = 1e-4,
    baseline: Optional[torch.Tensor] = None,
    slack: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """串起来：组内归一化 → 长度归一化 → 组过滤。返回 (advantages, 组掩码)。

    advantages 已经**按组掩码置 0**（被过滤的组不产生梯度），所以调用方
    直接拿去算 loss 就行，不用再管掩码。掩码同时返回是为了打日志：
    `kept_groups / total_groups` 是判断奖励设计好坏的第一手信号。
    """
    advantages = group_advantages(rewards, eps=eps)
    flat_len = lengths.reshape(-1).float()
    flat = length_normalize(advantages.reshape(-1), flat_len, mode=length_mode)
    keep = group_mask(rewards, baseline=baseline, slack=slack)
    return mask_advantages(flat, keep, rewards.shape[-1]), keep
