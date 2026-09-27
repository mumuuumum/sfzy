"""GRPO（Group Relative Policy Optimization）手写实现。

============================ 你要实现的文件 ============================

-------------------------------- 它和 PPO 差在哪 --------------------------------
PPO 需要一个 critic（价值网络）来估计"这个状态值多少"，好处是基线更准，
代价是要多训一个和策略同规模的模型 —— 显存直接翻倍。

GRPO 把基线换成**同一 prompt 下多个采样的平均奖励**：

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

不需要 critic。代价是每个 prompt 要采样 G 条（典型 4~16），
**用推理时间换显存**。在 4GB / 16GB 这种显存紧张的场景下是划算的。

-------------------------------- 为什么它适合摘要任务 --------------------------------
因为摘要的奖励是**可验证的**：ROUGE、法条命中、长度、格式全部可算，
同一条 prompt 采样 G 条各自的奖励有真实的方差，
组内归一化就能给出有意义的优势估计。

反过来，如果奖励是"人类打分"（噪声大、方差小），组内归一化会把噪声
也放大——这是 GRPO 在不适合的场景里的典型失败模式。
=====================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """组内归一化，返回同形状的 advantage。

    输入形状 (num_prompts, group_size)，输出同形状。

    ============================ 要写的步骤 ============================

    1. 沿 group 维度算均值和标准差（dim=1, keepdim=True）
    2. A = (r - mean) / (std + eps)

    ============================ 必要知识 ============================

    **为什么要减均值。**
    减掉组内基线后，advantage 的符号表示"这条采样比同组平均好还是差"。
    这是策略梯度的方差缩减手段——不减去基线的话，所有采样都会被
    "同等鼓励"，模型学不到方向。

    **为什么要除以标准差。**
    不同 prompt 的奖励尺度差很多（有的参考摘要长、ROUGE 天然高，
    有的短、天然低）。不归一化的话，尺度大的 prompt 会主导梯度。
    除以组内标准差把每个 prompt 的 advantage 拉到相近的尺度上。

    **eps 不能省。**
    如果一组采样恰好奖励全相同（std=0），除法会变成 nan。
    加 eps 之后 advantage 全变成 0——这其实是对的：**这一组没有提供
    任何偏好信息，不该产生梯度**。这个退化行为值得写进注释。

    **G=1 时会发生什么。**
    std=0 → advantage 全 0 → 没有梯度。所以 group_size 必须 ≥ 2，
    实践中至少 4 才有意义。
    """
    # TODO
    raise NotImplementedError("TODO: 实现 group_advantages")


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """返回标量 loss。

    ============================ 要写的步骤 ============================

    1. 算重要性比率 ratio = exp(logprobs - old_logprobs)
    2. 未裁剪项：ratio * advantages
    3. 裁剪项：clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
    4. 取两者的**逐元素最小值**（保守更新，PPO 的核心）
    5. 加负号求均值 → loss

    形状：logprobs / old_logprobs / advantages 都是 (num_prompts * group_size,)
    或 (num_prompts, group_size)，按你在 trainer 里的组织方式定。
    mask 用来屏蔽 padding / prompt 段。

    ============================ 必要知识 ============================

    **为什么要裁剪。**
    策略更新的步子太大会直接毁掉模型。裁剪把 ratio 限制在
    [1-ε, 1+ε] 内，等价于"一次更新不要让某个 token 的概率变化太多"。
    ε 通常 0.2（configs/grpo.yaml 里就是这个值）。

    **为什么取 min 而不是取裁剪后的值。**
    取 min 保证更新方向是保守的：当 advantage > 0（这条采样好）时，
    只按裁剪上限给梯度，不会因为 ratio 涨得特别大而过度奖励；
    当 advantage < 0 时同理。这是 PPO 论文里的 "pessimistic" 选择。

    **old_logprobs 是采样时刻的 logprob，必须 detach。**
    它是个常数，不参与求导。忘了 detach 会让梯度算错——
    而且不报错，只是训练效果变差。

    **KL 惩罚放哪。**
    严格说 GRPO 的 loss 里还有一项 β·KL(π_θ || π_ref)，
    用来防止策略跑离参考模型太远。本项目里的做法是把它作为
    **独立的惩罚项**在 trainer 里加上去（rl/trainer.py），
    这样 KL 系数可以在配置里单独调，不和 clip 纠缠在一起。

    **on-policy 的前提。**
    old_logprobs 必须来自**当前这一步用的那个策略**。如果用上一轮
    甚至更早的模型采样出来的 logprob，ratio 会偏离 1 很远，
    裁剪会一直生效、梯度几乎无效。所以 rollout 和 update 之间
    不能有别的参数更新——这是 GRPO 实现里最容易搞错的地方。
    """
    # TODO
    raise NotImplementedError("TODO: 实现 grpo_loss")
