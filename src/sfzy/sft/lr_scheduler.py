"""学习率调度：手写 warmup + cosine 衰减。

============================ 你要实现的文件 ============================

-------------------------------- 为什么要自己写 --------------------------------
A 那份参考脚本是**每一步手动把 lr 写进 param_group**：

    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

这比 `torch.optim.lr_scheduler` 更直白，也更好排查——你能随时打印出
"这一步的学习率到底是多少"。而且面试时能现场推导，不用背 API。

-------------------------------- 为什么要 warmup --------------------------------
训练最初几步，模型参数离预训练最优点还很远，梯度方向不稳定。
直接上大学习率容易把预训练权重带崩（LoRA 场景下后果尤其明显，
因为底座是冻结的，崩掉的部分学不回来）。
所以先用小学习率"试探"几百步，再逐步升到目标值。

-------------------------------- 契约（测试逐条检查） --------------------------------
    warmup_steps = int(total_steps * warmup_ratio)
    min_lr       = base_lr * min_lr_ratio

    step <  warmup_steps : 从 0 线性升到 base_lr
    step >= total_steps  : 恒为 min_lr
    其余                 : 从 base_lr 按余弦衰减到 min_lr

要求**在 warmup 边界处连续**：step = warmup_steps - 1 和 step = warmup_steps
都应当等于 base_lr，不能出现跳变。

        lr
    base │        ╭─────────╮
         │      ╱            ╲
         │    ╱                ╲___
    min  │  ╱                      ╲___
         └────────────────────────────── step
           ←warmup→ ←──── cosine ────→
=====================================================================
"""

from __future__ import annotations

import math


def get_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_ratio: float = 0.03,
    min_lr_ratio: float = 0.0,
) -> float:
    """返回第 step 步应当使用的学习率。

    参数：
        step         当前步数（从 0 开始）
        total_steps  总步数 = 每个 epoch 的步数 × epoch 数
        base_lr      峰值学习率
        warmup_ratio warmup 占总步数的比例，0.03 表示前 3% 用来升温
        min_lr_ratio 衰减下限相对于 base_lr 的比例，0.0 表示衰减到 0

    实现提示：
        * warmup 阶段用 `base_lr * (step + 1) / warmup_steps`，
          而不是 `base_lr * step / warmup_steps` —— 后者在第 0 步给出
          lr = 0，参数更新量为 0，第一步完全白跑。
        * cosine 的进度用 `(step - warmup_steps) / (total_steps - warmup_steps)`，
          分母要防 0（warmup_ratio 设成 1.0 时就会出现）。
        * `warmup_ratio=0` 时 warmup_steps 为 0，要直接走 cosine 分支，
          不能除零。

    有测试会检查：warmup 阶段严格递增、边界处等于 base_lr、
    末尾等于 min_lr、warmup 之后单调不增、以及各种边界参数不崩溃。
    """
    warmup_steps = int(total_steps * warmup_ratio)
    min_lr = base_lr * min_lr_ratio
    if step >= total_steps:
        return min_lr


    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    denom = total_steps - warmup_steps
    if denom <= 0:
        return min_lr
    progress = (step - warmup_steps) / denom
    cos_factor = (1.0 + math.cos(math.pi * progress)) / 2.0
    return min_lr + (base_lr - min_lr) * cos_factor
