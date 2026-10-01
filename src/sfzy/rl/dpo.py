"""DPO（Direct Preference Optimization）手写实现。

============================ 你要实现的文件 ============================

-------------------------------- 它是怎么绕开奖励模型的 --------------------------------
PPO 那套要先训一个 reward model，再用 RL 去优化；DPO 证明了一个结论：
**偏好数据本身就隐含了一个奖励函数**，可以直接把它写成分类损失：

    L = -log σ( β·log(π_θ(y_w)/π_ref(y_w))
                - β·log(π_θ(y_l)/π_ref(y_l)) )

其中 y_w 是 chosen、y_l 是 rejected、π_ref 是参考模型（通常是 SFT 后的模型）。
β 控制"允许偏离参考模型多少"：β 越大越保守。

-------------------------------- 这个任务上 DPO 的固有困难（要写进报告） --------------------------------
摘要任务没有天然的人类偏好对，只有两条造对的路，两条都不完美：

  * 模型自造 + ROUGE 排序 → 等于把规则奖励塞进 DPO，
    而规则奖励的正确用法是 GRPO，DPO 在这里是绕远路。
  * 参考摘要当 chosen、模型生成当 rejected → 会退化成"加权 SFT"，
    因为 chosen 恰好就是训练目标本身。

所以本项目的定位是：**DPO 作为对照实验，主线走 GRPO**。
讲清这个取舍比把 DPO 调到能跑更有价值。

-------------------------------- LoRA 场景的一个便宜技巧 --------------------------------
参考模型通常要额外加载一份完整模型（显存翻倍）。但 LoRA 场景下不需要：
**把 adapter 关掉，模型本身就是参考模型**（因为底座是冻结的、没被改过）。
PEFT 的 `model.disable_adapter()` 就是这个用途的上下文管理器。
=====================================================================
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F


def sequence_logprob(
    model: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    shift: bool = True,
    micro_batch: Optional[int] = None,
) -> torch.Tensor:
    """算每条序列上**答案部分**的对数概率之和，返回形状 (B,)。

    GRPO 和 DPO 都要用这个函数：前者要 rollout 时的 old_logprobs，
    后者要策略模型和参考模型的 logprob。所以它放在这里、两边共用。

    三个必须做对的地方：

    **一、错开一位。** 第 i 个位置的 logits 预测的是第 i+1 个 token。
    用 logits[:, :-1] 对 labels[:, 1:]。不错开的话算出来的是"复制输入"
    的置信度，数值看着正常但完全错误 —— 这是这类实现最常见的 bug。

    **二、只累加答案段。** labels == -100 的位置是 prompt 和 padding，
    要先用 masked_fill 换成 0 再 gather，否则 gather 会越界。
    最后乘 mask 求和，得到整条序列的 logprob。

    **三、不要让 fp32 的中间张量物化。** 直接 `log_softmax(logits.float())`
    会复制一份完整的 fp32 logits —— 6B 模型在 batch 4 / seq 2048 /
    vocab 65024 下是 2.1GB。改成"只 gather 目标 logit、logsumexp 用
    dtype 参数算 fp32"，数值稳定性和显存两头都占。

    `shift=False` 用于生成后的场景：此时 input_ids 已经是
    "prompt + 已生成 token"，labels 与之逐位对齐，不再错开。

    **`micro_batch` 是显存旋钮。** 一次 forward 会在 (B, L, vocab) 上物化完整
    logits —— ChatGLM3-6B 的 vocab 是 65024，B=8、L≈2400 时 fp16 就有 2.5GB，
    加上 logsumexp 的中间量，16GB 卡上就爆在这里。切分成 `micro_batch` 条一组
    后逐组前向、最后 cat 回 (B,)，峰值按组内条数线性下降。

    这样做**不改变数值**：logsumexp 只在最后一维（vocab）上做，每条序列的结果
    与同批的其它序列无关；拆开算再拼起来，和一次算是同一组数。
    `None`（默认）＝ 不切分，行为与之前完全一致。
    """
    if micro_batch is not None and 0 < micro_batch < input_ids.shape[0]:
        parts = [
            sequence_logprob(
                model,
                input_ids[start:start + micro_batch],
                labels[start:start + micro_batch],
                None if attention_mask is None else attention_mask[start:start + micro_batch],
                shift=shift,
                micro_batch=None,
            )
            for start in range(0, input_ids.shape[0], micro_batch)
        ]
        return torch.cat(parts, dim=0)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    if shift:
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]

    mask = labels != -100
    safe_targets = labels.masked_fill(~mask, 0)

    target_logits = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    # **不要给 logsumexp 传 dtype。** torch.logsumexp 没有这个参数（只有
    # log_softmax 有），传了直接 TypeError —— 而这条路径在本地从未被执行过，
    # 所以直到 GRPO 冒烟测试才发现（tests/test_smoke_grpo.py）。
    #
    # 也**不要写 logsumexp(logits.float())**：那会复制一份完整的 B×L×V fp32
    # 张量（batch 32 / seq 2432 / vocab 65024 ≈ 20GB），正是上面第三条要避免的。
    # PyTorch 的 logsumexp 对半精度输入内部按 fp32 累加，直接算就是稳的；
    # 需要 fp32 的只有最后这个 (B, L) 量级的减法。
    log_z = torch.logsumexp(logits, dim=-1)
    token_logprobs = target_logits.float() - log_z.float()

    return (token_logprobs * mask).sum(dim=-1)


@contextmanager
def reference_model_context(model: Any) -> Iterator[Any]:
    """在 LoRA 模型上临时关掉 adapter，得到参考模型的行为。

    用法：
        with reference_model_context(model):
            ref_logps = sequence_logprob(model, ...)

    非 LoRA 模型（没有 disable_adapter）时应当直接 yield model 不做任何事，
    这样同一个函数能同时服务两种场景。有测试会检查这一点。
    """
    # TODO
    raise NotImplementedError("TODO: 实现 reference_model_context")


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (loss, chosen_reward, rejected_reward)。

    ============================ 要写的步骤 ============================

    1. 算策略模型相对参考模型的**对数比**：
           policy_chosen_logps   - reference_chosen_logps
           policy_rejected_logps - reference_rejected_logps
       这个差值才是 DPO 真正在优化的东西，单独看 π_θ 的 logprob 没有意义。
    2. 乘 β，得到隐式奖励：
           chosen_reward   = β * (policy_chosen   - reference_chosen)
           rejected_reward = β * (policy_rejected - reference_rejected)
    3. loss = -logsigmoid(chosen_reward - rejected_reward).mean()

    ============================ 必要知识 ============================

    **为什么要减去参考模型。**
    不减的话，模型可以通过"把所有序列的 logprob 都抬高"来降低 loss，
    而这是没有意义的（它没学到偏好，只是变得更自信）。
    减掉参考模型之后，优化目标变成"相对参考模型，让 chosen 更可能、
    rejected 更不可能"——这个相对量才是偏好。

    **β 的作用。**
    它是对数比上的温度。β 大 → 隐式奖励被放大 → 梯度更强但也更容易
    跑偏；β 小 → 更保守。常用 0.1，你们 configs/dpo.yaml 用的也是这个。
    注意 β 和序列长度是耦合的（因为 logprob 是求和），所以换数据集
    长度分布时 β 可能要重调。

    **chosen_reward / rejected_reward 的用处。**
    它们不是 loss 的一部分，是用来监控的：正常训练下两者应该都缓慢上升
    且差距扩大。如果 chosen 不升反降，说明 β 太小或数据有问题；
    如果 rejected 也在快速上升，说明模型在"变自信"而不是"学会偏好"。

    **数值稳定性。**
    用 F.logsigmoid 而不是手写 -log(1+exp(-x))，前者在 |x| 很大时不会溢出。
    """
    # TODO
    raise NotImplementedError("TODO: 实现 dpo_loss")


def evaluate_preference_accuracy(
    chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
) -> float:
    """偏好准确率：chosen_reward > rejected_reward 的比例。

    这是 DPO 最直观的监控指标，会进 history 和 tracker。
    随机猜测是 0.5，收敛良好通常能到 0.7 以上。
    """
    # TODO
    raise NotImplementedError("TODO: 实现 evaluate_preference_accuracy")
