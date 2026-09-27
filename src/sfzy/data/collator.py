"""批量张量化：把样本列表变成可以喂给模型的 tensor 批次。

============================ 你要实现的文件 ============================

-------------------------------- 这个文件最重要的一件事 --------------------------------
**loss mask。**

对话样本是 "prompt + 答案" 拼起来的，但只有**答案部分**算 loss。
prompt 部分的 label 必须置成 -100（PyTorch 的 CrossEntropyLoss 会忽略 -100）。

不置 mask 会怎样：模型会去学习"如何生成 prompt 本身"，
既浪费容量又带偏训练目标，而且 loss 曲线看着还更"正常"——
这个 bug 不会报错，只会让指标莫名其妙地差。这是面试高频追问点。

-------------------------------- 拼装方式 --------------------------------
    prompt_ids = chat_template.encode_prompt(tokenizer, messages)   # 含生成提示符
    answer_ids = chat_template.encode_answer(tokenizer, answer)     # 末尾带 eos

    input_ids = prompt_ids + answer_ids
    labels    = [-100] * len(prompt_ids) + answer_ids

-------------------------------- 两个需要你做判断的地方 --------------------------------
1. **超长怎么截**？判决书很长，max_length 放不下是常态。
   从**尾部**截（丢掉后半段）还是从**头部**截（丢掉开头）？
   注意民事判决书里"本院认为""判决如下"都在**结尾**，
   而结尾恰恰是摘要最需要的部分。想清楚你的策略并在代码里注释理由。

2. **padding 用什么值**？input_ids 用 pad_token_id，但 **labels 必须用 -100**，
   不能用 pad_token_id——否则 padding 也参与算 loss。
=====================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from sfzy.models.chat_template import encode_answer, encode_prompt


class SFTCollator:
    """把一批样本拼成 (input_ids, attention_mask, labels) 的批次。

    输入 examples 是 dataset 产出的列表，每条形如
        {"id": str, "messages": List[dict], "answer": str}

    输出 dict（键名要和 trainer 里对得上）：
        input_ids      LongTensor (B, L)
        attention_mask LongTensor (B, L)
        labels         LongTensor (B, L)
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 2048,
        pad_to_multiple_of: int = 8,
    ) -> None:
        """pad_to_multiple_of 让序列长度对齐到 8 的倍数，对 GPU 更友好。

        提示：从 tokenizer 里取 pad_token_id。如果它是 None，
        在 loader.load_tokenizer 里就应该已经用 eos 顶上过了。
        """
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            raise ValueError(f"tokenizer.pad_token_id 应该已经被初始化，但是没有")
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        all_input_ids, all_labels = [], []

        # 第一步：逐条构造 input_ids 和 labels，并截断
        for ex in examples:
            p_ids = encode_prompt(
                self.tokenizer, ex["messages"], add_generation_prompt=True
            )
            a_ids = encode_answer(self.tokenizer, ex["answer"])
            p_ids, a_ids = self._truncate(p_ids, a_ids)

            input_ids = p_ids + a_ids
            labels = [-100] * len(p_ids) + a_ids   # 关键：prompt 置 -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # 第二步：统计 batch 内最长长度，对齐到 pad_to_multiple_of
        max_len = max(len(ids) for ids in all_input_ids)
        max_len = ceil_multiple(max_len, self.pad_to_multiple_of)

        # 第三步：padding
        pad_id = self.tokenizer.pad_token_id
        padded_input_ids = pad_maxlen(all_input_ids, max_len, pad_id)
        padded_labels = pad_maxlen(all_labels, max_len, -100)
        attention_mask = [
            [1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_input_ids
        ]

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        

    def _truncate(self, prompt_ids, answer_ids):
        """保证答案不被截断，优先砍 prompt；prompt 从头部砍，保留结尾。"""
        total = len(prompt_ids) + len(answer_ids)
        if total <= self.max_length:
            return prompt_ids, answer_ids
        keep = self.max_length - len(answer_ids)
        if keep <= 0:
            # 答案本身超长，只能截答案，但至少保留末尾 eos
            return [], answer_ids[-self.max_length:]
        return prompt_ids[-keep:], answer_ids


def ceil_multiple(x: int, n: int) -> int:
    """把 x 向上扩为 n 的倍数。"""
    return ((x + n - 1) // n) * n

def pad_maxlen(ids:List[List],maxlen:int,pad_id:int)->List[List]:
    return [id + (maxlen - len(id)) * [pad_id] for id in ids]