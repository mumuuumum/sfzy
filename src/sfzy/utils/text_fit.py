"""把过长的输入文本压到 token 预算内：保留头 + 尾，砍掉中间。

============================ 为什么不用 truncation_side ============================
抽取/判定 prompt 是 [system 指令][user 输入][assistant 生成标记]。
用 tokenizer 的 truncation_side 截**整段 prompt**，两头都有害：

  * 从左边切（Qwen 系常见默认）会把 system 指令切掉 → 模型接着复述 user 输入，
    实测症状就是抽取返回一段判决书原文、六要素全空；
  * 从右边切会把"该你回答了"的生成标记切掉。

所以这里只截 **user 输入本身**，system 与生成标记永远保留。判决书的关键信息
一头一尾（当事人/诉请 + 本院认为/判决结果），因此保留头尾、丢掉中间。
"""

from __future__ import annotations

from typing import Any

ELLIPSIS = "\n……（中间省略）……\n"


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_text(
    tokenizer: Any, text: str, max_tokens: int, tail_ratio: float = 0.5
) -> str:
    if max_tokens <= 0:
        return ""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    head = max(1, int(max_tokens * (1.0 - tail_ratio)))
    tail = max(0, max_tokens - head)
    parts = [tokenizer.decode(ids[:head], skip_special_tokens=True)]
    if tail:
        parts.append(tokenizer.decode(ids[-tail:], skip_special_tokens=True))
    return ELLIPSIS.join(p for p in parts if p)
