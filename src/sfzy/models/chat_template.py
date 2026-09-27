"""对话模板适配：把统一的 messages 转成具体模型能吃的 token 序列。

============================ 你要实现的文件 ============================

-------------------------------- 为什么需要这一层 --------------------------------
ChatGLM3-6B 和 Qwen2.5 的对话格式不一样，ChatGLM3 还自带建模代码
（加载时要 trust_remote_code=True）。上层模块不应该知道这些差异，
全部收敛到这里：

    Qwen2.5  : tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    ChatGLM3 : tokenizer.build_chat_input(...) 或它的 apply_chat_template

-------------------------------- 这一层的关键约束 --------------------------------
**loss mask 依赖"prompt 在哪里结束、答案从哪里开始"这个边界。**

如果先拼成一个大字符串再整体 tokenize，边界就丢了，collator 就没法知道
哪些 token 该算 loss。标准做法是**分别 tokenize prompt 和答案，再拼接**，
这样边界天然确定（就是 len(prompt_ids)）。

代价是拼接处的 tokenization 可能与整句 tokenize 略有差异，这是业界普遍接受的
近似，写报告时提一句即可。

-------------------------------- 统一约定 --------------------------------
输入 messages 一律是 [{"role": "system"|"user"|"assistant", "content": str}, ...]
（由 data/prompts.py 产出，且**只含 system 与 user，不含答案**）。
=====================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List


def encode_prompt(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> List[int]:
    """把 messages 编码成 prompt 的 token id 列表，**一律返回 list[int]**。

    要点：
      * add_generation_prompt=True 会追加模型专属的"该你说话了"标记
        （Qwen 是 <|im_start|>assistant\\n）。**训练和推理必须用同一个值**，
        否则训练时模型看到的上下文和推理时对不上。
      * 用 tokenizer.apply_chat_template(..., tokenize=True) 把不同模型的
        模板差异处理掉。

    ⚠️ **不同 tokenizer 的返回值类型不一致，必须在这里归一化：**

        Qwen2.5  → list[int]
        ChatGLM3 → BatchEncoding（字典形式，input_ids 是其中一个键）

    不归一化的话，collator 里的 `prompt_ids + answer_ids` 会直接报
        TypeError: unsupported operand type(s) for +: 'BatchEncoding' and 'list'

    这个坑在本地用 Qwen 调试时**永远暴露不出来** —— 只有换成 ChatGLM3
    才会出现，所以要靠 tests/test_chat_template.py 里的假 tokenizer 锁住。
    """
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )

    # BatchEncoding 支持属性访问（encoded.input_ids）
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    # 普通 dict 形式
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    # 已经是 list[int]
    return list(encoded)


def encode_answer(tokenizer: Any, answer: str) -> List[int]:
    """把参考摘要编码成答案的 token id 列表。

    要点：
      * **末尾必须补 eos**。模型要学会"说到这里就停"，缺了 eos 会导致
        推理时停不下来、生成一大段废话。
      * 通常用 tokenizer(answer, add_special_tokens=False).input_ids，
        再加 tokenizer.eos_token_id。
      * 注意有些 tokenizer 的 eos 在 encode 时会被自动加上、有些不会，
        先打印一次确认，别想当然。

    有测试会检查：返回的列表非空，且最后一个 id 等于 tokenizer.eos_token_id。
    """
    token_ids = tokenizer(answer, add_special_tokens=False).input_ids
    eos_id = tokenizer.eos_token_id
    if not token_ids or token_ids[-1] != eos_id:
        token_ids = token_ids + [eos_id]
    return token_ids


def decode(tokenizer: Any, token_ids: List[int]) -> str:
    """把 token id 解回文本，用于调试与推理后处理。

    要点：skip_special_tokens=True，否则输出里会混进 <|endoftext|> 这类标记。
    """
    return tokenizer.decode(token_ids, skip_special_tokens=True)
