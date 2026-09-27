"""models/chat_template.py 的验收测试。

重点是 `encode_prompt` 的**返回值归一化**：不同 tokenizer 的
`apply_chat_template(tokenize=True)` 返回类型不一致 ——

    Qwen2.5  → list[int]
    ChatGLM3 → BatchEncoding（字典形式，input_ids 是其中一个键）

不归一化的话，collator 里的 `prompt_ids + answer_ids` 会直接报
    TypeError: unsupported operand type(s) for +: 'BatchEncoding' and 'list'

**这个坑在本地用 Qwen 调试时永远暴露不出来**，只有换成 ChatGLM3 才出现。
所以必须用假 tokenizer 把这个差异造出来、锁进测试。
"""

from __future__ import annotations

import types

import pytest

from sfzy.models.chat_template import decode, encode_answer, encode_prompt

MESSAGES = [
    {"role": "system", "content": "系"},
    {"role": "user", "content": "文书"},
]


class _BaseTokenizer:
    """公共部分：把每个字符映射成一个 id。"""

    eos_token_id = 2
    pad_token_id = 0
    GENERATION_MARKER = 99

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[10 + (ord(c) % 100) for c in text])

    def _encode(self, messages, add_generation_prompt):
        ids = []
        for message in messages:
            ids.extend(self(message["content"], add_special_tokens=False).input_ids)
        if add_generation_prompt:
            ids.append(self.GENERATION_MARKER)
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(i - 10) for i in token_ids if i >= 10)


class ListTokenizer(_BaseTokenizer):
    """像 Qwen2.5：apply_chat_template 返回 list[int]。"""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return self._encode(messages, add_generation_prompt)


class BatchEncoding(dict):
    """像 transformers 的 BatchEncoding：是 dict，同时支持属性访问。"""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class ChatGLMTokenizer(_BaseTokenizer):
    """像 ChatGLM3：apply_chat_template 返回 BatchEncoding。"""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return BatchEncoding(input_ids=self._encode(messages, add_generation_prompt))


class PlainDictTokenizer(_BaseTokenizer):
    """返回普通 dict 的变体。"""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return {"input_ids": self._encode(messages, add_generation_prompt)}


# ---------------------------------------------------------------- 返回值归一化

@pytest.mark.parametrize("tokenizer_cls", [ListTokenizer, ChatGLMTokenizer, PlainDictTokenizer])
def test_encode_prompt_一律返回list(tokenizer_cls):
    """三种 tokenizer 的返回值形态不同，encode_prompt 必须统一成 list[int]。"""
    tokenizer = tokenizer_cls()
    result = encode_prompt(tokenizer, MESSAGES)

    assert isinstance(result, list), f"{tokenizer_cls.__name__} 的返回值没被归一化"
    assert all(isinstance(x, int) for x in result), "元素必须是 int"
    assert result[-1] == tokenizer.GENERATION_MARKER, "生成提示符应该在最后"


def test_encode_prompt_三种tokenizer结果一致():
    """归一化之后，不同 tokenizer 的语义应该一致（同一个 id 序列）。"""
    results = [
        encode_prompt(cls(), MESSAGES)
        for cls in (ListTokenizer, ChatGLMTokenizer, PlainDictTokenizer)
    ]
    assert results[0] == results[1] == results[2]


def test_encode_prompt_可以拼接():
    """真正的报错场景：归一化失败时 prompt_ids + answer_ids 会 TypeError。"""
    tokenizer = ChatGLMTokenizer()
    prompt_ids = encode_prompt(tokenizer, MESSAGES)
    answer_ids = encode_answer(tokenizer, "摘要")
    combined = prompt_ids + answer_ids          # 这一行曾经报 TypeError
    assert len(combined) == len(prompt_ids) + len(answer_ids)


def test_encode_prompt_可关闭生成提示符():
    tokenizer = ListTokenizer()
    with_marker = encode_prompt(tokenizer, MESSAGES, add_generation_prompt=True)
    without = encode_prompt(tokenizer, MESSAGES, add_generation_prompt=False)
    assert len(with_marker) == len(without) + 1


# ---------------------------------------------------------------- encode_answer

def test_encode_answer_末尾补eos():
    tokenizer = ListTokenizer()
    ids = encode_answer(tokenizer, "摘要内容")
    assert ids[-1] == tokenizer.eos_token_id


def test_encode_answer_已有eos时不重复补():
    tokenizer = ListTokenizer()
    base = tokenizer("摘要", add_special_tokens=False).input_ids
    already = base + [tokenizer.eos_token_id]
    # 直接构造一个"编码结果末尾已经是 eos"的场景
    tokenizer.__call__ = lambda text, add_special_tokens=False: types.SimpleNamespace(
        input_ids=already
    )
    ids = encode_answer(tokenizer, "摘要")
    assert ids.count(tokenizer.eos_token_id) == 1, "不该出现两个连续的 eos"


def test_encode_answer_空串也要有eos():
    tokenizer = ListTokenizer()
    tokenizer.__call__ = lambda text, add_special_tokens=False: types.SimpleNamespace(input_ids=[])
    ids = encode_answer(tokenizer, "")
    assert ids == [tokenizer.eos_token_id]


# ---------------------------------------------------------------- decode

def test_decode_跳过特殊token():
    tokenizer = ListTokenizer()
    text = decode(tokenizer, [10, 11, 12])
    assert isinstance(text, str)
