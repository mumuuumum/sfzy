"""data/collator.py 的验收测试。

核心是验证 **loss mask**：prompt 段的 label 必须是 -100，
只有答案段参与算 loss。这个 bug 不会报错，只会让指标莫名变差，
所以必须用测试锁住。

测试用一个假的 tokenizer（每个字符映射成一个 id），
不依赖任何模型下载，纯 CPU 秒跑：
    python -m pytest tests/test_collator.py -q
"""

from __future__ import annotations

import types

import pytest
import torch

from sfzy.data.collator import SFTCollator
from sfzy.models.chat_template import encode_answer, encode_prompt


class FakeTokenizer:
    """最小可用的假 tokenizer。

    * 每个字符映射成 10 + ord(c) % 200（范围 10~209，避开 0/2）
    * pad_token_id = 0，eos_token_id = 2
    * apply_chat_template 把各条消息拼起来，最后追加 999 当作生成提示符
    """

    pad_token_id = 0
    eos_token_id = 2
    GENERATION_MARKER = 999

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(
            input_ids=[10 + (ord(c) % 200) for c in text]
        )

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids = []
        for message in messages:
            ids.extend(self(message["content"]).input_ids)
        if add_generation_prompt:
            ids.append(self.GENERATION_MARKER)
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(i - 10) for i in token_ids if i >= 10)


def make_example(idx: int = 0, prompt_chars: int = 8, answer: str = "摘要内容") -> dict:
    return {
        "id": f"doc{idx}",
        "messages": [
            {"role": "system", "content": "系" * 3},
            {"role": "user", "content": "文" * prompt_chars},
        ],
        "answer": answer,
    }


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


# ---------------------------------------------------------------- 基本形状

def test_单条样本输出三个张量(tokenizer):
    batch = SFTCollator(tokenizer)([make_example()])
    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    for key in batch:
        assert isinstance(batch[key], torch.Tensor)
        assert batch[key].shape[0] == 1


def test_三个张量形状一致(tokenizer):
    batch = SFTCollator(tokenizer)([make_example(0), make_example(1)])
    shape = batch["input_ids"].shape
    assert batch["labels"].shape == shape
    assert batch["attention_mask"].shape == shape


def test_dtype是long(tokenizer):
    """不是 long 的话，embedding 层会报 dtype 不匹配。"""
    batch = SFTCollator(tokenizer)([make_example()])
    for key in batch:
        assert batch[key].dtype == torch.long


# ---------------------------------------------------------------- loss mask（核心）

def test_loss_mask_prompt段全为负100(tokenizer):
    example = make_example()
    batch = SFTCollator(tokenizer, pad_to_multiple_of=1)([example])

    prompt_len = len(encode_prompt(tokenizer, example["messages"]))
    labels = batch["labels"][0].tolist()

    assert labels[:prompt_len] == [-100] * prompt_len


def test_loss_mask_答案段等于答案的token(tokenizer):
    example = make_example()
    batch = SFTCollator(tokenizer, pad_to_multiple_of=1)([example])

    prompt_len = len(encode_prompt(tokenizer, example["messages"]))
    answer_ids = encode_answer(tokenizer, example["answer"])
    labels = batch["labels"][0].tolist()

    assert labels[prompt_len:] == answer_ids


def test_input_ids是prompt拼接答案(tokenizer):
    example = make_example()
    batch = SFTCollator(tokenizer, pad_to_multiple_of=1)([example])

    prompt_ids = encode_prompt(tokenizer, example["messages"])
    answer_ids = encode_answer(tokenizer, example["answer"])

    assert batch["input_ids"][0].tolist() == prompt_ids + answer_ids


def test_答案段labels与input_ids完全一致(tokenizer):
    """答案段既算 loss 又是输入，两边的 token 必须一模一样。"""
    example = make_example()
    batch = SFTCollator(tokenizer, pad_to_multiple_of=1)([example])

    prompt_len = len(encode_prompt(tokenizer, example["messages"]))
    input_tail = batch["input_ids"][0].tolist()[prompt_len:]
    label_tail = batch["labels"][0].tolist()[prompt_len:]

    assert input_tail == label_tail


def test_答案末尾必须是eos(tokenizer):
    """缺 eos 模型学不会停，推理时会生成一大段废话。"""
    example = make_example()
    batch = SFTCollator(tokenizer, pad_to_multiple_of=1)([example])

    assert batch["input_ids"][0][-1].item() == tokenizer.eos_token_id
    assert batch["labels"][0][-1].item() == tokenizer.eos_token_id


# ---------------------------------------------------------------- padding

def test_padding位置的label是负100(tokenizer):
    """padding 段若用 pad_token_id 当 label，padding 也会参与算 loss。"""
    short = make_example(0, prompt_chars=2, answer="短")
    long = make_example(1, prompt_chars=40, answer="很长的一段参考摘要内容")
    batch = SFTCollator(tokenizer, pad_to_multiple_of=1)([short, long])

    mask = batch["attention_mask"][0]
    pad_positions = (mask == 0).nonzero().flatten().tolist()
    assert pad_positions, "短样本应该产生 padding"

    for pos in pad_positions:
        assert batch["labels"][0][pos].item() == -100
        assert batch["input_ids"][0][pos].item() == tokenizer.pad_token_id


def test_attention_mask_非padding处为1(tokenizer):
    example = make_example()
    batch = SFTCollator(tokenizer, pad_to_multiple_of=1)([example])

    expected = len(encode_prompt(tokenizer, example["messages"])) + len(
        encode_answer(tokenizer, example["answer"])
    )
    assert batch["attention_mask"][0].sum().item() == expected


def test_pad_to_multiple_of_生效(tokenizer):
    batch = SFTCollator(tokenizer, pad_to_multiple_of=8)(
        [make_example(0, prompt_chars=3), make_example(1, prompt_chars=20)]
    )
    assert batch["input_ids"].shape[1] % 8 == 0


# ---------------------------------------------------------------- 截断

def test_超长样本被截断到max_length(tokenizer):
    example = make_example(prompt_chars=500, answer="摘要")
    batch = SFTCollator(tokenizer, max_length=64, pad_to_multiple_of=1)([example])
    assert batch["input_ids"].shape[1] <= 64


def test_答案不被截断(tokenizer):
    """答案平均只有 280 字，应当优先砍 prompt 而不是砍答案。"""
    example = make_example(prompt_chars=500, answer="这是一段参考摘要")
    batch = SFTCollator(tokenizer, max_length=64, pad_to_multiple_of=1)([example])

    answer_ids = encode_answer(tokenizer, example["answer"])
    labels = batch["labels"][0].tolist()

    assert labels[-len(answer_ids):] == answer_ids
    assert all(label != -100 for label in labels[-len(answer_ids):])


def test_不超长时不做截断(tokenizer):
    example = make_example(prompt_chars=5, answer="摘要")
    batch = SFTCollator(tokenizer, max_length=2048, pad_to_multiple_of=1)([example])

    expected = len(encode_prompt(tokenizer, example["messages"])) + len(
        encode_answer(tokenizer, example["answer"])
    )
    assert batch["input_ids"].shape[1] == expected


# ---------------------------------------------------------------- 批量

def test_批量多条样本(tokenizer):
    examples = [make_example(i) for i in range(4)]
    batch = SFTCollator(tokenizer)(examples)
    assert batch["input_ids"].shape[0] == 4


def test_批量中每条样本的mask各自正确(tokenizer):
    """batch 里短样本被 pad 之后，它的 mask 不能串到长样本上。"""
    examples = [
        make_example(0, prompt_chars=2, answer="短"),
        make_example(1, prompt_chars=40, answer="长一些的摘要"),
    ]
    batch = SFTCollator(tokenizer, pad_to_multiple_of=1)(examples)

    for i, example in enumerate(examples):
        valid = batch["attention_mask"][i].sum().item()
        prompt_len = len(encode_prompt(tokenizer, example["messages"]))
        answer_len = len(encode_answer(tokenizer, example["answer"]))
        assert valid == prompt_len + answer_len
        # 有效区间内，前 prompt_len 个 label 必须是 -100
        assert batch["labels"][i][:prompt_len].tolist() == [-100] * prompt_len
