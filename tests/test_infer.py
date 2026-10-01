"""sft/infer.py 的验收测试。

这个模块此前**一条测试都没有**，而它的 docstring 里列着三个"不报错、
只是结果莫名其妙"的坑：

  1. 没切掉 prompt 的 token → 输出里带着原文
  2. 没 strip → 输出带着模板残留的空白
  3. 批处理用了右 padding → 位置错位，输出错乱

这三条都不会抛异常，只会让评测和后续的事实提取悄悄失真，所以必须用
假模型把"喂进去的 tensor"和"切出来的结果"都锁住。
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from sfzy.sft.infer import (
    generate_batch,
    generate_one,
    summarize_records,
    summarize_records_batched,
)

# 内容用 ASCII：假 tokenizer 把字符映射成 `10 + ord(c) % 100`，
# 这对中文是有损的（解码回不来），而下面几条测试要比对解码出来的文本。
MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "doc"},
]


def has_subsequence(needle: list[int], haystack: list[int]) -> bool:
    """haystack 里是否包含连续的 needle —— 用来做 token 级断言，绕开有损解码。"""
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


class FakeTokenizer:
    """和 test_chat_template / test_collator 同一套约定：一个字符一个 id。"""

    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"
    GENERATION_MARKER = 99

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[10 + (ord(c) % 100) for c in text])

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids = []
        for message in messages:
            ids.extend(self(message["content"]).input_ids)
        if add_generation_prompt:
            ids.append(self.GENERATION_MARKER)
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        # id < 10 是 pad/eos，直接丢掉；其余按 __call__ 的映射反解
        return "".join(chr(i - 10) for i in token_ids if i >= 10)


class FakeModel(nn.Module):
    """记录每次 generate 的入参，并**回声**第一个真实 token 当作生成结果。

    为什么要回声而不是返回固定文本：批处理里每个 prompt 的第一真实 token
    不同（左 padding 时它在不同列），回声能证明"第 i 条输出确实来自第 i 个
    prompt"，而不是顺序错乱或者拼接错位。
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))   # generate_one 要读 .device
        self.calls: list[dict] = []

    def generate(self, input_ids=None, attention_mask=None, num_return_sequences=1, **kwargs):
        self.calls.append({
            "input_ids": input_ids.clone(),
            "attention_mask": None if attention_mask is None else attention_mask.clone(),
            "kwargs": kwargs,
        })
        rows = input_ids.repeat_interleave(num_return_sequences, dim=0)
        if attention_mask is not None:
            masks = attention_mask.repeat_interleave(num_return_sequences, dim=0)
            first_real = masks.argmax(dim=1, keepdim=True)
        else:
            first_real = torch.zeros((rows.shape[0], 1), dtype=torch.long)
        echo = rows.gather(1, first_real)
        suffix = torch.full_like(echo, 10 + ord("+"))
        return torch.cat([rows, echo, suffix], dim=1)


def make_model() -> FakeModel:
    return FakeModel()


# ---------------------------------------------------------------- 单条生成

def test_generate_one_只返回新生成的部分():
    """不切掉 prompt 的话，输出里会带上一整段原文 —— 不报错，只是长得莫名。"""
    model, tokenizer = make_model(), FakeTokenizer()
    prompt_len = len(tokenizer.apply_chat_template(MESSAGES))

    summary = generate_one(model, tokenizer, MESSAGES)

    # 假模型生成 2 个 token：回声第一个真实 token + 一个固定的 '+'
    # （不去断言回声具体是哪个字符 —— 假 tokenizer 的 `ord(c) % 100`
    #  映射对部分 ASCII 也是有损的，那属于假件的特性，不是被测行为）
    assert len(summary) == 2 and summary.endswith("+"), f"应当只含新生成了 2 个 token：{summary!r}"
    assert "sys" not in summary, "prompt 的内容不该出现在输出里"
    assert model.calls[-1]["input_ids"].shape == (1, prompt_len)


def test_generate_one_会strip首尾空白():
    """实测旧模型 100% 的输出以空白开头，那是模板拼接的产物。"""

    class SpaceModel(FakeModel):
        def generate(self, input_ids=None, attention_mask=None, num_return_sequences=1, **kw):
            out = super().generate(input_ids, attention_mask, num_return_sequences, **kw)
            space = torch.full_like(out[:, :1], 10 + ord(" "))
            return torch.cat([out, space], dim=1)   # 末尾补一个空格

    tokenizer = FakeTokenizer()
    out = generate_one(SpaceModel(), tokenizer, MESSAGES)
    assert not out.startswith(" ") and not out.endswith(" "), f"首尾空白必须 strip：{out!r}"


def test_generate_one_按max_length截断并保留尾部():
    """截断规则必须和训练时的 collator 一致：砍头部、保尾部。

    判决书的"本院认为""判决如下"都在结尾，砍掉就等于砍掉答案。
    """
    model, tokenizer = make_model(), FakeTokenizer()

    generate_one(model, tokenizer, MESSAGES, max_length=5)

    passed = model.calls[-1]["input_ids"][0].tolist()
    full = tokenizer.apply_chat_template(MESSAGES)
    assert len(passed) == 5
    assert passed == full[-5:], "截断后应当是原序列的尾部"


def test_贪心时显式清掉temperature():
    """贪心解码下 temperature/top_p/top_k 没有意义，但**不能只是不传**。

    Qwen2.5 的 `generation_config.json` 自带 temperature=0.7 / top_p=0.8 /
    top_k=20。只传 `do_sample=False` 的话，transformers 合并配置后会打印

        The following generation flags are not valid and may be ignored:
        ['temperature', 'top_p', 'top_k']

    所以贪心路径要显式把它们传成 None 清掉（见 compat.greedy_generation_kwargs）。
    采样路径反之 —— 那时 temperature/top_p 是真正要生效的参数。
    """
    model, tokenizer = make_model(), FakeTokenizer()

    generate_one(model, tokenizer, MESSAGES, do_sample=False)
    kwargs = model.calls[-1]["kwargs"]
    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] is None
    assert kwargs["top_p"] is None
    assert kwargs["top_k"] is None

    generate_one(model, tokenizer, MESSAGES, do_sample=True, temperature=0.9, top_p=0.95)
    assert model.calls[-1]["kwargs"]["temperature"] == 0.9
    assert model.calls[-1]["kwargs"]["top_p"] == 0.95


# ---------------------------------------------------------------- 批量生成

def test_generate_batch_用左padding():
    """右 padding 会让位置编码和注意力掩码错位，输出会莫名其妙地错乱。"""
    model, tokenizer = make_model(), FakeTokenizer()
    long_msgs = [
        {"role": "system", "content": "甲"},
        {"role": "user", "content": "很长的一段文书原文。"},
    ]

    generate_batch(model, tokenizer, [MESSAGES, long_msgs])

    call = model.calls[-1]
    input_ids, mask = call["input_ids"], call["attention_mask"]
    short_len = len(tokenizer.apply_chat_template(MESSAGES))
    long_len = len(tokenizer.apply_chat_template(long_msgs))
    assert input_ids.shape[1] == long_len, "应补齐到批内最长"
    assert input_ids[0, : long_len - short_len].eq(0).all(), "短的那条左边应当是 pad"
    assert input_ids[0, -1].item() == tokenizer.GENERATION_MARKER, "真实内容必须右对齐"
    assert mask[0].sum().item() == short_len
    assert mask[1].sum().item() == long_len


def test_generate_batch_每条输出对得上自己的prompt():
    """用"回声第一个真实 token"验证第 i 条输出确实来自第 i 个 prompt。"""
    model, tokenizer = make_model(), FakeTokenizer()
    jia = [{"role": "system", "content": "A"}, {"role": "user", "content": "doc"}]
    yi = [{"role": "system", "content": "B"}, {"role": "user", "content": "doc"}]

    out = generate_batch(model, tokenizer, [jia, yi])

    assert out[0].startswith("A"), f"第 0 条应当回声 A 的 prompt：{out[0]!r}"
    assert out[1].startswith("B"), f"第 1 条应当回声 B 的 prompt：{out[1]!r}"


def test_generate_batch_多采样时按prompt分组():
    """GRPO 的 rollout 要靠这个顺序把 G 条输出归回同一个 prompt。"""
    model, tokenizer = make_model(), FakeTokenizer()
    jia = [{"role": "system", "content": "A"}, {"role": "user", "content": "doc"}]
    yi = [{"role": "system", "content": "B"}, {"role": "user", "content": "doc"}]

    out = generate_batch(model, tokenizer, [jia, yi], num_return_sequences=3,
                         do_sample=True)

    assert len(out) == 6, "2 个 prompt × G=3"
    assert out[0].startswith("A") and out[1].startswith("A") and out[2].startswith("A")
    assert out[3].startswith("B") and out[4].startswith("B") and out[5].startswith("B")


def test_generate_batch_结束后恢复padding_side():
    """临时改 padding_side 之后必须还原，否则会污染后面的训练/评测。"""
    model, tokenizer = make_model(), FakeTokenizer()
    tokenizer.padding_side = "right"

    generate_batch(model, tokenizer, [MESSAGES])

    assert tokenizer.padding_side == "right"


def test_generate_batch_空输入返回空列表():
    model, tokenizer = make_model(), FakeTokenizer()
    assert generate_batch(model, tokenizer, []) == []
    assert model.calls == [], "不该白跑一次 generate"


# ---------------------------------------------------------------- 遍历接口

def make_records(n: int = 3) -> list[dict]:
    return [{"id": f"doc{i}", "source": f"文书原文{i}", "summary": "参考摘要"} for i in range(n)]


def test_summarize_records_batched_保持顺序和字段():
    model, tokenizer = make_model(), FakeTokenizer()

    results = list(summarize_records_batched(
        model, tokenizer, make_records(5), batch_size=2, max_new_tokens=8,
    ))

    assert [r["id"] for r in results] == [f"doc{i}" for i in range(5)]
    assert all(set(r) == {"id", "summary"} for r in results)
    assert len(model.calls) == 3, "5 条按 batch=2 分 3 批"


def test_summarize_records_顺序路径接口一致():
    """两条路径必须产出同样形状的结果，否则 generate_triples 的分支会不一致。"""
    model, tokenizer = make_model(), FakeTokenizer()

    results = list(summarize_records(model, tokenizer, make_records(3)))

    assert [r["id"] for r in results] == ["doc0", "doc1", "doc2"]
    assert all(set(r) == {"id", "summary"} for r in results)


def test_检索器存在时把contexts拼进prompt():
    """RAG 开关走的是 dataset 里同一个约定：retriever.retrieve(q, top_k)。"""

    class Doc:
        def __init__(self, text):
            self.text = text

    class Retriever:
        top_k = 2

        def __init__(self):
            self.queries = []

        def retrieve(self, query, k):
            self.queries.append((query, k))
            return [Doc("法条一"), Doc("法条二")]

    model, tokenizer = make_model(), FakeTokenizer()
    retriever = Retriever()

    list(summarize_records_batched(
        model, tokenizer, make_records(2), batch_size=1,
        retriever=retriever, prompt_style="rag",
    ))

    assert retriever.queries == [("文书原文0", 2), ("文书原文1", 2)]
    # 用 token 级子序列判断，不做有损的 decode（中文字符在这个假 tokenizer
    # 里会被 `ord(c) % 100` 折叠，解码回来已经不是一个汉字了）
    passed = model.calls[-1]["input_ids"][0].tolist()
    assert has_subsequence(tokenizer("法条一").input_ids, passed), "检索结果必须进 prompt"
