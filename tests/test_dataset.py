"""sft/dataset.py 的验收测试。

dataset 这一层最重要的职责不是"读文件"，而是**把 RAG 做成一个开关**：
retriever=None 就是纯 SFT，传了 retriever 就在取样本时检索并拼进 prompt。
消融实验因此不用改数据管线。

用一个假的 retriever，不依赖任何模型：
    python -m pytest tests/test_dataset.py -q
"""

from __future__ import annotations

import json
import types

import pytest

from sfzy.data.prompts import SUMMARY_RUBRIC
from sfzy.sft.dataset import SFTDataset, load_records


class FakeRetriever:
    """模拟 M3 的检索器：retrieve(query, k) -> list[有 .text 属性的对象]。"""

    def __init__(self, chunks=("第一条法条", "第二条法条", "第三条法条")):
        self.chunks = list(chunks)
        self.calls = []

    def retrieve(self, query, k=5):
        self.calls.append((query, k))
        return [types.SimpleNamespace(text=c) for c in self.chunks[:k]]


def make_records(n: int = 5) -> list[dict]:
    return [
        {
            "id": f"doc{i}",
            "source": f"第{i}号案件的文书原文" * 3,
            "summary": f"第{i}号案件的裁判摘要。",
        }
        for i in range(n)
    ]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- load_records

def test_load_records_读取jsonl(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, make_records(3))

    records = load_records(str(path))
    assert len(records) == 3
    assert records[0]["id"] == "doc0"
    assert set(records[0].keys()) == {"id", "source", "summary"}


# ---------------------------------------------------------------- 基本契约

def test_len等于样本数():
    assert len(SFTDataset(make_records(5))) == 5


def test_getitem_返回四个字段():
    example = SFTDataset(make_records(3))[0]
    assert set(example.keys()) == {"id", "messages", "answer", "contexts"}


def test_id与answer正确对应():
    dataset = SFTDataset(make_records(3))
    example = dataset[1]
    assert example["id"] == "doc1"
    assert example["answer"] == "第1号案件的裁判摘要。"


def test_messages只有system和user():
    """答案必须单独放在 answer 里。

    如果把答案也塞进 messages，collator 就没法确定 prompt 与答案的边界，
    loss mask 会连同答案一起置成 -100，模型永远学不会生成。
    """
    example = SFTDataset(make_records(3))[0]
    roles = [m["role"] for m in example["messages"]]
    assert "assistant" not in roles
    assert set(roles) <= {"system", "user"}


def test_答案不会泄漏进messages():
    dataset = SFTDataset(make_records(3))
    example = dataset[2]
    joined = "".join(m["content"] for m in example["messages"])
    assert example["answer"] not in joined


def test_原文出现在messages里():
    dataset = SFTDataset(make_records(3))
    example = dataset[0]
    joined = "".join(m["content"] for m in example["messages"])
    source = make_records(3)[0]["source"]
    assert source in joined


# ---------------------------------------------------------------- prompt_style

def test_默认用structured风格():
    """structured 的 user 消息里要带摘要要素清单。"""
    example = SFTDataset(make_records(1))[0]
    assert SUMMARY_RUBRIC in example["messages"][-1]["content"]


def test_可切换zeroshot风格():
    example = SFTDataset(make_records(1), prompt_style="zeroshot")[0]
    assert SUMMARY_RUBRIC not in example["messages"][-1]["content"]


# ---------------------------------------------------------------- RAG 开关

def test_无retriever时contexts为空():
    example = SFTDataset(make_records(3))[0]
    assert example["contexts"] == []


def test_有retriever时contexts被填充():
    retriever = FakeRetriever()
    example = SFTDataset(make_records(3), retriever=retriever)[0]
    assert example["contexts"] == ["第一条法条", "第二条法条", "第三条法条"]


def test_检索query来自原文():
    retriever = FakeRetriever()
    SFTDataset(make_records(3), retriever=retriever)[1]
    assert retriever.calls, "retriever 应该被调用过"
    query, _ = retriever.calls[0]
    assert "第1号案件" in query


def test_top_k透传给retriever():
    retriever = FakeRetriever()
    SFTDataset(make_records(1), retriever=retriever, top_k=2)[0]
    assert retriever.calls[0][1] == 2


def test_top_k限制contexts数量():
    retriever = FakeRetriever()
    example = SFTDataset(make_records(1), retriever=retriever, top_k=2)[0]
    assert len(example["contexts"]) == 2


def test_rag风格会把检索内容写进prompt():
    retriever = FakeRetriever()
    example = SFTDataset(
        make_records(1), retriever=retriever, prompt_style="rag"
    )[0]
    joined = "".join(m["content"] for m in example["messages"])
    assert "第一条法条" in joined


def test_每次取样本都会重新检索():
    """不能在 __init__ 里把所有样本都预先检索完——
    全量 1.2 万条 × top_k 次检索，启动时间会长到无法接受。"""
    retriever = FakeRetriever()
    dataset = SFTDataset(make_records(3), retriever=retriever)
    assert retriever.calls == []
    dataset[0]
    assert len(retriever.calls) == 1
    dataset[1]
    assert len(retriever.calls) == 2
