"""data/prepare.py 的验收测试。

注意：本模块依赖 schema.Document.joined()，所以**必须先完成 schema.py**。

跑法：
    python -m pytest tests/test_prepare.py -q
"""

from __future__ import annotations

import json

from sfzy.data.prepare import (
    build_splits,
    sample_documents,
    save_splits,
    split_documents,
    to_record,
)
from sfzy.data.schema import Document

# ---------------------------------------------------------------- 构造工具

def make_doc(idx: int, n_sent: int = 4) -> Document:
    return Document(
        id=f"doc{idx:04d}",
        sentences=[f"第{idx}号的第{i}句。" for i in range(n_sent)],
        summary=f"第{idx}号案件的裁判摘要。",
    )


def make_docs(n: int) -> list[Document]:
    return [make_doc(i) for i in range(n)]


# ---------------------------------------------------------------- to_record

def test_to_record_只有三个字段():
    assert set(to_record(make_doc(0)).keys()) == {"id", "source", "summary"}


def test_to_record_保留id与summary():
    doc = make_doc(7)
    record = to_record(doc)
    assert record["id"] == "doc0007"
    assert record["summary"] == doc.summary


def test_to_record_source是拼接后的全文():
    doc = Document(id="x", sentences=["甲", "乙", "丙"], summary="摘要")
    assert to_record(doc, joiner="")["source"] == "甲乙丙"
    assert to_record(doc, joiner="\n")["source"] == "甲\n乙\n丙"


def test_to_record_默认用空串拼接():
    doc = Document(id="x", sentences=["甲", "乙"], summary="摘要")
    assert to_record(doc)["source"] == "甲乙"


# ---------------------------------------------------------------- sample_documents

def test_sample_为None时返回全部():
    assert len(sample_documents(make_docs(10), None, seed=42)) == 10


def test_sample_数量大于总数时返回全部():
    assert len(sample_documents(make_docs(5), 100, seed=42)) == 5


def test_sample_按指定数量抽取且无重复():
    got = sample_documents(make_docs(100), 20, seed=42)
    assert len(got) == 20
    assert len({d.id for d in got}) == 20


def test_sample_同seed可复现():
    docs = make_docs(100)
    a = [d.id for d in sample_documents(docs, 20, seed=7)]
    b = [d.id for d in sample_documents(docs, 20, seed=7)]
    assert a == b


def test_sample_不同seed结果不同():
    docs = make_docs(200)
    a = [d.id for d in sample_documents(docs, 50, seed=1)]
    b = [d.id for d in sample_documents(docs, 50, seed=2)]
    assert a != b


# ---------------------------------------------------------------- split_documents

def test_split_按比例切分():
    train, dev = split_documents(make_docs(100), dev_ratio=0.2, seed=42)
    assert len(dev) == 20
    assert len(train) == 80


def test_split_不重不漏():
    docs = make_docs(100)
    train, dev = split_documents(docs, dev_ratio=0.2, seed=42)
    train_ids = {d.id for d in train}
    dev_ids = {d.id for d in dev}
    assert train_ids & dev_ids == set()
    assert train_ids | dev_ids == {d.id for d in docs}


def test_split_同seed可复现():
    docs = make_docs(50)
    a = [d.id for d in split_documents(docs, 0.2, seed=3)[1]]
    b = [d.id for d in split_documents(docs, 0.2, seed=3)[1]]
    assert a == b


def test_split_确实打乱了顺序():
    """不打乱就切，dev 会永远是文件末尾那批，两边分布会有系统性差异。"""
    _, dev = split_documents(make_docs(100), dev_ratio=0.2, seed=42)
    assert [d.id for d in dev] != [f"doc{i:04d}" for i in range(80, 100)]


def test_split_单样本时dev为空():
    train, dev = split_documents([make_doc(0)], dev_ratio=0.1, seed=42)
    assert len(train) == 1
    assert len(dev) == 0


def test_split_空输入不崩溃():
    train, dev = split_documents([], dev_ratio=0.1, seed=42)
    assert train == [] and dev == []


# ---------------------------------------------------------------- build_splits

def test_build_splits_端到端():
    splits = build_splits(make_docs(100), dev_ratio=0.2, seed=42)
    assert set(splits.keys()) == {"train", "dev"}
    assert len(splits["train"]) == 80
    assert len(splits["dev"]) == 20
    assert all(set(r.keys()) == {"id", "source", "summary"} for r in splits["train"])


def test_build_splits_采样生效():
    splits = build_splits(make_docs(1000), sample_size=100, dev_ratio=0.2, seed=42)
    assert len(splits["train"]) + len(splits["dev"]) == 100


def test_build_splits_不采样时用全量():
    splits = build_splits(make_docs(100), sample_size=None, dev_ratio=0.2, seed=42)
    assert len(splits["train"]) + len(splits["dev"]) == 100


# ---------------------------------------------------------------- save_splits

def test_save_splits_写出文件并返回条数(tmp_path):
    splits = {
        "train": [{"id": "a", "source": "原文A", "summary": "摘要A"}],
        "dev": [{"id": "b", "source": "原文B", "summary": "摘要B"}],
    }
    counts = save_splits(splits, tmp_path)
    assert counts == {"train": 1, "dev": 1}
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "dev.jsonl").exists()


def test_save_splits_中文不被转义(tmp_path):
    save_splits({"train": [{"id": "a", "source": "原文", "summary": "摘要"}]}, tmp_path)
    text = (tmp_path / "train.jsonl").read_text(encoding="utf-8")
    assert "摘要" in text
    assert "\\u" not in text


def test_save_splits_内容可被读回(tmp_path):
    save_splits({"train": [{"id": "a", "source": "原文", "summary": "摘要"}]}, tmp_path)
    line = (tmp_path / "train.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line) == {"id": "a", "source": "原文", "summary": "摘要"}


def test_save_splits_自动创建目录(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    save_splits({"train": []}, nested)
    assert (nested / "train.jsonl").exists()
