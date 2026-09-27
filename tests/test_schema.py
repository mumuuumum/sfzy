"""data/schema.py 的验收测试。

这些测试定义了你实现的模块必须满足的契约。跑法：
    python -m pytest tests/test_schema.py -q
"""

from __future__ import annotations

import json

import pytest

from sfzy.data.schema import (
    Document,
    parse_document,
    parse_line,
    read_jsonl,
    to_input_record,
    to_submission_record,
    write_submission,
)

# ---------------------------------------------------------------- 测试用原始数据

RAW_FULL = {
    "id": "abc123",
    "summary": "原被告民间借贷纠纷，法院判令被告归还借款。",
    "text": [
        {"sentence": "原告张三诉被告李四民间借贷纠纷一案。", "label": 1},
        {"sentence": "本院认为，借贷关系合法有效。", "label": 1},
        {"sentence": "审判员　王五", "label": 0},
    ],
}

RAW_TEST_STYLE = {
    "id": "abc123",
    "text": [{"sentence": "原告张三诉被告李四民间借贷纠纷一案。"}],
}


# ---------------------------------------------------------------- 解析

def test_parse_常规样本():
    doc = parse_document(RAW_FULL)
    assert isinstance(doc, Document)
    assert doc.id == "abc123"
    assert doc.sentences == [
        "原告张三诉被告李四民间借贷纠纷一案。",
        "本院认为，借贷关系合法有效。",
        "审判员　王五",
    ]
    assert doc.summary == RAW_FULL["summary"]
    assert doc.has_summary is True


def test_parse_忽略句子级label():
    """label 是官方的抽取式标注，本项目的生成式路线不使用它。

    关键点：解析结果里不应该出现 label 相关字段，
    但也不能因为 label 的存在而报错。
    """
    doc = parse_document(RAW_FULL)
    assert all(isinstance(s, str) for s in doc.sentences)
    # 句子内容里不应混入 label 的痕迹
    assert not hasattr(doc, "labels")


def test_parse_测试集口径没有summary():
    doc = parse_document(RAW_TEST_STYLE)
    assert doc.summary is None
    assert doc.has_summary is False


def test_parse_line_空行应报错():
    with pytest.raises(ValueError):
        parse_line("   \n")


# ---------------------------------------------------------------- 还原原文

def test_joined_空串可无损还原原文():
    doc = parse_document(RAW_FULL)
    expected = "".join(item["sentence"] for item in RAW_FULL["text"])
    assert doc.joined() == expected
    assert doc.joined("") == expected


def test_joined_可指定分隔符():
    doc = parse_document(RAW_FULL)
    assert doc.joined("\n") == "\n".join(item["sentence"] for item in RAW_FULL["text"])


def test_stats_统计字段():
    doc = parse_document(RAW_FULL)
    stats = doc.stats()
    assert stats["num_sentences"] == 3
    assert stats["source_chars"] == len(doc.joined())
    assert stats["summary_chars"] == len(RAW_FULL["summary"])


# ---------------------------------------------------------------- 导出格式

def test_to_input_record_剥离summary与label():
    """导出成测试集口径：只留 id 和句子。"""
    doc = parse_document(RAW_FULL)
    record = to_input_record(doc)
    assert set(record.keys()) == {"id", "text"}
    assert record["id"] == "abc123"
    assert record["text"] == [
        {"sentence": "原告张三诉被告李四民间借贷纠纷一案。"},
        {"sentence": "本院认为，借贷关系合法有效。"},
        {"sentence": "审判员　王五"},
    ]
    # 绝不能残留 label
    assert all("label" not in item for item in record["text"])


def test_to_submission_record_官方提交格式():
    record = to_submission_record("abc123", "预测摘要")
    assert record == {"id": "abc123", "summary": "预测摘要"}


def test_write_submission_写出JSONL(tmp_path):
    out = tmp_path / "result.json"
    count = write_submission(out, [("a", "摘要A"), ("b", "摘要B")])
    assert count == 2

    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first == {"id": "a", "summary": "摘要A"}
    # 中文不能被转义成 \uXXXX
    assert "摘要A" in lines[0]


def test_write_submission_支持非ascii路径(tmp_path):
    out = tmp_path / "中文目录" / "result.json"
    write_submission(out, [("a", "摘要")])
    assert out.exists()


# ---------------------------------------------------------------- 读取文件

def test_read_jsonl_正常读取(tmp_path):
    path = tmp_path / "train.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(RAW_FULL, ensure_ascii=False) + "\n")
        f.write(json.dumps(RAW_TEST_STYLE, ensure_ascii=False) + "\n")

    docs = list(read_jsonl(path))
    assert len(docs) == 2
    assert docs[0].has_summary is True
    assert docs[1].has_summary is False


def test_read_jsonl_坏行报错要带行号(tmp_path):
    path = tmp_path / "broken.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(RAW_FULL, ensure_ascii=False) + "\n")
        f.write("{ 这不是合法 json\n")

    with pytest.raises(ValueError, match="第[ ]*2[ ]*行"):
        list(read_jsonl(path))
