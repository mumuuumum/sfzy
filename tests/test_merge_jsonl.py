"""tools/merge_jsonl.py 的验收（纯标准库）。

拼分片看起来只是 ``cat``，但两件事错了都不报错、只是结果悄悄变样：

  1. **顺序** —— 分片按 0/N、1/N 给出来，合并后必须还是这个顺序；
  2. **去重语义** —— keep first/last 到底留哪条、放到哪个位置，得钉死。

    python -m pytest tests/test_merge_jsonl.py -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "merge_jsonl", ROOT / "tools" / "merge_jsonl.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _write(path: Path, records) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_concat_preserves_order_and_fields(tmp_path):
    a = _write(tmp_path / "a.jsonl", [{"id": "a1"}, {"id": "a2"}])
    b = _write(tmp_path / "b.jsonl", [{"id": "b1", "extra": 1}])

    merged, per_file, dropped, missing = TOOL.merge_records([a, b])

    assert [r["id"] for r in merged] == ["a1", "a2", "b1"]
    assert merged[-1] == {"id": "b1", "extra": 1}   # 字段不动
    assert per_file == [(a, 2), (b, 1)]
    assert dropped == 0 and missing == 0


def test_dedup_keep_first(tmp_path):
    a = _write(tmp_path / "a.jsonl", [{"id": "x", "v": 1}, {"id": "y", "v": 1}])
    b = _write(tmp_path / "b.jsonl", [{"id": "x", "v": 2}])

    merged, _, dropped, missing = TOOL.merge_records([a, b], dedup=True, key="id")

    assert [r["id"] for r in merged] == ["x", "y"]
    assert merged[0]["v"] == 1          # 保留先出现的
    assert dropped == 1 and missing == 0


def test_dedup_keep_last_keeps_original_position(tmp_path):
    a = _write(tmp_path / "a.jsonl", [{"id": "x", "v": 1}, {"id": "y", "v": 1}])
    b = _write(tmp_path / "b.jsonl", [{"id": "x", "v": 2}])

    merged, _, dropped, _ = TOOL.merge_records(
        [a, b], dedup=True, key="id", keep="last"
    )

    assert [r["id"] for r in merged] == ["x", "y"]   # 位置停在第一次出现处
    assert merged[0]["v"] == 2                        # 内容用后出现的
    assert dropped == 1


def test_dedup_keeps_records_without_key(tmp_path):
    a = _write(tmp_path / "a.jsonl", [{"id": "x"}, {"noid": True}, {"noid": True}])

    merged, _, dropped, missing = TOOL.merge_records([a], dedup=True, key="id")

    assert len(merged) == 3          # 缺 key 的两条都保留
    assert dropped == 0 and missing == 2


def test_expand_inputs_glob(tmp_path):
    _write(tmp_path / "sft_shard0of2.jsonl", [{"id": "1"}])
    _write(tmp_path / "sft_shard1of2.jsonl", [{"id": "2"}])

    paths = TOOL.expand_inputs([str(tmp_path / "sft_shard*.jsonl")])

    assert [p.name for p in paths] == ["sft_shard0of2.jsonl", "sft_shard1of2.jsonl"]


def test_expand_inputs_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        TOOL.expand_inputs([str(tmp_path / "nope.jsonl")])


def test_bad_json_reports_file_and_line(tmp_path):
    path = _write(tmp_path / "bad.jsonl", [{"id": "ok"}])
    with open(path, "a", encoding="utf-8") as f:
        f.write("not json\n")

    with pytest.raises(SystemExit) as exc:
        list(TOOL.iter_records(path))

    assert "bad.jsonl:2" in str(exc.value)
