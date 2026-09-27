"""通用文件读写工具。

与 data/schema.py 的分工：
  * 本模块只管"把文件读成 Python 对象 / 把对象写成文件"，不关心业务语义。
  * CAIL2020 格式的解析、校验放在 data/schema.py。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Union

import yaml

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """确保目录存在并返回 Path。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """逐行读取 JSONL，跳过空行。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: PathLike, records: Iterable[Dict[str, Any]]) -> int:
    """写出 JSONL，返回条数。"""
    path = Path(path)
    ensure_dir(path.parent)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_yaml(path: PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: PathLike, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def count_lines(path: PathLike) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
