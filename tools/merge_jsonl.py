"""把多个 JSONL 文件合并成一个。

分片推理（``generate_triples_vllm.py --shard I/N``、``score_reward_vllm.py``
的分片打分）都是"每个进程写一个文件、最后拼起来"，这个脚本负责最后那步。

默认行为是**按输入顺序原样拼接**，不解析业务字段、不排序、不去重；想要
``--dedup`` 时按 ``--key``（默认 ``id``）去重，保留先出现的那条。产物每行
仍是原来的 JSON，字段和顺序都不动。

用法：
    # 两个分片拼起来，写文件
    python tools/merge_jsonl.py \
        data/triples/sft_train_n4_shard0of2.jsonl \
        data/triples/sft_train_n4_shard1of2.jsonl \
        -o data/triples/sft_train_n4.jsonl

    # shell 没展开的通配符也能吃（注意加引号）
    python tools/merge_jsonl.py "data/triples/sft_train_n4_shard*.jsonl" \
        -o data/triples/sft_train_n4.jsonl

    # 按 id 去重（重复的保留先出现的），并打印统计
    python tools/merge_jsonl.py a.jsonl b.jsonl --dedup -o merged.jsonl

    # 不传 -o 就写到 stdout，方便接管道
    python tools/merge_jsonl.py a.jsonl b.jsonl | wc -l
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.utils.io import ensure_dir, write_jsonl  # noqa: E402
from sfzy.utils.logging import get_logger  # noqa: E402

logger = get_logger("merge_jsonl")

# 项目的日志默认写 stdout，而本工具支持把合并结果写到 stdout（管道用法）。
# 这个脚本是独立进程，直接把 sfzy logger 的流改到 stderr，保证 stdout 只有数据。
for _handler in logging.getLogger("sfzy").handlers:
    if isinstance(_handler, logging.StreamHandler):
        _handler.setStream(sys.stderr)


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    """把命令行参数展开成文件列表。

    支持带通配符的路径（``*`` ``?`` ``[...]``）—— 直接写字符串时由 shell
    展开，加了引号后 shell 不展开，这里用 ``glob`` 兜底。同一个文件被重复
    给出（比如 shell 和 glob 各展开一次）只保留第一次。
    """
    paths: List[Path] = []
    for raw in patterns:
        if any(ch in raw for ch in "*?["):
            matched = sorted(Path(m) for m in glob.glob(raw, recursive=True))
            if not matched:
                raise SystemExit(f"通配符没匹配到任何文件：{raw}")
            paths.extend(matched)
        else:
            path = Path(raw)
            if not path.is_file():
                raise SystemExit(f"输入文件不存在：{raw}")
            paths.append(path)

    seen = set()
    uniq: List[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)
    return uniq


def iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    """逐行读 JSONL，跳过空行；报错时带上 文件:行号。

    这里不复用 ``sfzy.utils.io.read_jsonl``：那个函数在 JSON 出错时只抛
    ``JSONDecodeError``，不知道是哪一行，拼分片时很难定位。
    """
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{lineno} 不是 JSON 对象（每行必须是 {{...}}）")
            yield obj


def merge_records(
    paths: Sequence[Path],
    *,
    dedup: bool = False,
    key: str = "id",
    keep: str = "first",
) -> Tuple[List[Dict[str, Any]], List[Tuple[Path, int]], int, int]:
    """按输入顺序拼接所有记录。

    返回 ``(合并后的记录, [(文件, 条数), ...], 去重丢掉的条数, 缺 key 的条数)``。

    去重语义：``key`` 取不到（缺字段）的记录无法判重，一律保留并计入
    "缺 key" 计数；``keep="first"`` 保留先出现的，``keep="last"`` 用后出现的
    覆盖前一条（位置仍停在第一次出现的地方，方便保持分片顺序）。
    """
    merged: List[Dict[str, Any]] = []
    per_file: List[Tuple[Path, int]] = []
    index_by_key: Dict[Any, int] = {}
    dropped = 0
    missing_key = 0

    for path in paths:
        count = 0
        for record in iter_records(path):
            count += 1
            if dedup:
                value = record.get(key)
                if value is None:
                    missing_key += 1
                elif value in index_by_key:
                    dropped += 1
                    if keep == "last":
                        merged[index_by_key[value]] = record
                    continue
                else:
                    # 只对可哈希的 key 建索引；list/dict 之类不可哈希就当缺 key
                    try:
                        index_by_key[value] = len(merged)
                    except TypeError:
                        missing_key += 1
            merged.append(record)
        per_file.append((path, count))
    return merged, per_file, dropped, missing_key


def write_records(path: Optional[Path], records: Sequence[Dict[str, Any]]) -> int:
    """写到文件或 stdout，返回写出的条数。"""
    if path is None:
        out = sys.stdout
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        return len(records)
    ensure_dir(path.parent)
    return write_jsonl(path, records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把多个 JSONL 合并成一个（默认按输入顺序原样拼接）"
    )
    parser.add_argument("inputs", nargs="+", metavar="INPUT.jsonl",
                        help="输入文件，支持通配符（加引号时由脚本自己展开）")
    parser.add_argument("-o", "--out", default=None,
                        help="输出 JSONL；不传则写到 stdout")
    parser.add_argument("--dedup", action="store_true",
                        help="按 --key 去重（默认关，纯拼接）")
    parser.add_argument("--key", default="id",
                        help="去重用的字段名，默认 id（只在 --dedup 时生效）")
    parser.add_argument("--keep", choices=["first", "last"], default="first",
                        help="去重保留哪一条：first 保留先出现的（默认），"
                             "last 用后出现的覆盖（位置不变）")
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    merged, per_file, dropped, missing_key = merge_records(
        paths, dedup=args.dedup, key=args.key, keep=args.keep
    )

    out_path = Path(args.out) if args.out else None
    written = write_records(out_path, merged)

    # 统计走 stderr（见文件顶部把 logger 改到 stderr 的那几行），stdout 只留数据
    total_read = sum(count for _, count in per_file)
    for path, count in per_file:
        logger.info("  %s: %d 条", path, count)
    if args.dedup:
        logger.info("去重键 %r（keep=%s）：读入 %d 条，丢掉重复 %d 条，缺 key %d 条",
                    args.key, args.keep, total_read, dropped, missing_key)
    logger.info("合并 %d 个文件、读入 %d 条，写出 %d 条 → %s",
                len(paths), total_read, written, out_path or "stdout")


if __name__ == "__main__":
    main()
