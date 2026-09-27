"""CLI：数据规范化。

这个脚本只做三件事：解析参数、发现输入文件、调用 sfzy.data.prepare。
真正的过滤/采样/切分逻辑在 src/sfzy/data/prepare.py 里，不要在脚本里重复实现。

用法：
    python scripts/prepare_data.py --config configs/data.yaml
    python scripts/prepare_data.py --config configs/data.yaml --input data/raw/sample
    python scripts/prepare_data.py --config configs/data.yaml --full   # 忽略 sample_size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfzy.config import load_config              # noqa: E402
from sfzy.data.prepare import build_splits, save_splits  # noqa: E402
from sfzy.data.schema import Document, read_jsonl  # noqa: E402
from sfzy.utils.logging import get_logger        # noqa: E402
from sfzy.utils.seed import set_seed             # noqa: E402

logger = get_logger("prepare_data")


def iter_input_files(input_path: Path) -> Iterator[Path]:
    """输入可以是单个 jsonl 文件，也可以是一个目录。"""
    if input_path.is_dir():
        yield from sorted(input_path.glob("*.jsonl"))
    elif input_path.exists():
        yield input_path
    else:
        raise FileNotFoundError(f"输入路径不存在：{input_path}")


def load_documents(paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    for path in paths:
        before = len(docs)
        docs.extend(read_jsonl(path))
        logger.info("读取 %s：%d 条", path.name, len(docs) - before)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="规范化 CAIL2020 数据")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--input", default=None, help="覆盖配置里的 raw_file，可传文件或目录")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--full", action="store_true", help="忽略 sample_size，使用全量数据")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    input_path = Path(args.input) if args.input else Path(cfg.path_("data.raw_file"))
    outdir = args.outdir or cfg.path_("data.processed_dir", "data/processed")
    sample_size = None if args.full else cfg.path_("data.sample_size")

    docs = load_documents(list(iter_input_files(input_path)))
    logger.info("共读入 %d 条原始样本", len(docs))

    splits = build_splits(
        docs,
        joiner=cfg.path_("data.joiner", ""),
        dev_ratio=cfg.path_("data.splits.dev", 0.1),
        sample_size=sample_size,
        seed=seed,
    )

    counts = save_splits(splits, outdir)
    for name, n in counts.items():
        logger.info("写出 %s/%s.jsonl：%d 条", outdir, name, n)

    train = splits.get("train", [])
    if train:
        avg_source = sum(len(r["source"]) for r in train) / len(train)
        avg_summary = sum(len(r["summary"] or "") for r in train) / len(train)
        logger.info("训练集平均长度：原文 %.0f 字 / 摘要 %.0f 字", avg_source, avg_summary)


if __name__ == "__main__":
    main()
