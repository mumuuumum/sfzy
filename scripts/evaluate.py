"""评测预测结果，按 CAIL2020 官方口径计算 ROUGE。

用法：
    python scripts/evaluate.py --pred outputs/pred.json --ref data/processed/dev.jsonl
    python scripts/evaluate.py --pred outputs/pred.json --ref data/processed/dev.jsonl --mode jieba
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfzy.eval.rouge import score_corpus       # noqa: E402
from sfzy.utils.io import read_jsonl           # noqa: E402
from sfzy.utils.logging import get_logger      # noqa: E402

logger = get_logger("evaluate")


def load_predictions(path: Path) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    for record in read_jsonl(path):
        if "id" not in record or "summary" not in record:
            raise ValueError(f"{path} 中的记录必须包含 id 与 summary 字段")
        preds[str(record["id"])] = record["summary"]
    return preds


def load_references(path: Path) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    for record in read_jsonl(path):
        if "summary" not in record:
            raise ValueError(f"{path} 中的记录必须包含 summary 字段（是否为测试集口径？）")
        refs[str(record["id"])] = record["summary"]
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(description="ROUGE 评测（官方口径）")
    parser.add_argument("--pred", required=True, help="预测文件，JSONL，含 id 与 summary")
    parser.add_argument("--ref", required=True, help="参考文件，JSONL，含 id 与 summary")
    parser.add_argument("--mode", default="char", choices=["char", "jieba"], help="分词方式")
    parser.add_argument("--save", default=None, help="把逐样本得分保存为 JSON 的路径")
    args = parser.parse_args()

    preds = load_predictions(Path(args.pred))
    refs = load_references(Path(args.ref))

    missing = sorted(set(refs) - set(preds))
    if missing:
        logger.warning("有 %d 条参考样本没有对应预测，示例：%s", len(missing), missing[:3])

    result = score_corpus(preds, refs, mode=args.mode)

    print()
    print(f"样本数：{result.get('num_samples', len(refs))}    分词：{args.mode}")
    print("-" * 46)
    print(f"{'指标':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for name in ("rouge-1", "rouge-2", "rouge-l"):
        print(
            f"{name:<12}"
            f"{result[f'{name}-p']:>10.4f}"
            f"{result[f'{name}-r']:>10.4f}"
            f"{result[f'{name}-f']:>10.4f}"
        )
    print("-" * 46)
    print(f"官方口径总分（0.2*R1 + 0.4*R2 + 0.4*RL）：{result['overall']:.4f}")

    if args.save:
        import json

        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("已保存到 %s", args.save)


if __name__ == "__main__":
    main()
