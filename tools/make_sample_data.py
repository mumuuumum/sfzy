"""生成开发用的小样本数据集，避免每次调试都加载 126MB 的全量数据。

从 `data/raw/sfzy_cail.json` 抽样，按比例切成 train / dev，
并额外产出一份**测试集口径**的 test.jsonl（去掉 summary 与 label），
用来验证"模型只能看到 id 和句子"这条链路。

用法：
    python tools/make_sample_data.py --n 200
    python tools/make_sample_data.py --n 200 --synthetic   # 没有全量数据时用
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "sfzy_cail.json"
DEFAULT_OUTDIR = ROOT / "data" / "raw" / "sample"

# 合成数据用的占位内容，只在 --synthetic 时使用
_SYNTH_SENTENCES = [
    "原告张三诉被告李四民间借贷纠纷一案，本院于2020年3月1日立案受理。",
    "原告诉称：被告向其借款人民币50000元，约定于2019年12月31日前归还，到期后经多次催要未果。",
    "被告辩称：借款属实，但已于2019年11月归还30000元，剩余部分因资金困难暂无力偿还。",
    "经审理查明：原被告之间的借贷关系合法有效，有借条及转账记录为证。",
    "本院认为，被告应当依照约定履行还款义务。",
    "依照《中华人民共和国合同法》第一百零七条、第二百零六条之规定，判决如下。",
    "一、被告李四于本判决生效之日起十日内归还原告张三借款人民币20000元。",
    "审判员　王五",
    "二○二○年六月十五日",
]


def synth_record(index: int, rng: random.Random) -> dict:
    """构造一条格式与 CAIL2020 一致的合成样本。"""
    k = rng.randint(3, len(_SYNTH_SENTENCES))
    sentences = rng.sample(_SYNTH_SENTENCES, k)
    return {
        "id": f"synthetic-{index:05d}",
        "summary": "原被告民间借贷纠纷。法院认定借贷关系有效，判令被告归还剩余借款。",
        "text": [{"sentence": s, "label": 0} for s in sentences],
    }


def load_records(source: Path, n: int, synthetic: bool, rng: random.Random) -> list[dict]:
    if synthetic or not source.exists():
        if not synthetic:
            print(f"[warn] 未找到 {source}，回退到合成数据")
        return [synth_record(i, rng) for i in range(n)]

    total = sum(1 for _ in open(source, "r", encoding="utf-8"))
    if n >= total:
        print(f"[warn] 请求 {n} 条，但源文件只有 {total} 条，将全部使用")
        n = total

    # 水库抽样：不把 126MB 全读进内存
    reservoir: list[str] = []
    with open(source, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = line
    return [json.loads(line) for line in reservoir]


def to_test_style(record: dict) -> dict:
    """转换成测试集口径：只保留 id 与句子，去掉 summary 与 label。"""
    return {
        "id": record["id"],
        "text": [{"sentence": item["sentence"]} for item in record["text"]],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成开发用小样本数据集")
    parser.add_argument("--n", type=int, default=200, help="抽样的样本总数")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="不读源文件，直接造合成数据")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_records(args.source, args.n, args.synthetic, rng)
    rng.shuffle(records)

    n_dev = max(1, int(len(records) * args.dev_ratio))
    dev, train = records[:n_dev], records[n_dev:]

    write_jsonl(args.outdir / "train.jsonl", train)
    write_jsonl(args.outdir / "dev.jsonl", dev)

    # 测试集口径的样本必须**放在 outdir 之外**。
    # 因为 scripts/prepare_data.py 读目录时会把 *.jsonl 全部收进去，
    # 而测试集口径没有 summary —— 混进训练集会让 collator 拿到 answer=None，
    # 报错位置离真正的原因很远。
    inference_path = args.outdir.parent / f"{args.outdir.name}_test_input.jsonl"
    write_jsonl(inference_path, [to_test_style(r) for r in dev])

    print(f"已生成到 {args.outdir}")
    print(f"  train.jsonl {len(train):>5} 条")
    print(f"  dev.jsonl   {len(dev):>5} 条")
    print(f"  (测试集口径的对照样本单独放在 {inference_path}，共 {len(dev)} 条)")


if __name__ == "__main__":
    main()
