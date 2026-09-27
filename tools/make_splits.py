"""把 CAIL2020 全量数据切成 train / val / test，按案由分层。

为什么要分层而不是纯随机：
  不同案由的摘要难度差异很大（继承纠纷涉及多名当事人和遗产明细，
  而简单的借款合同只有本金和利息）。纯随机切分有可能让某个案由
  在验证集里几乎不出现，导致指标受案由分布波动影响。
  分层能保证三个集合的案由分布一致。

输出（规范化后的三字段格式，和 data/processed/ 一致）：
    data/splits/train.jsonl   80%
    data/splits/val.jsonl     10%
    data/splits/test.jsonl    10%

用法：
    python tools/make_splits.py
    python tools/make_splits.py --ratios 0.9 0.05 0.05
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.data.prepare import to_record          # noqa: E402
from sfzy.data.schema import read_jsonl          # noqa: E402
from sfzy.utils.io import ensure_dir, write_jsonl  # noqa: E402

# 案由白名单（按长度降序匹配，避免"合同纠纷"抢先匹配掉"买卖合同纠纷"）。
# 用白名单而不是 `(.{1,10}纠纷)` 这类宽正则，是因为后者会把当事人名
# 一起吞进去 —— 实测会抽出 9000 多个"案由"，分层完全失效。
CAUSE_LIST = [
    "机动车交通事故责任纠纷", "生命权、健康权、身体权纠纷", "建设工程施工合同纠纷",
    "商品房销售合同纠纷", "房屋买卖合同纠纷", "物业服务合同纠纷", "装饰装修合同纠纷",
    "离婚后财产纠纷", "追索劳动报酬纠纷", "金融借款合同纠纷", "房屋租赁合同纠纷",
    "确认合同效力纠纷", "所有权确认纠纷", "股权转让纠纷", "保证合同纠纷",
    "民间借贷纠纷", "借款合同纠纷", "租赁合同纠纷", "劳动合同纠纷", "买卖合同纠纷",
    "承揽合同纠纷", "运输合同纠纷", "保管合同纠纷", "委托合同纠纷", "居间合同纠纷",
    "不当得利纠纷", "无因管理纠纷", "返还原物纠纷", "排除妨害纠纷",
    "离婚纠纷", "继承纠纷", "抚养费纠纷", "赡养纠纷", "分家析产纠纷",
    "侵权责任纠纷", "财产损害赔偿纠纷", "合伙协议纠纷",
    "合同纠纷", "服务合同纠纷", "供用热力合同纠纷", "供用电合同纠纷",
]
CAUSE_PATTERNS = sorted(CAUSE_LIST, key=len, reverse=True)

RARE_THRESHOLD = 20      # 少于这么多条的案由合并成"其他"，避免分层时出现空组


def extract_cause(source: str) -> str:
    """从文书开头按白名单抽案由。抽不到就归到"其他/未知"。

    只看前 150 字 —— 判决书标题和开头几行就写着案由，往后找反而会
    命中正文里提到的其他案由（比如引用的类案）。
    """
    head = source[:150]
    for cause in CAUSE_PATTERNS:
        if cause in head:
            return cause
    return "未知"


def main() -> None:
    parser = argparse.ArgumentParser(description="按案由分层切分数据集")
    parser.add_argument("--raw", default="data/raw/sfzy_cail.json")
    parser.add_argument("--outdir", default="data/splits")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    outdir = ensure_dir(ROOT / args.outdir if not Path(args.outdir).is_absolute() else args.outdir)

    print(f"读取 {raw_path} ...")
    docs = list(read_jsonl(raw_path))
    print(f"  共 {len(docs)} 条")

    # 抽取案由并分组
    buckets: dict[str, list] = defaultdict(list)
    for doc in docs:
        source = doc.joined("")
        buckets[extract_cause(source)].append(doc)

    # 稀有案由合并，否则分层时小类别可能被切出空集
    merged: dict[str, list] = defaultdict(list)
    for cause, group in buckets.items():
        key = cause if len(group) >= RARE_THRESHOLD else "其他"
        merged[key].extend(group)

    print(f"  案由类别 {len(buckets)} 个（>= {RARE_THRESHOLD} 条的保留，其余合并为'其他'）"
          f" → {len(merged)} 组")

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for cause, group in sorted(merged.items()):
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * args.ratios[0])
        n_val = int(n * args.ratios[1])
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])

    # 每个集合内部再打乱一次，避免同一案由扎堆
    for name in splits:
        rng.shuffle(splits[name])

    print()
    total = sum(len(v) for v in splits.values())
    for name in ("train", "val", "test"):
        recs = [to_record(d, "") for d in splits[name]]
        n = write_jsonl(outdir / f"{name}.jsonl", recs)
        print(f"  {name:<6} {n:>6} ({n/total:5.1%})  → {outdir / f'{name}.jsonl'}")

    # 校验：三集合 id 不重叠
    ids = {name: {d.id for d in docs_} for name, docs_ in splits.items()}
    assert not (ids["train"] & ids["val"]), "train / val 有重叠"
    assert not (ids["train"] & ids["test"]), "train / test 有重叠"
    assert not (ids["val"] & ids["test"]), "val / test 有重叠"
    print(f"\n  校验通过：三个集合互不重叠，合计 {total} 条")

    # 案由分布一致性检查
    print("\n  案由分布（Top 6）：")
    print(f"    {'案由':<16}{'train':>9}{'val':>9}{'test':>9}")
    for cause, _ in Counter(extract_cause(d.joined('')) for d in docs).most_common(6):
        row = []
        for name in ("train", "val", "test"):
            grp = [d for d in splits[name] if extract_cause(d.joined("")) == cause]
            row.append(f"{len(grp)/max(len(splits[name]),1):8.1%}")
        print(f"    {cause:<16}{row[0]:>9}{row[1]:>9}{row[2]:>9}")


if __name__ == "__main__":
    main()
