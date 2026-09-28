"""筛选 GRPO 的 prompt 池。

============================ 为什么必须筛 ============================
实测（335 条验证集抽样）：

    参考摘要里的事实个数
      0 个   60.0%   ← 六成的参考摘要没有任何可提取的事实
      1-2 个 23.9%
      >=3 个 16.1%

**事实覆盖这个奖励只对 40% 的样本有信号。** 剩下 60% 的参考摘要里根本
没有金额/日期可提取——不是提取器的问题，是标注风格（比如"判令被告支付
拖欠的租金"压根不写金额）。

这本身对 GRPO 无害：组内所有采样共享同一个参考，"有没有事实"是组内
常数，组内归一化时会抵消。**但随机抽 prompt 的话，60% 的算力花在事实项
恒为 0 的组上**，那些组的梯度只由 ROUGE 决定——等于白跑。

============================ 怎么筛 ============================
按"参考里的事实个数"降序排，取前 N 条。默认只保留含金额的样本
（金额占 35.2%，是三种事实里信号最强的）。

用法：
    python tools/select_rl_prompts.py --n 1000
    python tools/select_rl_prompts.py --n 1000 --kinds money date id
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.rl.reward import extract_facts     # noqa: E402
from sfzy.utils.io import ensure_dir, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="筛选 GRPO 的 prompt 池")
    parser.add_argument("--input", default="data/splits/train.jsonl")
    parser.add_argument("--out", default="data/splits/rl_prompts.jsonl")
    parser.add_argument("--n", type=int, default=1000,
                        help="选多少条；0 表示所有含事实的")
    parser.add_argument("--kinds", nargs="+", default=["money"],
                        help="按哪些事实类型筛。默认只看金额（信号最强、占比最高 35.2%%）")
    parser.add_argument("--shuffle", action="store_true",
                        help="选完之后打乱顺序（默认按事实个数降序，先练难的）")
    parser.add_argument("--triples", default=None,
                        help="SFT 三元组 jsonl（含 id/output）。给了就把 SFT 输出挂到 "
                             "每个 prompt 的 sft_output 字段上，训练时用来做基线锚")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    scored = []
    for r in records:
        facts = extract_facts(r["summary"], kinds=args.kinds)
        if facts:
            scored.append((len(facts), r))

    print(f"输入 {len(records)} 条")
    print(f"含 {args.kinds} 类事实的 {len(scored)} 条 = {len(scored)/max(len(records),1):.1%}")
    print(f"  事实个数分布: ", end="")
    from collections import Counter
    dist = Counter(c for c, _ in scored)
    print("  ".join(f"{k}个:{v}" for k, v in sorted(dist.items())[:6]))

    # 降序：先选事实最密集的，那是模型最可能失败的样本
    scored.sort(key=lambda x: -x[0])
    selected = [r for _, r in scored[: args.n or None]]

    if args.shuffle:
        random.Random(args.seed).shuffle(selected)

    # ---- 挂上 SFT 输出（可选的基线锚要用）----
    if args.triples:
        tpath = ROOT / args.triples if not Path(args.triples).is_absolute() else Path(args.triples)
        by_id: dict = {}
        for line in open(tpath, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                by_id[r["id"]] = r["output"]
        hit = 0
        for r in selected:
            if r["id"] in by_id:
                r["sft_output"] = by_id[r["id"]]
                hit += 1
        print(f"\n从 {tpath.name} 挂上 SFT 输出：{hit}/{len(selected)} 条")
        if hit < len(selected):
            print("  缺的那些在训练时不会被锚过滤（等价于不启用），不会报错")
            print("  ⚠ 常见原因：三元组是 **val** 上的，而 prompt 池来自 **train**，"
                  "两边 id 不重叠。\n"
                  "    要用基线锚就得给 prompt 池这批文书本身生成三元组：\n"
                  "      python scripts/generate_triples.py --config <cfg> "
                  "--adapter <sft.pt> \\\n"
                  f"          --input {args.out} "
                  "--out data/triples/sft_rlpool.jsonl\n"
                  "    （必须用 --input 指池子文件；--split train --limit 拿到的是"
                  " train 的前 N 条，和池子的 id 对不上）")

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    ensure_dir(out_path.parent)
    n = write_jsonl(out_path, selected)

    total_facts = sum(len(extract_facts(r["summary"], kinds=args.kinds)) for r in selected)
    print(f"\n选出 {n} 条 → {out_path}")
    print(f"  平均每条 {total_facts/max(n,1):.2f} 个事实"
          f"（全集平均 {sum(len(extract_facts(r['summary'], kinds=args.kinds)) for r in records)/max(len(records),1):.2f}）")
    if args.n and len(scored) < args.n:
        print(f"  注意：符合条件的只有 {len(scored)} 条，少于请求的 {args.n}")


if __name__ == "__main__":
    main()
