"""观察模型的输出：一份可以直接读的诊断报告。

============================ 看什么、为什么 ============================
按重要性排序，前两条是核心：

1. **漏事实率 ★** —— 旧模型的头号问题。它漏掉的数字里 98.5% 明确写在原文里，
   训练集上 28.8% 的样本至少漏一个，代价是 0.073 的分数。**这是 GRPO 要改的东西，
   所以必须先量出来。**

2. **ROUGE 官方总分** —— 和评测口径一致的主指标。

3. **长度分布** —— 过长/过短都是信号，而且长度是 ROUGE hacking 的第一表现。

4. **格式坍缩度** —— 比较模型和参考的"开头模式熵"。旧模型 92 种 → 11 种，
   97.8% 的输出以同一短语开头。熵跌得多说明模型退化成模板。

5. **分组差异** —— 按"参考里的事实个数"分组看。旧模型在信息密集的样本上
   崩得最厉害（参考含 ≥3 个数字的样本里 56% 漏了 3 个以上）。
   只看总数会把这个掩盖掉。

用法：
    python scripts/analyze_outputs.py --input data/triples/sft_val.jsonl
    python scripts/analyze_outputs.py --input data/triples/sft_val.jsonl --split-by-density
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.eval.rouge import score_pair                  # noqa: E402
from sfzy.rl.reward import (                            # noqa: E402
    check_gate,
    DEFAULT_REWARD_CFG,
    extract_facts,
    fact_coverage,
    fact_precision,
)


def load(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def opening_entropy(texts, n: int = 8) -> float:
    """开头 n 字的模式熵。熵越低说明模式越集中（格式坍缩）。"""
    counter = Counter(t.strip()[:n] for t in texts)
    total = sum(counter.values())
    return -sum((c / total) * math.log(c / total) for c in counter.values())


def q(values, p):
    a = sorted(values)
    return a[min(int(len(a) * p), len(a) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="观察模型输出的诊断报告")
    parser.add_argument("--input", required=True, help="三元组 jsonl")
    parser.add_argument("--rouge-mode", default="jieba", choices=["char", "jieba"])
    parser.add_argument("--top-examples", type=int, default=3)
    args = parser.parse_args()

    recs = load(ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input))
    n = len(recs)
    print(f"\n{'='*72}\n输入: {args.input}\n样本数: {n}\n{'='*72}")

    # ---------------- 1. ROUGE ----------------
    scores = [score_pair(r["output"].strip(), r["reference"].strip(), mode=args.rouge_mode)
              for r in recs]
    r1 = st.mean(s["rouge-1-f"] for s in scores)
    r2 = st.mean(s["rouge-2-f"] for s in scores)
    rl = st.mean(s["rouge-l-f"] for s in scores)
    official = 0.2 * r1 + 0.4 * r2 + 0.4 * rl
    print("\n【1】ROUGE（官方口径 0.2/0.4/0.4）")
    print(f"  ROUGE-1 {r1:.4f}   ROUGE-2 {r2:.4f}   ROUGE-L {rl:.4f}")
    print(f"  ★ 官方总分 {official:.4f}")
    print(f"  precision < recall? "
          f"R1 {st.mean(s['rouge-1-p'] for s in scores):.4f} vs {st.mean(s['rouge-1-r'] for s in scores):.4f}")

    # ---------------- 2. 事实（核心）----------------
    covs, precs, missed_total, ref_total = [], [], 0, 0
    per_sample_missed = []
    for r in recs:
        cov, tot, miss = fact_coverage(r["output"], r["reference"])
        covs.append(cov)
        missed_total += miss
        ref_total += tot
        per_sample_missed.append(miss)
        if "source" in r:
            precs.append(fact_precision(r["output"], r["source"]))

    has_facts = [i for i, r in enumerate(recs) if extract_facts(r["reference"])]
    print("\n【2】事实覆盖 ★（主要痛点）")
    print(f"  参考含事实的样本: {len(has_facts)}/{n} = {len(has_facts)/n:.1%}"
          f"   ← 事实信号只对这部分有效")
    if has_facts:
        sub_cov = [covs[i] for i in has_facts]
        sub_miss = [per_sample_missed[i] for i in has_facts]
        print(f"  这些样本的平均覆盖率: {st.mean(sub_cov):.4f}")
        print(f"  完全没漏的样本占比:   {sum(1 for i in has_facts if per_sample_missed[i]==0)/len(has_facts):.1%}")
        print(f"  漏 3 个以上的占比:    {sum(1 for i in has_facts if per_sample_missed[i]>=3)/len(has_facts):.1%}")
        print(f"  数字级漏掉比例:       {sum(sub_miss)/max(sum(len(extract_facts(recs[i]['reference'])) for i in has_facts),1):.1%}")
    if precs:
        print(f"  事实精确率（输出的事实有多少在原文里）: {st.mean(precs):.4f}")

    # ---------------- 3. 长度 ----------------
    ratios = [(len(r["output"].strip()) / max(len(r["reference"].strip()), 1)) for r in recs]
    print("\n【3】长度分布")
    print(f"  参考中位数 {st.median([len(r['reference'].strip()) for r in recs]):.0f} 字"
          f"   输出中位数 {st.median([len(r['output'].strip()) for r in recs]):.0f} 字")
    print(f"  比值: 中位数 {st.median(ratios):.2f}  P90 {q(ratios,0.9):.2f}")
    print(f"  长 30%+ 的占 {sum(1 for x in ratios if x>1.3)/n:.1%}"
          f"   短 30%+ 的占 {sum(1 for x in ratios if x<0.7)/n:.1%}")

    # ---------------- 4. 格式坍缩 ----------------
    out_ent = opening_entropy([r["output"] for r in recs])
    ref_ent = opening_entropy([r["reference"] for r in recs])
    out_modes = len(set(r["output"].strip()[:8] for r in recs))
    ref_modes = len(set(r["reference"].strip()[:8] for r in recs))
    print("\n【4】格式坍缩度（开头 8 字的模式熵）")
    print(f"  参考 {ref_ent:.3f}（{ref_modes} 种模式）   输出 {out_ent:.3f}（{out_modes} 种模式）")
    if ref_ent > 0:
        print(f"  熵的比值 {out_ent/ref_ent:.2f}   ← 明显小于 1 说明模型在收窄表达")

    # ---------------- 5. 门控（顺带看生成配置是否合理）----------------
    reasons = Counter()
    for r in recs:
        reason = check_gate(r["output"], r["reference"], DEFAULT_REWARD_CFG["gate"])
        if reason:
            reasons[reason] += 1
    print("\n【5】奖励门控会被拦下多少（衡量生成配置是否合理）")
    print(f"  会被拦下 {sum(reasons.values())}/{n} = {sum(reasons.values())/n:.1%}")
    for reason, c in reasons.most_common(5):
        print(f"    {reason:<24} {c}")

    # ---------------- 6. 分组：信息密度 ----------------
    print("\n【6】按「参考里的事实个数」分组（旧模型在这里崩得最厉害）")
    print(f"  {'组':<16}{'样本数':>8}{'ROUGE-L':>10}{'覆盖率':>10}{'长度比':>10}")
    for lo, hi, name in [(0, 1, "0 个"), (1, 3, "1-2 个"), (3, 99, "3 个以上")]:
        idx = [i for i in range(n) if lo <= len(extract_facts(recs[i]["reference"])) < hi]
        if not idx:
            continue
        print(f"  {name:<16}{len(idx):>8}"
              f"{st.mean(scores[i]['rouge-l-f'] for i in idx):>10.4f}"
              f"{st.mean(covs[i] for i in idx):>10.4f}"
              f"{st.mean(ratios[i] for i in idx):>10.2f}")

    # ---------------- 7. 最差样本 ----------------
    if args.top_examples:
        order = sorted(range(n), key=lambda i: 0.2 * scores[i]["rouge-1-f"]
                       + 0.4 * scores[i]["rouge-2-f"] + 0.4 * scores[i]["rouge-l-f"])
        print(f"\n【7】最差的 {args.top_examples} 条")
        for i in order[: args.top_examples]:
            r = recs[i]
            s = 0.2 * scores[i]["rouge-1-f"] + 0.4 * scores[i]["rouge-2-f"] + 0.4 * scores[i]["rouge-l-f"]
            miss = sorted(extract_facts(r["reference"]) - extract_facts(r["output"]))
            print(f"\n  [总分 {s:.3f}] 参考 {len(r['reference'])}字 / 输出 {len(r['output'].strip())}字")
            print(f"    参考: {r['reference'][:100]}")
            print(f"    输出: {r['output'].strip()[:100]}")
            if miss:
                print(f"    漏掉的事实: {miss[:6]}")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
