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


def _pearson(xs, ys) -> float:
    """皮尔逊相关。这一节要看的是"裁判分和 ROUGE 是不是同一个信号"，
    相关高说明新指标没带来新信息 —— 那就不值一个 7B 裁判的成本。"""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="观察模型输出的诊断报告")
    parser.add_argument("--input", required=True, nargs="+",
                        help="一个或多个三元组 jsonl，支持通配符。"
                             "多卡推理会产出多个分片文件，这里直接全部读进来，"
                             "不用手动 cat 合并（手动合并容易漏或重复）")
    parser.add_argument("--rouge-mode", default="jieba", choices=["char", "jieba"])
    parser.add_argument("--judge-file", nargs="*", default=None,
                        help="裁判分 jsonl（scripts/judge_score.py 的产物）。"
                             "给了就多打一节语义指标；多个文件按 id 合并")
    parser.add_argument("--top-examples", type=int, default=3)
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.input:
        p = Path(pattern) if Path(pattern).is_absolute() else ROOT / pattern
        matched = sorted(p.parent.glob(p.name))
        if not matched:
            raise FileNotFoundError(f"没有匹配到文件：{pattern}")
        paths.extend(matched)

    seen, recs = set(), []
    for p in paths:
        for r in load(p):
            if r["id"] not in seen:       # 分片之间有重叠时去重，避免重复计数
                seen.add(r["id"])
                recs.append(r)
    n = len(recs)
    print(f"\n{'='*72}\n输入 {len(paths)} 个文件，去重后 {n} 条")
    for p in paths:
        print(f"  {p.relative_to(ROOT) if str(p).startswith(str(ROOT)) else p}")
    print('='*72)

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

    # ---------------- 3.5 语义指标（裁判）----------------
    # 为什么单独一节：语义分和 ROUGE 是两套口径，混在一起算平均没有意义。
    # 判据是"语义涨、ROUGE 不跌"，所以必须并排看。
    judge_by_id: dict = {}
    for pattern in (args.judge_file or []):
        jp = Path(pattern) if Path(pattern).is_absolute() else ROOT / pattern
        for jpath in sorted(jp.parent.glob(jp.name)):
            for line in open(jpath, encoding="utf-8"):
                line = line.strip()
                if line:
                    j = json.loads(line)
                    if j.get("judge_score") is not None:
                        judge_by_id[str(j["id"])] = float(j["judge_score"])
    if judge_by_id:
        paired = [(judge_by_id[str(r["id"])], scores[i]["rouge-l-f"], covs[i], ratios[i])
                  for i, r in enumerate(recs) if str(r["id"]) in judge_by_id]
        js = [p[0] for p in paired]
        print("\n【3.5】语义指标（LLM 裁判）")
        print(f"  覆盖 {len(paired)}/{n} 条")
        print(f"  裁判分: 均值 {st.mean(js):.2f}  中位数 {st.median(js):.2f}  "
              f"标准差 {st.pstdev(js):.2f}  P10 {q(js,0.1):.1f}  P90 {q(js,0.9):.1f}")
        distinct = len({round(x, 1) for x in js})
        note = ""
        if distinct < 0.3 * len(js):
            note = "  ← 取值太集中，裁判在给模板分：检查 rubric 的档位描述和温度"
        print(f"  不同取值 {distinct} 个 / {len(js)} 条{note}")
        corr_rl = _pearson([p[0] for p in paired], [p[1] for p in paired])
        corr_cov = _pearson([p[0] for p in paired], [p[2] for p in paired])
        corr_len = _pearson([p[0] for p in paired], [p[3] for p in paired])
        print(f"  与 ROUGE-L 相关 {corr_rl:+.3f}  与事实覆盖 {corr_cov:+.3f}  "
              f"与长度比 {corr_len:+.3f}")
        print("  ← 和 ROUGE 相关太高（>0.9）说明它没带来新信息；"
              "和长度比相关太高说明它在奖励写长")

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
