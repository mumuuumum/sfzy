"""成对比较两次 run（SFT vs RL），按**预注册判据**给出结论。

============================ 为什么不能只比平均值 ============================
两次 run 的样本必须按 id 对齐、**成对**比较：

  * 只看平均值，样本集不同就会得出假结论（B 少跑了几十条长文本，
    平均值自己就动了）
  * 只看总平均会掩盖"一组涨一组跌" —— 参考含事实的 513 条和没事实的 827 条
    是两个不同的任务，混在一起平均出来的"持平"可能是两组各差 20 分的抵消

所以输出三块：整体成对、**参考含事实**、**参考无事实**。

============================ 判据是提前定好的 ============================
判据写在下面的 CRITERIA 里，**动手之前就定好**，跑完只读结果不调整阈值。
事后改阈值是自欺欺人的标准做法。

============================ 用法 ============================
    python tools/compare_runs.py \
        --a data/triples/sft_val_shard0of2.jsonl \
        --a-judge data/judge/sft_val.judge.jsonl --a-name SFT \
        --b data/triples/grpo_a4_val.jsonl \
        --b-judge data/judge/grpo_a4_val.judge.jsonl --b-name A4

    # 只看 ROUGE 和事实（还没跑裁判）就把 --*-judge 去掉
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.eval.rouge import score_pair          # noqa: E402
from sfzy.rl.reward import extract_facts, fact_coverage, fact_precision, fact_score  # noqa: E402

# ---------------------------------------------------------------------------
# 预注册判据：目标值来自 docs/semantic_metrics.md 第 4 节。
# 方向 '+' 表示越大越好，'-' 表示越小越好，'=' 表示不低于基线即可。
# ---------------------------------------------------------------------------
CRITERIA = [
    ("官方 ROUGE 总分", "official", "=", None, "掉超过 0.01 就是退化"),
    ("事实覆盖率", "fact_coverage", "+", 0.45, "只看参考含事实的子集"),
    ("数字级漏掉比例", "missed_ratio", "-", 0.55, "越低越好"),
    ("事实精确率", "fact_precision", "+", 0.99, None),
    ("长度比中位数", "len_ratio", "-", 1.15, None),
    # 六要素事实一致性（tools/score_fact_judge.py 产出）。和上面的"事实覆盖率"
    # 不是一回事：覆盖率是**规则口径**（金额/日期有没有抄到，只看参考），
    # 这一项是**裁判口径**（候选说的能不能被原文支持）。两个都看才分得清
    # "漏写"和"写错"。
    ("事实一致性（Judge）", "fact_judge", "=", None, "不跌即守，要的是涨"),
    ("裁判分（汇报用）", "judge", "=", None, "不跌即算守住，要的是涨"),
]


def load_jsonl(pattern: str, key: Optional[str] = None) -> Dict[str, dict]:
    """读 jsonl（支持通配），按 id 去重。key 用于裁判文件（取 judge_score）。"""
    out: Dict[str, dict] = {}
    # 绝对路径要单独处理：Path.glob 不接受绝对模式
    base = Path(pattern)
    paths = sorted(base.parent.glob(base.name)) if base.is_absolute() else sorted(ROOT.glob(pattern))
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r["id"])
            if key is None:
                out[rid] = r
            elif r.get(key) is not None:
                out[rid] = {"score": float(r[key])}
    return out


def metrics_for(rec: dict) -> Dict[str, float]:
    s = score_pair(rec["output"].strip(), rec["reference"].strip(), mode="jieba")
    ref_facts = extract_facts(rec["reference"])
    cov, n_ref, missed = fact_coverage(rec["output"], rec["reference"])
    m = {
        "rouge-1": s["rouge-1-f"],
        "rouge-2": s["rouge-2-f"],
        "rouge-l": s["rouge-l-f"],
        "official": 0.2 * s["rouge-1-f"] + 0.4 * s["rouge-2-f"] + 0.4 * s["rouge-l-f"],
        "fact_score": fact_score(rec["output"], rec["reference"]),
        "len_ratio": len(rec["output"].strip()) / max(len(rec["reference"].strip()), 1),
        "n_ref_facts": n_ref,
        "has_facts": 1.0 if ref_facts else 0.0,
    }
    if "source" in rec:
        m["fact_precision"] = fact_precision(rec["output"], rec["source"])
    # 覆盖率/漏掉比例只在"参考含事实"的样本上有定义，其余样本记为 None
    m["fact_coverage"] = cov if n_ref else None
    m["missed_ratio"] = (missed / n_ref) if n_ref else None
    return m


def paired_t(diffs: List[float]) -> Tuple[float, float]:
    """配对 t 统计量。不引 scipy：n=1340 时看 |t|>2 就够判断方向了。"""
    n = len(diffs)
    if n < 2:
        return 0.0, 0.0
    mean = st.mean(diffs)
    sd = st.pstdev(diffs)
    return mean, (mean / (sd / math.sqrt(n)) if sd > 0 else 0.0)


def fmt(x: Optional[float], nd: int = 4) -> str:
    return "  n/a " if x is None else f"{x:.{nd}f}"


def section(title: str, a: Dict[str, dict], b: Dict[str, dict],
            ids: List[str], judge_a: Dict, judge_b: Dict,
            fact_a: Optional[Dict] = None, fact_b: Optional[Dict] = None) -> None:
    print(f"\n{'='*78}\n{title}（{len(ids)} 条）\n{'='*78}")
    if not ids:
        print("  没有可比样本")
        return

    rows = []
    for rid in ids:
        ma = metrics_for(a[rid])
        mb = metrics_for(b[rid])
        if judge_a and judge_b and rid in judge_a and rid in judge_b:
            ma["judge"] = judge_a[rid]["score"]
            mb["judge"] = judge_b[rid]["score"]
        if fact_a and fact_b and rid in fact_a and rid in fact_b:
            ma["fact_judge"] = fact_a[rid]["score"]
            mb["fact_judge"] = fact_b[rid]["score"]
        rows.append((ma, mb))

    print(f"{'指标':<18}{'A':>10}{'B':>10}{'Δ':>10}{'改进占比':>10}{'t':>8}")
    print("-" * 66)
    for name, key, direction, target, _ in CRITERIA:
        pairs = [(m1[key], m2[key]) for m1, m2 in rows
                 if m1.get(key) is not None and m2.get(key) is not None]
        if not pairs:
            continue
        av = st.mean(p[0] for p in pairs)
        bv = st.mean(p[1] for p in pairs)
        diffs = [p[1] - p[0] for p in pairs]
        _, t = paired_t(diffs)
        better = sum(1 for d in diffs if (d > 0 if direction != "-" else d < 0))
        print(f"{name:<18}{av:>10.4f}{bv:>10.4f}{bv-av:>+10.4f}"
              f"{better/len(diffs):>9.1%}{t:>8.2f}")
    # 这几个不进判据表，但少了会误判
    for extra, label in [("rouge-1", "ROUGE-1"), ("rouge-2", "ROUGE-2"),
                         ("rouge-l", "ROUGE-L"), ("fact_score", "事实 F1")]:
        pairs = [(m1[extra], m2[extra]) for m1, m2 in rows]
        av = st.mean(p[0] for p in pairs)
        bv = st.mean(p[1] for p in pairs)
        diffs = [p[1] - p[0] for p in pairs]
        _, t = paired_t(diffs)
        better = sum(1 for d in diffs if d > 0)
        print(f"{label:<18}{av:>10.4f}{bv:>10.4f}{bv-av:>+10.4f}"
              f"{better/len(diffs):>9.1%}{t:>8.2f}")


def verdicts(rows_all: List[Tuple[dict, dict]], name_a: str, name_b: str) -> None:
    print(f"\n{'='*78}\n判据（B={name_b} 相对 A={name_a}）\n{'='*78}")
    for label, key, direction, target, note in CRITERIA:
        pairs = [(m1[key], m2[key]) for m1, m2 in rows_all
                 if m1.get(key) is not None and m2.get(key) is not None]
        if not pairs:
            print(f"  [跳过] {label:<16}（没有可比数据）")
            continue
        av = st.mean(p[0] for p in pairs)
        bv = st.mean(p[1] for p in pairs)
        if direction == "=":
            ok = bv >= av - 0.01
            rule = "不低于基线（允许 -0.01）"
        elif direction == "+":
            ok = bv >= (target if target is not None else av)
            rule = f"≥ {target}"
        else:
            ok = bv <= (target if target is not None else av)
            rule = f"≤ {target}"
        mark = "✓ 达标" if ok else "✗ 未达标"
        print(f"  {mark}  {label:<16} {av:.4f} → {bv:.4f}   要求 {rule}"
              + (f"   （{note}）" if note else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT 与 RL 的成对对比")
    ap.add_argument("--a", required=True, help="基线三元组 jsonl（glob）")
    ap.add_argument("--b", required=True, help="对比三元组 jsonl（glob）")
    ap.add_argument("--a-judge", default=None, help="基线裁判分 jsonl")
    ap.add_argument("--b-judge", default=None, help="对比裁判分 jsonl")
    ap.add_argument("--a-fact", default=None,
                    help="基线的事实一致性 jsonl（tools/score_fact_judge.py 产出）")
    ap.add_argument("--b-fact", default=None, help="对比的事实一致性 jsonl")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    args = ap.parse_args()

    a = load_jsonl(args.a)
    b = load_jsonl(args.b)
    judge_a = load_jsonl(args.a_judge, key="judge_score") if args.a_judge else {}
    judge_b = load_jsonl(args.b_judge, key="judge_score") if args.b_judge else {}
    fact_a = load_jsonl(args.a_fact, key="fact_reward") if args.a_fact else {}
    fact_b = load_jsonl(args.b_fact, key="fact_reward") if args.b_fact else {}

    common = sorted(set(a) & set(b))
    print(f"{args.a_name}: {len(a)} 条   {args.b_name}: {len(b)} 条   "
          f"交集 {len(common)} 条")
    if len(common) < 0.9 * min(len(a), len(b)):
        print("  ⚠ 交集明显小于两边样本数 —— 两次 run 用的不是同一批样本，"
              "下面的成对结论只对交集成立")
    if judge_a or judge_b:
        n_j = sum(1 for r in common if r in judge_a and r in judge_b)
        print(f"  两边都有裁判分的 {n_j} 条")

    has = [r for r in common if extract_facts(a[r]["reference"])]
    no = [r for r in common if not extract_facts(a[r]["reference"])]
    rows_all = [(metrics_for(a[r]), metrics_for(b[r])) for r in common]
    # 判据表也要能看到事实一致性 —— 它不在 metrics_for 里（那是纯规则口径），
    # 分数来自 tools/score_fact_judge.py 的离线输出。
    for rid, (ma, mb) in zip(common, rows_all):
        if fact_a and fact_b and rid in fact_a and rid in fact_b:
            ma["fact_judge"] = fact_a[rid]["score"]
            mb["fact_judge"] = fact_b[rid]["score"]

    section("整体", a, b, common, judge_a, judge_b, fact_a, fact_b)
    section("参考含事实（事实信号有效的子集）", a, b, has, judge_a, judge_b, fact_a, fact_b)
    section("参考无事实（事实项是常数的子集）", a, b, no, judge_a, judge_b, fact_a, fact_b)
    verdicts(rows_all, args.a_name, args.b_name)

    print("\n注：'改进占比' 是 B 优于 A 的样本比例。均值涨但占比接近 50%，"
          "说明是少数样本拉动的，不是整体变好。")


if __name__ == "__main__":
    main()
