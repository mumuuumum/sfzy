"""验证排序裁判：它能不能把"好的候选"排在"坏的候选"前面？

============================ 为什么要单独验证 ============================
点式裁判栽了三次（`configs/judge_rank_rubric.yaml` 顶部记了全过程），
所以排序裁判上线前必须证明它真的有分辨力，而不是又换了个形式的模板分。

验证方法不需要任何 GPU rollout：**用可控扰动当"同一个 prompt 的多条候选"**。
对每条三元组造一组候选：

    base      原样（好的）
    d_digit   把第一个数字改错一位
    d_party   当事人姓名换成"张某"
    d_halluc  掺一条参考里没有的事实
    e_order   调换两个相邻句子的顺序（语义不变，**好的**）

一组 5 条，正好是 GRPO 一个 group 的缩小版。判据很直接：

  * base 和 e_order 应该排前面（它们是对的）
  * d_digit / d_party / d_halluc 应该排后面（它们破坏了事实）

**注意 base 和 e_order 都是"好"的，不能要求 base 永远第一** ——
调换句序不改变语义，裁判把 e_order 排在 base 前面是合理的。
真正的判据是：**好的那一类（base + e_order）整体排在坏的那一类之前**。

============================ 用法 ============================
    # 服务器上，真裁判（约 1 分钟 / 20 组，repeat=2）
    python tools/check_ranker.py --input "data/triples/sft_val_shard0of2.jsonl" \
        --limit 20 --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --device cuda:1

    # 本地验证接线（不加载模型，用 ROUGE 假装排序）
    python tools/check_ranker.py --input "data/triples/sft_val_shard0of2.jsonl" \
        --limit 20 --stub
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bench_metrics import BREAKING, NEUTRAL   # noqa: E402  复用同一套扰动
from sfzy.eval.metrics import (               # noqa: E402
    ListwiseRankScorer,
    SemanticScorer,
    load_rubric,
    parse_ranking,
)
from sfzy.eval.rouge import score_pair         # noqa: E402

# 组内成员：base + 这一批扰动。选它们是因为在 1340 条上出现率都够高
# （d_halluc 100%、e_order 99.9%、d_party 95%、d_digit 68%）。
GOOD = ("base", "e_order")
BAD = ("d_digit", "d_party", "d_halluc")
MEMBERS = ("base", "d_digit", "d_party", "d_halluc", "e_order")


class RougeStubScorer(SemanticScorer):
    """不加载模型的假排序裁判：按 ROUGE-L 排序。

    只用来验证 `check_ranker.py` 自己的接线（分组、名次换算、统计），
    不能用来评价任何 rubric —— 它测的是 ROUGE，不是裁判。"""

    name = "rank_stub"
    group_size = len(MEMBERS)

    def score_batch(self, items):
        out = []
        for start in range(0, len(items), self.group_size):
            chunk = items[start:start + self.group_size]
            scores = [score_pair(c["candidate"], c["reference"], mode="jieba")["rouge-l-f"]
                      for c in chunk]
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            out.extend([100.0 * (len(scores) - 1 - order.index(i)) / (len(scores) - 1)
                        for i in range(len(scores))])
        return out


def build_groups(records: list, limit: int) -> list:
    """每条记录造一组候选项，返回 [(id, reference, source, [(kind, text), ...])]。"""
    groups = []
    for r in records:
        text = r["output"]
        made = [("base", text)]
        for kind in ("d_digit", "d_party", "d_halluc", "e_order"):
            fn = BREAKING.get(kind) or NEUTRAL.get(kind)
            t = fn(text)
            if t and t != text:
                made.append((kind, t))
        if len(made) == len(MEMBERS):
            groups.append((r["id"], r["reference"], r.get("source"), made))
        if len(groups) >= limit:
            break
    return groups


def report(groups: list, scores: list, raw_lines: list) -> None:
    ranks = {k: [] for k in MEMBERS}
    good_first = 0
    bad_before_good = 0
    for gi in range(len(groups)):
        chunk = scores[gi * len(MEMBERS):(gi + 1) * len(MEMBERS)]
        order = sorted(range(len(chunk)), key=lambda i: -chunk[i])
        rank_of = {i: pos for pos, i in enumerate(order)}
        for i, (kind, _) in enumerate(groups[gi][3]):
            ranks[kind].append(rank_of[i])
        best_kind = groups[gi][3][order[0]][0]
        good_first += best_kind in GOOD
        # 好的一类里**最差名次** vs 坏的一类里**最好名次**
        worst_good = max(rank_of[i] for i, (k, _) in enumerate(groups[gi][3]) if k in GOOD)
        best_bad = min(rank_of[i] for i, (k, _) in enumerate(groups[gi][3]) if k in BAD)
        bad_before_good += best_bad < worst_good

    n = len(groups)
    print(f"\n{'='*70}\n组数 {n}，每组 {len(MEMBERS)} 条候选（名次 0 = 最好）\n{'='*70}")
    print(f"{'候选类型':<12}{'平均名次':>10}{'排第一比例':>12}")
    for kind in MEMBERS:
        rs = ranks[kind]
        first = sum(1 for x in rs if x == 0) / len(rs)
        mark = "  ← 好的" if kind in GOOD else "  ← 坏的"
        print(f"{kind:<12}{st.mean(rs):>10.2f}{first:>11.0%}{mark}")

    print(f"\n最好的一条属于「好的那类」（base/e_order）的比例: {good_first/n:.0%}")
    print(f"坏候选抢在好候选前面的组数: {bad_before_good}/{n} = {bad_before_good/n:.0%}")
    ok = good_first / n >= 0.8 and bad_before_good / n <= 0.2
    print(f"\n判据：好候选排名第一 ≥80% 且 坏候选抢先 ≤20%  →  "
          f"{'✓ 通过' if ok else '✗ 不通过'}")
    print("\n注：base 和 e_order 都是好的（调换句序不改变语义），"
          "所以不要要求 base 永远第一。")


def main() -> None:
    ap = argparse.ArgumentParser(description="验证排序裁判的分辨力")
    ap.add_argument("--input", default="data/triples/sft_val_shard0of2.jsonl")
    ap.add_argument("--limit", type=int, default=20, help="跑多少组")
    ap.add_argument("--model", default=None, help="裁判模型路径；--stub 时可省略")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--repeat", type=int, default=2, help="不同候选顺序各排一次再平均")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--stub", action="store_true", help="不加载模型，用 ROUGE 假装排序")
    args = ap.parse_args()

    path = Path(args.input) if Path(args.input).is_absolute() else ROOT / args.input
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    groups = build_groups(recs, args.limit)
    if not groups:
        raise SystemExit("没有一条记录能凑齐这一组扰动，换个输入文件")
    print(f"从 {len(recs)} 条里凑出 {len(groups)} 组完整候选")

    items = []
    for _, ref, src, made in groups:
        for kind, text in made:
            items.append({"candidate": text, "reference": ref, "source": src, "kind": kind})

    if args.stub:
        scorer = RougeStubScorer()
        print("（stub 模式：用 ROUGE-L 假装排序，只验证接线）")
    else:
        if not args.model:
            raise SystemExit("要么 --model，要么 --stub")
        scorer = ListwiseRankScorer(
            model_path=args.model,
            group_size=len(MEMBERS),
            rubric=load_rubric("configs/judge_rank_rubric.yaml"),
            repeat=args.repeat,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )

    scores = scorer.score_batch(items)
    report(groups, scores, getattr(scorer, "last_raw", []))

    # 把裁判原文落盘，人工核查"它凭什么这么排"
    out = ROOT / "data/judge/ranker_check_raw.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for (gid, _, _, made), raw in zip(groups, getattr(scorer, "last_raw", [None] * len(groups))):
            f.write(json.dumps({"id": gid, "raw": raw}, ensure_ascii=False) + "\n")
    print(f"裁判原文 → {out}")


if __name__ == "__main__":
    main()
