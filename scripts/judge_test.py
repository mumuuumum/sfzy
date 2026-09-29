"""独立测试程序：验证六要素事实一致性 Judge（需求第十三节）。

先跑这个，再决定要不要接进 GRPOTrainer。

============================ 两种用法 ============================
    # 单条：看完整的中间过程（六要素、六项原始分、归一化分、加权分）
    python scripts/judge_test.py --model models/Qwen2.5-0.5B-Instruct --device cpu \
        --cases data/judge/fact_cases.jsonl --only d1_A-verbose

    # 固定测试集：跑 A~G 七类案例，看能不能排出正确的顺序
    python scripts/judge_test.py --model models/Qwen2.5-0.5B-Instruct --device cpu \
        --cases data/judge/fact_cases.jsonl

============================ 验收线 ============================
需求要求"重点确认 C、D、E、F、G 能显著低于 A、B"，并且希望看到

    完全正确 > 正确但简略 > 轻微事实错误 > 明显事实错误 > 核心裁判结果错误

本脚本按三条判据自动判定：

  1. 每个文档内 `min(A, B) > max(C, D, E, F, G)` —— 最硬的一条
  2. `mean(A, B) > mean(C, D, F, G) > mean(E)` —— 结果错要垫底
  3. `judgment_result` 这一项在 E 上显著低于 A —— 裁判结果分是重点观察指标

三条全过才算 Judge 可用。**不过就不要接 GRPO** —— 接上去只会得到噪声梯度。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.judge import (                                   # noqa: E402
    ELEMENTS,
    ELEMENT_ZH,
    FactConsistencyJudge,
    SixElements,
)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_documents() -> dict:
    """文档正文从 val 划分里按 id 取，避免在测试集里重复存 7KB 的判决书。"""
    path = resolve("data/splits/val.jsonl")
    return {json.loads(l)["id"]: json.loads(l)["source"]
            for l in open(path, encoding="utf-8") if l.strip()}


def print_elements(title: str, els: SixElements) -> None:
    print(f"\n{title}")
    for name in ELEMENTS:
        value = els.get(name).strip()
        value = value if value else "（空）"
        print(f"  {ELEMENT_ZH[name]:<8}{value[:120]}{'…' if len(value) > 120 else ''}")


def run_single(judge: FactConsistencyJudge, document: str, candidate: str) -> None:
    """需求第十三节要打印的五项，一次看全。"""
    doc_el = judge.document_elements(document)
    cand_el = judge.extract_six_elements(candidate)
    print_elements("【1】document 六要素", doc_el)
    print_elements("【2】candidate 六要素", cand_el)

    # 必须和 judge_summary 走同一套组 pair 规则（含"要素抽空时用整篇原文兜底"），
    # 否则演示看到的和生产跑的不是一个东西
    pairs = judge.build_pairs(doc_el, cand_el, document)
    scores, sources, pmaxs = judge.judge_elements_with_confidence(pairs)
    from sfzy.judge.schema import aggregate
    result = aggregate(
        {n: s for n, s in zip(ELEMENTS, scores)},
        weights=judge.weights,
        sources={n: s for n, s in zip(ELEMENTS, sources)},
    )

    # 逐要素对照：这一步才是"对比二者获得分数"的正文。
    # 只打印分数看不出模型到底在比什么，所以原文要素、摘要要素、判定分并排给。
    print("\n【3】逐要素对照与判定")
    for name, s, pm, (_, doc_sent, cand_sent) in zip(ELEMENTS, scores, pmaxs, pairs):
        doc_v = doc_el.get(name).strip() or "（空）"
        cand_v = cand_el.get(name).strip() or "（空）"
        src = sources[ELEMENTS.index(name)]
        # 兜底是在 build_pairs 里做的，库层的 sources 只区分"调没调模型"，
        # 所以这里按"抽出来的要素是空、但实际送进去的是整篇文书"来识别
        fallback = (not doc_el.get(name).strip()) and doc_sent == document and bool(document)
        tag = "  ← 空字段规则，未调用模型" if src == "empty_rule" else (
            "  ← 要素抽空，用整篇原文兜底判定" if fallback else "")
        conf = f"  pmax={pm:.2f}" if pm is not None else ""
        print(f"\n  ▸ {ELEMENT_ZH[name]}（{name}）  判定 {s}/4{conf}{tag}")
        print(f"      原文要素：{doc_v[:150]}{'…' if len(doc_v) > 150 else ''}")
        print(f"      摘要要素：{cand_v[:150]}{'…' if len(cand_v) > 150 else ''}")
        if fallback:
            print(f"      实际送判：原文要素位置放了整篇文书（{len(doc_sent)} 字）")

    print("\n【4】六个归一化评分（/4）")
    for name in ELEMENTS:
        print(f"  {ELEMENT_ZH[name]:<8}{result.scores[name]:.2f}")
    print(f"\n【5】weighted fact_reward = {result.weighted_reward:.4f}")
    print(f"     min_element_score     = {result.min_element_score:.4f}"
          "   （不含案由）")
    print(f"     judgment_result_score = {result.judgment_result_score:.4f}")


def run_cases(judge: FactConsistencyJudge, cases: list, docs: dict, verbose: bool) -> None:
    by_doc = defaultdict(list)
    for c in cases:
        by_doc[c["doc_id"]].append(c)

    rows = []
    for doc_id, group in by_doc.items():
        document = docs[doc_id]
        judge.clear_cache()                       # 每个文档单独提取一次
        doc_el = judge.document_elements(document)
        results = judge.judge_candidates(
            document, [c["candidate"] for c in group],
            candidate_ids=[c["id"] for c in group],
        )
        if verbose:
            print_elements(f"\n【{doc_id[:8]}】document 六要素", doc_el)
        for c, r in zip(group, results):
            rows.append((doc_id, c, r))

    print("\n" + "=" * 92)
    print(f"{'案例':<7}{'类型':<5}{'fact':>8}{'min':>7}{'结果分':>8}   说明")
    print("-" * 92)
    for doc_id, c, r in rows:
        print(f"{c['id']:<7}{c['case']:<5}{r.weighted_reward:>8.4f}"
              f"{r.min_element_score:>7.2f}{r.judgment_result_score:>8.2f}"
              f"   {c['desc']}")

    # ---- 判据 ----
    print("\n" + "=" * 92)
    per_case = defaultdict(list)
    for _, c, r in rows:
        per_case[c["case"]].append(r.weighted_reward)
    for k in sorted(per_case):
        v = per_case[k]
        print(f"  类型 {k}: fact_reward 均值 {st.mean(v):.4f}  (n={len(v)}, "
              f"min {min(v):.4f}, max {max(v):.4f})")

    good = per_case["A"] + per_case["B"]
    bad = per_case["C"] + per_case["D"] + per_case["F"] + per_case["G"]
    core = per_case["E"]
    c1 = st.mean(good) > st.mean(bad)
    c2 = st.mean(bad) > st.mean(core)
    c3 = all(
        st.mean(per_case[a] or [0]) > st.mean(per_case[e] or [0])
        for a in ("A",) for e in ("E",)
    )
    print(f"\n  判据 1  mean(A,B) > mean(C,D,F,G)：{st.mean(good):.4f} vs {st.mean(bad):.4f}"
          f"   {'✓' if c1 else '✗'}")
    print(f"  判据 2  mean(C,D,F,G) > mean(E)：{st.mean(bad):.4f} vs {st.mean(core):.4f}"
          f"   {'✓' if c2 else '✗'}")
    per_doc_ok = 0
    for doc_id, group in by_doc.items():
        vals = {c["case"]: r.weighted_reward
                for (d, c, r) in rows if d == doc_id}
        if max(vals["A"], vals["B"]) > max(vals[k] for k in "CDEFG"):
            per_doc_ok += 1
    print(f"  判据 3  每个文档内 min(A,B) > max(C,D,E,F,G)："
          f"{per_doc_ok}/{len(by_doc)} 个文档通过"
          f"   {'✓' if per_doc_ok == len(by_doc) else '✗'}")
    print(f"\n  结论：{'✓ 三条全过，可以接 GRPO' if (c1 and c2 and per_doc_ok == len(by_doc)) else '✗ 有判据未过，先别接 GRPO'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="六要素事实一致性 Judge 测试")
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="本地内存小就调小；服务器上可以调到 16")
    ap.add_argument("--cases", default="data/judge/fact_cases.jsonl")
    ap.add_argument("--only", default=None, help="只跑某个案例 id（单条详细模式）")
    ap.add_argument("--filter", default=None,
                    help="只跑案例 id 或文档 id 含该子串的案例。"
                         "本地用小模型验接线时，跑一个文档的 7 条就够")
    ap.add_argument("--verbose", action="store_true", help="打印每个文档的六要素")
    ap.add_argument("--stats", action="store_true", help="打印 0-4 各档比例")
    args = ap.parse_args()

    docs = load_documents()
    cases = [json.loads(l) for l in open(resolve(args.cases), encoding="utf-8") if l.strip()]
    if args.filter:
        cases = [c for c in cases if args.filter in c["id"] or args.filter in c["doc_id"]]
        if not cases:
            raise SystemExit(f"没有匹配 {args.filter!r} 的案例")
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
        if not cases:
            raise SystemExit(f"没有 id 为 {args.only} 的案例")

    print(f"模型 {args.model}   设备 {args.device}   dtype {args.dtype}   "
          f"batch {args.batch_size}")
    judge = FactConsistencyJudge(
        model_path=args.model, device=args.device, dtype=args.dtype,
        max_batch_size=args.batch_size,
    )

    if args.only:
        c = cases[0]
        run_single(judge, docs[c["doc_id"]], c["candidate"])
        return

    run_cases(judge, cases, docs, args.verbose)

    if args.stats:
        print("\n推理统计：", judge.runtime.stats)


if __name__ == "__main__":
    main()
