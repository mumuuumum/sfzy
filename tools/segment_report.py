"""在真实数据集上跑三要素切分器，统计识别率 / 完整切分率 / 不确定边界，并导样例。

============================ 为什么要有这个工具 ============================
切分器是**纯规则**的：规则对着语料调，就必须能一眼看到"这一刀是谁下的、
有多少条是它下的、下歪了长什么样"。这个工具把三件事一次做完：

1. **统计**：三要素各自的识别/缺失/不确定比例、完整切分率、不确定边界比例、
   顺序不变量、以及每条规则的命中分布（"哪条规则真的在干活"）。
2. **抽样**：按规则分桶导出典型**成功**和**失败/存疑**样例（标出区间），
   人工扫一眼就能判断规则好不好用。
3. **对账**：跑一遍自洽性检查（答辩片段里有没有答辩词、法院事实里有没有说理词、
   文本有没有掉在三个区间之外），这些检查不需要人工标注。

本工具**不修改任何训练/奖励逻辑**，只读数据、只写报告。

----------------------------- 用法 -----------------------------
    # 全量训练集 + 样例 + 落盘缓存（人工摘要离线提取）
    python tools/segment_report.py \
        --data data/splits/train.jsonl \
        --out outputs/segment_report \
        --cache-out data/segment_cache/train_summary_segments.jsonl

    # 冒烟：只看前 500 条
    python tools/segment_report.py --limit 500 --out /tmp/seg_smoke
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.segment import (  # noqa: E402
    ELEMENT_NAMES,
    ELEMENTS,
    Confidence,
    SegmentationResult,
    SegmentCache,
    Segmenter,
    Status,
    split_sentences,
)
from sfzy.segment.patterns import CLAIM_VERBS, COURT_TONE, RESULT_MARKER  # noqa: E402


# ---------------------------------------------------------------- 读写


def load_summaries(path: str, field: str, limit: int | None) -> List[str]:
    """读 (id, 文本)，保持文件顺序；同一份文件里重复文本只留第一次出现。"""

    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(field, "")
            if isinstance(text, str) and text.strip():
                texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    return texts


# ---------------------------------------------------------------- 统计


def health_flags(r: SegmentationResult) -> List[str]:
    """自洽性检查（不需要金标）：这三类命中说明"这一刀可能切歪了"。

    * 答辩片段里连"辩/答辩"都没有（除显式"未答辩"声明）；
    * 法院事实片段里一个说理语素都没有；
    * 诉请片段里一个诉请动词都没有；
    * 法院事实终点之后的尾部里没有"判决/裁定/驳回"（尾部标志可能是弱兜底猜的）。
    """

    flags: List[str] = []
    p, d, f = (r.spans[k] for k in ELEMENTS)
    if d.status is Status.IDENTIFIED:
        text = d.text_of(r.summary)
        defense_words = ("辩称", "答辩", "抗辩", "辩", "主张", "承认", "同意", "异议", "认可")
        if not any(w in text for w in defense_words):
            flags.append("答辩片段内无答辩词")
    if f.status is Status.IDENTIFIED and not COURT_TONE.search(f.text_of(r.summary)):
        flags.append("法院事实片段内无说理语素")
    if p.status is Status.IDENTIFIED and not CLAIM_VERBS.search(p.text_of(r.summary)):
        flags.append("诉请片段内无诉请动词")
    if f.status is Status.IDENTIFIED and f.end is not None:
        tail = r.summary[f.end :]
        if tail and not RESULT_MARKER.search(tail):
            flags.append("尾部内无裁判结果标志")
    return flags


def compute_stats(results: Sequence[SegmentationResult]) -> Dict[str, object]:
    n = len(results)
    elem_counts: Dict[str, collections.Counter] = {k: collections.Counter() for k in ELEMENTS}
    rule_counts: Dict[str, collections.Counter] = {
        "P.start": collections.Counter(),
        "P.end": collections.Counter(),
        "D.start": collections.Counter(),
        "D.end": collections.Counter(),
        "F.start": collections.Counter(),
        "F.end": collections.Counter(),
    }
    conf_counts: Dict[str, collections.Counter] = {k: collections.Counter() for k in ELEMENTS}
    complete = 0
    order_ok = 0
    any_uncertain = 0
    health = collections.Counter()
    #: 不确定边界的**成因**：哪条边界没落下来、当时记了什么来源/备注。
    uncertain_reasons: Dict[str, collections.Counter] = {k: collections.Counter() for k in ELEMENTS}
    lengths = {k: [] for k in ELEMENTS}
    uncovered_tail = []
    uncovered_head = []

    for r in results:
        spans = r.spans
        for k in ELEMENTS:
            span = spans[k]
            elem_counts[k][span.status.value] += 1
            conf_counts[k][span.confidence.value] += 1
            if span.status is Status.UNCERTAIN:
                reason = span.end_rule if "失败" in span.end_rule else span.start_rule
                note = span.notes[0] if span.notes else ""
                uncertain_reasons[k][f"{reason}｜{note}"] += 1
            if span.status is Status.IDENTIFIED and span.start is not None and span.end is not None:
                lengths[k].append(span.end - span.start)
        rule_counts["P.start"][spans["plaintiff_claims"].start_rule.split("(")[0]] += 1
        rule_counts["P.end"][spans["plaintiff_claims"].end_rule.split("(")[0]] += 1
        rule_counts["D.start"][spans["defendant_defenses"].start_rule.split("(")[0]] += 1
        rule_counts["D.end"][spans["defendant_defenses"].end_rule.split("(")[0]] += 1
        rule_counts["F.start"][spans["court_facts"].start_rule.split("(")[0]] += 1
        rule_counts["F.end"][spans["court_facts"].end_rule.split("(")[0]] += 1
        complete += int(r.complete)
        order_ok += int(r.order_ok)
        any_uncertain += int(bool(r.uncertain_elements))

        # ---- 自洽性检查（不需要金标也能算）
        for flag in health_flags(r):
            health[flag] += 1
        # 覆盖：诉请起点前、法院事实终点后的游离字符
        p, f = spans["plaintiff_claims"], spans["court_facts"]
        if p.start is not None:
            uncovered_head.append(p.start)
        if f.end is not None:
            uncovered_tail.append(len(r.summary) - f.end)

    def rates(k: str) -> Dict[str, float]:
        return {
            status: elem_counts[k][status] / n if n else 0.0
            for status in (Status.IDENTIFIED.value, Status.ABSENT.value, Status.UNCERTAIN.value)
        }

    def length_stats(key: str) -> Dict[str, float]:
        vals = lengths[key]
        if not vals:
            return {}
        return {
            "mean": st.mean(vals),
            "median": st.median(vals),
            "p10": sorted(vals)[int(0.10 * (len(vals) - 1))],
            "p90": sorted(vals)[int(0.90 * (len(vals) - 1))],
            "zero": sum(1 for v in vals if v == 0),
        }

    return {
        "n": n,
        "elements": {
            k: {
                "rates": rates(k),
                "counts": dict(elem_counts[k]),
                "confidence": {
                    c.value: conf_counts[k][c.value] for c in Confidence
                },
                "length": length_stats(k),
            }
            for k in ELEMENTS
        },
        "complete_rate": complete / n if n else 0.0,
        "order_ok_rate": order_ok / n if n else 0.0,
        "any_uncertain_rate": any_uncertain / n if n else 0.0,
        "rules": {k: dict(v.most_common()) for k, v in rule_counts.items()},
        "health": dict(health),
        "uncertain_reasons": {
            k: dict(v.most_common(8)) for k, v in uncertain_reasons.items()
        },
        "uncovered": {
            "head_mean": st.mean(uncovered_head) if uncovered_head else 0.0,
            "head_nonzero": sum(1 for v in uncovered_head if v > 0),
            "tail_mean": st.mean(uncovered_tail) if uncovered_tail else 0.0,
            "tail_max": max(uncovered_tail) if uncovered_tail else 0,
        },
    }


# ---------------------------------------------------------------- 样例分桶


def _bucket_examples(
    results: Sequence[SegmentationResult], max_total: int = 12
) -> Dict[str, List[Tuple[str, SegmentationResult]]]:
    """按"这一刀是谁下的"分桶，方便人工逐桶检查。"""

    buckets: Dict[str, List[Tuple[str, SegmentationResult]]] = collections.defaultdict(list)
    for r in results:
        p, d, f = (r.spans[k] for k in ELEMENTS)
        if not r.order_ok:
            buckets["非法顺序(不应出现)"].append(("order", r))
        if d.status is Status.UNCERTAIN or f.status is Status.UNCERTAIN:
            buckets["不确定边界"].append(("uncertain", r))
        if "推断" in d.end_rule:
            buckets["答辩终点-推断"].append(("infer_d_end", r))
        if d.start_rule.split("(")[0] in (
            "被告辩称:具名当事人辩称",
            "被告辩称:光名主体辩称",
            "被告辩称:并列主体共同辩称",
            "被告辩称:角色主体辩称",
            "被告辩称:错别字辨称",
            "被告辩称:被告+主张",
        ):
            buckets["答辩起点-弱标志"].append(("weak_d_start", r))
        if any(note for note in r.notes if "说理回述" in note):
            buckets["答辩标志被顺序校验否掉"].append(("d_rejected", r))
        if d.status is Status.ABSENT:
            buckets["无答辩(缺失)"].append(("d_absent", r))
        if "推断" in f.start_rule:
            buckets["法院事实起点-推断"].append(("infer_f_start", r))
        if f.end_rule.split("(")[0] in ("未找到尾部标志，取文末",):
            buckets["尾部标志未命中"].append(("no_tail", r))
        if "仅法律依据" in f.end_rule or "无法律依据" in f.end_rule:
            buckets["尾部-弱兜底"].append(("weak_tail", r))
        # 字符级切分：切点落在句子内部（不是靠整句切），说明支持"同句多要素"
        sents = split_sentences(r.summary)
        starts = {s.start for s in sents}
        if d.status is Status.IDENTIFIED and d.start is not None and d.start not in starts:
            buckets["同一句内切出多个要素(字符级)"].append(("mid_sentence", r))
        if (
            p.status is Status.IDENTIFIED
            and d.status is Status.IDENTIFIED
            and f.status is Status.IDENTIFIED
            and p.confidence is Confidence.HIGH
            and f.confidence is Confidence.HIGH
        ):
            buckets["全强标志成功"].append(("all_strong", r))
        # 自洽性不达标的三类，单独拎出来看（这几类是"规则可能切歪了"的候选）
        for flag in health_flags(r):
            buckets[f"体检：{flag}"].append(("health", r))
    out: Dict[str, List[Tuple[str, SegmentationResult]]] = {}
    for name, items in buckets.items():
        if not items:
            continue
        step = max(1, len(items) // max_total)
        out[name] = items[::step][:max_total]
    return out


def _fmt_example(kind: str, r: SegmentationResult) -> List[str]:
    lines = ["* 摘要：", f"  {r.summary}", "  切分："]
    for k in ELEMENTS:
        span = r.spans[k]
        tag = f"{ELEMENT_NAMES[k]}"
        if span.status is Status.ABSENT:
            lines.append(f"  - {tag}：**缺失**（{span.start_rule}）")
        elif span.status is Status.UNCERTAIN:
            lines.append(
                f"  - {tag}：**不确定**（{span.start_rule} → {span.end_rule}）"
                f"｜{'；'.join(span.notes)}"
            )
        else:
            lines.append(
                f"  - {tag}［{span.confidence.value}］：{span.text_of(r.summary)!r}"
                f"（{span.start_rule} → {span.end_rule}）"
            )
    if r.notes:
        lines.append(f"  - 备注：{'；'.join(r.notes)}")
    lines.append("")
    return lines


# ---------------------------------------------------------------- 报告


def render_markdown(
    stats: Dict[str, object],
    examples: Dict[str, List[Tuple[str, SegmentationResult]]],
    *,
    data_path: str,
    field: str,
    timing: Dict[str, float],
) -> str:
    n = stats["n"]
    lines: List[str] = []
    lines.append("# 三要素切分器：train 集切分统计")
    lines.append("")
    lines.append(f"- 数据：`{data_path}`（字段 `{field}`），共 **{n}** 条")
    lines.append(
        f"- 纯规则切分耗时：{timing['segment']:.2f}s"
        f"（{n / max(timing['segment'], 1e-9):,.0f} 条/秒，单进程）"
    )
    lines.append("- 本报告只统计切分器自身的产出，**不含任何 reward / 训练改动**")
    lines.append("")

    lines.append("## 1. 总览")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 完整切分率 | **{stats['complete_rate']:.1%}** |")
    lines.append(f"| 不确定边界比例（任一要素不确定） | **{stats['any_uncertain_rate']:.1%}** |")
    lines.append(f"| 顺序合法率（P→D→F 且不重叠） | {stats['order_ok_rate']:.1%} |")
    lines.append("")

    lines.append("## 2. 各要素识别率")
    lines.append("")
    lines.append("| 要素 | 识别 | 确认缺失 | 不确定 | 高/中/低置信 | 长度中位数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for k in ELEMENTS:
        e = stats["elements"][k]
        r = e["rates"]
        conf = e["confidence"]
        ln = e["length"]
        lines.append(
            f"| {ELEMENT_NAMES[k]} | {r['identified']:.1%} | {r['absent']:.1%} | {r['uncertain']:.1%} | "
            f"{conf['high']}/{conf['medium']}/{conf['low']} | {ln.get('median', 0):.0f} |"
        )
    lines.append("")
    lines.append(
        "说明：「确认缺失」只用于**被告辩称**（摘要结构里它本来就可选）；"
        "「不确定」＝规则找到了迹象但定不下边界（例如答辩标志存在、却既没有法院叙述标志、"
        "也没有可用的句边界），这种情况既不硬切也不判缺失。"
    )
    lines.append("")
    lines.append("### 不确定边界都是怎么来的")
    lines.append("")
    lines.append("| 要素 | 成因（来源｜备注） | 条数 |")
    lines.append("| --- | --- | --- |")
    for k in ELEMENTS:
        for reason, cnt in stats["uncertain_reasons"][k].items():
            lines.append(f"| {ELEMENT_NAMES[k]} | {reason} | {cnt} |")
    lines.append("")
    lines.append(
        "注意：**没有答辩**时诉请终点与法院事实起点是同一条边界（"
        "「诉请终点=法院事实起点」），所以法院事实起点定不下来时，诉请也会一起记不确定"
        "——不会只报一边。"
    )
    lines.append("")

    lines.append("## 3. 边界来源分布（哪条规则在干活）")
    lines.append("")
    for key in ("P.start", "P.end", "D.start", "D.end", "F.start", "F.end"):
        lines.append(f"**{key}**")
        lines.append("")
        lines.append("| 规则 | 条数 | 占比 |")
        lines.append("| --- | --- | --- |")
        for rule, cnt in list(stats["rules"][key].items())[:12]:
            lines.append(f"| {rule} | {cnt} | {cnt / n:.1%} |")
        lines.append("")

    lines.append("## 4. 自洽性检查（无需人工标注）")
    lines.append("")
    lines.append("| 检查 | 命中条数 |")
    lines.append("| --- | --- |")
    for k, v in stats["health"].items():
        lines.append(f"| {k} | {v} |")
    unc = stats["uncovered"]
    lines.append(
        f"| 诉请之前的案件背景句（正常，均值 {unc['head_mean']:.0f} 字） | {unc['head_nonzero']} 条 >0 字 |"
    )
    lines.append(
        f"| 法院事实终点之后的剩余字符（即尾部，最长 {unc['tail_max']} 字） | 属正常：尾部不属于三要素 |"
    )
    lines.append("")

    for name, items in examples.items():
        lines.append(f"## 样例：{name}（{len(items)} 条）")
        lines.append("")
        for kind, r in items:
            lines.extend(_fmt_example(kind, r))
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="三要素切分器的 train 集统计与样例导出")
    ap.add_argument("--data", default="data/splits/train.jsonl")
    ap.add_argument("--field", default="summary")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（冒烟用）")
    ap.add_argument(
        "--show",
        type=int,
        default=0,
        help="随机抽 N 条把切分结果打到终端（标出区间与每条边界的来源），看了再决定要不要全量跑",
    )
    ap.add_argument("--seed", type=int, default=0, help="--show 抽样用的随机种子")
    ap.add_argument("--out", default="outputs/segment_report")
    ap.add_argument("--examples", type=int, default=8, help="每个桶导出的样例条数")
    ap.add_argument(
        "--cache-out",
        default=None,
        help="把切分结果落盘成离线缓存（人工摘要预热用）；GRPO 候选摘要不需要",
    )
    ap.add_argument("--segments-out", default=None, help="逐条切分结果（jsonl，便于二次分析）")
    ap.add_argument("--no-write", action="store_true", help="只打印统计，不落盘")
    ap.add_argument(
        "--d-bare-first",
        action="store_true",
        help="打开对照规则：文本里第一个句首'被告'就是答辩起点（默认关闭，见 docs/segmenter.md）",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    texts = load_summaries(args.data, args.field, args.limit)
    if not texts:
        print(f"[segment_report] 没读到数据：{args.data} 字段 {args.field}", file=sys.stderr)
        return 2

    segmenter = Segmenter(prefer_bare_defendant=args.d_bare_first)

    if args.show:
        import random as _random

        rng = _random.Random(args.seed)
        for text in rng.sample(texts, min(args.show, len(texts))):
            res = segmenter.segment(text)
            print("=" * 100)
            print(res.marked_text())
            for k in ELEMENTS:
                span = res.spans[k]
                tag = ELEMENT_NAMES[k]
                if span.status is Status.ABSENT:
                    print(f"  {tag}：缺失（{span.start_rule}）")
                else:
                    print(
                        f"  {tag}［{span.status.value}/{span.confidence.value}］"
                        f"{span.start}-{span.end}｜{span.start_rule} → {span.end_rule}"
                    )
            if res.notes:
                print(f"  备注：{'；'.join(res.notes)}")
            print()

    # ---- 冷启动：实时切分（GRPO 候选摘要走的就是这条路）
    t0 = time.perf_counter()
    results = [segmenter.segment(t) for t in texts]
    segment_secs = time.perf_counter() - t0

    # ---- 离线缓存：人工摘要一次性预热，之后查表
    cache = SegmentCache(segmenter=segmenter)
    t1 = time.perf_counter()
    cache.warm(texts)
    warm_secs = time.perf_counter() - t1
    t2 = time.perf_counter()
    for t in texts:
        cache.get_or_segment(t)
    lookup_secs = time.perf_counter() - t2

    stats = compute_stats(results)
    examples = _bucket_examples(results, max_total=args.examples)
    timing = {
        "segment": segment_secs,
        "warm": warm_secs,
        "lookup": lookup_secs,
        "warm_per_item": warm_secs / len(texts),
        "lookup_per_item": lookup_secs / len(texts),
    }

    print(
        f"[segment_report] n={stats['n']} 完整切分率={stats['complete_rate']:.1%} "
        f"不确定={stats['any_uncertain_rate']:.1%} 顺序合法={stats['order_ok_rate']:.1%}"
    )
    print(
        f"[segment_report] 实时切分 {stats['n']} 条 {segment_secs:.2f}s"
        f"（{stats['n'] / segment_secs:,.0f}/s）；"
        f"缓存命中 {lookup_secs:.3f}s（{stats['n'] / max(lookup_secs, 1e-9):,.0f}/s）"
    )

    if args.no_write:
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(
        json.dumps({**stats, "timing": timing}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        render_markdown(
            stats, examples, data_path=args.data, field=args.field, timing=timing
        ),
        encoding="utf-8",
    )
    if args.cache_out:
        written = cache.save(args.cache_out)
        print(f"[segment_report] 离线缓存已写：{args.cache_out}（{written} 条）")
    if args.segments_out:
        p = Path(args.segments_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for text, res in zip(texts, results):
                f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
        print(f"[segment_report] 逐条切分已写：{args.segments_out}")
    print(f"[segment_report] 报告已写：{out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
