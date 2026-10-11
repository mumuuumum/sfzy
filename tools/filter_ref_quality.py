"""标记"参考摘要自身有问题"的样本（不调 API，可复现）。

只做高精度、可解释的三类判断：

  1) doc_mismatch —— 参考摘要与原文几乎不可对齐（整案/整段不匹配）。
     信号：greedy 最长匹配覆盖后，参考里"原文中找不到"的字占比 nov_ref。
     实测：正常样本 p90≈0.013，而整案不匹配样本 0.12~0.22。
  2) bad_number —— 参考里出现"金额型"数字，原文里完全不存在。
     （排除 第X条/年月日/案号 等上下文，提升精度）
  3) short_or_empty —— 参考缺失或过短。

输出：
  <out>.jsonl        每行 {id, nov_ref, len_ref, bad_numbers, flags}
  <out>.clean.jsonl  只保留未被标记的样本（可直接喂给 probe_fc_human_arm.py）

用法：
  python tools/filter_ref_quality.py \
      --input data/judge/sft_train_shard0.jsonl --limit 1000 \
      --out data/judge/sft_train_shard0.refaudit --doc-thr 0.08
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

NON_CJK = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]")
AMOUNT = re.compile(r"\d[\d,\.]*\d")
BAD_CTX = re.compile(r"[第条项款年月日号字]")
DATE_LIKE = re.compile(r"^(19|20)\d{2}(\d{2}){0,2}$")


def norm(x: str) -> str:
    return NON_CJK.sub("", x or "")


def novel_frac(ref: str, src: str) -> float:
    """参考里有多大比例的字，在原文中完全找不到（贪心最长匹配覆盖）。"""
    n = len(ref)
    if n == 0:
        return 1.0
    i = matched = 0
    while i < n:
        lo, hi, best = 1, n - i, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if ref[i:i + mid] in src:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if best == 0:
            i += 1
        else:
            matched += best
            i += best
    return 1 - matched / n


def bad_amounts(ref: str, src: str) -> List[str]:
    """参考里的“金额型”数字（排除法条/日期/案号上下文）在原文中找不到的。"""
    out: List[str] = []
    for m in AMOUNT.finditer(ref):
        s, e = m.span()
        if BAD_CTX.search(ref[max(0, s - 2):s] + ref[e:e + 2]):
            continue
        digits = re.sub(r"[^\d]", "", m.group())
        if len(digits) >= 3 and digits not in src:
            if DATE_LIKE.match(digits) or (len(digits) >= 6 and digits[:2] in ("19", "20")):
                continue  # 形如 2007.08 / 2016.9.16 / 20170809 的日期，不是金额
            out.append(digits)
    return sorted(set(out))


def audit(rows: List[Dict[str, Any]], doc_thr: float,
          min_len: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    audit_rows: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []
    for r in rows:
        ref_raw = r.get("reference") or ""
        ref, src = norm(ref_raw), norm(r.get("source") or "")
        nf = novel_frac(ref, src)
        bn = bad_amounts(ref_raw, src)
        flags: List[str] = []
        if len(ref) < min_len:
            flags.append("short_or_empty")
        if nf >= doc_thr:
            flags.append("doc_mismatch")
        if bn:
            flags.append("bad_number")
        audit_rows.append({
            "id": str(r.get("id")), "nov_ref": round(nf, 4),
            "len_ref": len(ref), "bad_numbers": bn, "flags": flags,
        })
        if not flags:
            clean_rows.append(r)
    return audit_rows, clean_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="标记参考摘要自身有问题的样本")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None, help="输出前缀；默认 <input>.refaudit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--doc-thr", type=float, default=0.08, help="nov_ref 阈值")
    ap.add_argument("--min-len", type=int, default=20, help="参考最短字数")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    prefix = Path(args.out or (args.input + ".refaudit"))
    audit_rows, clean_rows = audit(rows, args.doc_thr, args.min_len)

    with open(f"{prefix}.jsonl", "w", encoding="utf-8") as f:
        for rec in audit_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(f"{prefix}.clean.jsonl", "w", encoding="utf-8") as f:
        for r in clean_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    c = Counter(fl for rec in audit_rows for fl in rec["flags"])
    dirty = sum(1 for rec in audit_rows if rec["flags"])
    print("总 %d 条；标记 %d 条（%.1f%%），保留 %d 条"
          % (len(rows), dirty, 100 * dirty / max(1, len(rows)), len(clean_rows)))
    print("分类：", dict(c))
    for rec in [r for r in audit_rows if r["flags"]][:10]:
        print("  ", rec["id"], rec["flags"],
              "nov=%.3f" % rec["nov_ref"], rec["bad_numbers"][:2])
    print("→ %s.jsonl / %s.clean.jsonl" % (prefix, prefix))


if __name__ == "__main__":
    main()
