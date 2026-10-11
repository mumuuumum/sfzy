"""第二层：对"参考摘要"做一次语义忠实性审计（调 judge API）。

为什么需要它：残损串（离心式/贝茨恒人/原稿…）、错主体、错值这类问题只有
5~10 个字，字符覆盖率过滤（tools/filter_ref_quality.py）抓不到；只有语义判断
能抓。为了压住裁判自身的误判（把"遗漏/概括"当事实错误、或引错原文），本脚本
强制模型输出**两段逐字原文**（摘要里的问题句 + 原文里的对立句），再用程序
回连校验：任一段对不上，就判为 unverified 并**保留**该样本（宁可漏杀）。

输出：
  <out>.jsonl          {id, contradiction, verified, summary_span, source_span, reason}
  <out>.clean.jsonl    只保留"未被确认有问题"的样本

用法：
  python tools/audit_reference_fidelity.py \
      --config configs/grpo_fact_cov_ee_t4.yaml \
      --input data/judge/sft_train_shard0.jsonl --limit 1000 \
      --out data/judge/sft_train_shard0.refaudit_llm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import validate_reward_metrics as V                      # noqa: E402
from sfzy.judge.api_runtime import build_api_runtime     # noqa: E402

SYSTEM = (
    "你是判决书摘要审核员。判断【待审摘要】里是否存在与【判决书原文】明确的"
    "矛盾：主体、金额、时间、法律关系、判决结果被写反、写错或凭空编造。"
    "纯属省略细节、概括、表达不严谨、可能误导的，一律不算矛盾。"
    "只输出 JSON："
    '{"contradiction":0/1,"summary_span":"待审摘要里出问题的那句原话(逐字)",'
    '"source_span":"原文中与之矛盾的原文语句(逐字)","reason":"一句话"}。'
    "没有矛盾时两个 span 留空、contradiction=0。"
)
NON = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]")


def norm(x: str) -> str:
    return NON.sub("", x or "")


def covered(span: str, text: str, thr: float = 0.85) -> bool:
    """span 能否在 text 里找到（允许少量标点/字差异）。"""
    s = norm(span)
    t = norm(text)
    if not s:
        return False
    if s in t:
        return True
    n = len(s)
    best, lo, hi = 0, 1, n
    while lo <= hi:
        mid = (lo + hi) // 2
        if any(s[i:i + mid] in t for i in range(n - mid + 1)):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best >= thr * n


def parse(raw: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="参考摘要语义忠实性审计")
    ap.add_argument("--config", default="configs/grpo_fact_cov_ee_t4.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--dry-run", action="store_true", help="只打印第一条 prompt")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    prefix = Path(args.out or (args.input + ".refaudit_llm"))

    _, cfg_info, _ = V.load_config_bundle(V.resolve(args.config), ["fact_consistency"])
    api = dict(cfg_info["judge_api"])
    if args.judge_model:
        api["model"] = args.judge_model
    if args.judge_base_url:
        api["base_url"] = args.judge_base_url
        api.pop("extra_body", None)
    if args.concurrency:
        api["concurrency"] = int(args.concurrency)
    max_in = int(cfg_info["max_input_tokens"])

    def conv(r: Dict[str, Any]) -> List[Dict[str, str]]:
        src = (r.get("source") or "")[:max_in]
        ref = r.get("reference") or ""
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"【判决书原文】\n{src}\n\n【待审摘要】\n{ref}"}]

    if args.dry_run:
        print(json.dumps(conv(rows[0]), ensure_ascii=False)[:1200])
        return

    rt = build_api_runtime(api)
    audit_rows, clean_rows = [], []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        try:
            raws = rt.generate_messages([conv(r) for r in chunk], 512)
        except Exception as exc:  # noqa: BLE001
            print("批次失败，跳过：", exc)
            raws = [""] * len(chunk)
        for r, raw in zip(chunk, raws):
            d = parse(raw) or {}
            con = int(d.get("contradiction") or 0)
            sspan = str(d.get("summary_span") or "")
            xspan = str(d.get("source_span") or "")
            ok = con == 1 and covered(sspan, r.get("reference") or "") \
                and covered(xspan, r.get("source") or "")
            rec = {"id": str(r.get("id")), "contradiction": con,
                   "verified": int(ok), "summary_span": sspan,
                   "source_span": xspan, "reason": str(d.get("reason") or "")}
            audit_rows.append(rec)
            if not ok:            # 未被确认 → 保留（宁可漏杀）
                clean_rows.append(r)
        print(f"进度 {min(i + args.batch_size, len(rows))}/{len(rows)}")

    with open(f"{prefix}.jsonl", "w", encoding="utf-8") as f:
        for rec in audit_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(f"{prefix}.clean.jsonl", "w", encoding="utf-8") as f:
        for r in clean_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(audit_rows)
    v = sum(rec["verified"] for rec in audit_rows)
    print(f"审计 {n} 条：确认有问题 {v} 条（{v / max(1, n):.1%}），"
          f"保留 {len(clean_rows)} 条 → {prefix}.clean.jsonl")


if __name__ == "__main__":
    main()
