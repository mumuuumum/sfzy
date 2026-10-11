"""把**人工摘要当候选**再判一次事实一致性（FC），用于诊断 FC 的门控是否可信。

动机
====
新的事实一致性方案（``prompts.JUDGE_SYSTEM``）一次判定输出三样东西：

    {"has_factual_error": 0/1, "fact_consistency_score": 0~4, "error_reason": "..."}

其中 ``has_factual_error=1`` 会触发事实硬门控（整条 reward=0）。要判断这个门控
是否可信，最直接的办法是：**把人工摘要也当候选喂进同一个裁判**。人工摘要按定义
是正确答案，如果它也被判成"存在明确事实错误"，说明门控里有裁判噪声。

本脚本只调用 ``fact_consistency``（不碰 KEC / EE / INP），复用：

  * 配置里的 ``semantic.judge_api``（命令行可覆盖 base_url / model）
  * ``FactConsistencyJudge.judge_fact_batch``（内部用 prompts 的 system + parse_fact）

产物（默认 ``data/judge/fc_human_arm_<input_stem>/``）：
  * ``fc_scores.jsonl``  每条 (arm, group, index) 的整体分 / 门控 / 错误原因
  * ``summary.json``     逐臂 + 逐组对比统计（机器可读）
  * ``report.md``        人读报告

用法
====
    python scripts/probe_fc_human_arm.py \
        --config configs/grpo_fact_cov_ee_t4.yaml \
        --input data/triples/sft_val_100_n4.jsonl

    # 只判人工臂（省 API）
    python scripts/probe_fc_human_arm.py --arms human --limit 40

    # 换裁判（本机 key 属于别的服务时）
    python scripts/probe_fc_human_arm.py \
        --judge-base-url https://api.deepseek.com/v1 --judge-model deepseek-chat

    # 只重算报告：和一份旧的 scores.jsonl 做对比（不调 API）
    python scripts/probe_fc_human_arm.py --analysis-only \
        --compare-scores data/judge/metrics_sft_val_100_n4/scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import validate_reward_metrics as V                          # noqa: E402
import validate_iee_reward as base                           # noqa: E402

from sfzy.judge.api_runtime import build_api_runtime          # noqa: E402
from sfzy.judge.judge import FactConsistencyJudge            # noqa: E402
from sfzy.judge.schema import ELEMENTS, MAX_SCORE            # noqa: E402
from sfzy.utils.io import ensure_dir                         # noqa: E402
from sfzy.utils.logging import get_logger                    # noqa: E402

logger = get_logger("probe_fc_human_arm")


def resolve(p: Any) -> Path:
    return V.resolve(p)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------
def build_judge(cfg_info: Dict[str, Any], args: argparse.Namespace) -> FactConsistencyJudge:
    api = dict(cfg_info["judge_api"])
    if args.judge_base_url:
        api["base_url"] = args.judge_base_url
    if args.judge_model:
        api["model"] = args.judge_model
    if args.concurrency:
        api["concurrency"] = int(args.concurrency)
    # enable_thinking 之类的服务端私有字段只对特定服务有效，换服务商时去掉
    if args.judge_base_url:
        api.pop("extra_body", None)
    runtime = build_api_runtime(api)
    return FactConsistencyJudge(
        runtime=runtime,
        tasks=("fact_consistency",),
        six_max_new_tokens=int(cfg_info["six_max_new_tokens"]),
        max_input_tokens=int(cfg_info["max_input_tokens"]),
    )


def score_batch(judge: FactConsistencyJudge,
                items: Sequence[Tuple[str, str]]) -> List[Optional[Dict[str, Any]]]:
    """[(原文, 摘要)] → 每条 {fact_score, has_factual_error, error_reason, reward}。"""
    try:
        raws: List[Optional[Dict[str, Any]]] = list(judge.judge_fact_batch(list(items)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("批量判定失败（%d 条），退回逐条：%s", len(items), exc)
        raws = []
        for it in items:
            try:
                raws.append(judge.judge_fact_batch([it])[0])
            except Exception as exc2:  # noqa: BLE001
                logger.error("单条判定失败：%s", exc2)
                raws.append(None)
    out: List[Optional[Dict[str, Any]]] = []
    for raw in raws:
        if raw is None:
            out.append(None)
            continue
        score = int(raw.get("fact_consistency_score", 2))
        out.append({
            "fact_score": score,
            "has_factual_error": int(raw.get("has_factual_error", 0)),
            "error_reason": str(raw.get("error_reason") or ""),
            "reward": score / float(MAX_SCORE),
            "parse": raw.get("parse", "ok"),
        })
    return out


def run_scoring(args, groups, cfg_info, out_dir: Path) -> Path:
    scores_path = out_dir / "fc_scores.jsonl"
    done = {(r["group_id"], r["arm"], r["candidate_index"])
            for r in read_jsonl(scores_path)}
    jobs: List[Tuple[str, str, int, str, str]] = []       # gid, arm, idx, source, text
    for g in groups:
        if "human" in args.arms and (g["group_id"], "human", 0) not in done:
            jobs.append((g["group_id"], "human", 0, g["source"], g["reference"]))
        if "candidate" in args.arms:
            for i, cand in enumerate(g["candidates"]):
                if (g["group_id"], "candidate", i) not in done:
                    jobs.append((g["group_id"], "candidate", i, g["source"], cand))
    logger.info("待打分 %d 条（已完成 %d 条，arms=%s）", len(jobs), len(done), args.arms)
    if not jobs:
        return scores_path

    judge = build_judge(cfg_info, args)
    model = args.judge_model or cfg_info["judge_api"].get("model", "")
    started = time.time()
    written = 0
    with open(scores_path, "a", encoding="utf-8") as f:
        for s in range(0, len(jobs), args.batch_size):
            chunk = jobs[s:s + args.batch_size]
            res = score_batch(judge, [(src, txt) for _g, _a, _i, src, txt in chunk])
            for (gid, arm, idx, _src, txt), r in zip(chunk, res):
                if r is None:
                    continue
                rec = {
                    "group_id": gid, "arm": arm, "candidate_index": idx,
                    "judge_model": model,
                    "scored_at": datetime.now().isoformat(timespec="seconds"),
                    "chars": len(txt),
                    "fact_score": r["fact_score"],
                    "has_factual_error": r["has_factual_error"],
                    "error_reason": r["error_reason"],
                    "reward": r["reward"],
                    "parse": r["parse"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            el = time.time() - started
            rate = written / el if el else 0
            logger.info("进度 %d/%d | %.1f 条/秒 | 预计剩余 %.1f 分钟",
                        len(done) + written, len(done) + len(jobs), rate,
                        (len(jobs) - written) / rate / 60 if rate else float("inf"))
    logger.info("本轮写入 %d 条 → %s", written, scores_path)
    return scores_path


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------
def analyze(records: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)
    text_by_key = {}
    for g in groups:
        for i, cand in enumerate(g["candidates"]):
            text_by_key[(g["group_id"], "candidate", i)] = cand
        text_by_key[(g["group_id"], "human", 0)] = g["reference"]

    def arm_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(rows)
        out: Dict[str, Any] = {"n": n}
        if not n:
            return out
        scores = [int(r["fact_score"]) for r in rows]
        out["mean_reward"] = float(np.mean([r["reward"] for r in rows]))
        out["mean_fact_score"] = float(np.mean(scores))
        out["score_counts"] = {str(k): sum(1 for v in scores if v == k) for k in range(5)}
        out["error_count"] = sum(1 for r in rows if int(r["has_factual_error"]) == 1)
        out["error_rate"] = out["error_count"] / n
        out["fallback_ratio"] = sum(1 for r in rows if r.get("parse") == "fallback") / n
        return out

    summary: Dict[str, Any] = {"arms": {a: arm_stats(rows) for a, rows in by_arm.items()}}

    hum = {(r["group_id"],): r for r in by_arm.get("human", [])}
    cand = by_arm.get("candidate", [])
    comp: Dict[str, Any] = {"groups": len({r["group_id"] for r in cand})}
    pairs = [(r, hum.get((r["group_id"],))) for r in cand]
    pairs = [(c, h) for c, h in pairs if h is not None]
    if pairs:
        comp["n_pairs"] = len(pairs)
        comp["candidate_error"] = sum(1 for c, _h in pairs if int(c["has_factual_error"]) == 1)
        comp["human_error"] = sum(1 for _c, h in pairs if int(h["has_factual_error"]) == 1)
        comp["both_error"] = sum(1 for c, h in pairs
                                 if int(c["has_factual_error"]) == 1
                                 and int(h["has_factual_error"]) == 1)
        comp["mean_candidate"] = float(np.mean([c["reward"] for c, _h in pairs]))
        comp["mean_human"] = float(np.mean([h["reward"] for _c, h in pairs]))
        comp["human_gt_candidate_ratio"] = sum(
            1 for c, h in pairs if h["reward"] > c["reward"]) / len(pairs)
        # 候选被判错时，人工摘要的分数分布
        c_err = [(c, h) for c, h in pairs if int(c["has_factual_error"]) == 1]
        comp["candidate_error_human_scores"] = {
            str(k): sum(1 for _c, h in c_err if int(h["fact_score"]) == k) for k in range(5)
        }
        comp["candidate_error_human_error"] = sum(
            1 for _c, h in c_err if int(h["has_factual_error"]) == 1)
    summary["comparison"] = comp

    # 人工臂（金标准）被判"有事实错误"的组 —— 门控噪声的直接证据
    gold_err = [r for r in by_arm.get("human", []) if int(r["has_factual_error"]) == 1]
    summary["gold_errors"] = [
        {"group_id": r["group_id"], "fact_score": r["fact_score"],
         "error_reason": r["error_reason"], "chars": r["chars"]}
        for r in gold_err
    ]
    return summary


def load_existing_fc(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """读旧的 validate_reward_metrics scores.jsonl 的候选臂 FC 分。

    旧格式是六要素 ``sub_raw``；这里用"任一要素为 0"当作旧裁判的"有事实错误"代理。
    """
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in read_jsonl(path):
        fc = (r.get("metrics") or {}).get("fact_consistency")
        if not fc:
            continue
        subs = fc.get("sub_raw") or {}
        if not subs:
            continue
        out[(r["group_id"], r["candidate_index"])] = {
            "sub_raw": {e: int(subs[e]) for e in ELEMENTS if e in subs},
            "any_zero": any(int(subs[e]) == 0 for e in ELEMENTS if e in subs),
        }
    return out


def compare_judges(records: Sequence[Dict[str, Any]],
                   existing: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, Any]:
    """同一批候选、两个裁判的"有事实错误"判定对比。"""
    cand = [r for r in records if r["arm"] == "candidate"
            and (r["group_id"], r["candidate_index"]) in existing]
    both = only_old = only_new = neither = 0
    old_err_new_scores: Counter = Counter()
    for r in cand:
        old_err = existing[(r["group_id"], r["candidate_index"])]["any_zero"]
        new_err = int(r["has_factual_error"]) == 1
        if old_err and new_err:
            both += 1
        elif old_err:
            only_old += 1
            old_err_new_scores[int(r["fact_score"])] += 1
        elif new_err:
            only_new += 1
        else:
            neither += 1
    return {
        "n_candidates": len(cand),
        "both_error": both,
        "only_old_error": only_old,
        "only_new_error": only_new,
        "neither": neither,
        "old_error_new_score_dist": {str(k): old_err_new_scores.get(k, 0) for k in range(5)},
    }


def render_report(summary: Dict[str, Any]) -> str:
    arms = summary["arms"]
    meta = summary.get("meta") or {}
    L: List[str] = ["# FC 人工臂探针报告（人工摘要当候选再判一次）\n"]
    L.append(f"- 输入：{meta.get('input', '?')}（{meta.get('n_groups', '?')} 组）")
    L.append(f"- 裁判：`{meta.get('judge_model', '?')}` @ {meta.get('judge_base_url', '?')}")
    L.append(f"- arms：{list(arms)}\n")

    L.append("## 1. 两条臂：整体分与事实错误率\n")
    rows = []
    for arm in arms:
        a = arms[arm]
        dist = " / ".join(f"{k}:{a['score_counts'].get(str(k), 0)}" for k in range(5))
        rows.append([arm, a["n"], f"{a['mean_fact_score']:.3f}",
                     f"{a['mean_reward']:.4f}", a["error_count"],
                     f"{a['error_rate']:.1%}", f"{a['fallback_ratio']:.1%}", dist])
    L.append(base._table(["臂", "n", "整体分(0-4)", "归一奖励", "判错条数",
                          "错误率", "解析兜底率", "整体分分布(0/1/2/3/4)"], rows) + "\n")

    comp = summary.get("comparison") or {}
    if comp.get("n_pairs"):
        L.append("## 2. 候选 vs 人工（逐组配对）\n")
        L.append(f"- 配对 {comp['n_pairs']} 组；候选判错 {comp['candidate_error']} 条，"
                 f"人工摘要判错 {comp['human_error']} 条，两个都判错 {comp['both_error']} 条。")
        L.append(f"- 整条奖励：候选均值 {comp['mean_candidate']:.4f}，"
                 f"人工均值 {comp['mean_human']:.4f}，"
                 f"人工更高的比例 {comp['human_gt_candidate_ratio']:.1%}")
        dist = comp.get("candidate_error_human_scores") or {}
        L.append("- **候选被判“有事实错误”时，人工摘要的整体分分布**：" +
                 " / ".join(f"{k}:{dist.get(str(k), 0)}" for k in range(5)) +
                 f"；其中人工摘要也被判错的：{comp.get('candidate_error_human_error', 0)} 条\n")

    gold = summary.get("gold_errors") or []
    if gold:
        L.append(f"## 3. 人工摘要（金标准）被判“存在事实错误”的组：{len(gold)} 个\n")
        for g in gold[:20]:
            L.append(f"- `{g['group_id']}` 整体分={g['fact_score']} {g['error_reason']}")
        if len(gold) > 20:
            L.append(f"- ……其余 {len(gold) - 20} 个见 `summary.json`")
        L.append("")

    cmp_ = summary.get("compare")
    if cmp_:
        L.append("## 4. 与旧格式结果的对比（旧口径 = 六要素任一为 0）\n")
        L.append(f"- 对比候选数 {cmp_['n_candidates']}：两个都判错 {cmp_['both_error']}，"
                 f"只有旧裁判判错 **{cmp_['only_old_error']}**，只有本探针判错 "
                 f"{cmp_['only_new_error']}，都没判错 {cmp_['neither']}")
        dist = cmp_.get("old_error_new_score_dist") or {}
        L.append("- 旧裁判判错时，本探针给的整体分分布：" +
                 " / ".join(f"{k}:{dist.get(str(k), 0)}" for k in range(5)) + "\n")
    return "\n".join(L) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FC 人工臂探针（人工摘要当候选再判一次）")
    ap.add_argument("--config", default="configs/grpo_fact_cov_ee_t4.yaml")
    ap.add_argument("--input", default="data/triples/sft_val_100_n4.jsonl")
    ap.add_argument("--candidates-key", default="outputs")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--arms", nargs="+", default=["candidate", "human"],
                    choices=["candidate", "human"])
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 组")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--judge-base-url", default=None, help="覆盖配置里的 base_url")
    ap.add_argument("--judge-model", default=None, help="覆盖配置里的 model")
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--compare-scores", default=None,
                    help="一份 validate_reward_metrics 的 scores.jsonl，做跨裁判对比")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg_path, input_path = resolve(args.config), resolve(args.input)
    out_dir = resolve(args.out_dir) if args.out_dir else resolve(f"data/judge/fc_human_arm_{input_path.stem}")
    ensure_dir(out_dir)

    _weights, cfg_info, _rctx = V.load_config_bundle(cfg_path, ["fact_consistency"])
    groups = V.load_groups(input_path, args.candidates_key, args.limit)
    logger.info("读入 %d 组，候选 %d 条", len(groups), sum(len(g["candidates"]) for g in groups))

    scores_path = out_dir / "fc_scores.jsonl"
    if not args.analysis_only:
        run_scoring(args, groups, cfg_info, out_dir)
    records = read_jsonl(scores_path)
    if not records:
        raise SystemExit(f"没有打分结果 {scores_path}")

    summary = analyze(records, groups)
    if args.compare_scores:
        cmp_path = resolve(args.compare_scores)
        existing = load_existing_fc(cmp_path)
        if existing:
            summary["compare"] = compare_judges(records, existing)
            summary["compare"]["existing_path"] = str(cmp_path)
    summary["meta"] = {
        "input": str(input_path),
        "n_groups": len(groups),
        "arms": args.arms,
        "judge_model": args.judge_model or cfg_info["judge_api"].get("model"),
        "judge_base_url": args.judge_base_url or cfg_info["judge_api"].get("base_url"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    logger.info("逐条：%s\n统计：%s\n报告：%s", scores_path,
                out_dir / "summary.json", out_dir / "report.md")


if __name__ == "__main__":
    main()
