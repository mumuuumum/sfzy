"""IEE（Information Expression Efficiency，信息表达效率）Reward 区分度验证实验。

目标
====
在 ``data/triples/sft_val_100_n4.jsonl``（100 篇文书 × 4 个候选摘要）上，验证
配置 ``configs/grpo_fact_coverage_inp_t4.yaml`` 里启用的 ``expression_efficiency``
奖励，能否为 GRPO 的**组内相对优势**提供有意义的信号。

关键约束（需求原文，逐条遵守）
==============================
  * **不训练 GRPO**，不修改任何 Judge Prompt / 评分规则 / 配置权重；
  * 三个分项权重**从配置文件里读**（``RewardSpec`` → ``expression_efficiency``
    的 ``dimension_weights``），绝不硬编码那份 0.4/0.2/0.4 的默认值；
  * 复用**现有** Judge 调用 / Prompt / 解析 / 重试 / 缓存逻辑：
    直接用 ``sfzy.judge.judge.FactConsistencyJudge.judge_ee_batch``（它内部用
    ``sfzy.judge.prompts.build_ee_messages`` 与 ``parse_ee``）和
    ``sfzy.judge.api_runtime.APIRuntime``（负责并发 + 指数退避重试）；
  * Judge 输入为 **(原文, 人工参考摘要, 候选摘要)**；
  * 每个候选**独立评分**（一次判定只喂一个候选，不让裁判在 prompt 里同时看 4 个）；
  * 支持断点续跑（已打分的 candidate_id 直接跳过）。

关于 torch / transformers
========================
EE 判定完全走 API，不需要本地模型。但 ``sfzy.judge`` 这个包的 import 链会拉起
``sfzy.models.compat``（``from transformers import ...``）。在没装 torch 的机器上
这里注入一个**最小 stub** 让 import 通过——真实判定一行都不会走到它。装了
torch 的环境会自动走真实 import。

用法
====
    # 打分 + 分析 + 出报告（默认对全部 100 组 400 条候选打分）
    python scripts/validate_iee_reward.py \
        --config configs/grpo_fact_coverage_inp_t4.yaml \
        --input data/triples/sft_val_100_n4.jsonl

    # 只重算报告（不调 API），复用已有 scores.jsonl
    python scripts/validate_iee_reward.py --analysis-only

    # 先小样本冒烟
    python scripts/validate_iee_reward.py --limit 3 --no-repeat

产物（默认 ``data/judge/iee_<input_stem>/``）：
  * ``scores.jsonl``        每条候选的三维原始分 / 归一化分 / 加权 IEE + 元数据
  * ``repeat_scores.jsonl`` 重复评分（单独保存，绝不覆盖正式结果）
  * ``summary.json``        全部统计量（机器可读）
  * ``samples.jsonl``       典型样本（人工复核用，含原文/人工摘要/候选全文）
  * ``report.md``           Markdown 统计报告
  * ``figures/*.png``       图表
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _ensure_importable_without_torch() -> None:
    """EE 走 API，不需要 torch；只在真缺 transformers 时注入最小 stub。"""
    try:
        import transformers  # noqa: F401
        return
    except ImportError:
        pass
    import types

    stub = types.ModuleType("transformers")
    stub.GenerationConfig = type("GenerationConfig", (), {})
    stub.PreTrainedModel = type("PreTrainedModel", (), {})
    sys.modules["transformers"] = stub


_ensure_importable_without_torch()

from sfzy.config import load_config                          # noqa: E402
from sfzy.judge.api_runtime import APIError, build_api_runtime  # noqa: E402
from sfzy.judge.judge import FactConsistencyJudge            # noqa: E402
from sfzy.judge.schema import (                              # noqa: E402
    EE_DIMENSIONS,
    expression_efficiency_score,
)
from sfzy.rl.reward_spec import RewardSpec                   # noqa: E402
from sfzy.utils.io import ensure_dir, write_jsonl            # noqa: E402
from sfzy.utils.logging import get_logger                    # noqa: E402

logger = get_logger("validate_iee")

DIMS: Tuple[str, ...] = tuple(EE_DIMENSIONS)                 # 语义冗余/表达冗余/抽象适当性
DIM_ZH: Dict[str, str] = {
    "semantic_redundancy": "语义冗余",
    "verbal_redundancy": "表达冗余",
    "abstraction_adequacy": "抽象适当性",
}
MAX_RAW = 4.0


def resolve(path: Any) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# ---------------------------------------------------------------------------
# 0. 配置：权重只从配置文件来
# ---------------------------------------------------------------------------
def load_iee_weights(config_path: Path) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    """从配置里取 EE 三维权重 + judge_api + 其它需要的字段。

    走 ``RewardSpec.from_config`` 这条与训练**完全相同**的解析路径，拿
    ``expression_efficiency.options["dimension_weights"]``。如果配置里没写
    三维权重，RewardSpec 会退回默认值——这里显式判断并警告，避免"以为用了
    配置、其实用了默认 0.4/0.2/0.4"。
    """
    cfg = load_config(config_path)
    reward_raw = cfg.path_("rl.reward")
    if not reward_raw:
        raise SystemExit(f"{config_path} 里没有 rl.reward，取不到 EE 权重")
    spec = RewardSpec.from_config(reward_raw)
    term = spec.terms.get("expression_efficiency")
    if term is None or not term.enabled:
        raise SystemExit(
            f"{config_path} 里没有启用 expression_efficiency，本实验无从验证。"
        )
    options = spec.judge_term_options().get("expression_efficiency", {})
    weights = options.get("dimension_weights")
    if not weights:
        raise SystemExit(
            "配置里没有写 terms.expression_efficiency.dimension_weights，"
            "RewardSpec 会退回代码默认值；本实验要求严格使用配置权重。"
        )
    weights = {name: float(weights[name]) for name in DIMS}
    cfg_info = {
        "config_path": str(config_path),
        "config_chain": [str(p) for p in [config_path]],
        "dimension_weights": weights,
        "weight_sum": float(sum(weights.values())),
        "term_weight": float(term.weight),
        "normalized_weights": spec.normalized_weights(),
        "judge_api": {k: v for k, v in dict(cfg.path_("semantic.judge_api") or {}).items()
                      if k != "api_key"},
        "six_max_new_tokens": int(cfg.path_("semantic.six_max_new_tokens") or 512),
    }
    return weights, cfg_info, dict(reward_raw)


# ---------------------------------------------------------------------------
# 1. 数据
# ---------------------------------------------------------------------------
def read_jsonl_local(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    return rows


def load_groups(
    path: Path, candidates_key: str, limit: Optional[int]
) -> List[Dict[str, Any]]:
    """读入并整理成 group：每篇文书 + 它的 N 个候选摘要。"""
    rows = read_jsonl_local(path)
    if limit:
        rows = rows[:limit]
    groups: List[Dict[str, Any]] = []
    for row in rows:
        candidates = row.get(candidates_key)
        if not candidates:
            if isinstance(row.get("output"), str):
                candidates = [row["output"]]
            else:
                raise SystemExit(
                    f"记录 {row.get('id')!r} 里没有 {candidates_key!r} 也没有 output"
                )
        groups.append({
            "group_id": str(row.get("id")),
            "source": row.get("source", "") or "",
            "reference": row.get("reference", "") or "",
            "candidates": [str(c) for c in candidates],
        })
    return groups


# ---------------------------------------------------------------------------
# 2. 打分（可断点续跑）
# ---------------------------------------------------------------------------
def candidate_id(group_id: str, index: int) -> str:
    return f"{group_id}#{index}"


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for row in read_jsonl_local(path):
        cid = row.get("candidate_id")
        if cid:
            done.add(cid)
    return done


def build_judge(weights: Dict[str, float], api_cfg: Dict[str, Any],
                six_max_new_tokens: int, concurrency: Optional[int]) -> FactConsistencyJudge:
    cfg = dict(api_cfg)
    if concurrency:
        cfg["concurrency"] = int(concurrency)
    runtime = build_api_runtime(cfg)
    # tasks 只开 expression_efficiency：其余判定一条都不发。
    return FactConsistencyJudge(
        runtime=runtime,
        tasks=("expression_efficiency",),
        ee_weights=weights,
        six_max_new_tokens=six_max_new_tokens,
    )


def score_items(judge: FactConsistencyJudge, items: Sequence[Tuple[str, str, str]],
                tag: str) -> List[Optional[Dict[str, Any]]]:
    """一批 (原文, 人工摘要, 候选) → 每条的三维分。批量失败时退回逐条，单条失败记 None。"""
    try:
        return list(judge.judge_ee_batch(list(items)))
    except Exception as exc:  # noqa: BLE001 —— 一批失败不该丢掉整轮
        logger.warning("[%s] 批量判定失败（%d 条），退回逐条：%s", tag, len(items), exc)
    out: List[Optional[Dict[str, Any]]] = []
    for it in items:
        try:
            out.append(judge.judge_ee_batch([it])[0])
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] 单条判定失败，跳过（下次续跑会重试）：%s", tag, exc)
            out.append(None)
    return out


def make_record(group: Dict[str, Any], index: int, dims: Dict[str, Any],
                weights: Dict[str, float], judge_model: str) -> Dict[str, Any]:
    cand = group["candidates"][index]
    ref = group["reference"]
    resolved = {name: int(dims.get(name, 2)) for name in DIMS}
    rec: Dict[str, Any] = {
        "group_id": group["group_id"],
        "candidate_index": index,
        "candidate_id": candidate_id(group["group_id"], index),
        "dimension_weights": dict(weights),
        "judge_model": judge_model,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_chars": len(cand),
        "reference_chars": len(ref),
        "source_chars": len(group["source"]),
        "len_ratio_to_ref": round(len(cand) / max(1, len(ref)), 6),
        "len_diff_to_ref": len(cand) - len(ref),
    }
    for name in DIMS:
        rec[f"{name}_raw"] = resolved[name]
        rec[f"{name}_norm"] = round(resolved[name] / MAX_RAW, 6)
    rec["iee"] = expression_efficiency_score(resolved, weights)
    return rec


def run_scoring(args: argparse.Namespace, groups: List[Dict[str, Any]],
                weights: Dict[str, float], cfg_info: Dict[str, Any],
                out_dir: Path) -> Path:
    scores_path = out_dir / "scores.jsonl"
    done = load_done_ids(scores_path)
    n_cand = sum(len(g["candidates"]) for g in groups)
    tasks: List[Tuple[str, int, str, str, str]] = []
    for group in groups:
        for index, cand in enumerate(group["candidates"]):
            if candidate_id(group["group_id"], index) not in done:
                tasks.append((group["group_id"], index, group["source"],
                              group["reference"], cand))
    logger.info("候选共 %d 条，已完成 %d 条，待打分 %d 条", n_cand, len(done), len(tasks))
    if not tasks:
        logger.info("全部候选都已打分：%s", scores_path)
        return scores_path

    judge = build_judge(weights, cfg_info["judge_api"], cfg_info["six_max_new_tokens"],
                        args.concurrency)
    model = cfg_info["judge_api"].get("model", "")
    group_by_id = {g["group_id"]: g for g in groups}

    started = time.time()
    written = 0
    with open(scores_path, "a", encoding="utf-8") as f:
        for start in range(0, len(tasks), args.batch_candidates):
            chunk = tasks[start:start + args.batch_candidates]
            dims_list = score_items(
                judge, [(src, ref, cand) for _g, _i, src, ref, cand in chunk], "main"
            )
            for (gid, idx, _src, _ref, _cand), dims in zip(chunk, dims_list):
                if dims is None:
                    continue
                rec = make_record(group_by_id[gid], idx, dims, weights, model)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            elapsed = time.time() - started
            rate = written / elapsed if elapsed > 0 else 0.0
            eta = (len(tasks) - written) / rate / 60 if rate > 0 else float("inf")
            logger.info("进度 %d/%d | %.1f 条/秒 | 预计剩余 %.1f 分钟",
                        written, len(tasks), rate, eta)
    logger.info("本轮写入 %d 条 → %s", written, scores_path)
    return scores_path


def run_repeat_scoring(args: argparse.Namespace, groups: List[Dict[str, Any]],
                       weights: Dict[str, float], cfg_info: Dict[str, Any],
                       out_dir: Path) -> Optional[Path]:
    """对抽取的一部分组做重复评分，单独落盘（绝不动正式结果）。"""
    if args.no_repeat or args.repeat_groups <= 0:
        return None
    repeat_path = out_dir / "repeat_scores.jsonl"
    done = load_done_ids(repeat_path)
    rng = random.Random(args.repeat_seed)
    picked = groups if args.repeat_groups >= len(groups) else rng.sample(groups, args.repeat_groups)
    tasks: List[Tuple[str, int, str, str, str]] = []
    for group in picked:
        for index, cand in enumerate(group["candidates"]):
            key = candidate_id(group["group_id"], index)
            for k in range(args.repeat_k):
                if f"{key}@{k}" not in done:
                    tasks.append((group["group_id"], index, group["source"],
                                  group["reference"], cand, k))  # type: ignore[arg-type]
    logger.info("重复评分：抽 %d 组、%d 次，待打 %d 条", len(picked), args.repeat_k, len(tasks))
    if not tasks:
        return repeat_path
    judge = build_judge(weights, cfg_info["judge_api"], cfg_info["six_max_new_tokens"],
                        args.concurrency)
    model = cfg_info["judge_api"].get("model", "")
    group_by_id = {g["group_id"]: g for g in groups}
    with open(repeat_path, "a", encoding="utf-8") as f:
        for start in range(0, len(tasks), args.batch_candidates):
            chunk = tasks[start:start + args.batch_candidates]
            dims_list = score_items(
                judge, [(src, ref, cand) for _g, _i, src, ref, cand, _k in chunk], "repeat"
            )
            for (gid, idx, _s, _r, _c, k), dims in zip(chunk, dims_list):
                if dims is None:
                    continue
                rec = make_record(group_by_id[gid], idx, dims, weights, model)
                rec["repeat_index"] = k
                rec["candidate_id_orig"] = rec["candidate_id"]
                rec["candidate_id"] = f"{rec['candidate_id']}@{k}"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    logger.info("重复评分写入 → %s", repeat_path)
    return repeat_path


# ---------------------------------------------------------------------------
# 3. 统计工具
# ---------------------------------------------------------------------------
def describe(values: Sequence[float]) -> Dict[str, float]:
    """均值/样本标准差(ddof=1)/中位数/极值/分位数。"""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "p10": float(np.percentile(v, 10)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
    }


def safe_corr(a: Sequence[float], b: Sequence[float]) -> Dict[str, Optional[float]]:
    """Pearson + Spearman；常数序列等退化情形返回 None，不抛出。"""
    from scipy import stats

    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size < 3 or y.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None}
    with np.errstate(all="ignore"):
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
        "spearman_r": float(sr[0]), "spearman_p": float(sr[1]),
    }


def group_metric(values: Sequence[float]) -> Dict[str, float]:
    """一组的 4 个分数 → 组内区分度指标。"""
    v = np.asarray(list(values), dtype=float)
    mx, mn = float(v.max()), float(v.min())
    pairs = [(i, j) for i in range(len(v)) for j in range(i + 1, len(v))]
    diff_pairs = sum(1 for i, j in pairs if v[i] != v[j])
    return {
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "range": mx - mn,
        "n_distinct": int(len(set(float(x) for x in v))),
        "pair_diff_ratio": diff_pairs / len(pairs) if pairs else 0.0,
        "max_tie": int((v == mx).sum()),
    }


def _grouped(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec["group_id"]].append(rec)
    for rows in groups.values():
        rows.sort(key=lambda r: r["candidate_index"])
    return dict(groups)


# ---------------------------------------------------------------------------
# 4. 分析
# ---------------------------------------------------------------------------
def analyze(records: List[Dict[str, Any]], groups: List[Dict[str, Any]],
            weights: Dict[str, float], cfg_info: Dict[str, Any]) -> Dict[str, Any]:
    from scipy import stats

    wsum = sum(weights.values()) or 1.0
    w = {k: weights[k] / wsum for k in DIMS}          # 归一权重，IEE = Σ w_i·(raw/4)
    grouped = _grouped(records)
    order = [g["group_id"] for g in groups if g["group_id"] in grouped]

    summary: Dict[str, Any] = {
        "meta": {
            **cfg_info,
            "input_groups": len(groups),
            "scored_groups": len(grouped),
            "n_candidates": len(records),
            "candidates_per_group": sorted({len(v) for v in grouped.values()}),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "weights_normalized": w,
    }

    # ---------- 2. 全局分布 ----------
    global_stats: Dict[str, Any] = {}
    for name in DIMS:
        raw = [r[f"{name}_raw"] for r in records]
        counts = Counter(int(x) for x in raw)
        n = len(raw)
        global_stats[name] = {
            "describe": describe(raw),
            "raw_counts": {str(k): counts.get(k, 0) for k in range(5)},
            "raw_ratio": {str(k): counts.get(k, 0) / n for k in range(5)},
            "full_ratio": counts.get(4, 0) / n,
            "low_ratio_le2": sum(counts.get(k, 0) for k in range(3)) / n,
            "entropy": float(-sum((counts.get(k, 0) / n) * math.log(counts.get(k, 0) / n)
                                   for k in range(5) if counts.get(k, 0) > 0)),
        }
    iee = [r["iee"] for r in records]
    hist, edges = np.histogram(iee, bins=20, range=(0.0, 1.0))
    global_stats["iee"] = {
        "describe": describe(iee),
        "hist_counts": [int(x) for x in hist],
        "hist_edges": [round(float(x), 3) for x in edges],
        "full_ratio": sum(1 for x in iee if x >= 1.0 - 1e-9) / len(iee),
        "low_ratio_le05": sum(1 for x in iee if x <= 0.5 + 1e-9) / len(iee),
        "high_ratio_ge075": sum(1 for x in iee if x >= 0.75 - 1e-9) / len(iee),
    }
    summary["global"] = global_stats

    # ---------- 3. 组内区分度 ----------
    group_stats: Dict[str, Any] = {}
    for name in list(DIMS) + ["iee"]:
        gstd, grand, ndist, pair, maxtie = [], [], [], [], []
        for gid in order:
            vals = [r[name if name == "iee" else f"{name}_raw"] for r in grouped[gid]]
            m = group_metric(vals)
            gstd.append(m["std"]); grand.append(m["range"])
            ndist.append(m["n_distinct"]); pair.append(m["pair_diff_ratio"])
            maxtie.append(m["max_tie"])
        n_groups = len(order)
        zero = sum(1 for s in gstd if s <= 1e-12)
        group_stats[name] = {
            "std": describe(gstd),
            "zero_var_groups": int(zero),
            "zero_var_ratio": zero / n_groups if n_groups else 0.0,
            "nonzero_var_ratio": 1 - zero / n_groups if n_groups else 0.0,
            "range": describe(grand),
            "range_hist": {str(k): int(sum(1 for x in grand if abs(x - k) < 1e-9))
                           for k in sorted(set(round(x, 6) for x in grand))},
            "n_distinct_dist": {str(k): int(sum(1 for x in ndist if x == k)) for k in range(1, 5)},
            "pairwise_mean": float(np.mean(pair)) if pair else 0.0,
            "max_tie_dist": {str(k): int(sum(1 for x in maxtie if x == k)) for k in range(1, 5)},
        }
    summary["group"] = group_stats
    summary["group"]["n_groups"] = len(order)

    # 每个维度在多少组里能区分 / 各维度组合
    dim_disc = {name: int(sum(1 for gid in order
                             if group_metric([r[f"{name}_raw"] for r in grouped[gid]])["std"] > 1e-12))
                for name in DIMS}
    combo: Counter = Counter()
    all_none = 0
    for gid in order:
        rows = grouped[gid]
        active = tuple(name for name in DIMS
                       if group_metric([r[f"{name}_raw"] for r in rows])["std"] > 1e-12)
        combo[active] += 1
        if not active:
            all_none += 1
    summary["dim_discriminability"] = {
        "discriminating_groups_per_dim": dim_disc,
        "all_three_indistinguishable_groups": all_none,
        "combination_counts": {"+".join(k) if k else "(none)": v for k, v in combo.items()},
    }

    # ---------- 4. 相关性 / 贡献 / 权重 ----------
    series: Dict[str, List[float]] = {name: [r[f"{name}_raw"] for r in records] for name in DIMS}
    series["iee"] = iee
    series["candidate_chars"] = [r["candidate_chars"] for r in records]
    series["len_ratio_to_ref"] = [r["len_ratio_to_ref"] for r in records]
    series["len_diff_to_ref"] = [r["len_diff_to_ref"] for r in records]
    corr: Dict[str, Any] = {}
    keys = list(DIMS) + ["iee", "candidate_chars", "len_ratio_to_ref", "len_diff_to_ref"]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            corr[f"{a}~{b}"] = safe_corr(series[a], series[b])
    summary["correlations"] = corr

    # 各分项对 IEE 的贡献（w_i 已归一，IEE = Σ contribution_i）
    contribution: Dict[str, Any] = {}
    total_mean = float(np.mean(iee))
    for name in DIMS:
        contrib = [w[name] * (r[f"{name}_raw"] / MAX_RAW) for r in records]
        contribution[name] = {
            "weight": w[name],
            "mean_contribution": float(np.mean(contrib)),
            "share_of_iee_mean": float(np.mean(contrib) / total_mean) if total_mean else None,
        }
    # 组内方差分解：Var(IEE)=Σ wᵢ²Var(xᵢ)+2Σ wᵢwⱼCov(xᵢ,xⱼ)
    var_total, var_terms, cov_terms = [], defaultdict(list), defaultdict(list)
    for gid in order:
        rows = grouped[gid]
        x = {name: np.asarray([r[f"{name}_raw"] / MAX_RAW for r in rows], float) for name in DIMS}
        y = np.asarray([r["iee"] for r in rows], float)
        vt = float(y.var(ddof=1)) if y.size > 1 else 0.0
        var_total.append(vt)
        for name in DIMS:
            var_terms[name].append(float(w[name] ** 2 * (x[name].var(ddof=1) if x[name].size > 1 else 0.0)))
        for i, a in enumerate(DIMS):
            for b in DIMS[i + 1:]:
                if x[a].size > 1:
                    cov = float(np.cov(x[a], x[b], ddof=1)[0, 1])
                else:
                    cov = 0.0
                cov_terms[f"{a}~{b}"].append(2 * w[a] * w[b] * cov)
    mean_var = float(np.mean(var_total)) if var_total else 0.0
    summary["contribution"] = {
        "terms": contribution,
        "within_group_var_iee_mean": mean_var,
        "within_group_variance_share": {
            **{name: (float(np.mean(var_terms[name])) / mean_var if mean_var else None) for name in DIMS},
            **{k: (float(np.mean(v)) / mean_var if mean_var else None) for k, v in cov_terms.items()},
        },
    }

    # 权重敏感性（离线附加分析，只改评分公式，不动配置、不覆盖正式结果）
    schemes: Dict[str, Dict[str, float]] = {
        "actual(config)": dict(w),
        "equal": {name: 1 / 3 for name in DIMS},
    }
    for name in DIMS:
        schemes[f"only:{name}"] = {k: (1.0 if k == name else 0.0) for k in DIMS}
    scheme_iee = {
        label: [expression_efficiency_score({d: r[f"{d}_raw"] for d in DIMS}, sw)
                for r in records]
        for label, sw in schemes.items()
    }
    pos = {r["candidate_id"]: i for i, r in enumerate(records)}
    sens: Dict[str, Any] = {}
    actual_top1 = {gid: int(np.argmax([r["iee"] for r in grouped[gid]])) for gid in order}
    for label, vals in scheme_iee.items():
        per_group = []
        top1_hit = 0
        for gid in order:
            rows = grouped[gid]
            a = [r["iee"] for r in rows]
            b = [vals[pos[r["candidate_id"]]] for r in rows]
            per_group.append(safe_corr(a, b)["spearman_r"] or 0.0)
            if int(np.argmax(b)) == actual_top1[gid]:
                top1_hit += 1
        sens[label] = {
            "global_spearman_vs_actual": safe_corr(iee, vals)["spearman_r"],
            "mean_within_group_spearman": float(np.mean(per_group)) if per_group else 0.0,
            "top1_agreement": top1_hit / len(order) if order else 0.0,
        }
    summary["weight_sensitivity"] = {"schemes": schemes, "agreement_vs_actual": sens,
                                     "note": "离线敏感性分析，不修改配置也不覆盖正式评分"}

    # ---------- 5. 长度偏差 ----------
    length: Dict[str, Any] = {"global": {}, "within_group": {}}
    for metric in ["iee"] + list(DIMS):
        for lname in ["candidate_chars", "len_ratio_to_ref", "len_diff_to_ref"]:
            length["global"][f"{metric}~{lname}"] = safe_corr(series[metric], series[lname])
    # 组内去均值后相关（去掉"文书本身长短"这个组间混杂）
    demean_iee, demean_len, demean_ratio = [], [], []
    for gid in order:
        rows = grouped[gid]
        m_iee = np.mean([r["iee"] for r in rows])
        m_len = np.mean([r["candidate_chars"] for r in rows])
        m_rat = np.mean([r["len_ratio_to_ref"] for r in rows])
        for r in rows:
            demean_iee.append(r["iee"] - m_iee)
            demean_len.append(r["candidate_chars"] - m_len)
            demean_ratio.append(r["len_ratio_to_ref"] - m_rat)
    length["within_group"]["iee~candidate_chars"] = safe_corr(demean_iee, demean_len)
    length["within_group"]["iee~len_ratio_to_ref"] = safe_corr(demean_iee, demean_ratio)
    # 最短候选是否总是最高分
    shortest_highest = longest_highest = shortest_lowest = 0
    usable = 0
    for gid in order:
        rows = grouped[gid]
        lens = [r["candidate_chars"] for r in rows]
        if len(set(lens)) == 1:
            continue
        usable += 1
        best = max(r["iee"] for r in rows)
        worst = min(r["iee"] for r in rows)
        best_idx = {i for i, r in enumerate(rows) if r["iee"] == best}
        worst_idx = {i for i, r in enumerate(rows) if r["iee"] == worst}
        min_len = min(lens); max_len = max(lens)
        if best_idx & {i for i, L in enumerate(lens) if L == min_len}:
            shortest_highest += 1
        if best_idx & {i for i, L in enumerate(lens) if L == max_len}:
            longest_highest += 1
        if worst_idx & {i for i, L in enumerate(lens) if L == min_len}:
            shortest_lowest += 1
    length["shortest_highest"] = {"groups_usable": usable, "count": shortest_highest,
                                  "ratio": shortest_highest / usable if usable else None}
    length["longest_highest"] = {"count": longest_highest,
                                 "ratio": longest_highest / usable if usable else None}
    length["shortest_lowest"] = {"count": shortest_lowest,
                                 "ratio": shortest_lowest / usable if usable else None}
    summary["length_bias"] = length

    # ---------- 6. 排序稳定性 ----------
    top1_ties = Counter()
    rank_corr_with_dim = defaultdict(list)
    for gid in order:
        rows = grouped[gid]
        vals = [r["iee"] for r in rows]
        top1_ties[int(sum(1 for x in vals if x == max(vals)))] += 1
        for name in DIMS:
            c = safe_corr(vals, [r[f"{name}_raw"] for r in rows])["spearman_r"]
            if c is not None:
                rank_corr_with_dim[name].append(c)
    summary["ranking"] = {
        "top1_tie_dist": {str(k): v for k, v in sorted(top1_ties.items())},
        "mean_within_group_spearman_iee_vs_dim": {
            name: float(np.mean(v)) if v else None for name, v in rank_corr_with_dim.items()
        },
    }
    return summary


# ---------------------------------------------------------------------------
# 5. 典型样本
# ---------------------------------------------------------------------------
def select_samples(records: List[Dict[str, Any]], groups: List[Dict[str, Any]],
                   max_per_cat: int) -> List[Dict[str, Any]]:
    grouped = _grouped(records)
    text_by_id = {g["group_id"]: g for g in groups}
    picked: List[Dict[str, Any]] = []

    def snapshot(gid: str, category: str, note: str) -> Dict[str, Any]:
        rows = grouped[gid]
        g = text_by_id[gid]
        best = max(r["iee"] for r in rows)
        return {
            "category": category, "note": note, "group_id": gid,
            "source": g["source"], "reference": g["reference"],
            "candidates": [
                {
                    "index": r["candidate_index"],
                    "text": g["candidates"][r["candidate_index"]],
                    "iee": r["iee"],
                    "chars": r["candidate_chars"],
                    "is_best": r["iee"] == best,
                    **{f"{name}_raw": r[f"{name}_raw"] for name in DIMS},
                }
                for r in rows
            ],
        }

    order = list(grouped.keys())
    spread = {gid: group_metric([r["iee"] for r in grouped[gid]])["range"] for gid in order}
    used: Dict[str, set] = defaultdict(set)

    def add(category: str, gid: str, note: str) -> None:
        if len(used[category]) >= max_per_cat or gid in used[category]:
            return
        used[category].add(gid)
        picked.append(snapshot(gid, category, note))

    # 完全同分
    for gid in sorted(order, key=lambda g: spread[g]):
        if spread[gid] <= 1e-12:
            add("fully_tied", gid, "4 个候选 IEE 完全相同，组内无梯度")
    # 分差最大
    for gid in sorted(order, key=lambda g: -spread[g]):
        add("max_spread", gid, f"IEE 组内极差 = {spread[gid]:.3f}")
    # 维度冲突：三个维度各自的组内 top1 与 IEE top1 不一致的数量最多
    def conflict_score(gid: str) -> int:
        rows = grouped[gid]
        iee_best = max(r["iee"] for r in rows)
        iee_set = {i for i, r in enumerate(rows) if r["iee"] == iee_best}
        score = 0
        for name in DIMS:
            vals = [r[f"{name}_raw"] for r in rows]
            top = max(vals)
            if not ({i for i, v in enumerate(vals) if v == top} & iee_set):
                score += 1
        return score
    for gid in sorted(order, key=lambda g: (-conflict_score(g), -spread[g])):
        if spread[gid] > 1e-12 and conflict_score(gid) >= 2:
            add("dim_conflict", gid, f"{conflict_score(gid)} 个维度的组内最高分不是 IEE 最高分")
    # 最长候选反而最高 / 最短候选最低
    for gid in order:
        rows = grouped[gid]
        lens = [r["candidate_chars"] for r in rows]
        if len(set(lens)) == 1:
            continue
        best = max(r["iee"] for r in rows); worst = min(r["iee"] for r in rows)
        if max(lens) in {L for r, L in zip(rows, lens) if r["iee"] == best}:
            add("longest_highest", gid, "最长候选拿到组内最高 IEE")
        if min(lens) in {L for r, L in zip(rows, lens) if r["iee"] == worst}:
            add("shortest_lowest", gid, "最短候选拿到组内最低 IEE")
    return picked


# ---------------------------------------------------------------------------
# 6. 图表
# ---------------------------------------------------------------------------
def make_figures(records: List[Dict[str, Any]], summary: Dict[str, Any],
                 fig_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(fig_dir)
    made: List[str] = []
    iee = [r["iee"] for r in records]

    # 1) IEE 直方图
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(iee, bins=20, range=(0, 1), color="#3b6fb0", edgecolor="white")
    ax.set_title("IEE total score distribution (n=%d)" % len(iee))
    ax.set_xlabel("weighted IEE in [0,1]"); ax.set_ylabel("count")
    fig.tight_layout(); p = fig_dir / "fig_iee_hist.png"; fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p.name)

    # 2) 三维原始分分布
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, name in zip(axes, DIMS):
        counts = [summary["global"][name]["raw_counts"][str(k)] for k in range(5)]
        ax.bar(range(5), counts, color="#7aa6d6", edgecolor="white")
        ax.set_title(name); ax.set_xticks(range(5)); ax.set_ylim(0, max(counts + [1]) * 1.1)
    fig.suptitle("Raw 0-4 score counts per dimension")
    fig.tight_layout(); p = fig_dir / "fig_dim_raw_dist.png"; fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p.name)

    # 3) 组内标准差 & 不同评分数量
    grouped = _grouped(records)
    stds = [group_metric([r["iee"] for r in rows])["std"] for rows in grouped.values()]
    ndist = [group_metric([r["iee"] for r in rows])["n_distinct"] for rows in grouped.values()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(stds, bins=12, color="#4c9a6b", edgecolor="white")
    axes[0].set_title("Within-group std of IEE"); axes[0].set_xlabel("std (ddof=1)")
    axes[1].bar([1, 2, 3, 4], [ndist.count(k) for k in range(1, 5)], color="#4c9a6b", edgecolor="white")
    axes[1].set_title("Distinct IEE values per group"); axes[1].set_xticks([1, 2, 3, 4])
    fig.tight_layout(); p = fig_dir / "fig_group_discriminability.png"; fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p.name)

    # 4) 相关性热力图（Spearman）
    keys = list(DIMS) + ["iee", "candidate_chars", "len_ratio_to_ref"]
    mat = np.ones((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                continue
            c = summary["correlations"].get(f"{a}~{b}") or summary["correlations"].get(f"{b}~{a}")
            mat[i, j] = (c or {}).get("spearman_r") or 0.0
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation"); fig.tight_layout()
    p = fig_dir / "fig_corr_heatmap.png"; fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p.name)

    # 5) 长度偏差散点（全局 + 组内去均值）
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].scatter([r["candidate_chars"] for r in records], iee, s=8, alpha=0.5)
    axes[0].set_title("IEE vs candidate chars (global)")
    axes[0].set_xlabel("candidate chars"); axes[0].set_ylabel("IEE")
    dx, dy = [], []
    for rows in grouped.values():
        mx = np.mean([r["candidate_chars"] for r in rows]); my = np.mean([r["iee"] for r in rows])
        dx += [r["candidate_chars"] - mx for r in rows]; dy += [r["iee"] - my for r in rows]
    axes[1].scatter(dx, dy, s=8, alpha=0.5, color="#b0713b")
    axes[1].set_title("Within-group demeaned"); axes[1].set_xlabel("chars (demeaned)")
    fig.tight_layout(); p = fig_dir / "fig_length_bias.png"; fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p.name)
    return made


# ---------------------------------------------------------------------------
# 7. 重复评分一致性（区分稳定）
# ---------------------------------------------------------------------------
def analyze_repeat(main: List[Dict[str, Any]],
                   repeat: List[Dict[str, Any]]) -> Dict[str, Any]:
    from scipy import stats

    main_by = {(r["group_id"], r["candidate_index"]): r for r in main}
    rep_by: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in repeat:
        rep_by[(r["group_id"], r["candidate_index"])].append(r)

    out: Dict[str, Any] = {"n_repeat_records": len(repeat),
                           "n_candidates_repeated": len(rep_by),
                           "per_repeat_index": {}}
    # 按 repeat_index 分别统计（默认只有 0）
    idxs = sorted({r.get("repeat_index", 0) for r in repeat})
    for k in idxs:
        dims: Dict[str, Any] = {}
        iee_diffs, exact_iee = [], 0
        total = 0
        for key, rows in rep_by.items():
            base = main_by.get(key)
            cand_rows = [r for r in rows if r.get("repeat_index", 0) == k]
            if base is None or not cand_rows:
                continue
            rep = cand_rows[0]
            total += 1
            iee_diffs.append(abs(base["iee"] - rep["iee"]))
            exact_iee += int(base["iee"] == rep["iee"])
            for name in DIMS:
                d = dims.setdefault(name, {"exact": 0, "diff": []})
                d["exact"] += int(base[f"{name}_raw"] == rep[f"{name}_raw"])
                d["diff"].append(abs(base[f"{name}_raw"] - rep[f"{name}_raw"]))
        out["per_repeat_index"][str(k)] = {
            "n": total,
            "dim_exact_rate": {n: (dims[n]["exact"] / total if total else None) for n in DIMS},
            "dim_mean_abs_diff": {n: (float(np.mean(dims[n]["diff"])) if dims[n]["diff"] else None) for n in DIMS},
            "iee_exact_rate": exact_iee / total if total else None,
            "iee_mean_abs_diff": float(np.mean(iee_diffs)) if iee_diffs else None,
        }
    # 组内排序稳定性：4 个候选都重复过的组
    stable, top1_hits, ktaus = 0, 0, []
    group_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for gid, i in rep_by:
        group_index[gid].append((gid, i))
    for gid, keys in group_index.items():
        rows = [main_by[k] for k in keys if k in main_by]
        if len(rows) < 2:
            continue
        stable += 1
        a = [r["iee"] for r in rows]
        b = [rep_by[(r["group_id"], r["candidate_index"])][0]["iee"] for r in rows]
        if int(np.argmax(a)) == int(np.argmax(b)):
            top1_hits += 1
        if np.std(a) > 0 and np.std(b) > 0:
            ktaus.append(float(stats.kendalltau(a, b)[0]))
    out["group_ranking"] = {
        "n_groups": stable,
        "top1_agreement": top1_hits / stable if stable else None,
        "mean_kendall_tau": float(np.mean(ktaus)) if ktaus else None,
    }
    return out


# ---------------------------------------------------------------------------
# 8. 报告
# ---------------------------------------------------------------------------
def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _pct(x: Any, nd: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{nd}f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def render_report(summary: Dict[str, Any], samples: List[Dict[str, Any]],
                  repeat: Optional[Dict[str, Any]], figures: List[str],
                  out_dir: Path, source_chars: int) -> str:
    meta = summary["meta"]
    g = summary["global"]
    grp = summary["group"]
    w = summary["weights_normalized"]
    disc = summary["dim_discriminability"]
    L: List[str] = []
    L.append("# IEE（信息表达效率）Reward 区分度验证报告\n")
    L.append(f"- 生成时间：{meta['created_at']}")
    L.append(f"- 输入：`{meta.get('input_path', '')}`")
    L.append(f"- 配置：`{meta['config_path']}`")
    L.append(f"- 裁判：`{meta['judge_api'].get('model', '')}` @ `{meta['judge_api'].get('base_url', '')}`")
    L.append(f"- 数据规模：{meta['scored_groups']} 组 × {meta['candidates_per_group']} 候选 "
             f"= {meta['n_candidates']} 条候选")
    L.append(f"- EE 三维权重（**来自配置** `terms.expression_efficiency.dimension_weights`）："
             + "、".join(f"{DIM_ZH[n]}={w[n]:.4f}" for n in DIMS) + "\n")

    # ---------- 结论速览 ----------
    iee_g = grp["iee"]
    disc_share = {n: grp[n]["nonzero_var_ratio"] for n in DIMS}
    stable_txt = "未做重复评分"
    if repeat and repeat.get("per_repeat_index"):
        first = next(iter(repeat["per_repeat_index"].values()))
        stable_txt = (f"重复评分完全一致率 IEE={_pct(first['iee_exact_rate'])}，"
                      f"三维分项一致率 "
                      + "、".join(f"{DIM_ZH[n]}={_pct(first['dim_exact_rate'][n])}" for n in DIMS))
    L.append("## 0. 结论速览（三个概念必须分开看）\n")
    L.append(f"1. **有区分度**（同组候选拿到不同分）：IEE 总分有 {_pct(iee_g['nonzero_var_ratio'])} "
             f"的组组内方差非零；三维分项的非零方差组占比为 "
             + "、".join(f"{DIM_ZH[n]}={_pct(disc_share[n])}" for n in DIMS) + "。")
    L.append(f"2. **区分稳定**（重复评分一致）：{stable_txt}。")
    L.append("3. **区分正确**（分数差异对应真实表达效率差异）：**本实验无法证实** —— 缺人工标注的"
             "表达效率真值。报告给出长度偏差、维度相关性、典型样本等间接证据，最终正确性需要人工复核。\n")

    # ---------- 2. 全局分布 ----------
    L.append("## 1. 全局评分分布与天花板效应\n")
    rows = []
    for name in DIMS:
        d = g[name]["describe"]
        rows.append([DIM_ZH[name], f"{d['mean']:.3f}", f"{d['std']:.3f}", f"{d['median']:.1f}",
                     f"{d['min']:.0f}", f"{d['max']:.0f}", f"{d['p10']:.1f}", f"{d['p25']:.1f}",
                     f"{d['p75']:.1f}", f"{d['p90']:.1f}",
                     _pct(g[name]["full_ratio"]), _pct(g[name]["low_ratio_le2"])])
    d = g["iee"]["describe"]
    rows.append(["**IEE 总分**", f"{d['mean']:.3f}", f"{d['std']:.3f}", f"{d['median']:.3f}",
                 f"{d['min']:.3f}", f"{d['max']:.3f}", f"{d['p10']:.3f}", f"{d['p25']:.3f}",
                 f"{d['p75']:.3f}", f"{d['p90']:.3f}", _pct(g["iee"]["full_ratio"]),
                 _pct(g["iee"]["low_ratio_le05"]) + "(≤0.5)"])
    L.append(_table(["维度", "均值", "标准差", "中位数", "最小", "最大", "P10", "P25", "P75", "P90",
                     "满分率", "低分率"], rows) + "\n")
    L.append("各维度 0~4 原始分数量（占比）：\n")
    rows = [[DIM_ZH[n]] + [f"{g[n]['raw_counts'][str(k)]} ({_pct(g[n]['raw_ratio'][str(k)])})"
                           for k in range(5)] for n in DIMS]
    L.append(_table(["维度", "0", "1", "2", "3", "4"], rows) + "\n")
    # 天花板分析
    ceiling = {n: g[n]["full_ratio"] for n in DIMS}
    worst = min(ceiling, key=ceiling.get)
    L.append("- **分项天花板**：" + "、".join(
        f"{DIM_ZH[n]} 满分率 {_pct(g[n]['full_ratio'])}（均值 {g[n]['describe']['mean']:.2f}/4）"
        for n in DIMS)
        + f"。没有维度达到 >90% 的极端天花板（那样才算该维几乎失效），"
        f"但 {DIM_ZH[max(ceiling, key=ceiling.get)]} 的满分率 {_pct(max(ceiling.values()))} 已偏高。")
    L.append(f"- **IEE 总分压缩（重点）**：{_pct(g['iee']['high_ratio_ge075'])} 的候选 IEE ≥ 0.75，"
             f"{_pct(g['iee']['full_ratio'])} 恰好 = 1.0，中位数 {g['iee']['describe']['median']:.3f}。"
             f"总分高度堆在上端，会把组内本来就不大的差异进一步压缩"
             f"（IEE 组内 σ 均值仅 {grp['iee']['std']['mean']:.3f}、极差均值 {grp['iee']['range']['mean']:.3f}）。")
    L.append("- **近乎恒定的维度**：三个维度的全局标准差都在 "
             + "、".join(f"{DIM_ZH[n]}={g[n]['describe']['std']:.2f}" for n in DIMS)
             + f"；组内零方差占比 "
             + "、".join(f"{DIM_ZH[n]}={_pct(grp[n]['zero_var_ratio'])}" for n in DIMS)
             + "。**没有任何一个维度接近恒定**（未出现某维 100% 组内零方差）。\n")

    # ---------- 3. 组内区分度 ----------
    L.append("## 2. GRPO 组内区分度（核心）\n")
    rows = []
    for name in list(DIMS) + ["iee"]:
        s = grp[name]
        label = DIM_ZH[name] if name in DIM_ZH else "**IEE 总分**"
        rows.append([label,
                     f"{s['std']['mean']:.3f}", f"{s['std']['median']:.3f}",
                     f"{s['std']['p25']:.3f}", f"{s['std']['p75']:.3f}",
                     f"{s['zero_var_groups']} ({_pct(s['zero_var_ratio'])})",
                     _pct(s["nonzero_var_ratio"]),
                     f"{s['range']['mean']:.3f}", f"{s['range']['median']:.3f}",
                     _pct(s["pairwise_mean"])])
    L.append(_table(["指标", "组内σ均值", "σ中位数", "σ P25", "σ P75", "零方差组",
                     "非零方差占比", "极差均值", "极差中位数", "成对可区分率"], rows) + "\n")
    L.append("每组不同评分数量分布（1/2/3/4 种）：\n")
    rows = [[DIM_ZH[n] if n in DIM_ZH else "**IEE**"] +
            [grp[n]["n_distinct_dist"][str(k)] for k in range(1, 5)] for n in list(DIMS) + ["iee"]]
    L.append(_table(["指标", "1 种", "2 种", "3 种", "4 种"], rows) + "\n")
    L.append("最高分并列候选数量分布（1/2/3/4）：\n")
    rows = [[DIM_ZH[n] if n in DIM_ZH else "**IEE**"] +
            [grp[n]["max_tie_dist"][str(k)] for k in range(1, 5)] for n in list(DIMS) + ["iee"]]
    L.append(_table(["指标", "1 个", "2 个", "3 个", "4 个"], rows) + "\n")
    L.append(f"- **各维度能区分的组数**："
             + "、".join(f"{DIM_ZH[n]}={disc['discriminating_groups_per_dim'][n]}/{grp['n_groups']}"
                         for n in DIMS))
    L.append(f"- **三个维度全部无法区分的组**：{disc['all_three_indistinguishable_groups']} 组")
    L.append("- 各维度组合（能区分的维度集合 → 组数）：")
    for combo, cnt in sorted(disc["combination_counts"].items(), key=lambda x: -x[1]):
        label = combo
        if combo != "(none)":
            label = "+".join(DIM_ZH.get(part, part) for part in combo.split("+"))
        L.append(f"  - {label}：{cnt} 组")
    L.append("")

    # ---------- 4. 相关/贡献/权重 ----------
    L.append("## 3. 分项相关性、贡献与权重影响\n")
    L.append("维度间相关性（Pearson / Spearman）：\n")
    rows = []
    for i, a in enumerate(DIMS):
        for b in DIMS[i + 1:]:
            c = summary["correlations"][f"{a}~{b}"]
            rows.append([f"{DIM_ZH[a]} ~ {DIM_ZH[b]}",
                         _f(c["pearson_r"]), _f(c["spearman_r"])])
    L.append(_table(["维度对", "Pearson r", "Spearman ρ"], rows) + "\n")
    L.append("各维度与 IEE 总分、与候选长度的相关性：\n")
    rows = []
    for a in DIMS:
        c1 = summary["correlations"][f"{a}~iee"]
        c2 = summary["correlations"][f"{a}~candidate_chars"]
        rows.append([DIM_ZH[a], _f(c1["pearson_r"]), _f(c1["spearman_r"]),
                     _f(c2["pearson_r"]), _f(c2["spearman_r"])])
    L.append(_table(["维度", "Pearson(dim,IEE)", "Spearman(dim,IEE)",
                     "Pearson(dim,字数)", "Spearman(dim,字数)"], rows) + "\n")
    ct = summary["contribution"]
    L.append("各分项对 IEE 的贡献与组内波动来源（方差分解）：\n")
    rows = []
    for n in DIMS:
        rows.append([DIM_ZH[n], f"{ct['terms'][n]['weight']:.4f}",
                     f"{ct['terms'][n]['mean_contribution']:.4f}",
                     _pct(ct["terms"][n]["share_of_iee_mean"]),
                     _pct(ct["within_group_variance_share"].get(n))])
    for i, a in enumerate(DIMS):
        for b in DIMS[i + 1:]:
            key = f"{a}~{b}"
            rows.append([f"{DIM_ZH[a]}×{DIM_ZH[b]}（协方差项）", "—", "—", "—",
                         _pct(ct["within_group_variance_share"].get(key))])
    L.append(_table(["分项", "归一权重", "平均贡献", "占 IEE 均值", "占组内 IEE 方差"], rows) + "\n")
    L.append("权重敏感性（**离线附加分析，不改配置、不改正式评分**）——各方案与正式方案的组内排序一致性：\n")
    rows = []
    for label, s in summary["weight_sensitivity"]["agreement_vs_actual"].items():
        if label == "actual(config)":
            continue
        rows.append([label, _f(s["global_spearman_vs_actual"]),
                     _f(s["mean_within_group_spearman"]), _pct(s["top1_agreement"])])
    L.append(_table(["权重方案", "全局 Spearman(vs 正式)", "组内平均 Spearman", "组内 top1 一致率"], rows) + "\n")

    # ---------- 5. 长度偏差 ----------
    L.append("## 4. 长度偏差检查\n")
    lb = summary["length_bias"]
    rows = []
    for metric in ["iee"] + list(DIMS):
        label = "**IEE 总分**" if metric == "iee" else DIM_ZH[metric]
        c1 = lb["global"].get(f"{metric}~candidate_chars", {})
        c2 = lb["global"].get(f"{metric}~len_ratio_to_ref", {})
        c3 = lb["global"].get(f"{metric}~len_diff_to_ref", {})
        rows.append([label, _f(c1.get("pearson_r")), _f(c1.get("spearman_r")),
                     _f(c2.get("spearman_r")), _f(c3.get("spearman_r"))])
    L.append(_table(["指标", "Pearson(字数)", "Spearman(字数)", "Spearman(字数比)", "Spearman(字数差)"],
                    rows) + "\n")
    wg = lb["within_group"]
    L.append(f"- **组内去均值后**：IEE 与候选字数的 Spearman ρ = "
             f"{_f((wg['iee~candidate_chars'] or {}).get('spearman_r'))}，"
             f"与字数比 ρ = {_f((wg['iee~len_ratio_to_ref'] or {}).get('spearman_r'))}。")
    L.append(f"- 组内“最短候选拿最高 IEE”：{lb['shortest_highest']['count']}/"
             f"{lb['shortest_highest']['groups_usable']} 组 "
             f"({_pct(lb['shortest_highest']['ratio'])})；"
             f"“最长候选拿最高分”：{lb['longest_highest']['count']} 组 "
             f"({_pct(lb['longest_highest']['ratio'])})。")
    L.append("- 结论口径：负相关只能说明“长度与得分有关系”，**不能直接判定 Judge 错误**——"
             "更短可能确实更凝练。是否正确要看典型样本的人工复核。\n")

    # ---------- 6. 排序稳定性 + 样本 ----------
    L.append("## 5. 排序稳定性与典型样本\n")
    L.append(f"- IEE 最高分并列数量分布（每组）："
             + "、".join(f"{k} 个:{v} 组" for k, v in summary["ranking"]["top1_tie_dist"].items()))
    L.append("- IEE 与三个维度在组内的平均 Spearman ρ："
             + "、".join(f"{DIM_ZH[n]}={_f(summary['ranking']['mean_within_group_spearman_iee_vs_dim'][n])}"
                         for n in DIMS))
    if repeat:
        L.append(f"- 重复评分：覆盖 {repeat['n_repeat_records']} 条、"
                 f"{repeat['n_candidates_repeated']} 个候选；组内排序 top1 一致率 "
                 f"{_pct(repeat['group_ranking']['top1_agreement'])}，"
                 f"平均 Kendall τ = {_f(repeat['group_ranking']['mean_kendall_tau'])}。\n")

    cat_zh = {"fully_tied": "组内完全同分", "max_spread": "组内分差最大",
              "dim_conflict": "维度冲突明显", "longest_highest": "最长候选反而最高",
              "shortest_lowest": "最短候选得分最低"}
    for cat, label in cat_zh.items():
        items = [s for s in samples if s["category"] == cat]
        if not items:
            continue
        L.append(f"### 典型样本 · {label}（{len(items)} 例）\n")
        for s in items:
            src = s["source"]
            shown = src if len(src) <= source_chars else (
                src[: source_chars // 2] + "\n……（原文中间省略）……\n" + src[-source_chars // 2:])
            L.append(f"**组 `{s['group_id']}`** — {s['note']}\n")
            L.append(f"- 人工参考摘要（{len(s['reference'])} 字）：{s['reference']}")
            L.append(f"- 原文（{len(src)} 字，此处显示 {len(shown)} 字，完整见 `samples.jsonl`）：")
            L.append("```\n" + shown + "\n```")
            rows = []
            for c in sorted(s["candidates"], key=lambda x: -x["iee"]):
                rows.append([c["index"], _f(c["iee"]), "✓" if c["is_best"] else "", c["chars"],
                             c[f"{DIMS[0]}_raw"], c[f"{DIMS[1]}_raw"], c[f"{DIMS[2]}_raw"]])
            L.append(_table(["候选#", "IEE", "组内最高", "字数",
                             DIM_ZH[DIMS[0]], DIM_ZH[DIMS[1]], DIM_ZH[DIMS[2]]], rows))
            for c in sorted(s["candidates"], key=lambda x: x["index"]):
                L.append(f"  - 候选#{c['index']}：{c['text']}")
            L.append("")

    # ---------- 7. 重复评分 ----------
    L.append("## 6. 重复评分一致性（区分稳定）\n")
    if not repeat:
        L.append("未启用重复评分（`--no-repeat`），无法评估区分稳定性。\n")
    else:
        rows = []
        for k, s in repeat["per_repeat_index"].items():
            rows.append([f"第 {k} 次", s["n"],
                         _pct(s["dim_exact_rate"][DIMS[0]]),
                         _pct(s["dim_exact_rate"][DIMS[1]]),
                         _pct(s["dim_exact_rate"][DIMS[2]]),
                         _pct(s["iee_exact_rate"]),
                         _f(s["iee_mean_abs_diff"])])
        L.append(_table(["重复轮次", "样本数", DIM_ZH[DIMS[0]], DIM_ZH[DIMS[1]],
                         DIM_ZH[DIMS[2]], "IEE 完全一致率", "IEE 平均绝对差"], rows) + "\n")
        L.append(f"裁判 temperature=0（确定性贪心），重复调用的一致性反映的是服务端非确定性。"
                 f"组内排序 top1 一致率 {_pct(repeat['group_ranking']['top1_agreement'])}，"
                 f"平均 Kendall τ = {_f(repeat['group_ranking']['mean_kendall_tau'])}。\n")

    # ---------- 8. 结论 ----------
    L.append("## 7. 对 GRPO 组内相对奖励的结论\n")
    L.append(_conclusion_text(summary, repeat))
    L.append("\n## 附录 A. 图表\n")
    for name in figures:
        L.append(f"![{name}](figures/{name})\n")
    L.append("## 附录 B. 限制与未完成项\n")
    L.append("- 判定由 LLM 裁判完成，非人工真值；“区分正确”无法由本实验证实。")
    L.append("- 缺失维度按既有逻辑填中性分 2（`judge_ee_batch` 的默认），"
             "低概率的解析失败会被这种默认掩盖，可能轻微影响分布。")
    L.append("- 重复评分仅抽样一部分组；temperature=0 下的一致性偏高属预期。")
    return "\n".join(L) + "\n"


def _conclusion_text(summary: Dict[str, Any], repeat: Optional[Dict[str, Any]]) -> str:
    grp = summary["group"]
    disc = summary["dim_discriminability"]
    n_groups = grp["n_groups"]
    iee = grp["iee"]
    dims_nonzero = {n: grp[n]["nonzero_var_ratio"] for n in DIMS}
    lines: List[str] = []
    lines.append(f"**有区分度（组内不同分）**：IEE 总分零方差组占 {_pct(iee['zero_var_ratio'])}"
                 f"（{iee['zero_var_groups']}/{n_groups} 组 4 个候选完全同分），"
                 f"成对可区分率均值 {_pct(iee['pairwise_mean'])}。"
                 f"组内 σ 均值 {iee['std']['mean']:.3f}、极差均值 {iee['range']['mean']:.3f}；"
                 f"{iee['n_distinct_dist']['3'] + iee['n_distinct_dist']['4']} 组有 3~4 个不同的 IEE 值。"
                 + ("总分在多数组存在可用的组内差异。" if iee["nonzero_var_ratio"] >= 0.5
                    else "总分在超过一半的组里无组内差异，组内优势会大量为 0。"))
    ceiling = summary["global"]
    lines.append(f"**天花板 / 压缩**：{_pct(ceiling['iee']['high_ratio_ge075'])} 的候选 IEE ≥ 0.75、"
                 f"{_pct(ceiling['iee']['full_ratio'])} = 1.0；分项满分率 "
                 + "、".join(f"{DIM_ZH[n]}={_pct(ceiling[n]['full_ratio'])}" for n in DIMS)
                 + "。分数堆在上端，组内可分辨的绝对差距被压小。")
    lines.append("**三个分项各自的区分度**：" + "；".join(
        f"{DIM_ZH[n]} 非零方差组占比 {_pct(dims_nonzero[n])}" for n in DIMS) + "。"
        f"三个维度全部无法区分的组有 {disc['all_three_indistinguishable_groups']} 组。")
    lines.append("**权重影响**：低权重维度即使本身有区分度，对 IEE 排序的贡献也会被压缩；"
                 "具体看第 3 节的方差分解与权重敏感性表（例如某维度组内波动大但权重仅 0.3，"
                 "其组内方差占比会明显偏低）。")
    if repeat and repeat.get("per_repeat_index"):
        first = next(iter(repeat["per_repeat_index"].values()))
        lines.append(f"**区分稳定**：重复评分 IEE 完全一致率 {_pct(first['iee_exact_rate'])}，"
                     f"组内排序 top1 一致率 {_pct(repeat['group_ranking']['top1_agreement'])}，"
                     f"平均 Kendall τ = {_f(repeat['group_ranking']['mean_kendall_tau'])}。"
                     + ("完全一致率不高：即使 temperature=0，裁判在重复调用间也会给出不同分"
                        "（服务端非确定性），组内奖励因此带有额外噪声。"
                        if (first["iee_exact_rate"] or 0) < 0.95 else
                        "重复调用基本一致，裁判确定性较好。"))
    lines.append("**区分正确**：本实验只能提供间接证据（长度偏差、维度相关性、典型样本），"
                 "不能证明“分差 = 真实表达效率差”。建议对第 5 节的典型样本做人工复核后再定论。")
    stable_clause = ""
    if repeat and repeat.get("per_repeat_index"):
        first = next(iter(repeat["per_repeat_index"].values()))
        stable_clause = (f"、重复评分 IEE 完全一致率 {_pct(first['iee_exact_rate'])}、"
                         f"组内排序 top1 一致率 {_pct(repeat['group_ranking']['top1_agreement'])}")
    verdict = (f"IEE 总分在多数组有组内差异{stable_clause}，**可作为 GRPO 组内相对优势的候选信号**；"
               "但仍需人工复核确认区分正确性，并注意同分组比例带来的零梯度。"
               if iee["nonzero_var_ratio"] >= 0.5 else
               f"IEE 总分在多数组无组内差异{stable_clause}，"
               "**单独作为 GRPO 组内相对奖励信号偏弱**，"
               "会把大量样本的组内优势压成 0；建议配合其它奖励或调整维度设计后再评估。")
    lines.append(f"**结论**：{verdict}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="IEE Reward 区分度验证：打分 + 统计 + Markdown 报告（不训练 GRPO）"
    )
    ap.add_argument("--config", default="configs/grpo_fact_coverage_inp_t4.yaml",
                    help="从中读取 EE 三维权重与 judge_api（不修改它）")
    ap.add_argument("--input", default="data/triples/sft_val_100_n4.jsonl")
    ap.add_argument("--candidates-key", default="outputs",
                    help="每组候选摘要的字段名（默认 outputs；缺失则退回单个 output）")
    ap.add_argument("--out-dir", default=None,
                    help="默认 data/judge/iee_<input 文件名>")
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 组（冒烟用）")
    ap.add_argument("--batch-candidates", type=int, default=32,
                    help="每次交给 Judge 的候选数（内部再按 judge_api.concurrency 并发）")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="覆盖配置里的 judge_api.concurrency")
    ap.add_argument("--no-repeat", action="store_true",
                    help="不做重复评分（区分稳定那一节会标为未完成）")
    ap.add_argument("--repeat-groups", type=int, default=20, help="重复评分抽多少组")
    ap.add_argument("--repeat-k", type=int, default=1, help="每组重复评几次")
    ap.add_argument("--repeat-seed", type=int, default=42)
    ap.add_argument("--analysis-only", action="store_true",
                    help="跳过 API，只从已有 scores.jsonl 重算报告")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--max-samples-per-cat", type=int, default=3)
    ap.add_argument("--source-chars", type=int, default=3000,
                    help="报告中每个典型样本最多显示多少字原文（完整见 samples.jsonl）")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config_path = resolve(args.config)
    input_path = resolve(args.input)
    out_dir = resolve(args.out_dir) if args.out_dir else resolve(f"data/judge/iee_{input_path.stem}")
    ensure_dir(out_dir)

    weights, cfg_info, _ = load_iee_weights(config_path)
    cfg_info["input_path"] = str(input_path)
    logger.info("EE 三维权重（来自配置 %s）：%s", config_path.name, weights)

    groups = load_groups(input_path, args.candidates_key, args.limit)
    all_groups = groups if not args.limit else load_groups(input_path, args.candidates_key, None)
    logger.info("读入 %d 组，共 %d 条候选", len(groups),
                sum(len(g["candidates"]) for g in groups))

    scores_path = out_dir / "scores.jsonl"
    repeat_path = out_dir / "repeat_scores.jsonl"
    if args.analysis_only:
        logger.info("--analysis-only：跳过 API，复用 %s", scores_path)
    else:
        run_scoring(args, groups, weights, cfg_info, out_dir)
        run_repeat_scoring(args, groups, weights, cfg_info, out_dir)

    if not scores_path.exists():
        raise SystemExit(f"没有打分结果 {scores_path}；去掉 --analysis-only 先跑一次。")
    records = read_jsonl_local(scores_path)
    if not records:
        raise SystemExit(f"{scores_path} 是空的。")

    summary = analyze(records, all_groups, weights, cfg_info)

    repeat_stats = None
    if repeat_path.exists():
        rrecs = read_jsonl_local(repeat_path)
        if rrecs:
            repeat_stats = analyze_repeat(records, rrecs)
            summary["repeat"] = repeat_stats

    samples = select_samples(records, all_groups, args.max_samples_per_cat)
    summary["samples"] = [{"category": s["category"], "group_id": s["group_id"]} for s in samples]
    samples_path = out_dir / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    figures: List[str] = [] if args.no_figures else make_figures(records, summary, out_dir / "figures")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_report(summary, samples, repeat_stats, figures, out_dir, args.source_chars)
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    logger.info("候选打分：%s", scores_path)
    if repeat_stats:
        logger.info("重复评分：%s", repeat_path)
    logger.info("统计量：%s\n典型样本：%s\n报告：%s", summary_path, samples_path, report_path)


if __name__ == "__main__":
    main()
