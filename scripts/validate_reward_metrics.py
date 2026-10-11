"""Judge 三指标区分度验证：fact_consistency / element_coverage / expression_efficiency。

目标
====
在 ``data/triples/sft_val_100_n4.jsonl``（100 篇文书 × 4 个候选摘要）上，验证
**新的（便宜的）Judge 模型**能否在三个奖励指标上给 GRPO 提供有意义的
**组内相对优势**信号：

  * ``fact_consistency``     事实一致性：六要素加权分 ∈ [0,1]
  * ``element_coverage``     关键要素覆盖率：六要素覆盖加权分 ∈ [0,1]
  * ``expression_efficiency``表达效率：语义冗余/表达冗余/抽象适当性加权分 ∈ [0,1]

它和 ``scripts/validate_iee_reward.py`` 是同一套实验方法，只是把对象从
"EE 的三个子维度"扩展成"三个奖励指标（各自还带子项）"。

关键约束（逐条遵守）
==================
  * **不训练 GRPO**，不修改任何 Judge Prompt / 评分规则 / 配置权重；
  * 每个指标的内部权重**从配置文件里读**（``RewardSpec`` → 对应 term 的
    ``element_weights`` / ``dimension_weights``），绝不硬编码默认值；
  * 复用**现有** Judge 调用 / Prompt / 解析 / 重试逻辑：直接调
    ``FactConsistencyJudge.judge_fact_batch`` / ``judge_coverage_six_batch``
    / ``judge_ee_batch``（它们内部用 ``prompts.py`` 的 build_*_messages 与
    parse_*），以及 ``sfzy.judge.api_runtime.APIRuntime``（并发 + 指数退避重试）；
  * 每个候选**独立评分**（一次判定只喂一个候选，不让裁判在 prompt 里同时看 N 个）；
  * 支持断点续跑（已打分的 candidate_id 直接跳过）。

真实 Reward 与门控
=================
报告里的 raw / effective 奖励**不重新实现规则**，直接调用训练同一套代码：

  * ``RewardSpec.from_config(cfg.rl.reward)``  → 启用的 term、归一权重、门控配置
  * ``sfzy.rl.reward.compute_reward(...)``     → 文本门控（长度/前缀/结果标记）
                                                 + 事实硬门控 + 加权求和
  * ``check_gate`` / ``check_fact_error_gate`` → 门控原因（按原因分类统计）

三种口径：
  * ``raw_reward``          —— ``spec.gate.enabled=False``，即门控前的纯加权和；
  * ``effective_reward``    —— 配置里的真实门控口径（与 GRPO 训练完全一致）；
  * ``effective_reward_fc`` —— 同一套规则、把事实错误门控强制打开，
                               即"裁判判定存在明确事实错误 → 整条 reward=0"。

事实硬门控读的是裁判的 ``has_factual_error``（1=存在明确事实错误），不是
FC 分数阈值。若配置里 ``gate.fact_has_error`` 为 0（门控关闭），报告会显式提示，
并同时给出 ``effective_reward_fc`` 作为对照。

事实一致性（FC）指标现在只有**整体分 + 错误门控**，没有子项，所以第 2 节里
它没有逐要素明细；KEC / EE 仍然是多子项指标。

重复评分与排序稳定性
===================
``repeat_scores.jsonl`` 是**真正第二次请求 Judge**（不复用第一次缓存），用于：
完全一致率 / MAE / Top-1 集合一致率（并列全部纳入）/ Kendall τ-b（全并列记为
不可计算）/ 成对排序翻转率 / 保持率 / 并列关系变化率；对三个指标 + raw/effective
奖励分别统计。门控前后的排序变化是**同一轮评分内**的，单独放在报告第 4.4 节。

产物（默认 ``data/judge/metrics_<input_stem>/``）
================================================
  * ``scores.jsonl``        每条候选：三指标标量 + 全部子项原始分 + 长度元数据
                            + raw/effective 奖励 + 门控原因 + 逐要素 0 分记录
  * ``repeat_scores.jsonl`` 重复评分（单独保存，绝不覆盖正式结果）
  * ``summary.json``        全部统计量（机器可读）：metrics / cross_metric /
                            rewards / gate / gate_group / gate_impact /
                            consistency / repeat
  * ``samples.jsonl``       典型样本（人工复核用，含原文/人工摘要/候选全文）
  * ``report.md``           Markdown 统计报告
  * ``figures/*.png``       图表

用法
====
    # 打分 + 分析 + 出报告
    python scripts/validate_reward_metrics.py \
        --config configs/grpo_fact_cov_ee_t4.yaml \
        --input data/triples/sft_val_100_n4.jsonl

    # 只重算报告（不调 API），复用已有 scores.jsonl
    python scripts/validate_reward_metrics.py --analysis-only

    # 只验一个指标、先小样本冒烟
    python scripts/validate_reward_metrics.py --tasks expression_efficiency --limit 3 --no-repeat
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# 先导入同目录的 EE 验证脚本：它在 import 阶段会先注入 torch/transformers stub，
# 再 import sfzy.judge。这里复用它的通用工具（描述统计/相关/分组/表格/渲染）。
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import validate_iee_reward as base                            # noqa: E402

from sfzy.config import load_config                           # noqa: E402
from sfzy.eval.rouge import score_pair                        # noqa: E402
from sfzy.judge.api_runtime import build_api_runtime          # noqa: E402
from sfzy.judge.judge import FactConsistencyJudge             # noqa: E402
from sfzy.judge.schema import (                               # noqa: E402
    EE_DIMENSIONS,
    ELEMENT_ZH,
    ELEMENTS,
    MAX_SCORE,
    aggregate_coverage,
    expression_efficiency_score,
)
from sfzy.rl.reward import (                                  # noqa: E402
    check_fact_error_gate,
    check_gate,
    compute_reward,
)
from sfzy.rl.reward_spec import RewardSpec                    # noqa: E402
from sfzy.utils.io import ensure_dir                          # noqa: E402
from sfzy.utils.logging import get_logger                     # noqa: E402

describe = base.describe
safe_corr = base.safe_corr
group_metric = base.group_metric
_grouped = base._grouped
_f = base._f
_pct = base._pct
_table = base._table
read_jsonl_local = base.read_jsonl_local
load_groups = base.load_groups
candidate_id = base.candidate_id
load_done_ids = base.load_done_ids
MAX_RAW = base.MAX_RAW

logger = get_logger("validate_reward_metrics")

EE_DIM_ZH: Dict[str, str] = {
    "semantic_redundancy": "语义冗余",
    "verbal_redundancy": "表达冗余",
    "abstraction_adequacy": "抽象适当性",
}


@dataclass(frozen=True)
class MetricSpec:
    """一个奖励指标的静态描述：怎么打分、有哪些子项、权重写在配置的哪个字段。"""

    name: str
    label: str
    subs: Tuple[str, ...]
    sub_zh: Dict[str, str]
    weight_field: Optional[str]   # 配置项内部权重的字段名；None = 该指标没有子项权重
    aggregator: str          # "fact" | "coverage" | "ee"

    def sub_label(self, sub: str) -> str:
        return self.sub_zh.get(sub, sub)


METRICS: Tuple[MetricSpec, ...] = (
    # 事实一致性已改成"整体分 + 事实错误门控"（prompts.JUDGE_SYSTEM），不再有子项权重。
    MetricSpec("fact_consistency", "事实一致性", (), {}, None, "fact"),
    MetricSpec("element_coverage", "关键要素覆盖率", tuple(ELEMENTS), dict(ELEMENT_ZH),
               "element_weights", "coverage"),
    MetricSpec("expression_efficiency", "表达效率", tuple(EE_DIMENSIONS), EE_DIM_ZH,
               "dimension_weights", "ee"),
)
METRIC_BY_NAME: Dict[str, MetricSpec] = {m.name: m for m in METRICS}


def resolve(path: Any) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# ---------------------------------------------------------------------------
# 0. 配置：三指标的内部权重只从配置文件来
# ---------------------------------------------------------------------------
def load_config_bundle(
    config_path: Path, tasks: Sequence[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, Any]]:
    """从配置里取三指标内部权重 + judge_api + 混合奖励方案。

    走 ``RewardSpec.from_config``（与训练完全相同的解析路径）拿到结构，但
    **权重数值直接读 YAML**——因为 ``judge_term_options()`` 只包含 enabled 的
    term，而本实验要求即使 term 被关掉也能验证它的区分度（内部权重仍以配置为准）。
    """
    cfg = load_config(config_path)
    reward_raw = cfg.path_("rl.reward")
    if not reward_raw:
        raise SystemExit(f"{config_path} 里没有 rl.reward，取不到分项权重")
    spec = RewardSpec.from_config(reward_raw)
    terms_raw: Dict[str, Any] = dict(reward_raw.get("terms") or {})

    weights: Dict[str, Dict[str, float]] = {}
    for name in tasks:
        ms = METRIC_BY_NAME[name]
        if ms.weight_field is None:      # 该指标没有子项权重（新的事实一致性）
            weights[name] = {}
            continue
        entry = terms_raw.get(name) or {}
        raw_w = entry.get(ms.weight_field)
        if not isinstance(raw_w, dict):
            raise SystemExit(
                f"配置里没有 terms.{name}.{ms.weight_field}。\n"
                f"  本实验要求该指标的内部权重来自配置文件（键：{list(ms.subs)}），"
                f"不接受代码里的默认值。"
            )
        missing = [k for k in ms.subs if k not in raw_w]
        unknown = [k for k in raw_w if k not in ms.subs]
        if missing or unknown:
            raise SystemExit(
                f"terms.{name}.{ms.weight_field} 的键不对：缺少 {missing}，多余 {unknown}；"
                f"必须且只能是 {list(ms.subs)}"
            )
        w = {k: float(raw_w[k]) for k in ms.subs}
        if any(v < 0 for v in w.values()):
            raise SystemExit(f"terms.{name}.{ms.weight_field} 有负权重：{w}")
        if sum(w.values()) <= 0:
            raise SystemExit(f"terms.{name}.{ms.weight_field} 权重之和必须大于 0：{w}")
        weights[name] = w

    # ---- 真实训练 Reward 口径的上下文（不重新发明规则，直接用训练代码）----
    spec_gated = spec
    spec_ungated = copy.deepcopy(spec)
    spec_ungated.gate.enabled = False          # 门控前 = 纯加权和
    # "强制开启事实错误门控"口径：与训练同一函数，只把开关打开
    spec_fc = copy.deepcopy(spec)
    spec_fc.gate.cfg["fact_has_error"] = 1
    needed = set(spec.needed_signals)
    computable = set(tasks) | {"fact_has_error"}
    missing_signals = sorted(needed - computable)
    if missing_signals:
        logger.warning(
            "配置启用/门控还需要这些裁判信号，但本次 --tasks 不产出它们：%s\n"
            "  → 混合奖励（raw/effective）与门控统计将标记为不可用。",
            missing_signals,
        )
    reward_ctx: Dict[str, Any] = {
        "spec": spec_gated,
        "spec_ungated": spec_ungated,
        "spec_fc": spec_fc,
        "missing_signals": missing_signals,
    }
    cfg_info: Dict[str, Any] = {
        "config_path": str(config_path),
        "tasks": list(tasks),
        "metric_weights": weights,
        "judge_api": {k: v for k, v in dict(cfg.path_("semantic.judge_api") or {}).items()
                      if k != "api_key"},
        "six_max_new_tokens": int(cfg.path_("semantic.six_max_new_tokens") or 512),
        "max_input_tokens": int(cfg.path_("semantic.max_input_tokens") or 8192),
        "enabled_terms": [t.name for t in spec.enabled_terms],
        "normalized_term_weights": spec.normalized_weights(),
        # 真实 Reward / 门控口径（可 JSON 序列化的一份，进 summary.json）
        "reward_plan": {
            "enabled_terms": [t.name for t in spec.enabled_terms],
            "active_terms": [t.name for t in spec.active_terms],
            "consumed_terms": sorted(spec.consumed_terms),
            "normalized_weights": spec.normalized_weights(),
            "needed_signals": sorted(needed),
            "missing_signals": missing_signals,
            "gate_enabled": bool(spec.gate.enabled),
            "gate_cfg": dict(spec.gate.cfg),
            "fact_consistency_enabled": bool(spec.fact_consistency_enabled),
            "fact_has_error_gate": int(spec.gate.cfg.get("fact_has_error", 0) or 0),
            "fact_error_gate_enabled": bool(spec.fact_error_gate_enabled),
            "fc_forced_gate": int(spec_fc.gate.cfg["fact_has_error"]),
            "rouge_mode": spec.rouge_mode,
        },
    }
    return weights, cfg_info, reward_ctx


# 需要做区分度 / 排序稳定性分析的"奖励字段"：
#   三个指标本身 + 门控前混合奖励 + 全部门控后的最终有效奖励 + "任一要素 0 分"口径
REWARD_FIELDS: Tuple[str, ...] = ("raw_reward", "effective_reward", "effective_reward_fc")


def _rec_field(rec: Dict[str, Any], field: str) -> Optional[float]:
    """从一条记录里取某个字段的标量值（指标 or 奖励字段）。"""
    if field in METRIC_BY_NAME:
        return rec["metrics"][field]["reward"]
    return rec.get(field)


def judge_signals_of(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """把一条候选的三个指标结果整理成训练 Reward 需要的 judge_signals。"""
    signals: Dict[str, Optional[float]] = {
        name: float(m["reward"]) for name, m in metrics.items() if m is not None
    }
    fc = metrics.get("fact_consistency")
    if fc is not None:
        # 事实硬门控要读的信号：1 = 裁判判定存在明确事实错误（与训练同名）
        signals["fact_has_error"] = float(int(fc.get("has_factual_error") or 0))
    return signals


def reward_fields(
    candidate: str, reference: str, signals: Dict[str, Optional[float]],
    reward_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """用**训练代码**算 raw / effective / 强制开启事实门控三种奖励 + 门控诊断。

    * ``raw_reward``          —— 门控整块关掉后的加权和（等价于 spec.gate.enabled=False）
    * ``effective_reward``    —— 配置里真实的门控口径（文本门控 + 事实硬门控）
    * ``effective_reward_fc`` —— 同一套规则，但把事实错误门控强制打开
                                 （has_factual_error=1 → reward 0）

    门控本身**不重新实现**：直接调 ``sfzy.rl.reward`` 里的
    ``compute_reward`` / ``check_gate`` / ``check_fact_error_gate``。
    """
    spec = reward_ctx["spec"]
    has_error = signals.get("fact_has_error")
    out: Dict[str, Any] = {
        "raw_reward": None, "effective_reward": None, "effective_reward_fc": None,
        "gated": False, "gate_reason": "", "reward_terms": {}, "reward_error": "",
        "gate": {
            "text_reason": None,
            "fc_gate_enabled": bool(spec.fact_error_gate_enabled),
            "fc_forced": int(reward_ctx["spec_fc"].gate.cfg["fact_has_error"]),
            "fact_has_error": (None if has_error is None else int(has_error)),
            "fc_reason": None,
            "fc_strict_reason": None,
        },
    }
    gate = out["gate"]
    if spec.gate.enabled:
        gate["text_reason"] = check_gate(candidate, reference, spec.gate.cfg)
    gate["fc_reason"] = check_fact_error_gate(spec, signals)
    gate["fc_strict_reason"] = check_fact_error_gate(reward_ctx["spec_fc"], signals)
    if reward_ctx.get("missing_signals"):
        out["reward_error"] = "缺少信号 " + ",".join(reward_ctx["missing_signals"])
        return out
    try:
        raw, bd_u = compute_reward(candidate, reference, source=None,
                                   spec=reward_ctx["spec_ungated"], judge_signals=signals)
        eff, bd = compute_reward(candidate, reference, source=None,
                                 spec=spec, judge_signals=signals)
        eff_fc, _bd_fc = compute_reward(candidate, reference, source=None,
                                        spec=reward_ctx["spec_fc"], judge_signals=signals)
    except Exception as exc:  # noqa: BLE001 —— 缺信号等不该毁掉整条记录
        out["reward_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out.update({
        "raw_reward": float(raw),
        "effective_reward": float(eff),
        "effective_reward_fc": float(eff_fc),
        "gated": bool(bd.gated),
        "gate_reason": bd.gate_reason,
        "reward_terms": {k: float(v) for k, v in bd_u.values.items()},
    })
    return out


# ---------------------------------------------------------------------------
# 排序一致性工具（Top-1 集合 / Kendall τ-b / 成对关系）
# ---------------------------------------------------------------------------
def _cmp(a: float, b: float, eps: float = 1e-9) -> int:
    if abs(a - b) <= eps:
        return 0
    return 1 if a > b else -1


def top1_sets(a: Sequence[float], b: Sequence[float], eps: float = 1e-9) -> Tuple[set, set]:
    """各自的 Top-1 集合（并列全部纳入，不取第一个）。"""
    sa = {i for i, v in enumerate(a) if abs(v - max(a)) <= eps} if len(a) else set()
    sb = {i for i, v in enumerate(b) if abs(v - max(b)) <= eps} if len(b) else set()
    return sa, sb


def kendall_tau_b(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Kendall τ-b。全并列（任一侧方差为 0）时**不可定义**，返回 None（不记 0/1）。"""
    from scipy import stats

    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size < 2 or y.size != x.size:
        return None
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    tau = stats.kendalltau(x, y, variant="b")[0]
    if tau is None or (isinstance(tau, float) and math.isnan(tau)):
        return None
    return float(tau)


def pairwise_relation_stats(a: Sequence[float], b: Sequence[float],
                            eps: float = 1e-9) -> Dict[str, Any]:
    """两次评分（或门控前后）在同一组内的成对关系变化。

    对每一对候选 i<j：
      * 两次符号相同（含两次都是并列）→ keep
      * 两次符号相反（A>B → A<B 或反之）→ flip
      * 一次并列、一次不并列 → tie_change（tie_gain / tie_loss）
    """
    n = min(len(a), len(b))
    keep = flip = tie_gain = tie_loss = both_tied = 0
    for i in range(n):
        for j in range(i + 1, n):
            ca, cb = _cmp(a[i], a[j], eps), _cmp(b[i], b[j], eps)
            if ca == cb:
                keep += 1
                if ca == 0:
                    both_tied += 1
            elif ca != 0 and cb != 0:
                flip += 1
            elif ca == 0:
                tie_gain += 1      # 并列 → 非并列
            else:
                tie_loss += 1      # 非并列 → 并列
    total = n * (n - 1) // 2
    return {
        "pairs": total,
        "both_tied_pairs": both_tied,
        "comparable_pairs": total - both_tied,
        "keep": keep, "flip": flip,
        "tie_gain": tie_gain, "tie_loss": tie_loss,
        "tie_change": tie_gain + tie_loss,
        "keep_rate": keep / total if total else None,
        "flip_rate": flip / total if total else None,
        "tie_change_rate": (tie_gain + tie_loss) / total if total else None,
    }


# ---------------------------------------------------------------------------
# 1. 打分（可断点续跑）
# ---------------------------------------------------------------------------
def build_judge(weights: Dict[str, Dict[str, float]], api_cfg: Dict[str, Any],
                six_max_new_tokens: int, concurrency: Optional[int],
                max_input_tokens: int = 8192) -> FactConsistencyJudge:
    cfg = dict(api_cfg)
    if concurrency:
        cfg["concurrency"] = int(concurrency)
    runtime = build_api_runtime(cfg)
    return FactConsistencyJudge(
        runtime=runtime,
        tasks=tuple(weights.keys()),
        weights=weights.get("fact_consistency"),
        coverage_weights=weights.get("element_coverage"),
        ee_weights=weights.get("expression_efficiency"),
        six_max_new_tokens=six_max_new_tokens,
        max_input_tokens=max_input_tokens,
    )


def _metric_raws(judge: FactConsistencyJudge, ms: MetricSpec,
                 items: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """调 Judge 的**现有**批量接口。"""
    if ms.aggregator == "fact":
        return list(judge.judge_fact_batch([(it["source"], it["candidate"]) for it in items]))
    if ms.aggregator == "coverage":
        return list(judge.judge_coverage_six_batch([(it["reference"], it["candidate"]) for it in items]))
    return list(judge.judge_ee_batch(
        [(it["source"], it["reference"], it["candidate"]) for it in items]))


def score_metric(
    judge: FactConsistencyJudge, ms: MetricSpec,
    items: Sequence[Dict[str, str]], weights: Dict[str, float],
) -> List[Optional[Dict[str, Any]]]:
    """一批候选在**一个指标**上的分。批量失败退回逐条，单条失败记 None。"""
    try:
        raws: List[Optional[Dict[str, Any]]] = list(_metric_raws(judge, ms, items))
    except Exception as exc:  # noqa: BLE001 —— 一批失败不该丢掉整轮
        logger.warning("[%s] 批量判定失败（%d 条），退回逐条：%s", ms.name, len(items), exc)
        raws = []
        for it in items:
            try:
                raws.append(_metric_raws(judge, ms, [it])[0])
            except Exception as exc2:  # noqa: BLE001
                logger.error("[%s] 单条判定失败，跳过（下次续跑会重试）：%s", ms.name, exc2)
                raws.append(None)

    out: List[Optional[Dict[str, Any]]] = []
    for raw in raws:
        if raw is None:
            out.append(None)
            continue
        if ms.aggregator == "fact":
            score = int(raw.get("fact_consistency_score", 2))
            out.append({
                "reward": score / float(MAX_SCORE),
                "sub_raw": {},
                "fact_score": score,
                "has_factual_error": int(raw.get("has_factual_error", 0)),
                "error_reason": str(raw.get("error_reason") or ""),
            })
            continue
        subs = {k: int(raw.get(k, 2)) for k in ms.subs}
        if ms.aggregator == "coverage":
            reward = aggregate_coverage(subs, list(ms.subs), weights)
            out.append({"reward": (None if reward is None else float(reward)), "sub_raw": subs})
        else:
            out.append({"reward": expression_efficiency_score(subs, weights), "sub_raw": subs})
    return out


def score_items_all(
    judge: FactConsistencyJudge, weights: Dict[str, Dict[str, float]],
    items: Sequence[Dict[str, str]],
) -> List[Optional[Dict[str, Any]]]:
    """一批候选 → 每条的 {指标: {...}}。任一指标缺失则该候选记 None（下次续跑重试）。"""
    per_metric = {
        name: score_metric(judge, METRIC_BY_NAME[name], items, w)
        for name, w in weights.items()
    }
    out: List[Optional[Dict[str, Any]]] = []
    for i in range(len(items)):
        metrics: Dict[str, Any] = {}
        ok = True
        for name in weights:
            res = per_metric[name][i]
            if res is None or res.get("reward") is None:
                ok = False
            metrics[name] = res
        out.append(metrics if ok else None)
    return out


def make_record(group: Dict[str, Any], index: int, metrics: Dict[str, Any],
                reward_ctx: Dict[str, Any], judge_model: str) -> Dict[str, Any]:
    cand = group["candidates"][index]
    ref = group["reference"]
    rec: Dict[str, Any] = {
        "group_id": group["group_id"],
        "candidate_index": index,
        "candidate_id": candidate_id(group["group_id"], index),
        "judge_model": judge_model,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_chars": len(cand),
        "reference_chars": len(ref),
        "source_chars": len(group["source"]),
        "len_ratio_to_ref": round(len(cand) / max(1, len(ref)), 6),
        "len_diff_to_ref": len(cand) - len(ref),
        "metrics": metrics,
    }
    # ROUGE 只用来说明"是否与官方口径一致"，不是本实验的验证对象
    try:
        rec["rouge_l"] = float(
            score_pair(cand.strip(), ref.strip(),
                       mode=reward_ctx["spec"].rouge_mode)["rouge-l-f"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("ROUGE 计算失败：%s", exc)
        rec["rouge_l"] = None
    # ---- 真实训练 Reward：门控前 / 门控后 / 强制开启事实门控口径 ----
    signals = judge_signals_of(metrics)
    rec.update(reward_fields(cand, ref, signals, reward_ctx))
    return rec


def backfill_reward_fields(records: List[Dict[str, Any]], groups: List[Dict[str, Any]],
                           reward_ctx: Dict[str, Any]) -> int:
    """给**旧版** scores.jsonl 补算 raw/effective 奖励与门控信息。

    旧结果里只有三个指标的判分，没有奖励字段；这里直接用已存的 metric 分
    （外加配置里的权重/门控）重算，**不重新调 Judge**。
    """
    text_by_key: Dict[Tuple[str, int], Tuple[str, str, str]] = {}
    for g in groups:
        for i, cand in enumerate(g["candidates"]):
            text_by_key[(g["group_id"], i)] = (g["source"], g["reference"], cand)
    n = 0
    for r in records:
        if "gate" in r and "raw_reward" in r:
            continue
        texts = text_by_key.get((r.get("group_id"), r.get("candidate_index")))
        metrics = {k: v for k, v in (r.get("metrics") or {}).items() if v}
        if texts is None or not metrics:
            continue
        src, ref, cand = texts
        r.update(reward_fields(cand, ref, judge_signals_of(metrics), reward_ctx))
        n += 1
    if n:
        logger.info("为 %d 条旧结果补算了 raw/effective 奖励与门控信息（未重新调 Judge）", n)
    return n


def _iter_work(groups: List[Dict[str, Any]], done: set) -> List[Tuple[str, int, str, str, str]]:
    tasks: List[Tuple[str, int, str, str, str]] = []
    for group in groups:
        for index, cand in enumerate(group["candidates"]):
            if candidate_id(group["group_id"], index) not in done:
                tasks.append((group["group_id"], index, group["source"],
                              group["reference"], cand))
    return tasks


def run_scoring(args: argparse.Namespace, groups: List[Dict[str, Any]],
                weights: Dict[str, Dict[str, float]], cfg_info: Dict[str, Any],
                reward_ctx: Dict[str, Any], out_dir: Path) -> Path:
    scores_path = out_dir / "scores.jsonl"
    done = load_done_ids(scores_path)
    n_cand = sum(len(g["candidates"]) for g in groups)
    tasks = _iter_work(groups, done)
    logger.info("候选共 %d 条，已完成 %d 条，待打分 %d 条（指标：%s）",
                n_cand, len(done), len(tasks), list(weights))
    if not tasks:
        logger.info("全部候选都已打分：%s", scores_path)
        return scores_path

    judge = build_judge(weights, cfg_info["judge_api"], cfg_info["six_max_new_tokens"],
                        args.concurrency, cfg_info["max_input_tokens"])
    model = cfg_info["judge_api"].get("model", "")
    group_by_id = {g["group_id"]: g for g in groups}

    started = time.time()
    written = skipped = 0
    with open(scores_path, "a", encoding="utf-8") as f:
        for start in range(0, len(tasks), args.batch_candidates):
            chunk = tasks[start:start + args.batch_candidates]
            items = [{"source": s, "reference": r, "candidate": c} for _g, _i, s, r, c in chunk]
            results = score_items_all(judge, weights, items)
            for (gid, idx, _s, _r, _c), metrics in zip(chunk, results):
                if metrics is None:
                    skipped += 1
                    continue
                rec = make_record(group_by_id[gid], idx, metrics, reward_ctx, model)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            elapsed = time.time() - started
            rate = written / elapsed if elapsed > 0 else 0.0
            eta = (len(tasks) - written - skipped) / rate / 60 if rate > 0 else float("inf")
            logger.info("进度 %d/%d（跳过 %d） | %.1f 条/秒 | 预计剩余 %.1f 分钟",
                        written + skipped, len(tasks), skipped, rate, eta)
    logger.info("本轮写入 %d 条、跳过 %d 条 → %s", written, skipped, scores_path)
    return scores_path


def run_repeat_scoring(args: argparse.Namespace, groups: List[Dict[str, Any]],
                       weights: Dict[str, Dict[str, float]], cfg_info: Dict[str, Any],
                       reward_ctx: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    """对抽取的一部分组做重复评分，单独落盘（绝不动正式结果）。"""
    if args.no_repeat or args.repeat_groups <= 0:
        return None
    repeat_path = out_dir / "repeat_scores.jsonl"
    done = load_done_ids(repeat_path)
    rng = random.Random(args.repeat_seed)
    picked = groups if args.repeat_groups >= len(groups) else rng.sample(groups, args.repeat_groups)
    tasks = []
    for group in picked:
        for index, cand in enumerate(group["candidates"]):
            key = candidate_id(group["group_id"], index)
            for k in range(args.repeat_k):
                if f"{key}@{k}" not in done:
                    tasks.append((group["group_id"], index, group["source"],
                                  group["reference"], cand, k))
    logger.info("重复评分：抽 %d 组、重复 %d 次，待打 %d 条", len(picked), args.repeat_k, len(tasks))
    if not tasks:
        return repeat_path

    judge = build_judge(weights, cfg_info["judge_api"], cfg_info["six_max_new_tokens"],
                        args.concurrency, cfg_info["max_input_tokens"])
    model = cfg_info["judge_api"].get("model", "")
    group_by_id = {g["group_id"]: g for g in groups}
    with open(repeat_path, "a", encoding="utf-8") as f:
        for start in range(0, len(tasks), args.batch_candidates):
            chunk = tasks[start:start + args.batch_candidates]
            items = [{"source": s, "reference": r, "candidate": c}
                     for _g, _i, s, r, c, _k in chunk]
            results = score_items_all(judge, weights, items)
            for (gid, idx, _s, _r, _c, k), metrics in zip(chunk, results):
                if metrics is None:
                    continue
                rec = make_record(group_by_id[gid], idx, metrics, reward_ctx, model)
                rec["repeat_index"] = k
                rec["candidate_id_orig"] = rec["candidate_id"]
                rec["candidate_id"] = f"{rec['candidate_id']}@{k}"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    logger.info("重复评分写入 → %s", repeat_path)
    return repeat_path


# ---------------------------------------------------------------------------
# 2. 分析
# ---------------------------------------------------------------------------
def _reward_series(records: Sequence[Dict[str, Any]], metric: str) -> List[float]:
    return [r["metrics"][metric]["reward"] for r in records]


def _sub_series(records: Sequence[Dict[str, Any]], metric: str, sub: str) -> List[float]:
    return [r["metrics"][metric]["sub_raw"][sub] for r in records]


def discriminability(values_by_group: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """一组"每组一个分数列表" → 组内区分度统计（标量/子项通用）。"""
    stds, ranges, ndist, pair, maxtie = [], [], [], [], []
    for vals in values_by_group:
        m = group_metric(vals)
        stds.append(m["std"]); ranges.append(m["range"])
        ndist.append(m["n_distinct"]); pair.append(m["pair_diff_ratio"])
        maxtie.append(m["max_tie"])
    n = len(values_by_group)
    zero = sum(1 for s in stds if s <= 1e-12)
    kmax = max((len(v) for v in values_by_group), default=0)
    return {
        "std": describe(stds),
        "zero_var_groups": int(zero),
        "zero_var_ratio": zero / n if n else 0.0,
        "nonzero_var_ratio": (1 - zero / n) if n else 0.0,
        "range": describe(ranges),
        "range_hist": {str(k): int(sum(1 for x in ranges if abs(x - k) < 1e-9))
                       for k in sorted({round(x, 6) for x in ranges})},
        "n_distinct_dist": {str(k): int(sum(1 for x in ndist if x == k))
                            for k in range(1, kmax + 1)},
        "pairwise_mean": float(np.mean(pair)) if pair else 0.0,
        "max_tie_dist": {str(k): int(sum(1 for x in maxtie if x == k))
                         for k in range(1, kmax + 1)},
    }


def _metric_global(records: Sequence[Dict[str, Any]], ms: MetricSpec) -> Dict[str, Any]:
    reward = _reward_series(records, ms.name)
    n = len(reward)
    hist, edges = np.histogram(reward, bins=20, range=(0.0, 1.0))
    out: Dict[str, Any] = {
        "label": ms.label,
        "n": n,
        "reward": {
            "describe": describe(reward),
            "hist_counts": [int(x) for x in hist],
            "hist_edges": [round(float(x), 3) for x in edges],
            "full_ratio": sum(1 for x in reward if x >= 1 - 1e-9) / n,
            "high_ratio_ge075": sum(1 for x in reward if x >= 0.75 - 1e-9) / n,
            "low_ratio_le05": sum(1 for x in reward if x <= 0.5 + 1e-9) / n,
            "zero_ratio": sum(1 for x in reward if x <= 1e-9) / n,
        },
        "subs": {},
    }
    for sub in ms.subs:
        vals = [int(v) for v in _sub_series(records, ms.name, sub)]
        counts = Counter(vals)
        out["subs"][sub] = {
            "label": ms.sub_label(sub),
            "describe": describe(vals),
            "raw_counts": {str(k): counts.get(k, 0) for k in range(5)},
            "raw_ratio": {str(k): counts.get(k, 0) / n for k in range(5)},
            "full_ratio": counts.get(4, 0) / n,
            "low_ratio_le2": sum(counts.get(k, 0) for k in range(3)) / n,
            "entropy": float(-sum((counts.get(k, 0) / n) * math.log(counts.get(k, 0) / n)
                                  for k in range(5) if counts.get(k, 0) > 0)),
        }
    return out


def _metric_group(grouped: Dict[str, List[Dict[str, Any]]], order: List[str],
                  ms: MetricSpec) -> Dict[str, Any]:
    reward_by_group = [[r["metrics"][ms.name]["reward"] for r in grouped[g]] for g in order]
    out: Dict[str, Any] = {
        "n_groups": len(order),
        "reward": discriminability(reward_by_group),
        "subs": {
            sub: discriminability([[r["metrics"][ms.name]["sub_raw"][sub] for r in grouped[g]]
                                   for g in order])
            for sub in ms.subs
        },
    }
    combo: Counter = Counter()
    none = 0
    for g in order:
        active = tuple(sub for sub in ms.subs
                       if group_metric([r["metrics"][ms.name]["sub_raw"][sub]
                                        for r in grouped[g]])["std"] > 1e-12)
        combo[active] += 1
        if not active:
            none += 1
    out["discriminability"] = {
        "tied_groups_per_sub": {sub: out["subs"][sub]["zero_var_groups"] for sub in ms.subs},
        "all_subs_tied_groups": none,
        "combination_counts": {"+".join(k) if k else "(none)": v for k, v in combo.items()},
    }
    return out


def _metric_length_bias(grouped: Dict[str, List[Dict[str, Any]]], order: List[str],
                        records: Sequence[Dict[str, Any]], ms: MetricSpec) -> Dict[str, Any]:
    reward = _reward_series(records, ms.name)
    chars = [r["candidate_chars"] for r in records]
    ratio = [r["len_ratio_to_ref"] for r in records]
    diff = [r["len_diff_to_ref"] for r in records]
    out: Dict[str, Any] = {"global": {
        "reward~candidate_chars": safe_corr(reward, chars),
        "reward~len_ratio_to_ref": safe_corr(reward, ratio),
        "reward~len_diff_to_ref": safe_corr(reward, diff),
    }}
    dy, dc, dr = [], [], []
    for g in order:
        rows = grouped[g]
        my = float(np.mean([r["metrics"][ms.name]["reward"] for r in rows]))
        mc = float(np.mean([r["candidate_chars"] for r in rows]))
        mr = float(np.mean([r["len_ratio_to_ref"] for r in rows]))
        for r in rows:
            dy.append(r["metrics"][ms.name]["reward"] - my)
            dc.append(r["candidate_chars"] - mc)
            dr.append(r["len_ratio_to_ref"] - mr)
    out["within_group"] = {
        "reward~candidate_chars": safe_corr(dy, dc),
        "reward~len_ratio_to_ref": safe_corr(dy, dr),
    }
    shortest_highest = longest_highest = shortest_lowest = 0
    usable = 0
    for g in order:
        rows = grouped[g]
        lens = [r["candidate_chars"] for r in rows]
        if len(set(lens)) == 1:
            continue
        usable += 1
        vals = [r["metrics"][ms.name]["reward"] for r in rows]
        best = max(vals); worst = min(vals)
        best_idx = {i for i, v in enumerate(vals) if v == best}
        worst_idx = {i for i, v in enumerate(vals) if v == worst}
        min_len, max_len = min(lens), max(lens)
        if best_idx & {i for i, L in enumerate(lens) if L == min_len}:
            shortest_highest += 1
        if best_idx & {i for i, L in enumerate(lens) if L == max_len}:
            longest_highest += 1
        if worst_idx & {i for i, L in enumerate(lens) if L == min_len}:
            shortest_lowest += 1
    out["shortest_highest"] = {"groups_usable": usable, "count": shortest_highest,
                               "ratio": shortest_highest / usable if usable else None}
    out["longest_highest"] = {"count": longest_highest,
                              "ratio": longest_highest / usable if usable else None}
    out["shortest_lowest"] = {"count": shortest_lowest,
                              "ratio": shortest_lowest / usable if usable else None}
    return out


def _metric_contribution(grouped: Dict[str, List[Dict[str, Any]]], order: List[str],
                         records: Sequence[Dict[str, Any]], ms: MetricSpec,
                         weights: Dict[str, float]) -> Dict[str, Any]:
    wsum = sum(weights.values()) or 1.0
    w = {k: weights[k] / wsum for k in ms.subs}
    reward = _reward_series(records, ms.name)
    total_mean = float(np.mean(reward))
    terms: Dict[str, Any] = {}
    for sub in ms.subs:
        contrib = [w[sub] * (r["metrics"][ms.name]["sub_raw"][sub] / MAX_RAW) for r in records]
        terms[sub] = {
            "label": ms.sub_label(sub),
            "weight": w[sub],
            "mean_contribution": float(np.mean(contrib)),
            "share_of_reward_mean": float(np.mean(contrib) / total_mean) if total_mean else None,
        }
    var_total, var_terms, cov_terms = [], defaultdict(list), defaultdict(list)
    for g in order:
        rows = grouped[g]
        x = {sub: np.asarray([r["metrics"][ms.name]["sub_raw"][sub] / MAX_RAW for r in rows], float)
             for sub in ms.subs}
        y = np.asarray([r["metrics"][ms.name]["reward"] for r in rows], float)
        var_total.append(float(y.var(ddof=1)) if y.size > 1 else 0.0)
        for sub in ms.subs:
            var_terms[sub].append(
                float(w[sub] ** 2 * (x[sub].var(ddof=1) if x[sub].size > 1 else 0.0)))
        for i, a in enumerate(ms.subs):
            for b in ms.subs[i + 1:]:
                cov = float(np.cov(x[a], x[b], ddof=1)[0, 1]) if x[a].size > 1 else 0.0
                cov_terms[f"{a}~{b}"].append(2 * w[a] * w[b] * cov)
    mean_var = float(np.mean(var_total)) if var_total else 0.0
    return {
        "terms": terms,
        "within_group_var_reward_mean": mean_var,
        "within_group_variance_share": {
            **{sub: (float(np.mean(var_terms[sub])) / mean_var if mean_var else None)
               for sub in ms.subs},
            **{k: (float(np.mean(v)) / mean_var if mean_var else None)
               for k, v in cov_terms.items()},
        },
    }


def _metric_ranking(grouped: Dict[str, List[Dict[str, Any]]], order: List[str],
                    ms: MetricSpec) -> Dict[str, Any]:
    top1_ties: Counter = Counter()
    corr_sub: Dict[str, List[float]] = defaultdict(list)
    for g in order:
        rows = grouped[g]
        vals = [r["metrics"][ms.name]["reward"] for r in rows]
        top1_ties[int(sum(1 for x in vals if x == max(vals)))] += 1
        for sub in ms.subs:
            c = safe_corr(vals, [r["metrics"][ms.name]["sub_raw"][sub] for r in rows])["spearman_r"]
            if c is not None:
                corr_sub[sub].append(c)
    return {
        "top1_tie_dist": {str(k): v for k, v in sorted(top1_ties.items())},
        "mean_within_group_spearman_reward_vs_sub": {
            sub: (float(np.mean(v)) if v else None) for sub, v in corr_sub.items()
        },
    }


def _cross_metric(records: Sequence[Dict[str, Any]],
                  grouped: Dict[str, List[Dict[str, Any]]], order: List[str],
                  names: Sequence[str]) -> Dict[str, Any]:
    series: Dict[str, List[float]] = {n: _reward_series(records, n) for n in names}
    series["candidate_chars"] = [r["candidate_chars"] for r in records]
    keys = list(names) + ["candidate_chars"]
    corr: Dict[str, Any] = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            corr[f"{a}~{b}"] = safe_corr(series[a], series[b])

    combo: Counter = Counter()
    none = 0
    per_metric = {n: 0 for n in names}
    for g in order:
        rows = grouped[g]
        active = tuple(n for n in names
                       if group_metric([r["metrics"][n]["reward"] for r in rows])["std"] > 1e-12)
        combo[active] += 1
        if not active:
            none += 1
        for n in active:
            per_metric[n] += 1

    pairs: Dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sp: List[float] = []
            top1 = 0
            n_g = 0
            for g in order:
                rows = grouped[g]
                va = [r["metrics"][a]["reward"] for r in rows]
                vb = [r["metrics"][b]["reward"] for r in rows]
                c = safe_corr(va, vb)["spearman_r"]
                if c is not None:
                    sp.append(c)
                n_g += 1
                if int(np.argmax(va)) == int(np.argmax(vb)):
                    top1 += 1
            pairs[f"{a}~{b}"] = {
                "mean_within_group_spearman": float(np.mean(sp)) if sp else None,
                "top1_agreement": top1 / n_g if n_g else None,
            }
    return {
        "correlations": corr,
        "discriminating_groups_per_metric": per_metric,
        "all_metrics_tied_groups": none,
        "combination_counts": {"+".join(k) if k else "(none)": v for k, v in combo.items()},
        "pairwise": pairs,
    }


def _field_values(records: Sequence[Dict[str, Any]], field: str) -> List[Optional[float]]:
    return [_rec_field(r, field) for r in records]


def _field_by_group(grouped: Dict[str, List[Dict[str, Any]]], order: List[str],
                    field: str) -> List[List[float]]:
    """每组一条列表；缺失（None）直接剔除，只有 ≥2 个有效值的组参与区分度计算。"""
    out: List[List[float]] = []
    for g in order:
        vals = [float(v) for v in (_rec_field(r, field) for r in grouped[g]) if v is not None]
        if len(vals) >= 2:
            out.append(vals)
    return out


def _reward_field_stats(records: Sequence[Dict[str, Any]],
                        grouped: Dict[str, List[Dict[str, Any]]], order: List[str],
                        field: str) -> Dict[str, Any]:
    vals = _field_values(records, field)
    valid = [float(v) for v in vals if v is not None]
    n = len(valid)
    out: Dict[str, Any] = {
        "field": field,
        "n": n,
        "n_missing": len(vals) - n,
        "usable_groups": len(_field_by_group(grouped, order, field)),
    }
    if not n:
        return out
    hist, edges = np.histogram(valid, bins=20, range=(0.0, 1.0))
    out["global"] = {
        "describe": describe(valid),
        "hist_counts": [int(x) for x in hist],
        "hist_edges": [round(float(x), 3) for x in edges],
        "full_ratio": sum(1 for x in valid if x >= 1 - 1e-9) / n,
        "zero_ratio": sum(1 for x in valid if x <= 1e-9) / n,
        "high_ratio_ge075": sum(1 for x in valid if x >= 0.75 - 1e-9) / n,
        "low_ratio_le05": sum(1 for x in valid if x <= 0.5 + 1e-9) / n,
    }
    out["group"] = discriminability(_field_by_group(grouped, order, field))
    return out


def _gate_stats(records: Sequence[Dict[str, Any]], cfg_info: Dict[str, Any]) -> Dict[str, Any]:
    """门控触发统计：文本门控 / 事实错误门控 / 重叠 / 裁判判定的错误数。"""
    plan = cfg_info["reward_plan"]
    n = len(records)
    text_reasons: Counter = Counter()
    n_text = n_fc_cfg = n_fc_strict = n_any_cfg = n_any_strict = 0
    overlap: Counter = Counter()
    n_with_fc = n_has_error = 0
    for r in records:
        gate = r.get("gate") or {}
        text_reason = gate.get("text_reason")
        fc_reason = gate.get("fc_reason")
        fc_strict = gate.get("fc_strict_reason")
        if text_reason:
            n_text += 1
            text_reasons[text_reason] += 1
        if fc_reason:
            n_fc_cfg += 1
        if fc_strict:
            n_fc_strict += 1
        if r.get("gated"):
            n_any_cfg += 1
        if text_reason or fc_strict:
            n_any_strict += 1
        if text_reason and fc_reason:
            overlap["both"] += 1
        elif text_reason:
            overlap["only_text"] += 1
        elif fc_reason:
            overlap["only_fc"] += 1
        else:
            overlap["none"] += 1
        if r["metrics"].get("fact_consistency") is not None:
            n_with_fc += 1
            if int(gate.get("fact_has_error") or 0) == 1:
                n_has_error += 1

    def _rates(counter: Counter, total: int) -> Dict[str, Any]:
        return {k: {"count": v, "rate": (v / total if total else None)}
                for k, v in sorted(counter.items())}

    return {
        "n": n,
        "gate_enabled": plan["gate_enabled"],
        "text_gate": {"count": n_text, "rate": n_text / n if n else None,
                      "reasons": _rates(text_reasons, n)},
        "fc_gate_config": {
            "enabled": plan["fact_error_gate_enabled"],
            "switch": plan["fact_has_error_gate"],
            "fact_consistency_enabled": plan["fact_consistency_enabled"],
            "count": n_fc_cfg, "rate": n_fc_cfg / n if n else None,
        },
        "fc_gate_strict": {
            "rule": "裁判判定“存在明确事实错误”(has_factual_error=1) → 整条 reward = 0",
            "forced_switch": plan["fc_forced_gate"],
            "count": n_fc_strict, "rate": n_fc_strict / n if n else None,
        },
        "effective_gated": {"count": n_any_cfg, "rate": n_any_cfg / n if n else None},
        "effective_gated_strict": {"count": n_any_strict,
                                   "rate": n_any_strict / n if n else None},
        "overlap_config_scope": _rates(overlap, n),
        "fc_scored_records": n_with_fc,
        "fact_error": {"count": n_has_error,
                       "rate": (n_has_error / n_with_fc if n_with_fc else None)},
    }


def _gate_group_stats(grouped: Dict[str, List[Dict[str, Any]]],
                      order: List[str]) -> Dict[str, Any]:
    """组级的门控覆盖：全组被门控 / 部分被门控 / 门控前后区分度变化。"""
    n = len(order)
    all_gated = partial_gated = free = 0
    dist_before_tied_after = 0
    for g in order:
        rows = grouped[g]
        n_gated = sum(1 for r in rows if r.get("gated"))
        if n_gated == len(rows) and rows:
            all_gated += 1
        elif n_gated:
            partial_gated += 1
        else:
            free += 1
        raw_v = [r["raw_reward"] for r in rows if r.get("raw_reward") is not None]
        eff_v = [r["effective_reward"] for r in rows
                 if r.get("effective_reward") is not None]
        if len(raw_v) >= 2 and len(eff_v) >= 2:
            raw_var = group_metric(raw_v)["std"] > 1e-12
            eff_var = group_metric(eff_v)["std"] > 1e-12
            if raw_var and not eff_var:
                dist_before_tied_after += 1
    return {
        "n_groups": n,
        "all_gated_groups": all_gated,
        "all_gated_ratio": all_gated / n if n else None,
        "partial_gated_groups": partial_gated,
        "partial_gated_ratio": partial_gated / n if n else None,
        "never_gated_groups": free,
        "never_gated_ratio": free / n if n else None,
        "distinguishable_before_tied_after": dist_before_tied_after,
        "distinguishable_before_tied_after_ratio": (
            dist_before_tied_after / n if n else None),
    }


def _gate_impact(grouped: Dict[str, List[Dict[str, Any]]],
                 order: List[str]) -> Dict[str, Any]:
    """**同一轮评分内**门控前(raw) → 门控后(effective) 的排序变化。

    与"两次独立评分之间的排序稳定性"是两件事，报告里分开写。
    """
    top1_changed = top1_same = 0
    ktaus: List[float] = []
    kt_undefined = 0
    keep = flip = tie_gain = tie_loss = pairs = 0
    n_usable = 0
    for g in order:
        rows = grouped[g]
        a = [r["raw_reward"] for r in rows if r["raw_reward"] is not None]
        b = [r["effective_reward"] for r in rows if r["effective_reward"] is not None]
        if len(a) < 2 or len(a) != len(b):
            continue
        n_usable += 1
        sa, sb = top1_sets(a, b)
        if sa == sb:
            top1_same += 1
        else:
            top1_changed += 1
        tau = kendall_tau_b(a, b)
        if tau is None:
            kt_undefined += 1
        else:
            ktaus.append(tau)
        st = pairwise_relation_stats(a, b)
        keep += st["keep"]; flip += st["flip"]
        tie_gain += st["tie_gain"]; tie_loss += st["tie_loss"]
        pairs += st["pairs"]
    return {
        "n_usable_groups": n_usable,
        "top1_changed_groups": top1_changed,
        "top1_changed_ratio": top1_changed / n_usable if n_usable else None,
        "top1_same_groups": top1_same,
        "kendall_tau_b": {
            "mean": float(np.mean(ktaus)) if ktaus else None,
            "valid_groups": len(ktaus),
            "undefined_groups": kt_undefined,
        },
        "pairwise": {
            "pairs": pairs, "keep": keep, "flip": flip,
            "tie_gain": tie_gain, "tie_loss": tie_loss,
            "flip_rate": flip / pairs if pairs else None,
            "keep_rate": keep / pairs if pairs else None,
            "tie_change_rate": (tie_gain + tie_loss) / pairs if pairs else None,
        },
    }


def _consistency_checks(records: Sequence[Dict[str, Any]],
                        cfg_info: Dict[str, Any]) -> Dict[str, Any]:
    """自检：离线统计用的 Reward 与训练 `compute_reward` 是否同口径。

    全部通过 `sfzy.rl.reward` 的返回值做断言式的交叉检查，不做二次实现。
    """
    plan = cfg_info["reward_plan"]
    weights = plan["normalized_weights"]
    n = len(records)
    n_gated = sum(1 for r in records if r.get("gated"))
    gated_zero = sum(1 for r in records
                     if r.get("gated") and r.get("effective_reward") == 0.0)
    ungated_pairs = [(r["raw_reward"], r["effective_reward"]) for r in records
                     if not r.get("gated") and r.get("raw_reward") is not None
                     and r.get("effective_reward") is not None]
    ungated_equal = sum(1 for a, b in ungated_pairs if abs(a - b) < 1e-9)
    fc_reasoned = [r for r in records if (r.get("gate") or {}).get("fc_reason")]
    fc_reason_zero = sum(1 for r in fc_reasoned if r.get("effective_reward") == 0.0)
    text_reasoned = [r for r in records if (r.get("gate") or {}).get("text_reason")]
    text_reason_zero = sum(1 for r in text_reasoned if r.get("effective_reward") == 0.0)
    term_sum_ok = term_sum_total = 0
    for r in records:
        terms = r.get("reward_terms") or {}
        if r.get("raw_reward") is None or not terms:
            continue
        term_sum_total += 1
        expect = sum(weights.get(k, 0.0) * float(v) for k, v in terms.items())
        if abs(expect - r["raw_reward"]) < 1e-6:
            term_sum_ok += 1
    return {
        "n": n,
        "gated_implies_zero": {
            "n_checked": n_gated, "n_ok": gated_zero,
            "pass": gated_zero == n_gated,
        },
        "ungated_equals_raw": {
            "n_checked": len(ungated_pairs), "n_ok": ungated_equal,
            "pass": ungated_equal == len(ungated_pairs),
        },
        "fc_gate_reason_implies_zero": {
            "n_checked": len(fc_reasoned), "n_ok": fc_reason_zero,
            "pass": fc_reason_zero == len(fc_reasoned),
        },
        "text_gate_reason_implies_zero": {
            "n_checked": len(text_reasoned), "n_ok": text_reason_zero,
            "pass": text_reason_zero == len(text_reasoned),
        },
        "raw_equals_weighted_terms": {
            "n_checked": term_sum_total, "n_ok": term_sum_ok,
            "pass": term_sum_ok == term_sum_total,
        },
        "reward_source": "sfzy.rl.reward.compute_reward（与 GRPO 训练同一函数）",
    }


def analyze(records: List[Dict[str, Any]], groups: List[Dict[str, Any]],
            weights: Dict[str, Dict[str, float]], cfg_info: Dict[str, Any]) -> Dict[str, Any]:
    grouped = _grouped(records)
    order = [g["group_id"] for g in groups if g["group_id"] in grouped]
    names = list(weights.keys())
    summary: Dict[str, Any] = {
        "meta": {
            **cfg_info,
            "input_groups": len(groups),
            "scored_groups": len(grouped),
            "n_candidates": len(records),
            "candidates_per_group": sorted({len(v) for v in grouped.values()}),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": {},
    }
    for name in names:
        ms = METRIC_BY_NAME[name]
        summary["metrics"][name] = {
            "label": ms.label,
            "weight_field": ms.weight_field,
            "weights": weights[name],
            "global": _metric_global(records, ms),
            "group": _metric_group(grouped, order, ms),
            "length_bias": _metric_length_bias(grouped, order, records, ms),
            "contribution": _metric_contribution(grouped, order, records, ms, weights[name]),
            "ranking": _metric_ranking(grouped, order, ms),
        }
    summary["cross_metric"] = _cross_metric(records, grouped, order, names)
    # ---- 真实 Reward：门控前 / 门控后 / "任一要素 0 分"口径 ----
    rewards: Dict[str, Any] = {}
    for field in REWARD_FIELDS:
        stats = _reward_field_stats(records, grouped, order, field)
        if field == "raw_reward":
            corr: Dict[str, Any] = {}
            for n in names:
                xs, ys = [], []
                for r in records:
                    a, b = _rec_field(r, n), _rec_field(r, field)
                    if a is None or b is None:
                        continue
                    xs.append(float(a)); ys.append(float(b))
                corr[n] = safe_corr(xs, ys)
            stats["correlations"] = corr
        rewards[field] = stats
    summary["rewards"] = rewards
    # 兼容旧报告结构：mixed = 门控前的混合奖励（raw_reward，直接用训练 Reward 函数算）
    summary["mixed"] = rewards["raw_reward"] if rewards["raw_reward"].get("n") else None
    summary["gate"] = _gate_stats(records, cfg_info)
    summary["gate_group"] = _gate_group_stats(grouped, order)
    summary["gate_impact"] = _gate_impact(grouped, order)
    summary["consistency"] = _consistency_checks(records, cfg_info)
    return summary


def _paired_values(pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
                   field: str) -> Tuple[List[float], List[float]]:
    a, b = [], []
    for base_rec, rep in pairs:
        x, y = _rec_field(base_rec, field), _rec_field(rep, field)
        if x is None or y is None:
            continue
        a.append(float(x)); b.append(float(y))
    return a, b


def _ordering_stability(pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
                        field: str) -> Dict[str, Any]:
    """两次独立评分之间的排序稳定性（一组一组算，再汇总）。"""
    by_group: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for base_rec, rep in pairs:
        by_group[base_rec["group_id"]].append((base_rec, rep))

    n_groups = n_usable = 0
    strict_top1 = inter_top1 = 0
    all_tied_groups = 0
    ktaus: List[float] = []
    kt_undefined = 0
    flip = keep = tie_gain = tie_loss = both_tied = pairs_total = 0
    for _gid, group_pairs in by_group.items():
        n_groups += 1
        a, b = _paired_values(group_pairs, field)
        if len(a) < 2:
            continue
        n_usable += 1
        sa, sb = top1_sets(a, b)
        strict_top1 += int(sa == sb)
        inter_top1 += int(bool(sa & sb))
        if len(sa) == len(a) and len(sb) == len(b):
            all_tied_groups += 1
        tau = kendall_tau_b(a, b)
        if tau is None:
            kt_undefined += 1
        else:
            ktaus.append(tau)
        st = pairwise_relation_stats(a, b)
        flip += st["flip"]; keep += st["keep"]
        tie_gain += st["tie_gain"]; tie_loss += st["tie_loss"]
        both_tied += st["both_tied_pairs"]
        pairs_total += st["pairs"]
    return {
        "n_groups": n_groups,
        "n_usable_groups": n_usable,
        "top1_strict_agreement": strict_top1 / n_usable if n_usable else None,
        "top1_intersection_rate": inter_top1 / n_usable if n_usable else None,
        "top1_all_tied_groups": all_tied_groups,
        "kendall_tau_b_mean": float(np.mean(ktaus)) if ktaus else None,
        "kendall_tau_b_valid_groups": len(ktaus),
        "kendall_tau_b_undefined_groups": kt_undefined,
        "pairwise": {
            "pairs": pairs_total, "both_tied_pairs": both_tied,
            "keep": keep, "flip": flip,
            "tie_gain": tie_gain, "tie_loss": tie_loss,
            "flip_rate": flip / pairs_total if pairs_total else None,
            "keep_rate": keep / pairs_total if pairs_total else None,
            "tie_change_rate": (tie_gain + tie_loss) / pairs_total if pairs_total else None,
        },
    }


def analyze_repeat(main: List[Dict[str, Any]], repeat: List[Dict[str, Any]],
                   weights: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """两次独立 Judge 评分之间的一致性 + 排序稳定性（逐指标 + 奖励字段）。"""
    fields = list(weights.keys()) + [f for f in REWARD_FIELDS]
    main_by = {(r["group_id"], r["candidate_index"]): r for r in main}
    rep_by: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in repeat:
        rep_by[(r["group_id"], r["candidate_index"])].append(r)

    out: Dict[str, Any] = {"n_repeat_records": len(repeat),
                           "n_candidates_repeated": len(rep_by),
                           "fields": fields,
                           "per_repeat_index": {}}
    for k in sorted({r.get("repeat_index", 0) for r in repeat}):
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for key, rows in rep_by.items():
            base_rec = main_by.get(key)
            cand_rows = [r for r in rows if r.get("repeat_index", 0) == k]
            if base_rec is not None and cand_rows:
                pairs.append((base_rec, cand_rows[0]))
        total = len(pairs)
        per_field: Dict[str, Any] = {}
        for field in fields:
            a, b = _paired_values(pairs, field)
            diffs = [abs(x - y) for x, y in zip(a, b)]
            exact = sum(1 for x, y in zip(a, b) if x == y)
            ms = METRIC_BY_NAME.get(field)
            entry: Dict[str, Any] = {
                "n_pairs": len(a),
                "exact_rate": exact / len(a) if a else None,
                "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
                **_ordering_stability(pairs, field),
            }
            if ms is not None:      # 指标还要看子项一致率
                sub_exact = {s: 0 for s in ms.subs}
                sub_diff: Dict[str, List[float]] = {s: [] for s in ms.subs}
                for base_rec, rep in pairs:
                    if _rec_field(base_rec, field) is None or _rec_field(rep, field) is None:
                        continue
                    for s in ms.subs:
                        x = base_rec["metrics"][field]["sub_raw"][s]
                        y = rep["metrics"][field]["sub_raw"][s]
                        sub_exact[s] += int(x == y)
                        sub_diff[s].append(abs(x - y))
                entry["sub_exact_rate"] = {
                    s: (sub_exact[s] / len(a) if a else None) for s in ms.subs}
                entry["sub_mean_abs_diff"] = {
                    s: (float(np.mean(sub_diff[s])) if sub_diff[s] else None)
                    for s in ms.subs}
            per_field[field] = entry
        out["per_repeat_index"][str(k)] = {
            "n": total,
            "per_field": per_field,
            # 兼容旧的键名（旧报告读的是 per_metric / group_ranking）
            "per_metric": {n: per_field[n] for n in weights if n in per_field},
            "group_ranking": {
                n: {"n_groups": per_field[n]["n_usable_groups"],
                    "top1_agreement": per_field[n]["top1_strict_agreement"],
                    "mean_kendall_tau": per_field[n]["kendall_tau_b_mean"]}
                for n in weights if n in per_field
            },
        }
    return out


# ---------------------------------------------------------------------------
# 3. 典型样本
# ---------------------------------------------------------------------------
def select_samples(records: List[Dict[str, Any]], groups: List[Dict[str, Any]],
                   names: Sequence[str], max_per_cat: int) -> List[Dict[str, Any]]:
    grouped = _grouped(records)
    text_by_id = {g["group_id"]: g for g in groups}
    # 只看"打分结果"和"输入原文"都有的组（--analysis-only 复用一个子集时更稳）
    order = [gid for gid in grouped if gid in text_by_id]
    if len(order) < len(grouped):
        logger.warning("有 %d 个组的打分结果在输入文件里找不到对应原文，已跳过样本挑选",
                       len(grouped) - len(order))
    picked: List[Dict[str, Any]] = []
    used: Dict[Tuple[str, str], set] = defaultdict(set)

    def snapshot(gid: str, scope: str, category: str, note: str) -> Dict[str, Any]:
        rows = grouped[gid]
        g = text_by_id[gid]
        rewards = {n: [r["metrics"][n]["reward"] for r in rows] for n in names}
        best = {n: max(rewards[n]) for n in names}
        first = names[0]
        return {
            "scope": scope,
            "category": category,
            "note": note,
            "group_id": gid,
            "source": g["source"],
            "reference": g["reference"],
            "candidates": [
                {
                    "index": r["candidate_index"],
                    "text": g["candidates"][r["candidate_index"]],
                    "chars": r["candidate_chars"],
                    "reward": r["metrics"][first]["reward"],
                    "is_best": r["metrics"][first]["reward"] == best[first],
                    "rewards": {n: r["metrics"][n]["reward"] for n in names},
                    "sub_raw": {n: dict(r["metrics"][n]["sub_raw"]) for n in names},
                    "raw_reward": r.get("raw_reward"),
                    "effective_reward": r.get("effective_reward"),
                    "effective_reward_fc": r.get("effective_reward_fc"),
                    "gated": bool(r.get("gated")),
                    "gate_reason": r.get("gate_reason") or "",
                }
                for r in rows
            ],
            "rewards": rewards,
            "effective_rewards": [r.get("effective_reward") for r in rows],
            "raw_rewards": [r.get("raw_reward") for r in rows],
        }

    def add(scope: str, category: str, gid: str, note: str) -> None:
        if len(used[(scope, category)]) >= max_per_cat or gid in used[(scope, category)]:
            return
        used[(scope, category)].add(gid)
        picked.append(snapshot(gid, scope, category, note))

    for metric in names:
        spread = {g: group_metric([r["metrics"][metric]["reward"] for r in grouped[g]])["range"]
                  for g in order}
        # 完全同分
        for g in sorted(order, key=lambda x: spread[x]):
            if spread[g] <= 1e-12:
                add(metric, "fully_tied", g, f"{metric}: 组内候选得分完全相同，无梯度")
        # 分差最大
        for g in sorted(order, key=lambda x: -spread[x]):
            add(metric, "max_spread", g, f"{metric}: 组内极差 = {spread[g]:.3f}")
        # 子项冲突：至少 2 个子项的最高分不是该指标最高分
        def conflict(gid: str) -> int:
            rows = grouped[gid]
            vals = [r["metrics"][metric]["reward"] for r in rows]
            top = max(vals)
            top_idx = {i for i, v in enumerate(vals) if v == top}
            n = 0
            for sub in METRIC_BY_NAME[metric].subs:
                sv = [r["metrics"][metric]["sub_raw"][sub] for r in rows]
                smax = max(sv)
                if not ({i for i, v in enumerate(sv) if v == smax} & top_idx):
                    n += 1
            return n
        for g in sorted(order, key=lambda x: (-conflict(x), -spread[x])):
            if spread[g] > 1e-12 and conflict(g) >= 2:
                add(metric, "sub_conflict", g, f"{metric}: {conflict(g)} 个子项的最高分不是总分最高分")
        # 最长/最短
        for g in order:
            rows = grouped[g]
            lens = [r["candidate_chars"] for r in rows]
            if len(set(lens)) == 1:
                continue
            vals = [r["metrics"][metric]["reward"] for r in rows]
            best = max(vals); worst = min(vals)
            if max(lens) in {L for r, L in zip(rows, lens) if r["metrics"][metric]["reward"] == best}:
                add(metric, "longest_highest", g, f"{metric}: 最长候选拿到最高分")
            if min(lens) in {L for r, L in zip(rows, lens) if r["metrics"][metric]["reward"] == worst}:
                add(metric, "shortest_lowest", g, f"{metric}: 最短候选拿到最低分")

    # 指标间 top1 冲突：选冲突最多的一对
    if len(names) >= 2:
        worst_pair, worst_n = None, -1
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                n = 0
                for g in order:
                    rows = grouped[g]
                    if int(np.argmax([r["metrics"][a]["reward"] for r in rows])) != \
                       int(np.argmax([r["metrics"][b]["reward"] for r in rows])):
                        n += 1
                if n > worst_n:
                    worst_pair, worst_n = (a, b), n
        if worst_pair:
            a, b = worst_pair
            for g in order:
                rows = grouped[g]
                if int(np.argmax([r["metrics"][a]["reward"] for r in rows])) != \
                   int(np.argmax([r["metrics"][b]["reward"] for r in rows])):
                    add("cross", "metric_top1_conflict", g,
                        f"{a} 与 {b} 的组内最高分不是同一个候选")
    return picked


# ---------------------------------------------------------------------------
# 4. 图表
# ---------------------------------------------------------------------------
def make_figures(records: List[Dict[str, Any]], summary: Dict[str, Any],
                 fig_dir: Path, names: Sequence[str]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(fig_dir)
    made: List[str] = []
    grouped = _grouped(records)

    # 1) 三指标标量分布
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.6), squeeze=False)
    for ax, name in zip(axes[0], names):
        vals = _reward_series(records, name)
        ax.hist(vals, bins=20, range=(0, 1), color="#3b6fb0", edgecolor="white")
        ax.set_title(f"{name} (n={len(vals)})")
        ax.set_xlabel("reward in [0,1]")
    fig.tight_layout()
    p = fig_dir / "fig_reward_dist.png"; fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p.name)

    # 2) 组内可区分性：std + 不同分数个数
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 6.4), squeeze=False)
    for j, name in enumerate(names):
        stds = [group_metric([r["metrics"][name]["reward"] for r in rows])["std"]
                for rows in grouped.values()]
        ndist = [group_metric([r["metrics"][name]["reward"] for r in rows])["n_distinct"]
                 for rows in grouped.values()]
        kmax = max((len(rows) for rows in grouped.values()), default=4)
        axes[0][j].hist(stds, bins=12, color="#4c9a6b", edgecolor="white")
        axes[0][j].set_title(f"within-group std · {name}")
        axes[1][j].bar(range(1, kmax + 1), [ndist.count(k) for k in range(1, kmax + 1)],
                       color="#4c9a6b", edgecolor="white")
        axes[1][j].set_title(f"distinct reward values · {name}")
        axes[1][j].set_xticks(range(1, kmax + 1))
    fig.tight_layout()
    p = fig_dir / "fig_group_discriminability.png"; fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p.name)

    # 3) 三指标相关性热力图（Spearman）
    keys = list(names) + ["candidate_chars"]
    mat = np.ones((len(keys), len(keys)))
    corr = summary["cross_metric"]["correlations"]
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                continue
            c = corr.get(f"{a}~{b}") or corr.get(f"{b}~{a}") or {}
            mat[i, j] = c.get("spearman_r") or 0.0
    fig, ax = plt.subplots(figsize=(1.4 * len(keys) + 3, 1.2 * len(keys) + 3))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation")
    fig.tight_layout()
    p = fig_dir / "fig_corr_heatmap.png"; fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p.name)

    # 4) 每个指标的子项原始分分布
    for name in names:
        ms = METRIC_BY_NAME[name]
        if not ms.subs:            # 新的事实一致性没有子项
            continue
        fig, axes = plt.subplots(1, len(ms.subs), figsize=(2.4 * len(ms.subs), 3.2), squeeze=False)
        for ax, sub in zip(axes[0], ms.subs):
            vals = [int(v) for v in _sub_series(records, name, sub)]
            counts = [vals.count(k) for k in range(5)]
            ax.bar(range(5), counts, color="#7aa6d6", edgecolor="white")
            ax.set_title(sub, fontsize=9)
            ax.set_xticks(range(5))
        fig.suptitle(f"{name} raw 0-4 score counts")
        fig.tight_layout()
        p = fig_dir / f"fig_{name}_sub_raw.png"; fig.savefig(p, dpi=130); plt.close(fig)
        made.append(p.name)

    # 5) 长度偏差（全局 + 组内去均值）
    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 6.4), squeeze=False)
    for j, name in enumerate(names):
        chars = [r["candidate_chars"] for r in records]
        vals = _reward_series(records, name)
        axes[0][j].scatter(chars, vals, s=8, alpha=0.5)
        axes[0][j].set_title(f"{name} vs chars (global)")
        dx, dy = [], []
        for rows in grouped.values():
            mx = float(np.mean([r["candidate_chars"] for r in rows]))
            my = float(np.mean([r["metrics"][name]["reward"] for r in rows]))
            dx += [r["candidate_chars"] - mx for r in rows]
            dy += [r["metrics"][name]["reward"] - my for r in rows]
        axes[1][j].scatter(dx, dy, s=8, alpha=0.5, color="#b0713b")
        axes[1][j].set_title(f"{name} (within-group demeaned)")
    fig.tight_layout()
    p = fig_dir / "fig_length_bias.png"; fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p.name)

    # 6) 真实 Reward：门控前 / 门控后分布 + 组内 std
    fields = [f for f in REWARD_FIELDS
              if any(_rec_field(r, f) is not None for r in records)]
    if fields:
        fig, axes = plt.subplots(2, len(fields), figsize=(5 * len(fields), 6.4), squeeze=False)
        for j, field in enumerate(fields):
            vals = [v for v in _field_values(records, field) if v is not None]
            axes[0][j].hist(vals, bins=20, range=(0, 1), color="#8a5fb0", edgecolor="white")
            axes[0][j].set_title(f"{field} (n={len(vals)})")
            axes[0][j].set_xlabel("reward")
            stds = [group_metric([v for v in (_rec_field(r, field) for r in rows)
                                  if v is not None])["std"]
                    for rows in grouped.values()
                    if len([v for v in (_rec_field(r, field) for r in rows)
                            if v is not None]) >= 2]
            axes[1][j].hist(stds, bins=12, color="#4c9a6b", edgecolor="white")
            axes[1][j].set_title(f"within-group std · {field}")
        fig.tight_layout()
        p = fig_dir / "fig_reward_gate.png"; fig.savefig(p, dpi=130); plt.close(fig)
        made.append(p.name)
    return made


# ---------------------------------------------------------------------------
# 5. 报告
# ---------------------------------------------------------------------------
def _verdict(m: Dict[str, Any], repeat_metric: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    g = m["group"]["reward"]
    gl = m["global"]["reward"]
    usable = g["nonzero_var_ratio"] >= 0.5
    ceiling = gl["high_ratio_ge075"] >= 0.9
    stable = None
    if repeat_metric:
        stable = repeat_metric.get("exact_rate")
    text = (
        f"组内可区分率 {_pct(g['nonzero_var_ratio'])}（零方差组 {_pct(g['zero_var_ratio'])}），"
        f"成对可区分率均值 {_pct(g['pairwise_mean'])}，"
        f"高分占比(≥0.75) {_pct(gl['high_ratio_ge075'])}、满分占比 {_pct(gl['full_ratio'])}"
    )
    if stable is not None:
        text += f"，重复评分完全一致率 {_pct(stable)}"
    text += "。"
    if usable:
        verdict = "**可作为 GRPO 组内相对信号**（多数组有组内差异）"
        if ceiling:
            verdict += "，但分数堆在上端，注意天花板效应压缩组内差距"
    else:
        verdict = "**组内相对信号偏弱**（过半数组内无差异），大量样本的组内优势会被压成 0"
    return {"usable": usable, "ceiling": ceiling, "repeat_exact_rate": stable,
            "summary": text, "verdict": verdict}


def _field_label(field: str) -> str:
    if field in METRIC_BY_NAME:
        return METRIC_BY_NAME[field].label
    return {
        "raw_reward": "raw_reward（门控前）",
        "effective_reward": "effective_reward（配置口径）",
        "effective_reward_fc": "effective_reward_fc（任一要素 0 分）",
    }.get(field, field)


def render_report(summary: Dict[str, Any], samples: List[Dict[str, Any]],
                  repeat: Optional[Dict[str, Any]], figures: Sequence[str],
                  source_chars: int) -> str:
    meta = summary["meta"]
    names = list(summary["metrics"].keys())
    rep_by_metric: Dict[str, Any] = {}
    rep_fields: Dict[str, Any] = {}
    if repeat and repeat.get("per_repeat_index"):
        first = next(iter(repeat["per_repeat_index"].values()))
        rep_fields = dict(first.get("per_field") or {})
        rep_by_metric = {n: rep_fields[n] for n in names if n in rep_fields}

    L: List[str] = []
    sec = 0
    L.append("# Judge 三指标区分度验证报告\n")
    L.append(f"- 配置：`{meta['config_path']}`")
    L.append(f"- 输入：{meta.get('input_groups', '?')} 组，已打分 {meta['scored_groups']} 组，"
             f"共 {meta['n_candidates']} 条候选（每组 "
             f"{meta['candidates_per_group']} 条）")
    L.append(f"- 裁判：`{meta.get('judge_api', {}).get('model', '?')}`"
             f"（base_url={meta.get('judge_api', {}).get('base_url', '?')}）")
    L.append(f"- 验证指标：{'、'.join(METRIC_BY_NAME[n].label for n in names)}")
    L.append(f"- 生成时间：{meta['created_at']}\n")
    L.append("**内部权重（来自配置）**：")
    for n in names:
        ms = METRIC_BY_NAME[n]
        w = summary["metrics"][n]["weights"]
        if ms.subs:
            L.append(f"- {ms.label}（`{n}`）："
                     + "、".join(f"{ms.sub_label(s)}={w[s]:g}" for s in ms.subs))
        else:
            L.append(f"- {ms.label}（`{n}`）：无子项权重（整体分 + 事实错误门控）")
    rp = meta["reward_plan"]
    L.append("- 真实 Reward（来自配置，复用训练代码 `sfzy.rl.reward.compute_reward`）："
             f"启用项 {rp['enabled_terms']}，归一权重 {rp['normalized_weights']}；"
             f"事实错误门控开关 fact_has_error={rp['fact_has_error_gate']}"
             f"（{'启用' if rp['fact_error_gate_enabled'] else '未启用'}）。")
    if rp.get("missing_signals"):
        L.append(f"- ⚠️ 本次 `--tasks` 缺少这些信号，混合奖励不可用：{rp['missing_signals']}")
    L.append("")

    # ---------- 1. 总览 ----------
    sec += 1
    L.append(f"## {sec}. 总览（每个指标能不能区分）\n")
    rows = []
    for n in names:
        m = summary["metrics"][n]
        v = _verdict(m, rep_by_metric.get(n))
        rows.append([
            METRIC_BY_NAME[n].label,
            _f(m["group"]["reward"]["nonzero_var_ratio"] * 100, 1) + "%",
            _f(m["global"]["reward"]["full_ratio"] * 100, 1) + "%",
            _f(m["global"]["reward"]["high_ratio_ge075"] * 100, 1) + "%",
            _f(m["global"]["reward"]["describe"]["mean"]),
            _f(v["repeat_exact_rate"] * 100 if v["repeat_exact_rate"] is not None else None, 1) + "%"
            if v["repeat_exact_rate"] is not None else "—",
        ])
    L.append(_table(["指标", "组内可区分率", "满分占比", "≥0.75 占比", "均值", "重复一致率"], rows) + "\n")
    for n in names:
        v = _verdict(summary["metrics"][n], rep_by_metric.get(n))
        L.append(f"- **{METRIC_BY_NAME[n].label}**：{v['summary']} → {v['verdict']}")
    L.append("")

    # ---------- 2. 逐指标 ----------
    sec += 1
    L.append(f"## {sec}. 逐指标：全局分布 / 组内区分度 / 子项 / 长度偏差\n")
    for n in names:
        ms = METRIC_BY_NAME[n]
        m = summary["metrics"][n]
        L.append(f"### 2.{names.index(n) + 1} {ms.label}（`{n}`）\n")
        L.append("**全局分布**：" + ", ".join(
            f"{k}={_f(v)}" for k, v in m["global"]["reward"]["describe"].items()) + "\n")
        rows = []
        for sub in ms.subs:
            s = m["global"]["subs"][sub]
            rows.append([ms.sub_label(sub), _f(s["describe"]["mean"]),
                         _f(s["describe"]["std"]), _pct(s["full_ratio"]),
                         _pct(s["low_ratio_le2"]), _f(s["entropy"])])
        L.append(_table(["子项", "均值", "标准差", "满分率", "≤2 占比", "熵"], rows) + "\n")
        # 组内区分度
        rg = m["group"]["reward"]
        L.append(f"**组内区分度（总分）**：零方差组 {_pct(rg['zero_var_ratio'])}，"
                 f"成对可区分率均值 {_pct(rg['pairwise_mean'])}，"
                 f"组内 σ 均值 {_f(rg['std']['mean'])}，极差均值 {_f(rg['range']['mean'])}。")
        L.append("- 组内不同分数个数分布：" +
                 "、".join(f"{k} 个:{v}" for k, v in rg["n_distinct_dist"].items()))
        L.append("- 各子项零方差组占比：" + "、".join(
            f"{ms.sub_label(s)}={_pct(m['group']['subs'][s]['zero_var_ratio'])}" for s in ms.subs))
        L.append(f"- 全部子项都无法区分的组：{m['group']['discriminability']['all_subs_tied_groups']}")
        L.append("- 子项可区分组合（→ 组数）：" + "、".join(
            f"{'+'.join(ms.sub_label(p) for p in combo.split('+')) if combo != '(none)' else '(none)'}"
            f"={cnt}" for combo, cnt in sorted(
                m["group"]["discriminability"]["combination_counts"].items(), key=lambda x: -x[1])) + "\n")
        # 贡献 / 方差分解
        ct = m["contribution"]
        rows = []
        for sub in ms.subs:
            t = ct["terms"][sub]
            share = ct["within_group_variance_share"].get(sub)
            rows.append([ms.sub_label(sub), _f(t["weight"]), _f(t["mean_contribution"]),
                         _pct(t["share_of_reward_mean"]), _pct(share)])
        L.append("分项贡献与组内方差占比：\n")
        L.append(_table(["子项", "归一权重", "平均贡献", "占总分均值", "占组内方差"], rows) + "\n")
        # 长度偏差
        lb = m["length_bias"]
        L.append("**长度偏差**（Spearman）：全局 字数 "
                 f"{_f(lb['global']['reward~candidate_chars']['spearman_r'])}、"
                 f"字数比 {_f(lb['global']['reward~len_ratio_to_ref']['spearman_r'])}；"
                 f"组内去均值后 字数 "
                 f"{_f(lb['within_group']['reward~candidate_chars']['spearman_r'])}、"
                 f"字数比 {_f(lb['within_group']['reward~len_ratio_to_ref']['spearman_r'])}。")
        L.append(f"- 组内“最短最高”{lb['shortest_highest']['count']}/"
                 f"{lb['shortest_highest']['groups_usable']}"
                 f"（{_pct(lb['shortest_highest']['ratio'])}）；"
                 f"“最长最高”{_pct(lb['longest_highest']['ratio'])}；"
                 f"“最短最低”{_pct(lb['shortest_lowest']['ratio'])}。")
        if rep_by_metric.get(n):
            rm = rep_by_metric[n]
            L.append(f"- 重复评分：总分完全一致率 {_pct(rm['exact_rate'])}，"
                     f"平均绝对差 {_f(rm['mean_abs_diff'])}；"
                     + "、".join(f"{ms.sub_label(s)}一致率 {_pct(rm['sub_exact_rate'][s])}"
                                for s in ms.subs))
        L.append("")

    # ---------- 3. 跨指标 ----------
    sec += 1
    L.append(f"## {sec}. 跨指标：相关性与区分度重叠\n")
    cm = summary["cross_metric"]
    L.append("两两相关（Pearson / Spearman）：\n")
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = cm["correlations"][f"{a}~{b}"]
            p = cm["pairwise"][f"{a}~{b}"]
            rows.append([f"{METRIC_BY_NAME[a].label} ~ {METRIC_BY_NAME[b].label}",
                         _f(c["pearson_r"]), _f(c["spearman_r"]),
                         _f(p["mean_within_group_spearman"]), _pct(p["top1_agreement"])])
    L.append(_table(["指标对", "Pearson", "Spearman", "组内平均 Spearman", "组内 top1 一致率"],
                    rows) + "\n")
    L.append("- 各指标能区分的组数：" + "、".join(
        f"{METRIC_BY_NAME[n].label}={cm['discriminating_groups_per_metric'][n]}" for n in names))
    L.append(f"- 三个指标在组内都无差异的组：{cm['all_metrics_tied_groups']}")
    L.append("- 可区分组合（→ 组数）：" + "、".join(
        f"{'+'.join(METRIC_BY_NAME[p].label for p in combo.split('+')) if combo != '(none)' else '(none)'}"
        f"={cnt}" for combo, cnt in sorted(cm["combination_counts"].items(), key=lambda x: -x[1])))
    L.append("")

    # ---------- 4. 真实 Reward 与门控 ----------
    sec += 1
    L.append(f"## {sec}. 真实 Reward 与门控（复用训练 Reward 函数）\n")
    rp = meta["reward_plan"]
    L.append("**口径**（`RewardSpec.from_config` + `sfzy.rl.reward.compute_reward`）：")
    L.append(f"- 启用项 / 归一权重：{rp['enabled_terms']} / {rp['normalized_weights']}")
    L.append(f"- 文本门控：enabled={rp['gate_enabled']}，"
             f"min_chars={rp['gate_cfg'].get('min_chars')}，"
             f"length_ratio_range={rp['gate_cfg'].get('length_ratio_range')}，"
             f"require_result_marker={rp['gate_cfg'].get('require_result_marker')}，"
             f"forbidden_prefixes={rp['gate_cfg'].get('forbidden_prefixes')}")
    L.append(f"- 事实错误门控：fact_consistency_enabled={rp['fact_consistency_enabled']}，"
             f"开关 fact_has_error={rp['fact_has_error_gate']}"
             f"（{'生效' if rp['fact_error_gate_enabled'] else '不生效'}）；"
             f"强制开启口径 fact_has_error={rp['fc_forced_gate']}")
    if not rp["fact_error_gate_enabled"]:
        L.append(f"- ⚠️ 当前配置 `fact_has_error`={rp['fact_has_error_gate']} → "
                 f"**事实错误门控未生效**；报告额外给出 `effective_reward_fc`"
                 f"（强制 fact_has_error={rp['fc_forced_gate']}）作为对照。")
    if rp.get("missing_signals"):
        L.append(f"- ⚠️ 缺少信号 {rp['missing_signals']}，raw/effective 奖励不可用。")
    L.append("")

    rewards = summary.get("rewards") or {}
    if rewards:
        L.append("### 4.1 门控前 / 门控后（全局 + 组内区分度）\n")
        rows = []
        for field, label in (("raw_reward", "raw（门控全关）"),
                             ("effective_reward", "effective（配置口径）"),
                             ("effective_reward_fc", "effective_fc（任一要素 0 分即出局）")):
            st = rewards.get(field) or {}
            if not st.get("n"):
                rows.append([label, "—", "—", "—", "—", "—", "—"])
                continue
            gg = st["group"]
            rows.append([
                label,
                _f(st["global"]["describe"]["mean"]),
                _f(st["global"]["describe"]["std"]),
                _pct(st["global"]["zero_ratio"]),
                _pct(st["global"]["full_ratio"]),
                _pct(gg["nonzero_var_ratio"]),
                _pct(gg["pairwise_mean"]),
            ])
        L.append(_table(["奖励口径", "均值", "标准差", "零奖励占比", "满分占比",
                         "组内可区分率", "成对可区分率"], rows) + "\n")
        for field, label in (("raw_reward", "raw"), ("effective_reward", "effective"),
                             ("effective_reward_fc", "effective_fc")):
            st = rewards.get(field) or {}
            if not st.get("n"):
                continue
            gg = st["group"]
            L.append(f"- **{label}**：有效 {st['n']} 条（缺失 {st['n_missing']}），"
                     f"可用组 {st['usable_groups']}；组内 σ 均值 {_f(gg['std']['mean'])}、"
                     f"极差均值 {_f(gg['range']['mean'])}，零方差组 {_pct(gg['zero_var_ratio'])}；"
                     "组内不同奖励个数分布：" +
                     "、".join(f"{k}个:{v}" for k, v in gg["n_distinct_dist"].items()))
        if (rewards.get("raw_reward") or {}).get("correlations"):
            L.append("- raw_reward 与各指标相关（Spearman）：" + "、".join(
                f"{METRIC_BY_NAME[n].label}="
                f"{_f(rewards['raw_reward']['correlations'][n]['spearman_r'])}" for n in names))
        L.append("")

    gs = summary.get("gate") or {}
    if gs:
        L.append("### 4.2 门控触发统计\n")
        n_all = gs["n"]
        text_reasons = "、".join(f"{k}={v['count']}（{_pct(v['rate'])}）"
                               for k, v in gs["text_gate"]["reasons"].items())
        L.append(f"- 文本门控触发：{gs['text_gate']['count']}/{n_all}"
                 f"（{_pct(gs['text_gate']['rate'])}）"
                 + (f"；原因分布：{text_reasons}" if text_reasons else ""))
        fc = gs["fc_gate_config"]
        L.append(f"- 事实错误门控（配置开关 fact_has_error={fc['switch']}，"
                 f"{'生效' if fc['enabled'] else '未生效'}）："
                 f"触发 {fc['count']}/{n_all}（{_pct(fc['rate'])}）")
        fcs = gs["fc_gate_strict"]
        L.append(f"- 事实错误门控（{fcs['rule']}，强制开关={fcs['forced_switch']}）："
                 f"触发 {fcs['count']}/{n_all}（{_pct(fcs['rate'])}）")
        L.append(f"- 最终被门控（配置口径）：{gs['effective_gated']['count']}/{n_all}"
                 f"（{_pct(gs['effective_gated']['rate'])}）；"
                 f"「强制事实门控」口径：{gs['effective_gated_strict']['count']}/{n_all}"
                 f"（{_pct(gs['effective_gated_strict']['rate'])}）")
        ov = gs["overlap_config_scope"]
        L.append("- 重叠情况（配置口径）：" + "、".join(
            f"{k}={v['count']}（{_pct(v['rate'])}）" for k, v in ov.items()))
        fe = gs["fact_error"]
        L.append(f"- 裁判判定“存在明确事实错误”(has_factual_error=1) 的候选："
                 f"{fe['count']}/{gs['fc_scored_records']}（{_pct(fe['rate'])}）\n")

    gg = summary.get("gate_group") or {}
    if gg:
        L.append("### 4.3 组级门控覆盖\n")
        L.append(f"- 全组候选都被门控：{gg['all_gated_groups']}/{gg['n_groups']}"
                 f"（{_pct(gg['all_gated_ratio'])}）")
        L.append(f"- 部分候选被门控：{gg['partial_gated_groups']}/{gg['n_groups']}"
                 f"（{_pct(gg['partial_gated_ratio'])}）")
        L.append(f"- 完全没有门控：{gg['never_gated_groups']}/{gg['n_groups']}"
                 f"（{_pct(gg['never_gated_ratio'])}）")
        L.append(f"- **门控前可区分、门控后完全同分**："
                 f"{gg['distinguishable_before_tied_after']}/{gg['n_groups']}"
                 f"（{_pct(gg['distinguishable_before_tied_after_ratio'])}）")
        L.append("")

    gi = summary.get("gate_impact") or {}
    if gi:
        L.append("### 4.4 门控对排序的影响（同一轮评分内 raw → effective）\n")
        L.append(f"- 可用组 {gi['n_usable_groups']}；Top-1 候选发生变化的组："
                 f"{gi['top1_changed_groups']}（{_pct(gi['top1_changed_ratio'])}）")
        kt = gi["kendall_tau_b"]
        L.append(f"- 组内 Kendall τ-b 均值 {_f(kt['mean'])}"
                 f"（有效组 {kt['valid_groups']}，全并列不可计算 {kt['undefined_groups']}）")
        pw = gi["pairwise"]
        L.append(f"- 成对排序翻转率 {_pct(pw['flip_rate'])}（{pw['flip']}/{pw['pairs']}），"
                 f"保持率 {_pct(pw['keep_rate'])}，"
                 f"并列关系变化率 {_pct(pw['tie_change_rate'])}"
                 f"（并列→非并列 {pw['tie_gain']}，非并列→并列 {pw['tie_loss']}）")
        raw_g = (rewards.get("raw_reward") or {}).get("group", {})
        eff_g = (rewards.get("effective_reward") or {}).get("group", {})
        if raw_g and eff_g:
            L.append(f"- 零方差组比例变化：raw {_pct(raw_g['zero_var_ratio'])} → "
                     f"effective {_pct(eff_g['zero_var_ratio'])}")
        L.append("")

    cs = summary.get("consistency") or {}
    if cs:
        L.append("### 4.5 与训练 Reward 口径的一致性自检\n")
        L.append(f"- 奖励来源：`{cs['reward_source']}`")
        L.append("")
        checks = [
            ("被门控 → effective=0", cs["gated_implies_zero"]),
            ("未被门控 → effective=raw", cs["ungated_equals_raw"]),
            ("FC 门控原因 → effective=0", cs["fc_gate_reason_implies_zero"]),
            ("文本门控原因 → effective=0", cs["text_gate_reason_implies_zero"]),
            ("raw = Σ(归一权重 × 分项)", cs["raw_equals_weighted_terms"]),
        ]
        rows = [[name, "✓ 通过" if c["pass"] else "✗ 不通过",
                 f"{c['n_ok']}/{c['n_checked']}"] for name, c in checks]
        L.append(_table(["检查项", "结果", "通过/检查"], rows) + "\n")

    # ---------- 5. 重复评分：一致性与排序稳定性 ----------
    sec += 1
    L.append(f"## {sec}. 重复评分：一致性与排序稳定性（两次独立 Judge）\n")
    if not rep_fields:
        L.append("未启用重复评分（`--no-repeat`），本节的排序稳定性无法评估。\n")
    else:
        fields = list(rep_fields.keys())
        L.append("**完全一致率 / 平均绝对差 / Top-1 一致 / Kendall τ-b / 成对翻转**\n")
        rows = []
        for field in fields:
            e = rep_fields[field]
            kt = f"{_f(e['kendall_tau_b_mean'])}（{e['kendall_tau_b_valid_groups']}"
            if e["kendall_tau_b_undefined_groups"]:
                kt += f"+{e['kendall_tau_b_undefined_groups']}N"
            kt += "）"
            rows.append([
                _field_label(field),
                _pct(e["exact_rate"]), _f(e["mean_abs_diff"], 4),
                _pct(e["top1_strict_agreement"]), _pct(e["top1_intersection_rate"]),
                kt, _pct(e["pairwise"]["flip_rate"]),
            ])
        L.append(_table(["字段", "完全一致率", "MAE", "Top-1 严格一致率",
                         "Top-1 有交集", "Kendall τ-b（有效+N不可算）", "成对翻转率"],
                        rows) + "\n")
        L.append("**成对关系变化（keep / flip / 并列变化）**\n")
        rows = []
        for field in fields:
            pw = rep_fields[field]["pairwise"]
            rows.append([_field_label(field), pw["pairs"], pw["keep"], pw["flip"],
                         pw["tie_gain"], pw["tie_loss"],
                         _pct(pw["keep_rate"]), _pct(pw["flip_rate"]),
                         _pct(pw["tie_change_rate"])])
        L.append(_table(["字段", "候选对数", "保持", "翻转", "并列→非并列", "非并列→并列",
                         "保持率", "翻转率", "并列变化率"], rows) + "\n")
        L.append("- 参与重复评分的组："
                 + "、".join(f"{_field_label(f)}={rep_fields[f]['n_usable_groups']}"
                             for f in fields))
        L.append("- 说明：Top-1 用**集合**比较（并列全部纳入，不取第一个）；"
                 "`Top-1 严格一致率`= 两个集合完全相同，`Top-1 有交集`= 交集非空。")
        L.append("- 说明：Kendall τ-b 对**全并列**的分组不可定义，标为 N（不计 0/1）；"
                 "`成对翻转率`只在符号由 > 变 <（或反之）时计数，"
                 "并列↔非并列的变化单独统计。")
        L.append("- 本节的排序稳定性是**两次独立评分之间**的；"
                 "第 4.4 节的门控排序变化是**同一轮评分内** raw→effective 的，二者不可混用。\n")

    # ---------- 6. 典型样本 ----------
    sec += 1
    L.append(f"## {sec}. 典型样本（人工复核）\n")
    cat_zh = {"fully_tied": "组内完全同分", "max_spread": "组内分差最大",
              "sub_conflict": "子项冲突", "longest_highest": "最长候选反而最高",
              "shortest_lowest": "最短候选得分最低", "metric_top1_conflict": "指标间 top1 冲突"}
    scopes = list(names) + ["cross"]
    for scope in scopes:
        items = [s for s in samples if s["scope"] == scope]
        if not items:
            continue
        title = "跨指标" if scope == "cross" else METRIC_BY_NAME[scope].label
        L.append(f"### {title}\n")
        for s in items:
            src = s["source"]
            shown = src if len(src) <= source_chars else (
                src[: source_chars // 2] + "\n……（原文中间省略）……\n" + src[-source_chars // 2:])
            L.append(f"**组 `{s['group_id']}`** — {cat_zh.get(s['category'], s['category'])}：{s['note']}\n")
            L.append(f"- 人工参考摘要（{len(s['reference'])} 字）：{s['reference']}")
            L.append(f"- 原文（{len(src)} 字，此处显示 {len(shown)} 字，完整见 `samples.jsonl`）：")
            L.append("```\n" + shown + "\n```")
            for n in names:
                vals = s["rewards"][n]
                best = max(vals)
                L.append(f"- {METRIC_BY_NAME[n].label}：" +
                         "、".join(f"#{i}={_f(v)}{'★' if v == best else ''}"
                                   for i, v in enumerate(vals)))
            if s.get("raw_rewards"):
                L.append("- raw_reward：" + "、".join(
                    f"#{i}={_f(v)}" for i, v in enumerate(s["raw_rewards"])))
                L.append("- effective_reward：" + "、".join(
                    f"#{i}={_f(v)}" for i, v in enumerate(s["effective_rewards"])))
            L.append("\n候选全文：")
            for c in sorted(s["candidates"], key=lambda x: x["index"]):
                tags = " ".join(f"{n}={_f(c['rewards'][n])}" for n in names)
                gate_tag = f"，effective={_f(c['effective_reward'])}"
                if c.get("gated"):
                    gate_tag += f"，已门控({c.get('gate_reason')})"
                L.append(f"  - #{c['index']}（{c['chars']} 字，{tags}{gate_tag}）：{c['text']}")
            L.append("")

    # ---------- 6. 结论 ----------
    sec += 1
    L.append(f"## {sec}. 结论\n")
    for n in names:
        v = _verdict(summary["metrics"][n], rep_by_metric.get(n))
        L.append(f"- **{METRIC_BY_NAME[n].label}**：{v['verdict']}")
    rewards = summary.get("rewards") or {}
    raw_g = (rewards.get("raw_reward") or {}).get("group") or {}
    eff_g = (rewards.get("effective_reward") or {}).get("group") or {}
    fcg = (rewards.get("effective_reward_fc") or {}).get("group") or {}
    if raw_g and eff_g:
        L.append(
            f"- **实际参与 GRPO 优势计算的 effective_reward**：组内可区分率 "
            f"{_pct(eff_g['nonzero_var_ratio'])}（门控前 raw 为 {_pct(raw_g['nonzero_var_ratio'])}）"
            f" → " + ("仍具组内差异，可作为组内相对信号。"
                      if eff_g["nonzero_var_ratio"] >= 0.5 else
                      "过半数组内已无差异，大量样本的组内优势会被压成 0，需先解决门控/天花板。"))
    if fcg and eff_g:
        L.append(
            f"- **「任一要素 0 分即出局」口径（effective_reward_fc）**：组内可区分率 "
            f"{_pct(fcg['nonzero_var_ratio'])}，零奖励占比 "
            f"{_pct((rewards.get('effective_reward_fc') or {}).get('global', {}).get('zero_ratio'))}。")
    L.append("")
    L.append("**限制**：判定由 LLM 裁判完成，非人工真值；本实验只能给出间接证据"
             "（组内方差、长度偏差、指标相关性、典型样本），不能证明“分差 = 真实质量差”。"
             "建议对第 5 节典型样本做人工复核后再定论。\n")
    if figures:
        L.append("## 附录 A. 图表\n")
        for name in figures:
            L.append(f"![{name}](figures/{name})\n")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Judge 三指标区分度验证：打分 + 统计 + Markdown 报告（不训练 GRPO）"
    )
    ap.add_argument("--config", default="configs/grpo_fact_cov_ee_t4.yaml",
                    help="从中读取三指标内部权重与 judge_api（不修改它）")
    ap.add_argument("--input", default="data/triples/sft_val_100_n4.jsonl")
    ap.add_argument("--candidates-key", default="outputs",
                    help="每组候选摘要的字段名（默认 outputs；缺失则退回单个 output）")
    ap.add_argument("--tasks", nargs="+", default=[m.name for m in METRICS],
                    choices=[m.name for m in METRICS],
                    help="要验证的指标（默认全部三个）")
    ap.add_argument("--out-dir", default=None,
                    help="默认 data/judge/metrics_<input 文件名>")
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 组（冒烟用）")
    ap.add_argument("--batch-candidates", type=int, default=16,
                    help="每次交给 Judge 的候选数（内部再按 judge_api.concurrency 并发）")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="覆盖配置里的 judge_api.concurrency")
    ap.add_argument("--no-repeat", action="store_true", help="不做重复评分")
    ap.add_argument("--repeat-groups", type=int, default=20, help="重复评分抽多少组")
    ap.add_argument("--repeat-k", type=int, default=1, help="每组重复评几次")
    ap.add_argument("--repeat-seed", type=int, default=42)
    ap.add_argument("--analysis-only", action="store_true",
                    help="跳过 API，只从已有 scores.jsonl 重算报告")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--max-samples-per-cat", type=int, default=2)
    ap.add_argument("--source-chars", type=int, default=3000,
                    help="报告中每个典型样本最多显示多少字原文（完整见 samples.jsonl）")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config_path = resolve(args.config)
    input_path = resolve(args.input)
    out_dir = resolve(args.out_dir) if args.out_dir else resolve(f"data/judge/metrics_{input_path.stem}")
    ensure_dir(out_dir)

    weights, cfg_info, reward_ctx = load_config_bundle(config_path, args.tasks)
    cfg_info["input_path"] = str(input_path)
    for name in args.tasks:
        logger.info("权重（来自 %s）%s：%s", config_path.name, name, weights[name])

    groups = load_groups(input_path, args.candidates_key, args.limit)
    all_groups = groups if not args.limit else load_groups(input_path, args.candidates_key, None)
    logger.info("读入 %d 组，共 %d 条候选", len(groups),
                sum(len(g["candidates"]) for g in groups))

    scores_path = out_dir / "scores.jsonl"
    repeat_path = out_dir / "repeat_scores.jsonl"
    if args.analysis_only:
        logger.info("--analysis-only：跳过 API，复用 %s", scores_path)
    else:
        run_scoring(args, groups, weights, cfg_info, reward_ctx, out_dir)
        run_repeat_scoring(args, groups, weights, cfg_info, reward_ctx, out_dir)

    if not scores_path.exists():
        raise SystemExit(f"没有打分结果 {scores_path}；去掉 --analysis-only 先跑一次。")
    records = read_jsonl_local(scores_path)
    if not records:
        raise SystemExit(f"{scores_path} 是空的。")
    # 旧版结果没有奖励/门控字段 → 用已存的指标分补算（不重新调 API）
    n_backfilled = backfill_reward_fields(records, all_groups, reward_ctx)
    if n_backfilled:
        cfg_info.setdefault("reward_fields_reconstructed", n_backfilled)

    summary = analyze(records, all_groups, weights, cfg_info)

    repeat_stats = None
    if repeat_path.exists():
        rrecs = read_jsonl_local(repeat_path)
        if rrecs:
            backfill_reward_fields(rrecs, all_groups, reward_ctx)
            repeat_stats = analyze_repeat(records, rrecs, weights)
            summary["repeat"] = repeat_stats

    samples = select_samples(records, all_groups, list(weights.keys()),
                             args.max_samples_per_cat)
    summary["samples"] = [{"scope": s["scope"], "category": s["category"],
                           "group_id": s["group_id"]} for s in samples]
    samples_path = out_dir / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    figures: List[str] = []
    if not args.no_figures:
        try:
            figures = make_figures(records, summary, out_dir / "figures", list(weights.keys()))
        except Exception as exc:  # noqa: BLE001 —— 画图失败不该毁掉报告
            logger.warning("画图失败（跳过）：%s", exc)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_report(summary, samples, repeat_stats, figures, args.source_chars)
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    logger.info("候选打分：%s", scores_path)
    if repeat_stats:
        logger.info("重复评分：%s", repeat_path)
    logger.info("统计量：%s\n典型样本：%s\n报告：%s", summary_path, samples_path, report_path)


if __name__ == "__main__":
    main()
