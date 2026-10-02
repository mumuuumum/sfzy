"""奖励聚合：门控 + 多个 reward term 的（归一化）加权求和。

============================ 结构 ================================
    gate（全局开关）→ 逐个启用项求值 → 按归一化权重求和 → RewardBreakdown

哪些项参与、权重多少、要不要裁判，全部由 `RewardSpec`（`reward_spec.py`）
从 YAML 决定；每项从哪来、消费哪个信号，由注册表（`reward_terms.py`）决定。
**本文件只负责"把已经算好的分项拼成总分"**，不关心某个 reward 具体怎么算。

================================ 多条信号 ================================
规则项（rouge_l）在进程内直接算；裁判项（fact_consistency 等）消费模型 B
一次批量产出的**命名信号**：

    judge_signals = {"fact_consistency": [0.8, 0.3, ...], "element_coverage": [...]}

一个裁判项拿不到自己需要的信号时**直接报错**，不静默降级 —— 否则你会以为
在跑某个 reward，其实那一路权重是 0。组内补缺（用组内均值）由 trainer 做，
到这里必须已经是有值的。

================================ 门控 ================================
门控是全局的硬开关：不通过直接 0 分，后面所有项都不算。它的作用是切断
"格式烂一点、但某个 reward 多拿分"的补偿路径。`gate.enabled=false` 时
整块跳过（连最短长度都不看）。

================================ 为什么还保留 extract_facts ================================
`extract_facts` 是规则口径（金额/日期/编号）的抽取函数，它**不参与奖励**，
只给 `tools/select_rl_prompts.py` 用来筛选 GRPO 的 prompt 池。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sfzy.rl.reward_spec import RewardSpec
from sfzy.rl.reward_terms import compute_rule_term

# --------------------------------------------------------------------------
# 事实提取（只用于 prompt 池筛选，不参与奖励）
# --------------------------------------------------------------------------
# 金额：支持 元 / 万元 / 亿元 三种单位，统一折算到"元"。
# 不折算的话，"48000元" 和 "4.8万元" 会被算成两个不同事实，信号全是噪声。
MONEY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(亿元|万元|元)")
UNIT_TO_YUAN = {"元": 1.0, "万元": 1e4, "亿元": 1e8}

# 日期：年 / 年-月 / 年-月-日 三档，按书写精度归一化
DATE_RE = re.compile(r"(\d{4})\s*年(?:\s*(\d{1,2})\s*月)?(?:\s*(\d{1,2})\s*日)?")

# 编号（案号、身份证号等）。必须放在金额和日期之后提取，否则会把
# 48000 这样的金额、2017 这样的年份也当成编号。
ID_RE = re.compile(r"\d{4,}")

# 法条号。按项目决定不考虑法条，先挖掉，避免"第107条"里的数字污染事实集合。
ARTICLE_RE = re.compile(r"第\s*\d+\s*条")

RESULT_MARKERS = ("判决如下", "判令", "驳回", "本院认为", "判决", "裁定")


@dataclass
class RewardBreakdown:
    """一条样本的奖励分项。

    `values` 是**动态**字典：开关了哪些 reward 就有哪些键，新增 reward
    不需要改这个数据类。只看 total 判断不出"这一版好在哪"，所以必须留分项。
    """

    total: float = 0.0
    gated: bool = False
    gate_reason: str = ""
    values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {"reward": self.total, "gated": float(self.gated)}
        for name, value in self.values.items():
            out[f"reward_{name}"] = value
        return out


def _mask(text: str, spans: Iterable[Tuple[int, int]]) -> str:
    """把已匹配的片段挖成空格，避免同一段被两种事实重复提取。"""
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


def extract_facts(text: str, kinds: Sequence[str] = ("money", "date", "id")) -> set:
    """抽出文本里的事实，返回归一化后的集合。

    归一化规则：
        金额  "48000元" / "4.8万元"   → "money:48000.00"
        日期  "2015年7月19日"          → "date:2015-07-19"
              "2017年5月"               → "date:2017-05"
        编号  "1023"（案号片段）        → "id:1023"

    提取顺序很重要：**先挖掉法条号，再依次挖金额、日期，最后剩下的
    4 位以上数字才算编号**。顺序反了的话，金额和年份都会被当成编号。
    """
    facts: set = set()

    # 第一刀：法条号（"第107条"）
    work = _mask(text, [m.span() for m in ARTICLE_RE.finditer(text)])

    if "money" in kinds:
        spans = []
        for m in MONEY_RE.finditer(work):
            facts.add(f"money:{float(m.group(1)) * UNIT_TO_YUAN[m.group(2)]:.2f}")
            spans.append(m.span())
        work = _mask(work, spans)

    if "date" in kinds:
        spans = []
        for m in DATE_RE.finditer(work):
            year, month, day = m.group(1), m.group(2), m.group(3)
            key = f"date:{year}"
            if month:
                key += f"-{int(month):02d}"
                if day:
                    key += f"-{int(day):02d}"
            facts.add(key)
            spans.append(m.span())
        work = _mask(work, spans)

    if "id" in kinds:
        for m in ID_RE.finditer(work):
            facts.add(f"id:{m.group()}")

    return facts


def check_gate(candidate: str, reference: str, gate_cfg: Mapping[str, Any]) -> Optional[str]:
    """返回 None 表示通过，否则返回被拦下的原因（写进日志用）。"""
    text = candidate.strip()
    ref_len = max(len(reference.strip()), 1)

    if len(text) < gate_cfg.get("min_chars", 60):
        return "too_short"

    ratio = len(text) / ref_len
    lo, hi = gate_cfg.get("length_ratio_range", [0.5, 1.5])
    if ratio < lo:
        return "length_too_short"
    if ratio > hi:
        return "length_too_long"

    for prefix in gate_cfg.get("forbidden_prefixes", []):
        if text.startswith(prefix):
            return f"prefix:{prefix}"

    if gate_cfg.get("require_result_marker"):
        if not any(marker in text for marker in RESULT_MARKERS):
            return "no_result_marker"

    return None


def _as_spec(spec: Optional[Any]) -> RewardSpec:
    """允许直接传字典（测试/工具方便），但只解析一次。"""
    return spec if isinstance(spec, RewardSpec) else RewardSpec.from_config(spec)


def _compute_one(
    candidate: str,
    reference: str,
    source: Optional[str],
    spec: RewardSpec,
    judge_signals: Mapping[str, Optional[float]],
) -> Tuple[float, RewardBreakdown]:
    bd = RewardBreakdown()

    # ---- 第一层：全局硬门控 ----
    if spec.gate.enabled:
        reason = check_gate(candidate, reference, spec.gate.cfg)
        if reason is not None:
            bd.gated = True
            bd.gate_reason = reason
            bd.total = 0.0
            return 0.0, bd

    # ---- 第二层：逐项求值 + 归一化加权求和 ----
    weights = spec.normalized_weights()
    total = 0.0
    for term in spec.enabled_terms:
        ts = term.spec
        if ts.source == "rule":
            value = compute_rule_term(
                ts.name, candidate, reference, source, rouge_mode=spec.rouge_mode
            )
        else:
            value = judge_signals.get(ts.signal)
            if value is None:
                raise ValueError(
                    f"reward term {ts.name!r} 需要裁判信号 {ts.signal!r}，但收到 None。\n"
                    "  检查 semantic.backend=fact 的裁判是否接上、以及组内补缺是否生效。"
                )
            value = float(value)
            # 量纲护栏：judge 信号必须是 [0,1]，接成 0-100 会把其它项压成噪声。
            if not 0.0 <= value <= 1.0 + 1e-6:
                raise ValueError(
                    f"裁判信号 {ts.signal!r} 的值 {value} 不在 [0,1]。"
                    "六要素 Judge 的 weighted_reward 本来就是 [0,1]，不做量纲换算。"
                )
        bd.values[ts.name] = value
        total += weights[ts.name] * value

    bd.total = total
    return total, bd


def compute_reward(
    candidate: str,
    reference: str,
    source: Optional[str] = None,
    spec: Optional[Any] = None,
    judge_signals: Optional[Mapping[str, Optional[float]]] = None,
) -> Tuple[float, RewardBreakdown]:
    """算一条样本的奖励。`judge_signals` 是这条样本的 {信号名: 值}。"""
    return _compute_one(
        candidate, reference, source, _as_spec(spec), judge_signals or {}
    )


def compute_rewards(
    candidates: Sequence[str],
    references: Sequence[str],
    sources: Optional[Sequence[str]] = None,
    spec: Optional[Any] = None,
    judge_signals: Optional[Mapping[str, Sequence[Optional[float]]]] = None,
) -> List[RewardBreakdown]:
    """批量打分，返回分项明细列表（GRPO 的 rollout 用这个）。

    `judge_signals` 是 {信号名: 每条候选一个值}；裁判比规则慢几个数量级，
    必须由调用方**批量**算好再传进来。`None` 表示该条这一路缺失 —— 到这里
    还缺失会报错，因为 trainer 应该已经用组内均值补过。
    """
    resolved = _as_spec(spec)
    sources = sources or [None] * len(candidates)
    judge_signals = dict(judge_signals or {})
    n = len(candidates)
    for name, values in judge_signals.items():
        if len(values) != n:
            raise ValueError(
                f"裁判信号 {name!r} 的个数 {len(values)} 与候选数 {n} 不一致"
            )

    out: List[RewardBreakdown] = []
    for i, (cand, ref, src) in enumerate(zip(candidates, references, sources)):
        signals = {name: values[i] for name, values in judge_signals.items()}
        out.append(_compute_one(cand, ref, src, resolved, signals)[1])
    return out


def summarize_gate_reasons(breakdowns: Sequence[RewardBreakdown]) -> Dict[str, int]:
    """统计被门控的比例和原因分布 —— 进训练日志。

    如果门控比例很高（比如 50%），说明问题出在生成配置或 prompt，
    **不是奖励设计**。没有这个统计你就分不清。
    """
    stats: Dict[str, int] = {"total": len(breakdowns), "gated": 0}
    for bd in breakdowns:
        if bd.gated:
            stats["gated"] += 1
            stats[bd.gate_reason] = stats.get(bd.gate_reason, 0) + 1
    return stats
