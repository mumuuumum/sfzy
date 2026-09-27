"""规则奖励：门控 + 事实覆盖 + ROUGE-L。

================================ 为什么这么设计 ================================
从旧模型的 12172 条输出上分析出来的结论：

  * 模型漏掉的数字里 **98.5% 明确写在原文里** —— 不是幻觉，是漏抄
  * 训练集上 28.8% 的样本至少漏了一个原文里的事实
  * 漏事实的样本平均分 0.5443，干净样本 0.6175，**差 0.073**
  * 而且**输出比参考长 40% 时仍有 14.3% 漏事实** —— 和长度无关

所以痛点是"关键事实的遗漏"，不是幻觉、不是长度、不是格式。而 ROUGE 对
"漏一个 6 字符的金额"几乎不敏感（周围 90% 的文字都对），**所以拿 ROUGE
当奖励，模型根本没有动机去写数字**。

================================ 三层结构 ================================
  第一层  硬门控 —— 不满足直接 0 分，后面都不算。切断"牺牲格式换分数"的路径
  第二层  事实覆盖 ★ 主信号 —— 参考里的关键事实被覆盖了多少
  第三层  ROUGE-L —— 辅助，保持和官方评测口径一致

门控和加权求和的区别是**可补偿性**：加权求和下模型能学会"格式烂一点、
但 ROUGE 多拿分"，总分反而更高；门控切断的就是这条路。

================================ 防 hacking ================================
覆盖率有个明显的漏洞：模型可以把参考里所有数字都堆进去。
三道防线：
  1. 长度门控 —— 堆数字会把长度推高，直接撞上限
  2. `fact_precision` —— 输出的事实有多少真在原文里（监控指标，不进奖励）
  3. ROUGE 互补 —— 覆盖率和 ROUGE 反向变动就是报警

**前两条都要和覆盖率一起记录**，只看覆盖率会自欺欺人。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sfzy.eval.rouge import score_pair

# --------------------------------------------------------------------------
# 事实提取
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

DEFAULT_REWARD_CFG: Dict[str, Any] = {
    # flat  —— 所有项加权求和，可以互相补偿（对照组）
    # gated —— 门控项不满足直接 0 分，其余加权
    "mode": "gated",
    "weights": {"rouge_l": 0.5, "fact_coverage": 0.5, "length": 0.1},
    "fact_kinds": ["money", "date", "id"],
    "rouge_mode": "jieba",
    "gate": {
        "min_chars": 60,
        "length_ratio_range": [0.5, 1.5],
        "forbidden_prefixes": ["以下是", "摘要：", "摘要:", "本摘要", "这是"],
        "require_result_marker": True,
    },
}


@dataclass
class RewardBreakdown:
    """分项奖励。**必须返回分项** —— 只看总分判断不出"这一版好在哪"。"""

    total: float = 0.0
    gated: bool = False
    gate_reason: str = ""
    rouge_l: float = 0.0
    fact_coverage: float = 0.0
    fact_precision: float = 0.0
    length_ratio: float = 0.0
    n_ref_facts: int = 0
    n_missed_facts: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward": self.total,
            "reward_rouge_l": self.rouge_l,
            "reward_fact_coverage": self.fact_coverage,
            "fact_precision": self.fact_precision,
            "length_ratio": self.length_ratio,
            "n_ref_facts": self.n_ref_facts,
            "n_missed_facts": self.n_missed_facts,
            "gated": float(self.gated),
        }


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


def fact_coverage(
    candidate: str, reference: str, kinds: Sequence[str] = ("money", "date", "id")
) -> Tuple[float, int, int]:
    """参考里的事实被候选覆盖了多少。返回 (覆盖率, 参考事实数, 漏掉数)。

    参考摘要当金标准是合理的：实测它里面的数字 98.5% 都能在原文找到。
    参考为空时返回 0.0（无从判断）。
    """
    ref_facts = extract_facts(reference, kinds)
    if not ref_facts:
        return 0.0, 0, 0
    hit = len(ref_facts & extract_facts(candidate, kinds))
    return hit / len(ref_facts), len(ref_facts), len(ref_facts) - hit


def fact_precision(
    candidate: str, source: str, kinds: Sequence[str] = ("money", "date", "id")
) -> float:
    """候选里的事实有多少真在原文里。

    **这是监控指标，不是奖励项。** 用来发现"堆数字"型 hacking：
    模型为了刷覆盖率把参考里所有金额都塞进去时，它会下降。

    候选里一个事实都没有时返回 1.0 —— 那不是"不精确"，是"没写"，
    该由覆盖率去惩罚，不该在这里重复扣分。
    """
    cand_facts = extract_facts(candidate, kinds)
    if not cand_facts:
        return 1.0
    return len(cand_facts & extract_facts(source, kinds)) / len(cand_facts)


def length_reward(candidate: str, reference: str, tolerance: float = 0.5) -> float:
    """长度落在参考的 ±tolerance 内给满分，越远越低。**只在 flat 模式下用。**"""
    ref_len = len(reference.strip())
    if ref_len == 0:
        return 0.0
    deviation = abs(len(candidate.strip()) - ref_len) / ref_len
    return max(0.0, 1.0 - deviation / tolerance)


def check_gate(candidate: str, reference: str, gate_cfg: Dict[str, Any]) -> Optional[str]:
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


def compute_reward(
    candidate: str,
    reference: str,
    source: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, RewardBreakdown]:
    """算一条样本的奖励，返回 (总分, 分项明细)。

    `mode="flat"` 时所有项加权求和、长度是正向项（对照组）；
    `mode="gated"` 时先过门控、不通过直接 0 分。两种模式共用同一套
    分项计算，所以 A1/A2 的对比是干净的单变量。
    """
    full_cfg = _merge_cfg(cfg)
    weights = full_cfg["weights"]
    kinds = full_cfg["fact_kinds"]

    bd = RewardBreakdown()
    bd.rouge_l = score_pair(
        candidate.strip(), reference.strip(), mode=full_cfg["rouge_mode"]
    )["rouge-l-f"]
    bd.fact_coverage, bd.n_ref_facts, bd.n_missed_facts = fact_coverage(
        candidate, reference, kinds
    )
    bd.length_ratio = len(candidate.strip()) / max(len(reference.strip()), 1)
    if source:
        bd.fact_precision = fact_precision(candidate, source, kinds)

    if full_cfg["mode"] == "flat":
        bd.total = (
            weights["rouge_l"] * bd.rouge_l
            + weights["fact_coverage"] * bd.fact_coverage
            + weights["length"] * length_reward(candidate, reference)
        )
        return bd.total, bd

    reason = check_gate(candidate, reference, full_cfg["gate"])
    if reason is not None:
        bd.gated = True
        bd.gate_reason = reason
        bd.total = 0.0
        return 0.0, bd

    bd.total = (
        weights["rouge_l"] * bd.rouge_l
        + weights["fact_coverage"] * bd.fact_coverage
    )
    return bd.total, bd


def compute_rewards(
    candidates: Sequence[str],
    references: Sequence[str],
    sources: Optional[Sequence[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[RewardBreakdown]:
    """批量打分，返回分项明细列表（GRPO 的 rollout 用这个）。"""
    sources = sources or [None] * len(candidates)
    return [
        compute_reward(c, r, s, cfg)[1]
        for c, r, s in zip(candidates, references, sources)
    ]


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


def _merge_cfg(override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把配置和默认值深合并一层，允许只覆盖部分字段。"""
    import copy

    cfg = copy.deepcopy(DEFAULT_REWARD_CFG)
    if override:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg
