"""事实一致性奖励：门控 + 六要素 Judge 加权分 + ROUGE-L。

================================ 结构 ================================
  第一层  硬门控 —— 不满足直接 0 分，后面都不算。切断"牺牲格式换分数"的路径
  第二层  事实一致性 ★ 主信号 —— 六要素 Judge 的加权分 weighted_reward ∈ [0,1]
  第三层  ROUGE-L —— 辅助，保持和官方评测口径一致

门控和加权求和的区别是**可补偿性**：加权求和下模型能学会"格式烂一点、
但 ROUGE 多拿分"，总分反而更高；门控切断的就是这条路。

`reward.mode` 现在只有 `fact_judge` 一种。事实项**完全来自**
`sfzy/judge/` 的六要素 Judge，不做任何金额/日期/编号的规则匹配。

================================ 为什么还保留 extract_facts ================================
`extract_facts` 是规则口径（金额/日期/编号）的抽取函数，它**不参与奖励**，
只给 `tools/select_rl_prompts.py` 用来筛选 GRPO 的 prompt 池
（参考摘要里含金额的样本，是事实项最可能有信号的那一批）。
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sfzy.eval.rouge import score_pair

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

DEFAULT_REWARD_CFG: Dict[str, Any] = {
    # 现在只有 fact_judge 一种模式，见模块 docstring
    "mode": "fact_judge",
    "weights": {"rouge_l": 0.3, "fact": 0.7},
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
    fact_judge: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward": self.total,
            "reward_rouge_l": self.rouge_l,
            "reward_fact_judge": self.fact_judge,
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
    semantic: Optional[float] = None,
) -> Tuple[float, RewardBreakdown]:
    """算一条样本的奖励，返回 (总分, 分项明细)。

    `semantic` 是六要素事实一致性 Judge 的加权分，**本来就是 [0,1]**，
    不做任何量纲换算。它是这个奖励模式的唯一事实来源，缺了直接报错 ——
    静默降级只会让你以为在跑 Judge，其实没有。
    """
    full_cfg = _merge_cfg(cfg)
    weights = full_cfg["weights"]

    bd = RewardBreakdown()
    bd.rouge_l = score_pair(
        candidate.strip(), reference.strip(), mode=full_cfg["rouge_mode"]
    )["rouge-l-f"]
    if semantic is not None:
        # 量纲护栏。fact_judge 的分数是 [0,1]，接错了**不报错**，
        # 只是事实项凭空大 100 倍、把 ROUGE 压成噪声。
        if not 0.0 <= float(semantic) <= 1.0 + 1e-6:
            raise ValueError(
                f"mode=fact_judge 收到的事实一致性分 {semantic} 不在 [0,1]。"
                "这个模式只接受六要素 Judge 的 weighted_reward"
                "（semantic.backend=fact）。"
            )
        bd.fact_judge = float(semantic)

    reason = check_gate(candidate, reference, full_cfg["gate"])
    if reason is not None:
        bd.gated = True
        bd.gate_reason = reason
        bd.total = 0.0
        return 0.0, bd

    if semantic is None:
        raise ValueError(
            "mode=fact_judge 需要每条样本都有六要素事实一致性分，收到 None。"
            "检查 semantic.backend=fact 的裁判是否传给了 compute_rewards，"
            "以及脚本启动日志里的 '语义裁判' 行。"
        )

    bd.total = weights["rouge_l"] * bd.rouge_l + weights["fact"] * bd.fact_judge
    return bd.total, bd


def compute_rewards(
    candidates: Sequence[str],
    references: Sequence[str],
    sources: Optional[Sequence[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    semantic_scores: Optional[Sequence[Optional[float]]] = None,
) -> List[RewardBreakdown]:
    """批量打分，返回分项明细列表（GRPO 的 rollout 用这个）。

    `semantic_scores` 是六要素裁判分 ∈ [0,1]，由调用方先批量算好再传进来 ——
    裁判模型比规则慢几个数量级，必须批处理，不能在这里逐条调用。
    """
    sources = sources or [None] * len(candidates)
    if semantic_scores is None:
        semantic_scores = [None] * len(candidates)
    if len(semantic_scores) != len(candidates):
        raise ValueError(
            f"裁判分个数 {len(semantic_scores)} 与候选数 {len(candidates)} 不一致"
        )
    return [
        compute_reward(c, r, s, cfg, semantic=sem)[1]
        for c, r, s, sem in zip(candidates, references, sources, semantic_scores)
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
    """把配置和默认值深合并一层，允许只覆盖部分字段。

    现在只支持 `mode: fact_judge`。老配置里的 rouge_only / flat / gated /
    gated_judge 已经删除，写进来会直接报错，而不是静默跑成别的口径。
    """
    cfg = copy.deepcopy(DEFAULT_REWARD_CFG)
    if override:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    if cfg["mode"] != "fact_judge":
        raise ValueError(
            f"未知的 reward.mode={cfg['mode']!r}：现在只支持 fact_judge。"
        )
    return cfg
