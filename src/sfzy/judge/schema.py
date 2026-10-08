"""六要素事实一致性 Judge 的数据结构、权重与聚合规则。

============================ 这个 Judge 和之前的有什么不同 ============================
之前那套（`sfzy.eval.metrics`）是**点式打分**：把原文和候选一起丢给裁判，
让它给一个 0-100 的综合分。试了三版都栽在同一件事上 —— 裁判同时看到两份
文本时会以更短、更像摘要的那份（候选）为锚点，于是它算的是 precision
（候选写的有多少在原文里），而我们要的是 recall。实测表现为
`corr(裁判分, 长度比) = -0.35`，越短越占便宜。

这一版换了三个地方：

  1. **拆成六个要素**，每个要素单独判一次 —— 裁判每次只需要看两小段文字，
     没有"从哪份文本组织阅读"的空间
  2. **判据是"候选说的能不能被原文支持"**，明确不是覆盖率：
     省略不扣分，编造/写错才扣分
  3. **受限解码**：只在这五个数字 token 上取 argmax，不存在"输出不合法"
     这回事，也不用重试

============================ 六要素与权重 ============================
权重不是拍脑袋的：裁判结果 0.30（写反了整份摘要就废了）、法院查明 0.25
（事实主体）、诉讼请求与法律依据各 0.15、辩称 0.10、案由 0.05（最容易蒙对）。

从需求原文的公式：

    fact_reward = 0.05*case_type + 0.15*plaintiff_claims + 0.10*defendant_defenses
                + 0.25*court_facts + 0.15*legal_basis + 0.30*judgment_result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

# 顺序即打印顺序，也是 Judge 的调用顺序
ELEMENTS: tuple = (
    "case_type",
    "plaintiff_claims",
    "defendant_defenses",
    "court_facts",
    "legal_basis",
    "judgment_result",
)

ELEMENT_ZH: Dict[str, str] = {
    "case_type": "案件类型",
    "plaintiff_claims": "原告诉讼请求",
    "defendant_defenses": "被告辩称",
    "court_facts": "法院查明事实与说理",
    "legal_basis": "法律依据",
    "judgment_result": "裁判结果",
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "case_type": 0.05,
    "plaintiff_claims": 0.15,
    "defendant_defenses": 0.10,
    "court_facts": 0.25,
    "legal_basis": 0.15,
    "judgment_result": 0.30,
}

# 最低分不把案由算进来：案由只有五个字，蒙对的概率太高，会稀释这个信号
MIN_SCORE_ELEMENTS: tuple = tuple(e for e in ELEMENTS if e != "case_type")

MAX_SCORE = 4          # Judge 的量程 0-4


# 模型用来表达"这一项不存在"的占位词。需求写的是"返回空字符串"，
# 但实测模型会写"无"。
_PLACEHOLDERS = {
    "无", "无内容", "暂无", "不适用", "空", "没有", "未提及", "未涉及",
    "none", "null", "n/a", "na", "-", "—", "（无）", "(无)",
}


def coerce_element_value(value: Any) -> str:
    """把模型给的字段值统一成字符串。

    **0.5B 不会老实按 `"key": "字符串"` 输出**：实测它爱写成 JSON 数组
    （`"plaintiff_claims": ["要求偿还本金", "要求承担保证责任"]`），
    因为六个要素天然是多条并列的。数组本身是合理表达，所以这里把它接住，
    用分号连起来；不接的话正则兜底会一个字都匹配不到，六个要素全空。

    接不住的只有"整份 JSON 都坏了"，那由 `parse_six_json` 的三级降级处理。

    **占位词要归一成空字符串。** 实测模型对"不存在的要素"写的是 `"无"`，
    而 `empty_field_rule` 只把空白当空，于是它会以为"原文里确实有这一项"：

        文档被告辩称="无"、候选写了辩称 → 裁判看到"原文：无" → 判 0
        文档被告辩称="无"、候选没写     → 空字段规则 → 4

    又是"写了扣分、不写得分"的反向梯度。"无"和空字符串表达同一件事，
    必须在解析层拉平。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.strip("。.；;，,").lower() in _PLACEHOLDERS:
            return ""
        return stripped
    if isinstance(value, (list, tuple)):
        parts = [coerce_element_value(v) for v in value if v is not None]
        return "；".join(p for p in parts if p).strip()
    if isinstance(value, dict):
        return "；".join(
            f"{k}：{coerce_element_value(v)}" for k, v in value.items()
            if coerce_element_value(v)
        ).strip()
    return str(value).strip()


@dataclass(frozen=True)
class SixElements:
    """六要素。缺的要素是空字符串，不是 None —— 聚合时要区分"抽不出来"。"""

    case_type: str = ""
    plaintiff_claims: str = ""
    defendant_defenses: str = ""
    court_facts: str = ""
    legal_basis: str = ""
    judgment_result: str = ""

    def get(self, name: str) -> str:
        return getattr(self, name, "")

    def to_dict(self) -> Dict[str, str]:
        return {name: self.get(name) for name in ELEMENTS}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SixElements":
        """容错构造：缺键给空串、非字符串强转、None 当空串。

        小模型偶尔会把某个字段写成 null 或列表，不该因此让整条样本失败 ——
        那会静默丢掉一个训练样本，比多写几行容错代码贵得多。
        """
        out = {}
        for name in ELEMENTS:
            out[name] = coerce_element_value(data.get(name, ""))
        return cls(**out)

    def is_empty(self, name: str) -> bool:
        return not self.get(name).strip()

    def n_filled(self) -> int:
        """非空要素的个数。用来判断"这次提取是不是失败了"。"""
        return sum(1 for name in ELEMENTS if not self.is_empty(name))


@dataclass
class JudgeResult:
    """一条候选摘要的六要素判定结果。"""

    raw_scores: Dict[str, int] = field(default_factory=dict)         # 0-4
    scores: Dict[str, float] = field(default_factory=dict)           # 归一化 0-1
    weighted_reward: float = 0.0
    min_element_score: float = 0.0
    judgment_result_score: float = 0.0
    # 每个分数是怎么来的：judge（真调了模型）/ empty_rule（空字段规则）
    sources: Dict[str, str] = field(default_factory=dict)
    # 调试用：裁判原始输出（受限解码下就是那一位数字）与 5 档概率
    raw_outputs: Dict[str, str] = field(default_factory=dict)
    probs: Dict[str, List[float]] = field(default_factory=dict)
    candidate_id: str = ""
    # 这次判定产出的**所有信号**（事实一致性 / 关键要素覆盖率 / 以后新增的）。
    # 动态字典：加一个 reward 只需要往这里多放一个键，不用改数据类。
    signals: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "scores": dict(self.scores),
            "raw_scores": dict(self.raw_scores),
            "fact_reward": self.weighted_reward,
            "signals": dict(self.signals),
            "min_element_score": self.min_element_score,
            "judgment_result_score": self.judgment_result_score,
        }


# ---------------------------------------------------------------------------
# 空字段规则（需求第四节）
# ---------------------------------------------------------------------------
def empty_field_rule(document_element: str, candidate_element: str) -> Optional[int]:
    """返回免调 Judge 时该给的分；返回 None 表示**要调** Judge。

    三种情况（需求给的规则，一字不改）：

        doc 空 + cand 空   → 4   两边都没有，无从判断不一致
        doc 非空 + cand 空 → 4   **省略不是事实错误**，这是这套指标的核心立场
        doc 空 + cand 非空 → 调 Judge

    第三条为什么要调而不是直接判低分：六要素提取器的分类边界本身有噪声，
    "doc 里没有这一项" 很可能是提取时归到别的要素去了（比如被告的辩解被
    抽进了法院查明）。直接按 0 分处理会把提取器的错误算到策略模型头上。
    """
    doc_empty = not (document_element or "").strip()
    cand_empty = not (candidate_element or "").strip()
    if doc_empty and cand_empty:
        return MAX_SCORE
    if not doc_empty and cand_empty:
        return MAX_SCORE
    return None


# ---------------------------------------------------------------------------
# 聚合（需求第五、六、七节）
# ---------------------------------------------------------------------------
def aggregate(
    raw_scores: Dict[str, int],
    weights: Optional[Dict[str, float]] = None,
    sources: Optional[Dict[str, str]] = None,
    raw_outputs: Optional[Dict[str, str]] = None,
    probs: Optional[Dict[str, List[float]]] = None,
    candidate_id: str = "",
) -> JudgeResult:
    """把六个 0-4 整数分聚合成一条候选的奖励。

    缺任何一个要素都会报错 —— 静默用 0 补齐是这类聚合最危险的写法：
    少判一项会让奖励凭空掉一截，而且没人会发现。
    """
    weights = weights or DEFAULT_WEIGHTS
    missing = [name for name in ELEMENTS if name not in raw_scores]
    if missing:
        raise ValueError(f"缺少要素分数 {missing}，不能聚合")

    scores = {name: raw_scores[name] / float(MAX_SCORE) for name in ELEMENTS}
    weighted = sum(weights[name] * scores[name] for name in ELEMENTS)
    return JudgeResult(
        raw_scores={name: raw_scores[name] for name in ELEMENTS},
        scores=scores,
        weighted_reward=round(weighted, 6),
        min_element_score=min(scores[name] for name in MIN_SCORE_ELEMENTS),
        judgment_result_score=scores["judgment_result"],
        sources=dict(sources or {}),
        raw_outputs=dict(raw_outputs or {}),
        probs=dict(probs or {}),
        candidate_id=candidate_id,
    )


def aggregate_coverage(
    raw_scores: Dict[str, int],
    present: Sequence[str],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """关键要素覆盖率：R = Σ(w_i · s_i/4) / Σ w_i，只对 `present` 里的要素求和。

    `present` 是**参考摘要里真实存在的**要素。参考里没有的要素必须从分子和
    分母里同时去掉 —— 不能按 0 分算，那等于把"参考没写"记成"候选没覆盖"。
    一个要素都不存在时返回 None（这一条没有定义，交给调用方按缺失处理）。

    和事实一致性 `aggregate` 的区别：那个的权重和恒为 1，直接加权求和即可；
    这里的要素可能缺项，所以必须除以参与计算的权重和。
    """
    weights = weights or DEFAULT_WEIGHTS
    names = list(present)
    total = sum(weights[name] for name in names)
    if total <= 0:
        return None
    weighted = sum(weights[name] * raw_scores[name] / float(MAX_SCORE) for name in names)
    return round(weighted / total, 6)


# ---------------------------------------------------------------------------
# 训练日志统计（需求第十二节）
# ---------------------------------------------------------------------------
def summarize_scores(results: Sequence[JudgeResult]) -> Dict[str, float]:
    """训练日志要的那一组统计量。

    除了均值，还统计 **0/1/2/3/4 各档的比例** —— 只看均值分不清
    "大家都拿 3 分"和"一半 4 分一半 2 分"，而后者才说明判别力在起作用。
    """
    if not results:
        return {}
    n = len(results)
    stats: Dict[str, float] = {
        "mean_fact_reward": sum(r.weighted_reward for r in results) / n,
        "mean_min_element_score": sum(r.min_element_score for r in results) / n,
    }
    for name in ELEMENTS:
        stats[f"mean_{name}_score"] = sum(r.scores[name] for r in results) / n
    # 各档比例：把六要素所有判定合在一起看整体分布
    all_raw = [r.raw_scores[name] for r in results for name in ELEMENTS]
    for level in range(MAX_SCORE + 1):
        stats[f"ratio_score_{level}"] = sum(1 for v in all_raw if v == level) / len(all_raw)
    return stats
