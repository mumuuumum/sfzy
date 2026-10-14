"""逐要素 ROUGE：**候选要素** 对 **人工要素**，权重来自配置。

============================ 口径 ============================
每一项都取 F-score（和 `sfzy.eval.rouge.score_pair` 一致），总分是三项加权：

    score(要素) = w1·F1(ROUGE-1) + w2·F1(ROUGE-2) + wL·F1(ROUGE-L)

默认权重是官方口径 `0.2 / 0.4 / 0.4`（ROUGE-2/L 更重，词序与短语搭配比单词覆盖
更重要）。权重写在配置文件里，代码不存默认值。

三个要素之间再按 `element_weights` 合成"总分"（默认沿用 judge 六要素里这三项的
相对重要性 0.30 / 0.20 / 0.50）。

============================ 空要素怎么算 ============================
这是最容易把数字算歪的地方，所以口径写死并单独统计：

  * **两侧都空**（人工摘要本来就没有被告辩称、候选也没有）：`exclude` —— 不计入
    平均。把它算成 1.0 会奖励"两边都没写"，算成 0.0 会惩罚"本来就该没有"的样本，
    两种都会让模型学歪。
  * **只有一侧空**：按 `one_side_empty_score`（默认 0.0）——该说没说的是真丢分。
  * **抽取失败**（`ok=False`）：不进分数，单独计数，避免把"API 格式问题"混进指标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from sfzy.elements.config import RougeConfig
from sfzy.elements.schema import ELEMENTS, ELEMENT_ZH, ElementSet
from sfzy.eval.rouge import score_pair


@dataclass
class ElementScore:
    """一个要素的得分。"""

    element: str
    status: str                 # scored | both_empty | one_side_empty
    score: Optional[float]      # None = 不计入（两侧都空且口径为 exclude）
    rouge1_f: Optional[float] = None
    rouge2_f: Optional[float] = None
    rougel_f: Optional[float] = None
    ref_chars: int = 0
    hyp_chars: int = 0

    @property
    def counted(self) -> bool:
        return self.score is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element": self.element,
            "element_zh": ELEMENT_ZH.get(self.element, self.element),
            "status": self.status,
            "score": self.score,
            "rouge1_f": self.rouge1_f,
            "rouge2_f": self.rouge2_f,
            "rougel_f": self.rougel_f,
            "ref_chars": self.ref_chars,
            "hyp_chars": self.hyp_chars,
        }


def weighted_score(
    rouge1_f: float, rouge2_f: float, rougel_f: float, weights: Mapping[str, float]
) -> float:
    """三项 F 的加权和。权重不需要归一化——这里**按权重和归一**，这样配置里写
    `0.2/0.4/0.4` 还是 `2/4/4` 得到的分数一样。"""

    total = weights["rouge1"] + weights["rouge2"] + weights["rougel"]
    if total <= 0:                                            # pragma: no cover - 配置已校验
        raise ValueError("ROUGE 权重之和必须大于 0")
    return (
        weights["rouge1"] * rouge1_f
        + weights["rouge2"] * rouge2_f
        + weights["rougel"] * rougel_f
    ) / total


def score_elements(
    reference: ElementSet,
    hypothesis: ElementSet,
    *,
    config: RougeConfig,
) -> Dict[str, ElementScore]:
    """逐要素算分，返回 {要素: ElementScore}。"""

    out: Dict[str, ElementScore] = {}
    for name in ELEMENTS:
        ref_text = reference.get(name)
        hyp_text = hypothesis.get(name)
        ref_chars, hyp_chars = len(ref_text), len(hyp_text)

        if not ref_text and not hyp_text:
            if config.both_empty == "exclude":
                out[name] = ElementScore(name, "both_empty", None, ref_chars=0, hyp_chars=0)
                continue
            out[name] = ElementScore(name, "both_empty", float(config.one_side_empty_score))
            continue
        if not ref_text or not hyp_text:
            out[name] = ElementScore(
                name,
                "one_side_empty",
                float(config.one_side_empty_score),
                ref_chars=ref_chars,
                hyp_chars=hyp_chars,
            )
            continue

        detail = score_pair(hyp_text, ref_text, mode=config.mode)
        r1, r2, rl = detail["rouge-1-f"], detail["rouge-2-f"], detail["rouge-l-f"]
        out[name] = ElementScore(
            name,
            "scored",
            weighted_score(r1, r2, rl, config.weights),
            rouge1_f=r1,
            rouge2_f=r2,
            rougel_f=rl,
            ref_chars=ref_chars,
            hyp_chars=hyp_chars,
        )
    return out


def aggregate(
    scores: Mapping[str, ElementScore], element_weights: Mapping[str, float]
) -> Optional[float]:
    """把逐要素分合成总分。

    只对**计入的要素**做加权平均，并按这些要素的权重重新归一——否则"人工摘要
    本来没有被告辩称"这种样本会被系统性拉低，而它跟模型好坏无关。
    """

    total = 0.0
    weight_sum = 0.0
    for name, score in scores.items():
        if not score.counted:
            continue
        weight = float(element_weights.get(name, 0.0))
        if weight <= 0:
            continue
        total += weight * float(score.score)
        weight_sum += weight
    if weight_sum <= 0:
        return None
    return total / weight_sum


__all__ = [
    "ElementScore",
    "score_elements",
    "aggregate",
    "weighted_score",
]
