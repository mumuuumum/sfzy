"""规则奖励：把"摘要好不好"拆成几个可计算的项。

============================ 你要实现的文件 ============================

-------------------------------- 为什么用规则奖励而不是奖励模型 --------------------------------
训练一个 reward model 需要人类偏好数据，而我们没有。但摘要任务的
奖励恰好是**可验证**的：

    ROUGE          能算（eval/rouge.py 已经有了）
    法条引用命中    能匹配（99.3% 的参考摘要都写明了法律名称）
    长度偏离        能算（参考摘要平均 280 字）
    格式合规        能检查（有没有"判决如下"这类要素）

这些信号虽然不如人类偏好细腻，但**没有噪声、不需要标注、完全可复现**。
对 RLHF 来说，一个稳定的弱信号远好过一个不稳定的强信号。

-------------------------------- 分项设计（权重在 configs/grpo.yaml） --------------------------------
    rouge_l    0.5   与参考摘要的 ROUGE-L F1，主信号
    statute    0.3   生成摘要里提到的“全称+法条”能否覆盖应引用的那些
    length     0.1   长度偏离参考摘要的惩罚
    format     0.1   是否包含必备要素（案由、判决结果等）

**权重的含义要在报告里说清楚**：它不是"重要性排序"，而是"我们想让模型
优先优化什么"。ROUGE 占一半，说明主线仍是贴近参考摘要；
statute 占三成，说明我们希望模型把法条写出来（这正是 RAG 的价值所在）。

-------------------------------- 关于法条比对的重要决策 --------------------------------
我们**不做简称归一化**，而是直接比对“全称 + 具体法条”（如《中华人民共和国合同法》第六十条）。
理由：
1. 法律文书高度规范，裁判文书和人工摘要通常使用全称；
2. 模型输出也应遵守这一规范，若写简称则视为不合规，理应扣分；
3. 直接比对能更精确地衡量 RAG 是否检索到正确的法条，避免归一化带来的模糊匹配。
因此，`extract_law_articles` 只抽取以“中华人民共和国”或“最高人民法院”等开头的全称法名，
并提取其后的具体条文（第 X 条，支持“之一”）。
=====================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sfzy.eval.rouge import score_pair

# 匹配：全称法名 + 多个条文（允许中间有“、”连接），可带“款”等，但提取时去掉“款”
LAW_ARTICLE_RE = re.compile(
    r'《(中华人民共和国[^》]*?|最高人民法院[^》]*?)》'          # 法名全称
    r'((?:第[一二三四五六七八九十百千万零\d]+条(?:之一)?(?:[^，、《》]{0,10})?(?:、)?)+)'  # 条文部分
)

# 判决结果类表述。参考摘要里 72.5% 含这类词，是"摘要写全了"的最低标志。
RESULT_MARKERS = ("判决如下", "判令", "驳回", "本院认为", "判决")


@dataclass
class RewardBreakdown:
    """分项奖励，用来诊断"模型到底在往哪个方向优化"。"""

    total: float = 0.0
    rouge_l: float = 0.0
    statute: float = 0.0
    length: float = 0.0
    format: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward": self.total,
            "reward_rouge_l": self.rouge_l,
            "reward_statute": self.statute,
            "reward_length": self.length,
            "reward_format": self.format,
        }


def extract_law_articles(text: str) -> set[str]:
    """抽取文本里提到的所有“《全称法名》第 X 条”，返回集合。

    只匹配以“中华人民共和国”或“最高人民法院”等开头的全称法名，
    并提取其后的具体条文（支持“之一”）。不做简称归一化。
    同一法名 + 同一法条会去重。
    """
    results = set()
    matches = re.findall(LAW_ARTICLE_RE, text)

    for law_name, article_part in matches:
        # 拆分“第六十条第一款、第二百二十六条、第二百二十七条”这种结构
        articles = article_part.split("、")
        for art in articles:
            # 提取“第…条”，忽略“款”、“项”等
            article_match = re.match(r'(第[一二三四五六七八九十百千万零\d]+条(?:之一)?)', art)
            if article_match:
                article_str = article_match.group(1)
                results.add(f'《{law_name}》{article_str}')
            else:
                continue

    return results


def statute_reward(candidate: str, references: Sequence[str]) -> float:
    """法条命中率：候选摘要覆盖了应引用的法条中的多少。

    references 是从**参考摘要 + 原始文书**里抽出来的应引用法条集合
    （由 rag/citation.py 提供，M3 阶段接入）。每个元素形如
    “《中华人民共和国合同法》第六十条”。

    返回 [0, 1] 的覆盖率。references 为空时返回 0.0
    （无从判断，不给分也不扣分，权重要靠其它项撑）。

    实现：直接用 extract_law_articles 抽候选里的法条，再做集合覆盖计算。
    """
    if not references:
        return 0.0

    candidate_articles = extract_law_articles(candidate)
    ref_articles = set(references)

    if not ref_articles:
        return 0.0

    hit = candidate_articles & ref_articles
    return len(hit) / len(ref_articles)


def length_reward(candidate: str, reference: str, tolerance: float = 0.5) -> float:
    """长度惩罚，返回 [0, 1]：长度落在参考的 ±tolerance 内给满分，越远越低。

    为什么要这一项：ROUGE 是 F1，对长度的偏好很弱。模型很容易学会
    "多写一点、碰运气多命中几个 n-gram"——这在 ROUGE-2 上甚至可能涨分，
    但摘要会变得又臭又长。

    形状：1 - min(1, |Δ| / (reference_len * tolerance))
    当 |Δ| >= reference_len * tolerance 时得分为 0。
    tolerance=0.5 表示允许 ±50% 的长度偏差。
    """
    ref_len = len(reference)
    if ref_len == 0:
        return 0.0

    cand_len = len(candidate)
    diff = abs(cand_len - ref_len)
    # 当 diff 达到 ref_len * tolerance 时，得分为 0
    return max(0.0, 1.0 - diff / (ref_len * tolerance))


def format_reward(candidate: str) -> float:
    """格式合规度，返回 [0, 1]。

    最低要求：不能是空的、不能带"以下是摘要"这类前缀、要包含判决结果类表述。
    每满足一项给一部分分。

    这一项的作用是兜底——防止模型为了刷 ROUGE 输出一堆不成句的片段。
    """
    if not candidate or not candidate.strip():
        return 0.0

    score = 0.0

    # 1. 非空且长度合理（至少 10 个字）
    if len(candidate.strip()) >= 10:
        score += 0.3

    # 2. 不包含“以下是摘要”等前缀
    bad_prefixes = ("以下是摘要", "摘要：", "总结：")
    if not any(candidate.strip().startswith(p) for p in bad_prefixes):
        score += 0.3

    # 3. 包含判决结果类表述
    if any(marker in candidate for marker in RESULT_MARKERS):
        score += 0.4

    return min(1.0, score)


def compute_reward(
    candidate: str,
    reference: str,
    statutes: Sequence[str] = (),
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, RewardBreakdown]:
    """加权求和，返回 (总分, 分项明细)。

    默认权重见模块开头。总分是四项的加权和。

    **必须返回分项明细**：只看总分你没法判断"这一版比上一版好在哪"——
    是 ROUGE 涨了还是法条写全了？分项才是消融分析的最小单位。

    rouge_l 这一项直接用 eval/rouge.score_pair(candidate, reference)
    的 "rouge-l-f"，不要另写一套。
    """
    if weights is None:
        weights = {
            "rouge_l": 0.5,
            "statute": 0.3,
            "length": 0.1,
            "format": 0.1,
        }

    rouge_l = score_pair(candidate, reference).get("rouge-l-f", 0.0)
    statute = statute_reward(candidate, statutes)
    length = length_reward(candidate, reference)
    format_ = format_reward(candidate)

    total = (
        weights["rouge_l"] * rouge_l
        + weights["statute"] * statute
        + weights["length"] * length
        + weights["format"] * format_
    )

    breakdown = RewardBreakdown(
        total=total,
        rouge_l=rouge_l,
        statute=statute,
        length=length,
        format=format_,
    )

    return total, breakdown