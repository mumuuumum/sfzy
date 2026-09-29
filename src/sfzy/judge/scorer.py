"""把六要素事实一致性 Judge 接到 GRPO 奖励管线上的适配层。

============================ 为什么需要这一层 ============================
`FactConsistencyJudge` 的入口是 `judge_candidates(document, candidates)` ——
**一次吃一篇文书 + 一组候选**。而 GRPO 的裁判接口（`SemanticScorer`）是扁平的

    score_batch(items) -> List[Optional[float]]        # item = {candidate, reference, source}

中间的落差就是本文件。要做的只有两件事：

  1. 按 `source`（文书原文）把扁平的 items 分回各组
  2. 逐组调用 `judge_candidates`，把每条候选的 `weighted_reward` 摊回原位置

第 1 步不能省。不分组就等于逐条调用，文档六要素会被重复提取 G 次；而候选
六要素也失去批量 —— 需求第十节的"6 要素 × G 候选一次组 batch"直接作废。

============================ 和其它裁判后端的区别 ============================
`local` / `rank` / `api` 返回的是 0-100 的综合分；本后端返回的是
`JudgeResult.weighted_reward`，**已经是 [0,1]**，且它的每一项都来自
Qwen Judge 对"候选陈述能否被原文支持"的判断，不含任何金额/日期/法条规则匹配。

因此用它时奖励模式要选 `fact_judge` —— 那个模式不做 `semantic_scale` 换算
（值本来就是 0-1），见 `rl/reward.py`。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sfzy.eval.metrics import SemanticScorer
from sfzy.judge.judge import FactConsistencyJudge
from sfzy.judge.schema import JudgeResult, summarize_scores


class FactConsistencyScorer(SemanticScorer):
    """六要素事实一致性 Judge 的 `SemanticScorer` 适配器。

        judge = FactConsistencyJudge("models/Qwen2.5-1.5B-Instruct", device="cuda:1")
        scorer = FactConsistencyScorer(judge)
        rewards = scorer.score_batch(items)        # 每条候选 ∈ [0, 1]

    `score_batch` 的输入形态和 GRPO 的 rollout 完全对齐：同一篇文书会连续给出
    G 条候选，`source` 字段是文书原文（不是参考摘要）—— 事实一致性判的是
    "候选 vs 原文"，参考摘要在这一步用不上。
    """

    name = "judge_fact"
    # 本后端自己按 source 分组，不需要 trainer 对齐 group_size（那是排序裁判的要求）
    kind = "fact_consistency"

    def __init__(
        self,
        judge: FactConsistencyJudge,
        fail_soft: bool = True,
    ) -> None:
        """
        `fail_soft=True`（默认）时，某一篇文书的判定整体失败（比如提取器抛
        `ExtractionFailure`、显存抖动）只把这一组标成 `None`，交给 trainer 的
        `fill_semantic_gaps` 用组内均值补上，**不中断训练**。理由和
        `eval/metrics.py` 里点式裁判一样：一次失败不该毁掉整轮几小时的 run。
        整批失败时 `None` 会原样透传，调用方按"这条打分失败"处理。
        """
        self.judge = judge
        self.fail_soft = fail_soft
        # 最近一次 score_batch 的逐条判定结果，供训练日志取统计量（需求第十二节）
        self.last_results: List[JudgeResult] = []
        self.last_errors: List[str] = []

    def clear_cache(self) -> None:
        self.judge.clear_cache()

    def score_batch(
        self, items: Sequence[Dict[str, Any]]
    ) -> List[Optional[float]]:
        """返回每条候选的加权事实一致性奖励 ∈ [0, 1]，失败为 `None`。"""
        norm = [_normalize(it) for it in items]
        scores: List[Optional[float]] = [None] * len(norm)
        self.last_results = []
        self.last_errors = []

        # 按 source 分组，但保持每组内的原始顺序 —— 分数要摊回原位置，
        # 顺序错了不报错，只是奖励悄悄错位。
        groups: Dict[str, List[int]] = {}
        for i, it in enumerate(norm):
            groups.setdefault(it["source"], []).append(i)

        for source, idxs in groups.items():
            candidates = [norm[i]["candidate"] for i in idxs]
            ids = [norm[i]["id"] or str(i) for i in idxs]
            try:
                results = self.judge.judge_candidates(source, candidates, candidate_ids=ids)
            except Exception as exc:  # noqa: BLE001 — 见 fail_soft 的说明
                if not self.fail_soft:
                    raise
                self.last_errors.append(f"{source[:32]}…: {type(exc).__name__}: {exc}")
                continue
            for i, result in zip(idxs, results):
                scores[i] = float(result.weighted_reward)
                self.last_results.append(result)
        return scores

    # ---------------------------------------------------------------- 日志
    def summarize_last(self) -> Dict[str, float]:
        """最近一批的统计量：`mean_fact_reward`、各要素均值、0-4 各档比例。

        需求第十二节要的那一组，键名由 `sfzy.judge.schema.summarize_scores` 统一。
        trainer 只要 `hasattr(scorer, "summarize_last")` 就合并进日志，
        **不需要认识六要素**，耦合面只有一个方法名。
        """
        return summarize_scores(self.last_results)

    def iteration_stats(self) -> Dict[str, int]:
        return {
            "judge_pairs_graded": len(self.last_results) * 6,
            "judge_errors": len(self.last_errors),
        }


def _normalize(item: Dict[str, Any]) -> Dict[str, str]:
    """统一字段名：候选文本可能叫 `candidate` / `output` / `text`。"""
    return {
        "candidate": item.get("candidate") or item.get("output") or item.get("text") or "",
        "reference": item.get("reference", "") or "",
        "source": item.get("source", "") or "",
        "id": item.get("id") or "",
    }
