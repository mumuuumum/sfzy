"""把六要素事实一致性 Judge 接到 GRPO 奖励管线上的适配层。

============================ 为什么需要这一层 ============================
`FactConsistencyJudge` 的入口是 `judge_candidates(document, candidates, reference=…)`
—— **一次吃一篇文书 + 一组候选**。而 GRPO 侧的契约是扁平的

    items = [{candidate, reference, source, id}, ...]
    score_batch_signals(items) -> [{信号名: 分数 ∈ [0,1]}, ...]

中间的落差就是本文件。要做的只有三件事：

  1. 按 `source`（文书原文）把扁平的 items 分回各组
  2. 校验同组的候选共用同一份 `reference`（人工摘要）
  3. 逐组调用 `judge_candidates`，把每条候选的多路信号摊回原位置

第 1 步不能省。不分组就等于逐条调用，文档六要素会被重复提取 G 次；而候选
六要素也失去批量 —— "6 要素 × G 候选一次组 batch"直接作废。

============================ 契约在哪 ============================
契约由 `sfzy.eval.metrics.SignalScorer`（一个 `typing.Protocol`）描述，
本类**不继承**它 —— 结构上满足即可。产出哪几路信号由构造 Judge 时的 `tasks`
决定：加 reward 是加 task，不是加 scorer。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sfzy.judge.judge import FactConsistencyJudge
from sfzy.judge.schema import JudgeResult, summarize_scores


class SixElementScorer:
    """全项目**唯一**的 scorer：把六要素 Judge 接成多路命名信号。

        judge = FactConsistencyJudge("models/Qwen2.5-1.5B-Instruct", device="cuda:1")
        scorer = SixElementScorer(judge)
        signals = scorer.score_batch_signals(items)
        # [{"fact_consistency": 0.83, "element_coverage": 0.51}, ...]

    输入形态和 GRPO 的 rollout 完全对齐：同一篇文书会连续给出 G 条候选，
    `source` 是文书原文，`reference` 是人工摘要。产出哪几路信号由构造 Judge 时
    的 `tasks` 决定 —— 加 reward 是加 task，不是加 scorer。

    它不需要继承任何基类：契约（`available_signals` + `score_batch_signals`）
    由 `sfzy.eval.metrics.SignalScorer` 这个 Protocol 描述。
    """

    # 只是日志里的显示名，不参与任何分发
    name = "judge_six_element"

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

    def available_signals(self) -> set:
        """这个后端实际会产出的信号 = 构造时启用的任务。

        只开覆盖率时，事实一致性那一路根本不会被算，也就不会出现在这里。
        """
        return set(self.judge.tasks)

    def _score_item_signals(
        self, items: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Optional[float]]]:
        """真正算分的那一层：返回每条候选的**全部信号**，失败的那一组为 `{}`。"""
        norm = [_normalize(it) for it in items]
        signals: List[Dict[str, Optional[float]]] = [{} for _ in norm]
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
            # 同一篇原文的候选必须共用同一份人工摘要 —— 不然覆盖率判的就
            # 不是同一把尺子。这里显式拦住，而不是让第一份悄悄生效。
            references = {norm[i]["reference"] for i in idxs}
            if len(references) > 1:
                raise ValueError(
                    "同一个 source 下的候选必须共用同一个 reference（人工摘要），"
                    f"收到 {len(references)} 份不同的。"
                )
            reference = next(iter(references)) if references else ""
            try:
                results = self.judge.judge_candidates(
                    source, candidates, candidate_ids=ids, reference=reference
                )
            except Exception as exc:  # noqa: BLE001 — 见 fail_soft 的说明
                if not self.fail_soft:
                    raise
                self.last_errors.append(f"{source[:32]}…: {type(exc).__name__}: {exc}")
                continue
            for i, result in zip(idxs, results):
                signals[i] = {
                    name: (None if value is None else float(value))
                    for name, value in result.signals.items()
                }
                self.last_results.append(result)
        return signals

    def score_batch_signals(
        self, items: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Optional[float]]]:
        """每条候选 → `{"fact_consistency":…, "element_coverage":…}`。

        要素抽取（原文 / 人工摘要 / 候选）在这条路径上各做一次，
        两个任务共用；判定 prompt 合成一次批量前向。
        """
        return self._score_item_signals(items)

    # ---------------------------------------------------------------- 日志
    def summarize_last(self) -> Dict[str, float]:
        """最近一批的统计量：`mean_fact_reward`、各要素均值、0-4 各档比例。

        需求第十二节要的那一组，键名由 `sfzy.judge.schema.summarize_scores` 统一。
        trainer 只要 `hasattr(scorer, "summarize_last")` 就合并进日志，
        **不需要认识六要素**，耦合面只有一个方法名。
        """
        # 只开了覆盖率时，last_results 里没有一致性的逐要素分，统计量无从谈起
        if not self.last_results or not self.last_results[0].scores:
            return {}
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
