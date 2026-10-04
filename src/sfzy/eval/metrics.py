"""裁判接口：把"摘要好不好"交给一个 LLM 裁判来打分。

当前只有一个后端：**六要素事实一致性 Judge**（`semantic.backend=fact`），
由 `sfzy/judge/` 实现，返回 `weighted_reward ∈ [0, 1]`。

点式打分（0-100）、组内排序、API、缓存四个后端已删除。它们是被淘汰的
方案：裁判同时看到原文和候选时会以更短的那份（候选）为锚点，算出的其实是
precision 而不是 recall。六要素拆解 + 逐要素判定是替代方案。

所有裁判共用两个接口：

    score_batch(items)         -> List[Optional[float]]                 单信号（兼容用）
    score_batch_signals(items) -> List[Dict[str, Optional[float]]]      多信号（推荐）

多信号是为了支持"同一个模型 B 一次产出多路 reward"：抽取（贵）只做一次，
同一批 pair 上跑多个判定 task，每个 task 贡献一个**命名信号**，例如

    [{"fact_consistency": 0.8, "element_coverage": 0.4}, ...]

`available_signals()` 声明这个后端能产出哪些信号。`RewardSpec` 需要的信号
必须在里面，否则启动即报错 —— 否则一个配置写着要用的 reward 会静默地拿不到分。

`None` 表示这条打分失败（提取失败 / 显存抖动），**不能当 0 用** ——
失败和"很差"是两件事，混起来会让统计系统性偏低。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set


class SemanticScorer:
    """所有裁判后端的统一接口。"""

    name = "semantic"

    def score_batch(self, items: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
        raise NotImplementedError

    def available_signals(self) -> Set[str]:
        """这个后端能产出的信号名。默认就是它自己的 `name`。"""
        return {self.name}

    def score_batch_signals(
        self, items: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Optional[float]]]:
        """多信号接口。默认把单信号结果包一层，保证旧后端不用改就能用。"""
        return [{self.name: s} for s in self.score_batch(items)]


def build_scorer(
    cfg: Optional[Dict[str, Any]],
    override: Optional[str] = None,
    term_options: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[SemanticScorer]:
    """按配置构造裁判后端。

    现在只支持 `fact`（六要素事实一致性 Judge）与 `none`（不接裁判）。

    `term_options` 是 `RewardSpec.judge_term_options()` 的产物，
    `{信号名: {内部权重字段: 值}}` —— reward 的内部权重只能来自配置文件，
    这里只负责把它转交给对应的裁判后端。
    """
    spec = dict(cfg or {})
    backend = override or spec.get("backend", "none")
    if backend in ("none", "", None):
        return None

    # 六要素事实一致性 Judge 有自己的数据结构（SixElements / JudgeResult），
    # 不走任何 rubric。
    if backend in ("fact", "six", "fact_consistency"):
        return _build_fact_consistency_scorer(spec, dict(term_options or {}))

    raise ValueError(
        f"未知的 semantic.backend：{backend}（现在只支持 fact / none）"
    )


def _build_fact_consistency_scorer(
    spec: Dict[str, Any], term_options: Dict[str, Dict[str, Any]]
) -> SemanticScorer:
    """构造六要素事实一致性裁判（`semantic.backend=fact`）。

    延迟 import：`sfzy.judge` 会带上 torch 的数据结构，不需要裁判的路径
    不该为它付导入代价。
    """
    # 六要素权重曾经能写在 semantic.weights 下，现在只能写在 reward term 里。
    # 留着旧写法会变成"两个地方都能设权重"，正是要避免的事。
    if spec.get("weights"):
        raise ValueError(
            "semantic.weights 已废弃：六要素权重现在只能写在配置文件里，"
            "位置是 rl.reward.terms.fact_consistency.element_weights。"
        )

    from sfzy.judge.judge import FactConsistencyJudge
    from sfzy.judge.judge import SUPPORTED_TASKS
    from sfzy.judge.scorer import FactConsistencyScorer

    # 要跑哪些任务由 reward 配置决定：`judge_term_options()` 的键就是信号名。
    tasks = sorted(term_options) or ["fact_consistency"]
    unsupported = [t for t in tasks if t not in SUPPORTED_TASKS]
    if unsupported:
        raise ValueError(
            f"裁判后端还不支持这些信号：{unsupported}（支持：{list(SUPPORTED_TASKS)}）"
        )

    model = spec.get("model")
    if not model:
        raise ValueError("semantic.backend=fact 需要 semantic.model 指向裁判模型目录")
    judge = FactConsistencyJudge(
        model_path=model,
        device=spec.get("device", "cuda:1"),
        dtype=spec.get("dtype", "bfloat16"),
        max_batch_size=int(spec.get("max_batch_size", 8)),
        extract_max_new_tokens=int(spec.get("extract_max_new_tokens", 1024)),
        # 权重来自 reward 配置；直接调 build_scorer（探针 / 离线打分工具）
        # 不带 term_options 时，FactConsistencyJudge 会用 schema.DEFAULT_WEIGHTS。
        weights=(term_options.get("fact_consistency") or {}).get("element_weights"),
        coverage_weights=(term_options.get("element_coverage") or {}).get("element_weights"),
        tasks=tasks,
        min_document_elements=int(spec.get("min_document_elements", 2)),
        doc_fallback=bool(spec.get("doc_fallback", True)),
        # 裁判侧 4-bit（NF4）：7B 在 24GB 卡上量化后约 5~6GB，且位置由
        # device_map 定在 semantic.device 指的卡上，与策略分居两卡。
        load_in_4bit=bool(spec.get("load_in_4bit", False)),
        bnb_4bit_compute_dtype=spec.get("bnb_4bit_compute_dtype"),
        trust_remote_code=bool(spec.get("trust_remote_code", True)),
    )
    return FactConsistencyScorer(judge)
