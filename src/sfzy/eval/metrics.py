"""裁判接口：把"摘要好不好"交给一个 LLM 裁判来打分。

当前只有一个后端：**六要素事实一致性 Judge**（`semantic.backend=fact`），
由 `sfzy/judge/` 实现，返回 `weighted_reward ∈ [0, 1]`。

点式打分（0-100）、组内排序、API、缓存四个后端已删除。它们是被淘汰的
方案：裁判同时看到原文和候选时会以更短的那份（候选）为锚点，算出的其实是
precision 而不是 recall。六要素拆解 + 逐要素判定是替代方案。

所有裁判共用同一个接口：

    score_batch(items) -> List[Optional[float]]

`None` 表示这条打分失败（提取失败 / 显存抖动），**不能当 0 用** ——
失败和"很差"是两件事，混起来会让统计系统性偏低。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


class SemanticScorer:
    """所有裁判后端的统一接口。"""

    name = "semantic"

    def score_batch(self, items: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
        raise NotImplementedError


def build_scorer(
    cfg: Optional[Dict[str, Any]], override: Optional[str] = None
) -> Optional[SemanticScorer]:
    """按配置构造裁判后端。`override` 可以强制指定后端（"fact" / "none"）。

    现在只支持 `fact`（六要素事实一致性 Judge）与 `none`（不接裁判）。
    """
    spec = dict(cfg or {})
    backend = override or spec.get("backend", "none")
    if backend in ("none", "", None):
        return None

    # 六要素事实一致性 Judge 有自己的数据结构（SixElements / JudgeResult），
    # 不走任何 rubric。
    if backend in ("fact", "six", "fact_consistency"):
        return _build_fact_consistency_scorer(spec)

    raise ValueError(
        f"未知的 semantic.backend：{backend}（现在只支持 fact / none）"
    )


def _build_fact_consistency_scorer(spec: Dict[str, Any]) -> SemanticScorer:
    """构造六要素事实一致性裁判（`semantic.backend=fact`）。

    延迟 import：`sfzy.judge` 会带上 torch 的数据结构，不需要裁判的路径
    不该为它付导入代价。
    """
    from sfzy.judge.judge import FactConsistencyJudge
    from sfzy.judge.scorer import FactConsistencyScorer

    model = spec.get("model")
    if not model:
        raise ValueError("semantic.backend=fact 需要 semantic.model 指向裁判模型目录")
    judge = FactConsistencyJudge(
        model_path=model,
        device=spec.get("device", "cuda:1"),
        dtype=spec.get("dtype", "bfloat16"),
        max_batch_size=int(spec.get("max_batch_size", 8)),
        extract_max_new_tokens=int(spec.get("extract_max_new_tokens", 1024)),
        weights=spec.get("weights"),
        min_document_elements=int(spec.get("min_document_elements", 2)),
        judge_variant=spec.get("judge_variant", "spec"),
        doc_fallback=bool(spec.get("doc_fallback", True)),
        # 裁判侧 4-bit（NF4）：7B 在 24GB 卡上量化后约 5~6GB，且位置由
        # device_map 定在 semantic.device 指的卡上，与策略分居两卡。
        load_in_4bit=bool(spec.get("load_in_4bit", False)),
        bnb_4bit_compute_dtype=spec.get("bnb_4bit_compute_dtype"),
        trust_remote_code=bool(spec.get("trust_remote_code", True)),
    )
    return FactConsistencyScorer(judge)
