"""把"抽取 → 逐要素 ROUGE"串起来，并对多个候选做宏平均。

流程（一条样本）：

    人工摘要 ──抽取──► 三个要素（查缓存）        ┐
    候选摘要 ──抽取──► 三个要素（查缓存/现算）   ├─► 逐要素 ROUGE(候选, 人工)
                                                ┘

关键点：

* **人工摘要的抽取结果被复用**：同一份人工摘要会跟 G 个候选比，抽一次就够；
* 抽取是**批量**发出去的（人工 + 所有候选一起去重后一次并发），API 往返是主要开销；
* 抽取失败（`ok=False`）的样本**不进分数**，只计数——不把格式问题算成质量差异。
"""

from __future__ import annotations

import collections
import statistics as st
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sfzy.elements.config import ElementRougeConfig
from sfzy.elements.extract import ElementExtractor, Extraction
from sfzy.elements.schema import ELEMENTS, ElementSet
from sfzy.elements.score import aggregate, score_elements


@dataclass
class SampleResult:
    """一条样本：人工要素 + 每个候选的逐要素分。"""

    sample_id: str
    reference: ElementSet
    reference_ok: bool
    candidates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    reference_elements: Optional[ElementSet] = None   # 便于排查（就是 reference）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.sample_id,
            "reference_ok": self.reference_ok,
            "reference_elements": self.reference.to_dict(),
            "candidates": self.candidates,
        }


class ElementRougePipeline:
    """抽取 + 打分。"""

    def __init__(self, extractor: ElementExtractor, config: ElementRougeConfig) -> None:
        self.extractor = extractor
        self.config = config
        self.rouge = config.rouge

    # ---------------------------------------------------------------- 单条
    def evaluate(
        self, reference_text: str, candidates: Mapping[str, str]
    ) -> Tuple[Extraction, Dict[str, Dict[str, Any]]]:
        """抽一次人工摘要 + 所有候选，返回 (人工抽取, {候选名: 逐要素结果})。"""

        extractions = self.extract_all([reference_text, *candidates.values()])
        ref_extraction = extractions[reference_text]
        return ref_extraction, self.score_one(ref_extraction, candidates, extractions)

    def score_one(
        self,
        ref_extraction: Extraction,
        candidates: Mapping[str, str],
        extractions: Mapping[str, Extraction],
    ) -> Dict[str, Dict[str, Any]]:
        """给一条样本的候选打分（抽取结果已经拿到）。"""

        out: Dict[str, Dict[str, Any]] = {}
        for name, text in candidates.items():
            extraction = extractions[text]
            if not extraction.ok:
                out[name] = {
                    "ok": False,
                    "error": extraction.error,
                    "overall": None,
                    "elements": {},
                    "elements_text": extraction.elements.to_dict(),
                }
                continue
            if not ref_extraction.ok:
                out[name] = {
                    "ok": False,
                    "error": "人工摘要抽取失败",
                    "overall": None,
                    "elements": {},
                    "elements_text": extraction.elements.to_dict(),
                }
                continue
            scores = score_elements(ref_extraction.elements, extraction.elements, config=self.rouge)
            overall = aggregate(scores, self.rouge.normalized_element_weights())
            out[name] = {
                "ok": True,
                "error": "",
                "overall": overall,
                "elements": {k: v.to_dict() for k, v in scores.items()},
                "elements_text": extraction.elements.to_dict(),
            }
        return out

    def extract_all(self, texts: Sequence[str]) -> Dict[str, Extraction]:
        """一次抽完一批文本，返回 {文本: 抽取结果}。

        **必须批量发**：抽取是 I/O 密集、且这个模型一条要十几秒，逐条样本各发一次
        等于把并发压成 1~2（实测 20 条样本要 10 分钟以上）；攒成一批并发发出去，
        同样的量一分钟就能回来。
        """

        unique = list(dict.fromkeys(text for text in texts if text))
        extractions = self.extractor.extract_batch(unique)
        return dict(zip(unique, extractions))

    # ---------------------------------------------------------------- 多条
    def evaluate_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        candidate_names: Sequence[str],
    ) -> Tuple[List[SampleResult], Dict[str, Any]]:
        """rows: {id, reference, candidates: {名字: 文本}}（候选缺的按 None 跳过）。"""

        prepared: List[Tuple[str, str, Dict[str, str]]] = []
        all_texts: List[str] = []
        for row in rows:
            reference_text = str(row.get("reference", ""))
            candidates = {
                name: str(row["candidates"][name])
                for name in candidate_names
                if row.get("candidates", {}).get(name)
            }
            if not reference_text or not candidates:
                continue
            prepared.append((str(row.get("id", len(prepared))), reference_text, candidates))
            all_texts.append(reference_text)
            all_texts.extend(candidates.values())

        # ---- 一次性把所有人工摘要 + 候选摘要抽完（并发拉满），再逐条打分
        extractions = self.extract_all(all_texts)
        results: List[SampleResult] = []
        for sample_id, reference_text, candidates in prepared:
            ref_extraction = extractions.get(reference_text)
            if ref_extraction is None:                          # pragma: no cover - 逻辑兜底
                continue
            results.append(
                SampleResult(
                    sample_id=sample_id,
                    reference=ref_extraction.elements,
                    reference_ok=ref_extraction.ok,
                    candidates=self.score_one(ref_extraction, candidates, extractions),
                )
            )
        return results, self.summarize(results, candidate_names)

    # ---------------------------------------------------------------- 汇总
    def summarize(
        self, results: Sequence[SampleResult], candidate_names: Sequence[str]
    ) -> Dict[str, Any]:
        """宏平均：每条样本先算分，再对样本取平均（和 `score_corpus` 同口径）。"""

        report: Dict[str, Any] = {
            "n_samples": len(results),
            "rouge_mode": self.rouge.mode,
            "rouge_weights": dict(self.rouge.weights),
            "element_weights": self.rouge.normalized_element_weights(),
            "candidates": {},
        }
        for name in candidate_names:
            per_element: Dict[str, List[float]] = {k: [] for k in ELEMENTS}
            detail: Dict[str, Dict[str, List[float]]] = {
                k: collections.defaultdict(list) for k in ELEMENTS
            }
            overalls: List[float] = []
            status = collections.Counter()
            failed = 0
            for result in results:
                entry = result.candidates.get(name)
                if not entry:
                    continue
                if not entry.get("ok"):
                    failed += 1
                    continue
                if entry.get("overall") is not None:
                    overalls.append(float(entry["overall"]))
                for element, payload in entry["elements"].items():
                    status[f"{element}:{payload['status']}"] += 1
                    if payload["score"] is not None:
                        per_element[element].append(float(payload["score"]))
                        for key in ("rouge1_f", "rouge2_f", "rougel_f"):
                            if payload.get(key) is not None:
                                detail[element][key].append(float(payload[key]))

            def mean(values: Sequence[float]) -> Optional[float]:
                return float(st.mean(values)) if values else None

            report["candidates"][name] = {
                "n_scored": len(overalls),
                "n_failed": failed,
                "overall": mean(overalls),
                "overall_std": float(st.pstdev(overalls)) if len(overalls) > 1 else 0.0,
                "elements": {
                    element: {
                        "n": len(per_element[element]),
                        "score": mean(per_element[element]),
                        "rouge1_f": mean(detail[element]["rouge1_f"]),
                        "rouge2_f": mean(detail[element]["rouge2_f"]),
                        "rougel_f": mean(detail[element]["rougel_f"]),
                    }
                    for element in ELEMENTS
                },
                "status": dict(status),
            }

        # 人工摘要自身的抽取情况（空要素比例 = 抽取覆盖率，便于判断 prompt 好不好用）
        ref_empty = collections.Counter()
        ref_ok = 0
        for result in results:
            if result.reference_ok:
                ref_ok += 1
                for element in result.reference.empty_elements():
                    ref_empty[element] += 1
        report["reference"] = {
            "n_ok": ref_ok,
            "empty_rate": {
                element: (ref_empty[element] / ref_ok if ref_ok else None) for element in ELEMENTS
            },
        }
        report["extract_stats"] = self.extractor.stats.to_dict()
        return report


__all__ = ["SampleResult", "ElementRougePipeline"]
