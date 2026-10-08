"""LLM Judge 本体：一次判定直接吃全文，输出六要素分并聚合成奖励信号。

当前方案不再抽取六要素，也不再逐要素判定。每一步都换成"一次性判定"：

  * 事实一致性：``judge_fact_six_batch([(原文全文, 摘要全文)])``
  * 关键要素覆盖率：``judge_coverage_six_batch([(人工摘要, 候选摘要)])``
  * 信息必要性精确率：``judge_inp_batch([(原文, 人工摘要, 候选摘要)])``

判定的输入输出都通过 ``runtime`` 抽象（本地 TorchRuntime / vLLM / API 都可以），
所以这里只负责"怎么用模型 + 怎么把结果摊回每条候选"。

GRPO 里同一篇文书会配 G 个候选，``judge_candidates`` 是唯一入口：一次评一整组，
一次产出所有启用任务的信号。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sfzy.judge.prompts import (
    build_coverage_six_messages,
    build_fact_six_messages,
    build_inp_messages,
)
from sfzy.judge.runtime import (
    TorchRuntime,
    build_runtime,
    resolve_model_path,
)
from sfzy.judge.schema import (
    DEFAULT_WEIGHTS,
    ELEMENTS,
    MAX_SCORE,
    JudgeResult,
    aggregate,
    aggregate_coverage,
    summarize_scores,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

# 调试开关：设置环境变量 JUDGE_VERBOSE=1 即可打印模型输入/输出
_VERBOSE = os.environ.get("JUDGE_VERBOSE", "0") == "1"

# 这个 Judge 支持的任务（= 它产出的信号名）。
# 加一个新 reward：在 prompts.py 里写判定 prompt、在这里加一个分支、
# 在 `judge_term_options` 里声明要它，不需要动其它代码。
SUPPORTED_TASKS: tuple = (
    "fact_consistency",
    "element_coverage",
    "information_necessity_precision",
)


def parse_six_scores(text: str) -> Dict[str, int]:
    """解析"一次性六要素打分"的 JSON：`{"case_type": 3, ...}` → `{要素: 0-4}`。

    容错到"别让一条样本因为格式失败"：先 `json.loads`（含截 `{}`），再退回
    正则 `键: 数字`。缺失的键不在这里补，由调用方按中性分处理。
    """
    raw = (text or "").strip()
    fence = _FENCE_RE.search(raw)
    if fence:
        raw = fence.group(1).strip()
    candidates = [raw]
    if "{" in raw and "}" in raw:
        candidates.append(raw[raw.find("{"): raw.rfind("}") + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        out: Dict[str, int] = {}
        for name in ELEMENTS:
            value = data.get(name)
            if value is None:
                continue
            if isinstance(value, str):
                match = re.search(r"[0-4]", value)
                if not match:
                    continue
                value = match.group()
            try:
                out[name] = max(0, min(MAX_SCORE, int(value)))
            except (TypeError, ValueError):
                continue
        if out:
            return out

    fixed = raw.replace("“", '"').replace("”", '"')
    out = {}
    for match in re.finditer(r'"?([a-z_]+)"?\s*[:：]\s*([0-4])', fixed):
        if match.group(1) in ELEMENTS:
            out[match.group(1)] = int(match.group(2))
    return out


def _clamp_score(value: Any) -> Optional[int]:
    try:
        return max(0, min(MAX_SCORE, int(value)))
    except (TypeError, ValueError):
        return None


def parse_inp(text: str) -> Dict[str, Any]:
    """解析 INP 判定输出 → `{"propositions": [...], "groups": [...], "inp": float}`。

    约定模型输出：
        {"propositions": [{"text": "原子命题", "necessity": 0~4, "group": 1}, ...]}
    容错：没有 `group` 时按文本相同归组；必要性缺失按 2（中性）处理。

    INP = Σ(每个不重复命题组取其成员必要性得分的最大值) / (4 × 原子命题总数)。
    """
    raw = (text or "").strip()
    fence = _FENCE_RE.search(raw)
    if fence:
        raw = fence.group(1).strip()
    data: Any = None
    candidates = [raw]
    if "{" in raw and "}" in raw:
        candidates.append(raw[raw.find("{"): raw.rfind("}") + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            break
        data = None

    if not isinstance(data, dict):
        return {"propositions": [], "groups": [], "inp": 0.0, "parse": "failed"}

    raw_props = data.get("propositions") or data.get("items") or []
    texts = data.get("texts") or []
    scores = data.get("scores") or []
    groups_raw = data.get("groups") or []
    propositions: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_props):
        if isinstance(item, dict):
            propositions.append({
                "text": str(item.get("text") or item.get("proposition") or ""),
                "necessity": _clamp_score(
                    item.get("necessity", item.get("score"))
                ),
                "group": item.get("group", item.get("group_id")),
            })
        elif isinstance(item, str):
            propositions.append({"text": item, "necessity": None, "group": None})
    if not propositions and texts:
        for i, text_i in enumerate(texts):
            propositions.append({
                "text": str(text_i),
                "necessity": _clamp_score(scores[i] if i < len(scores) else None),
                "group": groups_raw[i] if i < len(groups_raw) else None,
            })
    if not propositions:
        return {"propositions": [], "groups": [], "inp": 0.0, "parse": "empty"}

    # ---- 归组：显式 group 优先；否则按规范化文本相同归组 ----
    group_key: Dict[Any, int] = {}
    txt_key: Dict[str, int] = {}
    groups: List[int] = []
    for prop in propositions:
        gid = prop.get("group")
        if gid is None:
            text_key = re.sub(r"\s+", "", str(prop.get("text") or ""))
            gid = txt_key.setdefault(text_key, len(txt_key))
        if gid not in group_key:
            group_key[gid] = len(group_key)
        prop["group"] = group_key[gid]
        prop["necessity"] = (
            prop["necessity"] if prop["necessity"] is not None else 2
        )
        groups.append(prop["group"])

    total = len(propositions)
    group_scores: Dict[int, int] = {}
    for prop in propositions:
        g = prop["group"]
        group_scores[g] = max(group_scores.get(g, 0), prop["necessity"])
    distinct_positions = [i for i, p in enumerate(propositions)
                          if p["group"] not in {q["group"] for q in propositions[:i]}]
    inp = sum(group_scores.values()) / (float(MAX_SCORE) * total) if total else 0.0
    return {
        "propositions": propositions,
        "groups": groups,
        "distinct_index": distinct_positions,
        "group_scores": group_scores,
        "inp": round(min(1.0, max(0.0, inp)), 6),
        "parse": "ok",
    }


class FactConsistencyJudge:
    """事实一致性 / 覆盖率 / INP 的 Judge 本体。

    三路信号都走"一次性判定"：直接喂全文，不再抽取六要素。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_batch_size: int = 4,
        weights: Optional[Dict[str, float]] = None,
        coverage_weights: Optional[Dict[str, float]] = None,
        tasks: Optional[Sequence[str]] = None,
        max_input_tokens: int = 8192,
        runtime: Optional[TorchRuntime] = None,
        six_max_new_tokens: int = 512,
        debug_prompts: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        """`runtime` 和 `model_path` 二选一。

        传 `runtime` 是为了支持"**用策略模型关掉 adapter 来当裁判**"：
        那种情况下模型已经加载好了，不该再加载一份（6B 就是 12GB）。
        """
        if runtime is None:
            if not model_path:
                raise ValueError("要么给 model_path，要么给一个已建好的 runtime")
            runtime = build_runtime(
                model_path=resolve_model_path(model_path),
                device=device,
                dtype=dtype,
                max_batch_size=max_batch_size,
                max_input_tokens=max_input_tokens,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                trust_remote_code=trust_remote_code,
            )
        self.runtime = runtime
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        # 覆盖率自己的六要素权重。没给就沿用一致性的那份（同一套要素）。
        self.coverage_weights = dict(coverage_weights or self.weights)
        # 这次要跑哪些任务。只跑需要的，能省掉对应的判定调用。
        self.tasks = tuple(tasks) if tasks else ("fact_consistency",)
        unknown = [t for t in self.tasks if t not in SUPPORTED_TASKS]
        if unknown:
            raise ValueError(
                f"未知的 judge task：{unknown}（支持：{list(SUPPORTED_TASKS)}）"
            )
        # 一次判定的生成预算（六要素 JSON / INP 命题列表都在这个量级）。
        self.six_max_new_tokens = int(six_max_new_tokens)
        # 是否把每次构造的 prompt（system+user）打到控制台（调试单条用）
        self.debug_prompts = bool(debug_prompts)
        self.max_input_tokens = max_input_tokens
        # 截断后的文档缓存：一次判定里原文可能被多条候选复用，避免重复截断。
        self._doc_fit_cache: Dict[str, str] = {}

    # ---------------------------------------------------------------- 材料裁剪
    def _fit_document_for_judge(self, document: str) -> str:
        """把送判的原文压到输入预算内（头 + 尾），system 指令始终保留。"""
        cached = self._doc_fit_cache.get(document)
        if cached is not None:
            return cached
        runtime = self.runtime
        if not hasattr(runtime, "truncate_text"):
            self._doc_fit_cache[document] = document
            return document
        try:
            system_tokens = runtime.count_tokens(
                runtime.render(build_fact_six_messages("", "")[:1])
            )
        except Exception:  # noqa: BLE001
            system_tokens = 1024
        # 1500 留给"候选摘要 + JSON 输出 + 模板"；原文按头+尾截断保住首尾事实。
        budget = max(256, self.max_input_tokens - system_tokens - 1500)
        fitted = runtime.truncate_text(document, budget)
        self._doc_fit_cache[document] = fitted
        return fitted

    def clear_cache(self) -> None:
        self._doc_fit_cache.clear()

    # ---------------------------------------------------------------- 内部工具
    def _dump_conversations(self, label: str, conversations: Sequence[List[Dict[str, str]]]) -> None:
        """把构造好的 prompt（system+user）打到控制台，调试单条用。"""
        if not (_VERBOSE or self.debug_prompts):
            return
        for i, conv in enumerate(conversations):
            print(f"\n{'=' * 20} {label} {i + 1} {'=' * 20}")
            for message in conv:
                print(f"[{message['role']}]\n{message['content']}\n")

    def _dump_outputs(self, label: str, raws: Sequence[str]) -> None:
        """把模型的原始返回打到控制台（API 输出观察用）。"""
        if not (_VERBOSE or self.debug_prompts):
            return
        for i, raw in enumerate(raws):
            print(f"\n{'-' * 20} {label} {i + 1} 模型输出 {'-' * 20}")
            print(raw)

    def _generate(
        self, conversations: Sequence[List[Dict[str, str]]], label: str
    ) -> List[str]:
        """一次批量前向。会话式 runtime（API）直接吃 messages，本地先 render。"""
        self._dump_conversations(label, conversations)
        if hasattr(self.runtime, "generate_messages"):
            raws = self.runtime.generate_messages(
                conversations, max_new_tokens=self.six_max_new_tokens
            )
        else:
            raws = self.runtime.generate_batch(
                [self.runtime.render(c) for c in conversations],
                max_new_tokens=self.six_max_new_tokens,
            )
        self._dump_outputs(label, raws)
        return list(raws)

    # ---------------------------------------------------------------- 三路判定
    def judge_fact_six_batch(
        self, items: Sequence[Tuple[str, str]]
    ) -> List[Dict[str, int]]:
        """事实一致性：`[(原文全文, 摘要全文)]` → `[{要素: 0-4}]`（顺序一致）。"""
        if not items:
            return []
        conversations = [
            build_fact_six_messages(self._fit_document_for_judge(doc), cand)
            for doc, cand in items
        ]
        raws = self._generate(conversations, "[Judge:fact_consistency]")
        results: List[Dict[str, int]] = []
        for raw in raws:
            parsed = parse_six_scores(raw)
            results.append({name: parsed.get(name, 2) for name in ELEMENTS})
        return results

    def judge_coverage_six_batch(
        self, items: Sequence[Tuple[str, str]]
    ) -> List[Dict[str, int]]:
        """关键要素覆盖率：`[(人工摘要, 候选摘要)]` → `[{要素: 0-4}]`（顺序一致）。"""
        if not items:
            return []
        conversations = [
            build_coverage_six_messages(ref, cand) for ref, cand in items
        ]
        raws = self._generate(conversations, "[Judge:element_coverage]")
        results: List[Dict[str, int]] = []
        for raw in raws:
            parsed = parse_six_scores(raw)
            results.append({name: parsed.get(name, 2) for name in ELEMENTS})
        return results

    def judge_inp_batch(
        self, items: Sequence[Tuple[str, str, str]]
    ) -> List[Dict[str, Any]]:
        """INP：`[(原文, 人工摘要, 候选摘要)]` → 每条的 `{propositions, groups, inp}`。"""
        if not items:
            return []
        conversations = [
            build_inp_messages(doc, ref, cand) for doc, ref, cand in items
        ]
        raws = self._generate(conversations, "[Judge:information_necessity_precision]")
        return [parse_inp(raw) for raw in raws]

    # ---------------------------------------------------------------- GRPO 入口
    def judge_candidates(
        self,
        document: str,
        candidates: Sequence[str],
        candidate_ids: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
    ) -> List[JudgeResult]:
        """一次评一批候选（就是 GRPO 的一个 group），**一次产出所有启用任务的信号**。

            results = judge.judge_candidates(原文, 8 条采样, reference=人工摘要)
            signals = [r.signals for r in results]

        事实一致性对每条候选判一次；覆盖率 / INP 需要人工摘要，且"候选=人工摘要"
        的自评臂直接短路（覆盖率满分，INP=1.0），不发请求。
        """
        candidates = list(candidates)
        ids = (
            list(candidate_ids)
            if candidate_ids is not None
            else [str(i) for i in range(len(candidates))]
        )

        fact_raw: List[Dict[str, int]] = []
        if "fact_consistency" in self.tasks:
            fact_raw = self.judge_fact_six_batch(
                [(document, text) for text in candidates]
            )

        cov_raw: List[Dict[str, int]] = []
        cov_pos: Dict[int, int] = {}
        if "element_coverage" in self.tasks:
            if reference is None:
                raise ValueError(
                    "element_coverage 需要人工摘要（reference），但调用时没有传。"
                )
            call_items = [(ci, t) for ci, t in enumerate(candidates) if t != reference]
            cov_raw = self.judge_coverage_six_batch(
                [(reference, text) for _ci, text in call_items]
            )
            cov_pos = {ci: k for k, (ci, _text) in enumerate(call_items)}

        inp_details: List[Dict[str, Any]] = []
        if "information_necessity_precision" in self.tasks:
            if reference is None:
                raise ValueError(
                    "information_necessity_precision 需要人工摘要（reference），"
                    "但调用时没有传。"
                )
            details = self.judge_inp_batch([
                (document, reference, text)
                for text in candidates if text != reference
            ])
            pos = 0
            for text in candidates:
                if text == reference:
                    inp_details.append({
                        "propositions": [], "groups": [], "inp": 1.0, "parse": "self",
                    })
                else:
                    inp_details.append(details[pos])
                    pos += 1

        results: List[JudgeResult] = []
        for ci in range(len(candidates)):
            signals: Dict[str, Optional[float]] = {}
            result = JudgeResult(candidate_id=ids[ci])

            if "fact_consistency" in self.tasks:
                result = aggregate(
                    dict(fact_raw[ci]),
                    weights=self.weights,
                    sources={name: "judge" for name in ELEMENTS},
                    candidate_id=ids[ci],
                )
                signals["fact_consistency"] = result.weighted_reward
                # 事实一致性硬门控要读的六要素原始分最小值。
                # 名字必须与 sfzy/rl/reward_spec.py 的 FACT_GATE_SIGNAL 一致。
                signals["fact_consistency_min_raw"] = float(
                    min(result.raw_scores.values())
                )

            if "element_coverage" in self.tasks:
                if ci in cov_pos:
                    raw = {name: cov_raw[cov_pos[ci]].get(name, 2) for name in ELEMENTS}
                else:
                    # 自评臂（候选=人工摘要）→ 覆盖率必然满分。
                    raw = {name: MAX_SCORE for name in ELEMENTS}
                signals["element_coverage"] = aggregate_coverage(
                    raw, list(ELEMENTS), self.coverage_weights
                )

            if "information_necessity_precision" in self.tasks:
                signals["information_necessity_precision"] = float(
                    inp_details[ci]["inp"]
                )

            result.signals = signals
            results.append(result)
        return results

    # ---------------------------------------------------------------- 日志
    @staticmethod
    def summarize(results: Sequence[JudgeResult]) -> Dict[str, float]:
        return summarize_scores(results)
