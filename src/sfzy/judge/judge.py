"""事实一致性 Judge 本体：六要素提取 → 逐要素判定 → 加权聚合。

============================ 为什么拆成"提取 + 逐要素判定" ============================
点式裁判（一次看全文给一个分）试了三版都栽在同一件事上：模型同时看到参考
和候选，会以更短的那份（候选）为锚点，于是算的是 precision 而不是 recall。

拆成六要素以后，每次判定只看**两小段文字**（比如"原告诉讼请求"的原文 vs
候选），没有"从哪一份文本组织阅读顺序"的余地。而且判据被明确写死成
"候选说的能不能被原文支持"——省略不算错。

============================ 缓存与批量（需求第十节） ============================
GRPO 里同一篇文书会配 8 个候选，文档的六要素**只允许提取一次**：

    document → extract → cache（按文本 sha1 索引）
    candidate × 8 → 各自提取
    6 × 8 = 48 个 pair → 一次性组 batch 判定

48 个 pair 如果不组 batch，按 0.5B 逐条跑是 48 次前向；组 batch 之后是
几次。这是需求里唯一一处"实现方式直接决定能不能跑"的地方。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sfzy.judge.prompts import build_extract_messages, build_judge_messages
from sfzy.judge.runtime import (
    DIGITS,
    TorchRuntime,
    build_runtime,
    resolve_model_path,
)
from sfzy.judge.schema import (
    DEFAULT_WEIGHTS,
    ELEMENTS,
    MAX_SCORE,
    JudgeResult,
    SixElements,
    aggregate,
    coerce_element_value,
    empty_field_rule,
    summarize_scores,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_KEY_RE = re.compile(r'"?([a-z_]+)"?\s*[:：]')

# 调试开关：设置环境变量 JUDGE_VERBOSE=1 即可打印模型输入/输出
_VERBOSE = os.environ.get("JUDGE_VERBOSE", "0") == "1"


class ExtractionFailure(RuntimeError):
    """文档要素提取基本失败。

    为什么把它做成异常而不是打个 warning：文档要素为空时，每条候选都会
    走进"原文没有、候选写了"的分支，被裁判判成 0 分 —— **整个 RL 的奖励
    恒为 0，而且不报错**。实测就是这么发生的：2600 字的文书被
    max_new_tokens 截断，六个要素只捞到一个 case_type，所有候选奖励归零。
    宁可当场停下，也不要让 GRPO 在零奖励上烧 13 小时。
    """


def _regex_extract(raw: str) -> Dict[str, str]:
    """按字段位置切片取值 —— 模型输出被截断时唯一能救回部分内容的方法。

    比"每个字段一条独立正则"更稳：它不要求值必须是字符串，数组、数字、
    嵌套都能接住，因为取的是"这个键到下一个键之间的那段文本"。
    """
    fixed = raw.replace("“", '"').replace("”", '"')
    # 中英文标点都要清：小模型经常用全角逗号分隔字段，只 strip(',') 会留下
    # 一个尾随的"，"，值就多带一个字符（实测踩过）
    junk = ' \t\r\n"\'“”,，。{}[]'
    hits = [(m.start(), m.group(1)) for m in _KEY_RE.finditer(fixed)
            if m.group(1) in ELEMENTS]
    out: Dict[str, str] = {}
    for i, (pos, name) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(fixed)
        chunk = fixed[pos:end]
        _, _, value = chunk.partition(":") if ":" in chunk else chunk.partition("：")
        value = value.strip(junk)
        if value.startswith("["):                      # 数组：按逗号切开
            value = "；".join(
                p.strip(junk)
                for p in re.split(r"[,，]", value)
                if p.strip(junk)
            )
        out[name] = value
    return out


def parse_six_json_debug(text: str) -> Tuple[SixElements, str]:
    """返回 (六要素, 解析方式)。解析方式是 `json` / `regex` / `empty`。

    调用方靠它判断"这次提取是不是降级了" —— 降级就意味着该重试，
    而不是拿着半截结果去算分。
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
        if isinstance(data, dict):
            return SixElements.from_dict(data), "json"

    found = _regex_extract(raw)
    if found:
        return SixElements.from_dict(found), "regex"
    return SixElements(), "empty"


def parse_six_json(text: str) -> SixElements:
    """把模型输出解析成六要素，容错到"绝不让一条样本因为格式失败"。

    三级降级，因为小模型的 JSON 输出经常会烂：

      1. 掐掉 ``` 围栏后直接 `json.loads`
      2. 截取第一个 `{` 到最后一个 `}` 再试（模型爱在前后加解释）
      3. 按字段位置切片兜底。全失败就返回空要素 —— **空要素是安全值**：
         按空字段规则会得到"4 分"或走 Judge，不会伪造出一个错误分数
    """
    return parse_six_json_debug(text)[0]


class FactConsistencyJudge:
    """六要素事实一致性 Judge。第一版提取器和判定器共用同一个 0.5B 模型。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_batch_size: int = 4,
        extract_max_new_tokens: int = 1024,
        weights: Optional[Dict[str, float]] = None,
        max_input_tokens: int = 4096,
        min_document_elements: int = 2,
        runtime: Optional[TorchRuntime] = None,
        judge_variant: str = "spec",
        doc_fallback: bool = True,
    ) -> None:
        """`runtime` 和 `model_path` 二选一。

        传 `runtime` 是为了支持"**用策略模型关掉 adapter 来当裁判**"：
        那种情况下模型已经加载好了，不该再加载一份（6B 就是 12GB）。
        见 `tools/` 或 `docs/rl_pipeline.md` 里的接法。
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
            )
        self.runtime = runtime
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.judge_variant = judge_variant
        self.doc_fallback = doc_fallback
        self.extract_max_new_tokens = extract_max_new_tokens
        # 文档要素少于这个数就认定提取失败（见 ExtractionFailure 的说明）。
        # 真实的判决书至少有案由、诉请、查明、结果四项，2 已经是很松的下限。
        self.min_document_elements = min_document_elements
        # 文档要素缓存：GRPO 里同一篇文书要配 G 个候选，只提取一次
        self._doc_cache: Dict[str, SixElements] = {}
        self.stats: Dict[str, int] = {
            "extract_calls": 0, "extract_degraded": 0, "extract_retry": 0,
        }
        # 最近一批判定的 pmax / 概率分布（与**真正发给模型的 pair**逐位对齐），
        # 供 CLI 打印置信度。不参与打分逻辑。
        self.last_pmax: List[float] = []
        self.last_probs: List[List[float]] = []

    # ---------------------------------------------------------------- 提取
    def _extract_once(
        self, texts: Sequence[str], max_new_tokens: int
    ) -> Tuple[List[SixElements], List[str]]:
        prompts = [
            self.runtime.render(build_extract_messages(t)) for t in texts
        ]
        # === 调试：打印提取阶段的输入 Prompt ===
        if _VERBOSE:
            for i, (t, p) in enumerate(zip(texts, prompts)):
                print(f"\n{'='*20} [Extract] 样本 {i+1} 输入 {'='*20}")
                print(f"原始文本（前300字）: {t[:300]}...")
                print(f"构造的完整 Prompt:\n{p}\n")
        # ====================================

        raws = self.runtime.generate_batch(prompts, max_new_tokens=max_new_tokens)
        parsed = [parse_six_json_debug(r) for r in raws]

        # === 调试：打印提取阶段的模型输出与解析结果 ===
        if _VERBOSE:
            for i, (raw, (el, mode)) in enumerate(zip(raws, parsed)):
                print(f"\n{'='*20} [Extract] 样本 {i+1} 输出 {'='*20}")
                print(f"模型原始输出:\n{raw}\n")
                print(f"解析方式: {mode}")
                print(f"解析出的六要素: {el.to_dict()}\n")
        # ============================================

        return [p[0] for p in parsed], [p[1] for p in parsed]

    def extract_six_elements_batch(self, texts: Sequence[str]) -> List[SixElements]:
        """批量提取六要素，**解析降级时自动重试**（预算翻倍）。

        重试不是锦上添花：实测 2600 字的文书在 512 token 预算下会被截断，
        六个要素只捞到一个，而"文档要素为空"会让所有候选奖励归零。
        重试的判断依据是解析方式 —— 只要没能干净地 `json.loads` 成功，
        就说明输出有问题，值得花一次更大的预算重来。
        """
        if not texts:
            return []
        todo = [i for i, t in enumerate(texts) if (t or "").strip()]
        out: List[SixElements] = [SixElements()] * len(texts)
        if not todo:
            return out

        sub = [texts[i] for i in todo]
        els, modes = self._extract_once(sub, self.extract_max_new_tokens)
        self.stats["extract_calls"] += len(sub)

        degraded = [j for j, m in enumerate(modes) if m != "json"]
        if degraded:
            self.stats["extract_degraded"] += len(degraded)
            els2, _ = self._extract_once(
                [sub[j] for j in degraded], self.extract_max_new_tokens * 2
            )
            self.stats["extract_retry"] += len(degraded)
            for k, j in enumerate(degraded):
                # 谁捞到的要素多就用谁 —— 不比较格式，只比较信息量
                if els2[k].n_filled() > els[j].n_filled():
                    els[j] = els2[k]

        for i, el in zip(todo, els):
            out[i] = el
        return out

    def extract_six_elements(self, text: str) -> SixElements:
        return self.extract_six_elements_batch([text])[0]

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

    def document_elements(
        self, document: str, strict: bool = True
    ) -> SixElements:
        """带缓存的文档要素提取。**GRPO 的热路径就是这里**。

        `strict=True` 时，提取结果过于稀疏会抛 `ExtractionFailure`。
        这不是洁癖：文档要素空 → 每条候选的每个要素都走"原文没有、候选写了"
        的分支 → 全部判 0 → 奖励恒为 0。宁可当场停下。
        """
        key = self._cache_key(document)
        if key not in self._doc_cache:
            el = self.extract_six_elements(document)
            if strict and el.n_filled() < self.min_document_elements:
                raise ExtractionFailure(
                    f"文档六要素只抽到 {el.n_filled()} 项（要求 ≥{self.min_document_elements}）：\n"
                    f"  {el.to_dict()}\n"
                    "  常见原因：extract_max_new_tokens 太小被截断、或文书太长被 max_input_tokens 截断。\n"
                    "  继续跑下去所有候选都会得 0 分，奖励恒为 0。"
                )
            self._doc_cache[key] = el
        return self._doc_cache[key]

    def clear_cache(self) -> None:
        self._doc_cache.clear()

    # ---------------------------------------------------------------- 判定
    def _judge_pairs_raw(
        self, pairs: Sequence[Tuple[str, str, str]]
    ) -> Tuple[List[int], List[float], List[List[float]]]:
        """真正调模型的那一层。**不做**空字段规则，调用方负责。"""
        if not pairs:
            return [], [], []
        prompts = [
            self.runtime.render(build_judge_messages(name, doc, cand, variant=self.judge_variant))
            for name, doc, cand in pairs
        ]

        # === 调试：打印判定阶段的输入 Prompt ===
        if _VERBOSE:
            for i, ((name, doc, cand), p) in enumerate(zip(pairs, prompts)):
                print(f"\n{'='*20} [Judge] Pair {i+1} 输入 {'='*20}")
                print(f"要素名称: {name}")
                print(f"原文要素: {doc}")
                print(f"候选要素: {cand}")
                print(f"构造的完整 Prompt:\n{p}\n")
        # =======================================

        scores, pmaxs, probs = self.runtime.score_digits_batch(prompts)

        # === 调试：打印判定阶段的模型输出与分数 ===
        if _VERBOSE:
            for i, (s, pm, pr) in enumerate(zip(scores, pmaxs, probs)):
                print(f"\n{'='*20} [Judge] Pair {i+1} 输出 {'='*20}")
                print(f"解析得分: {s}")
                print(f"最大概率 (pmax): {pm:.4f}")
                print(f"各数字概率分布: {pr}")
                print()
        # ==========================================

        return scores, pmaxs, probs

    def judge_elements_batch(
        self, pairs: Sequence[Tuple[str, str, str]]
    ) -> Tuple[List[int], List[str]]:
        """批量判定，返回 (分数, 来源)。来源是 'judge' 或 'empty_rule'。

        空字段规则在这里统一处理，调用方不必重复判断 —— 少一处判断就少一处
        "有的路径忘了处理"的机会。
        """
        scores: List[Optional[int]] = [None] * len(pairs)
        sources: List[str] = ["judge"] * len(pairs)
        pending, pending_idx = [], []
        for i, (_, doc_el, cand_el) in enumerate(pairs):
            rule = empty_field_rule(doc_el, cand_el)
            if rule is None:
                pending.append(pairs[i])
                pending_idx.append(i)
            else:
                scores[i] = rule
                sources[i] = "empty_rule"
        got, pmaxs, probs = self._judge_pairs_raw(pending)
        self.last_pmax, self.last_probs = pmaxs, probs
        for i, value in zip(pending_idx, got):
            scores[i] = value
        return [int(s) for s in scores], sources     # type: ignore[arg-type]

    def judge_elements_with_confidence(
        self, pairs: Sequence[Tuple[str, str, str]]
    ) -> Tuple[List[int], List[str], List[Optional[float]]]:
        """和 `judge_elements_batch` 一样，另外返回每条的最大概率。

        `pmax` 是"裁判有多确定"。**它不参与打分**，只用来判断这一批评分
        可不可信：整批 pmax 都很低（比如都在 0.3 附近）意味着模型在瞎猜，
        此时的分数不值得进奖励。生成路径拿不到这个信息，这是受限解码
        顺带带来的诊断能力。
        """
        scores, sources = self.judge_elements_batch(pairs)
        # last_pmax 只覆盖真正发给模型的那部分，按顺序还原到全量位置上
        pm: List[Optional[float]] = [None] * len(pairs)
        k = 0
        for i, (_, doc, cand) in enumerate(pairs):
            if empty_field_rule(doc, cand) is None:
                if k < len(self.last_pmax):
                    pm[i] = self.last_pmax[k]
                k += 1
        return scores, sources, pm

    def judge_element(self, element_name: str, document_element: str, candidate_element: str) -> int:
        scores, _ = self.judge_elements_batch(
            [(element_name, document_element, candidate_element)]
        )
        return scores[0]

    # ---------------------------------------------------------------- 聚合
    def judge_summary(
        self,
        document_elements: SixElements,
        candidate_elements: SixElements,
        document: Optional[str] = None,
    ) -> JudgeResult:
        """六要素逐项判定 + 加权聚合。返回的 `weighted_reward` ∈ [0, 1]。

        传了 `document` 就会启用"要素抽空时拿整篇原文兜底"，理由见
        `judge_candidates` 里的同一段注释。不传就是需求的原始行为。
        """
        pairs = self.build_pairs(document_elements, candidate_elements, document)
        scores, sources = self.judge_elements_batch(pairs)
        raw_scores = {name: s for name, s in zip(ELEMENTS, scores)}
        return aggregate(
            raw_scores,
            weights=self.weights,
            sources={name: src for name, src in zip(ELEMENTS, sources)},
        )

    def build_pairs(
        self,
        document_elements: SixElements,
        candidate_elements: SixElements,
        document: Optional[str] = None,
    ) -> List[Tuple[str, str, str]]:
        """按空字段规则组装 (要素名, 原文要素, 摘要要素) 三元组。

        抽成独立方法是因为它有**两条调用路径**：`judge_summary`（只给分数）
        和演示程序（还要打印 pmax）。两边必须用同一套规则，否则演示看到的
        和生产跑的就不是一个东西。
        """
        pairs: List[Tuple[str, str, str]] = []
        for name in ELEMENTS:
            doc_v = document_elements.get(name).strip()
            cand_v = candidate_elements.get(name).strip()
            if not cand_v:
                # 规则一 / 规则二：候选省略 → 4 分，不会真的发请求
                pairs.append((name, doc_v, ""))
            elif doc_v:
                pairs.append((name, doc_v, cand_v))
            elif self.doc_fallback and document:
                pairs.append((name, document, cand_v))
            else:
                pairs.append((name, "", cand_v))
        return pairs

    # ---------------------------------------------------------------- GRPO 入口
    def judge_candidates(
        self,
        document: str,
        candidates: Sequence[str],
        candidate_ids: Optional[Sequence[str]] = None,
    ) -> List[JudgeResult]:
        """一次评一批候选（就是 GRPO 的一个 group）。

        这是接入 GRPOTrainer 时唯一要调的入口：

            doc_el = judge.document_elements(原文)      # 命中缓存，不重复提取
            results = judge.judge_candidates(原文, 8 条采样)
            rewards = [r.weighted_reward for r in results]

        48 个 pair（6 要素 × 8 候选）会一次性组 batch 送进模型，
        空字段的那些根本不发请求。
        """
        candidates = list(candidates)
        cand_els = self.extract_six_elements_batch(candidates)
        doc_el = self.document_elements(document)

        pairs: List[Tuple[str, str, str]] = []
        slots: List[Tuple[int, str]] = []
        precomputed: Dict[Tuple[int, str], int] = {}
        fallback_slots: set = set()
        for ci, ce in enumerate(cand_els):
            for name in ELEMENTS:
                doc_v = doc_el.get(name).strip()
                cand_v = ce.get(name).strip()
                if not cand_v:
                    # 规则一 / 规则二：候选省略（或两边都空）→ 4 分，不发请求
                    precomputed[(ci, name)] = MAX_SCORE
                elif doc_v:
                    pairs.append((name, doc_v, cand_v))
                    slots.append((ci, name))
                elif self.doc_fallback and document:
                    # ★ 提取器漏了这个要素，但候选写了。
                    # 按需求规则三仍然调 Judge —— 但**不能把空字符串当原文要素**：
                    # 裁判看到空原文只能判 0，于是奖励变成"提取器漏过的要素
                    # 千万别写"（写了 0 分、不写 4 分），而且不报错。
                    # 拿整篇原文去判，裁判才有东西可核对。
                    pairs.append((name, document, cand_v))
                    slots.append((ci, name))
                    fallback_slots.add((ci, name))
                else:
                    pairs.append((name, "", cand_v))
                    slots.append((ci, name))

        scores, pmaxs, probs = self._judge_pairs_raw(pairs)
        got = {(ci, name): v for (ci, name), v in zip(slots, scores)}
        pget = {(ci, name): v for (ci, name), v in zip(slots, pmaxs)}
        prget = {(ci, name): v for (ci, name), v in zip(slots, probs)}

        ids = list(candidate_ids) if candidate_ids is not None else [str(i) for i in range(len(candidates))]
        results: List[JudgeResult] = []
        for ci in range(len(candidates)):
            raw_scores, sources, raw_outputs, prob_map = {}, {}, {}, {}
            for name in ELEMENTS:
                key = (ci, name)
                if key in precomputed:
                    raw_scores[name] = precomputed[key]
                    sources[name] = "empty_rule"
                else:
                    raw_scores[name] = got[key]
                    sources[name] = "doc_fallback" if key in fallback_slots else "judge"
                    raw_outputs[name] = str(got[key])
                    prob_map[name] = prget[key]
            results.append(aggregate(
                raw_scores,
                weights=self.weights,
                sources=sources,
                raw_outputs=raw_outputs,
                probs=prob_map,
                candidate_id=ids[ci],
            ))
        return results

    # ---------------------------------------------------------------- 日志
    @staticmethod
    def summarize(results: Sequence[JudgeResult]) -> Dict[str, float]:
        return summarize_scores(results)