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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sfzy.judge.prompts import (
    build_coverage_messages,
    build_coverage_six_messages,
    build_extract_messages,
    build_fact_six_messages,
    build_judge_messages,
    build_judge_messages_with_context,
)
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
    aggregate_coverage,
    coerce_element_value,
    empty_field_rule,
    summarize_scores,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_KEY_RE = re.compile(r'"?([a-z_]+)"?\s*[:：]')

# 调试开关：设置环境变量 JUDGE_VERBOSE=1 即可打印模型输入/输出
_VERBOSE = os.environ.get("JUDGE_VERBOSE", "0") == "1"

# 这个 Judge 支持的任务（= 它产出的信号名）。
# 加一个新 reward：在 prompts.py 里写判定 prompt、在这里加一个分支、
# 在 `term_options` 里声明要它，不需要动抽取和批处理的代码。
SUPPORTED_TASKS: tuple = ("fact_consistency", "element_coverage")


@dataclass(frozen=True)
class JudgeRequest:
    """一次要送进模型的判定：哪个任务、哪个要素、对照双方是什么。"""

    task: str
    element: str
    left: str    # 被对照的一方：原文要素（一致性）/ 参考摘要要素（覆盖率）
    right: str   # 候选摘要要素
    # 事实一致性开启"六要素上下文"时，附上两边的完整六要素（其余五项作辅助）。
    doc_elements: Optional[SixElements] = None
    cand_elements: Optional[SixElements] = None
    # 可选：人工摘要的六要素，作第二辅助参照（自评时不要传）
    ref_elements: Optional[SixElements] = None


class JudgePair(tuple):
    """`(name, left, right)` 三元组，额外携带**任务**和六要素上下文。

    做成 tuple 子类是为了不破坏按三元组解包的旧调用；`task` 决定渲染哪套
    判定 prompt（事实一致性 / 覆盖率），上下文属性只在事实一致性里用。

    `task` 默认 `fact_consistency`，旧代码（只传三元组）行为不变。
    """

    def __new__(cls, name, left, right, task="fact_consistency",
                doc_elements=None, cand_elements=None, ref_elements=None):
        self = super().__new__(cls, (name, left, right))
        self.task = task
        self.doc_elements = doc_elements
        self.cand_elements = cand_elements
        self.ref_elements = ref_elements
        return self


# 旧名字保留（build_pairs 的返回值类型，外部/测试引用过）
FactPair = JudgePair


def _pair_task(item) -> str:
    return getattr(item, "task", "fact_consistency")


def _pair_is_precomputed(item) -> bool:
    """这个 pair 是否按空字段规则直接给分、不送模型。

    **只对事实一致性**：覆盖率 pair 的两个要素本来就都非空（参考没有该项的
    情况在 build_coverage_pairs 里已剔除，候选没写的情况已预置 0）。
    """
    if _pair_task(item) != "fact_consistency":
        return False
    return empty_field_rule(item[1], item[2]) is not None


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


def parse_six_scores(text: str) -> Dict[str, int]:
    """解析"一次性六要素打分"的 JSON：`{"case_type": 3, ...}` → `{要素: 0-4}`。

    容错到"别让一条样本因为格式失败"：先 `json.loads`（含截 `{}`），再退回
    正则 `键: 数字`。缺失的键不在这里补，由调用方按空字段规则/中性分处理。
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
        coverage_weights: Optional[Dict[str, float]] = None,
        tasks: Optional[Sequence[str]] = None,
        max_input_tokens: int = 8192,
        min_document_elements: int = 2,
        runtime: Optional[TorchRuntime] = None,
        extract_runtime: Optional[TorchRuntime] = None,
        doc_fallback: bool = True,
        use_element_context: bool = False,
        use_reference_context: bool = False,
        six_shot_fact: bool = False,
        six_max_new_tokens: int = 512,
        six_shot_coverage: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: Optional[str] = None,
        trust_remote_code: bool = True,
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
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                trust_remote_code=trust_remote_code,
            )
        self.runtime = runtime
        # 抽取和判定可以用不同的模型。抽取这一步最容易出错（实测会把日期
        # 挪用/改写），想换更强、更大的抽取器时单独给一个 runtime 即可；
        # 不给就沿用裁判模型（老行为，零额外显存）。
        self.extract_runtime = extract_runtime or runtime
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        # 覆盖率自己的六要素权重。没给就沿用一致性的那份（同一套要素）。
        self.coverage_weights = dict(coverage_weights or self.weights)
        # 这次要跑哪些任务。只跑需要的，能省掉对应材料的抽取与判定。
        self.tasks = tuple(tasks) if tasks else ("fact_consistency",)
        unknown = [t for t in self.tasks if t not in SUPPORTED_TASKS]
        if unknown:
            raise ValueError(
                f"未知的 judge task：{unknown}（支持：{list(SUPPORTED_TASKS)}）"
            )
        self.doc_fallback = doc_fallback
        # 判定时是否附上两边的完整六要素（其余五项作辅助），弥补抽取边界误差
        self.use_element_context = bool(use_element_context)
        # 判定时是否再附上"人工摘要"作第二辅助参照（自评时调用方不要传 ref_elements）
        self.use_reference_context = bool(use_reference_context)
        # 事实一致性改成"一次给(原文, 摘要六要素) → 输出六要素分"
        self.six_shot_fact = bool(six_shot_fact)
        self.six_max_new_tokens = int(six_max_new_tokens)
        # 覆盖率也改成"一次给(人工摘要, 候选摘要) → 输出六要素分"
        self.six_shot_coverage = bool(six_shot_coverage)
        self.extract_max_new_tokens = extract_max_new_tokens
        self.max_input_tokens = max_input_tokens
        # 截断后的文档缓存（doc_fallback 每条要素都要用同一段截断文本，
        # 不缓存的话一篇文书要 tokenize 六次）
        self._doc_fit_cache: Dict[str, str] = {}
        # 文档要素少于这个数就认定提取失败（见 ExtractionFailure 的说明）。
        # 真实的判决书至少有案由、诉请、查明、结果四项，2 已经是很松的下限。
        self.min_document_elements = min_document_elements
        # 文档要素缓存：GRPO 里同一篇文书要配 G 个候选，只提取一次
        self._doc_cache: Dict[str, SixElements] = {}
        # 参考摘要（人工摘要）要素缓存：同一篇 prompt 的参考是固定的，
        # 也只需要抽一次。覆盖率任务才会用到。
        self._ref_cache: Dict[str, SixElements] = {}
        self.stats: Dict[str, int] = {
            "extract_calls": 0, "extract_degraded": 0, "extract_retry": 0,
        }
        # 最近一批判定的 pmax / 概率分布（与**真正发给模型的 pair**逐位对齐），
        # 供 CLI 打印置信度。不参与打分逻辑。
        self.last_pmax: List[float] = []
        self.last_probs: List[List[float]] = []

    # ---------------------------------------------------------------- 提取
    def _fit_for_extract(self, text: str, kind: str = "summary") -> str:
        """把抽取输入压到预算内（头 + 尾），system 指令始终保留。"""
        runtime = self.extract_runtime
        if not hasattr(runtime, "truncate_text"):
            return text
        try:
            system_tokens = runtime.count_tokens(
                runtime.render([build_extract_messages("", kind)[0]])
            )
        except Exception:  # noqa: BLE001
            system_tokens = 1024
        # 128 是模板开销的余量：system/user/generation 三段的 wrapper 都要算进去，
        # 只减 system 的 token 数会低估，边界上会被 runtime 的最后一道截断切尾。
        budget = max(256, self.max_input_tokens - system_tokens - 128)
        return runtime.truncate_text(text, budget)

    def _fit_document_for_judge(self, document: str) -> str:
        """doc_fallback 时把整篇原文压到判定预算内（同样的头+尾策略）。"""
        cached = self._doc_fit_cache.get(document)
        if cached is not None:
            return cached
        runtime = self.runtime
        if not hasattr(runtime, "truncate_text"):
            self._doc_fit_cache[document] = document
            return document
        try:
            system_tokens = runtime.count_tokens(
                runtime.render(build_judge_messages("case_type", "", "")[:1])
            )
        except Exception:  # noqa: BLE001
            system_tokens = 1024
        # 1500 留给"候选要素 + 另一边的其余五项 + 模板"：开了六要素上下文后，
        # prompt 里除了这段兜底原文，还要装下另外 5+6 个要素。
        budget = max(256, self.max_input_tokens - system_tokens - 1500)
        fitted = runtime.truncate_text(document, budget)
        self._doc_fit_cache[document] = fitted
        return fitted

    def _extract_once(
        self, texts: Sequence[str], max_new_tokens: int, kind: str = "summary"
    ) -> Tuple[List[SixElements], List[str]]:
        texts = [self._fit_for_extract(t, kind) for t in texts]
        conversations = [build_extract_messages(t, kind) for t in texts]
        # === 调试：打印提取阶段的输入 Prompt ===
        if _VERBOSE:
            for i, (t, conv) in enumerate(zip(texts, conversations)):
                print(f"\n{'='*20} [Extract] 样本 {i+1} 输入 {'='*20}")
                print(f"原始文本（前300字）: {t[:300]}...")
                print(f"抽取类型: {kind}")
                print(f"构造的完整 Prompt:\n{conv}\n")
        # ====================================

        # 会话式 runtime（如 API）直接吃 messages；本地 runtime 先 render 再前向。
        if hasattr(self.extract_runtime, "generate_messages"):
            raws = self.extract_runtime.generate_messages(
                conversations, max_new_tokens=max_new_tokens
            )
        else:
            prompts = [self.extract_runtime.render(c) for c in conversations]
            raws = self.extract_runtime.generate_batch(
                prompts, max_new_tokens=max_new_tokens
            )
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

    def extract_six_elements_batch(
        self, texts: Sequence[str], kind: str = "summary"
    ) -> List[SixElements]:
        """批量提取六要素，**解析降级时自动重试**（预算翻倍）。

        重试不是锦上添花：实测 2600 字的文书在 512 token 预算下会被截断，
        六个要素只捞到一个，而"文档要素为空"会让所有候选奖励归零。
        重试的判断依据是解析方式 —— 只要没能干净地 `json.loads` 成功，
        就说明输出有问题，值得花一次更大的预算重来。

        `kind`：`document`=判决书原文，`summary`=摘要（人工/候选）。
        """
        if not texts:
            return []
        todo = [i for i, t in enumerate(texts) if (t or "").strip()]
        out: List[SixElements] = [SixElements()] * len(texts)
        if not todo:
            return out

        sub = [texts[i] for i in todo]
        els, modes = self._extract_once(sub, self.extract_max_new_tokens, kind)
        self.stats["extract_calls"] += len(sub)

        degraded = [j for j, m in enumerate(modes) if m != "json"]
        if degraded:
            self.stats["extract_degraded"] += len(degraded)
            els2, _ = self._extract_once(
                [sub[j] for j in degraded], self.extract_max_new_tokens * 2, kind
            )
            self.stats["extract_retry"] += len(degraded)
            for k, j in enumerate(degraded):
                # 谁捞到的要素多就用谁 —— 不比较格式，只比较信息量
                if els2[k].n_filled() > els[j].n_filled():
                    els[j] = els2[k]

        for i, el in zip(todo, els):
            out[i] = el
        return out

    def extract_six_elements(self, text: str, kind: str = "summary") -> SixElements:
        return self.extract_six_elements_batch([text], kind)[0]

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
            el = self.extract_six_elements(document, kind="document")
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
        self._ref_cache.clear()
        self._doc_fit_cache.clear()

    def reference_elements(self, reference: str) -> SixElements:
        """参考摘要（人工摘要）的六要素，按 sha1 缓存。

        覆盖率要拿"候选 vs 人工摘要"逐项比，所以人工摘要的六要素也得抽一次。
        但它在同一篇 prompt 上是**固定**的，GRPO 每步换的只是候选 ——
        所以和原文要素一样按内容缓存，整个训练里只抽一次。
        """
        key = self._cache_key(reference)
        if key not in self._ref_cache:
            self._ref_cache[key] = self.extract_six_elements(reference)
        return self._ref_cache[key]

    # ---------------------------------------------------------------- 判定
    def _request_messages(self, request: "JudgeRequest") -> List[Dict[str, str]]:
        """按任务构造 messages（不渲染）—— 本地/API 两种 runtime 共用。"""
        if request.task == "element_coverage":
            return build_coverage_messages(
                request.element, request.left, request.right
            )
        if (
            self.use_element_context
            and request.doc_elements is not None
            and request.cand_elements is not None
        ):
            # 逐要素给分不变，只是把两边的六要素都摆出来、高亮本次对比项。
            return build_judge_messages_with_context(
                request.element, request.doc_elements, request.cand_elements,
                document_target=request.left, candidate_target=request.right,
                reference_elements=(
                    request.ref_elements if self.use_reference_context else None
                ),
            )
        return build_judge_messages(request.element, request.left, request.right)

    def _render_request(self, request: "JudgeRequest") -> str:
        return self.runtime.render(self._request_messages(request))

    def _score_requests(
        self, requests: Sequence["JudgeRequest"]
    ) -> Tuple[List[int], List[float], List[List[float]]]:
        """把所有任务的判定 prompt **合成一次批量前向**。

        事实一致性和关键要素覆盖率用的是两套 prompt，但它们打的是同一个
        受限解码头，可以放进同一个 batch —— 这样多一个 reward 不会多跑一遍
        模型，只是每批的序列数变多。
        """
        if not requests:
            return [], [], []
        conversations = [self._request_messages(r) for r in requests]
        prompts = [self.runtime.render(c) for c in conversations] if _VERBOSE else []
        if _VERBOSE:
            for i, (r, prompt) in enumerate(zip(requests, prompts)):
                print(f"\n{'='*20} [Judge:{r.task}] 请求 {i+1} {'='*20}")
                print(f"要素名称: {r.element}")
                print(f"对照左：{r.left}")
                print(f"对照右：{r.right}")
                print(f"构造的完整 Prompt:\n{prompt}\n")
        # API runtime 直接吃 messages（保住 system 里的评分 rubric）；
        # 本地 runtime 先 render 成字符串再前向。
        if hasattr(self.runtime, "score_digits_messages"):
            scores, pmaxs, probs = self.runtime.score_digits_messages(conversations)
        else:
            prompts = [self.runtime.render(c) for c in conversations]
            scores, pmaxs, probs = self.runtime.score_digits_batch(prompts)
        if _VERBOSE:
            for i, (s, pm, pr) in enumerate(zip(scores, pmaxs, probs)):
                print(f"\n{'='*20} [Judge] 请求 {i+1} 输出 {'='*20}")
                print(f"解析得分: {s}")
                print(f"最大概率 (pmax): {pm:.4f}")
                print(f"各数字概率分布: {pr}\n")
        return scores, pmaxs, probs

    def judge_fact_six_batch(
        self, items: Sequence[Tuple[str, SixElements]]
    ) -> List[Dict[str, int]]:
        """一次性六要素判定：`[(原文, 摘要六要素)]` → `[{要素: 0-4}]`（顺序一致）。

        输入是**整篇原文 + 摘要的六要素**，一次调用给出六个分。比原来"逐要素判 6 次"
        少了 5/6 的调用，而且不再依赖原文抽取（原文抽取有损是之前假 0 的主因）。
        """
        if not items:
            return []
        conversations = [
            build_fact_six_messages(self._fit_document_for_judge(doc), els)
            for doc, els in items
        ]
        if hasattr(self.runtime, "generate_messages"):
            raws = self.runtime.generate_messages(
                conversations, max_new_tokens=self.six_max_new_tokens
            )
        else:
            raws = self.runtime.generate_batch(
                [self.runtime.render(c) for c in conversations],
                max_new_tokens=self.six_max_new_tokens,
            )
        results: List[Dict[str, int]] = []
        for (_doc, els), raw in zip(items, raws):
            parsed = parse_six_scores(raw)
            full: Dict[str, int] = {}
            for name in ELEMENTS:
                if not els.get(name).strip():
                    full[name] = MAX_SCORE            # 空字段规则：候选省略 → 4
                else:
                    full[name] = parsed.get(name, 2)  # 解析缺失按"疑点"2，不判 0
            results.append(full)
        return results

    def judge_coverage_six_batch(
        self, items: Sequence[Tuple[str, str]]
    ) -> List[Dict[str, int]]:
        """一次性覆盖率判定：`[(人工摘要, 候选摘要)]` → `[{要素: 0-4}]`（顺序一致）。"""
        if not items:
            return []
        conversations = [
            build_coverage_six_messages(ref, cand) for ref, cand in items
        ]
        if hasattr(self.runtime, "generate_messages"):
            raws = self.runtime.generate_messages(
                conversations, max_new_tokens=self.six_max_new_tokens
            )
        else:
            raws = self.runtime.generate_batch(
                [self.runtime.render(c) for c in conversations],
                max_new_tokens=self.six_max_new_tokens,
            )
        results: List[Dict[str, int]] = []
        for _item, raw in zip(items, raws):
            parsed = parse_six_scores(raw)
            results.append({name: parsed.get(name, 2) for name in ELEMENTS})
        return results

    def _judge_pairs_raw(
        self, pairs: Sequence[Tuple[str, str, str]]
    ) -> Tuple[List[int], List[float], List[List[float]]]:
        """真正调模型的那一层。**不做**空字段规则，调用方负责。"""
        requests = [
            JudgeRequest(
                _pair_task(item), item[0], item[1], item[2],
                doc_elements=getattr(item, "doc_elements", None),
                cand_elements=getattr(item, "cand_elements", None),
                ref_elements=getattr(item, "ref_elements", None),
            )
            for item in pairs
        ]
        return self._score_requests(requests)

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
        for i, item in enumerate(pairs):
            if not _pair_is_precomputed(item):
                pending.append(pairs[i])
                pending_idx.append(i)
            else:
                scores[i] = empty_field_rule(item[1], item[2])
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
        for i, item in enumerate(pairs):
            if not _pair_is_precomputed(item):
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
        reference_elements: Optional[SixElements] = None,
    ) -> List[FactPair]:
        """按空字段规则组装 (要素名, 原文要素, 摘要要素) 三元组。

        抽成独立方法是因为它有**两条调用路径**：`judge_summary`（只给分数）
        和演示程序（还要打印 pmax）。两边必须用同一套规则，否则演示看到的
        和生产跑的就不是一个东西。

        返回的 `FactPair` 就是三元组（旧代码可照常解包），额外带上两边的
        六要素上下文，供"逐要素判定 + 其余五项辅助"用。

        `reference_elements` 传了就把人工摘要作为第二辅助参照带进判定 prompt；
        但**候选本身就是人工摘要（自评）时会自动忽略**，否则就成了自证。
        """
        if (
            reference_elements is not None
            and reference_elements.to_dict() == candidate_elements.to_dict()
        ):
            reference_elements = None
        pairs: List[JudgePair] = []
        for name in ELEMENTS:
            doc_v = document_elements.get(name).strip()
            cand_v = candidate_elements.get(name).strip()
            if not cand_v:
                # 规则一 / 规则二：候选省略 → 4 分，不会真的发请求
                pairs.append(JudgePair(
                    name, doc_v, "",
                    doc_elements=document_elements, cand_elements=candidate_elements,
                    ref_elements=reference_elements,
                ))
            elif doc_v:
                pairs.append(JudgePair(
                    name, doc_v, cand_v,
                    doc_elements=document_elements, cand_elements=candidate_elements,
                    ref_elements=reference_elements,
                ))
            elif self.doc_fallback and document:
                pairs.append(JudgePair(
                    name, self._fit_document_for_judge(document), cand_v,
                    doc_elements=document_elements, cand_elements=candidate_elements,
                    ref_elements=reference_elements,
                ))
            else:
                pairs.append(JudgePair(
                    name, "", cand_v,
                    doc_elements=document_elements, cand_elements=candidate_elements,
                    ref_elements=reference_elements,
                ))
        return pairs

    def build_coverage_pairs(
        self,
        reference_elements: SixElements,
        candidate_elements: SixElements,
    ) -> Tuple[List[Tuple[str, str, str]], List[str], Dict[str, int]]:
        """关键要素覆盖率的判定输入：候选摘要**对参考摘要**的覆盖程度。

        返回 `(要送进模型的 pair, 参与计算的要素, 免调模型直接给的分)`。

        规则（和事实一致性刻意相反）：

          * 参考摘要里**没有**这一项 → 该要素不参与，分子分母都不算
          * 参考有、候选空       → 未覆盖，0 分，不发请求
          * 两边都有             → 送进模型判 0~4

        注意方向：左是**参考摘要要素**，右是候选。候选多写不扣分
        （参考没写的内容由别的 reward 或 ROUGE 负责罚）。
        """
        pairs: List[JudgePair] = []
        present: List[str] = []
        precomputed: Dict[str, int] = {}
        for name in ELEMENTS:
            ref_v = reference_elements.get(name).strip()
            if not ref_v:
                continue                       # 参考里没有 → 不参与
            present.append(name)
            cand_v = candidate_elements.get(name).strip()
            if not cand_v:
                precomputed[name] = 0          # 未覆盖该要素
            else:
                # 覆盖率是**独立任务**：左=参考摘要要素，右=候选摘要要素，
                # 判定时走 build_coverage_messages，绝不能套事实一致性的 prompt。
                pairs.append(JudgePair(name, ref_v, cand_v, task="element_coverage"))
        return pairs, present, precomputed

    # ---------------------------------------------------------------- GRPO 入口
    def judge_candidates(
        self,
        document: str,
        candidates: Sequence[str],
        candidate_ids: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
    ) -> List[JudgeResult]:
        """一次评一批候选（就是 GRPO 的一个 group），**一次产出所有启用任务的信号**。

        这是接入 GRPOTrainer 时唯一要调的入口：

            results = judge.judge_candidates(原文, 8 条采样, reference=人工摘要)
            signals = [r.signals for r in results]   # {"fact_consistency":…, "element_coverage":…}

        算力只花一次：
          * 原文要素按 sha1 缓存（只在启用事实一致性时才抽）
          * 人工摘要要素按 sha1 缓存（只在启用覆盖率时才抽）
          * 候选要素一次批量抽完，**两个任务共用**
          * 两个任务的判定 prompt 合成**一次批量前向**
        空字段的那些 pair 根本不发请求。
        """
        candidates = list(candidates)
        cand_els = self.extract_six_elements_batch(candidates)
        ids = (
            list(candidate_ids)
            if candidate_ids is not None
            else [str(i) for i in range(len(candidates))]
        )

        # ---- 事实一致性：原文要素 vs 候选要素 ----
        cons_requests: List[JudgeRequest] = []
        cons_slots: List[Tuple[int, str]] = []
        cons_precomputed: Dict[Tuple[int, str], int] = {}
        fallback_slots: set = set()
        six_raw: List[Dict[str, int]] = []
        if "fact_consistency" in self.tasks and self.six_shot_fact:
            # 一次性判定：整篇原文 + 摘要六要素 → 六要素分（一次调用/候选）
            six_raw = self.judge_fact_six_batch([(document, ce) for ce in cand_els])
        elif "fact_consistency" in self.tasks:
            doc_el = self.document_elements(document)
            # 人工摘要作第二辅助参照（自评时逐条忽略，见下面的 ce 比较）
            ref_el = (
                self.reference_elements(reference)
                if (self.use_reference_context and reference is not None)
                else None
            )
            for ci, ce in enumerate(cand_els):
                ref_for_this = (
                    ref_el if ref_el is not None
                    and ref_el.to_dict() != ce.to_dict() else None
                )
                for name in ELEMENTS:
                    doc_v = doc_el.get(name).strip()
                    cand_v = ce.get(name).strip()
                    if not cand_v:
                        # 规则一 / 规则二：候选省略（或两边都空）→ 4 分，不发请求
                        cons_precomputed[(ci, name)] = MAX_SCORE
                    elif doc_v:
                        cons_requests.append(
                            JudgeRequest("fact_consistency", name, doc_v, cand_v,
                                         doc_el, ce, ref_for_this)
                        )
                        cons_slots.append((ci, name))
                    elif self.doc_fallback and document:
                        # ★ 提取器漏了这个要素，但候选写了。
                        # 不能把空字符串当原文要素：裁判看到空原文只能判 0，
                        # 奖励就变成"提取器漏过的要素千万别写"，而且不报错。
                        # 拿整篇原文去判，裁判才有东西可核对。
                        cons_requests.append(
                            JudgeRequest(
                                "fact_consistency", name,
                                self._fit_document_for_judge(document), cand_v,
                                doc_el, ce, ref_for_this,
                            )
                        )
                        cons_slots.append((ci, name))
                        fallback_slots.add((ci, name))
                    else:
                        cons_requests.append(
                            JudgeRequest("fact_consistency", name, "", cand_v,
                                         doc_el, ce, ref_for_this)
                        )
                        cons_slots.append((ci, name))

        # ---- 关键要素覆盖率：参考摘要要素 vs 候选要素 ----
        cov_requests: List[JudgeRequest] = []
        cov_slots: List[Tuple[int, str]] = []
        cov_precomputed: Dict[Tuple[int, str], int] = {}
        cov_present: Dict[int, List[str]] = {}
        cov_six_raw: List[Dict[str, int]] = []
        if "element_coverage" in self.tasks:
            if reference is None:
                raise ValueError(
                    "element_coverage 需要人工摘要（reference），但调用时没有传。"
                )
            ref_el = self.reference_elements(reference)
            if self.six_shot_coverage:
                # 一次性判定：(人工摘要全文, 候选摘要全文) → 六要素覆盖分
                cov_six_raw = self.judge_coverage_six_batch(
                    [(reference, text) for text in candidates]
                )
                for ci, ce in enumerate(cand_els):
                    present = [name for name in ELEMENTS if ref_el.get(name).strip()]
                    cov_present[ci] = present
                    for name in present:
                        if not ce.get(name).strip():
                            cov_precomputed[(ci, name)] = 0
            else:
                for ci, ce in enumerate(cand_els):
                    pairs, present, precomputed = self.build_coverage_pairs(ref_el, ce)
                    cov_present[ci] = present
                    for name, score in precomputed.items():
                        cov_precomputed[(ci, name)] = score
                    for name, left, right in pairs:
                        cov_requests.append(
                            JudgeRequest("element_coverage", name, left, right)
                        )
                        cov_slots.append((ci, name))

        # ---- 两个任务合成一次批量前向 ----
        scores, pmaxs, probs = self._score_requests(cons_requests + cov_requests)
        n_cons = len(cons_requests)
        cons_scores, cons_pmaxs, cons_probs = (
            scores[:n_cons], pmaxs[:n_cons], probs[:n_cons],
        )
        cov_scores = scores[n_cons:]
        # last_pmax 只覆盖一致性那一段：它是给 CLI 看置信度用的，语义不要变
        self.last_pmax, self.last_probs = cons_pmaxs, cons_probs

        got = {(ci, name): v for (ci, name), v in zip(cons_slots, cons_scores)}
        pget = {(ci, name): v for (ci, name), v in zip(cons_slots, cons_pmaxs)}
        prget = {(ci, name): v for (ci, name), v in zip(cons_slots, cons_probs)}
        cov_got = {(ci, name): v for (ci, name), v in zip(cov_slots, cov_scores)}

        results: List[JudgeResult] = []
        for ci in range(len(candidates)):
            signals: Dict[str, Optional[float]] = {}
            result = JudgeResult(candidate_id=ids[ci])

            if "fact_consistency" in self.tasks:
                if self.six_shot_fact:
                    raw_scores = dict(six_raw[ci])
                    sources = {name: "six_shot" for name in ELEMENTS}
                    raw_outputs, prob_map = {}, {}
                else:
                    raw_scores, sources, raw_outputs, prob_map = {}, {}, {}, {}
                    for name in ELEMENTS:
                        key = (ci, name)
                        if key in cons_precomputed:
                            raw_scores[name] = cons_precomputed[key]
                            sources[name] = "empty_rule"
                        else:
                            raw_scores[name] = got[key]
                            sources[name] = (
                                "doc_fallback" if key in fallback_slots else "judge"
                            )
                            raw_outputs[name] = str(got[key])
                            prob_map[name] = prget[key]
                result = aggregate(
                    raw_scores,
                    weights=self.weights,
                    sources=sources,
                    raw_outputs=raw_outputs,
                    probs=prob_map,
                    candidate_id=ids[ci],
                )
                signals["fact_consistency"] = result.weighted_reward
                # 事实一致性硬门控要读的六要素原始分最小值。
                # 名字必须与 sfzy/rl/reward_spec.py 的 FACT_GATE_SIGNAL 一致
                # （judge 层刻意不 import 那一层，免得把 torch 依赖倒灌进
                #  纯 CPU 的 reward 路径）；一致性由 tests/test_reward.py 钉住。
                signals["fact_consistency_min_raw"] = float(
                    min(result.raw_scores.values())
                )

            if "element_coverage" in self.tasks:
                present = cov_present[ci]
                if self.six_shot_coverage:
                    raw_scores = {
                        name: cov_precomputed.get(
                            (ci, name), cov_six_raw[ci].get(name, 2)
                        )
                        for name in present
                    }
                else:
                    raw_scores = {
                        name: cov_precomputed.get((ci, name), cov_got.get((ci, name)))
                        for name in present
                    }
                signals["element_coverage"] = aggregate_coverage(
                    raw_scores, present, self.coverage_weights
                )

            result.signals = signals
            results.append(result)
        return results

    # ---------------------------------------------------------------- 日志
    @staticmethod
    def summarize(results: Sequence[JudgeResult]) -> Dict[str, float]:
        return summarize_scores(results)
