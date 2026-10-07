"""用 vLLM 版六要素 Judge 给 val 三元组离线打「完整奖励」分（候选 vs 人工）。

============================ 这个脚本要验证的两条结论 ============================
`data/triples/sft_val_shard*of2.jsonl` 每行是

    {"id", "source"(文书原文), "reference"(人工摘要), "output"(SFT 候选摘要)}

用同一套 reward 给两臂打分（候选臂 candidate=output；人工臂 candidate=reference），
然后验证两条结论：

  结论 1  **候选摘要的奖励一般比人工摘要小。**
          配对差（人工 − 候选）> 0、胜率 > 0.5，且配对 t 检验 p < 0.05。
          不成立就说明 reward 没有判别力，接进 GRPO 只是噪声梯度。

  结论 2  **不同候选摘要之间，reward 与 ROUGE 分数正相关。**
          跨文书对「候选的 reward」和「候选 vs 人工摘要的 ROUGE-L F1」算
          Pearson / Spearman，要求 ρ > 0 且 p < 0.05。ROUGE 是官方评测口径，
          reward 若和它反向，优化目标就和最终指标打架。

两条结论连同支撑数据（均值、配对检验、相关系数、分项均值）都会落到
`*.summary.json` 和一份人读的 `*.report.md` 里。

============================ 和训练是同一个口径 ============================
奖励完全走训练那条路径：

    门控 check_gate（长度 / 前缀 / 结果标记）
      → 逐项：rouge_l（规则）+ fact_consistency / element_coverage（裁判）
      → 按 RewardSpec 归一化权重加权求和

变量只有一个：裁判（六要素 Judge）从 HF 换成 vLLM，用来把耗时压下来。
reward 配置直接从 `--config` 的 `rl.reward` 读，和 train_grpo.py 用的是同一段
代码（`RewardSpec.from_config` / `compute_reward`），不在这里另写一套。

============================ 为什么能复用 Judge ============================
`FactConsistencyJudge` 把"怎么用模型"抽象成了一个 `runtime`（见
`sfzy/judge/runtime.py` 的 `TorchRuntime`）。只需要实现一个接口相同的
`VLLMRuntime`（`render` / `generate_batch` / `score_digits_batch`），
六要素抽取、重试、空字段规则、覆盖率组装、0-4 受限解码这些逻辑一行都不用改。

关键实现选择（和 `scripts/generate_triples_vllm.py` 一致）：
  * 直接把 token id 喂给 vLLM（`prompt_token_ids`），不喂字符串 ——
    tokenization 收敛到我们的 tokenizer，vLLM 只负责前向；
  * 受限解码用 `max_tokens=1 + logprobs` 复刻：在 `{'0','1','2','3','4'}`
    五个 token 的 logprob 上做 softmax，取 argmax 当分数、max 当置信度。

============================ 分批，别一条一条喂 ============================
vLLM 的连续批处理是吞吐的来源。脚本按 `--chunk-size` 条记录切成一批：

  1. 把这一批里所有**不重复**的原文 / 人工摘要 / 候选摘要一次性抽取六要素
     （一次 generate，几十条序列一起批）；
  2. 两种臂 × 六要素 × 两个任务的判定 prompt 合成**一次批量前向**；
  3. 再按记录切回去聚合。

一条记录分两臂：

  候选臂（candidate）：candidate=output，  reference=reference
  人工臂（human）    ：candidate=reference，reference=reference

两臂共用同一份原文要素 / 人工摘要要素，所以原文六要素不会重复抽。

============================ T4 的显存（重要） ============================
vLLM **支持加载 4-bit**：AWQ / GPTQ / bitsandbytes 三条路都可以（官方硬件表
里 Turing=T4 是 ✅，见 https://docs.vllm.ai/en/latest/features/quantization/）。
Qwen2.5-7B 的 fp16 权重约 15GB，单张 T4（16GB）装不下，所以必须走 4-bit。

唯一要注意的是：vLLM 的 `LLM(...)` **不读**配置里的 `semantic.load_in_4bit`
（那是 transformers/bitsandbytes 的开关），量化方案得用 `--quantization` 显式给：

  * `--quantization bitsandbytes`：对现有 fp16 权重做 in-flight NF4 量化，
    不用另下一个量化好的 checkpoint。装 `bitsandbytes` 即可；新版 vLLM 把
    bnb 拆成了插件，还要 `pip install vllm-bnb-plugin`（0.6.x 内置，不用装）。
  * `--model <AWQ/GPTQ checkpoint> --quantization awq|gptq`：kernel 更快，
    prefill 吞吐通常好于 bnb，代价是要多下一份权重。

`--model` 不显式给时取 config 的 `semantic.model`；`--dtype float16` 是 T4
的默认。A100/4090 想跑全精度就 `--quantization none --dtype bfloat16`。
配置写了 load_in_4bit 却忘了传 --quantization 时，脚本会打印警告——因为那会
静默按 fp16 加载，在 T4 上直接 OOM。

============================ 用法（2×T4，一个分片一张卡） ============================
先装独立的 vLLM 环境（见 docs/vllm_triples.md，别污染 SFT 环境）：

    # 卡 0 处理 shard0，卡 1 处理 shard1（两个进程各占一张卡）
    # T4 上二选一：--quantization bitsandbytes（现量化现有 fp16 权重），
    # 或 --model <AWQ/GPTQ> --quantization awq|gptq（换量化好的权重）。
    CUDA_VISIBLE_DEVICES=0 python scripts/score_reward_vllm.py \
        --config configs/grpo_fact_coverage_t4.yaml --dtype float16 \
        --quantization bitsandbytes \
        --input data/triples/sft_val_shard0of2.jsonl

    CUDA_VISIBLE_DEVICES=1 python scripts/score_reward_vllm.py \
        --config configs/grpo_fact_coverage_t4.yaml --dtype float16 \
        --quantization bitsandbytes \
        --input data/triples/sft_val_shard1of2.jsonl

输出（默认 `data/judge/`）：

  * `<stem>.reward.jsonl`：每行一条记录的一臂奖励、分项与 ROUGE；
  * `<stem>.reward.summary.json`：两臂均值、分项均值、两条结论的检验结果；
  * `<stem>.reward.report.md`：把上面两条结论和支撑数据写成可读报告。

支持断点续跑（输出里已有的 id 跳过）、`--limit`、多个 `--input`。

两个分片跑完后，把逐条结果合成一份全量总报告（纯 CPU，不加载裁判）：

    python scripts/score_reward_vllm.py --merge \
        data/judge/sft_val_shard0of2.reward.jsonl \
        data/judge/sft_val_shard1of2.reward.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                          # noqa: E402
from sfzy.eval.rouge import score_pair                       # noqa: E402
from sfzy.rl.reward import compute_reward                    # noqa: E402
from sfzy.rl.reward_spec import RewardSpec                   # noqa: E402
from sfzy.utils.logging import get_logger                    # noqa: E402
from sfzy.utils.text_fit import count_tokens as _count_tokens  # noqa: E402
from sfzy.utils.text_fit import truncate_text as _truncate_text  # noqa: E402

# 六要素 Judge 的受限解码头，和 sfzy/judge/runtime.py 用同一个词表
from sfzy.judge.runtime import DIGITS                        # noqa: E402
from sfzy.judge.schema import (                              # noqa: E402
    ELEMENTS,
    MAX_SCORE,
    SixElements,
    aggregate,
    aggregate_coverage,
)

logger = get_logger("score_reward_vllm")

_ROUGE_WARNED = False

# 诊断开关：--dump-elements 打开后，逐条输出里带上原文/人工/候选三种六要素
# 的抽取结果。定位"某个要素为什么判 0"时用它（配合 --limit 跑一小批）。
_DUMP_ELEMENTS = False


def _warn_rouge_once(exc: Exception) -> None:
    """ROUGE 全线失败时说清楚原因，别让结论 2 静默变成“样本不足”。"""
    global _ROUGE_WARNED
    if _ROUGE_WARNED:
        return
    _ROUGE_WARNED = True
    logger.warning(
        "计算 ROUGE 失败（结论 2 会因此缺数据）：%s: %s。"
        "最常见的原因是 rouge_mode=jieba 但环境里没装 jieba —— pip install jieba。",
        type(exc).__name__, exc,
    )


def resolve(path: Any) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def is_local_dir(path: Path) -> bool:
    """权限不足（配置里写着别的机器的路径）时按"不是本地路径"处理。"""
    try:
        return path.is_dir()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# vLLM 运行时：接口和 sfzy/judge/runtime.py 的 TorchRuntime 一致
# ---------------------------------------------------------------------------
class VLLMRuntime:
    """把 vLLM 的 LLM 包成 Judge 认识的 runtime。

    只实现 Judge 真正调用的三个方法：

      * `render(messages)`             —— 套 chat 模板，返回 prompt 文本；
      * `generate_batch(prompts)`      —— 六要素抽取，输出 JSON 文本；
      * `score_digits_batch(prompts)`  —— 受限解码，只吐 0~4 的分数与概率。

    不 import 项目的 loader / lora，能在干净的 vLLM 环境里跑。
    """

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        *,
        max_input_tokens: int = 8192,
        max_logprobs: int = 50,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_logprobs = max_logprobs
        # 分数 token 的 id。取 encode 的**最后一个** id：有的 tokenizer 会在
        # 数字前加前缀 token，取错会让整批分数偏移（和 TorchRuntime 同一处理）。
        self._digit_ids = [
            tokenizer.encode(d, add_special_tokens=False)[-1] for d in DIGITS
        ]
        self.stats: Dict[str, int] = {"generate_calls": 0, "sequences": 0}

    # ---------------------------------------------------------------- 渲染
    def render(self, messages: List[Dict[str, str]]) -> str:
        """和 TorchRuntime.render 完全一致：优先用模型自己的 chat 模板。"""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "\n".join(m["content"] for m in messages)

    def _encode(self, prompts: Sequence[str]) -> List[List[int]]:
        """逐条 tokenize（不 padding），截断到 max_input_tokens。"""
        enc = self.tokenizer(
            list(prompts),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        return list(enc["input_ids"])

    # 和 TorchRuntime 同一套：只截输入文本、保留头尾，system 与生成标记不动
    def count_tokens(self, text: str) -> int:
        return _count_tokens(self.tokenizer, text)

    def truncate_text(self, text: str, max_tokens: int, tail_ratio: float = 0.5) -> str:
        return _truncate_text(self.tokenizer, text, max_tokens, tail_ratio)

    def _generate(self, prompts: Sequence[str], sampling: Any) -> List[Any]:
        ids = self._encode(prompts)
        outputs = self.llm.generate(
            [{"prompt_token_ids": x} for x in ids],
            sampling_params=sampling,
            use_tqdm=False,
        )
        self.stats["generate_calls"] += 1
        self.stats["sequences"] += len(ids)
        return outputs

    # ---------------------------------------------------------------- 抽取
    def generate_batch(
        self, prompts: Sequence[str], max_new_tokens: int = 512
    ) -> List[str]:
        """批量生成（六要素 JSON）。贪心，和 HF 路径的 do_sample=False 对齐。"""
        from vllm import SamplingParams

        sampling = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )
        return [o.outputs[0].text.strip() for o in self._generate(prompts, sampling)]

    # ---------------------------------------------------------------- 打分
    def score_digits_batch(
        self, prompts: Sequence[str], digits: str = DIGITS
    ) -> Tuple[List[int], List[float], List[List[float]]]:
        """受限解码：在 0~4 五个 token 的 logprob 上 softmax，取 argmax。

        和 HF 版（`TorchRuntime.score_digits_batch`）语义相同：只做 prefill、
        只取第一个生成位置的分布，不存在"生成非法输出"这回事。
        """
        from vllm import SamplingParams

        assert digits == DIGITS, "分数词表变了要同步改 _digit_ids"
        sampling = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            logprobs=self.max_logprobs,
            skip_special_tokens=True,
        )
        scores: List[int] = []
        pmaxs: List[float] = []
        probs_all: List[List[float]] = []
        for output in self._generate(prompts, sampling):
            logprobs = getattr(output.outputs[0], "logprobs", None)
            dist = logprobs[0] if logprobs else {}
            logp = []
            for tid in self._digit_ids:
                entry = dist.get(tid)
                logp.append(float(entry.logprob) if entry is not None else float("-inf"))

            if all(v == float("-inf") for v in logp):
                # 五个数字 token 一个都没进 top-logprobs：退回贪心 token 本身。
                # 正常不会发生（prompt 明确要求只输出一个整数）；这里只是兜底，
                # 保证不会因为 logprobs 截断把整条记录打成 0。
                token_ids = getattr(output.outputs[0], "token_ids", []) or []
                greedy = int(token_ids[0]) if token_ids else -1
                best = self._digit_ids.index(greedy) if greedy in self._digit_ids else 0
                probs = [0.0] * len(DIGITS)
                probs[best] = 1.0
                pmax = float("nan")
            else:
                top = max(logp)
                exps = [math.exp(v - top) if v != float("-inf") else 0.0 for v in logp]
                total = sum(exps) or 1.0
                probs = [e / total for e in exps]
                best = max(range(len(probs)), key=lambda i: probs[i])
                pmax = probs[best]

            scores.append(best)
            pmaxs.append(pmax)
            probs_all.append([round(p, 4) for p in probs])
        return scores, pmaxs, probs_all


def build_vllm(
    args: argparse.Namespace,
    model_name: str,
    trust_remote_code: bool,
    max_model_len: int,
) -> Tuple[VLLMRuntime, Any]:
    """按 generate_triples_vllm.py 的方式构造 vLLM + tokenizer。"""
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or model_name,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )

    llm_kwargs: Dict[str, Any] = dict(
        model=model_name,
        tokenizer=args.tokenizer or model_name,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        swap_space=args.swap_space,
        disable_log_stats=True,
        seed=args.seed,
        # 受限解码最多要看这么多个 logprob，引擎得提前放行
        max_logprobs=max(args.logprobs, 1),
    )
    if args.dtype and args.dtype != "auto":
        llm_kwargs["dtype"] = args.dtype
    if args.quantization and args.quantization != "none":
        # 量化在加载时决定：AWQ / GPTQ / bitsandbytes 都在这里选。
        llm_kwargs["quantization"] = args.quantization
        if args.quantization == "bitsandbytes":
            # vLLM 的 create_engine_config 有条硬校验：
            #   quantization="bitsandbytes" 必须搭配 load_format="bitsandbytes"，
            # 否则直接抛
            #   ValueError: BitsAndBytes quantization and QLoRA adapter only
            #               support 'bitsandbytes' load format, but got auto
            # 这个 loader 同时负责两件事：
            #   * 预量化 checkpoint（权重里带 quant_state.bitsandbytes__*）
            #   * in-flight 量化（fp16 权重没有 quant_state → 现 quantize_4bit）
            # 我们要的是后者，所以 load_format 必须显式给上。
            llm_kwargs["load_format"] = "bitsandbytes"
    # 允许显式覆盖 load_format（GGUF / 预量化 bnb checkpoint 等）。
    # 默认 "auto"，上面已经按量化方案填好，这里只在用户真的指定时才盖。
    if getattr(args, "load_format", "auto") not in (None, "", "auto"):
        llm_kwargs["load_format"] = args.load_format

    logger.info(
        "加载 vLLM 裁判：%s（dtype=%s, quantization=%s, load_format=%s, tp=%d, max_model_len=%d）",
        model_name, args.dtype, args.quantization,
        llm_kwargs.get("load_format", "auto"),
        args.tensor_parallel_size, max_model_len,
    )
    llm = LLM(**llm_kwargs)
    runtime = VLLMRuntime(
        llm=llm,
        tokenizer=tokenizer,
        max_input_tokens=args.max_input_tokens,
        max_logprobs=args.logprobs,
    )
    return runtime, tokenizer


# ---------------------------------------------------------------------------
# 分块打分
# ---------------------------------------------------------------------------
def _collect_unique(
    records: Sequence[Dict[str, Any]],
    candidate_key: str,
    reference_key: str,
    seen: Dict[str, SixElements],
) -> Tuple[List[str], List[str]]:
    """收集这一批里还没抽过要素的文本，分成 (原文, 摘要) 两组。

    判决书原文和摘要用**两套抽取 prompt**，所以必须分开批量。
    """
    docs: List[str] = []
    summaries: List[str] = []
    for rec in records:
        doc = rec.get("source") or ""
        if doc and doc not in seen and doc not in docs:
            docs.append(doc)
        for text in (rec.get(reference_key) or "", rec.get(candidate_key) or ""):
            if text and text not in seen and text not in docs and text not in summaries:
                summaries.append(text)
    return docs, summaries


def score_chunk(
    judge: Any,
    spec: RewardSpec,
    spec_ungated: RewardSpec,
    records: Sequence[Dict[str, Any]],
    els_cache: Dict[str, SixElements],
    *,
    candidate_key: str,
    reference_key: str,
    min_document_elements: int,
) -> List[Dict[str, Any]]:
    """给一批记录打分，返回每行含候选臂 / 人工臂的结果。"""
    tasks = set(judge.tasks)

    # ---- 1. 一次把这一批需要的文本都抽出来 -------------------------------
    docs_todo, summaries_todo = _collect_unique(
        records, candidate_key, reference_key, els_cache
    )
    for texts, kind in ((docs_todo, "document"), (summaries_todo, "summary")):
        if not texts:
            continue
        extracted = judge.extract_six_elements_batch(texts, kind=kind)
        for text, el in zip(texts, extracted):
            els_cache[text] = el

    # ---- 2. 组装两臂的判定请求（先攒起来，最后合成一次批量前向）---------
    all_pairs: List[Tuple[str, str, str]] = []
    entries: List[Dict[str, Any]] = []
    for rec in records:
        rid = str(rec.get("id", ""))
        document = rec.get("source") or ""
        reference = rec.get(reference_key) or ""
        output = rec.get(candidate_key) or ""

        doc_el = els_cache.get(document, SixElements())
        ref_el = els_cache.get(reference, SixElements())
        error = ""
        if doc_el.n_filled() < min_document_elements:
            error = (
                f"原文六要素只抽到 {doc_el.n_filled()} 项（要求 ≥{min_document_elements}）"
            )

        for arm, cand_text, cand_el in (
            ("candidate", output, els_cache.get(output, SixElements())),
            ("human", reference, ref_el),
        ):
            entry: Dict[str, Any] = {
                "id": rid,
                "arm": arm,
                "candidate": cand_text,
                "reference": reference,
                "error": error,
                # 诊断用（不落盘，除非 --dump-elements）
                "_cand_el": cand_el,
                "_doc_el": doc_el,
                "_ref_el": ref_el,
            }
            if "fact_consistency" in tasks:
                cons_pairs = judge.build_pairs(doc_el, cand_el, document)
                entry["cons_slice"] = (len(all_pairs), len(all_pairs) + len(cons_pairs))
                all_pairs.extend(cons_pairs)
            if "element_coverage" in tasks:
                cov_pairs, present, precomputed = judge.build_coverage_pairs(ref_el, cand_el)
                entry["cov_slice"] = (len(all_pairs), len(all_pairs) + len(cov_pairs))
                entry["cov_names"] = [p[0] for p in cov_pairs]
                entry["cov_present"] = list(present)
                entry["cov_pre"] = dict(precomputed)
                all_pairs.extend(cov_pairs)
            entries.append(entry)

    # ---- 3. 一次批量前向（空字段规则在 judge_elements_with_confidence 里处理）
    if all_pairs:
        scores, sources, _pmaxs = judge.judge_elements_with_confidence(all_pairs)
    else:
        scores, sources = [], []

    # ---- 4. 切回每条记录的两臂，聚合 + 算奖励 ---------------------------
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        signals: Dict[str, Optional[float]] = {}
        fact_result = None
        coverage = None
        cov_raw: Dict[str, int] = {}
        cov_present: List[str] = []
        error = entry["error"]
        try:
            if "fact_consistency" in tasks and "cons_slice" in entry:
                a, b = entry["cons_slice"]
                raw = {name: v for name, v in zip(ELEMENTS, scores[a:b])}
                src = {name: v for name, v in zip(ELEMENTS, sources[a:b])}
                fact_result = aggregate(
                    raw, weights=judge.weights, sources=src,
                    candidate_id=f"{entry['id']}:{entry['arm']}",
                )
                signals["fact_consistency"] = fact_result.weighted_reward
                # 事实硬门控要读的六要素最小原始分（与 reward_spec 的信号同名）
                signals["fact_consistency_min_raw"] = float(
                    min(fact_result.raw_scores.values())
                )
            if "element_coverage" in tasks and "cov_slice" in entry:
                a, b = entry["cov_slice"]
                cov_present = list(entry["cov_present"])
                # 分项原始分：参考里没有的要素不参与（不在 dict 里）；参考有、
                # 候选没写的要素预置 0；其余是模型判出来的 0~4。
                cov_raw = dict(entry["cov_pre"])
                for name, value in zip(entry["cov_names"], scores[a:b]):
                    cov_raw[name] = value
                coverage = aggregate_coverage(
                    cov_raw, cov_present, judge.coverage_weights
                )
                signals["element_coverage"] = coverage
        except Exception as exc:  # noqa: BLE001 — 单条失败不该毁掉整批
            error = error or f"{type(exc).__name__}: {exc}"

        reward = reward_ungated = None
        gated = False
        gate_reason = ""
        terms: Dict[str, float] = {}
        if not error:
            try:
                reward, bd = compute_reward(
                    entry["candidate"], entry["reference"], source=None,
                    spec=spec, judge_signals=signals,
                )
                gated, gate_reason = bd.gated, bd.gate_reason
                # 门控会把分项清空，所以再走一遍"不设门控"的规格拿分项明细
                reward_ungated, bd_un = compute_reward(
                    entry["candidate"], entry["reference"], source=None,
                    spec=spec_ungated, judge_signals=signals,
                )
                terms = dict(bd_un.values)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"

        # ROUGE 始终独立计算一遍：它是“结论 2”的自变量，不能依赖它是否被
        # 配成 reward 项（覆盖率的配置里 rouge_l 是关掉的）。人工臂的候选就是
        # 人工摘要本身，ROUGE 自然是 1.0。
        rouge: Dict[str, float] = {}
        if entry["candidate"].strip() and entry["reference"].strip():
            try:
                rouge = {
                    key: round(float(value), 6)
                    for key, value in score_pair(
                        entry["candidate"].strip(),
                        entry["reference"].strip(),
                        mode=spec.rouge_mode,
                    ).items()
                }
            except Exception as exc:  # noqa: BLE001 — ROUGE 失败不该毁掉 reward
                _warn_rouge_once(exc)
                rouge = {}

        row = {
            "id": entry["id"],
            "arm": entry["arm"],
            "reward": reward,
            "reward_ungated": reward_ungated,
            "gated": gated,
            "gate_reason": gate_reason,
            "terms": terms,
            "rouge": rouge,
            "rouge_l": rouge.get("rouge-l-f"),
            "fact_consistency": (fact_result.weighted_reward if fact_result else None),
            "min_element_score": (fact_result.min_element_score if fact_result else None),
            "judgment_result_score": (
                fact_result.judgment_result_score if fact_result else None
            ),
            "fact_scores": (dict(fact_result.scores) if fact_result else {}),
            "fact_raw": (dict(fact_result.raw_scores) if fact_result else {}),
            "element_coverage": coverage,
            # 覆盖率的分项：原始 0~4 / 归一化 0~1 / 参与计算（参考里真实存在）的要素。
            # 参考里没有的要素不进这两个 dict（分子分母都不算）。
            "coverage_raw": dict(cov_raw),
            "coverage_scores": {
                name: round(value / float(MAX_SCORE), 6)
                for name, value in cov_raw.items()
            },
            "coverage_present": cov_present,
            "error": error,
        }
        if _DUMP_ELEMENTS:
            row["elements"] = {
                "document": entry["_doc_el"].to_dict(),
                "reference": entry["_ref_el"].to_dict(),
                "candidate": entry["_cand_el"].to_dict(),
            }
        rows.append(row)
    return rows


def score_with_fallback(
    judge: Any,
    spec: RewardSpec,
    spec_ungated: RewardSpec,
    records: Sequence[Dict[str, Any]],
    els_cache: Dict[str, SixElements],
    *,
    candidate_key: str,
    reference_key: str,
    min_document_elements: int,
) -> List[Dict[str, Any]]:
    """整批失败就二分回退，一直到单条；保证不会因为一条坏记录丢掉一整批。"""
    try:
        return score_chunk(
            judge, spec, spec_ungated, records, els_cache,
            candidate_key=candidate_key, reference_key=reference_key,
            min_document_elements=min_document_elements,
        )
    except Exception as exc:  # noqa: BLE001
        if len(records) <= 1:
            rid = str(records[0].get("id", "")) if records else ""
            logger.warning("记录 %s 打分失败：%s", rid, exc)
            return [{
                "id": rid, "arm": arm, "reward": None, "reward_ungated": None,
                "gated": False, "gate_reason": "", "terms": {},
                "fact_consistency": None, "min_element_score": None,
                "judgment_result_score": None, "fact_scores": {}, "fact_raw": {},
                "element_coverage": None,
                "error": f"{type(exc).__name__}: {exc}",
            } for arm in ("candidate", "human")]
        mid = len(records) // 2
        logger.warning("整批 %d 条打分失败（%s），二分回退", len(records), exc)
        return score_with_fallback(
            judge, spec, spec_ungated, records[:mid], els_cache,
            candidate_key=candidate_key, reference_key=reference_key,
            min_document_elements=min_document_elements,
        ) + score_with_fallback(
            judge, spec, spec_ungated, records[mid:], els_cache,
            candidate_key=candidate_key, reference_key=reference_key,
            min_document_elements=min_document_elements,
        )


# ---------------------------------------------------------------------------
# IO / 汇总
# ---------------------------------------------------------------------------
def load_records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                ids.add(str(json.loads(line).get("id")))
            except json.JSONDecodeError:
                continue
    return ids


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else (clean[mid - 1] + clean[mid]) / 2.0


# ---------------------------------------------------------------------------
# 相关性与配对检验（纯标准库，不依赖 scipy / numpy）
# ---------------------------------------------------------------------------
def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _rank_avg(values: Sequence[float]) -> List[float]:
    """平均秩（并列取平均），Spearman 用它替换原值。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _fisher_z_p(r: Optional[float], n: int) -> Optional[float]:
    """Fisher z 变换的正态近似双侧 p 值（n 大时与 t 检验几乎一致）。"""
    if r is None or n < 4:
        return None
    if abs(r) >= 1.0:
        # 完全相关，z 发散；样本量够就直接判显著（小样本会在 n<4 被拦掉）
        return 0.0
    z = math.atanh(r) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2))


def _corr_block(
    xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]
) -> Dict[str, Any]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 2:
        return {"n": n, "pearson": None, "pearson_p": None,
                "spearman": None, "spearman_p": None}
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    pearson = _pearson(x, y)
    spearman = _pearson(_rank_avg(x), _rank_avg(y))
    return {
        "n": n,
        "pearson": pearson,
        "pearson_p": _fisher_z_p(pearson, n),
        "spearman": spearman,
        "spearman_p": _fisher_z_p(spearman, n),
    }


def _paired_delta_stats(deltas: Sequence[float]) -> Dict[str, Any]:
    """配对差（人工 − 候选）的均值 / 显著性。"""
    n = len(deltas)
    if n == 0:
        return {"n": 0, "mean_delta": None, "median_delta": None, "sd_delta": None,
                "win": 0, "tie": 0, "lose": 0, "win_rate": None,
                "t_stat": None, "p_value": None, "sign_test_p": None}
    mean = sum(deltas) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1)) if n > 1 else 0.0
    win = sum(1 for d in deltas if d > 1e-9)
    lose = sum(1 for d in deltas if d < -1e-9)
    tie = n - win - lose
    if n > 1 and sd > 0:
        t_stat = mean / (sd / math.sqrt(n))
        p_value = math.erfc(abs(t_stat) / math.sqrt(2))     # 正态近似双侧
    else:
        t_stat = None
        p_value = 1.0 if mean == 0 else 0.0
    return {
        "n": n, "mean_delta": mean, "median_delta": _median(deltas), "sd_delta": sd,
        "win": win, "tie": tie, "lose": lose, "win_rate": win / n,
        "t_stat": t_stat, "p_value": p_value,
        # 符号检验是精确的（不依赖正态假设），"候选一般比人工小"本质是符号命题，
        # 所以判定用它、t 检验只作参考。
        "sign_test_p": _sign_test_p(win, lose),
    }


def _sign_test_p(win: int, lose: int) -> Optional[float]:
    """双侧二项精确检验（p=0.5），只看胜/负、忽略平局。"""
    n = win + lose
    if n == 0:
        return None
    k = min(win, lose)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    # 用整数除法再转 float：tail 可能大到 1e300+，先乘 2.0 会溢出成 inf。
    # Python 的 int/int 真除法会自己正确处理大整数，商 ≤ 1 一定可表示。
    return min(1.0, 2 * tail / (2 ** n))


def _c1_verdict(c1: Dict[str, Any], passed: bool, subject: str = "候选摘要奖励") -> str:
    if c1["n"] == 0:
        return "样本不足，无法判定"
    if passed:
        return f"✓ 成立：{subject}显著低于人工摘要"
    if (c1["mean_delta"] or 0) > 0 and c1["win"] > c1["lose"]:
        return "△ 方向成立但不显著"
    return f"✗ 不成立：{subject}没有低于人工摘要"


def _c2_verdict(corr: Dict[str, Any], passed: bool, subject: str = "候选 reward") -> str:
    if corr["n"] < 3:
        return "样本不足，无法判定"
    if passed:
        return f"✓ 成立：{subject} 与 ROUGE-L 显著正相关"
    if (corr["spearman"] or 0) > 0:
        return "△ 正相关但不显著"
    return f"✗ 不成立：{subject} 与 ROUGE-L 非正相关"


# 分项单独当 reward 时的显示名
_TERM_LABELS: Dict[str, str] = {
    "fact_consistency": "候选摘要的事实一致性",
    "element_coverage": "候选摘要的要素覆盖率",
    "rouge_l": "候选摘要的 ROUGE-L",
}


def _term_label(name: str) -> str:
    return _TERM_LABELS.get(name, f"候选摘要的 {name}")


def _term_value(row: Dict[str, Any], name: str) -> Optional[float]:
    """取出某个分项的信号值（judge 信号在顶层，规则项在 terms 里）。"""
    value = row.get(name)
    if value is None:
        value = (row.get("terms") or {}).get(name)
    return float(value) if value is not None else None


def _term_value_gated(row: Dict[str, Any], name: str) -> Optional[float]:
    """把分项单独当 reward 时也套上全局门控：被拦下就是 0。"""
    value = _term_value(row, name)
    if value is None:
        return None
    return 0.0 if row.get("gated") else value


def summarize(
    rows_by_id: Dict[str, Dict[str, Dict[str, Any]]],
    terms: Sequence[str],
) -> Dict[str, Any]:
    """按 id 把两臂配对，产出两条待验证结论 + 分项统计。"""
    cand = [pair["candidate"] for pair in rows_by_id.values()]
    hum = [pair["human"] for pair in rows_by_id.values()]
    paired = [
        (pair["candidate"], pair["human"]) for pair in rows_by_id.values()
        if pair["candidate"].get("reward") is not None
        and pair["human"].get("reward") is not None
    ]

    def arm_stats(arms: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        ok = [a for a in arms if a.get("reward") is not None]
        out: Dict[str, Any] = {
            "n": len(arms),
            "n_ok": len(ok),
            "mean_reward": _mean([a["reward"] for a in ok]),
            "mean_reward_ungated": _mean([a.get("reward_ungated") for a in ok]),
            "gate_rate": (
                sum(1 for a in arms if a.get("gated")) / len(arms) if arms else None
            ),
            "gate_reasons": dict(Counter(
                a.get("gate_reason") or "?" for a in arms if a.get("gated")
            )),
            "mean_fact_consistency": _mean([a.get("fact_consistency") for a in ok]),
            "mean_element_coverage": _mean([a.get("element_coverage") for a in ok]),
            "mean_rouge_l": _mean([a.get("rouge_l") for a in ok]),
            "mean_rouge_overall": _mean(
                [a.get("rouge", {}).get("overall") for a in ok]
            ),
        }
        for term in terms:
            key = f"mean_{term}"
            # 别覆盖上面按信号直接算的那几个（terms 里缺键时会把它们冲成 None）
            if key not in out:
                out[key] = _mean([a.get("terms", {}).get(term) for a in ok])
        return out

    cand_stats, hum_stats = arm_stats(cand), arm_stats(hum)
    delta = None
    if cand_stats["mean_reward"] is not None and hum_stats["mean_reward"] is not None:
        delta = hum_stats["mean_reward"] - cand_stats["mean_reward"]

    # ---- 结论 1：候选摘要的奖励一般比人工摘要小（配对检验） ----
    c1 = _paired_delta_stats([h["reward"] - c["reward"] for c, h in paired])
    c1_pass = (
        c1["n"] > 0
        and c1["mean_delta"] is not None and c1["mean_delta"] > 0
        and c1["win"] > c1["lose"]
        and c1["sign_test_p"] is not None and c1["sign_test_p"] < 0.05
    )
    c1_verdict = _c1_verdict(c1, c1_pass)

    # ---- 结论 2：候选的 reward 与 ROUGE 正相关（跨文书） ----
    # 主口径用真正进训练的 gated reward；不含门控的那份一并报告，用来判断
    # 相关性是不是被门控的一堆 0 分"制造"出来的。
    corr_gated = _corr_block(
        [a.get("reward") for a in cand], [a.get("rouge_l") for a in cand]
    )
    corr_ungated = _corr_block(
        [a.get("reward_ungated") for a in cand], [a.get("rouge_l") for a in cand]
    )
    rho = corr_gated["spearman"]
    c2_pass = (
        corr_gated["n"] >= 3 and rho is not None and rho > 0
        and corr_gated["spearman_p"] is not None and corr_gated["spearman_p"] < 0.05
    )
    c2_verdict = _c2_verdict(corr_gated, c2_pass)

    # ---- 分项单独作为 reward：同样的两条结论各跑一遍 ----
    # 目的：看清楚"人工 > 候选"和"与 ROUGE 正相关"到底是哪个分项撑起来的。
    # element_coverage 要特别注意：人工臂是 ref vs ref，恒等于 1.0，
    # 所以它的结论 1 是**结构性成立**，不代表判别力。
    per_term: Dict[str, Any] = {}
    for name in terms:
        variants: Dict[str, Any] = {}
        for variant, value_fn in (
            ("gated", lambda row, n=name: _term_value_gated(row, n)),
            ("ungated", lambda row, n=name: _term_value(row, n)),
        ):
            deltas = [
                value_fn(h) - value_fn(c) for c, h in paired
                if value_fn(c) is not None and value_fn(h) is not None
            ]
            t1 = _paired_delta_stats(deltas)
            t1_pass = (
                t1["n"] > 0 and t1["mean_delta"] is not None and t1["mean_delta"] > 0
                and t1["win"] > t1["lose"]
                and t1["sign_test_p"] is not None and t1["sign_test_p"] < 0.05
            )
            corr = _corr_block(
                [value_fn(a) for a in cand], [a.get("rouge_l") for a in cand]
            )
            rho_t = corr["spearman"]
            t2_pass = (
                corr["n"] >= 3 and rho_t is not None and rho_t > 0
                and corr["spearman_p"] is not None and corr["spearman_p"] < 0.05
            )
            variants[variant] = {
                "conclusion1": {**t1, "pass": t1_pass,
                                "verdict": _c1_verdict(t1, t1_pass, _term_label(name))},
                "conclusion2": {**corr, "pass": t2_pass,
                                "verdict": _c2_verdict(corr, t2_pass, _term_label(name))},
            }
        human_values = [_term_value(h, name) for _, h in paired]
        variants["candidate_mean"] = _mean([_term_value(c, name) for c, _ in paired])
        variants["human_mean"] = _mean(human_values)
        # 人工臂在该分项上恒为 1.0 → 它的结论 1 是结构性成立，不代表判别力
        variants["human_constant_one"] = bool(human_values) and all(
            abs(v - 1.0) < 1e-9 for v in human_values
        )
        per_term[name] = variants

    if c1_pass and c2_pass:
        overall = "✓ 两条结论都成立"
    elif c1_pass or c2_pass:
        overall = "△ 只有一条结论成立"
    else:
        overall = "✗ 两条结论都不成立"

    return {
        "n_records": len(rows_by_id),
        "n_paired": len(paired),
        "candidate": cand_stats,
        "human": hum_stats,
        "delta_reward(human-candidate)": delta,
        "human_win": c1["win"],
        "tie": c1["tie"],
        "human_lose": c1["lose"],
        "human_win_rate": c1["win_rate"],
        "verdict": overall,
        "conclusions": {
            "candidate_reward_lt_human_reward": {**c1, "pass": c1_pass,
                                                 "verdict": c1_verdict},
            "reward_rouge_positive_correlation": {
                "primary": "reward(含门控) vs rouge-l-f",
                "gated": corr_gated,
                "ungated": corr_ungated,
                "pass": c2_pass,
                "verdict": c2_verdict,
            },
        },
        "per_term": per_term,
    }


def pair_rows(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["id"], {})[row["arm"]] = row
    return {
        rid: pair for rid, pair in out.items()
        if "candidate" in pair and "human" in pair
    }


def print_summary(title: str, stats: Dict[str, Any]) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if not stats.get("n_records"):
        print("  没有可统计的记录")
        return
    c, h = stats["candidate"], stats["human"]
    print(f"  记录 {stats['n_records']} 条，两臂都可用的配对 {stats['n_paired']} 条")
    print(f"  {'指标':<24}{'候选(output)':>16}{'人工(reference)':>18}")
    print("  " + "-" * 58)
    for key, label in (
        ("mean_reward", "奖励均值（含门控）"),
        ("mean_reward_ungated", "奖励均值（不含门控）"),
        ("mean_fact_consistency", "事实一致性均值"),
        ("mean_element_coverage", "要素覆盖率均值"),
        ("mean_rouge_l", "ROUGE-L 均值"),
        ("gate_rate", "门控拦截率"),
    ):
        cv, hv = c.get(key), h.get(key)
        cvs = f"{cv:.4f}" if isinstance(cv, float) else "  -  "
        hvs = f"{hv:.4f}" if isinstance(hv, float) else "  -  "
        print(f"  {label:<24}{cvs:>16}{hvs:>18}")
    if stats["delta_reward(human-candidate)"] is not None:
        print(f"\n  Δ 奖励（人工 − 候选）= {stats['delta_reward(human-candidate)']:+.4f}")
    if stats["human_win_rate"] is not None:
        print(
            f"  人工胜 / 平 / 负 = {stats['human_win']} / {stats['tie']} / "
            f"{stats['human_lose']}（胜率 {stats['human_win_rate']:.1%}）"
        )
    _print_conclusions(stats)
    print(f"\n  总判定：{stats['verdict']}")


def _print_conclusions(stats: Dict[str, Any]) -> None:
    if "conclusions" not in stats:
        return
    c1 = stats["conclusions"]["candidate_reward_lt_human_reward"]
    print(f"\n  结论 1（候选奖励 < 人工奖励）：{c1['verdict']}")
    print(f"    配对 n={c1['n']}  Δ均值={_fmt(c1['mean_delta'])}  "
          f"胜率={_fmt(c1['win_rate'])}  符号检验 p={_fmt(c1['sign_test_p'])}  "
          f"t 检验 p={_fmt(c1['p_value'])}")
    print(f"    门控原因：候选 {stats['candidate'].get('gate_reasons')} / "
          f"人工 {stats['human'].get('gate_reasons')}")

    c2 = stats["conclusions"]["reward_rouge_positive_correlation"]
    g, u = c2["gated"], c2["ungated"]
    print(f"\n  结论 2（候选 reward 与 ROUGE 正相关）：{c2['verdict']}")
    print(f"    {c2['primary']}：n={g['n']}  Pearson r={_fmt(g['pearson'])}"
          f"  Spearman ρ={_fmt(g['spearman'])}  p={_fmt(g['spearman_p'])}")
    print(f"    不含门控 reward vs rouge-l-f：n={u['n']}  "
          f"Pearson r={_fmt(u['pearson'])}  Spearman ρ={_fmt(u['spearman'])}  "
          f"p={_fmt(u['spearman_p'])}")

    per_term = stats.get("per_term") or {}
    if per_term:
        print("\n  分项单独作为 reward（含门控）：")
        for name, entry in per_term.items():
            blk = entry["gated"]
            c1, c2 = blk["conclusion1"], blk["conclusion2"]
            note = "（人工恒 1.0，结论1结构性成立）" if entry.get(
                "human_constant_one") else ""
            print(
                f"    {name:<18} 结论1 {'✓' if c1['pass'] else '✗'}"
                f"（Δ={_fmt(c1['mean_delta'])}）  "
                f"结论2 {'✓' if c2['pass'] else '✗'}"
                f"（ρ={_fmt(c2['spearman'])}）{note}"
            )


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, float) else "-"


def render_report(title: str, stats: Dict[str, Any]) -> str:
    """把两条结论连同支撑数据写成 Markdown。"""
    c, h = stats["candidate"], stats["human"]
    c1 = stats["conclusions"]["candidate_reward_lt_human_reward"]
    c2 = stats["conclusions"]["reward_rouge_positive_correlation"]
    g, u = c2["gated"], c2["ungated"]

    lines = [
        f"# reward 设计验证报告：{title}",
        "",
        f"- 记录数：{stats['n_records']}（两臂都可用的配对 {stats['n_paired']}）",
        f"- 输入：`{stats.get('input', '')}`",
        f"- 逐条输出：`{stats.get('output', '')}`",
        f"- 总判定：**{stats['verdict']}**",
        "",
        "## 结论 1：候选摘要的奖励一般比人工摘要小",
        "",
        f"**{c1['verdict']}**",
        "",
        "| 指标 | 候选(output) | 人工(reference) |",
        "|---|---:|---:|",
        f"| 奖励均值（含门控） | {_fmt(c['mean_reward'])} | {_fmt(h['mean_reward'])} |",
        f"| 奖励均值（不含门控） | {_fmt(c['mean_reward_ungated'])} | "
        f"{_fmt(h['mean_reward_ungated'])} |",
        f"| 门控拦截率 | {_fmt(c['gate_rate'])} | {_fmt(h['gate_rate'])} |",
        "",
        f"- 配对差（人工 − 候选）：均值 {_fmt(c1['mean_delta'])}，"
        f"中位数 {_fmt(c1['median_delta'])}，标准差 {_fmt(c1['sd_delta'])}",
        f"- 配对胜负：胜 {c1['win']} / 平 {c1['tie']} / 负 {c1['lose']}"
        f"（胜率 {_fmt(c1['win_rate'])}）",
        f"- 符号检验（精确，双侧）：p={_fmt(c1['sign_test_p'])} ← 判定依据",
        f"- 配对 t 检验（正态近似，双侧）：t={_fmt(c1['t_stat'])}，"
        f"p={_fmt(c1['p_value'])}（参考）",
        f"- 门控原因：候选 `{c['gate_reasons']}`；人工 `{h['gate_reasons']}`",
        "",
        "## 结论 2：候选的 reward 与 ROUGE 正相关",
        "",
        f"**{c2['verdict']}**",
        "",
        f"主口径：`{c2['primary']}`",
        "",
        "| 口径 | n | Pearson r | Pearson p | Spearman ρ | Spearman p |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 含门控 reward | {g['n']} | {_fmt(g['pearson'])} | {_fmt(g['pearson_p'])} "
        f"| {_fmt(g['spearman'])} | {_fmt(g['spearman_p'])} |",
        f"| 不含门控 reward | {u['n']} | {_fmt(u['pearson'])} | {_fmt(u['pearson_p'])} "
        f"| {_fmt(u['spearman'])} | {_fmt(u['spearman_p'])} |",
        "",
        "## 分项均值",
        "",
        "| 指标 | 候选(output) | 人工(reference) |",
        "|---|---:|---:|",
        f"| 事实一致性 | {_fmt(c['mean_fact_consistency'])} | "
        f"{_fmt(h['mean_fact_consistency'])} |",
        f"| 要素覆盖率 | {_fmt(c['mean_element_coverage'])} | "
        f"{_fmt(h['mean_element_coverage'])} |",
        f"| ROUGE-L F1 | {_fmt(c['mean_rouge_l'])} | {_fmt(h['mean_rouge_l'])} |",
        f"| ROUGE overall | {_fmt(c['mean_rouge_overall'])} | "
        f"{_fmt(h['mean_rouge_overall'])} |",
        "",
    ]
    per_term = stats.get("per_term") or {}
    if per_term:
        lines += [
            "## 分项单独作为 reward",
            "",
            "把每个分项单独当作 reward，重跑上面两条结论（同一套配对与相关检验）。"
            "「含门控」= 被门控拦下的候选按 0 分算；「不含门控」= 只看分项本身。",
            "",
        ]
        for name, entry in per_term.items():
            flag = (
                "（人工臂恒为 1.0 → 结论 1 属结构性成立，不代表判别力）"
                if entry.get("human_constant_one") else ""
            )
            lines += [
                f"### {_term_label(name)}",
                f"- 候选均值 {_fmt(entry.get('candidate_mean'))} / "
                f"人工均值 {_fmt(entry.get('human_mean'))}{flag}",
                "",
                "| 口径 | Δ均值 | 胜率 | 符号检验 p | 结论1 | "
                "Spearman ρ | ρ p | 结论2 |",
                "|---|---:|---:|---:|---|---:|---:|---|",
            ]
            for variant, label in (("gated", "含门控"), ("ungated", "不含门控")):
                blk = entry[variant]
                t1, t2 = blk["conclusion1"], blk["conclusion2"]
                lines.append(
                    f"| {label} | {_fmt(t1['mean_delta'])} | {_fmt(t1['win_rate'])} | "
                    f"{_fmt(t1['sign_test_p'])} | {'✓' if t1['pass'] else '✗'} | "
                    f"{_fmt(t2['spearman'])} | {_fmt(t2['spearman_p'])} | "
                    f"{'✓' if t2['pass'] else '✗'} |"
                )
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_input(
    args: argparse.Namespace,
    path: Path,
    judge: Any,
    spec: RewardSpec,
    spec_ungated: RewardSpec,
    min_document_elements: int,
) -> Dict[str, Any]:
    out_path = resolve(args.out_dir) / f"{path.stem}.reward.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(path)
    if args.ids:
        wanted = {str(x) for x in args.ids}
        records = [r for r in records if str(r.get("id")) in wanted]
        if not records:
            logger.warning("--ids 过滤后没有记录：%s（文件 %s）", sorted(wanted), path.name)
    if args.limit:
        records = records[: args.limit]
    done = set() if args.no_resume else load_done_ids(out_path)
    todo = [r for r in records if str(r.get("id")) not in done]
    logger.info(
        "%s：读入 %d 条，已完成 %d 条，本次打 %d 条",
        path.name, len(records), len(done), len(todo),
    )

    els_cache: Dict[str, SixElements] = {}
    started = time.time()
    count = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.chunk_size):
            chunk = todo[start:start + args.chunk_size]
            rows = score_with_fallback(
                judge, spec, spec_ungated, chunk, els_cache,
                candidate_key=args.candidate_key,
                reference_key=args.reference_key,
                min_document_elements=min_document_elements,
            )
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            count += len(chunk)
            elapsed = time.time() - started
            rate = count / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - count) / rate / 60 if rate > 0 else float("inf")
            logger.info(
                "进度 %d/%d | %.2f 条/秒 | 预计剩余 %.1f 分钟",
                count, len(todo), rate, eta,
            )

    rows_by_id = pair_rows(load_records(out_path))
    stats = summarize(rows_by_id, [t.name for t in spec.enabled_terms])
    stats["input"] = str(path)
    stats["output"] = str(out_path)
    stats["extract_model"] = getattr(args, "extract_model", None)
    stats["extract_api_model"] = getattr(args, "extract_api_model", None)
    stats["max_input_tokens"] = args.max_input_tokens
    try:  # 记录抽取 prompt 版本，换过 prompt 的结果才不会混在一起比
        from sfzy.judge.prompts import (
            EXTRACT_PROMPT_VERSION,
            EXTRACT_SUMMARY_PROMPT_VERSION,
        )

        stats["extract_prompt_version"] = EXTRACT_PROMPT_VERSION
        stats["extract_summary_prompt_version"] = EXTRACT_SUMMARY_PROMPT_VERSION
    except Exception:  # noqa: BLE001
        pass
    summary_path = Path(str(out_path.with_suffix("")) + ".summary.json")
    summary_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = Path(str(out_path.with_suffix("")) + ".report.md")
    report_path.write_text(render_report(path.name, stats), encoding="utf-8")
    print_summary(path.name, stats)
    logger.info("逐条输出：%s\n汇总：%s\n报告：%s", out_path, summary_path, report_path)
    return stats


def run_merge(args: argparse.Namespace) -> None:
    """把多份 `*.reward.jsonl` 合成一份总报告（不加载裁判）。

    两个分片是分别跑、分别落盘的，最终那两条结论要在**全量**上成立才有意义。
    这个模式只读逐条结果、重新统计，所以 CPU 上几秒就能出总报告：

        python scripts/score_reward_vllm.py --merge \
            data/judge/sft_val_shard0of2.reward.jsonl \
            data/judge/sft_val_shard1of2.reward.jsonl
    """
    rows: List[Dict[str, Any]] = []
    inputs: List[str] = []
    for raw in args.merge:
        path = resolve(raw)
        if not path.exists():
            raise SystemExit(f"输入文件不存在：{path}")
        rows.extend(load_records(path))
        inputs.append(str(path))

    rows_by_id = pair_rows(rows)
    terms = sorted({key for row in rows for key in (row.get("terms") or {})})
    stats = summarize(rows_by_id, terms)
    stats["input"] = inputs

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_jsonl = out_dir / f"{args.merge_name}.reward.jsonl"
    merged_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    stats["output"] = str(merged_jsonl)
    summary_path = out_dir / f"{args.merge_name}.reward.summary.json"
    summary_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = out_dir / f"{args.merge_name}.reward.report.md"
    report_path.write_text(render_report(args.merge_name, stats), encoding="utf-8")

    print_summary(f"{args.merge_name}（{len(inputs)} 个文件合并）", stats)
    logger.info("合并输出：%s\n汇总：%s\n报告：%s", merged_jsonl, summary_path, report_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="用 vLLM 六要素 Judge 给 val 三元组打完整奖励分（候选 vs 人工）"
    )
    ap.add_argument("--config", default="configs/grpo_fact_coverage_t4.yaml",
                    help="读 rl.reward（奖励口径）和 semantic（裁判配置）")
    ap.add_argument("--input", nargs="+", default=None,
                    help="三元组 jsonl（id/source/reference/output），可给多个")
    ap.add_argument("--merge", nargs="+", default=None, metavar="REWARD_JSONL",
                    help="合并多份 *.reward.jsonl 出一份总报告（不加载裁判，CPU 即可）")
    ap.add_argument("--merge-name", default="merged",
                    help="--merge 时的输出前缀，默认 merged")
    ap.add_argument("--out-dir", default="data/judge")
    ap.add_argument("--candidate-key", default="output", help="候选摘要字段名")
    ap.add_argument("--reference-key", default="reference", help="人工摘要字段名")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="+", default=None,
                    help="只跑这些 id（调试单条用，配合 --dump-elements）")
    ap.add_argument("--no-resume", action="store_true",
                    help="默认跳过输出里已有的 id（可断点续跑）")
    ap.add_argument("--chunk-size", type=int, default=16,
                    help="一批几条记录。显存紧就调小，吞吐优先就调大")
    ap.add_argument("--dump-elements", action="store_true",
                    help="逐条结果里带上抽取出的六要素（原文/人工/候选），"
                         "用来定位某个要素为什么判 0；配合 --limit 跑一小批")

    # ---- 裁判模型 / vLLM ----
    ap.add_argument("--model", default=None,
                    help="裁判模型目录或 HF 名，默认取 semantic.model")
    ap.add_argument("--tokenizer", default=None,
                    help="单独指定 tokenizer（AWQ 目录缺文件时用）")
    ap.add_argument("--extract-model", default=None,
                    help="可选的独立抽取模型（比裁判更强/更大）。不给就共用裁判模型。"
                         "两份权重同时在显存里，单卡放不下就别用")
    ap.add_argument("--extract-tokenizer", default=None)
    ap.add_argument("--extract-quantization", default=None,
                    help="抽取模型的量化方式，默认跟随 --quantization")
    ap.add_argument("--extract-dtype", default=None,
                    help="抽取模型的精度，默认跟随 --dtype")
    ap.add_argument("--quantization", default="none",
                    help="none | bitsandbytes | awq | gptq。T4 上 7B 装不下 fp16，"
                         "必须显式指定一个 4-bit 方案（bitsandbytes 可对现有 fp16 "
                         "权重现量化，awq/gptq 则要换成量化好的 checkpoint）")
    ap.add_argument("--load-format", default="auto",
                    help="auto | bitsandbytes | safetensors | gguf ...。"
                         "量化选 bitsandbytes 时会自动设成 bitsandbytes（vLLM 的硬校验）")
    ap.add_argument("--dtype", default="float16",
                    help="T4 必须 float16；4090/A100 可用 bfloat16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--swap-space", type=int, default=4)
    ap.add_argument("--max-input-tokens", type=int, default=None,
                    help="抽取/判定 prompt 的输入预算，默认取 semantic.max_input_tokens"
                         "（再默认 8192）。长判决书（val 最长约 1.2 万字）要调大；"
                         "超预算的输入按头+尾截断，system 不丢")
    ap.add_argument("--extract-max-new-tokens", type=int, default=None,
                    help="六要素抽取的生成预算，默认取 semantic.extract_max_new_tokens")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--logprobs", type=int, default=50,
                    help="受限解码时取多少个 logprob，够覆盖 0~4 即可")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global _DUMP_ELEMENTS
    _DUMP_ELEMENTS = bool(args.dump_elements)

    if args.merge:
        run_merge(args)
        return
    if not args.input:
        raise SystemExit("要么 --input（打分），要么 --merge（合并报告）")

    cfg = load_config(resolve(args.config))
    reward_raw = cfg.get("rl", {}).get("reward") if cfg.get("rl") else None
    if not reward_raw:
        raise SystemExit(f"{args.config} 里没有 rl.reward，无法确定奖励口径")
    spec = RewardSpec.from_config(reward_raw)
    # 不含门控的同一份规格，用来报告"分项分"（门控会把分项清空）
    spec_ungated = copy.deepcopy(spec)
    spec_ungated.gate.enabled = False

    semantic = cfg.get("semantic") or {}
    model_name = args.model or semantic.get("model")
    if not model_name:
        raise SystemExit("要么 --model，要么 config 里写 semantic.model")
    local = resolve(model_name)
    if is_local_dir(local):
        model_name = str(local)
    trust_remote_code = bool(semantic.get("trust_remote_code", False))

    extract_max_new_tokens = int(
        args.extract_max_new_tokens or semantic.get("extract_max_new_tokens", 1024)
    )
    # 输入预算：命令行优先，否则读配置（默认 8192）。长判决书（val 最长约
    # 1.2 万字）需要更大预算，否则头+尾截断会砍掉中间的事实。
    if args.max_input_tokens is None:
        args.max_input_tokens = int(semantic.get("max_input_tokens", 8192))
    max_model_len = int(
        args.max_model_len or (args.max_input_tokens + 2 * extract_max_new_tokens)
    )
    min_document_elements = int(semantic.get("min_document_elements", 2))

    # vLLM 支持 4-bit（AWQ / GPTQ / bitsandbytes），但**不读** transformers 风格
    # 的 semantic.load_in_4bit —— 量化方案必须在加载时用 --quantization 指定。
    # 不提醒的话，配置写着 load_in_4bit=true 却会静默按 fp16 加载，T4 上直接 OOM。
    if semantic.get("load_in_4bit") and args.quantization == "none":
        logger.warning(
            "config 里 semantic.load_in_4bit=True，但 vLLM 不读这个开关 —— "
            "不指定 --quantization 就会按 --dtype 加载 fp16（T4 上 7B 装不下）。"
            "vLLM 是能加载 4-bit 的，二选一："
            "① 对现有 fp16 权重现量化：--quantization bitsandbytes"
            "（需装 bitsandbytes；新版 vLLM 还要 vllm-bnb-plugin）；"
            "② 换成量化好的 checkpoint：--model <AWQ/GPTQ> --quantization awq|gptq。"
        )

    weights = spec.normalized_weights()
    logger.info(
        "奖励口径：%s",
        ", ".join(f"{t.name}(w={weights[t.name]:.3f})" for t in spec.enabled_terms),
    )

    from sfzy.judge.judge import SUPPORTED_TASKS, FactConsistencyJudge

    # 只把真正的 judge task 拿去构造 Judge；事实硬门控要读的
    # fact_consistency_min_raw 是 gate 辅助信号（在 score_chunk 里手工塞进
    # judge_signals），不是 task。
    judge_tasks = [
        t for t in sorted(spec.required_signals) if t in SUPPORTED_TASKS
    ] or ["fact_consistency"]
    logger.info("裁判任务：%s（事实硬门控阈值=%d）",
                judge_tasks, spec.fact_element_min_raw)

    if spec.fact_element_gate_enabled:
        logger.info(
            "事实一致性硬门控已启用：六要素原始分任一 < %d → 整条 reward=0",
            spec.fact_element_min_raw,
        )

    runtime, _tokenizer = build_vllm(args, model_name, trust_remote_code, max_model_len)

    # 可选的独立抽取器：抽取是最容易出错的一步（日期挪用/改写），换更强的模型
    # 通常比换裁判更值。代价是显存里要多放一份权重 —— 单张 T4 上两个 7B 装不下，
    # 只有显存够（例如 A100/4090 或小抽取器）才用得上。
    extract_runtime = None
    extract_model_name = args.extract_model
    if extract_model_name:
        local = resolve(extract_model_name)
        if is_local_dir(local):
            extract_model_name = str(local)
        eargs = copy.copy(args)
        eargs.tokenizer = args.extract_tokenizer
        if args.extract_quantization:
            eargs.quantization = args.extract_quantization
        if args.extract_dtype:
            eargs.dtype = args.extract_dtype
        logger.warning(
            "额外加载抽取模型 %s：两份权重会同时占用显存。单张 T4 放不下两个 7B，"
            "请确认显存预算；不够就只用 v2 抽取 prompt、不传 --extract-model。",
            extract_model_name,
        )
        extract_runtime, _ = build_vllm(
            eargs, extract_model_name,
            bool(semantic.get("extract_trust_remote_code", trust_remote_code)),
            max_model_len,
        )
    elif semantic.get("extract_api"):
        # 方案 A：抽取走 API，判定仍用本地 vLLM。
        from sfzy.judge.api_runtime import build_api_runtime

        extract_api = dict(semantic["extract_api"])
        extract_runtime = build_api_runtime(extract_api)
        args.extract_api_model = extract_api.get("model")
        logger.info(
            "抽取走 API：base_url=%s model=%s concurrency=%s（判定仍用本地 %s）",
            extract_api.get("base_url"), extract_api.get("model"),
            extract_api.get("concurrency", 16), model_name,
        )

    options = spec.judge_term_options()
    judge = FactConsistencyJudge(
        runtime=runtime,
        extract_runtime=extract_runtime,
        weights=(options.get("fact_consistency") or {}).get("element_weights"),
        coverage_weights=(options.get("element_coverage") or {}).get("element_weights"),
        tasks=judge_tasks,
        max_input_tokens=args.max_input_tokens,
        extract_max_new_tokens=extract_max_new_tokens,
        min_document_elements=min_document_elements,
        doc_fallback=bool(semantic.get("doc_fallback", True)),
        use_element_context=bool(semantic.get("element_context", False)),
    )

    all_stats = []
    for raw_path in args.input:
        path = resolve(raw_path)
        if not path.exists():
            raise SystemExit(f"输入文件不存在：{path}")
        all_stats.append(
            run_input(args, path, judge, spec, spec_ungated, min_document_elements)
        )

    if len(all_stats) > 1:
        print(f"\n{'=' * 78}\n跨文件汇总\n{'=' * 78}")
        print(f"  {'文件':<30}{'Δ(人-候)':>11}{'胜率':>8}{'ρ(reward,rouge)':>18}")
        print("  " + "-" * 66)
        for path, stats in zip(args.input, all_stats):
            delta = stats["delta_reward(human-candidate)"]
            win_rate = stats["human_win_rate"]
            rho = stats["conclusions"][
                "reward_rouge_positive_correlation"
            ]["gated"]["spearman"]
            delta_s = f"{delta:+.4f}" if delta is not None else "-"
            rate_s = f"{win_rate:.1%}" if win_rate is not None else "-"
            rho_s = f"{rho:+.4f}" if isinstance(rho, float) else "-"
            print(f"  {Path(path).name:<30}{delta_s:>11}{rate_s:>8}{rho_s:>18}")


if __name__ == "__main__":
    main()
