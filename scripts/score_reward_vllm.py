"""用 vLLM 版六要素 Judge 给 val 三元组离线打「完整奖励」分（候选 vs 人工）。

============================ 这个脚本回答什么问题 ============================
`data/triples/sft_val_shard*of2.jsonl` 每行是

    {"id", "source"(文书原文), "reference"(人工摘要), "output"(SFT 候选摘要)}

我们要验证 GRPO 的 reward 设计是否合理，判据是：

    **人工摘要（reference）拿到的奖励，应该稳定高于模型候选（output）。**

如果 reward 把人工摘要排在候选下面，或两者几乎没差距，那这个 reward 就
没有判别力，接进 GRPO 只会得到噪声梯度。这个脚本就是先把这条基线量出来。

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

  * `<stem>.reward.jsonl`：每行一条记录的一臂奖励与分项；
  * `<stem>.reward.summary.json`：两臂均值、分项均值、胜负率、门控率。

支持断点续跑（输出里已有的 id 跳过）、`--limit`、多个 `--input`。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                          # noqa: E402
from sfzy.rl.reward import compute_reward                    # noqa: E402
from sfzy.rl.reward_spec import RewardSpec                   # noqa: E402
from sfzy.utils.logging import get_logger                    # noqa: E402

# 六要素 Judge 的受限解码头，和 sfzy/judge/runtime.py 用同一个词表
from sfzy.judge.runtime import DIGITS                        # noqa: E402
from sfzy.judge.schema import (                              # noqa: E402
    ELEMENTS,
    SixElements,
    aggregate,
    aggregate_coverage,
)

logger = get_logger("score_reward_vllm")


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
        max_input_tokens: int = 4096,
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
        # vLLM 的量化是加载时决定的：AWQ / GPTQ 走这里，bitsandbytes 不走。
        llm_kwargs["quantization"] = args.quantization

    logger.info(
        "加载 vLLM 裁判：%s（dtype=%s, quantization=%s, tp=%d, max_model_len=%d）",
        model_name, args.dtype, args.quantization, args.tensor_parallel_size,
        max_model_len,
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
) -> List[str]:
    """收集这一批里还没抽过要素的文本（保持首次出现顺序）。"""
    todo: List[str] = []
    for rec in records:
        for text in (
            rec.get("source") or "",
            rec.get(reference_key) or "",
            rec.get(candidate_key) or "",
        ):
            if text and text not in seen and text not in todo:
                todo.append(text)
    return todo


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
    todo = _collect_unique(records, candidate_key, reference_key, els_cache)
    if todo:
        extracted = judge.extract_six_elements_batch(todo)
        for text, el in zip(todo, extracted):
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
            if "element_coverage" in tasks and "cov_slice" in entry:
                a, b = entry["cov_slice"]
                raw = dict(entry["cov_pre"])
                for name, value in zip(entry["cov_names"], scores[a:b]):
                    raw[name] = value
                coverage = aggregate_coverage(
                    raw, entry["cov_present"], judge.coverage_weights
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

        rows.append({
            "id": entry["id"],
            "arm": entry["arm"],
            "reward": reward,
            "reward_ungated": reward_ungated,
            "gated": gated,
            "gate_reason": gate_reason,
            "terms": terms,
            "fact_consistency": (fact_result.weighted_reward if fact_result else None),
            "min_element_score": (fact_result.min_element_score if fact_result else None),
            "judgment_result_score": (
                fact_result.judgment_result_score if fact_result else None
            ),
            "fact_scores": (dict(fact_result.scores) if fact_result else {}),
            "fact_raw": (dict(fact_result.raw_scores) if fact_result else {}),
            "element_coverage": coverage,
            "error": error,
        })
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


def summarize(
    rows_by_id: Dict[str, Dict[str, Dict[str, Any]]],
    terms: Sequence[str],
) -> Dict[str, Any]:
    """按 id 把两臂配对后统计。"""
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
            "mean_fact_consistency": _mean([a.get("fact_consistency") for a in ok]),
            "mean_element_coverage": _mean([a.get("element_coverage") for a in ok]),
            "mean_rouge_l": _mean([a.get("terms", {}).get("rouge_l") for a in ok]),
        }
        for term in terms:
            out[f"mean_{term}"] = _mean([a.get("terms", {}).get(term) for a in ok])
        return out

    wins = sum(1 for c, h in paired if h["reward"] > c["reward"] + 1e-9)
    ties = sum(1 for c, h in paired if abs(h["reward"] - c["reward"]) <= 1e-9)
    losses = len(paired) - wins - ties
    cand_stats, hum_stats = arm_stats(cand), arm_stats(hum)
    delta = None
    if cand_stats["mean_reward"] is not None and hum_stats["mean_reward"] is not None:
        delta = hum_stats["mean_reward"] - cand_stats["mean_reward"]
    win_rate = wins / len(paired) if paired else None

    return {
        "n_records": len(rows_by_id),
        "n_paired": len(paired),
        "candidate": cand_stats,
        "human": hum_stats,
        "delta_reward(human-candidate)": delta,
        "human_win": wins,
        "tie": ties,
        "human_lose": losses,
        "human_win_rate": win_rate,
        "verdict": _verdict(delta, win_rate),
    }


def _verdict(delta: Optional[float], win_rate: Optional[float]) -> str:
    if delta is None or win_rate is None:
        return "样本不足，无法判定"
    if delta > 0 and win_rate >= 0.9:
        return "✓ reward 能稳定把人工摘要排在候选之上，设计合理"
    if delta > 0 and win_rate >= 0.7:
        return "△ reward 方向正确但判别力偏弱，建议检查分项权重/门控"
    return "✗ reward 未能把人工摘要排在候选之上，接 GRPO 前必须重修"


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
    print(f"\n  结论：{stats['verdict']}")


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
    summary_path = Path(str(out_path.with_suffix("")) + ".summary.json")
    summary_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(path.name, stats)
    logger.info("逐条输出：%s\n汇总：%s", out_path, summary_path)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="用 vLLM 六要素 Judge 给 val 三元组打完整奖励分（候选 vs 人工）"
    )
    ap.add_argument("--config", default="configs/grpo_fact_coverage_t4.yaml",
                    help="读 rl.reward（奖励口径）和 semantic（裁判配置）")
    ap.add_argument("--input", nargs="+", required=True,
                    help="三元组 jsonl（id/source/reference/output），可给多个")
    ap.add_argument("--out-dir", default="data/judge")
    ap.add_argument("--candidate-key", default="output", help="候选摘要字段名")
    ap.add_argument("--reference-key", default="reference", help="人工摘要字段名")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true",
                    help="默认跳过输出里已有的 id（可断点续跑）")
    ap.add_argument("--chunk-size", type=int, default=16,
                    help="一批几条记录。显存紧就调小，吞吐优先就调大")

    # ---- 裁判模型 / vLLM ----
    ap.add_argument("--model", default=None,
                    help="裁判模型目录或 HF 名，默认取 semantic.model")
    ap.add_argument("--tokenizer", default=None,
                    help="单独指定 tokenizer（AWQ 目录缺文件时用）")
    ap.add_argument("--quantization", default="none",
                    help="none | bitsandbytes | awq | gptq。T4 上 7B 装不下 fp16，"
                         "必须显式指定一个 4-bit 方案（bitsandbytes 可对现有 fp16 "
                         "权重现量化，awq/gptq 则要换成量化好的 checkpoint）")
    ap.add_argument("--dtype", default="float16",
                    help="T4 必须 float16；4090/A100 可用 bfloat16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--swap-space", type=int, default=4)
    ap.add_argument("--max-input-tokens", type=int, default=4096,
                    help="prompt 截断长度（和 Judge 的 max_input_tokens 一致）")
    ap.add_argument("--extract-max-new-tokens", type=int, default=None,
                    help="六要素抽取的生成预算，默认取 semantic.extract_max_new_tokens")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--logprobs", type=int, default=50,
                    help="受限解码时取多少个 logprob，够覆盖 0~4 即可")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
    logger.info("裁判任务：%s", sorted(spec.required_signals) or ["(无)"])

    from sfzy.judge.judge import FactConsistencyJudge

    runtime, _tokenizer = build_vllm(args, model_name, trust_remote_code, max_model_len)
    options = spec.judge_term_options()
    judge = FactConsistencyJudge(
        runtime=runtime,
        weights=(options.get("fact_consistency") or {}).get("element_weights"),
        coverage_weights=(options.get("element_coverage") or {}).get("element_weights"),
        tasks=sorted(spec.required_signals) or ["fact_consistency"],
        max_input_tokens=args.max_input_tokens,
        extract_max_new_tokens=extract_max_new_tokens,
        min_document_elements=min_document_elements,
        doc_fallback=bool(semantic.get("doc_fallback", True)),
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
        for path, stats in zip(args.input, all_stats):
            delta = stats["delta_reward(human-candidate)"]
            win_rate = stats["human_win_rate"]
            delta_s = f"{delta:+.4f}" if delta is not None else "-"
            rate_s = f"{win_rate:.1%}" if win_rate is not None else "-"
            print(f"  {Path(path).name:<34} Δ={delta_s:>10}  胜率={rate_s:>8}")


if __name__ == "__main__":
    main()
