"""对若干个 LoRA step 在**验证集**上推理，并算 ROUGE 出报告（vLLM 加速）。

要解决的问题
============
GRPO 每 ``save_every_n_steps`` 步存一个 ``step_*.pt``。训练结束后常见的需求是：

    把其中任意几个 step 挂上去，在 val 上跑一遍，
    得到 (id, 裁判文书原文, 人工摘要, 候选摘要) 的 jsonl，
    再算 ROUGE，横向比较哪一步最好。

本脚本把这三件事串成一条命令：

  1. **导出**：项目自己的 ``step_*.pt``（自定义键名 ``...query_key_value.lora_A.weight``）
     → PEFT 目录（``adapter_config.json`` + ``adapter_model.safetensors``），
     实现在 ``sfzy.models.lora_export``，和 GRPO 训练时的热交换共用一份代码；
  2. **推理**：vLLM 连续批处理；**底座只加载一次**，逐个 adapter 生成
     （``LoRARequest`` 热挂载），不比"每个 step 各起一次进程"慢；
  3. **评测**：``sfzy.eval.rouge.score_corpus``（CAIL2020 官方口径，宏平均），
     写 ``rouge.json`` + ``summary.json`` + 人读的 ``report.md``。

产物（都在 ``--out-dir`` 下）
============================
    <out-dir>/
      step_001500.jsonl      每个 step（+ baseline）一份，含 id/source/reference/output
      lora_peft/<step>/      导出的 PEFT adapter（可复用，不用重复导出）
      rouge.json             逐 step 的完整 ROUGE 指标（机器可读）
      summary.json           本次运行的全部信息（配置 / 命令 / 指标 / 长度）
      report.md              横向对比表 + 最优 step + 说明

与训练口径对齐（重要）
====================
推理路径必须和训练时看到的一致，否则 ROUGE 不可比：

  * prompt 用 ``sfzy.data.prompts.build_messages(source, style)``，``style``
    取配置里的 ``data.prompt_style``（结构化提示）；
  * 编码用 ``sfzy.models.chat_template.encode_prompt(add_generation_prompt=True)``；
  * prompt 超预算时**保留尾部**（判决书的"本院认为/判决如下"在结尾）；
  * 贪心解码 + ``repetition_penalty``，与 SFT/GRPO 的生成默认值一致；
  * LoRA 的 ``alpha`` 取配置文件（rank 由权重形状反推）—— ``alpha/r`` 决定
    LoRA 强度，配错会静默改变结果。

底座必须和 checkpoint 匹配
==========================
``--config`` 同时提供底座模型和 LoRA 的 ``r``/``alpha``。如果配错了（例如拿
ChatGLM3 的配置去评 Qwen 的 checkpoint），**目标层名和 rank 会自动从 checkpoint
推断并告警**，但 ``alpha`` 无法从权重反推 —— 它只信配置。所以换 run 时务必确认
``--config`` 是那次训练用的那一份。

目标是层名也可以显式覆盖：``--target-modules q_proj k_proj v_proj o_proj``。

显存与速度
==========
* 底座只加载一次，逐个 adapter 生成；``--max-loras 1``（默认）最省显存，
  vLLM 会在切换 adapter 时换入换出。显存够可以设成 adapter 个数以省掉换页；
* T4 上 6B 必须 4-bit：``--dtype float16 --quantization bitsandbytes``。

用法
====
    # 几个 step 横向比（外加一次无 adapter 的底座作为 baseline）
    python scripts/eval_lora_steps_vllm.py \
        --config configs/grpo_fact_cov_ee_t4.yaml \
        --adapters outputs/exp/e5_full/step_001500.pt outputs/exp/e5_full/step_001511.pt \
        --out-dir outputs/eval_steps/e5_full --baseline

    # 通配符也行（按 step 号排序）
    python scripts/eval_lora_steps_vllm.py --config ... \
        --adapters 'outputs/exp/e5_full/step_*.pt' \
        --out-dir outputs/eval_steps/e5_full

    # 只导 adapter（不需要 GPU），先验证格式
    python scripts/eval_lora_steps_vllm.py --config ... \
        --adapters outputs/exp/e5_full/step_001511.pt --export-only

    # 只重算 ROUGE / 重出报告（不加载模型）
    python scripts/eval_lora_steps_vllm.py --config ... \
        --out-dir outputs/eval_steps/e5_full --rouge-only

    # 小样本冒烟
    python scripts/eval_lora_steps_vllm.py --config ... --adapters ... \
        --out-dir tmp/eval_smoke --limit 4 --max-new-tokens 32 --enforce-eager

环境
====
vLLM 与 transformers 版本强绑定（见 ``requirements-cloud.txt``）。本脚本刻意
只依赖纯 Python 的 prompt 拼接 + tokenizer + vLLM，不 import 项目的模型加载代码，
所以能在干净的 vLLM 环境里跑。``--rouge-only`` / ``--export-only`` 两个模式
不需要 vLLM。
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                        # noqa: E402
from sfzy.data.prompts import build_messages               # noqa: E402
from sfzy.eval.rouge import score_corpus                   # noqa: E402
from sfzy.models.chat_template import encode_prompt        # noqa: E402
from sfzy.utils.io import ensure_dir, read_jsonl           # noqa: E402
from sfzy.utils.logging import get_logger                  # noqa: E402

logger = get_logger("eval_lora_steps")

_STEP_RE = re.compile(r"step_(\d+)")


def resolve(path: Any) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def is_local_dir(path: Path) -> bool:
    """权限不足（配置里写的是别的机器的路径）时按"不是本地路径"处理。"""
    try:
        return path.is_dir()
    except OSError:
        return False


def load_done_ids(path: Path) -> set:
    """已完成（已写进结果 jsonl）的 id，用于断点续跑。"""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ---------------------------------------------------------------------------
# 输入数据 / adapter 解析
# ---------------------------------------------------------------------------
def resolve_split_file(data_cfg: Any, split: str, data_dir: Optional[str] = None) -> Path:
    """找 split 对应的 jsonl（和 generate_triples*.py 同一套候选顺序）。"""
    if data_dir:
        candidates = [resolve(data_dir) / f"{split}.jsonl"]
    else:
        processed = resolve(data_cfg.path_("data.processed_dir", "data/processed"))
        candidates = [processed / f"{split}.jsonl"]
        if split == "val":
            candidates.append(processed / data_cfg.path_("data.dev_file", "dev.jsonl"))
        if split == "train":
            candidates.append(processed / data_cfg.path_("data.train_file", "train.jsonl"))
        candidates.append(resolve("data/splits") / f"{split}.jsonl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"找不到 split={split} 的样本文件，试过：\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def resolve_model_cfg(cfg: Any) -> Dict[str, Any]:
    """取底座模型配置，兼容两种布局。

    项目里有两套写法：``model: {model_name_or_path: ...}`` 直接内联（GRPO 的
    自包含配置），以及 ``model_config: configs/model_xxx.yaml`` 指向另一份文件。
    两种都要认，内联的优先（子配置覆盖父配置的语义）。
    """
    from sfzy.config import deep_merge

    merged: Dict[str, Any] = {}
    ref = cfg.get("model_config")
    if ref:
        merged = dict(load_config(resolve(ref)).get("model") or {})
    return deep_merge(merged, dict(cfg.path_("model") or {}))


def resolve_data_cfg(cfg: Any) -> Any:
    """取数据配置；没有 ``data_config`` 时退回配置自身（同样的键都在顶层）。"""
    ref = cfg.get("data_config")
    return load_config(resolve(ref)) if ref else cfg


def step_of(name: str) -> Optional[int]:
    """从文件名里取 step 号（``step_001511`` → 1511）；取不到返回 None。"""
    match = _STEP_RE.search(name)
    return int(match.group(1)) if match else None


def expand_adapters(patterns: Sequence[str]) -> List[Path]:
    """展开通配符并去重；按 (step 号, 名字) 排序，没有 step 号的排最后。"""
    seen: Dict[str, Path] = {}
    for pattern in patterns:
        has_glob = any(ch in pattern for ch in "*?[")
        matches = [Path(p) for p in glob.glob(pattern)] if has_glob else [Path(pattern)]
        if not matches:
            raise FileNotFoundError(f"--adapters 里这一项没匹配到任何文件：{pattern}")
        for path in matches:
            seen[str(path)] = path
    paths = [resolve(p) for p in seen.values()]
    paths.sort(key=lambda p: (step_of(p.stem) is None, step_of(p.stem) or 0, p.name))
    return paths


def run_name_for(adapter: Path, used: Dict[str, int]) -> str:
    """adapter → 输出文件名（不含扩展名）。同名的加父目录前缀区分。"""
    name = adapter.stem
    if name in used:
        name = f"{adapter.parent.name}__{adapter.stem}"
    used[name] = used.get(name, 0) + 1
    return name


def probe_adapter(adapter: Path) -> Dict[str, Any]:
    """从 checkpoint 的键名/形状里读出 ``{"modules": [...], "rank": int|None}``。

    **比配置可靠**：配置里的 ``lora.target_modules`` 可能属于另一个底座
    （ChatGLM3 是 query_key_value/dense*，Qwen 是 q_proj/k_proj/v_proj/o_proj），
    而 adapter 自己的键名一定和它训练时的底座一致。导出时用错目标层，
    vLLM 要么加载失败、要么静默忽略掉这些层（"挂了 adapter 但效果没变"）。

    rank 同理：从 ``lora_A`` 的形状 (r, in_features) 反推，用来交叉检查配置是不是
    这次 run 用的那一份（``alpha`` 无法从权重反推，只能信配置）。
    """
    import torch

    from sfzy.models.lora_export import extract_lora_state

    payload = torch.load(adapter, map_location="cpu", weights_only=False)
    state = extract_lora_state(payload)
    modules: List[str] = []
    ranks: set = set()
    for key in state:
        name = str(key)
        if "lora_A" not in name and "lora_B" not in name:
            continue
        parts = name.split(".")
        if len(parts) >= 3 and parts[-3] not in modules:
            modules.append(parts[-3])
        if name.endswith("lora_A.weight"):
            try:
                ranks.add(int(state[key].shape[0]))
            except (AttributeError, IndexError, ValueError):
                pass
    return {"modules": modules, "rank": (ranks.pop() if len(ranks) == 1 else None)}


def export_adapters(adapters: Sequence[Path], out_dir: Path, *, alpha: int,
                    dropout: float, explicit_modules: Sequence[str],
                    config_modules: Sequence[str], config_rank: int, base_model: str,
                    force: bool) -> List[Dict[str, Any]]:
    """把每个 ``.pt`` 导出成 PEFT 目录；已导出过就直接复用。

    目标层的优先级：**命令行 ``--target-modules`` > checkpoint 自己推断 > 配置**。
    配置里的那份可能属于另一个底座（Qwen 的 q_proj vs ChatGLM3 的 query_key_value），
    而 checkpoint 的键名一定是训练时那一套。
    """
    from sfzy.models.lora_export import convert_checkpoint_to_peft

    runs: List[Dict[str, Any]] = []
    used: Dict[str, int] = {}
    for adapter in adapters:
        if not adapter.exists():
            raise FileNotFoundError(f"adapter 不存在：{adapter}")
        name = run_name_for(adapter, used)
        peft_dir = out_dir / "lora_peft" / name
        if (peft_dir / "adapter_config.json").exists() and not force:
            logger.info("复用已导出的 PEFT adapter：%s", peft_dir)
            modules = list(explicit_modules or config_modules)
            rank = adapter_rank(peft_dir, config_rank)
        else:
            probe = probe_adapter(adapter)
            detected, rank = probe["modules"], probe["rank"]
            modules = list(explicit_modules or detected or config_modules)
            if explicit_modules:
                rank = rank or config_rank
            # 只在"两套层名完全不相交"时告警：那基本就是底座架构不同
            # （Qwen 的 q_proj/k_proj/v_proj/o_proj vs ChatGLM3 的
            # query_key_value/dense*）。只导了部分层不算异常，不刷日志。
            if (detected and config_modules and not explicit_modules
                    and not (set(detected) & set(config_modules))):
                logger.warning(
                    "%s 里的 LoRA 目标层 %s 与配置里的 %s 完全不同 —— 按 checkpoint "
                    "推断结果导出。请确认 --config 里的底座模型与这份 checkpoint 匹配",
                    adapter.name, detected, list(config_modules))
            if rank is not None and rank != config_rank:
                logger.warning(
                    "%s 的实际 LoRA rank=%d，但配置 lora.r=%d —— 说明这份配置"
                    "很可能不是该 checkpoint 用的那份；alpha 是照配置取的值，"
                    "alpha/r 配错会静默改变 LoRA 强度，建议显式确认 alpha",
                    adapter.name, rank, config_rank)
            if not modules:
                raise SystemExit(
                    f"{adapter} 里推不出 LoRA 目标层，且配置里也没写 "
                    f"lora.target_modules —— 请用 --target-modules 显式指定")
            convert_checkpoint_to_peft(
                adapter, peft_dir, alpha=alpha, dropout=dropout,
                target_modules=modules, base_model=base_model,
            )
        runs.append({"name": name, "adapter": str(adapter), "step": step_of(adapter.stem),
                     "peft_dir": str(peft_dir), "baseline": False,
                     "target_modules": list(modules)})
    return runs


def adapter_rank(peft_dir: Path, fallback: int) -> int:
    try:
        cfg = json.loads((peft_dir / "adapter_config.json").read_text(encoding="utf-8"))
        return int(cfg.get("r", fallback))
    except (OSError, ValueError, KeyError):
        return fallback


# ---------------------------------------------------------------------------
# prompt 编码（与训练/其它生成脚本同一口径）
# ---------------------------------------------------------------------------
def encode_prompts(records: Sequence[Dict[str, Any]], tokenizer: Any,
                   prompt_style: str, max_prompt_len: int) -> Iterator[List[int]]:
    """逐条编码 prompt，**保留尾部**截断（判决书的结论在结尾）。"""
    for record in records:
        messages = build_messages(record["source"], prompt_style)
        ids = encode_prompt(tokenizer, messages, add_generation_prompt=True)
        if max_prompt_len and len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
        yield ids


def make_lora_request(peft_dir: Path, lora_id: int) -> Any:
    """构造 ``LoRARequest``，兼容 0.5.x 的 ``lora_local_path`` 与 0.6+ 的 ``lora_path``。"""
    from vllm.lora.request import LoRARequest

    try:
        return LoRARequest(peft_dir.name, lora_id, lora_path=str(peft_dir))
    except TypeError:
        return LoRARequest(peft_dir.name, lora_id, lora_local_path=str(peft_dir))


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
def build_llm(args: argparse.Namespace, model_name: str, model_cfg: Dict[str, Any],
              max_lora_rank: int, n_loras: int, max_prompt_len: int) -> Any:
    from vllm import LLM

    max_model_len = int(args.max_model_len or (max_prompt_len + args.max_new_tokens))
    llm_kwargs: Dict[str, Any] = dict(
        model=model_name,
        tokenizer=args.tokenizer or model_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code")),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        swap_space=args.swap_space,
        disable_log_stats=True,
        seed=args.seed,
    )
    if n_loras:
        llm_kwargs.update(
            enable_lora=True,
            max_lora_rank=int(max_lora_rank),
            max_loras=int(args.max_loras),
            max_cpu_loras=max(int(args.max_loras), n_loras),
        )
    dtype = args.dtype
    if dtype == "auto":
        cfg_dtype = model_cfg.get("torch_dtype")
        dtype = cfg_dtype if isinstance(cfg_dtype, str) and cfg_dtype else "auto"
    if args.quantization and args.quantization != "none":
        llm_kwargs["quantization"] = args.quantization
        if args.quantization == "bitsandbytes":
            # vLLM 的硬校验：bnb 量化必须搭配 load_format=bitsandbytes
            llm_kwargs["load_format"] = "bitsandbytes"
    if args.load_format and args.load_format != "auto":
        llm_kwargs["load_format"] = args.load_format
    if dtype and dtype != "auto":
        llm_kwargs["dtype"] = dtype

    from sfzy.utils.vllm_compat import filter_llm_kwargs

    llm_kwargs, dropped = filter_llm_kwargs(llm_kwargs)
    if dropped:
        logger.warning("当前 vLLM 不认识这些参数，已忽略：%s", dropped)
    logger.info(
        "加载 vLLM：%s（dtype=%s, tp=%d, max_model_len=%d, lora=%s, max_lora_rank=%d）",
        model_name, dtype, args.tensor_parallel_size, max_model_len,
        f"on(max_loras={args.max_loras})" if n_loras else "off", max_lora_rank,
    )
    return LLM(**llm_kwargs)


def generate_run(llm: Any, tokenizer: Any, records: Sequence[Dict[str, Any]],
                 args: argparse.Namespace, *, out_path: Path, prompt_style: str,
                 max_prompt_len: int, lora_request: Any) -> int:
    """跑一个 adapter（或 baseline）→ ``out_path``。返回本次新写入的条数。"""
    done = load_done_ids(out_path)
    todo = [r for r in records if str(r["id"]) not in done]
    if not todo:
        logger.info("已全部完成，跳过：%s（%d 条）", out_path.name, len(records))
        return 0
    logger.info("%s：共 %d 条，已完成 %d 条，本次生成 %d 条",
                out_path.name, len(records), len(done), len(todo))

    from vllm import SamplingParams

    sampling = SamplingParams(
        n=args.num,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )
    started = time.time()
    written = 0
    lengths: List[int] = []
    with open(out_path, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            prompt_ids = list(encode_prompts(chunk, tokenizer, prompt_style, max_prompt_len))
            outputs = llm.generate(
                prompts=[{"prompt_token_ids": ids} for ids in prompt_ids],
                sampling_params=sampling,
                lora_request=lora_request,
                use_tqdm=False,
            )
            for record, output in zip(chunk, outputs):
                candidates = [o.text.strip() for o in output.outputs]
                entry = {
                    "id": record["id"],
                    "source": record["source"],
                    "reference": record.get("summary", ""),
                    "output": candidates[0] if candidates else "",
                }
                if args.num > 1:
                    entry["outputs"] = candidates
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                lengths.extend(len(c) for c in candidates)
                written += 1
            f.flush()
            elapsed = time.time() - started
            rate = written / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - written) / rate / 60 if rate > 0 else float("inf")
            logger.info("  %s 进度 %d/%d | %.2f 条/秒 | 预计剩余 %.1f 分钟",
                        out_path.stem, written, len(todo), rate, eta)
    if lengths:
        logger.info("  %s 输出字数：平均 %.0f / 最短 %d / 最长 %d",
                    out_path.stem, sum(lengths) / len(lengths), min(lengths), max(lengths))
    return written


# ---------------------------------------------------------------------------
# ROUGE 与报告
# ---------------------------------------------------------------------------
def discover_run_files(out_dir: Path, extra: Sequence[str] = ()) -> List[Path]:
    """找出待评测的 jsonl：默认 ``<out-dir>/*.jsonl``，可用 ``--runs`` 显式指定。"""
    if extra:
        files = [resolve(p) for p in extra]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError("找不到这些结果文件：" + ", ".join(missing))
    else:
        files = [p for p in out_dir.glob("*.jsonl") if p.exists()]

    def key(p: Path):
        # baseline 排最前（报告里它的 Δ 那一列才有意义），其余按 step 号升序，
        # 没有 step 号的名字排最后。
        if p.stem == "baseline":
            return (-1, 0, p.name)
        step = step_of(p.stem)
        return (0, step or 0, p.name) if step is not None else (1, 0, p.name)

    return sorted(files, key=key)


def compute_rouge(run_files: Sequence[Path], references: Dict[str, str],
                  mode: str) -> List[Dict[str, Any]]:
    """逐文件算 ROUGE（逐样本宏平均，只统计两边都有的 id）。"""
    results: List[Dict[str, Any]] = []
    for path in run_files:
        preds: Dict[str, str] = {}
        lengths: List[int] = []
        for record in read_jsonl(str(path)):
            rid = str(record.get("id"))
            text = str(record.get("output") or "").strip()
            preds[rid] = text
            lengths.append(len(text))
        metrics = score_corpus(preds, references, mode=mode)
        results.append({
            "name": path.stem,
            "step": step_of(path.stem),
            "file": str(path),
            "n_pred": len(preds),
            "n_ref": len(references),
            "n_scored": int(metrics.get("num_samples", 0)),
            "mean_chars": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "empty_ratio": (sum(1 for v in preds.values() if not v) / len(preds))
            if preds else 0.0,
            **{k: float(v) for k, v in metrics.items() if k != "num_samples"},
        })
    return results


def render_report(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 4) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"

    baseline = next((r for r in rows if r["name"] == "baseline"), None)
    lines: List[str] = ["# LoRA step 验证集推理 + ROUGE 报告\n"]
    lines.append(f"- 生成时间：{meta.get('created_at', '?')}")
    lines.append(f"- 配置：`{meta.get('config', '?')}`")
    lines.append(f"- 底座：`{meta.get('model', '?')}`"
                 f"（dtype={meta.get('dtype')}, quantization={meta.get('quantization')}）")
    lines.append(f"- 数据：`{meta.get('input', '?')}`，共 {meta.get('n_records', '?')} 条"
                 f"（prompt_style={meta.get('prompt_style')}，"
                 f"max_new_tokens={meta.get('max_new_tokens')}）")
    lines.append(f"- 分词口径：`{meta.get('rouge_mode')}`；打分方式：逐样本宏平均")
    lines.append(f"- 输出目录：`{meta.get('out_dir', '?')}`\n")

    header = ["step", "条数", "ROUGE-1 F", "ROUGE-2 F", "ROUGE-L F", "官方总分",
              "平均字数", "空输出率"]
    if baseline is not None:
        header += ["Δ vs baseline", "Δ vs 前一行"]
    lines.append("## 结果总表\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    prev: Optional[Dict[str, Any]] = None
    for row in rows:
        cells = [
            str(row["step"]) if row["step"] is not None else row["name"],
            str(row["n_scored"]),
            fmt(row.get("rouge-1-f")), fmt(row.get("rouge-2-f")),
            fmt(row.get("rouge-l-f")), fmt(row.get("overall")),
            f"{row['mean_chars']:.0f}", f"{row['empty_ratio']:.1%}",
        ]
        if baseline is not None:
            is_baseline = row is baseline
            cells.append("-" if is_baseline else
                         fmt(row.get("overall", 0.0) - baseline.get("overall", 0.0)))
            cells.append(fmt(row.get("overall", 0.0) - prev.get("overall", 0.0))
                         if (prev is not None and not is_baseline) else "-")
        lines.append("| " + " | ".join(cells) + " |")
        prev = row
    lines.append("")

    scored = [r for r in rows if r["n_scored"]]
    if scored:
        best = max(scored, key=lambda r: r.get("overall", 0.0))
        lines.append("## 结论\n")
        lines.append(
            f"- 官方总分最高：**{best['step'] if best['step'] is not None else best['name']}**"
            f"（overall {best.get('overall', 0.0):.4f}，"
            f"ROUGE-L F {best.get('rouge-l-f', 0.0):.4f}）")
        if (baseline is not None and baseline.get("n_scored")
                and best is not baseline):
            delta = best.get("overall", 0.0) - baseline.get("overall", 0.0)
            lines.append(f"- 相对底座 baseline：{delta:+.4f}"
                         f"（{'提升' if delta > 0 else '下降或持平'}）")
        elif best is baseline:
            lines.append("- **没有任何 step 超过底座 baseline**："
                         "微调这一步的收益可能被采样噪声淹没，先扩大样本量再判断。")
        lines.append("- 关注点：若各 step 之间差异小于 0.005，多半是生成噪声，"
                     "建议加大样本量或换更稳的解码参数后再下结论。")
        lines.append("")
    lines.append("## 说明\n")
    lines.append("- 每份 jsonl 的字段：`id` / `source`（判决书原文）/ `reference`（人工摘要）"
                 "/ `output`（该 step 生成的候选摘要）；`--num>1` 时另有 `outputs` 列表。")
    lines.append("- ROUGE 只统计该文件与参考文件**共有的 id**，逐样本先算分再取宏平均"
                 "（CAIL2020 官方口径：0.2*R1 + 0.4*R2 + 0.4*RL）。")
    lines.append("- `条数`（n_scored）明显小于参考条数时说明该 step 没跑完，"
                 "重跑同一条命令会断点续跑补齐。")
    if baseline is not None:
        lines.append("- `baseline` 是不挂 adapter 的底座，作为参照；"
                     "它的 Δ 两列恒为 `-`。")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, rows: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    (out_dir / "rouge.json").write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "runs": list(rows)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(rows, meta), encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="对若干个 LoRA step 在验证集上推理并算 ROUGE（vLLM）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", default="configs/grpo_fact_cov_ee_t4.yaml")
    ap.add_argument("--adapters", nargs="+", default=[],
                    help="要评测的 LoRA checkpoint（.pt，本项目格式），可多个/可通配符")
    ap.add_argument("--baseline", action="store_true",
                    help="额外跑一次**不挂 adapter** 的底座作为对照（报告里显示 Δ）")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录（结果 jsonl / adapter / 报告都在这里）。"
                         "默认 outputs/eval_steps/<第一个 adapter 的父目录名>")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--input", default=None,
                    help="直接指定输入 jsonl（含 id/source/summary），覆盖 --split")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--force-convert", action="store_true",
                    help="即使 PEFT 目录已存在也重新导出 adapter")
    ap.add_argument("--export-only", action="store_true", help="只导出 adapter 然后退出")
    ap.add_argument("--rouge-only", action="store_true",
                    help="不推理：只对 --out-dir 里已有的 jsonl（或 --runs）算 ROUGE 出报告")
    ap.add_argument("--runs", nargs="+", default=[],
                    help="--rouge-only 时要评测的结果 jsonl（默认扫 --out-dir/*.jsonl）")
    ap.add_argument("--model", default=None, help="覆盖配置里的底座模型路径")
    ap.add_argument("--tokenizer", default=None, help="单独指定 tokenizer")
    ap.add_argument("--prompt-style", default=None)
    ap.add_argument("--target-modules", nargs="*", default=None,
                    help="LoRA 目标层名，显式覆盖自动推断。不传时按 checkpoint 的键名"
                         "推断（比配置里的 lora.target_modules 可靠）")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--num", type=int, default=1,
                    help="每条生成几个候选；1（默认）走贪心，产物与训练下游兼容")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-prompt-len", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-loras", type=int, default=1,
                    help="同时驻留 GPU 的 adapter 数。默认 1（逐个 step 生成，显存最省）；"
                         "调大可以省掉 adapter 换入换出的开销")
    ap.add_argument("--dtype", default="float16", help="float16 | bfloat16 | auto")
    ap.add_argument("--quantization", default="none",
                    help="none | bitsandbytes | awq | gptq（T4 上 7B 必须 4-bit）")
    ap.add_argument("--load-format", default="auto")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--swap-space", type=int, default=4)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rouge-mode", default="jieba", choices=["jieba", "char"],
                    help="ROUGE 分词口径；项目默认 jieba（与奖励/评测口径一致）")
    args = ap.parse_args(argv)

    if args.temperature is None:
        args.temperature = 0.0 if args.num == 1 else 0.7
    if args.num < 1:
        ap.error(f"--num 至少为 1，收到 {args.num}")
    if args.num > 1 and args.temperature <= 0:
        ap.error(f"--num={args.num}>1 需要采样：--temperature 必须 >0")
    if not (0.0 < args.top_p <= 1.0):
        ap.error(f"--top-p 要在 (0,1] 内，收到 {args.top_p}")
    if not args.rouge_only and not args.adapters and not args.baseline:
        ap.error("要么给 --adapters，要么给 --baseline（只跑底座）；"
                 "只想重算 ROUGE 就用 --rouge-only")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config(resolve(args.config))
    model_cfg = resolve_model_cfg(cfg)
    data_cfg = resolve_data_cfg(cfg)

    model_name = args.model or model_cfg.get("model_name_or_path", "")
    local = resolve(model_name)
    if is_local_dir(local):
        model_name = str(local)
    if not model_name:
        raise SystemExit("配置里没有 model.model_name_or_path，也没有传 --model")

    adapters = expand_adapters(args.adapters) if args.adapters else []
    if args.out_dir:
        out_dir = resolve(args.out_dir)
    else:
        parent = adapters[0].parent.name if adapters else "baseline"
        out_dir = resolve(f"outputs/eval_steps/{parent}")
    ensure_dir(out_dir)
    logger.info("输出目录：%s", out_dir)

    lora_alpha = int(cfg.path_("lora.alpha", 16))
    lora_r = int(cfg.path_("lora.r", 8))
    lora_dropout = float(cfg.path_("lora.dropout", 0.0))
    config_modules = list(cfg.path_("lora.target_modules", []) or [])
    explicit_modules = list(args.target_modules or [])
    logger.info("LoRA：r=%d alpha=%d dropout=%s 配置里的 target_modules=%s",
                lora_r, lora_alpha, lora_dropout, config_modules or "（空）")

    # ---- LoRA 导出（不需要 GPU；--export-only 到这里就结束）----
    runs: List[Dict[str, Any]] = []
    if adapters:
        runs = export_adapters(
            adapters, out_dir, alpha=lora_alpha, dropout=lora_dropout,
            explicit_modules=explicit_modules, config_modules=config_modules,
            config_rank=lora_r, base_model=model_name,
            force=args.force_convert,
        )
    if args.baseline:
        runs = [{"name": "baseline", "adapter": None, "step": None,
                 "peft_dir": None, "baseline": True}] + runs
    if args.export_only:
        logger.info("--export-only：已导出 %d 个 adapter，退出", len(runs))
        for run in runs:
            logger.info("  %s → %s", run["name"], run["peft_dir"])
        return

    # ---- 数据（ROUGE 也要参考摘要，所以 --rouge-only 时同样要读）----
    data_path = resolve(args.input) if args.input else resolve_split_file(
        data_cfg, args.split, args.data_dir)
    records = list(read_jsonl(str(data_path)))
    if args.limit:
        records = records[: args.limit]
    references = {str(r["id"]): str(r.get("summary", "")) for r in records}
    logger.info("数据：%s（%d 条）", data_path, len(records))

    prompt_style = args.prompt_style or data_cfg.path_("data.prompt_style", "structured")
    max_prompt_len = int(
        args.max_prompt_len
        or cfg.path_("sft.max_length")
        or cfg.path_("rl.max_length")
        or model_cfg.get("max_length")
        or 2048
    )

    meta: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(resolve(args.config)),
        "model": model_name,
        "input": str(data_path),
        "n_records": len(records),
        "prompt_style": prompt_style,
        "max_prompt_len": max_prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "num": args.num,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "rouge_mode": args.rouge_mode,
        "out_dir": str(out_dir),
        "lora": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout,
                 "target_modules": [r.get("target_modules") for r in runs] or None,
                 "config_target_modules": config_modules},
        "runs": [{k: v for k, v in run.items()} for run in runs],
    }

    # ---- 推理 ----
    if not args.rouge_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or model_name,
            trust_remote_code=bool(model_cfg.get("trust_remote_code")),
            padding_side="left",
        )
        max_lora_rank = max(
            [adapter_rank(Path(r["peft_dir"]), lora_r) for r in runs if r["peft_dir"]]
            or [lora_r]
        )
        n_loras = sum(1 for r in runs if r["peft_dir"])
        llm = build_llm(args, model_name, model_cfg, max_lora_rank, n_loras,
                        max_prompt_len)

        started = time.time()
        for index, run in enumerate(runs, start=1):
            lora_request = (
                make_lora_request(Path(run["peft_dir"]), index)
                if run["peft_dir"] else None
            )
            out_path = out_dir / f"{run['name']}.jsonl"
            generate_run(
                llm, tokenizer, records, args, out_path=out_path,
                prompt_style=prompt_style, max_prompt_len=max_prompt_len,
                lora_request=lora_request,
            )
            run["file"] = str(out_path)
        logger.info("全部生成完成，耗时 %.1f 分钟", (time.time() - started) / 60)

    # ---- ROUGE ----
    run_files = discover_run_files(
        out_dir, args.runs or [r["file"] for r in runs if r.get("file")])
    if not run_files:
        raise SystemExit(f"{out_dir} 下没有结果 jsonl，无法算 ROUGE")
    rows = compute_rouge(run_files, references, args.rouge_mode)
    meta["scored_files"] = [str(p) for p in run_files]
    write_outputs(out_dir, rows, meta)

    logger.info("ROUGE：%s\n汇总：%s\n报告：%s",
                out_dir / "rouge.json", out_dir / "summary.json", out_dir / "report.md")
    for row in rows:
        logger.info("  %-16s overall=%.4f  R1=%.4f R2=%.4f RL=%.4f  n=%d",
                    row["name"], row.get("overall", 0.0), row.get("rouge-1-f", 0.0),
                    row.get("rouge-2-f", 0.0), row.get("rouge-l-f", 0.0), row["n_scored"])


if __name__ == "__main__":
    main()
