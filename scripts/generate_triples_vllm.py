"""用 vLLM 批量生成三元组：(原文, 参考摘要, 模型输出)。

和 ``scripts/generate_triples.py`` 的关系
========================================
后者走 HuggingFace 的生成循环（``transformers.generate``，ChatGLM3 则借用它
自带的 ``stream_generate``），一次只跑一个 batch，6B 模型生成全量 train
（10738 条）要排队很久。这个脚本把**同一件事**换到 vLLM 上做：连续批处理
（continuous batching）+ PagedAttention，把 GPU 利用率吃满。

两条路径**在语义上必须对齐**，否则 vLLM 生成的 train 三元组和之前 HF 生成的
val/test 三元组不能混着用：

  1. prompt 用同一个函数构造（``data/prompts.build_messages``）；
  2. 编码用同一个函数（``models/chat_template.encode_prompt``，
     ``add_generation_prompt=True``）；
  3. 截断规则一致：prompt 超预算就**保留尾部**（判决书的"本院认为"
     "判决如下"在结尾，正是摘要最需要的部分）；
  4. 贪心解码（``do_sample=False``）、``repetition_penalty=1.1`` ——
     和 ``sft/infer.generate_one`` 的默认值一致。

关键实现选择：**直接把 token id 喂给 vLLM，不喂字符串**。
vLLM 自己也会 tokenize，但 ChatGLM 的 ``apply_chat_template`` 由远程代码
实现，两条路径再各自 tokenize 一次，就是在赌它们逐 token 相同 ——
不一致不会报错，只会让结果悄悄偏掉。传 ``prompt_token_ids`` 把 tokenization
收敛到一处（我们的 encoder），vLLM 只负责前向。

LoRA adapter 的格式
===================
本项目训练出来的 checkpoint（``step_*.pt`` / ``best.pt``）是自定义格式
（``...query_key_value.lora_A.weight``），vLLM 不认识。所以这里先把 LoRA
导出成 PEFT 目录（``adapter_config.json`` + ``adapter_model.safetensors``），
再用 ``LoRARequest`` 挂上去。vLLM 的解析函数 ``parse_fine_tuned_lora_name``
要求键名带 ``base_model.model.`` 前缀，导出时会补上，并校验每个目标层名都在
模型声明的 ``supported_lora_modules`` 里（ChatGLM3 正好是配置里的四个：
query_key_value / dense / dense_h_to_4h / dense_4h_to_h）。

环境（重要）
============
vLLM 的版本和 transformers 强绑定，**不要装进 ChatGLM3 的那个环境里硬凑**：
新版 vLLM 要求 ``transformers>=5``，而 ChatGLM3 的远程代码在 5.x 上加载不了
（见 ``requirements-cloud.txt`` 的说明）。两个选择：

  * 单环境（推荐，SFT 环境直接复用）：``vllm==0.5.5``（要求
    ``transformers>=4.43.2``、``torch==2.4.0``）。把 transformers 从 4.40.2
    升到 4.43.2 后 ChatGLM3 依然能加载，vLLM 与项目代码可以共存；
  * 双环境：一个跑 SFT/HF 推理，另一个只装 vLLM。本脚本刻意**不 import**
    项目的模型加载代码（loader / lora / infer），只依赖纯 Python 的
    prompt 拼接、tokenizer 和 vLLM，所以能在一个干净的 vLLM 环境里跑。

用法
====
    # 冒烟：先跑 4 条确认接线（本地 0.5B 就能验）
    python scripts/generate_triples_vllm.py --config configs/sft_local.yaml \
        --adapter outputs/sft_local/best.pt --split val --limit 4 \
        --max-new-tokens 32 --enforce-eager

    # AutoDL 2×4090：把 SFT 权重挂上去，跑全量 train
    python scripts/generate_triples_vllm.py --config configs/sft_cloud.yaml \
        --adapter outputs/sft_chatglm3/step_000336.pt \
        --split train --batch-size 128

    # 只想给 GRPO 的 prompt 池生成 sft_output（1000 条）
    python scripts/generate_triples_vllm.py --config configs/sft_cloud.yaml \
        --adapter outputs/sft_chatglm3/step_000336.pt \
        --input data/splits/rl_prompts.jsonl

支持断点续跑（输出文件里已有的 id 会被跳过）、``--shard I/N`` 数据分片，
输出格式与 ``generate_triples.py`` 完全一致：
``{"id", "source", "reference", "output"}``。

一个 prompt 生成多个候选（``--num N``）
======================================
默认 ``--num 1``，退化回贪心解码，产物和以前逐字节同构。传 ``--num N>1`` 时
用**采样**给每篇文书生成 N 个候选摘要（``n=N`` 交给 vLLM），产物多一个
``"outputs"`` 列表字段：``{"id", "source", "reference", "output", "outputs"}``。
``"output"`` 仍是第一个候选，保证 select_rl_prompts / score_fact_judge /
bench_metrics 这些只读 ``output`` 的下游不用改。

    # 每篇文书 4 个候选（默认写 data/triples/sft_{split}_n4.jsonl，避免和
    # 单候选产物混在同一个文件里）
    python scripts/generate_triples_vllm.py --config configs/sft_cloud.yaml \
        --adapter outputs/sft_chatglm3/step_000336.pt \
        --split train --num 4 --temperature 0.7

多候选必须采样：``--num>1`` 时 ``--temperature`` 不能为 0（贪心 N 次只会拿到
N 份一样的文本）。不显式传 ``--temperature`` 时，``--num>1`` 自动取 0.7。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                        # noqa: E402
from sfzy.data.prompts import build_messages              # noqa: E402
from sfzy.models.chat_template import encode_prompt       # noqa: E402
from sfzy.utils.io import read_jsonl                      # noqa: E402
from sfzy.utils.logging import get_logger                 # noqa: E402

logger = get_logger("generate_triples_vllm")


def resolve(path: Any) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def is_local_dir(path: Path) -> bool:
    """安全地判断目录是否存在。

    ``Path.is_dir()`` 在权限不足（比如配置里写的是别的机器的
    ``/root/autodl-tmp/...``）时会抛 PermissionError，而不是返回 False。
    这里要的是"本机能不能直接当本地路径用"，权限问题按"不是本地路径"处理。
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def load_done_ids(path: Path) -> set:
    """读已完成的 id，用于断点续跑（和 HF 版本同一套语义）。"""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def resolve_split_file(data_cfg: Any, split: str, data_dir: Optional[str] = None) -> Path:
    """找 split 对应的 jsonl，逻辑与 generate_triples.py 保持一致。

    项目里有两套数据布局（Kaggle 的 data/splits、本地的 data/processed），
    按候选列表取第一个存在的，一个都没有就把试过的路径打出来。
    """
    if data_dir:
        candidates = [resolve(data_dir) / f"{split}.jsonl"]
    else:
        processed = resolve(data_cfg.path_("data.processed_dir", "data/processed"))
        candidates = [processed / f"{split}.jsonl"]
        if split == "val":
            candidates.append(processed / data_cfg.path_("data.dev_file", "dev.jsonl"))
        if split == "train":
            candidates.append(processed / data_cfg.path_("data.train_file", "train.jsonl"))
        # tools/make_splits.py 的产物，本地调试也常用
        candidates.append(resolve("data/splits") / f"{split}.jsonl")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"找不到 split={split} 的样本文件，试过：\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\n本地冒烟加 --data-dir data/processed，跑全量切分用 --data-dir data/splits"
    )


# ---------------------------------------------------------------------------
# LoRA：自定义 checkpoint → PEFT 目录
# ---------------------------------------------------------------------------
# 实现已搬到 ``src/sfzy/models/lora_export.py``：GRPO 训练每步都要把当前 LoRA 导
# 出去做热交换，必须和这里共用同一份实现，否则两边会各自漂移（键名前缀、rank 推断、
# 目标层白名单这几处最容易分叉）。
from sfzy.models.lora_export import (  # noqa: E402
    CHATGLM3_TARGET_MODULES as _CHATGLM3_TARGET_MODULES,
    convert_checkpoint_to_peft as _convert_checkpoint_to_peft,
    extract_lora_state as _extract_lora_state,
)


def convert_checkpoint_to_peft(adapter_path, out_dir, *, r: int, alpha: int,
                               dropout: float, target_modules, base_model) -> Path:
    """兼容旧签名（`r=`）的转发 —— 实现在 `sfzy.models.lora_export`。"""
    return _convert_checkpoint_to_peft(
        adapter_path, out_dir, alpha=alpha, dropout=dropout,
        target_modules=target_modules or _CHATGLM3_TARGET_MODULES,
        base_model=base_model, rank=r,
    )



def ensure_peft_adapter(args: argparse.Namespace, cfg: Any, base_model: str) -> Optional[Path]:
    """把 ``--adapter``（.pt）或 ``--lora-dir``（已经是 PEFT 目录）统一成目录。"""
    if args.lora_dir:
        path = resolve(args.lora_dir)
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"--lora-dir 指向的不是 PEFT 目录（缺 adapter_config.json）：{path}"
            )
        return path
    if not args.adapter:
        return None

    adapter = resolve(args.adapter)
    if adapter.is_dir():
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"--adapter 是目录但不是 PEFT 目录：{adapter}")
        return adapter

    # 已经导出过就直接复用（不同 --out 反复跑时省一次转换）
    out_dir = resolve(args.lora_out) if args.lora_out else \
        resolve(f"outputs/lora_peft/{adapter.stem}")
    if (out_dir / "adapter_config.json").exists() and not args.force_convert:
        logger.info("复用已导出的 PEFT adapter：%s", out_dir)
        return out_dir

    return convert_checkpoint_to_peft(
        adapter, out_dir,
        r=cfg.path_("lora.r", 8), alpha=cfg.path_("lora.alpha", 16),
        dropout=cfg.path_("lora.dropout", 0.05),
        target_modules=cfg.path_("lora.target_modules", []),
        base_model=base_model,
    )


# ---------------------------------------------------------------------------
# prompt 编码
# ---------------------------------------------------------------------------
def encode_prompts(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    prompt_style: str,
    max_prompt_len: int,
) -> Iterator[List[int]]:
    """逐条编码 prompt 并返回 token id（保持输入顺序）。

    **保留尾部**截断，和训练/推理的 collator 一致；不做右截断。
    """
    for record in records:
        messages = build_messages(record["source"], prompt_style)
        ids = encode_prompt(tokenizer, messages, add_generation_prompt=True)
        if max_prompt_len and len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
        yield ids


def run_vllm(args: argparse.Namespace) -> None:
    cfg = load_config(resolve(args.config))
    model_cfg = load_config(resolve(cfg.get("model_config"))).get("model", {})
    data_cfg = load_config(resolve(cfg.get("data_config")))

    model_name = model_cfg.get("model_name_or_path", "")
    if args.model:
        model_name = args.model
    local = resolve(model_name)
    if is_local_dir(local):
        model_name = str(local)
    if not model_name:
        raise SystemExit("配置里没有 model.model_name_or_path，也没有传 --model")

    # vLLM 用不用 4-bit 由它自己的 quantization 参数决定，和配置里的
    # load_in_4bit 无关。这里显式提醒一次，免得有人以为 vLLM 会自动跟着降精度。
    if model_cfg.get("load_in_4bit"):
        logger.warning(
            "配置里 load_in_4bit=True，但 vLLM 不走 bitsandbytes 4-bit —— "
            "它会按 --dtype 加载（auto 时跟随模型配置的 torch_dtype）。"
            "T4 上请显式传 --dtype float16。"
        )

    # ---- LoRA adapter（放在最前面：--export-only 不需要数据/tokenizer）----
    lora_dir = ensure_peft_adapter(args, cfg, model_name)
    if args.export_only:
        logger.info("--export-only：只导出 adapter，不生成")
        return

    # ---- 数据 ----
    split_path = resolve(args.input) if args.input else resolve_split_file(
        data_cfg, args.split, args.data_dir
    )
    logger.info("数据: %s", split_path)
    records = list(read_jsonl(str(split_path)))
    if args.limit:
        records = records[: args.limit]

    # ---- 分片：按**下标**取模，保证同一输入下每个进程拿到的切片确定 ----
    if args.shard:
        try:
            shard_idx, shard_total = (int(x) for x in args.shard.split("/"))
        except ValueError as exc:
            raise SystemExit(f"--shard 要写成 I/N 的形式，收到：{args.shard!r}") from exc
        if not (0 <= shard_idx < shard_total):
            raise SystemExit(f"--shard 序号要在 [0, {shard_total}) 内，收到 {shard_idx}")
        before = len(records)
        records = [r for i, r in enumerate(records) if i % shard_total == shard_idx]
        logger.info("分片 %d/%d：%d 条 → %d 条", shard_idx, shard_total, before, len(records))
    else:
        shard_idx = shard_total = None

    if args.out:
        out_path = resolve(args.out)
    else:
        # 用 --input 指定任意子集（如 rl_prompts.jsonl）时，默认输出跟着输入文件
        # 命名，而不是 sft_{split}.jsonl —— 否则会把 val 的三元组覆盖掉。
        base = Path(args.input).stem if args.input else f"sft_{args.split}"
        suffix = f"_shard{shard_idx}of{shard_total}" if shard_total else ""
        # 多候选产物多加 _n{num}，避免和单候选文件落在同一个路径里互相覆盖 / 混淆
        num_suffix = f"_n{args.num}" if args.num > 1 else ""
        out_path = resolve(f"data/triples/{base}{num_suffix}{suffix}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path)
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        logger.info("全部 %d 条已生成完毕：%s", len(records), out_path)
        return
    logger.info("读入 %d 条，已完成 %d 条，待生成 %d 条", len(records), len(done), len(todo))

    # ---- tokenizer：只用来编码 prompt ----
    from transformers import AutoTokenizer

    trust_remote_code = bool(model_cfg.get("trust_remote_code"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, padding_side="left"
    )

    style = args.prompt_style or data_cfg.path_("data.prompt_style", "structured")
    max_length = args.max_prompt_len or cfg.path_("sft.max_length") or model_cfg.get("max_length")
    max_length = int(max_length or 2048)
    # vLLM 要求 prompt + 生成 ≤ max_model_len。HF 路径是 prompt 截到 max_length
    # 再生成 max_new_tokens，所以这里预留出生成位。
    max_model_len = args.max_model_len or (max_length + args.max_new_tokens)
    logger.info("prompt 截断长度 %d | vLLM max_model_len %d | max_new_tokens %d",
                max_length, max_model_len, args.max_new_tokens)

    lora_rank = cfg.path_("lora.r", 8)
    if lora_dir:
        # 以导出目录里的 rank 为准（导出的 adapter_config.json 才是 vLLM 实际读的）
        try:
            adapter_cfg = json.loads((lora_dir / "adapter_config.json").read_text(encoding="utf-8"))
            lora_rank = max(int(lora_rank), int(adapter_cfg.get("r", lora_rank)))
        except (OSError, ValueError, KeyError):
            pass
    if not lora_dir:
        logger.warning("没有指定 --adapter/--lora-dir，将用未微调的底座生成（只能当基线）")

    # ---- vLLM ----
    from vllm import LLM, SamplingParams

    llm_kwargs: Dict[str, Any] = dict(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        swap_space=args.swap_space,
        disable_log_stats=True,
        seed=args.seed,
    )
    if lora_dir:
        llm_kwargs.update(enable_lora=True, max_lora_rank=int(lora_rank), max_loras=1)

    # 精度默认跟随模型配置（和 HF 路径一致），而不是让 vLLM 去读模型自带的
    # config.json —— 后者可能是 fp16，而项目在 4090 上要的是 bf16。
    dtype = args.dtype
    if dtype == "auto":
        cfg_dtype = model_cfg.get("torch_dtype")
        if isinstance(cfg_dtype, str) and cfg_dtype not in ("auto", ""):
            dtype = cfg_dtype
    if dtype and dtype != "auto":
        llm_kwargs["dtype"] = dtype

    # 跨 vLLM 版本兼容：不同版本的 EngineArgs 字段不一样（V1 引擎删了 swap_space），
    # 把当前版本不认识的 kwargs 丢掉，而不是直接 TypeError。
    from sfzy.utils.vllm_compat import filter_llm_kwargs

    llm_kwargs, dropped = filter_llm_kwargs(llm_kwargs)
    if dropped:
        logger.warning(
            "当前 vLLM 的 EngineArgs 不认识这些参数，已忽略：%s（0.6.x 与 0.3x 的差异）",
            dropped,
        )

    logger.info("加载 vLLM：%s（dtype=%s, tp=%d, lora=%s）",
                model_name, dtype, args.tensor_parallel_size,
                lora_dir or "无")
    llm = LLM(**llm_kwargs)

    lora_request = None
    if lora_dir:
        from vllm.lora.request import LoRARequest

        # 0.5.x 的字段叫 lora_local_path，0.6+ 改名成 lora_path（旧名只留了
        # 一个会在未来版本删掉的兼容参数）。两种都试一次，免得踩弃用警告，
        # 也免得在新版本上直接 TypeError。
        try:
            lora_request = LoRARequest("sft", 1, lora_path=str(lora_dir))
        except TypeError:
            lora_request = LoRARequest("sft", 1, lora_local_path=str(lora_dir))

    # num==1 默认贪心（和 HF 路径的 do_sample=False 对齐，产物逐字节同构）；
    # num>1 必须采样，否则 n 份输出会一模一样。
    sampling = SamplingParams(
        n=args.num,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )
    logger.info("每条生成 %d 个候选（temperature=%.3f, top_p=%.3f）",
                args.num, args.temperature, args.top_p)

    started = time.time()
    count = 0
    out_lengths: List[int] = []
    # 分批提交：一次全塞进去虽然 vLLM 也吃得下，但中途被掐就一条都没落盘；
    # 每条 prompt 在开头就重新 tokenize 一次，内存占用与 batch 成正比。
    with open(out_path, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            prompt_ids = list(encode_prompts(chunk, tokenizer, style, max_length))
            # 用 TokensPrompt 字典而不是已弃用的 prompt_token_ids kwarg：
            # 0.6 起后者会告警，未来版本会删掉；prompts=[{...}] 从 0.5 一直
            # 到现在都支持。
            outputs = llm.generate(
                prompts=[{"prompt_token_ids": ids} for ids in prompt_ids],
                sampling_params=sampling,
                lora_request=lora_request,
                use_tqdm=False,
            )
            for record, output in zip(chunk, outputs):
                # vLLM 按采样顺序返回 n 个 completion；num==1 时就是 [0]。
                candidates = [o.text.strip() for o in output.outputs]
                entry = {
                    "id": record["id"],
                    "source": record["source"],
                    "reference": record.get("summary", ""),
                    # "output" 恒为第一个候选，兼容只读单字段的下游
                    "output": candidates[0] if candidates else "",
                }
                if args.num > 1:
                    entry["outputs"] = candidates
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                out_lengths.extend(len(c) for c in candidates)
                count += 1
            elapsed = time.time() - started
            rate = count / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - count) / rate / 60 if rate > 0 else float("inf")
            logger.info("进度 %d/%d | %.2f 条/秒 | 预计剩余 %.1f 分钟",
                        count, len(todo), rate, eta)

    elapsed = time.time() - started
    rate = count / elapsed if elapsed > 0 else 0.0
    logger.info("完成 %d 条，耗时 %.1f 分钟（%.2f 条/秒）", count, elapsed / 60, rate)
    if out_lengths:
        logger.info("输出长度 平均 %.0f 字 / 最短 %d / 最长 %d",
                    sum(out_lengths) / len(out_lengths), min(out_lengths), max(out_lengths))
    logger.info("输出: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="用 vLLM 生成 (原文, 参考摘要, 模型输出) 三元组")
    parser.add_argument("--config", default="configs/sft_kaggle.yaml")
    parser.add_argument("--adapter", default=None,
                        help="本项目格式的 LoRA checkpoint（.pt）；会自动导出成 PEFT 目录")
    parser.add_argument("--lora-dir", default=None,
                        help="已经是 PEFT 格式的 adapter 目录，直接用")
    parser.add_argument("--lora-out", default=None,
                        help="导出 PEFT adapter 的目录，默认 outputs/lora_peft/<checkpoint 名>")
    parser.add_argument("--force-convert", action="store_true",
                        help="即使 PEFT 目录已存在也重新导出")
    parser.add_argument("--export-only", action="store_true",
                        help="只导出 adapter 然后退出，用来单独验证格式")
    parser.add_argument("--model", default=None, help="覆盖配置里的底座模型路径")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--input", default=None,
                        help="直接指定输入 jsonl，覆盖 --split（如 data/splits/rl_prompts.jsonl）")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=None, help="默认 data/triples/sft_{split}.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="只生成所选集合的前 N 条。集合就是 --input 指定的文件，"
                             "没给 --input 时就是 --split 切分；按文件顺序截断，发生在"
                             "分片（--shard）和断点续跑过滤**之前**，所以两个分片合起来"
                             "正好覆盖这前 N 条（如 rl_prompts.jsonl 已按事实个数降序，"
                             "--limit 200 就是信号最强的 200 条）")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="数据分片。多卡不想用 tensor parallel 时，用两个进程各跑一个分片，"
                             "各自用 CUDA_VISIBLE_DEVICES 绑不同的卡，产物再拼起来")
    parser.add_argument("--prompt-style", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--num", type=int, default=1,
                        help="每篇文书生成多少个候选摘要。1（默认）走贪心，产物和以前完全一致；"
                             ">1 时走采样，产物多一个 outputs 列表字段（output 仍是第一个候选）")
    parser.add_argument("--temperature", type=float, default=None,
                        help="采样温度。默认：num==1 时 0.0（贪心），num>1 时 0.7。"
                             "num>1 时不能为 0，否则 N 个候选会完全相同")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="自 nucleus 采样阈值，仅在 temperature>0 时生效")
    parser.add_argument("--max-prompt-len", type=int, default=None,
                        help="prompt 截断长度，默认取配置的 sft.max_length")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="vLLM 的 max_model_len，默认 = max_prompt_len + max_new_tokens")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="每次提交给 vLLM 的 prompt 条数。vLLM 内部是连续批处理，"
                             "这个值只影响落盘粒度和峰值内存，128 起步即可")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="和 sft/infer.generate_one 的默认值保持一致，不要随意改")
    parser.add_argument("--dtype", default="auto",
                        help="auto | float16 | bfloat16。T4 不支持 bf16，必须显式传 float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--swap-space", type=int, default=4, help="每卡 CPU swap 空间 GiB")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="关掉 CUDA graph，省显存/好调试，代价是慢一点")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- num / temperature 的默认与校验（放在这里，run_vllm 直接用解析后的值）----
    if args.num < 1:
        parser.error(f"--num 至少为 1，收到 {args.num}")
    if args.temperature is None:
        # num==1 保持旧的贪心行为；num>1 需要采样才能拿到不同候选
        args.temperature = 0.0 if args.num == 1 else 0.7
    if args.num > 1 and args.temperature <= 0:
        parser.error(
            f"--num={args.num}>1 需要采样：--temperature 必须 >0（贪心会生成 "
            f"{args.num} 份完全相同的候选）。显式传一个正数，或省略让脚本取默认 0.7"
        )
    if not (0.0 < args.top_p <= 1.0):
        parser.error(f"--top-p 要在 (0, 1] 内，收到 {args.top_p}")

    run_vllm(args)


if __name__ == "__main__":
    main()
