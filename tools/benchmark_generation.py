"""生成速度基准：单条循环 vs 批处理。

为什么测这个：vLLM 的核心收益来自**批处理**（连续批处理 + PagedAttention），
而不是单请求的算子优化。所以在决定要不要上 vLLM 之前，先用 HF 自己的
批处理能力测出"批处理能带来多少"，这才是收益的上界来源。

三种模式：
  sequential  batch=1 顺序生成 —— 当前 scripts/generate_triples.py 的做法
  batched     batch=N 一次生成（左 padding）—— 三元组生成的优化方向
  multi_ret   1 个 prompt 采样 G 条 —— GRPO rollout 的形态

本地跑（0.5B）：
    python tools/benchmark_generation.py --config configs/sft_local.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch                                              # noqa: E402

from sfzy.config import load_config                       # noqa: E402
from sfzy.data.prompts import build_messages              # noqa: E402
from sfzy.models.chat_template import decode, encode_prompt  # noqa: E402
from sfzy.models.loader import build_quant_config, load_model, load_tokenizer  # noqa: E402
from sfzy.utils.distributed import pick_device            # noqa: E402


def load_setup(config_path: str):
    cfg = load_config(ROOT / config_path)
    model_cfg = load_config(ROOT / cfg.get("model_config")).get("model", {})
    name = model_cfg.get("model_name_or_path", "")
    local = ROOT / name
    if local.is_dir():
        model_cfg = {**model_cfg, "model_name_or_path": str(local)}
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, quant_config=build_quant_config(model_cfg),
                       gradient_checkpointing=False)
    if model_cfg.get("device_map") in (None, "", "none"):
        model = model.to(pick_device(cfg.get("device", "auto")))
    model.eval()
    return model, tokenizer, model_cfg


def bench_sequential(model, tokenizer, prompts, max_new_tokens, reps=3):
    started = time.time()
    total_tokens = 0
    for _ in range(reps):
        for p in prompts:
            ids = torch.tensor([encode_prompt(tokenizer, p)], device=model.device)
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
            total_tokens += out.shape[1] - ids.shape[1]
    return time.time() - started, total_tokens


def bench_batched(model, tokenizer, prompts, max_new_tokens, reps=3):
    # 批处理生成必须用左侧 padding，右侧 padding 会让位置编码和注意力掩码错位
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        started = time.time()
        total_tokens = 0
        for _ in range(reps):
            batch = tokenizer.apply_chat_template(
                prompts, tokenize=True, add_generation_prompt=True,
                padding=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(input_ids=batch, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
            total_tokens += (out.shape[1] - batch.shape[1]) * len(prompts)
        return time.time() - started, total_tokens
    finally:
        tokenizer.padding_side = old_side


def bench_multi_return(model, tokenizer, prompt, group_size, max_new_tokens, reps=3):
    ids = torch.tensor([encode_prompt(tokenizer, prompt)], device=model.device)
    started = time.time()
    total_tokens = 0
    for _ in range(reps):
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=0.9, top_p=0.95,
                                 num_return_sequences=group_size,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
        total_tokens += (out.shape[1] - ids.shape[1]) * group_size
    return time.time() - started, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="生成速度基准")
    parser.add_argument("--config", default="configs/sft_local.yaml")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    model, tokenizer, model_cfg = load_setup(args.config)
    print(f"模型: {model_cfg.get('model_name_or_path')}")
    print(f"设备: {next(model.parameters()).device}  dtype: {next(model.parameters()).dtype}")

    records = []
    with open(ROOT / "data/splits/val.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num_prompts:
                break
            records.append(json.loads(line))
    prompts = [build_messages(r["source"], "structured") for r in records]
    print(f"prompt 数 {len(prompts)}，每个约 {len(records[0]['source'])} 字\n")

    results = {}

    t, tok = bench_sequential(model, tokenizer, prompts, args.max_new_tokens, args.reps)
    results["sequential"] = (t, tok)
    print(f"  sequential (batch=1)    {t:6.1f}s  {tok/t:7.1f} tok/s")

    t, tok = bench_batched(model, tokenizer, prompts, args.max_new_tokens, args.reps)
    results["batched"] = (t, tok)
    print(f"  batched    (batch={len(prompts)})   {t:6.1f}s  {tok/t:7.1f} tok/s"
          f"   ← 加速 {(tok/t) / (results['sequential'][1]/results['sequential'][0]):.2f}x")

    t, tok = bench_multi_return(model, tokenizer, prompts[0], args.group_size,
                                args.max_new_tokens, args.reps)
    results["multi_return"] = (t, tok)
    print(f"  multi_ret  (G={args.group_size})      {t:6.1f}s  {tok/t:7.1f} tok/s"
          f"   ← 加速 {(tok/t) / (results['sequential'][1]/results['sequential'][0]):.2f}x")

    print()
    if torch.cuda.is_available():
        print(f"  显存峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB "
              f"/ {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")


if __name__ == "__main__":
    main()
