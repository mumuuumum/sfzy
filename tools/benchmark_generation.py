"""生成速度基准：单条循环 vs 批处理 vs 一组多采样。

为什么测这个：vLLM 的核心收益来自**批处理**（连续批处理 + PagedAttention），
而不是单请求的算子优化。所以在决定要不要上 vLLM 之前，先用 HF 自己的
批处理能力测出"批处理能带来多少"，这才是收益的上界来源。

三种模式：
  sequential  batch=1 顺序生成          —— sft/infer.generate_one
  batched     batch=N 一次生成（左 padding）—— sft/infer.generate_batch
  multi_ret   1 个 prompt 采样 G 条     —— GRPO rollout 的形态

**三种模式都直接调用 sft/infer.py 里的真实函数**，不另写一套生成逻辑 ——
否则量出来的是"我手写的 benchmark 有多快"，而不是"上线那条路径有多快"。

本地跑（0.5B，4GB 卡上要压小一点）：
    python tools/benchmark_generation.py --config configs/sft_local.yaml \\
        --num-prompts 4 --max-new-tokens 64 --reps 1

输出的 chars/s 是跨模式可比的（同一批 prompt），据此可以直接估
"生成 val/test 的三元组要几个小时"。
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
from sfzy.models.loader import (                          # noqa: E402
    build_quant_config,
    load_model,
    load_tokenizer,
    move_model_to_device,
)
from sfzy.sft.infer import generate_batch, generate_one   # noqa: E402
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
        # 量化模型不能 .to()：设备由 device_map 决定。见 loader.move_model_to_device
        model = move_model_to_device(model, pick_device(cfg.get("device", "auto")))
    model.eval()
    # 截断长度和训练/推理保持一致，否则量的不是真实场景
    max_length = cfg.path_("sft.max_length") or model_cfg.get("max_length")
    return model, tokenizer, model_cfg, max_length


def bench_sequential(model, tokenizer, messages_list, max_new_tokens, max_length, reps=3):
    """batch=1 顺序生成，走 generate_one（和生成三元组的默认路径一致）。"""
    started = time.time()
    chars = 0
    for _ in range(reps):
        for messages in messages_list:
            chars += len(generate_one(
                model, tokenizer, messages,
                max_new_tokens=max_new_tokens, max_length=max_length,
            ))
    return time.time() - started, chars


def bench_batched(model, tokenizer, messages_list, max_new_tokens, max_length, reps=3):
    """一次生成 N 条，走 generate_batch（内部已经处理左 padding）。

    注意一个**不利于批处理**的因素：批处理要把 prompt 补到批内最长，
    padding 那些位置也要参与前向。所以批越大、长度越不齐，单条的边际成本越高。
    """
    started = time.time()
    chars = 0
    for _ in range(reps):
        outputs = generate_batch(
            model, tokenizer, messages_list,
            max_new_tokens=max_new_tokens, max_length=max_length,
        )
        chars += sum(len(o) for o in outputs)
    return time.time() - started, chars


def bench_multi_return(model, tokenizer, messages, group_size,
                       max_new_tokens, max_length, reps=3):
    """一个 prompt 采样 G 条 —— GRPO rollout 的形态。

    这里必须 do_sample=True：G 条如果贪心解码，会得到 G 条一模一样的输出，
    组内奖励方差为 0，优势函数恒为 0（GRPO 就学不到任何东西）。
    """
    started = time.time()
    chars = 0
    for _ in range(reps):
        outputs = generate_batch(
            model, tokenizer, [messages],
            max_new_tokens=max_new_tokens, max_length=max_length,
            do_sample=True, temperature=0.9, top_p=0.95,
            num_return_sequences=group_size,
        )
        chars += sum(len(o) for o in outputs)
    return time.time() - started, chars


def main() -> None:
    parser = argparse.ArgumentParser(description="生成速度基准")
    parser.add_argument("--config", default="configs/sft_local.yaml")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    model, tokenizer, model_cfg, max_length = load_setup(args.config)
    print(f"模型: {model_cfg.get('model_name_or_path')}")
    print(f"设备: {next(model.parameters()).device}  dtype: {next(model.parameters()).dtype}")
    print(f"序列长度上限: {max_length}  每个 prompt 生成 <= {args.max_new_tokens} token")

    records = []
    with open(ROOT / "data/splits/val.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num_prompts:
                break
            records.append(json.loads(line))
    messages_list = [build_messages(r["source"], "structured") for r in records]
    print(f"prompt 数 {len(messages_list)}，原文平均 {sum(len(r['source']) for r in records)//len(records)} 字\n")

    rates = {}

    t, chars = bench_sequential(model, tokenizer, messages_list,
                                args.max_new_tokens, max_length, args.reps)
    rates["sequential"] = chars / t
    print(f"  sequential (batch=1)     {t:6.1f}s  {chars/t:7.1f} 字/秒")

    t, chars = bench_batched(model, tokenizer, messages_list,
                             args.max_new_tokens, max_length, args.reps)
    rates["batched"] = chars / t
    print(f"  batched    (batch={len(messages_list)})    {t:6.1f}s  {chars/t:7.1f} 字/秒"
          f"   ← 加速 {rates['batched']/rates['sequential']:.2f}x")

    t, chars = bench_multi_return(model, tokenizer, messages_list[0], args.group_size,
                                  args.max_new_tokens, max_length, args.reps)
    rates["multi_return"] = chars / t
    print(f"  multi_ret  (G={args.group_size})       {t:6.1f}s  {chars/t:7.1f} 字/秒"
          f"   ← 加速 {rates['multi_return']/rates['sequential']:.2f}x")

    # 把速率换算成"生成三元组要多久"，这才是决定要不要上 vLLM 的数
    chars_per_record = 300      # 参考摘要平均 ~290 字（实测 data/splits）
    print("\n按上面的速率外推（假设输出长度 = 参考摘要的 %d 字）：" % chars_per_record)
    for name, n_records in (("val 1340 条", 1340), ("test 1349 条", 1349),
                            ("train 10738 条", 10738)):
        rate = rates["batched"] if name.startswith("train") else rates["sequential"]
        hours = n_records * chars_per_record / rate / 3600
        print(f"  {name:16s} 用 {('batched' if name.startswith('train') else 'sequential'):10s}"
              f" ≈ {hours:5.1f} 小时")

    print()
    if torch.cuda.is_available():
        print(f"  显存峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB "
              f"/ {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")


if __name__ == "__main__":
    main()
