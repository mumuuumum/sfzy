"""vLLM rollout 的冒烟检查：**一条命令**确认"能不能用 vLLM 生成"。

它做的事情与 `scripts/train_grpo.py` 启动时**完全一致**（同一套配置、同一个
加载器、同一个 VLLMRollout），只是不训练：

    加载 tokenizer/模型 → 注入 LoRA → 载入 SFT adapter → 建 vLLM →
    把当前 LoRA 同步过去 → 生成 1 个 prompt × G 条 → 打印耗时与显存

用法（在 GPU 机器上）：

    python tools/check_vllm_rollout.py --config configs/grpo_fact_coverage_vGPU48.yaml

判据（都会打印出来）：
    * 能打印出非空摘要文本，且条数 == num_return_sequences；
    * `策略同步` 与 `vLLM 生成` 两段耗时；
    * `torch.cuda.max_memory_allocated()`（训练模型 + vLLM 引擎共存的峰值）。

失败时它会把最可能的三个原因直接写出来（版本/显存/量化白名单），不用去翻栈。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                                # noqa: E402
from sfzy.data.prompts import build_messages                       # noqa: E402
from sfzy.models.loader import (                                   # noqa: E402
    build_quant_config, load_model, load_tokenizer, move_model_to_device,
)
from sfzy.models.lora import inject_lora, mark_only_lora_trainable  # noqa: E402
from sfzy.sft.checkpoint import load_checkpoint                    # noqa: E402
from sfzy.utils.logging import get_logger                          # noqa: E402

logger = get_logger("check_vllm")

# 工具版本标记：这份脚本要在"开发机"和"GPU 机"之间来回拷，很容易跑成旧副本
# （现象是 argparse 报 "unrecognized arguments"）。启动时把版本与支持的参数打出来，
# 一眼就能确认自己跑的是不是最新的。
TOOL_REVISION = 2
TOOL_FLAGS = "--config / --prompt-file / --group-size / --max-new-tokens / --quantization"


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM rollout 冒烟检查")
    parser.add_argument("--config", default="configs/grpo_fact_coverage_vGPU48.yaml")
    parser.add_argument("--prompt-file", default=None, help="默认取 rl.prompt_file 的第一条")
    parser.add_argument("--group-size", type=int, default=None, help="默认取 rl.group_size")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--quantization", default=None,
        help="覆盖 rl.vllm.quantization。可选 bitsandbytes / awq / gptq / none。\n"
             "★ 用 `--quantization bitsandbytes` 可以就地验证『bnb 4-bit 底座 + LoRA』这个\n"
             "  官方未测试的组合到底能不能跑（vLLM 0.11.2 里它既没进支持清单、也没被禁止）；\n"
             "  能跑的话底座显存从 12.4GB 降到约 3.5GB。跑不通就回到默认的 bf16。",
    )
    args = parser.parse_args()
    print(f"[check_vllm_rollout r{TOOL_REVISION}] 支持参数: {TOOL_FLAGS}")

    cfg = load_config(resolve(args.config))
    rl = cfg.get("rl", {})
    model_cfg = dict(cfg.get("model") or {})
    lora_cfg = cfg.get("lora") or {}
    vcfg = dict(rl.get("vllm") or {})
    model_name = model_cfg.get("model_name_or_path")
    local = resolve(model_name)
    if local.is_dir():
        model_cfg["model_name_or_path"] = str(local)

    # ---- 与 train_grpo 相同的加载顺序 ----
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, quant_config=build_quant_config(model_cfg),
                       gradient_checkpointing=False)
    if model_cfg.get("device_map") in (None, "", "none"):
        model = move_model_to_device(model, "cuda")
    inject_lora(model, target_modules=lora_cfg.get("target_modules", []),
                r=lora_cfg.get("r", 8), alpha=lora_cfg.get("alpha", 16),
                dropout=lora_cfg.get("dropout", 0.05))
    mark_only_lora_trainable(model)
    adapter = resolve(rl.get("sft_adapter"))
    if adapter.exists():
        load_checkpoint(adapter, model=model)
        logger.info("已载入 SFT adapter: %s", adapter)
    else:
        logger.warning("找不到 rl.sft_adapter=%s —— 用未微调的 LoRA 冒烟（只验证管道）", adapter)

    # ---- 与 train_grpo 完全相同的后端装配 ----
    from sfzy.rl.vllm_rollout import VLLMRollout

    want_len = int(rl.get("max_length", 2048)) + int(rl.get("max_new_tokens", 512)) + 8
    seq_len = int(getattr(getattr(model, "config", None), "seq_length", 0) or 8192)
    quantization = args.quantization if args.quantization is not None else (
        vcfg.get("quantization") or None
    )
    if isinstance(quantization, str) and quantization.lower() in ("none", "", "null"):
        quantization = None
    backend = VLLMRollout(
        base_model=model_cfg["model_name_or_path"], tokenizer=tokenizer,
        lora_alpha=int(lora_cfg.get("alpha", 16)), lora_rank=int(lora_cfg.get("r", 8)),
        export_root=resolve(rl.get("output_dir", "outputs/grpo")) / "vllm_adapters",
        dtype=str(vcfg.get("dtype", "bfloat16")),
        quantization=quantization,
        gpu_memory_utilization=float(vcfg.get("gpu_memory_utilization", 0.30)),
        max_model_len=int(vcfg.get("max_model_len") or min(want_len, seq_len)),
        max_loras=int(vcfg.get("max_loras", 1)),
        target_modules=lora_cfg.get("target_modules"),
        trust_remote_code=False,
        extra_llm_kwargs=vcfg.get("extra_llm_kwargs"),
    )

    prompt_file = resolve(args.prompt_file or rl.get("prompt_file", "data/splits/rl_prompts.jsonl"))
    import json

    with open(prompt_file, encoding="utf-8") as f:
        record = json.loads(next(line for line in f if line.strip()))
    messages = build_messages(record["source"], str(rl.get("prompt_style", "structured")))
    group = int(args.group_size or rl.get("group_size", 4))

    try:
        t0 = time.perf_counter()
        backend.sync_policy(model, 0)
        sync_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        texts = backend.generate(
            [messages], num_return_sequences=group,
            max_new_tokens=args.max_new_tokens,
            max_length=int(rl.get("max_length", 2048)),
            temperature=float(rl.get("temperature", 0.9)),
            top_p=float(rl.get("top_p", 0.95)),
        )
        gen_s = time.perf_counter() - t1
    except Exception as exc:  # noqa: BLE001 - 冒烟脚本的价值就在于把原因讲清楚
        import torch
        logger.error("vLLM rollout 冒烟失败：%s: %s", type(exc).__name__, exc)
        print("\n最可能的三个原因（按概率排序）：")
        print("  1) 版本不匹配：vLLM 0.11.2 要求 torch==2.9.0、transformers<5,>=4.56。")
        print("     看 `pip show vllm torch transformers | grep -E 'Name|Version'`，")
        print("     按 requirements-vllm.txt 重装（别手动指定别的 torch）。")
        print("  2) 显存不够：vLLM 的 gpu_memory_utilization 默认 0.9；与训练同卡要压到 0.25~0.35，")
        print("     或将底座换成 AWQ（docs/vllm_rollout.md 第 3 节）。")
        print("  3) 量化×LoRA 组合未测试：vLLM 的 LoRA 白名单只有 AWQ/GPTQ；")
        print("     底座别用 bitsandbytes（rl.vllm.quantization 留空或填 awq）。")
        if torch.cuda.is_available():
            print(f"\n当前显存峰值：{torch.cuda.max_memory_allocated()/1024**3:.1f} GiB / "
                  f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB")
        raise SystemExit(1)

    import torch

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print("\n================ vLLM rollout 冒烟结果 ================")
    print(f"  条数        : {len(texts)}（期望 {group}）")
    print(f"  vLLM 量化   : {quantization or '（无，按 dtype 加载）'}")
    print(f"  策略同步    : {sync_s:.2f}s（导出 LoRA + 挂 LoRARequest）")
    print(f"  生成 {args.max_new_tokens:>3d} token: {gen_s:.2f}s（{group} 条合计）")
    print(f"  显存峰值    : {peak:.1f} GiB")
    print(f"  样例输出    : {texts[0][:120]!r}")
    print("  ✔ vLLM rollout 可用 —— 可以去跑 scripts/train_grpo.py 了")

    backend.close()


if __name__ == "__main__":
    main()
