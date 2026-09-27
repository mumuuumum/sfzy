"""GRPO 训练入口（装配层）。

流程：加载 SFT 模型 → 注入 LoRA 并载入 SFT 权重 → 读 prompt 池
      → 构造 GRPOTrainer → resume → train

和 train_sft.py 的分工完全一致：所有"环境相关"的决定都在这里，
rl/trainer.py 只管训练循环。

用法：
    # 最小闭环：先确认能跑（约 10 分钟）
    python scripts/train_grpo.py --config configs/grpo_cloud.yaml \
        --sft-adapter outputs/sft_chatglm3/best.pt --limit-prompts 8

    # 正式
    python scripts/train_grpo.py --config configs/grpo_cloud.yaml \
        --sft-adapter outputs/sft_chatglm3/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                       # noqa: E402
from sfzy.models.loader import build_quant_config, load_model, load_tokenizer  # noqa: E402
from sfzy.models.lora import inject_lora, mark_only_lora_trainable  # noqa: E402
from sfzy.rl.trainer import GRPOTrainer                   # noqa: E402
from sfzy.sft.checkpoint import load_checkpoint           # noqa: E402
from sfzy.utils.distributed import (                      # noqa: E402
    cleanup, describe_environment, init_distributed, is_main_process, pick_device,
)
from sfzy.utils.logging import get_logger                 # noqa: E402
from sfzy.utils.seed import set_seed                      # noqa: E402
from sfzy.utils.tracking import build_tracker, make_run_name  # noqa: E402

logger = get_logger("train_grpo")


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def apply_overrides(cfg, pairs) -> None:
    """就地覆盖配置项，形如 ["rl.kl_coef=0.01", "rl.group_size=4"]。

    三组对照实验（A1/A2/A3）就是靠它切出来的，不用改文件。
    """
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--override 需要 KEY=VALUE 形式，收到：{item!r}")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f"--override 的路径不存在：{dotted}")
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO 训练")
    parser.add_argument("--config", default="configs/grpo_cloud.yaml")
    parser.add_argument("--sft-adapter", required=True,
                        help="SFT 阶段产出的 LoRA checkpoint（GRPO 的起点）")
    parser.add_argument("--limit-prompts", type=int, default=None,
                        help="只用前 N 个 prompt，用来跑最小闭环")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    if args.override:
        apply_overrides(cfg, args.override)
        logger.info("配置覆盖: %s", args.override)

    model_cfg = load_config(resolve(cfg.get("model_config"))).get("model", {})
    lora_cfg = cfg.get("lora", {})
    rl_cfg = cfg.rl

    local_rank = init_distributed()
    set_seed(cfg.get("seed", 42))
    device = pick_device(cfg.get("device", "auto"))
    logger.info("设备: %s", device)
    logger.info("环境: %s", describe_environment())

    # ---------------- 模型 + LoRA + SFT 权重 ----------------
    model_name = model_cfg.get("model_name_or_path", "")
    local_path = resolve(model_name) if model_name else None
    if local_path is not None and local_path.is_dir():
        model_cfg = {**model_cfg, "model_name_or_path": str(local_path)}

    logger.info("加载模型: %s", model_cfg.get("model_name_or_path"))
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, quant_config=build_quant_config(model_cfg),
                       gradient_checkpointing=rl_cfg.get("gradient_checkpointing", True))
    if model_cfg.get("device_map") in (None, "", "none"):
        model = model.to(device)

    replaced = inject_lora(
        model, target_modules=lora_cfg.get("target_modules", []),
        r=lora_cfg.get("r", 8), alpha=lora_cfg.get("alpha", 16),
        dropout=lora_cfg.get("dropout", 0.05),
    )
    if replaced == 0:
        raise RuntimeError(f"LoRA 目标层一个都没匹配上：{lora_cfg.get('target_modules')}")
    mark_only_lora_trainable(model)

    adapter = resolve(args.sft_adapter)
    try:
        load_checkpoint(adapter, model=model)
    except RuntimeError as exc:
        # 形状不匹配是最常见的失败：SFT 和 GRPO 用了不同的 lora.r / alpha。
        # 这种错误如果不解释，报错信息（一串 size mismatch）很难定位到根因。
        raise RuntimeError(
            f"载入 SFT 权重失败：{adapter}\n"
            f"当前配置的 LoRA 是 r={lora_cfg.get('r')} alpha={lora_cfg.get('alpha')}，"
            f"target={lora_cfg.get('target_modules')}。\n"
            "**GRPO 的 LoRA 结构必须和 SFT 阶段完全一致** —— "
            "检查 configs/grpo_cloud.yaml 的 defaults 是否指向了 SFT 实际用的那份配置。\n"
            f"原始错误：{exc}"
        ) from exc
    logger.info("已载入 SFT 权重作为策略起点: %s", adapter)

    # ---------------- prompt 池 ----------------
    prompt_path = resolve(rl_cfg.get("prompt_file", "data/splits/rl_prompts.jsonl"))
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"找不到 prompt 池：{prompt_path}\n"
            "先跑：python tools/select_rl_prompts.py --n 1000"
        )
    prompts = []
    with open(prompt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    logger.info("prompt 池: %d 条", len(prompts))
    if len(prompts) < rl_cfg.get("prompts_per_step", 4):
        raise ValueError("prompt 数少于每步用量，没法训练")

    # ---------------- tracker ----------------
    tracker_cfg = cfg.get("tracking", {})
    tracker = build_tracker(
        backend=tracker_cfg.get("backend", "none"),
        project=tracker_cfg.get("project", "sfzy"),
        run_name=make_run_name(
            f"sfzy-grpo-{rl_cfg.get('reward', {}).get('mode', 'gated')}",
            G=rl_cfg.get("group_size"), kl=rl_cfg.get("kl_coef"), lr=rl_cfg.get("learning_rate"),
        ),
        enabled=is_main_process(),
    )

    # ---------------- 训练 ----------------
    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, cfg=cfg,
        output_dir=str(resolve(rl_cfg.get("output_dir", "outputs/grpo"))),
        tracker=tracker, is_main_process=is_main_process(),
    )
    trainer.resume(args.resume)
    state = trainer.train(prompts)
    tracker.finish()

    logger.info("GRPO 结束: step=%d history=%d 条", state.step, len(state.history))
    logger.info("checkpoint 目录: %s", trainer.output_dir)
    cleanup()


if __name__ == "__main__":
    main()
