"""SFT 训练入口（装配层）。

这个脚本只做装配，不实现任何训练逻辑：
    读配置 -> 初始化环境 -> 加载模型 -> 注入 LoRA -> 构建数据与 collator
    -> 构造 tracker -> 构造 trainer -> resume -> train

所有"环境相关"的决定（用哪块卡、是不是分布式、上报到哪里）都收敛在这里，
sft/trainer.py 只负责训练循环本身。

用法：
    python scripts/train_sft.py --config configs/sft_local.yaml
    python scripts/train_sft.py --config configs/sft_local.yaml --limit 32   # 只跑 32 条，先验证能跑通
    python scripts/train_sft.py --config configs/sft_local.yaml --resume outputs/sft_local/step_000020.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                      # noqa: E402
from sfzy.data.collator import SFTCollator               # noqa: E402
from sfzy.models.loader import (                         # noqa: E402
    build_quant_config,
    describe_model,
    load_model,
    load_tokenizer,
)
from sfzy.models.lora import inject_lora, mark_only_lora_trainable  # noqa: E402
from sfzy.sft.dataset import SFTDataset, load_records    # noqa: E402
from sfzy.sft.trainer import SFTTrainer                  # noqa: E402
from sfzy.utils.distributed import (                     # noqa: E402
    cleanup,
    describe_environment,
    init_distributed,
    is_main_process,
    pick_device,
)
from sfzy.utils.logging import get_logger                # noqa: E402
from sfzy.utils.seed import set_seed                     # noqa: E402
from sfzy.utils.tracking import build_tracker, make_run_name  # noqa: E402

logger = get_logger("train_sft")


def resolve(path: str | Path) -> Path:
    """把相对路径按项目根目录解析，避免依赖当前工作目录。"""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def apply_overrides(cfg, pairs) -> None:
    """就地覆盖配置项，形如 ["sft.num_epochs=1", "lora.r=4"]。

    调试时非常有用：把 grad_accum_steps 改成 1、把 num_epochs 改成 1、
    把 max_length 改小——都不用去动 yaml 文件，跑完也不用记得改回来。

    值用 yaml.safe_load 解析，所以 1 会变成 int、"1" 会变成 str、
    null 会变成 None，和直接在 yaml 里写一样。

    注意：调用必须在 `sft_cfg = cfg.sft` 之前 —— Config.__getattr__ 返回的是
    副本，先取出来再覆盖就改不到它了。
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
    parser = argparse.ArgumentParser(description="SFT 训练（手写训练循环）")
    parser.add_argument("--config", default="configs/sft_local.yaml")
    parser.add_argument("--resume", default=None,
                        help="覆盖配置里的 resume_from；不传则用配置值（null 表示从头训练）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只用前 N 条训练样本，用来快速验证链路能不能跑通")
    parser.add_argument("--dev-limit", type=int, default=None,
                        help="验证集只用前 N 条。**仅供调试** —— 冒烟测试时"
                             "全量验证要十几分钟，而且每轮都跑。正式训练不要设。")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="临时覆盖配置项，可重复。例如："
                             "--override sft.num_epochs=1 --override sft.grad_accum_steps=1")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    if args.override:
        apply_overrides(cfg, args.override)
        logger.info("配置覆盖: %s", args.override)

    model_cfg = load_config(resolve(cfg.get("model_config"))).get("model", {})
    data_cfg = load_config(resolve(cfg.get("data_config")))
    sft_cfg = cfg.sft
    lora_cfg = cfg.get("lora", {})

    # ---------- 精度只有一个真源：sft.dtype ----------
    # 模型配置里的 torch_dtype 单独写一份，很容易和 sft.dtype 不一致。
    # 不一致的后果不是报错，而是"权重 fp16 + autocast bf16"这种混合状态：
    # 每个算子都要多做一次类型转换，实测慢 2.3 倍（7m38s vs 17m19s），
    # 而且日志里看不出来——两边的 dtype_breakdown 会显示模型其实没换过精度。
    # 所以这里用 sft.dtype 覆盖模型配置，让配置项之间不可能打架。
    if sft_cfg.get("dtype"):
        if model_cfg.get("torch_dtype") and model_cfg["torch_dtype"] != sft_cfg["dtype"]:
            logger.info(
                "精度对齐: 模型配置的 torch_dtype=%s 与 sft.dtype=%s 不一致，按后者覆盖",
                model_cfg["torch_dtype"], sft_cfg["dtype"],
            )
        model_cfg = {
            **model_cfg,
            "torch_dtype": sft_cfg["dtype"],
            # 量化路径下这个也必须一致，否则 4-bit 的反量化精度和 autocast 又对不上
            "bnb_4bit_compute_dtype": sft_cfg["dtype"],
        }

    # ---------------- 1. 环境 ----------------
    local_rank = init_distributed()
    set_seed(cfg.get("seed", 42))
    device = pick_device(cfg.get("device", "auto"))
    logger.info("设备: %s", device)
    logger.info("环境: %s", describe_environment())

    # ---------------- 2. 模型 + LoRA ----------------
    # 模型名可能是 HF 仓库 id（THUDM/chatglm3-6b），也可能是项目内的相对路径
    # （models/Qwen2.5-0.5B-Instruct）。后者要按项目根目录解析成绝对路径，
    # 否则 from_pretrained 会按**当前工作目录**去找，换个目录启动就失败。
    model_name = model_cfg.get("model_name_or_path", "")
    local_path = resolve(model_name) if model_name else None
    if local_path is not None and local_path.is_dir():
        model_cfg = {**model_cfg, "model_name_or_path": str(local_path)}
    logger.info("加载模型: %s", model_cfg.get("model_name_or_path"))
    tokenizer = load_tokenizer(model_cfg)
    quant_config = build_quant_config(model_cfg)
    model = load_model(
        model_cfg,
        quant_config=quant_config,
        gradient_checkpointing=sft_cfg.get("gradient_checkpointing", False),
    )
    # device_map 为空时才手动搬运；device_map="auto" 已经由 accelerate 放好了，
    # 再调 .to() 会直接报错。
    if model_cfg.get("device_map") in (None, "", "none"):
        model = model.to(device)

    replaced = inject_lora(
        model,
        target_modules=lora_cfg.get("target_modules", []),
        r=lora_cfg.get("r", 8),
        alpha=lora_cfg.get("alpha", 16),
        dropout=lora_cfg.get("dropout", 0.05),
    )
    if replaced == 0:
        raise RuntimeError(
            f"没有匹配到任何层，target_modules={lora_cfg.get('target_modules')}。"
            "先打印 model.named_modules() 确认真实的层名。"
        )
    mark_only_lora_trainable(model)
    logger.info("注入 LoRA: %d 层", replaced)
    logger.info("模型概况: %s", describe_model(model))

    # ---------------- 3. 数据 ----------------
    processed_dir = resolve(data_cfg.path_("data.processed_dir", "data/processed"))
    # 文件名可配置：本地开发用 data/processed/{train,dev}.jsonl，
    # Kaggle 上用 data/splits/{train,val,test}.jsonl（由 tools/make_splits.py 生成）
    train_file = data_cfg.path_("data.train_file", "train.jsonl")
    dev_file = data_cfg.path_("data.dev_file", "dev.jsonl")
    train_records = load_records(str(processed_dir / train_file))
    dev_records = load_records(str(processed_dir / dev_file))
    if args.limit is not None:
        train_records = train_records[: args.limit]
        logger.info("--limit 生效，只使用前 %d 条训练样本", len(train_records))
    if args.dev_limit is not None:
        dev_records = dev_records[: args.dev_limit]
        logger.info("--dev-limit 生效，只使用前 %d 条验证样本（正式训练请不要设）",
                    len(dev_records))

    prompt_style = data_cfg.path_("data.prompt_style", "structured")
    train_dataset = SFTDataset(train_records, prompt_style=prompt_style)
    dev_dataset = SFTDataset(dev_records, prompt_style=prompt_style)
    logger.info("训练 %d 条 / 验证 %d 条", len(train_dataset), len(dev_dataset))

    # max_length 优先从 sft 配置读，好让 --override sft.max_length=512 能生效。
    # 夜间跑大规模数据时，把 1024 压到 512 能把耗时砍掉近一半，
    # 而我们关心的是"长跑会不会烂"，不是指标。
    max_length = sft_cfg.get("max_length") or model_cfg.get("max_length", 2048)
    logger.info("序列长度上限: %d", max_length)
    collator = SFTCollator(tokenizer, max_length=max_length)

    # ---------------- 4. 实验跟踪 ----------------
    tracker_cfg = cfg.get("tracking", {})
    tracker = build_tracker(
        backend=tracker_cfg.get("backend", "none"),
        project=tracker_cfg.get("project", "sfzy"),
        run_name=make_run_name(
            f"sfzy-sft-{model_cfg.get('model_name_or_path', '').split('/')[-1]}",
            lora_r=lora_cfg.get("r"),
            lr=sft_cfg.get("learning_rate"),
            epochs=sft_cfg.get("num_epochs"),
        ),
        enabled=is_main_process(),
    )

    # ---------------- 5. 训练 ----------------
    output_dir = resolve(sft_cfg.get("output_dir", "outputs/sft"))
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        collator=collator,
        cfg=cfg,
        output_dir=str(output_dir),
        tracker=tracker,
        is_main_process=is_main_process(),
    )

    # 断点续训必须显式指定：命令行 --resume 优先，否则用配置里的 resume_from。
    # 两者都为空就是从零开始，不会去目录里"找最新的"。
    trainer.resume(args.resume)

    state = trainer.train(train_dataset, dev_dataset)
    tracker.finish()

    logger.info(
        "训练结束: step=%d epoch=%d best_dev_loss=%.4f history=%d 条",
        state.step, state.epoch, state.best_dev_loss, len(state.history),
    )
    logger.info("checkpoint 目录: %s", output_dir)

    cleanup()


if __name__ == "__main__":
    main()
