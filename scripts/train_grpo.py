"""GRPO 训练入口（装配层）。

流程：加载 SFT 模型 → 注入 LoRA 并载入 SFT 权重 → 读 prompt 池
      → 构造 GRPOTrainer → resume → train

和 train_sft.py 的分工完全一致：所有"环境相关"的决定都在这里，
rl/trainer.py 只管训练循环。

用法：**一切都在配置文件里**，脚本只有一个 --config。

    # 2×T4 15GB：先测"只考虑事实一致性"的方案
    python scripts/train_grpo.py --config configs/grpo_fact_only_t4.yaml

    # 2×A100 40GB：正式跑
    python scripts/train_grpo.py --config configs/grpo_fact_only_a100.yaml

临时改某一项不用动文件，加 --override 即可（值按 YAML 解析）：

    python scripts/train_grpo.py --config configs/grpo_fact_only_t4.yaml \
        --override rl.limit_prompts=8 \
        --override rl.group_size=2 \
        --override tracking.backend=none
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
from sfzy.eval.metrics import build_scorer                # noqa: E402
from sfzy.models.loader import (                          # noqa: E402
    build_quant_config,
    load_model,
    load_tokenizer,
    move_model_to_device,
    resolve_device_index,
)
from sfzy.models.lora import inject_lora, mark_only_lora_trainable  # noqa: E402
from sfzy.rl.reward_spec import RewardSpec                     # noqa: E402
from sfzy.rl.trainer import GRPOTrainer                   # noqa: E402
from sfzy.sft.checkpoint import load_checkpoint           # noqa: E402
from sfzy.utils.distributed import (                      # noqa: E402
    cleanup, describe_environment, init_distributed, is_main_process, pick_device,
)
from sfzy.utils.logging import get_logger                 # noqa: E402
from sfzy.utils.seed import set_seed                      # noqa: E402
from sfzy.utils.tracking import (                         # noqa: E402
    build_tracker, flatten_config, make_run_name, metric_columns,
)

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
    parser.add_argument("--config", default="configs/grpo_fact_judge.yaml")
    # 下面的参数**全部可以不传**：默认值来自配置文件，命令行只用来覆盖。
    # 例：python scripts/train_grpo.py --config configs/grpo_fact_only_t4.yaml
    parser.add_argument("--sft-adapter", default=None,
                        help="SFT 阶段产出的 LoRA checkpoint（GRPO 的起点）。"
                             "不传就用配置里的 rl.sft_adapter")
    parser.add_argument("--limit-prompts", type=int, default=None,
                        help="只用前 N 个 prompt，用来跑最小闭环。不传就用配置里的 rl.limit_prompts")
    parser.add_argument("--resume", default=None,
                        help="续训 checkpoint。不传就用配置里的 rl.resume_from")
    parser.add_argument("--quantize", action="store_true",
                        help="允许 4-bit 量化加载。默认关闭 —— 策略精度必须和 SFT "
                             "训练时一致，只有显存真的不够才开")
    parser.add_argument("--judge-backend", default=None, choices=["fact", "none"],
                        help="覆盖 semantic.backend：fact（六要素事实一致性裁判）/ none（关掉）")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    if args.override:
        apply_overrides(cfg, args.override)
        logger.info("配置覆盖: %s", args.override)

    model_cfg = load_config(resolve(cfg.get("model_config"))).get("model", {})
    lora_cfg = cfg.get("lora", {})
    rl_cfg = cfg.rl

    # 命令行没给就用配置；两者都没有才报错。这样"跑一次实验"可以只写一个 yaml。
    sft_adapter = args.sft_adapter or rl_cfg.get("sft_adapter")
    if not sft_adapter:
        raise ValueError(
            "没有指定 SFT checkpoint：在配置里写 rl.sft_adapter，"
            "或命令行加 --sft-adapter <path>。"
        )
    limit_prompts = (
        args.limit_prompts if args.limit_prompts is not None else rl_cfg.get("limit_prompts")
    )
    resume_from = args.resume or rl_cfg.get("resume_from")
    logger.info("SFT 起点: %s", sft_adapter)
    logger.info("续训: %s", resume_from or "从头开始")

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
    # 和 generate_triples.py 同一条约束：**策略模型的精度必须和 SFT 一致**。
    # LoRA adapter 是在非量化底座上训出来的，换成 4-bit 底座会引入量化误差；
    # 而 RL 阶段是在这个（已经带误差的）策略上继续优化，误差会被放大。
    # 量化开关有两个来源：命令行 --quantize，或配置里的 rl.quantize。
    # 后者让"4-bit 策略"成为配置的一部分 —— 否则配置写着 load_in_4bit=true
    # 却因为忘了加 --quantize 而静默跑成 bf16，只在显存日志里看得出来。
    want_quant = bool(args.quantize or rl_cfg.get("quantize", False))
    if model_cfg.get("load_in_4bit") and not want_quant:
        logger.warning(
            "模型配置里 load_in_4bit=True，但 GRPO 默认强制关闭量化："
            "SFT 用的是非量化底座，策略必须一致。确实需要 4-bit 请显式传 --quantize，"
            "或在配置里写 rl.quantize: true"
        )
        model_cfg = {**model_cfg, "load_in_4bit": False}
    elif want_quant and not model_cfg.get("load_in_4bit"):
        # 反向的坑：配置没写 load_in_4bit 但传了 --quantize，量化不会发生，
        # 而你以为开了。这里补上，保持两个开关语义一致。
        model_cfg = {**model_cfg, "load_in_4bit": True}

    # 量化模型不能 .to()：位置必须在加载时用 device_map 定死（见 loader 的说明）。
    # 策略固定在 cfg.device 指的卡上（单进程双卡方案里就是卡 0，卡 1 留给裁判）。
    if want_quant and model_cfg.get("device_map") in (None, "", "none"):
        idx = resolve_device_index(device)
        if idx is not None:
            model_cfg = {**model_cfg, "device_map": {"": idx}}
    logger.info(
        "策略精度: %s（量化=%s，device_map=%s）",
        "4-bit nf4" if want_quant else model_cfg.get("torch_dtype", "auto"),
        want_quant, model_cfg.get("device_map"),
    )

    model = load_model(model_cfg, quant_config=build_quant_config(model_cfg),
                       gradient_checkpointing=rl_cfg.get("gradient_checkpointing", True))
    if model_cfg.get("device_map") in (None, "", "none"):
        # 量化模型不能 .to()：设备由 device_map 决定。见 loader.move_model_to_device
        model = move_model_to_device(model, device)

    replaced = inject_lora(
        model, target_modules=lora_cfg.get("target_modules", []),
        r=lora_cfg.get("r", 8), alpha=lora_cfg.get("alpha", 16),
        dropout=lora_cfg.get("dropout", 0.05),
    )
    if replaced == 0:
        raise RuntimeError(f"LoRA 目标层一个都没匹配上：{lora_cfg.get('target_modules')}")
    mark_only_lora_trainable(model)

    adapter = resolve(sft_adapter)
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
            "检查 configs/grpo_fact_judge.yaml 的 defaults 是否指向了 SFT 实际用的那份配置。\n"
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
    if limit_prompts:
        prompts = prompts[: int(limit_prompts)]
    logger.info("prompt 池: %d 条", len(prompts))
    n_with_sft = sum(1 for p in prompts if p.get("sft_output"))
    logger.info("其中带 SFT 输出（基线锚要用）: %d 条", n_with_sft)
    if len(prompts) < rl_cfg.get("prompts_per_step", 4):
        raise ValueError("prompt 数少于每步用量，没法训练")

    # ---------------- 语义裁判 ----------------
    # 裁判是独立模型，和策略**分卡放**：策略吃满卡 0，裁判在卡 1。
    # 需不需要裁判由 reward 规格决定：只要有一个启用的 judge 项就要加载它。
    reward_spec = RewardSpec.from_config(rl_cfg.get("reward"))
    want_judge = reward_spec.needs_judge() and args.judge_backend != "none"
    semantic_cfg = dict(cfg.get("semantic") or {})
    scorer = build_scorer(semantic_cfg, override=args.judge_backend) if want_judge else None
    if scorer is not None:
        missing = reward_spec.required_signals - set(scorer.available_signals())
        if missing:
            raise ValueError(
                f"奖励需要的裁判信号 {sorted(missing)} 这个裁判产不出来，"
                f"它只有 {sorted(scorer.available_signals())}。\n"
                "  要么换裁判后端，要么在 rl.reward.terms 里把对应项关掉。"
            )
        logger.info(
            "语义裁判: %s（backend=%s，信号=%s）",
            getattr(scorer, "name", "unknown"),
            args.judge_backend or cfg.path_("semantic.backend"),
            sorted(scorer.available_signals()),
        )
    else:
        if reward_spec.needs_judge():
            logger.warning(
                "奖励里有 judge 项但没有加载裁判，trainer 启动时会直接报错。"
            )
        else:
            logger.info("奖励不含 judge 项，不加载裁判。")
    logger.info(
        "奖励项: %s（归一化权重 %s）",
        [t.name for t in reward_spec.enabled_terms],
        {k: round(v, 4) for k, v in reward_spec.normalized_weights().items()},
    )

    # ---------------- tracker ----------------
    # 续训时要接回 swanlab 上原来那条 run，否则一次 11 小时的训练被掐断后
    # 重启，面板上会多出一条断掉的曲线，而你会以为训练从零开始了。
    # run_id 存在 checkpoint 的 state 里，所以这里先把它读出来。
    resume_id = None
    if resume_from:
        try:
            resume_id = (load_checkpoint(resolve(resume_from)).get("state") or {}).get(
                "tracker_run_id"
            )
            if resume_id:
                logger.info("接回实验跟踪的 run: %s", resume_id)
        except Exception as exc:  # noqa: BLE001 — 读不到就当新建一条，不该拦住训练
            logger.warning("读 resume checkpoint 的 tracker run_id 失败（将新建 run）: %s", exc)

    tracker_cfg = cfg.get("tracking", {})
    tracker = build_tracker(
        backend=tracker_cfg.get("backend", "none"),
        project=tracker_cfg.get("project", "sfzy"),
        resume_id=resume_id,
        # 超参写进面板：事后看一条曲线时，"这组数到底是什么配置跑出来的"
        # 只能从 run 记录里找，日志文件经常已经滚没了。
        config=flatten_config(cfg),
        # 预声明 GRPO 的列（swanlab 专用），面板分组和中文名固定下来。
        columns=metric_columns(),
        run_name=make_run_name(
            "sfzy-grpo-" + "+".join(t.name for t in reward_spec.enabled_terms),
            G=rl_cfg.get("group_size"), kl=rl_cfg.get("kl_coef"), lr=rl_cfg.get("learning_rate"),
        ),
        enabled=is_main_process(),
    )

    # ---------------- 训练 ----------------
    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, cfg=cfg,
        output_dir=str(resolve(rl_cfg.get("output_dir", "outputs/grpo"))),
        tracker=tracker, is_main_process=is_main_process(), scorer=scorer,
    )
    trainer.resume(resume_from)
    state = trainer.train(prompts)
    tracker.finish()

    logger.info("GRPO 结束: step=%d history=%d 条", state.step, len(state.history))
    logger.info("checkpoint 目录: %s", trainer.output_dir)
    cleanup()


if __name__ == "__main__":
    main()
