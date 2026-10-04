"""用训练好的 SFT 模型生成三元组：(原文, 参考摘要, 模型输出)。

产出用来做两件事：
  1. **观察 SFT 的失败模式** —— 看它漏了多少原文里的事实、长度分布如何、
     格式有没有坍缩。这一步要在 GRPO 之前做，否则你不知道该改什么。
  2. **阶段 1 的奖励函数验证** —— 在真实输出上算奖励，看它和 ROUGE
     的相关性是否合理，避免把 GPU 配额浪费在一个设计错了的奖励上。

支持断点续跑：输出文件已经存在的 id 会被跳过。这在 Kaggle 上很重要 ——
生成几小时被掐断是常态。

用法：
    python scripts/generate_triples.py --config configs/sft_kaggle.yaml \
        --adapter outputs/sft_chatglm3/best.pt --split val

    # 先跑 100 条看速度和输出质量
    python scripts/generate_triples.py --config configs/sft_kaggle.yaml \
        --adapter outputs/sft_chatglm3/best.pt --split val --limit 100

全量 train（10738 条）跑得慢的话，用 vLLM 版本做同一件事：
    scripts/generate_triples_vllm.py（见 docs/vllm_triples.md）。
产物格式、断点续跑、--shard 语义都与本脚本一致，可以直接接在一起用。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                        # noqa: E402
from sfzy.models.loader import (                          # noqa: E402
    build_quant_config,
    load_model,
    load_tokenizer,
    move_model_to_device,
)
from sfzy.models.lora import inject_lora, mark_only_lora_trainable  # noqa: E402
from sfzy.sft.checkpoint import load_checkpoint            # noqa: E402
from sfzy.sft.dataset import load_records                  # noqa: E402
from sfzy.sft.infer import summarize_records, summarize_records_batched  # noqa: E402
from sfzy.utils.distributed import pick_device              # noqa: E402
from sfzy.utils.logging import get_logger                  # noqa: E402

logger = get_logger("generate_triples")


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_done_ids(path: Path) -> set[str]:
    """读已完成的 id，用于断点续跑。"""
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


def resolve_split_file(data_cfg, split: str, data_dir: Optional[str] = None) -> Path:
    """按候选顺序找到该 split 的样本文件，返回第一个真实存在的。

    项目里有**两套数据布局**，同一份代码要都能跑：

      Kaggle（configs/data_kaggle.yaml）  data/splits/{train,val,test}.jsonl   全量、正式
      本地（configs/data.yaml）           data/processed/{train,dev}.jsonl     200 条、图快

    之前这里写的是"val 不存在就换成 data.dev_file"，而 data.yaml 里根本没有
    dev_file 这个键 —— `.path_(..., "val.jsonl")` 取到的默认值和原名一模一样，
    等于没回退，直接报 FileNotFoundError: data/processed/val.jsonl。
    **默认值和替换目标同名**是这类回退逻辑的经典失效方式，所以改成候选列表：
    全列出来、取第一个存在的，一个都不在就把试过的路径打出来。
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


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 (原文, 参考摘要, 模型输出) 三元组")
    parser.add_argument("--config", default="configs/sft_kaggle.yaml")
    parser.add_argument("--adapter", default=None,
                        help="LoRA checkpoint 路径；不传则用未微调的底座（当基线用）")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--input", default=None,
                        help="直接指定输入 jsonl，覆盖 --split。用来给**任意子集**生成"
                             "（比如 GRPO 的 prompt 池 —— 它是 train 里筛出来的，"
                             "不是 train 的前 N 条，用 --split train --limit 对不上 id）")
    parser.add_argument("--data-dir", default=None,
                        help="覆盖数据目录：本地冒烟 data/processed，全量切分 data/splits")
    parser.add_argument("--out", default=None, help="默认 data/triples/sft_{split}.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="数据分片，形如 0/2 或 1/2。**多卡推理用这个，不要用 DDP** —— "
                             "推理没有梯度，DDP 的 all-reduce 无事可做，只会引入 NCCL "
                             "初始化、进程组和超时这一堆失败面。两个进程各跑一个分片，"
                             "用 CUDA_VISIBLE_DEVICES 绑不同的卡即可。")
    parser.add_argument("--prompt-style", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批处理大小。实测 batch=8 相对 batch=1 有 5~6 倍加速；"
                            "6B 模型在 16GB 卡上建议从 4 起步，OOM 就往下调")
    parser.add_argument("--quantize", action="store_true",
                        help="允许 4-bit 量化加载。**默认关闭，而且不应该轻易打开** —— "
                             "推理的精度必须和 SFT 训练时一致：4-bit 底座配上 bf16 "
                             "训出来的 LoRA adapter，量化误差会直接进生成结果。"
                             "只有显存真的不够（比如 16GB 的 T4）才考虑。")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    model_cfg = load_config(resolve(cfg.get("model_config"))).get("model", {})
    data_cfg = load_config(resolve(cfg.get("data_config")))

    model_name = model_cfg.get("model_name_or_path", "")
    local = resolve(model_name)
    if local.is_dir():
        model_cfg = {**model_cfg, "model_name_or_path": str(local)}

    # ---- 数据 ----
    split_path = resolve(args.input) if args.input else resolve_split_file(data_cfg, args.split, args.data_dir)
    logger.info("数据: %s", split_path)
    records = load_records(str(split_path))
    if args.limit:
        records = records[: args.limit]

    # ---- 分片：按下标取模 ----
    # 用下标而不是哈希：同一个输入文件下每个进程拿到的切片是确定的，
    # 改了 --limit 也不会变（哈希分片会因为顺序变化而不稳定）。
    shard_idx = shard_total = None
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

    if args.out:
        out_path = resolve(args.out)
    elif shard_total:
        # 自动分名，免得两个进程写同一个文件互相覆盖
        out_path = resolve(f"data/triples/sft_{args.split}_shard{shard_idx}of{shard_total}.jsonl")
    else:
        out_path = resolve(f"data/triples/sft_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path)
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        logger.info("全部 %d 条已生成完毕：%s", len(records), out_path)
        return
    logger.info("读入 %d 条，已完成 %d 条，待生成 %d 条", len(records), len(done), len(todo))

    # ---- 模型 ----
    #
    # **推理精度由代码决定，不由配置文件决定。**
    #
    # 理由：精度不是用户偏好，是**正确性约束** —— LoRA adapter 是在 bf16 底座上
    # 训出来的，换成 4-bit 底座会引入额外的量化误差，直接进生成结果。
    # 把它留在共享的 model 配置里，等于让"为了 Kaggle 省显存"的一次编辑
    # 悄悄改掉云上的推理行为 —— 这正是刚才那个报错的成因。
    #
    # 但也不能硬编码 bf16：Kaggle 的 16GB T4 上推理确实需要 4-bit。
    # 所以做成"默认强制不量化 + 显式开关 + 大声警告"。
    if model_cfg.get("load_in_4bit") and not args.quantize:
        logger.warning(
            "模型配置里 load_in_4bit=True，但推理默认强制关闭量化："
            "SFT 用的是非量化底座，推理必须一致，否则量化误差会进生成结果。"
            "确实需要 4-bit（比如 16GB 显存）请显式传 --quantize"
        )
        model_cfg = {**model_cfg, "load_in_4bit": False}
    if args.quantize:
        logger.warning("已启用 4-bit 量化推理 —— 确认这与 SFT 训练时的精度一致")

    logger.info("加载模型: %s", model_cfg.get("model_name_or_path"))
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, quant_config=build_quant_config(model_cfg),
                       gradient_checkpointing=False)
    # device_map 为空时必须手动搬到设备上 —— 漏了这一步模型会留在 CPU 上，
    # 生成速度慢几十倍，而且日志里看不出来（只会觉得"怎么这么慢"）。
    if model_cfg.get("device_map") in (None, "", "none"):
        device = pick_device(cfg.get("device", "auto"))
        # 量化模型不能 .to()：设备由 device_map 决定。见 loader.move_model_to_device
        model = move_model_to_device(model, device)
        logger.info("模型已搬到 %s", device)
    replaced = inject_lora(model, target_modules=load_config(
        resolve(args.config)).path_("lora.target_modules", []),
                        r=cfg.path_("lora.r", 8), alpha=cfg.path_("lora.alpha", 16),
                        dropout=cfg.path_("lora.dropout", 0.05))
    if replaced == 0:
        raise RuntimeError("LoRA 目标层一个都没匹配上，检查 target_modules")
    mark_only_lora_trainable(model)

    if args.adapter:
        adapter = resolve(args.adapter)
        load_checkpoint(adapter, model=model)
        logger.info("已载入 LoRA 权重: %s", adapter)
    else:
        logger.warning("未指定 --adapter，将用未微调的底座生成（只能当基线）")

    model.eval()

    # ---- 生成 ----
    style = args.prompt_style or data_cfg.path_("data.prompt_style", "structured")
    max_length = cfg.path_("sft.max_length") or model_cfg.get("max_length")
    logger.info("序列长度上限: %s", max_length)
    by_id = {r["id"]: r for r in todo}
    started = time.time()
    count = 0
    # batch_size <= 1 时走顺序路径，> 1 时走批处理。
    # 批处理路径用左 padding，实测输出和顺序路径逐字节一致。
    generator = (
        summarize_records_batched(
            model, tokenizer, todo, prompt_style=style, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens, max_length=max_length, log_every=0,
        )
        if args.batch_size > 1 else
        summarize_records(
            model, tokenizer, todo, prompt_style=style,
            max_new_tokens=args.max_new_tokens, max_length=max_length, log_every=0,
        )
    )
    logger.info("生成模式: %s", f"批处理 batch={args.batch_size}"
                if args.batch_size > 1 else "顺序 batch=1")
    out_lengths: list[int] = []          # 每条输出的字符数，用来报告长度分布
    with open(out_path, "a", encoding="utf-8") as f:
        for result in generator:
            record = by_id[result["id"]]
            f.write(json.dumps({
                "id": result["id"],
                "source": record["source"],
                "reference": record["summary"],
                "output": result["summary"],
            }, ensure_ascii=False) + "\n")
            f.flush()
            out_lengths.append(len(result["summary"]))
            count += 1
            if count % 20 == 0:
                elapsed = time.time() - started
                # elapsed 理论上不会是 0，但除以它之前先算清楚：
                # ZeroDivisionError 会让跑了几个小时的生成在最后一行日志上崩掉
                rate = count / elapsed if elapsed > 0 else 0.0
                eta = (len(todo) - count) / rate / 60 if rate > 0 else float("inf")
                logger.info("进度 %d/%d | %.2f 条/秒 | 预计剩余 %.1f 分钟",
                            count, len(todo), rate, eta)

    elapsed = time.time() - started
    rate = count / elapsed if elapsed > 0 else 0.0
    logger.info("完成 %d 条，耗时 %.1f 分钟（%.2f 条/秒）", count, elapsed / 60, rate)
    if out_lengths:
        # 平均输出长度是个很好的"生成是否正常"的体检项：
        # 明显短于参考摘要（~290 字）说明模型在提前停，明显长说明停不下来。
        logger.info("批大小 %d | 输出长度 平均 %.0f 字 / 最短 %d / 最长 %d",
                    args.batch_size, sum(out_lengths) / len(out_lengths),
                    min(out_lengths), max(out_lengths))
    logger.info("输出: %s", out_path)


if __name__ == "__main__":
    main()
