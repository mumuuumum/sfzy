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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                        # noqa: E402
from sfzy.models.loader import build_quant_config, load_model, load_tokenizer  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 (原文, 参考摘要, 模型输出) 三元组")
    parser.add_argument("--config", default="configs/sft_kaggle.yaml")
    parser.add_argument("--adapter", default=None,
                        help="LoRA checkpoint 路径；不传则用未微调的底座（当基线用）")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default=None, help="覆盖配置里的 processed_dir")
    parser.add_argument("--out", default=None, help="默认 data/triples/sft_{split}.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-style", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批处理大小。实测 batch=8 相对 batch=1 有 5~6 倍加速；"
                             "6B 模型在 16GB 卡上建议从 4 起步，OOM 就往下调")
    parser.add_argument("--max-records", type=int, default=None,
                        help="调试用：只读前 N 条（和 --limit 的区别是不改文件顺序）")
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    model_cfg = load_config(resolve(cfg.get("model_config"))).get("model", {})
    data_cfg = load_config(resolve(cfg.get("data_config")))

    model_name = model_cfg.get("model_name_or_path", "")
    local = resolve(model_name)
    if local.is_dir():
        model_cfg = {**model_cfg, "model_name_or_path": str(local)}

    # ---- 数据 ----
    data_dir = resolve(args.data_dir or data_cfg.path_("data.processed_dir", "data/splits"))
    split_file = f"{args.split}.jsonl"
    if args.split == "val" and not (data_dir / split_file).exists():
        split_file = data_cfg.path_("data.dev_file", "val.jsonl")
    records = load_records(str(data_dir / split_file))
    if args.limit:
        records = records[: args.limit]

    out_path = resolve(args.out or f"data/triples/sft_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path)
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        logger.info("全部 %d 条已生成完毕：%s", len(records), out_path)
        return
    logger.info("读入 %d 条，已完成 %d 条，待生成 %d 条", len(records), len(done), len(todo))

    # ---- 模型 ----
    logger.info("加载模型: %s", model_cfg.get("model_name_or_path"))
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, quant_config=build_quant_config(model_cfg),
                       gradient_checkpointing=False)
    # device_map 为空时必须手动搬到设备上 —— 漏了这一步模型会留在 CPU 上，
    # 生成速度慢几十倍，而且日志里看不出来（只会觉得"怎么这么慢"）。
    if model_cfg.get("device_map") in (None, "", "none"):
        device = pick_device(cfg.get("device", "auto"))
        model = model.to(device)
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
            max_new_tokens=args.max_new_tokens, max_length=max_length, log_every=50,
        )
        if args.batch_size > 1 else
        summarize_records(
            model, tokenizer, todo, prompt_style=style,
            max_new_tokens=args.max_new_tokens, max_length=max_length, log_every=50,
        )
    )
    logger.info("生成模式: %s", f"批处理 batch={args.batch_size}"
                if args.batch_size > 1 else "顺序 batch=1")
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
            count += 1
            if count % 20 == 0:
                elapsed = time.time() - started
                rate = count / elapsed
                eta = (len(todo) - count) / rate / 60
                logger.info("进度 %d/%d | %.2f 条/秒 | 预计剩余 %.1f 分钟",
                            count, len(todo), rate, eta)

    elapsed = time.time() - started
    logger.info("完成 %d 条，耗时 %.1f 分钟（%.2f 条/秒）", count, elapsed / 60, count / elapsed)
    logger.info("输出: %s", out_path)


if __name__ == "__main__":
    main()
