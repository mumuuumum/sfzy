"""显存与设备分布诊断。

**不要猜，测。** 这个脚本回答三个问题：

  1. 模型到底落在几张卡上？（`device_map` 生效了吗）
  2. 梯度检查点真的开启了吗？（在 ChatGLM3 这种远程代码模型上不一定生效）
  3. 一次 forward+backward 之后，每张卡实际占了多少？

用法（Kaggle 上直接跑，不需要训练配额）：
    python scripts/check_memory.py --config configs/model_chatglm3_6b.yaml \\
        --batch-size 1 --seq-len 1536
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch                                              # noqa: E402

from sfzy.config import load_config                       # noqa: E402
from sfzy.data.collator import SFTCollator                # noqa: E402
from sfzy.models.loader import (                          # noqa: E402
    build_quant_config,
    load_model,
    load_tokenizer,
)
from sfzy.models.lora import inject_lora, mark_only_lora_trainable  # noqa: E402

MESSAGES = [
    {"role": "system", "content": "你是一名裁判文书摘要编辑。"},
    {"role": "user", "content": "原告张三诉被告李四借款合同纠纷一案。" * 20},
]


def report_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        resv = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        parts.append(f"卡{i}: 已分配 {alloc:.2f}G / 缓冲 {resv:.2f}G / 共 {total:.1f}G")
    print(f"      [{tag}] " + "   ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="显存与设备分布诊断")
    parser.add_argument("--config", default="configs/model_chatglm3_6b.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1536)
    args = parser.parse_args()

    model_cfg = load_config(ROOT / args.config).get("model", {})
    device_map = model_cfg.get("device_map")
    print(f"配置里的 device_map = {device_map!r}")

    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, quant_config=build_quant_config(model_cfg),
                       gradient_checkpointing=True)

    # ---------- 1. 模型落在几张卡上 ----------
    print("\n[1] 参数分布在哪些设备上")
    devices = Counter(str(p.device) for p in model.parameters())
    per_device_bytes = Counter()
    for p in model.parameters():
        per_device_bytes[str(p.device)] += p.numel()
    for dev, n in devices.most_common():
        print(f"      {dev:12s} {n:>4} 个参数张量，{per_device_bytes[dev]/1e9:.2f} B 元素")
    if len(devices) > 1:
        print("      ⚠️  模型跨了多张卡。训练时激活/梯度/优化器状态会把它们顶爆 ——")
        print("         auto 只按「权重能不能放下」分配，不含训练开销。")
        print('         修法：configs 里改成 device_map: {"": 0}')

    # ---------- 2. 梯度检查点真的开了吗 ----------
    print("\n[2] 梯度检查点是否真的生效")
    flag = getattr(model, "gradient_checkpointing", None)
    print(f"      顶层 model.gradient_checkpointing = {flag}")
    inner = [(n, m) for n, m in model.named_modules()
             if hasattr(m, "gradient_checkpointing")]
    enabled = [n for n, m in inner if m.gradient_checkpointing]
    print(f"      带这个标志的模块 {len(inner)} 个，其中已开启 {len(enabled)} 个")
    if not enabled:
        print("      ✗ 没有任何模块开启 —— 配置里写了 gradient_checkpointing: true")
        print("        但远程代码没接上。这时显存会按「全量激活」算，比预期大一倍以上。")
        print("        可以试试在 load_model 之后手动调 model.gradient_checkpointing_enable()")
    else:
        print(f"      示例：{enabled[:3]}")
    print(f"      use_cache = {getattr(model.config, 'use_cache', '（无此字段）')}")

    # ---------- 3. 一次前向 + 反向往多少显存 ----------
    print("\n[3] 一次 forward + backward 的显存开销")
    inject_lora(model, target_modules=load_config(ROOT / "configs/sft_kaggle.yaml")
                .path_("lora.target_modules", []), r=16, alpha=32)
    mark_only_lora_trainable(model)
    report_memory("模型加载后")

    collator = SFTCollator(tokenizer, max_length=args.seq_len)
    batch = collator([{
        "id": f"mem{i}",
        "messages": MESSAGES,
        "answer": "被告应于本判决生效之日起十日内归还原告借款。",
    } for i in range(args.batch_size)])
    batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
    print(f"      batch 形状 {tuple(batch['input_ids'].shape)}")
    report_memory("准备 batch 后")

    model.train()
    out = model(**batch)
    report_memory("forward 后")
    out.loss.backward()
    report_memory("backward 后")

    print("\n完成。把上面三段输出发出来，就能定位显存到底花在哪。")


if __name__ == "__main__":
    main()
