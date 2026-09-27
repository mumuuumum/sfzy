"""显存与设备分布诊断。

**不要猜，测。** 这个脚本回答三个问题：

  1. 模型到底落在几张卡上？（`device_map` 生效了吗）
  2. 梯度检查点真的开启了吗？（在 ChatGLM3 这种远程代码模型上不一定生效）
  3. 一次 forward+backward 之后，每张卡实际占了多少？

用法（Kaggle 上直接跑，不需要训练配额）：
    python scripts/check_memory.py --config configs/model_chatglm3_6b.yaml \\
        --batch-size 1 --seq-len 1536

**要把 --seq-len 设成真实长度，否则量出来的数字没有意义。**
train.jsonl 的 token 长度分布是 p50 1893 / p95 3487 / p99 4741，
所以要按 p95 甚至 p99 来量：

    python scripts/check_memory.py --batch-size 2 --seq-len 3500   # 常规批
    python scripts/check_memory.py --batch-size 2 --seq-len 4700   # 长样本批
    python scripts/check_memory.py --batch-size 4 --seq-len 2000   # 试 batch=4

判据：**只要"backward 后"的余量小于 1 GiB，就不要用这个 batch** ——
一个 batch 里最长的那条（p99）会再多吃一截，OOM 会挑长样本批发作，
而 9 小时会话里你必然会遇到长样本批。
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

SYSTEM = "你是一名裁判文书摘要编辑。"
ANSWER = "被告应于本判决生效之日起十日内归还原告借款本金十万元及利息。"


def build_messages(tokenizer: Any, target_tokens: int) -> list:
    """造一条「长度可控」的样本，把 user 内容凑到约 target_tokens 个 token。

    为什么要按 **token** 数而不是字符数来凑：显存跟 token 数成正比，而中文在
    ChatGLM3 的 sentencepiece 里大约是 1 字 = 0.6~0.7 token（实测 train.jsonl：
    平均 2600 字符的文书 ≈ 2047 token）。按字符凑会系统性偏短，
    量出来的余量会偏乐观。
    """
    unit = (
        "原告张三诉被告李四借款合同纠纷一案，本院于2020年3月1日立案后，"
        "依法适用简易程序，公开开庭进行了审理。原告请求判令被告归还借款本金"
        "十万元及利息。被告辩称双方之间不存在借贷关系。本院经审理查明以下事实："
    )
    content = ""
    while len(tokenizer(content, add_special_tokens=False)["input_ids"]) < target_tokens:
        content += unit
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content + "\n请生成摘要："},
    ]


def report_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        resv = torch.cuda.memory_reserved(i) / 1024**3
        peak = torch.cuda.max_memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        parts.append(
            f"卡{i}: 当前 {alloc:.2f}G / 峰值 {peak:.2f}G / 缓冲 {resv:.2f}G "
            f"/ 共 {total:.1f}G（余量 {total - peak:.2f}G）"
        )
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
    # LoRA 的超参从 sft_kaggle.yaml 读，避免"诊断脚本用 r=16、训练用 r=4"。
    sft_cfg = load_config(ROOT / "configs/sft_kaggle.yaml")
    lora_cfg = sft_cfg.get("lora", {})
    print("\n[3] 一次 forward + backward 的显存开销")
    replaced = inject_lora(
        model,
        target_modules=lora_cfg.get("target_modules", []),
        r=lora_cfg.get("r", 8),
        alpha=lora_cfg.get("alpha", 16),
    )
    mark_only_lora_trainable(model)
    print(f"      注入 LoRA: {replaced} 层（r={lora_cfg.get('r')}, "
          f"alpha={lora_cfg.get('alpha')}, targets={lora_cfg.get('target_modules')}）")
    report_memory("模型加载后")

    # 从这一行开始，"峰值"才算训练步的峰值。
    # 不重置的话，加载 12GB 权重时的瞬时峰值会混进来，把结论带偏 ——
    # 那是"能不能装下模型"的问题，和"batch 能开多大"是两件事。
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    print("      （已重置峰值统计，下面的峰值 = 本次 forward+backward 的峰值）")

    collator = SFTCollator(tokenizer, max_length=args.seq_len)
    messages = build_messages(tokenizer, args.seq_len)
    batch = collator([{
        "id": f"mem{i}",
        "messages": messages,
        "answer": ANSWER,
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
