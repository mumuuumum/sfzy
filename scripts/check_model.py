"""训练前自检：能不能加载、层名对不对、模板能不能生成、**反向能不能走通**。

**在 Kaggle 上换了环境之后第一件事就跑这个。** 它不需要 GPU 训练配额，
却能在开跑之前暴露绝大部分"模型相关但和训练逻辑无关"的问题。

为什么要有反向那一步：我们实际踩过的两个坑 —— LoRA 分支的设备不一致、
fp16 主权重配 GradScaler —— **都只在反向传播时才炸**。只测前向的话
要等到训练跑起来才发现，而那时已经烧了配额。

用法：
    python scripts/check_model.py --config configs/model_chatglm3_6b.yaml
    python scripts/check_model.py --config configs/model_debug_small.yaml \
        --target-modules q_proj,k_proj,v_proj,o_proj
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch                                              # noqa: E402

from sfzy.config import load_config                       # noqa: E402
from sfzy.models.loader import (                          # noqa: E402
    build_quant_config,
    load_model,
    load_tokenizer,
)
from sfzy.models.lora import inject_lora, mark_only_lora_trainable  # noqa: E402

DEMO_MESSAGES = [
    {"role": "system", "content": "你是一名裁判文书摘要编辑。"},
    {"role": "user", "content": "原告张三诉被告李四借款合同纠纷一案。判令被告归还借款。"},
]

PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes", "peft", "datasets")


def main() -> None:
    parser = argparse.ArgumentParser(description="模型连通性自检")
    parser.add_argument("--config", default="configs/model_chatglm3_6b.yaml")
    parser.add_argument("--target-modules", default=None,
                        help="覆盖配置里的 LoRA 目标层，逗号分隔")
    args = parser.parse_args()

    model_cfg = load_config(ROOT / args.config).get("model", {})
    name = model_cfg.get("model_name_or_path")

    print("[1/7] 环境版本")
    for pkg in PACKAGES:
        try:
            print(f"      {pkg:16s} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"      {pkg:16s} ✗ 未安装")
    print(f"      cuda available   {torch.cuda.is_available()}")

    print(f"[2/7] 加载 tokenizer: {name}")
    tokenizer = load_tokenizer(model_cfg)
    print(f"      pad_token_id={tokenizer.pad_token_id}  "
          f"eos_token_id={tokenizer.eos_token_id}  "
          f"padding_side={tokenizer.padding_side}")

    print("[3/7] 渲染 chat template")
    try:
        text = tokenizer.apply_chat_template(DEMO_MESSAGES, tokenize=False,
                                             add_generation_prompt=True)
    except Exception as exc:  # noqa: BLE001
        print(f"      ✗ apply_chat_template 失败：{type(exc).__name__}: {exc}")
        print("      ChatGLM3 可能需要退回它的 build_chat_input，见 models/chat_template.py")
        raise SystemExit(1)
    print(f"      渲染结果（前 200 字）：\n{text[:200]}")

    print("[4/7] 加载模型")
    quant_config = build_quant_config(model_cfg)
    # **必须和训练用同一个梯度检查点设置。**
    # 早先这里写死 False，于是"配置要求开、实际没开"这种情况结构上就不可能被发现
    # —— 结果一直等到 Kaggle 上训练 OOM 才暴露，而报错栈里全是 bitsandbytes。
    use_ckpt = load_config(ROOT / "configs/sft_kaggle.yaml").path_(
        "sft.gradient_checkpointing", False
    )
    model = load_model(model_cfg, quant_config=quant_config,
                       gradient_checkpointing=use_ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量 {n_params/1e9:.2f} B  dtype={next(model.parameters()).dtype}")
    from sfzy.models.compat import describe_gradient_checkpointing

    gc = describe_gradient_checkpointing(model)
    print(f"      梯度检查点: 配置={use_ckpt}")
    for key, value in gc.items():
        print(f"        {key:24s} = {value}")
    if use_ckpt and gc["modules_with_flag"] == 0:
        print("      ✗ 配置要求开梯度检查点，但一个模块都没开")
        print("        训练时激活值会按「每层完整保存」算，几乎必然 OOM")
        raise SystemExit(1)

    print("[5/7] 检查 LoRA 目标层是否存在")
    targets = (args.target_modules.split(",") if args.target_modules
               else load_config(ROOT / "configs/sft_kaggle.yaml").path_("lora.target_modules", []))
    replaced = inject_lora(model, target_modules=targets, r=8, alpha=16)
    if replaced == 0:
        print(f"      ✗ target_modules={targets} 一个都没匹配上")
        print("      实际存在的线性层名（部分）：")
        for n, _ in list(model.named_modules())[:400]:
            if n.endswith(("query_key_value", "q_proj", "k_proj", "v_proj", "o_proj", "dense")):
                print(f"        {n}")
        raise SystemExit(1)
    mark_only_lora_trainable(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      匹配 {replaced} 层，可训练参数 {trainable/1e6:.2f} M "
          f"（占比 {trainable/n_params:.3%}）")

    device = next(model.parameters()).device

    print("[6/7] 走一遍我们自己的 collator（训练真正用的路径）")
    # 前几步调的都是 tokenizer / 模型的**原生接口**，这一步才走我们自己的代码。
    # 实测踩过一次：ChatGLM3 的 apply_chat_template(tokenize=True) 返回
    # BatchEncoding，而 Qwen 返回 list —— 只有到了 collator 里做
    # `prompt_ids + answer_ids` 那一刻才报 TypeError。前几步完全看不出来。
    from sfzy.data.collator import SFTCollator

    collator = SFTCollator(tokenizer, max_length=256, pad_to_multiple_of=8)
    batch = collator([{
        "id": "check",
        "messages": DEMO_MESSAGES,
        "answer": "被告应于本判决生效之日起十日内归还原告借款。",
    }])
    labels = batch["labels"][0].tolist()
    masked = sum(1 for x in labels if x == -100)
    valid = sum(1 for x in labels if x != -100)
    print(f"      input_ids {tuple(batch['input_ids'].shape)}  "
          f"prompt 段 mask 掉 {masked} 个 token，答案段 {valid} 个")
    if masked == 0:
        print("      ✗ prompt 段没有被 mask —— collator 的 loss mask 有问题")
        raise SystemExit(1)
    if valid == 0:
        print("      ✗ 答案段全被 mask 掉了 —— 模型永远学不会生成")
        raise SystemExit(1)

    print("[7/7] 前向 + 反向一次，确认梯度能传下去")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    model.train()
    # 传 labels 让模型内部算 loss —— 和训练时走的是同一条路
    outputs = model(**inputs, labels=inputs["input_ids"])
    print(f"      logits 形状 {tuple(outputs.logits.shape)}  loss={outputs.loss.item():.4f}")

    outputs.loss.backward()
    total = sum(1 for p in model.parameters() if p.requires_grad)
    with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    nonzero = sum(1 for p in model.parameters()
                  if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    print(f"      可训练参数 {total} 个，拿到梯度 {with_grad} 个，梯度非零 {nonzero} 个")
    if with_grad == 0:
        print("      ✗ 没有任何参数拿到梯度 —— 检查 loss mask 或 requires_grad")
        raise SystemExit(1)
    if nonzero < total:
        print(
            f"      提示：有 {total - nonzero} 个参数梯度为 0。LoRA 场景下这是**正常的** ——"
            " lora_B 零初始化，所以 ∂L/∂lora_A = Bᵀ·… = 0。"
            "走一步更新后 A 才会有梯度信号。"
        )

    print("\n✓ 七步全部通过。可以开始训练了。")


if __name__ == "__main__":
    main()
