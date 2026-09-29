"""判定一个模型能不能当事实一致性裁判 —— 10 条探针，2 分钟出结论。

============================ 为什么单独做这个 ============================
实测 Qwen2.5-0.5B 在同一个判定上**无论输入什么都吐同一个值**：

    探针        期望   规范 prompt   少样本 prompt
    完全相同     4       4            3
    只省略      4       4            3
    金额写错     1       4            3      ← 完全没看出来
    结果反转     0       4            3      ← 完全没看出来

常量输出 = 没有判别力 = 奖励里加进去只会是噪声。这种事必须**在写进奖励之前**
用几分钟测出来，而不是跑完 13 小时 GRPO 之后。

探针是**要素级**的（直接给"原文要素 / 摘要要素"），绕开六要素提取器 ——
这样测的纯粹是"判定能力"，不受提取质量干扰。提取质量由 judge_test.py 测。

============================ 判据 ============================
  * 好的四条（完全相同 / 只省略 / 合理概括 / 换词改写）**都要 ≥ 3**
  * 坏的六条（金额错 / 主体颠倒 / 结果反转 / 编造 / 辩称当查明 / 案由错）
    **都要 ≤ 1**
  * 两组均值差 ≥ 2.0

任何一条不满足就不该拿它当奖励。

============================ 用法 ============================
    # ChatGLM3 底座（走项目自己的兼容层，不加载 adapter）
    python scripts/judge_probe.py --model-config configs/model_chatglm3_6b_bf16.yaml \
        --device cuda

    # Qwen2.5-7B（卡 1），普通 HF 路径
    python scripts/judge_probe.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
        --device cuda:1

    # 加上 SFT adapter，看看"微调过的模型"判得是不是比底座差（同源自评检验）
    python scripts/judge_probe.py --model-config configs/model_chatglm3_6b_bf16.yaml \
        --adapter outputs/sft_chatglm3/best.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                     # noqa: E402
from sfzy.judge.judge import FactConsistencyJudge        # noqa: E402
from sfzy.judge.runtime import TorchRuntime              # noqa: E402


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_model(args):
    """两条加载路径：项目自己的（带 ChatGLM3 兼容补丁）/ 原生 transformers。"""
    if args.model_config:
        from sfzy.models.loader import build_quant_config, load_model, load_tokenizer

        cfg = load_config(resolve(args.model_config))
        model_cfg = dict(cfg.get("model") or {})
        name = model_cfg.get("model_name_or_path", "")
        local = resolve(name)
        if local.exists():
            model_cfg["model_name_or_path"] = str(local)
        tokenizer = load_tokenizer(model_cfg)
        model = load_model(
            model_cfg,
            quant_config=build_quant_config(model_cfg),
            gradient_checkpointing=False,      # 只做推理，不开检查点
        )
        if model_cfg.get("device_map") in (None, "", "none"):
            model = model.to(args.device)
        return model, tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device)
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="事实一致性裁判的 10 条探针")
    ap.add_argument("--model", default=None, help="HF 模型路径（原生加载）")
    ap.add_argument("--model-config", default=None,
                    help="模型配置 yaml（走项目加载器，ChatGLM3 必须用这个）")
    ap.add_argument("--adapter", default=None, help="可选：载入 LoRA 后再测（同源自评检验）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--probes", default="data/judge/probe_cases.jsonl")
    ap.add_argument("--judge-variant", default="spec", choices=["spec", "fewshot"],
                    help="判定 Prompt 版本。小模型上先试 fewshot")
    args = ap.parse_args()

    if not args.model and not args.model_config:
        raise SystemExit("要么 --model，要么 --model-config")

    tag = args.model_config or args.model
    print(f"模型来源 {tag}   设备 {args.device}" + (f"   adapter {args.adapter}" if args.adapter else ""))
    model, tokenizer = build_model(args)

    if args.adapter:
        from sfzy.models.lora import inject_lora, mark_only_lora_trainable
        from sfzy.sft.checkpoint import load_checkpoint

        cfg = load_config(resolve(args.model_config)) if args.model_config else {}
        lora_cfg = (cfg.get("lora") or {}) if cfg else {}
        inject_lora(
            model,
            target_modules=lora_cfg.get("target_modules", []),
            r=lora_cfg.get("r", 8), alpha=lora_cfg.get("alpha", 16),
            dropout=0.0,
        )
        mark_only_lora_trainable(model)
        load_checkpoint(resolve(args.adapter), model=model)
        print("已载入 adapter —— 测的是微调过的模型")

    runtime = TorchRuntime(model=model, tokenizer=tokenizer, device=args.device)
    judge = FactConsistencyJudge(runtime=runtime, judge_variant=args.judge_variant)

    probes = [json.loads(l) for l in open(resolve(args.probes), encoding="utf-8") if l.strip()]
    pairs = [(p["element"], p["doc"], p["cand"]) for p in probes]
    scores, sources, pmaxs = judge.judge_elements_with_confidence(pairs)

    # ================= 修改后的打印逻辑 =================
    print(f"\n{'探针':<14}{'期望':<8}{'得分':>5}{'pmax':>8}   说明")
    print("-" * 78)
    good, bad, fails = [], [], []
    
    # 将 sources 也加入 zip 循环
    for p, s, src, pm in zip(probes, scores, sources, pmaxs):
        ok = s >= 3 if p["good"] else s <= 1
        (good if p["good"] else bad).append(s)
        if not ok:
            fails.append(p["id"])
        conf = f"{pm:.2f}" if pm is not None else "  - "
        print(f"{p['id']:<14}{'≥3' if p['good'] else '≤1':<8}{s:>5}{conf:>8}   "
              f"{'✓' if ok else '✗'} {p['note']}")

    # ================= 新增：详细推理过程（人类审查视角） =================
    print("\n" + "="*30 + " 详细推理过程 " + "="*30)
    for p, s, src, pm in zip(probes, scores, sources, pmaxs):
        print(f"\n【{p['id']}】{p['note']}")
        print(f"  ▶ 原文要素 (doc)   : {p['doc']}")
        print(f"  ▶ 摘要要素 (cand)  : {p['cand']}")
        
        # 处理 src 可能是 dict、list 或 None 的情况，确保美观打印
        if isinstance(src, (dict, list)):
            import json
            src_str = json.dumps(src, ensure_ascii=False)
        else:
            src_str = str(src).replace("\n", " ⏎ ") # 把换行符替换成可见符号，方便单行阅读
            
        print(f"  ▶ 模型原始输出 (src): {src_str}")
        print(f"  ▶ 解析得分 / 置信度 : {s} / {pm:.2f}" if pm is not None else f"  ▶ 解析得分 / 置信度 : {s} / -")
        print("-" * 60)
    # ==============================================================

    gap = st.mean(good) - st.mean(bad) if good and bad else 0.0
    print(f"\n好的四条均值 {st.mean(good):.2f}   坏的六条均值 {st.mean(bad):.2f}   "
          f"差值 {gap:.2f}（要求 ≥2.0）")
    print(f"不达标探针：{fails if fails else '无'}")
    verdict = not fails and gap >= 2.0
    print(f"\n结论：{'✓ 可以当裁判' if verdict else '✗ 不能当裁判，判别力不足'}")
    if not verdict:
        print("  表现是常量输出的话，问题在模型容量，不在提示词 —— 换更大的模型。")
    print("\n推理统计：", runtime.stats, " 提取统计：", judge.stats)

if __name__ == "__main__":
    main()
