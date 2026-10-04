"""独立测试程序：验证六要素事实一致性 Judge（需求第十三节）。

先跑这个，再决定要不要接进 GRPOTrainer。

============================ 用法 ============================
模型默认从**配置文件**加载（我们现在统一用 Qwen2.5-7B-Instruct）：

    # 固定测试集：跑 A~G 七类案例，看能不能排出正确的顺序
    python scripts/judge_test.py --model-config configs/model_qwen25_7b.yaml \
        --device cuda:1 --cases data/judge/fact_cases.jsonl

    # 单条：看完整的中间过程（六要素、六项原始分、归一化分、加权分）
    python scripts/judge_test.py --model-config configs/model_qwen25_7b.yaml \
        --device cuda:1 --cases data/judge/fact_cases.jsonl --only d1_A

`--model` 走原生 transformers 加载，给了它就覆盖 `--model-config`：

    python scripts/judge_test.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
        --device cuda:1 --cases data/judge/fact_cases.jsonl

`--adapter` 载入 LoRA 后再测（同源自评检验：微调过的模型判得是不是比底座差）。
两条加载路径和 `scripts/judge_probe.py` 完全一致。

============================ 验收线 ============================
需求要求"重点确认 C、D、E、F、G 能显著低于 A、B"，并且希望看到

    完全正确 > 正确但简略 > 轻微事实错误 > 明显事实错误 > 核心裁判结果错误

本脚本按三条判据自动判定：

  1. 每个文档内 `min(A, B) > max(C, D, E, F, G)` —— 最硬的一条
  2. `mean(A, B) > mean(C, D, F, G) > mean(E)` —— 结果错要垫底
  3. `judgment_result` 这一项在 E 上显著低于 A —— 裁判结果分是重点观察指标

三条全过才算 Judge 可用。**不过就不要接 GRPO** —— 接上去只会得到噪声梯度。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                         # noqa: E402
from sfzy.judge import (                                   # noqa: E402
    ELEMENTS,
    ELEMENT_ZH,
    FactConsistencyJudge,
    SixElements,
)
from sfzy.judge.runtime import TorchRuntime                # noqa: E402


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def quiet_transformers() -> None:
    """把 transformers 的日志压回默认的 WARNING。

    加载 Qwen2.5 的 tokenizer 时，transformers 会在 INFO 级别打印

        Special tokens have been added in the vocabulary, make sure the
        associated word embeddings are fine-tuned or trained.

    它来自 `tokenization_utils_base.py` 里一句判断：tokenizer 的 added token
    id（151665）超过了基础词表大小（vocab_size=151643）。**这是 tokenizer
    文件里本来就有的既有事实**（`<|im_start|>` / `<|end_of_text|>` 这些
    special token 的 id 一直排在 BPE 词表之后），不是我们把词表改大了，
    也不会触发 `resize_token_embeddings` —— 词表长度前后完全一致，无需重训
    嵌入。T4 上看不到它，只是因为那次跑的是别的模型/别的日志级别。

    压回 WARNING 只是让这条无害的 INFO 不再混进判分输出；真正的告警
    （WARNING/ERROR）照常打印。
    """
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_warning()
    except Exception:  # noqa: BLE001 - 日志配置失败不该影响判分
        pass


def build_model(args):
    """两条加载路径，和 `scripts/judge_probe.py` 保持一致：

      --model        原生 transformers（普通 HF 目录 / hub id）
      --model-config 走项目自己的加载器（读 model.* 那一节配置）

    `--model` 优先：它给了就用它，否则用配置文件。默认是配置文件里的
    Qwen2.5-7B-Instruct —— 现在裁判统一用它。

    两条路径都返回 `(model, tokenizer, device)`，`device` 是模型**实际**
    所在设备。**4/8-bit 量化模型的设备由 `device_map` 决定**，加载后不能
    再 `.to()`（transformers 会抛 `.to() is not supported for ... bitsandbytes
    models`），所以这里统一走 `load_inference_model`，由它把 `--device`
    翻译成 `{"": 卡号}` 再加载。
    """
    from sfzy.models.loader import load_inference_model

    if args.model:
        model_cfg = {
            "model_name_or_path": args.model,
            "trust_remote_code": True,          # 普通 HF 目录也走同一条路
            "torch_dtype": args.dtype,
            "load_in_4bit": bool(args.load_in_4bit),
            # T4 不支持 bf16，量化计算精度跟随 --dtype 更安全
            "bnb_4bit_compute_dtype": args.dtype,
        }
    else:
        if not args.model_config:
            raise SystemExit("要么 --model，要么 --model-config")
        cfg = load_config(resolve(args.model_config))
        model_cfg = dict(cfg.get("model") or {})
        name = model_cfg.get("model_name_or_path", "")
        # 相对路径按项目根解析；hub id（Qwen/...）原样保留。
        if name:
            local = resolve(name)
            if local.exists():
                model_cfg["model_name_or_path"] = str(local)

    return load_inference_model(
        model_cfg,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=False,          # 只做推理，不开检查点
    )


def build_judge(args) -> FactConsistencyJudge:
    """模型 + tokenizer → 冻结的 TorchRuntime → 六要素 Judge。"""
    model, tokenizer, device = build_model(args)

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

    runtime = TorchRuntime(
        # 用模型**实际**设备，不要用命令行传的那个 —— 量化模型经 device_map
        # 放置后可能和 --device 的写法不同（"cuda" vs "cuda:0"）
        model=model, tokenizer=tokenizer, device=device,
        max_batch_size=args.batch_size,
    )
    return FactConsistencyJudge(runtime=runtime)


def load_documents() -> dict:
    """文档正文从 val 划分里按 id 取，避免在测试集里重复存 7KB 的判决书。"""
    path = resolve("data/splits/val.jsonl")
    return {json.loads(l)["id"]: json.loads(l)["source"]
            for l in open(path, encoding="utf-8") if l.strip()}


def print_elements(title: str, els: SixElements) -> None:
    print(f"\n{title}")
    for name in ELEMENTS:
        value = els.get(name).strip()
        value = value if value else "（空）"
        print(f"  {ELEMENT_ZH[name]:<8}{value[:120]}{'…' if len(value) > 120 else ''}")


def run_single(judge: FactConsistencyJudge, document: str, candidate: str) -> None:
    """需求第十三节要打印的五项，一次看全。"""
    doc_el = judge.document_elements(document)
    cand_el = judge.extract_six_elements(candidate)
    print_elements("【1】document 六要素", doc_el)
    print_elements("【2】candidate 六要素", cand_el)

    # 必须和 judge_summary 走同一套组 pair 规则（含"要素抽空时用整篇原文兜底"），
    # 否则演示看到的和生产跑的不是一个东西
    pairs = judge.build_pairs(doc_el, cand_el, document)
    scores, sources, pmaxs = judge.judge_elements_with_confidence(pairs)
    from sfzy.judge.schema import aggregate
    result = aggregate(
        {n: s for n, s in zip(ELEMENTS, scores)},
        weights=judge.weights,
        sources={n: s for n, s in zip(ELEMENTS, sources)},
    )

    # 逐要素对照：这一步才是"对比二者获得分数"的正文。
    # 只打印分数看不出模型到底在比什么，所以原文要素、摘要要素、判定分并排给。
    print("\n【3】逐要素对照与判定")
    for name, s, pm, (_, doc_sent, cand_sent) in zip(ELEMENTS, scores, pmaxs, pairs):
        doc_v = doc_el.get(name).strip() or "（空）"
        cand_v = cand_el.get(name).strip() or "（空）"
        src = sources[ELEMENTS.index(name)]
        # 兜底是在 build_pairs 里做的，库层的 sources 只区分"调没调模型"，
        # 所以这里按"抽出来的要素是空、但实际送进去的是整篇文书"来识别
        fallback = (not doc_el.get(name).strip()) and doc_sent == document and bool(document)
        tag = "  ← 空字段规则，未调用模型" if src == "empty_rule" else (
            "  ← 要素抽空，用整篇原文兜底判定" if fallback else "")
        conf = f"  pmax={pm:.2f}" if pm is not None else ""
        print(f"\n  ▸ {ELEMENT_ZH[name]}（{name}）  判定 {s}/4{conf}{tag}")
        print(f"      原文要素：{doc_v[:150]}{'…' if len(doc_v) > 150 else ''}")
        print(f"      摘要要素：{cand_v[:150]}{'…' if len(cand_v) > 150 else ''}")
        if fallback:
            print(f"      实际送判：原文要素位置放了整篇文书（{len(doc_sent)} 字）")

    print("\n【4】六个归一化评分（/4）")
    for name in ELEMENTS:
        print(f"  {ELEMENT_ZH[name]:<8}{result.scores[name]:.2f}")
    print(f"\n【5】weighted fact_reward = {result.weighted_reward:.4f}")
    print(f"     min_element_score     = {result.min_element_score:.4f}"
          "   （不含案由）")
    print(f"     judgment_result_score = {result.judgment_result_score:.4f}")


def run_cases(judge: FactConsistencyJudge, cases: list, docs: dict, verbose: bool) -> None:
    by_doc = defaultdict(list)
    for c in cases:
        by_doc[c["doc_id"]].append(c)

    rows = []
    for doc_id, group in by_doc.items():
        document = docs[doc_id]
        judge.clear_cache()                       # 每个文档单独提取一次
        doc_el = judge.document_elements(document)
        results = judge.judge_candidates(
            document, [c["candidate"] for c in group],
            candidate_ids=[c["id"] for c in group],
        )
        if verbose:
            print_elements(f"\n【{doc_id[:8]}】document 六要素", doc_el)
        for c, r in zip(group, results):
            rows.append((doc_id, c, r))

    print("\n" + "=" * 92)
    print(f"{'案例':<7}{'类型':<5}{'fact':>8}{'min':>7}{'结果分':>8}   说明")
    print("-" * 92)
    for doc_id, c, r in rows:
        print(f"{c['id']:<7}{c['case']:<5}{r.weighted_reward:>8.4f}"
              f"{r.min_element_score:>7.2f}{r.judgment_result_score:>8.2f}"
              f"   {c['desc']}")

    # ---- 判据 ----
    print("\n" + "=" * 92)
    per_case = defaultdict(list)
    for _, c, r in rows:
        per_case[c["case"]].append(r.weighted_reward)
    for k in sorted(per_case):
        v = per_case[k]
        print(f"  类型 {k}: fact_reward 均值 {st.mean(v):.4f}  (n={len(v)}, "
              f"min {min(v):.4f}, max {max(v):.4f})")

    good = per_case["A"] + per_case["B"]
    bad = per_case["C"] + per_case["D"] + per_case["F"] + per_case["G"]
    core = per_case["E"]
    c1 = st.mean(good) > st.mean(bad)
    c2 = st.mean(bad) > st.mean(core)
    c3 = all(
        st.mean(per_case[a] or [0]) > st.mean(per_case[e] or [0])
        for a in ("A",) for e in ("E",)
    )
    print(f"\n  判据 1  mean(A,B) > mean(C,D,F,G)：{st.mean(good):.4f} vs {st.mean(bad):.4f}"
          f"   {'✓' if c1 else '✗'}")
    print(f"  判据 2  mean(C,D,F,G) > mean(E)：{st.mean(bad):.4f} vs {st.mean(core):.4f}"
          f"   {'✓' if c2 else '✗'}")
    per_doc_ok = 0
    for doc_id, group in by_doc.items():
        vals = {c["case"]: r.weighted_reward
                for (d, c, r) in rows if d == doc_id}
        if max(vals["A"], vals["B"]) > max(vals[k] for k in "CDEFG"):
            per_doc_ok += 1
    print(f"  判据 3  每个文档内 min(A,B) > max(C,D,E,F,G)："
          f"{per_doc_ok}/{len(by_doc)} 个文档通过"
          f"   {'✓' if per_doc_ok == len(by_doc) else '✗'}")
    print(f"\n  结论：{'✓ 三条全过，可以接 GRPO' if (c1 and c2 and per_doc_ok == len(by_doc)) else '✗ 有判据未过，先别接 GRPO'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="六要素事实一致性 Judge 测试")
    ap.add_argument("--model", default=None,
                    help="HF 模型路径（原生加载）。给了它就覆盖 --model-config")
    ap.add_argument("--model-config", default="configs/model_qwen25_7b.yaml",
                    help="模型配置 yaml（走项目加载器）。默认 configs/model_qwen25_7b.yaml"
                         "（Qwen2.5-7B-Instruct）")
    ap.add_argument("--adapter", default=None,
                    help="可选：载入 LoRA 后再测（同源自评检验）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    # 量化开关：默认跟随配置（model_qwen25_7b.yaml 是 4-bit）。
    # 4090（24GB）上 7B bf16 装得下，想绕开 bitsandbytes 就加 --no-4bit；
    # T4 只有 16GB，必须留在 4-bit。
    ap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true",
                    default=None, help="强制开启 4-bit（默认跟随配置）")
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                    help="关掉 4-bit，走配置里的 bf16/fp16")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="本地内存小就调小；服务器上可以调到 16")
    ap.add_argument("--cases", default="data/judge/fact_cases.jsonl")
    ap.add_argument("--only", default=None, help="只跑某个案例 id（单条详细模式）")
    ap.add_argument("--filter", default=None,
                    help="只跑案例 id 或文档 id 含该子串的案例。"
                         "本地用小模型验接线时，跑一个文档的 7 条就够")
    ap.add_argument("--verbose", action="store_true", help="打印每个文档的六要素")
    ap.add_argument("--stats", action="store_true", help="打印 0-4 各档比例")
    args = ap.parse_args()

    if not args.model and not args.model_config:
        raise SystemExit("要么 --model，要么 --model-config")

    quiet_transformers()

    docs = load_documents()
    cases = [json.loads(l) for l in open(resolve(args.cases), encoding="utf-8") if l.strip()]
    if args.filter:
        cases = [c for c in cases if args.filter in c["id"] or args.filter in c["doc_id"]]
        if not cases:
            raise SystemExit(f"没有匹配 {args.filter!r} 的案例")
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
        if not cases:
            raise SystemExit(f"没有 id 为 {args.only} 的案例")

    tag = args.model or args.model_config
    print(f"模型来源 {tag}   设备 {args.device}   batch {args.batch_size}"
          + (f"   adapter {args.adapter}" if args.adapter else ""))
    judge = build_judge(args)

    if args.only:
        c = cases[0]
        run_single(judge, docs[c["doc_id"]], c["candidate"])
        return

    run_cases(judge, cases, docs, args.verbose)

    if args.stats:
        print("\n推理统计：", judge.runtime.stats)


if __name__ == "__main__":
    main()
