"""筛选 GRPO 的 prompt 池。

============================ 为什么必须筛 ============================
实测（335 条验证集抽样）：

    参考摘要里的事实个数
      0 个   60.0%   ← 六成的参考摘要没有任何可提取的事实
      1-2 个 23.9%
      >=3 个 16.1%

**事实覆盖这个奖励只对 40% 的样本有信号。** 剩下 60% 的参考摘要里根本
没有金额/日期可提取——不是提取器的问题，是标注风格（比如"判令被告支付
拖欠的租金"压根不写金额）。

这本身对 GRPO 无害：组内所有采样共享同一个参考，"有没有事实"是组内
常数，组内归一化时会抵消。**但随机抽 prompt 的话，60% 的算力花在事实项
恒为 0 的组上**，那些组的梯度只由 ROUGE 决定——等于白跑。

============================ 怎么筛 ============================
按"参考里的事实个数"降序排，取前 N 条。默认只保留含金额的样本
（金额占 35.2%，是三种事实里信号最强的）。

用法：
    python tools/select_rl_prompts.py --n 1000
    python tools/select_rl_prompts.py --n 1000 --kinds money date id

要拿"全体训练集"当池子（不按事实筛）时，直接喂三元组文件 —— 它自带人工摘要
（`reference`）和 SFT 输出（`output`，正好给基线锚用）：

    python tools/select_rl_prompts.py --all \
        --input data/judge/sft_train_shard*.jsonl \
        --out data/splits/rl_prompts_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config               # noqa: E402
from sfzy.rl.reward import extract_facts     # noqa: E402
from sfzy.utils.io import ensure_dir, write_jsonl  # noqa: E402


def _resolve(spec: str) -> Path:
    path = Path(spec)
    return path if path.is_absolute() else ROOT / path


def _reference_of(record: dict) -> str:
    """人工摘要的键名：train/val.jsonl 用 `summary`，三元组文件用 `reference`。"""
    return str(record.get("summary") or record.get("reference") or "")


def _load_records(paths) -> list:
    """读一个或多个 jsonl，按 id 去重（保留首次出现）。

    允许多文件是给"分片生成的三元组"用的：`generate_triples.py --shard i/N`
    按 `i % N` 切开，几个 shard 拼起来才是完整训练集，所以要能一次喂进来。
    """
    records: list = []
    seen: set = set()
    dupes = 0
    for spec in paths:
        path = _resolve(spec)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = str(record.get("id") or "")
                if key and key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                records.append(record)
        print(f"  读入 {path.name}：累计 {len(records)} 条")
    if dupes:
        print(f"  按 id 去重：跳过 {dupes} 条重复")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="筛选 GRPO 的 prompt 池")
    parser.add_argument("--input", nargs="+", default=["data/splits/train.jsonl"],
                        help="输入 jsonl，可以给多个（如 data/judge/sft_train_shard*.jsonl）。"
                             "两种键名都认：train.jsonl 的 summary，或三元组的 reference/output")
    parser.add_argument("--out", default="data/splits/rl_prompts.jsonl")
    parser.add_argument("--n", type=int, default=None,
                        help="选多少条；0 表示不设上限。不传时默认 1000（配 --all 则不设上限）")
    parser.add_argument("--all", action="store_true",
                        help="不按事实筛，输入里有多少条就用多少条（全体训练集）")
    parser.add_argument("--kinds", nargs="+", default=["money"],
                        help="按哪些事实类型筛。默认只看金额（信号最强、占比最高 35.2%%）")
    parser.add_argument("--shuffle", action="store_true",
                        help="选完之后打乱顺序（默认按事实个数降序，先练难的；--all 时保持输入顺序）")
    parser.add_argument("--triples", default=None,
                        help="SFT 三元组 jsonl（含 id/output）。给了就把 SFT 输出挂到 "
                             "每个 prompt 的 sft_output 字段上，训练时用来做基线锚。"
                             "输入自带 output 时不需要再传")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="按**真实 tokenizer** 丢掉 prompt 超过这个 token 数的样本。\n"
                             "  用途：给 GRPO 的 rl.max_length 设一个安全上限（训练侧显存 ∝ 序列宽度）。\n"
                             "  用 `--prompt-style` 决定拼模板的方式，与训练严格一致。")
    parser.add_argument("--model-config", default="configs/model_chatglm3_6b.yaml",
                        help="提供 tokenizer 的模型配置（读它的 model.model_name_or_path）")
    parser.add_argument("--prompt-style", default="structured",
                        help="与 rl.prompt_style 一致（zeroshot/structured/rag）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _load_records(args.input)
    print(f"输入 {len(records)} 条")

    # 事实个数只用来观察分布；--all 时它不参与"选哪些"
    from collections import Counter
    fact_counts = [len(extract_facts(_reference_of(r), kinds=args.kinds)) for r in records]
    with_facts = [c for c in fact_counts if c]
    dist = Counter(with_facts)
    print(f"含 {args.kinds} 类事实的 {len(with_facts)} 条 "
          f"= {len(with_facts)/max(len(records),1):.1%}")
    print("  事实个数分布: " + "  ".join(f"{k}个:{v}" for k, v in sorted(dist.items())[:6]))

    if args.all:
        # 全量模式：一条都不筛，保持输入顺序（多文件时按传入顺序拼接）
        selected = records[: args.n] if args.n else list(records)
        print(f"\n--all：不按事实筛，取 {len(selected)} 条")
    else:
        # 降序：先选事实最密集的，那是模型最可能失败的样本
        scored = sorted(
            ((c, r) for c, r in zip(fact_counts, records) if c), key=lambda x: -x[0]
        )
        cap = 1000 if args.n is None else args.n
        selected = [r for _, r in scored[: cap or None]]
        print(f"\n按事实个数降序取 {len(selected)} 条")

    if args.shuffle:
        random.Random(args.seed).shuffle(selected)

    # ---- 可选：按真实 tokenizer 丢掉超长 prompt（控制训练侧显存）----
    if args.max_prompt_tokens:
        import sys as _sys

        _sys.path.insert(0, str(ROOT / "src"))
        from sfzy.data.prompts import build_messages
        from sfzy.models.chat_template import encode_prompt
        from sfzy.models.loader import load_tokenizer

        model_cfg = load_config(_resolve(args.model_config)).get("model", {})
        tokenizer = load_tokenizer(model_cfg)
        kept, dropped = [], 0
        for r in selected:
            # 必须用 **source（文书原文）** 拼模板，与训练时 rollout 的输入完全一致
            messages = build_messages(r.get("source", ""), args.prompt_style)
            n_tokens = len(encode_prompt(tokenizer, messages, add_generation_prompt=True))
            if n_tokens > args.max_prompt_tokens:
                dropped += 1
            else:
                kept.append(r)
        print(f"\n按 prompt ≤ {args.max_prompt_tokens} token 过滤："
              f"丢掉 {dropped} 条（{dropped / max(len(selected), 1):.1%}），剩 {len(kept)} 条")
        selected = kept

    # ---- 挂上 SFT 输出（可选的基线锚要用）----
    by_id: dict = {}
    if args.triples:
        tpath = _resolve(args.triples)
        for line in open(tpath, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                by_id[r["id"]] = r["output"]

    hit = 0
    for r in selected:
        # 优先用 --triples 的映射，其次用输入自带的 output（三元组文件就是这个格式）
        out = by_id.get(str(r.get("id"))) or r.get("output") or r.get("sft_output")
        if out:
            r["sft_output"] = out
            hit += 1
    print(f"\n挂上 SFT 输出（基线锚）：{hit}/{len(selected)} 条")
    if args.triples and hit < len(selected):
        print("  缺的那些在训练时不会被锚过滤（等价于不启用），不会报错。")
        print("  ⚠ 常见原因：三元组是 **val** 上的，而 prompt 池来自 **train**，"
              "两边 id 不重叠。")

    # ---- 只写训练真正读的键，不把 reference/output 重复写进池子 ----
    pool = []
    for r in selected:
        record = {
            "id": r.get("id"),
            "source": r.get("source") or "",
            "summary": _reference_of(r),
        }
        if r.get("sft_output"):
            record["sft_output"] = r["sft_output"]
        pool.append(record)

    out_path = _resolve(args.out)
    ensure_dir(out_path.parent)
    n = write_jsonl(out_path, pool)

    empty = sum(1 for r in pool if not r["source"] or not r["summary"])
    if empty:
        print(f"  ⚠ {empty} 条缺 source/summary，训练时会拿到空原文或空参考")

    total_facts = sum(len(extract_facts(r["summary"], kinds=args.kinds)) for r in pool)
    print(f"\n选出 {n} 条 → {out_path}")
    print(f"  平均每条 {total_facts/max(n,1):.2f} 个事实"
          f"（全集平均 {sum(len(extract_facts(_reference_of(r), kinds=args.kinds)) for r in records)/max(len(records),1):.2f}）")


if __name__ == "__main__":
    main()
