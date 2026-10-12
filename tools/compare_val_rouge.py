"""在**同一份验证集**上给多份模型输出算 ROUGE，并做成对比较。

============================ 为什么是"成对" ============================
两个模型跑同一套 val，比的应该是"同一条样本上谁好"，而不是两个独立平均分。
独立平均分里混进了样本难度差异（同一份数据其实抵消掉了，但方差大、也看不出
胜负分布）；成对比较能直接给出"赢/平/输多少条"和差值的置信区间。

三条口径必须和项目评测路径严格一致，否则数字对不上：

1. **逐样本先算分，再对样本取宏平均**（`score_corpus` 的做法）。把所有人的
   预测拼起来算一次是另一种口径，数值不同，不能混着报。
2. **只在共同 id 上比**（`score_corpus` 也是这个规则）。某个模型少跑了几条时，
   差值里不该混进"谁多做了一条"。
3. **综合分 = 0.2·ROUGE-1-F + 0.4·ROUGE-2-F + 0.4·ROUGE-L-F**（官方口径，
   `rouge.OFFICIAL_WEIGHTS`），不是三项简单平均。

分词口径有两套，默认两套都算：
  * `char`   —— `scripts/evaluate.py` 的默认值（对外报分用这个）；
  * `jieba`  —— `RewardSpec.rouge_mode` 的默认值（训练奖励/监控用这个）。
两套给出的绝对分不同、有时结论也不同，所以报告里必须写清是哪一套。

----------------------------- 用法 -----------------------------
    # SFT vs GRPO，两个分片自动按 id 合并
    python tools/compare_val_rouge.py \
        --pred sft="data/triples/sft_val_shard*of2.jsonl" \
               grpo_v48=data/triples/grpo_val_v48.jsonl \
        --ref data/splits/val.jsonl

    # 存成 JSON（含逐样本分数），便于二次分析
    python tools/compare_val_rouge.py ... --json outputs/val_rouge.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.eval.rouge import score_corpus, score_pair  # noqa: E402


def load_outputs(pattern: str) -> Tuple[Dict[str, str], Dict[str, str], int]:
    """读 triples 格式（id/reference/output）的文件（支持 glob 多分片）。

    同一 id 只保留第一次出现的那条 —— 分片跑重了或续跑追加时会有重复，
    静默取后者会让"哪一份输出被评了"变得不可复现。
    """
    outs: Dict[str, str] = {}
    refs: Dict[str, str] = {}
    dup = 0
    paths = _paths(pattern)
    if not paths:
        raise FileNotFoundError(f"没有匹配到文件：{pattern}")
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = str(record["id"])
            if doc_id in outs:
                dup += 1
                continue
            outs[doc_id] = record["output"]
            refs[doc_id] = record["reference"]
    return outs, refs, dup


def _paths(pattern: str) -> List[Path]:
    """按 pattern 找文件。相对路径先按 cwd 解析（= 仓库根），再退回仓库根。"""
    hits = [Path(p) for p in sorted(glob.glob(pattern))]
    if not hits:
        hits = sorted(ROOT.glob(pattern))
    return [p for p in hits if p.is_file()]


def load_reference(path: Optional[str]) -> Dict[str, str]:
    """读官方 split（id/summary）。给了就对一遍 reference 字段是否一致。"""
    if not path:
        return {}
    hits = _paths(path) or [Path(path)]
    refs: Dict[str, str] = {}
    for line in open(hits[0], encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        refs[str(record["id"])] = record.get("summary", record.get("reference", ""))
    return refs


def paired_bootstrap(
    diffs: Sequence[float], *, n_boot: int = 10000, seed: int = 42
) -> Tuple[float, float, float]:
    """成对 bootstrap：同一条样本的差值整块重采样，返回 (均值, 2.5%, 97.5%)。

    独立重采样两边会破坏配对结构、把置信区间撑大 —— 这里要的是"换一批同等
    难度的案件，差值还在不在同一侧"。
    """
    import numpy as np

    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_all(
    names: List[str], preds: Dict[str, Dict[str, str]], refs: Dict[str, str],
    mode: str, ids: List[str],
) -> Dict[str, Dict[str, dict]]:
    """每个模型每个样本一行分数；同时给出语料级宏平均。"""
    per_sample: Dict[str, Dict[str, dict]] = {}
    for name in names:
        per_sample[name] = {
            doc_id: score_pair(preds[name][doc_id].strip(), refs[doc_id].strip(), mode=mode)
            for doc_id in ids
        }
    return per_sample


def corpus_scores(per_sample: Dict[str, dict]) -> Dict[str, float]:
    """按 score_corpus 的口径做宏平均（这里手算，以便复用已有逐样本分数）。"""
    keys = [k for k in next(iter(per_sample.values())) if k != "num_samples"]
    return {k: st.fmean(s[k] for s in per_sample.values()) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="同一验证集上的 ROUGE 成对比较")
    parser.add_argument("--pred", nargs="+", required=True,
                        help="name=路径（或 name=glob），第一份作为基线")
    parser.add_argument("--ref", default="data/splits/val.jsonl",
                        help="官方 split（id/summary），用于核对 reference 字段")
    parser.add_argument("--mode", default="char,jieba", help="char,jieba 逗号分隔")
    parser.add_argument("--json", default=None, help="把逐样本结果写到这个路径")
    args = parser.parse_args()

    specs: List[Tuple[str, str]] = []
    for item in args.pred:
        if "=" not in item:
            raise SystemExit(f"--pred 要写成 name=路径 的形式，收到：{item}")
        name, pattern = item.split("=", 1)
        specs.append((name.strip(), pattern.strip()))
    names = [n for n, _ in specs]

    preds: Dict[str, Dict[str, str]] = {}
    refs_from_pred: Dict[str, Dict[str, str]] = {}
    for name, pattern in specs:
        outs, refs, dup = load_outputs(pattern)
        preds[name] = outs
        refs_from_pred[name] = refs
        print(f"[数据] {name}: {len(outs)} 条（{len(_paths(pattern))} 个文件，"
              f"重复 id {dup} 条）")

    official = load_reference(args.ref)
    if official:
        for name in names:
            bad = [i for i in preds[name]
                   if i in official and refs_from_pred[name][i] != official[i]]
            print(f"[核对] {name}: reference 与 {args.ref} 不一致 {len(bad)} 条")
        missing = {n: [i for i in official if i not in preds[n]] for n in names}
        for name, miss in missing.items():
            if miss:
                print(f"[核对] {name}: val 里有 {len(miss)} 条没有输出，示例 {miss[:2]}")

    common = set.intersection(*[set(preds[n]) for n in names])
    common &= set(official) if official else common
    ids = [i for i in official if i in common] if official else sorted(common)
    print(f"[口径] 共同可评样本 {len(ids)} 条（逐样本算分后取宏平均，只在共同 id 上比）\n")
    if not ids:
        raise SystemExit("没有共同 id，无法比较")

    result: Dict[str, object] = {"num_samples": len(ids), "ids": ids, "modes": {}}
    for mode in [m.strip() for m in args.mode.split(",") if m.strip()]:
        per_sample = score_all(names, preds, {i: refs_from_pred[names[0]][i] for i in ids}, mode, ids)
        # 每个模型的语料级分数（= score_corpus 的宏平均口径）
        corpus = {n: corpus_scores(per_sample[n]) for n in names}
        # 用官方实现再算一遍作为对账，防止这里手算的宏平均和项目路径有偏差
        for n in names:
            check = score_corpus(
                {i: preds[n][i] for i in ids},
                {i: refs_from_pred[names[0]][i] for i in ids},
                mode=mode,
            )
            assert abs(check["overall"] - corpus[n]["overall"]) < 1e-12, f"{n}/{mode} 口径不一致"

        print(f"===== 分词口径 {mode} =====")
        head = f"{'模型':<12}{'R1-F':>9}{'R2-F':>9}{'RL-F':>9}{'综合分':>10}{'输出均长':>10}"
        print(head)
        print("-" * len(head))
        for n in names:
            c = corpus[n]
            mean_len = st.fmean(len(preds[n][i]) for i in ids)
            print(f"{n:<12}{c['rouge-1-f']:>9.4f}{c['rouge-2-f']:>9.4f}"
                  f"{c['rouge-l-f']:>9.4f}{c['overall']:>10.4f}{mean_len:>10.1f}")

        base = names[0]
        for n in names[1:]:
            diffs = [per_sample[n][i]["overall"] - per_sample[base][i]["overall"] for i in ids]
            mean, lo, hi = paired_bootstrap(diffs)
            win = sum(1 for d in diffs if d > 1e-9)
            lose = sum(1 for d in diffs if d < -1e-9)
            tie = len(diffs) - win - lose
            delta = corpus[n]["overall"] - corpus[base]["overall"]
            print(f"\n{n} − {base}（综合分）：{delta:+.4f}"
                  f"    赢 {win} / 平 {tie} / 输 {lose}    "
                  f"中位差 {st.median(diffs):+.4f}    成对 bootstrap 95% 区间 [{lo:+.4f}, {hi:+.4f}]")
            for k, label in (("rouge-1-f", "R1-F"), ("rouge-2-f", "R2-F"), ("rouge-l-f", "RL-F")):
                d = corpus[n][k] - corpus[base][k]
                w = sum(1 for i in ids
                        if per_sample[n][i][k] - per_sample[base][i][k] > 1e-9)
                print(f"      {label}: {corpus[base][k]:.4f} → {corpus[n][k]:.4f}（{d:+.4f}，赢 {w} 条）")
        print()

        result["modes"][mode] = {  # type: ignore[index]
            "corpus": corpus,
            "per_sample": per_sample,
        }

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出 {out}")


if __name__ == "__main__":
    main()
