"""奖励体检：确认它在真实数据上"坏得对"。

单元测试（tests/test_reward.py）锁的是**接口**：归一化、提取顺序、门控分支。
这个工具锁的是**方向**：奖励在真实摘要上被破坏时，是不是按预期下降。

这是上 GRPO 之前最后一道闸 —— 奖励方向反了的话，训练会非常平稳地
朝着错的方向走，而且日志里看不出来（loss 照常下降）。

两种模式：

  合成扰动（默认，纯 CPU，几秒）：
      把参考摘要做几种**确定性**的破坏，看奖励是不是按预期掉：
        参考原文            → 基线，应该最高
        去掉所有数字        → fact_coverage 应该掉（漏事实是这个项目的主要痛点）
        砍掉后半段          → coverage 掉、长度比掉，可能被门控
        加"以下是摘要："前缀 → 应该被门控（格式约束）
        把原文里的数字全堆上 → fact_precision 应该掉（防 hacking 的监控项）

  真实输出（--triples，读 scripts/generate_triples.py 的产物）：
      分项统计 + 奖励与 ROUGE 的相关性 + 漏事实样本的奖励对比

用法：
    python tools/check_reward.py --config configs/grpo.yaml
    python tools/check_reward.py --config configs/grpo.yaml \
        --triples data/triples/sft_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                        # noqa: E402
from sfzy.rl.reward import (                               # noqa: E402
    compute_reward,
    extract_facts,
    summarize_gate_reasons,
)

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:亿元|万元|元|年|月|日)?")


def load_val_records(split: str, limit: int) -> list[dict]:
    path = ROOT / "data" / "splits" / f"{split}.jsonl"
    if not path.exists():
        path = ROOT / "data" / "processed" / ("dev.jsonl" if split == "val" else f"{split}.jsonl")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def drop_numbers(text: str) -> str:
    """去掉所有数字（含单位/日期）。"""
    return NUMBER_RE.sub("", text)


def stuff_numbers(source: str, reference: str) -> str:
    """把原文里出现、而参考摘要里没有的数字全堆到摘要末尾。

    这是覆盖率最明显的 hacking 路径：光看覆盖率它会涨分，
    所以必须同时看 fact_precision。
    """
    extra = [f.split(":", 1)[1] for f in extract_facts(source) - extract_facts(reference)]
    return reference + "，另涉及金额" + "、".join(extra[:20]) + "元。"


def perturb(reference: str, source: str) -> dict[str, str]:
    return {
        "参考原文（基线）": reference,
        "去掉所有数字": drop_numbers(reference),
        "砍掉后半段": reference[: max(len(reference) // 2, 1)],
        "加'以下是摘要：'前缀": "以下是摘要：" + reference,
        "把原文数字全堆上": stuff_numbers(source, reference),
    }


def spearman(xs: list[float], ys: list[float]) -> float:
    """秩相关。手写是为了不引入 scipy —— 这里只需要一个数。

    **并列值必须取平均名次。** 直接按排序后的位置给名次的话，一组
    "奖励全是 0"的样本会拿到 0,1,2,3… 这些人为的名次，于是算出一个
    纯属虚构的相关性（实测拿到过 -0.413）。并列取平均之后，
    全零奖励的方差为 0，函数会老老实实返回 nan。
    """
    if len(xs) < 3:
        return float("nan")

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mean_x, mean_y = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx) ** 0.5
    var_y = sum((b - mean_y) ** 2 for b in ry) ** 0.5
    return cov / (var_x * var_y) if var_x and var_y else float("nan")


def report_synthetic(records: list[dict], reward_cfg: dict) -> None:
    buckets: dict[str, list] = {}
    with_facts = 0
    for record in records:
        if extract_facts(record["summary"]):
            with_facts += 1
        for name, candidate in perturb(record["summary"], record["source"]).items():
            _, bd = compute_reward(candidate, record["summary"], record["source"], reward_cfg)
            buckets.setdefault(name, []).append(bd)

    # 先说清楚"主信号能覆盖多少样本"，否则下面那行覆盖率会看着像坏了：
    # 参考摘要里没有金额/日期/编号时，覆盖率按设计恒为 0（不给分也不假装满分），
    # 于是这些样本的奖励天花板只有 ROUGE 那一项。
    ratio = with_facts / len(records)
    print(f"参考摘要里含可验证事实（金额/日期/编号）的占比：{ratio*100:.1f}%"
          f" —— 其余 {(1-ratio)*100:.0f}% 的样本覆盖率恒为 0，奖励天花板只有 ROUGE 项。")
    print("（对 GRPO 无害：组内归一化会把「每条都一样的常数」消掉；"
          "但**跨实验比平均奖励**时不可比，报告里要说明。）\n")

    baseline = None
    print(f"{'扰动':22s} {'奖励':>7s} {'被门控':>7s} {'覆盖率':>7s} "
          f"{'精确率':>7s} {'ROUGE-L':>8s} {'长度比':>7s}")
    for name, breakdowns in buckets.items():
        mean = lambda key: sum(getattr(b, key) for b in breakdowns) / len(breakdowns)
        gated = sum(b.gated for b in breakdowns) / len(breakdowns)
        reward = mean("total")
        baseline = reward if baseline is None else baseline
        print(f"{name:22s} {reward:7.4f} {gated*100:6.0f}% {mean('fact_coverage'):7.3f} "
              f"{mean('fact_precision'):7.3f} {mean('rouge_l'):8.3f} {mean('length_ratio'):7.2f}")

    print("\n判据（任何一条不满足，先改奖励再上 GRPO）：")
    base = buckets["参考原文（基线）"]
    mean_total = lambda bs: sum(b.total for b in bs) / len(bs)
    ok = [
        ("去掉数字后奖励应当下降", mean_total(buckets["去掉所有数字"]) < mean_total(base)),
        ("砍半后奖励应当下降", mean_total(buckets["砍掉后半段"]) < mean_total(base)),
        ("加前缀应当被门控", all(b.gated for b in buckets["加'以下是摘要：'前缀"])),
        ("堆数字不应当涨分", mean_total(buckets["把原文数字全堆上"]) <= mean_total(base)),
        ("堆数字应当压低精确率",
         sum(b.fact_precision for b in buckets["把原文数字全堆上"])
         < sum(b.fact_precision for b in base)),
    ]
    for label, passed in ok:
        print(f"  {'✓' if passed else '✗'} {label}")


def report_triples(path: Path, reward_cfg: dict) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        print(f"{path} 是空的")
        return

    breakdowns, rouges = [], []
    for row in rows:
        _, bd = compute_reward(row["output"], row["reference"], row["source"], reward_cfg)
        breakdowns.append(bd)
        rouges.append(bd.rouge_l)

    mean = lambda key: sum(getattr(b, key) for b in breakdowns) / len(breakdowns)
    print(f"真实输出 {len(rows)} 条（{path}）")
    print(f"  平均奖励 {mean('total'):.4f} | 被门控 {sum(b.gated for b in breakdowns)/len(rows)*100:.0f}%"
          f" | 覆盖率 {mean('fact_coverage'):.3f} | 精确率 {mean('fact_precision'):.3f}"
          f" | ROUGE-L {mean('rouge_l'):.3f} | 长度比 {mean('length_ratio'):.2f}")
    print(f"  门控原因分布: {summarize_gate_reasons(breakdowns)}")

    missed = [b for b in breakdowns if b.n_missed_facts > 0]
    clean = [b for b in breakdowns if b.n_ref_facts > 0 and b.n_missed_facts == 0]
    if missed and clean:
        gap = (sum(b.total for b in clean) / len(clean)) - (sum(b.total for b in missed) / len(missed))
        print(f"  漏事实 {len(missed)} 条 / 干净 {len(clean)} 条，奖励差 {gap:+.4f}"
              f"（旧数据上实测差 +0.073，方向一致就说明主信号抓对了）")
    correlation = spearman([b.total for b in breakdowns], rouges)
    if correlation != correlation:  # nan
        print("  奖励与 ROUGE-L 的秩相关: 无意义（奖励在样本间没有方差，多半全被门控）")
    else:
        print(f"  奖励与 ROUGE-L 的秩相关: {correlation:+.3f}")
        print("  判据：明显为正但不是 1 —— 重合说明事实覆盖没起作用，"
              "为负说明奖励写反了。")
    if sum(b.gated for b in breakdowns) / len(rows) > 0.5:
        print("  ⚠️ 门控拦掉一半以上：问题多半在生成配置或 prompt，不在奖励设计。")


def main() -> None:
    parser = argparse.ArgumentParser(description="奖励体检")
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--triples", default=None,
                        help="generate_triples.py 的产物；给了就跑真实输出模式")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    reward_cfg = cfg.path_("rl.reward", {})
    print(f"奖励配置: mode={reward_cfg.get('mode')} weights={reward_cfg.get('weights')}\n")

    if args.triples:
        report_triples(Path(args.triples), reward_cfg)
    else:
        report_synthetic(load_val_records(args.split, args.limit), reward_cfg)


if __name__ == "__main__":
    main()
