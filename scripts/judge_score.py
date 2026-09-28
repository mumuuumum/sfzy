"""用一个 LLM 裁判给"候选摘要 vs 参考摘要"打 0-100 分（离线批量）。

真正的指标实现和三种后端在 `src/sfzy/eval/metrics.py`；这个脚本只是它的
命令行外壳。分工和理由见那份文件的 docstring。

============================ 什么时候用哪个后端 ============================
  RL 训练中（rollout 的候选是新生成的，没有缓存）
      → semantic.backend: local（卡 1）或 api，由 configs/grpo_cloud.yaml 指定
  RL 之后（给 val/test 的三元组打分，和 SFT 比）
      → 就是这个脚本，产出 jsonl 缓存，再由 analyze_outputs.py 汇总

============================ 用法 ============================
    # 服务器：policy 占卡 0，裁判放卡 1
    python scripts/judge_score.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
        --device cuda:1 --input data/triples/sft_val_shard0of2.jsonl \
        --output data/judge/sft_val_shard0.judge.jsonl

    # 作为 tools/bench_metrics.py 的后端（stdin 收 JSONL，stdout 每行一个分）
    python scripts/judge_score.py --model ... --stdin-jsonl

    # 本地冒烟：0.5B 也能把管线跑通（分数本身没有参考价值）
    python scripts/judge_score.py --model models/Qwen2.5-0.5B-Instruct \
        --device cpu --dtype bfloat16 --stdin-jsonl < /tmp/two.jsonl

============================ 打分前后各看什么 ============================
  * `失败 N 条` 必须接近 0。失败 = 截断或解析不出分，**不是 0 分**。
    失败率高就加大 --max-new-tokens。
  * `不同取值 K 个` 太少说明裁判在给模板分（rubric 的锚点白写了）。
  * `--samples 3` 给出的裁判自噪声是**指标的分辨率上限**：
    扰动实验里 Δ 小于它就没有意义。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.eval.metrics import LocalJudgeScorer, load_rubric, normalize_item  # noqa: E402


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 裁判打分")
    ap.add_argument("--model", required=True, help="裁判模型目录或 HF 名")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--load-in-4bit", action="store_true", help="T4 / 小显存时用")
    ap.add_argument("--rubric-file", default=None,
                    help="评分标准；汇报用的 judge 必须换一份（见 eval/metrics.py 的说明）")
    ap.add_argument("--input", default=None, help="三元组 jsonl（含 source/reference/output）")
    ap.add_argument("--limit", type=int, default=None,
                    help="只打前 N 条 —— 冒烟时用，别为了试 20 条加载完整份数据")
    ap.add_argument("--output", default=None, help="输出 jsonl（逐条分数与理由）")
    ap.add_argument("--stdin-jsonl", action="store_true",
                    help="从 stdin 读 {'candidate','reference'[,'source']}，stdout 每行一个分数")
    ap.add_argument("--samples", type=int, default=1, help="重复采样次数，>1 时报告裁判自噪声")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="裁判要逐项写理由，给少了会被截断；被截断的样本按失败处理")
    args = ap.parse_args()

    rubric = load_rubric(args.rubric_file)
    scorer = LocalJudgeScorer(
        model_path=args.model,
        rubric=rubric,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        samples=args.samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    scale = int(rubric.get("scale", 100))

    # ---------------- stdin 模式：给 bench_metrics 当后端 ----------------
    if args.stdin_jsonl:
        items = [json.loads(line) for line in sys.stdin if line.strip()]
        for score in scorer.score_batch(items):
            print("nan" if score is None else f"{score:.4f}")
        return

    # ---------------- 文件模式 ----------------
    if not args.input:
        raise SystemExit("要么 --stdin-jsonl，要么 --input/--output")
    in_path = resolve(args.input)
    out_path = resolve(args.output) if args.output else in_path.with_suffix(".judge.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()]
    if args.limit:
        recs = recs[: args.limit]
    items = [normalize_item(r) for r in recs]
    scores = scorer.score_batch(items)

    # 除了分数，把裁判的原文也存下来 —— 事后要人工核查"这个 85 分凭什么"
    raw_texts = scorer.last_raw or [[] for _ in recs]
    with open(out_path, "w", encoding="utf-8") as f:
        for r, it, s, raws in zip(recs, items, scores, raw_texts):
            f.write(json.dumps({
                "id": it["id"] or r.get("id"),
                "judge_score": s,
                "judge_scale": scale,
                "rubric": rubric.get("version", "unknown"),
                "judge_raw": raws,
            }, ensure_ascii=False) + "\n")

    ok = [s for s in scores if s is not None]
    print(f"\n裁判模型 {args.model}   rubric {rubric.get('version')}")
    print(f"打分 {len(ok)}/{len(scores)} 条，失败 {len(scores) - len(ok)} 条")
    if getattr(scorer, "n_truncated", 0):
        print(f"  ⚠ 被 max_new_tokens 截断 {scorer.n_truncated} 次，这些样本已丢弃 —— "
              f"加大 --max-new-tokens（现在 {args.max_new_tokens}）")
    if ok:
        print(f"  均值 {st.mean(ok):.2f}   标准差 {st.pstdev(ok):.2f}   "
              f"min {min(ok):.2f}   max {max(ok):.2f}")
        print(f"  不同取值 {len({round(s, 1) for s in ok})} 个（满分 {scale}）"
              "—— 太少说明裁判在给模板分")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
