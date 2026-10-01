"""用六要素事实一致性 Judge 给一批三元组离线打分（和 GRPO 的奖励同一个口径）。

============================ 为什么需要它 ============================
训练时 `fact_reward` 只在 rollout 上算过，**验证集上从来没算过**。
于是"SFT vs GRPO 谁的事实一致性更好"这个问题，用训练日志答不了 ——
两次 run 的 prompt 不同、采样也不同，日志里的均值根本不可比。

这个脚本补上那一列：拿同一批 val 记录、同一套 Judge、同一条
`judge_candidates` 路径，分别给两个模型的输出打分。

============================ 一次传两个文件有意想不到的好处 ============================
同一篇文书在两条臂里是**同一份 source**，而文档六要素的缓存是按
source 的 sha1 命中的。两个文件一起传：

  * 文档要素只提取一次，Judge 的生成开销直接省掉一半；
  * 两条臂是在**同一份要素**上判的，比较更干净。

============================ 用法 ============================
    python tools/score_fact_judge.py \
        --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
        --device cuda:1 --load-in-4bit --limit 100 \
        --input data/triples/sft_val.jsonl data/triples/grpo_val.jsonl \
        --out-dir data/judge

输出每个输入文件一份 `<stem>.fact.jsonl`，每行：

    {"id": "...", "fact_reward": 0.9375, "min_element_score": 0.75,
     "judgment_result_score": 1.0, "scores": {...}, "raw_scores": {...}}

失败的行（比如那篇文书六要素抽不出来）记 `"error"` 字段，**不写假分数** ——
把失败当 0 分会让统计系统性偏低。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                          # noqa: E402
from sfzy.judge.judge import FactConsistencyJudge            # noqa: E402
from sfzy.judge.schema import ELEMENT_ZH, ELEMENTS, JudgeResult, summarize_scores  # noqa: E402


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# ---------------------------------------------------------------------------
# 打分（和 Judge 解耦，便于在不加载模型的情况下测）
# ---------------------------------------------------------------------------
def score_records(
    judge: FactConsistencyJudge,
    records: Sequence[Dict[str, Any]],
    candidate_key: str = "output",
) -> List[Tuple[str, Optional[JudgeResult], Optional[str]]]:
    """逐条打分，返回 [(id, JudgeResult|None, error|None)]。

    一条失败**不中断整批**：val 集里总有几篇文书让六要素提取器抛
    `ExtractionFailure`（要素少于 2 项），为此丢掉前面几小时的打分不值。
    """
    out: List[Tuple[str, Optional[JudgeResult], Optional[str]]] = []
    for rec in records:
        rid = str(rec.get("id", ""))
        document = rec.get("source") or ""
        candidate = rec.get(candidate_key) or rec.get("candidate") or ""
        try:
            results = judge.judge_candidates(document, [candidate], candidate_ids=[rid])
        except Exception as exc:  # noqa: BLE001 — 见上面的说明
            out.append((rid, None, f"{type(exc).__name__}: {exc}"))
            continue
        out.append((rid, results[0], None))
    return out


def result_to_record(
    rid: str, result: Optional[JudgeResult], error: Optional[str]
) -> Dict[str, Any]:
    """把一条判定结果摊平成 jsonl 记录。"""
    if result is None:
        return {"id": rid, "fact_reward": None, "error": error}
    return {
        "id": rid,
        "fact_reward": result.weighted_reward,
        "min_element_score": result.min_element_score,
        "judgment_result_score": result.judgment_result_score,
        "scores": {k: round(v, 6) for k, v in result.scores.items()},
        "raw_scores": dict(result.raw_scores),
        "sources": dict(result.sources),
    }


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """把逐条记录汇总成一张表要的那几个数（复用需求的统计口径）。"""
    ok = [r for r in rows if r.get("fact_reward") is not None]
    if not ok:
        return {}
    n = len(ok)

    class _R:                                  # summarize_scores 只用到这三个字段
        def __init__(self, row):
            self.weighted_reward = float(row["fact_reward"])
            self.min_element_score = float(row.get("min_element_score", 0.0))
            self.scores = {k: float(v) for k, v in (row.get("scores") or {}).items()}
            self.raw_scores = {k: int(v) for k, v in (row.get("raw_scores") or {}).items()}

    stats = summarize_scores([_R(r) for r in ok])       # type: ignore[arg-type]
    stats["n"] = float(n)
    stats["n_failed"] = float(len(rows) - n)
    return stats


def print_summary(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    stats = summarize(rows)
    print(f"\n{'=' * 72}\n{title}（{len(rows)} 条）\n{'=' * 72}")
    if not stats:
        print("  全部失败，没有可统计的分数")
        return
    print(f"  事实一致性 fact_reward : {stats['mean_fact_reward']:.4f}")
    print(f"  最低要素分（不含案由） : {stats['mean_min_element_score']:.4f}")
    print(f"  裁判结果分             : {stats['mean_judgment_result_score']:.4f}")
    elements = "  ".join(
        f"{ELEMENT_ZH[k]} {stats[f'mean_{k}_score']:.2f}" for k in ELEMENTS
    )
    print(f"  六要素均值             : {elements}")
    ratio = "  ".join(f"{lv}:{stats[f'ratio_score_{lv}'] * 100:.0f}%" for lv in range(5))
    print(f"  评分档位分布           : {ratio}")
    print(f"  成功 {int(stats['n'])} 条，失败 {int(stats['n_failed'])} 条")


def compare_summaries(titles: Sequence[str], all_rows: Sequence[Sequence[Dict]]) -> None:
    """两份以上时打一张对照表 —— 这才是这个脚本存在的主要理由。"""
    if len(all_rows) < 2:
        return
    stats = [summarize(rows) for rows in all_rows]
    keys = ["mean_fact_reward", "mean_min_element_score", "mean_judgment_result_score"]
    keys += [f"mean_{k}_score" for k in ELEMENTS]
    label = {**{k: k for k in keys}, **{f"mean_{k}_score": ELEMENT_ZH[k] for k in ELEMENTS}}
    print(f"\n{'=' * 72}\n对照（注意：只有同一批 id 才可比）\n{'=' * 72}")
    header = "指标".ljust(22) + "".join(t[:16].rjust(12) for t in titles)
    print(header)
    print("-" * len(header))
    for key in keys:
        if not all(s and key in s for s in stats):
            continue
        cells = "".join(f"{s[key]:>12.4f}" for s in stats)
        print(label[key].ljust(22) + cells)
    delta = stats[1]["mean_fact_reward"] - stats[0]["mean_fact_reward"]
    print(f"\n  Δ fact_reward（{titles[1]} − {titles[0]}）= {delta:+.4f}")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            ids.add(str(json.loads(line).get("id")))
    return ids


def build_judge(args: argparse.Namespace) -> FactConsistencyJudge:
    model_cfg: Dict[str, Any] = {}
    if args.model_config:
        model_cfg = dict(load_config(resolve(args.model_config)).get("model", {}))
    model = args.model or model_cfg.get("model_name_or_path")
    if not model:
        raise SystemExit("要么给 --model，要么给 --model-config（里面有 model_name_or_path）")
    local = resolve(model)
    if local.is_dir():
        model = str(local)
    return FactConsistencyJudge(
        model_path=model,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        max_batch_size=args.max_batch_size,
        extract_max_new_tokens=args.extract_max_new_tokens,
        min_document_elements=args.min_document_elements,
        judge_variant=args.judge_variant,
        doc_fallback=not args.no_doc_fallback,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="六要素事实一致性 Judge 的批量打分")
    ap.add_argument("--input", nargs="+", required=True,
                    help="三元组 jsonl（含 id/source 和候选字段），可以给多个")
    ap.add_argument("--out-dir", default="data/judge",
                    help="输出目录，每个输入写一份 <文件名>.fact.jsonl")
    ap.add_argument("--candidate-key", default="output",
                    help="候选摘要的字段名（generate_triples 用 output）")
    ap.add_argument("--limit", type=int, default=None, help="每个文件只打前 N 条")
    ap.add_argument("--no-resume", action="store_true",
                    help="默认跳过输出里已有的 id（可断点续跑）")

    ap.add_argument("--model", default=None, help="裁判模型目录或 HF 名")
    ap.add_argument("--model-config", default=None,
                    help="从配置里取 model_name_or_path（可选）")
    ap.add_argument("--device", default="cuda:1", help="裁判卡的设备（默认卡 1）")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--load-in-4bit", action="store_true", help="T4 / 小显存时用")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--extract-max-new-tokens", type=int, default=1024)
    ap.add_argument("--min-document-elements", type=int, default=2)
    ap.add_argument("--judge-variant", default="spec", choices=["spec", "fewshot"])
    ap.add_argument("--no-doc-fallback", action="store_true",
                    help="关掉『要素抽空时拿整篇原文兜底』（默认开）")
    args = ap.parse_args()

    inputs = [resolve(p) for p in args.input]
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"输入文件不存在：{path}")

    judge = build_judge(args)
    print(f"裁判：{args.model or args.model_config}   设备 {args.device}   4bit={args.load_in_4bit}")

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    titles: List[str] = []
    all_rows: List[List[Dict[str, Any]]] = []
    for path in inputs:
        out_path = out_dir / f"{path.stem}.fact.jsonl"
        records = load_records(path)
        if args.limit:
            records = records[: args.limit]
        done = set() if args.no_resume else load_done_ids(out_path)
        todo = [r for r in records if str(r.get("id")) not in done]
        print(f"\n{path.name}: {len(records)} 条，已完成 {len(done)} 条，本次打 {len(todo)} 条")

        with open(out_path, "a", encoding="utf-8") as f:
            for start in range(0, len(todo), 16):        # 进度可见，别静默跑几小时
                chunk = todo[start:start + 16]
                for rid, result, error in score_records(judge, chunk, args.candidate_key):
                    f.write(json.dumps(result_to_record(rid, result, error),
                                       ensure_ascii=False) + "\n")
                f.flush()
                print(f"  …已打 {min(start + 16, len(todo))}/{len(todo)}", end="\r")
        print()

        rows = load_records(out_path)
        titles.append(path.stem)
        all_rows.append(rows)
        print_summary(path.stem, rows)

    compare_summaries(titles, all_rows)
    print(f"\n输出目录：{out_dir}")


if __name__ == "__main__":
    main()
