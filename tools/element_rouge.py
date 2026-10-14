"""LLM 抽三要素 + 逐要素 ROUGE：人工摘要 vs 候选摘要（含 GRPO）。

    # 探活（验证 base_url / 模型 / 鉴权，只发一次最小请求）
    python tools/element_rouge.py --config configs/element_extract.yaml --ping

    # 抽几条看看 prompt 效果
    python tools/element_rouge.py --config configs/element_extract.yaml \
        --ref data/splits/val.jsonl --cand grpo=data/triples/grpo_val_v48.jsonl \
        --limit 3 --show

    # 全量：多候选对比（同一份人工摘要，多个候选各算一遍）
    python tools/element_rouge.py --config configs/element_extract.yaml \
        --ref data/splits/val.jsonl \
        --cand sft=data/triples/sft_val_shard*of2.jsonl \
        --cand grpo=data/triples/grpo_val_v48.jsonl \
        --out outputs/element_rouge/val_sft_vs_grpo

产物：
    <out>/report.md     人看的（总览 + 逐要素 + 样例）
    <out>/summary.json  机读的同一批数字
    <out>/scores.jsonl  逐样本：人工要素、候选要素、逐要素分数

人工摘要的抽取结果会进 `cache_path`（离线抽一次，之后查表）；候选摘要（GRPO 每次
采样都是新文本）现算。**改了 prompt 缓存自动失效**（键里含 prompt 指纹）。
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.elements import (  # noqa: E402
    ELEMENTS,
    ELEMENT_ZH,
    ElementCache,
    ElementExtractor,
    ElementRougePipeline,
    ElementRougeConfig,
    build_runtime,
    explain_api_error,
    load_element_config,
)

logger = logging.getLogger("element_rouge")


# ---------------------------------------------------------------- 读数据


def _paths(pattern: str) -> List[Path]:
    hits = [Path(p) for p in sorted(glob.glob(pattern))]
    if not hits:
        hits = sorted(ROOT.glob(pattern))
    return [p for p in hits if p.is_file()]


def load_jsonl_field(pattern: str, field: str) -> Dict[str, str]:
    """读 {id: 文本}；同一 id 只留第一次出现（分片跑重了要能对齐）。"""

    out: Dict[str, str] = {}
    for path in _paths(pattern) or [Path(pattern)]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                doc_id = str(record.get("id", ""))
                text = record.get(field, "")
                if doc_id and isinstance(text, str) and text.strip():
                    out.setdefault(doc_id, text)
    return out


def parse_candidate_spec(spec: str) -> Tuple[str, str, str]:
    """`名字=文件#字段` → (名字, 文件, 字段)；字段默认 output（triples 格式）。"""

    if "=" not in spec:
        raise SystemExit(f"--cand 要写成 名字=文件[#字段]，收到 {spec!r}")
    name, rest = spec.split("=", 1)
    field = "output"
    if "#" in rest:
        rest, field = rest.rsplit("#", 1)
    return name.strip(), rest.strip(), field.strip()


# ---------------------------------------------------------------- 报告


def render_report(report: Dict[str, Any], examples: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# 三要素抽取 + 逐要素 ROUGE")
    lines.append("")
    lines.append(f"- 样本数：**{report['n_samples']}**")
    lines.append(
        f"- ROUGE 口径：`{report['rouge_mode']}` 分词；权重 "
        f"`{report['rouge_weights']}`；要素权重 `{report['element_weights']}`"
    )
    stats = report.get("extract_stats", {})
    lines.append(
        f"- 抽取：请求 {stats.get('requested')} 次、缓存命中 {stats.get('cache_hits')}、"
        f"实调 {stats.get('api_calls')}、失败 {stats.get('failed')}"
    )
    ref = report.get("reference", {})
    empty = ref.get("empty_rate", {})
    lines.append(
        "- 人工摘要里「抽取为空」的比例："
        + "、".join(
            f"{ELEMENT_ZH[k]} {(empty.get(k) or 0):.1%}" for k in ELEMENTS
        )
        + f"（抽取成功 {ref.get('n_ok')} 条）"
    )
    lines.append("")

    lines.append("## 总览")
    lines.append("")
    lines.append("| 候选 | 计入样本 | 抽取失败 | **总分** | 原告诉请 | 被告辩称 | 法院事实 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, payload in report["candidates"].items():
        cells = []
        for element in ELEMENTS:
            value = payload["elements"][element]["score"]
            cells.append("—" if value is None else f"{value:.4f}")
        overall = payload["overall"]
        lines.append(
            f"| {name} | {payload['n_scored']} | {payload['n_failed']} | "
            f"**{'—' if overall is None else f'{overall:.4f}'}** | " + " | ".join(cells) + " |"
        )
    lines.append("")

    lines.append("## 逐要素 ROUGE 明细（F1）")
    lines.append("")
    lines.append("| 候选 | 要素 | ROUGE-1 | ROUGE-2 | ROUGE-L | 加权分 | 计入/两侧空/单侧空 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, payload in report["candidates"].items():
        for element in ELEMENTS:
            item = payload["elements"][element]
            status = payload["status"]
            both = status.get(f"{element}:both_empty", 0)
            one = status.get(f"{element}:one_side_empty", 0)
            fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
            lines.append(
                f"| {name} | {ELEMENT_ZH[element]} | {fmt(item['rouge1_f'])} | "
                f"{fmt(item['rouge2_f'])} | {fmt(item['rougel_f'])} | {fmt(item['score'])} | "
                f"{item['n']}/{both}/{one} |"
            )
    lines.append("")

    for example in examples:
        lines.append(f"## 样例：{example['id']}")
        lines.append("")
        lines.append(f"**人工摘要**：{example['reference'][:400]}")
        lines.append("")
        lines.append(
            "**人工要素**："
            + "；".join(
                f"{ELEMENT_ZH[k]}={example['reference_elements'].get(k, '')[:60]!r}"
                for k in ELEMENTS
            )
        )
        lines.append("")
        for name, payload in example["candidates"].items():
            if not payload.get("ok"):
                lines.append(f"**{name}**：抽取失败（{payload.get('error')}）")
                lines.append("")
                continue
            scores = "、".join(
                f"{ELEMENT_ZH[k]} {payload['elements'][k]['score']:.3f}"
                if payload["elements"][k]["score"] is not None
                else f"{ELEMENT_ZH[k]} —"
                for k in ELEMENTS
            )
            overall = payload.get("overall")
            lines.append(
                f"**{name}**（总分 {'—' if overall is None else f'{overall:.3f}'}）：{scores}"
            )
            for k in ELEMENTS:
                text = payload["elements_text"].get(k, "")
                lines.append(f"  - {ELEMENT_ZH[k]}：{text[:120]!r}" if text else f"  - {ELEMENT_ZH[k]}：（空）")
            lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 主流程


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 抽三要素 + 逐要素 ROUGE")
    parser.add_argument("--config", default="configs/element_extract.yaml")
    parser.add_argument("--ref", default=None, help="人工摘要 jsonl（id/summary）")
    parser.add_argument("--ref-field", default="summary")
    parser.add_argument(
        "--cand",
        action="append",
        default=None,
        metavar="名字=文件[#字段]",
        help="候选摘要，可重复；字段默认 output（id/reference/output 的 triples 格式）",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--cache", default=None, help="覆盖配置里的 cache_path")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--show", type=int, default=0, help="打印前 N 条抽取结果")
    parser.add_argument("--ping", action="store_true", help="只验证 API 连通与鉴权")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    cfg_raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config: ElementRougeConfig = load_element_config(cfg_raw)
    if args.concurrency:
        config = ElementRougeConfig(
            api=type(config.api)(**{**config.api.__dict__, "concurrency": args.concurrency}),
            prompt=config.prompt,
            rouge=config.rouge,
            cache_path=config.cache_path,
            output_dir=config.output_dir,
        )

    try:
        runtime = build_runtime(config.api)
    except Exception as exc:                                    # noqa: BLE001
        print(explain_api_error(exc), file=sys.stderr)
        return 2

    if args.ping:
        try:
            out = runtime.ping()
            print(f"[ping] OK  base_url={runtime.base_url}  model={runtime.model}  -> {out[:60]!r}")
        except Exception as exc:                                # noqa: BLE001
            print(f"[ping] FAIL\n{explain_api_error(exc)}", file=sys.stderr)
            return 3
        # 再真抽一条：验证 prompt 能被模型解析成三个键（这一步才是真正的"能用"）
        probe = "原被告系借款合同纠纷。原告请求判令被告偿还借款十万元。被告辩称已经还清。经审理查明，借款属实。"
        extractor = ElementExtractor(
            runtime,
            config.prompt,
            cache=None,
            model=config.api.model,
            max_tokens=config.api.max_tokens,
            parse_retries=0,
        )
        extraction = extractor.extract(probe)
        print(
            f"[ping] 抽一条：解析{'成功' if extraction.ok else '失败'}　"
            f"要素字数 " + "、".join(
                f"{ELEMENT_ZH[k]}={len(extraction.elements.get(k))}" for k in ELEMENTS
            )
            + f"　tokens: prompt {extractor.stats.prompt_tokens} / completion "
            f"{extractor.stats.completion_tokens}"
        )
        if not extraction.ok:
            print(f"[ping] 抽取失败原因：{extraction.error}", file=sys.stderr)
            return 3
        return 0

    if not args.ref and not args.cand:
        parser.error("至少要给 --ref 或 --cand")

    references = load_jsonl_field(args.ref, args.ref_field) if args.ref else {}
    candidate_specs = [parse_candidate_spec(spec) for spec in (args.cand or [])]

    # ---- 组装样本：id → {reference, {候选名: 文本}}
    rows: Dict[str, Dict[str, Any]] = {}
    for doc_id, text in references.items():
        rows.setdefault(doc_id, {"id": doc_id, "reference": text, "candidates": {}})
    stats_join = {"ids_in_ref": len(references), "matched": 0, "from_embedded": 0, "dropped": 0}
    for name, path, field in candidate_specs:
        loaded = load_jsonl_field(path, field)
        # 候选文件里通常自带 reference（triples 格式）：人工摘要文件里没有这个 id 时用它兜
        refs_inside = load_jsonl_field(path, "reference")
        for doc_id, text in loaded.items():
            reference = references.get(doc_id)
            if reference:
                stats_join["matched"] += 1
            else:
                reference = refs_inside.get(doc_id, "")
                if reference:
                    stats_join["from_embedded"] += 1
            if not reference:
                stats_join["dropped"] += 1
                continue
            rows.setdefault(doc_id, {"id": doc_id, "reference": reference, "candidates": {}})
            rows[doc_id]["candidates"][name] = text

    # 只评"至少有一个候选"的样本：--ref 与 --cand 的 id 对不上时能一眼看出来
    ordered = sorted((r for r in rows.values() if r["candidates"]), key=lambda r: r["id"])
    print(
        f"[element_rouge] id 对齐：人工摘要 {stats_join['ids_in_ref']} 条 → 命中 "
        f"{stats_join['matched']} 条、用候选文件自带 reference 兜 {stats_join['from_embedded']} 条、"
        f"两条路都没有而丢弃 {stats_join['dropped']} 条；可评样本 {len(ordered)}"
    )
    if not ordered:
        print(
            "[element_rouge] 没有可评样本：通常是 --ref 与 --cand 不是同一批数据"
            "（例如 train 的 id 对 val 的 id），或字段名不对。",
            file=sys.stderr,
        )
        return 2
    if args.limit:
        ordered = ordered[: args.limit]

    cache = ElementCache.load(
        args.cache or config.cache_path, writable=not args.no_write
    )
    extractor = ElementExtractor(
        runtime,
        config.prompt,
        cache=cache,
        model=config.api.model,
        max_tokens=config.api.max_tokens,
        parse_retries=config.api.parse_retries,
    )
    pipeline = ElementRougePipeline(extractor, config)
    names = [spec[0] for spec in candidate_specs]
    if not references:
        print("[element_rouge] 注意：没给 --ref，人工摘要取自候选文件里的 reference 字段")

    print(
        f"[element_rouge] {len(ordered)} 条样本，候选 {names}；"
        f"prompt 指纹 {config.prompt.fingerprint()}；模型 {config.api.model}"
    )
    results, report = pipeline.evaluate_rows(ordered, names)

    if args.show:
        for result in results[: args.show]:
            print("=" * 100)
            print("人工摘要：", result.reference)
            for name, payload in result.candidates.items():
                if not payload.get("ok"):
                    print(f"  [{name}] 抽取失败：{payload.get('error')}")
                    continue
                overall = payload.get("overall")
                print(f"  [{name}] 总分 {'—' if overall is None else round(overall, 4)}")
                for element in ELEMENTS:
                    text = payload["elements_text"].get(element, "")
                    score = payload["elements"][element]["score"]
                    print(
                        f"    {ELEMENT_ZH[element]:10s} "
                        f"{'—' if score is None else round(score, 4)!s:8s} {text[:70]!r}"
                    )

    for name, payload in report["candidates"].items():
        overall = payload["overall"]
        detail = "  ".join(
            f"{ELEMENT_ZH[e]}={'—' if payload['elements'][e]['score'] is None else round(payload['elements'][e]['score'], 4)}"
            for e in ELEMENTS
        )
        print(
            f"[element_rouge] {name}: 总分 "
            f"{'—' if overall is None else round(overall, 4)} | {detail} | "
            f"计入 {payload['n_scored']} 条，抽取失败 {payload['n_failed']}"
        )
    stats = report.get("extract_stats", {})
    print(
        f"[element_rouge] 抽取：请求 {stats.get('requested')}、缓存命中 {stats.get('cache_hits')}、"
        f"实调 {stats.get('api_calls')}、解析失败 {stats.get('failed')}；"
        f"tokens: prompt {stats.get('prompt_tokens')} / completion {stats.get('completion_tokens')}"
    )
    if stats.get("api_errors"):
        print("[element_rouge] 调用报错：", file=sys.stderr)
        for message in stats["api_errors"]:
            print("  " + message.replace("\n", "\n  "), file=sys.stderr)
    n_scored = sum(payload["n_scored"] for payload in report["candidates"].values())

    if args.no_write:
        return 0 if n_scored else 3

    out_dir = Path(args.out or config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    # 样例：优先挑"总分差距最大"的几条（人工扫一眼就有信息量）
    examples: List[Dict[str, Any]] = []
    if len(names) >= 2:
        scored = [r for r in results if all(r.candidates[n].get("overall") is not None for n in names)]
        scored.sort(
            key=lambda r: abs(r.candidates[names[0]]["overall"] - r.candidates[names[1]]["overall"]),
            reverse=True,
        )
        chosen = scored[:3]
    else:
        chosen = [r for r in results if r.candidates[names[0]].get("overall") is not None][:3]
    for result in chosen:
        examples.append(
            {
                "id": result.sample_id,
                "reference": result.reference and next(
                    (row["reference"] for row in ordered if row["id"] == result.sample_id), ""
                ),
                "reference_elements": result.reference.to_dict(),
                "candidates": result.candidates,
            }
        )
    (out_dir / "report.md").write_text(
        render_report(report, examples), encoding="utf-8"
    )
    if config.cache_path and not args.no_write:
        written = cache.save(args.cache or config.cache_path)
        print(f"[element_rouge] 抽取缓存已写：{args.cache or config.cache_path}（{written} 条）")
    print(f"[element_rouge] 报告已写：{out_dir / 'report.md'}")
    if n_scored == 0:
        print(
            "[element_rouge] 一条都没算出来（全被计成抽取失败）；"
            "成功的抽取已经进缓存，修掉上面的调用报错后重跑不会重复花钱。",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
