"""同一个抽取任务，多个模型比速度（默认：teleai/xing4.0-29b-a4b vs 裁判那个 qwen3.7-flash）。

    # 默认比两个：$SFZY_ELEMENT_EXTRACT 那个模型 + configs/grpo_fact_cov_ee_t4.yaml 里的裁判模型
    python tools/bench_extract_models.py --limit 6 --rounds 3 --concurrency 6 \
        --out outputs/bench_extract_models

    # 自定义：名字=base_url|模型|密钥环境变量|extra_body(JSON)
    python tools/bench_extract_models.py \
        --model 'teleai=https://api.tokenplus.cn/v1|teleai/xing4.0-29b-a4b|SFZY_ELEMENT_EXTRACT|{"reasoning_effort":"none"}' \
        --model 'qwen=https://maas.qianwenaiapi.com/compatible-mode/v1|qwen3.7-flash|SFZY_JUDGE_API_KEY|{"enable_thinking":false}'

比的是"同一段摘要抽三要素"这件事本身，所以：
  * 用**同一个 prompt**（取 `--element-config` 里的那一段，也就是你要写的那份）；
  * 同一批摘要，跑 `--rounds` 轮（换个顺序/穿插着跑，避免某一轮正好撞上供应商抖动）；
  * 每轮都记录**单条延迟分布**与**解析成功率**——只看平均会被长尾带偏，所以中位/p90/最慢
    和"多少条按时返回、多少条能解析出三个键"都要看；
  * 轮次之间不共享缓存（缓存会把第二次变成 0.1s，比不出模型速度）。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.elements.client import (  # noqa: E402
    _load_api_runtime_module,
    ensure_api_key,
    explain_api_error,
)
from sfzy.elements.config import ApiConfig, ElementRougeConfig  # noqa: E402
from sfzy.elements.schema import ELEMENTS, parse_elements  # noqa: E402


def parse_spec(spec: str) -> Dict[str, Any]:
    """`名字=base_url|模型|密钥环境变量|extra_body(JSON, 可省)` → 配置字典。"""

    if "=" not in spec:
        raise SystemExit(f"--model 要写成 名字=base_url|模型|环境变量[|extra_body]，收到 {spec!r}")
    name, rest = spec.split("=", 1)
    parts = rest.split("|")
    if len(parts) < 3:
        raise SystemExit(f"--model 至少要有 base_url|模型|环境变量 三段，收到 {spec!r}")
    extra: Dict[str, Any] = {}
    if len(parts) >= 4 and parts[3].strip():
        extra = json.loads(parts[3])
    return {
        "name": name.strip(),
        "base_url": parts[0].strip(),
        "model": parts[1].strip(),
        "api_key_env": parts[2].strip(),
        "extra_body": extra,
    }


def default_models(element_config: str, judge_config: str) -> List[Dict[str, Any]]:
    """两个默认参比对象：抽取配置里的模型 + 裁判配置里的模型（都用同一段抽取 prompt）。"""

    element = yaml.safe_load(Path(element_config).read_text(encoding="utf-8"))["element_rouge"]
    api = element["api"]
    models = [
        {
            "name": "element(抽取配置)",
            "base_url": api["base_url"],
            "model": api["model"],
            "api_key_env": api.get("api_key_env", "SFZY_ELEMENT_EXTRACT"),
            "extra_body": dict(api.get("extra_body") or {}),
        }
    ]
    judge_yaml = yaml.safe_load(Path(judge_config).read_text(encoding="utf-8"))
    judge_api = (judge_yaml.get("semantic") or {}).get("judge_api") or {}
    if judge_api:
        models.append(
            {
                "name": f"judge({judge_api.get('model')})",
                "base_url": judge_api["base_url"],
                "model": judge_api["model"],
                "api_key_env": judge_api.get("api_key_env", "SFZY_JUDGE_API_KEY"),
                # 裁判那边关思考用的字段名不一样（Model Studio 是 enable_thinking）
                "extra_body": dict(judge_api.get("extra_body") or {}),
            }
        )
    return models


def run_one_call(
    runtime: Any, messages: List[Dict[str, str]], max_tokens: int
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        body = runtime._post(messages, max_tokens=max_tokens, json_mode=True)
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        content = (message.get("content") or "").strip()
        elements, ok = parse_elements(content)
        return {
            "latency": time.perf_counter() - t0,
            "ok": ok,
            "chars": sum(len(elements.get(name)) for name in ELEMENTS),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "reasoning_chars": len(message.get("reasoning_content") or ""),
            "finish_reason": str(choice.get("finish_reason") or ""),
            "error": "",
        }
    except Exception as exc:                                       # noqa: BLE001
        return {
            "latency": time.perf_counter() - t0,
            "ok": False,
            "chars": 0,
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "reasoning_chars": 0,
            "finish_reason": "",
            "error": explain_api_error(exc).splitlines()[0],
        }


def benchmark_model(
    spec: Dict[str, Any],
    prompt,
    texts: Sequence[str],
    *,
    rounds: int,
    concurrency: int,
    max_tokens: int,
    timeout_s: float,
) -> Dict[str, Any]:
    api = ApiConfig(
        base_url=spec["base_url"],
        model=spec["model"],
        api_key_env=spec["api_key_env"],
        timeout_s=timeout_s,
        max_retries=0,           # 比速度时不要重试掩盖真实延迟
        concurrency=concurrency,
    )
    filled = ensure_api_key(api)
    if filled is not None:
        print(f"  （{spec['name']}：密钥补读自 {filled}）")
    module = _load_api_runtime_module()
    cfg = api.as_runtime_config()
    cfg["extra_body"] = dict(spec.get("extra_body") or {})
    rt = module.build_api_runtime(cfg)
    conversations = [prompt.render(text) for text in texts]

    calls: List[Dict[str, Any]] = []
    round_walls: List[float] = []
    for round_index in range(rounds):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(len(conversations), concurrency)) as pool:
            results = list(pool.map(lambda c: run_one_call(rt, c, max_tokens), conversations))
        round_walls.append(time.perf_counter() - t0)
        calls.extend(results)
        ok = sum(1 for r in results if r["ok"])
        print(
            f"  [{spec['name']}] 第 {round_index + 1}/{rounds} 轮：{len(results)} 条 "
            f"{round_walls[-1]:.1f}s（解析成功 {ok}/{len(results)}）"
        )

    latencies = sorted(r["latency"] for r in calls)
    total_wall = sum(round_walls)
    n = len(calls)
    summary = {
        "name": spec["name"],
        "model": spec["model"],
        "base_url": spec["base_url"],
        "extra_body": spec.get("extra_body"),
        "concurrency": concurrency,
        "rounds": rounds,
        "n_calls": n,
        "wall_s": total_wall,
        "calls_per_min": n / total_wall * 60 if total_wall else 0.0,
        "latency_mean_s": st.mean(latencies) if latencies else 0.0,
        "latency_median_s": st.median(latencies) if latencies else 0.0,
        "latency_p90_s": latencies[int(0.9 * (n - 1))] if latencies else 0.0,
        "latency_max_s": latencies[-1] if latencies else 0.0,
        "parse_ok": sum(1 for r in calls if r["ok"]),
        "empty_elements": sum(1 for r in calls if r["ok"] and r["chars"] == 0),
        "errors": [r["error"] for r in calls if r["error"]][:3],
        "reasoning_chars": sum(r["reasoning_chars"] for r in calls),
        "prompt_tokens": sum(r["prompt_tokens"] for r in calls),
        "completion_tokens": sum(r["completion_tokens"] for r in calls),
        "round_walls": round_walls,
    }
    return summary


def render_markdown(report: Dict[str, Any]) -> str:
    lines = ["# 同一抽取任务：两个模型的速度对比", ""]
    lines.append(f"- 样本数：{report['n_texts']} 条摘要，跑 {report['rounds']} 轮"
                 f"（每轮同样的 {report['n_texts']} 条，合计每个模型 {report['n_calls_per_model']} 次调用）")
    lines.append(f"- 并发：{report['concurrency']}；prompt 平均 {report['prompt_chars']} 字；"
                 f"max_tokens {report['max_tokens']}；比速度时**不重试**")
    lines.append("")
    lines.append("| 模型 | 调用 | 总耗时 | 吞吐 | 单条 均值 | 中位 | p90 | 最慢 | 解析成功 | 失败 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in report["models"]:
        lines.append(
            f"| {item['name']}<br>`{item['model']}` | {item['n_calls']} | {item['wall_s']:.1f}s | "
            f"**{item['calls_per_min']:.0f}/分钟** | {item['latency_mean_s']:.1f}s | "
            f"{item['latency_median_s']:.1f}s | {item['latency_p90_s']:.1f}s | "
            f"{item['latency_max_s']:.1f}s | {item['parse_ok']}/{item['n_calls']} | "
            f"{len(item['errors'])} |"
        )
    lines.append("")
    lines.append("| 模型 | prompt tokens | completion tokens | 思维链字数 | 空要素条数 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in report["models"]:
        lines.append(
            f"| {item['name']} | {item['prompt_tokens']} | {item['completion_tokens']} | "
            f"{item['reasoning_chars']} | {item['empty_elements']} |"
        )
    lines.append("")
    for item in report["models"]:
        if item["errors"]:
            lines.append(f"**{item['name']} 失败样例**：{item['errors'][0]}")
            lines.append("")
    lines += ["## 结论", ""] + [f"- {text}" for text in report["conclusion"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="同一抽取任务、多模型速度对比")
    parser.add_argument("--element-config", default="configs/element_extract.yaml")
    parser.add_argument("--judge-config", default="configs/grpo_fact_cov_ee_t4.yaml")
    parser.add_argument("--data", default="data/triples/grpo_val_v48.jsonl")
    parser.add_argument("--field", default="reference", help="拿哪个字段当抽取输入（默认人工摘要）")
    parser.add_argument("--limit", type=int, default=6, help="每条样本抽一次")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--model", action="append", default=None,
                        metavar="名字=base_url|模型|环境变量[|extra_body]")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    element = yaml.safe_load(Path(args.element_config).read_text(encoding="utf-8"))["element_rouge"]
    prompt = ElementRougeConfig.from_config(element).prompt      # 同一份 prompt 给两边用
    models = ([parse_spec(spec) for spec in args.model] if args.model
              else default_models(args.element_config, args.judge_config))

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    texts = [str(row.get(args.field, "")) for row in rows[: args.limit]]
    texts = [t for t in texts if t.strip()]
    if not texts:
        print(f"[bench] {args.data} 的 {args.field} 字段没读到文本", file=sys.stderr)
        return 2
    prompt_chars = sum(len(prompt.render(t)[-1]["content"]) for t in texts) // len(texts)
    print(f"[bench] 样本 {len(texts)} 条，轮次 {args.rounds}，并发 {args.concurrency}"
          f"（prompt 平均 {prompt_chars} 字）")

    results = []
    for spec in models:
        print(f"[bench] 开始：{spec['name']}  model={spec['model']}")
        results.append(
            benchmark_model(spec, prompt, texts, rounds=args.rounds,
                            concurrency=args.concurrency, max_tokens=args.max_tokens,
                            timeout_s=args.timeout)
        )

    fastest = max(results, key=lambda r: r["calls_per_min"])
    steady = min(results, key=lambda r: r["latency_max_s"])
    report = {
        "n_texts": len(texts),
        "rounds": args.rounds,
        "n_calls_per_model": len(texts) * args.rounds,
        "concurrency": args.concurrency,
        "prompt_chars": prompt_chars,
        "max_tokens": args.max_tokens,
        "models": results,
        "conclusion": [
            f"最快：{fastest['name']}（{fastest['calls_per_min']:.0f} 次/分钟，"
            f"中位 {fastest['latency_median_s']:.1f}s）。",
            f"最稳（最慢单条最小）：{steady['name']}（最慢 {steady['latency_max_s']:.1f}s）"
            "——长尾决定全量跑会不会卡住，别只看均值。",
        ],
    }
    for item in results:
        print(f"[bench] {item['name']}: {item['wall_s']:.1f}s 总、"
              f"{item['calls_per_min']:.0f} 次/分钟、中位 {item['latency_median_s']:.1f}s、"
              f"最慢 {item['latency_max_s']:.1f}s、解析成功 {item['parse_ok']}/{item['n_calls']}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bench.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
        print(f"[bench] 报告已写：{out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
