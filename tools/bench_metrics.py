"""指标区分度压力测试台：用**可控扰动**取代人工标注来给指标打分。

============================ 为什么需要这个东西 ============================
"BERTScore 区分度不够" 是个含糊的说法。有区分度 = 对**该降分的扰动**降分、
对**不该降分的扰动**不降分。这两件事都能用规则构造出来，不需要人标。

所以把 1340 条 (source, reference, output) 每条复制成若干份，每份施加一种
扰动，比较指标相对基线（未扰动输出）的 Δ：

  破坏型扰动（指标必须明显掉分，掉得越多越好）
    d_digit   把输出里第一个数字改掉一位      —— 事实抄错
    d_fact    删掉一个含事实的句子            —— 关键事实遗漏
    d_verdict 判决结果反转（驳回 <-> 支持）    —— 法律结论相反
    d_party   当事人姓名换成"张某"            —— 主体张冠李戴
    d_halluc  掺入一条参考里没有的事实        —— 幻觉

  等价型扰动（指标不该掉分，掉得越少越好）
    e_order   调换两个相邻句子的顺序          —— 信息不变，只是语序
    e_unit    金额单位改写（48000元 -> 4.8万元）—— 表述换形，语义不变

------------------------------ 怎么读结果 ------------------------------
  区分度 = -mean(Δ_破坏) / max(|mean(Δ_等价)|, eps)

  ROUGE 的实测问题是**分母不小、分子不大**：它对语序和用词极度敏感
  （等价扰动掉分多），却几乎看不见数字抄错（破坏扰动掉分少）。这个比值
  一算就把"区分度不够"从感觉变成了数字。

------------------------------ 用例 ------------------------------
    # 只看 ROUGE（不需要模型，秒出）
    python tools/bench_metrics.py --input "data/triples/sft_val_shard*of2.jsonl"

    # 加上句向量余弦（需要 --encoder 指向本地模型目录）
    python tools/bench_metrics.py --input "..." --encoder models/hf/models--BAAI--bge-small-zh-v1.5/snapshots/<hash>

    # 导出扰动后的语料，交给服务器上的 judge 模型跑同一套测试
    python tools/bench_metrics.py --input "..." --dump data/metrics_bench/pressure.jsonl

============================ 指标接口约定 ============================
每个指标是 `score(candidate, reference) -> float`，越大越好。judge 指标只要
满足这个签名就能挂进来（`--judge-cmd`），本地不用装大模型。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.eval.rouge import score_pair  # noqa: E402
from sfzy.rl.reward import extract_facts  # noqa: E402


# --------------------------------------------------------------------------
# 扰动构造
# --------------------------------------------------------------------------
SENT_RE = re.compile(r"[^。；！？]*[。；！？]|[^。；！？]+")


def split_sentences(text: str) -> List[str]:
    return [s for s in (x.strip() for x in SENT_RE.findall(text)) if s]


def _flip_digit(text: str) -> Optional[str]:
    """把第一个连续数字串的最后一位改动 1。"""
    m = re.search(r"\d+", text)
    if not m:
        return None
    old = m.group()
    perturbed = old[:-1] + str((int(old[-1]) + 1) % 10)
    if perturbed == old:
        return None
    return text[: m.start()] + perturbed + text[m.end() :]


def _drop_fact_sentence(text: str) -> Optional[str]:
    """删掉一个含事实的句子，长度取中位数，避免总是删最长/最短的。"""
    cands = [s for s in split_sentences(text) if extract_facts(s)]
    if len(cands) < 1:
        return None
    cands.sort(key=len)
    s = cands[len(cands) // 2]
    out = text.replace(s, "", 1).strip()
    return out if out else None


_VERDICT_FLIPS = [
    ("驳回", "支持"),
    ("支持", "驳回"),
    ("不予支持", "予以支持"),
    ("予以支持", "不予支持"),
    ("解除", "维持"),
    ("维持", "解除"),
]


def _flip_verdict(text: str) -> Optional[str]:
    """反转判决结论。注意先长后短的替换顺序，否则 '不予支持' 会被拆坏。"""
    for src, dst in sorted(_VERDICT_FLIPS, key=lambda kv: -len(kv[0])):
        if src in text:
            return text.replace(src, dst, 1)
    return None


_PARTY_RE = re.compile(r"原告([\u4e00-\u9fa5]{2,4}?)(?=[，。、；,.:：]|与)")


def _swap_party(text: str) -> Optional[str]:
    """把第一个出现的原告姓名整体换成"张某"，语义上换了主体。"""
    m = _PARTY_RE.search(text)
    if not m:
        return None
    return text[: m.start(1)] + "张某" + text[m.end(1) :]


_HALLUC = "此外，被告还应支付精神损害抚慰金人民币五万元。"


def _inject_halluc(text: str) -> Optional[str]:
    """掺入一条参考里不可能有的、但形式合法的额外事实。"""
    return text + _HALLUC


def _swap_adjacent_sentences(text: str) -> Optional[str]:
    """调换两个相邻句子的顺序：信息完全不变，只有语序变了。"""
    sents = split_sentences(text)
    if len(sents) < 3:
        return None
    i = len(sents) // 2 - 1
    a, b = sents[i], sents[i + 1]
    return text.replace(a + b, b + a, 1) if (a + b) in text else None


_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*元")


def _rewrite_unit(text: str) -> Optional[str]:
    """48000元 -> 4.8万元：语义完全等价，字面几乎不重叠。"""
    for m in _UNIT_RE.finditer(text):
        value = float(m.group(1))
        if value >= 10000 and value % 1000 == 0:
            new = f"{value / 10000:g}万元"
            return text[: m.start()] + new + text[m.end() :]
        if value >= 1000:
            new = f"{value / 10000:g}万元"
            return text[: m.start()] + new + text[m.end() :]
    return None


BREAKING = {
    "d_digit": _flip_digit,
    "d_fact": _drop_fact_sentence,
    "d_verdict": _flip_verdict,
    "d_party": _swap_party,
    "d_halluc": _inject_halluc,
}
NEUTRAL = {
    "e_order": _swap_adjacent_sentences,
    "e_unit": _rewrite_unit,
}


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
class RougeMetric:
    name = "rouge_total"
    official = True

    def __init__(self, mode: str = "jieba"):
        self.mode = mode

    def __call__(self, candidate: str, reference: str) -> float:
        s = score_pair(candidate.strip(), reference.strip(), mode=self.mode)
        return 0.2 * s["rouge-1-f"] + 0.4 * s["rouge-2-f"] + 0.4 * s["rouge-l-f"]


class RougeLMetric(RougeMetric):
    name = "rouge_l"

    def __call__(self, candidate: str, reference: str) -> float:
        return score_pair(candidate.strip(), reference.strip(), mode=self.mode)["rouge-l-f"]


class EmbeddingMetric:
    """句向量余弦。比 BERTScore 更强，但**同样是整体语义相似度**，
    预期会在"数字抄错"上失灵 —— 这正是本测试要暴露的东西。"""

    name = "embed_cos"

    def __init__(self, encoder_path: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(encoder_path, device="cpu")

    def __call__(self, candidate: str, reference: str) -> float:
        import numpy as np

        emb = self.model.encode(
            [candidate.strip(), reference.strip()],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return float(np.dot(emb[0], emb[1]))


class JudgeCmdMetric:
    """把 (candidate, reference) 逐行喂给外部命令，读回一行一个分数。

    服务器上跑大模型 judge 时用这个，本地不需要装模型：
        --judge-cmd "python scripts/judge_score.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct"
    """

    def __init__(self, name: str, cmd: str):
        self.name = name
        self.cmd = cmd
        self.scores: Dict[Tuple[str, str], float] = {}

    def batch(self, pairs: List[Tuple[str, str]]) -> None:
        payload = "\n".join(
            json.dumps({"candidate": c, "reference": r}, ensure_ascii=False) for c, r in pairs
        )
        proc = subprocess.run(
            self.cmd, shell=True, input=payload, capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f"judge 命令失败：{proc.stderr[-500:]}")
        for (c, r), line in zip(pairs, proc.stdout.strip().splitlines()):
            self.scores[(c, r)] = float(line)

    def __call__(self, candidate: str, reference: str) -> float:
        return self.scores[(candidate, reference)]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def load_records(pattern: str, limit: Optional[int]) -> List[dict]:
    recs: List[dict] = []
    seen = set()
    for p in sorted(ROOT.glob(pattern)):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            recs.append(r)
    if limit:
        recs = recs[:limit]
    return recs


def build_cases(recs: List[dict], seed: int = 0) -> List[dict]:
    """产出 (perturbation, candidate, reference, source) 列表，基线扰动名为 'base'。"""
    random.seed(seed)
    cases = []
    for r in recs:
        ref, src, out = r["reference"], r["source"], r["output"]
        cases.append({"id": r["id"], "kind": "base", "text": out, "reference": ref, "source": src})
        for kind, fn in BREAKING.items():
            t = fn(out)
            if t and t != out:
                cases.append({"id": r["id"], "kind": kind, "text": t, "reference": ref, "source": src})
        for kind, fn in NEUTRAL.items():
            t = fn(out)
            if t and t != out:
                cases.append({"id": r["id"], "kind": kind, "text": t, "reference": ref, "source": src})
    return cases


def summarize(cases: List[dict], metrics: List[Callable], kinds: List[str]) -> None:
    # 打分
    for c in cases:
        c["scores"] = {m.name: m(c["text"], c["reference"]) for m in metrics}

    by_id: Dict[str, Dict[str, dict]] = {}
    for c in cases:
        by_id.setdefault(c["id"], {})[c["kind"]] = c

    print(f"\n样本 {len(by_id)} 条，扰动样本 {sum(len(v) for v in by_id.values())} 条")
    header = f"{'扰动':<10}{'n':>6}" + "".join(f"{m.name:>16}" for m in metrics)
    print("\nΔ 相对基线（破坏型越负越好；等价型越接近 0 越好）")
    print(header)
    print("-" * len(header))
    deltas: Dict[str, Dict[str, float]] = {}
    for kind, label in [("d_digit", "数字改错"), ("d_fact", "漏事实句"), ("d_verdict", "结论反转"),
                        ("d_party", "主体替换"), ("d_halluc", "幻觉注入"),
                        ("e_order", "句序调换"), ("e_unit", "单位改写")]:
        rows = [(k, v) for k, v in by_id.items() if kind in v and "base" in v]
        if not rows:
            continue
        cells = []
        for m in metrics:
            d = st.mean(v[kind]["scores"][m.name] - v["base"]["scores"][m.name] for k, v in rows)
            deltas.setdefault(m.name, {})[kind] = d
            cells.append(f"{d:>+16.4f}")
        print(f"{label:<10}{len(rows):>6}" + "".join(cells))

    print("\n区分度 = -mean(Δ破坏) / mean(|Δ等价|)")
    for m in metrics:
        d = deltas.get(m.name, {})
        brk = [d[k] for k in BREAKING if k in d]
        neu = [abs(d[k]) for k in NEUTRAL if k in d]
        if not brk or not neu:
            continue
        b, nscore = st.mean(brk), max(st.mean(neu), 1e-9)
        print(f"  {m.name:<16} 破坏 {b:+.4f}   等价 {st.mean(neu):.4f}   区分度 {-b / nscore:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="指标区分度压力测试")
    ap.add_argument("--input", required=True, help="三元组 jsonl 的 glob（相对项目根）")
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条，快速试跑")
    ap.add_argument("--rouge-mode", default="jieba", choices=["char", "jieba"])
    ap.add_argument("--encoder", default=None, help="句向量模型目录，加了才测 embed_cos")
    ap.add_argument("--judge-cmd", default=None, help="外部 judge 命令，配合 --judge-name")
    ap.add_argument("--judge-name", default="judge")
    ap.add_argument("--dump", default=None, help="把扰动语料写到这个 jsonl，供别处复用")
    args = ap.parse_args()

    recs = load_records(args.input, args.limit)
    if not recs:
        raise SystemExit(f"没读到数据：{args.input}")
    cases = build_cases(recs)
    if args.dump:
        p = ROOT / args.dump
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"已写出 {len(cases)} 条扰动语料到 {p}")

    metrics: List[Callable] = [
        RougeMetric(args.rouge_mode),
        RougeLMetric(args.rouge_mode),
    ]
    if args.encoder:
        metrics.append(EmbeddingMetric(args.encoder))
    judge = None
    if args.judge_cmd:
        judge = JudgeCmdMetric(args.judge_name, args.judge_cmd)
        pairs = [(c["text"], c["reference"]) for c in cases]
        judge.batch(pairs)
        metrics.append(judge)

    summarize(cases, metrics, list(BREAKING) + list(NEUTRAL))


if __name__ == "__main__":
    main()
