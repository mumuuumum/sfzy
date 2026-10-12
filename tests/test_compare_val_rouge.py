"""tools/compare_val_rouge.py 的验收。

这里钉的是三条"错了不会报错、只会让数字悄悄变样"的口径：

  1. **只在共同 id 上比** —— 某个模型少跑几条时，不能把"少做的那几条"算进差值；
  2. **宏平均**（逐样本先算分再平均），不是把所有预测拼起来算一次；
  3. **重复 id 只取第一次出现** —— 分片重跑/续跑追加时会重复，取后者就不可复现。

    python -m pytest tests/test_compare_val_rouge.py -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "compare_val_rouge", ROOT / "tools" / "compare_val_rouge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _write(path: Path, records) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def _records(ids, output="原告胜诉。", reference="原告胜诉。"):
    return [
        {"id": i, "source": "s", "reference": reference, "output": output}
        for i in ids
    ]


def test_重复id只保留第一次出现(tmp_path):
    path = _write(tmp_path / "a.jsonl", [
        {"id": "1", "reference": "甲", "output": "第一次"},
        {"id": "1", "reference": "甲", "output": "第二次"},
    ])
    outs, refs, dup = TOOL.load_outputs(str(path))
    assert outs == {"1": "第一次"} and dup == 1


def test_只在共同id上比较(tmp_path):
    a = _write(tmp_path / "a.jsonl", _records(["1", "2"]))
    b = _write(tmp_path / "b.jsonl", _records(["2", "3"]))
    outs_a, refs_a, _ = TOOL.load_outputs(str(a))
    outs_b, refs_b, _ = TOOL.load_outputs(str(b))
    common = set(outs_a) & set(outs_b)
    assert common == {"2"}          # "3" 只有 b 有、"" 只有 a 有，都不进比较
    assert set(refs_a) == set(outs_a) and set(refs_b) == set(outs_b)


def test_宏平均与score_corpus口径一致(tmp_path):
    """工具里的语料级分数必须和项目评测路径（`score_corpus`）完全一致。"""
    from sfzy.eval.rouge import score_corpus

    ids = ["1", "2", "3"]
    preds = {"1": "原告胜诉。被告承担诉讼费。", "2": "驳回。", "3": "原告与被告达成调解。"}
    refs = {"1": "原告胜诉。被告承担诉讼费。", "2": "驳回起诉。", "3": "原告与被告达成调解协议。"}
    per_sample = {
        i: TOOL.score_pair(preds[i], refs[i], mode="char") for i in ids
    }
    mine = TOOL.corpus_scores(per_sample)
    official = score_corpus(preds, refs, mode="char")
    for key in ("rouge-1-f", "rouge-2-f", "rouge-l-f", "overall"):
        assert abs(mine[key] - official[key]) < 1e-12, key


def test_成对bootstrap在同向差值下区间不含0():
    diffs = [0.1] * 40 + [0.2] * 40     # 全部为正 ⇒ 区间下界必须 > 0
    mean, lo, hi = TOOL.paired_bootstrap(diffs, n_boot=500, seed=0)
    assert lo > 0 and lo <= mean <= hi


def test_成对bootstrap在纯噪声下区间跨0():
    diffs = [0.1, -0.1] * 100
    _, lo, hi = TOOL.paired_bootstrap(diffs, n_boot=500, seed=0)
    assert lo < 0 < hi


def test_端到端跑通并核对reference(tmp_path, capsys, monkeypatch):
    """main() 能跑通、能对账 reference，并在 reference 不一致时说出来。"""
    val = _write(tmp_path / "val.jsonl", [
        {"id": "1", "source": "s", "summary": "甲。乙。"},
        {"id": "2", "source": "s", "summary": "丙。"},
    ])
    good = _write(tmp_path / "good.jsonl", [
        {"id": "1", "source": "s", "reference": "甲。乙。", "output": "甲。乙。"},
        {"id": "2", "source": "s", "reference": "丙。", "output": "丙。"},
    ])
    bad = _write(tmp_path / "bad.jsonl", _records(["1", "2"], reference="错的参考。"))

    monkeypatch.setattr(
        "sys.argv",
        ["compare_val_rouge.py", "--pred", f"good={good}", f"bad={bad}",
         "--ref", str(val), "--mode", "char"],
    )
    TOOL.main()
    out = capsys.readouterr().out
    assert "共同可评样本 2 条" in out
    assert "bad: reference 与" in out and "不一致 2 条" in out


def test_没有共同id时直接报错(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.jsonl", _records(["1"]))
    b = _write(tmp_path / "b.jsonl", _records(["2"]))
    monkeypatch.setattr(
        "sys.argv",
        ["compare_val_rouge.py", "--pred", f"a={a}", f"b={b}", "--mode", "char"],
    )
    with pytest.raises(SystemExit, match="没有共同 id"):
        TOOL.main()
