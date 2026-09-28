"""eval/metrics.py 的验收测试（不依赖任何模型，全部是纯函数与缓存）。

重点：
  1. **分数解析取最后一个"总分"** —— rubric 要求逐维度写理由，维度行里
     也会出现数字，取错了会把维度分当总分，而且**不报错**
  2. **超量程判失败（None）而不是截断到边界** —— 失败和"很差"是两件事
  3. **模板选择**：有原文才用带原文的模板（查幻觉必须给原文）

    python -m pytest tests/test_semantic.py -q
"""

from __future__ import annotations

import json

import pytest

from sfzy.eval.metrics import (
    CachedJudgeScorer,
    build_judge_prompt,
    load_rubric,
    normalize_item,
    parse_judge_score,
)


# ---------------------------------------------------------------- 解析

def test_解析_取最后一行总分():
    """维度行里也有分值，必须取最后那个"总分"。"""
    text = (
        "维度 1 事实准确性：得分 25\n"
        "维度 2 要素完整性：得分 15\n"
        "维度 3 取舍与篇幅：得分 20\n"
        "维度 4 法律表述规范：得分 25\n"
        "总分：85"
    )
    assert parse_judge_score(text) == 85.0


def test_解析_中文冒号也算():
    assert parse_judge_score("最终分数: 72") == 72.0


def test_解析_没有关键字时退化成最后一个数字():
    assert parse_judge_score("我觉得可以给 66 分吧") == 66.0


def test_解析_超量程判失败():
    """185 不是"比 100 还好的满分"，是解析错了 —— 宁可留空。"""
    assert parse_judge_score("总分：185") is None


def test_解析_没有数字判失败():
    assert parse_judge_score("这个摘要还行") is None


# ---------------------------------------------------------------- 模板

def test_模板_没有原文时用短模板():
    rubric = load_rubric()
    prompt = build_judge_prompt(rubric, reference="参考摘要", candidate="候选摘要")
    assert "参考摘要" in prompt and "候选摘要" in prompt
    assert "{source}" not in prompt and "{reference}" not in prompt


def test_模板_给了原文时把原文也塞进去():
    """查幻觉必须给原文，否则裁判没法判断金额是不是编的。"""
    rubric = load_rubric()
    prompt = build_judge_prompt(
        rubric, reference="参考摘要", candidate="候选", source="判决书原文内容"
    )
    assert "判决书原文内容" in prompt
    assert "{source}" not in prompt


def test_模板_必须给出总分格式要求():
    """rubric 里不写"最后一行给总分"，解析器就持续失败。"""
    rubric = load_rubric()
    assert "总分" in build_judge_prompt(rubric, "r", "c")


# ---------------------------------------------------------------- 字段归一

def test_字段归一_兼容三种写法():
    assert normalize_item({"output": "a"})["candidate"] == "a"
    assert normalize_item({"candidate": "b"})["candidate"] == "b"
    assert normalize_item({"text": "c"})["candidate"] == "c"


# ---------------------------------------------------------------- 缓存后端

def test_缓存后端_按id取分(tmp_path):
    p = tmp_path / "j.jsonl"
    p.write_text(
        "\n".join([
            json.dumps({"id": "a", "judge_score": 88.0}),
            json.dumps({"id": "b", "judge_score": None}),   # 失败样本
            json.dumps({"id": "c", "judge_score": 12.5}),
        ]),
        encoding="utf-8",
    )
    scorer = CachedJudgeScorer([str(p)])
    out = scorer.score_batch([{"id": "a"}, {"id": "b"}, {"id": "z"}])
    assert out == [88.0, None, None]


def test_缓存后端_空缓存不报错(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert CachedJudgeScorer([str(p)]).score_batch([{"id": "a"}]) == [None]


# ---------------------------------------------------------------- rubric 文件

def test_rubric_必须带版本号():
    """训练日志会记版本号；不记的话半年后分不清哪次实验用的哪版标准。"""
    rubric = load_rubric()
    assert rubric["version"] == "v1"
    assert int(rubric["scale"]) == 100


def test_rubric_四个维度都写了锚点():
    """没有锚点，"80 分"和"85 分"就是随机的。"""
    tpl = load_rubric()["template_with_source"]
    for kw in ("事实准确性", "要素完整性", "取舍与篇幅", "法律表述规范"):
        assert kw in tpl
    assert tpl.count("25 分") >= 4      # 每档都有具体描述


def test_rubric_载入不存在的文件要报错():
    with pytest.raises(FileNotFoundError):
        load_rubric("configs/不存在.yaml")
