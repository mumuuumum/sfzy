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
from typing import Any

import pytest

from sfzy.eval.metrics import (
    CachedJudgeScorer,
    ListwiseRankScorer,
    build_judge_prompt,
    build_rank_prompt,
    build_scorer,
    load_rubric,
    normalize_item,
    parse_judge_score,
    parse_ranking,
    verify_judge_output,
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
    assert str(rubric["version"]).startswith("v1")
    assert int(rubric["scale"]) == 100


def test_rubric_要求逐条抄录而不是整体印象打分():
    """v1 的锚点被裁判原样背进"理由"栏，20 条里 16 条满分。
    v1.1 改成"先枚举要素、再从候选里原样抄出对应句、分数由枚举结果算出来"
    —— 抄不出句子却给分是无效判定，而编不出没读到的句子。"""
    tpl = load_rubric()["template_with_source"]
    assert "逐字复制" in tpl
    assert "评分作废" in tpl                      # 伪造抄录的后果要写明
    assert "不要抄参考摘要里的句子" in tpl        # 实测 28% 的抄录抄的是参考
    assert "要素覆盖分" in tpl


def test_rubric_把可机械计算的项交给程序():
    """长度、金额、日期都是能算的，交给 LLM 评只会得到幻觉
    （实测：长度比 1.73 的候选被判"长度与参考相当"拿了满分）。"""
    tpl = load_rubric()["template_with_source"]
    assert "由程序另行核对" in tpl


def test_rubric_给了算分公式():
    """分数必须是枚举结果的函数，不能留"凭印象调分"的口子。"""
    tpl = load_rubric()["template_with_source"]
    assert "÷ (2 × 要素条数)" in tpl
    assert "最低 0" in tpl


def test_rubric_要求要素只从参考拆():
    """v1.1 的要素是从候选里挑的，于是必然全中、必然满分。
    实测：某条候选的 6 条要素里 5 条直接来自候选，参考里两条关键推理
    压根没被拆出来考。"""
    tpl = load_rubric()["template_with_source"]
    assert "只看【参考摘要】拆要素" in tpl
    assert "不超过 15 字" in tpl


# ---------------------------------------------------------------- 裁判质检

REF = "原被告系租赁合同纠纷，被告未按期支付租金，法院判决解除合同。"
CAND = "原被告系租赁合同关系，判决解除双方签订的租赁合同。"


def test_质检_真抄录算命中():
    raw = ('要素1 [2分] 抄录："原被告系租赁合同关系"\n'
           '要素2 [0分] 抄录：无\n'
           '总分：50')
    a = verify_judge_output(raw, REF, CAND)
    assert a["n_quotes"] == 1 and a["n_quote_hits"] == 1
    assert a["quote_hit_rate"] == 1.0


def test_质检_抄了参考的句子要被抓出来():
    """★ 这是实测抓到的真问题：83 条抄录里 23 条（28%）在候选里找不到、
    却能在参考里找到。裁判一边说"抄的是候选"，一边抄的是参考。"""
    raw = ('要素2 [1分] 抄录："被告未按期支付租金"\n'
           '总分：60')
    a = verify_judge_output(raw, REF, CAND)
    assert a["n_quote_hits"] == 0
    assert a["n_quotes_from_ref_only"] == 1
    assert "被告未按期支付租金" in a["suspect_quotes"][0]


def test_质检_标点差异不算没找到():
    """裁判抄录时常顺手改标点，不该因此判为编造。"""
    raw = '要素1 [2分] 抄录："原被告系租赁合同关系。"\n总分：90'
    assert verify_judge_output(raw, REF, CAND)["quote_hit_rate"] == 1.0


def test_质检_改写太多算没找到():
    """只保留 70% 以上连续片段才算命中；改得面目全非就是在编。"""
    raw = '要素1 [2分] 抄录："双方因房屋租赁产生争议并解除合同"\n总分：90'
    assert verify_judge_output(raw, REF, CAND)["quote_hit_rate"] == 0.0


def test_质检_要素出处率():
    raw = ("1. 被告未按期支付租金\n"
           "2. 法院支持了原告的全部请求\n"
           '要素1 [2分] 抄录："原被告系租赁合同关系"\n总分：80')
    a = verify_judge_output(raw, REF, CAND)
    assert a["n_elements"] == 2          # 第二步的"要素1 [2分] 抄录…"不算要素行
    assert a["n_element_hits"] == 1      # 第二条在参考里不存在


def test_质检_没有抄录时返回None不报错():
    a = verify_judge_output("总分：100", REF, CAND)
    assert a["n_quotes"] == 0 and a["quote_hit_rate"] is None


# ---------------------------------------------------------------- 组内排序裁判

def test_排序解析_取最后一行的字母序列():
    text = "理由：C 的判决结果与参考一致。\nA 太长了。\n\nC A F B D E G H"
    assert parse_ranking(text, 8) == [2, 0, 5, 1, 3, 4, 6, 7]


def test_排序解析_字母不全不算数():
    """理由里随口提到一个字母，不能被当成排序结果。"""
    assert parse_ranking("A 比 B 好，所以选 A。", 5) is None


def test_排序解析_没有输出返回None():
    assert parse_ranking("", 5) is None


def test_排序解析_忽略无关字母():
    """解释性文字里的大写英文单词不该污染结果。"""
    text = "理由：OK，C 覆盖了全部要素。\n\nC A F B D E G H"
    assert parse_ranking(text, 8) is not None


def test_排序提示词_包含全部候选和字母():
    rubric = load_rubric("configs/judge_rank_rubric.yaml")
    cands = ["候选一的内容", "候选二的内容", "候选三的内容", "候选四的内容", "候选五的内容"]
    p = build_rank_prompt(rubric, "参考", cands, "原文")
    for i, c in enumerate(cands):
        assert f"[{'ABCDE'[i]}] {c}" in p
    assert "{candidates}" not in p and "{n}" not in p


def test_排序提示词_强调参考覆盖而不是候选精确率():
    """点式裁判栽在算 precision（候选→参考）而不是 recall（参考→候选）。
    排序版必须把这一点写死在标准里。"""
    tpl = load_rubric("configs/judge_rank_rubric.yaml")["template_with_source"]
    assert "参考里有的有没有被写出来" in tpl
    assert "不要因为某条更长或更短就排前" in tpl


class _FakeRankScorer(ListwiseRankScorer):
    """不加载模型：按预设顺序返回生成结果，用来验证名次→分数的换算。"""

    def __init__(self, ranking_text: str, group_size: int, repeat: int = 1, **kw: Any):
        self.group_size = group_size
        self.repeat = repeat
        self.seed = 0
        self.rubric = load_rubric("configs/judge_rank_rubric.yaml")
        self.scale = 100
        self._text = ranking_text
        self.samples = 1
        self.n_truncated = 0
        self.last_raw = []

    def generate_texts(self, prompt: str):
        return [self._text]


def test_排序裁判_名次换算成分数():
    # 固定输出 "A B C D E"：A 最好 → 100 分，E 最差 → 0 分
    scorer = _FakeRankScorer("理由…\nA B C D E", group_size=5)
    scores = scorer.score_batch([{"candidate": f"c{i}", "reference": "r"} for i in range(5)])
    assert scores == [100.0, 75.0, 50.0, 25.0, 0.0]


def test_排序裁判_倒数第一名拿零分():
    scorer = _FakeRankScorer("理由…\nE D C B A", group_size=5)
    scores = scorer.score_batch([{"candidate": f"c{i}", "reference": "r"} for i in range(5)])
    assert scores == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_排序裁判_多组各自排名():
    scorer = _FakeRankScorer("A B C D E", group_size=5)
    items = [{"candidate": f"c{i}", "reference": "r"} for i in range(10)]
    scores = scorer.score_batch(items)
    assert scores[:5] == [100.0, 75.0, 50.0, 25.0, 0.0]
    assert scores[5:] == scores[:5]


def test_排序裁判_条数不是整数倍要报错():
    """错位分组会把不同 prompt 的候选排到一起，宁可不打分。"""
    scorer = _FakeRankScorer("A B C D E", group_size=5)
    with pytest.raises(ValueError, match="每 5 条一组"):
        scorer.score_batch([{"candidate": "c", "reference": "r"}] * 7)


def test_排序裁判_解析失败给None而不是猜():
    scorer = _FakeRankScorer("我不知道怎么排", group_size=5)
    assert scorer.score_batch([{"candidate": f"c{i}", "reference": "r"} for i in range(5)]) == [None] * 5


def test_排序裁判_组太小要报错():
    with pytest.raises(ValueError, match="至少要 2 条"):
        ListwiseRankScorer(model_path="x", group_size=1)


def test_rubric_载入不存在的文件要报错():
    with pytest.raises(FileNotFoundError):
        load_rubric("configs/不存在.yaml")


# ---------------------------------------------------------------- build_scorer 分流

def test_build_scorer_none后端返回None():
    assert build_scorer({"backend": "none"}) is None
    assert build_scorer(None) is None


def test_build_scorer_fact后端缺模型给明确报错():
    """fact 后端不吃 rubric，走的是六要素 Judge 那条路。
    没给模型时必须当场报错，而不是构造出一个跑不了的裁判。"""
    with pytest.raises(ValueError, match="backend=fact"):
        build_scorer({"backend": "fact"})


def test_build_scorer_未知后端的提示里列出fact():
    """后端名写错时要能一眼看出可选项 —— fact 必须出现在提示里。"""
    with pytest.raises(ValueError, match="fact"):
        build_scorer({"backend": "不存在的后端"})
