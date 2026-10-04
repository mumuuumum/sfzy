"""reward_spec.py 的验收测试：YAML → RewardSpec 的解析、组合与校验。

核心规矩：**所有权重只能来自配置文件**。
缺 term 权重、缺内部权重、键写错、和不为 1 —— 全部必须在启动时报错，
而不是悄悄退回一个藏在源码里的默认值。

    python -m pytest tests/test_reward_spec.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sfzy.rl.reward_spec import DEFAULT_GATE, RewardConfigError, RewardSpec

ROOT = Path(__file__).resolve().parents[1]

ELEMENT_WEIGHTS = {
    "case_type": 0.05,
    "plaintiff_claims": 0.15,
    "defendant_defenses": 0.10,
    "court_facts": 0.25,
    "legal_basis": 0.15,
    "judgment_result": 0.30,
}


def fact_term(**over):
    entry = {"enabled": True, "weight": 0.7, "element_weights": dict(ELEMENT_WEIGHTS)}
    entry.update(over)
    return entry


def spec(terms, **over):
    raw = {"terms": terms}
    raw.update(over)
    return RewardSpec.from_config(raw)


BOTH = {"rouge_l": {"enabled": True, "weight": 0.3}, "fact_consistency": fact_term()}


# ---------------------------------------------------------------- 解析

def test_解析两个reward():
    s = spec(dict(BOTH))
    assert [t.name for t in s.enabled_terms] == ["rouge_l", "fact_consistency"]
    assert s.required_signals == {"fact_consistency"}
    assert s.needs_judge() is True
    assert s.terms["rouge_l"].weight == pytest.approx(0.3)
    assert s.terms["fact_consistency"].weight == pytest.approx(0.7)


def test_可以单独开关():
    s = spec({"rouge_l": {"enabled": True, "weight": 1.0},
              "fact_consistency": {"enabled": False}})
    assert [t.name for t in s.enabled_terms] == ["rouge_l"]
    assert s.needs_judge() is False
    # 关掉的 term 不要求写权重，也不要求写内部权重
    assert s.terms["fact_consistency"].weight == 0.0
    assert s.terms["fact_consistency"].options == {}


def test_关掉的term也可以仍然写全():
    s = spec({"rouge_l": {"enabled": True, "weight": 1.0},
              "fact_consistency": fact_term(enabled=False)})
    assert s.needs_judge() is False
    assert s.terms["fact_consistency"].options["element_weights"]["court_facts"] == 0.25


# ---------------------------------------------------------------- 权重（term 级）

def test_term权重按启用项归一到一():
    s = spec({
        "rouge_l": {"weight": 30},
        "fact_consistency": fact_term(weight=70),
    })
    weights = s.normalized_weights()
    assert weights["rouge_l"] == pytest.approx(0.3)
    assert weights["fact_consistency"] == pytest.approx(0.7)


def test_关掉归一化时保留原值():
    s = spec({"rouge_l": {"weight": 30}}, normalize_weights=False)
    assert s.normalized_weights() == {"rouge_l": 30.0}


def test_归一化按实际启用的项算():
    s = spec({"rouge_l": {"weight": 30}, "fact_consistency": {"enabled": False}})
    assert s.normalized_weights() == {"rouge_l": pytest.approx(1.0)}


def test_启用却没写weight要报错():
    with pytest.raises(RewardConfigError, match="没有写 weight"):
        spec({"rouge_l": {"enabled": True}})


def test_权重非数字要报错():
    with pytest.raises(RewardConfigError, match="必须是数字"):
        spec({"rouge_l": {"weight": "高一点"}})


def test_负权重报错():
    with pytest.raises(RewardConfigError, match="不能为负"):
        spec({"rouge_l": {"weight": -1}})


def test_启用项权重全零报错():
    with pytest.raises(RewardConfigError, match="权重全是 0"):
        spec({"rouge_l": {"weight": 0.0}})


# ---------------------------------------------------------------- 权重（reward 内部）

def test_内部权重被解析出来():
    s = spec(dict(BOTH))
    assert s.terms["fact_consistency"].options["element_weights"] == ELEMENT_WEIGHTS
    assert s.judge_term_options() == {"fact_consistency": {"element_weights": ELEMENT_WEIGHTS}}


def test_启用fact却没写element_weights要报错():
    with pytest.raises(RewardConfigError, match="element_weights"):
        spec({"fact_consistency": {"enabled": True, "weight": 1.0}})


def test_内部权重缺少键要报错():
    bad = {"case_type": 0.1, "plaintiff_claims": 0.9}          # 少四个
    with pytest.raises(RewardConfigError, match="键不对"):
        spec({"fact_consistency": fact_term(element_weights=bad)})


def test_内部权重多了未知键要报错():
    bad = dict(ELEMENT_WEIGHTS, unknown_element=0.0)
    with pytest.raises(RewardConfigError, match="键不对"):
        spec({"fact_consistency": fact_term(element_weights=bad)})


def test_内部权重之和必须为1():
    bad = dict(ELEMENT_WEIGHTS)
    bad["judgment_result"] = 0.50                              # 和变成 1.2
    with pytest.raises(RewardConfigError, match="之和必须为 1"):
        spec({"fact_consistency": fact_term(element_weights=bad)})


def test_内部权重不能为负():
    bad = dict(ELEMENT_WEIGHTS)
    bad["case_type"] = -0.05
    bad["judgment_result"] = 0.40
    with pytest.raises(RewardConfigError, match="不能为负"):
        spec({"fact_consistency": fact_term(element_weights=bad)})


def test_内部权重不是映射要报错():
    with pytest.raises(RewardConfigError, match="必须是"):
        spec({"fact_consistency": fact_term(element_weights=[0.1, 0.9])})


def test_rouge_l没有内部权重字段():
    """写错字段名（比如照抄老的 semantic.weights）要当场报错。"""
    with pytest.raises(RewardConfigError, match="不认识的字段"):
        spec({"fact_consistency": {**fact_term(), "weights": dict(ELEMENT_WEIGHTS)}})


# ---------------------------------------------------------------- 两个 judge 项

def test_两个judge项各有各的内部权重():
    s = spec({
        "fact_consistency": fact_term(weight=0.6),
        "element_coverage": {"enabled": True, "weight": 0.4,
                             "element_weights": dict(ELEMENT_WEIGHTS)},
    })
    assert s.required_signals == {"fact_consistency", "element_coverage"}
    assert set(s.judge_term_options()) == {"fact_consistency", "element_coverage"}
    assert s.normalized_weights() == {
        "fact_consistency": pytest.approx(0.6), "element_coverage": pytest.approx(0.4),
    }


def test_启用覆盖率却没写内部权重要报错():
    with pytest.raises(RewardConfigError, match="element_weights"):
        spec({"fact_consistency": fact_term(),
              "element_coverage": {"enabled": True, "weight": 0.5}})


def test_覆盖率的内部权重和也必须为1():
    bad = dict(ELEMENT_WEIGHTS)
    bad["court_facts"] = 0.50
    with pytest.raises(RewardConfigError, match="之和必须为 1"):
        spec({"element_coverage": {"enabled": True, "weight": 1.0,
                                   "element_weights": bad}})


def test_未知字段要报错():
    with pytest.raises(RewardConfigError, match="不认识的字段"):
        spec({"rouge_l": {"enabled": True, "weight": 1.0, "typo": 1}})


# ---------------------------------------------------------------- 结构校验

def test_未知term报错并列出可用名():
    with pytest.raises(RewardConfigError, match="未知的 reward term"):
        spec({"不存在的项": {"weight": 1.0}})


def test_一个term都没启用报错():
    with pytest.raises(RewardConfigError, match="至少要有"):
        spec({"rouge_l": {"enabled": False}})


def test_没有terms报错():
    with pytest.raises(RewardConfigError, match="terms"):
        RewardSpec.from_config({"normalize_weights": True})


def test_没有reward配置报错():
    with pytest.raises(RewardConfigError, match="rl.reward"):
        RewardSpec.from_config(None)


def test_terms必须是映射():
    with pytest.raises(RewardConfigError, match="必须是"):
        RewardSpec.from_config({"terms": ["rouge_l"]})


# ---------------------------------------------------------------- 门控

def test_门控默认开启且带默认阈值():
    s = spec({"rouge_l": {"weight": 1.0}})
    assert s.gate.enabled is True
    assert s.gate.cfg["min_chars"] == DEFAULT_GATE["min_chars"]


def test_门控可整块关掉():
    s = spec({"rouge_l": {"weight": 1.0}}, gate={"enabled": False})
    assert s.gate.enabled is False


def test_门控可只覆盖部分阈值():
    s = spec({"rouge_l": {"weight": 1.0}}, gate={"min_chars": 10})
    assert s.gate.cfg["min_chars"] == 10
    assert s.gate.cfg["require_result_marker"] == DEFAULT_GATE["require_result_marker"]


# ---------------------------------------------------------------- 随仓库的配置

def _load(name: str):
    from sfzy.config import load_config

    return load_config(ROOT / "configs" / name)


@pytest.mark.parametrize("name", ["grpo_fact_only_t4.yaml", "grpo_fact_only_a100.yaml"])
def test_正式配置只开事实一致性(name):
    cfg = _load(name)
    s = RewardSpec.from_config(cfg.path_("rl.reward"))
    assert [t.name for t in s.enabled_terms] == ["fact_consistency"]
    assert s.normalized_weights() == {"fact_consistency": pytest.approx(1.0)}
    assert s.needs_judge() is True
    # 脚本不带参数也能跑：SFT 起点和裁判后端都写在配置里
    assert cfg.path_("rl.sft_adapter")
    assert cfg.path_("semantic.backend") == "fact"
    # 门控保留为全局开关
    assert s.gate.enabled is True
    # 六要素权重来自配置文件（在 grpo_fact_judge.yaml 里写全，子配置继承）
    assert s.terms["fact_consistency"].options["element_weights"] == ELEMENT_WEIGHTS


def test_旧的semantic_weights不再被接受():
    """权重只能写在 reward term 里 —— semantic.weights 这条路已经封掉。"""
    from sfzy.eval.metrics import build_scorer

    with pytest.raises(ValueError, match="semantic.weights"):
        build_scorer({
            "backend": "fact",
            "model": "不存在的模型",
            "weights": dict(ELEMENT_WEIGHTS),
        })


def test_t4配置是float16():
    """T4 是 Turing，不支持 bf16 —— 策略和裁判都必须 float16。"""
    cfg = _load("grpo_fact_only_t4.yaml")
    assert cfg.path_("semantic.dtype") == "float16"
    assert cfg.path_("semantic.bnb_4bit_compute_dtype") == "float16"
    assert cfg.path_("semantic.load_in_4bit") is True


def test_a100配置用bf16全精度裁判():
    cfg = _load("grpo_fact_only_a100.yaml")
    assert cfg.path_("semantic.dtype") == "bfloat16"
    assert cfg.path_("semantic.load_in_4bit") is False


@pytest.mark.parametrize("name", ["grpo_fact_coverage_t4.yaml", "grpo_fact_coverage_a100.yaml"])
def test_合并配置同时开两个reward(name):
    cfg = _load(name)
    s = RewardSpec.from_config(cfg.path_("rl.reward"))
    assert [t.name for t in s.enabled_terms] == ["fact_consistency", "element_coverage"]
    assert s.normalized_weights() == {
        "fact_consistency": pytest.approx(0.7), "element_coverage": pytest.approx(0.3),
    }
    assert s.required_signals == {"fact_consistency", "element_coverage"}
    # 两个 reward 的内部权重都是配置给的那一份
    for term in s.enabled_terms:
        assert term.options["element_weights"] == ELEMENT_WEIGHTS
