"""reward_spec.py 的验收测试：YAML → RewardSpec 的解析、组合与校验。

这些错误都必须在**启动时**报出来，而不是跑出一条其实没接裁判的曲线。

    python -m pytest tests/test_reward_spec.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sfzy.rl.reward_spec import (
    DEFAULT_GATE,
    PRESETS,
    RewardConfigError,
    RewardSpec,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 解析

def test_老的mode预设仍可用():
    spec = RewardSpec.from_config({"mode": "fact_judge"})
    assert [t.name for t in spec.enabled_terms] == ["rouge_l", "fact_consistency"]
    assert spec.required_signals == {"fact_consistency"}
    assert spec.needs_judge() is True


def test_不给配置时默认就是fact_judge预设():
    spec = RewardSpec.from_config(None)
    assert spec.needs_judge() is True


def test_显式terms可以单独开关():
    spec = RewardSpec.from_config({"terms": {
        "rouge_l": {"enabled": True, "weight": 1.0},
        "fact_consistency": {"enabled": False, "weight": 1.0},
    }})
    assert [t.name for t in spec.enabled_terms] == ["rouge_l"]
    assert spec.needs_judge() is False
    assert spec.required_signals == set()


def test_不写enabled默认是开():
    spec = RewardSpec.from_config({"terms": {"fact_consistency": {}}})
    assert spec.terms["fact_consistency"].enabled is True
    assert spec.terms["fact_consistency"].weight == pytest.approx(0.7)   # 来自注册表


# ---------------------------------------------------------------- 权重

def test_权重按启用项归一到一():
    spec = RewardSpec.from_config({"terms": {
        "rouge_l": {"weight": 30},
        "fact_consistency": {"weight": 70},
    }})
    weights = spec.normalized_weights()
    assert weights["rouge_l"] == pytest.approx(0.3)
    assert weights["fact_consistency"] == pytest.approx(0.7)


def test_关掉归一化时保留原值():
    spec = RewardSpec.from_config({
        "normalize_weights": False,
        "terms": {"rouge_l": {"weight": 30}},
    })
    assert spec.normalized_weights() == {"rouge_l": 30.0}


def test_归一化按实际启用的项算():
    """关掉一项后，另一项应该独占权重 —— 否则总分会凭空缩水。"""
    spec = RewardSpec.from_config({"terms": {
        "rouge_l": {"weight": 30},
        "fact_consistency": {"enabled": False, "weight": 70},
    }})
    assert spec.normalized_weights() == {"rouge_l": pytest.approx(1.0)}


# ---------------------------------------------------------------- 校验

def test_未知term报错并列出可用名():
    with pytest.raises(RewardConfigError, match="未知的 reward term"):
        RewardSpec.from_config({"terms": {"不存在的项": {"enabled": True}}})


def test_一个term都没启用报错():
    with pytest.raises(RewardConfigError, match="至少要有"):
        RewardSpec.from_config({"terms": {"rouge_l": {"enabled": False}}})


def test_启用项权重全零报错():
    with pytest.raises(RewardConfigError, match="权重全是 0"):
        RewardSpec.from_config({"terms": {"rouge_l": {"weight": 0.0}}})


def test_负权重报错():
    with pytest.raises(RewardConfigError, match="不能为负"):
        RewardSpec.from_config({"terms": {"rouge_l": {"weight": -1}}})


def test_未知mode报错():
    with pytest.raises(RewardConfigError, match="未知的 reward.mode"):
        RewardSpec.from_config({"mode": "rouge_only"})


def test_terms必须是映射():
    with pytest.raises(RewardConfigError, match="必须是"):
        RewardSpec.from_config({"terms": ["rouge_l"]})


# ---------------------------------------------------------------- 门控

def test_门控默认开启且带默认阈值():
    spec = RewardSpec.from_config({"terms": {"rouge_l": {}}})
    assert spec.gate.enabled is True
    assert spec.gate.cfg["min_chars"] == DEFAULT_GATE["min_chars"]


def test_门控可整块关掉():
    spec = RewardSpec.from_config({"gate": {"enabled": False}, "terms": {"rouge_l": {}}})
    assert spec.gate.enabled is False


def test_门控可只覆盖部分阈值():
    spec = RewardSpec.from_config({"gate": {"min_chars": 10}, "terms": {"rouge_l": {}}})
    assert spec.gate.cfg["min_chars"] == 10
    assert spec.gate.cfg["require_result_marker"] == DEFAULT_GATE["require_result_marker"]


def test_预设表里有fact_judge():
    assert "fact_judge" in PRESETS


# ---------------------------------------------------------------- 随仓库的配置

def _load(name: str):
    from sfzy.config import load_config

    return load_config(ROOT / "configs" / name)


@pytest.mark.parametrize("name", ["grpo_fact_only_t4.yaml", "grpo_fact_only_a100.yaml"])
def test_正式配置只开事实一致性(name):
    cfg = _load(name)
    spec = RewardSpec.from_config(cfg.path_("rl.reward"))
    assert [t.name for t in spec.enabled_terms] == ["fact_consistency"]
    assert spec.normalized_weights() == {"fact_consistency": pytest.approx(1.0)}
    assert spec.needs_judge() is True
    # 脚本不带参数也能跑：SFT 起点和裁判后端都写在配置里
    assert cfg.path_("rl.sft_adapter")
    assert cfg.path_("semantic.backend") == "fact"
    # 门控保留为全局开关
    assert spec.gate.enabled is True


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
