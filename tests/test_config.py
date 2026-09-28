"""config.py 的验收测试。

重点是 apply_overrides 的**路径校验** —— 它防的是一类静默事故：
`--override resume_from=...` 不带段名前缀，会被写到顶层，
而真正读它的是 cfg.sft.resume_from，于是覆盖失效、从零开始训练。
"""

from __future__ import annotations

import pytest

from sfzy.config import Config, apply_overrides, deep_merge, load_config


def make_cfg() -> Config:
    return Config({
        "seed": 42,
        "device": "auto",
        "sft": {"num_epochs": 3, "learning_rate": 2e-4, "resume_from": None,
                "output_dir": "outputs/x"},
        "lora": {"r": 16, "alpha": 32},
    })


# ---------------------------------------------------------------- 基本覆盖

def test_覆盖嵌套键():
    cfg = make_cfg()
    apply_overrides(cfg, ["sft.num_epochs=1"])
    assert cfg.path_("sft.num_epochs") == 1


def test_值是yaml解析而不是字符串():
    """1 要变成 int、null 要变成 None、2e-4 要变成 float，
    和直接在 yaml 里写一致。"""
    cfg = make_cfg()
    apply_overrides(cfg, ["sft.num_epochs=1", "sft.resume_from=null",
                          "sft.learning_rate=5e-5", "lora.r=8"])
    assert cfg.path_("sft.num_epochs") == 1
    assert cfg.path_("sft.resume_from") is None
    assert cfg.path_("sft.learning_rate") == pytest.approx(5e-5)
    assert cfg.path_("lora.r") == 8


def test_科学计数法不带小数点也是float():
    """YAML 1.1 的坑：`2e-4` 会被 safe_load 解析成**字符串**，
    只有 `2.0e-4` 才是 float。而前者是命令行里最自然的写法。"""
    import yaml

    assert isinstance(yaml.safe_load("2e-4"), str), "前置条件：YAML 确实把它当字符串"

    cfg = make_cfg()
    apply_overrides(cfg, ["sft.learning_rate=2e-4"])
    assert isinstance(cfg.path_("sft.learning_rate"), float)
    assert cfg.path_("sft.learning_rate") == pytest.approx(2e-4)


def test_真字符串不会被误转成数字():
    cfg = make_cfg()
    apply_overrides(cfg, ["sft.output_dir=outputs/sft-v2"])
    assert cfg.path_("sft.output_dir") == "outputs/sft-v2"


def test_可以覆盖多次():
    cfg = make_cfg()
    apply_overrides(cfg, ["lora.r=8", "lora.alpha=16"])
    assert cfg.get("lora") == {"r": 8, "alpha": 16}


# ---------------------------------------------------------------- 路径校验

def test_单段键不存在时拒绝():
    """这条是关键：`resume_from=...` 不带前缀会被写到顶层，
    而 trainer 读的是 cfg.sft.resume_from —— 覆盖静默失效。"""
    cfg = make_cfg()
    with pytest.raises(KeyError, match="顶层"):
        apply_overrides(cfg, ["resume_from=./x.pt"])


def test_末段键拼错时拒绝():
    cfg = make_cfg()
    with pytest.raises(KeyError, match="sft 下可用的键"):
        apply_overrides(cfg, ["sft.num_epochss=1"])


def test_中间段不存在时拒绝():
    cfg = make_cfg()
    with pytest.raises(KeyError, match="路径不存在"):
        apply_overrides(cfg, ["nope.num_epochs=1"])


def test_报错信息给出正确写法():
    """报错要能直接告诉人怎么写，而不是只说"键不存在"。"""
    cfg = make_cfg()
    with pytest.raises(KeyError, match="sft.resume_from="):
        apply_overrides(cfg, ["resume_from=./x.pt"])


def test_格式不对时拒绝():
    cfg = make_cfg()
    with pytest.raises(ValueError, match="KEY=VALUE"):
        apply_overrides(cfg, ["sft.num_epochs"])


def test_值里可以有等号():
    """路径里不能有 = 但值里可以有。"""
    cfg = make_cfg()
    apply_overrides(cfg, ["sft.output_dir=a=b"])
    assert cfg.path_("sft.output_dir") == "a=b"


# ---------------------------------------------------------------- deep_merge

def test_deep_merge_嵌套字典逐层合并():
    base = {"a": 1, "sft": {"x": 1, "y": 2}}
    over = {"sft": {"y": 99}}
    assert deep_merge(base, over) == {"a": 1, "sft": {"x": 1, "y": 99}}


def test_deep_merge_不修改入参():
    base = {"sft": {"x": 1}}
    deep_merge(base, {"sft": {"x": 2}})
    assert base == {"sft": {"x": 1}}, "deep_merge 不该改动原始字典"


# ---------------------------------------------------------------- 真实配置

def test_真实配置上覆盖resume_from():
    """用真配置跑一遍，确认路径是对的。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/sft_cloud.yaml")
    apply_overrides(cfg, ["sft.resume_from=./outputs/sft_chatglm3/step_000200.pt"])
    assert cfg.path_("sft.resume_from") == "./outputs/sft_chatglm3/step_000200.pt"
