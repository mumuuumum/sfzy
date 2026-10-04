"""config.py 的验收测试。

重点是 apply_overrides 的**路径校验** —— 它防的是一类静默事故：
`--override resume_from=...` 不带段名前缀，会被写到顶层，
而真正读它的是 cfg.sft.resume_from，于是覆盖失效、从零开始训练。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sfzy.config import Config, apply_overrides, deep_merge, load_config

ROOT = Path(__file__).resolve().parents[1]


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


# ---------------------------------------------------------------- 实际生效的配置

def test_继承链从根到叶():
    from sfzy.config import config_chain

    names = [p.name for p in config_chain(ROOT / "configs" / "grpo_fact_only_t4.yaml")]
    assert names == [
        "base.yaml",
        "sft.yaml",
        "sft_cloud.yaml",
        "grpo_fact_judge.yaml",
        "grpo_fact_only_t4.yaml",
    ]


def test_渲染出来的是展开后的值():
    """光看 t4 那份 yaml 看不到 seed / element_weights 这些继承来的键。"""
    from sfzy.config import render_config

    data = yaml.safe_load(render_config(load_config(ROOT / "configs" / "grpo_fact_only_t4.yaml")))

    # 来自 base / sft_cloud
    assert data["seed"] == 42
    assert data["lora"]["target_modules"] == [
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
    ]
    # 来自 grpo_fact_judge.yaml（子配置只覆盖了非继承项）
    assert data["rl"]["learning_rate"] == pytest.approx(1e-5)
    weights = data["rl"]["reward"]["terms"]["fact_consistency"]["element_weights"]
    assert weights["judgment_result"] == pytest.approx(0.30)
    assert data["rl"]["reward"]["terms"]["rouge_l"]["enabled"] is False
    # 本层自己写的
    assert data["rl"]["max_length"] == 1536
    assert data["semantic"]["dtype"] == "float16"


def test_渲染可以附加运行时信息():
    from sfzy.config import render_config

    cfg = load_config(ROOT / "configs" / "grpo_fact_only_t4.yaml")
    data = yaml.safe_load(render_config(cfg, {
        "_config_chain": "base → … → grpo_fact_only_t4",
        "_overrides": ["rl.group_size=2"],
    }))
    assert data["_config_chain"] == "base → … → grpo_fact_only_t4"
    assert data["_overrides"] == ["rl.group_size=2"]


def test_存档实际配置(tmp_path):
    from sfzy.config import save_config

    cfg = load_config(ROOT / "configs" / "grpo_fact_only_t4.yaml")
    path = save_config(cfg, tmp_path / "nested" / "resolved_config.yaml", {"_x": 1})
    assert path.exists()
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["_x"] == 1


def test_渲染结果带得来原始配置路径():
    from sfzy.config import render_config

    data = yaml.safe_load(render_config(load_config(ROOT / "configs" / "grpo_fact_judge.yaml")))
    assert data["_config_path"].endswith("grpo_fact_judge.yaml")
