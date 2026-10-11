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

    names = [p.name for p in config_chain(ROOT / "configs" / "sft_cloud.yaml")]
    assert names == ["base.yaml", "sft.yaml", "sft_cloud.yaml"]


GRPO_CONFIGS = (
    "grpo_fact_judge.yaml",
    "grpo_fact_only_t4.yaml",
    "grpo_fact_only_a100.yaml",
    "grpo_fact_coverage_inp_t4.yaml",
    "grpo_fact_coverage_vGPU48.yaml",
    "grpo_smoke_fact_judge.yaml",
    "grpo_cloud_qwen.yaml",
)


@pytest.mark.parametrize("name", GRPO_CONFIGS)
def test_grpo配置是自包含的(name):
    """GRPO 的运行配置不继承任何东西：一个文件里就是这次 run 的全部参数。

    继承链一长，"我改的到底是哪一层"就只能靠翻文件才能回答。
    """
    from sfzy.config import config_chain

    cfg = load_config(ROOT / "configs" / name)
    assert [p.name for p in config_chain(ROOT / "configs" / name)] == [name]
    assert "model_config" not in cfg, "自包含配置应该把底座直接写进 model: 块"
    for key in ("seed", "device", "model", "lora", "rl", "semantic", "tracking"):
        assert key in cfg, f"{name} 缺少 {key}"


def test_渲染出来的是展开后的值():
    """渲染出来的是这份配置的最终取值（自包含配置里就是文件里写的那份）。"""
    from sfzy.config import render_config

    data = yaml.safe_load(render_config(load_config(ROOT / "configs" / "grpo_fact_only_t4.yaml")))

    assert data["seed"] == 42
    assert data["lora"]["target_modules"] == [
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
    ]
    assert data["rl"]["learning_rate"] == pytest.approx(1e-5)
    # 新的事实一致性只有 enabled / weight（整体分 + 事实错误门控）
    assert data["rl"]["reward"]["terms"]["fact_consistency"]["weight"] == pytest.approx(1.0)
    assert data["rl"]["reward"]["terms"]["rouge_l"]["enabled"] is False
    assert data["rl"]["max_length"] == 1536
    assert data["semantic"]["dtype"] == "float16"
    # 底座是内联的，不再指向另一个 yaml
    assert data["model"]["torch_dtype"] == "float16"


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


# ---------------------------------------------------------------- YAML 科学计数法

def test_yaml里的科学计数法会被补成数字(tmp_path):
    """YAML 1.1 要求浮点尾数带小数点：`3e-6` 会被 PyYAML 解析成**字符串**，
    它一路走到 `torch.optim.AdamW` 才炸（`'<=' not supported between float and str`）。
    config.py 在读文件这一层就补正，所以配置里 `3e-6` 和 `3.0e-6` 等价。"""
    path = tmp_path / "c.yaml"
    path.write_text(
        "rl:\n"
        "  learning_rate: 3e-6\n"          # ← 尾数没小数点：PyYAML 给的是 str
        "  kl_coef: 1e-2\n"
        "  temperature: 0.9\n"             # ← 有小数点：本来就是 float
        "  nested:\n"
        "    xs: [2e-3, 0.5]\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert isinstance(cfg.path_("rl.learning_rate"), float)
    assert cfg.path_("rl.learning_rate") == pytest.approx(3e-6)
    assert cfg.path_("rl.kl_coef") == pytest.approx(1e-2)
    assert cfg.path_("rl.temperature") == pytest.approx(0.9)
    assert cfg.path_("rl.nested.xs") == [pytest.approx(2e-3), pytest.approx(0.5)]


def test_补正不会误伤普通字符串(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "device: auto\n"
        "model:\n"
        "  bnb_4bit_quant_type: nf4\n"
        "  model_name_or_path: /root/autodl-tmp/chatglm3-6b\n"
        "rl:\n"
        "  length_norm: sqrt\n"
        "  prompt_file: data/splits/rl_prompts_full.jsonl\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    for key, expected in (
        ("device", "auto"),
        ("model.bnb_4bit_quant_type", "nf4"),
        ("model.model_name_or_path", "/root/autodl-tmp/chatglm3-6b"),
        ("rl.length_norm", "sqrt"),
        ("rl.prompt_file", "data/splits/rl_prompts_full.jsonl"),
    ):
        assert cfg.path_(key) == expected, key


def test_真实配置的学习率是数字():
    """回归：vGPU48 那份配置曾写 `learning_rate: 3e-6`，在 AdamW 构造时报
    `TypeError: '<=' not supported between instances of 'float' and 'str'`。

    这里只断言"读进来是**数字**且量级合理"，不锁具体数值 —— 那份配置是实验用的，
    学习率会随时被调（例如为了压 clip 比例从 3e-6 降到 1e-6），锁死数值会让测试
    变成噪声。科学计数法本身的补正由上面的 `test_yaml里的科学计数法会被补成数字` 覆盖。
    """
    cfg = load_config(ROOT / "configs" / "grpo_fact_coverage_vGPU48.yaml")
    assert isinstance(cfg.path_("rl.learning_rate"), float)
    assert 1e-8 < cfg.path_("rl.learning_rate") < 1e-2


def test_所有配置读进来后都不残留科学计数法字符串():
    """审计整份 configs/：任何"看着像数字、实际是 str"的值都说明补正漏了。

    这条同时是 config.py 那个补正函数的回归测试 —— 谁把它删了，这里立刻红。
    """
    import re

    pattern = re.compile(r"^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$")
    offenders = []

    def walk(node, trail):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{trail}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{trail}[{i}]")
        elif isinstance(node, str) and pattern.match(node):
            offenders.append(f"{trail} = {node!r}")

    for path in sorted((ROOT / "configs").glob("*.yaml")):
        walk(load_config(path), path.name)
    assert not offenders, "这些值仍是字符串（会被当成非数字用）：\n  " + "\n  ".join(offenders)


def test_训练器对字符串学习率给出可读报错():
    """配置加载已兜住 `3e-6`；这条测的是残留情况（带引号的数字）在 trainer 里
    也会得到一句能看懂的话，而不是 AdamW 里那句 `'<=' not supported ...`。"""
    from sfzy.rl.trainer import _as_learning_rate

    assert _as_learning_rate(3e-6) == pytest.approx(3e-6)
    assert _as_learning_rate(1) == 1.0
    with pytest.raises(ValueError, match="必须是数字"):
        _as_learning_rate("3e-6")
    with pytest.raises(ValueError, match="3.0e-6"):
        _as_learning_rate({"oops": 1})


def test_yaml里的3e_6能一路走到优化器(tmp_path):
    """端到端锁死那次崩溃：yaml 写 `learning_rate: 3e-6` → load_config →
    GRPOTrainer → AdamW。修好之前，这里抛的是
    `TypeError: '<=' not supported between instances of 'float' and 'str'`。"""
    import re

    import test_kl_reference as h

    from sfzy.config import render_config

    dumped = render_config(h.make_cfg())
    # 把 learning_rate 改回"尾数没带小数点"的写法 —— PyYAML 会把它解析成 str
    dumped = re.sub(r"(?m)^(\s*learning_rate:) .*$", r"\1 3e-6", dumped)
    path = tmp_path / "c.yaml"
    path.write_text(dumped, encoding="utf-8")

    cfg = load_config(path)
    assert isinstance(cfg.path_("rl.learning_rate"), float), "补正没生效"

    import torch

    torch.manual_seed(0)
    trainer = h.GRPOTrainer(
        model=h.tiny_model(), tokenizer=h.FakeTokenizer(), cfg=cfg,
        output_dir=str(tmp_path), scorer=h.FactStubScorer(),
    )
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(3e-6)
