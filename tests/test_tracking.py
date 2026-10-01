"""utils/tracking.py 的验收测试（不联网、不装 swanlab 也能跑）。

三件事最容易写错、而且**写错了不报错**：

  1. 指标分组映射漏掉一个键 —— 曲线上少一根，你不会发现
  2. 认不出的键被丢掉 —— 新加的指标静默消失，比名字难看危险得多
  3. swanlab 的列预声明（`define_scalar`）没被调用 —— 面板分组退回默认，
     而日志里一切正常

    python -m pytest tests/test_tracking.py -q
"""

from __future__ import annotations

import sys
import types

import pytest

from sfzy.utils.tracking import (
    GRPO_METRICS,
    NullTracker,
    build_tracker,
    flatten_config,
    group_metrics,
    metric_columns,
)


# ---------------------------------------------------------------- 指标分组

def test_已知指标按分组前缀上报():
    out = group_metrics({"reward_mean": 0.6, "loss": 0.1, "judge_missing": 0.25})
    assert out == {
        "reward/mean": 0.6,
        "train/loss": 0.1,
        "judge/missing_ratio": 0.25,
    }


def test_六要素动态键被改名成中文():
    """`mean_court_facts_score` 这种键是 summarize_scores 动态产出的，
    登记表里没有它，必须靠模式匹配接住 —— 否则六要素的逐项曲线全丢。"""
    out = group_metrics({"mean_court_facts_score": 0.9, "mean_judgment_result_score": 1.0})
    assert out == {"judge/element/法院查明事实": 0.9, "judge/element/裁判结果": 1.0}


def test_档位比例被改名():
    out = group_metrics({f"ratio_score_{lv}": lv / 10 for lv in range(5)})
    assert out["judge/score_ratio/0"] == 0.0
    assert out["judge/score_ratio/4"] == 0.4


def test_认不出的键原样透传而不是丢掉():
    """漏报一个指标比名字难看得多，而且新指标不该因为忘了登记就消失。"""
    out = group_metrics({"reward_mean": 0.5, "某个新指标": 1.0})
    assert out["某个新指标"] == 1.0 and out["reward/mean"] == 0.5


def test_登记表里的每个键都能被指标名命中():
    """反向校验：登记表本身就是"GRPO 要看什么"的清单，
    里面的键必须真的存在于 trainer 产出的指标里（这里用一组代表性输入探一遍格式）。"""
    for key in GRPO_METRICS:
        assert group_metrics({key: 1.0}) == {GRPO_METRICS[key][0]: 1.0}


def test_列定义覆盖分组键和动态键():
    cols = {c[0] for c in metric_columns()}
    assert "reward/group_std" in cols            # 静态登记
    assert "judge/element/裁判结果" in cols       # 六要素动态键
    assert "judge/score_ratio/4" in cols         # 档位比例
    # 每列都要有中文名和图表分组名，否则 swanlab 面板上是英文 key
    for key, name, chart in metric_columns():
        assert name and chart and "/" in key


# ---------------------------------------------------------------- 配置扁平化

def test_配置被压成点号键():
    flat = flatten_config({"rl": {"group_size": 4, "reward": {"mode": "fact_judge"}}})
    assert flat == {"rl.group_size": 4, "rl.reward.mode": "fact_judge"}


def test_配置里的列表拼成逗号串():
    flat = flatten_config({"lora": {"target_modules": ["a", "b"]}})
    assert flat["lora.target_modules"] == "a,b"


def test_配置里的_None_和非映射被跳过():
    flat = flatten_config({"a": None, "b": 1, "c": object()})
    assert flat == {"b": 1}


# ---------------------------------------------------------------- tracker

def test_None_后端返回空实现():
    assert isinstance(build_tracker(backend="none"), NullTracker)
    assert isinstance(build_tracker(backend="swanlab", enabled=False), NullTracker)


def test_未知后端直接报错():
    with pytest.raises(ValueError, match="未知的 tracking 后端"):
        build_tracker(backend="不存在的后端")


def test_没装后端库时给的是可执行的提示(monkeypatch):
    """干瘪的 ModuleNotFoundError 会让人以为是自己代码写错了。"""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "swanlab":
            raise ImportError("No module named 'swanlab'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.delitem(sys.modules, "swanlab", raising=False)
    with pytest.raises(RuntimeError, match="pip install swanlab"):
        build_tracker(backend="swanlab", project="sfzy")


def _fake_swanlab(with_define: bool = True):
    calls = {"define": [], "log": [], "init": []}
    fake = types.ModuleType("swanlab")

    def _init(**kwargs):
        calls["init"].append(kwargs)
        return types.SimpleNamespace(id="run-1")

    fake.init = _init
    fake.log = lambda metrics, step=None: calls["log"].append((metrics, step))
    fake.finish = lambda: calls.setdefault("finish", True)
    if with_define:
        fake.define_scalar = lambda **kw: calls["define"].append(kw)
    return fake, calls


def test_swanlab_列被预声明且超参被记录(monkeypatch):
    """这是"能不能通过 swanlab 监控"的核心：列要先声明、超参要带上。"""
    fake, calls = _fake_swanlab()
    monkeypatch.setitem(sys.modules, "swanlab", fake)

    columns = [("reward/group_std", "组内奖励标准差", "reward")]
    tracker = build_tracker(
        backend="swanlab", project="sfzy", run_name="r1",
        config={"rl.group_size": 4}, columns=columns,
    )
    assert calls["define"] == [
        {"key": "reward/group_std", "name": "组内奖励标准差", "chart_name": "reward"}
    ]
    assert calls["init"][0]["config"] == {"rl.group_size": 4}

    tracker.log({"reward/group_std": 0.08}, step=7)
    assert calls["log"] == [({"reward/group_std": 0.08}, 7)]
    assert tracker.run_id == "run-1"


def test_后端没有_define_scalar_也不该崩(monkeypatch):
    """旧版 swanlab / wandb 没有这个 API。为它把训练搞挂是最蠢的失败模式。"""
    fake, calls = _fake_swanlab(with_define=False)
    monkeypatch.setitem(sys.modules, "swanlab", fake)

    tracker = build_tracker(
        backend="swanlab", project="sfzy",
        columns=[("reward/mean", "奖励均值", "reward")],
    )
    assert calls["define"] == []
    tracker.log({"reward/mean": 0.5}, step=1)     # 不该抛
