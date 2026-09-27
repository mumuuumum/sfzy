"""sft/lr_scheduler.py 的验收测试。

调度曲线写错了不会报错，只会让训练效果变差 —— 属于最难查的一类问题。
    python -m pytest tests/test_lr_scheduler.py -q
"""

from __future__ import annotations

import math

import pytest

from sfzy.sft.lr_scheduler import get_lr

BASE_LR = 2e-4
TOTAL = 1000
WARMUP_RATIO = 0.1          # warmup_steps = 100
MIN_RATIO = 0.1             # min_lr = 2e-5
WARMUP_STEPS = int(TOTAL * WARMUP_RATIO)
MIN_LR = BASE_LR * MIN_RATIO


def lr(step, **kwargs):
    params = dict(
        total_steps=TOTAL,
        base_lr=BASE_LR,
        warmup_ratio=WARMUP_RATIO,
        min_lr_ratio=MIN_RATIO,
    )
    params.update(kwargs)
    return get_lr(step, **params)


# ---------------------------------------------------------------- warmup

def test_warmup阶段严格递增():
    values = [lr(s) for s in range(0, WARMUP_STEPS)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_warmup第一步不为零():
    """用 step/warmup_steps 的写法会在第 0 步给出 lr=0，那一步等于白跑。"""
    assert lr(0) > 0


def test_warmup起点等于一个步长():
    assert lr(0) == pytest.approx(BASE_LR / WARMUP_STEPS, rel=1e-6)


def test_warmup结束时等于峰值():
    assert lr(WARMUP_STEPS - 1) == pytest.approx(BASE_LR, rel=1e-6)
    assert lr(WARMUP_STEPS) == pytest.approx(BASE_LR, rel=1e-6)


def test_warmup边界连续():
    """边界处不能有跳变。"""
    before = lr(WARMUP_STEPS - 1)
    after = lr(WARMUP_STEPS)
    assert after == pytest.approx(before, rel=1e-3)


# ---------------------------------------------------------------- cosine

def test_训练结束时等于下限():
    assert lr(TOTAL) == pytest.approx(MIN_LR, rel=1e-6)


def test_超出总步数仍为下限():
    assert lr(TOTAL + 500) == pytest.approx(MIN_LR, rel=1e-6)


def test_warmup之后单调不增():
    values = [lr(s) for s in range(WARMUP_STEPS, TOTAL + 1)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def test_中点是余弦半程():
    """warmup 后走一半时，lr 应为峰值与下限的中点。"""
    mid = WARMUP_STEPS + (TOTAL - WARMUP_STEPS) // 2
    expected = (BASE_LR + MIN_LR) / 2
    assert lr(mid) == pytest.approx(expected, rel=1e-2)


def test_全程不超过峰值():
    assert max(lr(s) for s in range(0, TOTAL + 1)) <= BASE_LR + 1e-9


# ---------------------------------------------------------------- 边界参数

def test_warmup_ratio为零时立即从峰值开始():
    assert lr(0, warmup_ratio=0.0) == pytest.approx(BASE_LR, rel=1e-6)


def test_min_lr_ratio为零时衰减到零():
    assert lr(TOTAL, min_lr_ratio=0.0) == pytest.approx(0.0, abs=1e-12)


def test_warmup_ratio为一时不崩溃():
    """warmup 占满全程，cosine 的分母会变成 0。"""
    for step in (0, TOTAL // 2, TOTAL):
        value = lr(step, warmup_ratio=1.0)
        assert math.isfinite(value)


def test_返回有限数值():
    for step in range(0, TOTAL + 1, 37):
        assert math.isfinite(lr(step))
