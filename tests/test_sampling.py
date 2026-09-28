"""utils/sampling.py 的验收测试。

核心是最后那个测试：**直接量化 padding 浪费减少了多少** ——
那才是这个采样器存在的理由，不该只靠"应该会快"。
"""

from __future__ import annotations

import random

import pytest

from sfzy.utils.sampling import LengthGroupedBatchSampler, infer_lengths


# ---------------------------------------------------------------- 分桶效果

def test_同一批的长度接近():
    """这是分桶的定义：一批里的长度应该聚在一起。"""
    lengths = [100, 4000, 110, 3900, 120, 3800, 105, 3950]
    sampler = LengthGroupedBatchSampler(lengths, batch_size=2, shuffle=False, jitter=0.0)

    for batch in sampler:
        lo = min(lengths[i] for i in batch)
        hi = max(lengths[i] for i in batch)
        assert hi / lo < 1.5, f"同一批长度差太大: {lo} ~ {hi}"


def test_分桶显著减少padding浪费():
    """用接近真实的长度分布量一下收益。

    真实分布：中位数 2833，P25 2263，P75 3542，P90 4096。
    这里用正态分布近似。
    """
    rng = random.Random(0)
    lengths = [min(4096, max(1000, int(rng.gauss(2900, 700)))) for _ in range(2000)]
    bs = 4

    def waste(batches) -> float:
        real = sum(lengths[i] for b in batches for i in b)
        padded = sum(len(b) * max(lengths[i] for i in b) for b in batches)
        return padded / real - 1

    # 随机组批：当前做法
    random_batches = [list(range(i, i + bs)) for i in range(0, len(lengths), bs)]
    # 分桶组批
    grouped_batches = list(
        LengthGroupedBatchSampler(lengths, bs, shuffle=False, jitter=0.0)
    )

    waste_random = waste(random_batches)
    waste_grouped = waste(grouped_batches)

    assert waste_random > 0.15, f"随机组批的浪费应该很明显，实际 {waste_random:.1%}"
    assert waste_grouped < 0.02, f"分桶后浪费应接近 0，实际 {waste_grouped:.1%}"
    # 这条是收益的下限：真实场景里至少该省这么多
    assert waste_random - waste_grouped > 0.15


# ---------------------------------------------------------------- 正确性

def test_不重不漏():
    """每个样本必须被恰好取到一次，否则会丢训练数据或重复训练。"""
    lengths = [i * 3 for i in range(100)]
    sampler = LengthGroupedBatchSampler(lengths, batch_size=4, shuffle=True, seed=0)
    flat = [i for batch in sampler for i in batch]
    assert sorted(flat) == list(range(100))


def test_最后一个批次可以不满():
    """drop_last=False 的语义：100 / 4 = 25 批，最后一批正好 4 条；
    换成 101 条时最后一批应该是 1 条而不是被丢掉。"""
    lengths = list(range(101))
    sampler = LengthGroupedBatchSampler(lengths, batch_size=4, shuffle=False)
    batches = list(sampler)
    assert len(batches) == 26
    assert len(batches[-1]) == 1


# ---------------------------------------------------------------- 随机性

def test_同seed同epoch可复现():
    lengths = list(range(200))
    a = list(LengthGroupedBatchSampler(lengths, 4, seed=7))
    b = list(LengthGroupedBatchSampler(lengths, 4, seed=7))
    assert a == b


def test_不同epoch批次组成不同():
    """纯排序会让每个 epoch 的批次构成几乎一样 —— 同一批样本反复一起训。
    加抖动就是为了避免这一点。"""
    lengths = [i % 50 * 80 + 1000 for i in range(200)]
    sampler = LengthGroupedBatchSampler(lengths, 4, seed=7)

    sampler.set_epoch(0)
    first = [tuple(sorted(b)) for b in sampler]
    sampler.set_epoch(1)
    second = [tuple(sorted(b)) for b in sampler]
    assert first != second


def test_关掉抖动时是纯排序():
    lengths = [500, 100, 300, 200]
    sampler = LengthGroupedBatchSampler(lengths, 2, shuffle=False, jitter=0.0)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert flat == [1, 3, 2, 0], "应按长度升序聚成两批"


# ---------------------------------------------------------------- DDP 分片

def test_ddp两个rank不重叠且覆盖全部():
    lengths = list(range(40))
    rank0 = [i for b in LengthGroupedBatchSampler(
        lengths, 4, shuffle=False, world_size=2, rank=0) for i in b]
    rank1 = [i for b in LengthGroupedBatchSampler(
        lengths, 4, shuffle=False, world_size=2, rank=1) for i in b]

    assert set(rank0) & set(rank1) == set(), "两个 rank 不该处理同一条数据"
    assert sorted(rank0 + rank1) == list(range(40))


def test_ddp各rank批次数量相近():
    lengths = list(range(40))
    n0 = len(LengthGroupedBatchSampler(lengths, 4, world_size=2, rank=0))
    n1 = len(LengthGroupedBatchSampler(lengths, 4, world_size=2, rank=1))
    assert abs(n0 - n1) <= 1


# ---------------------------------------------------------------- infer_lengths

def test_infer_lengths_从records推算():
    class _DS:
        records = [{"source": "abc", "summary": "de"}, {"source": "a", "summary": None}]

    assert infer_lengths(_DS()) == [5, 1]


def test_infer_lengths_优先用自带的lengths():
    class _DS:
        lengths = [7, 8]
        records = [{"source": "x" * 100, "summary": ""}]

    assert infer_lengths(_DS()) == [7, 8]


def test_infer_lengths_推不出时返回None():
    """推不出来就返回 None，让调用方退回普通采样 —— 宁可慢也不能崩。"""

    class _DS:
        pass

    assert infer_lengths(_DS()) is None


def test_infer_lengths_数据结构不符也不崩():
    class _DS:
        records = [{"wrong_key": 1}]

    assert infer_lengths(_DS()) == [0]


# ---------------------------------------------------------------- 参数校验

def test_batch_size必须为正():
    with pytest.raises(ValueError):
        LengthGroupedBatchSampler([1, 2, 3], batch_size=0)
