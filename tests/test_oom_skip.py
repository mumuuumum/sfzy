"""显存不足时"跳过该步"的护栏行为。

为什么要有这个功能：一次 40+ 小时的 run 因为某一步长文书 OOM 而全丢，代价远大于
跳一步。为什么要有护栏：如果 OOM 是**必然发生**的（显存预算根本不够），静默跳过
等于把长样本从数据集里悄悄删掉，跑完你还以为全覆盖了 —— 所以超过上限必须报错退出。

    python -m pytest tests/test_oom_skip.py -q
"""

from __future__ import annotations

import pytest
import torch

import test_kl_reference as h
from sfzy.rl.trainer import GRPOTrainer, _is_oom_like


def make_trainer(tmp_path, **rl_over):
    torch.manual_seed(0)
    return GRPOTrainer(
        model=h.lora_model(), tokenizer=h.FakeTokenizer(),
        cfg=h.make_cfg(**rl_over), output_dir=str(tmp_path), scorer=h.FactStubScorer(),
    )


def test_显存不足的步会跳过并继续(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, prompts_per_step=2)     # 4 条 prompt → 2 步
    real_step = trainer.step
    calls = {"n": 0}

    def flaky(chunk):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.64 GiB")
        return real_step(chunk)

    monkeypatch.setattr(trainer, "step", flaky)
    state = trainer.train(h.make_prompts(4))

    assert calls["n"] == 2, "第一步失败后必须继续跑第二步"
    assert state.step == 2, "被跳过的步也要推进步数，否则会一直用同一批数据重试"
    assert state.skipped_steps == 1
    assert len(state.history) == 1, "只有成功的那一步进 history"
    assert state.to_dict()["skipped_steps"] == 1, "跳过数要随 checkpoint 落盘"


def test_非显存类异常不会被吞掉(tmp_path, monkeypatch):
    """本项目自己的 RuntimeError 是 bug（例如裁判信号条数对不上），必须冒出来。"""
    trainer = make_trainer(tmp_path)

    def boom(chunk):
        raise RuntimeError("裁判返回 3 条信号，输入是 8 条，数量对不上。")

    monkeypatch.setattr(trainer, "step", boom)
    with pytest.raises(RuntimeError, match="数量对不上"):
        trainer.train(h.make_prompts(4))


def test_跳过超过上限就报错退出(tmp_path, monkeypatch):
    """OOM 必然发生 = 显存预算不对，这时该停下来改配置，而不是安静地丢数据。"""
    trainer = make_trainer(tmp_path, prompts_per_step=1, max_skipped_steps=2)

    def always_oom(chunk):
        raise torch.OutOfMemoryError("CUDA out of memory.")

    monkeypatch.setattr(trainer, "step", always_oom)
    with pytest.raises(RuntimeError, match="跳过的步数达到上限 2"):
        trainer.train(h.make_prompts(4))
    assert trainer.state.skipped_steps == 2


def test_关掉跳过开关时异常照样抛出(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, skip_on_oom=False)

    def oom(chunk):
        raise torch.OutOfMemoryError("CUDA out of memory.")

    monkeypatch.setattr(trainer, "step", oom)
    with pytest.raises(torch.OutOfMemoryError):
        trainer.train(h.make_prompts(2))


def test_哪些异常算显存类():
    assert _is_oom_like(torch.OutOfMemoryError("CUDA out of memory."))
    # vLLM 引擎子进程被 OOM 干掉时的报错
    assert _is_oom_like(RuntimeError("Engine core initialization failed. See root cause above."))
    assert _is_oom_like(RuntimeError("No available memory for the cache blocks"))
    assert not _is_oom_like(RuntimeError("裁判返回条数不对"))
    assert not _is_oom_like(ValueError("配置里的数值字段类型不对"))
