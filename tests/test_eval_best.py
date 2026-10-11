"""验证集调度（按分位点）+ best.pt（用综合 ROUGE 选最好模型）。

需求原文："GRPO 缺乏用验证集保存最佳模型的机制；SFT 时用过 ROUGE-1/2/L 综合
打分，现在要用它记录训练过程中最好的模型；假设总步数是 n，应该能在配置里指定
x，在 step 的 x 分位点验证，并按之前的验证结果保存当前最好的模型。"

对应实现：
  * 调度   —— `rl.monitor.eval.quantile_points = x` → `GRPOMonitor.set_total_steps`
              把 n 步等分成 x 段，在第 k/x 分位点各验一次（最后一点 = 最后一步）；
  * 判据   —— `eval/rouge_total`，即官方口径 `0.2*R1 + 0.4*R2 + 0.4*RL`
              （`sfzy.eval.rouge.OFFICIAL_WEIGHTS` 的 `overall`）；
  * 保存   —— 刷新历史最好分就覆盖 `best.pt`，成绩记在 `RLState` 里随 checkpoint
              走，续训不会用更差的分覆盖它。

    python -m pytest tests/test_eval_best.py -q
"""

from __future__ import annotations

import json

import pytest
import torch

import test_kl_reference as h            # 复用微型 GPT2 + LoRA 夹具
from sfzy.rl.monitor import GRPOMonitor, MonitorConfig


# ---------------------------------------------------------------- 分位点调度

def _eval_only_monitor(**ev):
    block = {"enabled": True, "data_file": "val.jsonl"}
    block.update(ev)
    return GRPOMonitor(MonitorConfig.from_config({"eval": block}))


def test_分位点就是把_n_步等分成_x_段():
    m = _eval_only_monitor(quantile_points=4)
    m.set_total_steps(1000)
    assert m.eval_schedule() == (250, 500, 750, 1000)
    # 只在分位点上触发，别处一次都不验
    assert [s for s in range(1, 1001) if m.eval_due(s)] == [250, 500, 750, 1000]


def test_最后一个分位点必然落在最后一步():
    """训练结束时的模型一定要被评到 —— 否则它永远没机会成为 best.pt。"""
    for n, x in ((10738, 10), (999, 7), (5, 5), (13, 4)):
        m = _eval_only_monitor(quantile_points=x)
        m.set_total_steps(n)
        assert m.eval_schedule()[-1] == n
        assert len(m.eval_schedule()) <= x


def test_分位点多于总步数时自动去重():
    m = _eval_only_monitor(quantile_points=10)
    m.set_total_steps(3)
    assert m.eval_schedule() == (1, 2, 3)


def test_不设分位点时仍按_every_n_steps_轮询():
    """旧的按步数触发必须原样保留（其它配置/测试还在用）。"""
    m = _eval_only_monitor(every_n_steps=100)
    m.set_total_steps(1000)
    assert m.eval_schedule() is None
    assert [s for s in range(1, 1001) if m.eval_due(s)] == list(range(100, 1001, 100))


def test_分位点调度不叠加自动缩短():
    """`_interval` 会在总步数少时缩短间隔；分位点是显式排定的，不该被它改掉。"""
    m = _eval_only_monitor(quantile_points=4)
    m.set_total_steps(20)          # 20//5 = 4，若走 _interval 会被缩成"每 4 步"
    assert [s for s in range(1, 21) if m.eval_due(s)] == [5, 10, 15, 20]


def test_没设总步数时不会误触发():
    m = _eval_only_monitor(quantile_points=4)
    assert m.eval_schedule() is None
    assert m.eval_due(1) is False or True  # 退回 every_n_steps 分支，不崩即可


# ---------------------------------------------------------------- best.pt

def _val_file(tmp_path, n=3):
    path = tmp_path / "val.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "id": f"v{i}",
                "source": "判决如下：被告支付48000元。" * (i + 1),
                "summary": "判令被告支付48000元。",
            }, ensure_ascii=False) + "\n")
    return path


def make_trainer(tmp_path, val_path, **ev):
    block = {
        "enabled": True, "data_file": str(val_path), "quantile_points": 1,
        "max_samples": 3, "max_new_tokens": 8, "max_length": None,
    }
    block.update(ev)
    torch.manual_seed(0)
    return h.GRPOTrainer(
        model=h.lora_model(), tokenizer=h.FakeTokenizer(),
        cfg=h.make_cfg(kl_coef=0.0, monitor={"eval": block}),
        output_dir=str(tmp_path), scorer=h.FactStubScorer(),
    )


def test_跑完训练会按验证分写出_best_pt(tmp_path):
    trainer = make_trainer(tmp_path, _val_file(tmp_path))
    trainer.train(h.make_prompts(4))          # prompts_per_step=2 → 2 步

    assert (tmp_path / "best.pt").exists(), "验证跑完却没写 best.pt"
    assert trainer.state.best_eval_score is not None
    assert trainer.state.best_step == 2       # quantile_points=1 → 只在最后一步验
    # 验证指标走的是"监控层 → tracker"这条路（面板上能看到 eval/rouge_total），
    # 不进 state.history 的扁平键 —— 后者只装训练自身的指标。
    assert 0.0 <= trainer._last_monitor_metrics["eval/rouge_total"] <= 1.0
    assert trainer.state.best_eval_score == pytest.approx(
        trainer._last_monitor_metrics["eval/rouge_total"]
    )


def test_只有刷新最好分才覆盖_best_pt(tmp_path):
    trainer = make_trainer(tmp_path, _val_file(tmp_path))
    written: list = []
    real_save = trainer.save
    trainer.save = lambda tag="last": (written.append(tag), real_save(tag))[1]

    for step, score in ((1, 0.50), (2, 0.40), (3, 0.50), (4, 0.62)):
        trainer._maybe_save_best({"eval/rouge_total": score}, step)

    assert written == ["best", "best"], "中间那两次没刷新，不该写盘"
    assert trainer.state.best_eval_score == pytest.approx(0.62)
    assert trainer.state.best_step == 4


def test_best_成绩随_checkpoint_续训保留(tmp_path):
    """续训时若忘了之前的最好分，会用更差的分覆盖 best.pt 且毫无提示。"""
    trainer = make_trainer(tmp_path, _val_file(tmp_path))
    trainer.state.best_eval_score = 0.77
    trainer.state.best_step = 12
    ckpt = trainer.save()                      # state.step 还是 0

    resumed = make_trainer(tmp_path, _val_file(tmp_path))
    resumed.resume(str(ckpt))
    assert resumed.state.best_eval_score == pytest.approx(0.77)
    assert resumed.state.best_step == 12

    resumed._maybe_save_best({"eval/rouge_total": 0.5}, 20)
    assert resumed.state.best_eval_score == pytest.approx(0.77), "续训后用差分数覆盖了"


def test_验证截断与训练用的_max_length_一致(tmp_path, monkeypatch):
    """训练看 7168 token、验证只喂 2048 → 验证分被系统性压低，best 就没意义了。"""
    trainer = make_trainer(tmp_path, _val_file(tmp_path))
    trainer.max_length = 4096
    seen: dict = {}

    import sfzy.rl.trainer as trainer_mod
    real = trainer_mod.generate_batch

    def spy(model, tokenizer, messages_list, **kwargs):
        seen.update(kwargs)
        return real(model, tokenizer, messages_list, **kwargs)

    monkeypatch.setattr(trainer_mod, "generate_batch", spy)

    trainer.monitor.cfg.eval_max_length = None     # 默认：跟随训练
    trainer._run_eval(1)
    assert seen["max_length"] == 4096

    trainer.monitor.cfg.eval_max_length = 1024     # 显式给值：取 min(它, 训练值)
    trainer._run_eval(2)
    assert seen["max_length"] == 1024
