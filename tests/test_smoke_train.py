"""sft/trainer.py 的冒烟测试：在 CPU 上跑通完整训练循环。

这是集成测试，串起 dataset → collator → trainer → checkpoint 整条链。
模型用 transformers 的配置随机初始化一个微型 GPT2（不下载任何权重），
tokenizer 用假的，所以整轮跑下来只要几秒。

    python -m pytest tests/test_smoke_train.py -q

注意两点：
  * 必须 fp16=False。CPU 上不支持 fp16，配了会直接报错。
  * 微型模型参数量极小，loss 不一定单调下降，所以只断言"末步低于首步"。
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch

from sfzy.config import load_config
from sfzy.data.collator import SFTCollator
from sfzy.sft.dataset import SFTDataset
from sfzy.sft.trainer import SFTTrainer

ROOT = Path(__file__).resolve().parents[1]

VOCAB = 1024  # 要大于假 tokenizer 可能产出的最大 id


class FakeTokenizer:
    """与 test_collator / test_dataset 里用的同一套约定。"""

    pad_token_id = 0
    eos_token_id = 2
    GENERATION_MARKER = 999

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[10 + (ord(c) % 200) for c in text])

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids = []
        for message in messages:
            ids.extend(self(message["content"]).input_ids)
        if add_generation_prompt:
            ids.append(self.GENERATION_MARKER)
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(i - 10) for i in token_ids if i >= 10)


def tiny_model():
    """随机初始化的微型因果语言模型，不需要下载权重。"""
    from transformers import GPT2Config, GPT2LMHeadModel

    # 必须固定随机初始化。不设种子的话初始 loss 会随机波动，
    # 而 test_loss在下降 断言的是"末步低于首步"——抽到不好的初值就会偶发失败。
    # 这个测试要测的是"梯度有没有真的传到参数上"，不是"这次初始化运气好不好"。
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
    )
    model = GPT2LMHeadModel(config)
    model.config.pad_token_id = FakeTokenizer.pad_token_id
    return model


def make_records(n: int = 4) -> list[dict]:
    return [
        {
            "id": f"doc{i}",
            "source": "文书原文" * (i % 3 + 1),
            "summary": "参考摘要内容。",
        }
        for i in range(n)
    ]


def make_cfg(**overrides):
    """从真实的 configs/sft.yaml 出发，只覆盖少数训练规模相关的键。"""
    cfg = load_config(ROOT / "configs" / "sft.yaml")
    cfg["sft"].update(
        {
            "num_epochs": 1,
            "per_device_batch_size": 2,
            "grad_accum_steps": 2,
            "learning_rate": 1e-2,
            "warmup_ratio": 0.0,
            "dtype": "float32",  # CPU 上用 fp32；fp16 的 autocast 在 CPU 上没意义
            "gradient_checkpointing": False,
            "log_every_n_steps": 1,
            "save_every_n_steps": 3,
            "keep_last_n_checkpoints": 2,
        }
    )
    cfg["sft"].update(overrides)
    return cfg


def make_trainer(tmp_path, **overrides):
    tokenizer = FakeTokenizer()
    collator = SFTCollator(tokenizer, max_length=64, pad_to_multiple_of=8)
    trainer = SFTTrainer(
        model=tiny_model(),
        tokenizer=tokenizer,
        collator=collator,
        cfg=make_cfg(**overrides),
        output_dir=str(tmp_path),
    )
    return trainer


# ---------------------------------------------------------------- 能跑通

def test_训练能跑完并返回状态(tmp_path):
    trainer = make_trainer(tmp_path)
    state = trainer.train(SFTDataset(make_records(4)))
    assert state.step > 0


def test_history非空且记录了loss(tmp_path):
    trainer = make_trainer(tmp_path)
    state = trainer.train(SFTDataset(make_records(4)))
    assert len(state.history) > 0
    assert "loss" in state.history[0]


def test_loss是有限数值(tmp_path):
    trainer = make_trainer(tmp_path)
    state = trainer.train(SFTDataset(make_records(4)))
    losses = [entry["loss"] for entry in state.history]
    # 这条断言不能省：history 为空时下面的 all() 会空过，测试变成永远通过
    assert losses, "history 不该为空"
    assert all(isinstance(v, float) and v == v for v in losses), "不该出现 nan"


def test_loss在下降(tmp_path):
    """在 4 条样本上反复拟合，末步的 loss 应该低于首步。

    这条测的是"梯度有没有真的传到参数上"。如果 loss mask 写错了、
    或者优化器拿到的是空参数列表，loss 会原地不动。

    学习率必须选小：微型模型的初始 loss 在 6.9 左右，lr=5e-2 会直接发散
    （实测 6.892 → 5.638 → 8.818），而 lr=1e-2 是稳定下降的。
    这条测试被 flaky 过一次，根因就是学习率选大了。
    """
    trainer = make_trainer(tmp_path, num_epochs=5, learning_rate=1e-2)
    state = trainer.train(SFTDataset(make_records(4)))

    first = state.history[0]["loss"]
    last = state.history[-1]["loss"]
    assert last < first, f"loss 没有下降：首步 {first:.4f}，末步 {last:.4f}"


def test_训练确实更新了参数(tmp_path):
    """直接查"参数变了没有"，不依赖 loss 的走势。

    比 test_loss在下降 更稳：微型模型的 loss 走势受初始化和 dropout 影响，
    偶尔会先涨后降。而"参数变没变"是确定的 —— 只要梯度传到了参数上、
    loss mask 没把答案段也屏蔽掉，参数就一定会变。

    这条能抓住的典型故障：
      * collator 把答案段的 label 也置成了 -100 → loss 恒为 0 → 没有梯度
      * 优化器拿到空参数列表（忘了 mark_only_lora_trainable 之类）
    """
    trainer = make_trainer(tmp_path, num_epochs=1, grad_accum_steps=1)
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

    trainer.train(SFTDataset(make_records(4)))

    changed = [
        n for n, p in trainer.model.named_parameters() if not torch.equal(before[n], p)
    ]
    assert changed, "训练一轮后没有任何参数被更新，说明梯度没有传下去"


# ---------------------------------------------------------------- 断点续训

def test_保存产生checkpoint文件(tmp_path):
    trainer = make_trainer(tmp_path, save_every_n_steps=2)
    trainer.train(SFTDataset(make_records(4)))

    files = list(Path(tmp_path).glob("*.pt"))
    assert files, "训练过程中应该落下 checkpoint"


def saved_checkpoint(tmp_path) -> str:
    """测试自己挑一个 checkpoint 文件。

    生产代码里没有"自动找最新"这种能力，所以测试也必须显式指定路径。
    文件名是零填充的，字符串序即数字序，取最后一个就是 step 最大的。
    """
    files = sorted(Path(tmp_path).glob("step_*.pt"))
    assert files, "训练过程中应该落下 checkpoint"
    return str(files[-1])


def test_resume_未指定路径时从头开始(tmp_path):
    """配置里 resume_from 为空 → 从头训练，不碰目录里已有的 checkpoint。"""
    trainer = make_trainer(tmp_path, save_every_n_steps=2)
    trainer.train(SFTDataset(make_records(4)))

    fresh = make_trainer(tmp_path)
    assert fresh.resume().step == 0


def test_resume_从指定文件恢复(tmp_path):
    trainer = make_trainer(tmp_path, save_every_n_steps=2)
    state = trainer.train(SFTDataset(make_records(4)))
    assert state.step > 0

    fresh = make_trainer(tmp_path)
    restored = fresh.resume(saved_checkpoint(tmp_path))
    assert restored.step == state.step


def test_resume_文件不存在时报错(tmp_path):
    """写了路径就说明意图是续训，文件不在必须报错，不能静默从头开始。"""
    fresh = make_trainer(tmp_path)
    with pytest.raises(FileNotFoundError):
        fresh.resume(str(tmp_path / "step_999999.pt"))


def test_断点续训从断点接着跑而不是重头来(tmp_path):
    """先跑 1 轮留下断点，再用 3 轮的配置恢复，应当只补跑剩下的 2 轮。

    这里刻意让第二次的 num_epochs 比第一次大 —— 否则"恢复之后没有剩余
    工作可做"，测不出任何东西。真实场景就是这样：Kaggle 上跑到一半被掐，
    重开一个会话接着跑。
    """
    trainer = make_trainer(tmp_path, num_epochs=1, save_every_n_steps=1)
    first = trainer.train(SFTDataset(make_records(4)))
    assert (first.step, first.epoch) == (1, 1)

    resumed = make_trainer(tmp_path, num_epochs=3, save_every_n_steps=1)
    resumed.resume(saved_checkpoint(tmp_path))
    second = resumed.train(SFTDataset(make_records(4)))

    assert second.epoch == 3
    assert second.step == 3, "应当只补跑第 2、3 轮，而不是把第 1 轮重跑一遍"
