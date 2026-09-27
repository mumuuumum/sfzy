"""DDP 冒烟测试的 worker。

**不是测试文件**（文件名不以 test_ 开头，pytest 不会收集它），
由 tests/test_ddp_smoke.py 用 torchrun 启动。这样做而不是用
torch.multiprocessing.spawn，有两个原因：

  1. 本机实测 spawn + torch 会触发 native 崩溃（free(): double free）；
  2. **torchrun 才是 Kaggle 上真实的启动方式**，用同一种方式测更有意义。

用法（由测试调用，不用手动跑）：
    CUDA_VISIBLE_DEVICES="" torchrun --nproc_per_node=2 --standalone \\
        tests/ddp_worker.py <输出目录>

注意不要用 CUDA_VISIBLE_DEVICES="" 来强制 CPU —— 本机实测这会让 torch
在 cuda.is_available() 里 native 崩溃。改用 gloo 后端即可：
init_distributed 现在只在 backend == "nccl" 时才 set_device。
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    out_dir = Path(sys.argv[1])
    rank = int(os.environ["RANK"])

    import torch
    from torch.utils.data import Dataset
    from transformers import GPT2Config, GPT2LMHeadModel

    from sfzy.config import load_config
    from sfzy.data.collator import SFTCollator
    from sfzy.sft.trainer import SFTTrainer
    from sfzy.utils.distributed import init_distributed, wrap_model

    local_rank = init_distributed(backend="gloo")

    # ---- 数据：记录这条 rank 实际取到了哪些样本索引 ----
    seen: list[int] = []

    class RecordingDataset(Dataset):
        def __len__(self) -> int:
            return 16

        def __getitem__(self, idx: int):
            seen.append(idx)
            return {
                "id": f"doc{idx}",
                "messages": [{"role": "user", "content": f"文书{idx}"}],
                "answer": "摘要",
            }

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            return types.SimpleNamespace(input_ids=[10 + (ord(c) % 100) for c in text])

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            ids = []
            for message in messages:
                ids.extend(self(message["content"]).input_ids)
            if add_generation_prompt:
                ids.append(99)
            return ids

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=256, n_positions=64, n_embd=32,
                                       n_layer=2, n_head=2, n_inner=64))
    model.config.pad_token_id = 0
    tokenizer = FakeTokenizer()

    cfg = load_config(ROOT / "configs" / "sft.yaml")
    cfg["sft"].update({
        "num_epochs": 1, "per_device_batch_size": 2, "grad_accum_steps": 2,
        "learning_rate": 1e-2, "warmup_ratio": 0.0, "dtype": "float32",
        "gradient_checkpointing": False, "log_every_n_steps": 1,
        "save_every_n_steps": 999, "num_workers": 0, "pin_memory": False,
    })

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        collator=SFTCollator(tokenizer, max_length=64, pad_to_multiple_of=8),
        cfg=cfg, output_dir=str(out_dir / f"rank{rank}"),
    )
    # 用和 train_sft.py 完全相同的封装函数（必须在注入 LoRA 之后调用）
    trainer.model = wrap_model(trainer.model, local_rank)

    state = trainer.train(RecordingDataset())

    # 参数和：两条 rank 应当完全一致 —— 梯度同步成功的证据
    param_sum = float(sum(p.detach().double().sum() for p in trainer.model.parameters()))

    (out_dir / f"rank{rank}.json").write_text(json.dumps({
        "rank": rank, "step": state.step, "seen": sorted(seen), "param_sum": param_sum,
    }), encoding="utf-8")


if __name__ == "__main__":
    main()
