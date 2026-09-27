"""DDP 冒烟测试：用 CPU + gloo + torchrun 起两个进程。

本机只有一张卡，跑不了真正的双卡 DDP。但 DDP 的正确性和后端无关，
gloo 在 CPU 上就能验证三件事：

  1. **数据分片** —— 两条 rank 拿到的样本索引不重叠、且合起来覆盖全部
  2. **梯度同步** —— 两条 rank 训练完的参数完全一致
  3. **步数计算** —— steps_per_epoch 除以了 world_size

第 3 条最容易写错：不除的话，学习率调度会按「单卡的数据量」走，
实际只走到 cosine 曲线的一半就结束了，而且不报错。

真正的 worker 在 tests/ddp_worker.py（用 torchrun 启动）。

跑法：
    python -m pytest tests/test_ddp_smoke.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests" / "ddp_worker.py"
WORLD_SIZE = 2


def test_双卡数据分片与梯度同步(tmp_path):
    # 用 `python -m torch.distributed.run` 而不是 torchrun 可执行文件 ——
    # 后者不一定在 PATH 里（本机的 conda 环境就没有），但模块一定随 torch 装好。
    try:
        import torch.distributed.run  # noqa: F401
    except ImportError:
        pytest.skip("环境里没有 torch.distributed.run")

    # 不设 CUDA_VISIBLE_DEVICES="" —— 本机实测那会让 torch native 崩溃。
    # gloo 后端下 init_distributed 不会碰 CUDA，所以直接跑即可。
    env = dict(**os.environ)

    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node", str(WORLD_SIZE), "--standalone", str(WORKER), str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, f"torchrun 失败：\n{proc.stderr[-2000:]}"

    results = [json.loads((tmp_path / f"rank{r}.json").read_text(encoding="utf-8"))
               for r in range(WORLD_SIZE)]
    r0, r1 = results

    # 1) 数据分片
    assert set(r0["seen"]) & set(r1["seen"]) == set(), "两条 rank 拿到重复样本，分片没生效"
    assert set(r0["seen"]) | set(r1["seen"]) == set(range(16)), "样本没被完整覆盖"

    # 2) 梯度同步
    assert r0["param_sum"] == pytest.approx(r1["param_sum"], rel=1e-9), \
        "两条 rank 的参数不一致，梯度没有同步"

    # 3) 步数：16 / (batch2 × accum2 × 2卡) = 2
    assert r0["step"] == r1["step"] == 2, \
        f"步数应为 2（已按 world_size 折算），实际 {r0['step']}"
