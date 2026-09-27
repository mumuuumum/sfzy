"""随机种子固定，保证实验可复现。

注意：Python 的哈希种子需要在进程启动时通过 PYTHONHASHSEED 环境变量设置，
运行时无法修改，因此这里只能设置 random / numpy / torch 三处。
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
