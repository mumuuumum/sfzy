"""手写 LoRA：低秩适配的注入与合并。

============================ 你要实现的文件 ============================

-------------------------------- 原理（一句话版） --------------------------------
冻结原权重 W，在旁路挂两个小矩阵 A、B，前向变成

    h = Wx + scaling * B(Ax),      scaling = alpha / r

其中 A 的形状是 (r, in_features)，B 是 (out_features, r)。
可训练参数量从 in*out 降到 r*(in+out)，r 通常取 8~64。

-------------------------------- 三个必须做对的地方 --------------------------------
1. **B 零初始化**。A 随机、B 全零，这样训练开始时 ΔW = B(Ax) 恒等于 0，
   模型行为与原始预训练权重完全一致，再逐步学出增量。
   如果两个都随机初始化，训练一开始就破坏了预训练模型的能力。

2. **原权重必须冻结**。base.requires_grad = False，否则就没省下参数量。

3. **scaling = alpha / r 不能省**。调 r 的时候保持 alpha/r 不变，
   学习率就不用重调，这是实践中让 r 的消融可控的关键。
   面试常问"为什么 alpha 常取 2r"，答案就在这个比例上。

-------------------------------- 动手前先确认 --------------------------------
ChatGLM3-6B 的注意力层名和 LLaMA 系不一样（它用融合的 query_key_value），
所以 target_modules 要按模型实际结构填，不要照抄 ["q_proj","v_proj"]。
用 `for name, _ in model.named_modules(): print(name)` 看一眼真实层名。
=====================================================================
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from pathlib import Path


class LoRALinear(nn.Module):
    """包装一个冻结的 nn.Linear，旁挂低秩分支 A、B。

    属性：
        base   —— 冻结的原线性层
        lora_A —— 形状 (r, in_features)
        lora_B —— 形状 (out_features, r)
        scaling —— alpha / r

    提示：dropout 只作用在旁路输入上（对 x 做 dropout），不要动主路。
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        # TODO: 保存 base、构造 lora_A / lora_B / dropout / scaling
        # 初始化要求见模块开头的第 1、2 条
        self.base = base
        base.requires_grad_(False)

        # 新构造的 nn.Linear 默认在 CPU / fp32 上，设备必须显式对齐到 base，
        # 否则底座搬到 GPU 之后 forward 会直接报
        # "Expected all tensors to be on the same device"。
        #
        # dtype 固定 fp32，**不跟随 base**。这不是保守，是必须：
        # fp16 的"主权重"配 GradScaler 会直接报
        # "Attempting to unscale FP16 gradients" —— GradScaler 的设计前提
        # 就是参数本身 fp32、只有中间计算走 fp16。
        # PEFT 的默认行为也是这个：底模可以是 fp16 / 4-bit，
        # 但 LoRA 的 A/B 始终是 fp32。
        lora_device = base.weight.device
        self.lora_A = nn.Linear(
            base.in_features, r, bias=False, device=lora_device, dtype=torch.float32,
        )
        nn.init.kaiming_normal_(self.lora_A.weight, a=5**0.5)
        self.lora_B = nn.Linear(
            r, base.out_features, bias=False, device=lora_device, dtype=torch.float32,
        )
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / r
           

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """h = base(x) + scaling * B(A(dropout(x)))。

        注意 shape：lora_A 是 (r, in)，所以 x @ A.T 得到 (..., r)，
        再 @ B.T 得到 (..., out)。
        可以把 x 展平再算以兼容任意前置维度（3D 输入在 transformer 里很常见）。
        """
        result = self.base(x)

        # 注意这里的两次 dtype 转换，缺一不可：
        #
        # 1. LoRA 参数固定是 fp32（见 __init__），而输入 x 是底座的精度
        #    （fp16/bf16）。不把 x 转成 LoRA 的 dtype，F.linear 会直接报
        #    "expected m1 and m2 to have the same dtype"。
        #    训练时有 autocast 兜着看不出问题，但**推理、评估、以及任何
        #    没包 autocast 的前向都会崩** —— 这类 bug 只在训练之外暴露。
        #    PEFT 的做法也是这一句 `x.to(lora_A.weight.dtype)`。
        # 2. 结果要转回 result 的 dtype，否则 fp16 底座 + fp32 旁路相加
        #    会向上提升成 fp32，污染后续层。
        lora_dtype = self.lora_A.weight.dtype
        lora_out = self.lora_B(self.lora_A(self.dropout(x).to(lora_dtype)))
        return result + (self.scaling * lora_out).to(result.dtype)

    def merged_weight(self) -> torch.Tensor:
        """返回合并后的等效权重 W + scaling * B @ A。

        用途：推理时把旁路合并进主路，前向不再有额外计算开销。
        形状必须是 (out_features, in_features)，与原权重一致。
        """
        return self.base.weight + self.scaling * (self.lora_B.weight @ self.lora_A.weight)


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> int:
    """把名字匹配 target_modules 的 nn.Linear 替换成 LoRALinear。

    实现思路（推荐，不容易出错）：
      1. 遍历 model.named_modules()，收集所有**叶子**线性层的完整名字
         （用 `isinstance(m, nn.Linear) and not any(isinstance(c, nn.Linear)
         for c in m.children())` 或者看名字后缀）。
      2. 只保留名字以 target_modules 中某一项结尾的层。
         用 `name.endswith(target)` 而不是 `==`，因为完整名字是
         "transformer.layers.0.attention.query_key_value" 这种。
      3. 用 setattr 递归替换。注意名字里带点号，要按点拆分逐级 setattr。

    返回替换的层数（有测试会检查这个数字是否等于匹配到的层数）。

    提示：替换时把父模块和子名字一起存下来，最后统一 setattr，
    边遍历边修改 dict 会出错。
    """
    linear_to_replace = [] # store (parent,child_name,old_linear)
    for name,module in model.named_modules():
        if (
            isinstance(module,nn.Linear) 
            and not any(isinstance(child,nn.Linear) for child in module.children())
            and name.split(".")[-1] in target_modules
        ):
            parent_name, _ , child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            linear_to_replace.append((parent, child_name, module))
    for parent,child_name,old_linear in linear_to_replace:
        # 替换的都是叶子节点，替换顺序不会影响其它节点
        new_linear = LoRALinear(old_linear,r,alpha,dropout)
        setattr(parent,child_name,new_linear)
    return len(linear_to_replace)

def mark_only_lora_trainable(model: nn.Module) -> None:
    """把所有非 LoRA 参数设为 requires_grad=False。

    判断依据：参数名里含 "lora_A" 或 "lora_B" 的才是可训练的。
    """
    for name, param in model.named_parameters():
        param.requires_grad = ("lora_A" in name or "lora_B" in name)

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。

    注入 LoRA 后调用它，可训练占比通常远低于 1%——
    这个数字是"为什么 LoRA 省显存"最直接的证据，写进实验报告里。
    """
    # requires_grad 才是 PyTorch 决定“这个参数是否参与训练”的唯一依据，名字匹配只是一个近似。
    # 假设有人做了以下操作之一：
    # 解冻了 base 的某些层（比如只解冻 lm_head 或最后几层做全量微调）。
    # 给 LoRA 层加了 bias（虽然标准实现没有，但自定义实现可能有）。
    # 解冻了整个 base 做全参微调，此时根本没有 lora_ 参数。
    # 在这些情况下，名字匹配会漏掉真正可训练的参数，或者误报不可训练的参数。而 param.requires_grad 永远反映真实状态。
    # total_num = 0
    # trainable_num = 0
    # for name, param in model.named_parameters():
    #     total_num += param.numel()
    #     if "lora_A" in name or "lora_B" in name:
    #         trainable_num += param.numel()
    # return (total_num, trainable_num)
    total_num = 0
    trainable_num = 0
    for _, param in model.named_parameters():
        total_num += param.numel()
        if param.requires_grad:
            trainable_num += param.numel()
    return (total_num, trainable_num)

def merge_lora(model: nn.Module) -> int:
    """把所有 LoRALinear 合并回普通 nn.Linear，返回合并的层数。

    合并后推理不再走旁路，速度与原始模型一致。
    注意：合并是不可逆的，只在推理前做。
    """
    linear_to_merge = []
    for name,module in model.named_modules():
        if isinstance(module,LoRALinear):
            parent_name,_,child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            linear_to_merge.append((parent,child_name,module))
    for parent,child_name,module in linear_to_merge:
        new_linear = nn.Linear(
            in_features=module.base.in_features,
            out_features=module.base.out_features,
            bias=module.base.bias is not None,   # 看原层有没有 bias
            dtype=module.base.weight.dtype, 
            device=module.base.weight.device
        )
        new_linear.weight.data.copy_(module.merged_weight()) # 原层的weight
        if module.base.bias is not None: # 原层的bias
            new_linear.bias.data.copy_(module.base.bias.data)
        setattr(parent,child_name,new_linear)
    return len(linear_to_merge)
            


def save_lora(model: nn.Module, path: str) -> None:
    """只保存 LoRA 参数（名字里含 lora_ 的那些），不保存整个 6B 模型。

    这是 LoRA 的实用价值之一：adapter 通常只有几十 MB，
    而完整模型是十几 GB。保存前先 state_dict() 再过滤。
    """
    # lora_dict = [{k:v} for k,v in model.state_dict() if "lora_" in k]
    lora_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_dict, path)


def load_lora(model: nn.Module, path: str) -> None:
    """把保存的 LoRA 参数载回模型。用 load_state_dict(..., strict=False)。"""
    lora_state = torch.load(path, map_location="cpu")
    # model.load_state_dict(state_dict) 会严格匹配：
    # state_dict 里的每个键都必须在模型里存在，模型里的每个参数也必须在 state_dict 里出现。否则报错。
    # 
    model.load_state_dict(lora_state, strict=False)
    # load_lora 通常的调用顺序是：
    # 加载基础模型
    # inject_lora(model, ...) 注入 LoRA 层
        # model.layers.0.self_attn.q_proj.base.weight    ← 原始权重，冻结
        # model.layers.0.self_attn.q_proj.base.bias      ← 原始 bias，冻结
        # model.layers.0.self_attn.q_proj.lora_A.weight  ← LoRA 参数
        # model.layers.0.self_attn.q_proj.lora_B.weight  ← LoRA 参数
    # load_lora(model, path) 把训练好的 LoRA 权重载进去（）
        # model.layers.0.self_attn.q_proj.lora_A.weight
        # model.layers.0.self_attn.q_proj.lora_B.weight
        # ... (只有 lora_ 的键)
