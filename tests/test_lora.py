"""models/lora.py 的验收测试。

LoRA 有三个"写错了也不会报错"的地方，测试全部锁住：
  1. B 必须零初始化 —— 否则训练一开始就破坏了预训练权重
  2. base 必须冻结 —— 否则没省下参数量
  3. scaling 必须是 alpha/r —— 否则调 r 时学习率跟着变，消融就不可控

纯 CPU，秒级：
    python -m pytest tests/test_lora.py -q
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sfzy.models.lora import (
    LoRALinear,
    count_parameters,
    inject_lora,
    load_lora,
    mark_only_lora_trainable,
    merge_lora,
    save_lora,
)

IN, OUT, R, ALPHA = 8, 4, 2, 4


class Tiny(nn.Module):
    """结构模仿 ChatGLM3 的注意力层命名（query_key_value 是融合 QKV）。"""

    def __init__(self) -> None:
        super().__init__()
        self.query_key_value = nn.Linear(IN, IN)
        self.dense = nn.Linear(IN, IN)
        self.head = nn.Linear(IN, OUT)

    def forward(self, x):
        return self.head(self.dense(self.query_key_value(x)))


def make_lora(r=R, alpha=ALPHA) -> LoRALinear:
    return LoRALinear(nn.Linear(IN, OUT), r=r, alpha=alpha)


def fill_lora_b(model: nn.Module, std: float = 0.1) -> None:
    """把 LoRA 的 B 矩阵填成非零，让旁路真正产生贡献。"""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.weight.normal_(mean=0.0, std=std)


# ---------------------------------------------------------------- LoRALinear 本身

def test_初始输出等于原始线性层():
    """B 零初始化 => ΔW = 0 => 输出与原始层逐位一致。

    如果这条不过，说明 B 不是零初始化的，训练起点就已经偏离预训练模型。
    """
    base = nn.Linear(IN, OUT)
    lora = LoRALinear(base, r=R, alpha=ALPHA)
    x = torch.randn(3, IN)

    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_base参数被冻结():
    lora = make_lora()
    assert lora.base.weight.requires_grad is False


def test_lora参数可训练():
    lora = make_lora()
    # lora_A / lora_B 是 nn.Linear 模块，requires_grad 在它们的 .weight 上
    assert lora.lora_A.weight.requires_grad is True
    assert lora.lora_B.weight.requires_grad is True


def test_A和B的形状():
    lora = make_lora()
    assert lora.lora_A.weight.shape == (R, IN)
    assert lora.lora_B.weight.shape == (OUT, R)


def test_scaling等于alpha除以r():
    assert LoRALinear(nn.Linear(IN, OUT), r=4, alpha=8).scaling == pytest.approx(2.0)
    assert LoRALinear(nn.Linear(IN, OUT), r=8, alpha=16).scaling == pytest.approx(2.0)
    assert LoRALinear(nn.Linear(IN, OUT), r=4, alpha=4).scaling == pytest.approx(1.0)


def test_可训练参数量只有r倍的in加out():
    lora = make_lora()
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    assert trainable == R * IN + OUT * R


def test_旁路生效后输出改变():
    lora = make_lora()
    x = torch.randn(3, IN)
    before = lora(x).detach()
    fill_lora_b(lora)
    assert not torch.allclose(before, lora(x).detach(), atol=1e-4)


def test_merged_weight形状与原权重一致():
    lora = make_lora()
    assert lora.merged_weight().shape == (OUT, IN)


def test_merged_weight等于W加scaling乘BA():
    lora = make_lora()
    fill_lora_b(lora)
    expected = lora.base.weight + lora.scaling * (lora.lora_B.weight @ lora.lora_A.weight)
    assert torch.allclose(lora.merged_weight(), expected, atol=1e-6)


def test_支持三维输入():
    """transformer 里送进线性层的通常是 (B, L, hidden)。"""
    lora = make_lora()
    x = torch.randn(2, 5, IN)
    assert lora(x).shape == (2, 5, OUT)


def test_无autocast时_fp16底座也能前向():
    """LoRA 参数固定 fp32，而底座可能是 fp16 —— 两者精度不同。

    不显式把输入转到 LoRA 的 dtype，`F.linear` 会直接报
    "expected m1 and m2 to have the same dtype"。

    训练时有 autocast 兜着看不出问题，**但推理、评估、以及任何没包
    autocast 的前向都会崩**。这个测试就是为了在没有 autocast 的
    情况下跑一次 forward。
    """
    base = nn.Linear(IN, OUT).half()
    lora = LoRALinear(base, r=R, alpha=ALPHA)
    assert lora.lora_A.weight.dtype == torch.float32, "前置条件：LoRA 参数是 fp32"

    x = torch.randn(2, IN, dtype=torch.float16)
    out = lora(x)                        # 不应抛 dtype 错误
    assert out.dtype == torch.float16, "输出精度应当跟随底座，不能被旁路抬高"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU 才能验证设备一致性")
def test_lora分支跟随base的设备():
    """新建的 lora_A / lora_B 默认落在 CPU 上。

    底座如果已经被搬到 GPU，而这里忘了指定 device，forward 会直接报
    "Expected all tensors to be on the same device"。
    这个 bug 在纯 CPU 的测试里永远暴露不出来，所以单独用 GPU 验一次。
    """
    base = nn.Linear(IN, OUT).to("cuda")
    lora = LoRALinear(base, r=R, alpha=ALPHA)

    assert lora.lora_A.weight.device.type == "cuda"
    assert lora.lora_B.weight.device.type == "cuda"
    # 参数必须是 fp32：fp16 主权重配 GradScaler 会直接报
    # "Attempting to unscale FP16 gradients"
    assert lora.lora_A.weight.dtype == torch.float32
    assert lora.lora_B.weight.dtype == torch.float32

    out = lora(torch.randn(2, IN, device="cuda"))
    assert out.device.type == "cuda"


# ---------------------------------------------------------------- inject_lora

def test_inject_lora_返回替换层数():
    model = Tiny()
    replaced = inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    assert replaced == 2


def test_inject_lora_确实替换了目标层():
    model = Tiny()
    inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    assert isinstance(model.query_key_value, LoRALinear)
    assert isinstance(model.dense, LoRALinear)


def test_inject_lora_不碰非目标层():
    model = Tiny()
    inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    assert type(model.head) is nn.Linear


def test_inject_lora_用endswith匹配完整层名():
    """完整层名是 "layers.0.query_key_value" 这种，必须按后缀匹配。"""
    model = nn.Sequential(nn.Linear(IN, IN), nn.Linear(IN, IN))
    replaced = inject_lora(model, ["0"], r=R, alpha=ALPHA)
    assert replaced == 1
    assert isinstance(model[0], LoRALinear)
    assert type(model[1]) is nn.Linear


def test_inject_lora_嵌套模块():
    """真实模型的层名是 "blocks.0.query_key_value" 这种带层级的完整路径。"""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_key_value = nn.Linear(IN, IN)
            self.dense = nn.Linear(IN, IN)

    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(3)])

    model = Nested()
    replaced = inject_lora(model, ["query_key_value"], r=R, alpha=ALPHA)
    assert replaced == 3
    assert all(isinstance(b.query_key_value, LoRALinear) for b in model.blocks)
    assert all(type(b.dense) is nn.Linear for b in model.blocks)


def test_inject_lora_无匹配时返回0():
    model = Tiny()
    assert inject_lora(model, ["不存在的层名"], r=R, alpha=ALPHA) == 0


def test_inject_lora_后前向仍可跑():
    model = Tiny()
    inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    assert model(torch.randn(2, IN)).shape == (2, OUT)


# ---------------------------------------------------------------- 可训练参数控制

def test_mark_only_lora_trainable():
    model = Tiny()
    inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    mark_only_lora_trainable(model)

    for name, param in model.named_parameters():
        expected = "lora_A" in name or "lora_B" in name
        assert param.requires_grad is expected, f"{name} 的 requires_grad 应为 {expected}"


def test_count_parameters_返回总数与可训练数():
    model = Tiny()
    inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    mark_only_lora_trainable(model)

    total, trainable = count_parameters(model)
    assert total > trainable > 0
    # 两层 LoRA，每层 (r*in + out*r)
    assert trainable == 2 * (R * IN + IN * R)


def test_lora占比远小于1():
    """这是"LoRA 为什么省显存"最直接的证据，写进报告里。"""
    model = Tiny()
    inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    mark_only_lora_trainable(model)
    total, trainable = count_parameters(model)
    assert trainable / total < 0.5


# ---------------------------------------------------------------- merge

def test_merge_lora_返回合并层数():
    model = Tiny()
    inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    assert merge_lora(model) == 2


def test_merge后不再有LoRALinear():
    model = Tiny()
    inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())


def test_merge前后输出一致():
    """合并是等价变换，输出必须逐位接近，否则说明 scaling 或矩阵乘法写错了。"""
    model = Tiny()
    inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    fill_lora_b(model)
    # 合并的等价性只在 dropout 关闭时成立：训练模式下旁路会先过 dropout，
    # 而 merged_weight() 是不带 dropout 的确定性权重。
    model.eval()

    x = torch.randn(2, IN)
    before = model(x).detach()
    merge_lora(model)
    after = model(x).detach()

    assert torch.allclose(before, after, atol=1e-5)


def test_merge保持原dtype():
    """新建的 nn.Linear 必须跟原层同精度。

    忘了指定 dtype 的话，fp16 的模型会在合并后被悄悄变成 fp32，
    显存直接翻倍——在 16GB 的 T4 上这就是 OOM 与不 OOM 的区别。
    """
    model = Tiny().half()
    inject_lora(model, ["query_key_value"], r=R, alpha=ALPHA)
    merge_lora(model)

    assert isinstance(model.query_key_value, nn.Linear)
    assert model.query_key_value.weight.dtype == torch.float16


# ---------------------------------------------------------------- 存取

def test_save_lora_只存lora参数(tmp_path):
    model = Tiny()
    inject_lora(model, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    path = tmp_path / "adapter.pt"
    save_lora(model, str(path))

    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    assert state, "保存的内容不该为空"
    assert all("lora_" in key for key in state), "不该存 base 权重"


def test_load_lora_能载回参数(tmp_path):
    src = Tiny()
    inject_lora(src, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    fill_lora_b(src)
    path = tmp_path / "adapter.pt"
    save_lora(src, str(path))

    dst = Tiny()
    inject_lora(dst, ["query_key_value", "dense"], r=R, alpha=ALPHA)
    load_lora(dst, str(path))

    assert torch.allclose(
        src.query_key_value.lora_A.weight, dst.query_key_value.lora_A.weight, atol=1e-6
    )
    assert torch.allclose(
        src.dense.lora_B.weight, dst.dense.lora_B.weight, atol=1e-6
    )
