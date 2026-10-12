"""LoRA 导出器（自定义格式 → vLLM/PEFT 目录）的验收测试。

这几条都是"错了不会报错、只会静默变差"的地方：
  * 少一个 `base_model.model.` 前缀 → vLLM 找不到层，按零 LoRA 生成；
  * rank 从配置抄而不是从 `lora_A` 形状推 → alpha/r 的 scaling 静默改变强度；
  * 导出空 adapter → vLLM 不报错，只是效果没变。
"""

from __future__ import annotations

import json

import pytest
import torch

import test_kl_reference as h
from sfzy.models.lora import inject_lora, mark_only_lora_trainable, save_lora
from sfzy.models.lora_export import (
    CHATGLM3_TARGET_MODULES,
    convert_checkpoint_to_peft,
    convert_peft_to_checkpoint,
    export_lora_from_model,
    lora_state_to_peft,
    peft_to_lora_state,
)
from sfzy.sft.checkpoint import save_checkpoint


def tiny_lora_model(alpha: int = 8, rank: int = 4):
    model = h.tiny_model()
    assert inject_lora(model, target_modules=["lm_head"], r=rank, alpha=alpha, dropout=0.0) == 1
    mark_only_lora_trainable(model)
    return model


def read_adapter(out_dir):
    from safetensors.torch import load_file

    cfg = json.loads((out_dir / "adapter_config.json").read_text(encoding="utf-8"))
    tensors = load_file(str(out_dir / "adapter_model.safetensors"))
    return cfg, tensors


def test_从活模型导出_键名带peft前缀且rank从形状推断(tmp_path):
    model = tiny_lora_model(alpha=8, rank=4)
    out = export_lora_from_model(
        model, tmp_path / "adapter", alpha=8,
        target_modules=["lm_head"], base_model="THUDM/chatglm3-6b",
    )
    cfg, tensors = read_adapter(out)

    assert cfg["r"] == 4 and cfg["lora_alpha"] == 8        # rank 由 lora_A 形状决定
    assert cfg["target_modules"] == ["lm_head"]
    assert cfg["base_model_name_or_path"] == "THUDM/chatglm3-6b"
    assert set(tensors) == {
        "base_model.model.lm_head.lora_A.weight",
        "base_model.model.lm_head.lora_B.weight",
    }
    assert all(t.dtype == torch.float32 for t in tensors.values())


def test_rank声明错时按实际值导出(tmp_path):
    """alpha/r 的 scaling 错了会静默改变 LoRA 强度 —— 必须让实际形状说了算。"""
    model = tiny_lora_model(alpha=8, rank=4)
    out = export_lora_from_model(
        model, tmp_path / "adapter", alpha=8, target_modules=["lm_head"], rank=16,
    )
    cfg, _ = read_adapter(out)
    assert cfg["r"] == 4, "应该用实际 rank，而不是配置里声称的 16"


def test_剥掉ddp的module前缀(tmp_path):
    state = {
        "module.transformer.encoder.layers.0.self_attention.query_key_value.lora_A.weight":
            torch.zeros(4, 8),
        "module.transformer.encoder.layers.0.self_attention.query_key_value.lora_B.weight":
            torch.zeros(16, 4),
    }
    out = lora_state_to_peft(state, tmp_path / "a", alpha=8,
                             target_modules=["query_key_value"])
    _, tensors = read_adapter(out)
    assert all(k.startswith("base_model.model.") for k in tensors)
    assert not any("module." in k for k in tensors)


def test_空state直接报错而不是导出空adapter(tmp_path):
    with pytest.raises(ValueError, match="没有任何 lora_A/lora_B"):
        lora_state_to_peft({"lm_head.base.weight": torch.zeros(2, 2)},
                           tmp_path / "a", alpha=8)


def test_rank不一致直接报错(tmp_path):
    state = {
        "lm_head.lora_A.weight": torch.zeros(4, 8),
        "other.lora_A.weight": torch.zeros(8, 8),
    }
    with pytest.raises(ValueError, match="rank 不一致"):
        lora_state_to_peft(state, tmp_path / "a", alpha=8, target_modules=["lm_head", "other"])


def test_不在白名单的层会告警但不中断(tmp_path):
    state = {"weird_module.lora_A.weight": torch.zeros(4, 8),
             "weird_module.lora_B.weight": torch.zeros(16, 4)}
    out = lora_state_to_peft(state, tmp_path / "a", alpha=8, target_modules=["lm_head"])
    assert (out / "adapter_config.json").exists()      # 仍然导出，交给 vLLM 去判


def test_两种checkpoint格式都能转(tmp_path):
    model = tiny_lora_model(alpha=8, rank=4)
    # ① save_checkpoint 格式（{"model": {...}}）
    ckpt = save_checkpoint(tmp_path / "step_000001.pt", model=model, only_trainable=True)
    out1 = convert_checkpoint_to_peft(ckpt, tmp_path / "p1", alpha=8,
                                      target_modules=["lm_head"])
    assert read_adapter(out1)[0]["r"] == 4
    # ② save_lora 格式（纯 {名字: 张量}）
    lora_pt = tmp_path / "lora_only.pt"
    save_lora(model, str(lora_pt))
    out2 = convert_checkpoint_to_peft(lora_pt, tmp_path / "p2", alpha=8,
                                      target_modules=["lm_head"])
    assert read_adapter(out2)[1].keys() == read_adapter(out1)[1].keys()


def test_chatglm3白名单常量与配置一致():
    """导出器的默认白名单必须是 ChatGLM3 的那四个叶子线性层。"""
    assert CHATGLM3_TARGET_MODULES == (
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
    )


def test_peft目录转回pt时键名剥掉前缀且与原模型对齐(tmp_path):
    """反向转换必须剥掉 `base_model.model.` 前缀：留着它键名就和模型对不上，
    载入时一个参数都进不去（`load_state_dict(strict=False)` 会静默跳过）。"""
    model = tiny_lora_model(alpha=8, rank=4)
    adapter = export_lora_from_model(
        model, tmp_path / "adapter", alpha=8, target_modules=["lm_head"],
    )

    pt = convert_peft_to_checkpoint(adapter, tmp_path / "lora.pt")
    payload = torch.load(pt, map_location="cpu", weights_only=False)

    assert payload["only_trainable"] is True          # load_checkpoint 认这一层
    assert set(payload["model"]) <= set(model.state_dict()), "键名必须能对上模型的 state_dict"
    for name, value in payload["model"].items():
        assert torch.equal(value, model.state_dict()[name]), f"{name} 的值变了"


def test_转回pt时空adapter直接报错():
    with pytest.raises(ValueError, match="没有任何 lora_A/lora_B"):
        peft_to_lora_state({"lm_head.base.weight": torch.zeros(2, 2)})


def test_转回pt时rank不一致直接报错():
    with pytest.raises(ValueError, match="rank 不一致"):
        peft_to_lora_state({
            "base_model.model.a.lora_A.weight": torch.zeros(4, 8),
            "base_model.model.b.lora_A.weight": torch.zeros(8, 8),
        })


def test_转回pt时两种前缀都剥掉():
    """正常导出的目录不会同时出现两种前缀，但手工拼的权重可能会有。"""
    state = peft_to_lora_state({
        "module.base_model.model.lm_head.lora_A.weight": torch.zeros(4, 8),
        "module.base_model.model.lm_head.lora_B.weight": torch.zeros(16, 4),
    })
    assert set(state) == {"lm_head.lora_A.weight", "lm_head.lora_B.weight"}


def test_导出再转回来是恒等变换(tmp_path):
    """export → convert 的往返必须还原（键名与数值都不许飘）。"""
    model = tiny_lora_model(alpha=8, rank=4)
    original = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}

    adapter = export_lora_from_model(
        model, tmp_path / "adapter", alpha=8, target_modules=["lm_head"],
    )
    payload = torch.load(
        convert_peft_to_checkpoint(adapter, tmp_path / "lora.pt"),
        map_location="cpu", weights_only=False,
    )
    assert set(payload["model"]) == set(original)
    assert all(torch.equal(payload["model"][k], v) for k, v in original.items())


def test_转回pt时把alpha记进payload_再导出不必手传(tmp_path):
    """alpha 无法从权重形状反推，丢了就会静默改强度 —— 必须随 .pt 一起保存。"""
    model = tiny_lora_model(alpha=8, rank=4)
    adapter = export_lora_from_model(
        model, tmp_path / "adapter", alpha=8, target_modules=["lm_head"],
    )
    pt = convert_peft_to_checkpoint(adapter, tmp_path / "lora.pt")
    payload = torch.load(pt, map_location="cpu", weights_only=False)
    assert payload["lora_config"]["lora_alpha"] == 8
    assert payload["lora_config"]["target_modules"] == ["lm_head"]

    # 不传 alpha 再导回 PEFT：应该直接沿用 .pt 里记的那一版
    back = convert_checkpoint_to_peft(pt, tmp_path / "back")
    cfg, tensors = read_adapter(back)
    assert cfg["lora_alpha"] == 8 and cfg["r"] == 4
    assert tensors.keys() == read_adapter(adapter)[1].keys()


def test_没有alpha来源时报错而不是猜一个(tmp_path):
    """裸的 save_lora 格式没有 alpha：宁可不转，也不要默认一个值改掉强度。"""
    model = tiny_lora_model(alpha=8, rank=4)
    bare = tmp_path / "lora_only.pt"
    save_lora(model, str(bare))
    with pytest.raises(ValueError, match="没有记录 lora_alpha"):
        convert_checkpoint_to_peft(bare, tmp_path / "out")
