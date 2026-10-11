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
    export_lora_from_model,
    lora_state_to_peft,
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
