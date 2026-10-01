"""models/loader.py 的设备放置与量化判定的验收测试（不加载真实模型）。

钉住的是那类**不报错、只是行为不对**的问题：

  1. **4/8-bit 量化模型不能 `.to()`** —— transformers 会抛
     "`.to()` is not supported for 4/8-bit bitsandbytes models"。踩过：
     `model_qwen25_7b.yaml` 开着 4-bit，脚本里 `device_map in (None, ...)`
     的放行判断在 device_map 写 null 时把 `.to()` 放进来了，4090 直接崩，
     而 T4 跑的是没量化的配置所以看不出。
  2. **`--device` 必须对量化模型生效** —— 量化模型的位置只能在加载时由
     `device_map` 决定，所以要把 `--device` 翻译成 `{"": 卡号}` 传进
     `from_pretrained`，否则模型和运行时/输入张量会分居两张卡。

    python -m pytest tests/test_loader.py -q
"""

from __future__ import annotations

import pytest

from sfzy.models import loader as L


# ---------------------------------------------------------------- 假模型

class _Param:
    def __init__(self, device):
        self.device = device


class _FakeModel:
    """够用就好：parameters() 给一个位置，to() 记录被调用过。"""

    def __init__(self, device="cpu", quant=False, eight=False, method=None):
        self._param = _Param(device)
        self.is_loaded_in_4bit = quant
        self.is_loaded_in_8bit = eight
        if method is not None:
            self.quantization_method = method
        self.to_calls: list = []

    def parameters(self):
        return iter([self._param])

    def to(self, device):
        self.to_calls.append(device)
        self._param = _Param(device)
        return self


# ---------------------------------------------------------------- 量化判定

def test_量化判定_4bit():
    assert L.is_quantized_model(_FakeModel(quant=True)) is True


def test_量化判定_8bit():
    assert L.is_quantized_model(_FakeModel(eight=True)) is True


def test_量化判定_普通模型():
    assert L.is_quantized_model(_FakeModel()) is False


def test_量化判定_看quantization_method():
    """不同 transformers 版本暴露的属性不一样，三个来源取或。"""
    assert L.is_quantized_model(_FakeModel(method="QuantizationMethod.BITS_AND_BYTES")) is True


# ---------------------------------------------------------------- 设备串解析

@pytest.mark.parametrize("spec,expected", [
    ("cuda", 0), ("cuda:0", 0), ("cuda:3", 3),
    ("cpu", None), ("auto", None), (None, None), (1, 1),
])
def test_卡号解析(spec, expected):
    assert L.resolve_device_index(spec) == expected


# ---------------------------------------------------------------- 安全搬运

def test_搬运_量化模型不调to():
    """★ 这一条就是 4090 上那个 ValueError 的根因。"""
    model = _FakeModel(quant=True)
    assert L.move_model_to_device(model, "cuda:1") is model
    assert model.to_calls == []


def test_搬运_普通模型正常to():
    model = _FakeModel()
    L.move_model_to_device(model, "cuda:1")
    assert model.to_calls == ["cuda:1"]


# ---------------------------------------------------------------- 推理加载

def _patch_loading(monkeypatch, captured):
    """拦掉真正的模型/tokenizer 加载，只记录传进去的配置。"""
    fake = _FakeModel(device="cuda:1")

    def fake_load_model(model_cfg, quant_config=None, gradient_checkpointing=True, **kw):
        captured["model_cfg"] = dict(model_cfg)
        captured["quant_config"] = quant_config
        return fake

    # 本地环境没装 bitsandbytes，构造 BitsAndBytesConfig 会抛 PackageNotFoundError，
    # 用哨兵代替：测的是"设备怎么放"，不是量化本身。
    monkeypatch.setattr(
        L, "build_quant_config",
        lambda cfg: "QUANT" if cfg.get("load_in_4bit") else None,
    )
    monkeypatch.setattr(L, "load_tokenizer", lambda cfg: "tok")
    monkeypatch.setattr(L, "load_model", fake_load_model)
    return fake


def test_推理加载_量化时把device翻译成device_map(monkeypatch):
    captured = {}
    model = _patch_loading(monkeypatch, captured)
    cfg = {"model_name_or_path": "x", "load_in_4bit": True}

    _, tok, device = L.load_inference_model(cfg, device="cuda:1")

    assert captured["quant_config"] is not None
    assert captured["model_cfg"]["device_map"] == {"": 1}   # --device 权威
    assert model.to_calls == []                              # 量化模型不搬
    assert device == "cuda:1"
    assert tok == "tok"


def test_推理加载_非量化也走device_map(monkeypatch):
    captured = {}
    _patch_loading(monkeypatch, captured)
    cfg = {"model_name_or_path": "x", "load_in_4bit": False}

    L.load_inference_model(cfg, device="cuda:0")

    assert captured["quant_config"] is None
    assert captured["model_cfg"]["device_map"] == {"": 0}


def test_推理加载_force_no_4bit覆盖配置(monkeypatch):
    """4090 上想绕开 bitsandbytes：--no-4bit 要能盖掉配置里的 load_in_4bit。"""
    captured = {}
    _patch_loading(monkeypatch, captured)
    cfg = {"model_name_or_path": "x", "load_in_4bit": True}

    L.load_inference_model(cfg, device="cuda:0", load_in_4bit=False)

    assert captured["quant_config"] is None
    assert captured["model_cfg"]["load_in_4bit"] is False


def test_推理加载_量化模型不允许cpu():
    """4-bit 需要 CUDA，早点报错比在 from_pretrained 里炸清楚。"""
    cfg = {"model_name_or_path": "x", "load_in_4bit": True}
    with pytest.raises(ValueError, match="需要 CUDA"):
        L.load_inference_model(cfg, device="cpu")
