"""models/compat.py 的验收测试。

这些补丁是给特定模型的远程代码填坑的，逻辑必须锁住。尤其是
`all_tied_weights_keys` **必须是 dict 不是 list** —— 这个类型错误
在第一处调用（`len(...) > 0`）看不出来，只会在 transformers 内部
第二处（`.keys()`）才暴露，而那时已经烧了一轮 Kaggle 时间。
"""

from __future__ import annotations

from transformers import PreTrainedModel

from sfzy.models.compat import (
    patch_pretrained_config,
    patch_tied_weights_keys,
    resolve_padding_side,
)


class FakeConfig:
    """模拟 ChatGLM3 的 config：只有 seq_length，没有 max_length / use_cache。"""

    def __init__(self, **kwargs):
        self.seq_length = 8192
        for key, value in kwargs.items():
            setattr(self, key, value)


# ---------------------------------------------------------------- tied weights

def test_tied_weights_keys_必须是dict不是list():
    """必须是 dict —— accelerate 那边会调 .keys()。

    填 [] 能通过 'len(...) > 0' 那处，但会在
    'getattr(model, "...", {}).keys()' 那处报
    'list' object has no attribute 'keys'。
    """
    patch_tied_weights_keys()
    value = PreTrainedModel.all_tied_weights_keys
    assert isinstance(value, dict), "必须是 dict，不能是 list"
    assert value == {}


def test_tied_weights_keys_重复调用不报错():
    patch_tied_weights_keys()
    patch_tied_weights_keys()
    assert isinstance(PreTrainedModel.all_tied_weights_keys, dict)


# ---------------------------------------------------------------- config 回填

def test_回填_generation_属性():
    config = FakeConfig()
    patch_pretrained_config(config)
    assert config.use_cache is True
    assert config.num_beams == 1
    assert config.temperature == 1.0


def test_max_length_取seq_length而不是generation默认值():
    """generation 的默认 max_length 是 20。

    直接用会让 ChatGLM3 的 self.max_sequence_length 变成 20，
    那模型就只能处理 20 个 token —— 而且不报错。
    """
    config = FakeConfig()
    patch_pretrained_config(config)
    assert config.max_length == 8192, "应该取 seq_length，不是 20"


def test_已有max_length时不覆盖():
    config = FakeConfig(max_length=4096)
    patch_pretrained_config(config)
    assert config.max_length == 4096


def test_已有use_cache时不覆盖():
    config = FakeConfig(use_cache=False)
    patch_pretrained_config(config)
    assert config.use_cache is False


def test_原始config没变_只加字段():
    """回填不能破坏模型自己定义的字段。"""
    config = FakeConfig(hidden_size=4096, num_layers=28)
    patch_pretrained_config(config)
    assert config.hidden_size == 4096
    assert config.num_layers == 28


# ---------------------------------------------------------------- 显式补丁

def test_config_patches_显式补丁生效():
    config = FakeConfig()
    patch_pretrained_config(config, {"config_patches": {"custom_field": 42}})
    assert config.custom_field == 42


def test_config_patches_不覆盖已有字段():
    config = FakeConfig(custom_field=1)
    patch_pretrained_config(config, {"config_patches": {"custom_field": 42}})
    assert config.custom_field == 1


def test_config_patches_缺省不报错():
    patch_pretrained_config(FakeConfig())
    patch_pretrained_config(FakeConfig(), {})


# ---------------------------------------------------------------- padding 方向

def test_padding_side_默认左():
    """decoder-only 的批量生成必须左 padding。"""
    assert resolve_padding_side({}) == "left"


def test_padding_side_可被配置覆盖():
    assert resolve_padding_side({"padding_side": "right"}) == "right"
