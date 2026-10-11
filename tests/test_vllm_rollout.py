"""vLLM rollout 后端 + trainer 接线的测试（**不需要装 vllm**）。

引擎、LoRARequest、SamplingParams 三者都允许注入：
  * 引擎：假的 `generate()` 记录调用参数、按 n 返回固定条数；
  * LoRARequest：假的工厂，只记录 (name, int_id, path)；
  * SamplingParams：假类，只记录 kwargs。

这样在 CPU 上就能把"每步热交换 + prompt 编码一致 + 条数/顺序 + trainer 接线"
全部钉住；真引擎的差异留给 `tools/check_vllm_rollout.py` 在 GPU 上冒烟。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import test_kl_reference as h
from sfzy.models.chat_template import encode_prompt
from sfzy.data.prompts import build_messages
from sfzy.rl.vllm_rollout import VLLMRollout


# ---------------------------------------------------------------- 假件

class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLoRARequest:
    def __init__(self, lora_name, lora_int_id, lora_path=""):
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path


class _Out:
    def __init__(self, items):
        self.outputs = list(items)


def _item(text: str, *, with_logprobs: bool, seed: int = 0):
    """造一条 CompletionOutput 形状的对象（text / token_ids / logprobs）。"""
    o = type("O", (), {})()
    o.text = text
    if not with_logprobs:
        o.token_ids = None
        o.logprobs = None
        return o
    ids = [10 + seed % 5, 11 + seed % 5, 12 + seed % 5]
    o.token_ids = ids
    # 每个位置是 {token_id: Logprob}，只给被采样到的那个
    o.logprobs = [
        {tid: type("L", (), {"logprob": -(0.5 + 0.1 * k)})()} for k, tid in enumerate(ids)
    ]
    return o


class FakeEngine:
    """记录每次 generate 的入参，并按 n 返回固定文本。"""

    def __init__(self, with_logprobs: bool = False):
        self.calls = []
        self.with_logprobs = with_logprobs

    def generate(self, prompts, sampling_params, lora_request=None):
        self.calls.append({
            "prompts": prompts,
            "sampling": sampling_params.kwargs,
            "lora": lora_request,
        })
        n = int(sampling_params.kwargs["n"])
        return [
            _Out([
                _item(f"prompt{i}-cand{j}", with_logprobs=self.with_logprobs, seed=i * 10 + j)
                for j in range(n)
            ])
            for i in range(len(prompts))
        ]


_UNSET = object()


def make_backend(tmp_path, tokenizer, engine=_UNSET, **over):
    """engine 默认给假引擎；显式传 engine=None 表示"留空、需要时自己建真引擎"。"""
    kwargs = dict(
        base_model="THUDM/chatglm3-6b", tokenizer=tokenizer,
        lora_alpha=32, lora_rank=16, export_root=tmp_path / "adapters",
        engine=FakeEngine() if engine is _UNSET else engine,
        lora_request_factory=FakeLoRARequest,
        sampling_params_factory=FakeSamplingParams,
        target_modules=["lm_head"],
    )
    kwargs.update(over)
    return VLLMRollout(**kwargs)


def build_engine_kwargs(tmp_path, monkeypatch, **over):
    """把假 vllm 模块塞进 sys.modules，捕获 LLM(**kwargs) 的实参。"""
    import sys
    import types

    captured: dict = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = types.ModuleType("vllm")
    fake.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake)
    backend = make_backend(tmp_path, h.FakeTokenizer(), engine=None, **over)
    _ = backend.engine            # 触发 _build_engine
    return captured


# ---------------------------------------------------------------- 引擎参数

def test_引擎参数默认放行远程代码(tmp_path, monkeypatch):
    """回归：漏了 trust_remote_code=True 会在建引擎时直接失败 ——

        ValueError: The repository <本地目录> contains custom code which must be
        executed to correctly load the model. ... pass `trust_remote_code=True`

    vLLM 自带 chatglm 的模型实现，但**读 config 仍要放行**（目录里有
    configuration_chatglm.py）。这条是开工前必须钉死的参数。
    """
    got = build_engine_kwargs(
        tmp_path, monkeypatch,
        gpu_memory_utilization=0.3, max_model_len=8192, max_loras=2, dtype="bfloat16",
    )
    assert got["trust_remote_code"] is True
    assert got["model"].endswith("chatglm3-6b")
    assert got["enable_lora"] is True
    assert got["max_lora_rank"] == 16              # 来自 make_backend 的 lora_rank
    assert got["max_loras"] == 2
    assert got["gpu_memory_utilization"] == 0.3    # 不能是默认 0.9（会挤爆训练）
    assert got["max_model_len"] == 8192
    assert got["dtype"] == "bfloat16"
    assert "quantization" not in got, "没指定量化时不要把这个参数传给 vLLM"


def test_引擎参数里的量化与远程代码开关(tmp_path, monkeypatch):
    got = build_engine_kwargs(tmp_path, monkeypatch, quantization="awq")
    assert got["quantization"] == "awq"
    got2 = build_engine_kwargs(tmp_path, monkeypatch, quantization="bitsandbytes",
                               trust_remote_code=False)
    assert got2["quantization"] == "bitsandbytes"
    assert got2["trust_remote_code"] is False, "显式关掉时不该被覆盖"


# ---------------------------------------------------------------- 同步策略

def test_每步导出adapter并换新的int_id(tmp_path):
    model = h.lora_model()
    backend = make_backend(tmp_path, h.FakeTokenizer())

    lora0 = backend.sync_policy(model, 0)
    assert lora0.lora_int_id == 1 and lora0.lora_name == "policy-0"   # int_id 必须 ≥1
    assert (tmp_path / "adapters" / "step0" / "adapter_config.json").exists()

    lora1 = backend.sync_policy(model, 1)
    assert lora1.lora_int_id == 2
    assert (tmp_path / "adapters" / "step1" / "adapter_config.json").exists()
    # 目录交替复用：磁盘上恒定只有两份
    assert sorted(p.name for p in (tmp_path / "adapters").iterdir()) == ["step0", "step1"]

    lora2 = backend.sync_policy(model, 2)
    assert lora2.lora_int_id == 3, "int_id 每步都要变（vLLM 按它缓存 adapter）"


def test_没同步就生成要报错(tmp_path):
    backend = make_backend(tmp_path, h.FakeTokenizer())
    with pytest.raises(RuntimeError, match="还没来得及|还没给 vLLM 同步策略"):
        backend.generate([[{"role": "user", "content": "x"}]], num_return_sequences=1,
                         max_new_tokens=4, max_length=64)


# ---------------------------------------------------------------- 生成

def test_prompt编码与截断必须和hf路径一致(tmp_path):
    tok = h.FakeTokenizer()
    backend = make_backend(tmp_path, tok)
    backend.sync_policy(h.lora_model(), 0)
    messages = build_messages("判决如下：被告支付48000元。", "structured")

    backend.generate([messages], num_return_sequences=2, max_new_tokens=8, max_length=32)
    got = backend.engine.calls[-1]["prompts"][0]
    expected = encode_prompt(tok, messages, add_generation_prompt=True)[-32:]   # 保留尾部
    # ★ 必须是 dict 形态（PromptType），不能是裸的 token id 列表：
    #   0.11 会对裸列表做 `"prompt_token_ids" in prompt` 然后 `.get()` →
    #   AttributeError: 'list' object has no attribute 'get'（实测踩过）
    assert got == {"prompt_token_ids": expected}


def test_采样参数与条数顺序(tmp_path):
    # collect_logprobs=False：这条只测"采样参数与顺序"，不掺 logprob 路径
    backend = make_backend(tmp_path, h.FakeTokenizer(), collect_logprobs=False)
    backend.sync_policy(h.lora_model(), 0)
    msgs = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]

    texts = backend.generate(msgs, num_return_sequences=3, max_new_tokens=16,
                             max_length=64, temperature=0.9, top_p=0.95)
    call = backend.engine.calls[-1]
    assert call["sampling"] == {
        "n": 3, "temperature": 0.9, "top_p": 0.95, "max_tokens": 16, "logprobs": None,
    }
    assert call["lora"] is not None
    # prompt 优先、组内连续 —— rollout() 按 responses[i*G + j] 取用
    assert texts == ["prompt0-cand0", "prompt0-cand1", "prompt0-cand2",
                     "prompt1-cand0", "prompt1-cand1", "prompt1-cand2"]


def test_默认让vLLM返回采样logprob(tmp_path):
    """默认 collect_logprobs=True：old_logprobs 可以直接用采样那一刻的值，省一次前向。"""
    backend = make_backend(tmp_path, h.FakeTokenizer(),
                           engine=FakeEngine(with_logprobs=True))
    backend.sync_policy(h.lora_model(), 0)
    backend.generate([[{"role": "user", "content": "a"}]], num_return_sequences=2,
                     max_new_tokens=8, max_length=64)
    assert backend.engine.calls[-1]["sampling"]["logprobs"] == 0
    assert backend.last_token_ids is not None and backend.last_logprobs is not None
    assert len(backend.last_token_ids[0]) == len(backend.last_logprobs[0]) == 3


def test_带logprobs时old_logprobs直接用采样值并省掉一次前向(tmp_path, monkeypatch):
    """这是这一步的收益来源：少跑一次 N×L 的前向（实测约 11% 步时）。"""
    import sfzy.rl.trainer as trainer_mod

    calls = {"n": 0}
    real = trainer_mod.sequence_logprob

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "sequence_logprob", counting)

    def run(with_logprobs: bool):
        calls["n"] = 0
        backend = make_backend(tmp_path, h.FakeTokenizer(),
                               engine=FakeEngine(with_logprobs=with_logprobs),
                               collect_logprobs=True)
        torch.manual_seed(0)
        trainer = h.GRPOTrainer(
            model=h.lora_model(), tokenizer=h.FakeTokenizer(),
            cfg=h.make_cfg(kl_coef=0.0, group_size=2, prompts_per_step=2, inner_iterations=2),
            output_dir=str(tmp_path / f"lp{with_logprobs}"),
            scorer=h.FactStubScorer(), rollout_backend=backend,
        )
        batch = trainer.rollout(h.make_prompts(2))
        metrics = trainer.step(h.make_prompts(2))
        return batch, metrics, calls["n"]

    batch_on, _, calls_on = run(True)
    batch_off, _, calls_off = run(False)

    assert batch_on["old_logprobs"] is not None, "有 logprob 时该直接用采样值"
    assert batch_off["old_logprobs"] is None, "没有 logprob 时该回退到本地前向"
    assert calls_on == calls_off - 1, "省掉的正是 old_logprobs 那一次前向"
    # 采样值就是"Σ 每 token logprob"，与 labels 的非 -100 位置一一对应
    assert batch_on["old_logprobs"].shape[0] == 4
    # 夹具给的每条 logprob 都是 [-0.5, -0.6, -0.7] → 和 = -1.8
    assert torch.allclose(batch_on["old_logprobs"], torch.full((4,), -1.8), atol=1e-5)


def test_返回条数不对时直接报错(tmp_path):
    class ShortEngine(FakeEngine):
        def generate(self, prompts, sampling_params, lora_request=None):
            # 少给了：每个 prompt 只回 1 条
            return [_Out([_item("只有一条", with_logprobs=False)]) for _ in prompts]

    backend = make_backend(tmp_path, h.FakeTokenizer(), engine=ShortEngine())
    backend.sync_policy(h.lora_model(), 0)
    with pytest.raises(ValueError, match="数量对不上"):
        backend.generate([[{"role": "user", "content": "a"}]], num_return_sequences=3,
                         max_new_tokens=8, max_length=32)


# ---------------------------------------------------------------- trainer 接线

def test_trainer用vllm后端生成且批次结构与hf一致(tmp_path, monkeypatch):
    """给定**同一批文本**，两条 rollout 路径产出的 input_ids / labels 必须逐位相同。

    这才是要钉的东西：vLLM 只换"谁来写这段文本"，tokenization、prompt 段 mask、
    组内顺序都不该有任何差别。所以这里把 HF 路径的 generate_batch 换成返回同样文本。
    """
    import sfzy.rl.trainer as trainer_mod

    tok = h.FakeTokenizer()
    prompts = h.make_prompts(2)
    fixed = ["prompt0-cand0", "prompt0-cand1", "prompt1-cand0", "prompt1-cand1"]

    def run_with(backend, texts=None):
        torch.manual_seed(0)
        trainer = h.GRPOTrainer(
            model=h.lora_model(), tokenizer=tok,
            cfg=h.make_cfg(kl_coef=0.0, group_size=2, prompts_per_step=2),
            output_dir=str(tmp_path / ("vllm" if backend else "hf")),
            scorer=h.FactStubScorer(), rollout_backend=backend,
        )
        if backend is None:
            monkeypatch.setattr(trainer_mod, "generate_batch", lambda *a, **k: list(texts))
        return trainer.rollout(prompts)

    backend = make_backend(tmp_path, tok)
    got = run_with(backend)
    ref = run_with(None, fixed)

    assert got["input_ids"].shape == ref["input_ids"].shape
    assert torch.equal(got["input_ids"], ref["input_ids"])
    assert torch.equal(got["labels"], ref["labels"])
    assert got["responses"] == ref["responses"]
    assert backend.engine.calls, "vLLM 后端必须真的被调用"
    assert backend.engine.calls[-1]["sampling"]["n"] == 2


def test_一步训练能走vllm后端并收尾(tmp_path):
    backend = make_backend(tmp_path, h.FakeTokenizer())
    torch.manual_seed(0)
    trainer = h.GRPOTrainer(
        model=h.lora_model(), tokenizer=h.FakeTokenizer(),
        cfg=h.make_cfg(kl_coef=0.0, group_size=2, prompts_per_step=2,
                       save_every_n_steps=1000),
        output_dir=str(tmp_path), scorer=h.FactStubScorer(), rollout_backend=backend,
    )
    metrics = trainer.step(h.make_prompts(2))
    assert "loss" in metrics and metrics["loss"] == metrics["loss"]
    trainer.train(h.make_prompts(2))
    assert backend.stats["sync_calls"] >= 2, "每步都要同步策略"
