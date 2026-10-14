"""scripts/eval_lora_steps_vllm.py 的验收测试（不需要 vLLM / GPU）。

覆盖四件容易悄悄错掉的事：

  1. **adapter 解析**：step 号解析、通配符展开、同名加前缀、`--adapters` 必填；
  2. **adapter 导出**：`.pt` → PEFT 目录，目标层名/rank 从 checkpoint 推断
     （不信配置），alpha 写进 adapter_config.json（vLLM 的缩放靠它）；
  3. **生成落盘**：jsonl 恰好是 id/source/reference/output 四字段，且断点续跑
     不重复调用模型；
  4. **ROUGE 报告**：只统计两边都有的 id、按 step 排序、总分口径正确。

vLLM 用假模块替身（``generate`` 的返回结构照抄 vLLM），所以这里跑的是真正的
落盘/评测代码路径。

    python -m pytest tests/test_eval_lora_steps.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "eval_lora_steps_vllm", ROOT / "scripts" / "eval_lora_steps_vllm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_script()


# ---------------------------------------------------------------- adapter 解析

def test_step号解析():
    assert EVAL.step_of("step_001511") == 1511
    assert EVAL.step_of("best") is None
    assert EVAL.step_of("e5_full__step_000200") == 200


def test_展开adapter_按步号排序且支持通配符(tmp_path):
    for name in ("step_000200.pt", "step_001500.pt", "best.pt"):
        (tmp_path / name).write_text("x")
    paths = EVAL.expand_adapters([str(tmp_path / "step_*.pt"), str(tmp_path / "best.pt")])
    assert [p.name for p in paths] == ["step_000200.pt", "step_001500.pt", "best.pt"]


def test_展开adapter_没匹配到就报错(tmp_path):
    with pytest.raises(FileNotFoundError, match="没匹配到"):
        EVAL.expand_adapters([str(tmp_path / "step_*.pt")])


def test_同名adapter加父目录前缀(tmp_path):
    used: dict = {}
    assert EVAL.run_name_for(tmp_path / "a" / "step_000001.pt", used) == "step_000001"
    assert EVAL.run_name_for(tmp_path / "b" / "step_000001.pt", used) == "b__step_000001"


def test_适配器与输出目录都是必填():
    with pytest.raises(SystemExit):
        EVAL.parse_args(["--config", "configs/grpo_fact_cov_ee_t4.yaml",
                         "--out-dir", "tmp/x"])
    with pytest.raises(SystemExit):
        EVAL.parse_args(["--config", "configs/grpo_fact_cov_ee_t4.yaml",
                         "--adapters", "a.pt"])


# ---------------------------------------------------------------- adapter 导出

def _save_ckpt(path: Path, *, r: int, modules, torch):
    state = {}
    for module in modules:
        state[f"transformer.encoder.layers.0.attention.{module}.lora_A.weight"] = torch.zeros(r, 8)
        state[f"transformer.encoder.layers.0.attention.{module}.lora_B.weight"] = torch.zeros(16, r)
    torch.save({"model": state, "only_trainable": True}, path)


def test_导出adapter写出peft目录(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    ckpt = tmp_path / "step_000042.pt"
    _save_ckpt(ckpt, r=4, modules=["query_key_value"], torch=torch)
    out_dir = tmp_path / "out"

    runs = EVAL.prepare_adapters([ckpt], out_dir, cfg_alpha=8, cfg_dropout=0.0,
                                 cfg_modules=["query_key_value"],
                                 cfg_rank=4, base_model="fake-base")
    assert runs[0]["name"] == "step_000042" and runs[0]["step"] == 42
    peft_dir = Path(runs[0]["peft_dir"])
    cfg = json.loads((peft_dir / "adapter_config.json").read_text(encoding="utf-8"))
    # alpha/r 决定缩放强度：alpha 来自 --config，r 由权重形状确认
    assert cfg["r"] == 4 and cfg["lora_alpha"] == 8
    assert cfg["target_modules"] == ["query_key_value"]
    assert EVAL.adapter_rank(peft_dir, fallback=999) == 4

    from safetensors.torch import load_file

    tensors = load_file(str(peft_dir / "adapter_model.safetensors"))
    # vLLM 只认带 base_model.model. 前缀的 PEFT 键名
    assert all(k.startswith("base_model.model.") for k in tensors)
    assert len(tensors) == 2


def test_导出adapter按checkpoint推断层名与rank_而不是配置(tmp_path):
    """配置写的是 ChatGLM3 的层名、r=16，checkpoint 是 Qwen 的 q_proj/r=8。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    ckpt = tmp_path / "step_000007.pt"
    _save_ckpt(ckpt, r=8, modules=["q_proj"], torch=torch)
    runs = EVAL.prepare_adapters(
        [ckpt], tmp_path / "out", cfg_alpha=32, cfg_dropout=0.0,
        cfg_modules=["query_key_value", "dense"], cfg_rank=16,
        base_model="fake-base")
    cfg = json.loads((Path(runs[0]["peft_dir"]) / "adapter_config.json").read_text("utf-8"))
    assert cfg["r"] == 8 and cfg["target_modules"] == ["q_proj"]
    assert cfg["lora_alpha"] == 32            # alpha 只能来自配置


def test_同目录adapter_config决定alpha与目标层(tmp_path):
    """你的场景：run 目录里放着 step_*.pt + 一份 adapter_config.json。

    这份 json 是训练时写下来的事实，优先级高于 --config —— config 里的
    r/alpha/target_modules 可能属于另一个 run。
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    ckpt = tmp_path / "step_001511.pt"
    _save_ckpt(ckpt, r=8, modules=["q_proj"], torch=torch)
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "base_model_name_or_path": "/models/Qwen2.5-0.5B-Instruct",
    }), encoding="utf-8")

    runs = EVAL.prepare_adapters(
        [ckpt], tmp_path / "out", cfg_alpha=32, cfg_dropout=0.0,
        cfg_modules=["query_key_value", "dense"], cfg_rank=16,
        base_model="fake-base")
    cfg = json.loads((Path(runs[0]["peft_dir"]) / "adapter_config.json").read_text("utf-8"))
    # alpha / dropout / target_modules 都来自旁边的 json，而不是配置
    assert cfg["lora_alpha"] == 16 and cfg["lora_dropout"] == pytest.approx(0.05)
    assert cfg["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert runs[0]["alpha"] == 16 and runs[0]["rank"] == 8
    assert "adapter_config.json" in runs[0]["meta_source"]


def test_同名json优先于同目录json(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    ckpt = tmp_path / "step_000100.pt"
    _save_ckpt(ckpt, r=8, modules=["q_proj"], torch=torch)
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 16,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}),
        encoding="utf-8")
    (tmp_path / "step_000100.adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 64, "target_modules": ["k_proj"]}),
        encoding="utf-8")
    runs = EVAL.prepare_adapters([ckpt], tmp_path / "out", cfg_alpha=1,
                                 cfg_dropout=0.0, cfg_modules=[], cfg_rank=8,
                                 base_model="b")
    cfg = json.loads((Path(runs[0]["peft_dir"]) / "adapter_config.json").read_text("utf-8"))
    assert cfg["lora_alpha"] == 64 and cfg["target_modules"] == ["k_proj"]


def test_每步子目录布局(tmp_path):
    """GRPO 的 vLLM 热交换布局：<dir>/<step>/adapter_config.json。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    ckpt = tmp_path / "step_000200.pt"
    _save_ckpt(ckpt, r=8, modules=["q_proj"], torch=torch)
    (tmp_path / "step_000200").mkdir()
    (tmp_path / "step_000200" / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 24, "target_modules": ["q_proj"]}),
        encoding="utf-8")
    runs = EVAL.prepare_adapters([ckpt], tmp_path / "out", cfg_alpha=1,
                                 cfg_dropout=0.0, cfg_modules=[], cfg_rank=8,
                                 base_model="b")
    assert runs[0]["alpha"] == 24


def test_checkpoint自带的lora_config也认(tmp_path):
    """PEFT 转回来的 .pt 里带 lora_config（没有旁边的 json 时靠它）。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    ckpt = tmp_path / "adapter_model.pt"
    state = {"transformer.encoder.layers.0.attention.query_key_value.lora_A.weight":
             torch.zeros(16, 8),
             "transformer.encoder.layers.0.attention.query_key_value.lora_B.weight":
             torch.zeros(16, 16)}
    torch.save({"model": state, "only_trainable": True,
                "lora_config": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
                                "target_modules": ["query_key_value"]}}, ckpt)
    runs = EVAL.prepare_adapters([ckpt], tmp_path / "out", cfg_alpha=1,
                                 cfg_dropout=0.0, cfg_modules=["dense"],
                                 cfg_rank=16, base_model="b")
    cfg = json.loads((Path(runs[0]["peft_dir"]) / "adapter_config.json").read_text("utf-8"))
    assert cfg["lora_alpha"] == 32 and cfg["target_modules"] == ["query_key_value"]
    assert "lora_config" in runs[0]["meta_source"]


def test_已经是peft目录就直接用(tmp_path):
    peft = tmp_path / "grpo_run"
    peft.mkdir()
    (peft / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}),
        encoding="utf-8")
    (peft / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    runs = EVAL.prepare_adapters([peft], tmp_path / "out", cfg_alpha=1,
                                 cfg_dropout=0.0, cfg_modules=[], cfg_rank=8,
                                 base_model="b")
    assert runs[0]["peft_dir"] == str(peft)
    assert runs[0]["alpha"] == 16
    assert EVAL.adapter_rank(peft, 1) == 8


def test_目录缺adapter_config就报错(tmp_path):
    (tmp_path / "not_peft").mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        EVAL.prepare_adapters([tmp_path / "not_peft"], tmp_path / "out", cfg_alpha=8,
                              cfg_dropout=0.0, cfg_modules=[], cfg_rank=8,
                              base_model="b")


def test_误传json文件会给出提示(tmp_path):
    wrong = tmp_path / "adapter_config.json"
    wrong.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="adapter_config"):
        EVAL.prepare_adapters([wrong], tmp_path / "out", cfg_alpha=8, cfg_dropout=0.0,
                              cfg_modules=[], cfg_rank=8, base_model="b")


# ---------------------------------------------------------------- 生成落盘

class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 2, 3]


@pytest.fixture()
def fake_vllm(monkeypatch):
    """把 vLLM 换成替身：generate 返回 vLLM 那套 output.outputs[i].text 结构。"""
    calls = {"n": 0, "prompts": 0, "loras": []}

    class _FakeLLM:
        def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=False):
            calls["n"] += 1
            calls["prompts"] += len(prompts)
            calls["loras"].append(getattr(lora_request, "lora_name", None)
                                  if lora_request else None)
            return [SimpleNamespace(outputs=[SimpleNamespace(text=f"摘要{i}")])
                    for i in range(len(prompts))]

    class _SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = _SamplingParams
    vllm.LLM = _FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    return _FakeLLM(), calls


def _ns(**over):
    base = dict(repetition_penalty=1.1, max_new_tokens=8, batch_size=2)
    base.update(over)
    return SimpleNamespace(**base)


def test_生成只写四字段并断点续跑(tmp_path, fake_vllm):
    llm, calls = fake_vllm
    records = [{"id": f"g{i}", "source": f"原文{i}", "summary": f"参考{i}"}
               for i in range(3)]
    out_path = tmp_path / "step_000100.jsonl"

    written = EVAL.generate_run(llm, _FakeTokenizer(), records, _ns(),
                                out_path=out_path, prompt_style="structured",
                                max_prompt_len=32, lora_request=None)
    assert written == 3
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    assert [set(r) for r in rows] == [{"id", "source", "reference", "output"}] * 3
    assert rows[0] == {"id": "g0", "source": "原文0", "reference": "参考0",
                       "output": "摘要0"}
    assert calls["prompts"] == 3

    # 第二遍：全部已完成 → 一条都不再调用模型
    assert EVAL.generate_run(llm, _FakeTokenizer(), records, _ns(),
                             out_path=out_path, prompt_style="structured",
                             max_prompt_len=32, lora_request=None) == 0
    assert calls["prompts"] == 3
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 3


def test_生成用贪心且带重罚(tmp_path, fake_vllm):
    llm, _calls = fake_vllm
    seen = {}

    real_generate = llm.generate

    def spy(prompts, sampling_params, lora_request=None, use_tqdm=False):
        seen["temperature"] = sampling_params.kwargs["temperature"]
        seen["repetition_penalty"] = sampling_params.kwargs["repetition_penalty"]
        return real_generate(prompts, sampling_params, lora_request, use_tqdm)

    llm.generate = spy
    EVAL.generate_run(llm, _FakeTokenizer(),
                      [{"id": "g0", "source": "原文", "summary": "参考"}], _ns(),
                      out_path=tmp_path / "step_000200.jsonl",
                      prompt_style="structured", max_prompt_len=32, lora_request=None)
    assert seen["temperature"] == 0.0
    assert seen["repetition_penalty"] == pytest.approx(1.1)


# ---------------------------------------------------------------- ROUGE 与报告

def _write_run(path: Path, records) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rid, text in records:
            f.write(json.dumps({"id": rid, "source": "原文", "reference": "参考",
                                "output": text}, ensure_ascii=False) + "\n")


def test_rouge只统计共有id(tmp_path):
    refs = {"a": "本院判决被告还款", "b": "驳回原告诉讼请求"}
    run = tmp_path / "step_000010.jsonl"
    _write_run(run, [("a", "本院判决被告还款"), ("c", "只有预测没有参考")])
    rows = EVAL.compute_rouge([run], refs, mode="char")
    assert rows[0]["n_pred"] == 2 and rows[0]["n_ref"] == 2
    assert rows[0]["n_scored"] == 1                  # 只算共有的 a
    assert rows[0]["overall"] == pytest.approx(1.0)  # 与参考逐字相同


def test_rouge按step排序(tmp_path):
    for name in ("step_000200.jsonl", "step_000100.jsonl"):
        _write_run(tmp_path / name, [("a", "x")])
    rows = EVAL.compute_rouge(
        [tmp_path / "step_000200.jsonl", tmp_path / "step_000100.jsonl"],
        {"a": "x"}, mode="char")
    assert [r["step"] for r in rows] == [100, 200]


def test_报告含总表与最优step():
    rows = [
        {"name": "step_000100", "step": 100, "n_scored": 10, "mean_chars": 120,
         "empty_ratio": 0.0, "overall": 0.55, "rouge-1-f": 0.56,
         "rouge-2-f": 0.54, "rouge-l-f": 0.55},
        {"name": "step_000200", "step": 200, "n_scored": 10, "mean_chars": 130,
         "empty_ratio": 0.0, "overall": 0.61, "rouge-1-f": 0.62,
         "rouge-2-f": 0.60, "rouge-l-f": 0.61},
    ]
    report = EVAL.render_report(rows, {"created_at": "now", "rouge_mode": "jieba"})
    assert "| 100 |" in report and "| 200 |" in report
    assert "官方总分最高：**200**" in report
    assert "0.6100" in report


def test_写出报告三件套(tmp_path):
    rows = [{"name": "step_000200", "step": 200, "n_scored": 1, "mean_chars": 5,
             "empty_ratio": 0.0, "overall": 0.5, "rouge-1-f": 0.5,
             "rouge-2-f": 0.5, "rouge-l-f": 0.5}]
    EVAL.write_outputs(tmp_path, rows, {"created_at": "now"})
    assert json.loads((tmp_path / "rouge.json").read_text(encoding="utf-8")) == rows
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["runs"] == rows
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith(
        "# LoRA step 验证集 ROUGE 报告")
