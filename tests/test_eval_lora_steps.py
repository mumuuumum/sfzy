"""scripts/eval_lora_steps_vllm.py 的验收测试（不需要 vLLM / GPU）。

覆盖三件容易悄悄错掉的事：

  1. **adapter 解析与排序**：step 号解析、通配符展开、报告里 baseline 排最前；
  2. **ROUGE 与报告**：只统计两边都有的 id、官方总分口径、最优 step 的判定；
  3. **生成落盘**：jsonl 字段是 id/source/reference/output，且**断点续跑**
     能跳过已完成的 id（跑第二遍不会重复调用模型）。

vLLM 用假模块替身（``generate`` 的返回结构照抄 vLLM），所以这里跑的
是真正的落盘/评测代码路径。

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


def test_发现结果文件时baseline排最前(tmp_path):
    for name in ("step_000200.jsonl", "baseline.jsonl", "step_000100.jsonl"):
        (tmp_path / name).write_text("")
    files = EVAL.discover_run_files(tmp_path)
    assert [p.stem for p in files] == ["baseline", "step_000100", "step_000200"]


# ---------------------------------------------------------------- ROUGE 与报告

def _write_run(path: Path, records, *, key: str = "output") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rid, text in records:
            f.write(json.dumps({"id": rid, "source": "原文", "reference": "参考",
                                key: text}, ensure_ascii=False) + "\n")


def test_rouge只统计共有id(tmp_path):
    refs = {"a": "本院判决被告还款", "b": "驳回原告诉讼请求"}
    run = tmp_path / "step_000010.jsonl"
    _write_run(run, [("a", "本院判决被告还款"), ("c", "只有预测没有参考")])
    rows = EVAL.compute_rouge([run], refs, mode="char")
    assert rows[0]["n_pred"] == 2 and rows[0]["n_ref"] == 2
    assert rows[0]["n_scored"] == 1                 # 只算共有的 a
    assert rows[0]["overall"] == pytest.approx(1.0)  # 与参考逐字相同


def test_rouge拿参考当预测时满分(tmp_path):
    refs = {"a": "本院判决被告还款", "b": "驳回原告诉讼请求"}
    run = tmp_path / "baseline.jsonl"
    _write_run(run, list(refs.items()))
    rows = EVAL.compute_rouge([run], refs, mode="char")
    assert rows[0]["rouge-l-f"] == pytest.approx(1.0)
    assert rows[0]["empty_ratio"] == 0.0


def test_报告含总表最优step与baseline():
    rows = [
        {"name": "baseline", "step": None, "n_scored": 10, "mean_chars": 100,
         "empty_ratio": 0.0, "overall": 0.40, "rouge-1-f": 0.41,
         "rouge-2-f": 0.39, "rouge-l-f": 0.40},
        {"name": "step_000100", "step": 100, "n_scored": 10, "mean_chars": 120,
         "empty_ratio": 0.0, "overall": 0.55, "rouge-1-f": 0.56,
         "rouge-2-f": 0.54, "rouge-l-f": 0.55},
    ]
    report = EVAL.render_report(rows, {"created_at": "now", "rouge_mode": "jieba"})
    assert "| baseline |" in report and "| 100 |" in report
    assert report.index("| baseline |") < report.index("| 100 |")   # baseline 在前
    assert "官方总分最高：**100**" in report
    assert "+0.1500" in report                                       # Δ vs baseline
    # baseline 自己那两列是 -
    baseline_row = next(l for l in report.splitlines() if l.startswith("| baseline |"))
    assert baseline_row.count("| - |") >= 1


def test_写出报告三件套(tmp_path):
    rows = [{"name": "baseline", "step": None, "n_scored": 1, "mean_chars": 5,
             "empty_ratio": 0.0, "overall": 0.5, "rouge-1-f": 0.5,
             "rouge-2-f": 0.5, "rouge-l-f": 0.5}]
    EVAL.write_outputs(tmp_path, rows, {"created_at": "now"})
    assert json.loads((tmp_path / "rouge.json").read_text(encoding="utf-8")) == rows
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["runs"] == rows
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith(
        "# LoRA step 验证集推理 + ROUGE 报告")


# ---------------------------------------------------------------- 生成落盘

class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 2, 3]


@pytest.fixture()
def fake_vllm(monkeypatch):
    """把 vLLM 换成替身：generate 返回 vLLM 那套 output.outputs[i].text 结构。"""
    calls = {"n": 0, "prompts": 0}

    class _FakeLLM:
        def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=False):
            calls["n"] += 1
            calls["prompts"] += len(prompts)
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
    base = dict(num=1, temperature=0.0, top_p=1.0, repetition_penalty=1.1,
                max_new_tokens=8, batch_size=2)
    base.update(over)
    return SimpleNamespace(**base)


def test_生成写出四字段并断点续跑(tmp_path, fake_vllm):
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
    written2 = EVAL.generate_run(llm, _FakeTokenizer(), records, _ns(),
                                 out_path=out_path, prompt_style="structured",
                                 max_prompt_len=32, lora_request=None)
    assert written2 == 0 and calls["prompts"] == 3
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 3


def test_生成num大于1时附带outputs列表(tmp_path, fake_vllm):
    llm, _calls = fake_vllm
    out_path = tmp_path / "step_000200.jsonl"
    EVAL.generate_run(llm, _FakeTokenizer(),
                      [{"id": "g0", "source": "原文", "summary": "参考"}],
                      _ns(num=2, temperature=0.7), out_path=out_path,
                      prompt_style="structured", max_prompt_len=32, lora_request=None)
    row = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert row["output"] == "摘要0" and isinstance(row["outputs"], list)


# ---------------------------------------------------------------- adapter 导出

def test_导出adapter为peft目录并复用(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    ckpt = tmp_path / "step_000042.pt"
    state = {
        "transformer.encoder.layers.0.attention.query_key_value.lora_A.weight":
            torch.zeros(4, 8),
        "transformer.encoder.layers.0.attention.query_key_value.lora_B.weight":
            torch.zeros(16, 4),
    }
    torch.save({"model": state, "only_trainable": True}, ckpt)

    out_dir = tmp_path / "out"
    runs = EVAL.export_adapters([ckpt], out_dir, alpha=8, dropout=0.0,
                                explicit_modules=None, config_modules=["query_key_value"],
                                config_rank=4, base_model="fake-base", force=False)
    peft_dir = Path(runs[0]["peft_dir"])
    assert runs[0]["name"] == "step_000042" and runs[0]["step"] == 42
    cfg = json.loads((peft_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg["r"] == 4 and cfg["lora_alpha"] == 8
    assert "query_key_value" in cfg["target_modules"]
    assert EVAL.adapter_rank(peft_dir, 999) == 4

    from safetensors.torch import load_file

    tensors = load_file(str(peft_dir / "adapter_model.safetensors"))
    # vLLM 只认带 base_model.model. 前缀的 PEFT 键名
    assert all(k.startswith("base_model.model.") for k in tensors)
    assert len(tensors) == 2

    # 已存在 → 默认复用（不报错、结果不变）
    again = EVAL.export_adapters([ckpt], out_dir, alpha=8, dropout=0.0,
                                 explicit_modules=None,
                                 config_modules=["query_key_value"],
                                 config_rank=4, base_model="fake-base", force=False)
    assert again[0]["peft_dir"] == runs[0]["peft_dir"]


def test_导出adapter自动推断目标层_忽略配置里的错误值(tmp_path):
    """配置写的是 ChatGLM3 的层名，checkpoint 是 Qwen 的 → 按 checkpoint 走。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    ckpt = tmp_path / "step_000007.pt"
    torch.save({"model": {
        "model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(8, 16),
        "model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(16, 8),
    }}, ckpt)
    runs = EVAL.export_adapters(
        [ckpt], tmp_path / "out", alpha=16, dropout=0.0, explicit_modules=None,
        config_modules=["query_key_value", "dense"], config_rank=16,
        base_model="fake-base", force=False)
    cfg = json.loads((Path(runs[0]["peft_dir"]) / "adapter_config.json").read_text("utf-8"))
    # rank 从形状反推（8），目标层从键名推断（q_proj）
    assert cfg["r"] == 8 and cfg["target_modules"] == ["q_proj"]


# ---------------------------------------------------------------- CLI 校验

def test_cli_必须给adapter或baseline():
    with pytest.raises(SystemExit):
        EVAL.parse_args(["--config", "configs/grpo_fact_cov_ee_t4.yaml"])


def test_cli_num大于1必须采样():
    with pytest.raises(SystemExit):
        EVAL.parse_args(["--baseline", "--num", "4", "--temperature", "0"])


def test_cli_num大于1默认温度():
    args = EVAL.parse_args(["--baseline", "--num", "4"])
    assert args.temperature == pytest.approx(0.7)
