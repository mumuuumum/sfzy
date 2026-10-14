"""对若干个 LoRA step 在验证集上推理，并算 ROUGE 出报告（vLLM 加速）。

产物只有两样（都在 ``--out-dir`` 下）：

    <out-dir>/<step>.jsonl   每行 {"id", "source", "reference", "output"}
                             —— 验证集的输出文件：id / 裁判文书原文 / 人工摘要 / 候选摘要
    <out-dir>/report.md      各 step 的 ROUGE 对比报告
    <out-dir>/rouge.json     同一份指标的机器可读版本
    <out-dir>/lora_peft/     导出给 vLLM 用的 PEFT adapter（中间产物，可复用）

一条命令做三件事
================
  1. **导出**：``step_*.pt``（本项目格式：``...query_key_value.lora_A.weight``）
     → PEFT 目录（``adapter_config.json`` + ``adapter_model.safetensors``），
     实现在 ``sfzy.models.lora_export``，和 GRPO 训练时的热交换共用同一份代码；
  2. **推理**：vLLM 连续批处理；**底座只加载一次**，多个 adapter 依次热挂载；
  3. **评测**：``sfzy.eval.rouge.score_corpus``（CAIL2020 官方口径，逐样本宏平均）。

关于 adapter_config.json（你不需要在命令行传它）
=============================================
vLLM 加载 LoRA 的单位是**目录**，不是文件：

    LoRARequest(name, lora_int_id, lora_path="<导出的目录>")

vLLM 自己会去读该目录里的 ``adapter_config.json``，从里面取

  * ``r`` 与 ``lora_alpha`` —— 决定缩放系数 ``scaling = alpha / r``，也就是
    **LoRA 的强度**。填错不会报错，只会让结果悄悄偏掉；
  * ``target_modules`` —— 决定把 LoRA 挂到哪些层上；
  * ``peft_type`` / ``task_type`` —— 格式校验。

所以本脚本导出时会把这些字段一并写好（见 ``sfzy/models/lora_export.py``）。
训练存档 ``step_*.pt`` 里**只有权重、没有 alpha**（``save_checkpoint`` 不存
``lora_config``），alpha 无法从权重反推 —— 它取自 ``--config`` 的 ``lora.alpha``，
rank 则从 ``lora_A`` 的形状反推并与配置交叉校验（对不上会告警）。

如果某个 adapter 已经是 PEFT 目录（例如 GRPO 训练时热交换留下的
``outputs/<run>/adapter_config.json``），``--adapters`` 直接给那个目录即可。

``--adapters`` 给 ``.pt`` 时，脚本会**自己去找**它对应的 ``adapter_config.json``
（按优先级）：

  1. ``<同目录>/<名字>.adapter_config.json`` 或 ``<同目录>/<名字>.json``（每步一份）；
  2. ``<同目录>/adapter_config.json``（一个 run 一份 —— r / alpha /
     target_modules 在整个训练里是常量，各 step 共用是对的）；
  3. ``<同目录>/<名字>/adapter_config.json``（每步一个子目录，GRPO 热交换的布局）；
  4. 都没有就看 ``.pt`` 里有没有 ``lora_config``（PEFT 转回来的存档会带）。

取到就用它：``alpha`` / ``dropout`` / ``target_modules`` / ``r`` 都以这份元信息
为准，**``--config`` 只兜底**。理由是这些值是训练时写下来的事实，而 ``--config``
可能属于另一次 run（例如拿 ChatGLM3 的配置去评 Qwen 的 checkpoint —— r 和层名
都对不上）。每一条都会在日志里写明取值来源。

与训练口径对齐
=============
  * prompt 用 ``sfzy.data.prompts.build_messages(source, style)``，style 取
    配置里的 ``data.prompt_style``；
  * 编码用 ``chat_template.encode_prompt(add_generation_prompt=True)``；
  * prompt 超预算时**保留尾部**（判决书的"本院认为/判决如下"在结尾）；
  * 贪心解码 + ``repetition_penalty``，与 SFT/GRPO 生成默认值一致。

用法
====
    # 几个 step 横向比
    python scripts/eval_lora_steps_vllm.py \
        --config configs/grpo_fact_cov_ee_t4.yaml \
        --adapters outputs/exp/e5_full/step_001500.pt outputs/exp/e5_full/step_001511.pt \
        --out-dir outputs/eval_steps/e5_full

    # 通配符（按 step 号排序）
    python scripts/eval_lora_steps_vllm.py --config configs/grpo_fact_cov_ee_t4.yaml \
        --adapters 'outputs/exp/e5_full/step_*.pt' --out-dir outputs/eval_steps/e5_full

    # 小样本冒烟
    python scripts/eval_lora_steps_vllm.py --config configs/grpo_fact_cov_ee_t4.yaml \
        --adapters outputs/exp/e5_full/step_001511.pt \
        --out-dir tmp/eval_smoke --limit 4 --max-new-tokens 32 --enforce-eager

    # T4：6B 必须 4-bit
    python scripts/eval_lora_steps_vllm.py --config ... --adapters ... --out-dir ... \
        --dtype float16 --quantization bitsandbytes

重跑同一条命令会**断点续跑**：已写进 jsonl 的 id 不再重复生成，只补缺的。

环境
====
vLLM 与 transformers 版本强绑定（见 ``requirements-cloud.txt``）。本脚本刻意只依赖
纯 Python 的 prompt 拼接 + tokenizer + vLLM，不 import 项目的模型加载代码，
所以能在干净的 vLLM 环境里跑。
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfzy.config import load_config                        # noqa: E402
from sfzy.data.prompts import build_messages               # noqa: E402
from sfzy.eval.rouge import score_corpus                   # noqa: E402
from sfzy.models.chat_template import encode_prompt        # noqa: E402
from sfzy.utils.io import ensure_dir, read_jsonl           # noqa: E402
from sfzy.utils.logging import get_logger                  # noqa: E402

logger = get_logger("eval_lora_steps")

_STEP_RE = re.compile(r"step_(\d+)")


def resolve(path: Any) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def is_local_dir(path: Path) -> bool:
    """权限不足（配置里写的是别的机器的路径）时按"不是本地路径"处理。"""
    try:
        return path.is_dir()
    except OSError:
        return False


def load_done_ids(path: Path) -> set:
    """已完成（已写进结果 jsonl）的 id，用于断点续跑。"""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ---------------------------------------------------------------------------
# 输入数据 / adapter
# ---------------------------------------------------------------------------
def resolve_split_file(data_cfg: Any, split: str) -> Path:
    """找 split 对应的 jsonl（和 generate_triples*.py 同一套候选顺序）。"""
    processed = resolve(data_cfg.path_("data.processed_dir", "data/processed"))
    candidates = [processed / f"{split}.jsonl"]
    if split == "val":
        candidates.append(processed / data_cfg.path_("data.dev_file", "dev.jsonl"))
    if split == "train":
        candidates.append(processed / data_cfg.path_("data.train_file", "train.jsonl"))
    candidates.append(resolve("data/splits") / f"{split}.jsonl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"找不到 split={split} 的样本文件，试过：\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def resolve_model_cfg(cfg: Any) -> Dict[str, Any]:
    """取底座模型配置，兼容两种布局。

    项目里有两套写法：``model: {model_name_or_path: ...}`` 直接内联（GRPO 的
    自包含配置），以及 ``model_config: configs/model_xxx.yaml`` 指向另一份文件。
    两种都要认，内联的优先（子配置覆盖父配置的语义）。
    """
    from sfzy.config import deep_merge

    merged: Dict[str, Any] = {}
    ref = cfg.get("model_config")
    if ref:
        merged = dict(load_config(resolve(ref)).get("model") or {})
    return deep_merge(merged, dict(cfg.path_("model") or {}))


def resolve_data_cfg(cfg: Any) -> Any:
    """取数据配置；没有 ``data_config`` 时退回配置自身（键都在顶层）。"""
    ref = cfg.get("data_config")
    return load_config(resolve(ref)) if ref else cfg


def step_of(name: str) -> Optional[int]:
    """从名字里取 step 号（``step_001511`` → 1511）；取不到返回 None。"""
    match = _STEP_RE.search(name)
    return int(match.group(1)) if match else None


def expand_adapters(patterns: Sequence[str]) -> List[Path]:
    """展开通配符并去重；按 (step 号, 名字) 排序，没有 step 号的排最后。"""
    seen: Dict[str, Path] = {}
    for pattern in patterns:
        has_glob = any(ch in pattern for ch in "*?[")
        matches = [Path(p) for p in glob.glob(pattern)] if has_glob else [Path(pattern)]
        if not matches:
            raise FileNotFoundError(f"--adapters 里这一项没匹配到任何文件：{pattern}")
        for path in matches:
            seen[str(path)] = path
    paths = [resolve(p) for p in seen.values()]
    paths.sort(key=lambda p: (step_of(p.stem) is None, step_of(p.stem) or 0, p.name))
    return paths


def run_name_for(adapter: Path, used: Dict[str, int]) -> str:
    """adapter → 输出文件名（不含扩展名）。同名的加父目录前缀区分。"""
    name = adapter.stem if not adapter.is_dir() else adapter.name
    if name in used:
        name = f"{adapter.parent.name}__{name}"
    used[name] = used.get(name, 0) + 1
    return name


def find_adapter_config(adapter: Path) -> Optional[Path]:
    """找 ``.pt`` 对应的 ``adapter_config.json``（按优先级）。

    支持的三种摆放方式（都是实际出现过的）：

      1. ``<名字>.adapter_config.json`` / ``<名字>.json`` —— 与 .pt 同名；
      2. ``<同目录>/adapter_config.json`` —— 一个 run 一份（r / alpha /
         target_modules 在整个训练里是常量，所以各 step 共用一份是对的）；
      3. ``<同目录>/<名字>/adapter_config.json`` —— 每步一个子目录
         （GRPO 的 vLLM 热交换就是这种布局）。
    """
    parent, stem = adapter.parent, adapter.stem
    for candidate in (
        parent / f"{stem}.adapter_config.json",
        parent / f"{stem}.json",
        parent / "adapter_config.json",
        parent / stem / "adapter_config.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def read_lora_config(path: Optional[Path]) -> Dict[str, Any]:
    """读 adapter_config.json（读不了就返回空，让调用方回退）。"""
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("adapter_config.json 读不了（%s）：%s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def probe_checkpoint(adapter: Path) -> Dict[str, Any]:
    """从 ``.pt`` 读出 ``{"modules", "rank", "lora_config", "sibling_config"}``。

    目标层名**从 checkpoint 自己推断**，而不是信配置：配置里的
    ``lora.target_modules`` 可能属于另一个底座（ChatGLM3 是
    query_key_value/dense*，Qwen 是 q_proj/k_proj/v_proj/o_proj），
    用错目标层 vLLM 要么加载失败、要么静默忽略（"挂了 adapter 但效果没变"）。
    rank 同理，从 ``lora_A`` 的形状 (r, in_features) 反推。

    另外把能拿到的**训练元信息**一并捞出来：``.pt`` 里可能带 ``lora_config``
    （PEFT 转回来的存档会有），旁边可能放着 ``adapter_config.json``（alpha 只
    能从这两处得到 —— 它无法从权重形状反推）。
    """
    import torch

    from sfzy.models.lora_export import extract_lora_state

    payload = torch.load(adapter, map_location="cpu", weights_only=False)
    state = extract_lora_state(payload)
    modules: List[str] = []
    ranks: set = set()
    for key in state:
        name = str(key)
        if "lora_A" not in name and "lora_B" not in name:
            continue
        parts = name.split(".")
        if len(parts) >= 3 and parts[-3] not in modules:
            modules.append(parts[-3])
        if name.endswith("lora_A.weight"):
            try:
                ranks.add(int(state[key].shape[0]))
            except (AttributeError, IndexError, ValueError):
                pass
    lora_config = payload.get("lora_config") if isinstance(payload, dict) else None
    return {
        "modules": modules,
        "rank": (ranks.pop() if len(ranks) == 1 else None),
        "lora_config": dict(lora_config) if isinstance(lora_config, dict) else {},
        "sibling_config": find_adapter_config(adapter),
    }


def resolve_settings(probe: Dict[str, Any], json_cfg: Dict[str, Any],
                     *, cfg_alpha: int, cfg_dropout: float,
                     cfg_modules: Sequence[str], cfg_rank: int
                     ) -> Dict[str, Any]:
    """定 LoRA 的 alpha / dropout / target_modules / rank，并记下每一项的来源。

    优先级：**同目录的 adapter_config.json > .pt 里的 lora_config > checkpoint
    的键名/形状 > --config**。前两者是训练时写下来的事实，配置可能属于另一个 run。
    """
    payload_cfg = probe.get("lora_config") or {}
    sources: Dict[str, str] = {}

    def pick(field: str, *options):
        for label, value in options:
            if value not in (None, "", [], {}):
                sources[field] = label
                return value
        return None

    alpha = pick("alpha",
                 ("adapter_config.json", json_cfg.get("lora_alpha")),
                 ("checkpoint.lora_config", payload_cfg.get("lora_alpha")),
                 ("--config lora.alpha", cfg_alpha))
    dropout = pick("dropout",
                   ("adapter_config.json", json_cfg.get("lora_dropout")),
                   ("checkpoint.lora_config", payload_cfg.get("lora_dropout")),
                   ("--config lora.dropout", cfg_dropout))
    modules = pick("target_modules",
                   ("adapter_config.json", json_cfg.get("target_modules")),
                   ("checkpoint.lora_config", payload_cfg.get("target_modules")),
                   ("checkpoint 键名", probe.get("modules")),
                   ("--config lora.target_modules", list(cfg_modules)))
    declared_r = pick("r",
                      ("adapter_config.json", json_cfg.get("r")),
                      ("checkpoint.lora_config", payload_cfg.get("r")))
    return {
        "alpha": int(alpha) if alpha is not None else int(cfg_alpha),
        "dropout": float(dropout) if dropout is not None else float(cfg_dropout),
        "target_modules": list(modules or cfg_modules or []),
        "actual_rank": probe.get("rank"),
        "declared_rank": int(declared_r) if declared_r is not None else None,
        "sources": sources,
    }


def prepare_adapters(adapters: Sequence[Path], out_dir: Path, *,
                     cfg_alpha: int, cfg_dropout: float,
                     cfg_modules: Sequence[str], cfg_rank: int,
                     base_model: str) -> List[Dict[str, Any]]:
    """把每个输入整理成 vLLM 能直接挂的 PEFT 目录。

    ``--adapters`` 给 ``.pt`` 就导出到 ``<out-dir>/lora_peft/<name>/``；
    给的本来就是 PEFT 目录（含 ``adapter_config.json``）就直接用。

    ``.pt`` 的 alpha / target_modules 优先取它自带的元信息（旁边的
    ``adapter_config.json`` 或 payload 里的 ``lora_config``），取不到才回退到
    ``--config`` —— 同一次训练里 r / alpha / target_modules 是常量，这几处
    必然一致；不一致时以 checkpoint 自带的为准并告警。
    """
    from sfzy.models.lora_export import convert_checkpoint_to_peft

    runs: List[Dict[str, Any]] = []
    used: Dict[str, int] = {}
    for adapter in adapters:
        if not adapter.exists():
            raise FileNotFoundError(f"adapter 不存在：{adapter}")
        if adapter.is_file() and adapter.suffix in (".json",):
            raise SystemExit(
                f"--adapters 收到的是配置文件而不是权重：{adapter}\n"
                "  请给 .pt 存档，或给含 adapter_config.json 的 PEFT 目录")
        name = run_name_for(adapter, used)

        if adapter.is_dir():
            if not (adapter / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"{adapter} 是目录，但不是 PEFT adapter（缺 adapter_config.json）")
            peft_dir = adapter
            cfg = read_lora_config(peft_dir / "adapter_config.json")
            logger.info("直接用已导出的 PEFT adapter：%s（r=%s, alpha=%s）",
                        peft_dir, cfg.get("r"), cfg.get("lora_alpha"))
            runs.append({"name": name, "adapter": str(adapter), "step": step_of(name),
                         "peft_dir": str(peft_dir), "alpha": cfg.get("lora_alpha"),
                         "rank": cfg.get("r"),
                         "target_modules": cfg.get("target_modules"),
                         "meta_source": "已导出的 PEFT 目录"})
            continue

        probe = probe_checkpoint(adapter)
        json_cfg = read_lora_config(probe.get("sibling_config"))
        settings = resolve_settings(probe, json_cfg, cfg_alpha=cfg_alpha,
                                    cfg_dropout=cfg_dropout,
                                    cfg_modules=cfg_modules, cfg_rank=cfg_rank)
        modules = settings["target_modules"]
        rank = settings["actual_rank"]
        declared = settings["declared_rank"]
        src = settings["sources"]

        if not modules:
            raise SystemExit(f"{adapter} 里推不出 LoRA 目标层，也没有 adapter_config.json "
                             f"或配置可依据 —— 这份文件可能不是 LoRA 存档")
        if probe.get("sibling_config"):
            logger.info("%s：元信息取自 %s（alpha=%s, r=%s, target_modules=%s）",
                        adapter.name, probe["sibling_config"], settings["alpha"],
                        declared, modules)
        else:
            logger.warning(
                "%s 旁边没有 adapter_config.json，也没带 lora_config —— alpha 只能"
                "取自 --config（%s）。若这不是该 checkpoint 用的那份配置，LoRA 强度"
                "会静默偏掉；建议把训练时的 adapter_config.json 放到它旁边",
                adapter.name, src.get("alpha"))
        if rank is not None and declared is not None and rank != declared:
            logger.warning("%s：adapter_config.json 声称 r=%d，权重形状是 r=%d —— "
                           "按实际形状导出", adapter.name, declared, rank)
        if (cfg_modules and src.get("target_modules") == "checkpoint 键名"
                and not (set(modules) & set(cfg_modules))):
            logger.warning(
                "%s 的 LoRA 目标层 %s 与 --config 里的 %s 完全不同 —— 按 checkpoint "
                "导出。请确认底座模型与这份 checkpoint 匹配",
                adapter.name, modules, list(cfg_modules))
        peft_dir = out_dir / "lora_peft" / name
        convert_checkpoint_to_peft(
            adapter, peft_dir, alpha=settings["alpha"], dropout=settings["dropout"],
            target_modules=modules, base_model=base_model,
        )
        runs.append({"name": name, "adapter": str(adapter), "step": step_of(name),
                     "peft_dir": str(peft_dir), "alpha": settings["alpha"],
                     "rank": rank if rank is not None else declared,
                     "target_modules": modules,
                     "meta_source": str(probe.get("sibling_config") or "checkpoint 键名"
                                        + ("/payload lora_config" if probe.get("lora_config")
                                           else "/--config"))})
    return runs


def adapter_rank(peft_dir: Path, fallback: int) -> int:
    """读 adapter_config.json 里的 r —— 这才是 vLLM 实际用的那个值。"""
    try:
        cfg = json.loads((Path(peft_dir) / "adapter_config.json").read_text(encoding="utf-8"))
        return int(cfg.get("r", fallback))
    except (OSError, ValueError, KeyError):
        return fallback


# ---------------------------------------------------------------------------
# prompt 编码（与训练/其它生成脚本同一口径）
# ---------------------------------------------------------------------------
def encode_prompts(records: Sequence[Dict[str, Any]], tokenizer: Any,
                   prompt_style: str, max_prompt_len: int) -> Iterator[List[int]]:
    """逐条编码 prompt，**保留尾部**截断（判决书的结论在结尾）。"""
    for record in records:
        messages = build_messages(record["source"], prompt_style)
        ids = encode_prompt(tokenizer, messages, add_generation_prompt=True)
        if max_prompt_len and len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
        yield ids


def make_lora_request(peft_dir: Path, lora_id: int) -> Any:
    """构造 ``LoRARequest`` —— 传的是**目录**，vLLM 自己去读里面的 adapter_config.json。

    兼容 0.5.x 的 ``lora_local_path`` 与 0.6+ 的 ``lora_path``。
    """
    from vllm.lora.request import LoRARequest

    try:
        return LoRARequest(Path(peft_dir).name, lora_id, lora_path=str(peft_dir))
    except TypeError:
        return LoRARequest(Path(peft_dir).name, lora_id, lora_local_path=str(peft_dir))


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
def build_llm(args: argparse.Namespace, model_name: str, model_cfg: Dict[str, Any],
              max_lora_rank: int, n_loras: int, max_prompt_len: int) -> Any:
    from vllm import LLM

    max_model_len = int(args.max_model_len or (max_prompt_len + args.max_new_tokens))
    llm_kwargs: Dict[str, Any] = dict(
        model=model_name,
        tokenizer=args.tokenizer or model_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code")),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        swap_space=args.swap_space,
        disable_log_stats=True,
        seed=args.seed,
        enable_lora=True,
        max_lora_rank=int(max_lora_rank),
        max_loras=int(args.max_loras),
        max_cpu_loras=max(int(args.max_loras), n_loras),
    )
    dtype = args.dtype
    if dtype == "auto":
        cfg_dtype = model_cfg.get("torch_dtype")
        dtype = cfg_dtype if isinstance(cfg_dtype, str) and cfg_dtype else "auto"
    if args.quantization and args.quantization != "none":
        llm_kwargs["quantization"] = args.quantization
        if args.quantization == "bitsandbytes":
            # vLLM 的硬校验：bnb 量化必须搭配 load_format=bitsandbytes
            llm_kwargs["load_format"] = "bitsandbytes"
    if dtype and dtype != "auto":
        llm_kwargs["dtype"] = dtype

    from sfzy.utils.vllm_compat import filter_llm_kwargs

    llm_kwargs, dropped = filter_llm_kwargs(llm_kwargs)
    if dropped:
        logger.warning("当前 vLLM 不认识这些参数，已忽略：%s", dropped)
    logger.info(
        "加载 vLLM：%s（dtype=%s, tp=%d, max_model_len=%d, max_loras=%d, max_lora_rank=%d）",
        model_name, dtype, args.tensor_parallel_size, max_model_len,
        args.max_loras, max_lora_rank,
    )
    return LLM(**llm_kwargs)


def generate_run(llm: Any, tokenizer: Any, records: Sequence[Dict[str, Any]],
                 args: argparse.Namespace, *, out_path: Path, prompt_style: str,
                 max_prompt_len: int, lora_request: Any) -> int:
    """跑一个 adapter → ``out_path``（``{"id","source","reference","output"}``）。"""
    done = load_done_ids(out_path)
    todo = [r for r in records if str(r["id"]) not in done]
    if not todo:
        logger.info("已全部完成，跳过：%s（%d 条）", out_path.name, len(records))
        return 0
    logger.info("%s：共 %d 条，已完成 %d 条，本次生成 %d 条",
                out_path.name, len(records), len(done), len(todo))

    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0,                     # 贪心：与训练/评测口径一致，结果可复现
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )
    started = time.time()
    written = 0
    lengths: List[int] = []
    with open(out_path, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            prompt_ids = list(encode_prompts(chunk, tokenizer, prompt_style, max_prompt_len))
            outputs = llm.generate(
                prompts=[{"prompt_token_ids": ids} for ids in prompt_ids],
                sampling_params=sampling,
                lora_request=lora_request,
                use_tqdm=False,
            )
            for record, output in zip(chunk, outputs):
                text = output.outputs[0].text.strip() if output.outputs else ""
                f.write(json.dumps({
                    "id": record["id"],
                    "source": record["source"],
                    "reference": record.get("summary", ""),
                    "output": text,
                }, ensure_ascii=False) + "\n")
                lengths.append(len(text))
                written += 1
            f.flush()
            elapsed = time.time() - started
            rate = written / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - written) / rate / 60 if rate > 0 else float("inf")
            logger.info("  %s 进度 %d/%d | %.2f 条/秒 | 预计剩余 %.1f 分钟",
                        out_path.stem, written, len(todo), rate, eta)
    if lengths:
        logger.info("  %s 输出字数：平均 %.0f / 最短 %d / 最长 %d",
                    out_path.stem, sum(lengths) / len(lengths), min(lengths), max(lengths))
    return written


# ---------------------------------------------------------------------------
# ROUGE 与报告
# ---------------------------------------------------------------------------
def compute_rouge(run_files: Sequence[Path], references: Dict[str, str],
                  mode: str) -> List[Dict[str, Any]]:
    """逐文件算 ROUGE（逐样本宏平均，只统计两边都有的 id）。"""
    results: List[Dict[str, Any]] = []
    for path in run_files:
        preds: Dict[str, str] = {}
        lengths: List[int] = []
        for record in read_jsonl(str(path)):
            rid = str(record.get("id"))
            text = str(record.get("output") or "").strip()
            preds[rid] = text
            lengths.append(len(text))
        metrics = score_corpus(preds, references, mode=mode)
        results.append({
            "name": path.stem,
            "step": step_of(path.stem),
            "file": str(path),
            "n_pred": len(preds),
            "n_ref": len(references),
            "n_scored": int(metrics.get("num_samples", 0)),
            "mean_chars": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "empty_ratio": (sum(1 for v in preds.values() if not v) / len(preds))
            if preds else 0.0,
            **{k: float(v) for k, v in metrics.items() if k != "num_samples"},
        })

    def key(row: Dict[str, Any]):
        step = row["step"]
        return (step is None, step or 0, row["name"])

    return sorted(results, key=key)


def render_report(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 4) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"

    lines: List[str] = ["# LoRA step 验证集 ROUGE 报告\n"]
    lines.append(f"- 生成时间：{meta.get('created_at', '?')}")
    lines.append(f"- 配置：`{meta.get('config', '?')}`")
    lines.append(f"- 底座：`{meta.get('model', '?')}`"
                 f"（dtype={meta.get('dtype')}, quantization={meta.get('quantization')}）")
    lines.append(f"- 数据：`{meta.get('input', '?')}`，共 {meta.get('n_records', '?')} 条"
                 f"（prompt_style={meta.get('prompt_style')}，"
                 f"max_new_tokens={meta.get('max_new_tokens')}）")
    lines.append(f"- 分词口径：`{meta.get('rouge_mode')}`；打分方式：逐样本宏平均")
    lines.append(f"- 输出目录：`{meta.get('out_dir', '?')}`\n")

    lines.append("## 结果\n")
    lines.append("| step | 条数 | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | 官方总分 | 平均字数 | 空输出率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['step'] if row['step'] is not None else row['name']} "
            f"| {row['n_scored']} | {fmt(row.get('rouge-1-f'))} "
            f"| {fmt(row.get('rouge-2-f'))} | {fmt(row.get('rouge-l-f'))} "
            f"| {fmt(row.get('overall'))} | {row['mean_chars']:.0f} "
            f"| {row['empty_ratio']:.1%} |")
    scored = [r for r in rows if r["n_scored"]]
    if scored:
        best = max(scored, key=lambda r: r.get("overall", 0.0))
        lines.append("")
        lines.append(f"官方总分最高：**{best['step'] if best['step'] is not None else best['name']}**"
                     f"（overall {best.get('overall', 0.0):.4f}，"
                     f"ROUGE-1/2/L F = {fmt(best.get('rouge-1-f'), 4)} / "
                     f"{fmt(best.get('rouge-2-f'), 4)} / {fmt(best.get('rouge-l-f'), 4)}）")

    lines.append("\n## 说明\n")
    lines.append("- 每份 jsonl 一行一条：`id` / `source`（裁判文书原文）/ `reference`"
                 "（人工摘要）/ `output`（该 step 生成的候选摘要）。")
    lines.append("- ROUGE 只统计该文件与参考文件**共有的 id**，逐样本先算分再取宏平均"
                 "（CAIL2020 官方口径：0.2*R1 + 0.4*R2 + 0.4*RL）。")
    lines.append("- 若各 step 之间差异小于 0.005，多半是生成噪声，"
                 "建议加大样本量后再下结论。")
    lines.append("- `条数` 小于参考条数说明该 step 没跑完；重跑同一条命令会断点续跑补齐。")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, rows: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    (out_dir / "rouge.json").write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "runs": list(rows)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(rows, meta), encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="对若干个 LoRA step 在验证集上推理并算 ROUGE（vLLM）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", default="configs/grpo_fact_cov_ee_t4.yaml",
                    help="提供底座模型、LoRA 的 r/alpha、prompt 风格与数据切分")
    ap.add_argument("--adapters", nargs="+", required=True,
                    help="要评测的 LoRA：可以是 .pt（本项目存档，会自动导出成 PEFT 目录），"
                         "也可以是已导出的 PEFT 目录；支持通配符")
    ap.add_argument("--out-dir", required=True,
                    help="输出目录：<step>.jsonl + rouge.json/summary.json/report.md，"
                         "导出给 vLLM 的 adapter 放在 <out-dir>/lora_peft/")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--input", default=None,
                    help="直接指定输入 jsonl（含 id/source/summary），覆盖 --split")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--model", default=None, help="覆盖配置里的底座模型路径")
    ap.add_argument("--tokenizer", default=None, help="单独指定 tokenizer")
    ap.add_argument("--prompt-style", default=None, help="默认取配置的 data.prompt_style")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--repetition-penalty", type=float, default=1.1,
                    help="与 SFT/GRPO 的生成默认值一致，不要随意改")
    ap.add_argument("--max-prompt-len", type=int, default=None,
                    help="prompt 截断长度，默认取配置的 sft/rl/model.max_length")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="vLLM 的 max_model_len，默认 = max_prompt_len + max_new_tokens")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="每次提交给 vLLM 的 prompt 条数（只影响落盘粒度与峰值内存）")
    ap.add_argument("--max-loras", type=int, default=1,
                    help="同时驻留 GPU 的 adapter 数。默认 1 最省显存；"
                         "设成 adapter 个数可省掉换入换出")
    ap.add_argument("--dtype", default="float16", help="float16 | bfloat16 | auto")
    ap.add_argument("--quantization", default="none",
                    help="none | bitsandbytes | awq | gptq（T4 上 6B 必须 4-bit）")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--swap-space", type=int, default=4)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="关掉 CUDA graph，省显存/好调试，代价是慢一点")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rouge-mode", default="jieba", choices=["jieba", "char"],
                    help="ROUGE 分词口径；项目默认 jieba（与奖励/评测口径一致）")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config(resolve(args.config))
    model_cfg = resolve_model_cfg(cfg)
    data_cfg = resolve_data_cfg(cfg)

    model_name = args.model or model_cfg.get("model_name_or_path", "")
    local = resolve(model_name)
    if is_local_dir(local):
        model_name = str(local)
    if not model_name:
        raise SystemExit("配置里没有 model.model_name_or_path，也没有传 --model")

    out_dir = resolve(args.out_dir)
    ensure_dir(out_dir)
    logger.info("输出目录：%s", out_dir)

    lora_alpha = int(cfg.path_("lora.alpha", 16))
    lora_r = int(cfg.path_("lora.r", 8))
    lora_dropout = float(cfg.path_("lora.dropout", 0.0))
    config_modules = list(cfg.path_("lora.target_modules", []) or [])
    logger.info("LoRA：r=%d alpha=%d dropout=%s 配置里的 target_modules=%s",
                lora_r, lora_alpha, lora_dropout, config_modules or "（空）")

    runs = prepare_adapters(
        expand_adapters(args.adapters), out_dir, cfg_alpha=lora_alpha,
        cfg_dropout=lora_dropout, cfg_modules=config_modules,
        cfg_rank=lora_r, base_model=model_name,
    )

    data_path = resolve(args.input) if args.input else resolve_split_file(data_cfg, args.split)
    records = list(read_jsonl(str(data_path)))
    if args.limit:
        records = records[: args.limit]
    references = {str(r["id"]): str(r.get("summary", "")) for r in records}
    logger.info("数据：%s（%d 条）", data_path, len(records))

    prompt_style = args.prompt_style or data_cfg.path_("data.prompt_style", "structured")
    max_prompt_len = int(
        args.max_prompt_len
        or cfg.path_("sft.max_length")
        or cfg.path_("rl.max_length")
        or model_cfg.get("max_length")
        or 2048
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or model_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code")),
        padding_side="left",
    )
    max_lora_rank = max(
        [adapter_rank(Path(r["peft_dir"]), lora_r) for r in runs] or [lora_r])
    llm = build_llm(args, model_name, model_cfg, max_lora_rank, len(runs), max_prompt_len)

    started = time.time()
    run_files: List[Path] = []
    for index, run in enumerate(runs, start=1):
        out_path = out_dir / f"{run['name']}.jsonl"
        generate_run(
            llm, tokenizer, records, args, out_path=out_path,
            prompt_style=prompt_style, max_prompt_len=max_prompt_len,
            lora_request=make_lora_request(Path(run["peft_dir"]), index),
        )
        run_files.append(out_path)
    logger.info("全部生成完成，耗时 %.1f 分钟", (time.time() - started) / 60)

    rows = compute_rouge(run_files, references, args.rouge_mode)
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(resolve(args.config)),
        "model": model_name,
        "input": str(data_path),
        "n_records": len(records),
        "prompt_style": prompt_style,
        "max_prompt_len": max_prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "rouge_mode": args.rouge_mode,
        "out_dir": str(out_dir),
        "lora": {"alpha": lora_alpha, "dropout": lora_dropout,
                 "config_r": lora_r, "config_target_modules": config_modules},
        "runs": runs,
    }
    write_outputs(out_dir, rows, meta)

    logger.info("输出：%s\n报告：%s\n指标：%s",
                ", ".join(str(p) for p in run_files),
                out_dir / "report.md", out_dir / "rouge.json")
    for row in rows:
        logger.info("  %-16s overall=%.4f  R1=%.4f R2=%.4f RL=%.4f  n=%d",
                    row["name"], row.get("overall", 0.0), row.get("rouge-1-f", 0.0),
                    row.get("rouge-2-f", 0.0), row.get("rouge-l-f", 0.0), row["n_scored"])


if __name__ == "__main__":
    main()
