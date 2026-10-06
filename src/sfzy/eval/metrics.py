"""裁判契约 + 裁判工厂。

============================ 只有一种裁判 ============================
整个项目只有**一个**裁判实现：六要素 Judge（`semantic.backend=fact`），
由 `sfzy/judge/scorer.py::SixElementScorer` 提供。它一次产出多路**命名信号**：

    [{"fact_consistency": 0.83, "element_coverage": 0.51}, ...]

点式打分（0-100）、组内排序、API、缓存四个后端属于已淘汰方案，已经删除；
现在的扩展方向是**加 task / 加信号**，不是加 scorer 种类。

============================ 契约用 Protocol，不用继承 ============================
`SignalScorer` 是这份契约的结构化描述。用 `typing.Protocol` 而不是抽象基类：
唯一的实现不需要为了"证明自己符合接口"去继承一个只服务于它的空壳，
测试里的桩也不需要。真正被依赖的只有两个方法 + 一个日志标签。

`build_scorer` 是唯一的构造入口：按配置决定要不要裁判，并把 reward 配置里的
内部权重转交给实现。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Set


class SignalScorer(Protocol):
    """裁判契约：把一批候选变成多路命名信号。

    实现者只要有下面两个方法即可（`name` 只是日志里的显示名，
    不参与任何分发；`available_signals()` 才是能力声明）。
    """

    name: str

    def available_signals(self) -> Set[str]:
        """这个裁判能产出哪些信号名。"""
        ...

    def score_batch_signals(
        self, items: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Optional[float]]]:
        """每条候选 → {信号名: 分数 ∈ [0,1]}。

        `None` 表示这条这一路打分失败（提取失败 / 显存抖动），**不能当 0 用**；
        整组失败时该条返回空字典，由 trainer 按缺失补。
        """
        ...


def build_scorer(
    cfg: Optional[Dict[str, Any]],
    override: Optional[str] = None,
    term_options: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[SignalScorer]:
    """按配置构造裁判。现在只支持 `fact`（六要素 Judge）与 `none`（不接）。

    `term_options` 是 `RewardSpec.judge_term_options()` 的产物，形如
    `{信号名: {内部权重字段: 值}}` —— reward 的内部权重只能来自配置文件，
    这里只负责把它转交给裁判实现。
    """
    spec = dict(cfg or {})
    backend = override or spec.get("backend", "none")
    if backend in ("none", "", None):
        return None

    if backend in ("fact", "six", "fact_consistency"):
        return _build_six_element_scorer(spec, dict(term_options or {}))

    raise ValueError(
        f"未知的 semantic.backend：{backend}（现在只支持 fact / none）"
    )


def _build_six_element_scorer(
    spec: Dict[str, Any], term_options: Dict[str, Dict[str, Any]]
) -> SignalScorer:
    """构造六要素 Judge 的 scorer（`semantic.backend=fact`）。

    延迟 import：`sfzy.judge` 会带上 torch / transformers，
    不需要裁判的路径不该为它付导入代价。
    """
    # 六要素权重曾经能写在 semantic.weights 下，现在只能写在 reward term 里。
    # 留着旧写法会变成"两个地方都能设权重"，正是要避免的事。
    if spec.get("weights"):
        raise ValueError(
            "semantic.weights 已废弃：六要素权重现在只能写在配置文件里，"
            "位置是 rl.reward.terms.<reward>.element_weights。"
        )

    from sfzy.judge.judge import SUPPORTED_TASKS, FactConsistencyJudge
    from sfzy.judge.scorer import SixElementScorer

    # 要跑哪些任务由 reward 配置决定：`judge_term_options()` 的键就是信号名。
    tasks = sorted(term_options) or ["fact_consistency"]
    unsupported = [t for t in tasks if t not in SUPPORTED_TASKS]
    if unsupported:
        raise ValueError(
            f"裁判还不支持这些信号：{unsupported}（支持：{list(SUPPORTED_TASKS)}）"
        )

    model = spec.get("model")
    if not model:
        raise ValueError("semantic.backend=fact 需要 semantic.model 指向裁判模型目录")

    # 抽取可以用一个单独的、更强的模型（`semantic.extract_model`）。
    # 抽取是最容易出错的一步（实测会把日期挪用/改写），换大模型比换裁判更值。
    # 不给就共用裁判模型。注意这会在显存里多放一份权重。
    extract_model = spec.get("extract_model")
    extract_runtime = None
    max_input_tokens = int(spec.get("max_input_tokens", 8192))
    if extract_model and extract_model != model:
        from sfzy.judge.runtime import build_runtime, resolve_model_path

        extract_runtime = build_runtime(
            model_path=resolve_model_path(extract_model),
            device=spec.get("extract_device", spec.get("device", "cuda:1")),
            dtype=spec.get("extract_dtype", spec.get("dtype", "bfloat16")),
            max_batch_size=int(spec.get("max_batch_size", 8)),
            max_input_tokens=max_input_tokens,
            load_in_4bit=bool(
                spec.get("extract_load_in_4bit", spec.get("load_in_4bit", False))
            ),
            bnb_4bit_compute_dtype=spec.get("extract_bnb_4bit_compute_dtype"),
            trust_remote_code=bool(
                spec.get("extract_trust_remote_code", spec.get("trust_remote_code", True))
            ),
        )

    judge = FactConsistencyJudge(
        model_path=model,
        device=spec.get("device", "cuda:1"),
        dtype=spec.get("dtype", "bfloat16"),
        max_batch_size=int(spec.get("max_batch_size", 8)),
        extract_max_new_tokens=int(spec.get("extract_max_new_tokens", 1024)),
        # 长文书（实测最长约 1.2 万字）在 4096 下会被截断，抽取直接残缺。
        max_input_tokens=max_input_tokens,
        extract_runtime=extract_runtime,
        # 权重来自 reward 配置；直接调 build_scorer（探针 / 离线打分工具）
        # 不带 term_options 时，FactConsistencyJudge 会用 schema.DEFAULT_WEIGHTS。
        weights=(term_options.get("fact_consistency") or {}).get("element_weights"),
        coverage_weights=(term_options.get("element_coverage") or {}).get("element_weights"),
        tasks=tasks,
        min_document_elements=int(spec.get("min_document_elements", 2)),
        doc_fallback=bool(spec.get("doc_fallback", True)),
        # 裁判侧 4-bit（NF4）：7B 在 24GB 卡上量化后约 5~6GB，位置由
        # device_map 定在 semantic.device 指的卡上，与策略分居两卡。
        load_in_4bit=bool(spec.get("load_in_4bit", False)),
        bnb_4bit_compute_dtype=spec.get("bnb_4bit_compute_dtype"),
        trust_remote_code=bool(spec.get("trust_remote_code", True)),
    )
    return SixElementScorer(judge)
