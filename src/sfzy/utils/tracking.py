"""实验跟踪的抽象层（工程代码，已实现）。

trainer 只调 `tracker.log(metrics, step)` 和 `tracker.finish()`，
不知道后端是 swanlab、wandb，还是只往 stdout 打印。

两个设计选择：

1. **本地调试默认用 NullTracker**，只打印、不联网、不用登录。
   跑单元测试和调试数据管线时不该被实验跟踪拖住。

2. **swanlab 和 wandb 的 API 是兼容的**（swanlab 刻意对齐了 wandb），
   所以两个后端共用同一段初始化代码。国内直连 swanlab 比 wandb 快很多，
   这是个实际的工程选择，不是随意换个库。

断点续训时要把 `resume_id` 传进来接回原来那次实验，
否则一次训练会在面板上被拆成好几条断掉的曲线。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple


class Tracker(Protocol):
    """实验跟踪的最小接口。trainer 只依赖这三个方法。"""

    def log(self, metrics: Dict[str, float], step: int) -> None:
        ...

    def finish(self) -> None:
        ...

    @property
    def run_id(self) -> Optional[str]:
        """当前 run 的 id，写进 checkpoint 供续训时接回去。"""
        ...


class NullTracker:
    """不做任何上报，只保证接口完整。"""

    def log(self, metrics: Dict[str, float], step: int) -> None:  # noqa: D102
        pass

    def finish(self) -> None:  # noqa: D102
        pass

    @property
    def run_id(self) -> Optional[str]:  # noqa: D102
        return None


class SwanlabWandbTracker:
    """swanlab / wandb 的共用实现 —— 两者 API 兼容。"""

    def __init__(
        self,
        backend: str,
        project: str,
        run_name: Optional[str] = None,
        resume_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        columns: Optional[List[Tuple[str, str, str]]] = None,
    ) -> None:
        try:
            if backend == "swanlab":
                import swanlab as module
            elif backend == "wandb":
                import wandb as module
            else:
                raise ValueError(f"未知的 tracking 后端: {backend}")
        except ImportError as exc:
            # 干瘪的 ModuleNotFoundError 会让人以为是自己代码写错了。
            # 这里在**训练开始之前**就失败（tracker 先于 trainer 构造），
            # 所以给出两条明确的出路就够了。
            raise RuntimeError(
                f"tracking.backend={backend} 需要安装 {backend}：\n"
                f"  pip install {backend} -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                "  或者把配置里的 tracking.backend 改成 none 先跑起来。"
            ) from exc

        self._module = module
        # resume_id 非空时用 resume="must"：宁可报错也不要静默新建一条 run，
        # 因为静默新建意味着你的续训曲线和原来的断开了，而你不会立刻发现。
        run = module.init(
            project=project,
            name=run_name,
            id=resume_id,
            resume="must" if resume_id else None,
            config=config or {},
        )
        self._run = run
        self._run_id = resume_id or getattr(run, "id", None)

        # 预声明要画的列。为什么值得做：不声明的话，面板上的列是**第一次
        # log 到它的时候**才冒出来的，顺序和分组全凭日志出现顺序；声明之后
        # 分组固定、中文名可见、训练一开始就能看到全部面板（哪怕还是空的）。
        # 只有 swanlab 有这个 API，wandb 靠 `/` 前缀自动分组，所以静默跳过。
        define = getattr(module, "define_scalar", None)
        for key, name, chart in (columns or []):
            if define is None:
                break
            try:
                define(key=key, name=name, chart_name=chart)
            except Exception:  # noqa: BLE001 — 后端版本差异不该让训练起不来
                break

    def log(self, metrics: Dict[str, float], step: int) -> None:  # noqa: D102
        # 显式传 step，否则各后端自己按调用次数编号，
        # 断点续训后会从头计数，曲线会出现重叠。
        self._module.log(dict(metrics), step=step)

    def finish(self) -> None:  # noqa: D102
        self._module.finish()

    @property
    def run_id(self) -> Optional[str]:  # noqa: D102
        return self._run_id


def build_tracker(
    backend: str = "none",
    project: str = "sfzy",
    run_name: Optional[str] = None,
    resume_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    columns: Optional[List[Tuple[str, str, str]]] = None,
    enabled: bool = True,
) -> Tracker:
    """按配置构建 tracker。

    enabled=False（或非主进程）时直接返回 NullTracker ——
    多进程下每个 rank 都去 init 一遍会把面板弄乱。
    """
    if not enabled or backend in ("", "none", None):
        return NullTracker()
    return SwanlabWandbTracker(
        backend=backend,
        project=project,
        run_name=run_name,
        resume_id=resume_id,
        config=config,
        columns=columns,
    )


def make_run_name(prefix: str = "sfzy", **hyperparams: Any) -> str:
    """把关键超参写进 run 名字，方便在面板上一眼分辨。

    例：sfzy-lora_r8-a16-lr2e-4-ep3
    """
    parts = [prefix]
    for key, value in hyperparams.items():
        if value is None:
            continue
        text = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{key}{text}")
    return "-".join(parts)


# ---------------------------------------------------------------------------
# GRPO 的指标注册表
# ---------------------------------------------------------------------------
# 训练循环产出的指标名是**扁平的**（`reward_mean`、`kept_groups`…），直接丢给
# swanlab 会得到一长条平铺的曲线，分不清哪几根是一组、该怎么一起读。
#
# 这里做两件事：
#   1. 给每个指标定一个带 `/` 前缀的上报名（swanlab / wandb 都按 `/` 自动分组），
#      面板上就是 reward/…、group/…、judge/… 几块；
#   2. 用一句话写清**这个指标该怎么读** —— 这份表同时就是"GRPO 要看什么"的清单。
#
# 注意：**只改上报名，不改 checkpoint history 里的原始键**。history 是给
# 离线分析（tools/compare_runs.py 之类）读的，键名一旦跟着后端走就锁死了。
GRPO_METRICS: Dict[str, Tuple[str, str]] = {
    # ---- 策略优化健康度：先看这四根，判断"训练有没有在正常进行" ----
    "loss": ("train/loss", "GRPO 损失（-min(ratio·A, clip(ratio)·A) 的均值）"),
    "grad_norm": ("train/grad_norm", "裁剪前梯度范数；尖峰 = 训练不稳"),
    "kl": ("train/kl", "相对参考策略的 KL；kl_coef=0 时恒为 0"),
    "clipped_frac": ("train/clip_frac", "ratio 被裁剪的序列比例；健康值 ≈ 0"),
    # ---- 奖励：GRPO 的全部学习信号都在这 ----
    "reward_mean": ("reward/mean", "奖励均值"),
    "reward_std": ("reward/std", "奖励标准差（整个 batch，不分组）"),
    "reward_group_std": ("reward/group_std", "★ 组内标准差均值 —— 有没有可学的信号"),
    "rouge_l": ("reward/rouge_l", "ROUGE-L（官方口径的锚）"),
    "fact_judge": ("reward/fact_judge", "六要素事实一致性奖励（fact_judge 模式的主信号）"),
    "fact_judge_group_std": ("reward/fact_judge_group_std", "★ 事实项的组内标准差"),
    "fact_score": ("reward/fact_f1", "规则事实 F1；fact_judge 模式下恒为 1，别读"),
    "fact_coverage": ("reward/fact_coverage", "事实覆盖率；只在参考含事实时有定义"),
    "fact_precision": ("reward/fact_precision", "事实精确率；掉 = 在堆数字"),
    "gating_rate": ("reward/gate_rate", "被门控拦下的比例；高 = 问题在生成不在奖励"),
    # ---- 组与优势：GRPO 特有的诊断 ----
    "advantage_abs_mean": ("group/advantage_abs_mean", "|优势| 均值；≈0 = 整组没方差"),
    "kept_ratio": ("group/kept_ratio", "★ 保留组比例；低 = 大量算力白烧"),
    "anchor_filtered": ("group/anchor_filtered", "被 SFT 基线锚过滤掉的组数"),
    # ---- 裁判（卡 1 那个模型）的状态 ----
    "judge_missing": ("judge/missing_ratio", "裁判整组失败的比例；>10% 必须先查"),
    "mean_fact_reward": ("judge/mean_fact_reward", "六要素加权事实一致性分（需求第十二节）"),
    "mean_min_element_score": ("judge/min_element_score", "六要素最低分（不含案由）"),
    # ---- 输出形态 ----
    "output_len_mean": ("length/output_mean", "输出平均字符数"),
}

# 由六要素 Judge 动态产出的键（`summarize_scores`）：按模式改名。
_ELEMENT_SCORE_RE = re.compile(r"^mean_([a-z_]+)_score$")
_RATIO_SCORE_RE = re.compile(r"^ratio_score_(\d)$")


def _element_label(name: str) -> str:
    """六要素的英文键 → 中文名。延迟 import，避免 utils 依赖 judge 模块。"""
    try:
        from sfzy.judge.schema import ELEMENT_ZH

        return ELEMENT_ZH.get(name, name)
    except Exception:  # noqa: BLE001 — 没装 judge 依赖时退化成英文键
        return name


def group_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """把训练循环的扁平指标名映射成带分组前缀的上报名。

    认不出来的键**原样透传**（而不是丢掉）：漏报一个指标比名字难看得多，
    而且新加的指标不该因为忘了登记就从面板上消失。
    """
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        spec = GRPO_METRICS.get(key)
        if spec is not None:
            out[spec[0]] = value
            continue

        match = _RATIO_SCORE_RE.match(key)
        if match:
            out[f"judge/score_ratio/{match.group(1)}"] = value
            continue

        match = _ELEMENT_SCORE_RE.match(key)
        if match:
            out[f"judge/element/{_element_label(match.group(1))}"] = value
            continue

        out[key] = value
    return out


def metric_columns() -> List[Tuple[str, str, str]]:
    """给 `swanlab.define_scalar` 用的列定义：[(上报键, 中文名, 图表分组), ...]。"""
    columns = [
        (tracked, name, tracked.split("/")[0])
        for tracked, name in GRPO_METRICS.values()
    ]
    # 六要素的逐项均值和 0-4 档位分布是动态键，这里按已知的六项预先声明。
    for name in ("case_type", "plaintiff_claims", "defendant_defenses",
                 "court_facts", "legal_basis", "judgment_result"):
        columns.append((f"judge/element/{_element_label(name)}",
                        f"{_element_label(name)} 的均值", "judge"))
    for level in range(5):
        columns.append((f"judge/score_ratio/{level}", f"{level} 分占比", "judge"))
    return columns


def flatten_config(cfg: Any, prefix: str = "") -> Dict[str, Any]:
    """把嵌套配置压成 `a.b.c` 的扁平字典，供实验跟踪记录超参。

    只保留标量和列表（列表拼成逗号串）：swanlab 的 config 面板对嵌套结构
    支持不好，塞个 dict 进去只会变成一坨不可读的东西。
    `None` 也跳过 —— 配置里一堆 `resume_from: null` 之类的空值只会把面板刷屏。
    """
    flat: Dict[str, Any] = {}
    if not isinstance(cfg, dict):
        return flat
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if value is None:
            continue
        if isinstance(value, dict):
            flat.update(flatten_config(value, prefix=f"{name}."))
        elif isinstance(value, (list, tuple)):
            flat[name] = ",".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            flat[name] = value
    return flat
