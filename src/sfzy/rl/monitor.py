"""GRPO 训练的 SwanLab 监控层（**纯只读派生**，不参与任何训练计算）。

============================ 设计边界 ============================
这个模块只做一件事：把 `GRPOTrainer.step()` 里**已经产生**的数据（奖励分项、
裁判信号、优势、logprob 比值、长度、生成文本……）摊平成一组监控指标。它

  * 不修改 advantage / 组过滤 / 长度归一化 / PPO 裁剪 / KL / LoRA 更新；
  * 不修改 Reward 权重、Judge Prompt、Gate 规则；
  * 不额外调用裁判、不重新生成摘要；
  * 只在主进程被调用（由 trainer 保证）。

所有指标名都是 `模块/指标` 形式，直接交给 `tracking.group_metrics()` 透传。

============================ 统计口径 ============================
为避免"曲线好看但口径说不清"，每条指标的分母都在实现处写明：

  * `raw`：门控前奖励 / 组内归一化后、长度缩放与 Mask 前的优势；
  * `final`：实际进入 GRPO 优势计算的奖励；
  * `effective`：经过长度缩放 + 组 Mask 后真正进 loss 的优势；
  * judge 分项统计只在**信号可用**的样本上算（`None` 不填 0），分母单独计；
  * 组内统计用 population std（`unbiased=False`），和 `grpo.group_advantages`
    的口径一致；
  * 相关系数遇到样本不足或常量向量时**返回缺失**，不写 0。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEFAULT_MONITOR: Dict[str, Any] = {
    "log_interval": 1,
    "reward_log_interval": 10,
    "relation_log_interval": 20,
    "entropy_log_interval": 20,
    "lora_update_log_interval": 20,
    "correlation_window": 512,
    "near_zero_std_eps": 1e-6,
    "near_upper_gate_margin": 0.1,
    "ee_low_threshold": 0.5,
    "ee_high_threshold": 0.75,
    "repetition_ngram": 3,
    "relation_enabled": True,
    "eval_interval": 100,
    "enabled": True,
}


@dataclass
class MonitorConfig:
    enabled: bool = True
    log_interval: int = 1
    reward_log_interval: int = 10
    relation_log_interval: int = 20
    entropy_log_interval: int = 20
    lora_update_log_interval: int = 20
    correlation_window: int = 512
    near_zero_std_eps: float = 1e-6
    near_upper_gate_margin: float = 0.1
    ee_low_threshold: float = 0.5
    ee_high_threshold: float = 0.75
    repetition_ngram: int = 3
    relation_enabled: bool = True
    # ---- 固定验证集（可选；项目没有内置验证流程时保持关闭）----
    eval_enabled: bool = False
    eval_data_file: Optional[str] = None
    eval_every_n_steps: int = 100
    # 按**分位点**验证：把总步数 n 等分成 x 段，在第 k/x 分位点各验一次（k=1..x），
    # 所以最后一个点就是训练结束那一步。x=0 表示不启用（退回 eval_every_n_steps）。
    eval_quantile_points: int = 0
    eval_max_samples: int = 100
    eval_max_new_tokens: int = 384
    # None = 与训练用同一个 rl.max_length（推荐：验证的截断规则必须和训练一致，
    # 否则"训练看 7168 token、验证只喂 2048"会让验证分系统性低于真实水平，
    # 白白压低基线）。给了整数则取 min(它, rl.max_length)。
    eval_max_length: Optional[int] = None
    eval_baseline: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: Optional[Dict[str, Any]]) -> "MonitorConfig":
        merged = dict(DEFAULT_MONITOR)
        merged.update(dict(raw or {}))
        ev = dict(merged.get("eval") or {})
        eval_interval = int(merged.get("eval_interval", 100))
        if "every_n_steps" in ev:
            eval_interval = int(ev["every_n_steps"])
        return cls(
            enabled=bool(merged["enabled"]),
            log_interval=max(1, int(merged["log_interval"])),
            reward_log_interval=max(1, int(merged["reward_log_interval"])),
            relation_log_interval=max(1, int(merged["relation_log_interval"])),
            entropy_log_interval=max(1, int(merged["entropy_log_interval"])),
            lora_update_log_interval=max(1, int(merged["lora_update_log_interval"])),
            correlation_window=max(8, int(merged["correlation_window"])),
            near_zero_std_eps=float(merged["near_zero_std_eps"]),
            near_upper_gate_margin=float(merged["near_upper_gate_margin"]),
            ee_low_threshold=float(merged["ee_low_threshold"]),
            ee_high_threshold=float(merged["ee_high_threshold"]),
            repetition_ngram=max(1, int(merged["repetition_ngram"])),
            relation_enabled=bool(merged["relation_enabled"]),
            eval_enabled=bool(ev.get("enabled", False)),
            eval_data_file=ev.get("data_file"),
            eval_every_n_steps=max(1, eval_interval),
            eval_quantile_points=max(0, int(ev.get("quantile_points", 0) or 0)),
            eval_max_samples=max(1, int(ev.get("max_samples", 100))),
            eval_max_new_tokens=int(ev.get("max_new_tokens", 384)),
            eval_max_length=(
                None if ev.get("max_length") is None else int(ev["max_length"])
            ),
            eval_baseline=dict(ev.get("baseline") or {}),
        )


# ---------------------------------------------------------------------------
# 基础统计工具（纯 python，不引入 numpy 依赖）
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _pstd(values: Sequence[float]) -> float:
    """population std（分母 N），与 `grpo.group_advantages` 的 std 口径一致。"""
    values = list(values)
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _percentile(values: Sequence[float], q: float) -> float:
    """线性插值分位数（与 numpy 默认口径一致）。`q ∈ [0,1]`。"""
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _rate(flags: Sequence[bool]) -> float:
    flags = list(flags)
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0


def _rankdata(values: Sequence[float]) -> List[float]:
    """平均秩（tie 取平均），用于 Spearman。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float], eps: float = 1e-12) -> Optional[float]:
    """Spearman 相关系数；样本 <3 或任一维度无方差时返回 None（**不写 0**）。"""
    pairs = [(float(a), float(b)) for a, b in zip(x, y)]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if max(xs) - min(xs) <= eps or max(ys) - min(ys) <= eps:
        return None
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    mx = _mean(rx)
    my = _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= eps or dy <= eps:
        return None
    value = num / (dx * dy)
    return max(-1.0, min(1.0, value))


def _group_slices(n: int, group_size: int) -> List[Tuple[int, int]]:
    return [(s, min(s + group_size, n)) for s in range(0, n, group_size)]


def _group_values(values: Sequence[float], group_size: int) -> List[List[float]]:
    return [list(values[s:e]) for s, e in _group_slices(len(values), group_size)]


def _put(out: Dict[str, float], name: str, value: Optional[float]) -> None:
    """写入一个指标；None 表示"当前不具备数据"，直接跳过而不是填 0。"""
    if value is None:
        return
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return
    out[name] = float(value)


# ---------------------------------------------------------------------------
# 一批次数据
# ---------------------------------------------------------------------------
@dataclass
class MonitorBatch:
    """一步训练里**已经算好**的数据快照（全部按候选顺序、group-major 排列）。"""

    step: int
    group_size: int
    responses: List[str]
    references: List[str]
    final_rewards: List[float]
    pre_gate_rewards: List[float]
    gate_flags: List[bool]
    gate_reasons: List[str]
    term_values: Dict[str, List[float]]
    term_weights: Dict[str, float]
    raw_signals: Dict[str, List[Optional[float]]]
    judge_results: List[Any]
    advantages_raw: List[float]
    advantages_effective: List[float]
    kept_mask: List[bool]
    output_lengths: List[int]
    token_lengths: List[int]
    ratios: List[float]
    normalization_factors: List[float]
    length_norm_mode: str
    clip_frac: float
    policy_loss: float
    total_loss: float
    grad_norm_preclip: float
    grad_norm_postclip: float
    clip_applied: float
    learning_rate: float
    clip_ratio: float
    fact_gate_enabled: bool
    upper_length_ratio: Optional[float]
    result_markers: Sequence[str] = ()
    forbidden_prefixes: Sequence[str] = ()
    max_new_tokens: int = 0
    entropy: Optional[List[float]] = None
    lora_update_norm: Optional[float] = None
    # 仅在 kl_coef>0 且真的算了参考 KL 时给出；否则保持 None（不写 0）。
    kl_estimate: Optional[float] = None
    kl_penalty: Optional[float] = None

    @property
    def n(self) -> int:
        return len(self.responses)

    @property
    def num_groups(self) -> int:
        return (self.n + self.group_size - 1) // self.group_size if self.group_size else 0


# ---------------------------------------------------------------------------
# 监控器
# ---------------------------------------------------------------------------
class GRPOMonitor:
    """把一步训练的数据变成一组带分组前缀的 SwanLab 指标。"""

    def __init__(
        self,
        cfg: MonitorConfig,
        fact_enabled: bool = False,
        coverage_enabled: bool = False,
        ee_enabled: bool = False,
    ) -> None:
        self.cfg = cfg
        self.fact_enabled = fact_enabled
        self.coverage_enabled = coverage_enabled
        self.ee_enabled = ee_enabled
        self.total_steps: Optional[int] = None
        # 按分位点验证的调度表（set_total_steps 时算出来），None = 用 every_n_steps
        self._eval_steps: Optional[Tuple[int, ...]] = None
        # 跨 step 的相关性窗口：每条候选一条记录
        self._window: deque = deque(maxlen=cfg.correlation_window)

    def set_total_steps(self, total_steps: int) -> None:
        """训练总步数已知后，若总步数很少则自动缩短日志间隔；顺便排定验证点。

        分位点调度（`eval.quantile_points = x`）就是"把 n 步等分成 x 段"：

            x=4, n=1000 → 在第 250 / 500 / 750 / 1000 步验证

        最后一个点必然落在最后一步（k=x），所以"训练结束时的模型"一定会被评到、
        也一定有机会成为 best。x 大于总步数时按 set 去重（不会重复验同一步）。
        """
        self.total_steps = max(1, int(total_steps))
        x = self.cfg.eval_quantile_points
        if x > 0:
            total = self.total_steps
            if x >= total:
                # 分位点比步数还多（冒烟测试/短 run）：相邻分位点只差一步，中间那些
                # 没有信息量，而**每一次验证都是 max_samples 条串行贪心生成**
                # （默认 100 条 × 最多几百 token）。按分位点逐个验会变成"每一步都跑
                # 一次完整验证"—— 实测一次验证十几分钟，于是"每步 16 分钟"。
                # 短 run 只需要最后那一个模型，所以退化成"只在最后一步验一次"。
                self._eval_steps = (total,)
            else:
                self._eval_steps = tuple(sorted({
                    max(1, min(total, int(round(total * k / x)))) for k in range(1, x + 1)
                }))
        else:
            self._eval_steps = None

    def eval_schedule(self) -> Optional[Tuple[int, ...]]:
        """本 run 的验证步（按分位点时才有；否则 None 表示按 every_n_steps 轮询）。"""
        return self._eval_steps

    def _interval(self, base: int) -> int:
        """步数少时缩短间隔：保证至少能记到几次扩展统计。"""
        if not self.total_steps:
            return base
        return max(1, min(base, max(1, self.total_steps // 5)))

    def entropy_due(self, step: int) -> bool:
        """要不要算策略熵。

        **刻意不走 `_interval` 的自适应缩短。** 熵要物化完整的 `(B, L, vocab)` fp32
        概率（L=2400、N=8 时峰值 5~10 GB），是整条链路里最吃显存的一项；而
        `_interval` 的规则是"短 run 至少记到几次扩展统计"，会把间隔缩到 1 ——
        于是 2~4 步的冒烟 run 里**每一步**都算熵，直接把训练推到 OOM（实测踩过）。
        长 run 上两者本来一致（间隔 20 就是每 20 步），所以这里只是把短 run 变安全。
        """
        return (
            self.cfg.enabled
            and step % max(1, int(self.cfg.entropy_log_interval)) == 0
        )

    def lora_due(self, step: int) -> bool:
        return self.cfg.enabled and step % self._interval(self.cfg.lora_update_log_interval) == 0

    def eval_due(self, step: int) -> bool:
        if not (self.cfg.enabled and self.cfg.eval_enabled and self.cfg.eval_data_file):
            return False
        if self._eval_steps is not None:
            # 分位点调度：显式列出来的步才验（不再叠加 `_interval` 的自动缩短，
            # 否则 1000 步的 run 里"每 250 步"会被缩成"每 200 步"，和配置不符）。
            return step in self._eval_steps
        return step % self._interval(self.cfg.eval_every_n_steps) == 0

    # ------------------------------------------------------------------
    def observe(self, batch: MonitorBatch) -> Dict[str, float]:
        """返回本步应上报的指标（按配置的频率决定记哪些块）。"""
        if not self.cfg.enabled:
            return {}
        step = batch.step
        out: Dict[str, float] = {}
        self._update_window(batch)
        if step % self._interval(self.cfg.log_interval) == 0:
            out.update(self._core_metrics(batch))
        if step % self._interval(self.cfg.reward_log_interval) == 0:
            out.update(self._reward_detail_metrics(batch))
        if (
            self.cfg.relation_enabled
            and step % self._interval(self.cfg.relation_log_interval) == 0
        ):
            out.update(self._relation_metrics(batch))
        return out

    # ------------------------------------------------------------------
    # 窗口（跨 step 相关性用）
    # ------------------------------------------------------------------
    def _update_window(self, b: MonitorBatch) -> None:
        fact = b.raw_signals.get("fact_consistency", [None] * b.n)
        cov = b.raw_signals.get("element_coverage", [None] * b.n)
        ee = b.raw_signals.get("expression_efficiency", [None] * b.n)
        for i in range(b.n):
            self._window.append({
                "final_reward": b.final_rewards[i],
                "length": b.output_lengths[i],
                "coverage": cov[i],
                "ee": ee[i],
                "fact": fact[i],
                "adv_eff": b.advantages_effective[i],
            })

    def _window_corr(self, key_x: str, key_y: str) -> Optional[float]:
        xs, ys = [], []
        for rec in self._window:
            a, b = rec.get(key_x), rec.get(key_y)
            if a is None or b is None:
                continue
            xs.append(a)
            ys.append(b)
        return _spearman(xs, ys)

    # ------------------------------------------------------------------
    # 一等块：训练稳定性 + 组 + 优势 + 总 reward + 长度 + 行为 + gate 原因
    # ------------------------------------------------------------------
    def _core_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        G = b.group_size
        n = b.n

        # ---- train/* ----
        out["train/loss"] = b.total_loss
        out["train/policy_loss"] = b.policy_loss
        out["train/learning_rate"] = b.learning_rate
        out["train/grad_norm_preclip"] = b.grad_norm_preclip
        out["train/grad_norm_postclip"] = b.grad_norm_postclip
        out["train/gradient_clip_applied_rate"] = b.clip_applied
        out["train/global_step"] = float(b.step)
        _put(out, "train/ratio_mean", _mean(b.ratios))
        _put(out, "train/ratio_std", _pstd(b.ratios))
        _put(out, "train/ratio_max", max(b.ratios) if b.ratios else None)
        _put(out, "train/ratio_p90", _percentile(b.ratios, 0.90) if b.ratios else None)
        out["train/clip_frac"] = b.clip_frac
        lower, upper = 1.0 - b.clip_ratio, 1.0 + b.clip_ratio
        out["train/clip_frac_low"] = _rate([r < lower for r in b.ratios])
        out["train/clip_frac_high"] = _rate([r > upper for r in b.ratios])
        if b.entropy is not None:
            _put(out, "train/entropy", _mean(b.entropy))
        if b.lora_update_norm is not None:
            out["train/lora_update_norm"] = b.lora_update_norm
        if b.kl_estimate is not None:
            out["train/ref_kl_estimate"] = b.kl_estimate
        if b.kl_penalty is not None:
            out["train/kl_penalty"] = b.kl_penalty

        # ---- grpo/*（组内最终 reward）----
        groups = _group_values(b.final_rewards, G)
        group_means = [_mean(g) for g in groups]
        group_stds = [_pstd(g) for g in groups]
        group_ranges = [(max(g) - min(g)) if g else 0.0 for g in groups]
        _put(out, "grpo/group_reward_mean", _mean(group_means))
        _put(out, "grpo/group_reward_std_mean", _mean(group_stds))
        _put(out, "grpo/group_reward_std_p10", _percentile(group_stds, 0.10))
        _put(out, "grpo/group_reward_std_p50", _percentile(group_stds, 0.50))
        _put(out, "grpo/group_reward_std_p90", _percentile(group_stds, 0.90))
        _put(out, "grpo/group_reward_range_mean", _mean(group_ranges))
        _put(out, "grpo/group_reward_range_p90", _percentile(group_ranges, 0.90))
        eps = self.cfg.near_zero_std_eps
        zero_std = [s == 0.0 for s in group_stds]
        near_zero = [s < eps for s in group_stds]
        all_zero = [all(x == 0.0 for x in g) for g in groups]
        all_equal_pos = [
            (max(g) == min(g) and max(g) > 0.0) if g else False for g in groups
        ]
        has_variance = [s > eps for s in group_stds]
        kept = list(b.kept_mask) if b.kept_mask else has_variance
        effective = [k and v for k, v in zip(kept, has_variance)]
        out["grpo/zero_std_group_rate"] = _rate(zero_std)
        out["grpo/near_zero_std_group_rate"] = _rate(near_zero)
        out["grpo/all_zero_group_rate"] = _rate(all_zero)
        out["grpo/all_equal_positive_group_rate"] = _rate(all_equal_pos)
        out["grpo/effective_group_rate"] = _rate(effective)
        out["grpo/kept_group_rate"] = _rate(kept)
        out["grpo/filtered_group_rate"] = _rate([not k for k in kept])
        out["grpo/groups_per_step"] = float(len(groups))
        out["grpo/effective_groups_per_step"] = float(sum(1 for e in effective if e))

        # ---- 组内输出多样性（仅 group 内文本）----
        unique_ratios, dup_groups = [], []
        for s, e in _group_slices(n, G):
            texts = b.responses[s:e]
            if not texts:
                continue
            unique_ratios.append(len(set(texts)) / len(texts))
            dup_groups.append(len(set(texts)) < len(texts))
        _put(out, "grpo/unique_completion_ratio", _mean(unique_ratios))
        _put(out, "grpo/duplicate_completion_group_rate", _rate(dup_groups))
        len_groups = _group_values([float(x) for x in b.output_lengths], G)
        _put(out, "grpo/completion_length_group_std", _mean([_pstd(g) for g in len_groups]))

        # ---- advantage/* ----
        out.update(_advantage_block("advantage/raw", b.advantages_raw))
        out.update(_advantage_block("advantage/effective", b.advantages_effective))

        # ---- reward/（总奖励：raw vs final）----
        out.update(_reward_block("reward/raw", b.pre_gate_rewards, G))
        out.update(_reward_block("reward/final", b.final_rewards, G))
        _put(out, "reward/raw_group_std",
             _mean([_pstd(g) for g in _group_values(b.pre_gate_rewards, G)]))
        _put(out, "reward/final_group_std",
             _mean([_pstd(g) for g in _group_values(b.final_rewards, G)]))
        out["reward/raw_zero_std_group_rate"] = _rate(
            [s == 0.0 for s in [_pstd(g) for g in _group_values(b.pre_gate_rewards, G)]]
        )
        out["reward/final_zero_std_group_rate"] = _rate(
            [s == 0.0 for s in [_pstd(g) for g in _group_values(b.final_rewards, G)]]
        )

        # ---- gate 原因分布（分母 = 本步全部候选）----
        out.update(self._gate_reason_metrics(b))

        # ---- length/* ----
        out.update(self._length_metrics(b))

        # ---- behavior/* ----
        out.update(self._behavior_metrics(b))
        return out

    # ------------------------------------------------------------------
    def _gate_reason_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        reasons = list(b.gate_reasons)
        total = len(reasons) or 1

        def cat_rate(pred) -> float:
            return sum(1 for r in reasons if r and pred(r)) / total

        out = {
            "gate/total_rate": _rate([bool(r) for r in reasons]),
            "gate/fact_rate": cat_rate(lambda r: r == "fact_has_error"),
            "gate/length_too_long_rate": cat_rate(lambda r: r == "length_too_long"),
            "gate/length_too_short_rate": cat_rate(lambda r: r == "length_too_short"),
            "gate/min_length_rate": cat_rate(lambda r: r == "too_short"),
            "gate/no_result_marker_rate": cat_rate(lambda r: r == "no_result_marker"),
            "gate/forbidden_prefix_rate": cat_rate(lambda r: r.startswith("prefix:")),
        }
        known = sum(
            out[k] for k in (
                "gate/fact_rate", "gate/length_too_long_rate", "gate/length_too_short_rate",
                "gate/min_length_rate", "gate/no_result_marker_rate",
                "gate/forbidden_prefix_rate",
            )
        )
        out["gate/other_rate"] = max(0.0, out["gate/total_rate"] - known)
        return out

    def _length_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        lens = [float(x) for x in b.output_lengths]
        toks = [float(x) for x in b.token_lengths]
        _put(out, "length/output_mean", _mean(lens))
        _put(out, "length/output_std", _pstd(lens))
        _put(out, "length/output_p10", _percentile(lens, 0.10))
        _put(out, "length/output_p50", _percentile(lens, 0.50))
        _put(out, "length/output_p90", _percentile(lens, 0.90))
        _put(out, "length/output_max", max(lens) if lens else None)
        _put(out, "length/tokens_mean", _mean(toks))
        _put(out, "length/tokens_std", _pstd(toks))
        _put(out, "length/tokens_p90", _percentile(toks, 0.90))

        ratios = []
        for cand, ref in zip(b.responses, b.references):
            ref_len = max(len(ref.strip()), 1)
            ratios.append(len(cand.strip()) / ref_len)
        _put(out, "length/ratio_mean", _mean(ratios))
        _put(out, "length/ratio_std", _pstd(ratios))
        _put(out, "length/ratio_p50", _percentile(ratios, 0.50))
        _put(out, "length/ratio_p90", _percentile(ratios, 0.90))
        out["length/ratio_over_1_2_rate"] = _rate([r > 1.2 for r in ratios])
        out["length/ratio_over_1_5_rate"] = _rate([r > 1.5 for r in ratios])
        if b.upper_length_ratio is not None:
            floor = b.upper_length_ratio - self.cfg.near_upper_gate_margin
            out["length/near_upper_gate_rate"] = _rate([r >= floor for r in ratios])
        _put(out, "length/normalization_factor_mean", _mean(b.normalization_factors))
        _put(out, "length/normalization_factor_p90", _percentile(b.normalization_factors, 0.90))
        return out

    def _behavior_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        n = b.n or 1
        if b.max_new_tokens > 0:
            truncated = [t >= b.max_new_tokens for t in b.token_lengths]
            out["behavior/truncation_rate"] = _rate(truncated)
            out["behavior/eos_termination_rate"] = _rate([not t for t in truncated])
        out["behavior/result_marker_rate"] = _rate(
            [any(m in r for m in b.result_markers) for r in b.responses]
        )
        out["behavior/forbidden_prefix_rate"] = _rate([
            any(r.strip().startswith(p) for p in b.forbidden_prefixes) for r in b.responses
        ])
        ngram = self.cfg.repetition_ngram
        reps = [_repetition_rate(r, ngram) for r in b.responses]
        _put(out, "behavior/repetition_rate", _mean(reps))
        return out

    # ------------------------------------------------------------------
    # 二等块：reward 分项 + fact/coverage/ee + gate 对学习信号的影响
    # ------------------------------------------------------------------
    def _reward_detail_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        G = b.group_size
        # ---- reward_terms/* ----
        for name, values in b.term_values.items():
            if not values:
                continue
            prefix = f"reward_terms/{name}"
            _put(out, f"{prefix}/mean", _mean(values))
            _put(out, f"{prefix}/std", _pstd(values))
            _put(out, f"{prefix}/p10", _percentile(values, 0.10))
            _put(out, f"{prefix}/p50", _percentile(values, 0.50))
            _put(out, f"{prefix}/p90", _percentile(values, 0.90))
            group_stds = [_pstd(g) for g in _group_values(values, G)]
            _put(out, f"{prefix}/group_std", _mean(group_stds))
            out[f"{prefix}/nonzero_group_std_rate"] = _rate(
                [s > self.cfg.near_zero_std_eps for s in group_stds]
            )
            weight = b.term_weights.get(name, 0.0)
            out[f"{prefix}/weighted_mean"] = _mean(values) * weight

        if self.fact_enabled:
            out.update(self._fact_metrics(b))
        if self.coverage_enabled:
            out.update(self._coverage_metrics(b))
        if self.ee_enabled:
            out.update(self._ee_metrics(b))
        out.update(self._gate_learning_metrics(b))
        return out

    def _fact_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        G = b.group_size
        sig = b.raw_signals.get("fact_consistency", [None] * b.n)
        valid = [v for v in sig if v is not None]
        if valid:
            _put(out, "fact/mean", _mean(valid))
            _put(out, "fact/std", _pstd(valid))
            _put(out, "fact/p10", _percentile(valid, 0.10))
            _put(out, "fact/p50", _percentile(valid, 0.50))
            _put(out, "fact/p90", _percentile(valid, 0.90))
            out["fact/zero_rate"] = _rate([v == 0.0 for v in valid])
            stds = [_pstd([v for v in g if v is not None])
                    for g in _group_values(sig, G)]
            stds = [s for s, g in zip(stds, _group_values(sig, G))
                    if sum(1 for v in g if v is not None) >= 2]
            _put(out, "fact/group_std", _mean(stds))
        # 事实硬门控：只统计被 fact 门控拦下的比例
        fact_gated = [r == "fact_has_error" for r in b.gate_reasons]
        out["fact/gate_rate"] = _rate(fact_gated)
        out["fact/pass_rate"] = 1.0 - out["fact/gate_rate"]
        out.update(self._fact_score_metrics(b))
        return out

    def _fact_score_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        """事实一致性：整体 0-4 分 + 事实错误门控信号（不再有六要素明细）。"""
        out: Dict[str, float] = {}
        G = b.group_size
        results = b.judge_results or []
        if not results:
            return out
        scores: List[int] = []
        fact_pass: List[Optional[bool]] = [None] * b.n
        for i, res in enumerate(results[: b.n]):
            score = getattr(res, "fact_score", None)
            has_error = getattr(res, "has_factual_error", None)
            if score is None:
                continue
            scores.append(int(score))
            fact_pass[i] = not bool(has_error) if b.fact_gate_enabled else True
        if not scores:
            return out
        _put(out, "fact/score_mean", _mean([s / 4.0 for s in scores]))
        for level in range(5):
            out[f"fact/score_ratio_{level}"] = _rate([v == level for v in scores])
        errors = [1 if (getattr(r, "has_factual_error", None) or 0) else 0
                  for r in results[: b.n] if getattr(r, "fact_score", None) is not None]
        if errors:
            out["fact/error_rate"] = _rate([e == 1 for e in errors])
        # 组级 gate 通过情况（分母 = 有裁判数据的组）
        group_flags: List[List[bool]] = []
        for s, e in _group_slices(b.n, G):
            flags = [fact_pass[i] for i in range(s, e) if i < len(fact_pass) and fact_pass[i] is not None]
            if flags:
                group_flags.append(flags)
        if group_flags:
            out["fact/group_all_pass_rate"] = _rate([all(f) for f in group_flags])
            out["fact/group_all_fail_rate"] = _rate([not any(f) for f in group_flags])
            out["fact/group_mixed_gate_rate"] = _rate(
                [any(f) and not all(f) for f in group_flags]
            )
        return out

    def _coverage_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        G = b.group_size
        sig = b.raw_signals.get("element_coverage", [None] * b.n)
        valid = [v for v in sig if v is not None]
        if valid:
            _put(out, "coverage/mean", _mean(valid))
            _put(out, "coverage/std", _pstd(valid))
            _put(out, "coverage/p10", _percentile(valid, 0.10))
            _put(out, "coverage/p50", _percentile(valid, 0.50))
            _put(out, "coverage/p90", _percentile(valid, 0.90))
            out["coverage/zero_rate"] = _rate([v == 0.0 for v in valid])
            out["coverage/full_score_rate"] = _rate([v >= 1.0 - 1e-9 for v in valid])
            stds = [_pstd([v for v in g if v is not None]) for g in _group_values(sig, G)]
            stds = [s for s, g in zip(stds, _group_values(sig, G))
                    if sum(1 for v in g if v is not None) >= 2]
            _put(out, "coverage/group_std", _mean(stds))
            out["coverage/nonzero_group_std_rate"] = _rate(
                [s > self.cfg.near_zero_std_eps for s in stds]
            )
        out.update(self._coverage_element_metrics(b))
        return out

    def _coverage_element_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        results = b.judge_results or []
        if not results:
            return out
        per_elem: Dict[str, List[float]] = {name: [] for name in _ELEMENTS}
        # 逐要素按组聚合（组内候选连续）
        grouped: Dict[str, List[List[float]]] = {name: [] for name in _ELEMENTS}
        for s, e in _group_slices(b.n, b.group_size):
            for name in _ELEMENTS:
                chunk = []
                for i in range(s, min(e, len(results))):
                    res = results[i]
                    raw = getattr(res, "coverage_raw", None) if res is not None else None
                    if raw and name in raw:
                        chunk.append(int(raw[name]) / 4.0)
                grouped[name].append(chunk)
                per_elem[name].extend(chunk)
        for name in _ELEMENTS:
            vals = per_elem[name]
            if not vals:
                continue
            _put(out, f"coverage/{name}/mean", _mean(vals))
            # 覆盖率 prompt 的语义：参考摘要没有的要素判 4；因此 0 分必然意味着
            # "参考里有、候选完全没覆盖"。missing_rate 用这个确定性的 0 分率定义。
            out[f"coverage/{name}/missing_rate"] = _rate([v == 0.0 for v in vals])
            stds = [_pstd(g) for g in grouped[name] if len(g) >= 2]
            _put(out, f"coverage/{name}/group_std", _mean(stds))
        return out

    def _ee_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        G = b.group_size
        sig = b.raw_signals.get("expression_efficiency", [None] * b.n)
        valid = [v for v in sig if v is not None]
        if valid:
            _put(out, "ee/mean", _mean(valid))
            _put(out, "ee/std", _pstd(valid))
            _put(out, "ee/p10", _percentile(valid, 0.10))
            _put(out, "ee/p50", _percentile(valid, 0.50))
            _put(out, "ee/p90", _percentile(valid, 0.90))
            out["ee/low_score_rate"] = _rate([v <= self.cfg.ee_low_threshold for v in valid])
            out["ee/high_score_rate"] = _rate([v >= self.cfg.ee_high_threshold for v in valid])
            stds = [_pstd([v for v in g if v is not None]) for g in _group_values(sig, G)]
            stds = [s for s, g in zip(stds, _group_values(sig, G))
                    if sum(1 for v in g if v is not None) >= 2]
            _put(out, "ee/group_std", _mean(stds))
            out["ee/nonzero_group_std_rate"] = _rate(
                [s > self.cfg.near_zero_std_eps for s in stds]
            )
        # 三个子维度：项目现有 EE 是"越高越好"的效率评分（0-4），保留真实名称。
        results = b.judge_results or []
        dims: Dict[str, List[float]] = {name: [] for name in _EE_DIMENSIONS}
        dim_groups: Dict[str, List[List[float]]] = {name: [] for name in _EE_DIMENSIONS}
        for s, e in _group_slices(b.n, b.group_size):
            for name in _EE_DIMENSIONS:
                chunk = []
                for i in range(s, min(e, len(results))):
                    res = results[i]
                    raw = getattr(res, "ee_dims", None) if res is not None else None
                    if raw and name in raw:
                        chunk.append(int(raw[name]) / 4.0)
                dim_groups[name].append(chunk)
                dims[name].extend(chunk)
        for name in _EE_DIMENSIONS:
            vals = dims[name]
            if not vals:
                continue
            _put(out, f"ee/{name}_mean", _mean(vals))
            _put(out, f"ee/{name}_std", _pstd(vals))
            stds = [_pstd(g) for g in dim_groups[name] if len(g) >= 2]
            _put(out, f"ee/{name}_group_std", _mean(stds))
        return out

    def _gate_learning_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        """Gate 对组内学习信号的影响（监控用优势计算，不改真实训练）。"""
        out: Dict[str, float] = {}
        G = b.group_size
        groups_flags = list(_group_values([1.0 if f else 0.0 for f in b.gate_flags], G))
        out["gate/all_gated_group_rate"] = _rate([all(x == 1.0 for x in g) for g in groups_flags])
        out["gate/all_passed_group_rate"] = _rate([all(x == 0.0 for x in g) for g in groups_flags])
        out["gate/mixed_group_rate"] = _rate([
            any(x == 1.0 for x in g) and any(x == 0.0 for x in g) for g in groups_flags
        ])
        _put(out, "gate/group_std_before",
             _mean([_pstd(g) for g in _group_values(b.pre_gate_rewards, G)]))
        _put(out, "gate/group_std_after",
             _mean([_pstd(g) for g in _group_values(b.final_rewards, G)]))
        # gate 前后优势符号变化率（监控用，同一组内归一化规则）
        raw_adv = _group_advantages(b.pre_gate_rewards, G)
        fin_adv = _group_advantages(b.final_rewards, G)
        changed = [
            (a > 0) != (c > 0)
            for a, c in zip(raw_adv, fin_adv)
            if abs(a) > 1e-12 and abs(c) > 1e-12
        ]
        _put(out, "gate/advantage_sign_change_rate", _rate(changed))
        # Gate 与质量分项的关系
        cov = b.raw_signals.get("element_coverage", [None] * b.n)
        ee = b.raw_signals.get("expression_efficiency", [None] * b.n)
        _put(out, "gate/coverage_mean_gated", _cond_mean(cov, b.gate_flags, True))
        _put(out, "gate/coverage_mean_passed", _cond_mean(cov, b.gate_flags, False))
        _put(out, "gate/ee_mean_gated", _cond_mean(ee, b.gate_flags, True))
        _put(out, "gate/ee_mean_passed", _cond_mean(ee, b.gate_flags, False))
        return out

    # ------------------------------------------------------------------
    # 三等块：相关性 + Coverage/EE 冲突 + 分项消融
    # ------------------------------------------------------------------
    def _relation_metrics(self, b: MonitorBatch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        # 窗口相关性（分母 = 窗口内有成对数据的样本）
        pairs = {
            "relation/coverage_ee_spearman": ("coverage", "ee"),
            "relation/coverage_fact_spearman": ("coverage", "fact"),
            "relation/ee_fact_spearman": ("ee", "fact"),
            "relation/coverage_length_spearman": ("coverage", "length"),
            "relation/ee_length_spearman": ("ee", "length"),
            "relation/final_reward_length_spearman": ("final_reward", "length"),
            "relation/effective_signal_length_spearman": ("adv_eff", "length"),
        }
        for name, (kx, ky) in pairs.items():
            _put(out, name, self._window_corr(kx, ky))
        # 逐要素覆盖率与长度的关系（6.3 要求独立命名，值相同）
        _put(out, "coverage/length_spearman", self._window_corr("coverage", "length"))
        _put(out, "coverage/group_length_spearman",
             _mean_group_spearman(b, "coverage", "length"))
        _put(out, "ee/length_spearman", self._window_corr("ee", "length"))
        _put(out, "ee/group_length_spearman", _mean_group_spearman(b, "ee", "length"))
        # 组内相关性
        _put(out, "relation/group_coverage_ee_spearman", _mean_group_spearman(b, "coverage", "ee"))
        _put(out, "relation/group_coverage_length_spearman", _mean_group_spearman(b, "coverage", "length"))
        _put(out, "relation/group_ee_length_spearman", _mean_group_spearman(b, "ee", "length"))
        _put(out, "relation/group_reward_length_spearman", _mean_group_spearman(b, "final_reward", "length"))
        # Coverage / EE 冲突：组内相对高低
        out.update(_conflict_rates(b))
        # 分项消融
        out.update(_ablation_metrics(b))
        return out


# ---------------------------------------------------------------------------
# 辅助：优势/奖励块
# ---------------------------------------------------------------------------
def eval_metrics(
    rouge: Optional[Dict[str, float]],
    length_ratios: Sequence[float],
    signals: Dict[str, Sequence[Optional[float]]],
    final_rewards: Sequence[float],
    baseline: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """固定验证集指标（`eval/*`）。所有量都用同一份数据、同一套协议计算。

    `rouge` 来自 `sfzy.eval.rouge.score_corpus`（逐样本、固定分词）。
    与训练步一样：缺失的裁判信号被排除，不填 0。
    """
    out: Dict[str, float] = {}
    if rouge:
        _put(out, "eval/rouge_1", rouge.get("rouge-1-f"))
        _put(out, "eval/rouge_2", rouge.get("rouge-2-f"))
        _put(out, "eval/rouge_l", rouge.get("rouge-l-f"))
        _put(out, "eval/rouge_total", rouge.get("overall"))
    _put(out, "eval/length_ratio_mean", _mean(length_ratios))
    _put(out, "eval/length_ratio_p50", _percentile(length_ratios, 0.50))
    fact = [v for v in signals.get("fact_consistency", []) if v is not None]
    cov = [v for v in signals.get("element_coverage", []) if v is not None]
    ee = [v for v in signals.get("expression_efficiency", []) if v is not None]
    if fact:
        _put(out, "eval/fact_mean", _mean(fact))
        out["eval/fact_zero_rate"] = _rate([v == 0.0 for v in fact])
    _put(out, "eval/coverage_mean", _mean(cov) if cov else None)
    _put(out, "eval/ee_mean", _mean(ee) if ee else None)
    vals = [float(v) for v in final_rewards]
    _put(out, "eval/final_reward_mean", _mean(vals) if vals else None)
    # 相对 SFT 基线的变化（只在配置里给了基线时才记录）
    base = dict(baseline or {})
    if base:
        current = {
            "rouge_l": out.get("eval/rouge_l"),
            "coverage": out.get("eval/coverage_mean"),
            "ee": out.get("eval/ee_mean"),
            "fact_mean": out.get("eval/fact_mean"),
            "length_ratio": out.get("eval/length_ratio_mean"),
        }
        for key, name in (
            ("rouge_l", "eval/delta_rouge_l"),
            ("coverage", "eval/delta_coverage"),
            ("ee", "eval/delta_ee"),
            ("fact_mean", "eval/delta_fact_mean"),
            ("length_ratio", "eval/delta_length_ratio"),
        ):
            if key in base and current.get(key) is not None:
                out[name] = current[key] - float(base[key])
    return out


def _positive_ratio(values: Sequence[float]) -> float:
    return _rate([v > 0 for v in values])


def _negative_ratio(values: Sequence[float]) -> float:
    return _rate([v < 0 for v in values])


def _zero_ratio(values: Sequence[float]) -> float:
    return _rate([v == 0 for v in values])


def _advantage_block(prefix: str, values: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    values = [float(v) for v in values]
    _put(out, f"{prefix}_mean", _mean(values))
    _put(out, f"{prefix}_std", _pstd(values))
    _put(out, f"{prefix}_min", min(values) if values else None)
    _put(out, f"{prefix}_max", max(values) if values else None)
    absv = [abs(v) for v in values]
    _put(out, f"{prefix}_abs_mean", _mean(absv))
    _put(out, f"{prefix}_abs_p90", _percentile(absv, 0.90) if absv else None)
    out[f"{prefix}_positive_ratio"] = _positive_ratio(values)
    out[f"{prefix}_negative_ratio"] = _negative_ratio(values)
    out[f"{prefix}_zero_ratio"] = _zero_ratio(values)
    return out


def _reward_block(prefix: str, values: Sequence[float], group_size: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    values = [float(v) for v in values]
    _put(out, f"{prefix}_mean", _mean(values))
    _put(out, f"{prefix}_std", _pstd(values))
    _put(out, f"{prefix}_min", min(values) if values else None)
    _put(out, f"{prefix}_max", max(values) if values else None)
    _put(out, f"{prefix}_p10", _percentile(values, 0.10))
    _put(out, f"{prefix}_p50", _percentile(values, 0.50))
    _put(out, f"{prefix}_p90", _percentile(values, 0.90))
    out[f"{prefix}_zero_rate"] = _zero_ratio(values)
    return out


def _group_advantages(values: Sequence[float], group_size: int) -> List[float]:
    """监控用的组内归一化优势，口径与 `grpo.group_advantages` 完全一致。"""
    out: List[float] = []
    for g in _group_values(values, group_size):
        mu = _mean(g)
        sd = _pstd(g)
        out.extend((v - mu) / (sd + 1e-4) for v in g)
    return out


def _repetition_rate(text: str, n: int) -> float:
    tokens = list(text)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


_ELEMENTS = (
    "case_type", "plaintiff_claims", "defendant_defenses",
    "court_facts", "legal_basis", "judgment_result",
)
_EE_DIMENSIONS = ("semantic_redundancy", "verbal_redundancy", "abstraction_adequacy")


def _cond_mean(values: Sequence[Optional[float]], flags: Sequence[bool], want: bool) -> Optional[float]:
    picked = [v for v, f in zip(values, flags) if f is want and v is not None]
    return _mean(picked) if picked else None


def _mean_group_spearman(b: MonitorBatch, key_x: str, key_y: str) -> Optional[float]:
    """组内相关性的组均值：只对有定义（≥3 有效且非恒定）的组求平均。"""
    values = _record_values(b, key_x)
    values_y = _record_values(b, key_y)
    corrs = []
    for s, e in _group_slices(b.n, b.group_size):
        xs = [values[i] for i in range(s, e) if values[i] is not None and values_y[i] is not None]
        ys = [values_y[i] for i in range(s, e) if values[i] is not None and values_y[i] is not None]
        c = _spearman(xs, ys)
        if c is not None:
            corrs.append(c)
    return _mean(corrs) if corrs else None


def _record_values(b: MonitorBatch, key: str) -> List[Optional[float]]:
    if key == "final_reward":
        return [float(v) for v in b.final_rewards]
    if key == "length":
        return [float(v) for v in b.output_lengths]
    if key in ("coverage", "ee", "fact"):
        return list(b.raw_signals.get(
            {"coverage": "element_coverage", "ee": "expression_efficiency",
             "fact": "fact_consistency"}[key], [None] * b.n
        ))
    if key == "adv_eff":
        return [float(v) for v in b.advantages_effective]
    return [None] * b.n


def _conflict_rates(b: MonitorBatch) -> Dict[str, float]:
    """Coverage / EE 组内相对高低组合比例（平局不计入高/低）。"""
    cov = _record_values(b, "coverage")
    ee = _record_values(b, "ee")
    high_cov_low_ee = low_cov_high_ee = high_cov_high_ee = low_cov_low_ee = 0
    denom = 0
    for s, e in _group_slices(b.n, b.group_size):
        c = [cov[i] for i in range(s, e)]
        d = [ee[i] for i in range(s, e)]
        c_ok = [x for x in c if x is not None]
        d_ok = [x for x in d if x is not None]
        if len(c_ok) < 2 or len(d_ok) < 2:
            continue
        c_med = _percentile(c_ok, 0.5)
        d_med = _percentile(d_ok, 0.5)
        for i in range(s, e):
            if cov[i] is None or ee[i] is None:
                continue
            ch = cov[i] > c_med
            cl = cov[i] < c_med
            eh = ee[i] > d_med
            el = ee[i] < d_med
            if not (ch or cl) or not (eh or el):
                continue
            denom += 1
            if ch and el:
                high_cov_low_ee += 1
            elif cl and eh:
                low_cov_high_ee += 1
            elif ch and eh:
                high_cov_high_ee += 1
            elif cl and el:
                low_cov_low_ee += 1
    if denom == 0:
        return {}
    return {
        "relation/high_coverage_low_ee_rate": high_cov_low_ee / denom,
        "relation/low_coverage_high_ee_rate": low_cov_high_ee / denom,
        "relation/high_coverage_high_ee_rate": high_cov_high_ee / denom,
        "relation/low_coverage_low_ee_rate": low_cov_low_ee / denom,
    }


def _ablation_metrics(b: MonitorBatch) -> Dict[str, float]:
    """分项消融：只构造假设性 Reward，不改变真实训练。"""
    out: Dict[str, float] = {}
    G = b.group_size
    soft = [name for name in ("fact_consistency", "element_coverage", "expression_efficiency")
            if name in b.term_values and name in b.term_weights and b.term_weights[name] > 0]
    if not soft:
        return out
    gated = list(b.gate_flags)
    orig = [float(v) for v in b.final_rewards]
    orig_adv = _group_advantages(orig, G)
    # 只在原始组内存在方差的组上比较（否则没有 top1/优势方向可言）
    group_has_var = [(_pstd(g) > 1e-12) for g in _group_values(orig, G)]
    label = {
        "fact_consistency": "fact_soft",
        "element_coverage": "coverage",
        "expression_efficiency": "ee",
    }
    for term in soft:
        remaining = {k: w for k, w in b.term_weights.items() if k != term}
        total_w = sum(remaining.values())
        if total_w <= 0:
            continue
        ablated = []
        for i in range(b.n):
            if gated[i]:
                ablated.append(0.0)
                continue
            score = sum(b.term_values[k][i] * w for k, w in remaining.items()) / total_w
            ablated.append(score)
        abl_adv = _group_advantages(ablated, G)
        # top1 flip：组内最优集合是否变化（并列最高算同一集合）
        flips, sign_flips, changes, sign_total = [], 0, 0.0, 0
        for gi, (s, e) in enumerate(_group_slices(b.n, G)):
            if not group_has_var[gi]:
                continue
            orig_max = max(orig[s:e])
            abl_max = max(ablated[s:e])
            orig_top = {i for i in range(s, e) if orig[i] == orig_max}
            abl_top = {i for i in range(s, e) if ablated[i] == abl_max}
            flips.append(orig_top != abl_top)
            for i in range(s, e):
                changes += abs(abl_adv[i] - orig_adv[i])
                if abs(orig_adv[i]) > 1e-12 and abs(abl_adv[i]) > 1e-12:
                    sign_total += 1
                    if (orig_adv[i] > 0) != (abl_adv[i] > 0):
                        sign_flips += 1
        name = label[term]
        _put(out, f"relation/{name}_top1_flip_rate", _rate(flips) if flips else None)
        _put(out, f"relation/{name}_advantage_sign_flip_rate",
             sign_flips / sign_total if sign_total else None)
        _put(out, f"relation/{name}_advantage_change_mean",
             changes / b.n if b.n else None)
    return out
