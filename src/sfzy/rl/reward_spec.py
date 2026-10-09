"""奖励规格：把 YAML 里的 `rl.reward` 解析成可校验、可组合的 RewardSpec。

============================ 权重只能来自配置文件 ============================
这个模块有一条硬规矩：**所有权重都只能写在配置文件里**。

  * 每个 reward 的权重      → `terms.<名字>.weight`
  * reward 内部的分项权重    → `terms.<名字>.<内部字段>`，例如
                              `terms.fact_consistency.element_weights`

代码里不存权重数值（注册表只声明"有哪些子项"），也没有命令行开关。
缺了就在**启动时**报错并告诉你该往哪一段写什么，而不是悄悄用一个藏在源码里
的默认值 —— 那种默认值是"我明明改了权重，曲线却一点没变"这类事故的源头。

============================ 配置形状 ============================
    rl:
      reward:
        normalize_weights: true        # 按启用项把权重归一到 1
        gate:                          # 全局硬门控，可整块关掉
          enabled: true
          min_chars: 60
          length_ratio_range: [0.5, 1.5]
          require_result_marker: true
        terms:                         # 想开几个开几个
          rouge_l:          {enabled: true,  weight: 0.3}
          fact_consistency:
            enabled: true
            weight: 0.7
            element_weights:           # ★ 这个 reward 内部的六要素权重
              case_type:          0.05
              plaintiff_claims:   0.15
              defendant_defenses: 0.10
              court_facts:        0.25
              legal_basis:        0.15
              judgment_result:    0.30

被启用的 term 必须写全 `weight`；有内部权重的 term 还必须写全那个字段
（键齐全、非负、之和为 1）。关掉的 term 不要求写，写了也不生效。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from sfzy.rl.reward_terms import TERM_REGISTRY, TermSpec, get_term

# 门控默认值。门控是筛选条件、不是奖励权重，所以这里保留默认。
# `enabled=False` 时整块不生效（连 min_chars 都不看）。
DEFAULT_GATE: Dict[str, Any] = {
    "min_chars": 60,
    "length_ratio_range": [0.5, 1.5],
    "forbidden_prefixes": ["以下是", "摘要：", "摘要:", "本摘要", "这是"],
    "require_result_marker": True,
    # 事实一致性硬门控：六个要素里只要有一个**原始分 < 这个值**，整条 reward
    # 直接 0（不再算覆盖率/ROUGE）。1 就是需求里的"任一个要素得 0 分即出局"；
    # 设 0 关闭这条规则。只在 fact_consistency 这个 term 被启用时生效。
    "fact_consistency_min_raw": 1,
}

# 事实一致性硬门控要读的裁判信号名：六个要素原始分（0-4）的最小值。
# judge.py / scorer.py 里也硬编码了同一个字符串 —— judge 层不 import 本模块，
# 否则会把 torch 拉进纯 CPU 的 reward 路径。两边一致性由 tests 钉住。
FACT_GATE_SIGNAL = "fact_consistency_min_raw"

# term 配置里除了内部权重字段、只能写这两个保留字段
_RESERVED_FIELDS = ("enabled", "weight")


class RewardConfigError(ValueError):
    """奖励配置有问题，启动时就该停下。"""


@dataclass
class GateSpec:
    enabled: bool = True
    cfg: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_GATE))


@dataclass
class TermConfig:
    name: str
    enabled: bool
    weight: float
    # 内部权重等（字段名 → 值），例如 {"element_weights": {...}}
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> TermSpec:
        return get_term(self.name)

    @property
    def source(self) -> str:
        return self.spec.source


def _as_float(value: Any, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RewardConfigError(f"{where} 必须是数字，收到 {value!r}") from exc


def _validate_internal_weights(
    term: str, field_name: str, required: Sequence[str], raw: Any
) -> Dict[str, float]:
    """校验一个 reward 的内部权重：键齐全、非负、之和为 1。"""
    if not isinstance(raw, Mapping):
        raise RewardConfigError(
            f"terms.{term}.{field_name} 必须是 {{键: 权重}} 形式的映射，收到 {raw!r}"
        )
    missing = [k for k in required if k not in raw]
    unknown = [k for k in raw if k not in required]
    if missing or unknown:
        raise RewardConfigError(
            f"terms.{term}.{field_name} 的键不对。\n"
            f"  缺少：{missing}\n"
            f"  多余：{unknown}\n"
            f"  必须且只能写：{list(required)}"
        )
    weights: Dict[str, float] = {}
    for key, value in raw.items():
        weight = _as_float(value, f"terms.{term}.{field_name}.{key}")
        if weight < 0:
            raise RewardConfigError(
                f"terms.{term}.{field_name}.{key} 不能为负：{weight}"
            )
        weights[key] = weight
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise RewardConfigError(
            f"terms.{term}.{field_name} 的权重之和必须为 1，当前是 {total:.6f}。"
        )
    return weights


def _validate_flexible_weights(
    term: str, field_name: str, required: Sequence[str], raw: Any
) -> Dict[str, float]:
    """校验"非负且和 > 0"的内部权重（如 EE 的三维权重）。

    和 `_validate_internal_weights` 的区别：**不要求和为 1**。EE 的 R_EE 公式
    自带按权重和归一，所以 0.4/0.2/0.4 和 4/2/4 等价。
    """
    if not isinstance(raw, Mapping):
        raise RewardConfigError(
            f"terms.{term}.{field_name} 必须是 {{键: 权重}} 形式的映射，收到 {raw!r}"
        )
    missing = [k for k in required if k not in raw]
    unknown = [k for k in raw if k not in required]
    if missing or unknown:
        raise RewardConfigError(
            f"terms.{term}.{field_name} 的键不对。\n"
            f"  缺少：{missing}\n"
            f"  多余：{unknown}\n"
            f"  必须且只能写：{list(required)}"
        )
    weights: Dict[str, float] = {}
    for key, value in raw.items():
        weight = _as_float(value, f"terms.{term}.{field_name}.{key}")
        if weight < 0:
            raise RewardConfigError(
                f"terms.{term}.{field_name}.{key} 不能为负：{weight}"
            )
        weights[key] = weight
    total = sum(weights.values())
    if total <= 0:
        raise RewardConfigError(
            f"terms.{term}.{field_name} 的权重之和必须大于 0，当前是 {total:.6f}。"
        )
    return weights


@dataclass
class RewardSpec:
    terms: Dict[str, TermConfig]
    gate: GateSpec = field(default_factory=GateSpec)
    normalize_weights: bool = True
    rouge_mode: str = "jieba"

    # ---------------------------------------------------------------- 解析
    @classmethod
    def from_config(cls, raw: Optional[Mapping[str, Any]]) -> "RewardSpec":
        if raw is None:
            raise RewardConfigError(
                "配置里没有 rl.reward：每个 reward 的开关和权重都要写在配置文件里。"
            )
        raw = dict(raw)

        terms_raw = raw.get("terms")
        if terms_raw is None:
            raise RewardConfigError(
                "rl.reward 里必须写 terms —— 每个 reward 的开关和权重都在这一段。"
            )
        if not isinstance(terms_raw, Mapping):
            raise RewardConfigError(
                "rl.reward.terms 必须是 {名字: {enabled, weight, ...}} 形式的映射"
            )

        terms: Dict[str, TermConfig] = {}
        for name, entry in terms_raw.items():
            if name not in TERM_REGISTRY:
                raise RewardConfigError(
                    f"未知的 reward term：{name!r}（可用：{sorted(TERM_REGISTRY)}）"
                )
            if entry is None:
                entry = {}
            if not isinstance(entry, Mapping):
                raise RewardConfigError(
                    f"reward term {name!r} 的配置必须是映射，收到 {entry!r}"
                )
            entry = dict(entry)
            term_spec = TERM_REGISTRY[name]
            enabled = bool(entry.get("enabled", True))

            allowed = (
                set(_RESERVED_FIELDS)
                | set(term_spec.internal_weight_fields)
                | set(term_spec.flexible_weight_fields)
                | set(term_spec.scalar_options)
            )
            unknown = sorted(set(entry) - allowed)
            if unknown:
                raise RewardConfigError(
                    f"terms.{name} 里有不认识的字段 {unknown}；可以写 {sorted(allowed)}"
                )

            # ---- 这个 reward 的权重：启用就必须写 ----
            if "weight" in entry:
                weight = _as_float(entry["weight"], f"terms.{name}.weight")
            elif enabled:
                raise RewardConfigError(
                    f"terms.{name} 被启用，但没有写 weight。\n"
                    f"  每个 reward 的权重只能写在配置文件里，"
                    f"加上一行 {name}: {{enabled: true, weight: <数值>}}。"
                )
            else:
                weight = 0.0
            if weight < 0:
                raise RewardConfigError(f"terms.{name}.weight 不能为负：{weight}")

            # ---- 这个 reward 内部的权重：同样只能来自配置 ----
            options: Dict[str, Any] = {}
            for field_name, required_keys in term_spec.internal_weight_fields.items():
                if field_name in entry:
                    options[field_name] = _validate_internal_weights(
                        name, field_name, required_keys, entry[field_name]
                    )
                elif enabled:
                    raise RewardConfigError(
                        f"terms.{name} 被启用，但没有写 {field_name}。\n"
                        f"  {name} 的内部权重只能写在配置文件的 "
                        f"terms.{name}.{field_name} 下，\n"
                        f"  键必须且只能是 {list(required_keys)}，权重之和必须为 1。"
                    )
            # 弹性内部权重（如 EE 的三维权重）：可省 → 用代码默认；写就必须写全，
            # 只校验非负 + 和 > 0。
            for field_name, required_keys in term_spec.flexible_weight_fields.items():
                if field_name in entry:
                    options[field_name] = _validate_flexible_weights(
                        name, field_name, required_keys, entry[field_name]
                    )
                else:
                    default = term_spec.default_weight_fields.get(field_name)
                    if default is not None:
                        options[field_name] = dict(default)
                    elif enabled:
                        raise RewardConfigError(
                            f"terms.{name} 被启用，但没有写 {field_name}，也没有默认值。"
                        )
            # 组合项/规则项的标量参数（如 quality_fbeta 的 beta / eps），可选
            for field_name in term_spec.scalar_options:
                if field_name in entry:
                    options[field_name] = _as_float(
                        entry[field_name], f"terms.{name}.{field_name}"
                    )
            terms[name] = TermConfig(
                name=name, enabled=enabled, weight=weight, options=options
            )

        enabled = [t for t in terms.values() if t.enabled]
        if not enabled:
            raise RewardConfigError("至少要有一个 enabled 的 reward term")
        if all(t.weight <= 0 for t in enabled):
            raise RewardConfigError("enabled 的 reward term 权重全是 0，等于没有奖励")

        gate_raw = dict(raw.get("gate") or {})
        gate_enabled = bool(gate_raw.pop("enabled", True))
        gate_cfg = copy.deepcopy(DEFAULT_GATE)
        gate_cfg.update(gate_raw)

        return cls(
            terms=terms,
            gate=GateSpec(enabled=gate_enabled, cfg=gate_cfg),
            normalize_weights=bool(raw.get("normalize_weights", True)),
            rouge_mode=raw.get("rouge_mode", "jieba"),
        )

    # ---------------------------------------------------------------- 派生
    @property
    def enabled_terms(self) -> List[TermConfig]:
        return [t for t in self.terms.values() if t.enabled]

    @property
    def consumed_terms(self) -> Set[str]:
        """被组合项（如 quality_fbeta）吞并的项名 —— 它们不再参与独立加权和。"""
        out: Set[str] = set()
        for term in self.enabled_terms:
            if term.source == "composite":
                out.update(term.spec.consumes)
        return out

    @property
    def active_terms(self) -> List[TermConfig]:
        """真正参与加权和的项 = 启用项 - 被组合项吞并的项。"""
        consumed = self.consumed_terms
        return [t for t in self.enabled_terms if t.name not in consumed]

    def normalized_weights(self) -> Dict[str, float]:
        """参与加权和的项的权重。`normalize_weights=True` 时归一到和为 1。"""
        enabled = self.active_terms
        total = sum(t.weight for t in enabled)
        if not self.normalize_weights or total <= 0:
            return {t.name: t.weight for t in enabled}
        return {t.name: t.weight / total for t in enabled}

    @property
    def required_signals(self) -> Set[str]:
        """启用项里，需要模型 B 产出的**任务**信号名集合。"""
        out: Set[str] = set()
        for term in self.enabled_terms:
            if term.source == "judge" and term.spec.signal:
                out.add(term.spec.signal)
            # 组合项依赖的信号（如 quality_fbeta 需要 coverage 与 INP）也要取回来
            if term.source == "composite":
                out.update(term.spec.depends_on)
        return out

    @property
    def gate_signals(self) -> Set[str]:
        """门控（不是 reward term）额外需要的裁判信号。"""
        out: Set[str] = set()
        if self.fact_element_gate_enabled:
            out.add(FACT_GATE_SIGNAL)
        return out

    @property
    def needed_signals(self) -> Set[str]:
        """要从裁判取回来的**全部**信号 = 任务信号 + 门控辅助信号。"""
        return self.required_signals | self.gate_signals

    def needs_judge(self) -> bool:
        return bool(self.needed_signals)

    # ---------------------------------------------------------------- 事实门控
    @property
    def fact_consistency_enabled(self) -> bool:
        term = self.terms.get("fact_consistency")
        return bool(term and term.enabled)

    @property
    def fact_element_min_raw(self) -> int:
        """六个要素原始分的最低要求；<=0 表示不做事实一致性硬门控。"""
        if not self.gate.enabled:
            return 0
        try:
            return int(self.gate.cfg.get("fact_consistency_min_raw", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def fact_element_gate_enabled(self) -> bool:
        return self.fact_consistency_enabled and self.fact_element_min_raw > 0

    def judge_term_options(self) -> Dict[str, Dict[str, Any]]:
        """{裁判信号名: 该 reward 的内部权重}，交给裁判后端构造时使用。"""
        out: Dict[str, Dict[str, Any]] = {}
        for term in self.enabled_terms:
            if term.source == "judge" and term.spec.signal:
                # 即使没有内部权重（如 INP）也要带上，调用方靠这些键决定
                # 要跑哪些 judge task。
                out[term.spec.signal] = dict(term.options)
        return out
