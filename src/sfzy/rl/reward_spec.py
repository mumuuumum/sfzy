"""奖励规格：把 YAML 里的 `rl.reward` 解析成可校验、可组合的 RewardSpec。

============================ 这一层解决什么 ============================
`terms` 是唯一的事实来源：某个 reward 开不开、权重多少、要不要裁判，
全部在这里定下来，并在**启动时**一次性校验完。校验放在这里而不是散在
trainer 里，是因为"配置说要接裁判却没有裁判"这类错误必须当场报错 ——
沉默地跑出一条其实没接裁判的曲线，是最贵的 bug。

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
          rouge_l:          { enabled: true,  weight: 0.3 }
          fact_consistency: { enabled: true,  weight: 0.7 }
          # 以后新增的项直接加在这里，聚合/训练/日志都不用改

老配置的 `mode: fact_judge` 作为**预设**保留：它等价于一组 terms，
方便不想改历史的实验继续跑。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set

from sfzy.rl.reward_terms import TERM_REGISTRY, TermSpec, get_term

# 门控默认值。`enabled=False` 时整块不生效（连 min_chars 都不看）。
DEFAULT_GATE: Dict[str, Any] = {
    "min_chars": 60,
    "length_ratio_range": [0.5, 1.5],
    "forbidden_prefixes": ["以下是", "摘要：", "摘要:", "本摘要", "这是"],
    "require_result_marker": True,
}

# 预设：一个名字展开成一组 terms。只为兼容老配置，新配置请直接写 terms。
PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "fact_judge": {
        "rouge_l": {"enabled": True, "weight": 0.3},
        "fact_consistency": {"enabled": True, "weight": 0.7},
    },
}


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

    @property
    def spec(self) -> TermSpec:
        return get_term(self.name)

    @property
    def source(self) -> str:
        return self.spec.source


@dataclass
class RewardSpec:
    terms: Dict[str, TermConfig]
    gate: GateSpec = field(default_factory=GateSpec)
    normalize_weights: bool = True
    rouge_mode: str = "jieba"

    # ---------------------------------------------------------------- 解析
    @classmethod
    def from_config(cls, raw: Optional[Mapping[str, Any]]) -> "RewardSpec":
        raw = dict(raw or {})

        terms_raw = raw.get("terms")
        if terms_raw is None:
            # 老配置：用 mode/preset 展开
            preset_name = raw.get("mode", "fact_judge")
            if preset_name not in PRESETS:
                raise RewardConfigError(
                    f"未知的 reward.mode/preset：{preset_name!r}（可用：{sorted(PRESETS)}）"
                )
            terms_raw = PRESETS[preset_name]
        if not isinstance(terms_raw, Mapping):
            raise RewardConfigError(
                "rl.reward.terms 必须是 {名字: {enabled, weight}} 形式的映射"
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
            term_spec = TERM_REGISTRY[name]
            enabled = bool(entry.get("enabled", True))
            weight = float(entry.get("weight", term_spec.default_weight))
            if weight < 0:
                raise RewardConfigError(
                    f"reward term {name!r} 的权重不能为负：{weight}"
                )
            terms[name] = TermConfig(name=name, enabled=enabled, weight=weight)

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

    def normalized_weights(self) -> Dict[str, float]:
        """启用项的权重。`normalize_weights=True` 时归一到和为 1。"""
        enabled = self.enabled_terms
        total = sum(t.weight for t in enabled)
        if not self.normalize_weights or total <= 0:
            return {t.name: t.weight for t in enabled}
        return {t.name: t.weight / total for t in enabled}

    @property
    def required_signals(self) -> Set[str]:
        """启用项里，需要模型 B 产出的信号名集合。"""
        out: Set[str] = set()
        for term in self.enabled_terms:
            if term.source == "judge" and term.spec.signal:
                out.add(term.spec.signal)
        return out

    def needs_judge(self) -> bool:
        return bool(self.required_signals)
