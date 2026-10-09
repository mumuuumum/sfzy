"""奖励项注册表：每个 term 声明自己是 rule 还是 judge、对应哪个信号。

============================ 为什么要有注册表 ============================
奖励项和被优化的东西是同一份代码里最容易悄悄长歪的部分：加一个 reward、
改一个权重、把某个项关掉，都不该动到聚合、trainer 和日志。

注册表把"有哪些 reward、它们从哪来"集中成一张表：

    source = "rule"   进程内纯函数，不需要模型
    source = "judge"  消费模型 B 产出的一个**命名信号**（signal）

============================ 注册表里没有权重 ============================
这里**只声明结构，不存任何权重数值**：某个 reward 的权重、以及它内部分项
的权重，全部只能写在配置文件里（见 `reward_spec.py`）。注册表最多声明
"这个 reward 有哪些内部分项"（键名），数值一律由 YAML 给。

新增一个 reward：
  * 规则项 —— 写一个函数，加进 `RULE_FUNCS`，再注册一条 `TermSpec`
  * 裁判项 —— 在 `sfzy/judge/` 里加一个 task 并声明信号名，再注册一条 `TermSpec`
两种情况都不需要改 `reward.py` 的聚合逻辑。

============================ 量纲约定 ============================
所有 term 的取值都必须是 **[0,1]**。权重的归一化交给 `reward_spec.py`，
这里只保证单项在统一量纲上，否则"加权求和"会变成"谁的量纲大谁说了算"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from sfzy.eval.rouge import score_pair

# 六要素的键名。
#
# 为什么不直接 import `sfzy.judge.schema.ELEMENTS`：那条 import 会一路把
# torch / transformers 拉进来，而 `rl/reward` 这条路径（以及它的纯 CPU 测试）
# 必须保持轻量。两边的一致性由 tests/test_judge.py 的一条断言钉住。
FACT_ELEMENTS: Tuple[str, ...] = (
    "case_type",
    "plaintiff_claims",
    "defendant_defenses",
    "court_facts",
    "legal_basis",
    "judgment_result",
)

# Expression Efficiency 的三个维度。和 judge/schema.py 的 EE_DIMENSIONS 必须一致
# （由 tests/test_judge.py 的一致性断言钉住）。这里同样抄一份，避免把 torch 拉进来。
EE_DIMENSIONS: Tuple[str, ...] = (
    "semantic_redundancy",
    "verbal_redundancy",
    "abstraction_adequacy",
)

# 三个维度的默认内部权重（非负、和 > 0 即可；R_EE 公式自带按权重和归一）。
EE_DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic_redundancy": 0.4,
    "verbal_redundancy": 0.2,
    "abstraction_adequacy": 0.4,
}


@dataclass(frozen=True)
class TermSpec:
    """一个奖励项的静态声明。**不含权重数值。**

    `internal_weight_fields` 声明这个 reward 内部的子权重有哪些字段、
    每个字段必须覆盖哪些键。例如事实一致性：

        {"element_weights": FACT_ELEMENTS}

    意思是配置文件里要写

        terms.fact_consistency.element_weights: {六个要素: 权重}

    数值由配置给，代码只认结构，并在启动时校验（齐全 + 和为 1）。
    """

    name: str
    source: str                     # "rule" | "judge" | "composite"
    signal: Optional[str] = None    # judge 项消费的信号名；rule 项为 None
    description: str = ""
    internal_weight_fields: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # 与 internal_weight_fields 的区别：这里只要求**非负且和 > 0**，不强制和为 1
    # （用于 EE 这类公式自带按权重和归一的分项）。缺省时用 default_weight_fields。
    flexible_weight_fields: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    default_weight_fields: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 组合项（composite）：依赖哪些信号、把哪些项"吞并"（被吞并的项不再参与独立加权和），
    # 以及除权重外的标量参数（如 beta / eps）。
    depends_on: Tuple[str, ...] = ()
    consumes: Tuple[str, ...] = ()
    scalar_options: Tuple[str, ...] = ()


TERM_REGISTRY: Dict[str, TermSpec] = {
    "rouge_l": TermSpec(
        name="rouge_l",
        source="rule",
        description="ROUGE-L（jieba 分词口径），和官方评测对齐的锚",
    ),
    "fact_consistency": TermSpec(
        name="fact_consistency",
        source="judge",
        signal="fact_consistency",
        description="六要素事实一致性 Judge 加权分 ∈ [0,1]",
        # 六要素的**内部权重**只能写在配置里，这里只声明键集合。
        internal_weight_fields={"element_weights": FACT_ELEMENTS},
    ),
    "element_coverage": TermSpec(
        name="element_coverage",
        source="judge",
        signal="element_coverage",
        description="关键要素覆盖率：候选摘要对人工摘要六要素的覆盖程度 ∈ [0,1]",
        # 同样六个要素，但权重可以单独设（两者共用一个 judge 后端，
        # 抽取共用，判定各一套 prompt）。
        internal_weight_fields={"element_weights": FACT_ELEMENTS},
    ),
    "information_necessity_precision": TermSpec(
        name="information_necessity_precision",
        source="judge",
        signal="information_necessity_precision",
        description="信息必要性精确率 INP：候选原子命题的必要性得分均值 ∈ [0,1]",
        # 按原子命题算，没有六要素内部权重。
    ),
    "expression_efficiency": TermSpec(
        name="expression_efficiency",
        source="judge",
        signal="expression_efficiency",
        description="表达效率 EE：语义冗余/表达冗余/抽象适当性三维加权分 ∈ [0,1]",
        # 三个维度权重可配置，缺省用默认；只要求非负、和 > 0。
        flexible_weight_fields={"dimension_weights": EE_DIMENSIONS},
        default_weight_fields={"dimension_weights": EE_DEFAULT_WEIGHTS},
    ),
    "quality_fbeta": TermSpec(
        name="quality_fbeta",
        source="composite",
        description="Coverage 与 INP 的 Fβ 组合 ∈ [0,1]（启用后 coverage/INP 不再独立加权）",
        depends_on=("element_coverage", "information_necessity_precision"),
        consumes=("element_coverage", "information_necessity_precision"),
        scalar_options=("beta", "eps"),
    ),
}


def known_terms() -> list[str]:
    return sorted(TERM_REGISTRY)


def get_term(name: str) -> TermSpec:
    try:
        return TERM_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"未知的 reward term：{name!r}（可用：{known_terms()}）"
        ) from exc


# --------------------------------------------------------------------------
# 规则项的实现
# --------------------------------------------------------------------------
def rouge_l(
    candidate: str, reference: str, source: Optional[str] = None, rouge_mode: str = "jieba"
) -> float:
    """ROUGE-L F1 ∈ [0,1]。`rouge_mode` 决定分词口径（中文用 jieba）。"""
    return float(
        score_pair(candidate.strip(), reference.strip(), mode=rouge_mode)["rouge-l-f"]
    )


RULE_FUNCS = {
    "rouge_l": rouge_l,
}


def compute_rule_term(name: str, candidate: str, reference: str, source: Optional[str], **opts: Any) -> float:
    """调一个规则项。未知/非规则项直接报错，不静默返回 0。"""
    fn = RULE_FUNCS.get(name)
    if fn is None:
        raise KeyError(f"{name!r} 不是规则项（没有注册对应的 RULE_FUNCS 实现）")
    return float(fn(candidate, reference, source=source, **opts))


# --------------------------------------------------------------------------
# 组合项：Coverage × INP 的 Fβ
# --------------------------------------------------------------------------
def fbeta_score(
    coverage: Optional[float],
    inp: Optional[float],
    beta: float = 1.0,
    eps: float = 1e-9,
) -> float:
    """`Fβ = (1+β²)·C·INP / (β²·INP + C + ε)`，∈[0,1]。

    `β>1` 偏重 Coverage，`β<1` 偏重 INP，`β=1` 是标准 F1。
    C=INP=0 直接返回 0；任一信号缺失按 0 处理。
    """
    c = float(coverage) if coverage is not None else 0.0
    i = float(inp) if inp is not None else 0.0
    c = min(1.0, max(0.0, c))
    i = min(1.0, max(0.0, i))
    if c <= 0.0 and i <= 0.0:
        return 0.0
    b2 = float(beta) ** 2
    denom = b2 * i + c + float(eps)
    if denom <= 0.0:
        return 0.0
    return min(1.0, max(0.0, (1.0 + b2) * c * i / denom))


COMPOSITE_FUNCS = {
    "quality_fbeta": lambda signals, opts: fbeta_score(
        signals.get("element_coverage"),
        signals.get("information_necessity_precision"),
        beta=float(opts.get("beta", 1.0)),
        eps=float(opts.get("eps", 1e-9)),
    ),
}


def compute_composite_term(
    name: str, signals: Mapping[str, Optional[float]], options: Mapping[str, Any]
) -> float:
    """组合项：从多个裁判信号算出综合分。"""
    fn = COMPOSITE_FUNCS.get(name)
    if fn is None:
        raise KeyError(f"{name!r} 不是组合项（没有注册对应的 COMPOSITE_FUNCS 实现）")
    return float(fn(signals, options))
