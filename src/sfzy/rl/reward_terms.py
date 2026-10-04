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
from typing import Any, Dict, Optional, Tuple

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
    source: str                     # "rule" | "judge"
    signal: Optional[str] = None    # judge 项消费的信号名；rule 项为 None
    description: str = ""
    internal_weight_fields: Dict[str, Tuple[str, ...]] = field(default_factory=dict)


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
