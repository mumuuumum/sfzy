"""分层奖励：门控 + 事实 F1 + 生成式语义分 + ROUGE-L。

================================ 为什么这么设计 ================================
从旧模型的 12172 条输出上分析出来的结论：

  * 模型漏掉的数字里 **98.5% 明确写在原文里** —— 不是幻觉，是漏抄
  * 训练集上 28.8% 的样本至少漏了一个原文里的事实
  * 漏事实的样本平均分 0.5443，干净样本 0.6175，**差 0.073**
  * 而且**输出比参考长 40% 时仍有 14.3% 漏事实** —— 和长度无关

所以痛点是"关键事实的遗漏"，不是幻觉、不是长度、不是格式。而 ROUGE 对
"漏一个 6 字符的金额"几乎不敏感（周围 90% 的文字都对），**所以拿 ROUGE
当奖励，模型根本没有动机去写数字**。

================================ 三层结构 ================================
  第一层  硬门控 —— 不满足直接 0 分，后面都不算。切断"牺牲格式换分数"的路径
  第二层  事实覆盖 ★ 主信号 —— 参考里的关键事实被覆盖了多少
  第三层  ROUGE-L —— 辅助，保持和官方评测口径一致

门控和加权求和的区别是**可补偿性**：加权求和下模型能学会"格式烂一点、
但 ROUGE 多拿分"，总分反而更高；门控切断的就是这条路。

================================ 四种模式 ================================
`mode` 是一个字符串，决定**哪些项参与、能不能互相补偿**。四种模式共用
同一套分项计算（`RewardBreakdown`），所以对比是干净的单变量：

  rouge_only    总分 = ROUGE-L。官方口径的裸基线，回答"不设计奖励能到哪"
  flat          所有项加权求和，长度是正向项。可补偿，对照组
  gated         门控 → 不通过直接 0；通过后 事实 F1 + ROUGE-L
  gated_judge   gated 再加上生成式语义分（LLM 裁判），本次的主方案
  fact_judge    门控 + **六要素事实一致性 Judge**（Qwen）+ ROUGE-L。
                事实项换成 Judge 的加权分，完全不跑金额/日期/法条规则匹配，
                见 `sfzy/judge/`。判据、权重、空字段规则都在需求里写死。

================================ 为什么事实项换成对称 F1 ================================
旧的 `fact_coverage` 在"参考里没有事实"时返回 0 —— 那些样本占 61.7%。
0 是**组内常数**，GRPO 的组内归一化会把它整个抵消，整组零梯度。

更要命的是方向：模型在参考**没有**事实的样本上多写数字，代价是 ROUGE 掉
0.099（实测），但 `fact_coverage` 对此**毫无惩罚**（返回 0，反正不加分）。
模型学不会"什么时候该写数字"。

对称 F1 把这件事补上：

    参考无事实 + 候选无事实 → 1.0    一致，满分
    参考无事实 + 候选写了   → 0.0    多写 = 错，有惩罚（组内不再是常数）
    参考有事实 + 候选没写   → 0.0    漏写 = 错
    都有 → 重叠的 F1

================================ 防 hacking ================================
事实项有个明显的漏洞：模型可以把参考里所有数字都堆进去。
三道防线：
  1. 长度门控 —— 堆数字会把长度推高，直接撞上限
  2. `fact_precision` —— 输出的事实有多少真在原文里（精确率进了 F1，但仍单独记录）
  3. ROUGE 互补 —— 事实项和 ROUGE 反向变动就是报警

**前两条都要和覆盖率一起记录**，只看覆盖率会自欺欺人。

裁判项同样会被 hack（模型学会讨好裁判），所以 `semantic` 必须和
`fact_precision`、`rouge_l` 一起看：三个一起涨才是真的好了。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sfzy.eval.rouge import score_pair

# --------------------------------------------------------------------------
# 事实提取
# --------------------------------------------------------------------------
# 金额：支持 元 / 万元 / 亿元 三种单位，统一折算到"元"。
# 不折算的话，"48000元" 和 "4.8万元" 会被算成两个不同事实，信号全是噪声。
MONEY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(亿元|万元|元)")
UNIT_TO_YUAN = {"元": 1.0, "万元": 1e4, "亿元": 1e8}

# 日期：年 / 年-月 / 年-月-日 三档，按书写精度归一化
DATE_RE = re.compile(r"(\d{4})\s*年(?:\s*(\d{1,2})\s*月)?(?:\s*(\d{1,2})\s*日)?")

# 编号（案号、身份证号等）。必须放在金额和日期之后提取，否则会把
# 48000 这样的金额、2017 这样的年份也当成编号。
ID_RE = re.compile(r"\d{4,}")

# 法条号。按项目决定不考虑法条，先挖掉，避免"第107条"里的数字污染事实集合。
ARTICLE_RE = re.compile(r"第\s*\d+\s*条")

RESULT_MARKERS = ("判决如下", "判令", "驳回", "本院认为", "判决", "裁定")

DEFAULT_REWARD_CFG: Dict[str, Any] = {
    # rouge_only | flat | gated | gated_judge | fact_judge，见模块 docstring
    "mode": "gated",
    # weights 的键是"项名"。事实项的键叫 fact（不叫 fact_coverage），
    # 因为它乘的到底是 coverage 还是 f1 由 fact_term 决定。
    # 老配置里的 fact_coverage 键会被自动识别，见 _merge_cfg。
    "weights": {"rouge_l": 0.3, "fact": 0.4, "semantic": 0.3, "length": 0.1},
    # 事实项用哪个函数：coverage（旧口径）/ f1（对称，能用组内梯度）
    "fact_term": "f1",
    "fact_kinds": ["money", "date", "id"],
    "rouge_mode": "jieba",
    # 裁判分数的量程。scorer 返回 0-100，这里除以它变成 0-1。
    "semantic_scale": 100.0,
    "gate": {
        "min_chars": 60,
        "length_ratio_range": [0.5, 1.5],
        "forbidden_prefixes": ["以下是", "摘要：", "摘要:", "本摘要", "这是"],
        "require_result_marker": True,
    },
}


@dataclass
class RewardBreakdown:
    """分项奖励。**必须返回分项** —— 只看总分判断不出"这一版好在哪"。"""

    total: float = 0.0
    gated: bool = False
    gate_reason: str = ""
    rouge_l: float = 0.0
    fact_coverage: float = 0.0
    fact_score: float = 0.0
    fact_judge: float = 0.0
    fact_precision: float = 0.0
    semantic: float = 0.0
    length_ratio: float = 0.0
    n_ref_facts: int = 0
    n_missed_facts: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward": self.total,
            "reward_rouge_l": self.rouge_l,
            "reward_fact_coverage": self.fact_coverage,
            "reward_fact_score": self.fact_score,
            "reward_fact_judge": self.fact_judge,
            "reward_semantic": self.semantic,
            "fact_precision": self.fact_precision,
            "length_ratio": self.length_ratio,
            "n_ref_facts": self.n_ref_facts,
            "n_missed_facts": self.n_missed_facts,
            "gated": float(self.gated),
        }

    def fact_value(self, term: str = "f1") -> float:
        """按 `fact_term` 取事实项的值。"""
        return self.fact_coverage if term == "coverage" else self.fact_score


def _mask(text: str, spans: Iterable[Tuple[int, int]]) -> str:
    """把已匹配的片段挖成空格，避免同一段被两种事实重复提取。"""
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


def extract_facts(text: str, kinds: Sequence[str] = ("money", "date", "id")) -> set:
    """抽出文本里的事实，返回归一化后的集合。

    归一化规则：
        金额  "48000元" / "4.8万元"   → "money:48000.00"
        日期  "2015年7月19日"          → "date:2015-07-19"
              "2017年5月"               → "date:2017-05"
        编号  "1023"（案号片段）        → "id:1023"

    提取顺序很重要：**先挖掉法条号，再依次挖金额、日期，最后剩下的
    4 位以上数字才算编号**。顺序反了的话，金额和年份都会被当成编号。
    """
    facts: set = set()

    # 第一刀：法条号（"第107条"）
    work = _mask(text, [m.span() for m in ARTICLE_RE.finditer(text)])

    if "money" in kinds:
        spans = []
        for m in MONEY_RE.finditer(work):
            facts.add(f"money:{float(m.group(1)) * UNIT_TO_YUAN[m.group(2)]:.2f}")
            spans.append(m.span())
        work = _mask(work, spans)

    if "date" in kinds:
        spans = []
        for m in DATE_RE.finditer(work):
            year, month, day = m.group(1), m.group(2), m.group(3)
            key = f"date:{year}"
            if month:
                key += f"-{int(month):02d}"
                if day:
                    key += f"-{int(day):02d}"
            facts.add(key)
            spans.append(m.span())
        work = _mask(work, spans)

    if "id" in kinds:
        for m in ID_RE.finditer(work):
            facts.add(f"id:{m.group()}")

    return facts


def fact_coverage(
    candidate: str, reference: str, kinds: Sequence[str] = ("money", "date", "id")
) -> Tuple[float, int, int]:
    """参考里的事实被候选覆盖了多少。返回 (覆盖率, 参考事实数, 漏掉数)。

    参考摘要当金标准是合理的：实测它里面的数字 98.5% 都能在原文找到。
    参考为空时返回 0.0（无从判断）。
    """
    ref_facts = extract_facts(reference, kinds)
    if not ref_facts:
        return 0.0, 0, 0
    hit = len(ref_facts & extract_facts(candidate, kinds))
    return hit / len(ref_facts), len(ref_facts), len(ref_facts) - hit


def fact_precision(
    candidate: str, source: str, kinds: Sequence[str] = ("money", "date", "id")
) -> float:
    """候选里的事实有多少真在原文里。

    **这是监控指标，不是奖励项。** 用来发现"堆数字"型 hacking：
    模型为了刷覆盖率把参考里所有金额都塞进去时，它会下降。

    候选里一个事实都没有时返回 1.0 —— 那不是"不精确"，是"没写"，
    该由覆盖率去惩罚，不该在这里重复扣分。
    """
    cand_facts = extract_facts(candidate, kinds)
    if not cand_facts:
        return 1.0
    return len(cand_facts & extract_facts(source, kinds)) / len(cand_facts)


def fact_score(
    candidate: str, reference: str, kinds: Sequence[str] = ("money", "date", "id")
) -> float:
    """事实项的**对称 F1**：既罚漏写，也罚多写。

    四种情况的取值（这是它和 `fact_coverage` 的全部区别）：

        参考无事实 + 候选无事实 → 1.0   两边一致，这组该拿满分
        参考无事实 + 候选写了   → 0.0   ★ 多写要扣分
        参考有事实 + 候选没写   → 0.0   漏写扣分（和 coverage 一致）
        都有事实               → 重叠的 F1

    ★ 那一行是这次改动的核心。旧口径返回 0，而 0 在"参考无事实"的组里
    是常数 —— GRPO 组内归一化会把它抵消掉，整组零梯度。改成 1/0 之后，
    "参考不写数字时也不该乱写数字"才第一次有了训练信号。

    实测依据：参考**无**事实的 827 条里，模型多写 3 个以上数字的样本
    官方总分从 0.5912 掉到 0.4921 —— 损失巨大却没有对应的惩罚项。

    注意 F1 用的是"参考"而不是"原文"：参考是金标准，原文里的事实未必该
    进摘要（写全文数字是 ROUGE 和长度都会罚的另一种错）。
    """
    ref_facts = extract_facts(reference, kinds)
    cand_facts = extract_facts(candidate, kinds)

    if not ref_facts and not cand_facts:
        return 1.0
    if not ref_facts or not cand_facts:
        return 0.0

    overlap = len(ref_facts & cand_facts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_facts)
    recall = overlap / len(ref_facts)
    return 2 * precision * recall / (precision + recall)


def length_reward(candidate: str, reference: str, tolerance: float = 0.5) -> float:
    """长度落在参考的 ±tolerance 内给满分，越远越低。**只在 flat 模式下用。**"""
    ref_len = len(reference.strip())
    if ref_len == 0:
        return 0.0
    deviation = abs(len(candidate.strip()) - ref_len) / ref_len
    return max(0.0, 1.0 - deviation / tolerance)


def check_gate(candidate: str, reference: str, gate_cfg: Dict[str, Any]) -> Optional[str]:
    """返回 None 表示通过，否则返回被拦下的原因（写进日志用）。"""
    text = candidate.strip()
    ref_len = max(len(reference.strip()), 1)

    if len(text) < gate_cfg.get("min_chars", 60):
        return "too_short"

    ratio = len(text) / ref_len
    lo, hi = gate_cfg.get("length_ratio_range", [0.5, 1.5])
    if ratio < lo:
        return "length_too_short"
    if ratio > hi:
        return "length_too_long"

    for prefix in gate_cfg.get("forbidden_prefixes", []):
        if text.startswith(prefix):
            return f"prefix:{prefix}"

    if gate_cfg.get("require_result_marker"):
        if not any(marker in text for marker in RESULT_MARKERS):
            return "no_result_marker"

    return None


def compute_reward(
    candidate: str,
    reference: str,
    source: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    semantic: Optional[float] = None,
) -> Tuple[float, RewardBreakdown]:
    """算一条样本的奖励，返回 (总分, 分项明细)。

    `semantic` 是裁判给出的原始分。量纲随模式而定：

      gated_judge   点式/排序裁判的 0-100 综合分，按 `semantic_scale` 归一
      fact_judge    六要素事实一致性 Judge 的加权分，**本来就是 [0,1]**，
                    不做 `semantic_scale` 换算（见 `bd.fact_judge`）

    其余模式即使传了也只记录进明细，不参与总分 —— 这样"同一个 checkpoint
    用不同奖励口径离线重打分"是免费的。

    四种模式的差别只在最后几行，**分项计算完全共用**，所以 A1/A2/A4 的
    对比是干净的单变量。
    """
    full_cfg = _merge_cfg(cfg)
    weights = full_cfg["weights"]
    term = full_cfg["fact_term"]
    mode = full_cfg["mode"]
    fact_judge_mode = mode == "fact_judge"
    # fact_judge 模式下事实项完全来自 Qwen Judge，**不跑任何规则匹配** ——
    # 需求明确要求金额/日期/法条/主体的判断全部交给裁判，不写额外 checker。
    # 空 kinds 让下面三条规则函数直接返回中性值，不产生误导性的日志字段。
    kinds = [] if fact_judge_mode else full_cfg["fact_kinds"]

    bd = RewardBreakdown()
    bd.semantic = 0.0 if semantic is None else float(semantic) / full_cfg["semantic_scale"]
    # 事实一致性分单独存一份：它是 [0,1]，和 semantic 的 0-100 量纲不同，
    # 混在一起会差 100 倍 —— 这是最容易悄悄写错的一处。
    bd.fact_judge = 0.0 if semantic is None else float(semantic)
    bd.rouge_l = score_pair(
        candidate.strip(), reference.strip(), mode=full_cfg["rouge_mode"]
    )["rouge-l-f"]
    bd.fact_coverage, bd.n_ref_facts, bd.n_missed_facts = fact_coverage(
        candidate, reference, kinds
    )
    bd.fact_score = fact_score(candidate, reference, kinds)
    bd.length_ratio = len(candidate.strip()) / max(len(reference.strip()), 1)
    if source:
        bd.fact_precision = fact_precision(candidate, source, kinds)

    # ---- A1：官方指标的裸基线。不过门控、不看事实，回答"传统做法能到哪" ----
    if mode == "rouge_only":
        bd.total = bd.rouge_l
        return bd.total, bd

    # ---- 对照组：所有项加权求和，可以互相补偿 ----
    if mode == "flat":
        bd.total = (
            weights["rouge_l"] * bd.rouge_l
            + weights["fact"] * bd.fact_value(term)
            + weights["length"] * length_reward(candidate, reference)
        )
        return bd.total, bd

    reason = check_gate(candidate, reference, full_cfg["gate"])
    if reason is not None:
        bd.gated = True
        bd.gate_reason = reason
        bd.total = 0.0
        return 0.0, bd

    # ---- 事实一致性主方案：门控 + 六要素 Judge 加权分 + ROUGE-L ----
    if fact_judge_mode:
        if semantic is None:
            # 和 gated_judge 同样的理由：静默降级只会让你以为在跑 Judge，其实没有。
            raise ValueError(
                "mode=fact_judge 需要每条样本都有六要素事实一致性分，收到 None。"
                "检查 semantic.backend=fact 的裁判是否传给了 compute_rewards，"
                "以及脚本启动日志里的 '语义裁判' 行。"
            )
        bd.total = weights["rouge_l"] * bd.rouge_l + weights["fact"] * bd.fact_judge
        return bd.total, bd

    # ---- 主方案：门控 + 事实 F1 + ROUGE-L（+ 可选的裁判语义分）----
    bd.total = weights["rouge_l"] * bd.rouge_l + weights["fact"] * bd.fact_value(term)
    if mode == "gated_judge":
        if semantic is None:
            # 静默降级成 gated 是最坏的选项：你会以为在跑 A4，其实跑的是 A2。
            raise ValueError(
                "mode=gated_judge 需要每条样本都有裁判分，收到 None。"
                "检查 semantic scorer 是否传给了 compute_rewards，"
                "以及脚本启动日志里的 '裁判' 行。"
            )
        bd.total += weights["semantic"] * bd.semantic
    return bd.total, bd


def compute_rewards(
    candidates: Sequence[str],
    references: Sequence[str],
    sources: Optional[Sequence[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    semantic_scores: Optional[Sequence[Optional[float]]] = None,
) -> List[RewardBreakdown]:
    """批量打分，返回分项明细列表（GRPO 的 rollout 用这个）。

    `semantic_scores` 是裁判分（原始量程，默认 0-100），由调用方先批量算好
    再传进来 —— 裁判模型比规则慢几个数量级，必须批处理，不能在这里逐条调用。
    """
    sources = sources or [None] * len(candidates)
    if semantic_scores is None:
        semantic_scores = [None] * len(candidates)
    if len(semantic_scores) != len(candidates):
        raise ValueError(
            f"裁判分个数 {len(semantic_scores)} 与候选数 {len(candidates)} 不一致"
        )
    return [
        compute_reward(c, r, s, cfg, semantic=sem)[1]
        for c, r, s, sem in zip(candidates, references, sources, semantic_scores)
    ]


def summarize_gate_reasons(breakdowns: Sequence[RewardBreakdown]) -> Dict[str, int]:
    """统计被门控的比例和原因分布 —— 进训练日志。

    如果门控比例很高（比如 50%），说明问题出在生成配置或 prompt，
    **不是奖励设计**。没有这个统计你就分不清。
    """
    stats: Dict[str, int] = {"total": len(breakdowns), "gated": 0}
    for bd in breakdowns:
        if bd.gated:
            stats["gated"] += 1
            stats[bd.gate_reason] = stats.get(bd.gate_reason, 0) + 1
    return stats


def _merge_cfg(override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把配置和默认值深合并一层，允许只覆盖部分字段。

    两处兼容处理：
      * 老配置写的 `weights.fact_coverage` 自动改名为 `weights.fact`
        （事实项乘 coverage 还是 f1 由 `fact_term` 决定，键名不该跟着变）
      * `mode: gated_judge` 但权重里没有 semantic → 补上默认权重，否则
        裁判分乘 0，等于没接
    """
    import copy

    cfg = copy.deepcopy(DEFAULT_REWARD_CFG)
    if override:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    if "fact_coverage" in cfg["weights"]:
        cfg["weights"]["fact"] = cfg["weights"].pop("fact_coverage")
    if cfg["mode"] == "gated_judge" and "semantic" not in cfg["weights"]:
        cfg["weights"]["semantic"] = DEFAULT_REWARD_CFG["weights"]["semantic"]
    return cfg
