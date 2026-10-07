"""Judge 用的几段 Prompt。

================ 哪些改了、哪些没改 ================
判定（事实一致性 JUDGE_SYSTEM / 覆盖率 COVERAGE_SYSTEM）**逐字照搬需求原文**，
不改措辞：那两段是要被验证的对象。

抽取（EXTRACT_SYSTEM）已经过两轮加强：
  * v2：原版只说"尽量保留…日期、金额…"，实测 7B 抽取器会把
    "10月19日出具了劳动合同于10月20日解除的离职证明"转述成
    "10月19日解除劳动合同"——日期被挪用，于是裁判拿这份被改坏的"原文要素"
    去比一条正确摘要，把正确摘要判成 0（人工摘要也未能幸免）。于是把
    "逐字摘录、不得改写日期/金额/行为/肯否"写成硬约束，并给了一个反例。
  * v3：补"程序事实优先"。判决书里常同时出现"何涛辩称…"（仲裁阶段/别处
    引用的陈述）和"何涛未答辩。本院认定事实如下…"（本案记录），抽取器会
    抓前者，把本案的 defendant_defenses 写错；现在明确要求以本案的
    "未答辩/未到庭/缺席审理"和"本院查明"为准。
另外原文和摘要**分两套 prompt**：`EXTRACT_SYSTEM` 抽判决书原文（长文、要素
齐全、要防"仲裁阶段的辩称"），`EXTRACT_SUMMARY_SYSTEM` 抽人工/候选摘要
（短、大量要素为空、要防"把别的要素挪过来硬凑"）。调用方用
`build_extract_messages(text, kind="document"|"summary")` 选择。
版本号见 EXTRACT_PROMPT_VERSION，换了 prompt 要重跑 judge_probe / judge_test
重新出基线。
"""

from __future__ import annotations

from typing import Dict, List

from sfzy.judge.schema import ELEMENT_ZH, ELEMENTS

# ---------------------------------------------------------------------------
# 六要素提取（需求第二节）
# ---------------------------------------------------------------------------
EXTRACT_PROMPT_VERSION = "v4-verbatim-procedure-reasoning"
EXTRACT_SUMMARY_PROMPT_VERSION = "s1-summary"

# EXTRACT_SYSTEM = """你是一个裁判文书信息抽取模型。

# 你的任务是从输入的裁判文书中提取裁判文书本身的六个核心要素（不要混淆其它案件描述：如裁判文书引用的仲裁、之前的裁判文书等）。

# 六个要素分别为：

# 1. case_type：案件类型或案由
# 2. plaintiff_claims：当前裁判中原告的诉讼请求
# 3. defendant_defenses：当前裁判中被告的辩称、抗辩意见
# 4. court_facts：当前裁判中法院审理查明、认定的案件事实（选择对最终判决真正有帮助的进行表述）
# 5. legal_basis：当前裁判中法院裁判所依据的法律、司法解释、法律条文
# 6. judgment_result：当前裁判中法院最终裁判结果

# 特别注意：

# - 一般六个要素是按顺序依次出现的，根据位置判断六个要素的提取范围；
# - 是当前裁判文书的六个要素还是其它案件的事实、观点、陈述；
# - 如果文本明确写了“被告未答辩”“被告未到庭”“缺席审理”，那么 defendant_defenses 就写“未答辩”；
# - plaintiff_claims、defendant_defenses、court_facts三部分选择对最终判决真正有帮助的进行表述。

# 写作要求：

# - 一句话包含多个事实点时，逐条分开写（用分号或编号并列），不要合并成一句概括。

# 如果某个要素在输入文本中不存在，返回空字符串。

# 只输出合法 JSON，不输出 Markdown，不解释。

# 输出格式严格为：

# {
#   "case_type": "",
#   "plaintiff_claims": "",
#   "defendant_defenses": "",
#   "court_facts": "",
#   "legal_basis": "",
#   "judgment_result": ""
# }"""

EXTRACT_SYSTEM = """你是一个裁判文书信息抽取模型。

你的任务是从输入的裁判文书中提取裁判文书本身的六个核心要素（不要混淆其它案件描述，如裁判文书引用的仲裁、之前的裁判文书等）。

六个要素分别为：

1. case_type：案件类型或案由
2. plaintiff_claims：当前裁判中原告的诉讼请求
3. defendant_defenses：当前裁判中被告的辩称、抗辩意见
4. court_facts：当前裁判中法院审理查明、认定的、直接支撑最终裁判结果的核心事实以及法院由这些事实得出的结论。仅保留决定裁判结果所必需的事实，不提取背景性、过程性、证据列举性事实，以及删除后仍不影响理解裁判结论的事实。保留法院由事实得出的结论。
5. legal_basis：当前裁判中法院裁判所依据的法律、司法解释、法律条文
6. judgment_result：当前裁判中法院最终裁判结果

特别注意：

- 一般六个要素按顺序依次出现，可根据位置辅助判断六个要素的提取范围；
- 注意区分当前裁判文书的六个要素与文中引用的其它案件、仲裁或先前裁判中的事实、观点和结论；
- 如果文本明确写了“被告未答辩”“被告未到庭”“缺席审理”，则 defendant_defenses 写“未答辩”；
- plaintiff_claims、defendant_defenses、court_facts 均只保留与当前裁判结果直接相关的核心内容；
- court_facts 必须高度精简。判断一个事实是否应保留：如果删除该事实，仍不影响理解法院为何作出当前裁判结果，则不要提取；
- 不要因为某事实出现在“法院查明”“本院认定”等部分就全部提取；
- court_facts 优先保留决定责任成立与否、责任范围、金额、比例、期限及裁判结果的事实；一般不展开完整时间线、证据来源、举证质证过程、程序经过、重复陈述和与裁判结果无直接关系的细节；
- 不得为了补全六个要素而推测、补充输入文本中不存在的信息。

写作要求：

- 对需要独立核验的核心事实点分开表述，但不要将同一事实关系过度拆分；
- 具有紧密因果、行为—结果或修饰关系的内容可以作为一个完整事实点；
- 在保证裁判核心事实完整的前提下，尽可能少地提取 court_facts。

如果某个要素在输入文本中不存在，返回空字符串。

只输出合法 JSON，不输出 Markdown，不解释。

输出格式严格为：

{
  "case_type": "",
  "plaintiff_claims": "",
  "defendant_defenses": "",
  "court_facts": "",
  "legal_basis": "",
  "judgment_result": ""
}"""


# ---------------------------------------------------------------------------
# 六要素提取：摘要版（和原文版分开）
# ---------------------------------------------------------------------------
# 为什么分开：判决书原文长、要素齐全，陷阱在"仲裁阶段引用的辩称 vs 本案未答辩"；
# 摘要很短、大量要素根本没写，陷阱在"把别的要素挪过来硬凑"。同一段 prompt 顾
# 不上两头，所以原文用 EXTRACT_SYSTEM，摘要用下面这段。
EXTRACT_SUMMARY_SYSTEM = """你是一个裁判文书摘要的信息抽取模型。

输入是一段裁判文书摘要（对判决书原文的压缩），请从这段摘要里抽取案件的六个核心要素。

六个要素分别为：

1. case_type：案件类型或案由
2. plaintiff_claims：原告的诉讼请求
3. defendant_defenses：被告的辩称、抗辩意见
4. court_facts：法院审理查明、认定的案件事实
5. legal_basis：法院裁判所依据的法律、司法解释、法律条文及主要裁判理由
6. judgment_result：法院最终裁判结果

抽取原则：

- 只需要抽取。
- 如果是被告未答辩，也要写在 defendant_defenses 里写出未答辩。

只输出合法 JSON，不输出 Markdown，不解释。

输出格式严格为：

{
  "case_type": "",
  "plaintiff_claims": "",
  "defendant_defenses": "",
  "court_facts": "",
  "legal_basis": "",
  "judgment_result": ""
}"""


# 原文 / 摘要两套抽取 prompt。默认 document，保持旧调用不变。
EXTRACT_PROMPTS: Dict[str, str] = {
    "document": EXTRACT_SYSTEM,
    "summary": EXTRACT_SUMMARY_SYSTEM,
}


def build_extract_messages(text: str, kind: str = "document") -> List[Dict[str, str]]:
    """`kind`：`document`=裁判文书原文，`summary`=判决书摘要（人工/候选）。"""
    try:
        system = EXTRACT_PROMPTS[kind]
    except KeyError as exc:
        raise ValueError(
            f"未知的抽取类型 {kind!r}（可用：{sorted(EXTRACT_PROMPTS)}）"
        ) from exc
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


# ---------------------------------------------------------------------------
# 事实一致性判定（需求第三节）
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """你是严格的裁判文书摘要事实一致性评价器。判断【摘要要素】中的事实是否受到【原文要素】支持。只评价事实一致性及表达是否足以确定事实，不评价完整性和语言风格。

核心原则：判断方向始终是“原文 → 摘要”。

1. 【允许省略】原文有、摘要没有的信息，不扣分。
2. 【允许概括】原文更具体、摘要更抽象时，只要原文能够推出摘要表述，就视为一致。
3. 【禁止新增】摘要主动陈述而原文无法支持的事实属于编造。
4. 【禁止改变】主体、行为、金额、日期、数量、法律依据、肯否关系、裁判结果等与原文不一致，属于事实错误。
5. 严格区分原告诉请、被告辩称和法院认定，不得相互替换。

评分：

0：存在事实错误或编造。包括原文无法支持的新增事实，以及主体、行为、金额、日期、数量、肯否关系、法律依据、法院认定、裁判结果等事实与原文不一致。核心事实错误、事实反转、主体或责任颠倒、虚构事实均为0分。

1：事实表达存在明显问题。没有确认存在事实错误或编造，但摘要存在明显表达模糊、语义代指不清、主体指向不明、句法或语义关系混乱等问题，导致事实含义无法可靠确定。

2：存在实质性疑点。没有发现明确事实错误或编造，但部分陈述能否由原文支持无法确定。合理概括不属于疑点。

3：基本一致。无事实错误或编造，仅有不影响主要事实含义的轻微偏差。

4：完全一致。摘要陈述均受原文支持；允许省略、同义改写、压缩和合理概括。

决策顺序：

发现事实错误或编造 → 0；
无事实错误，但表达模糊、代指不清或语病导致事实含义无法可靠确定 → 1；
无明确错误，但部分事实支持关系存在实质性疑点 → 2；
仅有轻微偏差 → 3；
全部受支持 → 4。

特别注意：

“原文有、摘要无”是省略，不是错误。
“摘要有、原文无”是编造，评0分。
“原文具体、摘要概括”只要原文能够推出摘要表述，就不是错误，可以评4分。
不要因为摘要其他内容正确而抵消任何已经确认的事实错误或编造。

只输出0、1、2、3或4。"""


# 事实一致性 + 六要素上下文（`semantic.element_context: true`）时用的 system。
# 判分规则全部留在 system 里，user 只放材料——否则 user 里再写一套规则，会
# 和这里的 rubric 打架、被模型当成"后出现的规则"覆盖掉。
JUDGE_SYSTEM_CONTEXT = JUDGE_SYSTEM + """

【本次判定的额外约定：六要素上下文】

本次你会看到三块内容（都在 user 消息里）：
  * 【判分依据】：原文中与你正要评价的摘要要素相对应的那一项；
  * 【辅助参考】：原文的其余五项；
  * 【待评价摘要的要素】：本次要评价的、摘要里的那一个要素。

判断时遵守：

1. 判分的主依据是【判分依据】。
2. 【辅助参考】只是辅助——它用来排查"抽取时把内容归到了别的要素里"这种边界误差，
   不能当作与【判分依据】同等强度的证据。
3. 如果摘要该要素的陈述在【判分依据】里找不到、但在【辅助参考】的任意一项里能找到：
   * 这属于抽取归类差异，**不是编造**，不得因此判 0。
4. 只有当摘要该要素的陈述在原文六项里**都找不到**支持时，才按上面的"编造/新增"判 0。

只输出0、1、2、3或4。"""


# ---------------------------------------------------------------------------
# 关键要素覆盖率判定
# ---------------------------------------------------------------------------
# 和事实一致性**方向相反**：那边"参考有、摘要没有"是省略，不扣分；
# 这边正是要罚省略。所以必须是独立 prompt，绝不能复用上面那套。
#
# 只做一版：rubric 本身已经把 0~4 每一档写死了，不提供示例变体。
COVERAGE_SYSTEM = """你是裁判文书摘要的要素覆盖率评价器。

给你【参考摘要要素】（人工撰写的正确摘要中的某一项）和【候选摘要要素】（待评价摘要中的同一项）。
请判断候选摘要对参考摘要这一项的覆盖程度。

评分（只输出一个整数）：

0：未覆盖该要素，或核心语义与参考摘要不一致。
1：仅涉及少量相关信息，存在明显遗漏。
2：部分覆盖，遗漏部分重要信息。
3：核心信息基本覆盖，仅遗漏次要信息。
4：完整覆盖核心信息。

判断要点：

1. 方向是“参考摘要 → 候选摘要”：看参考里有的信息，候选写出来了多少。
2. 允许换词、压缩和合理概括；只要核心语义一致就算覆盖，不要因为措辞不同扣分。
3. 候选多写了参考里没有的内容不扣分 —— 多写由别的指标负责。
4. 但如果候选与参考的核心语义矛盾（主体、行为、金额、日期、肯否关系、
   裁判结果等相反），给 0。
5. 逐档对齐：遗漏的是次要信息给 3，遗漏的是重要信息给 2，只沾到一点边给 1。

只输出一个整数（0/1/2/3/4）。"""


def build_coverage_messages(
    element_name: str, reference_element: str, candidate_element: str
) -> List[Dict[str, str]]:
    """覆盖率判定的两段 prompt。

    `reference_element` 是人工摘要的对应要素，`candidate_element` 是候选摘要的。
    参考摘要里不存在的要素根本不会走到这里（调用方直接跳过，见
    `FactConsistencyJudge.build_coverage_pairs`）。
    """
    zh = ELEMENT_ZH.get(element_name, element_name)
    user = f"""当前评价要素：
{zh}（{element_name}）

参考摘要要素：
{reference_element}

候选摘要要素：
{candidate_element}

请判断候选摘要要素对参考摘要要素的覆盖程度，只输出一个整数（0/1/2/3/4）。"""
    return [
        {"role": "system", "content": COVERAGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_judge_messages(
    element_name: str, document_element: str, candidate_element: str
) -> List[Dict[str, str]]:
    """`element_name` 用英文键名，但正文里给出中文名 —— 需求里就是这么写的。

    只有这一版 prompt：判定规则已经逐档写死在 system 里，不提供少样本变体。
    """
    zh = ELEMENT_ZH.get(element_name, element_name)

    user = f"""当前评价要素类型：
{zh}（{element_name}）

裁判文书原文要素：
{document_element}

待评价摘要要素：
{candidate_element}

    请判断待评价摘要要素与裁判文书原文要素的事实一致性。"""
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _as_values(elements) -> Dict[str, str]:
    if elements is None:
        return {}
    return elements.to_dict() if hasattr(elements, "to_dict") else dict(elements)


def _element_line(name: str, value: str) -> str:
    return f"- {ELEMENT_ZH[name]}（{name}）：{(value or '').strip() or '（空）'}"


def build_judge_messages_with_context(
    element_name: str,
    document_elements,
    candidate_elements=None,
    document_target: str = None,
    candidate_target: str = None,
) -> List[Dict[str, str]]:
    """带原文六要素上下文的判定 prompt（本次只判 `element_name` 这一项）。

    每次判定给：
      * 原文的**判分依据**（本次对比的那一项，单独成块）；
      * 原文的**其余五项**，单独放在"辅助参考"里，明确只用来兜抽取归类误差；
      * 摘要的**仅本次要判的那一个要素**。

    为什么这样不对称：抽取的边界有噪声，同一件事可能被原文抽进 court_facts、
    却被摘要抽进 legal_basis（或反过来）。只给"要对比的两小段"时，这种边界
    误差会被当成"编造/不一致"。所以把原文六项都摆出来排查；但另外五项只是
    辅助，摘要侧也只给要判的那一项，不引入无关信息。

    扣分口径：内容只在辅助项里能找到时——放对了位置视同一致；放错了位置
    算轻微问题给 3 分，**不是 0**（0 只留给六项里都找不到支持的编造）。

    `document_target` / `candidate_target` 是实际送判的对比文本。正常就是对应
    要素本身；原文该要素抽空、改用整篇原文兜底时，`document_target` 放兜底文本。
    """
    zh = ELEMENT_ZH.get(element_name, element_name)
    doc_values = _as_values(document_elements)
    primary = document_target if document_target is not None \
        else doc_values.get(element_name)
    other_lines = "\n".join(
        _element_line(name, doc_values.get(name))
        for name in ELEMENTS if name != element_name
    )
    if candidate_target is None and candidate_elements is not None:
        candidate_target = _as_values(candidate_elements).get(element_name) or ""
    cand_value = (candidate_target or "").strip() or "（空）"

    user = f"""当前评价要素类型：
{zh}（{element_name}）

【判分依据：原文的{zh}】
{_element_line(element_name, primary)}

【辅助参考：原文的其余五项】（**只是辅助**：用来排查抽取时的归类/边界误差，不作为判分依据）
{other_lines}

【待评价摘要的要素】
{zh}（{element_name}）：{cand_value}

请只评价摘要的【{zh}】与原文的事实一致性。"""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_CONTEXT},
        {"role": "user", "content": user},
    ]
