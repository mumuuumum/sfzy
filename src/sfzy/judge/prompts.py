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

# 六要素之一的 court_facts 定义（事实一致性 / 覆盖率 / 抽取共用一处，避免漂移）。
# 注意：这里把"裁判理由 / 法院说理"也算进 court_facts —— 之前只保留纯事实，
# 会把摘要里正确的"本院认为"内容误判成编造。
COURT_FACTS_DEF = "法院审理查明、认定的案件事实、裁判理由和法院说理"

# 六要素的中文定义（一次性判定 prompt 里要摆出来，模型才知道每个键指什么）
ELEMENT_DEFS_ZH = (
    "1. case_type：案件类型或案由\n"
    "2. plaintiff_claims：原告的诉讼请求\n"
    "3. defendant_defenses：被告的辩称、抗辩意见\n"
    f"4. court_facts：{COURT_FACTS_DEF}\n"
    "5. legal_basis：法院裁判所依据的法律、司法解释、法律条文及主要裁判理由\n"
    "6. judgment_result：法院最终裁判结果"
)

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
# 4. court_facts：法院审理查明、认定的案件事实、裁判理由和法院说理
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
4. court_facts：法院审理查明、认定的案件事实、裁判理由和法院说理。只保留与最终裁判结果相关的核心内容，不提取背景性、过程性、证据列举性事实。
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
4. court_facts：法院审理查明、认定的案件事实、裁判理由和法院说理
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
5. 「原告诉讼请求 / 被告辩称 / 法院查明事实与说理」这三项的 0 分门槛要更严（略微放宽）。
   这三项都是长段落，抽取本身有损，摘要往往只是压缩、概括、换词、省略。只有出现明确、可指认的冲突才判 0。

只输出0、1、2、3或4。"""


# ---------------------------------------------------------------------------
# 事实一致性：一次性给六个分（输入 = 原文全文 + 摘要六要素）
# ---------------------------------------------------------------------------
# 原来把六要素拆成 6 次判定，好处是每次只看两小段、不被锚定；代价是 6 倍调用，
# 而且原文侧抽取一旦有损（日期挪用、"本院认为"丢掉）就会把正确摘要判成 0。
# 这个版本改成"一次看全文 + 摘要六要素、输出六个分"：不再依赖原文抽取，
# 六要素仍分别给分（一次调用里给六个），不是点式总分。
SIX_SHOT_JUDGE_SYSTEM = JUDGE_SYSTEM + f"""

【输出格式（本次固定）】

你会看到【裁判文书原文】和【待评价摘要】（摘要原文，没有预先抽取要素）。
请你自己按下面六个要素逐项对照，对**六个要素分别**给出 0~4 的整数分，
判据同上（摘要该项是否受原文支持；摘要没写到的要素视为省略、给 4；省略、压缩、
同义改写不扣分；编造或与原文矛盾判 0）。

六个要素的定义：
{ELEMENT_DEFS_ZH}

只输出一个 JSON 对象，键必须是下面六个，值是 0~4 的整数，不要输出任何其他内容：

{{
  "case_type": 0,
  "plaintiff_claims": 0,
  "defendant_defenses": 0,
  "court_facts": 0,
  "legal_basis": 0,
  "judgment_result": 0
}}"""


def build_fact_six_messages(document: str, candidate: str) -> List[Dict[str, str]]:
    """一次性六要素判定：输入 (原文全文, 摘要全文)，输出六要素 0~4 分。

    不再预抽摘要六要素——prompt 里说明"按这六个要素打分"即可。
    """
    user = f"""【裁判文书原文】
{document}

【待评价摘要】
{candidate}

请按 case_type / plaintiff_claims / defendant_defenses / court_facts /
legal_basis / judgment_result 六个要素分别给出 0~4 的整数分，只输出 JSON。"""
    return [
        {"role": "system", "content": SIX_SHOT_JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


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


# ---------------------------------------------------------------------------
# 覆盖率：一次性给六个分（输入 = 人工摘要全文 + 候选摘要全文）
# ---------------------------------------------------------------------------
SIX_SHOT_COVERAGE_SYSTEM = COVERAGE_SYSTEM + f"""

【输出格式（本次固定）】

你会看到【参考摘要（人工）】和【候选摘要】。请对**六个要素分别**给出 0~4 的
覆盖程度（0=未覆盖/与参考矛盾，4=完整覆盖核心信息；措辞不同、压缩、换词不扣分）。
参考摘要里没有写到的那一项，直接给 4（该项无需覆盖）。

六个要素的定义：
{ELEMENT_DEFS_ZH}

只输出一个 JSON 对象，键必须是下面六个，值是 0~4 的整数，不要输出任何其他内容：

{{
  "case_type": 0,
  "plaintiff_claims": 0,
  "defendant_defenses": 0,
  "court_facts": 0,
  "legal_basis": 0,
  "judgment_result": 0
}}"""


def build_coverage_six_messages(reference: str, candidate: str) -> List[Dict[str, str]]:
    """一次性覆盖率判定：输入 (人工摘要全文, 候选摘要全文)，输出六要素 0~4 分。"""
    user = f"""【参考摘要（人工）】
{reference}

【候选摘要】
{candidate}

请对六个要素分别给出覆盖程度 0~4 的整数分，只输出 JSON。"""
    return [
        {"role": "system", "content": SIX_SHOT_COVERAGE_SYSTEM},
        {"role": "user", "content": user},
    ]


INP_JUDGE_PROMPT = """你是严格的裁判文书摘要信息必要性评价器。

你的任务是根据【裁判文书原文】和【人工摘要】，评价【候选摘要】中每个信息点对于裁判文书摘要是否具有保留必要性，并识别重复信息。

只评价信息必要性，不评价事实一致性、关键要素覆盖率和语言风格。

核心原则：

1. 【评价方向】只评价候选摘要已经陈述的信息，不因遗漏原文或人工摘要中的信息而扣分。
2. 【裁判相关性】判断信息是否有助于理解案件争议、当事人诉辩、法院认定的关键事实、法律关系、裁判理由及最终裁判结果。
3. 【允许概括】合理压缩、同义改写、跨句归纳不降低信息必要性。
4. 【人工摘要仅作参考】人工摘要用于辅助识别核心信息，但不是唯一标准。不得因为某信息未出现在人工摘要中就判为不必要，也不得因为其出现在人工摘要中就自动给高分。
5. 【独立评价】信息必要性由该信息在案件中的实际作用决定，不以信息长度、出现次数或文字相似度为依据。
6. 【不重复事实核验】事实正确性由独立模块评价。不要仅因某信息未能在原文中找到直接对应语句，就判定其不必要。
7. 【避免过度宽容】原文中真实存在的事实不一定值得写入摘要。与裁判结论关系较弱的背景、过程、时间地点、证据细节等，应降低必要性评分。

第一步：原子命题拆解

将候选摘要拆解为可以独立评价的信息命题。

要求：
- 每个命题保留明确的主体、行为、对象及必要限定。
- 保留影响法律关系或责任判断的金额、时间、数量、肯否关系等信息。
- 不将不同主体、不同诉请、不同责任或不同裁判事项合并。
- 不将紧密关联的行为及其结果过度拆分。
- 不增加、删除或改变候选摘要原有的信息。
- 对重复陈述的信息分别保留，不能提前去重。
- 案件类型、诉讼请求、抗辩意见、法院认定、法律依据和裁判结果均可构成独立信息命题。

第二步：逐命题评价信息必要性

对每个命题给出0至4的整数评分：

4：核心必要信息。
直接涉及案件主要争议、关键诉辩、决定性事实、法律关系认定、责任承担、核心法律依据或裁判结果。删除该信息会明显损害对案件核心内容或裁判逻辑的理解。

3：重要辅助信息。
与主要争议或裁判理由有明确关联，有助于解释核心事实、责任成立条件、责任范围或裁判结论，但不是不可缺少的核心信息。

2：一般相关信息。
与案件存在实际关联，具有一定说明作用，但对理解主要争议和裁判结果的贡献有限，通常可以进一步概括或省略。

1：低必要性信息。
主要属于背景、时间线、程序经过、证据细节或次要事实，删除后基本不影响对案件核心内容的理解。

0：无必要信息。
与案件核心争议及裁判逻辑无实质关系，属于无关细节、空泛陈述或没有实际信息价值的内容。

评分时特别注意：

- 判断信息必要性时，优先考虑其对当前案件争议和裁判结论的作用，而非其在原文中的篇幅。
- 金额、日期、地点、人物身份等细节，如果直接影响诉请、责任主体、法律适用或裁判结果，可以获得高分；否则不应仅因其具体而获得高分。
- 法院明确作出的法律关系认定、责任认定及裁判结论，通常具有较高必要性。
- 当事人的诉请和抗辩若涉及案件核心争议，也具有较高必要性，不应只关注法院查明事实。
- 不要求摘要完整复述事实经过、证据链和法院说理过程。
- 对人工摘要没有提及、但确实具有独立裁判价值的信息，应正常给予高分。
- 对人工摘要提及、但明显属于次要背景的信息，仍应根据实际必要性评分。
- 不因为某命题与其他命题重复就降低其necessity；重复由group单独处理，避免重复惩罚。

第三步：语义重复分组

为每个命题分配group编号：

- 表达相同核心事实、法律关系或裁判结论，且没有增加独立有效信息的命题，使用相同group。
- 同义改写、换词复述、重复强调同一事实，属于重复。
- 即使措辞相似，只要涉及不同主体、不同金额、不同责任、不同诉请或不同裁判事项，就不能归为同一组。
- 两个命题存在部分信息重叠，但各自包含不可省略的独立信息时，应分为不同组。
- group从1开始连续编号。

对于同一group中的重复命题，应保持必要性评分一致，并以该组共同表达的核心信息为依据评分。

输出要求：

只输出合法JSON，不输出Markdown、解释或其他内容。

严格按照以下格式：

{
  "propositions": [
    {
      "text": "原子命题",
      "necessity": 4,
      "group": 1
    },
    {
      "text": "原子命题",
      "necessity": 2,
      "group": 2
    }
  ]
}

约束：
- necessity只能是整数0、1、2、3、4。
- group必须是正整数。
- 每个原子命题必须且只能出现一次。
- 不输出最终INP分数，由外部程序计算。"""



def build_inp_messages(document: str, reference: str, candidate: str) -> List[Dict[str, str]]:
    """INP 判定：输入 (原文, 人工摘要, 候选摘要)。

    system 用 `INP_JUDGE_PROMPT`（当前是空串，留给使用者填写）；user 只摆三段材料。
    约定模型输出：
        {"propositions": [{"text": "原子命题", "necessity": 0~4, "group": 1}, ...]}
    `group` 相同即重复；没有 group 时按文本相同归组。
    """
    user = f"""【裁判文书原文】
{document}

【人工摘要】
{reference}

【候选摘要】
{candidate}"""
    return [
        {"role": "system", "content": INP_JUDGE_PROMPT},
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
    reference_elements=None,
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

    `reference_elements` 是**人工摘要**的六要素（可选）。人工摘要是人工核对过的
    可信参照：候选与它一致/能被它支持，说明是正常概括而非编造；与它矛盾则高度
    可疑。注意**不要在人工作为候选（自评）时传它**——那就成了自证。
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

    ref_block = ""
    ref_hint = ""
    if reference_elements is not None:
        ref_value = _as_values(reference_elements).get(element_name) or ""
        ref_value = ref_value.strip() or "（空）"
        ref_block = f"""
【辅助参考二：人工摘要的{zh}】（人工摘要是人工核对过的可信参照）
{_element_line(element_name, ref_value)}
"""
        ref_hint = """
（候选该项与人工摘要一致、或能由人工摘要支持 → 视为可靠，按 3 或 4 分；
与人工摘要矛盾 → 按事实错误判 0；人工摘要该项为空则本条不适用。）
"""

    user = f"""当前评价要素类型：
{zh}（{element_name}）

【判分依据：原文的{zh}】
{_element_line(element_name, primary)}

【辅助参考：原文的其余五项】（**只是辅助**：用来排查抽取时的归类/边界误差，不作为判分依据）
{other_lines}
{ref_block}
【待评价摘要的要素】
{zh}（{element_name}）：{cand_value}
{ref_hint}
请只评价摘要的【{zh}】与原文的事实一致性。"""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_CONTEXT},
        {"role": "user", "content": user},
    ]
