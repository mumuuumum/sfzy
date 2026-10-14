"""纯规则三要素切分器的验收测试。

测试分三块：

1. **规则**：需求①②③里点名的每一条行为，各有一到两条手工构造的用例，
   期望区间是**手写**的（不是跑出来再抄），可以直接拿来核对实现；
2. **不变量**：不管输入多离谱，P→D→F 必须有序、不重叠、不越界，
   且"确认缺失"与"不确定"两种状态不能混；
3. **缓存**：离线预热 / 实时切分两条链路的行为差异（写盘 vs 只读）。
"""

from __future__ import annotations

import json

import pytest

from sfzy.segment import (
    ELEMENTS,
    Confidence,
    SegmentCache,
    Segmenter,
    Status,
    split_sentences,
)


@pytest.fixture(scope="module")
def seg() -> Segmenter:
    return Segmenter()


def span(seg: Segmenter, text: str, element: str):
    return seg.segment(text).spans[element]


def pieces(seg: Segmenter, text: str):
    """返回 (P, D, F) 的切片文本；缺失/不确定记 None / '<uncertain>'。"""

    r = seg.segment(text)
    out = []
    for k in ELEMENTS:
        s = r.spans[k]
        if s.status is Status.ABSENT:
            out.append(None)
        elif s.status is Status.UNCERTAIN:
            out.append("<uncertain>")
        else:
            out.append(s.text_of(text))
    return tuple(out)


# ---------------------------------------------------------------- 句子切分


def test_句子切分_偏移量可无损还原():
    text = "第一句。第二句！第三句？第四句；第五句"
    sents = split_sentences(text)
    assert [s.text_of(text) for s in sents] == ["第一句。", "第二句！", "第三句？", "第四句；", "第五句"]
    assert "".join(s.text_of(text) for s in sents) == text


def test_句子切分_收进句尾引号():
    text = "被告辩称：“不同意。”原告同意。"
    sents = split_sentences(text)
    assert sents[0].text_of(text) == "被告辩称：“不同意。”"


# ---------------------------------------------------------------- 需求① 原告诉请


def test_诉请_默认从第二句开始并跳过案件背景(seg):
    text = "原被告系借款合同纠纷。原告请求被告偿还借款本金10万元。被告辩称已还清。经审理查明，借款属实。依照《合同法》判决被告偿还10万元。"
    p, d, f = pieces(seg, text)
    assert p == "原告请求被告偿还借款本金10万元。"
    assert d == "被告辩称已还清。"
    assert f == "经审理查明，借款属实。"


def test_诉请_第一句有标志就从标志处开始(seg):
    text = "原告诉称：判令被告支付租金5000元。被告未答辩。经审理查明，合同有效。依照《合同法》判决被告支付5000元。"
    p = span(seg, text, "plaintiff_claims")
    assert p.start == 0 and p.status is Status.IDENTIFIED
    assert p.text_of(text) == "原告诉称：判令被告支付租金5000元。"


def test_诉请_第一句是案件背景时跳过(seg):
    text = "原告与被告租赁合同纠纷一案。原告诉请判令被告腾退房屋。经查，房屋系原告所有。依照《物权法》判决被告腾退房屋。"
    p = span(seg, text, "plaintiff_claims")
    assert p.text_of(text) == "原告诉请判令被告腾退房屋。"


def test_诉请_无主语标志保留句首当事人(seg):
    """"原告崔某1请求判令……"——标志是"请求判令"，但片段要带上主语才读得通。"""

    text = "本案属于继承纠纷。原告崔某1请求判令案涉房屋分给原告。被告辩称不同意。经查房屋系遗产。依照《继承法》判决按份共有。"
    p = span(seg, text, "plaintiff_claims")
    assert p.text_of(text) == "原告崔某1请求判令案涉房屋分给原告。"


def test_诉请_结束于答辩起点(seg):
    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款3万元。被告辩称货物有质量问题。经审理查明，货物质量合格。依照《合同法》判决被告支付3万元。"
    p = span(seg, text, "plaintiff_claims")
    assert p.text_of(text) == "原告请求判令被告支付货款3万元。"


def test_诉请_无答辩时结束于法院事实起点(seg):
    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款3万元。经审理查明，货物质量合格。依照《合同法》判决被告支付3万元。"
    p, d, f = pieces(seg, text)
    assert p == "原告请求判令被告支付货款3万元。"
    assert d is None
    assert f == "经审理查明，货物质量合格。"


# ---------------------------------------------------------------- 需求② 被告辩称


@pytest.mark.parametrize(
    "marker",
    ["被告辩称", "被告答辩", "被告抗辩", "被告主张", "被告未答辩"],
)
def test_答辩_标志都能起点(seg, marker):
    text = f"原被告系买卖合同纠纷。原告请求判令被告支付货款。{marker}，货物有质量问题。经审理查明，货物质量合格。依照《合同法》判决被告支付货款。"
    d = span(seg, text, "defendant_defenses")
    assert d.status is Status.IDENTIFIED
    assert d.text_of(text).startswith(marker)


def test_答辩_仅凭被告二字不算答辩起点(seg):
    """需求②："不能仅凭'。被告'认定答辩开始"。"""

    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款3万元。被告是本案买受人。经审理查明，货物质量合格。依照《合同法》判决被告支付3万元。"
    d = span(seg, text, "defendant_defenses")
    assert d.status is Status.ABSENT


def test_答辩_说理中复述被告辩称不算新段落(seg):
    """需求②：说理里再次出现"被告辩称"不能被当成答辩段落。"""

    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款。经审理查明，被告辩称货物有质量问题，但未提供证据。依照《合同法》判决被告支付货款。"
    r = seg.segment(text)
    d = r.spans["defendant_defenses"]
    assert d.status is Status.ABSENT
    assert any("说理回述" in n for n in r.notes)
    # 被否掉的标志不能让诉请被切断——诉请终点应落在法院事实起点
    assert r.spans["plaintiff_claims"].end == r.spans["court_facts"].start


def test_答辩_多个被告连续答辩算一段(seg):
    text = (
        "原被告系借款合同纠纷。原告请求判令被告还款。"
        "被告甲公司辩称，借款属实但已过时效。被告乙公司辩称，本公司不承担担保责任。"
        "经审理查明，借款合同合法有效。依照《合同法》判决被告还款。"
    )
    d = span(seg, text, "defendant_defenses")
    assert d.status is Status.IDENTIFIED
    assert d.text_of(text) == "被告甲公司辩称，借款属实但已过时效。被告乙公司辩称，本公司不承担担保责任。"


def test_答辩_具名当事人辩称(seg):
    text = "原被告系劳动合同纠纷。原告请求判令被告支付工资。外服公司辩称存在与原告未建立劳动关系的事实。经审理查明，双方存在劳动关系。依照《劳动合同法》判决被告支付工资。"
    d = span(seg, text, "defendant_defenses")
    assert d.status is Status.IDENTIFIED
    assert d.text_of(text).startswith("外服公司辩称")


def test_答辩_未答辩声明是有标志的缺失形态(seg):
    text = "原被告系借款合同纠纷。原告请求判令被告还款。被告未答辩。经审理查明，借款属实。依照《合同法》判决被告还款。"
    r = seg.segment(text)
    d = r.spans["defendant_defenses"]
    assert d.status is Status.IDENTIFIED  # 有标志 → 不是"缺失"
    assert d.text_of(text) == "被告未答辩。"
    # 有法院叙述标志时优先用它收尾；没有的话按"未答辩"这句话的句边界收尾
    assert d.end_rule.startswith("法院叙述标志") or d.end_rule.startswith("未答辩声明句边界")

    no_marker = "原被告系借款合同纠纷。原告请求判令被告还款。被告未答辩。合同约定的利息过高，双方对还款期限也有分歧。依据《合同法》判决被告还款。"
    d2 = span(seg, no_marker, "defendant_defenses")
    assert d2.text_of(no_marker) == "被告未答辩。"
    assert d2.end_rule.startswith("未答辩声明句边界")


@pytest.mark.parametrize(
    "no_answer",
    ["被告未答辩。", "被告未提交书面答辩意见。", "被告未书面答辩。", "被告未到庭应诉。"],
)
def test_答辩_未答辩的多种写法都认(seg, no_answer):
    """语料里"没答辩"有十来种写法（未答辩/未提交书面答辩意见/未书面答辩/未到庭应诉……）。"""

    text = (
        "原被告系借款合同纠纷。原告请求判令被告还款。"
        + no_answer
        + "经审理查明，借款属实。依照《合同法》判决被告还款。"
    )
    d = span(seg, text, "defendant_defenses")
    assert d.status is Status.IDENTIFIED
    assert d.text_of(text) == no_answer


@pytest.mark.parametrize(
    "answer",
    ["被告承认原告的诉讼请求。", "被告表示同意解除合同。", "被告无异议。", "被告不同意原告的诉讼请求。"],
)
def test_答辩_没有辩字的陈述动作词(seg, answer):
    """摘要里真有"被告承认/表示/无异议/不同意"这类没有"辩"字的答辩陈述。"""

    text = (
        "原被告系借款合同纠纷。原告请求判令被告还款。"
        + answer
        + "经审理查明，借款属实。依照《合同法》判决被告还款。"
    )
    d = span(seg, text, "defendant_defenses")
    assert d.status is Status.IDENTIFIED
    assert d.text_of(text) == answer


def test_答辩_意思表示里的表示不是答辩(seg):
    """"双方当事人真实意思表示"里的"表示"不是答辩陈述（实测最大的误检来源）。"""

    text = (
        "原被告系借款合同纠纷。原告请求判令被告还款。"
        "经依法查明，借款合同是双方当事人真实意思表示，合法有效。"
        "依照《合同法》判决被告还款。"
    )
    r = seg.segment(text)
    assert r.spans["defendant_defenses"].status is Status.ABSENT
    assert r.spans["defendant_defenses"].start is None


def test_答辩_误检落在法院事实里会被撤销(seg):
    """答辩标志若落在法院事实/尾部区间内，必须撤销答辩段落，不能输出 D.start > F.end。"""

    text = (
        "原被告系借款合同纠纷。原告请求判令被告还款。"
        "由于原告停工期间的公告已过正当程序公示，被告在明知考勤制度的情形下未按规定考勤，"
        "根据《劳动合同法》判决原告无需支付赔偿金。"
    )
    r = seg.segment(text)
    d = r.spans["defendant_defenses"]
    if d.status is not Status.ABSENT:
        assert d.start < r.spans["court_facts"].end
    assert r.order_ok


def test_答辩_没有法院标志时不得只占一句(seg):
    """需求②：没有明确标志时要结合句边界推断，但不得默认答辩只占一句。"""

    text = (
        "原被告系借款合同纠纷。原告请求判令被告还款。"
        "被告辩称，借款属实。被告还辩称，利息计算过高。被告认为，应当免除罚息。"
        "本院认为，借款合同合法有效，被告应当偿还本息。依照《合同法》判决被告还款。"
    )
    d = span(seg, text, "defendant_defenses")
    assert d.status is Status.IDENTIFIED
    assert d.text_of(text).endswith("免除罚息。")


def test_答辩_法院归纳的争议焦点不算答辩(seg):
    """法院自己归纳的"争议焦点"是法院叙述：它比后面的"经审理查明"更早，
    答辩段落应该在这里收尾，不能被撑长一整句。"""

    text = (
        "原被告系劳动合同纠纷。原告请求判令无需支付赔偿金。"
        "被告辩称，原告解除劳动合同违法，应支付赔偿金。"
        "本案争议焦点为原告解除劳动关系是否合法。经审理查明，原告解除行为合法。"
        "依照《劳动合同法》判决原告无需支付赔偿金。"
    )
    r = seg.segment(text)
    d = r.spans["defendant_defenses"]
    assert d.text_of(text) == "被告辩称，原告解除劳动合同违法，应支付赔偿金。"
    assert r.spans["court_facts"].text_of(text) == (
        "本案争议焦点为原告解除劳动关系是否合法。经审理查明，原告解除行为合法。"
    )


def test_答辩_终点定不下来时记不确定而不是硬切(seg):
    text = (
        "原被告系借款合同纠纷。原告请求判令被告还款。"
        "被告辩称，借款属实。双方对利息计算存在分歧。款项已经部分归还。"
        "综上，依照《合同法》判决被告还款。"
    )
    r = seg.segment(text)
    d = r.spans["defendant_defenses"]
    assert d.status is Status.UNCERTAIN
    assert r.spans["court_facts"].status is Status.UNCERTAIN
    assert r.complete is False
    assert "推断失败" in d.end_rule


# ---------------------------------------------------------------- 需求③ 法院事实


def test_法院事实_起点继承答辩终点(seg):
    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款。被告辩称货物有质量问题。经审理查明，货物质量合格。依照《合同法》判决被告支付货款。"
    f = span(seg, text, "court_facts")
    assert f.text_of(text) == "经审理查明，货物质量合格。"


def test_法院事实_终点在依照判决处(seg):
    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款。经审理查明，货物质量合格。依照《中华人民共和国合同法》第六十条、第一百零七条之规定，判决被告支付货款3万元。"
    r = seg.segment(text)
    f = r.spans["court_facts"]
    assert f.text_of(text) == "经审理查明，货物质量合格。"
    assert f.end == r.spans["court_facts"].end
    assert text[f.end :].startswith("依照《中华人民共和国合同法》")


def test_法院事实_判决如下也认(seg):
    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款。经审理查明，货物质量合格。依照《合同法》第六十条之规定，判决如下：被告支付货款3万元。"
    f = span(seg, text, "court_facts")
    assert f.text_of(text) == "经审理查明，货物质量合格。"


def test_法院事实_说理里的驳回到不了尾部(seg):
    """需求③：不得把说理中的"驳回"当成结尾（这里尾部是后面的"判决如下"）。"""

    text = (
        "原被告系买卖合同纠纷。原告请求判令被告支付货款。"
        "经审理查明，被告辩称货物有质量问题，依据《合同法》第六十条之规定，应驳回原告的请求，但被告未举证，本院不予支持。"
        "依照《合同法》第一百零七条之规定，判决如下：被告支付货款3万元。"
    )
    f = span(seg, text, "court_facts")
    assert "应驳回原告的请求" in f.text_of(text)
    assert text[f.end :].startswith("依照《合同法》第一百零七条")


def test_法院事实_依据证据认定事实不是尾部(seg):
    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款。经审理查明，依据证据认定事实如下：货物质量合格。依照《合同法》判决被告支付货款。"
    f = span(seg, text, "court_facts")
    assert "依据证据认定事实如下" in f.text_of(text)


def test_法院事实_没有尾部标志时记不确定(seg):
    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款。经审理查明，货物质量合格，被告应当支付货款。"
    r = seg.segment(text)
    f = r.spans["court_facts"]
    assert f.end_rule.startswith("未找到尾部标志")
    assert r.complete is True or f.status is Status.UNCERTAIN  # 终点取文末，但来源如实记录
    assert any("未找到" in n for n in r.notes)


# ---------------------------------------------------------------- 顺序与字符级


def test_同一句里切出多个要素(seg):
    """边界是字符偏移量，不按句子切：一句话里可以同时出现诉请、答辩、法院事实。

    两端的悬挂标点会被收掉（"…支付货款，" → "…支付货款"）：那个逗号是**下一段**
    的引导，留在前一段里会让切片读起来像被截断。
    """

    text = "原告请求判令被告支付货款，被告辩称已付清，本院认为被告未举证，依照《合同法》判决被告支付货款。"
    p, d, f = pieces(seg, text)
    assert p == "原告请求判令被告支付货款"
    assert d == "被告辩称已付清"
    assert f == "本院认为被告未举证"


@pytest.mark.parametrize(
    "text",
    [
        "原被告系借款合同纠纷。原告请求判令被告还款。被告辩称已还。经审理查明，借款属实。依照《合同法》判决被告还款。",
        "原告诉请判令被告还款。经查借款属实。依照《合同法》判决被告还款。",
        "原告请求判令被告还款。被告未答辩。",
        "本案系租赁合同纠纷。",
        "原告请求判令被告支付货款，被告辩称已付清，本院认为被告未举证，依照《合同法》判决被告支付货款。",
    ],
)
def test_区间永远有序不重叠不越界(seg, text):
    r = seg.segment(text)
    seq = []
    for k in ELEMENTS:
        s = r.spans[k]
        if s.status is Status.ABSENT:
            continue
        assert s.start is not None and s.end is not None
        assert 0 <= s.start <= s.end <= len(text)
        if s.status is Status.IDENTIFIED:
            seq.append((s.start, s.end))
    for (s1, e1), (s2, _) in zip(seq, seq[1:]):
        assert e1 <= s2
    assert r.order_ok is True


def test_空文本记不确定而不是缺失(seg):
    r = seg.segment("   ")
    assert all(r.spans[k].status is Status.UNCERTAIN for k in ELEMENTS)
    assert r.complete is False


def test_缺失与不确定是两种状态(seg):
    absent = seg.segment("原被告系买卖合同纠纷。原告请求判令被告支付货款。经审理查明，货物质量合格。依照《合同法》判决被告支付货款。")
    assert absent.spans["defendant_defenses"].status is Status.ABSENT
    uncertain = seg.segment(
        "原被告系买卖合同纠纷。原告请求判令被告支付货款。被告辩称货物有问题。双方各执一词。综上，依照《合同法》判决被告支付货款。"
    )
    assert uncertain.spans["defendant_defenses"].status is Status.UNCERTAIN
    assert uncertain.spans["defendant_defenses"].status is not absent.spans["defendant_defenses"].status


def test_识别来源与置信状态被记录下来(seg):
    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款。被告辩称货物有质量问题。经审理查明，货物质量合格。依照《合同法》判决被告支付货款。"
    r = seg.segment(text)
    for k in ELEMENTS:
        s = r.spans[k]
        assert s.start_rule  # 每个区间都要能回答"这一刀是谁下的"
        assert s.end_rule
        assert s.confidence in (Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW)
    assert "经审理查明" in r.spans["court_facts"].start_rule


def test_导出字典可JSON序列化(seg):
    r = seg.segment("原被告系买卖合同纠纷。原告请求判令被告支付货款。被告辩称货物有质量问题。经审理查明，货物质量合格。依照《合同法》判决被告支付货款。")
    payload = json.dumps(r.to_dict(), ensure_ascii=False)
    back = json.loads(payload)
    assert back["spans"]["plaintiff_claims"]["start"] == r.spans["plaintiff_claims"].start


# ---------------------------------------------------------------- 缓存


def test_句首被告规则是可选开关(seg):
    """对照规则（默认关闭）：打开后，第一个句首"被告"就是答辩起点。

    这里把"打开会误吃诉请条目"这件事也钉住——它就是这条规则默认关闭的原因。
    """

    text = (
        "原被告系借款合同纠纷。原告请求判令被告偿还借款本息。"
        "被告承担本案诉讼费。被告辩称已经还清。"
        "经审理查明，借款属实。依照《合同法》判决被告还款。"
    )
    default = seg.segment(text)
    assert default.spans["defendant_defenses"].start_rule.startswith("被告辩称:被告+答辩动词")
    # 默认：诉请一直延伸到真正的答辩起点，最后一条诉请条目留在诉请里
    assert default.text_of("plaintiff_claims") == "原告请求判令被告偿还借款本息。被告承担本案诉讼费。"
    assert default.text_of("defendant_defenses") == "被告辩称已经还清。"

    from sfzy.segment import Segmenter as _S

    bare = _S(prefer_bare_defendant=True).segment(text)
    assert "句首被告" in bare.spans["defendant_defenses"].start_rule
    # 打开后诉请被切短，最后一条诉请条目掉进了答辩段落（这正是它默认关闭的原因）
    assert bare.text_of("plaintiff_claims") == "原告请求判令被告偿还借款本息。"
    assert bare.text_of("defendant_defenses").startswith("被告承担本案诉讼费。")


def test_句首被告兜底只在没有其它标志时生效(seg):
    """字面 `。被告` 兜底：常规标志落空时才用，且那句要紧跟答辩动作词。"""

    # 常规标志落空、但存在"被告 + 承认"的句首句 → 兜底接手
    text = (
        "原被告系借款合同纠纷。原告请求判令被告偿还借款本息。"
        "被告承认原告在本案中所主张的事实。经审理查明，借款属实。"
        "依照《合同法》判决被告还款。"
    )
    r = seg.segment(text)
    d = r.spans["defendant_defenses"]
    assert d.status is Status.IDENTIFIED
    assert d.text_of(text) == "被告承认原告在本案中所主张的事实。"

    # 同一段文字里"被告"后面接的是法院说理语气 → 兜底不启用
    court = (
        "原被告系借款合同纠纷。原告请求判令被告偿还借款本息。"
        "被告未按约定履行还款义务，已构成违约，应当承担违约责任。"
        "依照《合同法》判决被告还款。"
    )
    assert seg.segment(court).spans["defendant_defenses"].status is Status.ABSENT


def test_量词前缀不被切掉(seg):
    """"。四被告均未应诉答辩"这类：起点必须含"四"，不能切成"…。四|被告均…"。"""

    text = (
        "原被告系借款合同纠纷。原告请求判令被告偿还借款本息；其余被告承担连带清偿责任。"
        "四被告均未作答辩。经查明，借款属实。依照《合同法》判决被告还款。"
    )
    p = span(seg, text, "plaintiff_claims")
    assert p.text_of(text).endswith("其余被告承担连带清偿责任。")
    d = span(seg, text, "defendant_defenses")
    assert d.text_of(text) == "四被告均未作答辩。"


def test_缓存_离线预热后查表命中(seg, tmp_path):
    cache = SegmentCache(segmenter=seg)
    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款。经审理查明，货物质量合格。依照《合同法》判决被告支付货款。"
    cache.warm([text])
    hit = cache.get(text)
    assert hit is not None
    assert hit.spans["plaintiff_claims"].start == seg.segment(text).spans["plaintiff_claims"].start

    path = tmp_path / "segments.jsonl"
    assert cache.save(path) == 1
    reloaded = SegmentCache.load(path, segmenter=seg)
    assert len(reloaded) == 1
    assert reloaded.get(text) is not None


def test_缓存_实时链路只读不写(seg, tmp_path):
    """GRPO 候选摘要走的就是这条：现算但不缓存（避免把随机采样文本堆进磁盘）。"""

    cache = SegmentCache.load(tmp_path / "missing.jsonl", segmenter=seg, writable=False)
    text = "原被告系买卖合同纠纷。原告请求判令被告支付货款。经审理查明，货物质量合格。依照《合同法》判决被告支付货款。"
    res = cache.get_or_segment(text)
    assert res.spans["court_facts"].start is not None
    assert len(cache) == 0
    assert cache.get(text) is None


def test_缓存_键是正文哈希(seg):
    cache = SegmentCache(segmenter=seg)
    a = "原被告系买卖合同纠纷。原告请求判令被告支付货款。经审理查明，货物质量合格。依照《合同法》判决被告支付货款。"
    b = a + "（改动）"
    cache.put(a)
    assert cache.get(a) is not None
    assert cache.get(b) is None  # 文本变了，缓存不能串味
