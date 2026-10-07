"""六要素事实一致性 Judge 的验收测试（全部不加载模型）。

模型本身的判断力要靠 `scripts/judge_test.py` 在固定案例集上测，
这里只钉住三件"写错了不报错、只是分数悄悄变错"的事：

  1. **空字段规则**：省略不是事实错误 —— 这是整套指标的立场，写反了就全错
  2. **权重与最低分**：案由不参与最低分（它只有几个字，蒙对概率太高）
  3. **缓存与批量**：同一篇文档在 GRPO 里只允许提取一次

    python -m pytest tests/test_judge.py -q
"""

from __future__ import annotations

import pytest

from sfzy.judge.judge import FactConsistencyJudge, parse_six_json
from sfzy.judge.schema import (
    DEFAULT_WEIGHTS,
    ELEMENTS,
    MAX_SCORE,
    MIN_SCORE_ELEMENTS,
    SixElements,
    aggregate,
    aggregate_coverage,
    empty_field_rule,
    summarize_scores,
)


# ---------------------------------------------------------------- 权重

def test_权重之和为一():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


def test_裁判结果权重最高():
    """写反了整份摘要就废了，所以它必须是最大的一项。"""
    assert DEFAULT_WEIGHTS["judgment_result"] == max(DEFAULT_WEIGHTS.values())
    assert DEFAULT_WEIGHTS["judgment_result"] == 0.30


def test_最低分不含案由():
    assert "case_type" not in MIN_SCORE_ELEMENTS
    assert set(MIN_SCORE_ELEMENTS) == set(ELEMENTS) - {"case_type"}


# ---------------------------------------------------------------- 空字段规则

def test_空字段_两边都空给满分():
    assert empty_field_rule("", "") == MAX_SCORE


def test_空字段_原文有候选没写也给满分():
    """★ 整套指标的立场：**省略不是事实错误**。这一条写反，
    奖励就变成覆盖率了，而覆盖率是另一个指标。"""
    assert empty_field_rule("被告应支付租金48000元", "") == MAX_SCORE


def test_空字段_原文没有但候选写了要调模型():
    """不直接判 0：六要素提取器本身有噪声，"原文没有这一项"很可能是
    被归到别的要素去了，那是提取器的错，不该算到策略头上。"""
    assert empty_field_rule("", "被告支付了违约金") is None


def test_空字段_两边都有要调模型():
    assert empty_field_rule("甲方", "乙方") is None


def test_空字段_只有空白也算空():
    assert empty_field_rule("  \n ", "\t") == MAX_SCORE


# ---------------------------------------------------------------- 聚合

def test_聚合_按权重加权():
    raw = {name: MAX_SCORE for name in ELEMENTS}
    r = aggregate(raw)
    assert r.weighted_reward == pytest.approx(1.0)
    assert r.min_element_score == 1.0


def test_聚合_全零():
    assert aggregate({name: 0 for name in ELEMENTS}).weighted_reward == 0.0


def test_聚合_归一化除以四():
    raw = {name: 2 for name in ELEMENTS}
    r = aggregate(raw)
    assert all(v == pytest.approx(0.5) for v in r.scores.values())
    assert r.weighted_reward == pytest.approx(0.5)


def test_聚合_结果分零分时整条被拖到什么程度():
    """裁判结果写反（0 分）、其余全对：0.70 分。这就是"结果错但过程对"
    的实际代价 —— 面试时可以直接讲这个数。"""
    raw = {name: MAX_SCORE for name in ELEMENTS}
    raw["judgment_result"] = 0
    r = aggregate(raw)
    assert r.weighted_reward == pytest.approx(0.70)
    assert r.judgment_result_score == 0.0
    assert r.min_element_score == 0.0


def test_聚合_缺项必须报错():
    """静默用 0 补齐会让奖励凭空掉一截，而且没人发现。"""
    with pytest.raises(ValueError, match="缺少要素分数"):
        aggregate({"case_type": 4})


def test_聚合_记录来源():
    raw = {name: 4 for name in ELEMENTS}
    src = {name: "judge" for name in ELEMENTS}
    src["case_type"] = "empty_rule"
    r = aggregate(raw, sources=src)
    assert r.sources["case_type"] == "empty_rule"


# ---------------------------------------------------------------- 日志统计

def test_统计_给出各档比例且加起来是一():
    results = [aggregate({n: i % 5 for n in ELEMENTS}) for i in range(5)]
    stats = summarize_scores(results)
    total = sum(stats[f"ratio_score_{lv}"] for lv in range(MAX_SCORE + 1))
    assert total == pytest.approx(1.0)
    assert "mean_fact_reward" in stats
    for name in ELEMENTS:
        assert f"mean_{name}_score" in stats


def test_统计_空输入不报错():
    assert summarize_scores([]) == {}


# ---------------------------------------------------------------- JSON 解析

GOOD = ('{"case_type":"借款合同纠纷","plaintiff_claims":"偿还借款",'
        '"defendant_defenses":"不同意","court_facts":"借款属实",'
        '"legal_basis":"合同法","judgment_result":"判决还款"}')


def test_解析_干净的JSON():
    els = parse_six_json(GOOD)
    assert els.case_type == "借款合同纠纷" and els.judgment_result == "判决还款"


def test_解析_带markdown围栏():
    assert parse_six_json(f"```json\n{GOOD}\n```").case_type == "借款合同纠纷"


def test_解析_前后有多余解释():
    assert parse_six_json(f"好的，结果如下：\n{GOOD}\n以上。").case_type == "借款合同纠纷"


def test_解析_中文引号也能救回来():
    """小模型经常输出中文引号，json.loads 直接失败，要退到正则。"""
    raw = ('{“case_type”：“继承纠纷”，“plaintiff_claims”：“代位继承”，'
           '“defendant_defenses”：“依法判决”，“court_facts”：“共同生活”，'
           '“legal_basis”：“继承法”，“judgment_result”：“按份共有”}')
    assert parse_six_json(raw).case_type == "继承纠纷"
    # 字段之间是全角逗号，取值时不能把 "，" 带进值里（实测踩过）
    assert parse_six_json(raw).judgment_result == "按份共有"
    assert parse_six_json(raw).plaintiff_claims == "代位继承"


def test_解析_字段值是数组():
    """0.5B 实测不会老实输出 `"key": "字符串"` —— 它爱写成数组，
    因为六个要素天然是"多条并列"。不接住的话六个要素会全空。"""
    raw = ('{"case_type": "借款合同纠纷", '
           '"plaintiff_claims": ["要求被告偿还借款本金及利息", "要求承担保证责任"], '
           '"court_facts": ["借款属实"], "judgment_result": "判决还款"}')
    els = parse_six_json(raw)
    assert els.plaintiff_claims == "要求被告偿还借款本金及利息；要求承担保证责任"
    assert els.judgment_result == "判决还款"


def test_解析_彻底失败返回空要素():
    """空要素是安全值：走空字段规则会得满分或转 Judge，不会伪造错误分数。"""
    els = parse_six_json("我无法完成这个任务。")
    assert all(els.get(n) == "" for n in ELEMENTS)


def test_解析_多余字段被忽略():
    els = parse_six_json('{"case_type":"a","无关字段":"x"}')
    assert els.case_type == "a" and els.court_facts == ""


def test_解析_null当空串():
    assert parse_six_json('{"case_type": null}').case_type == ""


def test_解析_占位词归一成空串():
    """实测模型对"不存在的要素"写的是"无"，而 empty_field_rule 只把空白当空。
    不归一就会变成"原文里有这一项"，再叠加规则三 → 又是反向梯度。"""
    for placeholder in ("无", "无。", "暂无", "None", "n/a", "-"):
        assert parse_six_json(f'{{"court_facts": "{placeholder}"}}').court_facts == ""


def test_解析_数组里的占位词被丢掉():
    raw = '{"court_facts": ["借款属实", "无"]}'
    assert parse_six_json(raw).court_facts == "借款属实"


# ---------------------------------------------------------------- 缓存与批量

class _FakeJudge(FactConsistencyJudge):
    """不加载模型：记录哪些 pair 真的被"送进模型"，并按规则表返回分数。"""

    def __init__(self, score_map=None, extract_map=None,
                 tasks=("fact_consistency",), coverage_weights=None):
        self.weights = dict(DEFAULT_WEIGHTS)
        self.coverage_weights = dict(coverage_weights or DEFAULT_WEIGHTS)
        self.tasks = tuple(tasks)
        self._doc_cache = {}
        self._ref_cache = {}
        self.min_document_elements = 0          # 假裁判不检查提取质量
        self.doc_fallback = True
        self.stats = {}
        self.runtime = type("R", (), {"stats": {}})()
        self.sent_pairs = []        # 事实一致性真正送进模型的 pair
        self.sent_coverage = []     # 覆盖率真正送进模型的 pair
        self.extract_calls = 0
        self._score_map = score_map or {}
        self._extract_map = extract_map or {}

    def extract_six_elements_batch(self, texts):
        self.extract_calls += 1
        return [self._extract_map.get(t, SixElements(case_type="借款合同纠纷"))
                for t in texts]

    def _score_requests(self, requests):        # noqa: D102 — 见基类
        for req in requests:
            if req.task == "element_coverage":
                self.sent_coverage.append((req.element, req.left, req.right))
            else:
                self.sent_pairs.append((req.element, req.left, req.right))
        scores = [self._score_map.get(req.element, 4) for req in requests]
        return scores, [0.9] * len(requests), [[0.02] * 5 for _ in requests]


def test_文档要素只提取一次():
    """GRPO 里同一篇文书配 8 个候选，文档要素必须只提取一次。
    重复提取 = 每步多花 G 倍的钱，而且完全不报错。"""
    j = _FakeJudge()
    for _ in range(8):
        j.document_elements("同一篇判决书")
    assert j.extract_calls == 1
    assert len(j._doc_cache) == 1


def test_不同文档各自提取():
    j = _FakeJudge()
    j.document_elements("文书甲")
    j.document_elements("文书乙")
    j.document_elements("文书甲")
    assert j.extract_calls == 2


def test_候选要素批量提取只调一次():
    j = _FakeJudge()
    j.judge_candidates("文书", ["候选1", "候选2", "候选3"])
    assert j.extract_calls == 2      # 文档一次 + 候选一批一次


def test_抽取prompt_v2_含逐字保真约束():
    """v1 的抽取器会把日期挪用/改写，导致正确摘要被判 0；v2/v3 加了硬约束。"""
    from sfzy.judge.prompts import EXTRACT_PROMPT_VERSION, build_extract_messages

    assert EXTRACT_PROMPT_VERSION
    system = build_extract_messages("x")[0]["content"]
    for kw in ("逐字一致", "严禁改写", "日期"):
        assert kw in system
    # v3：程序事实优先 —— 本案"未答辩/未到庭"不能被仲裁阶段/别处的"辩称"顶掉
    for kw in ("未答辩", "未到庭", "本院查明"):
        assert kw in system


def test_原文与摘要用不同的抽取prompt():
    from sfzy.judge.prompts import (
        EXTRACT_SYSTEM,
        EXTRACT_SUMMARY_SYSTEM,
        build_extract_messages,
    )

    assert EXTRACT_SYSTEM != EXTRACT_SUMMARY_SYSTEM
    assert build_extract_messages("x", "document")[0]["content"] == EXTRACT_SYSTEM
    assert build_extract_messages("x", "summary")[0]["content"] == EXTRACT_SUMMARY_SYSTEM
    with pytest.raises(ValueError, match="未知的抽取类型"):
        build_extract_messages("x", "bogus")


def test_判官按类型选择抽取prompt():
    """document_elements 用原文 prompt；候选/参考摘要用摘要 prompt。"""
    from sfzy.judge.prompts import EXTRACT_SYSTEM, EXTRACT_SUMMARY_SYSTEM

    class _RT:
        def __init__(self):
            self.systems = []
            self.stats = {}

        def render(self, messages):
            self.systems.append(messages[0]["content"])
            return "prompt"

        def generate_batch(self, prompts, max_new_tokens=512):
            return ['{"case_type": "x", "court_facts": "y"}'] * len(prompts)

    rt = _RT()
    j = FactConsistencyJudge(runtime=rt, min_document_elements=0)
    j.document_elements("DOC")
    j.extract_six_elements("SUMMARY")          # 默认 kind="summary"
    assert rt.systems == [EXTRACT_SYSTEM, EXTRACT_SUMMARY_SYSTEM]


def test_上下文判定prompt_原文给六项摘要只给目标项():
    from sfzy.judge.prompts import ELEMENT_ZH, build_judge_messages_with_context

    doc = SixElements(case_type="劳动纠纷", court_facts="原文查明的事实",
                      legal_basis="原文法条")
    cand = SixElements(case_type="劳动纠纷", court_facts="摘要概括的事实",
                       legal_basis="摘要法条")
    msgs = build_judge_messages_with_context("court_facts", doc, cand)
    system, user = msgs[0]["content"], msgs[1]["content"]
    primary_block = user.split("【辅助参考")[0]
    assert "判分依据" in primary_block
    assert ELEMENT_ZH["court_facts"] in primary_block      # 判分依据是本项
    aux_block = user.split("【辅助参考")[1].split("【待评价摘要的要素】")[0]
    for name in ELEMENTS:
        if name != "court_facts":
            assert ELEMENT_ZH[name] in aux_block           # 其余五项都在辅助块
    assert ELEMENT_ZH["court_facts"] not in aux_block
    summary_block = user.split("【待评价摘要的要素】")[1]
    assert "court_facts" in summary_block
    assert "legal_basis" not in summary_block    # 摘要侧只给要判的那一项
    # 判分规则放在 system 里，user 只放材料，避免和 rubric 打架
    assert "判分规则" not in user
    assert "辅助参考" in system and "3 分" in system
    # 法律定性按"能否由原文整体推出"判断（案由/诉请/法条/结果体现同一定性即可）
    assert "法律定性" in system and "按 3 或 4 分" in system
    # 对"原告诉请/被告辩称/法院查明"这三项的一般性放宽：0 分门槛更严
    assert "0 分门槛" in system
    assert "建筑设备租赁" not in system      # 不是针对某个示例打的补丁


def test_开关开启时判定带上其它要素作辅助():
    class _RT:
        def __init__(self):
            self.rendered = []
            self.stats = {}

        def render(self, messages):
            self.rendered.append(messages[1]["content"])
            return "prompt"

        def score_digits_batch(self, prompts, digits=""):
            n = len(prompts)
            return [4] * n, [1.0] * n, [[1.0, 0, 0, 0, 0]] * n

    rt = _RT()
    j = FactConsistencyJudge(runtime=rt, use_element_context=True)
    doc = SixElements(court_facts="原文事实", legal_basis="《中华人民共和国劳动法》第七条")
    cand = SixElements(court_facts="摘要事实")
    j.judge_elements_batch(j.build_pairs(doc, cand, None))
    # 判 court_facts 时，原文的其它要素（legal_basis）也出现在 prompt 里
    assert any("《中华人民共和国劳动法》第七条" in p for p in rt.rendered)


def test_人工摘要作第二辅助_自评时忽略():
    from sfzy.judge.prompts import build_judge_messages_with_context

    doc = SixElements(court_facts="原文事实")
    cand = SixElements(court_facts="候选事实")
    ref = SixElements(court_facts="人工摘要事实")

    user = build_judge_messages_with_context(
        "court_facts", doc, cand, reference_elements=ref
    )[1]["content"]
    assert "辅助参考二" in user and "人工摘要事实" in user

    class _RT:
        def __init__(self):
            self.stats = {}

    j = FactConsistencyJudge(
        runtime=_RT(), use_element_context=True, use_reference_context=True
    )
    # 候选就是人工摘要本身（离线工具的 human 臂是自评）→ 不挂人工摘要辅助
    self_pairs = j.build_pairs(doc, ref, None, reference_elements=ref)
    assert all(getattr(p, "ref_elements", None) is None for p in self_pairs)
    # 候选不是人工摘要 → 挂上
    cand_pairs = j.build_pairs(doc, cand, None, reference_elements=ref)
    assert all(getattr(p, "ref_elements", None) is ref for p in cand_pairs)


def test_一次性六要素判定_prompt与解析():
    from sfzy.judge.judge import parse_six_scores
    from sfzy.judge.prompts import SIX_SHOT_JUDGE_SYSTEM, build_fact_six_messages

    cand = SixElements(case_type="劳动纠纷", court_facts="摘要事实")
    msgs = build_fact_six_messages("这是原文全文……", cand)
    assert msgs[0]["content"] == SIX_SHOT_JUDGE_SYSTEM
    user = msgs[1]["content"]
    assert "【裁判文书原文】" in user and "这是原文全文" in user
    assert "【待评价摘要的六要素】" in user and "court_facts" in user

    assert parse_six_scores('{"case_type": 3, "court_facts": 0}') == {
        "case_type": 3, "court_facts": 0,
    }
    assert parse_six_scores('```json\n{"court_facts": "4"}\n```') == {"court_facts": 4}

    class _RT:
        def __init__(self):
            self.stats = {}

        def render(self, messages):
            return "prompt"

        def generate_batch(self, prompts, max_new_tokens=512):
            return [
                '{"case_type":2,"plaintiff_claims":3,"defendant_defenses":4,'
                '"court_facts":1,"legal_basis":3,"judgment_result":4}'
            ] * len(prompts)

    j = FactConsistencyJudge(runtime=_RT())
    out = j.judge_fact_six_batch([("原文", SixElements(case_type="x", plaintiff_claims="y"))])
    assert out[0]["case_type"] == 2 and out[0]["court_facts"] == 1
    # 候选该项为空 → 空字段规则给满分 4（模型即使给了别的分也覆盖）
    assert out[0]["defendant_defenses"] == 4


def test_覆盖率pair走覆盖率prompt而不是事实一致性():
    """离线工具里 fact 和 coverage 的 pair 会合成一批送判；
    coverage 的 pair 必须走 COVERAGE_SYSTEM，左=参考、右=候选。"""
    from sfzy.judge.prompts import COVERAGE_SYSTEM, JUDGE_SYSTEM

    class _RT:
        def __init__(self):
            self.calls = []          # (system, user)
            self.stats = {}

        def render(self, messages):
            self.calls.append((messages[0]["content"], messages[1]["content"]))
            return "prompt"

        def score_digits_batch(self, prompts, digits=""):
            n = len(prompts)
            return [4] * n, [1.0] * n, [[1.0, 0, 0, 0, 0]] * n

    rt = _RT()
    j = FactConsistencyJudge(
        runtime=rt, tasks=("fact_consistency", "element_coverage")
    )
    doc = SixElements(court_facts="原文查明的事实")
    ref = SixElements(court_facts="参考摘要的事实")
    cand = SixElements(court_facts="候选摘要的事实")

    fact_pairs = j.build_pairs(doc, cand, None)
    cov_pairs, _present, _pre = j.build_coverage_pairs(ref, cand)
    j.judge_elements_batch(list(fact_pairs) + list(cov_pairs))

    systems = [s for s, _ in rt.calls]
    assert JUDGE_SYSTEM in systems              # 事实那一条
    assert COVERAGE_SYSTEM in systems           # 覆盖那一条（修复前这里是 JUDGE_SYSTEM）

    cov_user = next(u for s, u in rt.calls if s == COVERAGE_SYSTEM)
    assert "参考摘要要素" in cov_user and "参考摘要的事实" in cov_user
    assert "候选摘要要素" in cov_user and "候选摘要的事实" in cov_user


def test_一次性覆盖率判定_prompt():
    from sfzy.judge.prompts import (
        SIX_SHOT_COVERAGE_SYSTEM,
        build_coverage_six_messages,
    )

    msgs = build_coverage_six_messages("人工摘要全文", "候选摘要全文")
    assert msgs[0]["content"] == SIX_SHOT_COVERAGE_SYSTEM
    user = msgs[1]["content"]
    assert "【参考摘要（人工）】" in user and "人工摘要全文" in user
    assert "【候选摘要】" in user and "候选摘要全文" in user
    assert "case_type" in user and "judgment_result" in user


def test_抽取用独立的extract_runtime():
    """给 extract_runtime 时，抽取走它、判定仍走裁判模型。"""
    class _RT:
        def __init__(self, out, label):
            self.out, self.label, self.calls, self.stats = out, label, 0, {}

        def render(self, messages):
            return self.label

        def generate_batch(self, prompts, max_new_tokens=512):
            self.calls += 1
            return [self.out for _ in prompts]

    judge_rt = _RT("", "judge")
    extract_rt = _RT('{"case_type": "M"}', "extract")
    j = FactConsistencyJudge(runtime=judge_rt, extract_runtime=extract_rt)
    els, _modes = j._extract_once(["x"], 16)
    assert els[0].case_type == "M"
    assert extract_rt.calls == 1 and judge_rt.calls == 0


def test_抽取输入超预算时按头尾截断():
    """长判决书的抽取 prompt 一旦整段截断就会丢 system 指令、抽成原文片段；
    现在改成只把输入文本按头+尾压到预算内。"""
    class _RT:
        def __init__(self):
            self.seen = []
            self.stats = {}

        def render(self, messages):
            return messages[-1]["content"]

        def count_tokens(self, text):
            return len(text)

        def truncate_text(self, text, max_tokens, tail_ratio=0.5):
            self.seen.append((text, max_tokens))
            return text[:max_tokens]

        def generate_batch(self, prompts, max_new_tokens=512):
            return ['{"case_type": "X"}'] * len(prompts)

    rt = _RT()
    j = FactConsistencyJudge(runtime=rt, max_input_tokens=2000)
    j._extract_once(["文" * 5000], 16)
    assert rt.seen, "长输入没有被截断"
    budget = rt.seen[0][1]
    assert 0 < budget < 2000


def test_空字段不送进模型():
    """候选什么都没写时不该浪费一次前向。"""
    doc = SixElements(case_type="借款合同纠纷", court_facts="借款属实")
    cand = SixElements(case_type="借款合同纠纷", plaintiff_claims="请求还款",
                       judgment_result="判决还款")
    j = _FakeJudge(extract_map={"DOC": doc, "CAND": cand})
    j.judge_candidates("DOC", ["CAND"])
    sent_names = [name for name, _, _ in j.sent_pairs]
    assert sorted(sent_names) == ["case_type", "judgment_result", "plaintiff_claims"]
    # 两边都空 → 走规则给 4 分，不发请求
    assert "defendant_defenses" not in sent_names
    assert "court_facts" not in sent_names           # 原文有、候选省略 → 省略不算错


def test_要素抽空时拿整篇原文兜底():
    """★ 实测踩到的反向梯度：提取器漏了某个要素（写了"无"），而候选写对了，
    按需求规则三仍要调 Judge —— 但若把**空字符串**当原文要素喂进去，
    裁判只能判 0。于是规则的组合变成"提取器漏过的要素，写了得 0、不写反而得 4"，
    奖励在教模型删掉正确内容，而且不报错。

    修法：文档要素为空时，拿整篇原文去判。这样裁判才有东西可核对。
    """
    doc_el = SixElements(case_type="借款合同纠纷")        # 其余五项都没抽到
    cand_el = SixElements(case_type="借款合同纠纷",
                          defendant_defenses="被告蔡昌有不同意承担担保责任")
    full_doc = "……被告蔡昌有辩称：确实担保了，不同意承担担保责任。……"
    j = _FakeJudge(extract_map={"DOC": doc_el, "CAND": cand_el})
    j.doc_fallback = True
    sent = j.build_pairs(doc_el, cand_el, full_doc)
    name, doc_part, cand_part = [p for p in sent if p[0] == "defendant_defenses"][0]
    assert doc_part == full_doc, "抽空的要素必须拿整篇原文兜底"


def test_关掉兜底时退回原始行为():
    """兜底是个开关：关掉就完全按需求原文的规则走（空原文照样调 Judge）。"""
    doc_el = SixElements(case_type="借款合同纠纷")
    cand_el = SixElements(defendant_defenses="被告不同意承担担保责任")
    j = _FakeJudge()
    j.doc_fallback = False
    sent = j.build_pairs(doc_el, cand_el, "整篇原文……")
    doc_part = [p for p in sent if p[0] == "defendant_defenses"][0][1]
    assert doc_part == ""


def test_候选省略时不发请求():
    """文档有、候选没写 → 规则二给 4 分，不该浪费一次前向。"""
    doc_el = SixElements(court_facts="借款属实")
    cand_el = SixElements()
    j = _FakeJudge()
    sent = j.build_pairs(doc_el, cand_el, "原文")
    assert [p for p in sent if p[0] == "court_facts"][0][2] == ""


def test_批量打分对齐到候选():
    """6 要素 × N 候选的结果必须回到各自的候选上，不能错位。"""
    doc = SixElements(case_type="借款合同纠纷")
    cand = SixElements(case_type="借款合同纠纷", judgment_result="判决还款")
    j = _FakeJudge(score_map={"judgment_result": 0},
                   extract_map={"DOC": doc, "a": cand, "b": cand})
    results = j.judge_candidates("DOC", ["a", "b"], candidate_ids=["c1", "c2"])
    assert [r.candidate_id for r in results] == ["c1", "c2"]
    assert all(r.judgment_result_score == 0.0 for r in results)
    assert all(r.weighted_reward == pytest.approx(0.70) for r in results)


def test_单要素判定走同一条路径():
    j = _FakeJudge(score_map={"court_facts": 2})
    assert j.judge_element("court_facts", "原文", "候选") == 2
    assert j.judge_element("case_type", "", "") == MAX_SCORE     # 空字段规则


# ---------------------------------------------------------------- 关键要素覆盖率

def test_覆盖率聚合公式():
    raw = {name: 4 for name in ELEMENTS}
    both = ["court_facts", "judgment_result"]
    assert aggregate_coverage(raw, both) == pytest.approx(1.0)

    raw["judgment_result"] = 0
    # (0.25×1 + 0.30×0) / (0.25+0.30)
    assert aggregate_coverage(raw, both) == pytest.approx(0.25 / 0.55, abs=1e-5)


def test_覆盖率分母跟着参与要素走():
    """只算一个要素时，那一项自己的权重就是分母 —— 不能拿全局权重和去除。"""
    assert aggregate_coverage(
        {"case_type": 2}, ["case_type"]
    ) == pytest.approx(0.5)


def test_覆盖率没有要素参与时返回None():
    assert aggregate_coverage({}, []) is None


def test_覆盖率_参考没有的要素不参与():
    ref = SixElements(court_facts="借款48000元属实")          # 只抽到一项
    cand = SixElements(court_facts="借款48000元", judgment_result="判决还款")
    j = _FakeJudge(extract_map={"REF": ref, "CAND": cand}, tasks=("element_coverage",))
    res = j.judge_candidates("DOC", ["CAND"], reference="REF")[0]
    assert res.signals["element_coverage"] == pytest.approx(1.0)
    assert [p[0] for p in j.sent_coverage] == ["court_facts"]


def test_覆盖率_候选漏写该要素直接0分且不发请求():
    ref = SixElements(court_facts="借款48000元属实", judgment_result="判决还款")
    cand = SixElements(court_facts="借款48000元")             # 漏了裁判结果
    j = _FakeJudge(extract_map={"REF": ref, "CAND": cand}, tasks=("element_coverage",))
    res = j.judge_candidates("DOC", ["CAND"], reference="REF")[0]
    assert res.signals["element_coverage"] == pytest.approx(0.25 / 0.55, abs=1e-5)
    assert [p[0] for p in j.sent_coverage] == ["court_facts"]


def test_覆盖率_参考里一个要素都没有时返回None():
    j = _FakeJudge(
        extract_map={"REF": SixElements(), "CAND": SixElements(court_facts="x")},
        tasks=("element_coverage",),
    )
    res = j.judge_candidates("DOC", ["CAND"], reference="REF")[0]
    assert res.signals["element_coverage"] is None


def test_覆盖率_没传参考摘要直接报错():
    j = _FakeJudge(tasks=("element_coverage",))
    with pytest.raises(ValueError, match="reference"):
        j.judge_candidates("DOC", ["CAND"])


def test_覆盖率_参考摘要要素只抽一次():
    """同一个 prompt 的参考是固定的，每步只该抽候选。"""
    j = _FakeJudge(extract_map={"REF": SixElements(court_facts="x")},
                   tasks=("element_coverage",))
    j.judge_candidates("DOC", ["CAND"], reference="REF")
    first = j.extract_calls
    j.judge_candidates("DOC", ["CAND"], reference="REF")
    assert first == 2 and j.extract_calls == 3      # 第一次：候选+参考；第二次：只有候选
    assert len(j._ref_cache) == 1


def test_只开覆盖率时不抽原文要素():
    """只开覆盖率就不该为原文多付一次抽取 —— 抽取是这条路径上最贵的部分。"""
    j = _FakeJudge(extract_map={"REF": SixElements(court_facts="x")},
                   tasks=("element_coverage",))
    j.judge_candidates("DOC", ["CAND"], reference="REF")
    assert j._doc_cache == {}
    assert j.sent_pairs == []                      # 一致性那一路根本没跑


def test_两个任务共用一次候选抽取():
    """同时开两个 reward 时：原文 / 参考 / 候选三种要素各只抽一次。"""
    doc = SixElements(court_facts="借款48000元属实", judgment_result="判决还款")
    ref = SixElements(court_facts="借款48000元", judgment_result="判决还款")
    cand = SixElements(court_facts="借款48000元", judgment_result="判决还款")
    j = _FakeJudge(
        extract_map={"DOC": doc, "REF": ref, "CAND": cand},
        tasks=("fact_consistency", "element_coverage"),
    )
    results = j.judge_candidates("DOC", ["CAND"], reference="REF")
    assert j.extract_calls == 3                    # 文档 + 候选 + 参考，各一次
    assert set(results[0].signals) == {
        "fact_consistency", "fact_consistency_min_raw", "element_coverage",
    }
    # 两个任务的判定都发出去了，但走的是同一次批量前向
    assert [p[0] for p in j.sent_pairs] == ["court_facts", "judgment_result"]
    assert [p[0] for p in j.sent_coverage] == ["court_facts", "judgment_result"]


def test_事实门控信号_取六要素最小原始分():
    """fact_consistency_min_raw 供 reward 侧的事实硬门控使用。"""
    doc = SixElements(court_facts="借款属实")
    cand_bad = SixElements(court_facts="借款不属实")
    j = _FakeJudge(score_map={"court_facts": 0},
                   extract_map={"DOC": doc, "CAND": cand_bad})
    res = j.judge_candidates("DOC", ["CAND"])[0]
    assert res.signals["fact_consistency_min_raw"] == 0.0

    cand_ok = SixElements(court_facts="借款属实")
    j2 = _FakeJudge(extract_map={"DOC": doc, "CAND": cand_ok})
    assert j2.judge_candidates("DOC", ["CAND"])[0].signals[
        "fact_consistency_min_raw"] == 4.0


def test_覆盖率权重可单独配置():
    """覆盖率用自己的 element_weights，不和事实一致性共用一份。"""
    ref = SixElements(court_facts="借款属实", judgment_result="判决还款")
    cand = SixElements(court_facts="借款属实")     # 漏了裁判结果
    only_facts = {name: 0.0 for name in ELEMENTS}
    only_facts["court_facts"] = 1.0
    j = _FakeJudge(
        extract_map={"REF": ref, "CAND": cand},
        tasks=("element_coverage",),
        coverage_weights=only_facts,
    )
    res = j.judge_candidates("DOC", ["CAND"], reference="REF")[0]
    # 裁判结果权重为 0，漏写它不该影响分数
    assert res.signals["element_coverage"] == pytest.approx(1.0)
# ---------------------------------------------------------------- ChatGLM3 路径

class _NativeTokenizer:
    """自带生成循环的模型（ChatGLM3）用的假 tokenizer。"""

    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"        # 生成路径会临时改它，得先存在

    def __call__(self, text=None, return_tensors=None, add_special_tokens=False,
                 padding=False, truncation=False, max_length=None):
        ids = [5, 6, 7]
        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([[1] * len(ids)])}
        return type("E", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=True):
        return "生成结果"

    def encode(self, text, add_special_tokens=False):
        # 约定：数字 d 的 token id 是 15 + d（和真实 Qwen / ChatGLM3 一致）
        if len(text) == 1 and text.isdigit():
            return [15 + int(text)]
        return [5]


class _NativeModel:
    """模拟 ChatGLM3：有 stream_generate，forward 返回 logits。

    用它来钉住 `runtime.native` 分支的两件事：生成走逐条、打分走**无 padding**
    的单条前向。这条路径在本地没法用真的 6B 验证，所以用假模型把结构锁死。
    """

    def __init__(self, best_digit_index: int = 2):
        import torch

        self._torch = torch
        self.best = best_digit_index
        self.calls = []
        self.generate_called = False

    def parameters(self):
        import torch

        return iter([torch.zeros(1, requires_grad=True)])

    def eval(self):
        self.training = False
        return self

    def stream_generate(self, input_ids, max_new_tokens, do_sample=False):
        import torch

        self.generate_called = True
        cur = input_ids
        for _ in range(max_new_tokens):
            cur = torch.cat([cur, torch.tensor([[2]])], dim=1)
            yield cur

    def __call__(self, input_ids=None, attention_mask=None):
        import torch

        self.calls.append({"batch": input_ids.shape[0], "mask": attention_mask})
        vocab = 32
        logits = torch.full((input_ids.shape[0], 1, vocab), -10.0)
        logits[:, 0, 15 + self.best] = 10.0      # 分数 token id = 15 + 分数
        return type("O", (), {"logits": logits})()


def test_native模型_生成逐条走stream_generate():
    """ChatGLM3 不能批量生成（get_masks 假设没有 padding），必须逐条。"""
    from sfzy.judge.runtime import TorchRuntime

    model = _NativeModel()
    rt = TorchRuntime(model=model, tokenizer=_NativeTokenizer(), device="cpu")
    assert rt.native is True
    out = rt.generate_batch(["p1", "p2"], max_new_tokens=3)
    assert out == ["生成结果", "生成结果"]
    assert model.generate_called


def test_native模型_打分走无padding单条前向():
    """判定这一步不做生成，只读最后一个位置的 logits；
    逐条、无 padding，绕开 get_masks 对 padding 的假设。"""
    from sfzy.judge.runtime import TorchRuntime

    model = _NativeModel(best_digit_index=3)     # 期望 3 分
    rt = TorchRuntime(model=model, tokenizer=_NativeTokenizer(), device="cpu")
    scores, pmaxs, probs = rt.score_digits_batch(["a", "b"])
    assert scores == [3, 3]
    assert all(p > 0.99 for p in pmaxs)
    assert all(c["batch"] == 1 for c in model.calls), "必须是逐条，不能组批"
    assert all(c["mask"] is None for c in model.calls), "不能传 padding mask"


def test_adapter关闭上下文_会被进入():
    """用策略模型关掉 adapter 当裁判时，每次前向都必须进入那个上下文。"""
    from sfzy.judge.runtime import TorchRuntime

    entered = []
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        entered.append(1)
        yield

    model = _NativeModel()
    rt = TorchRuntime(model=model, tokenizer=_NativeTokenizer(), device="cpu", context=ctx)
    rt.score_digits_batch(["a", "b"])
    # 整批进出一次就够：上下文是"这一批都用关掉 adapter 的底座"，
    # 逐条进出只是徒增开销
    assert len(entered) == 1


# ---------------------------------------------------------------- GRPO 适配器

def test_适配器_按文书分组并摊回原位置():
    """GRPO 的 items 是扁平列表，同一篇文书的候选可能不连续。

    分组后分数必须摊回**原来的位置** —— 错位不报错，只是奖励悄悄换了对象。
    """
    from sfzy.judge.scorer import SixElementScorer

    doc_a = SixElements(case_type="借款合同纠纷")
    doc_b = SixElements(case_type="继承纠纷")
    cand = SixElements(case_type="借款合同纠纷", judgment_result="判决还款")
    j = _FakeJudge(extract_map={
        "文书甲": doc_a, "文书乙": doc_b,
        "候选1": cand, "候选2": cand, "候选3": cand, "候选4": cand, "候选5": cand,
    })
    items = [
        {"candidate": "候选1", "reference": "r", "source": "文书甲", "id": "c1"},
        {"candidate": "候选2", "reference": "r", "source": "文书甲", "id": "c2"},
        {"candidate": "候选3", "reference": "r", "source": "文书乙", "id": "c3"},
        {"candidate": "候选4", "reference": "r", "source": "文书甲", "id": "c4"},
        {"candidate": "候选5", "reference": "r", "source": "文书乙", "id": "c5"},
    ]
    signals = SixElementScorer(j).score_batch_signals(items)
    scores = [sig["fact_consistency"] for sig in signals]
    assert len(scores) == 5 and all(s is not None for s in scores)
    # 每篇文书：候选集中提取一次 + 文档提取一次 = 2 次；两篇 = 4 次。
    # 分组不成立（逐条调用）时这里会是 5 + 2 次。
    assert j.extract_calls == 4


def test_适配器_文档要素跨批只提取一次():
    """同一篇文书在 GRPO 的连续批次里反复出现，文档六要素必须命中缓存。"""
    from sfzy.judge.scorer import SixElementScorer

    j = _FakeJudge()
    scorer = SixElementScorer(j)
    item = {"candidate": "候选", "reference": "r", "source": "文书", "id": "c"}
    scorer.score_batch_signals([item])
    scorer.score_batch_signals([item])
    assert len(j._doc_cache) == 1


def test_适配器_汇总第十二节的统计量():
    """训练日志要的 mean_fact_reward / 各要素均值 / 0-4 各档比例都从这来。"""
    from sfzy.judge.scorer import SixElementScorer

    doc = SixElements(court_facts="被告向原告借款50000元")
    cand = SixElements(court_facts="被告向原告借款90000元")
    # 查明事实写错（0 分）→ 最低分 0；结果项两边都空 → 空字段规则给 4。
    j = _FakeJudge(score_map={"court_facts": 0},
                   extract_map={"文书": doc, "候选": cand})
    scorer = SixElementScorer(j)
    scorer.score_batch_signals(
        [{"candidate": "候选", "reference": "r", "source": "文书", "id": "c"}]
    )
    stats = scorer.summarize_last()
    assert "mean_fact_reward" in stats
    assert "mean_min_element_score" in stats
    assert "mean_judgment_result_score" in stats
    assert stats["mean_min_element_score"] == 0.0
    assert sum(stats[f"ratio_score_{lv}"] for lv in range(MAX_SCORE + 1)) == pytest.approx(1.0)


def test_适配器_单组失败只标None不中断训练():
    """一次提取失败不该毁掉整轮几小时的 run：标 None 交给组内均值补。"""
    from sfzy.judge.scorer import SixElementScorer

    class _BoomJudge(_FakeJudge):
        def judge_candidates(self, *a, **k):
            raise RuntimeError("显存抖动")

    scorer = SixElementScorer(_BoomJudge())
    signals = scorer.score_batch_signals(
        [{"candidate": "候选", "reference": "r", "source": "文书", "id": "c"}]
    )
    # 整组失败 → 该条没有任何信号（空字典），交给 trainer 按缺失补
    assert signals == [{}]
    assert scorer.last_errors and "RuntimeError" in scorer.last_errors[0]


# ---------------------------------------------------------------- 4-bit 裁判加载

def test_build_runtime_把4bit和设备传下去(monkeypatch):
    """裁判侧 4-bit 必须走 load_inference_model：device 翻译成 device_map，
    量化模型加载后不搬。写错的表现是 24GB 卡上直接抛
    ".to() is not supported for 4/8-bit bitsandbytes models"。"""
    import torch

    import sfzy.judge.runtime as R
    import sfzy.models.loader as L

    captured = {}

    class _FakeModel:
        def parameters(self):
            return iter([torch.zeros(1)])

        def eval(self):
            return self

    def fake_load(cfg, device="auto", gradient_checkpointing=False):
        captured["cfg"] = dict(cfg)
        captured["device"] = device
        return _FakeModel(), _NativeTokenizer(), "cuda:1"

    monkeypatch.setattr(L, "load_inference_model", fake_load)
    rt = R.build_runtime(
        "/path/Qwen2.5-7B-Instruct", device="cuda:1", dtype="bfloat16", load_in_4bit=True
    )
    assert captured["device"] == "cuda:1"
    assert captured["cfg"]["load_in_4bit"] is True
    assert captured["cfg"]["bnb_4bit_quant_type"] == "nf4"
    assert captured["cfg"]["bnb_4bit_compute_dtype"] == "bfloat16"
    # 用模型实际所在设备建运行时，避免输入张量送错卡
    assert rt.device == "cuda:1"


# ---------------------------------------------------------------- 贪婪生成的告警

class _FakeGenModel:
    """标准 HF 生成路径的假模型：只记录 generate() 收到的 kwargs。"""

    def __init__(self):
        import torch

        self._torch = torch
        self.gen_kwargs = None

    def parameters(self):
        import torch

        return iter([torch.zeros(1)])

    def eval(self):
        self.training = False
        return self

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        import torch

        self.gen_kwargs = kwargs
        extra = torch.zeros((input_ids.shape[0], 2), dtype=input_ids.dtype)
        return torch.cat([input_ids, extra], dim=1)


def test_贪婪生成_清掉模型自带的采样参数():
    """Qwen2.5 的 generation_config 自带 temperature/top_p/top_k。

    judge 走的是贪心（do_sample=False），不显式把它们传成 None 的话，
    transformers 会打印
        The following generation flags are not valid and may be ignored:
        ['temperature', 'top_p', 'top_k']
    这不是错误（参数确实被忽略），但每次判分都刷一行很干扰。这里钉住
    judge 的生成路径确实把它们清掉了。
    """
    from sfzy.judge.runtime import TorchRuntime

    model = _FakeGenModel()
    rt = TorchRuntime(model=model, tokenizer=_NativeTokenizer(), device="cpu")
    assert rt.native is False
    out = rt.generate_batch(["prompt"], max_new_tokens=2)
    assert out == ["生成结果"]
    assert model.gen_kwargs["do_sample"] is False
    assert model.gen_kwargs["temperature"] is None
    assert model.gen_kwargs["top_p"] is None
    assert model.gen_kwargs["top_k"] is None


def test_奖励侧与裁判侧的六要素键集合一致():
    """`rl/reward_terms.py` 为了不引入 torch，把六要素键名抄了一份。

    这份拷贝必须和 `judge/schema.py` 的 ELEMENTS 完全一致，否则配置里写的
    `element_weights` 会对着一套键校验、对另一套键聚合 —— 而且不报错。
    """
    from sfzy.judge.schema import ELEMENTS
    from sfzy.rl.reward_terms import FACT_ELEMENTS

    assert tuple(FACT_ELEMENTS) == tuple(ELEMENTS)
