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

    def __init__(self, score_map=None, extract_map=None):
        self.weights = dict(DEFAULT_WEIGHTS)
        self._doc_cache = {}
        self.min_document_elements = 0          # 假裁判不检查提取质量
        self.doc_fallback = True
        self.judge_variant = "spec"
        self.stats = {}
        self.runtime = type("R", (), {"stats": {}})()
        self.sent_pairs = []
        self.extract_calls = 0
        self._score_map = score_map or {}
        self._extract_map = extract_map or {}

    def extract_six_elements_batch(self, texts):
        self.extract_calls += 1
        return [self._extract_map.get(t, SixElements(case_type="借款合同纠纷"))
                for t in texts]

    def _judge_pairs_raw(self, pairs):
        self.sent_pairs.extend(list(pairs))
        scores = [self._score_map.get(name, 4) for name, _, _ in pairs]
        return scores, [0.9] * len(pairs), [[0.02] * 5 for _ in pairs]


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
    from sfzy.judge.scorer import FactConsistencyScorer

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
    scores = FactConsistencyScorer(j).score_batch(items)
    assert len(scores) == 5 and all(s is not None for s in scores)
    # 每篇文书：候选集中提取一次 + 文档提取一次 = 2 次；两篇 = 4 次。
    # 分组不成立（逐条调用）时这里会是 5 + 2 次。
    assert j.extract_calls == 4


def test_适配器_文档要素跨批只提取一次():
    """同一篇文书在 GRPO 的连续批次里反复出现，文档六要素必须命中缓存。"""
    from sfzy.judge.scorer import FactConsistencyScorer

    j = _FakeJudge()
    scorer = FactConsistencyScorer(j)
    item = {"candidate": "候选", "reference": "r", "source": "文书", "id": "c"}
    scorer.score_batch([item])
    scorer.score_batch([item])
    assert len(j._doc_cache) == 1


def test_适配器_汇总第十二节的统计量():
    """训练日志要的 mean_fact_reward / 各要素均值 / 0-4 各档比例都从这来。"""
    from sfzy.judge.scorer import FactConsistencyScorer

    doc = SixElements(court_facts="被告向原告借款50000元")
    cand = SixElements(court_facts="被告向原告借款90000元")
    # 查明事实写错（0 分）→ 最低分 0；结果项两边都空 → 空字段规则给 4。
    j = _FakeJudge(score_map={"court_facts": 0},
                   extract_map={"文书": doc, "候选": cand})
    scorer = FactConsistencyScorer(j)
    scorer.score_batch([{"candidate": "候选", "reference": "r", "source": "文书", "id": "c"}])
    stats = scorer.summarize_last()
    assert "mean_fact_reward" in stats
    assert "mean_min_element_score" in stats
    assert "mean_judgment_result_score" in stats
    assert stats["mean_min_element_score"] == 0.0
    assert sum(stats[f"ratio_score_{lv}"] for lv in range(MAX_SCORE + 1)) == pytest.approx(1.0)


def test_适配器_单组失败只标None不中断训练():
    """一次提取失败不该毁掉整轮几小时的 run：标 None 交给组内均值补。"""
    from sfzy.judge.scorer import FactConsistencyScorer

    class _BoomJudge(_FakeJudge):
        def judge_candidates(self, *a, **k):
            raise RuntimeError("显存抖动")

    scorer = FactConsistencyScorer(_BoomJudge())
    scores = scorer.score_batch(
        [{"candidate": "候选", "reference": "r", "source": "文书", "id": "c"}]
    )
    assert scores == [None]
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
