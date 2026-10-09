"""事实一致性 / 覆盖率 / INP Judge 的验收测试（全部不加载模型）。

模型本身的判断力靠 `scripts/judge_test.py` 在固定案例集上测；这里只钉住
三件"写错了不报错、只是分数悄悄变错"的事：

  1. **权重与最低分**：案由不参与最低分（它只有几个字，蒙对概率太高）
  2. **聚合**：缺项必须报错、覆盖率分母跟着参与要素走
  3. **一次性判定的接线**：六要素分 / INP 归组 / 人工摘要自评短路

    python -m pytest tests/test_judge.py -q
"""

from __future__ import annotations

import json

import pytest

from sfzy.judge.judge import FactConsistencyJudge, parse_inp, parse_six_scores
from sfzy.judge.schema import (
    DEFAULT_WEIGHTS,
    ELEMENTS,
    MAX_SCORE,
    MIN_SCORE_ELEMENTS,
    aggregate,
    aggregate_coverage,
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


# ---------------------------------------------------------------- 覆盖率聚合

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


# ---------------------------------------------------------------- 分数解析

def test_解析_一次性六要素分数():
    assert parse_six_scores('{"case_type": 3, "court_facts": 0}') == {
        "case_type": 3, "court_facts": 0,
    }
    assert parse_six_scores('```json\n{"court_facts": "4"}\n```') == {"court_facts": 4}
    # 前后有解释、键引号不规整时也要能救回来
    assert parse_six_scores('好的：\n{"judgment_result": 2}\n以上') == {"judgment_result": 2}
    assert parse_six_scores("我无法完成") == {}


def test_INP解析与重复分组():
    raw = json.dumps({"propositions": [
        {"text": "甲向乙借款10万元", "necessity": 4, "group": 1},
        {"text": "甲向乙借款10万元", "necessity": 3, "group": 1},   # 重复
        {"text": "案涉房屋位于某地", "necessity": 2, "group": 2},
    ]}, ensure_ascii=False)
    out = parse_inp(raw)
    # 去重后每个命题组取成员最大值：4 + 2；分母是原子命题总数 3
    assert out["inp"] == pytest.approx((4 + 2) / (4 * 3))
    assert out["groups"] == [0, 0, 1]
    assert len(out["propositions"]) == 3
    assert out["propositions"][0]["necessity"] == 4


# ---------------------------------------------------------------- Prompt

def test_一次性事实判定_prompt():
    from sfzy.judge.prompts import JUDGE_SYSTEM, build_fact_six_messages

    msgs = build_fact_six_messages("这是原文全文……", "这是摘要全文……")
    assert msgs[0]["content"] == JUDGE_SYSTEM
    user = msgs[1]["content"]
    assert "【裁判文书原文】" in user and "这是原文全文" in user
    assert "【待评价摘要】" in user and "这是摘要全文" in user
    assert "court_facts" in user and "judgment_result" in user


def test_一次性覆盖率判定_prompt():
    from sfzy.judge.prompts import (
        COVERAGE_SYSTEM,
        build_coverage_six_messages,
    )

    msgs = build_coverage_six_messages("人工摘要全文", "候选摘要全文")
    assert msgs[0]["content"] == COVERAGE_SYSTEM
    user = msgs[1]["content"]
    assert "【参考摘要（人工）】" in user and "人工摘要全文" in user
    assert "【候选摘要】" in user and "候选摘要全文" in user
    # 六个要素的键名和定义在 system 里（ELEMENT_DEFS_ZH + 输出格式模板），不在 user 里
    assert "case_type" in msgs[0]["content"] and "judgment_result" in msgs[0]["content"]
    # 覆盖率只吃"人工摘要 vs 候选摘要"，**不带裁判文书原文**
    assert "【裁判文书原文】" not in user


def test_INP_prompt_与材料():
    from sfzy.judge.prompts import INP_JUDGE_PROMPT, build_inp_messages

    assert INP_JUDGE_PROMPT.strip()      # 用户手写的 prompt，非空
    msgs = build_inp_messages("原文", "人工摘要", "候选摘要")
    assert msgs[0]["content"] == INP_JUDGE_PROMPT
    user = msgs[1]["content"]
    for block in ("【裁判文书原文】", "【人工摘要】", "【候选摘要】"):
        assert block in user


def test_EE_prompt_留空与材料():
    """EE prompt 由使用者自行填写 —— 这里只钉住"留空 + 三段材料"。"""
    from sfzy.judge.prompts import EE_JUDGE_PROMPT, build_ee_messages

    assert EE_JUDGE_PROMPT == ""          # 不要代写评价规则
    msgs = build_ee_messages("原文", "人工摘要", "候选摘要")
    assert msgs[0]["content"] == EE_JUDGE_PROMPT
    user = msgs[1]["content"]
    for block in ("【裁判文书原文】", "【人工参考摘要】", "【候选摘要】"):
        assert block in user


def test_EE解析与量程():
    from sfzy.judge.judge import parse_ee

    assert parse_ee(
        '{"semantic_redundancy":4,"verbal_redundancy":3,"abstraction_adequacy":2}'
    ) == {"semantic_redundancy": 4, "verbal_redundancy": 3, "abstraction_adequacy": 2}
    assert parse_ee('```json\n{"semantic_redundancy": "0"}\n```') == {
        "semantic_redundancy": 0,
    }
    # 前后有解释、键引号不规整时也要能救回来
    assert parse_ee("好的：\n{abstraction_adequacy: 3}\n以上") == {
        "abstraction_adequacy": 3,
    }
    assert parse_ee("我无法完成") == {}


def test_EE加权分公式():
    from sfzy.judge.schema import expression_efficiency_score

    dims = {"semantic_redundancy": 4, "verbal_redundancy": 0, "abstraction_adequacy": 4}
    assert expression_efficiency_score(dims) == pytest.approx(0.8)
    # 权重整体缩放不改变结果（R_EE 自带按权重和归一）
    assert expression_efficiency_score(
        dims, {"semantic_redundancy": 4, "verbal_redundancy": 2, "abstraction_adequacy": 4}
    ) == pytest.approx(0.8)
    # 权重和 <= 0 时返回 0
    assert expression_efficiency_score(
        dims, {"semantic_redundancy": 0, "verbal_redundancy": 0, "abstraction_adequacy": 0}
    ) == 0.0


def test_EE_维度常量与reward侧一致():
    """judge/schema 与 rl/reward_terms 各抄了一份 EE 维度，必须完全一致。"""
    from sfzy.judge.schema import EE_DEFAULT_WEIGHTS, EE_DIMENSIONS
    from sfzy.rl.reward_terms import EE_DEFAULT_WEIGHTS as R_DEFAULTS
    from sfzy.rl.reward_terms import EE_DIMENSIONS as R_DIMS

    assert tuple(R_DIMS) == tuple(EE_DIMENSIONS)
    assert dict(R_DEFAULTS) == dict(EE_DEFAULT_WEIGHTS)


def test_prompts模块只保留当前方案的三段():
    """不再有六要素抽取 / 逐要素判定的 prompt。"""
    from sfzy.judge import prompts as pm

    for gone in (
        "EXTRACT_SYSTEM",
        "EXTRACT_SUMMARY_SYSTEM",
        "build_extract_messages",
        "build_judge_messages",
        "build_judge_messages_with_context",
        "build_coverage_messages",
        "JUDGE_SYSTEM_CONTEXT",
    ):
        assert not hasattr(pm, gone), f"{gone} 应该已经删掉"


# ---------------------------------------------------------------- 一次性判定接线

class _FakeRT:
    """不加载模型：按 prompt 里的评价器身份选择要返回的 JSON。"""

    def __init__(self, fact=None, coverage=None, inp=None, ee=None):
        self.fact = fact if fact is not None else {n: 4 for n in ELEMENTS}
        self.coverage = coverage if coverage is not None else {n: 3 for n in ELEMENTS}
        self.inp = inp if inp is not None else {
            "propositions": [{"text": "甲向乙借款", "necessity": 4, "group": 1}]
        }
        self.ee = ee if ee is not None else {
            "semantic_redundancy": 4,
            "verbal_redundancy": 2,
            "abstraction_adequacy": 4,
        }
        self.calls: list = []          # 每次真正发出去的 prompt
        self.stats: dict = {}

    def render(self, messages):
        return messages[0]["content"] + "\n" + messages[-1]["content"]

    def generate_batch(self, prompts, max_new_tokens=512):
        out = []
        for prompt in prompts:
            self.calls.append(prompt)
            if "覆盖率评价器" in prompt:
                out.append(json.dumps(self.coverage, ensure_ascii=False))
            elif "信息必要性评价器" in prompt:
                out.append(json.dumps(self.inp, ensure_ascii=False))
            elif "【人工参考摘要】" in prompt:
                # EE 的 system 是空串，只能靠 user 里的材料块认出来
                out.append(json.dumps(self.ee, ensure_ascii=False))
            else:
                out.append(json.dumps(self.fact, ensure_ascii=False))
        return out


def test_一次性事实判定_六个分聚合成奖励():
    rt = _FakeRT(fact={**{n: 4 for n in ELEMENTS}, "judgment_result": 0})
    j = FactConsistencyJudge(runtime=rt)
    res = j.judge_candidates("原文", ["候选"])[0]
    assert res.signals["fact_consistency"] == pytest.approx(0.70)
    # 事实硬门控要读的最小原始分
    assert res.signals["fact_consistency_min_raw"] == 0.0


def test_覆盖率与INP_人工摘要自评短路():
    rt = _FakeRT(
        coverage={n: 2 for n in ELEMENTS},
        inp={"propositions": [
            {"text": "甲向乙借款10万元", "necessity": 4, "group": 1},
            {"text": "甲向乙借款10万元", "necessity": 3, "group": 1},
            {"text": "案涉房屋位于某地", "necessity": 2, "group": 2},
        ]},
    )
    j = FactConsistencyJudge(
        runtime=rt,
        tasks=("fact_consistency", "element_coverage", "information_necessity_precision"),
    )
    results = j.judge_candidates(
        "原文", ["候选摘要", "人工摘要"], reference="人工摘要",
    )
    # 候选臂：覆盖率 2/4 = 0.5，INP = (4+2)/(4×3)
    assert results[0].signals["element_coverage"] == pytest.approx(0.5)
    assert results[0].signals["information_necessity_precision"] == pytest.approx(0.5)
    # 人工臂：候选就是人工摘要 → 覆盖率满分、INP=1.0，且没有为它发请求
    assert results[1].signals["element_coverage"] == pytest.approx(1.0)
    assert results[1].signals["information_necessity_precision"] == pytest.approx(1.0)
    assert len([p for p in rt.calls if "覆盖率评价器" in p]) == 1
    assert len([p for p in rt.calls if "信息必要性评价器" in p]) == 1


def test_EE_人工摘要自评短路():
    """候选=人工摘要时短路为 1.0，不再发一条两段摘要完全相同的请求。"""
    rt = _FakeRT(ee={
        "semantic_redundancy": 4, "verbal_redundancy": 0, "abstraction_adequacy": 4,
    })
    j = FactConsistencyJudge(
        runtime=rt, tasks=("expression_efficiency",),
        ee_weights={"semantic_redundancy": 0.4, "verbal_redundancy": 0.2,
                    "abstraction_adequacy": 0.4},
    )
    results = j.judge_candidates(
        "原文", ["候选摘要", "人工摘要"], reference="人工摘要",
    )
    assert results[0].signals["expression_efficiency"] == pytest.approx(0.8)
    # 人工臂（候选=人工摘要）短路为满分，且没有为它发请求
    assert results[1].signals["expression_efficiency"] == pytest.approx(1.0)
    assert len([p for p in rt.calls if "【人工参考摘要】" in p]) == 1


def test_EE_没传参考摘要直接报错():
    j = FactConsistencyJudge(runtime=_FakeRT(), tasks=("expression_efficiency",))
    with pytest.raises(ValueError, match="reference"):
        j.judge_candidates("DOC", ["CAND"])


def test_只开覆盖率时不需要事实一致性():
    rt = _FakeRT(coverage={n: 4 for n in ELEMENTS})
    j = FactConsistencyJudge(runtime=rt, tasks=("element_coverage",))
    res = j.judge_candidates("原文", ["候选"], reference="人工摘要")[0]
    assert "fact_consistency" not in res.signals
    assert res.signals["element_coverage"] == pytest.approx(1.0)
    assert all("覆盖率评价器" in p for p in rt.calls)


def test_覆盖率_没传参考摘要直接报错():
    j = FactConsistencyJudge(runtime=_FakeRT(), tasks=("element_coverage",))
    with pytest.raises(ValueError, match="reference"):
        j.judge_candidates("DOC", ["CAND"])


def test_长原文判定时按头尾截断():
    """长判决书里送判的原文一旦整段截断就会丢掉首尾事实；
    现在只把输入文本按头+尾压到预算内。"""
    class _RT(_FakeRT):
        def __init__(self):
            super().__init__()
            self.seen = []

        def count_tokens(self, text):
            return len(text)

        def truncate_text(self, text, max_tokens, tail_ratio=0.5):
            self.seen.append((text, max_tokens))
            return text[:max_tokens]

    rt = _RT()
    j = FactConsistencyJudge(runtime=rt, max_input_tokens=2000)
    j.judge_fact_six_batch([("文" * 5000, "摘要")])
    assert rt.seen, "长输入没有被截断"
    budget = rt.seen[0][1]
    assert 0 < budget < 2000


# ---------------------------------------------------------------- GRPO 适配器

def test_适配器_按文书分组并摊回原位置():
    """GRPO 的 items 是扁平列表，同一篇文书的候选可能不连续。

    分组后分数必须摊回**原来的位置** —— 错位不报错，只是奖励悄悄换了对象。
    """
    from sfzy.judge.scorer import SixElementScorer

    rt = _FakeRT()

    class _CountingRT(_FakeRT):
        def __init__(self):
            super().__init__()
            self.batches = 0

        def generate_batch(self, prompts, max_new_tokens=512):
            self.batches += 1
            return super().generate_batch(prompts, max_new_tokens)

    rt = _CountingRT()
    j = FactConsistencyJudge(runtime=rt)
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
    # 按 source 分组 → 两篇文书各发一次事实判定批量（不分组会是 5 次）
    assert rt.batches == 2


def test_适配器_单组失败只标None不中断训练():
    """一次判定失败不该毁掉整轮几小时的 run：标 None 交给组内均值补。"""
    from sfzy.judge.scorer import SixElementScorer

    class _BoomJudge(FactConsistencyJudge):
        def judge_candidates(self, *a, **k):
            raise RuntimeError("显存抖动")

    scorer = SixElementScorer(_BoomJudge(runtime=_FakeRT()))
    signals = scorer.score_batch_signals(
        [{"candidate": "候选", "reference": "r", "source": "文书", "id": "c"}]
    )
    # 整组失败 → 该条没有任何信号（空字典），交给 trainer 按缺失补
    assert signals == [{}]
    assert scorer.last_errors and "RuntimeError" in scorer.last_errors[0]


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
    """模拟 ChatGLM3：有 stream_generate，forward 返回 logits。"""

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
    """Qwen2.5 的 generation_config 自带 temperature/top_p/top_k；
    judge 走贪心（do_sample=False），要显式清掉，免得每次判分刷告警。"""
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
