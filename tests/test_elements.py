"""LLM 抽取 + 逐要素 ROUGE 的验收测试（不联网：用一个假 runtime 顶替 API）。

覆盖四类东西：
  1. **解析容错**：围栏 / 多余文字 / null / 列表 / 整段坏掉时分清"要素为空"与"解析失败"；
  2. **权重口径**：ROUGE 三项加权（手算核对）、要素权重归一、空要素策略；
  3. **缓存**：键含模型与 prompt 指纹（换 prompt 必须失效），失败不写缓存；
  4. **环境变量文件**：非交互 shell 下从 `export KEY=VALUE` 文件里补读 key。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sfzy.elements.client import (
    ChatClient,
    Completion,
    diagnose_empty_completion,
    ensure_api_key,
    load_env_files,
    read_env_file,
)
from sfzy.elements.config import ApiConfig, ElementConfigError, ElementRougeConfig, RougeConfig
from sfzy.elements.extract import ElementCache, ElementExtractor, cache_key
from sfzy.elements.pipeline import ElementRougePipeline
from sfzy.elements.prompts import DEFAULT_SYSTEM, PromptTemplate
from sfzy.elements.schema import ELEMENTS, ElementSet, parse_elements
from sfzy.elements.score import aggregate, score_elements, weighted_score


# ---------------------------------------------------------------- 假 runtime


class FakeRuntime:
    """按"第几次调用"返回预设回复的假 runtime（`generate_messages` 返回一批）。"""

    model = "fake-model"

    def __init__(self, responses):
        self._batches = list(responses)
        self.calls = []
        self.stats = {"prompt_tokens": 0, "completion_tokens": 0}

    def generate_messages(self, conversations, max_new_tokens=512):
        self.calls.append([c[-1]["content"] for c in conversations])
        if not self._batches:
            raise AssertionError("假 runtime 没有预设回复了")
        batch = self._batches.pop(0)
        if isinstance(batch, str):
            batch = [batch] * len(conversations)
        assert len(batch) == len(conversations), "预设回复条数与请求条数不符"
        return list(batch)

    def ping(self):
        return "pong"


def _payload(**elements) -> str:
    data = {name: elements.get(name, "") for name in ELEMENTS}
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------- 解析容错


def test_解析_围栏和多余文字都能救回来():
    raw = '好的，结果如下：\n```json\n{"plaintiff_claims": "请求还款", ' \
          '"defendant_defenses": "已还清", "court_facts": "借款属实"}\n```\n希望有帮助'
    elements, ok = parse_elements(raw)
    assert ok
    assert elements.plaintiff_claims == "请求还款"
    assert elements.defendant_defenses == "已还清"
    assert elements.court_facts == "借款属实"


def test_解析_null和列表都能压成文本():
    raw = json.dumps(
        {
            "plaintiff_claims": ["判令还款", "承担诉讼费"],
            "defendant_defenses": None,
            "court_facts": {"事实": "借款属实"},
        },
        ensure_ascii=False,
    )
    elements, ok = parse_elements(raw)
    assert ok
    assert elements.plaintiff_claims == "判令还款；承担诉讼费"
    assert elements.defendant_defenses == ""
    assert "借款属实" in elements.court_facts


def test_解析_整段不是JSON时判失败而不是判空():
    """这里必须区分"模型说没有这一要素"和"这段回复没法用"。"""

    elements, ok = parse_elements("抱歉，我无法完成这个任务。")
    assert ok is False
    assert elements.to_dict() == {name: "" for name in ELEMENTS}


def test_解析_缺键时缺的那个当空串其余保留():
    raw = '{"plaintiff_claims": "请求还款"}'
    elements, ok = parse_elements(raw)
    assert ok
    assert elements.plaintiff_claims == "请求还款"
    assert elements.defendant_defenses == ""


# ---------------------------------------------------------------- prompt


def test_prompt_必须带summary占位符():
    with pytest.raises(ValueError, match="summary"):
        PromptTemplate(system="s", user="这段没有占位符")


def test_prompt_只替换占位符不动其它花括号():
    template = PromptTemplate(
        system=DEFAULT_SYSTEM,
        user='只输出 JSON：{"plaintiff_claims": ""}\n摘要：{summary}',
    )
    messages = template.render("原被告系借款合同纠纷。")
    assert messages[0]["role"] == "system"
    assert '{"plaintiff_claims": ""}' in messages[-1]["content"]
    assert "原被告系借款合同纠纷。" in messages[-1]["content"]
    assert "{summary}" not in messages[-1]["content"]


def test_prompt_指纹随内容变化():
    a = PromptTemplate(system="s", user="{summary}")
    b = PromptTemplate(system="s", user="{summary} ")
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == PromptTemplate(system="s", user="{summary}").fingerprint()


# ---------------------------------------------------------------- 配置校验


def _rouge_cfg(**overrides) -> RougeConfig:
    raw = {
        "mode": "char",
        "weights": {"rouge1": 0.2, "rouge2": 0.4, "rougel": 0.4},
    }
    raw.update(overrides)
    return RougeConfig.from_config(raw)


def test_配置_rouge权重必须写全():
    with pytest.raises(ElementConfigError, match="rouge.weights 必须写全"):
        RougeConfig.from_config({"mode": "char"})
    with pytest.raises(ElementConfigError, match="缺了 rougel"):
        RougeConfig.from_config(
            {"mode": "char", "weights": {"rouge1": 0.5, "rouge2": 0.5}}
        )


def test_配置_权重不能为负且和要大于零():
    with pytest.raises(ElementConfigError, match="不能为负"):
        RougeConfig.from_config(
            {"mode": "char", "weights": {"rouge1": -1, "rouge2": 1, "rougel": 1}}
        )
    with pytest.raises(ElementConfigError, match="大于 0"):
        RougeConfig.from_config(
            {"mode": "char", "weights": {"rouge1": 0, "rouge2": 0, "rougel": 0}}
        )


def test_配置_要素权重可省略且会归一化():
    cfg = _rouge_cfg()
    assert cfg.element_weights == {
        "plaintiff_claims": 0.30,
        "defendant_defenses": 0.20,
        "court_facts": 0.50,
    }
    other = _rouge_cfg(
        element_weights={"plaintiff_claims": 2, "defendant_defenses": 1, "court_facts": 1}
    )
    assert other.normalized_element_weights()["plaintiff_claims"] == pytest.approx(0.5)


def test_配置_要素权重写错要素名要报错():
    with pytest.raises(ElementConfigError, match="不认识的要素"):
        _rouge_cfg(element_weights={"plaintiff_claims": 1, "legal_basis": 1, "court_facts": 1})


def test_配置_整段配置能构造出来():
    cfg = ElementRougeConfig.from_config(
        {
            "api": {"base_url": "https://api.example.com/v1", "model": "m"},
            "prompt": {"system": "s", "user": "{summary}"},
            "rouge": {"mode": "char", "weights": {"rouge1": 1, "rouge2": 1, "rougel": 1}},
            "cache_path": None,
        }
    )
    assert cfg.api.model == "m"
    assert cfg.rouge.mode == "char"
    assert cfg.prompt.fingerprint()


def test_配置_缺api或prompt段时的报错要指路():
    with pytest.raises(ElementConfigError, match="element_rouge"):
        ElementRougeConfig.from_config({"rouge": {}})
    with pytest.raises(ElementConfigError, match="base_url"):
        ApiConfig.from_config({"model": "m"})


# ---------------------------------------------------------------- 打分


def test_加权分与手算一致():
    weights = {"rouge1": 0.2, "rouge2": 0.4, "rougel": 0.4}
    # 0.2*0.5 + 0.4*(1/3) + 0.4*0.5 = 0.4333…
    assert weighted_score(0.5, 1 / 3, 0.5, weights) == pytest.approx(0.4333333, abs=1e-6)
    # 权重整体放大不影响结果（按权重和归一）
    assert weighted_score(0.5, 1 / 3, 0.5, {"rouge1": 2, "rouge2": 4, "rougel": 4}) == pytest.approx(
        0.4333333, abs=1e-6
    )


def test_逐要素打分_手算一例():
    cfg = _rouge_cfg()
    # char 口径下 "原告诉请" = 原/告/诉/请，"原告请求" = 原/告/请/求：
    #   公共 1-gram = {原, 告, 请} = 3 → P = R = 3/4，F1 = 0.75
    #   公共 2-gram = {原告} = 1，各自 3 个 → F1 = 1/3
    #   LCS = "原告请" = 3 → P = R = 3/4，F1 = 0.75
    #   加权 = 0.2*0.75 + 0.4*(1/3) + 0.4*0.75 = 0.58333…
    ref = ElementSet(plaintiff_claims="原告诉请")
    hyp = ElementSet(plaintiff_claims="原告请求")
    scores = score_elements(ref, hyp, config=cfg)
    item = scores["plaintiff_claims"]
    assert item.status == "scored"
    assert item.rouge1_f == pytest.approx(0.75)
    assert item.rouge2_f == pytest.approx(1 / 3)
    assert item.rougel_f == pytest.approx(0.75)
    assert item.score == pytest.approx(0.5833333, abs=1e-6)


def test_逐要素打分_空要素的三种情形():
    cfg = _rouge_cfg()
    both_empty = score_elements(ElementSet(), ElementSet(), config=cfg)
    for name in ELEMENTS:
        assert both_empty[name].status == "both_empty"
        assert both_empty[name].score is None
        assert both_empty[name].counted is False

    one_side = score_elements(
        ElementSet(defendant_defenses="被告辩称已还清"), ElementSet(), config=cfg
    )
    assert one_side["defendant_defenses"].status == "one_side_empty"
    assert one_side["defendant_defenses"].score == 0.0


def test_总分只对计入的要素归一():
    cfg = _rouge_cfg()
    ref = ElementSet(plaintiff_claims="原告诉请", court_facts="借款属实")
    hyp = ElementSet(plaintiff_claims="原告诉请", court_facts="借款属实")
    scores = score_elements(ref, hyp, config=cfg)
    weights = cfg.normalized_element_weights()
    # 被告辩称两侧都空 → 不计入；剩下的两项应当归一后仍为 1.0
    assert scores["defendant_defenses"].score is None
    assert aggregate(scores, weights) == pytest.approx(1.0)


# ---------------------------------------------------------------- 抽取器


class FakeHTTPRuntime:
    """带 `_post` 的假 runtime：用来验证推理模型那条路径。"""

    model = "fake-reasoner"
    concurrency = 2

    def __init__(self, bodies):
        self._bodies = list(bodies)
        self.requests = []

    def _post(self, messages, *, max_tokens, json_mode=True, **kwargs):
        self.requests.append({"messages": messages, "max_tokens": max_tokens})
        return self._bodies.pop(0)


def _body(content="", reasoning="", finish="stop"):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "reasoning_content": reasoning},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def test_推理模型_思维链占满预算时的报错要能指出原因():
    completion = Completion(content="", reasoning="我先想想要不要…" * 10, finish_reason="length")
    message = diagnose_empty_completion(completion)
    assert "思维链" in message and "max_tokens" in message


def test_推理模型_JSON写在思维链里能被救回来():
    body = _body(content="", reasoning="思考…最后输出：" + _payload(plaintiff_claims="请求还款"))
    runtime = FakeHTTPRuntime([body])
    extractor = ElementExtractor(
        runtime, PromptTemplate(system="s", user="{summary}"), cache=ElementCache(),
        model="fake-reasoner", max_tokens=4096,
    )
    result = extractor.extract("摘要A")
    assert result.ok is True
    assert result.elements.plaintiff_claims == "请求还款"


def test_推理模型_只有思维链没有JSON时判失败且不写缓存():
    body = _body(content="", reasoning="我还在想…", finish="length")
    runtime = FakeHTTPRuntime([body, body])          # 重试一次还是这样
    extractor = ElementExtractor(
        runtime, PromptTemplate(system="s", user="{summary}"), cache=ElementCache(),
        model="fake-reasoner", max_tokens=16, parse_retries=1,
    )
    result = extractor.extract("摘要A")
    assert result.ok is False
    assert "思维链" in result.error
    assert len(extractor.cache) == 0


def test_聊天客户端_统计token并读finish_reason():
    runtime = FakeHTTPRuntime([_body(content="ok", finish="length")])
    client = ChatClient(runtime, max_tokens=128)
    completions = client.chat([[{"role": "user", "content": "hi"}]])
    assert completions[0].content == "ok"
    assert completions[0].truncated_by_length is True
    assert completions[0].prompt_tokens == 100
    assert runtime.requests[0]["max_tokens"] == 128


def test_聊天客户端_没有post的runtime退回标准接口():
    class Plain:
        model = "m"

        def generate_messages(self, conversations, max_new_tokens=512):
            return ["回复" for _ in conversations]

    completions = ChatClient(Plain(), max_tokens=8).chat([[{"role": "user", "content": "x"}]])
    assert completions[0].content == "回复"
    assert completions[0].reasoning == ""


def _extractor(responses, tmp_path=None, *, prompt=None, model="fake-model") -> ElementExtractor:
    cache = ElementCache() if tmp_path is None else ElementCache()
    return ElementExtractor(
        FakeRuntime(responses),
        prompt or PromptTemplate(system="s", user="{summary}"),
        cache=cache,
        model=model,
        parse_retries=1,
    )


def test_抽取_同文本只打一次API():
    extractor = _extractor([[_payload(plaintiff_claims="请求还款")]])
    results = extractor.extract_batch(["摘要A", "摘要A", "摘要A"])
    assert len(results) == 3
    assert all(r.elements.plaintiff_claims == "请求还款" for r in results)
    assert extractor.stats.api_calls == 1
    assert len(extractor.runtime.calls) == 1


def test_抽取_解析失败会重试():
    extractor = _extractor(
        ["这不是 JSON", [_payload(court_facts="借款属实")]]
    )
    result = extractor.extract("摘要A")
    assert result.ok is True
    assert result.elements.court_facts == "借款属实"
    assert extractor.stats.retried == 1
    assert extractor.stats.failed == 0


def test_抽取_重试后仍失败要标记失败而不是给空要素():
    extractor = _extractor(["坏回复", "还是坏回复"])
    result = extractor.extract("摘要A")
    assert result.ok is False
    assert result.error
    assert extractor.stats.failed == 1


def test_抽取_失败的条目不进缓存():
    extractor = _extractor(["坏回复", "还是坏回复"])
    extractor.extract("摘要A")
    assert len(extractor.cache) == 0


# ---------------------------------------------------------------- 缓存


def test_缓存_命中与落盘(tmp_path: Path):
    cache = ElementCache()
    extractor = ElementExtractor(
        FakeRuntime([[_payload(plaintiff_claims="请求还款")]]),
        PromptTemplate(system="s", user="{summary}"),
        cache=cache,
        model="m1",
    )
    extractor.extract("摘要A")
    path = tmp_path / "cache.jsonl"
    assert cache.save(path) == 1

    reloaded = ElementCache.load(path)
    again = ElementExtractor(
        FakeRuntime([]),                      # 不应再发请求
        PromptTemplate(system="s", user="{summary}"),
        cache=reloaded,
        model="m1",
    )
    result = again.extract("摘要A")
    assert result.from_cache is True
    assert result.elements.plaintiff_claims == "请求还款"
    assert again.stats.cache_hits == 1
    assert again.stats.api_calls == 0


def test_缓存_键含模型与prompt指纹():
    a = cache_key("摘要", "m1", "p1")
    assert a != cache_key("摘要", "m2", "p1")
    assert a != cache_key("摘要", "m1", "p2")
    assert a == cache_key("摘要", "m1", "p1")


def test_缓存_换prompt就不命中():
    cache = ElementCache()
    prompt_a = PromptTemplate(system="s", user="{summary}")
    prompt_b = PromptTemplate(system="s", user="新提示：{summary}")
    first = ElementExtractor(
        FakeRuntime([[_payload(plaintiff_claims="请求还款")]]), prompt_a, cache=cache, model="m"
    )
    first.extract("摘要A")
    second = ElementExtractor(
        FakeRuntime([[_payload(plaintiff_claims="换了prompt的结果")]]), prompt_b, cache=cache, model="m"
    )
    result = second.extract("摘要A")
    assert result.from_cache is False
    assert result.elements.plaintiff_claims == "换了prompt的结果"


# ---------------------------------------------------------------- 环境变量文件


def test_env文件_能读export与引号与注释(tmp_path: Path):
    path = tmp_path / "env"
    path.write_text(
        "# 注释行\n"
        "export SFZY_ELEMENT_EXTRACT=sk-abc123   # 行尾注释\n"
        "OTHER='带 空格 的值'\n"
        "EMPTY=\n",
        encoding="utf-8",
    )
    values = read_env_file(path)
    assert values["SFZY_ELEMENT_EXTRACT"] == "sk-abc123"
    assert values["OTHER"] == "带 空格 的值"
    assert "EMPTY" not in values


def test_env文件_不覆盖已有的环境变量(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SFZY_ELEMENT_EXTRACT", "已经设好了")
    path = tmp_path / "env"
    path.write_text("export SFZY_ELEMENT_EXTRACT=文件里的\n", encoding="utf-8")
    load_env_files([path])
    import os

    assert os.environ["SFZY_ELEMENT_EXTRACT"] == "已经设好了"


def test_env文件_进程环境里没有时能从文件补读到key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SFZY_ELEMENT_EXTRACT", raising=False)
    path = tmp_path / "env"
    path.write_text("export SFZY_ELEMENT_EXTRACT=sk-fromfile\n", encoding="utf-8")
    api = ApiConfig(base_url="https://api.example.com/v1", model="m", env_file=str(path))
    filled = ensure_api_key(api)
    assert filled == path
    import os

    assert os.environ["SFZY_ELEMENT_EXTRACT"] == "sk-fromfile"


def test_env文件_哪里都没有key时报错要指路(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SFZY_ELEMENT_EXTRACT", raising=False)
    # 关掉默认候选文件（否则会去读本机真实的 ~/.bashrc，测试就依赖机器状态了）
    monkeypatch.setattr("sfzy.elements.client.DEFAULT_ENV_FILES", ())
    api = ApiConfig(
        base_url="https://api.example.com/v1", model="m", env_file=str(tmp_path / "none")
    )
    assert ensure_api_key(api) is None


# ---------------------------------------------------------------- 端到端（假 runtime）


def test_端到端_多候选宏平均():
    reference = "原被告系借款合同纠纷。原告请求判令被告偿还借款十万元。被告辩称已经还清。经审理查明，借款属实。"
    good = "原被告系借款合同纠纷。原告请求判令被告偿还借款十万元。被告辩称已经还清。经审理查明，借款属实。"
    bad = "本案系买卖纠纷。原告主张货款。被告未答辩。法院认为证据不足。"

    elements_by_text = {
        reference: _payload(
            plaintiff_claims="原告请求判令被告偿还借款十万元。",
            defendant_defenses="被告辩称已经还清。",
            court_facts="经审理查明，借款属实。",
        ),
        good: _payload(
            plaintiff_claims="原告请求判令被告偿还借款十万元。",
            defendant_defenses="被告辩称已经还清。",
            court_facts="经审理查明，借款属实。",
        ),
        bad: _payload(
            plaintiff_claims="原告主张货款。",
            defendant_defenses="",
            court_facts="法院认为证据不足。",
        ),
    }

    class ByContentRuntime(FakeRuntime):
        """按"渲染后的 prompt 正文"查表——prompt 模板就是 `{summary}`，所以正文==摘要。"""

        def __init__(self):
            super().__init__([])
            self.stats = {"prompt_tokens": 0, "completion_tokens": 0}

        def generate_messages(self, conversations, max_new_tokens=512):
            return [elements_by_text[c[-1]["content"]] for c in conversations]

    prompt = PromptTemplate(system="s", user="{summary}")
    extractor = ElementExtractor(ByContentRuntime(), prompt, cache=ElementCache(), model="m")
    config = ElementRougeConfig.from_config(
        {
            "api": {"base_url": "https://api.example.com/v1", "model": "m"},
            "prompt": {"system": "s", "user": "{summary}"},
            "rouge": {"mode": "char", "weights": {"rouge1": 0.2, "rouge2": 0.4, "rougel": 0.4}},
        }
    )
    pipeline = ElementRougePipeline(extractor, config)
    rows = [
        {"id": "s1", "reference": reference, "candidates": {"good": good, "bad": bad}},
        {"id": "s2", "reference": reference, "candidates": {"good": good, "bad": bad}},
    ]
    results, report = pipeline.evaluate_rows(rows, ["good", "bad"])
    assert len(results) == 2
    good_score = report["candidates"]["good"]["overall"]
    bad_score = report["candidates"]["bad"]["overall"]
    assert good_score == pytest.approx(1.0)
    assert bad_score is not None and bad_score < good_score
    # 被告辩称：bad 侧为空 → 单侧空 = 0 分，但不是"不计入"
    assert report["candidates"]["bad"]["status"]["defendant_defenses:one_side_empty"] == 2
    assert report["reference"]["empty_rate"]["defendant_defenses"] == 0.0


def test_端到端_人工摘要抽取失败时不产生假分数():
    class BadRefRuntime(FakeRuntime):
        def __init__(self):
            super().__init__([])
            self.stats = {"prompt_tokens": 0, "completion_tokens": 0}

        def generate_messages(self, conversations, max_new_tokens=512):
            return ["不是 JSON"] * len(conversations)

    prompt = PromptTemplate(system="s", user="{summary}")
    extractor = ElementExtractor(BadRefRuntime(), prompt, cache=ElementCache(), model="m", parse_retries=0)
    config = ElementRougeConfig.from_config(
        {
            "api": {"base_url": "https://api.example.com/v1", "model": "m"},
            "prompt": {"system": "s", "user": "{summary}"},
            "rouge": {"mode": "char", "weights": {"rouge1": 1, "rouge2": 1, "rougel": 1}},
        }
    )
    pipeline = ElementRougePipeline(extractor, config)
    _, report = pipeline.evaluate_rows(
        [{"id": "s1", "reference": "人工摘要", "candidates": {"c": "候选摘要"}}], ["c"]
    )
    assert report["candidates"]["c"]["overall"] is None
    assert report["candidates"]["c"]["n_failed"] == 1
    assert report["reference"]["n_ok"] == 0
