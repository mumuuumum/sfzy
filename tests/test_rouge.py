"""eval/rouge.py 的验收测试。

所有期望值都是**手算**的，不是跑出来的。你可以拿这些数字直接检验实现。
官方口径：总分 = 0.2*F1(R1) + 0.4*F1(R2) + 0.4*F1(RL)
"""

from __future__ import annotations

import pytest

from sfzy.eval.rouge import rouge_l, rouge_n, score_corpus, score_pair, tokenize

APPROX = dict(abs=1e-6)


# ---------------------------------------------------------------- 分词

def test_tokenize_char_忽略空白():
    assert tokenize("a b c", mode="char") == ["a", "b", "c"]
    assert tokenize("中 文\t测试\n", mode="char") == ["中", "文", "测", "试"]


def test_tokenize_jieba_可运行():
    tokens = tokenize("原告与被告民间借贷纠纷", mode="jieba")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_jieba_模块被缓存():
    """jieba 的 import 外面套了告警抑制（Py3.12 下它会因自身源码发
    SyntaxWarning，并 import pkg_resources 触发弃用告警），用 lru_cache 保证
    只走一次 —— 重复进出 catch_warnings 会白白多写一次全局告警状态。
    """
    from sfzy.eval.rouge import _jieba_module

    assert _jieba_module() is _jieba_module()
    assert _jieba_module().lcut("原告与被告") == ["原告", "与", "被告"]


def test_tokenize_未知模式应报错():
    with pytest.raises(ValueError):
        tokenize("abc", mode="不存在的模式")


# ---------------------------------------------------------------- ROUGE-N

def test_rouge_n_完全相同():
    p, r, f = rouge_n(["a", "b", "c"], ["a", "b", "c"], n=1)
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_rouge_1_手算():
    # hyp={a,b,c} ref={a,b,d}，交集 {a,b} 共 2 个
    # p = 2/3, r = 2/3, f = 2/3
    p, r, f = rouge_n(["a", "b", "c"], ["a", "b", "d"], n=1)
    assert p == pytest.approx(2 / 3, **APPROX)
    assert r == pytest.approx(2 / 3, **APPROX)
    assert f == pytest.approx(2 / 3, **APPROX)


def test_rouge_2_手算():
    # hyp bigram = {(a,b),(b,c)}  ref = {(a,b),(b,d)}  交集 1 个
    p, r, f = rouge_n(["a", "b", "c"], ["a", "b", "d"], n=2)
    assert p == pytest.approx(0.5, **APPROX)
    assert r == pytest.approx(0.5, **APPROX)
    assert f == pytest.approx(0.5, **APPROX)


def test_rouge_n_完全无交集():
    p, r, f = rouge_n(["x", "y", "z"], ["a", "b", "c"], n=1)
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_rouge_n_空预测不崩溃():
    """空串会让 precision 的分母为 0，必须返回 0 而不是抛异常。"""
    p, r, f = rouge_n([], ["a", "b", "c"], n=1)
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_rouge_n_重复词按较小计数():
    # hyp 有 3 个 "a"，ref 有 1 个 -> 交集按较小值算，只算 1 个
    _, _, f = rouge_n(["a", "a", "a"], ["a"], n=1)
    assert f == pytest.approx(2 * (1 / 3) * (1 / 1) / ((1 / 3) + 1), **APPROX)


# ---------------------------------------------------------------- ROUGE-L

def test_rouge_l_子序列手算():
    # hyp="abcde" ref="ace"，LCS="ace" 长度 3
    # p = 3/5 = 0.6, r = 3/3 = 1.0, f = 2*0.6*1/(0.6+1) = 0.75
    p, r, f = rouge_l(list("abcde"), list("ace"))
    assert p == pytest.approx(0.6, **APPROX)
    assert r == pytest.approx(1.0, **APPROX)
    assert f == pytest.approx(0.75, **APPROX)


def test_rouge_l_不要求连续():
    # 连续匹配只有 "ab"，但 LCS 是 "abc"
    _, _, f_cont = rouge_n(list("abc"), list("abc"), n=1)
    _, _, f_lcs = rouge_l(list("abc"), list("acb"))
    assert f_cont == 1.0
    # LCS("abc","acb") = 2 ("ab" 或 "ac")
    assert f_lcs == pytest.approx(2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3)), **APPROX)


def test_rouge_l_空输入不崩溃():
    assert rouge_l([], ["a", "b"]) == (0.0, 0.0, 0.0)
    assert rouge_l(["a"], []) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- score_pair

def test_score_pair_完全相同():
    result = score_pair("abcabc", "abcabc")
    for metric in ("rouge-1", "rouge-2", "rouge-l"):
        assert result[f"{metric}-f"] == pytest.approx(1.0, **APPROX)
    assert result["overall"] == pytest.approx(1.0, **APPROX)


def test_score_pair_官方权重公式():
    result = score_pair("abc", "abd")
    assert result["rouge-1-f"] == pytest.approx(2 / 3, **APPROX)
    assert result["rouge-2-f"] == pytest.approx(0.5, **APPROX)
    assert result["rouge-l-f"] == pytest.approx(2 / 3, **APPROX)
    # 0.2*(2/3) + 0.4*(1/2) + 0.4*(2/3) = 0.6
    assert result["overall"] == pytest.approx(0.6, **APPROX)


def test_score_pair_overall_确实按权重加权():
    result = score_pair("abc", "abd")
    expected = (
        0.2 * result["rouge-1-f"]
        + 0.4 * result["rouge-2-f"]
        + 0.4 * result["rouge-l-f"]
    )
    assert result["overall"] == pytest.approx(expected, **APPROX)


def test_score_pair_空白不影响字级得分():
    assert score_pair("a b c", "abc")["overall"] == pytest.approx(1.0, **APPROX)


def test_score_pair_完全无关():
    assert score_pair("xyz", "abc")["overall"] == pytest.approx(0.0, **APPROX)


def test_score_pair_空预测():
    assert score_pair("", "abc")["overall"] == pytest.approx(0.0, **APPROX)


# ---------------------------------------------------------------- score_corpus

def test_score_corpus_是按样本取平均():
    preds = {"a": "abc", "b": "xyz"}
    refs = {"a": "abc", "b": "abc"}
    result = score_corpus(preds, refs)

    assert result["num_samples"] == 2
    # (1.0 + 0.0) / 2
    assert result["rouge-1-f"] == pytest.approx(0.5, **APPROX)
    assert result["overall"] == pytest.approx(0.5, **APPROX)


def test_score_corpus_只评测id交集部分():
    preds = {"a": "abc", "c": "不应被计入"}
    refs = {"a": "abc", "b": "缺预测"}
    result = score_corpus(preds, refs)
    # 只有 id "a" 同时存在于两边
    assert result["num_samples"] == 1
    assert result["overall"] == pytest.approx(1.0, **APPROX)


def test_score_corpus_返回官方口径总分():
    preds = {"a": "abc", "b": "abcde"}
    refs = {"a": "abd", "b": "ace"}
    result = score_corpus(preds, refs)

    expected = (
        0.2 * result["rouge-1-f"]
        + 0.4 * result["rouge-2-f"]
        + 0.4 * result["rouge-l-f"]
    )
    assert result["overall"] == pytest.approx(expected, **APPROX)


def test_score_corpus_空输入不崩溃():
    result = score_corpus({}, {})
    assert result["num_samples"] == 0
    assert result["overall"] == pytest.approx(0.0, **APPROX)
