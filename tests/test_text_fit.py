"""utils/text_fit.py 的验收测试：超预算时保留头+尾、砍中间。"""

from __future__ import annotations

from sfzy.utils.text_fit import ELLIPSIS, count_tokens, truncate_text


class _CharTokenizer:
    """每个字符一个 token 的假 tokenizer（纯 CPU，不依赖 transformers）。"""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


def test_count_tokens():
    assert count_tokens(_CharTokenizer(), "判决如下") == 4


def test_不超预算原样返回():
    tok = _CharTokenizer()
    assert truncate_text(tok, "abc", 10) == "abc"


def test_超预算保留头尾():
    tok = _CharTokenizer()
    text = "".join(chr(0x4E00 + i) for i in range(100))   # 100 个汉字
    out = truncate_text(tok, text, 10, tail_ratio=0.5)
    assert out.startswith(text[:5])          # 头保留
    assert out.endswith(text[-5:])           # 尾保留
    assert ELLIPSIS.strip() in out           # 中间有省略标记
    assert len(out) < len(text)


def test_预算为0返回空():
    assert truncate_text(_CharTokenizer(), "abc", 0) == ""
