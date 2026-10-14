"""三要素的数据结构与**容错解析**。

要素：`plaintiff_claims`（原告诉讼请求）、`defendant_defenses`（被告辩称）、
`court_facts`（法院认定事实及说理）。和规则切分器（`sfzy.segment`）用的是同一套
键名，方便两边结果直接对比。

============================ 为什么要"容错解析" ============================
抽取这一路是**调 API**：模型偶尔会包一层 ```json 围栏、前后加点解释文字、
把某个要素写成 null 或列表。这类格式瑕疵如果当成"抽取失败"，整条样本就白花钱了；
但如果当成"要素为空"，又会把一次格式问题静默变成"人工摘要里没有被告辩称"，
进而把 ROUGE 算错。所以这里的口径是：

  * **解析**尽量宽松（围栏 / 多余文字 / 非字符串值都能救回来）；
  * **解析失败**必须显式返回 `ok=False`，由调用方决定重试或丢弃——
    绝不悄悄降级成空要素。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

#: 本模块负责的三个要素（顺序即打印顺序）。
ELEMENTS: Tuple[str, ...] = ("plaintiff_claims", "defendant_defenses", "court_facts")

ELEMENT_ZH: Dict[str, str] = {
    "plaintiff_claims": "原告诉讼请求",
    "defendant_defenses": "被告辩称",
    "court_facts": "法院认定事实及说理",
}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def coerce_element_value(value: Any) -> str:
    """把模型给的任意形态压成一段文本。

    `None` → 空串；列表/字典 → 用中文分号接起来（模型偶尔会把诉求写成编号列表）。
    这与 `sfzy.judge.schema.coerce_element_value` 的口径一致，但本模块不 import
    那个包（它会把 torch 拉进来）。
    """

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [coerce_element_value(v) for v in value if v is not None]
        return "；".join(p for p in parts if p).strip()
    if isinstance(value, Mapping):
        return "；".join(
            f"{k}：{coerce_element_value(v)}" for k, v in value.items() if coerce_element_value(v)
        ).strip()
    return str(value).strip()


@dataclass(frozen=True)
class ElementSet:
    """一条摘要抽出来的三个要素。缺的要素是空串（不是 None）。"""

    plaintiff_claims: str = ""
    defendant_defenses: str = ""
    court_facts: str = ""

    def get(self, name: str) -> str:
        return getattr(self, name, "")

    def to_dict(self) -> Dict[str, str]:
        return {name: self.get(name) for name in ELEMENTS}

    def is_empty(self, name: str) -> bool:
        return not self.get(name)

    def empty_elements(self) -> List[str]:
        return [name for name in ELEMENTS if self.is_empty(name)]

    def chars(self) -> int:
        return sum(len(self.get(name)) for name in ELEMENTS)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ElementSet":
        return cls(**{name: coerce_element_value(data.get(name, "")) for name in ELEMENTS})


def _extract_json_object(raw: str) -> str:
    """从原始回复里截出最外层的 JSON 对象文本（先剥围栏，再取首尾花括号）。"""

    text = (raw or "").strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    if "{" in text and "}" in text:
        return text[text.find("{") : text.rfind("}") + 1]
    return text


def _fallback_fields(raw: str) -> Dict[str, str]:
    """JSON 整体解析失败时的兜底：按 `"键": "值"` 逐个抠。

    模型偶尔会在 JSON 里漏一个逗号、或在中文字符串里塞了未转义的换行。整段
    `json.loads` 会失败，但每个键值对本身往往还是好的——这时按行抠比整条丢掉划算。
    """

    out: Dict[str, str] = {}
    for name in ELEMENTS:
        pattern = re.compile(
            r'["\']' + name + r'["\']\s*[:：]\s*["\'](.*?)["\']\s*(?:,|\}|$)',
            re.S,
        )
        match = pattern.search(raw or "")
        if match:
            out[name] = match.group(1).strip()
    return out


def parse_elements(raw: str) -> Tuple[ElementSet, bool]:
    """模型回复 → (要素, 是否解析成功)。

    `ok=False` 表示"这段回复没法当抽取结果用"（调用方应当重试或丢弃），
    不代表"要素是空的"。
    """

    candidate = _extract_json_object(raw)
    if candidate:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, Mapping):
            return ElementSet.from_dict(data), True

    fields = _fallback_fields(raw)
    if fields:
        return ElementSet.from_dict(fields), True
    return ElementSet(), False


__all__ = [
    "ELEMENTS",
    "ELEMENT_ZH",
    "ElementSet",
    "coerce_element_value",
    "parse_elements",
]
