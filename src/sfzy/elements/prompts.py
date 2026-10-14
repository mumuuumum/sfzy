"""抽取用的 prompt 模板（**写在配置文件里**，用户可整段替换）。

============================ 契约 ============================
配置里是一段 prompt：

    prompt:
      system: |                  # 可选
        ...
      user: |                    # 必填，且必须含 {summary} 占位符
        下面是摘要：{summary}

渲染规则：只把**字面量 `{summary}`** 替换成摘要正文，其它花括号原样保留
（prompt 里通常有 `{"plaintiff_claims": ""}` 这样的 JSON 示例，用 `str.format`
会直接炸）。少了 `{summary}` 会在构造时就报错，不会安静地把摘要丢掉。

模型必须输出**只含三个键的 JSON**：

    {"plaintiff_claims": "...", "defendant_defenses": "...", "court_facts": "..."}

没写 prompt 时用 `DEFAULT_USER` 那一版；要点是**逐字保留原文表述**——下游要拿
两边抽出来的要素算 ROUGE，一旦模型改写/总结，ROUGE 量到的就不是"摘要之间像不像"，
而是"两个改写者像不像"。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, List, Mapping

from sfzy.elements.schema import ELEMENTS, ELEMENT_ZH

#: 占位符：摘要正文插到这里。用字面量替换而不是 str.format（见模块 docstring）。
SUMMARY_PLACEHOLDER = "{summary}"


DEFAULT_SYSTEM = (
    "你是裁判文书摘要的三要素切分器。你的输出会被程序直接解析并按字面比对，"
    "所以只允许输出 JSON，不要解释。"
)

DEFAULT_USER = """从下面的摘要里抽出三个要素，**逐字复制原文**（连标点、数字、
当事人代称都照抄；不要改写、不要总结、不要补全、不要调整顺序）：

- plaintiff_claims：原告诉讼请求。案件背景/案由（"原被告系××纠纷"）不算诉请；
  诉请从原告提出请求处开始，到被告开始答辩前结束。没有诉请内容时输出空字符串。
- defendant_defenses：被告辩称。包含"被告辩称…""被告未答辩""被告承认/同意…"
  这类被告方陈述；要保留**全部**被告的答辩（多个被告连续答辩算一段），
  一直到法院开始叙述（"经审理查明""本院认为"等）为止。没有则输出空字符串。
- court_facts：法院认定的事实、事实认定与裁判说理。从法院叙述开始，
  到法律依据与裁判结果（"依照《…》…判决/裁定/驳回""判决如下"）之前结束。
  没有则输出空字符串。

只输出 JSON，键名固定，值为字符串（示例里的内容全部是空串）：

{{"plaintiff_claims": "", "defendant_defenses": "", "court_facts": ""}}

摘要：
{summary}
"""


@dataclass(frozen=True)
class PromptTemplate:
    """一段可渲染的抽取 prompt。"""

    system: str
    user: str

    def __post_init__(self) -> None:
        if SUMMARY_PLACEHOLDER not in self.user:
            raise ValueError(
                f"抽取 prompt 的 user 模板里找不到占位符 {SUMMARY_PLACEHOLDER}；"
                "摘要正文没有地方插入，抽取会拿模型读不到的输入去跑。"
            )

    # ---------------------------------------------------------------- 渲染
    def render(self, summary: str) -> List[dict]:
        messages: List[dict] = []
        if self.system.strip():
            messages.append({"role": "system", "content": self.system})
        messages.append(
            {"role": "user", "content": self.user.replace(SUMMARY_PLACEHOLDER, summary)}
        )
        return messages

    def to_dict(self) -> dict:
        return {"system": self.system, "user": self.user}

    # ---------------------------------------------------------------- 指纹
    def fingerprint(self) -> str:
        """prompt 指纹，进缓存键：**改了 prompt 就必须让旧缓存失效**。

        抽取结果强依赖 prompt，拿旧 prompt 抽的结果去配新 prompt 的分数是静默错位。
        """

        payload = f"{self.system}\x00{self.user}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]

    @classmethod
    def from_config(cls, raw: Any) -> "PromptTemplate":
        if raw is None:
            return default_prompt()
        if isinstance(raw, str):
            return cls(system=DEFAULT_SYSTEM, user=raw)
        if not isinstance(raw, Mapping):
            raise ValueError(
                "prompt 必须是字符串，或 {system: ..., user: ...} 形式的映射，"
                f"收到 {type(raw).__name__}"
            )
        return cls(
            system=str(raw.get("system", DEFAULT_SYSTEM)),
            user=str(raw.get("user", DEFAULT_USER)),
        )


def default_prompt() -> PromptTemplate:
    """没配 prompt 时用的那一版（可跑通，但应当由使用方替换）。"""

    return PromptTemplate(system=DEFAULT_SYSTEM, user=DEFAULT_USER)


def elements_spec_zh() -> str:
    """三个要素的中文名，写进日志/报告用。"""

    return "、".join(f"{name}（{ELEMENT_ZH[name]}）" for name in ELEMENTS)


__all__ = [
    "PromptTemplate",
    "SUMMARY_PLACEHOLDER",
    "DEFAULT_SYSTEM",
    "DEFAULT_USER",
    "default_prompt",
    "elements_spec_zh",
]
