"""CAIL2020 数据结构的解析、校验与导出。

============================ 你要实现的文件 ============================
这个文件留给你写。下面是接口契约和提示，照着填实现即可。

验收方式：
    python -m pytest tests/test_schema.py -q

-------------------------------- 原始格式 --------------------------------
JSONL，每行一条：

    {"id": "abc",
     "summary": "参考摘要",                              # 测试集缺失
     "text": [{"sentence": "句子", "label": 1}, ...]}    # 测试集缺 label

要点：
  * text 是**按句子切分**后的列表，空串拼接可无损还原原文。
  * label 是官方的句子重要度标注（抽取式）。本项目走生成式路线，
    **解析时直接忽略 label**，不要把它读进 Document。
  * 测试集口径只有 id 与 text[].sentence，summary 缺失属正常情况，
    不能当成错误。
=====================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

PathLike = Union[str, Path]


@dataclass
class Document:
    """一条裁判文书样本。

    字段已经定好了（测试依赖这些名字），你只需要实现下面三个成员。
    """

    id: str
    sentences: List[str] = field(default_factory=list)
    summary: Optional[str] = None

    @property
    def has_summary(self) -> bool:
        """是否带有参考摘要（测试集样本为 False）。"""
        return self.summary is not None

    def joined(self, sep: str = "") -> str:
        """把句子拼回文书全文。

        参数 sep 默认空串 —— 因为官方是按句切分得到的 text，
        空串拼接才能无损还原原文；传 "\\n" 会让结果更可读，
        但会引入原文中不存在的字符，写报告时要说明用了哪种。

        提示：注意"空字符串"和"没传参数"要得到同样的结果。
        """
        return sep.join(self.sentences)

    def stats(self) -> Dict[str, int]:
        """返回 {"num_sentences", "source_chars", "summary_chars"}。

        提示：summary 缺失时 summary_chars 记 0，不要抛异常。
        source_chars 用哪种拼接结果来算？想一想，并在代码里注释说明你的选择。
        """
        # 选择用空串拼接的结果来计算 source_chars，因为空串拼接才能无损还原原文
        source_text = self.joined("")
        return {
          "num_sentences": len(self.sentences),
          "source_chars": len(source_text),
          "summary_chars": len(self.summary) if self.summary is not None else 0
        }


def parse_document(raw: Dict[str, Any]) -> Document:
    """把一行原始 JSON（已 json.loads 成 dict）解析为 Document。
    可选增强（不强求，但能让代码更健壮）：
      * 有的衍生版本把 text 直接存成字符串列表，可以考虑容错处理
      * id 可能是数字，统一转成 str 更安全
    """
    if "id" not in raw:
      raise  ValueError("缺少字段: id")
    if "text" not in raw:
      raise ValueError("缺少字段: text")
    
    doc_id = str(raw["id"])
    texts = raw["text"]
    doc_sentences : List[str] = []
    for text in texts:
      doc_sentences.append(str(text["sentence"]))
    
    doc_summary = raw.get("summary")
    
    return Document(doc_id,doc_sentences,doc_summary)
    


def parse_line(line: str) -> Document:
    """解析 JSONL 的单行文本。

    要求：
      * 空白行（strip 后为空）-> raise ValueError
      * JSON 语法错误时，让异常信息能定位问题
    """
    if not line.strip():
      raise ValueError("空白行")
    try:
      raw = json.loads(line)
    except json.JSONDecodeError as e:
      raise ValueError(f"JSON 语法错误: {e}") from e
    return parse_document(raw)


def read_jsonl(path: PathLike) -> Iterator[Document]:
    """逐行读取并 yield Document。

    要求：
      * 是生成器，不要一次性把所有样本读进内存
        （全量数据 126MB，逐行流式处理才能控制内存）
      * 某行解析失败时，异常信息必须带上**行号**，格式形如
        "data/xxx.jsonl: 第 2 行解析失败: ..."
        测试用 pytest.raises(match="第 2 行") 检查这一点。
    """
    path_str = str(path)
    with open(path,"r",encoding="utf-8") as f:
      for line_num, line in enumerate(f,start=1):
        try:
          doc = parse_line(line)
        except ValueError as e:
          raise ValueError(f"{path_str}：第{line_num}行解析失败：{e}") from e
        yield doc


def to_input_record(doc: Document) -> Dict[str, Any]:
    """导出**测试集口径**的输入记录：只保留 id 和句子，去掉 summary 与 label。

    期望结构：
        {"id": "...", "text": [{"sentence": "..."}, ...]}

    这个函数是"防止信息泄漏"的关键：任何送给模型的输入都必须走它，
    以免把参考摘要或标注混进 prompt。
    """
    return {
      "id": doc.id,
      "text": [{"sentence": s} for s in doc.sentences]
    }



def to_submission_record(doc_id: str, summary: str) -> Dict[str, Any]:
    """导出官方提交格式的一条记录：{"id": ..., "summary": ...}。"""
    return {"id": doc_id, "summary": summary}


def write_submission(path: PathLike, pairs: Any) -> int:
    """把 (id, summary) 序列写成 JSONL，返回写出条数。

    要求：
      * JSONL 格式（每行一个 JSON 对象）
      * ensure_ascii=False，中文不要被转义成 \\uXXXX
      * 目标目录不存在时自动创建（测试会用带中文的嵌套路径）
      * pairs 可能是列表、也可能是生成器，不要依赖下标访问

    提示：写文件的小工具在 sfzy.utils.io.write_jsonl，直接复用即可，
    不必在这里重复实现。
    """
    from sfzy.utils.io import write_jsonl

    records = (to_submission_record(doc_id, summary) for doc_id, summary in pairs)
    return write_jsonl(path, records)
