"""SFT 数据集：把规范化记录变成"prompt 消息 + 答案"的样本。

============================ 你要实现的文件 ============================

-------------------------------- 为什么不直接用 jsonl 塞给 collator --------------------------------
因为后面要接 RAG。

纯 SFT 时这个类确实很薄，但只要进到 M3，它就要在 __getitem__ 里
**按当前样本去检索法条**，把检索结果拼进 messages。检索是有开销的，
所以放在 __getitem__ 里按需触发，而不是启动时把全量样本都预处理一遍。

这个设计让 RAG 变成一个**开关**（retriever=None 就是纯 SFT），
消融实验时不用改数据管线。

-------------------------------- 数据格式 --------------------------------
输入是 data/processed/{train,dev}.jsonl，每行
    {"id": ..., "source": "文书全文", "summary": "参考摘要"}
=====================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from sfzy.data.prompts import build_messages
from sfzy.utils.io import read_jsonl


def load_records(path: str) -> List[Dict[str, Any]]:
    """读取规范化后的 jsonl，返回记录列表。

    提示：直接复用 sfzy.utils.io.read_jsonl，别自己写一遍。
    全量数据约 1.2 万条，读进内存没有问题（原始 126MB 里大头是 label 和
    已被我们丢掉的字段）。
    """
    return list(read_jsonl(path))


class SFTDataset(Dataset):
    """每条形如 {"id", "messages", "answer", "contexts"}。

    messages 由 data/prompts.build_messages 构造，且**只含 system 与 user**，
    答案单独放在 answer 里交给 collator —— 这样 collator 才能确定
    prompt 与答案的边界，进而构造 loss mask。
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        prompt_style: str = "structured",
        retriever: Optional[Any] = None,
        top_k: int = 5,
    ) -> None:
        """retriever 为 None 时退化为纯 SFT（不检索）。

        这个参数的存在让 RAG 成为可开关的消融项，而不是另写一套代码。
        真正的 retriever 在 M3 实现，接口约定为
            retriever.retrieve(query, k) -> list[RetrievedDoc]
        每个 RetrievedDoc 带 .text 属性。
        """
        self.records = records
        self.prompt_style = prompt_style
        self.retriever = retriever
        self.top_k = top_k
        

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """返回 {"id", "messages", "answer", "contexts"}。

        步骤：
          1. 取第 idx 条记录
          2. 若 self.retriever 不为 None，用 source 去检索，拿到 contexts
             （检索 query 用什么？整篇文书太长，直接拿全文当 query 效果差，
             可以考虑只取开头若干字，或者用文书里的关键词）
          3. 调 build_messages(source, style, contexts) 得到 messages
          4. answer 就是记录的 summary

        注意：一定不要返回原始记录里的多余字段，尤其别把答案混进 messages。
        """
        record = self.records[idx]
        contexts = []
        if self.retriever is not None:            # retriever 就是唯一开关
            query = record["source"] # TODO
            docs = self.retriever.retrieve(query, self.top_k)   # query 用 source，不是 ""
            contexts = [doc.text for doc in docs]                # 取出 .text
            self.prompt_style = "rag"
        
        messages = build_messages(record["source"],self.prompt_style,contexts)
        
        return {
            "id":record["id"], 
            "messages":messages, 
            "answer":record["summary"], 
            "contexts":contexts
        }
        