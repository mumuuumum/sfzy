"""数据加工：把 Document 变成模型能直接用的规范化数据集。

============================ 你要实现的文件 ============================
验收方式：
    python -m pytest tests/test_prepare.py -q

前提：数据是标准干净的，这里不做脏数据处理。
      长度超限由 collator 阶段 tokenizer 的 max_length 截断解决，
      那是模型侧的事，不是这一层的职责。

-------------------------------- 数据流向 --------------------------------
    schema.read_jsonl(path)   ->  List[Document]
    本模块                     ->  {"train": [...], "dev": [...]}
    save_splits               ->  data/processed/{train,dev}.jsonl
                                  每条形如 {"id", "source", "summary"}

-------------------------------- 这一层只做三件事 --------------------------------
1. **切分 train/dev** —— 不切分就没有可信的指标。切分前必须 shuffle：
   原始文件里相邻的文书可能来自同一批次，不打乱就切会让两边分布不一致。
2. **开发期采样** —— 调 prompt 时不该每次都加载 126MB 全量数据。
3. **统一字段** —— 下游模块只认 {"id", "source", "summary"}，
   不知道原始格式里还有 label 这种东西。

-------------------------------- 设计要点 --------------------------------
用 random.Random(seed) 实例，不要用全局 random。
全局随机状态会被其它模块污染，实验就不可复现了。
=====================================================================
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sfzy.data.schema import Document
from sfzy.utils.io import write_jsonl


def to_record(doc: Document, joiner: str = "") -> Dict[str, Any]:
    """把 Document 转成下游统一的记录格式，**只有三个字段**：

        {"id": ..., "source": "文书全文", "summary": "参考摘要"}

    这是整个项目的接口契约：prompts 读 source，eval 读 summary，
    没有任何模块需要访问原始 JSON。

    提示：joiner 直接透传给 doc.joined()。用空串是默认选择，
    理由在 schema.Document.joined 的文档里写了。
    """
    return {
        "id": doc.id,
        "source": doc.joined(joiner),
        "summary": doc.summary
    }


def sample_documents(
    docs: List[Document],
    sample_size: Optional[int],
    seed: int = 42,
) -> List[Document]:
    """开发期采样。

    * sample_size 为 None，或大于等于 docs 长度时，返回全部。
    * 否则无放回随机抽取 sample_size 条（不能有重复）。
    * 同一个 seed 必须得到完全相同的结果（有测试）。

    提示：rng = random.Random(seed)，然后 rng.sample(...)。
    用全局的 random.sample 会读全局随机状态，破坏可复现性。
    """
    rng = random.Random(seed)
    return rng.sample(docs, len(docs) if sample_size is None or sample_size >= len(docs) else sample_size)


def split_documents(
    docs: List[Document],
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Document], List[Document]]:
    """切分为 (train, dev)。

    要求：
      * 先 shuffle 再切（用 random.Random(seed)）
      * dev 条数 = max(1, int(len(docs) * dev_ratio))；样本数 <= 1 时 dev 为空
      * 同一 seed 结果可复现
      * 两个列表的并集恰好等于输入（不重不漏）
    """
    if len(docs) <= 1:
        return list(docs),[]
    
    shuffled = list(docs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    
    dev_size = max(1,int(len(docs) * dev_ratio))
    
    dev = shuffled[:dev_size]
    train = shuffled[dev_size:]
    return train,dev

def build_splits(
    docs: List[Document],
    joiner: str = "",
    dev_ratio: float = 0.1,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """串起完整流程，返回 {"train": [record, ...], "dev": [record, ...]}。

    顺序：采样 -> 切分 -> 转记录。
    每一步都调用上面已经定义好的函数，不要在这里重复实现逻辑。

    想一想：为什么是先采样再切分，而不是先切分再从各自里面采样？
    两种做法的 dev 集大小会有什么差别？
    """
    samples = sample_documents(docs, sample_size,seed)
    train,dev = split_documents(samples,dev_ratio,seed)

    train_records = [to_record(doc,joiner) for doc in train]
    dev_records = [to_record(doc,joiner) for doc in dev]
    
    return {
        "train": train_records,
        "dev": dev_records
    }

def save_splits(
    splits: Dict[str, List[Dict[str, Any]]],
    outdir: str | Path,
) -> Dict[str, int]:
    """把切分结果落盘为 {outdir}/{split_name}.jsonl，返回每个 split 的条数。

    例如 splits={"train": [...], "dev": [...]} 会写出
        data/processed/train.jsonl
        data/processed/dev.jsonl

    提示：写文件的工具是 sfzy.utils.io.write_jsonl（已在文件顶部 import），
    它会自动创建目录、保证中文不被转义，直接复用。
    """
    res = {}
    for name, split in splits.items():
        res[name] = write_jsonl(Path(outdir) / f"{name}.jsonl", split)
    return res