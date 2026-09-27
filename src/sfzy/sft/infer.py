"""批量推理：生成摘要。

============================ 你要实现的文件 ============================

-------------------------------- 训练和推理必须对齐的三件事 --------------------------------
1. **encode_prompt 的 add_generation_prompt 要一致**（都是 True）。
   不一致的话，模型训练时没见过推理时的上下文格式，生成质量会明显掉。
2. **prompt 模板要一致**。训练用 structured、推理用 zeroshot 是不可比的。
3. **system prompt 要一致**。

这三条是"离线指标好看、线上效果差"的常见原因，也是消融实验必须控制住的变量。

-------------------------------- 生成参数怎么调 --------------------------------
* max_new_tokens：参考摘要平均 280 字，取 256~384 比较合适。
  设太大会让模型有机会啰嗦，而且显著拖慢推理。
* repetition_penalty：中文生成很容易复读，1.1 左右是常用起点。
* 推理时用贪心（do_sample=False）还是采样？
  评测通常用贪心，因为要可复现；do_sample=True 时记得固定 seed。
=====================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from sfzy.data.prompts import build_messages


def generate_one(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
    do_sample: bool = False,
) -> str:
    """单条生成，返回**只含新生成部分**的摘要文本。

    两个容易踩的坑：
      * 必须把 prompt 部分的 token 切掉，只 decode 新生成的部分
        （input_ids.shape[-1] 之后才是答案）。不切的话输出里会带一大段原文。
      * 用 tokenizer.decode(..., skip_special_tokens=True)，
        否则 eos 标记会出现在结果里。

    提示：model.generate 要放在 torch.no_grad() 里，并且先 model.eval()。
    """
    # TODO
    raise NotImplementedError("TODO: 实现 generate_one")


def summarize_records(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    prompt_style: str = "structured",
    batch_size: int = 4,
    max_new_tokens: int = 256,
    retriever: Optional[Any] = None,
    **gen_kwargs: Any,
) -> List[Dict[str, str]]:
    """批量生成，返回 [{"id": ..., "summary": 预测}, ...]。

    输出格式刻意和 data/prepare.py 的 to_record 对齐（id + summary 两个字段），
    这样 scripts/evaluate.py 不用做任何转换就能直接读。

    提示：真正的 batch 推理需要左侧 padding 并对齐 prompt 长度，
    实现复杂度不低。第一版先用 batch_size=1 的循环跑通，
    等指标对得上了再优化吞吐 —— 先正确再快。
    """
    # TODO
    raise NotImplementedError("TODO: 实现 summarize_records")
