"""批量推理：用训练好的模型生成摘要。

训练和推理必须对齐的三件事，对不齐就会出现"离线指标好看、线上效果差"：
  1. encode_prompt 的 add_generation_prompt 要一致（都是 True）
  2. prompt 模板要一致（训练用 structured、推理用 zeroshot 是不可比的）
  3. system prompt 要一致

所以这里不重新拼 prompt，而是复用 data/prompts.build_messages —— 和训练时
走的是同一个函数。
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch

from sfzy.data.prompts import build_messages
from sfzy.models.chat_template import decode, encode_prompt
from sfzy.utils.logging import get_logger

logger = get_logger("infer")


def generate_one(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 384,
    max_length: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
    do_sample: bool = False,
) -> str:
    """单条生成，返回**只含新生成部分**的摘要文本。

    三个容易踩的坑：

    **必须切掉 prompt 的 token。** 只 decode `out[0][len(prompt_ids):]`，
    否则输出里会带上一大段原文。这个 bug 不报错，只是输出看起来莫明其妙地长。

    **必须 skip_special_tokens。** 不然 eos 之类的标记会混进结果里。

    **必须 strip。** 实测旧模型 100% 的输出以空白开头（`' \\n 原被告系...'`），
    那是模板拼接的产物。不 strip 会带进评测和后续的事实提取。

    评测默认用贪心（do_sample=False）—— 要可复现。做 RL 的 rollout 时
    才需要采样，那时显式传 do_sample=True 和 temperature。
    """
    prompt_ids = encode_prompt(tokenizer, messages, add_generation_prompt=True)

    # 截断方式必须和训练时一致：collator._truncate 保留 prompt 的**尾部**
    # （判决书的「本院认为」「判决如下」都在结尾，是最该保留的部分）。
    # 不截断的话，训练时模型只见过前 2048 个 token，推理时却喂 2600 个 ——
    # 这种不一致不会报错，只会让生成质量莫名其妙地变差，而且推理慢好几倍。
    if max_length is not None and len(prompt_ids) > max_length:
        prompt_ids = prompt_ids[-max_length:]

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # temperature / top_p 只在采样时有意义；贪心时传了会报 warning
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, **gen_kwargs)

    new_ids = output[0][len(prompt_ids):].tolist()
    return decode(tokenizer, new_ids).strip()


def summarize_records(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    prompt_style: str = "structured",
    retriever: Optional[Any] = None,
    max_new_tokens: int = 384,
    max_length: Optional[int] = None,
    log_every: int = 50,
    **gen_kwargs: Any,
) -> Iterator[Dict[str, str]]:
    """逐条生成，**边生成边 yield**。

    为什么是生成器而不是返回列表：三元组生成在 6B 模型上要跑几个小时，
    Kaggle 会话可能中途被掐。生成器让调用方可以边跑边写盘，
    中断后按已完成的 id 跳过，不用从头再来。

    retriever 为 None 就是纯 SFT；传了就在取样本时检索并拼进 prompt
    （和 sft/dataset.py 的约定一致）。

    先按 batch_size=1 实现。真正的 batch 推理需要左侧 padding 并对齐
    prompt 长度，复杂度不低 —— **先正确再快**，等指标对得上了再优化吞吐。
    """
    for index, record in enumerate(records, 1):
        contexts: List[str] = []
        if retriever is not None:
            docs = retriever.retrieve(record["source"], getattr(retriever, "top_k", 5))
            contexts = [d.text for d in docs]

        messages = build_messages(record["source"], prompt_style, contexts)
        summary = generate_one(
            model, tokenizer, messages, max_new_tokens=max_new_tokens,
            max_length=max_length, **gen_kwargs
        )

        if log_every and index % log_every == 0:
            logger.info("已生成 %d/%d 条", index, len(records))

        yield {"id": record["id"], "summary": summary}


# ---------------------------------------------------------------------------
# 批处理生成
# ---------------------------------------------------------------------------
def _encode_with_truncation(
    tokenizer: Any, messages: List[Dict[str, str]], max_length: Optional[int]
) -> List[int]:
    """编码单条 prompt 并按训练时的规则截断（保留尾部）。"""
    ids = encode_prompt(tokenizer, messages, add_generation_prompt=True)
    if max_length is not None and len(ids) > max_length:
        ids = ids[-max_length:]
    return ids


def generate_batch(
    model: Any,
    tokenizer: Any,
    messages_list: List[List[Dict[str, str]]],
    max_new_tokens: int = 384,
    max_length: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
    do_sample: bool = False,
    num_return_sequences: int = 1,
) -> List[str]:
    """批量生成，返回和输入等长的摘要列表。

    **为什么必须左 padding。** 因果语言模型自回归生成，只有左 padding 才能
    保证每个序列的真实内容都紧贴生成位置 —— 右 padding 会让位置编码和
    注意力掩码错位，输出会莫名其妙地错乱。而 `load_tokenizer` 里设的是
    `padding_side="right"`（批训练时希望 padding 在末尾），所以这里临时改。

    **实测收益**（本地 3050 + Qwen2.5-0.5B，8 条 × 128 token）：
        batch=1 顺序   13.9 tok/s
        batch=8        70.6 tok/s   ← 5.1x

    这条路径对三元组生成特别划算：prompt 都被截到同一个 max_length，
    长度接近，padding 浪费很小。

    `num_return_sequences=G` 时每个 prompt 返回 G 条连续的结果，
    这正是 GRPO rollout 需要的形态。
    """
    if not messages_list:
        return []

    encodings = [_encode_with_truncation(tokenizer, m, max_length) for m in messages_list]
    pad_id = tokenizer.pad_token_id
    padded_len = max(len(e) for e in encodings)

    input_ids = torch.tensor(
        [[pad_id] * (padded_len - len(e)) + e for e in encodings], dtype=torch.long
    ).to(next(model.parameters()).device)
    attention_mask = torch.tensor(
        [[0] * (padded_len - len(e)) + [1] * len(e) for e in encodings], dtype=torch.long
    ).to(input_ids.device)

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_return_sequences,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"          # 生成阶段必须是左 padding
    try:
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )
    finally:
        tokenizer.padding_side = old_side

    results = []
    for i in range(len(encodings)):
        chunk = output[i * num_return_sequences:(i + 1) * num_return_sequences]
        for row in chunk:
            results.append(decode(tokenizer, row[padded_len:].tolist()).strip())
    return results


def summarize_records_batched(
    model: Any,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    prompt_style: str = "structured",
    retriever: Optional[Any] = None,
    batch_size: int = 8,
    max_new_tokens: int = 384,
    max_length: Optional[int] = None,
    log_every: int = 50,
    **gen_kwargs: Any,
) -> Iterator[Dict[str, str]]:
    """批处理版本，边生成边 yield（保持和 summarize_records 一样的接口）。

    分批而不是一次全喂进去：6B 模型在 16GB 的 T4 上，
    一次塞太多条会 OOM，而且 OOM 了整批都要重来。
    """
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        messages_list = []
        for record in chunk:
            contexts: List[str] = []
            if retriever is not None:
                docs = retriever.retrieve(record["source"], getattr(retriever, "top_k", 5))
                contexts = [d.text for d in docs]
            messages_list.append(build_messages(record["source"], prompt_style, contexts))

        summaries = generate_batch(
            model, tokenizer, messages_list, max_new_tokens=max_new_tokens,
            max_length=max_length, **gen_kwargs,
        )
        for record, summary in zip(chunk, summaries):
            yield {"id": record["id"], "summary": summary}

        if log_every and (start + len(chunk)) % log_every < batch_size:
            logger.info("已生成 %d/%d 条", start + len(chunk), len(records))
