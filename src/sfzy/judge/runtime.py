"""Judge 的推理运行时：生成 + **受限解码打分**。

================================ 两种打分路径 ================================
**受限解码**（默认）：只做一次前向，取最后一个位置的 logits，在
`{'0','1','2','3','4'}` 五个 token 上 softmax，取 argmax。

（已确认 Qwen 与 ChatGLM3 的 "0"~"4" 各自都是单 token。）

============================ 为什么用受限解码而不是"生成再解析" ============================
需求写的是"max_new_tokens 设 2~4、解析成 0~4、非法输出重试"。那是
**文本生成**路径下必须做的事。但这里有一个更好的做法：

    只做一次前向，取最后一个位置的 logits，在 `{'0','1','2','3','4'}`
    这五个 token 上做 softmax，取 argmax。

（已确认 Qwen2.5 里 "0"~"4" 各自是单 token：id 15~19。）

这样：

  * **不可能产生非法输出** —— 采样空间就只有五档，不需要解析、不需要重试
  * **快得多** —— 只做 prefill，不逐 token 解码。48 个 pair 是 48 次前向，
    不是 48 × 4 次
  * **temperature=0 / do_sample=False 自动满足** —— argmax 就是确定性
  * **多了一个诊断信号**：`pmax`（最可能的那个分数占多少概率）。
    pmax 低说明裁判在犹豫，这是"这一批评分不可信"的早期警报，
    生成路径拿不到这个信息

生成路径仍然保留（`judge_mode="generate"`），用来做对照 —— 如果两条路径
给出的分数分布差很多，说明模型的行为不稳定，那本身就是个发现。

============================ 三种模型形态都支持 ============================
这个运行时**不自己加载模型**（`build_runtime` 是个便捷函数，仅此而已），
所以三种形态都能挂上来：

  1. 独立的 HF 模型（Qwen2.5-7B 在卡 1 当裁判）
  2. **策略模型关掉 LoRA adapter**（冻结的底座，零额外显存）——
     见 `context=` 参数，训练器里拿 KL 参考策略用的就是同一招
  3. 任何其他 HF 模型

ChatGLM3 那一类自带生成循环的模型走 `native_generation=True`：
**生成逐条走它自己的 `stream_generate`，打分走无 padding 的单条前向**。
两条路都绕开了 padding —— 它的 `get_masks` 假设输入没有 padding，
批量会直接崩（见 `models/compat.py` 记的四个坑）。

============================ 三处工程约束 ============================
  * 模型 `eval()` + `requires_grad_(False)`：需求明确要求完全冻结、不参与反传
  * 全程 `torch.inference_mode()`：不只是 `no_grad()`，前者连版本计数都不建
  * 左 padding + attention_mask：因果生成里右 padding 会让位置错位
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]

DIGITS = "01234"


class TorchRuntime:
    """一个冻结的模型，负责把 prompt 批量变成文本或分数。

    模型由调用方传进来，运行时只负责"怎么用" —— 这样同一份代码既能服务
    独立加载的裁判模型，也能服务**关掉 adapter 的策略模型**。
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
        max_batch_size: int = 8,
        max_input_tokens: int = 4096,
        context: Optional[Callable[[], Any]] = None,
        native_generation: Optional[bool] = None,
    ) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.device = device or str(next(model.parameters()).device)
        self.max_batch_size = max_batch_size
        self.max_input_tokens = max_input_tokens
        # context：每次前向/生成前要进入的上下文。用来表达"这次用的是关掉
        # adapter 的底座"。传 None 就是普通模型。
        self._context = context
        # 自带生成循环的模型（ChatGLM3）：生成不能批量、打分不能带 padding
        self.native = (
            hasattr(model, "stream_generate") if native_generation is None
            else native_generation
        )
        self.model.eval()
        for param in self.model.parameters():   # 需求第 8 条：Judge 完全冻结
            param.requires_grad_(False)

        self._torch = torch
        # 五个分数 token 的 id。取 encode 的**最后一个** id：有的 tokenizer
        # 会在数字前加前缀 token，取错就整批偏移。
        self._digit_ids = [
            self.tokenizer.encode(d, add_special_tokens=False)[-1] for d in DIGITS
        ]
        # 记录本进程累计调用了多少次、花了多少 token，便于估 GRPO 的预算
        self.stats = {"calls": 0, "sequences": 0}

    def _ctx(self) -> Any:
        return self._context() if self._context is not None else nullcontext()

    # ------------------------------------------------------------------
    def render(self, messages: List[Dict[str, str]]) -> str:
        """套模型自己的 chat 模板 —— 换 Judge 模型时这里不用改。"""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "\n".join(m["content"] for m in messages)

    def _encode(self, prompts: Sequence[str]) -> Tuple[Any, Any]:
        tok = self.tokenizer
        old_side = tok.padding_side
        tok.padding_side = "left"           # 因果生成必须左 padding
        try:
            enc = tok(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                add_special_tokens=False,
            )
        finally:
            tok.padding_side = old_side
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @staticmethod
    def _batched(items: Sequence[Any], size: int):
        for start in range(0, len(items), size):
            yield items[start:start + size]

    # ------------------------------------------------------------------
    def generate_batch(self, prompts: Sequence[str], max_new_tokens: int = 512) -> List[str]:
        """批量生成。用于六要素提取（输出是 JSON，长短不定）。"""
        torch = self._torch
        if self.native:
            # ChatGLM3 的 get_masks 假设输入没有 padding，批量必然崩；
            # 走它自己的 stream_generate，逐条来。慢，但只有提取这一步慢。
            return [self._generate_one_native(p, max_new_tokens) for p in prompts]
        out: List[str] = []
        with torch.inference_mode(), self._ctx():
            for chunk in self._batched(list(prompts), self.max_batch_size):
                input_ids, attention_mask = self._encode(chunk)
                gen = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,                 # 需求第 1、2 条：确定性
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
                new = gen[:, input_ids.shape[1]:]
                out.extend(
                    self.tokenizer.decode(row, skip_special_tokens=True).strip()
                    for row in new
                )
                self.stats["calls"] += 1
                self.stats["sequences"] += len(chunk)
        return out

    def _generate_one_native(self, prompt: str, max_new_tokens: int) -> str:
        """逐条生成，走模型自带的 stream_generate（它是生成器，取最后一次输出）。"""
        torch = self._torch
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(self.device)
        with torch.inference_mode(), self._ctx():
            last = ids
            for last in self.model.stream_generate(
                input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False
            ):
                pass
            self.stats["calls"] += 1
            self.stats["sequences"] += 1
        return self.tokenizer.decode(
            last[0][ids.shape[1]:], skip_special_tokens=True
        ).strip()

    # ------------------------------------------------------------------
    def score_digits_batch(
        self, prompts: Sequence[str], digits: str = DIGITS
    ) -> Tuple[List[int], List[float], List[List[float]]]:
        """受限解码：返回 (分数, 最大概率, 五档概率)。

        只做一次前向，取**最后一个位置**的 logits。因为输入是左 padding，
        最后一个位置对所有序列都是"prompt 结束、该吐答案"的位置。
        """
        torch = self._torch
        assert digits == DIGITS, "分数词表变了要同步改 _digit_ids"
        scores: List[int] = []
        pmaxs: List[float] = []
        probs_all: List[List[float]] = []
        with torch.inference_mode(), self._ctx():
            if self.native:
                # 无 padding 的单条前向：绕开 get_masks 对 padding 的假设
                for prompt in prompts:
                    ids = self.tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=self.max_input_tokens,
                    )["input_ids"].to(self.device)
                    logits = self.model(input_ids=ids).logits[0, -1, :]
                    probs = torch.softmax(logits[self._digit_ids].float(), dim=-1)
                    best = int(probs.argmax())
                    scores.append(best)
                    pmaxs.append(float(probs[best]))
                    probs_all.append([round(float(p), 4) for p in probs.tolist()])
                    self.stats["calls"] += 1
                    self.stats["sequences"] += 1
                return scores, pmaxs, probs_all
            for chunk in self._batched(list(prompts), self.max_batch_size):
                input_ids, attention_mask = self._encode(chunk)
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits[:, -1, :]                       # (B, vocab)
                digit_logits = logits[:, self._digit_ids].float()   # (B, 5)
                probs = torch.softmax(digit_logits, dim=-1)
                best = probs.argmax(dim=-1)
                scores.extend(int(x) for x in best.tolist())
                pmaxs.extend(float(x) for x in probs.max(dim=-1).values.tolist())
                probs_all.extend([[round(float(p), 4) for p in row] for row in probs.tolist()])
                self.stats["calls"] += 1
                self.stats["sequences"] += len(chunk)
        return scores, pmaxs, probs_all


def build_runtime(
    model_path: str,
    device: str = "auto",
    dtype: str = "bfloat16",
    max_batch_size: int = 8,
    max_input_tokens: int = 4096,
) -> TorchRuntime:
    """便捷函数：加载一个独立的 HF 模型当裁判。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    return TorchRuntime(
        model=model, tokenizer=tokenizer, device=device,
        max_batch_size=max_batch_size, max_input_tokens=max_input_tokens,
    )


# 旧名字保留：外部（含文档）引用过 QwenRuntime
QwenRuntime = TorchRuntime


def resolve_model_path(path: str) -> str:
    """相对路径按项目根解析，方便在任意目录下跑脚本。"""
    p = Path(path)
    if p.is_absolute() or not (ROOT / p).exists():
        return str(p)
    return str(ROOT / p)
