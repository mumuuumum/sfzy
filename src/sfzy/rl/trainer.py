"""GRPO 训练循环：rollout → 打分 → advantage → 更新。

============================ 和 SFT trainer 的区别 ============================
SFT 是"给定输入和答案，算 loss 反向传播"；GRPO 要多一步**在线采样**：

    for 每一批 prompt:
        1. rollout     —— 每个 prompt 采样 G 条输出（批处理生成）
        2. 打分        —— 规则奖励（门控 → 事实覆盖 + ROUGE）
        3. old_logprobs —— 采样那一刻的策略对数概率，**必须 detach**
        4. advantage   —— 组内归一化 + 长度归一化 + 组内过滤
        5. 更新        —— GRPO loss + KL 惩罚，反向传播

============================ 三个工程要点 ============================
**一、rollout 必须用批处理。** 实测 batch=8 相对 batch=1 有 5~6 倍加速。
GRPO 的采样量是 SFT 的 G 倍（G=8），不批处理根本跑不完。

**二、old_logprobs 必须在参数更新之前算。** 它是"采样时那个策略"的概率；
如果先更新再算，ratio 就不是从 1 出发，裁剪会一直生效、梯度几乎无效。
这是 GRPO 实现里最容易搞错的地方，而且不报错。

**三、组内过滤省的是真实算力。** 奖励全同的组 advantage 恒为 0，
反向传播纯属浪费。实测里被门控全拦的组会很多，过滤掉能省一大截。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from sfzy.config import Config
from sfzy.data.prompts import build_messages
from sfzy.models.chat_template import encode_prompt
from sfzy.rl.dpo import sequence_logprob
from sfzy.rl.grpo import compute_advantages, grpo_loss, kl_penalty
from sfzy.rl.reward import (
    compute_rewards,
    summarize_gate_reasons,
)
from sfzy.sft.checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint
from sfzy.sft.infer import generate_batch
from sfzy.utils.logging import get_logger

logger = get_logger("grpo")


@dataclass
class RLState:
    step: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "history": self.history}


class GRPOTrainer:
    """手写 GRPO 训练循环。"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cfg: Config,
        reference_model: Any = None,
        output_dir: Optional[str] = None,
        tracker: Any = None,
        is_main_process: bool = True,
    ) -> None:
        """参考模型不传时用 `disable_adapter()` 拿 —— LoRA 底座是冻结的，
        关掉 adapter 就是 SFT 后的参考策略，不用额外加载一份权重。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.reference_model = reference_model
        self.output_dir = Path(output_dir or cfg.path_("rl.output_dir", "outputs/grpo"))
        self.tracker = tracker
        self.is_main_process = is_main_process

        rl = cfg.rl
        self.group_size = rl.get("group_size", 8)
        self.max_new_tokens = rl.get("max_new_tokens", 384)
        self.max_length = rl.get("max_length", 2048)
        self.temperature = rl.get("temperature", 0.9)
        self.top_p = rl.get("top_p", 0.95)
        self.clip_ratio = rl.get("clip_ratio", 0.2)
        self.kl_coef = rl.get("kl_coef", 0.0)
        self.length_mode = rl.get("length_norm", "sqrt")
        self.prompts_per_step = rl.get("prompts_per_step", 4)
        self.accum_steps = rl.get("grad_accum_steps", 1)
        self.max_grad_norm = rl.get("max_grad_norm", 1.0)
        self.save_every_n_steps = rl.get("save_every_n_steps", 50)
        self.keep_last_n = rl.get("keep_last_n_checkpoints", 2)
        self.log_every_n_steps = rl.get("log_every_n_steps", 1)
        self.reward_cfg = rl.get("reward", None)
        self.resume_from = rl.get("resume_from") or None

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=rl.get("learning_rate", 1e-5))
        self.device = next(model.parameters()).device
        self.state = RLState()

    # ------------------------------------------------------------------
    def rollout(self, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """对每个 prompt 采样 G 条，返回拼好的张量和元信息。

        返回 dict：
            input_ids  (N, L)   prompt + 生成，左 padding
            labels     (N, L)   prompt 段 -100，只对生成段算 logprob
            responses  (N,)     生成的文本
            prompt_index (N,)   每条对应哪个 prompt（组内归一化要用）
        """
        messages_list = [build_messages(p["source"], self.cfg.path_("rl.prompt_style", "structured"))
                         for p in prompts]

        # generate_batch 的 num_return_sequences 会为每个 prompt 连续产出 G 条
        responses = generate_batch(
            self.model, self.tokenizer, messages_list,
            max_new_tokens=self.max_new_tokens, max_length=self.max_length,
            do_sample=True, temperature=self.temperature, top_p=self.top_p,
            num_return_sequences=self.group_size,
        )

        pad_id = self.tokenizer.pad_token_id
        sequences = []
        prompt_lens = []
        for i, (messages, _) in enumerate(zip(messages_list, prompts)):
            prompt_ids = encode_prompt(self.tokenizer, messages, add_generation_prompt=True)
            if len(prompt_ids) > self.max_length:
                prompt_ids = prompt_ids[-self.max_length:]
            for j in range(self.group_size):
                text = responses[i * self.group_size + j]
                gen_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                gen_ids = gen_ids[: self.max_new_tokens] + [self.tokenizer.eos_token_id]
                sequences.append(prompt_ids + gen_ids)
                prompt_lens.append(len(prompt_ids))

        width = max(len(s) for s in sequences)
        input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        labels = torch.full((len(sequences), width), -100, dtype=torch.long)
        for k, (seq, plen) in enumerate(zip(sequences, prompt_lens)):
            input_ids[k, : len(seq)] = torch.tensor(seq)
            labels[k, plen:len(seq)] = torch.tensor(seq[plen:])

        return {
            "input_ids": input_ids.to(self.device),
            "labels": labels.to(self.device),
            "responses": responses,
            "prompt_index": torch.arange(len(prompts)).repeat_interleave(self.group_size),
            "prompt_lens": prompt_lens,
        }

    # ------------------------------------------------------------------
    def step(self, prompts: List[Dict[str, Any]]) -> Dict[str, float]:
        """一个优化步：rollout → 打分 → advantage → 反向。"""
        batch = self.rollout(prompts)

        # ---- 打分 ----
        refs, sources = [], []
        for i in batch["prompt_index"].tolist():
            refs.append(prompts[i]["summary"])
            sources.append(prompts[i]["source"])
        breakdowns = compute_rewards(batch["responses"], refs, sources, self.reward_cfg)
        rewards = torch.tensor([b.total for b in breakdowns], dtype=torch.float32)
        rewards = rewards.reshape(len(prompts), self.group_size)

        # ---- advantage（组内归一化 + 长度归一化 + 组过滤）----
        lengths = torch.tensor(
            [len(r.strip()) for r in batch["responses"]], dtype=torch.float32
        ).reshape(len(prompts), self.group_size)
        advantages, keep = compute_advantages(rewards, lengths, length_mode=self.length_mode)

        # ---- old_logprobs：必须在任何参数更新之前算 ----
        with torch.no_grad():
            old_logprobs = sequence_logprob(
                self.model, batch["input_ids"], batch["labels"]
            ).detach()

        # ---- 策略 logprob（带梯度）----
        self.optimizer.zero_grad(set_to_none=True)
        logprobs = sequence_logprob(self.model, batch["input_ids"], batch["labels"])
        loss, clipped_frac = grpo_loss(
            logprobs, old_logprobs, advantages, clip_ratio=self.clip_ratio
        )

        # ---- KL 惩罚（可选）----
        kl_value = 0.0
        if self.kl_coef > 0:
            if self.reference_model is not None:
                ref_model = self.reference_model
            elif hasattr(self.model, "disable_adapter"):
                ref_model = self.model
            else:
                ref_model = None
            if ref_model is not None:
                ctx = (self.model.disable_adapter() if ref_model is self.model
                       else torch.no_grad())
                with torch.no_grad():
                    with (ctx if hasattr(ctx, "__enter__") else torch.no_grad()):
                        ref_logprobs = sequence_logprob(
                            ref_model, batch["input_ids"], batch["labels"]
                        ).detach()
                kl_term = kl_penalty(logprobs, ref_logprobs, coef=self.kl_coef)
                loss = loss + kl_term
                kl_value = float(kl_term.detach())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm
        )
        self.optimizer.step()

        gate_stats = summarize_gate_reasons(breakdowns)
        metrics = {
            "loss": float(loss.detach()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "advantage_abs_mean": float(advantages.abs().mean()),
            "clipped_frac": float(clipped_frac),
            "kl": kl_value,
            "kept_groups": int(keep.sum()),
            "total_groups": len(prompts),
            "gating_rate": gate_stats.get("gated", 0) / max(gate_stats.get("total", 1), 1),
            "output_len_mean": float(lengths.mean()),
        }
        # 事实覆盖和 ROUGE 的分项也该看 —— 只盯总分判断不出"好在哪"
        metrics["fact_coverage"] = float(
            sum(b.fact_coverage for b in breakdowns) / len(breakdowns)
        )
        metrics["rouge_l"] = float(sum(b.rouge_l for b in breakdowns) / len(breakdowns))
        return metrics

    # ------------------------------------------------------------------
    def train(self, prompts: List[Dict[str, Any]]) -> RLState:
        """主循环。每步处理 prompts_per_step 个 prompt。"""
        n_steps = len(prompts) // self.prompts_per_step
        logger.info(
            "GRPO 开始: %d 个 prompt，每步 %d 个，共 %d 步，G=%d",
            len(prompts), self.prompts_per_step, n_steps, self.group_size,
        )
        for step in range(self.state.step, n_steps):
            chunk = prompts[step * self.prompts_per_step:(step + 1) * self.prompts_per_step]
            metrics = self.step(chunk)
            self.state.step = step + 1

            if self.is_main_process and (step + 1) % self.log_every_n_steps == 0:
                self.state.history.append({"step": self.state.step, **metrics})
                logger.info(
                    "step %d/%d | reward %.4f±%.4f | ROUGE-L %.4f | 事实 %.4f | "
                    "门控 %.1f%% | clip %.1f%% | len %.0f",
                    self.state.step, n_steps, metrics["reward_mean"], metrics["reward_std"],
                    metrics["rouge_l"], metrics["fact_coverage"],
                    metrics["gating_rate"] * 100, metrics["clipped_frac"] * 100,
                    metrics["output_len_mean"],
                )
                if self.tracker is not None:
                    self.tracker.log(metrics, self.state.step)

            if self.is_main_process and self.state.step % self.save_every_n_steps == 0:
                self.save()

        if self.is_main_process:
            self.save()
        return self.state

    # ------------------------------------------------------------------
    def save(self, tag: str = "last") -> Path:
        if not self.is_main_process:
            return Path()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = (self.output_dir / "best.pt" if tag == "best"
                else self.output_dir / f"step_{self.state.step:06d}.pt")
        save_checkpoint(path=path, model=self.model, optimizer=self.optimizer,
                        state=self.state, only_trainable=True)
        prune_checkpoints(ckpt_dir=self.output_dir, keep_last_n=self.keep_last_n)
        return path

    def resume(self, path: Optional[str] = None) -> RLState:
        spec = path if path is not None else self.resume_from
        if not spec:
            logger.info("未指定 resume_from，从头开始")
            return self.state
        ckpt_path = Path(spec)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume_from 指定的 checkpoint 不存在：{ckpt_path}")
        info = load_checkpoint(ckpt_path, model=self.model, optimizer=self.optimizer)
        self.state = RLState(**(info.get("state") or {}))
        logger.info("已从 %s 恢复：step=%d", ckpt_path, self.state.step)
        return self.state
