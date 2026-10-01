"""GRPO 训练循环：rollout → 打分 → advantage → 更新。

============================ 和 SFT trainer 的区别 ============================
SFT 是"给定输入和答案，算 loss 反向传播"；GRPO 要多一步**在线采样**：

    for 每一批 prompt:
        1. rollout     —— 每个 prompt 采样 G 条输出（批处理生成）
        2. 打分        —— 规则奖励（门控 → 事实覆盖 + ROUGE）
        3. old_logprobs —— 采样那一刻的策略对数概率，**必须 detach**
        4. advantage   —— 组内归一化 + 长度归一化 + 组内过滤
        5. 更新        —— GRPO loss + KL 惩罚，反向传播

============================ 四个工程要点 ============================
**一、rollout 必须用批处理。** 实测 batch=8 相对 batch=1 有 5~6 倍加速。
GRPO 的采样量是 SFT 的 G 倍（G=8），不批处理根本跑不完。

**二、old_logprobs 必须在参数更新之前算。** 它是"采样时那个策略"的概率；
如果先更新再算，ratio 就不是从 1 出发，裁剪会一直生效、梯度几乎无效。
这是 GRPO 实现里最容易搞错的地方，而且不报错。

**三、组内过滤省的是真实算力。** 奖励全同的组 advantage 恒为 0，
反向传播纯属浪费。实测里被门控全拦的组会很多，过滤掉能省一大截。

**四、生成用 eval、训练前向用 train。** 见 `rollout()` 和 `step()` 里的说明：
漏掉 train，ChatGLM3 的梯度检查点会静默失效（激活按「每层完整保存」算，
长序列必然 OOM）；漏掉 eval，生成会因为 KV cache 被关掉而退化成 O(n²)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from sfzy.config import Config
from sfzy.data.prompts import build_messages
from sfzy.models.chat_template import encode_prompt
from sfzy.rl.dpo import sequence_logprob
from sfzy.rl.grpo import compute_advantages, group_mask, grpo_loss, kl_penalty
from sfzy.rl.reward import (
    compute_rewards,
    summarize_gate_reasons,
)
from sfzy.sft.checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint
from sfzy.sft.infer import generate_batch
from sfzy.utils.logging import get_logger
from sfzy.utils.tracking import group_metrics

logger = get_logger("grpo")


@dataclass
class RLState:
    step: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)
    # 实验跟踪的 run_id。存进 checkpoint 才能在续训时接回 swanlab 上原来那条
    # 曲线；不存的话一次断点重启就会在面板上多出一条断掉的新曲线。
    tracker_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "history": self.history,
            "tracker_run_id": self.tracker_run_id,
        }


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
        scorer: Any = None,
    ) -> None:
        """参考模型不传时用 `disable_adapter()` 拿 —— LoRA 底座是冻结的，
        关掉 adapter 就是 SFT 后的参考策略，不用额外加载一份权重。

        `scorer` 是裁判（`sfzy.judge.scorer.FactConsistencyScorer`）。现在只有
        `fact_judge` 一种奖励模式，没有裁判直接报错 —— 配置说要接裁判却没有
        裁判，静默降级是最坏的结果。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.reference_model = reference_model
        self.output_dir = Path(output_dir or cfg.path_("rl.output_dir", "outputs/grpo"))
        self.tracker = tracker
        self.is_main_process = is_main_process
        self.scorer = scorer

        rl = cfg.rl
        self.group_size = rl.get("group_size", 8)
        self.max_new_tokens = rl.get("max_new_tokens", 384)
        self.max_length = rl.get("max_length", 2048)
        self.temperature = rl.get("temperature", 0.9)
        self.top_p = rl.get("top_p", 0.95)
        self.clip_ratio = rl.get("clip_ratio", 0.2)
        self.kl_coef = rl.get("kl_coef", 0.0)
        self.length_mode = rl.get("length_norm", "sqrt")
        # logprob 前向的批内切分。None = 一次算完（默认，行为和以前一致）；
        # 设成 1~4 能把 (N, L, vocab) 的 logits 峰值按比例压下来 ——
        # 16GB 卡上 G=8 时这是最有效的一个显存旋钮，且不改变数值。
        self.logprob_micro_batch = rl.get("logprob_micro_batch") or None
        self.prompts_per_step = rl.get("prompts_per_step", 4)
        self.accum_steps = rl.get("grad_accum_steps", 1)
        self.max_grad_norm = rl.get("max_grad_norm", 1.0)
        self.save_every_n_steps = rl.get("save_every_n_steps", 50)
        self.keep_last_n = rl.get("keep_last_n_checkpoints", 2)
        self.log_every_n_steps = rl.get("log_every_n_steps", 1)
        self.reward_cfg = rl.get("reward", None)
        self.resume_from = rl.get("resume_from") or None

        # ---- SFT 基线锚：防止策略整体退化，见 grpo.group_mask ----
        anchor = rl.get("anchor") or {}
        self.anchor_enabled = bool(anchor.get("enabled", False))
        self.anchor_slack = float(anchor.get("slack", 0.05))
        self.baseline_rewards: Dict[str, float] = {}

        mode = (self.reward_cfg or {}).get("mode", "fact_judge")
        self.reward_mode = mode
        if mode != "fact_judge":
            raise ValueError(
                f"未知的 reward.mode={mode!r}：现在只支持 fact_judge。"
            )
        if self.scorer is None:
            raise ValueError(
                "reward.mode=fact_judge 但没传裁判（scorer）。\n"
                "  在配置里写 semantic.backend=fact。"
            )

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=rl.get("learning_rate", 1e-5))
        self.device = next(model.parameters()).device
        self.state = RLState()

    # ------------------------------------------------------------------
    def score_semantic(self, candidates, references, sources) -> Optional[List[Optional[float]]]:
        """批量算裁判分。返回 None 表示这次不接裁判（模式不需要）。"""
        if self.scorer is None:
            return None
        items = [
            {"candidate": c, "reference": r, "source": s}
            for c, r, s in zip(candidates, references, sources)
        ]
        return self.scorer.score_batch(items)

    def prepare_baseline(self, prompts: List[Dict[str, Any]]) -> None:
        """用 prompt 池里的 `sft_output` 预先把 SFT 的奖励算出来，当锚。

        只算一次（SFT 输出是固定的），之后每步查表。
        **必须和训练用同一个奖励函数**：锚和策略得分不在一个尺度上，
        比较就没有意义。所以这里也要过裁判。

        prompt 池没有 `sft_output` 就直接跳过并说明怎么补 ——
        `tools/select_rl_prompts.py --triples ...` 会把 SFT 输出带进来。
        """
        if not self.anchor_enabled:
            return
        with_out = [p for p in prompts if p.get("sft_output")]
        if not with_out:
            logger.warning(
                "anchor.enabled=true 但 prompt 池里没有 sft_output，基线锚不生效。\n"
                "  重新生成 prompt 池：python tools/select_rl_prompts.py --n 1000 \\\n"
                "      --triples data/triples/sft_val_shard0of2.jsonl"
            )
            return

        refs = [p["summary"] for p in with_out]
        srcs = [p["source"] for p in with_out]
        cands = [p["sft_output"] for p in with_out]
        semantic = self.score_semantic(cands, refs, srcs)
        bds = compute_rewards(cands, refs, srcs, self.reward_cfg, semantic)
        for p, bd in zip(with_out, bds):
            self.baseline_rewards[str(p.get("id"))] = bd.total
        vals = list(self.baseline_rewards.values())
        logger.info(
            "SFT 基线锚: %d 条，奖励均值 %.4f（锚 slack=%.3f）",
            len(vals), sum(vals) / len(vals), self.anchor_slack,
        )

    def fill_semantic_gaps(
        self, scores: List[Optional[float]], group_size: int
    ) -> Tuple[List[float], float]:
        """裁判个别失败时不中断训练，用**组内可用分的均值**补上，并返回缺失比例。

        为什么不是直接抛错：300 步的 run 里任何一次 API 超时或截断都会在
        第 137 步崩掉，前面几小时白跑。一个失败样本不该有这种权力。

        为什么用组内均值而不是 0 或全局均值：
          * 填 0 等于"这条很差"，会把优势估计带偏（模型被惩罚了一个和它
            无关的失误）
          * 全局均值跨 prompt 不可比 —— 不同文书的裁判分本身就有系统差
          * 组内均值是**该组的中性点**，归一化后它的 advantage 接近 0，
            既不奖励也不惩罚这一条

        整组都缺失时填 0：语义项在组内变成常数，GRPO 会自然忽略它，
        这一组仍能靠 ROUGE 和事实项学习。缺失比例记进日志 —— 超过 10%
        说明裁判配置有问题，该去查超时和 max_new_tokens，而不是接着跑。
        """
        filled: List[float] = [0.0] * len(scores)
        missing = 0
        for start in range(0, len(scores), group_size):
            chunk = scores[start:start + group_size]
            ok = [s for s in chunk if s is not None]
            fallback = sum(ok) / len(ok) if ok else 0.0
            missing += sum(1 for s in chunk if s is None)
            for i, s in enumerate(chunk):
                filled[start + i] = float(s) if s is not None else fallback
        return filled, missing / max(len(scores), 1)

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

        # 生成必须切到 eval，生成完再切回来。
        #
        # **为什么生成要 eval。** ChatGLM3 的 GLMTransformer.forward 里是
        #
        #     if self.gradient_checkpointing and self.training:
        #         if use_cache:
        #             logger.warning_once("`use_cache=True` is incompatible ...")
        #             use_cache = False
        #
        # 也就是说 **train 模式下它会为了梯度检查点把 KV cache 关掉**，
        # 自回归生成退化成每个 token 重算整个前缀。rollout 是 GRPO 的瓶颈
        # （G 倍生成量），这个代价受不了。dropout 也顺带在采样时关掉，
        # 生成的分布更稳定。
        #
        # **为什么生成完要切回来。** 见 `step()` 里那段说明：训练前向必须是
        # train，否则梯度检查点不生效。两个模式各司其职，谁都不能省。
        was_training = self.model.training
        self.model.eval()
        try:
            # generate_batch 的 num_return_sequences 会为每个 prompt 连续产出 G 条
            responses = generate_batch(
                self.model, self.tokenizer, messages_list,
                max_new_tokens=self.max_new_tokens, max_length=self.max_length,
                do_sample=True, temperature=self.temperature, top_p=self.top_p,
                num_return_sequences=self.group_size,
            )
        finally:
            if was_training:
                self.model.train()

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
        semantic = self.score_semantic(batch["responses"], refs, sources)
        judge_missing = 0.0
        if semantic is not None:
            semantic, judge_missing = self.fill_semantic_gaps(semantic, self.group_size)
        breakdowns = compute_rewards(
            batch["responses"], refs, sources, self.reward_cfg, semantic
        )
        # ★ 奖励/长度/基线必须建在**模型所在的设备**上。
        # 默认的 torch.tensor([...]) 建在 CPU；后续 advantages 也就留在 CPU，
        # 而 logprobs 在 GPU —— grpo_loss 里 `ratio * advantages` 会直接报
        # "Expected all tensors to be on the same device, but found at least
        # two devices, cuda:0 and cpu!"。CPU 冒烟测试永远碰不到（两边都是 CPU），
        # 只有真机跑 GRPO 才会暴露。
        rewards = torch.tensor(
            [b.total for b in breakdowns], dtype=torch.float32, device=self.device
        )
        rewards = rewards.reshape(len(prompts), self.group_size)

        # ---- advantage（组内归一化 + 长度归一化 + 组过滤）----
        lengths = torch.tensor(
            [len(r.strip()) for r in batch["responses"]],
            dtype=torch.float32, device=self.device,
        ).reshape(len(prompts), self.group_size)
        # 基线锚：查不到基线的 prompt 用 -1e9 填充 —— 阈值比较永远成立，
        # 等价于"这一组不参与锚过滤"，不会误伤。
        baseline = None
        if self.baseline_rewards:
            baseline = torch.tensor(
                [self.baseline_rewards.get(str(p.get("id")), -1e9) for p in prompts],
                dtype=torch.float32, device=self.device,
            )
        advantages, keep = compute_advantages(
            rewards, lengths, length_mode=self.length_mode,
            baseline=baseline, slack=self.anchor_slack,
        )
        # 把"没区分度"和"被锚挡住"分开记：前者是运气，后者是策略在退步
        anchor_filtered = 0
        if baseline is not None:
            has_var = group_mask(rewards)
            anchor_filtered = int((has_var & ~keep).sum())

        # ---- 训练前向必须显式切到 train 模式 ----
        # HF 的 from_pretrained 结尾会调 model.eval()（它自己的文档里写着
        # "The model is set in evaluation mode by default"），所以模型拿到手时
        # 是 eval。而 ChatGLM3 的 GLMTransformer.forward 里是
        #
        #     if self.gradient_checkpointing and self.training:
        #
        # training=False 会让整个条件短路 —— **梯度检查点设了也不生效**，
        # 激活值按「每层完整保存」算，长序列必然 OOM，而报错栈里全是
        # bitsandbytes 的调用，看不出是这里。
        #
        # 这不是假设：SFT 训练器踩过同一个坑并修了（见 sft/trainer.py 里
        # `self.model.train()` 上面那段注释："我们为此查了三轮"）；GRPO 这条
        # 路径此前漏掉了这一步，于是整个 RL 阶段都在 eval 下跑。
        #
        # 放在这里而不是 `__init__`：rollout 会临时切 eval（见 `rollout()`），
        # 每次进前向都要重新保证一次。dropout 也随之为训练打开。
        self.model.train()

        # ---- old_logprobs：必须在任何参数更新之前算 ----
        with torch.no_grad():
            old_logprobs = sequence_logprob(
                self.model, batch["input_ids"], batch["labels"],
                micro_batch=self.logprob_micro_batch,
            ).detach()

        # ---- 策略 logprob（带梯度）----
        self.optimizer.zero_grad(set_to_none=True)
        logprobs = sequence_logprob(
            self.model, batch["input_ids"], batch["labels"],
            micro_batch=self.logprob_micro_batch,
        )
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
                            ref_model, batch["input_ids"], batch["labels"],
                            micro_batch=self.logprob_micro_batch,
                        ).detach()
                kl_term = kl_penalty(logprobs, ref_logprobs, coef=self.kl_coef)
                loss = loss + kl_term
                kl_value = float(kl_term.detach())

        loss.backward()
        # 返回值是**裁剪前**的梯度总范数。它免费，而且是"训练稳不稳"最快的
        # 单一信号：突然窜到几十说明这一步的 advantage 尺度失控（常见于
        # 奖励被噪声污染、或被裁剪的 ratio 占了大头）。
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm
        )
        self.optimizer.step()

        gate_stats = summarize_gate_reasons(breakdowns)
        metrics = {
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "advantage_abs_mean": float(advantages.abs().mean()),
            "clipped_frac": float(clipped_frac),
            "kl": kl_value,
            "kept_groups": int(keep.sum()),
            "total_groups": len(prompts),
            "kept_ratio": int(keep.sum()) / max(len(prompts), 1),
            "anchor_filtered": anchor_filtered,
            "judge_missing": judge_missing,
            "gating_rate": gate_stats.get("gated", 0) / max(gate_stats.get("total", 1), 1),
            "output_len_mean": float(lengths.mean()),
        }
        # reward_std 是整个 batch 的标准差，只有在 prompts_per_step=1 时才等于
        # 组内标准差。GRPO 真正要的是**组内**方差（组间差异会被归一化消掉），
        # 所以它单独算一份，不管每步几个 prompt 都成立。
        metrics["reward_group_std"] = float(
            rewards.std(dim=-1, unbiased=False).mean()
        )
        # 事实一致性和 ROUGE 的分项也该看 —— 只盯总分判断不出"好在哪"
        metrics["fact_judge"] = float(
            sum(b.fact_judge for b in breakdowns) / len(breakdowns)
        )
        # 事实项的**组内**标准差：均值好看但组内没方差，归一化之后就是常数，
        # 等于白接一套裁判。
        fj = torch.tensor(
            [b.fact_judge for b in breakdowns], dtype=torch.float32, device=self.device
        )
        metrics["fact_judge_group_std"] = float(
            fj.reshape(len(prompts), self.group_size).std(dim=-1, unbiased=False).mean()
        )
        metrics["rouge_l"] = float(sum(b.rouge_l for b in breakdowns) / len(breakdowns))
        # 六要素裁判自己汇报的统计量（需求第十二节）：mean_fact_reward、
        # 各要素均值、0-4 各档比例。只有事实一致性后端有这个钩子，
        # **不需要 trainer 认识六要素**，耦合面就一个方法名。
        if hasattr(self.scorer, "summarize_last"):
            metrics.update(self.scorer.summarize_last())
        return metrics

    # ------------------------------------------------------------------
    def train(self, prompts: List[Dict[str, Any]]) -> RLState:
        """主循环。每步处理 prompts_per_step 个 prompt。"""
        if self.is_main_process:
            # 基线锚只在前 N 条 prompt 上用得上，但 prepare_baseline 会整池算 ——
            # 便宜（一次前向），换来的是每步都能查表。
            self.prepare_baseline(prompts)
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
                self._log_step(self.state.step, n_steps, metrics)
                if self.tracker is not None:
                    # 上报名带 `/<分组>` 前缀（见 tracking.GRPO_METRICS），
                    # 面板上按 reward/group/judge 分块；history 里仍是原始键名。
                    self.tracker.log(group_metrics(metrics), self.state.step)

            if self.is_main_process and self.state.step % self.save_every_n_steps == 0:
                self.save()

        if self.is_main_process:
            self.save()
        return self.state

    # ------------------------------------------------------------------
    def _log_step(self, step: int, n_steps: int, m: Dict[str, float]) -> None:
        """打这一步的日志。奖励只有 fact_judge 一种，列固定。"""
        head = (
            f"step {step}/{n_steps} | "
            f"reward {m['reward_mean']:.4f}±{m['reward_std']:.4f}"
            f"(组内σ{m['reward_group_std']:.4f}) | ROUGE-L {m['rouge_l']:.4f}"
        )
        head += (
            f" | 事实一致性 {m['fact_judge']:.4f}"
            f"(组内σ{m['fact_judge_group_std']:.4f})"
            f" | 裁判缺失 {m['judge_missing'] * 100:.1f}%"
        )
        head += (
            f" | 门控 {m['gating_rate'] * 100:.1f}%"
            f" | 保留组 {m['kept_groups']}/{m['total_groups']}"
            f" | clip {m['clipped_frac'] * 100:.1f}%"
            f" | |g| {m['grad_norm']:.2f} | len {m['output_len_mean']:.0f}"
        )
        logger.info(head)

        # 六要素的逐项明细（需求第十二节）：均值 + 0-4 各档比例。
        if "mean_fact_reward" in m:
            ratio = " ".join(
                f"{lv}:{m.get(f'ratio_score_{lv}', 0.0) * 100:.0f}%" for lv in range(5)
            )
            logger.info(
                "  └ 事实一致性 fact_reward %.4f | 结果分 %.3f | 最低分 %.3f | 档位 %s",
                m["mean_fact_reward"],
                m.get("mean_judgment_result_score", 0.0),
                m.get("mean_min_element_score", 0.0),
                ratio,
            )

        # 裁判整组失败时光看比例没法排查 —— 把最后一条异常打出来。
        # 实测最常见的两种：某篇文书六要素抽不到 2 项（ExtractionFailure），
        # 或者裁判侧那一批前向 OOM。
        errors = getattr(self.scorer, "last_errors", None)
        if m.get("judge_missing", 0.0) > 0 and errors:
            logger.warning(
                "  裁判有 %.0f%% 的组失败，最近一条：%s",
                m["judge_missing"] * 100, str(errors[-1])[:300],
            )

    # ------------------------------------------------------------------
    def save(self, tag: str = "last") -> Path:
        if not self.is_main_process:
            return Path()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = (self.output_dir / "best.pt" if tag == "best"
                else self.output_dir / f"step_{self.state.step:06d}.pt")
        self.state.tracker_run_id = (
            self.tracker.run_id if self.tracker is not None else None
        )
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
