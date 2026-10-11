"""GRPO 训练循环：rollout → 打分 → advantage → 更新。

============================ 和 SFT trainer 的区别 ============================
SFT 是"给定输入和答案，算 loss 反向传播"；GRPO 要多一步**在线采样**：

    for 每一批 prompt:
        1. rollout     —— 每个 prompt 采样 G 条输出（批处理生成）
        2. 打分        —— 规则奖励（门控 → 事实覆盖 + ROUGE）
        3. old_logprobs —— 采样那一刻的策略对数概率，**必须 detach**
        4. advantage   —— 组内归一化 + 长度归一化 + 组内过滤
        5. 更新 ×μ     —— GRPO loss + KL 惩罚，反向传播，做 μ 次（rl.inner_iterations）
                          μ=1 时 old/新 logprob 同参数 ⇒ ρ≡1 ⇒ clip 不可能生效，
                          此时它等价于"组内归一化的 REINFORCE"；μ>1 才会走 PPO 式的裁剪

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

import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from sfzy.config import Config
from sfzy.data.prompts import build_messages
from sfzy.models.chat_template import decode, encode_prompt
from sfzy.models.lora import lora_snapshot_scale, use_lora_weights
from sfzy.rl.dpo import sequence_logprob
from sfzy.rl.grpo import compute_advantages, group_mask, grpo_loss, kl_penalty
from sfzy.rl.grpo import group_advantages
from sfzy.rl.monitor import GRPOMonitor, MonitorBatch, MonitorConfig
from sfzy.rl.monitor import eval_metrics
from sfzy.rl.reward import (
    compute_rewards,
    compute_pre_gate_rewards,
    summarize_gate_reasons,
)
from sfzy.rl.reward import RESULT_MARKERS
from sfzy.rl.reward_spec import RewardSpec
from sfzy.sft.checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint
from sfzy.sft.infer import generate_batch
from sfzy.eval.rouge import score_corpus
from sfzy.utils.logging import get_logger
from sfzy.utils.tracking import group_metrics

logger = get_logger("grpo")


# 分阶段计时开关：`SFZY_PROFILE=1 python scripts/train_grpo.py ...`。
# 只在需要回答"这一步到底慢在哪"时打开；关着时每步只多一次 `if` 判断。
_PROFILE_ENABLED = os.environ.get("SFZY_PROFILE", "0") == "1"


def _sync() -> None:
    """计时前后都要同步：CUDA 是异步的，不同步量到的是"提交耗时"而不是"执行耗时"。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _timed(sink: Optional[Dict[str, float]], key: str) -> Iterator[None]:
    """把一段耗时累加进 `sink[key]`；`sink is None`（未开 profiling）时零开销。"""
    if sink is None:
        yield
        return
    _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        sink[key] = sink.get(key, 0.0) + (time.perf_counter() - start)


def _as_learning_rate(value: Any) -> float:
    """学习率必须是数字 —— 字符串这种最常见的来源见 config.py 顶部的 YAML 说明。

    真实的报错现场是 `TypeError: '<=' not supported between instances of 'float'
    and 'str'`，抛在 `torch.optim.AdamW.__init__` 里，而没有哪个词提示"去看 yaml"。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"rl.learning_rate 必须是数字，收到 {value!r}（{type(value).__name__}）。\n"
            "  最常见来源：YAML 里写 `learning_rate: 3e-6` —— YAML 1.1 会把**尾数不带"
            "小数点**的科学计数法解析成字符串，它一路走到 AdamW 才炸。\n"
            "  三种写法都行：① `3.0e-6`（带小数点）；② 命令行 "
            "--override rl.learning_rate=3e-6；③ 从文件读入时 config.py 已自动补正，"
            "所以只有**带引号的数字**（\"3e-6\"）才会落到这里。"
        )
    return float(value)


@dataclass
class RLState:
    step: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)
    # 实验跟踪的 run_id。存进 checkpoint 才能在续训时接回 swanlab 上原来那条
    # 曲线；不存的话一次断点重启就会在面板上多出一条断掉的新曲线。
    tracker_run_id: Optional[str] = None
    # 验证集最好成绩（含它发生在第几步）。**必须随 checkpoint 一起存**：
    # 续训时如果不记得之前的最好分，下一次验证就会用一个更差的分去覆盖 best.pt，
    # 而你不会收到任何提示。
    best_eval_score: Optional[float] = None
    best_step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "history": self.history,
            "tracker_run_id": self.tracker_run_id,
            "best_eval_score": self.best_eval_score,
            "best_step": self.best_step,
        }


class GRPOTrainer:
    """手写 GRPO 训练循环。"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cfg: Config,
        reference_model: Any = None,
        reference_lora: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        tracker: Any = None,
        is_main_process: bool = True,
        scorer: Any = None,
        rollout_backend: Any = None,
    ) -> None:
        """KL 的参考策略有两种给法，二选一即可，都不需要第二份 6B 权重：

        * `reference_lora` —— **本项目走这条**。策略与 SFT 策略共用同一个冻结
          底座、只有 LoRA 不同，所以只要在载入 SFT 权重之后 `snapshot_lora()`
          存一份 LoRA 副本（≈118 MB），算 KL 时临时换进去前向一次再换回来。
        * `reference_model` —— 外部已经备好一份冻结模型时直接传（占一份显存）。

        两者都没有时，退回到 `model.disable_adapter()`（那是 **peft.PeftModel**
        的方法；本仓库的手写 LoRA 没有它）。三条路都不通而 `kl_coef > 0`，
        直接报错 —— 静默跳过会让 KL 恒为 0，而曲线看上去"正在做 KL 约束"。

        `scorer` 是裁判（`sfzy.judge.scorer.SixElementScorer`）。奖励由
        `rl.reward.terms` 决定开哪些项；只要有 judge 项而没裁判，直接报错 ——
        配置说要接裁判却没有裁判，静默降级是最坏的结果。

        `rollout_backend` 是**可选的 rollout 后端对象**（目前只有
        `sfzy.rl.vllm_rollout.VLLMRollout`）：不传就用原来的 HF 逐条生成。
        对象由装配层（`scripts/train_grpo.py`）构造 —— 这样 trainer 完全不认识
        vLLM，没装 vllm 的环境也照常跑 `hf` 后端。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.reference_model = reference_model
        self.output_dir = Path(output_dir or cfg.path_("rl.output_dir", "outputs/grpo"))
        self.tracker = tracker
        self.is_main_process = is_main_process
        self.scorer = scorer
        self.rollout_backend = rollout_backend

        rl = cfg.rl
        self.group_size = rl.get("group_size", 8)
        self.max_new_tokens = rl.get("max_new_tokens", 384)
        self.max_length = rl.get("max_length", 2048)
        self.temperature = rl.get("temperature", 0.9)
        self.top_p = rl.get("top_p", 0.95)
        self.clip_ratio = rl.get("clip_ratio", 0.2)
        self.kl_coef = rl.get("kl_coef", 0.0)
        # μ：一次 rollout 做几次梯度更新（DeepSeekMath Algorithm 1 的内层循环）。
        #   μ=1（默认）：old/新 logprob 在同一参数状态下算 ⇒ ρ≡1 ⇒ clip 不可能生效，
        #                整个流程退化成"组内归一化 + 长度缩放的 REINFORCE"；
        #   μ>1：内层第 2 次起 π_θ ≠ π_θold，ρ≠1，clip/信任域才真正起作用；
        #        代价是同一批 rollout 被用来做 μ 次更新（生成是最贵的一环，所以
        #        μ 大省的是生成时间，不省反传时间）。
        raw_mu = rl.get("inner_iterations", 1)
        try:
            self.inner_iterations = int(raw_mu)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"rl.inner_iterations 必须是整数（μ），收到 {raw_mu!r}"
            ) from exc
        if self.inner_iterations < 1:
            raise ValueError(
                f"rl.inner_iterations 必须 ≥ 1，收到 {self.inner_iterations}"
                "（1 = 每个 rollout 只更新一次，即原来的行为）"
            )
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
        # `SFZY_PROFILE=1` 时，step() 会把本步各阶段耗时写进这里（默认 None）
        self._prof: Optional[Dict[str, float]] = None

        # ---- SFT 基线锚：防止策略整体退化，见 grpo.group_mask ----
        anchor = rl.get("anchor") or {}
        self.anchor_enabled = bool(anchor.get("enabled", False))
        self.anchor_slack = float(anchor.get("slack", 0.05))
        self.baseline_rewards: Dict[str, float] = {}

        # ---- KL 的参考策略：必须有，不许静默跳过 ----
        # 优先级：reference_model > reference_lora（换旁路）> disable_adapter()。
        # 最后那条只对 peft.PeftModel 有效：transformers 4.40 的 PreTrainedModel
        # 只有 disable_adapters（复数），本仓库用的又是手写 LoRA，所以实际上
        # 要么传 reference_lora，要么把 kl_coef 保持 0。
        self.reference_lora = dict(reference_lora) if reference_lora else None
        if self.kl_coef > 0:
            if self.reference_model is not None:
                logger.info("KL 参考策略 = 外部传入的 reference_model")
            elif self.reference_lora is not None:
                b_abs, total_abs = lora_snapshot_scale(self.reference_lora)
                logger.info(
                    "KL 参考策略 = SFT 的 LoRA 快照（%d 个张量，|B| 和 %.4g，|W| 和 %.4g）",
                    len(self.reference_lora), b_abs, total_abs,
                )
                if b_abs == 0.0:
                    logger.warning(
                        "LoRA 快照里的 lora_B 全是 0：参考策略等于**原始底座**，"
                        "不是 SFT 策略。检查快照是不是在载入 SFT 权重之前取的。"
                    )
            elif hasattr(self.model, "disable_adapter"):
                logger.info("KL 参考策略 = disable_adapter()（PEFT 模型）")
            else:
                raise ValueError(
                    "rl.kl_coef > 0，但拿不到参考策略。\n"
                    "  本项目的 LoRA 是手写的（models/lora.py），模型上没有 "
                    "disable_adapter（那是 peft.PeftModel 的方法），所以必须显式给：\n"
                    "    · 走 scripts/train_grpo.py：它会在载入 SFT adapter 之后自动 "
                    "snapshot_lora(model) 并传进来；\n"
                    "    · 自己构造 GRPOTrainer：传 reference_lora=snapshot_lora(模型) "
                    "或 reference_model=<冻结的 SFT 模型>。\n"
                    "  不想做 KL 约束就把 rl.kl_coef 保持 0.0。"
                )

        # ---- μ（rl.inner_iterations）的启动播报 ----
        if self.inner_iterations > 1:
            logger.info(
                "内层更新 μ=%d：一次 rollout 做 %d 次梯度更新（optimizer 更新数 = "
                "训练步数 × %d）。第 2 次起 ρ≠1，clip_ratio=%.2f 才开始真正生效。",
                self.inner_iterations, self.inner_iterations, self.inner_iterations,
                self.clip_ratio,
            )
            if self.clip_ratio <= 0 and self.kl_coef <= 0:
                logger.warning(
                    "μ=%d 但 clip_ratio 与 kl_coef 都是 0：同一批数据连续更新多次却"
                    "没有任何信任域约束，策略可能一步走太远。",
                    self.inner_iterations,
                )

        # ---- 奖励规格：开哪些项、权重、要不要裁判，全在这里定 ----
        self.reward_spec = RewardSpec.from_config(self.reward_cfg)
        self.reward_mode = "terms"
        if self.reward_spec.needs_judge():
            if self.scorer is None:
                raise ValueError(
                    "奖励里有 judge 项（"
                    f"{sorted(self.reward_spec.needed_signals)}），但没传裁判（scorer）。\n"
                    "  在配置里写 semantic.backend=fact。"
                )
            missing = self.reward_spec.needed_signals - set(self.scorer.available_signals())
            if missing:
                raise ValueError(
                    f"奖励需要的裁判信号 {sorted(missing)} 这个裁判产不出来，"
                    f"它只有 {sorted(self.scorer.available_signals())}。\n"
                    "  要么换裁判，要么把对应的 reward term 关掉。"
                )

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=_as_learning_rate(rl.get("learning_rate", 1e-5))
        )
        self.device = next(model.parameters()).device
        self.state = RLState()

        # ---- 监控层（只读派生，不参与任何训练计算）----
        enabled = {t.name for t in self.reward_spec.enabled_terms}
        self.monitor = GRPOMonitor(
            MonitorConfig.from_config(rl.get("monitor")),
            fact_enabled="fact_consistency" in enabled,
            coverage_enabled="element_coverage" in enabled,
            ee_enabled="expression_efficiency" in enabled,
        )
        # 本步要上报给 tracker 的**完整分组指标**（与 history 里的扁平键分开）。
        self._last_monitor_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def score_semantic(
        self, candidates, references, sources
    ) -> Optional[Dict[str, List[Optional[float]]]]:
        """批量算裁判信号。

        返回 `{信号名: 每条候选一个值}`；返回 None 表示这次不需要裁判
        （没有任何 judge 项被启用）。
        """
        if self.scorer is None or not self.reward_spec.needs_judge():
            return None
        items = [
            {"candidate": c, "reference": r, "source": s}
            for c, r, s in zip(candidates, references, sources)
        ]
        records = self.scorer.score_batch_signals(items)
        if len(records) != len(items):
            raise ValueError(
                f"裁判返回 {len(records)} 条信号，输入是 {len(items)} 条，数量对不上。"
            )
        return {
            name: [rec.get(name) for rec in records]
            for name in sorted(self.reward_spec.needed_signals)
        }

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
        signals = self.score_semantic(cands, refs, srcs)
        if signals is None:
            signals = {}
        # 基线锚是每 prompt 一条（不是每 prompt G 条），不能用组内均值补缺 ——
        # 缺信号的 SFT 输出直接跳过，填 0 会把它变成一个假的低锚。
        required = sorted(self.reward_spec.needed_signals)
        keep = [
            i for i in range(len(with_out))
            if all(signals.get(name, [None] * len(with_out))[i] is not None for name in required)
        ]
        if len(keep) < len(with_out):
            logger.warning(
                "基线锚：%d 条 SFT 输出缺裁判信号，已跳过（不填 0）。",
                len(with_out) - len(keep),
            )
        if not keep:
            logger.warning("基线锚：没有一条拿到完整裁判信号，锚不生效。")
            return
        kept = [with_out[i] for i in keep]
        kept_signals = {name: [signals[name][i] for i in keep] for name in required}
        bds = compute_rewards(
            [cands[i] for i in keep], [refs[i] for i in keep], [srcs[i] for i in keep],
            spec=self.reward_spec, judge_signals=kept_signals,
        )
        for p, bd in zip(kept, bds):
            self.baseline_rewards[str(p.get("id"))] = bd.total
        vals = list(self.baseline_rewards.values())
        logger.info(
            "SFT 基线锚: %d 条，奖励均值 %.4f（锚 slack=%.3f）",
            len(vals), sum(vals) / len(vals), self.anchor_slack,
        )

    @staticmethod
    def _fill_one_signal(scores: List[Optional[float]], group_size: int) -> Tuple[List[float], int]:
        """单路信号：按组用可用分均值补缺，返回 (补齐后的值, 缺失条数)。"""
        filled: List[float] = [0.0] * len(scores)
        missing = 0
        for start in range(0, len(scores), group_size):
            chunk = scores[start:start + group_size]
            ok = [s for s in chunk if s is not None]
            fallback = sum(ok) / len(ok) if ok else 0.0
            missing += sum(1 for s in chunk if s is None)
            for i, s in enumerate(chunk):
                filled[start + i] = float(s) if s is not None else fallback
        return filled, missing

    def fill_semantic_gaps(
        self,
        signals: Dict[str, List[Optional[float]]],
        group_size: int,
    ) -> Tuple[Dict[str, List[float]], float]:
        """多路信号各自补缺，返回 ({信号名: 补齐后的值}, 总缺失比例)。

        裁判个别失败时不中断训练，用**组内可用分的均值**补上。

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
        filled: Dict[str, List[float]] = {}
        missing = 0
        total = 0
        for name, scores in signals.items():
            values, miss = self._fill_one_signal(scores, group_size)
            filled[name] = values
            missing += miss
            total += len(scores)
        return filled, missing / max(total, 1)

    # ------------------------------------------------------------------
    def _reference_logprobs(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """参考策略的序列 logprob（已 detach）。返回 None = 没有参考策略。

        LoRA 快照那条是"共用底座、临时换旁路"：把 SFT 的 lora_A/B 就地
        `copy_` 进当前模型，跑一次 no_grad 前向，再恢复原值。所以不需要第二份
        6B 权重（省一份 12 GB 的显存），也不需要第二遍反传。

        **顺序很关键**：必须在策略 logprobs 之后调用。策略 logprobs 要带梯度、
        用当前（RL）权重；这里换的只是旁路权重，换进换出之间的梯度图不存在
        （no_grad）。
        """
        micro = self.logprob_micro_batch
        ids, labels = batch["input_ids"], batch["labels"]

        if self.reference_model is not None:
            with torch.no_grad():
                return sequence_logprob(
                    self.reference_model, ids, labels, micro_batch=micro
                ).detach()

        if self.reference_lora is not None:
            # 用 eval：参考分布应该是确定的（dropout 关掉）。这段没有反传，
            # 所以顺带关掉梯度检查点也不会让显存变差；结束后恢复原模式。
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad(), use_lora_weights(self.model, self.reference_lora):
                    ref = sequence_logprob(self.model, ids, labels, micro_batch=micro)
            finally:
                if was_training:
                    self.model.train()
            return ref.detach()

        if hasattr(self.model, "disable_adapter"):
            with torch.no_grad(), self.model.disable_adapter():
                return sequence_logprob(
                    self.model, ids, labels, micro_batch=micro
                ).detach()

        return None

    # ------------------------------------------------------------------
    def _inner_update(
        self,
        batch: Dict[str, Any],
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor],
        want_entropy: bool = False,
        prof: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """μ 循环的循环体：策略 logprob → GRPO loss(+KL) → 反传 → 裁剪 → 更新一步。

        **`old_logprobs` 与 `advantages` 整个循环里冻结**：前者是"采样那一刻的行为
        策略"π_θold 的对数概率，后者是这一组采样的相对好坏，两者都跟内层第几次
        迭代无关。于是：

            μ=1  → 循环体只跑一次，π_θ 与 π_θold 同参数 ⇒ ρ≡1 ⇒ clip 不生效；
            μ>1  → 第 2 次起 π_θ 已移动 ⇒ ρ≠1 ⇒ clip 才开始真正起作用。

        `want_entropy` 只在 μ 循环的第一次打开：完整词表的 fp32 概率很贵，
        而"这一步开始时策略有多随机"看第一次就够。
        """
        self.optimizer.zero_grad(set_to_none=True)

        with _timed(prof, "inner_forward"):
            # 监控用的策略熵在**同一次前向**里算（不额外跑模型）；只在低频步开启。
            entropy_values: Optional[List[float]] = None
            if want_entropy:
                logprobs, entropy_tensor = sequence_logprob(
                    self.model, batch["input_ids"], batch["labels"],
                    micro_batch=self.logprob_micro_batch, return_entropy=True,
                )
                entropy_values = [
                    float(x) for x in entropy_tensor.detach().reshape(-1).tolist()
                ]
            else:
                logprobs = sequence_logprob(
                    self.model, batch["input_ids"], batch["labels"],
                    micro_batch=self.logprob_micro_batch,
                )

        with _timed(prof, "inner_loss"):
            policy_loss, clipped_frac = grpo_loss(
                logprobs, old_logprobs, advantages, clip_ratio=self.clip_ratio
            )
            loss = policy_loss

            # KL 惩罚：参考 logprob 由调用方算好传进来（与策略参数无关，整步只算一次）。
            kl_value = 0.0
            if ref_logprobs is not None:
                kl_term = kl_penalty(logprobs, ref_logprobs, coef=self.kl_coef)
                loss = loss + kl_term
                kl_value = float(kl_term.detach())

            # 比值分布（μ>1 时它才是"漂移有多大"的真实读数：第 1 次恒为 1）
            ratios = [
                float(x)
                for x in torch.exp(logprobs.detach() - old_logprobs).reshape(-1).tolist()
            ]

        with _timed(prof, "inner_backward"):
            loss.backward()
            # 返回值是**裁剪前**的梯度总范数。它免费，而且是"训练稳不稳"最快的
            # 单一信号：突然窜到几十说明这一步的 advantage 尺度失控（常见于
            # 奖励被噪声污染、或被裁剪的 ratio 占了大头）。
            grad_norm_preclip = float(torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm
            ))
            self.optimizer.step()

        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "clipped_frac": float(clipped_frac),
            "grad_norm_preclip": grad_norm_preclip,
            "kl": kl_value,
            "ratios": ratios,
            "entropy": entropy_values,
        }

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
        if self.rollout_backend is not None:
            # vLLM 后端：先把当前策略的 LoRA 同步过去（热交换），再让它生成。
            # 这一步不占用训练模型的前向，所以不需要切 eval/train。
            self.rollout_backend.sync_policy(self.model, self.state.step)
            responses = self.rollout_backend.generate(
                messages_list,
                num_return_sequences=self.group_size,
                max_new_tokens=self.max_new_tokens, max_length=self.max_length,
                temperature=self.temperature, top_p=self.top_p,
            )
        else:
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
        # vLLM 后端如果带回了**原生 token id + logprob**，就直接用它们：
        #   1. 省掉"解码成文本再重新分词"的往返（两边 tokenization 不一致时会静默偏掉）；
        #   2. old_logprobs 直接用采样那一刻的 per-token logprob —— 省掉一次前向
        #      （N=8、L≈3300 时约 8s/步 ≈ 11%），而且它天然就是"行为策略"的概率。
        native_ids = getattr(self.rollout_backend, "last_token_ids", None)
        native_lp = getattr(self.rollout_backend, "last_logprobs", None)
        use_native_ids = (
            self.rollout_backend is not None
            and isinstance(native_ids, list)
            and len(native_ids) == len(responses)
        )
        use_vllm_lp = (
            use_native_ids
            and isinstance(native_lp, list)
            and len(native_lp) == len(responses)
        )
        if use_native_ids:
            responses = [
                decode(self.tokenizer, ids).strip() if ids else responses[k]
                for k, ids in enumerate(native_ids)
            ]
        sequences = []
        prompt_lens = []
        response_token_counts = []
        for i, (messages, _) in enumerate(zip(messages_list, prompts)):
            prompt_ids = encode_prompt(self.tokenizer, messages, add_generation_prompt=True)
            if len(prompt_ids) > self.max_length:
                prompt_ids = prompt_ids[-self.max_length:]
            for j in range(self.group_size):
                k = i * self.group_size + j
                if use_native_ids:
                    # 原生 token id：不再重新分词，也**不额外补 eos**（补了会让
                    # old_logprobs 的求和范围与 policy 前向的 labels 范围不一致）
                    gen_ids = list(native_ids[k])[: self.max_new_tokens]
                else:
                    text = responses[k]
                    gen_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                    gen_ids = gen_ids[: self.max_new_tokens] + [self.tokenizer.eos_token_id]
                # 监控用：记录**截断前**的生成 token 数，用于判断是否被 max_new_tokens 截断。
                response_token_counts.append(min(len(gen_ids), self.max_new_tokens))
                sequences.append(prompt_ids + gen_ids)
                prompt_lens.append(len(prompt_ids))

        width = max(len(s) for s in sequences)
        input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        labels = torch.full((len(sequences), width), -100, dtype=torch.long)
        for k, (seq, plen) in enumerate(zip(sequences, prompt_lens)):
            input_ids[k, : len(seq)] = torch.tensor(seq)
            labels[k, plen:len(seq)] = torch.tensor(seq[plen:])

        batch = {
            "input_ids": input_ids.to(self.device),
            "labels": labels.to(self.device),
            "responses": responses,
            "prompt_index": torch.arange(len(prompts)).repeat_interleave(self.group_size),
            "prompt_lens": prompt_lens,
            "response_token_counts": response_token_counts,
            "old_logprobs": None,
        }
        if use_vllm_lp:
            # Σ logp（只累加生成段）—— 与 labels 的非 -100 位置**一一对应**
            sums = [float(sum(lp[: len(native_ids[k])])) for k, lp in enumerate(native_lp)]
            batch["old_logprobs"] = torch.tensor(sums, dtype=torch.float32, device=self.device)
        return batch

    # ------------------------------------------------------------------
    def step(self, prompts: List[Dict[str, Any]]) -> Dict[str, float]:
        """一个优化步：rollout → 打分 → advantage → 反向。"""
        log_step = self.state.step + 1          # 实际 optimizer global_step
        want_entropy = self.monitor.entropy_due(log_step)
        want_lora = self.monitor.lora_due(log_step)
        # 分阶段计时（`SFZY_PROFILE=1`）。prof=None 时这些 with 都是空上下文，零开销。
        prof: Optional[Dict[str, float]] = {} if _PROFILE_ENABLED else None
        self._prof = prof
        if prof is not None and torch.cuda.is_available():
            # 每步重置峰值统计：这样 `_log_profile` 里的"本步峰值"就是**这一步**的
            # 真实峰值（训练前向/反向通常最高），而不是"进程启动至今"的累计值。
            torch.cuda.reset_peak_memory_stats()
        step_start = time.perf_counter()
        with _timed(prof, "rollout"):
            batch = self.rollout(prompts)

        # ---- 打分 ----
        refs, sources = [], []
        for i in batch["prompt_index"].tolist():
            refs.append(prompts[i]["summary"])
            sources.append(prompts[i]["source"])
        # `signals_raw` 是裁判的**原始**输出（缺失处为 None），只喂给监控；
        # `signals` 是补缺后的，喂给奖励（与改动前完全一致）。
        with _timed(prof, "judge"):
            signals_raw = self.score_semantic(batch["responses"], refs, sources)
        judge_results: List[Any] = []
        if signals_raw is not None:
            get_results = getattr(self.scorer, "aligned_results", None)
            if callable(get_results):
                judge_results = list(get_results())
        signals = signals_raw
        judge_missing = 0.0
        if signals is not None:
            signals, judge_missing = self.fill_semantic_gaps(signals, self.group_size)
        breakdowns = compute_rewards(
            batch["responses"], refs, sources,
            spec=self.reward_spec, judge_signals=signals,
        )
        # 门控前奖励 / 分项得分（仅监控；不改变上面 `breakdowns` 的任何数值）
        pre_gate = compute_pre_gate_rewards(
            batch["responses"], refs, sources,
            spec=self.reward_spec, judge_signals=signals,
        )
        pre_gate_rewards = [float(total) for total, _ in pre_gate]
        term_values: Dict[str, List[float]] = {
            term.name: [float(values.get(term.name, 0.0)) for _, values in pre_gate]
            for term in self.reward_spec.active_terms
        }
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

        # ---- 参考策略的 logprob：只与 batch 有关，与策略参数无关 ----
        # 所以整个 μ 循环只算一次（μ=4 时省下 3 次参考前向）。
        with _timed(prof, "reference_logprobs"):
            ref_logprobs = self._reference_logprobs(batch) if self.kl_coef > 0 else None

        # ---- old_logprobs：必须在**任何**参数更新之前算，μ 次内层更新共用 ----
        # 它就是"采样那一刻的行为策略" π_θold，在整个内层循环里是常数 —— 也正因为
        # 如此，μ=1 时 ρ≡1（clip 不生效），μ>1 时第 2 次起 ρ≠1 才开始生效。
        with _timed(prof, "old_logprobs"):
            if batch.get("old_logprobs") is not None:
                # 后端（vLLM）已经给了采样那一刻的 per-token logprob：直接用，
                # 不再跑这一趟前向（省 ~11% 步时，而且更严格）。这一段计时会接近 0。
                old_logprobs = batch["old_logprobs"]
            else:
                with torch.no_grad():
                    old_logprobs = sequence_logprob(
                        self.model, batch["input_ids"], batch["labels"],
                        micro_batch=self.logprob_micro_batch,
                    ).detach()

        # 监控用：LoRA 更新幅度（低频）。在循环之前快照、循环之后比差 ——
        # 量到的是整步漂移（μ 次更新累积），这才是"这一步把策略推了多远"。
        lora_before: Optional[Dict[str, torch.Tensor]] = None
        if want_lora:
            snapshot = {
                name: param.detach().clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad and "lora_" in name
            }
            # 没有 LoRA 参数时保持 None（不适用），而不是记一个 0。
            lora_before = snapshot or None

        # ---- μ 次内层更新（μ=1 时与改动前逐步等价）----
        iters = [
            self._inner_update(
                batch, old_logprobs, advantages, ref_logprobs,
                want_entropy=want_entropy and it == 0, prof=prof,
            )
            for it in range(self.inner_iterations)
        ]
        losses = [x["loss"] for x in iters]
        policy_losses = [x["policy_loss"] for x in iters]
        clipped = [x["clipped_frac"] for x in iters]
        grad_norms = [x["grad_norm_preclip"] for x in iters]
        kl_values = [x["kl"] for x in iters]
        # 比值把 μ 个内层迭代全留着：μ>1 时这才是真实的漂移分布（第 1 次恒为 1、
        # 越往后越偏），μ=1 时与改动前完全一致。
        ratio_values = [r for x in iters for r in x["ratios"]]
        entropy_values = next(
            (x["entropy"] for x in iters if x["entropy"] is not None), None
        )

        # 聚合口径（写清楚，避免"曲线好看但口径说不清"）：
        #   loss / policy_loss / clipped_frac / kl → μ 次的**均值**
        #   grad_norm → μ 次的**最大值**（稳定性看最坏的那一次）
        #   clip_applied → 只要有一次被裁到就是 1
        loss = float(sum(losses) / len(losses))
        policy_loss_value = float(sum(policy_losses) / len(policy_losses))
        clipped_frac = float(sum(clipped) / len(clipped))
        grad_norm = float(max(grad_norms))
        grad_norm_preclip = grad_norm
        grad_norm_postclip = min(grad_norm_preclip, float(self.max_grad_norm))
        clip_applied = 1.0 if grad_norm_preclip > float(self.max_grad_norm) else 0.0
        kl_value = float(sum(kl_values) / len(kl_values))
        # k3 无偏估计 = 加权惩罚 / 系数；两者分开记录，避免混淆。
        kl_estimate_value = kl_value / self.kl_coef if ref_logprobs is not None else None
        kl_penalty_value = kl_value if ref_logprobs is not None else None

        lora_update_norm: Optional[float] = None
        if lora_before is not None:
            sq = 0.0
            for name, param in self.model.named_parameters():
                if name in lora_before:
                    delta = (param.detach() - lora_before[name]).float()
                    sq += float((delta * delta).sum())
            lora_update_norm = float(sq ** 0.5)

        gate_stats = summarize_gate_reasons(breakdowns)
        metrics = {
            "loss": loss,
            # μ>1 时同一批数据更新多次：loss 会逐步下降，最后一轮的数值更能
            # 说明"这一步结束时策略拟合到什么程度"；两个都留着，口径不混。
            "loss_last": float(losses[-1]),
            "grad_norm": float(grad_norm),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "advantage_abs_mean": float(advantages.abs().mean()),
            "clipped_frac": float(clipped_frac),
            "clipped_frac_first": float(clipped[0]),
            "clipped_frac_last": float(clipped[-1]),
            "inner_iterations": int(self.inner_iterations),
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
        # ---- 逐项分项：每个启用的 reward 都要有均值 + 组内标准差 ----
        # 只盯总分判断不出"好在哪"；组内标准差才是"这一项在 GRPO 里有没有用"
        # 的判据 —— 均值好看但组内没方差，归一化之后就是常数，等于白接。
        for term in self.reward_spec.enabled_terms:
            values = torch.tensor(
                [b.values.get(term.name, 0.0) for b in breakdowns],
                dtype=torch.float32, device=self.device,
            )
            metrics[f"term_{term.name}"] = float(values.mean())
            metrics[f"term_{term.name}_group_std"] = float(
                values.reshape(len(prompts), self.group_size)
                .std(dim=-1, unbiased=False).mean()
            )
        # 六要素裁判自己汇报的统计量（需求第十二节）：mean_fact_reward、
        # 各要素均值、0-4 各档比例。只有事实一致性后端有这个钩子，
        # **不需要 trainer 认识六要素**，耦合面就一个方法名。
        if hasattr(self.scorer, "summarize_last"):
            metrics.update(self.scorer.summarize_last())

        # ---- 监控层：把本步已产生的数据摊平成 SwanLab 指标 ----
        gate_cfg = self.reward_spec.gate.cfg if self.reward_spec.gate.enabled else {}
        upper_ratio = None
        if self.reward_spec.gate.enabled:
            rng = gate_cfg.get("length_ratio_range", [0.5, 1.5])
            if rng and len(rng) >= 2:
                upper_ratio = float(rng[1])
        flat_lengths = lengths.reshape(-1).clamp(min=1)
        if self.length_mode == "none":
            norm_factors = [1.0] * flat_lengths.numel()
        elif self.length_mode == "mean":
            norm_factors = [float(x) for x in flat_lengths.tolist()]
        else:
            norm_factors = [float(math.sqrt(x)) for x in flat_lengths.tolist()]
        raw_adv = group_advantages(rewards).reshape(-1)
        monitor_batch = MonitorBatch(
            step=log_step,
            group_size=self.group_size,
            responses=list(batch["responses"]),
            references=list(refs),
            final_rewards=[float(b.total) for b in breakdowns],
            pre_gate_rewards=pre_gate_rewards,
            gate_flags=[bool(b.gated) for b in breakdowns],
            gate_reasons=[str(b.gate_reason or "") for b in breakdowns],
            term_values=term_values,
            term_weights=self.reward_spec.normalized_weights(),
            raw_signals=dict(signals_raw or {}),
            judge_results=judge_results,
            advantages_raw=[float(x) for x in raw_adv.tolist()],
            advantages_effective=[float(x) for x in advantages.reshape(-1).tolist()],
            kept_mask=[bool(x) for x in keep.tolist()],
            output_lengths=[int(x) for x in lengths.reshape(-1).tolist()],
            token_lengths=list(batch["response_token_counts"]),
            ratios=ratio_values,
            normalization_factors=norm_factors,
            length_norm_mode=self.length_mode,
            clip_frac=float(clipped_frac),
            policy_loss=policy_loss_value,
            total_loss=loss,
            grad_norm_preclip=grad_norm_preclip,
            grad_norm_postclip=grad_norm_postclip,
            clip_applied=clip_applied,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            clip_ratio=float(self.clip_ratio),
            fact_gate_enabled=self.reward_spec.fact_error_gate_enabled,
            upper_length_ratio=upper_ratio,
            result_markers=tuple(gate_cfg.get("require_result_marker") and RESULT_MARKERS or ()),
            forbidden_prefixes=tuple(gate_cfg.get("forbidden_prefixes", [])),
            max_new_tokens=self.max_new_tokens,
            entropy=entropy_values,
            lora_update_norm=lora_update_norm,
            kl_estimate=kl_estimate_value,
            kl_penalty=kl_penalty_value,
        )
        self._last_monitor_metrics = self.monitor.observe(monitor_batch)
        metrics.update(self._last_monitor_metrics)
        if prof is not None:
            self._log_profile(log_step, prof, batch, time.perf_counter() - step_start)
        return metrics

    # ------------------------------------------------------------------
    def train(self, prompts: List[Dict[str, Any]]) -> RLState:
        """主循环。每步处理 prompts_per_step 个 prompt。"""
        if self.is_main_process:
            # 基线锚只在前 N 条 prompt 上用得上，但 prepare_baseline 会整池算 ——
            # 便宜（一次前向），换来的是每步都能查表。
            self.prepare_baseline(prompts)
        n_steps = len(prompts) // self.prompts_per_step
        self.monitor.set_total_steps(max(1, n_steps))
        logger.info(
            "GRPO 开始: %d 个 prompt，每步 %d 个，共 %d 步，G=%d，μ=%d（共 %d 次梯度更新）",
            len(prompts), self.prompts_per_step, n_steps, self.group_size,
            self.inner_iterations, n_steps * self.inner_iterations,
        )
        # 验证集调度：x 个分位点 = "把 n 步等分成 x 段"，最后一点落在最后一步，
        # 保证训练结束时的模型一定会被评到（也一定有机会成为 best.pt）。
        if self.monitor.cfg.eval_enabled and self.monitor.cfg.eval_data_file:
            schedule = self.monitor.eval_schedule()
            key = self._best_metric_key()
            if schedule:
                preview = ", ".join(str(s) for s in schedule[:10])
                if len(schedule) > 10:
                    preview += " …"
                note = ""
                if self.monitor.cfg.eval_quantile_points >= n_steps:
                    note = (
                        f"（x={self.monitor.cfg.eval_quantile_points} ≥ 步数 {n_steps}，"
                        "已退化为只在最后一步验证）"
                    )
                logger.info(
                    "验证集：x=%d 个分位点 → 共 %d 次，落在第 %s 步%s；best.pt 由 %s 决定",
                    self.monitor.cfg.eval_quantile_points, len(schedule), preview,
                    note,
                    key or "（save_best=false，不存 best）",
                )
                # 验证成本要**开跑前**就可见：它是串行贪心生成，量级 = 次数 × 条数 × token
                logger.info(
                    "  验证成本 ≈ %d 次 × %d 条 × 最多 %d token（ChatGLM3 逐条串行生成，"
                    "按 30ms/token 估约 %.1f 小时）；嫌贵就降 quantile_points / max_samples / max_new_tokens",
                    len(schedule), self.monitor.cfg.eval_max_samples,
                    min(self.monitor.cfg.eval_max_new_tokens, self.max_new_tokens),
                    len(schedule) * self.monitor.cfg.eval_max_samples
                    * min(self.monitor.cfg.eval_max_new_tokens, self.max_new_tokens)
                    * 0.03 / 3600,
                )
            else:
                logger.info(
                    "验证集：每 %d 步一次（未设 quantile_points）；best.pt 由 %s 决定",
                    self.monitor.cfg.eval_every_n_steps,
                    key or "（save_best=false，不存 best）",
                )
        for step in range(self.state.step, n_steps):
            chunk = prompts[step * self.prompts_per_step:(step + 1) * self.prompts_per_step]
            metrics = self.step(chunk)
            self.state.step = step + 1

            if self.is_main_process and self.monitor.eval_due(self.state.step):
                eval_result = self._run_eval(self.state.step)
                if eval_result:
                    # 先决定要不要更新 best.pt，再并入上报指标（顺序无关，但这里
                    # 写在前面，日志里"刷新最好成绩"会紧跟在那行验证日志之后）。
                    self._maybe_save_best(eval_result, self.state.step)
                    self._last_monitor_metrics.update(eval_result)

            if self.is_main_process and (step + 1) % self.log_every_n_steps == 0:
                self.state.history.append({"step": self.state.step, **metrics})
                self._log_step(self.state.step, n_steps, metrics)
                if self.tracker is not None:
                    # 只上报监控层产出的**完整分组指标**（`模块/指标`），
                    # 避免与 history 里的扁平键重复成两条曲线。
                    self.tracker.log(
                        group_metrics(dict(self._last_monitor_metrics)), self.state.step
                    )

            if self.is_main_process and self.state.step % self.save_every_n_steps == 0:
                self.save()

        if self.is_main_process:
            self.save()
        # 收尾：vLLM 引擎的显存不回收会一直占着（下次同卡启动会 OOM）
        if self.rollout_backend is not None:
            self.rollout_backend.close()
        return self.state

    # ------------------------------------------------------------------
    def _log_profile(
        self,
        step: int,
        prof: Dict[str, float],
        batch: Dict[str, Any],
        total: float,
    ) -> None:
        """`SFZY_PROFILE=1` 时打一张分阶段耗时表（附本步的真实形状）。

        诊断"这一步为什么这么慢"靠的就是这三样：**各阶段秒数**、**序列宽度**
        （= prompt 截断后的长度 + 本组最长生成）与**生成长度**。例如：

            总耗时几乎全在 `rollout`  → 生成太长/太慢（ChatGLM3 是逐条串行生成）
            `judge` 占大头              → 裁判 API 慢（输入是整篇原文、并发被限）
            `inner_*` 占大头            → 序列太宽（max_length 开太大）或 μ 太大
        """
        gen = list(batch.get("response_token_counts") or [])
        gen_median = sorted(gen)[len(gen) // 2] if gen else 0
        vram = ""
        if torch.cuda.is_available():
            # 本步峰值（step 开头 reset 过）+ 整卡当前用量（含 vLLM 引擎子进程）
            vram = f"，本步本进程峰值 {torch.cuda.max_memory_allocated() / 1024**3:.1f}G"
            try:
                # ★ 变量名不能叫 total：它是本函数的入参（本步秒数），被覆盖后
                #   日志里的"总 X s"会打印成显存字节数（实测踩过：打印 50864390144.0s）。
                free_b, total_b = torch.cuda.mem_get_info()
                vram += (
                    f"，整卡 {((total_b - free_b) / 1024**3):.1f}"
                    f"/{total_b / 1024**3:.1f}G 已用"
                )
            except Exception:  # noqa: BLE001 - 只是日志
                pass
        ranked = " | ".join(f"{k} {v:.1f}s" for k, v in sorted(prof.items(), key=lambda x: -x[1]))
        logger.info(
            "[profile] step %d 总 %.1fs（N=%d，序列宽度 %d，生成长度 中位 %d/上限 %d%s）"
            "\n    %s",
            step, total, int(batch["input_ids"].shape[0]), int(batch["input_ids"].shape[1]),
            gen_median, self.max_new_tokens, vram, ranked,
        )

    # ------------------------------------------------------------------
    def _best_metric_key(self) -> Optional[str]:
        """`best.pt` 的判据指标（完整上报键）；`save_best: false` 时返回 None。

        默认 `rouge_total` —— 就是 SFT 阶段那套官方口径综合分
        `0.2*ROUGE-1 + 0.4*ROUGE-2 + 0.4*ROUGE-L`（见 `sfzy.eval.rouge.OFFICIAL_WEIGHTS`
        与它的 `overall` 键）。
        """
        spec = self.cfg.path_("rl.monitor.eval") or {}
        if not bool(spec.get("save_best", True)):
            return None
        metric = str(spec.get("best_metric", "rouge_total"))
        return metric if "/" in metric else f"eval/{metric}"

    def _maybe_save_best(self, eval_result: Dict[str, float], step: int) -> None:
        """验证分刷新历史最好成绩 → 覆盖 `best.pt`。

        "历史最好"记在 `self.state` 里并随 checkpoint 落盘，所以续训时不会用一个
        更差的分数把 best.pt 覆盖掉（这点静默发生的话很难发现）。

        比较口径只有一条：**越大越好**，阈值/多目标都不藏在这里。
        """
        if not eval_result:
            return
        key = self._best_metric_key()
        if key is None:
            return
        value = eval_result.get(key)
        if value is None:
            logger.warning(
                "验证集没有产出 %s，本次不更新 best.pt（本次可用的：%s）",
                key, sorted(k for k in eval_result if k.startswith("eval/")),
            )
            return

        best = self.state.best_eval_score
        if best is not None and value <= best:
            logger.info(
                "验证集 %s=%.4f 未刷新最好成绩（%.4f @step %s），best.pt 不动",
                key, value, best, self.state.best_step,
            )
            return

        self.state.best_eval_score = float(value)
        self.state.best_step = int(step)
        self.save(tag="best")
        logger.info(
            "★ 验证集 %s=%.4f 刷新最好成绩（旧值 %s）→ 已写 best.pt（step %d）",
            key, value, "无" if best is None else f"{best:.4f}", step,
        )

    # ------------------------------------------------------------------
    def _log_step(self, step: int, n_steps: int, m: Dict[str, float]) -> None:
        """打这一步的日志。启用了哪些 reward，就打印哪几列。"""
        parts = [
            f"step {step}/{n_steps}",
            f"reward {m['reward_mean']:.4f}±{m['reward_std']:.4f}"
            f"(组内σ{m['reward_group_std']:.4f})",
        ]
        for term in self.reward_spec.enabled_terms:
            parts.append(
                f"{term.name} {m[f'term_{term.name}']:.4f}"
                f"(σ{m[f'term_{term.name}_group_std']:.4f})"
            )
        if self.reward_spec.needs_judge():
            parts.append(f"裁判缺失 {m['judge_missing'] * 100:.1f}%")
        parts += [
            f"门控 {m['gating_rate'] * 100:.1f}%",
            f"保留组 {m['kept_groups']}/{m['total_groups']}",
            f"clip {m['clipped_frac'] * 100:.1f}%",
            f"|g| {m['grad_norm']:.2f}",
            f"len {m['output_len_mean']:.0f}",
        ]
        mu = int(m.get("inner_iterations", 1))
        if mu > 1:
            # 内层漂移的读数：clip 从第 1 次迭代（ρ≡1，必然 0%）到第 μ 次的变化。
            # 最后一轮就已经几十个百分点，说明 μ 开大了（策略在一步内走太远）。
            parts.append(
                f"μ{mu} clip {m['clipped_frac_first'] * 100:.0f}%→"
                f"{m['clipped_frac_last'] * 100:.0f}%"
            )
        logger.info(" | ".join(parts))

        # 事实一致性的整体分布（需求第十二节）：均值 + 0-4 各档比例 + 错误率。
        if "mean_fact_reward" in m:
            ratio = " ".join(
                f"{lv}:{m.get(f'ratio_score_{lv}', 0.0) * 100:.0f}%" for lv in range(5)
            )
            logger.info(
                "  └ 事实一致性 fact_reward %.4f | 整体分 %.3f | 事实错误率 %.1f%% | 档位 %s",
                m["mean_fact_reward"],
                m.get("mean_fact_score", 0.0),
                100.0 * m.get("fact_error_rate", 0.0),
                ratio,
            )

        # 裁判整组失败时光看比例没法排查 —— 把最后一条异常打出来。
        # 实测最常见的两种：裁判 API 超时/报错，或者裁判侧那一批前向 OOM。
        errors = getattr(self.scorer, "last_errors", None)
        if m.get("judge_missing", 0.0) > 0 and errors:
            logger.warning(
                "  裁判有 %.0f%% 的组失败，最近一条：%s",
                m["judge_missing"] * 100, str(errors[-1])[:300],
            )

    # ------------------------------------------------------------------
    def _load_eval_records(self) -> List[Dict[str, Any]]:
        """读固定验证集（只读一次，之后缓存在内存）。"""
        cached = getattr(self, "_eval_records", None)
        if cached is not None:
            return cached
        spec = self.monitor.cfg.eval_data_file
        path = Path(spec)
        if not path.is_absolute():
            root = Path(__file__).resolve().parents[3]
            candidate = root / path
            path = candidate if candidate.exists() else path
        records: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        records = records[: self.monitor.cfg.eval_max_samples]
        self._eval_records = records
        return records

    def _run_eval(self, step: int) -> Dict[str, float]:
        """固定验证集评估（可选；默认关闭，不影响训练）。

        - 固定数据、固定贪心解码、固定分词口径 → 各 checkpoint 可比；
        - 复用已有的 `generate_batch` / `score_corpus` / `compute_rewards`；
        - 生成阶段临时切 eval，结束后恢复原模式，不写梯度。
        """
        try:
            records = self._load_eval_records()
        except (OSError, ValueError) as exc:
            logger.warning("验证集读取失败，跳过本次 eval：%s", exc)
            return {}
        if not records:
            return {}
        cf = self.monitor.cfg
        prompt_style = self.cfg.path_("rl.prompt_style", "structured")
        messages_list = [build_messages(r["source"], prompt_style) for r in records]
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                responses = generate_batch(
                    self.model, self.tokenizer, messages_list,
                    # 不超过训练生成长度，保证验证与训练同一解码预算。
                    max_new_tokens=min(cf.eval_max_new_tokens, self.max_new_tokens),
                    # 截断规则必须与训练一致：没显式配 max_length 时就用 rl.max_length，
                    # 配了则取 min(它, rl.max_length) —— 绝不允许验证比训练更宽松，
                    # 也不该反过来（"训练看 7168 token、验证只喂 2048"会把验证分
                    # 系统性压低，best 的选择就失去意义）。
                    max_length=(
                        self.max_length if cf.eval_max_length is None
                        else min(int(cf.eval_max_length), self.max_length)
                    ),
                    do_sample=False,
                    num_return_sequences=1,
                )
        finally:
            if was_training:
                self.model.train()
        refs = [r["summary"] for r in records]
        sources = [r["source"] for r in records]
        preds = {str(r["id"]): resp for r, resp in zip(records, responses)}
        ref_map = {str(r["id"]): r["summary"] for r in records}
        try:
            rouge = score_corpus(preds, ref_map, mode=self.reward_spec.rouge_mode)
        except Exception as exc:  # noqa: BLE001 — 缺 id 等不该毁掉训练
            logger.warning("验证集 ROUGE 计算失败，跳过：%s", exc)
            rouge = None
        signals_raw = self.score_semantic(responses, refs, sources)
        if signals_raw is not None:
            filled, _ = self.fill_semantic_gaps(signals_raw, group_size=1)
        else:
            filled = None
        breakdowns = compute_rewards(
            responses, refs, sources, spec=self.reward_spec, judge_signals=filled
        )
        length_ratios = [
            len(c.strip()) / max(len(r.strip()), 1) for c, r in zip(responses, refs)
        ]
        metrics = eval_metrics(
            rouge, length_ratios, dict(signals_raw or {}),
            [float(b.total) for b in breakdowns], cf.eval_baseline,
        )
        if metrics:
            logger.info("验证集 step %d: %s", step, {
                k.split("/")[1]: round(v, 4) for k, v in metrics.items()
                if k.split("/")[1] in ("rouge_l", "coverage_mean", "ee_mean", "fact_mean")
            })
        return metrics

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
