"""SFT 训练循环（手写）。

============================ 你要实现的文件 ============================

不用 Trainer 类的理由：训练循环是这个项目最该自己写的地方。
学习率、梯度累积、混合精度、梯度裁剪、断点续训——每一样面试都问得到，
调包就答不上来。

-------------------------------- fp16 的五个顺序问题（最容易全写错） --------------------------------
1. **loss 要除以梯度累积步数**再 backward。
   不除的话等效学习率会放大 accum_steps 倍，loss 曲线会抖得没法看。
2. **scaler.scale(loss).backward()**，不是直接 loss.backward()。
   fp16 下梯度会下溢成 0，必须靠 GradScaler 放大再缩回。
3. 梯度裁剪**之前**必须 scaler.unscale_(optimizer)，
   否则裁的是被放大过的梯度，裁剪阈值形同虚设。
4. 只有到了累积边界才 scaler.step(optimizer); scaler.update()，
   中间步骤不要动 optimizer。
5. **每步都要 optimizer.zero_grad(set_to_none=True)**，且要放在
   backward 之前——放在之后的写法在遇到梯度累积时很容易写错位置。

-------------------------------- 另外两个 Kaggle 相关的坑 --------------------------------
* **梯度检查点与 use_cache 冲突**：开了 gradient_checkpointing 就必须
  model.config.use_cache = False，否则会报 "gradient checkpointing
  和 cache 不能同时启用"。推理前再打开。
* **单次会话 9 小时**：save_every_n_steps 必须配合 checkpoint.py 的
  断点续训，否则跑到一半被掐就白干了。
=====================================================================
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

from sfzy.config import Config
from sfzy.sft.checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint
from sfzy.sft.lr_scheduler import get_lr
from sfzy.utils.distributed import get_world_size, is_distributed
from sfzy.utils.logging import get_logger

logger = get_logger("trainer")

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class TrainState:
    """训练进度。断点续训要能精确恢复到"第几步、第几个 epoch"。"""

    step: int = 0 # 全局已完成的优化步数。 不是 epoch 内的索引，也不是 batch 数。
    epoch: int = 0 # 已经完整跑完的 epoch 数
    best_dev_loss: float = float("inf") # 历史最优的验证 loss，用于挑最好的 checkpoint 存下来。
    history: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "best_dev_loss": self.best_dev_loss,
            "history": self.history,
        }


class SFTTrainer:
    """手写训练循环。"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        collator: Any,
        cfg: Config,
        output_dir: Optional[str] = None,
        tracker: Any = None,
        is_main_process: bool = True,
    ) -> None:
        """============================ 要写的步骤 ============================

        1. 读超参：从 cfg.sft 取 learning_rate / weight_decay / dtype /
           grad_accum_steps / num_workers / pin_memory / ...
        2. 收集可训练参数，并**分成两组**：需要权重衰减的、不需要的。
           判断依据是参数名里有没有 "bias" 或 "LayerNorm"。
        3. 构造 AdamW，把两组分别传进去。
        4. 构造 GradScaler（只有 fp16 才需要启用，bf16 不需要）。
        5. 保存 collator / tokenizer / output_dir / tracker / is_main_process
           这些后面要用的引用。
        6. 处理 gradient_checkpointing 与 use_cache 的冲突。
        7. 初始化 self.state = TrainState()。

        ============================ 必要知识 ============================

        **为什么优化器只能拿到 requires_grad=True 的参数。**
        把冻结参数传进去不会立刻报错，但 optimizer.step() 会去访问它们的
        grad（为 None），或者白白占显存。LoRA 场景下底座是冻结的，
        只有 A/B 矩阵该进优化器。如果你忘了 mark_only_lora_trainable，
        这里会传进去一堆冻结参数，或者一个可训练参数都没有而直接报
        "optimizer got an empty parameter list"。

        **为什么要分权重衰减组。**
        weight_decay 的作用是抑制权重变大，而 bias 和 LayerNorm 的
        缩放参数（gamma）本身就不该被抑制——把 LayerNorm 的 gamma
        往 0 拉会直接破坏归一化。LoRA 场景下所有可训练参数都是 A/B 矩阵，
        名称里不含 bias/LayerNorm，所以第二组通常是空的，但写法要保留——
        将来做全量微调时它会立刻派上用场。

        **AdamW 里的 W 是什么意思。**
        AdamW = Adam + 解耦权重衰减（decoupled weight decay）。
        原版 Adam 的 L2 正则混在梯度里，和自适应学习率耦合在一起，
        导致大权重的有效衰减被削弱。AdamW 把衰减项独立出来，
        这也是它成为 LLM 训练默认选择的原因。

        **GradScaler 不是混合精度本身。**
        autocast 负责"哪些算子用 fp16 算"，GradScaler 负责"梯度的动态缩放"。
        fp16 的表示范围很窄（最小正规数约 6e-5），反向传播时梯度很容易
        下溢成 0。GradScaler 先把 loss 放大若干倍再 backward，
        让梯度落在可表示范围内，step 前再缩回来。
        **bf16 的动态范围和 fp32 一样宽，不会下溢，所以不需要 scaler。**
        这就是为什么 enabled 要跟 dtype 挂钩，而不是无脑开。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.collator = collator
        self.cfg = cfg
        self.output_dir = Path(output_dir or cfg.path_("sft.output_dir", "outputs/sft"))
        self.tracker = tracker
        self.is_main_process = is_main_process
        self.trainable_param = []
        self.seed = cfg.get("seed", 42)
        
        sft_cfg = cfg.sft

        # ---- 混合精度：dtype 决定要不要用 scaler ----
        # 兼容旧的 fp16: true 写法，新的写法是 sft.dtype: float16|bfloat16
        if sft_cfg.get("dtype") is not None:
            self.dtype = DTYPE_MAP[str(sft_cfg.get("dtype")).lower()]
        else:
            self.dtype = torch.float16 if sft_cfg.get("fp16") else torch.float32
        self.use_scaler = self.dtype == torch.float16

        # ---- 优化器：按是否做权重衰减分组 ----
        no_decay = ("bias", "LayerNorm.weight", "layernorm.weight")
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay_params if any(nd in name for nd in no_decay) else decay_params).append(param)
            self.trainable_param.append(param)
        groups = [{"params": decay_params, "weight_decay": sft_cfg.get("weight_decay", 0.0)}]
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(groups, lr=sft_cfg.get("learning_rate", 1e-4))
        self.lr = sft_cfg.get("learning_rate", 1e-4)

        # 设备必须先确定：scaler 和 autocast 的构造都依赖它
        self.device = next(model.parameters()).device

        # GradScaler 只在 fp16 下启用；bf16 与 fp32 都不需要。
        # 用新的 torch.amp.GradScaler(device_type, ...)：旧的
        # torch.cuda.amp.GradScaler 在 torch 2.14 上会打 FutureWarning，
        # 而且它硬编码了 cuda，CPU 上语义不对。
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_scaler)

        # ---- 梯度检查点与 use_cache 互斥 ----
        if sft_cfg.get("gradient_checkpointing") and hasattr(model, "config"):
            model.config.use_cache = False

        # autocast 规则：目标精度是 fp32 时套上去没有意义（还会触发
        # "target dtype 已是默认值" 的警告），直接退化成 nullcontext。
        # 上下文对象可以重复进入，所以建一次、每步复用。
        if self.dtype == torch.float32:
            self.autocast_ctx = nullcontext()
        else:
            self.autocast_ctx = torch.autocast(device_type=self.device.type, dtype=self.dtype)
    
        self.state = TrainState()
        
        self.batch_size = sft_cfg.get("per_device_batch_size", 1)
        self.grad_accum_steps = sft_cfg.get("grad_accum_steps", 1)
        self.num_epochs = sft_cfg.get("num_epochs", 3)
        self.num_workers = sft_cfg.get("num_workers", 0)
        self.pin_memory = sft_cfg.get("pin_memory", False)
        self.max_grad_norm = sft_cfg.get("max_grad_norm", 1.0)
        self.log_every_n_steps = sft_cfg.get("log_every_n_steps", 10)
        self.save_every_n_steps = sft_cfg.get("save_every_n_steps", 200)
        self.keep_last_n_checkpoints = sft_cfg.get("keep_last_n_checkpoints", 3)
        self.warmup_ratio = sft_cfg.get("warmup_ratio", 0.0)
        self.min_lr_ratio = sft_cfg.get("min_lr_ratio", 0.0)

        # 按「墙钟时间」存档，和 save_every_n_steps 是**或**的关系。
        #
        # 为什么步数之外还要有这一条：Kaggle 会话到点是被直接 kill 的，
        # 没有"训练结束再存一次"的机会；而"9 小时能跑到第几步"取决于
        # 每步耗时，每步耗时又取决于这个 batch 里判决书的长度
        # （短的 300 字，长的 14000 字，p99 有 6700 字）。
        # 用时间触发就不用预先算这个数，也不会因为漏算而丢掉最后两小时。
        # 设为 null / 0 表示关闭，只按步数存档。
        self.save_every_n_minutes = sft_cfg.get("save_every_n_minutes") or None

        # ---- 验证的节奏与成本 ----
        #
        # eval_every_n_steps：每隔多少个**优化步**验一次。
        #   默认 null = 只在每个 epoch 末尾验一次（原来就有的行为）。
        #
        #   为什么 Kaggle 上必须按步数触发：一轮 10738 条要跑十几个小时，
        #   而单次会话只有 9 小时 —— "epoch 末尾验一次"等于**这个会话里永远
        #   不会发生**，best.pt 也就永远存不下。
        #
        #   代价可以自己算：一次验证 ≈ 样本数 × 每样本秒数，
        #   而每个优化步 ≈ grad_accum_steps × 每样本秒数。
        #   所以 eval_max_samples=200、log 间隔 15 步时，
        #   一次验证约等于 13 个优化步的时间（约 6%）。
        self.eval_every_n_steps = sft_cfg.get("eval_every_n_steps") or None

        # eval_max_samples：验证时**最多**用多少条，null = 全量。
        #   验证集 1340 条全跑一遍要十几分钟，而我们要的只是"train 还在降、
        #   dev 有没有开始涨"这个趋势 —— 200 条足够，且每次成本固定。
        #   取样是**等间隔**取的，不是取前 N 条。实测本项目的切分文件
        #   （tools/make_splits.py 产出）已经是打散的：前 200 条的案由分布
        #   和全量一致（借款合同 51/200 vs 全量 23.9%），所以 head 切片在这里
        #   不会明显偏。之所以还是用等间隔：它不依赖"文件恰好被打散"这个前提，
        #   换切分工具或换数据集时不会突然变成一个安静的偏差。
        self.eval_max_samples = sft_cfg.get("eval_max_samples") or None

        # 上一次验证落在了第几步、结果是多少。
        # 周期验证（step % eval_every_n_steps）和 epoch 末尾那次**可能落在同一步**：
        # 不去重的话会在同一个 step 上把同一批样本验两遍 —— 结果一样、成本翻倍，
        # 而且 history 里会出现两条一模一样的记录，画曲线时看着像"验了两次"。
        self._last_eval_step: Optional[int] = None
        self._last_dev_loss: Optional[float] = None

        # 只用于日志的计时与计数（吞吐、峰值显存），不参与训练逻辑
        self._run_start_time = time.monotonic()
        self._last_save_time = self._run_start_time
        self._samples_this_run = 0

        # 断点续训的起点。None 表示从头训练；字符串表示具体的 checkpoint 路径。
        # 刻意不支持"自动找最新"—— 见 resume() 的说明。
        self.resume_from: Optional[str] = sft_cfg.get("resume_from") or None

        # DDP 下每条 rank 只处理 1/world_size 的数据，步数计算要用到
        self.world_size = get_world_size()

    # ------------------------------------------------------------------
    def train(self, train_dataset: Any, dev_dataset: Optional[Any] = None) -> TrainState:
        """主循环。

        ============================ 要写的步骤 ============================

        1. 根据 len(train_dataset) / per_device_batch_size 算出每个 epoch 的
           步数，再乘 epoch 数得到 total_steps——学习率调度要用。
        2. 构造 DataLoader：batch_size、collate_fn=self.collator、
           shuffle=True、num_workers、pin_memory。
        3. 从 self.state 读 start_epoch 和 start_step（初始都是 0）。
        4. 外层 for epoch：
             a. 断点续训时，把已经跑完的 step **跳过**，不要重跑
             b. 内层 for step, batch：
                  - self._forward_backward(batch)
                  - 到达累积边界就 self._optimizer_step()
                  - 按 log_every_n_steps 记录 loss 与当前学习率
                  - 按 save_every_n_steps 存 checkpoint
             c. 内层结束后处理**尾部不足一个累积步的梯度**（见下）
             d. 有 dev 集就在 epoch 末尾 self.evaluate()
        5. 返回 self.state。

        ============================ 必要知识 ============================

        **梯度累积的边界怎么判断。**
        用 `(step + 1) % grad_accum_steps == 0`，不是 `step % accum == 0`。
        差别在第 0 步：后者会在还没有任何梯度时就更新一次，
        等效 batch 少一半。这个 off-by-one 不报错，只是效果差。

        **尾部不完整的累积组必须补救。**
        如果 min(loader 长度) 不是 accum 的整数倍，最后那一小组梯度
        会永远停在 .grad 里没被 step 掉——它们会被下一轮 epoch 的
        zero_grad 清掉，等于白算。所以在内层循环结束后要判断：
        如果最后一步没落在累积边界上，手动再走一次 _optimizer_step()。
        我的冒烟测试目前没覆盖这种情况（样本数刚好整除），
        你自己加一组"样本数不是 accum 整数倍"的数据验证一下。

        **断点续训怎么跳过已完成的 step。**
        最简单的是 `itertools.islice(loader, skip_steps)`——直接消耗掉
        前 N 个 batch 不处理。前提是 DataLoader 的 shuffle 必须可复现
        （固定 generator），否则"跳过的"和你上次"跑过的"不是同一批数据。
        更严谨的做法是把当前 epoch 内的样本顺序存进 checkpoint，
        但对这个项目来说 islice 足够。

        **为什么要自己算 total_steps 而不是用 len(scheduler)。**
        如果最后一个 epoch 不满（比如中途被 Kaggle 掐断后 resume），
        按 epoch 算出来的总步数会和实际跑的步数不一致，
        cosine 曲线就走不到最低点。用实际步数更稳。

        **DataLoader 的两个参数。**
        num_workers 在 Kaggle 上设 2 就够，设大了会因为共享内存不足
        被 OOM killer 干掉（报错信息通常是 "DataLoader worker (pid xxx)
        is killed by signal: Killed"，看起来和显存无关，很容易误判）。
        pin_memory=True 让数据在 CPU 侧锁页，传到 GPU 更快。
        """
        # 必须显式切到 train 模式。
        #
        # HF 的 from_pretrained 结尾会调 model.eval()（它自己的文档里写着
        # "The model is set in evaluation mode by default"），所以模型拿到手时
        # 是 eval。不切回来的话有两个后果，而且**两个都不报错**：
        #
        #   1. **梯度检查点不生效。** ChatGLM3 的 GLMTransformer.forward 里是
        #          if self.gradient_checkpointing and self.training:
        #      training=False 会让整个条件短路 —— 激活值按「每层完整保存」算，
        #      16GB 的 T4 上必然 OOM，而报错栈里全是 bitsandbytes 的调用。
        #      （我们为此查了三轮：先猜 device_map、再猜 max_length、
        #        最后才发现 flag 设对了但 training 是 False。）
        #   2. dropout 全程关闭，正则化失效，模型更容易过拟合。
        self.model.train()

        # 计时/计数起点。峰值显存特意在这里清零，日志里那一行才是
        # "训练一步要多少显存"，而不是把加载模型、注入 LoRA 的一次性峰值算进去。
        self._run_start_time = time.monotonic()
        self._last_save_time = self._run_start_time
        self._samples_this_run = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # DDP 下每条 rank 只处理 1/world_size 的数据，所以每条 rank 的优化步数
        # 也要除以 world_size —— 否则学习率调度会按「单卡的数据量」走，
        # 实际只走到 cosine 曲线的一半就结束了。
        steps_per_epoch = math.ceil(
            len(train_dataset) / (self.batch_size * self.grad_accum_steps * self.world_size)
        )
        total_steps = steps_per_epoch * self.num_epochs
        start_epoch = self.state.epoch
        steps_done_in_epoch = self.state.step - start_epoch * steps_per_epoch

        # DDP 下用 DistributedSampler 把数据分给各个 rank。必须 set_epoch，
        # 否则每个 epoch 的分片方式完全一样 —— 每轮都拿同一批数据，
        # 而且两张卡之间也不会轮换。
        sampler = None
        if is_distributed():
            sampler = DistributedSampler(train_dataset, shuffle=True, seed=self.seed)

        log_loss_sum = 0.0
        log_step_count = 0
        for epoch in range(start_epoch, self.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            batches_to_skip = steps_done_in_epoch * self.grad_accum_steps
            skip = batches_to_skip if epoch == start_epoch else 0
            
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                # shuffle 与 sampler 互斥，同时给会直接报错
                shuffle=sampler is None,
                sampler=sampler,
                collate_fn=self.collator,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
                generator=g # keep epoch res same
            )
            loader_iter = islice(loader, skip, None) # skip batch
            
            batches_this_epoch = 0
            accum_loss = 0
            accum_count = 0
            for batch_idx, batch in enumerate(loader_iter, start=skip):
                accum_loss += self._forward_backward(batch) # loss用于记录，已经反向传播过了
                # 只统计本进程真正处理过的样本数（分母是墙钟时间），
                # 乘以 world_size 就是全局吞吐 —— 用来回答"9 小时能跑多少条"
                self._samples_this_run += batch["input_ids"].size(0)
                batches_this_epoch +=1
                accum_count +=1
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    lr = get_lr(self.state.step, total_steps, self.lr, self.warmup_ratio, self.min_lr_ratio)
                    self._optimizer_step(lr)
                    steps_done_in_epoch += 1
                    self.state.step += 1
                    
                    step_loss = accum_loss / accum_count
                    accum_loss = 0.0
                    accum_count = 0
                    log_loss_sum += step_loss
                    log_step_count += 1
                    
                    if self.state.step % self.log_every_n_steps == 0 and self.is_main_process:
                        avg_log_loss = log_loss_sum / log_step_count
                        # 吞吐和峰值显存直接打在日志里：
                        # 调 batch_size 靠它、算存档间隔靠它、估"这一轮能不能
                        # 在单次会话内跑完"也靠它。事后补算是算不出来的。
                        elapsed = time.monotonic() - self._run_start_time
                        rate = (self._samples_this_run * self.world_size / elapsed
                                if elapsed > 0 else 0.0)
                        peak_gib = (
                            torch.cuda.max_memory_allocated(self.device) / 2**30
                            if self.device.type == "cuda" else 0.0
                        )
                        logger.info(
                            f"epoch {epoch} | step {self.state.step}/{total_steps} | "
                            f"loss {avg_log_loss:.4f} | lr {lr:.2e} | "
                            f"{rate:.2f} 样本/秒 | 峰值显存 {peak_gib:.2f} GiB"
                        )
                        record = {
                            "step": self.state.step,
                            "epoch": epoch + 1,
                            "loss": avg_log_loss,
                            "lr": lr,
                            "dev_loss": None,
                        }
                        self.state.history.append(record)
                        if self.tracker is not None:
                            self.tracker.log(record, self.state.step)
                        log_loss_sum = 0.0
                        log_step_count = 0

                    # ===== 保存 =====
                    if self.is_main_process and self._should_save():
                        self.save()

                    # ===== 周期验证 =====
                    # 放在存档之后：这样 best.pt 和 step_xxxxxx.pt 的 step 号一致，
                    # 事后对着 history 就能知道"最好的那一步"是不是已经存下来了。
                    if (dev_dataset is not None
                            and self.eval_every_n_steps
                            and self.state.step % self.eval_every_n_steps == 0):
                        self._run_eval(dev_dataset, epoch + 1)
                    
            
            # 尾部补救
            if batches_this_epoch % self.grad_accum_steps != 0:
                step_loss = accum_loss / accum_count if accum_count > 0 else 0.0
                lr = get_lr(self.state.step, total_steps, self.lr, self.warmup_ratio, self.min_lr_ratio)
                self._optimizer_step(lr)
                steps_done_in_epoch += 1
                self.state.step += 1

                log_loss_sum += step_loss
                log_step_count += 1

            # epoch 末尾：把剩下的日志区间刷掉（否则不足一个 log 间隔的步会被吞掉）
            if log_step_count > 0 and self.is_main_process:
                avg_log_loss = log_loss_sum / log_step_count
                logger.info(
                    f"epoch {epoch} | step {self.state.step}/{total_steps} | "
                    f"loss {avg_log_loss:.4f} (epoch end)"
                )
                log_loss_sum = 0.0
                log_step_count = 0

            # 先把"已完成几轮"记上，再跑验证 —— 顺序很重要。
            # best.pt 是在验证改善的那一刻落盘的，如果此时 epoch 还没 +1，
            # best.pt 里的 epoch 会比实际进度少一：同一个 step 的两个快照
            # （best.pt 与末尾的 step_xxxxxx.pt）会给出互相矛盾的 epoch。
            # 更实际的后果是：从 best.pt 续训时会多跑一个"空 epoch"，
            # 而 islice 仍然会真的把那 100 个 batch 取出来过一遍 collator，
            # 白白等上一分钟。
            self.state.epoch = epoch + 1

            # ---------- 验证集 ----------
            # evaluate() 的返回值以前被直接丢掉了，这里给它三个去处：
            #   1. 更新历史最佳，并用 tag="best" 把最好的那轮落盘
            #      —— Kaggle 单次会话只有 9 小时，训练被掐是常态，
            #      "哪一轮最好"必须落盘，否则只能靠猜。
            #   2. 上报 tracker，和 train loss 画在同一张图上。
            #   3. 写入 history，事后能判断"train 还在降但 dev 开始涨"
            #      的过拟合点。
            if dev_dataset is not None:
                self._run_eval(dev_dataset, epoch + 1)

            steps_done_in_epoch = 0
        
        # 训练结束落一次最终权重。
        # 少了这一步，当 save_every_n_steps 大于总步数时
        # （短训练很常见，比如 save=200 但整轮只跑了 30 步），
        # 整个过程一个 checkpoint 都不会产生 —— Kaggle 上被掐就全白干。
        if self.is_main_process:
            self.save()

        return self.state
        
        
        
    # ------------------------------------------------------------------
    def _forward_backward(self, batch: Dict[str, torch.Tensor]) -> float:
        """一次前向 + 反向，返回**用于记录的真实 loss**。

        ============================ 要写的步骤 ============================

        1. 把 batch 里的每个 tensor 搬到 self.device（用 non_blocking=True）。
        2. 在 autocast 上下文里跑 forward：
               outputs = self.model(**batch)
               loss = outputs.loss
        3. 记录用的 loss 先存下来：loss_value = loss.detach().item()
        4. 除以累积步数：(loss / grad_accum_steps)
        5. self.scaler.scale(scaled_loss).backward()
        6. 返回 loss_value（**未经除法**的那个）。

        ============================ 必要知识 ============================

        **为什么要用 outputs.loss 而不是自己写 CrossEntropyLoss。**
        因果语言模型的 loss 需要把 logits 和 labels **错开一位**
        （第 i 个位置的 logits 预测第 i+1 个 token）。自己写极容易
        写成同位的，那样模型学的是"复制输入"，loss 会降得很快但
        生成出来全是废话。让模型内部处理这件事。

        **为什么返回值不是除过法的那个。**
        除以累积步数只是为了让梯度尺度正确，不是真实 loss。
        如果日志里记的是除过法的值，你会看到 loss 莫名小 16 倍，
        然后误以为训练收敛得很好。

        **为什么 device 搬运要用 non_blocking=True。**
        配合 pin_memory 使用时，拷贝可以和计算重叠。不加也不会错，
        只是慢一点。

        **autocast 的上下文对象在 __init__ 里建还是这里建。**
        建议在 __init__ 里建好存成 self.autocast_ctx，因为 CPU 上必须
        用 nullcontext()（CPU 不支持 fp16 autocast），而设备在构造时
        就确定了。注意 bf16 在 CPU 上是支持的，fp16 不是。

        **loss.detach().item() 的 detach 不能省。**
        .item() 本身会取出标量，但张量如果还挂在计算图上，
        持有它会阻止计算图释放。累积步数大的时候这会明显吃显存。
        """
        batch = {
            k: v.to(self.device, non_blocking=True)
            for k, v in batch.items()
        }
        
        with self.autocast_ctx:
            outputs = self.model(**batch)
            loss = outputs.loss
        loss_value = loss.detach().item() # .item() 只用来“读数值”，绝不能参与反向传播。除以 grad_accum_steps 必须对张量做，不能对 float 做。
        scaled_loss = loss / self.grad_accum_steps
        self.scaler.scale(scaled_loss).backward() # 对张量 backward
        return loss_value
        # TODO
        # 自己实现loss计算
            


    # ------------------------------------------------------------------
    def _optimizer_step(self, lr: float) -> None:
        """一次参数更新。

        ============================ 要写的步骤 ============================

        按顺序做五件事，顺序不能换：

            1. self.scaler.unscale_(self.optimizer)
            2. torch.nn.utils.clip_grad_norm_(可训练参数, max_grad_norm)
            3. self.scaler.step(self.optimizer)
            4. self.scaler.update()
            5. self.optimizer.zero_grad(set_to_none=True)

        ============================ 必要知识 ============================

        **为什么 unscale 必须在 clip 之前。**
        backward 时梯度被 GradScaler 放大过（比如 65536 倍）。
        直接按 max_grad_norm=1.0 裁剪，等于把阈值变成了 1.0/65536，
        梯度会被裁到几乎为 0，训练直接停滞——而这个过程完全不报错。

        **scaler.step() 不一定真的更新参数。**
        GradScaler 会检查梯度里有没有 inf/nan。如果有，它会**跳过这一步**，
        并调低缩放因子。这是它的保护机制，不是 bug。所以看到
        "某几步 loss 没变化"通常是正常的。
        如果想确认，可以记录 scaler.get_scale()，正常训练中它应该是
        稳定或缓慢下降的；如果一直剧烈下降，说明 fp16 数值不稳定。

        **为什么 zero_grad 要带 set_to_none=True。**
        默认行为是把 .grad 张量就地填 0，但张量本身还在，显存照占。
        set_to_none=True 直接释放掉，下一轮 backward 时重新分配，
        更省显存也更快。这是 PyTorch 官方推荐的写法。

        注意这一条要放在 backward **之后**、下一轮 backward **之前**。
        最稳妥的位置是本函数末尾——这样它在循环里的位置就固定了，
        不会因为累积边界的判断写错而破坏"梯度先累积后清零"的语义。
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.trainable_param,self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        
        
    # ------------------------------------------------------------------
    def _run_eval(self, dev_dataset: Any, epoch: int) -> float:
        """跑一次验证，把结果送到该去的地方。返回 dev_loss。

        **epoch 末尾**和**按步数触发的周期验证**走的是同一个函数。
        分成两份写的话，"什么算新最好、什么时候存 best.pt"这套判定逻辑
        会长出两份，而两份迟早会不一致 —— 那种 bug 表现为
        "history 里 best 明明是第 40 步，但 best.pt 是第 80 步的"。

        同一个 step 上重复调用直接返回上次的结果（见 __init__ 里的说明）。
        """
        if self._last_eval_step == self.state.step:
            return self._last_dev_loss

        dev_loss = self.evaluate(dev_dataset)
        self._last_eval_step = self.state.step
        self._last_dev_loss = dev_loss

        if dev_loss < self.state.best_dev_loss:
            self.state.best_dev_loss = dev_loss
            if self.is_main_process:
                self.save(tag="best")

        self.state.history.append({
            "step": self.state.step,
            "epoch": epoch,
            "loss": None,
            "lr": None,
            "dev_loss": dev_loss,
        })

        if self.is_main_process:
            logger.info(
                "step %d | epoch %d | dev_loss %.4f | best %.4f",
                self.state.step, epoch, dev_loss, self.state.best_dev_loss,
            )
            if self.tracker is not None:
                self.tracker.log(
                    {
                        "dev_loss": dev_loss,
                        "best_dev_loss": self.state.best_dev_loss,
                    },
                    self.state.step,
                )
        return dev_loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, dataset: Any, max_samples: Optional[int] = None) -> float:
        """在验证集上算平均 loss。

        ============================ 要写的步骤 ============================

        1. self.model.eval()
        2. 构造 DataLoader（shuffle=False，验证集不需要打乱）
        3. 用 try/finally 包住循环，finally 里 self.model.train()
        4. 累加 `loss * batch_size` 和样本数
        5. 返回 总损失 / 总样本数

        ============================ 必要知识 ============================

        **为什么必须 model.eval()。**
        训练模式下 dropout 会随机丢弃激活，验证 loss 会带随机噪声，
        每次算出来的值都不一样，就没法用它判断"是不是变好了"。
        同理 BatchNorm 也会用当前 batch 的统计量而不是滑动平均。

        **为什么必须切回 model.train()。**
        这个最容易漏。切不回去的话，后续训练里 dropout 一直是关闭的——
        模型会过拟合得更快，但你只会觉得"验证 loss 怎么这么快就变差了"。
        用 try/finally 而不是在函数末尾写一句，是因为中途抛异常时
        末尾那句根本不会执行。

        **为什么要按样本数加权平均，而不是各 batch 求平均。**
        最后一个 batch 通常不满（比如 1000 个样本、batch_size=8，
        最后一批只有 4 个）。直接对 batch 求平均会让那一批的权重
        和满批一样大，指标会有系统性偏差。

        **@torch.no_grad() 装饰器 vs with 语句。**
        两者等价。装饰器写在函数上更不容易忘。但注意它只关掉梯度记录，
        **不会**自动切换 eval 模式——两件事都要做。

        **max_samples 为什么是"等间隔取"而不是"取前 N 条"。**
        如果验证集是按案由**排序**的（先全部借款合同，再全部劳动合同……），
        `records[:200]` 就会把某一类案由全取进来、另一类一条不取，dev_loss
        的含义随之漂移 —— 而且这种偏差不会报错，只会让"这轮比上轮好"的
        判断失去意义。等间隔取样（每 len/N 条取一条）保住了原始分布。
        本项目的切分文件目前是打散的（实测前 200 条与全量分布一致），
        但这属于数据的性质、不是代码的保证，所以不依赖它。
        """
        # 默认用配置里的上限；显式传参可以覆盖（测试用）
        max_samples = max_samples if max_samples is not None else self.eval_max_samples
        if max_samples and len(dataset) > max_samples:
            stride = len(dataset) / max_samples
            indices = [int(i * stride) for i in range(max_samples)]
            dataset = Subset(dataset, indices)

        self.model.eval()
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        
        loss = 0.0
        samples_num = 0
        try:
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                with self.autocast_ctx:
                    loss += self.model(**batch).loss.detach().item() * batch["input_ids"].size(0)
                samples_num+=batch["input_ids"].size(0)
        finally:
            self.model.train()
        return loss / samples_num

    # ------------------------------------------------------------------
    def _should_save(self) -> bool:
        """到存档点了吗？步数触发和时间触发是「或」的关系。

        返回 True 的两种情形：

            step % save_every_n_steps == 0        步数到了
            距上次存档 >= save_every_n_minutes    墙钟时间到了

        只在**优化步**（梯度累积的边界）上被调用，不会每处理一个 batch 就调一次，
        所以"时间到了"最多晚一个优化步落盘。

        计时器在每次 save() 里重置，因此时间触发不会被步数触发的存档打乱节奏：
        谁先存，谁的计时器就归零。
        """
        if self.save_every_n_steps and self.state.step % self.save_every_n_steps == 0:
            return True
        if self.save_every_n_minutes:
            return (time.monotonic() - self._last_save_time) >= self.save_every_n_minutes * 60
        return False

    # ------------------------------------------------------------------
    def save(self, tag: str = "last") -> Path:
        r"""保存 checkpoint。

        ============================ 要写的步骤 ============================

        1. 非主进程直接返回（多进程会互相覆盖同一个文件）
        2. 组装路径：tag="best" 时用 best.pt，否则用
           self.output_dir / f"step_{self.state.step:06d}.pt"
        3. 调 save_checkpoint(...)，传 model / optimizer / scaler /
           scheduler / state，并保持 only_trainable=True
        4. 按 keep_last_n_checkpoints 调 prune_checkpoints 清理旧文件
        5. 返回路径

        ============================ 必要知识 ============================

        **只存 adapter，不存整个模型。**
        LoRA 的可训练参数通常只有几十 MB，而完整的 6B 模型是十几 GB。
        Kaggle 的 /kaggle/working 有输出配额，每次存全量很快就会顶满。
        checkpoint.py 的 only_trainable=True 已经处理了这件事。

        **但断点续训还需要优化器和 scaler 的状态。**
        优化器带 Adam 的一阶/二阶动量，scaler 带当前的缩放因子。
        少了它们，resume 之后相当于换了个优化器从头开始——
        损失曲线会出现一个明显的回升。这两件事和"推理时只存 adapter"
        是两个不同的用途，别混在一起。

        **scheduler 状态存不存？**
        我们的学习率是 get_lr(step, total_steps) 纯函数算出来的，
        没有内部状态，所以不用存。如果改用 torch.optim.lr_scheduler，
        就必须存 state_dict，否则续训后学习率会从头开始走。
        这也是自己写调度的好处之一——状态更少，更不容易出错。

        **文件名里的 step 编号有什么用。**
        一是让 prune_checkpoints 能用 `step_(\d+)\.pt` 这个正则识别并按数字排序，
        二是人工在目录里翻的时候也是数字序。
        必须用 zero-padding（%06d）：字符串排序和数字排序在位数不同时结果不一样，
        "step_1000.pt" 会排在 "step_400.pt" 前面。
        """
        if not self.is_main_process:
            return Path()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if tag == "best":
            # 固定文件名 —— prune_checkpoints 只匹配 step_*.pt，
            # 所以最佳模型不会被按数量清理掉。
            path = self.output_dir / "best.pt"
        else:
            path = self.output_dir / f"step_{self.state.step:06d}.pt"
        
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            state=self.state,
            only_trainable=True
        )
        
        prune_checkpoints(
            ckpt_dir=self.output_dir,
            keep_last_n=self.keep_last_n_checkpoints
        )

        # 计时器归零：下一次「时间到了」从这次存档重新计时
        self._last_save_time = time.monotonic()

        return path

    # ------------------------------------------------------------------
    def resume(self, path: Optional[str] = None) -> TrainState:
        """从指定的 checkpoint 继续训练；没有指定就从零开始。

        ============================ 只有两种模式 ============================

            path（或配置里的 sft.resume_from）为空
                → 从头训练。**不会**去目录里找"最新的那个"，
                  也不会碰任何已有文件。

            path 指向一个具体文件
                → 从它继续。文件不存在直接抛 FileNotFoundError。

        ============================ 为什么不做"自动找最新" ============================

        看起来方便，但风险很大，而且三种风险都不报错：

        1. **接错实验**。output_dir 里混着不同实验的 checkpoint 时
           （比如 lora_r8 和 lora_r16 共用 outputs/sft），
           按 step 编号取最大的那个会毫不犹豫地接错。
        2. **超参不匹配**。改了 learning_rate 之后接上旧的 checkpoint，
           AdamW 的动量是在旧学习率下累积的，scaler 的缩放因子也是旧的，
           接上去不是"继续训练"，是两个实验拼在一起。
        3. **计数错乱**。state.step 从中间起步，实验跟踪面板上会看到
           一条从 4000 步开始的曲线，看起来像实验被截断成两段。

        所以续训必须被**明确指定**。想从头训练就把配置留空，
        而不是指望程序猜。

        ============================ 必要知识 ============================

        **strict=False 是必须的。**
        save() 只保存可训练参数（LoRA 的 A/B 矩阵），模型里那一大片
        冻结的底座权重不在 checkpoint 里。strict=True 会因为缺键直接报错。

        **为什么"文件不存在"要报错，而不是静默从头开始。**
        你既然写了路径，意图就是续训。文件不在说明环境出了问题——
        最常见的是 Kaggle 上忘了挂载存有 checkpoint 的 Dataset。
        静默从头开始会让你在几小时后才发现，白烧一堆 GPU 时间。
        "我想从头训练"的表达方式是**把配置留空**，不是写一个错的路径。

        **TrainState 可以直接从 dict 重建。**
        save() 存的是 state.to_dict()，这里用 TrainState(**state) 就能还原，
        不需要手写字段映射。

        **恢复之后还要保证数据顺序一致。**
        resume 能恢复参数和 step，但跳过的那些 batch 要和上次跑过的是同一批，
        否则等于跳错了位置。这依赖 DataLoader 的 shuffle 种子固定
        （train() 里用 seed + epoch 手动设了 generator），
        所以 set_seed 必须在构造 DataLoader 之前调用。
        """
        # 传入的 path 优先，否则用配置里的 resume_from
        spec = path if path is not None else self.resume_from

        # ---------- 模式一：从头训练 ----------
        if not spec:
            logger.info("未指定 resume_from，从头开始训练")
            return self.state

        # ---------- 模式二：从指定文件继续 ----------
        ckpt_path = Path(spec)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"resume_from 指定的 checkpoint 不存在：{ckpt_path}。"
                "如果想从头训练，请把 configs/sft.yaml 里的 resume_from 设为 null。"
            )

        info = load_checkpoint(
            ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
        )
        self.state = TrainState(**(info.get("state") or {}))
        logger.info(
            "已从 %s 恢复：step=%d, epoch=%d",
            ckpt_path,
            self.state.step,
            self.state.epoch,
        )
        return self.state
