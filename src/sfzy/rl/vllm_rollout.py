"""vLLM 版 rollout 后端（可选）：一次采样 G 条 + 每步热交换策略 LoRA。

====================== 它解决的是什么问题 ======================
GRPO 的瓶颈是 rollout：每个 prompt 要采 G 条，且**策略每步都在变**。HF 那条路
对 ChatGLM3 只能逐条串行生成（`stream_generate` 不支持带 padding 的批输入），
所以 vLLM 的价值在于它把"同时生成很多条"做到极致（连续批处理 + PagedAttention）。

====================== 每步怎么把新策略交给 vLLM ======================
**不需要**新版才有的 HTTP 接口，也不需要在磁盘上留一堆 adapter：

1. 把当前模型里的 LoRA 导出成一个 PEFT 目录（`models/lora_export.py`）；
2. 用一个**新的 `lora_int_id`** 构造 `LoRARequest`（vLLM 按 int_id 缓存 adapter，
   `max_loras=1` / `max_cpu_loras=1` 时旧的会被 LRU 淘汰）；
3. 生成请求带 `lora_request=...` 即可。

磁盘上只保留 2 份（交替复用 `step0/`、`step1/`），每份约 60~120 MB。

====================== 为什么不用它算 old_logprobs ======================
vLLM 确实能返回 per-token logprobs，但本项目 `GRPOTrainer.rollout()` 的既有约定是
"生成只产文本，logprob 由**训练侧模型**在同一批 token 上重算"。后者让 μ 内层循环、
clip、KL 的语义完全不受后端影响 —— 换热交换后端就只换"谁来写这段文本"。
（想省掉那次前向再说，那是另一个更大的改动。）

====================== 依赖 ======================
vllm 只在真正构造后端时才 import（`hf` 后端、CI、没有 GPU 的机器都不需要装它）。
版本与精度选择见 `docs/vllm_rollout.md`：**vLLM 0.11.2 + transformers 4.56.x**、
底座用 **bf16**（最省心）或 **AWQ**（显存紧张时）。
**不要用 bitsandbytes 量化底座** —— vLLM 的 LoRA 只对 AWQ/GPTQ 做过测试。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from sfzy.models.chat_template import encode_prompt
from sfzy.models.lora_export import CHATGLM3_TARGET_MODULES, export_lora_from_model
from sfzy.utils.logging import get_logger

logger = get_logger("vllm_rollout")


class VLLMRollout:
    """vLLM 生成 + 动态 LoRA 热交换。

    `engine` 参数用于测试注入假引擎（真引擎是 `vllm.LLM`）；不传就自己建。
    """

    def __init__(
        self,
        *,
        base_model: str,
        tokenizer: Any,
        lora_alpha: int,
        lora_rank: int,
        export_root: Union[str, Path],
        dtype: str = "bfloat16",
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.30,
        max_model_len: Optional[int] = None,
        max_loras: int = 1,
        target_modules: Sequence[str] = CHATGLM3_TARGET_MODULES,
        trust_remote_code: bool = True,
        extra_llm_kwargs: Optional[Dict[str, Any]] = None,
        engine: Any = None,
        lora_request_factory: Any = None,
        sampling_params_factory: Any = None,
        collect_logprobs: bool = False,
    ) -> None:
        self.base_model = str(base_model)
        self.tokenizer = tokenizer
        self.lora_alpha = int(lora_alpha)
        self.lora_rank = int(lora_rank)
        self.export_root = Path(export_root)
        self.dtype = dtype
        self.quantization = quantization
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_model_len = max_model_len
        self.max_loras = int(max_loras)
        self.target_modules = tuple(target_modules or CHATGLM3_TARGET_MODULES)
        # **必须默认 True**：vLLM 自带 chatglm 的 config 类与模型实现，但配置解析那一步
        # 仍会走 transformers 的 AutoConfig；而本地目录里有 configuration_chatglm.py +
        # config.json 的 auto_map，`trust_remote_code=False` 时 transformers 会直接拒绝：
        #
        #     ValueError: The repository ... contains custom code which must be executed to
        #     correctly load the model. ... Please pass the argument `trust_remote_code=True`
        #
        # 注意这只放行**配置类**的加载；模型前向仍走 vLLM 自己的 ChatGLMForCausalLM
        # （registry 里 ChatGLMModel → chatglm 模块），不会执行远程的 modeling 代码。
        self.trust_remote_code = bool(trust_remote_code)
        self.extra_llm_kwargs = dict(extra_llm_kwargs or {})
        self._engine = engine
        # 测试可注入假的 LoRARequest 工厂（没装 vllm 的机器也能验证这条链路）
        self._lora_request_factory = lora_request_factory
        # 同理：采样参数对象也允许注入（没装 vllm 的机器上测试用）
        self._sampling_params_factory = sampling_params_factory
        # 要不要让 vLLM 顺带返回**采样 token 的 logprob**（省掉训练侧那次 old_logprobs 前向）。
        #
        # ★ 默认 **False**，理由是实测出来的：vLLM 与 HF 是两套实现（kernel/attention/
        #   温度语义都不同），同一条几百 token 的序列，两边 logprob 的累计差很容易超过
        #   ±20% 的裁剪带 → `μ2 clip` 的第一个数（第 1 次内层迭代）从 0% 变成 25~50%，
        #   而被裁的样本在 `clamp` 带外**梯度为 0**。即"省 3.4s/步"换来"1/4~1/2 的样本
        #   不产生梯度"，不划算。要做对照实验可以打开（`rl.vllm.use_sampling_logprobs`）。
        #
        #   注意：**token id 仍然照常带回**（用于跳过 decode→re-encode 的往返），
        #   这个开关只决定"old_logprobs 用 vLLM 的还是本地重算"。
        self.collect_logprobs = bool(collect_logprobs)
        self._lora: Any = None
        # 最近一次生成的原生结果（token id / 文本 / logprob），供 trainer 直接用
        self.last_token_ids: Optional[List[List[int]]] = None
        self.last_logprobs: Optional[List[List[float]]] = None
        self.stats: Dict[str, float] = {
            "sync_calls": 0, "sync_seconds": 0.0,
            "generate_calls": 0, "generate_seconds": 0.0,
            "init_seconds": 0.0,
        }

    # ------------------------------------------------------------------ 引擎
    def _build_engine(self) -> Any:
        """真正建 vLLM 引擎（只在第一次用到时 import vllm）。"""
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise RuntimeError(
                "配置了 rl.rollout_backend: vllm，但当前环境没有装 vllm。\n"
                "  pip install -r requirements-vllm.txt   # 版本组合见 docs/vllm_rollout.md\n"
                "  或者把 rl.rollout_backend 改回 hf（原来的逐条生成）。"
            ) from exc

        kwargs: Dict[str, Any] = dict(
            model=self.base_model,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
            # LoRA 只能配 max_loras 个；我们每步换一个 int_id，旧的按 LRU 淘汰
            enable_lora=True,
            max_lora_rank=self.lora_rank,
            max_loras=self.max_loras,
            # 不设的话默认 0.9 —— 会把同一张卡上的训练进程挤爆
            disable_log_stats=True,
        )
        if self.quantization:
            kwargs["quantization"] = self.quantization
        if self.max_model_len:
            kwargs["max_model_len"] = int(self.max_model_len)
        kwargs.update(self.extra_llm_kwargs)
        logger.info(
            "初始化 vLLM：model=%s dtype=%s quantization=%s gpu_memory_utilization=%.2f "
            "max_model_len=%s lora_rank=%d",
            self.base_model, self.dtype, self.quantization or "（无）",
            self.gpu_memory_utilization, self.max_model_len, self.lora_rank,
        )
        if not self.quantization:
            # 没指定量化 → vLLM 会按 dtype 加载**整个**底座（6B bf16 ≈ 12.4GB 显存），
            # 而不是训练侧那种 bnb 4-bit（≈3.5GB）。这一步差出来的 ~9GB 会直接决定
            # 能不能和训练同卡 —— 实测就有人把冒烟时用的 `--quantization bitsandbytes`
            # 忘了写进配置，于是训练时 vLLM 悄悄吃了 15.5GB 并 OOM。
            logger.warning(
                "vLLM 未指定量化：会按 %s 加载完整底座（6B 约 12.4GB）+ KV cache。"
                "与训练同卡时要么把 gpu_memory_utilization 提到 ~0.30，"
                "要么改用 rl.vllm.quantization: bitsandbytes（约 3.5GB，已验证可用）。",
                self.dtype,
            )
        start = time.perf_counter()
        self._engine = LLM(**kwargs)
        # 引擎初始化是**一次性**开销（加载权重 + KV cache + CUDA graph 捕获），
        # 实测 25~45 秒。必须单独计时：混进"生成耗时"里会让第一步看起来慢十倍。
        self.stats["init_seconds"] += time.perf_counter() - start
        logger.info("vLLM 引擎初始化完成：%.1fs（一次性开销）", self.stats["init_seconds"])
        return self._engine

    @property
    def engine(self) -> Any:
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    # ------------------------------------------------------------------ 同步策略
    def sync_policy(self, model: Any, step: int) -> Any:
        """把当前策略的 LoRA 交给 vLLM，返回这次请求要带的 LoRARequest。"""
        factory = self._lora_request_factory
        if factory is None:
            # 真的用到 vLLM 时才 import
            from vllm.lora.request import LoRARequest  # noqa: PLC0415

            factory = LoRARequest

        start = time.perf_counter()
        # 交替用 step0/step1 两个目录，磁盘占用恒定（只保留最近两份）
        adapter_dir = self.export_root / f"step{int(step) % 2}"
        export_lora_from_model(
            model, adapter_dir,
            alpha=self.lora_alpha, target_modules=self.target_modules,
            base_model=self.base_model, rank=self.lora_rank,
        )
        # int_id 必须 ≥ 1（0 在 vLLM 里表示"不用 LoRA"），且每步都不同
        self._lora = factory(
            lora_name=f"policy-{int(step)}",
            lora_int_id=int(step) + 1,
            lora_path=str(adapter_dir),
        )
        self.stats["sync_calls"] += 1
        self.stats["sync_seconds"] += time.perf_counter() - start
        logger.info("策略已同步给 vLLM：step=%d adapter=%s", step, adapter_dir)
        return self._lora

    # ------------------------------------------------------------------ 生成
    def generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        *,
        num_return_sequences: int,
        max_new_tokens: int,
        max_length: int,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> List[str]:
        """每个 prompt 采 `num_return_sequences` 条，返回**prompt 优先**展平的文本。

        顺序与 `sft/infer.generate_batch` 一致（第 i 个 prompt 的 G 条连续排列），
        `GRPOTrainer.rollout()` 按 `responses[i * G + j]` 取用。
        """
        if self._lora is None:
            raise RuntimeError(
                "还没给 vLLM 同步策略（sync_policy 没调）—— rollout 会用到未挂 LoRA 的底座。"
            )
        factory = self._sampling_params_factory
        if factory is None:
            from vllm import SamplingParams  # noqa: PLC0415

            factory = SamplingParams

        # prompt 的编码与截断规则必须和 HF 路径**逐条一致**：同一个 encode_prompt、
        # 同样的"保留尾部"。不一致不会报错，只会让两边生成的分布悄悄不同。
        #
        # ★ 形态：vLLM 0.11 的 `generate(prompts=...)` 只认 `str` 或 dict 形态的
        #   `PromptType`（`{"prompt_token_ids": [...]}` / `{"prompt": "..."}`）。
        #   **裸的 token id 列表（`[[1, 2, 3]]`）是 0.5.x 的老写法**：0.11 会先做
        #   `"prompt_token_ids" in prompt`（对 list 求成员、恒为 False），随后对它
        #   调 `.get()` → `AttributeError: 'list' object has no attribute 'get'`。
        #   踩过一次，别再改回裸列表。
        prompt_ids: List[Dict[str, Any]] = []
        for messages in messages_list:
            ids = list(encode_prompt(self.tokenizer, messages, add_generation_prompt=True))
            if len(ids) > max_length:
                ids = ids[-max_length:]
            prompt_ids.append({"prompt_token_ids": ids})

        sampling = factory(
            n=int(num_return_sequences),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_new_tokens),
            # logprobs=0 ⇒ 只返回**被采样到的那个 token** 的 logprob（正是我们要的）
            logprobs=0 if self.collect_logprobs else None,
        )
        # ★ 先把引擎取出来（懒加载会在这里发生），再开始计时 —— 否则第一次的
        #   初始化开销会被算进"生成耗时"（踩过：40.7s 里 40s 是初始化）。
        engine = self.engine
        start = time.perf_counter()
        outputs = engine.generate(prompt_ids, sampling, lora_request=self._lora)
        self.stats["generate_calls"] += 1
        elapsed = time.perf_counter() - start
        self.stats["generate_seconds"] += elapsed

        texts: List[str] = []
        token_ids: List[List[int]] = []
        logprobs: List[List[float]] = []
        ids_complete = True      # 每条都有 token_ids（用于直通建序列）
        lp_complete = True       # 每条都有完整 logprob（用于省掉 old_logprobs 前向）
        for out in outputs:
            for item in out.outputs:
                texts.append(str(item.text).strip())
                ids = [int(t) for t in (getattr(item, "token_ids", None) or [])]
                token_ids.append(ids)
                if self.collect_logprobs:
                    # `item.logprobs` 是"每个位置一个 {token_id: Logprob}" 的列表
                    step_lp: List[float] = []
                    for entry in (getattr(item, "logprobs", None) or []):
                        if not entry:
                            step_lp.append(0.0)
                            continue
                        picked = (
                            entry.get(ids[len(step_lp)]) if len(ids) > len(step_lp) else None
                        )
                        if picked is None:
                            picked = max(
                                entry.values(), key=lambda v: getattr(v, "logprob", -1e30)
                            )
                        step_lp.append(float(getattr(picked, "logprob", 0.0)))
                    logprobs.append(step_lp)
                    if not ids or len(step_lp) != len(ids):
                        lp_complete = False
                if not ids:
                    ids_complete = False
        expected = len(messages_list) * int(num_return_sequences)
        if len(texts) != expected:
            raise ValueError(
                f"vLLM 返回 {len(texts)} 条，期望 {expected} 条（{len(messages_list)} 个 prompt × "
                f"{num_return_sequences}）—— 数量对不上说明接口用错了，先别继续训练。"
            )
        # token id 只要"每条都有"就交给 trainer（它用来直接建序列）；
        # logprob 只有在**每一条都完整**时才给（否则回退到本地前向）。
        self.last_token_ids = token_ids if ids_complete else None
        self.last_logprobs = logprobs if (self.collect_logprobs and lp_complete) else None
        if self.collect_logprobs and not lp_complete:
            logger.warning(
                "vLLM 没有为所有结果返回 token_ids+logprobs，本次回退到"
                "『训练侧自己跑 old_logprobs』（多一次前向，但结果仍正确）。"
            )
        if not ids_complete:
            logger.warning(
                "vLLM 没有为所有结果返回 token_ids，本次回退到『解码文本再重新分词』"
                "构造训练序列。"
            )
        logger.info(
            "vLLM 生成完成：%d 个 prompt × %d 条，用时 %.1fs（%.2f 条/秒）",
            len(messages_list), num_return_sequences, elapsed,
            expected / max(elapsed, 1e-6),
        )
        if self.stats["generate_calls"] == 1:
            # 第一次生成后打一次**整卡**显存：vLLM 引擎跑在子进程里，训练进程的
            # `torch.cuda.memory_allocated()` 看不到它 —— 调 gpu_memory_utilization
            # 时必须看整卡，否则就是在猜。
            try:
                import torch

                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    logger.info(
                        "显存：整卡 %.1f/%.1f GiB 已用（本进程 %.1f GiB；引擎在子进程，"
                        "gpu_memory_utilization=%.2f 决定它占多少）",
                        (total - free) / 1024**3, total / 1024**3,
                        torch.cuda.memory_allocated() / 1024**3,
                        self.gpu_memory_utilization,
                    )
            except Exception:  # noqa: BLE001 - 只是日志，失败不影响训练
                pass
        return texts

    # ------------------------------------------------------------------ 收尾
    def close(self) -> None:
        """释放引擎（vLLM 的显存不显式回收会一直占着）。"""
        if self._engine is None:
            return
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:  # noqa: BLE001 - 版本差异，收尾失败不该影响训练结果落盘
            pass
        self._engine = None
