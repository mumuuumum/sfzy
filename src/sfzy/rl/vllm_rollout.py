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
        self._lora: Any = None
        self.stats: Dict[str, float] = {"sync_calls": 0, "sync_seconds": 0.0, "generate_calls": 0}

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
        self._engine = LLM(**kwargs)
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
        prompt_ids: List[List[int]] = []
        for messages in messages_list:
            ids = list(encode_prompt(self.tokenizer, messages, add_generation_prompt=True))
            if len(ids) > max_length:
                ids = ids[-max_length:]
            prompt_ids.append(ids)

        sampling = factory(
            n=int(num_return_sequences),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_new_tokens),
        )
        start = time.perf_counter()
        outputs = self.engine.generate(
            prompt_ids, sampling, lora_request=self._lora
        )
        self.stats["generate_calls"] += 1
        elapsed = time.perf_counter() - start

        texts: List[str] = []
        for out in outputs:
            for item in out.outputs:
                texts.append(str(item.text).strip())
        expected = len(messages_list) * int(num_return_sequences)
        if len(texts) != expected:
            raise ValueError(
                f"vLLM 返回 {len(texts)} 条，期望 {expected} 条（{len(messages_list)} 个 prompt × "
                f"{num_return_sequences}）—— 数量对不上说明接口用错了，先别继续训练。"
            )
        logger.info(
            "vLLM 生成完成：%d 个 prompt × %d 条，用时 %.1fs（%.2f 条/秒）",
            len(messages_list), num_return_sequences, elapsed,
            expected / max(elapsed, 1e-6),
        )
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
