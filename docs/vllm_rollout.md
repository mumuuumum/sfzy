# 用 vLLM 做 GRPO 的 rollout

目标：把 `rollout`（每个 prompt 采 G 条）从"逐条串行"换成"连续批处理"。
ChatGLM3 在 HF 里只能逐条生成（`stream_generate` 不支持带 padding 的批输入），
vLLM 自带 ChatGLM3 实现 + 连续批处理 + PagedAttention，是这条路的上限。

## 0. 版本决策（已定，别再试别的组合）

| 依赖 | 版本 | 依据 |
|---|---|---|
| vLLM | `0.11.2` | 支持 ChatGLM3；文档里有运行时动态 LoRA（`VLLM_ALLOW_RUNTIME_LORA_UPDATING` / `load_inplace`） |
| torch | `==2.9.0` | vLLM 0.11.2 的 wheel METADATA 硬 pin，别手动改 |
| transformers | `>=4.56.0,<5` | vLLM 0.11.2 自己的上限；**4.x 是仓库 compat 层的主场** |
| 底座精度 | **bf16**（默认）或 AWQ | vLLM 的 LoRA 白名单只有 AWQ/GPTQ；bitsandbytes 底座 + LoRA 是未测试组合 |

关于"ChatGLM3 挑 transformers 版本"：这个问题**已经解决**。仓库的 `models/compat.py`
补齐了 5.x 删掉的两个生成辅助方法（`_get_logits_warper`、
`_extract_past_from_model_output`）与 `generation_config` 默认值 —— 实测在
transformers **5.19.0** 上 `GRPOTrainer.step()` / `train()` 都能跑通，仓库全量测试
580 passed（唯一的 failed 在 4.40.2 上同样失败，是 caplog 的环境问题）。
只是 vLLM 0.11.2 自己把上限定在 `<5`，所以这一版环境用 **4.56.x**；
将来 vLLM 放开到 5.x 时，那部分移植已经就位。

## 1. 建环境

```bash
conda create -n sfzy-vllm python=3.12 -y && conda activate sfzy-vllm
pip install -r requirements-vllm.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
```

AutoDL 镜像自带 torch。若自带的是 2.8/2.7，**不要去手动装别的 torch**（会装成 CUDA
不匹配的 wheel，GPU 直接不可用），让 `pip install vllm==0.11.2` 自己把 torch 拉到 2.9.0。

## 2. 冒烟（1~2 分钟，先跑这个）

```bash
export SFZY_JUDGE_API_KEY=...        # 训练要用；冒烟只需要 rollout，不填也行
# 把 worktree 拉起来，确保数据与 adapter 路径都在
python tools/check_vllm_rollout.py --config configs/grpo_fact_coverage_vGPU48.yaml
```

它会打印：条数、策略同步耗时、生成耗时、显存峰值、一条样例输出。看到
`✔ vLLM rollout 可用` 就可以开训；失败时它会把三个最可能的原因写出来。

## 3. 显存不够时的两条路

同卡共存：训练侧（4-bit QLoRA）峰值约 22~26 GB，vLLM 侧 bf16 底座 12.4 GB + KV
≈ 14~15 GB（`gpu_memory_utilization: 0.30`）→ 合计 40 GB 上下，48 GB 可用但偏紧。

**先试这些免费的旋钮**（都不改数值，只降峰值）：

```yaml
rl:
  prompts_per_step: 1          # 每步 1 个 prompt（批量 4 条 → 不再 ×2）
  logprob_micro_batch: 1       # logits 峰值按比例下降
  vllm:
    gpu_memory_utilization: 0.28
    extra_llm_kwargs: {enforce_eager: true}   # 省掉 CUDA graph 的额外占用
```

还不行就把底座转 AWQ（vLLM 的 LoRA 白名单里有它，权重降到 ~5.5 GB）：

```bash
pip install autoawq          # 注意：autoawq 对新版 torch 跟进较慢，装不上就跳过这条
python - <<'PY'
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
src, dst = "/root/autodl-tmp/chatglm3-6b", "/root/autodl-tmp/chatglm3-6b-awq"
model = AutoAWQForCausalLM.from_pretrained(src, trust_remote_code=True)
AutoTokenizer.from_pretrained(src, trust_remote_code=True).save_pretrained(dst)
model.quantize(AutoTokenizer.from_pretrained(src, trust_remote_code=True), quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})
model.save_quantized(dst)
PY
# 然后配置里：model.model_name_or_path 指向 -awq 目录，rl.vllm.quantization: awq
```

**注意**：AWQ 是另一套 4-bit 数值，与你 SFT 基线的 nf4 不是同一套 kernel；
换过去等于换了底座精度，"精度与 SFT 一致"这条要重新表述。所以**优先 bf16**。

### 3.1 顺便验证一下"bnb 4-bit 底座 + LoRA"（2 分钟，可能省 7GB）

vLLM 0.11.2 的 LoRA 支持的底座类型是有源码清单的（`vllm/lora/layers/utils.py`）：
未量化 Linear、compressed-tensors（`weight_packed`）、**GPTQ/AWQ**（`qweight`）、
**Marlin**（`B`）、**HQQ marlin**（`W_q`）；其它一律 `raise ValueError("Unsupported
base layer")`。**bitsandbytes 不在这个清单里**（0.5.5 里那句 "not tested with LoRA
yet" 的警告在 0.11.2 已被删掉 —— 既没列入支持也没明确禁止；bnb 的权重是
`Params4bit`，带 `.weight` 属性，所以它可能碰巧落进"未量化"那条分支而能用）。

想把它从"未知"变成"数据"，就在同一次开机里多跑一条：

```bash
python tools/check_vllm_rollout.py --config configs/grpo_fact_coverage_vGPU48.yaml \
    --quantization bitsandbytes
```

* 能跑 → 在配置里填 `rl.vllm.quantization: bitsandbytes`：底座显存从 ~12.4GB 降到 ~3.5GB，
  48GB 同卡共存宽裕很多（代价：bnb 在 vLLM 里强制 eager、且这条组合没被上游测过）。
* 跑不通（或结果明显乱） → 保持默认 bf16，必要时按上面的 AWQ 走。

## 4. 训练

```bash
SFZY_PROFILE=1 python scripts/train_grpo.py --config configs/grpo_fact_coverage_vGPU48.yaml
```

- `rl.rollout_backend: vllm`（配置里已经写好）。想临时回退：`--override rl.rollout_backend=hf`。
- 启动日志会打印后端与参数：`rollout 后端: vLLM（每步热交换 LoRA；base=... dtype=bfloat16 ...）`。
- 每步日志里多两行：`策略已同步给 vLLM：step=N adapter=...` 与
  `vLLM 生成完成：1 个 prompt × 4 条，用时 X s（Y 条/秒）`。
- `SFZY_PROFILE=1` 的分阶段表可直接对比：`rollout` 那一项应该从"几十秒"掉到"几秒"。

## 5. 实现要点（为什么这样接是安全的）

1. **只换"谁来写这段文本"**：vLLM 生成只产文本；`input_ids/labels` 与
   `old_logprobs` 仍由训练侧模型在同一批 token 上重算 → μ 内层循环、clip、KL 的
   语义完全不变（`GRPOTrainer.rollout()` 的后半段一行没改）。
2. **prompt 编码/截断与 HF 路径逐条一致**：同一个 `encode_prompt`、同样"保留尾部"。
3. **每步热交换**：把当前 LoRA 导出成 PEFT 目录（`src/sfzy/models/lora_export.py`，
   与离线三元组脚本共用同一份实现）+ 新的 `lora_int_id`；`max_loras=1` 时旧的按 LRU
   淘汰，磁盘上只交替保留 2 份。
4. **不依赖 vLLM 的新接口**：走 in-process `LoRARequest`，0.5.5 就有；
   服务版另有 `POST /v1/load_lora_adapter?load_inplace=true`（更适合异步训练）。
5. `hf` 后端默认不变；`vllm` 相关代码只在真的选择它时才 import —— 没装 vllm 的环境
   （CI、本地 CPU）照常跑全部测试。

## 6. 已知差异与坑

- **`trust_remote_code` 必须为 true**（`rl.vllm.trust_remote_code`，默认已是 true）。
  vLLM 自带 chatglm 的**模型实现**（registry 里 `ChatGLMModel → chatglm`），但**读
  config** 那一步仍走 transformers 的 `AutoConfig`；本地目录里有
  `configuration_chatglm.py` + `config.json` 的 `auto_map`，关闭时直接报：

  ```
  ValueError: The repository /path/to/chatglm3-6b contains custom code which must be
  executed to correctly load the model. ... Please pass the argument `trust_remote_code=True`
  ```

  放行的只是配置类；前向仍由 vLLM 自己的实现算（不会执行远程 modeling 代码）。
- **采样不跟随 torch 全局种子**：HF 路径受 `set_seed` 影响，vLLM 有自己的采样器。
  训练仍然可复现到"分布"，但具体样本不同。
- **LoRA × chunked prefill 互斥**（vLLM 源码里是硬报错）；不要手动去开 chunked prefill。
- **不要用 bitsandbytes 底座**：vLLM 的 LoRA 只对 AWQ/GPTQ 做过测试
  （源码：`if quantization not in ["awq","gptq"]: logger.warning("... not tested with LoRA yet")`）。
- 首次初始化 vLLM 要几十秒（加载权重 + 编译/捕获），这部分只在训练开始时付一次。
