# 用 vLLM 裁判给 val 三元组打奖励分（验证 reward 设计）

`scripts/score_reward_vllm.py` 回答一个问题：**GRPO 的 reward 会不会把
人工摘要排在模型候选之上？** 它在验证集上分别给两臂打分：

```
候选臂 candidate：candidate=output，  reference=reference
人工臂 human    ：candidate=reference，reference=reference
```

奖励口径完全走训练那条路径（`RewardSpec.from_config` + `compute_reward`）：

```
门控（长度 / 前缀 / 结果标记）→ ROUGE-L / fact_consistency / element_coverage
→ 按归一化权重加权求和
```

唯一的差别是裁判从 HF 换成 vLLM，用来把耗时压下来。六要素抽取、重试、空字段
规则、覆盖率组装、0-4 受限解码全部复用 `sfzy/judge`，不另写一套。

## 一、先装独立的 vLLM 环境

和 `docs/vllm_triples.md` 完全一樣：vLLM 和 transformers 强绑定，别装进
ChatGLM3 的那个 SFT 环境。推荐

```bash
pip install "vllm==0.6.6" "transformers==4.46.3"
```

## 二、T4 的显存与 4-bit（必须看）

Qwen2.5-7B 的 fp16 权重约 15GB，单张 T4（16GB）装不下。**vLLM 是能加载
4-bit 模型的**，只是它不认识配置里的 `semantic.load_in_4bit` —— 那是
transformers / bitsandbytes 的加载期开关，vLLM 的 `LLM(...)` 不读它；不显式
传 `--quantization`，它就会按 `--dtype` 加载 fp16。T4 上有两条路（都配
`--dtype float16`）。

**A. bitsandbytes NF4（和项目里 GRPO 裁判同一套量化语义）**

```bash
# vLLM 0.6.x ~ 0.10.x 内置 bnb，装依赖即可
pip install "bitsandbytes>=0.45.0"
# 更新的 vLLM（2026 那条线）把 bnb 拆成了插件：
# pip install vllm-bnb-plugin
```

```bash
--model Qwen/Qwen2.5-7B-Instruct --quantization bitsandbytes
```

vLLM 的 bnb 支持 **in-flight 量化**（加载时把 fp16 权重现量化成 NF4）和
预量化 checkpoint 两种，Turing(sm_75)/T4 在官方支持列表里。代价是 bnb 走
"反量化再算"的通用内核，吞吐不如 AWQ/GPTQ/Marlin。

> 脚本会在 `--quantization bitsandbytes` 时自动补上 `--load-format bitsandbytes`。
> 这不是可选项：vLLM 的 `create_engine_config` 有一条硬校验，bnb 量化必须配
> `bitsandbytes` load format，否则直接抛
> `ValueError: BitsAndBytes quantization ... only support 'bitsandbytes' load format`。
> 这个 loader 对没有 `quant_state` 的 fp16 权重做 `quantize_4bit`，也就是
> in-flight 量化。

**B. AWQ / GPTQ 4-bit checkpoint（serving 内核更快，推荐）**

```bash
--model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq
```

脚本会在 `semantic.load_in_4bit=True` 而 `--quantization none` 时打印警告，
但不会替你下载模型或装依赖 —— 提前把权重放到本地、把 bitsandbytes / 插件
装好。

> 因为要"一个分片一张卡"，这里**不要用 tensor parallel**：两个进程各占一张
> T4，各跑一个分片。`semantic.device` 会被忽略，卡由 `CUDA_VISIBLE_DEVICES`
> 决定（vLLM 只看得到可见设备）。

## 三、跑法（2×T4，一个分片一张卡）

```bash
# 卡 0：shard0（例：AWQ；换成 --quantization bitsandbytes 即是上面的 A 方案）
CUDA_VISIBLE_DEVICES=0 python scripts/score_reward_vllm.py \
    --config configs/grpo_fact_coverage_t4.yaml \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --dtype float16 \
    --input data/triples/sft_val_shard0of2.jsonl

# 卡 1：shard1
CUDA_VISIBLE_DEVICES=1 python scripts/score_reward_vllm.py \
    --config configs/grpo_fact_coverage_t4.yaml \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --dtype float16 \
    --input data/triples/sft_val_shard1of2.jsonl
```


```bash
# 卡 0：shard0（例：BNB；换成 --quantization bitsandbytes ）
# pip install vllm-bnb-plugin

CUDA_VISIBLE_DEVICES=0 python scripts/score_reward_vllm.py \
    --config configs/grpo_fact_coverage_t4.yaml \
    --model Qwen/Qwen2.5-7B-Instruct --quantization bitsandbytes --dtype float16 \
    --input data/triples/sft_val_shard0of2.jsonl

# 卡 1：shard1
CUDA_VISIBLE_DEVICES=1 python scripts/score_reward_vllm.py \
    --config configs/grpo_fact_coverage_t4.yaml \
    --model Qwen/Qwen2.5-7B-Instruct --quantization bitsandbytes --dtype float16 \
    --input data/triples/sft_val_shard1of2.jsonl
```

等两个进程都结束后，产物在 `data/judge/`：

```
data/judge/sft_val_shard0of2.reward.jsonl          每行一条记录的一臂奖励 / ROUGE
data/judge/sft_val_shard0of2.reward.summary.json   两条结论的检验结果 + 分项统计
data/judge/sft_val_shard0of2.reward.report.md      人读报告（结论 + 支撑数据）
data/judge/sft_val_shard1of2.reward.jsonl
data/judge/sft_val_shard1of2.reward.summary.json
data/judge/sft_val_shard1of2.reward.report.md
```

两个分片跑完后，把逐条结果合成一份**全量**总报告（纯 CPU，几秒，不加载裁判）：

```bash
python scripts/score_reward_vllm.py --merge \
    data/judge/sft_val_shard0of2.reward.jsonl \
    data/judge/sft_val_shard1of2.reward.jsonl
# -> data/judge/merged.reward.summary.json / merged.reward.report.md / merged.reward.jsonl
```

最终对外汇报用这份 `merged.reward.report.md`，各分片的报告留作排查。

### 常用参数

| 参数 | 作用 |
|---|---|
| `--config` | 读 `rl.reward`（奖励口径）和 `semantic`（裁判抽取预算等），默认 `configs/grpo_fact_coverage_t4.yaml` |
| `--model` / `--tokenizer` | 覆盖 `semantic.model`；AWQ 目录缺 tokenizer 文件时单独指 |
| `--quantization` | `none` / `bitsandbytes` / `awq` / `gptq`。T4 必须显式指定一个 4-bit 方案 |
| `--load-format` | 默认 `auto`；选 `bitsandbytes` 时脚本会自动设成 `bitsandbytes`（vLLM 硬校验） |
| `--dtype` | T4 用 `float16`；4090/A100 可用 `bfloat16` |
| `--chunk-size` | 一批几条记录（默认 16）。显存紧调小，吞吐优先调大 |
| `--limit` / `--no-resume` | 冒烟 / 强制重跑。默认按 id 断点续跑 |
| `--candidate-key` / `--reference-key` | 字段名，默认 `output` / `reference` |

先冒烟 8 条确认接线（几十秒）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/score_reward_vllm.py \
    --config configs/grpo_fact_coverage_t4.yaml \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --dtype float16 \
    --input data/triples/sft_val_shard0of2.jsonl --limit 8 --no-resume
```

## 四、怎么读结果

脚本会验证并输出**两条结论**。`*.reward.report.md` 是给人看的版本，
`*.reward.summary.json` 里的 `conclusions` 是同一份数据的机器可读版。

### 结论 1：候选摘要的奖励一般比人工摘要小

同一篇文书上做配对比较（人工奖励 − 候选奖励），判据是：

- 配对差均值 > 0，且人工胜多于负；
- **符号检验**（精确二项，双侧）p < 0.05 → 判定成立。

用符号检验而不是只看均值，是因为"一般更小"本质是个方向性命题；t 检验也一起
报了，但当样本量小（`--limit`）时它的正态近似偏乐观，不作为判定依据。

```json
{
  "candidate_reward_lt_human_reward": {
    "mean_delta": 0.13, "win_rate": 0.94,
    "sign_test_p": 1.2e-40, "p_value": 3e-42,
    "pass": true,
    "verdict": "✓ 成立：候选摘要奖励显著低于人工摘要"
  }
}
```

### 结论 2：候选的 reward 与 ROUGE 正相关

对每条记录取候选摘要的 `reward` 和它相对人工摘要的 `rouge-l-f`，跨文书算
Pearson 和 Spearman：

- Spearman ρ > 0 且 p < 0.05 → 判定成立。

主口径是**真正进训练的 gated reward**；同时报告 `reward_ungated`（不含门控）
的口径，用来判断相关性是不是被门控的一堆 0 分"制造"出来的。ROUGE 是官方
评测口径，reward 若和它反向，优化目标就和最终指标打架。

```json
{
  "reward_rouge_positive_correlation": {
    "primary": "reward(含门控) vs rouge-l-f",
    "gated":   {"n": 670, "pearson": 0.41, "spearman": 0.44, "spearman_p": 1e-30},
    "ungated": {"n": 670, "pearson": 0.52, "spearman": 0.55, "spearman_p": 1e-50},
    "pass": true,
    "verdict": "✓ 成立：候选 reward 与 ROUGE-L 显著正相关"
  }
}
```

### 总判定与排查

- 两条都成立 → `verdict` 为「✓ 两条结论都成立」，可以接 GRPO；
- 只有一条成立 → 去看报告里的分项：结论 1 弱通常是
  `mean_fact_consistency` / `mean_element_coverage` 两臂差距太小，或
  `gate_rate` 误伤人工摘要（人工摘要没有"判决如下"这类结果标记时会这样）；
  结论 2 弱通常是 reward 被门控的 0 分主导，或裁判分与 ROUGE 系统性相悖；
- 两条都不成立 → reward 没有判别力，接 GRPO 只会得到噪声梯度。

逐条 `*.reward.jsonl` 里每行是 `arm=candidate|human` 的一条，字段包括
`reward`（含门控）、`reward_ungated`（不含门控）、`terms`（各分项）、
`rouge`（rouge-1/2/l 的 p/r/f 与 overall）、`rouge_l`、`fact_raw`（六要素
0-4 原始分）、`element_coverage`、`gate_reason`。要定位"是哪一篇文书、哪个
要素判反了"，直接看这一列。

## 五、和 HF 版的关系

`tools/score_fact_judge.py` 是 HF 版：慢，但只算事实一致性、且一次只打一份
输入。本脚本是 vLLM 版：更快、一次打两臂、并且把 `rouge_l` / `element_coverage`
和门控一起算成**训练同口径的完整奖励**。两者都用同一份提示词和同一套聚合，
数值上只差 vLLM 与 HF 算子的细微差异 —— 不要拿两份数字做逐位对比。
