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

### 抽取器与抽取 prompt 版本

事实一致性 = 抽取六要素 → 逐要素判定。**抽取这一步最容易出错**：实测 v1 的
抽取 prompt 会把日期挪用/改写（例如把"某日出具证明、另一日合同解除"合并成
"某日解除劳动合同"），裁判只看抽取结果，于是把逐字正确的摘要判成 0——人工
摘要也会中招。现在的抽取 prompt 是 **v3**：在 v2 的"逐字保真"之上又加了
"程序事实优先"——文本写明"被告未答辩/未到庭/缺席审理"时，`defendant_defenses`
必须写"未答辩"，不能被仲裁阶段/别处引用的"辩称"顶掉。版本号记录在
`sfzy/judge/prompts.py` 的 `EXTRACT_PROMPT_VERSION`，并写进每份
`*.reward.summary.json` 的 `extract_prompt_version`。**换过 prompt 的结果不要
和旧结果混着比**；同时要重跑 `judge_probe` / `judge_test` 重新出基线。

原文和摘要是**两套 prompt**：

- `EXTRACT_SYSTEM`（`kind="document"`）：抽判决书原文。长文、要素齐全，重点是
  防止"仲裁阶段引用的辩称 vs 本案未答辩"、日期挪用。
- `EXTRACT_SUMMARY_SYSTEM`（`kind="summary"`）：抽人工/候选摘要。短、大量要素
  为空，重点是"没写就留空，别把别的要素挪过来硬凑"，同时保持逐字摘录。

调用方用 `build_extract_messages(text, kind=...)` 选择；`document_elements` 走
原文 prompt，候选/参考摘要走摘要 prompt。换抽取 prompt 后不能和旧结果混着比。

**方案 A：抽取走 API，判定留本地。** 抽取是最难的一步，可以换更强的 API 模型；
判定要跑 `6要素 × 候选数` 次、还要受限解码的 `pmax`，放本地更划算。在配置里加：

```yaml
semantic:
  extract_api:
    base_url: https://api.deepseek.com/v1   # 或自建 vLLM 的 OpenAI 端口
    api_key_env: SFZY_JUDGE_API_KEY         # 密钥只从环境变量读
    model: deepseek-chat
    concurrency: 16                         # 线程池并发（API 是 I/O 密集）
    max_retries: 3
    timeout_s: 60
```

原来的 `EXTRACT_SYSTEM` 也**不能过度删减**：v1 的"省略不扣分/合理概括"判定
仍然由裁判负责，抽取只负责把原文/摘要里的要素如实抄下来。

### 判定时附上六要素上下文（`semantic.element_context`）

抽取的要素边界本来就有噪声：同一件事可能原文抽进 `court_facts`、摘要抽进
`legal_basis`，或者一边干脆没抽到。只把"要对比的两小段"给裁判，这种边界误差
就会被当成"编造/不一致"，产生假 0（实测：候选 court_facts 里的"不存在…事实/
无需承担责任"出自原文"本院认为"，但原文抽出的 court_facts 只有纯事实，于是被
判 0）。

打开 `semantic.element_context: true` 后，**逐要素给分的方式不变**，只是每次
判定的输入变成：

- 原文的**完整六要素**——高亮本次要对比的那一项，其余五项作辅助；
- 摘要**只给本次要判的那一个要素**（不给其它五项，避免引入无关信息）；
- 明确告诉裁判：只要摘要该项的陈述在原文**任意一项**里出现过，就按一致处理；
  但"当事人主张"和"法院认定"互换仍算不一致。

六要素基本就是整篇文书的信息量，所以这个 prompt 只比原来长一点（远小于把整篇
原文塞进去），不影响"逐要素判定"的初衷。要对比效果就把它关掉跑一份、开着重跑
一份，用 `judge_probe` / `judge_test` 看判别力有没有变差。

**输入长度**：val 原文最长约 1.2 万字，`max_input_tokens=4096/8192` 都会把长
文书的抽取 prompt 截断（头+尾保住了，中间的事实没了 → 抽取残缺、判 0）。现在
`max_input_tokens` 从 `semantic.max_input_tokens` 读（T4 配置里是 **16384**），
`extract_max_new_tokens` 提到 **1536**（v3 要求逐字保留、分条列举，输出更长）。
超预算的输入仍按头+尾截断，system 指令不丢。

如果显存够，也可以给抽取单独挂一个**本地**更强的模型（判定仍用原裁判）：

```bash
--extract-model <更强的抽取模型> --extract-quantization awq --extract-dtype float16
```

但这会在显存里**同时放两份权重** —— 单张 T4 放不下两个 7B，所以 T4 上一般
只用 v2 prompt，不传 `--extract-model`。

### 输入预算与截断（v2 之后的第二个坑）

抽取/判定的输入预算 `max_input_tokens` 默认从 4096 提到了 **8192**。原因：一篇
5102 字的判决书 + v2 的 system 指令会超过 4096，整段 prompt 被 tokenizer 截断后
**system 指令丢失**，抽取直接返回一段判决书原文、`解析方式: empty`、六要素全空，
后面每条要素都走兜底并被判 0。8192 覆盖 val 原文长度的 p99（约 6500 字）。

仍然超预算的输入,现在按**头 + 尾**截断（中间省略），system 指令和生成标记永远
保留 —— 不再用 `truncation_side` 截整段 prompt（从左切丢指令、从右切丢生成标记，
两头都会出问题）。实现见 `src/sfzy/utils/text_fit.py`。

代价是上下文变长、KV 占用上升。T4 上如果显存吃紧，把 `--chunk-size` 调小
（例如 8），否则 vLLM 会频繁 preempt/重算，吞吐反而更差。T4 配置里的
`extract_max_new_tokens` 也从 768 提到 1024（v2 要求逐字保留细节，输出更长）。

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
| `--extract-model` | 可选的独立抽取模型（更强/更大）；默认共用裁判模型。两份权重同占显存 |
| `--quantization` | `none` / `bitsandbytes` / `awq` / `gptq`。T4 必须显式指定一个 4-bit 方案 |
| `--load-format` | 默认 `auto`；选 `bitsandbytes` 时脚本会自动设成 `bitsandbytes`（vLLM 硬校验） |
| `--dtype` | T4 用 `float16`；4090/A100 可用 `bfloat16` |
| `--max-input-tokens` | 抽取/判定输入预算，默认取 `semantic.max_input_tokens`（T4 配置 16384）；超预算按头+尾截断 |
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

### 分项单独作为 reward

报告里还有一节「分项单独作为 reward」:把 `fact_consistency`、`element_coverage`
各自单独当作 reward,重跑上面两条结论(含门控 / 不含门控两种口径),
`*.reward.summary.json` 里对应 `per_term`。这一步用来回答"到底是哪个分项在
撑着结论"。

有两个必须注意的读法:

* **`element_coverage` 的人工臂恒为 1.0**(ref vs ref),所以它的结论 1 是
  **结构性成立**,不能当作判别力;它真正有意义的是结论 2(与 ROUGE 的相关性)。
* **`fact_consistency` 才是需要检验判别力的那一项**:如果它单独当 reward 时
  结论 1 不过,说明"人工 > 候选"其实只由覆盖率撑着,事实一致性这一路没有把
  人工排上去。

### 总判定与排查

#### 门控（决定 reward 是否为 0）

`gate` 是全局硬开关，命中任意一条整条 reward 直接 0，后面所有分项都不算。
当前（所有配置默认）有两大类：

1. **文本门控**：候选 < 60 字；候选/人工长度比不在 `[0.5, 1.5]`；以
   `以下是/摘要：/摘要:/本摘要/这是` 开头；不含 `判决如下/判令/驳回/本院认为/
   判决/裁定` 任一标记。对应 `gate_reason`：`too_short` / `length_too_short` /
   `length_too_long` / `prefix:…` / `no_result_marker`。
2. **事实一致性硬门控**（`gate.fact_consistency_min_raw: 1`）：六要素的
   **原始分（0-4）** 只要有任意一个 `< 1`（即得 0 分），整条 reward 直接 0，
   覆盖率 / ROUGE 都不再参与；六项都 ≥ 1 时，`fact_consistency` 才按配置权重
   进入混合奖励。把该值设成 `0` 就关闭这条规则。对应 `gate_reason`：
   `fact_element_raw<1`。

注意这是**全局**门控，离线验证时人工臂（candidate=reference）也会被同一条
规则拦下；`*.reward.jsonl` 的 `gated`/`gate_reason` 和报告里的「门控原因」
分布能看出各臂被拦在哪一条。

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
