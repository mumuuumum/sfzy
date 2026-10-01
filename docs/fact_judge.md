# 六要素事实一致性 Judge

代码在 `src/sfzy/judge/`，测试程序 `scripts/judge_test.py`，
模型筛选工具 `scripts/judge_probe.py`。

---

## 1. 它和之前那套点式裁判的关系

之前那条线（`sfzy.eval.metrics` + `configs/judge_rubric_v1.yaml`）做的是
**点式打分**：把原文和候选一起给裁判，让它给一个 0-100 的综合分。
试了三版都栽在同一件事上 —— 裁判同时看到两份文本时，会以更短、更像摘要的
那份（候选）为锚点组织阅读，于是它算的是 precision（候选写的有多少在原文里），
而我们要的是 recall。实测 `corr(裁判分, 长度比) = -0.35`，越短越占便宜。

这一版换了三个地方：

| | 点式（旧） | 六要素（本版） |
|---|---|---|
| 输入 | 原文全篇 + 候选全篇 | 两小段要素 |
| 判据 | 综合印象分 | "候选说的能否被原文支持" |
| 打分 | 生成 0-100 再解析 | **受限解码**，五档 token 取 argmax |
| 对省略的态度 | 未明确 → 短摘要占便宜 | 明确写死：省略不算错 |

---

## 2. 模型容量是硬约束：0.5B 不能用

需求要求"Qwen 0.5B 级别"。用 10 条要素级探针实测（`scripts/judge_probe.py`）：

```
探针            期望   0.5B得分   0.5B的pmax
P1_完全相同      ≥3        0        0.97
P2_只省略        ≥3        0        0.98
P3_合理概括      ≥3        0        0.68
P4_换词改写      ≥3        0        0.75
P5_金额写错      ≤1        0        0.72
P6_主体颠倒      ≤1        0        0.86
P7_结果反转      ≤1        0        0.86
P8_编造事实      ≤1        0        0.68
P9_辩称当查明    ≤1        0        0.80
P10_案由串门     ≤1        0        0.99
```

**十条全是 0，而且 pmax 高达 0.97~0.99。** 它不是"拿不准"，是**自信地
输出常量**。加少样本示例只会把常量从 4 挪到 3（见下面的对照），不会产生判别力。

| 探针 | 期望 | 规范 prompt | 加少样本 prompt | 简化 prompt |
|---|---|---|---|---|
| 完全相同 | 4 | 4 | 3 | 1 |
| 金额写错 | 1 | 4 | 3 | 1 |
| 结果反转 | 0 | 4 | 3 | 1 |

三套提示词下都是**常量**。这是容量问题，不是提示词问题。

另外发现一个小坑：规范 prompt 结尾是 `0 / 1 / 2 / 3 / 或 / 4` 的枚举，
会明显影响输出（同一批输入，改掉这个结尾后答案从全 4 变成全 3）。
所以判定一律用**受限解码**：只在这五个 token 上取 argmax，
不存在"输出不合法"，也不受枚举写法影响。

---

## 3. 用什么模型当裁判：三个选项

| 方案 | 额外显存 | 每步额外耗时 | 独立性 |
|---|---|---|---|
| Qwen2.5-0.5B | 1 GB | 快 | 好，但**判别力不足，已否决** |
| **ChatGLM3-6B 关 adapter** | **0 GB**（同一份权重） | 生成逐条，约 +90% | 差 —— 与策略同源 |
| Qwen2.5-7B（卡 1） | 15 GB（卡 1 本来空着） | 可与策略并行 | 好 |

### ChatGLM3 关 adapter 当裁判

LoRA 是可插拔的，`model.disable_adapter()` 拿到的就是**冻结的底座** ——
训练器里已经在用这一招取 KL 的参考策略，所以**不需要额外加载任何权重**。

三条注意事项：

1. **同源自评。** 策略是底座+LoRA，裁判是底座。底座分不清原告被告的地方，
   裁判也分不清 —— 奖励会继承策略的盲区，而策略正好可以往盲区里钻。
   这是这个方案唯一的硬伤，用 Qwen 当裁判就没这个问题。
2. **省显存但不省时间。** 裁判要做 G 次候选六要素提取（是生成，不是前向）。
   ChatGLM3 逐条生成约 15 秒/条，G=8 就是每步多 ~120 秒，训练时间大约翻倍。
   判定那 48 次是纯前向，很快。
3. 走项目自己的兼容层（`models/compat.py`）：**生成逐条走 `stream_generate`，
   打分走无 padding 的单条前向** —— 它的 `get_masks` 假设输入没有 padding，
   批量会直接崩。这条路径有测试钉住（`tests/test_judge.py` 的 `_NativeModel`）。

### 建议

**奖励用 ChatGLM3 底座，汇报用 Qwen2.5-7B。** 省下卡 1 的 15GB，
而"语义真的提升了吗"这个对外结论由独立模型给出 —— 这正是双裁判设计的用途。
前提是 ChatGLM3 能过探针。

---

## 4. 怎么用

```bash
# ① 先筛模型（2 分钟，10 条探针，不接 GRPO）
python scripts/judge_probe.py --model-config configs/model_chatglm3_6b_bf16.yaml --device cuda
python scripts/judge_probe.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --device cuda:1
# 对照：微调过的模型判得是不是比底座差
python scripts/judge_probe.py --model-config configs/model_chatglm3_6b_bf16.yaml \
    --adapter outputs/sft_chatglm3/best.pt --device cuda

# ② 过了再做全流程测试（21 条人工案例，含六要素提取）
python scripts/judge_test.py --model-config configs/model_chatglm3_6b_bf16.yaml \
    --cases data/judge/fact_cases.jsonl

# ③ 都过了才接 GRPOTrainer
```

**判据**：好的四条 ≥3、坏的六条 ≤1、两组均值差 ≥2.0。任何一条不过就不要接进奖励。

接进训练时唯一要调的入口是 `judge_candidates(document, candidates)`，
文档要素有缓存（按文本 sha1），一次提取、G 个候选共用。

---

## 5. 实测踩到的两个坑（都是"不报错、只是奖励全错"）

**坑一：提取被截断 → 文档要素全空 → 奖励恒为 0。**
2600 字的文书在 `max_new_tokens=512` 下被截断，六个要素只捞到一个
`case_type`；于是每条候选的每个要素都走"原文没有、候选写了"的分支，
裁判全部判 0。现在：解析降级会自动用双倍预算重试，文档要素少于 2 项直接抛
`ExtractionFailure`。

**坑二：字段值是数组。** 0.5B 不会老实输出 `"key": "字符串"`，
它爱写成 `["要求偿还本金", "要求承担保证责任"]`。不接住的话降级路径
一个字都匹配不到，六个要素全空（等于坑一）。

两个坑都是跑一次真实数据才暴露的，单测和纸面推演都看不出来。

---

## 6. 接进 GRPO 奖励（事实一致性部分）

上面几节把 Judge 做出来了，这一节说它**怎么进 GRPO 的奖励**。链路只有三跳：

```
GRPOTrainer.step()
  rollout → G 条候选
  score_batch(items)              # FactConsistencyScorer：按文书分组
    → judge_candidates(原文, 候选×G)
    → {weighted_reward ∈ [0,1]}   # 每候选一个事实一致性分
  compute_rewards(..., mode=fact_judge)
    → total = 0.3·ROUGE-L + 0.7·fact_judge   （门控通过后）
```

### 三个新增的接口

| 位置 | 作用 |
|---|---|
| `sfzy/judge/scorer.py` `FactConsistencyScorer` | 把扁平的 GRPO items 按 `source` 分组，逐组调 `judge_candidates`，把 `weighted_reward` 摊回原位置 |
| `eval/metrics.py` `build_scorer(backend="fact")` | 配置层入口：`semantic.backend=fact` 就构造这个裁判 |
| `rl/reward.py` `mode="fact_judge"` | 新奖励模式：事实项**完全来自 Judge**，不跑任何金额/日期/法条规则匹配 |

### 为什么单开一个奖励模式，而不是复用它

默认的 `gated` / `gated_judge` 里，"事实项"是**规则口径的对称 F1**
（金额/日期/编号的正则匹配）。需求明确要求事实一致性**全部交给 Judge**，
不写额外 checker。所以 `fact_judge` 模式的第一件事就是把 `fact_kinds` 清空 ——
规则提取根本不跑，`n_ref_facts` 恒为 0，事实项只认 `weighted_reward`。

顺带钉住一个量纲坑：`gated_judge` 的裁判分是 0-100，要除 `semantic_scale`；
而六要素 Judge 返回的 `weighted_reward` **本来就是 [0,1]**。配错了不报错，
只是事实项缩水 100 倍。`fact_judge` 模式因此**不做量纲换算**。

### 怎么跑

```bash
# 先过验收（不接 GRPO）
python scripts/judge_probe.py --model models/Qwen2.5-1.5B-Instruct --device cuda:1
python scripts/judge_test.py  --model models/Qwen2.5-1.5B-Instruct --device cuda:1 \
    --cases data/judge/fact_cases.jsonl

# 三条判据全过再接 GRPO
python scripts/train_grpo.py --config configs/grpo_fact_judge.yaml \
    --sft-adapter outputs/sft_chatglm3/best.pt
```

`configs/grpo_fact_judge.yaml` 只覆盖两处（`rl.reward.mode=fact_judge`、
`semantic.backend=fact`），其余全部继承 `grpo_cloud.yaml`。

### 4-bit 与设备（4090 / T4 的差异）

`configs/model_qwen25_7b.yaml` 默认开着 `load_in_4bit`。这带来一条硬约束：

> **bitsandbytes 4/8-bit 量化模型不能调 `.to()`。** transformers 会直接抛
> `ValueError: .to() is not supported for 4/8-bit bitsandbytes models`，
> 因为量化权重是 int8/uint8 + 分片 metadata，搬动会破坏它。设备位置只能在
> `from_pretrained` 时用 `device_map` 定死。

所以 `judge_test.py` / `judge_probe.py`（以及训练、三元组、基准脚本）现在统一
走 `sfzy.models.loader.load_inference_model`：它把 `--device` 翻译成
`{"": 卡号}` 再加载，加载完一个字都不搬。

```bash
# 4090（24GB）：7B bf16 装得下，绕开 bitsandbytes 更省心
python scripts/judge_test.py --model-config configs/model_qwen25_7b.yaml \
    --device cuda:1 --no-4bit

# T4（16GB）：必须留在 4-bit
python scripts/judge_test.py --model-config configs/model_qwen25_7b.yaml \
    --device cuda:0 --dtype float16
```

这条差异正是"T4 能跑、4090 崩"的来源：两边跑的不是同一个配置（一个没量化、
一个量化），而旧的 `device_map in (None, "", "none")` 判断不看"是否量化",
在 device_map 写成 null 的配置上会把 `.to()` 放行。

### 关于 "Special tokens have been added in the vocabulary"

加载 Qwen2.5 的 tokenizer 时 transformers 会打这一行（INFO 级别）：

```
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
```

**它无害，也不需要做任何事。** 触发条件在 `tokenization_utils_base.py`：
tokenizer 的 added token id（151665）超过了基础 BPE 词表大小（`vocab_size=151643`）。
Qwen2.5 的 `<|im_start|>`、`<|end_of_text|>` 这些 special token 的 id 本来就排在
BPE 词表之后，是 tokenizer 文件里的既有事实 —— 加载前后 `len(tokenizer)` 完全一致，
没有触发 `resize_token_embeddings`，模型的嵌入矩阵本来就有这么多行。

T4 上看不到，只是因为那次日志级别不是 INFO（或跑的是别的 tokenizer）。两个脚本现在
在启动时把 transformers 的日志压回默认的 WARNING，这条 INFO 不再混进判分输出，
真正的 WARNING/ERROR 照常打印。

### 训练日志里能看到的（需求第十二节）

每个优化步打印一行：

```
└ 事实一致性 fact_reward 0.9312 | 结果分 1.000 | 最低分 0.750 | 档位 0:1% 1:2% 2:6% 3:14% 4:77%
```

`mean_fact_reward`、六个 `mean_<要素>_score`、`mean_min_element_score`、
`ratio_score_0..4` 全部由 `FactConsistencyScorer.summarize_last()` 汇总，
trainer 只要 `hasattr(scorer, "summarize_last")` 就合并进 metrics ——
**trainer 不需要认识六要素**，耦合面只有一个方法名。

### 失败与降级

某一篇文书的判定整体失败（提取器抛 `ExtractionFailure`、显存抖动）时，
`FactConsistencyScorer` 只把这一组标成 `None`，由 trainer 的
`fill_semantic_gaps` 用**组内均值**补上，不中断训练；缺失比例记在
`judge_missing` 里。整组失败意味着这一组的事实项是常数，GRPO 自然忽略它，
仍能靠 ROUGE 学习。超过 10% 就该去查 `extract_max_new_tokens` 和显存，
而不是接着跑。

### 门控与 fact_judge 的关系

`fact_judge` 仍然过门控（长度区间、禁止前缀、结果标记）。这是项目级设计：
格式不达标的输出直接 0，切断"牺牲格式换分数"的路。但要注意**门控是加在
总分上的，不是加在 Judge 上的** —— 一组采样若全长太短，会被整组拦下、
advantage 恒 0。真遇到这种情况，先放宽 `gate.length_ratio_range`，
而不是去动 Judge。

---

## 7. 双卡分居部署（2× RTX 4090 24GB）

```
RTX 4090 #0（24GB）                     RTX 4090 #1（24GB）
├── ChatGLM3-6B Base：4bit NF4，冻结      └── Qwen2.5-7B-Instruct Judge
├── SFT LoRA：可训练                        ├── 4bit NF4
├── GRPO optimizer / activations            ├── 完全冻结、inference_mode
└── rollout / candidate generation          └── 批量评价 candidate
```

**单进程**跑，两张卡各司其职 —— 不要用 `torchrun`（DDP 会给每张卡各放一份
策略，卡 1 就装不下裁判了）。

### 配置：一份文件搞定

`configs/grpo_fact_judge.yaml` 就是这套布局，关键四项：

```yaml
model_config: configs/model_chatglm3_6b_nf4.yaml   # 策略：4bit 钉卡 0
rl:
  quantize: true                                   # 让"策略 4bit"写进配置
  reward: { mode: fact_judge }
semantic:
  backend: fact
  model: /root/autodl-tmp/models/Qwen2.5-7B-Instruct
  device: cuda:1                                   # ★ 裁判钉卡 1
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
```

策略侧底座见 `configs/model_chatglm3_6b_nf4.yaml`：`load_in_4bit: true` +
`device_map: {"": 0}`。**`device_map` 必须显式写**，因为 4bit 模型加载后
不能 `.to()`，位置只能在加载时定死；写 `"auto"` 会把 6B 摊到两张卡。

启动：

```bash
python scripts/train_grpo.py --config configs/grpo_fact_judge.yaml \
    --sft-adapter outputs/sft_chatglm3/step_000336.pt
```

### 两个开关的语义（易踩）

| 开关 | 作用 | 坑 |
|---|---|---|
| `rl.quantize: true` / `--quantize` | 策略底座走 4bit | 配置里写了 `load_in_4bit: true` 但没开这个，脚本会**静默降级成 bf16**（原设计是为了和 bf16 SFT 对齐），只在日志的"策略精度"行看得出来 |
| `semantic.load_in_4bit: true` | 裁判走 4bit | 走 `load_inference_model`：`device=cuda:1` 被翻译成 `device_map={"":1}`，加载后不搬 |

启动日志里会有两行确认，先看它们再去泡茶：

```
策略精度: 4-bit nf4（量化=True，device_map={'': 0}）
语义裁判: judge_fact（backend=fact）
```

### 显存与调参

* 卡 0 的大头不是权重（4bit 约 3.5GB），是**词表投影**：ChatGLM3 的
  vocab=65024，一次前向里 fp16 logits + fp32 副本 + 梯度约 `B×L×0.9MB`。
  G=8、L≈2432 时这一项就 17GB 左右，是最可能 OOM 的地方。先跑
  `python scripts/check_memory.py` 量一遍，撑不住就按顺序降
  `rl.max_new_tokens` → `rl.max_length` → `rl.group_size`。
* 卡 1 的裁判 7B 4bit 约 5~6GB，48 个 pair 组 batch 前向也很快；
  真紧张就把 `semantic.max_batch_size` 从 8 调到 4。

### 上线前的两步验收（都在卡 1，不接 GRPO）

```bash
python scripts/judge_probe.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --device cuda:1 --load-in-4bit
python scripts/judge_test.py  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --device cuda:1 --load-in-4bit --cases data/judge/fact_cases.jsonl
```

三条判据全过（好的四条 ≥3、坏的六条 ≤1、每个文档内 A/B 都高于 C~G）
才接 GRPO；否则奖励只是噪声，跑到一半才发现就晚了。
