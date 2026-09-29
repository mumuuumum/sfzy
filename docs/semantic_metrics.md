# 语义指标：为什么换、换成什么、怎么证明它有用

本文所有数字都是在本仓库 `data/triples/sft_val_shard{0,1}of2.jsonl`
（1340 条 SFT 输出）上实测的，复现命令写在每节末尾。

---

## 1. 官方 ROUGE 不适配这个任务：可控扰动实验

"指标不适配"必须有证据。设 1340 条 SFT 输出为基线，对每条施加**可控扰动**，
测指标相对基线的 Δ。分两类：

* **破坏型**：语义真的变了，指标**必须**明显掉分
* **等价型**：语义没变，指标**不该**掉分

`rouge_total` 是官方总分（0.2/0.4/0.4），`rouge_l` 是 ROUGE-L F1，
`embed_cos` 是 BAAI/bge-small-zh-v1.5 的句向量余弦（BERTScore 的同类替代，
但更强；BERTScore 只会更钝）。

| 扰动 | 类型 | n | ΔROUGE 总分 | ΔROUGE-L | Δ余弦* |
|---|---|---|---|---|---|
| 数字改错（金额抄错一位） | 破坏 | 916 | **-0.0027** | -0.0022 | -0.0004 |
| 漏掉一个含事实的句子 | 破坏 | 551 | -0.0440 | -0.0471 | -0.0090 |
| 判决结论反转（驳回↔支持） | 破坏 | 885 | **-0.0049** | -0.0043 | -0.0007 |
| 当事人张冠李戴 | 破坏 | 1274 | -0.0050 | -0.0043 | -0.0019 |
| 掺入一条不存在的事实（幻觉） | 破坏 | 1340 | -0.0162 | -0.0175 | -0.0032 |
| **调换两个相邻句子的顺序** | 等价 | 1339 | **-0.0137** | -0.0333 | -0.0011 |
| 金额单位改写（48000元→4.8万元） | 等价 | 304 | -0.0049 | -0.0042 | -0.0002 |

\* 余弦只在 300 条子集上跑（CPU），趋势一致。

### 读出来的三件事

**（1）ROUGE 惩罚"语序"的力度是惩罚"金额抄错"的 5.1 倍。**
调换两个相邻句子（信息一个字没变）掉 0.0137，把金额写错掉 0.0027。
方向完全反了 —— 这正是官方指标不能当 RL 主奖励的根本原因。

**（2）判决结论反转只掉 0.0049。** 而一个把"驳回"写成"支持"的摘要，
在法律意义上是废的。ROUGE-2 靠局部二元组，抓不住"结论"这种全局语义。

**（3）余弦相似度不是解法，它的动态范围塌了。**
数字改错只掉 0.0004 = 满分的 0.04%，而且**它的"破坏惩罚"只有"语序惩罚"的 36%**，
排序同样错乱。原因很清楚：摘要和参考本来就高度同源（同一份判决书），
余弦恒在 0.95~0.99 的顶部区间，所有信息挤在最后 5% 的量程里。
GRPO 组内归一化会把这点差异除以组内标准差 —— 放大的全是噪声。

> 所以"BERTScore 区分度不够"的准确表述是：
> **语义选择性（信号方向）和动态范围（信噪比）两个都不合格。**
> 后者才是不能用作 reward 的致命伤。

复现：
```bash
python tools/bench_metrics.py --input "data/triples/sft_val_shard*of2.jsonl"
python tools/bench_metrics.py --input "data/triples/sft_val_shard*of2.jsonl" \
    --limit 300 --encoder models/hf/models--BAAI--bge-small-zh-v1.5/snapshots/<hash>
```

---

## 2. 换什么：三层语义信号，各管一段

没有任何单一自动指标能同时做到"看得懂事实"和"分得清好坏的连续程度"。
实测已经把分工划清楚了，按层实现：

| 层 | 指标 | 覆盖什么 | 实测优势 | 不能做什么 |
|---|---|---|---|---|
| L1 事实层 | 规则事实 F1（金额/日期/编号，可扩展到要素） | 数字抄错、漏事实、幻觉 | 数字改错 **1.0→0.0**（精确率），判决级不可骗 | 对同义改写无感 |
| L2 语义层 | **LLM 生成式裁判**（`scripts/judge_score.py` + rubric） | 结论反转、要素完整性、表述规范 | 逐项核对，能识别"驳回↔支持" | 慢；有自噪声；可能被 reward hacking |
| L3 表层 | ROUGE-1/2/L | 与官方口径对齐、语序/短语 | 与比赛分数一致，对外可解释 | 上表已证明的盲区 |

余弦/BERTScore **不进 reward**，只作为监控项 —— 它的价值是"便宜且和 ROUGE
正相关度高"，一旦它和裁判分反向变动，说明裁判可能被 hack 了。

### L2 裁判要高区分度，靠这五条（都写在 `configs/judge_rubric_v1.yaml`）

1. **四维各 0-25，先写理由再给总分**（CoT 后再汇总，直接问总分只会得到模板分）
2. **每档写锚点描述**：没有锚点，"80 分"和"85 分"是随机的
3. **同一 group 共用同一上下文**，候选顺序打乱，去掉位置偏置
4. **测噪声地板**：`--samples 3` 报告裁判自噪声；扰动 Δ 必须 > 2σ 才算可分辨
5. **截断即丢弃**：裁判被 max_new_tokens 截断时最后一行是维度分，
   正则会把它当总分抄走 —— 脚本已加检测（这条是实现时踩到的真坑）

---

## 3. 怎么保证"至少语义指标突破，ROUGE 不退化"

### 3.1 一个必须知道的数学事实

GRPO 的 advantage 是组内归一化：`A_i = (r_i - mean_j r_j) / std_j r_j`。
**给组内所有候选同时加上/减去任何常数，梯度完全不变。**

推论：想用"奖励里减去 SFT 基线的分数"来防止退化，是无效的 ——
基线常数会被归一化抹掉。防退化只能用**非线性**手段：

* **硬门控**：候选在某项上不达标 → 该项直接 0 分（不可被其他项补偿）
* **组过滤**：整组的质量都不如 SFT 基线 → 这一组不产生梯度
* **KL 约束**：限制策略漂移（`grpo` 里已有 β，别调到 0）

### 3.2 落地到奖励结构

```
L0 门控（已有）  格式 / 长度 / 结果标记      —— 不通过 → 0 分
L1 事实层        fact_score（对称 F1）★改    —— 参考无事实时也'有梯度'
L2 语义层        judge_score / 100           —— 新增，主信号
L3 表层          ROUGE-L，权重压到 0.1~0.2   —— 锚住官方口径
```

四种模式做单变量对比（写进 `configs/grpo*.yaml` 的一个字段）：

| 模式 | 构成 | 说明 |
|---|---|---|
| `rouge_only` | ROUGE-L | 对照组，证明"不这么做" |
| `flat` | 加权求和 | 可补偿：可以用 ROUGE 换事实 |
| `gated` | 门控 + 加权 | 不可补偿（当前实现） |
| `gated_judge` | 上面 + L2 | 本次要验证的 |

### 3.3 语义指标提升要"可采信"，必须双裁判

reward 里放了裁判，再去汇报"裁判分涨了"＝循环论证。所以：

| 角色 | 模型 / rubric | 用途 |
|---|---|---|
| reward judge | Qwen2.5-7B-Instruct + rubric v1 | 进奖励，卡 1 |
| 汇报 judge | 换模型（GLM-4-9B / DeepSeek / GLM-4-Flash API）+ rubric v2 | 只用于对外汇报 |
| 人工校验 | 200 条分层抽样双盲 | 确认两个裁判都没跑偏 |
| 无争议指标 | 事实覆盖率/精确率（规则，不可被裁判带偏） | 一票否决 |

---

## 4. 验收判据（跑完 GRPO 先看这张表）

| 指标 | SFT 基线 | 目标 | 失败信号 |
|---|---|---|---|
| 事实覆盖率 | 0.3508 | > 0.45 | 不升 |
| 数字级漏掉 | 68.0% | < 55% | — |
| 事实精确率 | 0.9995 | > 0.99 | **下降 = 堆数字 hacking** |
| 官方 ROUGE 总分 | 0.5653 | ≥ 0.5653（持平即可） | 掉超过 0.01 |
| 汇报 judge 分（rubric v2） | 待测 | **≥ 基线 + 噪声地板** | 只有 rubric v1 涨 |
| 长度比中位数 | 1.05 | ≤ 1.15 | 涨 = 靠写长刷分 |
| 余弦监控 | 待测 | 与裁判同向 | 反向 = 裁判被 hack |

**必须分组报告**：参考含事实的 513 条 / 无事实的 827 条分开看。
总分持平但两组一涨一跌，是"平均掉了两个错误"的假象。

ROUGE 的可改进空间已单独算过：事实覆盖补到全中组水平 +0.0151，
长度拉回最佳组水平 +0.0114，**乐观上界 +0.0265**。所以对 ROUGE 的
合理期待是"微涨"，把"突破"押在语义指标上是对的判断。

---

## 5. 服务器 runbook

每步都有**闸门**：上一步的判据不过，就不要做下一步。GPU 时间比你的时间贵。

时间按实测的 ~16 秒/条（ChatGLM3 逐条生成，384 token × ~40ms）估。

---

### 步骤 1 · 生成 prompt 池（秒级）

```bash
python tools/select_rl_prompts.py --n 1000 --shuffle --out data/splits/rl_prompts.jsonl
```

**实验内容**：从 train 划分里筛出"参考摘要含金额"的样本当 RL 的 prompt 池。
事实项只在参考含事实的样本上有信号，随机抽 prompt 会让六成算力白花。

`--shuffle` 不能省：不加的话池子按事实密度降序，`--limit-prompts 300`
拿到的是**最难的那 300 条**，和全量不可比。

**判据**：输出里"含 money 类事实的 3497 条 = 32.6%"这两个数对得上即可。

---

### 步骤 2 · 裁判冒烟（3 分钟）

```bash
python scripts/judge_score.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --device cuda:1 --rubric-file configs/judge_rubric_v1.yaml \
    --input data/triples/sft_val_shard0of2.jsonl --limit 20 \
    --output data/judge/smoke.jsonl
```

**实验内容**：验证裁判管线（加载、模板、解析、写盘）通不通。
用已有的 SFT 三元组，不需要任何训练。

**闸门**（2026-09-28 实测记录：rubric v1 在这里**没通过**）：

| 检查 | 判据 | v1 的实测 |
|---|---|---|
| `失败 0 条` | 失败 = 截断或解析不出分，**不是 0 分** | ✓ 20/20 |
| 分数不能堆在满分 | 20 条里满分不超过一半 | ✗ **16 条满分** |
| 分布要和客观指标对得上 | ROUGE-L 最低的几条不该拿 100 | ✗ 最低 5 条里 3 条拿 100 |
| 可机械验证的项不许评错 | 长度比 1.73（>1.5）不该拿满分的"篇幅"分 | ✗ 拿了 25/25 |
| 人工读 2 条 `judge_raw` | 理由里要能看到**从候选里抄出来的句子** | ✗ 抄的是 rubric 的锚点原文 |

**v1 的失败模式：裁判在背锚点。** 它的"理由"栏写的是
"篇幅得当，长度与参考摘要相当（0.8~1.2 倍）"——和 rubric 里的 25 分锚点
一字不差，而那条摘要的实际长度比是 1.73。**它没在评，它在抄。**

这不是"分数偏高"，是**分数不携带信息**：组内 8 条全 100，GRPO 归一化后
语义项被抹成常数，A4 会退化成 A3，13 小时白烧。所以判据不是"均值好看"，
而是"分数和客观质量对得上"。

### 5.2.1 v1.1 的结果：有进步，但暴露两个新缺陷

v1.1 把分数改成"枚举要素 → 逐条抄录 → 按公式算"，18/20 有效（2 条被
`max_new_tokens=512` 截断，截断守卫正确地丢掉了它们，没有记成 0 分）：

| 检查 | v1 | v1.1 |
|---|---|---|
| 满分占比 | 16/20 = 80% | 8/18 = 44% |
| 取值个数 | 4 | 8 |
| 长度比 >1.5 的三条 | 两条满分 | 全部掉出满分（83 / 33 / 33） |
| corr(裁判, ROUGE-L) | +0.537（靠唯一的 40 分撑着） | +0.315 |

方向对了，但读 raw 发现两个缺陷：

**缺陷一：要素是从候选里挑的，不是从参考里拆的。** `f4752814` 的 6 条要素里
5 条直接来自候选，判 100 分；而参考里的"原被告均系第一顺序继承人"
"二分之一份额由四人平均继承"这两条关键法律推理，候选压根没写，
也就没被拆出来考。**裁判挑的都是候选写过的东西，于是必然全中。**

**缺陷二：伪造抄录。** 把裁判声称"抄自候选"的 83 条片段拿去候选里找：
23 条（28%）找不到，却能在**参考**里找到。它一边说抄的是候选，一边抄的是参考，
然后据此给分。这种错误不报错、分数看着正常。

```
参考：……原告有权解除与被告的租赁合同……
候选：原被告系租赁合同关系。原告诉求：解除合同。……判决解除……
裁判：要素2 [1分] 抄录："现被告在原告限定的合理期限内未向原告履行支付租金的义务"
                                                    ↑ 候选里没有这句
```

### 5.2.2 v1.2 的两处改动 + 自动质检

对应上面两个缺陷（见 `configs/judge_rubric_v1.yaml` 顶部注释）：

* **要素只从参考拆**，拆的时候不要看候选；每条要素要附上**参考里的原句片段**
* **抄录必须是候选里的连续片段、不超过 15 字、逐字复制**；抄参考会判为无效

更重要的是把质检**程序化**（`sfzy.eval.metrics.verify_judge_output`）——
人工读 2 条根本抓不住 28% 的伪造率。`judge_score.py` 现在会自动输出：

```
自动质检（抄录/要素是不是真的存在于它声称的来源里）
  抄录命中率 84.6%（18 条有抄录）   ← 声称抄自候选的句子有多少真在候选里
  命中率低于 80% 的 5 条
  ⚠ 裁判在编造依据：把参考摘要的句子当成候选的抄录了。
```

**这两个比率是"裁判能不能用"的硬指标**：引用都能编，后面的推理更不用谈。
v1.1 的要素出处率只有 36.9% —— 因为那一版允许裁判改写要素；
v1.2 要求附出处之后，这个数必须跳上去，否则说明裁判没照做。

重跑这条命令即可验证：

```bash
python scripts/judge_score.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --device cuda:1 --rubric-file configs/judge_rubric_v1.yaml \
    --input data/triples/sft_val_shard0of2.jsonl --limit 20 \
    --output data/judge/smoke_v11.jsonl

# 用客观指标对一遍：相关性该为正，且满分不该扎堆
python scripts/analyze_outputs.py --input "data/triples/sft_val_shard0of2.jsonl" \
    --judge-file data/judge/smoke_v11.jsonl | sed -n '/3.5/,/^$/p'
```

**验收线**（v1.2 起，四条一起看）：

1. `失败 0 条` —— 截断的样本按失败丢弃，**不要为凑数把它记成 0 分**
2. `抄录命中率 ≥ 80%` —— 低于这条说明裁判在编造依据，分数不可信
3. `要素出处率 ≥ 80%` —— 要素确实是参考里的东西，不是从候选里挑的
4. 满分不超过一半，且 `与 ROUGE-L 相关` 不要只靠一两个极端值撑着

前三条不过就别往下跑 —— 理由见 5.3。

### 5.3 组内排序裁判（已实现，点式三版失败后的主方案）

**实测记录**：点式打分试了三版，都没解决同一个结构性问题。

| 版本 | 改动 | 结果 |
|---|---|---|
| v1 | 四维各 25 分 + 锚点 | 16/20 满分，锚点被原样背进"理由"栏 |
| v1.1 | 枚举要素 + 逐条抄录 + 算分 | 满分降到 8/18，但要素是**从候选里挑的** |
| v1.2 | 要素只从参考拆 + 附出处 | 20/20 有效、分数散开，但**要素出处率 2.9%** |

根因：模型同时看到参考和候选时，会以更短、更像摘要的那份（**候选**）为锚点
组织阅读，于是它算的是 precision（候选→参考），而我们要的是 recall
（参考→候选）。实测表现为 `corr(裁判分, 长度比) = -0.35` —— 越短越占便宜。
连着两版用指令纠正都没用，因为这是在要求它报告一个我们**无法验证**的
内部过程。

如果点式打分（每条独立给 0-100）怎么调都堆在满分，换成**组内排序**：
把同一 prompt 的 G 条候选一次性给裁判，让它排序。

为什么这招对 GRPO 特别合适 —— **GRPO 只需要组内相对分数**：

    A_i = (r_i - mean_j r_j) / std_j r_j

绝对的 0-100 标定、宽松偏差、"80 分和 85 分有什么区别"，在组内归一化里
**全都会被减掉**。需要的只是"这 8 条谁比谁好"，而排序恰恰是 LLM 裁判最擅长、
最不容易失手的任务：

  * 不用标定（没有绝对尺度问题）
  * 是**比较**不是**打分**，宽松偏差自然消失
  * 成本更低：1 次调用/组，比点式的 G 次更便宜

代价与注意事项：
  * **位置偏置** —— 候选顺序要打乱；稳妥做法是正序倒序各排一次取平均
  * **并列** —— 允许并列，并列时用规则奖励打破
  * 只适用于 RL 的奖励；**对外汇报还得用点式**（跨 run 比较需要绝对分）

**实现**：`src/sfzy/eval/metrics.py` 的 `ListwiseRankScorer`，
`score_batch` 按 group_size 切块（rollout 输出的分组天然连续）。
配置：`semantic.backend: rank`，标准在 `configs/judge_rank_rubric.yaml`。

**上线前的验证**（不需要任何 GPU rollout）：

```bash
# 用可控扰动冒充"同一 prompt 的 5 条候选"：
#   base 和 e_order 是好的，d_digit/d_party/d_halluc 是坏的
python tools/check_ranker.py --input "data/triples/sft_val_shard0of2.jsonl" \
    --limit 20 --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --device cuda:1

# 本地先验证接线（用 ROUGE 假装排序）
python tools/check_ranker.py --limit 20 --stub
```

`--stub`（按 ROUGE 排序）的结果是**不通过**，正好说明这个测试有分辨力：

```
base        平均名次 0.25      d_digit  1.50
e_order     平均名次 3.45  ←   d_halluc 2.60
坏候选抢在好候选前面的组数: 19/20 = 95%   ✗ 不通过
```

ROUGE 把"只调换了句序"的候选排到最差（3.45 名），正是 §1 记录过的那个病。
判据是"好候选排名第一 ≥80% 且 坏候选抢先 ≤20%"，真裁判必须比这个 stub 好。

---

### 步骤 3 · 裁判区分度（约 25 分钟）

```bash
python tools/bench_metrics.py --input "data/triples/sft_val_shard*of2.jsonl" \
    --limit 200 --judge-cmd "python scripts/judge_score.py \
    --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --device cuda:1 \
    --rubric-file configs/judge_rubric_v1.yaml --stdin-jsonl"
```

**实验内容**：用可控扰动给裁判本身打分 —— 破坏型扰动（数字改错、结论反转、
幻觉注入）**必须**掉分，等价型扰动（调换句序、改写金额单位）**不该**掉分。
同时得到裁判的"区分度"比值，和 ROUGE 的 1.57 对比。

**闸门**（不达标就别跑 A4，换模型或改 rubric 重来）：

* 裁判对"数字改错"的降幅要**明显大于**对"句序调换"的降幅 ——
  ROUGE 在这两项上是 0.0027 vs 0.0137，方向完全反了，裁判必须把它纠过来
* 区分度（破坏/等价）要 > ROUGE 的 1.57

---

### 步骤 4 · 显存实测（5 分钟）

```bash
python scripts/check_memory.py --config configs/model_chatglm3_6b_bf16.yaml \
    --batch-size 8 --seq-len 2432
```

**实验内容**：GRPO 每个优化步要一次 forward+backward 处理
`prompts_per_step × group_size = 8` 条、每条最长 `2048 + 384` token 的序列。
先量出峰值，再决定 `prompts_per_step` 能不能加。

**闸门**：backward 之后余量 < 1 GiB 就不要加 `prompts_per_step`，
而且要注意一批里最长的那条还会再多吃一截。

---

### 步骤 5 · GRPO 最小闭环（约 20 分钟）

```bash
python scripts/train_grpo.py --config configs/grpo_cloud.yaml \
    --sft-adapter outputs/sft_chatglm3/best.pt --limit-prompts 8
```

**实验内容**：8 个 prompt × G=8 走完整链路（rollout → 裁判 → 奖励 →
advantage → 更新 → 存档）。这是唯一一次允许出错的运行。

**闸门**：

1. 日志里出现 `语义裁判: judge_local`（没出现说明裁判没接上）
2. `保留组 x/8` 的 x 不为 0（全 0 = 组内奖励没方差，GRPO 学不到东西）
3. `门控 0.0%`，不是几十个百分点（门控率高说明生成配置或 prompt 有问题，
   不是奖励设计的问题）
4. `judge_missing` 接近 0
5. `outputs/grpo_chatglm3/step_000008.pt` 真的写出来了

---

### 步骤 6 · A4 主方案正式跑（300 prompt，约 13 小时）

```bash
nohup python scripts/train_grpo.py --config configs/grpo_cloud.yaml \
    --sft-adapter outputs/sft_chatglm3/best.pt --limit-prompts 300 \
    > logs/grpo_a4.log 2>&1 &
```

**实验内容**：主方案 = 门控 + 事实 F1 + ROUGE + 生成式裁判
（权重 0.4 / 0.3 / 0.3）。**这一轮不跑 A1/A2/A3** —— 四臂全跑是 50+ 小时，
先拿一个结果；A4 相对 SFT 有效，再补 A3 做归因（"涨的是裁判还是事实项"）。

**训练中盯**（`tail -f logs/grpo_a4.log`）：

* `reward` 的均值该缓慢上升，`reward std` 不该塌到 0
* `ROUGE-L` 相对 SFT 的 0.5920 **不该持续下滑**（跌 0.01 就该警惕）
* `事实F1` 上升、`裁判` 上升、`len` 不涨 —— 三个一起动才是真变好

**中途可停**：`step_*.pt` 每 50 步存一次，随时可以拿来做评估。
不用等全部跑完 —— 如果第 100 步就已经看到 ROUGE 在跌，停下比跑完更省。

---

### 步骤 7 · 生成 RL 模型的三元组（val 子集 400 条，双卡约 1 小时）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_triples.py \
    --config configs/sft_cloud.yaml --adapter outputs/grpo_chatglm3/best.pt \
    --split val --limit 400 --shard 0/2 --batch-size 8 \
    --out data/triples/a4_val_shard0.jsonl > logs/gen_a4_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_triples.py \
    --config configs/sft_cloud.yaml --adapter outputs/grpo_chatglm3/best.pt \
    --split val --limit 400 --shard 1/2 --batch-size 8 \
    --out data/triples/a4_val_shard1.jsonl > logs/gen_a4_1.log 2>&1 &
```

**实验内容**：让 RL 后的模型在**同一批** val 样本上重新生成摘要。
`--limit 400` + `--shard` 的顺序是"先取前 400 条再分片"，所以两个分片合起来
正好是 SFT 三元组的前 400 条 —— `compare_runs.py` 按 id 取交集，天然对齐。

**注意**：`--adapter` 指向 RL 的 checkpoint（**不是** SFT 的），
`--quantize` 不要加（和训练精度保持一致）。

---

### 步骤 8 · 给 RL 的输出打两种分（各约 8 分钟）

```bash
# 8a. 汇报用裁判（rubric v2，和奖励里用的 v1 不同）
python scripts/judge_score.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --device cuda:1 --rubric-file configs/judge_rubric_v2.yaml \
    --input "data/triples/a4_val_shard*.jsonl" \
    --output data/judge/a4_val_v2.judge.jsonl

# 8b. 同一批 SFT 输出也重打一遍（同一把尺子才可比）
python scripts/judge_score.py --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --device cuda:1 --rubric-file configs/judge_rubric_v2.yaml \
    --input "data/triples/sft_val_shard*of2.jsonl" --limit 400 \
    --output data/judge/sft_val_v2.judge.jsonl
```

**实验内容**：**换一把尺子**给两次 run 打分。奖励里用的是 rubric v1，
如果汇报也用 v1，"语义涨了"只是"我们优化的函数涨了"。v2 的维度划分和
提问方式都不同（先列漏项/多项再打分），它是独立的第二个裁判。

**注意**：8a 和 8b 必须用同一份 rubric、同一个模型、同样的采样参数，
否则比的是两把尺子，不是两次 run。8b 加 `--limit 400` 是为了和 8a 对齐。

---

### 步骤 9 · 出结论（秒级）

```bash
python tools/compare_runs.py \
    --a "data/triples/sft_val_shard*of2.jsonl" --a-name SFT \
    --a-judge data/judge/sft_val_v2.judge.jsonl \
    --b "data/triples/a4_val_shard*.jsonl" --b-name A4 \
    --b-judge data/judge/a4_val_v2.judge.jsonl
```

**实验内容**：成对比较（只算两边都有的 id），并**分组**给出结论：
整体 / 参考含事实的 / 参考无事实的。最后按预注册判据逐条打勾。

看三件事：

1. **裁判分涨了、ROUGE 没跌** → 语义指标上的突破成立
2. **两组分开看**：总分持平而两组一涨一跌，是"平均掉了两个错误"的假象
3. **`改进占比`**：均值涨但占比接近 50% 说明是少数样本拉动的

---

### 5.x 两条不能违的纪律

**一、单进程，不要 torchrun。** 裁判和策略在同一个进程里分卡放
（策略 cuda:0、裁判 cuda:1）。`torchrun` 起两个进程会让两个都去占 cuda:1，
直接 OOM。而且两个 rank 用同一个 seed 会采出**完全相同**的样本，DDP 只会
重复算一遍梯度，没有任何加速。

**二、别急着改判据。** CRITERIA 写在 `tools/compare_runs.py` 里，
动手之前就定好了。跑完只读结果，不调阈值。
