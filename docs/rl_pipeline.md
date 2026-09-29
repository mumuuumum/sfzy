# SFT → 观察 → 奖励 → GRPO：跑法与判据

这份文档只讲**顺序和判据**：每一步跑什么、产出什么、看到什么才允许进下一步。
方法和结论写在各自的代码注释里（`rl/reward.py`、`sft/infer.py`）。

> 语义指标（为什么不用 ROUGE 当主奖励、裁判怎么搭、验收判据）
> 见 `docs/semantic_metrics.md`。本文第 3 步的奖励设计以那份文档为准。

---

## 为什么必须"先观察再上 RL"

2025-03 那版的教训：SFT 之后直接上 DPO，效果不好，但**说不清是哪一环的问题** ——
是奖励信号不对、样本不够，还是 SFT 本身就没学会写摘要？

所以这次把"观察"单独做成一步：先用 SFT 模型生成三元组
`(原文, 参考摘要, 模型输出)`，在真实输出上算奖励，**确认奖励的方向和 ROUGE
一致、且确实能区分好坏之后**，再花 GPU 配额去跑 GRPO。

不然你会在一个设计错了的奖励上烧掉一整周配额。

---

## 第 0 步：SFT（见 docs/kaggle_sft.md）

产物是 `outputs/sft_chatglm3/best.pt`（或某个 `step_*.pt`）。
**记住文件名**，下一步要用。

---

## 第 1 步：生成三元组

```python
# 先跑 4 条确认接线（本地 0.5B 就能验，几十秒）
!python scripts/generate_triples.py --config configs/sft_local.yaml \
    --split val --limit 4 --batch-size 2 --max-new-tokens 24

# Kaggle 上跑正式的：val + test 两个集合
!python scripts/generate_triples.py --config configs/sft_kaggle.yaml \
    --adapter outputs/sft_chatglm3/best.pt --split val --batch-size 8
!python scripts/generate_triples.py --config configs/sft_kaggle.yaml \
    --adapter outputs/sft_chatglm3/best.pt --split test --batch-size 8
```

产出 `data/triples/sft_{val,test}.jsonl`，每行：

```json
{"id": "...", "source": "文书全文", "reference": "参考摘要", "output": "模型生成的摘要"}
```

**三个必须注意的点：**

1. **断点续跑是内建的**：文件里已有的 id 会被跳过，会话被掐之后重跑同一条命令即可。
   别用 `>` 覆盖，脚本自己用的是追加模式。
2. **`--adapter` 不传就是未微调的底座** —— 这不是错误，是**基线**：
   拿它和微调后的输出比，才知道 SFT 到底带来了什么。
3. **batch_size 先从小往大试**：`--batch-size 4` 起步，OOM 就降到 2。
   实测批处理相对单条有 4~5 倍加速（`tools/benchmark_generation.py`）。

## 第 2 步：先量速度，再决定跑多少

```python
!python tools/benchmark_generation.py --config configs/sft_kaggle.yaml \
    --num-prompts 8 --max-new-tokens 128 --reps 1
```

它会用**同一条真实代码路径**（`sft/infer.py`）量三种模式，并外推出
"生成 val/test/train 各要多久"。**先用这个数决定要不要上 vLLM**，
而不是先上 vLLM 再想值不值。

---

## 第 3 步：在真实输出上验奖励

```python
# 合成扰动（纯 CPU，几秒）：把参考摘要按确定性的方式破坏，看奖励是不是按预期掉
!python tools/check_reward.py --config configs/grpo.yaml

# 真实输出：分项统计 + 奖励与 ROUGE 的相关性 + 漏事实组的奖励差
!python tools/check_reward.py --config configs/grpo.yaml --triples data/triples/sft_val.jsonl
```

合成扰动这一遍是上 GRPO 之前**最后一道闸**。奖励方向反了的话，训练会
非常平稳地朝错的方向走，日志里完全看不出来（loss 照常下降）。
实测（200 条真实参考摘要）：

| 扰动 | 奖励 | 被门控 | 覆盖率 | 精确率 |
|---|---|---|---|---|
| 参考原文（基线） | 0.675 | 0% | 0.350 | 0.993 |
| 去掉所有数字 | 0.491 | 0% | 0.000 | 1.000 |
| 砍掉后半段 | 0.100 | 76% | 0.212 | 0.978 |
| 加"以下是摘要："前缀 | 0.000 | 100% | — | — |
| 把原文里数字全堆上 | 0.177 | 76% | 0.350 | 0.369 |

最后一行是**防 hacking 的证据**：堆数字没有涨分（反而被长度门控拦掉 76%），
同时 `fact_precision` 从 0.993 掉到 0.369 —— 这正是"必须同时记录精确率"
的原因，光看覆盖率它和基线一模一样。

拿到三元组之后，**不要直接开 GRPO**，先做这三件事：

| 检查 | 期望 | 不满足说明什么 |
|---|---|---|
| 门控被拦的比例 | 低（<10%） | 高的话问题在生成配置或 prompt，不在奖励设计 |
| 奖励与 ROUGE-L 的相关性 | 正相关、但不等价 | 完全重合说明事实覆盖没起作用；负相关说明奖励写反了 |
| 漏事实的样本奖励是否更低 | 是 | 否的话主信号没抓住痛点，回去改 `fact_coverage` |

`compute_rewards()` 返回的就是**分项**（`RewardBreakdown`），
所以这三项都能直接从同一遍打分里读出来，不用跑三遍。

⚠️ **覆盖率只对 38.3% 的验证样本有定义。** 实测 1340 条参考摘要里，
只有 38.3% 含金额/日期/编号这三类可验证事实（平均 0.94 个），
另外 61.7% 一条都没有 —— 这些样本的覆盖率按设计恒为 0，奖励天花板只剩
ROUGE 那一项（原文里平均有 21.9 个这样的数字，参考只写 0.94 个）。
对 GRPO 无害：组内归一化会把"组内每条都一样的常数"消掉。
但**跨实验比平均奖励没有意义**，报告里必须写清楚，
否则"gated 平均奖励比 flat 低"这种结论会被样本构成直接带偏。

⚠️ **只看总分一定会自欺欺人**：覆盖率有个明显的 hacking 路径
（把参考里的数字全堆进去），所以必须同时记录
`fact_precision`（输出的事实有多少真在原文里）。

---

## 第 4 步：GRPO，先跑 A1/A2 对照

唯一的开关是 `rl.reward.mode`（见 `configs/grpo.yaml`）：

| 组 | mode | 含义 |
|---|---|---|
| A1 | `flat` | 全部加权求和，各项可互相补偿 |
| A2 | `gated` | 硬门控 + 加权，格式不达标直接 0 |

```python
!python scripts/train_grpo.py --config configs/grpo.yaml \
    --override rl.reward.mode=flat  --override rl.output_dir=outputs/grpo_flat
!python scripts/train_grpo.py --config configs/grpo.yaml \
    --override rl.reward.mode=gated --override rl.output_dir=outputs/grpo_gated
```

两种模式**共用同一套分项计算**（`compute_reward`），所以差异只有一个变量：
可补偿性。这一点在面试里比"我们做了 A/B"重要得多 ——
能说清单变量是什么，才说明实验是可控的。

判据不是"gated 一定更好"，而是：**格式坍缩是否被抑制，且 ROUGE 没有掉**。
如果 gated 把 ROUGE 拉低了，说明门控设得太紧，先把
`gate.length_ratio_range` 放宽再看。

### 事实一致性作奖励（`mode=fact_judge`）

上面两种模式的"事实项"是金额/日期/编号的**规则匹配**。如果要求事实一致性
全部由模型判断（不写额外 checker），改用六要素事实一致性 Judge：

```bash
python scripts/train_grpo.py --config configs/grpo_fact_judge.yaml \
    --sft-adapter outputs/sft_chatglm3/best.pt
```

这时 `reward.total = 0.3·ROUGE-L + 0.7·fact_judge`，`fact_judge` 是
`sfzy/judge/` 那套 Judge 的加权分（∈ [0,1]），规则事实提取完全不跑。
数据流、日志字段、失败降级见 `docs/fact_judge.md` 第 6 节。
