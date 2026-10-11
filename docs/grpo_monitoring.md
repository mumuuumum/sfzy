# GRPO 训练监控体系（SwanLab）

本文件是"完整 SwanLab 监控"任务的交付说明：指标清单、统计口径、代码位置、
新增/复用情况、命名与频率、以及对手写 GRPO 关键量的解释。

代码入口：

- 监控层：[`src/sfzy/rl/monitor.py`](../src/sfzy/rl/monitor.py)（`GRPOMonitor` / `MonitorBatch`）
- 训练接线：[`src/sfzy/rl/trainer.py`](../src/sfzy/rl/trainer.py)（`rollout` / `step` / `train` / `_run_eval`）
- 面板分组：[`src/sfzy/utils/tracking.py`](../src/sfzy/utils/tracking.py)（`MONITOR_COLUMNS` / `group_metrics`）
- 算法内核（未改动）：[`src/sfzy/rl/grpo.py`](../src/sfzy/rl/grpo.py)
- Reward（未改动，仅新增只读的门控前计算）：[`src/sfzy/rl/reward.py`](../src/sfzy/rl/reward.py)
- 测试：[`tests/test_monitor.py`](../tests/test_monitor.py)

## 0. 设计边界与保证

- 监控层**只读派生**：不改 advantage、组过滤、长度归一化、PPO 裁剪、KL、LoRA 更新。
- Reward 的权重、Judge Prompt、评分标准、Gate 规则全部不变。
- 不额外调用裁判、不重新生成摘要。所有 per-step 指标都来自 `step()` 已产生的数据。
- 只由主进程上报（`is_main_process` 守卫），断点续训沿用原有 run 恢复逻辑，
  `global_step` 取 `state.step + 1`，与优化步一一对应。
- 监控开关（`monitor.enabled=false`）时，Reward、优势、loss、参数更新与开启前逐位一致
  （由 `test_监控开关不改变训练结果` 钉住）。
- 无法计算的指标直接跳过，不写 0、不编造（例如 `kl_coef=0` 时不写 KL，缺裁判信号时不写对应分项）。

## 1. 命名与分组

统一 `模块/指标`：

| 一级目录 | 内容 |
|---|---|
| `grpo/` | 组内奖励分布、有效组、组内输出多样性 |
| `advantage/` | 原始与有效优势 |
| `train/` | loss、学习率、梯度、Ratio、裁剪、KL、熵、LoRA 更新 |
| `reward/` | 门控前 / 最终总奖励 |
| `reward_terms/` | 动态 Reward 分项 |
| `fact/` | 事实一致性（FC） |
| `coverage/` | 关键要素覆盖率（KEC） |
| `ee/` | 语言表达效率（EE） |
| `gate/` | Gate 行为与对学习信号的影响 |
| `relation/` | 多目标相关性及分项对组内排序的贡献 |
| `length/` | 长度、长度比、长度归一化 |
| `behavior/` | 生成行为 |
| `eval/` | 固定验证集 |

## 2. 日志频率与配置

配置块在 `rl.monitor` 下（见 `configs/grpo_fact_judge.yaml`）：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `log_interval` | 1 | 基础训练 / 组 / 优势 / 总奖励 / 长度 / 行为 / Gate 原因 |
| `reward_log_interval` | 10 | Reward 分项、六要素、Coverage、EE、Gate 深度统计 |
| `relation_log_interval` | 20 | 相关性、Coverage/EE 冲突、分项消融 |
| `entropy_log_interval` | 20 | 策略熵（同一次前向） |
| `lora_update_log_interval` | 20 | LoRA 更新范数 |
| `eval_interval` / `eval.every_n_steps` | 100 | 固定验证集（按步数轮询） |
| `eval.quantile_points` | 0（关） | 固定验证集（按分位点：n 步等分成 x 段，第 k/x 分位点各验一次） |
| `correlation_window` | 512 | 跨步相关性窗口样本数 |
| `near_zero_std_eps` | 1e-6 | "近零方差组"阈值 |
| `ee_low_threshold` / `ee_high_threshold` | 0.5 / 0.75 | EE 低/高分阈值 |
| `repetition_ngram` | 3 | 重复 n-gram 的 n |
| `near_upper_gate_margin` | 0.1 | 距长度比上界多近算"接近上界" |
| `relation_enabled` | true | 可整体关掉关系/消融分析 |
| `eval.enabled` / `eval.data_file` / `eval.max_samples` / `eval.baseline` | 关闭 / 无 | 可选固定验证集 |
| `eval.save_best` / `eval.best_metric` | true / `rouge_total` | 按验证分维护 `best.pt` |
| `eval.max_length` | null | null = 与训练的 `rl.max_length` 一致（**不要填更小的值**） |

步数很少时，`GRPOMonitor._interval` 自动缩短上述间隔（至多 `total_steps//5`），
避免短跑一条扩展曲线都记不到。

## 3. 指标清单、口径与代码位置

### 3.1 GRPO 组内 Reward（步频：`log_interval`）— `monitor._core_metrics`

分组 = 同一原文的 G 个候选（rollout 按 group-major 排列，一组连续）。

| 指标 | 定义 / 分母 |
|---|---|
| `grpo/group_reward_mean` | 各组最终 Reward 均值，再对组取平均 |
| `grpo/group_reward_std_mean` | 各组最终 Reward 的 **population std** 均值（与 `group_advantages` 一致） |
| `grpo/group_reward_std_p10/p50/p90` | 组内 std 的分位数 |
| `grpo/group_reward_range_mean` | 各组 `max-min` 的均值 |
| `grpo/group_reward_range_p90` | 组内极差的 P90 |
| `grpo/zero_std_group_rate` | 组内 std 恰好为 0 的组占比 |
| `grpo/near_zero_std_group_rate` | 组内 std < `near_zero_std_eps` 的组占比 |
| `grpo/all_zero_group_rate` | 组内全部候选 Reward = 0 的组占比 |
| `grpo/all_equal_positive_group_rate` | 组内全部相同且 > 0 的组占比 |
| `grpo/effective_group_rate` | 被保留 **且** 组内 std > eps 的组占比 |
| `grpo/kept_group_rate` | 经过实际组过滤（`keep`）后保留的组占比 |
| `grpo/filtered_group_rate` | `1 - kept_group_rate` |
| `grpo/groups_per_step` | 本步原始组数 P |
| `grpo/effective_groups_per_step` | 保留且非零方差的组数 |
| `grpo/unique_completion_ratio` | 每组 `唯一文本数 / G` 的均值 |
| `grpo/duplicate_completion_group_rate` | 组内存在重复文本的组占比 |
| `grpo/completion_length_group_std` | 组内字符长度 population std 的均值 |

保留过滤前的组内指标：`group_std_*`、`zero_std_group_rate` 等都在**过滤前**的
全量组上计算，`kept_group_rate` 单独反映过滤，避免过滤掩盖零方差问题。

### 3.2 Advantage（步频：`log_interval`）— `_advantage_block`

| 指标 | 定义 |
|---|---|
| `advantage/raw_mean/std/min/max` | 组内归一化后（`group_advantages`）、长度缩放与 Mask **之前**的优势，分母 = 全部候选 |
| `advantage/raw_abs_mean` / `raw_abs_p90` | 上者绝对值的均值 / P90 |
| `advantage/raw_positive_ratio` / `raw_negative_ratio` / `raw_zero_ratio` | >0 / <0 / ==0 的比例 |
| `advantage/effective_mean/std/abs_mean/abs_p90` | 经过 `length_normalize` + `group_mask` 后真正进 loss 的优势，分母 = 全部候选（被过滤组为 0） |
| `advantage/effective_zero_ratio` / `positive_ratio` / `negative_ratio` | 上者的符号分布 |

`raw` 与 `effective` 都用与训练**同一个** `grpo.group_advantages` / `length_normalize`
口径在监控端重算（纯函数，不改变训练值）。

### 3.3 训练稳定性（步频：`log_interval`）— `_core_metrics`

| 指标 | 定义 / 来源 |
|---|---|
| `train/loss` | 实际反向的总损失（含 KL） |
| `train/policy_loss` | GRPO 策略损失（`grpo_loss` 返回，未加 KL） |
| `train/learning_rate` | `optimizer.param_groups[0]["lr"]` |
| `train/grad_norm_preclip` | `clip_grad_norm_` 返回的裁剪前总范数 |
| `train/grad_norm_postclip` | `min(preclip, max_grad_norm)` |
| `train/gradient_clip_applied_rate` | 本步是否触发裁剪（1/0） |
| `train/global_step` | 实际优化步 = `state.step + 1` |
| `train/ratio_mean/std/max/p90` | `ratio = exp(logprobs - old_logprobs)` 的分布 |
| `train/clip_frac` | `grpo_loss` 返回的裁剪比例（与算法一致） |
| `train/clip_frac_low` / `clip_frac_high` | `ratio < 1-ε` / `ratio > 1+ε` 的比例，ε=`clip_ratio` |
| `train/entropy` | 生成段逐 token 平均策略熵（低频；同一次前向计算） |
| `train/ref_kl_estimate` | k3 KL 估计 = 加权惩罚 / `kl_coef`，仅 `kl_coef>0` 且真的算了参考 KL 时 |
| `train/kl_penalty` | 实际计入 loss 的加权 KL，仅上述条件 |
| `train/lora_update_norm` | 优化前后 LoRA 参数差的 L2 范数（低频） |

### 3.4 总 Reward（步频：`log_interval`）— `_reward_block`

| 指标 | 定义 |
|---|---|
| `reward/raw_mean/std/min/max/p10/p50/p90` | **门控前**奖励：`Σ w_i·term_i / Σ w_i`，用同一套分项与归一化权重、不施加 Gate |
| `reward/raw_zero_rate` | 门控前奖励为 0 的比例 |
| `reward/raw_group_std` / `raw_zero_std_group_rate` | 门控前组内 std 均值 / 零方差组占比 |
| `reward/final_*` | 实际进入优势计算的奖励（门控后） |
| `reward/final_group_std` / `final_zero_std_group_rate` | 上者组内 std 均值 / 零方差组占比 |

门控前奖励由 `reward.compute_pre_gate_rewards` 计算（新增、仅监控），与训练
共用同一份补缺后的裁判信号，因此分项口径与训练一致。组合项按 `RewardSpec.active_terms`
处理，被 `quality_fbeta` 吞并的项不会重复计入。

### 3.5 Reward 分项（步频：`reward_log_interval`）

对 `spec.active_terms`（真正参与加权和的项，含组合项）逐项：

`reward_terms/{name}/mean`、`/std`、`/p10`、`/p50`、`/p90`、`/group_std`、
`/nonzero_group_std_rate`、`/weighted_mean`（= 该项均值 × 归一化权重）。

`reward_terms/*/mean` 是**门控前**的原始分项得分（`compute_pre_gate_rewards`）；
`weighted_mean` 是乘配置权重后的贡献；`group_std` 反映该项在 GRPO 里是否有区分度。
权重的实际取值写进 run 配置（`_resolved_reward.normalized_weights`）。

### 3.6 事实一致性 FC（步频：`reward_log_interval`）— `_fact_metrics`

总体（分母 = 有裁判信号的候选）：

`fact/mean`、`/std`、`/p10`、`/p50`、`/p90`、`/zero_rate`、`/group_std`

门控：

`fact/gate_rate`（被事实硬门控拦下的候选比例，分母 = 全部候选）、`fact/pass_rate`

六要素（`ELEMENTS`），分母 = 有该要素原始分的候选：

`fact/{element}/mean`（原始分 /4）、`/zero_rate`、`/group_std`

聚合：

`fact/score_ratio_0..4`（所有要素原始分的档位分布）、
`fact/zero_element_count_mean`（每条候选 0 分要素数的均值）、
`fact/multi_zero_element_rate`（≥2 个要素 0 分的候选比例）、
`fact/min_element_score_mean`（`JudgeResult.min_element_score` 的均值，已排除案由）、
`fact/group_mixed_gate_rate` / `group_all_pass_rate` / `group_all_fail_rate`
（按组，分母 = 有裁判数据的组）。

### 3.7 关键要素覆盖率 KEC（步频：`reward_log_interval`）— `_coverage_metrics`

总体（分母 = 有覆盖率信号的候选）：

`coverage/mean/std/p10/p50/p90`、`coverage/zero_rate`、`coverage/full_score_rate`、
`coverage/group_std`、`coverage/nonzero_group_std_rate`

逐要素（分母 = 有该要素覆盖率原始分的候选）：

`coverage/{element}/mean`（原始分 /4）、`coverage/{element}/missing_rate`、`coverage/{element}/group_std`

> 口径说明：覆盖率 prompt 规定"参考摘要没有写到的要素直接给 4（无需覆盖）"，
> 因此逐要素 4 分无法区分"参考没有"与"完整覆盖"。监控据此外只把 **0 分**当作
> 确定性的"参考有、候选完全没覆盖"，`missing_rate` 即其比例；请求里要求的
> "仅参考含该要素样本上的条件均值"因当前判定不输出"参考是否含该要素"而**不可分离**，
> 未编造。逐要素 `mean` 是判定原始分的直接均值。

长度关系：`coverage/length_spearman`、`coverage/group_length_spearman`
（后者是组内相关性的组均值，常量组被排除）。

### 3.8 表达效率 EE（步频：`reward_log_interval`）— `_ee_metrics`

总体：`ee/mean/std/p10/p50/p90`、`ee/low_score_rate`（≤`ee_low_threshold`）、
`ee/high_score_rate`（≥`ee_high_threshold`）、`ee/group_std`、`ee/nonzero_group_std_rate`

三个子维度（项目当前 EE 的真实维度，**均为越高越好的效率分，0-4**）：

`ee/semantic_redundancy_{mean,std,group_std}`、
`ee/verbal_redundancy_{mean,std,group_std}`、
`ee/abstraction_adequacy_{mean,std,group_std}`

> 命名说明：需求文本列出 `ee/semantic_*` / `ee/factual_*` / `ee/expression_*`，
> 但项目当前 Judge 的真实三维是"语义冗余 / 表达冗余 / 抽象适当性"，
> 没有"事实冗余"维度；因此按真实维度命名并标注方向，未造 `factual` 曲线。

长度关系：`ee/length_spearman`、`ee/group_length_spearman`。

### 3.9 Gate（`reward_log_interval`：深度统计）— `_gate_reason_metrics` / `_gate_learning_metrics`

原因分布（分母 = 全部候选，只记**第一个**失败原因，与 `check_gate` 一致）：

`gate/total_rate`、`gate/fact_rate`、`gate/length_too_long_rate`、
`gate/length_too_short_rate`、`gate/min_length_rate`、`gate/no_result_marker_rate`、
`gate/forbidden_prefix_rate`、`gate/other_rate`

对学习信号的影响（监控用优势，基于 `grpo.group_advantages` 同口径重算，不改训练）：

`gate/all_gated_group_rate`、`gate/mixed_group_rate`、`gate/all_passed_group_rate`、
`gate/group_std_before`、`gate/group_std_after`、`gate/advantage_sign_change_rate`
（门控前/后奖励各自做组内归一化，符号在两次都不为 0 的候选里统计变化比例）

质量分项关系（仅在对应分项有评分时）：

`gate/coverage_mean_gated` / `_passed`、`gate/ee_mean_gated` / `_passed`

### 3.10 Reward 之间的关系与贡献（步频：`relation_log_interval`）— `_relation_metrics`

窗口相关性（分母 = 窗口内该对变量都非缺失的样本；样本 <3 或任一维度无方差则**不写**）：

`relation/coverage_ee_spearman`、`coverage_fact_spearman`、`ee_fact_spearman`、
`coverage_length_spearman`、`ee_length_spearman`、`final_reward_length_spearman`、
`effective_signal_length_spearman`

组内相关性（对每个有定义的组算，再取组均值）：

`relation/group_coverage_ee_spearman`、`group_coverage_length_spearman`、
`group_ee_length_spearman`、`group_reward_length_spearman`

Coverage / EE 冲突（高低按**同组内中位数**的相对高低，平局不计入，分母 = 非平局的候选）：

`relation/high_coverage_low_ee_rate`、`low_coverage_high_ee_rate`、
`high_coverage_high_ee_rate`、`low_coverage_low_ee_rate`

分项消融（只构造假设性 Reward，保持真实 Gate 不变，绝不改训练）：

`relation/{coverage,ee,fact_soft}_top1_flip_rate`
`relation/{coverage,ee}_advantage_sign_flip_rate`（`fact_soft` 同样记录）
`relation/{coverage,ee}_advantage_change_mean`

消融构造：对每个软分项，`reward_abl = Σ_{k≠term} w_k·term_k / Σ_{k≠term} w_k`
（被门控候选仍为 0），再按 `group_advantages` 同口径重算优势。
`top1_flip` 用"组内最优集合是否改变"判定，并列最高视为同一集合；
只在原始组内有方差的组上统计（分母 = 这些组）。真实 `reward` 全程不变
（`test_消融不改变真实奖励且处理并列` 钉住）。

> 过滤口径：真实组过滤（`group_mask`）依赖 Reward，移除分项后某些组的方差可能消失。
> 监控消融**不重新过滤**，而是固定"原始组内有方差"的组集合做对比（分母就是这个集合），
> 这样翻转率/变化量反映的是同一批候选在去掉某分项后的排序变化，而不是过滤集合变化。

### 3.11 长度与生成行为（步频：`log_interval`）— `_length_metrics` / `_behavior_metrics`

长度（字符）：`length/output_mean/std/p10/p50/p90/max`
token：`length/tokens_mean/std/p90`
长度比（与 Gate 同口径 `len(cand)/len(ref)`）：`length/ratio_mean/std/p50/p90`、
`length/ratio_over_1_2_rate`、`length/ratio_over_1_5_rate`、`length/near_upper_gate_rate`
（距 `gate.length_ratio_range[1]` 不超过 `near_upper_gate_margin`；Gate 关闭时不记）
长度归一化：`length/normalization_factor_mean/p90`（按 `length_norm` 模式复算的缩放因子，
`sqrt`/`mean`/`none`）；`<mode>` 作为 run 配置记录在 `_resolved_monitor.length_norm`。

生成行为：`behavior/truncation_rate`（生成 token 数 ≥ `max_new_tokens`）、
`behavior/eos_termination_rate`、`behavior/result_marker_rate`、
`behavior/forbidden_prefix_rate`、`behavior/repetition_rate`
（`1 - 唯一 n-gram / 总 n-gram`，n=`repetition_ngram`）。

### 3.12 固定验证集（步频：`eval_interval`，可选）— `eval_metrics`

默认关闭。开启且提供 `data_file` 时，在固定验证集上用**贪心解码**生成摘要，
固定分词口径算 ROUGE，并复用同一 Reward / 裁判：

`eval/rouge_1`、`eval/rouge_2`、`eval/rouge_l`、`eval/rouge_total`、
`eval/length_ratio_mean`、`eval/length_ratio_p50`、`eval/fact_mean`、`eval/fact_zero_rate`、
`eval/coverage_mean`、`eval/ee_mean`、`eval/final_reward_mean`；
配置 `eval.baseline` 时另记 `eval/delta_rouge_l`、`delta_coverage`、`delta_ee`、
`delta_fact_mean`、`delta_length_ratio`。

缺信号的项不写（例如裁判只有 fact 时不写 `eval/coverage_mean`）。

**验证调度**。两种二选一：

* `eval.every_n_steps`（或旧的 `eval_interval`）：每 N 步一次，按步数轮询；
* `eval.quantile_points = x`：**按分位点** —— 把总步数 n 等分成 x 段，在第
  k/x 分位点各验一次（k=1..x）。x=10、n=10738 → 第 1074/2148/…/10738 步。
  最后一个点必然落在最后一步，所以"训练结束时的模型"一定会被评到。设了
  `quantile_points` 就以它为准，`every_n_steps` 被忽略。

**best.pt**。每次验证后，用 `eval.best_metric`（默认 `rouge_total`，即官方口径
`0.2*R1 + 0.4*R2 + 0.4*RL`）与历史最好分比较（越大越好）；刷新了就覆盖
`<output_dir>/best.pt`，并在日志里打一行 `★ 验证集 eval/rouge_total=… 刷新最好成绩`。
历史最好分存在 `RLState` 里随 checkpoint 落盘，所以**续训不会用更差的分覆盖
best.pt**；`best.pt` 不参与 `keep_last_n_checkpoints` 的清理。

验证的截断/生成预算必须与训练可比：`eval.max_length: null` 表示跟训练用同一个
`rl.max_length`（给了整数则取 `min(它, rl.max_length)`），`eval.max_new_tokens`
则取 `min(它, rl.max_new_tokens)`。

## 4. 未记录 / 不适用

| 需求项 | 处理与原因 |
|---|---|
| `train/epoch` | 不适用：GRPO 单遍遍历 prompt 池，没有 epoch 概念 |
| `ee/factual_*` | 不适用：当前 EE 没有"事实冗余"维度（见 3.8） |
| 覆盖率"仅参考含该要素"的条件均值 | 不可计算：覆盖率 Judge 把"参考无该要素"编码为 4 分，与满分不可区分；不编造 |
| `train/ref_kl_estimate` / `train/kl_penalty` | `kl_coef=0` 时不计算也不上报（不写 0） |
| `train/entropy` | 低频；仅存在时上报（未开启的频率步不写） |
| 基线锚相关指标（`anchor_filtered` 等） | 按任务要求**不监控**（基线锚逻辑不在本次范围） |
| `eval/delta_*` 缺基线 | 不写，不做无依据的对比 |
| 事实精确率类指标（`fact_precision`） | 不设固定阈值、不把 Fact Judge 分命名为精确率 |

## 5. 手写 GRPO 关键量解释

- **old/current logprob**：`rollout` 后、任何参数更新前先算 `old_logprobs`（detach，
  **整个 μ 内层循环共用**），再用同一批参数算带梯度的 `logprobs`。因此**在
  dropout 关闭的确定性前向下**：`rl.inner_iterations`（μ）=1 时 `ratio≡1`、
  `clip_frac≡0` —— 这是"每批只更新一次"的结构性结果，不是实现出错；μ>1 时第 1 轮
  仍是 1，第 2 轮起才出现真实漂移，这时 Ratio/Clip 才是在看"策略被这一步推了多远"。
- **advantage**：`A_i=(r_i-mean)/ (std+eps)`，`eps` 保证组内全同退化为 0 而非 nan。
  `advantage/raw_*` 是其原始值；`advantage/effective_*` 是再经长度缩放与组 Mask 后
  真正进 loss 的值。
- **长度归一化**：本项目 `length_normalize` **作用在 advantage 上**（非 logprob），
  默认 `sqrt`：`A ← A / sqrt(len)`；`length/normalization_factor_*` 与之一致。
  `relation/effective_signal_length_spearman` 用有效优势与长度的相关观察长度偏置。
- **PPO 裁剪**：`loss=-mean(min(ratio·A, clip(ratio)·A))`，`clip_frac` 是算法返回值，
  `clip_frac_low/high` 按 `[1-ε,1+ε]` 分界统计。μ=1 时它必然为 0（裁剪不起作用），
  μ>1 才会真正生效；多个内层迭代的 ratio **拼接**在一起统计，所以 μ>1 时
  `ratio_*` / `clip_frac_*` 的口径是"整步全部迭代"，日志里另有
  `μ{μ} clip 首轮%→末轮%` 一行看漂移怎么长起来。
- **KL**：k3 无偏估计 `mean(exp(ref-policy)-(ref-policy)-1)`；`ref_kl_estimate` 是估计值，
  `kl_penalty` 是乘系数后计入 loss 的值，两者分开记录。参考策略 = **同一底座 +
  SFT 的 LoRA 快照**（不额外加载 6B 权重），参考 logprob 与策略参数无关，所以整个
  μ 循环只算一次。`kl_coef=0` 时不加载参考策略、不记 0。

## 6. 单元测试

`tests/test_monitor.py`（共 20 项）覆盖需求第十四节：

1. 同一原文多 Rollout 正确分组；2. 不同组不互相参与组内标准化；3. 组内全 0；
4. 组内全等正；5. 微小差异；6. 明显差异；7. 过滤前后有效组比例；
8. raw/effective 优势口径；9. Gate 前后奖励与组内 std；10. 六要素 Fact=0 统计；
11. 覆盖率逐要素/漏覆盖；12. EE 三维度与方向；13. 常量组相关系数不写 0；
14. 消融并列处理；15. 消融不改真实奖励；16. 首次更新前 ratio≈1；
17. KL 未计算不写 0；18. 长度归一化统计与实现一致；19. global_step 对齐；
20. 监控开关不改训练结果（Reward/优势/loss/参数一致）。

运行：`python -m pytest tests/test_monitor.py -q`。

## 7. 确认未改动的东西

手写 GRPO 算法（优势、组过滤、长度归一化、PPO 裁剪、KL、LoRA 更新）、
Reward 权重与聚合、Judge Prompt 与评分标准、Gate 规则、生成（rollout）行为均未改动。
唯一进入奖励路径的新增函数是只读的 `compute_pre_gate_rewards`（监控用），
不参与 `compute_rewards` 的任何结果。
