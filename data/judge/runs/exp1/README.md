# exp1：第一次验证（历史基线）

- 时间：2026-10-05
- 输入：`data/triples/sft_val_shard0of2.jsonl` / `sft_val_shard1of2.jsonl`
- 裁判：本地 Qwen2.5-7B-Instruct（bnb 4-bit，vLLM），`max_input_tokens=8192`
- 产物：两个分片的 `*.reward.jsonl`，以及 `merged.reward.{jsonl,summary.json,report.md}`

结论（当时）：两条都成立——Δ(人工−候选)=+0.1159、人工胜率 72.8%；
reward vs ROUGE-L 的 Spearman ρ≈0.29。

## 为什么只当历史基线

这次跑在几个修复之前，数字**不能**和后面的实验直接比：

1. `element_coverage` 当时被误用事实一致性的 prompt 计算（左=参考摘要被当成原文），
   **覆盖率口径是错的**；
2. 抽取只是一版 prompt，还没分"原文 / 摘要"两套，也没修日期挪用、"未答辩"被判成
   "辩称"的问题；
3. 判定还没有六要素上下文（`element_context`）。

要复现/对比请用后续实验目录。
