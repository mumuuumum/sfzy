# exp2：第二次验证（六要素上下文 + API 抽取）

目标：用修好之后的流水线重新验证两条结论——

1. 候选摘要的奖励一般比人工摘要小；
2. 候选摘要的 reward 与它相对人工摘要的 ROUGE 正相关。

和 exp1 的差别（都在 `run_config.json` 里固化）：

- 抽取分两套 prompt：判决书原文 `EXTRACT_PROMPT_VERSION`、摘要
  `EXTRACT_SUMMARY_PROMPT_VERSION`；
- 判定带六要素上下文（`semantic.element_context: true`）：判分依据 + 辅助参考，
  内容串了要素给 3 分而不是 0；
- 覆盖率走自己的 `COVERAGE_SYSTEM`（exp1 里被误用事实一致性 prompt）；
- 抽取走 API（`semantic.extract_api`），判定仍用本地 vLLM。

## 跑法

```bash
export SFZY_JUDGE_API_KEY=sk-...

# 卡 0：shard0
CUDA_VISIBLE_DEVICES=0 python scripts/score_reward_vllm.py --run exp2 \
    --config configs/grpo_fact_coverage_t4.yaml --dtype float16 --quantization bitsandbytes \
    --input data/triples/sft_val_shard0of2.jsonl

# 卡 1：shard1
CUDA_VISIBLE_DEVICES=1 python scripts/score_reward_vllm.py --run exp2 \
    --config configs/grpo_fact_coverage_t4.yaml --dtype float16 --quantization bitsandbytes \
    --input data/triples/sft_val_shard1of2.jsonl

# 合并出全量结论
python scripts/score_reward_vllm.py --run exp2 --merge \
    data/judge/runs/exp2/sft_val_shard0of2.reward.jsonl \
    data/judge/runs/exp2/sft_val_shard1of2.reward.jsonl
```

产物：`<input-stem>.reward.{jsonl,summary.json,report.md}` + `merged.reward.{jsonl,summary.json,report.md}`
+ `run_config.json`。最终对外用 `merged.reward.report.md`。

断点续跑：默认跳过已有 id；本次要重跑就加 `--no-resume`（或删掉本目录里对应的
`.reward.jsonl`）。
