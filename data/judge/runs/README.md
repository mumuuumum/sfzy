# 验证实验目录

每次用 `scripts/score_reward_vllm.py` 验证"候选奖励 < 人工奖励 + reward 与 ROUGE
正相关"这两条结论，都放一个独立子目录：`data/judge/runs/<exp>/`。

## 目录约定

```
data/judge/runs/
  exp1/                     第一次（历史基线，见 exp1/README.md）
  exp2/                     第二次（六要素上下文 + API 抽取，见 exp2/README.md）
  <exp>/
    run_config.json         当次生效的配置（模型/量化/prompt 版本/奖励权重/抽取器）
    <input-stem>.reward.jsonl   每个输入分片一份逐臂结果
    <input-stem>.reward.summary.json
    <input-stem>.reward.report.md
    merged.reward.jsonl      --merge 出来的全量合并
    merged.reward.summary.json
    merged.reward.report.md
```

## 跑法

```bash
# 卡 0 / 卡 1 各跑一个分片，结果直接落进 <runs-dir>/<--run>/
CUDA_VISIBLE_DEVICES=0 python scripts/score_reward_vllm.py --run exp2 \
    --config configs/grpo_fact_coverage_t4.yaml --dtype float16 --quantization bitsandbytes \
    --input data/triples/sft_val_shard0of2.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/score_reward_vllm.py --run exp2 \
    --config configs/grpo_fact_coverage_t4.yaml --dtype float16 --quantization bitsandbytes \
    --input data/triples/sft_val_shard1of2.jsonl

# 两个分片跑完后合成全量结论（CPU 即可，不加载裁判）
python scripts/score_reward_vllm.py --run exp2 --merge \
    data/judge/runs/exp2/sft_val_shard0of2.reward.jsonl \
    data/judge/runs/exp2/sft_val_shard1of2.reward.jsonl
```

`--out-dir` 仍然可用（优先于 `--run`）；不传 `--run`/`--out-dir` 时退回
`data/judge/`（旧行为，不推荐）。

## 注意

不同实验之间**不要直接比数字**：抽取 prompt、裁判上下文、覆盖率口径都改过。
每次的 `run_config.json` 记录了当次到底用了什么，比较前先看它。
