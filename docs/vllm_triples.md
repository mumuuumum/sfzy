# 用 vLLM 生成训练集三元组

`scripts/generate_triples.py` 走 HuggingFace 的生成循环，一次一个 batch，
6B 模型生成全量 train（10738 条）很慢。`scripts/generate_triples_vllm.py`
把同一件事换到 vLLM 上，用连续批处理 + PagedAttention 把 GPU 吃满。

两条路径的产物格式完全一致：

```json
{"id": "...", "source": "文书全文", "reference": "参考摘要", "output": "模型生成的摘要"}
```

并且断点续跑（已完成的 id 跳过）、`--shard I/N`、`--input` 任意子集都和
HF 版本对齐，所以输出的 `data/triples/sft_train.jsonl` 可以直接喂给
`tools/select_rl_prompts.py --triples ...` 做 GRPO 的 SFT 基线锚。

需要**一篇文书多个候选**时加 `--num N`：产物多一个 `outputs` 列表字段
（`output` 仍是第一个候选，读单字段的下游不受影响）：

```json
{"id": "...", "source": "文书全文", "reference": "参考摘要",
 "output": "候选1", "outputs": ["候选1", "候选2", "候选3", "候选4"]}
```

---

## 一、为什么不能直接用现有环境

vLLM 的版本和 transformers 强绑定，而 ChatGLM3 的远程代码又在 transformers
5.x 上加载不了（见 `requirements-cloud.txt`）。两边对 transformers 的要求
是冲突的：

| | transformers | torch | 备注 |
|---|---|---|---|
| SFT 环境（`requirements-cloud.txt`） | ==4.40.2 | 镜像自带 | ChatGLM3 的口径 |
| vLLM 0.6.6（推荐） | >=4.45.2 | ==2.5.1 | `outlines` 用 airportsdata |
| vLLM 0.5.5（本机实测） | >=4.43.2 | ==2.4.0 | `outlines` 依赖 pyairports |
| vLLM 0.30（2026） | >=5.10.4 | 新版 | 会和 ChatGLM3 冲突 |

**必须显式把 transformers 钉在 5.0 以下**，否则 pip 会给你装 5.x：
5.x 在 torch 2.4 上会直接"禁用 PyTorch"，ChatGLM3 的远程代码也加载不了。

```bash
# 推荐的独立环境（和 SFT 环境隔离，互不影响）
pip install "vllm==0.6.6" "transformers==4.46.3"

# 或者复用 SFT 环境（把 transformers 从 4.40.2 抬到 4.44）—— 本机实测这条能跑
pip install "vllm==0.5.5" "transformers==4.44.2"
```

两个环境相关的坑：

* vLLM 0.5.5 的 `outlines==0.0.46` 依赖 `pyairports`，而 PyPI 上现在只有
  一个空的同名 0.0.1（装了也不提供 `pyairports` 模块），会让 `import vllm`
  直接 `ModuleNotFoundError`。**所以优先用 0.6.6**（它改用 `airportsdata`）；
  实在要用 0.5.5，就在 site-packages 里补一个 `pyairports/airports.py`，
  内容一行 `AIRPORT_LIST = []`。
* 本脚本刻意不 import 项目的模型加载代码（`sfzy.models.loader` / `lora` /
  `sft.infer`），只依赖纯 Python 的 prompt 拼接、tokenizer 和 vLLM，
  所以能在干净环境里跑，不需要装 peft / 项目那套兼容补丁。

> 注意 T4：vLLM 上 ChatGLM3 默认按 config 的 bf16 加载，但 Turing 不支持
> bf16，Kaggle 上必须显式加 `--dtype float16`。AutoDL 4090 不用管。

精度默认跟随项目模型配置里的 `torch_dtype`（`--dtype auto` 时），不会去
读 ChatGLM3 自带 config.json 里的值；T4 上仍要用 `--dtype float16` 覆盖。

## 二、LoRA adapter：`.pt` 先导出成 PEFT 目录

项目训练出来的 checkpoint 是自定义格式（`...query_key_value.lora_A.weight`），
vLLM 不认识。脚本会自动把它导出成 PEFT 目录
（`adapter_config.json` + `adapter_model.safetensors`），补上 vLLM 要求的
`base_model.model.` 前缀，并把 rank/alpha/target_modules 写进
`adapter_config.json`。产物默认落在 `outputs/lora_peft/<checkpoint 名>/`，
再次运行会复用。

只想验证导出格式、不生成：

```bash
python scripts/generate_triples_vllm.py --config configs/sft_cloud.yaml \
    --adapter outputs/sft_chatglm3/step_000336.pt --export-only
```

## 三、跑法

```bash
# 冒烟：4 条确认接线（本地 0.5B 就能验，几十秒）
python scripts/generate_triples_vllm.py --config configs/sft_local.yaml \
    --adapter outputs/sft_local/best.pt --split val --limit 4 \
    --max-new-tokens 32 --enforce-eager

# AutoDL 2×4090：给全量 train 生成 SFT 输出（默认写 data/triples/sft_train.jsonl）
python scripts/generate_triples_vllm.py --config configs/sft_cloud.yaml \
    --adapter outputs/sft_chatglm3/step_000336.pt \
    --split train --batch-size 128

# 只给 GRPO 的 prompt 池（1000 条）生成 sft_output
python scripts/generate_triples_vllm.py --config configs/sft_cloud.yaml \
    --adapter outputs/sft_chatglm3/step_000336.pt \
    --input data/splits/rl_prompts.jsonl
```

多卡两种用法，二选一：

```bash
# A. tensor parallel：一个进程，一张逻辑卡
python scripts/generate_triples_vllm.py --config configs/sft_cloud.yaml \
    --adapter outputs/sft_chatglm3/step_000336.pt --split train \
    --tensor-parallel-size 2

# B. 数据并行：两个进程各绑一张卡、各跑一个分片，最后按 id 合并
CUDA_VISIBLE_DEVICES=0 python scripts/generate_triples_vllm.py ... --shard 0/2
CUDA_VISIBLE_DEVICES=1 python scripts/generate_triples_vllm.py ... --shard 1/2
```

6B 模型在单张 4090 上装得下，**数据并行（B）通常比 tensor parallel 更快**，
因为省掉了跨卡通信；tensor parallel 更适合单卡装不下的情况。

### 一篇文书多个候选：`--num N`

多候选是采样的产物，`n=N` 直接交给 vLLM：

```bash
# 每篇 4 个候选，默认写 data/triples/sft_train_n4.jsonl
python scripts/generate_triples_vllm.py --config configs/sft_cloud.yaml \
    --adapter outputs/sft_chatglm3/step_000336.pt \
    --split train --num 4 --temperature 0.7
```

* `--num 1`（默认）走贪心，产物和以前**逐字节同构**（只有 4 个字段）。
* `--num >1` 必须采样：`--temperature` 不能为 0，否则 N 个候选完全相同。
  不显式传 `--temperature` 时，`--num>1` 自动取 `0.7`；`--top-p` 默认 `1.0`。
* `--num>1` 时产物是 5 个字段，多了 `outputs` 列表；`output` = `outputs[0]`。
  只读 `output` 的 `tools/select_rl_prompts.py`、`tools/score_fact_judge.py`、
  `tools/bench_metrics.py` 不用改就能继续用。
* 默认输出路径会带 `_n{num}`（如 `sft_train_n4.jsonl`），不会覆盖单候选文件。
* 想固定随机种子做可复现的多候选：`--seed`（默认 42）会传给 vLLM 引擎。

## 四、和 HF 版本的差异（写报告时要提）

* 贪心解码在两边都是确定性的，但 vLLM 的算子、批处理顺序和 HF 不同，
  **逐字节结果不保证一致**。三元组是用来观察失败模式和验证奖励的，
  这点数值差异不影响结论；不要把两份文件当作可精确对比的复现实验。
* 之前用 HF 生成的 `sft_val.jsonl` / `sft_test.jsonl` 可以继续用。
  vLLM 只为 train 生成即可（默认就是同一个输出路径，id 不会重复）。

## 五、给 val 三元组打奖励分（验证 reward 设计）

生成了 `sft_val_shard*of2.jsonl` 之后，用
`scripts/score_reward_vllm.py` 在两张 T4 上各打一个分片，比较**候选摘要
（output）**和**人工摘要（reference）**的奖励分。同样是 vLLM 裁判、同样
直接把 token id 喂给 vLLM。跑法和 T4 的显存注意事项见
`docs/vllm_reward_eval.md`。
