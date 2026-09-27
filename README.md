# sfzy — 裁判文书摘要（CAIL2020 司法摘要赛道）

把 2025 年 3 月基于 LLaMA-Factory 的裁判文书摘要研究，重写为**自实现代码**的项目。

目标不是"跑通一个框架"，而是让每个核心模块的选择都能讲清为什么：数据怎么构造、
LoRA 为什么这么注入、loss mask 为什么必要、检索为什么用混合召回、RL 为什么从 DPO
换成规则奖励。

具体分工：数据构造、LoRA、训练循环、检索算法、RL 算法、评测指标由本人手写；
配置加载、目录脚手架、CLI 入口、测试用例由工程化代码提供。

---

## 任务定义

输入一篇**民事一审判决书**全文，输出该文书的司法摘要。

---

## 数据

### 来源与规模

CAIL2020 司法摘要赛道，约 1 万篇民事一审判决书及对应参考摘要。
本仓库已合并为 `data/raw/sfzy_cail.json`，去重后共 **13427 条**。

### 原始格式（JSONL，每行一条）

```json
{"id": "...",
 "summary": "参考摘要",
 "text": [{"sentence": "句子", "label": 1}, {"sentence": "句子", "label": 0}]}
```

* `text` 是按句子切分后的列表，**空串拼接可无损还原原文**。
* `label` 是官方给出的句子重要度（抽取式标注）。本项目的生成式路线**不使用**该字段，
  解析时直接忽略。
* 测试集口径只有 `id` 与 `text[].sentence`。

### 提交格式（每行一条）

```json
{"id": "...", "summary": "预测摘要"}
```

### 评测口径

官方 ROUGE，三项均取 F-score：

```
总分 = 0.2 * F1(ROUGE-1) + 0.4 * F1(ROUGE-2) + 0.4 * F1(ROUGE-L)
```

`eval/rouge.py` 为自实现版本，不使用第三方 `rouge` 包，以保证口径可控、分词方式可校准。

### 法条语料（RAG 知识库）

`data/raw/laws/` 下为 12 个分类、共 1385 个法条文本（Markdown）。

**关键发现**：统计数据集中文书实际引用的法律后，结果与直觉相反：

| 法律 | 被引用篇数 | 所属目录 |
|---|---|---|
| 民事诉讼法 | 11617 (86%) | 诉讼与非诉讼程序法 |
| 合同法 | 6288 (47%) | **停用** |
| 担保法 | 1976 | **停用** |
| 继承法 | 1662 | **停用** |
| 侵权责任法 | 1478 | **停用** |
| 民法通则 | 1147 | **停用** |
| 物权法 | 749 | **停用** |
| 婚姻法 | 357 | **停用** |
| 民法总则 | 251 | **停用** |
| 民法典 | 0 | 民法典 |

CAIL2020 是民法典生效（2021-01-01）**之前**的判决书，因此文书引用的全是旧法。

结论：`停用/` 目录（被民法典取代的旧法）其实是本任务**最核心**的检索语料，
而 `民法典/` 的检索价值接近于零。构建检索语料时应按"数据集中实际被引用的法律"来筛，
而不是按现行法律的完整性来筛。

---

## 环境

### 本地开发（CPU）

```bash
conda activate minimind
```

解释器路径：`/home/yukino/anaconda3/envs/minimind/bin/python`

* `data/` 与 `eval/` 只用标准库实现，**不装重依赖、不开 GPU** 即可跑通数据管线与评测。
* 本地 GPU **不可用**：环境内 torch 为 `2.14.0+cu130`，而本机驱动仅支持到 CUDA 12.9，
  `torch.cuda.is_available()` 返回 `False`。本地一律按 CPU 调试处理。

### 训练环境（Kaggle）

2 × Tesla T4（16GB，Turing, sm_75）。约束见 `requirements-kaggle.txt`，其中三条最关键：

1. **不要重装 torch** —— 装错版本会毁掉环境。
2. **T4 不支持 bf16** —— 必须 fp16 + GradScaler。
3. **默认无外网** —— 需手动开启，或把权重挂成 Kaggle Dataset。

本地调试模型与线上模型的分工：

| 用途 | 模型 | 验证什么 |
|---|---|---|
| 训练逻辑 smoke test | MiniMind（本机已有） | loss mask、梯度累积、checkpoint 存取 |
| HF 集成链路 | Qwen2.5-0.5B-Instruct | 模板、generate 参数、评测闭环 |
| 正式实验 | ChatGLM3-6B | QLoRA 全流程 |

---

## 目录结构

```
configs/           # 配置驱动，换模型/换阶段只改配置
data/
  raw/             # sfzy_cail.json（不入库）+ laws/（法条语料，入库）
  processed/       # 规范化后的 JSONL
  index/           # 检索索引
src/sfzy/
  data/            # 解析、规范化、prompt 模板、collator
  models/          # 模型加载、对话模板、手写 LoRA
  sft/             # SFT 训练循环、断点续训、推理
  rag/             # 语料、分块、BM25、稠密检索、融合、重排
  rl/              # 规则奖励、偏好数据、DPO、GRPO
  eval/            # ROUGE、指标、批量评测、报告
scripts/           # 薄 CLI 入口
tests/             # 测试即验收标准
```

---

## 里程碑

| 阶段 | 内容 | 产出 |
|---|---|---|
| M0 | 工程基建 | 配置、脚手架、调试样本生成 |
| M1 | 数据层与评测地基 | schema、prompt、ROUGE |
| M2 | SFT | collator、LoRA、训练循环 |
| M3 | RAG | BM25 + 稠密检索 + 融合 |
| M4 | RLHF | 规则奖励 + GRPO / DPO |
| M5 | Kaggle 全量实验 | 消融与最终报告 |

数据使用策略：**先用小子集把全流程跑通，确认可行后再上全量数据。**

---

## 快速开始

```bash
conda activate minimind

# 1. 生成开发用的小样本（不必等全量数据）
python tools/make_sample_data.py --n 200

# 2. 规范化数据
python scripts/prepare_data.py --config configs/data.yaml

# 3. 跑测试（验收标准）
python -m pytest -q

# 4. 评测预测结果
python scripts/evaluate.py --pred outputs/pred.json --ref data/processed/dev.jsonl
```

---

## 实验结论

（待补）
