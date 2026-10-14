# 三要素抽取（LLM API）+ 逐要素 ROUGE

用大模型 API 从**人工摘要**和**候选摘要（含 GRPO 输出）**里各抽三个要素，
再逐要素算 ROUGE。规则版切分器（`sfzy.segment` / `docs/segmenter.md`）保留在仓库里，
两条路可以互相验证。

| 文件 | 职责 |
| --- | --- |
| [configs/element_extract.yaml](../configs/element_extract.yaml) | 端点、**你要写的 prompt**、ROUGE 权重、缓存路径 |
| `src/sfzy/elements/schema.py` | 三要素结构 + 容错解析（围栏/多余文字/null/列表） |
| `src/sfzy/elements/prompts.py` | prompt 模板（`{summary}` 占位符）与指纹 |
| `src/sfzy/elements/client.py` | 环境变量文件补读、HTTP 客户端、**报错翻译** |
| `src/sfzy/elements/extract.py` | 批量抽取 + 缓存 + 单条失败隔离 |
| `src/sfzy/elements/score.py` | 逐要素 ROUGE（权重来自配置） |
| `src/sfzy/elements/pipeline.py` | 串起来 + 宏平均 |
| `tools/element_rouge.py` | 命令行入口 |

## 1. 跑法

```bash
# 0) 探活：验证 base_url / 模型 / 鉴权，并真抽一条看能不能解析出三个键
python tools/element_rouge.py --config configs/element_extract.yaml --ping

# 1) 抽几条看看 prompt 效果（不落盘）
python tools/element_rouge.py --config configs/element_extract.yaml \
    --cand grpo=data/triples/grpo_val_v48.jsonl --limit 5 --show 5 --no-write

# 2) 多候选对比（人工摘要取自 --ref，没有则用候选文件里的 reference 字段）
python tools/element_rouge.py --config configs/element_extract.yaml \
    --ref data/splits/val.jsonl \
    --cand sft=data/triples/sft_val_shard*of2.jsonl \
    --cand grpo=data/triples/grpo_val_v48.jsonl \
    --out outputs/element_rouge/val_sft_vs_grpo
```

产物：`report.md`（人看）、`summary.json`（机读）、`scores.jsonl`（逐样本：人工要素、
候选要素、逐要素分数）。人工摘要的抽取结果写进 `cache_path`，候选摘要现算。

## 2. prompt 是你写的，但有两条硬契约

1. `user` 模板里必须有 `{summary}` 占位符——少了会在构造时直接报错，
   不会安静地把摘要丢掉；其它花括号（JSON 示例）原样保留。
2. 要求模型只输出 JSON，键名固定 `plaintiff_claims` / `defendant_defenses` /
   `court_facts`，值为字符串。

建议写"**逐字复制**（连标点、数字都照抄）"：下游是按字面算 ROUGE 的，
模型一旦改写、总结、或顺手去掉标点，量到的就不只是"两份摘要像不像"。

## 3. 用哪个模型：默认用裁判那套（qwen3.7-flash）

同一个 prompt、同一批摘要，各跑几轮（`outputs/bench_extract_models/report.md` 与直测）：

| 模型 | 端点 / 密钥 | 单条中位 | 最慢 | 吞吐 | 解析 |
| --- | --- | --- | --- | --- | --- |
| **`qwen3.7-flash`**（默认） | maas.qianwenaiapi.com / `$SFZY_JUDGE_API_KEY` | **1.3s** | **1.5s** | **204~320 次/分钟** | 15/15 |
| `teleai/xing4.0-29b-a4b`（备选，注释里） | api.tokenplus.cn / `$SFZY_ELEMENT_EXTRACT` | 7.7s | 36.4s | 11 次/分钟 | 15/15 |

差约 18 倍，**关键差别在长尾**（teleai 会突然一条卡到几十秒甚至超时；全量 2680 次调用时
这就是"卡不卡住"的问题）。两者 token 量级相当（prompt ~460/条、completion ~140/条）。
切换只是 `api:` 那一段的四行（备选块就在配置里，注释状态）。

注意 `qwen3.7-flash` 属于 Qwen3 系，**默认开思考**，必须用 `extra_body.enable_thinking: false`
关掉；而 `teleai` 那个供应商只认 `reasoning_effort: none`（写 `enable_thinking` 会被 400 拒）。

实测（20 条样本、40 次新调用）：**7.5 秒**跑完，抽取失败 0，
总分 0.5458（原告诉请 0.6683 / 被告辩称 0.5152 / 法院事实 0.4846）。

## 4. 踩过的三个坑（都已处理，改配置即可）

### 4.1 推理模型：思维链吃掉预算，`content` 是空的

备选模型 `teleai/xing4.0-29b-a4b` 会把思考写进 `reasoning_content`。`max_tokens` 给小了
就会出现"思维链写满、`content` 为空、`finish_reason=length`"——**看起来像模型什么都没输出**。

* 配置里 `max_tokens: 4096`；
* 客户端会读 `reasoning_content` 与 `finish_reason`，并给出具体诊断
  （"模型只输出了思维链…被 max_tokens 截断…"），而不是笼统的"解析失败"；
* 思维链里恰好写了 JSON 时会被救回来（但会记下来源）。

### 4.2 关思维链：两个供应商的开关名字不一样

同一段摘要、`teleai` 那个供应商（`api.tokenplus.cn`）的实测：

| 开关 | 耗时 | 思维链长度 | 结果 |
| --- | --- | --- | --- |
| 不改（基线） | 35.0s | 2121 字 | — |
| `enable_thinking: false` | 1.7s | — | **HTTP 400**：供应商要求用 `thinking`/`reasoning_effort` |
| `chat_template_kwargs.enable_thinking` | 66.8s | 2108 字 | 被忽略 |
| prompt 里加 `/no_think` | 34.9s | 2105 字 | 无效 |
| `thinking: {type: disabled}` | 127.5s | 0 | 有效但更慢 |
| **`reasoning_effort: none`** | **6.0s** | **0** | ✅ 抽出来的 JSON 与基线完全一致 |

所以**备选块**里写的是：

```yaml
    extra_body:
      reasoning_effort: none
```

效果：**35s → 6s**，输出 token 从 ~1960/条 降到 ~155/条。而默认的 `qwen3.7-flash`
用 `enable_thinking: false`（见配置注释），它本来就不产思维链。

### 4.3 批量粒度与并发：别按样本发

第一版是"逐条样本"各发一次（人工 + 该样本的候选），等于把并发压成 1~2：
20 条样本跑了 10 分钟以上。改成**先把所有人工摘要 + 所有候选摘要攒成一批发出去**，
再逐条打分。另外：

* **单条失败要隔离**：40 条里一条超时，另外 39 条的结果必须留下（早先的实现会让整批失败）；
* **超时给足**：推理模型单条 20~60s，配置里 `timeout_s: 180`；
* **并发不是越高越好**：在 `teleai` 那条路上实测 16 并发 ≈ 21 条/分钟，
  32 并发反而掉到 6 条/分钟（供应商侧排队/限流）。默认的 `qwen3.7-flash` 没这个问题，
  配置里给的是 16。

实测（`data/splits/val.jsonl` + `data/triples/grpo_val_v48.jsonl`）：
20 条样本 40 次新调用 **7.5 秒**跑完。

## 5. 打分的口径

每项取 F1，总分三项加权（**权重写在配置里**，代码不存默认值）：

```
score(要素) = w1·F1(ROUGE-1) + w2·F1(ROUGE-2) + wL·F1(ROUGE-L)
```

* 默认（配置模板里）是官方口径 `0.2 / 0.4 / 0.4`；权重按和归一，写 `2/4/4` 结果一样；
* `mode`: `jieba`（训练/奖励侧口径）或 `char`（官方评测脚本口径）；
* 三个要素再按 `element_weights` 合成总分，默认沿用 judge 六要素里这三项的比例
  `0.30 / 0.20 / 0.50`（可省略）。

空要素的三种情形分开统计，避免把口径问题算成质量问题：

| 情形 | 处理 |
| --- | --- |
| 两侧都空（人工摘要本来没有被告辩称） | `exclude`：不计入平均；按计入的要素重新归一权重 |
| 只有一侧空 | `one_side_empty_score`（默认 0.0） |
| 抽取失败（`ok=False`） | **不进分数**，单独计数，并打印失败原因 |

## 6. 缓存

键 = `sha1(正文 + 模型名 + prompt 指纹)`。三者缺一不可：

* 换了模型 → 重抽；
* **改了 prompt → 旧缓存自动失效**（这是最容易静默出错的地方）；
* 只缓存**成功**的抽取（失败被缓存下来，之后每次跑都会复现同一个空结果）；
* 候选摘要也会命中缓存（同一条候选被多个实验评到时就省下来了）。

## 7. 成本与时间估计

按实测（`qwen3.7-flash`，关思考）：单条 ~140 completion tokens、~460 prompt tokens；
16 并发 **200+ 条/分钟**。

* 全量 val（1340 条 × 2 = 2680 次调用）≈ **10~15 分钟**；
* 之后再评别的候选：人工摘要那 1340 次全部命中缓存，只需付候选那一半；
* 缓存键含模型名：换模型（例如切回 teleai）会重抽，不会拿旧结果顶。

## 8. 测试

```bash
python -m pytest tests/test_elements.py -q     # 35 条：解析容错 / 权重口径 / 缓存 / 环境变量 / 推理模型
```

全部不联网：用一个假 runtime 顶替 API（`FakeRuntime` / `FakeHTTPRuntime`）。
