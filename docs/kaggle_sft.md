# 在 Kaggle 上跑 ChatGLM3-6B 的 QLoRA 微调

## 总览：三段式，别一次到位

Kaggle 每周只有 30 GPU 小时，一次会话上限 9 小时。所以**不要一上来就跑全量**：

| 阶段 | 成本 | 目的 | 通过判据 |
|---|---|---|---|
| ① 连通性检查 | 0 GPU 小时（CPU） | ChatGLM3 能不能被当前 transformers 加载 | `check_model.py` 全绿 |
| ② 冒烟测试 | ~0.5 小时 | 显存够不够、速度多少、能不能存盘 | 跑完 200 条且落了 checkpoint |
| ③ 正式训练 | 6~9 小时 | 拿到 SFT 模型 | 训练完成 + 产出已 Save Version |

**阶段 ① 是最容易被跳过、但代价最大的一步。** ChatGLM3 的远程代码是 2023 年写的，
和 Kaggle 上较新版本的 transformers 组合经常出问题。等下了 12GB 权重、跑了半小时
才发现不兼容，就白烧配额了。

---

## 第一步：准备两个 Kaggle Dataset

### 一个 Dataset 装两样东西

本地执行：

```bash
python tools/make_splits.py          # → data/splits/{train,val,test}.jsonl  (105MB)
python tools/make_kaggle_bundle.py   # → dist/sfzy-code.zip                  (130KB)
unzip -o dist/sfzy-code.zip -d dist/sfzy-code    # 解压成目录
```

然后建一个 Kaggle Dataset（命名比如 `sfzy-sft`），把下面这些放进去：

```
train.jsonl
val.jsonl
test.jsonl
sfzy-code/          ← dist/sfzy-code/ 的内容
```

**挂载后的实际路径是 `/kaggle/input/datasets/<用户名>/<数据集名>/`** —— 这是
Kaggle 较新的格式，早期是 `/kaggle/input/<数据集名>/`。两种都可能遇到，
所以下面 notebook 里**不写死路径，按文件名自动探测**。

> 代码上传成 zip 或目录都可以，格 1 里的探测逻辑对两种都成立。
> 每次本地改完代码都要重新打包并上传新版本。

---

## 第二步：建 notebook 并配置

New Notebook，然后在右侧设置面板里：

- **Accelerator**: `GPU T4 x2`（必须是 x2，单卡也能跑但慢一倍）
- **Internet**: `On`（要下载 ChatGLM3）
- **Persistence**: `Files only`（保住 /kaggle/working）

---

## 第三步：环境准备（notebook 里逐格跑）

### 格 1 — 部署代码、链接数据

```python
import glob, pathlib, shutil

INPUT = pathlib.Path("/kaggle/input")

# Kaggle 的挂载路径格式变过两次：
#     /kaggle/input/<数据集名>/
#     /kaggle/input/datasets/<用户名>/<数据集名>/
# 所以不写死路径，按内容自动找。
train_jsonl = next(INPUT.rglob("train.jsonl"))
DATA = train_jsonl.parent
print("数据目录:", DATA)

CODE_SRC = next(p for p in INPUT.rglob("sfzy-code") if p.is_dir())
CODE = pathlib.Path("/kaggle/working/sfzy")

# 必须拷到 /kaggle/working：/kaggle/input 是只读的，而训练要写 outputs/
# 顺便这也让你能在 Kaggle 上直接改 configs/ 而不用重新上传
if CODE.exists():
    shutil.rmtree(CODE)
shutil.copytree(CODE_SRC, CODE)
print("代码目录:", CODE_SRC, "->", CODE)

# 数据文件是只读的，软链到代码期望的相对路径（configs/data_kaggle.yaml 里写的是 data/splits）
splits = CODE / "data" / "splits"
splits.mkdir(parents=True, exist_ok=True)
for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
    link = splits / name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(DATA / name)

print("就绪:", sorted(p.name for p in splits.iterdir()))
```

### 格 2 — 装依赖、设缓存路径

```python
import os, importlib.metadata as md

# 必须加 -U！
# Kaggle 镜像**预装了** bitsandbytes，但版本可能低于 transformers 的要求。
# 不加 -U 的话 pip 会说 "Requirement already satisfied" 直接跳过，
# 然后在加载 4-bit 模型时抛
#     ImportError: Using bitsandbytes 4-bit quantization requires ...
# 你以为装好了，其实还是旧版本 —— 这是 pip 不升级已装包的经典陷阱。
!pip install -q -U "bitsandbytes>=0.46.1"

# 顺便补齐另外两个：loader.py 会 import peft（用它的
# prepare_model_for_kbit_training），device_map="auto" 需要 accelerate
!pip install -q -U peft accelerate

print("依赖版本：")
for pkg in ("torch", "transformers", "accelerate", "bitsandbytes", "peft", "datasets"):
    try:
        print(f"  {pkg:16s} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"  {pkg:16s} ✗ 未安装")

# 关键：把 HuggingFace 缓存放进 /kaggle/working，
# 这样 Save Version 之后模型权重能跟着 notebook 输出一起留下来，
# 下次会话挂载回来就不必重新下载 12GB
os.environ["HF_HOME"] = "/kaggle/working/hf"
print("HF_HOME =", os.environ["HF_HOME"])
```

> **如果装完还是报同样的 ImportError**，说明当前内核里已经载入了旧版
> bitsandbytes 的 C 扩展，需要 `Run → Restart Session`，然后**从格 1 重新跑**
> （重启会清掉 `/kaggle/working`，代码和数据要重新部署一次，但只有几秒）。

### 格 3 — 连通性检查（先跑这个！）

```python
%cd /kaggle/working/sfzy
!python scripts/check_model.py --config configs/model_chatglm3_6b.yaml
```

这一步会做五件事：加载 tokenizer、渲染 chat template、4-bit 加载模型、
检查 `query_key_value` 层名、前向一次。

**预期输出**（只看最后几行）：

```
[4/5] 检查 LoRA 目标层是否存在
      匹配 28 层，可训练参数 XX M
[5/5] 前向一次，确认端到端能跑
      logits 形状 (1, N, 65024)

✓ 全部通过。可以开始训练了。
```

**如果这一步失败**，先别往下走。最常见的三种失败：

| 报错 | 原因 | 处理 |
|---|---|---|
| `ImportError` / `AttributeError` 出现在 modeling_chatglm 里 | transformers 版本和 ChatGLM3 远程代码不兼容 | 降级 transformers，或换 ChatGLM3 的 ModelScope 版本 |
| `target_modules` 匹配 0 层 | ChatGLM3 的层名和预期不符 | 脚本会打印真实层名，把它填进 `configs/sft_kaggle.yaml` |
| `apply_chat_template` 失败 | ChatGLM3 的 tokenizer 没有这个接口 | 改用它的 `build_chat_input`，见 `models/chat_template.py` 的注释 |

### 已知兼容性问题（实际踩过的，按出现顺序）

这五个是我们真实遇到并修掉的。记在这里是为了换环境、换模型、重建
notebook 时能快速定位 —— 不用再花五个来回重新发现一遍。

| # | 报错 | 根因 | 修法 |
|---|---|---|---|
| 1 | `ImportError: ... requires bitsandbytes>=0.46.1` | Kaggle **预装**了 bitsandbytes，`pip install` 不加 `-U` 会被 "already satisfied" 跳过 | 格 2 里用 `pip install -U` |
| 2 | `AttributeError: 'ChatGLMConfig' object has no attribute 'max_length'` | 它的 modeling 代码读 `config.max_length`，而 config 里只有 `seq_length` —— 官方仓库两处字段名不一致 | `models/compat.py::patch_pretrained_config` |
| 3 | `AttributeError: ... no attribute 'all_tied_weights_keys'` | 新版 transformers 的量化路径要读这个属性，2023 年的远程代码没提供 | `models/compat.py::patch_tied_weights_keys` |
| 4 | `AttributeError: 'list' object has no attribute 'keys'` | 上面那条的补丁填错了类型 —— 同一属性 transformers 内部两处要求不同，**必须填 dict** | 同上，默认值写成 `{}` |
| 5 | `AssertionError` at `tokenization_chatglm.py:300 in _pad` | 它的 tokenizer 里写死了 `assert self.padding_side == "left"` | `models/compat.py::resolve_padding_side`，默认 left |

**这一组的共同特征是：根因全部在 ChatGLM3 的远程代码里，不在我们的代码里。**
它们的代码停留在 2023 年，而 transformers 一直在演进。

> **为什么本地发现不了？**
> 本机的 transformers 是 4.57.6，而 Kaggle 上装的是更新版本 —— 量化路径
> 的实现不同（比如 `get_keys_to_not_convert` 在两个版本里位于不同文件）。
> 所以**本地的 `check_model.py` 能验证逻辑，但验证不了版本兼容性**，
> 这一类问题只能在 Kaggle 上跑一次才会暴露。这也是为什么把
> `check_model.py` 放在流程的第一步、而不是直接开训练。

---

## 第四步：冒烟测试（200 条，约 30 分钟）

```python
%cd /kaggle/working/sfzy
!python scripts/train_sft.py --config configs/sft_kaggle.yaml \
    --limit 200 \
    --override sft.num_epochs=1 \
    --override sft.save_every_n_steps=20 \
    --override sft.output_dir=outputs/smoke
```

**这一轮要看三件事**（都在日志里）：

1. **模型概况**那一行里的 `trainable_ratio` —— 应该远小于 1%（LoRA 的正常范围）
2. **每步耗时** —— 从日志时间戳算。这是决定正式训练能覆盖多少数据的关键
3. **有没有 OOM** —— T4 只有 16GB，如果 batch=2 / max_length=1536 撑不住，
   按下面的顺序降：先降 `max_length` 到 1024 → 再降 `per_device_batch_size` 到 1
   （同时把 `grad_accum_steps` 翻倍，保持等效 batch 不变）

> ⚠️ **先看「模型概况」里的 `devices` 是不是只有一张卡。**
>
> ```
> devices: ['cuda:0', 'cuda:1']   ← 错！模型被摊到两张卡上了，训练时必 OOM
> devices: ['cuda:0']             ← 对
> ```
>
> 出现两张卡说明 `device_map` 是 `"auto"`。**`auto` 是给推理用的模型并行**，
> 它按「权重能不能放下」分配，完全不考虑训练还要放激活、梯度、优化器状态 ——
> 所以开训后必然把第二张卡顶爆，而报错信息里满是 bitsandbytes 的调用栈，
> 第一眼看不出是 device_map 的问题（**我们实际踩过，OOM 在 GPU 1**）。
>
> 修法：`configs/model_chatglm3_6b.yaml` 里改成
> ```yaml
> device_map:
>   "": 0        # 固定卡 0；卡 1 空着留给后面 judge 模型
> ```

**算一下正式训练能覆盖多少样本**：

```
单次会话可训练样本数 ≈ (9小时 × 3600) / (每样本秒数)
每样本秒数 = 每步耗时 / (per_device_batch_size × grad_accum_steps)
```

如果算出来覆盖不了 10738 条训练集，就**用 `--limit` 取一个子集**，而不是硬上导致中途被掐。

---

## 第五步：正式训练

```python
%cd /kaggle/working/sfzy
!python scripts/train_sft.py --config configs/sft_kaggle.yaml 2>&1 | tee outputs/train.log
```

训练过程中：

- checkpoint 会按 `save_every_n_steps` 落在 `outputs/sft_chatglm3/`
- `best.pt` 是验证集 loss 最好的那一轮（固定文件名，不会被 prune 掉）
- `step_XXXXXX.pt` 是定期存档

**被 9 小时上限掐断也没关系** —— 开新会话后重新执行格 1、格 2，
然后用断点续训：

```python
!python scripts/train_sft.py --config configs/sft_kaggle.yaml \
    --resume outputs/sft_chatglm3/best.pt
```

---

## 第六步：保存产出（**别忘了这一步**）

Kaggle 的 `/kaggle/working` 在会话结束后会清空，**必须 Save Version 才会保留**。

点右上角 `Save Version` → `Save & Run All (Commit)`。

保存后，这次会话的 `/kaggle/working` 会变成一个新的 Dataset，
下次可以通过 `/kaggle/input/<notebook-slug>/` 访问，里面有：

```
hf/                      HuggingFace 缓存（含 ChatGLM3 权重，下次不用重下）
sfzy/outputs/            checkpoint + 训练日志
```

如果不保存，下次会话要从零开始下模型、重跑训练。

---

## 常见问题

**Q: 一定要用 2×T4 吗？**
单卡也能跑，就是慢一倍。另外 2×T4 的第二张卡现在是空着的——后面接
judge 模型（LLM-as-a-judge）时会用到它，policy 在卡 0、judge 在卡 1，
两者真并行。

**Q: 为什么 dtype 必须是 float16？**
T4 是 Turing 架构，**不支持 bf16**。而 fp16 需要 GradScaler 配合
（梯度会下溢）。`configs/sft_kaggle.yaml` 里已经设好了。

**Q: 训练中断了，step 会重跑吗？**
不会。`resume` 会按 checkpoint 里的 step 跳过已完成的批次
（`train_sft.py` 用 `islice` 实现）。但前提是 data 的 shuffle 种子固定，
这一点代码里已经保证了。

**Q: 为什么 checkpoint 只存几十 MB？**
LoRA 只保存可训练的 adapter（A/B 矩阵），不存整个 6B 底座。
`checkpoint.py` 的 `only_trainable=True` 做的这件事。
