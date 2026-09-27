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

## 第一步：数据走 Dataset，代码走 Git

**这两样东西的更新频率差了两个数量级，所以同步方式必须分开。**

| | 数据 | 代码 |
|---|---|---|
| 体积 | 105MB | 150KB |
| 变化频率 | 几乎不变 | 一天改十几次 |
| 同步方式 | **Kaggle Dataset** | **Git 仓库** |

一开始我们把代码也打包成 zip 塞进 Dataset，结果是每改一行都要重新打包、
重新上传、等几分钟。中间还图快改用"在 Kaggle 上就地打补丁"，
结果攒了六七个补丁，和本地代码越对越不上 —— 这是个错配：
**Dataset 是为大数据设计的，版本化慢、只读、没有 diff**。

### A. 数据（Kaggle Dataset）

本地执行：

```bash
python tools/make_splits.py     # → data/splits/{train,val,test}.jsonl  (105MB)
```

建一个 Kaggle Dataset（命名比如 `sfzy-sft`），**只放这三个文件**：

```
train.jsonl
val.jsonl
test.jsonl
```

> 挂载后的实际路径是 `/kaggle/input/datasets/<用户名>/<数据集名>/`（Kaggle
> 较新的格式，早期是 `/kaggle/input/<数据集名>/`）。下面 notebook 里
> **不写死路径，按文件名自动探测**，两种格式都能用。

### B. 代码（Git 仓库）

本地执行一次：

```bash
cd ~/projects/sfzy
git remote add origin git@github.com:<你的用户名>/sfzy.git
git push -u origin main
```

之后每次改完代码：

```bash
git add -A && git commit -m "..." && git push
```

Kaggle 那边 `git clone`（格 1）。**代码版本就是 commit hash**，
实验记录里写清楚用的是哪个 commit，结果就可复现了。

> **建议公开仓库**：一是 Kaggle clone 时不用配 token，二是这是个面试项目，
> GitHub 本身就该给人看。
> 如果要用私有仓库，clone 时用 `https://<token>@github.com/user/repo.git`，
> token 放 Kaggle 的 Secrets 里而不是硬编码在 notebook。

---

## 第二步：建 notebook 并配置

New Notebook，然后在右侧设置面板里：

- **Accelerator**: `GPU T4 x2`（必须是 x2，单卡也能跑但慢一倍）
- **Internet**: `On`（要下载 ChatGLM3）
- **Persistence**: `Files only`（保住 /kaggle/working）

---

## 第三步：环境准备（notebook 里逐格跑）

### 格 1 — 拉代码、链接数据

```python
import pathlib, shutil, subprocess, os

REPO = "https://github.com/<你的用户名>/sfzy.git"     # ← 改这里
CODE = pathlib.Path("/kaggle/working/sfzy")

# 每次会话都是新的容器，所以每次都要重新 clone（代码 150KB，一两秒）
if CODE.exists():
    shutil.rmtree(CODE)
subprocess.run(["git", "clone", "--depth", "1", REPO, str(CODE)], check=True)

REV = subprocess.run(["git", "-C", str(CODE), "log", "--oneline", "-1"],
                     capture_output=True, text=True).stdout.strip()
print("代码版本:", REV)     # ← 把这一行记进实验记录，结果才可复现

# 数据仍然从 Dataset 挂载。路径格式变过两次，所以按文件名自动探测：
#     /kaggle/input/<数据集名>/
#     /kaggle/input/datasets/<用户名>/<数据集名>/
train_jsonl = next(pathlib.Path("/kaggle/input").rglob("train.jsonl"))
DATA = train_jsonl.parent
print("数据目录:", DATA)

# 数据是只读的，软链到代码期望的相对路径（configs/data_kaggle.yaml 写的是 data/splits）
splits = CODE / "data" / "splits"
splits.mkdir(parents=True, exist_ok=True)
for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
    link = splits / name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(DATA / name)

print("就绪:", sorted(p.name for p in splits.iterdir()))
```

> **同一会话里想用最新代码**：本地 push 之后，在 notebook 里跑
> `!git -C /kaggle/working/sfzy pull` 即可，不用重跑整格。
>
> **不建议在 Kaggle 上直接改代码** —— 那样又会出现"两边不一致"的老问题。
> 临时调参可以改 `configs/`（不提交），代码改动一律走本地 → push → pull。

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

# 不要把 HF_HOME 设到 /kaggle/working！
#
# 我原来写的理由是"缓存跟着 Save Version 留下来，下次不用重下 12.5GB"，
# 但实测 Kaggle 下载这个模型只要 **36 秒**（385 MB/s）。
# 而 /kaggle/working 只有 20GB 的保存配额，塞 12.5GB 的模型缓存进去，
# checkpoint 和日志就没多少空间了。
#
# 所以用默认缓存（/root/.cache/huggingface）更划算：每次会话重下，36 秒。
print("HF_HOME =", os.environ.get("HF_HOME", "（未设置，用默认 /root/.cache）"))
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
| 6 | **训练时 OOM，但报错栈里全是 bitsandbytes** | **两个原因叠加**（见下） | `compat.py::ensure_gradient_checkpointing` + `trainer.train()` 开头的 `self.model.train()` |

**这一组的共同特征是：根因全部在 ChatGLM3 的远程代码里，不在我们的代码里。**
它们的代码停留在 2023 年，而 transformers 一直在演进。

### 第 6 个坑：两个原因叠加，而且一个在我们的代码里

它是**唯一一个不报自己名字的**。前五个都有明确的异常信息；这个只报 OOM，
而调用栈一路穿过 `bnb.matmul_4bit` → `gemm_4bit` → `_dequant_linear_fallback`，
**看起来像是在量化库里炸的**。

ChatGLM3 的 `GLMTransformer.forward` 里，梯度检查点的条件是：

```python
if self.gradient_checkpointing and self.training:   # ← 两个条件缺一不可
    layer_ret = torch.utils.checkpoint.checkpoint(layer, ...)
```

**两个条件我们各踩了一个：**

**原因一，flag 没设上（ChatGLM3 的锅）。** 它的
`ChatGLMPreTrainedModel.gradient_checkpointing_enable` 重写了基类方法，
但**函数体是空的** —— 只做了 `supports_gradient_checkpointing` 校验就返回，
一个 flag 都没设。所以 `prepare_model_for_kbit_training` 调完之后，
`GLMTransformer.gradient_checkpointing` 仍然是 `False`。

**原因二，模型不在 train 模式（我们的锅）。** HF 的 `from_pretrained`
结尾会调 `model.eval()`（它自己的文档里写着 "The model is set in evaluation
mode by default"），而**我们的 `SFTTrainer.train()` 从来没有切回 train 模式** ——
`model.train()` 只出现在 `evaluate()` 的 finally 里。

所以 `self.training` 全程是 `False`，条件短路。

> **顺带的影响**：dropout 也全程关闭了，正则化失效 —— 同样不报错。

**判据只能靠对比**：LLaMA-Factory 用同一个模型、**同样 `cutoff_len=8192`、
batch 还大一倍**（`per_device_train_batch_size: 2`），在同样的 2×T4 上跑得通。
序列一样长、batch 更大却没事，只能是梯度检查点这个数量级的差异。

**这一坑我们连续猜错了两轮**（先猜 `device_map`、再猜 `max_length`），
最后是去读 ChatGLM3 的 `modeling_chatglm.py` 源码才确认的。
教训是：**别猜，去读源码；写了修复，就用测试证明它能抓住 bug。**

相应的两处可见化：
- `check_model.py` 会把梯度检查点的**实际状态**整个打印出来（它之前用
  `gradient_checkpointing=False` 加载，结构上就不可能发现这件事）
- `test_smoke_train.py::test_训练前把模型切回train模式` 用 forward hook 在
  **训练过程中**读 `model.training`（不能等训练结束再读，那时 `evaluate()`
  的 finally 已经把它切回来了，会把 bug 掩盖掉）

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

---

## 双卡（DDP）训练

单卡实测 10.4 秒/样本，全量训练集 10738 条要 **31 小时** —— 超过单次 10 小时，
也超过每周 30 小时的配额。所以双卡不是"想更快"，是**必须做**。

### 速度预期：接近 2×，但不是精确 2×

LoRA 让 DDP 的效率很高：**需要 all-reduce 的梯度只有 3.9M 个参数 ≈ 15.6MB**，
而 T4 之间走 PCIe 传 15MB 是大材小用，通信开销几乎可以忽略。

损失的 10~30% 来自：数据加载的串行部分、每步的同步等待、以及一个 batch 里
长短样本混杂导致的卡间负载不均。

> 反过来说，**如果换成全参数微调，梯度是 12GB 量级，PCIe 就会变成瓶颈**，
> 加速比会明显下降。LoRA 在这里帮了大忙。

### 启动方式

```python
%cd /kaggle/working/sfzy-sft
!python -m torch.distributed.run --nproc_per_node 2 --standalone \
    scripts/train_sft.py --config configs/sft_kaggle.yaml \
    --limit 200 --dev-limit 50 \
    --override sft.num_epochs=1 \
    --override sft.per_device_batch_size=2 \
    --override sft.grad_accum_steps=4 \
    --override sft.output_dir=outputs/smoke_ddp
```

用 `python -m torch.distributed.run` 而不是 `torchrun` 可执行文件 —— 后者不一定在 PATH 里。

### batch 怎么配

单卡 11GB/15GB 只剩 4GB 余量，直接加 batch 有风险。但 DDP 下每张卡只需要放
一份完整的 4-bit 模型（3.5GB）+ 自己的激活，余量比单卡宽。

```yaml
per_device_batch_size: 2      # 每卡 2 条
grad_accum_steps: 4           # 等效 batch = 2 × 4 × 2卡 = 16
```

**等效 batch 保持 16 不变**是有意的：超参（学习率）不用重调，单卡的 smoke
结果还能当基线。有余量再把 `per_device_batch_size` 提到 4、`grad_accum_steps` 降到 2。

### 代码上做了什么

| 位置 | 改动 |
|---|---|
| `models/loader.py` | `load_model(..., local_rank=)` —— ≥0 时用 `device_map={"": local_rank}` 把整模型钉在本进程的卡上 |
| `utils/distributed.py` | `wrap_model()` 封装 DDP；`init_distributed` 只在 NCCL 时才 `set_device` |
| `sft/trainer.py` | `DistributedSampler` + `set_epoch`；`steps_per_epoch` 除以 `world_size` |
| `scripts/train_sft.py` | 串起来：加载 → 注入 LoRA → **再**包 DDP |

**三个容易错的点：**

1. **DDP 必须在注入 LoRA 之后包。** 先包再注入的话，新加的模块不在 DDP 管辖内，
   梯度不会同步，两张卡各训各的 —— 而且不报错。
2. **`steps_per_epoch` 必须除以 `world_size`。** 不然学习率调度会按单卡的数据量走，
   实际只走到 cosine 曲线的一半就结束了。同样不报错。
3. **`device_ids` 要看模型在哪，而不是 CUDA 可不可用。** 机器有 GPU 但模型在 CPU
   （DDP 的 CPU 测试就是这种情况）时会直接报错。

第 2、3 条都是 `tests/test_ddp_smoke.py` 抓出来的 —— 它用 **CPU + gloo** 起两个
进程，验证分片、梯度同步、步数折算。本机只有一张卡跑不了真双卡，但 DDP 的
正确性和后端无关，gloo 就能验。
