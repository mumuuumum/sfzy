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

单卡实测 9.4 秒/样本（200 条冒烟：151 秒/优化步 ÷ 16 条/步），
全量训练集 10738 条要 **28 小时** —— 超过单次 10 小时，也超过每周 30 小时的配额。
所以双卡不是"想更快"，是**必须做**。

### 速度预期：接近 2×，但不是精确 2×

LoRA 让 DDP 的效率很高：**需要 all-reduce 的梯度只有 7.4M 个参数 ≈ 30MB**
（r=4 打在 qkv/dense/dense_h_to_4h/dense_4h_to_h 上），
而 T4 之间走 PCIe 传 30MB 是大材小用，通信开销几乎可以忽略。

损失的 10~30% 来自：数据加载的串行部分、每步的同步等待、以及一个 batch 里
长短样本混杂导致的卡间负载不均。

> 反过来说，**如果换成全参数微调，梯度是 12GB 量级，PCIe 就会变成瓶颈**，
> 加速比会明显下降。LoRA 在这里帮了大忙。

### DDP 到底在每张卡上放什么

**数据并行 = 每张卡一份完整的模型副本**（权重、LoRA 参数、优化器动量、梯度），
只有喂进去的数据不同；每一步结束用 all-reduce 把各卡的梯度求平均。

所以有三件事必须记住：

1. **每卡显存 ≈ 单卡训练的显存**，不会因为多了张卡就变小。
   这就是 `load_model(..., local_rank=)` 要用 `device_map={"": local_rank}`
   把整个模型钉在本进程自己那张卡上的原因 —— 用 `auto` 会变成"模型并行"，
   把层摊到两张卡上，跟 DDP 的语义正好相反（每个进程只有一部分层，没法各自前向）。
2. `per_device_batch_size` 是**每张卡**的批量。全局批量
   = per_device_batch_size × grad_accum_steps × 卡数。
3. **单步不会变快**：DDP 提速的方式是"同样时间处理两倍数据"。
   一次优化步的耗时和单卡一样（再加几个百分点的通信），
   但它现在吃掉两倍的样本 —— 所以按 epoch 算的墙钟时间才接近减半。

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

`--dev-limit 50` 不是可选项：验证集有 1340 条，每轮到 epoch 结束都要完整跑一遍
（约 1 秒/条，单卡 20 分钟左右），冒烟阶段没必要付这个时间。
正式训练时才放开（它同时决定 `best.pt` 是什么时候存的）。

正式训练去掉 `--limit` / `--dev-limit`：

```python
!python -m torch.distributed.run --nproc_per_node 2 --standalone \
    scripts/train_sft.py --config configs/sft_kaggle.yaml
```

### batch 怎么配

单卡实测 11 GiB/14.6 GiB，只剩约 4 GiB 余量。DDP 不会让这个余量变宽：
每张卡都要放**一整份** 4-bit 权重（约 3.5 GiB）+ 未被量化的 fp32 词表/输出层
（0.54 B × 4 B ≈ 2.1 GiB）+ 自己的激活。

```yaml
per_device_batch_size: 2      # 每卡 2 条
grad_accum_steps: 8           # 等效 batch = 2 × 8 × 2卡 = 32
```

**`per_device_batch_size` 停在 2，不要再往上调。** 原因不是权重，而是
**词表投影**：ChatGLM3 的 vocab 是 65024，官方 `modeling_chatglm.py` 里

```python
lm_logits = lm_logits.to(torch.float32)      # fp16 → fp32，多一份
shift_logits = lm_logits[..., :-1, :].contiguous()   # 又一份 fp32 拷贝
```

一次前向里同时存在 fp16 logits、fp32 logits、fp32 的 shift 副本，
反传时还有同尺寸的 fp32 梯度，合计约 `B×L×0.9 MB`：

| 配置 | 均值长度（B×L≈2×2000） | p95 长度（B×L≈2×3500） |
|---|---|---|
| B=2 | ≈ 3.6 GiB | ≈ 6.3 GiB |
| B=4 | ≈ 7.2 GiB | ≈ 12.6 GiB |

加上 5 GiB 静态权重，B=4 在 15 GiB 的卡上必然 OOM（长样本批更早）。
想验证就跑：

```python
!python scripts/check_memory.py --batch-size 4 --seq-len 3500
```

判据：**"backward 后"的余量 < 1 GiB 就不要用**。9 小时会话里必然遇到长样本批，
而 OOM 会挑那种批发作 —— 你不会想在第 150 步丢掉整个会话。

要更大的等效批量就加 `grad_accum_steps`：梯度累积和真 batch 在数学上等价
（我们没有 BatchNorm，只有 LoRA 的 dropout 分布略有区别），代价只是慢一点。

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

---

## 参数依据（实测 + 推算，改配置前先看这里）

### 一次会话能跑多少

| 量 | 数值 | 来源 |
|---|---|---|
| 每优化步耗时（单卡，16 条/步） | 151 s | 06:22–06:54 的 13 步冒烟 |
| 每样本耗时 | 9.4 s | 151 ÷ 16 |
| 每优化步耗时（双卡，32 条/步） | ≈ 160 s | 单卡步时 + 几个百分点的 all-reduce |
| 9 小时会话 | ≈ 190 步 ≈ **6100 条** | 扣除 clone / 装依赖 / 加载模型的约 20 分钟 |
| 全量一轮 | 336 步 ≈ **15 小时** | 10738 ÷ 32 |

**结论：一轮训练要跨两次 Kaggle 会话。** 这不是意外情况，是常态，
所以 `resume_from` + 每 40 分钟落一次盘是这套配置的核心，不是补丁。
第一次会话结束前记得把 `outputs/` 存成 Dataset（Kaggle 的 Save Version），
下一次会话挂回来再用 `--resume` 指到具体的 `step_*.pt`。

### 存档间隔为什么是 15 步 / 45 分钟

目标只有一个：**会话被掐的时候，最新的 checkpoint 离被掐点尽可能近**。

* Kaggle 的 9 小时从 **notebook 启动**算起，不是从训练开始算 ——
  所以按"墙钟时间"存档比按"步数"存档更贴近真实限制，
  `save_every_n_minutes: 45` 就是干这个的（两者是「或」的关系）。
* `save_every_n_steps: 15` ≈ 40 分钟一存，作用是在训练比预估快时也不至于
  攒太久才落一次盘。
* 最坏情况丢失 40~45 分钟 ≈ 15 步 ≈ 480 条。相对 15 小时的总量是 5%。
* 单个 checkpoint 只存可训练参数 + 优化器动量 ≈ 90 MB，写盘几秒，
  所以"存得勤"几乎没有代价 —— 真正有代价的是**丢掉 8 小时的成果**。
* `keep_last_n_checkpoints: 3` 给出 90 分钟冗余；`best.pt` 是固定文件名，
  永远不会被按数量清理。

### 关键超参

| 参数 | 值 | 依据 |
|---|---|---|
| `per_device_batch_size` | 2 | 实测 11 GiB/14.6 GiB；B=4 的词表投影要多吃 ~7 GiB，必 OOM |
| `grad_accum_steps` | 8 | 等效 batch = 2×8×2卡 = 32，与 2025-03 那版 LLaMA-Factory 一致 |
| `learning_rate` | 2.0e-4 | QLoRA 推荐带宽 1e-4~3e-4 的中上值；只跑 1 轮、步数少，取偏大一侧。**不按 batch 线性放大**（线性缩放律是全参微调的经验） |
| `max_length` | 8192 | 模型自身 `seq_length`；实测 prompt+answer 的 p99 只有 4741 token，截断率 0.05% |
| `lora.r` / `alpha` | 4 / 8 | scaling = alpha/r = 2，与原 `16/32` 相同 → 学习率不用重调 |
| `target_modules` | qkv + dense + dense_h_to_4h + dense_4h_to_h | 等价于 LLaMA-Factory 的 `lora_target: all`（7.4M 可训练参数，占 6.25B 的 0.119%） |
| `save_every_n_steps` / `_minutes` | 15 / 45 | 见上 |

两个与 ChatGLM3 结构绑定的数字，换模型时一定要重算：

* `query_key_value` 是**融合**的 QKV（out = 4096 + 2×128×2 = 4608），
  不是 `q_proj/k_proj/v_proj` 三个独立层；
* MLP 是 SwiGLU，`dense_h_to_4h` 的输出维度是 `ffn_hidden_size * 2 = 27392`，
  而 `dense_4h_to_h` 的输入是 13696 —— 两个形状不对称。

按这几个形状算出来的 LoRA 参数量，和 Kaggle 日志对得上：
r=16 只打 `query_key_value` = 3,899,392（日志里的 `trainable_params`），
r=8 = 1,949,696（`check_model.py` 打印的 1.95 M），
r=4 打满四个目标层 = 7,411,712（占 6.25 B 的 0.119%）。
