# AutoDL 部署：2× RTX 4090

## 一、选服务器时的镜像要求

选镜像时看这四项，**任何一项不满足都别开**：

| 项 | 要求 | 为什么 |
|---|---|---|
| PyTorch | **≥ 2.1** | 4090 是 Ada (sm_89)，太老的 torch 不支持这个架构 |
| CUDA | **12.1 ~ 12.4** | 下限是 sm_89 的要求；上限别碰 13.x——社区库（vLLM、bitsandbytes）对 13 的支持还不完整，而你的驱动未必跟得上 |
| Python | **3.10 ~ 3.12** | 和我们的代码一致（本机是 3.12.7） |
| 基础镜像自带的 transformers | **无所谓，反正要重装** | 镜像里常见 4.5x/5.x，ChatGLM3 在 5.x 上加载不了 |

典型命名形如：`PyTorch 2.x.x / CUDA 12.x / Python 3.10 / Ubuntu 22.04`

**不要选**：CUDA 11.x（对 4090 支持不完整）、CUDA 13.x（生态没跟上）、
miniconda 里预装了 transformers 5.x 的镜像（会覆盖麻烦）。

## 二、磁盘与路径

AutoDL 的系统盘只有 30~50GB，**12GB 的模型权重放不下**。数据盘挂在
`/root/autodl-tmp`（通常 50~100GB），所有大文件都放这里：

```bash
mkdir -p /root/autodl-tmp/sfzy
export PROJECT=/root/autodl-tmp/sfzy
cd $PROJECT
```

**注意**：AutoDL 的"无卡模式开机"可以省钱（约 ¥0.1/h），适合传数据、
下模型、配环境。**下 12GB 模型时用无卡模式，不要开着 GPU 等。**

## 三、环境准备

```bash
cd /root/autodl-tmp/sfzy
pip install -r requirements-cloud.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`requirements-cloud.txt` 里刻意不包含 torch 和 bitsandbytes，原因写在文件里了。

验证环境：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
# 期望：2.x.x / 12.x / True / 2
python -c "import torch; print('bf16:', torch.cuda.is_bf16_supported())"
# 期望：True  —— 这是选 4090 的核心红利
```

## 四、模型下载（国内最慢的一步）

HF 在国内云上常常不通。**用 ModelScope**（魔搭），ChatGLM3 有官方镜像：

```bash
python tools/download_model.py \
    --repo-id ZhipuAI/chatglm3-6b \
    --backend modelscope \
    --local-dir models/chatglm3-6b
```

12GB 大约几分钟。**这一步用无卡模式做**，不烧 GPU 时长。

如果一定要走 HF，AutoDL 有"学术资源加速"（控制台里开启），
开启后 `HF_ENDPOINT=https://hf-mirror.com` 也能用。

## 五、连通性检查（别跳）

```bash
python scripts/check_model.py --config configs/model_chatglm3_6b_bf16.yaml
```

重点看两行：

```
[4/5] 检查 LoRA 目标层是否存在
      匹配 28 层，可训练参数 XX M
[5/5] 前向一次，确认端到端能跑
```

**如果 transformers 版本和 ChatGLM3 不兼容，这里就会暴露。** 那时候
按报错降 transformers 版本重装，比训练到一半才发现要划算得多。

## 六、冒烟测试（决定 batch 开多大）

```bash
python scripts/train_sft.py --config configs/sft_cloud.yaml \
    --limit 200 --override sft.save_every_n_steps=20 \
    --override sft.output_dir=outputs/smoke
```

跑的过程中开另一个终端看显存：

```bash
watch -n 2 nvidia-smi
```

**要拿到两个数**：

1. **显存峰值** —— 如果离 24GB 还有很大余量，把 `per_device_batch_size`
   往上加（4 → 6 → 8），每次加完重跑冒烟
2. **每步耗时** —— 用它估算全量训练时间：
   `总时间 ≈ 10738 / (batch × accum) × 每步耗时`

撑不住时的降级顺序（**先降序列长度，它对显存影响最大**）：

```
max_length 2048 → 1536 → 1024
per_device_batch_size 4 → 2（同时 grad_accum_steps 翻倍，保持等效 batch）
```

## 七、正式训练

```bash
python scripts/train_sft.py --config configs/sft_cloud.yaml 2>&1 | tee outputs/train.log
```

## 八、双卡怎么用

**SFT 阶段只用一张卡。** 理由：

- LoRA 的梯度同步量很小，DDP 的实际加速到不了 2x（PCIe 通信 + 启动开销），
  而我们的 trainer 目前不支持 DDP，加它要改 `DistributedSampler`、
  启动方式、checkpoint 的 rank 判断，是一整块工作
- 单卡 batch 4 已经能把 24GB 用满，双卡的好处主要是"更大的等效 batch"，
  而这对 LoRA 的收益有限

**第二张卡的真实用途在后面**：

- 跑 `check_model.py` 验证 vLLM 的 LoRA adapter 格式
- GRPO 阶段：卡 0 跑 policy 的 rollout，卡 1 跑 judge 打分，**真并行**

所以现在开双卡，第二张卡先闲着是可以接受的；如果预算紧，
SFT 阶段开单卡也行，等 GRPO 再换双卡。
