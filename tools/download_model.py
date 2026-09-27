"""把模型下载到项目目录内（不落到 ~/.cache）。

为什么要这个工具：
  1. huggingface.co 在国内直连不通，需要走 hf-mirror.com 镜像；
  2. 默认的 HF 缓存位置（~/.cache/huggingface/hub）不在工作区内，
     模型文件散落在外面既不好清理，也不方便打包带走。
     用 snapshot_download(local_dir=...) 会把文件平铺到一个普通目录里，
     from_pretrained 直接指向它就行，不需要任何环境变量。

用法：
    # 本机（HF 直连不通、走 hf-mirror）
    python tools/download_model.py --repo-id Qwen/Qwen2.5-0.5B-Instruct

    # AutoDL 等国内云（走 ModelScope，比 HF 镜像还快）
    python tools/download_model.py --repo-id ZhipuAI/chatglm3-6b \
        --backend modelscope --local-dir models/chatglm3-6b
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

# 必须在 import huggingface_hub 之前设置：HF_ENDPOINT 是在模块导入时读取的
DEFAULT_ENDPOINT = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", DEFAULT_ENDPOINT)

ROOT = Path(__file__).resolve().parents[1]

# 只下推理和训练需要的文件。不加限制的话，有些仓库会同时带上
# pytorch_model.bin 和 model.safetensors 两份权重，体积直接翻倍。
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt"]


def download_from_modelscope(repo_id: str, local_dir: Path) -> str:
    """走 ModelScope（魔搭）下载。

    AutoDL 等国内云上 HF 常常不通或极慢，而 ModelScope 有 ChatGLM3 的
    官方镜像，速度差一个数量级。12GB 的模型这一步的体验差别很大。

    缓存放进 local_dir 而不是 ~/.cache —— 和 HF 那条路一样，
    保持所有大文件都在工作区里，方便打包和迁移。
    """
    from modelscope import snapshot_download

    kwargs: dict = {"cache_dir": str(local_dir.parent)}
    # 不同版本的 ModelScope 接口不一样：新版支持 local_dir（把文件平铺到
    # 指定目录），老版只有 cache_dir。按签名探测，两条路都能走。
    if "local_dir" in inspect.signature(snapshot_download).parameters:
        kwargs["local_dir"] = str(local_dir)
        kwargs.pop("cache_dir", None)

    print(f"下载中（ModelScope）: {repo_id}")
    return snapshot_download(repo_id, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载模型到项目目录内")
    parser.add_argument("--repo-id", required=True, help="例如 Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--local-dir", default=None, help="默认 models/<repo 名>")
    parser.add_argument("--backend", default="hf", choices=["hf", "modelscope"],
                        help="国内云上用 modelscope，HF 常常不通")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--all-files", action="store_true", help="不做文件类型过滤")
    args = parser.parse_args()

    local_dir = Path(args.local_dir) if args.local_dir else ROOT / "models" / args.repo_id.split("/")[-1]
    if not local_dir.is_absolute():
        local_dir = ROOT / local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"后端: {args.backend}")
    print(f"仓库: {args.repo_id}")
    print(f"目标: {local_dir}")

    if args.backend == "modelscope":
        path = download_from_modelscope(args.repo_id, local_dir)
    else:
        os.environ["HF_ENDPOINT"] = args.endpoint
        from huggingface_hub import snapshot_download  # 必须在设置 endpoint 之后导入

        print(f"镜像: {args.endpoint}")
        path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(local_dir),
            allow_patterns=None if args.all_files else ALLOW_PATTERNS,
            max_workers=4,
        )

    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"完成: {path}  ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
