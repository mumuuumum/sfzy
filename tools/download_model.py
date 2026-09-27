"""把模型下载到项目目录内（不落到 ~/.cache）。

为什么要这个工具：
  1. huggingface.co 在国内直连不通，需要走 hf-mirror.com 镜像；
  2. 默认的 HF 缓存位置（~/.cache/huggingface/hub）不在工作区内，
     模型文件散落在外面既不好清理，也不方便打包带走。
     用 snapshot_download(local_dir=...) 会把文件平铺到一个普通目录里，
     from_pretrained 直接指向它就行，不需要任何环境变量。

用法：
    python tools/download_model.py --repo-id Qwen/Qwen2.5-0.5B-Instruct
    python tools/download_model.py --repo-id THUDM/chatglm3-6b --local-dir models/chatglm3-6b
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# 必须在 import huggingface_hub 之前设置：HF_ENDPOINT 是在模块导入时读取的
DEFAULT_ENDPOINT = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", DEFAULT_ENDPOINT)

ROOT = Path(__file__).resolve().parents[1]

# 只下推理和训练需要的文件。不加限制的话，有些仓库会同时带上
# pytorch_model.bin 和 model.safetensors 两份权重，体积直接翻倍。
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt"]


def main() -> None:
    parser = argparse.ArgumentParser(description="下载模型到项目目录内")
    parser.add_argument("--repo-id", required=True, help="例如 Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--local-dir", default=None, help="默认 models/<repo 名>")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--all-files", action="store_true", help="不做文件类型过滤")
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import snapshot_download  # 必须在设置 endpoint 之后导入

    local_dir = Path(args.local_dir) if args.local_dir else ROOT / "models" / args.repo_id.split("/")[-1]
    if not local_dir.is_absolute():
        local_dir = ROOT / local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"镜像: {args.endpoint}")
    print(f"仓库: {args.repo_id}")
    print(f"目标: {local_dir}")

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
