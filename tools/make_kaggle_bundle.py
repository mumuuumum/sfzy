"""把代码打包成可以上传到 Kaggle 的 zip。

为什么需要：Kaggle notebook 里改代码很痛苦，而 Dataset 挂载是只读的、
正好适合放代码。每次本地改完重新跑一遍这个脚本、更新 Dataset 版本即可。

排除：data/ models/ outputs/ .git/ 等 —— 数据和模型单独作为 Dataset，
代码包应该只有几百 KB。

用法：
    python tools/make_kaggle_bundle.py
    # 产出 dist/sfzy-code.zip
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "sfzy-code.zip"

# 只打包这些目录/文件
INCLUDE_DIRS = ["src", "scripts", "configs", "tools", "tests", "docs"]
INCLUDE_FILES = ["pyproject.toml", "requirements-kaggle.txt", "README.md"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".vscode"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_PARTS for part in path.parts) or path.suffix in EXCLUDE_SUFFIXES


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in INCLUDE_DIRS:
            base = ROOT / d
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if path.is_file() and not should_skip(path):
                    zf.write(path, path.relative_to(ROOT))
                    count += 1
        for f in INCLUDE_FILES:
            path = ROOT / f
            if path.exists():
                zf.write(path, Path(f))
                count += 1

    size = OUT.stat().st_size / 1024
    print(f"打包完成: {OUT}")
    print(f"  {count} 个文件，{size:.0f} KB")


if __name__ == "__main__":
    main()
