"""把代码打包成 zip（**备用方案**）。

⚠️ 正式流程已经改成 **Git**：本地 push、Kaggle `git clone`。
   见 docs/kaggle_sft.md 的「第一步：数据走 Dataset，代码走 Git」。

   改成 git 的原因：代码是高频小改动（一天十几次），而 Dataset 是
   为大数据设计的 —— 版本化慢、只读、没有 diff。继续用它同步代码，
   最后一定会滑向"在 Kaggle 上就地打补丁"，然后两边对不上。
   我们用这个 zip 方案实际走过一遍，最终在 Kaggle 上攒了六七个补丁。

保留这个脚本是因为两种情况还能用上：
  1. Kaggle 那边不能联网（无法 git clone）时的离线部署；
  2. 想把"某个特定版本的代码 + 数据"一起打包归档。

排除：data/ models/ outputs/ .git/ 等 —— 数据和模型单独作为 Dataset，
代码包应该只有一两百 KB。

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
