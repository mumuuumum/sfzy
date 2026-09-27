"""配置加载：YAML 读取、defaults 继承、点号访问。

用法：
    from sfzy.config import load_config
    cfg = load_config("configs/sft.yaml")
    cfg.sft.learning_rate          # 属性访问
    cfg.path_("paths.outputs_dir") # 点号路径访问

设计约定：
  * 子配置用 `defaults: base` 声明继承，父配置位于同目录下的 base.yaml。
  * 子配置的键会**深合并**覆盖父配置，嵌套字典不会整体替换。
  * 直接运行本文件可打印合并后的配置，便于排查继承问题：
        python -m sfzy.config configs/sft.yaml
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import yaml


class Config(dict):
    """支持属性访问的字典，便于写 `cfg.sft.learning_rate`。"""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(
                f"配置中没有 '{item}'。可用键：{sorted(self.keys())}"
            ) from exc
        return Config(value) if isinstance(value, Mapping) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def path_(self, dotted: str, default: Any = None) -> Any:
        """按 "a.b.c" 取值，缺失时返回 default。"""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并：override 覆盖 base，字典逐层合并而非整体替换。"""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: Union[str, Path],
    _stack: Tuple[Path, ...] = (),
) -> Config:
    """加载配置文件并沿 `defaults` 链向上合并。"""
    path = Path(path).resolve()
    if path in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, path))
        raise ValueError(f"配置继承出现循环：{chain}")
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"配置文件顶层必须是映射：{path}")

    raw = dict(raw)
    parent_name = raw.pop("defaults", None)

    merged: Dict[str, Any] = {}
    if parent_name:
        parent_path = path.parent / f"{parent_name}.yaml"
        parent = load_config(parent_path, (*_stack, path))
        merged = deep_merge(merged, parent)

    merged = deep_merge(merged, raw)
    merged["_config_path"] = str(path)
    return Config(merged)


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python -m sfzy.config <配置文件路径>", file=sys.stderr)
        raise SystemExit(2)
    cfg = load_config(sys.argv[1])
    print(yaml.safe_dump(dict(cfg), allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
