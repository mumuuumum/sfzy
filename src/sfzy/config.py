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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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


def apply_overrides(cfg: Config, pairs: Sequence[str]) -> None:
    """就地覆盖配置项，形如 ["sft.num_epochs=1", "lora.r=4"]。

    调试时很有用：把 grad_accum_steps 改成 1、把 max_length 改小 ——
    都不用去动 yaml 文件，跑完也不用记得改回来。

    值用 yaml.safe_load 解析，所以 1 会变成 int、"1" 会变成 str、
    null 会变成 None，和直接在 yaml 里写一样。

    **每一段路径都要存在，包括最后一段。** 这不是洁癖，是防一类
    静默事故：`--override resume_from=...` 这种写法不带段名前缀，
    会被写到**顶层**，而真正读它的是 `cfg.sft.resume_from` ——
    覆盖静默失效、直接从零开始训练，你只会在几小时后才发现。
    末段校验能把这个错误变成一条立刻可见的报错。

    注意：调用必须在 `sft_cfg = cfg.sft` 之前 —— Config.__getattr__
    返回的是副本，先取出来再覆盖就改不到它了。
    """
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--override 需要 KEY=VALUE 形式，收到：{item!r}")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")

        node: Any = cfg
        walked: List[str] = []
        for part in parts[:-1]:
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(
                    f"--override 的路径不存在：{dotted}\n"
                    f"  '{part}' 不在 {'.'.join(walked) or '(顶层)'} 下，"
                    f"可用的键：{sorted(node.keys()) if isinstance(node, Mapping) else '（不是字典）'}"
                )
            node = node[part]
            walked.append(part)

        if not isinstance(node, Mapping) or parts[-1] not in node:
            raise KeyError(
                f"--override 的键不存在：{dotted}\n"
                f"  {'.'.join(parts[:-1]) or '(顶层)'} 下可用的键："
                f"{sorted(node.keys()) if isinstance(node, Mapping) else '（不是字典）'}\n"
                f"  提示：resume_from / num_epochs / per_device_batch_size 这些"
                f"都在 sft 段下，要写成 sft.resume_from=..."
            )

        node[parts[-1]] = parse_override_value(raw)


def parse_override_value(raw: str) -> Any:
    """把 --override 的字符串值转成合适的类型。

    用 yaml.safe_load 解析，所以 1 → int、"1" → str、null → None、
    true → bool，和直接在 yaml 里写一致。

    **额外补一刀处理科学计数法。** YAML 1.1 规定浮点数的尾数必须带小数点：

        yaml.safe_load("2e-4")   → "2e-4"（字符串！）
        yaml.safe_load("2.0e-4") → 0.0002（float）

    而 `learning_rate=2e-4` 是命令行里最自然的写法，它会悄悄变成字符串，
    然后在 optimizer 构造时才炸 —— 报错位置离原因很远。
    所以对"看起来是数字的字符串"再试一次 float()。
    """
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value
