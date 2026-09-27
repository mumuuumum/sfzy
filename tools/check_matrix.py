"""跑一组场景，检查各种路径能否跑通。

不是单元测试的替代品 —— 单元测试验证的是"逻辑对不对"，
这个脚本验证的是"端起命令行来到底能不能跑"，覆盖配置解析、
断点续训、设备选择、错误提示这些只有真跑才会碰到的东西。

用法：
    python tools/check_matrix.py            # 全部
    python tools/check_matrix.py -k resume  # 只跑名字含 resume 的
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/yukino/anaconda3/envs/minimind/bin/python"
CKPT_DIR = ROOT / "outputs" / "sft_local"

CFG = "configs/sft_local.yaml"
QUICK = ["--limit", "1", "--override", "sft.num_epochs=1",
         "--override", "sft.grad_accum_steps=1"]


def newest_checkpoint() -> str | None:
    files = sorted(CKPT_DIR.glob("step_*.pt"))
    return str(files[-1]) if files else None


def run(args: list[str], timeout: int = 900) -> tuple[int, str]:
    proc = subprocess.run(
        [PY, str(ROOT / "scripts" / "train_sft.py"), *args],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode, proc.stdout + proc.stderr


SCENARIOS: list[dict] = [
    {
        "name": "缺配置文件",
        "args": ["--config", "configs/根本不存在.yaml"],
        "fail": True, "contains": "配置文件不存在",
    },
    {
        "name": "override 格式错（缺等号）",
        "args": ["--config", CFG, "--override", "sft.num_epochs"],
        "fail": True, "contains": "KEY=VALUE",
    },
    {
        "name": "override 路径不存在",
        "args": ["--config", CFG, "--override", "根本没有这段.x=1"],
        "fail": True, "contains": "路径不存在",
    },
    {
        "name": "resume 指向不存在的文件",
        "args": ["--config", CFG, "--resume", "outputs/不存在.pt", *QUICK],
        "fail": True, "contains": "checkpoint 不存在",
    },
    {
        "name": "target_modules 写错（匹配不到层）",
        "args": ["--config", CFG, "--override", 'lora.target_modules=["根本不存在的层"]', *QUICK],
        "fail": True, "contains": "没有匹配到任何层",
    },
    {
        "name": "从头训练（1 条样本 1 步）",
        "args": ["--config", CFG, *QUICK],
        "fail": False, "contains": "未指定 resume_from",
    },
    {
        "name": "续训（用上一步的 checkpoint）",
        "args": ["--config", CFG, *QUICK],   # args 在运行时替换
        "dynamic": "resume",
        "fail": False, "contains": "已从",
    },
    {
        "name": "dtype 换成 float32",
        "args": ["--config", CFG, "--override", "sft.dtype=float32", *QUICK],
        "fail": False, "contains": "训练结束",
    },
    {
        "name": "关掉梯度检查点",
        "args": ["--config", CFG, "--override", "sft.gradient_checkpointing=false", *QUICK],
        "fail": False, "contains": "训练结束",
    },
    {
        "name": "纯 CPU 跑（fp32 + max_length 256）",
        "args": ["--config", CFG, "--override", "model_config=configs/model_debug_cpu.yaml",
                 "--override", "device=cpu", *QUICK],
        "fail": False, "contains": "设备: cpu",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", default=None, help="只跑名字含该关键字的场景")
    opts = parser.parse_args()

    scenarios = [s for s in SCENARIOS if not opts.k or opts.k in s["name"]]
    results = []

    for i, sc in enumerate(scenarios, 1):
        args = list(sc["args"])
        if sc.get("dynamic") == "resume":
            ckpt = newest_checkpoint()
            if not ckpt:
                results.append((sc["name"], "SKIP", "没有可用的 checkpoint", 0.0))
                continue
            args = ["--config", CFG, "--resume", ckpt,
                    "--limit", "1", "--override", "sft.num_epochs=2",
                    "--override", "sft.grad_accum_steps=1"]

        print(f"[{i}/{len(scenarios)}] {sc['name']} ...", flush=True)
        start = time.time()
        try:
            code, out = run(args)
            elapsed = time.time() - start
            ok = (code != 0) if sc["fail"] else (code == 0)
            hit = sc["contains"] in out
            if ok and hit:
                status, detail = "PASS", ""
            else:
                why = []
                if not ok:
                    why.append(f"退出码 {code}（期望{'非零' if sc['fail'] else '零'}）")
                if not hit:
                    why.append(f"输出里没有 {sc['contains']!r}")
                status, detail = "FAIL", "；".join(why)
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            status, detail = "TIMEOUT", "超过 900 秒"
        results.append((sc["name"], status, detail, elapsed))

    print("\n" + "=" * 78)
    print(f"{'场景':<34}{'结果':<8}{'耗时':<8}说明")
    print("-" * 78)
    for name, status, detail, elapsed in results:
        print(f"{name:<34}{status:<8}{elapsed:>6.0f}s  {detail}")
    print("=" * 78)

    bad = [r for r in results if r[1] != "PASS"]
    print(f"{len(results) - len(bad)}/{len(results)} 通过")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
