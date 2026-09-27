"""分批实验：每批一条命令，单独可跑，互不依赖（除了同批内准备数据的先后）。

三个设计点，都是为了无人值守：

  1. **单个实验失败不中断后续**。一个配置写错不该让整批白费。
  2. **每个实验一个日志文件**（outputs/exp_logs/<name>.log），
     跑完直接翻日志定位，不用去终端历史里找。
  3. **批次内按依赖排序**：准备数据的步骤排在真正训练的前面。

用法：
    python tools/run_experiments.py --list              # 看有哪些批次
    python tools/run_experiments.py --batch 1 --dry-run # 看第 1 批要跑什么
    python tools/run_experiments.py --batch 1           # 跑第 1 批
    python tools/run_experiments.py --only e2_lora_r8   # 只跑某一个实验
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "exp_logs"
PYTHON = "/home/yukino/anaconda3/envs/minimind/bin/python"

FINAL_RE = re.compile(r"训练结束: step=(\d+) epoch=(\d+) best_dev_loss=([\d.]+) history=(\d+) 条")
LOSS_RE = re.compile(r"loss ([\d.]+)")

TRAIN = "scripts/train_sft.py"
COMMON = ["--config", "configs/sft_local.yaml"]


def train_cmd(name: str, *, data: str = "configs/data_2000.yaml", limit: int | None = None,
              extra: list[str] | None = None, resume: str | None = None) -> list[str]:
    cmd = [TRAIN, *COMMON, "--override", f"data_config={data}",
           "--override", f"sft.output_dir=outputs/exp/{name}"]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if resume is not None:
        cmd += ["--resume", resume]
    return cmd + (extra or [])


def lora_ablation(r: int) -> dict:
    return {
        "name": f"e2_lora_r{r}",
        "desc": f"LoRA 消融 r={r}（alpha=2r），200 条 1 轮",
        "cmd": train_cmd(f"e2_lora_r{r}", limit=200, extra=[
            "--override", f"lora.r={r}",
            "--override", f"lora.alpha={2 * r}",
            "--override", "sft.num_epochs=1",
            "--override", "sft.save_every_n_steps=50",
            "--override", "sft.log_every_n_steps=5",
        ]),
        "timeout": 2400,
    }


def dtype_run(dtype: str) -> dict:
    return {
        "name": f"e3_dtype_{dtype}",
        "desc": f"混合精度 {dtype}，200 条 1 轮",
        "cmd": train_cmd(f"e3_dtype_{dtype}", limit=200, extra=[
            "--override", f"sft.dtype={dtype}",
            "--override", "sft.num_epochs=1",
            "--override", "sft.save_every_n_steps=50",
            "--override", "sft.log_every_n_steps=5",
        ]),
        "timeout": 2400,
    }


# 批次 1/2/3 都要用这份 2000 条的数据。放在每批开头而不是只在批次 1 里做，
# 是为了让每一批都能**单独运行**——否则单独跑批次 2 会因为数据目录不存在而失败。
# 重复生成是幂等的（同一个种子、同一份原始数据），代价只有 1 分钟。
PREPARE_2000 = {
    "name": "prepare_2000",
    "desc": "准备 2000 条数据（幂等，可重复执行）",
    "cmd": ["scripts/prepare_data.py", "--config", "configs/data_2000.yaml"],
    "timeout": 900,
}


BATCHES: dict[str, dict] = {
    "1": {
        "title": "回归 + LoRA 消融",
        "eta": "实测 24m22s",
        "why": "先确认改完代码链路没坏，再拿到 LoRA rank 的消融表。",
        "experiments": [
            {
                "name": "s0_pytest",
                "desc": "全量单元测试（140 个用例）",
                "cmd": ["-m", "pytest", "-q"],
                "timeout": 900,
            },
            {
                "name": "e1_regression",
                "desc": "回归：4 条样本 1 步，确认训练链路还能跑",
                "cmd": train_cmd("e1_regression", data="configs/data.yaml", limit=4, extra=[
                    "--override", "sft.num_epochs=1",
                    "--override", "sft.grad_accum_steps=1",
                    "--override", "sft.save_every_n_steps=1",
                ]),
                "timeout": 900,
            },
            {
                **PREPARE_2000,
            },
            lora_ablation(4),
            lora_ablation(8),
            lora_ablation(16),
        ],
    },
    "2": {
        "title": "精度路径 + 断点续训",
        "eta": "实测 40m35s",
        "why": "本地 3050 支持 bf16 而 Kaggle 的 T4 不支持，两条路径都要成立；续训是 Kaggle 9 小时限制下的刚需。",
        "experiments": [
            {**PREPARE_2000},
            dtype_run("float16"),
            dtype_run("bfloat16"),
            {
                "name": "e6a_resume_base",
                "desc": "断点续训上半场：200 条 1 轮，留下 best.pt",
                "cmd": train_cmd("e6", limit=200, extra=[
                    "--override", "sft.num_epochs=1",
                    "--override", "sft.save_every_n_steps=50",
                ]),
                "timeout": 2400,
            },
            {
                "name": "e6b_resume_continue",
                "desc": "断点续训下半场：从 best.pt 恢复，配置改成 2 轮",
                "cmd": train_cmd("e6", limit=200, resume="outputs/exp/e6/best.pt", extra=[
                    "--override", "sft.num_epochs=2",
                    "--override", "sft.save_every_n_steps=50",
                ]),
                "timeout": 2400,
            },
        ],
    },
    "3": {
        "title": "中等规模长跑",
        "eta": "实测 8m57s",
        "why": "2000 条 1 轮，验证 checkpoint 频率、prune 清理、显存是否稳定、loss 有没有发散。max_length 压到 512 是为了把耗时砍半——这一轮关心的是稳定性，不是指标。",
        "experiments": [
            {**PREPARE_2000},
            {
                "name": "e4_medium_2000",
                "desc": "2000 条 1 轮，max_length=512",
                "cmd": train_cmd("e4_medium_2000", extra=[
                    "--override", "sft.num_epochs=1",
                    "--override", "sft.max_length=512",
                    "--override", "sft.save_every_n_steps=20",
                    "--override", "sft.log_every_n_steps=10",
                    "--override", "sft.keep_last_n_checkpoints=3",
                ]),
                "timeout": 14400,
            },
        ],
    },
    "4": {
        "title": "全量数据长跑",
        "eta": "约 1 小时",
        "why": "13427 条 1 轮，最接近 Kaggle 的真实条件（只是模型小一个量级、序列短一半）。验证真实规模下的数据管线、磁盘占用、几百个 step 之后的数值稳定性。",
        "experiments": [
            {
                "name": "prepare_full",
                "desc": "准备全量 13427 条数据",
                "cmd": ["scripts/prepare_data.py", "--config", "configs/data_full.yaml"],
                "timeout": 900,
            },
            {
                "name": "e5_full",
                "desc": "全量 13427 条 1 轮，max_length=512",
                "cmd": train_cmd("e5_full", data="configs/data_full.yaml", extra=[
                    "--override", "sft.num_epochs=1",
                    "--override", "sft.max_length=512",
                    "--override", "sft.save_every_n_steps=100",
                    "--override", "sft.log_every_n_steps=50",
                    "--override", "sft.keep_last_n_checkpoints=2",
                ]),
                "timeout": 28800,
            },
        ],
    },
}


def fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def extract_metrics(log_text: str) -> dict:
    out: dict = {}
    if m := FINAL_RE.search(log_text):
        out.update(step=int(m.group(1)), epoch=int(m.group(2)),
                   best_dev_loss=float(m.group(3)), history=int(m.group(4)))
    losses = [float(x) for x in LOSS_RE.findall(log_text)]
    if losses:
        out["first_loss"] = losses[0]
        out["last_loss"] = losses[-1]
    return out


def run_one(exp: dict, index: int, total: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{exp['name']}.log"
    cmd = [PYTHON, *exp["cmd"]]

    print(f"\n{'=' * 72}")
    print(f"[{index}/{total}] {exp['name']} — {exp['desc']}")
    print(f"       日志: {log_path.relative_to(ROOT)}   超时: {fmt_duration(exp['timeout'])}")
    print(f"{'=' * 72}", flush=True)

    started = time.time()
    status = "ok"
    deadline = started + exp["timeout"]

    # 输出**直接写文件**，而不是先 capture 再落盘。
    # 之前用 capture_output 有个实际麻烦：日志只在实验结束后才出现，
    # 跑一小时的长实验时完全看不到进度，出问题也只能干等。
    # 现在随时可以 tail -f 看进度。
    timed_out = False
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_file,
                                stderr=subprocess.STDOUT, text=True)
        last_beat = time.time()
        while proc.poll() is None:
            now = time.time()
            if now > deadline:
                proc.kill()
                timed_out = True
                break
            if now - last_beat >= 120:
                last_beat = now
                print(f"    ... 运行中 {fmt_duration(now - started)}", flush=True)
            time.sleep(3)
        proc.wait()

    output = log_path.read_text(encoding="utf-8", errors="replace")
    if timed_out:
        status = "timeout"
    elif proc.returncode != 0:
        status = f"exit={proc.returncode}"

    elapsed = time.time() - started

    result = {"name": exp["name"], "desc": exp["desc"], "status": status,
              "seconds": round(elapsed, 1), "log": str(log_path.relative_to(ROOT)),
              **extract_metrics(output)}

    mark = "✓" if status == "ok" else "✗"
    print(f"{mark} {exp['name']}: {status}  耗时 {fmt_duration(elapsed)}", flush=True)
    return result


def print_batches() -> None:
    print("可用的批次：\n")
    for key, batch in BATCHES.items():
        print(f"  批次 {key}  {batch['title']:<20} {batch['eta']}")
        print(f"          {batch['why']}")
        print(f"          实验: {', '.join(e['name'] for e in batch['experiments'])}\n")
    print("运行方式：python tools/run_experiments.py --batch <编号>")


def main() -> None:
    parser = argparse.ArgumentParser(description="分批实验")
    parser.add_argument("--batch", default=None, help="要跑的批次编号，见 --list")
    parser.add_argument("--list", action="store_true", help="列出所有批次")
    parser.add_argument("--only", default=None, help="只跑名字匹配的实验（子串匹配），忽略批次")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    args = parser.parse_args()

    if args.list or (args.batch is None and args.only is None):
        print_batches()
        return

    if args.only:
        todo = [e for b in BATCHES.values() for e in b["experiments"] if args.only in e["name"]]
        label = f"--only {args.only}"
    else:
        if args.batch not in BATCHES:
            print(f"没有批次 {args.batch}。用 --list 看可用的编号。")
            sys.exit(1)
        batch = BATCHES[args.batch]
        todo = batch["experiments"]
        label = f"批次 {args.batch} · {batch['title']}"

    if not todo:
        print("没有匹配的实验")
        sys.exit(1)

    print(f"{label}：{len(todo)} 个实验，{BATCHES[args.batch]['eta'] if args.batch else ''}")
    for i, e in enumerate(todo, 1):
        print(f"  {i:>2}. {e['name']:<22} {e['desc']}")
    if args.dry_run:
        return

    started = time.time()
    results = [run_one(e, i, len(todo)) for i, e in enumerate(todo, 1)]
    elapsed = time.time() - started

    print(f"\n{'=' * 72}\n汇总（总耗时 {fmt_duration(elapsed)}）\n{'=' * 72}")
    print(f"{'实验':<24}{'状态':<10}{'耗时':<10}{'best_dev':<12}loss")
    for r in results:
        first, last = r.get("first_loss"), r.get("last_loss")
        loss = f"{first:.4f} -> {last:.4f}" if first is not None and last is not None else "-"
        dev = f"{r['best_dev_loss']:.4f}" if "best_dev_loss" in r else "-"
        print(f"{r['name']:<24}{r['status']:<10}{fmt_duration(r['seconds']):<10}{dev:<12}{loss}")

    tag = f"batch{args.batch}" if args.batch else "single"
    summary = LOG_DIR / f"summary_{tag}.json"
    summary.write_text(json.dumps(
        {"label": label, "elapsed_seconds": round(elapsed, 1), "results": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已写入 {summary.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
