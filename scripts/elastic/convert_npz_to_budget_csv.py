#!/usr/bin/env python3
# scripts/elastic/convert_npz_to_budget_csv.py
#
# 把 elastic_memory pipeline 产出的 npz 转成 BudgetWatcher 能读的两列 CSV：
#   (time_sec, budget_mb)
#
# npz 里 budget 字段已经是 LLM 可用预算（剩余可用内存），脚本只做透传 +
# 单位/形状校验，不再做任何 dram_total - kernel - sys - app 的减法。
#
# 字段名默认：time_sec、budget_mb。pipeline 那边如果叫别的，用
# --time-field / --budget-field 覆盖。
#
# 用例：
#   ./convert_npz_to_budget_csv.py path/to/window.npz -o traces/window.csv
#   ./convert_npz_to_budget_csv.py path/to/window.npz -o traces/window.csv \
#       --budget-field available_mb
#
# 自测（不需要真 trace）：
#   ./convert_npz_to_budget_csv.py --make-synthetic /tmp/synthetic.npz
#   ./convert_npz_to_budget_csv.py /tmp/synthetic.npz -o /tmp/synthetic.csv

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def make_synthetic(path: str) -> None:
    # 15 分钟、1 Hz；budget 在 ~7.5 GB 上下慢慢起伏，覆盖典型移动场景
    n = 15 * 60
    t = np.arange(n, dtype=float)
    budget = (
        7500.0
        + 1500.0 * np.sin(2 * np.pi * t / 600.0)
        + 800.0 * np.sin(2 * np.pi * t / 137.0)
        + 200.0 * np.sin(2 * np.pi * t / 41.0)
    )
    budget = np.clip(budget, 500.0, 14000.0)
    np.savez(path, time_sec=t, budget_mb=budget)
    print(f"已写合成 npz: {path} (n={n}, B(t) 范围 [{budget.min():.1f}, {budget.max():.1f}] MB)")


def convert(npz_path: str, out_csv: str, time_field: str, budget_field: str) -> None:
    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.files)
    if time_field not in keys:
        sys.exit(f"错误：npz 缺少时间字段 '{time_field}'（现有字段：{keys}）")
    if budget_field not in keys:
        sys.exit(f"错误：npz 缺少预算字段 '{budget_field}'（现有字段：{keys}）")

    t = np.asarray(z[time_field], dtype=float)
    b = np.asarray(z[budget_field], dtype=float)
    if t.shape != b.shape:
        sys.exit(f"错误：{time_field} 与 {budget_field} 形状不匹配 {t.shape} vs {b.shape}")
    if t.ndim != 1:
        sys.exit(f"错误：{time_field} 必须是 1D（当前 {t.ndim}D）")
    if np.any(b < 0):
        n_neg = int(np.sum(b < 0))
        print(f"警告：{budget_field} 有 {n_neg} 个负值，钳到 0", file=sys.stderr)
        b = np.clip(b, 0.0, None)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "budget_mb"])
        for ti, bi in zip(t, b):
            w.writerow([f"{ti:.6f}", f"{bi:.3f}"])

    print(
        f"写入 {out_path}: {len(t)} 行；"
        f"B(t) 范围 [{b.min():.1f}, {b.max():.1f}] MB；"
        f"M_floor = {b.min():.1f} MB"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="elastic_memory npz → (time_sec, budget_mb) CSV")
    p.add_argument("npz", nargs="?", help="输入 npz 路径（--make-synthetic 模式下当输出路径）")
    p.add_argument("-o", "--output", help="输出 CSV 路径")
    p.add_argument("--time-field", default="time_sec", help="npz 中的时间字段名，默认 time_sec")
    p.add_argument("--budget-field", default="budget_mb", help="npz 中的预算字段名，默认 budget_mb")
    p.add_argument(
        "--make-synthetic",
        action="store_true",
        help="生成一份合成 npz 写到 npz 参数指定路径后退出（自测用）",
    )
    args = p.parse_args()

    if args.make_synthetic:
        if not args.npz:
            sys.exit("错误：--make-synthetic 需要给一个输出 npz 路径作为位置参数")
        make_synthetic(args.npz)
        return

    if not args.npz or not args.output:
        sys.exit("错误：需要 <npz> 与 -o <output.csv>")
    convert(args.npz, args.output, args.time_field, args.budget_field)


if __name__ == "__main__":
    main()
