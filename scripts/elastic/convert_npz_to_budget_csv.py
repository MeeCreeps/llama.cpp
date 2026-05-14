#!/usr/bin/env python3
# scripts/elastic/convert_npz_to_budget_csv.py
#
# 把 elastic_memory pipeline 产出的 npz（含 time_sec、app_mem_mb）
# 转成 BudgetWatcher 能读的 CSV: (time_sec, budget_mb)。
#
# 预算公式（见 docs/elastic/baseline_external_execution.md §7 Test 2）：
#   budget_mb(t) = dram_total - kernel_overhead - system_baseline - app_mem(t)
#
# 三个常数从设备 profile yaml 读（configs/devices/<device>.yaml），命令行
# 可覆盖。负值钳到 0。
#
# 用例：
#   ./convert_npz_to_budget_csv.py path/to/window.npz \
#       --device configs/devices/oneplus12.yaml \
#       -o traces/window.csv
#
# 自测用例（不需要真 trace，生成一份合成 npz 再转 CSV）：
#   ./convert_npz_to_budget_csv.py --make-synthetic synthetic.npz
#   ./convert_npz_to_budget_csv.py synthetic.npz \
#       --device configs/devices/oneplus12.yaml \
#       -o synthetic.csv

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_profile(path: str) -> dict:
    # 极简 YAML：只支持 "key: value" 行 + # 注释，避免引入 PyYAML 依赖
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if not v:
                continue
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def pick(name: str, cli_val, profile: dict):
    if cli_val is not None:
        return float(cli_val)
    if name in profile:
        return float(profile[name])
    sys.exit(f"错误：缺少参数 {name}（用 --{name.replace('_', '-')} 或 --device profile 提供）")


def make_synthetic(path: str) -> None:
    # 15 分钟，1 Hz 采样，app_mem 在 1500~6000 MB 之间慢慢起伏
    n = 15 * 60
    t = np.arange(n, dtype=float)
    # 三个正弦叠加 + 漂移，确保 min < max
    app = (
        3500.0
        + 1500.0 * np.sin(2 * np.pi * t / 600.0)
        + 800.0 * np.sin(2 * np.pi * t / 137.0)
        + 200.0 * np.sin(2 * np.pi * t / 41.0)
    )
    app = np.clip(app, 1000.0, 7000.0)
    np.savez(path, time_sec=t, app_mem_mb=app)
    print(f"已写合成 npz: {path} (n={n}, app_mem 范围 [{app.min():.1f}, {app.max():.1f}] MB)")


def convert(npz_path: str, out_csv: str, dram: float, kern: float, sysb: float) -> None:
    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.files)
    if "time_sec" not in keys:
        sys.exit(f"错误：npz 缺少字段 time_sec（现有字段：{keys}）")
    if "app_mem_mb" not in keys:
        sys.exit(f"错误：npz 缺少字段 app_mem_mb（现有字段：{keys}）")

    t = np.asarray(z["time_sec"], dtype=float)
    app = np.asarray(z["app_mem_mb"], dtype=float)
    if t.shape != app.shape:
        sys.exit(f"错误：time_sec 与 app_mem_mb 形状不匹配 {t.shape} vs {app.shape}")
    if t.ndim != 1:
        sys.exit(f"错误：time_sec 必须是 1D（当前 {t.ndim}D）")

    budget = dram - kern - sysb - app
    budget = np.clip(budget, 0.0, None)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "budget_mb"])
        for ti, bi in zip(t, budget):
            w.writerow([f"{ti:.6f}", f"{bi:.3f}"])

    print(
        f"写入 {out_path}: {len(t)} 行；"
        f"B(t) 范围 [{budget.min():.1f}, {budget.max():.1f}] MB；"
        f"M_floor = {budget.min():.1f} MB"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="npz → (time_sec, budget_mb) CSV")
    p.add_argument("npz", nargs="?", help="输入 npz 路径（--make-synthetic 模式下当输出路径）")
    p.add_argument("-o", "--output", help="输出 CSV 路径")
    p.add_argument("--device", help="设备 profile yaml（configs/devices/<device>.yaml）")
    p.add_argument("--dram-total-mb", type=float, help="覆盖 profile 的 dram_total_mb")
    p.add_argument("--kernel-overhead-mb", type=float, help="覆盖 profile 的 kernel_overhead_mb")
    p.add_argument("--system-baseline-mb", type=float, help="覆盖 profile 的 system_baseline_mb")
    p.add_argument(
        "--make-synthetic",
        action="store_true",
        help="生成一份合成 npz 写到 npz 参数指定的路径，然后退出（自测用）",
    )
    args = p.parse_args()

    if args.make_synthetic:
        if not args.npz:
            sys.exit("错误：--make-synthetic 需要给一个输出 npz 路径作为位置参数")
        make_synthetic(args.npz)
        return

    if not args.npz or not args.output:
        sys.exit("错误：需要 <npz> 与 -o <output.csv>")

    profile = load_profile(args.device) if args.device else {}
    dram = pick("dram_total_mb", args.dram_total_mb, profile)
    kern = pick("kernel_overhead_mb", args.kernel_overhead_mb, profile)
    sysb = pick("system_baseline_mb", args.system_baseline_mb, profile)
    convert(args.npz, args.output, dram, kern, sysb)


if __name__ == "__main__":
    main()
