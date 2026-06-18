#!/usr/bin/env python3
"""Build an offline budget table directory for PlanProvider::create_table."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build offline elastic budget plan table")
    ap.add_argument("--model-meta", type=Path, required=True)
    ap.add_argument("--cost-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--budgets", required=True, help="comma-separated MiB budgets")
    ap.add_argument("--kv-mib", type=int, default=128)
    ap.add_argument("--misc-mib", type=int, default=256)
    ap.add_argument("--safety-mib", type=int, default=64)
    ap.add_argument("--time-limit-ms", type=int, default=20)
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--transition-weight", type=float, default=1.0)
    args = ap.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    if not budgets:
        raise SystemExit("no budgets provided")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    solver = Path(__file__).with_name("dynamic_budget_solver.py")
    index = []

    for b in budgets:
        out = args.out_dir / f"plan_{b}MiB.json"
        cmd = [
            sys.executable,
            str(solver),
            "--model-meta", str(args.model_meta),
            "--cost-dir", str(args.cost_dir),
            "--budget-mib", str(b),
            "--kv-mib", str(args.kv_mib),
            "--misc-mib", str(args.misc_mib),
            "--safety-mib", str(args.safety_mib),
            "--time-limit-ms", str(args.time_limit_ms),
            "--transition-weight", str(args.transition_weight),
            "--out", str(out),
        ]
        if args.allow_cpu_fallback:
            cmd.append("--allow-cpu-fallback")
        subprocess.run(cmd, check=True)
        index.append({"budget_mib": b, "file": out.name})

    (args.out_dir / "index.json").write_text(json.dumps({"index": index}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "plans": len(index)}, indent=2))


if __name__ == "__main__":
    main()
