#!/usr/bin/env python3
"""Build an offline budget table directory for PlanProvider::create_table."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def plan_to_state(plan_path: Path, state_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    rows = []
    for w in plan.get("weights", []):
        if not isinstance(w, dict):
            continue
        loc = str(w.get("location", "disk")).lower()
        flags = ["disk_available"]
        if loc == "gpu":
            flags.append("gpu_compute_resident")
        elif loc == "cpu":
            flags.append("cpu_compute_resident")
        rows.append({
            "name": w.get("name", ""),
            "flags": flags,
        })
    state_path.write_text(json.dumps({"weights": rows}, indent=2, sort_keys=True) + "\n")


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
    ap.add_argument("--prefetch-distance", type=int, default=1)
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--transition-weight", type=float, default=0.1)
    ap.add_argument("--disk-reload-multiplier", type=float, default=1.0)
    ap.add_argument("--disk-gpu-reload-multiplier", type=float, default=4.0)
    ap.add_argument("--overlap-model", choices=("pipeline", "none"), default="pipeline")
    ap.add_argument("--cp-objective", choices=("resource_makespan", "interval_makespan", "sum"), default="resource_makespan")
    ap.add_argument("--allowed-placements", default="cpu,gpu,disk_cpu,disk_gpu")
    ap.add_argument("--chain-state", action="store_true",
                    help="build each budget using the previous lower-budget plan as current state")
    args = ap.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    if not budgets:
        raise SystemExit("no budgets provided")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    solver = Path(__file__).with_name("dynamic_budget_solver.py")
    index = []

    with tempfile.TemporaryDirectory(prefix="offline_chain_state_") as td:
        prev_state: Path | None = None
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
                "--prefetch-distance", str(args.prefetch_distance),
                "--transition-weight", str(args.transition_weight),
                "--disk-reload-multiplier", str(args.disk_reload_multiplier),
                "--disk-gpu-reload-multiplier", str(args.disk_gpu_reload_multiplier),
                "--overlap-model", str(args.overlap_model),
                "--cp-objective", str(args.cp_objective),
                "--allowed-placements", str(args.allowed_placements),
                "--out", str(out),
            ]
            if args.chain_state and prev_state is not None:
                cmd.extend(["--state", str(prev_state)])
            if args.allow_cpu_fallback:
                cmd.append("--allow-cpu-fallback")
            subprocess.run(cmd, check=True)
            index.append({"budget_mib": b, "file": out.name})
            if args.chain_state:
                next_state = Path(td) / f"state_{b}MiB.json"
                plan_to_state(out, next_state)
                prev_state = next_state

    (args.out_dir / "index.json").write_text(json.dumps({"index": index}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "plans": len(index)}, indent=2))


if __name__ == "__main__":
    main()
