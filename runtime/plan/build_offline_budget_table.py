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
    ap.add_argument("--top-k", type=int, default=1,
                    help="number of diverse candidate plans to preserve per budget")
    ap.add_argument("--candidate-placement-specs", default="",
                    help="semicolon-separated allowed-placement specs used to diversify candidates")
    ap.add_argument("--candidate-min-distance", type=int, default=32,
                    help="minimum weight-placement Hamming distance between top-k candidates when placement specs are not provided")
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
        placement_specs = [s.strip() for s in args.candidate_placement_specs.split(";") if s.strip()]
        use_distance_diversity = not placement_specs
        if use_distance_diversity:
            placement_specs = [args.allowed_placements]
        for b in budgets:
            candidates = []
            n_candidates = max(1, args.top_k)
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
            ]
            if args.chain_state and prev_state is not None:
                cmd.extend(["--state", str(prev_state)])
            if args.allow_cpu_fallback:
                cmd.append("--allow-cpu-fallback")
            for k in range(n_candidates):
                out = args.out_dir / (f"plan_{b}MiB.json" if k == 0 else f"plan_{b}MiB_cand{k}.json")
                cand_cmd = list(cmd)
                cand_cmd.extend([
                    "--allowed-placements", placement_specs[k % len(placement_specs)],
                    "--disk-gpu-reload-multiplier", str(args.disk_gpu_reload_multiplier * (1.0 + 0.20 * (k // len(placement_specs)))),
                ])
                if use_distance_diversity and k > 0 and args.candidate_min_distance > 0:
                    for prev in candidates:
                        cand_cmd.extend(["--exclude-plan", str(args.out_dir / prev["file"])])
                    cand_cmd.extend(["--min-placement-distance", str(args.candidate_min_distance)])
                cand_cmd.extend(["--out", str(out)])
                subprocess.run(cand_cmd, check=True)
                plan = json.loads(out.read_text())
                candidates.append({
                    "candidate_id": k,
                    "file": out.name,
                    "allowed_placements": placement_specs[k % len(placement_specs)],
                    "diversity": "distance" if use_distance_diversity else "placement_specs",
                    "min_placement_distance": args.candidate_min_distance if use_distance_diversity and k > 0 else 0,
                    "pred_per_token_ms": plan.get("pred_per_token_ms", 0.0),
                    "bottleneck": plan.get("bottleneck", ""),
                })
            index.append({"budget_mib": b, "file": candidates[0]["file"], "candidates": candidates})
            if args.chain_state:
                next_state = Path(td) / f"state_{b}MiB.json"
                plan_to_state(args.out_dir / candidates[0]["file"], next_state)
                prev_state = next_state

    (args.out_dir / "index.json").write_text(json.dumps({
        "index": index,
        "top_k": max(1, args.top_k),
        "candidate_diversity": "distance" if use_distance_diversity else "placement_specs",
        "candidate_min_distance": args.candidate_min_distance if use_distance_diversity else 0,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "budgets": len(index), "top_k": max(1, args.top_k)}, indent=2))


if __name__ == "__main__":
    main()
