#!/usr/bin/env python3
"""Materialize the selected mixed frontier over an absolute-budget trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from runtime.plan.select_mixed_boundary_profile import mode_counts
    from runtime.plan.super_tensor_planner import GranularityCostModel
except ModuleNotFoundError:
    from select_mixed_boundary_profile import mode_counts  # type: ignore
    from super_tensor_planner import GranularityCostModel  # type: ignore


def read_trace(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            time_value = row.get("t_sec", row.get("time_sec"))
            budget_value = row.get(
                "mem_available_mb", row.get("budget_mb"))
            if time_value is None or budget_value is None:
                raise ValueError(
                    f"{path}: expected time and absolute-budget columns")
            rows.append((float(time_value), float(budget_value)))
    if not rows:
        raise ValueError(f"{path}: empty trace")
    if any(second[0] < first[0] for first, second in zip(rows, rows[1:])):
        raise ValueError(f"{path}: non-monotonic timestamps")
    return rows


def available_plan_budgets(plan_dir: Path) -> list[int]:
    budgets = sorted({
        int(path.stem.removeprefix("plan_").removesuffix("MiB"))
        for path in plan_dir.glob("plan_*MiB.json")
    })
    if not budgets:
        raise ValueError(f"{plan_dir}: no budget plans")
    return budgets


def bucket_for_budget(
    budget_mib: float,
    bucket_mib: int,
    available: list[int],
) -> int:
    quantized = math.floor(budget_mib / bucket_mib) * bucket_mib
    quantized = min(max(quantized, available[0]), available[-1])
    if quantized in available:
        return quantized
    return min(available, key=lambda candidate: abs(candidate - quantized))


def dominant_mode(counts: dict[str, int | float]) -> str:
    # Multi decisions cover two complete weights, whereas Cut counts represent
    # tensors whose two row tiles execute independently.
    coverage = {
        "multi": 2 * int(counts["multi"]),
        "tensor": int(counts["tensor"]),
        "cut": int(counts["cut"]),
    }
    return max(coverage, key=coverage.get)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--model-meta", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--bucket-mib", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.bucket_mib <= 0:
        parser.error("--bucket-mib must be positive")
    trace = read_trace(args.trace)
    budgets = available_plan_budgets(args.plan_dir)
    model_meta: dict[str, Any] = json.loads(args.model_meta.read_text())
    profile = GranularityCostModel(
        json.loads(args.profile.read_text()), args.backend)
    counts_by_budget = {
        budget: mode_counts(
            args.plan_dir / f"plan_{budget}MiB.json",
            model_meta,
            profile,
        )
        for budget in budgets
    }

    timeline = []
    duration_by_bucket = {budget: 0.0 for budget in budgets}
    duration_by_dominant = {
        mode: 0.0 for mode in ("multi", "tensor", "cut")}
    for index, (time_sec, raw_budget_mib) in enumerate(trace):
        bucket = bucket_for_budget(
            raw_budget_mib, args.bucket_mib, budgets)
        counts = counts_by_budget[bucket]
        duration_sec = (
            max(0.0, trace[index + 1][0] - time_sec)
            if index + 1 < len(trace) else 0.0
        )
        dominant = dominant_mode(counts)
        duration_by_bucket[bucket] += duration_sec
        duration_by_dominant[dominant] += duration_sec
        timeline.append({
            "time_sec": time_sec,
            "raw_budget_mib": raw_budget_mib,
            "plan_budget_mib": bucket,
            "multi_decisions": counts["multi"],
            "tensor_decisions": counts["tensor"],
            "cut_tensors": counts["cut"],
            "cut_physical_units": 2 * int(counts["cut"]),
            "cut_boundaries": counts["cut_boundaries"],
            "merge_boundaries": counts["merge_boundaries"],
            "mode_runs": counts["mode_runs"],
            "dominant_mode": dominant,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.backend}_frontier_timeline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timeline[0]))
        writer.writeheader()
        writer.writerows(timeline)

    sample_deltas = [
        second[0] - first[0]
        for first, second in zip(trace, trace[1:])
        if second[0] > first[0]
    ]
    payload = {
        "backend": args.backend,
        "trace": str(args.trace),
        "trace_sha256": hashlib.sha256(
            args.trace.read_bytes()).hexdigest(),
        "trace_start_sec": trace[0][0],
        "trace_span_sec": trace[-1][0] - trace[0][0],
        "trace_samples": len(trace),
        "median_sample_delta_sec": (
            statistics.median(sample_deltas) if sample_deltas else 0.0),
        "raw_budget_min_mib": min(row[1] for row in trace),
        "raw_budget_max_mib": max(row[1] for row in trace),
        "duration_by_plan_bucket_sec": {
            str(budget): duration
            for budget, duration in duration_by_bucket.items()
            if duration > 0.0
        },
        "duration_by_dominant_mode_sec": duration_by_dominant,
        "frontier_by_plan_bucket": {
            str(budget): counts
            for budget, counts in counts_by_budget.items()
        },
        "timeline_csv": str(csv_path),
    }
    json_path = args.output_dir / f"{args.backend}_frontier_timeline.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "csv": str(csv_path),
        "json": str(json_path),
    }, indent=2))


if __name__ == "__main__":
    main()
