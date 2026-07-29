#!/usr/bin/env python3
"""Summarize Elastic token latency by real-trace time phase.

The token timestamp is the completion time.  The first one-token decode in
each phase is excluded because its evaluation interval can straddle the phase
boundary.  Statistics remain process-level; tokens are not treated as
independent experimental repeats.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_PHASES = (
    "high_before:0:4",
    "decline:4:6",
    "low:6:22",
    "recovery:22:40",
    "high_after:40:44.001",
)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = rank - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise SystemExit(f"no rows generated for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_phases(specs: list[str]) -> list[tuple[str, float, float]]:
    phases: list[tuple[str, float, float]] = []
    for spec in specs:
        label, start, end = spec.split(":", 2)
        phase = (label, float(start), float(end))
        if phase[2] <= phase[1]:
            raise SystemExit(f"invalid phase interval: {spec}")
        phases.append(phase)
    phases.sort(key=lambda phase: phase[1])
    for left, right in zip(phases, phases[1:]):
        if left[2] > right[1]:
            raise SystemExit(f"overlapping phases: {left[0]} and {right[0]}")
    return phases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--phase", action="append", dest="phases",
                        help="label:start_s:end_s; may be repeated")
    parser.add_argument(
        "--allow-empty-phases",
        action="store_true",
        help=("record a missing-phase row instead of aborting when a short "
              "phase contains no usable completed token"),
    )
    args = parser.parse_args()
    phases = parse_phases(args.phases or list(DEFAULT_PHASES))

    with (args.result_dir / "dynamic_runs.csv").open(
            newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    run_meta = {row["run_id"]: row for row in runs}

    result_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for token_path in sorted((args.result_dir / "tokens").glob("*.csv")):
        run_id = token_path.stem
        meta = run_meta.get(run_id)
        if meta is None:
            raise SystemExit(f"missing dynamic_runs.csv row for {run_id}")
        valid = (
            meta.get("returncode") == "0"
            and meta.get("valid_device_idle", "").lower() == "true"
            and meta.get("valid_device_awake", "").lower() == "true"
            and meta.get("valid_runtime", "").lower() == "true"
            and meta.get("valid_token_sequence", "").lower() == "true"
        )
        if not valid:
            raise SystemExit(f"refusing to analyze invalid run: {run_id}")
        with token_path.open(newline="", encoding="utf-8") as handle:
            tokens = list(csv.DictReader(handle))

        values: dict[str, list[dict[str, str]]] = defaultdict(list)
        dropped: dict[str, int] = defaultdict(int)
        phase_seen: set[str] = set()
        previous_t: float | None = None
        for token in tokens:
            current_t = float(token["t_sec"])
            phase = next((item for item in phases
                          if item[1] <= current_t < item[2]), None)
            if phase is not None and int(token["n_tokens"]) == 1:
                label, start, _ = phase
                crosses_start = previous_t is None or previous_t < start
                if label not in phase_seen or crosses_start:
                    dropped[label] += 1
                    phase_seen.add(label)
                else:
                    values[label].append(token)
            previous_t = current_t

        for ordinal, (label, start, end) in enumerate(phases):
            phase_tokens = values[label]
            if not phase_tokens:
                if not args.allow_empty_phases:
                    raise SystemExit(f"{run_id}: no usable tokens in phase {label}")
                missing_rows.append({
                    "run_id": run_id,
                    "backend": meta["backend"],
                    "repeat": meta["repeat"],
                    "mode": meta["mode"],
                    "phase": label,
                    "phase_ordinal": ordinal,
                    "start_s": start,
                    "end_s": end,
                    "n_boundary_tokens_dropped": dropped[label],
                    "reason": "no_usable_completed_token",
                })
                continue
            latency = [float(token["latency_ms"]) for token in phase_tokens]
            budgets = [int(token["budget_mib"]) for token in phase_tokens]
            result_rows.append({
                "run_id": run_id,
                "backend": meta["backend"],
                "repeat": meta["repeat"],
                "mode": meta["mode"],
                "phase": label,
                "phase_ordinal": ordinal,
                "start_s": start,
                "end_s": end,
                "n_boundary_tokens_dropped": dropped[label],
                "n_decode": len(phase_tokens),
                "budget_min_mib": min(budgets),
                "budget_median_mib": statistics.median(budgets),
                "budget_max_mib": max(budgets),
                "median_latency_ms": statistics.median(latency),
                "p95_latency_ms": percentile(latency, 95),
                "mean_latency_ms": statistics.mean(latency),
            })

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_repeat: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in result_rows:
        key = (str(row["mode"]), str(row["phase"]))
        grouped[key].append(row)
        by_repeat[(str(row["repeat"]), str(row["phase"]))][str(row["mode"])] = row

    phase_order = {phase[0]: index for index, phase in enumerate(phases)}
    expected_modes = {row["mode"] for row in runs}
    aggregate_rows: list[dict[str, object]] = []
    for (mode, label), rows in sorted(
            grouped.items(), key=lambda item: (phase_order[item[0][1]], item[0][0])):
        def win_count(metric: str) -> int:
            wins = 0
            for (_, repeat_label), modes in by_repeat.items():
                if repeat_label != label:
                    continue
                if set(modes) != expected_modes:
                    continue
                winner = min(modes, key=lambda name: float(modes[name][metric]))
                wins += winner == mode
            return wins

        medians = [float(row["median_latency_ms"]) for row in rows]
        p95s = [float(row["p95_latency_ms"]) for row in rows]
        means = [float(row["mean_latency_ms"]) for row in rows]
        aggregate_rows.append({
            "mode": mode,
            "phase": label,
            "n_runs": len(rows),
            "n_decode_total": sum(int(row["n_decode"]) for row in rows),
            "median_of_run_medians_ms": statistics.median(medians),
            "median_of_run_p95_ms": statistics.median(p95s),
            "median_of_run_means_ms": statistics.median(means),
            "p50_win_count": win_count("median_latency_ms"),
            "mean_win_count": win_count("mean_latency_ms"),
            "p95_win_count": win_count("p95_latency_ms"),
        })

    write_csv(args.result_dir / "dynamic_trace_phase_runs.csv", result_rows)
    write_csv(args.result_dir / "dynamic_trace_phase_aggregate.csv", aggregate_rows)
    if missing_rows:
        write_csv(args.result_dir / "dynamic_trace_phase_missing.csv", missing_rows)


if __name__ == "__main__":
    main()
