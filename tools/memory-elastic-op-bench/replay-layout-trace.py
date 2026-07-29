#!/usr/bin/env python3
"""Replay measured layout promotion costs over dynamic memory traces.

This is a characterization model, not an end-to-end runtime simulator. A
budget threshold partitions a real trace into intervals where a promoted
layout is feasible or unavailable. The replay compares immediate budget-only
promotion with keeping the generic layout and a per-window oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Window:
    trace: str
    percentile: float
    threshold_mib: float
    window_id: int
    start_sec: float
    end_sec: float
    executions: int
    promotion_ms: float
    gross_compute_saving_ms: float
    greedy_net_over_generic_ms: float
    oracle_promotes: int


@dataclass
class Event:
    trace: str
    percentile: float
    threshold_mib: float
    window_id: int
    t_sec: float
    event: str
    executions_in_window: int
    greedy_layout_after: str
    immediate_latency_delta_ms: float
    completed_window_net_ms: float | str


def read_trace(path: Path, duration_sec: float) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            t = float(row.get("t_sec", row.get("time_sec", i)) or i)
            budget = float(row.get("mem_available_mb", row.get("budget_mib", 0)) or 0)
            if t <= duration_sec:
                rows.append((t, budget))
    if not rows:
        raise ValueError(f"empty trace window: {path}")
    # For duplicate timestamps, the last observation is the state visible after
    # all updates at that timestamp.
    collapsed: dict[float, float] = {}
    for t, budget in rows:
        collapsed[t] = budget
    return sorted(collapsed.items())


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1)*p/100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    alpha = pos - lo
    return ordered[lo]*(1-alpha) + ordered[hi]*alpha


def sample_budgets(
    rows: list[tuple[float, float]], duration_sec: float, execution_interval_sec: float
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    row_index = 0
    current = rows[0][1]
    count = int(math.floor(duration_sec/execution_interval_sec))
    for i in range(count):
        t = i*execution_interval_sec
        while row_index + 1 < len(rows) and rows[row_index + 1][0] <= t + 1e-12:
            row_index += 1
            current = rows[row_index][1]
        result.append((t, current))
    return result


def find_high_windows(samples: list[tuple[float, float]], threshold: float) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start: int | None = None
    for i, (_, budget) in enumerate(samples):
        high = budget >= threshold
        if high and start is None:
            start = i
        elif not high and start is not None:
            windows.append((start, i))
            start = None
    if start is not None:
        windows.append((start, len(samples)))
    return windows


def replay(
    path: Path,
    rows: list[tuple[float, float]],
    duration_sec: float,
    interval_sec: float,
    p: float,
    generic_ms: float,
    promoted_ms: float,
    promotion_ms: float,
) -> tuple[dict[str, float | int | str], list[Window], list[Event]]:
    threshold = percentile([budget for _, budget in rows], p)
    samples = sample_budgets(rows, duration_sec, interval_sec)
    high_windows = find_high_windows(samples, threshold)
    per_use_saving = generic_ms - promoted_ms
    windows: list[Window] = []
    events: list[Event] = []
    high_executions = 0
    oracle_promotions = 0
    oracle_delta = 0.0
    losing_promotions = 0
    losing_promotion_waste = 0.0

    for window_id, (begin, end) in enumerate(high_windows, 1):
        executions = end - begin
        high_executions += executions
        gross_saving = executions*per_use_saving
        net = promotion_ms - gross_saving
        oracle_promotes = int(net < 0)
        oracle_promotions += oracle_promotes
        oracle_delta += min(0.0, net)
        if net >= 0:
            losing_promotions += 1
            losing_promotion_waste += net
        start_sec = samples[begin][0]
        end_sec = samples[end][0] if end < len(samples) else duration_sec
        windows.append(Window(
            trace=path.name,
            percentile=p,
            threshold_mib=threshold,
            window_id=window_id,
            start_sec=start_sec,
            end_sec=end_sec,
            executions=executions,
            promotion_ms=promotion_ms,
            gross_compute_saving_ms=gross_saving,
            greedy_net_over_generic_ms=net,
            oracle_promotes=oracle_promotes,
        ))
        events.append(Event(
            trace=path.name,
            percentile=p,
            threshold_mib=threshold,
            window_id=window_id,
            t_sec=start_sec,
            event="budget_up_promote",
            executions_in_window=executions,
            greedy_layout_after="promoted",
            immediate_latency_delta_ms=promotion_ms - per_use_saving,
            completed_window_net_ms="",
        ))
        if end < len(samples):
            events.append(Event(
                trace=path.name,
                percentile=p,
                threshold_mib=threshold,
                window_id=window_id,
                t_sec=end_sec,
                event="budget_down_evict",
                executions_in_window=executions,
                greedy_layout_after="generic",
                immediate_latency_delta_ms=per_use_saving,
                completed_window_net_ms=net,
            ))

    total_executions = len(samples)
    static_total = total_executions*generic_ms
    greedy_delta = len(high_windows)*promotion_ms - high_executions*per_use_saving
    greedy_total = static_total + greedy_delta
    oracle_total = static_total + oracle_delta
    break_even = promotion_ms/per_use_saving if per_use_saving > 0 else math.inf
    summary: dict[str, float | int | str] = {
        "trace": path.name,
        "percentile": p,
        "threshold_mib": threshold,
        "duration_sec": duration_sec,
        "execution_interval_sec": interval_sec,
        "total_executions": total_executions,
        "high_executions": high_executions,
        "high_windows": len(high_windows),
        "break_even_executions": break_even,
        "static_generic_total_ms": static_total,
        "budget_greedy_total_ms": greedy_total,
        "window_oracle_total_ms": oracle_total,
        "greedy_over_static_ms": greedy_delta,
        "greedy_over_oracle_ms": greedy_total - oracle_total,
        "greedy_promotions": len(high_windows),
        "oracle_promotions": oracle_promotions,
        "losing_promotions": losing_promotions,
        "losing_promotion_waste_ms": losing_promotion_waste,
        "promotion_time_ms": len(high_windows)*promotion_ms,
        "promotion_time_fraction": len(high_windows)*promotion_ms/greedy_total if greedy_total else 0.0,
    }
    return summary, windows, events


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--duration-sec", type=float, default=180.0)
    ap.add_argument("--execution-interval-ms", type=float, default=200.0)
    ap.add_argument("--threshold-percentile", type=float, action="append", default=[])
    ap.add_argument("--generic-ms", type=float, default=2.314)
    ap.add_argument("--promoted-ms", type=float, default=2.211)
    ap.add_argument("--promotion-ms", type=float, default=10.299)
    args = ap.parse_args()
    percentiles = args.threshold_percentile or [25.0, 50.0, 75.0]
    interval_sec = args.execution_interval_ms/1000.0
    if interval_sec <= 0 or args.duration_sec <= 0:
        raise SystemExit("duration and execution interval must be positive")

    summaries: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    patterns_dir = args.out_dir/"patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)
    for trace in args.trace:
        trace_rows = read_trace(trace, args.duration_sec)
        for p in percentiles:
            summary, trace_windows, trace_events = replay(
                trace, trace_rows, args.duration_sec, interval_sec, p,
                args.generic_ms, args.promoted_ms, args.promotion_ms,
            )
            summaries.append(summary)
            windows.extend(asdict(row) for row in trace_windows)
            events.extend(asdict(row) for row in trace_events)
            threshold = percentile([budget for _, budget in trace_rows], p)
            samples = sample_budgets(trace_rows, args.duration_sec, interval_sec)
            pattern_rows = [
                {"execution": i, "t_sec": t, "budget_mib": budget, "high": int(budget >= threshold)}
                for i, (t, budget) in enumerate(samples)
            ]
            write_csv(patterns_dir/f"{trace.stem}_p{p:g}.csv", pattern_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir/"summary.csv", summaries)
    write_csv(args.out_dir/"windows.csv", windows)
    write_csv(args.out_dir/"events.csv", events)
    metadata = {
        "generic_ms": args.generic_ms,
        "promoted_ms": args.promoted_ms,
        "promotion_ms": args.promotion_ms,
        "execution_interval_ms": args.execution_interval_ms,
        "duration_sec": args.duration_sec,
        "threshold_percentiles": percentiles,
        "traces": [str(path) for path in args.trace],
        "interpretation": "measured-cost characterization; not end-to-end runtime simulation",
    }
    (args.out_dir/"metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.out_dir/"summary.csv")


if __name__ == "__main__":
    main()
