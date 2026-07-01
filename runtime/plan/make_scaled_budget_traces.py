#!/usr/bin/env python3
"""Rescale real memory traces into model-specific budget pressure windows.

The output keeps the source trace timing and shape, but linearly maps memory
values into a requested target min/max range. This is useful when comparing
planner baselines across model sizes while preserving realistic temporal
budget trends from the collected traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def read_trace(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            t = float(row.get("t_sec", row.get("time_sec", i)) or i)
            m = float(row.get("mem_available_mb", row.get("budget_mib", 0)) or 0)
            rows.append((t, m))
    if not rows:
        raise ValueError(f"empty trace: {path}")
    rows.sort(key=lambda x: x[0])
    return rows


def planned_weight_mib(path: Path) -> float:
    data = json.loads(path.read_text())
    return sum(int(w.get("byte_size", w.get("bytes", 0)) or 0) for w in data.get("weights", [])) / 1024.0 / 1024.0


def trace_id_user(path: Path, fallback_id: int) -> tuple[int, int]:
    m = re.match(r"trace_(\d+)_user_(\d+)", path.stem)
    if not m:
        return fallback_id, 0
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Scale real traces into target memory-budget ranges")
    ap.add_argument("--source", type=Path, action="append", required=True, help="source trace; may be repeated")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-min-mib", type=float, required=True, help="target raw memory trace minimum")
    ap.add_argument("--target-max-mib", type=float, required=True, help="target raw memory trace maximum")
    ap.add_argument("--trace-id-offset", type=int, default=30)
    ap.add_argument("--tag", default="", help="optional suffix before .csv")
    ap.add_argument("--model-meta", type=Path)
    ap.add_argument("--kv-mib", type=float, default=512.0)
    ap.add_argument("--misc-mib", type=float, default=256.0)
    ap.add_argument("--safety-mib", type=float, default=64.0)
    ap.add_argument("--pinned-extra-mib", type=float, default=410.0)
    args = ap.parse_args()

    if args.target_max_mib <= args.target_min_mib:
        raise SystemExit("--target-max-mib must be greater than --target-min-mib")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []
    overhead = args.kv_mib + args.misc_mib + args.safety_mib + args.pinned_extra_mib
    planned = planned_weight_mib(args.model_meta) if args.model_meta else 0.0

    for idx, source in enumerate(args.source):
        rows = read_trace(source)
        vals = [m for _, m in rows]
        src_min, src_max = min(vals), max(vals)
        if src_max <= src_min:
            scaled = [(t, args.target_min_mib) for t, _ in rows]
        else:
            scaled = [
                (
                    t,
                    args.target_min_mib
                    + (m - src_min) / (src_max - src_min) * (args.target_max_mib - args.target_min_mib),
                )
                for t, m in rows
            ]

        src_trace_id, user_id = trace_id_user(source, args.trace_id_offset + idx)
        out_trace_id = args.trace_id_offset + idx
        tag = f"_{args.tag}" if args.tag else ""
        out = args.out_dir / f"trace_{out_trace_id:02d}_user_{user_id}{tag}.csv"
        with out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_sec", "mem_available_mb"])
            for t, m in scaled:
                writer.writerow([f"{t:.3f}".rstrip("0").rstrip("."), f"{m:.1f}"])

        out_vals = [m for _, m in scaled]
        summary_rows.append(
            {
                "source": str(source),
                "output": str(out),
                "source_trace_id": str(src_trace_id),
                "source_min_mean_max": f"{src_min:.1f}/{sum(vals)/len(vals):.1f}/{src_max:.1f}",
                "target_min_mean_max": f"{min(out_vals):.1f}/{sum(out_vals)/len(out_vals):.1f}/{max(out_vals):.1f}",
                "planner_weight_budget_min_max": f"{min(out_vals)-overhead:.1f}/{max(out_vals)-overhead:.1f}",
                "planned_weight_mib": f"{planned:.1f}" if planned else "",
                "max_weight_gap_mib": f"{planned-(max(out_vals)-overhead):.1f}" if planned else "",
            }
        )

    summary = args.out_dir / "SUMMARY.md"
    lines = [
        "# Scaled Budget Traces",
        "",
        f"- target raw min/max: `{args.target_min_mib:.1f}/{args.target_max_mib:.1f} MiB`",
        f"- overhead subtracted for planner weight budget: `{overhead:.1f} MiB`",
    ]
    if planned:
        lines.append(f"- planned weights: `{planned:.1f} MiB`")
    lines += [
        "",
        "| source | output | source min/mean/max | scaled min/mean/max | planner weight budget min/max | max weight gap |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['source']} | {row['output']} | {row['source_min_mean_max']} | "
            f"{row['target_min_mean_max']} | {row['planner_weight_budget_min_max']} | {row['max_weight_gap_mib']} |"
        )
    summary.write_text("\n".join(lines) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
