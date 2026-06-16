#!/usr/bin/env python3
"""Build an elastic memory cost model from GGML_ELASTIC_PROFILE_CSV output.

The input CSV is intentionally simple so it can be produced on Android and then
pulled to the host. This script has no pandas dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


STAGE_KINDS = {"LOAD", "TRANSFER", "XFORM", "RELOAD_ENSURE"}
COMPUTE_KINDS = {"COMPUTE", "COMPUTE_GRAPH"}
BOUNDARY_KINDS = {"BOUNDARY"}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - rank) + xs[hi] * (rank - lo)


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, "") or default)
    except ValueError:
        return default


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def shape_of(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (
        as_int(row, "ne0"),
        as_int(row, "ne1"),
        as_int(row, "ne2"),
        as_int(row, "ne3"),
    )


def group_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        row.get("backend", ""),
        row.get("kind", ""),
        row.get("name", ""),
        row.get("op", ""),
        row.get("quant", ""),
        shape_of(row),
        as_int(row, "bytes"),
        row.get("extra", ""),
    )


def summarize(rows: list[dict[str, str]], kind_filter: set[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("kind", "") not in kind_filter:
            continue
        if as_int(row, "ok", 1) == 0:
            continue
        grouped[group_key(row)].append(row)

    records: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        backend, kind, name, op, quant, shape, byte_size, extra = key
        ms = [as_float(r, "ms") for r in items]
        total_bytes = sum(as_int(r, "bytes") for r in items)
        total_ms = sum(ms)
        records.append(
            {
                "backend": backend,
                "kind": kind,
                "name": name,
                "op": op,
                "quant": quant,
                "shape": list(shape),
                "bytes": byte_size,
                "extra": extra,
                "samples": len(items),
                "median_ms": median(ms) if ms else 0.0,
                "p90_ms": percentile(ms, 0.90),
                "mean_ms": total_ms / len(ms) if ms else 0.0,
                "total_ms": total_ms,
                "total_bytes": total_bytes,
                "mbps": (total_bytes / 1024.0 / 1024.0) / (total_ms / 1000.0) if total_ms > 0 else 0.0,
            }
        )
    return records


def load_csv(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build elastic cost model JSON from profile CSV")
    ap.add_argument("csv", nargs="+", type=Path, help="GGML_ELASTIC_PROFILE_CSV file(s)")
    ap.add_argument("--out-dir", type=Path, required=True, help="output cost model directory")
    ap.add_argument("--device", default="android-opencl", help="device tag")
    ap.add_argument("--model", default="", help="model tag/name")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    meta = {
        "device": args.device,
        "model": args.model,
        "source_csv": [str(p) for p in args.csv],
        "input_rows": len(rows),
    }

    write_json(args.out_dir / "stage_costs.json", {**meta, "records": summarize(rows, STAGE_KINDS)})
    write_json(args.out_dir / "op_costs.json", {**meta, "records": summarize(rows, COMPUTE_KINDS)})
    write_json(args.out_dir / "boundary_costs.json", {**meta, "records": summarize(rows, BOUNDARY_KINDS)})

    summary = {
        **meta,
        "stage_records": len(summarize(rows, STAGE_KINDS)),
        "op_records": len(summarize(rows, COMPUTE_KINDS)),
        "boundary_records": len(summarize(rows, BOUNDARY_KINDS)),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

