#!/usr/bin/env python3
"""Create a minimal weights_ops.json from elastic profile CSV.

This is a bootstrap helper for phone experiments. A fuller extractor should
eventually read GGUF metadata directly, but profile CSV already contains the
weight names and byte sizes needed by the first solver baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


WEIGHT_KINDS = {"LOAD", "TRANSFER", "XFORM", "RELOAD_ENSURE"}


def layer_of(name: str) -> int:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser(description="Build minimal model meta from profile CSV")
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    by_name: dict[str, dict] = {}
    compute_order: list[str] = []

    for path in args.csv:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("name", "")
                if not name or name == "ggml_cgraph":
                    continue
                kind = row.get("kind", "")
                if kind in WEIGHT_KINDS:
                    cur = by_name.setdefault(name, {"name": name, "byte_size": 0, "quant": ""})
                    cur["byte_size"] = max(cur["byte_size"], int(row.get("bytes", "0") or 0))
                if kind == "COMPUTE" and name not in compute_order:
                    compute_order.append(name)
                    cur = by_name.setdefault(name, {"name": name, "byte_size": 0, "quant": ""})
                    cur["byte_size"] = max(cur["byte_size"], int(row.get("bytes", "0") or 0))
                    cur["quant"] = row.get("quant", "") or cur.get("quant", "")

    ordered_names = compute_order + sorted(n for n in by_name if n not in set(compute_order))
    weights = []
    ops = []
    for i, name in enumerate(ordered_names):
        rec = by_name[name]
        weights.append(
            {
                "weight_id": i,
                "name": name,
                "layer": layer_of(name),
                "byte_size": int(rec.get("byte_size", 0)),
                "quant": rec.get("quant", ""),
            }
        )
        ops.append(
            {
                "op_id": i,
                "name": name,
                "layer": layer_of(name),
                "weight_id": i,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"weights": weights, "ops": ops}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "weights": len(weights), "ops": len(ops)}, indent=2))


if __name__ == "__main__":
    main()

