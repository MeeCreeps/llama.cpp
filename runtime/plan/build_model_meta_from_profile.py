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


def order_key(name: str) -> tuple[int, int, str]:
    layer = layer_of(name)
    if layer >= 0:
        return (0, layer, name)
    return (1, 0, name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build minimal model meta from profile CSV")
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-meta", type=Path, default=None,
                    help="optional existing weights_ops.json to merge with profile-derived weights")
    ap.add_argument("--include-output-weight", action="store_true",
                    help="append output.weight when result_output compute profile exists")
    ap.add_argument("--output-weight-byte-size", type=int, default=0,
                    help="byte size for output.weight; required when --include-output-weight is used without GGUF extraction")
    ap.add_argument("--max-layer-exclusive", type=int, default=None,
                    help="drop blk.N weights with N >= this value; useful for GGUFs that carry inactive extra tensors")
    args = ap.parse_args()

    by_name: dict[str, dict] = {}
    if args.base_meta:
        base = json.loads(args.base_meta.read_text())
        for w in base.get("weights", []):
            name = str(w.get("name", ""))
            if not name:
                continue
            by_name[name] = {
                "name": name,
                "byte_size": int(w.get("byte_size", w.get("bytes", 0)) or 0),
                "quant": str(w.get("quant", "")),
            }

    saw_result_output_compute = False
    for path in args.csv:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("name", "")
                if not name or name == "ggml_cgraph":
                    continue
                kind = row.get("kind", "")
                if name == "result_output" and kind == "COMPUTE":
                    saw_result_output_compute = True
                if kind in WEIGHT_KINDS:
                    cur = by_name.setdefault(name, {"name": name, "byte_size": 0, "quant": ""})
                    cur["byte_size"] = max(cur["byte_size"], int(row.get("bytes", "0") or 0))

    if args.include_output_weight and saw_result_output_compute and "output.weight" not in by_name:
        if args.output_weight_byte_size <= 0:
            raise SystemExit("--output-weight-byte-size is required to append output.weight")
        by_name["output.weight"] = {
            "name": "output.weight",
            "byte_size": int(args.output_weight_byte_size),
            "quant": "",
        }

    ordered_names = sorted(by_name, key=order_key)
    weights = []
    ops = []
    for i, name in enumerate(ordered_names):
        layer = layer_of(name)
        if args.max_layer_exclusive is not None and layer >= args.max_layer_exclusive:
            continue
        rec = by_name[name]
        weight_id = len(weights)
        weights.append(
            {
                "weight_id": weight_id,
                "name": name,
                "layer": layer,
                "byte_size": int(rec.get("byte_size", 0)),
                "quant": rec.get("quant", ""),
            }
        )
        ops.append(
            {
                "op_id": weight_id,
                "name": name,
                "layer": layer,
                "weight_id": weight_id,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"weights": weights, "ops": ops}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "weights": len(weights), "ops": len(ops)}, indent=2))


if __name__ == "__main__":
    main()
