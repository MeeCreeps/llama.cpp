#!/usr/bin/env python3
"""Create elastic planner weights_ops.json directly from a GGUF file."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))

from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402


CORE_WEIGHT_RE = re.compile(
    r"^blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$"
)


def layer_of(name: str) -> int:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def order_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    name = str(row.get("name", ""))
    layer = int(row.get("layer", layer_of(name)))
    component_order = {
        "attn_k": 0,
        "attn_output": 1,
        "attn_q": 2,
        "attn_v": 3,
        "ffn_down": 4,
        "ffn_gate": 5,
        "ffn_up": 6,
    }
    comp = 99
    m = CORE_WEIGHT_RE.match(name)
    if m:
        comp = component_order.get(m.group(2), 99)
    return (0 if layer >= 0 else 1, layer, comp, name)


def quant_numeric_string(tensor_type: Any) -> str:
    try:
        return str(int(tensor_type))
    except Exception:
        try:
            return str(int(tensor_type.value))
        except Exception:
            return ""


def quant_name(tensor_type: Any) -> str:
    try:
        return GGMLQuantizationType(tensor_type).name
    except Exception:
        return str(tensor_type)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--include-all-tensors",
        action="store_true",
        help="include every GGUF tensor instead of only core layer matmul weights",
    )
    args = parser.parse_args()

    reader = GGUFReader(str(args.gguf))
    rows: list[dict[str, Any]] = []
    skipped = 0
    for tensor in reader.tensors:
        name = str(tensor.name)
        match = CORE_WEIGHT_RE.match(name)
        if not args.include_all_tensors and not match:
            skipped += 1
            continue
        rows.append(
            {
                "name": name,
                "layer": int(match.group(1)) if match else layer_of(name),
                "byte_size": int(tensor.n_bytes),
                # Existing planner metadata historically uses the numeric GGML
                # quantization id, e.g. "2" for Q4_0.
                "quant": quant_numeric_string(tensor.tensor_type),
                "quant_name": quant_name(tensor.tensor_type),
                "shape": [int(x) for x in tensor.shape],
            }
        )

    rows.sort(key=order_key)
    weights = []
    ops = []
    for i, row in enumerate(rows):
        weight = {
            "weight_id": i,
            "name": row["name"],
            "layer": row["layer"],
            "byte_size": row["byte_size"],
            "quant": row["quant"],
            "quant_name": row["quant_name"],
            "shape": row["shape"],
        }
        weights.append(weight)
        ops.append(
            {
                "op_id": i,
                "name": row["name"],
                "layer": row["layer"],
                "weight_id": i,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"weights": weights, "ops": ops}, indent=2, sort_keys=True) + "\n")
    total_bytes = sum(int(w["byte_size"]) for w in weights)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "weights": len(weights),
                "ops": len(ops),
                "planned_bytes": total_bytes,
                "planned_mib": total_bytes / 1024 / 1024,
                "skipped": skipped,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
