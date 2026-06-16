#!/usr/bin/env python3
"""Generate a native ExecPlan for a dynamic memory budget baseline.

This is intentionally a conservative first implementation:
  * one weight tensor is the movement unit;
  * compute backend is selected per op from cost table when available;
  * resident selection is a 0/1 budget problem;
  * OR-Tools CP-SAT is used if installed, otherwise deterministic greedy is used.

Expected model meta schema:
{
  "weights": [
    {"weight_id": 0, "name": "blk.0.ffn_down.weight", "layer": 0,
     "byte_size": 1234, "quant": "Q4_0"}
  ],
  "ops": [
    {"op_id": 0, "name": "blk.0.ffn_down.weight", "layer": 0, "weight_id": 0}
  ]
}
If ops are omitted, one op per weight is inferred.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MB = 1024 * 1024


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_cost_records(cost_dir: Path, filename: str) -> list[dict[str, Any]]:
    path = cost_dir / filename
    if not path.exists():
        return []
    return load_json(path).get("records", [])


class CostModel:
    def __init__(self, cost_dir: Path):
        self.stage = load_cost_records(cost_dir, "stage_costs.json")
        self.ops = load_cost_records(cost_dir, "op_costs.json")

    def stage_ms(self, backend: str, kind: str, name: str, byte_size: int) -> float:
        best = None
        for rec in self.stage:
            if rec.get("backend") != backend:
                continue
            if rec.get("kind") != kind:
                continue
            if rec.get("name") == name:
                return float(rec.get("median_ms", 0.0))
            if int(rec.get("bytes", 0)) == byte_size and best is None:
                best = rec
        if best:
            return float(best.get("median_ms", 0.0))
        # Phone-first fallback constants. These are deliberately rough and are
        # replaced by measured CSV data as soon as it exists.
        mb = byte_size / MB
        if kind in ("LOAD", "RELOAD_ENSURE"):
            return 0.15 + mb / 1800.0 * 1000.0
        if kind == "TRANSFER":
            return 0.05 + mb / 6000.0 * 1000.0
        if kind == "XFORM":
            return 0.10 + mb / 8000.0 * 1000.0
        return 0.0

    def compute_ms(self, backend: str, name: str, byte_size: int) -> float:
        best = None
        for rec in self.ops:
            if rec.get("backend") != backend:
                continue
            if rec.get("name") == name:
                return float(rec.get("median_ms", 0.0))
            if int(rec.get("bytes", 0)) == byte_size and best is None:
                best = rec
        if best:
            return float(best.get("median_ms", 0.0))
        mb = byte_size / MB
        if backend == "OpenCL":
            return 0.03 + mb * 0.020
        return 0.05 + mb * 0.045


def normalize_meta(meta: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weights = list(meta.get("weights", []))
    for i, w in enumerate(weights):
        w.setdefault("weight_id", i)
        w.setdefault("name", f"weight_{i}")
        w.setdefault("layer", -1)
        w.setdefault("byte_size", int(w.get("bytes", 0)))
        w.setdefault("quant", "")

    ops = list(meta.get("ops", []))
    if not ops:
        ops = [
            {
                "op_id": i,
                "name": w["name"],
                "layer": w.get("layer", -1),
                "weight_id": w["weight_id"],
            }
            for i, w in enumerate(weights)
        ]
    for i, op in enumerate(ops):
        op.setdefault("op_id", i)
        op.setdefault("name", f"op_{i}")
        op.setdefault("layer", -1)
        op.setdefault("weight_id", i if i < len(weights) else -1)
    return weights, ops


def load_state(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    data = load_json(path)
    rows = data.get("weights", data if isinstance(data, list) else [])
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        if name:
            out[name] = row
    return out


def state_has_backend(row: dict[str, Any] | None, backend: str) -> bool:
    if not row:
        return False
    flags = row.get("flags", [])
    if isinstance(flags, int):
        if backend == "GPU":
            return bool(flags & (1 << 4))
        return bool(flags & (1 << 2))
    if not isinstance(flags, list):
        return False
    if backend == "GPU":
        return "gpu_compute_resident" in flags or "GPU_COMPUTE_RESIDENT" in flags
    return "cpu_compute_resident" in flags or "CPU_COMPUTE_RESIDENT" in flags


def state_any_resident(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    flags = row.get("flags", [])
    if isinstance(flags, int):
        return bool(flags & ((1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)))
    if not isinstance(flags, list):
        return False
    resident_names = {
        "cpu_raw_resident", "CPU_RAW_RESIDENT",
        "cpu_compute_resident", "CPU_COMPUTE_RESIDENT",
        "gpu_raw_resident", "GPU_RAW_RESIDENT",
        "gpu_compute_resident", "GPU_COMPUTE_RESIDENT",
    }
    return any(f in resident_names for f in flags)


def choose_backends(weights: list[dict[str, Any]], ops: list[dict[str, Any]], cm: CostModel) -> dict[int, str]:
    by_id = {int(w["weight_id"]): w for w in weights}
    out: dict[int, str] = {}
    for op in ops:
        wid = int(op.get("weight_id", -1))
        w = by_id.get(wid)
        if not w:
            out[int(op["op_id"])] = "GPU"
            continue
        name = str(w["name"])
        size = int(w.get("byte_size", 0))
        gpu_ms = cm.compute_ms("OpenCL", name, size)
        cpu_ms = cm.compute_ms("CPU_Elastic", name, size)
        out[int(op["op_id"])] = "GPU" if gpu_ms <= cpu_ms else "CPU"
    return out


def weight_value(weight: dict[str, Any], backend: str, cm: CostModel,
                 state: dict[str, dict[str, Any]]) -> float:
    name = str(weight["name"])
    size = int(weight.get("byte_size", 0))
    row = state.get(name)
    if state_has_backend(row, backend):
        # Keeping an already-correct resident tensor avoids both reload and
        # evict/reload churn, so bias the resident selector toward it.
        return 1e9 + size / MB
    if backend == "GPU":
        return (
            cm.stage_ms("OpenCL", "LOAD", name, size)
            + cm.stage_ms("OpenCL", "TRANSFER", name, size)
            + cm.stage_ms("OpenCL", "XFORM", name, size)
        )
    return (
        cm.stage_ms("CPU_Elastic", "LOAD", name, size)
        + cm.stage_ms("CPU_Elastic", "XFORM", name, size)
    )


def select_resident_cp(items: list[tuple[int, int, float]], budget_bytes: int, time_limit_ms: int) -> set[int] | None:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        return None

    model = cp_model.CpModel()
    x = {wid: model.NewBoolVar(f"resident_{wid}") for wid, _, _ in items}
    model.Add(sum(size * x[wid] for wid, size, _ in items) <= budget_bytes)
    # CP-SAT integer objective; preserve three decimal places.
    model.Maximize(sum(int(value * 1000.0) * x[wid] for wid, _, value in items))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(time_limit_ms, 1) / 1000.0
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return set()
    return {wid for wid, _, _ in items if solver.Value(x[wid]) == 1}


def select_resident_greedy(items: list[tuple[int, int, float]], budget_bytes: int) -> set[int]:
    ordered = sorted(items, key=lambda it: (it[2] / max(it[1], 1), it[2]), reverse=True)
    total = 0
    keep: set[int] = set()
    for wid, size, _ in ordered:
        if total + size <= budget_bytes:
            keep.add(wid)
            total += size
    return keep


def xform_for_backend(backend: str, quant: str) -> str:
    if backend == "GPU":
        if quant in ("Q4_0", "Q8_0", "MXFP4") or not quant:
            return "gpu_convert"
        return "none"
    return "cpu_repack"


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    meta = load_json(args.model_meta)
    weights, ops = normalize_meta(meta)
    state = load_state(args.state)
    cm = CostModel(args.cost_dir)

    backend_by_op = choose_backends(weights, ops, cm)
    backend_by_weight: dict[int, str] = {}
    for op in ops:
        wid = int(op.get("weight_id", -1))
        backend_by_weight.setdefault(wid, backend_by_op[int(op["op_id"])])

    budget_bytes = max(0, (args.budget_mib - args.kv_mib - args.misc_mib - args.safety_mib) * MB)
    items = []
    for w in weights:
        wid = int(w["weight_id"])
        size = int(w.get("byte_size", 0))
        be = backend_by_weight.get(wid, "GPU")
        items.append((wid, size, weight_value(w, be, cm, state)))

    keep = select_resident_cp(items, budget_bytes, args.time_limit_ms)
    solver_kind = "cp_sat"
    if keep is None:
        keep = select_resident_greedy(items, budget_bytes)
        solver_kind = "greedy"

    by_id = {int(w["weight_id"]): w for w in weights}
    first_consumer: dict[int, int] = {}
    for op in ops:
        wid = int(op.get("weight_id", -1))
        first_consumer.setdefault(wid, int(op["op_id"]))

    plan_weights = []
    timeline = []
    pred_ms = 0.0
    for w in weights:
        wid = int(w["weight_id"])
        be = backend_by_weight.get(wid, "GPU")
        resident = wid in keep
        quant = str(w.get("quant", ""))
        location = be.lower() if resident else "disk"
        xform = xform_for_backend(be, quant) if resident else "none"
        plan_weights.append(
            {
                "weight_id": wid,
                "name": w["name"],
                "layer": int(w.get("layer", -1)),
                "byte_size": int(w.get("byte_size", 0)),
                "location": location,
                "pinned": False,
                "xform": xform,
            }
        )
        if not resident:
            anchor = max(0, first_consumer.get(wid, 0) - args.prefetch_distance)
            size = int(w.get("byte_size", 0))
            if state_any_resident(state.get(str(w["name"]))):
                from_loc = "gpu" if state_has_backend(state.get(str(w["name"])), "GPU") else "cpu"
                timeline.append(
                    {"kind": "evict", "weight_id": wid, "from_loc": from_loc, "to_loc": "disk", "engine": "cpu", "anchor_op_id": 0, "overlap_group": -1}
                )
            if be == "GPU":
                pred_ms += cm.stage_ms("OpenCL", "LOAD", str(w["name"]), size)
                pred_ms += cm.stage_ms("OpenCL", "TRANSFER", str(w["name"]), size)
                pred_ms += cm.stage_ms("OpenCL", "XFORM", str(w["name"]), size)
                timeline.extend(
                    [
                        {"kind": "load", "weight_id": wid, "from_loc": "disk", "to_loc": "cpu", "engine": "disk", "anchor_op_id": anchor, "overlap_group": -1},
                        {"kind": "transfer", "weight_id": wid, "from_loc": "cpu", "to_loc": "gpu", "engine": "transfer", "anchor_op_id": anchor, "overlap_group": -1},
                        {"kind": "xform", "weight_id": wid, "from_loc": "gpu", "to_loc": "gpu", "engine": "gpu", "anchor_op_id": anchor, "overlap_group": -1},
                    ]
                )
            else:
                pred_ms += cm.stage_ms("CPU_Elastic", "LOAD", str(w["name"]), size)
                pred_ms += cm.stage_ms("CPU_Elastic", "XFORM", str(w["name"]), size)
                timeline.extend(
                    [
                        {"kind": "load", "weight_id": wid, "from_loc": "disk", "to_loc": "cpu", "engine": "disk", "anchor_op_id": anchor, "overlap_group": -1},
                        {"kind": "xform", "weight_id": wid, "from_loc": "cpu", "to_loc": "cpu", "engine": "cpu", "anchor_op_id": anchor, "overlap_group": -1},
                    ]
                )

    plan_ops = []
    for op in ops:
        wid = int(op.get("weight_id", -1))
        be = backend_by_op[int(op["op_id"])]
        w = by_id.get(wid, {})
        pred_ms += cm.compute_ms("OpenCL" if be == "GPU" else "CPU_Elastic", str(w.get("name", op["name"])), int(w.get("byte_size", 0)))
        plan_ops.append(
            {
                "op_id": int(op["op_id"]),
                "name": op["name"],
                "layer": int(op.get("layer", -1)),
                "compute_backend": be.lower(),
                "weight_id": wid,
                "dispatch": "static",
                "migrate": False,
                "migrate_from": "cpu",
                "migrate_xform": "none",
            }
        )

    return {
        "schema_version": 1,
        "budget_mib": args.budget_mib,
        "kv_bytes": args.kv_mib * MB,
        "misc_bytes": args.misc_mib * MB,
        "weights": plan_weights,
        "ops": plan_ops,
        "timeline": timeline,
        "pred_per_token_ms": pred_ms,
        "bottleneck": solver_kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an elastic native ExecPlan for one budget")
    ap.add_argument("--model-meta", type=Path, required=True)
    ap.add_argument("--cost-dir", type=Path, required=True)
    ap.add_argument("--state", type=Path, default=None, help="optional current residency state JSON for online baseline")
    ap.add_argument("--budget-mib", type=int, required=True)
    ap.add_argument("--kv-mib", type=int, default=128)
    ap.add_argument("--misc-mib", type=int, default=256)
    ap.add_argument("--safety-mib", type=int, default=64)
    ap.add_argument("--prefetch-distance", type=int, default=1)
    ap.add_argument("--time-limit-ms", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan = build_plan(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "budget_mib": args.budget_mib, "weights": len(plan["weights"]), "ops": len(plan["ops"]), "timeline": len(plan["timeline"]), "solver": plan["bottleneck"]}, indent=2))


if __name__ == "__main__":
    main()
