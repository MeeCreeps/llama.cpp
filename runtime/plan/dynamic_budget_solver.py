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
import re
from pathlib import Path
from typing import Any


MB = 1024 * 1024
BACKENDS = ("CPU", "GPU")
PLACEMENTS = ("cpu", "gpu", "disk_cpu", "disk_gpu")


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

    def measured_stage_ms(self, backend: str, kind: str, name: str, byte_size: int) -> float | None:
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
        return None

    def stage_ms(self, backend: str, kind: str, name: str, byte_size: int) -> float:
        measured = self.measured_stage_ms(backend, kind, name, byte_size)
        if measured is not None:
            return measured
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

    @staticmethod
    def compute_profile_names(name: str) -> list[str]:
        """Map llama weight names to ggml/OpenCL graph node names.

        OpenCL profiling records compute on graph nodes such as ``Qcur-0`` and
        ``ffn_gate-0``. The planner operates on model weights such as
        ``blk.0.attn_q.weight``. Bytes cannot be used as the primary key here:
        compute rows report activation/output bytes, not weight-file bytes.
        """
        out = [name]
        m = re.match(r"blk\.(\d+)\.(.+)\.weight$", name)
        if not m:
            if name == "output.weight":
                out.append("result_output")
            return out
        layer = m.group(1)
        suffix = m.group(2)
        mapped = {
            "attn_q": f"Qcur-{layer}",
            "attn_k": f"Kcur-{layer}",
            "attn_v": f"Vcur-{layer}",
            "attn_output": f"attn_out-{layer}",
            "ffn_gate": f"ffn_gate-{layer}",
            "ffn_up": f"ffn_up-{layer}",
            "ffn_down": f"ffn_out-{layer}",
        }.get(suffix)
        if mapped:
            out.append(mapped)
        return out

    def measured_compute_ms(self, backend: str, name: str, byte_size: int) -> float | None:
        candidates = set(self.compute_profile_names(name))
        exact = None
        best_mul_mat = None
        best = None
        for rec in self.ops:
            if rec.get("backend") != backend:
                continue
            if rec.get("kind") != "COMPUTE":
                continue
            if rec.get("name") in candidates:
                if rec.get("op") == "MUL_MAT":
                    if best_mul_mat is None or int(rec.get("samples", 0)) > int(best_mul_mat.get("samples", 0)):
                        best_mul_mat = rec
                    continue
                if exact is None:
                    exact = rec
            if rec.get("name") == name and int(rec.get("bytes", 0)) == byte_size and best is None:
                best = rec
        if best_mul_mat:
            return float(best_mul_mat.get("median_ms", 0.0))
        if exact:
            return float(exact.get("median_ms", 0.0))
        if best:
            return float(best.get("median_ms", 0.0))
        return None

    def compute_ms(self, backend: str, name: str, byte_size: int) -> float:
        measured = self.measured_compute_ms(backend, name, byte_size)
        if measured is not None:
            return measured
        mb = byte_size / MB
        if backend == "OpenCL":
            return 0.03 + mb * 0.020
        return 0.05 + mb * 0.045

    def has_measured_backend_compute(self, backend: str) -> bool:
        return any(rec.get("backend") == backend and rec.get("kind") == "COMPUTE" for rec in self.ops)

    def has_measured_stage_backend(self, backend: str) -> bool:
        return any(rec.get("backend") == backend for rec in self.stage)

    def has_measured_stage_kind(self, backend: str, kind: str) -> bool:
        return any(rec.get("backend") == backend and rec.get("kind") == kind for rec in self.stage)

    def gpu_reload_ms(self, name: str, byte_size: int) -> float:
        measured = self.measured_stage_ms("OpenCL", "RELOAD_ENSURE", name, byte_size)
        if measured is not None:
            return measured
        return (
            self.stage_ms("OpenCL", "LOAD", name, byte_size)
            + self.stage_ms("OpenCL", "TRANSFER", name, byte_size)
            + self.stage_ms("OpenCL", "XFORM", name, byte_size)
        )

    def cpu_reload_ms(self, name: str, byte_size: int) -> float:
        measured = self.measured_stage_ms("CPU_Elastic", "RELOAD_ENSURE", name, byte_size)
        if measured is not None:
            return measured
        return (
            self.stage_ms("CPU_Elastic", "LOAD", name, byte_size)
            + self.stage_ms("CPU_Elastic", "XFORM", name, byte_size)
        )

    def cpu_to_gpu_ms(self, name: str, byte_size: int) -> float:
        return (
            self.stage_ms("OpenCL", "TRANSFER", name, byte_size)
            + self.stage_ms("OpenCL", "XFORM", name, byte_size)
        )

    def compute_backend_ms(self, backend: str, name: str, byte_size: int, allow_cpu_fallback: bool) -> float:
        if backend == "CPU":
            if not allow_cpu_fallback and not self.has_measured_backend_compute("CPU_Elastic"):
                return math.inf
            return self.compute_ms("CPU_Elastic", name, byte_size)
        return self.compute_ms("OpenCL", name, byte_size)


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


def state_has_disk(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    flags = row.get("flags", [])
    if isinstance(flags, int):
        return bool(flags & 1)
    if not isinstance(flags, list):
        return False
    return "disk_available" in flags or "DISK_AVAILABLE" in flags


def backend_from_state(row: dict[str, Any] | None, fallback: str, quant: str) -> str:
    if quant:
        return fallback
    gpu = state_has_backend(row, "GPU")
    cpu = state_has_backend(row, "CPU")
    if gpu and not cpu:
        return "GPU"
    if cpu and not gpu:
        return "CPU"
    return fallback


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


def placement_location(choice: str) -> str:
    return "disk" if choice.startswith("disk_") else choice


def placement_backend(choice: str) -> str:
    if choice.endswith("_cpu") or choice == "cpu":
        return "CPU"
    return "GPU"


def placement_resident_bytes(choice: str, byte_size: int) -> int:
    return 0 if choice.startswith("disk_") else byte_size


def current_location(row: dict[str, Any] | None) -> str:
    if state_has_backend(row, "GPU"):
        return "gpu"
    if state_has_backend(row, "CPU"):
        return "cpu"
    return "disk"


def transition_ms(choice: str, name: str, size: int, row: dict[str, Any] | None, cm: CostModel) -> float:
    src = current_location(row)
    dst = placement_location(choice)
    backend = placement_backend(choice)
    if src == dst:
        return 0.0
    if dst == "disk":
        return 0.0
    if dst == "gpu":
        if src == "cpu":
            return cm.cpu_to_gpu_ms(name, size)
        return cm.gpu_reload_ms(name, size)
    # dst == cpu
    if src == "gpu":
        # No GPU->CPU readback stage exists in the current runtime. Model
        # weights are disk-backed, so represent GPU->CPU placement as evict +
        # disk->CPU reload/repack.
        return cm.cpu_reload_ms(name, size)
    return cm.cpu_reload_ms(name, size)


def steady_ms(choice: str, name: str, size: int, cm: CostModel, allow_cpu_fallback: bool) -> float:
    backend = placement_backend(choice)
    compute = cm.compute_backend_ms(backend, name, size, allow_cpu_fallback)
    if math.isinf(compute):
        return math.inf
    if choice == "disk_gpu":
        return compute + cm.gpu_reload_ms(name, size)
    if choice == "disk_cpu":
        return compute + cm.cpu_reload_ms(name, size)
    return compute


def placement_costs(weight: dict[str, Any], cm: CostModel, state: dict[str, dict[str, Any]],
                    allow_cpu_fallback: bool, transition_weight: float) -> dict[str, float]:
    name = str(weight["name"])
    size = int(weight.get("byte_size", 0))
    row = state.get(name)
    out: dict[str, float] = {}
    for choice in PLACEMENTS:
        steady = steady_ms(choice, name, size, cm, allow_cpu_fallback)
        if math.isinf(steady):
            out[choice] = math.inf
            continue
        out[choice] = steady + transition_weight * transition_ms(choice, name, size, row, cm)
    return out


def select_placements_cp(items: list[tuple[int, int, dict[str, float]]], budget_bytes: int,
                         time_limit_ms: int) -> dict[int, str] | None:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        return None

    model = cp_model.CpModel()
    x: dict[tuple[int, str], Any] = {}
    for wid, _, costs in items:
        feasible = [choice for choice in PLACEMENTS if not math.isinf(costs.get(choice, math.inf))]
        if not feasible:
            feasible = ["disk_gpu"]
        for choice in feasible:
            x[(wid, choice)] = model.NewBoolVar(f"{choice}_{wid}")
        model.Add(sum(x[(wid, choice)] for choice in feasible) == 1)
    model.Add(
        sum(size * x[(wid, choice)]
            for wid, size, costs in items
            for choice in PLACEMENTS
            if (wid, choice) in x and placement_resident_bytes(choice, size) > 0) <= budget_bytes
    )
    model.Minimize(
        sum(int(max(0.0, costs.get(choice, math.inf)) * 1000.0) * x[(wid, choice)]
            for wid, _, costs in items
            for choice in PLACEMENTS
            if (wid, choice) in x)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(time_limit_ms, 1) / 1000.0
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {}
    out: dict[int, str] = {}
    for wid, _, _ in items:
        for choice in PLACEMENTS:
            if (wid, choice) in x and solver.Value(x[(wid, choice)]) == 1:
                out[wid] = choice
                break
    return out


def select_placements_greedy(items: list[tuple[int, int, dict[str, float]]], budget_bytes: int) -> dict[int, str]:
    out: dict[int, str] = {}
    used = 0
    upgrades = []
    for wid, size, costs in items:
        disk_choices = [c for c in ("disk_cpu", "disk_gpu") if not math.isinf(costs.get(c, math.inf))]
        base = min(disk_choices or ["disk_gpu"], key=lambda c: costs.get(c, math.inf))
        out[wid] = base
        for resident in ("cpu", "gpu"):
            if math.isinf(costs.get(resident, math.inf)):
                continue
            saving = costs.get(base, math.inf) - costs[resident]
            if saving > 0:
                upgrades.append((saving / max(size, 1), saving, wid, resident, size))
    for _, _, wid, resident, size in sorted(upgrades, reverse=True):
        if out.get(wid) in ("cpu", "gpu"):
            continue
        if used + size <= budget_bytes:
            out[wid] = resident
            used += size
    return out


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
    has_state = args.state is not None

    budget_bytes = max(0, (args.budget_mib - args.kv_mib - args.misc_mib - args.safety_mib) * MB)
    objective_transition_weight = float(args.transition_weight) if has_state else 0.0
    items = []
    for w in weights:
        wid = int(w["weight_id"])
        size = int(w.get("byte_size", 0))
        costs = placement_costs(
            w, cm, state,
            allow_cpu_fallback=bool(args.allow_cpu_fallback),
            transition_weight=objective_transition_weight,
        )
        items.append((wid, size, costs))

    placements = select_placements_cp(items, budget_bytes, args.time_limit_ms)
    solver_kind = "cp_sat"
    if placements is None:
        placements = select_placements_greedy(items, budget_bytes)
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
        choice = placements.get(wid, "disk_gpu")
        be = placement_backend(choice)
        quant = str(w.get("quant", ""))
        location = placement_location(choice)
        xform = xform_for_backend(be, quant) if location != "disk" else "none"
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
        name = str(w["name"])
        size = int(w.get("byte_size", 0))
        row = state.get(name)
        src = current_location(row)
        consumer = first_consumer.get(wid, 0)
        anchor = consumer if consumer <= args.prefetch_distance else max(0, consumer - args.prefetch_distance)
        pred_ms += steady_ms(choice, name, size, cm, bool(args.allow_cpu_fallback))

        if location == "disk":
            if state_any_resident(row):
                timeline.append({"kind": "evict", "weight_id": wid, "from_loc": src, "to_loc": "disk", "engine": "cpu", "anchor_op_id": 0, "overlap_group": -1})
            if be == "GPU":
                timeline.extend([
                    {"kind": "load", "weight_id": wid, "from_loc": "disk", "to_loc": "cpu", "engine": "disk", "anchor_op_id": anchor, "overlap_group": -1},
                    {"kind": "transfer", "weight_id": wid, "from_loc": "cpu", "to_loc": "gpu", "engine": "transfer", "anchor_op_id": anchor, "overlap_group": -1},
                    {"kind": "xform", "weight_id": wid, "from_loc": "gpu", "to_loc": "gpu", "engine": "gpu", "anchor_op_id": anchor, "overlap_group": -1},
                ])
            else:
                timeline.extend([
                    {"kind": "load", "weight_id": wid, "from_loc": "disk", "to_loc": "cpu", "engine": "disk", "anchor_op_id": anchor, "overlap_group": -1},
                    {"kind": "xform", "weight_id": wid, "from_loc": "cpu", "to_loc": "cpu", "engine": "cpu", "anchor_op_id": anchor, "overlap_group": -1},
                ])
        elif has_state and location == "gpu" and src != "gpu":
            if src == "disk":
                timeline.append({"kind": "load", "weight_id": wid, "from_loc": "disk", "to_loc": "cpu", "engine": "disk", "anchor_op_id": anchor, "overlap_group": -1})
            timeline.append({"kind": "transfer", "weight_id": wid, "from_loc": "cpu", "to_loc": "gpu", "engine": "transfer", "anchor_op_id": anchor, "overlap_group": -1})
            timeline.append({"kind": "xform", "weight_id": wid, "from_loc": "gpu", "to_loc": "gpu", "engine": "gpu", "anchor_op_id": anchor, "overlap_group": -1})
        elif has_state and location == "cpu" and src != "cpu":
            if src == "gpu":
                timeline.append({"kind": "evict", "weight_id": wid, "from_loc": "gpu", "to_loc": "disk", "engine": "cpu", "anchor_op_id": 0, "overlap_group": -1})
            timeline.append({"kind": "load", "weight_id": wid, "from_loc": "disk", "to_loc": "cpu", "engine": "disk", "anchor_op_id": anchor, "overlap_group": -1})
            timeline.append({"kind": "xform", "weight_id": wid, "from_loc": "cpu", "to_loc": "cpu", "engine": "cpu", "anchor_op_id": anchor, "overlap_group": -1})

    plan_ops = []
    for op in ops:
        wid = int(op.get("weight_id", -1))
        choice = placements.get(wid, "disk_gpu")
        be = placement_backend(choice)
        loc = placement_location(choice)
        w = by_id.get(wid, {})
        migrate = loc in ("cpu", "gpu") and loc != be.lower()
        plan_ops.append(
            {
                "op_id": int(op["op_id"]),
                "name": op["name"],
                "layer": int(op.get("layer", -1)),
                "compute_backend": be.lower(),
                "weight_id": wid,
                "dispatch": "static",
                "migrate": migrate,
                "migrate_from": loc if loc in ("cpu", "gpu") else "cpu",
                "migrate_xform": xform_for_backend(be, str(w.get("quant", ""))) if migrate else "none",
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
        "cost_model": {
            "backend_choice": "multi_backend_placement",
            "gpu_movement_cost": "reload_ensure" if cm.has_measured_stage_kind("OpenCL", "RELOAD_ENSURE") else "load_transfer_xform",
            "opencl_compute_measured": cm.has_measured_backend_compute("OpenCL"),
            "cpu_elastic_compute_measured": cm.has_measured_backend_compute("CPU_Elastic"),
            "opencl_stage_measured": cm.has_measured_stage_backend("OpenCL"),
            "cpu_elastic_stage_measured": cm.has_measured_stage_backend("CPU_Elastic"),
            "cpu_fallback_enabled": bool(args.allow_cpu_fallback),
            "transition_weight": objective_transition_weight,
        },
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
    ap.add_argument("--allow-cpu-fallback", action="store_true",
                    help="allow estimated CPU per-op compute when measured CPU_Elastic COMPUTE rows are unavailable")
    ap.add_argument("--transition-weight", type=float, default=1.0,
                    help="weight applied to current-state transition/migration cost in the online objective")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan = build_plan(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "budget_mib": args.budget_mib, "weights": len(plan["weights"]), "ops": len(plan["ops"]), "timeline": len(plan["timeline"]), "solver": plan["bottleneck"]}, indent=2))


if __name__ == "__main__":
    main()
