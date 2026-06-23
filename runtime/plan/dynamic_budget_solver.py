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
ENGINES = ("compute_cpu", "compute_gpu", "disk", "prepare_cpu", "prepare_gpu", "transfer", "xform_cpu", "xform_gpu", "sync")
CP_OBJECTIVES = ("resource_makespan", "interval_makespan", "sum")


def parse_allowed_placements(s: str | None) -> set[str]:
    if not s:
        return set(PLACEMENTS)
    out = {p.strip() for p in s.split(",") if p.strip()}
    bad = out.difference(PLACEMENTS)
    if bad:
        raise ValueError(f"unknown placement(s): {sorted(bad)}")
    return out or set(PLACEMENTS)


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
        self.used_legacy_reload_ensure = False

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
            # Fine-grained profiling may not yet contain standalone transform
            # records. On OP12/OpenCL, GPU transform measured during staged
            # execution is much closer to about 0.8-2.0 GB/s than the old
            # optimistic 8 GB/s fallback, while the generic CPU repack path
            # currently falls back to a no-op for llama.cpp weights.
            if backend == "CPU_Elastic":
                return 0.02
            return 0.10 + mb / 1200.0 * 1000.0
        if kind == "SYNC":
            return 0.03
        if kind == "EVICT":
            return 0.01
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
        return self.path_sum_ms("GPU", "disk", "gpu", name, byte_size)

    def cpu_reload_ms(self, name: str, byte_size: int) -> float:
        return self.path_sum_ms("CPU", "disk", "cpu", name, byte_size)

    def cpu_to_gpu_ms(self, name: str, byte_size: int) -> float:
        return self.path_sum_ms("GPU", "cpu", "gpu", name, byte_size)

    def stage_path(self, backend: str, src: str, dst: str, name: str, byte_size: int) -> list[tuple[str, str, str, float]]:
        """Return fine-grained stages as (kind, backend, engine, ms).

        RELOAD_ENSURE deliberately is not part of the primary model: it is a
        legacy black box and hides disk, transfer, transform and synchronization.
        """
        if dst == "disk" or src == dst:
            return []
        if backend == "GPU":
            stages: list[tuple[str, str, str, float]] = []
            if src == "disk":
                stages.append(("LOAD", "OpenCL", "disk", self.stage_ms("OpenCL", "LOAD", name, byte_size)))
            stages.append(("TRANSFER", "OpenCL", "transfer", self.stage_ms("OpenCL", "TRANSFER", name, byte_size)))
            stages.append(("XFORM", "OpenCL", "gpu", self.stage_ms("OpenCL", "XFORM", name, byte_size)))
            stages.append(("SYNC", "OpenCL", "gpu", self.stage_ms("OpenCL", "SYNC", name, byte_size)))
            return stages
        if src == "gpu":
            # The runtime does not have GPU->CPU readback for model weights.
            # Represent this as evict + reload from the disk-backed source.
            return [
                ("EVICT", "OpenCL", "cpu", self.stage_ms("OpenCL", "EVICT", name, byte_size)),
                ("LOAD", "CPU_Elastic", "disk", self.stage_ms("CPU_Elastic", "LOAD", name, byte_size)),
                ("XFORM", "CPU_Elastic", "cpu", self.stage_ms("CPU_Elastic", "XFORM", name, byte_size)),
            ]
        return [
            ("LOAD", "CPU_Elastic", "disk", self.stage_ms("CPU_Elastic", "LOAD", name, byte_size)),
            ("XFORM", "CPU_Elastic", "cpu", self.stage_ms("CPU_Elastic", "XFORM", name, byte_size)),
        ]

    def legacy_reload_ensure_ms(self, backend: str, name: str, byte_size: int) -> float | None:
        measured = self.measured_stage_ms("OpenCL" if backend == "GPU" else "CPU_Elastic",
                                          "RELOAD_ENSURE", name, byte_size)
        if measured is not None:
            self.used_legacy_reload_ensure = True
        return measured

    def path_sum_ms(self, backend: str, src: str, dst: str, name: str, byte_size: int) -> float:
        return sum(stage[3] for stage in self.stage_path(backend, src, dst, name, byte_size))

    def path_pipeline_ms(self, backend: str, src: str, dst: str, name: str, byte_size: int,
                         compute_ms: float, overlap_model: str, stage_multiplier: float = 1.0) -> float:
        stages = self.stage_path(backend, src, dst, name, byte_size)
        if not stages:
            return compute_ms
        if overlap_model == "none":
            return compute_ms + stage_multiplier * sum(stage[3] for stage in stages)
        by_engine: dict[str, float] = {"compute": compute_ms}
        for _, _, engine, ms in stages:
            by_engine[engine] = by_engine.get(engine, 0.0) + stage_multiplier * ms
        return max(by_engine.values())

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
            return bool(flags & ((1 << 3) | (1 << 4)))
        return bool(flags & ((1 << 1) | (1 << 2)))
    if not isinstance(flags, list):
        return False
    if backend == "GPU":
        return ("gpu_compute_resident" in flags or "GPU_COMPUTE_RESIDENT" in flags
                or "gpu_raw_resident" in flags or "GPU_RAW_RESIDENT" in flags)
    return ("cpu_compute_resident" in flags or "CPU_COMPUTE_RESIDENT" in flags
            or "cpu_raw_resident" in flags or "CPU_RAW_RESIDENT" in flags)


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


def disk_stage_multiplier(choice: str, disk_reload_multiplier: float, disk_gpu_reload_multiplier: float) -> float:
    if choice == "disk_gpu":
        return disk_reload_multiplier * disk_gpu_reload_multiplier
    return disk_reload_multiplier


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


def steady_engine_ms(choice: str, name: str, size: int, cm: CostModel, allow_cpu_fallback: bool,
                     disk_reload_multiplier: float = 1.0,
                     disk_gpu_reload_multiplier: float = 4.0) -> dict[str, float]:
    backend = placement_backend(choice)
    compute = cm.compute_backend_ms(backend, name, size, allow_cpu_fallback)
    if math.isinf(compute):
        return {engine: math.inf for engine in ENGINES}
    out = {engine: 0.0 for engine in ENGINES}
    out["compute_gpu" if backend == "GPU" else "compute_cpu"] += compute
    reload_mult = disk_stage_multiplier(choice, disk_reload_multiplier, disk_gpu_reload_multiplier)
    if choice == "disk_gpu":
        for kind, _, engine, ms in cm.stage_path("GPU", "disk", "gpu", name, size):
            key = {
                "disk": "disk",
                "transfer": "transfer",
                "gpu": "xform_gpu" if kind == "XFORM" else "sync",
                "cpu": "xform_cpu",
            }.get(engine, engine)
            out[key] = out.get(key, 0.0) + reload_mult * ms
    elif choice == "disk_cpu":
        for kind, _, engine, ms in cm.stage_path("CPU", "disk", "cpu", name, size):
            key = "disk" if engine == "disk" else "xform_cpu"
            out[key] = out.get(key, 0.0) + reload_mult * ms
    return out


def engine_makespan_ms(engine_ms: dict[str, float]) -> float:
    return max(engine_ms.values()) if engine_ms else 0.0


def steady_ms(choice: str, name: str, size: int, cm: CostModel, allow_cpu_fallback: bool,
              disk_reload_multiplier: float = 1.0, disk_gpu_reload_multiplier: float = 4.0,
              overlap_model: str = "pipeline") -> float:
    engine_ms = steady_engine_ms(choice, name, size, cm, allow_cpu_fallback,
                                 disk_reload_multiplier, disk_gpu_reload_multiplier)
    if any(math.isinf(v) for v in engine_ms.values()):
        return math.inf
    if overlap_model == "none":
        return sum(engine_ms.values())
    return engine_makespan_ms(engine_ms)


def transition_engine_ms(choice: str, name: str, size: int, row: dict[str, Any] | None,
                         cm: CostModel) -> dict[str, float]:
    src = current_location(row)
    dst = placement_location(choice)
    backend = placement_backend(choice)
    out = {engine: 0.0 for engine in ENGINES}
    if src == dst or dst == "disk":
        return out
    for kind, _, engine, ms in cm.stage_path(backend, src, dst, name, size):
        key = {
            "disk": "disk",
            "transfer": "transfer",
            "gpu": "xform_gpu" if kind == "XFORM" else "sync",
            "cpu": "xform_cpu",
        }.get(engine, engine)
        out[key] = out.get(key, 0.0) + ms
    return out


def stage_engine_key(kind: str, engine: str, backend: str) -> str:
    if engine == "disk":
        return "disk"
    if engine == "transfer":
        return "transfer"
    if kind == "SYNC":
        return "sync"
    if engine == "gpu" or backend == "OpenCL":
        return "xform_gpu"
    return "xform_cpu"


def selected_choice_intervals(choice: str, name: str, size: int, cm: CostModel,
                              allow_cpu_fallback: bool,
                              disk_reload_multiplier: float,
                              disk_gpu_reload_multiplier: float) -> list[dict[str, Any]]:
    backend = placement_backend(choice)
    compute = cm.compute_backend_ms(backend, name, size, allow_cpu_fallback)
    if math.isinf(compute):
        return []
    out: list[dict[str, Any]] = []
    reload_mult = disk_stage_multiplier(choice, disk_reload_multiplier, disk_gpu_reload_multiplier)
    prepare_ms = 0.0
    prepare_engine = "prepare_gpu" if backend == "GPU" else "prepare_cpu"
    if choice == "disk_gpu":
        for kind, stage_backend, engine, ms in cm.stage_path("GPU", "disk", "gpu", name, size):
            if kind == "LOAD":
                out.append({
                    "kind": "load",
                    "engine": "disk",
                    "ms": reload_mult * ms,
                })
            else:
                # Treat host->device write + convert/transpose/materialization
                # as one backend prepare stage.  The lower OpenCL runtime may
                # internally split this into writes/kernels, but planner-visible
                # scheduling should reason about "prepare next weight" rather
                # than a standalone transfer stage.
                prepare_ms += reload_mult * ms
    elif choice == "disk_cpu":
        for kind, stage_backend, engine, ms in cm.stage_path("CPU", "disk", "cpu", name, size):
            if kind == "LOAD":
                out.append({
                    "kind": "load",
                    "engine": "disk",
                    "ms": reload_mult * ms,
                })
            else:
                prepare_ms += reload_mult * ms
    if prepare_ms > 0.0:
        out.append({
            "kind": "prepare",
            "engine": prepare_engine,
            "ms": prepare_ms,
        })
    out.append({
        "kind": "compute",
        "engine": "compute_gpu" if backend == "GPU" else "compute_cpu",
        "ms": compute,
    })
    return out


def build_index_pipeline_schedule(
    items: list[tuple[int, int, int, str, dict[str, float], dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]],
    placements: dict[int, str],
    prefetch_distance: int,
) -> tuple[list[dict[str, Any]], float]:
    """Build a deterministic op-index anchored schedule for the selected plan.

    CP-SAT interval placement can time out for a full 8B table within the tight
    online/offline limits.  Runtime execution is index based anyway, so this
    fallback preserves the important contract: every load/prepare stage
    is anchored before its consumer op, while start/end are diagnostic resource
    coordinates only.
    """
    resource_ready: dict[str, float] = {engine: 0.0 for engine in ENGINES}
    last_compute_end = 0.0
    events: list[dict[str, Any]] = []
    lead = max(0, int(prefetch_distance))
    stage_leads = {
        "load": lead * 2,
        "prepare": lead,
        "transfer": lead,
        "xform": lead,
        "sync": 0,
    }
    for wid, _, order, name, _, _, choice_ops in sorted(items, key=lambda item: item[2]):
        choice = placements.get(wid, "disk_gpu")
        ops = list(choice_ops.get(choice, []))
        if not ops:
            continue
        prev_end = 0.0
        for op in ops:
            kind = str(op.get("kind", "compute"))
            engine = str(op.get("engine", "compute_gpu"))
            duration_ms = max(0.0, float(op.get("ms", 0.0)))
            if kind == "compute":
                start_ms = max(prev_end, last_compute_end, resource_ready.get(engine, 0.0))
            else:
                start_ms = max(prev_end, resource_ready.get(engine, 0.0))
            end_ms = start_ms + duration_ms
            resource_ready[engine] = end_ms
            prev_end = end_ms
            if kind == "compute":
                last_compute_end = end_ms
                anchor_op_id = int(order)
            else:
                anchor_op_id = max(0, int(order) - int(stage_leads.get(kind, lead)))
            events.append({
                "weight_id": wid,
                "anchor_op_id": anchor_op_id,
                "consumer_op_id": int(order),
                "weight_name": name,
                "choice": choice,
                "kind": kind,
                "engine": engine,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
            })
    events.sort(key=lambda row: (
        int(row.get("anchor_op_id", 0)),
        float(row.get("start_ms", 0.0)),
        float(row.get("end_ms", 0.0)),
        int(row.get("weight_id", 0)),
        str(row.get("kind", "")),
    ))
    return events, max([float(e.get("end_ms", 0.0)) for e in events] or [0.0])


def placement_costs(weight: dict[str, Any], cm: CostModel, state: dict[str, dict[str, Any]],
                    allow_cpu_fallback: bool, transition_weight: float,
                    disk_reload_multiplier: float, disk_gpu_reload_multiplier: float,
                    overlap_model: str) -> dict[str, float]:
    name = str(weight["name"])
    size = int(weight.get("byte_size", 0))
    row = state.get(name)
    out: dict[str, float] = {}
    for choice in PLACEMENTS:
        steady = steady_ms(choice, name, size, cm, allow_cpu_fallback,
                           disk_reload_multiplier, disk_gpu_reload_multiplier, overlap_model)
        if math.isinf(steady):
            out[choice] = math.inf
            continue
        out[choice] = steady + transition_weight * transition_ms(choice, name, size, row, cm)
    return out


def placement_engine_costs(weight: dict[str, Any], cm: CostModel, state: dict[str, dict[str, Any]],
                           allow_cpu_fallback: bool, transition_weight: float,
                           disk_reload_multiplier: float,
                           disk_gpu_reload_multiplier: float) -> dict[str, dict[str, float]]:
    name = str(weight["name"])
    size = int(weight.get("byte_size", 0))
    row = state.get(name)
    out: dict[str, dict[str, float]] = {}
    for choice in PLACEMENTS:
        engines = steady_engine_ms(choice, name, size, cm, allow_cpu_fallback,
                                   disk_reload_multiplier, disk_gpu_reload_multiplier)
        if any(math.isinf(v) for v in engines.values()):
            out[choice] = {engine: math.inf for engine in ENGINES}
            continue
        if transition_weight > 0.0:
            trans = transition_engine_ms(choice, name, size, row, cm)
            for engine, ms in trans.items():
                engines[engine] = engines.get(engine, 0.0) + transition_weight * ms
        out[choice] = engines
    return out


def select_placements_cp(items: list[tuple[int, int, int, str, dict[str, float], dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]],
                         budget_bytes: int, time_limit_ms: int,
                         objective: str = "resource_makespan",
                         prefetch_distance: int = 1) -> dict[str, Any] | None:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        return None

    model = cp_model.CpModel()
    x: dict[tuple[int, str], Any] = {}
    for wid, _, _, _, costs, _, _ in items:
        feasible = [choice for choice in PLACEMENTS if not math.isinf(costs.get(choice, math.inf))]
        if not feasible:
            feasible = ["disk_gpu"]
        for choice in feasible:
            x[(wid, choice)] = model.NewBoolVar(f"{choice}_{wid}")
        model.Add(sum(x[(wid, choice)] for choice in feasible) == 1)
    model.Add(
        sum(size * x[(wid, choice)]
            for wid, size, _, _, costs, _, _ in items
            for choice in PLACEMENTS
            if (wid, choice) in x and placement_resident_bytes(choice, size) > 0) <= budget_bytes
    )
    total_cost_terms = [
        int(max(0.0, costs.get(choice, math.inf)) * 1000.0) * x[(wid, choice)]
        for wid, _, _, _, costs, _, _ in items
        for choice in PLACEMENTS
        if (wid, choice) in x
    ]
    if objective == "sum":
        model.Minimize(sum(total_cost_terms))
    else:
        engine_loads = []
        total_engine_work_terms = []
        total_compute_work_terms = []
        for engine in ENGINES:
            terms = []
            for wid, _, _, _, _, engine_costs, _ in items:
                for choice in PLACEMENTS:
                    if (wid, choice) not in x:
                        continue
                    ms = engine_costs.get(choice, {}).get(engine, math.inf)
                    if math.isinf(ms):
                        continue
                    terms.append(int(max(0.0, ms) * 1000.0) * x[(wid, choice)])
            load = model.NewIntVar(0, 10**12, f"load_{engine}")
            model.Add(load == sum(terms))
            engine_loads.append(load)
            total_engine_work_terms.extend(terms)
            if engine in ("compute_cpu", "compute_gpu"):
                total_compute_work_terms.extend(terms)
        if objective == "resource_makespan":
            makespan = model.NewIntVar(0, 10**12, "resource_makespan")
            for load in engine_loads:
                model.Add(makespan >= load)
            # Primary objective is overlapped resource makespan. Tie-break first
            # on compute work, then on total work. This avoids systematically
            # choosing disk_cpu over disk_gpu when disk load dominates both paths
            # but GPU compute is measured faster and transfer/xform can overlap.
            model.Minimize(makespan * 1000000 + sum(total_compute_work_terms) * 1000 + sum(total_engine_work_terms) + sum(total_cost_terms))
        else:
            resource_intervals: dict[str, list[Any]] = {engine: [] for engine in ENGINES}
            interval_meta: dict[tuple[int, str, int], dict[str, Any]] = {}
            compute_vars: dict[tuple[int, str], tuple[int, Any, Any]] = {}
            horizon = max(1, sum(
                int(max(1.0, op.get("ms", 0.0) * 1000.0))
                for _, _, _, _, _, _, choice_ops in items
                for ops in choice_ops.values()
                for op in ops
            ) + 1)
            for wid, _, order, name, _, _, choice_ops in items:
                for choice, ops in choice_ops.items():
                    if (wid, choice) not in x:
                        continue
                    prev_end = None
                    for idx, op in enumerate(ops):
                        dur = int(max(1.0, float(op.get("ms", 0.0)) * 1000.0))
                        start = model.NewIntVar(0, horizon, f"s_{wid}_{choice}_{idx}")
                        end = model.NewIntVar(0, horizon, f"e_{wid}_{choice}_{idx}")
                        interval = model.NewOptionalIntervalVar(
                            start, dur, end, x[(wid, choice)], f"iv_{wid}_{choice}_{idx}"
                        )
                        engine = str(op.get("engine", "compute_gpu"))
                        resource_intervals.setdefault(engine, []).append(interval)
                        if prev_end is not None:
                            model.Add(start >= prev_end).OnlyEnforceIf(x[(wid, choice)])
                        prev_end = end
                        interval_meta[(wid, choice, idx)] = {
                            "weight_id": wid,
                            "anchor_op_id": order,
                            "weight_name": name,
                            "choice": choice,
                            "kind": op.get("kind"),
                            "engine": engine,
                            "duration_ms": dur / 1000.0,
                            "start": start,
                            "end": end,
                        }
                        if op.get("kind") == "compute":
                            compute_vars[(wid, choice)] = (order, start, end)
            for intervals in resource_intervals.values():
                if intervals:
                    model.AddNoOverlap(intervals)
            ordered_items = sorted(items, key=lambda item: item[2])
            for prev_item, next_item in zip(ordered_items, ordered_items[1:]):
                prev_wid = prev_item[0]
                next_wid = next_item[0]
                for prev_choice in PLACEMENTS:
                    prev_key = (prev_wid, prev_choice)
                    if prev_key not in compute_vars or prev_key not in x:
                        continue
                    for next_choice in PLACEMENTS:
                        next_key = (next_wid, next_choice)
                        if next_key not in compute_vars or next_key not in x:
                            continue
                        _, _, prev_end = compute_vars[prev_key]
                        _, next_start, _ = compute_vars[next_key]
                        # Approximate llama graph dependency: op i+1 cannot
                        # start before op i finishes. This is conditional on
                        # the two placement choices actually being selected.
                        model.Add(next_start >= prev_end).OnlyEnforceIf([x[prev_key], x[next_key]])
            makespan = model.NewIntVar(0, horizon, "interval_makespan")
            for _, _, end in compute_vars.values():
                model.Add(makespan >= end)
            model.Minimize(makespan * 1000000 + sum(total_compute_work_terms) * 1000 + sum(total_engine_work_terms) + sum(total_cost_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(time_limit_ms, 1) / 1000.0
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"placements": {}, "schedule": [], "objective_ms": None, "status": solver.StatusName(status)}
    out: dict[int, str] = {}
    for wid, _, _, _, _, _, _ in items:
        for choice in PLACEMENTS:
            if (wid, choice) in x and solver.Value(x[(wid, choice)]) == 1:
                out[wid] = choice
                break
    schedule = []
    if objective == "interval_makespan":
        raw_schedule = []
        for (wid, choice, _), meta in interval_meta.items():
            if out.get(wid) != choice:
                continue
            raw_schedule.append({
                "weight_id": wid,
                "anchor_op_id": int(meta.get("anchor_op_id", wid)),
                "weight_name": meta["weight_name"],
                "choice": choice,
                "kind": meta["kind"],
                "engine": meta["engine"],
                "start_ms": solver.Value(meta["start"]) / 1000.0,
                "end_ms": solver.Value(meta["end"]) / 1000.0,
                "duration_ms": meta["duration_ms"],
            })
        # Runtime execution is index based.  Each non-compute stage is
        # anchored relative to the true consumer op of that weight, not to the
        # virtual interval start.  start_ms/end_ms are diagnostic CP-SAT
        # coordinates only; runtime never waits for virtual time.
        lead = max(0, int(prefetch_distance))
        stage_leads = {
            "load": lead * 2,
            "transfer": lead,
            "xform": lead,
            "sync": 0,
        }
        for row in raw_schedule:
            consumer = int(row.get("anchor_op_id", row.get("weight_id", 0)))
            if row.get("kind") == "compute":
                row["consumer_op_id"] = consumer
                continue
            row["consumer_op_id"] = consumer
            row["anchor_op_id"] = max(0, consumer - int(stage_leads.get(str(row.get("kind")), lead)))
        schedule = raw_schedule
        schedule.sort(key=lambda row: (row["anchor_op_id"], row["start_ms"], row["end_ms"], row["weight_id"], row["kind"]))
    objective_ms = solver.ObjectiveValue() / 1000000.0 if objective == "interval_makespan" else None
    return {"placements": out, "schedule": schedule, "objective_ms": objective_ms, "status": solver.StatusName(status)}


def select_placements_greedy(items: list[tuple[int, int, int, str, dict[str, float], dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]], budget_bytes: int) -> dict[int, str]:
    out: dict[int, str] = {}
    used = 0
    upgrades = []
    for wid, size, _, _, costs, _, _ in items:
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
        return "gpu_convert" if quant in ("Q4_0", "Q8_0", "MXFP4", "2", "8") else "none"
    return "cpu_repack" if quant else "none"


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    meta = load_json(args.model_meta)
    weights, ops = normalize_meta(meta)
    state = load_state(args.state)
    cm = CostModel(args.cost_dir)
    has_state = args.state is not None
    allowed_placements = parse_allowed_placements(getattr(args, "allowed_placements", None))

    budget_bytes = max(0, (args.budget_mib - args.kv_mib - args.misc_mib - args.safety_mib) * MB)
    objective_transition_weight = float(args.transition_weight) if has_state else 0.0
    by_id = {int(w["weight_id"]): w for w in weights}
    first_consumer: dict[int, int] = {}
    for op in ops:
        wid = int(op.get("weight_id", -1))
        first_consumer.setdefault(wid, int(op["op_id"]))
    items = []
    for w in weights:
        wid = int(w["weight_id"])
        size = int(w.get("byte_size", 0))
        name = str(w["name"])
        costs = placement_costs(
            w, cm, state,
            allow_cpu_fallback=bool(args.allow_cpu_fallback),
            transition_weight=objective_transition_weight,
            disk_reload_multiplier=float(args.disk_reload_multiplier),
            disk_gpu_reload_multiplier=float(getattr(args, "disk_gpu_reload_multiplier", 4.0)),
            overlap_model=str(args.overlap_model),
        )
        for choice in PLACEMENTS:
            if choice not in allowed_placements:
                costs[choice] = math.inf
            if name == "output.weight" and choice in ("cpu", "disk_cpu") and not bool(getattr(args, "allow_output_cpu", False)):
                costs[choice] = math.inf
            if name == "output.weight" and choice in ("disk_cpu", "disk_gpu") and not bool(getattr(args, "allow_output_disk", False)):
                costs[choice] = math.inf
        engine_costs = placement_engine_costs(
            w, cm, state,
            allow_cpu_fallback=bool(args.allow_cpu_fallback),
            transition_weight=objective_transition_weight,
            disk_reload_multiplier=float(args.disk_reload_multiplier),
            disk_gpu_reload_multiplier=float(getattr(args, "disk_gpu_reload_multiplier", 4.0)),
        )
        for choice in PLACEMENTS:
            if choice not in allowed_placements:
                engine_costs[choice] = {engine: math.inf for engine in ENGINES}
            if name == "output.weight" and choice in ("cpu", "disk_cpu") and not bool(getattr(args, "allow_output_cpu", False)):
                engine_costs[choice] = {engine: math.inf for engine in ENGINES}
            if name == "output.weight" and choice in ("disk_cpu", "disk_gpu") and not bool(getattr(args, "allow_output_disk", False)):
                engine_costs[choice] = {engine: math.inf for engine in ENGINES}
        choice_ops = {
            choice: selected_choice_intervals(
                choice, name, size, cm, bool(args.allow_cpu_fallback),
                float(args.disk_reload_multiplier), float(getattr(args, "disk_gpu_reload_multiplier", 4.0))
            )
            for choice in PLACEMENTS
            if choice in allowed_placements and not math.isinf(costs.get(choice, math.inf))
        }
        items.append((wid, size, first_consumer.get(wid, wid), name, costs, engine_costs, choice_ops))

    cp_result = select_placements_cp(
        items, budget_bytes, args.time_limit_ms, str(args.cp_objective), int(args.prefetch_distance)
    )
    solver_kind = "cp_sat"
    cp_schedule: list[dict[str, Any]] = []
    cp_objective_ms = None
    cp_status = None
    placements = cp_result.get("placements", {}) if cp_result else None
    if cp_result:
        cp_schedule = list(cp_result.get("schedule", []))
        cp_objective_ms = cp_result.get("objective_ms")
        cp_status = cp_result.get("status")
    if not placements:
        placements = select_placements_greedy(items, budget_bytes)
        solver_kind = "greedy"
    if str(args.cp_objective) == "interval_makespan" and not cp_schedule:
        cp_schedule, cp_objective_ms = build_index_pipeline_schedule(
            items, placements, int(args.prefetch_distance)
        )
        if cp_status and cp_status not in ("OPTIMAL", "FEASIBLE"):
            cp_status = f"{cp_status}+INDEX_PIPELINE_FALLBACK"
        else:
            cp_status = "INDEX_PIPELINE_FALLBACK"

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
        pred_ms += steady_ms(choice, name, size, cm, bool(args.allow_cpu_fallback),
                             float(args.disk_reload_multiplier),
                             float(args.disk_gpu_reload_multiplier),
                             str(args.overlap_model))

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
        "schedule": {
            "kind": ("index_pipeline" if cp_status and "INDEX_PIPELINE_FALLBACK" in str(cp_status)
                     else "interval_cp_sat") if str(args.cp_objective) == "interval_makespan" and cp_schedule else "none",
            "objective_ms": cp_objective_ms,
            "status": cp_status,
            "events": cp_schedule,
        },
        "pred_per_token_ms": pred_ms,
        "bottleneck": solver_kind,
        "cost_model": {
            "backend_choice": "multi_backend_placement",
            "movement_cost": "fine_stage_path",
            "legacy_reload_ensure_available": cm.has_measured_stage_kind("OpenCL", "RELOAD_ENSURE") or cm.has_measured_stage_kind("CPU_Elastic", "RELOAD_ENSURE"),
            "legacy_reload_ensure_used": cm.used_legacy_reload_ensure,
            "overlap_model": str(args.overlap_model),
            "cp_objective": str(args.cp_objective),
            "opencl_compute_measured": cm.has_measured_backend_compute("OpenCL"),
            "cpu_elastic_compute_measured": cm.has_measured_backend_compute("CPU_Elastic"),
            "opencl_stage_measured": cm.has_measured_stage_backend("OpenCL"),
            "cpu_elastic_stage_measured": cm.has_measured_stage_backend("CPU_Elastic"),
            "cpu_fallback_enabled": bool(args.allow_cpu_fallback),
            "allowed_placements": sorted(allowed_placements),
            "transition_weight": objective_transition_weight,
            "disk_reload_multiplier": float(args.disk_reload_multiplier),
            "disk_gpu_reload_multiplier": float(getattr(args, "disk_gpu_reload_multiplier", 4.0)),
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
    ap.add_argument("--allow-output-cpu", action="store_true",
                    help="allow output.weight to be placed on CPU/disk_cpu; disabled by default because OP12 end-to-end CPU output is much slower than the isolated op profile")
    ap.add_argument("--allow-output-disk", action="store_true",
                    help="allow output.weight to be evicted to disk; disabled by default because logits output reload dominates decode")
    ap.add_argument("--allowed-placements", default=",".join(PLACEMENTS),
                    help=f"comma-separated placement choices to allow; valid={','.join(PLACEMENTS)}")
    ap.add_argument("--transition-weight", type=float, default=0.1,
                    help="weight applied to current-state transition/migration cost in the online objective")
    ap.add_argument("--disk-reload-multiplier", type=float, default=1.0,
                    help="multiplier applied to steady-state disk/on-demand reload cost")
    ap.add_argument("--disk-gpu-reload-multiplier", type=float, default=4.0,
                    help="extra multiplier for disk_gpu on-demand disk/prepare stages")
    ap.add_argument("--overlap-model", choices=("pipeline", "none"), default="pipeline",
                    help="pipeline uses max per engine for disk/prepare/compute; none sums all stages")
    ap.add_argument("--cp-objective", choices=CP_OBJECTIVES, default="resource_makespan",
                    help="CP-SAT objective: interval_makespan uses optional intervals and NoOverlap resources")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan = build_plan(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "budget_mib": args.budget_mib, "weights": len(plan["weights"]), "ops": len(plan["ops"]), "timeline": len(plan["timeline"]), "solver": plan["bottleneck"]}, indent=2))


if __name__ == "__main__":
    main()
