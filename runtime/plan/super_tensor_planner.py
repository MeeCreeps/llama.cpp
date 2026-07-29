#!/usr/bin/env python3
"""Mixed working-unit planner for Elastic llama.cpp.

The physical representation is a set of pre-provisioned row tiles.  Planning
changes only the partition over those tiles:

* CUT: the two tiles of one logical tensor are independent units;
* TENSOR: both tiles share one unit id;
* MULTI: tiles of two adjacent compatible tensors share one unit id.

This module is deliberately independent of placement solving.  It consumes an
already generated ExecPlan, predicts the three-stage pipeline makespan, and
attaches ``working_unit.units``.  Offline mode searches mixed partitions.
Online Diff-tree mode starts from the current partition and accepts a bounded
number of local split/merge edits only when their horizon gain pays for
transition cost and hysteresis. Full Online reoptimizes the complete frontier
under the same cost model and serves as the dynamic-planning performance bound.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any, Iterable, Sequence


MIB = 1024 * 1024
TENSOR_ID_STRIDE = 4
MULTI_ID_BASE = 1_000_000_000
_OFFLINE_CACHE_LIMIT = 64
_OFFLINE_CACHE: OrderedDict[
    tuple[Any, ...], tuple[tuple[Decision, ...], PipelineState]
] = OrderedDict()
_OFFLINE_CACHE_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class WeightSpec:
    weight_id: int
    name: str
    byte_size: int
    shape: tuple[int, ...]
    quant_name: str
    rows: int
    layer: int
    backend: str
    resident: bool
    plan_resident_ratio: float
    eligible_cut: bool


@dataclasses.dataclass(frozen=True)
class Decision:
    kind: str
    weight_ids: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class UnitCost:
    io_ms: float
    prepare_ms: float
    compute_ms: float

    def scaled(self, factor: float) -> "UnitCost":
        factor = max(0.05, factor)
        return UnitCost(
            self.io_ms * factor,
            self.prepare_ms * factor,
            self.compute_ms * factor,
        )

    def with_pipeline_efficiency(self, factor: float) -> "UnitCost":
        """Correct non-resident pipeline exposure without changing kernels.

        The residual curve is fitted after comparing the explicit pipeline
        critical-stage prediction with end-to-end latency. It therefore
        describes how non-resident preparation is exposed by unit boundaries;
        it is not a second measurement of resident GEMV throughput. Scaling
        compute here made a 90%-resident Multi unit look slower than Tensor
        even though the fixed-budget measurement shows the opposite, and
        caused the online planner to over-split high-residency frontiers.
        """
        factor = max(0.05, factor)
        return UnitCost(
            self.io_ms * factor,
            self.prepare_ms * factor,
            self.compute_ms,
        )


@dataclasses.dataclass
class PipelineState:
    io_end_ms: float = 0.0
    prepare_end_ms: float = 0.0
    compute_end_ms: float = 0.0
    exposed_wait_ms: float = 0.0
    units: int = 0

    def append(self, cost: UnitCost) -> None:
        self.io_end_ms += cost.io_ms
        prepare_start = max(self.prepare_end_ms, self.io_end_ms)
        self.prepare_end_ms = prepare_start + cost.prepare_ms
        compute_start = max(self.compute_end_ms, self.prepare_end_ms)
        self.exposed_wait_ms += max(0.0, self.prepare_end_ms - self.compute_end_ms)
        self.compute_end_ms = compute_start + cost.compute_ms
        self.units += 1


@dataclasses.dataclass
class SearchState:
    index: int
    decisions: list[Decision]
    pipeline: PipelineState
    cut_boundaries: int = 0
    merge_boundaries: int = 0


@dataclasses.dataclass(frozen=True)
class LocalEdit:
    operation: str
    index: int
    remove_count: int
    replacement: tuple[Decision, ...]
    changed_weights: tuple[int, ...]


def _float_map(value: Any, defaults: dict[str, float]) -> dict[str, float]:
    result = dict(defaults)
    if isinstance(value, dict):
        for key, item in value.items():
            try:
                result[str(key)] = float(item)
            except (TypeError, ValueError):
                pass
    return result


def _curve_map(value: Any) -> dict[str, tuple[tuple[float, float], ...]]:
    """Normalize per-mode piecewise-linear curves.

    A profile curve is a list of ``{"resident_ratio", "scale"}`` rows.  Pair
    lists are accepted as well so hand-authored profiles remain concise.
    Invalid points are ignored and duplicate ratios keep the last value.
    """
    if not isinstance(value, dict):
        return {}
    result: dict[str, tuple[tuple[float, float], ...]] = {}
    for mode, raw_points in value.items():
        if not isinstance(raw_points, list):
            continue
        points: dict[float, float] = {}
        for raw in raw_points:
            ratio: Any = None
            scale: Any = None
            if isinstance(raw, dict):
                ratio = raw.get("resident_ratio", raw.get("ratio"))
                scale = raw.get("scale")
            elif isinstance(raw, (list, tuple)) and len(raw) == 2:
                ratio, scale = raw
            try:
                normalized_ratio = min(1.0, max(0.0, float(ratio)))
                normalized_scale = max(0.05, float(scale))
            except (TypeError, ValueError):
                continue
            if math.isfinite(normalized_ratio) and math.isfinite(
                    normalized_scale):
                points[normalized_ratio] = normalized_scale
        if points:
            result[str(mode)] = tuple(sorted(points.items()))
    return result


def _interpolate_curve(
    points: Sequence[tuple[float, float]],
    ratio: float,
) -> float:
    if not points:
        return 1.0
    ratio = min(1.0, max(0.0, ratio))
    if ratio <= points[0][0]:
        return points[0][1]
    if ratio >= points[-1][0]:
        return points[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if left_x <= ratio <= right_x:
            if right_x <= left_x:
                return right_y
            fraction = (ratio - left_x) / (right_x - left_x)
            return left_y + fraction * (right_y - left_y)
    return points[-1][1]


class GranularityCostModel:
    """Calibrated pipeline cost model.

    Profiles may provide backend-specific entries under ``profiles.cpu`` and
    ``profiles.gpu``.  Per-weight records are optional and override the
    size-based base cost.  Mode scales describe *total useful work*, while
    request/launch costs are charged per actual working unit.
    """

    def __init__(self, data: dict[str, Any] | None, backend: str):
        data = data or {}
        profiles = data.get("profiles", data)
        selected = profiles.get(backend, profiles.get(backend.lower(), {}))
        if not isinstance(selected, dict):
            selected = {}
        self.backend = backend.lower()
        self.source = str(data.get("source", "uncalibrated"))
        self.multi_implementation = str(
            selected.get("multi_implementation", "multi"))
        self.multi_fusion_enabled = (
            self.multi_implementation == "multi_fused")
        self.io_bandwidth_mib_s = float(
            selected.get("io_bandwidth_mib_s", 2200.0))
        self.prepare_bandwidth_mib_s = float(
            selected.get("prepare_bandwidth_mib_s", 500.0))
        self.compute_ms_per_mib = float(
            selected.get("compute_ms_per_mib", 0.30))
        self.io_request_ms = float(selected.get("io_request_ms", 0.04))
        self.prepare_launch_ms = float(
            selected.get("prepare_launch_ms", 0.04))
        self.compute_launch_ms = float(
            selected.get("compute_launch_ms", 0.04))
        self.cut_compute_grouping = str(selected.get(
            "cut_compute_grouping",
            (
                "fused_pair"
                if self.backend == "gpu"
                else "independent_halves"
            ),
        ))
        if self.cut_compute_grouping not in {
            "independent_halves", "fused_pair",
        }:
            raise ValueError(
                "cut_compute_grouping must be independent_halves or "
                f"fused_pair, got {self.cut_compute_grouping!r}")
        self.mode_io_scale = _float_map(
            selected.get("mode_io_scale"),
            {"multi": 1.0, "tensor": 1.0, "cut": 1.0})
        self.mode_prepare_scale = _float_map(
            selected.get("mode_prepare_scale"),
            {"multi": 1.0, "tensor": 1.0, "cut": 1.0})
        self.mode_compute_scale = _float_map(
            selected.get("mode_compute_scale"),
            {"multi": 1.0, "tensor": 1.0, "cut": 1.0})
        # End-to-end fixed sweeps can expose a logical-unit bookkeeping or
        # head-of-line cost that is absent from backend compute counters. Keep
        # it explicit instead of distorting the measured per-MiB kernel rate.
        self.mode_unit_overhead_ms = _float_map(
            selected.get("mode_unit_overhead_ms"),
            {"multi": 0.0, "tensor": 0.0, "cut": 0.0})
        # Switching granularity inside one operator sequence can fragment
        # direct-I/O batches and shorten useful pipeline runs even though no
        # plan transition occurs.  Pure-mode calibration cannot expose this
        # term, so mixed-frontier calibration supplies it separately.  This
        # is deliberately distinct from transition_fixed_ms, which is paid
        # when an online plan replaces the previous frontier.
        self.cut_boundary_ms = max(
            0.0, float(selected.get(
                "cut_boundary_ms",
                selected.get("mode_boundary_ms", 0.0))))
        self.merge_boundary_ms = max(
            0.0, float(selected.get("merge_boundary_ms", 0.0)))
        self.mode_fusion_scale = _float_map(
            selected.get("mode_fusion_scale"),
            {"multi": 1.0, "tensor": 1.0, "cut": 1.0})
        # Rate scales model time per processed MiB. Work curves separately
        # model how much reload/layout work a granularity creates at a given
        # resident ratio (packing waste, partial-tile retention and turnover).
        # Keeping them separate prevents double counting Cut's low-budget
        # advantage.
        self.mode_io_work_curve = _curve_map(
            selected.get("mode_io_work_curve"))
        self.mode_prepare_work_curve = _curve_map(
            selected.get("mode_prepare_work_curve"))
        # Residual efficiency is calibrated from end-to-end fixed sweeps
        # after the explicit I/O/PREPARE/COMPUTE model has been evaluated.  It
        # captures how much of those stage costs is actually exposed on the
        # pipeline critical path (for example Cut's smaller wait bubbles).
        self.mode_pipeline_efficiency_curve = _curve_map(
            selected.get("mode_pipeline_efficiency_curve"))
        self.transition_fixed_ms = float(
            selected.get("transition_fixed_ms", 0.02))
        self.transition_ms_per_mib = float(
            selected.get("transition_ms_per_mib", 0.005))
        self.hysteresis_ms = float(selected.get("hysteresis_ms", 0.05))
        self.wait_feedback_weight = float(
            selected.get("wait_feedback_weight", 0.25))
        # A transition-aware prefix must beat the globally calibrated
        # steady-state target by more than the profile's relative uncertainty.
        # Otherwise a sub-millisecond modeled gain can select a materially
        # different frontier and overfit one calibration run.
        self.online_confidence_margin_ratio = max(
            0.0, float(selected.get(
                "online_confidence_margin_ratio", 0.01)))
        self.cut_row_alignment = max(
            1, int(selected.get("cut_row_alignment", 1)))
        self.max_merge_bytes = int(
            float(selected.get("max_merge_mib", 64.0)) * MIB)
        self.weights = selected.get("weights", {})
        if not isinstance(self.weights, dict):
            self.weights = {}

    @classmethod
    def load(cls, path: Path | None, backend: str) -> "GranularityCostModel":
        data = json.loads(path.read_text()) if path else {}
        return cls(data, backend)

    def _weight_override(self, weight: WeightSpec, key: str) -> float | None:
        row = self.weights.get(weight.name)
        if not isinstance(row, dict) or key not in row:
            return None
        try:
            return float(row[key])
        except (TypeError, ValueError):
            return None

    def signature(self) -> str:
        payload = {
            "backend": self.backend,
            "source": self.source,
            "multi_implementation": self.multi_implementation,
            "io_bandwidth_mib_s": self.io_bandwidth_mib_s,
            "prepare_bandwidth_mib_s": self.prepare_bandwidth_mib_s,
            "compute_ms_per_mib": self.compute_ms_per_mib,
            "io_request_ms": self.io_request_ms,
            "prepare_launch_ms": self.prepare_launch_ms,
            "compute_launch_ms": self.compute_launch_ms,
            "cut_compute_grouping": self.cut_compute_grouping,
            "mode_io_scale": self.mode_io_scale,
            "mode_prepare_scale": self.mode_prepare_scale,
            "mode_compute_scale": self.mode_compute_scale,
            "mode_unit_overhead_ms": self.mode_unit_overhead_ms,
            "cut_boundary_ms": self.cut_boundary_ms,
            "merge_boundary_ms": self.merge_boundary_ms,
            "mode_fusion_scale": self.mode_fusion_scale,
            "mode_io_work_curve": self.mode_io_work_curve,
            "mode_prepare_work_curve": self.mode_prepare_work_curve,
            "mode_pipeline_efficiency_curve":
                self.mode_pipeline_efficiency_curve,
            "online_confidence_margin_ratio":
                self.online_confidence_margin_ratio,
            "cut_row_alignment": self.cut_row_alignment,
            "max_merge_bytes": self.max_merge_bytes,
            "weights": self.weights,
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    def mode_work_scale(
        self,
        kind: str,
        phase: str,
        resident_ratio: float,
    ) -> float:
        curves = (
            self.mode_io_work_curve
            if phase == "io" else self.mode_prepare_work_curve
        )
        return _interpolate_curve(
            curves.get(kind, ()), resident_ratio)

    def mode_pipeline_efficiency(
        self,
        kind: str,
        resident_ratio: float,
    ) -> float:
        return _interpolate_curve(
            self.mode_pipeline_efficiency_curve.get(kind, ()),
            resident_ratio)

    def base_io_ms(self, weight: WeightSpec) -> float:
        override = self._weight_override(weight, "io_ms")
        if override is not None:
            return override
        return weight.byte_size / MIB / max(self.io_bandwidth_mib_s, 1e-9) * 1000.0

    def base_prepare_ms(self, weight: WeightSpec) -> float:
        override = self._weight_override(weight, "prepare_ms")
        if override is not None:
            return override
        return weight.byte_size / MIB / max(
            self.prepare_bandwidth_mib_s, 1e-9) * 1000.0

    def base_compute_ms(self, weight: WeightSpec) -> float:
        override = self._weight_override(weight, "compute_ms")
        if override is not None:
            return override
        return weight.byte_size / MIB * self.compute_ms_per_mib

    def decision_costs(
        self,
        decision: Decision,
        by_id: dict[int, WeightSpec],
    ) -> list[UnitCost]:
        weights = [by_id[wid] for wid in decision.weight_ids]
        kind = decision.kind
        resident_ratio = (
            weights[0].plan_resident_ratio if weights else 0.0)
        io_work_scale = self.mode_work_scale(
            kind, "io", resident_ratio)
        prepare_work_scale = self.mode_work_scale(
            kind, "prepare", resident_ratio)
        pipeline_efficiency = self.mode_pipeline_efficiency(
            kind, resident_ratio)
        if kind == "cut":
            weight = weights[0]
            total_io = (
                self.base_io_ms(weight) * self.mode_io_scale["cut"] *
                io_work_scale
                if not weight.resident else 0.0)
            total_prepare = (
                self.base_prepare_ms(weight) *
                self.mode_prepare_scale["cut"] * prepare_work_scale
                if not weight.resident else 0.0)
            total_compute = (
                self.base_compute_ms(weight) *
                self.mode_compute_scale["cut"])
            halves = [
                UnitCost(
                    total_io / 2.0 +
                    (self.io_request_ms if not weight.resident else 0.0),
                    total_prepare / 2.0 +
                    (self.prepare_launch_ms if not weight.resident else 0.0),
                    total_compute / 2.0 + self.compute_launch_ms +
                    self.mode_unit_overhead_ms["cut"],
                ).with_pipeline_efficiency(pipeline_efficiency)
                for _ in range(2)
            ]
            if self.cut_compute_grouping == "fused_pair":
                # The OpenCL fused dual-half path independently loads and
                # prepares both physical tiles, then submits one GEMV only
                # after both halves are ready.  Preserve the two pipeline
                # boundaries without inventing compute/prepare overlap inside
                # one tensor or charging two kernel launches.
                per_unit_overhead = self.mode_unit_overhead_ms["cut"]
                halves[0] = dataclasses.replace(
                    halves[0], compute_ms=per_unit_overhead)
                halves[1] = dataclasses.replace(
                    halves[1],
                    compute_ms=(
                        total_compute + self.compute_launch_ms +
                        per_unit_overhead),
                )
            return halves

        nonresident = [weight for weight in weights if not weight.resident]
        io_ms = sum(self.base_io_ms(weight) for weight in nonresident)
        prepare_ms = sum(
            self.base_prepare_ms(weight) for weight in nonresident)
        compute_ms = sum(self.base_compute_ms(weight) for weight in weights)
        io_ms *= self.mode_io_scale[kind] * io_work_scale
        prepare_ms *= (
            self.mode_prepare_scale[kind] * prepare_work_scale)
        compute_ms *= self.mode_compute_scale[kind]
        if kind == "multi" and compatible_fusion(
                weights[0], weights[1], self):
            compute_ms *= self.mode_fusion_scale["multi"]
        return [UnitCost(
            io_ms + (self.io_request_ms if nonresident else 0.0),
            prepare_ms + (
                self.prepare_launch_ms if nonresident else 0.0),
            compute_ms + self.compute_launch_ms +
            self.mode_unit_overhead_ms[kind],
        ).with_pipeline_efficiency(pipeline_efficiency)]


def normalize_weights(
    model_meta: dict[str, Any],
    plan: dict[str, Any],
    backend: str,
) -> list[WeightSpec]:
    placement = {
        int(row.get("weight_id", -1)): row
        for row in plan.get("weights", [])
        if isinstance(row, dict)
    }
    op_order = {
        int(row.get("weight_id", -1)): index
        for index, row in enumerate(model_meta.get("ops", []))
        if isinstance(row, dict)
    }
    resident_bytes = 0
    total_bytes = 0
    for ordinal, row in enumerate(model_meta.get("weights", [])):
        if not isinstance(row, dict):
            continue
        weight_id = int(row.get("weight_id", ordinal))
        byte_size = int(row.get(
            "byte_size",
            placement.get(weight_id, {}).get("byte_size", 0)) or 0)
        total_bytes += byte_size
        location = str(
            placement.get(weight_id, {}).get("location", "disk")).lower()
        if not location.startswith("disk"):
            resident_bytes += byte_size
    plan_resident_ratio = (
        resident_bytes / total_bytes if total_bytes > 0 else 0.0)
    result = []
    for ordinal, row in enumerate(model_meta.get("weights", [])):
        if not isinstance(row, dict):
            continue
        weight_id = int(row.get("weight_id", ordinal))
        shape = row.get("shape", [])
        normalized_shape = tuple(
            int(dim) for dim in shape
        ) if isinstance(shape, list) else ()
        rows = (
            normalized_shape[1]
            if len(normalized_shape) == 2 else 0
        )
        plan_row = placement.get(weight_id, {})
        location = str(plan_row.get("location", "disk")).lower()
        quant_name = str(
            row.get("quant_name", row.get("quant", ""))).upper()
        gpu_cut_compatible = (
            backend.lower() != "gpu" or
            quant_name == "Q4_0" or quant_name == "2"
        )
        cut_retention_compatible = (
            str(row.get("name", "")).removesuffix(".weight")
            != "token_embd"
        )
        result.append(WeightSpec(
            weight_id=weight_id,
            name=str(row.get("name", plan_row.get("name", ""))),
            byte_size=int(row.get(
                "byte_size", plan_row.get("byte_size", 0)) or 0),
            shape=normalized_shape,
            quant_name=quant_name,
            rows=rows,
            layer=int(row.get("layer", -1)),
            backend=backend.lower(),
            resident=not location.startswith("disk"),
            plan_resident_ratio=plan_resident_ratio,
            # Current CPU_Elastic/OpenCL row-tile paths accept contiguous
            # two-dimensional weights. Packed MoE expert tensors are 3-D and
            # must remain Tensor/Multi scheduling units.
            eligible_cut=(
                len(normalized_shape) == 2 and rows >= 2 and
                int(row.get("byte_size", 0) or 0) > 0 and
                gpu_cut_compatible and cut_retention_compatible
            ),
        ))
    result.sort(key=lambda weight: (
        op_order.get(weight.weight_id, 1 << 30), weight.weight_id))
    return result


def compatible_multi(
    first: WeightSpec,
    second: WeightSpec,
    model: GranularityCostModel,
) -> bool:
    if first.backend != second.backend:
        return False
    if first.byte_size + second.byte_size > model.max_merge_bytes:
        return False
    # Keep merge local in the operator graph. Cross-layer grouping creates
    # head-of-line blocking and makes diff patches unnecessarily large.
    return first.layer == second.layer


def compatible_fusion(
    first: WeightSpec,
    second: WeightSpec,
    model: GranularityCostModel,
) -> bool:
    """Whether the backend can legally fuse this pair's physical layout."""
    return (
        # The current fused pair path is implemented only by CPU_Elastic.
        # OpenCL still receives a real coarse Multi unit, but must not be
        # credited with or advertise fusion until its pair kernel exists.
        model.multi_fusion_enabled and
        first.backend == "cpu" and second.backend == "cpu" and
        first.eligible_cut and second.eligible_cut and
        len(first.shape) == 2 and first.shape == second.shape and
        first.byte_size == second.byte_size
    )


def simulate_partition(
    decisions: Sequence[Decision],
    weights: Sequence[WeightSpec],
    model: GranularityCostModel,
    wait_ratio: float = 0.0,
) -> PipelineState:
    by_id = {weight.weight_id: weight for weight in weights}
    state = PipelineState()
    for decision in decisions:
        for cost in model.decision_costs(decision, by_id):
            state.append(cost)
    if wait_ratio > 0.0:
        state.compute_end_ms += (
            min(1.0, max(0.0, wait_ratio)) *
            model.wait_feedback_weight * state.exposed_wait_ms)
    state.compute_end_ms += mode_boundary_cost_ms(decisions, model)
    return state


def mode_boundary_count(decisions: Sequence[Decision]) -> int:
    """Count adjacent changes of working-unit kind in graph order."""
    return sum(
        first.kind != second.kind
        for first, second in zip(decisions, decisions[1:])
    )


def mode_boundary_counts(
    decisions: Sequence[Decision],
) -> tuple[int, int]:
    """Return physical Cut/whole and logical Tensor/Multi boundaries.

    Cut changes the physical execution representation from one whole-tensor
    unit to independent row tiles. Tensor and Multi both retain whole-tensor
    tiles; their boundary only changes logical grouping. Keeping the two
    penalties separate avoids charging ordinary Tensor fallbacks inside a
    coarse Multi frontier as if they were Cut layout transitions.
    """
    cut = 0
    merge = 0
    for first, second in zip(decisions, decisions[1:]):
        if first.kind == second.kind:
            continue
        if (first.kind == "cut") != (second.kind == "cut"):
            cut += 1
        else:
            merge += 1
    return cut, merge


def mode_boundary_cost_ms(
    decisions: Sequence[Decision],
    model: GranularityCostModel,
) -> float:
    cut, merge = mode_boundary_counts(decisions)
    return (
        cut * model.cut_boundary_ms +
        merge * model.merge_boundary_ms
    )


def _search_state_score(
    state: SearchState,
    model: GranularityCostModel,
) -> float:
    # Beam states hold the physical pipeline state without the regularizer so
    # appending another unit preserves the exact three-stage recurrence.
    return (
        state.pipeline.compute_end_ms +
        state.cut_boundaries * model.cut_boundary_ms +
        state.merge_boundaries * model.merge_boundary_ms
    )


_NEG_INF = -1.0e300
_IDENTITY_MATRIX = (
    (0.0, _NEG_INF, _NEG_INF),
    (_NEG_INF, 0.0, _NEG_INF),
    (_NEG_INF, _NEG_INF, 0.0),
)


def _compose_pipeline_matrix(
    first: tuple[tuple[float, ...], ...],
    second: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Return ``second(first(state))`` in max-plus algebra."""
    return tuple(
        tuple(
            max(second[row][mid] + first[mid][column]
                for mid in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _unit_pipeline_matrix(
    cost: UnitCost,
) -> tuple[tuple[float, ...], ...]:
    io = cost.io_ms
    prepare = cost.prepare_ms
    compute = cost.compute_ms
    return (
        (io, _NEG_INF, _NEG_INF),
        (io + prepare, prepare, _NEG_INF),
        (
            io + prepare + compute,
            prepare + compute,
            compute,
        ),
    )


def _decision_pipeline_matrix(
    decision: Decision,
    by_id: dict[int, WeightSpec],
    model: GranularityCostModel,
) -> tuple[tuple[float, ...], ...]:
    result = _IDENTITY_MATRIX
    for cost in model.decision_costs(decision, by_id):
        result = _compose_pipeline_matrix(
            result, _unit_pipeline_matrix(cost))
    return result


def _pipeline_matrix_end(
    matrix: tuple[tuple[float, ...], ...],
) -> float:
    # PipelineState starts at I=P=C=0.
    return max(matrix[2])


def _partition_prefix_suffix(
    decisions: Sequence[Decision],
    by_id: dict[int, WeightSpec],
    model: GranularityCostModel,
) -> tuple[
    list[tuple[tuple[float, ...], ...]],
    list[tuple[tuple[float, ...], ...]],
]:
    matrices = [
        _decision_pipeline_matrix(decision, by_id, model)
        for decision in decisions
    ]
    prefix = [_IDENTITY_MATRIX]
    for matrix in matrices:
        prefix.append(_compose_pipeline_matrix(prefix[-1], matrix))
    suffix = [_IDENTITY_MATRIX] * (len(matrices) + 1)
    for index in range(len(matrices) - 1, -1, -1):
        suffix[index] = _compose_pipeline_matrix(
            matrices[index], suffix[index + 1])
    return prefix, suffix


def _edited_pipeline_end(
    edit: LocalEdit,
    prefix: Sequence[tuple[tuple[float, ...], ...]],
    suffix: Sequence[tuple[tuple[float, ...], ...]],
    by_id: dict[int, WeightSpec],
    model: GranularityCostModel,
) -> float:
    matrix = prefix[edit.index]
    for decision in edit.replacement:
        matrix = _compose_pipeline_matrix(
            matrix, _decision_pipeline_matrix(decision, by_id, model))
    matrix = _compose_pipeline_matrix(
        matrix, suffix[edit.index + edit.remove_count])
    return _pipeline_matrix_end(matrix)


def _clone_pipeline(state: PipelineState) -> PipelineState:
    return dataclasses.replace(state)


def offline_partition(
    weights: Sequence[WeightSpec],
    model: GranularityCostModel,
    beam_width: int = 128,
    refine_edits: int = 64,
) -> tuple[list[Decision], PipelineState]:
    """Beam-DP over Tensor, 2-way Cut, and adjacent 2-tensor Multi."""
    by_id = {weight.weight_id: weight for weight in weights}
    states: dict[int, list[SearchState]] = {
        0: [SearchState(0, [], PipelineState())]
    }
    for index in range(len(weights)):
        current_states = states.get(index, [])
        if not current_states:
            continue
        current_states.sort(key=lambda state: (
            _search_state_score(state, model),
            state.pipeline.exposed_wait_ms,
            len(state.decisions)))
        for state in current_states[:beam_width]:
            weight = weights[index]
            options: list[tuple[Decision, int]] = [
                (Decision("tensor", (weight.weight_id,)), 1)
            ]
            if weight.eligible_cut:
                options.append((Decision("cut", (weight.weight_id,)), 1))
            if (
                index + 1 < len(weights) and
                compatible_multi(weight, weights[index + 1], model)
            ):
                options.append((Decision(
                    "multi",
                    (weight.weight_id, weights[index + 1].weight_id)), 2))
            for decision, consumed in options:
                pipeline = _clone_pipeline(state.pipeline)
                for cost in model.decision_costs(decision, by_id):
                    pipeline.append(cost)
                target = index + consumed
                cut_boundaries = state.cut_boundaries
                merge_boundaries = state.merge_boundaries
                if (
                    state.decisions and
                    state.decisions[-1].kind != decision.kind
                ):
                    if (
                        (state.decisions[-1].kind == "cut") !=
                        (decision.kind == "cut")
                    ):
                        cut_boundaries += 1
                    else:
                        merge_boundaries += 1
                states.setdefault(target, []).append(SearchState(
                    target, state.decisions + [decision], pipeline,
                    cut_boundaries, merge_boundaries))
        # Bound every frontier, not only the state being expanded.
        for target in (index + 1, index + 2):
            if len(states.get(target, [])) > beam_width * 4:
                states[target].sort(key=lambda state: (
                    _search_state_score(state, model),
                    state.pipeline.exposed_wait_ms,
                    len(state.decisions)))
                states[target] = states[target][:beam_width]
    finals = states.get(len(weights), [])
    if not finals:
        decisions = [
            Decision("tensor", (weight.weight_id,)) for weight in weights]
        return decisions, simulate_partition(decisions, weights, model)
    best = min(finals, key=lambda state: (
        _search_state_score(state, model),
        state.pipeline.exposed_wait_ms,
        len(state.decisions)))
    return refine_partition(
        best.decisions, weights, model, max_edits=refine_edits)


def cached_offline_partition(
    weights: Sequence[WeightSpec],
    model: GranularityCostModel,
    *,
    beam_width: int,
    refine_edits: int,
) -> tuple[list[Decision], PipelineState, bool]:
    key = (
        model.signature(),
        int(beam_width),
        int(refine_edits),
        tuple((
            weight.weight_id,
            weight.name,
            weight.byte_size,
            weight.shape,
            weight.quant_name,
            weight.layer,
            weight.resident,
            weight.eligible_cut,
        ) for weight in weights),
    )
    with _OFFLINE_CACHE_LOCK:
        cached = _OFFLINE_CACHE.get(key)
        if cached is not None:
            _OFFLINE_CACHE.move_to_end(key)
            decisions, state = cached
            return list(decisions), _clone_pipeline(state), True
    decisions, state = offline_partition(
        weights, model, beam_width=beam_width,
        refine_edits=refine_edits)
    with _OFFLINE_CACHE_LOCK:
        _OFFLINE_CACHE[key] = (
            tuple(decisions), _clone_pipeline(state))
        _OFFLINE_CACHE.move_to_end(key)
        while len(_OFFLINE_CACHE) > _OFFLINE_CACHE_LIMIT:
            _OFFLINE_CACHE.popitem(last=False)
    return decisions, state, False


def fixed_partition(
    weights: Sequence[WeightSpec],
    model: GranularityCostModel,
    kind: str,
) -> tuple[list[Decision], PipelineState]:
    """Construct a homogeneous reference on the common physical tiles."""
    decisions: list[Decision] = []
    index = 0
    while index < len(weights):
        weight = weights[index]
        if kind == "cut" and weight.eligible_cut:
            decisions.append(Decision("cut", (weight.weight_id,)))
            index += 1
            continue
        if (
            kind == "multi" and
            index + 1 < len(weights) and
            compatible_multi(weight, weights[index + 1], model)
        ):
            decisions.append(Decision(
                "multi",
                (weight.weight_id, weights[index + 1].weight_id)))
            index += 2
            continue
        decisions.append(Decision("tensor", (weight.weight_id,)))
        index += 1
    return decisions, simulate_partition(decisions, weights, model)


def stable_unit_id(decision: Decision, part: int = 0) -> int:
    first = decision.weight_ids[0]
    if decision.kind == "cut":
        return first * TENSOR_ID_STRIDE + 1 + part
    if decision.kind == "multi":
        return MULTI_ID_BASE + first
    return first * TENSOR_ID_STRIDE


def split_rows(rows: int, alignment: int) -> list[tuple[int, int]]:
    if rows < 2:
        return [(0, -1)]
    alignment = max(1, alignment)
    first = (rows // 2 // alignment) * alignment
    if first <= 0 or first >= rows:
        first = rows // 2
    if first <= 0 or first >= rows:
        return [(0, -1)]
    return [(0, first), (first, rows - first)]


def tile_rows(
    weight: WeightSpec,
    model: GranularityCostModel,
) -> list[dict[str, Any]]:
    parts = split_rows(weight.rows, model.cut_row_alignment)
    if len(parts) == 1:
        return [{
            "weight_id": weight.weight_id,
            "weight_name": weight.name,
            "row_start": 0,
            "row_count": -1,
            "byte_offset": 0,
            "byte_size": weight.byte_size,
        }]
    result = []
    used_bytes = 0
    for index, (start, count) in enumerate(parts):
        if index + 1 == len(parts):
            byte_size = weight.byte_size - used_bytes
        else:
            byte_size = int(round(weight.byte_size * count / weight.rows))
        result.append({
            "weight_id": weight.weight_id,
            "weight_name": weight.name,
            "row_start": start,
            "row_count": count,
            "byte_offset": used_bytes,
            "byte_size": byte_size,
        })
        used_bytes += byte_size
    return result


def partition_to_working_unit(
    decisions: Sequence[Decision],
    weights: Sequence[WeightSpec],
    model: GranularityCostModel,
    *,
    policy: str,
    predicted_ms: float,
    switch_cost_ms: float = 0.0,
    state_aware: bool = False,
) -> dict[str, Any]:
    by_id = {weight.weight_id: weight for weight in weights}
    units = []
    for decision in decisions:
        if decision.kind == "cut":
            weight = by_id[decision.weight_ids[0]]
            for part, tile in enumerate(tile_rows(weight, model)):
                units.append({
                    "unit_id": stable_unit_id(decision, part),
                    "fuse_layout": False,
                    "fuse_compute": False,
                    "tiles": [tile],
                })
            continue
        tiles = []
        for weight_id in decision.weight_ids:
            tiles.extend(tile_rows(by_id[weight_id], model))
        fused = (
            decision.kind == "multi" and
            compatible_fusion(
                by_id[decision.weight_ids[0]],
                by_id[decision.weight_ids[1]],
                model)
        )
        units.append({
            "unit_id": stable_unit_id(decision),
            "fuse_layout": fused,
            "fuse_compute": fused,
            "tiles": tiles,
        })
    return {
        "enabled": True,
        "mode": "tensor",
        "cut_parts": 2,
        "multi_tensors": 2,
        "policy": policy,
        "state_aware": state_aware,
        "predicted_ms": predicted_ms,
        "switch_cost_ms": switch_cost_ms,
        "units": units,
    }


def working_unit_to_partition(
    working_unit: dict[str, Any] | None,
    weights: Sequence[WeightSpec],
) -> list[Decision]:
    if not working_unit or not isinstance(working_unit.get("units"), list):
        return [Decision("tensor", (weight.weight_id,)) for weight in weights]
    unit_to_weights: list[tuple[int, ...]] = []
    weight_to_units: dict[int, set[int]] = {}
    for unit in working_unit.get("units", []):
        if not isinstance(unit, dict):
            continue
        unit_id = int(unit.get("unit_id", -1))
        ordered = []
        seen = set()
        for tile in unit.get("tiles", []):
            if not isinstance(tile, dict):
                continue
            weight_id = int(tile.get("weight_id", -1))
            if weight_id < 0 or weight_id in seen:
                continue
            seen.add(weight_id)
            ordered.append(weight_id)
            weight_to_units.setdefault(weight_id, set()).add(unit_id)
        if ordered:
            unit_to_weights.append(tuple(ordered))
    multi_by_first = {
        ids[0]: ids for ids in unit_to_weights if len(ids) > 1
    }
    decisions = []
    consumed = set()
    for weight in weights:
        weight_id = weight.weight_id
        if weight_id in consumed:
            continue
        if weight_id in multi_by_first:
            ids = multi_by_first[weight_id]
            decisions.append(Decision("multi", ids))
            consumed.update(ids)
        elif len(weight_to_units.get(weight_id, set())) > 1:
            decisions.append(Decision("cut", (weight_id,)))
            consumed.add(weight_id)
        else:
            decisions.append(Decision("tensor", (weight_id,)))
            consumed.add(weight_id)
    return decisions


def partition_labels(
    decisions: Sequence[Decision],
) -> dict[int, tuple[str, tuple[int, ...]]]:
    result = {}
    for decision in decisions:
        label = (decision.kind, decision.weight_ids)
        for weight_id in decision.weight_ids:
            result[weight_id] = label
    return result


def partition_distance(
    first: Sequence[Decision],
    second: Sequence[Decision],
) -> int:
    a = partition_labels(first)
    b = partition_labels(second)
    return sum(a.get(weight_id) != b.get(weight_id)
               for weight_id in set(a) | set(b))


_TREE_DEPTH = {"cut": 0, "tensor": 1, "multi": 2}


def _tree_label_distance(
    current: tuple[str, tuple[int, ...]] | None,
    target: tuple[str, tuple[int, ...]] | None,
) -> int:
    if current is None or target is None:
        return 3
    current_kind, current_ids = current
    target_kind, target_ids = target
    distance = abs(
        _TREE_DEPTH.get(current_kind, 1) -
        _TREE_DEPTH.get(target_kind, 1))
    if (
        current_kind == target_kind == "multi"
        and current_ids != target_ids
    ):
        distance += 1
    return distance


def partition_tree_distance(
    first: Sequence[Decision],
    second: Sequence[Decision],
) -> int:
    """Number of IFF-tree edges between two working-unit frontiers.

    A coarse/fine transition is hierarchical:
    ``Cut <-> Tensor <-> Multi``.  A boolean label mismatch incorrectly gives
    Cut and Tensor the same distance from Multi and therefore rejects the
    necessary intermediate edit in a Cut-to-Multi transition.
    """
    current = partition_labels(first)
    target = partition_labels(second)
    return sum(
        _tree_label_distance(
            current.get(weight_id), target.get(weight_id))
        for weight_id in set(current) | set(target)
    )


def local_edits(
    decisions: Sequence[Decision],
    by_id: dict[int, WeightSpec],
    model: GranularityCostModel,
) -> Iterable[LocalEdit]:
    for index, decision in enumerate(decisions):
        if decision.kind == "tensor":
            weight = by_id[decision.weight_ids[0]]
            if weight.eligible_cut:
                yield LocalEdit(
                    "split", index, 1,
                    (Decision("cut", decision.weight_ids),),
                    decision.weight_ids)
            if index + 1 < len(decisions):
                following = decisions[index + 1]
                if (
                    following.kind == "tensor" and
                    compatible_multi(
                        weight, by_id[following.weight_ids[0]], model)
                ):
                    merged_ids = decision.weight_ids + following.weight_ids
                    yield LocalEdit(
                        "merge", index, 2,
                        (Decision("multi", merged_ids),),
                        merged_ids)
        elif decision.kind == "cut":
            yield LocalEdit(
                "merge", index, 1,
                (Decision("tensor", decision.weight_ids),),
                decision.weight_ids)
        elif decision.kind == "multi":
            replacement = [
                Decision("tensor", (weight_id,))
                for weight_id in decision.weight_ids
            ]
            yield LocalEdit(
                "split", index, 1, tuple(replacement),
                decision.weight_ids)


def apply_local_edit(
    decisions: Sequence[Decision],
    edit: LocalEdit,
) -> list[Decision]:
    return (
        list(decisions[:edit.index]) +
        list(edit.replacement) +
        list(decisions[edit.index + edit.remove_count:])
    )


def local_neighbors(
    decisions: Sequence[Decision],
    by_id: dict[int, WeightSpec],
    model: GranularityCostModel,
) -> Iterable[tuple[str, list[Decision], tuple[int, ...]]]:
    for edit in local_edits(decisions, by_id, model):
        yield (
            edit.operation,
            apply_local_edit(decisions, edit),
            edit.changed_weights,
        )


def refine_partition(
    decisions: Sequence[Decision],
    weights: Sequence[WeightSpec],
    model: GranularityCostModel,
    *,
    max_edits: int = 64,
) -> tuple[list[Decision], PipelineState]:
    """Deterministic local refinement of the beam-DP result.

    Pipeline completion state is continuous, so a finite beam can prune a
    prefix that later becomes useful.  Refinement guarantees the published
    Offline target is at least a local optimum under the same split/merge
    neighborhood used by Diff-tree.  This also prevents Online from appearing
    to beat Offline merely because of beam truncation.
    """
    by_id = {weight.weight_id: weight for weight in weights}
    chosen = list(decisions)
    chosen_state = simulate_partition(chosen, weights, model)
    for _ in range(max(0, max_edits)):
        prefix, suffix = _partition_prefix_suffix(
            chosen, by_id, model)
        current_end = (
            _pipeline_matrix_end(prefix[-1]) +
            mode_boundary_cost_ms(chosen, model))
        best: tuple[
            tuple[float, int],
            list[Decision],
        ] | None = None
        for edit in local_edits(chosen, by_id, model):
            candidate = apply_local_edit(chosen, edit)
            candidate_end = _edited_pipeline_end(
                edit, prefix, suffix, by_id, model)
            candidate_end += mode_boundary_cost_ms(candidate, model)
            improvement = current_end - candidate_end
            key = (
                improvement,
                -len(candidate),
            )
            if best is None or key > best[0]:
                best = (key, candidate)
        if best is None or best[0][0] <= 1e-9:
            break
        _, chosen = best
        # Replay only the winning edit to preserve the complete diagnostics;
        # max-plus prefix/suffix composition already scored every neighbor
        # exactly on the pipeline completion-time objective.
        chosen_state = simulate_partition(chosen, weights, model)
    return chosen, chosen_state


def transition_cost_ms(
    changed_weights: Iterable[int],
    by_id: dict[int, WeightSpec],
    model: GranularityCostModel,
) -> float:
    changed = set(changed_weights)
    if not changed:
        return 0.0
    bytes_changed = sum(by_id[weight_id].byte_size for weight_id in changed)
    return (
        model.transition_fixed_ms +
        bytes_changed / MIB * model.transition_ms_per_mib)


def partition_compute_service_ms(
    decisions: Sequence[Decision],
    weights: Sequence[WeightSpec],
    model: GranularityCostModel,
) -> tuple[float, int]:
    by_id = {weight.weight_id: weight for weight in weights}
    service_ms = 0.0
    physical_units = 0
    for decision in decisions:
        costs = model.decision_costs(decision, by_id)
        service_ms += sum(cost.compute_ms for cost in costs)
        physical_units += len(costs)
    return service_ms, physical_units


def telemetry_wait_ratio(
    telemetry: dict[str, Any] | None,
    *,
    estimated_compute_us: float = 0.0,
) -> float:
    if not telemetry:
        return 0.0
    try:
        wait_us = float(telemetry.get("pipeline_wait_us", 0.0))
        prepare_us = float(telemetry.get("prepare_us", 0.0))
        compute_us = float(
            telemetry.get("compute_us", estimated_compute_us)
            or estimated_compute_us)
    except (TypeError, ValueError):
        return 0.0
    denominator = max(1.0, wait_us + prepare_us + compute_us)
    return max(0.0, min(1.0, wait_us / denominator))


def diff_tree_partition(
    current: Sequence[Decision],
    offline_target: Sequence[Decision],
    weights: Sequence[WeightSpec],
    model: GranularityCostModel,
    *,
    telemetry: dict[str, Any] | None = None,
    horizon_tokens: float = 8.0,
    min_gain_ms: float = 0.0,
    max_edits: int = 4,
) -> tuple[list[Decision], PipelineState, dict[str, Any]]:
    """Bounded local IFF-tree reconfiguration.

    The offline target is guidance, not an unconditional replacement. Every
    edit is rescored against current residency and observed pipeline wait.
    """
    by_id = {weight.weight_id: weight for weight in weights}
    chosen = list(current)
    # A large budget transition can visit hundreds of IFF-tree edges.  The
    # model cost of an immutable Decision does not change during one callback,
    # so compute it once instead of rebuilding UnitCost objects while scoring
    # every candidate and replaying every accepted prefix.
    decision_cost_cache: dict[Decision, tuple[UnitCost, ...]] = {}

    def decision_costs_cached(
        decision: Decision,
    ) -> tuple[UnitCost, ...]:
        costs = decision_cost_cache.get(decision)
        if costs is None:
            costs = tuple(model.decision_costs(decision, by_id))
            decision_cost_cache[decision] = costs
        return costs

    compute_service_ms = 0.0
    physical_units = 0
    for decision in chosen:
        costs = decision_costs_cached(decision)
        compute_service_ms += sum(cost.compute_ms for cost in costs)
        physical_units += len(costs)
    telemetry_units = 0.0
    if telemetry:
        try:
            telemetry_units = float(telemetry.get("units", 0.0))
        except (TypeError, ValueError):
            telemetry_units = 0.0
    passes = max(
        1.0,
        telemetry_units / max(1, physical_units),
    )
    wait_ratio = telemetry_wait_ratio(
        telemetry,
        estimated_compute_us=compute_service_ms * 1000.0 * passes)
    # With a zero feedback weight telemetry is diagnostic only. Replaying the
    # entire partition for several candidates after every edit is then
    # mathematically identical to the exact max-plus prefix/suffix score, but
    # used to add hundreds of milliseconds to a budget-change callback.
    feedback_ratio = (
        wait_ratio if model.wait_feedback_weight > 0.0 else 0.0)

    def simulate_cached(
        decisions: Sequence[Decision],
    ) -> PipelineState:
        state = PipelineState()
        for decision in decisions:
            for cost in decision_costs_cached(decision):
                state.append(cost)
        if feedback_ratio > 0.0:
            state.compute_end_ms += (
                min(1.0, max(0.0, feedback_ratio)) *
                model.wait_feedback_weight * state.exposed_wait_ms)
        state.compute_end_ms += mode_boundary_cost_ms(decisions, model)
        return state

    chosen_state = simulate_cached(chosen)
    initial_state = _clone_pipeline(chosen_state)
    initial_distance = partition_tree_distance(
        chosen, offline_target)
    initial_labels = partition_labels(chosen)
    current_distance = initial_distance
    total_switch_ms = 0.0
    edits: list[dict[str, Any]] = []
    best_chosen = list(chosen)
    best_state = _clone_pipeline(chosen_state)
    best_switch_ms = 0.0
    best_edits: list[dict[str, Any]] = []
    best_net_gain_ms = 0.0
    best_distance = initial_distance
    target_labels = partition_labels(offline_target)
    horizon_tokens = max(1.0, horizon_tokens)

    def decision_service_ms(decision: Decision) -> float:
        return sum(
            cost.io_ms + cost.prepare_ms + cost.compute_ms
            for cost in decision_costs_cached(decision))

    for _ in range(max(0, max_edits)):
        current_labels = partition_labels(chosen)
        selected: tuple[tuple[float, ...], LocalEdit, int] | None = None
        for edit in local_edits(chosen, by_id, model):
            replacement_labels = partition_labels(edit.replacement)
            target_distance = current_distance
            for weight_id in edit.changed_weights:
                target_label = target_labels.get(weight_id)
                target_distance += _tree_label_distance(
                    replacement_labels.get(weight_id), target_label)
                target_distance -= _tree_label_distance(
                    current_labels.get(weight_id), target_label)
            if target_distance >= current_distance:
                continue
            removed_service = sum(
                decision_service_ms(decision)
                for decision in chosen[
                    edit.index:edit.index + edit.remove_count])
            replacement_service = sum(
                decision_service_ms(decision)
                for decision in edit.replacement)
            service_gain = removed_service - replacement_service
            key = (
                float(current_distance - target_distance),
                service_gain,
                edit.operation == "merge",
                float(-edit.index),
            )
            if selected is None or key > selected[0]:
                selected = (key, edit, target_distance)
        if selected is None:
            break

        _, best_edit, target_distance = selected
        previous_state = chosen_state
        chosen = apply_local_edit(chosen, best_edit)
        chosen_state = simulate_cached(chosen)
        chosen_labels = partition_labels(chosen)
        changed_from_initial = {
            weight_id
            for weight_id in set(initial_labels) | set(chosen_labels)
            if initial_labels.get(weight_id)
                != chosen_labels.get(weight_id)
        }
        # The complete prefix is published atomically at one graph boundary.
        # Charge the fixed transition once and each changed weight once,
        # rather than charging every IFF-tree intermediate (for example both
        # Cut->Tensor and Tensor->Multi for the same weight).
        prefix_switch_ms = transition_cost_ms(
            changed_from_initial, by_id, model)
        incremental_switch_ms = prefix_switch_ms - total_switch_ms
        total_switch_ms = prefix_switch_ms
        steady_gain = (
            previous_state.compute_end_ms -
            chosen_state.compute_end_ms)
        edits.append({
            "operation": best_edit.operation,
            "weight_ids": list(best_edit.changed_weights),
            "steady_gain_ms": steady_gain,
            "incremental_switch_cost_ms": incremental_switch_ms,
            "prefix_switch_cost_ms": prefix_switch_ms,
        })
        current_distance = target_distance

        # All edits in this prefix are installed atomically at a graph
        # boundary. Score the complete prefix rather than requiring the
        # Tensor intermediate of Cut<->Multi to win on its own.
        prefix_steady_gain_ms = (
            initial_state.compute_end_ms -
            chosen_state.compute_end_ms)
        prefix_net_gain_ms = (
            prefix_steady_gain_ms * horizon_tokens
            - total_switch_ms
            - model.hysteresis_ms
            - min_gain_ms)
        edits[-1]["prefix_horizon_net_gain_ms"] = prefix_net_gain_ms
        score = (
            prefix_net_gain_ms,
            -current_distance,
            prefix_steady_gain_ms,
        )
        best_score = (
            best_net_gain_ms,
            -best_distance,
            initial_state.compute_end_ms - best_state.compute_end_ms,
        )
        if prefix_net_gain_ms > 0.0 and score > best_score:
            best_chosen = list(chosen)
            best_state = _clone_pipeline(chosen_state)
            best_switch_ms = total_switch_ms
            best_edits = [dict(edit) for edit in edits]
            best_net_gain_ms = prefix_net_gain_ms
            best_distance = current_distance

    if len(best_edits) <= 16:
        published_edits = best_edits
        edits_truncated = False
    else:
        published_edits = best_edits[:8] + best_edits[-8:]
        edits_truncated = True
    return best_chosen, best_state, {
        "policy": "diff-tree-mixed",
        "wait_ratio": wait_ratio,
        "wait_feedback_active": feedback_ratio > 0.0,
        "horizon_tokens": horizon_tokens,
        "switch_cost_ms": best_switch_ms,
        "horizon_net_gain_ms": best_net_gain_ms,
        "edit_count": len(best_edits),
        "edits_truncated": edits_truncated,
        "offline_distance_before": initial_distance,
        "offline_distance_after": best_distance,
        "edits": published_edits,
    }


def online_partition(
    current: Sequence[Decision],
    target: Sequence[Decision],
    weights: Sequence[WeightSpec],
    model: GranularityCostModel,
    *,
    telemetry: dict[str, Any] | None = None,
    horizon_tokens: float = 8.0,
    min_gain_ms: float = 0.0,
) -> tuple[list[Decision], PipelineState, dict[str, Any]]:
    """Choose the best transition-aware frontier without an edit-count bound.

    The offline target is the globally searched steady-state frontier. Online
    evaluates every prefix of the complete hierarchical path from the current
    frontier to that target and installs the best prefix atomically. This is
    important because the best finite-horizon plan may stop before the offline
    target, and because a profitable Cut<->Multi change passes through a
    temporarily slower Tensor representation. Diff-tree uses the same
    objective but explores only ``max_edits`` steps.
    """
    by_id = {weight.weight_id: weight for weight in weights}
    target_distance = partition_tree_distance(current, target)
    compute_service_ms, physical_units = partition_compute_service_ms(
        current, weights, model)
    telemetry_units = 0.0
    if telemetry:
        try:
            telemetry_units = float(telemetry.get("units", 0.0))
        except (TypeError, ValueError):
            telemetry_units = 0.0
    passes = max(
        1.0, telemetry_units / max(1, physical_units))
    wait_ratio = telemetry_wait_ratio(
        telemetry,
        estimated_compute_us=compute_service_ms * 1000.0 * passes)
    feedback_ratio = (
        wait_ratio if model.wait_feedback_weight > 0.0 else 0.0)
    current_state = simulate_partition(
        current, weights, model, feedback_ratio)
    target_state = simulate_partition(
        target, weights, model, feedback_ratio)
    current_labels = partition_labels(current)
    target_labels = partition_labels(target)
    target_changed = {
        weight_id
        for weight_id in set(current_labels) | set(target_labels)
        if current_labels.get(weight_id) != target_labels.get(weight_id)
    }
    target_switch_ms = transition_cost_ms(
        target_changed, by_id, model)
    target_net_gain_ms = (
        (current_state.compute_end_ms - target_state.compute_end_ms) *
        max(1.0, horizon_tokens)
        - target_switch_ms
        - model.hysteresis_ms
        - min_gain_ms
    )
    confidence_margin_ms = (
        target_state.compute_end_ms *
        model.online_confidence_margin_ratio)

    # With a constant nonzero switch cost, zero per-byte transition cost, and
    # no telemetry correction, every nonempty prefix pays exactly the same
    # reconfiguration charge.  The refined global target therefore dominates
    # every intermediate prefix whenever it beats the current frontier.
    # Avoid replaying hundreds of hierarchical edits on each budget callback;
    # this is mathematically equivalent to the exhaustive path for the
    # calibrated OP12 profiles and removes planner latency from execution.
    if (
        model.transition_ms_per_mib == 0.0 and
        feedback_ratio == 0.0
    ):
        accepted = bool(target_changed) and target_net_gain_ms > 0.0
        chosen = list(target if accepted else current)
        chosen_state = target_state if accepted else current_state
        switch_ms = target_switch_ms if accepted else 0.0
        changed = target_changed if accepted else set()
        steady_gain_ms = (
            current_state.compute_end_ms -
            chosen_state.compute_end_ms)
        return (
            chosen,
            chosen_state,
            {
                "policy": "online-global-mixed",
                "wait_ratio": wait_ratio,
                "wait_feedback_active": False,
                "horizon_tokens": max(1.0, horizon_tokens),
                "switch_cost_ms": switch_ms,
                "candidate_switch_cost_ms": target_switch_ms,
                "target_horizon_net_gain_ms": target_net_gain_ms,
                "prefix_advantage_ms": (
                    target_state.compute_end_ms -
                    chosen_state.compute_end_ms),
                "confidence_margin_ms": confidence_margin_ms,
                "confidence_fallback_to_offline_target": False,
                "constant_switch_fast_path": True,
                "steady_gain_ms": steady_gain_ms,
                "horizon_net_gain_ms": (
                    target_net_gain_ms if accepted else 0.0),
                "changed_weights": len(changed),
                "target_changed_weights": len(target_changed),
                "optimizer_edit_count": (
                    target_distance if accepted else 0),
                "offline_distance_before": target_distance,
                "offline_distance_after": (
                    0 if accepted else target_distance),
                "accepted": accepted,
            },
        )

    chosen, chosen_state, search = diff_tree_partition(
        current,
        target,
        weights,
        model,
        telemetry=telemetry,
        horizon_tokens=horizon_tokens,
        min_gain_ms=min_gain_ms,
        # Every edit reduces the hierarchical target distance by at least one,
        # so this reaches the target without Diff-tree's practical edit cap.
        max_edits=target_distance,
    )

    wait_ratio = float(search["wait_ratio"])
    feedback_ratio = (
        wait_ratio if model.wait_feedback_weight > 0.0 else 0.0)
    prefix_advantage_ms = (
        target_state.compute_end_ms - chosen_state.compute_end_ms)
    confidence_fallback = (
        bool(target_changed) and target_net_gain_ms > 0.0 and
        prefix_advantage_ms < confidence_margin_ms)
    if confidence_fallback:
        chosen = list(target)
        chosen_state = target_state
        search = dict(search)
        search["switch_cost_ms"] = target_switch_ms
        search["horizon_net_gain_ms"] = target_net_gain_ms
        search["offline_distance_after"] = 0

    chosen_labels = partition_labels(chosen)
    changed = {
        weight_id
        for weight_id in set(current_labels) | set(chosen_labels)
        if current_labels.get(weight_id) != chosen_labels.get(weight_id)
    }
    steady_gain_ms = (
        current_state.compute_end_ms - chosen_state.compute_end_ms)
    switch_ms = float(search["switch_cost_ms"])
    accepted = bool(changed)
    return (
        chosen,
        chosen_state,
        {
            "policy": "online-global-mixed",
            "wait_ratio": wait_ratio,
            "wait_feedback_active": bool(
                search["wait_feedback_active"]),
            "horizon_tokens": max(1.0, horizon_tokens),
            "switch_cost_ms": switch_ms,
            "candidate_switch_cost_ms": transition_cost_ms(
                target_changed, by_id, model),
            "target_horizon_net_gain_ms": target_net_gain_ms,
            "prefix_advantage_ms": prefix_advantage_ms,
            "confidence_margin_ms": confidence_margin_ms,
            "confidence_fallback_to_offline_target":
                confidence_fallback,
            "steady_gain_ms": steady_gain_ms,
            "horizon_net_gain_ms": float(
                search["horizon_net_gain_ms"]),
            "changed_weights": len(changed),
            "target_changed_weights": len(target_changed),
            "optimizer_edit_count": int(search["edit_count"]),
            "offline_distance_before": int(
                search["offline_distance_before"]),
            "offline_distance_after": int(
                search["offline_distance_after"]),
            "accepted": accepted,
        },
    )


def attach_working_unit(
    model_meta: dict[str, Any],
    plan: dict[str, Any],
    model: GranularityCostModel,
    *,
    policy: str,
    current_working_unit: dict[str, Any] | None = None,
    offline_working_unit: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
    horizon_tokens: float = 8.0,
    min_gain_ms: float = 0.0,
    max_edits: int = 4,
    beam_width: int = 128,
) -> dict[str, Any]:
    weights = normalize_weights(model_meta, plan, model.backend)
    diagnostics: dict[str, Any] = {
        "cost_profile_source": model.source,
    }
    if policy in ("fixed-multi", "fixed-tensor", "fixed-cut"):
        decisions, state = fixed_partition(
            weights, model, policy.removeprefix("fixed-"))
        switch_cost = 0.0
        state_aware = False
    else:
        # Offline and full Online receive global beam search plus local
        # refinement. Diff-tree instead uses that frontier only as guidance
        # for a bounded number of incremental edits.
        bootstrap_mixed = (
            policy in ("diff-tree", "online") and
            current_working_unit is None
        )
        if offline_working_unit is not None:
            # The offline budget table was already globally searched and
            # refined. Reuse its partition as Diff-tree guidance or the full
            # Online starting frontier, then rescore it using current
            # placement/residency. Online performs unrestricted local
            # refinement below, whereas Diff-tree retains its bounded edit
            # budget. This removes multi-second beam/refinement work from the
            # latency-critical callback without weakening Online's frontier.
            offline = working_unit_to_partition(
                offline_working_unit, weights)
            if policy == "online":
                # The table frontier was already globally searched and locally
                # refined for this budget.  Re-refining it against every
                # transient placement both delayed the async callback and
                # overfit the frontier to a one-token residency snapshot.
                # Full Online remains unrestricted because online_partition()
                # may replace the complete current frontier with this target.
                offline_state = simulate_partition(
                    offline, weights, model)
                offline_target_source = "precomputed-budget-table"
            else:
                offline_state = simulate_partition(
                    offline, weights, model)
                offline_target_source = "precomputed-budget-table"
            offline_cache_hit = True
            offline_refined = True
        else:
            offline, offline_state, offline_cache_hit = (
                cached_offline_partition(
                    weights, model, beam_width=beam_width,
                    refine_edits=(
                        64
                        if policy in ("offline", "online")
                        or bootstrap_mixed
                        else 0)))
            offline_refined = (
                policy in ("offline", "online") or bootstrap_mixed)
            offline_target_source = "online-search"
        offline_cut_boundaries, offline_merge_boundaries = (
            mode_boundary_counts(offline))
        diagnostics.update({
            "offline_predicted_ms": offline_state.compute_end_ms,
            "offline_exposed_wait_ms": offline_state.exposed_wait_ms,
            "offline_mode_boundaries": mode_boundary_count(offline),
            "offline_cut_boundaries": offline_cut_boundaries,
            "offline_merge_boundaries": offline_merge_boundaries,
            "offline_mode_boundary_cost_ms":
                mode_boundary_cost_ms(offline, model),
            "offline_refined": offline_refined,
            "offline_cache_hit": offline_cache_hit,
            "offline_target_source": offline_target_source,
        })
    if policy == "diff-tree":
        if current_working_unit is None:
            # Pre-decode initialization has no live buffers to regroup and no
            # transition to amortize. Start at the refined offline partition;
            # the bounded Diff-tree edit budget applies only after inference
            # begins.
            decisions = offline
            state = offline_state
            switch_cost = 0.0
            diagnostics.update({
                "policy": "diff-tree-mixed",
                "bootstrap_offline": True,
                "switch_cost_ms": 0.0,
                "edit_count": 0,
                "edits_truncated": False,
                "edits": [],
            })
        else:
            current = working_unit_to_partition(
                current_working_unit, weights)
            decisions, state, diff = diff_tree_partition(
                current, offline, weights, model,
                telemetry=telemetry,
                horizon_tokens=horizon_tokens,
                min_gain_ms=min_gain_ms,
                max_edits=max_edits)
            diagnostics.update(diff)
            diagnostics["bootstrap_offline"] = False
            switch_cost = float(diff["switch_cost_ms"])
        state_aware = True
    elif policy == "online":
        if current_working_unit is None:
            decisions = offline
            state = offline_state
            switch_cost = 0.0
            diagnostics.update({
                "policy": "online-global-mixed",
                "bootstrap_offline": True,
                "switch_cost_ms": 0.0,
                "accepted": True,
            })
        else:
            current = working_unit_to_partition(
                current_working_unit, weights)
            decisions, state, online = online_partition(
                current, offline, weights, model,
                telemetry=telemetry,
                horizon_tokens=horizon_tokens,
                min_gain_ms=min_gain_ms)
            diagnostics.update(online)
            diagnostics["bootstrap_offline"] = False
            switch_cost = float(online["switch_cost_ms"])
        state_aware = True
    elif policy == "offline":
        decisions = offline
        state = offline_state
        switch_cost = 0.0
        state_aware = False
    cut_boundaries, merge_boundaries = mode_boundary_counts(decisions)
    diagnostics.update({
        "mode_boundaries": mode_boundary_count(decisions),
        "cut_boundaries": cut_boundaries,
        "merge_boundaries": merge_boundaries,
        "cut_boundary_ms": model.cut_boundary_ms,
        "merge_boundary_ms": model.merge_boundary_ms,
        "mode_boundary_cost_ms":
            mode_boundary_cost_ms(decisions, model),
    })
    plan["working_unit"] = partition_to_working_unit(
        decisions, weights, model,
        policy=(
            "diff-tree-mixed" if policy == "diff-tree"
            else "online-global-mixed" if policy == "online"
            else policy),
        predicted_ms=state.compute_end_ms,
        switch_cost_ms=switch_cost,
        state_aware=state_aware)
    plan.setdefault("cost_model", {})[
        "working_unit"] = diagnostics
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach a mixed Super-Tensor partition to an ExecPlan")
    parser.add_argument("--model-meta", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--policy",
        choices=(
            "fixed-multi", "fixed-tensor", "fixed-cut",
            "offline", "online", "diff-tree"),
        default="offline")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--current-plan", type=Path)
    parser.add_argument(
        "--offline-target-plan", type=Path,
        help="precomputed Offline-Mixed plan used as Diff-tree guidance")
    parser.add_argument("--horizon-tokens", type=float, default=8.0)
    parser.add_argument("--min-gain-ms", type=float, default=0.0)
    parser.add_argument("--max-edits", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta = json.loads(args.model_meta.read_text())
    plan = json.loads(args.plan.read_text())
    state = json.loads(args.state.read_text()) if args.state else {}
    current_plan = (
        json.loads(args.current_plan.read_text())
        if args.current_plan else {})
    current_working_unit = (
        current_plan.get("working_unit") or
        state.get("working_unit"))
    offline_working_unit = None
    if args.offline_target_plan:
        offline_target_plan = json.loads(
            args.offline_target_plan.read_text())
        candidate = offline_target_plan.get("working_unit")
        if isinstance(candidate, dict):
            offline_working_unit = candidate
    model = GranularityCostModel.load(args.profile, args.backend)
    attach_working_unit(
        meta, plan, model,
        policy=args.policy,
        current_working_unit=current_working_unit,
        offline_working_unit=offline_working_unit,
        telemetry=state.get("granularity"),
        horizon_tokens=args.horizon_tokens,
        min_gain_ms=args.min_gain_ms,
        max_edits=args.max_edits,
        beam_width=args.beam_width)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "policy": args.policy,
        "units": len(plan["working_unit"]["units"]),
        "predicted_ms": plan["working_unit"]["predicted_ms"],
        "switch_cost_ms": plan["working_unit"]["switch_cost_ms"],
    }, indent=2))


if __name__ == "__main__":
    main()
