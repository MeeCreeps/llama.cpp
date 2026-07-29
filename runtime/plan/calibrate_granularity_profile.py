#!/usr/bin/env python3
"""Fit a split/merge granularity cost profile from fixed-budget model runs.

The fixed sweep is calibration data. Dynamic nodes and real traces must remain
held out.  The fitted model keeps two effects separate:

* per-MiB/per-request cost for I/O, layout preparation, and computation;
* a resident-ratio curve for the amount of work and exposed pipeline cost.

This separation prevents a mode that reloads fewer bytes from receiving that
benefit twice in the Diff-tree objective.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Iterable


MODES = ("multi", "tensor", "cut")
MEASURED_MODES = ("multi", "multi_fused", "tensor", "cut")


def strip_pipeline_residual_profile(
    root: dict[str, object],
    backend: str,
    source: str = "",
) -> dict[str, object]:
    """Build Diff-before by removing only the selected residual curve."""
    result = copy.deepcopy(root)
    profiles = result.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"profile has no {backend} profile")
    selected = profiles.get(backend)
    if not isinstance(selected, dict):
        raise ValueError(f"profile has no {backend} profile")
    selected.pop("mode_pipeline_efficiency_curve", None)
    result["source"] = (
        source
        or f"{result.get('source', 'granularity-profile')}-phase-only"
    )
    calibrations = result.get("calibration")
    calibration = (
        calibrations.get(backend)
        if isinstance(calibrations, dict)
        else None
    )
    if isinstance(calibration, dict):
        residual = calibration.get("pipeline_efficiency_residual_curve")
        if isinstance(residual, dict):
            residual["enabled"] = False
            residual["baseline"] = "Diff-before"
    return result


def number(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fastest_mode_by_ratio(
    rows: list[dict[str, str]],
) -> dict[str, str]:
    """Return measured winners after Multi implementation selection."""
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        latency = number(row, "steady_median_latency_ms")
        if latency is None:
            continue
        key = (int(float(row["ratio_pct"])), row["mode"])
        grouped.setdefault(key, []).append(latency)
    ratios = sorted({ratio for ratio, _ in grouped})
    winners: dict[str, str] = {}
    for ratio in ratios:
        candidates = {
            mode: statistics.median(samples)
            for (candidate_ratio, mode), samples in grouped.items()
            if candidate_ratio == ratio
        }
        if candidates:
            winners[str(ratio)] = min(
                candidates,
                key=lambda mode: (candidates[mode], mode),
            )
    return winners


def valid_row(row: dict[str, str], backend: str) -> bool:
    return (
        row.get("backend", "").lower() == backend
        and row.get("mode", "") in MEASURED_MODES
        and row.get("returncode", "") == "0"
        and row.get("valid_runtime", "").lower() == "true"
        and row.get(
            "decode_wall_contract_valid", "").lower() == "true"
        and row.get(
            "decode_phase_contract_valid", "").lower() == "true"
        and row.get("decode_phase_counter_scope", "")
            == "decode-only-backend-counter-delta"
        and row.get("latency_timing_source", "") == "decode-wall"
        and row.get("valid_token_sequence", "").lower() == "true"
        and row.get("valid_device_idle", "").lower() == "true"
        and (number(row, "measurement_monitor_samples") or 0.0) > 0.0
        and (number(row, "measurement_monitor_failures") or 0.0) == 0.0
        and row.get(
            "monitor_thread_incomplete", "false").lower() != "true"
        and row.get("measurement_llama_pid", "").strip() != ""
        and row.get("budget_trace_sha256", "").strip() != ""
        and row.get("remote_budget_trace_sha256", "")
            == row.get("budget_trace_sha256")
        and row.get("valid_device_awake", "").lower() == "true"
        and (number(row, "steady_median_latency_ms") or 0.0) > 0.0
    )


def read_rows(paths: Iterable[Path], backend: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(
                row for row in csv.DictReader(handle)
                if valid_row(row, backend)
            )
    if not rows:
        raise ValueError(f"no valid {backend} calibration rows")
    return rows


def require_calibration_matrix(
    rows: list[dict[str, str]],
    *,
    ratios: Iterable[int],
    modes: Iterable[str],
) -> None:
    """Require at least one valid device row for every requested cell."""
    expected = {
        (mode.strip(), int(ratio))
        for mode in modes if mode.strip()
        for ratio in ratios
    }
    observed = {
        (row.get("mode", ""), int(float(row.get("ratio_pct", "-1"))))
        for row in rows
    }
    missing = sorted(expected - observed, key=lambda cell: (cell[1], cell[0]))
    if missing:
        detail = ", ".join(
            f"{mode}@{ratio}%" for mode, ratio in missing)
        raise ValueError(
            "incomplete valid fixed-budget calibration matrix: "
            f"{detail}")


def require_input_lineage(
    rows: list[dict[str, str]],
) -> tuple[str, str]:
    """Require every calibration cell to use one model and one binary."""
    binary_hashes = {
        row.get("remote_binary_sha256", "").strip()
        for row in rows
    }
    model_hashes = {
        row.get("remote_model_sha256", "").strip()
        for row in rows
    }
    config_hashes = {
        row.get("fixed_run_config_sha256", "").strip()
        for row in rows
    }
    if (
        "" in binary_hashes or len(binary_hashes) != 1
        or "" in model_hashes or len(model_hashes) != 1
        or "" in config_hashes or len(config_hashes) != 1
    ):
        raise ValueError(
            "fixed-budget calibration mixes or omits runtime input "
            f"lineage: binary={sorted(binary_hashes)} "
            f"model={sorted(model_hashes)} "
            f"config={sorted(config_hashes)}")
    return next(iter(binary_hashes)), next(iter(model_hashes))


def require_physical_tiling_contract(
    rows: list[dict[str, str]],
) -> dict[str, object]:
    """Require one physical representation behind every logical mode.

    The mixed planner can only interpret fixed-sweep phase costs when Multi,
    Tensor, and Cut were measured over the same pre-provisioned base tiles.
    Validate this again at the calibration consumer, rather than trusting a
    producer-side ``valid_runtime`` flag or a manually assembled CSV.
    """
    physical: set[tuple[int, int, float]] = set()
    targets: dict[int, set[float]] = {}
    for row in rows:
        tensors = number(row, "physical_cut_tensors")
        parts = number(row, "physical_cut_parts")
        total_mib = number(row, "wbm_total_mib")
        target_mib = number(row, "weight_target_mib")
        if (
            tensors is None or parts is None or total_mib is None
            or target_mib is None
        ):
            raise ValueError(
                "fixed-budget calibration omits physical tiling or "
                "weight-target evidence")
        tensor_count = int(tensors)
        part_count = int(parts)
        if (
            tensor_count <= 0
            or part_count != 2 * tensor_count
            or total_mib <= 0.0
            or target_mib <= 0.0
        ):
            raise ValueError(
                "invalid common physical representation: "
                f"tensors={tensor_count} parts={part_count} "
                f"wbm_total_mib={total_mib} "
                f"weight_target_mib={target_mib}")
        physical.add((
            tensor_count,
            part_count,
            round(total_mib, 6),
        ))
        ratio = int(float(row.get("ratio_pct", "-1")))
        targets.setdefault(ratio, set()).add(round(target_mib, 6))
    if len(physical) != 1:
        raise ValueError(
            "fixed-budget calibration mixes physical representations: "
            f"{sorted(physical)}")
    mismatched_targets = {
        ratio: sorted(values)
        for ratio, values in targets.items()
        if len(values) != 1
    }
    if mismatched_targets:
        raise ValueError(
            "logical modes used different physical targets at the same "
            f"budget ratio: {mismatched_targets}")
    tensors, parts, total_mib = next(iter(physical))
    return {
        "physical_cut_tensors": tensors,
        "physical_cut_parts": parts,
        "wbm_total_mib": total_mib,
        "weight_target_mib_by_ratio": {
            str(ratio): next(iter(values))
            for ratio, values in sorted(targets.items())
        },
    }


def select_multi_implementation(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str, dict[str, object]]:
    """Select one globally consistent coarse-unit implementation.

    Multi and Multi-fused have identical logical unit boundaries.  When both
    are measured, select the implementation with the lower paired geometric
    mean latency across the common fixed-budget ratios, then expose it to the
    three-way cost model as ``multi``.  Selection is global rather than
    budget-specific so the planner cannot gain an unmodelled fourth action.
    """
    by_implementation: dict[str, dict[int, list[float]]] = {
        "multi": {},
        "multi_fused": {},
    }
    for row in rows:
        implementation = row.get("mode", "")
        if implementation not in by_implementation:
            continue
        latency = number(row, "steady_median_latency_ms")
        ratio = number(row, "ratio_pct")
        if latency is None or latency <= 0.0 or ratio is None:
            continue
        by_implementation[implementation].setdefault(
            int(ratio), []).append(latency)

    ordinary = by_implementation["multi"]
    fused = by_implementation["multi_fused"]
    common_ratios = sorted(set(ordinary) & set(fused))
    selected = "multi"
    paired_fused_over_multi: list[float] = []
    if common_ratios:
        paired_fused_over_multi = [
            statistics.median(fused[ratio]) /
            statistics.median(ordinary[ratio])
            for ratio in common_ratios
        ]
        geometric_ratio = math.exp(statistics.mean(
            math.log(value) for value in paired_fused_over_multi))
        if geometric_ratio < 1.0:
            selected = "multi_fused"
    else:
        geometric_ratio = None
        if fused and not ordinary:
            selected = "multi_fused"

    normalized: list[dict[str, str]] = []
    for row in rows:
        mode = row.get("mode", "")
        if mode in {"multi", "multi_fused"} and mode != selected:
            continue
        copied = dict(row)
        if mode == selected:
            copied["mode"] = "multi"
            copied["measured_multi_implementation"] = selected
        normalized.append(copied)
    details: dict[str, object] = {
        "selected": selected,
        "common_ratios": common_ratios,
        "paired_fused_over_multi": paired_fused_over_multi,
        "geometric_fused_over_multi": geometric_ratio,
        "selection_scope": "one implementation across all budgets",
    }
    return normalized, selected, details


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system with partial-pivot Gaussian elimination."""
    n = len(vector)
    aug = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular calibration fit")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [
                    aug[row][i] - factor * aug[col][i]
                    for i in range(n + 1)
                ]
    return [aug[i][-1] for i in range(n)]


def least_squares(features: list[list[float]], target: list[float]) -> list[float]:
    width = len(features[0])
    gram = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for x, y in zip(features, target):
        for i in range(width):
            rhs[i] += x[i] * y
            for j in range(width):
                gram[i][j] += x[i] * x[j]
    # A tiny ridge only resolves exact degeneracy; it is far below timer noise.
    for i in range(width):
        gram[i][i] += 1e-12
    return solve_linear(gram, rhs)


def fit_phase(
    rows: list[dict[str, str]],
    *,
    time_key: str,
    work_key: str,
    calls_key: str,
    work_override: dict[int, float] | None = None,
) -> tuple[float, dict[str, float], int]:
    usable: list[tuple[str, float, float, float]] = []
    for index, row in enumerate(rows):
        elapsed = number(row, time_key)
        calls = number(row, calls_key)
        work = (
            work_override.get(index)
            if work_override is not None else number(row, work_key)
        )
        if elapsed is None or calls is None or work is None or work <= 0:
            continue
        usable.append((row["mode"], elapsed, calls, work))
    if not usable:
        raise ValueError(f"no samples for {time_key}/{work_key}")

    features: list[list[float]] = []
    target: list[float] = []
    for mode, elapsed, calls, work in usable:
        features.append([
            calls,
            work if mode == "multi" else 0.0,
            work if mode == "tensor" else 0.0,
            work if mode == "cut" else 0.0,
        ])
        target.append(elapsed)
    try:
        fitted = least_squares(features, target)
    except ValueError:
        fitted = [-1.0, 0.0, 0.0, 0.0]

    launch = fitted[0]
    rates = dict(zip(MODES, fitted[1:]))
    if launch < 0.0 or any(rate <= 0.0 for rate in rates.values()):
        # Non-negative fallback. A negative intercept is not a meaningful
        # launch cost and destabilizes small Cut units in the planner.
        launch = 0.0
        rates = {}
        for mode in MODES:
            samples = [
                elapsed / work
                for sample_mode, elapsed, _, work in usable
                if sample_mode == mode and work > 0
            ]
            if not samples:
                raise ValueError(f"missing {mode} samples for {time_key}")
            rates[mode] = statistics.median(samples)
    return launch, rates, len(usable)


def fit_gpu_compute_from_resident_tail(
    rows: list[dict[str, str]],
    *,
    planned_weight_mib: float = 0.0,
    unplanned_pinned_mib: float = 0.0,
) -> tuple[float, dict[str, float], int, dict[str, object]]:
    """Estimate undisturbed GPU compute from the highest-residency runs.

    OpenCL event profiling inserts ``clFinish`` and changes the pipeline being
    measured, so production GPU sweeps deliberately leave it disabled.  The
    CPU_Elastic fallback still emits a tiny ``compute_pt`` counter, but that is
    not GPU kernel time and must never train the GPU cost model.

    At the highest calibrated residency, subtract the runtime's exposed
    unit-pipeline wait from steady per-token latency.  The remainder is the
    backend compute/service floor under the real asynchronous execution path.
    It includes ordinary graph overhead, which is exactly what the planner
    must preserve when non-resident work approaches zero.
    """
    if planned_weight_mib <= 0.0:
        totals = [
            value for row in rows
            if (value := number(row, "wbm_total_mib")) is not None
            and value > 0.0
        ]
        if not totals:
            raise ValueError(
                "planned weight size is required for GPU compute fit")
        planned_weight_mib = statistics.median(totals)

    ratios = [
        resident_ratio(
            row,
            planned_weight_mib=planned_weight_mib,
            unplanned_pinned_mib=unplanned_pinned_mib,
        )
        for row in rows
    ]
    highest_ratio = max(ratios)
    estimates: dict[str, list[float]] = {mode: [] for mode in MODES}
    details: dict[str, list[dict[str, float]]] = {
        mode: [] for mode in MODES
    }
    for row, ratio in zip(rows, ratios):
        if not math.isclose(ratio, highest_ratio, abs_tol=1e-9):
            continue
        observed = number(row, "steady_median_latency_ms")
        wait_total = number(row, "decode_phase_pipeline_wait_ms")
        forward = number(row, "decode_phase_runs") or 0.0
        if (
            observed is None or wait_total is None or
            forward <= 1.0
        ):
            continue
        wait_per_forward = wait_total / forward
        compute_per_forward = observed - wait_per_forward
        if compute_per_forward <= 0.0:
            continue
        mode = row["mode"]
        estimates[mode].append(compute_per_forward)
        details[mode].append({
            "resident_ratio": ratio,
            "observed_ms_per_token": observed,
            "pipeline_wait_ms_per_forward": wait_per_forward,
            "compute_service_ms_per_forward": compute_per_forward,
        })
    if any(not estimates[mode] for mode in MODES):
        raise ValueError(
            "highest-residency GPU rows require steady latency and "
            "decode_phase_pipeline_wait_ms for every mode")
    per_forward = {
        mode: statistics.median(estimates[mode])
        for mode in MODES
    }
    rates = {
        mode: per_forward[mode] / planned_weight_mib
        for mode in MODES
    }
    return 0.0, rates, sum(map(len, estimates.values())), {
        "source": (
            "highest-residency steady latency minus exposed "
            "unit-pipeline wait; OpenCL event profiling disabled"
        ),
        "highest_resident_ratio": highest_ratio,
        "planned_weight_mib": planned_weight_mib,
        "mode_compute_service_ms_per_forward": per_forward,
        "samples": details,
    }


def grouped_median(
    rows: list[dict[str, str]],
    key: str,
    *,
    planned_weight_mib: float = 0.0,
    unplanned_pinned_mib: float = 0.0,
) -> dict[float, dict[str, float]]:
    grouped: dict[tuple[float, str], list[float]] = {}
    for row in rows:
        ratio = resident_ratio(
            row,
            planned_weight_mib=planned_weight_mib,
            unplanned_pinned_mib=unplanned_pinned_mib,
        )
        value = number(row, key)
        if value is not None:
            grouped.setdefault((ratio, row["mode"]), []).append(value)
    result: dict[float, dict[str, float]] = {}
    for (ratio, mode), values in grouped.items():
        result.setdefault(ratio, {})[mode] = statistics.median(values)
    return result


def normalized_curve(
    rows: list[dict[str, str]],
    key: str,
    *,
    planned_weight_mib: float = 0.0,
    unplanned_pinned_mib: float = 0.0,
) -> dict[str, list[dict[str, float]]]:
    grouped = grouped_median(
        rows,
        key,
        planned_weight_mib=planned_weight_mib,
        unplanned_pinned_mib=unplanned_pinned_mib,
    )
    curves = {mode: [] for mode in MODES}
    for ratio in sorted(grouped):
        baseline = grouped[ratio].get("tensor")
        if not baseline or any(mode not in grouped[ratio] for mode in MODES):
            continue
        for mode in MODES:
            curves[mode].append({
                "resident_ratio": ratio,
                "scale": grouped[ratio][mode] / baseline,
            })
    if any(not values for values in curves.values()):
        raise ValueError(f"incomplete tensor-normalized curve for {key}")
    return curves


def pipeline_curve(
    rows: list[dict[str, str]],
    *,
    prepare_time_key: str,
    planned_weight_mib: float = 0.0,
    unplanned_pinned_mib: float = 0.0,
    compute_per_forward_ms: dict[str, float] | None = None,
) -> tuple[
    dict[str, list[dict[str, float]]],
    dict[str, object],
]:
    groups: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for row in rows:
        forward = number(row, "decode_phase_runs") or 0.0
        observed = number(row, "steady_median_latency_ms")
        io_ms = number(row, "decode_phase_load_ms")
        prepare_ms = number(row, prepare_time_key)
        compute_ms = (
            compute_per_forward_ms.get(row["mode"], 0.0) * forward
            if compute_per_forward_ms is not None
            else number(row, "decode_phase_compute_ms")
        )
        if (
            forward <= 1.0 or observed is None or io_ms is None
            or prepare_ms is None or compute_ms is None
        ):
            continue
        # The planner already composes I/O -> preparation -> compute with a
        # max-plus pipeline.  Calibrating its residual against the additive
        # phase sum counts overlap twice and systematically over-rewards Cut,
        # whose finer units expose more overlap.  The aggregate critical
        # resource is the proper first-order prediction for a long pipeline;
        # the residual below then captures only fill/drain, dependencies, and
        # other non-ideal overlap that the phase model does not explain.
        critical = max(io_ms, prepare_ms, compute_ms) / forward
        ratio = resident_ratio(
            row,
            planned_weight_mib=planned_weight_mib,
            unplanned_pinned_mib=unplanned_pinned_mib,
        )
        groups.setdefault((ratio, row["mode"]), []).append(
            (observed, critical))

    curves = {mode: [] for mode in MODES}
    details: dict[str, object] = {}
    ratios = sorted({ratio for ratio, _ in groups})
    for ratio in ratios:
        summary: dict[str, tuple[float, float]] = {}
        for mode in MODES:
            values = groups.get((ratio, mode), [])
            if values:
                summary[mode] = (
                    statistics.median(value[0] for value in values),
                    statistics.median(value[1] for value in values),
                )
        if len(summary) != len(MODES):
            continue
        tensor_observed, tensor_critical = summary["tensor"]
        point: dict[str, object] = {}
        for mode in MODES:
            observed, critical = summary[mode]
            observed_scale = observed / tensor_observed
            phase_scale = critical / tensor_critical
            residual = observed_scale / phase_scale
            curves[mode].append({
                "resident_ratio": ratio,
                "scale": residual,
            })
            point[mode] = {
                "observed_scale": observed_scale,
                "critical_stage_scale": phase_scale,
                "residual_scale": residual,
            }
        details[str(ratio)] = point
    if any(not values for values in curves.values()):
        raise ValueError("incomplete pipeline residual curve")
    return curves, details


def resident_ratio(
    row: dict[str, str],
    *,
    planned_weight_mib: float = 0.0,
    unplanned_pinned_mib: float = 0.0,
) -> float:
    """Map a physical WBM target to the planner's weight-only ratio."""
    if planned_weight_mib > 0.0:
        target_mib = number(row, "weight_target_mib")
        if target_mib is None:
            raise ValueError(
                "weight_target_mib is required for planned-weight ratio "
                "calibration")
        return min(
            1.0,
            max(
                0.0,
                (target_mib - unplanned_pinned_mib)
                / planned_weight_mib,
            ),
        )
    return min(1.0, max(0.0, float(row["ratio_pct"]) / 100.0))


def write_cpu_placement_cost_dir(
    *,
    output_dir: Path,
    base_cost_dir: Path,
    source: str,
    io_launch_ms: float,
    io_ms_per_mib: float,
    prepare_launch_ms: float,
    prepare_ms_per_mib: float,
) -> None:
    """Create a CPU-only placement profile from fixed-sweep stage counters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("op_costs.json", "boundary_costs.json"):
        source_path = base_cost_dir / name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / name)

    stage_payload = {
        "device": "android-cpu-elastic",
        "model": source,
        "source_csv": [],
        "input_rows": 0,
        "records": [],
        "note": (
            "CPU LOAD/XFORM are supplied by calibration.json linear models "
            "fitted from fixed-budget Tensor-mode runtime counters."
        ),
    }
    (output_dir / "stage_costs.json").write_text(
        json.dumps(stage_payload, indent=2, sort_keys=True) + "\n")
    calibration = {
        "source": source,
        "stage_fallback_models": {
            "CPU_Elastic:LOAD": {
                "fixed_ms": max(0.0, io_launch_ms),
                "ms_per_mib": max(0.0, io_ms_per_mib),
            },
            "CPU_Elastic:XFORM": {
                "fixed_ms": max(0.0, prepare_launch_ms),
                "ms_per_mib": max(0.0, prepare_ms_per_mib),
            },
        },
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    summary = {
        "device": "android-cpu-elastic",
        "model": source,
        "base_cost_dir": str(base_cost_dir),
        "stage_records": 0,
        "stage_model_source": "fixed-budget Tensor-mode runtime counters",
        "cpu_load_fixed_ms": max(0.0, io_launch_ms),
        "cpu_load_ms_per_mib": max(0.0, io_ms_per_mib),
        "cpu_xform_fixed_ms": max(0.0, prepare_launch_ms),
        "cpu_xform_ms_per_mib": max(0.0, prepare_ms_per_mib),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")


def write_gpu_placement_cost_dir(
    *,
    output_dir: Path,
    base_cost_dir: Path,
    source: str,
    io_launch_ms: float,
    io_ms_per_mib: float,
    prepare_launch_ms: float,
    prepare_ms_per_mib: float,
) -> None:
    """Create an OP12 GPU placement profile from fixed-sweep stage counters.

    The production sweep intentionally avoids OpenCL event profiling. Its
    preparation counter is therefore host-visible issue/enqueue time, not pure
    device conversion-kernel time. Represent it once as OpenCL:XFORM and let
    the end-to-end pipeline residual capture exposed device work. TRANSFER and
    SYNC remain zero so the same host counter is never double counted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("op_costs.json", "boundary_costs.json"):
        source_path = base_cost_dir / name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / name)

    stage_payload = {
        "device": "android-opencl-elastic",
        "model": source,
        "source_csv": [],
        "input_rows": 0,
        "records": [],
        "note": (
            "OpenCL LOAD and combined layout preparation are supplied by "
            "calibration.json linear models fitted from fixed-budget "
            "Tensor-mode runtime counters."
        ),
    }
    (output_dir / "stage_costs.json").write_text(
        json.dumps(stage_payload, indent=2, sort_keys=True) + "\n")
    calibration = {
        "source": source,
        "stage_fallback_models": {
            "OpenCL:LOAD": {
                "fixed_ms": max(0.0, io_launch_ms),
                "ms_per_mib": max(0.0, io_ms_per_mib),
            },
            "OpenCL:TRANSFER": {
                "fixed_ms": 0.0,
                "ms_per_mib": 0.0,
            },
            "OpenCL:XFORM": {
                "fixed_ms": max(0.0, prepare_launch_ms),
                "ms_per_mib": max(0.0, prepare_ms_per_mib),
            },
            "OpenCL:SYNC": {
                "fixed_ms": 0.0,
                "ms_per_mib": 0.0,
            },
        },
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    summary = {
        "device": "android-opencl-elastic",
        "model": source,
        "base_cost_dir": str(base_cost_dir),
        "stage_records": 0,
        "stage_model_source": "fixed-budget Tensor-mode runtime counters",
        "gpu_load_fixed_ms": max(0.0, io_launch_ms),
        "gpu_load_ms_per_mib": max(0.0, io_ms_per_mib),
        "gpu_prepare_fixed_ms": max(0.0, prepare_launch_ms),
        "gpu_prepare_ms_per_mib": max(0.0, prepare_ms_per_mib),
        "gpu_prepare_semantics": (
            "host-visible preparation issue/enqueue; device event profiling "
            "disabled in authoritative production-overlap latency runs"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", action="append", type=Path, default=[])
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase-only-output",
        type=Path,
        help=(
            "also write the previous phase-only model, without the learned "
            "pipeline-efficiency residual; this is the Diff-before baseline"
        ),
    )
    parser.add_argument("--base-profile", type=Path)
    parser.add_argument("--source")
    parser.add_argument(
        "--placement-cost-base-dir",
        type=Path,
        help="existing cost directory supplying backend compute observations",
    )
    parser.add_argument(
        "--placement-cost-output-dir",
        type=Path,
        help=(
            "write a device-specific placement cost directory whose "
            "LOAD/preparation models are fitted from Tensor-mode "
            "fixed-sweep counters"
        ),
    )
    parser.add_argument(
        "--planned-weight-mib",
        type=float,
        default=0.0,
        help=(
            "planner-visible weight bytes; when set, curve ratios are derived "
            "from weight_target_mib after subtracting unplanned pinned bytes"
        ),
    )
    parser.add_argument(
        "--unplanned-pinned-mib",
        type=float,
        default=0.0,
        help="pinned WBM bytes omitted from the planner's model metadata",
    )
    parser.add_argument(
        "--required-ratios",
        default="",
        help=(
            "comma-separated fixed-budget ratios that must each contain all "
            "required measured modes"),
    )
    parser.add_argument(
        "--required-measured-modes",
        default="",
        help=(
            "comma-separated measured modes required at every ratio, e.g. "
            "multi,multi_fused,tensor,cut"),
    )
    parser.add_argument(
        "--cut-boundary-ms",
        "--mode-boundary-ms",
        dest="cut_boundary_ms",
        type=float,
        default=0.0,
        help=(
            "calibrated latency regularizer for each Cut/whole-tensor "
            "boundary inside one graph; obtain this from short "
            "mixed-frontier runs, not from the held-out dynamic trace"
        ),
    )
    parser.add_argument(
        "--merge-boundary-ms",
        type=float,
        default=0.0,
        help=(
            "optional regularizer for each Tensor/Multi grouping boundary; "
            "keep zero unless short mixed-frontier runs isolate this cost"
        ),
    )
    parser.add_argument(
        "--strip-residual-from",
        type=Path,
        help=(
            "derive an exact Diff-before profile from an existing profile by "
            "removing only the selected backend's pipeline residual"
        ),
    )
    args = parser.parse_args()
    if args.cut_boundary_ms < 0.0:
        parser.error("--cut-boundary-ms must be non-negative")
    if args.merge_boundary_ms < 0.0:
        parser.error("--merge-boundary-ms must be non-negative")

    if args.strip_residual_from:
        root = json.loads(args.strip_residual_from.read_text())
        root = strip_pipeline_residual_profile(
            root, args.backend, args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(root, indent=2, sort_keys=True) + "\n")
        return

    if not args.runs_csv:
        parser.error("--runs-csv is required for calibration")
    if not args.source:
        parser.error("--source is required for calibration")
    measured_rows = read_rows(args.runs_csv, args.backend)
    binary_sha256, model_sha256 = require_input_lineage(
        measured_rows)
    physical_tiling_audit = require_physical_tiling_contract(
        measured_rows)
    fixed_run_config_sha256 = next(iter({
        row["fixed_run_config_sha256"] for row in measured_rows
    }))
    if args.required_ratios or args.required_measured_modes:
        if not args.required_ratios or not args.required_measured_modes:
            parser.error(
                "--required-ratios and --required-measured-modes must be "
                "provided together")
        required_ratios = [
            int(value.strip())
            for value in args.required_ratios.split(",")
            if value.strip()
        ]
        required_modes = [
            value.strip()
            for value in args.required_measured_modes.split(",")
            if value.strip()
        ]
        unknown_modes = sorted(
            set(required_modes) - set(MEASURED_MODES))
        if unknown_modes:
            parser.error(
                "unknown --required-measured-modes: "
                + ", ".join(unknown_modes))
        require_calibration_matrix(
            measured_rows,
            ratios=required_ratios,
            modes=required_modes,
        )
    rows, multi_implementation, multi_selection = (
        select_multi_implementation(measured_rows))
    fixed_fastest_modes = fastest_mode_by_ratio(rows)
    prepare_keys = (
        "decode_phase_prepare_ms",
        "decode_phase_prepare_mib",
        "decode_phase_prepare_calls",
    )
    for key in prepare_keys:
        if not any(number(row, key) is not None for row in rows):
            raise ValueError(
                f"{key} is missing; rebuild the sweep CSV with device-event "
                "layout-preparation counters")

    io_launch, io_rates, io_samples = fit_phase(
        rows,
        time_key="decode_phase_load_ms",
        work_key="decode_phase_load_mib",
        calls_key="decode_phase_nonresident_units",
    )
    prepare_launch, prepare_rates, prepare_samples = fit_phase(
        rows,
        time_key=prepare_keys[0],
        work_key=prepare_keys[1],
        calls_key="decode_phase_nonresident_units",
    )
    gpu_compute_details: dict[str, object] | None = None
    gpu_compute_weight_mib = args.planned_weight_mib
    if args.backend == "gpu":
        (
            compute_launch,
            compute_rates,
            compute_samples,
            gpu_compute_details,
        ) = fit_gpu_compute_from_resident_tail(
            rows,
            planned_weight_mib=args.planned_weight_mib,
            unplanned_pinned_mib=args.unplanned_pinned_mib,
        )
        gpu_compute_weight_mib = float(
            gpu_compute_details["planned_weight_mib"])
    else:
        compute_work: dict[int, float] = {}
        for index, row in enumerate(rows):
            wbm_mib = number(row, "wbm_total_mib")
            forward = number(row, "decode_phase_runs") or 0.0
            if wbm_mib is not None and forward > 0.0:
                compute_work[index] = wbm_mib * forward
        compute_launch, compute_rates, compute_samples = fit_phase(
            rows,
            time_key="decode_phase_compute_ms",
            work_key="wbm_total_mib",
            calls_key="decode_phase_units",
            work_override=compute_work,
        )

    io_curve = normalized_curve(
        rows,
        "decode_phase_load_mib",
        planned_weight_mib=args.planned_weight_mib,
        unplanned_pinned_mib=args.unplanned_pinned_mib,
    )
    prepare_curve = normalized_curve(
        rows,
        prepare_keys[1],
        planned_weight_mib=args.planned_weight_mib,
        unplanned_pinned_mib=args.unplanned_pinned_mib,
    )
    residual_curve, residual_details = pipeline_curve(
        rows,
        prepare_time_key=prepare_keys[0],
        planned_weight_mib=args.planned_weight_mib,
        unplanned_pinned_mib=args.unplanned_pinned_mib,
        compute_per_forward_ms=(
            {
                mode: compute_rates[mode] * gpu_compute_weight_mib
                for mode in MODES
            }
            if args.backend == "gpu"
            else None
        ),
    )

    def rate_scales(rates: dict[str, float]) -> dict[str, float]:
        baseline = rates["tensor"]
        return {mode: rates[mode] / baseline for mode in MODES}

    profile = {
        "multi_implementation": multi_implementation,
        "pipeline_stage_order": ["load", "prepare", "compute"],
        "io_cost_source": "decode_phase_load",
        "pure_io_diagnostic_source": "decode_phase_direct_read",
        "stage_launch_count_source": {
            "load": "decode_phase_nonresident_units",
            "prepare": "decode_phase_nonresident_units",
            "compute": "decode_phase_units",
        },
        "io_bandwidth_mib_s": 1000.0 / io_rates["tensor"],
        "prepare_bandwidth_mib_s": 1000.0 / prepare_rates["tensor"],
        "compute_ms_per_mib": compute_rates["tensor"],
        "io_request_ms": io_launch,
        "prepare_launch_ms": prepare_launch,
        "compute_launch_ms": compute_launch,
        "cut_compute_grouping": (
            "fused_pair"
            if args.backend == "gpu"
            else "independent_halves"
        ),
        "mode_io_scale": rate_scales(io_rates),
        "mode_prepare_scale": rate_scales(prepare_rates),
        "mode_compute_scale": rate_scales(compute_rates),
        "mode_unit_overhead_ms": {mode: 0.0 for mode in MODES},
        "cut_boundary_ms": args.cut_boundary_ms,
        "merge_boundary_ms": args.merge_boundary_ms,
        "mode_fusion_scale": {mode: 1.0 for mode in MODES},
        "mode_io_work_curve": io_curve,
        "mode_prepare_work_curve": prepare_curve,
        "mode_pipeline_efficiency_curve": residual_curve,
        "transition_fixed_ms": 0.02,
        "transition_ms_per_mib": 0.0,
        "hysteresis_ms": 0.05,
        # The resident-ratio residual curve is already fitted from exposed
        # pipeline latency. Feeding the absolute runtime wait ratio back into
        # the same objective would count stalls twice and over-split the
        # frontier during budget recovery. Runtime history remains available
        # for diagnostics; future correction should use prediction error,
        # rather than the absolute wait ratio.
        "wait_feedback_weight": 0.0,
        "cut_row_alignment": 64,
        "max_merge_mib": 64.0,
    }
    root = (
        json.loads(args.base_profile.read_text())
        if args.base_profile else {}
    )
    root["schema_version"] = 1
    root["source"] = args.source
    root.setdefault("held_out", "dynamic nodes and full trace are not used")
    root.setdefault("profiles", {})[args.backend] = profile
    root.setdefault("calibration", {})[args.backend] = {
        "runs_csv": [str(path) for path in args.runs_csv],
        "valid_rows": len(rows),
        "measured_valid_rows": len(measured_rows),
        "remote_binary_sha256": binary_sha256,
        "remote_model_sha256": model_sha256,
        "fixed_run_config_sha256": fixed_run_config_sha256,
        "physical_tiling": physical_tiling_audit,
        "ratios": sorted({int(float(row["ratio_pct"])) for row in rows}),
        "multi_implementation_selection": multi_selection,
        "fixed_fastest_mode_by_ratio": fixed_fastest_modes,
        "pipeline_stage_order": ["load", "prepare", "compute"],
        "io_cost_source": "decode_phase_load",
        "pure_io_diagnostic_source": "decode_phase_direct_read",
        "stage_launch_count_source": {
            "load": "decode_phase_nonresident_units",
            "prepare": "decode_phase_nonresident_units",
            "compute": "decode_phase_units",
        },
        "cut_compute_grouping": (
            "fused_pair"
            if args.backend == "gpu"
            else "independent_halves"
        ),
        "planned_weight_mib": args.planned_weight_mib,
        "unplanned_pinned_mib": args.unplanned_pinned_mib,
        "cut_boundary_ms": args.cut_boundary_ms,
        "merge_boundary_ms": args.merge_boundary_ms,
        "mode_boundary_source": (
            "short mixed-frontier calibration"
            if (
                args.cut_boundary_ms > 0.0 or
                args.merge_boundary_ms > 0.0
            )
            else "disabled until mixed-frontier calibration"
        ),
        "io_fit": {
            "launch_ms": io_launch,
            "mode_rate_ms_per_mib": io_rates,
            "samples": io_samples,
        },
        "prepare_fit": {
            "launch_ms": prepare_launch,
            "mode_rate_ms_per_mib": prepare_rates,
            "samples": prepare_samples,
            "time_key": prepare_keys[0],
            "measurement_semantics": (
                "synchronous CPU layout-preparation wall time"
                if args.backend == "cpu"
                else (
                    "host-visible OpenCL preparation issue/enqueue time; "
                    "not pure device kernel time"
                )
            ),
        },
        "compute_fit": {
            "launch_ms": compute_launch,
            "mode_rate_ms_per_mib": compute_rates,
            "samples": compute_samples,
            "details": gpu_compute_details,
        },
        "pipeline_efficiency_residual_curve": {
            "enabled": True,
            "method": "observed_latency/pipeline_critical_stage",
            "points": residual_details,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n")
    if bool(args.placement_cost_base_dir) != bool(
        args.placement_cost_output_dir
    ):
        parser.error(
            "--placement-cost-base-dir and --placement-cost-output-dir "
            "must be provided together")
    if args.placement_cost_output_dir:
        writer = (
            write_cpu_placement_cost_dir
            if args.backend == "cpu"
            else write_gpu_placement_cost_dir
        )
        writer(
                output_dir=args.placement_cost_output_dir,
                base_cost_dir=args.placement_cost_base_dir,
                source=args.source,
                io_launch_ms=io_launch,
                io_ms_per_mib=io_rates["tensor"],
                prepare_launch_ms=prepare_launch,
                prepare_ms_per_mib=prepare_rates["tensor"],
            )
    if args.phase_only_output:
        phase_only = copy.deepcopy(root)
        phase_only["source"] = f"{args.source}-phase-only"
        phase_only["profiles"][args.backend].pop(
            "mode_pipeline_efficiency_curve", None)
        phase_only["calibration"][args.backend][
            "pipeline_efficiency_residual_curve"
        ]["enabled"] = False
        phase_only["calibration"][args.backend][
            "pipeline_efficiency_residual_curve"
        ]["baseline"] = "Diff-before"
        args.phase_only_output.parent.mkdir(parents=True, exist_ok=True)
        args.phase_only_output.write_text(
            json.dumps(phase_only, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
