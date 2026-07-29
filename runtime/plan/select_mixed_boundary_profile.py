#!/usr/bin/env python3
"""Select a boundary-cost profile from held-out fixed-node device runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from runtime.plan.super_tensor_planner import (
        GranularityCostModel,
        mode_boundary_counts,
        normalize_weights,
        working_unit_to_partition,
    )
except ModuleNotFoundError:
    from super_tensor_planner import (  # type: ignore
        GranularityCostModel,
        mode_boundary_counts,
        normalize_weights,
        working_unit_to_partition,
    )


def truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def finite_number(row: dict[str, str], key: str) -> float:
    value = float(row.get(key, "nan"))
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def valid_device_row(row: dict[str, str]) -> bool:
    """Validate one run while allowing a missing single-method peer flag."""
    return (
        row.get("status") == "ok"
        and str(row.get("rc", "0")) == "0"
        and truth(row.get("valid_token_trace", "false"))
        and truth(row.get(
            "token_trace_timing_contract_valid", "false"))
        and truth(row.get("decode_wall_contract_valid", "false"))
        and truth(row.get("decode_phase_contract_valid", "false"))
        and row.get("decode_phase_counter_scope", "")
            == "decode-only-backend-counter-delta"
        # A one-method calibration artifact has no local peer against which
        # the matrix runner can set valid_token_sequence=true. Reject an
        # explicit mismatch; all candidate prefix hashes are checked later.
        and str(
            row.get("valid_token_sequence", "")
        ).strip().lower() != "false"
        and truth(row.get("valid_measurement_isolation", "false"))
        and int(float(
            row.get("measurement_monitor_samples", "0") or 0)) > 0
        and int(float(
            row.get("measurement_monitor_failures", "0") or 0)) == 0
        and str(row.get("measurement_llama_pid", "")).strip() != ""
        and not truth(row.get("monitor_thread_incomplete", "false"))
        and int(float(
            row.get("pipeline_budget_violations", "1") or 1)) == 0
        and truth(row.get("timed_run_contract_valid", "false"))
        and truth(row.get("pin_contract_valid", "false"))
        and truth(row.get("fused_lane_contract_valid", "false"))
        and truth(row.get(
            "unit_pipeline_stage_authority_contract_valid", "false"))
        and truth(row.get(
            "backend_compute_contract_valid", "false"))
        and truth(row.get("frontier_contract_valid", "false"))
        and int(float(
            row.get("pin_outside_budget_count", "0") or 0)) == 0
        and str(row.get(
            "pipeline_plan_protection_relaxations", "")).strip() != ""
        and int(float(
            row.get(
                "pipeline_plan_protection_relaxations", "") or -1)) == 0
        and int(float(row.get("physical_cut_tensors", "0") or 0)) > 0
        and int(float(row.get("physical_cut_parts", "0") or 0))
            == 2 * int(float(
                row.get("physical_cut_tensors", "0") or 0))
        and float(row.get("wbm_total_mib", "0") or 0) > 0.0
        and float(row.get(
            "expected_wbm_total_mib_configured", "0") or 0) > 0.0
        and row.get("bench_exit_reason") == "duration"
        and int(float(row.get("bench_time_done", "0") or 0)) == 1
        and int(float(
            row.get("frontier_apply_count", "0") or 0)) > 0
        and float(
            row.get("frontier_publish_ms_max", "0") or 0) > 0.0
        and int(float(row.get(
            "frontier_transition_publish_count", "0") or 0)) > 0
        and float(row.get(
            "frontier_transition_publish_ms_max", "0") or 0) > 0.0
    )


def measured_frontier_transition_cost(
    row: dict[str, str],
) -> dict[str, float | int | str]:
    """Extract the isolated working-unit publication cost."""
    apply_count = int(float(
        row.get("frontier_apply_count", "0") or 0))
    publication_total_ms = finite_number(row, "frontier_publish_ms_total")
    publication_max_ms = finite_number(row, "frontier_publish_ms_max")
    transition_count = int(float(
        row.get("frontier_transition_publish_count", "0") or 0))
    transition_total_ms = finite_number(
        row, "frontier_transition_publish_ms_total")
    transition_max_ms = finite_number(
        row, "frontier_transition_publish_ms_max")
    if (
        apply_count <= 0
        or publication_total_ms <= 0.0
        or publication_max_ms <= 0.0
        or transition_count <= 0
        or transition_total_ms <= 0.0
        or transition_max_ms <= 0.0
    ):
        raise ValueError(
            "mixed-boundary row has no measured frontier transition "
            "publication cost")
    return {
        "frontier_apply_count": apply_count,
        "frontier_publish_ms_total": publication_total_ms,
        "frontier_publish_ms_mean":
            publication_total_ms / apply_count,
        "frontier_publish_ms_max": publication_max_ms,
        "frontier_transition_publish_count": transition_count,
        "frontier_transition_publish_ms_total": transition_total_ms,
        "frontier_transition_publish_ms_mean":
            transition_total_ms / transition_count,
        "frontier_transition_publish_ms_max": transition_max_ms,
        "selected_transition_fixed_ms": transition_max_ms,
        "source": (
            "max isolated changed-frontier publication time, excluding "
            "initial installation, from the selected held-in fixed-node "
            "candidate"
        ),
    }


def require_calibration_lineage(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Require every boundary candidate to share one runtime experiment."""
    if not rows:
        raise ValueError("mixed-boundary calibration has no device rows")
    for row in rows:
        if not valid_device_row(row):
            raise ValueError(
                "mixed-boundary calibration contains an invalid device row")
        for name in ("model_meta", "cost_dir"):
            local = row.get(f"{name}_sha256", "").strip()
            remote = row.get(f"remote_{name}_sha256", "").strip()
            if not local or remote != local:
                raise ValueError(
                    f"mixed-boundary calibration has invalid {name} "
                    "device lineage")
        source_hash = row.get("source_file_sha256", "").strip()
        if (
            not source_hash
            or row.get(
                "expected_source_trace_sha256", "").strip()
                != source_hash
        ):
            raise ValueError(
                "mixed-boundary calibration has invalid source trace "
                "lineage")
        runtime_hash = row.get("runtime_trace_sha256", "").strip()
        if (
            not runtime_hash
            or row.get(
                "remote_runtime_trace_sha256", "").strip()
                != runtime_hash
        ):
            raise ValueError(
                "mixed-boundary calibration has invalid runtime trace "
                "lineage")
        expected_total = float(
            row.get("expected_wbm_total_mib_configured", "0") or 0)
        observed_total = float(row.get("wbm_total_mib", "0") or 0)
        if abs(expected_total - observed_total) > max(
                0.01, expected_total * 1e-5):
            raise ValueError(
                "mixed-boundary calibration has invalid physical WBM "
                "total")

    common_keys = (
        "remote_binary_sha256",
        "remote_model_sha256",
        "model_meta_sha256",
        "cost_dir_sha256",
        "dynamic_run_config_sha256",
        "dynamic_run_core_config_sha256",
        "source_file_sha256",
        "source_window_sha256",
        "runtime_trace_sha256",
        "token_prefix_sha256",
    )
    common: dict[str, str] = {}
    for key in common_keys:
        values = {
            row.get(key, "").strip()
            for row in rows
        }
        if "" in values or len(values) != 1:
            raise ValueError(
                "mixed-boundary calibration does not share one "
                f"{key}: {sorted(values)}")
        common[key] = next(iter(values))
    physical = {
        (
            int(float(row["physical_cut_tensors"])),
            int(float(row["physical_cut_parts"])),
            round(float(row["wbm_total_mib"]), 6),
        )
        for row in rows
    }
    if len(physical) != 1:
        raise ValueError(
            "mixed-boundary calibration mixes physical representations: "
            f"{sorted(physical)}")
    tensors, parts, total_mib = next(iter(physical))
    return {
        "rows": len(rows),
        "common": common,
        "physical_cut_tensors": tensors,
        "physical_cut_parts": parts,
        "wbm_total_mib": total_mib,
    }


def token_budget_latency_breakdown(
    path: Path,
    expected_budgets: set[int] | None = None,
) -> dict[str, dict[str, float | int]]:
    groups: dict[int, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # Sequence zero contains prompt evaluation plus the first decode
            # token, so it is not a comparable steady decode sample.
            if int(row.get("sequence", "0")) == 0:
                continue
            budget = int(float(row["budget_mib"]))
            if expected_budgets is not None and budget not in expected_budgets:
                continue
            groups.setdefault(budget, []).append(float(row["latency_ms"]))
    result: dict[str, dict[str, float | int]] = {}
    for budget, samples in sorted(groups.items()):
        ordered = sorted(samples)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        result[str(budget)] = {
            "samples": len(samples),
            "mean_ms": sum(samples) / len(samples),
            "median_ms": statistics.median(samples),
            "p95_ms": ordered[p95_index],
        }
    return result


def row_node_latency(
    row: dict[str, str],
    expected_budgets: set[int] | None = None,
) -> dict[str, dict[str, float | int]]:
    path = Path(row.get("token_trace", ""))
    if not path.is_file():
        raise ValueError(f"missing calibration token trace: {path}")
    return token_budget_latency_breakdown(path, expected_budgets)


def node_means_text(
    breakdown: dict[str, dict[str, float | int]],
) -> str:
    return "/".join(
        f"{float(node['mean_ms']):.1f}"
        for node in breakdown.values()
    )


def speedup_pct(
    reference: dict[str, float | int],
    candidate: dict[str, float | int],
) -> float:
    reference_ms = float(reference["mean_ms"])
    candidate_ms = float(candidate["mean_ms"])
    if reference_ms <= 0.0:
        raise ValueError("reference node mean must be positive")
    return (reference_ms - candidate_ms) / reference_ms * 100.0


def mode_counts(
    plan_path: Path,
    model_meta: dict[str, Any],
    profile: GranularityCostModel,
) -> dict[str, int | float | str]:
    plan = json.loads(plan_path.read_text())
    weights = normalize_weights(model_meta, plan, profile.backend)
    flexible_weight_ids = {
        weight.weight_id for weight in weights if weight.eligible_cut
    }
    decisions = working_unit_to_partition(
        plan.get("working_unit"), weights)
    exact_frontier = [
        [decision.kind, list(decision.weight_ids)]
        for decision in decisions
    ]
    exact_frontier_bytes = json.dumps(
        exact_frontier, separators=(",", ":")).encode("utf-8")
    result: dict[str, int | float | str] = {
        mode: sum(decision.kind == mode for decision in decisions)
        for mode in ("multi", "tensor", "cut")
    }
    result["frontier_sha256"] = hashlib.sha256(
        exact_frontier_bytes).hexdigest()
    result.update({
        f"{mode}_flexible_weight_coverage": sum(
            weight_id in flexible_weight_ids
            for decision in decisions
            if decision.kind == mode
            for weight_id in decision.weight_ids
        )
        for mode in ("multi", "tensor", "cut")
    })
    cut_boundaries, merge_boundaries = mode_boundary_counts(decisions)
    run_lengths: list[int] = []
    last_kind: str | None = None
    for decision in decisions:
        if decision.kind != last_kind:
            run_lengths.append(1)
        else:
            run_lengths[-1] += 1
        last_kind = decision.kind
    result.update({
        "cut_boundaries": cut_boundaries,
        "merge_boundaries": merge_boundaries,
        "mode_boundaries": cut_boundaries + merge_boundaries,
        "mode_runs": len(run_lengths),
        "mean_run_units": (
            len(decisions) / len(run_lengths) if run_lengths else 0.0),
        "max_run_units": max(run_lengths, default=0),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--model-meta", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--low-budget-mib", type=int, required=True)
    parser.add_argument("--high-budget-mib", type=int, required=True)
    parser.add_argument("--low-fixed-ratio", type=int, required=True)
    parser.add_argument("--high-fixed-ratio", type=int, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    args = parser.parse_args()

    model_meta = json.loads(args.model_meta.read_text())
    candidates: list[dict[str, Any]] = []
    tensor_reference: dict[str, Any] | None = None
    expected_low_mode: str | None = None
    expected_high_mode: str | None = None
    calibration_rows: list[dict[str, str]] = []
    for directory in sorted(args.calibration_root.glob("cut_*")):
        results_path = directory / "summary" / "results.csv"
        profile_path = (
            args.calibration_root / "profiles" /
            f"{directory.name}.json"
        )
        if not results_path.exists() or not profile_path.exists():
            continue
        rows = list(csv.DictReader(results_path.open()))
        reference_for_directory: dict[str, str] | None = None
        reference_matches = [
            row for row in rows
            if row.get("method") == "offline"
        ]
        if reference_matches:
            if len(reference_matches) != 1 or tensor_reference is not None:
                raise ValueError(
                    "expected exactly one fixed-Tensor reference")
            reference_row = reference_matches[0]
            reference_for_directory = reference_row
            calibration_rows.append(reference_row)
            tensor_reference = {
                "directory": str(directory),
                "exec_ms_per_token": finite_number(
                    reference_row, "exec_ms_per_token"),
                "raw_ms_per_token": finite_number(
                    reference_row, "raw_ms_per_token"),
                "token_prefix_sha256": reference_row.get(
                    "token_prefix_sha256", ""),
                "valid": valid_device_row(reference_row),
            }
        matches = [
            row for row in rows
            if row.get("method") == "offline-mixed"
        ]
        if len(matches) != 1:
            continue
        row = matches[0]
        calibration_rows.append(row)
        profile_root = json.loads(profile_path.read_text())
        selected = profile_root["profiles"][args.backend]
        fixed_winners = (
            profile_root.get("calibration", {})
            .get(args.backend, {})
            .get("fixed_fastest_mode_by_ratio", {})
        )
        candidate_low_mode = fixed_winners.get(
            str(args.low_fixed_ratio))
        candidate_high_mode = fixed_winners.get(
            str(args.high_fixed_ratio))
        if candidate_low_mode not in {"multi", "tensor", "cut"}:
            raise ValueError(
                f"{profile_path}: missing measured fixed winner at "
                f"{args.low_fixed_ratio}%")
        if candidate_high_mode not in {"multi", "tensor", "cut"}:
            raise ValueError(
                f"{profile_path}: missing measured fixed winner at "
                f"{args.high_fixed_ratio}%")
        if expected_low_mode is None:
            expected_low_mode = candidate_low_mode
            expected_high_mode = candidate_high_mode
        elif (
            expected_low_mode != candidate_low_mode
            or expected_high_mode != candidate_high_mode
        ):
            raise ValueError(
                "boundary candidates do not share the same measured "
                "fixed-budget endpoint winners")
        cut_boundary_ms = float(
            selected.get("cut_boundary_ms", 0.0))
        merge_boundary_ms = float(
            selected.get("merge_boundary_ms", 0.0))
        valid = valid_device_row(row)
        profile = GranularityCostModel(profile_root, args.backend)
        mixed_dir = directory / "offline_mixed_table"
        available_budgets = {
            int(path.stem.removeprefix("plan_").removesuffix("MiB"))
            for path in mixed_dir.glob("plan_*MiB.json")
        }
        node_latency = row_node_latency(row, available_budgets)
        if reference_for_directory is not None:
            tensor_reference["node_latency"] = row_node_latency(
                reference_for_directory, available_budgets)
        node_mode_counts = {
            budget: mode_counts(
                mixed_dir / f"plan_{budget}MiB.json",
                model_meta, profile,
            )
            for budget in node_latency
        }
        low = node_mode_counts[str(args.low_budget_mib)]
        high = node_mode_counts[str(args.high_budget_mib)]
        assert expected_low_mode is not None
        assert expected_high_mode is not None
        crossover_valid = (
            low[f"{expected_low_mode}_flexible_weight_coverage"] >
                max(
                    low[f"{mode}_flexible_weight_coverage"]
                    for mode in ("multi", "tensor", "cut")
                    if mode != expected_low_mode
                )
            and high[f"{expected_high_mode}_flexible_weight_coverage"] >
                max(
                    high[f"{mode}_flexible_weight_coverage"]
                    for mode in ("multi", "tensor", "cut")
                    if mode != expected_high_mode
                )
        )
        candidates.append({
            "directory": str(directory),
            "profile": str(profile_path),
            "cut_boundary_ms": cut_boundary_ms,
            "merge_boundary_ms": merge_boundary_ms,
            "exec_ms_per_token": finite_number(
                row, "exec_ms_per_token"),
            "raw_ms_per_token": finite_number(
                row, "raw_ms_per_token"),
            "token_prefix_sha256": row.get(
                "token_prefix_sha256", ""),
            "valid": valid,
            "frontier_transition_calibration":
                measured_frontier_transition_cost(row),
            "crossover_valid": crossover_valid,
            "endpoint_winner_alignment_valid": crossover_valid,
            "node_latency": node_latency,
            "node_mode_counts": node_mode_counts,
            "low_mode_counts": low,
            "high_mode_counts": high,
        })

    calibration_lineage = require_calibration_lineage(
        calibration_rows)
    if expected_low_mode is None or expected_high_mode is None:
        raise ValueError("no boundary candidate exposes fixed-sweep winners")
    if expected_low_mode == expected_high_mode:
        raise ValueError(
            "fixed-budget measurements do not show an endpoint granularity "
            f"crossover: {args.low_fixed_ratio}% and "
            f"{args.high_fixed_ratio}% are both {expected_low_mode}")

    frontier_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        for budget, counts in candidate["node_mode_counts"].items():
            signature = str(counts["frontier_sha256"])
            frontier_groups.setdefault((budget, signature), []).append({
                "cut_boundary_ms": candidate["cut_boundary_ms"],
                "mean_ms": candidate["node_latency"][budget]["mean_ms"],
            })
    repeatability_groups = []
    for (budget, signature), samples in frontier_groups.items():
        if len(samples) < 2:
            continue
        values = [float(sample["mean_ms"]) for sample in samples]
        mean = sum(values) / len(values)
        repeatability_groups.append({
            "budget_mib": int(budget),
            "frontier_sha256": signature,
            "samples": samples,
            "relative_spread_pct": (
                (max(values) - min(values)) / mean * 100.0
                if mean > 0.0 else 0.0
            ),
        })
    repeatability_noise_pct = max(
        (
            group["relative_spread_pct"]
            for group in repeatability_groups
        ),
        default=0.0,
    )

    if tensor_reference is None or not tensor_reference["valid"]:
        raise ValueError("missing or invalid fixed-Tensor reference")
    tensor_nodes = tensor_reference.get("node_latency", {})
    low_key = str(args.low_budget_mib)
    high_key = str(args.high_budget_mib)
    if low_key not in tensor_nodes or high_key not in tensor_nodes:
        raise ValueError(
            "fixed-Tensor reference does not cover both calibration "
            "endpoints")
    for candidate in candidates:
        candidate_nodes = candidate["node_latency"]
        endpoint_coverage = (
            low_key in candidate_nodes and high_key in candidate_nodes)
        if endpoint_coverage:
            candidate["low_speedup_vs_fixed_tensor_pct"] = speedup_pct(
                tensor_nodes[low_key], candidate_nodes[low_key])
            candidate["high_speedup_vs_fixed_tensor_pct"] = speedup_pct(
                tensor_nodes[high_key], candidate_nodes[high_key])
        else:
            candidate["low_speedup_vs_fixed_tensor_pct"] = None
            candidate["high_speedup_vs_fixed_tensor_pct"] = None
        candidate["measured_endpoint_advantage"] = (
            endpoint_coverage
            and candidate["low_speedup_vs_fixed_tensor_pct"] > 0.0
            and candidate["high_speedup_vs_fixed_tensor_pct"] > 0.0
        )
    eligible = [
        candidate for candidate in candidates
        if (
            candidate["valid"]
            and candidate["endpoint_winner_alignment_valid"]
            and candidate["measured_endpoint_advantage"]
        )
    ]
    token_hashes = {
        candidate["token_prefix_sha256"]
        for candidate in eligible
        if candidate["token_prefix_sha256"]
    }
    if tensor_reference["token_prefix_sha256"]:
        token_hashes.add(tensor_reference["token_prefix_sha256"])
    if len(token_hashes) != 1:
        raise ValueError(
            "eligible candidates do not share one token prefix hash")
    if not eligible:
        raise ValueError(
            "no valid candidate aligned with measured fixed-budget endpoint "
            "winners has an advantage over fixed Tensor at both endpoints")
    chosen = min(
        eligible,
        key=lambda candidate: (
            candidate["exec_ms_per_token"],
            candidate["cut_boundary_ms"],
            candidate["merge_boundary_ms"],
        ),
    )
    chosen["speedup_vs_fixed_tensor_pct"] = (
        (
            tensor_reference["exec_ms_per_token"] -
            chosen["exec_ms_per_token"]
        ) / tensor_reference["exec_ms_per_token"] * 100.0
    )
    if chosen["speedup_vs_fixed_tensor_pct"] <= 0.0:
        raise ValueError(
            "no fixed-winner-aligned mixed candidate improves the "
            "fixed-Tensor reference")

    selected_root = json.loads(Path(chosen["profile"]).read_text())
    selected_profile = selected_root["profiles"][args.backend]
    transition_calibration = chosen[
        "frontier_transition_calibration"]
    selected_profile["transition_fixed_ms"] = transition_calibration[
        "selected_transition_fixed_ms"]
    selected_profile["transition_ms_per_mib"] = 0.0
    selected_profile["transition_cost_source"] = transition_calibration[
        "source"]
    calibration = selected_root.setdefault(
        "calibration", {}).setdefault(args.backend, {})
    calibration["mixed_boundary"].update({
        "status": "selected",
        "selection_artifact": str(args.calibration_root),
        "selection_metric": "minimum valid exec_ms_per_token",
        "selected_exec_ms_per_token": chosen["exec_ms_per_token"],
        "frontier_transition_calibration": transition_calibration,
        "endpoint_winner_constraint": (
            f"{expected_low_mode} at {args.low_fixed_ratio}% maps to the "
            f"low node; {expected_high_mode} at "
            f"{args.high_fixed_ratio}% maps to the high node; both winners "
            "come from this run's fixed-budget measurements"
        ),
    })
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        json.dumps(selected_root, indent=2, sort_keys=True) + "\n")

    payload = {
        "backend": args.backend,
        "selection_data": (
            "fixed-node calibration; original 600-second trace held out"
        ),
        "low_budget_mib": args.low_budget_mib,
        "high_budget_mib": args.high_budget_mib,
        "low_fixed_ratio": args.low_fixed_ratio,
        "high_fixed_ratio": args.high_fixed_ratio,
        "measured_low_fixed_winner": expected_low_mode,
        "measured_high_fixed_winner": expected_high_mode,
        "token_prefix_sha256": next(iter(token_hashes)),
        "fixed_tensor_reference": tensor_reference,
        "calibration_lineage": calibration_lineage,
        "identical_frontier_repeatability": repeatability_groups,
        "identical_frontier_max_relative_spread_pct":
            repeatability_noise_pct,
        "candidates": candidates,
        "selected": chosen,
        "output_profile": str(args.output_profile),
    }
    (args.calibration_root / "selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Mixed-boundary calibration",
        "",
        "The original 600-second trace was held out.",
        "",
        "Fixed-Tensor reference: "
        f"`{tensor_reference['exec_ms_per_token']:.3f}` ms/token.",
        " Node means (low→high): "
        f"`{node_means_text(tensor_reference['node_latency'])}` ms.",
        "",
        "| Cut boundary (ms) | Merge boundary (ms) | "
        "Exec (ms/token) | Node mean ms (low→high) | "
        "Low flexible-weight coverage M/T/C | Low Cut/merge edges | "
        "High flexible-weight coverage M/T/C | High Cut/merge edges | Valid |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for candidate in candidates:
        low = candidate["low_mode_counts"]
        high = candidate["high_mode_counts"]
        lines.append(
            f"| {candidate['cut_boundary_ms']:.3f} | "
            f"{candidate['merge_boundary_ms']:.3f} | "
            f"{candidate['exec_ms_per_token']:.3f} | "
            f"{node_means_text(candidate['node_latency'])} | "
            f"{low['multi_flexible_weight_coverage']}/"
            f"{low['tensor_flexible_weight_coverage']}/"
            f"{low['cut_flexible_weight_coverage']} | "
            f"{low['cut_boundaries']}/{low['merge_boundaries']} | "
            f"{high['multi_flexible_weight_coverage']}/"
            f"{high['tensor_flexible_weight_coverage']}/"
            f"{high['cut_flexible_weight_coverage']} | "
            f"{high['cut_boundaries']}/{high['merge_boundaries']} | "
            f"{'yes' if candidate['valid'] and candidate['endpoint_winner_alignment_valid'] and candidate['measured_endpoint_advantage'] else 'no'} |"
        )
    lines.extend([
        "",
        f"Selected Cut-boundary cost: "
        f"{chosen['cut_boundary_ms']:.3f} ms.",
        "",
        "Selected mixed frontier speedup over the fixed-Tensor reference: "
        f"`{chosen['speedup_vs_fixed_tensor_pct']:.2f}%`.",
        " Endpoint speedups (low/high): "
        f"`{chosen['low_speedup_vs_fixed_tensor_pct']:.2f}%` / "
        f"`{chosen['high_speedup_vs_fixed_tensor_pct']:.2f}%`.",
        "",
        "Calibrated working-unit transition cost: "
        f"`{float(transition_calibration['selected_transition_fixed_ms']):.3f}` "
        "ms (maximum isolated changed-frontier publication time, excluding "
        "initial installation, in the selected fixed-node candidate).",
        "",
        "Maximum observed per-node spread between structurally identical "
        f"frontiers: `{repeatability_noise_pct:.2f}%`.",
        "",
    ])
    (args.calibration_root / "SUMMARY.md").write_text(
        "\n".join(lines))


if __name__ == "__main__":
    main()
