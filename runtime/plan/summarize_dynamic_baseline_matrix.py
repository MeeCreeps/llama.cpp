#!/usr/bin/env python3
"""Consolidate the seven-method real-trace matrix after validity checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

try:
    from runtime.plan.run_dynamic_budget_matrix import (
        parse_log,
        token_sequence,
        token_trace_timing,
        token_trace_timing_contract_failures,
    )
except ModuleNotFoundError:
    from run_dynamic_budget_matrix import (  # type: ignore
        parse_log,
        token_sequence,
        token_trace_timing,
        token_trace_timing_contract_failures,
    )


LABELS = {
    "static-min": "Static-Min",
    "static-max": "Static-Max",
    "mru": "MRU",
    "offline-mixed": "Offline",
    "online": "Online",
}
ORDER = (
    "Static-Min", "Static-Max", "MRU", "Offline",
    "Online", "Diff-before", "Diff-now",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def only_result(
        path: Path, method: str | None = None) -> dict[str, str]:
    rows = read_rows(path)
    if method is not None:
        rows = [row for row in rows if row.get("method") == method]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one result, found {len(rows)}")
    return rows[0]


def truth(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").lower() == "true"


def validate(label: str, row: dict[str, str]) -> None:
    failures = []
    if row.get("status") != "ok" or row.get("rc") != "0":
        failures.append("runtime")
    if not truth(row, "valid_token_trace"):
        failures.append("token trace")
    if not truth(row, "token_trace_timing_contract_valid"):
        failures.append("token trace timing")
    if not truth(row, "decode_wall_contract_valid"):
        failures.append("decode wall timing")
    if not truth(row, "decode_phase_contract_valid"):
        failures.append("decode phase counters")
    if (
        truth(row, "fixed_performance_mode_configured")
        and not truth(row, "fixed_performance_restore_valid")
    ):
        failures.append("fixed-performance restore")
    # A one-method Diff artifact has no local peer against which the runner can
    # mark valid_token_sequence=true.  Reject an explicit failure here; the
    # cross-artifact SHA check below provides the actual seven-method check.
    if row.get("valid_token_sequence", "").lower() == "false":
        failures.append("cross-method token prefix")
    if not truth(row, "valid_measurement_isolation"):
        failures.append("measurement isolation")
    if int(float(row.get("measurement_monitor_samples") or 0)) <= 0:
        failures.append("measurement isolation samples")
    if int(float(row.get("measurement_monitor_failures") or 0)) != 0:
        failures.append("measurement isolation monitor failure")
    if row.get(
            "monitor_thread_incomplete", "false").lower() == "true":
        failures.append("measurement monitor thread incomplete")
    if not row.get("measurement_llama_pid", "").strip():
        failures.append("measured llama process identity")
    if not truth(row, "require_device_idle_configured"):
        failures.append("device-idle configuration")
    thermal_limit_raw = row.get(
        "cooldown_thermal_max_c_configured", "").strip()
    if not thermal_limit_raw or float(thermal_limit_raw) <= 0:
        failures.append("cooldown temperature configuration")
    else:
        thermal_limit = float(thermal_limit_raw)
        temperatures = [
            float(row[key])
            for key in (
                "thermal_cpu_max_c_before",
                "thermal_gpu_max_c_before",
                "thermal_skin_max_c_before",
            )
            if row.get(key, "").strip() != ""
        ]
        status_raw = row.get("thermal_status_before", "").strip()
        status_limit_raw = row.get(
            "cooldown_thermal_status_max_configured", "").strip()
        if not temperatures:
            failures.append("clean-start temperature audit missing")
        elif max(temperatures) > thermal_limit:
            failures.append("clean-start temperature")
        if not status_raw or not status_limit_raw:
            failures.append("clean-start thermal-status audit missing")
        elif int(float(status_raw)) > int(float(status_limit_raw)):
            failures.append("clean-start thermal status")
    battery_min_raw = row.get(
        "min_battery_level_pct_configured", "").strip()
    battery_before_raw = row.get(
        "battery_level_pct_before", "").strip()
    if not battery_min_raw or int(float(battery_min_raw)) <= 0:
        failures.append("battery configuration")
    elif (
        not battery_before_raw
        or int(float(battery_before_raw))
            < int(float(battery_min_raw))
    ):
        failures.append("clean-start battery")
    for field, configured_field in (
        (
            "cpu_freq_limit_min_khz",
            "min_cpu_freq_limit_khz_configured",
        ),
        (
            "cpu_freq_mean_khz",
            "min_cpu_mean_freq_khz_configured",
        ),
        (
            "cpu_freq_median_sample_min_khz",
            "min_cpu_median_sample_min_khz_configured",
        ),
    ):
        configured = float(row.get(configured_field) or 0)
        if configured > 0:
            value = row.get(field, "").strip()
            if not value or float(value) < configured:
                failures.append(f"{field} audit")
    if int(float(row.get("pipeline_budget_violations") or -1)) != 0:
        failures.append("pipeline budget")
    for label_prefix in ("model", "binary"):
        expected_hash = row.get(
            f"expected_{label_prefix}_sha256", "").strip()
        remote_hash = row.get(
            f"remote_{label_prefix}_sha256", "").strip()
        if not expected_hash or remote_hash != expected_hash:
            failures.append(f"remote {label_prefix} lineage")
    for label_prefix in ("model_meta", "cost_dir"):
        local_hash = row.get(
            f"{label_prefix}_sha256", "").strip()
        remote_hash = row.get(
            f"remote_{label_prefix}_sha256", "").strip()
        if not local_hash or remote_hash != local_hash:
            failures.append(f"remote {label_prefix} lineage")
    if row.get("pin_contract_valid", "").lower() != "true":
        failures.append("pin contract")
    if row.get("fused_lane_contract_valid", "").lower() != "true":
        failures.append("fused PREPARE lane contract")
    if row.get(
            "unit_pipeline_stage_authority_contract_valid",
            "").lower() != "true":
        failures.append("unit-pipeline stage authority contract")
    if row.get(
            "backend_compute_contract_valid", "").lower() != "true":
        failures.append("backend compute contract")
    physical_cut_tensors = int(float(
        row.get("physical_cut_tensors") or 0))
    physical_cut_parts = int(float(
        row.get("physical_cut_parts") or 0))
    if physical_cut_tensors <= 0:
        failures.append("physical tiling audit missing")
    elif physical_cut_parts != 2 * physical_cut_tensors:
        failures.append("physical tiling audit mismatch")
    observed_wbm_total_mib = float(
        row.get("wbm_total_mib") or 0.0)
    expected_wbm_total_mib = float(
        row.get("expected_wbm_total_mib_configured") or 0.0)
    planner_stream_reserve_mib = float(
        row.get("planner_stream_reserve_mib_configured") or 0.0)
    if observed_wbm_total_mib <= 0.0:
        failures.append("physical WBM total audit missing")
    if expected_wbm_total_mib <= 0.0:
        failures.append("expected physical WBM total configuration")
    elif abs(observed_wbm_total_mib - expected_wbm_total_mib) > max(
            0.01, expected_wbm_total_mib * 1e-5):
        failures.append("physical WBM total")
    if planner_stream_reserve_mib <= 0.0:
        failures.append("planner stream reserve configuration")
    if int(float(row.get("pin_outside_budget_count") or 0)) != 0:
        failures.append("outside-budget pin")
    plan_relaxations = row.get(
        "pipeline_plan_protection_relaxations", "").strip()
    if not plan_relaxations:
        failures.append("plan-protection audit missing")
    elif int(float(plan_relaxations)) != 0:
        failures.append("plan-protection relaxation")
    if abs(float(row.get("source_start_sec") or -1.0)) > 1e-6:
        failures.append("trace start")
    source_file_hash = row.get("source_file_sha256", "").strip()
    expected_source_hash = row.get(
        "expected_source_trace_sha256", "").strip()
    if not source_file_hash:
        failures.append("original source trace lineage")
    if not expected_source_hash or expected_source_hash != source_file_hash:
        failures.append("normalized source input lineage")
    if (
        row.get("normalized_source_input_sha256", "").strip()
        != source_file_hash
        or row.get(
            "expected_normalized_source_input_sha256", "").strip()
            != source_file_hash
    ):
        failures.append("explicit normalized source input lineage")
    historical_hash = row.get(
        "original_source_trace_sha256", "").strip()
    expected_historical_hash = row.get(
        "expected_original_source_trace_sha256", "").strip()
    if not row.get("original_source_trace", "").strip():
        failures.append("historical original trace path")
    if (
        not historical_hash
        or not expected_historical_hash
        or historical_hash != expected_historical_hash
    ):
        failures.append("historical original trace lineage")
    if row.get(
            "original_source_trace_semantically_equal",
            "").lower() != "true":
        failures.append("historical/normalized trace semantic equality")
    if float(row.get("source_span_sec") or 0) < 599.999:
        failures.append("trace coverage")
    if float(row.get("replay_span_sec") or 0) < 599.999:
        failures.append("replay coverage")
    runtime_trace_hash = row.get(
        "runtime_trace_sha256", "").strip()
    if (
        not runtime_trace_hash
        or row.get(
            "remote_runtime_trace_sha256", "").strip()
            != runtime_trace_hash
    ):
        failures.append("remote runtime trace lineage")
    if row.get("bench_exit_reason") != "duration":
        failures.append("timed benchmark exit")
    if int(float(row.get("bench_time_done") or 0)) != 1:
        failures.append("timed benchmark coverage")
    if int(float(row.get("budget_decode_reset_count") or 0)) != 1:
        failures.append("decode-boundary budget reset count")
    if int(float(row.get("budget_backend_reset_count") or 0)) != 1:
        failures.append("backend budget reset count")
    if int(float(
            row.get("budget_decode_reset_generated", -1))) != 0:
        failures.append("decode-boundary budget replay origin")
    if label in {"Static-Min", "Static-Max"}:
        table_hash = row.get("offline_table_sha256", "").strip()
        if (
            not table_hash
            or row.get("remote_offline_table_sha256", "").strip()
                != table_hash
        ):
            failures.append("offline plan table device lineage")
    if label in {
        "Offline", "Online", "Diff-before", "Diff-now",
    }:
        mixed_table_hash = row.get(
            "mixed_offline_table_sha256", "").strip()
        if (
            not mixed_table_hash
            or row.get(
                "remote_mixed_offline_table_sha256", "").strip()
                != mixed_table_hash
        ):
            failures.append("mixed offline plan table device lineage")
        if not row.get("granularity_profile", "").strip():
            failures.append("granularity profile path")
        if not row.get("granularity_profile_sha256", "").strip():
            failures.append("granularity profile hash")
        if row.get("multi_implementation_configured") not in {
            "multi", "multi_fused",
        }:
            failures.append("coarse-unit implementation contract")
        if row.get("cut_compute_grouping_configured") not in {
            "independent_halves", "fused_pair",
        }:
            failures.append("Cut compute grouping contract")
        if float(
                row.get("transition_fixed_ms_configured") or 0.0) <= 0.0:
            failures.append("measured frontier transition cost")
        if not row.get(
                "transition_cost_source_configured", "").strip():
            failures.append("frontier transition cost source")
        expected_placement_source = (
            "offline-table"
            if label == "Offline" else "stateful-cp"
        )
        if (
            row.get("granularity_placement_source")
            != expected_placement_source
        ):
            failures.append("granularity placement source")
        if not row.get("granularity_policy_configured", "").strip():
            failures.append("granularity policy")
        expected_effective_policy = {
            "Offline": "offline",
            "Online": "online",
            "Diff-before": "diff-tree",
            "Diff-now": "diff-tree",
        }[label]
        if (
            row.get("granularity_policy_effective")
            != expected_effective_policy
        ):
            failures.append("effective granularity policy")
        if int(float(row.get("frontier_apply_count") or 0)) <= 0:
            failures.append("working-unit frontier not applied")
        active_modes = sum(
            int(float(row.get(
                f"frontier_{mode}_units_max") or 0)) > 0
            for mode in ("multi", "tensor", "cut")
        )
        if active_modes < 2:
            failures.append("mixed-granularity frontier not exercised")
        if int(float(row.get(
                "frontier_mixed_mode_generations") or 0)) <= 0:
            failures.append(
                "no applied frontier simultaneously used multiple modes")
        if not row.get("frontier_trace_sha256", "").strip():
            failures.append("applied-frontier trace hash missing")
        if not row.get(
                "frontier_transition_trace_sha256", "").strip():
            failures.append("frontier transition-path hash missing")
        frontier_changes = int(float(
            row.get("frontier_state_changes") or 0))
        transition_publications = int(float(
            row.get("frontier_transition_publish_count") or 0))
        if transition_publications != frontier_changes:
            failures.append(
                "changed-frontier publication count mismatch")
        if (
            frontier_changes > 0
            and (
                float(row.get(
                    "frontier_transition_publish_ms_total") or 0.0) <= 0.0
                or float(row.get(
                    "frontier_transition_publish_ms_max") or 0.0) <= 0.0
                or int(float(row.get(
                    "frontier_transition_delta_weights_total") or 0)) <= 0
                or int(float(row.get(
                    "frontier_transition_delta_weights_max") or 0)) <= 0
            )
        ):
            failures.append(
                "changed-frontier publication audit missing")
    if label in {"Online", "Diff-before", "Diff-now"}:
        if int(float(row.get("online_calls") or 0)) <= 0:
            failures.append("online planner not exercised")
        if row.get(
                "remote_solver_fresh_process", "").lower() != "true":
            failures.append("fresh remote solver process")
        if row.get(
                "remote_solver_contract_valid", "").lower() != "true":
            failures.append("remote solver isolation contract")
    if failures:
        raise ValueError(f"{label}: invalid {', '.join(failures)}")


def validate_profile_lineage(
    selected: dict[str, dict[str, str]]) -> None:
    """Prove the Diff attribution from recorded profile content hashes."""
    implementations = {
        row.get("multi_implementation_configured", "")
        for row in selected.values()
    }
    if "" in implementations or len(implementations) != 1:
        raise ValueError(
            "all methods must share one coarse-unit implementation, "
            f"got {sorted(implementations)}")
    transition_costs = {
        round(float(
            row.get("transition_fixed_ms_configured") or 0.0), 6)
        for row in selected.values()
    }
    transition_sources = {
        row.get("transition_cost_source_configured", "").strip()
        for row in selected.values()
    }
    if (
        len(transition_costs) != 1
        or next(iter(transition_costs)) <= 0.0
        or "" in transition_sources
        or len(transition_sources) != 1
    ):
        raise ValueError(
            "all methods must share one positive measured frontier "
            "transition cost and source, "
            f"costs={sorted(transition_costs)} "
            f"sources={sorted(transition_sources)}")
    residual_labels = ("Offline", "Online", "Diff-now")
    residual_hashes = {
        selected[label].get("granularity_profile_sha256", "")
        for label in residual_labels
    }
    if "" in residual_hashes or len(residual_hashes) != 1:
        raise ValueError(
            "Offline, Online, and Diff-now must share one granularity "
            f"profile hash, got {sorted(residual_hashes)}")
    for label in residual_labels:
        if (
            selected[label].get("granularity_profile_kind")
            != "pipeline-residual"
        ):
            raise ValueError(
                f"{label}: expected pipeline-residual granularity profile")
    residual_table_hashes = {
        selected[label].get("mixed_offline_table_sha256", "")
        for label in residual_labels
    }
    if (
        "" in residual_table_hashes
        or len(residual_table_hashes) != 1
    ):
        raise ValueError(
            "Offline, Online, and Diff-now must share one mixed offline "
            f"table hash, got {sorted(residual_table_hashes)}")

    before = selected["Diff-before"]
    if before.get("granularity_profile_kind") != "phase-only":
        raise ValueError(
            "Diff-before: expected phase-only granularity profile")
    before_hash = before.get("granularity_profile_sha256", "")
    if not before_hash or before_hash in residual_hashes:
        raise ValueError(
            "Diff-before and Diff-now must use different profile content")
    before_frontier_hash = before.get(
        "frontier_transition_trace_sha256", "").strip()
    now_frontier_hash = selected["Diff-now"].get(
        "frontier_transition_trace_sha256", "").strip()
    if (
        not before_frontier_hash
        or not now_frontier_hash
        or before_frontier_hash == now_frontier_hash
    ):
        raise ValueError(
            "Diff-before and Diff-now must execute different ordered "
            "split/merge frontier decisions")
    if not before.get("mixed_offline_table_sha256", ""):
        raise ValueError(
            "Diff-before: mixed offline table hash missing")


def validate_cut_compute_lineage(
    backend: str,
    selected: dict[str, dict[str, str]],
) -> None:
    """Require one explicit runtime/profile Cut dependency for every method."""
    expected = (
        "fused_pair" if backend == "gpu" else "independent_halves")
    observed = {
        row.get("cut_compute_grouping_configured", "")
        for row in selected.values()
    }
    if observed != {expected}:
        raise ValueError(
            f"{backend} methods do not use the required Cut compute "
            f"grouping {expected!r}: {sorted(observed)}")
    if backend != "gpu":
        return
    for label, row in selected.items():
        fallback = row.get("cut_dual_budget_fallbacks", "")
        if fallback == "":
            raise ValueError(
                f"{label}: missing fused Cut budget-fallback audit")
        if int(float(fallback)) != 0:
            raise ValueError(
                f"{label}: fused Cut used {fallback} budget fallback(s)")


def validate_runtime_input_lineage(
        selected: dict[str, dict[str, str]]) -> None:
    """Require every baseline to run the same exact runtime inputs."""
    for input_name in ("model", "binary"):
        hashes = {
            row.get(f"remote_{input_name}_sha256", "")
            for row in selected.values()
        }
        expected = {
            row.get(f"expected_{input_name}_sha256", "")
            for row in selected.values()
        }
        if "" in hashes or len(hashes) != 1 or hashes != expected:
            raise ValueError(
                f"all methods must share one verified {input_name} "
                f"SHA-256, got remote={sorted(hashes)} "
                f"expected={sorted(expected)}")
    for input_name in ("model_meta", "cost_dir"):
        remote = {
            row.get(f"remote_{input_name}_sha256", "")
            for row in selected.values()
        }
        local = {
            row.get(f"{input_name}_sha256", "")
            for row in selected.values()
        }
        if "" in remote or len(remote) != 1 or remote != local:
            raise ValueError(
                f"all methods must share one verified {input_name} "
                f"SHA-256, got remote={sorted(remote)} "
                f"local={sorted(local)}")
    config_hashes = {
        row.get("dynamic_run_config_sha256", "")
        for row in selected.values()
    }
    if "" in config_hashes or len(config_hashes) != 1:
        raise ValueError(
            "all methods must share one dynamic runtime configuration "
            f"SHA-256, got {sorted(config_hashes)}")
    core_config_hashes = {
        row.get("dynamic_run_core_config_sha256", "")
        for row in selected.values()
    }
    if "" in core_config_hashes or len(core_config_hashes) != 1:
        raise ValueError(
            "all methods must share one dynamic runtime core configuration "
            f"SHA-256, got {sorted(core_config_hashes)}")
    historical_hashes = {
        row.get("original_source_trace_sha256", "")
        for row in selected.values()
    }
    expected_historical_hashes = {
        row.get("expected_original_source_trace_sha256", "")
        for row in selected.values()
    }
    if (
        "" in historical_hashes
        or len(historical_hashes) != 1
        or historical_hashes != expected_historical_hashes
        or any(
            row.get(
                "original_source_trace_semantically_equal",
                "").lower() != "true"
            for row in selected.values()
        )
    ):
        raise ValueError(
            "all methods must share one verified historical original "
            f"trace, got actual={sorted(historical_hashes)} "
            f"expected={sorted(expected_historical_hashes)}")
    physical = {
        (
            int(float(row.get("physical_cut_tensors") or 0)),
            int(float(row.get("physical_cut_parts") or 0)),
            round(float(row.get("wbm_total_mib") or 0.0), 6),
        )
        for row in selected.values()
    }
    if len(physical) != 1:
        raise ValueError(
            "all methods must share one physical tiling and WBM total, "
            f"got {sorted(physical)}")
    tensors, parts, total_mib = next(iter(physical))
    if tensors <= 0 or parts != 2 * tensors or total_mib <= 0.0:
        raise ValueError(
            "invalid common physical tiling/WBM total: "
            f"{(tensors, parts, total_mib)}")
    expected_totals = {
        round(float(
            row.get("expected_wbm_total_mib_configured") or 0.0), 6)
        for row in selected.values()
    }
    expected_total = (
        next(iter(expected_totals))
        if len(expected_totals) == 1 else 0.0)
    if (
        len(expected_totals) != 1
        or expected_total <= 0.0
        or abs(total_mib - expected_total) > max(
            0.01, expected_total * 1e-5)
    ):
        raise ValueError(
            "all methods must share and match one explicitly configured "
            "physical WBM total, "
            f"configured={sorted(expected_totals)} observed={total_mib}")
    stream_reserves = {
        round(float(
            row.get("planner_stream_reserve_mib_configured") or 0.0), 6)
        for row in selected.values()
    }
    if (
        len(stream_reserves) != 1
        or next(iter(stream_reserves)) <= 0.0
    ):
        raise ValueError(
            "all methods must share one positive planner stream reserve, "
            f"got {sorted(stream_reserves)}")


def as_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key) or 0.0)


def enrich_from_raw_log(row: dict[str, str]) -> dict[str, str]:
    """Reparse the authoritative raw log with the latest audit schema.

    Long device sweeps may have been launched before a new diagnostic column
    was added to the matrix runner.  The raw log remains authoritative, so the
    final consolidation should not require rerunning a valid 600-second trace
    solely to populate a derived CSV column.
    """
    enriched = dict(row)
    log = Path(row.get("log", ""))
    if not log.is_file():
        return enriched
    parsed = parse_log(log, row.get("method", ""))
    for key, value in parsed.items():
        if value != "":
            # The matrix runner applies cross-log validity gates (for example,
            # device frequency and thermal isolation) after parsing the raw
            # benchmark log.  Re-parsing here must never turn one of those
            # rejected rows back into an apparently valid result.
            if key == "status" and enriched.get("status"):
                continue
            enriched[key] = str(value)
    return enriched


def with_source(
        row: dict[str, str], source: Path) -> dict[str, str]:
    enriched = enrich_from_raw_log(row)
    enriched["_result_source"] = str(source)
    return enriched


def per_forward(row: dict[str, str], key: str) -> float:
    runs = as_float(row, "decode_phase_runs")
    return as_float(row, key) / runs if runs > 0 else 0.0


def format_metric(value: object) -> str:
    if value in ("", None):
        return "N/A"
    return f"{float(value):.3f}"


def budget_trace_points(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        time_value = row.get("t_sec", row.get("time_sec", ""))
        budget_value = row.get(
            "mem_available_mb", row.get("budget_mb", ""))
        if time_value == "" or budget_value == "":
            raise ValueError(f"{path}: invalid budget trace schema")
        points.append((float(time_value), float(budget_value)))
    if not points:
        raise ValueError(f"{path}: empty budget trace")
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument(
        "--replacement-results",
        type=Path,
        action="append",
        default=[],
        help=(
            "optional newer matrix whose Offline/Online rows replace the "
            "corresponding rows from --main-results; repeat as needed"
        ),
    )
    parser.add_argument("--diff-before-results", type=Path, required=True)
    parser.add_argument("--diff-now-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected: dict[str, dict[str, str]] = {}
    for row in read_rows(args.main_results):
        method = row.get("method", "")
        if method in LABELS:
            label = LABELS[method]
            if label in selected:
                raise ValueError(
                    f"{args.main_results}: duplicate {label} row")
            selected[label] = with_source(row, args.main_results)
    for replacement in args.replacement_results:
        for row in read_rows(replacement):
            method = row.get("method", "")
            if method in LABELS:
                selected[LABELS[method]] = with_source(
                    row, replacement)
    selected["Diff-before"] = with_source(
        only_result(args.diff_before_results, "diff-tree-mixed"),
        args.diff_before_results,
    )
    selected["Diff-now"] = with_source(
        only_result(args.diff_now_results, "diff-tree-mixed"),
        args.diff_now_results,
    )
    if (
        args.diff_before_results.resolve()
        == args.diff_now_results.resolve()
    ):
        raise ValueError(
            "Diff-before and Diff-now must come from independent artifacts")
    missing = [label for label in ORDER if label not in selected]
    if missing:
        raise ValueError(f"missing results: {', '.join(missing)}")
    validate_profile_lineage(selected)
    validate_runtime_input_lineage(selected)
    validate_cut_compute_lineage(args.backend, selected)

    gate_hashes = set()
    token_paths: dict[str, Path] = {}
    source_window_paths: dict[str, Path] = {}
    runtime_trace_paths: dict[str, Path] = {}
    original_source_paths: dict[str, Path] = {}
    trace_metadata: set[tuple[str, ...]] = set()
    output_rows = []
    for label in ORDER:
        row = selected[label]
        validate(label, row)
        gate_hashes.add(row.get("token_prefix_sha256", ""))
        token_path = Path(row.get("token_trace", ""))
        if not token_path.is_file():
            raise ValueError(f"{label}: missing token trace {token_path}")
        token_paths[label] = token_path
        token_timing = token_trace_timing(token_path)
        token_timing_failures = token_trace_timing_contract_failures(
            token_timing, 600.0)
        if token_timing_failures:
            raise ValueError(
                f"{label}: invalid token timing coverage: "
                + "; ".join(token_timing_failures))
        result_root = Path(row["_result_source"]).parent.parent
        source_window_path = (
            result_root / "traces" / row.get("trace", ""))
        if not source_window_path.is_file():
            raise ValueError(
                f"{label}: missing source window {source_window_path}")
        source_window_paths[label] = source_window_path
        original_source_path = Path(
            row.get("original_source_trace", ""))
        if not original_source_path.is_file():
            raise ValueError(
                f"{label}: missing historical original trace "
                f"{original_source_path}")
        actual_original_hash = hashlib.sha256(
            original_source_path.read_bytes()).hexdigest()
        if (
            actual_original_hash
            != row.get("original_source_trace_sha256", "")
        ):
            raise ValueError(
                f"{label}: historical original trace hash changed")
        if (
            budget_trace_points(original_source_path)
            != budget_trace_points(source_window_path)
        ):
            raise ValueError(
                f"{label}: normalized source window is not point-for-point "
                "equal to the historical original trace")
        original_source_paths[label] = original_source_path
        runtime_trace_path = (
            result_root / "traces" / row.get("runtime_trace", ""))
        if not row.get("runtime_trace") or not runtime_trace_path.is_file():
            raise ValueError(
                f"{label}: missing effective runtime trace "
                f"{runtime_trace_path}")
        runtime_trace_paths[label] = runtime_trace_path
        trace_metadata.add(tuple(
            row.get(key, "")
            for key in (
                "source_trace", "source_start_sec", "source_span_sec",
                "replay_span_sec", "min_mib", "mean_mib", "max_mib",
                "min_bucket_mib", "max_bucket_mib",
            )
        ))
        output_rows.append({
            "backend": args.backend,
            "method": label,
            "status": row.get("status", ""),
            "rc": row.get("rc", ""),
            "result_source": row.get("_result_source", ""),
            "expected_model_sha256":
                row.get("expected_model_sha256", ""),
            "remote_model_sha256":
                row.get("remote_model_sha256", ""),
            "expected_binary_sha256":
                row.get("expected_binary_sha256", ""),
            "remote_binary_sha256":
                row.get("remote_binary_sha256", ""),
            "model_meta_sha256":
                row.get("model_meta_sha256", ""),
            "remote_model_meta_sha256":
                row.get("remote_model_meta_sha256", ""),
            "cost_dir_sha256":
                row.get("cost_dir_sha256", ""),
            "remote_cost_dir_sha256":
                row.get("remote_cost_dir_sha256", ""),
            "dynamic_run_config_sha256":
                row.get("dynamic_run_config_sha256", ""),
            "dynamic_run_core_config_sha256":
                row.get("dynamic_run_core_config_sha256", ""),
            "granularity_profile":
                row.get("granularity_profile", ""),
            "granularity_profile_sha256":
                row.get("granularity_profile_sha256", ""),
            "granularity_profile_kind":
                row.get("granularity_profile_kind", ""),
            "multi_implementation_configured":
                row.get("multi_implementation_configured", ""),
            "cut_compute_grouping_configured":
                row.get("cut_compute_grouping_configured", ""),
            "transition_fixed_ms_configured":
                row.get("transition_fixed_ms_configured", ""),
            "transition_cost_source_configured":
                row.get("transition_cost_source_configured", ""),
            "granularity_placement_source":
                row.get("granularity_placement_source", ""),
            "granularity_placement_source_configured":
                row.get(
                    "granularity_placement_source_configured", ""),
            "granularity_policy_configured":
                row.get("granularity_policy_configured", ""),
            "granularity_policy_effective":
                row.get("granularity_policy_effective", ""),
            "mixed_offline_table_sha256":
                row.get("mixed_offline_table_sha256", ""),
            "offline_table_sha256":
                row.get("offline_table_sha256", ""),
            "remote_offline_table_sha256":
                row.get("remote_offline_table_sha256", ""),
            "remote_mixed_offline_table_sha256":
                row.get("remote_mixed_offline_table_sha256", ""),
            "exec_ms_per_token": row.get("exec_ms_per_token", ""),
            "exec_timing_source": row.get("exec_timing_source", ""),
            "exec_component_sum_ms_per_token":
                row.get("exec_component_sum_ms_per_token", ""),
            "decode_wall_ms_total":
                row.get("decode_wall_ms_total", ""),
            "decode_wall_runs": row.get("decode_wall_runs", ""),
            "decode_phase_contract_valid":
                row.get("decode_phase_contract_valid", ""),
            "decode_phase_counter_scope":
                row.get("decode_phase_counter_scope", ""),
            "decode_phase_runs": row.get("decode_phase_runs", ""),
            "fixed_performance_mode_configured":
                row.get("fixed_performance_mode_configured", ""),
            "fixed_performance_restore_attempted":
                row.get("fixed_performance_restore_attempted", ""),
            "fixed_performance_restore_valid":
                row.get("fixed_performance_restore_valid", ""),
            "decode_wall_unattributed_ms_total":
                row.get("decode_wall_unattributed_ms_total", ""),
            "raw_ms_per_token": row.get("raw_ms_per_token", ""),
            "generated_tokens": row.get("generated_tokens", ""),
            "bench_exit_reason": row.get("bench_exit_reason", ""),
            "bench_time_done": row.get("bench_time_done", ""),
            "budget_decode_reset_count":
                row.get("budget_decode_reset_count", ""),
            "budget_decode_reset_generated":
                row.get("budget_decode_reset_generated", ""),
            "budget_backend_reset_count":
                row.get("budget_backend_reset_count", ""),
            "token_trace_decode_rows":
                token_timing["token_trace_decode_rows"],
            "token_trace_t_first_sec":
                token_timing["token_trace_t_first_sec"],
            "token_trace_t_last_sec":
                token_timing["token_trace_t_last_sec"],
            "token_trace_t_span_sec":
                token_timing["token_trace_t_span_sec"],
            "token_trace_first_budget_mib":
                token_timing["token_trace_first_budget_mib"],
            "token_trace_last_budget_mib":
                token_timing["token_trace_last_budget_mib"],
            "token_trace_max_latency_ms":
                token_timing["token_trace_max_latency_ms"],
            "token_trace_time_monotonic":
                token_timing["token_trace_time_monotonic"],
            "token_trace_budget_provider_valid":
                token_timing["token_trace_budget_provider_valid"],
            "direct_read_mib_per_forward":
                per_forward(row, "decode_phase_direct_read_mib"),
            "direct_read_ms_per_forward":
                per_forward(row, "decode_phase_direct_read_ms"),
            "load_stage_ms_per_forward":
                per_forward(row, "decode_phase_load_ms"),
            "nonresident_units_per_forward":
                per_forward(row, "decode_phase_nonresident_units"),
            "layout_prepare_ms_per_forward": (
                per_forward(row, "decode_phase_prepare_ms")
                if args.backend == "cpu" else ""),
            "prepare_host_issue_ms_per_forward": (
                per_forward(row, "decode_phase_prepare_ms")
                if args.backend == "gpu" else ""),
            "compute_ms_per_forward": (
                per_forward(row, "decode_phase_compute_ms")
                if args.backend == "cpu" else ""
            ),
            "pipeline_wait_ms_per_forward":
                per_forward(row, "decode_phase_pipeline_wait_ms"),
            "pipeline_residency_ms_per_forward":
                per_forward(row, "decode_phase_pipeline_residency_ms"),
            "plan_apply_ms_per_forward":
                per_forward(row, "apply_ms_total"),
            "plan_apply_ms_max": row.get("apply_ms_max", ""),
            "provider_get_ms_max": row.get(
                "provider_get_ms_max", ""),
            "remote_wall_ms_max": row.get(
                "remote_wall_ms_max", ""),
            "remote_server_ms_max": row.get(
                "remote_server_ms_max", ""),
            "pipeline_wait_ms":
                row.get("decode_phase_pipeline_wait_ms", ""),
            "layout_prepare_ms":
                row.get("decode_phase_prepare_ms", ""),
            "pipeline_residency_ms":
                row.get("decode_phase_pipeline_residency_ms", ""),
            "pipeline_missing_mib": row.get("pipeline_missing_mib", ""),
            "pipeline_unissued_mib": row.get("pipeline_unissued_mib", ""),
            "pipeline_budget_violations":
                row.get("pipeline_budget_violations", ""),
            "pin_contract_valid": row.get("pin_contract_valid", ""),
            "pin_token_embd_inside_budget_count":
                row.get("pin_token_embd_inside_budget_count", ""),
            "pin_output_inside_budget_count":
                row.get("pin_output_inside_budget_count", ""),
            "pin_outside_budget_count":
                row.get("pin_outside_budget_count", ""),
            "plan_protection_relaxations":
                row.get("pipeline_plan_protection_relaxations", ""),
            "frontier_contract_valid":
                row.get("frontier_contract_valid", ""),
            "frontier_mixed_mode_generations":
                row.get("frontier_mixed_mode_generations", ""),
            "unit_pipeline_stage_authority_contract_valid":
                row.get(
                    "unit_pipeline_stage_authority_contract_valid", ""),
            "plan_stage_defer_apply_count":
                row.get("plan_stage_defer_apply_count", ""),
            "plan_stage_defer_min":
                row.get("plan_stage_defer_min", ""),
            "plan_stage_defer_max":
                row.get("plan_stage_defer_max", ""),
            "plan_anchor_fired":
                row.get("plan_anchor_fired", ""),
            "plan_anchor_load_events":
                row.get("plan_anchor_load_events", ""),
            "plan_anchor_transfer_events":
                row.get("plan_anchor_transfer_events", ""),
            "plan_anchor_xform_events":
                row.get("plan_anchor_xform_events", ""),
            "backend_compute_contract_valid":
                row.get("backend_compute_contract_valid", ""),
            "fused_layout_pair_calls":
                row.get("fused_layout_pair_calls", ""),
            "fused_kernel_pair_candidates":
                row.get("fused_kernel_pair_candidates", ""),
            "fused_kernel_pair_calls":
                row.get("fused_kernel_pair_calls", ""),
            "fused_kernel_pair_fallbacks":
                row.get("fused_kernel_pair_fallbacks", ""),
            "fused_kernel_pair_errors":
                row.get("fused_kernel_pair_errors", ""),
            "remote_solver_fresh_process":
                row.get("remote_solver_fresh_process", ""),
            "remote_solver_contract_valid":
                row.get("remote_solver_contract_valid", ""),
            "remote_solver_event_count":
                row.get("remote_solver_event_count", ""),
            "remote_solver_run_ids":
                row.get("remote_solver_run_ids", ""),
            "remote_solver_working_unit_policies":
                row.get(
                    "remote_solver_working_unit_policies", ""),
            "remote_solver_placement_sources":
                row.get(
                    "remote_solver_placement_sources", ""),
            "remote_solver_diff_edit_count_max":
                row.get(
                    "remote_solver_diff_edit_count_max", ""),
            "remote_solver_log": row.get("remote_solver_log", ""),
            "cut_dual_candidates":
                row.get("cut_dual_candidates", ""),
            "cut_dual_calls": row.get("cut_dual_calls", ""),
            "cut_dual_budget_fallbacks":
                row.get("cut_dual_budget_fallbacks", ""),
            "cut_dual_shape_fallbacks":
                row.get("cut_dual_shape_fallbacks", ""),
            "cut_dual_queue_errors":
                row.get("cut_dual_queue_errors", ""),
            "cut_dual_image_creates":
                row.get("cut_dual_image_creates", ""),
            "cut_dual_image_cache_hits":
                row.get("cut_dual_image_cache_hits", ""),
            "cut_dual_image_releases":
                row.get("cut_dual_image_releases", ""),
            "cut_dual_image_errors":
                row.get("cut_dual_image_errors", ""),
            "physical_cut_tensors":
                row.get("physical_cut_tensors", ""),
            "physical_cut_parts":
                row.get("physical_cut_parts", ""),
            "wbm_total_mib":
                row.get("wbm_total_mib", ""),
            "expected_wbm_total_mib_configured":
                row.get("expected_wbm_total_mib_configured", ""),
            "planner_stream_reserve_mib_configured":
                row.get("planner_stream_reserve_mib_configured", ""),
            "plan_protection_relaxed_mib_per_forward":
                per_forward(
                    row, "pipeline_plan_protection_relaxed_mib"),
            "apply_count": row.get("apply_count", ""),
            "frontier_multi_units_last":
                row.get("frontier_multi_units_last", ""),
            "frontier_tensor_units_last":
                row.get("frontier_tensor_units_last", ""),
            "frontier_cut_units_last":
                row.get("frontier_cut_units_last", ""),
            "frontier_multi_units_max":
                row.get("frontier_multi_units_max", ""),
            "frontier_tensor_units_max":
                row.get("frontier_tensor_units_max", ""),
            "frontier_cut_units_max":
                row.get("frontier_cut_units_max", ""),
            "frontier_switch_ms_total":
                row.get("frontier_switch_ms_total", ""),
            "frontier_switch_ms_max":
                row.get("frontier_switch_ms_max", ""),
            "frontier_publish_ms_total":
                row.get("frontier_publish_ms_total", ""),
            "frontier_publish_ms_max":
                row.get("frontier_publish_ms_max", ""),
            "frontier_transition_publish_count":
                row.get("frontier_transition_publish_count", ""),
            "frontier_transition_publish_ms_total":
                row.get("frontier_transition_publish_ms_total", ""),
            "frontier_transition_publish_ms_max":
                row.get("frontier_transition_publish_ms_max", ""),
            "frontier_delta_weights_total":
                row.get("frontier_delta_weights_total", ""),
            "frontier_delta_weights_max":
                row.get("frontier_delta_weights_max", ""),
            "frontier_transition_delta_weights_total":
                row.get("frontier_transition_delta_weights_total", ""),
            "frontier_transition_delta_weights_max":
                row.get("frontier_transition_delta_weights_max", ""),
            "frontier_only_apply_count":
                row.get("frontier_only_apply_count", ""),
            "frontier_trace_sha256":
                row.get("frontier_trace_sha256", ""),
            "frontier_transition_trace_sha256":
                row.get("frontier_transition_trace_sha256", ""),
            "frontier_shape_trace_sha256":
                row.get("frontier_shape_trace_sha256", ""),
            "frontier_unique_states":
                row.get("frontier_unique_states", ""),
            "frontier_unique_shape_states":
                row.get("frontier_unique_shape_states", ""),
            "frontier_state_changes":
                row.get("frontier_state_changes", ""),
            "frontier_shape_state_changes":
                row.get("frontier_shape_state_changes", ""),
            "online_calls": row.get("online_calls", ""),
            "token_prefix_sha256": row.get("token_prefix_sha256", ""),
            "measurement_monitor_samples":
                row.get("measurement_monitor_samples", ""),
            "measurement_monitor_failures":
                row.get("measurement_monitor_failures", ""),
            "measurement_llama_pid":
                row.get("measurement_llama_pid", ""),
            "monitor_thread_incomplete":
                row.get("monitor_thread_incomplete", ""),
            "valid_measurement_isolation":
                row.get("valid_measurement_isolation", ""),
            "measurement_overlap":
                row.get("measurement_overlap", ""),
            "thermal_status_before":
                row.get("thermal_status_before", ""),
            "thermal_cpu_max_c_before":
                row.get("thermal_cpu_max_c_before", ""),
            "thermal_gpu_max_c_before":
                row.get("thermal_gpu_max_c_before", ""),
            "thermal_skin_max_c_before":
                row.get("thermal_skin_max_c_before", ""),
            "thermal_status_after":
                row.get("thermal_status_after", ""),
            "thermal_cpu_max_c_after":
                row.get("thermal_cpu_max_c_after", ""),
            "thermal_gpu_max_c_after":
                row.get("thermal_gpu_max_c_after", ""),
            "thermal_skin_max_c_after":
                row.get("thermal_skin_max_c_after", ""),
            "battery_level_pct_before":
                row.get("battery_level_pct_before", ""),
            "battery_level_pct_after":
                row.get("battery_level_pct_after", ""),
            "cpu_freq_limit_min_khz":
                row.get("cpu_freq_limit_min_khz", ""),
            "cpu_freq_mean_khz": row.get("cpu_freq_mean_khz", ""),
            "cpu_freq_median_sample_min_khz":
                row.get("cpu_freq_median_sample_min_khz", ""),
            "require_device_idle_configured":
                row.get("require_device_idle_configured", ""),
            "cooldown_thermal_max_c_configured":
                row.get("cooldown_thermal_max_c_configured", ""),
            "cooldown_thermal_status_max_configured":
                row.get(
                    "cooldown_thermal_status_max_configured", ""),
            "min_battery_level_pct_configured":
                row.get("min_battery_level_pct_configured", ""),
            "min_cpu_freq_limit_khz_configured":
                row.get("min_cpu_freq_limit_khz_configured", ""),
            "min_cpu_mean_freq_khz_configured":
                row.get("min_cpu_mean_freq_khz_configured", ""),
            "min_cpu_median_sample_min_khz_configured":
                row.get(
                    "min_cpu_median_sample_min_khz_configured", ""),
            "source_trace": row.get("source_trace", ""),
            "source_window": row.get("trace", ""),
            "source_window_sha256":
                row.get("source_window_sha256", ""),
            "source_file_sha256":
                row.get("source_file_sha256", ""),
            "normalized_source_input_sha256":
                row.get("normalized_source_input_sha256", ""),
            "expected_normalized_source_input_sha256":
                row.get(
                    "expected_normalized_source_input_sha256", ""),
            "expected_source_trace_sha256":
                row.get("expected_source_trace_sha256", ""),
            "original_source_trace":
                row.get("original_source_trace", ""),
            "original_source_trace_sha256":
                row.get("original_source_trace_sha256", ""),
            "expected_original_source_trace_sha256":
                row.get(
                    "expected_original_source_trace_sha256", ""),
            "original_source_trace_semantically_equal":
                row.get(
                    "original_source_trace_semantically_equal", ""),
            "runtime_trace": row.get("runtime_trace", ""),
            "runtime_trace_kind": row.get("runtime_trace_kind", ""),
            "runtime_trace_budget_mib":
                row.get("runtime_trace_budget_mib", ""),
            "runtime_trace_sha256":
                row.get("runtime_trace_sha256", ""),
            "remote_runtime_trace_sha256":
                row.get("remote_runtime_trace_sha256", ""),
            "source_start_sec": row.get("source_start_sec", ""),
            "source_span_sec": row.get("source_span_sec", ""),
            "replay_span_sec": row.get("replay_span_sec", ""),
            "trace_min_mib": row.get("min_mib", ""),
            "trace_mean_mib": row.get("mean_mib", ""),
            "trace_max_mib": row.get("max_mib", ""),
        })
    if "" in gate_hashes or len(gate_hashes) != 1:
        raise ValueError(
            "cross-artifact 32-token gate mismatch: "
            f"{sorted(gate_hashes)}")

    token_counts = {
        label: token_sequence(path)[0]
        for label, path in token_paths.items()
    }
    common_token_count = min(token_counts.values(), default=0)
    if common_token_count <= 0:
        raise ValueError("no common decode-token prefix")
    common_hashes = {
        label: token_sequence(
            path, common_token_count)[1]
        for label, path in token_paths.items()
    }
    unique_common_hashes = set(common_hashes.values())
    if None in unique_common_hashes or len(unique_common_hashes) != 1:
        raise ValueError(
            "longest-common-prefix token mismatch: "
            f"{common_hashes}")
    common_token_hash = next(iter(unique_common_hashes))
    source_window_hashes = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in source_window_paths.items()
    }
    if (
        len(set(source_window_hashes.values())) != 1
        or len(trace_metadata) != 1
    ):
        raise ValueError(
            "methods do not derive from one identical absolute-budget "
            f"source window: hashes={source_window_hashes} "
            f"metadata={sorted(trace_metadata)}")
    source_window_hash = next(iter(source_window_hashes.values()))
    original_source_hashes = {
        row.get("source_file_sha256", "")
        for row in selected.values()
    }
    if "" in original_source_hashes or len(original_source_hashes) != 1:
        raise ValueError(
            "methods do not share one original source trace SHA-256: "
            f"{sorted(original_source_hashes)}")
    original_source_hash = next(iter(original_source_hashes))
    historical_original_hashes = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in original_source_paths.items()
    }
    if (
        len(set(historical_original_hashes.values())) != 1
        or next(iter(historical_original_hashes.values()))
            != next(iter({
                row["original_source_trace_sha256"]
                for row in output_rows
            }))
    ):
        raise ValueError(
            "methods do not share the same historical original trace: "
            f"{historical_original_hashes}")
    historical_original_hash = next(
        iter(historical_original_hashes.values()))
    runtime_trace_hashes = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in runtime_trace_paths.items()
    }
    for label in ORDER:
        row = selected[label]
        if row.get("source_window_sha256") != source_window_hashes[label]:
            raise ValueError(
                f"{label}: recorded source-window hash does not match file")
        if (
            row.get("runtime_trace_sha256")
            != runtime_trace_hashes[label]
        ):
            raise ValueError(
                f"{label}: recorded runtime-trace hash does not match file")
        points = budget_trace_points(runtime_trace_paths[label])
        if abs(points[0][0]) > 1e-6 or points[-1][0] < 599.999:
            raise ValueError(
                f"{label}: effective runtime trace does not cover 0--600 s")
        if label in {"Static-Min", "Static-Max"}:
            expected_kind = label.lower()
            expected_key = (
                "min_bucket_mib"
                if label == "Static-Min" else "max_bucket_mib")
            expected_budget = float(row.get(expected_key, "nan"))
            budgets = {budget for _, budget in points}
            if (
                row.get("runtime_trace_kind") != expected_kind
                or len(budgets) != 1
                or abs(next(iter(budgets)) - expected_budget) > 1e-6
                or abs(float(
                    row.get("runtime_trace_budget_mib", "nan"))
                    - expected_budget) > 1e-6
            ):
                raise ValueError(
                    f"{label}: effective trace is not the constant "
                    f"{expected_budget:g} MiB baseline")
        else:
            if row.get("runtime_trace_kind") != "dynamic":
                raise ValueError(
                    f"{label}: expected the dynamic runtime trace")
            if runtime_trace_hashes[label] != source_window_hashes[label]:
                raise ValueError(
                    f"{label}: runtime trace differs from the original "
                    "dynamic source window")
    trace_meta = next(iter(trace_metadata))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.backend}_full10min_baselines.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    fastest = min(
        output_rows, key=lambda row: float(row["exec_ms_per_token"]))
    rows_by_method = {
        row["method"]: row for row in output_rows
    }
    diff_now_latency = float(
        rows_by_method["Diff-now"]["exec_ms_per_token"])
    diff_now_speedup_pct = {
        method: (
            (float(rows_by_method[method]["exec_ms_per_token"]) -
             diff_now_latency) /
            float(rows_by_method[method]["exec_ms_per_token"]) * 100.0
        )
        for method in (
            "Static-Min", "Static-Max", "MRU", "Offline",
            "Online", "Diff-before",
        )
    }
    lines = [
        f"# {args.backend.upper()} original 10-minute source-trace baselines",
        "",
        "| Method | ms/token | I/O MiB/fwd | I/O ms/fwd | LOAD ms/fwd | "
        + (
            "Layout ms/fwd"
            if args.backend == "cpu" else "Prepare host ms/fwd"
        )
        + " | Compute ms/fwd | Pipeline wait ms/fwd | Missing units/fwd | "
        "Relaxed plan MiB/fwd | Apply ms/fwd | "
        "Final M/T/C | Tokens |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in output_rows:
        frontier = "/".join(
            str(row[f"frontier_{mode}_units_last"] or "-")
            for mode in ("multi", "tensor", "cut"))
        lines.append(
            f"| {row['method']} | "
            f"{float(row['exec_ms_per_token']):.3f} | "
            f"{format_metric(row['direct_read_mib_per_forward'])} | "
            f"{format_metric(row['direct_read_ms_per_forward'])} | "
            f"{format_metric(row['load_stage_ms_per_forward'])} | "
            f"{format_metric(
                row['layout_prepare_ms_per_forward']
                if args.backend == 'cpu'
                else row['prepare_host_issue_ms_per_forward'])} | "
            f"{format_metric(row['compute_ms_per_forward'])} | "
            f"{format_metric(row['pipeline_wait_ms_per_forward'])} | "
            f"{format_metric(row['nonresident_units_per_forward'])} | "
            f"{format_metric(row['plan_protection_relaxed_mib_per_forward'])} | "
            f"{format_metric(row['plan_apply_ms_per_forward'])} | "
            f"{frontier} | "
            f"{row['generated_tokens']} |"
        )
    lines.extend([
        "",
        "## Mixed-granularity planning and transition audit",
        "",
        "| Method | Max M/T/C units | Mixed decisions | Frontier state "
        "changes | Transition changed weights total/max | Predicted switch ms "
        "total/max | Measured changed-frontier publish ms total/max | "
        "Full plan apply ms/fwd/max |",
        "|:--|:--|--:|--:|:--|:--|:--|--:|",
    ])
    for row in output_rows:
        frontier_max = "/".join(
            str(row[f"frontier_{mode}_units_max"] or "-")
            for mode in ("multi", "tensor", "cut"))
        changed_weights = (
            f"{row['frontier_transition_delta_weights_total'] or 0}/"
            f"{row['frontier_transition_delta_weights_max'] or 0}")
        predicted_switch = (
            f"{format_metric(row['frontier_switch_ms_total'])}/"
            f"{format_metric(row['frontier_switch_ms_max'])}")
        measured_publish = (
            f"{format_metric(row['frontier_transition_publish_ms_total'])}/"
            f"{format_metric(row['frontier_transition_publish_ms_max'])}")
        lines.append(
            f"| {row['method']} | {frontier_max} | "
            f"{row['frontier_mixed_mode_generations'] or 0} | "
            f"{row['frontier_state_changes'] or 0} | "
            f"{changed_weights} | {predicted_switch} | "
            f"{measured_publish} | "
            f"{format_metric(row['plan_apply_ms_per_forward'])}/"
            f"{format_metric(row['plan_apply_ms_max'])} |")
    lines.extend([
        "",
        f"Fastest: **{fastest['method']}** at "
        f"{float(fastest['exec_ms_per_token']):.3f} ms/token.",
        "",
        "Diff-now speedup (positive means Diff-now is faster): "
        + ", ".join(
            f"`{method} {speedup:+.2f}%`"
            for method, speedup in diff_now_speedup_pct.items()
        )
        + ".",
        "",
        "All rows derive from the same original 0--600 second "
        "absolute-budget source trace, share "
        f"one {common_token_count}-decode-token common-prefix hash "
        f"(`{common_token_hash}`), and pass continuous device-isolation "
        "checks.",
        "",
        "Runtime trace contract: MRU, Offline, Online, Diff-before and "
        "Diff-now replay the dynamic source at 1x. Static-Min and Static-Max "
        f"hold its bucketed minimum/maximum at {float(trace_meta[7]):g}/"
        f"{float(trace_meta[8]):g} MiB for the same 600-second duration.",
        "",
        "Trace audit: source "
        f"`{trace_meta[0]}`, uncompressed span `{trace_meta[1]}–"
        f"{float(trace_meta[1]) + float(trace_meta[2]):g}` s, "
        f"budget min/mean/max `{float(trace_meta[4]):.1f}/"
        f"{float(trace_meta[5]):.3f}/{float(trace_meta[6]):.1f}` MiB, "
        f"historical raw SHA-256 `{historical_original_hash}`, normalized "
        f"host-input SHA-256 `{original_source_hash}`, and generated "
        f"source-window SHA-256 `{source_window_hash}`. Historical and "
        "generated traces are point-for-point equal.",
        "",
        "Pure I/O, full LOAD service, layout preparation, pipeline wait, and "
        "plan apply are measured decode-only backend-counter deltas from the "
        "first generated-token boundary. LOAD includes direct reads plus "
        "staging/pool service and is the planner's first pipeline stage; pure "
        "I/O is a diagnostic subcomponent. These are signals from an "
        "overlapped pipeline and are not additive latency components.",
        "",
        "The transition audit separates the planner-predicted split/merge "
        "switch cost, isolated working-unit frontier publication, and full "
        "plan-apply time. The calibrated switch cost comes from held-in "
        "fixed-node publication measurements, not the 600-second trace. "
        "`M/T/C` means logical "
        "Multi/Tensor/Cut units over the common physical two-tile "
        "representation; it does not change the bytes available to a method.",
        "",
        "The ranking metric is the independent decode critical-path wall "
        "timer (scheduler/provider/apply through logits synchronization). "
        "The standard eval timer plus provider/apply measurements are kept "
        "only as a containment audit.",
    ])
    if args.backend == "gpu":
        lines.extend([
            "",
            "GPU `Prepare host` is host-visible issue/enqueue time, not pure "
            "device layout-kernel time; production runs keep OpenCL event "
            "profiling disabled to preserve the measured overlap path.",
        ])
    (args.output_dir / f"{args.backend}_full10min_baselines.md").write_text(
        "\n".join(lines) + "\n")
    audit_path = args.output_dir / f"{args.backend}_full10min_audit.json"
    audit_path.write_text(json.dumps({
        "backend": args.backend,
        "common_decode_token_count": common_token_count,
        "common_decode_token_sha256": common_token_hash,
        "token_counts": token_counts,
        "gate_token_count": 32,
        "gate_token_sha256": next(iter(gate_hashes)),
        "trace": {
            "source": trace_meta[0],
            "source_start_sec": float(trace_meta[1]),
            "source_span_sec": float(trace_meta[2]),
            "replay_span_sec": float(trace_meta[3]),
            "min_mib": float(trace_meta[4]),
            "mean_mib": float(trace_meta[5]),
            "max_mib": float(trace_meta[6]),
            "min_bucket_mib": float(trace_meta[7]),
            "max_bucket_mib": float(trace_meta[8]),
            "source_window_sha256": source_window_hash,
            "normalized_source_input_sha256": original_source_hash,
            "legacy_source_file_sha256": original_source_hash,
            "historical_original_source_file_sha256":
                historical_original_hash,
            "per_method_historical_original_sha256":
                historical_original_hashes,
            "per_method_source_window_sha256": source_window_hashes,
            "per_method_runtime_trace_sha256": runtime_trace_hashes,
            "per_method_remote_runtime_trace_sha256": {
                row["method"]: row[
                    "remote_runtime_trace_sha256"]
                for row in output_rows
            },
            "per_method_runtime_trace_kind": {
                label: selected[label].get("runtime_trace_kind", "")
                for label in ORDER
            },
            "per_method_runtime_trace_budget_mib": {
                label: selected[label].get(
                    "runtime_trace_budget_mib", "")
                for label in ORDER
            },
        },
        "result_sources": {
            row["method"]: row["result_source"]
            for row in output_rows
        },
        "replay_clock": {
            row["method"]: {
                "decode_reset_count":
                    row["budget_decode_reset_count"],
                "decode_reset_generated":
                    row["budget_decode_reset_generated"],
                "backend_reset_count":
                    row["budget_backend_reset_count"],
                "decode_rows": row["token_trace_decode_rows"],
                "first_t_sec": row["token_trace_t_first_sec"],
                "last_t_sec": row["token_trace_t_last_sec"],
                "span_sec": row["token_trace_t_span_sec"],
                "first_budget_mib":
                    row["token_trace_first_budget_mib"],
                "last_budget_mib":
                    row["token_trace_last_budget_mib"],
                "max_decode_latency_ms":
                    row["token_trace_max_latency_ms"],
                "monotonic": row["token_trace_time_monotonic"],
                "budget_provider_valid":
                    row["token_trace_budget_provider_valid"],
            }
            for row in output_rows
        },
        "latency_clock": {
            row["method"]: {
                "ranking_source": row["exec_timing_source"],
                "decode_wall_ms_total": row["decode_wall_ms_total"],
                "decode_wall_runs": row["decode_wall_runs"],
                "decode_wall_ms_per_token":
                    row["exec_ms_per_token"],
                "component_sum_ms_per_token":
                    row["exec_component_sum_ms_per_token"],
                "wall_minus_component_sum_ms_total":
                    row["decode_wall_unattributed_ms_total"],
            }
            for row in output_rows
        },
        "phase_counter_window": {
            row["method"]: {
                "scope": row["decode_phase_counter_scope"],
                "runs": row["decode_phase_runs"],
                "contract_valid":
                    row["decode_phase_contract_valid"],
            }
            for row in output_rows
        },
        "fixed_performance_restore": {
            row["method"]: {
                "configured":
                    row["fixed_performance_mode_configured"],
                "attempted":
                    row["fixed_performance_restore_attempted"],
                "valid": row["fixed_performance_restore_valid"],
            }
            for row in output_rows
        },
        "runtime_input_lineage": {
            "dynamic_run_config_sha256": next(iter({
                row["dynamic_run_config_sha256"]
                for row in output_rows
            })),
            "dynamic_run_core_config_sha256": next(iter({
                row["dynamic_run_core_config_sha256"]
                for row in output_rows
            })),
            "model_sha256": next(iter({
                row["remote_model_sha256"] for row in output_rows
            })),
            "binary_sha256": next(iter({
                row["remote_binary_sha256"] for row in output_rows
            })),
            "model_meta_sha256": next(iter({
                row["remote_model_meta_sha256"]
                for row in output_rows
            })),
            "cost_dir_sha256": next(iter({
                row["remote_cost_dir_sha256"]
                for row in output_rows
            })),
            "per_method": {
                row["method"]: {
                    "model_sha256": row["remote_model_sha256"],
                    "binary_sha256": row["remote_binary_sha256"],
                    "model_meta_sha256":
                        row["remote_model_meta_sha256"],
                    "cost_dir_sha256":
                        row["remote_cost_dir_sha256"],
                }
                for row in output_rows
            },
        },
        "granularity_profile_lineage": {
            row["method"]: {
                "path": row["granularity_profile"],
                "sha256": row["granularity_profile_sha256"],
                "kind": row["granularity_profile_kind"],
                "placement_source":
                    row["granularity_placement_source"],
                "policy": row["granularity_policy_configured"],
                "effective_policy":
                    row["granularity_policy_effective"],
                "multi_implementation":
                    row["multi_implementation_configured"],
                "cut_compute_grouping":
                    row["cut_compute_grouping_configured"],
                "transition_fixed_ms":
                    row["transition_fixed_ms_configured"],
                "transition_cost_source":
                    row["transition_cost_source_configured"],
                "mixed_offline_table_sha256":
                    row["mixed_offline_table_sha256"],
                "offline_table_sha256":
                    row["offline_table_sha256"],
                "remote_offline_table_sha256":
                    row["remote_offline_table_sha256"],
                "remote_mixed_offline_table_sha256":
                    row["remote_mixed_offline_table_sha256"],
            }
            for row in output_rows
        },
        "physical_tiling": {
            row["method"]: {
                "cut_capable_tensors": row[
                    "physical_cut_tensors"],
                "physical_parts": row["physical_cut_parts"],
                "wbm_total_mib": row["wbm_total_mib"],
                "expected_wbm_total_mib": row[
                    "expected_wbm_total_mib_configured"],
                "planner_stream_reserve_mib": row[
                    "planner_stream_reserve_mib_configured"],
            }
            for row in output_rows
        },
        "runtime_status": {
            row["method"]: {
                "status": row["status"],
                "rc": int(float(row["rc"])),
                "bench_exit_reason": row["bench_exit_reason"],
                "bench_time_done": int(float(row["bench_time_done"])),
            }
            for row in output_rows
        },
        "measurement_isolation": {
            row["method"]: {
                "valid": str(
                    row["valid_measurement_isolation"]
                ).lower() == "true",
                "monitor_samples": int(float(
                    row["measurement_monitor_samples"] or 0)),
                "monitor_failures": int(float(
                    row["measurement_monitor_failures"] or 0)),
                "measured_llama_pid":
                    row["measurement_llama_pid"],
                "monitor_thread_incomplete": str(
                    row["monitor_thread_incomplete"]
                ).lower() == "true",
                "overlap": row["measurement_overlap"],
            }
            for row in output_rows
        },
        "thermal_and_battery": {
            row["method"]: {
                "thermal_status_before":
                    row["thermal_status_before"],
                "thermal_cpu_max_c_before":
                    row["thermal_cpu_max_c_before"],
                "thermal_gpu_max_c_before":
                    row["thermal_gpu_max_c_before"],
                "thermal_skin_max_c_before":
                    row["thermal_skin_max_c_before"],
                "thermal_status_after":
                    row["thermal_status_after"],
                "thermal_cpu_max_c_after":
                    row["thermal_cpu_max_c_after"],
                "thermal_gpu_max_c_after":
                    row["thermal_gpu_max_c_after"],
                "thermal_skin_max_c_after":
                    row["thermal_skin_max_c_after"],
                "battery_level_pct_before":
                    row["battery_level_pct_before"],
                "battery_level_pct_after":
                    row["battery_level_pct_after"],
                "cooldown_thermal_max_c_configured":
                    row["cooldown_thermal_max_c_configured"],
                "cooldown_thermal_status_max_configured":
                    row[
                        "cooldown_thermal_status_max_configured"],
                "min_battery_level_pct_configured":
                    row["min_battery_level_pct_configured"],
            }
            for row in output_rows
        },
        "cpu_frequency": {
            row["method"]: {
                "limit_min_khz":
                    row["cpu_freq_limit_min_khz"],
                "mean_khz": row["cpu_freq_mean_khz"],
                "median_sample_min_khz":
                    row["cpu_freq_median_sample_min_khz"],
                "required_limit_min_khz":
                    row["min_cpu_freq_limit_khz_configured"],
                "required_mean_khz":
                    row["min_cpu_mean_freq_khz_configured"],
                "required_median_sample_min_khz":
                    row[
                        "min_cpu_median_sample_min_khz_configured"],
            }
            for row in output_rows
        },
        "pipeline_budget_violations": {
            row["method"]: int(float(
                row["pipeline_budget_violations"]))
            for row in output_rows
        },
        "pipeline_plan_protection_relaxations": {
            row["method"]: int(float(
                row["plan_protection_relaxations"]))
            for row in output_rows
        },
        "frontier_contract_valid": {
            row["method"]: str(
                row.get("frontier_contract_valid", "")
            ).lower() == "true"
            for row in output_rows
        },
        "unit_pipeline_stage_authority": {
            row["method"]: {
                "valid": str(row.get(
                    "unit_pipeline_stage_authority_contract_valid",
                    "")).lower() == "true",
                "full_plan_applies": int(float(
                    row["plan_stage_defer_apply_count"] or 0)),
                "stage_defer_min": int(float(
                    row["plan_stage_defer_min"] or 0)),
                "stage_defer_max": int(float(
                    row["plan_stage_defer_max"] or 0)),
                "anchor_fired": int(float(
                    row["plan_anchor_fired"] or 0)),
                "load_anchors": int(float(
                    row["plan_anchor_load_events"] or 0)),
                "transfer_anchors": int(float(
                    row["plan_anchor_transfer_events"] or 0)),
                "prepare_anchors": int(float(
                    row["plan_anchor_xform_events"] or 0)),
            }
            for row in output_rows
        },
        "backend_compute_contract": {
            row["method"]: {
                "valid": str(
                    row["backend_compute_contract_valid"]
                ).lower() == "true",
                "cut_dual_candidates": int(float(
                    row["cut_dual_candidates"] or 0)),
                "cut_dual_calls": int(float(
                    row["cut_dual_calls"] or 0)),
                "cut_dual_budget_fallbacks": int(float(
                    row["cut_dual_budget_fallbacks"] or 0)),
                "cut_dual_shape_fallbacks": int(float(
                    row["cut_dual_shape_fallbacks"] or 0)),
                "cut_dual_queue_errors": int(float(
                    row["cut_dual_queue_errors"] or 0)),
                "cut_dual_image_creates": int(float(
                    row["cut_dual_image_creates"] or 0)),
                "cut_dual_image_cache_hits": int(float(
                    row["cut_dual_image_cache_hits"] or 0)),
                "cut_dual_image_releases": int(float(
                    row["cut_dual_image_releases"] or 0)),
                "cut_dual_image_errors": int(float(
                    row["cut_dual_image_errors"] or 0)),
            }
            for row in output_rows
        },
        "remote_solver_isolation": {
            row["method"]: {
                "fresh_process": str(
                    row["remote_solver_fresh_process"]
                ).lower() == "true",
                "contract_valid": str(
                    row["remote_solver_contract_valid"]
                ).lower() == "true",
                "event_count": int(float(
                    row["remote_solver_event_count"] or 0)),
                "run_ids": row["remote_solver_run_ids"],
                "working_unit_policies":
                    row["remote_solver_working_unit_policies"],
                "placement_sources":
                    row["remote_solver_placement_sources"],
                "diff_edit_count_max": int(float(
                    row[
                        "remote_solver_diff_edit_count_max"] or 0)),
                "log": row["remote_solver_log"],
            }
            for row in output_rows
        },
        "frontier_units_max": {
            row["method"]: {
                mode: int(float(
                    row.get(f"frontier_{mode}_units_max", 0) or 0))
                for mode in ("multi", "tensor", "cut")
            }
            for row in output_rows
        },
        "frontier_mixed_mode_generations": {
            row["method"]: int(float(
                row["frontier_mixed_mode_generations"] or 0))
            for row in output_rows
        },
        "frontier_decision_trace": {
            row["method"]: {
                "sha256": row["frontier_trace_sha256"],
                "transition_path_sha256":
                    row["frontier_transition_trace_sha256"],
                "shape_sha256": row[
                    "frontier_shape_trace_sha256"],
                "unique_states": int(float(
                    row["frontier_unique_states"] or 0)),
                "unique_shape_states": int(float(
                    row["frontier_unique_shape_states"] or 0)),
                "state_changes": int(float(
                    row["frontier_state_changes"] or 0)),
                "shape_state_changes": int(float(
                    row["frontier_shape_state_changes"] or 0)),
            }
            for row in output_rows
        },
        "frontier_delta_weights": {
            row["method"]: {
                "total": int(float(
                    row["frontier_delta_weights_total"] or 0)),
                "max_per_apply": int(float(
                    row["frontier_delta_weights_max"] or 0)),
                "frontier_only_apply_count": int(float(
                    row["frontier_only_apply_count"] or 0)),
                "transition_total": int(float(
                    row["frontier_transition_delta_weights_total"] or 0)),
                "transition_max_per_apply": int(float(
                    row["frontier_transition_delta_weights_max"] or 0)),
            }
            for row in output_rows
        },
        "frontier_publication": {
            row["method"]: {
                "predicted_switch_ms_total": float(
                    row["frontier_switch_ms_total"] or 0.0),
                "predicted_switch_ms_max": float(
                    row["frontier_switch_ms_max"] or 0.0),
                "measured_publish_ms_total": float(
                    row["frontier_publish_ms_total"] or 0.0),
                "measured_publish_ms_max": float(
                    row["frontier_publish_ms_max"] or 0.0),
                "measured_transition_publish_count": int(float(
                    row["frontier_transition_publish_count"] or 0)),
                "measured_transition_publish_ms_total": float(
                    row["frontier_transition_publish_ms_total"] or 0.0),
                "measured_transition_publish_ms_max": float(
                    row["frontier_transition_publish_ms_max"] or 0.0),
                "calibrated_transition_fixed_ms": float(
                    row["transition_fixed_ms_configured"] or 0.0),
                "calibration_source":
                    row["transition_cost_source_configured"],
            }
            for row in output_rows
        },
        "frontier_switch_cost": {
            row["method"]: {
                "predicted_total_ms": float(
                    row["frontier_switch_ms_total"] or 0.0),
                "predicted_max_ms": float(
                    row["frontier_switch_ms_max"] or 0.0),
                "actual_plan_apply_ms_per_forward": float(
                    row["plan_apply_ms_per_forward"] or 0.0),
                "actual_plan_apply_ms_max": float(
                    row["plan_apply_ms_max"] or 0.0),
                "provider_get_ms_max": float(
                    row["provider_get_ms_max"] or 0.0),
                "remote_wall_ms_max": float(
                    row["remote_wall_ms_max"] or 0.0),
                "remote_server_ms_max": float(
                    row["remote_server_ms_max"] or 0.0),
            }
            for row in output_rows
        },
        "diff_now_speedup_pct": diff_now_speedup_pct,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "csv": str(csv_path),
        "markdown": str(
            args.output_dir / f"{args.backend}_full10min_baselines.md"),
        "audit": str(audit_path),
    }, indent=2))


if __name__ == "__main__":
    main()
