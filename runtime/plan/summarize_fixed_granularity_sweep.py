#!/usr/bin/env python3
"""Audit and summarize fixed-budget production-pipeline granularity runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable

try:
    from runtime.plan.calibrate_granularity_profile import (
        number,
        require_calibration_matrix,
        require_input_lineage,
        require_physical_tiling_contract,
        select_multi_implementation,
        valid_row,
    )
except ModuleNotFoundError:
    from calibrate_granularity_profile import (  # type: ignore
        number,
        require_calibration_matrix,
        require_input_lineage,
        require_physical_tiling_contract,
        select_multi_implementation,
        valid_row,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def median_metric(
    rows: list[dict[str, str]],
    key: str,
    transform: Callable[[float, dict[str, str]], float] | None = None,
) -> float | str:
    values: list[float] = []
    for row in rows:
        value = number(row, key)
        if value is None:
            continue
        values.append(transform(value, row) if transform else value)
    return statistics.median(values) if values else ""


def per_decode_phase(value: float, row: dict[str, str]) -> float:
    runs = number(row, "decode_phase_runs") or 0.0
    return value / runs if runs > 0.0 else 0.0


def summarize_rows(
    rows: list[dict[str, str]],
    *,
    backend: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(
            (int(float(row["ratio_pct"])), row["mode"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (ratio, mode), group in sorted(groups.items()):
        output.append({
            "backend": backend,
            "ratio_pct": ratio,
            "mode": mode,
            "samples": len(group),
            "steady_median_latency_ms": median_metric(
                group, "steady_median_latency_ms"),
            "weight_target_mib": median_metric(group, "weight_target_mib"),
            "wbm_total_mib": median_metric(group, "wbm_total_mib"),
            "direct_read_mib_per_forward": median_metric(
                group, "decode_phase_direct_read_mib",
                per_decode_phase),
            "direct_read_ms_per_forward": median_metric(
                group, "decode_phase_direct_read_ms",
                per_decode_phase),
            "load_stage_ms_per_forward": median_metric(
                group, "decode_phase_load_ms",
                per_decode_phase),
            "nonresident_units_per_forward": median_metric(
                group, "decode_phase_nonresident_units",
                per_decode_phase),
            "layout_prepare_ms_per_forward": (
                median_metric(
                    group, "decode_phase_prepare_ms",
                    per_decode_phase)
                if backend == "cpu" else ""),
            "prepare_host_issue_ms_per_forward": (
                median_metric(
                    group, "decode_phase_prepare_ms",
                    per_decode_phase)
                if backend == "gpu" else ""),
            "compute_ms_per_forward": (
                median_metric(
                    group, "decode_phase_compute_ms",
                    per_decode_phase)
                if backend == "cpu" else ""),
            "pipeline_wait_ms_per_forward": median_metric(
                group, "decode_phase_pipeline_wait_ms",
                per_decode_phase),
            "pipeline_residency_ms_per_forward": median_metric(
                group, "decode_phase_pipeline_residency_ms",
                per_decode_phase),
            "unit_count": median_metric(group, "unit_count"),
            "cut_op_count": median_metric(group, "cut_op_count"),
            "peak_unit_mib": median_metric(group, "peak_unit_mib"),
            "physical_cut_tensors": median_metric(
                group, "physical_cut_tensors"),
            "physical_cut_parts": median_metric(
                group, "physical_cut_parts"),
            "cut_dual_calls": median_metric(group, "cut_dual_calls"),
            "cut_dual_image_creates": median_metric(
                group, "cut_dual_image_creates"),
            "cut_dual_image_cache_hits": median_metric(
                group, "cut_dual_image_cache_hits"),
            "cut_dual_image_releases": median_metric(
                group, "cut_dual_image_releases"),
            "cut_dual_image_errors": median_metric(
                group, "cut_dual_image_errors"),
            "pin_token_embd_mib": median_metric(
                group, "pin_token_embd_inside_budget_mib"),
            "pin_output_mib": median_metric(
                group, "pin_output_inside_budget_mib"),
            "pipeline_budget_violations": max(
                int(float(row["pipeline_budget_violations"]))
                for row in group),
            "plan_protection_relaxations": max(
                int(float(
                    row["pipeline_plan_protection_relaxations"]))
                for row in group),
        })
    return output


def summarize_granularity_schemes(
    rows: list[dict[str, str]],
    *,
    backend: str,
) -> tuple[list[dict[str, Any]], str, dict[str, object]]:
    """Collapse Multi implementation ablations into three planner schemes."""
    normalized, implementation, selection = select_multi_implementation(
        rows)
    return (
        summarize_rows(normalized, backend=backend),
        implementation,
        selection,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "N/A" if value in ("", None) else f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-ratios", required=True)
    parser.add_argument("--expected-modes", required=True)
    args = parser.parse_args()

    ratios = [
        int(value.strip())
        for value in args.expected_ratios.split(",") if value.strip()]
    modes = [
        value.strip()
        for value in args.expected_modes.split(",") if value.strip()]
    raw_rows = read_rows(args.runs_csv)
    rows = [row for row in raw_rows if valid_row(row, args.backend)]
    require_calibration_matrix(rows, ratios=ratios, modes=modes)
    binary_sha256, model_sha256 = require_input_lineage(rows)
    physical_tiling_audit = require_physical_tiling_contract(rows)
    fixed_run_config_sha256 = next(iter({
        row["fixed_run_config_sha256"] for row in rows
    }))
    for ratio in ratios:
        physical_contracts = {
            (
                int(float(row["physical_cut_tensors"])),
                int(float(row["physical_cut_parts"])),
                round(float(row["wbm_total_mib"]), 3),
            )
            for row in rows
            if int(float(row["ratio_pct"])) == ratio
        }
        if len(physical_contracts) != 1:
            raise ValueError(
                f"{ratio}% modes do not share one physical tiling/WBM "
                f"contract: {sorted(physical_contracts)}")

    token_contract = {
        (int(float(row.get("token_id_count", "0"))),
         row.get("token_ids_sha256", ""))
        for row in rows
    }
    if (
        len(token_contract) != 1
        or next(iter(token_contract))[0] <= 0
        or not next(iter(token_contract))[1]
    ):
        raise ValueError(
            "fixed sweep does not share one complete decode-token sequence")

    # Multi and Multi-fused are two implementations of the same coarse
    # granularity, not two planner actions. Use the same global implementation
    # selection as profile calibration before reporting a three-granularity
    # winner. Keeping the raw four-way rows in the breakdown still exposes
    # whether fusion helped, without cherry-picking it independently at each
    # budget.
    summary = summarize_rows(rows, backend=args.backend)
    scheme_summary, multi_implementation, multi_selection = (
        summarize_granularity_schemes(rows, backend=args.backend))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fixed_pipeline_breakdown.csv"
    md_path = args.output_dir / "fixed_pipeline_breakdown.md"
    audit_path = args.output_dir / "fixed_pipeline_audit.json"
    write_csv(csv_path, summary)

    fastest = {
        ratio: min(
            (
                row for row in scheme_summary
                if row["ratio_pct"] == ratio
            ),
            key=lambda row: float(row["steady_median_latency_ms"]),
        )["mode"]
        for ratio in ratios
    }
    lines = [
        f"# {args.backend.upper()} fixed-budget granularity breakdown",
        "",
        "| Budget | Mode | Latency ms/token | I/O MiB/fwd | I/O ms/fwd | "
        "LOAD ms/fwd | "
        + (
            "Layout ms/fwd"
            if args.backend == "cpu" else "Prepare host ms/fwd"
        )
        + " | Compute ms/fwd | Wait ms/fwd | Missing units/fwd | Units | "
        "Peak unit MiB |",
        "|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['ratio_pct']}% | {row['mode']} | "
            f"{fmt(row['steady_median_latency_ms'])} | "
            f"{fmt(row['direct_read_mib_per_forward'])} | "
            f"{fmt(row['direct_read_ms_per_forward'])} | "
            f"{fmt(row['load_stage_ms_per_forward'])} | "
            f"{fmt(
                row['layout_prepare_ms_per_forward']
                if args.backend == 'cpu'
                else row['prepare_host_issue_ms_per_forward'])} | "
            f"{fmt(row['compute_ms_per_forward'])} | "
            f"{fmt(row['pipeline_wait_ms_per_forward'])} | "
            f"{fmt(row['nonresident_units_per_forward'])} | "
            f"{fmt(row['unit_count'])} | {fmt(row['peak_unit_mib'])} |")
    lines += [
        "",
        "Globally selected coarse implementation: "
        f"`{multi_implementation}`. Multi-fused is an implementation "
        "ablation of Multi, not a fourth granularity; this choice is fixed "
        "across all budgets.",
        "",
        "Fastest granularity by budget after that global selection: "
        + ", ".join(f"`{ratio}% {fastest[ratio]}`" for ratio in ratios)
        + ".",
        "",
        "Pure I/O, full LOAD service, layout preparation, compute, and "
        "pipeline wait are overlapping production-path signals and must not "
        "be added together. LOAD includes direct reads plus staging/pool "
        "service and is the planner's first pipeline stage; pure I/O remains "
        "a diagnostic subcomponent. Every phase column is a decode-only "
        "backend-counter delta divided by the exact decode run count; prompt "
        "ingestion is excluded.",
    ]
    if args.backend == "gpu":
        lines += [
            "",
            "GPU `Prepare host` is host-visible write/conversion enqueue "
            "overhead, not pure layout-kernel time. Pure device stages require "
            "a separate OpenCL-event profiling run.",
            "",
            "GPU compute is reported as N/A: intrusive OpenCL event profiling "
            "is disabled in the production-overlap run. End-to-end latency "
            "remains the authoritative GPU performance metric.",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    token_count, token_hash = next(iter(token_contract))
    audit_path.write_text(json.dumps({
        "backend": args.backend,
        "source": str(args.runs_csv),
        "required_ratios": ratios,
        "required_modes": modes,
        "valid_rows": len(rows),
        "complete_token_count": token_count,
        "complete_token_sha256": token_hash,
        "remote_binary_sha256": binary_sha256,
        "remote_model_sha256": model_sha256,
        "fixed_run_config_sha256": fixed_run_config_sha256,
        "latency_timing_source": next(iter({
            row["latency_timing_source"] for row in rows
        })),
        "decode_wall_contract_valid": all(
            row.get(
                "decode_wall_contract_valid", "").lower() == "true"
            for row in rows
        ),
        "decode_phase_contract_valid": all(
            row.get(
                "decode_phase_contract_valid", "").lower() == "true"
            for row in rows
        ),
        "phase_counter_scope": next(iter({
            row["decode_phase_counter_scope"] for row in rows
        })),
        "decode_wall_runs_by_cell": {
            f"{row.get('repeat', '0')}:{row['ratio_pct']}:{row['mode']}":
                int(float(row["decode_wall_runs"]))
            for row in rows
        },
        "physical_tiling": physical_tiling_audit,
        "multi_implementation_selection": multi_selection,
        "physical_tiling_by_ratio": {
            str(ratio): {
                "cut_capable_tensors": int(next(
                    row["physical_cut_tensors"]
                    for row in summary
                    if row["ratio_pct"] == ratio)),
                "physical_parts": int(next(
                    row["physical_cut_parts"]
                    for row in summary
                    if row["ratio_pct"] == ratio)),
                "wbm_total_mib": float(next(
                    row["wbm_total_mib"]
                    for row in summary
                    if row["ratio_pct"] == ratio)),
            }
            for ratio in ratios
        },
        "fastest_granularity_by_ratio": fastest,
        "budget_violations": {
            f"{row['ratio_pct']}:{row['mode']}":
                row["pipeline_budget_violations"]
            for row in summary
        },
        "plan_protection_relaxations": {
            f"{row['ratio_pct']}:{row['mode']}":
                row["plan_protection_relaxations"]
            for row in summary
        },
        "gpu_tiled_fused_compute": {
            f"{row['ratio_pct']}:{row['mode']}": {
                "dual_calls": row["cut_dual_calls"],
                "image_creates": row["cut_dual_image_creates"],
                "image_cache_hits": row["cut_dual_image_cache_hits"],
                "image_releases": row["cut_dual_image_releases"],
                "image_errors": row["cut_dual_image_errors"],
            }
            for row in summary
            if args.backend == "gpu"
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "csv": str(csv_path),
        "markdown": str(md_path),
        "audit": str(audit_path),
    }, indent=2))


if __name__ == "__main__":
    main()
