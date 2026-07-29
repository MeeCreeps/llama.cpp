#!/usr/bin/env python3
"""Audit OP12 plan-residency and streaming-headroom screening runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from runtime.plan.run_dynamic_budget_matrix import parse_log, token_sequence
except ModuleNotFoundError:
    from run_dynamic_budget_matrix import parse_log, token_sequence  # type: ignore


POLICIES = (
    "strict",
    "reuse-stream",
    "cache-managed",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value.lower() == "true"


def row_contract_failures(
    raw: dict[str, str],
    parsed: dict[str, Any],
) -> list[str]:
    """Independently reject an invalid residency-screening measurement."""
    failures: list[str] = []
    if raw.get("status") != "ok" or raw.get("rc") != "0":
        failures.append("runtime")
    for field, label in (
        ("valid_token_trace", "token-trace"),
        (
            "token_trace_timing_contract_valid",
            "token-trace-timing-contract",
        ),
        ("decode_wall_contract_valid", "decode-wall-contract"),
        ("decode_phase_contract_valid", "decode-phase-contract"),
        ("valid_measurement_isolation", "isolation"),
        ("pin_contract_valid", "pin-contract"),
        ("fused_lane_contract_valid", "fused-lane-contract"),
        ("frontier_contract_valid", "frontier-contract"),
        (
            "unit_pipeline_stage_authority_contract_valid",
            "unit-pipeline-stage-authority-contract",
        ),
        ("backend_compute_contract_valid", "backend-compute-contract"),
        ("remote_solver_contract_valid", "remote-solver-contract"),
        ("timed_run_contract_valid", "timed-run-contract"),
    ):
        if not truth(raw.get(field, "")):
            failures.append(label)
    if int(float(raw.get("measurement_monitor_samples") or 0)) <= 0:
        failures.append("isolation-samples")
    if int(float(raw.get("measurement_monitor_failures") or 0)) != 0:
        failures.append("isolation-monitor-failure")
    if not str(raw.get("measurement_llama_pid") or "").strip():
        failures.append("isolation-process-identity")
    if truth(raw.get("monitor_thread_incomplete", "false")):
        failures.append("isolation-monitor-thread")
    if int(float(raw.get("pipeline_budget_violations") or -1)) != 0:
        failures.append("budget")
    plan_relaxations = str(
        raw.get("pipeline_plan_protection_relaxations") or "").strip()
    if not plan_relaxations:
        failures.append("plan-protection-audit-missing")
    elif int(float(plan_relaxations)) != 0:
        failures.append("plan-protection-relaxation")
    if int(float(raw.get("pin_outside_budget_count") or 0)) != 0:
        failures.append("outside-budget-pin")
    physical_tensors = int(float(
        raw.get("physical_cut_tensors")
        or parsed.get("physical_cut_tensors")
        or 0))
    physical_parts = int(float(
        raw.get("physical_cut_parts")
        or parsed.get("physical_cut_parts")
        or 0))
    if physical_tensors <= 0:
        failures.append("physical-tiling-audit-missing")
    elif physical_parts != 2 * physical_tensors:
        failures.append("physical-tiling-mismatch")
    observed_total_mib = float(
        raw.get("wbm_total_mib")
        or parsed.get("wbm_total_mib")
        or 0.0)
    expected_total_mib = float(
        raw.get("expected_wbm_total_mib_configured") or 0.0)
    if (
        expected_total_mib <= 0.0
        or abs(observed_total_mib - expected_total_mib) > max(
            0.01, expected_total_mib * 1e-5)
    ):
        failures.append("physical-wbm-total")
    if float(raw.get("source_span_sec") or 0) < 119.999:
        failures.append("trace-coverage")
    if float(raw.get("replay_span_sec") or 0) < 119.999:
        failures.append("replay-coverage")
    if abs(float(raw.get("source_start_sec") or -1.0)) > 1e-6:
        failures.append("trace-start")
    runtime_trace_hash = str(
        raw.get("runtime_trace_sha256") or "").strip()
    if (
        not runtime_trace_hash
        or str(raw.get("remote_runtime_trace_sha256") or "").strip()
            != runtime_trace_hash
    ):
        failures.append("runtime-trace-lineage")
    for label in ("model", "binary"):
        expected = str(raw.get(f"expected_{label}_sha256") or "").strip()
        remote = str(raw.get(f"remote_{label}_sha256") or "").strip()
        if not expected or remote != expected:
            failures.append(f"{label}-lineage")
    for label in ("model_meta", "cost_dir"):
        local = str(raw.get(f"{label}_sha256") or "").strip()
        remote = str(raw.get(f"remote_{label}_sha256") or "").strip()
        if not local or remote != local:
            failures.append(f"{label}-lineage")
    if not str(raw.get("dynamic_run_config_sha256") or "").strip():
        failures.append("runtime-config-lineage")
    if not str(
            raw.get("dynamic_run_core_config_sha256") or "").strip():
        failures.append("runtime-core-config-lineage")
    return failures


def common_candidate_audit(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Prove that residency policy is the only screened runtime difference."""
    if len(rows) != len(POLICIES):
        raise ValueError(
            f"expected {len(POLICIES)} candidate rows, got {len(rows)}")

    def one(field: str) -> str:
        values = {str(row.get(field) or "").strip() for row in rows}
        if "" in values or len(values) != 1:
            raise ValueError(
                f"residency candidates differ in {field}: "
                f"{sorted(values)}")
        return next(iter(values))

    audit = {
        "dynamic_run_core_config_sha256":
            one("dynamic_run_core_config_sha256"),
        "expected_model_sha256": one("expected_model_sha256"),
        "remote_model_sha256": one("remote_model_sha256"),
        "expected_binary_sha256": one("expected_binary_sha256"),
        "remote_binary_sha256": one("remote_binary_sha256"),
        "model_meta_sha256": one("model_meta_sha256"),
        "remote_model_meta_sha256": one("remote_model_meta_sha256"),
        "cost_dir_sha256": one("cost_dir_sha256"),
        "remote_cost_dir_sha256": one("remote_cost_dir_sha256"),
        "granularity_profile_sha256":
            one("granularity_profile_sha256"),
        "multi_implementation_configured":
            one("multi_implementation_configured"),
        "mixed_offline_table_sha256":
            one("mixed_offline_table_sha256"),
        "source_file_sha256": one("source_file_sha256"),
        "expected_source_trace_sha256":
            one("expected_source_trace_sha256"),
        "source_window_sha256": one("source_window_sha256"),
        "planner_stream_reserve_mib_configured":
            one("planner_stream_reserve_mib_configured"),
        "expected_wbm_total_mib_configured":
            one("expected_wbm_total_mib_configured"),
    }
    if audit["remote_model_sha256"] != audit["expected_model_sha256"]:
        raise ValueError("candidate model lineage is not verified")
    if audit["remote_binary_sha256"] != audit["expected_binary_sha256"]:
        raise ValueError("candidate binary lineage is not verified")
    if audit["remote_model_meta_sha256"] != audit["model_meta_sha256"]:
        raise ValueError("candidate model-meta lineage is not verified")
    if audit["remote_cost_dir_sha256"] != audit["cost_dir_sha256"]:
        raise ValueError("candidate cost-dir lineage is not verified")
    if (
        audit["source_file_sha256"]
        != audit["expected_source_trace_sha256"]
    ):
        raise ValueError("candidate source-trace lineage is not verified")
    full_config_hashes = {
        str(row.get("dynamic_run_config_sha256") or "").strip()
        for row in rows
    }
    if "" in full_config_hashes or len(full_config_hashes) != len(POLICIES):
        raise ValueError(
            "each residency policy must have one distinct full runtime "
            f"configuration hash, got {sorted(full_config_hashes)}")
    physical = {
        (
            int(float(row.get("physical_cut_tensors") or 0)),
            int(float(row.get("physical_cut_parts") or 0)),
            round(float(row.get("wbm_total_mib") or 0.0), 6),
        )
        for row in rows
    }
    if len(physical) != 1:
        raise ValueError(
            f"residency candidates used different physical tiling: "
            f"{sorted(physical)}")
    tensors, parts, total_mib = next(iter(physical))
    expected_total_mib = float(
        audit["expected_wbm_total_mib_configured"])
    if (
        tensors <= 0 or parts != 2 * tensors or total_mib <= 0.0
        or expected_total_mib <= 0.0
        or abs(total_mib - expected_total_mib) > max(
            0.01, expected_total_mib * 1e-5)
    ):
        raise ValueError(
            "residency candidates lack a valid common physical "
            f"representation: {(tensors, parts, total_mib)}")
    audit["physical_cut_tensors"] = tensors
    audit["physical_cut_parts"] = parts
    audit["wbm_total_mib"] = total_mib
    stream_reserve_mib = float(
        audit["planner_stream_reserve_mib_configured"])
    if stream_reserve_mib <= 0.0:
        raise ValueError(
            "residency candidates lack a positive common planner "
            "stream reserve")
    audit["planner_stream_reserve_mib"] = stream_reserve_mib
    audit["candidate_full_config_sha256"] = sorted(full_config_hashes)
    return audit


def planned_nonresident_mib(
    row: dict[str, str],
    artifact: Path,
) -> float:
    token_rows = read_rows(Path(row["token_trace"]))
    plans: dict[int, float] = {}
    for path in (artifact / "offline_mixed_table").glob(
            "plan_*MiB.json"):
        budget = int(path.stem.split("_")[1][:-3])
        plan = json.loads(path.read_text(encoding="utf-8"))
        plans[budget] = sum(
            int(weight.get("byte_size", 0))
            for weight in plan.get("weights", [])
            if weight.get("location") == "disk"
        ) / (1024.0 * 1024.0)
    if not token_rows or not plans:
        return 0.0
    minimum, maximum = min(plans), max(plans)
    values = []
    for token in token_rows:
        budget = int(float(token["budget_mib"]) // 128) * 128
        budget = max(minimum, min(maximum, budget))
        values.append(plans[budget])
    return statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()

    selected: list[dict[str, Any]] = []
    candidate_audit_rows: list[dict[str, str]] = []
    token_paths: dict[str, Path] = {}
    trace_hashes: set[str] = set()
    for policy in POLICIES:
        artifact = args.root / policy
        result_path = artifact / "summary" / "results.csv"
        for raw in read_rows(result_path):
            method = raw["method"]
            if method == "mru" and policy != "strict":
                continue
            log = Path(raw["log"])
            parsed = parse_log(log, method)
            label = "MRU" if method == "mru" else policy
            failures = row_contract_failures(raw, parsed)
            trace_path = artifact / "traces" / raw["trace"]
            trace_hashes.add(hashlib.sha256(
                trace_path.read_bytes()).hexdigest())
            token_paths[label] = Path(raw["token_trace"])
            forwards = float(parsed.get("decode_phase_runs") or 0)
            direct = float(
                parsed.get("decode_phase_direct_read_mib") or 0)
            relaxed = float(
                parsed.get(
                    "pipeline_plan_protection_relaxed_mib") or 0)
            predicted = (
                planned_nonresident_mib(raw, artifact)
                if method != "mru" else 0.0)
            relaxations = int(parsed.get(
                "pipeline_plan_protection_relaxations") or 0)
            if method != "mru":
                candidate_audit_rows.append({
                    **raw,
                    "physical_cut_tensors": str(
                        parsed.get("physical_cut_tensors") or
                        raw.get("physical_cut_tensors") or ""),
                    "physical_cut_parts": str(
                        parsed.get("physical_cut_parts") or
                        raw.get("physical_cut_parts") or ""),
                    "wbm_total_mib": str(
                        parsed.get("wbm_total_mib") or
                        raw.get("wbm_total_mib") or ""),
                })
            selected.append({
                "method": label,
                "policy": (
                    "native-mru" if method == "mru" else policy),
                "valid": not failures,
                "failures": ",".join(failures),
                "exec_ms_per_token": float(
                    raw["exec_ms_per_token"]),
                "generated_tokens": int(raw["generated_tokens"]),
                "direct_read_mib_per_forward":
                    direct / forwards if forwards else 0.0,
                "planned_nonresident_mib_per_forward": predicted,
                "excess_direct_mib_per_forward": (
                    direct / forwards - predicted
                    if forwards and method != "mru" else 0.0),
                "plan_protection_relaxations": relaxations,
                "plan_protection_relaxed_mib_per_forward":
                    relaxed / forwards if forwards else 0.0,
                "pipeline_wait_ms_per_forward": (
                    float(parsed.get(
                        "decode_phase_pipeline_wait_ms") or 0) /
                    forwards if forwards else 0.0),
                "layout_prepare_ms_per_forward": (
                    float(parsed.get("decode_phase_prepare_ms") or 0) /
                    forwards if forwards else 0.0),
                "token_trace": str(raw["token_trace"]),
                "log": str(log),
            })

    if len(trace_hashes) != 1:
        raise ValueError(
            f"pilot methods used different traces: {trace_hashes}")
    counts = {
        label: token_sequence(path)[0]
        for label, path in token_paths.items()
    }
    common_count = min(counts.values())
    hashes = {
        label: token_sequence(path, common_count)[1]
        for label, path in token_paths.items()
    }
    if len(set(hashes.values())) != 1:
        raise ValueError(f"token mismatch: {hashes}")
    common_audit = common_candidate_audit(candidate_audit_rows)
    candidates = [
        row for row in selected
        if row["valid"] and row["method"] != "MRU"
    ]
    if len(candidates) != len(POLICIES):
        raise ValueError("one or more residency candidates are invalid")
    best = min(candidates, key=lambda row: row["exec_ms_per_token"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                key for key in selected[0]
                if key not in {"token_trace", "log"}
            ],
        )
        writer.writeheader()
        writer.writerows({
            key: value for key, value in row.items()
            if key not in {"token_trace", "log"}
        } for row in selected)
    lines = [
        f"# OP12 {args.backend.upper()} plan-residency pilot",
        "",
        "| Method | ms/token | I/O MiB/fwd | Planned nonresident "
        "MiB/fwd | Excess I/O MiB/fwd | Relaxed plan MiB/fwd | "
        "Wait ms/fwd | Layout ms/fwd |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['method']} | {row['exec_ms_per_token']:.3f} | "
            f"{row['direct_read_mib_per_forward']:.3f} | "
            f"{row['planned_nonresident_mib_per_forward']:.3f} | "
            f"{row['excess_direct_mib_per_forward']:.3f} | "
            f"{row['plan_protection_relaxed_mib_per_forward']:.3f} | "
            f"{row['pipeline_wait_ms_per_forward']:.3f} | "
            f"{row['layout_prepare_ms_per_forward']:.3f} |")
    lines.extend([
        "",
        f"Selected candidate: **{best['policy']}** at "
        f"{best['exec_ms_per_token']:.3f} ms/token.",
        "",
        f"All rows share a {common_count}-token common-prefix hash and "
        "the same 120-second four-node trace. Every policy reserves the "
        "measured 16 MiB streaming window inside the unchanged physical "
        "budget, uses the same policy-excluded runtime-core hash, and reports "
        f"{common_audit['physical_cut_parts']} common physical tiles over "
        f"{common_audit['physical_cut_tensors']} cut-capable tensors at "
        f"{common_audit['wbm_total_mib']:.3f} MiB total. Any "
        "plan-protection relaxation invalidates a candidate. "
        "This is a screening result, not a final trace claim.",
    ])
    (args.output_dir / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "selection.json").write_text(
        json.dumps({
            "selected_policy": best["policy"],
            "selected_exec_ms_per_token":
                best["exec_ms_per_token"],
            "planner_stream_reserve_mib":
                common_audit["planner_stream_reserve_mib"],
            "common_token_count": common_count,
            "common_token_sha256": next(iter(hashes.values())),
            "trace_sha256": next(iter(trace_hashes)),
            "common_candidate_audit": common_audit,
            "rows": selected,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
