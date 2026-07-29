#!/usr/bin/env python3
"""Run real llama.cpp Elastic granularity experiments on one Android device.

The fixed sweep converts each requested weight-residency ratio into the total
budget consumed by BudgetWatcher:

    total budget = ceil(weight_bytes * ratio) + KV MiB + misc MiB

Each subprocess runs llama-cli through CPU_Elastic or OpenCL Elastic. Raw logs,
the exact budget traces, per-run rows, and median summaries are retained.
Dynamic mode additionally pulls GGML_ELASTIC_TOKEN_CSV and summarizes decode
latency at every observed budget point.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import threading
import time
from typing import Iterable


MODES = ("multi", "multi_fused", "tensor", "cut")
MODE_ORDERS = (
    ("multi", "multi_fused", "tensor", "cut"),
    ("cut", "tensor", "multi_fused", "multi"),
    ("tensor", "cut", "multi", "multi_fused"),
    ("multi_fused", "multi", "cut", "tensor"),
)

_ADB_PORTS = tuple(dict.fromkeys(
    int(value)
    for value in (
        os.environ.get("ANDROID_ADB_SERVER_PORT", ""),
        "5037",
        "5038",
    )
    if value.isdigit()
))
_ADB_PORT_BY_SERIAL: dict[str, int] = {}
_ADB_MISSING_PATTERNS = (
    "no devices/emulators found",
    "device offline",
    "device unauthorized",
)


def _adb_run(
    port: int,
    serial: str,
    args: tuple[str, ...],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["adb", "-P", str(port), "-s", serial, *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False,
    )


def adb(serial: str, *args: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run adb while tolerating the device moving between local adb servers.

    The lab host has independent adb servers on ports 5037 and 5038.  USB
    ownership can move between them without the Android device disconnecting.
    Retry only transport-level visibility failures; an ordinary remote-command
    failure is returned unchanged and a timed-out command is never replayed.
    """
    preferred = _ADB_PORT_BY_SERIAL.get(serial)
    ports = ([preferred] if preferred is not None else []) + [
        port for port in _ADB_PORTS if port != preferred
    ]
    last: subprocess.CompletedProcess[str] | None = None
    for port in ports:
        proc = _adb_run(port, serial, args, timeout)
        last = proc
        output = (proc.stdout or "").lower()
        missing = proc.returncode != 0 and any(
            pattern in output for pattern in _ADB_MISSING_PATTERNS)
        missing = missing or (
            proc.returncode != 0 and "device" in output and "not found" in output)
        if not missing:
            _ADB_PORT_BY_SERIAL[serial] = port
            return proc
    assert last is not None
    return last


def push(serial: str, local: Path, remote: str) -> None:
    proc = adb(serial, "push", str(local), remote)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)


def remote_sha256(serial: str, remote_path: str, timeout: int) -> str:
    proc = adb(
        serial,
        "shell",
        f"sha256sum {shlex.quote(remote_path)}",
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to hash remote input {remote_path}: {proc.stdout}")
    match = re.search(r"\b([0-9a-fA-F]{64})\b", proc.stdout or "")
    if match is None:
        raise RuntimeError(
            f"invalid remote SHA-256 output for {remote_path}: "
            f"{proc.stdout!r}")
    return match.group(1).lower()


def input_lineage(args: argparse.Namespace) -> dict[str, str]:
    return {
        "remote_binary_sha256": args.remote_binary_sha256,
        "remote_model_sha256": args.remote_model_sha256,
    }


FIXED_RUN_CONFIG_KEYS = (
    "serial", "backend", "remote_binary", "remote_model",
    "remote_lib_dir", "remote_work_dir", "weight_mib", "ratios",
    "repeats", "modes", "n_predict", "fixed_warmup_decode_tokens",
    "fixed_min_measure_tokens", "threads", "cpu_repack_threads",
    "cpu_mask", "cpu_strict", "poll", "context", "batch", "ubatch",
    "prompt", "seed", "ignore_eos", "kv_mib", "misc_mib", "pin",
    "expected_pin_token_embd_mib", "expected_pin_output_mib",
    "multi_tensors", "cut_parts", "cut_dual_compute", "gpu_unit_sync",
    "pipeline", "pipeline_lookahead", "pipeline_lookahead_mib",
    "pipeline_graph_lookahead", "pipeline_max_pending",
    "pipeline_max_pending_mib", "require_real_batch_io",
    "cpu_pipeline_staging", "cpu_staged_baseline",
    "pipeline_load_cpu", "pipeline_prepare_cpu", "pipeline_copy_cpu",
    "pipeline_copy_min_kib", "max_cpu_start_c", "max_gpu_start_c",
    "max_skin_start_c", "require_device_idle",
    "min_mem_available_mib", "min_battery_level_pct",
    "min_cpu_freq_limit_khz", "min_cpu_mean_freq_khz",
    "min_cpu_median_sample_min_khz", "max_compute_call_ms",
    "compute_recovery_ratio", "fixed_performance_mode",
    "keep_device_awake", "reboot_between_runs",
)


def fixed_run_config_sha256(args: argparse.Namespace) -> str:
    """Fingerprint every setting that can change a fixed-sweep result.

    Resume is intentionally conservative: a host-runner change or any
    execution, pipeline, placement-budget, affinity, or validity-gate change
    reruns and replaces the same run ids instead of mixing configurations in
    one calibration CSV.
    """
    payload = {
        key: getattr(args, key, None)
        for key in FIXED_RUN_CONFIG_KEYS
    }
    payload["host_runner_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def pin_policy_requests(policy: str, tensor_suffix: str) -> bool:
    tokens = {token.strip() for token in policy.split(",") if token.strip()}
    return "all" in tokens or tensor_suffix in tokens


def pinned_inside_budget(text: str, tensor_suffix: str) -> tuple[int, float]:
    matches = re.findall(
        rf"(?:ggml_opencl )?elastic: pin {re.escape(tensor_suffix)} "
        r"inside weight budget \(([0-9.]+) MiB",
        text,
    )
    return len(matches), sum(float(value) for value in matches)


def parse_log(text: str) -> dict[str, object]:
    reload_count = parse_int(r"reload calls:\s*([0-9]+)", text)
    if reload_count is None:
        reload_count = parse_int(r"reload_count\s*:\s*([0-9]+)", text)
    direct_read_ms = parse_float(r"direct O_DIRECT read.*?total=([0-9.]+) ms", text)
    direct_read_mib = parse_float(r"direct O_DIRECT read.*?MB=([0-9.]+)", text)
    if direct_read_ms is None:
        direct_read_ms = parse_float(r"direct_read\s*:.*?total=([0-9.]+) ms", text)
    if direct_read_mib is None:
        direct_read_mib = parse_float(r"direct_read\s*:.*?MB=([0-9.]+)", text)
    unit_count = parse_int(r"\[elastic granularity\].*?units=([0-9]+)", text)
    cut_op_count = parse_int(r"\[elastic granularity\].*?cut_ops=([0-9]+)", text)
    peak_unit_mib = parse_float(r"\[elastic granularity\].*?peak_unit=([0-9.]+) MiB", text)
    physical_cut_tensors = parse_int(
        r"\[elastic granularity\].*?cut_tensors=([0-9]+)", text)
    physical_cut_parts = parse_int(
        r"\[elastic granularity\].*?cut_parts=([0-9]+)", text)
    wbm_total_mib = parse_float(r"\[elastic granularity\].*?wbm_total=([0-9.]+) MiB", text)
    if unit_count is None:
        unit_count = parse_int(r"granularity\s*:.*?units=([0-9]+)", text)
        cut_op_count = parse_int(r"granularity\s*:.*?cut_ops=([0-9]+)", text)
        peak_unit_mib = parse_float(r"granularity\s*:.*?peak_unit=([0-9.]+) MiB", text)
        physical_cut_tensors = parse_int(
            r"granularity\s*:.*?cut_tensors=([0-9]+)", text)
        physical_cut_parts = parse_int(
            r"granularity\s*:.*?(?:cut_parts|parts)=([0-9]+)", text)
        wbm_total_mib = parse_float(r"granularity\s*:.*?wbm_total=([0-9.]+) MiB", text)
    pin_token_embd_count, pin_token_embd_mib = pinned_inside_budget(
        text, "token_embd")
    pin_output_count, pin_output_mib = pinned_inside_budget(text, "output")
    row: dict[str, object] = {
        "eval_total_ms": parse_float(r"(?m)^llama_perf_context_print:\s+eval time\s*=\s*([0-9.]+) ms", text),
        "eval_runs": parse_int(
            r"(?m)^llama_perf_context_print:\s+eval time\s*=.*?/\s*"
            r"([0-9]+)\s+runs",
            text),
        "eval_ms_per_token": parse_float(r"(?m)^llama_perf_context_print:\s+eval time\s*=.*?\(\s*([0-9.]+) ms per token", text),
        "decode_wall_ms_total": parse_float(
            r"elastic decode wall summary:\s+runs=[0-9]+\s+"
            r"wall_ms=([0-9.]+)",
            text),
        "decode_wall_runs": parse_int(
            r"elastic decode wall summary:\s+runs=([0-9]+)",
            text),
        "prompt_eval_total_ms": parse_float(r"(?m)^llama_perf_context_print:\s+prompt eval time\s*=\s*([0-9.]+) ms", text),
        "prompt_eval_ms_per_token": parse_float(r"(?m)^llama_perf_context_print:\s+prompt eval time\s*=.*?\(\s*([0-9.]+) ms per token", text),
        "load_ms": parse_float(r"load time\s*=\s*([0-9.]+) ms", text),
        "pin_token_embd_inside_budget_count": pin_token_embd_count,
        "pin_token_embd_inside_budget_mib": pin_token_embd_mib,
        "pin_output_inside_budget_count": pin_output_count,
        "pin_output_inside_budget_mib": pin_output_mib,
        "pin_outside_budget_count": len(re.findall(
            r"(?:ggml_opencl )?elastic: pin (?:unplanned output|token_embd)"
            r".*?outside budget",
            text,
        )),
        "reload_count": reload_count,
        "direct_read_ms": direct_read_ms,
        "direct_read_mib": direct_read_mib,
        "direct_read_calls": parse_int(
            r"(?:direct O_DIRECT read|direct_read)\s*:?\s*calls=([0-9]+)",
            text),
        "compute_total_ms": parse_float(
            r"compute_pt\s*:.*?total=([0-9.]+) ms", text),
        "compute_calls": parse_int(r"compute_pt\s*:.*?calls=([0-9]+)", text),
        "compute_avg_call_ms": parse_float(
            r"compute_pt\s*:.*?avg=([0-9.]+) ms", text),
        "direct_batch_units": parse_int(
            r"direct_batch\s*:.*?units=([0-9]+)", text),
        "direct_batch_tensors": parse_int(
            r"direct_batch\s*:.*?tensors=([0-9]+)", text),
        "direct_batch_fallbacks": parse_int(
            r"direct_batch\s*:.*?fallbacks=([0-9]+)", text),
        "direct_batch_errors": parse_int(
            r"direct_batch\s*:.*?errors=([0-9]+)", text),
        "multi_io_parallel_units": parse_int(
            r"\[elastic cpu multi io\].*?parallel_units=([0-9]+)", text),
        "multi_io_parallel_tensors": parse_int(
            r"\[elastic cpu multi io\].*?parallel_tensors=([0-9]+)", text),
        "pipeline_load_affinity_cpu": parse_int(
            r"elastic: LOAD worker affinity cpu=([0-9]+)", text),
        "pipeline_load_affinity_rc": parse_int(
            r"elastic: LOAD worker affinity cpu=[0-9]+ rc=(-?[0-9]+)", text),
        "pipeline_prepare_affinity_cpu": parse_int(
            r"elastic: PREPARE worker affinity cpu=([0-9]+)", text),
        "pipeline_prepare_affinity_rc": parse_int(
            r"elastic: PREPARE worker affinity cpu=[0-9]+ rc=(-?[0-9]+)", text),
        "pipeline_copy_affinity_cpu": parse_int(
            r"elastic: COPY worker affinity cpu=([0-9]+)", text),
        "pipeline_copy_affinity_rc": parse_int(
            r"elastic: COPY worker affinity cpu=[0-9]+ rc=(-?[0-9]+)", text),
        "stage_prepare_ms": parse_float(r"stage_xform\s*:.*?total=([0-9.]+) ms", text),
        "stage_prepare_mib": parse_float(r"stage_xform\s*:.*?MB=([0-9.]+)", text),
        "stage_prepare_calls": parse_int(
            r"stage_xform\s*:.*?calls=([0-9]+)", text),
        "initial_repack_calls": parse_int(
            r"\[elastic cpu layout\].*?initial_repack_calls=([0-9]+)", text),
        "initial_repack_ms": parse_float(
            r"\[elastic cpu layout\].*?initial_repack_ms=([0-9.]+)", text),
        "initial_repack_mib": parse_float(
            r"\[elastic cpu layout\].*?initial_repack_mib=([0-9.]+)", text),
        "stage_repack_calls": parse_int(
            r"\[elastic cpu layout\].*?\brepack_calls=([0-9]+)", text),
        "stage_repack_ok": parse_int(
            r"\[elastic cpu layout\].*?\brepack_ok=([0-9]+)", text),
        "stage_repack_ms": parse_float(
            r"\[elastic cpu layout\].*?\brepack_ms=([0-9.]+)", text),
        "stage_repack_mib": parse_float(
            r"\[elastic cpu layout\].*?\brepack_mib=([0-9.]+)", text),
        "stage_materialize_calls": parse_int(
            r"\[elastic cpu layout\].*?materialize_calls=([0-9]+)", text),
        "stage_materialize_ms": parse_float(
            r"\[elastic cpu layout\].*?materialize_ms=([0-9.]+)", text),
        "stage_materialize_mib": parse_float(
            r"\[elastic cpu layout\].*?materialize_mib=([0-9.]+)", text),
        "fused_layout_pair_calls": parse_int(
            r"\[elastic cpu multi_fused\].*?layout_pairs=([0-9]+)", text),
        "fused_layout_pair_ok": parse_int(
            r"\[elastic cpu multi_fused\].*?\bok=([0-9]+)", text),
        "fused_layout_pair_fallbacks": parse_int(
            r"\[elastic cpu multi_fused\].*?\bfallback=([0-9]+)", text),
        "fused_layout_parallel_calls": parse_int(
            r"\[elastic cpu multi_fused\].*?layout_pairs=[0-9]+ "
            r"ok=[0-9]+ fallback=[0-9]+ parallel=([0-9]+)", text),
        "fused_kernel_pair_candidates": parse_int(
            r"\[elastic cpu multi_fused\].*?kernel_candidates=([0-9]+)", text),
        "fused_kernel_pair_calls": parse_int(
            r"\[elastic cpu multi_fused\].*?kernel_candidates=[0-9]+ calls=([0-9]+)", text),
        "fused_kernel_pair_fallbacks": parse_int(
            r"\[elastic cpu multi_fused\].*?kernel_candidates=[0-9]+ calls=[0-9]+ fallback=([0-9]+)", text),
        "fused_kernel_pair_errors": parse_int(
            r"\[elastic cpu multi_fused\].*?errors=([0-9]+)", text),
        "fused_workspace_allocations": parse_int(
            r"\[elastic cpu multi_fused\].*?workspace_alloc=([0-9]+)", text),
        "fused_workspace_reuses": parse_int(
            r"\[elastic cpu multi_fused\].*?\breuse=([0-9]+)", text),
        "stage_copy_parallel_calls": parse_int(
            r"stage_copy\s*:.*?parallel_calls=([0-9]+)", text),
        "stage_copy_parallel_mib": parse_float(
            r"stage_copy\s*:.*?parallel_MB=([0-9.]+)", text),
        "staging_pool_hits": parse_int(
            r"staging_pool\s*:.*?hit=([0-9]+)", text),
        "staging_pool_misses": parse_int(
            r"staging_pool\s*:.*?miss=([0-9]+)", text),
        "soa_pool_hits": parse_int(
            r"pool stats:\s*soa_hit=([0-9]+)", text),
        "soa_pool_misses": parse_int(
            r"pool stats:\s*soa_hit=[0-9]+\s+soa_miss=([0-9]+)", text),
        "unit_count": unit_count,
        "cut_op_count": cut_op_count,
        "peak_unit_mib": peak_unit_mib,
        "physical_cut_tensors": physical_cut_tensors,
        "physical_cut_parts": physical_cut_parts,
        "wbm_total_mib": wbm_total_mib,
        "gpu_unit_boundaries": parse_int(
            r"\[elastic granularity sync\].*?boundaries=([0-9]+)", text),
        "gpu_unit_flushes": parse_int(
            r"\[elastic granularity sync\].*?flushes=([0-9]+)", text),
        "gpu_unit_finishes": parse_int(
            r"\[elastic granularity sync\].*?finishes=([0-9]+)", text),
        "gpu_unit_sync_errors": parse_int(
            r"\[elastic granularity sync\].*?errors=([0-9]+)", text),
        "cut_dual_candidates": parse_int(
            r"\[elastic cut dual\].*?candidates=([0-9]+)", text),
        "cut_dual_calls": parse_int(
            r"\[elastic cut dual\].*?calls=([0-9]+)", text),
        "cut_dual_budget_fallbacks": parse_int(
            r"\[elastic cut dual\].*?budget_fallbacks=([0-9]+)", text),
        "cut_dual_shape_fallbacks": parse_int(
            r"\[elastic cut dual\].*?shape_fallbacks=([0-9]+)", text),
        "cut_dual_queue_errors": parse_int(
            r"\[elastic cut dual\].*?queue_errors=([0-9]+)", text),
        "cut_dual_image_creates": parse_int(
            r"\[elastic cut dual\].*?image_creates=([0-9]+)", text),
        "cut_dual_image_cache_hits": parse_int(
            r"\[elastic cut dual\].*?image_cache_hits=([0-9]+)", text),
        "cut_dual_image_releases": parse_int(
            r"\[elastic cut dual\].*?image_releases=([0-9]+)", text),
        "cut_dual_image_errors": parse_int(
            r"\[elastic cut dual\].*?image_errors=([0-9]+)", text),
        "unit_pipeline_issued": parse_int(
            r"\[elastic unit pipeline\].*?issued=([0-9]+)", text),
        "unit_pipeline_ready": parse_int(
            r"\[elastic unit pipeline\].*?ready=([0-9]+)", text),
        "unit_pipeline_waits": parse_int(
            r"\[elastic unit pipeline\].*?waits=([0-9]+)", text),
        "unit_pipeline_wait_ms": parse_float(
            r"\[elastic unit pipeline\].*?wait_ms=([0-9.]+)", text),
        "unit_pipeline_residency_ms": parse_float(
            r"\[elastic unit pipeline residency\].*?total_ms=([0-9.]+)", text),
        "unit_pipeline_missing_units": parse_int(
            r"\[elastic unit pipeline residency\].*?missing_units=([0-9]+)", text),
        "unit_pipeline_missing_tensors": parse_int(
            r"\[elastic unit pipeline residency\].*?missing_tensors=([0-9]+)", text),
        "unit_pipeline_missing_mib": parse_float(
            r"\[elastic unit pipeline residency\].*?missing_mib=([0-9.]+)", text),
        "unit_pipeline_unissued_units": parse_int(
            r"\[elastic unit pipeline residency\].*?unissued_units=([0-9]+)", text),
        "unit_pipeline_unissued_tensors": parse_int(
            r"\[elastic unit pipeline residency\].*?unissued_tensors=([0-9]+)", text),
        "unit_pipeline_unissued_mib": parse_float(
            r"\[elastic unit pipeline residency\].*?unissued_mib=([0-9.]+)", text),
        "unit_pipeline_unissued_ms": parse_float(
            r"\[elastic unit pipeline residency\].*?unissued_ms=([0-9.]+)", text),
        "unit_pipeline_prepare_cap_declines": parse_int(
            r"\[elastic unit pipeline residency\].*?prepare_cap_declines=([0-9]+)", text),
        "unit_pipeline_prepare_cap_tensors": parse_int(
            r"\[elastic unit pipeline residency\].*?prepare_cap_tensors=([0-9]+)", text),
        "pipeline_window_avg_units": parse_float(
            r"\[elastic unit pipeline window\].*?avg_units=([0-9.]+)", text),
        "pipeline_window_max_units": parse_int(
            r"\[elastic unit pipeline window\].*?max_units=([0-9]+)", text),
        "pipeline_window_avg_mib": parse_float(
            r"\[elastic unit pipeline window\].*?avg_mib=([0-9.]+)", text),
        "pipeline_window_max_mib": parse_float(
            r"\[elastic unit pipeline window\].*?max_mib=([0-9.]+)", text),
        "pipeline_window_oversize": parse_int(
            r"\[elastic unit pipeline window\].*?oversize=([0-9]+)", text),
        "pipeline_cross_issued": parse_int(
            r"\[elastic unit pipeline cross-graph\].*?issued=([0-9]+)", text),
        "pipeline_cross_ready": parse_int(
            r"\[elastic unit pipeline cross-graph\].*?ready=([0-9]+)", text),
        "pipeline_cross_waits": parse_int(
            r"\[elastic unit pipeline cross-graph\].*?waits=([0-9]+)", text),
        "pipeline_cross_staged_mib": parse_float(
            r"\[elastic unit pipeline cross-graph\].*?staged_mib=([0-9.]+)", text),
        "pipeline_budget_samples": parse_int(
            r"\[elastic unit pipeline budget\].*?samples=([0-9]+)", text),
        "pipeline_budget_violations": parse_int(
            r"\[elastic unit pipeline budget\].*?violations=([0-9]+)", text),
        "pipeline_plan_protection_relaxations": parse_int(
            r"\[elastic unit pipeline budget\].*?"
            r"plan_protection_relaxations=([0-9]+)", text),
        "pipeline_resident_peak_mib": parse_float(
            r"\[elastic unit pipeline budget\].*?resident_peak_mib=([0-9.]+)", text),
        "pipeline_pinned_peak_mib": parse_float(
            r"\[elastic unit pipeline budget\].*?pinned_peak_mib=([0-9.]+)", text),
        "pipeline_over_budget_peak_mib": parse_float(
            r"\[elastic unit pipeline budget\].*?over_peak_mib=([0-9.]+)", text),
        "async_load_units_enqueued": parse_int(
            r"async_load\s*:.*?units=([0-9]+)/", text),
        "async_load_units_completed": parse_int(
            r"async_load\s*:.*?units=[0-9]+/([0-9]+)", text),
        "async_load_enqueued": parse_int(
            r"async_load\s*:.*?tensors=([0-9]+)/", text),
        "async_load_completed": parse_int(
            r"async_load\s*:.*?tensors=[0-9]+/([0-9]+)", text),
        "async_prepare_units_enqueued": parse_int(
            r"async_prepare\s*:.*?units=([0-9]+)/", text),
        "async_prepare_units_completed": parse_int(
            r"async_prepare\s*:.*?units=[0-9]+/([0-9]+)", text),
        "async_prepare_enqueued": parse_int(
            r"async_prepare\s*:.*?tensors=([0-9]+)/", text),
        "async_prepare_completed": parse_int(
            r"async_prepare\s*:.*?tensors=[0-9]+/([0-9]+)", text),
        "async_load_wait_ms": parse_float(
            r"async_load\s*:.*?wait_total=([0-9.]+) ms", text),
        "async_prepare_wait_ms": parse_float(
            r"async_prepare\s*:.*?wait_total=([0-9.]+) ms", text),
    }
    detail_names = (
        "soa_pool_lookup", "parent_alloc", "staging_alloc", "host_src",
        "write_enqueue", "subbuffer", "barrier_enqueue", "convert_enqueue",
        "transpose_enqueue", "final_wait_enqueue", "mark_resident",
        "cpu_xform")
    detail_ms = {
        name: parse_float(
            rf"detail {name}\s+calls=[0-9]+ total=([0-9.]+) ms",
            text)
        for name in detail_names
    }
    for name, value in detail_ms.items():
        row[f"gpu_prepare_{name}_ms"] = value
    # HOST_SRC is the LOAD dependency/source-resolution interval inside the
    # materialization callback.  Exclude it from PREPARE so an async LOAD wait
    # or a foreground direct read is not charged to layout preparation again.
    row["gpu_load_host_source_ms"] = detail_ms["host_src"]
    available_detail = [
        value for name, value in detail_ms.items()
        if name != "host_src" and value is not None]
    row["gpu_prepare_host_total_ms"] = (
        sum(available_detail) if available_detail else None)
    row["gpu_prepare_mib"] = parse_float(
        r"detail mark_resident\s+calls=[0-9]+ total=[0-9.]+ ms "
        r"avg=[0-9.]+ ms MB=([0-9.]+)",
        text)
    row["gpu_prepare_calls"] = parse_int(
        r"detail mark_resident\s+calls=([0-9]+)", text)
    origins = re.findall(
        r"\[elastic-bench\] decode phase origin generated=(\d+)"
        r"\s+backend=(cpu|gpu)",
        text,
    )
    quiesce = re.findall(
        r"\[elastic-bench\] decode phase quiesce "
        r"boundary=(begin|end) rc=(-?\d+)",
        text,
    )
    row["decode_phase_quiesce_count"] = len(quiesce)
    row["decode_phase_quiesce_sequence"] = ",".join(
        boundary for boundary, _ in quiesce)
    for boundary, rc in quiesce:
        row[f"decode_phase_quiesce_{boundary}_rc"] = int(rc)
    phase_payloads = [
        payload
        for payload in re.findall(
            r"\[elastic decode phase summary\]\s+([^\n]+)", text)
        if not payload.startswith("unavailable")
    ]
    row["decode_phase_origin_count"] = len(origins)
    row["decode_phase_origin_generated"] = (
        int(origins[-1][0]) if origins else None)
    row["decode_phase_origin_backend"] = (
        origins[-1][1] if origins else None)
    row["decode_phase_summary_count"] = len(phase_payloads)
    if phase_payloads:
        values = dict(re.findall(
            r"([a-z_]+)=([A-Za-z0-9.]+)", phase_payloads[-1]))
        row["decode_phase_backend"] = values.get("backend")
        for key in (
            "runs", "units", "nonresident_units", "reloads", "reload_bytes",
            "direct_read_calls", "load_calls", "prepare_calls",
            "compute_available", "compute_calls", "pipeline_waits",
        ):
            row[f"decode_phase_{key}"] = (
                int(values[key]) if key in values else None)
        for key in (
            "direct_read_us", "load_us", "prepare_us", "compute_us",
            "pipeline_wait_us", "pipeline_residency_us",
            "pipeline_unissued_us", "pipeline_stage_us",
            "pipeline_retire_us", "evict_us",
        ):
            row[f"decode_phase_{key.removesuffix('_us')}_ms"] = (
                int(values[key]) / 1000.0 if key in values else None)
        for key in (
            "direct_read_bytes", "load_bytes", "prepare_bytes",
            "resident_bytes",
        ):
            row[f"decode_phase_{key.removesuffix('_bytes')}_mib"] = (
                int(values[key]) / 1024.0 / 1024.0
                if key in values else None)
        row["decode_phase_counter_scope"] = (
            "decode-only-backend-counter-delta")
    else:
        row["decode_phase_counter_scope"] = None
    return row


def token_sequence(path: Path, max_ids: int | None = None) -> tuple[int, str | None]:
    if not path.exists():
        return 0, None
    token_ids: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for item in str(row.get("token_ids", "")).split(";"):
                if item:
                    token_ids.append(int(item))
    if not token_ids:
        return 0, None
    if max_ids is not None:
        token_ids = token_ids[:max(0, max_ids)]
    if not token_ids:
        return 0, None
    payload = ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    return len(token_ids), hashlib.sha256(payload).hexdigest()


def fixed_decode_metrics(path: Path, warmup_tokens: int) -> dict[str, object]:
    values: list[float] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["n_tokens"]) == 1:
                    values.append(float(row["latency_ms"]))
    measured = values[max(0, warmup_tokens):]
    return {
        "fixed_decode_tokens_total": len(values),
        "fixed_warmup_tokens_discarded": min(len(values), max(0, warmup_tokens)),
        "steady_decode_tokens": len(measured),
        "steady_median_latency_ms": statistics.median(measured) if measured else None,
        "steady_mean_latency_ms": statistics.mean(measured) if measured else None,
        "steady_p95_latency_ms": percentile(measured, 95),
        "startup_median_latency_ms": statistics.median(values[:warmup_tokens])
            if warmup_tokens > 0 and values[:warmup_tokens] else None,
    }


def mark_fixed_token_correctness(rows: list[dict[str, object]], modes: set[str]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("repeat")), str(row.get("ratio_pct"))), []).append(row)
    for values in groups.values():
        by_mode = {str(row.get("mode")): row for row in values
                   if row.get("token_ids_sha256") not in (None, "")}
        if not modes.issubset(by_mode):
            continue
        hashes = {str(by_mode[mode]["token_ids_sha256"]) for mode in modes}
        counts = {int(by_mode[mode]["token_id_count"]) for mode in modes}
        valid = len(hashes) == 1 and len(counts) == 1
        for mode in modes:
            by_mode[mode]["valid_token_sequence"] = valid


def mark_dynamic_token_correctness(rows: list[dict[str, object]], modes: set[str]) -> None:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("repeat")), []).append(row)
    for values in groups.values():
        by_mode = {str(row.get("mode")): row for row in values
                   if row.get("token_prefix_sha256") not in (None, "")}
        if not modes.issubset(by_mode):
            continue
        hashes = {str(by_mode[mode]["token_prefix_sha256"]) for mode in modes}
        valid = len(hashes) == 1
        for mode in modes:
            by_mode[mode]["valid_token_sequence"] = valid


def validate_pipeline_run(
        args: argparse.Namespace, mode: str,
        row: dict[str, object]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(row.get("returncode") or 0) != 0:
        reasons.append(f"process exit code {int(row['returncode'])}")
    if int(row.get("token_trace_pull_returncode") or 0) != 0:
        reasons.append(
            f"token trace pull exit code {int(row['token_trace_pull_returncode'])}")
    budget_trace_hash = str(
        row.get("budget_trace_sha256") or "").strip()
    if (
        not budget_trace_hash
        or str(row.get("remote_budget_trace_sha256") or "").strip()
            != budget_trace_hash
    ):
        reasons.append("remote budget trace lineage mismatch")
    timing_required = (
        row.get("eval_ms_per_token") not in (None, "")
        or int(row.get("fixed_decode_tokens_total") or 0) > 0
    )
    timing_reasons: list[str] = []
    if timing_required:
        eval_runs = int(row.get("eval_runs") or 0)
        wall_runs = int(row.get("decode_wall_runs") or 0)
        eval_total = float(row.get("eval_total_ms") or 0.0)
        wall_total = float(row.get("decode_wall_ms_total") or 0.0)
        if eval_runs <= 0:
            timing_reasons.append("standard eval timer has no decode runs")
        if wall_runs <= 0:
            timing_reasons.append("decode wall timer is missing")
        elif wall_runs != eval_runs:
            timing_reasons.append(
                f"decode wall runs={wall_runs}, eval runs={eval_runs}")
        if wall_total <= 0.0:
            timing_reasons.append("decode wall time is non-positive")
        if wall_total + max(5.0, eval_total * 0.001) < eval_total:
            timing_reasons.append(
                f"decode wall={wall_total:.3f}ms is shorter than "
                f"eval={eval_total:.3f}ms")
        token_runs = int(row.get("fixed_decode_tokens_total") or 0)
        if token_runs > 0 and wall_runs > 0 and token_runs != wall_runs:
            timing_reasons.append(
                f"token CSV decode rows={token_runs}, "
                f"decode wall runs={wall_runs}")
    row["decode_wall_contract_valid"] = not timing_reasons
    row["decode_wall_contract_invalid_reason"] = " | ".join(
        timing_reasons)
    row["latency_timing_source"] = (
        "decode-wall" if not timing_reasons and timing_required else "")
    reasons.extend(timing_reasons)
    phase_reasons: list[str] = []
    if timing_required:
        backend = str(getattr(args, "backend", "cpu"))
        expected_backend = "cpu" if backend == "cpu" else "gpu"
        if int(row.get("decode_phase_origin_count") or 0) != 1:
            phase_reasons.append("decode phase origin count is not one")
        if int(row.get("decode_phase_quiesce_count") or 0) != 2:
            phase_reasons.append(
                "decode phase quiescence boundary count is not two")
        if row.get("decode_phase_quiesce_sequence") != "begin,end":
            phase_reasons.append(
                "decode phase quiescence sequence is not begin,end")
        if (
            row.get("decode_phase_quiesce_begin_rc") in (None, "")
            or int(row["decode_phase_quiesce_begin_rc"]) != 0
        ):
            phase_reasons.append(
                "decode phase begin quiescence failed")
        if (
            row.get("decode_phase_quiesce_end_rc") in (None, "")
            or int(row["decode_phase_quiesce_end_rc"]) != 0
        ):
            phase_reasons.append(
                "decode phase end quiescence failed")
        origin_generated = row.get(
            "decode_phase_origin_generated")
        if origin_generated in (None, "") or int(origin_generated) != 0:
            phase_reasons.append("decode phase origin is not token zero")
        if row.get("decode_phase_origin_backend") != expected_backend:
            phase_reasons.append("decode phase origin backend mismatch")
        if int(row.get("decode_phase_summary_count") or 0) != 1:
            phase_reasons.append("decode phase summary count is not one")
        if row.get("decode_phase_backend") != expected_backend:
            phase_reasons.append("decode phase summary backend mismatch")
        phase_runs = int(row.get("decode_phase_runs") or 0)
        eval_runs = int(row.get("eval_runs") or 0)
        wall_runs = int(row.get("decode_wall_runs") or 0)
        token_runs = int(row.get("fixed_decode_tokens_total") or 0)
        if (
            phase_runs <= 0
            or phase_runs != eval_runs
            or phase_runs != wall_runs
            or (token_runs > 0 and phase_runs != token_runs)
        ):
            phase_reasons.append(
                f"decode phase runs={phase_runs}, eval={eval_runs}, "
                f"wall={wall_runs}, token CSV={token_runs}")
        if int(row.get("decode_phase_units") or 0) <= 0:
            phase_reasons.append("decode phase has no working units")
        if row.get("decode_phase_counter_scope") != (
                "decode-only-backend-counter-delta"):
            phase_reasons.append(
                "decode phase counter scope is not decode-only")
        required_phase_metrics = (
            "decode_phase_units",
            "decode_phase_nonresident_units",
            "decode_phase_reloads",
            "decode_phase_reload_bytes",
            "decode_phase_direct_read_calls",
            "decode_phase_direct_read_ms",
            "decode_phase_direct_read_mib",
            "decode_phase_load_calls",
            "decode_phase_load_ms",
            "decode_phase_load_mib",
            "decode_phase_prepare_calls",
            "decode_phase_prepare_ms",
            "decode_phase_prepare_mib",
            "decode_phase_pipeline_waits",
            "decode_phase_pipeline_wait_ms",
            "decode_phase_pipeline_residency_ms",
            "decode_phase_pipeline_unissued_ms",
            "decode_phase_pipeline_stage_ms",
            "decode_phase_pipeline_retire_ms",
            "decode_phase_evict_ms",
            "decode_phase_resident_mib",
        )
        missing_phase_metrics = [
            name for name in required_phase_metrics
            if row.get(name) in (None, "")
        ]
        if missing_phase_metrics:
            phase_reasons.append(
                "decode phase metrics missing: "
                + ",".join(missing_phase_metrics))
        compute_available = int(
            row.get("decode_phase_compute_available") or 0)
        if backend == "cpu":
            if compute_available != 1:
                phase_reasons.append(
                    "CPU decode compute timer unavailable")
            if (
                int(row.get("decode_phase_compute_calls") or 0) <= 0
                or float(
                    row.get("decode_phase_compute_ms") or 0.0) <= 0.0
            ):
                phase_reasons.append("CPU decode compute phase is empty")
        elif (
            compute_available != 0
            or int(row.get("decode_phase_compute_calls") or 0) != 0
            or float(row.get("decode_phase_compute_ms") or 0.0) != 0.0
        ):
            phase_reasons.append(
                "GPU decode compute must be N/A without event profiling")
    row["decode_phase_contract_valid"] = not phase_reasons
    row["decode_phase_contract_invalid_reason"] = " | ".join(
        phase_reasons)
    reasons.extend(phase_reasons)
    if bool(getattr(args, "require_device_idle", False)):
        if int(row.get("measurement_monitor_samples") or 0) <= 0:
            reasons.append("measurement isolation monitor had no samples")
        if int(row.get("measurement_monitor_failures") or 0) != 0:
            reasons.append("measurement isolation monitor failed")
        if str(row.get(
                "monitor_thread_incomplete", "false")).lower() == "true":
            reasons.append("measurement monitor thread did not exit")
        if row.get("measurement_llama_pid") in (None, ""):
            reasons.append(
                "measurement isolation monitor did not observe llama-cli")
    pin_policy = str(getattr(args, "pin", ""))
    row["pin_policy"] = pin_policy
    for tensor_suffix in ("token_embd", "output"):
        requested = pin_policy_requests(pin_policy, tensor_suffix)
        row[f"pin_{tensor_suffix}_requested"] = requested
        observed = int(
            row.get(f"pin_{tensor_suffix}_inside_budget_count") or 0)
        if requested and observed <= 0:
            reasons.append(
                f"requested {tensor_suffix} pin was not observed inside "
                "the weight budget")
        expected_mib = float(getattr(
            args, f"expected_pin_{tensor_suffix}_mib", 0.0))
        observed_mib = float(
            row.get(f"pin_{tensor_suffix}_inside_budget_mib") or 0.0)
        tolerance_mib = max(0.1, expected_mib * 0.001)
        if (
            requested and observed > 0 and expected_mib > 0.0
            and abs(observed_mib - expected_mib) > tolerance_mib
        ):
            reasons.append(
                f"{tensor_suffix} pinned {observed_mib:.2f} MiB, "
                f"expected {expected_mib:.2f} MiB")
    if int(row.get("pin_outside_budget_count") or 0) != 0:
        reasons.append(
            f"{int(row['pin_outside_budget_count'])} tensor pin(s) enlarged "
            "the weight budget")
    if args.pipeline != "on":
        return not reasons, reasons
    required = [
        "unit_pipeline_issued", "async_load_units_enqueued",
        "async_load_units_completed", "async_prepare_units_enqueued",
        "async_prepare_units_completed", "pipeline_window_max_mib",
        "pipeline_window_oversize", "compute_calls", "compute_avg_call_ms",
        "pipeline_budget_samples", "pipeline_budget_violations",
        "pipeline_plan_protection_relaxations",
        "pipeline_resident_peak_mib", "pipeline_pinned_peak_mib",
        "pipeline_over_budget_peak_mib", "wbm_total_mib",
        "physical_cut_tensors", "physical_cut_parts"]
    required.append("staging_pool_hits" if args.backend == "cpu" else "soa_pool_hits")
    if args.backend == "gpu":
        required.extend((
            "gpu_unit_boundaries", "gpu_unit_flushes",
            "gpu_unit_finishes", "gpu_unit_sync_errors"))
    for key in required:
        if row.get(key) in (None, ""):
            reasons.append(f"missing {key}")
    if reasons:
        return False, reasons

    if int(row["unit_pipeline_issued"]) <= 0:
        reasons.append("no future unit issued")
    total_tolerance_mib = max(0.1, float(args.weight_mib) * 0.001)
    if abs(float(row["wbm_total_mib"]) - float(args.weight_mib)) > (
            total_tolerance_mib):
        reasons.append(
            f"WBM total {float(row['wbm_total_mib']):.3f} MiB differs "
            f"from expected {float(args.weight_mib):.3f} MiB")
    physical_cut_tensors = int(row["physical_cut_tensors"])
    physical_cut_parts = int(row["physical_cut_parts"])
    if physical_cut_tensors <= 0:
        reasons.append(
            "common dynamic-capable physical tiling did not execute")
    elif physical_cut_parts != physical_cut_tensors * args.cut_parts:
        reasons.append(
            "physical tile count does not match the common Cut-capable "
            f"representation: tensors={physical_cut_tensors} "
            f"parts={physical_cut_parts} cut_parts={args.cut_parts}")
    budget_samples = int(row["pipeline_budget_samples"])
    budget_violations = int(row["pipeline_budget_violations"])
    plan_relaxations = int(
        row["pipeline_plan_protection_relaxations"])
    over_budget_peak_mib = float(row["pipeline_over_budget_peak_mib"])
    if budget_samples <= 0:
        reasons.append("no pipeline budget samples")
    if budget_violations != 0 or over_budget_peak_mib > 0.1:
        reasons.append(
            f"pipeline exceeded its weight budget "
            f"{budget_violations} times "
            f"(peak={over_budget_peak_mib:.2f} MiB)")
    if plan_relaxations != 0:
        reasons.append(
            f"pipeline relaxed {plan_relaxations} protected plan units")
    if int(row["async_load_units_enqueued"]) != int(row["async_load_units_completed"]):
        reasons.append("LOAD unit queue not drained")
    if int(row["async_prepare_units_enqueued"]) != int(row["async_prepare_units_completed"]):
        reasons.append("PREPARE unit queue not drained")
    if int(row.get("direct_batch_errors") or 0) != 0:
        reasons.append("direct batch I/O error")
    has_weight_load = (
        int(row.get("reload_count") or 0) > 0 or
        int(row.get("direct_read_calls") or 0) > 0 or
        int(row.get("async_load_units_enqueued") or 0) > 0)
    if args.backend == "cpu":
        # A true 100%-resident run performs no LOAD and therefore cannot reuse
        # the staging pool.  That is the expected steady state, not a missing
        # pipeline optimization.
        if has_weight_load and int(row["staging_pool_hits"]) <= 0:
            reasons.append("CPU staging pool was not reused")
        if mode in ("multi", "multi_fused") and int(
                row.get("reload_count") or 0) > 0:
            if row.get("direct_batch_units") in (None, ""):
                reasons.append("missing Multi unit-I/O counters")
            elif int(row["direct_batch_units"]) <= 0:
                reasons.append("Multi unit-I/O path did not execute")
            elif int(row.get("direct_batch_tensors") or 0) < (
                    2 * int(row["direct_batch_units"])):
                reasons.append("Multi unit-I/O did not cover two tensors")
        for worker, expected_cpu in (
                ("load", args.pipeline_load_cpu),
                ("prepare", args.pipeline_prepare_cpu)):
            if expected_cpu < 0:
                continue
            observed = row.get(f"pipeline_{worker}_affinity_cpu")
            rc = row.get(f"pipeline_{worker}_affinity_rc")
            if observed in (None, "") or rc in (None, ""):
                reasons.append(
                    f"missing CPU pipeline {worker.upper()} affinity result")
            elif int(observed) != expected_cpu or int(rc) != 0:
                reasons.append(
                    f"CPU pipeline {worker.upper()} affinity mismatch: "
                    f"cpu={int(observed)} rc={int(rc)} expected_cpu="
                    f"{expected_cpu}")
    elif int(row["soa_pool_hits"]) <= 0:
        reasons.append("OpenCL SOA retain pool was not reused")
    if args.backend == "gpu":
        boundaries = int(row["gpu_unit_boundaries"])
        flushes = int(row["gpu_unit_flushes"])
        finishes = int(row["gpu_unit_finishes"])
        errors = int(row["gpu_unit_sync_errors"])
        if boundaries <= 0:
            reasons.append("no GPU working-unit completion boundaries")
        if errors != 0:
            reasons.append(f"GPU working-unit completion errors={errors}")
        if args.gpu_unit_sync == "flush" and (
                flushes != boundaries or finishes != 0):
            reasons.append(
                f"GPU unit sync mismatch for flush: boundaries={boundaries} "
                f"flushes={flushes} finishes={finishes}")
        if args.gpu_unit_sync == "finish" and (
                finishes != boundaries or flushes != 0):
            reasons.append(
                f"GPU unit sync mismatch for finish: boundaries={boundaries} "
                f"flushes={flushes} finishes={finishes}")
        if args.gpu_unit_sync == "none" and (flushes != 0 or finishes != 0):
            reasons.append(
                f"GPU unit sync mismatch for none: flushes={flushes} "
                f"finishes={finishes}")
        # Dynamic-capable GPU runs provision the same two-tile physical Q4
        # representation for every logical mode. Multi and Tensor must use
        # the same fused dual-half compute optimization as Cut; otherwise the
        # logical-granularity comparison would penalize them artificially.
        if args.cut_dual_compute != "off":
            if row.get("cut_dual_calls") in (None, ""):
                reasons.append("missing tiled dual-compute counters")
            elif int(row["cut_dual_calls"]) <= 0:
                reasons.append("tiled fused dual-compute did not execute")
            if row.get("cut_dual_budget_fallbacks") in (None, ""):
                reasons.append(
                    "missing tiled dual-compute budget-fallback counter")
            elif int(row["cut_dual_budget_fallbacks"]) != 0:
                reasons.append(
                    "tiled weights fell back from fused dual-compute "
                    "because both tiles did not fit the weight budget: "
                    f"{int(row['cut_dual_budget_fallbacks'])}")
            if int(row.get("cut_dual_queue_errors") or 0) != 0:
                reasons.append(
                    f"tiled fused dual-compute queue errors="
                    f"{int(row['cut_dual_queue_errors'])}")
            if int(row.get("cut_dual_image_errors") or 0) != 0:
                reasons.append(
                    f"tiled fused dual-compute image errors="
                    f"{int(row['cut_dual_image_errors'])}")
            if row.get("cut_dual_image_creates") in (None, ""):
                reasons.append("missing tiled dual image-cache counters")
            elif int(row["cut_dual_image_creates"]) <= 0:
                reasons.append(
                    "tiled dual image-cache path did not execute")
            if row.get("cut_dual_image_releases") in (None, ""):
                reasons.append("missing tiled dual image-release counter")
            if (
                int(row.get("cut_dual_image_releases") or 0)
                > int(row.get("cut_dual_image_creates") or 0)
            ):
                reasons.append(
                    "tiled dual image-cache released more views than it "
                    "created")
    compute_avg_ms = float(row.get("compute_avg_call_ms") or 0.0)
    compute_slow_state = (
        args.max_compute_call_ms > 0 and
        compute_avg_ms > args.max_compute_call_ms)
    startup_ms = float(row.get("startup_median_latency_ms") or 0.0)
    steady_ms = float(row.get("steady_median_latency_ms") or 0.0)
    compute_recovered = (
        compute_slow_state and args.fixed_warmup_decode_tokens > 0 and
        startup_ms > 0.0 and steady_ms > 0.0 and
        startup_ms >= args.compute_recovery_ratio * steady_ms)
    row["compute_slow_state_observed"] = compute_slow_state
    row["compute_recovered_after_warmup"] = compute_recovered
    if compute_slow_state and not compute_recovered:
        reasons.append(
            f"delegate compute call {compute_avg_ms:.4f} ms exceeds "
            f"{args.max_compute_call_ms:.4f} ms without post-warmup recovery")
    if args.backend == "cpu" and args.min_cpu_freq_limit_khz > 0:
        if int(row.get("cpu_freq_sample_count") or 0) <= 0:
            reasons.append("no active CPU frequency samples")
        elif int(row.get("cpu_freq_limit_min_khz") or 0) < args.min_cpu_freq_limit_khz:
            reasons.append(
                f"CPU frequency limit {int(row['cpu_freq_limit_min_khz'])} kHz below "
                f"{args.min_cpu_freq_limit_khz} kHz")
    if args.backend == "cpu" and args.min_cpu_mean_freq_khz > 0:
        if int(row.get("cpu_freq_sample_count") or 0) <= 0:
            if "no active CPU frequency samples" not in reasons:
                reasons.append("no active CPU frequency samples")
        elif float(row.get("cpu_freq_mean_khz") or 0.0) < args.min_cpu_mean_freq_khz:
            reasons.append(
                f"CPU mean active frequency "
                f"{float(row['cpu_freq_mean_khz']):.0f} kHz below "
                f"{args.min_cpu_mean_freq_khz} kHz")
    if args.backend == "cpu" and args.min_cpu_median_sample_min_khz > 0:
        if int(row.get("cpu_freq_sample_count") or 0) <= 0:
            if "no active CPU frequency samples" not in reasons:
                reasons.append("no active CPU frequency samples")
        elif float(row.get("cpu_freq_median_sample_min_khz") or 0.0) < (
                args.min_cpu_median_sample_min_khz):
            reasons.append(
                f"CPU median per-sample minimum frequency "
                f"{float(row['cpu_freq_median_sample_min_khz']):.0f} kHz below "
                f"{args.min_cpu_median_sample_min_khz} kHz")

    max_window = float(row["pipeline_window_max_mib"])
    allowed_window = float(args.pipeline_lookahead_mib) + 0.1
    if max_window > allowed_window:
        reasons.append(
            f"pipeline window {max_window:.2f} MiB exceeds {allowed_window:.2f} MiB")

    load_units = int(row["async_load_units_enqueued"])
    load_tensors = int(row.get("async_load_enqueued") or 0)
    if (mode in ("multi", "multi_fused") and load_units > 0 and
            load_tensors <= load_units):
        reasons.append("Multi LOAD tasks did not contain multiple tensors")
    if mode == "multi_fused":
        if int(row.get("fused_kernel_pair_calls") or 0) <= 0:
            reasons.append("multi_fused did not execute a fused MUL_MAT pair")
        if int(row.get("fused_kernel_pair_errors") or 0) != 0:
            reasons.append("multi_fused kernel reported an error")
        if (int(row.get("reload_count") or 0) > 0 and
                int(row.get("fused_layout_pair_ok") or 0) <= 0):
            reasons.append("multi_fused reloads did not execute fused layout preparation")
        if (
            args.pipeline_copy_cpu >= 0 and
            int(row.get("fused_layout_pair_ok") or 0) > 0 and
            int(row.get("fused_layout_parallel_calls") or 0) <= 0
        ):
            reasons.append(
                "multi_fused layout pairs did not use the persistent "
                "second PREPARE lane")
        if (
            args.pipeline_copy_cpu >= 0 and
            int(row.get("fused_layout_parallel_calls") or 0) > 0
        ):
            observed = row.get("pipeline_copy_affinity_cpu")
            rc = row.get("pipeline_copy_affinity_rc")
            if observed in (None, "") or rc in (None, ""):
                reasons.append(
                    "missing CPU pipeline COPY affinity result")
            elif (
                int(observed) != args.pipeline_copy_cpu or int(rc) != 0
            ):
                reasons.append(
                    "CPU pipeline COPY affinity mismatch: "
                    f"cpu={int(observed)} rc={int(rc)} expected_cpu="
                    f"{args.pipeline_copy_cpu}")
    if args.require_real_batch_io and mode in ("multi", "multi_fused"):
        if int(row.get("direct_batch_units") or 0) <= 0:
            reasons.append("no Multi direct-I/O batch executed")
        if int(row.get("direct_batch_fallbacks") or 0) != 0:
            reasons.append("Multi direct-I/O batch fell back to sequential reads")
    return not reasons, reasons


def thermal_snapshot(serial: str) -> str:
    proc = adb(serial, "shell", "dumpsys thermalservice | head -80", timeout=30)
    return proc.stdout.strip().replace("\n", " | ")


def parse_current_temperatures(
    text: str,
) -> tuple[float | None, float | None, float | None]:
    marker = "Current temperatures from HAL:"
    if marker not in text:
        # Do not accept stale "Cached temperatures" as a clean-start signal.
        return None, None, None
    current = text.rsplit(marker, 1)[1]
    current = current.split("Current cooling devices from HAL:", 1)[0]
    cpu_values: list[float] = []
    gpu_values: list[float] = []
    skin_value: float | None = None
    pattern = re.compile(
        r"Temperature\{mValue=([0-9.]+),\s*mType=([0-9]+),\s*mName=([^,}]+)")
    for value, type_id, name in pattern.findall(current):
        if type_id == "0":
            cpu_values.append(float(value))
        if type_id == "1":
            gpu_values.append(float(value))
        if type_id == "3" and name.strip() == "skin":
            skin_value = float(value)
    return (max(cpu_values) if cpu_values else None), \
        (max(gpu_values) if gpu_values else None), skin_value


def current_temperatures(
    serial: str,
) -> tuple[float | None, float | None, float | None]:
    proc = adb(serial, "shell", "dumpsys thermalservice", timeout=30)
    return parse_current_temperatures(proc.stdout)


def external_inference_processes(
    serial: str,
    *,
    timeout: int = 30,
) -> list[str]:
    proc = adb(
        serial,
        "shell",
        "ps -A -o USER,PID,PPID,ARGS",
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "failed to sample inference processes: "
            f"rc={proc.returncode} output={(proc.stdout or '').strip()}")
    matches = []
    for line in proc.stdout.splitlines():
        fields = line.split(None, 3)
        if len(fields) < 4:
            continue
        argv = fields[3]
        executable = argv.split(None, 1)[0].strip("[]").lower()
        # Match the process executable, not arbitrary path arguments.  For
        # example `sha256sum .../llama-layersplit` is not inference and must
        # not keep the device-idle gate closed.
        executable_name = Path(executable).name.lower()
        if (
            "llama" not in executable_name
            and executable_name not in {
                "memory-elastic-op-bench",
                "granularity-pipeline-bench",
            }
        ):
            continue
        matches.append(line.strip())
    return matches


def mem_available_mib(serial: str) -> float | None:
    proc = adb(serial, "shell", "cat /proc/meminfo", timeout=30)
    match = re.search(r"(?m)^MemAvailable:\s+([0-9]+)\s+kB", proc.stdout)
    return float(match.group(1)) / 1024.0 if match else None


def battery_level_pct(serial: str) -> int | None:
    proc = adb(
        serial, "shell",
        "dumpsys battery | grep -m 1 '^  level:'", timeout=30)
    match = re.search(r"(?m)^\s*level:\s*([0-9]+)", proc.stdout)
    return int(match.group(1)) if match else None


def cpu_mask_cpus(mask: str) -> list[int]:
    if not mask:
        return []
    try:
        value = int(mask, 16)
    except ValueError:
        return []
    return [cpu for cpu in range(value.bit_length()) if value & (1 << cpu)]


def cpu_frequency_snapshot(
    serial: str,
    cpus: list[int],
    *,
    timeout: int = 30,
) -> list[tuple[int, int, int]]:
    if not cpus:
        return []
    cpu_words = " ".join(str(cpu) for cpu in cpus)
    command = (
        f"for c in {cpu_words}; do "
        "d=/sys/devices/system/cpu/cpu$c/cpufreq; "
        "printf '%s ' $c; cat $d/scaling_cur_freq $d/scaling_max_freq 2>/dev/null "
        "| tr '\\n' ' '; echo; done")
    proc = adb(serial, "shell", command, timeout=timeout)
    values: list[tuple[int, int, int]] = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and all(field.isdigit() for field in fields[:3]):
            values.append((int(fields[0]), int(fields[1]), int(fields[2])))
    return values


def device_power_state(serial: str, *, timeout: int = 30) -> str:
    """Return Android's current wakefulness state without changing it."""
    proc = adb(
        serial,
        "shell",
        "dumpsys power | grep -m 1 'mWakefulness='",
        timeout=timeout,
    )
    match = re.search(r"mWakefulness=([A-Za-z]+)", proc.stdout)
    return match.group(1) if match else "Unknown"


def set_fixed_performance_mode(serial: str, enabled: bool) -> subprocess.CompletedProcess[str]:
    value = "true" if enabled else "false"
    return adb(
        serial, "shell",
        f"cmd power set-fixed-performance-mode-enabled {value}", timeout=30)


def rearm_fixed_performance_mode(args: argparse.Namespace) -> bool:
    """Reset Android's fixed-performance hint before every measured process.

    On the OP13 the hint can retain a stale scheduling/DVFS state across a
    long sequence of otherwise independent llama-cli processes. Disabling and
    enabling it at each run boundary gives every order position the same
    initial control state.
    """
    if not args.fixed_performance_mode:
        return False
    disabled = set_fixed_performance_mode(args.serial, False)
    enabled = set_fixed_performance_mode(args.serial, True)
    if disabled.returncode != 0 or enabled.returncode != 0:
        raise RuntimeError(
            "failed to rearm fixed-performance mode: "
            f"disable={disabled.returncode} {disabled.stdout!r}; "
            f"enable={enabled.returncode} {enabled.stdout!r}")
    return True


def reboot_device_and_wait(args: argparse.Namespace, run_id: str) -> None:
    """Cold-reset persistent scheduler/DVFS state before an independent run."""
    proc = adb(args.serial, "reboot", timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to reboot device before {run_id}: {proc.stdout}")

    deadline = time.monotonic() + args.boot_wait_timeout_s
    # A TCP transport commonly remains registered as "offline" across boot.
    # Recreate it explicitly; USB serials simply become visible again through
    # the ordinary adb() retry path.
    while time.monotonic() < deadline:
        if ":" in args.serial:
            # Connecting through both lab adb servers makes the TCP transport
            # bounce between owners. Stay on the port that successfully issued
            # the reboot (or the first configured port before it is cached).
            port = _ADB_PORT_BY_SERIAL.get(
                args.serial, _ADB_PORTS[0])
            try:
                subprocess.run(
                    ["adb", "-P", str(port), "reconnect", "offline"],
                    text=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=10, check=False)
                subprocess.run(
                    ["adb", "-P", str(port), "connect", args.serial],
                    text=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=10, check=False)
            except subprocess.TimeoutExpired:
                # The adbd TCP listener can be absent during early boot.
                # Continue polling rather than aborting the experiment.
                pass
        ready = adb(
            args.serial, "shell", "getprop sys.boot_completed",
            timeout=15)
        if ready.returncode == 0 and ready.stdout.strip() == "1":
            print(json.dumps({
                "run_id": run_id,
                "device_rebooted": True,
                "boot_completed": True,
            }), flush=True)
            return
        time.sleep(2)
    raise RuntimeError(
        f"device did not finish boot before {run_id} within "
        f"{args.boot_wait_timeout_s:.0f}s")


def device_is_awake(serial: str, *, timeout: int = 30) -> bool:
    return device_power_state(serial, timeout=timeout).lower() == "awake"


def enable_device_awake(serial: str) -> dict[str, object]:
    """Keep a USB-powered benchmark device awake and return restoration state."""
    setting = adb(
        serial, "shell", "settings get global stay_on_while_plugged_in", timeout=30)
    restore = {
        "stay_on_while_plugged_in": setting.stdout.strip(),
        "was_awake": device_is_awake(serial),
    }
    proc = adb(
        serial, "shell", "settings put global stay_on_while_plugged_in 3", timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to enable stay-awake mode: {proc.stdout}")
    proc = adb(serial, "shell", "input keyevent KEYCODE_WAKEUP", timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to wake benchmark device: {proc.stdout}")
    # dumpsys can lag the key event very briefly.
    for _ in range(20):
        if device_is_awake(serial):
            return restore
        time.sleep(0.1)
    raise RuntimeError("benchmark device did not enter the Awake state")


def restore_device_awake(serial: str, restore: dict[str, object]) -> None:
    setting = str(restore["stay_on_while_plugged_in"])
    if setting.lower() in ("", "null", "none"):
        adb(serial, "shell", "settings delete global stay_on_while_plugged_in", timeout=30)
    else:
        adb(
            serial, "shell",
            f"settings put global stay_on_while_plugged_in {shlex.quote(setting)}",
            timeout=30)
    if not bool(restore["was_awake"]) and device_is_awake(serial):
        adb(serial, "shell", "input keyevent KEYCODE_SLEEP", timeout=30)


def wait_for_start_temperature(args: argparse.Namespace, run_id: str) -> None:
    if args.keep_device_awake and not device_is_awake(args.serial):
        proc = adb(args.serial, "shell", "input keyevent KEYCODE_WAKEUP", timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(f"failed to wake device before {run_id}: {proc.stdout}")
    if (args.max_cpu_start_c <= 0 and args.max_gpu_start_c <= 0 and
            args.max_skin_start_c <= 0 and args.min_mem_available_mib <= 0 and
            args.min_battery_level_pct <= 0 and not args.require_device_idle):
        return
    deadline = time.monotonic() + args.thermal_wait_timeout_s
    while True:
        # No process from this runner has been launched yet, so every llama
        # process is external here.  Do not exclude a process merely because
        # another runner uses the same binary/model paths.
        external = (external_inference_processes(args.serial)
                    if args.require_device_idle else [])
        cpu_c, gpu_c, skin_c = current_temperatures(args.serial)
        available_mib = mem_available_mib(args.serial)
        battery_pct = battery_level_pct(args.serial)
        cpu_ok = args.max_cpu_start_c <= 0 or (cpu_c is not None and cpu_c <= args.max_cpu_start_c)
        gpu_ok = args.max_gpu_start_c <= 0 or (gpu_c is not None and gpu_c <= args.max_gpu_start_c)
        skin_ok = args.max_skin_start_c <= 0 or (skin_c is not None and skin_c <= args.max_skin_start_c)
        memory_ok = (args.min_mem_available_mib <= 0 or
                     (available_mib is not None and
                      available_mib >= args.min_mem_available_mib))
        battery_ok = (
            args.min_battery_level_pct <= 0 or
            (battery_pct is not None and
             battery_pct >= args.min_battery_level_pct))
        if (cpu_ok and gpu_ok and skin_ok and memory_ok and battery_ok and
                not external):
            if args.keep_device_awake and not device_is_awake(args.serial):
                proc = adb(
                    args.serial, "shell",
                    "input keyevent KEYCODE_WAKEUP", timeout=30)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"failed to wake device before {run_id}: "
                        f"{proc.stdout}")
                # Validate the gates once more after the power-state
                # transition rather than starting on a stale snapshot.
                time.sleep(1.0)
                continue
            print(json.dumps({"run_id": run_id, "thermal_ready": True,
                              "cpu_c": cpu_c, "gpu_c": gpu_c, "skin_c": skin_c,
                              "mem_available_mib": available_mib,
                              "battery_level_pct": battery_pct}), flush=True)
            return
        if (args.sleep_while_waiting_for_battery and not battery_ok and
                device_is_awake(args.serial)):
            adb(
                args.serial, "shell",
                "input keyevent KEYCODE_SLEEP", timeout=30)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"start gate timeout for {run_id}: cpu={cpu_c}C gpu={gpu_c}C "
                f"skin={skin_c}C MemAvailable={available_mib}MiB "
                f"battery={battery_pct}% external={external}")
        print(json.dumps({"run_id": run_id, "thermal_ready": False,
                          "cpu_c": cpu_c, "gpu_c": gpu_c, "skin_c": skin_c,
                          "mem_available_mib": available_mib,
                          "battery_level_pct": battery_pct,
                          "external_inference": external}), flush=True)
        time.sleep(args.thermal_poll_s)


def run_with_interference_monitor(
        args: argparse.Namespace, command: str
        ) -> tuple[subprocess.CompletedProcess[str], list[str], list[str], dict[str, object]]:
    if not args.require_device_idle and not args.keep_device_awake:
        return adb(args.serial, "shell", command, timeout=args.timeout), [], [], {}
    stop = threading.Event()
    seen: set[str] = set()
    power_violations: set[str] = set()
    frequency_samples: list[list[tuple[int, int, int]]] = []
    monitored_cpus = cpu_mask_cpus(args.cpu_mask) if args.backend == "cpu" else []
    measured_llama_pid: int | None = None
    measurement_monitor_samples = 0
    measurement_monitor_failures = 0
    poll_s = max(0.5, min(1.0, args.device_idle_poll_s))
    monitor_adb_timeout_s = 5

    def monitor() -> None:
        nonlocal measured_llama_pid
        nonlocal measurement_monitor_samples
        nonlocal measurement_monitor_failures
        while not stop.wait(poll_s):
            try:
                if args.require_device_idle:
                    processes = external_inference_processes(
                        args.serial, timeout=monitor_adb_timeout_s)
                    measurement_monitor_samples += 1
                    non_llama = [
                        line for line in processes
                        if "llama" not in Path(
                            line.split(None, 3)[3]
                            .split(None, 1)[0]
                            .strip("[]")
                        ).name.lower()
                    ]
                    seen.update(non_llama)
                    by_pid = {
                        int(line.split(None, 3)[1]): line
                        for line in processes
                        if "llama" in Path(
                            line.split(None, 3)[3]
                            .split(None, 1)[0]
                            .strip("[]")
                        ).name.lower()
                    }
                    # The start gate observed an empty process set.  Treat the
                    # first sole llama executable as this measured invocation;
                    # any concurrent executable is interference even when it
                    # uses identical binary/model paths.
                    if measured_llama_pid is None and len(by_pid) == 1:
                        measured_llama_pid = next(iter(by_pid))
                    if len(by_pid) > 1:
                        seen.update(processes)
                    elif (
                        measured_llama_pid is not None
                        and by_pid
                        and measured_llama_pid not in by_pid
                    ):
                        seen.update(processes)
                    # MemAvailable is a pre-run cleanliness gate. The measured
                    # process legitimately consumes that memory after it
                    # starts; competing inference processes remain monitored
                    # above.
                if (
                    args.keep_device_awake
                    and not device_is_awake(
                        args.serial, timeout=monitor_adb_timeout_s)
                ):
                    power_violations.add(
                        "device left Awake state during inference")
                snapshot = cpu_frequency_snapshot(
                    args.serial,
                    monitored_cpus,
                    timeout=monitor_adb_timeout_s,
                )
                if snapshot:
                    frequency_samples.append(snapshot)
            except (OSError, RuntimeError, subprocess.SubprocessError):
                measurement_monitor_failures += 1

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        proc = adb(args.serial, "shell", command, timeout=args.timeout)
    finally:
        stop.set()
        worker.join(timeout=3 * monitor_adb_timeout_s + 2.0)
    monitor_thread_incomplete = worker.is_alive()
    if monitor_thread_incomplete:
        measurement_monitor_failures += 1
    if args.keep_device_awake and not device_is_awake(args.serial):
        power_violations.add("device was not Awake after inference")
    frequency: dict[str, object] = {
        "cpu_freq_sample_count": len(frequency_samples),
        "measurement_monitor_samples": measurement_monitor_samples,
        "measurement_monitor_failures": measurement_monitor_failures,
        "monitor_thread_incomplete": monitor_thread_incomplete,
        "measurement_llama_pid": (
            measured_llama_pid if measured_llama_pid is not None else ""),
    }
    if frequency_samples:
        current = [freq for sample in frequency_samples for _, freq, _ in sample]
        limits = [limit for sample in frequency_samples for _, _, limit in sample]
        per_sample_min = [min(freq for _, freq, _ in sample)
                          for sample in frequency_samples]
        frequency.update({
            "cpu_freq_min_khz": min(current),
            "cpu_freq_mean_khz": statistics.mean(current),
            "cpu_freq_median_sample_min_khz": statistics.median(per_sample_min),
            "cpu_freq_limit_min_khz": min(limits),
        })
    return proc, sorted(seen), sorted(power_violations), frequency


def remote_command(args: argparse.Namespace, mode: str, trace_remote: str,
                   token_csv_remote: str | None) -> str:
    env = {
        "LD_LIBRARY_PATH": f"{args.remote_lib_dir}:/system/vendor/lib64:/vendor/lib64",
        "GGML_ELASTIC_GRANULARITY": mode,
        # Mixed-granularity planning cannot allocate or free model tensors
        # when the budget changes. Provision one common Cut-capable physical
        # representation for every logical fixed mode, matching the real
        # dynamic runtime. Tensor/Multi group these tiles atomically; only Cut
        # exposes them as independent working units.
        "GGML_ELASTIC_GRANULARITY_DYNAMIC": "1",
        "GGML_ELASTIC_MULTI_TENSORS": str(args.multi_tensors),
        "GGML_ELASTIC_CUT_PARTS": str(args.cut_parts),
        "GGML_ELASTIC_CUT_DUAL_COMPUTE": args.cut_dual_compute,
        "GGML_ELASTIC_BUDGET_CSV": trace_remote,
        "GGML_ELASTIC_DYNAMIC": "1",
        "GGML_ELASTIC_KV_MB": str(args.kv_mib),
        "GGML_ELASTIC_MISC_MB": str(args.misc_mib),
        "GGML_ELASTIC_DIRECT_IO": "1",
        "GGML_ELASTIC_PROFILE": "1" if args.backend == "cpu" else "0",
        "GGML_ELASTIC_PIN": args.pin,
        "GGML_ELASTIC_PREFETCH": "0",
        "GGML_ELASTIC_CHUNK_SIZE": "0",
        "GGML_CPU_REPACK_THREADS": str(args.cpu_repack_threads),
        "GGML_ELASTIC_ASYNC_STAGE_LOAD": "0",
        "GGML_ELASTIC_ASYNC_STAGE_PREPARE": "0",
        "GGML_ELASTIC_SYNC_STAGE_LOAD": "0",
        "GGML_ELASTIC_SYNC_STAGE_WORKERS": "0",
        "GGML_ELASTIC_UNIT_PIPELINE": "0",
        "GGML_ELASTIC_UNIT_PIPELINE_LOOKAHEAD": "0",
    }
    if args.pipeline == "on":
        # Granularity-aware pipeline: stage future real working units while
        # the current unit computes. Keep the legacy node-distance prefetch
        # disabled so granularity is the only scheduling-unit difference.
        env.update({
            "GGML_ELASTIC_UNIT_PIPELINE": "1",
            "GGML_ELASTIC_UNIT_PIPELINE_LOOKAHEAD": str(args.pipeline_lookahead),
            "GGML_ELASTIC_UNIT_PIPELINE_LOOKAHEAD_MB": str(
                args.pipeline_lookahead_mib),
            "GGML_ELASTIC_UNIT_PIPELINE_GRAPH_LOOKAHEAD": str(
                args.pipeline_graph_lookahead),
            "GGML_ELASTIC_ASYNC_STAGE_LOAD": "1",
            "GGML_ELASTIC_ASYNC_STAGE_PREPARE": "1",
            "GGML_ELASTIC_ASYNC_PREPARE_MAX_PENDING": str(args.pipeline_max_pending),
            "GGML_ELASTIC_HOST_STAGING_POOL_MB": str(args.pipeline_max_pending_mib),
        })
        if args.backend == "cpu":
            # `pooled` exercises the explicit LOAD -> PREPARE pipeline using
            # the runtime's reusable size-class staging buffers. `direct` is a
            # zero-copy diagnostic path that reads into the stable final CPU
            # residency buffer.
            env["GGML_ELASTIC_ASYNC_STAGE_LOAD"] = (
                "1" if args.cpu_pipeline_staging == "pooled" else "0")
            if args.cpu_pipeline_staging == "pooled":
                # Any current-unit fallback uses the same reusable staged
                # LOAD/PREPARE path as prefetched units.
                env["GGML_ELASTIC_SYNC_STAGE_LOAD"] = "1"
            if args.pipeline_load_cpu >= 0:
                env["GGML_ELASTIC_PIPELINE_LOAD_CPU"] = str(args.pipeline_load_cpu)
            if args.pipeline_prepare_cpu >= 0:
                env["GGML_ELASTIC_PIPELINE_PREPARE_CPU"] = str(args.pipeline_prepare_cpu)
            if (args.cpu_pipeline_staging == "pooled" and
                    args.pipeline_copy_cpu >= 0):
                # Reuse the same pooled staging allocation.  A persistent
                # helper copies one half while PREPARE copies the other; no
                # per-unit thread or temporary buffer is created.
                env["GGML_ELASTIC_CPU_STAGE_COPY_PARALLEL"] = "1"
                env["GGML_ELASTIC_CPU_STAGE_COPY_MIN_KB"] = str(
                    args.pipeline_copy_min_kib)
                env["GGML_ELASTIC_PIPELINE_COPY_CPU"] = str(args.pipeline_copy_cpu)
    if (args.backend == "cpu" and args.pipeline == "off" and
            args.cpu_staged_baseline):
        # Controlled no-overlap baseline: identical pooled LOAD/PREPARE work,
        # executed synchronously before compute.
        env["GGML_ELASTIC_SYNC_STAGE_LOAD"] = "1"
        env["GGML_ELASTIC_SYNC_STAGE_WORKERS"] = "1"
        env["GGML_ELASTIC_ASYNC_STAGE_LOAD"] = "1"
        env["GGML_ELASTIC_ASYNC_STAGE_PREPARE"] = "1"
        env["GGML_ELASTIC_ASYNC_PREPARE_MAX_PENDING"] = str(
            args.pipeline_max_pending)
        env["GGML_ELASTIC_HOST_STAGING_POOL_MB"] = str(
            args.pipeline_max_pending_mib)
        if args.pipeline_load_cpu >= 0:
            env["GGML_ELASTIC_PIPELINE_LOAD_CPU"] = str(args.pipeline_load_cpu)
        if args.pipeline_prepare_cpu >= 0:
            env["GGML_ELASTIC_PIPELINE_PREPARE_CPU"] = str(
                args.pipeline_prepare_cpu)
        if args.pipeline_copy_cpu >= 0:
            env["GGML_ELASTIC_CPU_STAGE_COPY_PARALLEL"] = "1"
            env["GGML_ELASTIC_CPU_STAGE_COPY_MIN_KB"] = str(
                args.pipeline_copy_min_kib)
            env["GGML_ELASTIC_PIPELINE_COPY_CPU"] = str(args.pipeline_copy_cpu)
    if token_csv_remote:
        env["GGML_ELASTIC_TOKEN_CSV"] = token_csv_remote
    if args.duration_s > 0:
        env["LLAMA_ELASTIC_BENCH_SECONDS"] = str(args.duration_s)
    if args.backend == "gpu":
        env.update({
            "GGML_OPENCL_ELASTIC": "1",
            "GGML_ELASTIC_GPU_UNIT_SYNC": args.gpu_unit_sync,
            "GGML_ELASTIC_TIMING": "1",
            "GGML_ELASTIC_CL_RETAIN": "0",
            "GGML_ELASTIC_PIN_UNPLANNED_OUTPUT": "0",
            "GGML_ELASTIC_Q4_IMAGE_DEBUG": "1",
            "GGML_ELASTIC_GPU_PREFETCH": "0",
            "GGML_ELASTIC_ASYNC_STAGE_LOAD": "0",
            "GGML_ELASTIC_ASYNC_STAGE_PREPARE": "0",
        })
        if args.pipeline == "on":
            env.update({
                # Reuse the backend's existing pools.  Pipeline results are
                # invalid if every reload recreates host/cl_mem temporaries.
                "GGML_ELASTIC_CL_RETAIN": "1",
                "GGML_ELASTIC_CL_RETAIN_MB": str(args.pipeline_max_pending_mib),
                "GGML_ELASTIC_ASYNC_STAGE_LOAD": "1",
                "GGML_ELASTIC_ASYNC_STAGE_PREPARE": "1",
                "GGML_ELASTIC_ASYNC_LOAD_BOUNDED": "1",
                "GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD": "0",
                "GGML_ELASTIC_ASYNC_LOAD_MAX_PENDING_MB": str(
                    args.pipeline_max_pending_mib),
                "GGML_ELASTIC_ASYNC_LOAD_MAX_STAGED_MB": str(
                    args.pipeline_max_pending_mib),
                "GGML_ELASTIC_ASYNC_LOAD_MAX_PENDING_COUNT": str(
                    args.pipeline_max_pending),
                "GGML_ELASTIC_ASYNC_XFER": "1",
                "GGML_ELASTIC_RELOAD_ON_XFER": "1",
                "GGML_ELASTIC_RELEASE_STAGE_AFTER_XFORM": "1",
                "GGML_ELASTIC_GPU_PREFETCH_INFLIGHT_MB": str(
                    args.pipeline_max_pending_mib),
                "GGML_ELASTIC_STAGE_DETAIL": "1",
            })

    argv = [
        args.remote_binary, "-m", args.remote_model,
        "-t", str(args.threads), "-c", str(args.context),
        "-b", str(args.batch), "-ub", str(args.ubatch),
        "-p", args.prompt, "-n", str(args.n_predict),
        "--seed", str(args.seed), "--temp", "0", "--no-warmup", "-no-cnv",
    ]
    if args.ignore_eos:
        argv.append("--ignore-eos")
    if args.cpu_mask:
        argv += ["--cpu-mask", args.cpu_mask,
                 "--cpu-strict", str(args.cpu_strict),
                 "--poll", str(args.poll)]
    if args.backend == "cpu":
        argv += ["-ngl", "0", "-dev", "CPU_Elastic"]
    else:
        argv += ["-ngl", "99"]
    invocation = " ".join(
        [f"{key}={shlex.quote(value)}" for key, value in env.items()]
        + [shlex.quote(value) for value in argv]
    )
    return f"cd {shlex.quote(args.remote_work_dir)} && {invocation}"


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def prepare_dynamic_trace(args: argparse.Namespace, out_dir: Path) -> tuple[Path, float]:
    """Create the exact total-budget trace consumed by BudgetWatcher."""
    inputs = out_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    local_trace = inputs / "dynamic_trace.csv"

    if args.dynamic_trace is None:
        points: list[tuple[float, int]] = []
        for item in args.dynamic_points.split(","):
            seconds, ratio = item.split(":", 1)
            points.append((float(seconds), int(ratio)))
        with local_trace.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("time_sec", "budget_mb"))
            # BudgetWatcher interpolates between adjacent trace samples. Duplicate
            # each plateau immediately before the following transition so this is
            # a true step experiment instead of an unintended budget ramp.
            trace_points: list[tuple[float, int]] = []
            for index, (seconds, ratio) in enumerate(points):
                budget = int(math.ceil(
                    args.weight_mib * ratio / 100.0 + args.kv_mib + args.misc_mib))
                trace_points.append((seconds, budget))
                if index + 1 < len(points):
                    next_seconds = points[index + 1][0]
                    if next_seconds <= seconds:
                        raise ValueError("dynamic point times must be strictly increasing")
                    trace_points.append((max(seconds, next_seconds - 0.001), budget))
            for seconds, budget in trace_points:
                writer.writerow((f"{seconds:.3f}", budget))
        return local_trace, points[-1][0]

    source_rows: list[tuple[float, float]] = []
    with args.dynamic_trace.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError(f"dynamic trace needs two columns: {args.dynamic_trace}")
        time_key = next((key for key in ("t_sec", "time_sec") if key in reader.fieldnames),
                        reader.fieldnames[0])
        budget_key = next((key for key in ("mem_available_mb", "budget_mib", "budget_mb")
                           if key in reader.fieldnames), reader.fieldnames[1])
        for row in reader:
            seconds = float(row[time_key])
            if seconds < args.dynamic_trace_start_s:
                continue
            if args.dynamic_trace_end_s > 0 and seconds > args.dynamic_trace_end_s:
                continue
            source_rows.append((seconds - args.dynamic_trace_start_s,
                                float(row[budget_key])))
    if not source_rows:
        raise ValueError(f"empty dynamic trace: {args.dynamic_trace}")
    source_rows.sort(key=lambda item: item[0])
    source_min = min(value for _, value in source_rows)
    source_max = max(value for _, value in source_rows)
    if (args.dynamic_trace_budget_mode == "rescale" and
            source_max <= source_min):
        raise ValueError(f"constant source trace cannot be rescaled: {args.dynamic_trace}")

    total_min = math.ceil(
        args.weight_mib * args.dynamic_min_ratio / 100.0 + args.kv_mib + args.misc_mib)
    total_max = math.ceil(
        args.weight_mib * args.dynamic_max_ratio / 100.0 + args.kv_mib + args.misc_mib)
    with local_trace.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_sec", "budget_mb"))
        for seconds, value in source_rows:
            if args.dynamic_trace_budget_mode == "native":
                # The trace was already constructed for this model. Preserve
                # its absolute total-memory budgets instead of performing a
                # second model/range mapping.
                budget = value
            else:
                normalized = (value - source_min) / (source_max - source_min)
                budget = round(total_min + normalized * (total_max - total_min))
            writer.writerow((f"{seconds:.3f}", budget))
    provenance = {
        "source": str(args.dynamic_trace),
        "budget_mode": args.dynamic_trace_budget_mode,
        "source_min_mib": source_min,
        "source_max_mib": source_max,
        "source_window_start_s": args.dynamic_trace_start_s,
        "source_window_end_s": args.dynamic_trace_end_s or None,
        "source_time_rebased": args.dynamic_trace_start_s > 0,
        "duration_s": source_rows[-1][0],
    }
    if args.dynamic_trace_budget_mode == "native":
        provenance.update({
            "output_min_mib": source_min,
            "output_max_mib": source_max,
            "transform": "absolute budget values copied without rescaling",
        })
    else:
        provenance.update({
            "target_total_min_mib": total_min,
            "target_total_max_mib": total_max,
            "target_weight_ratio_min_pct": args.dynamic_min_ratio,
            "target_weight_ratio_max_pct": args.dynamic_max_ratio,
            "transform": "source min/max linearly rescaled to target totals",
        })
    (inputs / "trace_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    return local_trace, source_rows[-1][0]


def summarize_fixed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        if (str(row.get("returncode", "")) != "0" or
                row.get("eval_ms_per_token") in (None, "") or
                row.get("steady_median_latency_ms") in (None, "") or
                str(row.get("valid_device_idle", "true")).lower() == "false" or
                str(row.get("valid_device_awake", "true")).lower() == "false" or
                str(row.get("valid_runtime", "true")).lower() == "false" or
                str(row.get(
                    "decode_wall_contract_valid", "")).lower() != "true" or
                str(row.get(
                    "decode_phase_contract_valid", "")).lower() != "true" or
                str(row.get("valid_token_sequence", "")).lower() != "true"):
            continue
        groups.setdefault((str(row["mode"]), int(row["ratio_pct"])), []).append(row)
    summary = []
    for (mode, ratio), values in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        out: dict[str, object] = {
            "mode": mode,
            "ratio_pct": ratio,
            "budget_mib": values[0]["budget_mib"],
            "n": len(values),
        }
        for key in ("steady_median_latency_ms", "steady_mean_latency_ms",
                    "steady_p95_latency_ms", "startup_median_latency_ms",
                    "eval_ms_per_token", "prompt_eval_ms_per_token",
                    "decode_phase_direct_read_ms",
                    "decode_phase_direct_read_mib",
                    "decode_phase_prepare_ms",
                    "decode_phase_compute_ms",
                    "decode_phase_pipeline_wait_ms", "wall_s"):
            samples = [float(row[key]) for row in values if row.get(key) is not None]
            out[f"median_{key}"] = statistics.median(samples) if samples else None
        summary.append(out)
    return summary


def summarize_dynamic_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for mode in MODES:
        values = [row for row in rows if str(row.get("mode")) == mode
                  and str(row.get("returncode")) == "0"
                  and str(row.get("valid_device_idle", "true")).lower() != "false"
                  and str(row.get("valid_device_awake", "true")).lower() != "false"
                  and str(row.get("valid_runtime", "true")).lower() != "false"
                  and str(row.get(
                      "decode_wall_contract_valid", "")).lower() == "true"
                  and str(row.get(
                      "decode_phase_contract_valid", "")).lower() == "true"
                  and str(row.get("valid_token_sequence", "")).lower() == "true"]
        if not values:
            continue
        out: dict[str, object] = {"mode": mode, "n_runs": len(values)}
        for key in (
            "n_decode", "median_latency_ms", "p95_latency_ms",
            "mean_latency_ms", "decode_tokens_per_s",
            "decode_phase_direct_read_mib",
            "decode_phase_prepare_ms",
            "decode_phase_pipeline_wait_ms",
        ):
            samples = [float(row[key]) for row in values if row.get(key) not in (None, "")]
            out[f"median_{key}"] = statistics.median(samples) if samples else None
            out[f"min_{key}"] = min(samples) if samples else None
            out[f"max_{key}"] = max(samples) if samples else None
        summary.append(out)
    return summary


def summarize_dynamic_bins(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["mode"]), int(row["ratio_bin_pct"])), []).append(row)
    summary = []
    for (mode, ratio), values in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        medians = [float(row["median_latency_ms"]) for row in values]
        p95s = [float(row["p95_latency_ms"]) for row in values]
        summary.append({
            "mode": mode, "ratio_bin_pct": ratio, "n_runs": len(values),
            "n_decode_total": sum(int(row["n_decode"]) for row in values),
            "median_of_run_medians_ms": statistics.median(medians),
            "median_of_run_p95_ms": statistics.median(p95s),
            "min_run_median_ms": min(medians), "max_run_median_ms": max(medians),
        })
    return summary


def fixed_sweep(args: argparse.Namespace, out_dir: Path, remote_dir: str) -> None:
    ratios = [int(item) for item in args.ratios.split(",")]
    budgets = {
        ratio: int(math.ceil(args.weight_mib * ratio / 100.0 + args.kv_mib + args.misc_mib))
        for ratio in ratios
    }
    traces: dict[int, tuple[Path, str]] = {}
    trace_sha256: dict[int, str] = {}
    inputs = out_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for ratio, budget in budgets.items():
        local = inputs / f"fixed_{ratio:02d}pct_{budget}MiB.csv"
        local.write_text(f"time_sec,budget_mb\n0,{budget}\n3600,{budget}\n", encoding="utf-8")
        remote = f"{remote_dir}/{local.name}"
        push(args.serial, local, remote)
        local_sha256 = hashlib.sha256(local.read_bytes()).hexdigest()
        actual_sha256 = remote_sha256(
            args.serial, remote, min(args.timeout, 60))
        if actual_sha256 != local_sha256:
            raise RuntimeError(
                f"remote fixed-budget trace SHA-256 mismatch at "
                f"{ratio}%: local={local_sha256} remote={actual_sha256}")
        traces[ratio] = (local, remote)
        trace_sha256[ratio] = local_sha256

    rows: list[dict[str, object]] = (
        read_csv_rows(out_dir / "runs.csv") if args.resume else [])
    completed = {
        str(row["run_id"]) for row in rows
        if str(row.get("returncode", "")) == "0"
        and row.get("eval_ms_per_token") not in (None, "")
        and row.get("steady_median_latency_ms") not in (None, "")
        and str(row.get("valid_device_idle", "")).lower() == "true"
        and str(row.get("valid_device_awake", "")).lower() == "true"
        and str(row.get("valid_runtime", "")).lower() == "true"
        and str(row.get(
            "decode_wall_contract_valid", "")).lower() == "true"
        and str(row.get(
            "decode_phase_contract_valid", "")).lower() == "true"
        and str(row.get("valid_token_sequence", "")).lower() == "true"
        and (
            not args.require_device_idle
            or (
                int(row.get("measurement_monitor_samples") or 0) > 0
                and int(row.get(
                    "measurement_monitor_failures") or 0) == 0
                and str(row.get(
                    "monitor_thread_incomplete", "false")
                ).lower() != "true"
                and row.get("measurement_llama_pid") not in (None, "")
            )
        )
        and row.get("remote_binary_sha256")
            == args.remote_binary_sha256
        and row.get("remote_model_sha256")
            == args.remote_model_sha256
        and row.get("fixed_run_config_sha256")
            == args.fixed_run_config_sha256
        and row.get("budget_trace_sha256")
            == trace_sha256.get(int(float(row.get("ratio_pct") or -1)), "")
        and row.get("remote_budget_trace_sha256")
            == row.get("budget_trace_sha256")
    }
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    tokens_dir = out_dir / "tokens"
    tokens_dir.mkdir(exist_ok=True)
    for repeat in range(args.repeats):
        ratio_order = ratios if repeat % 2 == 0 else list(reversed(ratios))
        for ratio_position, ratio in enumerate(ratio_order):
            # A one-round sweep must not always put Multi on a cold device and
            # Cut on a warm device. Rotate the Latin-square mode order across
            # budget points; additional repeats still reverse the budget order.
            mode_order = MODE_ORDERS[
                (repeat + ratio_position) % len(MODE_ORDERS)]
            for mode in mode_order:
                if mode not in args.modes:
                    continue
                run_id = f"cpu" if args.backend == "cpu" else "gpu"
                run_id += f"_r{repeat}_{ratio:02d}pct_{mode}"
                if run_id in completed:
                    print(json.dumps({"run_id": run_id, "resume_skip": True}), flush=True)
                    continue
                if args.reboot_between_runs:
                    reboot_device_and_wait(args, run_id)
                wait_for_start_temperature(args, run_id)
                performance_mode_rearmed = rearm_fixed_performance_mode(args)
                before = thermal_snapshot(args.serial)
                battery_before = battery_level_pct(args.serial)
                power_before = device_power_state(args.serial)
                # Replace a previous interrupted or contaminated attempt with
                # this id. Clean rows were skipped above.
                rows = [row for row in rows if str(row.get("run_id")) != run_id]
                remote_tokens = f"{remote_dir}/{run_id}_tokens.csv"
                adb(args.serial, "shell", f"rm -f {shlex.quote(remote_tokens)}", timeout=30)
                command = remote_command(args, mode, traces[ratio][1], remote_tokens)
                start = time.monotonic()
                performance_restore_error = ""
                try:
                    proc, external_overlap, power_violations, frequency = run_with_interference_monitor(
                        args, command)
                finally:
                    if performance_mode_rearmed:
                        disabled = set_fixed_performance_mode(args.serial, False)
                        if disabled.returncode != 0:
                            performance_restore_error = disabled.stdout.strip()
                wall = time.monotonic() - start
                after = thermal_snapshot(args.serial)
                battery_after = battery_level_pct(args.serial)
                power_after = device_power_state(args.serial)
                (raw_dir / f"{run_id}.log").write_text(proc.stdout, encoding="utf-8")
                local_tokens = tokens_dir / f"{run_id}.csv"
                pull = adb(args.serial, "pull", remote_tokens, str(local_tokens), timeout=60)
                row: dict[str, object] = {
                    "run_id": run_id, "backend": args.backend, "repeat": repeat,
                    "mode": mode, "ratio_pct": ratio, "budget_mib": budgets[ratio],
                    "pipeline": args.pipeline,
                    "cpu_pipeline_staging": args.cpu_pipeline_staging,
                    "cpu_staged_baseline": args.cpu_staged_baseline,
                    "pipeline_lookahead": args.pipeline_lookahead,
                    "pipeline_lookahead_mib": args.pipeline_lookahead_mib,
                    "pipeline_max_pending": args.pipeline_max_pending,
                    "pipeline_max_pending_mib": args.pipeline_max_pending_mib,
                    "pipeline_load_cpu": args.pipeline_load_cpu,
                    "pipeline_prepare_cpu": args.pipeline_prepare_cpu,
                    "pipeline_copy_cpu": args.pipeline_copy_cpu,
                    "cut_dual_compute": args.cut_dual_compute,
                    "performance_mode_rearmed": performance_mode_rearmed,
                    "performance_mode_restore_error": performance_restore_error,
                    "weight_target_mib": args.weight_mib * ratio / 100.0,
                    "fixed_run_config_sha256":
                        args.fixed_run_config_sha256,
                    "budget_trace_sha256": trace_sha256[ratio],
                    "remote_budget_trace_sha256": trace_sha256[ratio],
                    "returncode": proc.returncode, "wall_s": wall,
                    "thermal_before": before, "thermal_after": after,
                    "battery_level_pct_before": battery_before,
                    "battery_level_pct_after": battery_after,
                    "device_power_before": power_before,
                    "device_power_after": power_after,
                    "valid_device_idle": not external_overlap,
                    "external_overlap": " | ".join(external_overlap),
                    "valid_device_awake": not power_violations,
                    "power_state_violation": " | ".join(power_violations),
                }
                row.update(input_lineage(args))
                row.update(parse_log(proc.stdout))
                row.update(frequency)
                token_count, token_hash = token_sequence(local_tokens)
                row["token_trace_pull_returncode"] = pull.returncode
                row["token_id_count"] = token_count
                row["token_ids_sha256"] = token_hash
                prefix_count, prefix_hash = token_sequence(
                    local_tokens, args.dynamic_token_prefix_count)
                row["token_prefix_count"] = prefix_count
                row["token_prefix_sha256"] = prefix_hash
                row.update(fixed_decode_metrics(
                    local_tokens, args.fixed_warmup_decode_tokens))
                valid_runtime, runtime_reasons = validate_pipeline_run(args, mode, row)
                if performance_restore_error:
                    valid_runtime = False
                    runtime_reasons.append(
                        "failed to restore fixed-performance mode: "
                        f"{performance_restore_error}")
                if int(row.get("steady_decode_tokens") or 0) < args.fixed_min_measure_tokens:
                    valid_runtime = False
                    runtime_reasons.append(
                        f"only {int(row.get('steady_decode_tokens') or 0)} steady decode tokens")
                row["valid_runtime"] = valid_runtime
                row["runtime_invalid_reason"] = " | ".join(runtime_reasons)
                rows.append(row)
                mark_fixed_token_correctness(rows, set(args.modes))
                write_csv(out_dir / "runs.csv", rows)
                write_csv(out_dir / "summary.csv", summarize_fixed(rows))
                print(json.dumps({k: row.get(k) for k in (
                    "run_id", "returncode", "eval_ms_per_token",
                    "steady_median_latency_ms", "startup_median_latency_ms",
                    "direct_read_mib",
                    "unit_count", "cut_op_count", "wall_s", "valid_device_idle",
                    "external_overlap", "valid_device_awake", "valid_runtime",
                    "runtime_invalid_reason", "valid_token_sequence",
                    "power_state_violation")}, ensure_ascii=False), flush=True)
                if args.cooldown_s > 0:
                    time.sleep(args.cooldown_s)


def dynamic_sweep(args: argparse.Namespace, out_dir: Path, remote_dir: str) -> None:
    local_trace, trace_duration_s = prepare_dynamic_trace(args, out_dir)
    remote_trace = f"{remote_dir}/{local_trace.name}"
    push(args.serial, local_trace, remote_trace)
    budget_trace_sha256 = hashlib.sha256(
        local_trace.read_bytes()).hexdigest()
    remote_budget_trace_sha256 = remote_sha256(
        args.serial, remote_trace, min(args.timeout, 60))
    if remote_budget_trace_sha256 != budget_trace_sha256:
        raise RuntimeError(
            "remote dynamic-budget trace SHA-256 mismatch: "
            f"local={budget_trace_sha256} "
            f"remote={remote_budget_trace_sha256}")

    summary_rows: list[dict[str, object]] = (
        read_csv_rows(out_dir / "dynamic_summary.csv") if args.resume else [])
    run_rows: list[dict[str, object]] = (
        read_csv_rows(out_dir / "dynamic_runs.csv") if args.resume else [])
    bin_rows: list[dict[str, object]] = (
        read_csv_rows(out_dir / "dynamic_budget_bins.csv") if args.resume else [])
    completed = {
        str(row["run_id"]) for row in run_rows
        if str(row.get("returncode", "")) == "0" and str(row.get("n_decode", "")) not in ("", "0")
        and str(row.get("valid_device_idle", "")).lower() == "true"
        and str(row.get("valid_device_awake", "")).lower() == "true"
        and str(row.get("valid_runtime", "")).lower() == "true"
        and str(row.get(
            "decode_wall_contract_valid", "")).lower() == "true"
        and str(row.get(
            "decode_phase_contract_valid", "")).lower() == "true"
        and str(row.get("valid_token_sequence", "")).lower() == "true"
        and (
            not args.require_device_idle
            or (
                int(row.get("measurement_monitor_samples") or 0) > 0
                and int(row.get(
                    "measurement_monitor_failures") or 0) == 0
                and str(row.get(
                    "monitor_thread_incomplete", "false")
                ).lower() != "true"
                and row.get("measurement_llama_pid") not in (None, "")
            )
        )
        and row.get("remote_binary_sha256")
            == args.remote_binary_sha256
        and row.get("remote_model_sha256")
            == args.remote_model_sha256
        and row.get("budget_trace_sha256") == budget_trace_sha256
        and row.get("remote_budget_trace_sha256")
            == budget_trace_sha256
    }
    raw_dir = out_dir / "raw"
    tokens_dir = out_dir / "tokens"
    raw_dir.mkdir(exist_ok=True)
    tokens_dir.mkdir(exist_ok=True)
    for repeat in range(args.repeats):
        for mode in MODE_ORDERS[repeat % len(MODE_ORDERS)]:
            if mode not in args.modes:
                continue
            run_id = f"{args.backend}_dynamic_r{repeat}_{mode}"
            if run_id in completed:
                print(json.dumps({"run_id": run_id, "resume_skip": True}), flush=True)
                continue
            if args.reboot_between_runs:
                reboot_device_and_wait(args, run_id)
            remote_tokens = f"{remote_dir}/{run_id}_tokens.csv"
            adb(args.serial, "shell", f"rm -f {shlex.quote(remote_tokens)}", timeout=30)
            wait_for_start_temperature(args, run_id)
            performance_mode_rearmed = rearm_fixed_performance_mode(args)
            before = thermal_snapshot(args.serial)
            battery_before = battery_level_pct(args.serial)
            power_before = device_power_state(args.serial)
            # A previous interrupted/contaminated attempt with this id is
            # replaced on resume; successful clean rows were skipped above.
            run_rows = [row for row in run_rows if str(row.get("run_id")) != run_id]
            summary_rows = [row for row in summary_rows if str(row.get("run_id")) != run_id]
            bin_rows = [row for row in bin_rows if str(row.get("run_id")) != run_id]
            command = remote_command(args, mode, remote_trace, remote_tokens)
            start = time.monotonic()
            performance_restore_error = ""
            try:
                proc, external_overlap, power_violations, frequency = run_with_interference_monitor(
                    args, command)
            finally:
                if performance_mode_rearmed:
                    disabled = set_fixed_performance_mode(args.serial, False)
                    if disabled.returncode != 0:
                        performance_restore_error = disabled.stdout.strip()
            wall = time.monotonic() - start
            after = thermal_snapshot(args.serial)
            battery_after = battery_level_pct(args.serial)
            power_after = device_power_state(args.serial)
            (raw_dir / f"{run_id}.log").write_text(proc.stdout, encoding="utf-8")
            local_tokens = tokens_dir / f"{run_id}.csv"
            pull = adb(args.serial, "pull", remote_tokens, str(local_tokens), timeout=60)
            parsed = parse_log(proc.stdout)
            run_row: dict[str, object] = {
                "run_id": run_id, "backend": args.backend, "repeat": repeat,
                "mode": mode, "pipeline": args.pipeline,
                "cpu_pipeline_staging": args.cpu_pipeline_staging,
                "cpu_staged_baseline": args.cpu_staged_baseline,
                "pipeline_lookahead": args.pipeline_lookahead,
                "pipeline_lookahead_mib": args.pipeline_lookahead_mib,
                "pipeline_max_pending": args.pipeline_max_pending,
                "pipeline_max_pending_mib": args.pipeline_max_pending_mib,
                "pipeline_load_cpu": args.pipeline_load_cpu,
                "pipeline_prepare_cpu": args.pipeline_prepare_cpu,
                "pipeline_copy_cpu": args.pipeline_copy_cpu,
                "cut_dual_compute": args.cut_dual_compute,
                "performance_mode_rearmed": performance_mode_rearmed,
                "performance_mode_restore_error": performance_restore_error,
                "returncode": proc.returncode, "wall_s": wall,
                "trace_duration_s": trace_duration_s,
                "measurement_duration_s": args.duration_s or trace_duration_s,
                "budget_trace_sha256": budget_trace_sha256,
                "remote_budget_trace_sha256":
                    remote_budget_trace_sha256,
                "thermal_before": before, "thermal_after": after,
                "battery_level_pct_before": battery_before,
                "battery_level_pct_after": battery_after,
                "device_power_before": power_before,
                "device_power_after": power_after,
                "valid_device_idle": not external_overlap,
                "external_overlap": " | ".join(external_overlap),
                "valid_device_awake": not power_violations,
                "power_state_violation": " | ".join(power_violations),
            }
            run_row.update(input_lineage(args))
            run_row.update(parsed)
            run_row.update(frequency)
            token_count, token_hash = token_sequence(local_tokens)
            run_row["token_trace_pull_returncode"] = pull.returncode
            run_row["token_id_count"] = token_count
            run_row["token_ids_sha256"] = token_hash
            prefix_count, prefix_hash = token_sequence(
                local_tokens, args.dynamic_token_prefix_count)
            run_row["token_prefix_count"] = prefix_count
            run_row["token_prefix_sha256"] = prefix_hash
            token_rows: list[dict[str, str]] = []
            if pull.returncode == 0 and local_tokens.exists():
                with local_tokens.open(newline="", encoding="utf-8") as handle:
                    token_rows = list(csv.DictReader(handle))
            decode_times = [
                float(token["t_sec"]) for token in token_rows
                if int(token["n_tokens"]) == 1
            ]
            trace_coverage_s = max(decode_times) if decode_times else 0.0
            required_coverage_s = float(args.duration_s or trace_duration_s)
            # A time-based trace must not silently pass when generation stops
            # early (for example, after exhausting --ctx-size).  Allow only a
            # small end-of-window scheduling tolerance.
            coverage_tolerance_s = max(2.0, 0.01 * required_coverage_s)
            run_row["trace_coverage_s"] = trace_coverage_s
            run_row["required_trace_coverage_s"] = required_coverage_s
            valid_runtime, runtime_reasons = validate_pipeline_run(args, mode, run_row)
            if performance_restore_error:
                valid_runtime = False
                runtime_reasons.append(
                    "failed to restore fixed-performance mode: "
                    f"{performance_restore_error}")
            if trace_coverage_s + coverage_tolerance_s < required_coverage_s:
                valid_runtime = False
                runtime_reasons.append(
                    f"token trace covered only {trace_coverage_s:.3f}s of "
                    f"{required_coverage_s:.3f}s")
            run_row["valid_runtime"] = valid_runtime
            run_row["runtime_invalid_reason"] = " | ".join(runtime_reasons)
            if (proc.returncode == 0 and pull.returncode == 0 and local_tokens.exists()
                    and not external_overlap and not power_violations and valid_runtime):
                by_budget: dict[int, list[float]] = {}
                by_bin: dict[int, list[float]] = {}
                decode_values: list[float] = []
                previous_budget: int | None = None
                previous_ratio_bin: int | None = None
                for token in token_rows:
                    if int(token["n_tokens"]) != 1:
                        continue
                    t_sec = float(token["t_sec"])
                    if t_sec < args.dynamic_warmup_s:
                        continue
                    if args.duration_s > 0 and t_sec > args.duration_s + 1.0:
                        continue
                    budget = int(token["budget_mib"])
                    latency = float(token["latency_ms"])
                    decode_values.append(latency)
                    weight_ratio = 100.0 * max(
                        0.0, budget - args.kv_mib - args.misc_mib) / args.weight_mib
                    step = args.dynamic_bin_step_pct
                    ratio_bin = min(
                        args.dynamic_max_ratio,
                        max(args.dynamic_min_ratio,
                            int(round(weight_ratio / step) * step)))
                    # Exact-budget plateaus still require an unchanged budget.
                    # For a continuous real trace, however, small changes within
                    # the same ratio band are the regime being measured. Exclude
                    # only tokens that cross a bin boundary instead of dropping
                    # every token whose integer-MiB budget changed.
                    if previous_ratio_bin is not None and ratio_bin == previous_ratio_bin:
                        by_bin.setdefault(ratio_bin, []).append(latency)
                    if previous_budget is not None and budget == previous_budget:
                        by_budget.setdefault(budget, []).append(latency)
                    previous_budget = budget
                    previous_ratio_bin = ratio_bin
                run_row.update({
                    "n_decode": len(decode_values),
                    "median_latency_ms": statistics.median(decode_values) if decode_values else None,
                    "p95_latency_ms": percentile(decode_values, 95),
                    "mean_latency_ms": statistics.mean(decode_values) if decode_values else None,
                    "decode_tokens_per_s": (1000.0 / statistics.mean(decode_values)) if decode_values else None,
                })
                for budget, values in sorted(by_budget.items()):
                    summary_rows.append({
                        "run_id": run_id, "backend": args.backend, "repeat": repeat,
                        "mode": mode, "budget_mib": budget, "n_decode": len(values),
                        "median_latency_ms": statistics.median(values),
                        "mean_latency_ms": statistics.mean(values), "wall_s": wall,
                    })
                for ratio_bin, values in sorted(by_bin.items()):
                    bin_rows.append({
                        "run_id": run_id, "backend": args.backend, "repeat": repeat,
                        "mode": mode, "ratio_bin_pct": ratio_bin, "n_decode": len(values),
                        "median_latency_ms": statistics.median(values),
                        "p95_latency_ms": percentile(values, 95),
                        "mean_latency_ms": statistics.mean(values),
                    })
            run_rows.append(run_row)
            mark_dynamic_token_correctness(run_rows, set(args.modes))
            write_csv(out_dir / "dynamic_runs.csv", run_rows)
            write_csv(out_dir / "dynamic_summary.csv", summary_rows)
            write_csv(out_dir / "dynamic_budget_bins.csv", bin_rows)
            write_csv(out_dir / "dynamic_aggregate.csv", summarize_dynamic_runs(run_rows))
            write_csv(out_dir / "dynamic_budget_bin_aggregate.csv", summarize_dynamic_bins(bin_rows))
            print(json.dumps({"run_id": run_id, "returncode": proc.returncode,
                              "wall_s": wall, "n_decode": run_row.get("n_decode"),
                              "median_latency_ms": run_row.get("median_latency_ms"),
                              "p95_latency_ms": run_row.get("p95_latency_ms"),
                              "valid_device_idle": run_row.get("valid_device_idle"),
                              "external_overlap": run_row.get("external_overlap"),
                              "valid_device_awake": run_row.get("valid_device_awake"),
                              "valid_runtime": run_row.get("valid_runtime"),
                              "runtime_invalid_reason": run_row.get("runtime_invalid_reason"),
                              "valid_token_sequence": run_row.get("valid_token_sequence"),
                              "power_state_violation": run_row.get("power_state_violation")},
                             ensure_ascii=False), flush=True)
            if args.cooldown_s > 0:
                time.sleep(args.cooldown_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="3C15AU002CL00000")
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--remote-binary", default="/data/local/tmp/hyzheng/elastic/granularity/llama-cli")
    parser.add_argument("--remote-model", default="/data/local/tmp/hyzheng/elastic/Llama-3.2-3B-Instruct-q4_0.gguf")
    parser.add_argument(
        "--expected-binary-sha256", default="",
        help="fail unless the remote llama-cli has this SHA-256")
    parser.add_argument(
        "--expected-model-sha256", default="",
        help="fail unless the remote GGUF has this SHA-256")
    parser.add_argument("--remote-lib-dir", default="/data/local/tmp/hyzheng/elastic")
    parser.add_argument("--remote-work-dir", default="/data/local/tmp/hyzheng/elastic/granularity")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-mib", type=float, default=1825.40)
    parser.add_argument("--ratios", default="30,40,50,60,70,80,90")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES),
                        help="granularity modes to run (default: multi multi_fused tensor cut)")
    parser.add_argument("--n-predict", type=int, default=8)
    parser.add_argument("--fixed-warmup-decode-tokens", type=int, default=0,
                        help="discard this many initial decode tokens from fixed-budget latency")
    parser.add_argument("--fixed-min-measure-tokens", type=int, default=1,
                        help="minimum post-warmup decode tokens required for a valid fixed run")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-repack-threads", type=int, default=1,
                        help="native CPU layout-repack threads per PREPARE unit")
    parser.add_argument("--cpu-mask", default="",
                        help="llama threadpool affinity mask, e.g. 0xfc on OP13")
    parser.add_argument("--cpu-strict", type=int, choices=(0, 1), default=1)
    parser.add_argument("--poll", type=int, default=50,
                        help="llama threadpool busy-poll percentage")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--ubatch", type=int, default=32)
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-eos", action="store_true",
                        help="continue time-based trace replay after an EOG token")
    parser.add_argument("--kv-mib", type=int, default=64)
    parser.add_argument("--misc-mib", type=int, default=128)
    parser.add_argument("--pin", default="norm,k,v")
    parser.add_argument("--expected-pin-token-embd-mib", type=float, default=0)
    parser.add_argument("--expected-pin-output-mib", type=float, default=0)
    parser.add_argument("--multi-tensors", type=int, default=2)
    parser.add_argument("--cut-parts", type=int, default=2)
    parser.add_argument(
        "--cut-dual-compute", choices=("off", "queue", "fused"),
        default="off",
        help=("GPU Cut decode compute: off, paired queues, or one fused "
              "two-cut GEMV dispatch"))
    parser.add_argument(
        "--gpu-unit-sync", choices=("none", "flush", "finish"), default="flush",
        help=("OpenCL logical-unit completion: nonblocking flush (default), "
              "none, or legacy blocking finish diagnostic"))
    parser.add_argument("--pipeline", choices=("off", "on"), default="off",
                        help="enable granularity-aware load/prepare/compute overlap")
    parser.add_argument("--pipeline-lookahead", type=int, default=1,
                        help="fallback unit-count lookahead when byte lookahead is zero")
    parser.add_argument("--pipeline-lookahead-mib", type=int, default=64,
                        help="equal future working-set byte window for every granularity")
    parser.add_argument("--pipeline-graph-lookahead", type=int, default=4,
                        help="maximum backend-graph hops in the rolling byte window")
    parser.add_argument("--pipeline-max-pending", type=int, default=256,
                        help="non-binding safety cap; byte lookahead is the primary fair limit")
    parser.add_argument("--pipeline-max-pending-mib", type=int, default=128,
                        help="OpenCL host/in-flight staging cap")
    parser.add_argument("--require-real-batch-io", action="store_true",
                        help="invalidate Multi if io_uring batch falls back to sequential I/O")
    parser.add_argument("--cpu-pipeline-staging", choices=("pooled", "direct"),
                        default="pooled",
                        help="CPU pipeline LOAD buffer: reusable pool or direct final-buffer read")
    parser.add_argument("--cpu-staged-baseline", action="store_true",
                        help="pipeline-off uses the identical pooled LOAD/PREPARE path synchronously")
    parser.add_argument("--pipeline-load-cpu", type=int, default=-1,
                        help="pin CPU pipeline LOAD worker to this logical CPU")
    parser.add_argument("--pipeline-prepare-cpu", type=int, default=-1,
                        help="pin CPU pipeline PREPARE worker to this logical CPU")
    parser.add_argument("--pipeline-copy-cpu", type=int, default=-1,
                        help="enable pooled dual-lane copy and pin its persistent helper")
    parser.add_argument("--pipeline-copy-min-kib", type=int, default=1024,
                        help="minimum pooled buffer size for dual-lane copy")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--cooldown-s", type=float, default=0)
    parser.add_argument("--max-cpu-start-c", type=float, default=0,
                        help="wait until the hottest CPU sensor is at or below this value")
    parser.add_argument("--max-gpu-start-c", type=float, default=0,
                        help="wait until the hottest GPU sensor is at or below this value")
    parser.add_argument("--max-skin-start-c", type=float, default=0,
                        help="wait until the skin sensor is at or below this value")
    parser.add_argument("--thermal-poll-s", type=float, default=10)
    parser.add_argument("--thermal-wait-timeout-s", type=float, default=1200)
    parser.add_argument("--resume", action="store_true",
                        help="skip successful dynamic run ids already present in the output")
    parser.add_argument("--require-device-idle", action="store_true",
                        help="wait for no other llama processes and record mid-run overlap")
    parser.add_argument("--device-idle-poll-s", type=float, default=5)
    parser.add_argument("--min-mem-available-mib", type=float, default=0,
                        help="wait for this much real MemAvailable before starting")
    parser.add_argument(
        "--min-battery-level-pct", type=int, default=0,
        help="wait until the physical device battery reaches this percentage")
    parser.add_argument(
        "--sleep-while-waiting-for-battery", action="store_true",
        help=("turn the display off while a battery gate is blocking, then "
              "wake and revalidate the device immediately before inference"))
    parser.add_argument("--min-cpu-freq-limit-khz", type=int, default=0,
                        help="invalidate a CPU run if thermal policy lowers scaling_max_freq")
    parser.add_argument(
        "--min-cpu-mean-freq-khz", type=int, default=0,
        help=("invalidate a CPU run when its measured mean active frequency "
              "is below this value"))
    parser.add_argument(
        "--min-cpu-median-sample-min-khz", type=int, default=0,
        help=("invalidate a CPU run when the median, across monitoring "
              "samples, of the slowest measured benchmark CPU is below this value"))
    parser.add_argument("--max-compute-call-ms", type=float, default=0,
                        help="flag known delegate-compute slow states above this full-run average")
    parser.add_argument("--compute-recovery-ratio", type=float, default=2.0,
                        help="accept a full-run compute flag when startup/steady latency proves recovery")
    parser.add_argument("--fixed-performance-mode", action="store_true",
                        help="enable Android fixed-performance mode for this sweep and restore it on exit")
    parser.add_argument("--keep-device-awake", action="store_true",
                        help="wake the USB-powered device, prevent screen-off during the sweep, and restore its prior state")
    parser.add_argument(
        "--reboot-between-runs", action="store_true",
        help=("reboot and wait for Android boot completion before every "
              "measured process to reset persistent scheduler/DVFS state"))
    parser.add_argument(
        "--boot-wait-timeout-s", type=float, default=300,
        help="maximum time to wait for a rebooted benchmark device")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--dynamic-points", default="0:90,3:30,9:60,15:90")
    parser.add_argument("--dynamic-trace", type=Path,
                        help="budget trace to replay")
    parser.add_argument(
        "--dynamic-trace-budget-mode", choices=("rescale", "native"),
        default="rescale",
        help=("rescale source min/max to --dynamic-min/max-ratio (default), "
              "or preserve model-matched absolute budget values with native"))
    parser.add_argument("--dynamic-trace-start-s", type=float, default=0,
                        help="first source-trace timestamp to retain; retained time is rebased to zero")
    parser.add_argument("--dynamic-trace-end-s", type=float, default=0,
                        help="last source-trace timestamp to retain; zero keeps the remainder")
    parser.add_argument("--dynamic-min-ratio", type=int, default=30)
    parser.add_argument("--dynamic-max-ratio", type=int, default=90)
    parser.add_argument("--dynamic-bin-step-pct", type=int, default=10,
                        help="weight-ratio width used by dynamic budget-bin summaries")
    parser.add_argument("--dynamic-warmup-s", type=float, default=0,
                        help="retain but exclude initial decode seconds from dynamic summaries")
    parser.add_argument("--dynamic-token-prefix-count", type=int, default=16,
                        help="common token-ID prefix length used for dynamic correctness")
    parser.add_argument("--duration-s", type=float, default=0,
                        help="stop generation after this many decode seconds")
    args = parser.parse_args()
    if args.backend == "gpu" and "multi_fused" in args.modes:
        parser.error(
            "OpenCL multi_fused is not implemented; refusing to label ordinary "
            "Multi execution as fused")
    if args.dynamic_bin_step_pct <= 0:
        parser.error("--dynamic-bin-step-pct must be positive")
    if args.pipeline_lookahead <= 0:
        parser.error("--pipeline-lookahead must be positive")
    if args.pipeline_graph_lookahead < 0:
        parser.error("--pipeline-graph-lookahead must be non-negative")
    if args.pipeline_max_pending <= 0 or args.pipeline_max_pending_mib <= 0:
        parser.error("pipeline pending limits must be positive")
    if args.pipeline_copy_min_kib < 0:
        parser.error("--pipeline-copy-min-kib must be non-negative")
    if args.cpu_repack_threads <= 0:
        parser.error("--cpu-repack-threads must be positive")
    if args.fixed_warmup_decode_tokens < 0 or args.fixed_min_measure_tokens <= 0:
        parser.error("fixed warmup must be non-negative and minimum tokens positive")
    if args.compute_recovery_ratio <= 1.0:
        parser.error("compute recovery ratio must be greater than one")
    if not 0 <= args.min_battery_level_pct <= 100:
        parser.error("--min-battery-level-pct must be between 0 and 100")
    if args.dynamic_warmup_s < 0:
        parser.error("dynamic warmup must be non-negative")
    if args.dynamic_trace_start_s < 0 or args.dynamic_trace_end_s < 0:
        parser.error("dynamic trace window bounds must be non-negative")
    if (args.dynamic_trace_end_s > 0 and
            args.dynamic_trace_end_s <= args.dynamic_trace_start_s):
        parser.error("dynamic trace end must be greater than its start")
    if args.dynamic_token_prefix_count <= 0:
        parser.error("dynamic token prefix count must be positive")
    for label, value in (
        ("--expected-binary-sha256", args.expected_binary_sha256),
        ("--expected-model-sha256", args.expected_model_sha256),
    ):
        if value and not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            parser.error(
                f"{label} must contain exactly 64 hexadecimal digits")
    args.expected_binary_sha256 = (
        args.expected_binary_sha256.lower())
    args.expected_model_sha256 = (
        args.expected_model_sha256.lower())

    # Serialize experiments per physical device.  The idle monitor cannot
    # distinguish two runners that intentionally use the exact same
    # binary/model paths once both remote processes are live, so acquire a
    # host-side advisory lock before touching device state.  The kernel
    # releases this lock automatically if the runner exits or is killed.
    safe_serial = re.sub(r"[^A-Za-z0-9_.-]", "_", args.serial)
    lock_path = Path("/tmp") / f"llama-elastic-sweep-{safe_serial}.lock"
    with lock_path.open("a+", encoding="utf-8") as device_lock:
        try:
            fcntl.flock(device_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another Elastic sweep already owns device {args.serial}; "
                f"lock={lock_path}") from exc
        device_lock.seek(0)
        device_lock.truncate()
        device_lock.write(
            f"pid={os.getpid()} output={args.output} started={time.time()}\n")
        device_lock.flush()

        args.output.mkdir(parents=True, exist_ok=True)
        args.remote_binary_sha256 = remote_sha256(
            args.serial, args.remote_binary, timeout=args.timeout)
        args.remote_model_sha256 = remote_sha256(
            args.serial, args.remote_model, timeout=args.timeout)
        if (
            args.expected_binary_sha256
            and args.remote_binary_sha256
                != args.expected_binary_sha256
        ):
            raise RuntimeError(
                "remote llama-cli SHA-256 mismatch: "
                f"expected={args.expected_binary_sha256} "
                f"actual={args.remote_binary_sha256}")
        if (
            args.expected_model_sha256
            and args.remote_model_sha256
                != args.expected_model_sha256
        ):
            raise RuntimeError(
                "remote model SHA-256 mismatch: "
                f"expected={args.expected_model_sha256} "
                f"actual={args.remote_model_sha256}")
        args.fixed_run_config_sha256 = fixed_run_config_sha256(args)
        metadata = vars(args).copy()
        for key, value in metadata.items():
            if isinstance(value, Path):
                metadata[key] = str(value)
        metadata["started_unix"] = time.time()
        metadata["device_lock_path"] = str(lock_path)
        metadata["initial_device_power_state"] = device_power_state(args.serial)
        metadata["device_props"] = adb(
            args.serial, "shell",
            "getprop ro.product.model; getprop ro.build.version.release; getprop ro.build.fingerprint",
            timeout=30,
        ).stdout
        (args.output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        adb(args.serial, "shell", f"mkdir -p {shlex.quote(args.remote_work_dir)}", timeout=30)
        awake_restore: dict[str, object] | None = None
        try:
            if args.keep_device_awake:
                awake_restore = enable_device_awake(args.serial)
            if args.dynamic:
                dynamic_sweep(args, args.output, args.remote_work_dir)
            else:
                fixed_sweep(args, args.output, args.remote_work_dir)
        finally:
            if args.fixed_performance_mode:
                set_fixed_performance_mode(args.serial, False)
            if awake_restore is not None:
                restore_device_awake(args.serial, awake_restore)


if __name__ == "__main__":
    main()
