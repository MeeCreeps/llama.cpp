#!/usr/bin/env python3
"""Run dynamic-budget planner baselines on Android and summarize decode speed.

Baselines:

* offline    : table CP-SAT plan, dynamic budget switches choose table bands.
* online     : remote CP-SAT, current residency state is sent to the host solver.
* mru        : native online MRU eviction baseline.
* static-min : one fixed plan built for the lowest budget bucket of each trace.

The runner creates low-memory 10-minute source windows from trace CSVs, optionally
replays them faster for practical phone experiments, pushes all needed artifacts,
runs llama-cli on the phone, and writes log/CSV/Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = (
    ROOT
    / ".wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_10min_baselines"
)
DEFAULT_MODEL_HOST_PATH = Path("/home/myid/hz85760/model/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_0.gguf")


def default_libomp_path() -> Path | None:
    candidates: list[Path] = []
    for env_name in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT"):
        ndk = os.environ.get(env_name)
        if ndk:
            candidates += sorted(Path(ndk).glob("toolchains/llvm/prebuilt/*/lib*/clang/*/lib/linux/aarch64/libomp.so"))
            candidates += sorted(Path(ndk).glob("toolchains/llvm/prebuilt/*/lib/clang/*/lib/linux/aarch64/libomp.so"))
    candidates += [
        Path("/home/myid/hz85760/env/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"),
        Path("/home/myid/hz85760/env/hexgon/6.5.0.1/tools/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7/lib/linux/aarch64/libomp.so"),
        Path("/home/myid/hz85760/ndk/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/19/lib/linux/aarch64/libomp.so"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def default_libcxx_path() -> Path | None:
    candidates: list[Path] = []
    for env_name in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT"):
        ndk = os.environ.get(env_name)
        if ndk:
            candidates += sorted(Path(ndk).glob("toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"))
    candidates += [
        Path("/home/myid/hz85760/env/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"),
        Path("/home/myid/hz85760/ndk/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@dataclass
class TraceWindow:
    source: Path
    local: Path
    remote_name: str
    source_start: float
    source_end: float
    source_span: float
    replay_span: float
    rows: int
    min_mib: float
    mean_mib: float
    max_mib: float
    min_bucket_mib: int
    max_bucket_mib: int


def run(cmd: list[str], *, check: bool = True, timeout: float | None = None, stdout: Any = subprocess.PIPE) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=check, timeout=timeout, text=True, stdout=stdout, stderr=subprocess.STDOUT)


def adb(serial: str, args: list[str], *, check: bool = True, timeout: float | None = None, stdout: Any = subprocess.PIPE) -> subprocess.CompletedProcess[str]:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    return run(cmd, check=check, timeout=timeout, stdout=stdout)


def adb_shell(serial: str, script: str, *, check: bool = True, timeout: float | None = None, stdout: Any = subprocess.PIPE) -> subprocess.CompletedProcess[str]:
    return adb(serial, ["shell", script], check=check, timeout=timeout, stdout=stdout)


def adb_shell_retry(
    serial: str,
    script: str,
    *,
    timeout: float,
    retries: int,
    check: bool = True,
    stdout: Any = subprocess.PIPE,
) -> subprocess.CompletedProcess[str]:
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return adb_shell(serial, script, check=check, timeout=timeout, stdout=stdout)
        except subprocess.TimeoutExpired as exc:
            last_error = exc
            print(f"adb shell timeout attempt={attempt}/{retries}: {script}", flush=True)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"adb shell failed attempt={attempt}/{retries}: rc={exc.returncode} {script}", flush=True)
            if check and attempt >= retries:
                raise
        time.sleep(1.0)
    if last_error:
        raise last_error
    raise RuntimeError(f"adb shell retry failed without exception: {script}")


def read_thermal_snapshot(serial: str, timeout: float) -> dict[str, Any]:
    proc = adb_shell(serial, "dumpsys thermalservice 2>/dev/null", check=False, timeout=timeout)
    text = proc.stdout or ""
    temps: dict[str, float] = {}
    for m in re.finditer(r"Temperature\{mValue=([-0-9.]+).*?mName=([^,}]+)", text):
        value = float(m.group(1))
        name = m.group(2)
        if -50.0 < value < 200.0:
            temps[name] = value

    cpu = [v for k, v in temps.items() if k.startswith("CPU")]
    gpu = [v for k, v in temps.items() if k.startswith("GPU")]
    skin = [v for k, v in temps.items() if "skin" in k.lower()]
    status = ""
    m = re.search(r"Thermal Status:\s*(\d+)", text)
    if m:
        status = int(m.group(1))
    return {
        "thermal_status": status,
        "thermal_cpu_max_c": max(cpu) if cpu else "",
        "thermal_gpu_max_c": max(gpu) if gpu else "",
        "thermal_skin_max_c": max(skin) if skin else "",
    }


def thermal_max(snapshot: dict[str, Any]) -> float:
    values = []
    for key in ("thermal_cpu_max_c", "thermal_gpu_max_c", "thermal_skin_max_c"):
        value = snapshot.get(key, "")
        if value != "":
            values.append(float(value))
    return max(values) if values else 0.0


def wait_for_cooldown(args: argparse.Namespace, trace_name: str, method: str) -> dict[str, Any]:
    snapshot = read_thermal_snapshot(args.adb_serial, args.adb_timeout_s)
    if args.cooldown_thermal_max_c <= 0:
        return snapshot

    deadline = time.monotonic() + args.cooldown_timeout_s
    while thermal_max(snapshot) > args.cooldown_thermal_max_c and time.monotonic() < deadline:
        print(
            "=== cooldown trace=%s method=%s thermal_max=%.1fC target=%.1fC status=%s ==="
            % (trace_name, method, thermal_max(snapshot), args.cooldown_thermal_max_c, snapshot.get("thermal_status", "")),
            flush=True,
        )
        time.sleep(args.cooldown_poll_s)
        snapshot = read_thermal_snapshot(args.adb_serial, args.adb_timeout_s)
    return snapshot


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def bucket_floor(value_mib: float, bucket_mib: int) -> int:
    return int(math.floor(value_mib / bucket_mib) * bucket_mib)


def bucket_ceil(value_mib: float, bucket_mib: int) -> int:
    return int(math.ceil(value_mib / bucket_mib) * bucket_mib)


def read_trace(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row.get("t_sec", row.get("time_sec", "0")) or 0)
            m = float(row.get("mem_available_mb", row.get("budget_mib", "0")) or 0)
            rows.append((t, m))
    rows.sort(key=lambda x: x[0])
    return rows


def select_low_window(rows: list[tuple[float, float]], window_sec: float, stride_sec: float) -> tuple[float, float]:
    if not rows:
        raise ValueError("empty trace")
    first = rows[0][0]
    last = rows[-1][0]
    if last - first <= window_sec:
        return first, last

    best_start = first
    best_score: tuple[float, float] | None = None
    t = first
    while t + window_sec <= last + 1e-9:
        vals = [m for ts, m in rows if t <= ts <= t + window_sec]
        if vals:
            score = (sum(vals) / len(vals), min(vals))
            if best_score is None or score < best_score:
                best_score = score
                best_start = t
        t += stride_sec
    return best_start, best_start + window_sec


def write_window_trace(
    source: Path,
    out_dir: Path,
    *,
    window_sec: float,
    stride_sec: float,
    replay_speedup: float,
    bucket_mib: int,
) -> TraceWindow:
    rows = read_trace(source)
    start, end = select_low_window(rows, window_sec, stride_sec)
    window_rows = [(t, m) for t, m in rows if start <= t <= end]
    if not window_rows:
        raise ValueError(f"no rows selected for {source}")
    replay_rows = [((t - start) / replay_speedup, m) for t, m in window_rows]
    stem = source.stem
    speed_tag = f"x{int(replay_speedup)}" if float(replay_speedup).is_integer() else f"x{replay_speedup:g}"
    out = out_dir / f"{stem}_10min_{speed_tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec", "mem_available_mb"])
        for t, m in replay_rows:
            writer.writerow([f"{t:.3f}".rstrip("0").rstrip("."), f"{m:.1f}"])

    vals = [m for _, m in window_rows]
    min_mib = min(vals)
    max_mib = max(vals)
    return TraceWindow(
        source=source,
        local=out,
        remote_name=out.name,
        source_start=start,
        source_end=end,
        source_span=end - start,
        replay_span=(end - start) / replay_speedup,
        rows=len(window_rows),
        min_mib=min_mib,
        mean_mib=sum(vals) / len(vals),
        max_mib=max_mib,
        min_bucket_mib=bucket_floor(min_mib, bucket_mib),
        max_bucket_mib=bucket_ceil(max_mib, bucket_mib),
    )


def build_budget_list(traces: list[TraceWindow], bucket_mib: int, extra_max_mib: int) -> list[int]:
    lo = min(t.min_bucket_mib for t in traces)
    hi = max(max(t.max_bucket_mib for t in traces), extra_max_mib)
    if lo <= 0:
        lo = bucket_mib
    return list(range(lo, hi + 1, bucket_mib))


def parse_log(path: Path, method: str) -> dict[str, Any]:
    text = path.read_text(errors="replace") if path.exists() else ""
    result: dict[str, Any] = {
        "status": "ok",
        "eval_ms_total": "",
        "eval_runs": "",
        "raw_ms_per_token": "",
        "decode_ms_per_token": "",
        "exec_ms_per_token": "",
        "remote_wall_ms": 0.0,
        "remote_server_ms": 0.0,
        "provider_get_ms_total": 0.0,
        "apply_ms_total": 0.0,
        "apply_count": 0,
        "evict_planned": 0,
        "load_planned": 0,
        "transfer_planned": 0,
        "xform_planned": 0,
        "anchor_fired": "",
        "anchor_load": "",
        "anchor_transfer": "",
        "anchor_xform": "",
        "anchor_failures": "",
        "reload_host_issue_ms": "",
        "reload_host_issue_ms_per_forward": "",
        "reload_calls": "",
        "direct_read_ms": "",
        "direct_read_ms_per_forward": "",
        "direct_read_calls": "",
        "direct_read_mb": "",
        "direct_read_mb_per_forward": "",
        "online_calls": "",
        "online_failures": "",
    }
    if "TIMEOUT" in text:
        result["status"] = "timeout"
    if "CANNOT LINK EXECUTABLE" in text or "No such file" in text or "failed" in text.lower() and "failures=0" not in text:
        result["status"] = "check_log"

    m = re.search(r"eval time =\s*([0-9.]+) ms /\s*(\d+) runs\s*\(\s*([0-9.]+) ms per token", text)
    if m:
        eval_total = float(m.group(1))
        runs = int(m.group(2))
        raw = float(m.group(3))
        result["eval_ms_total"] = eval_total
        result["eval_runs"] = runs
        result["raw_ms_per_token"] = raw
        result["decode_ms_per_token"] = raw
    else:
        result["status"] = "no_perf" if result["status"] == "ok" else result["status"]

    remote_pairs = re.findall(r"remote_wall_ms=([0-9.]+).*?server_solve_ms=([0-9.]+)", text)
    remote_wall = sum(float(a) for a, _ in remote_pairs)
    remote_server = sum(float(b) for _, b in remote_pairs)
    summary = re.search(r"remote_wall_ms=([0-9.]+)\s+remote_server_ms=([0-9.]+)", text)
    if summary:
        remote_wall = max(remote_wall, float(summary.group(1)))
        remote_server = max(remote_server, float(summary.group(2)))
    result["remote_wall_ms"] = remote_wall
    result["remote_server_ms"] = remote_server
    for m in re.finditer(r"provider_get_ms=([0-9.]+)\s+apply_ms=([0-9.]+)", text):
        result["provider_get_ms_total"] += float(m.group(1))
        result["apply_ms_total"] += float(m.group(2))
        result["apply_count"] += 1
    if result["eval_ms_total"] != "" and result["eval_runs"]:
        runs = int(result["eval_runs"])
        if runs > 0:
            # Execution-only metric requested for online CP-SAT experiments:
            # exclude provider lookup / remote solve time, but keep apply_ms
            # because plan application performs real residency changes.
            exec_total = max(
                0.0,
                float(result["eval_ms_total"])
                - float(result["provider_get_ms_total"])
                + float(result["apply_ms_total"]),
            )
            result["exec_ms_per_token"] = exec_total / runs

    for m in re.finditer(r"apply_exec_plan:.*?evict=(\d+).*?load=(\d+).*?transfer=(\d+).*?xform=(\d+)", text):
        result["evict_planned"] += int(m.group(1))
        result["load_planned"] += int(m.group(2))
        result["transfer_planned"] += int(m.group(3))
        result["xform_planned"] += int(m.group(4))

    m = re.search(r"elastic anchor summary:.*?fired=(\d+).*?load=(\d+).*?transfer=(\d+).*?xform=(\d+).*?failures=(\d+)", text)
    if m:
        result["anchor_fired"] = int(m.group(1))
        result["anchor_load"] = int(m.group(2))
        result["anchor_transfer"] = int(m.group(3))
        result["anchor_xform"] = int(m.group(4))
        result["anchor_failures"] = int(m.group(5))

    m = re.search(r"reload host-issue total:\s*([0-9.]+) ms", text)
    if m:
        result["reload_host_issue_ms"] = float(m.group(1))
    m = re.search(r"reload calls:\s*(\d+)", text)
    if m:
        result["reload_calls"] = int(m.group(1))
    m = re.search(r"direct O_DIRECT read calls=(\d+).*?total=([0-9.]+) ms.*?MB=([0-9.]+)", text)
    if m:
        result["direct_read_calls"] = int(m.group(1))
        result["direct_read_ms"] = float(m.group(2))
        result["direct_read_mb"] = float(m.group(3))
    if result["eval_runs"]:
        runs = int(result["eval_runs"])
        if runs > 0:
            if result["reload_host_issue_ms"] != "":
                result["reload_host_issue_ms_per_forward"] = float(result["reload_host_issue_ms"]) / runs
            if result["direct_read_ms"] != "":
                result["direct_read_ms_per_forward"] = float(result["direct_read_ms"]) / runs
            if result["direct_read_mb"] != "":
                result["direct_read_mb_per_forward"] = float(result["direct_read_mb"]) / runs
    m = re.search(r"\[elastic-online\] calls=(\d+) failures=(\d+)", text)
    if m:
        result["online_calls"] = int(m.group(1))
        result["online_failures"] = int(m.group(2))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(v: Any) -> str:
    if v == "":
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def write_markdown(path: Path, rows: list[dict[str, Any]], traces: list[TraceWindow], args: argparse.Namespace) -> None:
    lines = [
        "# Dynamic Budget 10-Min Baseline Matrix",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "## Configuration",
        "",
        f"- device: `{args.adb_serial}`",
        f"- remote dir: `{args.remote_dir}`",
        f"- model: `{args.model}`",
        f"- source window: `{args.window_sec}` seconds",
        f"- replay speedup: `{args.replay_speedup}x`",
        f"- bench seconds: `{args.bench_seconds}`",
        f"- n_predict fallback: `{args.n_pred}`",
        f"- budget bucket: `{args.bucket_mib}` MiB",
        f"- offline table: `{args.phone_plan_dir}`",
        f"- cooldown thermal max: `{args.cooldown_thermal_max_c}` C",
        "",
        "## Trace Windows",
        "",
        "| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in traces:
        lines.append(
            f"| {t.local.name} | {t.source_start:.0f} | {t.source_span:.0f} | {t.replay_span:.1f} | {t.rows} | "
            f"{t.min_mib:.1f} | {t.mean_mib:.1f} | {t.max_mib:.1f} | {t.min_bucket_mib} |"
        )
    lines += [
        "",
        "## Results",
        "",
        "`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.",
        "",
        "| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        planned = f"{r.get('evict_planned', 0)}/{r.get('load_planned', 0)}/{r.get('transfer_planned', 0)}/{r.get('xform_planned', 0)}"
        failures = r.get("anchor_failures", r.get("online_failures", ""))
        before = thermal_max(
            {
                "thermal_cpu_max_c": r.get("thermal_cpu_max_c_before", ""),
                "thermal_gpu_max_c": r.get("thermal_gpu_max_c_before", ""),
                "thermal_skin_max_c": r.get("thermal_skin_max_c_before", ""),
            }
        )
        after = thermal_max(
            {
                "thermal_cpu_max_c": r.get("thermal_cpu_max_c_after", ""),
                "thermal_gpu_max_c": r.get("thermal_gpu_max_c_after", ""),
                "thermal_skin_max_c": r.get("thermal_skin_max_c_after", ""),
            }
        )
        thermal = "n/a" if before == 0.0 and after == 0.0 else f"{before:.1f}/{after:.1f}"
        lines.append(
            f"| {r['trace']} | {r['method']} | {r['status']} | {fmt(r.get('raw_ms_per_token', ''))} | "
            f"{fmt(r.get('exec_ms_per_token', ''))} | {thermal} | {fmt(r.get('remote_wall_ms', ''))} | "
            f"{fmt(r.get('provider_get_ms_total', ''))} | {fmt(r.get('apply_count', ''))} | {planned} | {fmt(r.get('direct_read_ms', ''))} | "
            f"{fmt(r.get('direct_read_calls', ''))} | {fmt(failures)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def remote_shell_env(remote_dir: str, env: dict[str, str], argv: list[str]) -> str:
    exports = " ".join(f"export {k}={shell_quote(v)};" for k, v in env.items() if v is not None)
    return f"cd {shell_quote(remote_dir)} && {exports} " + " ".join(shell_quote(x) for x in argv)


def make_method_env(args: argparse.Namespace, method: str, trace: TraceWindow, remote_trace: str, work_dir: str) -> dict[str, str]:
    effective_safety_mib = args.safety_mib + args.pinned_extra_mib
    common = {
        "LD_LIBRARY_PATH": args.remote_dir,
        "GGML_OPENCL_DISABLE_ALLOC_HOST_PTR": "1",
        "GGML_OPENCL_USE_SVM": "1",
        "GGML_OPENCL_ELASTIC": "1",
        "GGML_ELASTIC_TIMING": "1",
        "GGML_ELASTIC_STAGE_DETAIL": "1",
        "GGML_ELASTIC_BUDGET_CSV": remote_trace,
        "GGML_ELASTIC_BUDGET_BUCKET_MB": str(args.bucket_mib),
        "GGML_ELASTIC_KV_MB": str(args.kv_mib),
        "GGML_ELASTIC_MISC_MB": str(args.misc_mib),
        "GGML_ELASTIC_SAFETY_MB": str(effective_safety_mib),
        "LLAMA_ELASTIC_DEFER_STAGE": "0",
    }
    if args.pinned_extra_mib > 0:
        common["GGML_ELASTIC_PIN_UNPLANNED_OUTPUT_COUNTS_BUDGET"] = "1"
    use_interval_schedule = (
        args.use_interval_schedule == "1"
        or (args.use_interval_schedule == "auto" and args.cp_objective == "interval_makespan")
    )
    if use_interval_schedule:
        common["LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE"] = "1"
        common["LLAMA_ELASTIC_DEFER_STAGE"] = "1"
        if args.overlap_model == "none":
            common["LLAMA_ELASTIC_INTERVAL_STAGE_KINDS"] = "none"
            common["GGML_ELASTIC_ASYNC_STAGE_LOAD"] = "0"
            common["GGML_ELASTIC_ASYNC_STAGE_PREPARE"] = "0"
            common["GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD"] = "0"
            common["GGML_ELASTIC_RELOAD_ON_XFER"] = "0"
            common["GGML_ELASTIC_XFER_EXTRA"] = "0"
        else:
            common.setdefault("GGML_ELASTIC_ASYNC_STAGE_LOAD", "1")
            common.setdefault("GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD", str(max(4, args.prefetch_distance)))
            common.setdefault("GGML_ELASTIC_RELOAD_ON_XFER", "1")
            common.setdefault("GGML_ELASTIC_XFER_EXTRA", "1")
            common.setdefault("LLAMA_ELASTIC_PREPARE_LEAD_OPS", "16")
            common.setdefault("GGML_ELASTIC_ASYNC_XFORM_MAX_PENDING", "128")
        common.setdefault("GGML_ELASTIC_NO_AUTO_EVICT", "1")
        common.setdefault("LLAMA_ELASTIC_ANCHOR_REPEAT_PER_GRAPH", "1")
        common.setdefault("LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH", "0")
        common.setdefault("LLAMA_ELASTIC_INTERVAL_INCLUDE_TRANSITIONS", "0")
        common.setdefault("GGML_ELASTIC_RELEASE_STAGE_AFTER_XFORM", "0")
        common.setdefault("GGML_ELASTIC_CACHE_FOREGROUND_LOAD", "1")
        common.setdefault("GGML_ELASTIC_SOA_STAGING_SLOTS", "4")
        common.setdefault("GGML_ELASTIC_CL_RETAIN", "1")
        common.setdefault("GGML_ELASTIC_CL_RETAIN_MB", "1024")
        if args.interval_stage_kinds and args.overlap_model != "none":
            common["LLAMA_ELASTIC_INTERVAL_STAGE_KINDS"] = str(args.interval_stage_kinds)
            stage_kinds = {s.strip().lower() for s in str(args.interval_stage_kinds).split(",") if s.strip()}
            if "prepare" in stage_kinds or "all" in stage_kinds:
                common.setdefault("GGML_ELASTIC_ASYNC_STAGE_PREPARE", "1")
                common.setdefault("LLAMA_ELASTIC_ENABLE_CPU_XFORM_STAGE", "1")
    if method in {"offline", "online", "mru", "candidate-select", "diff-graph-expand"}:
        common["GGML_ELASTIC_DYNAMIC"] = "1"
    if method == "offline":
        common["LLAMA_ELASTIC_DIR"] = args.phone_plan_dir
    elif method == "static-min":
        common.update(
            {
                "LLAMA_ELASTIC_APPLY": f"{args.phone_plan_dir}/plan_{trace.min_bucket_mib}MiB.json",
                "LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE": "0",
                "LLAMA_ELASTIC_DEFER_STAGE": "0",
                "GGML_ELASTIC_ASYNC_STAGE_LOAD": "0",
                "GGML_ELASTIC_ASYNC_STAGE_PREPARE": "0",
                "GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD": "0",
                "GGML_ELASTIC_RELOAD_ON_XFER": "0",
                "GGML_ELASTIC_XFER_EXTRA": "0",
            }
        )
    elif method == "online":
        common.update(
            {
                "LLAMA_ELASTIC_ONLINE": "1",
                "LLAMA_ELASTIC_ONLINE_MODE": "remote",
                "LLAMA_ELASTIC_ONLINE_REMOTE_URL": f"http://127.0.0.1:{args.port}/solve",
                "LLAMA_ELASTIC_MODEL_META": args.phone_model_meta,
                "LLAMA_ELASTIC_COST_DIR": args.phone_cost_dir,
                "LLAMA_ELASTIC_ONLINE_WORK_DIR": work_dir,
                "LLAMA_ELASTIC_ONLINE_KV_MB": str(args.kv_mib),
                "LLAMA_ELASTIC_ONLINE_MISC_MB": str(args.misc_mib),
                "LLAMA_ELASTIC_ONLINE_SAFETY_MB": str(effective_safety_mib),
                "LLAMA_ELASTIC_ONLINE_TIME_LIMIT_MS": str(args.time_limit_ms),
                "LLAMA_ELASTIC_PREFETCH_DISTANCE": str(args.prefetch_distance),
                "LLAMA_ELASTIC_TRANSITION_WEIGHT": str(args.transition_weight),
                "LLAMA_ELASTIC_DISK_RELOAD_MULTIPLIER": str(args.disk_reload_multiplier),
                "LLAMA_ELASTIC_DISK_GPU_RELOAD_MULTIPLIER": str(args.disk_gpu_reload_multiplier),
                "LLAMA_ELASTIC_OVERLAP_MODEL": str(args.overlap_model),
                "LLAMA_ELASTIC_CP_OBJECTIVE": str(args.cp_objective),
                "LLAMA_ELASTIC_ALLOWED_PLACEMENTS": str(args.allowed_placements),
            }
        )
    elif method in {"candidate-select", "diff-graph-expand"}:
        common.update(
            {
                "LLAMA_ELASTIC_ONLINE": "1",
                "LLAMA_ELASTIC_ONLINE_MODE": method,
                "LLAMA_ELASTIC_CANDIDATE_DIR": args.phone_plan_dir,
                "LLAMA_ELASTIC_MODEL_META": args.phone_model_meta,
                "LLAMA_ELASTIC_COST_DIR": args.phone_cost_dir,
                "LLAMA_ELASTIC_ONLINE_WORK_DIR": work_dir,
                "LLAMA_ELASTIC_ONLINE_KV_MB": str(args.kv_mib),
                "LLAMA_ELASTIC_ONLINE_MISC_MB": str(args.misc_mib),
                "LLAMA_ELASTIC_ONLINE_SAFETY_MB": str(effective_safety_mib),
                "LLAMA_ELASTIC_PREFETCH_DISTANCE": str(args.prefetch_distance),
                "LLAMA_ELASTIC_TRANSITION_WEIGHT": str(args.transition_weight),
                "LLAMA_ELASTIC_DISK_RELOAD_MULTIPLIER": str(args.disk_reload_multiplier),
                "LLAMA_ELASTIC_DISK_GPU_RELOAD_MULTIPLIER": str(args.disk_gpu_reload_multiplier),
                "LLAMA_ELASTIC_OVERLAP_MODEL": str(args.overlap_model),
                "LLAMA_ELASTIC_CP_OBJECTIVE": str(args.cp_objective),
                "LLAMA_ELASTIC_ALLOWED_PLACEMENTS": str(args.allowed_placements),
            }
        )
    elif method == "mru":
        common.update(
            {
                "LLAMA_ELASTIC_ONLINE": "1",
                "LLAMA_ELASTIC_ONLINE_MODE": "mru-cache",
                "LLAMA_ELASTIC_RUNTIME_MRU_CACHE": "1",
                "LLAMA_ELASTIC_MRU_EVICT_IMMEDIATE": "1",
                "LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE": "0",
                "LLAMA_ELASTIC_DEFER_STAGE": "0",
                "GGML_ELASTIC_ASYNC_STAGE_LOAD": "0",
                "GGML_ELASTIC_ASYNC_STAGE_PREPARE": "0",
                "GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD": "0",
                "GGML_ELASTIC_RELOAD_ON_XFER": "0",
                "GGML_ELASTIC_XFER_EXTRA": "0",
                "LLAMA_ELASTIC_MODEL_META": args.phone_model_meta,
                "LLAMA_ELASTIC_COST_DIR": args.phone_cost_dir,
                "LLAMA_ELASTIC_ONLINE_WORK_DIR": work_dir,
                "LLAMA_ELASTIC_ONLINE_KV_MB": str(args.kv_mib),
                "LLAMA_ELASTIC_ONLINE_MISC_MB": str(args.misc_mib),
                "LLAMA_ELASTIC_ONLINE_SAFETY_MB": str(effective_safety_mib),
                "LLAMA_ELASTIC_PREFETCH_DISTANCE": str(args.prefetch_distance),
                "LLAMA_ELASTIC_TRANSITION_WEIGHT": str(args.transition_weight),
                "LLAMA_ELASTIC_DISK_RELOAD_MULTIPLIER": str(args.disk_reload_multiplier),
                "LLAMA_ELASTIC_DISK_GPU_RELOAD_MULTIPLIER": str(args.disk_gpu_reload_multiplier),
                "LLAMA_ELASTIC_OVERLAP_MODEL": str(args.overlap_model),
                "LLAMA_ELASTIC_CP_OBJECTIVE": str(args.cp_objective),
                "LLAMA_ELASTIC_ALLOWED_PLACEMENTS": "cpu,disk_cpu",
            }
        )
    else:
        raise ValueError(f"unknown method: {method}")
    for item in args.extra_env:
        if "=" not in item:
            raise ValueError(f"--extra-env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--extra-env has empty key: {item}")
        common[key] = value
    return common


def start_remote_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen[str] | None:
    if "online" not in {m.strip() for m in args.methods.split(",") if m.strip()}:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    cmd = [
        sys.executable,
        str(PLAN_DIR / "remote_dynamic_budget_server.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--model-meta",
        str(args.model_meta),
        "--cost-dir",
        str(args.cost_dir),
        "--kv-mib",
        str(args.kv_mib),
        "--misc-mib",
        str(args.misc_mib),
        "--safety-mib",
        str(args.effective_safety_mib),
        "--time-limit-ms",
        str(args.time_limit_ms),
        "--prefetch-distance",
        str(args.prefetch_distance),
        "--transition-weight",
        str(args.transition_weight),
        "--disk-reload-multiplier",
        str(args.disk_reload_multiplier),
        "--disk-gpu-reload-multiplier",
        str(args.disk_gpu_reload_multiplier),
        "--overlap-model",
        str(args.overlap_model),
        "--cp-objective",
        str(args.cp_objective),
        "--allowed-placements",
        str(args.allowed_placements),
    ]
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT)
    time.sleep(1.0)
    adb(args.adb_serial, ["reverse", f"tcp:{args.port}", f"tcp:{args.port}"], check=False, timeout=args.adb_timeout_s)
    return proc


def methods_need_offline_table(methods: str) -> bool:
    selected = {m.strip() for m in methods.split(",") if m.strip()}
    return bool(selected & {"offline", "static-min", "candidate-select", "diff-graph-expand"})


def stop_remote_server(args: argparse.Namespace, proc: subprocess.Popen[str] | None) -> None:
    try:
        adb(args.adb_serial, ["reverse", "--remove", f"tcp:{args.port}"], check=False, timeout=args.adb_timeout_s)
    except subprocess.TimeoutExpired:
        print(f"adb reverse remove timed out for tcp:{args.port}", flush=True)
    if not proc:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 10-minute dynamic-budget baseline matrix on Android")
    ap.add_argument("--adb-serial", default="172.20.115.151:5555")
    ap.add_argument("--remote-dir", default="/data/local/tmp/hyzheng/elastic")
    ap.add_argument("--model", default="Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    ap.add_argument("--model-host-path", type=Path, default=DEFAULT_MODEL_HOST_PATH)
    ap.add_argument("--llama-cli", type=Path, default=ROOT / "build-android-llama/bin/llama-cli")
    ap.add_argument("--libomp-path", type=Path, default=None)
    ap.add_argument("--libcxx-path", type=Path, default=None)
    ap.add_argument("--model-meta", type=Path, default=ROOT / "runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json")
    ap.add_argument("--cost-dir", type=Path, default=ROOT / "runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01")
    ap.add_argument("--trace-glob", default="trace/traces_9g/trace_*.csv")
    ap.add_argument("--trace-filter", default="", help="substring filter for trace filenames")
    ap.add_argument("--methods", default="offline,online,mru,static-min")
    ap.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    ap.add_argument("--window-sec", type=float, default=600.0)
    ap.add_argument("--window-stride-sec", type=float, default=10.0)
    ap.add_argument("--replay-speedup", type=float, default=1.0)
    ap.add_argument("--bench-seconds", type=float, default=None, help="wall-clock decode duration per run; default = window-sec / replay-speedup")
    ap.add_argument("--bucket-mib", type=int, default=256)
    ap.add_argument("--extra-max-budget-mib", type=int, default=8192)
    ap.add_argument("--kv-mib", type=int, default=512)
    ap.add_argument("--misc-mib", type=int, default=256)
    ap.add_argument("--safety-mib", type=int, default=64)
    ap.add_argument("--pinned-extra-mib", type=int, default=410,
                    help="extra pinned non-planned model bytes counted inside the budget, e.g. output.weight for Llama-3 8B Q4_0")
    ap.add_argument("--time-limit-ms", type=int, default=250)
    ap.add_argument("--prefetch-distance", type=int, default=1)
    ap.add_argument("--transition-weight", type=float, default=0.1)
    ap.add_argument("--disk-reload-multiplier", type=float, default=1.0)
    ap.add_argument("--disk-gpu-reload-multiplier", type=float, default=4.0)
    ap.add_argument("--overlap-model", choices=("pipeline", "none"), default="pipeline")
    ap.add_argument("--cp-objective", choices=("resource_makespan", "interval_makespan", "sum"), default="resource_makespan")
    ap.add_argument("--allowed-placements", default="cpu,gpu,disk_cpu,disk_gpu",
                    help="comma-separated solver placement choices")
    ap.add_argument("--top-k", type=int, default=1,
                    help="number of candidate plans per budget for candidate-select / diff-graph-expand")
    ap.add_argument("--candidate-placement-specs", default="",
                    help="semicolon-separated allowed-placement specs for offline candidate diversity")
    ap.add_argument("--use-interval-schedule", choices=("auto", "0", "1"), default="auto",
                    help="whether runtime uses schedule.events anchors; auto enables it for interval_makespan")
    ap.add_argument("--interval-stage-kinds", default="load,prepare",
                    help="comma-separated interval stages to trigger at runtime, e.g. load or load,prepare")
    ap.add_argument("--extra-env", action="append", default=[],
                    help="additional Android env var as KEY=VALUE; may be repeated")
    ap.add_argument("--llama-extra-arg", action="append", default=[],
                    help="extra llama-cli argument appended verbatim; repeat for option/value pairs")
    ap.add_argument("--n-pred", type=int, default=96, help="used only when --bench-seconds 0 disables timed mode")
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prompt", default="Summarize dynamic elastic memory planning for mobile LLM inference.")
    ap.add_argument("--timeout-s", type=int, default=480)
    ap.add_argument("--adb-timeout-s", type=int, default=30)
    ap.add_argument("--adb-retries", type=int, default=2)
    ap.add_argument("--cooldown-thermal-max-c", type=float, default=0.0,
                    help="if >0, wait before each run until max CPU/GPU/skin temperature is below this value")
    ap.add_argument("--cooldown-poll-s", type=float, default=30.0)
    ap.add_argument("--cooldown-timeout-s", type=float, default=1800.0)
    ap.add_argument("--port", type=int, default=18082)
    ap.add_argument("--no-resume", action="store_true", help="rerun rows already present in summary/results.csv")
    ap.add_argument("--offline-chain-state", action="store_true",
                    help="build offline budget table with previous bucket as synthetic state")
    ap.add_argument("--skip-build-table", action="store_true")
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--skip-push-binary", action="store_true")
    ap.add_argument("--skip-push-model", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.artifact_root = args.artifact_root.resolve()
    trace_dir = args.artifact_root / "traces"
    log_dir = args.artifact_root / "logs"
    summary_dir = args.artifact_root / "summary"
    table_dir = args.artifact_root / "offline_table"
    args.phone_plan_dir = f"{args.remote_dir}/plans_matrix10min"
    args.phone_model_meta = f"{args.remote_dir}/model_meta_matrix10min.json"
    args.phone_cost_dir = f"{args.remote_dir}/cost_matrix10min"
    if args.libomp_path is None:
        args.libomp_path = default_libomp_path()
    if args.libcxx_path is None:
        args.libcxx_path = default_libcxx_path()
    if args.bench_seconds is None:
        args.bench_seconds = args.window_sec / args.replay_speedup
    args.effective_safety_mib = args.safety_mib + args.pinned_extra_mib

    source_name_re = re.compile(r"^trace_\d+_user_\d+\.csv$")
    sources = sorted((ROOT / p).resolve() for p in Path(ROOT).glob(args.trace_glob))
    sources = [p for p in sources if source_name_re.match(p.name)]
    if args.trace_filter:
        sources = [p for p in sources if args.trace_filter in p.name]
    if not sources:
        raise SystemExit(f"no traces matched {args.trace_glob!r}")

    traces = [
        write_window_trace(
            p,
            trace_dir,
            window_sec=args.window_sec,
            stride_sec=args.window_stride_sec,
            replay_speedup=args.replay_speedup,
            bucket_mib=args.bucket_mib,
        )
        for p in sources
    ]
    budgets = build_budget_list(traces, args.bucket_mib, args.extra_max_budget_mib)

    need_offline_table = methods_need_offline_table(args.methods)
    if not args.skip_build_table and need_offline_table:
        cmd = [
            sys.executable,
            str(PLAN_DIR / "build_offline_budget_table.py"),
            "--model-meta",
            str(args.model_meta),
            "--cost-dir",
            str(args.cost_dir),
            "--out-dir",
            str(table_dir),
            "--budgets",
            ",".join(str(b) for b in budgets),
            "--kv-mib",
            str(args.kv_mib),
            "--misc-mib",
            str(args.misc_mib),
            "--safety-mib",
            str(args.effective_safety_mib),
            "--time-limit-ms",
            str(args.time_limit_ms),
            "--prefetch-distance",
            str(args.prefetch_distance),
            "--transition-weight",
            str(args.transition_weight),
            "--disk-reload-multiplier",
            str(args.disk_reload_multiplier),
            "--disk-gpu-reload-multiplier",
            str(args.disk_gpu_reload_multiplier),
            "--overlap-model",
            str(args.overlap_model),
            "--cp-objective",
            str(args.cp_objective),
            "--allowed-placements",
            str(args.allowed_placements),
            "--top-k",
            str(args.top_k),
        ]
        if args.candidate_placement_specs:
            cmd.extend(["--candidate-placement-specs", str(args.candidate_placement_specs)])
        if args.offline_chain_state:
            cmd.append("--chain-state")
        if not args.dry_run:
            run(cmd)
        else:
            print("+ " + " ".join(shlex.quote(c) for c in cmd))
    elif not need_offline_table:
        print("skip offline table build: selected methods do not require it", flush=True)

    if not args.skip_push and not args.dry_run:
        adb(args.adb_serial, ["devices"], timeout=args.adb_timeout_s)
        adb_shell_retry(args.adb_serial, f"mkdir -p {shell_quote(args.remote_dir)}", timeout=args.adb_timeout_s, retries=args.adb_retries)
        if not args.skip_push_binary:
            if not args.llama_cli.exists():
                raise SystemExit(f"llama-cli not found: {args.llama_cli}")
            adb(args.adb_serial, ["push", str(args.llama_cli), f"{args.remote_dir}/llama-cli"])
            adb_shell_retry(args.adb_serial, f"chmod 755 {shell_quote(args.remote_dir + '/llama-cli')}", timeout=args.adb_timeout_s, retries=args.adb_retries)
            if args.libomp_path and args.libomp_path.exists():
                adb(args.adb_serial, ["push", str(args.libomp_path), f"{args.remote_dir}/libomp.so"])
            else:
                print("warning: libomp.so not found; set --libomp-path if llama-cli links OpenMP dynamically", flush=True)
            if args.libcxx_path and args.libcxx_path.exists():
                adb(args.adb_serial, ["push", str(args.libcxx_path), f"{args.remote_dir}/libc++_shared.so"])
            else:
                print("warning: libc++_shared.so not found; set --libcxx-path if llama-cli links libc++ dynamically", flush=True)
        if not args.skip_push_model:
            if not args.model_host_path.exists():
                raise SystemExit(f"model host path not found: {args.model_host_path}")
            adb(args.adb_serial, ["push", str(args.model_host_path), f"{args.remote_dir}/{args.model}"])
        adb(args.adb_serial, ["push", str(args.model_meta), args.phone_model_meta])
        adb_shell_retry(
            args.adb_serial,
            f"rm -rf {shell_quote(args.phone_cost_dir)} && mkdir -p {shell_quote(args.phone_cost_dir)}",
            timeout=args.adb_timeout_s,
            retries=args.adb_retries,
        )
        adb(args.adb_serial, ["push", str(args.cost_dir) + "/.", args.phone_cost_dir + "/"])
        if need_offline_table:
            adb_shell_retry(
                args.adb_serial,
                f"rm -rf {shell_quote(args.phone_plan_dir)} && mkdir -p {shell_quote(args.phone_plan_dir)}",
                timeout=args.adb_timeout_s,
                retries=args.adb_retries,
            )
            adb(args.adb_serial, ["push", str(table_dir) + "/.", args.phone_plan_dir + "/"])
        for t in traces:
            adb(args.adb_serial, ["push", str(t.local), f"{args.remote_dir}/{t.remote_name}"])

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows: list[dict[str, Any]] = [] if args.no_resume else read_existing_csv(summary_dir / "results.csv")
    completed = {(r.get("trace", ""), r.get("method", "")) for r in rows}
    server = None
    try:
        if not args.dry_run:
            server = start_remote_server(args, log_dir / "remote_solver.log")
        for t in traces:
            for method in methods:
                timed_mode = args.bench_seconds and args.bench_seconds > 0
                log_tag = f"s{int(args.bench_seconds)}" if timed_mode and float(args.bench_seconds).is_integer() else (f"s{args.bench_seconds:g}" if timed_mode else f"n{args.n_pred}")
                work = f"{args.remote_dir}/online_matrix10min_{method}_{t.local.stem}_{log_tag}"
                if (t.local.name, method) in completed:
                    print(f"=== skip existing trace={t.local.name} method={method} ===", flush=True)
                    continue
                if method in {"online", "mru", "candidate-select", "diff-graph-expand"} and not args.dry_run:
                    adb_shell_retry(
                        args.adb_serial,
                        f"rm -rf {shell_quote(work)} && mkdir -p {shell_quote(work)}",
                        timeout=args.adb_timeout_s,
                        retries=args.adb_retries,
                    )
                env = make_method_env(args, method, t, f"{args.remote_dir}/{t.remote_name}", work)
                argv = [
                    "./llama-cli",
                    "-m",
                    f"{args.remote_dir}/{args.model}" if not args.model.startswith("/") else args.model,
                    "-p",
                    args.prompt,
                    "-n",
                    str(args.n_pred),
                    "-c",
                    str(args.ctx_size),
                    "-b",
                    str(args.batch),
                    "-ub",
                    str(args.batch),
                    "-t",
                    str(args.threads),
                    "--temp",
                    "0",
                    "--no-warmup",
                    "-ngl",
                    "99",
                    "-fa",
                    "on",
                    "-no-cnv",
                ]
                argv.extend(args.llama_extra_arg)
                if timed_mode:
                    env["LLAMA_ELASTIC_BENCH_SECONDS"] = f"{args.bench_seconds:.3f}".rstrip("0").rstrip(".")
                if timed_mode:
                    argv[argv.index("-n") + 1] = "-1"
                script = remote_shell_env(args.remote_dir, env, argv)
                log_path = log_dir / f"{method}_{t.local.stem}_{log_tag}.log"
                print(f"=== run trace={t.local.name} method={method} log={log_path} ===", flush=True)
                if args.dry_run:
                    print(script)
                    rc = 0
                    thermal_before = {"thermal_status": "", "thermal_cpu_max_c": "", "thermal_gpu_max_c": "", "thermal_skin_max_c": ""}
                    thermal_after = {"thermal_status": "", "thermal_cpu_max_c": "", "thermal_gpu_max_c": "", "thermal_skin_max_c": ""}
                else:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    thermal_before = wait_for_cooldown(args, t.local.name, method)
                    with log_path.open("w") as f:
                        f.write("[thermal-before] " + json.dumps(thermal_before, sort_keys=True) + "\n")
                        f.flush()
                        try:
                            proc = adb_shell(args.adb_serial, script, check=False, timeout=args.timeout_s, stdout=f)
                            rc = proc.returncode
                        except subprocess.TimeoutExpired:
                            rc = 124
                            f.write(f"\nTIMEOUT after {args.timeout_s}s\n")
                        thermal_after = read_thermal_snapshot(args.adb_serial, args.adb_timeout_s)
                        f.write("[thermal-after] " + json.dumps(thermal_after, sort_keys=True) + "\n")
                    print(f"=== done trace={t.local.name} method={method} rc={rc} ===", flush=True)
                parsed = parse_log(log_path, method) if not args.dry_run else {"status": "dry-run"}
                parsed.update({f"{k}_before": v for k, v in thermal_before.items()})
                parsed.update({f"{k}_after": v for k, v in thermal_after.items()})
                parsed.update(
                    {
                        "trace": t.local.name,
                        "source_trace": t.source.name,
                        "method": method,
                        "rc": rc,
                        "source_start_sec": t.source_start,
                        "source_span_sec": t.source_span,
                        "replay_span_sec": t.replay_span,
                        "min_mib": t.min_mib,
                        "mean_mib": t.mean_mib,
                        "max_mib": t.max_mib,
                        "min_bucket_mib": t.min_bucket_mib,
                        "log": str(log_path),
                    }
                )
                rows.append(parsed)
                completed.add((t.local.name, method))
                write_csv(summary_dir / "results.csv", rows)
                write_markdown(summary_dir / "SUMMARY.md", rows, traces, args)
    finally:
        if not args.dry_run:
            stop_remote_server(args, server)

    write_csv(summary_dir / "results.csv", rows)
    write_markdown(summary_dir / "SUMMARY.md", rows, traces, args)
    trace_meta = [t.__dict__ | {"source": str(t.source), "local": str(t.local)} for t in traces]
    (summary_dir / "trace_windows.json").write_text(json.dumps(trace_meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(summary_dir / "SUMMARY.md"), "results": str(summary_dir / "results.csv")}, indent=2))


if __name__ == "__main__":
    main()
