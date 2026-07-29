#!/usr/bin/env python3
"""Run dynamic-budget planner baselines on Android and summarize decode speed.

Baselines:

* offline    : table CP-SAT placement with fixed Tensor units (legacy baseline).
* online     : remote CP-SAT plus full mixed-frontier reoptimization.
* mru        : native online MRU eviction baseline.
* static-min : one fixed plan built for the lowest budget bucket of each trace.
* static-max : one fixed plan built for the highest budget bucket of each trace.
* offline-mixed  : offline placement plus an offline mixed-granularity plan.
* diff-tree-mixed: online placement plus bounded online unit split/merge.

The runner creates low-memory 10-minute source windows from trace CSVs, optionally
replays them faster for practical phone experiments, pushes all needed artifacts,
runs llama-cli on the phone, and writes log/CSV/Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
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


@dataclass
class StaticTrace:
    local: Path
    remote_name: str
    budget_mib: int


def run(
    cmd: list[str],
    *,
    check: bool = True,
    timeout: float | None = None,
    stdout: Any = subprocess.PIPE,
    announce: bool = True,
) -> subprocess.CompletedProcess[str]:
    if announce:
        print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=check, timeout=timeout, text=True, stdout=stdout, stderr=subprocess.STDOUT)


def adb(
    serial: str,
    args: list[str],
    *,
    check: bool = True,
    timeout: float | None = None,
    stdout: Any = subprocess.PIPE,
    announce: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = ["adb"]
    if serial:
        preferred = _ADB_PORT_BY_SERIAL.get(serial)
        ports = ([preferred] if preferred is not None else []) + [
            port for port in _ADB_PORTS if port != preferred
        ]
        for port in ports:
            probe = subprocess.run(
                ["adb", "-P", str(port), "-s", serial, "get-state"],
                text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False, timeout=10)
            if probe.returncode == 0 and probe.stdout.strip() == "device":
                _ADB_PORT_BY_SERIAL[serial] = port
                cmd += ["-P", str(port)]
                break
        cmd += ["-s", serial]
    cmd += args
    return run(
        cmd, check=check, timeout=timeout, stdout=stdout,
        announce=announce)


def adb_shell(
    serial: str,
    script: str,
    *,
    check: bool = True,
    timeout: float | None = None,
    stdout: Any = subprocess.PIPE,
    announce: bool = True,
) -> subprocess.CompletedProcess[str]:
    return adb(
        serial, ["shell", script], check=check, timeout=timeout,
        stdout=stdout, announce=announce)


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


def validate_remote_sha256(
    serial: str,
    remote_path: str,
    expected_sha256: str,
    *,
    label: str,
    timeout_s: float,
    retries: int,
) -> str:
    """Fail closed unless a remote experiment input has the expected bytes."""
    expected = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(
            f"invalid expected {label} SHA-256: {expected_sha256!r}")
    proc = adb_shell_retry(
        serial,
        f"sha256sum {shell_quote(remote_path)}",
        timeout=timeout_s,
        retries=retries,
    )
    match = re.search(r"\b([0-9a-fA-F]{64})\b", proc.stdout or "")
    if match is None:
        raise RuntimeError(
            f"could not read remote {label} SHA-256 for {remote_path}")
    actual = match.group(1).lower()
    if actual != expected:
        raise RuntimeError(
            f"remote {label} SHA-256 mismatch: "
            f"expected={expected} actual={actual} path={remote_path}")
    print(
        f"validated remote {label} sha256={actual} path={remote_path}",
        flush=True,
    )
    return actual


def read_thermal_snapshot(serial: str, timeout: float) -> dict[str, Any]:
    proc = adb_shell(
        serial,
        "dumpsys thermalservice 2>/dev/null; "
        "echo __ELASTIC_BATTERY__; dumpsys battery 2>/dev/null; "
        "echo __ELASTIC_POWER__; "
        "dumpsys power 2>/dev/null | grep -m 1 'mWakefulness='",
        check=False,
        timeout=timeout,
    )
    text = proc.stdout or ""
    thermal_text, _, device_text = text.partition("__ELASTIC_BATTERY__")
    battery_text, _, power_text = device_text.partition("__ELASTIC_POWER__")
    # Newer Android thermalservice dumps both a stale "Cached temperatures"
    # snapshot and the live HAL readings.  Mixing the two can leave a cached
    # sensor such as shell_skin in the set even when it is absent from the
    # current HAL snapshot, causing the clean-start gate to wait forever.
    # Prefer the explicitly live section when it is available; retain the
    # whole-output fallback for older Android versions.
    temperature_text = thermal_text
    current_marker = "Current temperatures from HAL:"
    if current_marker in thermal_text:
        temperature_text = thermal_text.split(current_marker, 1)[1]
        temperature_text = temperature_text.split(
            "Current cooling devices from HAL:", 1)[0]
    temps: dict[str, float] = {}
    for m in re.finditer(
            r"Temperature\{mValue=([-0-9.]+).*?mName=([^,}]+)",
            temperature_text):
        value = float(m.group(1))
        name = m.group(2)
        if -50.0 < value < 200.0:
            temps[name] = value

    cpu = [v for k, v in temps.items() if k.startswith("CPU")]
    gpu = [v for k, v in temps.items() if k.startswith("GPU")]
    skin = [v for k, v in temps.items() if "skin" in k.lower()]
    status = ""
    m = re.search(r"Thermal Status:\s*(\d+)", thermal_text)
    if m:
        status = int(m.group(1))
    battery_level: int | str = ""
    m = re.search(r"(?m)^\s*level:\s*(\d+)", battery_text)
    if m:
        battery_level = int(m.group(1))
    wakefulness = ""
    m = re.search(r"mWakefulness=([A-Za-z]+)", power_text)
    if m:
        wakefulness = m.group(1)
    return {
        "thermal_status": status,
        "thermal_cpu_max_c": max(cpu) if cpu else "",
        "thermal_gpu_max_c": max(gpu) if gpu else "",
        "thermal_skin_max_c": max(skin) if skin else "",
        "battery_level_pct": battery_level,
        "device_wakefulness": wakefulness,
    }


def thermal_max(snapshot: dict[str, Any]) -> float:
    values = []
    for key in ("thermal_cpu_max_c", "thermal_gpu_max_c", "thermal_skin_max_c"):
        value = snapshot.get(key, "")
        if value != "":
            values.append(float(value))
    return max(values) if values else 0.0


def thermal_status(snapshot: dict[str, Any]) -> int | None:
    value = snapshot.get("thermal_status", "")
    if value == "":
        return None
    return int(value)


def active_ffn_worker_pids(
    serial: str,
    pids: set[int],
    timeout: float,
    *,
    quiet: bool,
) -> set[int]:
    """Distinguish an active HTP workload from an idle resident endpoint."""
    if not pids:
        return set()
    pid_list = " ".join(str(pid) for pid in sorted(pids))
    snapshot = (
        "for p in " + pid_list + "; do "
        "[ -r /proc/$p/stat ] || continue; "
        "ticks=$(awk '{print $14+$15}' /proc/$p/stat); "
        "target=$(readlink /proc/$p/fd/1 2>/dev/null); "
        "bytes=$(stat -c %s \"$target\" 2>/dev/null || echo 0); "
        "echo"
    )
    script = (
        snapshot + " A $p $ticks $bytes; done; "
        "sleep 0.5; "
        + snapshot + " B $p $ticks $bytes; done"
    )
    proc = adb_shell(
        serial, script, check=False, timeout=max(timeout, 2.0),
        announce=not quiet)
    if proc.returncode != 0:
        # The workers were visible but their activity probe failed. Treat all
        # of them as active rather than silently clearing the idle gate.
        return set(pids)
    samples: dict[tuple[str, int], tuple[int, int]] = {}
    for line in (proc.stdout or "").splitlines():
        match = re.fullmatch(r"([AB])\s+(\d+)\s+(\d+)\s+(\d+)", line.strip())
        if match:
            samples[(match.group(1), int(match.group(2)))] = (
                int(match.group(3)), int(match.group(4)))
    active: set[int] = set()
    for pid in pids:
        before = samples.get(("A", pid))
        after = samples.get(("B", pid))
        # A process that vanished is no longer competing. If sampling a live
        # process failed, remain conservative and classify it as active.
        if before is None and after is None:
            continue
        if before is None or after is None:
            active.add(pid)
            continue
        tick_delta = after[0] - before[0]
        byte_delta = after[1] - before[1]
        if tick_delta > 0 or byte_delta > 0:
            active.add(pid)
    return active


def external_inference_processes(
    serial: str,
    timeout: float,
    *,
    quiet: bool = False,
) -> list[str]:
    proc = adb_shell(
        serial,
        "ps -A -o USER,PID,PPID,ARGS",
        check=False,
        timeout=timeout,
        announce=not quiet)
    if proc.returncode != 0:
        raise RuntimeError(
            "failed to sample inference processes: "
            f"rc={proc.returncode} output={(proc.stdout or '').strip()}")
    executable_names = {
        "llama-cli",
        "ffn_worker",
        "memory-elastic-op-bench",
        "granularity-pipeline-bench",
    }
    matches: list[str] = []
    for line in (proc.stdout or "").splitlines():
        fields = line.split(None, 3)
        if len(fields) < 4 or not fields[1].isdigit():
            continue
        # Inspect the process executable rather than searching its complete
        # argv.  The adb shell wrapper contains the literal "./llama-cli" in
        # its `sh -c` script, but it is not a second inference process.
        executable = fields[3].split(None, 1)[0].strip("[]")
        if Path(executable).name.lower() in executable_names:
            matches.append(line.strip())
    ffn_pids: set[int] = set()
    for line in matches:
        if "ffn_worker" not in line:
            continue
        fields = line.split(None, 3)
        if len(fields) >= 2 and fields[1].isdigit():
            ffn_pids.add(int(fields[1]))
    active_ffn = active_ffn_worker_pids(
        serial, ffn_pids, timeout, quiet=quiet)
    filtered: list[str] = []
    for line in matches:
        if "ffn_worker" not in line:
            filtered.append(line)
            continue
        fields = line.split(None, 3)
        pid = int(fields[1]) if len(fields) >= 2 and fields[1].isdigit() else -1
        if pid in active_ffn:
            filtered.append(line)
    return filtered


def wait_for_clean_start(
    args: argparse.Namespace,
    trace_name: str,
    method: str,
) -> dict[str, Any]:
    """Require both an idle device and a cold device at the same instant."""
    while True:
        wait_for_device_idle(args, trace_name, method)
        snapshot = wait_for_cooldown(args, trace_name, method)
        if not args.require_device_idle:
            return snapshot
        processes = external_inference_processes(
            args.adb_serial, args.adb_timeout_s)
        if not processes:
            return snapshot
        print(
            "=== device became busy during cooldown; restarting clean-start "
            f"gate trace={trace_name} method={method}: "
            + " | ".join(processes),
            flush=True)


def monitor_measurement_overlap(
    args: argparse.Namespace,
    stop: threading.Event,
    result: dict[str, Any],
) -> None:
    """Detect a competing benchmark that starts after the clean-start gate."""
    seen: set[str] = set()
    samples = 0
    failures = 0
    frequency_samples: list[list[tuple[int, int, int]]] = []
    measured_llama_pid: int | None = None
    monitored_cpus: list[int] = []
    if args.execution_backend == "cpu" and args.cpu_mask:
        try:
            mask = int(args.cpu_mask, 16)
            monitored_cpus = [
                cpu for cpu in range(mask.bit_length())
                if mask & (1 << cpu)]
        except ValueError:
            monitored_cpus = []
    poll_s = max(1.0, min(10.0, args.device_idle_poll_s))
    monitor_adb_timeout_s = max(
        1.0, min(5.0, getattr(args, "adb_timeout_s", 5.0)))
    while not stop.wait(poll_s):
        try:
            processes = external_inference_processes(
                args.adb_serial, monitor_adb_timeout_s, quiet=True)
            if monitored_cpus:
                cpu_words = " ".join(str(cpu) for cpu in monitored_cpus)
                freq_proc = adb_shell(
                    args.adb_serial,
                    f"for c in {cpu_words}; do "
                    "d=/sys/devices/system/cpu/cpu$c/cpufreq; "
                    "printf '%s ' $c; "
                    "cat $d/scaling_cur_freq $d/scaling_max_freq "
                    "2>/dev/null | tr '\\n' ' '; echo; done",
                    check=False,
                    timeout=monitor_adb_timeout_s,
                    announce=False,
                )
                snapshot: list[tuple[int, int, int]] = []
                for line in (freq_proc.stdout or "").splitlines():
                    fields = line.split()
                    if (len(fields) >= 3 and
                            all(field.isdigit() for field in fields[:3])):
                        snapshot.append(
                            (int(fields[0]), int(fields[1]), int(fields[2])))
                if snapshot:
                    frequency_samples.append(snapshot)
            samples += 1
        except (OSError, RuntimeError, subprocess.SubprocessError):
            failures += 1
            continue
        non_llama = [
            line for line in processes
            if Path(line.split(None, 3)[3].split(None, 1)[0]
                    .strip("[]")).name.lower() != "llama-cli"
        ]
        llama = [
            line for line in processes
            if Path(line.split(None, 3)[3].split(None, 1)[0]
                    .strip("[]")).name.lower() == "llama-cli"
        ]
        seen.update(non_llama)
        llama_by_pid = {
            int(line.split(None, 3)[1]): line
            for line in llama
        }
        # The clean-start gate guarantees that no llama-cli predates this
        # invocation.  The first sole executable process is therefore the
        # measured child.  A second executable process is real overlap even
        # if it uses the same binary/model paths.
        if measured_llama_pid is None and len(llama_by_pid) == 1:
            measured_llama_pid = next(iter(llama_by_pid))
        if len(llama_by_pid) > 1:
            seen.update(llama)
        elif (
            measured_llama_pid is not None
            and llama_by_pid
            and measured_llama_pid not in llama_by_pid
        ):
            seen.update(llama)
    result.update({
        "measurement_monitor_samples": samples,
        "measurement_monitor_failures": failures,
        "measurement_overlap": " | ".join(sorted(seen)),
        "valid_measurement_isolation": (
            not seen
            and failures == 0
            and (
                not args.require_device_idle
                or (
                    samples > 0
                    and measured_llama_pid is not None
                )
            )
        ),
        "measurement_llama_pid": (
            measured_llama_pid if measured_llama_pid is not None else ""),
        "cpu_freq_sample_count": len(frequency_samples),
    })
    if frequency_samples:
        active = [
            current for sample in frequency_samples
            for _, current, _ in sample]
        limits = [
            limit for sample in frequency_samples
            for _, _, limit in sample]
        per_sample_min = [
            min(current for _, current, _ in sample)
            for sample in frequency_samples]
        result.update({
            "cpu_freq_min_khz": min(active),
            "cpu_freq_mean_khz": statistics.mean(active),
            "cpu_freq_median_sample_min_khz": statistics.median(
                per_sample_min),
            "cpu_freq_limit_min_khz": min(limits),
        })


def wait_for_device_idle(
    args: argparse.Namespace,
    trace_name: str,
    method: str,
) -> None:
    if not args.require_device_idle:
        return
    deadline = time.monotonic() + args.device_idle_timeout_s
    while True:
        processes = external_inference_processes(
            args.adb_serial, args.adb_timeout_s)
        if not processes:
            return
        print(
            "=== device busy "
            f"trace={trace_name} method={method}: "
            + " | ".join(processes),
            flush=True)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "device did not become idle before timeout: "
                + " | ".join(processes))
        time.sleep(args.device_idle_poll_s)


def set_fixed_performance_mode(
    args: argparse.Namespace,
    enabled: bool,
) -> None:
    adb_shell_retry(
        args.adb_serial,
        "cmd power set-fixed-performance-mode-enabled "
        + ("true" if enabled else "false"),
        timeout=args.adb_timeout_s,
        retries=args.adb_retries,
    )


@contextmanager
def fixed_performance_measurement(args: argparse.Namespace):
    """Bound fixed-performance mode to exactly one measured invocation.

    The audit dictionary is deliberately mutable so the caller can persist the
    cleanup result after the context exits.  A cleanup failure never masks an
    exception from the measured process, but it invalidates an otherwise
    successful row.
    """
    configured = bool(args.fixed_performance_mode)
    audit: dict[str, Any] = {
        "fixed_performance_mode_configured": configured,
        "fixed_performance_mode_enabled_for_measurement": False,
        "fixed_performance_restore_attempted": False,
        "fixed_performance_restore_valid": not configured,
        "fixed_performance_restore_error": "",
    }
    if not configured:
        yield audit
        return

    try:
        # Start every measured method from the same explicit transition rather
        # than inheriting an Android power-mode state from an earlier process.
        set_fixed_performance_mode(args, False)
        set_fixed_performance_mode(args, True)
        audit["fixed_performance_mode_enabled_for_measurement"] = True
        yield audit
    finally:
        audit["fixed_performance_restore_attempted"] = True
        try:
            set_fixed_performance_mode(args, False)
            audit["fixed_performance_restore_valid"] = True
        except Exception as exc:  # preserve the measured-process exception
            audit["fixed_performance_restore_valid"] = False
            audit["fixed_performance_restore_error"] = (
                f"{type(exc).__name__}: {exc}")
            print(
                "=== fixed-performance restore failed: "
                f"{audit['fixed_performance_restore_error']} ===",
                flush=True,
            )


def wait_for_cooldown(args: argparse.Namespace, trace_name: str, method: str) -> dict[str, Any]:
    if args.fixed_performance_mode:
        # This is a defensive reset for interrupted/legacy invocations.  The
        # measured-run context also restores the state in its finally block.
        set_fixed_performance_mode(args, False)
    snapshot = read_thermal_snapshot(args.adb_serial, args.adb_timeout_s)
    if (args.cooldown_thermal_max_c <= 0 and
            args.min_battery_level_pct <= 0):
        return snapshot

    deadline = time.monotonic() + args.cooldown_timeout_s
    while (
        not cooldown_snapshot_ready(args, snapshot)
        and time.monotonic() < deadline
    ):
        print(
            "=== clean-start trace=%s method=%s thermal_max=%.1fC "
            "target=%.1fC status=%s target_status<=%d battery=%s%% "
            "target_battery>=%d%% ==="
            % (
                trace_name,
                method,
                thermal_max(snapshot),
                args.cooldown_thermal_max_c,
                snapshot.get("thermal_status", ""),
                args.cooldown_thermal_status_max,
                snapshot.get("battery_level_pct", ""),
                args.min_battery_level_pct,
            ),
            flush=True,
        )
        time.sleep(args.cooldown_poll_s)
        snapshot = read_thermal_snapshot(args.adb_serial, args.adb_timeout_s)
    if not cooldown_snapshot_ready(args, snapshot):
        raise TimeoutError(
            "clean-start gate timed out "
            f"trace={trace_name} method={method}: "
            f"thermal_max={thermal_max(snapshot):.1f}C "
            f"thermal_status={snapshot.get('thermal_status', '')} "
            f"battery={snapshot.get('battery_level_pct', '')}%")
    return snapshot


def cooldown_snapshot_ready(
    args: argparse.Namespace,
    snapshot: dict[str, Any],
) -> bool:
    """Evaluate a clean-start snapshot without treating missing data as cold."""
    thermal_ready = True
    if args.cooldown_thermal_max_c > 0:
        temperature_available = any(
            snapshot.get(key, "") not in ("", None)
            for key in (
                "thermal_cpu_max_c",
                "thermal_gpu_max_c",
                "thermal_skin_max_c",
            )
        )
        status = thermal_status(snapshot)
        thermal_ready = (
            temperature_available
            and status is not None
            and status <= args.cooldown_thermal_status_max
            and thermal_max(snapshot) <= args.cooldown_thermal_max_c
        )
    battery = snapshot.get("battery_level_pct", "")
    battery_ready = (
        args.min_battery_level_pct <= 0
        or (
            battery not in ("", None)
            and int(battery) >= args.min_battery_level_pct
        )
    )
    return thermal_ready and battery_ready


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
            # ``budget_mb`` is the native model-matched trace schema emitted
            # by run_granularity_model_sweep.py. Preserve its absolute values;
            # this matrix only windows/rebases time and must not remap budget.
            m = float(row.get(
                "mem_available_mb",
                row.get("budget_mib", row.get("budget_mb", "0"))) or 0)
            rows.append((t, m))
    rows.sort(key=lambda x: x[0])
    return rows


def select_low_window(
    rows: list[tuple[float, float]],
    window_sec: float,
    stride_sec: float,
    start_sec: float | None = None,
) -> tuple[float, float]:
    if not rows:
        raise ValueError("empty trace")
    first = rows[0][0]
    last = rows[-1][0]
    if start_sec is not None:
        start = max(first, float(start_sec))
        end = min(start + window_sec, last)
        if end <= start:
            raise ValueError(f"requested window start {start_sec} is outside trace range {first}..{last}")
        return start, end
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
    window_start_sec: float | None = None,
) -> TraceWindow:
    rows = read_trace(source)
    start, end = select_low_window(rows, window_sec, stride_sec, window_start_sec)
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


def write_static_trace(trace: TraceWindow, out_dir: Path, kind: str) -> StaticTrace:
    if kind not in {"min", "max"}:
        raise ValueError(f"unknown static trace kind: {kind}")
    budget = trace.min_bucket_mib if kind == "min" else trace.max_bucket_mib
    out = out_dir / f"{trace.local.stem}_static_{kind}_{budget}MiB.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec", "mem_available_mb"])
        writer.writerow(["0", str(budget)])
        writer.writerow([f"{max(trace.replay_span, 1.0):.3f}".rstrip("0").rstrip("."), str(budget)])
    return StaticTrace(local=out, remote_name=out.name, budget_mib=budget)


def build_budget_list(traces: list[TraceWindow], bucket_mib: int, extra_max_mib: int) -> list[int]:
    lo = min(t.min_bucket_mib for t in traces)
    hi = max(max(t.max_bucket_mib for t in traces), extra_max_mib)
    if lo <= 0:
        lo = bucket_mib
    return list(range(lo, hi + 1, bucket_mib))


def count_q4_moe_expert_tensors(model_meta: Path) -> int:
    try:
        data = json.loads(model_meta.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    pattern = re.compile(r"ffn_(?:gate|up|down)_exps\.weight$")
    return sum(
        1
        for weight in data.get("weights", [])
        if pattern.search(str(weight.get("name", "")))
        and str(weight.get("quant_name", "")).upper() == "Q4_0"
    )


def derive_q4_moe_cache_base_slots(
    model_meta: Path,
    floor_plan: Path,
    *,
    floor_budget_mib: int,
    reserved_mib: int,
    total_experts: float,
) -> tuple[int, dict[str, float]]:
    """Size the retained expert cache from the floor budget's residual bytes."""
    if total_experts <= 0:
        return 0, {}
    try:
        meta = json.loads(model_meta.read_text())
        plan = json.loads(floor_plan.read_text())
    except (OSError, json.JSONDecodeError):
        return 0, {}

    expert_pattern = re.compile(r"ffn_(?:gate|up|down)_exps\.weight$")
    expert_weights = [
        weight
        for weight in meta.get("weights", [])
        if expert_pattern.search(str(weight.get("name", "")))
        and str(weight.get("quant_name", "")).upper() == "Q4_0"
    ]
    if not expert_weights:
        return 0, {}

    expert_pool_bytes = sum(int(weight.get("byte_size", 0)) for weight in expert_weights)
    bytes_per_slot = expert_pool_bytes / total_experts
    if bytes_per_slot <= 0:
        return 0, {}

    expert_names = {str(weight.get("name", "")) for weight in expert_weights}
    fixed_resident_bytes = sum(
        int(weight.get("byte_size", 0))
        for weight in plan.get("weights", [])
        if str(weight.get("name", "")) not in expert_names
        and str(weight.get("location", "disk")) != "disk"
    )
    floor_bytes = floor_budget_mib * 1024 * 1024
    reserved_bytes = reserved_mib * 1024 * 1024
    residual_bytes = max(0, floor_bytes - reserved_bytes - fixed_resident_bytes)
    slots = max(1, min(int(total_experts), int(residual_bytes // bytes_per_slot)))
    details = {
        "floor_budget_mib": float(floor_budget_mib),
        "reserved_mib": float(reserved_mib),
        "fixed_resident_mib": fixed_resident_bytes / (1024 * 1024),
        "expert_slot_mib": bytes_per_slot / (1024 * 1024),
        "residual_mib": residual_bytes / (1024 * 1024),
    }
    return slots, details


def pin_policy_requests(policy: str, tensor_suffix: str) -> bool:
    tokens = {token.strip() for token in policy.split(",") if token.strip()}
    return "all" in tokens or tensor_suffix in tokens


def effective_granularity_policy(
    method: str,
    configured_policy: str,
) -> str:
    if method == "offline-mixed":
        return "offline"
    if method == "online" and configured_policy != "none":
        return "online"
    if method == "diff-tree-mixed":
        return "diff-tree"
    # Static placement and MRU retain one complete tensor as their logical
    # working unit even though the invocation also builds mixed tables for
    # the proposed methods.
    return "fixed-tensor"


def effective_granularity_placement_source(
    method: str,
    configured_source: str,
) -> str:
    """Report the placement algorithm that actually produced each plan.

    Offline-Mixed always replays its precomputed budget table. Online and
    Diff-tree use the configured online placement source. Static and MRU do
    not participate in mixed-granularity placement planning.
    """
    if method == "offline-mixed":
        return "offline-table"
    if method in {"online", "diff-tree-mixed"}:
        return configured_source
    return "fixed-tensor"


def remote_server_granularity_policy(
    method: str,
    configured_policy: str,
) -> str:
    """Select the solver policy for the one method this server will serve.

    A matrix invocation can contain both Online and Diff-tree, but each
    measured method receives a fresh server process.  Deriving this value from
    the full matrix method set would silently configure Online as Diff-tree
    whenever Diff-tree is also present.
    """
    if method == "online":
        if configured_policy == "none":
            return "none"
        return "online"
    if method == "diff-tree-mixed":
        return "diff-tree"
    raise ValueError(
        f"remote server is not used by method {method!r}")


def granularity_profile_kind(
    path: Path | None,
    backend: str,
) -> str:
    if path is None or not path.is_file():
        return "none"
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", data)
    selected = (
        profiles.get(backend, {})
        if isinstance(profiles, dict) else {})
    curves = (
        selected.get("mode_pipeline_efficiency_curve", {})
        if isinstance(selected, dict) else {})
    has_residual = (
        isinstance(curves, dict)
        and any(
            isinstance(points, list) and len(points) > 0
            for points in curves.values()
        )
    )
    return "pipeline-residual" if has_residual else "phase-only"


def granularity_profile_multi_implementation(
    path: Path | None,
    backend: str,
) -> str:
    """Return the coarse-unit implementation promised by a profile."""
    if path is None or not path.is_file():
        return "multi"
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", data)
    selected = (
        profiles.get(backend, {})
        if isinstance(profiles, dict) else {})
    if (
        not isinstance(selected, dict)
        or "multi_implementation" not in selected
    ):
        raise ValueError(
            f"{backend} cost profile {path} does not explicitly encode "
            "multi_implementation")
    implementation = str(selected["multi_implementation"])
    if implementation not in {"multi", "multi_fused"}:
        raise ValueError(
            f"invalid {backend} multi_implementation in {path}: "
            f"{implementation!r}")
    if backend != "cpu" and implementation == "multi_fused":
        raise ValueError(
            f"{backend} profile advertises unsupported multi_fused")
    return implementation


def granularity_profile_cut_compute_grouping(
    path: Path | None,
    backend: str,
) -> str:
    """Return the Cut compute dependency encoded by the cost profile."""
    expected = (
        "fused_pair" if backend == "gpu" else "independent_halves")
    if path is None or not path.is_file():
        return expected
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", data)
    selected = (
        profiles.get(backend, {})
        if isinstance(profiles, dict) else {})
    if (
        not isinstance(selected, dict)
        or "cut_compute_grouping" not in selected
    ):
        raise ValueError(
            f"{backend} cost profile {path} does not explicitly encode "
            "cut_compute_grouping")
    grouping = str(selected["cut_compute_grouping"])
    if grouping not in {"independent_halves", "fused_pair"}:
        raise ValueError(
            f"invalid {backend} cut_compute_grouping in {path}: "
            f"{grouping!r}")
    if grouping != expected:
        raise ValueError(
            f"{backend} cost profile uses {grouping!r} Cut compute, "
            f"but the formal runtime requires {expected!r}")
    return grouping


def granularity_profile_transition_contract(
    path: Path | None,
    backend: str,
) -> tuple[float, str]:
    """Return the profile's measured working-unit publication cost."""
    if path is None or not path.is_file():
        return 0.0, ""
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", data)
    selected = (
        profiles.get(backend, {})
        if isinstance(profiles, dict) else {})
    if not isinstance(selected, dict):
        return 0.0, ""
    return (
        float(selected.get("transition_fixed_ms", 0.0) or 0.0),
        str(selected.get("transition_cost_source", "")),
    )


def directory_sha256(path: Path) -> str:
    """Content-address one generated plan table, independent of mtimes."""
    if not path.is_dir():
        return ""
    digest = hashlib.sha256()
    files = sorted(
        child for child in path.rglob("*") if child.is_file())
    for child in files:
        relative = child.relative_to(path).as_posix().encode("utf-8")
        payload = child.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest() if files else ""


def directory_file_manifest(
    path: Path,
) -> dict[str, tuple[int, str]]:
    """Return exact per-file size/hash entries for remote byte validation."""
    if not path.is_dir():
        return {}
    return {
        child.relative_to(path).as_posix(): (
            child.stat().st_size,
            hashlib.sha256(child.read_bytes()).hexdigest(),
        )
        for child in sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file()
        )
    }


def validate_remote_directory_contents(
    serial: str,
    local_dir: Path,
    remote_dir: str,
    *,
    label: str,
    timeout_s: float,
    retries: int,
) -> str:
    """Fail closed unless a remote directory is byte-identical to local."""
    expected = directory_file_manifest(local_dir)
    if not expected:
        raise RuntimeError(
            f"local {label} directory is missing or empty: {local_dir}")
    proc = adb_shell_retry(
        serial,
        f"cd {shell_quote(remote_dir)} && "
        "find . -type f | sort | while IFS= read -r p; do "
        "sha=$(sha256sum \"$p\"); sha=${sha%% *}; "
        "bytes=$(wc -c < \"$p\" | tr -d ' '); "
        "printf '%s\\t%s\\t%s\\n' \"$sha\" \"$bytes\" \"${p#./}\"; "
        "done",
        timeout=timeout_s,
        retries=retries,
    )
    actual: dict[str, tuple[int, str]] = {}
    for raw in (proc.stdout or "").splitlines():
        fields = raw.rstrip("\r").split("\t", 2)
        if (
            len(fields) != 3
            or re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]) is None
            or not fields[1].isdigit()
            or not fields[2]
        ):
            raise RuntimeError(
                f"invalid remote {label} manifest row: {raw!r}")
        actual[fields[2]] = (int(fields[1]), fields[0].lower())
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            path for path in set(expected) & set(actual)
            if expected[path] != actual[path]
        )
        raise RuntimeError(
            f"remote {label} content mismatch: "
            f"missing={missing[:8]} extra={extra[:8]} "
            f"changed={changed[:8]}")
    content_sha256 = directory_sha256(local_dir)
    print(
        f"validated remote {label} sha256={content_sha256} "
        f"files={len(expected)} path={remote_dir}",
        flush=True,
    )
    return content_sha256


def parse_log(path: Path, method: str) -> dict[str, Any]:
    text = path.read_text(errors="replace") if path.exists() else ""
    result: dict[str, Any] = {
        "status": "ok",
        "eval_ms_total": "",
        "eval_runs": "",
        "prompt_runs": "",
        "forward_runs": "",
        "generated_tokens": "",
        "bench_exit_reason": "",
        "bench_time_done": "",
        "budget_decode_reset_count": 0,
        "budget_decode_reset_generated": "",
        "budget_backend_reset_count": 0,
        "raw_ms_per_token": "",
        "decode_ms_per_token": "",
        "exec_ms_per_token": "",
        "exec_timing_source": "",
        "exec_component_sum_ms_per_token": "",
        "decode_wall_ms_total": "",
        "decode_wall_runs": "",
        "decode_wall_ms_per_token": "",
        "decode_wall_unattributed_ms_total": "",
        "decode_phase_origin_count": 0,
        "decode_phase_quiesce_count": 0,
        "decode_phase_origin_generated": "",
        "decode_phase_origin_backend": "",
        "decode_phase_summary_count": 0,
        "decode_phase_backend": "",
        "decode_phase_runs": "",
        "decode_phase_units": "",
        "decode_phase_nonresident_units": "",
        "decode_phase_reloads": "",
        "decode_phase_reload_bytes": "",
        "decode_phase_direct_read_calls": "",
        "decode_phase_direct_read_ms": "",
        "decode_phase_direct_read_mib": "",
        "decode_phase_load_calls": "",
        "decode_phase_load_ms": "",
        "decode_phase_load_mib": "",
        "decode_phase_prepare_calls": "",
        "decode_phase_prepare_ms": "",
        "decode_phase_prepare_mib": "",
        "decode_phase_compute_available": "",
        "decode_phase_compute_calls": "",
        "decode_phase_compute_ms": "",
        "decode_phase_pipeline_waits": "",
        "decode_phase_pipeline_wait_ms": "",
        "decode_phase_pipeline_residency_ms": "",
        "decode_phase_pipeline_unissued_ms": "",
        "decode_phase_pipeline_stage_ms": "",
        "decode_phase_pipeline_retire_ms": "",
        "decode_phase_evict_ms": "",
        "decode_phase_resident_mib": "",
        "decode_phase_counter_scope": "",
        "remote_wall_ms": 0.0,
        "remote_wall_ms_max": 0.0,
        "remote_server_ms": 0.0,
        "remote_server_ms_max": 0.0,
        "provider_get_ms_total": 0.0,
        "provider_get_ms_max": 0.0,
        "provider_get_calls": 0,
        "apply_ms_total": 0.0,
        "apply_ms_max": 0.0,
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
        "reload_bytes": "",
        "evict_bytes": "",
        "pipeline_wait_ms": "",
        "pipeline_waits": "",
        "pipeline_stage_ms": "",
        "pipeline_retire_ms": "",
        "pipeline_evict_ms": "",
        "pipeline_residency_ms": "",
        "pipeline_unissued_ms": "",
        "pipeline_missing_mib": "",
        "pipeline_unissued_mib": "",
        "pipeline_budget_samples": "",
        "pipeline_budget_violations": "",
        "pipeline_plan_protection_relaxations": "",
        "pipeline_plan_protection_relaxed_mib": "",
        "pipeline_resident_peak_mib": "",
        "pipeline_pinned_peak_mib": "",
        "pipeline_over_budget_peak_mib": "",
        "pipeline_load_affinity_cpu": "",
        "pipeline_load_affinity_rc": "",
        "pipeline_prepare_affinity_cpu": "",
        "pipeline_prepare_affinity_rc": "",
        "pipeline_copy_affinity_cpu": "",
        "pipeline_copy_affinity_rc": "",
        "granularity_units": "",
        "granularity_cut_ops": "",
        "physical_cut_tensors": "",
        "physical_cut_parts": "",
        "wbm_total_mib": "",
        "cpu_delegate_compute_calls": "",
        "cpu_delegate_compute_ms": "",
        "cpu_delegate_compute_avg_ms": "",
        "stage_xform_calls": "",
        "stage_xform_ms": "",
        "stage_xform_mib": "",
        "async_prepare_waits": "",
        "async_prepare_wait_ms": "",
        "async_xform_waits": "",
        "async_xform_wait_ms": "",
        "cpu_layout_initial_repack_calls": "",
        "cpu_layout_initial_repack_ms": "",
        "cpu_layout_initial_repack_mib": "",
        "cpu_layout_repack_calls": "",
        "cpu_layout_repack_ms": "",
        "cpu_layout_repack_mib": "",
        "cpu_layout_materialize_calls": "",
        "cpu_layout_materialize_ms": "",
        "cpu_layout_materialize_mib": "",
        "fused_layout_pair_calls": "",
        "fused_layout_parallel_calls": "",
        "fused_layout_pair_ms": "",
        "fused_layout_pair_mib": "",
        "fused_kernel_pair_calls": "",
        "fused_kernel_pair_candidates": "",
        "fused_kernel_pair_fallbacks": "",
        "fused_kernel_pair_errors": "",
        "fused_kernel_pair_ms": "",
        "cut_dual_candidates": "",
        "cut_dual_calls": "",
        "cut_dual_budget_fallbacks": "",
        "cut_dual_shape_fallbacks": "",
        "cut_dual_queue_errors": "",
        "cut_dual_image_creates": "",
        "cut_dual_image_cache_hits": "",
        "cut_dual_image_releases": "",
        "cut_dual_image_errors": "",
        "frontier_predicted_ms_last": "",
        "frontier_switch_ms_total": 0.0,
        "frontier_switch_ms_max": 0.0,
        "frontier_publish_ms_total": 0.0,
        "frontier_publish_ms_max": 0.0,
        "frontier_transition_publish_count": 0,
        "frontier_transition_publish_ms_total": 0.0,
        "frontier_transition_publish_ms_max": 0.0,
        "frontier_delta_weights_total": 0,
        "frontier_delta_weights_max": 0,
        "frontier_transition_delta_weights_total": 0,
        "frontier_transition_delta_weights_max": 0,
        "frontier_only_apply_count": 0,
        "frontier_apply_count": 0,
        "frontier_mixed_mode_generations": 0,
        "frontier_trace_sha256": "",
        "frontier_transition_trace_sha256": "",
        "frontier_shape_trace_sha256": "",
        "frontier_unique_states": 0,
        "frontier_unique_shape_states": 0,
        "frontier_state_changes": 0,
        "frontier_shape_state_changes": 0,
        "frontier_multi_units_last": "",
        "frontier_tensor_units_last": "",
        "frontier_cut_units_last": "",
        "frontier_multi_units_max": "",
        "frontier_tensor_units_max": "",
        "frontier_cut_units_max": "",
        "plan_stage_defer_apply_count": 0,
        "plan_stage_defer_min": "",
        "plan_stage_defer_max": "",
        "plan_anchor_fired": "",
        "plan_anchor_load_events": "",
        "plan_anchor_transfer_events": "",
        "plan_anchor_xform_events": "",
        "moe_cache_mib": "",
        "moe_cache_hits": "",
        "moe_cache_misses": "",
        "moe_cache_hit_rate": "",
        "moe_cache_disk_mb": "",
        "moe_cache_disk_ms": "",
        "moe_cache_resize_syncs": "",
        "online_calls": "",
        "online_failures": "",
        "pin_token_embd_inside_budget_count": 0,
        "pin_token_embd_inside_budget_mib": 0.0,
        "pin_output_inside_budget_count": 0,
        "pin_output_inside_budget_mib": 0.0,
        "pin_outside_budget_count": 0,
    }
    if "TIMEOUT" in text:
        result["status"] = "timeout"
    inner_rc = re.search(r"__LLAMA_INNER_RC__=(\d+)", text)
    fatal_markers = (
        "CANNOT LINK EXECUTABLE",
        "No such file or directory",
        "error loading model",
        "failed to load model",
        "failed to create context",
        "load failed for",
        "[elastic-fw] failed to load plan",
        "apply_exec_plan failed",
        "moe cache: upload failed",
        "moe cache: read active mask failed",
        "moe cache: allocation failed",
    )
    if any(marker.lower() in text.lower() for marker in fatal_markers) or (inner_rc and int(inner_rc.group(1)) != 0):
        result["status"] = "check_log"
    if text and "__LLAMA_INNER_RC__" not in text and result["status"] == "ok":
        result["status"] = "remote_incomplete"

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

    prompt = re.search(r"prompt eval time =\s*[0-9.]+ ms /\s*(\d+) tokens", text)
    prompt_runs = 1 if prompt and int(prompt.group(1)) > 0 else 0
    result["prompt_runs"] = prompt_runs
    if result["eval_runs"] != "":
        result["forward_runs"] = int(result["eval_runs"]) + prompt_runs
    generated = re.search(
        r"\[elastic-bench\] final exit reason=(\S+) generated=(\d+)"
        r".*?time_done=(\d+)",
        text,
    )
    if generated:
        result["bench_exit_reason"] = generated.group(1)
        result["generated_tokens"] = int(generated.group(2))
        result["bench_time_done"] = int(generated.group(3))
    budget_decode_resets = re.findall(
        r"\[elastic-bench\] budget replay reset "
        r"boundary=first_decode generated=(\d+)",
        text,
    )
    result["budget_decode_reset_count"] = len(budget_decode_resets)
    if budget_decode_resets:
        result["budget_decode_reset_generated"] = int(
            budget_decode_resets[-1])
    result["budget_backend_reset_count"] = len(re.findall(
        r"BudgetWatcher replay clock reset", text))
    decode_phase_origins = re.findall(
        r"\[elastic-bench\] decode phase origin generated=(\d+)"
        r"\s+backend=(cpu|gpu)",
        text,
    )
    result["decode_phase_origin_count"] = len(decode_phase_origins)
    if decode_phase_origins:
        result["decode_phase_origin_generated"] = int(
            decode_phase_origins[-1][0])
        result["decode_phase_origin_backend"] = (
            decode_phase_origins[-1][1])
    decode_phase_quiesce = re.findall(
        r"\[elastic-bench\] decode phase quiesce "
        r"boundary=(begin|end) rc=(-?\d+)",
        text,
    )
    result["decode_phase_quiesce_count"] = len(decode_phase_quiesce)
    result["decode_phase_quiesce_sequence"] = ",".join(
        boundary for boundary, _ in decode_phase_quiesce)
    for boundary, rc in decode_phase_quiesce:
        result[f"decode_phase_quiesce_{boundary}_rc"] = int(rc)
    decode_phase_payloads = [
        payload
        for payload in re.findall(
            r"\[elastic decode phase summary\]\s+([^\n]+)", text)
        if not payload.startswith("unavailable")
    ]
    result["decode_phase_summary_count"] = len(decode_phase_payloads)
    if decode_phase_payloads:
        values = dict(re.findall(
            r"([a-z_]+)=([A-Za-z0-9.]+)",
            decode_phase_payloads[-1],
        ))
        integer_fields = {
            "runs": "decode_phase_runs",
            "units": "decode_phase_units",
            "nonresident_units": "decode_phase_nonresident_units",
            "reloads": "decode_phase_reloads",
            "reload_bytes": "decode_phase_reload_bytes",
            "direct_read_calls": "decode_phase_direct_read_calls",
            "load_calls": "decode_phase_load_calls",
            "prepare_calls": "decode_phase_prepare_calls",
            "compute_available": "decode_phase_compute_available",
            "compute_calls": "decode_phase_compute_calls",
            "pipeline_waits": "decode_phase_pipeline_waits",
        }
        microsecond_fields = {
            "direct_read_us": "decode_phase_direct_read_ms",
            "load_us": "decode_phase_load_ms",
            "prepare_us": "decode_phase_prepare_ms",
            "compute_us": "decode_phase_compute_ms",
            "pipeline_wait_us": "decode_phase_pipeline_wait_ms",
            "pipeline_residency_us":
                "decode_phase_pipeline_residency_ms",
            "pipeline_unissued_us":
                "decode_phase_pipeline_unissued_ms",
            "pipeline_stage_us": "decode_phase_pipeline_stage_ms",
            "pipeline_retire_us": "decode_phase_pipeline_retire_ms",
            "evict_us": "decode_phase_evict_ms",
        }
        byte_fields = {
            "direct_read_bytes": "decode_phase_direct_read_mib",
            "load_bytes": "decode_phase_load_mib",
            "prepare_bytes": "decode_phase_prepare_mib",
            "resident_bytes": "decode_phase_resident_mib",
        }
        result["decode_phase_backend"] = values.get("backend", "")
        for source, target in integer_fields.items():
            if source in values:
                result[target] = int(values[source])
        for source, target in microsecond_fields.items():
            if source in values:
                result[target] = int(values[source]) / 1000.0
        for source, target in byte_fields.items():
            if source in values:
                result[target] = (
                    int(values[source]) / 1024.0 / 1024.0)
        result["decode_phase_counter_scope"] = (
            "decode-only-backend-counter-delta")

    remote_pairs = re.findall(r"remote_wall_ms=([0-9.]+).*?server_solve_ms=([0-9.]+)", text)
    remote_wall = sum(float(a) for a, _ in remote_pairs)
    remote_server = sum(float(b) for _, b in remote_pairs)
    result["remote_wall_ms_max"] = max(
        (float(a) for a, _ in remote_pairs), default=0.0)
    result["remote_server_ms_max"] = max(
        (float(b) for _, b in remote_pairs), default=0.0)
    summary = re.search(r"remote_wall_ms=([0-9.]+)\s+remote_server_ms=([0-9.]+)", text)
    if summary:
        remote_wall = max(remote_wall, float(summary.group(1)))
        remote_server = max(remote_server, float(summary.group(2)))
    result["remote_wall_ms"] = remote_wall
    result["remote_server_ms"] = remote_server
    apply_pairs = [
        (float(provider_ms), float(apply_ms))
        for provider_ms, apply_ms in re.findall(
            r"provider_get_ms=([0-9.]+)\s+apply_ms=([0-9.]+)",
            text,
        )
    ]
    for provider_ms, apply_ms in apply_pairs:
        result["provider_get_ms_total"] += provider_ms
        result["apply_ms_total"] += apply_ms
        result["apply_count"] += 1
    result["provider_get_ms_max"] = max(
        (provider_ms for provider_ms, _ in apply_pairs), default=0.0)
    result["apply_ms_max"] = max(
        (apply_ms for _, apply_ms in apply_pairs), default=0.0)
    timing_summary = re.search(
        r"elastic plan timing summary:\s+provider_calls=(\d+)"
        r"\s+provider_get_ms=([0-9.]+)\s+apply_calls=(\d+)"
        r"\s+apply_ms=([0-9.]+)",
        text,
    )
    if timing_summary:
        result["provider_get_calls"] = int(timing_summary.group(1))
        result["provider_get_ms_total"] = float(timing_summary.group(2))
        result["apply_count"] = int(timing_summary.group(3))
        result["apply_ms_total"] = float(timing_summary.group(4))
    decode_wall_summary = re.search(
        r"elastic decode wall summary:\s+runs=(\d+)"
        r"\s+wall_ms=([0-9.]+)",
        text,
    )
    if decode_wall_summary:
        result["decode_wall_runs"] = int(decode_wall_summary.group(1))
        result["decode_wall_ms_total"] = float(
            decode_wall_summary.group(2))
        if result["decode_wall_runs"] > 0:
            result["decode_wall_ms_per_token"] = (
                float(result["decode_wall_ms_total"])
                / int(result["decode_wall_runs"])
            )
    if result["eval_ms_total"] != "" and result["eval_runs"]:
        runs = int(result["eval_runs"])
        if runs > 0:
            # maybe_apply_plan() runs before llama_context starts its eval
            # timer. Both synchronous provider lookup/wait and residency
            # changes are therefore decode-critical and must be added back.
            # remote_wall_ms is not added separately because asynchronous
            # solving may overlap compute; its blocking portion is already
            # contained in provider_get_ms_total.
            exec_total = (
                float(result["eval_ms_total"])
                + float(result["provider_get_ms_total"])
                + float(result["apply_ms_total"])
            )
            result["exec_component_sum_ms_per_token"] = exec_total / runs
            if result["decode_wall_runs"] not in ("", 0):
                result["exec_ms_per_token"] = (
                    float(result["decode_wall_ms_total"])
                    / int(result["decode_wall_runs"])
                )
                result["exec_timing_source"] = "decode-wall"
                result["decode_wall_unattributed_ms_total"] = (
                    float(result["decode_wall_ms_total"]) - exec_total
                )
            else:
                # Compatibility fallback for logs produced by binaries before
                # the independent decode-wall audit was added. Formal rows
                # reject this fallback via decode_wall_contract_failures().
                result["exec_ms_per_token"] = exec_total / runs
                result["exec_timing_source"] = "component-sum-fallback"

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
    m = re.search(
        r"(?:direct O_DIRECT read calls=|direct_read\s*:\s*calls=)(\d+)"
        r".*?total=([0-9.]+) ms.*?MB=([0-9.]+)",
        text,
    )
    if m:
        result["direct_read_calls"] = int(m.group(1))
        result["direct_read_ms"] = float(m.group(2))
        result["direct_read_mb"] = float(m.group(3))
    m = re.search(r"reload_count\s*:\s*\d+\s*\((\d+) bytes total\)", text)
    if m:
        result["reload_bytes"] = int(m.group(1))
    m = re.search(r"evict_count\s*:\s*\d+\s*\((\d+) bytes total\)", text)
    if m:
        result["evict_bytes"] = int(m.group(1))
    m = re.search(r"\[elastic unit pipeline\].*?waits=(\d+)\s+wait_ms=([0-9.]+)", text)
    if m:
        result["pipeline_waits"] = int(m.group(1))
        result["pipeline_wait_ms"] = float(m.group(2))
    m = re.search(
        r"\[elastic unit pipeline timing\].*?stage_ms=([0-9.]+)"
        r".*?retire_ms=([0-9.]+).*?evict_ms=([0-9.]+)",
        text,
    )
    if m:
        result["pipeline_stage_ms"] = float(m.group(1))
        result["pipeline_retire_ms"] = float(m.group(2))
        result["pipeline_evict_ms"] = float(m.group(3))
    m = re.search(
        r"\[elastic unit pipeline residency\].*?total_ms=([0-9.]+)"
        r".*?missing_mib=([0-9.]+).*?unissued_mib=([0-9.]+)"
        r".*?unissued_ms=([0-9.]+)",
        text,
    )
    if m:
        result["pipeline_residency_ms"] = float(m.group(1))
        result["pipeline_missing_mib"] = float(m.group(2))
        result["pipeline_unissued_mib"] = float(m.group(3))
        result["pipeline_unissued_ms"] = float(m.group(4))
    m = re.search(
        r"\[elastic unit pipeline budget\].*?samples=(\d+)"
        r".*?violations=(\d+).*?resident_peak_mib=([0-9.]+)"
        r".*?pinned_peak_mib=([0-9.]+).*?over_peak_mib=([0-9.]+)",
        text,
    )
    if m:
        result["pipeline_budget_samples"] = int(m.group(1))
        result["pipeline_budget_violations"] = int(m.group(2))
        result["pipeline_resident_peak_mib"] = float(m.group(3))
        result["pipeline_pinned_peak_mib"] = float(m.group(4))
        result["pipeline_over_budget_peak_mib"] = float(m.group(5))
    m = re.search(
        r"\[elastic unit pipeline budget\].*?"
        r"plan_protection_relaxations=(\d+)"
        r"\s+relaxed_mib=([0-9.]+)",
        text,
    )
    if m:
        result["pipeline_plan_protection_relaxations"] = int(m.group(1))
        result["pipeline_plan_protection_relaxed_mib"] = float(m.group(2))
    for worker in ("LOAD", "PREPARE", "COPY"):
        m = re.search(
            rf"elastic: {worker} worker affinity cpu=(\d+) rc=(-?\d+)",
            text,
        )
        if m:
            prefix = f"pipeline_{worker.lower()}_affinity"
            result[f"{prefix}_cpu"] = int(m.group(1))
            result[f"{prefix}_rc"] = int(m.group(2))
    m = re.search(
        r"granularity\s*:\s*mode=\S+\s+units=(\d+)\s+cut_ops=(\d+)",
        text,
    )
    if m:
        result["granularity_units"] = int(m.group(1))
        result["granularity_cut_ops"] = int(m.group(2))
    m = re.search(
        r"\[elastic granularity\].*?cut_tensors=(\d+)"
        r"\s+cut_parts=(\d+).*?wbm_total=([0-9.]+)\s+MiB",
        text,
    )
    if not m:
        # CPU_Elastic reports the same physical representation in its profile
        # dump as ``cut_tensors=N parts=M``; OpenCL uses
        # ``cut_tensors=N cut_parts=M``.
        m = re.search(
            r"granularity\s*:.*?cut_tensors=(\d+)"
            r"\s+(?:cut_parts|parts)=(\d+).*?"
            r"wbm_total=([0-9.]+)\s+MiB",
            text,
        )
    if m:
        result["physical_cut_tensors"] = int(m.group(1))
        result["physical_cut_parts"] = int(m.group(2))
        result["wbm_total_mib"] = float(m.group(3))
    m = re.search(
        r"compute_pt\s*:\s*[0-9.]+\s+ms/graph.*?"
        r"total=([0-9.]+)\s+ms\s+calls=(\d+)\s+avg=([0-9.]+)\s+ms",
        text,
    )
    if m:
        result["cpu_delegate_compute_ms"] = float(m.group(1))
        result["cpu_delegate_compute_calls"] = int(m.group(2))
        result["cpu_delegate_compute_avg_ms"] = float(m.group(3))

    # The CPU and OpenCL backends use slightly different labels for the same
    # preparation stage.  Select the non-zero active-backend record rather
    # than summing backend diagnostic dumps.
    stage_xform = [
        (int(calls), float(ms), float(mib))
        for calls, ms, mib in re.findall(
            r"stage[ _]xform\s*:?\s*calls=(\d+)\s+ok=\d+"
            r"\s+total=([0-9.]+)\s+ms.*?MB=([0-9.]+)",
            text,
        )
    ]
    if stage_xform:
        calls, ms, mib = max(stage_xform, key=lambda values: values[1])
        result["stage_xform_calls"] = calls
        result["stage_xform_ms"] = ms
        result["stage_xform_mib"] = mib
    m = re.search(
        r"async_prepare:.*?waits=(\d+)\s+wait_total=([0-9.]+)\s+ms",
        text,
    )
    if m:
        result["async_prepare_waits"] = int(m.group(1))
        result["async_prepare_wait_ms"] = float(m.group(2))
    m = re.search(
        r"async xform worker.*?waits=(\d+)\s+"
        r"wait_total=([0-9.]+)\s+ms",
        text,
    )
    if m:
        result["async_xform_waits"] = int(m.group(1))
        result["async_xform_wait_ms"] = float(m.group(2))
    m = re.search(
        r"\[elastic cpu layout\]\s+"
        r"initial_repack_calls=(\d+)\s+initial_repack_ms=([0-9.]+)"
        r"\s+initial_repack_mib=([0-9.]+)\s+"
        r"repack_calls=(\d+)\s+repack_ok=\d+\s+"
        r"repack_ms=([0-9.]+)\s+repack_mib=([0-9.]+)\s+"
        r"materialize_calls=(\d+)\s+materialize_ms=([0-9.]+)"
        r"\s+materialize_mib=([0-9.]+)",
        text,
    )
    if m:
        result["cpu_layout_initial_repack_calls"] = int(m.group(1))
        result["cpu_layout_initial_repack_ms"] = float(m.group(2))
        result["cpu_layout_initial_repack_mib"] = float(m.group(3))
        result["cpu_layout_repack_calls"] = int(m.group(4))
        result["cpu_layout_repack_ms"] = float(m.group(5))
        result["cpu_layout_repack_mib"] = float(m.group(6))
        result["cpu_layout_materialize_calls"] = int(m.group(7))
        result["cpu_layout_materialize_ms"] = float(m.group(8))
        result["cpu_layout_materialize_mib"] = float(m.group(9))
    m = re.search(
        r"\[elastic cpu multi_fused\]\s+layout_pairs=(\d+)"
        r"\s+ok=\d+\s+fallback=\d+(?:\s+parallel=(\d+))?"
        r"\s+ms=([0-9.]+)"
        r"\s+mib=([0-9.]+).*?kernel_candidates=(\d+)"
        r"\s+calls=(\d+)\s+fallback=(\d+)\s+errors=(\d+)"
        r"\s+ms=([0-9.]+)",
        text,
    )
    if m:
        result["fused_layout_pair_calls"] = int(m.group(1))
        result["fused_layout_parallel_calls"] = int(m.group(2) or 0)
        result["fused_layout_pair_ms"] = float(m.group(3))
        result["fused_layout_pair_mib"] = float(m.group(4))
        result["fused_kernel_pair_candidates"] = int(m.group(5))
        result["fused_kernel_pair_calls"] = int(m.group(6))
        result["fused_kernel_pair_fallbacks"] = int(m.group(7))
        result["fused_kernel_pair_errors"] = int(m.group(8))
        result["fused_kernel_pair_ms"] = float(m.group(9))
    m = re.search(
        r"\[elastic cut dual\].*?candidates=(\d+)"
        r"\s+calls=(\d+)\s+budget_fallbacks=(\d+)"
        r"\s+shape_fallbacks=(\d+)\s+queue_errors=(\d+)"
        r"(?:\s+image_creates=(\d+)\s+image_cache_hits=(\d+)"
        r"(?:\s+image_releases=(\d+))?"
        r"\s+image_errors=(\d+))?",
        text,
    )
    if m:
        result["cut_dual_candidates"] = int(m.group(1))
        result["cut_dual_calls"] = int(m.group(2))
        result["cut_dual_budget_fallbacks"] = int(m.group(3))
        result["cut_dual_shape_fallbacks"] = int(m.group(4))
        result["cut_dual_queue_errors"] = int(m.group(5))
        result["cut_dual_image_creates"] = int(m.group(6) or 0)
        result["cut_dual_image_cache_hits"] = int(m.group(7) or 0)
        result["cut_dual_image_releases"] = int(m.group(8) or 0)
        result["cut_dual_image_errors"] = int(m.group(9) or 0)

    frontier_costs = [
        (float(predicted), float(switch))
        for predicted, switch in re.findall(
            r"working_unit .*?predicted_ms=([0-9.]+)"
            r"\s+switch_ms=([0-9.]+)",
            text,
        )
    ]
    if frontier_costs:
        result["frontier_predicted_ms_last"] = frontier_costs[-1][0]
        result["frontier_switch_ms_total"] = sum(
            switch for _, switch in frontier_costs)
        result["frontier_switch_ms_max"] = max(
            switch for _, switch in frontier_costs)
    frontier_counts = [
        tuple(int(value) for value in match)
        for match in re.findall(
            r"mixed_units=\d+\s+multi_units=(\d+)\s+"
            r"tensor_units=(\d+)\s+cut_units=(\d+)",
            text,
        )
    ]
    if frontier_counts:
        result["frontier_apply_count"] = len(frontier_counts)
        result["frontier_mixed_mode_generations"] = sum(
            sum(value > 0 for value in counts) > 1
            for counts in frontier_counts)
        last = frontier_counts[-1]
        for index, mode in enumerate(("multi", "tensor", "cut")):
            result[f"frontier_{mode}_units_last"] = last[index]
            result[f"frontier_{mode}_units_max"] = max(
                counts[index] for counts in frontier_counts)
    # Hash both the human-readable shape sequence and the exact on-device
    # working-unit identity sequence. ``frontier_sig`` covers unit ids, fusion
    # flags, weight/tile ids, row ranges, byte ranges, and physical tiling,
    # while excluding predicted latency and switch cost.
    frontier_states = [
        tuple(int(value) for value in match)
        for match in re.findall(
            r"working_unit [^\n]*\bmixed_units=(\d+)\s+"
            r"multi_units=(\d+)\s+tensor_units=(\d+)\s+"
            r"cut_units=(\d+)\s+tiles=(\d+)",
            text,
        )
    ]
    if frontier_states:
        encoded_states = json.dumps(
            frontier_states, separators=(",", ":")).encode("utf-8")
        result["frontier_shape_trace_sha256"] = hashlib.sha256(
            encoded_states).hexdigest()
        result["frontier_unique_shape_states"] = len(set(frontier_states))
        result["frontier_shape_state_changes"] = sum(
            current != previous
            for previous, current in zip(
                frontier_states, frontier_states[1:])
        )
    frontier_identities = [
        value.lower()
        for value in re.findall(
            r"working_unit [^\n]*\bfrontier_sig=([0-9a-fA-F]{16})\b",
            text,
        )
    ]
    if (
        frontier_identities
        and len(frontier_identities) == len(frontier_states)
    ):
        encoded_identities = json.dumps(
            frontier_identities, separators=(",", ":")).encode("utf-8")
        result["frontier_trace_sha256"] = hashlib.sha256(
            encoded_identities).hexdigest()
        transition_identities = [frontier_identities[0]]
        transition_identities.extend(
            current
            for previous, current in zip(
                frontier_identities, frontier_identities[1:])
            if current != previous
        )
        encoded_transition_identities = json.dumps(
            transition_identities,
            separators=(",", ":"),
        ).encode("utf-8")
        result["frontier_transition_trace_sha256"] = hashlib.sha256(
            encoded_transition_identities).hexdigest()
        result["frontier_unique_states"] = len(
            set(frontier_identities))
        result["frontier_state_changes"] = sum(
            current != previous
            for previous, current in zip(
                frontier_identities, frontier_identities[1:])
        )
    frontier_delta_weights = [
        int(value)
        for value in re.findall(
            r"working_unit [^\n]*\bdelta_weights=(\d+)",
            text,
        )
    ]
    if frontier_delta_weights:
        result["frontier_delta_weights_total"] = sum(
            frontier_delta_weights)
        result["frontier_delta_weights_max"] = max(
            frontier_delta_weights)
    frontier_publish_ms = [
        float(value)
        for value in re.findall(
            r"(?:working_unit [^\n]*|cleared working-unit frontier "
            r"[^\n]*)\bpublish_ms=([0-9.]+)",
            text,
        )
    ]
    if frontier_publish_ms:
        result["frontier_publish_ms_total"] = sum(
            frontier_publish_ms)
        result["frontier_publish_ms_max"] = max(
            frontier_publish_ms)
    # The first publication installs the initial frontier and is not a
    # split/merge transition. Charge only publications whose exact frontier
    # signature differs from the preceding applied frontier. This separates
    # dynamic edit cost from one-time startup installation cost.
    frontier_publications = [
        (signature.lower(), float(publish_ms), int(delta_weights))
        for signature, publish_ms, delta_weights in re.findall(
            r"working_unit [^\n]*\bfrontier_sig=([0-9a-fA-F]{16})\b"
            r"[^\n]*\bpublish_ms=([0-9.]+)"
            r"[^\n]*\bdelta_weights=(\d+)",
            text,
        )
    ]
    transition_publications = [
        current
        for previous, current in zip(
            frontier_publications, frontier_publications[1:])
        if current[0] != previous[0]
    ]
    if transition_publications:
        transition_publish_ms = [
            publication[1] for publication in transition_publications
        ]
        result["frontier_transition_publish_count"] = len(
            transition_publications)
        result["frontier_transition_publish_ms_total"] = sum(
            transition_publish_ms)
        result["frontier_transition_publish_ms_max"] = max(
            transition_publish_ms)
        transition_delta_weights = [
            publication[2] for publication in transition_publications
        ]
        result["frontier_transition_delta_weights_total"] = sum(
            transition_delta_weights)
        result["frontier_transition_delta_weights_max"] = max(
            transition_delta_weights)
    result["frontier_only_apply_count"] = len(re.findall(
        r"applied frontier-only delta_weights=\d+",
        text,
    ))
    stage_defer_values = [
        int(value)
        for value in re.findall(
            r"apply_exec_plan: applied plan [^\n]*\bstage_defer=(\d+)",
            text,
        )
    ]
    if stage_defer_values:
        result["plan_stage_defer_apply_count"] = len(
            stage_defer_values)
        result["plan_stage_defer_min"] = min(stage_defer_values)
        result["plan_stage_defer_max"] = max(stage_defer_values)
    anchor_summary = re.search(
        r"elastic anchor summary: requests=\d+\s+hits=\d+\s+"
        r"duplicate=\d+\s+fired=(\d+)\s+load=(\d+)\s+"
        r"transfer=(\d+)\s+xform=(\d+)\s+failures=\d+",
        text,
    )
    if anchor_summary:
        result["plan_anchor_fired"] = int(anchor_summary.group(1))
        result["plan_anchor_load_events"] = int(
            anchor_summary.group(2))
        result["plan_anchor_transfer_events"] = int(
            anchor_summary.group(3))
        result["plan_anchor_xform_events"] = int(
            anchor_summary.group(4))
    if result["forward_runs"]:
        runs = int(result["forward_runs"])
        if runs > 0:
            if result["reload_host_issue_ms"] != "":
                result["reload_host_issue_ms_per_forward"] = float(result["reload_host_issue_ms"]) / runs
            if result["direct_read_ms"] != "":
                result["direct_read_ms_per_forward"] = float(result["direct_read_ms"]) / runs
            if result["direct_read_mb"] != "":
                result["direct_read_mb_per_forward"] = float(result["direct_read_mb"]) / runs
    m = re.search(
        r"moe expert cache: tensors=\d+ bytes=([0-9.]+)MB hits=(\d+) misses=(\d+) "
        r"hit_rate=([0-9.]+)% disk=([0-9.]+)MB/([0-9.]+)ms",
        text,
    )
    if m:
        result["moe_cache_mib"] = float(m.group(1))
        result["moe_cache_hits"] = int(m.group(2))
        result["moe_cache_misses"] = int(m.group(3))
        result["moe_cache_hit_rate"] = float(m.group(4))
        result["moe_cache_disk_mb"] = float(m.group(5))
        result["moe_cache_disk_ms"] = float(m.group(6))
    m = re.search(r"moe cache resize: syncs=(\d+)", text)
    if m:
        result["moe_cache_resize_syncs"] = int(m.group(1))
    m = re.search(r"\[elastic-online\] calls=(\d+) failures=(\d+)", text)
    if m:
        result["online_calls"] = int(m.group(1))
        result["online_failures"] = int(m.group(2))
    for tensor_suffix in ("token_embd", "output"):
        values = re.findall(
            rf"(?:ggml_opencl )?elastic: pin {tensor_suffix} "
            r"inside weight budget \(([0-9.]+) MiB",
            text,
        )
        result[f"pin_{tensor_suffix}_inside_budget_count"] = len(values)
        result[f"pin_{tensor_suffix}_inside_budget_mib"] = sum(
            float(value) for value in values)
    result["pin_outside_budget_count"] = len(re.findall(
        r"(?:ggml_opencl )?elastic: pin (?:unplanned output|token_embd)"
        r".*?outside budget",
        text,
    ))
    return result


def pipeline_worker_was_required(
        parsed: dict[str, Any], worker: str) -> bool:
    """Return whether a configured asynchronous worker had runtime work.

    Fully resident plans legitimately create neither LOAD nor PREPARE workers.
    Missing affinity diagnostics are therefore invalid only when the log also
    shows work for that worker.
    """
    if worker == "load":
        fields = ("load_planned", "direct_read_calls", "reload_calls")
    elif worker == "prepare":
        fields = ("xform_planned", "stage_xform_calls")
    elif worker == "copy":
        fields = ("fused_layout_parallel_calls",)
    else:
        raise ValueError(f"unknown pipeline worker: {worker}")
    return any(
        value not in ("", None) and float(value) > 0
        for value in (parsed.get(field, "") for field in fields)
    )


def frontier_contract_failures(
        method: str, parsed: dict[str, Any]) -> list[str]:
    """Return missing mixed-planner execution evidence for one device run."""
    failures: list[str] = []
    if method in {"offline-mixed", "online", "diff-tree-mixed"}:
        if int(parsed.get("frontier_apply_count") or 0) <= 0:
            failures.append("working-unit frontier was not applied")
        active_modes = sum(
            int(parsed.get(f"frontier_{mode}_units_max") or 0) > 0
            for mode in ("multi", "tensor", "cut")
        )
        if active_modes < 2:
            failures.append(
                "mixed-granularity frontier was not exercised")
        if int(parsed.get(
                "frontier_mixed_mode_generations") or 0) <= 0:
            failures.append(
                "no applied frontier simultaneously used multiple modes")
    if method in {"online", "diff-tree-mixed"}:
        if int(parsed.get("online_calls") or 0) <= 0:
            failures.append("online planner was not exercised")
    return failures


def unit_pipeline_stage_authority_failures(
    parsed: dict[str, Any],
) -> list[str]:
    """Reject whole-weight plan staging in a working-unit experiment.

    Placement reconciliation may still load a newly persistent weight once.
    Recurring DISK materialization, however, must be owned exclusively by the
    backend's physical unit pipeline. Otherwise plan apply eagerly stages
    whole weights while Cut/Tensor/Multi are also staged by physical unit,
    double-counting transitions and collapsing Cut's fine-grained overlap.
    """
    failures: list[str] = []
    if int(parsed.get("plan_stage_defer_apply_count") or 0) <= 0:
        failures.append("plan stage-defer audit is missing")
    elif (
        int(parsed.get("plan_stage_defer_min") or 0) != 1
        or int(parsed.get("plan_stage_defer_max") or 0) != 1
    ):
        failures.append(
            "whole-weight plan stages were issued eagerly")
    for field, label in (
        ("plan_anchor_fired", "plan anchor"),
        ("plan_anchor_load_events", "plan LOAD anchor"),
        ("plan_anchor_transfer_events", "plan TRANSFER anchor"),
        ("plan_anchor_xform_events", "plan PREPARE anchor"),
    ):
        value = parsed.get(field, "")
        if value in ("", None):
            failures.append(f"{label} audit is missing")
        elif int(value) != 0:
            failures.append(
                f"{label} fired alongside the physical unit pipeline")
    return failures


def backend_compute_contract_failures(
    backend: str,
    parsed: dict[str, Any],
    expected_wbm_total_mib: float = 0.0,
    require_cpu_multi_fused: bool = False,
) -> list[str]:
    """Audit optimized backend paths required by an exercised frontier."""
    failures: list[str] = []
    physical_cut_tensors = int(
        parsed.get("physical_cut_tensors") or 0)
    physical_cut_parts = int(
        parsed.get("physical_cut_parts") or 0)
    if physical_cut_tensors <= 0:
        failures.append(
            "common dynamic-capable physical tiling did not execute")
    elif physical_cut_parts != 2 * physical_cut_tensors:
        failures.append(
            "common physical representation is not exactly two tiles "
            "per cut-capable logical tensor")
    wbm_total_mib = float(parsed.get("wbm_total_mib") or 0.0)
    if expected_wbm_total_mib > 0.0:
        tolerance_mib = max(0.01, expected_wbm_total_mib * 1e-5)
        if abs(wbm_total_mib - expected_wbm_total_mib) > tolerance_mib:
            failures.append(
                "physical WBM total differs from expected model weight "
                f"bytes: observed={wbm_total_mib:.6f} "
                f"expected={expected_wbm_total_mib:.6f} MiB")
    if backend == "cpu":
        if (
            require_cpu_multi_fused
            and int(parsed.get("frontier_multi_units_max") or 0) > 0
        ):
            if int(parsed.get(
                    "fused_kernel_pair_candidates") or 0) <= 0:
                failures.append(
                    "Multi-fused frontier produced no fused MUL_MAT "
                    "candidate")
            if int(parsed.get("fused_kernel_pair_calls") or 0) <= 0:
                failures.append(
                    "Multi-fused frontier did not execute fused MUL_MAT")
            if int(parsed.get("fused_kernel_pair_errors") or 0) != 0:
                failures.append(
                    "Multi-fused frontier reported a fused MUL_MAT error")
        return failures
    if backend != "gpu":
        return failures
    # All GPU methods share the same pre-provisioned two-tile physical Q4
    # representation so split/merge is allocation-free. Logical Multi and
    # Tensor must therefore receive the same fused dual-half GEMV path as Cut.
    if int(parsed.get("cut_dual_candidates") or 0) <= 0:
        failures.append("tiled weights did not reach dual-GEMV candidates")
    if int(parsed.get("cut_dual_calls") or 0) <= 0:
        failures.append("tiled weights did not execute fused dual GEMV")
    if int(parsed.get("cut_dual_budget_fallbacks") or 0) != 0:
        failures.append(
            "tiled weights fell back from fused dual GEMV because both "
            "tiles did not fit the active weight budget")
    if int(parsed.get("cut_dual_image_creates") or 0) <= 0:
        failures.append("fused dual-GEMV image cache did not execute")
    if int(parsed.get("cut_dual_queue_errors") or 0) != 0:
        failures.append("Cut fused dual-GEMV queue error")
    if int(parsed.get("cut_dual_image_errors") or 0) != 0:
        failures.append("Cut fused dual-GEMV image-view error")
    if (
        int(parsed.get("cut_dual_image_releases") or 0)
        > int(parsed.get("cut_dual_image_creates") or 0)
    ):
        failures.append("fused dual-GEMV image release accounting error")
    return failures


def remote_solver_log_contract(
    method: str,
    path: Path | None,
    expected_run_id: str,
    max_diff_edits: int | None = None,
    expected_placement_source: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Prove Online/Diff used one fresh, method-correct solver process."""
    remote_methods = {"online", "diff-tree-mixed"}
    if method not in remote_methods:
        return [], {
            "remote_solver_event_count": 0,
            "remote_solver_run_ids": "",
            "remote_solver_working_unit_policies": "",
            "remote_solver_placement_sources": "",
            "remote_solver_diff_edit_count_max": 0,
        }
    failures: list[str] = []
    events: list[dict[str, Any]] = []
    if path is None or not path.is_file():
        failures.append("fresh remote solver log missing")
    else:
        for line in path.read_text(
                encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") in {"solve", "solve-cache"}:
                events.append(record)
    run_ids = {
        str(record.get("run_id", ""))
        for record in events
    }
    policies = {
        str(record.get("working_unit_policy", ""))
        for record in events
    }
    if not any(record.get("event") == "solve" for record in events):
        failures.append("fresh remote solver performed no cold solve")
    if run_ids != {expected_run_id}:
        failures.append(
            "remote solver log contains another or missing run id")
    # These are the canonical policies serialized into ExecPlan.working_unit
    # by attach_working_unit(), not the shorter command-line policy names.
    expected_policy = (
        "online-global-mixed"
        if method == "online" else "diff-tree-mixed")
    if policies != {expected_policy}:
        failures.append(
            f"remote solver policy is not {expected_policy}")
    cold_events = [
        record for record in events
        if record.get("event") == "solve"
    ]
    placement_sources = {
        str(record.get("online_placement_source", ""))
        for record in cold_events
    }
    if expected_placement_source is not None:
        expected_runtime_source = {
            "offline-table": "precomputed-offline-budget-table",
            "stateful-cp": "stateful-runtime-cp",
        }.get(expected_placement_source)
        if expected_runtime_source is None:
            failures.append(
                "unknown expected remote placement source")
        elif placement_sources != {expected_runtime_source}:
            failures.append(
                "remote solver placement source is not "
                f"{expected_runtime_source}")
    diff_edit_counts: list[int] = []
    if method == "diff-tree-mixed":
        for record in cold_events:
            value = record.get("edit_count")
            if not isinstance(value, int) or value < 0:
                failures.append(
                    "Diff-tree solver edit-count audit missing")
                continue
            diff_edit_counts.append(value)
            if (
                max_diff_edits is not None
                and value > max_diff_edits
            ):
                failures.append(
                    "Diff-tree solver exceeded configured edit bound")
    return failures, {
        "remote_solver_event_count": len(events),
        "remote_solver_run_ids": ",".join(sorted(run_ids)),
        "remote_solver_working_unit_policies":
            ",".join(sorted(policies)),
        "remote_solver_placement_sources":
            ",".join(sorted(placement_sources)),
        "remote_solver_diff_edit_count_max":
            max(diff_edit_counts, default=0),
    }


def timed_run_contract_failures(
    parsed: dict[str, Any], bench_seconds: float | None,
) -> list[str]:
    """Require a duration-limited run to reach its real timed boundary."""
    if bench_seconds is None or bench_seconds <= 0.0:
        return []
    failures: list[str] = []
    if parsed.get("bench_exit_reason") != "duration":
        failures.append("duration-limited benchmark exited early")
    if int(parsed.get("bench_time_done") or 0) != 1:
        failures.append("duration-limited benchmark did not reach time limit")
    reset_count = int(parsed.get("budget_decode_reset_count") or 0)
    if reset_count != 1:
        failures.append(
            f"decode-boundary budget reset count={reset_count}, expected 1")
    backend_reset_count = int(
        parsed.get("budget_backend_reset_count") or 0)
    if backend_reset_count != 1:
        failures.append(
            f"backend budget reset count={backend_reset_count}, expected 1")
    if (
        reset_count == 1
        and int(parsed.get("budget_decode_reset_generated", -1)) != 0
    ):
        failures.append("budget replay did not start at first decode")
    return failures


def decode_wall_contract_failures(
    parsed: dict[str, Any],
) -> list[str]:
    """Require an independent end-to-end decode critical-path timer.

    llama.cpp's standard eval timer begins after scheduler/provider hooks.
    Formal dynamic-budget comparisons therefore use the interval from entry
    to ``llama_context::decode()`` through the synchronization that makes its
    logits consumable.  The component sum remains a diagnostic cross-check.
    """
    failures: list[str] = []
    eval_runs = int(parsed.get("eval_runs") or 0)
    wall_runs = int(parsed.get("decode_wall_runs") or 0)
    wall_total = float(parsed.get("decode_wall_ms_total") or 0.0)
    if eval_runs <= 0:
        failures.append("standard eval timer has no decode runs")
    if wall_runs <= 0:
        failures.append("decode wall timer is missing")
    elif wall_runs != eval_runs:
        failures.append(
            f"decode wall runs={wall_runs}, eval runs={eval_runs}")
    if wall_total <= 0.0:
        failures.append("decode wall time is non-positive")
    component_total = (
        float(parsed.get("eval_ms_total") or 0.0)
        + float(parsed.get("provider_get_ms_total") or 0.0)
        + float(parsed.get("apply_ms_total") or 0.0)
    )
    # All components lie inside the wall interval. Permit only log-rounding
    # noise; a shorter wall interval proves a timer-boundary or aggregation
    # bug and must never rank a formal result.
    tolerance_ms = max(5.0, component_total * 0.001)
    if wall_total + tolerance_ms < component_total:
        failures.append(
            f"decode wall={wall_total:.3f}ms is shorter than contained "
            f"components={component_total:.3f}ms")
    if parsed.get("exec_timing_source") != "decode-wall":
        failures.append("exec latency is not sourced from decode wall timer")
    return failures


def decode_phase_contract_failures(
    parsed: dict[str, Any],
    execution_backend: str,
) -> list[str]:
    """Validate the prompt-excluding backend counter delta."""
    failures: list[str] = []
    expected_backend = "cpu" if execution_backend == "cpu" else "gpu"
    if int(parsed.get("decode_phase_origin_count") or 0) != 1:
        failures.append("decode phase origin count is not one")
    if int(parsed.get("decode_phase_quiesce_count") or 0) != 2:
        failures.append(
            "decode phase quiescence boundary count is not two")
    if parsed.get("decode_phase_quiesce_sequence") != "begin,end":
        failures.append(
            "decode phase quiescence sequence is not begin,end")
    if (
        parsed.get("decode_phase_quiesce_begin_rc") in (None, "")
        or int(parsed["decode_phase_quiesce_begin_rc"]) != 0
    ):
        failures.append("decode phase begin quiescence failed")
    if (
        parsed.get("decode_phase_quiesce_end_rc") in (None, "")
        or int(parsed["decode_phase_quiesce_end_rc"]) != 0
    ):
        failures.append("decode phase end quiescence failed")
    origin_generated = parsed.get(
        "decode_phase_origin_generated", "")
    if (
        origin_generated in ("", None)
        or int(origin_generated) != 0
    ):
        failures.append("decode phase origin is not generated token zero")
    if parsed.get("decode_phase_origin_backend") != expected_backend:
        failures.append("decode phase origin backend mismatch")
    if int(parsed.get("decode_phase_summary_count") or 0) != 1:
        failures.append("decode phase summary count is not one")
    if parsed.get("decode_phase_backend") != expected_backend:
        failures.append("decode phase summary backend mismatch")
    phase_runs = int(parsed.get("decode_phase_runs") or 0)
    eval_runs = int(parsed.get("eval_runs") or 0)
    wall_runs = int(parsed.get("decode_wall_runs") or 0)
    if phase_runs <= 0 or phase_runs != eval_runs or phase_runs != wall_runs:
        failures.append(
            f"decode phase runs={phase_runs}, eval runs={eval_runs}, "
            f"wall runs={wall_runs}")
    if int(parsed.get("decode_phase_units") or 0) <= 0:
        failures.append("decode phase has no working units")
    if parsed.get("decode_phase_counter_scope") != (
            "decode-only-backend-counter-delta"):
        failures.append("decode phase counter scope is not decode-only")
    required_metrics = (
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
    missing_metrics = [
        name for name in required_metrics
        if parsed.get(name, "") in ("", None)
    ]
    if missing_metrics:
        failures.append(
            "decode phase metrics missing: "
            + ",".join(missing_metrics))
    compute_available = int(
        parsed.get("decode_phase_compute_available") or 0)
    compute_calls = int(parsed.get("decode_phase_compute_calls") or 0)
    compute_ms = float(parsed.get("decode_phase_compute_ms") or 0.0)
    if execution_backend == "cpu":
        if compute_available != 1:
            failures.append("CPU decode compute timer is unavailable")
        if compute_calls <= 0 or compute_ms <= 0.0:
            failures.append("CPU decode compute phase is empty")
    elif compute_available != 0 or compute_calls != 0 or compute_ms != 0.0:
        failures.append(
            "GPU decode compute phase must be unavailable without event "
            "profiling")
    return failures


def cpu_frequency_contract_failures(
    parsed: dict[str, Any],
    *,
    min_limit_khz: int,
    min_mean_khz: int,
    min_median_sample_min_khz: int,
) -> list[str]:
    """Fail closed when a requested CPU-frequency audit is unavailable."""
    failures: list[str] = []
    for field, threshold in (
        ("cpu_freq_limit_min_khz", min_limit_khz),
        ("cpu_freq_mean_khz", min_mean_khz),
        (
            "cpu_freq_median_sample_min_khz",
            min_median_sample_min_khz,
        ),
    ):
        if threshold <= 0:
            continue
        value = parsed.get(field, "")
        if value in ("", None):
            failures.append(f"{field}=missing<{threshold}")
        elif float(value) < threshold:
            failures.append(f"{field}={float(value):.0f}<{threshold}")
    return failures


DYNAMIC_RUN_CONFIG_KEYS = (
    "execution_backend", "window_sec", "window_stride_sec",
    "window_start_sec", "replay_speedup", "bench_seconds", "bucket_mib",
    "original_source_trace", "expected_original_source_trace_sha256",
    "extra_max_budget_mib", "kv_mib", "misc_mib", "safety_mib",
    "pinned_extra_mib", "expected_pin_token_embd_mib",
    "expected_pin_output_mib", "expected_wbm_total_mib",
    "planner_stream_reserve_mib",
    "release_stage_after_xform", "time_limit_ms", "prefetch_distance",
    "transition_weight", "diff_horizon_tokens", "plan_up_step_buckets",
    "plan_switch_min_gap_steps", "online_plan_switch_min_gain_ms",
    "disk_reload_multiplier", "disk_gpu_reload_multiplier",
    "overlap_model", "static_overlap_model", "cp_objective",
    "allowed_placements", "force_weight_placement",
    "allow_cpu_fallback", "granularity_policy", "granularity_backend",
    "granularity_placement_source", "granularity_horizon_tokens",
    "granularity_min_gain_ms", "granularity_max_edits",
    "granularity_beam_width", "granularity_pipeline_lookahead",
    "granularity_pipeline_lookahead_mib", "plan_residency_policy",
    "online_ignore_state", "top_k", "candidate_placement_specs",
    "candidate_min_distance", "candidate_min_gain_ms",
    "mru_allowed_placements", "static_weight_mirror",
    "moe_expert_cache_adaptive", "moe_expert_cache_base_slots",
    "moe_retain_workload_high_water", "dynamic_active_experts",
    "dynamic_total_experts", "use_interval_schedule",
    "interval_stage_kinds", "extra_env", "llama_extra_arg", "n_pred",
    "ctx_size", "batch", "threads", "cpu_mask", "cpu_strict", "poll",
    "min_cpu_freq_limit_khz", "min_cpu_mean_freq_khz",
    "min_cpu_median_sample_min_khz", "max_pipeline_budget_violations",
    "max_plan_protection_relaxations", "fixed_performance_mode",
    "n_gpu_layers", "prompt", "show_token_output", "token_prefix_count",
    "require_device_idle", "cooldown_thermal_max_c",
    "cooldown_thermal_status_max", "min_battery_level_pct",
)


def dynamic_run_config_payload(
    args: argparse.Namespace,
    *,
    omit: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return the canonical dynamic-runtime configuration payload."""
    payload = {
        key: getattr(args, key, None)
        for key in DYNAMIC_RUN_CONFIG_KEYS
        if key not in omit
    }
    payload = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in payload.items()
    }
    payload["host_runner_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    return payload


def dynamic_run_config_sha256(args: argparse.Namespace) -> str:
    """Fingerprint the complete runtime contract for resume/consolidation.

    Method list, output paths, and granularity-profile bytes are intentionally
    excluded: Diff-before is a separate one-method artifact and its distinct
    profile hash is audited independently. Everything that changes the common
    runtime, planner, pipeline, budget, affinity, or validity contract is
    included.
    """
    payload = dynamic_run_config_payload(args)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dynamic_run_core_config_sha256(args: argparse.Namespace) -> str:
    """Fingerprint the common contract used by residency-policy screening.

    The residency pilot deliberately varies only ``plan_residency_policy``.
    Its full configuration hashes must therefore differ, while this core hash
    must remain identical.  Keeping the stream reserve in this hash proves
    that every candidate receives the same physical headroom.
    """
    payload = dynamic_run_config_payload(
        args, omit=frozenset({"plan_residency_policy"}))
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resume_environment_contract_failures(
    args: argparse.Namespace,
    row: dict[str, Any],
) -> list[str]:
    """Reject cached rows collected under weaker or unaudited conditions."""
    failures: list[str] = []
    if (
        args.require_device_idle
        and str(row.get(
            "require_device_idle_configured", ""
        )).lower() != "true"
    ):
        failures.append("device-idle configuration missing")
    if args.cooldown_thermal_max_c > 0:
        configured = row.get(
            "cooldown_thermal_max_c_configured", "")
        if (
            configured in ("", None)
            or float(configured) > args.cooldown_thermal_max_c
        ):
            failures.append("cooldown temperature configuration weaker")
        status_configured = row.get(
            "cooldown_thermal_status_max_configured", "")
        if (
            status_configured in ("", None)
            or int(float(status_configured))
                > args.cooldown_thermal_status_max
        ):
            failures.append("thermal-status configuration weaker")
        before = {
            key.removesuffix("_before"): row.get(key, "")
            for key in (
                "thermal_status_before",
                "thermal_cpu_max_c_before",
                "thermal_gpu_max_c_before",
                "thermal_skin_max_c_before",
                "battery_level_pct_before",
            )
        }
        if not cooldown_snapshot_ready(args, before):
            failures.append("recorded clean-start snapshot invalid")
    if args.min_battery_level_pct > 0:
        configured = row.get(
            "min_battery_level_pct_configured", "")
        if (
            configured in ("", None)
            or int(float(configured)) < args.min_battery_level_pct
        ):
            failures.append("battery configuration weaker")
    if args.execution_backend == "cpu":
        failures.extend(cpu_frequency_contract_failures(
            row,
            min_limit_khz=args.min_cpu_freq_limit_khz,
            min_mean_khz=args.min_cpu_mean_freq_khz,
            min_median_sample_min_khz=(
                args.min_cpu_median_sample_min_khz),
        ))
    return failures


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


def token_sequence(
    path: Path,
    max_ids: int | None = None,
    *,
    decode_only: bool = True,
) -> tuple[int, str | None]:
    """Return the number and SHA-256 of recorded decode token ids.

    The first CSV row is normally the shared prompt batch.  Excluding it makes
    the correctness check sensitive to generated-output divergence rather than
    merely confirming that all methods received the same prompt.
    """
    if not path.exists():
        return 0, None
    token_ids: list[int] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if decode_only and int(row.get("n_tokens") or 0) != 1:
                    continue
                for item in str(row.get("token_ids", "")).split(";"):
                    if item:
                        token_ids.append(int(item))
    except (OSError, ValueError):
        return 0, None
    if max_ids is not None:
        token_ids = token_ids[:max(0, max_ids)]
    if not token_ids:
        return 0, None
    payload = ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    return len(token_ids), hashlib.sha256(payload).hexdigest()


def token_trace_timing(path: Path) -> dict[str, Any]:
    """Summarize decode-row timing from the authoritative token CSV."""
    result: dict[str, Any] = {
        "token_trace_decode_rows": 0,
        "token_trace_t_first_sec": "",
        "token_trace_t_last_sec": "",
        "token_trace_t_span_sec": "",
        "token_trace_first_budget_mib": "",
        "token_trace_last_budget_mib": "",
        "token_trace_first_latency_ms": "",
        "token_trace_max_latency_ms": "",
        "token_trace_time_monotonic": False,
        "token_trace_budget_provider_valid": False,
    }
    if not path.exists():
        return result
    times: list[float] = []
    budgets: list[int] = []
    latencies: list[float] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row.get("n_tokens") or 0) != 1:
                    continue
                times.append(float(row["t_sec"]))
                budgets.append(int(float(row["budget_mib"])))
                latencies.append(float(row["latency_ms"]))
    except (KeyError, OSError, TypeError, ValueError):
        return result
    if not times:
        return result
    result.update({
        "token_trace_decode_rows": len(times),
        "token_trace_t_first_sec": times[0],
        "token_trace_t_last_sec": times[-1],
        "token_trace_t_span_sec": times[-1] - times[0],
        "token_trace_first_budget_mib": budgets[0],
        "token_trace_last_budget_mib": budgets[-1],
        "token_trace_first_latency_ms": latencies[0],
        "token_trace_max_latency_ms": max(latencies),
        "token_trace_time_monotonic": all(
            current >= previous
            for previous, current in zip(times, times[1:])
        ),
        "token_trace_budget_provider_valid": all(
            budget >= 0 for budget in budgets),
    })
    return result


def token_trace_timing_contract_failures(
    timing: dict[str, Any],
    bench_seconds: float | None,
) -> list[str]:
    """Prove that decode samples cover the same wall-clock replay interval."""
    if bench_seconds is None or bench_seconds <= 0.0:
        return []
    rows = int(timing.get("token_trace_decode_rows") or 0)
    if rows <= 0:
        return ["decode token timing trace is empty"]
    failures: list[str] = []
    if not bool(timing.get("token_trace_time_monotonic")):
        failures.append("decode token timestamps are not monotonic")
    if not bool(timing.get("token_trace_budget_provider_valid")):
        failures.append("decode token trace has no valid budget provider")
    first = float(timing.get("token_trace_t_first_sec") or -1.0)
    last = float(timing.get("token_trace_t_last_sec") or -1.0)
    first_latency = max(
        0.0, float(timing.get("token_trace_first_latency_ms") or 0.0)
    ) / 1000.0
    max_latency = max(
        0.0, float(timing.get("token_trace_max_latency_ms") or 0.0)
    ) / 1000.0
    # The timestamp is written after synchronization, so the first point can
    # legitimately be one decode latency after the replay origin. Likewise,
    # the duration check is sampled immediately after the previous decode.
    start_tolerance = max(2.0, 2.0 * first_latency)
    end_tolerance = max(2.0, 2.0 * max_latency)
    if first < 0.0 or first > start_tolerance:
        failures.append(
            f"first decode timestamp={first:.3f}s exceeds "
            f"{start_tolerance:.3f}s origin tolerance")
    if last < float(bench_seconds) - end_tolerance:
        failures.append(
            f"last decode timestamp={last:.3f}s does not cover "
            f"{float(bench_seconds):.3f}s replay")
    return failures


def mark_token_correctness(
    rows: list[dict[str, Any]],
    methods: set[str],
    prefix_count: int,
) -> None:
    """Compare equal-length deterministic token prefixes across all methods."""
    if len(methods) < 2:
        return
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("trace", "")), []).append(row)
    for values in groups.values():
        by_method = {
            str(row.get("method", "")): row
            for row in values
            if str(row.get("method", "")) in methods
            and row.get("token_prefix_sha256") not in (None, "")
            and int(row.get("token_prefix_count") or 0) == prefix_count
        }
        if not methods.issubset(by_method):
            continue
        hashes = {
            str(by_method[method]["token_prefix_sha256"])
            for method in methods
        }
        valid = len(hashes) == 1
        for method in methods:
            row = by_method[method]
            row["valid_token_sequence"] = valid
            if not valid and row.get("status") == "ok":
                row["status"] = "token_mismatch"
            elif valid and row.get("status") == "token_mismatch":
                # Resume may replace the one divergent method after the
                # mismatch had marked every compared row. Recover only this
                # derived status; never erase thermal/budget/runtime errors.
                row["status"] = "ok"


def matrix_completion_failures(
    rows: list[dict[str, Any]],
    *,
    trace_names: set[str],
    methods: set[str],
    require_device_idle: bool,
    require_fixed_performance_restore: bool = False,
) -> list[str]:
    """Return fail-closed reasons before a matrix invocation exits."""
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("trace", "")), str(row.get("method", "")))
        if key[0] in trace_names and key[1] in methods:
            by_key.setdefault(key, []).append(row)
    failures: list[str] = []
    for trace_name in sorted(trace_names):
        for method in sorted(methods):
            key = (trace_name, method)
            matches = by_key.get(key, [])
            label = f"{trace_name}:{method}"
            if len(matches) != 1:
                failures.append(f"{label}: rows={len(matches)}")
                continue
            row = matches[0]
            reasons: list[str] = []
            if row.get("status") != "ok":
                reasons.append(f"status={row.get('status')}")
            if str(row.get("rc", "")) != "0":
                reasons.append(f"rc={row.get('rc')}")
            if str(row.get(
                    "valid_token_trace", "")).lower() != "true":
                reasons.append("token-trace")
            if str(row.get(
                    "token_trace_timing_contract_valid",
                    "")).lower() != "true":
                reasons.append("token-trace-timing")
            if str(row.get(
                    "decode_wall_contract_valid",
                    "")).lower() != "true":
                reasons.append("decode-wall-timing")
            if str(row.get(
                    "decode_phase_contract_valid",
                    "")).lower() != "true":
                reasons.append("decode-phase")
            if str(row.get(
                    "valid_token_sequence", "")).lower() == "false":
                reasons.append("token-sequence")
            if (
                require_fixed_performance_restore
                and str(row.get(
                    "fixed_performance_restore_valid", "")
                ).lower() != "true"
            ):
                reasons.append("fixed-performance-restore")
            if require_device_idle:
                if str(row.get(
                        "valid_measurement_isolation", "")
                       ).lower() != "true":
                    reasons.append("measurement-isolation")
                if int(float(
                        row.get("measurement_monitor_samples") or 0)) <= 0:
                    reasons.append("monitor-samples")
                if int(float(
                        row.get("measurement_monitor_failures") or 0)) != 0:
                    reasons.append("monitor-failures")
                if not str(
                        row.get("measurement_llama_pid") or "").strip():
                    reasons.append("measurement-process")
                if str(row.get(
                        "monitor_thread_incomplete", "false")
                       ).lower() == "true":
                    reasons.append("monitor-thread")
            if reasons:
                failures.append(f"{label}: {','.join(reasons)}")
    return failures


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
        f"- cooldown thermal status max: `{args.cooldown_thermal_status_max}`",
        f"- minimum battery level: `{args.min_battery_level_pct}`%",
        f"- adaptive MoE expert cache: `{args.moe_expert_cache_adaptive}`",
        f"- MoE workload high-water retention: `{args.moe_retain_workload_high_water}`",
        f"- Q4 MoE expert tensors: `{args.moe_expert_tensor_count}`",
        f"- MoE expert cache floor slots: `{args.moe_expert_cache_base_slots}`",
        f"- dynamic experts active/total: `{args.dynamic_active_experts}/{args.dynamic_total_experts}`",
        f"- online plan switch minimum gain: `{args.online_plan_switch_min_gain_ms}` ms/token",
        f"- plan residency policy: `{getattr(args, 'plan_residency_policy', 'strict')}`",
        "",
        "## Trace Windows",
        "",
        "| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in traces:
        lines.append(
            f"| {t.local.name} | {t.source_start:.0f} | {t.source_span:.0f} | {t.replay_span:.1f} | {t.rows} | "
            f"{t.min_mib:.1f} | {t.mean_mib:.1f} | {t.max_mib:.1f} | {t.min_bucket_mib} | {t.max_bucket_mib} |"
        )
    lines += [
        "",
        "## Results",
        "",
        "`raw ms/token` is llama's eval timer. `exec ms/token` adds "
        "decode-critical provider lookup/wait plus plan apply/movement time. "
        "Asynchronous remote solve wall time is reported separately and is "
        "not double counted.",
        "",
        "| trace | method | status | token prefix valid | raw ms/token | exec ms/token | thermal C | battery % | mean CPU MHz | cache MiB / hit % | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|",
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
        battery = (
            f"{fmt(r.get('battery_level_pct_before', ''))}/"
            f"{fmt(r.get('battery_level_pct_after', ''))}")
        mean_cpu_mhz = (
            float(r["cpu_freq_mean_khz"]) / 1000.0
            if r.get("cpu_freq_mean_khz", "") not in ("", None)
            else "")
        cache = f"{fmt(r.get('moe_cache_mib', ''))} / {fmt(r.get('moe_cache_hit_rate', ''))}"
        lines.append(
            f"| {r['trace']} | {r['method']} | {r['status']} | {fmt(r.get('valid_token_sequence', ''))} | "
            f"{fmt(r.get('raw_ms_per_token', ''))} | "
            f"{fmt(r.get('exec_ms_per_token', ''))} | {thermal} | "
            f"{battery} | {fmt(mean_cpu_mhz)} | {cache} | "
            f"{fmt(r.get('remote_wall_ms', ''))} | "
            f"{fmt(r.get('provider_get_ms_total', ''))} | {fmt(r.get('apply_count', ''))} | {planned} | {fmt(r.get('direct_read_ms', ''))} | "
            f"{fmt(r.get('direct_read_calls', ''))} | {fmt(failures)} |"
        )
    frontier_rows = [
        row for row in rows
        if int(row.get("frontier_apply_count") or 0) > 0
    ]
    if frontier_rows:
        lines += [
            "",
            "## Mixed-frontier audit",
            "",
            "| trace | method | frontier applies | generations using >1 mode | max Multi/Tensor/Cut units | final Multi/Tensor/Cut units |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in frontier_rows:
            maximum = "/".join(str(row.get(
                f"frontier_{mode}_units_max", ""))
                for mode in ("multi", "tensor", "cut"))
            final = "/".join(str(row.get(
                f"frontier_{mode}_units_last", ""))
                for mode in ("multi", "tensor", "cut"))
            lines.append(
                f"| {row['trace']} | {row['method']} | "
                f"{row.get('frontier_apply_count', 0)} | "
                f"{row.get('frontier_mixed_mode_generations', 0)} | "
                f"{maximum} | {final} |")
    path.write_text("\n".join(lines) + "\n")


def remote_shell_env(remote_dir: str, env: dict[str, str], argv: list[str], *, suppress_stdout: bool = False) -> str:
    exports = " ".join(f"export {k}={shell_quote(v)};" for k, v in env.items() if v is not None)
    command = " ".join(shell_quote(x) for x in argv)
    if suppress_stdout:
        command += " 1>/dev/null"
    script = (
        f"cd {shell_quote(remote_dir)} && {exports} "
        f"{command}; rc=$?; echo __LLAMA_INNER_RC__=$rc; exit $rc"
    )
    return script


def backend_safety_mib(args: argparse.Namespace) -> int:
    """Return the safety reserve enforced by the physical weight backend.

    Both CPU and OpenCL now register the requested token_embd physical units
    in the same WBM whose target this value controls.  The WBM therefore
    charges those bytes through resident-byte accounting.  The planner
    separately subtracts pinned_extra_mib only because token_embd is absent
    from model metadata; subtracting it again from the backend target would
    double-count the same inside-budget pin.
    """
    return args.safety_mib


def apply_plan_residency_policy(
    env: dict[str, str],
    method: str,
    policy: str,
) -> None:
    """Select how a placement plan and the live WBM share residency control.

    ``strict`` preserves the historical experiment: every plan-DISK weight is
    retired at the graph boundary and planned residents are protected.

    ``reuse-stream`` keeps the same protected resident identity but allows the
    last legal streaming working set to survive the graph boundary.  The unit
    pipeline still enforces the physical byte budget before every issue.

    ``cache-managed`` keeps the plan's mixed working-unit frontier but lets the
    WBM choose all movable victims inside the same hard byte budget.  It is an
    explicit co-design candidate, not a silent change to the MRU baseline.
    """
    valid = {"strict", "reuse-stream", "cache-managed"}
    if policy not in valid:
        raise ValueError(
            f"unknown plan residency policy {policy!r}; "
            f"expected one of {sorted(valid)}")
    env["LLAMA_ELASTIC_PLAN_RESIDENCY_POLICY"] = policy
    planned_methods = {
        "offline", "offline-mixed", "online", "diff-tree-mixed",
        "candidate-select", "diff-graph-expand", "diff-tree-ideal",
    }
    if method not in planned_methods or policy == "strict":
        return
    if policy == "reuse-stream":
        env["GGML_ELASTIC_NO_AUTO_EVICT"] = "1"
        env["LLAMA_ELASTIC_FORCE_PLAN_EVICT"] = "1"
        env["LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH"] = "0"
        return
    env["GGML_ELASTIC_NO_AUTO_EVICT"] = "0"
    env["LLAMA_ELASTIC_FORCE_PLAN_EVICT"] = "0"
    env["LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH"] = "0"


def configure_unit_pipeline_stage_authority(
    env: dict[str, str],
) -> None:
    """Make the physical unit pipeline the sole recurring stage producer."""
    env["LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE"] = "0"
    env["LLAMA_ELASTIC_DEFER_STAGE"] = "1"
    env["LLAMA_ELASTIC_INTERVAL_STAGE_KINDS"] = "none"


def make_method_env(args: argparse.Namespace, method: str, trace: TraceWindow, remote_trace: str, work_dir: str) -> dict[str, str]:
    effective_safety_mib = args.safety_mib + args.pinned_extra_mib
    allowed_for_method = args.mru_allowed_placements if method == "mru" else args.allowed_placements
    placement_set = {s.strip().lower() for s in str(allowed_for_method).split(",") if s.strip()}
    trace_rows = read_trace(trace.local)
    initial_budget_mib = bucket_floor(trace_rows[0][1], args.bucket_mib) if trace_rows else trace.min_bucket_mib
    common = {
        "LD_LIBRARY_PATH": args.remote_dir,
        "GGML_ELASTIC_TIMING": "1",
        "GGML_ELASTIC_STAGE_DETAIL": "1",
        "GGML_ELASTIC_TOKEN_CSV": f"{work_dir}/tokens.csv",
        "GGML_ELASTIC_BUDGET_CSV": remote_trace,
        # Model load and prompt ingestion are outside the measured replay.
        # Start B(t) exactly once at the first generated-token boundary and
        # never rewind it on an Online/Diff plan transition.
        "LLAMA_ELASTIC_RESET_BUDGET_ON_START": "0",
        "LLAMA_ELASTIC_RESET_BUDGET_ON_DECODE": "1",
        "LLAMA_ELASTIC_RESET_BUDGET_AFTER_INITIAL_APPLY": "0",
        "LLAMA_ELASTIC_RESET_BUDGET_AFTER_ONLINE_APPLY": "0",
        "GGML_ELASTIC_BUDGET_BUCKET_MB": str(args.bucket_mib),
        "GGML_ELASTIC_KV_MB": str(args.kv_mib),
        "GGML_ELASTIC_MISC_MB": str(args.misc_mib),
        "GGML_ELASTIC_SAFETY_MB": str(backend_safety_mib(args)),
        # Static traces are flattened to one budget, so their observed floor
        # is not a valid adaptive-cache baseline. Keep all methods anchored to
        # the source window's floor for equal slots at equal budgets.
        "GGML_ELASTIC_MOE_EXPERT_CACHE_BASE_BUDGET_MIB": str(trace.min_bucket_mib),
        "LLAMA_ELASTIC_DEFER_STAGE": "0",
        # Placement plans are authoritative in this comparison. Do not let a
        # backend silently keep output.weight or the CPU default K/V set
        # resident after the solver placed them on disk. Legacy experiments
        # can opt back in explicitly through --extra-env.
        "GGML_ELASTIC_PIN_UNPLANNED_OUTPUT": "0",
        "GGML_ELASTIC_PIN": "",
    }
    if args.execution_backend == "gpu":
        common.update({
            "GGML_OPENCL_DISABLE_ALLOC_HOST_PTR": "1",
            "GGML_OPENCL_USE_SVM": "1",
            "GGML_OPENCL_ELASTIC": "1",
        })
    else:
        common.update({
            "GGML_ELASTIC_PROFILE": "1",
            "GGML_ELASTIC_HOST_STAGING_POOL_MB": "128",
        })
    if args.granularity_policy != "none":
        # Provision a common two-tile physical representation before the first
        # plan is applied. Every method then changes only logical unit ids;
        # split/merge never reallocates a live model tensor.
        common.update({
            "GGML_ELASTIC_GRANULARITY_DYNAMIC": "1",
            "GGML_ELASTIC_GRANULARITY": "tensor",
            "GGML_ELASTIC_CUT_PARTS": "2",
            "GGML_ELASTIC_MULTI_TENSORS": "2",
            "GGML_ELASTIC_UNIT_PIPELINE": "1",
            "GGML_ELASTIC_UNIT_PIPELINE_LOOKAHEAD": str(
                args.granularity_pipeline_lookahead),
            "GGML_ELASTIC_UNIT_PIPELINE_LOOKAHEAD_MB": str(
                args.granularity_pipeline_lookahead_mib),
            "LLAMA_ELASTIC_GRANULARITY_BACKEND":
                args.granularity_backend,
            "LLAMA_ELASTIC_GRANULARITY_HORIZON_TOKENS": str(
                args.granularity_horizon_tokens),
            "LLAMA_ELASTIC_GRANULARITY_MIN_GAIN_MS": str(
                args.granularity_min_gain_ms),
        })
    adaptive_moe_cache = (
        args.moe_expert_cache_adaptive == "1"
        or (args.moe_expert_cache_adaptive == "auto" and args.moe_expert_tensor_count > 0)
    )
    if adaptive_moe_cache:
        common["GGML_ELASTIC_MOE_EXPERT_CACHE"] = "1"
        common["GGML_ELASTIC_MOE_EXPERT_CACHE_ADAPTIVE"] = "1"
        if args.moe_expert_cache_base_slots > 0:
            common["GGML_ELASTIC_MOE_EXPERT_CACHE_SLOTS"] = str(args.moe_expert_cache_base_slots)
        # Keep the packed-expert replacement policy identical across the
        # planner and static baselines. MRU is the intentional exception: its
        # replacement policy is the baseline being measured.
        if method != "mru":
            common["GGML_ELASTIC_MOE_EXPERT_CACHE_POLICY"] = "lfu"
        workload_aware_methods = {
            "online", "diff-tree-mixed", "candidate-select",
            "diff-graph-expand", "diff-tree-ideal"}
        workload_exceeds_floor_cache = (
            args.moe_expert_cache_base_slots <= 0
            or args.dynamic_active_experts <= 0
            or args.dynamic_active_experts > args.moe_expert_cache_base_slots
        )
        retain_workload_high_water = (
            args.moe_retain_workload_high_water == "1"
            or (
                args.moe_retain_workload_high_water == "auto"
                and method in workload_aware_methods
                and workload_exceeds_floor_cache
            )
        )
        common["GGML_ELASTIC_MOE_RETAIN_WORKLOAD_HIGH_WATER"] = "1" if retain_workload_high_water else "0"
    if args.moe_expert_tensor_count > 0:
        common["GGML_ELASTIC_MOE_EXPERT_TENSOR_COUNT"] = str(args.moe_expert_tensor_count)
    if args.dynamic_active_experts > 0 and args.dynamic_total_experts > 0:
        common["LLAMA_ELASTIC_DYNAMIC_ACTIVE_EXPERTS"] = str(args.dynamic_active_experts)
        common["LLAMA_ELASTIC_DYNAMIC_TOTAL_EXPERTS"] = str(args.dynamic_total_experts)
    if (
        args.static_weight_mirror == "1" or
        (
            args.static_weight_mirror == "auto" and
            "cpu" in placement_set and "gpu" in placement_set
        )
    ):
        common["GGML_SCHED_STATIC_WEIGHT_MIRROR"] = "1"
    if args.pinned_extra_mib > 0:
        common["GGML_ELASTIC_PIN_UNPLANNED_OUTPUT_COUNTS_BUDGET"] = "1"
    def enable_plan_pipeline(env: dict[str, str]) -> None:
        env["LLAMA_ELASTIC_DEFER_STAGE"] = "1"
        env.setdefault("GGML_ELASTIC_ASYNC_STAGE_LOAD", "1")
        env.setdefault("GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD", str(max(4, args.prefetch_distance)))
        env.setdefault("GGML_ELASTIC_RELOAD_ON_XFER", "1")
        env.setdefault("GGML_ELASTIC_XFER_EXTRA", "1")
        env.setdefault("GGML_ELASTIC_NO_AUTO_EVICT", "1")
        env.setdefault("LLAMA_ELASTIC_ANCHOR_REPEAT_PER_GRAPH", "1")
        env.setdefault("LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH", "0")
        env.setdefault("GGML_ELASTIC_RELEASE_STAGE_AFTER_XFORM", "0")
        env.setdefault("GGML_ELASTIC_CACHE_FOREGROUND_LOAD", "1")
        env.setdefault("GGML_ELASTIC_SOA_STAGING_SLOTS", "4")
        env.setdefault("GGML_ELASTIC_CL_RETAIN", "1")
        env.setdefault("GGML_ELASTIC_CL_RETAIN_MB", "1024")
        if args.interval_stage_kinds:
            env["LLAMA_ELASTIC_INTERVAL_STAGE_KINDS"] = str(args.interval_stage_kinds)
            stage_kinds = {s.strip().lower() for s in str(args.interval_stage_kinds).split(",") if s.strip()}
            if "prepare" in stage_kinds or "all" in stage_kinds:
                env.setdefault("GGML_ELASTIC_ASYNC_STAGE_PREPARE", "1")
                env.setdefault("LLAMA_ELASTIC_ENABLE_CPU_XFORM_STAGE", "1")
    def enable_one_shot_transition_stages(env: dict[str, str]) -> None:
        # Candidate/diff-tree plans encode transition work for a budget switch.
        # That work must be paid once when the switch is applied.  Keep the
        # per-graph disk eviction path enabled so DISK weights do not remain
        # resident across tokens and accidentally exceed the active budget.
        env["LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE"] = "0"
        env["LLAMA_ELASTIC_DEFER_STAGE"] = "0"
        env["LLAMA_ELASTIC_INTERVAL_STAGE_KINDS"] = "none"
        env["LLAMA_ELASTIC_ANCHOR_REPEAT_PER_GRAPH"] = "1"
        env["LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH"] = "1"
        env["LLAMA_ELASTIC_PRELOAD_DISK_WEIGHTS_PER_GRAPH"] = "0"
        env["GGML_ELASTIC_ASYNC_STAGE_LOAD"] = "0"
        env["GGML_ELASTIC_ASYNC_STAGE_PREPARE"] = "0"
        env["GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD"] = "0"
        env["GGML_ELASTIC_RELOAD_ON_XFER"] = "0"
        env["GGML_ELASTIC_XFER_EXTRA"] = "0"
        env.setdefault("GGML_ELASTIC_NO_AUTO_EVICT", "1")
        env.setdefault("GGML_ELASTIC_CACHE_FOREGROUND_LOAD", "1")
        env.setdefault("GGML_ELASTIC_CL_RETAIN", "1")
        env.setdefault("GGML_ELASTIC_CL_RETAIN_MB", "500")
        env.setdefault("GGML_ELASTIC_RELEASE_STAGE_AFTER_XFORM", str(args.release_stage_after_xform))
    def disable_plan_pipeline(env: dict[str, str], *, strict_cache: bool = False) -> None:
        # Keep stage events deferred but do not project/trigger a staged
        # pipeline.  Disk weights then reach the normal foreground ensure path.
        env["LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE"] = "0"
        env["LLAMA_ELASTIC_DEFER_STAGE"] = "1"
        env["LLAMA_ELASTIC_INTERVAL_STAGE_KINDS"] = "none"
        env["GGML_ELASTIC_ASYNC_STAGE_LOAD"] = "0"
        env["GGML_ELASTIC_ASYNC_STAGE_PREPARE"] = "0"
        env["GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD"] = "0"
        env["GGML_ELASTIC_RELOAD_ON_XFER"] = "0"
        env["GGML_ELASTIC_XFER_EXTRA"] = "0"
        if strict_cache:
            env["GGML_ELASTIC_CACHE_FOREGROUND_LOAD"] = "0"
            env["GGML_ELASTIC_CL_RETAIN"] = "0"
            env["GGML_ELASTIC_CL_RETAIN_MB"] = "0"
            env["GGML_ELASTIC_NO_AUTO_EVICT"] = "0"
            env["GGML_ELASTIC_RELEASE_STAGE_AFTER_XFORM"] = "1"
    def enable_sync_plan_stages(env: dict[str, str], *, strict_cache: bool = False) -> None:
        # Execute plan anchors synchronously: no async LOAD/PREPARE overlap, but
        # still fire the load/prepare anchors so disk placements pay real IO and
        # transform cost.  This is the strict fixed-budget baseline path.
        env["LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE"] = "1"
        env["LLAMA_ELASTIC_DEFER_STAGE"] = "1"
        env["LLAMA_ELASTIC_INTERVAL_STAGE_KINDS"] = "load,prepare"
        env["GGML_ELASTIC_ASYNC_STAGE_LOAD"] = "0"
        env["GGML_ELASTIC_ASYNC_STAGE_PREPARE"] = "0"
        env["GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD"] = "0"
        env["GGML_ELASTIC_RELOAD_ON_XFER"] = "0"
        env["GGML_ELASTIC_XFER_EXTRA"] = "0"
        env["LLAMA_ELASTIC_ANCHOR_REPEAT_PER_GRAPH"] = "1"
        env["LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH"] = "1"
        env["LLAMA_ELASTIC_ENABLE_CPU_XFORM_STAGE"] = "1"
        env["LLAMA_ELASTIC_INTERVAL_INCLUDE_TRANSITIONS"] = "0"
        if strict_cache:
            # Strict correctness path: dynamic budget changes can evict/reload
            # Q8 SOA weights at different graph steps.  Retaining parent cl_mem
            # across those dynamic steps has produced token drift on device, so
            # keep planned baselines on fresh allocations until the pool path is
            # separately proven bit-stable.
            env["GGML_ELASTIC_CACHE_FOREGROUND_LOAD"] = "0"
            env["GGML_ELASTIC_CL_RETAIN"] = "0"
            env["GGML_ELASTIC_CL_RETAIN_MB"] = "0"
            env["GGML_ELASTIC_NO_AUTO_EVICT"] = "0"
            env["GGML_ELASTIC_RELEASE_STAGE_AFTER_XFORM"] = "1"
    use_interval_schedule = (
        args.use_interval_schedule == "1"
        or (args.use_interval_schedule == "auto" and args.cp_objective == "interval_makespan")
    )
    if use_interval_schedule:
        common["LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE"] = "1"
        if args.overlap_model == "none":
            common["LLAMA_ELASTIC_INTERVAL_STAGE_KINDS"] = "none"
            common["GGML_ELASTIC_ASYNC_STAGE_LOAD"] = "0"
            common["GGML_ELASTIC_ASYNC_STAGE_PREPARE"] = "0"
            common["GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD"] = "0"
            common["GGML_ELASTIC_RELOAD_ON_XFER"] = "0"
            common["GGML_ELASTIC_XFER_EXTRA"] = "0"
        else:
            enable_plan_pipeline(common)
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
    if method in {
        "offline", "offline-mixed", "online", "diff-tree-mixed", "mru",
        "candidate-select", "diff-graph-expand", "diff-tree-ideal",
    }:
        common["GGML_ELASTIC_DYNAMIC"] = "1"
    if method in {"offline", "offline-mixed"}:
        plan_dir = (
            args.phone_mixed_plan_dir
            if method == "offline-mixed"
            else args.phone_plan_dir
        )
        common["LLAMA_ELASTIC_DIR"] = plan_dir
        # Offline tables do not account for the separately allocated dynamic
        # expert cache. Let the backend enforce the combined budget when that
        # cache is active; otherwise the offline baseline silently exceeds the
        # same budget respected by MRU and online methods.
        common["GGML_ELASTIC_NO_AUTO_EVICT"] = "0" if adaptive_moe_cache else "1"
        # Build the prompt KV under the same routing/residency state used by
        # decode. Applying the first table plan only after token 0 leaves a KV
        # prefix computed with a different backend plan and causes output drift.
        common["LLAMA_ELASTIC_APPLY"] = (
            f"{plan_dir}/plan_{initial_budget_mib}MiB.json"
        )
    elif method in {"static-min", "static-max"}:
        static_budget = trace.min_bucket_mib if method == "static-min" else trace.max_bucket_mib
        common.update(
            {
                "LLAMA_ELASTIC_APPLY": f"{args.phone_plan_dir}/plan_{static_budget}MiB.json",
                "LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE": "0",
                "LLAMA_ELASTIC_DEFER_STAGE": "0",
                "GGML_ELASTIC_ASYNC_STAGE_LOAD": "0",
                "GGML_ELASTIC_ASYNC_STAGE_PREPARE": "0",
                "GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD": "0",
                "GGML_ELASTIC_RELOAD_ON_XFER": "0",
                "GGML_ELASTIC_XFER_EXTRA": "0",
            }
        )
    elif method in {"online", "diff-tree-mixed"}:
        granularity_online_policy = (
            "diff-tree" if method == "diff-tree-mixed"
            else ("online" if args.granularity_policy != "none"
                  else "none"))
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
                "LLAMA_ELASTIC_DIFF_HORIZON_TOKENS": str(args.diff_horizon_tokens),
                "LLAMA_ELASTIC_PLAN_UP_STEP_BUCKETS": str(args.plan_up_step_buckets),
                # Full Online is the unconstrained dynamic-planning bound:
                # apply every completed bucket transition. Diff-tree keeps
                # the configured gap as part of its practical transition
                # control.
                "LLAMA_ELASTIC_PLAN_SWITCH_MIN_GAP_STEPS": str(
                    0 if method == "online"
                    else args.plan_switch_min_gap_steps),
                # Online and Diff-tree already apply a measured transition
                # cost plus hysteresis in their frontier objective. A second
                # generic two-hit debounce delayed an async plan by another
                # decode and caused it to miss short but stable budget bands.
                "LLAMA_ELASTIC_PLAN_SWITCH_STABLE_STEPS": "1",
                "LLAMA_ELASTIC_ONLINE_INITIAL_BUDGET_MIB": str(initial_budget_mib),
                "LLAMA_ELASTIC_ONLINE_PREAPPLY_INITIAL": "1",
                "LLAMA_ELASTIC_DISK_RELOAD_MULTIPLIER": str(args.disk_reload_multiplier),
                "LLAMA_ELASTIC_DISK_GPU_RELOAD_MULTIPLIER": str(args.disk_gpu_reload_multiplier),
                "LLAMA_ELASTIC_OVERLAP_MODEL": str(args.overlap_model),
                "LLAMA_ELASTIC_CP_OBJECTIVE": str(args.cp_objective),
                "LLAMA_ELASTIC_ALLOWED_PLACEMENTS": str(args.allowed_placements),
                "LLAMA_ELASTIC_ALLOW_CPU_FALLBACK": "1" if args.allow_cpu_fallback else "0",
                "GGML_ELASTIC_CALLBACK_ASYNC": "0" if args.overlap_model == "none" else "1",
                # Publish only an exact-budget result, but keep a rising
                # budget solve off the decode critical path: the previous
                # smaller plan is memory-safe for one graph while the solver
                # overlaps with compute. Budget decreases remain blocking in
                # CallbackProvider because the prior plan may exceed capacity.
                "GGML_ELASTIC_CALLBACK_ASYNC_WAIT_MS":
                    "0" if args.overlap_model == "none" else "100",
                "LLAMA_ELASTIC_GRANULARITY_POLICY":
                    granularity_online_policy,
            }
        )
    elif method in {"candidate-select", "diff-graph-expand", "diff-tree-ideal"}:
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
                "LLAMA_ELASTIC_DIFF_HORIZON_TOKENS": str(args.diff_horizon_tokens),
                "LLAMA_ELASTIC_PLAN_UP_STEP_BUCKETS": str(args.plan_up_step_buckets),
                "LLAMA_ELASTIC_PLAN_SWITCH_MIN_GAP_STEPS": str(args.plan_switch_min_gap_steps),
                "LLAMA_ELASTIC_ONLINE_INITIAL_BUDGET_MIB": str(initial_budget_mib),
                "LLAMA_ELASTIC_ONLINE_PREAPPLY_INITIAL": "1",
                "LLAMA_ELASTIC_DISK_RELOAD_MULTIPLIER": str(args.disk_reload_multiplier),
                "LLAMA_ELASTIC_DISK_GPU_RELOAD_MULTIPLIER": str(args.disk_gpu_reload_multiplier),
                "LLAMA_ELASTIC_OVERLAP_MODEL": str(args.overlap_model),
                "LLAMA_ELASTIC_CP_OBJECTIVE": str(args.cp_objective),
                "LLAMA_ELASTIC_ALLOWED_PLACEMENTS": str(args.allowed_placements),
                "LLAMA_ELASTIC_CANDIDATE_MIN_GAIN_MS": str(args.candidate_min_gain_ms),
                "LLAMA_ELASTIC_ALLOW_CPU_FALLBACK": "1" if args.allow_cpu_fallback else "0",
                # Candidate expansion can take hundreds of milliseconds when
                # the budget changes. Keep it off the decode critical path,
                # just like remote online solving; the provider continues to
                # serve the last ready conservative plan while it computes.
                "GGML_ELASTIC_CALLBACK_ASYNC": "0" if args.overlap_model == "none" else "1",
            }
        )
    elif method == "mru":
        common.update(
            {
                "LLAMA_ELASTIC_ONLINE": "1",
                "LLAMA_ELASTIC_ONLINE_MODE": "mru-cache",
                # Use the OpenCL WBM's native MRU policy. It records a weight's
                # last-use event and releases victims only after the consuming
                # op completes. The context-level immediate path can evict an
                # input while the backend is still preparing the same op.
                "LLAMA_ELASTIC_RUNTIME_MRU_CACHE": "0",
                "GGML_ELASTIC_EVICT_POLICY": "mru",
                # Apply the baseline policy to packed dynamic expert slices as
                # well as ordinary WBM weights. Otherwise MoE MRU silently uses
                # LRU for the majority of model bytes.
                "GGML_ELASTIC_MOE_EXPERT_CACHE_POLICY": "mru",
                "GGML_ELASTIC_NO_AUTO_EVICT": "0",
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
                "LLAMA_ELASTIC_ONLINE_INITIAL_BUDGET_MIB": str(initial_budget_mib),
                "LLAMA_ELASTIC_ONLINE_PREAPPLY_INITIAL": "1",
                "LLAMA_ELASTIC_DISK_RELOAD_MULTIPLIER": str(args.disk_reload_multiplier),
                "LLAMA_ELASTIC_DISK_GPU_RELOAD_MULTIPLIER": str(args.disk_gpu_reload_multiplier),
                "LLAMA_ELASTIC_OVERLAP_MODEL": str(args.overlap_model),
                "LLAMA_ELASTIC_CP_OBJECTIVE": str(args.cp_objective),
                "LLAMA_ELASTIC_ALLOWED_PLACEMENTS": str(args.mru_allowed_placements),
                "LLAMA_ELASTIC_ALLOW_CPU_FALLBACK": "1" if args.allow_cpu_fallback else "0",
            }
        )
    else:
        raise ValueError(f"unknown method: {method}")
    if args.granularity_policy != "none":
        # Every granularity comparison uses the real three-stage unit
        # pipeline, including static and MRU baselines. The same backend
        # LOAD/PREPARE/COMPUTE path is authoritative for every method.
        common["GGML_ELASTIC_UNIT_PIPELINE"] = "1"
        common["GGML_ELASTIC_ASYNC_STAGE_LOAD"] = "1"
        common["GGML_ELASTIC_ASYNC_STAGE_PREPARE"] = "1"
        common["GGML_ELASTIC_ASYNC_PREPARE_MAX_PENDING"] = "256"
        common["GGML_ELASTIC_HOST_STAGING_POOL_MB"] = "128"
        # The placement timeline is expressed per complete weight, whereas
        # this experiment's real streaming path is expressed per physical
        # Cut/Tensor/Multi unit. Letting both paths stage DISK weights makes a
        # plan switch eagerly LOAD/PREPARE complete tensors and then stage the
        # same storage again in the unit pipeline. It also turns Cut into a
        # tensor-sized prefetch and charges frequently switching Online/Diff
        # methods extra work. Placement reconciliation remains enabled for
        # one-time persistent promotions; only recurring whole-weight stage
        # anchors are disabled.
        configure_unit_pipeline_stage_authority(common)
        if args.execution_backend == "cpu":
            common["GGML_ELASTIC_SYNC_STAGE_LOAD"] = "1"
            common["LLAMA_ELASTIC_ROUTE_CPU_ELASTIC"] = "1"
        else:
            common["GGML_ELASTIC_CL_RETAIN"] = "1"
            common["GGML_ELASTIC_CL_RETAIN_MB"] = "128"
            common["GGML_ELASTIC_ASYNC_LOAD_BOUNDED"] = "1"
            common["GGML_ELASTIC_ASYNC_LOAD_MAX_PENDING_MB"] = "128"
            common["GGML_ELASTIC_ASYNC_LOAD_MAX_STAGED_MB"] = "128"
            common["GGML_ELASTIC_ASYNC_XFER"] = "1"
            common["GGML_ELASTIC_RELOAD_ON_XFER"] = "1"
            common["GGML_ELASTIC_RELEASE_STAGE_AFTER_XFORM"] = "1"
        if method != "mru":
            # Preserve the plan's resident identity. A disk weight is staged
            # for its consumer and retired after each graph; retaining it after
            # token 0 silently turns every plan into the backend MRU cache.
            common["GGML_ELASTIC_NO_AUTO_EVICT"] = "1"
            common["LLAMA_ELASTIC_FORCE_PLAN_EVICT"] = "1"
            common["LLAMA_ELASTIC_ANCHOR_REPEAT_PER_GRAPH"] = "1"
            common["LLAMA_ELASTIC_EVICT_DISK_WEIGHTS_PER_GRAPH"] = "1"
            common["LLAMA_ELASTIC_PRELOAD_DISK_WEIGHTS_PER_GRAPH"] = "0"
            common["GGML_ELASTIC_CACHE_FOREGROUND_LOAD"] = "1"
    apply_plan_residency_policy(
        common,
        method,
        str(getattr(args, "plan_residency_policy", "strict")),
    )
    for item in args.extra_env:
        if "=" not in item:
            raise ValueError(f"--extra-env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--extra-env has empty key: {item}")
        common[key] = value
    return common


def start_remote_server(
    args: argparse.Namespace,
    log_path: Path,
    method: str,
) -> subprocess.Popen[str] | None:
    if not ({"online", "diff-tree-mixed"} &
            {m.strip() for m in args.methods.split(",") if m.strip()}):
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
        "--planner-stream-reserve-mib",
        str(args.planner_stream_reserve_mib),
        "--time-limit-ms",
        str(args.time_limit_ms),
        "--prefetch-distance",
        str(args.prefetch_distance),
        "--transition-weight",
        str(args.transition_weight),
        "--transition-horizon-tokens",
        str(args.diff_horizon_tokens),
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
        "--dynamic-active-experts",
        str(args.dynamic_active_experts),
        "--dynamic-total-experts",
        str(args.dynamic_total_experts),
        "--plan-switch-min-gain-ms",
        str(args.online_plan_switch_min_gain_ms),
        "--granularity-policy",
        remote_server_granularity_policy(
            method, args.granularity_policy),
        "--granularity-backend",
        str(args.granularity_backend),
        "--granularity-horizon-tokens",
        str(args.granularity_horizon_tokens),
        "--granularity-min-gain-ms",
        str(args.granularity_min_gain_ms),
        "--granularity-max-edits",
        str(args.granularity_max_edits),
        "--granularity-beam-width",
        str(args.granularity_beam_width),
        "--granularity-placement-source",
        str(args.granularity_placement_source),
    ]
    for forced in args.force_weight_placement:
        cmd.extend(["--force-weight-placement", forced])
    if args.granularity_profile:
        cmd.extend([
            "--granularity-profile",
            str(args.granularity_profile),
        ])
    if {"online", "diff-tree-mixed"} & {
        method.strip() for method in args.methods.split(",")
    }:
        mixed_dir = args.artifact_root / "offline_mixed_table"
        if mixed_dir.exists():
            cmd.extend([
                "--granularity-offline-dir",
                str(mixed_dir),
            ])
    if args.granularity_backend == "cpu":
        cmd.extend(["--allow-output-cpu", "--allow-output-disk"])
    if args.allow_cpu_fallback:
        cmd.append("--allow-cpu-fallback")
    if args.online_ignore_state:
        cmd.append("--ignore-state")
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT)
    log.close()

    health_url = f"http://127.0.0.1:{args.port}/health"
    deadline = time.monotonic() + 10.0
    last_error = "server did not become healthy"
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            detail = log_path.read_text(errors="replace").strip()
            raise RuntimeError(
                f"remote solver exited before health check rc={rc}: {detail or 'no log output'}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and payload.get("ok") is True:
                break
            last_error = f"unexpected health response: status={response.status} payload={payload}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(0.1)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
        raise RuntimeError(f"remote solver health check failed: {last_error}")

    adb(args.adb_serial, ["reverse", f"tcp:{args.port}", f"tcp:{args.port}"], check=False, timeout=args.adb_timeout_s)
    return proc


def methods_need_offline_table(methods: str) -> bool:
    selected = {m.strip() for m in methods.split(",") if m.strip()}
    return bool(selected & {"offline", "static-min", "static-max", "candidate-select", "diff-graph-expand", "diff-tree-ideal"})


def methods_need_mixed_offline_table(methods: str) -> bool:
    selected = {m.strip() for m in methods.split(",") if m.strip()}
    # Online and Diff-tree use the offline mixed plan at each budget as their
    # common target. Building that table must not depend on also measuring the
    # offline-mixed oracle in the same invocation.
    return bool(selected & {"offline-mixed", "online", "diff-tree-mixed"})


def required_plan_budgets(
    methods: list[str],
    traces: list[TraceWindow],
    table_budgets: list[int],
    *,
    mixed: bool = False,
) -> list[int]:
    selected = set(methods)
    required: set[int] = set()
    if mixed:
        if selected & {"offline-mixed", "online", "diff-tree-mixed"}:
            required.update(table_budgets)
        return sorted(required)
    if selected & {"offline", "candidate-select", "diff-graph-expand", "diff-tree-ideal"}:
        required.update(table_budgets)
    if "static-min" in selected:
        required.update(trace.min_bucket_mib for trace in traces)
    if "static-max" in selected:
        required.update(trace.max_bucket_mib for trace in traces)
    return sorted(required)


def validate_remote_plan_table(
    args: argparse.Namespace,
    methods: list[str],
    traces: list[TraceWindow],
    table_budgets: list[int],
    *,
    local_dir: Path,
    mixed: bool = False,
) -> str:
    required = required_plan_budgets(
        methods, traces, table_budgets, mixed=mixed)
    if not required:
        return ""
    phone_dir = (
        args.phone_mixed_plan_dir if mixed else args.phone_plan_dir)
    checks = " ".join(
        shell_quote(f"{phone_dir}/plan_{budget}MiB.json")
        for budget in required
    )
    proc = adb_shell_retry(
        args.adb_serial,
        f"for p in {checks}; do [ -f \"$p\" ] || basename \"$p\"; done",
        timeout=args.adb_timeout_s,
        retries=args.adb_retries,
    )
    missing = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if missing:
        raise SystemExit(
            "remote offline plan table is incomplete; missing "
            + ", ".join(missing)
            + ". Rebuild and push the table before running these baselines."
        )

    index_proc = adb_shell_retry(
        args.adb_serial,
        f"cat {shell_quote(phone_dir + '/index.json')}",
        timeout=args.adb_timeout_s,
        retries=args.adb_retries,
    )
    try:
        index = json.loads(index_proc.stdout or "")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"remote offline plan index is invalid JSON: {exc}") from exc

    expected = {item.strip() for item in args.allowed_placements.split(",") if item.strip()}
    primary_specs: set[tuple[str, ...]] = set()
    for row in index.get("index", []):
        candidates = row.get("candidates", []) if isinstance(row, dict) else []
        primary = next(
            (candidate for candidate in candidates if int(candidate.get("candidate_id", -1)) == 0),
            candidates[0] if candidates else {},
        )
        spec = str(primary.get("allowed_placements", ""))
        if spec:
            primary_specs.add(tuple(sorted(item.strip() for item in spec.split(",") if item.strip())))
    expected_spec = tuple(sorted(expected))
    if primary_specs and primary_specs != {expected_spec}:
        actual = "; ".join(",".join(spec) for spec in sorted(primary_specs))
        raise SystemExit(
            "remote offline plan table placement space does not match this run: "
            f"table={actual} requested={','.join(expected_spec)}. "
            "Rebuild and push the table, or use matching --allowed-placements."
        )
    return validate_remote_directory_contents(
        args.adb_serial,
        local_dir,
        phone_dir,
        label=("mixed offline plan table"
               if mixed else "offline plan table"),
        timeout_s=max(args.adb_timeout_s, 60.0),
        retries=args.adb_retries,
    )


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
    ap.add_argument(
        "--expected-model-sha256", default="",
        help=(
            "fail unless the remote GGUF has this SHA-256; strongly "
            "recommended with --skip-push-model"))
    ap.add_argument("--llama-cli", type=Path, default=ROOT / "build-android-llama/bin/llama-cli")
    ap.add_argument(
        "--expected-binary-sha256", default="",
        help=(
            "fail unless the remote llama-cli has this SHA-256; strongly "
            "recommended with --skip-push-binary"))
    ap.add_argument("--libomp-path", type=Path, default=None)
    ap.add_argument("--libcxx-path", type=Path, default=None)
    ap.add_argument("--model-meta", type=Path, default=ROOT / "runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json")
    ap.add_argument("--cost-dir", type=Path, default=ROOT / "runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01")
    ap.add_argument("--trace-glob", default="trace/traces_9g/trace_*.csv")
    ap.add_argument(
        "--expected-source-trace-sha256",
        default="",
        help=(
            "fail unless every selected replay-input trace has this "
            "SHA-256"))
    ap.add_argument(
        "--original-source-trace",
        type=Path,
        default=None,
        help=(
            "optional historical raw trace from which the normalized "
            "--trace-glob input was derived; semantic equality is required"))
    ap.add_argument(
        "--expected-original-source-trace-sha256",
        default="",
        help=(
            "fail unless --original-source-trace has this SHA-256"))
    ap.add_argument("--trace-filter", default="", help="substring filter for trace filenames")
    ap.add_argument("--methods", default="offline,online,mru,static-min,static-max")
    ap.add_argument(
        "--execution-backend", choices=("cpu", "gpu"), default="gpu",
        help="run CPU_Elastic-only or OpenCL Elastic-only; never heterogeneous")
    ap.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    ap.add_argument("--window-sec", type=float, default=600.0)
    ap.add_argument("--window-stride-sec", type=float, default=10.0)
    ap.add_argument("--window-start-sec", type=float, default=None,
                    help="if set, use this source-trace start time instead of selecting the lowest-memory window")
    ap.add_argument("--replay-speedup", type=float, default=1.0)
    ap.add_argument("--bench-seconds", type=float, default=None, help="wall-clock decode duration per run; default = window-sec / replay-speedup")
    ap.add_argument("--bucket-mib", type=int, default=256)
    ap.add_argument("--extra-max-budget-mib", type=int, default=8192)
    ap.add_argument("--kv-mib", type=int, default=512)
    ap.add_argument("--misc-mib", type=int, default=256)
    ap.add_argument("--safety-mib", type=int, default=64)
    ap.add_argument("--pinned-extra-mib", type=int, default=410,
                    help="extra pinned non-planned model bytes counted inside the budget, e.g. output.weight for Llama-3 8B Q4_0")
    ap.add_argument(
        "--expected-pin-token-embd-mib", type=float, default=0.0)
    ap.add_argument(
        "--expected-pin-output-mib", type=float, default=0.0)
    ap.add_argument(
        "--expected-wbm-total-mib", type=float, default=0.0,
        help=(
            "fail unless the runtime registers this exact physical weight "
            "total in its WBM; 0 disables the explicit total check"))
    ap.add_argument(
        "--planner-stream-reserve-mib", type=int, default=0,
        help=(
            "reserve physical weight capacity for current/prefetched units; "
            "subtracted from persistent plan residency only"))
    ap.add_argument("--release-stage-after-xform", choices=("0", "1"), default="1",
                    help="release host staging after prepare/xform consumption; 0 retains staged disk weights across graphs")
    ap.add_argument("--time-limit-ms", type=int, default=250)
    ap.add_argument("--prefetch-distance", type=int, default=1)
    ap.add_argument("--transition-weight", type=float, default=0.1)
    ap.add_argument("--diff-horizon-tokens", type=float, default=8.0,
                    help="token horizon used by diff-tree / diff-graph accept decisions")
    ap.add_argument("--plan-up-step-buckets", type=int, default=1,
                    help="limit upward budget movement to this many buckets per accepted switch; 0 disables")
    ap.add_argument("--plan-switch-min-gap-steps", type=int, default=16,
                    help="minimum decode steps between dynamic plan switches; 0 disables")
    ap.add_argument("--online-plan-switch-min-gain-ms", type=float, default=0.1,
                    help="remote online CP keeps a feasible previous plan unless predicted gain reaches this value")
    ap.add_argument("--disk-reload-multiplier", type=float, default=1.0)
    ap.add_argument("--disk-gpu-reload-multiplier", type=float, default=4.0)
    ap.add_argument("--overlap-model", choices=("pipeline", "none"), default="pipeline")
    ap.add_argument("--static-overlap-model", choices=("pipeline", "none"), default="pipeline",
                    help="execution overlap for static-min/static-max fixed offline plans")
    ap.add_argument("--cp-objective", choices=("resource_makespan", "interval_makespan", "sum"), default="resource_makespan")
    ap.add_argument("--allowed-placements", default="cpu,gpu,disk_cpu,disk_gpu",
                    help="comma-separated solver placement choices")
    ap.add_argument(
        "--force-weight-placement", action="append", default=[],
        metavar="WEIGHT=PLACEMENT",
        help="repeatable hard placement constraint shared by every method")
    ap.add_argument("--allow-cpu-fallback", action="store_true",
                    help="allow the solver to estimate CPU_Elastic compute when measured CPU compute rows are missing")
    ap.add_argument(
        "--granularity-policy",
        choices=(
            "none", "fixed-multi", "fixed-tensor", "fixed-cut",
            "offline", "diff-tree"),
        default="none",
        help="enable common tiled representation and mixed working-unit planning")
    ap.add_argument(
        "--granularity-backend", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument(
        "--granularity-profile", type=Path,
        help="OP13 CPU/GPU working-unit calibration JSON")
    ap.add_argument(
        "--granularity-placement-source",
        choices=("stateful-cp", "offline-table"),
        default="stateful-cp",
        help=(
            "whether Online/Diff re-solve placement or reuse the common "
            "offline placement while adapting working-unit granularity"))
    ap.add_argument("--granularity-horizon-tokens", type=float, default=8.0)
    ap.add_argument("--granularity-min-gain-ms", type=float, default=0.0)
    ap.add_argument("--granularity-max-edits", type=int, default=4)
    ap.add_argument("--granularity-beam-width", type=int, default=128)
    ap.add_argument("--granularity-pipeline-lookahead", type=int, default=1)
    ap.add_argument(
        "--granularity-pipeline-lookahead-mib", type=int, default=64)
    ap.add_argument(
        "--plan-residency-policy",
        choices=("strict", "reuse-stream", "cache-managed"),
        default="strict",
        help=(
            "strict retires every plan-DISK weight per graph; reuse-stream "
            "keeps a budget-safe streaming working set; cache-managed keeps "
            "the mixed frontier but delegates movable residency to WBM"))
    ap.add_argument("--online-ignore-state", action="store_true",
                    help="remote online CP ignores runtime residency state and solves budget-only plans")
    ap.add_argument("--top-k", type=int, default=1,
                    help="number of candidate plans per budget for candidate-select / diff-graph-expand / diff-tree-ideal")
    ap.add_argument("--candidate-placement-specs", default="",
                    help="semicolon-separated allowed-placement specs for offline candidate diversity")
    ap.add_argument("--candidate-min-distance", type=int, default=32,
                    help="minimum weight-placement Hamming distance between top-k candidates")
    ap.add_argument("--candidate-min-gain-ms", type=float, default=5.0,
                    help="candidate-select only accepts a non-base candidate if predicted score gain is at least this many ms/token")
    ap.add_argument("--mru-allowed-placements", default="gpu,disk_gpu",
                    help="placement choices for the runtime MRU baseline; default keeps MRU on GPU/disk rather than CPU")
    ap.add_argument("--static-weight-mirror", choices=("auto", "0", "1"), default="auto",
                    help="cache scheduler static weight copies for mixed CPU/GPU plans; auto enables it when CPU placement is allowed")
    ap.add_argument("--moe-expert-cache-adaptive", choices=("auto", "0", "1"), default="auto",
                    help="scale retained Q4 MoE expert slots with the budget; auto enables it when model metadata contains Q4 expert tensors")
    ap.add_argument("--moe-expert-cache-base-slots", type=int, default=0,
                    help="retained expert slots at the trace floor; 0 derives them from residual floor-budget bytes")
    ap.add_argument("--moe-retain-workload-high-water", choices=("auto", "0", "1"), default="auto",
                    help="retain each budget tier's observed expert-cache high-water; auto enables it for workload-aware planners only when mean active demand exceeds the floor cache")
    ap.add_argument("--dynamic-active-experts", type=float, default=0.0,
                    help="profiled active experts per layer used to budget dynamic expert weights; 0 disables scaling")
    ap.add_argument("--dynamic-total-experts", type=float, default=0.0,
                    help="total experts per layer paired with --dynamic-active-experts")
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
    ap.add_argument("--cpu-mask", default="")
    ap.add_argument("--cpu-strict", type=int, default=1)
    ap.add_argument("--poll", type=int, default=50)
    ap.add_argument(
        "--min-cpu-freq-limit-khz", type=int, default=0,
        help="invalidate a CPU run if thermal policy lowers scaling_max_freq")
    ap.add_argument(
        "--min-cpu-mean-freq-khz", type=int, default=0,
        help="invalidate a CPU run if measured mean active frequency is lower")
    ap.add_argument(
        "--min-cpu-median-sample-min-khz", type=int, default=0,
        help=(
            "invalidate a CPU run if the median slowest monitored CPU "
            "frequency is lower"))
    ap.add_argument(
        "--max-pipeline-budget-violations", type=int, default=-1,
        help=(
            "invalidate a run when the pipeline budget audit exceeds this "
            "count; -1 disables the check"))
    ap.add_argument(
        "--max-plan-protection-relaxations", type=int, default=-1,
        help=(
            "invalidate a run when streaming admission overrides more than "
            "this many protected plan residents; -1 disables the check"))
    ap.add_argument(
        "--fixed-performance-mode", action=argparse.BooleanOptionalAction,
        default=False)
    ap.add_argument("--n-gpu-layers", type=int, default=99,
                    help="llama-cli -ngl value used for benchmark runs")
    ap.add_argument("--prompt", default="Summarize dynamic elastic memory planning for mobile LLM inference.")
    ap.add_argument("--show-token-output", action="store_true",
                    help="keep llama-cli generated tokens in logs; default suppresses stdout and keeps stderr/perf logs")
    ap.add_argument(
        "--token-prefix-count", type=int, default=16,
        help="equal-length token prefix compared across timed methods")
    ap.add_argument("--timeout-s", type=int, default=480)
    ap.add_argument("--adb-timeout-s", type=int, default=30)
    ap.add_argument("--adb-retries", type=int, default=2)
    ap.add_argument(
        "--require-device-idle", action="store_true",
        help="wait instead of overlapping another inference/benchmark process")
    ap.add_argument("--device-idle-poll-s", type=float, default=5.0)
    ap.add_argument("--device-idle-timeout-s", type=float, default=3600.0)
    ap.add_argument("--cooldown-thermal-max-c", type=float, default=0.0,
                    help="if >0, wait before each run until max CPU/GPU/skin temperature is below this value")
    ap.add_argument("--cooldown-thermal-status-max", type=int, default=0,
                    help="when cooldown is enabled, also wait until Android thermal status is at most this value")
    ap.add_argument("--cooldown-poll-s", type=float, default=30.0)
    ap.add_argument("--cooldown-timeout-s", type=float, default=1800.0)
    ap.add_argument(
        "--min-battery-level-pct", type=int, default=0,
        help="wait until the physical device battery reaches this percentage")
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
    if not 0 <= args.min_battery_level_pct <= 100:
        ap.error("--min-battery-level-pct must be between 0 and 100")
    for label, value in (
        ("--expected-model-sha256", args.expected_model_sha256),
        ("--expected-binary-sha256", args.expected_binary_sha256),
        (
            "--expected-source-trace-sha256",
            args.expected_source_trace_sha256,
        ),
        (
            "--expected-original-source-trace-sha256",
            args.expected_original_source_trace_sha256,
        ),
    ):
        if value and not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            ap.error(f"{label} must contain exactly 64 hexadecimal digits")
    args.expected_model_sha256 = args.expected_model_sha256.lower()
    args.expected_binary_sha256 = args.expected_binary_sha256.lower()
    args.expected_source_trace_sha256 = (
        args.expected_source_trace_sha256.lower())
    args.expected_original_source_trace_sha256 = (
        args.expected_original_source_trace_sha256.lower())
    if bool(args.original_source_trace) != bool(
            args.expected_original_source_trace_sha256):
        ap.error(
            "--original-source-trace and "
            "--expected-original-source-trace-sha256 must be supplied "
            "together")
    if (
        args.granularity_policy != "none" and
        args.execution_backend != args.granularity_backend
    ):
        ap.error(
            "--execution-backend and --granularity-backend must match "
            "for a fixed-backend granularity experiment")

    if (args.dynamic_active_experts > 0) != (args.dynamic_total_experts > 0):
        raise SystemExit("--dynamic-active-experts and --dynamic-total-experts must be set together")
    if args.dynamic_active_experts > args.dynamic_total_experts:
        raise SystemExit("--dynamic-active-experts cannot exceed --dynamic-total-experts")
    if args.token_prefix_count <= 0:
        raise SystemExit("--token-prefix-count must be positive")
    if args.expected_wbm_total_mib < 0.0:
        raise SystemExit("--expected-wbm-total-mib must be non-negative")

    # Share the same per-device host lock as the fixed-budget runner. Without
    # it, two runners can both pass an empty clean-start gate before either
    # launches llama-cli, wasting a full trace before overlap is detected.
    safe_serial = re.sub(r"[^A-Za-z0-9_.-]", "_", args.adb_serial)
    device_lock_path = (
        Path("/tmp") / f"llama-elastic-sweep-{safe_serial}.lock")
    device_lock = device_lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            device_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        device_lock.close()
        raise RuntimeError(
            f"another Elastic sweep already owns device "
            f"{args.adb_serial}; lock={device_lock_path}") from exc
    device_lock.seek(0)
    device_lock.truncate()
    device_lock.write(
        f"pid={os.getpid()} artifact={args.artifact_root} "
        f"started={time.time()}\n")
    device_lock.flush()

    args.artifact_root = args.artifact_root.resolve()
    trace_dir = args.artifact_root / "traces"
    log_dir = args.artifact_root / "logs"
    tokens_dir = args.artifact_root / "tokens"
    summary_dir = args.artifact_root / "summary"
    table_dir = args.artifact_root / "offline_table"
    mixed_table_dir = args.artifact_root / "offline_mixed_table"
    args.phone_plan_dir = f"{args.remote_dir}/plans_matrix10min"
    args.phone_mixed_plan_dir = (
        f"{args.remote_dir}/plans_matrix10min_mixed")
    args.phone_model_meta = f"{args.remote_dir}/model_meta_matrix10min.json"
    args.phone_cost_dir = f"{args.remote_dir}/cost_matrix10min"
    if args.libomp_path is None:
        args.libomp_path = default_libomp_path()
    if args.libcxx_path is None:
        args.libcxx_path = default_libcxx_path()
    if args.bench_seconds is None:
        args.bench_seconds = args.window_sec / args.replay_speedup
    args.effective_safety_mib = args.safety_mib + args.pinned_extra_mib
    args.moe_expert_tensor_count = count_q4_moe_expert_tensors(args.model_meta)

    source_name_re = re.compile(
        r"^(?:trace_\d+_user_\d+(?:_[^.]+)?|dynamic_trace)\.csv$")
    sources = sorted((ROOT / p).resolve() for p in Path(ROOT).glob(args.trace_glob))
    sources = [p for p in sources if source_name_re.match(p.name)]
    if args.trace_filter:
        sources = [p for p in sources if args.trace_filter in p.name]
    if not sources:
        raise SystemExit(f"no traces matched {args.trace_glob!r}")
    if args.expected_source_trace_sha256:
        actual_source_hashes = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        }
        mismatches = {
            path: actual
            for path, actual in actual_source_hashes.items()
            if actual
                != args.expected_source_trace_sha256
        }
        if mismatches:
            raise SystemExit(
                "normalized replay-input trace SHA-256 mismatch: "
                f"expected={args.expected_source_trace_sha256} "
                f"actual={mismatches}")
    original_source_trace = (
        args.original_source_trace.resolve()
        if args.original_source_trace is not None else None)
    args.original_source_trace = original_source_trace
    original_source_trace_sha256 = ""
    if original_source_trace is not None:
        if len(sources) != 1:
            raise SystemExit(
                "--original-source-trace requires exactly one selected "
                f"replay input, found {len(sources)}")
        if not original_source_trace.is_file():
            raise SystemExit(
                f"historical original trace not found: "
                f"{original_source_trace}")
        original_source_trace_sha256 = hashlib.sha256(
            original_source_trace.read_bytes()).hexdigest()
        if (
            original_source_trace_sha256
            != args.expected_original_source_trace_sha256
        ):
            raise SystemExit(
                "historical original source trace SHA-256 mismatch: "
                f"expected={args.expected_original_source_trace_sha256} "
                f"actual={original_source_trace_sha256}")
        original_points = read_trace(original_source_trace)
        normalized_points = read_trace(sources[0])
        if original_points != normalized_points:
            raise SystemExit(
                "normalized replay input is not point-for-point equal to "
                f"historical original trace: original="
                f"{original_source_trace} normalized={sources[0]}")
        print(
            "validated historical trace provenance: "
            f"original_sha256={original_source_trace_sha256} "
            f"normalized_sha256="
            f"{hashlib.sha256(sources[0].read_bytes()).hexdigest()} "
            f"points={len(original_points)}",
            flush=True,
        )

    traces = [
        write_window_trace(
            p,
            trace_dir,
            window_sec=args.window_sec,
            stride_sec=args.window_stride_sec,
            replay_speedup=args.replay_speedup,
            bucket_mib=args.bucket_mib,
            window_start_sec=args.window_start_sec,
        )
        for p in sources
    ]
    static_traces = {
        (t.local.name, "static-min"): write_static_trace(t, trace_dir, "min")
        for t in traces
    }
    static_traces.update(
        {
            (t.local.name, "static-max"): write_static_trace(t, trace_dir, "max")
            for t in traces
        }
    )
    budgets = build_budget_list(traces, args.bucket_mib, args.extra_max_budget_mib)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.dynamic_run_config_sha256 = dynamic_run_config_sha256(args)
    args.dynamic_run_core_config_sha256 = (
        dynamic_run_core_config_sha256(args))
    need_offline_table = methods_need_offline_table(args.methods)
    need_mixed_offline_table = methods_need_mixed_offline_table(
        args.methods)
    if (
        {"offline-mixed", "online", "diff-tree-mixed"} & set(methods)
        and args.granularity_policy == "none"
    ):
        ap.error(
            "mixed-granularity methods require --granularity-policy "
            "diff-tree (or offline)")

    def build_table(
        out_dir: Path,
        granularity_policy: str,
        *,
        mixed: bool,
    ) -> None:
        cmd = [
            sys.executable,
            str(PLAN_DIR / "build_offline_budget_table.py"),
            "--model-meta",
            str(args.model_meta),
            "--cost-dir",
            str(args.cost_dir),
            "--out-dir",
            str(out_dir),
            "--budgets",
            ",".join(str(b) for b in budgets),
            "--kv-mib",
            str(args.kv_mib),
            "--misc-mib",
            str(args.misc_mib),
            "--safety-mib",
            str(args.effective_safety_mib),
            "--planner-stream-reserve-mib",
            str(args.planner_stream_reserve_mib),
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
            "--dynamic-active-experts",
            str(args.dynamic_active_experts),
            "--dynamic-total-experts",
            str(args.dynamic_total_experts),
            "--granularity-policy",
            granularity_policy,
            "--granularity-backend",
            str(args.granularity_backend),
            "--granularity-beam-width",
            str(args.granularity_beam_width),
            "--top-k",
            str(args.top_k),
        ]
        for forced in args.force_weight_placement:
            cmd.extend(["--force-weight-placement", forced])
        if args.granularity_profile:
            cmd.extend([
                "--granularity-profile",
                str(args.granularity_profile),
            ])
        if args.candidate_placement_specs and not mixed:
            cmd.extend(["--candidate-placement-specs", str(args.candidate_placement_specs)])
        cmd.extend(["--candidate-min-distance", str(args.candidate_min_distance)])
        if args.allow_cpu_fallback:
            cmd.append("--allow-cpu-fallback")
        if args.execution_backend == "cpu":
            cmd.extend(["--allow-output-cpu", "--allow-output-disk"])
        if args.offline_chain_state:
            cmd.append("--chain-state")
        if not args.dry_run:
            run(cmd)
        else:
            print("+ " + " ".join(shlex.quote(c) for c in cmd))

    if not args.skip_build_table and need_offline_table:
        # Traditional placement baselines all use one complete tensor as the
        # logical working unit. This isolates the proposed granularity policy
        # from the existing Static/Online/Offline/MRU placement policies.
        baseline_policy = (
            "fixed-tensor"
            if args.granularity_policy in {"offline", "diff-tree"}
            else args.granularity_policy
        )
        build_table(table_dir, baseline_policy, mixed=False)
    elif not need_offline_table:
        print("skip offline table build: selected methods do not require it", flush=True)
    if not args.skip_build_table and need_mixed_offline_table:
        build_table(mixed_table_dir, "offline", mixed=True)
    elif not need_mixed_offline_table:
        print(
            "skip mixed offline table build: no mixed-granularity method selected",
            flush=True,
        )

    args.moe_expert_cache_slot_details = {}
    if (
        args.moe_expert_cache_base_slots == 0
        and args.moe_expert_tensor_count > 0
        and args.dynamic_active_experts > 0
        and (need_offline_table or need_mixed_offline_table)
    ):
        floor_budget_mib = min(t.min_bucket_mib for t in traces)
        cache_plan_dir = (
            table_dir if need_offline_table else mixed_table_dir)
        slots, details = derive_q4_moe_cache_base_slots(
            args.model_meta,
            cache_plan_dir / f"plan_{floor_budget_mib}MiB.json",
            floor_budget_mib=floor_budget_mib,
            reserved_mib=args.kv_mib + args.misc_mib + args.effective_safety_mib,
            total_experts=args.dynamic_total_experts,
        )
        if slots > 0:
            args.moe_expert_cache_base_slots = slots
            args.moe_expert_cache_slot_details = details
            print(json.dumps({"moe_expert_cache_base_slots": slots, **details}, sort_keys=True), flush=True)

    model_meta_sha256 = hashlib.sha256(
        args.model_meta.read_bytes()).hexdigest()
    cost_dir_sha256 = directory_sha256(args.cost_dir)
    if not cost_dir_sha256:
        raise SystemExit(
            f"cost directory is missing or empty: {args.cost_dir}")
    remote_model_sha256 = ""
    remote_binary_sha256 = ""
    remote_model_meta_sha256 = ""
    remote_cost_dir_sha256 = ""
    remote_trace_sha256: dict[str, str] = {}
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
        if need_mixed_offline_table:
            adb_shell_retry(
                args.adb_serial,
                f"rm -rf {shell_quote(args.phone_mixed_plan_dir)} && "
                f"mkdir -p {shell_quote(args.phone_mixed_plan_dir)}",
                timeout=args.adb_timeout_s,
                retries=args.adb_retries,
            )
            adb(
                args.adb_serial,
                [
                    "push", str(mixed_table_dir) + "/.",
                    args.phone_mixed_plan_dir + "/",
                ],
            )
        for t in traces:
            adb(args.adb_serial, ["push", str(t.local), f"{args.remote_dir}/{t.remote_name}"])
            for method in ("static-min", "static-max"):
                st = static_traces[(t.local.name, method)]
                adb(args.adb_serial, ["push", str(st.local), f"{args.remote_dir}/{st.remote_name}"])

    if not args.dry_run:
        remote_model_meta_sha256 = validate_remote_sha256(
            args.adb_serial,
            args.phone_model_meta,
            model_meta_sha256,
            label="model metadata",
            timeout_s=args.adb_timeout_s,
            retries=args.adb_retries,
        )
        remote_cost_dir_sha256 = validate_remote_directory_contents(
            args.adb_serial,
            args.cost_dir,
            args.phone_cost_dir,
            label="placement cost directory",
            timeout_s=max(args.adb_timeout_s, 60.0),
            retries=args.adb_retries,
        )
        if args.expected_binary_sha256:
            remote_binary_sha256 = validate_remote_sha256(
                args.adb_serial,
                f"{args.remote_dir}/llama-cli",
                args.expected_binary_sha256,
                label="llama-cli",
                timeout_s=args.adb_timeout_s,
                retries=args.adb_retries,
            )
        if args.expected_model_sha256:
            # The GGUF is several GiB, so its content hash receives the
            # benchmark timeout rather than the short metadata timeout.
            remote_model_sha256 = validate_remote_sha256(
                args.adb_serial,
                (
                    f"{args.remote_dir}/{args.model}"
                    if not args.model.startswith("/") else args.model
                ),
                args.expected_model_sha256,
                label="model",
                timeout_s=args.timeout_s,
                retries=args.adb_retries,
            )
        trace_inputs: dict[str, Path] = {}
        for trace in traces:
            trace_inputs[
                f"{args.remote_dir}/{trace.remote_name}"
            ] = trace.local
            for method in ("static-min", "static-max"):
                static_trace = static_traces[(trace.local.name, method)]
                trace_inputs[
                    f"{args.remote_dir}/{static_trace.remote_name}"
                ] = static_trace.local
        for remote_path, local_path in trace_inputs.items():
            remote_trace_sha256[remote_path] = validate_remote_sha256(
                args.adb_serial,
                remote_path,
                hashlib.sha256(local_path.read_bytes()).hexdigest(),
                label="budget trace",
                timeout_s=args.adb_timeout_s,
                retries=args.adb_retries,
            )

    remote_offline_table_sha256 = ""
    remote_mixed_offline_table_sha256 = ""
    if not args.dry_run:
        remote_offline_table_sha256 = validate_remote_plan_table(
            args,
            methods,
            traces,
            budgets,
            local_dir=table_dir,
            mixed=False,
        )
        remote_mixed_offline_table_sha256 = validate_remote_plan_table(
            args,
            methods,
            traces,
            budgets,
            local_dir=mixed_table_dir,
            mixed=True,
        )

    granularity_profile_path = args.granularity_profile
    granularity_profile_sha256 = (
        hashlib.sha256(
            granularity_profile_path.read_bytes()).hexdigest()
        if granularity_profile_path is not None
        and granularity_profile_path.is_file()
        else ""
    )
    configured_granularity_profile_kind = granularity_profile_kind(
        granularity_profile_path, args.granularity_backend)
    configured_multi_implementation = (
        granularity_profile_multi_implementation(
            granularity_profile_path, args.granularity_backend))
    configured_cut_compute_grouping = (
        granularity_profile_cut_compute_grouping(
            granularity_profile_path, args.granularity_backend))
    (
        configured_transition_fixed_ms,
        configured_transition_cost_source,
    ) = granularity_profile_transition_contract(
        granularity_profile_path, args.granularity_backend)
    offline_table_sha256 = directory_sha256(table_dir)
    mixed_offline_table_sha256 = directory_sha256(mixed_table_dir)
    source_trace_sha256_by_name = {
        trace.local.name: hashlib.sha256(
            trace.local.read_bytes()).hexdigest()
        for trace in traces
    }
    source_file_sha256_by_name = {
        trace.local.name: hashlib.sha256(
            trace.source.read_bytes()).hexdigest()
        for trace in traces
    }
    runtime_trace_sha256_by_key: dict[tuple[str, str], str] = {}
    for trace in traces:
        dynamic_sha256 = source_trace_sha256_by_name[trace.local.name]
        for method in methods:
            if method in {"static-min", "static-max"}:
                runtime_local = static_traces[(trace.local.name, method)].local
                runtime_trace_sha256_by_key[(trace.local.name, method)] = (
                    hashlib.sha256(runtime_local.read_bytes()).hexdigest()
                )
            else:
                runtime_trace_sha256_by_key[(trace.local.name, method)] = (
                    dynamic_sha256
                )
    rows: list[dict[str, Any]] = [] if args.no_resume else read_existing_csv(summary_dir / "results.csv")
    completed = {
        (r.get("trace", ""), r.get("method", ""))
        for r in rows
        if r.get("status") == "ok" and str(r.get("rc", "0")) == "0"
        and (
            not any(
                pin_policy_requests(
                    item.split("=", 1)[1], tensor_suffix)
                for item in args.extra_env
                if item.startswith("GGML_ELASTIC_PIN=")
                for tensor_suffix in ("token_embd", "output")
            )
            or str(r.get("pin_contract_valid", "")).lower() == "true"
        )
        and (
            args.max_pipeline_budget_violations < 0
            or (
                str(r.get("pipeline_budget_violations", "")).strip() != ""
                and int(float(r["pipeline_budget_violations"]))
                <= args.max_pipeline_budget_violations
            )
        )
        and (
            args.max_plan_protection_relaxations < 0
            or (
                str(r.get(
                    "pipeline_plan_protection_relaxations", "")
                ).strip() != ""
                and int(float(
                    r["pipeline_plan_protection_relaxations"]))
                <= args.max_plan_protection_relaxations
            )
        )
        and str(
            r.get("fused_lane_contract_valid", "")
        ).lower() == "true"
        and str(
            r.get("frontier_contract_valid", "")
        ).lower() == "true"
        and str(
            r.get(
                "unit_pipeline_stage_authority_contract_valid", "")
        ).lower() == "true"
        and str(
            r.get("backend_compute_contract_valid", "")
        ).lower() == "true"
        and str(
            r.get("remote_solver_contract_valid", "")
        ).lower() == "true"
        and str(
            r.get("timed_run_contract_valid", "")
        ).lower() == "true"
        and str(r.get("source_window_sha256", "")).strip() != ""
        and str(r.get("runtime_trace_sha256", "")).strip() != ""
        and str(r.get("granularity_profile_sha256", ""))
            == granularity_profile_sha256
        and str(r.get("granularity_profile_kind", ""))
            == configured_granularity_profile_kind
        and str(r.get("multi_implementation_configured", ""))
            == configured_multi_implementation
        and str(r.get("cut_compute_grouping_configured", ""))
            == configured_cut_compute_grouping
        and float(
            r.get("transition_fixed_ms_configured") or -1.0
        ) == configured_transition_fixed_ms
        and str(r.get("transition_cost_source_configured", ""))
            == configured_transition_cost_source
        and float(
            r.get("planner_stream_reserve_mib_configured") or -1.0
        ) == float(args.planner_stream_reserve_mib)
        and str(r.get("granularity_placement_source", ""))
            == effective_granularity_placement_source(
                str(r.get("method", "")),
                str(args.granularity_placement_source),
            )
        and str(r.get("granularity_policy_effective", ""))
            == effective_granularity_policy(
                str(r.get("method", "")), args.granularity_policy)
        and str(r.get("mixed_offline_table_sha256", ""))
            == mixed_offline_table_sha256
        and str(r.get("offline_table_sha256", ""))
            == offline_table_sha256
        and str(r.get("remote_offline_table_sha256", ""))
            == remote_offline_table_sha256
        and str(r.get("remote_mixed_offline_table_sha256", ""))
            == remote_mixed_offline_table_sha256
        and str(r.get("expected_model_sha256", ""))
            == args.expected_model_sha256
        and str(r.get("remote_model_sha256", ""))
            == remote_model_sha256
        and str(r.get("expected_binary_sha256", ""))
            == args.expected_binary_sha256
        and str(r.get("remote_binary_sha256", ""))
            == remote_binary_sha256
        and str(r.get("model_meta_sha256", ""))
            == model_meta_sha256
        and str(r.get("remote_model_meta_sha256", ""))
            == remote_model_meta_sha256
        and str(r.get("cost_dir_sha256", ""))
            == cost_dir_sha256
        and str(r.get("remote_cost_dir_sha256", ""))
            == remote_cost_dir_sha256
        and str(r.get("dynamic_run_config_sha256", ""))
            == args.dynamic_run_config_sha256
        and str(r.get("dynamic_run_core_config_sha256", ""))
            == args.dynamic_run_core_config_sha256
        and str(r.get("source_window_sha256", ""))
            == source_trace_sha256_by_name.get(
                str(r.get("trace", "")), "")
        and str(r.get("source_file_sha256", ""))
            == source_file_sha256_by_name.get(
                str(r.get("trace", "")), "")
        and str(r.get("normalized_source_input_sha256", ""))
            == source_file_sha256_by_name.get(
                str(r.get("trace", "")), "")
        and str(r.get(
            "expected_normalized_source_input_sha256", ""))
            == args.expected_source_trace_sha256
        and str(r.get("expected_source_trace_sha256", ""))
            == args.expected_source_trace_sha256
        and (
            original_source_trace is None
            or (
                str(r.get("original_source_trace_sha256", ""))
                    == original_source_trace_sha256
                and str(r.get(
                    "expected_original_source_trace_sha256", ""))
                    == args.expected_original_source_trace_sha256
                and str(r.get(
                    "original_source_trace_semantically_equal", "")
                ).lower() == "true"
            )
        )
        and str(r.get("runtime_trace_sha256", ""))
            == runtime_trace_sha256_by_key.get(
                (str(r.get("trace", "")), str(r.get("method", ""))),
                "",
            )
        and str(r.get("remote_runtime_trace_sha256", ""))
            == str(r.get("runtime_trace_sha256", ""))
        and not resume_environment_contract_failures(args, r)
        and str(r.get("valid_token_sequence", "true")).lower() != "false"
        and str(r.get("valid_token_trace", "true")).lower() != "false"
        and str(r.get(
            "token_trace_timing_contract_valid", "")).lower() == "true"
        and str(r.get(
            "decode_wall_contract_valid", "")).lower() == "true"
        and str(r.get(
            "decode_phase_contract_valid", "")).lower() == "true"
        and (
            not args.fixed_performance_mode
            or str(r.get(
                "fixed_performance_restore_valid", "")
            ).lower() == "true"
        )
        and (
            not args.require_device_idle
            or (
                str(r.get(
                    "valid_measurement_isolation", "")).lower() == "true"
                and int(float(
                    r.get("measurement_monitor_samples") or 0)) > 0
                and int(float(
                    r.get("measurement_monitor_failures") or 0)) == 0
                and str(r.get(
                    "monitor_thread_incomplete", "false")
                ).lower() != "true"
                and str(r.get("measurement_llama_pid") or "").strip() != ""
            )
        )
        and str(r.get("token_trace_pull_rc", "0")) in {"", "0"}
    }
    server = None
    try:
        for t in traces:
            for method in methods:
                timed_mode = args.bench_seconds and args.bench_seconds > 0
                log_tag = f"s{int(args.bench_seconds)}" if timed_mode and float(args.bench_seconds).is_integer() else (f"s{args.bench_seconds:g}" if timed_mode else f"n{args.n_pred}")
                work_key = f"{method}:{t.local.stem}:{log_tag}"
                work_hash = hashlib.sha1(work_key.encode("utf-8")).hexdigest()[:10]
                work = f"{args.remote_dir}/om_{method}_{work_hash}_{log_tag}"
                if (t.local.name, method) in completed:
                    print(f"=== skip existing trace={t.local.name} method={method} ===", flush=True)
                    continue
                if not args.dry_run:
                    adb_shell_retry(
                        args.adb_serial,
                        f"rm -rf {shell_quote(work)} && mkdir -p {shell_quote(work)}",
                        timeout=args.adb_timeout_s,
                        retries=args.adb_retries,
                    )
                remote_trace = f"{args.remote_dir}/{t.remote_name}"
                runtime_trace_local = t.local
                runtime_trace_kind = "dynamic"
                runtime_trace_budget_mib: int | str = ""
                if method in {"static-min", "static-max"}:
                    static_trace = static_traces[(t.local.name, method)]
                    remote_trace = (
                        f"{args.remote_dir}/{static_trace.remote_name}")
                    runtime_trace_local = static_trace.local
                    runtime_trace_kind = method
                    runtime_trace_budget_mib = static_trace.budget_mib
                env = make_method_env(args, method, t, remote_trace, work)
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
                    ("0" if args.execution_backend == "cpu"
                     else str(args.n_gpu_layers)),
                    "-fa",
                    "on",
                    "-no-cnv",
                ]
                if args.execution_backend == "cpu":
                    argv.extend(["-dev", "CPU_Elastic"])
                    if args.cpu_mask:
                        argv.extend([
                            "--cpu-mask", args.cpu_mask,
                            "--cpu-strict", str(args.cpu_strict),
                            "--poll", str(args.poll),
                        ])
                argv.extend(args.llama_extra_arg)
                if timed_mode:
                    env["LLAMA_ELASTIC_BENCH_SECONDS"] = f"{args.bench_seconds:.3f}".rstrip("0").rstrip(".")
                if timed_mode:
                    argv[argv.index("-n") + 1] = "-1"
                script = remote_shell_env(args.remote_dir, env, argv, suppress_stdout=False)
                log_path = log_dir / f"{method}_{t.local.stem}_{log_tag}.log"
                remote_solver_log = ""
                remote_solver_fresh_process = False
                solver_log_path: Path | None = None
                if (
                    not args.dry_run
                    and method in {"online", "diff-tree-mixed"}
                ):
                    if server is not None:
                        raise RuntimeError(
                            "remote solver from a previous method is still "
                            "running")
                    solver_log_path = (
                        log_dir /
                        f"remote_solver_{method}_{t.local.stem}_"
                        f"{log_tag}.log"
                    )
                    remote_solver_log = str(solver_log_path)
                print(f"=== run trace={t.local.name} method={method} log={log_path} ===", flush=True)
                if args.dry_run:
                    print(script)
                    rc = 0
                    thermal_before = {"thermal_status": "", "thermal_cpu_max_c": "", "thermal_gpu_max_c": "", "thermal_skin_max_c": ""}
                    thermal_after = {"thermal_status": "", "thermal_cpu_max_c": "", "thermal_gpu_max_c": "", "thermal_skin_max_c": ""}
                else:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    thermal_before = wait_for_clean_start(
                        args, t.local.name, method)
                    fixed_performance_audit: dict[str, Any] = {}
                    if solver_log_path is not None:
                        # Start after a potentially long battery/thermal
                        # gate, so "fresh process" describes the solver that
                        # immediately serves this measured run rather than a
                        # process left idle throughout cooldown.
                        server = start_remote_server(
                            args, solver_log_path, method)
                        if server is None:
                            raise RuntimeError(
                                f"failed to start fresh solver for {method}")
                        remote_solver_fresh_process = True
                    with fixed_performance_measurement(
                            args) as fixed_performance_audit:
                        with log_path.open("w") as f:
                            f.write(
                                "[thermal-before] "
                                + json.dumps(
                                    thermal_before, sort_keys=True)
                                + "\n")
                            f.flush()
                            monitor_stop = threading.Event()
                            monitor_result: dict[str, Any] = {}
                            monitor = threading.Thread(
                                target=monitor_measurement_overlap,
                                args=(
                                    args, monitor_stop, monitor_result),
                                daemon=True,
                            )
                            monitor.start()
                            try:
                                proc = adb_shell(
                                    args.adb_serial,
                                    script,
                                    check=False,
                                    timeout=args.timeout_s,
                                    stdout=f,
                                )
                                rc = proc.returncode
                            except subprocess.TimeoutExpired:
                                rc = 124
                                f.write(
                                    f"\nTIMEOUT after {args.timeout_s}s\n")
                            finally:
                                monitor_stop.set()
                                monitor.join(
                                    timeout=3 * max(
                                        1.0,
                                        min(5.0, args.adb_timeout_s),
                                    ) + 2.0)
                                if monitor.is_alive():
                                    monitor_result[
                                        "monitor_thread_incomplete"] = True
                                    monitor_result[
                                        "valid_measurement_isolation"] = False
                            thermal_after = read_thermal_snapshot(
                                args.adb_serial, args.adb_timeout_s)
                            f.write(
                                "[thermal-after] "
                                + json.dumps(
                                    thermal_after, sort_keys=True)
                                + "\n")
                            f.write(
                                "[measurement-isolation] "
                                + json.dumps(
                                    monitor_result, sort_keys=True)
                                + "\n")
                    with log_path.open("a") as f:
                        f.write(
                            "[fixed-performance-audit] "
                            + json.dumps(
                                fixed_performance_audit, sort_keys=True)
                            + "\n")
                    print(f"=== done trace={t.local.name} method={method} rc={rc} ===", flush=True)
                    if server is not None:
                        stop_remote_server(args, server)
                        server = None
                parsed = parse_log(log_path, method) if not args.dry_run else {"status": "dry-run"}
                if not args.dry_run:
                    parsed.update(monitor_result)
                    parsed.update(fixed_performance_audit)
                    if (
                        args.fixed_performance_mode
                        and not bool(parsed.get(
                            "fixed_performance_restore_valid", False))
                        and parsed.get("status") == "ok"
                    ):
                        parsed["status"] = (
                            "fixed_performance_restore_invalid")
                    solver_failures, solver_audit = (
                        remote_solver_log_contract(
                            method,
                            (
                                Path(remote_solver_log)
                                if remote_solver_log else None
                            ),
                            work,
                            args.granularity_max_edits,
                            args.granularity_placement_source,
                        )
                    )
                    if (
                        method in {"online", "diff-tree-mixed"}
                        and not remote_solver_fresh_process
                    ):
                        solver_failures.append(
                            "remote solver process was not fresh")
                    parsed.update(solver_audit)
                    parsed["remote_solver_contract_valid"] = (
                        not solver_failures)
                    parsed[
                        "remote_solver_contract_invalid_reason"
                    ] = "; ".join(solver_failures)
                    if solver_failures and parsed.get("status") == "ok":
                        parsed["status"] = "remote_solver_invalid"
                    timed_failures = timed_run_contract_failures(
                        parsed, args.bench_seconds)
                    parsed["timed_run_contract_valid"] = (
                        not timed_failures)
                    parsed["timed_run_contract_invalid_reason"] = "; ".join(
                        timed_failures)
                    if timed_failures and parsed.get("status") == "ok":
                        parsed["status"] = "timed_run_incomplete"
                    decode_wall_failures = (
                        decode_wall_contract_failures(parsed)
                    )
                    parsed["decode_wall_contract_valid"] = (
                        not decode_wall_failures)
                    parsed[
                        "decode_wall_contract_invalid_reason"
                    ] = "; ".join(decode_wall_failures)
                    if (
                        decode_wall_failures
                        and parsed.get("status") == "ok"
                    ):
                        parsed["status"] = "decode_wall_invalid"
                    decode_phase_failures = (
                        decode_phase_contract_failures(
                            parsed, args.execution_backend)
                    )
                    parsed["decode_phase_contract_valid"] = (
                        not decode_phase_failures)
                    parsed[
                        "decode_phase_contract_invalid_reason"
                    ] = "; ".join(decode_phase_failures)
                    if (
                        decode_phase_failures
                        and parsed.get("status") == "ok"
                    ):
                        parsed["status"] = "decode_phase_invalid"
                    pin_policy = env.get("GGML_ELASTIC_PIN", "")
                    pin_failures: list[str] = []
                    parsed["pin_policy"] = pin_policy
                    for tensor_suffix in ("token_embd", "output"):
                        requested = pin_policy_requests(
                            pin_policy, tensor_suffix)
                        parsed[f"pin_{tensor_suffix}_requested"] = requested
                        observed = int(parsed.get(
                            f"pin_{tensor_suffix}_inside_budget_count", 0))
                        if requested and observed <= 0:
                            pin_failures.append(
                                f"requested {tensor_suffix} pin not observed "
                                "inside weight budget")
                        expected_mib = float(getattr(
                            args,
                            f"expected_pin_{tensor_suffix}_mib",
                            0.0,
                        ))
                        observed_mib = float(parsed.get(
                            f"pin_{tensor_suffix}_inside_budget_mib", 0.0))
                        tolerance_mib = max(0.1, expected_mib * 0.001)
                        if (
                            requested and observed > 0
                            and expected_mib > 0.0
                            and abs(observed_mib - expected_mib)
                                > tolerance_mib
                        ):
                            pin_failures.append(
                                f"{tensor_suffix} pinned "
                                f"{observed_mib:.2f} MiB, expected "
                                f"{expected_mib:.2f} MiB")
                    outside_count = int(
                        parsed.get("pin_outside_budget_count", 0))
                    if outside_count != 0:
                        pin_failures.append(
                            f"{outside_count} tensor pin(s) enlarged weight "
                            "budget")
                    parsed["pin_contract_valid"] = not pin_failures
                    parsed["pin_contract_invalid_reason"] = "; ".join(
                        pin_failures)
                    if pin_failures and parsed.get("status") == "ok":
                        parsed["status"] = "pin_contract_invalid"
                    fused_lane_failures = []
                    if (
                        env.get(
                            "GGML_ELASTIC_CPU_STAGE_COPY_PARALLEL") == "1"
                        and int(parsed.get(
                            "fused_layout_pair_calls") or 0) > 0
                        and int(parsed.get(
                            "fused_layout_parallel_calls") or 0) <= 0
                    ):
                        fused_lane_failures.append(
                            "fused layout pairs did not use the persistent "
                            "second PREPARE lane")
                    parsed["fused_lane_contract_valid"] = (
                        not fused_lane_failures)
                    parsed["fused_lane_contract_invalid_reason"] = "; ".join(
                        fused_lane_failures)
                    if (
                        fused_lane_failures
                        and parsed.get("status") == "ok"
                    ):
                        parsed["status"] = "fused_lane_invalid"
                    frontier_failures = frontier_contract_failures(
                        method, parsed)
                    parsed["frontier_contract_valid"] = (
                        not frontier_failures)
                    parsed["frontier_contract_invalid_reason"] = "; ".join(
                        frontier_failures)
                    if (
                        frontier_failures
                        and parsed.get("status") == "ok"
                    ):
                        parsed["status"] = "frontier_invalid"
                    stage_authority_failures = (
                        unit_pipeline_stage_authority_failures(
                            parsed)
                        if args.granularity_policy != "none"
                        else []
                    )
                    parsed[
                        "unit_pipeline_stage_authority_contract_valid"
                    ] = not stage_authority_failures
                    parsed[
                        "unit_pipeline_stage_authority_contract_invalid_reason"
                    ] = "; ".join(stage_authority_failures)
                    if (
                        stage_authority_failures
                        and parsed.get("status") == "ok"
                    ):
                        parsed["status"] = (
                            "unit_pipeline_stage_authority_invalid")
                    backend_compute_failures = (
                        backend_compute_contract_failures(
                            args.execution_backend,
                            parsed,
                            args.expected_wbm_total_mib,
                            (
                                configured_multi_implementation
                                == "multi_fused"
                            ),
                        )
                    )
                    parsed["backend_compute_contract_valid"] = (
                        not backend_compute_failures)
                    parsed[
                        "backend_compute_contract_invalid_reason"
                    ] = "; ".join(backend_compute_failures)
                    if (
                        backend_compute_failures
                        and parsed.get("status") == "ok"
                    ):
                        parsed["status"] = "backend_compute_invalid"
                    configured_env = {
                        item.split("=", 1)[0].strip(): item.split("=", 1)[1]
                        for item in args.extra_env if "=" in item
                    }
                    for worker, env_key in (
                            ("load", "GGML_ELASTIC_PIPELINE_LOAD_CPU"),
                            ("prepare", "GGML_ELASTIC_PIPELINE_PREPARE_CPU"),
                            ("copy", "GGML_ELASTIC_PIPELINE_COPY_CPU")):
                        if env_key not in configured_env:
                            continue
                        expected_cpu = int(configured_env[env_key])
                        observed_cpu = parsed.get(
                            f"pipeline_{worker}_affinity_cpu", "")
                        affinity_rc = parsed.get(
                            f"pipeline_{worker}_affinity_rc", "")
                        if (
                            observed_cpu == ""
                            and affinity_rc == ""
                            and not pipeline_worker_was_required(
                                parsed, worker)
                        ):
                            continue
                        if (observed_cpu == "" or affinity_rc == "" or
                                int(observed_cpu) != expected_cpu or
                                int(affinity_rc) != 0):
                            parsed["status"] = "affinity_invalid"
                    if (
                        parsed.get("status") == "ok"
                        and not parsed.get(
                            "valid_measurement_isolation", False)
                    ):
                        parsed["status"] = "external_overlap"
                    if (
                        args.execution_backend == "cpu"
                        and parsed.get("status") == "ok"
                    ):
                        frequency_failures = (
                            cpu_frequency_contract_failures(
                                parsed,
                                min_limit_khz=(
                                    args.min_cpu_freq_limit_khz),
                                min_mean_khz=(
                                    args.min_cpu_mean_freq_khz),
                                min_median_sample_min_khz=(
                                    args.min_cpu_median_sample_min_khz),
                            )
                        )
                        if frequency_failures:
                            parsed["status"] = "frequency_invalid"
                            parsed["frequency_invalid_reason"] = "; ".join(
                                frequency_failures)
                    if (
                        parsed.get("status") == "ok"
                        and args.max_pipeline_budget_violations >= 0
                    ):
                        violations = parsed.get(
                            "pipeline_budget_violations", "")
                        if violations in ("", None):
                            parsed["status"] = "budget_audit_missing"
                        elif (
                            int(violations)
                            > args.max_pipeline_budget_violations
                        ):
                            parsed["status"] = "budget_violation"
                    if (
                        parsed.get("status") == "ok"
                        and args.max_plan_protection_relaxations >= 0
                    ):
                        relaxations = parsed.get(
                            "pipeline_plan_protection_relaxations", "")
                        if relaxations in ("", None):
                            parsed["status"] = (
                                "plan_residency_audit_missing")
                        elif (
                            int(relaxations)
                            > args.max_plan_protection_relaxations
                        ):
                            parsed["status"] = (
                                "plan_residency_invalid")
                if not args.dry_run:
                    tokens_dir.mkdir(parents=True, exist_ok=True)
                    local_tokens = (
                        tokens_dir /
                        f"{method}_{t.local.stem}_{log_tag}.csv"
                    )
                    local_tokens.unlink(missing_ok=True)
                    pull = adb(
                        args.adb_serial,
                        ["pull", f"{work}/tokens.csv", str(local_tokens)],
                        check=False,
                        timeout=args.adb_timeout_s,
                    )
                    token_count, token_hash = token_sequence(local_tokens)
                    prefix_count, prefix_hash = token_sequence(
                        local_tokens, args.token_prefix_count)
                    token_timing = token_trace_timing(local_tokens)
                    token_timing_failures = (
                        token_trace_timing_contract_failures(
                            token_timing, args.bench_seconds)
                    )
                    parsed.update({
                        "token_trace": str(local_tokens),
                        "token_trace_pull_rc": pull.returncode,
                        "token_id_count": token_count,
                        "token_ids_sha256": token_hash,
                        "token_prefix_count": prefix_count,
                        "token_prefix_sha256": prefix_hash,
                        "valid_token_trace": (
                            pull.returncode == 0
                            and prefix_count == args.token_prefix_count
                            and prefix_hash is not None
                        ),
                        **token_timing,
                        "token_trace_timing_contract_valid":
                            not token_timing_failures,
                        "token_trace_timing_contract_invalid_reason":
                            "; ".join(token_timing_failures),
                    })
                    if (
                        parsed["status"] == "ok"
                        and not parsed["valid_token_trace"]
                    ):
                        parsed["status"] = "token_incomplete"
                    if (
                        parsed["status"] == "ok"
                        and token_timing_failures
                    ):
                        parsed["status"] = "token_timing_invalid"
                parsed.update({f"{k}_before": v for k, v in thermal_before.items()})
                parsed.update({f"{k}_after": v for k, v in thermal_after.items()})
                parsed.update(
                    {
                        "require_device_idle_configured":
                            args.require_device_idle,
                        "device_lock_path": str(device_lock_path),
                        "cooldown_thermal_max_c_configured":
                            args.cooldown_thermal_max_c,
                        "cooldown_thermal_status_max_configured":
                            args.cooldown_thermal_status_max,
                        "min_battery_level_pct_configured":
                            args.min_battery_level_pct,
                        "min_cpu_freq_limit_khz_configured":
                            args.min_cpu_freq_limit_khz,
                        "min_cpu_mean_freq_khz_configured":
                            args.min_cpu_mean_freq_khz,
                        "min_cpu_median_sample_min_khz_configured":
                            args.min_cpu_median_sample_min_khz,
                        "remote_solver_fresh_process":
                            remote_solver_fresh_process,
                        "remote_solver_log": remote_solver_log,
                        "granularity_profile": (
                            str(granularity_profile_path)
                            if granularity_profile_path is not None
                            else ""),
                        "granularity_profile_sha256":
                            granularity_profile_sha256,
                        "granularity_profile_kind":
                            configured_granularity_profile_kind,
                        "multi_implementation_configured":
                            configured_multi_implementation,
                        "cut_compute_grouping_configured":
                            configured_cut_compute_grouping,
                        "transition_fixed_ms_configured":
                            configured_transition_fixed_ms,
                        "transition_cost_source_configured":
                            configured_transition_cost_source,
                        "granularity_placement_source":
                            effective_granularity_placement_source(
                                method,
                                args.granularity_placement_source,
                            ),
                        "granularity_placement_source_configured":
                            args.granularity_placement_source,
                        "granularity_policy_configured":
                            args.granularity_policy,
                        "granularity_policy_effective":
                            effective_granularity_policy(
                                method, args.granularity_policy),
                        "mixed_offline_table_sha256":
                            mixed_offline_table_sha256,
                        "offline_table_sha256":
                            offline_table_sha256,
                        "remote_offline_table_sha256":
                            remote_offline_table_sha256,
                        "remote_mixed_offline_table_sha256":
                            remote_mixed_offline_table_sha256,
                        "expected_model_sha256":
                            args.expected_model_sha256,
                        "remote_model_sha256":
                            remote_model_sha256,
                        "expected_binary_sha256":
                            args.expected_binary_sha256,
                        "remote_binary_sha256":
                            remote_binary_sha256,
                        "model_meta_sha256":
                            model_meta_sha256,
                        "remote_model_meta_sha256":
                            remote_model_meta_sha256,
                        "cost_dir_sha256":
                            cost_dir_sha256,
                        "remote_cost_dir_sha256":
                            remote_cost_dir_sha256,
                        "dynamic_run_config_sha256":
                            args.dynamic_run_config_sha256,
                        "dynamic_run_core_config_sha256":
                            args.dynamic_run_core_config_sha256,
                        "planner_stream_reserve_mib_configured":
                            args.planner_stream_reserve_mib,
                        "expected_wbm_total_mib_configured":
                            args.expected_wbm_total_mib,
                        "trace": t.local.name,
                        "runtime_trace": runtime_trace_local.name,
                        "runtime_trace_kind": runtime_trace_kind,
                        "runtime_trace_budget_mib":
                            runtime_trace_budget_mib,
                        "runtime_trace_sha256": hashlib.sha256(
                            runtime_trace_local.read_bytes()).hexdigest(),
                        "remote_runtime_trace_sha256":
                            remote_trace_sha256.get(remote_trace, ""),
                        "source_window_sha256": hashlib.sha256(
                            t.local.read_bytes()).hexdigest(),
                        "source_file_sha256":
                            source_file_sha256_by_name[t.local.name],
                        "normalized_source_input_sha256":
                            source_file_sha256_by_name[t.local.name],
                        "expected_normalized_source_input_sha256":
                            args.expected_source_trace_sha256,
                        "expected_source_trace_sha256":
                            args.expected_source_trace_sha256,
                        "original_source_trace": (
                            str(original_source_trace)
                            if original_source_trace is not None else ""),
                        "original_source_trace_sha256":
                            original_source_trace_sha256,
                        "expected_original_source_trace_sha256":
                            args.expected_original_source_trace_sha256,
                        "original_source_trace_semantically_equal": (
                            original_source_trace is not None),
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
                        "max_bucket_mib": t.max_bucket_mib,
                        "log": str(log_path),
                    }
                )
                rows = [
                    row for row in rows
                    if (row.get("trace"), row.get("method")) !=
                    (t.local.name, method)
                ]
                rows.append(parsed)
                mark_token_correctness(
                    rows, set(methods), args.token_prefix_count)
                if (
                    parsed.get("status") == "ok"
                    and str(parsed.get("rc", "0")) == "0"
                    and str(
                        parsed.get("valid_token_sequence", "true")
                    ).lower() != "false"
                    and str(
                        parsed.get("valid_token_trace", "true")
                    ).lower() != "false"
                    and str(
                        parsed.get(
                            "token_trace_timing_contract_valid", "")
                    ).lower() == "true"
                    and str(
                        parsed.get(
                            "decode_phase_contract_valid", "")
                    ).lower() == "true"
                    and (
                        not args.require_device_idle
                        or str(
                            parsed.get(
                                "valid_measurement_isolation", "")
                        ).lower() == "true"
                    )
                ):
                    completed.add((t.local.name, method))
                write_csv(summary_dir / "results.csv", rows)
                write_markdown(summary_dir / "SUMMARY.md", rows, traces, args)
    finally:
        if not args.dry_run:
            stop_remote_server(args, server)
            if args.fixed_performance_mode:
                try:
                    # Last-resort cleanup for failures before/around the
                    # per-method measurement context.
                    set_fixed_performance_mode(args, False)
                except Exception as exc:
                    print(
                        "=== final fixed-performance cleanup failed: "
                        f"{type(exc).__name__}: {exc} ===",
                        flush=True,
                    )

    mark_token_correctness(rows, set(methods), args.token_prefix_count)
    write_csv(summary_dir / "results.csv", rows)
    write_markdown(summary_dir / "SUMMARY.md", rows, traces, args)
    trace_meta = [t.__dict__ | {"source": str(t.source), "local": str(t.local)} for t in traces]
    (summary_dir / "trace_windows.json").write_text(json.dumps(trace_meta, indent=2, sort_keys=True) + "\n")
    if not args.dry_run:
        completion_failures = matrix_completion_failures(
            rows,
            trace_names={trace.local.name for trace in traces},
            methods=set(methods),
            require_device_idle=args.require_device_idle,
            require_fixed_performance_restore=(
                args.fixed_performance_mode),
        )
        if completion_failures:
            raise RuntimeError(
                "dynamic matrix did not complete validly: "
                + " | ".join(completion_failures))
    print(json.dumps({"summary": str(summary_dir / "SUMMARY.md"), "results": str(summary_dir / "results.csv")}, indent=2))


if __name__ == "__main__":
    main()
