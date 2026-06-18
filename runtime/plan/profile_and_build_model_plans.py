#!/usr/bin/env python3
"""Profile a new model on Android, then build elastic planner artifacts.

This is the repeatable "new model" entry point:

1. run phone profiles for OpenCL compute / reload path / CPU_Elastic compute
2. pull profile CSVs to the host
3. build model_meta and measured cost model
4. build an offline budget table from those measured artifacts
5. optionally push the generated table back to the phone
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = Path(__file__).resolve().parent


def run(cmd: list[str], *, dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def adb(serial: str, args: list[str], *, dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    return run(cmd, dry_run=dry_run, check=check)


def adb_shell(serial: str, script: str, *, dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return adb(serial, ["shell", script], dry_run=dry_run, check=check)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def remote_env(env: dict[str, str], argv: list[str], remote_dir: str) -> str:
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items() if v is not None)
    return f"cd {shlex.quote(remote_dir)} && {exports} {shell_join(argv)}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return max(0, sum(1 for _ in f) - 1)


def profile_command(
    *,
    remote_dir: str,
    model: str,
    csv_name: str,
    prompt: str,
    n_predict: int,
    ctx_size: int,
    batch: int,
    threads: int,
    mode: str,
    remote_trace: str | None,
) -> str:
    env = {
        "LD_LIBRARY_PATH": remote_dir,
        "GGML_ELASTIC_PROFILE": "1",
        "GGML_ELASTIC_PROFILE_CSV": f"{remote_dir}/{csv_name}",
    }

    argv = [
        "./llama-cli",
        "-m",
        model,
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(ctx_size),
        "-b",
        str(batch),
        "-ub",
        str(batch),
        "-t",
        str(threads),
        "--temp",
        "0",
        "--no-warmup",
    ]

    if mode == "opencl-compute":
        env.update(
            {
                "GGML_OPENCL_DISABLE_ALLOC_HOST_PTR": "1",
                "GGML_OPENCL_USE_SVM": "1",
                "GGML_OPENCL_ELASTIC": "1",
            }
        )
        argv += ["-ngl", "99", "-fa", "on"]
    elif mode == "opencl-reload":
        env.update(
            {
                "GGML_OPENCL_DISABLE_ALLOC_HOST_PTR": "1",
                "GGML_OPENCL_USE_SVM": "1",
                "GGML_OPENCL_ELASTIC": "1",
                "GGML_ELASTIC_TIMING": "1",
                "GGML_ELASTIC_DYNAMIC": "1",
                "GGML_ELASTIC_BUDGET_BUCKET_MB": "256",
            }
        )
        if remote_trace:
            env["GGML_ELASTIC_BUDGET_CSV"] = remote_trace
        argv += ["-ngl", "99", "-fa", "on"]
    elif mode == "cpu-compute":
        env.update(
            {
                "GGML_ELASTIC_CHUNK_SIZE": "1",
                "GGML_ELASTIC_PROFILE": "1",
            }
        )
        argv += ["-ngl", "0", "-dev", "CPU_Elastic"]
    else:
        raise ValueError(f"unknown profile mode: {mode}")

    return remote_env(env, argv, remote_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile Android model and build elastic CP-SAT planner artifacts")
    ap.add_argument("--adb-serial", default="", help="adb serial, e.g. 172.20.115.151:5555")
    ap.add_argument("--remote-dir", default="/data/local/tmp/hyzheng/elastic")
    ap.add_argument("--model", required=True, help="remote GGUF path or filename under remote-dir")
    ap.add_argument("--model-tag", required=True, help="stable artifact tag, e.g. Meta-Llama-3-8B-Instruct-Q4_0")
    ap.add_argument("--model-meta", type=Path, help="existing complete weights_ops.json to use instead of deriving meta from trace-limited CSV")
    ap.add_argument("--trace", type=Path, help="local dynamic budget trace CSV to push/use during reload profiling")
    ap.add_argument("--out-root", type=Path, default=Path("runtime/plan/generated"))
    ap.add_argument("--budgets", required=True, help="comma-separated MiB budgets for offline table")
    ap.add_argument("--kv-mib", type=int, default=512)
    ap.add_argument("--misc-mib", type=int, default=256)
    ap.add_argument("--safety-mib", type=int, default=64)
    ap.add_argument("--time-limit-ms", type=int, default=200)
    ap.add_argument("--n-opencl", type=int, default=16)
    ap.add_argument("--n-reload", type=int, default=16)
    ap.add_argument("--n-cpu", type=int, default=4)
    ap.add_argument("--ctx-size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--allow-cpu-fallback", action="store_true", help="allow solver synthetic CPU cost if CPU profile is incomplete")
    ap.add_argument("--skip-phone-profile", action="store_true", help="build from existing local CSVs in out-root/model-tag/csv")
    ap.add_argument("--push-plan-dir", action="store_true", help="push generated offline table back to remote-dir/plans_<model-tag>")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_remote = args.model if args.model.startswith("/") else f"{args.remote_dir}/{args.model}"
    artifact_root = (ROOT / args.out_root / args.model_tag).resolve()
    csv_dir = artifact_root / "csv"
    generated_meta_path = ROOT / "runtime" / "plan" / "model_meta" / f"{args.model_tag}.weights_ops.json"
    meta_path = args.model_meta.resolve() if args.model_meta else generated_meta_path
    cost_dir = ROOT / "runtime" / "plan" / "profiles" / "android-opencl" / args.model_tag
    table_dir = artifact_root / "offline_table"
    manifest_path = artifact_root / "manifest.json"
    csv_dir.mkdir(parents=True, exist_ok=True)

    remote_trace = None
    commands: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    if not args.skip_phone_profile:
        adb(args.adb_serial, ["devices"], dry_run=args.dry_run)
        adb_shell(args.adb_serial, f"mkdir -p {shlex.quote(args.remote_dir)}", dry_run=args.dry_run)
        if args.trace:
            remote_trace = f"{args.remote_dir}/{args.trace.name}"
            adb(args.adb_serial, ["push", str(args.trace), remote_trace], dry_run=args.dry_run)

        profiles = [
            ("opencl_compute.csv", "opencl-compute", args.n_opencl),
            ("opencl_reload.csv", "opencl-reload", args.n_reload),
            ("cpu_compute.csv", "cpu-compute", args.n_cpu),
        ]
        for csv_name, mode, n_predict in profiles:
            script = profile_command(
                remote_dir=args.remote_dir,
                model=model_remote,
                csv_name=csv_name,
                prompt=args.prompt,
                n_predict=n_predict,
                ctx_size=args.ctx_size,
                batch=args.batch,
                threads=args.threads,
                mode=mode,
                remote_trace=remote_trace,
            )
            commands.append({"phase": mode, "remote_shell": script})
            proc = adb_shell(args.adb_serial, script, dry_run=args.dry_run, check=False)
            if proc.returncode != 0 and mode == "cpu-compute":
                print("warning: CPU_Elastic profile failed; solver will require --allow-cpu-fallback or a later CPU profile", file=sys.stderr)
            elif proc.returncode != 0:
                raise SystemExit(proc.stdout)
            if not args.dry_run:
                print(proc.stdout)
                adb(args.adb_serial, ["pull", f"{args.remote_dir}/{csv_name}", str(csv_dir / csv_name)], check=False)

    csv_paths = [csv_dir / "opencl_compute.csv", csv_dir / "opencl_reload.csv", csv_dir / "cpu_compute.csv"]
    existing_csvs = [p for p in csv_paths if p.exists() and csv_rows(p) > 0]
    if args.dry_run:
        existing_csvs = csv_paths
    if not existing_csvs and not args.dry_run:
        raise SystemExit(f"no non-empty profile CSVs found in {csv_dir}")

    if args.model_meta:
        commands.append({"phase": "use_existing_model_meta", "path": str(meta_path)})
    else:
        build_meta_cmd = [
            sys.executable,
            str(PLAN_DIR / "build_model_meta_from_profile.py"),
            *[str(p) for p in existing_csvs],
            "--out",
            str(meta_path),
        ]
        commands.append({"phase": "build_model_meta", "cmd": build_meta_cmd})
        run(build_meta_cmd, dry_run=args.dry_run)

    build_cost_cmd = [
        sys.executable,
        str(PLAN_DIR / "build_cost_model.py"),
        *[str(p) for p in existing_csvs],
        "--out-dir",
        str(cost_dir),
        "--device",
        "android-opencl",
        "--model",
        args.model_tag,
    ]
    commands.append({"phase": "build_cost_model", "cmd": build_cost_cmd})
    run(build_cost_cmd, dry_run=args.dry_run)

    build_table_cmd = [
        sys.executable,
        str(PLAN_DIR / "build_offline_budget_table.py"),
        "--model-meta",
        str(meta_path),
        "--cost-dir",
        str(cost_dir),
        "--out-dir",
        str(table_dir),
        "--budgets",
        args.budgets,
        "--kv-mib",
        str(args.kv_mib),
        "--misc-mib",
        str(args.misc_mib),
        "--safety-mib",
        str(args.safety_mib),
        "--time-limit-ms",
        str(args.time_limit_ms),
    ]
    if args.allow_cpu_fallback:
        build_table_cmd.append("--allow-cpu-fallback")
    commands.append({"phase": "build_offline_table", "cmd": build_table_cmd})
    run(build_table_cmd, dry_run=args.dry_run)

    if args.push_plan_dir:
        remote_table = f"{args.remote_dir}/plans_{args.model_tag}"
        adb_shell(args.adb_serial, f"rm -rf {shlex.quote(remote_table)} && mkdir -p {shlex.quote(remote_table)}", dry_run=args.dry_run)
        adb(args.adb_serial, ["push", str(table_dir) + "/.", remote_table + "/"], dry_run=args.dry_run)
        commands.append({"phase": "push_offline_table", "remote_dir": remote_table})

    manifest = {
        "started": started,
        "model": model_remote,
        "model_tag": args.model_tag,
        "remote_dir": args.remote_dir,
        "trace": str(args.trace) if args.trace else None,
        "csv_rows": {p.name: csv_rows(p) for p in csv_paths},
        "model_meta": str(meta_path),
        "cost_dir": str(cost_dir),
        "offline_table": str(table_dir),
        "commands": commands,
    }
    if not args.dry_run:
        if meta_path.exists():
            meta = read_json(meta_path)
            manifest["weights"] = len(meta.get("weights", []))
            manifest["ops"] = len(meta.get("ops", []))
        if (cost_dir / "summary.json").exists():
            manifest["cost_summary"] = read_json(cost_dir / "summary.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
