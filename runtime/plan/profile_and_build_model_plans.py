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


def write_stage_probe_plan(meta_path: Path, out_path: Path, backend: str, budget_mib: int,
                           samples_per_size: int) -> None:
    meta = read_json(meta_path)
    all_weights = list(meta.get("weights", []))
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for w in all_weights:
        key = (int(w.get("byte_size", w.get("bytes", 0))), str(w.get("quant", "")))
        grouped.setdefault(key, []).append(w)

    weights = []
    for _, group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        weights.extend(group[:max(1, samples_per_size)])

    # Reindex the compact probe plan densely. ExecPlan::weight_by_id expects a
    # vector index, so sparse original model ids would be unsafe here.
    for i, w in enumerate(weights):
        w = dict(w)
        w["weight_id"] = i
        weights[i] = w

    ops = [
        {
            "op_id": i,
            "name": w.get("name", f"weight_{i}"),
            "layer": w.get("layer", -1),
            "weight_id": i,
        }
        for i, w in enumerate(weights)
    ]

    first_consumer: dict[int, int] = {}
    for i, op in enumerate(ops):
        wid = int(op.get("weight_id", i))
        first_consumer.setdefault(wid, int(op.get("op_id", i)))

    be = backend.lower()
    if be not in {"gpu", "cpu"}:
        raise ValueError(f"unsupported stage probe backend: {backend}")

    plan_weights = []
    timeline = []
    for i, w in enumerate(weights):
        wid = int(w.get("weight_id", i))
        name = str(w.get("name", f"weight_{wid}"))
        byte_size = int(w.get("byte_size", w.get("bytes", 0)))
        anchor = first_consumer.get(wid, 0)
        plan_weights.append(
            {
                "weight_id": wid,
                "name": name,
                "layer": int(w.get("layer", -1)),
                "byte_size": byte_size,
                "location": "disk",
                "pinned": False,
                "xform": "none",
            }
        )
        timeline.append(
            {
                "kind": "load",
                "weight_id": wid,
                "from_loc": "disk",
                "to_loc": "cpu",
                "engine": "disk",
                "anchor_op_id": anchor,
                "overlap_group": -1,
            }
        )
        if be == "gpu":
            timeline.append(
                {
                    "kind": "transfer",
                    "weight_id": wid,
                    "from_loc": "cpu",
                    "to_loc": "gpu",
                    "engine": "transfer",
                    "anchor_op_id": anchor,
                    "overlap_group": -1,
                }
            )
            timeline.append(
                {
                    "kind": "xform",
                    "weight_id": wid,
                    "from_loc": "gpu",
                    "to_loc": "gpu",
                    "engine": "gpu",
                    "anchor_op_id": anchor,
                    "overlap_group": -1,
                }
            )
        else:
            timeline.append(
                {
                    "kind": "xform",
                    "weight_id": wid,
                    "from_loc": "cpu",
                    "to_loc": "cpu",
                    "engine": "cpu",
                    "anchor_op_id": anchor,
                    "overlap_group": -1,
                }
            )

    plan_ops = []
    for i, op in enumerate(ops):
        wid = int(op.get("weight_id", i))
        plan_ops.append(
            {
                "op_id": int(op.get("op_id", i)),
                "name": op.get("name", weights[wid].get("name", f"weight_{wid}") if 0 <= wid < len(weights) else f"op_{i}"),
                "layer": int(op.get("layer", -1)),
                "compute_backend": be,
                "weight_id": wid,
                "dispatch": "static",
                "migrate": False,
                "migrate_from": be,
                "migrate_xform": "none",
            }
        )

    write_json(
        out_path,
        {
            "schema_version": 1,
            "budget_mib": budget_mib,
            "kv_bytes": 0,
            "misc_bytes": 0,
            "weights": plan_weights,
            "ops": plan_ops,
            "timeline": timeline,
            "pred_per_token_ms": 0.0,
            "bottleneck": f"{be}_stage_probe",
            "cost_model": {
                "purpose": "fine_stage_profile_probe",
                "backend": be,
            },
        },
    )


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
    remote_plan: str | None = None,
    flash_attn: str = "on",
) -> str:
    env = {
        "LD_LIBRARY_PATH": remote_dir,
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
        "-no-cnv",
    ]

    if mode == "opencl-compute":
        env.update(
            {
                "GGML_ELASTIC_PROFILE": "1",
                "GGML_OPENCL_DISABLE_ALLOC_HOST_PTR": "1",
                "GGML_OPENCL_USE_SVM": "1",
                "GGML_OPENCL_ELASTIC": "1",
            }
        )
        argv += ["-ngl", "99", "-fa", flash_attn]
    elif mode == "opencl-reload":
        env.update(
            {
                "GGML_ELASTIC_PROFILE": "1",
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
        argv += ["-ngl", "99", "-fa", flash_attn]
    elif mode == "opencl-stage":
        if not remote_plan:
            raise ValueError("opencl-stage requires remote_plan")
        env.update(
            {
                "GGML_OPENCL_DISABLE_ALLOC_HOST_PTR": "1",
                "GGML_OPENCL_USE_SVM": "1",
                "GGML_OPENCL_ELASTIC": "1",
                "GGML_ELASTIC_TIMING": "1",
                "GGML_ELASTIC_STAGE_DETAIL": "1",
                "LLAMA_ELASTIC_APPLY": remote_plan,
                "LLAMA_ELASTIC_DEFER_STAGE": "0",
            }
        )
        argv += ["-ngl", "99", "-fa", flash_attn]
    elif mode == "cpu-compute":
        env.update(
            {
                "GGML_ELASTIC_CHUNK_SIZE": "1",
                "GGML_ELASTIC_PROFILE": "1",
            }
        )
        argv += ["-ngl", "0", "-dev", "CPU_Elastic"]
    elif mode == "cpu-stage":
        if not remote_plan:
            raise ValueError("cpu-stage requires remote_plan")
        env.update(
            {
                "LLAMA_ELASTIC_APPLY": remote_plan,
                "LLAMA_ELASTIC_DEFER_STAGE": "0",
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
    ap.add_argument("--stage-samples-per-size", type=int, default=4,
                    help="number of representative weights per byte-size/quant group in stage probe plans")
    ap.add_argument("--ctx-size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--flash-attn", choices=("on", "off", "auto"), default="on",
                    help="flash-attention mode for OpenCL profiling phases")
    ap.add_argument("--allow-cpu-fallback", action="store_true", help="allow solver synthetic CPU cost if CPU profile is incomplete")
    ap.add_argument("--cp-objective", choices=("resource_makespan", "interval_makespan", "sum"), default="resource_makespan")
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
    probe_dir = artifact_root / "stage_probe"
    opencl_stage_plan = probe_dir / "opencl_stage_probe.json"
    cpu_stage_plan = probe_dir / "cpu_stage_probe.json"
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

        remote_opencl_stage_plan = None
        remote_cpu_stage_plan = None
        if args.model_meta:
            write_stage_probe_plan(
                meta_path,
                opencl_stage_plan,
                "gpu",
                max(int(args.budgets.split(",")[0]), args.kv_mib + args.misc_mib + args.safety_mib),
                args.stage_samples_per_size,
            )
            write_stage_probe_plan(
                meta_path,
                cpu_stage_plan,
                "cpu",
                max(int(args.budgets.split(",")[0]), args.kv_mib + args.misc_mib + args.safety_mib),
                args.stage_samples_per_size,
            )
            remote_opencl_stage_plan = f"{args.remote_dir}/{opencl_stage_plan.name}"
            remote_cpu_stage_plan = f"{args.remote_dir}/{cpu_stage_plan.name}"
            adb(args.adb_serial, ["push", str(opencl_stage_plan), remote_opencl_stage_plan], dry_run=args.dry_run)
            adb(args.adb_serial, ["push", str(cpu_stage_plan), remote_cpu_stage_plan], dry_run=args.dry_run)
        else:
            print("warning: --model-meta not provided; skipping fine-stage probe profiles in the first pass", file=sys.stderr)

        profiles = [
            ("opencl_compute.csv", "opencl-compute", args.n_opencl),
            ("opencl_stage.csv", "opencl-stage", args.n_reload),
            ("opencl_reload_legacy.csv", "opencl-reload", args.n_reload),
            ("cpu_compute.csv", "cpu-compute", args.n_cpu),
            ("cpu_stage.csv", "cpu-stage", args.n_cpu),
        ]
        for csv_name, mode, n_predict in profiles:
            if mode == "opencl-stage" and not remote_opencl_stage_plan:
                continue
            if mode == "cpu-stage" and not remote_cpu_stage_plan:
                continue
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
                remote_plan=remote_opencl_stage_plan if mode == "opencl-stage" else remote_cpu_stage_plan if mode == "cpu-stage" else None,
                flash_attn=args.flash_attn,
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

    csv_paths = [
        csv_dir / "opencl_compute.csv",
        csv_dir / "opencl_stage.csv",
        csv_dir / "opencl_reload_legacy.csv",
        csv_dir / "cpu_compute.csv",
        csv_dir / "cpu_stage.csv",
    ]
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
        "--cp-objective",
        str(args.cp_objective),
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
