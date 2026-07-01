#!/usr/bin/env python3
"""Run a repeatable elastic-memory pipeline contention benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


SINGLE_RE = re.compile(
    r"^single\s+(?P<stage>\S+)\s+med=\s*(?P<med>[0-9.]+)\s+ms.*"
    r"size=\s*(?P<size>[0-9.]+)\s+MiB\s+rate=\s*(?P<rate>[0-9.]+)\s+MiB/s"
)
OVERLAP_RE = re.compile(
    r"^overlap\s+(?P<pair>\S+)\s+med=\s*(?P<med>[0-9.]+)\s+ms\s+"
    r"ideal=\s*(?P<ideal>[0-9.]+)\s+ms\s+sum=\s*(?P<serial>[0-9.]+)\s+ms\s+"
    r"competition=\s*(?P<competition>[0-9.]+)\s+speedup_vs_serial=\s*(?P<speedup>[0-9.]+)"
)
PIPELINE_RE = re.compile(
    r"^pipeline wall=\s*(?P<wall>[0-9.]+)\s+ms\s+item_avg=\s*(?P<item_avg>[0-9.]+)\s+ms\s+"
    r"serial_est=\s*(?P<serial>[0-9.]+)\s+ms\s+speedup_vs_serial=\s*(?P<speedup>[0-9.]+)\s+"
    r"steady_items/s=\s*(?P<items_s>[0-9.]+)"
)
PIPELINE_HEADER_RE = re.compile(r"^pipeline\s+.*\bmode=(?P<mode>\S+)\b")


PRESETS = {
    "op12-llm": [
        {
            "name": "llm-8b-attn-q4_0",
            "args": [
                "--iters", "20", "--warmup", "3",
                "--load-mb", "16",
                "--cpu-xform-mb", "9",
                "--cpu-compute-mb", "16",
                "--gpu-compute-mb", "16",
                "--k", "4096", "--m", "4096",
                "--cpu-rounds", "4",
                "--gpu-rounds", "64",
            ],
        },
        {
            "name": "llm-8b-ffn-q4_0",
            "args": [
                "--iters", "20", "--warmup", "3",
                "--load-mb", "32",
                "--cpu-xform-mb", "32",
                "--cpu-compute-mb", "32",
                "--gpu-compute-mb", "32",
                "--k", "4096", "--m", "14336",
                "--cpu-rounds", "8",
                "--gpu-rounds", "128",
            ],
        },
    ],
    "op12": [
        {
            "name": "short-gpu-compute",
            "args": [
                "--iters", "20", "--warmup", "3",
                "--load-mb", "16",
                "--cpu-xform-mb", "32",
                "--cpu-compute-mb", "32",
                "--gpu-compute-mb", "16",
                "--k", "4096", "--m", "4096",
                "--cpu-rounds", "4",
                "--gpu-rounds", "64",
            ],
        },
        {
            "name": "balanced-gpu-compute",
            "args": [
                "--iters", "20", "--warmup", "3",
                "--load-mb", "32",
                "--cpu-xform-mb", "32",
                "--cpu-compute-mb", "32",
                "--gpu-compute-mb", "32",
                "--k", "4096", "--m", "4096",
                "--cpu-rounds", "8",
                "--gpu-rounds", "128",
            ],
        },
        {
            "name": "long-gpu-compute",
            "args": [
                "--iters", "20", "--warmup", "3",
                "--load-mb", "32",
                "--cpu-xform-mb", "32",
                "--cpu-compute-mb", "32",
                "--gpu-compute-mb", "32",
                "--k", "4096", "--m", "4096",
                "--cpu-rounds", "16",
                "--gpu-rounds", "256",
            ],
        },
    ],
}


def classify(competition: float, speedup: float) -> str:
    if competition <= 1.10 and speedup >= 1.20:
        return "good"
    if competition <= 1.35 and speedup >= 1.05:
        return "partial"
    return "poor"


def parse_output(text: str, run_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pipeline_mode = ""
    for line in text.splitlines():
        hm = PIPELINE_HEADER_RE.match(line)
        if hm:
            pipeline_mode = hm.group("mode")
            continue
        m = SINGLE_RE.match(line)
        if m:
            rows.append({
                "run": run_name,
                "kind": "single",
                "name": m.group("stage"),
                "med_ms": m.group("med"),
                "ideal_ms": "",
                "serial_ms": "",
                "competition": "",
                "speedup_vs_serial": "",
                "classification": "",
                "size_mib": m.group("size"),
                "rate_mib_s": m.group("rate"),
            })
            continue
        m = OVERLAP_RE.match(line)
        if m:
            competition = float(m.group("competition"))
            speedup = float(m.group("speedup"))
            rows.append({
                "run": run_name,
                "kind": "overlap",
                "name": m.group("pair"),
                "med_ms": m.group("med"),
                "ideal_ms": m.group("ideal"),
                "serial_ms": m.group("serial"),
                "competition": m.group("competition"),
                "speedup_vs_serial": m.group("speedup"),
                "classification": classify(competition, speedup),
                "size_mib": "",
                "rate_mib_s": "",
            })
            continue
        m = PIPELINE_RE.match(line)
        if m:
            name = "disk_load->cpu_xform->gpu_xform->gpu_compute"
            if pipeline_mode:
                name += f":{pipeline_mode}"
            rows.append({
                "run": run_name,
                "kind": "pipeline",
                "name": name,
                "med_ms": m.group("wall"),
                "ideal_ms": "",
                "serial_ms": m.group("serial"),
                "competition": "",
                "speedup_vs_serial": m.group("speedup"),
                "classification": "pipeline",
                "size_mib": "",
                "rate_mib_s": m.group("items_s"),
            })
    return rows


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    overlap = [r for r in rows if r["kind"] == "overlap"]
    pipeline = [r for r in rows if r["kind"] == "pipeline"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Pipeline Contention Matrix Summary\n\n")
        f.write("| run | pair | med ms | ideal ms | serial ms | competition | speedup | class |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for r in overlap:
            f.write(
                f"| {r['run']} | `{r['name']}` | {r['med_ms']} | {r['ideal_ms']} | "
                f"{r['serial_ms']} | {r['competition']} | {r['speedup_vs_serial']} | "
                f"{r['classification']} |\n"
            )
        if pipeline:
            f.write("\n## Pipeline Runs\n\n")
            f.write("| run | wall ms | serial ms | speedup | steady items/s |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for r in pipeline:
                f.write(
                    f"| {r['run']} | {r['med_ms']} | {r['serial_ms']} | "
                    f"{r['speedup_vs_serial']} | {r['rate_mib_s']} |\n"
                )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="Path to llama-pipeline-contention-bench")
    ap.add_argument("--out-dir", required=True, help="Directory for stdout logs and summaries")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="op12-llm")
    ap.add_argument("--file", help="Large file for O_DIRECT disk_load and pipeline tests")
    ap.add_argument("--kernel-dir", default="ggml/src/ggml-opencl/kernels")
    ap.add_argument("--platform", default="0")
    ap.add_argument("--device", default="0")
    ap.add_argument("--pipeline", action="store_true", help="Also run full pipeline mode; requires --file")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = Path(args.bench)
    rows: list[dict[str, str]] = []

    for case in PRESETS[args.preset]:
        cmd = [
            str(bench),
            "--kernel-dir", args.kernel_dir,
            "--platform", args.platform,
            "--device", args.device,
            *case["args"],
        ]
        if args.file:
            cmd += ["--file", args.file]
        if args.pipeline:
            cmd += ["--pipeline"]
        log_path = out_dir / f"{case['name']}.log"
        print("running:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        log_path.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit(f"{case['name']} failed with exit code {proc.returncode}; see {log_path}")
        rows.extend(parse_output(proc.stdout, case["name"]))

    csv_path = out_dir / "pipeline_contention_matrix.csv"
    fieldnames = [
        "run", "kind", "name", "med_ms", "ideal_ms", "serial_ms",
        "competition", "speedup_vs_serial", "classification", "size_mib", "rate_mib_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(out_dir / "pipeline_contention_matrix.md", rows)
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
