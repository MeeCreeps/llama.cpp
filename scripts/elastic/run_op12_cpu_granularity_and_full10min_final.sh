#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

serial="${ELASTIC_ADB_SERIAL:-5ae7a43d}"
fixed_root="${ELASTIC_FIXED_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_pipeline_fixed_30_90_budgetfair_i8mm_workerpin_prep2_op12_20260728}"
profile="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_cpu_pipeline_budgetfair_i8mm_20260728.json}"
phase_profile="${ELASTIC_PHASE_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_cpu_pipeline_budgetfair_i8mm_phase_only_20260728.json}"
placement_cost_dir="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-cpu/Meta-Llama-3-8B-Instruct-Q4_0_op12_i8mm_20260728}"
base_cost_dir="runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01"
main_root="${ELASTIC_MAIN_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_full10min_all_baselines_op12_20260728_budgetfair_cpucost_v9}"
before_root="${ELASTIC_BEFORE_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_full10min_diff_before_op12_20260728_budgetfair_cpucost_v9}"
smoke_main_root="${ELASTIC_SMOKE_MAIN_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_trace_smoke190_280_op12_20260728_budgetfair_cpucost_v9}"
smoke_before_root="${ELASTIC_SMOKE_BEFORE_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_trace_smoke190_280_diff_before_op12_20260728_budgetfair_cpucost_v9}"

# OP12/SM8650 resource partition selected by a held-out Tensor diagnostic:
# CPU0 handles file reads, CPU2 handles layout preparation, and CPUs3-7 run
# the common five-thread compute pool. The same partition is used by every
# granularity and every dynamic-plan baseline.
threads=5
cpu_mask=0xf8
load_cpu=0
prepare_cpu=2

restore_device_state() {
  adb -s "$serial" shell \
    'cmd power set-fixed-performance-mode-enabled false; svc wifi enable; svc data enable; settings put global stay_on_while_plugged_in 3' \
    >/dev/null 2>&1 || true
}
trap restore_device_state EXIT

adb -s "$serial" shell \
  'settings put global stay_on_while_plugged_in 0; svc wifi disable; svc data disable; input keyevent KEYCODE_SLEEP' \
  >/dev/null

python3 scripts/elastic/run_granularity_model_sweep.py \
  --serial "$serial" \
  --backend cpu \
  --remote-binary /data/local/tmp/hyzheng/elastic/llama-cli \
  --remote-model /data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --remote-lib-dir /data/local/tmp/hyzheng/elastic \
  --remote-work-dir /data/local/tmp/hyzheng/elastic/granularity_cpu_fixed_op12_final \
  --output "$fixed_root" \
  --modes multi tensor cut \
  --ratios 30,40,50,60,70,80,90 \
  --weight-mib 4437.8 \
  --kv-mib 512 \
  --misc-mib 256 \
  --pin token_embd,output \
  --pipeline on \
  --pipeline-lookahead 1 \
  --pipeline-lookahead-mib 128 \
  --pipeline-graph-lookahead 1 \
  --pipeline-max-pending 256 \
  --pipeline-max-pending-mib 128 \
  --cpu-pipeline-staging pooled \
  --cpu-repack-threads 1 \
  --pipeline-load-cpu "$load_cpu" \
  --pipeline-prepare-cpu "$prepare_cpu" \
  --cut-parts 2 \
  --cut-dual-compute off \
  --threads "$threads" \
  --cpu-mask "$cpu_mask" \
  --cpu-strict 1 \
  --batch 32 \
  --ubatch 32 \
  --context 4096 \
  --n-predict 14 \
  --ignore-eos \
  --seed 42 \
  --prompt Hi \
  --repeats 1 \
  --fixed-warmup-decode-tokens 4 \
  --fixed-min-measure-tokens 9 \
  --timeout 600 \
  --max-cpu-start-c 35 \
  --max-gpu-start-c 35 \
  --max-skin-start-c 31 \
  --thermal-poll-s 10 \
  --thermal-wait-timeout-s 14400 \
  --require-device-idle \
  --device-idle-poll-s 2 \
  --min-mem-available-mib 4800 \
  --min-battery-level-pct 20 \
  --min-cpu-freq-limit-khz 1800000 \
  --fixed-performance-mode \
  --resume

python3 runtime/plan/calibrate_granularity_profile.py \
  --runs-csv "$fixed_root/runs.csv" \
  --backend cpu \
  --output "$profile" \
  --phase-only-output "$phase_profile" \
  --placement-cost-base-dir "$base_cost_dir" \
  --placement-cost-output-dir "$placement_cost_dir" \
  --planned-weight-mib 4154.9765625 \
  --unplanned-pinned-mib 281.8125 \
  --source OP12-Llama3-8B-Q4-CPU-pipeline-budgetfair-I8MM-workerpin-20260728

ELASTIC_ADB_SERIAL="$serial" \
ELASTIC_ARTIFACT_ROOT="$smoke_main_root" \
ELASTIC_METHODS=offline-mixed,online,diff-tree-mixed \
ELASTIC_WINDOW_SEC=90 \
ELASTIC_WINDOW_START_SEC=190 \
ELASTIC_BENCH_SECONDS=90 \
ELASTIC_GRANULARITY_PROFILE="$profile" \
ELASTIC_COST_DIR="$placement_cost_dir" \
ELASTIC_CPU_THREADS="$threads" \
ELASTIC_CPU_MASK="$cpu_mask" \
ELASTIC_PIPELINE_LOAD_CPU="$load_cpu" \
ELASTIC_PIPELINE_PREPARE_CPU="$prepare_cpu" \
ELASTIC_COOLDOWN_THERMAL_MAX_C=35 \
ELASTIC_MIN_CPU_FREQ_LIMIT_KHZ=1800000 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op13_cpu_full10min_difftree.sh

ELASTIC_ADB_SERIAL="$serial" \
ELASTIC_ARTIFACT_ROOT="$smoke_before_root" \
ELASTIC_WINDOW_SEC=90 \
ELASTIC_WINDOW_START_SEC=190 \
ELASTIC_BENCH_SECONDS=90 \
ELASTIC_GRANULARITY_PROFILE="$phase_profile" \
ELASTIC_COST_DIR="$placement_cost_dir" \
ELASTIC_CPU_THREADS="$threads" \
ELASTIC_CPU_MASK="$cpu_mask" \
ELASTIC_PIPELINE_LOAD_CPU="$load_cpu" \
ELASTIC_PIPELINE_PREPARE_CPU="$prepare_cpu" \
ELASTIC_COOLDOWN_THERMAL_MAX_C=35 \
ELASTIC_MIN_CPU_FREQ_LIMIT_KHZ=1800000 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op13_cpu_full10min_diff_before.sh

# Fail fast on an invalid held-out diagnostic before spending hours on the
# full trace.  Method ordering is deliberately not a gate: the measured
# ordering is an experimental result, not a condition the implementation may
# enforce.
python3 - \
  "$smoke_main_root/summary/results.csv" \
  "$smoke_before_root/summary/results.csv" <<'PY'
import csv
import sys

main_path, before_path = sys.argv[1:]
main = {
    row["method"]: row
    for row in csv.DictReader(open(main_path, newline="", encoding="utf-8"))
}
before_rows = list(csv.DictReader(
    open(before_path, newline="", encoding="utf-8")))
required = ("offline-mixed", "online", "diff-tree-mixed")
for method in required:
    row = main.get(method)
    if row is None or row.get("status") != "ok":
        raise SystemExit(f"smoke gate: invalid {method}: {row}")
    if int(row.get("pipeline_budget_violations", "-1")) != 0:
        raise SystemExit(f"smoke gate: budget violation in {method}")
if len(before_rows) != 1 or before_rows[0].get("status") != "ok":
    raise SystemExit("smoke gate: invalid Diff-before row")
if int(before_rows[0].get("pipeline_budget_violations", "-1")) != 0:
    raise SystemExit("smoke gate: budget violation in Diff-before")

latency = {
    "offline": float(main["offline-mixed"]["raw_ms_per_token"]),
    "online": float(main["online"]["raw_ms_per_token"]),
    "diff-now": float(main["diff-tree-mixed"]["raw_ms_per_token"]),
    "diff-before": float(before_rows[0]["raw_ms_per_token"]),
}
print({"smoke_latency_ms_per_token": latency}, flush=True)
PY

ELASTIC_ADB_SERIAL="$serial" \
ELASTIC_ARTIFACT_ROOT="$main_root" \
ELASTIC_GRANULARITY_PROFILE="$profile" \
ELASTIC_COST_DIR="$placement_cost_dir" \
ELASTIC_CPU_THREADS="$threads" \
ELASTIC_CPU_MASK="$cpu_mask" \
ELASTIC_PIPELINE_LOAD_CPU="$load_cpu" \
ELASTIC_PIPELINE_PREPARE_CPU="$prepare_cpu" \
ELASTIC_COOLDOWN_THERMAL_MAX_C=35 \
ELASTIC_MIN_CPU_FREQ_LIMIT_KHZ=1800000 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op13_cpu_full10min_difftree.sh

ELASTIC_ADB_SERIAL="$serial" \
ELASTIC_ARTIFACT_ROOT="$before_root" \
ELASTIC_GRANULARITY_PROFILE="$phase_profile" \
ELASTIC_COST_DIR="$placement_cost_dir" \
ELASTIC_CPU_THREADS="$threads" \
ELASTIC_CPU_MASK="$cpu_mask" \
ELASTIC_PIPELINE_LOAD_CPU="$load_cpu" \
ELASTIC_PIPELINE_PREPARE_CPU="$prepare_cpu" \
ELASTIC_COOLDOWN_THERMAL_MAX_C=35 \
ELASTIC_MIN_CPU_FREQ_LIMIT_KHZ=1800000 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op13_cpu_full10min_diff_before.sh
