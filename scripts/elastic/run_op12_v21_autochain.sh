#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

chain_log="${ELASTIC_CHAIN_LOG:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/op12_v21_autochain_20260729.log}"
mkdir -p "$(dirname "$chain_log")"
exec > >(tee -a "$chain_log") 2>&1

serial="${ELASTIC_ADB_SERIAL:-5ae7a43d}"
export ELASTIC_ADB_SERIAL="$serial"
export ELASTIC_PIPELINE_COPY_CPU="${ELASTIC_PIPELINE_COPY_CPU:-1}"
ports=("${ANDROID_ADB_SERVER_PORT:-5037}" 5037 5038 5039)
local_binary="${ELASTIC_LOCAL_BINARY:-build-android-llama/bin/llama-cli}"
remote_root="/data/local/tmp/hyzheng/elastic"
remote_binary="$remote_root/llama-cli"
remote_temporary="$remote_root/llama-cli.outputpin-v21.tmp"
remote_backup="$remote_root/llama-cli.before-outputpin-v21"

active_port=""
wait_for_adb() {
  while true; do
    for port in "${ports[@]}"; do
      if [[ -n "$port" ]] && [[ "$(adb -P "$port" -s "$serial" get-state \
          2>/dev/null || true)" == "device" ]]; then
        active_port="$port"
        export ANDROID_ADB_SERVER_PORT="$port"
        echo "[$(date --iso-8601=seconds)] OP12 adb ready on port $port"
        return
      fi
    done
    echo "[$(date --iso-8601=seconds)] waiting for OP12 adb permission"
    sleep 30
  done
}

wait_for_no_inference() {
  while true; do
    local pids
    pids="$(adb -P "$active_port" -s "$serial" shell \
      'pidof llama-cli 2>/dev/null || true' 2>/dev/null | tr -d '\r')"
    if [[ -z "$pids" ]]; then
      return
    fi
    echo "[$(date --iso-8601=seconds)] OP12 llama-cli still running: $pids"
    sleep 30
    wait_for_adb
  done
}

deploy_binary() {
  if [[ ! -x "$local_binary" ]]; then
    echo "missing Android binary: $local_binary" >&2
    exit 1
  fi
  wait_for_no_inference
  local local_sha remote_sha installed_sha
  local_sha="$(sha256sum "$local_binary" | awk '{print $1}')"
  adb -P "$active_port" -s "$serial" push \
    "$local_binary" "$remote_temporary"
  remote_sha="$(adb -P "$active_port" -s "$serial" shell \
    "sha256sum '$remote_temporary'" | awk '{print $1}' | tr -d '\r')"
  if [[ "$local_sha" != "$remote_sha" ]]; then
    echo "binary upload hash mismatch: local=$local_sha remote=$remote_sha" >&2
    exit 1
  fi
  adb -P "$active_port" -s "$serial" shell \
    "if [ -f '$remote_binary' ] && [ ! -f '$remote_backup' ]; then
       cp -p '$remote_binary' '$remote_backup';
     fi;
     chmod 755 '$remote_temporary' &&
     mv -f '$remote_temporary' '$remote_binary'"
  installed_sha="$(adb -P "$active_port" -s "$serial" shell \
    "sha256sum '$remote_binary'" | awk '{print $1}' | tr -d '\r')"
  if [[ "$local_sha" != "$installed_sha" ]]; then
    echo "installed binary hash mismatch: local=$local_sha remote=$installed_sha" >&2
    exit 1
  fi
  # Pin every downstream row to the exact bytes deployed at chain start,
  # even if another local build finishes while the multi-hour sweep runs.
  export ELASTIC_EXPECTED_BINARY_SHA256="$local_sha"
  echo "[$(date --iso-8601=seconds)] deployed OP12 binary sha256=$local_sha"
  adb -P "$active_port" -s "$serial" shell \
    "dumpsys battery | grep -E 'level:|status:|powered:'; \
     dumpsys thermalservice | grep -m 1 'Thermal Status'" || true
}

read_residency_selection() {
  python3 - "$1" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    document["selected_policy"],
    document["planner_stream_reserve_mib"],
)
PY
}

wait_for_adb
deploy_binary

# CPU is validated first.  Only after its fusion/pin contract and complete
# fixed-budget calibration pass do we start the GPU backend.
ELASTIC_FIXED_ROOT=exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_outputpin_fusion_contract_smoke_op12_20260729_v21 \
ELASTIC_RATIOS=30 \
ELASTIC_N_PREDICT=6 \
ELASTIC_WARMUP_TOKENS=2 \
ELASTIC_MIN_MEASURE_TOKENS=3 \
ELASTIC_SKIP_CALIBRATION=1 \
bash scripts/elastic/run_op12_cpu_fixed_calibration_v21.sh

# Held-in fixed-budget calibration. CPU includes both coarse implementations;
# the calibration selects Multi or Multi-fused globally across all budgets.
bash scripts/elastic/run_op12_cpu_fixed_calibration_v21.sh

cpu_fixed_profile="runtime/plan/profiles/granularity/op12_llama3_8b_q4_cpu_pipeline_budgetfair_outputpin_multifused_20260729_v21.json"
cpu_phase_profile="runtime/plan/profiles/granularity/op12_llama3_8b_q4_cpu_pipeline_budgetfair_outputpin_multifused_phase_only_20260729_v21.json"
cpu_cost_dir="runtime/plan/profiles/android-cpu/Meta-Llama-3-8B-Instruct-Q4_0_op12_outputpin_multifused_20260729_v21"
cpu_pilot_root="exp-results/Meta-Llama-3-8B-Instruct-Q4_0/plan_residency_pilot_cpu_op12_20260729_v21"

ELASTIC_ARTIFACT_ROOT="$cpu_pilot_root" \
ELASTIC_GRANULARITY_PROFILE="$cpu_fixed_profile" \
ELASTIC_COST_DIR="$cpu_cost_dir" \
bash scripts/elastic/run_op12_cpu_plan_residency_pilot_v20.sh
read -r cpu_policy cpu_reserve < <(
  read_residency_selection "$cpu_pilot_root/summary/selection.json")
echo "[$(date --iso-8601=seconds)] CPU residency policy=$cpu_policy reserve=${cpu_reserve}MiB"

cpu_boundary_root="exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_mixed_boundary_nodes_outputpin_op12_20260729_v21"
cpu_selected_profile="runtime/plan/profiles/granularity/op12_llama3_8b_q4_cpu_pipeline_mixedboundary_outputpin_20260729_v21.json"
ELASTIC_BASE_GRANULARITY_PROFILE="$cpu_fixed_profile" \
ELASTIC_COST_DIR="$cpu_cost_dir" \
ELASTIC_CALIBRATION_ROOT="$cpu_boundary_root" \
ELASTIC_SELECTED_PROFILE="$cpu_selected_profile" \
ELASTIC_PLAN_RESIDENCY_POLICY="$cpu_policy" \
ELASTIC_PLANNER_STREAM_RESERVE_MIB="$cpu_reserve" \
bash scripts/elastic/run_op12_cpu_mixed_boundary_calibration.sh

cpu_selected_phase_profile="runtime/plan/profiles/granularity/op12_llama3_8b_q4_cpu_pipeline_mixedboundary_outputpin_phase_only_20260729_v21.json"
python3 runtime/plan/calibrate_granularity_profile.py \
  --backend cpu \
  --strip-residual-from "$cpu_selected_profile" \
  --output "$cpu_selected_phase_profile" \
  --source OP12-Llama3-8B-Q4-CPU-mixedboundary-outputpin-phase-only-20260729-v21

cpu_main_root="exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_full10min_outputpin_op12_20260729_v21"
cpu_before_root="exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_full10min_outputpin_diff_before_op12_20260729_v21"
final_root="exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_full10min_final_outputpin_op12_20260729_v21"
ELASTIC_ARTIFACT_ROOT="$cpu_main_root" \
ELASTIC_METHODS=static-min,static-max,mru,offline-mixed,online,diff-tree-mixed \
ELASTIC_GRANULARITY_PROFILE="$cpu_selected_profile" \
ELASTIC_COST_DIR="$cpu_cost_dir" \
ELASTIC_GRANULARITY_PLACEMENT_SOURCE=stateful-cp \
ELASTIC_PLAN_RESIDENCY_POLICY="$cpu_policy" \
ELASTIC_PLANNER_STREAM_RESERVE_MIB="$cpu_reserve" \
ELASTIC_CPU_THREADS=5 \
ELASTIC_CPU_MASK=0xf8 \
ELASTIC_PIPELINE_LOAD_CPU=0 \
ELASTIC_PIPELINE_PREPARE_CPU=2 \
ELASTIC_PIPELINE_GRAPH_LOOKAHEAD=1 \
ELASTIC_COOLDOWN_THERMAL_MAX_C=35 \
ELASTIC_MIN_CPU_FREQ_LIMIT_KHZ=1400000 \
ELASTIC_MIN_CPU_MEAN_FREQ_KHZ=1550000 \
ELASTIC_MIN_CPU_MEDIAN_SAMPLE_MIN_KHZ=1500000 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op13_cpu_full10min_difftree.sh

ELASTIC_ARTIFACT_ROOT="$cpu_before_root" \
ELASTIC_GRANULARITY_PROFILE="$cpu_selected_phase_profile" \
ELASTIC_COST_DIR="$cpu_cost_dir" \
ELASTIC_GRANULARITY_PLACEMENT_SOURCE=stateful-cp \
ELASTIC_PLAN_RESIDENCY_POLICY="$cpu_policy" \
ELASTIC_PLANNER_STREAM_RESERVE_MIB="$cpu_reserve" \
ELASTIC_CPU_THREADS=5 \
ELASTIC_CPU_MASK=0xf8 \
ELASTIC_PIPELINE_LOAD_CPU=0 \
ELASTIC_PIPELINE_PREPARE_CPU=2 \
ELASTIC_COOLDOWN_THERMAL_MAX_C=35 \
ELASTIC_MIN_CPU_FREQ_LIMIT_KHZ=1400000 \
ELASTIC_MIN_CPU_MEAN_FREQ_KHZ=1550000 \
ELASTIC_MIN_CPU_MEDIAN_SAMPLE_MIN_KHZ=1500000 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op13_cpu_full10min_diff_before.sh

python3 runtime/plan/summarize_dynamic_baseline_matrix.py \
  --backend cpu \
  --main-results "$cpu_main_root/summary/results.csv" \
  --diff-before-results "$cpu_before_root/summary/results.csv" \
  --diff-now-results "$cpu_main_root/summary/results.csv" \
  --output-dir "$final_root/cpu"
echo "[$(date --iso-8601=seconds)] OP12 v21 CPU consolidation complete: $final_root/cpu"

# CPU fixed, pilot, boundary calibration, Diff-now/baselines, and Diff-before
# are complete before GPU starts. GPU remains a separate Elastic-only
# backend: no heterogeneous placement.
bash scripts/elastic/run_op12_gpu_pin_contract_smoke_v21.sh
bash scripts/elastic/run_op12_gpu_fixed_calibration_v21.sh

gpu_fixed_profile="runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_budgetfair_outputpin_20260729_v21.json"
gpu_cost_dir="runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_op12_gpu_outputpin_20260729_v21"
gpu_pilot_root="exp-results/Meta-Llama-3-8B-Instruct-Q4_0/plan_residency_pilot_gpu_op12_20260729_v21"

ELASTIC_ARTIFACT_ROOT="$gpu_pilot_root" \
ELASTIC_GRANULARITY_PROFILE="$gpu_fixed_profile" \
ELASTIC_COST_DIR="$gpu_cost_dir" \
bash scripts/elastic/run_op12_gpu_plan_residency_pilot_v21.sh
read -r gpu_policy gpu_reserve < <(
  read_residency_selection "$gpu_pilot_root/summary/selection.json")
echo "[$(date --iso-8601=seconds)] GPU residency policy=$gpu_policy reserve=${gpu_reserve}MiB"

gpu_boundary_root="exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_gpu_mixed_boundary_nodes_outputpin_op12_20260729_v21"
gpu_selected_profile="runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_mixedboundary_outputpin_20260729_v21.json"
ELASTIC_BASE_GRANULARITY_PROFILE="$gpu_fixed_profile" \
ELASTIC_COST_DIR="$gpu_cost_dir" \
ELASTIC_CALIBRATION_ROOT="$gpu_boundary_root" \
ELASTIC_SELECTED_PROFILE="$gpu_selected_profile" \
ELASTIC_PLAN_RESIDENCY_POLICY="$gpu_policy" \
ELASTIC_PLANNER_STREAM_RESERVE_MIB="$gpu_reserve" \
bash scripts/elastic/run_op12_gpu_mixed_boundary_calibration.sh

gpu_selected_phase_profile="runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_mixedboundary_outputpin_phase_only_20260729_v21.json"
python3 runtime/plan/calibrate_granularity_profile.py \
  --backend gpu \
  --strip-residual-from "$gpu_selected_profile" \
  --output "$gpu_selected_phase_profile" \
  --source OP12-Llama3-8B-Q4-GPU-mixedboundary-outputpin-phase-only-20260729-v21

gpu_main_root="exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_gpu_native_full10min_outputpin_op12_20260729_v21"
gpu_before_root="exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_gpu_native_full10min_outputpin_diff_before_op12_20260729_v21"
ELASTIC_ARTIFACT_ROOT="$gpu_main_root" \
ELASTIC_METHODS=static-min,static-max,mru,offline-mixed,online,diff-tree-mixed \
ELASTIC_GRANULARITY_PROFILE="$gpu_selected_profile" \
ELASTIC_COST_DIR="$gpu_cost_dir" \
ELASTIC_GRANULARITY_PLACEMENT_SOURCE=stateful-cp \
ELASTIC_PLAN_RESIDENCY_POLICY="$gpu_policy" \
ELASTIC_PLANNER_STREAM_RESERVE_MIB="$gpu_reserve" \
ELASTIC_PIPELINE_GRAPH_LOOKAHEAD=1 \
ELASTIC_COOLDOWN_THERMAL_MAX_C=42 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op12_gpu_full10min_difftree.sh

ELASTIC_ARTIFACT_ROOT="$gpu_before_root" \
ELASTIC_GRANULARITY_PROFILE="$gpu_selected_phase_profile" \
ELASTIC_COST_DIR="$gpu_cost_dir" \
ELASTIC_GRANULARITY_PLACEMENT_SOURCE=stateful-cp \
ELASTIC_PLAN_RESIDENCY_POLICY="$gpu_policy" \
ELASTIC_PLANNER_STREAM_RESERVE_MIB="$gpu_reserve" \
ELASTIC_COOLDOWN_THERMAL_MAX_C=42 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op12_gpu_full10min_diff_before_v21.sh

python3 runtime/plan/summarize_dynamic_baseline_matrix.py \
  --backend gpu \
  --main-results "$gpu_main_root/summary/results.csv" \
  --diff-before-results "$gpu_before_root/summary/results.csv" \
  --diff-now-results "$gpu_main_root/summary/results.csv" \
  --output-dir "$final_root/gpu"

echo "[$(date --iso-8601=seconds)] OP12 v21 CPU/GPU autochain complete: $final_root/{cpu,gpu}"
