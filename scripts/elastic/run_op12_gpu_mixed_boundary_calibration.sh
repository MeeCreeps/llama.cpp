#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

serial="${ELASTIC_ADB_SERIAL:-5ae7a43d}"
base_profile="${ELASTIC_BASE_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_budgetfair_20260729_v19.json}"
cost_dir="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_op12_gpu_20260729_v19}"
calibration_root="${ELASTIC_CALIBRATION_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_gpu_mixed_boundary_nodes_op12_20260729_v19}"
selected_profile="${ELASTIC_SELECTED_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_mixedboundary_20260729_v19.json}"
penalties="${ELASTIC_CUT_BOUNDARY_CANDIDATES:-0,0.2,0.5,1.0}"
trace_glob="scripts/elastic/inputs/op12_mixed_boundary_calibration/dynamic_trace.csv"
trace_sha256="$(sha256sum "$trace_glob" | awk '{print $1}')"
python_bin="${ELASTIC_PYTHON:-python3}"

mkdir -p "$calibration_root/profiles"
IFS=',' read -r -a penalty_values <<< "$penalties"
for penalty in "${penalty_values[@]}"; do
  tag="cut_${penalty//./p}"
  profile="$calibration_root/profiles/$tag.json"
  "$python_bin" runtime/plan/set_granularity_boundary_cost.py \
    --base "$base_profile" \
    --output "$profile" \
    --backend gpu \
    --cut-boundary-ms "$penalty" \
    --merge-boundary-ms 0 \
    --source-suffix op12-fixed-node

  methods="offline-mixed"
  if [[ "$penalty" == "0" || "$penalty" == "0.0" ]]; then
    methods="offline,offline-mixed"
  fi

  ANDROID_ADB_SERVER_PORT="$ANDROID_ADB_SERVER_PORT" \
  ELASTIC_ADB_SERIAL="$serial" \
  ELASTIC_ARTIFACT_ROOT="$calibration_root/$tag" \
  ELASTIC_METHODS="$methods" \
  ELASTIC_TRACE_GLOB="$trace_glob" \
  ELASTIC_EXPECTED_SOURCE_TRACE_SHA256="$trace_sha256" \
  ELASTIC_WINDOW_SEC=120 \
  ELASTIC_WINDOW_START_SEC=0 \
  ELASTIC_BENCH_SECONDS=120 \
  ELASTIC_REPLAY_SPEEDUP=1 \
  ELASTIC_GRANULARITY_PROFILE="$profile" \
  ELASTIC_COST_DIR="$cost_dir" \
  ELASTIC_GRANULARITY_MAX_EDITS=16 \
  ELASTIC_PLAN_SWITCH_MIN_GAP_STEPS=4 \
  ELASTIC_PIPELINE_GRAPH_LOOKAHEAD=1 \
  ELASTIC_COOLDOWN_THERMAL_MAX_C=42 \
  ELASTIC_MIN_BATTERY_PCT=20 \
  bash scripts/elastic/run_op12_gpu_full10min_difftree.sh
done

"$python_bin" runtime/plan/select_mixed_boundary_profile.py \
  --calibration-root "$calibration_root" \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0_with_output_shapes.weights_ops.json \
  --backend gpu \
  --low-budget-mib 3712 \
  --high-budget-mib 5504 \
  --low-fixed-ratio 60 \
  --high-fixed-ratio 90 \
  --output-profile "$selected_profile"

echo "calibration: $calibration_root"
echo "selected profile: $selected_profile"
