#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

profile="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_mixedboundary_20260729_v19.json}"
artifact_root="${ELASTIC_ARTIFACT_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_gpu_native_full10min_mixedboundary_op12_20260729_v19}"
if [[ ! -s "$profile" ]]; then
  echo "selected GPU mixed-boundary profile is missing: $profile" >&2
  exit 1
fi

ANDROID_ADB_SERVER_PORT="$ANDROID_ADB_SERVER_PORT" \
ELASTIC_ADB_SERIAL="${ELASTIC_ADB_SERIAL:-5ae7a43d}" \
ELASTIC_ARTIFACT_ROOT="$artifact_root" \
ELASTIC_METHODS="${ELASTIC_METHODS:-static-min,static-max,mru,offline-mixed,online,diff-tree-mixed}" \
ELASTIC_GRANULARITY_PROFILE="$profile" \
ELASTIC_COST_DIR="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_op12_gpu_20260729_v19}" \
ELASTIC_GRANULARITY_MAX_EDITS=16 \
ELASTIC_PLAN_SWITCH_MIN_GAP_STEPS=4 \
ELASTIC_PIPELINE_GRAPH_LOOKAHEAD="${ELASTIC_PIPELINE_GRAPH_LOOKAHEAD:-1}" \
ELASTIC_WINDOW_SEC=600 \
ELASTIC_WINDOW_START_SEC=0 \
ELASTIC_BENCH_SECONDS=600 \
ELASTIC_REPLAY_SPEEDUP=1 \
ELASTIC_COOLDOWN_THERMAL_MAX_C=42 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op12_gpu_full10min_difftree.sh

python3 runtime/plan/summarize_granularity_frontier_trace.py \
  --trace "$artifact_root/traces/dynamic_trace_10min_x1.csv" \
  --plan-dir "$artifact_root/offline_mixed_table" \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0_with_output_shapes.weights_ops.json \
  --profile "$profile" \
  --backend gpu \
  --output-dir "$artifact_root/analysis"
