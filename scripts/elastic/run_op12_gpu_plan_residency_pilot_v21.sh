#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

serial="${ELASTIC_ADB_SERIAL:-5ae7a43d}"
trace="${ELASTIC_TRACE_GLOB:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/plan_residency_pilot_op12_20260729/inputs/dynamic_budget_4nodes.csv}"
trace_sha256="$(sha256sum "$trace" | awk '{print $1}')"
root="${ELASTIC_ARTIFACT_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/plan_residency_pilot_gpu_op12_20260729_v21}"
profile="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_budgetfair_outputpin_20260729_v21.json}"
cost_dir="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_op12_gpu_outputpin_20260729_v21}"

for variant in \
    strict \
    reuse-stream \
    cache-managed; do
  policy="$variant"
  # Reserve the measured maximum 15.75 MiB working-unit window inside the
  # unchanged physical budget for every policy.
  stream_reserve_mib=16
  methods="offline-mixed"
  if [[ "$variant" == "strict" ]]; then
    methods="mru,offline-mixed"
  fi
  ANDROID_ADB_SERVER_PORT="$ANDROID_ADB_SERVER_PORT" \
  ELASTIC_ADB_SERIAL="$serial" \
  ELASTIC_ARTIFACT_ROOT="$root/$variant" \
  ELASTIC_METHODS="$methods" \
  ELASTIC_TRACE_GLOB="$trace" \
  ELASTIC_EXPECTED_SOURCE_TRACE_SHA256="$trace_sha256" \
  ELASTIC_GRANULARITY_PROFILE="$profile" \
  ELASTIC_COST_DIR="$cost_dir" \
  ELASTIC_PLAN_RESIDENCY_POLICY="$policy" \
  ELASTIC_PLANNER_STREAM_RESERVE_MIB="$stream_reserve_mib" \
  ELASTIC_PIPELINE_GRAPH_LOOKAHEAD=1 \
  ELASTIC_WINDOW_SEC=120 \
  ELASTIC_WINDOW_START_SEC=0 \
  ELASTIC_BENCH_SECONDS=120 \
  ELASTIC_REPLAY_SPEEDUP=1 \
  ELASTIC_COOLDOWN_THERMAL_MAX_C=42 \
  ELASTIC_MIN_BATTERY_PCT=20 \
  bash scripts/elastic/run_op12_gpu_full10min_difftree.sh
done

python3 runtime/plan/summarize_plan_residency_pilot.py \
  --root "$root" \
  --backend gpu \
  --output-dir "$root/summary"
