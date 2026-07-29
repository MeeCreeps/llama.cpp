#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

serial="${ELASTIC_ADB_SERIAL:-5ae7a43d}"
trace="${ELASTIC_TRACE_GLOB:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/plan_residency_pilot_op12_20260729/inputs/dynamic_budget_4nodes.csv}"
trace_sha256="$(sha256sum "$trace" | awk '{print $1}')"
root="${ELASTIC_ARTIFACT_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/plan_residency_pilot_op12_20260729}"
profile="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_cpu_pipeline_mixedboundary_20260729.json}"
cost_dir="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-cpu/Meta-Llama-3-8B-Instruct-Q4_0_op12_i8mm_20260728}"

for variant in \
    strict \
    reuse-stream \
    cache-managed; do
  policy="$variant"
  # The largest measured OP12 working unit is 15.75 MiB.  This is execution
  # capacity inside the same hard budget, not an optimization knob: without
  # it, a full keep set must be relaxed before the first streamed unit can
  # enter the pipeline.
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
  ELASTIC_CPU_THREADS=5 \
  ELASTIC_CPU_MASK=0xf8 \
  ELASTIC_PIPELINE_LOAD_CPU=0 \
  ELASTIC_PIPELINE_PREPARE_CPU=2 \
  ELASTIC_PIPELINE_GRAPH_LOOKAHEAD=1 \
  ELASTIC_WINDOW_SEC=120 \
  ELASTIC_WINDOW_START_SEC=0 \
  ELASTIC_BENCH_SECONDS=120 \
  ELASTIC_REPLAY_SPEEDUP=1 \
  ELASTIC_COOLDOWN_THERMAL_MAX_C=35 \
  ELASTIC_MIN_CPU_FREQ_LIMIT_KHZ=1400000 \
  ELASTIC_MIN_CPU_MEAN_FREQ_KHZ=1550000 \
  ELASTIC_MIN_CPU_MEDIAN_SAMPLE_MIN_KHZ=1500000 \
  ELASTIC_MIN_BATTERY_PCT=20 \
  bash scripts/elastic/run_op13_cpu_full10min_difftree.sh
done

python3 runtime/plan/summarize_plan_residency_pilot.py \
  --root "$root" \
  --backend cpu \
  --output-dir "$root/summary"
