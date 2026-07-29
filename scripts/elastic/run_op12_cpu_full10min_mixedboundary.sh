#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

profile="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_cpu_pipeline_mixedboundary_20260729.json}"
if [[ ! -s "$profile" ]]; then
  echo "selected mixed-boundary profile is missing: $profile" >&2
  exit 1
fi

ANDROID_ADB_SERVER_PORT="$ANDROID_ADB_SERVER_PORT" \
ELASTIC_ADB_SERIAL="${ELASTIC_ADB_SERIAL:-5ae7a43d}" \
ELASTIC_ARTIFACT_ROOT="${ELASTIC_ARTIFACT_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_full10min_mixedboundary_op12_20260729}" \
ELASTIC_METHODS="${ELASTIC_METHODS:-offline-mixed,online,diff-tree-mixed}" \
ELASTIC_GRANULARITY_PROFILE="$profile" \
ELASTIC_COST_DIR="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-cpu/Meta-Llama-3-8B-Instruct-Q4_0_op12_i8mm_20260728}" \
ELASTIC_CPU_THREADS=5 \
ELASTIC_CPU_MASK=0xf8 \
ELASTIC_PIPELINE_LOAD_CPU=0 \
ELASTIC_PIPELINE_PREPARE_CPU=2 \
ELASTIC_PIPELINE_GRAPH_LOOKAHEAD="${ELASTIC_PIPELINE_GRAPH_LOOKAHEAD:-1}" \
ELASTIC_WINDOW_SEC=600 \
ELASTIC_WINDOW_START_SEC=0 \
ELASTIC_BENCH_SECONDS=600 \
ELASTIC_REPLAY_SPEEDUP=1 \
ELASTIC_COOLDOWN_THERMAL_MAX_C=35 \
ELASTIC_MIN_CPU_FREQ_LIMIT_KHZ=0 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op13_cpu_full10min_difftree.sh
