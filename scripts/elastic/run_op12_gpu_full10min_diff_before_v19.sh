#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

ANDROID_ADB_SERVER_PORT="$ANDROID_ADB_SERVER_PORT" \
ELASTIC_ADB_SERIAL="${ELASTIC_ADB_SERIAL:-5ae7a43d}" \
ELASTIC_ARTIFACT_ROOT="${ELASTIC_ARTIFACT_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_gpu_native_full10min_diff_before_op12_20260729_v19}" \
ELASTIC_METHODS=diff-tree-mixed \
ELASTIC_GRANULARITY_PROFILE="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_budgetfair_phase_only_20260729_v19.json}" \
ELASTIC_COST_DIR="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_op12_gpu_20260729_v19}" \
ELASTIC_GRANULARITY_MAX_EDITS=16 \
ELASTIC_PLAN_SWITCH_MIN_GAP_STEPS=4 \
ELASTIC_PIPELINE_GRAPH_LOOKAHEAD=1 \
ELASTIC_WINDOW_SEC=600 \
ELASTIC_WINDOW_START_SEC=0 \
ELASTIC_BENCH_SECONDS=600 \
ELASTIC_REPLAY_SPEEDUP=1 \
ELASTIC_COOLDOWN_THERMAL_MAX_C=42 \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op12_gpu_full10min_difftree.sh
