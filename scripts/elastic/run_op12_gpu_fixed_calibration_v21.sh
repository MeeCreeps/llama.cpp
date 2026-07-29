#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

ELASTIC_ADB_SERIAL="${ELASTIC_ADB_SERIAL:-5ae7a43d}" \
ELASTIC_FIXED_ROOT="${ELASTIC_FIXED_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_gpu_pipeline_fixed_30_90_budgetfair_cutdual_outputpin_op12_20260729_v21}" \
ELASTIC_GRANULARITY_PROFILE="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_budgetfair_outputpin_20260729_v21.json}" \
ELASTIC_PHASE_PROFILE="${ELASTIC_PHASE_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_budgetfair_outputpin_phase_only_20260729_v21.json}" \
ELASTIC_COST_DIR="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_op12_gpu_outputpin_20260729_v21}" \
ELASTIC_SOURCE="${ELASTIC_SOURCE:-OP12-Llama3-8B-Q4-GPU-pipeline-budgetfair-cutdual-outputpin-20260729-v21}" \
bash scripts/elastic/run_op12_gpu_fixed_calibration.sh
