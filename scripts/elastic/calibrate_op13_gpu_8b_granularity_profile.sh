#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python3 runtime/plan/calibrate_granularity_profile.py \
  --runs-csv exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_gpu_pipeline_calibration_8b_70_90_op13_20260727/runs.csv \
  --backend gpu \
  --output runtime/plan/profiles/granularity/op13_llama3_8b_q4_gpu_pipeline_20260727.json \
  --phase-only-output runtime/plan/profiles/granularity/op13_llama3_8b_q4_gpu_pipeline_phase_only_20260727.json \
  --source OP13-Llama3-8B-Q4-GPU-pipeline-residual-20260727
