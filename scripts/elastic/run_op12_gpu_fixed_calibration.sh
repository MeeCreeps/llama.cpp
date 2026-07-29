#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

serial="${ELASTIC_ADB_SERIAL:-5ae7a43d}"
fixed_root="${ELASTIC_FIXED_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_gpu_pipeline_fixed_30_90_budgetfair_cutdual_op12_20260728}"
profile="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_budgetfair_20260728.json}"
phase_profile="${ELASTIC_PHASE_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_budgetfair_phase_only_20260728.json}"
base_cost_dir="${ELASTIC_BASE_COST_DIR:-runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01}"
placement_cost_dir="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_op12_gpu_20260728}"
ratios="${ELASTIC_RATIOS:-30,40,50,60,70,80,90}"
source_name="${ELASTIC_SOURCE:-OP12-Llama3-8B-Q4-GPU-pipeline-budgetfair-cutdual-20260728}"
python_bin="${ELASTIC_PYTHON:-python3}"
expected_model_sha256="${ELASTIC_EXPECTED_MODEL_SHA256:-2b4675c2208f09ad8762d8cf1b6a4a26bf65e6f0641aba324ec65143c0b4ad9f}"
expected_binary_sha256="${ELASTIC_EXPECTED_BINARY_SHA256:-$(sha256sum build-android-llama/bin/llama-cli | awk '{print $1}')}"

# The model contains 4436.7890625 MiB of physical WBM weights. The
# planner-visible
# Metadata covers 4154.9765625 MiB, including output.weight, while token_embd
# contributes the remaining 281.8125 MiB as an inside-budget WBM pin. All
# three modes therefore receive the same physical weight target at every
# ratio.
"$python_bin" scripts/elastic/run_granularity_model_sweep.py \
  --serial "$serial" \
  --backend gpu \
  --remote-binary /data/local/tmp/hyzheng/elastic/llama-cli \
  --remote-model /data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --expected-binary-sha256 "$expected_binary_sha256" \
  --expected-model-sha256 "$expected_model_sha256" \
  --remote-lib-dir /data/local/tmp/hyzheng/elastic \
  --remote-work-dir /data/local/tmp/hyzheng/elastic/granularity_gpu_fixed_op12 \
  --output "$fixed_root" \
  --modes multi tensor cut \
  --ratios "$ratios" \
  --weight-mib 4436.7890625 \
  --kv-mib 512 \
  --misc-mib 256 \
  --pin token_embd,output \
  --expected-pin-token-embd-mib 281.8125 \
  --expected-pin-output-mib 410.9765625 \
  --pipeline on \
  --pipeline-lookahead 1 \
  --pipeline-lookahead-mib 128 \
  --pipeline-graph-lookahead 1 \
  --pipeline-max-pending 256 \
  --pipeline-max-pending-mib 128 \
  --multi-tensors 2 \
  --cut-parts 2 \
  --cut-dual-compute fused \
  --gpu-unit-sync none \
  --threads 5 \
  --cpu-mask 0xf8 \
  --cpu-strict 1 \
  --poll 50 \
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
  --timeout 900 \
  --max-cpu-start-c 42 \
  --max-gpu-start-c 42 \
  --max-skin-start-c 35 \
  --thermal-poll-s 10 \
  --thermal-wait-timeout-s 14400 \
  --require-device-idle \
  --device-idle-poll-s 5 \
  --min-mem-available-mib 4800 \
  --min-battery-level-pct 20 \
  --fixed-performance-mode \
  --resume

"$python_bin" runtime/plan/calibrate_granularity_profile.py \
  --runs-csv "$fixed_root/runs.csv" \
  --backend gpu \
  --output "$profile" \
  --phase-only-output "$phase_profile" \
  --placement-cost-base-dir "$base_cost_dir" \
  --placement-cost-output-dir "$placement_cost_dir" \
  --planned-weight-mib 4154.9765625 \
  --unplanned-pinned-mib 281.8125 \
  --required-ratios "$ratios" \
  --required-measured-modes multi,tensor,cut \
  --source "$source_name"

"$python_bin" runtime/plan/summarize_fixed_granularity_sweep.py \
  --runs-csv "$fixed_root/runs.csv" \
  --backend gpu \
  --output-dir "$fixed_root" \
  --expected-ratios "$ratios" \
  --expected-modes multi,tensor,cut

echo "fixed results: $fixed_root"
echo "Diff-now profile: $profile"
echo "Diff-before profile: $phase_profile"
echo "OP12 GPU placement costs: $placement_cost_dir"
