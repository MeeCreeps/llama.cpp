#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5038}"

# Replays the unmodified prefix containing the first sharp budget decrease.
# This is long enough to reproduce the dynamic SOA residency failure without
# spending a full ten-minute measurement on a diagnostic binary.
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 172.20.173.218:5555 \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --model Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0_with_output_shapes.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01 \
  --trace-glob exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_pipeline_final_native_trace01_full10min_op13_20260723/inputs/dynamic_trace.csv \
  --methods mru \
  --execution-backend gpu \
  --artifact-root exp-results/Meta-Llama-3-8B-Instruct-Q4_0/gpu_mru_lifecycle_diagnostic_execpin_op13_20260728 \
  --window-sec 150 \
  --window-start-sec 0 \
  --replay-speedup 1 \
  --bench-seconds 150 \
  --bucket-mib 128 \
  --extra-max-budget-mib 0 \
  --kv-mib 512 \
  --misc-mib 256 \
  --safety-mib 283 \
  --pinned-extra-mib 0 \
  --release-stage-after-xform 1 \
  --time-limit-ms 250 \
  --prefetch-distance 1 \
  --transition-weight 0.1 \
  --diff-horizon-tokens 8 \
  --plan-up-step-buckets 1 \
  --plan-switch-min-gap-steps 16 \
  --online-plan-switch-min-gain-ms 0.1 \
  --disk-reload-multiplier 1 \
  --disk-gpu-reload-multiplier 4 \
  --overlap-model pipeline \
  --static-overlap-model pipeline \
  --cp-objective resource_makespan \
  --allowed-placements gpu,disk_gpu \
  --force-weight-placement output.weight=gpu \
  --granularity-policy diff-tree \
  --granularity-backend gpu \
  --granularity-profile runtime/plan/profiles/granularity/op13_llama3_8b_q4_gpu_pipeline_20260727.json \
  --granularity-horizon-tokens 8 \
  --granularity-min-gain-ms 0 \
  --granularity-max-edits 16 \
  --granularity-beam-width 128 \
  --granularity-pipeline-lookahead 1 \
  --granularity-pipeline-lookahead-mib 128 \
  --mru-allowed-placements gpu,disk_gpu \
  --threads 6 \
  --cpu-mask 0xfc \
  --cpu-strict 1 \
  --poll 50 \
  --fixed-performance-mode \
  --n-gpu-layers 99 \
  --prompt Hi \
  --show-token-output \
  --token-prefix-count 32 \
  --timeout-s 500 \
  --adb-timeout-s 30 \
  --adb-retries 2 \
  --require-device-idle \
  --device-idle-poll-s 5 \
  --device-idle-timeout-s 7200 \
  --cooldown-thermal-max-c 48 \
  --cooldown-thermal-status-max 0 \
  --cooldown-poll-s 10 \
  --cooldown-timeout-s 3600 \
  --port 18082 \
  --skip-push-binary \
  --skip-push-model \
  --extra-env GGML_ELASTIC_PIN=token_embd,output \
  --extra-env GGML_ELASTIC_GPU_UNIT_SYNC=none \
  --extra-env GGML_ELASTIC_EVICT_DEFER_REUSE=1 \
  --extra-env GGML_ELASTIC_CUT_DUAL_COMPUTE=fused \
  --extra-env GGML_ELASTIC_Q4_IMAGE_DEBUG=1
