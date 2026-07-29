#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5038}"

python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 172.20.173.218:5555 \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --model Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0_with_output_shapes.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01 \
  --trace-glob exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_pipeline_final_native_trace01_full10min_op13_20260723/inputs/dynamic_trace.csv \
  --methods diff-tree-mixed \
  --execution-backend cpu \
  --artifact-root exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_full10min_diffnow_edits16_multi_overhead_op13_20260727 \
  --window-sec 600 \
  --window-start-sec 0 \
  --replay-speedup 1 \
  --bench-seconds 600 \
  --bucket-mib 128 \
  --extra-max-budget-mib 0 \
  --kv-mib 512 \
  --misc-mib 256 \
  --safety-mib 283 \
  --pinned-extra-mib 0 \
  --release-stage-after-xform 0 \
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
  --allowed-placements cpu,disk_cpu \
  --force-weight-placement output.weight=cpu \
  --granularity-policy diff-tree \
  --granularity-backend cpu \
  --granularity-profile runtime/plan/profiles/granularity/op13_llama3_8b_q4_pipeline_20260727.json \
  --granularity-horizon-tokens 8 \
  --granularity-min-gain-ms 0 \
  --granularity-max-edits 16 \
  --granularity-beam-width 128 \
  --granularity-pipeline-lookahead 1 \
  --granularity-pipeline-lookahead-mib 128 \
  --mru-allowed-placements cpu,disk_cpu \
  --threads 6 \
  --cpu-mask 0xfc \
  --cpu-strict 1 \
  --poll 50 \
  --fixed-performance-mode \
  --n-gpu-layers 0 \
  --prompt Hi \
  --show-token-output \
  --token-prefix-count 32 \
  --timeout-s 1500 \
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
  --extra-env GGML_ELASTIC_UNIT_PIPELINE_GRAPH_LOOKAHEAD=1
