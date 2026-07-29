#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5038}"

# The native 10-minute trace covers effective 70%/80%/90% weight-residency
# bins. Calibrate all three optimized GPU implementations only on those bins;
# the dynamic trace remains held out.
python3 scripts/elastic/run_granularity_model_sweep.py \
  --serial 172.20.173.218:5555 \
  --backend gpu \
  --remote-binary /data/local/tmp/hyzheng/elastic/llama-cli \
  --remote-model /data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --remote-lib-dir /data/local/tmp/hyzheng/elastic \
  --remote-work-dir /data/local/tmp/hyzheng/elastic/granularity_llama8_gpu_cal \
  --output exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_gpu_pipeline_calibration_8b_70_90_op13_20260727 \
  --weight-mib 4154.98 \
  --ratios 70,80,90 \
  --repeats 1 \
  --modes multi tensor cut \
  --n-predict -1 \
  --fixed-warmup-decode-tokens 4 \
  --fixed-min-measure-tokens 16 \
  --duration-s 25 \
  --threads 6 \
  --cpu-mask 0xfc \
  --cpu-strict 1 \
  --poll 50 \
  --context 4096 \
  --batch 32 \
  --ubatch 32 \
  --prompt Hi \
  --seed 42 \
  --kv-mib 512 \
  --misc-mib 256 \
  --pin token_embd,output \
  --multi-tensors 2 \
  --cut-parts 2 \
  --cut-dual-compute fused \
  --gpu-unit-sync none \
  --pipeline on \
  --pipeline-lookahead 1 \
  --pipeline-lookahead-mib 128 \
  --pipeline-graph-lookahead 4 \
  --pipeline-max-pending 256 \
  --pipeline-max-pending-mib 128 \
  --timeout 600 \
  --max-cpu-start-c 48 \
  --max-gpu-start-c 48 \
  --max-skin-start-c 39 \
  --thermal-poll-s 5 \
  --thermal-wait-timeout-s 1800 \
  --require-device-idle \
  --device-idle-poll-s 5 \
  --fixed-performance-mode \
  --keep-device-awake
