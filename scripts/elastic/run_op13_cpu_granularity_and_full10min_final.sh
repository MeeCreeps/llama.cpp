#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5038}"

serial="${ELASTIC_ADB_SERIAL:-3C15AU002CL00000}"
fixed_root="${ELASTIC_FIXED_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_pipeline_fixed_30_90_budgetfair_i8mm_workerpin_screenoff_battery20_op13_20260728_r2}"
profile="runtime/plan/profiles/granularity/op13_llama3_8b_q4_cpu_pipeline_budgetfair_i8mm_20260728.json"
phase_profile="runtime/plan/profiles/granularity/op13_llama3_8b_q4_cpu_pipeline_budgetfair_i8mm_phase_only_20260728.json"

restore_device_state() {
  local port
  for port in 5038 5037; do
    if adb -P "$port" -s "$serial" get-state >/dev/null 2>&1; then
      adb -P "$port" -s "$serial" shell \
        'cmd power set-fixed-performance-mode-enabled false; svc wifi enable; svc data enable; settings put global stay_on_while_plugged_in 3' \
        >/dev/null 2>&1 || true
      return
    fi
  done
}
trap restore_device_state EXIT

python3 scripts/elastic/run_granularity_model_sweep.py \
  --serial "$serial" \
  --backend cpu \
  --remote-binary /data/local/tmp/hyzheng/elastic/llama-cli \
  --remote-model /data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --remote-lib-dir /data/local/tmp/hyzheng/elastic \
  --remote-work-dir /data/local/tmp/hyzheng/elastic/granularity_cpu_fixed_workerpin_screenoff_final \
  --output "$fixed_root" \
  --modes multi tensor cut \
  --ratios 30,40,50,60,70,80,90 \
  --weight-mib 4437.8 \
  --kv-mib 512 \
  --misc-mib 256 \
  --pin token_embd,output \
  --pipeline on \
  --pipeline-lookahead 1 \
  --pipeline-lookahead-mib 128 \
  --pipeline-graph-lookahead 1 \
  --pipeline-max-pending 256 \
  --pipeline-max-pending-mib 128 \
  --cpu-pipeline-staging pooled \
  --cpu-repack-threads 1 \
  --pipeline-load-cpu 0 \
  --pipeline-prepare-cpu 1 \
  --cut-parts 2 \
  --cut-dual-compute off \
  --threads 6 \
  --cpu-mask 0xfc \
  --cpu-strict 1 \
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
  --timeout 600 \
  --max-cpu-start-c 35 \
  --max-gpu-start-c 35 \
  --max-skin-start-c 31 \
  --thermal-poll-s 60 \
  --thermal-wait-timeout-s 43200 \
  --require-device-idle \
  --device-idle-poll-s 2 \
  --min-mem-available-mib 8000 \
  --min-battery-level-pct 20 \
  --sleep-while-waiting-for-battery \
  --min-cpu-freq-limit-khz 1900000 \
  --fixed-performance-mode \
  --resume

python3 runtime/plan/calibrate_granularity_profile.py \
  --runs-csv "$fixed_root/runs.csv" \
  --backend cpu \
  --output "$profile" \
  --phase-only-output "$phase_profile" \
  --planned-weight-mib 4154.9765625 \
  --unplanned-pinned-mib 281.8125 \
  --source OP13-Llama3-8B-Q4-CPU-pipeline-budgetfair-I8MM-workerpin-screenoff-20260728

ELASTIC_ADB_SERIAL="$serial" \
ELASTIC_GRANULARITY_PROFILE="$profile" \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op13_cpu_full10min_difftree.sh

ELASTIC_ADB_SERIAL="$serial" \
ELASTIC_GRANULARITY_PROFILE="$phase_profile" \
ELASTIC_MIN_BATTERY_PCT=20 \
bash scripts/elastic/run_op13_cpu_full10min_diff_before.sh
