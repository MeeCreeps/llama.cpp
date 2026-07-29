#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

serial="${ELASTIC_ADB_SERIAL:-5ae7a43d}"
fixed_root="${ELASTIC_FIXED_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_pipeline_fixed_30_90_budgetfair_outputpin_multifused_op12_20260729_v21}"
profile="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_cpu_pipeline_budgetfair_outputpin_multifused_20260729_v21.json}"
phase_profile="${ELASTIC_PHASE_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_cpu_pipeline_budgetfair_outputpin_multifused_phase_only_20260729_v21.json}"
placement_cost_dir="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-cpu/Meta-Llama-3-8B-Instruct-Q4_0_op12_outputpin_multifused_20260729_v21}"
base_cost_dir="${ELASTIC_BASE_COST_DIR:-runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01}"
ratios="${ELASTIC_RATIOS:-30,40,50,60,70,80,90}"
modes="${ELASTIC_MODES:-multi,multi_fused,tensor,cut}"
n_predict="${ELASTIC_N_PREDICT:-14}"
warmup_tokens="${ELASTIC_WARMUP_TOKENS:-4}"
minimum_tokens="${ELASTIC_MIN_MEASURE_TOKENS:-9}"
skip_calibration="${ELASTIC_SKIP_CALIBRATION:-0}"
python_bin="${ELASTIC_PYTHON:-python3}"
expected_model_sha256="${ELASTIC_EXPECTED_MODEL_SHA256:-2b4675c2208f09ad8762d8cf1b6a4a26bf65e6f0641aba324ec65143c0b4ad9f}"
expected_binary_sha256="${ELASTIC_EXPECTED_BINARY_SHA256:-$(sha256sum build-android-llama/bin/llama-cli | awk '{print $1}')}"
IFS=',' read -r -a mode_values <<< "$modes"

# OP12 resource partition: CPU0 LOAD, CPU1 persistent pooled-copy helper,
# CPU2 PREPARE, CPUs3-7 common five-thread compute pool. All four measured
# implementations use the same partition and the same byte-window pipeline.
"$python_bin" scripts/elastic/run_granularity_model_sweep.py \
  --serial "$serial" \
  --backend cpu \
  --remote-binary /data/local/tmp/hyzheng/elastic/llama-cli \
  --remote-model /data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --expected-binary-sha256 "$expected_binary_sha256" \
  --expected-model-sha256 "$expected_model_sha256" \
  --remote-lib-dir /data/local/tmp/hyzheng/elastic \
  --remote-work-dir /data/local/tmp/hyzheng/elastic/granularity_cpu_fixed_op12_v21 \
  --output "$fixed_root" \
  --modes "${mode_values[@]}" \
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
  --cpu-pipeline-staging pooled \
  --cpu-repack-threads 1 \
  --pipeline-load-cpu 0 \
  --pipeline-copy-cpu 1 \
  --pipeline-prepare-cpu 2 \
  --pipeline-copy-min-kib 1024 \
  --multi-tensors 2 \
  --cut-parts 2 \
  --cut-dual-compute off \
  --threads 5 \
  --cpu-mask 0xf8 \
  --cpu-strict 1 \
  --poll 50 \
  --batch 32 \
  --ubatch 32 \
  --context 4096 \
  --n-predict "$n_predict" \
  --ignore-eos \
  --seed 42 \
  --prompt Hi \
  --repeats 1 \
  --fixed-warmup-decode-tokens "$warmup_tokens" \
  --fixed-min-measure-tokens "$minimum_tokens" \
  --timeout 900 \
  --max-cpu-start-c 35 \
  --max-gpu-start-c 35 \
  --max-skin-start-c 31 \
  --thermal-poll-s 10 \
  --thermal-wait-timeout-s 14400 \
  --require-device-idle \
  --device-idle-poll-s 2 \
  --min-mem-available-mib 4800 \
  --min-battery-level-pct 20 \
  --min-cpu-freq-limit-khz 1800000 \
  --fixed-performance-mode \
  --resume

if [[ "$skip_calibration" == "1" ]]; then
  "$python_bin" - "$fixed_root/runs.csv" "$modes" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
expected_modes = set(sys.argv[2].split(","))
by_mode = {row["mode"]: row for row in rows}
if set(by_mode) != expected_modes:
    raise SystemExit(
        f"CPU pin-contract smoke is incomplete: {sorted(by_mode)}")
for mode, row in by_mode.items():
    if row.get("valid_runtime", "").lower() != "true":
        raise SystemExit(
            f"{mode} failed runtime audit: "
            f"{row.get('runtime_invalid_reason')}")
    if row.get("valid_token_sequence", "").lower() != "true":
        raise SystemExit(f"{mode} token sequence mismatch")
    if int(row["pin_token_embd_inside_budget_count"]) < 1:
        raise SystemExit(f"{mode} did not pin token_embd inside the budget")
    if int(row["pin_output_inside_budget_count"]) < 1:
        raise SystemExit(f"{mode} did not pin output inside the budget")
    if int(row["pin_outside_budget_count"]) != 0:
        raise SystemExit(f"{mode} enlarged the weight budget through a pin")
# Performance is deliberately not a smoke-gate condition. The complete
# 30--90% matrix measures both coarse implementations and globally selects
# their lower paired geometric-mean latency. A one-round 30% comparison is
# too noisy to decide whether fusion is beneficial.
print("OP12 CPU pin/fusion-contract smoke passed", flush=True)
PY
  exit 0
fi

"$python_bin" runtime/plan/calibrate_granularity_profile.py \
  --runs-csv "$fixed_root/runs.csv" \
  --backend cpu \
  --output "$profile" \
  --phase-only-output "$phase_profile" \
  --placement-cost-base-dir "$base_cost_dir" \
  --placement-cost-output-dir "$placement_cost_dir" \
  --planned-weight-mib 4154.9765625 \
  --unplanned-pinned-mib 281.8125 \
  --required-ratios "$ratios" \
  --required-measured-modes "$modes" \
  --source OP12-Llama3-8B-Q4-CPU-pipeline-budgetfair-outputpin-multifused-20260729-v21

"$python_bin" runtime/plan/summarize_fixed_granularity_sweep.py \
  --runs-csv "$fixed_root/runs.csv" \
  --backend cpu \
  --output-dir "$fixed_root" \
  --expected-ratios "$ratios" \
  --expected-modes "$modes"

echo "fixed results: $fixed_root"
echo "Diff-now profile: $profile"
echo "Diff-before profile: $phase_profile"
echo "OP12 CPU placement costs: $placement_cost_dir"
