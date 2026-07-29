#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5038}"

# Reproduce the pre-residual Diff-tree cost model on the same 10-minute
# absolute-budget trace and with the same runtime controls as Diff-now.
adb_serial="${ELASTIC_ADB_SERIAL:-172.20.173.218:5555}"
artifact_root="${ELASTIC_ARTIFACT_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_full10min_diff_before_phaseonly_op13_20260728_budgetfair_freqaudit_v2}"
granularity_profile="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op13_llama3_8b_q4_cpu_pipeline_budgetfair_i8mm_phase_only_20260728.json}"
cost_dir="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01}"
min_battery_pct="${ELASTIC_MIN_BATTERY_PCT:-20}"
threads="${ELASTIC_CPU_THREADS:-6}"
cpu_mask="${ELASTIC_CPU_MASK:-0xfc}"
pipeline_load_cpu="${ELASTIC_PIPELINE_LOAD_CPU:-0}"
pipeline_prepare_cpu="${ELASTIC_PIPELINE_PREPARE_CPU:-1}"
pipeline_copy_cpu="${ELASTIC_PIPELINE_COPY_CPU:--1}"
cooldown_thermal_max_c="${ELASTIC_COOLDOWN_THERMAL_MAX_C:-48}"
min_cpu_freq_limit_khz="${ELASTIC_MIN_CPU_FREQ_LIMIT_KHZ:-0}"
min_cpu_mean_freq_khz="${ELASTIC_MIN_CPU_MEAN_FREQ_KHZ:-0}"
min_cpu_median_sample_min_khz="${ELASTIC_MIN_CPU_MEDIAN_SAMPLE_MIN_KHZ:-0}"
bench_seconds="${ELASTIC_BENCH_SECONDS:-600}"
window_sec="${ELASTIC_WINDOW_SEC:-600}"
window_start_sec="${ELASTIC_WINDOW_START_SEC:-0}"
replay_speedup="${ELASTIC_REPLAY_SPEEDUP:-1}"
plan_residency_policy="${ELASTIC_PLAN_RESIDENCY_POLICY:-strict}"
planner_stream_reserve_mib="${ELASTIC_PLANNER_STREAM_RESERVE_MIB:-0}"
granularity_placement_source="${ELASTIC_GRANULARITY_PLACEMENT_SOURCE:-stateful-cp}"
python_bin="${ELASTIC_PYTHON:-python3}"
expected_model_sha256="${ELASTIC_EXPECTED_MODEL_SHA256:-2b4675c2208f09ad8762d8cf1b6a4a26bf65e6f0641aba324ec65143c0b4ad9f}"
expected_binary_sha256="${ELASTIC_EXPECTED_BINARY_SHA256:-$(sha256sum build-android-llama/bin/llama-cli | awk '{print $1}')}"
original_source_trace="${ELASTIC_ORIGINAL_SOURCE_TRACE:-.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_10min_baselines_full_timebased_op12_ctx4096/traces/trace_01_user_147_10min_x1.csv}"
expected_original_source_trace_sha256="${ELASTIC_EXPECTED_ORIGINAL_SOURCE_TRACE_SHA256:-ac69401a9089b7371921dcdfd8040c12909eb87eac25fa90986af6ccf2cc30ec}"
copy_env_args=()
source_provenance_args=()
if [[ -n "$original_source_trace" || -n "$expected_original_source_trace_sha256" ]]; then
  source_provenance_args+=(
    --original-source-trace "$original_source_trace"
    --expected-original-source-trace-sha256 \
      "$expected_original_source_trace_sha256"
  )
fi
if (( pipeline_copy_cpu >= 0 )); then
  copy_env_args+=(
    --extra-env GGML_ELASTIC_CPU_STAGE_COPY_PARALLEL=1
    --extra-env GGML_ELASTIC_CPU_STAGE_COPY_MIN_KB=1024
    --extra-env GGML_ELASTIC_PIPELINE_COPY_CPU="$pipeline_copy_cpu"
  )
fi

"$python_bin" runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial "$adb_serial" \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --model Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --expected-model-sha256 "$expected_model_sha256" \
  --expected-binary-sha256 "$expected_binary_sha256" \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0_with_output_shapes.weights_ops.json \
  --cost-dir "$cost_dir" \
  --trace-glob exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_pipeline_final_native_trace01_full10min_op13_20260723/inputs/dynamic_trace.csv \
  --expected-source-trace-sha256 95ebfa2c8ca60d9f1adac25a1856a40269011b7eb8c8bc75533df4cb6e1306f2 \
  "${source_provenance_args[@]}" \
  --methods diff-tree-mixed \
  --execution-backend cpu \
  --artifact-root "$artifact_root" \
  --window-sec "$window_sec" \
  --window-start-sec "$window_start_sec" \
  --replay-speedup "$replay_speedup" \
  --bench-seconds "$bench_seconds" \
  --bucket-mib 128 \
  --extra-max-budget-mib 0 \
  --kv-mib 512 \
  --misc-mib 256 \
  --safety-mib 283 \
  --pinned-extra-mib 282 \
  --expected-pin-token-embd-mib 281.8125 \
  --expected-pin-output-mib 410.9765625 \
  --expected-wbm-total-mib 4436.7890625 \
  --release-stage-after-xform 0 \
  --time-limit-ms 250 \
  --prefetch-distance 1 \
  --transition-weight 0.1 \
  --diff-horizon-tokens 8 \
  --plan-up-step-buckets 0 \
  --plan-switch-min-gap-steps 4 \
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
  --granularity-profile "$granularity_profile" \
  --granularity-placement-source "$granularity_placement_source" \
  --granularity-horizon-tokens 8 \
  --granularity-min-gain-ms 0 \
  --granularity-max-edits 128 \
  --granularity-beam-width 128 \
  --granularity-pipeline-lookahead 1 \
  --granularity-pipeline-lookahead-mib 128 \
  --plan-residency-policy "$plan_residency_policy" \
  --planner-stream-reserve-mib "$planner_stream_reserve_mib" \
  --mru-allowed-placements cpu,disk_cpu \
  --threads "$threads" \
  --cpu-mask "$cpu_mask" \
  --cpu-strict 1 \
  --poll 50 \
  --min-cpu-freq-limit-khz "$min_cpu_freq_limit_khz" \
  --min-cpu-mean-freq-khz "$min_cpu_mean_freq_khz" \
  --min-cpu-median-sample-min-khz \
    "$min_cpu_median_sample_min_khz" \
  --max-pipeline-budget-violations 0 \
  --max-plan-protection-relaxations 0 \
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
  --cooldown-thermal-max-c "$cooldown_thermal_max_c" \
  --cooldown-thermal-status-max 0 \
  --cooldown-poll-s 10 \
  --cooldown-timeout-s 14400 \
  --min-battery-level-pct "$min_battery_pct" \
  --port 18082 \
  --skip-push-binary \
  --skip-push-model \
  --extra-env GGML_ELASTIC_PIN=token_embd,output \
  --extra-env GGML_ELASTIC_PIPELINE_LOAD_CPU="$pipeline_load_cpu" \
  --extra-env GGML_ELASTIC_PIPELINE_PREPARE_CPU="$pipeline_prepare_cpu" \
  --extra-env GGML_ELASTIC_UNIT_PIPELINE_GRAPH_LOOKAHEAD=1 \
  "${copy_env_args[@]}"
