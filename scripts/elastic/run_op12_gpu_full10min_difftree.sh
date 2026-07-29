#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

serial="${ELASTIC_ADB_SERIAL:-5ae7a43d}"
methods="${ELASTIC_METHODS:-static-min,static-max,mru,offline-mixed,online,diff-tree-mixed}"
artifact_root="${ELASTIC_ARTIFACT_ROOT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_gpu_native_full10min_all_baselines_op12_20260728_budgetfair}"
profile="${ELASTIC_GRANULARITY_PROFILE:-runtime/plan/profiles/granularity/op12_llama3_8b_q4_gpu_pipeline_budgetfair_20260728.json}"
cost_dir="${ELASTIC_COST_DIR:-runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_op12_gpu_20260728}"
bench_seconds="${ELASTIC_BENCH_SECONDS:-600}"
window_sec="${ELASTIC_WINDOW_SEC:-600}"
window_start_sec="${ELASTIC_WINDOW_START_SEC:-0}"
replay_speedup="${ELASTIC_REPLAY_SPEEDUP:-1}"
trace_glob="${ELASTIC_TRACE_GLOB:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_pipeline_final_native_trace01_full10min_op13_20260723/inputs/dynamic_trace.csv}"
expected_source_trace_sha256="${ELASTIC_EXPECTED_SOURCE_TRACE_SHA256:-95ebfa2c8ca60d9f1adac25a1856a40269011b7eb8c8bc75533df4cb6e1306f2}"
original_source_trace="${ELASTIC_ORIGINAL_SOURCE_TRACE:-}"
expected_original_source_trace_sha256="${ELASTIC_EXPECTED_ORIGINAL_SOURCE_TRACE_SHA256:-}"
if [[ -z "$original_source_trace" && -z "$expected_original_source_trace_sha256" \
      && "$expected_source_trace_sha256" == "95ebfa2c8ca60d9f1adac25a1856a40269011b7eb8c8bc75533df4cb6e1306f2" \
      && "$trace_glob" == "exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_pipeline_final_native_trace01_full10min_op13_20260723/inputs/dynamic_trace.csv" ]]; then
  original_source_trace=".wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_10min_baselines_full_timebased_op12_ctx4096/traces/trace_01_user_147_10min_x1.csv"
  expected_original_source_trace_sha256="ac69401a9089b7371921dcdfd8040c12909eb87eac25fa90986af6ccf2cc30ec"
fi
granularity_max_edits="${ELASTIC_GRANULARITY_MAX_EDITS:-16}"
plan_switch_min_gap_steps="${ELASTIC_PLAN_SWITCH_MIN_GAP_STEPS:-4}"
cooldown_thermal_max_c="${ELASTIC_COOLDOWN_THERMAL_MAX_C:-42}"
pipeline_graph_lookahead="${ELASTIC_PIPELINE_GRAPH_LOOKAHEAD:-1}"
plan_residency_policy="${ELASTIC_PLAN_RESIDENCY_POLICY:-strict}"
planner_stream_reserve_mib="${ELASTIC_PLANNER_STREAM_RESERVE_MIB:-0}"
granularity_placement_source="${ELASTIC_GRANULARITY_PLACEMENT_SOURCE:-stateful-cp}"
min_battery_pct="${ELASTIC_MIN_BATTERY_PCT:-20}"
python_bin="${ELASTIC_PYTHON:-python3}"
expected_model_sha256="${ELASTIC_EXPECTED_MODEL_SHA256:-2b4675c2208f09ad8762d8cf1b6a4a26bf65e6f0641aba324ec65143c0b4ad9f}"
expected_binary_sha256="${ELASTIC_EXPECTED_BINARY_SHA256:-$(sha256sum build-android-llama/bin/llama-cli | awk '{print $1}')}"
source_provenance_args=()
if [[ -n "$original_source_trace" || -n "$expected_original_source_trace_sha256" ]]; then
  source_provenance_args+=(
    --original-source-trace "$original_source_trace"
    --expected-original-source-trace-sha256 \
      "$expected_original_source_trace_sha256"
  )
fi

"$python_bin" runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial "$serial" \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --model Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --expected-model-sha256 "$expected_model_sha256" \
  --expected-binary-sha256 "$expected_binary_sha256" \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0_with_output_shapes.weights_ops.json \
  --cost-dir "$cost_dir" \
  --trace-glob "$trace_glob" \
  --expected-source-trace-sha256 "$expected_source_trace_sha256" \
  "${source_provenance_args[@]}" \
  --methods "$methods" \
  --execution-backend gpu \
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
  --release-stage-after-xform 1 \
  --time-limit-ms 250 \
  --prefetch-distance 1 \
  --transition-weight 0.1 \
  --diff-horizon-tokens 8 \
  --plan-up-step-buckets 0 \
  --plan-switch-min-gap-steps "$plan_switch_min_gap_steps" \
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
  --granularity-profile "$profile" \
  --granularity-placement-source "$granularity_placement_source" \
  --granularity-horizon-tokens 8 \
  --granularity-min-gain-ms 0 \
  --granularity-max-edits "$granularity_max_edits" \
  --granularity-beam-width 128 \
  --granularity-pipeline-lookahead 1 \
  --granularity-pipeline-lookahead-mib 128 \
  --plan-residency-policy "$plan_residency_policy" \
  --planner-stream-reserve-mib "$planner_stream_reserve_mib" \
  --mru-allowed-placements gpu,disk_gpu \
  --threads 5 \
  --cpu-mask 0xf8 \
  --cpu-strict 1 \
  --poll 50 \
  --max-pipeline-budget-violations 0 \
  --max-plan-protection-relaxations 0 \
  --fixed-performance-mode \
  --n-gpu-layers 99 \
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
  --extra-env GGML_ELASTIC_GPU_UNIT_SYNC=none \
  --extra-env GGML_ELASTIC_UNIT_PIPELINE_GRAPH_LOOKAHEAD="$pipeline_graph_lookahead" \
  --extra-env GGML_ELASTIC_EVICT_DEFER_REUSE=1 \
  --extra-env GGML_ELASTIC_CUT_DUAL_COMPUTE=fused
