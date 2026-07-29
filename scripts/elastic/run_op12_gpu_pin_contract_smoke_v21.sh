#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export ANDROID_ADB_SERVER_PORT="${ANDROID_ADB_SERVER_PORT:-5037}"

serial="${ELASTIC_ADB_SERIAL:-5ae7a43d}"
output="${ELASTIC_OUTPUT:-exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_gpu_outputpin_contract_smoke_op12_20260729_v21}"
python_bin="${ELASTIC_PYTHON:-python3}"
expected_model_sha256="${ELASTIC_EXPECTED_MODEL_SHA256:-2b4675c2208f09ad8762d8cf1b6a4a26bf65e6f0641aba324ec65143c0b4ad9f}"
expected_binary_sha256="${ELASTIC_EXPECTED_BINARY_SHA256:-$(sha256sum build-android-llama/bin/llama-cli | awk '{print $1}')}"

"$python_bin" scripts/elastic/run_granularity_model_sweep.py \
  --serial "$serial" \
  --backend gpu \
  --remote-binary /data/local/tmp/hyzheng/elastic/llama-cli \
  --remote-model /data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --expected-binary-sha256 "$expected_binary_sha256" \
  --expected-model-sha256 "$expected_model_sha256" \
  --remote-lib-dir /data/local/tmp/hyzheng/elastic \
  --remote-work-dir /data/local/tmp/hyzheng/elastic/granularity_gpu_pin_smoke_op12 \
  --output "$output" \
  --modes multi tensor cut \
  --ratios 30 \
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
  --n-predict 6 \
  --ignore-eos \
  --seed 42 \
  --prompt Hi \
  --repeats 1 \
  --fixed-warmup-decode-tokens 2 \
  --fixed-min-measure-tokens 3 \
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
  --fixed-performance-mode

"$python_bin" - "$output/runs.csv" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
expected_modes = {"multi", "tensor", "cut"}
by_mode = {row["mode"]: row for row in rows}
if set(by_mode) != expected_modes:
    raise SystemExit(
        f"pin-contract smoke is incomplete: {sorted(by_mode)}")
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
    if int(row["pipeline_budget_violations"]) != 0:
        raise SystemExit(f"{mode} exceeded the weight budget")
print("OP12 GPU pin-contract smoke passed", flush=True)
PY

echo "pin-contract smoke: $output"
