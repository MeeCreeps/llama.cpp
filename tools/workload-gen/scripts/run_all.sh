#!/usr/bin/env bash
# Runs the M4 acceptance suite end-to-end:
#   for each scenario YAML in tools/workload-gen/configs/*.yaml:
#     1. gen_workload.py --config -> workloads/m4-<scenario>.json
#     2. adb push trace and (re)push the binary to /data/local/tmp/hyzheng/multi-lora/
#     3. adb shell run multi-lora-bench
#     4. adb pull csv + summary.json + output dump
#     5. summarize.py + plot_results.py
#
# Run from repo root:
#     ./tools/workload-gen/scripts/run_all.sh
#
# Configurable via env vars:
#     DEV               adb device serial (default: first device)
#     PY                python interpreter (default: /home/myid/hz85760/env/llama-py/bin/python)
#     SCENARIOS         space-separated list of scenarios to run (default: all .yaml files)
#     N_SLOTS           --n-slots passed to bench (default: 4)
#     MAX_MEM_MB        --max-adapter-mem-mb passed to bench (default: 0 = unbounded)
#     NGL               --n-gpu-layers (default: 99)
#     CTX               --n-ctx (default: 4096)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DEV="${DEV:-$(adb devices | awk 'NR==2 {print $1}')}"
PY="${PY:-/home/myid/hz85760/env/llama-py/bin/python}"
N_SLOTS="${N_SLOTS:-4}"
MAX_MEM_MB="${MAX_MEM_MB:-0}"
NGL="${NGL:-99}"
CTX="${CTX:-4096}"

DEV_ROOT=/data/local/tmp/hyzheng/multi-lora
HOST_OUT=results/M4

if [[ -z "$DEV" || "$DEV" == "List" ]]; then
    echo "no adb device attached" >&2
    exit 1
fi

if [[ -n "${SCENARIOS:-}" ]]; then
    YAMLS=()
    for s in $SCENARIOS; do
        YAMLS+=("tools/workload-gen/configs/${s}.yaml")
    done
else
    mapfile -t YAMLS < <(ls tools/workload-gen/configs/*.yaml)
fi

echo "[run_all] device=$DEV  scenarios=${#YAMLS[@]}  n_slots=$N_SLOTS  budget=${MAX_MEM_MB} MiB  ngl=$NGL  ctx=$CTX"

# Ensure binary on device is current.
adb -s "$DEV" push build-android/bin/multi-lora-bench "$DEV_ROOT/bin/" 2>&1 | tail -1
adb -s "$DEV" shell "mkdir -p $DEV_ROOT/workloads $DEV_ROOT/results/m4"

mkdir -p "$HOST_OUT"

for yaml in "${YAMLS[@]}"; do
    scenario=$(basename "$yaml" .yaml)
    echo
    echo "==================== $scenario ===================="

    # 1. Generate trace
    "$PY" tools/workload-gen/gen_workload.py --config "$yaml"

    trace_host="workloads/m4-${scenario}.json"
    trace_dev="$DEV_ROOT/workloads/m4-${scenario}.json"

    # 2. Push trace
    adb -s "$DEV" push "$trace_host" "$DEV_ROOT/workloads/" 2>&1 | tail -1

    # 3. Run bench on device
    csv_dev="$DEV_ROOT/results/m4/${scenario}.csv"
    out_dev="$DEV_ROOT/results/m4/out_${scenario}"
    adb -s "$DEV" shell "rm -rf $out_dev && mkdir -p $out_dev && cd $DEV_ROOT && \
        export LD_LIBRARY_PATH=$DEV_ROOT/lib:\$LD_LIBRARY_PATH && \
        time ./bin/multi-lora-bench \
            -m models/llama-3.2-3b-instruct-q4_0.gguf \
            -a models/lora \
            -w workloads/m4-${scenario}.json \
            -o results/m4/${scenario}.csv \
            --output-dir results/m4/out_${scenario} \
            --max-adapter-mem-mb $MAX_MEM_MB \
            --n-slots $N_SLOTS \
            -c $CTX -ngl $NGL --quiet 2>&1 | tail -2"

    # 4. Pull artefacts
    mkdir -p "$HOST_OUT/$scenario"
    adb -s "$DEV" pull "$csv_dev" "$HOST_OUT/$scenario/" 2>&1 | tail -1
    adb -s "$DEV" pull "${csv_dev}.summary.json" "$HOST_OUT/$scenario/" 2>&1 | tail -1
    adb -s "$DEV" pull "$out_dev" "$HOST_OUT/$scenario/output_dump" 2>&1 | tail -1

    # 5. Summarize + plot
    "$PY" tools/workload-gen/summarize.py "$HOST_OUT/$scenario/${scenario}.csv" \
        | tee "$HOST_OUT/$scenario/summary.txt"
    "$PY" tools/workload-gen/plot_results.py \
        "$HOST_OUT/$scenario/${scenario}.csv" \
        --out-dir "$HOST_OUT/$scenario/figures"
done

echo
echo "[run_all] done. artefacts under $HOST_OUT/"
