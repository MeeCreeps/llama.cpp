#!/usr/bin/env bash
# Reproduce the 5-request vanilla-vs-ours demo from the README.
# Assumes the prerequisites in README.md (Setup section) are already in place:
#   - build-android/bin/multi-lora-bench cross-compiled for arm64-v8a
#   - base GGUF + 3 LoRA GGUFs pushed to /data/local/tmp/hyzheng/multi-lora/
#   - libomp.so pushed to /data/local/tmp/hyzheng/multi-lora/lib/
#   - tools/workload-gen/scripts/demo_vanilla.sh present on the device
# Run from repo root:
#     ./docs/multi-lora/quickstart/reproduce.sh
#
# Configurable env:
#   DEV     adb device serial (default: first connected)
#   DUMP    where to save artefacts (default: results/demo)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DEV="${DEV:-$(adb devices | awk 'NR==2 {print $1}')}"
DUMP="${DUMP:-results/demo}"
DEV_ROOT=/data/local/tmp/hyzheng/multi-lora

if [[ -z "$DEV" || "$DEV" == "List" ]]; then
    echo "no adb device attached" >&2; exit 1
fi

mkdir -p "$DUMP"

echo "device: $DEV"
echo

# Make sure trace + vanilla loop script are on device.
adb -s "$DEV" push "$REPO_ROOT/docs/multi-lora/quickstart/data/demo-5req.json" \
    "$DEV_ROOT/workloads/" 2>&1 | tail -1
adb -s "$DEV" push "$REPO_ROOT/tools/workload-gen/scripts/demo_vanilla.sh" \
    "$DEV_ROOT/" 2>&1 | tail -1
adb -s "$DEV" shell "chmod +x $DEV_ROOT/demo_vanilla.sh"

echo
echo "===== A. vanilla per-process loop (5 × llama-completion) ====="
adb -s "$DEV" shell "$DEV_ROOT/demo_vanilla.sh" 2>&1 | tail -3

echo
echo "===== B. ours (multi-lora-bench --n-slots=4) ====="
adb -s "$DEV" shell "
    rm -rf $DEV_ROOT/results/demo/ours_dump &&
    mkdir -p $DEV_ROOT/results/demo/ours_dump &&
    cd $DEV_ROOT &&
    export LD_LIBRARY_PATH=$DEV_ROOT/lib:\$LD_LIBRARY_PATH &&
    START=\$(date +%s%3N) &&
    ./bin/multi-lora-bench \
        -m models/llama-3.2-3b-instruct-q4_0.gguf \
        -a models/lora \
        -w workloads/demo-5req.json \
        -o results/demo/ours.csv \
        --output-dir results/demo/ours_dump \
        --max-adapter-mem-mb 0 --n-slots 4 -c 4096 -ngl 99 --quiet 2>&1 | tail -1 &&
    END=\$(date +%s%3N) &&
    echo \"OURS total ms: \$((END-START))\""

echo
echo "===== pull artefacts ====="
mkdir -p "$DUMP/vanilla" "$DUMP/ours_dump"
adb -s "$DEV" pull "$DEV_ROOT/results/demo/vanilla"   "$DUMP/" 2>&1 | tail -1
adb -s "$DEV" pull "$DEV_ROOT/results/demo/ours_dump" "$DUMP/" 2>&1 | tail -1
adb -s "$DEV" pull "$DEV_ROOT/results/demo/ours.csv"  "$DUMP/" 2>&1 | tail -1
adb -s "$DEV" pull "$DEV_ROOT/results/demo/ours.csv.summary.json" "$DUMP/" 2>&1 | tail -1

echo
echo "===== ours.csv ====="
column -t -s, "$DUMP/ours.csv"
echo
echo "===== outputs landed in $DUMP/ ====="
echo "  $DUMP/vanilla/        — 5 vanilla per-process outputs"
echo "  $DUMP/ours_dump/      — 5 multi-lora-bench detokenized outputs"
echo "  $DUMP/ours.csv*       — bench metrics"
