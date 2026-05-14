#!/usr/bin/env bash
# scripts/elastic/run_test3_tight.sh
#
# Test 3 (规范 §7)：极端紧 trace，M_floor 取值刚够 KV + 少数 block。
# 验证：合规率 1.0、token 与 baseline 一致、tok/s 在 B(t)~M_floor 时被 flash
# 带宽限速但不崩。
#
# 用法：./run_test3_tight.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEV="${DEVICE_DIR:-/data/local/tmp/elastic}"
MODEL="${MODEL_PATH:-/data/local/tmp/hyzheng/llama.cpp/exp-model/Llama-3.2-1B-f16.gguf}"
N_PREDICT="${N_PREDICT:-32}"
PROMPT="${PROMPT:-The story begins:}"
ADB="adb ${ADB_DEVICE:+-s $ADB_DEVICE}"

OUT="$ROOT/results/test3"
mkdir -p "$OUT"

# 极端紧 trace：B(t) 始终在 1000-1200 MB 窄带，远低于 model resident (~2357 MB)
# + kv (128) + misc (256)。M_floor = 1000 MB → target = 616 MB，能装 ~12 个
# block。整个 trace 全程都在流式模式。
CSV="$OUT/budget.csv"
cat > "$CSV" <<'EOF'
time_sec,budget_mb
0,1200
2,1000
4,1200
6,1000
60,1200
EOF
$ADB push "$CSV" "$DEV/test3_budget.csv" >/dev/null

echo "[test3] 跑 baseline (elastic=0) ..."
$ADB shell "cd '$DEV' && LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 ./llama-cli \
    -m '$MODEL' -p '$PROMPT' -n $N_PREDICT --seed 42 --temp 0 -ngl 99 -no-cnv --no-warmup" \
    > "$OUT/baseline.out.txt" 2> "$OUT/baseline.err.txt"

echo "[test3] 跑 elastic + 极端紧 trace ..."
$ADB shell "rm -f '$DEV/test3_metrics.jsonl'" >/dev/null 2>&1 || true
$ADB shell "cd '$DEV' && LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
    GGML_OPENCL_ELASTIC=1 \
    GGML_ELASTIC_BUDGET_CSV=./test3_budget.csv \
    GGML_ELASTIC_METRICS_JSONL=./test3_metrics.jsonl \
    ./llama-cli -m '$MODEL' -p '$PROMPT' -n $N_PREDICT --seed 42 --temp 0 -ngl 99 -no-cnv --no-warmup" \
    > "$OUT/elastic.out.txt" 2> "$OUT/elastic.err.txt"

$ADB pull "$DEV/test3_metrics.jsonl" "$OUT/" >/dev/null

echo "[test3] 对比 token 序列 ..."
if diff -q "$OUT/baseline.out.txt" "$OUT/elastic.out.txt" >/dev/null; then
    echo "  ✓ token 输出完全一致"
else
    echo "  ✗ token 输出不一致"
    diff "$OUT/baseline.out.txt" "$OUT/elastic.out.txt" | head -20
    exit 1
fi

echo "[test3] 计算 summary ..."
python3 "$ROOT/scripts/elastic/summary_from_jsonl.py" \
    --baseline-tokens-file "$OUT/baseline.out.txt" \
    --elastic-tokens-file  "$OUT/elastic.out.txt" \
    "$OUT/test3_metrics.jsonl" \
    > "$OUT/summary.json"
cat "$OUT/summary.json"
echo
echo "[test3] 全部产物在 $OUT/"
