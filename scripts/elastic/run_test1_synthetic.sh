#!/usr/bin/env bash
# scripts/elastic/run_test1_synthetic.sh
#
# Test 1 (规范 §7)：合成 step-down B(t) trace，验证：
#   - t<10s: 缓存模式，正常速度
#   - t∈[10,20]: 流式模式，tok/s 掉下来
#   - t>20: 回到缓存
#   - 不崩；与 baseline token 完全一致
#
# 用法：./run_test1_synthetic.sh
#
# env 可覆盖：
#   ADB_DEVICE      ：默认空（adb 单设备自动选）
#   DEVICE_DIR      ：默认 /data/local/tmp/elastic
#   MODEL_PATH      ：默认 /data/local/tmp/hyzheng/llama.cpp/exp-model/Llama-3.2-1B-f16.gguf
#   N_PREDICT       ：默认 60
#   PROMPT          ：默认 "The story begins:"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEV="${DEVICE_DIR:-/data/local/tmp/elastic}"
MODEL="${MODEL_PATH:-/data/local/tmp/hyzheng/llama.cpp/exp-model/Llama-3.2-1B-f16.gguf}"
N_PREDICT="${N_PREDICT:-60}"
PROMPT="${PROMPT:-The story begins:}"
ADB="adb ${ADB_DEVICE:+-s $ADB_DEVICE}"

OUT="$ROOT/results/test1"
mkdir -p "$OUT"

CSV="$OUT/budget.csv"
# 短时间 step-down：avg infer ~50ms/tok 时 60 tok 跑 3 秒，trace 在前 2 秒内
# 完整经历 "缓存 → 流式 → 缓存" 三阶段，足以触发 evict。规范 §7 Test 1 原 CSV
# 用 0/10/20s 节点，但那是按慢速 trace 假设；我们这边 inference 太快撑不到，
# 改成 0/0.8/1.6 让 trace 在测试窗口内变化两次。
cat > "$CSV" <<EOF
time_sec,budget_mb
0,4000
0.8,1500
1.6,4000
60,4000
EOF
$ADB push "$CSV" "$DEV/test1_budget.csv" >/dev/null

# 基线（elastic=0）
echo "[test1] 跑 baseline (elastic=0) ..."
$ADB shell "cd '$DEV' && LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 ./llama-cli \
    -m '$MODEL' -p '$PROMPT' -n $N_PREDICT --seed 42 --temp 0 -ngl 99 -no-cnv --no-warmup" \
    > "$OUT/baseline.out.txt" 2> "$OUT/baseline.err.txt"

# Elastic（带 trace + metrics 写盘）
echo "[test1] 跑 elastic + B(t) trace ..."
$ADB shell "rm -f '$DEV/test1_metrics.jsonl'" >/dev/null 2>&1 || true
$ADB shell "cd '$DEV' && LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
    GGML_OPENCL_ELASTIC=1 \
    GGML_ELASTIC_BUDGET_CSV=./test1_budget.csv \
    GGML_ELASTIC_METRICS_JSONL=./test1_metrics.jsonl \
    ./llama-cli -m '$MODEL' -p '$PROMPT' -n $N_PREDICT --seed 42 --temp 0 -ngl 99 -no-cnv --no-warmup" \
    > "$OUT/elastic.out.txt" 2> "$OUT/elastic.err.txt"

$ADB pull "$DEV/test1_metrics.jsonl" "$OUT/" >/dev/null

# Token 一致性 diff
echo "[test1] 对比 token 序列 ..."
if diff -q "$OUT/baseline.out.txt" "$OUT/elastic.out.txt" >/dev/null; then
    echo "  ✓ token 输出完全一致"
else
    echo "  ✗ token 输出不一致 —— diff:"
    diff "$OUT/baseline.out.txt" "$OUT/elastic.out.txt" | head -20
    exit 1
fi

# Summary
echo "[test1] 计算 summary ..."
python3 "$ROOT/scripts/elastic/summary_from_jsonl.py" \
    --baseline-tokens-file "$OUT/baseline.out.txt" \
    --elastic-tokens-file  "$OUT/elastic.out.txt" \
    "$OUT/test1_metrics.jsonl" \
    > "$OUT/summary.json"
cat "$OUT/summary.json"
echo
echo "[test1] 全部产物在 $OUT/"
