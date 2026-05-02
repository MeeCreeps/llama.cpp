#!/system/bin/sh
# Run on device; loops a fixed 5-request demo through standalone
# llama-completion invocations (vanilla per-process baseline).
set -eu
cd /data/local/tmp/hyzheng/multi-lora
export LD_LIBRARY_PATH=/data/local/tmp/hyzheng/multi-lora/lib:${LD_LIBRARY_PATH:-}

OUT=results/demo/vanilla
rm -rf "$OUT" && mkdir -p "$OUT"

run_one() {
    local idx="$1" adapter="$2" prompt="$3"
    ./bin/llama-completion \
        -m models/llama-3.2-3b-instruct-q4_0.gguf \
        --lora "models/lora/${adapter}.gguf" \
        -p "$prompt" \
        -n 24 --temp 0 --seed 42 --no-warmup -c 4096 -ngl 99 \
        2>/dev/null > "$OUT/req_${idx}_${adapter}.out"
}

START=$(date +%s%3N)
run_one 00000 reasoning "A car travels 60 miles in 1 hour. How far does it travel in 2.5 hours? Show the math."
run_one 00001 hebrew    "Write a one-sentence inspirational quote about perseverance."
run_one 00002 summary   "In one paragraph, summarize what photosynthesis is."
run_one 00003 reasoning "If a rectangle has length 8 and width 5, what is its area and perimeter?"
run_one 00004 reasoning "Convert 100 degrees Fahrenheit to Celsius. Show the formula."
END=$(date +%s%3N)
echo "VANILLA total ms: $((END-START))"
