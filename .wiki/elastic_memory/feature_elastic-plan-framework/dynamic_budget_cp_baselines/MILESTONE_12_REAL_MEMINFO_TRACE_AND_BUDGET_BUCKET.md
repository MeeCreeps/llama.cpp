# Milestone 12: real phone meminfo trace and budget bucket

Date: 2026-06-16
Device: `172.20.115.151:5555` (OnePlus 15 / Adreno 840)
Workspace on device: `/data/local/tmp/hyzheng/elastic`

## Goal

Run online native-greedy against the phone's real memory signal instead of a synthetic BudgetWatcher CSV.

This run does not set `GGML_ELASTIC_BUDGET_CSV`; `llama_context::elastic_budget_mib()` falls back to `/proc/meminfo` `MemAvailable`.

In parallel, a shell sampler writes the real phone memory trace:

```csv
t_ms,budget_mb
```

## Fix: optional budget bucket

Real `/proc/meminfo` fluctuates by small 1-10 MiB amounts during inference. That caused repeated online replanning even though the plan did not materially change.

`src/llama-context.cpp::maybe_apply_plan()` now supports:

```bash
GGML_ELASTIC_BUDGET_BUCKET_MB=64
```

When set, the runtime rounds the observed budget down to the nearest bucket before provider lookup. The default is `1`, so existing behavior is unchanged unless the env var is set.

## Commands

Real meminfo, no bucket:

```bash
adb -s 172.20.115.151:5555 shell 'cd /data/local/tmp/hyzheng/elastic && sh -c '"'"'echo t_ms,budget_mb > real_meminfo_trace_online_native_n64.csv; START=$(date +%s); (while true; do NOW=$(date +%s); T=$(( (NOW - START) * 1000 )); B=$(awk "/MemAvailable:/ {print int(\$2/1024)}" /proc/meminfo); echo ${T},${B}; sleep 0.1; done >> real_meminfo_trace_online_native_n64.csv) & SPID=$!; LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 GGML_OPENCL_DEVICE=0 GGML_OPENCL_DISABLE_HOST_PTR=1 GGML_OPENCL_SVM_DISABLE=1 GGML_ELASTIC_PROFILE_CSV=run_online_native_real_meminfo_n64_provider_skip_profile.csv LLAMA_ELASTIC_ONLINE=1 LLAMA_ELASTIC_ONLINE_MODE=native-greedy LLAMA_ELASTIC_MODEL_META=model_meta.json LLAMA_ELASTIC_COST_DIR=profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 LLAMA_ELASTIC_ONLINE_WORK_DIR=online_work_native_real_meminfo_provider_skip ./llama-cli -m Llama-3.2-3B-Instruct-q4_0.gguf -p "Summarize dynamic elastic memory planning for mobile LLM inference." -n 64 -ngl 999 --ctx-size 512 --batch-size 64 --ubatch-size 64 --no-warmup --temp 0.0 -no-cnv 2>&1 | tee run_online_native_real_meminfo_n64_provider_skip.log; RC=${PIPESTATUS:-0}; kill $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true; exit $RC'"'"''
```

Real meminfo, 64 MiB bucket:

```bash
adb -s 172.20.115.151:5555 shell 'cd /data/local/tmp/hyzheng/elastic && sh -c '"'"'echo t_ms,budget_mb > real_meminfo_trace_online_native_n64_bucket64.csv; START=$(date +%s); (while true; do NOW=$(date +%s); T=$(( (NOW - START) * 1000 )); B=$(awk "/MemAvailable:/ {print int(\$2/1024)}" /proc/meminfo); echo ${T},${B}; sleep 0.1; done >> real_meminfo_trace_online_native_n64_bucket64.csv) & SPID=$!; LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 GGML_OPENCL_DEVICE=0 GGML_OPENCL_DISABLE_HOST_PTR=1 GGML_OPENCL_SVM_DISABLE=1 GGML_ELASTIC_BUDGET_BUCKET_MB=64 GGML_ELASTIC_PROFILE_CSV=run_online_native_real_meminfo_n64_bucket64_profile.csv LLAMA_ELASTIC_ONLINE=1 LLAMA_ELASTIC_ONLINE_MODE=native-greedy LLAMA_ELASTIC_MODEL_META=model_meta.json LLAMA_ELASTIC_COST_DIR=profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 LLAMA_ELASTIC_ONLINE_WORK_DIR=online_work_native_real_meminfo_bucket64 ./llama-cli -m Llama-3.2-3B-Instruct-q4_0.gguf -p "Summarize dynamic elastic memory planning for mobile LLM inference." -n 64 -ngl 999 --ctx-size 512 --batch-size 64 --ubatch-size 64 --no-warmup --temp 0.0 -no-cnv 2>&1 | tee run_online_native_real_meminfo_n64_bucket64.log; RC=${PIPESTATUS:-0}; kill $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true; exit $RC'"'"''
```

## Results

### Real meminfo, no bucket, n64

- Exit: `0`
- Real trace samples: 199
- Real trace budget range: 7352-10091 MiB
- Provider budgets during decode: 7352-7370 MiB, 15 distinct budget values
- Online calls/failures: 23 / 0
- `gen_ms`: n=23, median=4.952 ms, mean=4.587 ms, min=1.786 ms, max=5.101 ms
- `provider_get_ms`: n=23, median=9.988 ms, mean=9.574 ms, min=3.631 ms, max=10.614 ms
- `apply_ms`: n=23, median=0.173 ms, mean=0.170 ms
- Prompt eval: 16.48 ms/token
- Eval: 48.92 ms/token
- Total: 7561.19 ms / 77 tokens
- Anchor summary: requests=985, hits=111, duplicate=444, fired=0, failures=0

### Real meminfo, 64 MiB bucket, n64

- Exit: `0`
- Real trace samples: 197
- Real trace budget range: 7297-10071 MiB
- Provider budget during decode: 7296 MiB, 1 distinct budget value
- Online calls/failures: 1 / 0
- `gen_ms`: n=1, 1.685 ms
- `provider_get_ms`: n=1, 3.545 ms
- `apply_ms`: n=1, 0.072 ms
- Prompt eval: 18.25 ms/token
- Eval: 49.31 ms/token
- Total: 7426.35 ms / 77 tokens
- Anchor summary: requests=12608, hits=111, duplicate=6993, fired=0, failures=0

## Notes

- The real meminfo run stayed far above the low-memory thresholds, so no load/transfer/xform staging was expected.
- The synthetic low-budget traces from Milestone 11 remain the better stress tests for budget switching under constrained memory.
- The real trace collection path is now working and artifacts are stored under `artifacts/`.
