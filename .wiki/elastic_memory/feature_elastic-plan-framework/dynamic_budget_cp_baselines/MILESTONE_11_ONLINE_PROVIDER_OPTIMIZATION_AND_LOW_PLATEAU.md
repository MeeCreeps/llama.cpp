# Milestone 11: online provider optimization and low-budget plateau validation

Date: 2026-06-16
Device: `172.20.115.151:5555` (OnePlus 15 / Adreno 840)
Workspace on device: `/data/local/tmp/hyzheng/elastic`
Branch/commit under test: `feature/elastic-plan-framework` / `a10bd13ed`

## Why this milestone exists

Two issues showed up while continuing the phone baseline:

1. The online native solver was spending about 840 ms in `gen_ms` / `provider_get_ms` before each token because it rescanned JSON cost tables for every weight on every provider call.
2. The low-budget tail could hang when `GGML_ELASTIC_CALLBACK_NOCACHE=1` returned a fresh plan pointer for the same budget every token and `maybe_apply_plan()` repeatedly applied the same 1500 MiB staging plan.

Also, the correct BudgetWatcher variable is `GGML_ELASTIC_BUDGET_CSV`, not `GGML_ELASTIC_BUDGET_TRACE`.

## Fixes

### 1. Precompute online native item bases

`tools/main/main.cpp` now precomputes per-weight static fields when the online provider is initialized:

- weight id/name/layer/bytes/quant
- selected backend from OpenCL vs CPU_Elastic compute cost
- resident value from stage cost tables

`elastic_online_generate_native()` now queries only live residency state per token and reuses the precomputed cost metadata.

Observed effect on phone:

- before: `gen_ms` / `provider_get_ms` around 840 ms per online call
- after: `gen_ms` median about 4.9-5.0 ms, `provider_get_ms` median about 10 ms for changing budgets

### 2. Avoid shell `mkdir -p` on Android

`tools/main/main.cpp` now uses a local `mkdir_p_local()` helper instead of `std::system("mkdir -p ...")`.

Reason: Android system tools inherited the benchmark `LD_LIBRARY_PATH` and could fail with vendor library loader errors.

### 3. Debounce same-budget callback plans

`src/llama-context.cpp::maybe_apply_plan()` now skips provider generation and plan application when the current budget is already the applied plan's budget.

`GGML_ELASTIC_CALLBACK_APPLY_SAME_BUDGET=1` restores the old behavior for forced stress/reproduction.

## Trace files

All traces in this milestone are synthetic BudgetWatcher inputs, not real phone memory traces.

`budget_trace_fast.csv` exercises rapid interpolation over 1500/1800/2200/3000 MiB.

`budget_trace_decode_low_plateau.csv` was added to force decode into a stable 1500 MiB plateau:

```csv
t_ms,budget_mb
0,3000
1000,3000
1500,1500
8000,1500
9000,3000
11000,3000
12000,1500
600000,1500
```

## Commands

Build/deploy:

```bash
cmake --build build-android-llama --target llama-cli -j2
adb -s 172.20.115.151:5555 push build-android-llama/bin/llama-cli /data/local/tmp/hyzheng/elastic/llama-cli
adb -s 172.20.115.151:5555 shell 'chmod 755 /data/local/tmp/hyzheng/elastic/llama-cli'
```

Fast varying trace:

```bash
adb -s 172.20.115.151:5555 shell 'cd /data/local/tmp/hyzheng/elastic && timeout 120 sh -c '"'"'LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 GGML_OPENCL_DEVICE=0 GGML_OPENCL_DISABLE_HOST_PTR=1 GGML_OPENCL_SVM_DISABLE=1 GGML_ELASTIC_BUDGET_CSV=budget_trace_fast.csv GGML_ELASTIC_PROFILE_CSV=run_online_native_fasttrace_n32_after_debounce_csv_profile.csv LLAMA_ELASTIC_ONLINE=1 LLAMA_ELASTIC_ONLINE_MODE=native-greedy LLAMA_ELASTIC_MODEL_META=model_meta.json LLAMA_ELASTIC_COST_DIR=profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 LLAMA_ELASTIC_ONLINE_WORK_DIR=online_work_native_fast_debounce_csv ./llama-cli -m Llama-3.2-3B-Instruct-q4_0.gguf -p "Summarize dynamic elastic memory planning for mobile LLM inference." -n 32 -ngl 999 --ctx-size 512 --batch-size 64 --ubatch-size 64 --no-warmup --temp 0.0 -no-cnv'"'"' 2>&1 | tee run_online_native_fasttrace_n32_after_debounce_csv.log; echo EXIT:$?'
```

Low-budget plateau after provider-skip optimization:

```bash
adb -s 172.20.115.151:5555 shell 'cd /data/local/tmp/hyzheng/elastic && timeout 150 sh -c '"'"'LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 GGML_OPENCL_DEVICE=0 GGML_OPENCL_DISABLE_HOST_PTR=1 GGML_OPENCL_SVM_DISABLE=1 GGML_ELASTIC_BUDGET_CSV=budget_trace_decode_low_plateau.csv GGML_ELASTIC_PROFILE_CSV=run_online_native_lowplateau_n64_provider_skip_profile.csv LLAMA_ELASTIC_ONLINE=1 LLAMA_ELASTIC_ONLINE_MODE=native-greedy LLAMA_ELASTIC_MODEL_META=model_meta.json LLAMA_ELASTIC_COST_DIR=profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 LLAMA_ELASTIC_ONLINE_WORK_DIR=online_work_native_lowplateau_provider_skip ./llama-cli -m Llama-3.2-3B-Instruct-q4_0.gguf -p "Summarize dynamic elastic memory planning for mobile LLM inference." -n 64 -ngl 999 --ctx-size 512 --batch-size 64 --ubatch-size 64 --no-warmup --temp 0.0 -no-cnv'"'"' 2>&1 | tee run_online_native_lowplateau_n64_provider_skip.log; echo EXIT:$?'
```

## Results

### Fast varying trace, n32

- Exit: `0`
- Budget range observed by online provider: 1582-2976 MiB, 30 distinct budgets
- `gen_ms`: n=32, median=4.954 ms, mean=4.883 ms, min=1.784 ms, max=5.725 ms
- `provider_get_ms`: n=32, median=10.242 ms, mean=10.024 ms, min=3.628 ms, max=10.911 ms
- `apply_ms`: n=32, median=0.184 ms, mean=1.269 ms, max=35.240 ms
- Prompt eval: 18.40 ms/token
- Eval: 55.18 ms/token
- Total: 6319.61 ms / 45 tokens
- Online calls/failures: 32 / 0
- Anchor summary: requests=197, hits=111, duplicate=0, fired=0, failures=0

### Low 1500 MiB plateau before provider-skip, n64

- Exit: `0`
- Budget range: exactly 1500 MiB
- Online calls/failures: 64 / 0
- `gen_ms`: n=64, median=4.969 ms, mean=4.936 ms
- `provider_get_ms`: only one logged switch, 3.722 ms
- `apply_ms`: one applied switch, 23.172 ms
- Eval: 48.83 ms/token
- Total: 8320.85 ms / 77 tokens

### Low 1500 MiB plateau after provider-skip, n64

- Exit: `0`
- Budget range: exactly 1500 MiB
- Online calls/failures: 1 / 0
- `gen_ms`: n=1, 1.741 ms
- `provider_get_ms`: n=1, 3.643 ms
- `apply_ms`: n=1, 23.151 ms
- Prompt eval: 33.10 ms/token
- Eval: 49.33 ms/token
- Total: 7724.47 ms / 77 tokens
- Anchor summary: requests=12608, hits=111, duplicate=6993, fired=0, failures=0

## Caveats

- These traces are synthetic. Real phone memory trace collection is still a separate step.
- The low-plateau runs generated a timeline with load/transfer/xform events in the applied plan, but the runtime anchor summary reported `fired=0`. The profile CSV for this run contains `RELOAD_ENSURE` and `COMPUTE_GRAPH`, not direct disk read or stage load/transfer/xform rows. So this validates online provider/apply stability and overhead, but not actual stage I/O throughput.
- The phone has no `python` or `python3` in PATH, so the external CP-SAT baseline remains skipped on-device.

## Verification

- Android build: passed
- Deployed `llama-cli` to `/data/local/tmp/hyzheng/elastic`
- `git diff --check`: passed
- No lingering `llama-cli` process after runs
