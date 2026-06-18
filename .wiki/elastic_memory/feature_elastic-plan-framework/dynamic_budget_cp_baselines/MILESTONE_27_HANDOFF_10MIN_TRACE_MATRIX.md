# Milestone 27: Handoff for 10-Min Dynamic Trace Matrix

Date: 2026-06-18

## Current Goal

Run the 8B model on OP13 with real dynamic memory traces from `trace/traces_9g`.
Each trace should use a 10-minute dynamic window, not a fixed token count.

Baselines:

- `offline`: precomputed budget table, selected by current budget bucket.
- `online`: remote CP-SAT solver on the host, current weight residency sent by phone.
- `mru`: native MRU migration baseline.
- `static-min`: fixed plan for the minimum memory budget bucket in the trace window.

Reported online decode speed should exclude remote plan query/solve wall time:

```text
online_decode_ms_per_token = (eval_ms_total - remote_wall_ms) / eval_runs
```

## Implemented Since Previous Runs

### Time-based llama-cli benchmark mode

`tools/main/main.cpp` now supports:

```bash
LLAMA_ELASTIC_BENCH_SECONDS=600 ./llama-cli ... -n -1
```

The timer starts at the first generated token. It stops after the requested decode duration and still emits the normal llama.cpp perf footer.

This is required because `-n 64` was not testing trace dynamics; it only generated a fixed number of tokens.

### Matrix runner

New script:

```text
runtime/plan/run_dynamic_budget_matrix.py
```

Main behavior:

- Selects low-memory 10-minute windows from raw `trace/traces_9g/trace_XX_user_YY.csv`.
- Builds/pushes offline budget table.
- Pushes model meta, cost model, and trace CSVs.
- Runs `offline`, `online`, `mru`, and `static-min`.
- Writes:
  - `summary/results.csv`
  - `summary/SUMMARY.md`
  - per-run logs in `logs/`
- Supports resume by default; use `--no-resume` to rerun everything.
- Adds adb timeout/retry for short shell setup commands to avoid hanging forever.

## Last Known Phone State

The TCP adb device `172.20.115.151:5555` became unhealthy:

```text
adb devices showed it as offline after reconnect.
adb shell commands timed out before that.
```

USB serial was healthy and identified as OP13:

```text
serial: 5ae7a43d
model:  CPH2583
device: OP595DL1
```

Use USB serial if available:

```bash
adb -s 5ae7a43d shell 'echo ok; getprop ro.product.model; getprop ro.product.vendor.device'
```

Target deploy directory remains:

```text
/data/local/tmp/hyzheng/elastic
```

## Important Deployment Fix

The Android build of `llama-cli` needs `libomp.so` in `LD_LIBRARY_PATH`.
The last full matrix attempt failed immediately because the new deploy directory did not contain it:

```text
CANNOT LINK EXECUTABLE "./llama-cli": library "libomp.so" not found
```

The current `run_dynamic_budget_matrix.py` can push this automatically.
If deploying manually, push the aarch64 OpenMP runtime before running:

```bash
adb -s 5ae7a43d push \
  /home/myid/hz85760/env/android-sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so \
  /data/local/tmp/hyzheng/elastic/libomp.so
```

Then verify:

```bash
adb -s 5ae7a43d shell \
  'cd /data/local/tmp/hyzheng/elastic && LD_LIBRARY_PATH=. ./llama-cli --help >/dev/null && echo llama-ok'
```

## Model Location

Host-side 8B model used:

```text
/home/myid/hz85760/model/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_0.gguf
```

Phone-side expected model path:

```text
/data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct.Q4_0.gguf
```

Model size is about 4.3 GiB. Do not commit it to git.

## Command to Resume on a New Server

From repo root:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --methods offline,online,mru,static-min \
  --window-sec 600 \
  --replay-speedup 1 \
  --bench-seconds 600 \
  --ctx-size 4096 \
  --timeout-s 900 \
  --adb-timeout-s 30 \
  --adb-retries 2 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_10min_baselines_full_timebased_usb_op13_ctx4096
```

By default this command now also pushes:

- `build-android-llama/bin/llama-cli`
- `/home/myid/hz85760/model/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_0.gguf`
- detected aarch64 `libomp.so`

Override with:

```bash
--llama-cli <path>
--model-host-path <path>
--libomp-path <path>
```

Use `--no-resume` only if the artifact directory contains invalid rows that should be discarded.

## Last Invalid Attempt

Artifact directory:

```text
.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_10min_baselines_full_timebased_usb_op13_ctx4096
```

It produced 40 rows, but all are invalid:

```text
status=check_log
rc=1
root cause: missing libomp.so
```

Do not use those rows as performance results.

## Expected Next Step

1. Pull this branch on the new server.
2. Build Android OpenCL + CPU_Elastic if needed.
3. Deploy:
   - `llama-cli`
   - `libomp.so`
   - 8B GGUF model
   - model meta
   - cost model
   - offline budget table
   - trace CSVs
4. Run the matrix command above.
5. Inspect `summary/results.csv` and `summary/SUMMARY.md`.
6. If results still show tiny online/offline differences, inspect per-log plan transitions:
   - `provider_get_ms`
   - `apply_ms`
   - load/transfer/xform counts
   - direct disk read ms
   - backend distribution in generated plan JSONs
