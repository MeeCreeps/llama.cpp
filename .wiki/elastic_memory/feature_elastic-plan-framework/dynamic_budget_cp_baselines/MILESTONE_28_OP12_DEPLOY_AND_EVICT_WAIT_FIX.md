# Milestone 28: OP12 Deploy and Adreno Evict Wait Fix

Date: 2026-06-18

## Device

USB device used:

```text
adb serial: 5ae7a43d
lsusb: OPPO Electronics Corp. OnePlus 12
OpenCL: QUALCOMM Adreno(TM) 750
```

Target directory:

```text
/data/local/tmp/hyzheng/elastic
```

## Fixes Made

### Deploy libc++_shared.so

`llama-cli` depends on:

```text
libomp.so
libc++_shared.so
```

OP12 failed without `libc++_shared.so`:

```text
CANNOT LINK EXECUTABLE "./llama-cli":
cannot locate symbol "_ZTTNSt6__ndk114basic_ifstreamIcNS_11char_traitsIcEEEE"
```

`runtime/plan/run_dynamic_budget_matrix.py` now auto-detects and pushes both:

```text
libomp.so
libc++_shared.so
```

### Rebuilt Android llama-cli

The first deployed binary reported:

```text
version: 7120 (a10bd13ed)
```

That was too old for timed benchmark mode.

Rebuilt with:

```bash
cmake --build build-android-llama --target llama-cli -j8
```

Verified on phone:

```text
version: 7121 (6cdc479eb)
```

### Adreno cl_mem Evict Wait

The OP12 10-minute trace crashed with `rc=139` when budget transitions reached low-memory plans with load/transfer/xform events.

Tombstone showed crash in the Adreno driver thread:

```text
tid: AdrenoOsLib
signal 11 (SIGSEGV)
fault addr 0x00000000000001a8
backtrace: /vendor/lib64/libCB.so
```

Crash occurred after plan switches such as:

```text
apply_exec_plan: applied plan budget=3840MiB ... evict=27 ... load=27 transfer=27 xform=27
Segmentation fault
```

Fix:

```text
runtime/weight_buffer_manager_opencl.cpp
```

Changed `wbmcl_evict()` and `wbmcl_evict_batch()` to wait for the compute queue by default before releasing or retaining `cl_mem`.

Use this only for aggressive diagnostics:

```bash
GGML_ELASTIC_EVICT_WAIT=0
```

Default is now conservative correctness:

```text
evict waits unless GGML_ELASTIC_EVICT_WAIT=0
```

## Validation

### 20-second smoke

Command used:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --methods offline \
  --trace-filter trace_01_user_147 \
  --window-sec 600 \
  --replay-speedup 1 \
  --bench-seconds 20 \
  --ctx-size 4096 \
  --timeout-s 180 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/smoke_op12_trace01_offline_s20 \
  --skip-push-model \
  --no-resume
```

Result:

```text
rc=0
eval = 182.29 ms/token
```

### 600-second trace01/offline after evict wait

Command used:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --methods offline \
  --trace-filter trace_01_user_147 \
  --window-sec 600 \
  --replay-speedup 1 \
  --bench-seconds 600 \
  --ctx-size 4096 \
  --timeout-s 900 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/debug_op12_trace01_offline_evictwait_s600 \
  --skip-push-model \
  --no-resume
```

Result:

```text
rc=0
no segfault after low-budget transitions
apply_count=11
evict_planned=175
load/transfer/xform planned=113/113/113
```

The result still had `status=no_perf`; the log ended without the llama perf footer. Immediately after this run, the OP12 disappeared from USB:

```text
adb devices: no devices
lsusb: no OnePlus device
```

So the most likely cause is physical USB/device disconnect, not a llama crash. No new tombstone was available after that point.

## Current Blocker

The phone is no longer visible to the server:

```bash
adb devices -l
lsusb
```

Both show no OP12 / `5ae7a43d`.

## Next Resume Command

Once OP12 is visible again:

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
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_10min_baselines_full_timebased_op12_ctx4096_v3 \
  --skip-push-model \
  --no-resume
```

If the model is missing on the phone, remove `--skip-push-model`.

