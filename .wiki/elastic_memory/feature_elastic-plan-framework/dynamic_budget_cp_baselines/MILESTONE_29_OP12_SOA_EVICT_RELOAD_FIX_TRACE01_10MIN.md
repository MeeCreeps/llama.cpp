# Milestone 29: OP12 SOA Evict/Reload Fix and Trace01 10-Min Results

Date: 2026-06-18

Device:

- ADB serial: `5ae7a43d`
- Phone: OP12 / `CPH2583`
- Remote dir: `/data/local/tmp/hyzheng/elastic`
- Model: `Meta-Llama-3-8B-Instruct.Q4_0.gguf`
- Trace window: `trace_01_user_147.csv`, source start `100s`, span `600s`, replay speed `1.0x`
- Budget range in selected window: min `3752.1 MiB`, mean `4907.2 MiB`, max `6198.6 MiB`, min bucket `3584 MiB`

## Problem Found

The previous OP12 online and MRU runs failed after dynamic budget switches:

1. First failure mode: Adreno driver thread segfaulted after plan transitions with SOA Q4 weights.
2. After routing SOA evict through callbacks, the failure became deterministic `CL_INVALID_MEM_OBJECT` at Q4 matmul image creation from `extra->q`.

Root causes:

- `wbmcl_evict()` and `wbmcl_evict_batch()` treated SOA Q4/Q8 tensors as normal `cl_mem` blocks. This skipped the registered SOA `evict_fn`, leaving `parent/q/d` tensor extra state inconsistent across dynamic plan transitions.
- Decode fallback reload for non-resident SOA tensors still called the generic `wbmcl_ensure_resident()`. That creates a raw parent `cl_mem` but does not rebuild SOA `q/d` sub-buffers, so the next Q4 matmul can see an invalid `extra->q`.
- Full SOA triple reuse (`parent+q+d`) is unsafe on OP12/Adreno in this dynamic path. The default now reuses only the parent buffer and recreates `q/d` sub-buffers on reload. `GGML_ELASTIC_POOL_PARENT_ONLY=0` can restore the aggressive diagnostic path.

## Code Changes

- `runtime/weight_buffer_manager_opencl.cpp`
  - `wbmcl_evict()` calls the SOA `evict_fn` when registered.
  - `wbmcl_evict_batch()` calls the SOA `evict_fn` per victim when registered.
  - Host staging is released after successful SOA eviction.

- `ggml/src/ggml-opencl/ggml-opencl.cpp`
  - Decode fallback reload uses SOA `reload_fn` for Q4/Q8/MXFP4 tensors instead of generic `wbmcl_ensure_resident()`.
  - SOA pool default changed to parent-only reuse. Fresh `q/d` sub-buffers are created on each reload.

## Validation Commands

Short smoke tests:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --methods online \
  --trace-filter trace_01_user_147 \
  --window-sec 600 \
  --replay-speedup 1 \
  --bench-seconds 60 \
  --ctx-size 4096 \
  --timeout-s 600 \
  --adb-timeout-s 30 \
  --adb-retries 2 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/debug_op12_trace01_online_soa_fallback_smoke_s60 \
  --skip-push-model \
  --no-resume

python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --methods mru \
  --trace-filter trace_01_user_147 \
  --window-sec 600 \
  --replay-speedup 1 \
  --bench-seconds 60 \
  --ctx-size 4096 \
  --timeout-s 600 \
  --adb-timeout-s 30 \
  --adb-retries 2 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/debug_op12_trace01_mru_soa_fallback_smoke_s60 \
  --skip-push-model \
  --no-resume
```

10-minute trace01 matrix:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --methods online,mru,static-min \
  --trace-filter trace_01_user_147 \
  --window-sec 600 \
  --replay-speedup 1 \
  --bench-seconds 600 \
  --ctx-size 4096 \
  --timeout-s 2400 \
  --adb-timeout-s 30 \
  --adb-retries 2 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_op12_trace01_10min_after_soa_fix \
  --skip-push-model \
  --no-resume

python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --methods offline \
  --trace-filter trace_01_user_147 \
  --window-sec 600 \
  --replay-speedup 1 \
  --bench-seconds 600 \
  --ctx-size 4096 \
  --timeout-s 2400 \
  --adb-timeout-s 30 \
  --adb-retries 2 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_op12_trace01_10min_after_soa_fix_offline \
  --skip-push-model \
  --no-resume
```

## Results

All four trace01 10-minute runs completed with `rc=0`.

`online` decode ms/token excludes remote CP-SAT wall time. Other methods use raw eval ms/token.

| method | status | decode ms/token | raw ms/token | eval runs | apply count | planned evict/load/xfer/xform | remote wall ms | direct read ms | direct read calls | reload calls |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| online | ok | 204.17 | 209.85 | 2728 | 19 | 128/563/293/563 | 15514.27 | 34052.93 | 3046 | 2483 |
| mru | ok | 527.79 | 527.79 | 1121 | 18 | 339/315/313/315 | 0.00 | 22807.69 | 2068 | 1753 |
| static-min | ok | 863.85 | 863.85 | 694 | 0 | 100/36/36/36 | 0.00 | 205137.19 | 20188 | 20152 |
| offline | ok | 754.69 | 754.69 | 786 | 18 | 228/222/222/222 | 0.00 | 18805.02 | 992 | 770 |

Artifact roots:

- `.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_op12_trace01_10min_after_soa_fix`
- `.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_op12_trace01_10min_after_soa_fix_offline`

## Observations

- Online is now runnable on OP12 and is clearly faster on this dynamic trace window.
- The measured ordering for trace01 is:
  - online: `204.17 ms/token`
  - mru: `527.79 ms/token`
  - offline: `754.69 ms/token`
  - static-min: `863.85 ms/token`
- Online performed more direct disk reads than offline, but it avoided the much worse decode behavior caused by offline/static plan mismatch over this memory window.
- Static-min produced the most direct disk activity: `20188` direct reads and `205137.19 ms` direct read time.
- Offline remains slower than MRU in this window despite fewer direct reads than online/MRU. The fixed binary reduced the old offline result from `1027.35 ms/token` to `754.69 ms/token`, but online still has a large advantage.

## Next Steps

- Run all `trace/traces_9g/*.csv` with the same 10-minute window configuration.
- Keep this OP12 SOA path as the default for mobile experiments.
- If optimizing later, test `GGML_ELASTIC_POOL_PARENT_ONLY=0` only as a diagnostic/performance experiment, not as the correctness default.
