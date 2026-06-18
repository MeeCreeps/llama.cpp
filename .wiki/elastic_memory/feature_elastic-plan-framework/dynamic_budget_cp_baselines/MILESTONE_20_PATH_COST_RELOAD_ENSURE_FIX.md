# MILESTONE 20: Path-cost planner fix with real OpenCL reload cost

Date: 2026-06-17

## Why this milestone exists

The previous backend choice was not scientifically valid for CPU/GPU planning:

```text
OpenCL:      0.03 + MB * 0.020
CPU_Elastic: 0.05 + MB * 0.045
```

That fallback made GPU almost always cheaper before considering the real path:

```text
CPU path: disk -> CPU + CPU transform/repack + CPU compute
GPU path: disk -> CPU + CPU->GPU/write-buffer + GPU transform + GPU compute
```

It also ignored the actual execution behavior observed on OP13: most low-budget decode misses are recorded as `OpenCL/RELOAD_ENSURE`, not as separate `LOAD/TRANSFER/XFORM` stage events.

## Code changes

### `runtime/plan/dynamic_budget_solver.py`

- Added explicit weight-name to OpenCL-node-name mapping:
  - `blk.N.attn_q.weight` -> `Qcur-N`
  - `blk.N.attn_k.weight` -> `Kcur-N`
  - `blk.N.attn_v.weight` -> `Vcur-N`
  - `blk.N.attn_output.weight` -> `attn_out-N`
  - `blk.N.ffn_gate.weight` -> `ffn_gate-N`
  - `blk.N.ffn_up.weight` -> `ffn_up-N`
  - `blk.N.ffn_down.weight` -> `ffn_out-N`
- Compute lookup now prefers measured `MUL_MAT` rows with more samples, so decode-shape records are preferred over prompt-shape records.
- CPU backend is no longer allowed to win from fallback constants if there is no measured per-op `CPU_Elastic/COMPUTE`.
- GPU non-resident path cost now uses measured `OpenCL/RELOAD_ENSURE` when present:

```text
C_gpu_path(w) = C_opencl_compute(w) + C_reload_ensure(w)
```

Only if `RELOAD_ENSURE` is unavailable does it fall back to:

```text
LOAD + TRANSFER + XFORM
```

### `tools/main/main.cpp`

- Native online/MRU cost lookup now mirrors the Python solver:
  - same weight-name to OpenCL-node-name mapping,
  - same `MUL_MAT`/sample preference,
  - same CPU per-op measurement guard,
  - same `RELOAD_ENSURE` preference for GPU non-resident cost.
- Native backend initialization now compares full path cost, not compute-only cost.

## New measured cost model

Built from:

```bash
python3 runtime/plan/build_cost_model.py \
  .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/profile_path_8b_opencl_n4.csv \
  .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/profile_path_stage_user147_n64.csv \
  --out-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_path_reload \
  --device android-opencl-op13 \
  --model Meta-Llama-3-8B-Instruct-Q4_0
```

Summary:

```text
input_rows:    44384
op_records:    1562
stage_records: 159
stage kind:    OpenCL / RELOAD_ENSURE
```

Example measured reload records:

```text
blk.25.ffn_gate.weight      median 13.689 ms, 29 samples, 33.0 MB
blk.26.attn_q.weight        median  4.228 ms, 26 samples,  9.4 MB
blk.26.attn_k.weight        median  1.513 ms, 26 samples,  2.4 MB
blk.31.ffn_down.weight      median 12.176 ms,  2 samples, 33.0 MB
```

## Plan sanity check

Generated with the new `*_path_reload` cost dir:

```text
budget 3584 MiB: pred 436.41 ms/token, gpu 191, disk 33, timeline 99
budget 3840 MiB: pred 337.27 ms/token, gpu 198, disk 26, timeline 78
budget 4096 MiB: pred 237.73 ms/token, gpu 208, disk 16, timeline 48
budget 4352 MiB: pred 141.28 ms/token, gpu 215, disk  9, timeline 27
```

Plan metadata now reports:

```json
{
  "backend_choice": "path_cost",
  "gpu_movement_cost": "reload_ensure",
  "opencl_compute_measured": true,
  "opencl_stage_measured": true,
  "cpu_elastic_compute_measured": false
}
```

## OP13 experiment: user147 min-start low-budget window

Trace window:

```text
source: trace/traces_9g/trace_01_user_147.csv
window: starts at original t=294s
first budget: 3752.1 MB
range: 3752.1 MB -> 6135.2 MB
```

This is a real trace window selected to start at the minimum budget, so the test begins below full 8B model residency.

### Offline table, n64

```text
first apply: budget 4096 MiB, evict 16, load/transfer/xform 16
second apply: budget 4352 MiB, evict 6, load/transfer/xform 9
eval: 424.49 ms/token
reload host-issue: 24993.9 ms, 3068 calls
direct O_DIRECT read: 19602.22 ms, 3034 calls, 51951.7 MB
stage load: 370.16 ms, 25 calls
stage xform: 105.83 ms, 17 ok calls
```

### Offline table, n32

```text
eval: 453.96 ms/token
reload host-issue: 13492.9 ms, 1540 calls
direct O_DIRECT read: 10016.44 ms, 1556 calls, 26931.7 MB
```

### Online remote CP-SAT, n16

```text
remote budget 4096:
  remote_wall_ms: 151.533
  server_solve_ms: 107.365
  provider_get_ms: 178.355
  apply_ms: 391.429
  evict 16, load/transfer/xform 16

eval: 467.16 ms/token
reload host-issue: 7132.5 ms, 768 calls
direct O_DIRECT read: 5215.22 ms, 784 calls, 13881.7 MB
```

### Online remote CP-SAT, n32/n64 status

The online remote n32/n64 runs did not complete within the timeout after the second budget switch. Logs show the run was not stuck in HTTP; it completed remote solving and apply:

```text
budget 4096: server_solve_ms about 118 ms, apply_ms about 318 ms
budget 4352: server_solve_ms about 113 ms, apply_ms about 169 ms
```

After that, decode remained too slow to finish inside the timeout. This needs follow-up before using long online remote numbers.

## Important finding

The current OP13 runtime still spends most miss cost in direct on-demand reload:

```text
OpenCL/RELOAD_ENSURE + direct O_DIRECT read
```

The planned `LOAD/TRANSFER/XFORM` timeline is present in the plan and sometimes in apply, but the dominant decode-time cost is still direct disk reload from `RELOAD_ENSURE`. Therefore the correct cost model for the current implementation is `RELOAD_ENSURE`-based, not a synthetic split-stage model.

## Current limitation

This is still not a full heterogeneous CPU/GPU planner result:

- OpenCL per-op compute is measured.
- OpenCL reload/direct-read cost is measured.
- CPU_Elastic currently reports only graph-level `COMPUTE_GRAPH`, not per-weight/per-op CPU compute.
- Therefore CPU is deliberately disabled for scientific backend selection until per-op CPU compute profiling exists.

## Invalidated runs

Two n16 runs were accidentally started in parallel on the same phone (offline and MRU). They caused GPU/IO contention and then adb instability. Those logs are invalid and must not be used for results. After that, adb/pull became unreliable, so I stopped further phone benchmarking rather than mixing polluted numbers into the table.

## Next fix needed

1. Add CPU_Elastic per-node/per-weight compute profiling if CPU placement is required.
2. Investigate why online remote after the second switch is much slower than offline, despite similar evict/load counts.
3. Run future phone benchmarks strictly one process at a time, with a cool-down or device restart after any timeout.
4. Re-run all trace windows after the device is stable:
   - offline table,
   - online remote CP-SAT,
   - native MRU.

