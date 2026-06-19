# Milestone 41: Index-Anchor Pipeline + Xfer Queue on OP12 Trace01

## Why this milestone exists

The previous interval CP-SAT implementation used virtual interval start/end times for schedule diagnostics, but runtime execution should be anchored by graph op index. The earlier mapping could place stages too close to the consumer op, so `load/xform` often behaved like on-demand reload instead of prefetch pipeline.

This milestone changes the execution contract:

- `start_ms` / `end_ms`: CP-SAT diagnostic coordinates only.
- `anchor_op_id`: real runtime trigger, expressed as graph op index.
- `consumer_op_id`: the true consumer op for the weight.
- non-compute stages are triggered before `consumer_op_id` by stage-specific index lead.

## Code changes

- `runtime/plan/dynamic_budget_solver.py`
  - CP-SAT `interval_makespan` now emits `consumer_op_id` and index-based `anchor_op_id`.
  - `prefetch_distance` is passed into CP-SAT schedule emission.
  - Default stage leads currently used by the solver:
    - `load`: `2 * prefetch_distance`
    - `transfer`: `prefetch_distance`
    - `xform`: `prefetch_distance`
    - `sync`: `0`

- `runtime/plan/build_offline_budget_table.py`
  - Added `--prefetch-distance` and forwards it to `dynamic_budget_solver.py`.

- `runtime/plan/run_dynamic_budget_matrix.py`
  - Added `--prefetch-distance`.
  - Added `--offline-chain-state`; offline table is now unchained by default.
  - For online runs, exports `LLAMA_ELASTIC_PREFETCH_DISTANCE`.
  - Starts `remote_dynamic_budget_server.py` with the same budget/cost/objective/prefetch parameters.

- `tools/main/main.cpp`
  - Online remote request now includes `prefetch_distance`.
  - External solver command also includes `--prefetch-distance`.

- `src/llama-context.cpp`
  - CPU xform stage is skipped, not counted as failure, unless `LLAMA_ELASTIC_ENABLE_CPU_XFORM_STAGE=1`.
  - This fixes MRU/static timelines that contain CPU xform events while CPU staged xform is not enabled.

## Key semantic fix: offline table is no longer chained by default

The old matrix runner always passed `--chain-state` when building the offline table. That made neighboring budget buckets partially stateful and reduced the difference between offline and online.

For the intended baseline:

- offline: precomputed table per budget, does not know current runtime residency.
- online: CP-SAT receives current runtime state and penalizes migration.
- MRU: stateful heuristic, no measured global optimization.
- static-min: fixed plan for the minimum trace budget.

`--offline-chain-state` is still available for diagnostics, but it is no longer the default experiment setting.

## Short-window OP12 trace01 result

Device: OP12 `5ae7a43d`  
Trace: `trace_01_user_147`, 60 s replay window, 20 s decode bench  
Model: Meta-Llama-3-8B-Instruct Q4_0  
Config: `prefetch_distance=16`, `interval-stage-kinds=load,xform`, `GGML_ELASTIC_RELOAD_ON_XFER=1`, unchained offline table, placements `cpu,gpu,disk_gpu`.

| baseline | decode ms/token | direct read ms/forward | reload issue ms/forward | apply ms total | anchor load/xform | failures |
|---|---:|---:|---:|---:|---:|---:|
| online remote CP-SAT | 757.46 | 343.30 | 402.75 | 2.475 | 8 / 8 | 0 |
| offline table | 887.68 | 392.52 | 487.02 | 120.159 | 35 / 35 | 0 |
| static-min | 1086.00 | 534.69 | 667.44 | 0.000 | 53 / 53 | 0 |
| MRU | 4091.95 | 696.75 | 0.00 | 2.678 | 53 / 53 | 0 |

Interpretation:

- Online is about 14.7% faster than offline in this short window.
- Online is faster because it applies fewer transitions and has lower per-forward direct read/reload issue cost.
- Static-min is slower because it cannot exploit higher budget periods.
- MRU is correctness-clean after the CPU-xform skip fix, but remains a weak heuristic on this trace.

## Pipeline A/B notes

Unchained `load-only`, `prefetch_distance=8`:

| baseline | decode ms/token | direct read ms/forward | reload issue ms/forward |
|---|---:|---:|---:|
| online | 794.84 | 340.33 | 429.49 |
| offline | 910.87 | 390.97 | 522.36 |
| static-min | 1142.54 | 557.27 | 784.23 |
| MRU | 3587.86 | 669.36 | 0.00 |

Best pipeline setting found so far (`prefetch_distance=16`, `GGML_ELASTIC_RELOAD_ON_XFER=1`) improves both online and offline over this load-only baseline in the short window:

- online: 794.84 -> 757.46 ms/token
- offline: 910.87 -> 887.68 ms/token
- static-min: 1142.54 -> 1086.00 ms/token

## Current long-window validation

Started a 240 s trace window with 180 s decode bench using the best short-window config:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --methods offline,online,mru,static-min \
  --trace-filter trace_01_user_147 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/trace01_pd16_load_xform_xfer_unchained_3min \
  --window-sec 240 --window-stride-sec 240 --bench-seconds 180 \
  --prefetch-distance 16 \
  --interval-stage-kinds load,xform \
  --allowed-placements cpu,gpu,disk_gpu \
  --extra-env GGML_ELASTIC_RELOAD_ON_XFER=1
```

The long-window result should be used to confirm the short-window direction under a more realistic trace duration and thermal state.

## Counter Fix Follow-up

After the first long-window run, dynamic methods showed `anchor_load/xform=0/0` in the final CSV even though logs showed plans with schedule events. Root cause: `apply_exec_plan()` cleared cumulative anchor counters every time a new plan was applied. If the last switched plan had no events, the final summary hid all earlier pipeline activity.

Fix:

- keep `elastic_anchor_fired.clear()` on plan switch, so the new plan can trigger the same op-index anchors again;
- do not reset cumulative anchor counters on plan switch.

This makes final run metrics cumulative across the whole process.

## 3-minute OP12 validation after counter fix

Device: OP12 `5ae7a43d`  
Trace: `trace_01_user_147`, 240 s replay window, 180 s decode bench  
Config: `prefetch_distance=16`, `interval-stage-kinds=load,xform`, `GGML_ELASTIC_RELOAD_ON_XFER=1`, unchained offline table, placements `cpu,gpu,disk_gpu`.

Artifact:

```text
.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/trace01_pd16_xfer_counterfix_3min_online_offline
```

| baseline | decode ms/token | direct read ms/forward | reload issue ms/forward | apply ms total | apply count | anchor load/xform | failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| online remote CP-SAT | 422.75 | 101.21 | 118.91 | 71.407 | 9 | 213 / 213 | 0 |
| offline table | 729.59 | 214.26 | 260.18 | 154.357 | 7 | 178 / 178 | 0 |

Memory trace window:

- min available: 3752.1 MiB
- mean available: 4810.95 MiB
- max available: 5544.5 MiB
- min bucket: 3584 MiB

Interpretation:

- Online is about 42% faster than offline on the 3-minute window.
- Online has lower direct disk read per forward and lower reload issue time per forward.
- Online also has fewer expensive plan-application costs despite more budget queries/switch opportunities.
- Anchor counters now correctly show that both online and offline executed indexed `load/xform` pipeline events with zero failures.

Remaining validation work:

- Run the same 3-minute counter-fixed matrix for `mru` and `static-min` if full four-baseline long-window counters are required.
- Repeat on several `trace/traces_9g` traces after thermal cooldown to avoid order/thermal bias.

## Complete four-baseline 3-minute comparison

After the counter fix, the same artifact was extended with `mru` and `static-min` under the same config.

Artifact:

```text
.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/trace01_pd16_xfer_counterfix_3min_online_offline
```

Trace/config:

- OP12 `5ae7a43d`
- `trace_01_user_147`
- 240 s trace window, 180 s decode bench
- min / mean / max memory: 3752.1 / 4810.95 / 5544.5 MiB
- min budget bucket: 3584 MiB
- `prefetch_distance=16`
- `interval-stage-kinds=load,xform`
- `GGML_ELASTIC_RELOAD_ON_XFER=1`
- placements: `cpu,gpu,disk_gpu`
- offline table: unchained

| baseline | decode ms/token | direct read ms/forward | reload issue ms/forward | apply ms total | apply count | anchor load/xform | failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| online remote CP-SAT | 422.75 | 101.21 | 118.91 | 71.407 | 9 | 213 / 213 | 0 |
| offline table | 729.59 | 214.26 | 260.18 | 154.357 | 7 | 178 / 178 | 0 |
| static-min | 1270.39 | 623.30 | 783.28 | 0.000 | 0 | 71 / 71 | 0 |
| MRU | 3339.57 | 94.22 | 0.00 | 5.686 | 7 | 166 / 166 | 0 |

Ordering by decode speed:

```text
online remote CP-SAT  <  offline table  <  static-min  <  MRU
422.75 ms/token       <  729.59          <  1270.39     <  3339.57
```

Relative speedups:

- online vs offline: 42.0% lower decode ms/token
- online vs static-min: 66.7% lower decode ms/token
- online vs MRU: 87.3% lower decode ms/token
- offline vs static-min: 42.6% lower decode ms/token

Notes:

- Online is fastest because it solves with current runtime residency and avoids much of the offline table migration/reload churn.
- Offline still improves over static-min because it follows the dynamic memory budget buckets, but it does not solve from the current runtime state.
- Static-min is slow because it is pinned to the lowest bucket plan (`3584MiB`) for the whole window.
- MRU has low direct-read metrics here but very poor decode time, so its bottleneck is not direct disk IO; it is choosing a poor backend/layout execution path without measured global optimization.
- All four baselines have zero anchor failures after the CPU-xform skip and cumulative counter fixes.


## No-pipeline / load-only comparison

To isolate the effect of the indexed `load+xform` pipeline, I reran the same
trace window with only indexed `load` stages enabled.

No-pipeline artifact:

```text
.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/trace01_pd16_load_only_counterfix_3min
```

No-pipeline config differences:

- `LLAMA_ELASTIC_INTERVAL_STAGE_KINDS=load`
- no `xform` scheduled as a deferred pipeline stage
- no explicit transfer/xform reload path in the interval schedule
- same OP12 device, same trace, same 240 s trace window, same 180 s decode bench
- same `prefetch_distance=16`, placements, bucket size, KV/misc memory, and offline table mode

Raw no-pipeline results:

| baseline | decode ms/token | direct read ms/forward | reload issue ms/forward | apply ms total | apply count | anchor load/xform | failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| online remote CP-SAT | 692.94 | 154.75 | 207.01 | 154.486 | 9 | 221 / 0 | 0 |
| offline table | 997.63 | 244.89 | 330.19 | 276.080 | 7 | 178 / 0 | 0 |
| static-min | 1418.96 | 606.59 | 882.22 | 0.000 | 0 | 71 / 0 | 0 |
| MRU | 3669.36 | 99.48 | 0.00 | 3.835 | 7 | 166 / 0 | 0 |

Pipeline vs no-pipeline:

| baseline | no-pipeline decode | pipeline decode | change | no-pipeline direct/reload | pipeline direct/reload | no-pipeline anchors | pipeline anchors |
|---|---:|---:|---:|---:|---:|---:|---:|
| online remote CP-SAT | 692.94 | 422.75 | -39.0% | 154.75 / 207.01 | 101.21 / 118.91 | 221 / 0 | 213 / 213 |
| offline table | 997.63 | 729.59 | -26.9% | 244.89 / 330.19 | 214.26 / 260.18 | 178 / 0 | 178 / 178 |
| static-min | 1418.96 | 1270.39 | -10.5% | 606.59 / 882.22 | 623.30 / 783.28 | 71 / 0 | 71 / 71 |
| MRU | 3669.36 | 3339.57 | -9.0% | 99.48 / 0.00 | 94.22 / 0.00 | 166 / 0 | 166 / 166 |

Interpretation:

- The indexed pipeline improves all four baselines on this trace01 window.
- The largest improvement is on online remote CP-SAT: `692.94 -> 422.75 ms/token`.
- Offline also improves substantially: `997.63 -> 729.59 ms/token`.
- Static-min and MRU improve less because their plan quality is still poor; pipeline hides part of the load/transform latency, but it cannot fix bad placement decisions.
- In both configurations, the ordering is preserved:

```text
online remote CP-SAT  <  offline table  <  static-min  <  MRU
```

This is the expected direction: pipeline reduces execution latency, while online
planning still wins over offline because it solves from the current runtime
residency state instead of applying a precomputed table entry blindly.
