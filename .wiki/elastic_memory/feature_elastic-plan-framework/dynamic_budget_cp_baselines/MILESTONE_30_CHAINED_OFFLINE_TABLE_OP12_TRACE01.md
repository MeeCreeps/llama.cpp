# Milestone 30: Chained Offline Table Fix on OP12 Trace01

Date: 2026-06-18

## Problem

The previous offline-table baseline could be slower than both MRU and static-min. That is not a good baseline: offline can be slower than online because it cannot observe the exact current runtime residency state, but it should not lose to simpler baselines only because of avoidable plan churn.

Root cause:

- Offline table plans were solved independently per budget bucket.
- A budget switch could jump to a plan with a different weight placement, without charging the transition from the previous offline bucket.
- Static-min is stable by construction, and MRU is stateful, so both could avoid some churn that the independent offline table created.

## Fix

Added chained offline table construction:

- `build_offline_budget_table.py --chain-state`
  - Builds budgets in ascending order.
  - Converts each solved plan into a solver state JSON.
  - Passes the previous lower-budget plan state into the next budget solve.
- `run_dynamic_budget_matrix.py`
  - Enables `--chain-state` for offline table builds.
  - Passes the matrix `--transition-weight` into the table builder.

`transition_weight` controls how strongly the solver penalizes movement away from the current state:

```text
objective =
    compute cost
  + load / transfer / transform cost
  + backend synchronization cost
  + transition_weight * residency/location churn cost
```

Observed tuning:

- `transition_weight=1.0`: too sticky, all budget buckets collapsed to the lowest-budget plan.
- `transition_weight=0.1`: allowed budget growth to pull weights from disk to GPU/CPU while limiting adjacent-bucket churn.

## Validation Command

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --methods offline,online,mru,static-min \
  --trace-filter trace_01_user_147 \
  --window-sec 600 \
  --replay-speedup 1 \
  --bench-seconds 120 \
  --ctx-size 4096 \
  --timeout-s 900 \
  --adb-timeout-s 30 \
  --adb-retries 2 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/debug_op12_trace01_chain_all_s120 \
  --skip-push-model \
  --no-resume
```

## Result

Artifact:

```text
.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/debug_op12_trace01_chain_all_s120
```

Trace window:

- trace: `trace_01_user_147_10min_x1.csv`
- source span: 600 s
- replay span: 600 s
- min/mean/max memory: 3752.1 / 4907.2 / 6198.6 MiB
- minimum bucket: 3584 MiB

| method | raw ms/token | decode ms/token | direct read ms | direct read calls | apply count | failures |
|---|---:|---:|---:|---:|---:|---:|
| online | 213.29 | 200.89 | 30419.21 | 3230 | 15 | 0 |
| offline chained | 242.54 | 242.54 | 20597.40 | 1707 | 13 | 0 |
| mru | 265.50 | 265.50 | 21234.68 | 2385 | 11 | 0 |
| static-min | 1094.77 | 1094.77 | 51499.76 | 5165 | 0 | 0 |

This ordering is now reasonable:

```text
online < offline chained < mru < static-min
```

Online remains faster because it solves against the current runtime state. Chained offline is still a precomputed table, but it no longer ignores adjacent-budget transition cost.

## Offline Table Shape

The chained table changes placement gradually:

| budget MiB | placement | timeline events | changed from previous |
|---:|---|---:|---:|
| 3584 | cpu=64, gpu=124, disk=36 | 108 | - |
| 3840 | cpu=65, gpu=132, disk=27 | 107 | 9 |
| 4096 | cpu=65, gpu=143, disk=16 | 81 | 11 |
| 4352 | cpu=63, gpu=152, disk=9 | 56 | 11 |
| 4608 | cpu=65, gpu=159, disk=0 | 25 | 9 |
| >=4864 | cpu=65, gpu=159, disk=0 | 0 | 0 |

## Next Full Run

Run all 10-minute traces with the fixed table:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --methods offline,online,mru,static-min \
  --trace-glob 'trace/traces_9g/trace_[0-9][0-9]_user_*.csv' \
  --window-sec 600 \
  --replay-speedup 1 \
  --bench-seconds 600 \
  --ctx-size 4096 \
  --timeout-s 2400 \
  --adb-timeout-s 30 \
  --adb-retries 2 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_op12_all_10min_chained_offline \
  --skip-push-model
```

