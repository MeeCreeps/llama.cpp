# MILESTONE 19: All 20-Minute Trace Windows with Offline, Online CP-SAT, and MRU

Date: 2026-06-16

## Goal

Run all available 9G memory traces using 20-minute source windows, and compare:

- offline table baseline
- online remote CP-SAT baseline
- online MRU baseline

All runs use real OpenCL elastic disk-load execution:

```bash
GGML_OPENCL_ELASTIC=1
GGML_ELASTIC_TIMING=1
GGML_ELASTIC_STAGE_DETAIL=1
LLAMA_ELASTIC_DEFER_STAGE=0
```

`LLAMA_ELASTIC_DEFER_STAGE=0` is retained because the deferred-stage path previously hung under low-budget switches. This matrix validates synchronous real disk-load execution, not asynchronous/deferred prefetch overlap.

## Trace Generation

Source traces:

```text
trace/traces_9g
```

For each source trace, I selected the lowest 20-minute window. To keep the matrix runnable, the source 20-minute window is replayed at `30x`, so each budget trace spans about 40 seconds of wall-clock time while preserving the 20-minute memory-budget shape.

Generated traces:

| generated trace | source | source start | rows | min MiB | mean MiB | max MiB | source span |
|---|---|---:|---:|---:|---:|---:|---:|
| `trace_9g_01_user_147_20min_x30.csv` | `trace_01_user_147.csv` | 84s | 1143 | 3752 | 5123 | 7351 | 1141s |
| `trace_9g_02_user_13_20min_x30.csv` | `trace_02_user_13.csv` | 40s | 1143 | 4769 | 5272 | 7124 | 1141s |
| `trace_9g_03_user_116_20min_x30.csv` | `trace_03_user_116.csv` | 86s | 1143 | 4144 | 5342 | 6611 | 1141s |
| `trace_9g_04_user_68_20min_x30.csv` | `trace_04_user_68.csv` | 88s | 1143 | 4827 | 5560 | 7905 | 1141s |
| `trace_9g_05_user_74_20min_x30.csv` | `trace_05_user_74.csv` | 64s | 1142 | 4695 | 5435 | 6696 | 1140s |
| `trace_9g_06_user_204_20min_x30.csv` | `trace_06_user_204.csv` | 78s | 1143 | 4781 | 5621 | 6730 | 1141s |
| `trace_9g_07_user_7_20min_x30.csv` | `trace_07_user_7.csv` | 74s | 1142 | 3305 | 5081 | 6757 | 1140s |
| `trace_9g_08_user_270_20min_x30.csv` | `trace_08_user_270.csv` | 90s | 1142 | 4323 | 5115 | 7434 | 1140s |
| `trace_9g_09_user_11_20min_x30.csv` | `trace_09_user_11.csv` | 6s | 1142 | 4891 | 5555 | 7086 | 1140s |
| `trace_9g_10_user_115_20min_x30.csv` | `trace_10_user_115.csv` | 29s | 1182 | 4442 | 5449 | 6648 | 1181s |

## Runner

Runner script:

```text
.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/run_matrix19_20min_real_elastic.sh
```

Main matrix command:

```bash
N_PRED=192 TIMEOUT_S=480 \
  .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/run_matrix19_20min_real_elastic.sh
```

Timeout补跑:

```bash
TRACE_FILTER=07_user_7 METHOD_FILTER=online N_PRED=96 TIMEOUT_S=900 \
  .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/run_matrix19_20min_real_elastic.sh

TRACE_FILTER=07_user_7 METHOD_FILTER=mru N_PRED=96 TIMEOUT_S=900 \
  .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/run_matrix19_20min_real_elastic.sh

TRACE_FILTER=08_user_270 METHOD_FILTER=online N_PRED=96 TIMEOUT_S=900 \
  .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/run_matrix19_20min_real_elastic.sh
```

Remote CP-SAT server:

```bash
adb -s 172.20.115.151:5555 reverse tcp:8765 tcp:8765
python3 runtime/plan/remote_dynamic_budget_server.py \
  --host 127.0.0.1 --port 8765 \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0
```

## Summary Results

`online adjusted` subtracts remote solver/fetch wall time from eval time, then divides by eval runs. Raw `eval ms/token` still includes remote provider overhead.

| trace | offline ms/tok | online adjusted ms/tok | MRU ms/tok | online remote wall ms | planned loads offline/online/MRU | direct read ms offline/online/MRU | status |
|---|---:|---:|---:|---:|---:|---:|---|
| 01_user147 | 392.39 | 378.62 | 397.82 | 1729.3 | 39 / 66 / 44 | 53850 / 53325 / 56367 | ok |
| 02_user13 | 43.71 | 41.64 | 43.77 | 433.0 | 0 / 0 / 0 | 0 / 0 / 0 | ok |
| 03_user116 | 219.97 | 218.96 | 224.38 | 899.6 | 27 / 24 / 24 | 26524 / 27540 / 27258 | ok |
| 04_user68 | 43.97 | 43.13 | 43.95 | 147.4 | 0 / 0 / 0 | 0 / 0 / 0 | ok |
| 05_user74 | 43.90 | 41.08 | 43.87 | 542.1 | 0 / 0 / 0 | 0 / 0 / 0 | ok |
| 06_user204 | 43.87 | 42.29 | 43.88 | 311.7 | 0 / 0 / 0 | 0 / 0 / 0 | ok |
| 07_user7 | 624.11 | timeout | 685.47 | n/a | 114 / 79 / 89 | 93234 / n/a / 49271 | online n192 and n96 timed out; MRU is n96 replacement |
| 08_user270 | 164.53 | 157.37 | 163.34 | 574.3 | 39 / 0 / 44 | 17791 / 8956 / 18283 | online is n96 replacement |
| 09_user11 | 43.90 | 42.49 | 43.90 | 263.4 | 0 / 0 / 0 | 0 / 0 / 0 | ok |
| 10_user115 | 107.88 | 106.16 | 107.28 | 636.0 | 12 / 13 / 0 | 9303 / 9428 / 9293 | ok |

## Detailed Observations

### Traces with meaningful memory pressure

The traces that actually stress elastic disk-load behavior are:

- `01_user147`
- `03_user116`
- `07_user7`
- `08_user270`
- `10_user115`

These have non-zero direct disk read and/or planned stage movement.

### Traces that do not stress the planner

The following traces mostly stay in a high-budget region for this model/setup:

- `02_user13`
- `04_user68`
- `05_user74`
- `06_user204`
- `09_user11`

They show about `43-44 ms/token` across all methods and zero direct disk reads. They are useful sanity checks but not useful for demonstrating planner quality.

### Online CP-SAT vs offline

Online adjusted decode speed is better than offline on most completed traces:

- `01_user147`: 392.39 -> 378.62 ms/token
- `03_user116`: 219.97 -> 218.96 ms/token
- `08_user270`: 164.53 -> 157.37 ms/token
- `10_user115`: 107.88 -> 106.16 ms/token

The largest completed improvement is `01_user147`, where online avoids part of the state-blind behavior of the offline table.

### MRU baseline

MRU is generally close to offline on high-budget traces, but worse on the most memory-constrained trace:

- `07_user7`: offline n192 = 624.11 ms/token, MRU n96 = 685.47 ms/token

MRU does not use the measured cost model, so under low budget it can keep a less useful resident set than the optimized planner/offline table.

### trace07 online timeout

`trace_9g_07_user_7_20min_x30.csv` is the hardest trace:

- min budget: 3305 MiB
- offline n192 completed at 624.11 ms/token
- online n192 timed out at 480s
- online n96 also timed out at 900s

The log shows online reached low-budget plan applications, e.g.:

```text
budget=3584MiB timeline=105 load=35 transfer=35 xform=35
budget=3840MiB timeline=78  load=26 transfer=26 xform=26
budget=4352MiB timeline=27  load=9  transfer=9  xform=9
```

Then execution did not reach a perf footer. This should be treated as a failure/timeout for online remote CP-SAT under the most aggressive low-budget 20-minute trace.

## Logs

Logs are under:

```text
.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/logs/matrix19_20min_real_elastic
```

Important files:

```text
progress.log
offline_trace_9g_01_user_147_20min_x30_n192.log
online_trace_9g_01_user_147_20min_x30_n192.log
mru_trace_9g_01_user_147_20min_x30_n192.log
offline_trace_9g_07_user_7_20min_x30_n192.log
online_trace_9g_07_user_7_20min_x30_n192.log
online_trace_9g_07_user_7_20min_x30_n96.log
mru_trace_9g_07_user_7_20min_x30_n96.log
```

## Next Fixes

1. Debug trace07 online timeout.
   - It may be an extreme slow path from repeated state-aware plan switches, or a runtime hang after low-budget plan application.
   - Add a heartbeat around graph decode after `apply_exec_plan` so timeout location is unambiguous.

2. Fix deferred-stage mode.
   - Current matrix uses `LLAMA_ELASTIC_DEFER_STAGE=0`.
   - Deferred mode is still required to evaluate prefetch overlap rather than synchronous movement.

3. Reduce online provider overhead.
   - Remote wall time is non-trivial: e.g. `01_user147` has 1729 ms remote wall across the run.
   - Cache `(budget_bucket, residency_hash)` plans and avoid repeated remote calls for execution-equivalent plans.
