# Milestone 26: Time-Based 10-Min Baseline Matrix Runner

## Goal

The matrix benchmark should not stop after a fixed number of generated tokens.

Required behavior:

- each trace is a 10-minute memory-budget window,
- each method runs for the trace window duration,
- memory changes during the window drive planner behavior,
- online CP-SAT reports decode speed excluding remote plan query / solve time,
- baselines:
  - offline table CP-SAT,
  - online remote CP-SAT,
  - online native MRU,
  - static-min-budget plan.

## Code Changes

### Timed llama-cli exit

File:

```text
tools/main/main.cpp
```

Added:

```text
LLAMA_ELASTIC_BENCH_SECONDS=<seconds>
```

When set, `llama-cli` should be invoked with:

```text
-n -1
```

The CLI then generates until the decode-duration timer reaches the requested number of seconds. It exits naturally, so the normal perf footer and elastic timing footer are printed.

Important detail:

```text
The timer starts at the first generated token.
```

So model load and prompt evaluation are not counted as part of the timed decode window.

### Matrix runner

File:

```text
runtime/plan/run_dynamic_budget_matrix.py
```

Responsibilities:

1. select a low-memory 10-minute source window from each raw trace,
2. write replay trace CSVs,
3. build an offline CP-SAT table covering all needed budget buckets,
4. push model metadata, cost model, table, and traces to the phone,
5. start / stop the remote CP-SAT server for online,
6. run each baseline,
7. parse logs,
8. write `results.csv` and `SUMMARY.md`.

## Baseline Definitions

### offline

```text
LLAMA_ELASTIC_DIR=<offline_table_dir>
GGML_ELASTIC_DYNAMIC=1
```

The table provider picks the largest budget band `<= current bucket`.

### online

```text
LLAMA_ELASTIC_ONLINE=1
LLAMA_ELASTIC_ONLINE_MODE=remote
LLAMA_ELASTIC_ONLINE_REMOTE_URL=http://127.0.0.1:<port>/solve
```

The phone sends current residency state to the host CP-SAT server. Summary output includes:

```text
raw_ms_per_token
decode_ms_per_token
remote_wall_ms
remote_server_ms
```

For `online`, `decode_ms_per_token` is computed as:

```text
(eval_ms_total - remote_wall_ms) / eval_runs
```

This is the number to use when solver/query time should not count.

### mru

```text
LLAMA_ELASTIC_ONLINE=1
LLAMA_ELASTIC_ONLINE_MODE=mru
```

Native online MRU plan generator. No remote solver.

### static-min

For each trace window, the runner finds:

```text
min_bucket_mib = floor(min(memory_budget_mib) / bucket) * bucket
```

Then applies only:

```text
LLAMA_ELASTIC_APPLY=<offline_table_dir>/plan_<min_bucket_mib>MiB.json
```

`GGML_ELASTIC_DYNAMIC` is intentionally not set for this baseline, so the WBM target is the trace floor/static target. This models a fixed plan chosen for the worst memory point in that trace.

## Full 10-Minute Command

This is the real experiment requested here. It runs 10 raw traces × 4 baselines × 10 minutes, so it can take several hours.

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --methods offline,online,mru,static-min \
  --window-sec 600 \
  --replay-speedup 1 \
  --bench-seconds 600 \
  --ctx-size 4096 \
  --timeout-s 900 \
  --artifact-root .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_10min_baselines_full
```

Outputs:

```text
.wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/matrix_10min_baselines_full/
  traces/
  offline_table/
  logs/
  summary/results.csv
  summary/SUMMARY.md
  summary/trace_windows.json
```

## Faster Rehearsal Command

For debugging the benchmark harness only, use replay compression:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --methods offline,online,mru,static-min \
  --window-sec 600 \
  --replay-speedup 30 \
  --bench-seconds 20 \
  --timeout-s 180 \
  --artifact-root /tmp/elastic_matrix_rehearsal
```

This still uses a 10-minute source window, but replays the memory trace over 20 seconds. This is not the final result; it is a harness check.

## Context Size

Use `--ctx-size 4096` for the full 10-minute run. A smaller context such as 512 can stop early at the context limit before the 600-second timer fires. The 4096-token context also matches the `kv_mib=512` budget assumption used by the planner table.

## Verification

Built and pushed updated Android binary:

```bash
cmake --build build-android-llama --target llama-cli -j 8
adb -s 172.20.115.151:5555 push build-android-llama/bin/llama-cli /data/local/tmp/hyzheng/elastic/llama-cli
```

Smoke command:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --trace-filter trace_01_user_147 \
  --methods offline \
  --bench-seconds 8 \
  --replay-speedup 30 \
  --timeout-s 180 \
  --artifact-root /tmp/elastic_matrix_timed_smoke2
```

Smoke log confirmed:

```text
[elastic-bench] duration limit enabled: 8.000 seconds
[elastic-bench] reached duration 8.000 s after 57 generated tokens
llama_perf_context_print: eval time = 5866.82 ms / 56 runs (104.76 ms per token)
```

After this smoke, the C++ timer was refined so the duration starts at the first generated token instead of before prompt evaluation.

## Notes

The earlier `n64` matrix run was stopped and should not be treated as the requested final experiment. It used fixed generated-token count and is superseded by this time-based runner.
