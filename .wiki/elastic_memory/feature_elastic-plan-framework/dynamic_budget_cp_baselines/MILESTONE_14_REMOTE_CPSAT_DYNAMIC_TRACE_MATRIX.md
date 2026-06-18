# MILESTONE 14: Remote CP-SAT Dynamic Trace Matrix

Date: 2026-06-16

## Goal

Run dynamic memory traces on OP13 and compare:

- offline table baseline
- online remote CP-SAT baseline

The phone has no `python`/`python3`, so CP-SAT runs on the server over `adb reverse tcp:8765 tcp:8765`. The phone waits for the returned plan and executes it. Reported adjusted decode excludes remote solver wall time.

## Fixes Before Measurement

- Added execution-equivalent plan debounce in `llama_context::maybe_apply_plan()`.
  - Different budget buckets often generate the same execution intent.
  - Before this fix every budget bucket reinstalled dispatch/anchors and invalidated the graph.
  - After this fix, equivalent plans log `execution-equivalent, skipped apply`.
- Kept `GGML_ELASTIC_BUDGET_BUCKET_MB=256` for dynamic traces.
  - This preserves real budget changes while avoiding one solve per tiny interpolated budget step.

## Commands

Common phone environment:

```bash
cd /data/local/tmp/hyzheng/elastic
export LD_LIBRARY_PATH=$PWD
export GGML_OPENCL_DISABLE_ALLOC_HOST_PTR=1
export GGML_OPENCL_USE_SVM=1
export GGML_ELASTIC_BUDGET_BUCKET_MB=256
```

Offline:

```bash
export LLAMA_ELASTIC_DIR=/data/local/tmp/hyzheng/elastic/plans_dynamic_offline_cpsat_matrix
export GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/trace_sawtooth_900_3000.csv
./llama-cli -m Llama-3.2-3B-Instruct-q4_0.gguf \
  -p "Summarize dynamic elastic memory planning for mobile LLM inference." \
  -n 64 -c 512 -b 64 -ub 64 -ngl 99 -fa on -t 8 -no-cnv --temp 0
```

Online remote:

```bash
adb reverse tcp:8765 tcp:8765
python3 runtime/plan/remote_dynamic_budget_server.py \
  --host 127.0.0.1 --port 8765 \
  --model-meta runtime/plan/model_meta/Llama-3.2-3B-Instruct-q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0

export LLAMA_ELASTIC_MODEL_META=model_meta.json
export LLAMA_ELASTIC_COST_DIR=profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0
export LLAMA_ELASTIC_ONLINE=1
export LLAMA_ELASTIC_ONLINE_MODE=remote
export LLAMA_ELASTIC_ONLINE_REMOTE_URL=http://127.0.0.1:8765/solve
export LLAMA_ELASTIC_ONLINE_WORK_DIR=online_work_matrix14_saw_defer_b256
```

## Dynamic Traces

- `trace_sawtooth_900_3000.csv`: repeated 3000 -> 900 -> 3000 -> 1200 -> 3000 -> 900 -> 1500 -> 3000 -> 900.
- `trace_pulse_900_3000.csv`: faster pulse trace with repeated 900/1000/1200 low-budget windows and recovery to 2200/2600/3000.

Both traces include budgets below full model residency.

## Results

Adjusted decode = `(eval_time_ms - remote_wall_ms) / eval_runs`. This excludes remote CP-SAT solve/network wait, as intended for phone execution timing.

| Trace | Baseline | Raw eval ms/token | Adjusted decode ms/token | Solver wall ms | Switches | Skipped equivalent applies | Stage events load/transfer/xform | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sawtooth 900-3000 | offline table | 64.91 | 64.91 | 0.00 | 11 | 2 | 120 / 120 / 120 | 0 |
| sawtooth 900-3000 | online remote CP-SAT | 55.57 | 45.03 | 664.21 | 3 | 11 | 1 / 0 / 1 | 0 |
| pulse 900-3000 | offline table | 61.02 | 61.02 | 0.00 | 10 | 3 | 134 / 134 / 134 | 0 |
| pulse 900-3000 | online remote CP-SAT | 55.36 | 46.81 | 538.67 | 3 | 9 | 1 / 0 / 1 | 0 |

## Profile CSV: Pulse Trace

`GGML_ELASTIC_PROFILE_CSV` was collected with unique filenames for the pulse trace.

| Baseline | COMPUTE_GRAPH ms | RELOAD_ENSURE count | RELOAD_ENSURE ms |
|---|---:|---:|---:|
| offline table | 951.147 | 5 | 958.179 |
| online remote CP-SAT | 962.548 | 3 | 587.715 |

`RELOAD_ENSURE` is the closest available direct disk/read reload timing in the current profile CSV schema. Stage load/transfer/xform are available from plan/apply logs, not yet as separate CSV timing rows.

## Interpretation

Online remote CP-SAT now shows the expected behavior:

- It reacts to current runtime state.
- It produces far fewer stage events than the offline table.
- It avoids the invalid/offline-heavy low-budget event plans.
- Excluding solver wait, adjusted decode is clearly faster:
  - sawtooth: 45.03 vs 64.91 ms/token
  - pulse: 46.81 vs 61.02 ms/token

The remaining bottleneck is solver latency. On-device Python is unavailable, but the server-side CP-SAT path works and is explicitly accounted separately.

## Artifacts

Logs:

- `artifacts/logs/matrix14/offline_trace_sawtooth_900_3000_defer_n64_bucket256.log`
- `artifacts/logs/matrix14/online_remote_trace_sawtooth_900_3000_defer_n64_bucket256.log`
- `artifacts/logs/matrix14/offline_trace_pulse_900_3000_defer_n64_bucket256_profiled.log`
- `artifacts/logs/matrix14/online_remote_trace_pulse_900_3000_defer_n64_bucket256_profiled.log`

Profiles:

- `artifacts/logs/matrix14/profile_pulse_offline_b256.csv`
- `artifacts/logs/matrix14/profile_pulse_online_b256.csv`
