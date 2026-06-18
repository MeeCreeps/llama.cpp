# MILESTONE 18: Retest Offline vs Online with Real OpenCL Elastic Disk Loads

Date: 2026-06-16

## Goal

Retest the 8B offline-table and online remote CP-SAT baselines after discovering that earlier 8B results did not fully exercise real OpenCL elastic disk reloads.

The corrected runs explicitly set:

```bash
GGML_OPENCL_ELASTIC=1
GGML_ELASTIC_TIMING=1
GGML_ELASTIC_STAGE_DETAIL=1
```

I also used:

```bash
LLAMA_ELASTIC_DEFER_STAGE=0
```

because the default deferred-stage path hung in an offline `user147 n=128` smoke run after applying a lower-budget plan. The synchronous stage path completes and exposes the real `LOAD / TRANSFER / XFORM` and direct disk read costs.

## Device and Setup

Device:

```text
172.20.115.151:5555
```

Deploy directory:

```text
/data/local/tmp/hyzheng/elastic
```

Model:

```text
/data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct.Q4_0.gguf
```

Offline table:

```text
/data/local/tmp/hyzheng/elastic/plans_8b_9g_offline_kv512
```

Remote CP-SAT server:

```bash
adb -s 172.20.115.151:5555 reverse tcp:8765 tcp:8765
python3 runtime/plan/remote_dynamic_budget_server.py \
  --host 127.0.0.1 --port 8765 \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0
```

Common llama-cli shape:

```bash
cd /data/local/tmp/hyzheng/elastic
export LD_LIBRARY_PATH=$PWD
export GGML_OPENCL_DISABLE_ALLOC_HOST_PTR=1
export GGML_OPENCL_USE_SVM=1
export GGML_OPENCL_ELASTIC=1
export GGML_ELASTIC_TIMING=1
export GGML_ELASTIC_STAGE_DETAIL=1
export LLAMA_ELASTIC_DEFER_STAGE=0
export GGML_ELASTIC_BUDGET_BUCKET_MB=256

./llama-cli \
  -m Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  -p "Summarize dynamic elastic memory planning for mobile LLM inference." \
  -n <N> -c 4096 -b 64 -ub 64 -ngl 99 -fa on -t 8 -no-cnv --temp 0
```

Offline adds:

```bash
export GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/<trace>.csv
export LLAMA_ELASTIC_DIR=/data/local/tmp/hyzheng/elastic/plans_8b_9g_offline_kv512
```

Online adds:

```bash
export GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/<trace>.csv
export LLAMA_ELASTIC_MODEL_META=/data/local/tmp/hyzheng/elastic/model_meta_8b.json
export LLAMA_ELASTIC_COST_DIR=/data/local/tmp/hyzheng/elastic/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0
export LLAMA_ELASTIC_ONLINE=1
export LLAMA_ELASTIC_ONLINE_MODE=remote
export LLAMA_ELASTIC_ONLINE_REMOTE_URL=http://127.0.0.1:8765/solve
export LLAMA_ELASTIC_ONLINE_KV_MB=512
export LLAMA_ELASTIC_ONLINE_WORK_DIR=/data/local/tmp/hyzheng/elastic/<work_dir>
```

## Valid Results

`adjusted excl remote` subtracts online `remote_wall_ms` from eval time and divides by eval runs. This reports phone execution excluding remote CP-SAT/fetch wall time.

| trace | method | eval ms/token | adjusted excl remote | direct read ms/calls/MB | stage load calls/ms/MB | xform calls(ok)/ms | event loads | apply sum ms | provider sum ms | remote wall/server ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| user147_n64 | offline-table | 426.39 | 426.39 | 20299.2 / 3172 / 54523.5 | 39 / 753.4 / 1154.2 | 39(3) / 95.9 | 39 | 862.9 | 34.2 | 0.0 / 0.0 |
| user147_n64 | online remote CP-SAT | 374.14 | 359.94 | 17513.8 / 2883 / 45899.2 | 25 / 463.0 / 706.5 | 25(2) / 82.2 | 25 | 546.8 | 587.9 | 894.7 / 363.4 |
| user116_n64 | offline-table | 237.22 | 237.22 | 10253.2 / 1596 / 26398.5 | 0 / 0.0 / 0.0 | 0(0) / 0.0 | 0 | 0.2 | 5.1 | 0.0 / 0.0 |
| user116_n64 | online remote CP-SAT | 237.43 | 231.00 | 10410.5 / 1596 / 26398.5 | 0 / 0.0 / 0.0 | 0(0) / 0.0 | 0 | 0.1 | 44.7 | 405.1 / 46.1 |

## Interpretation

### user147

This is the useful corrected comparison.

The online CP-SAT plan performs less real movement than the offline table:

- offline planned stage loads: 39
- online planned stage loads: 25
- offline direct O_DIRECT read time: 20299.2 ms
- online direct O_DIRECT read time: 17513.8 ms
- offline eval: 426.39 ms/token
- online eval: 374.14 ms/token
- online adjusted excluding remote wall: 359.94 ms/token

So, under this trace segment, online is faster because it uses the actual runtime residency state and avoids part of the offline table's state-blind movement.

### user116

This trace segment is less useful for demonstrating planner differences. Both offline and online apply essentially one high-budget/equivalent plan and have no planned `LOAD / TRANSFER / XFORM` stage events.

The remaining direct disk reads are runtime reload/ensure traffic, not planned stage movement differences. As expected, offline and online eval ms/token are nearly identical.

### user7

The `n=64` online remote run generated many low-budget online plans but its log did not include the final perf footer, so I did not count it as a valid result.

A shorter `n=32` pair completed:

| trace | method | eval ms/token | adjusted excl remote | direct read ms/calls/MB | planned stage loads |
|---|---|---:|---:|---:|---:|
| user7_n32 | offline-table | 604.40 | 604.40 | 15290.8 / 2474 / 41199.8 | 0 |
| user7_n32 | online remote CP-SAT | 599.18 | 593.02 | 15107.7 / 2474 / 41199.8 | 0 |

This `n=32` window is too short to hit the important low-budget planned-stage region, so it is mainly a smoke result, not a good planner comparison.

## Important Fix/Workaround

Earlier 8B online/offline numbers around 60-90 ms/token were invalid for disk-load comparison because OpenCL elastic was not enabled. The corrected runs show real disk read costs:

```text
GGML_OPENCL_ELASTIC=1
direct O_DIRECT read calls > 0
stage load / transfer / xform counters > 0 when low-budget plans are applied
```

The default deferred-stage path should still be debugged. In this retest, it hung in:

```text
run_offline_elastic_8b_user147_n128.log
```

after switching to lower budget plans with deferred stage events:

```text
stage_defer=1
events=81 load=27 transfer=27 xform=27
```

The completed retest uses:

```bash
LLAMA_ELASTIC_DEFER_STAGE=0
```

so the result is valid for real disk-load execution, but not yet a validation of asynchronous/deferred prefetch overlap.

## Logs

Valid logs:

```text
artifacts/logs/run_offline_elastic_8b_user147_n64_defer0.log
artifacts/logs/run_online_remote_elastic_8b_user147_n64_defer0.log
artifacts/logs/run_offline_elastic_8b_user116_n64_defer0.log
artifacts/logs/run_online_remote_elastic_8b_user116_n64_defer0.log
artifacts/logs/run_offline_elastic_8b_user7_n32_defer0.log
artifacts/logs/run_online_remote_elastic_8b_user7_n32_defer0.log
```

Invalid/incomplete logs:

```text
artifacts/logs/run_offline_elastic_8b_user147_n128.log
artifacts/logs/run_online_remote_elastic_8b_user7_n64_defer0.log
```
