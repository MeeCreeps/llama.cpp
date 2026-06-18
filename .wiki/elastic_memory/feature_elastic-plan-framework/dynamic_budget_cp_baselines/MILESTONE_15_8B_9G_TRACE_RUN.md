# MILESTONE 15: 8B Model on 9GB Memory Traces

Date: 2026-06-16

## Goal

Run an 8B model on OP13 with traces from `trace/traces_9g`, using about a 5 minute memory window, and compare offline table vs online remote CP-SAT.

## Model

Downloaded and deployed:

```text
/data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct.Q4_0.gguf
```

Source:

```text
QuantFactory/Meta-Llama-3-8B-Instruct-GGUF
Meta-Llama-3-8B-Instruct.Q4_0.gguf
```

Smoke result:

- model type: 8B
- GGUF size: 4.33 GiB
- OpenCL model buffer: 4155.99 MiB
- CPU_Elastic model buffer: 281.81 MiB

## Trace Windows

Selected `trace/traces_9g/trace_07_user_7.csv` because it had the lowest 5 minute window among the available traces.

Generated:

```text
trace_9g_user7_window_5min.csv
```

Properties:

- source window start: 728s
- duration: about 300s
- min budget: 3305 MiB
- max budget: 5484 MiB

The lowest region appears around window time 130-152s, so long runs need at least about 160s of decode to cover it.

Also generated a shorter pressure-focused slice:

```text
trace_9g_user7_low_2min.csv
```

Properties:

- duration: about 120s
- min budget: 3305 MiB
- max budget: 4636 MiB

## 8B Metadata And Plans

The existing profile-to-meta path did not work for 8B because bootstrap profile CSV contained only `COMPUTE_GRAPH` rows and no weight stage rows.

Generated model meta directly from GGUF tensor metadata:

```text
runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json
```

Meta contents:

- 224 managed matrix weights
- total managed matrix bytes: 3744 MiB

Built cost model from 8B bootstrap profile:

```text
runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0
```

The cost model currently has compute rows only; stage costs use solver fallback constants.

Built offline table with `kv_mib=512` for `-c 4096`:

```text
/data/local/tmp/hyzheng/elastic/plans_8b_9g_offline_kv512
```

Offline table timeline sizes:

| Budget MiB | Timeline events |
|---:|---:|
| 3000 | 153 |
| 3300 | 126 |
| 3500 | 111 |
| 3800 | 81 |
| 4200 | 36 |
| 4600 | 0 |
| 5000 | 0 |
| 5500 | 0 |
| 6000 | 0 |

## Fix

Online CP-SAT initially produced `timeline=0` even at 3072/3328 MiB. The dumped state had empty flags for all weights:

```json
{"flags": [], "name": "blk.0.attn_k.weight"}
```

The solver incorrectly treated state-present but no `disk_available` flag as immovable. Fixed:

- before: force keep when `state_provided and not state_has_disk(row)`
- after: force keep only when `state_any_resident(row) and not state_has_disk(row)`

After the fix, the same 3072 MiB state produced:

- timeline: 156
- disk weights: 52
- GPU weights: 172

## Commands

Common:

```bash
cd /data/local/tmp/hyzheng/elastic
export LD_LIBRARY_PATH=$PWD
export GGML_OPENCL_DISABLE_ALLOC_HOST_PTR=1
export GGML_OPENCL_USE_SVM=1
export GGML_ELASTIC_BUDGET_BUCKET_MB=256
```

Offline:

```bash
export GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/trace_9g_user7_window_5min.csv
export LLAMA_ELASTIC_DIR=/data/local/tmp/hyzheng/elastic/plans_8b_9g_offline_kv512
./llama-cli -m Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  -p "Summarize dynamic elastic memory planning for mobile LLM inference." \
  -n 3500 -c 4096 -b 64 -ub 64 -ngl 99 -fa on -t 8 -no-cnv --temp 0
```

Online remote:

```bash
adb reverse tcp:8765 tcp:8765
python3 runtime/plan/remote_dynamic_budget_server.py \
  --host 127.0.0.1 --port 8765 \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0

export LLAMA_ELASTIC_MODEL_META=model_meta_8b.json
export LLAMA_ELASTIC_COST_DIR=profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0
export LLAMA_ELASTIC_ONLINE=1
export LLAMA_ELASTIC_ONLINE_MODE=remote
export LLAMA_ELASTIC_ONLINE_REMOTE_URL=http://127.0.0.1:8765/solve
export LLAMA_ELASTIC_ONLINE_KV_MB=512
```

## Results: 5 Minute Window

`adjusted` subtracts remote solver wall time from eval time.

| Run | Raw eval ms/token | Adjusted ms/token | Switches | Stage events load/transfer/xform | Remote wall ms |
|---|---:|---:|---:|---:|---:|
| offline table | 102.15 | 102.15 | 17 | 348 / 348 / 348 | 0 |
| online remote CP-SAT fixed | 120.90 | 120.56 | 18 | 345 / 345 / 345 | 1191.764 |

Interpretation:

- This long pair is not thermally fair. Offline ran first; online ran after a long 8B run.
- Phone thermal readings reached a 95C zone.
- Online and offline planned almost the same number of stage events, so the current online CP-SAT model did not find a better plan for this trace.

## Results: Low 2 Minute Slice

Online was run first, then offline.

| Run | Raw eval ms/token | Adjusted ms/token | Switches | Stage events load/transfer/xform | Remote wall ms |
|---|---:|---:|---:|---:|---:|
| online remote CP-SAT fixed | 70.40 | 70.15 | 10 | 309 / 309 / 309 | 305.986 |
| offline table | 84.99 | 84.99 | 9 | 300 / 300 / 300 | 0 |

Interpretation:

- In the pressure-focused slice, online-first was faster than offline-second.
- This result has order/thermal caveats in the opposite direction.
- Both runs still reported stage request failures:
  - online: 48 failures
  - offline: 81 failures

## Current Status

8B execution on real 9GB trace windows works end-to-end:

- model deployed
- trace converted and deployed
- 8B meta generated
- offline table generated
- online remote CP-SAT works
- state guard bug fixed

The online-vs-offline speed result is not yet clean enough to call final because thermal order strongly affects 8B runs, and stage requests still fail in deferred execution.

## Artifacts

Logs:

- `artifacts/logs/matrix15/offline_8b_user7_5min_n3500_c4096.log`
- `artifacts/logs/matrix15/online_8b_user7_5min_n3500_c4096_fixed.log`
- `artifacts/logs/matrix15/online_8b_user7_low2min_n1200_c4096_fixed.log`
- `artifacts/logs/matrix15/offline_8b_user7_low2min_n1200_c4096.log`

Traces:

- `trace_9g_user7_window_5min.csv`
- `trace_9g_user7_low_2min.csv`
