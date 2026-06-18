# MILESTONE 16: 8B Multi-Trace 9G Dynamic Matrix

Date: 2026-06-16

## Goal

Run multiple dynamic 9G memory traces on OP13 with the 8B Q4_0 model, using larger real trace windows, and compare:

- offline precomputed table baseline
- online remote CP-SAT baseline

The online CP-SAT solve runs on the server through `adb reverse`; phone timing includes plan fetch wall time in raw eval, and the adjusted number subtracts that wall time.

## Setup

Phone:

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

Model/meta facts:

- model: Meta-Llama-3-8B-Instruct Q4_0
- GGUF size: 4.33 GiB
- OpenCL model buffer: 4155.99 MiB
- KV for `-c 4096`: 512 MiB
- managed matrix weights: 224
- managed matrix bytes: 3744 MiB

Offline table:

```text
/data/local/tmp/hyzheng/elastic/plans_8b_9g_offline_kv512
```

Online remote server:

```bash
adb -s 172.20.115.151:5555 reverse tcp:8765 tcp:8765
python3 runtime/plan/remote_dynamic_budget_server.py \
  --host 127.0.0.1 --port 8765 \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0
```

## Trace Windows

The source traces are from:

```text
trace/traces_9g
```

Each selected source window is about 5 minutes. The generated test traces replay the same dynamic budget shape at 3x speed so `-n 1600` covers the whole window on the phone.

| Generated trace | Source trace | Source start | Source min MiB | Source mean MiB | Source max MiB |
|---|---|---:|---:|---:|---:|
| `trace_9g_07_user_7_5min_x3.csv` | `trace_07_user_7.csv` | 728s | 3305 | 4453 | 5484 |
| `trace_9g_01_user_147_5min_x3.csv` | `trace_01_user_147.csv` | 86s | 3752 | 4674 | 6199 |
| `trace_9g_03_user_116_5min_x3.csv` | `trace_03_user_116.csv` | 334s | 4144 | 4954 | 6261 |

Scan result for all available 5 minute low windows:

```text
trace_01_user_147.csv: start=86.0s min=3752 mean=4674 max=6199
trace_02_user_13.csv:  start=998.0s min=4769 mean=5018 max=5336
trace_03_user_116.csv: start=334.0s min=4144 mean=4954 max=6261
trace_04_user_68.csv:  start=850.0s min=4827 mean=5038 max=5493
trace_05_user_74.csv:  start=594.0s min=4695 mean=5237 max=6001
trace_06_user_204.csv: start=614.0s min=4781 mean=5284 max=6427
trace_07_user_7.csv:   start=728.0s min=3305 mean=4453 max=5484
trace_08_user_270.csv: start=918.0s min=4323 mean=4823 max=5116
trace_09_user_11.csv:  start=382.0s min=4891 mean=5353 max=5809
trace_10_user_115.csv: start=28.0s min=4442 mean=5257 max=5702
```

## Commands

Offline common command shape:

```bash
cd /data/local/tmp/hyzheng/elastic
export LD_LIBRARY_PATH=$PWD
export GGML_OPENCL_DISABLE_ALLOC_HOST_PTR=1
export GGML_OPENCL_USE_SVM=1
export GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/<trace>.csv
export GGML_ELASTIC_BUDGET_BUCKET_MB=256
export LLAMA_ELASTIC_DIR=/data/local/tmp/hyzheng/elastic/plans_8b_9g_offline_kv512
./llama-cli -m Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  -p "Summarize dynamic elastic memory planning for mobile LLM inference." \
  -n 1600 -c 4096 -b 64 -ub 64 -ngl 99 -fa on -t 8 -no-cnv --temp 0
```

Online common additions:

```bash
export LLAMA_ELASTIC_MODEL_META=model_meta_8b.json
export LLAMA_ELASTIC_COST_DIR=profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0
export LLAMA_ELASTIC_ONLINE=1
export LLAMA_ELASTIC_ONLINE_MODE=remote
export LLAMA_ELASTIC_ONLINE_REMOTE_URL=http://127.0.0.1:8765/solve
export LLAMA_ELASTIC_ONLINE_KV_MB=512
```

## Results

`adjusted ms/token = (eval_ms - remote_wall_ms) / tokens`.

| Trace | Mode | Raw eval ms/token | Adjusted ms/token | Remote calls | Remote wall ms | Switches | Skips | Stage events | Stage load/transfer/xform |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| user7 5min_x3 | online remote CP-SAT | 68.60 | 67.88 | 27 | 1152.295 | 18 | 9 | 1035 | 345 / 345 / 345 |
| user7 5min_x3 | offline table | 69.19 | 69.19 | 0 | 0 | 17 | 5 | 1044 | 348 / 348 / 348 |
| user147 5min_x3 | online remote CP-SAT | 68.59 | 68.06 | 29 | 843.385 | 18 | 11 | 657 | 219 / 219 / 219 |
| user147 5min_x3 | offline table | 78.52 | 78.52 | 0 | 0 | 15 | 6 | 660 | 220 / 220 / 220 |
| user147 5min_x3 | online repeat, hot | 69.07 | 68.57 | 29 | 795.850 | 18 | 11 | 657 | 219 / 219 / 219 |
| user116 5min_x3 | online remote CP-SAT | 95.32 | 94.84 | 25 | 771.931 | 4 | 21 | 75 | 25 / 25 / 25 |
| user116 5min_x3 | offline table | 95.84 | 95.84 | 0 | 0 | 4 | 11 | 117 | 39 / 39 / 39 |

## Observations

- user147 is the clearest 5 minute dynamic trace result:
  - offline: 78.52 ms/token
  - online first pass: 68.59 ms/token
  - online hot repeat after more 8B load: 69.07 ms/token
  - online remains about 12% faster than offline even in the repeat.
- user7 has the lowest budget valley, but the 5min_x3 pair was only slightly separated:
  - online: 68.60 ms/token
  - offline: 69.19 ms/token
  - stage event counts were almost identical.
- user116 is a high-budget control:
  - min budget is 4144 MiB, so most of the run can keep all weights on GPU.
  - online and offline speed are effectively equal under heavy thermal load.
  - online still reduced stage events from 117 to 75 by solving exact 4096/4352 budgets instead of using coarser offline table entries.
- The offline table is coarse:
  - at 4096 MiB, offline falls to the 3800 MiB plan with 81 events.
  - online solves exactly at 4096 MiB with 48 events.
  - at 4352 MiB, offline falls to the 4200 MiB plan with 36 events.
  - online solves exactly at 4352 MiB with 27 events.
- Remote CP-SAT overhead is stable and small compared with full decode:
  - 25-29 calls per run
  - 0 failures
  - 0.77-1.15s total remote wall per 1600-token run
  - server solve time is smaller than wall time; adb/HTTP/file handling dominates.
- Main logs do not currently expose direct disk read milliseconds separately. They expose stage event counts and `provider_get_ms`/`apply_ms`; a precise `direct disk read ms` field needs another instrumentation hook.
- Thermal is still a major confound. After repeated 8B runs, two thermal zones still reported `95000`, while most other zones were around 47-50C. The user147 online repeat was run hot and still stayed close to the first online result, which improves confidence for that trace.

## Artifacts

Logs:

```text
artifacts/logs/matrix15/online_8b_trace_9g_07_user_7_5min_x3_n1600_c4096.log
artifacts/logs/matrix15/offline_8b_trace_9g_07_user_7_5min_x3_n1600_c4096.log
artifacts/logs/matrix15/online_8b_trace_9g_01_user_147_5min_x3_n1600_c4096.log
artifacts/logs/matrix15/offline_8b_trace_9g_01_user_147_5min_x3_n1600_c4096.log
artifacts/logs/matrix15/online_repeat_8b_trace_9g_01_user_147_5min_x3_n1600_c4096.log
artifacts/logs/matrix15/online_8b_trace_9g_03_user_116_5min_x3_n1600_c4096.log
artifacts/logs/matrix15/offline_8b_trace_9g_03_user_116_5min_x3_n1600_c4096.log
```

Generated traces:

```text
trace_9g_07_user_7_5min_x3.csv
trace_9g_01_user_147_5min_x3.csv
trace_9g_03_user_116_5min_x3.csv
```

## Status

The multi-trace 8B dynamic run now has a clear positive case:

- user147: online remote CP-SAT is consistently faster than offline table.

It also has two useful controls:

- user7: very low budget but similar plans at this replay length.
- user116: wider budget, online/offline mostly equal.

Next best improvement is instrumentation, not another planner change: add per-stage direct disk read timing to the runtime log/profile so the stage cost model can be validated against actual disk movement time.
