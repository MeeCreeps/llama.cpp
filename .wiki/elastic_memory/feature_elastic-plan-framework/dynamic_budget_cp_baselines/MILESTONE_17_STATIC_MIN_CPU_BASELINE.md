# MILESTONE 17: Static Minimum Budget CPU Baseline

Date: 2026-06-16

## Goal

Add the simplest conservative baseline requested by the user:

- take each dynamic trace's minimum budget
- turn it into a static constant-budget trace
- run the 8B model entirely on `CPU_Elastic`
- compare this lower-bound baseline against the online remote CP-SAT and offline table results from MILESTONE 16

This baseline does not use online or offline plan switching. It answers: "what if we size for the worst memory point and just run on CPU?"

## Static Traces

Generated and pushed to:

```text
/data/local/tmp/hyzheng/elastic
```

| Static trace | Source dynamic trace | Minimum budget MiB | Static target printed by runtime |
|---|---|---:|---:|
| `trace_9g_07_user_7_static_min.csv` | `trace_9g_07_user_7_5min_x3.csv` | 3305 | 2921 |
| `trace_9g_01_user_147_static_min.csv` | `trace_9g_01_user_147_5min_x3.csv` | 3752 | 3368 |
| `trace_9g_03_user_116_static_min.csv` | `trace_9g_03_user_116_5min_x3.csv` | 4144 | 3760 |

Each static trace contains:

```text
t_ms,budget_mb
0,<min_budget>
300000,<min_budget>
```

## Command

Example for user147:

```bash
cd /data/local/tmp/hyzheng/elastic
export LD_LIBRARY_PATH=$PWD
export GGML_ELASTIC_PROFILE_CSV=profile_static_cpu_8b_01_user_147_n128.csv
export GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/trace_9g_01_user_147_static_min.csv

./llama-cli -m Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  -p "Summarize dynamic elastic memory planning for mobile LLM inference." \
  -n 128 -c 4096 -b 64 -ub 64 -ngl 99 --device CPU_Elastic \
  -fa on -t 8 -no-cnv --temp 0
```

The important part is `--device CPU_Elastic`. The runtime confirmed:

- `llama_model_load_from_file_impl: using device CPU_Elastic`
- `CPU_Elastic model buffer size = 4437.80 MiB`
- `llama_kv_cache: CPU_Elastic KV buffer size = 512.00 MiB`
- `llama_context: CPU_Elastic compute buffer size = 32.31 MiB`
- `graph splits = 1`
- memory breakdown reports the model/context/compute under Host, with CPU_Elastic as the selected backend

## Results

The n=128 results use 127 decode eval runs.

| Trace | Mode | Eval ms/token | Prompt ms/token | Notes |
|---|---|---:|---:|---|
| user7 static min | static min CPU_Elastic | 1294.99 | 251.52 | min budget 3305 MiB |
| user147 static min | static min CPU_Elastic | 1067.26 | 237.76 | min budget 3752 MiB |
| user116 static min | static min CPU_Elastic | 757.10 | 231.52 | min budget 4144 MiB |

Smoke check:

| Trace | Mode | Eval ms/token | Runs |
|---|---|---:|---:|
| user147 static min | static min CPU_Elastic | 1012.78 | 31 |

## Comparison With Online And Offline

| Trace | Online remote CP-SAT ms/token | Offline table ms/token | Static min CPU ms/token |
|---|---:|---:|---:|
| user7 5min_x3 | 68.60 | 69.19 | 1294.99 |
| user147 5min_x3 | 68.59 | 78.52 | 1067.26 |
| user116 5min_x3 | 95.32 | 95.84 | 757.10 |

Approximate slowdown of static min CPU:

| Trace | vs online | vs offline |
|---|---:|---:|
| user7 | 18.9x slower | 18.7x slower |
| user147 | 15.6x slower | 13.6x slower |
| user116 | 7.9x slower | 7.9x slower |

## Interpretation

- The static-min CPU baseline is much slower than both online and offline elastic GPU baselines.
- It is a useful correctness/safety lower bound, but not a competitive runtime policy for 8B on OP13.
- It also explains why the planner should avoid "everything on CPU" except as a fallback: even under the lowest observed budgets, keeping useful weight subsets on GPU is worth an order of magnitude in decode speed.
- The different CPU ms/token values across traces should not be interpreted as budget sensitivity. All three runs used the same all-CPU execution path; the differences are likely thermal/frequency/scheduler effects from back-to-back long CPU runs.

## Artifacts

Logs:

```text
artifacts/logs/matrix15/static_cpu_8b_trace_9g_07_user_7_static_min_n128_c4096.log
artifacts/logs/matrix15/static_cpu_8b_trace_9g_01_user_147_static_min_n128_c4096.log
artifacts/logs/matrix15/static_cpu_8b_trace_9g_03_user_116_static_min_n128_c4096.log
artifacts/logs/matrix15/static_cpu_8b_trace_9g_01_user_147_static_min_n32_c4096.log
```

Generated traces:

```text
trace_9g_07_user_7_static_min.csv
trace_9g_01_user_147_static_min.csv
trace_9g_03_user_116_static_min.csv
```
