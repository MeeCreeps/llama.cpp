# Milestone 25: Profiled Trace01 Phone Online vs Offline

## Target

Phone:

```text
OP13 adb serial: 172.20.115.151:5555
remote dir: /data/local/tmp/hyzheng/elastic
model: Meta-Llama-3-8B-Instruct.Q4_0.gguf
trace: trace/traces_9g/trace_01_user_147_minstart_5min.csv
```

## Profile-First Pipeline

Ran:

```bash
python3 runtime/plan/profile_and_build_model_plans.py \
  --adb-serial 172.20.115.151:5555 \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --model Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --model-tag Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01 \
  --trace trace/traces_9g/trace_01_user_147_minstart_5min.csv \
  --budgets 3328,3584,3840,4096,4352,4608,4864,5120,5376,5632 \
  --kv-mib 512 \
  --misc-mib 256 \
  --safety-mib 64 \
  --time-limit-ms 250 \
  --n-opencl 16 \
  --n-reload 16 \
  --n-cpu 4 \
  --ctx-size 512 \
  --batch 32 \
  --threads 8 \
  --push-plan-dir
```

Profile output:

```text
opencl_compute.csv rows: 8245
opencl_reload.csv  rows: 8602
cpu_compute.csv    rows: 5795
cost records: stage=55, op=2815
```

Important issue found:

```text
CSV-derived model_meta only had 55 weights.
```

Reason: this 5-minute trace only triggered reload for a subset of weights, so deriving `weights_ops.json` only from reload/profile CSV is trace-limited.

Fix/workaround used for the actual table:

```text
Use the existing complete 8B model_meta:
runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json

Use the newly measured cost model:
runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01
```

The orchestration script now supports:

```text
--model-meta <complete weights_ops.json>
```

so future short-trace profiling can reuse complete metadata safely.

## Full-Meta Offline Table

Command:

```bash
python3 runtime/plan/build_offline_budget_table.py \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01 \
  --out-dir runtime/plan/generated/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01/offline_table_fullmeta \
  --budgets 3328,3584,3840,4096,4352,4608,4864,5120,5376,5632 \
  --kv-mib 512 \
  --misc-mib 256 \
  --safety-mib 64 \
  --time-limit-ms 250
```

Pushed to:

```text
/data/local/tmp/hyzheng/elastic/plans_Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01_fullmeta
```

Representative plans:

```text
3328MiB: loc gpu=118 cpu=64 disk=42, backend gpu=160 cpu=64, timeline=126
3584MiB: loc gpu=124 cpu=64 disk=36, backend gpu=160 cpu=64, timeline=108
3840MiB: loc gpu=132 cpu=65 disk=27, backend gpu=159 cpu=65, timeline=81
4096MiB: loc gpu=143 cpu=65 disk=16, backend gpu=159 cpu=65, timeline=48
4352MiB: loc gpu=152 cpu=63 disk=9,  backend gpu=161 cpu=63, timeline=27
4608MiB+: loc gpu=159 cpu=65 disk=0, timeline=0
```

## Phone Test Commands

Offline table:

```bash
LLAMA_ELASTIC_DIR=/data/local/tmp/hyzheng/elastic/plans_Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01_fullmeta
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/trace_01_user_147_minstart_5min.csv
GGML_ELASTIC_DYNAMIC=1
GGML_ELASTIC_BUDGET_BUCKET_MB=256
GGML_ELASTIC_TIMING=1
```

Online remote CP-SAT:

```bash
python3 runtime/plan/remote_dynamic_budget_server.py \
  --host 127.0.0.1 \
  --port 18081 \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01

adb -s 172.20.115.151:5555 reverse tcp:18081 tcp:18081
```

Phone env:

```text
LLAMA_ELASTIC_ONLINE=1
LLAMA_ELASTIC_ONLINE_MODE=remote
LLAMA_ELASTIC_ONLINE_REMOTE_URL=http://127.0.0.1:18081/solve
LLAMA_ELASTIC_MODEL_META=/data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json
LLAMA_ELASTIC_COST_DIR=/data/local/tmp/hyzheng/elastic/profiled_trace01_cost
LLAMA_ELASTIC_TRANSITION_WEIGHT=0.1
LLAMA_ELASTIC_ONLINE_KV_MB=512
LLAMA_ELASTIC_ONLINE_MISC_MB=256
LLAMA_ELASTIC_ONLINE_SAFETY_MB=64
LLAMA_ELASTIC_ONLINE_TIME_LIMIT_MS=250
```

## Results

### n=16

Offline:

```text
eval: 257.10 ms/token
provider_get_ms: 8.873
apply_ms: 0.341
apply: evict=81 route=224 events=48 load=16 transfer=16 xform=16
anchor summary: fired=30 load=10 transfer=10 xform=10 failures=0
reload host issue: 1786.9 ms
reload calls: 143
direct O_DIRECT read: 1360.30 ms, 153 calls, 3604.5 MB
```

Online remote CP-SAT:

```text
eval: 212.98 ms/token
remote_wall_ms: 632.876
server_solve_ms: 547.754
provider_get_ms: 654.258
apply_ms: 0.308
apply: evict=16 route=224 events=64 load=16 transfer=16 xform=16
anchor summary: fired=64 load=16 transfer=16 xform=16 failures=0
reload host issue: 2793.7 ms
reload calls: 323
direct O_DIRECT read: 2196.43 ms, 339 calls, 5852.2 MB
```

Speedup:

```text
257.10 / 212.98 = 1.207x
online is about 17.2% lower ms/token
```

### n=32

Offline:

```text
eval: 317.83 ms/token
provider_get_ms: 6.987
apply_ms: 0.266
apply: evict=81 route=224 events=48 load=16 transfer=16 xform=16
anchor summary: fired=30 load=10 transfer=10 xform=10 failures=0
reload host issue: 3651.6 ms
reload calls: 285
direct O_DIRECT read: 2644.86 ms, 295 calls, 6750.0 MB
```

Online remote CP-SAT:

```text
eval: 213.79 ms/token
remote_wall_ms: 399.008
server_solve_ms: 264.093
provider_get_ms: 421.780
apply_ms: 0.335
apply: evict=16 route=224 events=64 load=16 transfer=16 xform=16
anchor summary: fired=64 load=16 transfer=16 xform=16 failures=0
reload host issue: 5589.1 ms
reload calls: 659
direct O_DIRECT read: 4344.35 ms, 675 calls, 11468.2 MB
```

Speedup:

```text
317.83 / 213.79 = 1.487x
online is about 32.7% lower ms/token
```

## Plan Difference at 4096MiB

Offline table plan:

```text
location: gpu=143 cpu=65 disk=16
backend:  gpu=159 cpu=65
timeline: load=16 transfer=16 xform=16
```

Online remote CP-SAT plan:

```text
location: gpu=208 disk=16
backend:  gpu=224
timeline: evict=16 load=16 transfer=16 xform=16
```

Interpretation:

```text
Offline is state-blind.
It moves many weights away from the current GPU-resident state and routes 65 ops to CPU.

Online is state-aware.
It observes that the weights are currently GPU-resident, pays only the necessary evictions, and keeps all ops on GPU.
```

This is the expected effect: online CP-SAT is faster because it optimizes against the current residency state, not just the current budget.

## Cleanup

Stopped the remote solver and removed adb reverse:

```bash
adb -s 172.20.115.151:5555 reverse --remove tcp:18081
```
