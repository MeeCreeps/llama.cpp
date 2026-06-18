# MILESTONE 23: OP13 multi-backend offline vs online test

Date: 2026-06-17

## Goal

Actually run the multi-backend CP-SAT planner on OP13 and compare offline table vs online remote CP-SAT.

This milestone uses the new multi-backend solver from Milestone 22:

```text
placement choices: cpu, gpu, disk_cpu, disk_gpu
```

Because the 8B cost model still lacks measured per-op CPU_Elastic compute, these tests enable CPU fallback explicitly:

```text
--allow-cpu-fallback
LLAMA_ELASTIC_ALLOW_CPU_FALLBACK=1
```

So the run validates the multi-backend solver/runtime path, but CPU costs are still estimated.

## Offline table

Generated on host:

```bash
python3 runtime/plan/build_offline_budget_table.py \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_path_reload \
  --out-dir /tmp/plans_8b_multibackend_cpu_fallback_kv512 \
  --budgets 3328,3584,3840,4096,4352,4608,4864,5120,5376,5632 \
  --kv-mib 512 \
  --misc-mib 256 \
  --safety-mib 64 \
  --time-limit-ms 200 \
  --allow-cpu-fallback
```

Representative offline plans:

```text
4096 MiB:
  location: gpu 144, cpu 64, disk 16
  backend:  gpu 160, cpu 64
  pred:     233.53 ms/token

4352 MiB:
  location: gpu 153, cpu 62, disk 9
  backend:  gpu 162, cpu 62
  pred:     137.22 ms/token
```

## Trace window

```text
trace: trace_01_user_147_minstart_5min.csv
first budget: 3752.1 MB
first applied bucket: 4096 MiB
```

This window starts at the minimum budget point of user147.

## Offline table execution, n16

Phone command used:

```bash
LLAMA_ELASTIC_DIR=/data/local/tmp/hyzheng/elastic/plans_8b_multibackend_cpu_fallback_kv512
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/trace/traces_9g/trace_01_user_147_minstart_5min.csv
LLAMA_ELASTIC_DEFER_STAGE=0
./llama-cli ... -n 16 -c 4096 -ngl 99
```

Result:

```text
apply:
  budget 4096 MiB
  evict=80
  route=224
  events=48
  load=16 transfer=16 xform=16
  apply_ms=328.306

eval:
  555.24 ms/token

reload:
  host issue total: 5738.3 ms
  reload calls: 474
  direct O_DIRECT read: 4304.38 ms, 490 calls, 11607.0 MB
```

Interpretation:

Offline table is state-blind. It moves many weights out of the initial GPU-resident state to realize its CPU/GPU/disk placement:

```text
offline 4096 target: gpu 144, cpu 64, disk 16
runtime apply: evict 80
```

## Online remote CP-SAT execution, n16

Server:

```bash
python3 runtime/plan/remote_dynamic_budget_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_path_reload \
  --kv-mib 512 \
  --misc-mib 256 \
  --safety-mib 64 \
  --time-limit-ms 200 \
  --allow-cpu-fallback
```

Phone:

```bash
LLAMA_ELASTIC_ONLINE=1
LLAMA_ELASTIC_ONLINE_MODE=remote
LLAMA_ELASTIC_ONLINE_REMOTE_URL=http://127.0.0.1:8765/solve
LLAMA_ELASTIC_ALLOW_CPU_FALLBACK=1
LLAMA_ELASTIC_TRANSITION_WEIGHT=1.0
```

Result:

```text
remote solve:
  server_solve_ms=497.523
  remote_wall_ms=651.882

apply:
  budget 4096 MiB
  evict=16
  route=224
  events=64
  load=16 transfer=16 xform=16
  apply_ms=339.457

eval:
  459.94 ms/token

reload:
  host issue total: 6981.2 ms
  reload calls: 768
  direct O_DIRECT read: 5178.71 ms, 784 calls, 13881.7 MB
```

Interpretation:

Online sees the current state: model weights are initially GPU resident. It avoids offline's state-blind CPU placement and only evicts the 16 weights that do not fit.

```text
offline n16: 555.24 ms/token, evict=80
online  n16: 459.94 ms/token, evict=16
```

This is the expected multi-backend online advantage: offline's new plan moves many weights across backend/disk state; online accounts for current state and avoids that churn.

## Offline table execution, n32

Result:

```text
apply:
  budget 4096 MiB
  evict=80
  apply_ms=305.463

eval:
  403.17 ms/token

reload:
  host issue total: 10575.2 ms
  reload calls: 950
  direct O_DIRECT read: 8145.63 ms, 966 calls, 22371.0 MB
```

## Online n32 status

Online n32 with `transition_weight=1.0` timed out after the second budget switch.

Observed:

```text
4096 solve:
  server_solve_ms=198.019
  apply_ms=341.070
  evict=16

4352 solve:
  server_solve_ms=160.704
  apply_ms=574.175
  generated plan: gpu 175, disk 49
  eval did not finish before timeout
```

The corresponding state dump before the 4352 solve showed:

```text
gpu_compute_resident: 175 weights
disk only:            49 weights
```

So during decode after the first 4096 plan, the runtime ended up with more disk-resident weights than the target plan's 16 disk weights. Online then kept that state too conservatively at transition_weight=1.0.

Local transition-weight sweep using the real 4352 state:

```text
tw=1.0:
  location: gpu 175, disk 49
  pred: 402.4

tw=0.1:
  location: gpu 203, cpu 12, disk 9
  pred: 142.2

tw=0.05:
  location: gpu 201, cpu 14, disk 9
  pred: 142.0
```

A phone n32 rerun with `transition_weight=0.1` reduced the second switch plan churn but still did not complete within the timeout:

```text
4352 apply:
  prefetch=27
  events=128
  load=46 transfer=36 xform=46
  apply_ms=495.383
```

This indicates the second-switch runtime path still has a bottleneck. The n16 result is valid; longer online windows need more runtime tuning around budget-increase prefetch / resident restoration.

## Current conclusion

The multi-backend solver is implemented and executes on OP13.

For the first low-budget switch:

```text
offline: 555.24 ms/token
online:  459.94 ms/token
speedup: 17.2%
```

This confirms the intended effect:

```text
offline is state-blind and migrates too many weights;
online sees current state and avoids unnecessary backend/disk movement.
```

Remaining work:

```text
1. Add measured CPU_Elastic per-op compute instead of CPU fallback.
2. Tune online transition weighting for budget increase.
3. Fix/runtime-profile why second switch can leave 49 weights disk-resident and make n32 online timeout.
4. Re-run all traces after the above fixes.
```

