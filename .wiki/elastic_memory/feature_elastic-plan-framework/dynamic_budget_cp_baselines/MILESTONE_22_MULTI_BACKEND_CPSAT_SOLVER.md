# MILESTONE 22: Multi-backend CP-SAT planner

Date: 2026-06-17

## Motivation

The planner must not be only a GPU residency planner. The intended problem is heterogeneous placement:

```text
weight w can be resident on CPU, resident on GPU, or left on disk
operator o can execute on CPU or GPU
budget changes can require CPU <-> GPU / disk <-> CPU/GPU movement
```

Offline plans are budget-indexed and state-blind. Online remote CP-SAT should use the current weight state to avoid unnecessary migration and churn.

## Solver change

File:

```text
runtime/plan/dynamic_budget_solver.py
```

The CP-SAT decision is now per-weight placement, not binary keep/drop.

For each weight `w`, the solver chooses exactly one placement:

```text
cpu       : resident in CPU layout, compute on CPU
gpu       : resident in GPU layout, compute on GPU
disk_cpu  : not resident, load/repack on demand, compute on CPU
disk_gpu  : not resident, load/transfer/convert on demand, compute on GPU
```

Budget constraint:

```text
sum_w size(w) * [x_w,cpu + x_w,gpu] <= available_model_budget
```

Objective:

```text
minimize sum_w steady_cost(w, placement)
       + transition_weight * transition_cost(current_state(w), placement)
```

For offline table generation, there is no current state, so:

```text
transition_weight = 0
```

For online remote CP-SAT, the request includes the current state, so transition/migration cost is included.

## Cost paths

CPU resident:

```text
C_cpu_compute(w)
```

GPU resident:

```text
C_gpu_compute(w)
```

Disk -> CPU:

```text
C_cpu_reload(w) + C_cpu_compute(w)
```

Disk -> GPU:

```text
C_gpu_reload_ensure(w) + C_gpu_compute(w)
```

CPU -> GPU transition:

```text
C_transfer_cpu_to_gpu(w) + C_gpu_xform(w)
```

GPU -> CPU transition:

```text
current runtime has no GPU->CPU readback stage;
model weights are disk-backed, so this is represented as evict + disk->CPU reload/repack
```

## Runtime output

The generated ExecPlan now contains real multi-backend placement:

```json
{
  "weights": [
    {"location": "cpu" | "gpu" | "disk"}
  ],
  "ops": [
    {"compute_backend": "cpu" | "gpu"}
  ]
}
```

The existing Plan IR already supports this:

```text
WeightPlan.location
OpPlan.compute_backend
OpPlan.migrate / migrate_from / migrate_xform
```

## Remote server / phone integration

Files:

```text
runtime/plan/remote_dynamic_budget_server.py
runtime/plan/build_offline_budget_table.py
tools/main/main.cpp
```

New controls:

```text
--allow-cpu-fallback
--transition-weight
LLAMA_ELASTIC_ALLOW_CPU_FALLBACK=1
LLAMA_ELASTIC_TRANSITION_WEIGHT=<float>
```

`allow_cpu_fallback` is intentionally explicit. Current 8B profile has measured OpenCL compute and measured OpenCL reload, but does not yet have measured per-op CPU_Elastic compute.

## Smoke test: 8B, 4096 MiB model budget bucket

Command without CPU fallback:

```bash
python3 runtime/plan/dynamic_budget_solver.py \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_path_reload \
  --budget-mib 4096 --kv-mib 512 --misc-mib 256 \
  --time-limit-ms 200 \
  --out /tmp/mb_no_cpu.json
```

Result:

```text
location: gpu 208, disk 16
backend:  gpu 224
timeline: load 16, transfer 16, xform 16
pred: 237.73 ms/token
```

Command with CPU fallback:

```bash
python3 runtime/plan/dynamic_budget_solver.py \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_path_reload \
  --budget-mib 4096 --kv-mib 512 --misc-mib 256 \
  --time-limit-ms 200 \
  --allow-cpu-fallback \
  --out /tmp/mb_cpu.json
```

Result:

```text
location: gpu 144, cpu 64, disk 16
backend:  gpu 160, cpu 64
timeline: load 16, transfer 16, xform 16
pred: 233.53 ms/token
```

This confirms that the CP-SAT formulation is now multi-backend. The CPU numbers in this smoke test are not final scientific numbers because CPU per-op compute is still estimated.

## Online state smoke test

Synthetic mixed state:

```text
1/3 weights currently CPU resident
2/3 weights currently GPU resident
budget 4096 MiB
allow_cpu_fallback=true
transition_weight=1.0
```

Result:

```text
location: gpu 149, cpu 59, disk 16
backend:  gpu 165, cpu 59
timeline events: 64
```

The online objective now uses current state and transition cost while still optimizing CPU/GPU/disk placement.

## Remaining scientific gap

The formulation is now multi-backend, but the 8B Android cost model still lacks measured per-op CPU_Elastic compute.

Required next profiling work:

```text
CPU_Elastic per-op compute timing
CPU disk->CPU reload/repack timing per weight
CPU<->GPU migration timing if runtime support is expanded beyond CPU->GPU
```

Until that is available, CPU placement experiments must be labeled as fallback-estimated.

