# Milestone 4: Online State Query Baseline

## 目标

为 `online-cp` baseline 提供当前 weight residency 状态。

新增公开 C API：

```c
enum llama_elastic_weight_state_flags {
    LLAMA_ELASTIC_WEIGHT_DISK_AVAILABLE       = 1u << 0,
    LLAMA_ELASTIC_WEIGHT_CPU_RAW_RESIDENT     = 1u << 1,
    LLAMA_ELASTIC_WEIGHT_CPU_COMPUTE_RESIDENT = 1u << 2,
    LLAMA_ELASTIC_WEIGHT_GPU_RAW_RESIDENT     = 1u << 3,
    LLAMA_ELASTIC_WEIGHT_GPU_COMPUTE_RESIDENT = 1u << 4,
};

struct llama_elastic_weight_state {
    uint32_t flags;
    void *   host_ptr;
};

int llama_weight_get_state(
    struct llama_context * ctx,
    const char * tensor_name,
    struct llama_elastic_weight_state * out_state);
```

## Backend Providers

OpenCL provider reports:

```text
DISK_AVAILABLE
GPU_COMPUTE_RESIDENT
```

CPU_Elastic provider reports:

```text
DISK_AVAILABLE
CPU_RAW_RESIDENT
CPU_COMPUTE_RESIDENT
```

当前 OpenCL WBM 第一版没有稳定暴露 raw vs compute layout 的独立状态，所以先只标 `GPU_COMPUTE_RESIDENT`。后续如果 GPU raw staging 和 transformed layout 分离，需要补 `GPU_RAW_RESIDENT`。

## Internal Registry

新增 internal registry：

```text
llama_weight_state_register()
llama_weight_state_query()
```

位置：

```text
src/llama-mmap.h
src/llama-mmap.cpp
```

公开 API 实现在：

```text
src/llama-context.cpp
include/llama.h
```

## Current State JSON

`dynamic_budget_solver.py` now accepts:

```text
--state current_state.json
```

Schema:

```json
{
  "weights": [
    {
      "name": "blk.0.ffn_down.weight",
      "flags": [
        "disk_available",
        "gpu_compute_resident"
      ]
    }
  ]
}
```

The solver also accepts integer bitmask flags matching the C API.

## Online Solver Behavior

When `--state` is present:

```text
1. If a tensor is already resident on the selected backend, resident selection gives it a strong keep bias.
2. If a currently resident tensor is not kept under the new budget, the emitted plan includes an EVICT event.
3. Movement cost is still estimated from the shared phone cost model.
```

This is the first online baseline. It is not a fully embedded in-process CP solver yet, but it exercises the important difference from offline-table:

```text
offline-table:
  solve with canonical all-disk state

online-cp:
  solve with actual current CPU/GPU residency state
```

## External Callback Baseline

The simplest runtime baseline is:

```text
callback provider:
  1. enumerate model weights from model_meta
  2. call llama_weight_get_state(ctx, weight.name)
  3. write current_state.json
  4. run dynamic_budget_solver.py --state current_state.json
  5. load returned plan with llama_plan_load_json()
```

This intentionally measures online solver overhead. Later optimization can embed the solver in process and add state-hash caching.

## Limitations

1. There is no built-in model weight enumerator API yet; the callback should use the same `weights_ops.json` passed to the solver.
2. OpenCL raw-layout state is not separated from compute-layout state.
3. CP-SAT is still limited to resident selection; full optional-interval scheduling is a later milestone.

