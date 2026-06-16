# Milestone 3: Solver Prototype And Offline Table

## 目标

实现一个可以生成 native `ExecPlan` JSON 的动态 budget solver 原型，并生成 `PlanProvider::create_table()` 可加载的 offline budget table。

新增脚本：

```text
runtime/plan/dynamic_budget_solver.py
runtime/plan/build_offline_budget_table.py
```

## Solver 输入

```text
--model-meta weights_ops.json
--cost-dir runtime/plan/profiles/android-opencl/<model>
--budget-mib B
--out plan_B.json
```

`weights_ops.json` schema：

```json
{
  "weights": [
    {
      "weight_id": 0,
      "name": "blk.0.ffn_down.weight",
      "layer": 0,
      "byte_size": 37748736,
      "quant": "Q4_0"
    }
  ],
  "ops": [
    {
      "op_id": 0,
      "name": "blk.0.ffn_down.weight",
      "layer": 0,
      "weight_id": 0
    }
  ]
}
```

If `ops` is missing, the solver infers one op per weight.

## Solver 行为

第一版范围控制：

```text
movement unit = whole weight tensor
compute backend = CPU_Elastic or OpenCL
dispatch = STATIC
sub-tensor split = disabled
NPU = disabled
KV cache = fixed reserve
```

流程：

```text
1. Load model meta.
2. Load stage_costs.json and op_costs.json.
3. Pick CPU/GPU backend per op from measured compute cost when available.
4. Compute per-weight resident value from measured LOAD/TRANSFER/XFORM cost.
5. Select resident weights under budget:
   - OR-Tools CP-SAT if installed.
   - deterministic greedy fallback otherwise.
6. Emit native ExecPlan JSON.
```

`bottleneck` field records which resident selector was used:

```text
cp_sat
greedy
```

## Offline Table Command

```sh
python3 runtime/plan/build_offline_budget_table.py \
  --model-meta runtime/plan/model_meta/Llama-3.2-3B-Instruct-q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 \
  --out-dir /tmp/plans_dynamic_offline \
  --budgets 1500,1800,2200,3000,4144,4912,6144,7600,8192
```

Output:

```text
/tmp/plans_dynamic_offline/
  index.json
  plan_1500MiB.json
  plan_1800MiB.json
  ...
```

Runtime use:

```text
LLAMA_ELASTIC_PROVIDER=table
LLAMA_ELASTIC_PLANS_DIR=/data/local/tmp/elastic/plans_dynamic_offline
```

## Current Limitations

1. CP-SAT is only used for resident selection, not full op precedence / optional interval scheduling yet.
2. If OR-Tools is not installed, greedy fallback keeps the pipeline usable.
3. CPU_Elastic compute cost is graph-level in current profile data, so per-op CPU route selection is approximate.
4. Activation boundary cost is still zero unless future `BOUNDARY` profile records are added.

