# Milestone 5: Validation Status

## Build Results

Native build:

```text
cmake --build build-native -j2
```

Result:

```text
PASS
```

Android phone-target build with OpenCL + CPU_Elastic:

```text
cmake --build build-android-llama -j2
```

Result:

```text
PASS
```

This build covers:

```text
ggml-opencl
ggml-cpu-elastic
runtime/elastic_profile_writer.cpp
include/llama.h public state API
PlanProvider native/offline-table loading
llama-cli Android binary
```

Standalone runtime build:

```text
cmake -S runtime -B runtime/build -DELASTIC_BUILD_TESTS=ON
cmake --build runtime/build -j2
```

Result:

```text
PASS
```

## Script Smoke

Synthetic profile CSV -> cost model:

```text
python3 runtime/plan/build_cost_model.py profile.csv --out-dir cost
```

Result:

```text
PASS
```

Synthetic profile CSV -> minimal model meta:

```text
python3 runtime/plan/build_model_meta_from_profile.py profile.csv --out meta.json
```

Result:

```text
PASS
```

Cost model -> single native ExecPlan:

```text
python3 runtime/plan/dynamic_budget_solver.py \
  --model-meta meta.json \
  --cost-dir cost \
  --budget-mib 512 \
  --out plan.json
```

Result:

```text
PASS
```

Cost model -> offline table:

```text
python3 runtime/plan/build_offline_budget_table.py \
  --model-meta meta.json \
  --cost-dir cost \
  --out-dir table \
  --budgets 256,512
```

Result:

```text
PASS
```

Generated native offline table -> `PlanProvider::create_table()`:

```text
runtime/build/tests_elastic/test_plan_provider /tmp/generated_table
```

Result:

```text
PASS
```

This required a runtime fix:

```text
runtime/plan_provider.cpp now loads native ExecPlan JSON first,
then falls back to old make_plan JSON.
```

## Known Test Gaps

Full standalone `ctest` currently reports failures unrelated to this implementation:

```text
test_weight_buffer_manager:
  existing assertion mismatch: "oldest should be 3", got 2

test_plan_ir:
  default fixture runtime/plan/plans/plan_4144MiB.json missing
  also contains old make_plan-specific assertions

test_plan_provider:
  default fixture runtime/plan/plans/index.json missing
```

Targeted provider test with a generated native offline table passes.

## Current Baseline Coverage

Implemented:

```text
M1 profile CSV writer
M2 cost model builder
M3 solver prototype + offline table builder
M4 online state query API + solver --state input
M5 build/script validation
```

Not fully implemented yet:

```text
in-process embedded CP solver
state-hash online plan cache
activation boundary timing records
anchor-time stage execution for solver-generated plans
full optional-interval CP scheduling
```

These are intentionally left as next optimization steps after the first phone-target baseline can run end to end.
