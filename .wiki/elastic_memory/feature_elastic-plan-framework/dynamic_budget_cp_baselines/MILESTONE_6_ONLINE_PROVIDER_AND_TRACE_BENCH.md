# Milestone 6: Online Provider And Trace Bench Path

## 目标

把 `online-cp` baseline 从“有状态查询 + solver 脚本”推进到 `llama-cli` 可直接启用的 runtime provider。

新增 `llama-cli` env:

```text
LLAMA_ELASTIC_ONLINE=1
LLAMA_ELASTIC_ONLINE_MODE=native-greedy | external
LLAMA_ELASTIC_MODEL_META=/data/local/tmp/elastic/model_meta.json
LLAMA_ELASTIC_COST_DIR=/data/local/tmp/elastic/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0
LLAMA_ELASTIC_ONLINE_WORK_DIR=/data/local/tmp/elastic/online
```

## Two Online Modes

### native-greedy

Default mode:

```text
LLAMA_ELASTIC_ONLINE_MODE=native-greedy
```

Runs entirely inside `llama-cli`:

```text
1. Query current state with llama_weight_get_state().
2. Load stage_costs.json and op_costs.json.
3. Pick CPU/GPU backend per weight from measured compute cost.
4. Greedily keep resident weights under current budget.
5. Prefer tensors already resident on the selected backend.
6. Emit native ExecPlan JSON into LLAMA_ELASTIC_ONLINE_WORK_DIR.
7. Load the plan with llama_plan_load_json().
8. Return it through callback provider.
```

This mode is intended for actual phone memory-trace latency runs because it avoids Python process startup and does not require Python on Android.

### external

CP-SAT / Python mode:

```text
LLAMA_ELASTIC_ONLINE_MODE=external
LLAMA_ELASTIC_ONLINE_PYTHON=python3
LLAMA_ELASTIC_ONLINE_SOLVER=/data/local/tmp/elastic/dynamic_budget_solver.py
```

Runs:

```text
dynamic_budget_solver.py --state current_state.json ...
```

This mode is useful for comparing solver quality and solver wall time, but it may be too slow or unavailable on a bare Android shell.

## Runtime Fix

`PlanProvider::create_callback()` originally cached callback results by exact budget. That is wrong for online planning because:

```text
same budget, different current residency -> different best plan
```

Added:

```text
GGML_ELASTIC_CALLBACK_NOCACHE=1
```

`llama-cli` sets this automatically for `LLAMA_ELASTIC_ONLINE=1`.

Also fixed `maybe_apply_plan()`:

```text
table provider:
  same budget_mib can skip apply

online no-cache callback:
  new pointer must apply even when budget_mib is unchanged
```

## Phone Setup

Files needed on phone:

```text
/data/local/tmp/elastic/llama-cli
/data/local/tmp/elastic/Llama-3.2-3B-Instruct-q4_0.gguf
/data/local/tmp/elastic/model_meta.json
/data/local/tmp/elastic/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0/stage_costs.json
/data/local/tmp/elastic/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0/op_costs.json
/data/local/tmp/elastic/budget_trace.csv
```

`model_meta.json` can be bootstrapped from profile CSV:

```sh
python3 runtime/plan/build_model_meta_from_profile.py \
  profile_opencl_1500.csv \
  profile_cpu_elastic_1500.csv \
  --out model_meta.json
```

## Native Online Trace Command

```sh
cd /data/local/tmp/elastic

LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
GGML_OPENCL_ELASTIC=1 \
GGML_ELASTIC_TIMING=1 \
GGML_ELASTIC_STAGE_DETAIL=1 \
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/elastic/budget_trace.csv \
LLAMA_ELASTIC_ONLINE=1 \
LLAMA_ELASTIC_ONLINE_MODE=native-greedy \
LLAMA_ELASTIC_MODEL_META=/data/local/tmp/elastic/model_meta.json \
LLAMA_ELASTIC_COST_DIR=/data/local/tmp/elastic/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 \
LLAMA_ELASTIC_ONLINE_WORK_DIR=/data/local/tmp/elastic/online \
./llama-cli -m /data/local/tmp/elastic/Llama-3.2-3B-Instruct-q4_0.gguf \
  -p hello -n 32 -ngl 99 --seed 42 --temp 0 --no-warmup -no-cnv
```

## Offline Table Trace Command

```sh
cd /data/local/tmp/elastic

LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
GGML_OPENCL_ELASTIC=1 \
GGML_ELASTIC_TIMING=1 \
GGML_ELASTIC_STAGE_DETAIL=1 \
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/elastic/budget_trace.csv \
LLAMA_ELASTIC_DIR=/data/local/tmp/elastic/plans_dynamic_offline \
./llama-cli -m /data/local/tmp/elastic/Llama-3.2-3B-Instruct-q4_0.gguf \
  -p hello -n 32 -ngl 99 --seed 42 --temp 0 --no-warmup -no-cnv
```

## Validation

Compiled:

```text
cmake --build build-android-llama --target llama-cli -j2
```

Result:

```text
PASS
```

This covers:

```text
tools/main/main.cpp online provider
native-greedy generator
external solver callback path
PlanProvider callback no-cache mode
maybe_apply_plan same-budget online apply fix
online provider gen_ms log
maybe_apply_plan provider_get_ms/apply_ms log
```

## Current Bench Blocker

`adb devices` currently reports:

```text
List of devices attached
4fd79abf    unauthorized
```

So I cannot push the new `llama-cli` or run the actual memory trace from this environment yet. Once the phone authorizes this host, the next step is:

```sh
adb push build-android-llama/bin/llama-cli /data/local/tmp/elastic/
adb push runtime/plan/dynamic_budget_solver.py /data/local/tmp/elastic/
adb shell chmod +x /data/local/tmp/elastic/llama-cli
adb shell 'cd /data/local/tmp/elastic && <native online trace command>'
```

## Remaining Work

1. Run actual phone trace for:

```text
offline-table
online native-greedy
online external CP-SAT, if Python is available
```

2. Record:

```text
plan generation ms
provider_get_ms
apply_exec_plan ms
stage movement ms
eval ms/token
budget switch count
fallback/failure count
```

3. Add activation `BOUNDARY` profile records for mixed CPU/GPU plans.
