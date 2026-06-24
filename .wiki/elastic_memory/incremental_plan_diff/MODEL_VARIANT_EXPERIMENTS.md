# Model Variant Experiment Plan

## Motivation

The current main result uses:

```text
Meta-Llama-3-8B-Instruct.Q4_0.gguf
```

This model is useful for debugging because it fits near the OP12 dynamic-memory
range, but the Q4_0 file is only about 4.66 GB.  To make the elastic-memory
claim stronger, the experiment matrix should include:

```text
1. max-memory baseline
2. larger resident footprint models
3. different quantization types
```

The goal is to show whether the optimized incremental planner still beats
state-unaware offline switching when the model pressure is higher.

## Added Baseline: Static Max

The runner now supports:

```text
static-max
```

Meaning:

```text
For each trace window, compute max_bucket = ceil(max(trace_memory) / bucket) * bucket.
Apply plan_<max_bucket>MiB.json once.
Do not follow the dynamic trace during decode.
```

This complements:

```text
static-min:
    fixed worst-budget plan

static-max:
    fixed best-budget plan within this trace window
```

Interpretation:

```text
static-min:
    lower-bound memory baseline

static-max:
    upper-bound "always have the trace's maximum memory" baseline

offline / online / candidate:
    dynamic-budget policies that must react to the actual trace
```

Suggested method list:

```text
offline,online,mru,static-min,static-max,candidate-select
```

or for the latest optimized mode:

```text
offline,online,mru,static-min,static-max,candidate-select,diff-tree-ideal
```

## Existing Local Model Candidates

Current local GGUF files:

| model file | size | use |
|---|---:|---|
| `/home/myid/hz85760/model/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_0.gguf` | 4.66 GB | current main baseline |
| `/home/myid/hz85760/model/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_0.gguf` | 1.92 GB | smaller control |
| `/home/myid/hz85760/model/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0.gguf` | 3.42 GB | quantization control |
| `/home/myid/hz85760/model/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf` | 6.43 GB | larger memory-footprint control |

Important caveat:

```text
The 3B f16 file is larger in memory footprint than 8B Q4_0, but it is not a
larger parameter-count model.  It is useful for memory-pressure experiments,
not for claiming larger-model scaling.
```

For a true larger-model experiment, add a 13B/14B Q4/Q5 GGUF and generate its
planner artifacts.

## Required Artifacts Per Model / Quantization

Every model variant needs matching planner artifacts:

```text
GGUF model file
model_meta weights_ops.json
OpenCL/CPU cost profile directory
offline budget table
```

The runner can already accept these through:

```text
--model
--model-host-path
--model-meta
--cost-dir
```

Existing ready-to-use artifacts:

```text
8B Q4_0:
    model_meta:
        runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json
    cost_dir:
        runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01

3B Q4_0:
    model_meta:
        runtime/plan/model_meta/Llama-3.2-3B-Instruct-q4_0.weights_ops.json
    cost_dir:
        runtime/plan/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0
```

For 3B Q8_0 and 3B f16, first run the profiling/artifact pipeline.

## Profiling A New Model Variant

Use `profile_and_build_model_plans.py` to generate model-specific profiles and
planner artifacts.

Example for 3B Q8_0:

```bash
python3 runtime/plan/profile_and_build_model_plans.py \
  --adb-serial 5ae7a43d \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --model Llama-3.2-3B-Instruct-Q8_0.gguf \
  --model-tag Llama-3.2-3B-Instruct-Q8_0 \
  --budgets 2048,2304,2560,2816,3072,3328,3584,3840,4096,4352,4608,4864,5120,5376,5632,5888,6144,6400 \
  --trace trace/traces_9g/trace_01_user_147.csv \
  --allow-cpu-fallback
```

Example for 3B f16:

```bash
python3 runtime/plan/profile_and_build_model_plans.py \
  --adb-serial 5ae7a43d \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --model Llama-3.2-3B-Instruct-f16.gguf \
  --model-tag Llama-3.2-3B-Instruct-f16 \
  --budgets 3840,4096,4352,4608,4864,5120,5376,5632,5888,6144,6400,6656,6912,7168,7424,7680,7936,8192 \
  --trace trace/traces_9g/trace_01_user_147.csv \
  --allow-cpu-fallback
```

Before profiling, push the GGUF to the phone if it is not already there:

```bash
adb -s 5ae7a43d push \
  /home/myid/hz85760/model/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0.gguf \
  /data/local/tmp/hyzheng/elastic/Llama-3.2-3B-Instruct-Q8_0.gguf
```

## Running The Baseline Matrix

Current 8B Q4_0 experiment with static-max:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --methods offline,online,mru,static-min,static-max,candidate-select \
  --trace-filter '01_user_147' \
  --window-sec 180 \
  --bench-seconds 60 \
  --bucket-mib 256 \
  --cooldown-thermal-max-c 42 \
  --top-k 4 \
  --candidate-min-distance 32 \
  --transition-weight 0.5 \
  --extra-env LLAMA_ELASTIC_KEEP_CURRENT_MARGIN_MS=50 \
  --extra-env LLAMA_ELASTIC_CANDIDATE_PREWARM=1 \
  --extra-env LLAMA_ELASTIC_CANDIDATE_LOG_EVENTS=0 \
  --artifact-root .wiki/elastic_memory/incremental_plan_diff/artifacts/trace01_8bq4_staticmax
```

3B Q4_0 control:

```bash
python3 runtime/plan/run_dynamic_budget_matrix.py \
  --adb-serial 5ae7a43d \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --model Llama-3.2-3B-Instruct-Q4_0.gguf \
  --model-host-path /home/myid/hz85760/model/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_0.gguf \
  --model-meta runtime/plan/model_meta/Llama-3.2-3B-Instruct-q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 \
  --methods offline,online,mru,static-min,static-max,candidate-select \
  --trace-filter '01_user_147' \
  --window-sec 180 \
  --bench-seconds 60 \
  --bucket-mib 256 \
  --cooldown-thermal-max-c 42 \
  --top-k 4 \
  --candidate-min-distance 32 \
  --transition-weight 0.5 \
  --extra-env LLAMA_ELASTIC_KEEP_CURRENT_MARGIN_MS=50 \
  --extra-env LLAMA_ELASTIC_CANDIDATE_PREWARM=1 \
  --extra-env LLAMA_ELASTIC_CANDIDATE_LOG_EVENTS=0 \
  --artifact-root .wiki/elastic_memory/incremental_plan_diff/artifacts/trace01_3bq4_staticmax
```

After Q8_0/f16 artifacts are generated, use the same command shape with:

```text
--model <variant>.gguf
--model-host-path /home/myid/hz85760/model/.../<variant>.gguf
--model-meta runtime/plan/model_meta/<variant>.weights_ops.json
--cost-dir runtime/plan/profiles/android-opencl/<variant>
```

## Recommended First Matrix

Keep the first pass small:

```text
traces:
    trace_01_user_147
    trace_06_user_204 oscillating window

models:
    8B Q4_0
    3B Q4_0
    3B Q8_0, after profiling
    3B f16, after profiling if OP12 memory permits

methods:
    static-min
    static-max
    mru
    offline
    online
    candidate-select
```

Expected interpretation:

```text
If static-max is much faster than offline/optimized:
    the dynamic trace has enough max memory to avoid many reloads, so dynamic
    planning is still paying real pressure.

If optimized approaches static-max while respecting the trace:
    strong result; the planner uses transient high-memory opportunities without
    blindly paying offline switching cost.

If larger footprint variants increase the optimized-vs-offline gap:
    stronger evidence that state-aware incremental planning matters more as
    model memory pressure increases.
```
