# Milestone 24: Profile-First Plan-Build Pipeline

## Goal

For a new model, planner artifacts should not be hand-written or based on a fixed fallback table. The repeatable path is:

1. profile the model on the target phone,
2. build `model_meta`,
3. build measured stage / compute cost models,
4. build CP-SAT offline budget plans,
5. optionally push the generated plan table back to the phone.

This is also required for the multi-backend CP-SAT planner, because CPU and GPU choices must be compared with measured per-op costs.

## Code Changes

### CPU_Elastic per-op compute profile

File:

```text
ggml/src/ggml-cpu-elastic/ggml-cpu-elastic.cpp
```

Added `CPU_Elastic,COMPUTE` CSV records when running with:

```text
GGML_ELASTIC_CHUNK_SIZE=1
GGML_ELASTIC_PROFILE=1
GGML_ELASTIC_PROFILE_CSV=...
```

Each chunk is a single graph node, so the profile captures:

- backend: `CPU_Elastic`
- kind: `COMPUTE`
- op name: e.g. `MUL_MAT`, `MUL`, `GET_ROWS`
- node name: e.g. `Qcur-0`
- shape / bytes
- measured ms

The existing `CPU_Elastic,COMPUTE_GRAPH` record is still emitted for graph-level summary.

### New profile/build orchestration script

File:

```text
runtime/plan/profile_and_build_model_plans.py
```

The script performs:

1. `adb devices`
2. push dynamic budget trace
3. OpenCL compute profile
4. OpenCL reload / movement profile using the trace
5. CPU_Elastic per-op compute profile
6. pull CSV files
7. run `build_model_meta_from_profile.py`
8. run `build_cost_model.py`
9. run `build_offline_budget_table.py`
10. optionally push the offline table to `/data/local/tmp/hyzheng/elastic`

Example command:

```bash
python3 runtime/plan/profile_and_build_model_plans.py \
  --adb-serial 172.20.115.151:5555 \
  --remote-dir /data/local/tmp/hyzheng/elastic \
  --model Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  --model-tag Meta-Llama-3-8B-Instruct-Q4_0_autoprofile \
  --trace trace/traces_9g/trace_01_user_147_minstart_5min.csv \
  --budgets 3328,3584,3840,4096,4352 \
  --kv-mib 512 \
  --misc-mib 256 \
  --safety-mib 64 \
  --push-plan-dir
```

Generated artifacts:

```text
runtime/plan/generated/<model-tag>/csv/
runtime/plan/generated/<model-tag>/offline_table/
runtime/plan/generated/<model-tag>/manifest.json
runtime/plan/model_meta/<model-tag>.weights_ops.json
runtime/plan/profiles/android-opencl/<model-tag>/
```

## Verification

### Build

Command:

```bash
python3 -m py_compile runtime/plan/*.py
cmake --build build-android-llama --target llama-cli -j 8
```

Result:

```text
Python compile: pass
Android llama-cli build: pass
```

One existing warning remains:

```text
unused variable 'pre_resident'
```

### Phone CPU profile smoke

Pushed the rebuilt binary:

```bash
adb -s 172.20.115.151:5555 push \
  build-android-llama/bin/llama-cli \
  /data/local/tmp/hyzheng/elastic/llama-cli
```

Ran a short CPU_Elastic profile:

```bash
adb -s 172.20.115.151:5555 shell '
cd /data/local/tmp/hyzheng/elastic &&
chmod 755 ./llama-cli &&
rm -f cpu_compute_smoke.csv &&
LD_LIBRARY_PATH=$PWD \
GGML_ELASTIC_PROFILE=1 \
GGML_ELASTIC_PROFILE_CSV=/data/local/tmp/hyzheng/elastic/cpu_compute_smoke.csv \
GGML_ELASTIC_CHUNK_SIZE=1 \
./llama-cli \
  -m /data/local/tmp/hyzheng/elastic/Meta-Llama-3-8B-Instruct.Q4_0.gguf \
  -p smoke \
  -n 1 \
  -c 128 \
  -b 8 \
  -ub 8 \
  -t 8 \
  --temp 0 \
  --no-warmup \
  -ngl 0 \
  -dev CPU_Elastic
'
```

Pulled and inspected:

```bash
adb -s 172.20.115.151:5555 pull \
  /data/local/tmp/hyzheng/elastic/cpu_compute_smoke.csv \
  /tmp/elastic_cpu_smoke/cpu_compute_smoke.csv
```

Result:

```text
rows: 3477
CPU_Elastic,COMPUTE:       3282
CPU_Elastic,COMPUTE_GRAPH: 195
```

Example measured per-op row:

```text
CPU_Elastic,COMPUTE,Qcur-0,MUL_MAT,...,ms=9.721771,extra=chunk_size_1
```

## Current Status

The missing profile-first path is now present:

- a new model can be profiled first on OP13,
- CPU per-op compute costs are measured instead of guessed,
- OpenCL stage/reload and compute CSVs feed the same cost model builder,
- CP-SAT offline table generation is one command after profiling.

The next full experiment should run the orchestration script without `--dry-run` for the selected model/trace, then compare offline table and remote online CP-SAT with the same measured cost directory.
