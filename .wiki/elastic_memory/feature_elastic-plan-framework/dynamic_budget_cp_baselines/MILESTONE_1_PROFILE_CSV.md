# Milestone 1: Phone Profile CSV

## 目标

为动态 budget CP baseline 提供真实手机端 cost profile 数据。

新增 env：

```text
GGML_ELASTIC_PROFILE_CSV=/data/local/tmp/elastic/profile.csv
```

只有设置该 env 时才写 CSV。默认路径不启用 profile writer，OpenCL / CPU_Elastic hot path 使用初始化时缓存的 `profile_csv` bool，避免默认 decode 每个 op 读 env 或拿 mutex。

## CSV Schema

```text
backend,kind,name,op,quant,token,op_id,weight_id,ne0,ne1,ne2,ne3,bytes,ms,ok,extra
```

字段语义：

```text
backend:
  OpenCL
  CPU_Elastic

kind:
  LOAD
  TRANSFER
  XFORM
  RELOAD_ENSURE
  COMPUTE
  COMPUTE_GRAPH

name:
  tensor / op name

op:
  ggml op name, compute records only

quant:
  ggml type name, compute records only

weight_id:
  WBM idx for weight stage/reload records

op_id:
  graph node index for OpenCL COMPUTE
  graph node count for CPU_Elastic COMPUTE_GRAPH

extra:
  plan_stage
  decode_ensure
  clFinish_profiled
  delegate_cpu_graph
```

## OpenCL Records

OpenCL writes:

```text
LOAD           plan-triggered disk/mmap -> host staging
TRANSFER       plan-triggered host staging -> GPU-visible raw/backend buffer
XFORM          plan-triggered GPU layout conversion
RELOAD_ENSURE  decode-time wbmcl_ensure_resident()
COMPUTE        per OpenCL graph node compute
```

`COMPUTE` records use `clFinish()` after the op, because the goal is cost-model calibration, not low-overhead production timing. Therefore `GGML_ELASTIC_PROFILE_CSV` should be used in dedicated profiling runs, not final latency runs.

## CPU_Elastic Records

CPU_Elastic writes:

```text
LOAD           plan-triggered disk/mmap -> CPU raw staging
XFORM          plan-triggered CPU raw staging -> CPU resident slot
RELOAD_ENSURE  decode-time direct/mmap reload into resident slot
COMPUTE_GRAPH  delegated CPU graph compute wall time
```

CPU_Elastic currently delegates the whole graph to the CPU backend. Without a larger scheduler/runtime change, this path cannot produce truthful per-op CPU compute timing. The first baseline therefore uses graph-level CPU compute records and per-weight reload/stage records.

## Phone Command

Example all-GPU profiling run:

```sh
cd /data/local/tmp/elastic

LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
GGML_OPENCL_ELASTIC=1 \
GGML_ELASTIC_TIMING=1 \
GGML_ELASTIC_STAGE_DETAIL=1 \
GGML_ELASTIC_PROFILE_CSV=/data/local/tmp/elastic/profile_opencl_1500.csv \
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/elastic/const_1500_budget.csv \
LLAMA_ELASTIC_APPLY=/data/local/tmp/elastic/plan_1500_all_gpu_streaming.json \
./llama-cli -m Llama-3.2-3B-Instruct-q4_0.gguf \
  -p hello -n 16 -ngl 99 --seed 42 --temp 0 --no-warmup -no-cnv
```

Example CPU_Elastic profiling run:

```sh
cd /data/local/tmp/elastic

LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
GGML_OPENCL_ELASTIC=1 \
GGML_ELASTIC_TIMING=1 \
GGML_ELASTIC_STAGE_DETAIL=1 \
GGML_ELASTIC_PROFILE_CSV=/data/local/tmp/elastic/profile_cpu_elastic_1500.csv \
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/elastic/const_1500_budget.csv \
LLAMA_ELASTIC_APPLY=/data/local/tmp/elastic/plan_1500_all_cpu_streaming.json \
./llama-cli -m Llama-3.2-3B-Instruct-q4_0.gguf \
  -p hello -n 16 -ngl 99 --device CPU_Elastic --seed 42 --temp 0 --no-warmup -no-cnv
```

## Implementation Files

```text
runtime/elastic_profile_writer.h
runtime/elastic_profile_writer.cpp
ggml/src/ggml-opencl/ggml-opencl.cpp
ggml/src/ggml-cpu-elastic/ggml-cpu-elastic.cpp
ggml/src/ggml-opencl/CMakeLists.txt
ggml/src/ggml-cpu-elastic/CMakeLists.txt
runtime/CMakeLists.txt
```

## Notes

This milestone is intentionally phone-first. Desktop runs are useful only for compile/smoke testing; the cost model should be built from the Android target CSV.

