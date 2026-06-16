# Milestone 2: Cost Model Builder

## 目标

把 Milestone 1 的手机端 profile CSV 聚合成 CP solver 可读取的 cost table。

新增脚本：

```text
runtime/plan/build_cost_model.py
runtime/plan/build_model_meta_from_profile.py
```

脚本无 pandas 依赖，方便在普通 host 环境直接运行。

## 输入

一个或多个 `GGML_ELASTIC_PROFILE_CSV` 产出的 CSV：

```text
profile_opencl_1500.csv
profile_cpu_elastic_1500.csv
profile_opencl_2200.csv
...
```

## 输出

```text
profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0/
  stage_costs.json
  op_costs.json
  boundary_costs.json
  summary.json
```

当前会聚合：

```text
stage_costs:
  LOAD
  TRANSFER
  XFORM
  RELOAD_ENSURE

op_costs:
  COMPUTE
  COMPUTE_GRAPH

boundary_costs:
  BOUNDARY
```

`BOUNDARY` 记录目前预留，后续接 CPU/GPU activation boundary timing。

## 聚合 Key

```text
backend
kind
name
op
quant
shape = [ne0, ne1, ne2, ne3]
bytes
extra
```

每组输出：

```text
samples
median_ms
p90_ms
mean_ms
total_ms
total_bytes
mbps
```

## Host Command

从手机拉回 CSV 后：

```sh
python3 runtime/plan/build_cost_model.py \
  /tmp/profile_opencl_1500.csv \
  /tmp/profile_cpu_elastic_1500.csv \
  --device android-opencl \
  --model Llama-3.2-3B-Instruct-q4_0 \
  --out-dir runtime/plan/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0
```

Bootstrap minimal model meta from the same CSV:

```sh
python3 runtime/plan/build_model_meta_from_profile.py \
  /tmp/profile_opencl_1500.csv \
  /tmp/profile_cpu_elastic_1500.csv \
  --out runtime/plan/model_meta/Llama-3.2-3B-Instruct-q4_0.weights_ops.json
```

This profile-derived meta is enough for the first solver baseline. A later extractor should read GGUF tensor metadata directly so it does not depend on which tensors appeared in a profiling run.

## 限制

1. CPU_Elastic 目前只有 graph-level compute timing，不能代表 per-op CPU matmul cost。
2. OpenCL `COMPUTE` profile 会 `clFinish()`，只能用于 cost calibration。
3. 当前还没有 activation boundary timing，mixed plan 的 switch cost 需要 Milestone 3/4 后继续补。
