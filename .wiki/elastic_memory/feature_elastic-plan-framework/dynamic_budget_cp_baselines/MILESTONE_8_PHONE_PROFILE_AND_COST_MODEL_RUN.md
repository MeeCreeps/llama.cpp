# Milestone 8: Phone Profile And Cost Model Run

## Target

```text
ADB_SERIAL=172.20.115.151:5555
device=OnePlus 15 / CPH2749 / OP611FL1 / Android 16 / canoe
DEV=/data/local/tmp/hyzheng/elastic
commit=a10bd13ed
```

`op13` was requested, but no OnePlus 13 device was visible from this server.

## Commands

OpenCL profile:

```sh
cd /data/local/tmp/hyzheng/elastic

LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
GGML_OPENCL_ELASTIC=1 \
GGML_ELASTIC_TIMING=1 \
GGML_ELASTIC_STAGE_DETAIL=1 \
GGML_ELASTIC_PROFILE_CSV=/data/local/tmp/hyzheng/elastic/profile_opencl_1500.csv \
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/const_1500_budget.csv \
./llama-cli -m /data/local/tmp/hyzheng/elastic/Llama-3.2-3B-Instruct-q4_0.gguf \
  -p hello -n 16 -ngl 99 --seed 42 --temp 0 --no-warmup -no-cnv
```

CPU_Elastic profile:

```sh
cd /data/local/tmp/hyzheng/elastic

LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
GGML_OPENCL_ELASTIC=1 \
GGML_ELASTIC_TIMING=1 \
GGML_ELASTIC_STAGE_DETAIL=1 \
GGML_ELASTIC_PROFILE_CSV=/data/local/tmp/hyzheng/elastic/profile_cpu_elastic_1500.csv \
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/const_1500_budget.csv \
./llama-cli -m /data/local/tmp/hyzheng/elastic/Llama-3.2-3B-Instruct-q4_0.gguf \
  -p hello -n 8 -ngl 99 --device CPU_Elastic --seed 42 --temp 0 --no-warmup -no-cnv
```

Host cost/model meta:

```sh
python3 runtime/plan/build_cost_model.py \
  artifacts/profile_opencl_1500.csv \
  artifacts/profile_cpu_elastic_1500.csv \
  --device android-opencl \
  --model Llama-3.2-3B-Instruct-q4_0 \
  --out-dir runtime/plan/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0

python3 runtime/plan/build_model_meta_from_profile.py \
  artifacts/profile_opencl_1500.csv \
  artifacts/profile_cpu_elastic_1500.csv \
  --out runtime/plan/model_meta/Llama-3.2-3B-Instruct-q4_0.weights_ops.json
```

## Results

OpenCL profile:

```text
prompt eval: 352.34 ms/token
eval:        487.06 ms/token
direct O_DIRECT read: calls=1605 ok=1605 fail=0 total=4351.04 ms avg=2.711 ms MB=11008.7 MB/s=2530.1
```

CPU_Elastic profile:

```text
prompt eval: 126.27 ms/token
eval:        592.80 ms/token
direct_read: calls=259 ok=259 fail=0 total=3009.25 ms MB=4992.6 MB/s=1659.1
```

Cost/model output:

```text
input_rows=8624
stage_records=149
op_records=771
boundary_records=0
model_meta weights=112 ops=112
```

## Fixes

1. Fixed `runtime/elastic_profile_writer.cpp`: CSV writer emitted an extra comma after `quant`, shifting numeric columns.
2. Fixed `runtime/plan/build_model_meta_from_profile.py`: model meta now uses only weight stage/reload records. It no longer treats compute temporaries such as `Qcur-*`, `cache_*`, or `ffn_out-*` as model weights.

## Artifacts

```text
.wiki/.../artifacts/profile_opencl_1500.csv
.wiki/.../artifacts/profile_cpu_elastic_1500.csv
.wiki/.../artifacts/logs/run_profile_opencl.log
.wiki/.../artifacts/logs/run_profile_cpu_elastic.log
runtime/plan/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0/
runtime/plan/model_meta/Llama-3.2-3B-Instruct-q4_0.weights_ops.json
```
