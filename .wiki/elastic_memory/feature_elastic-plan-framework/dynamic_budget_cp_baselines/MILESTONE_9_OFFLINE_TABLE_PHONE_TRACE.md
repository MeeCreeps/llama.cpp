# Milestone 9: Offline Table Phone Trace

## Setup

```text
DEV=/data/local/tmp/hyzheng/elastic
model=Llama-3.2-3B-Instruct-q4_0.gguf
budget_trace=/data/local/tmp/hyzheng/elastic/budget_trace.csv
plans=/data/local/tmp/hyzheng/elastic/plans_dynamic_offline
```

`budget_trace.csv` is a synthetic BudgetWatcher input, not a real sampled phone memory trace. It was used to force budget switches while validating provider/apply/stage behavior.

Offline table generation:

```sh
python3 runtime/plan/build_offline_budget_table.py \
  --model-meta runtime/plan/model_meta/Llama-3.2-3B-Instruct-q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 \
  --out-dir /tmp/plans_dynamic_offline \
  --budgets 1500,1800,2200,3000
```

Generated plans:

```text
1500MiB: weights=112 timeline=3 loc={gpu:111,disk:1}
1800MiB: weights=112 timeline=0 loc={gpu:112}
2200MiB: weights=112 timeline=0 loc={gpu:112}
3000MiB: weights=112 timeline=0 loc={gpu:112}
```

## Phone Command

```sh
cd /data/local/tmp/hyzheng/elastic

LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
GGML_OPENCL_ELASTIC=1 \
GGML_ELASTIC_TIMING=1 \
GGML_ELASTIC_STAGE_DETAIL=1 \
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/budget_trace.csv \
LLAMA_ELASTIC_DIR=/data/local/tmp/hyzheng/elastic/plans_dynamic_offline \
./llama-cli -m /data/local/tmp/hyzheng/elastic/Llama-3.2-3B-Instruct-q4_0.gguf \
  -p hello -n 32 -ngl 99 --seed 42 --temp 0 --no-warmup -no-cnv
```

## Results

```text
prompt eval: 424.59 ms/token
eval:        396.38 ms/token
total:       16380.19 ms / 33 tokens
budget switches: 3
budgets seen: 1680, 1861, 2273 MiB
provider_get_ms: n=3 median=5.697 mean=4.520 min=1.953 max=5.910
apply_ms:        n=3 median=0.268 mean=10.484 min=0.265 max=30.918
```

Stage/direct read:

```text
stage load:     calls=0 ok=0 total=0.00 ms
stage transfer: calls=0 ok=0 total=0.00 ms
stage xform:    calls=1 ok=1 total=0.00 ms avg=0.001 ms MB=308.2
direct O_DIRECT read: calls=3204 ok=3204 fail=0 total=8430.56 ms avg=2.631 ms MB=21682.2 MB/s=2571.9
```

Runtime summary:

```text
anchor requests=9576 hits=112 duplicate=5170 fired=0 load=0 transfer=0 xform=0 failures=0
```

## Notes

The `-n 32` run lasted about 16.4 seconds, so it covered the rising budget section only and did not reach the later falling trace entries.

Artifact:

```text
.wiki/.../artifacts/logs/run_offline_table.log
```
