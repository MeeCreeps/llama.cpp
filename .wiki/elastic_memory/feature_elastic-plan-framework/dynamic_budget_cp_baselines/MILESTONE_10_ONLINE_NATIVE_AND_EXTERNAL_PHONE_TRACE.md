# Milestone 10: Online Native And External Phone Trace

## Native Greedy Command

Clean completed run used `-n 30` to avoid the repeated final 1500MiB hang seen with `-n 32`.
The budget trace used here is synthetic and was hand-written to force budget switching; it is not a real phone memory availability trace.

```sh
cd /data/local/tmp/hyzheng/elastic

LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
GGML_OPENCL_ELASTIC=1 \
GGML_ELASTIC_TIMING=1 \
GGML_ELASTIC_STAGE_DETAIL=1 \
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/hyzheng/elastic/budget_trace.csv \
LLAMA_ELASTIC_ONLINE=1 \
LLAMA_ELASTIC_ONLINE_MODE=native-greedy \
LLAMA_ELASTIC_MODEL_META=/data/local/tmp/hyzheng/elastic/model_meta.json \
LLAMA_ELASTIC_COST_DIR=/data/local/tmp/hyzheng/elastic/profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 \
LLAMA_ELASTIC_ONLINE_WORK_DIR=/data/local/tmp/hyzheng/elastic/online \
./llama-cli -m /data/local/tmp/hyzheng/elastic/Llama-3.2-3B-Instruct-q4_0.gguf \
  -p hello -n 30 -ngl 99 --seed 42 --temp 0 --no-warmup -no-cnv
```

## Native Greedy Results

```text
prompt eval: 328.84 ms/token
eval:        430.48 ms/token
total:       41593.25 ms / 31 tokens
online calls=30 failures=0
budget switches/provider calls=30
provider_get_ms: n=30 median=844.369 mean=842.146 min=670.627 max=914.388
online gen_ms:   n=30 median=842.556 mean=840.299 min=668.788 max=912.494
apply_ms:        n=30 median=0.092 mean=0.092 min=0.085 max=0.105
```

Stage/direct read:

```text
stage load:     calls=11 ok=11 total=103.69 ms avg=9.426 ms MB=148.5 MB/s=1432.2
stage transfer: calls=11 ok=11 total=0.01 ms avg=0.001 ms MB=148.5
stage xform:    calls=11 ok=3  total=10.21 ms avg=0.928 ms MB=40.5 MB/s=3968.6
direct O_DIRECT read: calls=2989 ok=2989 fail=0 total=8225.41 ms avg=2.752 ms MB=20146.6 MB/s=2449.3
```

Runtime summary:

```text
anchor requests=504 hits=112 duplicate=166 fired=18 load=6 transfer=6 xform=6 failures=5
```

## External CP-SAT

Skipped on this phone because no `python3` is available:

```sh
adb shell 'which python3; python3 --version'
```

Result:

```text
no python3 in PATH
```

## Issues And Fixes

1. `LLAMA_ELASTIC_ONLINE_WORK_DIR` creation used `system("mkdir -p ...")`. On Android, this inherited the test `LD_LIBRARY_PATH` and failed with:

```text
CANNOT LINK EXECUTABLE "mkdir": "/vendor/lib64/libcrypto.so" phdr mmap failed: Permission denied
```

Fixed by replacing this call with an in-process recursive `mkdir` helper in `tools/main/main.cpp`.

2. A clean `-n 32` online run hung after 31 provider calls at the repeated low-budget tail. `debuggerd -b` could not be used because root is required. The process was terminated and the partial log was saved.

3. Native online provider overhead is dominated by plan generation/JSON load. `provider_get_ms` tracks `gen_ms` closely at about 0.84s/call. This is much larger than `apply_ms`.

4. Low-budget stage execution still has partial xform failures: `stage xform calls=11 ok=3`, with anchor `failures=5`. Runtime fallback allowed generation to complete.

## Artifacts

```text
.wiki/.../artifacts/logs/run_online_native_greedy_n30.log
.wiki/.../artifacts/logs/run_online_native_greedy_hung_n32.log
/data/local/tmp/hyzheng/elastic/online/*.json
```
