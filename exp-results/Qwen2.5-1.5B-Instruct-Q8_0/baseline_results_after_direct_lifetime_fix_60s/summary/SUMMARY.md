# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-01 01:47:36 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `qwen2.5-1.5b-instruct-q8_0.gguf`
- source window: `60.0` seconds
- replay speedup: `1.0x`
- bench seconds: `60.0`
- n_predict fallback: `16`
- budget bucket: `128` MiB
- offline table: `/data/local/tmp/hyzheng/elastic/plans_matrix10min`
- cooldown thermal max: `0.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_410_user_116__qwen15_correctness_10min_x1.csv | 470 | 60 | 60.0 | 62 | 1100.0 | 1233.5 | 1358.1 | 1024 | 1408 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_410_user_116__qwen15_correctness_10min_x1.csv | static-max | no_perf | n/a | n/a | 37.2/59.7 | 0 | 0.00 | 0 | 0/0/0/0 | n/a | n/a | n/a |
