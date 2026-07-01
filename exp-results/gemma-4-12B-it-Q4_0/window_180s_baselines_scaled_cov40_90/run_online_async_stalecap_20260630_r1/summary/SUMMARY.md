# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-30 00:33:53 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/gemma-4-12B-it-Q4_0.gguf`
- source window: `180.0` seconds
- replay speedup: `1.0x`
- bench seconds: `180.0`
- n_predict fallback: `96`
- budget bucket: `256` MiB
- offline table: `/data/local/tmp/hyzheng/elastic/plans_matrix10min`
- cooldown thermal max: `0.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_100_user_116_gemma_cov40_90_10min_x1.csv | 0 | 180 | 180.0 | 181 | 6409.5 | 8508.5 | 9873.0 | 6400 | 9984 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_100_user_116_gemma_cov40_90_10min_x1.csv | online | ok | 1406.07 | 1333.01 | 44.9/51.9 | 41051.60 | 9557.05 | 2 | 0/115/115/115 | 83596.19 | 8260 | 0 |
