# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-30 02:21:28 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/gemma-4-E4B-it-Q8_0.gguf`
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
| trace_120_user_116_gemma_e4b_q8_cov40_90_10min_x1.csv | 0 | 180 | 180.0 | 181 | 6420.2 | 8522.7 | 9889.5 | 6400 | 9984 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_120_user_116_gemma_e4b_q8_cov40_90_10min_x1.csv | online | ok | 832.25 | 828.43 | 44.2/52.7 | 2178.18 | 948.41 | 3 | 0/192/192/192 | 74691.75 | 10692 | 8344 |
