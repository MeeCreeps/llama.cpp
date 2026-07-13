# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-01 20:38:59 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `OLMoE-1B-7B-0125-Instruct-Q4_0.gguf`
- source window: `1200.0` seconds
- replay speedup: `1.0x`
- bench seconds: `1200.0`
- n_predict fallback: `96`
- budget bucket: `256` MiB
- offline table: `/data/local/tmp/hyzheng/elastic/plans_matrix10min`
- cooldown thermal max: `65.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | 0 | 1200 | 1200.0 | 1201 | 1995.0 | 3439.2 | 4488.0 | 1792 | 4608 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | mru | ok | 1000.61 | 1001.34 | 50.8/56.2 | 0 | 6744.26 | 299 | 0/0/0/0 | 677907.18 | 81921 | 0 |
