# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-01 20:16:48 -0400

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
- cooldown thermal max: `0.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | 0 | 1200 | 1200.0 | 1201 | 1995.0 | 3439.2 | 4488.0 | 1792 | 4608 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | static-min | ok | 1815.24 | 1815.24 | 41.5/59.7 | 0 | 0.00 | 0 | 0/56/0/56 | 766669.02 | 29872 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | static-max | ok | 324.20 | 324.20 | 59.7/63.2 | 0 | 0.00 | 0 | 0/14/0/14 | 377022.36 | 16656 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | offline | ok | 911.10 | 1216.12 | 63.2/54.6 | 0 | 20.43 | 248 | 0/5084/0/5084 | 469216.36 | 18160 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | diff-tree-ideal | ok | 808.30 | 824.65 | 54.6/61.2 | 0 | 1179.70 | 16 | 2/412/0/412 | 546182.05 | 20359 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | online | ok | 752.42 | 760.70 | 61.2/56.6 | 7127.68 | 353.45 | 8 | 0/261/0/261 | 670706.81 | 25137 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | mru | remote_incomplete | n/a | n/a | 56.2/0.0 | 0 | 6929.87 | 301 | 0/0/0/0 | n/a | n/a | n/a |
