# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-01 12:58:59 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `OLMoE-1B-7B-0125-Instruct-Q4_0.gguf`
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
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | 0 | 180 | 180.0 | 181 | 2913.8 | 3867.8 | 4488.0 | 2816 | 4608 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | static-max | ok | 300.94 | 300.94 | 44.9/60.8 | 0 | 0.00 | 0 | 0/14/0/14 | 59790.25 | 2706 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | mru | ok | 824.90 | 827.65 | 60.8/53.5 | 0 | 1015.35 | 49 | 0/0/0/0 | 112394.39 | 14979 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | static-min | ok | 1075.48 | 1075.48 | 53.1/64.7 | 0 | 0.00 | 0 | 0/37/0/37 | 120255.31 | 5082 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | offline | ok | 689.19 | 892.85 | 64.7/61.6 | 0 | 18.44 | 41 | 0/840/0/840 | 86399.99 | 3620 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | diff-tree-ideal | ok | 616.46 | 632.93 | 61.6/64.3 | 0 | 274.77 | 4 | 1/125/0/125 | 100294.41 | 4081 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | online | ok | 696.54 | 706.91 | 64.3/57.0 | 1415.92 | 199.04 | 2 | 0/69/0/69 | 104968.67 | 4115 | 0 |
