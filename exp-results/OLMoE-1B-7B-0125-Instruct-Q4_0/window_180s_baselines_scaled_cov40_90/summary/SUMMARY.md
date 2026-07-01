# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-30 09:07:39 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/OLMoE-1B-7B-0125-Instruct-Q4_0.gguf`
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
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | static-min | ok | 1249.35 | 1249.35 | 40.3/59.7 | 0 | 0.00 | 0 | 0/37/0/37 | 119658.33 | 6010 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | static-max | ok | 365.20 | 365.20 | 59.7/60.5 | 0 | 0.00 | 0 | 0/14/0/14 | 94612.85 | 7908 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | offline | ok | 827.33 | 868.94 | 60.1/61.6 | 0 | 45.41 | 8 | 0/178/0/178 | 87027.68 | 3399 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | diff-tree-ideal | ok | 843.01 | 867.58 | 61.6/60.1 | 0 | 490.17 | 6 | 0/137/0/137 | 78547.37 | 3063 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | online | ok | 761.56 | 759.54 | 60.1/55.4 | 1147.71 | 649.13 | 2 | 0/49/0/49 | 56632.37 | 2974 | 0 |
| trace_170_user_116_olmoe_q4_0_cov40_90_10min_x1.csv | mru | ok | 635.51 | 635.65 | 55.0/62.4 | 0 | 161.80 | 10 | 0/0/0/0 | 100556.53 | 13041 | 0 |
