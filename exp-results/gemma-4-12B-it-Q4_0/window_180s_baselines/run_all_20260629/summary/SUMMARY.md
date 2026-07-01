# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-29 22:02:38 -0400

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
| trace_03_user_116_cap7300_10min_x1.csv | 0 | 180 | 180.0 | 181 | 5954.0 | 6769.7 | 7300.0 | 5888 | 7424 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_03_user_116_cap7300_10min_x1.csv | static-min | ok | 3193.22 | 3193.22 | 40.3/52.7 | 0 | 0.00 | 0 | 0/248/248/248 | 121567.72 | 22926 | 960 |
| trace_03_user_116_cap7300_10min_x1.csv | static-max | ok | 2514.30 | 2514.30 | 52.7/54.6 | 0 | 0.00 | 0 | 0/192/192/192 | 114558.63 | 18307 | 1200 |
| trace_03_user_116_cap7300_10min_x1.csv | offline | ok | 1044.27 | 1189.61 | 54.6/54.3 | 0 | 133.10 | 8 | 0/1835/1835/1835 | 87107.10 | 19138 | 0 |
| trace_03_user_116_cap7300_10min_x1.csv | online | ok | 3783.52 | 3397.62 | 54.6/64.3 | 18913.54 | 19156.81 | 3 | 0/691/691/691 | 141821.79 | 10311 | 752 |
| trace_03_user_116_cap7300_10min_x1.csv | diff-tree-ideal | ok | 2138.42 | 2236.72 | 65.5/58.9 | 0 | 2548.51 | 3 | 0/530/211/530 | 50456.61 | 7419 | 0 |
| trace_03_user_116_cap7300_10min_x1.csv | mru | ok | 1796.64 | 1796.59 | 58.9/57.7 | 0 | 202.07 | 6 | 0/0/0/0 | 89267.45 | 11145 | 0 |
