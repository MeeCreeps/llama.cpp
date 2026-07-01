# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-29 23:35:29 -0400

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
| trace_100_user_116_gemma_cov40_90_10min_x1.csv | static-min | ok | 3060.01 | 3060.01 | 41.8/53.1 | 0 | 0.00 | 0 | 0/240/240/240 | 118313.07 | 21913 | 992 |
| trace_100_user_116_gemma_cov40_90_10min_x1.csv | static-max | ok | 625.67 | 625.67 | 53.1/53.5 | 0 | 0.00 | 0 | 0/29/29/29 | 107716.16 | 11349 | 0 |
| trace_100_user_116_gemma_cov40_90_10min_x1.csv | offline | ok | 222.12 | 265.62 | 53.1/51.5 | 0 | 190.67 | 20 | 0/1562/1562/1562 | 22992.67 | 2992 | 0 |
| trace_100_user_116_gemma_cov40_90_10min_x1.csv | online | timeout | n/a | n/a | 51.5/47.3 | 29282.54 | 29547.03 | 5 | 0/493/493/493 | n/a | n/a | n/a |
| trace_100_user_116_gemma_cov40_90_10min_x1.csv | diff-tree-ideal | ok | 969.95 | 999.78 | 49.2/56.6 | 0 | 2769.26 | 7 | 0/809/44/809 | 55234.58 | 9864 | 0 |
| trace_100_user_116_gemma_cov40_90_10min_x1.csv | mru | ok | 1528.89 | 1529.08 | 56.6/53.1 | 0 | 398.20 | 12 | 0/0/0/0 | 95870.65 | 11281 | 0 |
