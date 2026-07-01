# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-30 03:20:19 -0400

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
| trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42_10min_x1.csv | 0 | 180 | 180.0 | 181 | 5891.1 | 7820.2 | 9074.4 | 5888 | 9216 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42_10min_x1.csv | static-min | ok | 2292.23 | 2292.23 | 40.3/51.5 | 0 | 0.00 | 0 | 0/216/216/216 | 103956.88 | 22468 | 0 |
| trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42_10min_x1.csv | static-max | ok | 532.37 | 532.37 | 51.2/52.3 | 0 | 0.00 | 0 | 0/33/33/33 | 92807.59 | 14663 | 0 |
| trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42_10min_x1.csv | offline | ok | 1328.38 | 1403.45 | 52.7/52.3 | 0 | 134.08 | 9 | 0/753/753/753 | 77548.58 | 12888 | 0 |
| trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42_10min_x1.csv | diff-tree-ideal | ok | 677.34 | 685.24 | 52.3/56.6 | 0 | 2517.36 | 3 | 0/300/41/300 | 47462.25 | 11610 | 0 |
| trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42_10min_x1.csv | online | ok | 815.72 | 812.78 | 56.2/55.8 | 2441.32 | 766.52 | 2 | 0/154/154/154 | 77844.24 | 10768 | 0 |
| trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42_10min_x1.csv | mru | ok | 777.98 | 777.68 | 55.4/52.3 | 0 | 344.37 | 12 | 0/0/0/0 | 74293.98 | 12121 | 0 |
