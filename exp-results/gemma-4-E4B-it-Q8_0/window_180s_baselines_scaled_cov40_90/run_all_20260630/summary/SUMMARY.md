# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-30 02:10:39 -0400

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
| trace_120_user_116_gemma_e4b_q8_cov40_90_10min_x1.csv | static-min | ok | 2110.64 | 2110.64 | 36.4/48.4 | 0 | 0.00 | 0 | 0/243/243/243 | 104542.44 | 24119 | 0 |
| trace_120_user_116_gemma_e4b_q8_cov40_90_10min_x1.csv | static-max | ok | 365.13 | 365.13 | 48.4/54.3 | 0 | 0.00 | 0 | 0/31/31/31 | 86622.01 | 13860 | 0 |
| trace_120_user_116_gemma_e4b_q8_cov40_90_10min_x1.csv | offline | ok | 1047.64 | 1134.92 | 54.3/51.9 | 0 | 145.08 | 17 | 0/1266/1266/1266 | 81949.25 | 11283 | 0 |
| trace_120_user_116_gemma_e4b_q8_cov40_90_10min_x1.csv | diff-tree-ideal | ok | 605.24 | 620.46 | 51.9/56.2 | 0 | 4594.30 | 15 | 0/1566/53/1566 | 41890.03 | 9361 | 0 |
| trace_120_user_116_gemma_e4b_q8_cov40_90_10min_x1.csv | online | ok | 874.24 | 831.56 | 56.2/52.3 | 42237.55 | 8867.08 | 2 | 0/128/128/128 | 77571.94 | 10939 | 8038 |
| trace_120_user_116_gemma_e4b_q8_cov40_90_10min_x1.csv | mru | ok | 594.78 | 593.62 | 52.3/51.9 | 0 | 668.24 | 20 | 0/0/0/0 | 69248.89 | 11386 | 0 |
