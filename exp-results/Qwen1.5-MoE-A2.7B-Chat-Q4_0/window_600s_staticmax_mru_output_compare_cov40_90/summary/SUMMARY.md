# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-30 13:59:16 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`
- source window: `600.0` seconds
- replay speedup: `1.0x`
- bench seconds: `600.0`
- n_predict fallback: `96`
- budget bucket: `256` MiB
- offline table: `/data/local/tmp/hyzheng/elastic/plans_matrix10min`
- cooldown thermal max: `0.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_210_user_116_qwen15moe_q4_0_cov40_90_10min_x1.csv | 0 | 600 | 600.0 | 601 | 3603.0 | 6028.0 | 8108.0 | 3584 | 8192 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_210_user_116_qwen15moe_q4_0_cov40_90_10min_x1.csv | static-max | ok | 586.30 | 586.30 | 36.0/56.6 | 0 | 0.00 | 0 | 0/10/0/10 | 225452.32 | 9224 | 0 |
| trace_210_user_116_qwen15moe_q4_0_cov40_90_10min_x1.csv | mru | ok | 1884.28 | 1889.77 | 56.6/55.0 | 0 | 784.30 | 27 | 0/0/0/0 | 286623.30 | 23557 | 0 |
