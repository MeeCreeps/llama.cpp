# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-12 21:09:53 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`
- source window: `600.0` seconds
- replay speedup: `1.0x`
- bench seconds: `90.0`
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
| trace_210_user_116_qwen15moe_q4_0_cov40_90_10min_x1.csv | static-max | ok | 450.51 | 450.51 | 57.7/59.3 | 0 | 0.00 | 0 | 0/9/0/9 | 45861.39 | 1420 | 0 |
