# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-13 13:05:14 -0400

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
- cooldown thermal max: `45.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | 0 | 600 | 600.0 | 601 | 5405.0 | 7102.2 | 8558.0 | 5376 | 8704 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | static-max | ok | 163.21 | 163.21172347968366 | 35.0/60.8 | 0 | 0.0 | 0 | 0/11/1/11 | 228.84 | 81 | 0 |
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | diff-tree-ideal | ok | 221.8 | 221.73479273422566 | 44.9/57.7 | 0 | 760.141 | 2 | 1/10/0/10 | 1421.57 | 140 | 0 |
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | offline | ok | 363.1 | 930.8167679127725 | 44.9/0.0 | 0 | 117.35799999999993 | 130 | 1655/2217/0/2217 | 260971.82 | 3439 | 0 |
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | online | remote_incomplete | n/a | n/a | 44.9/0.0 | 3806.69 | 454.15 | 4 | 90/131/0/131 | n/a | n/a | n/a |
