# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-12 21:29:31 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`
- source window: `600.0` seconds
- replay speedup: `1.0x`
- bench seconds: `60.0`
- n_predict fallback: `96`
- budget bucket: `256` MiB
- offline table: `/data/local/tmp/hyzheng/elastic/plans_matrix10min`
- cooldown thermal max: `0.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | 0 | 600 | 600.0 | 601 | 5405.0 | 7102.2 | 8558.0 | 5376 | 8704 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | static-max | ok | 95.27 | 95.27 | 41.8/62.0 | 0 | 0.00 | 0 | 0/11/1/11 | 515.66 | 83 | 0 |
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | mru | ok | 971.81 | 973.76 | 61.2/51.2 | 0 | 548.10 | 15 | 0/0/0/0 | 36861.38 | 3786 | 0 |
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | diff-tree-ideal | ok | 740.24 | 730.09 | 50.8/55.4 | 0 | 1193.33 | 1 | 0/10/0/10 | 15640.20 | 2281 | 0 |
