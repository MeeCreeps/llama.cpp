# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-13 09:29:43 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`
- source window: `600.0` seconds
- replay speedup: `1.0x`
- bench seconds: `30.0`
- n_predict fallback: `96`
- budget bucket: `256` MiB
- offline table: `/data/local/tmp/hyzheng/elastic/plans_matrix10min`
- cooldown thermal max: `45.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | 450 | 600 | 600.0 | 602 | 5405.0 | 7253.0 | 8558.0 | 5376 | 8704 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | diff-tree-ideal | ok | 315.91 | 312.74 | 41.5/57.7 | 0 | 729.54 | 2 | 1/13/1/13 | 5258.31 | 2485 | 0 |
