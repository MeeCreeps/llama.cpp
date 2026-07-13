# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-12 22:47:12 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`
- source window: `600.0` seconds
- replay speedup: `1.0x`
- bench seconds: `120.0`
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
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | diff-tree-ideal | ok | 228.92 | 228.25 | 39.5/60.1 | 0 | 749.52 | 2 | 1/11/1/11 | 1456.83 | 185 | 0 |
