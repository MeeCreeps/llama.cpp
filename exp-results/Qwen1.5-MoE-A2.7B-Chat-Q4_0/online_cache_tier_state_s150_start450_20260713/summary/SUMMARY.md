# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-13 13:53:30 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`
- source window: `150.0` seconds
- replay speedup: `1.0x`
- bench seconds: `150.0`
- n_predict fallback: `96`
- budget bucket: `256` MiB
- offline table: `/data/local/tmp/hyzheng/elastic/plans_matrix10min`
- cooldown thermal max: `45.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | 450 | 150 | 150.0 | 152 | 5405.0 | 6350.4 | 6884.6 | 5376 | 6912 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_220_user_116_qwen15moe_q4_0_cov60_95_10min_x1.csv | online | ok | 423.77 | 464.19 | 44.6/77.9 | 2259.27 | 445.18 | 3 | 100/152/0/152 | 11339.06 | 243 | 0 |
