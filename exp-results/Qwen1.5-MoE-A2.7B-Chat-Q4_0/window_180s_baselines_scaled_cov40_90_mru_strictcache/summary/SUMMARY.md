# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-30 11:50:11 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`
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
| trace_210_user_116_qwen15moe_q4_0_cov40_90_10min_x1.csv | 0 | 180 | 180.0 | 181 | 5263.3 | 6987.3 | 8108.0 | 5120 | 8192 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_210_user_116_qwen15moe_q4_0_cov40_90_10min_x1.csv | mru | ok | 1062.45 | 1065.70 | 36.0/57.7 | 0 | 179.67 | 9 | 0/0/0/0 | 98058.02 | 7261 | 0 |
