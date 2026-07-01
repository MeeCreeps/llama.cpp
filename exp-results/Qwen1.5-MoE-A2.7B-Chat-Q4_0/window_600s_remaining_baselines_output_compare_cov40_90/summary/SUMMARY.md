# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-30 16:06:13 -0400

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
| trace_210_user_116_qwen15moe_q4_0_cov40_90_10min_x1.csv | static-min | ok | 3905.29 | 3905.29 | 36.0/56.2 | 0 | 0.00 | 0 | 0/69/0/69 | 277843.21 | 10642 | 0 |
| trace_210_user_116_qwen15moe_q4_0_cov40_90_10min_x1.csv | offline | ok | 2139.88 | 2364.27 | 56.2/60.8 | 0 | 283.17 | 22 | 1/817/0/817 | 261168.91 | 9901 | 0 |
| trace_210_user_116_qwen15moe_q4_0_cov40_90_10min_x1.csv | diff-tree-ideal | ok | 2012.97 | 2129.17 | 60.8/64.0 | 0 | 2725.75 | 12 | 0/461/0/461 | 256962.77 | 7652 | 0 |
| trace_210_user_116_qwen15moe_q4_0_cov40_90_10min_x1.csv | online | ok | 2338.05 | 2337.81 | 63.6/60.1 | 4130.84 | 736.32 | 4 | 0/185/0/185 | 155059.80 | 7940 | 0 |
