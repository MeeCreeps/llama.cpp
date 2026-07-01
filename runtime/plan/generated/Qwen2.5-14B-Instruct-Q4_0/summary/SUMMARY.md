# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-06-24 15:13:07 -0400

## Configuration

- device: `5ae7a43d`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `qwen2.5-14b-instruct-q4_0.gguf`
- source window: `30.0` seconds
- replay speedup: `1.0x`
- bench seconds: `0.0`
- n_predict fallback: `1`
- budget bucket: `256` MiB
- offline table: `/data/local/tmp/hyzheng/elastic/plans_matrix10min`
- cooldown thermal max: `0.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_01_user_147_10min_x1.csv | 290 | 30 | 30.0 | 32 | 3752.1 | 4179.9 | 4477.2 | 3584 | 4608 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_01_user_147_10min_x1.csv | static-max | no_perf | n/a | n/a | 42.2/0.0 | 0 | 0.00 | 0 | 0/0/0/0 | n/a | n/a | n/a |
