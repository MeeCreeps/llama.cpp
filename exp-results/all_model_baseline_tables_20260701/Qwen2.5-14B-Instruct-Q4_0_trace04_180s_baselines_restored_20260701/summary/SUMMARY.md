# Dynamic Budget 10-Min Baseline Matrix

Generated: 2026-07-01 12:34:14 -0400

## Configuration

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `qwen2.5-14b-instruct-q4_0.gguf`
- source window: `180.0` seconds
- replay speedup: `1.0x`
- bench seconds: `180.0`
- n_predict fallback: `96`
- budget bucket: `128` MiB
- offline table: `/data/local/tmp/hyzheng/elastic/plans_matrix10min`
- cooldown thermal max: `0.0` C

## Trace Windows

| trace | source start s | source span s | replay span s | rows | min MiB | mean MiB | max MiB | min bucket | max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_04_user_68_10min_x1.csv | 0 | 180 | 180.0 | 181 | 6639.6 | 6743.1 | 6883.6 | 6528 | 6912 |

## Results

`raw ms/token` is llama's eval timer. `exec ms/token` subtracts provider_get/query time but keeps plan apply/movement time.

| trace | method | status | raw ms/token | exec ms/token | thermal before/after C | remote wall ms | provider get ms | apply count | planned evict/load/xfer/prepare | direct read ms | direct read calls | failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| trace_04_user_68_10min_x1.csv | static-max | ok | 757.28 | 757.28 | 38.7/50.4 | 0 | 0.00 | 0 | 0/69/1/69 | 87213.99 | 23401 | 0 |
| trace_04_user_68_10min_x1.csv | mru | ok | 974.46 | 973.37 | 49.6/51.5 | 0 | 600.34 | 19 | 0/0/0/0 | 92845.67 | 10447 | 0 |
| trace_04_user_68_10min_x1.csv | static-min | ok | 1059.19 | 1059.19 | 51.2/53.1 | 0 | 0.00 | 0 | 0/93/1/93 | 93451.68 | 22813 | 0 |
| trace_04_user_68_10min_x1.csv | offline | ok | 839.36 | 915.74 | 53.1/53.1 | 0 | 37.94 | 17 | 0/1231/17/1231 | 80255.84 | 20887 | 0 |
| trace_04_user_68_10min_x1.csv | diff-tree-ideal | ok | 615.47 | 642.93 | 53.1/58.5 | 0 | 1597.05 | 13 | 0/1080/1/1080 | 74873.30 | 17461 | 0 |
| trace_04_user_68_10min_x1.csv | online | ok | 875.46 | 886.29 | 57.7/53.1 | 2966.05 | 495.95 | 3 | 0/336/4/336 | 89781.68 | 13930 | 0 |
