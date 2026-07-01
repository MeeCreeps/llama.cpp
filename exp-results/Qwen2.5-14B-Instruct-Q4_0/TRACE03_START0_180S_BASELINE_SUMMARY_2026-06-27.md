# Qwen2.5-14B-Instruct-Q4_0 Trace03 Start0 180s Baseline Summary

Generated: 2026-06-27

## Setup

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `qwen2.5-14b-instruct-q4_0.gguf`
- trace: `trace_03_user_116_cap7300.csv`
- source window: start `0s`, span `180s`
- window stats: min/mean/max `5954.0 / 6769.7 / 7300.0 MiB`
- buckets: min `5888 MiB`, max `7424 MiB`
- placement family: `gpu,disk_gpu`
- timed decode: `180s`
- unified budget overhead: `KV=512 MiB`, `misc=256 MiB`, `pinned=410 MiB`, `safety=64 MiB`
- full-residency requirement: `8329.5 MiB` (`weights=7087.5`, `KV=512`, `misc=256`, `pinned=410`, `safety=64`)
- cost model: freshly reprofiled on `2026-06-27` from current `llama-cli` and this 14B Q4 model
- profile rows: opencl_compute `40343`, opencl_stage `32706`, opencl_reload `30481`, stage_records `702`

## Headline Baselines

| method | source artifact | status | raw ms/token | exec ms/token | eval runs | apply count | planned evict/load/xfer/xform | direct read MB | direct read MB/token | online calls/failures | GPU before/after C |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| static-min | `trace03_cap7300_start0_180s_all_reprofile_20260627` | ok | 2943.32 | 2943.32 | 62 | 0 | 0/69/69/69 | 296036.7 | 4774.79 | n/a/n/a | 33.7/46.5 |
| static-max fair | `trace03_cap7300_start0_180s_static_max_fair_20260627` | ok | 596.52 | 596.52 | 302 | 0 | 0/31/31/31 | 276825.9 | 916.64 | n/a/n/a | 43.8/53.9 |
| offline | `trace03_cap7300_start0_180s_all_reprofile_20260627` | ok | 1151.93 | 1213.66 | 149 | 8 | 0/369/369/369 | 267143.9 | 1792.91 | n/a/n/a | 52.3/52.3 |
| diff-tree-ideal fair | `trace03_cap7300_start0_180s_diff_tree_fair_20260627` | ok | 646.60 | 664.99 | 271 | 3 | 0/174/35/174 | 105960.9 | 391.00 | 3/0 | 38.0/55.0 |
| online remote CP | `trace03_cap7300_start0_180s_all_reprofile_20260627` | ok | 816.56 | 733.78 | 205 | 4 | 0/185/185/185 | 346507.0 | 1690.28 | 4/0 | 53.9/71.3 |
| mru runtime-cache kv512 | `trace03_cap7300_start0_180s_all_reprofile_20260627` | ok | 1485.28 | 1484.51 | 121 | 10 | 0/0/0/0 | 207923.9 | 1718.38 | 10/0 | 70.9/50.8 |

## Notes

- `static-max fair` is the fastest result under the foreground/per-graph-evict comparison path: `596.52 ms/token`.
- `diff-tree-ideal fair` fixes transition replay while keeping per-graph DISK-weight eviction enabled: `646.60 ms/token`, 3 online calls, 0 failures. Its direct read/token is `391.00 MB`, so it is no longer using the non-fair retained-resident upper-bound path.
- The earlier diff-tree top-k4 run replayed transition anchors every graph, causing `6718` stage loads and `1551.19 ms/token`; the one-shot debug run removed that replay but retained too much resident state (`348.60 ms/token`), so it is not used as a headline baseline.
- `offline` is slower than online because it applies 8 plans and schedules 369 load/xfer/xform movements.
- `online remote CP` remains valid but slower than fixed diff-tree in this run: `816.56 ms/token`, 4 online calls, 0 failures.
- `mru` uses the unified `kv=512` budget accounting and performs real direct reads; it is slower than online/offline here.

## Source Files

- final merged baseline CSV: `window_180s_baselines/baseline_results.csv`
- exact 180s trace window CSV: `window_180s_baselines/trace03_start0_180s.csv`
- full trace plot: `trace_plots/trace_03_user_116_cap7300.svg`
- 180s window trace plot: `window_180s_baselines/trace03_start0_180s.svg`
- all-baseline first pass raw results: `window_180s_baselines/raw_results/all_baselines_firstpass_results.csv`
- fair static-max raw results: `window_180s_baselines/raw_results/static_max_fair_results.csv`
- fair diff-tree raw results: `window_180s_baselines/raw_results/diff_tree_fair_results.csv`
- reprofile manifest: `runtime/plan/generated/Qwen2.5-14B-Instruct-Q4_0_reprofile_20260627/manifest.json`
