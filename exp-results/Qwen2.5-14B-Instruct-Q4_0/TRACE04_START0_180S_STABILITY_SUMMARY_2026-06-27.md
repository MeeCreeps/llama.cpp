# Qwen2.5-14B-Instruct-Q4_0 Trace04 Start0 180s Stability Summary

Generated: 2026-06-27

## Setup

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `qwen2.5-14b-instruct-q4_0.gguf`
- trace: `trace_04_user_68.csv`
- source window: start `0s`, span `180s`
- window stats: min/mean/max `6639.6 / 6743.1 / 6883.6 MiB`
- buckets: min `6400 MiB`, max `6912 MiB`
- placement family: `gpu,disk_gpu`
- timed decode: `180s`
- unified budget overhead: `KV=512 MiB`, `misc=256 MiB`, `pinned=410 MiB`, `safety=64 MiB`
- full-residency requirement: `8329.5 MiB` (`weights=7087.5`, `KV=512`, `misc=256`, `pinned=410`, `safety=64`)
- cost model: `Qwen2.5-14B-Instruct-Q4_0_reprofile_20260627`

## Headline Baselines

| method | source artifact | status | raw ms/token | exec ms/token | eval runs | apply count | planned evict/load/xfer/xform | direct read MB | direct read MB/token | online calls/failures | GPU before/after C |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| static-min | `trace04_start0_180s_core_20260627` | ok | 1780.24 | 1780.24 | 102 | 0 | 0/110/1/110 | 233228.0 | 2286.55 | n/a/n/a | 37.2/49.2 |
| static-max fair | `trace04_start0_180s_static_max_fair_20260627` | ok | 1337.73 | 1337.73 | 135 | 0 | 0/94/1/94 | 174425.6 | 1292.04 | n/a/n/a | 46.5/51.2 |
| offline | `trace04_start0_180s_offline_rerun_20260627` | ok | 1282.20 | 1318.99 | 138 | 3 | 0/260/3/260 | 167997.7 | 1217.37 | n/a/n/a | 40.3/50.8 |
| diff-tree-ideal fair | `trace04_start0_180s_diff_tree_fair_20260627` | ok | 1124.86 | 1162.95 | 156 | 3 | 0/198/1/198 | 106484.1 | 682.59 | 3/0 | 44.6/56.2 |
| online remote CP | `trace04_start0_180s_online_rerun_20260627` | ok | 1229.31 | 1140.75 | 139 | 3 | 0/150/150/150 | 239355.0 | 1721.98 | 3/0 | 43.4/63.6 |
| mru runtime-cache kv512 | `trace04_start0_180s_mru_20260627` | ok | 1612.44 | 1611.96 | 112 | 3 | 0/0/0/0 | 191684.5 | 1711.47 | 3/0 | 52.3/50.8 |

## Stability Notes

- `diff-tree-ideal fair` remains the best dynamic method on this second trace: `1124.86 ms/token`, 3 calls, 0 failures.
- The absolute numbers are slower than trace03 because this window is a tighter mid-budget band (`6400-6912` buckets), so every method has more recurring disk pressure.
- The trend is still sane: `diff-tree` < `online` < `offline` < `mru` / `static-min` by raw ms/token. `static-max fair` is close to offline but no longer dominates because the max bucket is only `6912 MiB`, not `7424 MiB`.
- A first continuous multi-method run made the phone drop from ADB after `offline`; rerunning methods one-by-one from cooler thermal state produced rc=0 for all headline rows. The table uses only the successful single-method reruns where needed.
- `online remote CP` had one hot-state rc=137 attempt and then passed after cooldown. This suggests online is more thermally/memory sensitive than diff-tree on trace04, even though it is functionally correct after cooldown.

## Cross-Trace Trend

| method | trace03 raw ms/token | trace04 raw ms/token | trend |
|---|---:|---:|---|
| static-min | 2943.32 | 1780.24 | faster on trace04 because the min bucket is higher |
| static-max fair | 596.52 | 1337.73 | slower on trace04 because max bucket is lower |
| offline | 1151.93 | 1282.20 | similar order, slightly slower |
| diff-tree-ideal fair | 646.60 | 1124.86 | still best dynamic method |
| online remote CP | 816.56 | 1229.31 | still better than offline by exec ms/token, but thermally sensitive |
| mru runtime-cache kv512 | 1485.28 | 1612.44 | consistently slower than planner methods |

## Source Files

- final merged baseline CSV: `trace04_start0_180s_baselines/baseline_results.csv`
- exact 180s trace window CSV: `trace04_start0_180s_baselines/trace04_start0_180s.csv`
- full trace plot: `trace04_plots/trace_04_user_68.svg`
- 180s window trace plot: `trace04_start0_180s_baselines/trace04_start0_180s.svg`
- raw result CSVs: `trace04_start0_180s_baselines/raw_results/`
