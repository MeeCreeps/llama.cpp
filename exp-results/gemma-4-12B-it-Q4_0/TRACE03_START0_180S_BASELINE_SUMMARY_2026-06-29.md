# gemma-4-12B-it-Q4_0 Trace03 Start0 180s Baseline Summary

Generated: 2026-06-29

## Setup

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/gemma-4-12B-it-Q4_0.gguf`
- trace: `trace_03_user_116_cap7300.csv`
- source window: start `0s`, span `180s`
- window stats: min/mean/max `5954.0 / 6769.7 / 7300.0 MiB`
- buckets: min `5888 MiB`, max `7424 MiB`
- placement family: `gpu,disk_gpu`
- timed decode: `180s`
- context/batch/thread settings: `-c 4096 -b 32 -ub 32 -t 8 -ngl 99 -fa on -no-cnv`
- unified budget overhead: `KV=512 MiB`, `misc=256 MiB`, `pinned=410 MiB`, `safety=64 MiB`
- full-residency requirement: `10969.8 MiB` (`weights=9727.8`, `KV=512`, `misc=256`, `pinned=410`, `safety=64`)
- cost model: freshly reprofiled on `2026-06-29` from current `llama-cli` and this Gemma4 12B Q4 model
- profile rows: opencl_compute `48513`, opencl_stage `39874`, opencl_reload `39060`, cpu_compute `24322`, cpu_stage `14996`, stage_records `967`

## Headline Baselines

| method | source artifact | status | raw ms/token | exec ms/token | eval runs | apply count | planned evict/load/xfer/xform | direct read MB | direct read MB/token | online calls/failures | anchor failures | GPU before/after C |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| static-min | `run_all_20260629` | ok+anchor_failures | 3193.22 | 3193.22 | 57 | 0 | 0/248/248/248 | 290336.1 | 5093.62 | n/a/n/a | 960 | 39.1/51.9 |
| static-max | `run_all_20260629` | ok+anchor_failures | 2514.30 | 2514.30 | 72 | 0 | 0/192/192/192 | 288138.5 | 4001.92 | n/a/n/a | 1200 | 51.2/53.1 |
| offline | `run_all_20260629` | ok | 1044.27 | 1189.61 | 156 | 8 | 0/1835/1835/1835 | 197174.0 | 1263.94 | n/a/n/a | 0 | 52.7/53.5 |
| online | `run_all_20260629` | ok+anchor_failures | 3783.52 | 3397.62 | 44 | 3 | 0/691/691/691 | 151689.4 | 3447.49 | 3/0 | 752 | 52.7/58.5 |
| diff-tree-ideal | `run_all_20260629` | ok | 2138.42 | 2236.72 | 80 | 3 | 0/530/211/530 | 102308.9 | 1278.86 | 3/0 | 0 | 58.5/57.4 |
| mru | `run_all_20260629` | ok | 1796.64 | 1796.59 | 101 | 6 | 0/0/0/0 | 205568.2 | 2035.33 | 6/0 | 0 | 56.6/57.0 |

## Notes

- All six runs completed with process `rc=0` on the phone.
- This first run is diagnostic, not the final headline table: the original
  trace is too memory-constrained for Gemma4 and several methods report anchor
  failures.
- `offline` is the fastest headline result in this first Gemma4 run: `1044.27 ms/token` raw, `1189.61 ms/token` exec, with 8 plan applies.
- `mru` is slower than offline here (`1796.64 ms/token`) and performs real direct reads: `205568.2 MB` total, `2035.33 MB/token`.
- `online remote CP` is much slower in this run (`3783.52 ms/token`) because remote solve overhead is large: `18.91 s` total remote wall over 3 calls, plus 752 anchor failures.
- `diff-tree-ideal` avoids the remote solver tax but still pays heavy transition/apply cost: `2138.42 ms/token`, 3 calls, `530/211/530` planned load/xfer/xform.
- `static-min` and `static-max` completed but have anchor failures (`960` and `1200`), so they are kept as diagnostic fixed-plan baselines rather than clean planner wins.
- The trace is highly memory-constrained for Gemma4: the 180s window covers only `54.3-66.5%` of full residency.
- A revised scaled trace was prepared with minimum coverage above 50%:
  `trace_inputs_scaled_min50/trace_90_user_116_gemma_min50.csv`, with
  window coverage `51.3-81.7%`; use that trace for the next fair rerun.

## Source Files

- final merged baseline CSV: `window_180s_baselines/baseline_results.csv`
- exact 180s trace window CSV: `window_180s_baselines/trace03_start0_180s.csv`
- full trace plot: `trace_plots/trace_03_user_116_cap7300.svg`
- 180s window trace plot: `window_180s_baselines/trace03_start0_180s.svg`
- raw all-baseline results: `window_180s_baselines/raw_results/all_baselines_20260629_results.csv`
- runner artifacts/logs/offline table: `window_180s_baselines/run_all_20260629/`
- revised min50 trace plot: `window_180s_baselines_scaled_min50/trace90_start0_180s.svg`
- reprofile manifest: `runtime/plan/generated/gemma-4-12B-it-Q4_0_reprofile_20260629/manifest.json`
