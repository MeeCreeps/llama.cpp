# gemma-4-E4B-it-Q8_0 Trace120 Start0 180s Baseline Summary

> Superseded: use `../window_180s_baselines_scaled_cov40_90_nlayer42/SUMMARY.md`.
> This earlier run used a meta that included inactive Gemma E4B extra tensors
> (`blk.42..47`) even though runtime reports `n_layer = 42`, which caused
> online anchor failures and overestimated full residency.

Generated: 2026-06-30

## Setup

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/gemma-4-E4B-it-Q8_0.gguf`
- local model: `models/gemma-4-E4B-it-Q8_0/gemma-4-E4B-it-Q8_0.gguf`
- trace: `trace_120_user_116_gemma_e4b_q8_cov40_90.csv`
- source window: start `0s`, span `180s`
- window stats: min/mean/max `6420.2 / 8522.7 / 9889.5 MiB`
- buckets: min `6400 MiB`, max `9984 MiB`
- placement family: `gpu,disk_gpu`
- timed decode: `180s`
- context/batch/thread settings: `-c 4096 -b 32 -ub 32 -t 8 -ngl 99 -fa on -no-cnv`
- unified budget overhead: `KV=512 MiB`, `misc=256 MiB`, `pinned=410 MiB`, `safety=64 MiB`
- full-residency requirement: `10988.4 MiB` (`weights=9746.4`, `KV=512`, `misc=256`, `pinned=410`, `safety=64`)
- cost model: freshly reprofiled on `2026-06-30` from current `llama-cli` and this Gemma4 E4B Q8 model
- profile rows: opencl_compute `/home/myid/hz85760/workspace/project/elastic/llama.cpp/runtime/plan/generated/gemma-4-E4B-it-Q8_0_reprofile_20260630/csv/opencl_compute.csv`, total input rows `206612`, stage_records `1194`
- online CP optimization: indexed `CostModel` lookup in `runtime/plan/dynamic_budget_solver.py`; single solve dropped from about `8.5-9.0s` to `0.78-0.88s` on host probe

## Baseline Results

| method | status | rc | raw ms/token | exec ms/token | eval runs | apply | online calls/fails | remote wall ms | anchor failures | direct read MB/token | planned evict/load/xfer/xform | GPU before/after C |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| static-min | ok | 0 | 2110.64 | 2110.64 | 86 | 0 | n/a/n/a | 0.00 | 0 | 2827.25 | 0/243/243/243 | 36.00/47.30 |
| static-max | ok | 0 | 365.13 | 365.13 | 492 | 0 | n/a/n/a | 0.00 | 0 | 438.28 | 0/31/31/31 | 46.90/53.10 |
| offline | ok | 0 | 1047.64 | 1134.92 | 159 | 17 | n/a/n/a | 0.00 | 0 | 1311.08 | 0/1266/1266/1266 | 52.70/51.20 |
| diff-tree-ideal | ok | 0 | 605.24 | 620.46 | 277 | 15 | 15/0 | 0.00 | 0 | 304.03 | 0/1566/53/1566 | 50.80/54.30 |
| online | ok | 0 | 749.01 | 745.81 | 240 | 3 | 8/0 | 4127.06 | 9126 | 1000.28 | 0/192/192/192 | 45.30/54.30 |
| mru | ok | 0 | 594.78 | 593.62 | 301 | 20 | 20/0 | 0.00 | 0 | 549.86 | 0/0/0/0 | 50.80/50.80 |

## Notes

- Valid headline online result is from `run_online_clean_20260630` with rc=0. The earlier bounded async-load run is diagnostic only: it reached `832.25 ms/token` but exited rc=134 during teardown (`pthread_mutex_lock called on a destroyed mutex`) and triggered the phone GPU memory watchdog, so it is excluded.
- Online CP is now faster than offline on raw decode (`749.01` vs `1047.64 ms/token`) after indexing cost-model lookup, but still slower than diff-tree and MRU on this start0 window.
- `static-max` is fastest because the selected start0 180s window is relatively high-memory (`6420.2-9889.5 MiB`), so it sits near full residency for much of the run.
- `mru` is not a fake-fast result here: it performs real direct reads, about `549.86 MB/token`.
- Online clean still reports anchor failures (`9126`) even with rc=0; the pattern matches unhandled optional stage events rather than O_DIRECT failures. Treat online as performance-valid but needing stage-provider cleanup before claiming zero-error anchors.

## Source Files

- scaled full trace: `trace_inputs_scaled_cov40_90/trace_120_user_116_gemma_e4b_q8_cov40_90.csv`
- scaled trace plot: `trace_plots_scaled_cov40_90/trace_120_user_116_gemma_e4b_q8_cov40_90.svg`
- merged baseline CSV: `window_180s_baselines_scaled_cov40_90/baseline_results.csv`
- full baseline run artifacts: `window_180s_baselines_scaled_cov40_90/run_all_20260630/`
- valid online rerun artifacts: `window_180s_baselines_scaled_cov40_90/run_online_clean_20260630/`
- excluded bounded async diagnostic: `window_180s_baselines_scaled_cov40_90/run_online_bounded_20260630/`
- reprofile manifest: `runtime/plan/generated/gemma-4-E4B-it-Q8_0_reprofile_20260630/manifest.json`
