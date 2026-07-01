# gemma-4-E4B-it-Q8_0 Corrected Baseline Summary

Generated: 2026-06-30

## Setup

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `/data/local/tmp/unifer/llamacpp/gemma-4-E4B-it-Q8_0.gguf`
- local model: `models/gemma-4-E4B-it-Q8_0/gemma-4-E4B-it-Q8_0.gguf`
- model sha256: `fb8f0c032de00b18c710824af3c7e5777c71e5fb60b13f13575f0a9e92ddecd0`
- trace: `trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42.csv`
- trace plot: `../trace_plots_scaled_cov40_90_nlayer42/trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42.svg`
- source window: start `0s`, span `180s`
- window stats: min/mean/max `5891.1 / 7820.2 / 9074.4 MiB`
- buckets: min `5888 MiB`, max `9216 MiB`
- placement family: `gpu,disk_gpu`
- timed decode: `180s`
- context/batch/thread settings: `-c 4096 -b 32 -ub 32 -t 8 -ngl 99 -fa on -no-cnv`
- unified budget overhead: `KV=512 MiB`, `misc=256 MiB`, `pinned=410 MiB`, `safety=64 MiB`
- corrected managed weights: `8846.4 MiB`, full residency `10088.4 MiB`
- cost model: freshly profiled on `2026-06-30` using this Gemma E4B Q8 model
- model meta: `runtime/plan/model_meta/gemma-4-E4B-it-Q8_0_reprofile_20260630_nlayer42.weights_ops.json`

## Correction

Gemma E4B Q8 reports `n_layer = 42` at runtime, but the GGUF contains extra `blk.42` to `blk.47` tensors. The first meta included those inactive tensors, which caused online anchor failures because the OpenCL runtime did not register them in WBM. The corrected meta filters `blk.N` tensors with `N >= 42`, then rescales the 40-90% trace from the corrected full-residency requirement.

## Baseline Results

| method | status | rc | raw ms/token | exec ms/token | eval runs | apply | online calls/fails | remote wall ms | anchor failures | direct read MB/token | planned evict/load/xfer/xform | GPU before/after C |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| static-min | ok | 0 | 2292.23 | 2292.23 | 79 | 0 | n/a/n/a | 0.00 | 0 | 3050.06 | 0/216/216/216 | 39.5/50.8 |
| static-max | ok | 0 | 532.37 | 532.37 | 338 | 0 | n/a/n/a | 0.00 | 0 | 687.38 | 0/33/33/33 | 50.0/51.5 |
| offline | ok | 0 | 1328.38 | 1403.45 | 129 | 9 | n/a/n/a | 0.00 | 0 | 1460.90 | 0/753/753/753 | 51.2/51.5 |
| diff-tree-ideal | ok | 0 | 677.34 | 685.24 | 261 | 3 | 3/0 | 0.00 | 0 | 372.45 | 0/300/41/300 | 50.8/55.4 |
| online | ok | 0 | 815.72 | 812.78 | 221 | 2 | 5/0 | 2441.32 | 0 | 917.96 | 0/154/154/154 | 54.6/53.9 |
| mru | ok | 0 | 777.98 | 777.68 | 231 | 12 | 12/0 | 0.00 | 0 | 752.67 | 0/0/0/0 | 53.5/51.5 |

## Notes

- All corrected baselines completed with `rc=0` and `anchor failures=0`.
- `static-max` is fastest because the selected start0 window is relatively high-memory after correction.
- `diff-tree-ideal` is slower than `static-max` but faster than `offline`, which is the expected fair trend for this trace.
- `online` is slower than `diff-tree-ideal`; this is mostly runtime IO/plan choice, not server overhead. Server wall time is only `2441.32 ms` total.
- `mru` is not skipping IO: it performs `752.67 MB/token` of direct reads and `12113` xform calls.
- The previous `window_180s_baselines_scaled_cov40_90` run is superseded because its meta included inactive `blk.42..47` tensors and produced online anchor failures.

## Artifacts

- baseline CSV: `baseline_results.csv`
- full run summary: `run_all_20260630/summary/SUMMARY.md`
- full run CSV: `run_all_20260630/summary/results.csv`
- full logs: `run_all_20260630/logs/`
- scaled trace: `../trace_inputs_scaled_cov40_90_nlayer42/trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42.csv`
- scaled trace plot: `../trace_plots_scaled_cov40_90_nlayer42/trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42.svg`
