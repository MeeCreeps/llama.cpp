# Qwen2.5-14B-Instruct-Q4_0 Elastic Memory Baselines

## Purpose

This directory contains the trace03 start0 180s Android baseline run for Qwen2.5-14B-Instruct-Q4_0, using a freshly reprofiled cost model from the current local build.

## Runtime Settings

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `qwen2.5-14b-instruct-q4_0.gguf`
- source window: start `0s`, span `180s`
- trace replay speedup: `1x`
- context/batch/thread settings: `-c 4096 -b 32 -ub 32 -t 8`
- budget bucket: `256 MiB`
- unified budget overhead: `KV=512 MiB`, `misc=256 MiB`, `pinned=410 MiB`, `safety=64 MiB`
- placements: `gpu,disk_gpu`

## Result Layout

- `trace_inputs/`: selected source CSV trace
- `trace_plots/`: full trace three-panel SVG and summary
- `window_180s_baselines/trace03_start0_180s.csv`: exact 180s replay window
- `window_180s_baselines/trace03_start0_180s.svg`: exact 180s replay window three-panel SVG
- `window_180s_baselines/baseline_results.csv`: final merged baseline data
- `window_180s_baselines/raw_results/`: raw result CSVs used to build the final table
- `TRACE03_START0_180S_BASELINE_SUMMARY_2026-06-27.md`: final readable setup, interpretation, and baseline table
- `trace04_inputs/`, `trace04_plots/`, `trace04_start0_180s_baselines/`: second-trace stability check
- `TRACE04_START0_180S_STABILITY_SUMMARY_2026-06-27.md`: second-trace stability table and cross-trace comparison
