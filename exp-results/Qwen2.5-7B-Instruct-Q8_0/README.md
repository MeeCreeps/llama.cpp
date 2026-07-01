# Qwen2.5-7B-Instruct-Q8_0 Elastic Memory Baselines

## Purpose

This experiment reruns the elastic memory baselines on real dynamic memory traces with enough budget range to expose differences between fixed offline plans, online CP, current-state diff-tree planning, static min/max plans, and MRU.

## Environment

- repo: `/home/myid/hz85760/workspace/project/elastic/llama.cpp`
- device: `3C15AU002CL00000`
- phone model: `CPH2749`
- phone directory: `/data/local/tmp/hyzheng/elastic`
- remote model: `/data/local/tmp/hyzheng/elastic/Qwen2.5-7B-Instruct-Q8_0.gguf`
- model size on phone: about `7.5G`
- planned weights: `6611.9 MiB`
- budget overhead: `KV=512 MiB`, `misc=256 MiB`, `pinned=410 MiB`, `safety=64 MiB`
- full-residency raw requirement: `7853.9 MiB`

## Trace Selection

Source trace is copied into `trace_inputs`.

| trace | window | min/mean/max MiB | min/max bucket |
|---|---:|---:|---:|
| `trace_03_user_116_cap7300.csv` | start `0s`, span `180s` | `5954.0 / 6769.7 / 7300.0` | `5888 / 7424` |

Full-trace plots are in `trace_plots`. The exact 180s window CSV/SVG used for
the final table is in `window_180s_baselines`.

## Runtime Settings

- decode duration: `180 s`
- trace replay speedup: `1x`
- source window: start `0s`, span `180 s`
- budget bucket: `256 MiB`
- OpenCL GPU layers: `-ngl 99`
- context/batch/thread settings: `-c 4096 -b 32 -ub 32 -t 8`
- static-max/static-min: fixed offline plan with pipeline execution
- online CP: fastest fixed remote-host solver result,
  `trace03_cap7300_start0_180s_online_timeline_kv256_guard80_tl50_b32`
- diff-tree-ideal: current-runtime-state based online diff selection
- MRU: strict `kv=512` Q8 direct-read no-retain runtime-cache baseline

## Result Layout

- `trace_inputs/`: selected source CSV trace
- `trace_plots/`: source trace SVG and summary
- `window_180s_baselines/trace03_start0_180s.csv`: exact 180s replay window
- `window_180s_baselines/trace03_start0_180s.svg`: exact 180s replay window plot
- `window_180s_baselines/baseline_results.csv`: final merged baseline data
- `window_180s_baselines/raw_results/`: raw `results.csv` files used to build
  the final table
- `TRACE03_START0_180S_BASELINE_SUMMARY_2026-06-26.md`: final readable
  experiment setup, interpretation, and baseline table
