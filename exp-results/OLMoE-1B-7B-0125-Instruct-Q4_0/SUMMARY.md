# OLMoE-1B-7B-0125-Instruct Q4_0 Elastic Baselines

Generated: 2026-06-30

## Model And Artifacts

- model repo: `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`
- local GGUF: `/home/myid/hz85760/workspace/project/elastic/models/OLMoE-1B-7B-0125-Instruct-Q4_0/OLMoE-1B-7B-0125-Instruct-Q4_0.gguf`
- phone GGUF: `/data/local/tmp/unifer/llamacpp/OLMoE-1B-7B-0125-Instruct-Q4_0.gguf`
- SHA256: `df43bd738dfdcbe889c5f854120c271e9e53e2c624f212ac2523f8bc11ae9bf3`
- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`

## Why Q4_0

`Q4_K_M` was not a fair GPU-MoE test because OpenCL `MUL_MAT_ID` does not support the expert `Q4_K/Q6_K` tensors, so most expert weights stayed in `CPU_Elastic`. `Q4_0` fixes that for this model:

- `Q4_0` smoke run:
  - `OpenCL model buffer size = 3689.10 MiB`
  - `CPU_Elastic model buffer size = 55.27 MiB`
- complete GGUF meta:
  - weights/ops: `195/195`
  - planned weight size: `3744.37 MiB`
  - expert tensors:
    - `ffn_down_exps`: `1152.0 MiB`, `Q4_0`
    - `ffn_gate_exps`: `1152.0 MiB`, `Q4_0`
    - `ffn_up_exps`: `1152.0 MiB`, `Q4_0`

## Trace

- source trace: `exp-results/Qwen2.5-7B-Instruct-Q8_0/trace_inputs/trace_03_user_116_cap7300.csv`
- scaled trace: `trace_inputs_scaled_cov40_90/trace_170_user_116_olmoe_q4_0_cov40_90.csv`
- plot: `trace_plots_scaled_cov40_90/trace_170_user_116_olmoe_q4_0_cov40_90.svg`
- full residency requirement: `4986.37 MiB`
- scaled raw budget range: `1995.0-4488.0 MiB`, i.e. `40.0%-90.0%`
- 180s window used by baselines: min/mean/max `2913.8/3867.8/4488.0 MiB`
- usable planner weight budget in the full scaled trace: `753.0-3246.0 MiB`

## Cost Model

- model meta: `runtime/plan/model_meta/OLMoE-1B-7B-0125-Instruct-Q4_0_gguf_all.weights_ops.json`
- cost dir: `runtime/plan/profiles/android-opencl/OLMoE-1B-7B-0125-Instruct-Q4_0_gguf_all_20260630`
- profile CSV rows:
  - `opencl_compute.csv`: `7152`
  - `opencl_stage.csv`: `9288`
  - `opencl_reload_legacy.csv`: `7152`
  - `cpu_compute.csv`: `4060`
  - `cpu_stage.csv`: `3796`
- cost summary: `31448` input rows, `2502` op records, `229` stage records

## Baseline Results

All runs use `KV=512 MiB`, `misc=256 MiB`, `pinned-extra=410 MiB`, `safety=64 MiB`, 180s wall-clock decode, bucket size `256 MiB`, and allowed placements `cpu,gpu,disk_cpu,disk_gpu`.

Thermals reached status `3` during this run, so the table is a valid first full pass but should be considered warm-device data.

| method | rc | raw ms/token | exec ms/token | apply count | apply ms total | online calls | remote/server ms | direct read MB/token | anchor failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static-min | 0 | 1249.35 | 1249.35 | 0 | 0.00 |  | 0/0 | 2485.44 | 0 |
| static-max | 0 | 365.20 | 365.20 | 0 | 0.00 |  | 0/0 | 524.14 | 0 |
| offline | 0 | 827.33 | 868.94 | 8 | 8701.96 |  | 0/0 | 1099.32 | 0 |
| diff-tree-ideal | 0 | 843.01 | 867.58 | 6 | 5600.89 | 6 | 0/0 | 1031.45 | 0 |
| online | 0 | 761.56 | 759.54 | 2 | 167.26 | 3 | 1147.71/1092.30 | 662.19 | 0 |
| mru | 0 | 635.51 | 635.65 | 10 | 199.17 | 10 | 0/0 | 986.20 | 0 |

Detailed raw CSV: `window_180s_baselines_scaled_cov40_90/summary/results.csv`

Detailed run summary: `window_180s_baselines_scaled_cov40_90/summary/SUMMARY.md`

## Initial Takeaways

- `Q4_0` is the correct MoE GPU test for this OpenCL backend: expert weights now mostly reside in OpenCL, unlike `Q4_K_M`.
- `static-max` remains fastest on this 180s window because the window's min bucket is not extremely low and fixed high-residency avoids plan switching.
- `online` is the best planner-style dynamic result in this run: `761.56 ms/token`, faster than `offline` and `diff-tree-ideal`.
- CP server solve is not the main cost for online: total remote server time is about `1.09 s`.
- MRU is faster than planner-style dynamic methods but still does real budget tracking: final summary shows `23004` accesses, `13717` misses, `2446` evictions, and resident bytes `2851.85 MiB` for budget `2854.00 MiB`.
- There is still a small CPU_Elastic residual path around `55 MiB`; planner variants can repeatedly reload/xform it during switches.
