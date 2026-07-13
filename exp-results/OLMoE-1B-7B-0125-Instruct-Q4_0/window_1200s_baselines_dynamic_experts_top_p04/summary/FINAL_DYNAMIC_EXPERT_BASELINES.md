# OLMoE Dynamic Experts 20-Min Baselines

- model: `OLMoE-1B-7B-0125-Instruct-Q4_0.gguf`
- device: `3C15AU002CL00000`
- trace: `trace_170_user_116_olmoe_q4_0_cov40_90.csv`
- window: start `0s`, span `1200s`
- memory budget: min `1995.0 MiB`, mean `3439.2 MiB`, max `4488.0 MiB`, min bucket `1792 MiB`
- dynamic experts: `LLAMA_MOE_DYNAMIC_TOP_P=0.4`, `LLAMA_MOE_DYNAMIC_GPU_DECODE=1`
- note: `mru` in the first full-matrix run returned `remote_incomplete`; the valid `mru` row below is from the retry with `--ignore-eos`.

| method | status | rc | raw ms/token | exec ms/token | eval runs | apply count | planned load | direct read MB/forward | provider get ms | source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| static-min | ok | 0 | 1815.24 | 1815.24 | 662 | 0 | 56 | 3103.5 | 0.00 | full matrix |
| static-max | ok | 0 | 324.20 | 324.20 | 3698 | 0 | 14 | 271.8 | 0.00 | full matrix |
| offline | ok | 0 | 911.10 | 1216.12 | 987 | 248 | 5084 | 1235.5 | 20.43 | full matrix |
| diff-tree-ideal | ok | 0 | 808.30 | 824.65 | 1452 | 16 | 412 | 975.2 | 1179.70 | full matrix |
| online | ok | 0 | 752.42 | 760.70 | 1576 | 8 | 261 | 1085.5 | 353.45 | full matrix |
| mru | ok | 0 | 1000.61 | 1001.34 | 1185 | 299 | 0 | 1344.1 | 6744.26 | retry ignore-eos |

## Result Files

- full matrix summary: `summary/SUMMARY.md`
- full matrix csv: `summary/results.csv`
- valid mru retry summary: `../window_1200s_baselines_dynamic_experts_top_p04_mru_retry_ignore_eos/summary/SUMMARY.md`
- valid mru retry csv: `../window_1200s_baselines_dynamic_experts_top_p04_mru_retry_ignore_eos/summary/results.csv`
