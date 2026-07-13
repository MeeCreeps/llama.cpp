# MoE All-Expert Dynamic Top-p Sweep

Model: `OLMoE-1B-7B-0125-Instruct-Q4_0.gguf`
Tokens per run: `1000`
Mode: `LLAMA_MOE_DYNAMIC_ALL_EXPERTS=1`, so dynamic Top-P selects from all 64 experts per layer instead of only the original fixed top-8 candidates.
Definition: expert weight working set only. KV cache, activation tensors, allocator overhead, and dense non-expert weights are excluded from the top subplot.
Expert-count columns use the per-token average across MoE layers, so the value is in the `1..64` per-layer expert range.

| top_p | avg experts/layer mean | avg experts/layer min/max | max experts in one layer | expert MB mean | expert MB min/max | expert / total active mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 3.61 | 2.25 / 4.88 | 9 | 195.12 | 121.50 / 263.25 | 0.4018 |
| 0.25 | 4.65 | 2.69 / 6.44 | 12 | 250.95 | 145.12 / 347.62 | 0.4625 |
| 0.30 | 6.20 | 3.00 / 9.31 | 13 | 334.61 | 162.00 / 502.88 | 0.5324 |
| 0.35 | 7.28 | 3.31 / 10.88 | 16 | 393.00 | 178.88 / 587.25 | 0.5725 |
| 0.40 | 9.02 | 4.75 / 12.50 | 20 | 486.88 | 256.50 / 675.00 | 0.6239 |

## Token-to-token Reload Demand

Reload demand is computed from expert ids as active experts in token `t` that were not active in token `t-1`, summed across layers. The cold-start token is excluded from the mean/min/max below.

| top_p | reload MB mean | reload MB min/max | new expert fraction mean |
|---:|---:|---:|---:|
| 0.20 | 145.69 | 47.25 / 249.75 | 0.7430 |
| 0.25 | 177.21 | 57.38 / 310.50 | 0.7015 |
| 0.30 | 230.82 | 64.12 / 432.00 | 0.6837 |
| 0.35 | 259.42 | 70.88 / 472.50 | 0.6530 |
| 0.40 | 303.08 | 77.62 / 583.88 | 0.6184 |

Artifacts:

- `per_top_p_active/top_p_*_active_weight_and_ratio.svg`
- `per_top_p_active/top_p_*_active_weight_and_ratio.png`
- `per_top_p_expert_count/top_p_*_active_expert_count.svg`
- `per_top_p_expert_count/top_p_*_active_expert_count.png`
- `per_top_p_reload/top_p_*_reload_demand.svg`
- `per_top_p_reload/top_p_*_reload_demand.png`
- `top_p_sweep_active_weight_timeseries.csv`
- `top_p_sweep_summary.csv`
- `top_p_sweep_reload_demand_timeseries.csv`
- `top_p_sweep_reload_demand_summary.csv`
- raw logs: `top_p_*.out`, `top_p_*.err`
