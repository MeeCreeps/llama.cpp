# Scaled Budget Traces

- target raw min/max: `1995.0/4488.0 MiB`
- overhead subtracted for planner weight budget: `1242.0 MiB`
- planned weights: `3744.4 MiB`

| source | output | source min/mean/max | scaled min/mean/max | planner weight budget min/max | max weight gap |
|---|---|---:|---:|---:|---:|
| exp-results/Qwen2.5-7B-Instruct-Q8_0/trace_inputs/trace_03_user_116_cap7300.csv | exp-results/OLMoE-1B-7B-0125-Instruct-Q4_0/trace_inputs_scaled_cov40_90/trace_170_user_116_olmoe_q4_0_cov40_90.csv | 5168.4/6395.1/7300.0 | 1995.0/3429.7/4488.0 | 753.0/3246.0 | 498.4 |
