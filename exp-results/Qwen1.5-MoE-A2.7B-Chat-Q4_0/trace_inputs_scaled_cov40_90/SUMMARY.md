# Scaled Budget Traces

- target raw min/max: `3603.0/8108.0 MiB`
- overhead subtracted for planner weight budget: `1242.0 MiB`
- planned weights: `7766.6 MiB`

| source | output | source min/mean/max | scaled min/mean/max | planner weight budget min/max | max weight gap |
|---|---|---:|---:|---:|---:|
| exp-results/Qwen2.5-7B-Instruct-Q8_0/trace_inputs/trace_03_user_116_cap7300.csv | exp-results/Qwen1.5-MoE-A2.7B-Chat-Q4_0/trace_inputs_scaled_cov40_90/trace_210_user_116_qwen15moe_q4_0_cov40_90.csv | 5168.4/6395.1/7300.0 | 3603.0/6195.6/8108.0 | 2361.0/6866.0 | 900.6 |
