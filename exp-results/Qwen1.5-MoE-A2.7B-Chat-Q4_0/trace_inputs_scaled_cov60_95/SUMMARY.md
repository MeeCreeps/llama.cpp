# Scaled Budget Traces

- target raw min/max: `5405.0/8558.0 MiB`
- overhead subtracted for planner weight budget: `1242.0 MiB`
- planned weights: `7766.6 MiB`

| source | output | source min/mean/max | scaled min/mean/max | planner weight budget min/max | max weight gap |
|---|---|---:|---:|---:|---:|
| exp-results/Qwen2.5-7B-Instruct-Q8_0/trace_inputs/trace_03_user_116_cap7300.csv | exp-results/Qwen1.5-MoE-A2.7B-Chat-Q4_0/trace_inputs_scaled_cov60_95/trace_220_user_116_qwen15moe_q4_0_cov60_95.csv | 5168.4/6395.1/7300.0 | 5405.0/7219.5/8558.0 | 4163.0/7316.0 | 450.6 |
