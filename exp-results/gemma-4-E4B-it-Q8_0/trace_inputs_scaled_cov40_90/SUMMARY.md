# Scaled Budget Traces

- target raw min/max: `4395.4/9889.5 MiB`
- overhead subtracted for planner weight budget: `1242.0 MiB`
- planned weights: `9746.4 MiB`

| source | output | source min/mean/max | scaled min/mean/max | planner weight budget min/max | max weight gap |
|---|---|---:|---:|---:|---:|
| exp-results/Qwen2.5-7B-Instruct-Q8_0/trace_inputs/trace_03_user_116_cap7300.csv | exp-results/gemma-4-E4B-it-Q8_0/trace_inputs_scaled_cov40_90/trace_120_user_116_gemma_e4b_q8_cov40_90.csv | 5168.4/6395.1/7300.0 | 4395.4/7557.2/9889.5 | 3153.4/8647.5 | 1098.9 |
