# Scaled Budget Traces

- target raw min/max: `4033.1/9074.4 MiB`
- overhead subtracted for planner weight budget: `1242.0 MiB`
- planned weights: `8846.4 MiB`

| source | output | source min/mean/max | scaled min/mean/max | planner weight budget min/max | max weight gap |
|---|---|---:|---:|---:|---:|
| exp-results/Qwen2.5-7B-Instruct-Q8_0/trace_inputs/trace_03_user_116_cap7300.csv | exp-results/gemma-4-E4B-it-Q8_0/trace_inputs_scaled_cov40_90_nlayer42/trace_130_user_116_gemma_e4b_q8_cov40_90_nlayer42.csv | 5168.4/6395.1/7300.0 | 4033.1/6934.3/9074.4 | 2791.1/7832.4 | 1014.0 |
