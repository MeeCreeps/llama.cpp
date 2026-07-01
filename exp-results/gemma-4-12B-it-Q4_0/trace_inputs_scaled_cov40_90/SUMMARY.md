# Scaled Budget Traces

- target raw min/max: `4388.0/9873.0 MiB`
- overhead subtracted for planner weight budget: `1242.0 MiB`
- planned weights: `9727.8 MiB`

| source | output | source min/mean/max | scaled min/mean/max | planner weight budget min/max | max weight gap |
|---|---|---:|---:|---:|---:|
| exp-results/Qwen2.5-14B-Instruct-Q4_0/trace_inputs/trace_03_user_116_cap7300.csv | exp-results/gemma-4-12B-it-Q4_0/trace_inputs_scaled_cov40_90/trace_100_user_116_gemma_cov40_90.csv | 5168.4/6395.1/7300.0 | 4388.0/7544.6/9873.0 | 3146.0/8631.0 | 1096.8 |
