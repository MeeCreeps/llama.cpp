# Trace Memory Gap Plots: Qwen2.5-7B-Instruct-Q8_0

- source trace: `trace/traces_10g/trace_03_user_116.csv`
- source png: `trace/traces_10g/trace_03_user_116.png`
- scaled trace CSV: `../trace_inputs/trace_03_user_116_cap7300.csv`
- plot: [trace_03_user_116_cap7300.svg](trace_03_user_116_cap7300.svg)
- transform: cap only, `mem_available_mb = min(original, 7300.0)`; no global shift
- capped samples: `27` / `1228`
- original min/mean/max: `5168.4` / `6399.0` / `7634.6` MiB
- capped min/mean/max: `5168.4` / `6395.1` / `7300.0` MiB
- full-residency requirement: `7853.9 MiB`
- weights/KV/misc/pinned/safety: `6611.9/512/256/410/64 MiB`

| trace | min/mean/max MiB | min/max gap MiB | coverage % | floor buckets | static max bucket |
|---|---:|---:|---:|---|---:|
| `trace_03_user_116_cap7300.csv` | 5168.4/6395.1/7300.0 | -2685.5/-553.9 | 65.8-92.9 | 5120:15, 5632:217, 5888:155, 6144:116, 6400:351, 6656:289, 6912:29, 7168:56 | 7424 |
