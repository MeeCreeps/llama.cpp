# MoE Dynamic Expert Trace

Model: `OLMoE-1B-7B-0125-Instruct-Q4_0`

Run: normal generation, `LLAMA_MOE_DYNAMIC_TOP_P=0.4`, `LLAMA_MOE_DYNAMIC_GPU_DECODE=1`, `-ngl 99`, `-n 1000`, `--ignore-eos`. No external dynamic memory budget trace was used.

Memory definition: expert weight demand only. KV cache, dense/shared weights, attention weights, norms, activations, and allocator overhead are excluded.

Decode token records parsed: `999`
MoE layers: `16`
Expert count per layer: `64`
Expert weight per active expert per layer: `3.375 MiB`

| Metric | Value |
|---|---:|
| Total active experts/token, mean | 115.90 |
| Total active experts/token, min | 83 |
| Total active experts/token, max | 128 |
| Avg active experts/layer, mean | 7.24 |
| Expert weight demand/token mean, no KV | 391.15 MiB |
| Expert weight demand/token min, no KV | 280.12 MiB |
| Expert weight demand/token max, no KV | 432.00 MiB |

Artifacts:

- `moe_dyn_1000.out`
- `moe_dyn_1000.err`
- `moe_dyn_1000_active_experts.csv`
- `active_experts_over_tokens.svg`
- `active_experts_over_tokens.png`
- `expert_weight_memory_demand_no_kv.svg`
- `expert_weight_memory_demand_no_kv.png`
