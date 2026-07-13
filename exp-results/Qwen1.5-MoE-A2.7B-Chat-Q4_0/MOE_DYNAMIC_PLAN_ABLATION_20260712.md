# Qwen1.5-MoE Q4_0 Dynamic Plan Ablation

## Setup

- Device: `3C15AU002CL00000`
- Model: `/data/local/tmp/unifer/llamacpp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`
- Cost model: `runtime/plan/profiles/android-opencl/Qwen1.5-MoE-A2.7B-Chat-Q4_0_gguf_all_20260630`
- Dynamic expert accounting: `LLAMA_ELASTIC_DYNAMIC_ACTIVE_EXPERTS=4`, `LLAMA_ELASTIC_DYNAMIC_TOTAL_EXPERTS=60`
- Fast protocol: short timed decode first, compare concrete plans against `static-max` and `mru`; run full 600s only after the plan choice is stable.

## cov40_90, 90s

| Method / concrete plan | ms/token | eval runs | direct read ms/fwd | direct read MB/fwd | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed `plan_8192MiB.json` | 424.86 | 212 | 229.13 | 654.51 | best fixed plan in this short ablation |
| fixed `plan_8192MiB_cand2.json` | 450.51 | 200 | 229.31 | 655.26 | diversity candidate slower in real runtime |
| `mru` | 1601.07 | 56 | 935.57 | 2332.38 | large foreground reload churn |
| `diff-tree-ideal`, dynamic base-only target | 409.42 | 214 | 234.38 | 682.95 | selects `plan_8192MiB.json` |

## cov60_95, 60s

| Method / concrete plan | ms/token | eval runs | direct read ms/fwd | direct read MB/fwd | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `static-max`, fixed `plan_8704MiB.json`, static max trace | 95.27 | 627 | 0.82 | 1.05 | upper-bound fixed max behavior |
| `mru`, dynamic trace | 971.81 | 61 | 604.28 | 1509.31 | does not converge to max-like residency |
| `diff-tree-ideal`, diversity candidate enabled by previous behavior | 740.24 | 78 | 200.52 | 478.27 | selected `plan_8704MiB_cand2.json` |
| `diff-tree-ideal`, dynamic base-only target | 384.87 | 152 | 174.76 | 446.58 | selects `plan_8704MiB.json` |
| fixed `plan_8704MiB.json`, dynamic trace, no online | 385.61 | 117 | ~181.46 | ~461.23 | same bottleneck without planner |

## Initial Finding

The candidate-selection bug is fixed: dynamic workload scan now defaults to base/steady plans and leaves diversity candidates opt-in through `LLAMA_ELASTIC_DYNAMIC_INCLUDE_DIVERSITY_CANDIDATES=1`.

At this stage, the remaining gap to `static-max` came from packed MoE tensors being reloaded at raw tensor granularity. This motivated the expert-slice runtime below. Later experiments also exposed a separate candidate-selection failure at high source budgets; the final placement guard is documented at the end of this file.

## Expert-Slice Runtime

The Q4_0 OpenCL decode path now has an opt-in expert-slice cache. It evicts the packed 60-expert parent, loads only router-selected raw Q4_0 expert slices, maps router indices to cache slots, and adapts cache capacity to the current budget. The full-memory and slice-cache runs use the same router IDs and math; a deterministic 12-token comparison is byte-identical.

After restoring the best synchronous-route implementation and pushing the final binary, a second 12-token phone smoke completed with `rc=0` and reproduced the same full-memory output prefix. Its `726.74 ms/token` is intentionally excluded from the performance table because the 11 decode samples include first-plan generation/apply and a completely cold expert cache. Steady-state comparisons use the timed 60-second runs below.

| Variant, cov60_95 | Window | ms/token | Cache MiB | Hit rate | Expert disk MiB | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| packed tensor, base-only plan | 60s | 384.87 | n/a | n/a | ~67,739 foreground total | original runtime granularity |
| Q4 SOA slice cache, 24 fixed slots | 20s | 362.52 | 2673.0 | 50.07% | 12,464.7 | physically bounded at low budget |
| adaptive SOA slice cache | 60s | 220.56 | 4900.5 | 80.47% | 23,060.8 | 24 minimum, up to 52 slots |
| adaptive SOA, one interleaved upload | 60s | 215.53 | 4900.5 | 80.63% | 23,388.8 | removes one write per miss |
| adaptive raw-Q4 slice cache | 60s | **195.39** | 4900.5 | 81.18% | 25,077.9 | best validated; CPU transform removed |
| raw-Q4 + layer router reuse | 60s | 199.17 | 4900.5 | 80.98% | 24,832.0 | correct, but waits shift to route upload |
| raw-Q4 + async route upload | 60s | 278.01 | 4900.5 | 79.82% | 18,970.9 | rejected: Adreno queue backlog |
| raw-Q4 + verified gate/up/down router reuse | 60s | 196.56 | 4900.5 | 81.12% | 24,989.8 | router reads fall 3x, but synchronization shifts to cache/route writes |
| raw-Q4 + inline scalar route | 60s | 835.67 | 4900.5 | 71.12% | 9,140.5 | rejected: extra scalar arguments severely regress the Adreno kernel |
| raw-Q4 + preserve-on-grow, repeat 1 / 2 | 60s | 184.94 / 203.77 | 4900.5 | 81.40% / 80.83% | 26,265.9 / 24,513.3 | mean 194.36; 1.8-1.9 GiB cache copy cancels the saved cold misses |

Best validated result at this stage: `195.39 ms/token`, down 49.2% from the base-only packed-tensor result and 2.05x the `95.27 ms/token` static-max upper bound. Planner/provider cost is below 1 ms/token in this run; the remaining gap is dominated by expert cache misses and synchronous disk-to-GPU uploads.

The first router-reuse attempt reduced blocking ID reads from about 72 to 24 calls/token without changing the generated prefix, but moved the wait into blocking expert/route writes. Passing four route slots as kernel scalar arguments also changed Adreno kernel code generation enough to regress throughput by over 4x. Those initial variants were rejected. The later implementation safely reuses one router result across each gate/up/down triplet and uses an event-protected nonblocking route upload.

Preserving resident slots across cache growth is retained only as the opt-in `GGML_ELASTIC_MOE_PRESERVE_GROW=1`. It performed 27-28 grow copies per run and copied 1.84-1.91 GiB on the GPU. Two cooled repeats averaged `194.36 ms/token`, statistically indistinguishable from the simpler `195.39 ms/token` path, so it is disabled by default. This identifies the next planner trade-off: retain a slice only when its expected avoided disk reload exceeds the device-copy cost, rather than copying every resident slot.

Artifacts:

- best performance: `diff_tree_moe_raw_cache_s60_cov60_95/`
- deterministic output checks: `moe_expert_cache_correctness/`
- final pushed-binary smoke: `moe_expert_cache_final_sanity_20260712/`
- triplet-router experiment: `diff_tree_moe_triplet_router_reuse_s60_cov60_95/`
- rejected inline-route experiment: `diff_tree_moe_inline_route_s60_cov60_95/`
- preserve-grow repeats: `diff_tree_moe_preserve_grow_s60_cov60_95/`, `diff_tree_moe_preserve_grow_repeat2_s60_cov60_95/`
- stage profile: `moe_cache_stage_profile_s30_cov60_95/`

## Final Candidate Guard and Runtime Result

### Root cause

The offline plan sequence is non-monotonic at the high-budget boundary. Increasing the source budget from `8448` to `8704 MiB` unexpectedly moves most non-dynamic weights from GPU to CPU, while the cost model predicts an improvement.

| Source budget MiB | Predicted ms/token | GPU / CPU / disk weights | Non-dynamic GPU weights | Non-dynamic GPU MiB |
| ---: | ---: | ---: | ---: | ---: |
| 7168 | 1328.73 | 320 / 70 / 21 | 266 | 755.2 |
| 7424 | 1220.31 | 316 / 76 / 19 | 261 | 913.7 |
| 7680 | 979.61 | 304 / 84 / 23 | 244 | 708.4 |
| 7936 | 813.48 | 328 / 72 / 11 | 266 | 755.2 |
| 8192 | 622.47 | 329 / 72 / 10 | 264 | 754.7 |
| 8448 | 446.06 | 331 / 70 / 10 | 263 | 736.7 |
| 8704 | 345.24 | 83 / 317 / 11 | 15 | 244.5 |

The unguarded online scan therefore selected `plan_8704MiB.json` and measured `792.20 ms/token`, despite a cool run and `rc=0`. Manually capping the scan at `8448 MiB` recovered `113.07 ms/token` for 30 seconds and `105.48 ms/token` for 60 seconds, confirming that the target-plan choice, rather than thermal state or plan apply, caused the regression.

### Generic rule

For dynamic-workload candidate scans, the selector now tracks non-dynamic GPU-resident bytes across increasing source budgets. It rejects a candidate when those bytes fall below `50%` of the best lower-budget candidate. Dynamic weights are excluded because their effective demand is handled by the runtime expert cache. The rule is model-independent, does not encode a particular budget, and can be configured with `LLAMA_ELASTIC_DYNAMIC_GPU_PLACEMENT_GUARD_RATIO`; setting it to `0` disables the guard.

The selected plan records the guard ratio, selected non-dynamic GPU bytes, and prior peak in `online_selection`. In the validated run, the selector rejected `8704` (`244.5 MiB` versus a `913.7 MiB` prior peak) and automatically selected `plan_8448MiB.json` (`736.7 MiB`).

### Final phone validation

| Configuration, cov60_95 | Window | ms/token | Exec ms/token | Expert hit rate | Online failures | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `static-max` reference | 60s | 95.27 | 95.27 | n/a | 0 | fixed full-memory upper bound |
| bounded expert cache, before router optimization | 60s | 131.46 | 130.63 | 94% class | 1 | one earlier online failure |
| router reuse + safe nonblocking route upload | 60s | 119.02 | 118.34 | 94.79% | 0 | still vulnerable to selecting `8704` |
| guard enabled, default cap `8704` | 30s | 116.99 | 116.57 | 92.10% | 0 | cold-to-warm transition |
| guard enabled, default cap `8704` | 60s | **99.19** | **98.81** | **95.12%** | **0** | automatically selects `8448` |

The final 60-second result is only `4.1%` slower than `static-max`, versus `8.31x` slower for the unguarded `8704` selection. A deterministic 12-token phone check completed with `rc=0`, no OpenCL/planner/apply failures, and the same generated token prefix as `static-max`. The host and phone binary SHA-256 both equal `639fa15842cdc556e90d93e432a34a6dcef3eb1d5c32547ec0fc821aff3a2650`.

Final artifacts:

- correctness and selected-plan metadata: `moe_gpu_guard_correctness_20260713/`
- 30-second default-cap run: `diff_tree_moe_gpu_guard_defaultcap_s30_cov60_95/`
- 60-second default-cap run: `diff_tree_moe_gpu_guard_defaultcap_s60_cov60_95/`
- unguarded failure: `diff_tree_moe_route_async_upload_repeat2_slots36_max56_w1_s60_cov60_95/`
- manual-cap controls: `diff_tree_moe_cap8448_route_slots36_max56_w1_s30_cov60_95/`, `diff_tree_moe_cap8448_route_slots36_max56_w1_s60_cov60_95/`
