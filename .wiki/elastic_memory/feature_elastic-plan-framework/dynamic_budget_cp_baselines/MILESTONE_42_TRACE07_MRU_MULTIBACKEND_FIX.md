# Milestone 42: Trace07 MRU anomaly and multi-backend fix

Date: 2026-06-19
Device: OP12 5ae7a43d
Remote dir: /data/local/tmp/hyzheng/elastic
Model: Meta-Llama-3-8B-Instruct.Q4_0.gguf
Trace: trace_07_user_7, 240 s source window, 180 s decode bench

## Why this run was needed

Trace01 showed MRU at about 3339.57 ms/token, much slower than both offline and static-min. That was suspicious because MRU should be a simple stateful baseline, not a catastrophically slow CPU-only path.

I reran a different dynamic trace, trace_07_user_7, whose selected window is more memory constrained.

| min MiB | mean MiB | max MiB | min bucket |
|---:|---:|---:|---:|
| 3304.9 | 4462.6 | 5483.6 | 3072 |

## Trace07 result before fixing MRU

Artifact: .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/trace07_pd16_xfer_counterfix_3min

| baseline | decode ms/token | direct read ms/forward | reload issue ms/forward | apply count | anchor load/xform | failures |
|---|---:|---:|---:|---:|---:|---:|
| offline table | 344.78 | 90.27 | 104.94 | 13 | 580 / 580 | 0 |
| online remote CP-SAT | 400.96 | 125.51 | 145.02 | 14 | 457 / 457 | 0 |
| MRU old | 3955.85 | 270.02 | 0.00 | 10 | 323 / 323 | 0 |
| static-min | 2001.88 | 908.35 | 1140.13 | 0 | 98 / 98 | 0 |

This reproduced the anomaly: old MRU was even slower than static-min.

## Root cause

The old native MRU plan was not a multi-backend MRU baseline. Pulled MRU plans showed CPU-only routing.

| plan | op routing | residency | schedule |
|---|---|---|---:|
| old MRU B=3072 | cpu 224 | cpu 126, disk 98 | 0 |
| old MRU B=3584 | cpu 224 | cpu 157, disk 67 | 0 |
| old MRU B=5120 | cpu 224 | cpu 224 | 0 |

Runtime log confirmed that old MRU did not execute the same OpenCL xform pipeline as CP-SAT plans.

| metric | old MRU |
|---|---:|
| eval ms/token | 3955.85 |
| stage load calls | 323 |
| stage xform calls | 0 |
| schedule_used | 0 |

Why it happened:

- native MRU selected each weight target backend by comparing cold path cost, including compute plus load, transfer, and transform.
- for this profile, cold CPU path looked cheaper for all 224 weights.
- MRU then kept or evicted according to recency, but all compute routes were CPU.
- therefore the baseline measured a CPU-only route, not MRU movement over CPU and GPU placements.

## Code fix

Changed native MRU target backend selection in tools/main/main.cpp.

- before: choose backend by cold path cost, including reload and transform.
- after: choose backend by measured compute cost only.
- MRU still decides which weights remain resident under the budget.
- added prefetch_distance to native timeline anchor placement, so MRU load anchors are not always only one op before use.

This makes MRU a stateful eviction baseline over the same CPU/GPU target backend split used by the cost model, instead of degenerating into CPU-only execution.

## Trace07 MRU after fix

Artifact: .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/trace07_pd16_xfer_mru_multibackend_fix_3min

| baseline | decode ms/token | direct read ms/forward | reload issue ms/forward | apply count | anchor load/xform | failures |
|---|---:|---:|---:|---:|---:|---:|
| MRU fixed | 382.83 | 118.51 | 140.61 | 13 | 423 / 423 | 0 |

Pulled fixed MRU plans now show multi-backend routing.

| plan | op routing | residency | timeline | schedule |
|---|---|---|---:|---:|
| fixed MRU B=3072 | cpu 65, gpu 159 | cpu 39, gpu 87, disk 98 | 296 | 0 |
| fixed MRU B=3328 | cpu 65, gpu 159 | cpu 45, gpu 99, disk 80 | 231 | 0 |
| fixed MRU B=3840 | cpu 65, gpu 159 | cpu 52, gpu 119, disk 53 | 150 | 0 |
| fixed MRU B=4864 | cpu 65, gpu 159 | cpu 65, gpu 159 | 0 | 0 |

Runtime stage counters now show real xform execution.

| metric | fixed MRU |
|---|---:|
| eval ms/token | 382.83 |
| stage load calls | 423 |
| stage xform calls | 328 |
| direct O_DIRECT read total ms | 55698.29 |

## Updated interpretation

The old MRU numbers from trace01 and trace07 should not be used as final baseline results because old MRU was CPU-only by accident.

On trace07 after the fix:

| method | decode ms/token |
|---|---:|
| offline table | 344.78 |
| MRU fixed | 382.83 |
| online remote CP-SAT | 400.96 |
| static-min | 2001.88 |

This trace does not show online faster than offline. Online made more state-aware transitions, but it also incurred more reload/direct-read per forward in this window.

| method | direct read ms/forward | reload issue ms/forward | apply count |
|---|---:|---:|---:|
| offline table | 90.27 | 104.94 | 13 |
| online remote CP-SAT | 125.51 | 145.02 | 14 |
| MRU fixed | 118.51 | 140.61 | 13 |

So trace07 is useful as a correctness/debug trace: it exposed the MRU implementation bug. It is not yet a clean example of online beating offline.

## Next validation

- Re-run trace01 MRU with the fixed binary, because old trace01 MRU was also CPU-only and should be discarded.
- Then re-run a four-baseline matrix on at least trace01 and trace07 with the fixed MRU implementation.
- If online remains slower than offline on some traces, inspect plan transitions and current-state penalties instead of assuming online always wins.

## Trace01 MRU recheck after the same fix

Artifact: .wiki/elastic_memory/feature_elastic-plan-framework/dynamic_budget_cp_baselines/artifacts/trace01_pd16_xfer_mru_multibackend_fix_3min

Old trace01 MRU from the earlier full matrix was 3339.57 ms/token. After the multi-backend MRU fix:

| baseline | decode ms/token | direct read ms/forward | reload issue ms/forward | apply count | anchor load/xform | failures |
|---|---:|---:|---:|---:|---:|---:|
| MRU fixed trace01 | 734.62 | 205.05 | 258.20 | 7 | 166 / 166 | 0 |

Pulled trace01 fixed MRU plans also show multi-backend routing:

| plan | op routing | residency | timeline | schedule |
|---|---|---|---:|---:|
| fixed MRU B=3584 | cpu 65, gpu 159 | cpu 48, gpu 109, disk 67 | 216 | 0 |
| fixed MRU B=4096 | cpu 65, gpu 159 | cpu 58, gpu 131, disk 35 | 120 | 0 |
| fixed MRU B=4608 | cpu 65, gpu 159 | cpu 65, gpu 153, disk 6 | 24 | 0 |
| fixed MRU B=4864 | cpu 65, gpu 159 | cpu 65, gpu 159 | 0 | 0 |

Runtime stage counters:

| metric | fixed MRU trace01 |
|---|---:|
| eval ms/token | 734.62 |
| stage load calls | 166 |
| stage xform calls | 134 |
| direct O_DIRECT read total ms | 50236.36 |

Trace01 updated comparison, using the previous pipeline full matrix plus fixed MRU:

| method | decode ms/token |
|---|---:|
| online remote CP-SAT | 422.75 |
| offline table | 729.59 |
| MRU fixed | 734.62 |
| static-min | 1270.39 |
| old MRU invalid | 3339.57 |

This confirms the old MRU result was invalid. With the fix, MRU is no longer catastrophically slow; it is close to offline on trace01 and between offline and online/static-min depending on trace.
