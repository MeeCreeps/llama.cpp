# Offline vs Online Plan Differences by Trace

This document explains how the offline-table plans, online CP-SAT plans, and MRU plans differ under the 20-minute trace-window matrix from:

```text
MILESTONE_19_ALL_20MIN_TRACES_MRU_REAL_ELASTIC.md
```

The focus here is plan structure, not only speed.

## How to Read the Tables

Plan signature:

```text
(gpu, cpu, disk, load, transfer, xform, evict)
```

Example:

```text
(197, 0, 27, 27, 27, 27, 0)
```

means:

- 197 weights are planned GPU-resident.
- 27 weights are planned disk-backed / loaded on demand.
- The timeline contains 27 LOAD, 27 TRANSFER, and 27 XFORM events.

Important distinction:

- **Offline table** chooses from fixed budget bands: `3000, 3300, 3500, 3800, 4200, 4600, 5000, 5500, 6000 MiB`.
- **Online CP-SAT** solves for the current bucketed budget and current residency state.
- Therefore online can produce plans like `3840`, `4096`, `4352`, `4608`, etc., while offline maps these to the nearest table band below the current budget.

## High-Level Difference

Offline plans are **budget-aware but state-blind**:

```text
plan = f(budget_band)
```

Online plans are **budget-aware and state-aware**:

```text
plan = f(current_budget, current_residency_state)
```

MRU plans are **budget-aware and heuristic/stateful**:

```text
plan = f(current_budget, recent residency/order heuristic)
```

MRU does not use the measured cost model. It decides what to move based on a recency-style eviction policy, not on predicted reload / transfer / transform / compute impact.

This causes two major differences:

1. Online often keeps weights that are already resident, avoiding unnecessary churn.
2. Online places LOAD / TRANSFER / XFORM events for different weights than offline, because the current runtime state changes the marginal value of each weight.

In the low-budget traces, the concrete pattern is:

- Offline tends to preserve a canonical set of later-layer weights from the precomputed table.
- Online often keeps more early/currently-resident weights and evicts a different set.
- MRU may keep a recency-shaped set that is neither the offline canonical optimum nor the online CP-SAT cost-aware set.

This is visible in the `online_gpu_only` and `offline_gpu_only` examples below.

## Trace 01: user147

Performance summary:

```text
offline: 392.39 ms/token
online:  378.62 ms/token, excluding remote solver wall time
MRU:     397.82 ms/token
```

Runtime plan switches:

```text
offline applied:
  4600(load=0), 3800(load=27), 4600(load=0),
  4200(load=12), 4600(load=0)

online applied:
  3840(load=32), 4096(load=16), 4352(load=9),
  4608(load=0), 4352(load=9), 4608(load=0)

MRU applied:
  4352(load=10), 4096(load=24), 4608(load=0),
  4352(load=10), 4608(load=0)
```

Plan-level differences:

| online budget | offline band | offline signature | online signature | online-only GPU weights | offline-only GPU weights |
|---:|---:|---|---|---:|---:|
| 3840 | 3800 | `(197,0,27,27,27,27,0)` | `(192,0,32,32,32,32,0)` | 26 | 31 |
| 4096 | 3800 | `(197,0,27,27,27,27,0)` | `(208,0,16,16,16,16,0)` | 26 | 15 |
| 4352 | 4200 | `(212,0,12,12,12,12,0)` | `(215,0,9,9,9,9,0)` | 12 | 9 |
| 4608 | 4600 | `(224,0,0,0,0,0,0)` | `(224,0,0,0,0,0,0)` | 0 | 0 |

Example at `online budget=4352` vs `offline band=4200`:

```text
online keeps on GPU but offline does not:
  blk.0.ffn_down.weight
  blk.0.ffn_gate.weight
  blk.0.ffn_up.weight

offline keeps on GPU but online does not:
  blk.12.ffn_down.weight
  blk.12.ffn_up.weight
  blk.13.attn_k.weight
```

Interpretation:

Online uses the actual current residency state and can fit 215 GPU weights at `4352 MiB`, while offline's `4200 MiB` table keeps 212. The exact resident set also differs: online preserves some early/current-state weights that offline's canonical table does not.

This trace shows the clearest useful plan difference. Online performs more fine-grained budget adaptation and is faster than offline.

MRU comparison:

| budget | offline/table signature | online CP-SAT signature | MRU signature |
|---:|---|---|---|
| 4352 | `4200 -> (212,12,12,12,0)` | `(215,9,9,9,0)` | `(214,10,10,10,1)` |
| 4096 | `3800 -> (197,27,27,27,0)` | `(208,16,16,16,0)` | `(200,24,24,24,1)` |

Here the compact signature is:

```text
(gpu, disk, load, xform, evict)
```

At `4096 MiB`, MRU keeps 200 GPU weights, while CP-SAT keeps 208 and needs fewer planned loads. Example difference:

```text
MRU keeps on GPU but CP-SAT does not:
  blk.0.attn_q.weight
  blk.25.ffn_down.weight
  blk.25.ffn_gate.weight

CP-SAT keeps on GPU but MRU does not:
  blk.28.attn_q.weight
  blk.29.attn_output.weight
  blk.29.attn_q.weight
```

Interpretation:

MRU has some state awareness, but it is not cost-aware. It can retain weights because of recency/order even when CP-SAT prefers a different set with lower expected reload cost.

## Trace 02: user13

Performance summary:

```text
offline: 43.71 ms/token
online:  41.64 ms/token, excluding remote solver wall time
MRU:     43.77 ms/token
```

Runtime plan switches:

```text
offline applied:
  4600(load=0)

online applied:
  4864(load=0)
```

Plan comparison:

```text
offline band 4600: (224,0,0,0,0,0,0)
online budget 4864: (224,0,0,0,0,0,0)
```

Interpretation:

No meaningful plan difference. Both methods keep all managed weights GPU-resident, so this trace does not stress the planner.

## Trace 03: user116

Performance summary:

```text
offline: 219.97 ms/token
online:  218.96 ms/token, excluding remote solver wall time
MRU:     224.38 ms/token
```

Runtime plan switches:

```text
offline applied:
  5000(load=0), 3800(load=27), 4600(load=0)

online applied:
  5120(load=0), 4096(load=24), 4608(load=0)

MRU applied:
  5120(load=0), 4096(load=24), 4608(load=0)
```

Plan-level differences:

| online budget | offline band | offline signature | online signature | online-only GPU weights | offline-only GPU weights |
|---:|---:|---|---|---:|---:|
| 5120 | 5000 | `(224,0,0,0,0,0,0)` | `(224,0,0,0,0,0,0)` | 0 | 0 |
| 4096 | 3800 | `(197,0,27,27,27,27,0)` | `(200,0,24,24,24,24,3)` | 24 | 21 |
| 4608 | 4600 | `(224,0,0,0,0,0,0)` | `(224,0,0,0,0,0,0)` | 0 | 0 |

Example at `online budget=4096` vs `offline band=3800`:

```text
online keeps on GPU but offline does not:
  blk.0.attn_k.weight
  blk.0.attn_q.weight
  blk.0.ffn_up.weight

offline keeps on GPU but online does not:
  blk.22.ffn_down.weight
  blk.23.attn_k.weight
  blk.23.attn_output.weight
```

Interpretation:

The online solver uses a less conservative budget than the offline band: `4096` maps to offline `3800`, so online can keep 200 GPU weights instead of 197. It also changes which weights are preserved based on current state. The speed difference is small because this trace has only one significant low-budget region.

MRU comparison at `4096 MiB`:

```text
offline band 3800: (197,27,27,27,0)
online CP-SAT:     (200,24,24,24,3)
MRU:               (200,24,24,24,24)
```

MRU and CP-SAT keep the same number of GPU weights, but not the same set.

```text
MRU keeps on GPU but CP-SAT does not:
  blk.0.attn_output.weight
  blk.0.ffn_down.weight
  blk.0.ffn_gate.weight

CP-SAT keeps on GPU but MRU does not:
  blk.27.ffn_up.weight
  blk.28.attn_q.weight
  blk.28.ffn_down.weight
```

Interpretation:

This is a clean example where aggregate counts are similar but plan content differs. MRU is not simply a worse-size version of CP-SAT; it is choosing a different resident set because its objective is implicit recency, not predicted cost.

## Trace 04: user68

Runtime plan switches:

```text
offline applied:
  5500(load=0)

online applied:
  5632(load=0)
```

Plan comparison:

```text
offline band 5500: (224,0,0,0,0,0,0)
online budget 5632: (224,0,0,0,0,0,0)
```

Interpretation:

No meaningful plan difference. This is a high-budget trace for this model/setup.

## Trace 05: user74

Runtime plan switches:

```text
offline applied:
  6000(load=0)

online applied:
  5376(load=0)
```

Plan comparison:

```text
offline band 6000: (224,0,0,0,0,0,0)
online budget 5376: (224,0,0,0,0,0,0)
```

Interpretation:

No meaningful plan difference. Both keep all managed weights GPU-resident.

## Trace 06: user204

Runtime plan switches:

```text
offline applied:
  5500(load=0)

online applied:
  5632(load=0)
```

Plan comparison:

```text
offline band 5500: (224,0,0,0,0,0,0)
online budget 5632: (224,0,0,0,0,0,0)
```

Interpretation:

No meaningful plan difference. This trace is also high-budget for the 8B setup.

## Trace 07: user7

Performance summary:

```text
offline: 624.11 ms/token
online:  timeout, no perf footer at n192 or n96
MRU:     685.47 ms/token, n96 replacement
```

Runtime plan switches:

```text
offline applied:
  5500(load=0), 4200(load=12), 3000(load=51),
  3800(load=27), 4200(load=12), 4600(load=0),
  4200(load=12), 4600(load=0)

online applied before timeout:
  5376(load=0), 4352(load=9), 4608(load=0),
  3584(load=35), 3840(load=26), 4352(load=9)

MRU n96 applied:
  5632(load=0), 3584(load=55), 4096(load=24),
  4608(load=0), 4352(load=10), 4608(load=0)
```

Plan-level differences:

| online budget | offline band | offline signature | online signature | online-only GPU weights | offline-only GPU weights |
|---:|---:|---|---|---:|---:|
| 4352 | 4200 | `(212,0,12,12,12,12,0)` | `(215,0,9,9,9,9,0)` | 12 | 9 |
| 3584 | 3500 | `(187,0,37,37,37,37,0)` | `(189,0,35,35,35,35,0)` | 37 | 35 |
| 3840 | 3800 | `(197,0,27,27,27,27,0)` | `(198,0,26,26,26,26,0)` | 25 | 24 |

Examples:

```text
online budget=3584 keeps on GPU but offline band=3500 does not:
  blk.0.attn_k.weight
  blk.0.attn_v.weight
  blk.0.ffn_down.weight

offline band=3500 keeps on GPU but online budget=3584 does not:
  blk.13.ffn_down.weight
  blk.14.attn_k.weight
  blk.14.attn_output.weight
```

Interpretation:

This is the most memory-constrained trace. Online creates reasonable state-aware plans on paper: it keeps slightly more GPU weights than the coarser offline band at comparable budgets. However, the online execution timed out before reaching a perf footer.

So for this trace, the plan difference exists, but the runtime path is not yet robust enough to complete the online baseline under the 20-minute low-budget replay.

This trace should be used as the primary debug target for online timeout behavior.

MRU comparison:

| budget | offline/table signature | online CP-SAT signature | MRU signature |
|---:|---|---|---|
| 3584 | `3500 -> (187,37,37,37,0)` | `(189,35,35,35,0)` | `(169,55,55,55,55)` |
| 4096 | `3800 -> (197,27,27,27,0)` | no completed matching online plan in n96 log | `(200,24,24,24,24)` |
| 4352 | `4200 -> (212,12,12,12,0)` | `(215,9,9,9,0)` | `(214,10,10,10,10)` |

At `3584 MiB`, MRU is substantially different:

```text
offline table: 187 GPU, 37 disk, 37 planned loads
online CP-SAT: 189 GPU, 35 disk, 35 planned loads
MRU:           169 GPU, 55 disk, 55 planned loads
```

Example at `3584 MiB`:

```text
MRU keeps on GPU but CP-SAT does not:
  blk.13.ffn_down.weight
  blk.14.attn_k.weight
  blk.14.attn_output.weight

CP-SAT keeps on GPU but MRU does not:
  blk.23.ffn_gate.weight
  blk.23.ffn_up.weight
  blk.24.attn_output.weight
```

Interpretation:

This is the strongest MRU weakness case. Under severe pressure, MRU keeps only 169 GPU weights and schedules 55 loads, while CP-SAT's plan keeps 189 GPU weights and schedules 35 loads. MRU finished at `685.47 ms/token` for the n96 replacement, worse than offline n192 at `624.11 ms/token`.

This is useful for a slide: MRU is simple and robust enough to finish, but the plan quality is visibly worse under the hardest memory trace.

## Trace 08: user270

Performance summary:

```text
offline: 164.53 ms/token
online:  157.37 ms/token, n96 replacement, excluding remote wall time
MRU:     163.34 ms/token
```

Runtime plan switches:

```text
offline applied:
  4600(load=0), 4200(load=12), 4600(load=0),
  3800(load=27), 4600(load=0)

online n96 replacement applied:
  4608(load=0)

MRU applied:
  4608(load=0), 4352(load=10), 4608(load=0),
  4096(load=24), 4608(load=0), 4352(load=10), 4608(load=0)
```

Plan comparison for completed online run:

```text
offline band 4600: (224,0,0,0,0,0,0)
online budget 4608: (224,0,0,0,0,0,0)
```

Interpretation:

The completed online replacement run did not apply the low-budget plans that offline saw in the full n192 run, so this trace is not a clean plan-level comparison. The n192 online run timed out, while n96 completed in a high-residency state.

Use this trace cautiously in slides: it is useful as evidence that online can avoid low-budget table movement in the completed run, but not as a perfectly matched n192 offline-vs-online plan comparison.

MRU comparison:

| budget | offline/table signature | MRU signature |
|---:|---|---|
| 4352 | `4200 -> (212,12,12,12,0)` | `(214,10,10,10,10)` |
| 4096 | `3800 -> (197,27,27,27,0)` | `(200,24,24,24,24)` |

Example at `4096 MiB`:

```text
MRU keeps on GPU but offline table does not:
  blk.0.attn_k.weight
  blk.0.attn_output.weight
  blk.0.attn_q.weight

offline table keeps on GPU but MRU does not:
  blk.27.ffn_up.weight
  blk.28.attn_q.weight
  blk.28.ffn_down.weight
```

Interpretation:

MRU differs from offline in a predictable way: it tends to preserve a different, recency-shaped resident set. In this trace MRU completed and is close to offline in speed, but its plan is still not the same as the cost-model table.

## Trace 09: user11

Runtime plan switches:

```text
offline applied:
  5000(load=0)

online applied:
  5376(load=0)
```

Plan comparison:

```text
offline band 5000: (224,0,0,0,0,0,0)
online budget 5376: (224,0,0,0,0,0,0)
```

Interpretation:

No meaningful plan difference. High-budget trace.

## Trace 10: user115

Performance summary:

```text
offline: 107.88 ms/token
online:  106.16 ms/token, excluding remote wall time
MRU:     107.28 ms/token
```

Runtime plan switches:

```text
offline applied:
  4200(load=12), 5000(load=0)

online applied:
  4352(load=13), 5376(load=0)

MRU applied:
  4864(load=0)
```

Plan-level differences:

| online budget | offline band | offline signature | online signature | online-only GPU weights | offline-only GPU weights |
|---:|---:|---|---|---:|---:|
| 4352 | 4200 | `(212,0,12,12,12,12,0)` | `(211,0,13,13,13,13,4)` | 8 | 9 |
| 5376 | 5000 | `(224,0,0,0,0,0,0)` | `(224,0,0,0,0,0,0)` | 0 | 0 |

Example at `online budget=4352` vs `offline band=4200`:

```text
online keeps on GPU but offline does not:
  blk.1.ffn_gate.weight
  blk.1.ffn_up.weight
  blk.2.ffn_down.weight

offline keeps on GPU but online does not:
  blk.0.attn_k.weight
  blk.0.attn_q.weight
  blk.0.attn_v.weight
```

Interpretation:

This trace has a modest low-budget region. Online and offline both require around a dozen planned load events, but choose different resident sets. Online is slightly faster after excluding remote wall time.

MRU comparison:

MRU never entered a low-load plan in the completed run:

```text
MRU budget 4864: (224,0,0,0,0)
```

So for trace10, MRU is not useful as a low-budget resident-set comparison. It is mainly a completed runtime baseline.

## Overall Takeaways

1. **High-budget traces have no meaningful plan difference.**

   In traces 02, 04, 05, 06, and 09, both offline and online keep all 224 managed weights GPU-resident:

   ```text
   (224,0,0,0,0,0,0)
   ```

2. **Low-budget traces show real offline/online plan differences.**

   In traces 01, 03, 07, and 10, online uses exact bucketed budgets and current residency state. This changes both:

   - how many weights stay GPU-resident
   - which specific layers/weights stay GPU-resident

3. **Offline table is coarse and state-blind.**

   Example:

   ```text
   online budget 4096 -> online solves at 4096
   offline maps 4096 -> table band 3800
   ```

   That gap can make offline more conservative than necessary.

4. **Online CP-SAT can improve plan quality, but runtime robustness is still an issue.**

   Trace07 shows this clearly: online plans are reasonable structurally, but execution times out under the hardest low-budget 20-minute replay.

5. **MRU is stateful but not cost-aware.**

   MRU is not the same as offline and not the same as online CP-SAT. It can preserve a recency-shaped resident set, but it does not optimize measured reload / transfer / xform cost.

   The clearest example is trace07 at `3584 MiB`:

   ```text
   offline table: 187 GPU, 37 planned loads
   online CP-SAT: 189 GPU, 35 planned loads
   MRU:           169 GPU, 55 planned loads
   ```

   This explains why MRU can be significantly worse under severe memory pressure.

6. **For presentation, use trace01 as the cleanest positive example.**

   Trace01 has:

   - completed offline and online runs
   - meaningful low-budget plan differences
   - online faster than offline
   - concrete weight-level resident-set differences

   Trace07 is better as a limitation / stress-test slide, and also the best slide for showing why MRU is not enough.
