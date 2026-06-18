# Concrete Plan Examples: Offline vs Online CP-SAT vs MRU

The aggregate speed differences are not always large, so this document focuses on concrete plan examples. Each example fixes one trace and one memory budget region, then compares what the three planners actually choose.

Compact signature:

```text
(gpu, disk, load, xform, evict)
```

where:

- `gpu`: number of managed weights planned GPU-resident
- `disk`: number of managed weights planned disk-backed / loaded on demand
- `load`: number of planned LOAD events
- `xform`: number of planned layout transform events
- `evict`: number of planned evictions

## Example 1: trace01_user147 at ~4096 MiB

This is a useful example because all three methods complete, and the plan difference is real even though speed is close.

### Plan Counts

| method | effective budget | signature | meaning |
|---|---:|---|---|
| offline table | 3800 band | `(197, 27, 27, 27, 0)` | 197 GPU weights, 27 planned loads |
| online CP-SAT | 4096 | `(208, 16, 16, 16, 0)` | 208 GPU weights, 16 planned loads |
| MRU | 4096 | `(200, 24, 24, 24, 1)` | 200 GPU weights, 24 planned loads |

### What Online Keeps That Offline Does Not

Online keeps 26 GPU weights that offline does not. They are mostly early-layer FFN weights:

```text
L0 blk.0.attn_k.weight
L0 blk.0.attn_output.weight
L0 blk.0.ffn_down.weight
L0 blk.0.ffn_gate.weight
L0 blk.0.ffn_up.weight
L1 blk.1.ffn_down.weight
L1 blk.1.ffn_gate.weight
L1 blk.1.ffn_up.weight
```

Offline keeps 15 GPU weights that online does not, mostly later-layer FFN weights:

```text
L25 blk.25.ffn_down.weight
L25 blk.25.ffn_gate.weight
L25 blk.25.ffn_up.weight
L26 blk.26.ffn_down.weight
L26 blk.26.ffn_gate.weight
L26 blk.26.ffn_up.weight
L27 blk.27.ffn_down.weight
L27 blk.27.ffn_gate.weight
```

Interpretation:

```text
offline table is tied to a canonical 3800 MiB band
online solves the actual 4096 MiB budget and current state
```

So online can keep 11 more GPU weights and reduce planned loads from 27 to 16.

### MRU vs Online

MRU keeps 9 GPU weights that online does not:

```text
L0 blk.0.attn_q.weight
L25 blk.25.ffn_down.weight
L25 blk.25.ffn_gate.weight
L25 blk.25.ffn_up.weight
L26 blk.26.ffn_down.weight
L26 blk.26.ffn_gate.weight
L26 blk.26.ffn_up.weight
L27 blk.27.ffn_down.weight
```

Online keeps 17 GPU weights that MRU does not:

```text
L28 blk.28.attn_q.weight
L29 blk.29.attn_output.weight
L29 blk.29.attn_q.weight
L30 blk.30.attn_k.weight
L30 blk.30.attn_output.weight
L30 blk.30.attn_q.weight
L30 blk.30.attn_v.weight
L30 blk.30.ffn_down.weight
```

Interpretation:

MRU is not simply between offline and online. It has its own recency-shaped resident set. CP-SAT uses the cost model and current state to choose a different set.

### Load Timeline Difference

Offline planned loads start from early layers:

```text
blk.0.attn_k.weight        anchor 0
blk.0.attn_output.weight   anchor 1
blk.0.attn_q.weight        anchor 1
blk.0.ffn_down.weight      anchor 3
blk.0.ffn_gate.weight      anchor 4
blk.0.ffn_up.weight        anchor 5
```

Online planned loads are shifted to later weights:

```text
blk.0.attn_q.weight        anchor 1
blk.25.ffn_down.weight     anchor 178
blk.25.ffn_gate.weight     anchor 179
blk.25.ffn_up.weight       anchor 180
blk.26.ffn_down.weight     anchor 185
blk.26.ffn_gate.weight     anchor 186
```

MRU planned loads are even later:

```text
blk.27.ffn_up.weight       anchor 194
blk.28.attn_q.weight       anchor 197
blk.28.ffn_down.weight     anchor 199
blk.28.ffn_gate.weight     anchor 200
blk.28.ffn_up.weight       anchor 201
```

This is the clearest visual point:

```text
offline reloads early weights
online reloads a different set based on current state
MRU reloads a recency-shaped tail set
```

## Example 2: trace01_user147 at ~4352 MiB

This is a cleaner, smaller example for one PPT slide.

### Plan Counts

| method | effective budget | signature | meaning |
|---|---:|---|---|
| offline table | 4200 band | `(212, 12, 12, 12, 0)` | 212 GPU weights, 12 planned loads |
| online CP-SAT | 4352 | `(215, 9, 9, 9, 0)` | 215 GPU weights, 9 planned loads |
| MRU | 4352 | `(214, 10, 10, 10, 1)` | 214 GPU weights, 10 planned loads |

### Online vs Offline

Online keeps these early FFN weights on GPU:

```text
L0 blk.0.ffn_down.weight
L0 blk.0.ffn_gate.weight
L0 blk.0.ffn_up.weight
L1 blk.1.ffn_down.weight
L1 blk.1.ffn_gate.weight
L1 blk.1.ffn_up.weight
L2 blk.2.ffn_down.weight
L2 blk.2.ffn_gate.weight
```

Offline instead keeps mid-layer weights:

```text
L12 blk.12.ffn_down.weight
L12 blk.12.ffn_up.weight
L13 blk.13.attn_k.weight
L13 blk.13.attn_v.weight
L13 blk.13.ffn_down.weight
L13 blk.13.ffn_gate.weight
L13 blk.13.ffn_up.weight
```

### MRU vs Online

MRU keeps mid-layer weights:

```text
L12 blk.12.ffn_down.weight
L12 blk.12.ffn_up.weight
L13 blk.13.attn_k.weight
L13 blk.13.attn_v.weight
L13 blk.13.ffn_down.weight
```

Online keeps late-layer weights that MRU does not:

```text
L30 blk.30.attn_output.weight
L30 blk.30.attn_q.weight
L30 blk.30.ffn_down.weight
L30 blk.30.ffn_gate.weight
L30 blk.30.ffn_up.weight
L31 blk.31.attn_output.weight
L31 blk.31.attn_q.weight
L31 blk.31.ffn_down.weight
```

### Load Timeline Difference

Offline loads early FFN weights:

```text
blk.0.ffn_down.weight      anchor 3
blk.0.ffn_gate.weight      anchor 4
blk.0.ffn_up.weight        anchor 5
blk.1.ffn_down.weight      anchor 10
blk.1.ffn_gate.weight      anchor 11
```

Online loads mid-layer weights:

```text
blk.12.ffn_down.weight     anchor 87
blk.12.ffn_up.weight       anchor 89
blk.13.attn_k.weight       anchor 90
blk.13.attn_v.weight       anchor 93
blk.13.ffn_down.weight     anchor 94
```

MRU loads late-layer weights:

```text
blk.30.attn_output.weight  anchor 210
blk.30.attn_q.weight       anchor 211
blk.30.ffn_down.weight     anchor 213
blk.30.ffn_gate.weight     anchor 214
blk.30.ffn_up.weight       anchor 215
```

This example is useful because the counts are close:

```text
offline: 12 loads
online:   9 loads
MRU:     10 loads
```

but the actual loaded weights are clearly different.

## Example 3: trace07_user7 at ~3584 MiB

This is the strongest low-budget example. It is also the online timeout stress case.

### Plan Counts

| method | effective budget | signature | meaning |
|---|---:|---|---|
| offline table | 3500 band | `(187, 37, 37, 37, 0)` | 187 GPU weights, 37 planned loads |
| online CP-SAT | 3584 | `(189, 35, 35, 35, 0)` | 189 GPU weights, 35 planned loads |
| MRU | 3584 | `(169, 55, 55, 55, 55)` | 169 GPU weights, 55 planned loads |

This is the best example for explaining why MRU is not enough.

### Online vs Offline

Online keeps early/current-state weights that offline does not:

```text
L0 blk.0.attn_k.weight
L0 blk.0.attn_v.weight
L0 blk.0.ffn_down.weight
L0 blk.0.ffn_gate.weight
L0 blk.0.ffn_up.weight
L1 blk.1.attn_k.weight
L1 blk.1.ffn_down.weight
L1 blk.1.ffn_gate.weight
```

Offline keeps a different mid/late set:

```text
L13 blk.13.ffn_down.weight
L14 blk.14.attn_k.weight
L14 blk.14.attn_output.weight
L14 blk.14.attn_v.weight
L14 blk.14.ffn_down.weight
L14 blk.14.ffn_gate.weight
L14 blk.14.ffn_up.weight
```

Online keeps 2 more GPU weights and schedules 2 fewer planned loads than offline at this budget region.

### MRU vs Online

MRU keeps these weights that CP-SAT does not:

```text
L13 blk.13.ffn_down.weight
L14 blk.14.attn_k.weight
L14 blk.14.attn_output.weight
L14 blk.14.attn_v.weight
L14 blk.14.ffn_down.weight
L14 blk.14.ffn_gate.weight
L14 blk.14.ffn_up.weight
```

CP-SAT keeps many later-layer weights that MRU does not:

```text
L23 blk.23.ffn_gate.weight
L23 blk.23.ffn_up.weight
L24 blk.24.attn_output.weight
L24 blk.24.attn_q.weight
L24 blk.24.ffn_down.weight
L24 blk.24.ffn_gate.weight
L24 blk.24.ffn_up.weight
L25 blk.25.attn_output.weight
```

The set difference is large:

```text
MRU-only GPU weights:    25
CP-SAT-only GPU weights: 45
```

### Load Timeline Difference

Offline loads early weights:

```text
blk.0.attn_k.weight      anchor 0
blk.0.attn_v.weight      anchor 2
blk.0.ffn_down.weight    anchor 3
blk.0.ffn_gate.weight    anchor 4
blk.0.ffn_up.weight      anchor 5
```

Online loads mid-layer weights:

```text
blk.13.ffn_down.weight   anchor 94
blk.14.attn_k.weight     anchor 97
blk.14.attn_output.weight anchor 98
blk.14.attn_v.weight     anchor 100
blk.14.ffn_down.weight   anchor 101
```

MRU loads later weights:

```text
blk.23.ffn_gate.weight   anchor 165
blk.23.ffn_up.weight     anchor 166
blk.24.attn_output.weight anchor 168
blk.24.attn_q.weight     anchor 169
blk.24.ffn_down.weight   anchor 171
```

### Why This Matters

The performance result for this trace:

```text
offline n192: 624.11 ms/token
online:       timeout at n192 and n96
MRU n96:      685.47 ms/token
```

The plan tells a clearer story than speed alone:

```text
MRU schedules 55 loads at 3584 MiB
CP-SAT schedules 35 loads at 3584 MiB
offline schedules 37 loads at the 3500 band
```

MRU finishes, but its plan quality is visibly worse under severe memory pressure.

## Slide-Friendly Takeaway

If the speed numbers look close, the plan examples still show the planner behavior:

```text
Offline:
  fixed budget-band plan
  state-blind
  can reload weights that are already less useful for current state

Online CP-SAT:
  exact current budget
  state-aware
  cost-aware resident set

MRU:
  stateful heuristic
  not cost-aware
  can preserve a recency-shaped set and trigger more reloads
```

The strongest single example:

```text
trace07_user7, budget around 3584 MiB

offline table: 187 GPU, 37 loads
online CP-SAT: 189 GPU, 35 loads
MRU:           169 GPU, 55 loads
```

This is the cleanest evidence that the planners are making different decisions, even when end-to-end speed is partially masked by runtime overhead, direct disk reads, and synchronous stage execution.

