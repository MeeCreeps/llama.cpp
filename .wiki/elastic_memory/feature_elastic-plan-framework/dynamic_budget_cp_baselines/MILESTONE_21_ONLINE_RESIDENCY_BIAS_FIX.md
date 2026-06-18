# MILESTONE 21: Fix online residency bias that made online slower than offline

Date: 2026-06-17

## Problem

Online remote CP-SAT was sometimes slower than the offline table on the same dynamic trace. That is not expected: online may pay extra provider/apply overhead, but the selected resident set should not be worse than offline when the current state does not provide useful information.

Root cause:

```text
resident weight value = 1e9 + size_mb
```

This made online almost pin the previous resident set. Under a budget drop, the solver kept cheap stale tensors only because they were already resident, and evicted tensors with much higher future reload cost. That can be worse than the offline global selection.

There was a second issue in the online/native path:

```text
missing disk_available flag => not manageable => forced keep
```

If the Android state dump is incomplete or transient, this also pins online to stale state.

## Fix

### Python remote CP-SAT solver

File:

```text
runtime/plan/dynamic_budget_solver.py
```

Changed resident value from an infinite bonus to a finite churn bonus:

```text
value(w) = reload_cost(w) + min(0.25 * reload_cost(w), 2.0 ms)
```

This still slightly favors keeping a useful resident tensor, but it no longer overrides the real reload-cost objective.

Also removed forced keep for resident tensors that lack `disk_available` in the state dump. Model weights in this planner are disk-backed elastic weights, so missing state metadata should not make them immovable.

### Native online / MRU generator

File:

```text
tools/main/main.cpp
```

Mirrored the same finite churn bonus and made model-meta items manageable by default, instead of pinning them when a transient disk flag is missing.

## Local plan sanity check

Simulated the first online solve after model load:

```text
state: all 224 weights disk_available + gpu_compute_resident
budget: 4096 MiB
kv: 512 MiB
misc: 256 MiB
```

Result:

```text
offline table 4096:
  disk weights: 16
  timeline events: 48

online CP-SAT 4096 with state:
  disk weights: 16
  timeline events: 64
  disk set equal to offline: true
```

The extra 16 online events are evicts from the current GPU-resident state. The resident/disk selection itself now matches offline instead of being distorted by stale state.

## OP13 validation: user147 min-start low-budget window

Trace:

```text
trace_01_user_147_minstart_5min.csv
first budget: 3752.1 MB
first applied plan bucket: 4096 MiB
```

### Offline table, n16

```text
apply_ms: 292.608
eval: 467.39 ms/token
reload host-issue: 7095.7 ms, 768 calls
direct O_DIRECT read: 5229.03 ms, 784 calls, 13881.7 MB
```

### Online remote CP-SAT, n16

```text
server_solve_ms: 390.690
remote_wall_ms: 545.337
provider_get_ms: 570.144
apply_ms: 338.651
eval: 463.06 ms/token
reload host-issue: 7010.0 ms, 768 calls
direct O_DIRECT read: 5164.53 ms, 784 calls, 13881.7 MB
```

## Interpretation

After this fix, online is no longer slower because of a worse resident set. The n16 validation is slightly faster online:

```text
offline: 467.39 ms/token
online:  463.06 ms/token
```

The difference is small because both plans select the same 16 disk weights at the first low-budget bucket. This is expected for the first solve: online has no beneficial history yet beyond the initial all-resident state.

Larger online gains should come from later budget changes where state-aware planning avoids unnecessary churn, but those runs need to be re-run after this fix and with one llama-cli process at a time.

