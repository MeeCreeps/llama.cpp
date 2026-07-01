# Structured Diff Design for Dynamic Memory Budget Inference

## 1. Problem Framing

In mobile LLM inference, the available memory budget is not fixed.  The budget
can shrink or expand because of OS pressure, foreground apps, thermal/runtime
policy, or other system activity.  A planner therefore cannot only answer:

```text
What is the best plan under budget B?
```

It must answer a stateful question:

```text
Given the current runtime memory state, what should the next plan be under the
new budget B_t, and how much adaptation cost should we pay now?
```

The key research trade-off is:

```text
budget-optimal target quality
    The plan that gives good steady-state inference performance under B_t.

adjustment cost from current state
    The one-time or short-term cost to transform the current runtime state into
    that plan.
```

This trade-off is the center of the design.  The system should not blindly
chase the best target plan if the transition cost dominates the benefit over
the expected future decoding horizon.  It also should not stay too close to the
old state when the budget change creates a meaningful opportunity to improve
decode performance.

## 2. Baseline Story

### Offline Budget Table

The offline baseline precomputes a plan for each memory budget.  It is cheap at
runtime because budget changes become table lookups.

Its weakness is that it is state-unaware.  Two executions can have the same
new budget but very different current residency states.  The offline table
does not know whether a weight is already resident on GPU, resident on CPU,
still in disk layout, partially materialized, or waiting for deferred eviction.

This means the offline plan may look optimal in isolation but expensive to
reach from the actual runtime state.

### Online CP-SAT

The online baseline solves a CP-SAT problem when the budget changes.  This is
not a different solver family from the offline CP formulation; the difference
is the problem input and objective.  Online solving can include current
runtime residency and transition costs, so it can select a plan that is more
aware of actual system state.

Its weakness is online latency.  A full solve at every budget change is too
expensive for the runtime path, especially on a phone where the planner shares
resources with inference and OS activity.

### Research Gap

We want most of the state-awareness benefit of online CP-SAT while keeping the
runtime overhead close to a table lookup.  This motivates an incremental
method:

```text
offline: prepare a small, diverse set of good candidate plans
online: choose and adjust candidates using the current runtime state
```

## 3. State Definitions

The design should keep three objects separate:

```text
P_prev
    The last logical plan selected by the planner.

S_runtime
    The actual current residency/materialization state observed by runtime.

P_target(B_t)
    A candidate target plan for the current budget.
```

The real transition cost is:

```text
Diff(S_runtime, P_target)
```

not only:

```text
Diff(P_prev, P_target)
```

`P_prev` is still useful as a structural reference: it tells us what the
planner intended and can help identify stable plan identity across budget
changes.  But it should not be treated as complete truth because runtime can
diverge from the logical plan through deferred evictions, missed
materialization, failed transitions, or keep-current decisions.

## 4. Offline Candidate Preparation

Instead of saving only one optimal plan per budget, preserve top-k diverse
plans:

```text
C(B) = {P_1, P_2, ..., P_k}
```

Each candidate should be good under the budget, but diversity is essential.
If all candidates have similar placement, online adaptation has no useful
choice.  Candidate diversity can come from:

```text
different CPU/GPU/disk placement ratios
different layer groups kept resident
different attention-vs-FFN placement choices
objective perturbation
distance constraints between selected plans
```

The offline artifact should store enough metadata for cheap online scoring:

```text
steady predicted decode cost
CPU/GPU memory footprint
placement bitsets
layout/transform requirements
load and prepare bytes
coarse layer/component summaries
```

We can also precompute plan-to-plan transition summaries:

```text
T(P_i -> P_j)
```

This is useful when `P_prev` is close to runtime state.  But online scoring
must still be able to recompute transition cost from `S_runtime`.

## 5. Online Candidate Selection

When budget changes to `B_t`, the online planner first evaluates the small
candidate set:

```text
C(B_t) = {P_1, P_2, ..., P_k}
```

For each candidate:

```text
score(P) =
    steady_cost(P)
  + alpha * transition_cost(S_runtime -> P)
  + beta  * memory_pressure_risk(P, B_t)
  + gamma * pipeline_disruption(P)
```

This step chooses a base target:

```text
P_base = argmin score(P)
```

This gives a simple first research contribution: online does not solve CP-SAT,
but it is still state-aware because it scores each offline candidate against
the real runtime state.

However, choosing one whole candidate is still coarse.  A candidate may contain
some changes that are worth applying and some that are not.  This is where the
fourth design point becomes important.

## 6. Core Design: Structured Plan Diff

The fourth point should be framed as:

```text
Do not treat the plan difference as a flat list of independent weight moves.
Represent it as a structured diff, then accept, reject, or refine parts of the
diff according to benefit, transition cost, memory feasibility, and expected
future horizon.
```

The planner constructs:

```text
D = Diff(S_runtime, P_base)
```

This diff contains placement, layout, and materialization changes needed to
move from the actual runtime state toward the selected candidate plan.

### 6.1 Diff Unit

A diff unit is the smallest change the planner may eventually decide on:

```text
tensor placement:
    blk.12.attn_q.weight CPU -> GPU

layout transform:
    raw CPU layout -> GPU transformed layout

materialization:
    disk-only -> CPU staged

eviction:
    GPU resident -> evicted or demoted
```

But the online planner should not begin at this smallest granularity.  Per
tensor decisions are too noisy and can miss shared cost.

### 6.2 Diff Hierarchy

Start from coarse semantic groups and expand only when needed:

```text
model
  layer group
    layer
      attention
        qkv group
        output projection
      ffn
        gate/up group
        down projection
      individual tensor
```

This hierarchy is useful because LLM inference has repeated structure.  A
whole attention block promotion or FFN demotion often has a clearer cost and
benefit than many isolated tensor moves.

### 6.3 Diff Graph

The hierarchy gives a tree, but the real system also has cross edges.  A graph
better captures coupling:

```text
execution locality
    tensors used close together in the decode graph

resource sharing
    changes share disk load, CPU prepare, GPU transform, or eviction path

file locality
    tensors are near each other in the GGUF file

memory coupling
    accepting one promotion may require another eviction

pipeline overlap
    multiple transforms can be hidden under the same compute window
```

Research framing: the graph is not just an implementation data structure.  It
encodes the reason incremental planning can be both cheaper and higher quality
than flat candidate selection.  It lets the system make decisions at the
granularity where cost and benefit are actually shared.

## 7. Accept, Reject, Expand

For each diff node or connected subgraph, compute:

```text
steady_gain(node, H)
    Expected decode-time improvement over horizon H tokens.

transition_cost(node)
    Load, prepare, transform, eviction, and synchronization cost.

effective_transition_cost(node)
    The visible transition cost after pipeline overlap.

memory_delta(node)
    Additional CPU/GPU memory pressure if this diff is accepted.

risk(node)
    Risk from budget instability, fragmentation, or uncertain profile data.
```

A simple score:

```text
score(node) =
    steady_gain(node, H)
  - effective_transition_cost(node)
  - memory_pressure_penalty(node)
  - risk_penalty(node)
```

Decision rule:

```text
if score(node) is clearly positive and memory is feasible:
    accept the whole node

elif score(node) is clearly negative:
    reject the whole node and keep the current runtime state for this region

else:
    expand the node into smaller structured children
```

This is the main design idea for point 4.  The planner spends online work only
where the decision is uncertain.  Easy large decisions stay coarse; difficult
regions are refined.

## 8. Horizon-Aware Trade-off

The horizon `H` is important.  If the budget is likely to last for many future
tokens, a costly transition may be worth paying.  If the budget is unstable,
the planner should be conservative.

For example:

```text
steady_gain(node, H) =
    H * per_token_gain(node)
```

Then:

```text
accept if H * per_token_gain > visible_transition_cost + penalties
```

This gives a clean systems argument:

```text
The planner adapts not only to the current memory budget, but also to the
expected lifetime of that budget.
```

For an ASPLOS-style system paper, this is a strong angle because it connects
algorithm design to runtime dynamics.

## 9. Example: Partial Candidate Adoption

Suppose `P_base` wants to promote an entire layer's attention and FFN weights
from CPU to GPU.

A whole-plan switch applies every change.  A flat diff evaluates many tensor
moves independently.

The structured diff first evaluates:

```text
Layer 12 CPU -> GPU promotion
```

If this is too expensive, it expands:

```text
Layer 12
  attention promotion
  FFN promotion
```

The planner may find:

```text
attention promotion:
    high gain, good locality, transition cost can be overlapped
    -> accept

FFN promotion:
    larger memory delta, higher transform cost, low benefit under short H
    -> reject or expand further
```

The result is not exactly `S_runtime` and not exactly `P_base`.  It is an
adjusted plan:

```text
P_adjusted = S_runtime + accepted structured diffs
```

This is the research distinction from candidate selection.  Candidate
selection asks which precomputed plan is closest to the current state.
Structured diff asks which parts of that candidate are actually worth paying
for now.

## 10. Runtime Algorithm Sketch

```text
input:
    current budget B_t
    runtime state S_runtime
    previous logical plan P_prev
    offline candidates C(B_t)
    expected horizon H

1. Score each candidate P in C(B_t) against S_runtime.
2. Select P_base with the best whole-plan score.
3. Construct D = Diff(S_runtime, P_base).
4. Initialize frontier with coarse diff nodes.
5. While frontier is non-empty and online time budget remains:
       pop a node
       estimate gain, cost, memory delta, risk
       accept, reject, or expand
6. Repair feasibility if accepted diffs exceed memory budget.
7. Emit P_adjusted and log accepted/rejected/expanded decisions.
```

The online time budget is part of the design.  The method should degrade
gracefully: if the planner runs out of time, it can apply already accepted
coarse decisions and keep the rest of the runtime state unchanged.

## 11. What To Measure

The evaluation should show both performance and mechanism.

Baselines:

```text
offline-single
    one precomputed plan per budget

online-cpsat
    full state-aware online solve

candidate-select
    top-k offline candidates with runtime transition scoring

structured-diff
    candidate selection plus accept/reject/expand over structured diff
```

Metrics:

```text
decode ms/token
planner decision time
transition load/prepare/evict bytes
visible transition stall
number of accepted/rejected/expanded nodes
distance from online CP-SAT plan quality
memory budget violation count
budget-change recovery time
```

A good headline claim would be:

```text
Structured-diff planning preserves most of the state-aware quality of online
CP-SAT while reducing budget-switch planning overhead to bounded candidate
scoring and selective graph refinement.
```

## 12. Open Research Questions

1. What is the best initial hierarchy: layer-first, component-first, or
   profile-derived clusters?
2. How should the horizon `H` be estimated from recent budget stability?
3. Should top-k offline candidates be diverse by placement distance, by
   predicted transition behavior, or by semantic layer/component patterns?
4. How much graph structure is necessary before the method beats whole-plan
   candidate selection?
5. Can the system learn accept/reject thresholds from measured phone traces
   instead of manually tuning them?

## 13. Current Working Hypothesis

The most promising design is:

```text
offline:
    preserve top-k diverse budget candidates and compact placement summaries

online:
    select a target candidate using S_runtime-aware transition cost

incremental:
    transform Diff(S_runtime, P_base) into a structured hierarchy/graph
    accept obvious beneficial subgraphs
    reject obviously bad subgraphs
    expand uncertain subgraphs until the online planning budget is reached
```

This directly targets the core trade-off: it can move toward a
budget-appropriate target plan when the benefit is large, while avoiding
unnecessary transition work when the current runtime state is already good
enough for the expected future horizon.
