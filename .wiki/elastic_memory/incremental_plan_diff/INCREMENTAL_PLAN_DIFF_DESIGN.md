# Incremental Plan Diff Design

## Motivation

The current online planner can produce a better plan than an offline budget
table because it observes the current runtime weight residency.  However, a
full online CP-SAT solve is expensive.  The research goal is therefore:

```text
Keep most of the quality advantage of online planning, but make online
decision time close to a lightweight table lookup.
```

The key trade-off is between:

```text
budget-optimal target plan
    A plan optimized for the current budget, ignoring current runtime state.

adjustment from last/current plan
    A plan that reuses current resident/transformed weights and avoids
    redundant load/prepare/evict work.
```

The central observation is that runtime cost is not only the steady-state cost
of the selected target plan.  The runtime must also pay the adaptation cost:

```text
current residency state -> selected plan
```

This adaptation cost is structured.  Weight placement changes inside the same
layer, component, file locality region, or pipeline window often share load,
prepare, memory pressure, and overlap behavior.  Therefore, online adaptation
should not treat all weight-level diffs as independent flat choices.

## Four-Step Design

### 1. Offline: Preserve Top-K Diverse Plans

For each memory budget bucket, offline planning should preserve multiple
candidate plans instead of only the single predicted optimum:

```text
plans[bucket] = [P0, P1, ..., P{K-1}]
```

Each candidate should be high quality under the budget, but candidates should
also have diverse placements.  The diversity matters because online selection
needs alternatives that differ in transition cost from the current runtime
state.

Candidate diversity can be created by:

```text
objective perturbation
    Add small random or structured perturbations to placement costs.

distance constraints
    Require a new candidate to differ from existing candidates by a minimum
    placement Hamming distance.

placement profile constraints
    Force different CPU/GPU/disk ratios or different layer groups to stay
    resident.

multiple solvers/heuristics
    Mix CP-SAT, greedy, and local-search candidates.
```

Each plan should store:

```text
budget_mib
steady_cost_ms_per_token
predicted_pipeline_makespan
placement bitsets:
    GPU resident weights
    CPU resident weights
    DISK weights
memory bytes:
    gpu_bytes
    cpu_bytes
    disk_bytes
schedule summary:
    load bytes
    prepare bytes
    expected overlap window
```

### 2. Offline: Precompute Transition Overhead Between Plans

For each budget bucket and candidate pair, precompute the transition overhead:

```text
T(P_i -> P_j)
```

This can be computed from placement bitset differences:

```text
GPU_i xor GPU_j
CPU_i xor CPU_j
DISK_i xor DISK_j
```

Transition cost should include:

```text
disk load cost
    disk -> CPU staging

CPU prepare cost
    CPU staging -> CPU resident compute layout

GPU prepare cost
    CPU staging / raw source -> GPU transformed layout

evict cost
    releasing old CPU/GPU residency

pipeline disruption
    work that cannot be hidden by the upcoming compute window
```

The precomputed pairwise cost is useful when the previous plan is known:

```text
score(P_j) = steady_cost(P_j) + alpha * T(P_prev -> P_j)
```

However, runtime state can differ from the last logical plan because of:

```text
deferred evictions
foreground misses
failed or skipped materialization
retained allocations whose contents are not resident
budget oscillation
```

So the online path should also be able to compute:

```text
T(S_runtime -> P_j)
```

where `S_runtime` is the actual current residency state.

### 3. Online: Select Candidate By Current Runtime State

When a new budget bucket arrives, online does not solve CP-SAT.  It evaluates a
small candidate set:

```text
candidates = plans[current_bucket]
```

For each candidate:

```text
score(P) =
    steady_cost(P)
  + alpha * transition_cost(S_runtime -> P)
  + beta  * memory_pressure_risk(P)
  + gamma * pipeline_disruption(P)
```

Then choose the lowest score:

```text
P_base = argmin score(P)
```

This keeps online overhead low:

```text
O(K * N_weights)
```

where `K` is small, e.g. 4 or 8.

This step is already stronger than offline single-plan selection because it
uses actual runtime residency.  It is much cheaper than online CP-SAT because
it only scores precomputed candidates.

### 4. Online: Graph-Structured Plan Diff Expansion

Candidate selection alone still requires either fully switching to `P_base` or
staying close to the previous state.  The fourth step refines this by operating
on the plan difference:

```text
D = Diff(S_runtime, P_base)
```

The diff should be organized as an adaptation graph, not as a flat list of
independent weights.

#### Diff Graph

Each node is a diff unit:

```text
weight placement change:
    blk.12.ffn_up.weight: CPU -> GPU
```

Edges encode coupling between diff units:

```text
execution adjacency
    weights are used close together in the decode graph

layer/component relation
    weights belong to the same layer, attention block, or FFN block

shared transition resource
    weights use the same disk load, CPU prepare, GPU prepare, or evict path

file locality
    weights are adjacent in the GGUF file and may benefit from locality

memory coupling
    accepting one diff can force eviction of another resident weight

pipeline overlap relation
    prepare work can be hidden under the same compute window
```

This graph can be built first using a fixed LLM hierarchy:

```text
root
  layer group: blk.0-3
    layer: blk.0
      attention
        q/k/v/o
      ffn
        gate/up/down
```

Later it can be extended with profile-derived edges:

```text
edge weight = execution proximity + resource sharing + file locality
```

#### Expand Operation

`expand` is not just tree traversal.  It is an adaptive refinement operation:

```text
Only expose finer-grained diff decisions when the coarse decision is uncertain.
```

The planner starts from coarse supernodes:

```text
layer group
layer
component
tensor group
weight
```

For each supernode, compute:

```text
transition_cost(node)
steady_benefit(node)
memory_delta(node)
overlap_ratio(node)
confidence(node)
```

Then:

```text
if node is clearly beneficial:
    accept the whole connected subgraph

elif node is clearly bad:
    reject the whole connected subgraph

else:
    expand the node into smaller connected subgraphs
```

This gives a tunable online overhead:

```text
coarse decisions when easy
fine decisions only when needed
```

#### Cost Model

For a diff node or subgraph:

```text
transition_cost_ms =
    disk_load_bytes / disk_bw
  + cpu_prepare_bytes / cpu_prepare_bw
  + gpu_prepare_bytes / gpu_prepare_bw
  + evict_cost_ms
```

The steady-state gain over a horizon `H` tokens:

```text
steady_gain_ms =
    H * (old_compute_ms_per_token - new_compute_ms_per_token)
```

The effective transition cost should consider overlap:

```text
effective_transition_cost =
    transition_cost_ms * (1 - overlap_ratio)
```

Node score:

```text
score(node) =
    steady_gain_ms
  - effective_transition_cost
  - memory_pressure_penalty
  - fragmentation_penalty
```

Accept/reject/expand:

```text
accept if:
    score(node) > accept_threshold
    and memory is feasible

reject if:
    score(node) < reject_threshold

expand if:
    reject_threshold <= score(node) <= accept_threshold
    or children have high cost variance
    or memory_delta(node) is too large for one coarse decision
```

#### Example

Suppose the candidate wants to move three FFN weights:

```text
blk.12.ffn_gate.weight: CPU -> GPU
blk.12.ffn_up.weight:   CPU -> GPU
blk.12.ffn_down.weight: CPU -> GPU
```

A flat policy evaluates three independent changes.

The diff graph first evaluates:

```text
FFN(blk.12): CPU -> GPU
```

If the whole FFN subgraph has high score, accept all three weights.  If the
score is uncertain, expand:

```text
FFN(blk.12)
  gate + up
  down
```

This captures shared execution locality and pipeline overlap that flat
weight-level decisions miss.

## Runtime Algorithm

```text
input:
    current budget B_t
    current runtime state S_runtime
    previous selected plan P_prev
    candidate set plans[B_t]

1. candidate scoring:
    P_base = argmin_P score(S_runtime -> P)

2. diff construction:
    D = Diff(S_runtime, P_base)

3. graph expansion:
    initialize frontier with coarse diff supernodes

4. decision loop:
    while frontier not empty and online_time_budget not exhausted:
        node = pop(frontier)
        compute node score

        if accept(node):
            apply node diff to adjusted plan
        elif reject(node):
            keep current placement for this subgraph
        else:
            expand node and push children

5. feasibility repair:
    if adjusted plan exceeds memory budget:
        accept demotion/eviction diffs by lowest future loss per byte

output:
    adjusted plan P_adjusted
```

## Expected Baselines

The evaluation should compare:

```text
offline-single
    One precomputed plan per budget.  Does not consider current residency.

online-cpsat
    Full online solve.  Best quality, high solve overhead.

candidate-select
    Top-K precomputed plans, runtime transition scoring only.

candidate-select + diff-graph-expand
    Top-K selection plus graph-structured incremental adaptation.
```

Expected qualitative result:

```text
decode speed:
    online-cpsat ~= diff-graph-expand > candidate-select > offline-single

online overhead:
    candidate-select < diff-graph-expand << online-cpsat
```

The target research claim is:

```text
Graph-structured diff expansion preserves most of the benefit of residency-aware
online planning while reducing solve overhead from CP-SAT scale to lightweight
candidate scoring plus bounded graph refinement.
```

## Minimal Implementation Plan

### Phase 1: Candidate Table

```text
Extend offline table builder:
    output top-K candidate plans per budget
    store placement bitsets and steady cost
```

### Phase 2: Candidate Selection

```text
Add online mode:
    LLAMA_ELASTIC_ONLINE_MODE=candidate-select

Runtime:
    read current residency
    score K candidates
    apply best candidate
```

### Phase 3: Diff Graph Expansion

```text
Add online mode:
    LLAMA_ELASTIC_ONLINE_MODE=diff-graph-expand

Runtime:
    build structured diff graph
    run expand/accept/reject loop
    emit adjusted plan
```

### Phase 4: Instrumentation

Log per budget switch:

```text
selected candidate id
candidate steady cost
candidate transition cost
accepted diff nodes
rejected diff nodes
expanded node count
online decision time
load/prepare/evict bytes caused by accepted diffs
```

This instrumentation is essential for showing that the method is not just
faster, but is making interpretable residency-aware decisions.

## Implementation Status

Initial implementation:

```text
commit base:
    705d8f014 elastic: save pipeline runtime and incremental diff design

offline builder:
    runtime/plan/build_offline_budget_table.py

new options:
    --top-k
    --candidate-placement-specs

index format:
    index.json keeps the legacy `file` field and adds `candidates`.
```

Runtime support:

```text
LLAMA_ELASTIC_ONLINE_MODE=candidate-select
LLAMA_ELASTIC_CANDIDATE_DIR=<plan-table-dir>
```

The first runtime selector:

```text
1. reads candidates for the current budget bucket
2. evaluates transition_cost(S_runtime -> P_candidate)
3. scores steady_cost + transition_weight * transition_cost
4. writes selected plan into the normal online work directory
5. attaches `online_selection` metadata to the generated plan
```

The `diff-graph-expand` mode is wired to the same selector as a placeholder for
the next stage.

Second implementation:

```text
LLAMA_ELASTIC_ONLINE_MODE=diff-graph-expand
```

The first graph expansion pass is intentionally conservative:

```text
1. Build layer/component diff groups from Diff(S_runtime, P_candidate).
2. Consider only safe promotion diffs:
       current=disk -> target=cpu/gpu
   Rejecting these diffs can only reduce memory pressure.
3. Compute group score:
       steady_gain(horizon) - transition_weight * transition_cost
4. Reject low-score groups by keeping those weights on disk and recomputing a
   simple load/xform timeline.
```

This is not the full graph method yet.  It is the first executable expand /
accept / reject loop and provides the runtime metadata needed to analyze
whether graph-level decisions are useful.

Smoke artifact:

```text
.wiki/elastic_memory/incremental_plan_diff/artifacts/candidate_select_smoke_10s_v2
.wiki/elastic_memory/incremental_plan_diff/artifacts/diff_graph_expand_smoke_10s
```

Smoke result:

```text
method             status   rc   generated tokens   raw ms/token
candidate-select   ok       0    7                  1662.94
```

The log confirms runtime candidate selection without remote CP-SAT:

```text
[elastic-candidate] budget=4096 table_budget=4096 candidate=0
score=516.315 transition=89.562 changed=100
load=112.5MB prepare=112.5MB evict=828.0MB
```

Diff graph smoke:

```text
method: diff-graph-expand
rc: 0
raw ms/token: 1561.18
provider_get_ms_total: 160.058
```
