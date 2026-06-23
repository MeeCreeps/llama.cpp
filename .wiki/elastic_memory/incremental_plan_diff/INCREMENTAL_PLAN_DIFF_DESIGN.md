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

The runtime selector:

```text
1. loads and caches the candidate table on first use
2. evaluates transition_cost(S_runtime -> P_candidate)
3. scores steady_cost + transition_weight * transition_cost
4. returns a cached `llama_plan*` directly for unmodified candidate-select
```

The selector keeps these runtime caches:

```text
index.json cache
candidate JSON plan cache
candidate `llama_plan*` handle cache
runtime cost lookup cache
```

The cost cache is important.  Without it, each budget change repeatedly scans
JSON profile records for every candidate weight.  With it, the first call fills
the cache and later candidate scoring is mostly O(K * N_weights) arithmetic and
state-bit checks.

Top-K generation now supports distance-based candidate diversity:

```text
dynamic_budget_solver.py:
    --exclude-plan <plan.json>
    --min-placement-distance <N>

build_offline_budget_table.py:
    --candidate-min-distance 32
```

For candidate `k > 0`, the offline builder excludes all previous candidates
under the same budget by requiring a placement Hamming distance of at least
`candidate_min_distance`.  This keeps all candidates in the same allowed
placement space and avoids the older behavior where diversity was created by
forbidding placement classes such as `disk_cpu` or `disk_gpu`.

Local top-K smoke with `candidate_min_distance=32`:

```text
budget 4096:
    pairwise placement distances: all 32
    pred_per_token_ms: 542.395, 542.450, 542.423, 542.624

budget 4352:
    pairwise placement distances: all 32
    pred_per_token_ms: 401.323, 401.502, 401.858, 401.697

budget 4608:
    pairwise placement distances: all 32
    pred_per_token_ms: 259.896, 261.043, 261.401, 261.419
```

This is the desired candidate shape: meaningfully different placements with
small steady-cost gaps, instead of candidates that are diverse only because
some placement classes were disabled.

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

Current optimized smoke:

```text
artifact:
    .wiki/elastic_memory/incremental_plan_diff/artifacts/cost_cache_smoke_20s_candidate

trace:
    trace_05_user_74_10min_x1.csv
    source window: 20 s
    budget bucket: 256 MiB
    min/mean/max budget: 4301.7 / 4309.3 / 4320.6 MiB

candidate-select:
    raw ms/token: 220.79
    provider_get_ms_total: 178.80
    online calls: 10
    failures: 0
```

Per-call candidate-select overhead after cache warmup:

```text
first call:
    91.965 ms

later calls:
    24.860 ms
    17.175 ms
    15.291 ms
    16.862 ms
    4.790 ms
    4.601 ms
    9.807 ms
    21.525 ms
    10.684 ms
```

Diff graph optimized smoke:

```text
artifact:
    .wiki/elastic_memory/incremental_plan_diff/artifacts/cost_cache_smoke_20s_diff

diff-graph-expand:
    raw ms/token: 183.42
    provider_get_ms_total: 339.47
    online calls: 11
    failures: 0
```

Per-call diff-graph overhead after cache warmup:

```text
first call:
    165.964 ms

later calls:
    23.760 ms
    14.982 ms
    10.954 ms
    10.073 ms
    11.296 ms
    8.731 ms
    19.106 ms
    36.545 ms
    13.863 ms
    16.364 ms
```

This confirms that the online incremental path no longer depends on remote
CP-SAT solve time.  The remaining large runtime cost is plan application and
actual load/prepare/evict work, not online decision overhead.

Distance-topK score-log smoke:

```text
artifact:
    .wiki/elastic_memory/incremental_plan_diff/artifacts/distance32_scores_20s_candidate

env:
    LLAMA_ELASTIC_CANDIDATE_LOG_SCORES=1

candidate-select:
    provider_get_ms_total: 213.39
    online calls: 1
```

The first budget switch selected candidate 0:

```text
budget=4096
candidate 0: steady=542.395 transition=114.290 score=553.824
candidate 1: steady=542.450 transition=114.290 score=553.879
candidate 2: steady=542.423 transition=114.290 score=553.852
candidate 3: steady=542.624 transition=120.665 score=554.690
```

Interpretation:

```text
On the first apply, runtime residency is almost equally far from all candidates,
so the transition term cannot distinguish the diverse placements.  Candidate 0
wins because it has the best steady cost.
```

To show the incremental advantage clearly, the evaluation trace should include
multiple budget changes after a previous plan has already been materialized.
The useful regime is:

```text
P_prev is already partially resident/transformed
budget changes to B_t
candidate j reuses more of S_runtime than offline P0
candidate j wins despite a slightly higher steady cost
```

Oscillating trace result:

```text
artifact:
    .wiki/elastic_memory/incremental_plan_diff/artifacts/oscillating_distance32_scores_120s_candidate

trace:
    trace_06_user_204_10min_x1.csv
    source window: 180 s
    bench seconds: 120 s
    min/mean/max budget: 4008.2 / 4615.7 / 5705.9 MiB
    buckets: 3840..5632 MiB

candidate-select, transition_weight=0.1:
    online calls: 26
    selected candidates: candidate 0 = 25, candidate 3 = 1
    provider_get_ms_total: 367.63
    apply_count: 24
    planned load/xform: 366 / 366
    direct_read: 13999.08 ms, 415 calls, 11191.5 MiB
```

The first switch selected a non-zero candidate because it had much lower
transition cost:

```text
budget=4608
candidate 0 score=271.946 transition=120.495 changed=77
candidate 1 score=270.599 transition=95.558  changed=63
candidate 2 score=270.245 transition=88.433  changed=59
candidate 3 score=269.803 transition=83.839  changed=59
selected: candidate 3
```

This validates the core mechanism: the online selector can choose a slightly
worse steady plan when it better matches current residency.

Transition-weight sweep smoke:

```text
artifact:
    .wiki/elastic_memory/incremental_plan_diff/artifacts/oscillating_distance32_tw1_120s_candidate

candidate-select, transition_weight=1.0:
    online calls: 27
    selected candidates: candidate 0 = 21, candidate 1 = 3, candidate 2 = 1, candidate 3 = 2
    provider_get_ms_total: 358.58
    apply_count: 26
    planned load/xform: 424 / 424
    direct_read: 14056.00 ms, 473 calls, 10737.0 MiB
```

Interpretation:

```text
Increasing transition_weight makes the selector choose non-zero candidates
more often, but it is not automatically faster.  A too-large transition weight
can prefer plans with lower immediate transition score but more future load
events under oscillating budgets.
```

The next evaluation should sweep:

```text
transition_weight in {0.1, 0.25, 0.5, 1.0}
candidate_min_distance in {16, 32, 48}
```

and compare:

```text
candidate id distribution
planned load/xform/evict
direct read ms and MiB
apply_ms_total
raw and exec ms/token under cooled runs
```

Initial alpha sweep on the same distance-32 table:

```text
artifacts:
    .wiki/elastic_memory/incremental_plan_diff/artifacts/oscillating_distance32_scores_120s_candidate
    .wiki/elastic_memory/incremental_plan_diff/artifacts/oscillating_distance32_tw025_120s_candidate
    .wiki/elastic_memory/incremental_plan_diff/artifacts/oscillating_distance32_tw05_120s_candidate
    .wiki/elastic_memory/incremental_plan_diff/artifacts/oscillating_distance32_tw1_120s_candidate

summary generated by:
    runtime/plan/summarize_incremental_candidates.py
```

```text
transition_weight  raw ms/tok  provider ms  apply count  load/xform  direct read ms/calls/MB       selected candidates
0.10               545.27      367.63       24           366/366     13999.08 / 415 / 11191.5   c0=25, c3=1
0.25               397.97      331.40       22           332/332     13575.71 / 381 / 10179.0   c0=25, c3=1
0.50               335.72      283.00       25           378/378      8837.18 / 427 / 11569.5   c0=28, c3=1
1.00               462.98      358.58       26           424/424     14056.00 / 473 / 10737.0   c0=21, c1=3, c2=1, c3=2
```

Same-trace offline reference with the same pushed distance-32 table:

```text
offline:
    raw ms/token: 346.74
    provider_get_ms_total: 45.44
    apply_count: 25
    load/xform: 378 / 378
    direct_read: 15436.33 ms, 427 calls, 11569.5 MiB
```

Preliminary interpretation:

```text
transition_weight=0.5 is the first run where candidate-select is slightly
faster than offline on raw ms/token and much lower on direct read time.
However, the runs were not thermally controlled:
    offline CPU max: 41.1 -> 58.5 C
    tw=0.5 CPU max: 50.0 -> 71.3 C

So this is promising but not yet a rigorous speed result.  It should be
repeated with cooldown or randomized run order.
```

The sweep also shows why alpha tuning matters:

```text
alpha too low:
    selector mostly chooses P0, so it behaves like offline with small runtime
    scoring overhead.

alpha too high:
    selector chooses more non-P0 candidates, but may increase future load/xform
    under budget oscillation.

alpha around 0.25-0.5:
    current trace suggests the best trade-off between reuse and steady plan
    quality.
```

## Sticky current-plan branch

The first candidate-select implementation still made a fresh table choice at
every budget tick.  That is useful for validating the transition cost, but it
can overreact on traces where the budget oscillates upward after a low-budget
event.  In that case the current resident/transformed plan is already feasible
under the larger budget, and switching to the larger budget's steady-optimal
candidate may pay load/transform cost that is not recovered before the next
budget change.

This motivates a conservative branch inside step 4:

```text
Given:
    P_keep = currently applied candidate plan
    P_best = best candidate under the current budget table

If:
    budget(P_keep) <= current_budget
    steady_cost(P_keep) <= score(P_best) + keep_margin_ms

Then:
    reject the candidate diff and keep P_keep.
```

Runtime knob:

```text
LLAMA_ELASTIC_KEEP_CURRENT_MARGIN_MS=<margin>
```

This is a simple graph decision:

```text
root: compare current plan and candidate table winner
edge accept:
    apply diff(P_keep -> P_best)
edge reject:
    keep current residency and skip plan apply
```

The branch is deliberately one-sided:

```text
It only keeps a previous plan when that plan was built for a budget not larger
than the current budget.  It does not keep an infeasible high-budget plan after
a budget drop.
```

Smoke result on the oscillating trace:

```text
artifact:
    .wiki/elastic_memory/incremental_plan_diff/artifacts/oscillating_distance32_tw05_keep150_60s_candidate

trace:
    trace_06_user_204_10min_x1.csv

settings:
    candidate_min_distance=32
    transition_weight=0.5
    keep_current_margin_ms=150
    bench_seconds=60
```

```text
raw ms/token:            223.74
exec ms/token:           227.48
provider_get_ms_total:   146.03
apply_count:             2
planned load/xform:      34 / 34
direct_read:             698.36 ms, 34 calls, 1012.5 MiB
online calls:            156
selected candidates:     c-2=154, c0=1, c3=1
real switches:           1
```

Here `candidate=-2` means "keep current plan".  The online provider still
receives many budget checks, but almost all of them are resolved by rejecting
the diff and keeping the already materialized plan.  This is the first result
where the mechanism strongly reduces load/transform instead of merely choosing
among alternate steady plans.

Current caveat:

```text
This is only a 60-second smoke run and not a cooled A/B comparison.  The next
step is to sweep keep_margin_ms in {25, 50, 100, 150} and compare against
offline on the same trace with cooldown/randomized order.
```

## Provider-cache refinement

The sticky branch exposed a second overhead source.  Even when the selected
plan remains unchanged, the runtime used to call the online provider every
token whenever:

```text
raw/effective budget B != applied_plan.budget_mib
```

This happens naturally with sticky keep:

```text
current raw budget: 4608 / 4864 / ...
applied plan:       4352
decision:           keep 4352 because transition is not worth it
```

The old context-level cache only skipped provider calls when `B` matched the
applied plan's own budget.  The refinement adds a provider-result cache:

```text
if provider(B) previously returned the currently applied plan:
    skip provider(B) on the next token with the same B
```

This keeps the online decision exact at the first occurrence of a budget, but
avoids repeatedly proving the same keep decision while the budget remains
unchanged.

Additional runtime knob:

```text
LLAMA_ELASTIC_CANDIDATE_LOG_EVENTS=0
```

This disables per-budget candidate event logs for production-speed runs.  The
default remains enabled for debugging and summary scripts.

Short A/B on the same 60-second oscillating window:

```text
trace:
    trace_06_user_204_10min_x1.csv

offline:
    raw ms/token:          224.79
    exec ms/token:         243.08
    provider_get_ms_total: 36.00
    apply_count:           15
    planned load/xform:    208 / 208
    direct_read:           4315.24 ms, 257 calls, 6507.0 MiB

candidate-select:
    transition_weight:     0.5
    keep_current_margin:   50 ms
    provider result cache: on
    candidate event logs:  off
    raw ms/token:          224.08
    exec ms/token:         228.61
    provider_get_ms_total: 143.96
    online calls:          13
    apply_count:           2
    planned load/xform:    34 / 34
    direct_read:           961.38 ms, 34 calls, 1012.5 MiB
```

Interpretation:

```text
The online plan is now slightly faster on raw ms/token in this short run, and
clearly faster on execution-side ms/token.  It also reduces materialization
work by roughly 6x and direct-read volume by roughly 6.4x.

The remaining provider time is dominated by the first few calls, which still
load/cache candidate plan data lazily.  A future improvement is to prewarm the
candidate table before timed decode starts.
```

## Candidate prewarm

The provider-cache run reduced online calls from about 150 to 13, but the
remaining provider time was still high because the first calls lazily loaded:

```text
candidate index/json plans
candidate llama_plan handles
```

The runtime now supports:

```text
LLAMA_ELASTIC_CANDIDATE_PREWARM=1
```

When enabled for `candidate-select` or `diff-graph-expand`, initialization
loads the candidate table and all candidate plan handles before timed decode.

Prewarm result on the same 60-second oscillating window:

```text
candidate-select:
    transition_weight:       0.5
    keep_current_margin:     50 ms
    provider result cache:   on
    candidate event logs:    off
    candidate prewarm:       on

    raw ms/token:            224.12
    exec ms/token:           229.54
    provider_get_ms_total:   36.67
    online calls:            13
    apply_count:             2
    planned load/xform:      34 / 34
    direct_read:             989.35 ms, 34 calls, 1012.5 MiB
```

The log confirmed:

```text
cached 36 plans
prewarmed 36 plan handles
```

Compared with no-prewarm:

```text
provider_get_ms_total:
    no prewarm: 143.96 ms
    prewarm:     36.67 ms
```

This moves the online provider overhead back to the same scale as offline while
preserving the lower movement count:

```text
offline load/xform:     208 / 208
candidate load/xform:    34 / 34
```

## Cooled 60-second A/B

Artifact:

```text
.wiki/elastic_memory/incremental_plan_diff/artifacts/oscillating_distance32_keep50_prewarm_cooled_ab_60s
```

Settings:

```text
trace:                   trace_06_user_204_10min_x1.csv
source window:           180 s
bench seconds:           60 s
cooldown target:         42 C
candidate_min_distance:  32
transition_weight:       0.5
keep_current_margin:     50 ms
candidate prewarm:       on
candidate event logs:    off
```

Results:

```text
offline:
    raw ms/token:          255.88
    exec ms/token:         283.10
    provider_get_ms_total: 33.90
    apply_count:           15
    planned load/xform:    208 / 208
    direct_read:           5593.18 ms, 257 calls, 6507.0 MiB
    thermal CPU max:       38.7 -> 56.6 C

candidate-select:
    raw ms/token:          209.85
    exec ms/token:         214.33
    provider_get_ms_total: 26.83
    online calls:          13
    apply_count:           2
    planned load/xform:    34 / 34
    direct_read:           890.63 ms, 34 calls, 1012.5 MiB
    thermal CPU max:       41.3 -> 60.5 C
```

Interpretation:

```text
This is the clearest current evidence for the research claim:

    offline switches to the budget-optimal target plan and repeatedly pays
    movement/materialization.

    online candidate-select uses the previous/current resident plan as state,
    accepts only high-value diffs, and rejects most budget-up oscillation diffs.

In this window, online reduces load/xform by 6.1x and direct-read volume by
6.4x, while raw decode improves by about 18%.
```

## Cooled 120-second order-swap A/B

Artifacts:

```text
.wiki/elastic_memory/incremental_plan_diff/artifacts/oscillating_distance32_keep50_prewarm_cooled_offline_candidate_120s
.wiki/elastic_memory/incremental_plan_diff/artifacts/oscillating_distance32_keep50_prewarm_cooled_candidate_offline_120s
```

Settings:

```text
trace:                   trace_06_user_204_10min_x1.csv
source window:           180 s
bench seconds:           120 s
cooldown target:         42 C
candidate_min_distance:  32
transition_weight:       0.5
keep_current_margin:     50 ms
candidate prewarm:       on
candidate event logs:    off
```

Order 1: offline -> candidate-select

```text
offline:
    raw ms/token:          568.36
    exec ms/token:         626.44
    provider_get_ms_total: 40.53
    apply_count:           21
    planned load/xform:    310 / 310
    direct_read:           10992.30 ms, 359 calls, 9544.5 MiB
    thermal CPU max:       37.2 -> 55.4 C

candidate-select:
    raw ms/token:          233.12
    exec ms/token:         235.86
    provider_get_ms_total: 37.14
    online calls:          13
    apply_count:           2
    planned load/xform:    34 / 34
    direct_read:           1032.12 ms, 34 calls, 1012.5 MiB
    thermal CPU max:       38.7 -> 53.9 C
```

Order 2: candidate-select -> offline

```text
candidate-select:
    raw ms/token:          232.63
    exec ms/token:         235.02
    provider_get_ms_total: 38.30
    online calls:          13
    apply_count:           2
    planned load/xform:    34 / 34
    direct_read:           862.02 ms, 34 calls, 1012.5 MiB
    thermal CPU max:       41.5 -> 54.3 C

offline:
    raw ms/token:          592.65
    exec ms/token:         636.64
    provider_get_ms_total: 37.55
    apply_count:           17
    planned load/xform:    257 / 257
    direct_read:           8156.99 ms, 306 calls, 8030.2 MiB
    thermal CPU max:       41.5 -> 64.3 C
```

Interpretation:

```text
The order-swap result is consistent:

    candidate-select stays around 233 ms/token in both orders.
    offline stays around 568-593 ms/token in both orders.

The speedup is not a run-order artifact.  The main difference is movement:
candidate-select applies two plans and keeps the low-budget/current-resident
plan through budget-up oscillations; offline repeatedly switches to the
budget-optimal target plan and pays hundreds of load/xform events.
```

## Keep-current margin sweep

`keep_current_margin_ms` controls how much steady-state loss the incremental
online planner is willing to tolerate before switching away from the current
resident plan.

Decision rule:

```text
P_keep = currently applied plan
P_best = best scored candidate under current budget

if budget(P_keep) <= current_budget
   and steady_cost(P_keep) <= score(P_best) + keep_current_margin_ms:
       keep P_keep and reject the diff
else:
       accept the selected candidate diff
```

Interpretation:

```text
margin = 0:
    no sticky keep-current branch; behaves like candidate-select without
    incremental diff rejection.

small/moderate margin:
    allows the runtime to keep an already resident/transformed plan when the
    predicted steady loss is smaller than the movement avoided.

too large margin:
    may keep a feasible but overly conservative low-budget plan too long.
```

Sweep settings:

```text
trace:                   trace_06_user_204_10min_x1.csv
source window:           180 s
bench seconds:           60 s
cooldown target:         42 C
candidate_min_distance:  32
transition_weight:       0.5
candidate prewarm:       on
candidate event logs:    off
```

Results:

```text
margin  raw ms/tok  exec ms/tok  provider ms  apply  load/xform  direct read ms/calls/MB
0       290.97      322.87       132.73       15     208/208     6041.59 / 257 / 6507.0
25      210.63      214.63        29.06        2      34/34       811.54 / 34  / 1012.5
50      207.03      210.76        26.06        2      34/34       765.13 / 34  / 1012.5
100     231.60      236.78        37.55        2      34/34       975.53 / 34  / 1012.5
```

This supports the intended trade-off:

```text
margin=0 is too reactive:
    it follows the candidate table and repeatedly pays movement.

margin=25/50 is the sweet spot on this window:
    it rejects low-value budget-up diffs and avoids most reload/transform.

margin=100 still avoids movement:
    but it starts to lose speed, consistent with keeping a lower-budget plan
    longer than necessary.
```

## Cross-trace 60-second A/B

Artifact:

```text
.wiki/elastic_memory/incremental_plan_diff/artifacts/cross_trace_9g_keep50_prewarm_60s
```

Settings:

```text
traces:                  trace_01_user_147, trace_05_user_74, trace_06_user_204
source window:           runner-selected lowest 180 s window
bench seconds:           60 s
cooldown target:         42 C
candidate_min_distance:  32
transition_weight:       0.5
keep_current_margin:     50 ms
candidate prewarm:       on
candidate event logs:    off
```

Results:

```text
trace_01_user_147:
    window min/mean/max:      3752.1 / 4451.9 / 4809.4 MiB

    offline:
        raw ms/token:          381.71
        exec ms/token:         442.74
        apply_count:           8
        planned load/xform:    187 / 187
        direct_read:           7583.73 ms, 187 calls, 5602.5 MiB

    candidate-select:
        raw ms/token:          257.27
        exec ms/token:         265.60
        online calls:          8
        apply_count:           3
        planned load/xform:    73 / 73
        direct_read:           1353.08 ms, 73 calls, 2166.8 MiB

trace_05_user_74:
    window min/mean/max:      4695.1 / 5174.6 / 6487.4 MiB

    offline:
        raw ms/token:          200.31
        exec ms/token:         203.29
        apply_count:           8
        planned load/xform:    52 / 52
        direct_read:           1017.08 ms, 150 calls, 2133.0 MiB

    candidate-select:
        raw ms/token:          175.52
        exec ms/token:         177.28
        online calls:          4
        apply_count:           2
        planned load/xform:    19 / 19
        direct_read:           417.69 ms, 19 calls, 501.8 MiB

trace_06_user_204:
    window min/mean/max:      4781.0 / 5171.9 / 6427.0 MiB

    offline:
        raw ms/token:          198.62
        exec ms/token:         200.16
        apply_count:           3
        planned load/xform:    19 / 19
        direct_read:           458.53 ms, 70 calls, 960.8 MiB

    candidate-select:
        raw ms/token:          197.66
        exec ms/token:         198.83
        online calls:          3
        apply_count:           2
        planned load/xform:    12 / 12
        direct_read:           402.16 ms, 63 calls, 837.0 MiB
```

Interpretation:

```text
The benefit scales with how much unnecessary movement offline performs:

    trace_01:
        large memory pressure and many budget changes.  Online avoids most
        movement and substantially improves speed.

    trace_05:
        moderate pressure.  Online still reduces movement and improves speed.

    trace_06:
        high-budget window with little movement.  Online mostly matches offline
        speed while still reducing apply/load/xform counts.

This supports the broader claim that incremental planning is most valuable
when budget-optimal offline switching creates redundant materialization.
```

## Final 60-second baseline matrix

Artifacts:

```text
.wiki/elastic_memory/incremental_plan_diff/artifacts/final_cross_trace_mixed_baselines_60s
.wiki/elastic_memory/incremental_plan_diff/artifacts/final_cross_trace_mru_cpu_60s
.wiki/elastic_memory/incremental_plan_diff/artifacts/final_oscillating_mixed_baselines_60s
.wiki/elastic_memory/incremental_plan_diff/artifacts/final_oscillating_mru_cpu_60s
```

Common settings:

```text
device:                  OP12, adb serial 5ae7a43d
model:                   Meta-Llama-3-8B-Instruct.Q4_0.gguf
context / batch:          -c 4096 -b 32 -ub 32
decode window:            60 s measured decode
trace replay:             180 s trace window, replay speedup 1
bucket size:              256 MiB
kv / misc / safety:       512 / 256 / 474 MiB
cooldown target:          42 C before every run
offline/candidate plans:  top-k=4, min-distance=32
candidate policy:         keep_current_margin=50 ms, prewarm on, event logs off
MRU policy:               cpu,disk_cpu only
online CP-SAT:            remote solve, solver time excluded from exec ms/token
```

The `raw ms/token` column is the end-to-end decode speed from llama.cpp timing.
The `exec ms/token` column subtracts planner provider time in the runner.  For
old online CP-SAT, this makes the speed comparable to prior reports where solve
time is not charged to decode; the solve overhead is shown separately in
`provider ms total` and `remote wall ms`.  For candidate-select, `provider
ms/call` is the online candidate decision latency.

| trace | method | raw ms/tok | exec ms/tok | tokens | provider ms total | provider ms/call | remote wall ms | calls | apply | load/xform | read MB | min/mean/max MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oscillating trace06 | static-min | 1965.8 | 1965.8 | 31 | 0.00 |  | 0.0 |  | 0 | 39/39 | 1147.5 | 4008.2/4615.7/5705.9 |
| oscillating trace06 | mru-cpu | 2143.5 | 2143.6 | 28 | 38.89 | 12.96 | 0.0 | 3 | 3 | 0/0 | 9315.0 | 4008.2/4615.7/5705.9 |
| oscillating trace06 | offline | 271.8 | 294.9 | 212 | 37.55 |  | 0.0 |  | 15 | 208/208 | 6507.0 | 4008.2/4615.7/5705.9 |
| oscillating trace06 | candidate-select | 207.7 | 211.5 | 286 | 29.72 | 2.29 | 0.0 | 13 | 2 | 34/34 | 1012.5 | 4008.2/4615.7/5705.9 |
| oscillating trace06 | online CP-SAT | 183.6 | 141.1 | 223 | 14970.36 | 787.91 | 14467.3 | 19 | 19 | 305/400 | 6581.2 | 4008.2/4615.7/5705.9 |
| trace_01_user_147 | static-min | 3705.6 | 3705.6 | 17 | 0.00 |  | 0.0 |  | 0 | 46/46 | 1404.0 | 3752.1/4451.9/4809.4 |
| trace_01_user_147 | mru-cpu | 2582.5 | 2580.7 | 24 | 81.69 | 13.62 | 0.0 | 6 | 6 | 0/0 | 15417.0 | 3752.1/4451.9/4809.4 |
| trace_01_user_147 | offline | 356.9 | 395.5 | 160 | 33.96 |  | 0.0 |  | 8 | 187/187 | 5602.5 | 3752.1/4451.9/4809.4 |
| trace_01_user_147 | candidate-select | 288.3 | 298.7 | 202 | 49.66 | 6.21 | 0.0 | 8 | 3 | 73/73 | 2166.8 | 3752.1/4451.9/4809.4 |
| trace_01_user_147 | online CP-SAT | 194.1 | 191.7 | 249 | 6807.77 | 756.42 | 6555.2 | 9 | 9 | 216/264 | 6480.0 | 3752.1/4451.9/4809.4 |
| trace_05_user_74 | static-min | 199.5 | 199.5 | 301 | 0.00 |  | 0.0 |  | 0 | 12/12 | 378.0 | 4695.1/5174.6/6487.4 |
| trace_05_user_74 | mru-cpu | 962.2 | 961.3 | 63 | 74.30 | 12.38 | 0.0 | 6 | 6 | 0/0 | 3181.5 | 4695.1/5174.6/6487.4 |
| trace_05_user_74 | offline | 193.9 | 199.0 | 302 | 26.53 |  | 0.0 |  | 8 | 52/52 | 2133.0 | 4695.1/5174.6/6487.4 |
| trace_05_user_74 | candidate-select | 173.6 | 175.1 | 343 | 23.37 | 5.84 | 0.0 | 4 | 2 | 19/19 | 501.8 | 4695.1/5174.6/6487.4 |
| trace_05_user_74 | online CP-SAT | 186.8 | 171.0 | 287 | 5951.82 | 743.98 | 5723.9 | 8 | 8 | 128/156 | 1516.5 | 4695.1/5174.6/6487.4 |
| trace_06_user_204 | static-min | 192.2 | 192.2 | 312 | 0.00 |  | 0.0 |  | 0 | 12/12 | 378.0 | 4781.0/5171.9/6427.0 |
| trace_06_user_204 | mru-cpu | 804.6 | 804.2 | 75 | 27.28 | 13.64 | 0.0 | 2 | 2 | 0/0 | 378.0 | 4781.0/5171.9/6427.0 |
| trace_06_user_204 | offline | 179.2 | 180.6 | 332 | 24.88 |  | 0.0 |  | 3 | 19/19 | 960.8 | 4781.0/5171.9/6427.0 |
| trace_06_user_204 | candidate-select | 215.2 | 216.5 | 277 | 31.29 | 10.43 | 0.0 | 3 | 2 | 12/12 | 837.0 | 4781.0/5171.9/6427.0 |
| trace_06_user_204 | online CP-SAT | 201.9 | 197.0 | 287 | 2189.52 | 729.84 | 2104.5 | 3 | 3 | 81/86 | 641.2 | 4781.0/5171.9/6427.0 |

Summary:

```text
oscillating trace06:
    candidate-select is 1.43x faster than offline by exec ms/token
    and uses only 34 load/xform operations instead of 208.  It is still
    slower than online CP-SAT's decode-only number, but candidate selection
    takes only 2.29 ms/call instead of about 788 ms/call.

trace_05_user_74:
    candidate-select is 1.14x faster than offline by exec ms/token and is
    close to online CP-SAT decode-only speed.  It is also faster than online
    in raw wall timing because no solver wall time is paid.

trace_01_user_147:
    candidate-select is 1.32x faster than offline and far faster than MRU/static.
    However, it is not close to online CP-SAT.  The current plan-level candidate
    selector reduces movement from 187 to 73 load/xform operations, while CP-SAT
    finds a better compute/movement trade-off for this window.

trace_06_user_204:
    this is a high-budget, low-movement window.  Offline and static-min are already
    strong, so candidate-select has little transition cost to remove and is slower.
```

Research interpretation:

```text
The current incremental version proves the intended fast-online mechanism:

    offline top-k candidates + runtime residency scoring
        -> millisecond-level online decisions
        -> large movement reduction when budget changes would otherwise force
           redundant materialization

It does not yet fully reproduce online CP-SAT quality on every trace.  The weak
case is trace_01, where choosing among whole precomputed candidate plans is too
coarse.  The next research step should use the graph/tree diff expansion from
Step 4 to locally accept only the valuable sub-diffs from the CP-SAT target plan:

    root: target candidate plan difference from current residency
    internal nodes: layer or layer-range placement changes
    leaves: individual weight placement changes
    accept rule: apply a sub-diff only when
        saved decode/compute cost - extra load/transform/read cost > margin
        and the resulting placement remains under the current budget

That should keep the current candidate-select latency profile while closing the
quality gap to CP-SAT on difficult traces like trace_01.
```
