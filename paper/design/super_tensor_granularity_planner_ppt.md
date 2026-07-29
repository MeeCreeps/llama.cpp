# Super-Tensor Granularity Planner — PPT Version

这份文档只使用普通文本和 Unicode 符号，可以直接复制到 PowerPoint。
详细推导仍保留在 `super_tensor_granularity_planner.md`，不需要放入主讲 PPT。

---

## Slide 1 — Why Granularity Must Be Planned

### Title

**Dynamic Memory Budgets Change the Best Working-Unit Granularity**

### Left: three granularity choices

**Multi-Tensor**

- Groups adjacent tensors into one pipeline unit
- Fewer I/O requests and backend submissions
- Better overhead amortization
- Larger unit may delay the whole pipeline

**Tensor**

- One tensor per pipeline unit
- Moderate request and scheduling overhead
- Limited flexibility for overlap

**Cut**

- Splits one tensor into independently prepared row ranges
- Smaller units become ready earlier
- Better layout locality and pipeline overlap
- More requests and bookkeeping

### Fair backend implementation

```text
GPU Multi / Tensor / Cut
= same two-tile physical layout
+ same fused dual-half GEMV
+ different logical pipeline boundaries
```

This isolates working-unit granularity from an accidental compute-kernel
quality difference.

For GPU Cut, both tiles are prepared independently but the fused GEMV starts
only after both are ready: **2× LOAD/PREPARE → 1× fused Compute**.

### Right: core observation

**Low weight residency**

I/O and layout preparation are exposed on the critical path.  
Fine-grained Cut units can reduce pipeline waiting.

**High weight residency**

Most weights are already available.  
Coarse-grained Multi units can reduce per-unit overhead.

### Bottom takeaway

> No single granularity is optimal for every memory budget, tensor, operator,
> and backend.

### Speaker note

Granularity does not only change the isolated I/O, layout, or compute cost.
It changes when each unit becomes ready and therefore changes pipeline overlap.

---

## Slide 2 — A Unified Super-Tensor Abstraction

### Title

**Super-Tensor Tree Unifies Multi, Tensor, and Cut**

### Main figure

```text
                         Multi(A, B)
                         /         \
                   Tensor(A)     Tensor(B)
                    /    \        /    \
                  Cut A0 Cut A1 Cut B0 Cut B1
```

### Labels beside the tree

- Select the parent → use **Multi**
- Select a tensor node → use **Tensor**
- Select its children → use **Cut**
- Select different levels in different subtrees → obtain a **mixed plan**

### Validity rule

```text
Every physical weight range must be covered exactly once.
```

This prevents:

- missing weight ranges;
- duplicate computation;
- selecting a Tensor together with its Cut children.

### What the tree does

```text
Tree = legal decision space
```

The tree does not predict performance. It only defines which split/merge
choices are legal.

### Bottom takeaway

> We turn three separate execution modes into one hierarchical partitioning
> problem.

---

## Slide 3 — Budget-Aware Granularity Planning

### Title

**Select the Mixed Plan with the Shortest Pipeline Completion Time**

### Input boxes

```text
Memory budget B(t)
Current weight residency
Tensor/operator/backend features
Recent pipeline history
```

All four boxes point to:

```text
Super-Tensor Planner
```

### Planner output

```text
Plan at time t
= mixed working units
+ weight residency
+ pipeline schedule
```

### Main objective — plain-text version

```text
Predicted total cost
= predicted pipeline completion time
+ intra-plan fragmentation cost
+ λ × plan-switch cost
```

```text
Fragmentation cost coalesces profitable Cut choices into long contiguous runs;
it does not globally suppress Cut. Its calibrated coefficient may be zero when
coalescing would remove still-profitable Cut choices.
```

```text
Choose the feasible plan with the lowest predicted total cost.
```

### One-line equation for a PowerPoint equation box

The following is PowerPoint linear input, not LaTeX:

```text
BestPlan(t) = arg min [ PipelineCompletionTime(t)
                        + FragmentationCost(t)
                        + λ·SwitchCost(t) ]
```

### Two different overheads

```text
Fragmentation cost (paid every token)
= αcut × #Cut↔Whole boundaries
+ αmerge × #Tensor↔Multi boundaries
```

```text
Switch cost (paid when the plan changes)
= split/merge edits + layout regroup + pipeline drain + residency changes
```

Cut↔Whole and Tensor↔Multi boundaries are calibrated separately because only
the former changes the physical row-unit lifecycle. This avoids penalizing the
legal Tensor fallbacks inside a Multi plan as if they were Cut transitions.

Implementation note for the slide:

```text
Granularity switch coefficient
= measured changed-frontier publication on held-in boundary nodes
```

The one-time initial frontier installation is excluded. The complete
placement/residency apply remains on the measured end-to-end critical path
and is not charged again as a granularity switch.

### Memory constraint — plain-text version

```text
Resident weights
+ pinned execution units
+ in-flight staging buffers
≤ current memory budget
```

### Why we predict pipeline completion instead of summing stages

```text
I/O time + Layout time + Compute time ≠ End-to-end latency
```

I/O, layout preparation, and compute overlap. The planner must predict the
finish time of the resource-constrained pipeline, not add three independently
measured stage times.

### Suggested pipeline inset

```text
Time →

I/O:       [ unit 1 ][ unit 2 ][ unit 3 ]
Layout:             [ unit 1 ][ unit 2 ][ unit 3 ]
Compute:                      [ unit 1 ][ unit 2 ][ unit 3 ]

End-to-end latency = finish time of the last required Compute unit
```

### Bottom takeaway

> The cost model evaluates pipeline behavior; the optimizer selects the
> feasible mixed plan.

---

## Slide 4 — How the Planner Predicts a Candidate

### Title

**Estimate Stage Demand, Then Simulate Pipeline Overlap**

### Per-unit information

Each candidate working unit stores:

```text
Weight bytes
Resident or non-resident
Operator type
Backend
Granularity
Estimated I/O demand
Estimated layout-preparation demand
Estimated compute demand
Per-unit launch and bookkeeping overhead
```

### Residency is also a working-set decision

```text
Current residency
= protected keep set
+ rolling stream set
```

Both sets share the same hard memory budget. Retaining a prepared stream unit
must replace another movable unit; it never creates extra capacity.

```text
Persistent keep bytes
≤ B(t) − KV − misc − pinned − safety − streaming-window reserve
```

The streaming reserve is the maximum current/lookahead window implied by the
selected frontier. It changes the partition of the same budget; it does not
increase the physical budget.

Do not cold-flush every nonresident unit at a graph boundary. Doing so can fill
the budget with protected weights and then evict/reload them to admit the first
unit of the next token. The cost model tracks this failure mode as:

```text
excess direct-read bytes
+ plan-protection-relaxed bytes
+ exposed pipeline wait
```

### Simple cost expressions

```text
I/O demand
= non-resident bytes / effective I/O bandwidth
+ request overhead
```

```text
Layout demand
= non-resident bytes / effective preparation bandwidth
+ preparation overhead
```

```text
Compute demand
= operator work / effective throughput
+ kernel-submission overhead
```

### Pipeline recurrence in words

For every unit:

1. I/O starts after the I/O resource becomes free.
2. Layout starts after its I/O finishes and the layout resource becomes free.
3. Compute starts after its layout finishes, its dependencies finish, and the
   backend compute queue becomes free.
4. The final completion time is the predicted pipeline latency.

### Recent-history correction

```text
Recent wait ratio
= observed pipeline wait
/ (wait + layout preparation + compute)
```

Use a bounded moving average to correct offline estimates without reacting to
single-token noise.

Important modeling rule:

```text
Pipeline residual corrects exposed non-resident I/O and layout work.
It must not rescale resident kernel compute.
```

Otherwise the planner can erase Multi's measured high-budget compute advantage
and over-split a mostly resident frontier.

### Bottom takeaway

> The same bytes can produce different latency because their unit boundaries
> create different pipeline schedules.

---

## Slide 5 — Offline Plan and Online Diff-Tree

### Title

**Offline Search Finds a Strong Plan; Online Diff-Tree Adapts It**

### Left: offline

```text
For each memory-budget bucket:

1. Generate legal Multi / Tensor / Cut candidates
2. Predict their pipeline completion time
3. Search for a good mixed partition
4. Store the selected plan
```

Search actions:

```text
Tensor(Wi)
Cut(Wi)
Multi(Wi, Wi+1)
```

### Right: online

Start from the current frontier and evaluate only local edits:

```text
Tensor → Cut
Cut → Tensor
Tensor + Tensor → Multi
Multi → Tensor + Tensor
```

Accept an edit only when:

```text
Expected latency reduction over the horizon
> switching cost + stability margin
```

When the budget enters a new bucket:

```text
1. Re-evaluate the current frontier under the new residency
2. Compare it with the new bucket's offline target
3. Switch only if the horizon net gain is positive
```

The bucket selects a candidate target; it does not force a granularity change.
This avoids unnecessary split/merge when placement changes but both frontiers
have the same predicted critical path.

### What changes at runtime

At a graph boundary, publish one atomic frontier delta:

- compare old/new slices by weight;
- replace only changed weight entries;
- keep unchanged buffers and pool objects;
- increment generation once;
- log changed-weight count and isolated publication time.

If placement, dispatch, and event schedule are unchanged, the runtime skips
weight reconciliation and graph invalidation. A real placement change still
uses the full safe apply path.

### Bottom takeaway

> Offline planning provides a high-quality target; online Diff-Tree makes
> bounded, profitable split/merge updates.

---

## Slide 6 — Expected Adaptation Behavior

### Title

**Budget Changes Shift the Pipeline Bottleneck**

### Budget decreases

```text
More non-resident weights
→ more exposed I/O and layout work
→ large units cause longer ready-time stalls
→ split critical units when the saved wait exceeds switch cost
```

Likely action: **Multi/Tensor → Cut**

### Budget increases

```text
More resident weights
→ less exposed I/O and layout work
→ request and dispatch overhead become relatively larger
→ merge compatible units when overhead saving exceeds switch cost
```

Likely action: **Cut/Tensor → Multi**

### Important qualification

Do not write:

```text
Low budget always uses Cut; high budget always uses Multi.
```

Write:

```text
Budget changes the bottleneck, while the final choice also depends on tensor
size, operator, backend, residency, recent pipeline behavior, and switch cost.
```

### Bottom takeaway

> Adapt granularity according to the predicted bottleneck, not a fixed budget
> threshold.

---

## Slide 7 — System Flow

### Title

**End-to-End Granularity-Aware Elastic Inference**

### Flow diagram text

```text
Dynamic memory trace
        ↓
Budget watcher
        ↓
Placement / residency plan
        ↓
Super-Tensor mixed-granularity planner
        ↓
Incremental plan diff
        ↓
Granularity-aware I/O → Layout → Compute pipeline
        ↓
Runtime stage counters
        └──────────── feedback to planner
```

### Runtime safety box

```text
Coverage correctness
Token-prefix correctness
Physical budget enforcement
Execution pin safety
Same backend and retention policy
```

### Bottom takeaway

> Granularity becomes part of the Elastic plan rather than a global runtime
> switch.

---

## Slide 8 — Evaluation Plan

### Title

**Evaluate Placement and Granularity on the Original 10-Minute Trace**

### Backends

```text
CPU Elastic-only
GPU/OpenCL Elastic-only
```

No CPU+GPU heterogeneous placement in this stage.

### Baselines

```text
Static-Min
Static-Max
MRU
Offline
Online
Diff-before
Diff-now
```

### Metrics

```text
End-to-end ms/token
I/O bytes and time
Layout preparation
Exposed pipeline wait
Working-unit / dispatch count
Plan-switch count and cost
Resident / pinned / in-flight peak
Budget violations
Token correctness
```

### Required expectation

```text
Static-Max should provide the performance upper bound because it retains the
largest weight set and does not represent a deployable low-budget policy.
```

### Main comparison

```text
Diff-now vs Diff-before:
Does the new pipeline-aware mixed-granularity design improve the full trace?
```

---

## Slide 9 — Original 10-Minute Trace Results

> **Historical preliminary table — do not cite.** These values predate the
> OP12 equal-budget, mixed-boundary and GPU compute-fit audit. Replace this
> slide from
> `exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_full10min_final_op12_20260729_v19`
> after both serialized CPU/GPU runs pass token, budget, thermal and isolation
> validation.

### Title

**Mixed Granularity Improves the GPU Dynamic Plan**

### CPU Elastic-only

```text
Method         ms/token
Static-Max       296.69
MRU              427.59
Offline          542.12
Online           575.26
Diff-before      638.51
Diff-now         575.30
Static-Min      1676.29
```

CPU Diff-now is 9.9% faster than Diff-before, but is approximately tied with
Online.

### GPU/OpenCL Elastic-only

```text
Method         ms/token
Static-Max       388.74
Diff-now         637.17
Offline          760.42
Diff-before      788.94
Online           804.25
MRU              843.61
Static-Min      1500.06
```

GPU Diff-now improves over:

```text
Diff-before: 19.2%
Offline:     16.2%
Online:      20.8%
MRU:         24.5%
```

### Mechanism

```text
High budget:
147 coarse units preserve Multi-style overhead amortization

Lower budget:
about 188–207 units expose more overlap opportunities
```

### Validation

```text
Original 600-second absolute-budget trace
No time compression or budget remapping
Same 32-token prefix
Continuous device isolation
Zero physical GPU budget violations
Static-Max remains the performance upper bound
```

### Bottom takeaway

> Granularity should adapt with the pipeline bottleneck, but the residual model
> must not sacrifice resident compute efficiency.

---

## Backup Slide — Compact Design Formula Without LaTeX

Use the following block directly:

```text
Plan(t) = {MixedUnits(t), Residency(t), PipelineSchedule(t)}

BestPlan(t)
= the feasible Plan(t) that minimizes:

    Predicted pipeline completion time
  + λ × split/merge and plan-transition cost

subject to:

    Resident weights
  + pinned execution units
  + in-flight staging buffers
  ≤ memory budget B(t)
```

## One-Sentence Design Summary

> The Super-Tensor tree defines legal mixed-granularity choices, the cost
> model predicts their pipeline completion times, and Diff-Tree applies only
> profitable local split/merge updates under the current memory budget.
