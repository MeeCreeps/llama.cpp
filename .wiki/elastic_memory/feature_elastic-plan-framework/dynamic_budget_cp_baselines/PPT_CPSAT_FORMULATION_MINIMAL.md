# CP-SAT Planner Formulation

This note focuses only on the CP-SAT design: variables, constraints, and objective function. It is written in a slide-friendly style.

## Slide 1: Planner as a Constrained Optimization Problem

Goal:

> Given a dynamic memory budget, decide where each weight should reside, which backend each operator should use, and which memory movement actions are needed, so that decode latency is minimized.

Inputs:

```text
W: set of weight tensors
O: set of operators
B_t: current memory budget
S_t[w]: current residency state of weight w
C: measured cost model
```

The measured cost model contains:

```text
C_compute[o,b]: compute cost of operator o on backend b
C_load[w]: disk -> CPU staging cost
C_transfer[w]: CPU -> GPU transfer cost
C_xform[w,b]: backend-specific layout transform cost
C_sync[o,b1,b2]: backend switch / activation synchronization cost
C_evict[w]: eviction / bookkeeping cost
```

The effective memory budget for model weights is:

```text
B_weight(t) = B_t - M_kv - M_misc - M_safety
```

## Slide 2: Decision Variables

### 1. Weight location variables

For each weight `w`:

```text
x_disk[w] in {0,1}
x_cpu[w]  in {0,1}
x_gpu[w]  in {0,1}
```

Meaning:

```text
x_disk[w] = 1: w is only available from disk
x_cpu[w]  = 1: w is resident in CPU_Elastic memory
x_gpu[w]  = 1: w is resident in OpenCL/GPU memory
```

One-hot location constraint:

```text
x_disk[w] + x_cpu[w] + x_gpu[w] = 1
```

### 2. Operator backend variables

For each operator `o`:

```text
y_cpu[o] in {0,1}
y_gpu[o] in {0,1}
```

Meaning:

```text
y_cpu[o] = 1: operator o runs on CPU_Elastic
y_gpu[o] = 1: operator o runs on OpenCL/GPU
```

One backend per operator:

```text
y_cpu[o] + y_gpu[o] = 1
```

### 3. Memory movement variables

For each weight `w`:

```text
l[w]     in {0,1}: load from disk to CPU staging
t[w]     in {0,1}: transfer from CPU staging to GPU
f_cpu[w] in {0,1}: CPU layout transform / repack
f_gpu[w] in {0,1}: GPU layout transform / convert
e[w]     in {0,1}: evict an existing resident copy
```

These variables explicitly model memory movement costs.

### 4. Backend switch / synchronization variables

For adjacent operators `o` and `o+1`:

```text
s_cpu_gpu[o] in {0,1}: op o on CPU, op o+1 on GPU
s_gpu_cpu[o] in {0,1}: op o on GPU, op o+1 on CPU
```

Meaning:

> If two consecutive operators use different backends, the runtime may pay activation visibility, data copy, command queue barrier, or synchronization overhead.

### 5. Time / anchor variables

For each weight `w`, we also need variables that specify when movement happens:

```text
tau_load[w]     in {0, ..., |O|-1}: operator index where LOAD is issued
tau_transfer[w] in {0, ..., |O|-1}: operator index where TRANSFER is issued
tau_xform[w]    in {0, ..., |O|-1}: operator index where XFORM is issued
tau_evict[w]    in {0, ..., |O|-1}: operator index where EVICT is issued
```

These are not wall-clock timestamps. They are graph-order time indices, or anchor points, used to place memory events relative to operator execution.

Meaning:

```text
tau_load[w] = k
```

means:

> Start loading weight `w` around operator index `k`, before the first operator that needs `w`.

## Slide 3: Feasibility Constraints

### 1. Memory capacity

If CPU and GPU share mobile DRAM, we can use a unified memory budget:

```text
sum_w size_cpu[w] * x_cpu[w]
+ sum_w size_gpu[w] * x_gpu[w]
<= B_weight(t)
```

If backend pools are modeled separately:

```text
sum_w size_cpu[w] * x_cpu[w] <= B_cpu(t)
sum_w size_gpu[w] * x_gpu[w] <= B_gpu(t)
```

### 2. Operator must match weight residency

Let `w(o)` be the main weight consumed by operator `o`.

```text
y_cpu[o] <= x_cpu[w(o)]
y_gpu[o] <= x_gpu[w(o)]
```

Meaning:

> If an operator runs on CPU, its weight must be CPU-resident. If it runs on GPU, its weight must be GPU-resident.

### 3. Layout compatibility

CPU backend:

```text
y_cpu[o] = 1 -> f_cpu[w(o)] = 1 if CPU requires a repacked layout
```

GPU backend:

```text
y_gpu[o] = 1 -> f_gpu[w(o)] = 1 if GPU requires a converted layout
```

Compact form:

```text
f_cpu[w] >= x_cpu[w] * need_cpu_repack[w]
f_gpu[w] >= x_gpu[w] * need_gpu_convert[w]
```

In CP-SAT, this is usually linearized as implications:

```text
x_cpu[w] = 1 and need_cpu_repack[w] = 1 -> f_cpu[w] = 1
x_gpu[w] = 1 and need_gpu_convert[w] = 1 -> f_gpu[w] = 1
```

### 4. Movement induced by the current runtime state

`S_t[w]` is the current runtime state.

If the target location is CPU, but there is no usable CPU copy:

```text
x_cpu[w] = 1 and not S_t[w].cpu_resident -> l[w] = 1
```

If the target location is GPU, but there is no usable GPU copy:

```text
x_gpu[w] = 1 and not S_t[w].gpu_resident -> l[w] = 1
x_gpu[w] = 1 and not S_t[w].gpu_resident -> t[w] = 1
```

If the weight is currently resident but the new plan places it on disk:

```text
S_t[w].resident = 1 and x_disk[w] = 1 -> e[w] = 1
```

### 5. Forced keep for unsafe eviction

If the runtime reports that a tensor is currently resident but has no reliable disk backing:

```text
S_t[w].resident = 1 and S_t[w].has_disk_backing = 0
    -> x_disk[w] = 0
```

Equivalently:

```text
x_cpu[w] + x_gpu[w] = 1
```

Meaning:

> The planner must not generate a plan that loses the only valid copy of a tensor.

## Slide 4: Backend Switch / Synchronization Constraints

Backend switches between adjacent operators:

```text
s_cpu_gpu[o] = 1 iff y_cpu[o] = 1 and y_gpu[o+1] = 1
s_gpu_cpu[o] = 1 iff y_gpu[o] = 1 and y_cpu[o+1] = 1
```

CP-SAT linearization:

```text
s_cpu_gpu[o] <= y_cpu[o]
s_cpu_gpu[o] <= y_gpu[o+1]
s_cpu_gpu[o] >= y_cpu[o] + y_gpu[o+1] - 1

s_gpu_cpu[o] <= y_gpu[o]
s_gpu_cpu[o] <= y_cpu[o+1]
s_gpu_cpu[o] >= y_gpu[o] + y_cpu[o+1] - 1
```

Synchronization cost:

```text
sync_cost =
  sum_o C_sync[o, CPU->GPU] * s_cpu_gpu[o]
+ sum_o C_sync[o, GPU->CPU] * s_gpu_cpu[o]
```

Slide note:

> Backend choice is not independent per operator. A locally faster backend can still be globally worse if it introduces extra activation movement or synchronization.

## Slide 5: Time Index and Prefetch Constraints

The planner can model movement timing using operator-index time rather than wall-clock time.

For each weight `w`:

```text
tau_load[w]:     when LOAD(w) is issued
tau_transfer[w]: when TRANSFER(w) is issued
tau_xform[w]:    when XFORM(w) is issued
tau_evict[w]:    when EVICT(w) is issued
```

Let:

```text
first_use[w] = first operator that consumes weight w
last_use[w]  = last operator that consumes weight w
```

### 1. Movement precedence

```text
tau_load[w] <= tau_transfer[w] <= tau_xform[w] <= first_use[w]
```

For CPU execution, transfer may be disabled:

```text
x_cpu[w] = 1 -> t[w] = 0
```

For GPU execution:

```text
x_gpu[w] = 1 -> tau_transfer[w] <= tau_xform[w] <= first_use[w]
```

### 2. Prefetch window

To avoid loading too early or too late:

```text
first_use[w] - D_prefetch <= tau_load[w] <= first_use[w]
```

where `D_prefetch` is the maximum prefetch distance in operator indices.

### 3. Residency lifetime

If a weight is loaded and later evicted:

```text
tau_load[w] <= first_use[w]
last_use[w] <= tau_evict[w]
```

If the weight remains resident after the planning window:

```text
e[w] = 0
```

### 4. Memory-over-time constraint

For each operator-index time `k`, define:

```text
alive_cpu[w,k] = 1 if w occupies CPU memory at time k
alive_gpu[w,k] = 1 if w occupies GPU memory at time k
```

Then:

```text
sum_w size_cpu[w] * alive_cpu[w,k]
+ sum_w size_gpu[w] * alive_gpu[w,k]
<= B_weight(t,k), for all k
```

This is stricter than the final-residency constraint because it also accounts for temporary staging buffers during LOAD / TRANSFER / XFORM.

Slide note:

> The time index tells the planner when a weight is brought in. This is necessary for modeling prefetching, temporary memory pressure, and overlap with compute.

## Slide 6: Path-Based Objective Function

Full objective:

```text
minimize
    exposed_path_cost
  + sync_cost
  + eviction_cost
  + churn_penalty
  + live_memory_pressure
```

### 1. Backend path cost, not bare compute cost

The backend decision should not compare only:

```text
C_compute[o, CPU] vs C_compute[o, GPU]
```

Instead, each operator/weight pair should compare the full exposed path cost under the current state `S_t`.

Let `w(o)` be the weight consumed by operator `o`.

CPU path:

```text
C_path[o, CPU] =
    C_disk_to_cpu[w(o)]       * l_cpu[w(o)]
  + C_cpu_xform[w(o)]         * f_cpu[w(o)]
  + C_compute[o, CPU]
```

GPU path:

```text
C_path[o, GPU] =
    C_disk_to_cpu[w(o)]       * l_gpu[w(o)]
  + C_cpu_to_gpu[w(o)]        * t_gpu[w(o)]
  + C_gpu_xform[w(o)]         * f_gpu[w(o)]
  + C_compute[o, GPU]
```

The exposed path cost is:

```text
exposed_path_cost =
  sum_o C_path[o, CPU] * y_cpu[o]
+ sum_o C_path[o, GPU] * y_gpu[o]
- sum_w C_overlap[w, tau_load[w], first_use[w]]
```

### 2. Movement variables induced by backend choice

The movement variables are backend-dependent:

```text
y_cpu[o] = 1 and no usable CPU copy in S_t:
  l_cpu[w(o)] = 1
  f_cpu[w(o)] = need_cpu_repack[w(o)]

y_gpu[o] = 1 and no usable GPU copy in S_t:
  l_gpu[w(o)] = 1
  t_gpu[w(o)] = 1
  f_gpu[w(o)] = need_gpu_convert[w(o)]
```

This matters because CPU and GPU have different movement pipelines:

```text
CPU execution:
  disk -> CPU staging -> CPU transform/repack -> CPU compute

GPU execution:
  disk -> CPU staging -> write buffer / CPU->GPU transfer
       -> GPU transform/convert -> GPU compute
```

### 3. Backend synchronization cost

```text
sync_cost =
  sum_o C_sync[o, CPU->GPU] * s_cpu_gpu[o]
+ sum_o C_sync[o, GPU->CPU] * s_gpu_cpu[o]
```

### 4. Eviction, churn, and live-memory pressure

```text
eviction_cost = sum_w C_evict[w] * e[w]
```

```text
churn_penalty =
  lambda_churn * number_of_changed_locations_from_S_t
```

Early prefetch can hide movement latency, but it occupies memory for longer:

```text
live_memory_pressure =
  lambda_live * sum_w live_duration[w] * size[w]
```

Final objective:

```text
minimize
  sum_o C_path[o,CPU] * y_cpu[o]
+ sum_o C_path[o,GPU] * y_gpu[o]
- sum_w C_overlap[w, tau_load[w], first_use[w]]
+ sum_o C_sync[o,CPU->GPU] * s_cpu_gpu[o]
+ sum_o C_sync[o,GPU->CPU] * s_gpu_cpu[o]
+ sum_w C_evict[w] * e[w]
+ lambda_churn * churn[w]
+ lambda_live * live_duration[w] * size[w]
```

Slide note:

> The planner should choose CPU or GPU using end-to-end path cost, not bare compute time. A GPU op may compute faster, but it can still be worse if it requires disk load, CPU-to-GPU write buffer, GPU layout transform, and synchronization. A CPU op may compute slower, but it can win when the CPU copy is already resident or avoids GPU movement.

## Slide 7: Engine-Aware Scheduling Extension

To make the planner closer to a scheduling problem, each movement event can be assigned an anchor operator:

```text
a_load[w]     in operator index domain
a_transfer[w] in operator index domain
a_xform[w]    in operator index domain
```

Precedence:

```text
a_load[w] <= a_transfer[w] <= a_xform[w] <= first_use[w]
```

Prefetch window:

```text
first_use[w] - prefetch_distance <= a_load[w] <= first_use[w]
```

Engine capacity can be modeled with cumulative or optional interval constraints:

```text
DISK engine handles LOAD events
TRANSFER engine handles CPU->GPU events
CPU engine handles CPU xform and CPU compute
GPU engine handles GPU xform and GPU compute
```

Slide note:

> The basic formulation chooses the final plan. A richer formulation can explicitly schedule load, transfer, transform, and compute events on different engines and estimate the overlapped critical path.

## Slide 8: Online vs Offline in This Formulation

Offline table:

```text
solve once for each budget band B
use a canonical initial state S0:
  all weights start from disk
```

Online CP-SAT:

```text
solve at each budget change using the actual runtime state S_t
```

The difference appears in the movement variables:

```text
offline:
  l[w], t[w], f[w] are estimated from a canonical state

online:
  l[w], t[w], f[w], e[w] are derived from the real current residency state
```

Therefore, online planning can avoid:

- reloading weights that are already resident
- repeating layout transforms that have already been done
- evicting a weight that will be needed again soon
- unnecessary CPU/GPU backend switching

Slide note:

> Offline and online planning may use the same objective, but they optimize from different states. The online planner is state-aware, so its movement cost is closer to the real runtime cost.

## Slide 9: Relation to the Current Implementation

The full formulation above is path-based and can choose CPU vs GPU. The current Android experiments are more limited: they mainly evaluate GPU residency planning under dynamic memory pressure.

In the current deployed implementation, compute is effectively GPU-first:

```text
op backend: mostly GPU
weight location: GPU resident vs disk-backed reload
```

So the first deployed version simplifies the full formulation into resident-set selection:

```text
x[w] in {0,1}
```

Constraint:

```text
sum_w size[w] * x[w] <= B_weight(t)
```

Objective:

```text
maximize sum_w value[w] * x[w]
```

where:

```text
value[w] =
    avoided_load_cost[w]
  + avoided_transfer_cost[w]
  + avoided_xform_cost[w]
  + approximate_compute_priority[w]
  + state_stability_bonus[w]
```

This is a projection of the full objective:

```text
keeping a weight resident
  -> avoids future memory movement
  -> reduces expected decode latency
```

Slide note:

> The current experiment should be described as GPU residency planning, not full heterogeneous CPU/GPU placement. The full CP-SAT formulation is path-based, but the current cost table does not yet contain reliable per-op CPU-vs-GPU path measurements. Therefore the current solver compresses movement and approximate compute priority into the resident-set value function.

Current cost-model limitation:

```text
The measured 8B cost table does not yet contain full per-op CPU/GPU path costs.
If the solver falls back to rough constants, GPU is almost always selected.
That fallback is an engineering placeholder, not a scientific model.
```

Required next profiling step:

```text
For each weight/op:
  CPU path = disk->CPU + CPU xform + CPU compute
  GPU path = disk->CPU + write buffer + GPU xform + GPU compute

The solver should choose backend using these measured path costs.
```

## One-Slide Summary

```text
Variables:
  x_cpu[w], x_gpu[w], x_disk[w]     weight location
  y_cpu[o], y_gpu[o]                operator backend
  l[w], t[w], f_cpu[w], f_gpu[w]    load / transfer / transform
  tau_load[w], tau_transfer[w],
  tau_xform[w], tau_evict[w]        movement time indices
  s_cpu_gpu[o], s_gpu_cpu[o]        backend synchronization
  e[w]                              eviction

Constraints:
  x_cpu[w] + x_gpu[w] + x_disk[w] = 1
  y_cpu[o] + y_gpu[o] = 1
  memory capacity constraint
  memory-over-time constraint
  backend execution feasibility
  layout compatibility
  movement precedence and prefetch window
  movement variables derived from S_t
  unsafe eviction is forbidden

Objective:
  minimize compute cost
         + load / transfer / transform cost
         + backend synchronization cost
         + eviction / churn cost
```

## Polished Single-Slide Version

Use this version to replace the current PPT page.

### Title

```text
Formulation of the CP-SAT Planner
```

### Left block: Inputs

```text
Inputs
  W        weight tensors
  O        operators
  B_t      current memory budget
  S_t(w)   current residency state
  C        measured cost model

Effective weight budget
  B_weight = B_t - M_kv - M_misc - M_safety
```

### Middle block: Decision variables

```text
Weight placement
  x_disk(w), x_cpu(w), x_gpu(w) in {0,1}

Operator backend
  y_cpu(o), y_gpu(o) in {0,1}

Movement actions
  l(w)       load from disk
  t(w)       transfer CPU -> GPU
  f_cpu(w)   CPU layout transform
  f_gpu(w)   GPU layout transform
  e(w)       eviction

Time anchors
  tau_load(w), tau_transfer(w), tau_xform(w)
```

### Right block: Main constraints

```text
One location per weight
  x_disk(w) + x_cpu(w) + x_gpu(w) = 1

One backend per op
  y_cpu(o) + y_gpu(o) = 1

Memory budget
  sum_w size_cpu(w)x_cpu(w)
      + size_gpu(w)x_gpu(w) <= B_weight

Execution feasibility
  y_cpu(o) <= x_cpu(w(o))
  y_gpu(o) <= x_gpu(w(o))

Movement ordering
  tau_load(w) <= tau_transfer(w)
              <= tau_xform(w) <= first_use(w)
```

### Bottom block: Objective

```text
minimize
    sum_o C_compute(o,CPU) y_cpu(o)
  + sum_o C_compute(o,GPU) y_gpu(o)
  + sum_w C_load(w)        l(w)
  + sum_w C_transfer(w)    t(w)
  + sum_w C_xform(w,CPU)   f_cpu(w)
  + sum_w C_xform(w,GPU)   f_gpu(w)
  + sum_o C_sync(o)        backend_switch(o)
  + sum_w C_evict(w)       e(w)
  + lambda_churn * churn
```

### Speaker note

```text
The planner jointly optimizes three coupled choices:
1. where each weight resides,
2. which backend executes each operator,
3. when memory movement happens.

The key point is that backend placement cannot be optimized by compute time alone:
moving a weight to GPU may reduce compute time but introduce load, transfer,
layout transform, and synchronization costs. The CP-SAT objective captures this
global trade-off under the current memory budget and runtime state.
```

### Layout suggestion

Use a 3-column layout:

```text
[Inputs]        [Variables]             [Constraints]

                 [Objective across bottom]
```

Recommended visual simplification:

- Remove the old `x_w` variable from the slide, or put it only in a small note as "simplified deployed version".
- Use `x_disk/x_cpu/x_gpu` as the main placement variables.
- Put `tau_*` variables beside movement actions to show that movement is scheduled, not just selected.
- Keep formulas in normal math font if possible; avoid monospaced text for every line.
