# 独立设计方案：Adaptation Capsules for Dynamic Memory Budgets

## 1. 重新定义第四点

如果不完全 follow 之前图片里的流程，我会把第四点设计成一个更独立的系统机制：

```text
不要把 online adaptation 看成“从 previous plan 修改到某个 target plan”。
而是把它看成：在当前 runtime state 和新 memory budget 下，选择一组小而可组合
的 adaptation actions，使系统快速恢复到一个足够好的执行状态。
```

这个设计可以叫：

```text
Adaptation Capsules
```

核心思想是：

```text
offline 阶段不只是保存 top-k complete plans，
而是从这些 plans 和 profiling 结果中提取可复用的 adaptation capsules。

online 阶段不再完整求解 plan，也不一定完整追随某个 P_base，
而是根据当前 S_runtime、budget direction、expected horizon，
选择一组 capsules 来执行。
```

这里的 capsule 是一个语义完整、成本可估计、可独立或半独立执行的 adaptation
单元，比如：

```text
promote one attention block to GPU
demote one FFN block from GPU to CPU
materialize a future layer group on CPU
evict a cold tensor group
convert a layer group to GPU layout
keep current resident group and skip target demotion
```

换句话说，第四点的 novelty 不是“diff graph 怎么剪枝”，而是：

```text
把 online replanning 转化成 capsule selection：
runtime 选择一组预分析过的 adaptation capsules，而不是求解完整 placement。
```

## 2. 为什么我觉得这个方案更好讲

### 2.1 它比 target-plan patch 更自由

`Structured Plan Patch` 是从一个 selected candidate `P_base` 出发，然后决定
哪些 diff 要接受。这个思路合理，但它仍然被 `P_base` 限制：

```text
如果 P_base 没有包含某个 action，patch search 通常不会考虑它。
```

例如，当前状态下最划算的动作可能不是向 `P_base` 靠近，而是：

```text
暂时保留当前 GPU-resident attention；
只 materialize 下一个 layer group 的 CPU staging；
同时 evict 一个 cold FFN 来防止下一次 budget shrink。
```

这些动作可能不是任何一个 complete plan diff 的直接子集，但它们对动态
runtime 更合理。

### 2.2 它更符合动态系统

动态 budget 的困难不是每次都找到一个静态最优 plan，而是持续处理：

```text
budget increase
budget shrink
budget oscillation
runtime state drift
partial materialization
deferred eviction
pipeline overlap opportunity
```

这些情况更像一个在线控制问题，而不是离散 plan lookup 问题。因此把 online
决策设计成 action/capsule selection，会比“选一个 plan 再 patch”更自然。

### 2.3 它更 lightweight

capsule 的成本和收益可以 offline 预计算或半预计算。online 阶段只需要根据
当前状态修正少量参数：

```text
当前是否已经 resident
当前 memory pressure
当前 budget direction
当前 horizon
当前 pipeline window
```

这比 full CP-SAT 轻，也比在完整 diff graph 上搜索更容易控制开销。

## 3. Capsule 是什么

一个 capsule 是一个带 predicate、action、cost model、effect model 的结构：

```text
Capsule {
    id
    semantic_scope
    action_type
    precondition
    effect
    resource_delta
    transition_model
    benefit_model
    risk_model
    dependencies
    conflicts
    priority_hint
}
```

### 3.1 semantic_scope

表示这个 capsule 作用在哪个语义区域：

```text
layer group
layer
attention block
QKV group
FFN block
tensor group
individual tensor
```

我会建议主要以 component-level capsule 为主，而不是 tensor-level：

```text
attention capsule
FFN capsule
layer-group staging capsule
cold-group eviction capsule
```

原因是 component-level 更符合 LLM execution structure，也更容易解释收益和
成本。

### 3.2 action_type

可以定义几类 capsule：

```text
PromoteCapsule
    CPU/disk -> GPU，用于 budget increase 或性能恢复。

DemoteCapsule
    GPU -> CPU/disk，用于 budget shrink。

StageCapsule
    disk/raw -> CPU staged，用于提前准备未来 layer。

TransformCapsule
    CPU staged -> backend-specific layout。

EvictCapsule
    释放 cold resident region。

PreserveCapsule
    显式保留当前 resident region，即使 target plan 想 demote。

DeferCapsule
    延迟某个 expensive transition，等待 horizon 更清楚。
```

这里 `PreserveCapsule` 和 `DeferCapsule` 是我觉得比较有意思的地方。它们让
系统不只是“做改变”，也能把“不做改变”作为一个显式决策，并记录原因。这对
动态 budget 很重要。

### 3.3 precondition

每个 capsule 只在某些 runtime state 下可用：

```text
PromoteCapsule(attn L12):
    precondition:
        L12 attention is CPU resident or disk materializable
        GPU free memory >= required bytes or evict capsule available

DemoteCapsule(ffn L20):
    precondition:
        L20 FFN is GPU resident
        CPU/disk fallback exists

PreserveCapsule(attn L8):
    precondition:
        L8 attention is already GPU resident
        keeping it does not violate hard memory budget
```

precondition 的作用是让 online selection 很快过滤掉不可行 action。

### 3.4 effect

effect 描述执行后状态怎么变：

```text
GPU bytes +120 MiB
CPU bytes -80 MiB
attention L12 backend becomes GPU
layout becomes GPU-transformed
future decode cost decreases by X
```

这让 online planner 可以做轻量 state update，而不是重新构造完整 plan。

## 4. Offline 阶段做什么

offline 不只是保存完整 plans，而是构建 capsule library。

### 4.1 从 top-k plans 中挖 capsule

对相邻 budget、不同 candidate plans 做 diff：

```text
Diff(P_i(B_low), P_j(B_high))
Diff(P_i(B_high), P_j(B_low))
```

从这些 diff 中抽取高频、有语义边界的变化：

```text
attention promote patterns
FFN demote patterns
layer-group staging patterns
cold-region eviction patterns
preserve-current patterns
```

如果某类变化在很多 plan transition 中反复出现，说明它是一个稳定 adaptation
primitive，适合做成 capsule。

### 4.2 给 capsule 建 profile

offline/profile 阶段估计：

```text
transition_cost distribution
per-token benefit
memory delta
overlap opportunity
failure/risk probability
dependency/conflict relation
```

这里不要只保存一个平均值，最好保存区间：

```text
cost_low, cost_high
benefit_low, benefit_high
overlap_low, overlap_high
```

因为 online 状态会变，区间可以支持 conservative / optimistic decision。

### 4.3 构建 capsule graph

capsule 之间有关系：

```text
dependency:
    transform depends on stage

conflict:
    two promote capsules compete for same GPU memory

substitution:
    promote attention and preserve existing attention may be alternative ways
    to protect decode performance

synergy:
    QKV promote + O promote together has better scheduling benefit

repair:
    promote capsule may require one demote/evict capsule for memory feasibility
```

这个 graph 是 offline 构建的 capsule-level graph，而不是 online 临时从完整
plan diff 里建大图。

## 5. Online 阶段做什么

online 阶段可以分成三层：

```text
Layer 1. Safety response
    如果 budget shrink，先确保 hard memory budget 不被违反。

Layer 2. Performance recovery
    在安全基础上选择能提升 decode 的 promote/stage/transform capsules。

Layer 3. Stability control
    根据 budget volatility/horizon 决定 preserve/defer，避免 oscillation。
```

### 5.1 Safety response

当 budget 下降时，第一目标不是性能，而是可行性：

```text
free required memory with minimum future loss
```

选择 demote/evict capsules：

```text
loss(c) =
    H * lost_perf_per_token(c)
  + visible_transition_cost(c)
  + risk_penalty(c)

loss_per_byte(c) =
    loss(c) / freed_bytes(c)
```

优先执行低 `loss_per_byte` capsule，直到：

```text
memory_used <= safe_budget
```

这个部分可以作为 deterministic fast path，非常 lightweight。

### 5.2 Performance recovery

当 budget 增加，或者 safety repair 后还有空间，选择 promote/stage/transform
capsules：

```text
utility(c) =
    H * gain_per_token(c)
  - visible_transition_cost(c)
  - memory_pressure_penalty(c)
  - risk_penalty(c)
```

接受条件：

```text
utility(c) > threshold
and precondition(c) is true
and memory feasible after effect(c)
```

如果多个 capsule 互相冲突，选 utility 更高或 utility_per_byte 更高的。

### 5.3 Stability control

动态 budget 里最容易出问题的是 oscillation：

```text
budget 上升 -> promote
budget 下降 -> demote
budget 又上升 -> promote
```

这样系统可能一直在做 transition。为了解决这个问题，我会把 stability 做成
显式设计，而不是事后调参。

定义：

```text
volatility = recent_budget_change_rate
H = expected stable horizon
```

当 volatility 高或 H 短时：

```text
提高 promote threshold
降低 preserve/defer threshold
偏向小 capsule 而不是大 capsule
```

当 volatility 低或 H 长时：

```text
允许更大的 promote/transform capsule
追求更接近 steady-state optimal plan
```

这可以形成一个很好的系统故事：

```text
the planner adapts not only to the current budget, but also to budget stability.
```

## 6. Online selection algorithm

整体算法可以写成：

```text
Algorithm: CapsuleAdapt

Input:
    S_runtime
    B_t
    capsule_library
    capsule_graph
    H
    volatility
    online_bound

Output:
    P_adjusted
    capsule_trace

1. active = Filter capsules whose preconditions match S_runtime.

2. if memory_used(S_runtime) > safe_budget(B_t):
       repair_set = Select demote/evict capsules by loss_per_byte.
       Apply repair_set to S_runtime.

3. Update active capsules after repair.

4. Score performance capsules:
       utility(c) =
           H * gain_per_token(c)
         - visible_transition_cost(c)
         - memory_pressure_penalty(c)
         - risk_penalty(c)
         - volatility_penalty(c)

5. Select capsules under constraints:
       dependency satisfied
       conflict avoided
       memory feasible
       online_bound not exceeded

6. Add preserve/defer capsules for regions where transition is not worth it.

7. Emit P_adjusted = Apply selected capsule effects to S_runtime.
```

这个算法本质上是一个小规模 constrained selection。为了保持 lightweight，不
需要 CP-SAT；可以用 greedy + repair：

```text
budget shrink:
    greedy by lowest loss_per_byte

budget increase:
    greedy by highest utility_per_byte or utility_per_ms

conflict:
    local winner-take-all

dependency:
    include prerequisite if combined utility remains positive
```

如果想更 formal，可以说在线做的是：

```text
bounded capsule selection under dependency and memory constraints
```

## 7. Cost function

### 7.1 Capsule utility

对 performance capsule：

```text
U(c) =
    H * G(c, S_runtime)
  - V(c, S_runtime)
  - M(c, B_t)
  - R(c)
  - O(c, volatility)
```

含义：

```text
G:
    per-token gain，取决于当前状态。如果目标已经 resident，则 gain 可能为 0。

V:
    visible transition cost，即无法被 pipeline overlap hide 的部分。

M:
    memory pressure penalty，越接近 budget penalty 越高。

R:
    risk penalty，包括 profile uncertainty、fragmentation、failure retry。

O:
    oscillation penalty，budget 越不稳定，越惩罚大规模 promote/transform。
```

### 7.2 Visible transition cost

```text
V(c) =
    raw_cost(c) * (1 - overlap(c, S_runtime))
```

其中：

```text
raw_cost(c) =
    disk_load
  + CPU_prepare
  + GPU_transform
  + evict
  + sync
```

overlap 取决于 capsule 的执行位置：

```text
near-term layer:
    overlap 小，因为马上要用。

future layer group:
    overlap 大，因为可以被当前 decode compute hide。
```

### 7.3 Oscillation penalty

这个是我认为比较 novel 的点。动态 budget 下，transition 不只是一次性成本，
还可能因为 budget 抖动反复发生。

可以定义：

```text
O(c, volatility) =
    lambda_osc * volatility * reversibility_cost(c)
```

其中：

```text
reversibility_cost(c):
    如果 budget 很快反向变化，撤销该 capsule 需要付出的成本。
```

例如：

```text
large FFN promote:
    reversibility_cost 高，因为 demote/evict 又要付出较大代价。

small attention preserve:
    reversibility_cost 低。
```

这让 planner 在 budget 不稳定时自然偏保守，在 budget 稳定时才做大动作。

### 7.4 Safety loss

对 budget shrink 的 demote/evict capsule：

```text
L(c) =
    H * lost_gain_per_token(c)
  + V(c)
  + R(c)
```

选择目标：

```text
minimize L(selected)
subject to freed_bytes(selected) >= required_free_bytes
```

lightweight greedy：

```text
choose lowest L(c) / freed_bytes(c)
```

## 8. 为什么这个方案 novel

我觉得这个方案的 novelty 可以这样讲。

### 8.1 从 plan selection 到 action selection

现有 baseline 的共同点是选择 plan：

```text
offline:
    选 budget 对应的 plan。

online CP-SAT:
    在线求一个完整 plan。

candidate-select:
    从 top-k 中选一个完整 plan。
```

我们的 design 不是选择完整 plan，而是选择 adaptation capsules：

```text
runtime chooses what to change, what to preserve, and what to defer.
```

这更贴近动态系统。

### 8.2 显式建模 preserve/defer

很多系统只把 adaptation 看成“做哪些迁移”。但动态 budget 下，“不迁移”也
应该是一个有成本模型的 decision：

```text
preserve:
    当前 resident state 已经不错，保留它比追 target 更好。

defer:
    当前 horizon 不确定，推迟昂贵 transition，等待更多 budget evidence。
```

把 preserve/defer 做成 capsule，可以让系统避免无意义 oscillation。

### 8.3 显式建模 budget stability

offline plan 和普通 online solver 通常只看当前 budget。这个 design 把
budget stability 放进 utility：

```text
same budget, different volatility -> different adaptation action
```

这对 mobile setting 很重要，也比较符合 ASPLOS 系统味道。

### 8.4 离线挖掘 reusable adaptation primitives

不是手写一堆 heuristic，而是从 offline top-k plan transitions 和 profile 中
挖出 reusable capsules。这让方法比 rule-based policy 更可信：

```text
offline heavy analysis extracts adaptation primitives;
online lightweight policy composes them.
```

## 9. 一个完整例子

假设 budget 从 4096 MiB 上升到 4608 MiB，但最近 30 秒内 budget 已经上下波动
多次，因此：

```text
H = 32 tokens
volatility = high
```

当前状态：

```text
S_runtime:
    L12 attention: CPU resident
    L12 FFN:       CPU resident
    L18 attention: already GPU resident
    L20 FFN:       GPU resident but cold
    GPU free:      380 MiB
```

如果 follow 一个 complete candidate `P_base`，它可能想做：

```text
promote L12 attention
promote L12 FFN
keep L18 attention GPU
keep L20 FFN GPU
```

CapsuleAdapt 会看到这些 active capsules：

```text
C1: Promote L12 attention
    gain = 0.35 ms/token
    visible_cost = 7 ms
    memory_delta = +120 MiB
    reversibility_cost = low

C2: Promote L12 FFN
    gain = 0.20 ms/token
    visible_cost = 36 ms
    memory_delta = +360 MiB
    reversibility_cost = high

C3: Preserve L18 attention
    gain = avoid future reload
    visible_cost = 0 ms
    memory_delta = 0 MiB
    reversibility_cost = low

C4: Evict cold L20 FFN
    lost_gain = 0.05 ms/token
    visible_cost = 4 ms
    memory_delta = -220 MiB
    reversibility_cost = medium
```

utility：

```text
U(C1) = 32 * 0.35 - 7 - memory_penalty(120) - osc_penalty(low)
      ~= positive

U(C2) = 32 * 0.20 - 36 - memory_penalty(360) - osc_penalty(high)
      ~= strongly negative

C3 is preserve:
      accept because it avoids unnecessary transition and does not hurt budget.

C4 is optional repair:
      only needed if accepting other capsules would exceed safe budget.
```

最终选择：

```text
accept C1: promote L12 attention
accept C3: preserve L18 attention
defer C2: do not promote L12 FFN under high volatility
skip C4: no need to evict cold L20 FFN yet
```

输出：

```text
P_adjusted:
    L12 attention -> GPU
    L12 FFN       -> CPU
    L18 attention -> keep GPU
    L20 FFN       -> keep current
```

如果之后 budget 稳定，`H` 变成 256 tokens、volatility 低，那么同一个 C2 可能
变成 positive：

```text
U(C2) = 256 * 0.20 - 36 - memory_penalty - small_osc_penalty
      > 0
```

此时系统再 promote FFN。这个例子能清楚说明：

```text
同一个 budget 下，系统会因为 expected horizon 和 volatility 不同而做不同
adaptation。
```

这就是动态 memory budget 里很有价值的 design point。

## 10. 和 Structured Plan Patch 的关系

这两个方案不是互斥的。

```text
Structured Plan Patch:
    从 selected P_base 的 diff 出发，做 structured accept/prune/expand。

Adaptation Capsules:
    从 capsule library 出发，选择当前最值得执行、保留或推迟的 actions。
```

如果要做成更强的系统，可以组合：

```text
1. candidate selection 给出 P_base，提供目标方向。
2. P_base diff 激活相关 capsules。
3. capsule policy 也可以激活 P_base 之外的 preserve/defer/repair capsules。
4. online 选择 capsules，输出 P_adjusted。
```

这样既保留了 offline candidate 的方向感，又不被单个 target plan 限制。

## 11. 我更推荐的最终设计

如果目标是 ASPLOS system paper，我更推荐把第四点最终表述成：

```text
State-aware Capsule-based Adaptation
```

而不是单纯的：

```text
diff graph patching
```

原因是 capsule-based adaptation 更容易讲出系统贡献：

```text
1. offline heavy analysis extracts reusable adaptation capsules from top-k
   plans and profiles;

2. online lightweight policy selects capsules based on current runtime state,
   budget direction, horizon, memory pressure, and volatility;

3. the runtime can explicitly choose to promote, demote, preserve, or defer
   regions instead of blindly switching complete plans;

4. this converts expensive online re-planning into bounded action composition.
```

这个思路比较新，也比较合理。它不是为了 graph 而 graph，而是围绕动态 budget
的本质：系统每次不一定需要一个新 plan，而是需要一组最合适的 adaptation
actions。

## 12. 还需要进一步设计的问题

1. capsule library 是完全从 offline top-k plan transitions 挖出来，还是结合
   手写 semantic templates？
2. capsule 的 semantic granularity 应该固定在 attention/FFN，还是允许按
   profile 自动合并 layer groups？
3. preserve/defer capsule 怎么在 plan representation 里表达？
4. volatility 和 H 怎么估计，是否需要从真实手机 memory trace 学？
5. online selection 用 greedy 是否足够，还是需要一个很小的 bounded knapsack？
6. evaluation 怎么证明 capsule selection 比 target-plan patch 更强？
