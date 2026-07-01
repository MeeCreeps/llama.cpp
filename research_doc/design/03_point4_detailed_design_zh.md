# 第四点详细设计：State-aware Structured Plan Patching

## 1. 第四点的定位

前三个阶段解决的是：

```text
offline:
    为每个 budget 预先准备 top-k candidate plans。

online candidate selection:
    根据当前 runtime state S_runtime，从 top-k candidates 中选出 P_base。
```

第四点要解决的是：

```text
P_base 是一个完整 target plan，但完整切换到 P_base 不一定划算。
```

原因是动态 memory budget 下，planner 面对的不是一次静态 placement，而是一个
带状态的转换问题：

```text
current runtime state S_runtime  ->  candidate target plan P_base
```

这个转换里面有些部分值得做，有些部分不值得做，还有些部分必须组合起来才
有意义。因此第四点的核心不是重新求一个完整 plan，而是：

```text
在 S_runtime 和 P_base 之间选择一组最值得执行的 structured patch。
```

输出不是完整的 `P_base`，而是一个 adjusted plan：

```text
P_adjusted = S_runtime + AcceptedPatches(Diff(S_runtime, P_base))
```

这一步可以称为：

```text
State-aware Structured Plan Patching
```

## 2. 为什么这样设计

### 2.1 为什么不直接使用 P_base

candidate selection 已经比 offline single-plan 更好，因为它根据
`S_runtime` 选择了一个比较合适的 candidate。但它仍然是 whole-plan 级别：

```text
要么切到 P_base，
要么不切到 P_base。
```

这会漏掉一个重要机会：`P_base` 里面不同区域的收益和转换成本并不一样。

例如：

```text
attention promote:
    可能 gain 高、transform bytes 小、overlap 好。

FFN promote:
    可能 memory delta 大、transform bytes 大、短 horizon 下不划算。
```

如果 whole-plan 切换，这两个部分会被绑在一起；如果 structured patch，就能
只接受 attention promote，拒绝或延迟 FFN promote。

### 2.2 为什么不重新 online solve

online CP-SAT 可以从全局角度求出 state-aware plan，但代价太大。第四点的
目标是复用 offline candidate 的质量，同时避免在线完整搜索 placement 空间。

所以 online 搜索空间从：

```text
所有 tensors 的 backend/layout/residency 组合
```

缩小成：

```text
S_runtime 和 P_base 之间发生变化的 diff subgraphs
```

这就是 lightweight 的来源。

### 2.3 为什么需要结构化 diff

如果只把 diff 展平成 tensor list，planner 会错误地假设每个 tensor move
独立。但真实系统里成本、收益、memory feasibility 都有耦合：

```text
共享成本:
    多个 tensor 共享 disk load、CPU staging、GPU transform。

共享收益:
    attention 或 FFN 作为 component 更能反映 decode gain。

memory coupling:
    一个 GPU promotion 可能需要另一个 eviction。

pipeline overlap:
    多个 transfer/transform 可能被同一个 compute window hide。
```

因此 diff 需要组织成结构化对象，让 planner 在合适的粒度做决策。

## 3. 总体流程

第四点可以分成五个步骤：

```text
Step 1. Build raw diff
    比较 S_runtime 和 P_base，找出所有 placement/layout/materialization 变化。

Step 2. Build hierarchical diff DAG
    把 raw diff 聚合到 layer/component/tensor 层级，并加入 resource、
    dependency、conflict、overlap edges。

Step 3. Score patch nodes
    对每个 node/subgraph 估计 horizon-aware benefit 和 visible transition cost。

Step 4. Bounded patch search
    在 online time/node budget 内做 accept / reject / expand / contract / prune。

Step 5. Feasibility repair and emit plan
    确保 accepted patches 满足 CPU/GPU memory budget，输出 P_adjusted。
```

这五步可以写进系统设计图中：

```text
S_runtime + P_base
        |
        v
Raw Diff
        |
        v
Hierarchical Diff DAG
        |
        v
Bounded Patch Search
        |
        v
Accepted Patches + Repair
        |
        v
P_adjusted
```

## 4. 输入和输出

### 4.1 输入

```text
S_runtime:
    当前真实 runtime state，包括每个 weight 是否 CPU/GPU resident，
    是否已经 materialized，是否有 backend-specific layout。

P_base:
    online candidate selection 选出的 target plan。

B_t:
    当前 memory budget，包括 CPU/GPU 可用 budget。

H:
    expected horizon，表示当前 budget/state 预计会持续多少 token。

profile:
    offline/online profile 信息，包括 load bandwidth、transform throughput、
    compute cost、overlap estimate。

online_bound:
    在线搜索的时间预算或 node expansion 预算。
```

### 4.2 输出

```text
P_adjusted:
    基于当前状态和 accepted patches 生成的 adjusted plan。

PatchTrace:
    记录每个 node 是 accepted、rejected、expanded、contracted 还是 pruned。
```

`PatchTrace` 很重要，因为论文里需要证明这个方法不是黑盒 heuristic，而是有
可解释机制。

## 5. Raw Diff 表示

对每个 tensor `w`，比较 runtime state 和 target plan：

```text
runtime state:
    backend_runtime(w)
    layout_runtime(w)
    resident_runtime(w)

target plan:
    backend_target(w)
    layout_target(w)
    resident_target(w)
```

如果任一字段不同，就产生一个 raw diff item：

```text
DiffItem(w) = {
    tensor_id,
    layer_id,
    component,
    old_backend,
    new_backend,
    old_layout,
    new_layout,
    old_residency,
    new_residency,
    bytes,
    action_type
}
```

`action_type` 可以是：

```text
promote:
    CPU/disk -> GPU

demote:
    GPU -> CPU/disk

materialize:
    disk/raw -> CPU resident/staged

transform:
    raw/staged -> backend-specific layout

evict:
    resident -> non-resident

keep:
    runtime 和 target 一致，不需要变化
```

注意：`keep` 通常不需要作为 patch 执行，但可以作为 graph 里的边界信息，
比如判断一个 layer 里已经有哪些 tensor 在 GPU 上。

## 6. Hierarchical Diff DAG 结构

### 6.1 为什么是 DAG

推荐把结构称为：

```text
Hierarchical Diff DAG
```

它不是完整模型图，而是只覆盖当前 diff 的局部图。它包含两类关系：

```text
hierarchical edges:
    表达模型结构上的父子关系。

cross edges:
    表达系统代价上的耦合关系。
```

### 6.2 层级节点

层级可以这样组织：

```text
root
  layer_group[0-3]
    layer[0]
      attention
        qkv
          q_proj
          k_proj
          v_proj
        o_proj
      ffn
        gate_up
          gate_proj
          up_proj
        down_proj
```

每个节点代表一个 patch candidate。比如：

```text
attention node:
    表示接受该 attention component 内所有 relevant diff。

qkv node:
    表示接受 q/k/v 相关 diff。

tensor node:
    表示接受单个 tensor diff。
```

### 6.3 Cross edges

除了父子边，还需要这些 cross edges：

```text
resource-sharing edge:
    两个 node 共享 load/prepare/transform 资源。

dependency edge:
    一个 node 必须依赖另一个 node 先完成。
    例如 GPU compute layout 依赖 CPU materialization。

conflict edge:
    两个 node 不能同时接受，或者同时接受会超过 memory budget。

overlap edge:
    两个 node 的 transfer/transform 可以被同一段 compute window hide。

ordering edge:
    某些 patch 必须在特定 layer decode 前完成。
```

### 6.4 Node schema

每个 node 可以记录：

```text
PatchNode {
    id
    type                  // promote, demote, materialize, transform, evict, mixed
    level                 // layer_group, layer, component, tensor
    tensors               // covered tensor ids
    children
    parents
    edges

    bytes_read
    bytes_write
    bytes_transform
    memory_delta_cpu
    memory_delta_gpu

    old_compute_ms_per_token
    new_compute_ms_per_token
    profile_confidence

    dependency_set
    conflict_set
    overlap_group

    lower_score
    upper_score
    decision_state        // unknown, accepted, rejected, expanded, pruned
}
```

这个 schema 的好处是设计上很清楚：node 既是图结构的一部分，也是 cost model
的计算单位。

## 7. Cost function 设计

第四点的 cost function 应该表达一句话：

```text
接受一个 patch 是否值得，取决于未来 H 个 token 的稳态收益是否超过现在可见
的 transition cost 和风险。
```

### 7.1 Steady-state benefit

对于 patch node `v`：

```text
benefit(v, H) = H * gain_per_token(v)
```

其中：

```text
gain_per_token(v) =
    old_compute_ms_per_token(v) - new_compute_ms_per_token(v)
```

如果是 promote 到 GPU，`gain_per_token` 通常为正；如果是 demote/evict，它
可能为负，但可以释放 memory，为其他高收益 patch 创造可行性。

对于 mixed node，可以按 covered tensors/component 汇总：

```text
gain_per_token(v) =
    sum gain_per_token(child)
  + interaction_gain(v)
```

`interaction_gain` 表示 group-level 效果，比如 QKV 一起 promote 后调度更
稳定，或者整个 attention block backend 一致后减少同步。

### 7.2 Transition cost

原始 transition cost：

```text
raw_transition_cost(v) =
    disk_load_cost(v)
  + cpu_prepare_cost(v)
  + gpu_transform_cost(v)
  + evict_cost(v)
  + sync_cost(v)
```

各项可以展开为：

```text
disk_load_cost(v) =
    bytes_read_from_disk(v) / disk_bandwidth

cpu_prepare_cost(v) =
    cpu_prepare_bytes(v) / cpu_prepare_bandwidth

gpu_transform_cost(v) =
    gpu_transform_bytes(v) / gpu_transform_bandwidth

evict_cost(v) =
    evict_bytes(v) / evict_bandwidth + allocator_overhead

sync_cost(v) =
    backend_sync_count(v) * sync_latency
```

### 7.3 Pipeline overlap

不是所有 transition cost 都暴露在关键路径上。假设 `overlap_ratio(v)` 表示
有多少比例能被 compute hide：

```text
visible_transition_cost(v) =
    raw_transition_cost(v) * (1 - overlap_ratio(v))
```

`overlap_ratio` 可以来自 profile：

```text
overlap_ratio(v) = min(
    available_compute_window_ms(v) / raw_transition_cost(v),
    max_overlap_cap
)
```

如果 patch 必须在马上要执行的 layer 之前完成，overlap ratio 应该更低；如果
patch 是未来 layer group 的 prepare，overlap ratio 可以更高。

### 7.4 Memory pressure penalty

接受 patch 后会改变 memory：

```text
gpu_after = gpu_current + sum memory_delta_gpu(accepted)
cpu_after = cpu_current + sum memory_delta_cpu(accepted)
```

如果接近 budget，需要加 penalty：

```text
memory_pressure_penalty(v) =
    lambda_gpu * pressure(gpu_after, gpu_budget)
  + lambda_cpu * pressure(cpu_after, cpu_budget)
```

一个简单 pressure 函数：

```text
pressure(used, budget) =
    0,                                  if used <= safe_ratio * budget
    ((used - safe_ratio * budget) /
     ((1 - safe_ratio) * budget))^2,    otherwise
```

如果超过 hard budget，则 node/subgraph infeasible：

```text
used > budget  =>  infeasible
```

### 7.5 Risk penalty

动态 budget 下，一些 patch 的收益不确定。风险可以包括：

```text
budget_volatility:
    budget 可能很快又变化，H 估计不稳定。

profile_uncertainty:
    profile 对当前手机状态、thermal、后台负载不准。

fragmentation_risk:
    大块 allocation/eviction 可能失败或导致后续 allocator pressure。

retry_risk:
    materialization/transform 失败后需要 fallback。
```

可以写成：

```text
risk_penalty(v) =
    lambda_vol * budget_volatility
  + lambda_prof * (1 - profile_confidence(v))
  + lambda_frag * fragmentation_risk(v)
  + lambda_retry * retry_risk(v)
```

### 7.6 Final score

最终 score：

```text
score(v) =
    H * gain_per_token(v)
  - visible_transition_cost(v)
  - memory_pressure_penalty(v)
  - risk_penalty(v)
```

解释：

```text
score > 0:
    预计未来收益大于现在成本，倾向接受。

score < 0:
    当前不值得支付这个 patch。
```

但为了避免 profile 误差，最好使用上下界：

```text
lower_score(v):
    conservative estimate，收益低估，成本高估。

upper_score(v):
    optimistic estimate，收益高估，成本低估。
```

决策：

```text
if lower_score(v) > accept_threshold and feasible(v):
    accept(v)

elif upper_score(v) < reject_threshold or infeasible(v):
    prune(v)

else:
    expand(v)
```

这个上下界设计能解释为什么方法 lightweight：很多 node 可以在 coarse level
直接 accept/prune，不需要展开到底。

## 8. Search policy 设计

### 8.1 Frontier

planner 维护一个 frontier：

```text
frontier = candidate patch nodes whose decision is unknown
```

初始化时不放所有 tensor，而是放 coarse nodes：

```text
layer groups 或 layers
```

这样一开始搜索空间很小。

### 8.2 Node priority

每轮选择哪个 node 处理，可以用：

```text
priority(v) =
    max(upper_score(v), uncertainty(v), memory_relief(v))
```

其中：

```text
upper_score(v):
    有潜在收益的 node 优先看。

uncertainty(v):
    lower_score 和 upper_score 差距大的 node 优先展开。

memory_relief(v):
    budget shrink 时，能释放 memory 的 demote/evict node 优先。
```

### 8.3 Accept

accept 一个 node 的含义是：

```text
把该 node 覆盖的 diff items 加入 accepted patch set。
```

同时要处理 dependency：

```text
如果 accept GPU promote，
必须保证 materialize/transform dependency 也被 accept 或已经满足。
```

如果 accept 后 memory 仍 feasible，就保留；否则触发 repair 或 rollback。

### 8.4 Prune

prune 一个 node 的含义是：

```text
该 node 覆盖区域保持 S_runtime 的状态，不向 P_base 切换。
```

如果一个 parent 被 prune，它的 children 默认不再展开。这样可以减少在线工作。

### 8.5 Expand

expand 一个 node 的含义是：

```text
把粗粒度 node 替换成更细粒度 children。
```

例如：

```text
layer node
    -> attention node
    -> ffn node

attention node
    -> qkv node
    -> o_proj node

qkv node
    -> q tensor
    -> k tensor
    -> v tensor
```

expand 的触发条件：

```text
lower_score <= accept_threshold
and upper_score >= reject_threshold
```

或者：

```text
node 内部 gain/cost variance 高
memory_delta 太大
dependency/conflict 太多
```

### 8.6 Contract

contract 是把多个 child nodes 合并成一个 group patch。它适合以下情况：

```text
children 都明显 positive
children 共享 resource edge
children 同属一个 component
contract 后 memory feasible
```

例子：

```text
q_proj promote + k_proj promote + v_proj promote
    -> qkv promote
```

contract 的意义是防止系统陷入 tensor-level 微管理，同时更准确估计共享成本。

### 8.7 Feasibility repair

即使每个 accepted patch 局部看起来可行，组合后也可能超过 memory budget。
repair 可以按两种方式做。

第一种：在线搜索过程中强约束：

```text
accept(v) only if current_memory + delta(v) <= budget
```

优点是简单安全，缺点是可能错过“先 promote 后 evict repair”的组合。

第二种：允许临时候选，然后最后 repair：

```text
如果 gpu_after > gpu_budget:
    从 accepted promote patches 中，按最低 benefit_per_byte 移除；
    或加入 demote/evict patches，直到 memory feasible。
```

推荐设计上使用混合方式：

```text
hard upper bound:
    绝不超过 runtime 无法承受的 hard budget。

soft budget:
    在 soft budget 内做 repair，以获得更好组合。
```

repair 的排序可以用：

```text
loss_per_byte(v) =
    lost_score(v) / freed_bytes(v)
```

优先移除或 demote `loss_per_byte` 最低的 patch。

## 9. 完整算法

```text
Algorithm: StructuredPlanPatch

Input:
    S_runtime, P_base, B_t, H, profile, online_bound

Output:
    P_adjusted, PatchTrace

1. raw_diff = BuildRawDiff(S_runtime, P_base)

2. G = BuildHierarchicalDiffDAG(raw_diff)

3. frontier = InitializeCoarseNodes(G)
   accepted = {}
   rejected = {}

4. while frontier not empty and online_bound not exhausted:
       v = PopByPriority(frontier)

       Estimate lower_score(v), upper_score(v)

       if Infeasible(v, accepted, B_t):
           Prune(v)
           rejected.add(v)
           continue

       if lower_score(v) > accept_threshold:
           Accept(v)
           accepted.add(v)
           TryContractNeighbors(v, accepted)
           continue

       if upper_score(v) < reject_threshold:
           Prune(v)
           rejected.add(v)
           continue

       if CanExpand(v):
           children = Expand(v)
           frontier.add(children)
       else:
           // leaf but uncertain: use conservative decision
           if score(v) > 0 and Feasible(v):
               accepted.add(v)
           else:
               rejected.add(v)

5. accepted = FeasibilityRepair(accepted, B_t)

6. P_adjusted = ApplyPatches(S_runtime, accepted)

7. return P_adjusted, PatchTrace
```

复杂度可以这样写：

```text
O(M + E + L log L)
```

其中：

```text
M:
    raw diff item 数量，通常小于全部 tensor 数量。

E:
    diff DAG edges 数量。

L:
    实际被展开/访问的 patch nodes 数量，由 online_bound 限制。
```

这比 online CP-SAT 的完整组合搜索轻，因为 `L` 是 bounded 的，而且搜索空间
只来自 `Diff(S_runtime, P_base)`。

## 10. 具体例子

假设当前 budget 从 `B_old = 4096 MiB` 上升到 `B_t = 4608 MiB`。offline
candidate selection 选出 `P_base`，它希望更多 weight 回到 GPU。

当前 runtime state：

```text
S_runtime:
    Layer 12 attention: CPU resident, CPU layout
    Layer 12 FFN:       CPU resident, CPU layout
    GPU used:           4200 MiB
    GPU budget:         4608 MiB
```

`P_base` 的目标：

```text
P_base:
    Layer 12 attention: GPU resident, GPU layout
    Layer 12 FFN:       GPU resident, GPU layout
```

raw diff：

```text
D1: L12.attn.q CPU -> GPU
D2: L12.attn.k CPU -> GPU
D3: L12.attn.v CPU -> GPU
D4: L12.attn.o CPU -> GPU
D5: L12.ffn.gate CPU -> GPU
D6: L12.ffn.up   CPU -> GPU
D7: L12.ffn.down CPU -> GPU
```

构造 coarse DAG：

```text
Layer12Promote
  AttentionPromote
    QKVPromote
      D1, D2, D3
    OPromote
      D4
  FFNPromote
    GateUpPromote
      D5, D6
    DownPromote
      D7
```

假设 profile 给出：

```text
H = 128 tokens

AttentionPromote:
    gain_per_token = 0.35 ms/token
    raw_transition_cost = 18 ms
    overlap_ratio = 0.60
    visible_cost = 7.2 ms
    memory_delta_gpu = 120 MiB
    risk_penalty = 2 ms

FFNPromote:
    gain_per_token = 0.20 ms/token
    raw_transition_cost = 52 ms
    overlap_ratio = 0.30
    visible_cost = 36.4 ms
    memory_delta_gpu = 360 MiB
    risk_penalty = 4 ms
```

计算 score：

```text
score(AttentionPromote)
    = 128 * 0.35 - 7.2 - memory_penalty(120 MiB) - 2
    ~= 44.8 - 7.2 - 1 - 2
    = 34.6 ms

score(FFNPromote)
    = 128 * 0.20 - 36.4 - memory_penalty(360 MiB) - 4
    ~= 25.6 - 36.4 - 10 - 4
    = -24.8 ms
```

决策：

```text
AttentionPromote:
    score 明显为正，accept。

FFNPromote:
    score 明显为负，prune。
```

输出：

```text
P_adjusted:
    Layer 12 attention -> GPU
    Layer 12 FFN       -> keep CPU
```

这说明 structured patch 和 candidate selection 的区别：

```text
candidate selection:
    选中 P_base 后会倾向完整执行 attention + FFN promote。

structured patch:
    只接受 attention promote，拒绝短 horizon 下不划算的 FFN promote。
```

再看一个短 horizon 情况。如果 `H = 16 tokens`：

```text
score(AttentionPromote)
    = 16 * 0.35 - 7.2 - 1 - 2
    = -4.6 ms
```

此时即使 attention promote 稳态有收益，也不值得立即做。planner 会 prune 或
继续 expand 到更细粒度，例如只考虑 QKV：

```text
AttentionPromote
    -> QKVPromote
    -> OPromote
```

如果 QKV 的 visible cost 更低、收益更集中，可能只 accept QKV。这体现了
`expand` 的作用：当 coarse node 不划算或不确定时，寻找更小、更便宜的 patch。

## 11. 另一个例子：budget shrink

如果 budget 从 `4608 MiB` 降到 `4096 MiB`，目标可能不是 promote，而是释放
GPU memory。

当前状态：

```text
S_runtime:
    Layer 20 attention: GPU
    Layer 20 FFN:       GPU
    GPU used:           4520 MiB
    GPU budget:         4096 MiB
```

需要释放至少：

```text
424 MiB + safety margin
```

候选 patch：

```text
Demote Layer20Attention:
    freed_gpu = 140 MiB
    lost_gain_per_token = 0.30 ms/token
    transition_cost = 6 ms

Demote Layer20FFN:
    freed_gpu = 390 MiB
    lost_gain_per_token = 0.18 ms/token
    transition_cost = 12 ms

Evict cold Layer5FFN:
    freed_gpu = 220 MiB
    lost_gain_per_token = 0.05 ms/token
    transition_cost = 4 ms
```

这里 cost function 要反过来：不是 maximize promote gain，而是在满足 memory
释放要求下 minimize future loss：

```text
loss(v) =
    H * lost_gain_per_token(v)
  + visible_transition_cost(v)
  + risk_penalty(v)

loss_per_byte(v) =
    loss(v) / freed_gpu_bytes(v)
```

planner 会优先选择低 `loss_per_byte` 的 demote/evict patch。例如先 evict
cold Layer5FFN，再 demote Layer20FFN，而不是 demote 高收益 attention。

这个例子说明同一个 structured patch framework 可以处理：

```text
budget increase:
    选择值得 promote/materialize 的 patch。

budget shrink:
    选择损失最小的 demote/evict patch。
```

## 12. 论文里怎么讲 novelty

可以把第四点写成三个 design insights。

### Insight 1: Re-planning can be converted to patching

动态 budget change 后，不需要在线重解完整 placement。offline candidate 已经
提供了一个高质量方向，runtime 只需要决定哪些 diff 值得执行。

### Insight 2: Patch value is structured

patch 的 cost/benefit 不是 tensor-independent，而是按 layer/component、
resource sharing、pipeline overlap、memory conflict 组织。因此 structured
diff DAG 是必要的。

### Insight 3: Online search should be uncertainty-driven

planner 不应该展开所有节点，而应该：

```text
明显好的 coarse patch 直接 accept；
明显差的 coarse patch 直接 prune；
只有不确定的区域才 expand。
```

这样 online overhead 被限制在少量 uncertain subgraphs 上。

## 13. 需要记录的日志

为了后面 evaluation 能证明机制有效，需要记录：

```text
budget change event:
    old_budget, new_budget, H

candidate selection:
    selected P_base id
    whole-plan steady cost
    whole-plan transition cost

patch search:
    number of raw diff items
    number of DAG nodes
    expanded node count
    accepted node count
    pruned node count
    contracted node count
    online search time

patch result:
    accepted patch bytes
    visible transition cost
    predicted benefit
    actual transition stall
    actual ms/token after switch
```

这些日志可以支撑几个关键图表：

```text
1. structured patch 比 candidate-select 少做了多少无效 transition。
2. structured patch 的 online overhead 比 CP-SAT 小多少。
3. accepted/pruned nodes 是否符合 cost model 预测。
4. graph expansion 是否只集中在少量 uncertain regions。
```

## 14. 可以直接放进 design section 的表述

第四点的详细设计可以压缩成下面这段：

```text
After selecting an offline candidate, our runtime does not blindly switch to
the entire plan.  Instead, it constructs a hierarchical diff DAG between the
observed runtime state and the selected candidate.  Each DAG node represents a
candidate patch at a semantic granularity, such as a layer, attention block,
FFN block, tensor group, or individual tensor.  Edges encode structural
containment as well as resource sharing, dependency, conflict, and pipeline
overlap.  The runtime scores each patch by comparing horizon-weighted
steady-state benefit against visible transition cost, memory pressure, and
risk.  It then performs a bounded search: confidently beneficial nodes are
accepted, clearly unprofitable nodes are pruned, and only uncertain nodes are
expanded to finer granularity.  The final plan is the current runtime state
plus the accepted patches, repaired for memory feasibility.
```

中文版本：

```text
在选出 offline candidate 后，runtime 不会直接完整切换到该 plan，而是在当前
真实 runtime state 和 candidate 之间构造一个 hierarchical diff DAG。DAG 中
每个节点代表一个语义粒度上的 patch，例如 layer、attention block、FFN
block、tensor group 或单个 tensor；边表示层级包含、资源共享、依赖、冲突和
pipeline overlap。runtime 使用 horizon-weighted benefit、visible transition
cost、memory pressure 和 risk 对 patch 打分，并执行 bounded search：明显有
收益的节点直接接受，明显不划算的节点剪枝，只有不确定节点才展开到更细粒度。
最终输出的 plan 是当前 runtime state 加上 accepted patches，并经过 memory
feasibility repair。
```
