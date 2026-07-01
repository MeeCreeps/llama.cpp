# Dynamic Memory Budget 下的 Structured Diff 设计

## 1. 问题定义

我们现在面对的问题不是普通的静态 LLM inference placement。手机上的
memory budget 会随着系统环境变化而变化，比如 OS memory pressure、前台
应用、后台任务、thermal/runtime policy 等。因此 planner 不能只回答一个
静态问题：

```text
在 budget B 下，最优 plan 是什么？
```

更准确的问题应该是：

```text
给定当前真实 runtime memory state，当 budget 变成 B_t 时，
系统应该切换到什么 plan，并且现在值得支付多少 adjustment cost？
```

这里的核心 trade-off 是：

```text
budget-optimal target plan
    在当前 budget B_t 下，从稳态推理性能看最好的 plan。

adjustment from current state
    从当前 runtime state 切到这个 target plan 需要支付的 load、
    transform、evict、同步和 pipeline disruption 成本。
```

也就是说，系统不能只追求 target plan 的稳态最优。如果 budget 变化很快，
或者切换成本很大，那么马上切过去可能并不划算。但系统也不能一直保守地
停留在旧状态，因为新的 budget 可能提供了明显的 decode 性能提升空间。

所以我们的 design 应该围绕一句话展开：

```text
在动态 budget 下，planner 要在 target plan quality 和 adaptation cost
之间做在线权衡。
```

## 2. 两个 baseline 为什么不够

### 2.1 Offline budget table

offline baseline 会为不同 memory budget 预先求解 plan。runtime 只需要根据
当前 budget 查表，所以 online overhead 很低。

它的问题是没有考虑当前 runtime memory state。同样是切到 4608 MiB budget，
当前状态可能完全不同：

```text
一些 weight 已经在 GPU resident
一些 weight 在 CPU resident
一些 weight 只有 disk/raw layout
一些 transform 已经做完但 logical plan 不知道
一些 evict 被 deferred
一些 materialization 失败或者被跳过
```

offline plan 只知道 budget，不知道这些状态。因此它可能在静态意义上是
optimal，但从当前状态切过去的实际代价很大。

### 2.2 Online CP-SAT

online baseline 在 budget 变化时在线求解 CP-SAT。这里要特别注意：online
并不是另一个 solver family。offline 和 online 都可以是 CP-SAT；区别在于
online solver 的输入和 objective 可以包含当前 runtime state 和 transition
cost。

所以 online CP-SAT 的优势是 state-aware：它能够根据当前 weight residency
选择更合适的 plan。

但它的问题是求解时间太长。对于手机 runtime 来说，budget change 可能在
inference 过程中发生，planner 的求解会和 decode、memory transfer、系统
负载竞争资源。完整在线求解很难放进关键路径。

### 2.3 我们的 research gap

我们想要的是：

```text
接近 online CP-SAT 的 state-aware plan quality，
但 runtime overhead 接近 offline table lookup。
```

因此设计方向是：

```text
offline 阶段：
    不只保存一个最优 plan，而是保存 top-k diverse candidate plans。

online 阶段：
    利用当前 runtime state 选择和调整 candidate plan，
    避免重新完整求解 CP-SAT。
```

## 3. 必须区分三个状态

这个设计里最容易混淆的是 previous plan 和 current memory state。我们需要
把三个对象分开：

```text
P_prev
    planner 上一次选择的 logical plan。

S_runtime
    runtime 当前真实观测到的 residency/materialization state。

P_target(B_t)
    当前 budget B_t 下的某个 candidate target plan。
```

真正的 transition cost 应该基于：

```text
Diff(S_runtime, P_target)
```

而不是只看：

```text
Diff(P_prev, P_target)
```

原因是 `P_prev` 只是 planner 的上一轮意图，不一定等于真实状态。runtime
可能因为 deferred eviction、foreground miss、materialization failure、
keep-current decision 等原因偏离 logical plan。

因此 `P_prev` 可以作为结构参考，比如帮助判断 plan identity 和上一次的
decision history；但真正算代价时必须锚定 `S_runtime`。

## 4. Offline 阶段：保存 top-k diverse plans

如果每个 budget 只保存一个最优 plan，那么 online 阶段几乎没有选择空间。
因此 offline 阶段应该为每个 budget 保存一组 candidate：

```text
C(B) = {P_1, P_2, ..., P_k}
```

这些 candidate 要满足两个条件：

```text
1. 每个 plan 在 budget B 下本身质量不错。
2. plan 之间 placement pattern 足够多样。
```

多样性很重要。因为动态 budget 下，最好的 plan 不一定是稳态预测最优的
那个，而可能是“从当前 runtime state 切过去最划算”的那个。

candidate diversity 可以来自：

```text
不同 CPU/GPU/disk placement ratio
不同 layer group resident pattern
不同 attention/FFN placement trade-off
objective perturbation
plan 之间的 placement distance constraint
```

offline artifact 至少应该保存：

```text
steady decode cost
CPU/GPU memory footprint
placement bitsets
layout/transform requirements
load/prepare bytes
layer/component coarse summary
```

也可以预计算 plan-to-plan transition summary：

```text
T(P_i -> P_j)
```

但这个只是在 `P_prev` 接近真实 runtime state 时有用。online 阶段仍然需要
能基于 `S_runtime` 重新估计 transition cost。

## 5. Online 阶段第一步：candidate selection

当 budget 变成 `B_t` 时，online planner 先取出对应的候选集合：

```text
C(B_t) = {P_1, P_2, ..., P_k}
```

然后对每个 candidate 打分：

```text
score(P) =
    steady_cost(P)
  + alpha * transition_cost(S_runtime -> P)
  + beta  * memory_pressure_risk(P, B_t)
  + gamma * pipeline_disruption(P)
```

得到一个 base target：

```text
P_base = argmin score(P)
```

这一步已经比 offline single-plan 更强，因为它不是只看 budget，而是看当前
runtime state。但它仍然有一个问题：它选择的是一个完整 candidate plan。

一个 candidate 里面可能有些 diff 很值得做，有些 diff 不值得做。整 plan
切换仍然太粗。因此第四点才是我们设计的关键。

## 6. 第四点核心：Structured Plan Diff

第四点可以这样定义：

```text
不要把 plan diff 看成一组 flat、彼此独立的 weight moves。
把 Diff(S_runtime, P_base) 组织成有结构的 tree/graph，
然后对每个结构化 diff 单元做 accept / reject / expand。
```

planner 首先构造：

```text
D = Diff(S_runtime, P_base)
```

这个 diff 表示：如果要从当前真实 runtime state 走向 `P_base`，需要做哪些
placement、layout、materialization、eviction 改变。

### 6.1 Diff unit

最小粒度的 diff unit 可以是：

```text
tensor placement:
    blk.12.attn_q.weight CPU -> GPU

layout transform:
    raw CPU layout -> GPU transformed layout

materialization:
    disk-only -> CPU staged

eviction:
    GPU resident -> evicted/demoted
```

但是 online planner 不应该一开始就在这个粒度做决策。逐 tensor 决策很容易
丢掉共享成本，比如同一 layer 的 transform、同一 GGUF file locality、同一
pipeline overlap window。

### 6.2 Diff hierarchy

我们可以先用 LLM 的自然结构组织 diff：

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

这样 planner 可以先判断 coarse-grained decision：

```text
是否值得把某个 layer 的 attention 从 CPU promote 到 GPU？
是否值得把某个 FFN block 保留在 CPU 而不是 GPU？
是否应该整体接受某个 layer group 的 demotion？
```

如果 coarse decision 很明确，就不需要继续细分。只有在收益和代价接近时，
才 expand 到更细粒度。

### 6.3 Diff graph

tree 能表达模型层级，但系统代价里还有很多 cross-edge，所以更完整的结构
应该是 graph：

```text
execution locality
    decode graph 中使用时间接近的 tensors。

resource sharing
    共享 disk load、CPU prepare、GPU transform、evict path。

file locality
    在 GGUF 文件中相邻，可能共享顺序读取或 page/cache 行为。

memory coupling
    接受一个 GPU promotion 可能必须 evict 另一个 resident weight。

pipeline overlap
    多个 transform/load 能否被同一个 compute window hide。
```

这里的 research point 是：graph 不是普通实现细节，而是告诉读者为什么
incremental planning 可以比 flat candidate selection 更好。因为 transition
cost 和 steady benefit 本来就是结构化的，不是每个 weight 独立发生的。

## 7. Accept / Reject / Expand

对一个 diff node 或 connected subgraph，planner 估计：

```text
steady_gain(node, H)
    在未来 H 个 token 内的 decode 收益。

transition_cost(node)
    load、prepare、transform、evict、sync 的总成本。

effective_transition_cost(node)
    考虑 pipeline overlap 后真正暴露在关键路径上的成本。

memory_delta(node)
    接受该 diff 后增加或释放的 CPU/GPU memory。

risk(node)
    budget 不稳定、fragmentation、profile 不准等风险。
```

一个简单的 score 是：

```text
score(node) =
    steady_gain(node, H)
  - effective_transition_cost(node)
  - memory_pressure_penalty(node)
  - risk_penalty(node)
```

决策规则：

```text
if score(node) 明显为正，并且 memory feasible:
    accept 整个 node/subgraph

elif score(node) 明显为负:
    reject 整个 node/subgraph，保持当前 runtime state

else:
    expand 到更细的 children/subgraphs
```

这就是第四点最重要的形式：planner 的 online 工作量不是固定扫完整个 weight
list，而是集中在 uncertain region。容易判断的大块 diff 直接接受或拒绝；
只有难判断的局部才展开。

## 8. Horizon-aware trade-off

`H` 表示我们预计当前 budget/state 会持续多少 token。这个量很重要。

如果 budget 可能稳定很久，那么一次昂贵的 transition 可能值得做；如果
budget 很快又会变化，那么 planner 应该更保守。

可以写成：

```text
steady_gain(node, H) =
    H * per_token_gain(node)
```

接受条件可以表达为：

```text
accept if H * per_token_gain > visible_transition_cost + penalties
```

这给 ASPLOS-style system design 一个比较清晰的论点：

```text
我们的 planner 不只是适配当前 budget，还适配这个 budget 的预期生命周期。
```

这比单纯说“我们减少 diff cost”更强，因为它解释了为什么同一个 diff 在
不同 runtime 环境下可能应该做出不同决策。

## 9. 一个例子：不是全盘接受 candidate

假设 `P_base` 希望把某一层的 attention 和 FFN 都从 CPU promote 到 GPU。

whole-plan switch 会把所有变化都执行。flat diff 会逐 tensor 看每个 move。
structured diff 会先看：

```text
Layer 12 CPU -> GPU promotion
```

如果这一层整体收益和成本不确定，再 expand：

```text
Layer 12
  attention promotion
  FFN promotion
```

planner 可能发现：

```text
attention promotion:
    per-token gain 高
    transition cost 可以被 pipeline overlap hide
    memory delta 可接受
    -> accept

FFN promotion:
    memory delta 更大
    transform cost 更高
    在短 horizon 下收益不够
    -> reject 或继续 expand
```

最后得到的不是原来的 `S_runtime`，也不是完整的 `P_base`，而是：

```text
P_adjusted = S_runtime + accepted structured diffs
```

这就是 structured diff 相对 candidate selection 的本质区别。candidate
selection 问的是：

```text
哪个 candidate plan 最适合当前状态？
```

structured diff 进一步问：

```text
这个 candidate 里面哪些部分现在值得付费执行？
```

## 10. Runtime algorithm sketch

```text
input:
    current budget B_t
    runtime state S_runtime
    previous logical plan P_prev
    offline candidates C(B_t)
    expected horizon H

1. 对 C(B_t) 中每个 candidate，基于 S_runtime 打分。
2. 选择 whole-plan score 最好的 P_base。
3. 构造 D = Diff(S_runtime, P_base)。
4. 用 coarse diff nodes 初始化 frontier。
5. 在 online time budget 内循环：
       pop 一个 node
       估计 gain、cost、memory delta、risk
       accept / reject / expand
6. 如果 accepted diffs 超过 memory budget，做 feasibility repair。
7. 输出 P_adjusted，并记录 accepted/rejected/expanded decisions。
```

这里 online time budget 本身也是系统设计的一部分。即使 planner 时间不够，
它也可以应用已经明确接受的大块 diff，其余部分保持当前 runtime state，从而
gracefully degrade。

## 11. Evaluation 应该证明什么

需要比较的 baseline：

```text
offline-single
    每个 budget 一个预计算 plan，不看当前状态。

online-cpsat
    完整 state-aware 在线求解，质量高但 overhead 大。

candidate-select
    top-k offline candidates + runtime transition scoring。

structured-diff
    candidate selection + structured accept/reject/expand。
```

需要看的指标：

```text
decode ms/token
planner decision time
transition load/prepare/evict bytes
visible transition stall
accepted/rejected/expanded node 数量
和 online CP-SAT plan quality 的距离
memory budget violation 次数
budget-change recovery time
```

我们希望最后能形成的 claim：

```text
Structured-diff planning 保留了 online CP-SAT 的大部分 state-aware 质量，
但把 budget switch 时的在线开销降低为 bounded candidate scoring 和
selective graph refinement。
```

## 12. 现在最值得继续细化的问题

1. 初始结构应该怎么选：layer-first、component-first，还是 profile-derived
   cluster？
2. `H` 怎么估计：用最近 budget 稳定时间、OS signal，还是 trace predictor？
3. top-k candidate 的 diversity 应该按 placement distance、transition
   behavior，还是 semantic component pattern 来定义？
4. graph structure 需要做到多复杂，才能明显超过 whole-plan candidate
   selection？
5. accept/reject threshold 是手动调参，还是可以从手机 trace/profile 自动学？

## 13. 当前工作假设

当前最有希望的 design 是：

```text
offline:
    为每个 budget 保存 top-k diverse candidate plans 和 compact summaries。

online:
    用 S_runtime-aware transition cost 选择一个 base target candidate。

incremental:
    把 Diff(S_runtime, P_base) 组织成 hierarchy/graph。
    对明显有收益的 subgraph 直接 accept。
    对明显不划算的 subgraph 直接 reject。
    对 uncertain subgraph 继续 expand。
    在 online planning budget 内得到 P_adjusted。
```

这套方法直接对应我们的核心 trade-off：当 target plan 的收益足够大时，系统
愿意向它移动；当当前 runtime state 已经足够好，或者 transition cost 在短
horizon 下不划算时，系统避免不必要的调整。
