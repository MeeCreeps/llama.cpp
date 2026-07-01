# 第四点设计讨论：Structured Plan Patch

## 1. 第四点到底要解决什么

前三点做完以后，系统已经有了：

```text
offline:
    每个 budget 下的 top-k candidate plans

online:
    根据 S_runtime 选择一个最相关的 candidate P_base
```

但这还不够。因为 `P_base` 是一个完整 plan，直接切到它仍然可能太粗：

```text
有些 diff 很值得做：
    比如某些 attention weight promote 到 GPU 后 decode 收益明显。

有些 diff 不值得做：
    比如短 horizon 下的大量 FFN transform，成本比收益大。

有些 diff 只有组合起来才有意义：
    单个 tensor move 看起来不划算，但整个 attention group 共享 load/transform
    后是划算的。

有些 diff 会互相冲突：
    接受一个 GPU promotion 可能要求 evict 另一个 resident region。
```

所以第四点的核心任务不是再选一个 plan，而是：

```text
在已经选出的 P_base 上，做 lightweight partial modification，
只接受当前状态下值得支付的 plan diff。
```

可以把这一步命名成：

```text
Structured Plan Patch
```

也就是从 `S_runtime` 出发，只把 `P_base` 里面一部分有价值的 diff patch 到
当前状态上：

```text
P_adjusted = S_runtime + accepted_patch(Diff(S_runtime, P_base))
```

这一步的出发点是：

```text
whole-plan candidate selection 仍然太粗；
full online CP-SAT 又太重；
所以需要一个 bounded、结构化、可解释的中间层。
```

## 2. 第四点的 research claim

第四点可以承担 paper 里的主要 novelty：

```text
Dynamic memory changes do not require solving a full placement problem online.
They can be handled as a bounded structured patch over an offline candidate,
where the patch is selected according to state-aware transition cost and
horizon-aware steady-state benefit.
```

换成中文就是：

```text
动态 memory budget 下，在线 planner 不需要从零求一个完整 plan。
它只需要在 offline candidate 和当前 runtime state 之间，选择一组最值得
执行的结构化 patch。
```

这个 claim 比“我们用了 graph/tree”更强。graph/tree 只是实现这个 claim 的
方法，真正的贡献是：

```text
把 online re-planning 转化为 bounded structured patch selection。
```

## 3. 为什么不能 flat diff

最直接的方法是列出所有 changed tensors：

```text
blk.10.attn_q CPU -> GPU
blk.10.attn_k CPU -> GPU
blk.10.attn_v CPU -> GPU
blk.10.ffn_up CPU -> GPU
...
```

然后逐个决定做不做。但 flat diff 有三个问题。

### 3.1 成本不是独立的

多个 tensor 可能共享同一次 disk read、CPU staging、GPU transform、pipeline
overlap window。单独看每个 tensor，会高估或低估真实成本。

### 3.2 收益不是独立的

LLM 的计算结构是 layer/component 级别的。比如 attention 里的 q/k/v/o 常常
应该作为一个 group 讨论，FFN 的 gate/up/down 也有组合关系。单个 tensor 的
per-token gain 可能不稳定，但整个 component 的收益更有解释力。

### 3.3 memory feasibility 有耦合

接受一个 patch 可能导致 GPU memory 超出 budget，需要同时接受另一个 evict
或 demote patch。因此 patch decision 本身存在 coupling。

所以第四点需要把 diff 组织起来，而不是当成无结构 list。

## 4. 推荐的数据结构：Hierarchical Diff DAG

我建议不要把它只叫 tree，也不要一开始就叫一般 graph。更合适的说法是：

```text
Hierarchical Diff DAG
```

原因：

```text
hierarchical:
    LLM 本身有天然层级，model -> layer group -> layer -> component -> tensor。

DAG:
    除了层级父子关系，还需要表达 resource sharing、memory conflict、
    pipeline overlap 等 cross edges。

diff:
    这个结构不是完整 execution graph，而是只覆盖 S_runtime 到 P_base 的变化。
```

节点可以分几类：

```text
keep_identity
    当前状态和 target 一致，不需要动作，但可以作为边界和上下文。

promote
    CPU/disk -> GPU，通常带来 decode gain，但消耗 GPU memory 和 transform cost。

demote
    GPU -> CPU/disk，释放 memory，但可能增加 future compute/load cost。

materialize
    disk/raw -> CPU staged/transformed。

transform
    CPU raw/layout -> backend-specific layout。

evict
    release resident memory，通常作为 memory repair 或 budget shrink 动作。
```

边可以分几类：

```text
parent-child edge
    layer group 包含 layer，layer 包含 attention/FFN，component 包含 tensor。

resource-sharing edge
    多个 patch 共享 load/prepare/transform path。

conflict edge
    两个 patch 不能同时接受，或接受一个必须拒绝另一个。

dependency edge
    某个 GPU compute patch 依赖 CPU materialization 或 layout transform。

overlap edge
    多个 transfer/transform 可以被相同 compute window hide。
```

这个结构的重点是：在线阶段只在 diff 上建图，不在完整模型上做全局优化。
所以图规模天然小于 full planning problem。

## 5. Patch node 需要记录什么

每个 node/subgraph 至少需要这些属性：

```text
steady_gain_ms_per_token
    接受这个 patch 后每 token 预计能省多少时间。

transition_cost_ms
    执行 patch 需要的 load/prepare/transform/evict/sync 成本。

visible_cost_ms
    考虑 pipeline overlap 后，真正暴露在关键路径上的成本。

memory_delta_cpu / memory_delta_gpu
    接受 patch 后 CPU/GPU memory 的变化。

risk
    profile 不确定性、budget 波动、fragmentation、失败重试等风险。

children
    如果这个 node 不确定，能 expand 成哪些更细粒度节点。

constraints
    dependency/conflict/feasibility 条件。
```

核心 score 可以先写成：

```text
score(v) =
    H * steady_gain_ms_per_token(v)
  - visible_cost_ms(v)
  - memory_pressure_penalty(v)
  - risk_penalty(v)
```

其中 `H` 是当前 budget/state 的 expected horizon。它让第四点变成
horizon-aware，而不是只看单次 transition cost。

## 6. 三个核心操作：Expand / Contract / Prune

你提到的 expand、收缩、剪枝是很好的方向。我建议把第四点的 online search
描述成三个操作。

### 6.1 Expand：不确定时细化

当一个 coarse node 的收益和成本接近，或者内部 variance 很大时，展开它：

```text
Layer 12 promotion
    -> Attention promotion
    -> FFN promotion

Attention promotion
    -> QKV group
    -> O projection
```

触发 expand 的条件可以是：

```text
score 接近 0
children 的 cost/gain variance 大
memory_delta 太大，一次接受风险高
profile confidence 低
```

research 上的解释：

```text
planner 只在 ambiguous region 花更多在线计算。
```

### 6.2 Contract：确定时合并

如果一组 child nodes 共享资源，而且方向一致，就把它们合并成一个 patch：

```text
Q + K + V promote
    -> QKV promote patch
```

contract 的条件可以是：

```text
children 都明显 positive
children 共享 load/transform/overlap cost
合并后 memory feasible
```

这一步很重要，因为它防止 planner 退化成逐 tensor 操作。它也能表达系统里
真实存在的共享成本。

### 6.3 Prune：明显不划算时剪掉

如果某个 node/subgraph 明显不值得做，就直接剪掉：

```text
H * gain << visible transition cost
```

或者：

```text
memory_delta 导致不可行
risk 太高
依赖条件无法满足
```

prune 后，该区域保持 `S_runtime` 的当前 placement。

这一步是 lightweight 的关键：planner 不需要完整探索所有 diff，只保留有
希望的 patch frontier。

## 7. 在线算法可以怎么讲

可以把第四点写成一个 bounded best-first patch search：

```text
input:
    S_runtime
    P_base
    budget B_t
    horizon H
    online_time_budget

1. 构造 D = Diff(S_runtime, P_base)。
2. 把 D 聚合成 coarse patch nodes，初始化 frontier。
3. 对 frontier 中的 node 计算 optimistic score 和 conservative score。
4. 循环直到 frontier 为空或 online_time_budget 用完：
       取出最有希望或最不确定的 node。
       如果明显 positive:
           accept，并尝试 contract 相关 positive nodes。
       如果明显 negative/infeasible:
           prune。
       否则:
           expand 成更细 nodes。
5. 对 accepted patches 做 memory feasibility repair。
6. 输出 P_adjusted。
```

这里可以引入两个 bound：

```text
optimistic_score
    假设 overlap 更好、收益更高，用来判断是否值得继续探索。

conservative_score
    假设 overlap 较差、风险较高，用来判断是否可以直接接受。
```

决策可以写成：

```text
if conservative_score(v) > accept_threshold:
    accept(v)

elif optimistic_score(v) < reject_threshold:
    prune(v)

else:
    expand(v)
```

这个形式很适合讲 “bounded lightweight search”，因为它不是全局 CP-SAT，而是
带上下界的局部 patch refinement。

## 8. 为什么这个设计讲得通

这个设计有几个比较 solid 的系统理由。

### 8.1 它利用了 offline 的重计算

offline 已经帮我们找到了 budget 下的好 candidate。online 不再搜索完整
placement 空间，只在 `S_runtime` 和 `P_base` 的差异上做选择。

### 8.2 它利用了 LLM 的结构

LLM placement 不是任意图问题。layer、attention、FFN、tensor group 都有
稳定语义。structured patch 可以用这些语义降低搜索空间，并且让 decision
可解释。

### 8.3 它适配 runtime state

transition cost 基于 `S_runtime`，所以它能处理 previous plan 和真实 memory
state 不一致的问题。

### 8.4 它是 horizon-aware

同一个 patch 在长 horizon 下可能值得做，在短 horizon 下可能不值得做。
这正好对应 dynamic memory budget 的核心 runtime 特征。

### 8.5 它有 bounded overhead

online search 有 time budget 和 frontier budget。即使没有探索完整 diff，
也能输出当前已接受的 patch，并保持剩余区域不变。

## 9. 一个可以放进 paper 的例子

假设 budget 从高变低后又回升。offline candidate `P_base` 想把某些 attention
和 FFN 从 CPU promote 回 GPU。

flat switch 会全部执行：

```text
attention promote + FFN promote + layout transform + evict repair
```

structured patch 会先看 coarse layer node：

```text
Layer 18 promote patch
```

如果整体不确定，expand：

```text
Layer 18
  Attention promote
  FFN promote
```

然后发现：

```text
Attention promote:
    gain 高
    transform bytes 小
    overlap ratio 高
    -> accept

FFN promote:
    memory_delta 大
    transform bytes 高
    当前 horizon 短
    -> prune
```

结果：

```text
P_adjusted 接受 attention patch，但保留 FFN 当前状态。
```

这个例子能很好说明第四点不是简单地“找最近 plan”，而是：

```text
选择性吸收 target plan 中值得支付的部分。
```

## 10. 和 baseline 的区别

### Offline-single

```text
只看 budget，不看 S_runtime。
```

### Online CP-SAT

```text
看 S_runtime，但在线完整求解，overhead 高。
```

### Candidate-select

```text
看 S_runtime，选择整个 P_base，但不能部分修改 candidate。
```

### Structured Plan Patch

```text
看 S_runtime，选择 P_base，然后只执行 P_base 中值得执行的 structured diff。
```

这就是第四点可以成为主要 contribution 的原因。

## 11. 目前我建议的最终表述

可以把第四点最终写成：

```text
State-aware Structured Plan Patching.

Given the selected offline candidate, the runtime planner constructs a
hierarchical diff DAG between the observed runtime state and the candidate.
It then performs a bounded patch search that accepts, contracts, expands, or
prunes diff subgraphs according to horizon-weighted benefit, visible transition
cost, memory feasibility, and risk.  The output is an adjusted plan that
partially adopts the candidate only where the expected benefit justifies the
adaptation cost.
```

中文理解是：

```text
第四点不是重新规划，而是在当前状态和候选 plan 之间做一个有界、结构化、
可解释的 patch selection。
```

## 12. 还需要继续讨论的设计选择

1. patch graph 是先用 model hierarchy 建，再加 resource/conflict edge；
   还是直接根据 profile clustering 建？
2. `contract` 是否需要作为显式操作，还是只作为 accept parent node 的自然
   结果？
3. score 里 `H` 怎么估计，是否需要把 budget 变化 trace 做成 predictor？
4. memory feasibility repair 是作为 patch search 的约束，还是最后单独修复？
5. graph search 的 online bound 应该是 time budget、node budget，还是两者
   都有？
6. evaluation 里怎样证明 graph/patch 结构本身有用，而不是 top-k candidate
   已经足够？
