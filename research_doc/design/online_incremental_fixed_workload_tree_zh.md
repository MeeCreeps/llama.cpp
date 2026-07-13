# Fixed Workload Online Incremental Plan: Diff Patch Tree

## Tree Definition

在 fixed model workload 下，online planner 已经得到当前 runtime state `S_t`、当前 budget `B_t`，以及一个由 offline candidates 选出的 base plan `P_base`。

我们不直接 apply `P_base`，而是先构造：

```text
D = Diff(S_t, P_base)
```

`D` 被组织成一组结构化 patch：

```text
P = {p1, p2, ..., pk}
```

每个 patch 表示一个局部 plan change，例如 demotion、promotion、reload、eviction 或 layout drop。

## Tree Node

tree 中每个 node 表示一个 partial adjusted plan：

```text
N_i = S_t + accepted_patches_i
```

也就是说，node 不是完整重新求解出来的 plan，而是在当前 runtime state 上应用了一部分 patch 之后得到的中间状态。

## Tree Edge

每条 edge 是对一个 patch 的二元决策：

```text
+p_i: apply this patch
-p_i: keep current runtime state for this group
```

这样 tree 的一条 root-to-leaf path 就是一组 patch selection。

## Search Strategy

planner 按 patch 的收益和代价排序展开：

```text
priority(p) = memory_relief + compute_benefit - transition_cost - churn_cost
```

搜索中执行 pruning：

- 当前 path 已经超过 best feasible leaf 的 cost。
- 当前 memory 超过 `B_t`，且剩余 patch 已经无法修复。
- patch 之间有 conflict。
- 当前 memory 已经满足 budget，并且剩余 patch 的收益低于切换代价。

## Leaf Selection

每个 leaf 是一个 candidate adjusted plan：

```text
P_leaf = S_t + selected_patches
```

最终选择：

```text
P_adjusted =
argmin cost(P_leaf)
s.t. memory(P_leaf) <= B_t
```

其中：

```text
cost = compute_cost + transition_cost + apply_overhead + churn_penalty
```

## Key Point

这个 tree 的目的不是预测未来，也不是重新完整求解 plan，而是把 whole-plan replacement 转换成 selective patch selection。

最终输出：

```text
P_t = P_adjusted
```

它通常既不等于 `S_t`，也不等于 `P_base`，而是在二者之间选择出的低切换成本可行 plan。
