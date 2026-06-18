# Dynamic Budget Planner: PPT Outline

这份稿子面向偏学术的汇报，重点解释 planner 的问题建模和 CP-SAT 设计，而不是代码实现。可以直接压缩成 10 页短汇报，也可以展开成 15 页组会版。

## Slide 1: 标题

**Dynamic Memory Budget Planner for Mobile LLM Inference**

核心信息：

- 目标：在移动端可用内存动态变化时，自动决定哪些权重常驻、哪些按需从 disk 加载，以及每个算子在哪个 backend 执行。
- 方法：用实测 cost model + CP-SAT/knapsack-style optimization 生成执行计划。
- 对比：static CPU、MRU cache、offline table、online state-aware CP-SAT。

建议图示：

```text
memory trace  ->  planner  ->  ExecPlan  ->  elastic runtime  ->  decode latency
```

讲稿：

> 这个工作关注的不是单次 offload，而是动态内存环境下的持续 decode。手机上可用内存会随系统状态波动，因此 planner 需要不断把模型权重在 disk、CPU、GPU 之间重新布局，并尽量降低 decode latency。

## Slide 2: Motivation

**为什么需要 planner？**

核心信息：

- 大模型权重超过手机可稳定使用的内存预算。
- 可用 memory budget 是 time-varying 的，不是固定常数。
- 简单策略有明显缺陷：
  - 全 CPU：安全但慢。
  - 固定 offline plan：查询快，但不知道当前哪些权重已经 resident。
  - MRU/LRU cache：开销低，但不理解 layer cost 和 reload cost。

建议图示：

```text
available memory
^
|        high budget
|   /\      /\       /\
|__/  \____/  \_____/  \____ time
| low budget: model cannot fully fit
```

讲稿：

> 如果 memory budget 足够大，所有权重都放在快的 backend 上即可。但真正困难的情况是 budget 低于模型部分权重大小，planner 必须选择保留哪些权重，以及未来需要付出多少 disk load、transfer、xform 的代价。

## Slide 3: Problem Definition

**动态 budget 下的 planning problem**

输入：

```text
W: weight tensors
O: compute operators
B_t: current available memory budget at time t
S_t: current residency state of each weight
C: measured cost tables
M_kv, M_misc, M_safety: reserved memory
```

可用于权重的预算：

```text
B_weight(t) = B_t - M_kv - M_misc - M_safety
```

输出：

```text
P_t = {weight locations, backend assignment, movement timeline}
```

讲稿：

> Planner 的输入不是只有 budget，还包括当前状态 S_t。S_t 记录每个 weight 当前是否已经在 CPU/GPU、是否已经完成 xform、是否还有 disk backing。online planner 的优势正来自于它利用了这个状态。

## Slide 4: System Overview

**从 profiling 到 runtime apply**

建议图示：

```text
        offline profiling
              |
              v
   stage costs / op costs / model meta
              |
memory trace + residency state
              |
              v
       CP-SAT planner
              |
              v
           ExecPlan
              |
              v
 elastic runtime: load -> transfer -> xform -> compute
```

核心信息：

- Cost model 来自目标设备实测，而不是假设带宽。
- Planner 输出统一的 ExecPlan。
- Runtime 只负责执行 plan，不在执行路径里重新做复杂优化。

讲稿：

> 学术上可以把系统分成两个层次：第一层是 measured cost model，估计每个 movement 和 compute 的代价；第二层是 constrained optimization，根据当前 budget 和 state 选择最优或近似最优的 plan。

## Slide 5: Plan Abstraction

**ExecPlan 把 planning 和 runtime 解耦**

每个 weight 的 plan：

```text
location[w] in {DISK, CPU, GPU}
xform[w]    in {NONE, CPU_REPACK, GPU_CONVERT}
```

每个 op 的 plan：

```text
backend[o] in {CPU, GPU}
```

每个 movement event：

```text
LOAD(w):     disk -> CPU staging
TRANSFER(w): CPU -> GPU
XFORM(w):    raw layout -> backend layout
EVICT(w):    resident -> disk-only state
```

讲稿：

> 这个抽象很重要，因为 planner 不需要关心具体 runtime API；它只输出目标 residency、算子 backend 和一组带 anchor 的 movement event。runtime 可以用同一套接口执行 offline、online、MRU 等不同策略产生的 plan。

## Slide 6: CP-SAT Variables

**决策变量：保留谁、放哪里、怎么执行**

完整问题中的变量：

```text
x_cpu[w]  in {0,1}: weight w resident on CPU
x_gpu[w]  in {0,1}: weight w resident on GPU
x_disk[w] in {0,1}: weight w only backed by disk

y_cpu[o]  in {0,1}: op o runs on CPU
y_gpu[o]  in {0,1}: op o runs on GPU

l[w]      in {0,1}: need disk load
t[w]      in {0,1}: need CPU-to-GPU transfer
f[w]      in {0,1}: need backend-specific transform
e[w]      in {0,1}: need eviction
```

当前落地实现的简化：

```text
x[w] in {0,1}: keep weight w resident under current budget
```

讲稿：

> 完整模型可以同时选择 location、backend 和 timeline。当前实现为了先跑通真实手机实验，把核心 CP-SAT 收缩成 resident-set selection，也就是一个带实测价值函数的 0/1 budget problem。这样可以稳定接入 online runtime，同时保留向完整 CP 扩展的空间。

## Slide 7: Constraints

**约束：内存容量 + 可执行性 + 状态一致性**

内存约束：

```text
sum_w size[w] * x[w] <= B_weight(t)
```

完整模型中的位置约束：

```text
x_cpu[w] + x_gpu[w] + x_disk[w] = 1
```

backend 可执行性：

```text
y_gpu[o] = 1 -> x_gpu[weight(o)] = 1 before op o
y_cpu[o] = 1 -> x_cpu[weight(o)] = 1 before op o
```

online state-aware movement：

```text
l[w] = 1 if target resident but S_t[w] has no usable resident copy
e[w] = 1 if S_t[w] resident but planner chooses not to keep it
```

不可丢弃约束：

```text
if S_t[w] is resident and has no disk backing:
    x[w] = 1
```

讲稿：

> 这类问题的核心不是只有容量约束，还包括状态一致性。比如一个 tensor 已经 resident 但没有可靠 disk backing，就不能被 planner 随意 evict；否则 runtime 会得到一个理论上合法但实际不可执行的 plan。

## Slide 8: Objective Function

**目标：最小化下一阶段 decode 的预测代价**

完整形式：

```text
minimize
    compute_cost
  + movement_cost
  + backend_switch_cost
  + churn_penalty
```

其中：

```text
compute_cost  = sum_o C_compute[o, backend[o]]
movement_cost = sum_w (
    C_load[w] * l[w]
  + C_transfer[w] * t[w]
  + C_xform[w] * f[w]
)
```

当前 CP-SAT resident-set 形式：

```text
maximize sum_w value[w] * x[w]
subject to sum_w size[w] * x[w] <= B_weight(t)
```

value[w] 的含义：

- 保留 w 能避免未来 disk LOAD / TRANSFER / XFORM。
- 如果 w 已经 resident，保留它还能避免 unnecessary churn。
- 如果 w 对 decode compute 更关键，其 value 更高。

讲稿：

> 当前 solver 的形式看起来像 knapsack，但 value 不是人工指定的重要性，而是由实测 cost table 和当前 residency state 派生出来的。也就是说，保留某个 weight 的收益等于避免它将来重新加载和转换所节省的时间。

## Slide 9: Why CP-SAT?

**为什么不用单纯 greedy？**

核心信息：

- Greedy 按 value/size 排序，开销低但只看局部比值。
- CP-SAT 能处理：
  - binary residency variables
  - hard capacity constraints
  - forced keep / pinned tensors
  - future extension to precedence and optional intervals
  - time limit 下返回 feasible solution

对比：

```text
Greedy:
  fast, simple, locally optimal

CP-SAT:
  constrained global optimization,
  extensible to richer scheduling constraints
```

讲稿：

> 现在这个问题的一阶版本确实接近 knapsack，但我们选择 CP-SAT 是因为后面要加入 anchor、load/transfer/xform precedence、engine overlap 和 optional interval。这些都是 CP-SAT 比普通 greedy 更自然的地方。

## Slide 10: Offline Table Baseline

**offline-table：预先为 budget bands 求 plan**

流程：

```text
for B in budget_bands:
    P_B = solve(B, canonical_state)

runtime:
    observe B_t
    choose nearest feasible band
    apply P_B
```

优点：

- runtime overhead 很低。
- 没有在线求解依赖。

缺点：

- 不知道真实 S_t。
- budget band 量化会损失可用内存。
- 可能重复 load 已经 resident 的 weight，或 evict 刚刚转换好的 weight。

讲稿：

> offline-table 的核心问题是 state-blind。它只知道 budget，不知道当前手机上哪些权重已经在 CPU/GPU。动态 trace 下，budget 在多个档位之间来回变化，state-blind plan 很容易产生不必要的数据移动。

## Slide 11: Online CP-SAT Planner

**online：每次 budget 变化时利用当前 state 求解**

流程：

```text
runtime observes budget change
runtime dumps residency state S_t
server/host solves CP-SAT with (B_t, S_t)
runtime applies returned ExecPlan
```

为什么可行：

- 手机上没有 Python/OR-Tools 时，可以 remote solve。
- evaluation 可以只统计 plan execution time，不把 solver wall time 算进 decode execution。
- solver timeout 时可以 fallback 到 greedy/offline plan。

为什么更好：

- 精确使用当前 budget，而不是固定 band。
- 避免对 resident weights 做重复 load/xform。
- 在低 budget 时选择最有价值的 resident subset。

讲稿：

> online 版本本质上把 planning 从静态查表变成 closed-loop control：每次 memory trace 改变，planner 都根据当前状态重新优化。为了避免手机上 Python 依赖，我们可以让 server 跑 CP-SAT，手机只接收 plan 并执行。

## Slide 12: Runtime Execution Model

**movement 是真实执行成本，不只是 plan 里的数字**

Stage pipeline：

```text
DISK LOAD      : model file -> CPU staging buffer
TRANSFER       : CPU staging -> OpenCL/device-visible buffer
XFORM          : raw quant layout -> backend compute layout
COMPUTE        : matmul / op execution
RELOAD_ENSURE  : runtime checks and restores required residency
```

关键 counters：

- eval ms/token
- online gen_ms
- provider_get_ms
- apply_ms
- stage load / transfer / xform ms
- direct disk read ms
- budget switch count
- failures / fallbacks

讲稿：

> 实验必须确认 LOAD/TRANSFER/XFORM 真的发生。否则 planner 看似在调度 disk movement，但实际 runtime 可能仍把完整模型保留在普通 OpenCL buffer 里，测出来的 online/offline 差异就不可信。

## Slide 13: Baselines

**四个 baseline 的学术定位**

Static min CPU：

- 按 trace 最低 budget 配置。
- 全部 CPU 执行。
- 保守、稳定，但通常最慢。

MRU cache：

- 按最近使用顺序移入移出。
- 不需要 cost model。
- 不能识别某些 weight 的 reload cost 更高。

Offline table：

- budget-aware but state-blind。
- runtime 查询非常快。

Online CP-SAT：

- budget-aware and state-aware。
- 计划质量最好，但有 solver overhead。

建议表格：

```text
method          budget-aware   state-aware   solver overhead   expected speed
Static CPU      no             no            none              low
MRU             yes            weak          none              medium
Offline table   yes            no            offline only      medium
Online CP-SAT   yes            yes           online/remote     high
```

讲稿：

> 这几个 baseline 不是随便选的。它们分别代表安全但慢、启发式缓存、离线优化和在线优化四个层次。我们希望证明 state-aware optimization 在 dynamic trace 下比 state-blind table 更适合。

## Slide 14: Experimental Methodology

**怎样证明 planner 真的有效？**

Trace 要求：

- 使用 dynamic memory trace，而不是单一固定 budget。
- trace 中必须包含低于模型部分权重可驻留容量的区间。
- 多条 trace 覆盖不同波动模式和低 budget 持续时间。

运行设置：

- 同一个 model、prompt、decode length。
- 同一个 cost model 和 model meta。
- 对比 online/offline/MRU/static CPU。
- 记录平均 decode speed，也记录 stage breakdown。

结果表建议：

```text
trace     min/max budget   method      avg ms/token   LOAD ms   XFER ms   XFORM ms   switches
trace A   ...              offline     ...
trace A   ...              online      ...
trace B   ...              offline     ...
trace B   ...              online      ...
```

讲稿：

> 只看平均 ms/token 不够，因为平均值可能掩盖 planner 是否真的触发 disk load。必须同时给 stage breakdown 和 budget switch count，确认 online 的优势来自减少真实 movement，而不是实验配置差异。

## Slide 15: Validation Lesson

**实验可信性的关键：确认 elastic backend 真正启用**

已发现的问题：

- 如果没有启用 OpenCL elastic mode，plan 中可能有 LOAD/TRANSFER/XFORM event，但 runtime 实际没有从 disk 重新加载权重。
- 这种情况下 online/offline 的 decode speed 接近，不能作为 disk-load baseline 结论。

修正后的检查：

```text
GGML_OPENCL_ELASTIC=1
GGML_ELASTIC_TIMING=1
GGML_ELASTIC_STAGE_DETAIL=1
```

需要看到：

```text
LOAD / TRANSFER / XFORM counters > 0
direct disk read counters > 0
anchor summary fired > 0
```

讲稿：

> 这是很重要的 negative result。之前一些 69 ms/token 左右的结果没有真实 disk load，因此只能说明 plan provider 的开销，而不能说明 elastic movement 的性能。启用 OpenCL elastic 后，低 budget case 中 LOAD/TRANSFER/XFORM 会实际出现，latency 也会显著上升，这才是可信实验。

## Slide 16: Expected Result Pattern

**我们希望看到什么现象？**

理论预期：

- Static min CPU 最慢，因为它完全按最差 budget 保守执行。
- MRU 在短期局部性好时可用，但在重要 weight 被误 evict 时会退化。
- Offline table 在 budget 来回变化时会产生多余 movement。
- Online CP-SAT 利用 S_t 减少 reload/xform，平均 decode speed 应更好。

可展示的结论句：

> Under dynamic memory traces, the online planner improves decode throughput by converting memory-budget adaptation into a state-aware optimization problem, while offline table lookup remains state-blind and may pay unnecessary movement costs.

讲稿：

> 这页可以放最终实验结果。如果 online 和 offline 的差距不明显，就要回到 stage breakdown 检查：是不是 trace 不够动态、budget 太高、offline band 太密，或者 disk load 没有真正触发。

## Slide 17: Limitations and Next Steps

**当前模型的边界**

限制：

- 当前 CP-SAT 主要解决 resident-set selection，还不是完整 timeline scheduling。
- solver latency 仍是问题，因此需要 remote solve、cache、debounce 和 fallback。
- cost model 对设备和 backend 绑定，换手机需要重新 profiling。
- budget trace 的 representativeness 会影响结论。

下一步：

- 加入 engine-aware overlap model：

```text
predicted_ms = max(disk_time, transfer_time, cpu_time, gpu_time)
```

- 引入 optional interval 和 precedence，显式建模 prefetch window。
- 用 previous solution 做 warm start / plan cache。
- 把 solver time 与 execution time 分开报告。

讲稿：

> 这个方向后续最自然的扩展是从 knapsack-style resident selection 走向真正的 CP scheduling，把 disk、transfer、CPU、GPU 看成不同资源，并建模它们之间的 overlap。

## Optional Slide: Mathematical Formulation

如果需要更学术，可以单独放这一页。

Sets：

```text
W: weights
O: operators
B = {CPU, GPU}: compute backends
```

Variables：

```text
x_w in {0,1}
y_o,b in {0,1}
m_w,k in {0,1}, k in {load, transfer, xform, evict}
```

Constraints：

```text
sum_w size_w x_w <= B_weight

sum_b y_o,b = 1

y_o,GPU <= x_weight(o),GPU
y_o,CPU <= x_weight(o),CPU
```

Objective：

```text
min sum_o sum_b C_o,b y_o,b
  + sum_w sum_k C_w,k m_w,k
  + lambda * churn(S_t, x)
```

Current resident-set CP-SAT：

```text
max sum_w v_w x_w
s.t. sum_w size_w x_w <= B_weight
```

where:

```text
v_w = avoided_reload_cost(w, S_t)
    + compute_value(w)
    + resident_stability_bonus(w, S_t)
```

## Optional Slide: One-Sentence Takeaways

- Planner 把 mobile LLM memory adaptation 建模为 constrained optimization。
- Offline table 是 budget-aware，但不是 state-aware。
- Online CP-SAT 同时使用当前 budget 和 residency state。
- 真正可信的实验必须验证 disk LOAD / TRANSFER / XFORM 被触发。
- 当前实现是 resident-set CP-SAT，后续可扩展成 engine-aware CP scheduling。
