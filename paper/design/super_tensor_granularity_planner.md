# Budget-Aware Super-Tensor Planning

> PPT 可直接复制版本（无 LaTeX）：
> `paper/design/super_tensor_granularity_planner_ppt.md`

## 1. Design objective

端侧 LLM 的可用 weight memory budget 会随相机、浏览器和后台应用动态变化。
当更多 weights non-resident 时，I/O、layout preparation 和 pipeline waiting
占比上升；当大部分 weights resident 时，working-unit submission、kernel 和
bookkeeping overhead 的占比上升。因此，固定使用 Multi、Tensor 或 Cut
都不能在所有 budget 下最优。

本设计将 granularity 纳入 Elastic execution plan。系统不再为全模型选择一个
全局 mode，而是为不同 weights 选择 mixed working units，并随 memory budget
进行局部 split/merge。

论文中的核心表述为：

> We represent legal weight partitions as a Super-Tensor tree and select a
> mixed working-unit plan that minimizes the resource-constrained pipeline
> critical path under the current memory budget.

当前研究范围为 CPU Elastic-only 和 GPU/OpenCL Elastic-only；两者使用相同
抽象和 planner，但分别校准 cost profile。CPU+GPU heterogeneous placement
不属于本阶段。

## 2. Motivation translated into design requirements

已有测量给出三个直接要求。

1. 颗粒度影响的不只是单阶段吞吐，还会改变 pipeline overlap。CPU 30% fixed
   budget 下，Multi/Tensor/Cut 每个 decode 实际读取约 1.50 GiB weights，
   但 exposed pipeline wait 分别为 672.529、620.199 和 469.952 ms。因此
   planner 必须优化 pipeline critical path，而不是简单相加 I/O、Layout 和
   Compute。

2. 颗粒度收益依赖 budget。non-resident work 较多时，较小 unit 可以更早
   ready；non-resident work 较少时，request、preparation 和 dispatch overhead
   更容易成为主导。因此不能使用一个全局 granularity，也不能只依赖
   instantaneous budget 的单阈值。

3. granularity change 本身有成本。Split/merge 可能触发 layout regroup、
   pipeline drain、residency adjustment 和 metadata update。系统需要对稳定态
   收益与 transition cost 做 horizon amortization，并使用 hysteresis 防止
   budget 抖动造成反复切换。

对应数据位于：

- `exp-results/important_results/motivation/granularity/cpu_fixed_30pct_pipeline_breakdown_op13.csv`
- `exp-results/important_results/motivation/granularity/cpu_fixed_90pct_pipeline_breakdown_op13.csv`
- `exp-results/important_results/motivation/granularity/gpu_8b_fixed_90pct_pipeline_breakdown_op13.csv`

## 3. Super-Tensor abstraction

### 3.1 What the tree represents

Super-Tensor tree 不是模型计算图，也不是 pipeline dependency graph。它是同一组
weights 的合法 working-unit partitions 所形成的层次化决策空间。

```text
                         Multi(A, B)
                         /         \
                   Tensor(A)     Tensor(B)
                    /    \        /    \
                  Cut A0 Cut A1 Cut B0 Cut B1
```

- 选择 Multi parent：多个相邻 tensors 共享一个 working-unit lifecycle；
- 选择 Tensor node：一个完整 tensor 是一个 unit；
- 选择 Cut children：一个 tensor 的 row ranges 是独立 units；
- 选择不同子树的不同层级：形成 mixed granularity。

树有四个作用：

1. 统一表示 Multi、Tensor 和 Cut；
2. 在构建时排除不合法的 split/merge；
3. 允许 per-tensor mixed granularity；
4. 将 online re-planning 限制为局部 parent/children edits。

当前代码使用隐式三层 interval forest，而不是保存一个通用 tree object：
`Decision(kind, weight_ids)` 表示当前选择的 node。当前版本支持二路 Cut 和相邻
两 tensor Multi；后续可以将相邻 Multi nodes 继续组成更高层 balanced tree。

### 3.2 Ordered weight-use stream

系统从 model metadata 和真实 graph execution order 中提取：

```text
(operator, weight, first_use, last_use, backend, layout)
```

当前实现位于 `runtime/plan/super_tensor_planner.py::normalize_weights()`：

- 从 `model_meta["ops"]` 获取 weight-use order；
- 从 placement plan 获取 CPU/GPU/Disk location；
- 按 operator order、weight id 形成稳定序列；
- 为每个 weight 记录 bytes、shape、quantization、layer、backend 和 residency。

逻辑上，在以下位置建立 segment barrier：

- backend 改变；
- mandatory synchronization 或不可跨越的 dependency boundary；
- 特殊长期 resident weights；
- working-unit lifecycle 不兼容；
- 合并后超过 unit/pipeline byte cap。

当前 `compatible_multi()` 进一步要求相邻 weights 位于同一 layer，并且合并后的
bytes 不超过 profile 中的 `max_merge_bytes`，从而避免跨层 head-of-line
blocking 和过大的 diff patch。

### 3.3 Cut construction

对每个 weight \(W_i\) 先创建 Tensor node。若满足以下条件，则沿输出 rows 创建
Cut leaves：

- weight 是非空二维 tensor；
- row ranges 可以独立执行；
- quantization block 和 backend kernel 支持该类型；
- row boundary 满足 layout/kernel alignment；
- 输出 row ranges 不重叠；
- 不是不允许拆分的特殊共享 weight。

当前实现对 eligible weight 使用两块预分配 physical row tiles：

```text
Tensor(Wi)
  ├── Cut(Wi, rows [0, h/2))
  └── Cut(Wi, rows [h/2, h))
```

CPU 和 OpenCL 的 Tensor/Multi 也可以在相同 physical tiles 上分组执行。因此
granularity change 主要修改 unit metadata、pin/retire boundary 和 dispatch
方式，不要求为每次切换重新创建 backend buffers。

### 3.4 Multi construction

对 execution stream 中相邻的 Tensor nodes \(T_i,T_{i+1}\)，满足
`can_group` 时创建 Multi parent：

\[
M_{i,i+1}=\mathrm{Multi}(T_i,T_{i+1}).
\]

必须区分：

\[
\mathrm{can\_group}(T_i,T_j)
\quad\text{and}\quad
\mathrm{can\_fuse}(T_i,T_j).
\]

普通 Multi 只合并 working-unit scheduling、pipeline reservation 和 retire
boundary，并不自动表示 layout/kernel fusion。只有 shape、dtype、layout、
backend kernel 和 output dependency 都兼容时，才设置 `fuse_layout` 或
`fuse_kernel`。当前 CPU 实现支持受限 cross-tensor pair fusion；OpenCL Multi
仍是真实粗粒度 unit，但不能在没有实现 cross-tensor pair kernel 时虚报该收益。
为避免另一个方向的不公平，OpenCL 的 Multi、Tensor 和 Cut 都使用相同的
two-tile physical representation 和 fused dual-half GEMV；三者只在 logical
LOAD/PREPARE/retire frontier 上不同。每个 resident tile 的 image view 被缓存，
并在 WBM eviction 时释放，因而不会形成 budget 外的隐藏 residency。
这里的 GPU Cut dependency 必须写成“两块 tile 分别 LOAD/PREPARE，随后等待
两块都 ready，再提交一次 fused dual-half GEMV”，不能写成两个可独立计算的
half-GEMV；后者会虚构同一个 tensor 内部的 compute/layout overlap。CPU Cut
没有该 fused pair dependency，仍按两个 independent half computes 建模。

### 3.5 Valid mixed partition

令 tree 的 physical leaves 为 \(\mathcal L\)，selected units 为
\(\mathcal F_t\)。有效 partition 满足：

\[
\forall l\in\mathcal L,\qquad
\sum_{u:\,l\in Leaves(u)} z_u^t=1.
\]

即每个 physical weight range 恰好被一个 selected node 覆盖，不能遗漏，也不能
被 Tensor 和其 Cut children 重复覆盖。

例：

```text
Selected units = {Tensor(A), Cut(B0), Cut(B1)}
```

表示 A 使用 Tensor granularity，B 使用 Cut granularity。

## 4. Elastic plan representation

时刻 \(t\) 的完整 plan 定义为：

\[
P_t=(\mathcal F_t,\mathcal R_t,\mathcal S_t),
\]

其中：

- \(\mathcal F_t\)：mixed working-unit partition；
- \(\mathcal R_t\)：weight residency/placement；
- \(\mathcal S_t\)：load-layout-compute schedule 和 transition timeline。

当前工程采用分解式求解：

1. placement solver 先生成 weight placement/residency 和 transition events；
2. Super-Tensor planner 在该 residency 上搜索 mixed partition；
3. `working_unit.units` 被附加到同一个 ExecPlan；
4. runtime 根据 placement timeline 和 selected units 执行 pipeline。

因此 granularity 已进入同一个 runtime plan 和在线 update path，但当前
`super_tensor_planner.py` 的搜索是 conditional on placement，而不是一个单体
MILP 同时枚举全部 placement 与 granularity。论文可以将整体目标写成联合优化，
实现部分应诚实说明采用 decomposition 控制在线求解开销。

每个 plan 中的 working-unit row 至少包含：

```text
unit_id
kind = multi | tensor | cut
weight_ids
physical tile/range ids
backend
fuse_layout / fuse_kernel
predicted cost
generation
```

Stable unit ids 使 runtime 可以仅更新发生变化的 eviction groups 和 pin/retire
boundaries。

## 5. Cost model

### 5.1 Per-node feature vector

每个 Super-Tensor node 保存：

\[
\phi_u=
\{m_u,\hat I_u,\hat P_u,\hat C_u,\hat H_u,
op_u,g_u,b_u,r_u\},
\]

其中：

- \(m_u\)：weight bytes；
- \(\hat I_u\)：I/O service demand；
- \(\hat P_u\)：layout preparation demand；
- \(\hat C_u\)：compute service demand；
- \(\hat H_u\)：request、launch、dispatch 和 bookkeeping overhead；
- \(op_u,g_u,b_u,r_u\)：operator、granularity、backend 和 residency。

### 5.2 Stage estimates

对 non-resident unit：

\[
\hat I_u=
(1-x_u)
\left[
\alpha_{io}(g_u)+
\frac{m_u}{BW_{io}(m_u,g_u)}
\right],
\]

\[
\hat P_u=
(1-x_u)
\left[
\alpha_{prep}(g_u,b_u)+
\frac{m_u}{BW_{prep}(m_u,g_u,b_u)}
\right].
\]

Compute cost 为：

\[
\hat C_u=
\frac{FLOPs_u}
{Throughput(op_u,m_u,g_u,b_u)}
+N_{dispatch}(u)\alpha_{submit}(b_u).
\]

父节点 cost 不能机械等于 children cost 之和：

- Multi 可以减少 I/O request 和 unit submissions；
- Cut 增加 units，但可能提高局部 preparation 效率和 overlap；
- fused node 会改变 preparation 或 compute service；
- resident ratio 会改变 I/O/Layout 在端到端 latency 中的权重。

当前 profile 使用 base phase costs、mode scales、launch overhead、
resident-ratio curves、pipeline-efficiency residual 和 transition coefficients。
校准入口为 `runtime/plan/calibrate_granularity_profile.py`。

### 5.3 Online telemetry correction

在线路径读取近期：

```text
pipeline_wait_us
prepare_us
compute_us
executed_units
```

并计算：

\[
\rho_{wait}=
\frac{T_{wait}}
{T_{wait}+T_{prepare}+T_{compute}}.
\]

该值用于修正离线 profile 对 exposed waiting 的估计。稳定版本应使用有界 EMA：

\[
\theta_t=(1-\eta)\theta_{t-1}+\eta\theta_t^{obs},
\]

并为样本量不足、thermal state 改变或 backend 改变设置 confidence guard。

Pipeline-efficiency residual 只修正 non-resident I/O 和 Layout demand 在
pipeline 上的暴露程度，不能再次缩放 Compute。Compute 必须保留 fixed-budget
kernel counter 校准出的 throughput scale。否则在全驻留时也会把 residual
乘到 GEMV cost，错误抹去 Multi 的实测 kernel/submission 优势，并导致高预算
frontier 过度 split。

### 5.4 Mixed-frontier fragmentation

纯 Multi/Tensor/Cut sweep 可以校准每种 mode 的阶段 demand，却无法观察同一个
graph 中不同 unit representation 交错产生的代价。短 mixed-frontier 运行显示，
大量局部最优的 Cut/Tensor/Multi 交替会缩短连续 pipeline run、打散 direct-I/O
batch，并增加 eviction-group turnover。若忽略该项，planner 可能为每 token
仅数毫秒的预测收益制造数十到上百个边界。

因此定义 intra-plan fragmentation cost：

\[
C_{intra}(\mathcal F_t)=
\alpha_{cut}\,N_{Cut\leftrightarrow Whole}+
\alpha_{merge}\,N_{Tensor\leftrightarrow Multi}.
\]

Cut 与 whole-tensor 的边界改变 physical row-unit lifecycle；Tensor 与 Multi
边界只改变 whole-tensor grouping，所以两者必须分别校准。普通 Multi 中因
shape/layer constraint 留下的 Tensor fallback 不应默认按 Cut layout transition
处罚。当前实现将 `cut_boundary_ms` 和 `merge_boundary_ms` 保存在 backend
profile 中。

本轮 OP12 校准先 sweep 物理意义更强的
`cut_boundary_ms`，并将 `merge_boundary_ms` 固定为 0：前者会改变 row-tile
lifecycle，已有短 mixed run 直接显示其 fragmentation；后者目前只是逻辑
Tensor/Multi grouping，尚无可独立归因的正代价。两项在模型中保持分离，避免
用 Cut 的实测惩罚人为压低合法的 Tensor fallback。

该项必须允许校准为 0，而不能假设 fragmentation 一定主导。OP12 CPU 的独立
fixed-node sweep 中，较大 `cut_boundary_ms` 虽然形成了更长 Cut runs，却在
中预算节点移除了仍然有收益的 Cut choices；因此本轮 CPU 选择 0。这个负结果
说明 planner 必须同时比较 phase benefit、unit overhead 和 boundary effect，
不能用一个“越平滑越好”的启发式替代 cost model。GPU 使用自己的 profile
重新校准，不能继承 CPU 的 0。

这项成本与跨时间的 plan transition 不同：

- \(C_{intra}\) 每个 token 都由当前 mixed frontier 的结构产生；
- \(C_{switch}\) 仅在 \(P_{t-1}\rightarrow P_t\) 时支付一次。

边界系数使用独立的短 fixed-node mixed runs 选择，完整 600 秒动态 trace 保持
held out。低、高预算端点的 frontier 必须分别与本轮 fixed sweep 的实测最快
granularity 对齐；选择器不预设 Cut 或 Multi 必须在哪个端点获胜。如果两个端点
的实测 winner 相同，则本轮数据不构成 crossover evidence，实验链直接停止，
而不是强行生成 mixed plan。

这个 regularizer 的作用不是全局压低 Cut 数量，而是让相邻、收益相似的
weights 形成较长的同类 run。例如，低预算可以选择一个连续 Cut region，
而不是在每个 tensor 周围反复 Cut→Tensor→Cut；因此既保留局部 preparation
优势，又恢复 direct-I/O batching 和跨 unit pipeline overlap。

## 6. Objective function

### 6.1 Decision variables

\[
z_u^t\in\{0,1\}
\]

表示 node \(u\) 是否属于当前 partition；

\[
x_u^t\in\{0,1\}
\]

表示 selected unit 是否 resident。

因此：

\[
\mathcal F_t=\{u\mid z_u^t=1\},\qquad
\mathcal R_t=\{u\mid x_u^t=1\}.
\]

### 6.2 Resource-constrained pipeline latency

对每个 selected physical unit 建立：

```text
I/O(u) -> Layout(u) -> Compute(u)
```

GPU fused Cut 是上述普通 unit recurrence 的一个 grouped-compute 特例：

```text
I/O(c0) -> Layout(c0) ┐
                      ├-> FusedCompute(c0, c1)
I/O(c1) -> Layout(c1) ┘
```

因此 Cut 仍有两个细粒度 load/layout boundaries，但只产生一个 compute
submission；该 fused compute 的 release time 是两块 tile 的 layout completion
的最大值。

并加入 operator dependency、I/O resource、prepare workers 和 backend compute
queue 的资源约束。单 I/O、单 Prepare、单 Compute resource 的递推为：

\[
f_u^{io}=
\max(a_u,A^{io})+\hat I_u,
\]

\[
f_u^{prep}=
\max(f_u^{io},A^{prep})+\hat P_u,
\]

\[
f_u^{comp}=
\max\left(
f_u^{prep},
A^{comp},
\max_{v\in pred(u)}f_v^{comp}
\right)+\hat C_u.
\]

\(A^{io},A^{prep},A^{comp}\) 是各资源的下一可用时间。资源约束下的
pipeline critical path 为：

\[
\max_{u\in\mathcal F_t}f_u^{comp}
=
\mathrm{CP}(DAG_t).
\]

加入 mixed-frontier fragmentation 后，planner 使用的 steady-state latency
估计为：

\[
\hat L(P_t,B_t)=
\mathrm{CP}(DAG_t)+C_{intra}(\mathcal F_t).
\]

当前 `PipelineState.append()` 使用等价的三阶段 earliest-start recurrence；
offline search 进一步使用 max-plus matrices 对局部 edit 进行 prefix/suffix
增量评分。

### 6.3 Transition cost

\[
C_{switch}(P_{t-1},P_t)=
C_{edit}+C_{repack}+C_{drain}+C_{residency}.
\]

当前 granularity transition 近似为：

\[
C_{edit}=
\alpha_{transition}+
\beta_{transition}
\sum_{w\in ChangedWeights}bytes(w).
\]

实现中，\(\alpha_{transition}\) 不是人工指定，也不从最终 10 分钟 trace 拟合。
系统在 held-in mixed-boundary 节点上单独计时 working-unit frontier 的构造、
验证和原子发布。首次完整安装不是 split/merge；系统通过相邻 exact
`frontier_sig` 的变化识别真实 transition，并使用该候选实测
`frontier_transition_publish_ms_max` 作为每次 frontier change 的保守固定
成本。计时从 scheduler quiescence 之后开始，并在 frontier map replace
之后结束，因此不包含 placement/residency apply。
这里比较的是 exact `frontier_sig`，而不是只比较 Multi/Tensor/Cut 的数量；
因此 unit counts 相同但所选 weights 不同的变化仍会计费，完全相同的重复
frontier 则不会计费。
Diff-before 与 Diff-now 的 attribution 进一步比较去除连续重复签名后的
transition-path hash，避免 callback 重复次数不同被误认为 split/merge
decision 不同。

完整 runtime 还会发生：

- changed unit metadata 和 eviction-group update；
- 必要的 layout regroup；
- in-flight task drain/cancel；
- residency load/eviction；
- backend synchronization。

其中 shared placement/residency apply、provider wait 和 solver time 已经位于
端到端 critical path 中，不能再次全部加入 granularity
\(C_{switch}\)，否则会重复计费。最终审计分别报告 predicted switch、
measured frontier publication、完整 `apply_exec_plan`、provider wait 和
solver wall time。

### 6.4 Single-point objective

\[
\boxed{
P_t^*=
\arg\min_{P_t}
\mathrm{CP}
\left(
DAG_{I/O\rightarrow Layout\rightarrow Compute}(P_t)
\right)
+C_{intra}(\mathcal F_t)
+\lambda C_{switch}(P_{t-1},P_t)
}
\]

### 6.5 Horizon objective

若 future budget trace 已知或可预测：

\[
\boxed{
P_{t:t+H}^*=
\arg\min
\sum_{k=0}^{H}
\gamma^k
\left[
\mathrm{CP}(DAG_{t+k})
+C_{intra}(\mathcal F_{t+k})
+\lambda C_{switch}(P_{t+k-1},P_{t+k})
\right]
}
\]

- Offline 使用真实 trace 或每个 budget bucket 的预计算候选；
- Online 使用短 horizon budget prediction；
- 完全不可预测时令 \(H=0\)，但仍使用 hysteresis。

当前 Diff-Tree 的单 edit 接受条件等价于：

\[
H\cdot\Delta L_{steady}
-
C_{switch}
-
C_{hysteresis}
-
C_{min\_gain}
>0.
\]

### 6.6 Constraints

有效 tree partition：

\[
\forall l\in\mathcal L,\qquad
\sum_{u:\,l\in Leaves(u)}z_u^t=1.
\]

Residency 只属于 selected representation：

\[
x_u^t\le z_u^t.
\]

真实 memory constraint 必须包含 transient working set：

\[
\sum_u x_u^t m_u+
M_{pinned}^t+
M_{inflight\ staging}^t
\le B_t.
\]

还需满足：

\[
z_u^t\le Compatible(u,b_t),
\]

以及 row alignment、atomic eviction group、output range disjointness 和
dependency safety。

由于 plan budget 以整 MiB 表示而 Q4 blocks/sub-buffers 是离散的，runtime 允许
固定 64 KiB alignment slack；该 slack 远小于普通 Tensor unit，不能掩盖一次
真实 working-unit overshoot。

### 6.1 Protected residency and rolling stream residency

Residency 不能只表示成“长期固定保留哪些 weights”。真实 pipeline 中还需要区分：

\[
\mathcal R_t=
\mathcal R_t^{keep}\cup\mathcal R_t^{stream},
\]

其中 \(\mathcal R_t^{keep}\) 是 planner 希望跨 token 保留的 weights，
\(\mathcal R_t^{stream}\) 是最近已经 prepare、并可能被下一轮顺序访问复用的
rolling working set。两者仍受同一个 hard byte constraint，stream residency
不是额外内存。

在实现中，planner 显式预留 \(H_t^{stream}\)，并对 persistent set 使用：

\[
\sum_{u\in\mathcal R_t^{keep}}m_u
\le
B_t-M_{KV}-M_{misc}-M_{safety}-M_{pinned}-H_t^{stream}.
\]

backend 仍以未减去 \(H_t^{stream}\) 的真实 weight target 执行 hard-budget
检查。因此该预留只是把同一 budget 在 keep 与 stream 之间重新分配，并没有给
方法额外内存。\(H_t^{stream}\) 由 selected frontier 的最大 current/lookahead
window 决定；OP12 CPU 的首个筛选值为审计得到的 15.75 MiB，向上取整为
16 MiB。

一次 OP12 CPU 审计显示，若每个 graph 结束时强制清空所有 plan-DISK units，
planner 又将 \(\mathcal R^{keep}\) 填满 budget，下一 token 的第一个较大 unit
只能临时逐出 plan-resident weight。Online 的理论 non-resident volume 为
371.92 MiB/token，实际 direct read 为 441.65 MiB/token；其中
65.48 MiB/token 来自 runtime 记录的 plan-protection relaxation，可解释
93.9% 的额外读取。该结果用于定位实现瓶颈，不作为最终方法性能。

因此 mixed-granularity plan 同时输出：

```text
protected keep set
rolling stream policy
working-unit frontier
```

runtime 在相同 budget 内优先复用合法的 rolling stream units；只有当 keep set
与当前/in-flight unit 仍无法同时满足约束时，才允许有计数的 protection
relaxation。Planner 将 `relaxed bytes`、额外 direct-read bytes 和 exposed wait
作为反馈，而不是把固定 resident identity 当成不可调整的事实。

## 7. Offline planning

### 7.1 Candidate generation

对 ordered weights 做 beam-DP。位置 \(i\) 的候选动作是：

```text
Tensor(Wi)                    consume 1 tensor
Cut(Wi)                       consume 1 tensor, if eligible
Multi(Wi, Wi+1)               consume 2 tensors, if compatible
```

每个 SearchState 保存：

```text
current index
selected decisions
pipeline state
Cut/whole and Tensor/Multi boundary counts
```

扩展一个 decision 时，直接更新三阶段 pipeline recurrence。每个 index 仅保留
beam width 内包含 fragmentation regularizer 的 predicted makespan、
exposed wait 和 unit count 最好的 states。
当前默认 beam width 为 128。

### 7.2 Local refinement

Beam-DP 之后使用局部 split/merge refinement：

- Tensor \(\rightarrow\) Cut；
- Cut \(\rightarrow\) Tensor；
- adjacent Tensors \(\rightarrow\) Multi；
- Multi \(\rightarrow\) component Tensors。

Prefix/suffix max-plus matrices 使单 edit 的 makespan 更新不需要重放整个
partition。

### 7.3 Budget-indexed offline table

对每个 budget bucket：

1. placement solver 生成 placement/residency；
2. mixed planner 在该 residency 上生成 offline partition；
3. 将 `working_unit.units` 写入 bucket plan；
4. 保存 index 和相邻 transition metadata。

已知完整 memory trace 时，可进一步对 bucket plans 做时间维 DP，将相邻 plan
switch cost 纳入整个 trace 的最优路径，而不是逐点贪心。

## 8. Online Diff-Tree

### 8.1 Why a diff tree

预算每次变化时重跑完整 beam search 会增加在线延迟，并可能大范围改变 unit
ids、pins 和 residency。Diff-Tree 从当前 partition 出发，只评估局部
parent/children edits。

Tree 定义合法 edit space；offline partition 只作为 guidance，不会被无条件
应用。每个 edit 都在当前 residency 和 recent telemetry 上重新评分。

### 8.2 Trigger

online adjustment 可以由以下事件触发：

- budget 进入新的 bucket；
- current plan 不再满足 memory constraint；
- observed wait/prepare/compute 与 profile 偏差超过阈值；
- backend 或 thermal operating point 发生稳定变化；
- 当前 plan 的 predicted gain 超过 switch threshold。

瞬时 budget noise 不直接触发 plan change；先经过 bucketization、minimum gap
和 hysteresis。

### 8.3 Bootstrap

推理开始前不存在需要保留的 live working-unit state，因此：

1. 读取当前 budget 的 refined offline mixed partition；
2. 作为初始 frontier；
3. switch cost 记为 0；
4. 首个 decode 后才启用 bounded online edits。

### 8.4 Local edit search

对当前 partition：

1. 枚举所有合法 local edits；
2. 使用 max-plus prefix/suffix 计算 approximate steady gain；
3. 若 telemetry correction 非零，只精确重放排名最高的少量 candidates；
4. 计算 horizon net gain；
5. 接受 net gain 最大且为正的 edit；
6. 最多执行 `max_edits` 次。

Offline target distance 只用于极小的 deterministic tie-break，不能将亏损 edit
变成盈利 edit。

预算变化后，Online 不会把“新 bucket 的 offline frontier”直接替换进 runtime。
它先在新的 residency \(R_t\) 下重新评估当前 frontier：

\[
C(\mathcal F_{t-1};R_t)
\quad\text{vs.}\quad
C(\mathcal F_t^{target};R_t)
\lambda C_{switch}.
\]

只有 target 在 planning horizon 内仍有正的 net gain，才执行 split/merge。
例如一次 3968→4224 MiB 的实测调整中，两种 frontier 在新 residency 下的
predicted latency 都为 824.324 ms，而切换后的 horizon net gain 为
\(-0.07\) ms；因此 Online 只更新 residency/placement，保留原 frontier。
这避免了“每跨一个 budget bucket 就重建颗粒度”的无效切换。

### 8.5 Budget-dependent behavior

Budget 降低、non-resident ratio 或 exposed wait 增大时：

- 优先 split critical-path 上的大 Tensor/Multi；
- 让部分 weight rows 更早完成 load/layout；
- 缩短 compute 等待 ready unit 的时间。

Budget 升高、大部分 weights resident 时：

- 优先 merge compatible siblings；
- 降低 working-unit、request 和 dispatch 数；
- 减少 pipeline bookkeeping 和 driver submission overhead。

这不是固定的 low-budget=Cut、high-budget=Multi 规则。真正的 decision 还依赖
tensor size、operator、backend、当前 residency、history 和 transition cost。

## 9. Runtime application

### 9.1 Applying a plan diff

planner 输出：

```text
removed unit ids
added unit ids
changed weight/tile ids
new fusion flags
new generation
```

runtime 在 graph boundary 先完成已有提交，再原子发布 frontier diff：

1. 将新 plan 的 slices 按 weight 分组；
2. 在一个锁区间内比较 old/new slice lists；
3. 只删除或替换发生变化的 weight entries；
4. 保留未改变 entries、physical buffers 和 pool objects；
5. frontier 相同则不增加 generation；否则一次性发布新 generation。

若 placement、dispatch 和 event schedule 的 core signature 未变化，runtime
走 frontier-only fast path：不重复 reconcile weights，不清空 routing，也不
invalidate/rebuild graph。CPU/OpenCL backend 在每个 graph 入口读取新的
generation 和 unit map，因此 metadata-only split/merge 可以直接作用于下一
个 graph。日志记录 `delta_weights`、`publish_ms` 和 frontier-only apply
count，使 split/merge 的 isolated publication cost 能与完整 `apply_ms`
分开验证。

若 core signature 也发生变化，则仍执行完整 plan apply、residency reconcile
和 graph invalidation；不能为了降低切换开销跳过真实 placement transition。

### 9.2 Execution pins and budget safety

每个 current unit 从 ensure-resident 前到 kernel submission 后持有 execution
pin。Lookahead pin 和 execution pin 使用独立 reference，防止当前 tensor 被
post-ensure eviction 选中并导致 invalid OpenCL memory object。

当 plan 使用 `NO_AUTO_EVICT` 时，placement-aware victim selector 负责正常
plan transition；但 current non-resident unit 必须执行。runtime 在 ensure 前：

1. 计算 current unit missing bytes；
2. 在 budget 中预留该 working set；
3. 临时使用 WBM MRU victim 为 current unit 腾出物理空间；
4. 仍遵守 pin 和 atomic group；
5. ensure 后验证 resident bytes；
6. kernel submission 完成后释放 execution pin。

这一 reserve path 只保证 plan 能在真实物理 budget 内执行，不改变 planner 的
长期 residency objective。

### 9.3 Pipeline

Granularity-aware pipeline 的 lookahead 单位是真实 working unit，不是 graph
node distance。一个 lookahead 表示一个 Multi、Tensor 或 Cut unit。

Cross-graph successor 只在引入新的 WBM indices 时 materialize；仅 graph
signature 不同但覆盖同一 weights 的 successor 不得重复 stage。当前动态实验
使用 `GRAPH_LOOKAHEAD=1`，即一个真实 next backend-subgraph successor。

## 10. End-to-end algorithm

```text
Offline:
    weights = NormalizeModelWeights(model_meta)
    forest  = BuildImplicitSuperTensorForest(weights)

    for budget bucket B:
        placement_B = SolvePlacement(B)
        profile_B   = ConditionProfile(placement_B.residency)
        units_B     = BeamDPAndRefine(forest, profile_B)
        plan_B      = AttachUnits(placement_B, units_B)
        Save(plan_B)

Online at time t:
    B_t       = BudgetWatcher()
    base_plan = PlacementPlan(B_t)
    target    = OfflineMixedPlan(B_t)

    if no current plan:
        current = target
    else:
        telemetry = RecentPipelineCounters()
        candidates = LocalSplitMergeEdits(current)
        current = AcceptUpToKProfitableEdits(
            candidates,
            target,
            telemetry,
            horizon,
            switch_cost,
            hysteresis)

    diff = Diff(previous_plan, current)
    ApplyPlacementAndWorkingUnitDiff(diff)
    ExecuteGranularityAwarePipeline()
```

## 11. Correctness and fairness invariants

任何性能结果进入论文前必须满足：

1. **Coverage:** selected units 覆盖相同 logical weights 和 operator results；
2. **No overlap:** Cut output row ranges 不重复；
3. **Token correctness:** 相同 prompt/seed 的 token prefix 一致；
4. **Budget safety:** resident+pinned+in-flight 不超过 budget 与 alignment slack；
5. **Pin safety:** current execution unit 在 kernel submission 前不可被驱逐；
6. **Unit isolation:** Multi/Tensor/Cut 使用同一 backend、prompt、trace 和保留
   策略；
7. **Pipeline enabled:** 三种模式均使用真实 load-layout-compute overlap；
8. **No hidden initialization:** kernel compile、backend init、buffer creation 与
   measured stage boundary分开；
9. **Stage semantics:** integrated `direct_read`、`layout preparation` 和
   `pipeline wait` 允许 overlap，不能相加解释总 latency；
10. **Device isolation:** OP12 单设备互斥、相同 thermal-status start、无外部
    active inference。

## 12. Baselines and ablations

完整 10 分钟原始 trace 使用以下方法：

- **Static-Min:** 全程使用 trace 最低 budget 的 plan；
- **Static-Max:** 全程使用 trace 最高 budget，作为性能上界；
- **MRU:** 无预测的 online cache replacement；
- **Offline:** 已知 budget table 的 placement 加完整 mixed-frontier
  granularity cost model；
- **Online:** 在线 placement adjustment 加完整 mixed-frontier reoptimization；
- **Diff-before:** placement + mixed Diff-Tree，使用 phase-only profile；
- **Diff-now:** placement + bounded mixed Diff-Tree，加入 measured
  exposed-pipeline efficiency residual、intra-plan fragmentation cost 和
  低开销局部 split/merge。

CPU 和 GPU 分开运行，暂不进行 heterogeneous placement。

需要报告：

- end-to-end ms/token 和 tokens/s；
- I/O bytes/time；
- layout preparation；
- exposed pipeline wait；
- unit/dispatch counts；
- switch count/cost；
- resident/pinned/in-flight peak；
- budget violations；
- token correctness 和 device isolation。

关键 ablations：

1. fixed Multi/Tensor/Cut；
2. mixed offline vs fixed Tensor offline；
3. Diff-before vs Diff-now；
4. without switch cost/hysteresis；
5. without telemetry wait correction；
6. graph lookahead 1 vs duplicate cross-graph staging；
7. Cut sequential vs dual fused dispatch；
8. Multi scheduling-only vs legal layout/kernel fusion。

## 13. Complexity

令 weights 数为 \(N\)，beam width 为 \(K\)。

- fixed three-level candidate generation 为 \(O(N)\)；
- beam-DP 约为 \(O(NK)\)；
- max-plus prefix/suffix 构造为 \(O(N)\)；
- 每轮 local edit 的 approximate evaluation 为 \(O(N)\)；
- online 最多接受 \(E\) 个 edits，约为 \(O(EN)\)；
- telemetry 非零时只精确重放 top-\(q\) candidates。

当前 \(N\approx225\)、\(K=128\)、\(E\le16\)，适合在 budget bucket change 时
运行，而不是每个 operator 或每个 token 重跑全局 search。

## 14. Limitations and next steps

1. 当前 Multi 仅合并相邻两个 tensors；更大 Super-Tensor 需要 balanced
   hierarchy 和更强的 head-of-line blocking guard。
2. 当前 granularity search conditional on placement；真正 joint optimizer 可
   通过 placement/granularity alternating optimization 或统一 CP-SAT 实现。
3. GPU host-visible prepare time 不是纯 device kernel time；需要 OpenCL event
   profiling 分离 write、conversion kernel 和 synchronization。
4. Cost profile 对 thermal/DVFS 状态敏感；需要 confidence-aware online
   calibration。
5. 当前 Cut 固定两路；未来可按 tensor size、operator 和 backend 支持 2/4-way
   adaptive cuts。
6. MoE 3-D expert tensors、shared weights 和跨分支 graph 需要独立 compatibility
   rules。
7. CPU/GPU heterogeneous placement 应在单 backend 结果稳定后再加入。

## 15. PPT-ready summary

主 design slide 只保留：

\[
P_t=
\{\text{mixed working units},
\text{residency},
\text{pipeline schedule}\},
\]

\[
\boxed{
P_t^*=
\arg\min_{P_t}
\mathrm{CP}(DAG_{I/O\rightarrow Layout\rightarrow Compute}(P_t))
+\lambda C_{switch}(P_{t-1},P_t)
}
\]

\[
M_{resident}+M_{pinned}+M_{inflight}\le B_t.
\]

配套三步：

1. Generate mixed candidates through legal tree split/merge；
2. Predict the resource-constrained pipeline critical path；
3. Select the feasible plan with the lowest latency plus switching cost。

一句话总结：

> The tree defines the legal choices; the cost model evaluates their
> pipeline behavior; the optimizer selects and incrementally updates the
> mixed plan.
