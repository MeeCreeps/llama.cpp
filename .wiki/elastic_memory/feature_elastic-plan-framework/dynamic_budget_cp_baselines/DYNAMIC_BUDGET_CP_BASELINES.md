# Dynamic Budget CP Baselines

## 目标

当前 elastic runtime 已经能执行一个给定 `ExecPlan`：

```text
disk / mmap
  -> LOAD
  -> TRANSFER
  -> XFORM
  -> op compute
```

并且已有：

```text
PlanProvider(table/callback)
PlanExecutor
BudgetWatcher
OpenCL WBM
CPU_Elastic WBM
direct disk reload
explicit CPU/GPU transform
```

本阶段目标不是继续手工构造单个 plan，而是实现两个动态 memory budget baseline：

1. `offline-table` baseline：离线为不同 budget 求好 plan，runtime 在 budget 变化时按 budget 查表并执行差量 movement。
2. `online-cp` baseline：runtime 读取当前每个 weight 已在哪个 backend，然后按当前 budget 在线构建并求解 plan。

这两个 baseline 都有已知缺陷：

```text
offline-table:
  优点: runtime 查询快，只需要选档 + apply plan
  缺点: plan 只针对 budget，不针对当前 residency；budget 变化时可能做多余 movement

online-cp:
  优点: 能把当前 residency / transformed layout 纳入初始状态，movement 最少
  缺点: 每次 budget 变化在线 CP/ILP 求解可能太慢，尤其 op 级别变量多
```

所以本阶段的判定标准不是“谁最优”，而是：

```text
1. 两种 baseline 都能生成可执行 ExecPlan。
2. 两种 baseline 都能在真实 runtime 下随 budget 变化切换 plan。
3. 记录 plan 查询/求解耗时、apply 耗时、movement 耗时、decode ms/token。
4. 给出不同 tensor shape / backend / transform 的实测 cost model。
```

## 现有接口映射

现有 `ExecPlan` 已经足够承载第一版 baseline：

```text
WeightPlan:
  weight_id
  name
  layer
  byte_size
  location = GPU / CPU / DISK
  xform    = NONE / CPU_REPACK / GPU_CONVERT

OpPlan:
  op_id
  name
  layer
  compute_backend = CPU / GPU
  weight_id
  dispatch = STATIC / RUNTIME
  migrate
  migrate_from
  migrate_xform

PlanEvent:
  kind = LOAD / TRANSFER / XFORM / EVICT
  weight_id
  from_loc
  to_loc
  engine = DISK / TRANSFER / CPU / GPU
  anchor_op_id
  overlap_group
```

当前需要补齐的 runtime 语义：

```text
1. offline-table 直接复用 PlanProvider::create_table。
2. online-cp 走 PlanProvider::create_callback。
3. budget 变化时 maybe_apply_plan() 触发 apply_exec_plan()。
4. 如果要做到 op 级别 prefetch / overlap，需要把 defer_stage_events=true 路径接到 graph compute 的 anchor op。
5. 第一版 baseline 可以先沿用 apply-time stage，下一个阶段再打开 anchor-time stage。
```

## 成本模型

CP 求解前必须先有 profile 数据。第一版不要假设论文里的固定 bandwidth，而是用目标设备实测表。

### 需要测的对象

按 tensor 形状、量化类型、backend、stage 拆开：

```text
LOAD:
  disk -> CPU raw staging
  metrics: bytes, ms, MB/s, direct_io flag, page_cache flag

TRANSFER:
  CPU raw staging -> GPU backend-visible raw buffer
  CPU raw staging -> CPU_Elastic resident slot
  metrics: bytes, ms, MB/s, blocking/non-blocking, event wait ms

XFORM:
  CPU raw -> CPU_REPACK layout
  GPU raw -> GPU_CONVERT layout
  metrics: bytes, ms, output bytes, temp bytes, queue wait ms

COMPUTE:
  CPU matmul on generic/repack layout
  GPU matmul on raw/SOA layout
  metrics: op name, layer, ne/nb shape, quant type, ms

BOUNDARY:
  CPU -> GPU activation visibility/copy
  GPU -> CPU activation visibility/copy
  metrics: activation bytes, wait ms, copy ms
```

### Profile key

实测表用稳定 key，避免只按 bytes 拟合导致误差太大：

```text
tensor_cost_key:
  weight_name_pattern
  op_kind
  quant_type
  ne0, ne1, ne2, ne3
  byte_size
  source_location
  target_backend
  xform_kind
```

如果 key 未命中，fallback 到分段线性模型：

```text
stage_ms = alpha_stage + bytes / bandwidth_stage
compute_ms = alpha_backend_op + flops_or_bytes / throughput_backend_op
```

### Profile 输出文件

建议新增：

```text
runtime/plan/profiles/<device>/<model>/stage_costs.json
runtime/plan/profiles/<device>/<model>/op_costs.json
runtime/plan/profiles/<device>/<model>/boundary_costs.json
```

示例 schema：

```json
{
  "device": "android-opencl-target",
  "model": "Llama-3.2-3B-Instruct-q4_0.gguf",
  "records": [
    {
      "name": "blk.0.ffn_down.weight",
      "quant": "Q4_0",
      "shape": [8192, 8192, 1, 1],
      "bytes": 37748736,
      "stage": "GPU_CONVERT",
      "backend": "GPU",
      "median_ms": 1.72,
      "p90_ms": 1.95,
      "samples": 20
    }
  ]
}
```

### Profile 实施步骤

1. 复用现有 `GGML_ELASTIC_TIMING=1`、`GGML_ELASTIC_STAGE_DETAIL=1`、OpenCL direct read counters、CPU_Elastic stage counters。
2. 增加 per-weight/per-op CSV 输出，至少包含 `weight_id/name/bytes/kind/backend/ms`。
3. 对典型 plan 做 warmup 后重复 decode，取 median/p90。
4. 额外增加 synthetic microbench：对不同 shape 的 Q4_0/Q8_0/Q5_K/Q6_K weight 单独跑 LOAD/TRANSFER/XFORM。
5. 把 profile CSV 汇总成 JSON cost table，供 offline 和 online solver 共用。

## CP 问题定义

### 输入

```text
W: weight tensor 集合
O: op 集合，每个 op 消费一个主 weight
B: 当前可用 memory budget
S0: 当前 residency 状态，online-cp 使用；offline-table 固定为空或固定 canonical state
C: cost table
M_kv, M_misc: KV cache 和 runtime misc 预留
```

预算可用于 weight 的部分：

```text
B_weight = B - M_kv - M_misc - safety_margin
```

### 决策变量

每个 weight：

```text
loc[w] in {DISK, CPU, GPU}
xform[w] in {NONE, CPU_REPACK, GPU_CONVERT}
resident_cpu[w] in {0,1}
resident_gpu[w] in {0,1}
```

每个 op：

```text
be[o] in {CPU, GPU}
dispatch[o] in {STATIC, RUNTIME}
```

每个 weight/op 的 movement：

```text
need_load[w]       in {0,1}
need_transfer[w]   in {0,1}
need_xform[w]      in {0,1}
need_evict[w]      in {0,1}
anchor[w]          in op_id domain
```

对于 online-cp，movement 由当前状态决定：

```text
need_load[w] = 1 iff S0[w] does not contain loc/xform needed by selected op route
need_evict[w] = 1 iff S0[w] resident and new plan cannot keep it under budget
```

对于 offline-table，movement 按 canonical initial state 估计：

```text
canonical S0:
  all weights are DISK
  no transformed backend layout resident
```

这就是 offline baseline 的主要缺陷：真实 runtime 可能已经有 CPU/GPU resident data，但离线 plan 不知道。

### 约束

内存约束：

```text
sum_w gpu_bytes[w] * resident_gpu[w]
  + gpu_pool_reserved
  <= B_weight_gpu

sum_w cpu_bytes[w] * resident_cpu[w]
  + cpu_stage_reserved
  <= B_weight_cpu
```

第一版可以用单一 budget：

```text
sum_w memory_bytes(loc[w], xform[w]) <= B_weight
```

后续再拆 CPU/GPU budget，因为移动 SoC 上 DRAM 共享但 backend pool 仍有各自 cap。

可执行性约束：

```text
be[o] = GPU -> weight(o) must be GPU resident before op o compute
be[o] = CPU -> weight(o) must be CPU resident before op o compute
GPU resident -> xform[w] in {NONE, GPU_CONVERT}
CPU resident -> xform[w] in {NONE, CPU_REPACK}
DISK resident -> xform[w] = NONE
```

layout 约束：

```text
Q4_0/Q8_0 GPU route -> GPU_CONVERT if OpenCL matmul requires SOA
Q5_K/Q6_K GPU route -> NONE or backend-specific raw path
CPU_Elastic first baseline -> CPU_REPACK is explicit but may be identity copy
```

op 顺序约束：

```text
anchor[w] <= first_consumer_op[w]
LOAD(w) before TRANSFER(w)
TRANSFER(w) before XFORM(w)
XFORM(w) before compute(first_consumer_op[w])
EVICT(w) after last_consumer_op[w] if not resident after plan
```

backend boundary 约束：

```text
be[o] != be[o+1] -> boundary_cost[o] is paid
if HostMapped compute activation enabled -> boundary_cost may be 0 or visibility-only
```

### 目标函数

第一版目标使用加权和：

```text
minimize
  predicted_token_ms
  + lambda_move * movement_ms
  + lambda_switch * backend_switch_count
  + lambda_solve * solve_time_penalty
```

其中：

```text
predicted_token_ms =
  critical_path(
    compute_ms[o, be[o]],
    load_ms[w],
    transfer_ms[w],
    xform_ms[w],
    boundary_ms[o],
    engine_capacity
  )
```

为了第一版 CP 可落地，可以先用线性近似：

```text
predicted_token_ms =
  sum_o compute_ms[o, be[o]]
  + sum_boundary boundary_ms[o]
  + sum_w movement_ms[w]
```

第二版再做 engine-aware overlap：

```text
disk_engine_time     = sum LOAD
transfer_engine_time = sum TRANSFER
cpu_engine_time      = CPU compute + CPU_REPACK
gpu_engine_time      = GPU compute + GPU_CONVERT
predicted_token_ms   = max(engine_time on critical windows)
```

### CP-SAT 实现选择

建议先在 Python 侧实现 solver：

```text
runtime/plan/dynamic_budget_solver.py
```

使用 OR-Tools CP-SAT。如果目标函数保持线性，也可以切换到 MILP，但 CP-SAT 更适合后续表达 anchor、precedence、optional interval。

第一版变量规模控制：

```text
按 weight/op 一一对应求解，不做 sub-tensor。
只允许 CPU/GPU 两个 compute backend。
只允许每个 weight 一个 resident location。
anchor 先固定为 first_consumer_op - prefetch_distance，不作为变量。
```

这样 solver 输出可以直接转成现有 `ExecPlan`。

## Baseline 1: offline-table

### 设计

离线选择一组 budget 档：

```text
budgets = [1500, 1800, 2200, 3000, 4144, 4912, 6144, 7600, 8192]
```

每个 budget 独立求解：

```text
plan_B = solve_cp(
  budget=B,
  current_state=canonical_disk_state,
  cost_table=profile_json,
  objective=token_latency + movement_penalty
)
```

输出：

```text
plans_dynamic_offline/
  index.json
  plan_1500MiB.json
  plan_1800MiB.json
  ...
```

runtime：

```text
LLAMA_ELASTIC_PROVIDER=table
LLAMA_ELASTIC_PLANS_DIR=/data/local/tmp/elastic/plans_dynamic_offline
GGML_ELASTIC_BUDGET_CSV=/data/local/tmp/elastic/budget_trace.csv
```

当 budget 变化：

```text
BudgetWatcher -> maybe_apply_plan()
  -> PlanProvider table get(B)
  -> selected nearest budget band <= B
  -> apply_exec_plan()
  -> graph invalidate / rebuild
  -> movement according to plan
```

### 缺陷

offline-table 不读取真实 `S0`：

```text
真实状态:
  blk.10.ffn_up.weight 已经在 GPU transformed layout

离线 plan:
  只知道 budget=2200，不知道 blk.10 已经在 GPU
  可能选择 CPU route 或重新 LOAD/TRANSFER/XFORM
```

所以 offline-table 的移动成本可能高于 online-cp。

### 需要记录的指标

```text
provider_get_ms
plan_band
apply_exec_plan_ms
evict_count
load_count
transfer_count
xform_count
stage_load_ms
stage_transfer_ms
stage_xform_ms
direct_read_ms
decode_ms_per_token
budget_switch_count
graph_rebuild_count
```

## Baseline 2: online-cp

### 设计

runtime 每次 budget 变化时采集当前状态：

```text
S0[w]:
  current location: DISK / CPU / GPU
  has_cpu_raw
  has_cpu_repack
  has_gpu_raw
  has_gpu_compute_layout
  bytes held in each pool
  last_used_op/token
```

然后在线求解：

```text
plan = solve_cp(
  budget=current_B,
  current_state=S0,
  cost_table=profile_json,
  objective=token_latency + incremental_movement_cost
)
```

输出不需要落盘，直接通过 callback provider 返回 `ExecPlan`：

```text
PlanProvider::create_callback(fn)
fn(B) -> llama_plan / ExecPlan
```

### 状态采集接口

需要给 CPU_Elastic 和 OpenCL WBM 增加只读 query：

```cpp
struct elastic_weight_state {
    int weight_id;
    bool disk_available;
    bool cpu_raw_resident;
    bool cpu_compute_resident;
    bool gpu_raw_resident;
    bool gpu_compute_resident;
    size_t cpu_bytes;
    size_t gpu_bytes;
    uint64_t version;
};
```

桥接层提供：

```cpp
llama_elastic_query_weight_states(ctx, out_states)
```

第一版可以只 query：

```text
is_resident_gpu
is_resident_cpu
has_transformed_layout
byte_size
```

### 在线求解性能控制

online-cp 很可能慢，所以必须加限制：

```text
time_limit_ms = 5 / 10 / 20 / 50 / 100 sweep
relative_gap_limit = 0.05
warm_start = previous plan
budget debounce = only solve if budget band changed by >= N MiB
state hash cache = (budget_band, residency_bitmap_hash) -> ExecPlan
fallback = use offline-table plan if solver timeout
```

CP-SAT 求解超时策略：

```text
if feasible solution exists:
  use best feasible plan
else:
  use offline-table plan
```

### 缺陷

online-cp 的计划质量最好，但 runtime 开销可能抵消收益：

```text
solve_ms + apply_ms + movement_ms
```

如果 budget 波动频繁，online-cp 可能每秒求解多次，导致 token latency 抖动。

因此最终报告必须单独展示：

```text
solver wall time
solver feasible/optimal status
objective gap
cache hit rate
fallback count
```

## Plan 生成细节

solver 输出内部决策后，需要转成 `ExecPlan`。

### WeightPlan

```text
if selected backend is GPU:
  location = GPU
  xform = GPU_CONVERT or NONE

if selected backend is CPU:
  location = CPU
  xform = CPU_REPACK or NONE

if budget cannot keep resident:
  location = DISK
  xform = NONE
```

### OpPlan

第一版使用 static route：

```text
compute_backend = selected be[o]
dispatch = STATIC
migrate = false
```

如果某些 op 需要 runtime dispatch：

```text
dispatch = RUNTIME
```

但 baseline 优先使用 `STATIC`，因为可测性更好。

### Timeline

对每个需要从 disk 进入 CPU/GPU 的 weight：

CPU route：

```text
LOAD:
  from_loc=DISK
  to_loc=CPU
  engine=DISK

XFORM:
  from_loc=CPU
  to_loc=CPU
  engine=CPU
  xform=CPU_REPACK via WeightPlan
```

GPU route：

```text
LOAD:
  from_loc=DISK
  to_loc=CPU
  engine=DISK

TRANSFER:
  from_loc=CPU
  to_loc=GPU
  engine=TRANSFER

XFORM:
  from_loc=GPU
  to_loc=GPU
  engine=GPU
  xform=GPU_CONVERT via WeightPlan
```

对于 budget 降低需要释放的 weight：

```text
EVICT:
  from_loc=GPU or CPU
  to_loc=DISK
```

anchor 策略：

```text
anchor_op_id = max(0, first_consumer_op[w] - prefetch_distance)
```

第一版：

```text
defer_stage_events = false
apply-time stage
```

第二版：

```text
defer_stage_events = true
anchor-time stage
```

## 实验矩阵

### 固定 budget 单点

目的：验证 solver plan 本身是否合理。

```text
budget = 1500 / 2200 / 4144 / 8192 MiB
mode = all_gpu / all_cpu / mixed_cp
provider = offline-table / online-cp
```

每组记录：

```text
predicted_ms
actual_eval_ms_per_token
movement_ms
compute_ms
switch_count
stage counts
```

### 动态 budget trace

目的：验证 budget 变化时的 plan 切换成本。

trace：

```text
const_1500.csv
step_8192_to_1500.csv
step_1500_to_8192.csv
oscillate_1500_2200_4144.csv
real_phone_trace.csv
```

对比：

```text
offline-table
online-cp no-cache
online-cp state-cache
online-cp fallback-to-offline
```

### shape microbench

目的：让 cost model 覆盖真实 tensor shape。

至少覆盖：

```text
attention:
  q_proj / k_proj / v_proj / o_proj

ffn:
  ffn_gate
  ffn_up
  ffn_down

norm / small tensors:
  attention_norm
  ffn_norm
```

量化：

```text
Q4_0
Q8_0
Q5_K
Q6_K
```

backend：

```text
CPU_Elastic
OpenCL GPU
```

stage：

```text
LOAD
TRANSFER
XFORM
COMPUTE
BOUNDARY
```

## 实施步骤

### Step 1: profile 数据落盘

新增 per-weight/per-op timing dump：

```text
GGML_ELASTIC_PROFILE_CSV=/data/local/tmp/elastic/profile.csv
```

输出字段：

```text
token
op_id
weight_id
name
layer
kind
backend
bytes
shape
quant
ms
extra
```

完成标准：

```text
能从一次 llama-cli run 得到 LOAD/TRANSFER/XFORM/COMPUTE 的 per-weight/per-op CSV。
```

### Step 2: profile 聚合器

新增：

```text
runtime/plan/build_cost_model.py
```

输入 CSV，输出 `stage_costs.json`、`op_costs.json`、`boundary_costs.json`。

完成标准：

```text
同一个 model/device 的 cost table 可被 solver 加载。
```

### Step 3: CP solver 原型

新增：

```text
runtime/plan/dynamic_budget_solver.py
```

功能：

```text
--model-meta weights_ops.json
--cost-dir profiles/<device>/<model>
--budget-mib B
--state current_state.json or canonical
--out plan_B.json
```

完成标准：

```text
生成 native ExecPlan JSON，并能被 llama_plan_load_json() 加载。
```

### Step 4: offline-table 生成器

新增：

```text
runtime/plan/build_offline_budget_table.py
```

功能：

```text
for B in budgets:
  call dynamic_budget_solver.py --state canonical
write index.json
```

完成标准：

```text
PlanProvider::create_table 可以加载生成目录。
```

### Step 5: online callback baseline

新增 C/C++ bridge：

```text
llama_elastic_query_weight_states()
llama_elastic_set_online_solver()
```

第一版为了降低实现风险，可以让 callback provider 调外部 solver 进程：

```text
online callback -> dump current_state.json -> run solver -> load plan json
```

后续再把 solver 嵌入进程内。

完成标准：

```text
budget 变化时 callback provider 能生成并应用新 plan。
```

### Step 6: solver 速度优化

逐步增加：

```text
budget band debounce
state hash cache
previous plan warm start
time limit fallback
coarse layer-level pre-solve
op-level local refinement
```

完成标准：

```text
online-cp 在给定 time_limit_ms 下有稳定 feasible plan，timeout 时可回退 offline-table。
```

### Step 7: anchor-time stage

把 stage event 从 apply-time 推到 op anchor：

```text
ExecSinks.defer_stage_events = true
graph compute at op_id:
  events = executor.events_for_anchor(op_id)
  enqueue LOAD/TRANSFER/XFORM
```

完成标准：

```text
同一个 plan 下，apply-time 与 anchor-time 结果一致；
anchor-time 在 streaming plan 下减少 decode stall 或提前暴露 overlap。
```

## 第一版范围控制

第一版不要同时解决所有问题。建议固定以下约束：

```text
1. 只支持 CPU_Elastic + OpenCL GPU。
2. 不做 sub-tensor split。
3. 不做 NPU。
4. 不做 speculative prefetch。
5. 不把 KV cache 作为 CP 变量，只作为固定 reserve。
6. offline-table 和 online-cp 共用同一个 cost model。
7. online-cp 如果超过 time limit，直接 fallback offline-table。
```

## 预期输出

本阶段最终应产出：

```text
1. cost profile CSV/JSON
2. offline-table plans directory
3. online-cp provider path
4. dynamic budget benchmark report
5. predicted vs actual error table
```

核心表格：

| mode | budget trace | solve/query ms | apply ms | movement ms | eval ms/token | switches | fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| offline-table | step | query only | TBD | TBD | TBD | TBD | 0 |
| online-cp | step | TBD | TBD | TBD | TBD | TBD | TBD |
| online-cp-cache | step | TBD | TBD | TBD | TBD | TBD | TBD |

## 风险

1. CP-SAT op-level 求解可能太慢，需要 layer-level coarse plan + op-level local refinement。
2. cost model 如果只按 bytes 拟合，会低估 layout transform 和 small tensor overhead。
3. offline-table 的 plan 切换可能因为忽略 current residency，导致 movement 过多。
4. online-cp 如果频繁触发 graph rebuild，decode latency 抖动可能比 movement 优化更严重。
5. mixed CPU/GPU 仍有 correctness 风险；Phase 22 里 mixed full run 还不稳定，baseline 需要先用 all-GPU/all-CPU 验证，再逐步打开 mixed。

## 推荐里程碑

```text
M1: profile CSV + cost JSON
M2: solver 生成单 budget all-GPU/all-CPU/mixed plan
M3: offline-table provider 跑动态 trace
M4: online-cp 外部进程 callback 跑动态 trace
M5: online-cp cache/fallback/time-limit sweep
M6: anchor-time stage + overlap 实验
```

