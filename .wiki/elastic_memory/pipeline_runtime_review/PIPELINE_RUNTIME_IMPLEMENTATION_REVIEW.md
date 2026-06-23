# Pipeline Runtime Implementation Review

本文档说明当前 elastic planner / runtime 里 pipeline 功能是怎么实现的，供代码 review 使用。

当前结论先放前面：

- 当前实现已经把 planner-visible stage 改成 **LOAD + BACKEND_PREPARE + COMPUTE**。
- planner 新生成的是 `load / prepare / compute` schedule events。
- runtime 可以按 op anchor 延迟触发 stage events，而不是 apply plan 时一次性执行。
- OpenCL WBM 里已经有异步 host load worker，可以把 disk direct read 提前到计算前若干 op。
- `prepare` 对 GPU 表示 backend materialization：write buffer + convert / transpose / SOA prepare；对 CPU 表示 repack / transform。
- `transfer` 不再是 planner 的核心 stage；它只是 GPU prepare 内部可能使用的实现细节。

## 1. 目标

我们希望 planner 不只是告诉 runtime “哪些 weight 常驻 GPU / 哪些放 disk”，还要告诉 runtime：

- 第几个 op 前开始从 disk load 某个 weight；
- 第几个 op 前做 backend prepare；
- compute 到该 op 时尽量不再同步等待 disk read。

所以 pipeline 的核心不是引入真实 wall-clock start time，而是引入 **op-index based anchor**：

```text
anchor_op_id = 在第几个 op 附近触发这个 stage
```

这样 planner 的时间只作为 cost / ordering 参考，runtime 真正执行时按 op 到达来触发。

## 2. Planner 侧：生成 index pipeline schedule

主要文件：

- `runtime/plan/dynamic_budget_solver.py`

planner 当前有两条 schedule 来源：

1. CP-SAT interval schedule
   - 在 `cp_objective=interval_makespan` 时尝试构建。
   - schedule event 包含：
     - `kind`: `load`, `prepare`, `compute`
     - `engine`: `disk`, `prepare_gpu`, `prepare_cpu`, `compute_gpu` 等
     - `weight_id`
     - `anchor_op_id`
     - diagnostic `start_ms / end_ms`

2. index pipeline fallback
   - 如果 interval CP-SAT 没有可用 schedule，则构造 deterministic fallback。
   - fallback 不依赖真实时间，而是根据 weight 的 consumer op 和 `prefetch_distance` 提前挂 anchor。

关键思想：

```text
consumer_op = 第一次使用该 weight 的 op
anchor_op   = max(0, consumer_op - prefetch_distance)
```

对于 `disk_gpu` placement，现在 planner-visible stage chain 是：

```text
LOAD             disk -> host-accessible bytes / staging
BACKEND_PREPARE  GPU materialization: write + convert/transpose/SOA layout
COMPUTE          op uses prepared GPU weight
```

对于 `disk_cpu` placement：

```text
LOAD             disk -> host-accessible bytes / staging
BACKEND_PREPARE  CPU repack / transform
COMPUTE          op uses prepared CPU weight
```

当前默认 stage 投影：

```text
LLAMA_ELASTIC_INTERVAL_STAGE_KINDS=load,prepare
```

旧 plan 或旧 timeline 里的 `xform` 会被兼容当成 `prepare`；`transfer` 保留为内部/legacy stage，不再由 `prepare` 自动包含。

## 3. Plan IR：把 schedule 带到 runtime

主要文件：

- `runtime/plan_ir.h`
- `runtime/plan_ir.cpp`

`ExecPlan` 里有两类事件：

```text
timeline          : legacy / transition timeline
schedule_events   : interval/index pipeline schedule
```

`schedule_events` 是 pipeline 关键。它来自 JSON plan 的 `schedule.events` 字段。

一个典型 event 逻辑上类似：

```json
{
  "kind": "load",
  "weight_id": 123,
  "anchor_op_id": 115,
  "engine": "disk",
  "duration_ms": 12.3
}
```

以及：

```json
{
  "kind": "prepare",
  "weight_id": 123,
  "anchor_op_id": 123,
  "engine": "prepare_gpu",
  "duration_ms": 36.7
}
```

注意：

- `start_ms / end_ms / duration_ms` 用于 planner cost 和 debug。
- runtime 触发不按 wall-clock 等待这些时间。
- runtime 真正使用的是 `anchor_op_id`。

## 4. PlanExecutor：把 schedule 投影成 anchor events

主要文件：

- `runtime/plan_executor.h`
- `runtime/plan_executor.cpp`

`PlanExecutor::apply(plan)` 做几件事：

1. residency reconcile
   - 对 plan 期望常驻 GPU 的 weight 调 `set_resident(weight_id, true)`。
   - 对 plan 不期望 GPU 常驻的 weight 调 evict。

2. op routing
   - 给每个 op 写入 static backend route。

3. migration metadata
   - 标记 op 是否需要跨 backend migrate。

4. build anchor index
   - 这是 pipeline 的核心。
   - `schedule_events` 或 `timeline` 被投影成：

```text
anchor_op_id -> [PlanEvent, PlanEvent, ...]
```

相关函数：

```cpp
PlanExecutor::build_anchor_index()
PlanExecutor::events_for_anchor(op_id)
```

当启用：

```text
LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE=1
LLAMA_ELASTIC_DEFER_STAGE=1
```

`apply()` 不会立刻执行 load/transfer/xform，而只是建立 anchor index。

这样可以避免 apply plan 时一次性同步加载所有 weight。

## 5. llama-context：在 op 到达时触发 stage

主要文件：

- `src/llama-context.cpp`
- `src/llama-context.h`

runtime 有两种触发 anchor 的路径：

1. pre-op index trigger
   - 当前 graph 执行到某个 op 前，调用：

```cpp
llama_context::elastic_fire_anchor_op(op_id, "pre_op_index")
```

2. weight anchor callback
   - backend 根据 weight / tensor anchor 回调到 llama context。

核心函数：

```cpp
llama_context::elastic_fire_anchor_op(int op_id, const char * reason)
```

它做的事情是：

```text
events = elastic_executor->events_for_anchor(op_id)
for event in events:
    if event.kind == LOAD:
        llama_weight_stage_request(weight_name, "load")
    if event.kind == XFORM/PREPARE:
        llama_weight_stage_request(weight_name, "prepare")
        fallback: llama_weight_transform_request(weight_name, transform_kind)
```

实现上暂时复用 `EvKind::XFORM` 来承载 `prepare` event，避免破坏旧 Plan IR enum；但 scheduler JSON 和 runtime stage request 已经使用 `prepare` 语义。

为了避免同一个 graph 中重复触发同一个 anchor，runtime 维护：

```text
elastic_anchor_fired
elastic_anchor_requests
elastic_anchor_hits
elastic_anchor_duplicates
elastic_anchor_events_fired
```

实验里为了每个 decode graph 都能重新触发 stage，使用：

```text
LLAMA_ELASTIC_ANCHOR_REPEAT_PER_GRAPH=1
```

这会在每次 graph reset / decode step 之间清理 anchor fired 状态。

## 6. WBM/OpenCL：LOAD 和 PREPARE stage

主要文件：

- `runtime/weight_buffer_manager_opencl.h`
- `runtime/weight_buffer_manager_opencl.cpp`
- `ggml/src/ggml-opencl/ggml-opencl.cpp`

当前实现有两个 planner-visible stage：

```text
LOAD     disk -> host staging
PREPARE  backend materialization for compute
```

### 6.1 stage load 请求

当 llama-context 触发：

```text
llama_weight_stage_request(weight, "load")
```

OpenCL backend 最终进入 WBM 的 load stage。

核心函数：

```cpp
wbmcl_load_host_async(wbm_opencl_ctx * octx, int idx)
```

如果开启：

```text
GGML_ELASTIC_ASYNC_STAGE_LOAD=1
```

它不会同步读 disk，而是：

```text
async_load_queue.push_back(idx)
async_load_cv.notify_one()
```

后台 worker 负责执行：

```cpp
wbmcl_load_host(octx, idx)
```

`wbmcl_load_host()` 做的事：

- 分配 / 复用 host staging buffer；
- 优先通过 direct read callback 做 O_DIRECT read；
- 失败时 fallback 到 memcpy；
- 记录 stage load timing。

### 6.2 compute 前 ensure

当 compute 真正需要该 weight 时，OpenCL path 会 ensure resident。

如果前面的 LOAD stage 已经完成：

```text
disk -> host staging
```

那么 compute path 不需要再做 direct disk read，只需要使用 staged data 继续 backend materialization。

如果 LOAD stage 还没完成，则 wait：

```cpp
wbmcl_wait_host_load(octx, idx)
```

实验日志中能看到：

```text
async load worker enqueued=...
completed=...
waits=...
wait_total=...
direct read split: async_load calls=...
foreground calls=0
```

理想情况是：

```text
foreground direct read calls = 0
wait_total 尽量小
```

这说明 direct disk read 被提前到了 compute 之前。

### 6.3 backend prepare 请求

当 llama-context 触发 prepare event 时，优先调用：

```text
llama_weight_stage_request(weight, "prepare")
```

OpenCL backend 对应：

```cpp
wbmcl_prepare_backend(wbm_opencl_ctx * octx, int idx)
```

语义：

```text
GPU/SOA weight:
    prepare = write buffer + convert + transpose / SOA materialization

generic WBM block:
    prepare = dma_to_backend + transform_backend
```

开启：

```text
GGML_ELASTIC_ASYNC_STAGE_PREPARE=1
```

时，OpenCL/SOA prepare 不在 anchor 线程里同步完成，而是：

```text
prepare anchor:
    wbmcl_soa_reload_async(idx)
    return

compute ensure:
    wbmcl_wait_soa_reload(idx)
    if not ready/resident:
        fallback reload_fn()
```

这样 `LOAD` worker、`PREPARE` worker 和前台 compute 可以形成 producer/consumer overlap：

```text
disk direct read -> host staging
backend prepare  -> GPU materialized layout
compute          -> wait only if producer has not finished
```

如果 backend 不支持 `"prepare"`，llama-context 会 fallback 到旧的：

```text
llama_weight_transform_request(...)
```

CPU_Elastic 现在也实现了 `"prepare"` stage：`LOAD` 把 raw bytes 放入 `staged_raw_by_idx`，`PREPARE` 把 staged bytes materialize 到 CPU resident buffer，并更新 runtime residency map。

开启：

```text
GGML_ELASTIC_ASYNC_STAGE_LOAD=1
GGML_ELASTIC_ASYNC_STAGE_PREPARE=1
```

时，CPU_Elastic 也有独立 worker queue：

```text
CPU load worker:
    disk/direct read -> staged_raw_by_idx

CPU prepare worker:
    wait load if needed
    staged_raw_by_idx -> CPU resident buffer

CPU compute ensure:
    wait async prepare
    wait async load
    foreground prepare only as fallback
```

因此现在 CPU/GPU 两个 backend 都是同一套 planner-visible pipeline 语义：

```text
LOAD -> PREPARE -> COMPUTE
```

区别只在 PREPARE 的 backend materialization 细节：

- GPU/OpenCL: write buffer + convert/transpose/SOA layout;
- CPU_Elastic: raw staging -> CPU resident layout.

注意：Plan IR 里暂时复用 `EvKind::XFORM` 统计 prepare，所以一些旧日志字段仍显示 `xform=...`。在新语义里这代表 prepare event，而不是单独的老 xform stage。

## 7. 当前实验默认开关

当前验证过、用于 baseline 的推荐配置：

```bash
LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE=1
LLAMA_ELASTIC_DEFER_STAGE=1
LLAMA_ELASTIC_INTERVAL_STAGE_KINDS=load,prepare

GGML_ELASTIC_ASYNC_STAGE_LOAD=1
GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD=8
GGML_ELASTIC_ASYNC_STAGE_PREPARE=1

LLAMA_ELASTIC_ANCHOR_REPEAT_PER_GRAPH=1
GGML_ELASTIC_CACHE_FOREGROUND_LOAD=1

GGML_ELASTIC_PIN_UNPLANNED_OUTPUT=0
```

matrix runner 中对应逻辑在：

```text
runtime/plan/run_dynamic_budget_matrix.py
```

当：

```text
--cp-objective interval_makespan
--use-interval-schedule 1
--interval-stage-kinds load,prepare
```

runner 会自动设置这些 env。特别地，只要 `--interval-stage-kinds` 包含 `prepare`，runner 会默认打开：

```text
GGML_ELASTIC_ASYNC_STAGE_PREPARE=1
LLAMA_ELASTIC_ENABLE_CPU_XFORM_STAGE=1
```

## 8. 为什么不再把 transfer 单独作为核心 stage

之前的拆法是：

```text
LOAD
TRANSFER
XFORM
COMPUTE
```

这个拆法对 OpenCL/SOA 不够贴近真实执行，因为 host -> GPU write、convert、transpose 经常是一个 backend materialization 过程，很难把 “transfer” 当成独立的优化目标。

现在的拆法是：

```text
LOAD
BACKEND_PREPARE
COMPUTE
```

其中 `BACKEND_PREPARE` 覆盖：

- GPU write buffer；
- GPU convert / transpose；
- OpenCL SOA materialization；
- CPU repack / transform。

旧实验入口：

```text
GGML_ELASTIC_SOA_ASYNC_RELOAD_ON_TRANSFER=1
wbmcl_soa_reload_async()
wbmcl_wait_soa_reload()
```

现在已经收敛成明确的：

```text
GGML_ELASTIC_ASYNC_STAGE_PREPARE=1
```

旧的 `GGML_ELASTIC_SOA_ASYNC_RELOAD_ON_TRANSFER` 只作为兼容 fallback。

短 smoke 中看到 async prepare 确实被提交并被 compute 等待消费：

```text
async prepare worker enqueued=175 completed=175 waits=4 wait_total=252.757 ms
```

但它是否带来最终 speedup 还需要用同一个 offline table 做固定对照。之前短窗口中 async/sync 直接比较会被 CP-SAT time-limit 的 plan 差异、动态 budget wall-clock 推进差异污染。

prepare 仍可能慢的原因是：

- SOA materialization work 本身很重；
- async worker 和 foreground compute 竞争 OpenCL queue / bandwidth；
- wait 点仍然可能落在 compute 前；
- 当前 planner 的 interval schedule 还没有精确建模 OpenCL queue contention。

所以现在的结论是：

```text
planner-visible pipeline = LOAD + PREPARE + COMPUTE
validated IO overlap     = async LOAD
prepare overlap          = async OpenCL PREPARE implemented and correctness-tested
```

## 9. pipeline 和 CP-SAT planner 的关系

pipeline runtime 本身不决定哪些 weight 放 GPU / disk。

planner 决定：

```text
placement[w] = gpu / disk_gpu / cpu / disk_cpu
```

然后根据 placement 生成 stage chain：

```text
gpu      : 常驻 GPU，不需要 per-token disk load
disk_gpu : 需要 LOAD + TRANSFER + XFORM + GPU compute
```

更新后的 planner-visible chain：

```text
gpu      : 常驻 GPU，不需要 per-token disk load/prepare
disk_gpu : 需要 LOAD + PREPARE + GPU compute
disk_cpu : 需要 LOAD + PREPARE + CPU compute
```

在当前实验中 allowed placements 通常限制为：

```text
gpu,disk_gpu
```

这样避免 CPU backend 的不完整 profile 干扰，同时聚焦 GPU memory pressure。

online CP-SAT 和 offline table 的区别：

- offline table 预先为每个 budget bucket 生成 plan；
- online remote CP-SAT 会拿 runtime state 生成 plan；
- runtime state 包括 weight 当前是否在 CPU/GPU/disk；
- online 的 provider query/solve time 在最终 `exec ms/token` 里不计入，但 `apply_ms` 和实际 movement 计入。

## 10. 关键日志怎么看

### 10.1 是否用了 interval schedule

```text
apply_exec_plan: ... schedule_events=40 schedule_used=1 load=20 transfer=0 xform=20 stage_defer=1
```

说明：

- 使用了 schedule events；
- 当前投影了 20 个 load event 和 20 个 prepare event；
- `xform=20` 是当前 legacy counter 名字，语义上是 prepare；
- `transfer=0` 表示没有 standalone transfer stage；
- stage 是 deferred 到 anchor 触发的。

### 10.2 anchor 是否触发

```text
elastic anchor summary: requests=... hits=... duplicate=... fired=... load=... transfer=0 xform=... failures=0
```

重点看：

- `failures=0`
- `load > 0`
- `transfer=0`
- `xform > 0` 在新语义下表示 prepare anchors 被触发。

### 10.3 direct read 是否提前

```text
direct read split: async_load calls=62 total=...
foreground calls=0 total=0.00 ms
async load worker enqueued=62 completed=62 waits=...
async prepare worker enqueued=... completed=... waits=...
```

重点看：

- `foreground calls=0`
- `async_load calls` 与 stage load calls 对齐；
- `wait_total` 是否较小。
- `async prepare worker enqueued/completed` 是否非零且没有 stage failure。

## 11. 当前已知限制

1. prepare overlap 还需要继续调优

现在 planner 和 runtime stage 已经支持异步 OpenCL `prepare`，但 prepare 是否真正带来 speedup，还取决于 OpenCL queue、SOA materialization、prefetch distance、plan 稳定性和 wait 点。短 smoke 已验证 correctness，不代表最终 speedup。

2. `start_ms / end_ms` 不是 runtime wall-clock

它们是 planner 诊断和 CP-SAT objective 的结果。runtime 触发依据是 `anchor_op_id`。

3. online raw timer 会包含 provider query

所以实验 summary 里需要看：

```text
exec ms/token
```

而不是只看 raw llama eval timer。

4. pipeline 效果依赖 prefetch distance

如果 `prefetch_distance` 太小，LOAD 可能来不及完成，compute 前仍会 wait。

如果太大，可能提前占用 host staging / memory bandwidth。

## 12. Review 建议重点

建议优先 review 这些点：

1. `runtime/plan_executor.cpp`
   - `schedule_events` 到 `PlanEvent` 的投影是否正确；
   - `LLAMA_ELASTIC_INTERVAL_STAGE_KINDS=load,prepare` 是否正确过滤；
   - anchor dedup key 是否会误去重。

2. `src/llama-context.cpp`
   - `elastic_fire_anchor_op()` 的触发时机；
   - anchor fired 生命周期是否按 graph reset；
   - stage request / transform request 是否正确更新 runtime state。

3. `runtime/weight_buffer_manager_opencl.cpp`
   - async load worker 生命周期；
   - host staging buffer 的复用和释放；
   - `wbmcl_wait_host_load()` 是否会产生隐藏同步；
   - `wbmcl_prepare_backend()` 是否正确覆盖 SOA materialization 和 generic block prepare；
   - direct read fallback 是否正确。

4. `ggml/src/ggml-opencl/ggml-opencl.cpp`
   - compute ensure path 是否优先使用 staged host data；
   - foreground direct read 是否已经避免；
   - `"prepare"` stage request 是否正确进入 WBM prepare；
   - SOA materialization 是否仍是主要瓶颈。

5. `runtime/plan/run_dynamic_budget_matrix.py`
   - env 是否一致；
   - `exec_ms_per_token` 统计是否符合实验定义。

## 13. Smoke test

已在 OP12 `5ae7a43d` 做 3 秒 correctness smoke：

```text
artifact: .wiki/elastic_memory/pipeline_runtime_review/artifacts/prepare_smoke_trace01_3s_v2
trace: trace_01_user_147
method: offline
stage kinds: load,prepare
```

关键日志：

```text
apply_exec_plan: ... schedule_events=40 schedule_used=1 load=20 transfer=0 xform=20 stage_defer=1
elastic anchor summary: ... load=200 transfer=0 xform=200 failures=0
direct read split: async_load calls=20 ... foreground calls=0
```

说明：

- 新 plan 生成的是 `load / prepare / compute` schedule；
- runtime 投影后没有 standalone transfer stage；
- prepare 事件通过 legacy `xform` counter 体现；
- stage request 无 failure；
- direct disk read 仍走 async load worker。

更新后的 async prepare smoke：

```text
artifact: .wiki/elastic_memory/pipeline_runtime_review/artifacts/prepare_async_after_cpu_prepare_smoke_trace01_3s
trace: trace_01_user_147
method: offline
stage kinds: load,prepare
env: GGML_ELASTIC_ASYNC_STAGE_PREPARE=1
```

关键结果：

```text
status: ok
raw ms/token: 712.49
exec ms/token: 705.48
planned evict/load/xfer/prepare: 60/60/0/60
direct read split: async_load calls=60 total=667.68 ms; foreground calls=0 total=0.00 ms
async load worker enqueued=60 completed=60 waits=280 wait_total=128.208 ms
async prepare worker enqueued=175 completed=175 waits=4 wait_total=252.757 ms
anchor failures=0
```

这个 smoke 证明的是实现路径正确：

- `load` 和 `prepare` 都 deferred 到 anchor；
- `prepare` 不是同步 fallback，而是进入 async prepare worker；
- foreground direct disk read 为 0；
- compute only waits at consumer boundary；
- 没有 stage failures。

它还不是最终性能结论。要做性能结论，需要复用同一个 offline table，对比：

```text
same plan + sync prepare
same plan + async prepare
```

否则短窗口内重新求解 offline table 可能得到不同 plan，动态 budget 也会因为 wall-clock 速度差异走到不同状态。

同 table 对照已补跑：

```text
sync artifact : .wiki/elastic_memory/pipeline_runtime_review/artifacts/prepare_sync_same_table_smoke_trace01_3s
async artifact: .wiki/elastic_memory/pipeline_runtime_review/artifacts/prepare_async_after_cpu_prepare_smoke_trace01_3s
same offline table: yes
```

结果：

```text
async prepare: exec 705.48 ms/token, direct read 667.48 ms, foreground read 0, failures 0
sync prepare : exec 693.19 ms/token, direct read 710.05 ms, foreground read 0, failures 0
```

解释：

- 两者都已经消除了 foreground direct disk read；
- async prepare 确实提交了 worker：`enqueued=175 completed=175 waits=4`；
- 但这个短窗口里 async prepare 还没有带来 speedup，主要是 prepare worker 与 foreground compute/queue/bandwidth 存在竞争，且当前 schedule 还没有精确限制 prepare 的并发和距离。

因此当前状态是：

```text
功能正确性: passed
完整路径: CPU/GPU LOAD async + CPU/GPU PREPARE async + COMPUTE wait-consume
性能状态: needs tuning, not yet a reliable speedup
```

## 14. CPU/GPU pipeline routing fix

一次 CPU placement smoke 暴露了一个重要问题：

```text
allowed placements = cpu,disk_cpu
plan schedule      = load + prepare_cpu
actual runtime     = OpenCL load worker handled "load"
```

原因是早期 `load` event 的 engine 是通用 `disk`，llama-context 发出的 stage 是 `"load"`，provider chain 里 OpenCL 先接住了这个请求。修复后：

```text
disk_cpu load -> stage "load_cpu"
disk_gpu load -> stage "load_gpu"
```

同时 runner 在 `load,prepare` 下默认设置：

```text
LLAMA_ELASTIC_ENABLE_CPU_XFORM_STAGE=1
```

否则 `prepare_cpu` 会被旧的 CPU xform stage guard 过滤掉。

## 15. CPU/GPU smoke after full pipeline

### GPU/OpenCL path

```text
artifact: .wiki/elastic_memory/pipeline_runtime_review/artifacts/full_pipeline_gpu_smoke_trace01_2s
model placement: GPU/OpenCL weights
stage kinds: load,prepare
status: ok
exec ms/token: 755.97
planned evict/load/xfer/prepare: 32/32/0/32
foreground direct read: 0
anchor failures: 0
```

Key runtime counters:

```text
async load worker enqueued=32 completed=32 waits=207 wait_total=24.901 ms
async prepare worker enqueued=116 completed=116 waits=0 wait_total=0.111 ms
```

### CPU_Elastic path

This smoke forces CPU weights with trailing `-ngl 0`:

```text
artifact: .wiki/elastic_memory/pipeline_runtime_review/artifacts/full_pipeline_cpu_ngl0_smoke_trace01_1s
model placement: CPU_Elastic model buffer, offloaded 0/33 layers to GPU
stage kinds: load,prepare
status: ok
exec ms/token: 1653.89
planned evict/load/xfer/prepare: 224/110/0/55
anchor failures: 0
```

Key runtime counters:

```text
CPU_Elastic model buffer size = 4437.80 MiB
direct_read  : calls=262 ok=262 total=4518.96 ms MB=4761.0
stage_load   : calls=55 ok=55 total=1227.07 ms MB=992.2
stage_xform  : calls=262 ok=262 total=4639.02 ms MB=4761.0
async_load   : enqueued=55 completed=55 waits=43 wait_total=699.53 ms
async_prepare: enqueued=55 completed=55 waits=1 wait_total=148.97 ms
```

This verifies that CPU load and CPU prepare are now both routed to CPU_Elastic and consumed by CPU compute.

## 16. Pipeline vs no-pipeline tests

After enabling CPU/GPU pipeline support, I ran OP12 GPU-path comparisons with the same trace window and the same offline table.

Important metric note:

```text
exec_ms_per_token = (eval_ms_total - provider_get_ms_total + apply_ms_total) / eval_runs
```

This matters because no-pipeline mode can move a large amount of weight movement into `apply_exec_plan` before llama's eval timer, while pipeline mode moves that work into decode anchors. The runner has been fixed to include `apply_ms_total` in future `exec_ms_per_token`.

### Trace01, 8s window, GPU path

Artifacts:

```text
load-only pipeline: .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_vs_none_gpu_trace01_8s_load_only
no pipeline       : .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_vs_none_gpu_trace01_8s_off
full prepare async: .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_vs_none_gpu_trace01_8s_on
```

Results:

```text
load-only pipeline:
    raw eval       = 318.82 ms/token
    provider/apply = 34.66 / 1.41 ms
    direct read    = 670.90 ms, foreground read = 0
    reload calls   = 586

no pipeline:
    raw eval       = 452.91 ms/token
    provider/apply = 32.99 / 1722.53 ms
    direct read    = 4259.20 ms, foreground read = 437 calls
    reload calls   = 377
```

Conclusion:

```text
load-only pipeline wins clearly on trace01.
It eliminates foreground disk read and avoids the large apply-time movement.
```

Full async prepare on this trace was correct but slower:

```text
full prepare async raw eval = 473.23 ms/token
async prepare worker enqueued=394 completed=394 waits=3
```

This means the current GPU prepare worker is over-eager: it reduces foreground IO, but it also competes with foreground compute/OpenCL queues.

### Trace03, 8s window, GPU path

Artifacts:

```text
load-only pipeline: .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_vs_none_gpu_trace03_8s_load_only
no pipeline       : .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_vs_none_gpu_trace03_8s_off
```

Results:

```text
load-only pipeline:
    raw eval       = 103.64 ms/token
    provider/apply = 36.34 / 1.30 ms
    direct read    = 207.66 ms, foreground read = 0
    reload calls   = 107

no pipeline:
    raw eval       = 100.36 ms/token
    provider/apply = 31.79 / 758.46 ms
    direct read    = 1044.65 ms, foreground read = 61 calls
    reload calls   = 61
```

Conclusion:

```text
raw eval alone is slightly better for no-pipeline on trace03,
but that is because no-pipeline moved ~758 ms into apply time.
With apply included, load-only pipeline is better.
```

### Current performance conclusion

```text
Stable speedup path today:
    async LOAD pipeline + foreground/on-demand backend prepare from staged bytes

Correct but not yet fast enough:
    eager async GPU PREPARE worker
```

The next performance task is not to remove prepare pipeline, but to make GPU prepare queue-aware:

- do not flood OpenCL queue while compute is active;
- pace prepare by anchor distance and budget pressure;
- only prepare weights that are likely to survive eviction until use;
- avoid repeatedly preparing the same disk_gpu weights across decode graphs.

An experimental `GGML_ELASTIC_ASYNC_PREPARE_MAX_PENDING` cap was tried, but a naive "skip enqueue when full" policy can break SOA reload correctness. Its default is therefore `0` (unlimited) until queue-aware fallback is implemented correctly.

## 17. 最简 mental model

可以把当前 pipeline 理解成：

```text
CP-SAT / fallback schedule:
    weight w should be loaded at op i
    weight w should be prepared at op j

PlanExecutor:
    anchor[i].append(load(w))
    anchor[j].append(prepare(w))

llama-context:
    when op i is reached:
        request load(w)
    when op j is reached:
        request prepare(w)

OpenCL WBM:
    async worker reads w from disk into host staging
    prepare materializes backend-ready representation

later compute:
    if backend representation ready:
        avoid foreground disk read
    else:
        wait or fallback ensure
```

当前已经把 planner/runtime 抽象对齐到 `LOAD + PREPARE + COMPUTE`。下一步性能优化重点是让 `prepare` 的 SOA write/convert/transpose 更稳定地与 compute/IO overlap，并让 CP-SAT objective 反映 OpenCL queue contention。

## 18. 最新修正：plan-driven eviction + paced prepare

### 问题

之前 `load,prepare` pipeline 有两个混在一起的问题：

```text
1. backend auto eviction 仍然可能在 graph_compute 内部按 budget 驱逐权重；
2. plan-level PREPARE 直接走 lower-level SOA async reload queue，容易一次性提交过多 GPU transform。
```

第一个问题会破坏 plan 的 residency 语义。CP plan 已经决定哪些 weight 在 GPU、哪些在 disk；如果 backend 又按自己的 LRU/budget 自动驱逐，可能把 plan 认为应该 GPU resident 的权重驱逐掉，后续 compute 再 ensure 时就会出现错误或者额外 reload。

因此 interval schedule / plan-driven pipeline 下现在默认：

```text
GGML_ELASTIC_NO_AUTO_EVICT=1
```

也就是说：

```text
plan apply / plan anchors = residency authority
backend ensure           = correctness fallback only
backend auto eviction    = disabled in this mode
```

第二个问题是 GPU prepare 的 pacing。OpenCL Q4_0 SOA prepare 实际包含：

```text
host staging source
clEnqueueWriteBuffer
convert kernel
optional Adreno transpose kernels
final queue dependency for compute
```

如果把很多 prepare 全部提前提交，虽然它是 async 的，但它会和 decode kernels 竞争 GPU/device memory bandwidth。最新实现把 OpenCL `"prepare"` stage 改为走 paced async transform worker：

```text
GGML_ELASTIC_ASYNC_XFORM_MAX_INFLIGHT=1 by default
```

这样 PREPARE 仍然是 pipeline 的一部分，但同时限制 in-flight transform 数量，避免 transform 洪峰拖慢 foreground compute。

### Trace01 8s smoke after fix

Artifacts:

```text
pipeline paced prepare:
  .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_vs_none_gpu_trace01_8s_prepare_paced1_noautoevict

no interval pipeline, same no-auto-evict:
  .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_vs_none_gpu_trace01_8s_nopipe_new
```

Results:

```text
pipeline paced prepare:
    raw eval       = 84.63 ms/token
    exec eval      = 84.34 ms/token
    provider/apply = 29.49 / 1.66 ms
    direct read    = 636.42 ms, foreground read = 0
    stage xform    = 25 calls, 791.01 ms total

no interval pipeline:
    raw eval       = 77.26 ms/token
    exec eval      = 86.13 ms/token
    provider/apply = 32.72 / 902.17 ms
    direct read    = 616.73 ms, foreground read = 0
    stage xform    = 60 calls, 223.89 ms total
```

Interpretation:

```text
raw eval alone is misleading here.
No-pipeline does most movement in apply time, so raw eval looks slightly faster.
The fair execution metric is:

    exec_ms_per_token = (eval_ms_total - provider_get_ms_total + apply_ms_total) / eval_runs

Under that metric, paced prepare pipeline is slightly faster:

    84.34 ms/token vs 86.13 ms/token
```

This is the first configuration where full `LOAD + PREPARE` pipeline is both correct and faster than the no-pipeline apply-heavy path on trace01.

### Trace01 30s validation

Artifacts:

```text
pipeline paced prepare:
  .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_vs_none_gpu_trace01_30s_prepare_paced1

no interval pipeline:
  .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_vs_none_gpu_trace01_30s_nopipe_new
```

Results:

```text
pipeline paced prepare:
    raw eval       = 83.21 ms/token
    exec eval      = 83.08 ms/token
    provider/apply = 47.47 / 1.996 ms
    direct read    = 651.77 ms, foreground read = 0
    stage xform    = 25 calls, 771.45 ms total
    failures       = 0

no interval pipeline:
    raw eval       = 80.74 ms/token
    exec eval      = 83.49 ms/token
    provider/apply = 64.94 / 1058.09 ms
    direct read    = 725.67 ms, foreground read = 0
    stage xform    = 70 calls, 266.28 ms total
    failures       = 0
```

Interpretation:

```text
The 30s result is close, but still favors the paced pipeline when apply time is included:

    83.08 ms/token vs 83.49 ms/token

The larger architectural difference is apply cost:

    pipeline apply ~= 2 ms
    no-pipeline apply ~= 1058 ms

So the pipeline path has moved movement/materialization out of budget-switch apply
and into overlapped per-op anchors.  This is the behavior we want before comparing
online/offline/MRU over longer traces.
```

## 19. Strict disk residency mode

The fast results above use plan-driven eviction at plan switch time, with backend
auto eviction disabled.  That fixes the earlier incorrect interaction where the
backend could evict a GPU-resident plan weight.  However, it also means a
`disk_gpu` weight may remain GPU-resident after it is first prepared inside a
plan window, until the next plan switch evicts it.

For strict budget semantics, interval runtime now supports synthetic disk-weight
evict events:

```text
LLAMA_ELASTIC_INTERVAL_EVICT_DISK_AFTER_USE=1
    Insert EVICT at consumer_op + 1.

LLAMA_ELASTIC_INTERVAL_EVICT_DISK_AT_GRAPH_END=1
    Insert EVICT at the last op anchor instead of immediately after use.
```

Implementation:

```text
PlanExecutor:
    for each plan weight with location=disk:
        add LOAD/PREPARE before consumer
        add EVICT after consumer or at graph end

llama-context anchor:
    EVICT -> llama_weight_movement_request(weight, evict=true)
           -> runtime map marks the weight as DISK
```

Trace01 8s strict results:

```text
after-use evict:
    artifact  = .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_strict_evict_trace01_8s
    raw eval  = 400.58 ms/token
    exec eval = 398.94 ms/token
    direct read foreground = 0
    reload calls = 597

graph-end evict:
    artifact  = .wiki/elastic_memory/pipeline_runtime_review/artifacts/pipeline_graphend_evict_trace01_8s
    raw eval  = 357.17 ms/token
    exec eval = 355.71 ms/token
    direct read foreground = 0
    reload calls = 617
```

Interpretation:

```text
Strict mode is much slower than relaxed mode because disk_gpu weights are
materialized every forward again.  Disk read itself is mostly hidden or cached
in host staging; the remaining cost is repeated GPU materialization
(write/convert/transpose) and eviction synchronization.

Graph-end evict is better than after-use evict because it avoids evicting in
the middle of the decode graph, but it still pays the per-forward materialize
cost.
```

Current recommendation:

```text
Use strict mode when validating memory-budget semantics.
Use relaxed mode only as a performance upper bound for the pipeline machinery.

For final online/offline/MRU comparison, results should state explicitly which
mode is used.  The next optimization target is reducing repeated GPU
materialization in strict mode, not disk read.
```

## 20. 2026-06-22 static strict speed pass

This pass focused on one fixed 4096 MiB static plan on OP12, not dynamic traces.
The important runtime switches for a *valid strict per-graph pipeline* are:

```text
LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE=1
LLAMA_ELASTIC_ANCHOR_REPEAT_PER_GRAPH=1
LLAMA_ELASTIC_PREPARE_LEAD_OPS=32
GGML_ELASTIC_ASYNC_STAGE_LOAD=1
GGML_ELASTIC_ASYNC_STAGE_PREPARE=1
GGML_ELASTIC_ASYNC_XFORM_MAX_PENDING=32
GGML_ELASTIC_CL_RETAIN=1
GGML_ELASTIC_CL_RETAIN_MB=1024
```

Two easy-to-miss pitfalls:

```text
Without LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE=1:
    the plan loads schedule=280 from JSON, but PlanExecutor projects
    schedule_events=0 and falls back to the transition timeline.

Without LLAMA_ELASTIC_ANCHOR_REPEAT_PER_GRAPH=1:
    anchors fire only once; the run looks close to compute-only because
    strict disk weights are not rematerialized each decode graph.
```

CPU Q4_0 pretransform was implemented as an experimental path:

```text
GGML_ELASTIC_CPU_PRETRANSFORM_Q4_0=1

raw q4_0 AOS -> CPU NEON q/d SOA -> clEnqueueWriteBuffer(q,d)
then optional Adreno transpose remains GPU-side.
```

The CPU path now skips the raw staging allocation in this mode, so
`detail staging_alloc` is 0 for CPU-pretransform reloads.  It is still slower
than the GPU materialization path on OP12:

```text
CPU pretransform + strict schedule:
    eval ~= 600 ms/token
    stage xform avg ~= 15.7 ms
    cpu_xform avg ~= 4.6 ms/call
    q/d write avg ~= 6.1-6.7 ms/call

CPU pretransform + GGML_ELASTIC_NO_TRANSPOSE=1:
    still ~= 600 ms/token
```

Interpretation:

```text
CPU xform itself can overlap GPU compute in the microbench, but the runtime
path still pays q/d host->GPU writes.  Those writes compete with GPU compute
and dominate enough that CPU pretransform does not beat the existing GPU
convert/transpose path for 8B Q4_0 on OP12.
```

The best static strict result in this pass came from the GPU pipeline with
explicit retain-pool capacity and aggressive SOA reuse:

```text
GGML_ELASTIC_CL_RETAIN=1
GGML_ELASTIC_CL_RETAIN_MB=1024
GGML_ELASTIC_POOL_PARENT_ONLY=0
GGML_ELASTIC_EVICT_WAIT=0
```

4-second OP12 smoke, 8B Q4_0, 4096 MiB fixed plan:

```text
eval time              ~= 281.0 ms/token
stage xform            312 calls, total ~= 275.6 ms, avg ~= 0.88 ms
pool stats             soa_hit=456, soa_miss=0, parent_create=0
foreground direct read 0
```

8-second repeat:

```text
eval time              ~= 288.0 ms/token
stage xform            559 calls, total ~= 467.3 ms, avg ~= 0.84 ms
pool stats             soa_hit=820, soa_miss=0, parent_create=0
foreground direct read 0
```

Retain-cap sweep:

```text
1024 MB: ~= 281 ms/token, parent_create=0
1536 MB: ~= 287 ms/token, parent_create=0
2048 MB: ~= 285 ms/token, parent_create=0
3072 MB: ~= 286 ms/token, parent_create=0
```

So 1024 MB is enough for this fixed plan.  If `GGML_ELASTIC_CL_RETAIN_MB` is
left on auto, the backend currently chooses roughly one max tensor size; in
this strict pipeline that caused repeated parent creation and slower decode.

The safer profile keeps parent-only SOA reuse and normal evict waits, but still
sets the retain cap explicitly:

```text
GGML_ELASTIC_CL_RETAIN=1
GGML_ELASTIC_CL_RETAIN_MB=1024
```

4-second OP12 smoke:

```text
eval time              ~= 313.3 ms/token
stage xform            274 calls, total ~= 254.7 ms, avg ~= 0.93 ms
pool stats             soa_hit=400, soa_miss=0, parent_create=0
```

This is the recommended correctness-first static profile to compare against
the older ~316 ms/token pipeline result.  The aggressive profile is a speed
ceiling until full SOA triple reuse and no-wait evict are revalidated.

Correctness note:

```text
GGML_ELASTIC_POOL_PARENT_ONLY=0 and GGML_ELASTIC_EVICT_WAIT=0 are aggressive
performance switches.  Earlier OP12 notes marked full q/d sub-buffer reuse as
diagnostic because Adreno can be sensitive to stale sub-buffer/image state.
Use this profile for speed experiments; keep the safer parent-only / evict-wait
profile for final correctness runs unless the aggressive profile is revalidated
with longer generations and output checks.
```

## 21. Fixed mixed CPU/GPU plan: pipeline vs no-overlap

The desired experiment is not "CPU transform beats GPU transform".  The desired
experiment is:

```text
same fixed plan
same CPU/GPU routing
same disk/load/prepare work
pipeline enabled  >  foreground synchronous load/prepare
```

The existing 4096 MiB table plan is not a mixed CPU/GPU compute plan:

```text
plan_4096MiB.json:
    op backends = gpu:224
    weight locations = gpu:196, disk:28
```

So I generated a fixed mixed plan for this test by disabling `disk_gpu` and
allowing CPU fallback:

```text
python3 runtime/plan/dynamic_budget_solver.py \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir runtime/plan/profiles/android-opencl/Meta-Llama-3-8B-Instruct-Q4_0_profiled_trace01 \
  --budget-mib 4096 --kv-mib 512 --misc-mib 256 --safety-mib 64 \
  --time-limit-ms 1000 --prefetch-distance 4 \
  --overlap-model pipeline --cp-objective interval_makespan \
  --allowed-placements cpu,gpu,disk_cpu --allow-cpu-fallback \
  --out /tmp/mixed_4096.json
```

Generated plan:

```text
weights: cpu=58, disk=28, gpu=138
ops:     cpu=86, gpu=138
schedule engines:
    compute_cpu=86
    compute_gpu=138
    disk=28
    prepare_cpu=28
```

It was pushed to OP12 as:

```text
/data/local/tmp/hyzheng/elastic/plan_mixed_cpu_gpu_4096MiB.json
```

Pipeline runtime profile:

```text
LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE=1
LLAMA_ELASTIC_ANCHOR_REPEAT_PER_GRAPH=1
LLAMA_ELASTIC_INTERVAL_STAGE_KINDS=load,prepare
LLAMA_ELASTIC_ENABLE_CPU_XFORM_STAGE=1
LLAMA_ELASTIC_PREPARE_LEAD_OPS=32
GGML_ELASTIC_ASYNC_STAGE_LOAD=1
GGML_ELASTIC_ASYNC_STAGE_PREPARE=1
GGML_ELASTIC_ASYNC_XFORM_MAX_PENDING=32
GGML_ELASTIC_CL_RETAIN=1
GGML_ELASTIC_CL_RETAIN_MB=1024
```

No-overlap runtime profile:

```text
same plan, same routing, same load/prepare stage kinds
GGML_ELASTIC_ASYNC_STAGE_LOAD=0
GGML_ELASTIC_ASYNC_STAGE_PREPARE=0
```

OP12 8-second fixed-plan result:

```text
pipeline:
    eval        ~= 473.3 ms/token
    generated   18 tokens in 8 s window
    load calls  10 async-stage calls
    xform calls 498 async transform calls

no-overlap:
    eval        ~= 895.5 ms/token
    generated   10 tokens in 8 s window
    load calls  560 foreground/synchronous stage calls
    xform calls 280 foreground/synchronous stage calls
```

Interpretation:

```text
For a fixed mixed CPU/GPU plan, enabling the staged pipeline is about 1.9x faster
than doing the same stage work synchronously at the anchors.

This validates the runtime effect we want: disk/load/prepare can be overlapped
with ongoing compute and should be scheduled ahead of use.  The result is not a
claim that CPU transform is faster than GPU transform; it is a pipeline vs
no-overlap comparison under the same CPU/GPU placement plan.
```

## 22. Same-budget all-GPU compute plan

Do not confuse the no-budget full-offload baseline with a same-budget all-GPU
compute plan.

No-budget upper bound:

```text
-ngl 99, no elastic plan
all weights resident/offloaded
eval ~= 73.8 ms/token
```

Same 4096 MiB budget all-GPU compute plan:

```text
plans_matrix10min/plan_4096MiB.json
ops:     gpu=224
weights: gpu=196, disk=28
schedule engines:
    compute_gpu=224
    disk=28
    prepare_gpu=28
```

This plan still has disk load and GPU prepare because the budget cannot hold
all weights resident.

OP12 8-second fixed-plan result, safe retain profile:

```text
pipeline:
    eval        ~= 316.1 ms/token
    generated   27 tokens in 8 s window
    stage load  28 calls,  foreground direct read = 0
    stage xform 521 calls
    pool        parent_create=0

no-overlap:
    eval        ~= 968.1 ms/token
    generated   10 tokens in 8 s window
    stage load  560 calls, foreground direct read ~= 634 ms
    stage xform 280 calls
    pool        parent_create=0
```

Interpretation:

```text
Under the same 4096 MiB budget, "pure GPU compute" is not the same as full
resident GPU.  The plan has disk_gpu weights, and pipeline is what hides most
of their load/prepare latency.

This same-budget all-GPU compute comparison shows the pipeline benefit even
more clearly than the mixed CPU/GPU plan: ~=316 ms/token with pipeline vs
~=968 ms/token with synchronous no-overlap.
```

## 23. Runtime-calibrated CP-SAT replan

Question:

```text
The online CP-SAT planner was designed to construct plans at runtime.  Are the
old plans/costs still correct after the pipeline/runtime changes?
```

Finding:

```text
The old 4096 MiB all-GPU plan is valid, but its cost model is stale.
```

Old plan:

```text
plans_matrix10min/plan_4096MiB.json
pred_per_token_ms ~= 1666 ms
actual pipeline   ~= 316 ms/token
placement         gpu=196, disk=28
```

The biggest cost-model issues were:

```text
1. disk_gpu_reload_multiplier=4.0 over-penalizes disk_gpu for the current
   staged pipeline runtime.

2. OpenCL TRANSFER in the old fine-stage profile is host enqueue time
   (~0.001 ms), not the true device/materialization cost.

3. Current retain-pool runtime makes GPU prepare much cheaper than the old
   profile: parent_create=0 and SOA pool hits dominate.

4. The old plan predicts resource time, but real decode depends heavily on
   anchor lead and whether stage work is hidden before use.
```

I built a runtime-calibrated profile from the latest OP12 observations:

```text
artifact:
.wiki/elastic_memory/pipeline_runtime_review/artifacts/op12_calibrated_replan_20260622/runtime_calibrated_profile

LOAD:
    0.15 ms + bytes / 2000 MiB/s

OpenCL TRANSFER:
    0.02 ms

OpenCL XFORM/PREPARE:
    0.10 ms + bytes / 18000 MiB/s

CPU_Elastic XFORM:
    kept from measured profile
```

Then I solved a new same-budget all-GPU compute plan:

```text
python3 runtime/plan/dynamic_budget_solver.py \
  --model-meta runtime/plan/model_meta/Meta-Llama-3-8B-Instruct-Q4_0.weights_ops.json \
  --cost-dir .wiki/elastic_memory/pipeline_runtime_review/artifacts/op12_calibrated_replan_20260622/runtime_calibrated_profile \
  --budget-mib 4096 --kv-mib 512 --misc-mib 256 --safety-mib 64 \
  --time-limit-ms 1000 --prefetch-distance 64 \
  --transition-weight 0.0 \
  --disk-reload-multiplier 1.0 --disk-gpu-reload-multiplier 1.0 \
  --overlap-model pipeline --cp-objective interval_makespan \
  --allowed-placements gpu,disk_gpu \
  --out plan_allgpu_lead64_4096MiB.json
```

New plan:

```text
artifact:
.wiki/elastic_memory/pipeline_runtime_review/artifacts/op12_calibrated_replan_20260622/plan_allgpu_lead64_4096MiB.json

pred_per_token_ms ~= 444 ms
placement         gpu=200, disk=24
schedule events   compute_gpu=224, disk=24, prepare_gpu=24
disk bytes        481.5 MiB
```

The new plan moved from the old contiguous early-layer disk set to a distributed
disk set across later layers, while keeping the same disk byte budget.

Lead sweep on OP12, 4-second smokes:

```text
lead=16:
    eval ~= 317.2 ms/token

lead=32:
    eval ~= 312.4 ms/token

lead=64:
    eval ~= 305.6 ms/token
```

8-second OP12 run for lead=64:

```text
pipeline:
    eval        ~= 310.7 ms/token
    stage load  24 calls, 481.5 MiB, foreground direct read = 0
    stage xform 466 calls, 9022.5 MiB
    pool        parent_create=0

no-overlap, same new plan:
    eval        ~= 810.8 ms/token
    stage load  288 calls, foreground direct read ~= 274 ms
    stage xform 144 calls
    pool        parent_create=0
```

Interpretation:

```text
Re-solving with a runtime-calibrated cost model gives a small but real
improvement over the old same-budget all-GPU plan:

    old plan pipeline ~= 316.1 ms/token
    new plan pipeline ~= 310.7 ms/token

The improvement is not huge because both plans stream the same total disk bytes
and the pipeline already hides disk load well.  The remaining gap is dominated
by prepare/write/transpose pressure and graph compute.

The much larger difference remains pipeline vs no-overlap:

    new plan pipeline   ~= 310.7 ms/token
    new plan no-overlap ~= 810.8 ms/token
```

Next planner fix:

```text
The online CP-SAT solver should be fed current runtime-calibrated stage costs
and current residency state.  It should also prefer schedule anchor leads that
are validated on-device, because anchor placement changes real performance even
when placement is unchanged.
```

## 24. 3-minute dynamic trace: online/offline/MRU/static-min

User-requested comparison:

```text
Run decode for 3 minutes.
Replay the budget trace at real decode time (1x), not compressed.
Do not count online solver time in the final execution speed.

Expected ordering:
    online > offline > MRU > static-min

Reason:
    online should be better than offline because it solves using current runtime
    weight residency, not just budget.
```

Trace window:

```text
source trace:
    trace/tmp_3min_matrix/trace_01_user_147.csv

runner output, offline/online:
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/trace01_3min_online_offline_mru_static_calibrated

runner output, MRU/static-min no-overlap:
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/trace01_3min_mru_static_nooverlap_calibrated

source span:
    180 seconds

replay speedup:
    1.0x

bench seconds:
    180

budget:
    min=3752 MiB, mean=4862 MiB, max=5544 MiB, min bucket=3584 MiB
```

Method policy:

```text
offline:
    calibrated CP-SAT table + interval pipeline

online:
    remote CP-SAT + current runtime residency state + interval pipeline
    provider/solver time is subtracted from exec ms/token

MRU:
    no-overlap baseline
    LLAMA_ELASTIC_INTERVAL_STAGE_KINDS=none
    GGML_ELASTIC_ASYNC_STAGE_LOAD=0
    GGML_ELASTIC_ASYNC_STAGE_PREPARE=0

static-min:
    fixed min-budget plan, no-overlap baseline
```

Result:

```text
method      raw ms/token   exec ms/token   notes
online          131.63          125.64     15 online solves, 0 failures
offline         132.50          132.43     5 applies
MRU             134.22          134.13     no-overlap baseline
static-min      971.11          971.11     no-overlap, fixed 3584 MiB plan
```

Stage / movement diagnostics:

```text
online:
    provider_get_ms_total ~= 7522 ms
    remote_wall_ms        ~= 15025 ms
    remote_server_ms      ~= 14819 ms
    apply_count           = 5
    direct_read_ms        ~= 608 ms
    direct_read_calls     = 74
    reload_calls          = 1991
    anchor load/xform     = 4508 / 2254

offline:
    provider_get_ms_total ~= 102 ms
    apply_count           = 5
    direct_read_ms        ~= 644 ms
    direct_read_calls     = 74
    reload_calls          = 3821
    anchor load/xform     = 7648 / 3824

MRU no-overlap:
    direct_read_ms        ~= 28864 ms
    direct_read_calls     = 3388
    anchor load/xform     = 0 / 0

static-min no-overlap:
    direct_read_ms        ~= 88981 ms
    direct_read_calls     = 10285
    anchor load/xform     = 0 / 0
```

Interpretation:

```text
The ordering is the desired one by exec ms/token:

    online (125.64) > offline (132.43) > MRU (134.13) >> static-min (971.11)

Online beats offline because it generates plans from current residency state.
That shows up in much less repeated reload work:

    online reload_calls = 1991
    offline reload_calls = 3821

The win is about 5% on this 3-minute window, not "much better".  The likely
reason is this trace rises quickly into higher budget buckets, and many later
offline budget switches are execution-equivalent, so the offline table is not
punished much after the initial low-budget phase.

MRU is close to offline in raw speed here despite no overlap, but it performs
much more foreground direct read.  It is not a better planner; this window is
generous enough that MRU avoids the catastrophic static-min behavior.

static-min is the expected lower baseline: it stays at the min 3584 MiB plan
while available memory rises, so it repeatedly reloads far more data.
```

Next test to make online's advantage larger:

```text
Use a trace window with repeated budget drops/rises or a longer low-budget
phase, and keep online state-aware while offline remains budget-table-only.
That should amplify the benefit of preserving current residency.
```

## 25. 3-minute oscillating trace: residency-aware online advantage

The first 3-minute window above was a useful correctness check, but it rose
quickly into higher memory budgets.  That made the offline budget table look
stronger than it should: after the initial low-budget phase many table switches
were execution-equivalent.

To stress the difference between stateless budget plans and current-residency
planning, I scanned the local traces for a 180 second window with repeated
bucket changes.  The strongest candidate was:

```text
source trace:
    trace/traces_small/trace_06_user_204.csv

window:
    source start ~= 1002 s, rebased to t=0
    written to trace/tmp_3min_oscillating_std/trace_06_user_204.csv

budget:
    min=4008 MiB, mean=4616 MiB, max=5706 MiB, min bucket=3840 MiB

bucket movement:
    69 bucket changes in 180 seconds
    35 drops, 34 rises
```

Run policy stayed the same as section 24:

```text
online/offline:
    calibrated CP-SAT + interval pipeline
    LLAMA_ELASTIC_INTERVAL_STAGE_KINDS=load,prepare
    GGML_ELASTIC_ASYNC_STAGE_LOAD=1
    GGML_ELASTIC_ASYNC_STAGE_PREPARE=1

MRU/static-min:
    no-overlap baseline
    LLAMA_ELASTIC_INTERVAL_STAGE_KINDS=none
    GGML_ELASTIC_ASYNC_STAGE_LOAD=0
    GGML_ELASTIC_ASYNC_STAGE_PREPARE=0

online metric:
    exec ms/token subtracts provider_get/query time
    plan apply and real movement are still counted
```

Artifacts:

```text
online/offline:
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/trace06_3min_osc_online_offline_calibrated

MRU/static-min no-overlap:
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/trace06_3min_osc_mru_static_nooverlap_calibrated
```

Result:

```text
method       raw ms/token   exec ms/token   notes
online           145.78           83.96     47 remote solves, 0 failures
MRU              148.57          147.75     no-overlap baseline
offline          150.60          150.61     stateless budget table
static-min       733.32          733.32     fixed 3840 MiB plan, no-overlap
```

Movement diagnostics:

```text
online:
    provider_get_ms_total ~= 51346 ms
    remote_wall_ms        ~= 57400 ms
    remote_server_ms      ~= 56695 ms
    apply_count           = 37
    reload_calls          = 2295
    direct_read_ms        ~= 341 ms
    direct_read_calls     = 38
    anchor load/xform     = 5020 / 2510

offline:
    provider_get_ms_total ~= 95 ms
    apply_count           = 45
    reload_calls          = 5223
    direct_read_ms        ~= 344 ms
    direct_read_calls     = 38
    anchor load/xform     = 10510 / 5255

MRU no-overlap:
    provider_get_ms_total ~= 1003 ms
    apply_count           = 41
    reload_calls          = 5016
    direct_read_ms        ~= 44599 ms
    direct_read_calls     = 5016

static-min no-overlap:
    reload_calls          = 9386
    direct_read_ms        ~= 86356 ms
    direct_read_calls     = 9386
```

Interpretation:

```text
This is the desired planner ordering:

    online (83.96 exec ms/token) > MRU (147.75) ~= offline (150.61) >> static-min (733.32)

Online wins because it solves from the current runtime residency state.  In this
oscillating window, that matters a lot: online performs less than half the
reload work of offline:

    online reload_calls = 2295
    offline reload_calls = 5223

The same pattern appears in planned anchors:

    online anchor load/xform = 5020 / 2510
    offline anchor load/xform = 10510 / 5255

MRU is close to offline in raw speed, but it is not doing the same work profile.
Because it has no pipeline overlap, it pays large foreground direct-read time:

    MRU direct_read_ms ~= 44.6 seconds

Static-min is the lower bound: it ignores the extra budget available during
most of the window and repeatedly reloads from the fixed 3840 MiB plan.

One caveat: online's raw ms/token is only slightly better than offline because
the current remote CP-SAT provider is synchronous and expensive here:

    provider_get_ms_total ~= 51.3 seconds

Under the requested metric, solver time is excluded.  For an end-to-end mobile
runtime, the next engineering step is to make online solving incremental,
cached, or asynchronous so this provider time does not block decode.
```

### 25.1 Constant 3840 MiB MRU sanity check

The oscillating trace above spends about half its time at bucket >= 4608 MiB,
where usable weight budget is already enough for all managed Q4_0 weights:

```text
managed weights = 3744 MiB
reserved KV/misc/safety = 512 + 256 + 64 = 832 MiB

raw bucket 3840 -> usable 3008 MiB
raw bucket 4096 -> usable 3264 MiB
raw bucket 4352 -> usable 3520 MiB
raw bucket 4608 -> usable 3776 MiB, enough for all 3744 MiB
```

So I also ran a fixed 3840 MiB trace for 60 seconds, with MRU and static-min
both using no-overlap mode:

```text
trace:
    trace/tmp_const3840/trace_06_user_204.csv

artifact:
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/const3840_mru_static_nooverlap_60s
```

Result:

```text
method       raw ms/token   exec ms/token   reload_calls   direct_read_ms
MRU              470.62          470.52          4902          38922
static-min       729.31          729.31          3192          29236
```

MRU is no longer near the ~150 ms/token mixed-budget result.  It becomes slow
under sustained strict pressure, which is the expected direction.

Why is MRU still faster than static-min at the same 3840 MiB bucket?  Both keep
the same amount resident:

```text
MRU:        gpu 3006 MiB, disk 738 MiB
static-min: gpu 3006 MiB, disk 738 MiB
```

But they choose very different disk sets.  MRU places mostly the tail layers on
disk:

```text
MRU disk set:
    mostly blk.25 - blk.31
```

The static CP plan places scattered early/middle weights on disk:

```text
static-min disk set:
    blk.0, blk.1, blk.2, blk.3, blk.4, blk.7, blk.8, blk.15 - blk.18, plus some blk.31
```

The disk-set overlap is only 5 / 38 weights.  This likely changes the foreground
reload access pattern: MRU's disk weights are more contiguous in layer/model
order, while the static plan creates more scattered reads.  That means MRU can
be faster without having pipeline overlap, but it is still far slower than a
good pipelined/state-aware plan under strict budget.
```

### 25.2 True runtime MRU fix and rerun

The previous `mru` baseline was not a real runtime MRU cache.  It generated a
static resident set by sorting weights in op order and usually left tail layers
on disk.  Those old MRU numbers should not be used as the MRU baseline.

The new true MRU path is enabled by:

```text
LLAMA_ELASTIC_ONLINE_MODE=mru-cache
LLAMA_ELASTIC_RUNTIME_MRU_CACHE=1
```

It updates recency on every weight access, loads a missed weight before the op,
and evicts the most recently used resident weight other than the current op
input when the logical resident bytes exceed the current usable budget.

Important bug fixed during this pass: a disk miss originally called the generic
`llama_weight_movement_request(name, false)`.  For Q4_0 OpenCL SOA tensors that
can report success without rebuilding the `q/d/parent` buffers needed by the
matmul image path.  The symptom was:

```text
clCreateImage failed tensor=blk.25.ffn_gate.weight q=0x0 d=0x0 parent=0x0
```

The true MRU miss path now first calls `stage_request("prepare")`, falling back
only if the backend does not support staged prepare.  This forces the same
Q4_0 materialization path used by offline/online pipeline plans.

Quant mapping was also fixed for native and Python plan generation: model meta
uses numeric `quant="2"` for Q4_0, so `2` now maps to `gpu_convert`.

#### Dynamic 3-minute trace, min bucket 3840 MiB

Source window:

```text
trace/tmp_user147_bucket3840_3min/trace_01_user_147.csv
min/mean/max = 3989.8 / 4762.0 / 6198.6 MiB
min bucket   = 3840 MiB
```

Artifacts:

```text
.wiki/elastic_memory/pipeline_runtime_review/artifacts/trace01_bucket3840_3min_true_mru_pipeline_online_offline
.wiki/elastic_memory/pipeline_runtime_review/artifacts/trace01_bucket3840_3min_true_mru_nooverlap_mru_min
```

Results:

```text
method       mode         raw ms/token   exec ms/token   direct reads   direct read MB
online       pipeline          124.51          111.25            38           738.0
offline      pipeline          130.61          130.55            38           738.0
true MRU     no-overlap        130.60          130.41          4027         67144.5
static-min   no-overlap        734.08          734.08          9348        181548.0
```

Why is true MRU still fast here?  This is not a strict-budget-only trace.  The
window reaches high buckets:

```text
raw bucket 4608 -> usable 3776 MiB
managed Q4_0 weights = 3744 MiB
```

So in high-budget phases true MRU can reload all managed weights and then run
near full-resident speed.  The run confirms that:

```text
true-mru summary:
    accesses=305760 hits=301733 misses=4027 evictions=3879
    resident_bytes=3744.00 MiB
    budget_bytes=3776.00 MiB
```

This dynamic-trace result should therefore not be interpreted as "MRU is strong
under sustained 3840 MiB pressure."  It is fast because the trace gives it
large-budget recovery periods.

#### Constant 3840 MiB, 60 seconds

Artifacts:

```text
.wiki/elastic_memory/pipeline_runtime_review/artifacts/const3840_60s_true_mru_pipeline_online_offline
.wiki/elastic_memory/pipeline_runtime_review/artifacts/const3840_60s_true_mru_nooverlap_mru_min
```

Results:

```text
method       mode         raw ms/token   exec ms/token   direct reads   direct read MB
offline      pipeline          459.24          459.17            38           738.0
true MRU     no-overlap        461.04          460.97          6001         97614.0
online       pipeline          614.92          595.72            38           738.0
static-min   no-overlap        736.70          736.70          3154         61254.0
```

The strict constant-budget MRU run behaves as expected: it is no longer near the
dynamic-trace ~130 ms/token result and shows heavy foreground reload traffic:

```text
true-mru summary:
    accesses=29344 hits=23343 misses=6001 evictions=6027
    resident_bytes=3006.00 MiB
    budget_bytes=3008.00 MiB
```

Online is unexpectedly worse than offline in this fixed-budget 60 second run
because the remote solver path still produces many `xform` stage submissions
for the same 3840 plan (`stage xform calls=3762`) while offline submits only 38
stage xforms.  This is now a separate online-plan/runtime-application bug to
investigate; the true MRU correctness issue is fixed.

#### Strict pinned-output correction

The fixed-budget results above still used the old runtime convention where
`output.weight` was pinned but treated as outside the memory budget:

```text
pin unplanned output (410 MB) outside budget -> target=...
```

That was too generous.  It made true MRU look faster than a strict memory-budget
baseline should.  The runtime now supports:

```text
GGML_ELASTIC_PIN_UNPLANNED_OUTPUT_COUNTS_BUDGET=1
```

With this enabled, `output.weight` is still pinned, but it is counted inside the
budget.  The runner also adds `--pinned-extra-mib` to the planner/online/MRU
safety reserve.  For the Llama-3 8B Q4_0 experiment the default pinned extra is
410 MiB, matching `output.weight`.

Constant 3840 MiB, 30 second sanity:

```text
artifact:
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/const3840_30s_true_mru_pinned_output_inside_budget
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/const3840_30s_static_min_pinned_output_inside_budget

strict reserve:
    KV 512 + misc 256 + safety 64 + output.weight 410 = 1242 MiB
    managed-weight budget at raw 3840 = 2598 MiB
```

Result:

```text
method       raw ms/token   exec ms/token   direct reads   direct read MB
true MRU          671.38          671.46          3183         53649.0
static-min       1121.18         1121.18          1848         32130.0
```

True MRU summary:

```text
budget_bytes=2598.00 MiB
resident_bytes=2574.00 MiB
misses=3183
evictions=3187
```

This is the expected direction: after counting pinned output inside the budget,
MRU is no longer near the old ~460 ms/token fixed-budget result.

#### Strict disk-residency sanity: evict must force disk refill

After the pinned-output correction, the fixed-budget offline result was still
too good in one important way: direct disk reads were much lower than the
number of repeated disk-weight materializations implied by the plan.  That was
not a planner property.  It came from runtime cache paths that reused old data
after a logical evict.

Correct strict semantics:

```text
evict(weight):
    the weight is no longer resident
    the next miss must refill the contents from disk
    CPU/GPU allocation objects may be reused, but their old contents must not
    count as valid resident data
```

The problematic non-strict paths were:

```text
host_staging_by_idx        cached LOAD-stage bytes after PREPARE
GGML_ELASTIC_CACHE_FOREGROUND_LOAD=1
                            foreground direct reads were written back into host staging
SOA pool hit               parent/q/d triples could be handed back without disk refill
GGML_ELASTIC_CL_RETAIN=1   retained GPU buffers could hide stale-data reuse bugs
```

Constant 3840 MiB, 30 second offline sanity runs:

```text
artifact:
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/const3840_30s_offline_strict_release_stage
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/const3840_30s_offline_strict_no_stale_cache
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/const3840_30s_offline_strict_reuse_alloc_only

common strict budget reserve:
    KV 512 + misc 256 + output.weight 410 = 1178 MiB
```

Results:

```text
mode                                      raw ms/token   direct reads   direct read MB   note
old non-strict 60s                              657.88             66          1147.5   stale staging/SOA reuse
release host staging only                       640.50            132          2295.0   partial fix only
strict no stale cache, CL retain off           1669.32           1542         21528.0   correct disk-refill direction
strict alloc-reuse attempt, CL retain on      18563.66            388          5427.0   invalid: DOUBLE-EVICT parent bugs
```

The decisive check is `strict no stale cache, CL retain off`:

```text
stage load calls    = 1464
stage xform calls   = 1462
direct read calls   = 1542
direct read split   = async_load 1464 + foreground 78
SOA pool hits       = 0
```

This matches the intended semantics: scheduled LOAD does the disk refill and
PREPARE rematerializes the backend data.  The earlier `66` or `132` direct-read
counts were therefore not valid strict-budget pipeline measurements.

The `CL_RETAIN=1 + GGML_ELASTIC_NO_SOA_HIT=1` run is not a usable optimization
baseline yet.  It produced many:

```text
[soa-pool] DOUBLE-EVICT parent=...
```

So allocation reuse needs a separate correctness fix before it can be used in
the strict disk-residency experiment.  Until then, fair offline/online/MRU/static
comparisons should use the strict no-stale-cache configuration:

```text
GGML_ELASTIC_RELEASE_STAGE_AFTER_XFORM=1
GGML_ELASTIC_CACHE_FOREGROUND_LOAD=0
GGML_ELASTIC_CL_RETAIN=0
GGML_ELASTIC_NO_SOA_HIT=1
GGML_ELASTIC_PIN_UNPLANNED_OUTPUT_COUNTS_BUDGET=1
```

## 2026-06-22 MRU Strict Immediate And Bigger-Trace Sanity

The MRU baseline was changed from a plan-time static "tail on disk" shape into
a runtime cache baseline:

```text
LLAMA_ELASTIC_ONLINE_MODE=mru-cache
LLAMA_ELASTIC_RUNTIME_MRU_CACHE=1
LLAMA_ELASTIC_MRU_EVICT_IMMEDIATE=1
```

Important runtime semantics:

```text
miss:
    materialize the current weight with prepare_gpu or prepare_cpu, matching
    the planned xform target

over budget:
    choose the most-recently-used resident victim other than the current op
    synchronize the scheduler
    evict immediately, instead of waiting until graph end
```

The matrix runner also forces MRU to run without interval pipeline overlap:

```text
LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE=0
LLAMA_ELASTIC_DEFER_STAGE=0
GGML_ELASTIC_ASYNC_STAGE_LOAD=0
GGML_ELASTIC_ASYNC_STAGE_PREPARE=0
GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD=0
```

Static-min now receives the same no-overlap override, so the intended baseline
comparison is:

```text
online/offline:    planner + pipeline
MRU/static-min:    no interval pipeline overlap
```

Sanity run on a larger 9g trace window:

```text
source trace:
    trace/tmp_3min_bigger_mru/trace_08_user_270.csv

selected 60 second replay window:
    source_start = 60s
    raw min/mean/max = 4590.5 / 4700.3 / 4854.3 MiB
    reserve = KV 512 + misc 256 + output.weight 410 = 1178 MiB
    managed min/mean/max = 3412.5 / 3522.3 / 3676.3 MiB
    min bucket = 4352 MiB
```

Artifacts:

```text
.wiki/elastic_memory/pipeline_runtime_review/artifacts/trace08_9g_bigger_60s_strict_mru_current
.wiki/elastic_memory/pipeline_runtime_review/artifacts/trace08_9g_bigger_60s_static_min_nopipe
```

Results:

```text
method       raw ms/token   exec ms/token   direct read calls   direct read MB   notes
online            188.46          120.95                  33          573.8   provider_get excluded from exec
offline           188.43          188.37                  32          571.5   pipeline, table plans
MRU               212.46          212.31                 440         9427.5   strict immediate evict, no pipeline
static-min        222.09          222.09                  32          571.5   no pipeline override
```

This is the first run in the desired direction:

```text
online(exec) < offline < MRU < static-min
```

The earlier worry that "MRU is too fast" needs to be interpreted together with
the selected budget window.  On `trace_09_user_11` with a 4997 MiB minimum, the
5376 MiB bucket leaves about 4198 MiB for managed weights after reserves, while
all managed weights are only about 3744 MiB.  In that case MRU is fast because
almost everything legitimately fits.  The trace08 window above keeps the
managed budget around 3.4-3.7 GiB, which forces real disk reloads without
dropping back to the too-small 2.x GiB regime.

Follow-up: the trace08 window was still too large for a convincing pressure
test because it stays close to the 3744 MiB managed-weight total.  A tighter
middle-budget sanity was run with:

```text
source trace:
    trace/tmp_mid_budget_60s/trace_05_user_74.csv

selected 60 second replay window:
    raw min/mean/max = 4277.7 / 4573.8 / 5671.8 MiB
    reserve = 1178 MiB
    managed raw min/mean = 3099.7 / 3395.8 MiB
    min bucket = 4096 MiB
    managed min-bucket budget = 2918 MiB

artifact:
    .wiki/elastic_memory/pipeline_runtime_review/artifacts/trace05_midbudget_60s_strict_mru_current
```

Results:

```text
method       raw ms/token   exec ms/token   direct read calls   direct read MB   notes
online            701.80          164.31                  53          893.2   provider_get excluded from exec
offline           290.82          290.45                  52          891.0   pipeline, table plans
MRU               292.94          292.24                1583        28435.5   strict immediate evict, no pipeline
static-min       1619.17         1619.17                  50          828.0   no pipeline, fixed 4096 plan
```

This window is a better pressure point than trace08:

```text
online(exec) < offline < MRU << static-min
```

MRU is no longer artificially fast here.  It repeatedly reloads from disk
because the runtime budget dips to the 4096 MiB bucket, leaving only about
2918 MiB for managed elastic weights after reserves.
