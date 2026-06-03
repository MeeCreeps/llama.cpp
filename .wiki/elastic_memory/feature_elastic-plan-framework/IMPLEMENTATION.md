# Elastic Memory 执行框架 — 实现文档

> 目标分支:`feature/elastic-plan-framework`(从 `feature/elastic-op-sched-runtime` 切出)
> 状态:**设计 + 落地分步,待实现**。本文档先定义框架,代码按 §8 分步落。
> 关联:`docs/elastic/unified_scheduler_design.md`(统一 scheduler 4-hook 提案)、`runtime/plan/`(CP 求解器)、`.wiki/research/layout_transform_cost_2026-06-03.md`(变换代价实测)。

---

## 0. 一句话

把「内存预算 B(t) 不断变化」下的异构(CPU/GPU/disk,未来 NPU)权重调度,抽象成一条

```
   B(t)  ──►  PlanProvider.get_plan(B)  ──►  Plan(IR)  ──►  PlanExecutor.apply(Plan)
 (变化的预算)     (查表 / 启发式 / 未来 CP)    (正式定义)        (对齐 residency+routing+overlap)
```

的主线。**Plan 是核心可序列化中间表示**;求解器(现在概念、未来 CP)和执行器**解耦**——换求解器不碰执行器,换 backend 不碰 Plan 定义。

---

## 1. 设计目标与范围

### 1.1 要解决的问题(用户需求拆解)

研究背景:手机上 memory budget **动态变化**,而以往工作都假设固定 budget。本框架做 external-memory 执行 + 在线调度,scheduler 需联合决策三件事:

| # | 决策 | 在 Plan 里的体现 | 现状缺口 |
|---|---|---|---|
| **D1** | 预算变化时哪些 weight 移进/移出 | `Plan.residency` + `Plan.timeline`(prefetch/evict 事件) | 现在散在 `request_evict/prefetch` + WBM 自动 MRU,无统一时间线 |
| **D2a** | 每个 **weight** 当前放在哪(GPU/CPU/disk),且这个位置会随时间变 | `Plan.weights[i].location` | `placement` JSON 有,但与 compute 耦在一起 |
| **D2b** | 每个 **op** 哪个 processor 算 + **运行时**能否改 + 跨后端迁移(CPU→GPU layout 转换) | `Plan.ops[j].compute_backend` + `dispatch_mode` + `migrate` | graph-build 静态 routing 有(`routes`);**runtime per-op dispatch + 迁移机制已存在(ggml-backend.cpp)但没被 plan 驱动**,二者割裂 |
| **D3** | transfer / layout-transform / compute 的 overlap(per-engine busy,§5.5) | `Plan.timeline` 事件带 `engine` 归属 + `anchor_op_id` + `Plan.weights[i].xform` | 无显式 overlap 表示;转换的引擎归属(CPU repack ∥matmul vs GPU convert 入 GPU_busy)未进 plan |

> **D2 拆成两面**:D2a = weight **当前在哪**(存储位置,GPU/CPU/disk,**动态**——一开始有些 weight 在 CPU、有些在 GPU,运行中可迁移);D2b = op **谁算 + 怎么取**(计算面)。二者解耦——op 可在 GPU 算但 weight 这一刻在 CPU(→ 触发 CPU→GPU 迁移 + layout 转换,之后就在 GPU 上算)。D2b 里又分**静态 routing**(graph-build 时定,复用 graph,便宜)和**运行时 per-op dispatch**(compute 时按现场状态改 + 跨后端迁移,贵但灵活)——详见 §5.3。

### 1.2 范围

- **In**:CPU + GPU(OpenCL/Adreno UMA)两 backend;disk 作为第三层 location;Plan IR + Provider + Executor 三层;table provider(查现有 CP plan)+ heuristic provider(static/dynamic baseline);Executor 接现有 WBM / op-routing(静态)/ **per-op runtime dispatch + 跨后端迁移(layout 转换)** / overlap。
- **粒度**:放置/迁移的单元 = **整个 weight tensor**(如 `blk.5.ffn_down.weight`),不做 sub-tensor 切分。
- **Out(本期不做,留扩展点)**:NPU backend(DeviceProfile 已留位);CP 求解器的在线版(先用离线预生成表,`runtime/plan/` 已有)。
- **不破坏**:现有 `llama_set_op_schedule` 等 API 保留,变成 Plan Executor 的薄封装(见 §6.3)。

### 1.2.1 本期 Definition of Done(核心目标)

> **求解器(CP)不是本期重点**。本期就两件事,做到即算成:

1. **给定一个 plan,能严格按它执行**:喂一个 `exec_plan`(手写 / 预生成 JSON),Executor 把 residency(谁在 GPU)、op routing(谁算)、per-op backend + 跨后端迁移、搬运 overlap 全部对齐到 plan 描述的状态,decode 出正确结果。
2. **dynamic memory trace 下,每次内存变化喂一个新 plan,能切过去按新 plan 执行**:trace 给出 B(t) 序列,每当预算变化 → 外部(provider / 测试 harness)给一个对应的 plan → Executor `apply()` 切到新 plan(reconcile 差量 + 重 build routing),后续 decode 按新 plan 跑。

即:**Plan 当输入、Executor 当主角**。plan 哪来的(查表 / 手写 / 未来 CP)本期不关心——只要「给我一个 plan 我能执行」+「换一个 plan 我能切」。

### 1.3 设计原则

1. **Plan 是 backend-无关的纯数据**:用逻辑 id(weight_id、location、backend_id)描述,不含任何 cl_mem / 指针。可 JSON 序列化、可单测、可离线生成。
2. **求解器与执行器解耦**:Provider 产出 Plan,Executor 消费 Plan,中间只有 Plan IR 这一个契约。
3. **现有零件复用**:WBM(residency+victim)、BudgetWatcher(B(t) 信号)、calib.json(cost model)、`runtime/plan/`(CP 表)全部接进来,不重写。
4. **先框架后性能**:本期目标是「能按一个 plan 跑通 + 预算变化能换 plan」,不追 SOTA 速度。

---

## 2. 三层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  online loop  (llama_decode 每步 / BudgetWatcher 触发)                  │
│     B(t) ──► band 变了? ──► provider.get_plan(B) ──► executor.apply()    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │  Plan (IR, §3)
        ┌───────────────────┴───────────────────┐
        ▼                                        ▼
┌─────────────────────┐                ┌──────────────────────────────┐
│  PlanProvider (§4)   │                │   PlanExecutor (§5)          │
│  get_plan(B,ctx)→Plan│                │   apply(Plan) → 驱动:        │
│  - table   (查 CP表) │                │    • residency 对齐 (WBM)    │
│  - heuristic(static/ │                │    • op routing (set_op_sched)│
│      dynamic baseline│                │    • prefetch/evict timeline │
│  - cp (未来在线求解) │                │    • overlap 编排 (DMA/transform)│
└─────────┬───────────┘                └──────────┬───────────────────┘
          │ 输入                                    │ 落到
          ▼                                        ▼
┌─────────────────────┐                ┌──────────────────────────────┐
│ DeviceProfile (§7)   │                │ 现有 runtime 零件             │
│ = calib.json 形式化   │                │  weight_buffer_manager(_cl/_cpu)│
│ 异构 compute/mem/    │                │  budget_watcher              │
│ transform 速度        │                │  ggml-{cpu-elastic,opencl}   │
└─────────────────────┘                └──────────────────────────────┘
```

三层各自独立可测:Provider 给 (B, DeviceProfile) 出 Plan(纯函数,可单测);Executor 给 Plan 出一串 WBM/routing 动作(mock WBM 可单测);DeviceProfile 是只读数据。

---

## 3. Plan IR — 正式定义(框架核心)

**这是用户要的「plan 的具体定义」。** 放 `runtime/plan_ir.{h,cpp}`(纯逻辑,不依赖 llama.h/ggml,可单测 + JSON 互转)。

### 3.1 概念模型

- **weight**:放置/迁移的单元 = 一个完整 weight tensor(如 `blk.5.ffn_down.weight`),IR 用 `weight_id` 索引。
- **location**:weight 当前在哪。`GPU`(backend buffer)/ `CPU`(host RAM,mmap 常驻或 page cache)/ `DISK`(GGUF on flash,需 reload)。**动态**:一开始有些 weight 在 CPU、有些在 GPU,运行中随 plan/预算变化迁移(CPU→GPU 等)。
- **backend**:计算单元。`CPU` / `GPU`(/ 未来 `NPU`)。一个 op 的 compute backend 与其输入 weight 的 location **可不同**:GPU 算但 weight 在 CPU → 先 CPU→GPU 迁移(+ layout 转换)再算;CPU 算但 weight 在 disk → 先 reload。
- **transform(xform)**:layout 变换。target backend 的 kernel 要求特定 layout(GPU 强制 SOA+transpose;CPU 可选 repack,有 generic 退路)。代价见 layout_transform_cost 实测。

### 3.2 数据结构

```c
// runtime/plan_ir.h   namespace elastic

enum location_t { LOC_GPU = 0, LOC_CPU = 1, LOC_DISK = 2 };  // weight 当前在哪
enum backend_t { BE_CPU = 0, BE_GPU = 1, BE_NPU = 2 /*未来*/ };

// 物理引擎:overlap = 各引擎 busy 的 max。一段活儿落哪条引擎,决定它能跟谁并行。
enum engine_t { ENG_CPU = 0, ENG_GPU = 1, ENG_DISK = 2, ENG_DMA = 3 };

// layout 变换:把一个 weight 取到某 backend 算时需要的变换 + 它跑在哪条引擎
enum xform_kind_t {
    XF_NONE      = 0,  // 已是目标 layout,无变换
    XF_CPU_REPACK= 1,  // q4_0→4x8/8x8,引擎=CPU,~15GB/s;可 generic 退路免掉;∥GPU/disk,不∥CPU matmul
    XF_GPU_CONVERT=2,  // q4_0→SOA+transpose,引擎=GPU,强制;∥CPU matmul/disk,**不∥GPU matmul**
};

// ── 放置:每个 weight 当前在哪 / 取用时的变换 ──  (决策 D2a 的存储面)
struct weight_plan {
    int          weight_id;      // 稳定 id(= tensor 注册序)
    const char  *name;           // "blk.5.ffn_down.weight"(JSON 用;运行时按 weight_id 索引)
    int          layer;          // -1 = 非分层
    size_t       byte_size;
    location_t   location;       // 本 plan 下该 weight 放哪(GPU 常驻 / CPU 常驻 / DISK 流式);随 plan/预算变
    bool         pinned;         // location=GPU 且永不 evict(小权重)
    xform_kind_t xform;          // 从 location 取到其 op 的 compute backend 需要的变换
};

// ── 路由:每个 op 谁算 + 怎么取 ──  (决策 D2b 的计算面)
enum dispatch_mode_t {
    DISP_STATIC  = 0,  // graph-build 时定 backend,复用 graph(便宜,默认)
    DISP_RUNTIME = 1,  // compute 时 per-op 再决策(接 runtime_dispatch hook,可按现场状态改)
};
struct op_plan {
    int        op_id;            // graph 内稳定序(layer*N_OP + op_kind)
    const char *name;            // "blk.5.ffn_down"(对应输出 tensor)
    int        layer;
    backend_t  compute_backend;  // 这个 op 在哪算
    int        weight_id;        // 它消费的主 weight(连到 weight_plan)
    dispatch_mode_t dispatch;    // 静态 vs 运行时(D2b 两面)
    // 跨后端迁移:compute_backend 与 weight 当前 location 的 backend 不一致时,
    // 该 op 的输入要从 migrate_from 搬到 compute_backend 才能算。
    // 搬运 = transfer + layout 转换(GPU 侧 set_tensor 触发 convert+transpose)。
    bool       migrate;          // 是否需要跨后端迁移该 op 的输入
    backend_t  migrate_from;     // 输入当前所在 backend(weight 的 location 对应 backend)
    xform_kind_t migrate_xform;  // 迁移时的 layout 变换(= 目标 backend 的 layout 要求)
};

// ── 时间线:搬运事件 + overlap 分组 ──  (决策 D1 + D3)
enum ev_kind_t { EV_PREFETCH=0, EV_EVICT=1, EV_DMA=2, EV_XFORM=3 };
struct plan_event {
    ev_kind_t  kind;
    int        weight_id;
    location_t from_loc, to_loc;    // PREFETCH: DISK→CPU; DMA: CPU→GPU; EVICT: GPU→(drop)
    engine_t   engine;              // 这段活儿落哪条引擎 → 进哪个 *_busy,决定能跟谁并行
                                    //   EV_XFORM+ENG_GPU ⇒ 进 GPU_busy,只与非-GPU-matmul 并行
    int        anchor_op_id;        // 应在哪个 op 的 compute 窗口启动(prefetch 距离);受 work-buffer 容量界
    int        overlap_group;       // 同 group 可并发;-1 = 串行屏障(本期 greedy 执行可暂不依赖)
};

// ── 整个 plan ──
struct exec_plan {
    int    schema_version;
    int64_t budget_mib;             // 本 plan 适用的预算档
    size_t  kv_bytes, misc_bytes;   // 算 target 用
    int     n_weights, n_ops, n_events;
    weight_plan *weights;           // [n_weights]   按 weight_id 索引
    op_plan    *ops;                // [n_ops]
    plan_event *timeline;           // [n_events]  按执行顺序
    // 预测指标(求解器填,执行端只读,用于日志/对比)
    double pred_per_token_ms;
    const char *bottleneck;         // "disk"/"gpu"/"chain"
};
```

### 3.3 与现有 `plan_XXXXMiB.json` 的映射

现有 `runtime/plan/plans*/plan_*.json` 已经有 `placement`(per-op-kind pin/n_gpu)+ `routes`(per-tensor "cpu"/"gpu")+ 指标。table provider 直接把它**装载成 `exec_plan`**:

| JSON 字段 | → exec_plan |
|---|---|
| `routes["blk.0.attn_q.weight"]="cpu"` | `op_plan.compute_backend=BE_CPU`,对应 `weight_plan` |
| `placement.q.n_gpu` / `pin` | `weight_plan.location`(GPU vs DISK)+ `pinned` |
| `per_token_ms` / `bottleneck` | `pred_per_token_ms` / `bottleneck` |
| (新增,JSON 升级)`timeline` | 离散事件模拟已算出搬运顺序(make_plan.py simulate 段),序列化进来 |
| (新增)`weights[].xform` | 由 quant 类型 + compute_backend 推导(q4→GPU=XF_GPU_CONVERT, →CPU=XF_NONE/可选 repack) |

→ **make_plan.py 需小升级**:把 simulate 阶段已有的搬运时间线 + 变换标注一并 dump 进 JSON(§8 M4)。

---

## 4. PlanProvider — 拿 plan 的 API

放 `runtime/plan_provider.{h,cpp}`。Provider 是策略,Plan 是结果。

```c
// runtime/plan_provider.h

struct plan_provider;   // opaque

// 工厂:按名字建 provider
//  "table"     : 从 dir 加载预生成 CP plan,按 budget 选最近档(现有 plans*/index.json)
//  "static"    : 启发式,budget 恒定 min(B),全 GPU 装得下 + MRU(最保守 baseline)
//  "dynamic"   : 启发式,budget 跟 B(t),MRU
//  "cp"        : 未来,在线调 CP 求解(本期 stub → 退化到 table)
plan_provider *plan_provider_create(const char *kind,
                                    const char *plans_dir,   // table 用
                                    const device_profile *dp);// 启发式/cp 用

// 核心 API:给定当前预算,返回该用的 plan。
//  返回的 plan 由 provider 拥有(table: 缓存复用;heuristic: 内部 rebuild)。
//  budget 落在同一档 → 返回同一指针(executor 据此判断"要不要换 plan")。
const exec_plan *plan_provider_get(plan_provider *pp,
                                   int64_t budget_mib,
                                   size_t kv_bytes, size_t misc_bytes);

void plan_provider_destroy(plan_provider *pp);
```

**在线主循环里怎么用**(伪码,落在 llama-context 的 decode 路径):

```
on each decode step:
    B   = budget_watcher_get()              // 现有信号源
    plan= plan_provider_get(pp, B, kv, misc)
    if (plan != last_plan) {                // band 切换才动
        plan_executor_apply(ex, plan)       // §5
        llama_graph_invalidate(ctx)         // routing 变 → 重 build graph(现有 API)
        last_plan = plan
    }
```

→ band 不变时 0 开销(指针比较);band 变才 re-apply,符合现有 `LLAMA_KEEP_GRAPH_REUSE` 机制。

---

## 5. PlanExecutor — 按 plan 执行

放 `runtime/plan_executor.{h,cpp}`。Executor 把一个 `exec_plan` **翻译成对现有零件的动作**,是 §1.1 三决策的落地点。

```c
// runtime/plan_executor.h

struct plan_executor;

// 绑定到现有零件(WBM + op-routing sink + overlap hooks)。
// sinks 用函数指针,executor 不直接依赖 llama/ggml(跨 TU 解耦,同 victim_fn 风格)。
struct exec_sinks {
    // residency(D1):让 weight 进/出 GPU。底层 = wbmcl_prefetch/evict_batch。
    void (*set_resident)(int weight_id, bool want_resident, void *ud);
    bool (*is_resident )(int weight_id, void *ud);
    // 静态 routing(D2b-static):告诉 graph build 这个 op 走哪个 backend。底层 = op_schedule_fn 查表。
    void (*set_op_backend)(int op_id, backend_t be, void *ud);
    // 运行时 dispatch(D2b-runtime):compute 时 per-op 查 plan 决定 backend + 是否迁移。
    //   底层 = ggml_backend_sched_set_runtime_dispatch 的 callback。返回 -1=用默认。
    int  (*runtime_dispatch)(int op_id, int default_backend, void *ud);
    // 跨后端迁移(D2b 迁移):该 op 输入从 from→to backend 搬(含 layout 转换)。
    //   底层 = ggml-backend.cpp 的 per-op input migration(GGML_SCHED_RUNTIME_DISPATCH_MIGRATE)。
    void (*set_op_migrate)(int op_id, bool migrate, backend_t from, xform_kind_t xf, void *ud);
    // overlap(D3):把一个搬运事件挂到某 op 的 compute 上并行。底层 = prefetch 入队 + event。
    void (*enqueue_overlapped)(const plan_event *ev, void *ud);
    void *ud;
};

plan_executor *plan_executor_create(const exec_sinks *sinks);

// 应用一个 plan:
//  1. 算 target = budget - kv - misc(plan 自带或现场算)
//  2. reconcile residency:当前 resident 集合 → plan 期望 GPU 常驻集合
//       want 但不在 → set_resident(true);在但 plan 要它 DISK/CPU → set_resident(false)
//  3. 灌 op routing(D2b):static op → set_op_backend;runtime op 进 runtime_dispatch 查表
//  4. 灌迁移表(D2b 迁移):op_plan.migrate → set_op_migrate(from, xform)
//  5. 注册 timeline 给 overlap 编排器(D3):后续 ensure_resident 时按 anchor_op 并行搬
void plan_executor_apply(plan_executor *ex, const exec_plan *plan);

void plan_executor_destroy(plan_executor *ex);
```

### 5.1 三个决策怎么落

- **D1 residency reconcile**:plan 给出「GPU 常驻 weight 集合」(`location==LOC_GPU`)。Executor diff 当前 WBM resident 集合,只对差集发 prefetch/evict——**这就是「哪些 weight 移进移出」的统一出口**,替代手工枚举 `request_evict`。WBM 的 `victim_fn` 在自动兜底超预算时仍可由 plan 注入(plan 给淘汰序)。
- **D2a location**:`weight_plan.location` 决定该 weight 的取用源(GPU 常驻免搬 / CPU 走 DMA / DISK 走 reload)——**「weight 当前在哪」**。
- **D2b routing + 迁移**:见 §5.3。
- **D3 overlap**:见 §5.5(per-engine 模型 —— overlap 不是「把搬运挂到某 op」这么简单,而是各物理引擎 busy 的并行问题,layout 转换的引擎归属是关键)。

### 5.3 D2b — per-op backend 选择 + 跨后端迁移(CPU↔GPU layout 转换)

**这是「按 op 选 backend 执行,包括 CPU→GPU 转换」的落点。** 机制在当前 branch 已实现,本框架把它**接到 plan 驱动**:

| 已有机制(current branch) | 代码位置 | plan 框架里怎么用 |
|---|---|---|
| per-op runtime backend 选择 | `ggml_backend_sched_set_runtime_dispatch` + compute_splits hook (`ggml-backend.cpp:1589`) | `runtime_dispatch` sink 改成「查 plan 的 `op_plan.compute_backend`」,而非用户手写 callback |
| 跨后端 input/output 迁移 | per-op migration + v4/v5 buffer pool (`ggml-backend.cpp:1664/1708`),env `GGML_SCHED_RUNTIME_DISPATCH_MIGRATE` | `set_op_migrate` sink 按 `op_plan.migrate/from/xform` 开关;不再靠 env |
| layout 转换 | `ggml_backend_tensor_copy` → OpenCL set_tensor 触发 q4_0→SOA+transpose | `op_plan.migrate_xform` 标注代价(GPU 强制 convert),进 DeviceProfile / overlap 编排 |

**两种 dispatch 模式,plan 按 op 选**:
- `DISP_STATIC`(默认,便宜):backend 在 graph-build 时定,graph 可复用。适合稳态——大部分 op 走这。
- `DISP_RUNTIME`(灵活,贵):op compute 前查 plan 再定 backend,可跨后端迁移。适合「这个 op 的 weight 这一刻在 CPU,但想在 GPU 算」→ migrate=true,搬运时做 layout 转换。**注册 runtime dispatch 会强制 per-op 单 op 执行(吞吐降),所以 plan 应只把少量 op 标 `DISP_RUNTIME`**(实测 alternate 全切 882ms/tok,慢在 CPU GEMM 本身,不是 dispatch overhead——见 `project_v3_v4_cross_backend`)。

**迁移成本进 plan / 求解器**:`migrate_xform` 的代价取自 DeviceProfile(GPU convert 3B 层 15.4ms 强制抢算力 / CPU repack 6.6ms 可选)。CP 求解时,「op 在 GPU 算 + weight 在 CPU」这条边的代价 = DMA + GPU convert;solve() 据此权衡是迁移去 GPU 算、还是就地 CPU 算。**硬约束**:per-token 高频迁移走 GPU 病态 → 迁移优先 CPU 侧(免变换),GPU 迁移只给常驻/低频。

### 5.4 与现有散 API 的关系(收编,不删)

| 旧散 API | Plan 框架里的位置 |
|---|---|
| `llama_set_op_schedule(fn)` | Executor 内部建一张 op→backend 表,旧 fn 变成「读这张表」的默认实现(D2b-static) |
| `llama_set_op_runtime_dispatch(fn)` | Executor 的 `runtime_dispatch` sink;旧 callback → 查 plan 的 `op_plan`(D2b-runtime) |
| `GGML_SCHED_RUNTIME_DISPATCH_MIGRATE` env | 由 `op_plan.migrate` 逐 op 控制,取代全局 env 开关 |
| `llama_weight_request_prefetch/evict` | residency reconcile 的底层动作;用户仍可手动调(plan 外的临时覆盖) |
| `llama_set_weight_pin` | `weight_plan.pinned` 的等价;pinned weight 不进 victim 候选 |
| `llama_set_scheduler`(通知式) | 升级:不再是「通知」,而是 online loop 里 provider+executor 的固定流程 |
| `victim_fn`(WBM 已有) | plan 可注入「按 plan 淘汰序」的 victim_fn |

### 5.5 D3 — overlap 是「per-engine busy」问题(含 layout 转换)

overlap **不是**「把一次搬运挂到某个 op 的 compute 上」这么单线。它是**多条物理引擎同时忙**的并行问题:

```
CPU 引擎 :  Σ CPU matmul   + Σ CPU repack
GPU 引擎 :  Σ GPU matmul   + Σ GPU convert+transpose
disk 引擎:  Σ disk read
dma 引擎 :  Σ cpu↔gpu 上传   (UMA 上廉价;但见下方 upload caveat)
一个 token 耗时 ≈ max(CPU_busy, GPU_busy, disk_busy, dma_busy)   —— 不是相加
```

compute 链本身是**串行数据依赖**(op_{k+1} 等 op_k 的输出),但**未来 weight 的 transfer/transform 可以挂在当前 op 的 compute 窗口里跑** —— 这才是 overlap 的来源。关键是每段活儿落在哪条引擎:

**layout 转换的引擎归属(核心,别搞错)**:
| 转换 | 引擎 | 能和谁并行 | **不能和谁并行** | 计入 |
|---|---|---|---|---|
| **CPU repack** | CPU | GPU matmul / disk / dma | CPU matmul | `CPU_busy`(或免掉) |
| **GPU convert+transpose** | GPU | **CPU matmul** / disk / dma | **GPU matmul**(同引擎) | `GPU_busy` |

→ **GPU convert 能藏进 CPU-op 的窗口**(和 CPU compute 并行),只是藏不进 GPU matmul。所以它不是"白亏",而是"看 GPU 引擎那一刻空不空"。

**调度杠杆(D2b × D3 耦合)**:既然 GPU 引擎在「链里那段 op 跑 CPU」时是空的,plan 可以**故意交错 CPU/GPU op,让 GPU 引擎在 CPU-op 窗口里 pre-convert 后面要在 GPU 算的 weight**:

```
chain :  [op_k CPU][op_{k+1} CPU][op_{k+2} GPU matmul]...
GPU   :   └ convert W_{k+2} ┘└ convert W_{k+5} ┘[op_{k+2}]
          ↑ GPU 空(因 op_k/op_{k+1} 在 CPU)→ 拿来转换,白嫖
```

即 backend 分配不止决定"谁算快",还决定"哪条引擎此刻空、能拿去搬/转"。solver 里体现为:`GPU_busy` 只累加 GPU matmul + 那些**无法被 CPU-op 窗口掩盖**的 convert。

**实测 caveat(必须区分 convert vs upload)**:理想模型下 `GPU convert ∥ CPU compute` 成立;但 layout_transform 实测发现 **host→GPU 上传会串行掐住 GPU kernel**(同 command queue 排队,`K+W ×0.00`)。所以:
- `GPU convert kernel ∥ CPU compute` → 成立,可白嫖;
- `host→GPU upload ∥ GPU matmul` → 实测互掐,需第二 queue 才可能并行。

→ DeviceProfile 把 **convert(GPU compute 子项)和 upload(可能掐 matmul 的子项)拆开**(见 §7);upload 能否和 GPU matmul 并行**得目标设备实测拍板**,本期保守当"掐"(算进 `GPU_busy`)。

**plan 怎么表示 + executor 怎么落**:
- `plan_event` 带 `engine`(CPU/GPU/disk/dma):决定这段进哪条引擎的 busy,从而能跟谁并行。`EV_XFORM` + `engine==GPU` ⇒ 进 `GPU_busy`,只与非-GPU-matmul 并行。
- `anchor_op_id` = 该 transfer/convert 应在哪个 op 的 compute 窗口启动(prefetch 距离),受 work-buffer 容量上界。
- **正确性靠 event**(WBM `prefetch_event`/`last_use_event`):op 消费 weight 前 `clWaitForEvents`,没到就 stall —— 即使 plan 的时间预测偏了结果仍对,只是没完全 overlap。
- 本期 executor 先做 **greedy prefetch**(按 `location==LOC_DISK` 集合 + op use-order 尽量超前,容量卡 work-buffer),`anchor_op_id` 作为未来精确调度(CP)的落点保留。

---

## 6. 在线主循环接线

落在 `src/llama-context.cpp` 的 decode 入口(现有 `maybe_run_scheduler` 位置)。

```c
// llama-context 持有(新增字段,llama-context.h)
plan_provider  *elastic_pp = nullptr;
plan_executor  *elastic_ex = nullptr;
const exec_plan*elastic_last_plan = nullptr;

// 每 decode 前(现有 maybe_run_scheduler 改写):
void llama_context::maybe_apply_plan() {
    if (!elastic_pp) return;
    int64_t B = elastic_budget_mib();        // BudgetWatcher,失败回退 /proc/meminfo
    const exec_plan *p = plan_provider_get(elastic_pp, B, kv_bytes(), misc_bytes());
    if (p != elastic_last_plan) {
        plan_executor_apply(elastic_ex, p);
        graph_reuse_disable = true;          // 现有机制,触发重 build
        elastic_last_plan = p;
    }
}
```

### 6.1 新的对外 API(llama.h,薄封装)

```c
// ── DoD#1:直接喂一个 plan 让 ctx 按它执行(最核心入口)──
//  plan 来源不限:从 JSON 加载,或调用方手构。立即 apply(reconcile + 重 build routing)。
LLAMA_API int llama_elastic_apply_plan(
        struct llama_context     *ctx,
        const struct llama_plan  *plan);       // = exec_plan 的对外句柄
LLAMA_API struct llama_plan * llama_plan_load_json(const char *path);  // JSON → plan
LLAMA_API void                llama_plan_free(struct llama_plan *plan);

// ── DoD#2:开启 dynamic 模式,内存变化时自动换 plan ──
//  provider 负责"每次 B(t) 变化给一个 plan":
//   "table"   = 从 plans_dir 按 budget 查预生成 plan(对应"每次给你一个 plan")
//   "callback"= 调用方注册 fn(budget)→plan,完全自定义"内存变化喂哪个 plan"
LLAMA_API int llama_elastic_enable(
        struct llama_context *ctx,
        const char           *provider_kind,   // "table"/"callback"/"static"/"dynamic"
        const char           *plans_dir);      // table 用,可 NULL
typedef const struct llama_plan * (*llama_plan_provider_fn)(int64_t budget_mib, void *ud);
LLAMA_API void llama_elastic_set_plan_provider(
        struct llama_context *ctx, llama_plan_provider_fn fn, void *ud);

// 调试/外部执行:拿当前预算下该用的 plan(只读视图)
LLAMA_API const struct llama_plan * llama_elastic_get_plan(
        struct llama_context *ctx, int64_t budget_mib);
```

→ 对应用户的两个 DoD:`llama_elastic_apply_plan` = **「给一个 plan 就能执行」**;`llama_elastic_enable("table"/"callback")` + provider = **「内存变化每次给一个 plan 就能切」**。`get_plan` 还能导给外部 runner(`tools/main`)对照。

---

## 7. DeviceProfile — 异构 cost model(CP 的输入)

放 `runtime/device_profile.{h,cpp}`,**就是 `calib.json` 的形式化**。这是「每个 CPU/GPU 有各自 compute 和 memory 速度」的载体,也是 CP 求解器的唯一输入。

```c
struct backend_cost {            // 每个 compute backend 一份
    backend_t backend;
    double    matmul_gbps;       // 该 backend 算 GEMM 的有效吞吐(per op-kind 可细分)
    double    xform_gbps;        // 变换吞吐(CPU repack ~15 / GPU convert ~3.4–7.5)
    bool      xform_optional;    // CPU=true(generic 退路)/ GPU=false(强制)
    engine_t  xform_engine;      // 变换跑哪条引擎(CPU repack→ENG_CPU / GPU convert→ENG_GPU)
                                 //  → 决定它进哪条引擎 busy、能跟谁并行(§5.5)
    // upload 与 convert 拆开:convert 是 GPU compute 子项(∥CPU matmul);
    // upload(host→GPU DMA)实测会掐 GPU matmul(同 queue),单列、可单独标是否∥matmul。
    double    upload_gbps;       // host→GPU 上传吞吐(pooled ~25)
    bool      upload_blocks_gpu_matmul; // 实测 true(本期保守);有第二 queue 分离可设 false
};
struct link_cost {               // location 间传输
    location_t from, to;
    double bw_gbps;              // DISK→CPU(odirect ~3.67)/ CPU→GPU DMA(pooled ~25)
};
struct device_profile {
    backend_cost backends[3];    // CPU/GPU/(NPU 留位)
    link_cost    links[…];
    size_t       reserved_bytes_per_ctx;  // kv + misc
};
device_profile *device_profile_load(const char *calib_json);
```

实测锚点(已在 calib.json / 研究笔记,直接引):CPU GEMM vs GPU ~1.8×(非 5.9×);DMA pooled 25GB/s vs realloc 3.7;disk odirect ~3.67GB/s;CPU repack 15GB/s(可选)vs GPU convert 3.4–7.5GB/s(强制、抢算力)。见 `.wiki/research/layout_transform_cost_2026-06-03.md`、`project_offline_plan_cp` 记忆。

---

## 8. CP 求解器接口(概念,本期不做在线版)

求解器 = 「给定 `device_profile` + `budget` → 产出 `exec_plan`」的纯函数。本期**复用 `runtime/plan/make_plan.py` 离线生成表**,框架只定义契约:

```
solve(device_profile, budget, ctx) → exec_plan
    目标: min makespan = max(各引擎 busy)        ── per-engine 模型,见 §5.5
          CPU_busy = Σ CPU matmul + Σ CPU repack
          GPU_busy = Σ GPU matmul + Σ(无法被 CPU-op 窗口掩盖的)GPU convert (+ upload,若掐 matmul)
          disk_busy= Σ disk read
    变量: per-weight {location, pin/stream}, per-op {compute_backend, migrate}, prefetch order
    约束: Σ resident_bytes ≤ budget - reserved; location/backend 兼容性; work-buffer 容量
    机会: 交错 CPU/GPU op,让 GPU convert 藏进 CPU-op 窗口(§5.5 调度杠杆)→ 压低 GPU_busy
    lexicographic: 二级目标让常驻权重跨 budget 档稳定(减少 re-plan 抖动)
```

现状 `make_plan.py` 已实现 CP-SAT(solve_decision)+ 离散事件模拟(simulate)。**本期只需让它多 dump `timeline` + `xform`**(§3.3),输出就是合法 `exec_plan` JSON。在线 CP(`"cp"` provider)留作后续——接口已固定,届时实现 `solve()` 即可,不动 Plan IR / Executor。

---

## 9. 落地分步(milestones)

每步独立可编可测;先桌面(无 OpenCL,纯逻辑)再上设备。**主线 = DoD#1(给 plan 能执行)→ DoD#2(内存变化能换 plan)**,其余是增强。

| M | 内容 | 产出/验证 | 依赖 |
|---|---|---|---|
| **M0** | 切分支 `feature/elastic-plan-framework`;建空 `runtime/plan_ir.{h,cpp}` + CMake | 编过 | — |
| **M1** | **Plan IR + 加载一个 plan**:struct + JSON→exec_plan 加载(+ 手写一个 plan)+ 单测 | `tests/elastic/test_plan_ir.cpp` 装载一个 plan_*.json 通过 | M0 |
| **M2** | **PlanExecutor**:`apply(plan)` = residency reconcile + static routing 灌表(D1+D2a+D2b-static);mock sinks 单测验证动作序列 | 给 plan → 期望 prefetch/evict/routing 动作正确 | M1 |
| **M3 ★DoD#1** | **接线 llama-context:喂一个 plan 直接执行**。`llama_elastic_apply_plan(ctx, plan)`(从 JSON 加载或 API 传入)+ executor sinks 桥接 WBM/op-routing | 桌面 cpu-elastic 按一个**固定 plan** 跑通 decode,结果正确,residency/routing 符合 plan | M2 |
| **M4 ★DoD#2** | **dynamic trace 换 plan**。online loop:BudgetWatcher trace 驱动 B(t),每次变化由 provider(table:budget→plan 文件,或测试 harness 直接喂)给新 plan → `apply()` 切换(reconcile 差量 + graph 重建) | trace 跑完,每次 B(t) 变化都切到对应 plan 并继续 decode;日志显示 plan 切换 + residency 跟随 | M3 |
| **M5** | **D2b runtime dispatch + 跨后端迁移**(设备):`runtime_dispatch`/`set_op_migrate` sink 接 `ggml_backend_sched_set_runtime_dispatch` + 已有 migration pool,plan 驱动 per-op backend + CPU↔GPU layout 转换 | 设备上某 op 按 plan 切 backend + 迁移生效 | M4 |
| **M6** | **overlap 编排(D3)**:timeline anchor → prefetch/DMA/迁移挂到 op compute;接 opencl WBM | 设备上 overlap 生效,对比无 overlap | M5 |
| **M7** | **端到端实测**:真 trace,plan-driven vs static baseline,出指标 | `scripts/elastic/` 跑 + metrics | M6 |

**支撑件(不在 DoD 主线,按需并行)**:
- **DeviceProfile**(`runtime/device_profile`,calib.json 形式化)+ **make_plan.py 升级**(dump timeline/xform)——这俩是**生成 plan** 用的;本期 plan 当输入,可先手写/用现有 `plans*/` JSON,等需要新 plan 再补。

**★ = 用户定义的两个核心里程碑**。最小达成 = **M1→M2→M3→M4**:能加载 plan、按 plan 对齐状态、桌面跑通固定 plan、内存 trace 下换 plan。M5/M6(runtime 迁移、overlap)是设备上的增强。

---

## 10. 文件清单

新增(`runtime/`,纯逻辑优先,跨 TU 用函数指针解耦):
```
runtime/plan_ir.{h,cpp}            # M1  exec_plan 定义 + JSON 加载
runtime/plan_executor.{h,cpp}      # M2  apply(plan) → sinks(DoD 核心)
runtime/plan_provider.{h,cpp}      # M4  budget→plan(table/callback)
tests/elastic/test_plan_ir.cpp     # M1
tests/elastic/test_plan_executor.cpp # M2
runtime/device_profile.{h,cpp}     # 支撑(生成 plan 用,按需)
```
改动:
```
include/llama.h                    # M3  llama_elastic_apply_plan/enable/get_plan
src/llama-context.{h,cpp}          # M3/M4 apply_plan + online loop + provider/executor;M5 runtime_dispatch/migrate sink
runtime/weight_buffer_manager*.cpp # M3/M6 executor sinks 桥接(set_resident 等)
ggml/src/ggml-backend.cpp          # M5  per-op migration 由 plan 逐 op 控制(取代 env 全局开关)
runtime/plan/make_plan.py          # 支撑 dump timeline+xform(生成 plan 用,按需)
runtime/CMakeLists.txt             # 加新 TU
```

## 11. 风险与开放问题

- **粒度**:放置/迁移单元 = 整个 weight tensor,不做 sub-tensor 切分。
- **re-plan 抖动**:budget 在档边界来回 → 频繁重 build graph。缓解:provider 加滞回(hysteresis)+ lexicographic 让相邻档 plan 尽量同构。
- **GPU 流式病态(实测)**:per-token GPU reload ~17s/tok 且崩设备。**plan 应让流式那部分走 CPU(可免变换),GPU 只放常驻**——这是硬约束,要写进 solve() 约束 + provider 校验(见 `project_q4_elastic_findings`)。
- **D3 overlap 正确性**:event 的 anchor_op 错位会算错结果(weight 没到就算)。Executor 必须在 op compute 前 `ensure_resident` 兜底(现有机制),overlap 只是把它提前,不替代正确性检查。
- **Plan schema 版本**:`schema_version` 字段;旧 JSON 缺 timeline 时 executor 退化到「无 overlap、串行 reload」。

---

## 12. 与现有设计文档的关系

本文档 = `docs/elastic/unified_scheduler_design.md` 的**落地具体化 + 用户三决策(weight 位置 / per-op backend + 迁移 / overlap / plan API)的补全**:
- 那份的 4-hook(budget_target/pick_victim/route_op/on_step)→ 本框架里:budget_target/route_op 合进 **Plan**(provider 算好),pick_victim 留 WBM `victim_fn`(plan 可注入),on_step → **online loop**(固定流程)。
- 那份缺的「plan 正式定义 + 拿 plan 的 API + weight 位置 + 跨后端迁移/transform overlap」由本文 §3/§4/§5 补上。
