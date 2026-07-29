// runtime/plan_ir.h
//
// Plan IR —— elastic memory 执行框架的核心中间表示(见
// .wiki/elastic_memory/feature_elastic-plan-framework/IMPLEMENTATION.md §3)。
//
// 一个 ExecPlan 完整描述「在某个内存预算 B 下,一个 decode token 怎么执行」:
//   - WeightPlan : 每个 weight 当前在哪(GPU/CPU/DISK)+ 取用时的 layout 变换
//   - OpPlan     : 每个 op 哪个 backend 算 + 静态/运行时 dispatch + 是否跨后端迁移
//   - PlanEvent  : 搬运/变换事件的时间线 + 引擎归属(overlap = per-engine busy,§5.5)
//
// 设计约束:
//   * 纯逻辑,不依赖 llama.h / ggml —— 可在桌面无 OpenCL 环境编译 + 单测。
//   * 用逻辑 id(weight_id / op_id / engine)描述,不含任何 cl_mem / 指针。
//   * 可 JSON 序列化(native schema)+ 可加载现有 make_plan.py 产出的 plan_*.json。
//
// WorkingUnitPlan additionally selects how physical weight tiles are grouped
// into schedulable units (sub-tensor / tensor / multi-tensor).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace elastic {

// weight 当前在哪一层 location(动态:随 plan/预算变化迁移)
enum class Location { GPU = 0, CPU = 1, DISK = 2 };

// 计算单元
enum class Backend { CPU = 0, GPU = 1, NPU = 2 /*未来*/ };

// 物理引擎:overlap = 各引擎 busy 的 max。一段活儿落哪条引擎,决定它能跟谁并行。
enum class Engine { CPU = 0, GPU = 1, DISK = 2, TRANSFER = 3 };

// layout 变换 + 它跑在哪条引擎
//   CPU_REPACK  : 引擎=CPU,可被 generic 退路免掉;∥ GPU matmul/disk,不 ∥ CPU matmul
//   GPU_CONVERT : 引擎=GPU,强制(SOA+transpose);∥ CPU matmul/disk,**不 ∥ GPU matmul**
enum class Xform { NONE = 0, CPU_REPACK = 1, GPU_CONVERT = 2 };

// op backend 决策的时机
//   STATIC  : graph-build 时定,复用 graph(便宜,默认)
//   RUNTIME : compute 时 per-op 再决策(接 runtime_dispatch hook,可跨后端迁移)
enum class Dispatch { STATIC = 0, RUNTIME = 1 };

// 时间线事件类型
//   LOAD     : disk / mmap -> host staging, 不创建 backend layout
//   TRANSFER : host staging -> backend-visible raw/staging buffer
//   XFORM    : backend/raw buffer -> compute layout (OpenCL SOA+transpose, CPU repack)
//   PREFETCH : legacy combined event; executor/backend may implement as LOAD+TRANSFER+XFORM
enum class EvKind { LOAD = 0, PREFETCH = 1, EVICT = 2, TRANSFER = 3, XFORM = 4 };

// ── 放置:每个 weight 当前在哪 / 取用时的变换 ── (决策 D2a)
struct WeightPlan {
    int         weight_id = -1;      // 稳定 id(= weights[] 下标)
    std::string name;                // "blk.5.ffn_down.weight"
    int         layer     = -1;      // -1 = 非分层
    size_t      byte_size = 0;       // 0 = 未知(loader 没字节信息时,执行端用 WBM 实际大小)
    Location    location  = Location::DISK;  // 本 plan 下放哪
    bool        pinned    = false;   // location=GPU 且永不 evict
    Xform       xform     = Xform::NONE;  // 取到其 op 的 compute backend 需要的变换
};

// ── 路由:每个 op 谁算 + 怎么取 ── (决策 D2b)
struct OpPlan {
    int         op_id           = -1;  // 稳定序(= ops[] 下标)
    std::string name;                  // 对应消费的 weight 名
    int         layer           = -1;
    Backend     compute_backend = Backend::GPU;  // 这个 op 在哪算
    int         weight_id       = -1;  // 消费的主 weight(连到 WeightPlan)
    Dispatch    dispatch        = Dispatch::STATIC;
    // 跨后端迁移:compute_backend 与 weight 当前 location 的 backend 不一致时,
    // 输入要从 migrate_from 搬到 compute_backend(transfer + layout 转换)才能算。
    bool        migrate         = false;
    Backend     migrate_from    = Backend::CPU;
    Xform       migrate_xform   = Xform::NONE;
};

// ── 时间线:搬运/变换事件 + 引擎归属 ── (决策 D1 + D3)
struct PlanEvent {
    EvKind   kind          = EvKind::PREFETCH;
    int      weight_id     = -1;
    Location from_loc      = Location::DISK;
    Location to_loc        = Location::CPU;
    Engine   engine        = Engine::DISK;  // 进哪条引擎 busy → 决定能跟谁并行
    int      anchor_op_id  = -1;  // 应在哪个 op 的 compute 窗口启动(prefetch 距离)
    int      overlap_group = -1;  // 同 group 可并发;-1 = 串行屏障
};

// Optional interval schedule emitted by the CP-SAT planner.  The solver may
// use interval variables internally, but runtime execution is driven by
// anchor_op_id: the graph op index where this stage should be triggered.
// start_ms/end_ms are diagnostic cost-model coordinates only.
struct ScheduleEvent {
    int         weight_id   = -1;
    int         anchor_op_id = -1;
    std::string weight_name;
    std::string choice;
    std::string kind;       // load / transfer / xform / sync / compute
    std::string engine;     // disk / transfer / xform_cpu / xform_gpu / compute_cpu / compute_gpu / sync
    double      start_ms    = 0.0;
    double      end_ms      = 0.0;
    double      duration_ms = 0.0;
};

// Runtime-managed working set. Unlike WeightPlan, an entry represents a
// bounded set of interchangeable slices/objects whose demand can change per
// token (for example expert slices or multimodal encoder tiles).
struct WorkingSetPlan {
    std::string name;                 // stable logical resource name
    std::string kind;                 // executor/backend capability key
    int         target_capacity = -1; // -1 lets the backend choose
    int         budget_capacity = -1; // hard ceiling under this plan's budget
    int         min_capacity    = 0;  // correctness floor
    int         max_capacity    = -1; // -1 means no plan-side ceiling
    std::string policy;               // replacement/retention policy
    bool        state_aware = false;  // target used observed runtime state
    bool        coupled_to_core = false; // capacity change requires weight-plan reconciliation
};

// Smallest pre-provisioned row tile referenced by a schedulable Super-Tensor.
// A whole-tensor tile uses row_start=0,row_count=-1. byte_size=0 lets the
// backend derive the physical size.
struct WorkingUnitTilePlan {
    int         weight_id  = -1;
    std::string weight_name;
    int64_t     row_start  = 0;
    int64_t     row_count  = -1;
    size_t      byte_offset = 0;
    size_t      byte_size   = 0;
};

// One schedulable unit. A unit containing part of one weight is CUT; all tiles
// of one weight form Tensor; tiles from multiple weights form Multi. Thus one
// plan can mix all three granularities.
struct SuperTensorUnitPlan {
    int unit_id = -1;
    std::vector<WorkingUnitTilePlan> tiles;
    bool fuse_layout  = false;
    bool fuse_compute = false;
};

// Reconfigurable mixed partition. `mode` remains a backward-compatible
// fallback for plans without explicit units.
struct WorkingUnitPlan {
    bool        enabled       = false;
    std::string mode          = "tensor";
    int         cut_parts     = 2;
    int         multi_tensors = 2;
    std::string policy;
    bool        state_aware   = false;
    double      predicted_ms  = 0.0;
    double      switch_cost_ms = 0.0;
    std::vector<SuperTensorUnitPlan> units;
};

// ── 整个 plan ──
struct ExecPlan {
    int         schema_version = 1;
    int64_t     budget_mib     = 0;
    size_t      kv_bytes       = 0;
    size_t      misc_bytes     = 0;

    std::vector<WeightPlan> weights;   // 按 weight_id 索引
    std::vector<OpPlan>     ops;       // 按 op_id 索引
    std::vector<PlanEvent>  timeline;  // 按执行顺序
    std::vector<ScheduleEvent> schedule_events; // optional interval CP-SAT schedule
    std::vector<WorkingSetPlan> working_sets;   // runtime-varying resources
    WorkingUnitPlan working_unit;                // split/merge configuration

    std::string schedule_kind;
    std::string schedule_status;
    double      schedule_objective_ms = 0.0;

    // 预测指标(求解器填,执行端只读;用于日志/对比)
    double      pred_per_token_ms = 0.0;
    std::string bottleneck;            // "disk"/"gpu"/"chain"

    // ── 查询辅助 ──
    const WeightPlan * weight_by_id(int id) const;
    const WeightPlan * weight_by_name(const std::string & name) const;
    int                weight_id_of(const std::string & name) const;  // -1 = 没有
    const OpPlan *     op_by_id(int id) const;

    // 本 plan 期望「常驻在 GPU」的 weight 集合(executor residency reconcile 用)
    std::vector<int>   resident_gpu_ids() const;
};

// ── native schema 序列化(round-trips ExecPlan)──
bool plan_to_json_string(const ExecPlan & plan, std::string & out);
bool plan_to_json_file  (const ExecPlan & plan, const std::string & path);
bool plan_from_json_string(const std::string & s, ExecPlan & out, std::string * err = nullptr);
bool plan_from_json_file  (const std::string & path, ExecPlan & out, std::string * err = nullptr);

// ── 加载现有 make_plan.py 产出格式(routes/placement/schedule/resident_in_memory)──
// 把 plan_XXXXMiB.json 映射成 ExecPlan(见 IMPLEMENTATION.md §3.3)。
bool plan_from_make_plan_file(const std::string & path, ExecPlan & out, std::string * err = nullptr);

// ── 枚举 ⇄ 字符串 ──
const char * to_string(Location);
const char * to_string(Backend);
const char * to_string(Engine);
const char * to_string(Xform);
const char * to_string(Dispatch);
const char * to_string(EvKind);

// weight location 对应的 backend(DISK reload 后落 CPU)
Backend backend_of_location(Location);

}  // namespace elastic
