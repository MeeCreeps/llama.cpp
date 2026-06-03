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
// 实现单位 = 整个 weight tensor(不做 sub-tensor 切分)。

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
enum class Engine { CPU = 0, GPU = 1, DISK = 2, DMA = 3 };

// layout 变换 + 它跑在哪条引擎
//   CPU_REPACK  : 引擎=CPU,可被 generic 退路免掉;∥ GPU matmul/disk,不 ∥ CPU matmul
//   GPU_CONVERT : 引擎=GPU,强制(SOA+transpose);∥ CPU matmul/disk,**不 ∥ GPU matmul**
enum class Xform { NONE = 0, CPU_REPACK = 1, GPU_CONVERT = 2 };

// op backend 决策的时机
//   STATIC  : graph-build 时定,复用 graph(便宜,默认)
//   RUNTIME : compute 时 per-op 再决策(接 runtime_dispatch hook,可跨后端迁移)
enum class Dispatch { STATIC = 0, RUNTIME = 1 };

// 时间线事件类型
enum class EvKind { PREFETCH = 0, EVICT = 1, DMA = 2, XFORM = 3 };

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

// ── 整个 plan ──
struct ExecPlan {
    int         schema_version = 1;
    int64_t     budget_mib     = 0;
    size_t      kv_bytes       = 0;
    size_t      misc_bytes     = 0;

    std::vector<WeightPlan> weights;   // 按 weight_id 索引
    std::vector<OpPlan>     ops;       // 按 op_id 索引
    std::vector<PlanEvent>  timeline;  // 按执行顺序

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
