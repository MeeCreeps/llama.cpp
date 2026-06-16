// runtime/plan_executor.h
//
// PlanExecutor —— 把一个 ExecPlan 翻译成对现有零件(WBM residency / op routing /
// 跨后端迁移 / overlap 编排)的动作(见 IMPLEMENTATION.md §5)。
//
// 设计:executor 纯逻辑,不直接依赖 llama/ggml。通过 ExecSinks(一组回调)落到
// 真实后端 —— 测试时喂 mock sink 验证动作序列;llama-context 喂桥接到 WBM/
// op_schedule 的 lambda。
//
//   apply(plan) 做四件事:
//     1. residency reconcile (D1)  : 当前 GPU 驻留集合 → plan 期望集合,只搬差量
//     2. static routing      (D2b) : 每个 STATIC op → set_op_backend
//     3. migration           (D2b) : 每个 migrate op → set_op_migrate(from, xform)
//     4. 记录 timeline       (D3)  : 建 anchor_op → events 索引,供 decode 时 overlap

#pragma once

#include <functional>
#include <unordered_map>
#include <vector>

#include "plan_ir.h"

namespace elastic {

// executor 落地用的回调集合。任意成员可留空(nullptr-like:未设的 std::function);
// 留空 = 该维度不驱动(executor 跳过)。
struct ExecSinks {
    // residency(D1):让 weight 进/出 GPU。底层 = wbmcl_prefetch / evict。
    std::function<void(int weight_id, bool want_resident)> set_resident;
    std::function<bool(int weight_id)>                     is_resident;

    // 静态 routing(D2b-static):graph-build 这个 op 走哪个 backend。底层 = op_schedule 查表。
    std::function<void(int op_id, Backend be)>             set_op_backend;

    // 跨后端迁移(D2b):该 op 输入是否需 from→compute_backend 迁移(含 layout 转换)。
    std::function<void(int op_id, bool migrate, Backend from, Xform xf)> set_op_migrate;

    // overlap(D3):把一个搬运/变换事件挂到某 op 的 compute 上并行。底层 = 异步 prefetch 入队。
    std::function<void(const PlanEvent & ev)>             enqueue_overlapped;

    // 分阶段执行(D3):LOAD / TRANSFER / TRANSFORM 可分别落到 disk / transfer / transform engine。
    // 未设置时 executor 只记录 timeline；设置后 apply() 默认会按 plan.timeline 顺序下发。
    // defer_stage_events=true 时 apply() 只建 anchor index，不立即下发；decode/runtime
    // 到达 anchor 后再通过 events_for_anchor() 触发。
    std::function<void(const PlanEvent & ev)>             enqueue_load;
    std::function<void(const PlanEvent & ev)>             enqueue_transfer;
    std::function<void(const PlanEvent & ev)>             enqueue_transform;
    bool                                                  defer_stage_events = false;
};

// apply 的统计(单测断言 + 日志用)
struct ReconcileStats {
    int n_prefetch       = 0;  // 搬进 GPU 的 weight 数(want resident & 当前不在)
    int n_evict          = 0;  // 搬出 GPU 的 weight 数(当前在 & plan 不要)
    int n_already        = 0;  // 已符合,无需动
    int n_route_static   = 0;  // 灌了静态 backend 的 op 数
    int n_route_runtime  = 0;  // 交给 runtime dispatch 的 op 数(apply 不灌)
    int n_migrate        = 0;  // 标了迁移的 op 数
    int n_overlap_events = 0;  // 记录的 timeline 事件数
    int n_load_events    = 0;  // LOAD stage 数
    int n_transfer_events = 0;  // TRANSFER stage 数
    int n_xform_events   = 0;  // XFORM stage 数
};

class PlanExecutor {
public:
    PlanExecutor() = default;
    explicit PlanExecutor(ExecSinks sinks) : sinks_(std::move(sinks)) {}

    void set_sinks(ExecSinks sinks) { sinks_ = std::move(sinks); }

    // 应用一个 plan。plan 必须在 executor 使用期间保持存活(executor 只持指针,
    // 用于后续 runtime_dispatch / events_for_anchor 查询)。
    ReconcileStats apply(const ExecPlan & plan);

    // runtime-dispatch op:compute 时回调。给 op_id + split 默认 backend,
    // 返回 plan 指定的 backend 索引(Backend 的整数值);plan 没说或非 RUNTIME → 返回 default。
    int runtime_dispatch(int op_id, int default_backend) const;

    // anchor 在某个 op 上的所有 timeline 事件(decode 到该 op 时拿去 overlap enqueue)。
    const std::vector<const PlanEvent *> & events_for_anchor(int op_id) const;
    bool defer_stage_events() const { return sinks_.defer_stage_events; }

    const ExecPlan * current() const { return plan_; }

private:
    void build_anchor_index();

    ExecSinks       sinks_;
    const ExecPlan *plan_ = nullptr;
    std::unordered_map<int, std::vector<const PlanEvent *>> anchor_index_;
    static const std::vector<const PlanEvent *>            empty_events_;
};

}  // namespace elastic
