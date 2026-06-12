// runtime/plan_executor.cpp —— PlanExecutor 实现(见 plan_executor.h / IMPLEMENTATION.md §5)。

#include "plan_executor.h"

namespace elastic {

const std::vector<const PlanEvent *> PlanExecutor::empty_events_;

void PlanExecutor::build_anchor_index() {
    anchor_index_.clear();
    if (!plan_) return;
    for (const auto & e : plan_->timeline) {
        anchor_index_[e.anchor_op_id].push_back(&e);
    }
}

ReconcileStats PlanExecutor::apply(const ExecPlan & plan) {
    plan_ = &plan;
    build_anchor_index();

    ReconcileStats st;

    // 1) residency reconcile (D1):只对「当前状态 ≠ plan 期望」的 weight 发动作。
    //    plan 期望 GPU 常驻 = (location == GPU)。其余(CPU/DISK)= 不该占 GPU buffer。
    if (sinks_.set_resident) {
        for (const auto & w : plan.weights) {
            const bool want = (w.location == Location::GPU);
            bool cur = want;  // 没 is_resident 探针时,假设需要时就搬(保守:发 set_resident)
            if (sinks_.is_resident) {
                cur = sinks_.is_resident(w.weight_id);
            }
            if (sinks_.is_resident && cur == want) {
                st.n_already++;
                continue;
            }
            if (want) {
                sinks_.set_resident(w.weight_id, true);
                st.n_prefetch++;
            } else {
                // 只有「当前在 GPU」才需要发 evict;没探针时无法判断,保守发一次。
                if (!sinks_.is_resident || cur) {
                    sinks_.set_resident(w.weight_id, false);
                    st.n_evict++;
                } else {
                    st.n_already++;
                }
            }
        }
    }

    // 2) routing (D2b):STATIC op 灌 backend;RUNTIME op 留给 runtime_dispatch(apply 不灌)。
    for (const auto & o : plan.ops) {
        if (o.dispatch == Dispatch::RUNTIME) {
            st.n_route_runtime++;
        } else {
            if (sinks_.set_op_backend) sinks_.set_op_backend(o.op_id, o.compute_backend);
            st.n_route_static++;
        }
        // 3) migration (D2b):每个 op 都同步迁移意图(migrate=false 也下发,覆盖上个 plan 的残留)。
        if (sinks_.set_op_migrate) {
            sinks_.set_op_migrate(o.op_id, o.migrate, o.migrate_from, o.migrate_xform);
            if (o.migrate) st.n_migrate++;
        } else if (o.migrate) {
            st.n_migrate++;
        }
    }

    // 4) timeline (D3):已建 anchor 索引;若 sink 想立刻登记也下发一遍。
    st.n_overlap_events = (int) plan.timeline.size();
    for (const auto & e : plan.timeline) {
        if (sinks_.enqueue_overlapped) {
            sinks_.enqueue_overlapped(e);
        }
        switch (e.kind) {
            case EvKind::LOAD:
                st.n_load_events++;
                if (sinks_.enqueue_load) sinks_.enqueue_load(e);
                break;
            case EvKind::DMA:
                st.n_dma_events++;
                if (sinks_.enqueue_dma) sinks_.enqueue_dma(e);
                break;
            case EvKind::XFORM:
                st.n_xform_events++;
                if (sinks_.enqueue_transform) sinks_.enqueue_transform(e);
                break;
            case EvKind::PREFETCH:
            case EvKind::EVICT:
                break;
        }
    }

    return st;
}

int PlanExecutor::runtime_dispatch(int op_id, int default_backend) const {
    if (!plan_) return default_backend;
    const OpPlan * o = plan_->op_by_id(op_id);
    if (!o || o->dispatch != Dispatch::RUNTIME) return default_backend;
    return (int) o->compute_backend;
}

const std::vector<const PlanEvent *> & PlanExecutor::events_for_anchor(int op_id) const {
    auto it = anchor_index_.find(op_id);
    if (it == anchor_index_.end()) return empty_events_;
    return it->second;
}

}  // namespace elastic
