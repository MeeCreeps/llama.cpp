// runtime/plan_executor.cpp —— PlanExecutor 实现(见 plan_executor.h / IMPLEMENTATION.md §5)。

#include "plan_executor.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <unordered_map>

namespace elastic {

const std::vector<const PlanEvent *> PlanExecutor::empty_events_;

bool PlanExecutor::use_interval_schedule() const {
    if (!plan_ || plan_->schedule_events.empty()) return false;
    const char * e = std::getenv("LLAMA_ELASTIC_USE_INTERVAL_SCHEDULE");
    return e && *e && *e != '0';
}

static bool interval_enable_cpu_xform_stage() {
    const char * e = std::getenv("LLAMA_ELASTIC_ENABLE_CPU_XFORM_STAGE");
    return e && *e && *e != 0;
}

static bool interval_project_stage_kind(const std::string & kind) {
    const char * e = std::getenv("LLAMA_ELASTIC_INTERVAL_STAGE_KINDS");
    if (!e || !*e) {
        return kind == "load";
    }
    std::string spec(e);
    size_t pos = 0;
    while (pos <= spec.size()) {
        size_t comma = spec.find(',', pos);
        std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c); }), item.end());
        if (item == kind || item == "all") return true;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return false;
}


static const char * event_kind_name(EvKind kind) {
    switch (kind) {
        case EvKind::LOAD:     return "load";
        case EvKind::TRANSFER: return "transfer";
        case EvKind::XFORM:    return "xform";
        case EvKind::PREFETCH: return "prefetch";
        case EvKind::EVICT:    return "evict";
    }
    return "";
}

void PlanExecutor::build_anchor_index() {
    anchor_index_.clear();
    projected_schedule_events_.clear();
    if (!plan_) return;

    if (use_interval_schedule()) {
        std::unordered_map<int, int> weight_to_op;
        for (const auto & op : plan_->ops) {
            if (op.weight_id >= 0 && weight_to_op.find(op.weight_id) == weight_to_op.end()) {
                weight_to_op.emplace(op.weight_id, op.op_id);
            }
        }

        std::unordered_map<int, int> compute_anchor_by_weight;
        const char * gpu_only_env = std::getenv("LLAMA_ELASTIC_INTERVAL_GPU_ANCHORS_ONLY");
        const bool gpu_only = gpu_only_env && *gpu_only_env && *gpu_only_env != '0';
        for (const auto & ev : plan_->schedule_events) {
            if (ev.kind != "compute") continue;
            if (gpu_only && ev.engine != "compute_gpu") continue;
            int op_id = ev.anchor_op_id;
            if (op_id < 0) {
                auto it = weight_to_op.find(ev.weight_id);
                if (it == weight_to_op.end()) continue;
                op_id = it->second;
            }
            auto hit = compute_anchor_by_weight.find(ev.weight_id);
            if (hit == compute_anchor_by_weight.end() || op_id < hit->second) {
                compute_anchor_by_weight[ev.weight_id] = op_id;
            }
        }

        for (const auto & ev : plan_->schedule_events) {
            if (ev.kind != "load" && ev.kind != "transfer" && ev.kind != "xform") {
                continue;
            }
            if (!interval_project_stage_kind(ev.kind)) {
                continue;
            }
            if (ev.kind == "xform" && ev.engine == "xform_cpu" && !interval_enable_cpu_xform_stage()) {
                continue;
            }
            PlanEvent pe;
            pe.weight_id = ev.weight_id;
            pe.anchor_op_id = ev.anchor_op_id;
            if (pe.anchor_op_id < 0) {
                auto cit = compute_anchor_by_weight.find(ev.weight_id);
                if (cit != compute_anchor_by_weight.end()) {
                    pe.anchor_op_id = cit->second;
                } else {
                    auto wit = weight_to_op.find(ev.weight_id);
                    pe.anchor_op_id = wit == weight_to_op.end() ? 0 : wit->second;
                }
            }
            pe.overlap_group = pe.anchor_op_id;
            if (ev.kind == "load") {
                pe.kind = EvKind::LOAD;
                pe.from_loc = Location::DISK;
                pe.to_loc = Location::CPU;
                pe.engine = ev.choice.find("gpu") != std::string::npos ? Engine::GPU : Engine::CPU;
            } else if (ev.kind == "transfer") {
                pe.kind = EvKind::TRANSFER;
                pe.from_loc = Location::CPU;
                pe.to_loc = Location::GPU;
                pe.engine = Engine::TRANSFER;
            } else {
                pe.kind = EvKind::XFORM;
                pe.from_loc = (ev.engine == "xform_gpu" || ev.choice.find("gpu") != std::string::npos) ? Location::GPU : Location::CPU;
                pe.to_loc = pe.from_loc;
                pe.engine = pe.to_loc == Location::GPU ? Engine::GPU : Engine::CPU;
            }
            projected_schedule_events_.push_back(pe);
        }

        for (const auto & e : projected_schedule_events_) {
            anchor_index_[e.anchor_op_id].push_back(&e);
        }
        return;
    }

    for (const auto & e : plan_->timeline) {
        if ((e.kind == EvKind::LOAD || e.kind == EvKind::TRANSFER || e.kind == EvKind::XFORM) &&
            !interval_project_stage_kind(event_kind_name(e.kind))) {
            continue;
        }
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

    // 4) timeline (D3):已建 anchor 索引;默认兼容旧路径, sink 可立刻下发。
    // defer_stage_events=true 时只计数/建索引, 由 decode/runtime 到达 anchor 后
    // 通过 events_for_anchor() 精确触发。
    st.n_schedule_events = (int) projected_schedule_events_.size();
    st.used_interval_schedule = use_interval_schedule();
    st.n_overlap_events = st.used_interval_schedule ? st.n_schedule_events : (int) plan.timeline.size();
    if (st.used_interval_schedule) {
        for (const auto & e : projected_schedule_events_) {
            switch (e.kind) {
                case EvKind::LOAD:     st.n_load_events++; break;
                case EvKind::TRANSFER: st.n_transfer_events++; break;
                case EvKind::XFORM:    st.n_xform_events++; break;
                case EvKind::PREFETCH:
                case EvKind::EVICT:    break;
            }
        }
    }
    for (const auto & e : plan.timeline) {
        if (!sinks_.defer_stage_events && sinks_.enqueue_overlapped) {
            sinks_.enqueue_overlapped(e);
        }
        switch (e.kind) {
            case EvKind::LOAD:
                if (!st.used_interval_schedule) st.n_load_events++;
                if (!sinks_.defer_stage_events && sinks_.enqueue_load) sinks_.enqueue_load(e);
                break;
            case EvKind::TRANSFER:
                if (!st.used_interval_schedule) st.n_transfer_events++;
                if (!sinks_.defer_stage_events && sinks_.enqueue_transfer) sinks_.enqueue_transfer(e);
                break;
            case EvKind::XFORM:
                if (!st.used_interval_schedule) st.n_xform_events++;
                if (!sinks_.defer_stage_events && sinks_.enqueue_transform) sinks_.enqueue_transform(e);
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
