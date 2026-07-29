// tests/elastic/test_plan_executor.cpp
//
// PlanExecutor 单测:喂 mock sinks,验证 apply(plan) 产出的 residency / routing /
// migration / overlap 动作序列正确,以及 plan 切换时的差量 reconcile。

#include "plan_executor.h"
#include "plan_ir.h"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace elastic;

static int g_fail = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// mock 后端:用一个 set 模拟 GPU 驻留状态,记录所有动作。
struct MockBackend {
    std::set<int>                            resident;       // 当前在 GPU 的 weight_id
    std::vector<std::pair<int, bool>>        set_resident_log;
    std::vector<std::pair<int, Backend>>     route_log;
    std::vector<int>                         migrate_log;    // 标了 migrate 的 op_id
    int                                      overlap_count = 0;
    int                                      load_count = 0;
    int                                      transfer_count = 0;
    int                                      xform_count = 0;

    ExecSinks make_sinks() {
        ExecSinks s;
        s.set_resident = [this](int wid, bool want) {
            set_resident_log.push_back({wid, want});
            if (want) resident.insert(wid);
            else      resident.erase(wid);
        };
        s.is_resident = [this](int wid) { return resident.count(wid) > 0; };
        s.set_op_backend = [this](int op_id, Backend be) { route_log.push_back({op_id, be}); };
        s.set_op_migrate = [this](int op_id, bool mig, Backend, Xform) {
            if (mig) migrate_log.push_back(op_id);
        };
        s.enqueue_overlapped = [this](const PlanEvent &) { overlap_count++; };
        s.enqueue_load  = [this](const PlanEvent &) { load_count++; };
        s.enqueue_transfer = [this](const PlanEvent &) { transfer_count++; };
        s.enqueue_transform = [this](const PlanEvent &) { xform_count++; };
        return s;
    }
};

// 造一个 plan:n_layers 层,每层 1 个 weight+op。前 n_gpu 个 weight 放 GPU,其余 DISK。
static ExecPlan make_plan(int n_layers, int n_gpu) {
    ExecPlan p;
    p.budget_mib = 4096;
    for (int i = 0; i < n_layers; i++) {
        WeightPlan w;
        w.weight_id = i;
        w.name      = "blk." + std::to_string(i) + ".w";
        w.layer     = i;
        w.byte_size = 1000;
        w.location  = (i < n_gpu) ? Location::GPU : Location::DISK;
        w.pinned    = (i < n_gpu);
        w.xform     = (i < n_gpu) ? Xform::GPU_CONVERT : Xform::NONE;
        p.weights.push_back(w);

        OpPlan o;
        o.op_id           = i;
        o.name            = w.name;
        o.layer           = i;
        o.weight_id       = i;
        o.compute_backend = (i < n_gpu) ? Backend::GPU : Backend::CPU;
        o.dispatch        = Dispatch::STATIC;
        // DISK weight 在 CPU 算 → 无迁移;若强制在 GPU 算才迁移。这里 DISK→CPU 算,不迁移。
        p.ops.push_back(o);

        // 给 DISK weight 一个 load 事件,anchor 到前一个 op
        if (i >= n_gpu) {
            PlanEvent e;
            e.kind = EvKind::LOAD; e.weight_id = i;
            e.from_loc = Location::DISK; e.to_loc = Location::CPU; e.engine = Engine::DISK;
            e.anchor_op_id = (i > 0) ? i - 1 : 0;
            p.timeline.push_back(e);
        }
    }
    return p;
}

// 全新 ctx(无探针默认)apply:GPU weight 应全部 prefetch。
static void test_fresh_apply() {
    MockBackend mb;
    PlanExecutor ex(mb.make_sinks());
    ExecPlan p = make_plan(10, 4);  // 4 GPU + 6 DISK

    ReconcileStats st = ex.apply(p);
    CHECK(st.n_prefetch == 4);     // 4 个 GPU weight 从无到有
    CHECK(st.n_evict == 0);
    CHECK(st.n_route_static == 10);
    CHECK(st.n_route_runtime == 0);
    CHECK(st.n_overlap_events == 6);
    CHECK(st.n_load_events == 6);
    CHECK(mb.load_count == 6);
    CHECK((int) mb.resident.size() == 4);
    CHECK(mb.resident.count(0) && mb.resident.count(3));
    CHECK(!mb.resident.count(4));
}

// plan 切换:从 {0..3 GPU} 切到 {2..5 GPU} → evict 0,1;prefetch 4,5;保留 2,3。
static void test_replan_diff() {
    MockBackend mb;
    PlanExecutor ex(mb.make_sinks());

    ExecPlan p1 = make_plan(10, 4);   // GPU = {0,1,2,3}
    ex.apply(p1);
    CHECK(mb.resident.size() == 4);

    // 造 p2:GPU = {2,3,4,5}
    ExecPlan p2 = make_plan(10, 0);   // 先全 DISK
    for (int i = 2; i <= 5; i++) {
        p2.weights[i].location = Location::GPU;
        p2.weights[i].pinned   = true;
        p2.ops[i].compute_backend = Backend::GPU;
    }
    mb.set_resident_log.clear();
    ReconcileStats st = ex.apply(p2);

    // 期望:evict {0,1}(2 个),prefetch {4,5}(2 个),{2,3} 已在 → already
    CHECK(st.n_prefetch == 2);
    CHECK(st.n_evict == 2);
    CHECK(mb.resident.count(2) && mb.resident.count(3));
    CHECK(mb.resident.count(4) && mb.resident.count(5));
    CHECK(!mb.resident.count(0) && !mb.resident.count(1));
    CHECK(mb.resident.size() == 4);
}

// runtime dispatch:RUNTIME op 不在 apply 灌 backend,改由 runtime_dispatch 查 plan。
static void test_runtime_dispatch() {
    MockBackend mb;
    PlanExecutor ex(mb.make_sinks());
    ExecPlan p = make_plan(4, 0);
    // 把 op 2 设成 RUNTIME + GPU
    p.ops[2].dispatch = Dispatch::RUNTIME;
    p.ops[2].compute_backend = Backend::GPU;

    ReconcileStats st = ex.apply(p);
    CHECK(st.n_route_runtime == 1);
    CHECK(st.n_route_static == 3);

    // runtime_dispatch:op2 → GPU(=1);其他 op 非 RUNTIME → 返回 default
    CHECK(ex.runtime_dispatch(2, 0) == (int) Backend::GPU);
    CHECK(ex.runtime_dispatch(1, 7) == 7);   // op1 STATIC → default 透传
    CHECK(ex.runtime_dispatch(99, 5) == 5);  // 不存在 → default
}

// anchor 索引:events_for_anchor 把事件按 anchor op 归类。
static void test_anchor_index() {
    MockBackend mb;
    PlanExecutor ex(mb.make_sinks());
    ExecPlan p = make_plan(10, 4);  // DISK weight 4..9,各一个 prefetch,anchor = i-1
    ex.apply(p);

    // weight 5 的 prefetch anchor 到 op 4
    auto & evs = ex.events_for_anchor(4);
    CHECK(evs.size() == 1);
    CHECK(evs[0]->weight_id == 5);
    CHECK(evs[0]->kind == EvKind::LOAD);

    // 没有事件 anchor 到 op 8 的 weight 9 → anchor 8
    CHECK(ex.events_for_anchor(8).size() == 1);
    // op 0 没有 prefetch anchor 到它(weight4 anchor 到 op3)
    CHECK(ex.events_for_anchor(100).size() == 0);
}

// defer_stage_events:apply 只建 timeline/计数，不立即调用 stage sinks。
static void test_deferred_stage_events() {
    MockBackend mb;
    ExecSinks sinks = mb.make_sinks();
    sinks.defer_stage_events = true;
    PlanExecutor ex(std::move(sinks));
    ExecPlan p = make_plan(10, 4);

    ReconcileStats st = ex.apply(p);
    CHECK(st.n_overlap_events == 6);
    CHECK(st.n_load_events == 6);
    CHECK(mb.overlap_count == 0);
    CHECK(mb.load_count == 0);
    CHECK(mb.transfer_count == 0);
    CHECK(mb.xform_count == 0);
    CHECK(ex.events_for_anchor(4).size() == 1);
}

// The mixed-granularity runner delegates recurring LOAD/PREPARE to the
// backend's physical-unit pipeline.  "none" must therefore remove the legacy
// complete-weight stage anchors even though their diagnostic counts remain in
// the placement plan.
static void test_unit_pipeline_stage_authority() {
    const char * previous =
        std::getenv("LLAMA_ELASTIC_INTERVAL_STAGE_KINDS");
    const bool had_previous = previous != nullptr;
    const std::string previous_value =
        previous ? previous : "";
    setenv("LLAMA_ELASTIC_INTERVAL_STAGE_KINDS", "none", 1);

    MockBackend mb;
    ExecSinks sinks = mb.make_sinks();
    sinks.defer_stage_events = true;
    PlanExecutor ex(std::move(sinks));
    ExecPlan p = make_plan(10, 4);

    ReconcileStats st = ex.apply(p);
    CHECK(st.n_load_events == 6);
    CHECK(mb.load_count == 0);
    CHECK(ex.events_for_anchor(4).empty());
    CHECK(ex.events_for_anchor(8).empty());

    if (had_previous) {
        setenv(
            "LLAMA_ELASTIC_INTERVAL_STAGE_KINDS",
            previous_value.c_str(), 1);
    } else {
        unsetenv("LLAMA_ELASTIC_INTERVAL_STAGE_KINDS");
    }
}

int main() {
    test_fresh_apply();
    test_replan_diff();
    test_runtime_dispatch();
    test_anchor_index();
    test_deferred_stage_events();
    test_unit_pipeline_stage_authority();

    if (g_fail == 0) {
        std::printf("test_plan_executor: ALL PASS\n");
        return 0;
    }
    std::printf("test_plan_executor: %d FAILED\n", g_fail);
    return 1;
}
