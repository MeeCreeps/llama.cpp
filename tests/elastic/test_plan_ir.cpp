// tests/elastic/test_plan_ir.cpp
//
// Plan IR 单测:native schema round-trip + 加载现有 make_plan.py 格式 plan_*.json。

#include "plan_ir.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

using namespace elastic;

static int g_fail = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// 手构一个小 plan,验证 native JSON round-trip。
static void test_native_roundtrip() {
    ExecPlan p;
    p.schema_version    = 1;
    p.budget_mib        = 4096;
    p.kv_bytes          = 1234;
    p.misc_bytes        = 5678;
    p.pred_per_token_ms = 12.5;
    p.bottleneck        = "disk";

    WeightPlan w0; w0.weight_id = 0; w0.name = "blk.0.attn_q.weight"; w0.layer = 0;
    w0.byte_size = 1000; w0.location = Location::GPU; w0.pinned = true; w0.xform = Xform::GPU_CONVERT;
    WeightPlan w1; w1.weight_id = 1; w1.name = "blk.1.ffn_up.weight"; w1.layer = 1;
    w1.byte_size = 2000; w1.location = Location::DISK; w1.pinned = false; w1.xform = Xform::NONE;
    p.weights = {w0, w1};

    OpPlan o0; o0.op_id = 0; o0.name = "blk.0.attn_q.weight"; o0.layer = 0;
    o0.compute_backend = Backend::GPU; o0.weight_id = 0; o0.dispatch = Dispatch::STATIC;
    OpPlan o1; o1.op_id = 1; o1.name = "blk.1.ffn_up.weight"; o1.layer = 1;
    o1.compute_backend = Backend::GPU; o1.weight_id = 1; o1.dispatch = Dispatch::RUNTIME;
    o1.migrate = true; o1.migrate_from = Backend::CPU; o1.migrate_xform = Xform::GPU_CONVERT;
    p.ops = {o0, o1};

    PlanEvent e0; e0.kind = EvKind::LOAD; e0.weight_id = 1; e0.from_loc = Location::DISK;
    e0.to_loc = Location::CPU; e0.engine = Engine::DISK; e0.anchor_op_id = 0;
    PlanEvent e1; e1.kind = EvKind::XFORM; e1.weight_id = 1; e1.from_loc = Location::GPU;
    e1.to_loc = Location::GPU; e1.engine = Engine::GPU; e1.anchor_op_id = 0;
    p.timeline = {e0, e1};

    std::string s;
    CHECK(plan_to_json_string(p, s));
    CHECK(!s.empty());

    ExecPlan q;
    std::string err;
    CHECK(plan_from_json_string(s, q, &err));
    if (!err.empty()) std::fprintf(stderr, "  decode err: %s\n", err.c_str());

    CHECK(q.budget_mib == 4096);
    CHECK(q.kv_bytes == 1234);
    CHECK(q.misc_bytes == 5678);
    CHECK(q.bottleneck == "disk");
    CHECK(q.weights.size() == 2);
    CHECK(q.ops.size() == 2);
    CHECK(q.timeline.size() == 2);

    CHECK(q.weights[0].location == Location::GPU);
    CHECK(q.weights[0].pinned == true);
    CHECK(q.weights[0].xform == Xform::GPU_CONVERT);
    CHECK(q.weights[1].location == Location::DISK);

    CHECK(q.ops[1].dispatch == Dispatch::RUNTIME);
    CHECK(q.ops[1].migrate == true);
    CHECK(q.ops[1].migrate_from == Backend::CPU);
    CHECK(q.ops[1].migrate_xform == Xform::GPU_CONVERT);

    CHECK(q.timeline[0].engine == Engine::DISK);
    CHECK(q.timeline[0].kind == EvKind::LOAD);
    CHECK(q.timeline[1].engine == Engine::GPU);
    CHECK(q.timeline[1].kind == EvKind::XFORM);

    // 查询辅助
    CHECK(q.weight_id_of("blk.1.ffn_up.weight") == 1);
    CHECK(q.weight_id_of("nope") == -1);
    CHECK(q.weight_by_id(0) != nullptr);
    CHECK(q.weight_by_id(99) == nullptr);
    CHECK(q.resident_gpu_ids().size() == 1);
    CHECK(q.resident_gpu_ids()[0] == 0);
}

// 加载真实 make_plan.py 产出(若提供路径)。
static void test_make_plan_load(const char * path) {
    if (!path) {
        std::fprintf(stderr, "  (skip make_plan load: no path given)\n");
        return;
    }
    ExecPlan p;
    std::string err;
    bool ok = plan_from_make_plan_file(path, p, &err);
    if (!ok) {
        std::fprintf(stderr, "FAIL load %s: %s\n", path, err.c_str());
        ++g_fail;
        return;
    }
    std::fprintf(stderr,
                 "  loaded %s: budget=%lld weights=%zu ops=%zu timeline=%zu resident_gpu=%zu per_token=%.1fms bottleneck=%s\n",
                 path, (long long) p.budget_mib, p.weights.size(), p.ops.size(),
                 p.timeline.size(), p.resident_gpu_ids().size(),
                 p.pred_per_token_ms, p.bottleneck.c_str());

    CHECK(p.budget_mib > 0);
    CHECK(!p.weights.empty());
    CHECK(!p.ops.empty());
    // routes 有 197 个 → 至少有这么多 op(3B model)
    CHECK(p.ops.size() >= 100);
    // 每个 op 的 weight_id 必须有效
    for (const auto & o : p.ops) {
        CHECK(o.weight_id >= 0 && o.weight_id < (int) p.weights.size());
    }
    // GPU 算的 op,其 weight 若不在 GPU 应标 migrate
    for (const auto & o : p.ops) {
        const WeightPlan * w = p.weight_by_id(o.weight_id);
        CHECK(w != nullptr);
        if (o.compute_backend == Backend::GPU && w->location != Location::GPU) {
            CHECK(o.migrate == true);
        }
    }
    // timeline 事件的 weight_id / anchor 合法
    bool saw_load = false;
    bool saw_xform = false;
    for (const auto & e : p.timeline) {
        CHECK(e.weight_id >= 0 && e.weight_id < (int) p.weights.size());
        saw_load  = saw_load  || e.kind == EvKind::LOAD;
        saw_xform = saw_xform || e.kind == EvKind::XFORM;
    }
    CHECK(saw_load);
    CHECK(saw_xform);
    // round-trip native:加载后导出再加载,结构一致
    std::string s;
    CHECK(plan_to_json_string(p, s));
    ExecPlan q;
    CHECK(plan_from_json_string(s, q, &err));
    CHECK(q.weights.size() == p.weights.size());
    CHECK(q.ops.size() == p.ops.size());
    CHECK(q.timeline.size() == p.timeline.size());
}

static void test_make_plan_synth_gpu_stages() {
    const char * path = "/tmp/elastic_make_plan_synth_gpu_stages.json";
    {
        std::ofstream f(path);
        f << R"JSON({
  "budget_mib": 1024,
  "per_token_ms": 1.0,
  "bottleneck": "disk",
  "routes": {
    "blk.0.attn_q.weight": "gpu",
    "blk.0.attn_k.weight": "cpu"
  },
  "resident_in_memory": [],
  "schedule": [
    {
      "compute": {"weight": "blk.0.attn_q.weight", "backend": "gpu"},
      "disk_in": [
        {"weight": "blk.0.attn_q.weight"},
        {"weight": "blk.0.attn_k.weight"}
      ]
    }
  ]
})JSON";
    }

    ExecPlan p;
    std::string err;
    CHECK(plan_from_make_plan_file(path, p, &err));
    if (!err.empty()) std::fprintf(stderr, "  synth decode err: %s\n", err.c_str());

    int q_load = 0, q_transfer = 0, q_xform = 0;
    int k_load = 0, k_transfer = 0, k_xform = 0;
    const int qid = p.weight_id_of("blk.0.attn_q.weight");
    const int kid = p.weight_id_of("blk.0.attn_k.weight");
    CHECK(qid >= 0);
    CHECK(kid >= 0);
    for (const auto & e : p.timeline) {
        if (e.weight_id == qid) {
            q_load     += e.kind == EvKind::LOAD;
            q_transfer += e.kind == EvKind::TRANSFER;
            q_xform    += e.kind == EvKind::XFORM;
        }
        if (e.weight_id == kid) {
            k_load     += e.kind == EvKind::LOAD;
            k_transfer += e.kind == EvKind::TRANSFER;
            k_xform    += e.kind == EvKind::XFORM;
        }
    }
    CHECK(q_load == 1);
    CHECK(q_transfer == 1);
    CHECK(q_xform == 1);
    CHECK(k_load == 1);
    CHECK(k_transfer == 0);
    CHECK(k_xform == 0);
}

int main(int argc, char ** argv) {
    test_native_roundtrip();
    test_make_plan_synth_gpu_stages();
    // 可传一个真实 plan_*.json 路径;CMake 会传 runtime/plan/plans/plan_4144MiB.json
    test_make_plan_load(argc > 1 ? argv[1] : nullptr);

    if (g_fail == 0) {
        std::printf("test_plan_ir: ALL PASS\n");
        return 0;
    }
    std::printf("test_plan_ir: %d FAILED\n", g_fail);
    return 1;
}
