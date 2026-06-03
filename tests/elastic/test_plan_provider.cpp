// tests/elastic/test_plan_provider.cpp
//
// PlanProvider 单测:table(查 index.json 选档 + 指针稳定)+ callback。

#include "plan_provider.h"
#include "plan_ir.h"

#include <cstdio>
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

static void test_table(const char * plans_dir) {
    if (!plans_dir) {
        std::fprintf(stderr, "  (skip table provider: no plans_dir given)\n");
        return;
    }
    std::string err;
    auto pp = PlanProvider::create_table(plans_dir, &err);
    if (!pp) {
        std::fprintf(stderr, "FAIL create_table(%s): %s\n", plans_dir, err.c_str());
        ++g_fail;
        return;
    }
    std::fprintf(stderr, "  table provider: %d bands\n", pp->n_bands());
    CHECK(pp->n_bands() > 1);

    // 大预算 → 拿到一个 plan
    const ExecPlan * hi = pp->get(7000);
    CHECK(hi != nullptr);
    if (hi) {
        std::fprintf(stderr, "  get(7000) → budget=%lld weights=%zu ops=%zu\n",
                     (long long) hi->budget_mib, hi->weights.size(), hi->ops.size());
        CHECK(hi->budget_mib <= 7000);   // 选 ≤ 预算的档
        CHECK(!hi->ops.empty());
    }

    // 同档指针稳定:同预算多次 get → 同指针
    const ExecPlan * a = pp->get(4200);
    const ExecPlan * b = pp->get(4200);
    CHECK(a == b);

    // 不同档 → 不同 plan(预算差很大时)
    const ExecPlan * lo = pp->get(2500);
    CHECK(lo != nullptr);
    if (lo && hi) CHECK(lo->budget_mib <= hi->budget_mib);

    // 极小预算 → 退回最小档(非空)
    const ExecPlan * tiny = pp->get(1);
    CHECK(tiny != nullptr);
}

static void test_callback() {
    int call_count = 0;
    auto pp = PlanProvider::create_callback([&](int64_t budget, ExecPlan & out) {
        call_count++;
        out.budget_mib = budget;
        WeightPlan w; w.weight_id = 0; w.name = "w0";
        w.location = (budget >= 4000) ? Location::GPU : Location::DISK;
        out.weights.push_back(w);
        OpPlan o; o.op_id = 0; o.name = "w0"; o.weight_id = 0;
        o.compute_backend = (budget >= 4000) ? Backend::GPU : Backend::CPU;
        out.ops.push_back(o);
        return true;
    });
    CHECK(pp != nullptr);

    const ExecPlan * p1 = pp->get(5000);
    CHECK(p1 != nullptr);
    CHECK(p1->weights[0].location == Location::GPU);

    // 同 budget → 同指针,callback 只调一次
    const ExecPlan * p1b = pp->get(5000);
    CHECK(p1 == p1b);
    CHECK(call_count == 1);

    // 不同 budget → 新 plan
    const ExecPlan * p2 = pp->get(3000);
    CHECK(p2 != nullptr);
    CHECK(p2 != p1);
    CHECK(p2->weights[0].location == Location::DISK);
    CHECK(call_count == 2);
}

int main(int argc, char ** argv) {
    test_table(argc > 1 ? argv[1] : nullptr);
    test_callback();

    if (g_fail == 0) {
        std::printf("test_plan_provider: ALL PASS\n");
        return 0;
    }
    std::printf("test_plan_provider: %d FAILED\n", g_fail);
    return 1;
}
