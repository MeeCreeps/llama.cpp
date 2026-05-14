// tests/elastic/test_budget_watcher.cpp
//
// 覆盖 BudgetWatcher 的纯逻辑（内插）与端到端（CSV 加载 + 后台线程 +
// 原子读）。失败用断言直接退出。不引入 gtest，与 llama.cpp 上游 tests/
// 风格一致。

#include "budget_watcher.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using elastic::budget_watcher;
using elastic::budget_watcher_config;
using elastic::budget_watcher_get;
using elastic::budget_watcher_init;
using elastic::budget_watcher_interp_at;
using elastic::budget_watcher_load_csv;
using elastic::budget_watcher_shutdown;

namespace {

// 写一份临时 CSV 文件，返回路径
std::string write_tmp(const char *name, const std::string &content) {
    std::string path = std::string("/tmp/") + name;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << content;
    f.close();
    return path;
}

void check_eq_sz(size_t got, size_t want, const char *msg) {
    if (got != want) {
        std::fprintf(stderr, "断言失败 [%s]: got=%zu want=%zu\n", msg, got, want);
        std::abort();
    }
}

void check_near_sz(size_t got, size_t want, size_t tol, const char *msg) {
    size_t d = got > want ? got - want : want - got;
    if (d > tol) {
        std::fprintf(stderr,
            "断言失败 [%s]: got=%zu want=%zu tol=%zu\n", msg, got, want, tol);
        std::abort();
    }
}

// —— 用例 1：纯线性插值 ——
void test_interp_linear() {
    std::vector<std::pair<double, size_t>> sched = {
        {0.0, 1000}, {10.0, 2000},
    };
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR,  0.0), 1000, "linear@0");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR, 10.0), 2000, "linear@10");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR,  5.0), 1500, "linear@5");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR,  2.5), 1250, "linear@2.5");
    std::printf("[OK] test_interp_linear\n");
}

// —— 用例 2：纯 step 插值 ——
void test_interp_step() {
    std::vector<std::pair<double, size_t>> sched = {
        {0.0, 1000}, {10.0, 2000}, {20.0, 500},
    };
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_STEP,  0.0), 1000, "step@0");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_STEP,  5.0), 1000, "step@5");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_STEP,  9.999), 1000, "step@9.999");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_STEP, 10.0), 2000, "step@10");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_STEP, 15.0), 2000, "step@15");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_STEP, 20.0), 500,  "step@20");
    std::printf("[OK] test_interp_step\n");
}

// —— 用例 3：边界钳位 ——
void test_clamp() {
    std::vector<std::pair<double, size_t>> sched = {
        {5.0, 800}, {15.0, 1200},
    };
    // 早于首点
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR, -1.0), 800, "clamp head -1");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR,  0.0), 800, "clamp head 0");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR,  4.9), 800, "clamp head 4.9");
    // 晚于尾点
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR, 15.0),  1200, "clamp tail 15");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR, 99.0),  1200, "clamp tail 99");
    std::printf("[OK] test_clamp\n");
}

// —— 用例 4：CSV header 容错 + # 注释 + 空行 ——
void test_csv_header_and_comments() {
    std::string csv =
        "# 这是一份带 header 的 CSV，watcher 应该跳过 header\n"
        "time_sec,budget_mb\n"
        "\n"
        "0,4000\n"
        "# 中途注释\n"
        "10,1500\n"
        "20,4000\n";
    std::string p = write_tmp("bw_test_header.csv", csv);

    budget_watcher bw{};
    [[maybe_unused]] int rc = budget_watcher_load_csv(&bw, p.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "load_csv 失败 rc=%d\n", rc);
        std::abort();
    }
    assert(bw.schedule.size() == 3);
    check_eq_sz(bw.schedule[0].second, 4000, "csv row0");
    check_eq_sz(bw.schedule[1].second, 1500, "csv row1");
    check_eq_sz(bw.schedule[2].second, 4000, "csv row2");
    check_eq_sz(bw.m_floor_mb, 1500, "m_floor");
    std::printf("[OK] test_csv_header_and_comments\n");
}

// —— 用例 5：CSV 乱序，加载后内部按时间升序排好 ——
void test_csv_unsorted() {
    std::string csv = "time_sec,budget_mb\n20,500\n0,3000\n10,1500\n";
    std::string p = write_tmp("bw_test_unsorted.csv", csv);

    budget_watcher bw{};
    [[maybe_unused]] int rc = budget_watcher_load_csv(&bw, p.c_str());
    assert(rc == 0);
    assert(bw.schedule.size() == 3);
    assert(bw.schedule[0].first == 0.0);
    assert(bw.schedule[1].first == 10.0);
    assert(bw.schedule[2].first == 20.0);
    check_eq_sz(bw.m_floor_mb, 500, "m_floor unsorted");
    std::printf("[OK] test_csv_unsorted\n");
}

// —— 用例 6：端到端 init + 后台线程 + 原子读 ——
//
// 用一段短 CSV，0s=1000，0.5s=2000。等约 250 ms 后读，期望约 1500（±200）。
void test_end_to_end() {
    std::string csv = "time_sec,budget_mb\n0.0,1000\n0.5,2000\n";
    std::string p = write_tmp("bw_test_e2e.csv", csv);

    budget_watcher bw{};
    budget_watcher_config cfg;
    cfg.mode           = budget_watcher::INTERP_LINEAR;
    cfg.tick_period_ms = 10;
    [[maybe_unused]] int rc = budget_watcher_init(&bw, p.c_str(), cfg);
    assert(rc == 0);

    // init 时立刻 get 应为 t=0 处的 1000（init 已把 t=0 处的值预填进 atomic）
    check_eq_sz(budget_watcher_get(&bw), 1000, "t=0 即刻");

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    size_t mid = budget_watcher_get(&bw);
    // 给比较宽的容忍：线程调度抖动 + 时间不精确
    check_near_sz(mid, 1500, 250, "t~0.25 中点");

    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    size_t end = budget_watcher_get(&bw);
    check_eq_sz(end, 2000, "t>=0.5 钳尾");

    budget_watcher_shutdown(&bw);
    std::printf("[OK] test_end_to_end\n");
}

// —— 用例 7：单点 schedule 处处返回该值 ——
void test_single_point() {
    std::vector<std::pair<double, size_t>> sched = {{0.0, 1234}};
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR, -10.0), 1234, "single -10");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR,   0.0), 1234, "single 0");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_LINEAR, 999.0), 1234, "single 999");
    check_eq_sz(budget_watcher_interp_at(sched, budget_watcher::INTERP_STEP,    50.0), 1234, "single step 50");
    std::printf("[OK] test_single_point\n");
}

}  // namespace

int main() {
    test_interp_linear();
    test_interp_step();
    test_clamp();
    test_csv_header_and_comments();
    test_csv_unsorted();
    test_single_point();
    test_end_to_end();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
