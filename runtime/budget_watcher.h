// runtime/budget_watcher.h
//
// 从 CSV trace 文件加载 (time_sec, budget_mb) 预算序列，后台线程跟随
// steady_clock 时间，按 step 或 linear 方式内插出当前 B(t)，原子地暴露给
// 主线程非阻塞读取。
//
// 设计点（与 docs/elastic/baseline_external_execution.md §3 对齐）：
//   - 时间原点 t0 = budget_watcher_init() 返回前一刻的 steady_clock::now()
//   - t < schedule[0].first 时钳到首点，t >= schedule.back().first 时钳到尾点
//   - 首行若不能解析为数值则当作 header 跳过，# 开头与空行忽略
//   - schedule 在加载后按 time_sec 升序稳定排序
//   - m_floor_mb = min over schedule 的 budget_mb，启动时计算一次

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <pthread.h>
#include <string>
#include <vector>

namespace elastic {

struct budget_watcher {
    enum interp_mode {
        INTERP_STEP   = 0,
        INTERP_LINEAR = 1,
    };

    // 公开字段（保持 C 风格直接访问，方便从 llama.cpp 老路径使用）
    std::atomic<size_t> current_budget_mb;
    std::atomic<bool>   stop_flag;
    pthread_t           thread;
    bool                thread_started;

    std::vector<std::pair<double, size_t>> schedule;  // (time_sec, budget_mb)
    size_t              m_floor_mb;                   // min B(t)，启动时算好
    interp_mode         mode;
    int                 tick_period_ms;               // 后台线程采样周期

    std::chrono::steady_clock::time_point t0;
};

struct budget_watcher_config {
    budget_watcher::interp_mode mode = budget_watcher::INTERP_LINEAR;
    int tick_period_ms               = 50;
};

// 加载 CSV 并启动后台线程。成功返回 0，失败返回非 0 且不启动线程。
int  budget_watcher_init(budget_watcher *bw,
                         const char *csv_path,
                         const budget_watcher_config &cfg = budget_watcher_config{});

// 非阻塞读当前 B(t)。
size_t budget_watcher_get(const budget_watcher *bw);

// 将 replay 时间原点重置为当前时刻，并立即写入 t=0 的预算。
void budget_watcher_reset_clock(budget_watcher *bw);

// 停止线程并清理。可重入安全。
void budget_watcher_shutdown(budget_watcher *bw);

// —— 仅供单测 / 离线 replay 使用的纯函数 —— //
//
// 在给定的 schedule 上按 mode 内插 t_sec 处的预算值。schedule 必须已按
// time_sec 升序排序且至少有 1 个元素。
size_t budget_watcher_interp_at(const std::vector<std::pair<double, size_t>> &schedule,
                                budget_watcher::interp_mode mode,
                                double t_sec);

// 仅加载 CSV，不启动后台线程。便于纯逻辑测试。失败返回非 0。
int  budget_watcher_load_csv(budget_watcher *bw, const char *csv_path);

}  // namespace elastic
