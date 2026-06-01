// runtime/budget_watcher.cpp —— 见 budget_watcher.h 的实现注释

#include "budget_watcher.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace elastic {

namespace {

// 把字符串两端的空白去掉
std::string trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

// 尝试把字符串解析为 double；成功写到 out 并返回 true
bool try_parse_double(const std::string &s, double &out) {
    if (s.empty()) return false;
    const char *p = s.c_str();
    char *end     = nullptr;
    errno         = 0;
    double v      = std::strtod(p, &end);
    if (end == p || *end != '\0' || errno == ERANGE) return false;
    out = v;
    return true;
}

void *budget_watcher_thread_main(void *arg) {
    auto *bw = static_cast<budget_watcher *>(arg);
    while (!bw->stop_flag.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        const double t_sec =
            std::chrono::duration<double>(now - bw->t0).count();
        const size_t b = budget_watcher_interp_at(bw->schedule, bw->mode, t_sec);
        bw->current_budget_mb.store(b, std::memory_order_release);

        // 用 nanosleep 这样停时延迟最多一个 tick；不需要更精细的唤醒
        struct timespec ts;
        ts.tv_sec  = bw->tick_period_ms / 1000;
        ts.tv_nsec = (bw->tick_period_ms % 1000) * 1000000L;
        nanosleep(&ts, nullptr);
    }
    return nullptr;
}

}  // namespace

size_t budget_watcher_interp_at(const std::vector<std::pair<double, size_t>> &schedule,
                                budget_watcher::interp_mode mode,
                                double t_sec) {
    if (schedule.empty()) return 0;
    if (t_sec <= schedule.front().first) return schedule.front().second;
    if (t_sec >= schedule.back().first)  return schedule.back().second;

    // 二分查找第一个 time > t_sec 的位置
    auto it = std::upper_bound(
        schedule.begin(), schedule.end(), t_sec,
        [](double v, const std::pair<double, size_t> &row) {
            return v < row.first;
        });
    // 因为上面已经处理了 t >= back，所以 it 不会是 end
    const auto &hi = *it;
    const auto &lo = *(it - 1);

    if (mode == budget_watcher::INTERP_STEP) {
        return lo.second;
    }

    // 线性插值：注意预算单位 size_t，避免无符号下溢
    const double dt = hi.first - lo.first;
    if (dt <= 0.0) return lo.second;
    const double frac = (t_sec - lo.first) / dt;
    const double lo_v = static_cast<double>(lo.second);
    const double hi_v = static_cast<double>(hi.second);
    const double v    = lo_v + (hi_v - lo_v) * frac;
    if (v <= 0.0) return 0;
    return static_cast<size_t>(v + 0.5);
}

int budget_watcher_load_csv(budget_watcher *bw, const char *csv_path) {
    if (!bw || !csv_path) return -1;

    std::ifstream f(csv_path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[budget_watcher] 无法打开 CSV: %s\n", csv_path);
        return -2;
    }

    bw->schedule.clear();

    // 时间列单位换算成秒. CSV header 决定: "t_ms" → ÷1000, "t_sec"/"time_sec" → ×1.
    // 历史上所有 trace CSV 都是 t_ms (0,100,...,60000), 但 schedule + 线程查询用秒,
    // 不换算的话 t_sec(0~30) 永远落在头两行之间 → budget 冻在第一个值 (老 bug).
    // 无 header 时默认按 ms (现存 CSV 全是 ms), 避免再踩冻结坑.
    double time_scale = 0.001;

    std::string line;
    bool first_data_line = true;
    size_t line_no       = 0;
    while (std::getline(f, line)) {
        ++line_no;
        std::string t = trim(line);
        if (t.empty()) continue;
        if (t[0] == '#') continue;

        // 切两列
        std::string col0, col1;
        const size_t comma = t.find(',');
        if (comma == std::string::npos) {
            std::fprintf(stderr,
                "[budget_watcher] 第 %zu 行缺少逗号: %s\n", line_no, t.c_str());
            return -3;
        }
        col0 = trim(t.substr(0, comma));
        col1 = trim(t.substr(comma + 1));

        double ts = 0.0, bmb = 0.0;
        if (!try_parse_double(col0, ts) || !try_parse_double(col1, bmb)) {
            if (first_data_line) {
                // 首行不能解析当作 header 跳过. 顺便看时间列单位: 含 "sec" 用秒,
                // 含 "ms" 用毫秒. ("t_ms" 命中 ms; "t_sec"/"time_sec" 命中 sec)
                std::string h = col0;
                for (auto &c : h) c = (char)tolower((unsigned char)c);
                if (h.find("sec") != std::string::npos)      time_scale = 1.0;
                else if (h.find("ms") != std::string::npos)  time_scale = 0.001;
                first_data_line = false;
                continue;
            }
            std::fprintf(stderr,
                "[budget_watcher] 第 %zu 行解析失败: %s\n", line_no, t.c_str());
            return -4;
        }
        first_data_line = false;
        if (bmb < 0.0) bmb = 0.0;
        bw->schedule.emplace_back(ts * time_scale, static_cast<size_t>(bmb + 0.5));
    }

    if (bw->schedule.empty()) {
        std::fprintf(stderr, "[budget_watcher] CSV 没有数据行\n");
        return -5;
    }

    std::stable_sort(bw->schedule.begin(), bw->schedule.end(),
        [](const std::pair<double, size_t> &a, const std::pair<double, size_t> &b) {
            return a.first < b.first;
        });

    size_t m_floor = bw->schedule.front().second;
    for (const auto &row : bw->schedule) {
        if (row.second < m_floor) m_floor = row.second;
    }
    bw->m_floor_mb = m_floor;
    return 0;
}

int budget_watcher_init(budget_watcher *bw,
                        const char *csv_path,
                        const budget_watcher_config &cfg) {
    if (!bw) return -1;

    bw->current_budget_mb.store(0);
    bw->stop_flag.store(false);
    bw->thread_started  = false;
    bw->schedule.clear();
    bw->m_floor_mb      = 0;
    bw->mode            = cfg.mode;
    bw->tick_period_ms  = cfg.tick_period_ms > 0 ? cfg.tick_period_ms : 50;

    int rc = budget_watcher_load_csv(bw, csv_path);
    if (rc != 0) return rc;

    // 启动前先把 t0 设好并把 t=0 处的 B 写到原子量，保证立刻 get 也有合理值
    bw->t0 = std::chrono::steady_clock::now();
    bw->current_budget_mb.store(
        budget_watcher_interp_at(bw->schedule, bw->mode, 0.0),
        std::memory_order_release);

    if (pthread_create(&bw->thread, nullptr, budget_watcher_thread_main, bw) != 0) {
        std::fprintf(stderr, "[budget_watcher] pthread_create 失败\n");
        return -6;
    }
    bw->thread_started = true;
    return 0;
}

size_t budget_watcher_get(const budget_watcher *bw) {
    if (!bw) return 0;
    return bw->current_budget_mb.load(std::memory_order_acquire);
}

void budget_watcher_shutdown(budget_watcher *bw) {
    if (!bw) return;
    if (bw->thread_started) {
        bw->stop_flag.store(true, std::memory_order_release);
        pthread_join(bw->thread, nullptr);
        bw->thread_started = false;
    }
    bw->schedule.clear();
}

}  // namespace elastic
