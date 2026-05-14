// runtime/metrics_logger.h
//
// 每 decode step 写一行 JSONL。字段与
// docs/elastic/baseline_external_execution.md §5 验收标准第 6 条对齐：
//   {t, B_t_mb, resident_bytes, n_blocks_resident, token_id,
//    layer_load_latency_ms, decode_latency_ms, flash_bytes_read}
//
// 设计：不引入 JSON 库依赖，手写序列化；按 flush_threshold（默认 64）条
// 调用 fflush，避免对 ggml/llama.cpp 引入额外编译期依赖。close 时强制 flush。
// 线程不安全 —— 调用方保证 write/close 之间没有竞争（decode 主循环是单线程的）。

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace elastic {

struct metrics_record {
    double   t_sec;                 // 相对启动的秒数
    size_t   B_t_mb;                // 当前预算
    size_t   resident_bytes;        // 当前所有权重在 GPU 上的字节
    int      n_blocks_resident;     // 当前驻留的 block 数
    int      token_id;              // 本 step 的 token id（采样后）
    double   layer_load_latency_ms; // 本 step 所有 ensure_resident 累计时间
    double   decode_latency_ms;     // 本 step 端到端时间
    size_t   flash_bytes_read;      // 本 step 从 flash 读到的字节
};

struct metrics_logger {
    FILE  *fp;
    size_t buffered_records;
    size_t flush_threshold;
};

// 成功返回 0，失败返回非 0 且不修改 lg。
int  metrics_logger_init(metrics_logger *lg,
                         const char *path,
                         size_t flush_threshold = 64);

void metrics_logger_write(metrics_logger *lg, const metrics_record &r);

void metrics_logger_flush(metrics_logger *lg);

void metrics_logger_close(metrics_logger *lg);

}  // namespace elastic
