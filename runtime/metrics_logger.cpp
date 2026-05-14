// runtime/metrics_logger.cpp —— 见 metrics_logger.h

#include "metrics_logger.h"

#include <cstdio>
#include <cstring>

namespace elastic {

int metrics_logger_init(metrics_logger *lg, const char *path, size_t flush_threshold) {
    if (!lg || !path) return -1;
    FILE *fp = std::fopen(path, "wb");
    if (!fp) {
        std::fprintf(stderr, "[metrics_logger] 无法打开日志文件: %s\n", path);
        return -2;
    }
    lg->fp                = fp;
    lg->buffered_records  = 0;
    lg->flush_threshold   = flush_threshold > 0 ? flush_threshold : 1;
    return 0;
}

void metrics_logger_write(metrics_logger *lg, const metrics_record &r) {
    if (!lg || !lg->fp) return;
    // 手写 JSON：固定字段顺序，浮点保留较高精度方便后处理
    std::fprintf(lg->fp,
        "{\"t\":%.6f,"
        "\"B_t_mb\":%zu,"
        "\"resident_bytes\":%zu,"
        "\"n_blocks_resident\":%d,"
        "\"token_id\":%d,"
        "\"layer_load_latency_ms\":%.4f,"
        "\"decode_latency_ms\":%.4f,"
        "\"flash_bytes_read\":%zu}\n",
        r.t_sec, r.B_t_mb, r.resident_bytes, r.n_blocks_resident,
        r.token_id, r.layer_load_latency_ms, r.decode_latency_ms,
        r.flash_bytes_read);
    ++lg->buffered_records;
    if (lg->buffered_records >= lg->flush_threshold) {
        std::fflush(lg->fp);
        lg->buffered_records = 0;
    }
}

void metrics_logger_flush(metrics_logger *lg) {
    if (!lg || !lg->fp) return;
    std::fflush(lg->fp);
    lg->buffered_records = 0;
}

void metrics_logger_close(metrics_logger *lg) {
    if (!lg || !lg->fp) return;
    std::fflush(lg->fp);
    std::fclose(lg->fp);
    lg->fp               = nullptr;
    lg->buffered_records = 0;
}

}  // namespace elastic
