// ggml-cpu-elastic.h
//
// 隔离的 CPU elastic 后端：复用 CPU compute 路径，套上 WBM 弹性内存管理。
// set_tensor 时把 mmap 源指针记到 WBM，graph_compute 前 ensure_resident，
// 周期 evict 用 madvise(MADV_DONTNEED) 回收 anonymous 页，再用 memcpy 从
// mmap 源 reload。
//
// 触发：调 ggml_backend_cpu_elastic_init() 显式启用。也通过 ggml backend
// registry 注册为 "CPU_Elastic" 设备。

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t            ggml_backend_cpu_elastic_init(void);
GGML_BACKEND_API bool                      ggml_backend_is_cpu_elastic(ggml_backend_t backend);

struct ggml_backend_elastic_granularity_state {
    int      mode;
    uint64_t units;
    uint64_t nonresident_units;
    uint64_t pipeline_issued;
    uint64_t pipeline_ready;
    uint64_t pipeline_waits;
    uint64_t pipeline_wait_us;
    uint64_t reloads;
    uint64_t reload_bytes;
    uint64_t prepare_us;
    uint64_t direct_read_calls;
    uint64_t direct_read_us;
    uint64_t direct_read_bytes;
    uint64_t load_calls;
    uint64_t load_us;
    uint64_t load_bytes;
    uint64_t prepare_calls;
    uint64_t prepare_bytes;
    uint64_t compute_calls;
    uint64_t compute_us;
    int      compute_timing_available;
    uint64_t pipeline_residency_us;
    uint64_t pipeline_unissued_us;
    uint64_t pipeline_stage_us;
    uint64_t pipeline_retire_us;
    uint64_t evict_us;
    size_t   resident_bytes;
};

GGML_BACKEND_API bool ggml_backend_cpu_elastic_get_granularity_state(
        ggml_backend_t backend,
        struct ggml_backend_elastic_granularity_state * state);
GGML_BACKEND_API bool ggml_backend_cpu_elastic_synchronize_pipeline(
        ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cpu_elastic_buffer_type(void);
GGML_BACKEND_API ggml_backend_reg_t        ggml_backend_cpu_elastic_reg(void);

#ifdef  __cplusplus
}
#endif
