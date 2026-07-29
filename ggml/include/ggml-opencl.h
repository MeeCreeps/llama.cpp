#ifndef GGML_OPENCL_H
#define GGML_OPENCL_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);

struct ggml_backend_opencl_working_set_state {
    int      active_capacity;
    int      pending_capacity;
    int      target_capacity;
    int      observed_required_capacity;
    uint64_t accesses;
    uint64_t hits;
    uint64_t misses;
    uint64_t capacity_changes;
    size_t   resident_bytes;
};

struct ggml_backend_opencl_granularity_state {
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

// Query/control the OpenCL runtime-managed expert-slice working set. A target
// of -1 restores backend-local automatic sizing. The target is a retention
// capacity; active slices required for correctness may temporarily exceed it.
GGML_BACKEND_API bool ggml_backend_opencl_get_working_set_state(
        ggml_backend_t backend,
        struct ggml_backend_opencl_working_set_state * state);
GGML_BACKEND_API bool ggml_backend_opencl_set_working_set_target(
        ggml_backend_t backend,
        int target_capacity);
GGML_BACKEND_API bool ggml_backend_opencl_get_granularity_state(
        ggml_backend_t backend,
        struct ggml_backend_opencl_granularity_state * state);
GGML_BACKEND_API bool ggml_backend_opencl_synchronize_elastic_pipeline(
        ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL_H
