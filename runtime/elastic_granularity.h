#pragma once

#include <cstdint>
#include <vector>

namespace elastic {

enum class granularity_mode {
    MULTI,
    MULTI_FUSED,
    TENSOR,
    CUT,
};

struct granularity_config {
    granularity_mode mode = granularity_mode::TENSOR;
    int cut_parts = 2;
    int multi_tensors = 2;
    bool explicitly_enabled = false;
};

struct granularity_runtime_state {
    granularity_config config;
    uint64_t generation = 0;
    bool plan_override = false;
};

struct row_partition {
    int64_t row_start = 0;
    int64_t row_count = 0;
};

// Parse GGML_ELASTIC_GRANULARITY=multi|multi_fused|tensor|cut,
// GGML_ELASTIC_CUT_PARTS and GGML_ELASTIC_MULTI_TENSORS. Invalid values use
// the documented defaults and emit one diagnostic per process.
granularity_config granularity_from_env();

// Parse a plan/CLI spelling without consulting process environment. Returns
// false for unknown modes and leaves `mode` unchanged.
bool granularity_mode_from_string(const char * value, granularity_mode & mode);

// A plan may change granularity at a synchronized graph boundary. Backends
// snapshot this process-wide state at graph entry; model construction uses
// granularity_dynamic_prepare_cut() to provision the common row-tile
// representation before any plan can select CUT.
void granularity_runtime_apply(const granularity_config & config);
void granularity_runtime_clear();
granularity_runtime_state granularity_runtime_get();

// GGML_ELASTIC_GRANULARITY_DYNAMIC=1 provisions CUT-capable base tiles even
// when the initial mode is tensor or multi. This is required for a later plan
// to split a tensor without reallocating live backend buffers.
bool granularity_dynamic_prepare_cut();

const char * granularity_mode_name(granularity_mode mode);

// Divide rows into at most `parts` non-empty, contiguous partitions. Interior
// boundaries are aligned to `row_alignment`; the last partition consumes the
// exact tail. An empty result means that a meaningful split is impossible.
std::vector<row_partition> granularity_partition_rows(
        int64_t rows, int parts, int64_t row_alignment = 1);

} // namespace elastic
