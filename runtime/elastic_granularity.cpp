#include "elastic_granularity.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace elastic {

namespace {

std::atomic<int> g_runtime_mode{-1};
std::atomic<int> g_runtime_cut_parts{2};
std::atomic<int> g_runtime_multi_tensors{2};
std::atomic<uint64_t> g_runtime_generation{0};

int parse_bounded_positive(const char * name, int fallback, int maximum) {
    const char * value = std::getenv(name);
    if (!value || !*value) return fallback;
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 1 || parsed > maximum) {
        std::fprintf(stderr, "elastic granularity: invalid %s='%s'; using %d\n",
                     name, value, fallback);
        return fallback;
    }
    return static_cast<int>(parsed);
}

} // namespace

bool granularity_mode_from_string(const char * value, granularity_mode & mode) {
    if (!value) return false;
    if (std::strcmp(value, "multi") == 0) {
        mode = granularity_mode::MULTI;
    } else if (std::strcmp(value, "multi_fused") == 0) {
        mode = granularity_mode::MULTI_FUSED;
    } else if (std::strcmp(value, "tensor") == 0) {
        mode = granularity_mode::TENSOR;
    } else if (std::strcmp(value, "cut") == 0) {
        mode = granularity_mode::CUT;
    } else {
        return false;
    }
    return true;
}

granularity_config granularity_from_env() {
    granularity_config config;
    const char * mode = std::getenv("GGML_ELASTIC_GRANULARITY");
    config.explicitly_enabled = mode && *mode;
    if (config.explicitly_enabled) {
        if (!granularity_mode_from_string(mode, config.mode)) {
            std::fprintf(stderr,
                         "elastic granularity: invalid GGML_ELASTIC_GRANULARITY='%s'; using tensor\n",
                         mode);
            config.mode = granularity_mode::TENSOR;
        }
    }
    config.cut_parts = parse_bounded_positive("GGML_ELASTIC_CUT_PARTS", 2, 64);
    if (config.cut_parts < 2) {
        std::fprintf(stderr,
                     "elastic granularity: GGML_ELASTIC_CUT_PARTS must be >= 2; using 2\n");
        config.cut_parts = 2;
    }
    config.multi_tensors = parse_bounded_positive("GGML_ELASTIC_MULTI_TENSORS", 2, 64);
    if (config.multi_tensors < 2) {
        std::fprintf(stderr,
                     "elastic granularity: GGML_ELASTIC_MULTI_TENSORS must be >= 2; using 2\n");
        config.multi_tensors = 2;
    }
    return config;
}

void granularity_runtime_apply(const granularity_config & config) {
    g_runtime_cut_parts.store(
        std::max(2, config.cut_parts), std::memory_order_relaxed);
    g_runtime_multi_tensors.store(
        std::max(2, config.multi_tensors), std::memory_order_relaxed);
    g_runtime_mode.store(
        static_cast<int>(config.mode), std::memory_order_release);
    g_runtime_generation.fetch_add(1, std::memory_order_acq_rel);
#if defined(_WIN32)
    _putenv_s("GGML_ELASTIC_GRANULARITY",
              granularity_mode_name(config.mode));
    _putenv_s("GGML_ELASTIC_CUT_PARTS",
              std::to_string(std::max(2, config.cut_parts)).c_str());
    _putenv_s("GGML_ELASTIC_MULTI_TENSORS",
              std::to_string(std::max(2, config.multi_tensors)).c_str());
#else
    setenv("GGML_ELASTIC_GRANULARITY",
           granularity_mode_name(config.mode), 1);
    const std::string cut_parts =
        std::to_string(std::max(2, config.cut_parts));
    const std::string multi_tensors =
        std::to_string(std::max(2, config.multi_tensors));
    setenv("GGML_ELASTIC_CUT_PARTS", cut_parts.c_str(), 1);
    setenv("GGML_ELASTIC_MULTI_TENSORS", multi_tensors.c_str(), 1);
#endif
}

void granularity_runtime_clear() {
    g_runtime_mode.store(-1, std::memory_order_release);
    g_runtime_generation.fetch_add(1, std::memory_order_acq_rel);
}

granularity_runtime_state granularity_runtime_get() {
    granularity_runtime_state state;
    const int mode = g_runtime_mode.load(std::memory_order_acquire);
    state.generation =
        g_runtime_generation.load(std::memory_order_acquire);
    if (mode >= static_cast<int>(granularity_mode::MULTI) &&
        mode <= static_cast<int>(granularity_mode::CUT)) {
        state.plan_override = true;
        state.config.mode = static_cast<granularity_mode>(mode);
        state.config.cut_parts =
            g_runtime_cut_parts.load(std::memory_order_relaxed);
        state.config.multi_tensors =
            g_runtime_multi_tensors.load(std::memory_order_relaxed);
        state.config.explicitly_enabled = true;
    } else {
        state.config = granularity_from_env();
    }
    return state;
}

bool granularity_dynamic_prepare_cut() {
    const char * value = std::getenv("GGML_ELASTIC_GRANULARITY_DYNAMIC");
    return value && *value && *value != '0';
}

const char * granularity_mode_name(granularity_mode mode) {
    switch (mode) {
        case granularity_mode::MULTI:       return "multi";
        case granularity_mode::MULTI_FUSED: return "multi_fused";
        case granularity_mode::TENSOR:      return "tensor";
        case granularity_mode::CUT:         return "cut";
    }
    return "tensor";
}

std::vector<row_partition> granularity_partition_rows(
        int64_t rows, int parts, int64_t row_alignment) {
    std::vector<row_partition> result;
    if (rows <= 0 || parts < 2 || row_alignment <= 0) return result;
    const int64_t aligned_rows = rows / row_alignment;
    if (aligned_rows < parts) return result;

    int64_t start = 0;
    for (int part = 0; part < parts; ++part) {
        const int remaining_parts = parts - part;
        const int64_t remaining_rows = rows - start;
        int64_t count = remaining_rows;
        if (remaining_parts > 1) {
            const int64_t target = (remaining_rows + remaining_parts - 1) / remaining_parts;
            count = std::max<int64_t>(row_alignment,
                    ((target + row_alignment - 1) / row_alignment) * row_alignment);
            const int64_t minimum_tail = static_cast<int64_t>(remaining_parts - 1) * row_alignment;
            count = std::min(count, remaining_rows - minimum_tail);
        }
        if (count <= 0) return {};
        result.push_back({start, count});
        start += count;
    }
    if (start != rows) return {};
    return result;
}

} // namespace elastic
