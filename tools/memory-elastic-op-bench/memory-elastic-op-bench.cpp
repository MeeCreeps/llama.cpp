#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(clock_type::now().time_since_epoch()).count();
}

struct stats {
    double min;
    double med;
    double avg;
    double max;
};

static stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {
        v.front(),
        v[v.size()/2],
        std::accumulate(v.begin(), v.end(), 0.0) / double(v.size()),
        v.back(),
    };
}

static ggml_backend_t init_opencl() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("OPENCL");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        throw std::runtime_error("OPENCL backend is unavailable");
    }
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    if (!backend) {
        throw std::runtime_error("failed to initialize OPENCL backend");
    }
    return backend;
}

static void fill_weight(std::vector<uint8_t> & data, int64_t k, int64_t m) {
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, k);
    std::vector<float> row(size_t(k), 0.0f);
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t c = 0; c < k; ++c) {
            row[size_t(c)] = float(((c*17 + r*13) % 127) - 63) / 64.0f;
        }
        const size_t written = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, row.data(), data.data() + size_t(r)*row_bytes, 0, 1, k, nullptr);
        if (written != row_bytes) {
            throw std::runtime_error("unexpected Q4_0 row size from ggml_quantize_chunk");
        }
    }
}

int main(int argc, char ** argv) {
    int64_t k = 4096;
    int64_t m = 4096;
    int warmup = 20;
    int iters = 100;
    int materialize_iters = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char * name) -> const char * {
            if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[i];
        };
        if (arg == "--k") k = std::atoll(value("--k"));
        else if (arg == "--m") m = std::atoll(value("--m"));
        else if (arg == "--warmup") warmup = std::atoi(value("--warmup"));
        else if (arg == "--iters") iters = std::atoi(value("--iters"));
        else if (arg == "--materialize-iters") materialize_iters = std::atoi(value("--materialize-iters"));
        else if (arg == "--help") {
            std::printf("usage: %s [--k K] [--m M] [--warmup N] [--iters N] [--materialize-iters N]\n", argv[0]);
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (k <= 0 || m <= 0 || k % 32 != 0 || warmup < 0 || iters <= 0 || materialize_iters != 1) {
        throw std::runtime_error("invalid arguments; K must be a positive multiple of 32 and materialize-iters must be 1");
    }

    ggml_backend_t backend = init_opencl();

    const size_t mem_size = ggml_tensor_overhead()*8 + ggml_graph_overhead_custom(16, false);
    ggml_init_params params = { mem_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, m);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(weight, "bench.weight");
    ggml_set_name(input,  "bench.input");
    ggml_set_name(output, "bench.output");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("backend tensor allocation failed");

    std::vector<uint8_t> weight_data(ggml_nbytes(weight));
    std::vector<float> input_data(size_t(k), 0.0f);
    fill_weight(weight_data, k, m);
    for (int64_t i = 0; i < k; ++i) input_data[size_t(i)] = float((i % 31) - 15) / 31.0f;
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));

    std::vector<double> materialize_ms;
    materialize_ms.reserve(size_t(materialize_iters));
    for (int i = 0; i < materialize_iters; ++i) {
        const double t0 = now_ms();
        ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());
        ggml_backend_synchronize(backend);
        materialize_ms.push_back(now_ms() - t0);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    for (int i = 0; i < warmup; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("warmup graph compute failed");
        }
    }
    ggml_backend_synchronize(backend);

    std::vector<double> compute_ms;
    compute_ms.reserve(size_t(iters));
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_ms();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph compute failed");
        }
        ggml_backend_synchronize(backend);
        compute_ms.push_back(now_ms() - t0);
    }

    const double batch_t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("batched graph compute failed");
        }
    }
    ggml_backend_synchronize(backend);
    const double batch_avg_ms = (now_ms() - batch_t0) / double(iters);

    std::vector<float> out_data(size_t(m), 0.0f);
    ggml_backend_tensor_get(output, out_data.data(), 0, ggml_nbytes(output));
    double checksum = 0.0;
    for (float v : out_data) {
        if (std::isfinite(v)) checksum += v;
    }

    const stats sm = summarize(materialize_ms);
    const stats sc = summarize(compute_ms);
    const char * disable = std::getenv("GGML_OPENCL_DISABLE_ADRENO_KERNELS");
    const char * layout = disable && *disable && *disable != '0' ? "generic_soa" : "adreno_transposed_soa";
    std::printf(
        "RESULT layout=%s K=%lld M=%lld raw_mib=%.3f backend_mib=%.3f "
        "materialize_med_ms=%.6f materialize_avg_ms=%.6f "
        "compute_med_ms=%.6f compute_avg_ms=%.6f batch_avg_ms=%.6f "
        "compute_min_ms=%.6f compute_max_ms=%.6f checksum=%.9g\n",
        layout, (long long) k, (long long) m,
        double(weight_data.size()) / 1048576.0,
        double(ggml_backend_buffer_get_size(buffer)) / 1048576.0,
        sm.med, sm.avg, sc.med, sc.avg, batch_avg_ms, sc.min, sc.max, checksum);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
