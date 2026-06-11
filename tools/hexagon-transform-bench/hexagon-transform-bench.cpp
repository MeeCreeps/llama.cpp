#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct bench_type {
    ggml_type type;
    const char * name;
};

struct bench_shape {
    const char * name;
    int64_t k;
    int64_t m;
};

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

static ggml_type parse_type(const std::string & s) {
    static const std::map<std::string, ggml_type> map = {
        {"f16", GGML_TYPE_F16},
        {"f32", GGML_TYPE_F32},
        {"q4_0", GGML_TYPE_Q4_0},
        {"q4_1", GGML_TYPE_Q4_1},
        {"q8_0", GGML_TYPE_Q8_0},
        {"iq4_nl", GGML_TYPE_IQ4_NL},
        {"mxfp4", GGML_TYPE_MXFP4},
        {"q4_k", GGML_TYPE_Q4_K},
        {"q5_k", GGML_TYPE_Q5_K},
        {"q6_k", GGML_TYPE_Q6_K},
    };
    auto it = map.find(s);
    if (it == map.end()) {
        throw std::runtime_error("unknown type: " + s);
    }
    return it->second;
}

static const char * type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_IQ4_NL: return "IQ4_NL";
        case GGML_TYPE_MXFP4: return "MXFP4";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        default: return ggml_type_name(t);
    }
}

static void fill_source(std::vector<uint8_t> & data) {
    uint32_t x = 0x12345678u;
    for (uint8_t & b : data) {
        x = 1664525u*x + 1013904223u;
        b = (uint8_t)(x >> 24);
    }
}

static ggml_backend_dev_t find_hexagon_dev() {
    ggml_backend_load_all();

    ggml_backend_reg_t hex_reg = ggml_backend_reg_by_name("HEXAGON");
    if (hex_reg && ggml_backend_reg_dev_count(hex_reg) > 0) {
        return ggml_backend_reg_dev_get(hex_reg, 0);
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        std::string reg_name = reg ? ggml_backend_reg_name(reg) : "";
        std::string dev_name = ggml_backend_dev_name(dev);
        std::string dev_desc = ggml_backend_dev_description(dev);
        auto lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
            return s;
        };
        std::string all = lower(reg_name + " " + dev_name + " " + dev_desc);
        if (all.find("hexagon") != std::string::npos || all.find("htp") != std::string::npos) {
            return dev;
        }
    }
    return nullptr;
}

static ggml_backend_buffer_type_t get_repack_buft(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (fn) {
        ggml_backend_buffer_type_t * bufts = fn(dev);
        if (bufts && bufts[0]) {
            return bufts[0];
        }
    }
    return ggml_backend_dev_buffer_type(dev);
}

static bool type_is_block_aligned(ggml_type type, int64_t k) {
    const int64_t bs = ggml_blck_size(type);
    return bs <= 1 || (k % bs) == 0;
}

static void run_one(ggml_backend_buffer_type_t buft, ggml_type type, const bench_shape & shape, int warmup, int iters) {
    if (!type_is_block_aligned(type, shape.k)) {
        return;
    }

    const size_t meta_size = 16*1024;
    ggml_init_params params = { meta_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed");
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, shape.k, shape.m);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        ggml_free(ctx);
        throw std::runtime_error("failed to allocate Hexagon tensor buffer");
    }

    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> src(nbytes);
    fill_source(src);

    for (int i = 0; i < warmup; ++i) {
        ggml_backend_tensor_set(t, src.data(), 0, nbytes);
    }

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_ms();
        ggml_backend_tensor_set(t, src.data(), 0, nbytes);
        const double t1 = now_ms();
        times.push_back(t1 - t0);
    }

    const double med = median(times);
    const double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    const double gib = (double)nbytes / (1024.0*1024.0*1024.0);
    const double elems = (double)shape.k * (double)shape.m;
    const double gibs = gib / (med / 1000.0);
    const double gelems = elems / 1e9 / (med / 1000.0);

    printf("%s,%s,%" PRId64 ",%" PRId64 ",%.3f,%.4f,%.4f,%.2f,%.2f\n",
           type_name(type), shape.name, shape.k, shape.m,
           (double)nbytes / (1024.0*1024.0), med, avg, gibs, gelems);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    std::string type_arg = "all";
    int warmup = 3;
    int iters = 10;
    int64_t k = 0;
    int64_t m = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * opt) -> const char * {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + opt);
            return argv[++i];
        };
        if (a == "--type") type_arg = need("--type");
        else if (a == "--warmup") warmup = std::atoi(need("--warmup"));
        else if (a == "--iters") iters = std::atoi(need("--iters"));
        else if (a == "--k") k = std::atoll(need("--k"));
        else if (a == "--m") m = std::atoll(need("--m"));
        else if (a == "--help") {
            printf("usage: %s [--type all|q4_0|q4_1|q8_0|iq4_nl|mxfp4|q4_k|q5_k|q6_k|f16|f32] [--k K --m M] [--iters N] [--warmup N]\n", argv[0]);
            return 0;
        } else {
            throw std::runtime_error("unknown arg: " + a);
        }
    }

    ggml_backend_dev_t dev = find_hexagon_dev();
    if (!dev) {
        fprintf(stderr, "no HEXAGON backend device found\n");
        return 2;
    }

    ggml_backend_buffer_type_t buft = get_repack_buft(dev);
    printf("device=%s backend=%s buft=%s\n",
           ggml_backend_dev_description(dev),
           ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)),
           ggml_backend_buft_name(buft));
    printf("timing includes ggml_backend_tensor_set only; tensor/source/backend buffers are preallocated\n");
    printf("type,shape,K,M,MiB,med_ms,avg_ms,GiB/s,Gelem/s\n");

    std::vector<bench_type> types = {
        {GGML_TYPE_Q4_0, "q4_0"},
        {GGML_TYPE_Q4_1, "q4_1"},
        {GGML_TYPE_Q8_0, "q8_0"},
        {GGML_TYPE_IQ4_NL, "iq4_nl"},
        {GGML_TYPE_MXFP4, "mxfp4"},
        {GGML_TYPE_Q4_K, "q4_k"},
        {GGML_TYPE_Q5_K, "q5_k"},
        {GGML_TYPE_Q6_K, "q6_k"},
        {GGML_TYPE_F16, "f16"},
        {GGML_TYPE_F32, "f32"},
    };
    if (type_arg != "all") {
        types = {{parse_type(type_arg), type_arg.c_str()}};
    }

    std::vector<bench_shape> shapes;
    if (k > 0 && m > 0) {
        shapes.push_back({"custom", k, m});
    } else {
        shapes = {
            {"1B_hidden", 2048, 2048}, {"1B_ffn_up", 2048, 5632}, {"1B_ffn_down", 5632, 2048}, {"1B_lm_head_32k", 2048, 32000},
            {"3B_hidden", 3072, 3072}, {"3B_ffn_up", 3072, 8192}, {"3B_ffn_down", 8192, 3072}, {"3B_lm_head_32k", 3072, 32000},
            {"4B_hidden", 2560, 2560}, {"4B_ffn_up", 2560, 6912}, {"4B_ffn_down", 6912, 2560}, {"4B_lm_head_32k", 2560, 32000},
            {"7B_hidden", 4096, 4096}, {"7B_ffn_up", 4096, 11008}, {"7B_ffn_down", 11008, 4096}, {"7B_lm_head_32k", 4096, 32000},
            {"8B_hidden", 4096, 4096}, {"8B_ffn_up", 4096, 14336}, {"8B_ffn_down", 14336, 4096}, {"8B_lm_head_32k", 4096, 32000},
        };
    }

    for (const bench_type & bt : types) {
        for (const bench_shape & s : shapes) {
            run_one(buft, bt.type, s, warmup, iters);
        }
    }

    return 0;
}
