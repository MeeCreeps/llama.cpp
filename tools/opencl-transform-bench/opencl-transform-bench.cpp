#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef LLAMA_OPENCL_KERNEL_DIR
#define LLAMA_OPENCL_KERNEL_DIR "."
#endif

#define CL_CHECK(expr) do { \
    cl_int err_ = (expr); \
    if (err_ != CL_SUCCESS) { \
        throw std::runtime_error(std::string("OpenCL error ") + std::to_string(err_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)

static uint64_t ceil_div(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static uint64_t align_up(uint64_t a, uint64_t b) {
    return ceil_div(a, b) * b;
}

static std::string read_text(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static double event_ms(cl_event evt) {
    cl_ulong t0 = 0;
    cl_ulong t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr));
    return double(t1 - t0) * 1e-6;
}

static const char * type_name(const std::string & type) {
    return type.c_str();
}

struct stats {
    double min = 0.0;
    double med = 0.0;
    double avg = 0.0;
    double max = 0.0;
};

static stats calc_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    stats s;
    s.min = v.front();
    s.med = v[v.size() / 2];
    s.max = v.back();
    s.avg = std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
    return s;
}

struct params {
    std::string type = "all";
    std::string mode = "all";
    std::string kernel_dir = LLAMA_OPENCL_KERNEL_DIR;
    int platform = 0;
    int device = 0;
    int iters = 100;
    int warmup = 10;
    int k = 4096; // tensor ne0
    int m = 4096; // tensor ne1
    double flops_per_elem = 1.0;
};

static void usage(const char * argv0) {
    std::printf(
        "usage: %s [options]\n"
        "\n"
        "OpenCL weight transform microbenchmark. Buffer creation, host upload,\n"
        "subbuffer/image creation, and kernel compilation are outside timed regions.\n"
        "\n"
        "options:\n"
        "  --type <all|q4_0|q4_1|q5_0|q5_1|q8_0|iq4_nl|q4_k|q5_k|q6_k|mxfp4>\n"
        "  --mode <all|standard|dense|dense_adaptive|dense_tile32|dense_nocopy|dense_tile32_nocopy|dense_scalar|dense_tiled|dense_fused|dense_fused_wall|dense_fused_check|moe|transpose|transpose_tiled|transpose_check|transpose_check32|reload_chain_ab>\n"
        "       standard: AOS->SOA convert kernel only\n"
        "       dense:    Adreno-style convert/noshuffle + tiled transpose kernel + transpose copy\n"
        "       dense_adaptive: dense mode with the same 16x16/32x32 heuristic as the backend\n"
        "       dense_tile32: dense mode with 32x32 tiled transpose + copy-back\n"
        "       dense_nocopy: dense mode without transpose copy-back; measures upper bound for final-buffer ownership\n"
        "       dense_tile32_nocopy: compare 32x32 transpose tile size without copy-back\n"
        "       dense_scalar: dense mode with the original scalar-strided transpose kernels\n"
        "       dense_tiled: alias for dense\n"
        "       dense_fused: experimental direct convert into dense-transposed layout\n"
        "       dense_fused_wall: pre-set dense_fused kernel, then time CPU wall average\n"
        "       dense_fused_check: compare dense_fused output against dense_tiled output\n"
        "       moe:      *_trans4_ns kernel only; clCreateImage is not timed\n"
        "       transpose: pre-set dense transpose kernels, then compare CPU wall and event profiling\n"
        "       transpose_tiled: same as transpose, but uses 16x16 local-memory tiled kernels\n"
        "       transpose_check: compare scalar and tiled transpose output byte-for-byte\n"
        "       transpose_check32: compare scalar and 32x32 tiled transpose output byte-for-byte\n"
        "  --k <int>              tensor ne0, default 4096\n"
        "  --m <int>              tensor ne1, default 4096\n"
        "  --iters <int>          timed iterations, default 100\n"
        "  --warmup <int>         warmup iterations, default 10\n"
        "  --platform <int>       OpenCL platform index, default 0\n"
        "  --device <int>         OpenCL device index within platform, default 0\n"
        "  --kernel-dir <path>    directory containing cvt.cl and transpose.cl\n"
        "  --flops-per-elem <f>   effective FLOPs per K*M element, default 1.0\n",
        argv0);
}

static params parse(int argc, char ** argv) {
    params p;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "--type") {
            p.type = need("--type");
        } else if (a == "--mode") {
            p.mode = need("--mode");
        } else if (a == "--k") {
            p.k = std::atoi(need("--k"));
        } else if (a == "--m") {
            p.m = std::atoi(need("--m"));
        } else if (a == "--iters") {
            p.iters = std::atoi(need("--iters"));
        } else if (a == "--warmup") {
            p.warmup = std::atoi(need("--warmup"));
        } else if (a == "--platform") {
            p.platform = std::atoi(need("--platform"));
        } else if (a == "--device") {
            p.device = std::atoi(need("--device"));
        } else if (a == "--kernel-dir") {
            p.kernel_dir = need("--kernel-dir");
        } else if (a == "--flops-per-elem") {
            p.flops_per_elem = std::atof(need("--flops-per-elem"));
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (p.k <= 0 || p.m <= 0 || p.iters <= 0 || p.warmup < 0 || p.flops_per_elem < 0.0) {
        throw std::runtime_error("invalid non-positive shape or iteration parameter");
    }
    return p;
}

struct cl_ctx {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_command_queue xfer_queue = nullptr;
    cl_program cvt = nullptr;
    cl_program transpose = nullptr;

    ~cl_ctx() {
        if (transpose) clReleaseProgram(transpose);
        if (cvt) clReleaseProgram(cvt);
        if (xfer_queue) clReleaseCommandQueue(xfer_queue);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

static cl_program build_program(cl_context ctx, cl_device_id dev, const std::string & src, const char * name) {
    const char * csrc = src.c_str();
    size_t len = src.size();
    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &len, &err);
    CL_CHECK(err);

    err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
        clReleaseProgram(prog);
        throw std::runtime_error(std::string("failed to build ") + name + ":\n" + log);
    }
    return prog;
}

static cl_ctx init_opencl(const params & p) {
    cl_uint n_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &n_platforms));
    if (n_platforms == 0 || p.platform >= int(n_platforms)) {
        throw std::runtime_error("OpenCL platform index out of range");
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    CL_CHECK(clGetPlatformIDs(n_platforms, platforms.data(), nullptr));

    cl_uint n_devices = 0;
    CL_CHECK(clGetDeviceIDs(platforms[p.platform], CL_DEVICE_TYPE_ALL, 0, nullptr, &n_devices));
    if (n_devices == 0 || p.device >= int(n_devices)) {
        throw std::runtime_error("OpenCL device index out of range");
    }
    std::vector<cl_device_id> devices(n_devices);
    CL_CHECK(clGetDeviceIDs(platforms[p.platform], CL_DEVICE_TYPE_ALL, n_devices, devices.data(), nullptr));

    cl_ctx c;
    c.platform = platforms[p.platform];
    c.device = devices[p.device];

    cl_int err = CL_SUCCESS;
    c.context = clCreateContext(nullptr, 1, &c.device, nullptr, nullptr, &err);
    CL_CHECK(err);

#ifdef CL_VERSION_2_0
    const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    c.queue = clCreateCommandQueueWithProperties(c.context, c.device, props, &err);
#else
    c.queue = clCreateCommandQueue(c.context, c.device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    CL_CHECK(err);
#ifdef CL_VERSION_2_0
    c.xfer_queue = clCreateCommandQueueWithProperties(c.context, c.device, props, &err);
#else
    c.xfer_queue = clCreateCommandQueue(c.context, c.device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    CL_CHECK(err);

    c.cvt = build_program(c.context, c.device, read_text(p.kernel_dir + "/cvt.cl"), "cvt.cl");
    c.transpose = build_program(c.context, c.device, read_text(p.kernel_dir + "/transpose.cl"), "transpose.cl");

    char dev_name[256] = {};
    char plat_name[256] = {};
    clGetPlatformInfo(c.platform, CL_PLATFORM_NAME, sizeof(plat_name), plat_name, nullptr);
    clGetDeviceInfo(c.device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    std::printf("platform=%s device=%s\n", plat_name, dev_name);

    return c;
}

struct buffer {
    cl_mem mem = nullptr;
    size_t size = 0;
    cl_context ctx = nullptr;

    buffer() = default;
    buffer(cl_context c, size_t n, cl_mem_flags flags = CL_MEM_READ_WRITE) : size(n), ctx(c) {
        cl_int err = CL_SUCCESS;
        mem = clCreateBuffer(ctx, flags, std::max<size_t>(size, 1), nullptr, &err);
        CL_CHECK(err);
    }
    buffer(const buffer &) = delete;
    buffer & operator=(const buffer &) = delete;
    buffer(buffer && other) noexcept {
        mem = other.mem;
        size = other.size;
        ctx = other.ctx;
        other.mem = nullptr;
    }
    buffer & operator=(buffer && other) noexcept {
        if (this != &other) {
            if (mem) clReleaseMemObject(mem);
            mem = other.mem;
            size = other.size;
            ctx = other.ctx;
            other.mem = nullptr;
        }
        return *this;
    }
    ~buffer() {
        if (mem) clReleaseMemObject(mem);
    }
};

struct kernel {
    cl_kernel k = nullptr;
    kernel() = default;
    kernel(cl_program prog, const char * name) {
        cl_int err = CL_SUCCESS;
        k = clCreateKernel(prog, name, &err);
        CL_CHECK(err);
    }
    kernel(const kernel &) = delete;
    kernel & operator=(const kernel &) = delete;
    kernel(kernel && other) noexcept {
        k = other.k;
        other.k = nullptr;
    }
    ~kernel() {
        if (k) clReleaseKernel(k);
    }
};

static void set_arg(cl_kernel k, cl_uint i, cl_mem mem) {
    CL_CHECK(clSetKernelArg(k, i, sizeof(mem), &mem));
}

template <typename T>
static void set_scalar(cl_kernel k, cl_uint i, const T & v) {
    CL_CHECK(clSetKernelArg(k, i, sizeof(T), &v));
}

struct run_times {
    double convert_ms = 0.0;
    double transpose_kernel_ms = 0.0;
    double transpose_copy_ms = 0.0;
    double total_ms() const {
        return convert_ms + transpose_kernel_ms + transpose_copy_ms;
    }
};

static double enqueue_kernel(cl_command_queue q, cl_kernel k, cl_uint dims, const size_t * gws, const size_t * lws) {
    cl_event evt = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(q, k, dims, nullptr, gws, lws, 0, nullptr, &evt));
    CL_CHECK(clWaitForEvents(1, &evt));
    double ms = event_ms(evt);
    CL_CHECK(clReleaseEvent(evt));
    return ms;
}

static double enqueue_copy(cl_command_queue q, cl_mem src, cl_mem dst, size_t n) {
    cl_event evt = nullptr;
    CL_CHECK(clEnqueueCopyBuffer(q, src, dst, 0, 0, n, 0, nullptr, &evt));
    CL_CHECK(clWaitForEvents(1, &evt));
    double ms = event_ms(evt);
    CL_CHECK(clReleaseEvent(evt));
    return ms;
}

static run_times transpose_as(cl_ctx & c, const char * kernel_name, buffer & src_dst, buffer & tmp, size_t bytes, int stride, int rows) {
    kernel kt(c.transpose, kernel_name);
    set_arg(kt.k, 0, src_dst.mem);
    set_arg(kt.k, 1, tmp.mem);
    set_scalar(kt.k, 2, stride);
    set_scalar(kt.k, 3, rows);
    const size_t gws[3] = { size_t(stride), size_t(rows), 1 };
    const size_t lws[3] = { 64, 1, 1 };
    run_times t;
    t.transpose_kernel_ms = enqueue_kernel(c.queue, kt.k, 3, gws, lws);
    t.transpose_copy_ms = enqueue_copy(c.queue, tmp.mem, src_dst.mem, bytes);
    return t;
}

static int kernel_tile_size(const char * kernel_name) {
    const std::string name(kernel_name);
    if (name.find("tiled32") != std::string::npos) {
        return 32;
    }
    return 16;
}

static run_times transpose_as_tiled(cl_ctx & c, const char * kernel_name, buffer & src_dst, buffer & tmp, size_t bytes, int stride, int rows) {
    kernel kt(c.transpose, kernel_name);
    set_arg(kt.k, 0, src_dst.mem);
    set_arg(kt.k, 1, tmp.mem);
    set_scalar(kt.k, 2, stride);
    set_scalar(kt.k, 3, rows);
    const int tile = kernel_tile_size(kernel_name);
    const size_t gws[3] = { size_t(align_up(stride, tile)), size_t(align_up(rows, tile)), 1 };
    const size_t lws[3] = { size_t(tile), size_t(tile), 1 };
    run_times t;
    t.transpose_kernel_ms = enqueue_kernel(c.queue, kt.k, 3, gws, lws);
    t.transpose_copy_ms = enqueue_copy(c.queue, tmp.mem, src_dst.mem, bytes);
    return t;
}

static run_times transpose_as_tiled_nocopy(cl_ctx & c, const char * kernel_name, buffer & src, buffer & tmp, int stride, int rows) {
    kernel kt(c.transpose, kernel_name);
    set_arg(kt.k, 0, src.mem);
    set_arg(kt.k, 1, tmp.mem);
    set_scalar(kt.k, 2, stride);
    set_scalar(kt.k, 3, rows);
    const int tile = kernel_tile_size(kernel_name);
    const size_t gws[3] = { size_t(align_up(stride, tile)), size_t(align_up(rows, tile)), 1 };
    const size_t lws[3] = { size_t(tile), size_t(tile), 1 };
    run_times t;
    t.transpose_kernel_ms = enqueue_kernel(c.queue, kt.k, 3, gws, lws);
    return t;
}

struct prepared_transpose {
    kernel kt;
    cl_mem src_dst = nullptr;
    cl_mem tmp = nullptr;
    size_t bytes = 0;
    size_t gws[3] = {};
    size_t lws[3] = { 64, 1, 1 };

    prepared_transpose() = default;
    prepared_transpose(
            cl_program prog, const char * kernel_name,
            buffer & src_dst_buf, buffer & tmp_buf,
            size_t nbytes, int stride, int rows,
            bool tiled = false) :
        kt(prog, kernel_name),
        src_dst(src_dst_buf.mem),
        tmp(tmp_buf.mem),
        bytes(nbytes) {
        set_arg(kt.k, 0, src_dst);
        set_arg(kt.k, 1, tmp);
        set_scalar(kt.k, 2, stride);
        set_scalar(kt.k, 3, rows);
        if (tiled) {
            gws[0] = size_t(align_up(stride, 16));
            gws[1] = size_t(align_up(rows, 16));
            lws[0] = 16;
            lws[1] = 16;
        } else {
            gws[0] = size_t(stride);
            gws[1] = size_t(rows);
            lws[0] = 64;
            lws[1] = 1;
        }
        gws[2] = 1;
    }

    prepared_transpose(const prepared_transpose &) = delete;
    prepared_transpose & operator=(const prepared_transpose &) = delete;
    prepared_transpose(prepared_transpose && other) noexcept :
        kt(std::move(other.kt)),
        src_dst(other.src_dst),
        tmp(other.tmp),
        bytes(other.bytes) {
        std::copy(std::begin(other.gws), std::end(other.gws), std::begin(gws));
        std::copy(std::begin(other.lws), std::end(other.lws), std::begin(lws));
        other.src_dst = nullptr;
        other.tmp = nullptr;
        other.bytes = 0;
    }
};

static run_times enqueue_prepared_transpose(cl_command_queue q, prepared_transpose & tcmd) {
    run_times t;
    t.transpose_kernel_ms = enqueue_kernel(q, tcmd.kt.k, 3, tcmd.gws, tcmd.lws);
    t.transpose_copy_ms = enqueue_copy(q, tcmd.tmp, tcmd.src_dst, tcmd.bytes);
    return t;
}

static void enqueue_prepared_transpose_no_profile(cl_command_queue q, prepared_transpose & tcmd) {
    CL_CHECK(clEnqueueNDRangeKernel(q, tcmd.kt.k, 3, nullptr, tcmd.gws, tcmd.lws, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueCopyBuffer(q, tcmd.tmp, tcmd.src_dst, 0, 0, tcmd.bytes, 0, nullptr, nullptr));
}

struct bench_case {
    std::string type;
    int block = 0;
    size_t src_block_bytes = 0;
    size_t d_bytes = 0;
    size_t m_bytes = 0;
    size_t dm_bytes = 0;
    size_t q_bytes = 0;
    size_t qs_bytes = 0;
    size_t qh_bytes = 0;
    size_t ql_bytes = 0;
    size_t s_bytes = 0;
    size_t e_bytes = 0;
};

static bench_case make_case(const std::string & type, int k, int m, const std::string & mode) {
    bench_case b;
    b.type = type;
    const uint64_t elems = uint64_t(k) * uint64_t(m);
    auto blocks = [&](int blck) {
        if (k % blck != 0) {
            throw std::runtime_error(type + " requires --k multiple of " + std::to_string(blck));
        }
        return elems / blck;
    };

    if (type == "q4_0" || type == "q4_1" || type == "q5_0" || type == "q5_1" || type == "q8_0" || type == "iq4_nl" || type == "mxfp4") {
        b.block = 32;
    } else if (type == "q4_k" || type == "q5_k" || type == "q6_k") {
        b.block = 256;
    } else {
        throw std::runtime_error("unsupported type: " + type);
    }

    const uint64_t nblk = blocks(b.block);

    if (type == "q4_0" || type == "iq4_nl") {
        b.src_block_bytes = 18;
        b.d_bytes = nblk * 2;
        b.q_bytes = nblk * 16;
    } else if (type == "q4_1") {
        b.src_block_bytes = 20;
        b.d_bytes = nblk * 2;
        b.m_bytes = nblk * 2;
        b.q_bytes = nblk * 16;
    } else if (type == "q5_0") {
        b.src_block_bytes = 22;
        b.d_bytes = nblk * 2;
        b.qh_bytes = nblk * 4;
        b.qs_bytes = nblk * 16;
    } else if (type == "q5_1") {
        b.src_block_bytes = 24;
        b.d_bytes = nblk * 2;
        b.m_bytes = nblk * 2;
        b.qh_bytes = nblk * 4;
        b.qs_bytes = nblk * 16;
    } else if (type == "q8_0") {
        b.src_block_bytes = 34;
        b.d_bytes = nblk * 2;
        b.q_bytes = nblk * 32;
    } else if (type == "mxfp4") {
        b.src_block_bytes = 17;
        b.e_bytes = nblk;
        b.q_bytes = nblk * 16;
    } else if (type == "q4_k") {
        b.src_block_bytes = 2 + 2 + 12 + 128;
        b.d_bytes = nblk * 2;
        b.dm_bytes = nblk * 2;
        b.s_bytes = nblk * 12;
        b.q_bytes = nblk * 128;
    } else if (type == "q5_k") {
        b.src_block_bytes = 2 + 2 + 12 + 32 + 128;
        b.d_bytes = nblk * 2;
        b.dm_bytes = nblk * 2;
        b.s_bytes = nblk * 12;
        b.q_bytes = nblk * 128;
        b.qh_bytes = nblk * 32;
    } else if (type == "q6_k") {
        b.src_block_bytes = 128 + 64 + 16 + 2;
        b.ql_bytes = nblk * 128;
        b.qh_bytes = nblk * 64;
        b.s_bytes = nblk * 16;
        b.d_bytes = nblk * 2;
        if (mode == "moe") {
            b.ql_bytes = elems / 8 * sizeof(uint32_t);
            b.qh_bytes = elems / 16 * sizeof(uint32_t);
        }
    }
    return b;
}

static std::vector<uint8_t> make_input(size_t bytes) {
    std::vector<uint8_t> data(bytes);
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> dist(0, 255);
    for (uint8_t & v : data) {
        v = uint8_t(dist(rng));
    }
    return data;
}

static void add(run_times & a, const run_times & b) {
    a.convert_ms += b.convert_ms;
    a.transpose_kernel_ms += b.transpose_kernel_ms;
    a.transpose_copy_ms += b.transpose_copy_ms;
}

struct bench_alloc {
    buffer src;
    buffer d;
    buffer m;
    buffer dm;
    buffer q;
    buffer qs;
    buffer qh;
    buffer ql;
    buffer s;
    buffer e;
    buffer tmp_q;
    buffer tmp_qs;
    buffer tmp_qh;
    buffer tmp_ql;
    buffer tmp_d;
    buffer tmp_m;
    buffer tmp_dm;
    buffer tmp_s;
};

static bench_alloc allocate_case(cl_ctx & c, const bench_case & bc, size_t src_bytes) {
    bench_alloc a;
    a.src = buffer(c.context, src_bytes);
    if (bc.d_bytes)  a.d  = buffer(c.context, bc.d_bytes);
    if (bc.m_bytes)  a.m  = buffer(c.context, bc.m_bytes);
    if (bc.dm_bytes) a.dm = buffer(c.context, bc.dm_bytes);
    if (bc.q_bytes)  a.q  = buffer(c.context, bc.q_bytes);
    if (bc.qs_bytes) a.qs = buffer(c.context, bc.qs_bytes);
    if (bc.qh_bytes) a.qh = buffer(c.context, bc.qh_bytes);
    if (bc.ql_bytes) a.ql = buffer(c.context, bc.ql_bytes);
    if (bc.s_bytes)  a.s  = buffer(c.context, bc.s_bytes);
    if (bc.e_bytes)  a.e  = buffer(c.context, bc.e_bytes);

    if (bc.q_bytes)  a.tmp_q  = buffer(c.context, bc.q_bytes);
    if (bc.qs_bytes) a.tmp_qs = buffer(c.context, bc.qs_bytes);
    if (bc.qh_bytes) a.tmp_qh = buffer(c.context, bc.qh_bytes);
    if (bc.ql_bytes) a.tmp_ql = buffer(c.context, bc.ql_bytes);
    if (bc.d_bytes)  a.tmp_d  = buffer(c.context, bc.d_bytes);
    if (bc.m_bytes)  a.tmp_m  = buffer(c.context, bc.m_bytes);
    if (bc.dm_bytes) a.tmp_dm = buffer(c.context, bc.dm_bytes);
    if (bc.s_bytes)  a.tmp_s  = buffer(c.context, bc.s_bytes);
    return a;
}

static run_times run_standard(cl_ctx & c, const std::string & type, int k_dim, int m_dim, bench_alloc & a) {
    const uint64_t elems = uint64_t(k_dim) * uint64_t(m_dim);
    run_times t;
    size_t gws[3] = {};
    size_t lws[3] = { 64, 1, 1 };

    if (type == "q4_0") {
        kernel kc(c.cvt, "kernel_convert_block_q4_0");
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.d.mem);
        gws[0] = elems / 32;
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else if (type == "q4_1") {
        kernel kc(c.cvt, "kernel_convert_block_q4_1");
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.d.mem); set_arg(kc.k, 3, a.m.mem);
        gws[0] = elems / 32;
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else if (type == "q5_0") {
        kernel kc(c.cvt, "kernel_convert_block_q5_0");
        cl_ulong nblk = elems / 32;
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.qs.mem); set_arg(kc.k, 2, a.qh.mem); set_arg(kc.k, 3, a.d.mem); set_scalar(kc.k, 4, nblk);
        gws[0] = align_up(nblk, 64);
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else if (type == "q5_1") {
        kernel kc(c.cvt, "kernel_convert_block_q5_1");
        cl_ulong nblk = elems / 32;
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.qs.mem); set_arg(kc.k, 2, a.qh.mem); set_arg(kc.k, 3, a.d.mem); set_arg(kc.k, 4, a.m.mem); set_scalar(kc.k, 5, nblk);
        gws[0] = align_up(nblk, 64);
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else if (type == "q8_0") {
        kernel kc(c.cvt, "kernel_convert_block_q8_0");
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.d.mem);
        gws[0] = elems / 32;
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else if (type == "iq4_nl") {
        kernel kc(c.cvt, "kernel_convert_block_iq4_nl");
        cl_ulong nblk = elems / 32;
        cl_uchar mask0 = 0x0f, maskf = 0xf0;
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.d.mem);
        set_scalar(kc.k, 3, mask0); set_scalar(kc.k, 4, maskf); set_scalar(kc.k, 5, nblk);
        gws[0] = align_up(nblk, 64);
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else if (type == "mxfp4") {
        kernel kc(c.cvt, "kernel_convert_block_mxfp4");
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.e.mem);
        gws[0] = elems / 32;
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else if (type == "q4_k") {
        kernel kc(c.cvt, "kernel_convert_block_q4_K");
        cl_uchar mask0 = 0x0f, maskf = 0xf0;
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.s.mem); set_arg(kc.k, 3, a.d.mem); set_arg(kc.k, 4, a.dm.mem);
        set_scalar(kc.k, 5, mask0); set_scalar(kc.k, 6, maskf);
        gws[0] = elems / 256;
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else if (type == "q5_k") {
        kernel kc(c.cvt, "kernel_convert_block_q5_K");
        cl_uchar mask0 = 0x0f, maskf = 0xf0;
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.qh.mem); set_arg(kc.k, 3, a.s.mem); set_arg(kc.k, 4, a.d.mem); set_arg(kc.k, 5, a.dm.mem);
        set_scalar(kc.k, 6, mask0); set_scalar(kc.k, 7, maskf);
        gws[0] = elems / 256;
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else if (type == "q6_k") {
        kernel kc(c.cvt, "kernel_convert_block_q6_K");
        cl_uchar mask = 0xff;
        cl_ulong nblk = elems / 256;
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.ql.mem); set_arg(kc.k, 2, a.qh.mem); set_arg(kc.k, 3, a.s.mem); set_arg(kc.k, 4, a.d.mem);
        set_scalar(kc.k, 5, mask); set_scalar(kc.k, 6, nblk);
        gws[0] = align_up(nblk, 64);
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else {
        throw std::runtime_error("unsupported standard type " + type);
    }
    return t;
}

static bool prefer_tile32(int stride, int rows) {
    return rows % 1024 == 0 || stride >= 2048;
}

static run_times run_dense(cl_ctx & c, const std::string & type, int k_dim, int m_dim, bench_alloc & a, bool tiled = true, bool copy_back = true, int tile_size = 16) {
    if (m_dim % 64 != 0) {
        throw std::runtime_error("dense mode requires --m multiple of 64 for local size compatibility");
    }
    run_times t;
    run_times conv;

    if (type == "q5_0" || type == "q5_1" || type == "mxfp4") {
        return run_standard(c, type, k_dim, m_dim, a);
    }

    const uint64_t elems = uint64_t(k_dim) * uint64_t(m_dim);
    size_t gws[3] = {};
    size_t lws[3] = { 64, 1, 1 };
    auto tr8 = [&](buffer & src_dst, buffer & tmp, size_t bytes, int stride, int rows) {
        const int field_tile = tile_size == 0 ? (prefer_tile32(stride, rows) ? 32 : 16) : tile_size;
        const char * kname = field_tile == 32 ? "kernel_transpose_8_buf_tiled32" : "kernel_transpose_8_buf_tiled";
        if (tiled) {
            return copy_back ? transpose_as_tiled(c, kname, src_dst, tmp, bytes, stride, rows)
                             : transpose_as_tiled_nocopy(c, kname, src_dst, tmp, stride, rows);
        }
        return transpose_as(c, "kernel_transpose_8_buf", src_dst, tmp, bytes, stride, rows);
    };
    auto tr16 = [&](buffer & src_dst, buffer & tmp, size_t bytes, int stride, int rows) {
        const int field_tile = tile_size == 0 ? (prefer_tile32(stride, rows) ? 32 : 16) : tile_size;
        const char * kname = field_tile == 32 ? "kernel_transpose_16_buf_tiled32" : "kernel_transpose_16_buf_tiled";
        if (tiled) {
            return copy_back ? transpose_as_tiled(c, kname, src_dst, tmp, bytes, stride, rows)
                             : transpose_as_tiled_nocopy(c, kname, src_dst, tmp, stride, rows);
        }
        return transpose_as(c, "kernel_transpose_16_buf", src_dst, tmp, bytes, stride, rows);
    };
    auto tr32 = [&](buffer & src_dst, buffer & tmp, size_t bytes, int stride, int rows) {
        const int field_tile = tile_size == 0 ? (prefer_tile32(stride, rows) ? 32 : 16) : tile_size;
        const char * kname = field_tile == 32 ? "kernel_transpose_32_buf_tiled32" : "kernel_transpose_32_buf_tiled";
        if (tiled) {
            return copy_back ? transpose_as_tiled(c, kname, src_dst, tmp, bytes, stride, rows)
                             : transpose_as_tiled_nocopy(c, kname, src_dst, tmp, stride, rows);
        }
        return transpose_as(c, "kernel_transpose_32_buf", src_dst, tmp, bytes, stride, rows);
    };

    if (type == "q4_0") {
        kernel kc(c.cvt, "kernel_convert_block_q4_0_noshuffle");
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.d.mem);
        gws[0] = elems / 32;
        conv.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        add(t, conv);
        add(t, tr16(a.q, a.tmp_q, a.q.size, k_dim / 4,  m_dim));
        add(t, tr16(a.d, a.tmp_d, a.d.size, k_dim / 32, m_dim));
    } else if (type == "q4_1") {
        kernel kc(c.cvt, "kernel_convert_block_q4_1_noshuffle");
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.d.mem); set_arg(kc.k, 3, a.m.mem);
        gws[0] = elems / 32;
        conv.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        add(t, conv);
        add(t, tr16(a.q, a.tmp_q, a.q.size, k_dim / 4,  m_dim));
        add(t, tr16(a.d, a.tmp_d, a.d.size, k_dim / 32, m_dim));
        add(t, tr16(a.m, a.tmp_m, a.m.size, k_dim / 32, m_dim));
    } else if (type == "q8_0") {
        conv = run_standard(c, type, k_dim, m_dim, a);
        add(t, conv);
        add(t, tr32(a.q, a.tmp_q, a.q.size, k_dim / 4,  m_dim));
        add(t, tr16(a.d, a.tmp_d, a.d.size, k_dim / 32, m_dim));
    } else if (type == "iq4_nl") {
        kernel kc(c.cvt, "kernel_convert_block_iq4_nl_noshuffle");
        cl_ulong nblk = elems / 32;
        cl_uchar mask0 = 0x0f, maskf = 0xf0;
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.d.mem);
        set_scalar(kc.k, 3, mask0); set_scalar(kc.k, 4, maskf); set_scalar(kc.k, 5, nblk);
        gws[0] = align_up(nblk, 64);
        conv.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        add(t, conv);
        add(t, tr16(a.q, a.tmp_q, a.q.size, k_dim / 4,  m_dim));
        add(t, tr16(a.d, a.tmp_d, a.d.size, k_dim / 32, m_dim));
    } else if (type == "q4_k") {
        kernel kc(c.cvt, "kernel_convert_block_q4_K_noshuffle");
        cl_uchar mask0 = 0x0f, maskf = 0xf0;
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.s.mem); set_arg(kc.k, 3, a.d.mem); set_arg(kc.k, 4, a.dm.mem);
        set_scalar(kc.k, 5, mask0); set_scalar(kc.k, 6, maskf);
        gws[0] = elems / 256;
        conv.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        add(t, conv);
        add(t, tr16(a.q,  a.tmp_q,  a.q.size,  k_dim / 4,   m_dim));
        add(t, tr16(a.d,  a.tmp_d,  a.d.size,  k_dim / 256, m_dim));
        add(t, tr16(a.dm, a.tmp_dm, a.dm.size, k_dim / 256, m_dim));
    } else if (type == "q5_k") {
        kernel kc(c.cvt, "kernel_convert_block_q5_K_noshuffle");
        cl_uchar mask0 = 0x0f, maskf = 0xf0;
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.qh.mem); set_arg(kc.k, 3, a.s.mem); set_arg(kc.k, 4, a.d.mem); set_arg(kc.k, 5, a.dm.mem);
        set_scalar(kc.k, 6, mask0); set_scalar(kc.k, 7, maskf);
        gws[0] = elems / 256;
        conv.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        add(t, conv);
        add(t, tr16(a.q,  a.tmp_q,  a.q.size,  k_dim / 4,   m_dim));
        add(t, tr8(a.qh, a.tmp_qh, a.qh.size, k_dim / 8,   m_dim));
        add(t, tr16(a.d,  a.tmp_d,  a.d.size,  k_dim / 256, m_dim));
        add(t, tr16(a.dm, a.tmp_dm, a.dm.size, k_dim / 256, m_dim));
    } else if (type == "q6_k") {
        kernel kc(c.cvt, "kernel_convert_block_q6_K_noshuffle");
        cl_uchar mask = 0xff;
        cl_ulong nblk = elems / 256;
        set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.ql.mem); set_arg(kc.k, 2, a.qh.mem); set_arg(kc.k, 3, a.s.mem); set_arg(kc.k, 4, a.d.mem);
        set_scalar(kc.k, 5, mask); set_scalar(kc.k, 6, nblk);
        gws[0] = align_up(nblk, 64);
        conv.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        add(t, conv);
        add(t, tr16(a.ql, a.tmp_ql, a.ql.size, k_dim / 4,      m_dim));
        add(t, tr8(a.qh, a.tmp_qh, a.qh.size, k_dim / 4,      m_dim));
        add(t, tr16(a.s,  a.tmp_s,  a.s.size,  k_dim / 16 / 2, m_dim));
        add(t, tr16(a.d,  a.tmp_d,  a.d.size,  k_dim / 256,    m_dim));
    } else {
        throw std::runtime_error("unsupported dense type " + type);
    }
    return t;
}

static run_times run_dense_fused(cl_ctx & c, const std::string & type, int k_dim, int m_dim, bench_alloc & a) {
    if (m_dim % 64 != 0) {
        throw std::runtime_error("dense_fused mode requires --m multiple of 64 for local size compatibility");
    }

    const uint64_t elems = uint64_t(k_dim) * uint64_t(m_dim);
    size_t gws[3] = {};
    size_t lws[3] = { 64, 1, 1 };
    run_times t;

    if (type == "q4_0") {
        kernel kc(c.cvt, "kernel_convert_block_q4_0_noshuffle_transpose");
        set_arg(kc.k, 0, a.src.mem);
        set_arg(kc.k, 1, a.q.mem);
        set_arg(kc.k, 2, a.d.mem);
        set_scalar(kc.k, 3, cl_uint(k_dim));
        set_scalar(kc.k, 4, cl_uint(m_dim));
        gws[0] = align_up(m_dim, 64);
        gws[1] = k_dim / 32;
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else if (type == "q8_0") {
        kernel kc(c.cvt, "kernel_convert_block_q8_0_transpose");
        set_arg(kc.k, 0, a.src.mem);
        set_arg(kc.k, 1, a.q.mem);
        set_arg(kc.k, 2, a.d.mem);
        set_scalar(kc.k, 3, cl_uint(k_dim));
        set_scalar(kc.k, 4, cl_uint(m_dim));
        gws[0] = align_up(m_dim, 64);
        gws[1] = k_dim / 32;
        t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
    } else {
        (void) elems;
        throw std::runtime_error("dense_fused currently supports q4_0 and q8_0");
    }

    return t;
}

struct prepared_fused {
    kernel kt;
    size_t gws[3] = {};
    size_t lws[3] = { 64, 1, 1 };

    prepared_fused(cl_program prog, const std::string & type, int k_dim, int m_dim, bench_alloc & a) :
        kt(prog, type == "q4_0" ? "kernel_convert_block_q4_0_noshuffle_transpose" : "kernel_convert_block_q8_0_transpose") {
        if (type != "q4_0" && type != "q8_0") {
            throw std::runtime_error("dense_fused_wall currently supports q4_0 and q8_0");
        }
        set_arg(kt.k, 0, a.src.mem);
        set_arg(kt.k, 1, a.q.mem);
        set_arg(kt.k, 2, a.d.mem);
        set_scalar(kt.k, 3, cl_uint(k_dim));
        set_scalar(kt.k, 4, cl_uint(m_dim));
        gws[0] = size_t(align_up(m_dim, 64));
        gws[1] = size_t(k_dim / 32);
        gws[2] = 1;
    }
};

static void enqueue_prepared_fused_no_profile(cl_command_queue q, prepared_fused & fcmd) {
    CL_CHECK(clEnqueueNDRangeKernel(q, fcmd.kt.k, 3, nullptr, fcmd.gws, fcmd.lws, 0, nullptr, nullptr));
}

static std::vector<prepared_transpose> make_prepared_transposes(
        cl_ctx & c, const std::string & type, int k_dim, int m_dim, bench_alloc & a, bool tiled);

struct prepared_convert {
    kernel kc;
    size_t gws[3] = {};
    size_t lws[3] = { 64, 1, 1 };

    prepared_convert(cl_program prog, const std::string & type, int k_dim, int m_dim, bench_alloc & a) :
        kc(prog, type == "q4_0" ? "kernel_convert_block_q4_0_noshuffle" : "kernel_convert_block_q8_0") {
        if (type != "q4_0" && type != "q8_0") {
            throw std::runtime_error("reload_chain_ab currently supports q4_0 and q8_0");
        }
        const uint64_t elems = uint64_t(k_dim) * uint64_t(m_dim);
        set_arg(kc.k, 0, a.src.mem);
        set_arg(kc.k, 1, a.q.mem);
        set_arg(kc.k, 2, a.d.mem);
        gws[0] = size_t(elems / 32);
        gws[1] = 1;
        gws[2] = 1;
    }
};

static void enqueue_prepared_convert_no_profile(
        cl_command_queue q,
        prepared_convert & cmd,
        cl_uint wait_n = 0,
        const cl_event * wait = nullptr,
        cl_event * out = nullptr) {
    CL_CHECK(clEnqueueNDRangeKernel(q, cmd.kc.k, 3, nullptr, cmd.gws, cmd.lws, wait_n, wait, out));
}

struct reload_chain_sample {
    double wall_ms = 0.0;
};

static reload_chain_sample run_reload_chain_once(
        cl_ctx & c,
        prepared_convert & conv,
        std::vector<prepared_transpose> & transposes,
        buffer & staging,
        const std::vector<uint8_t> & input,
        bool transform_on_xfer) {
    const auto t0 = std::chrono::steady_clock::now();

    cl_event write_ev = nullptr;
    CL_CHECK(clEnqueueWriteBuffer(c.xfer_queue, staging.mem, CL_FALSE, 0, input.size(), input.data(), 0, nullptr, &write_ev));

    if (transform_on_xfer) {
        cl_event convert_ev = nullptr;
        enqueue_prepared_convert_no_profile(c.xfer_queue, conv, 1, &write_ev, &convert_ev);
        CL_CHECK(clReleaseEvent(write_ev));
        CL_CHECK(clReleaseEvent(convert_ev));
        for (prepared_transpose & tr : transposes) {
            enqueue_prepared_transpose_no_profile(c.xfer_queue, tr);
        }
        cl_event final_ev = nullptr;
        CL_CHECK(clEnqueueMarkerWithWaitList(c.xfer_queue, 0, nullptr, &final_ev));
        CL_CHECK(clFlush(c.xfer_queue));
        CL_CHECK(clEnqueueBarrierWithWaitList(c.queue, 1, &final_ev, nullptr));
        CL_CHECK(clReleaseEvent(final_ev));
    } else {
        CL_CHECK(clFlush(c.xfer_queue));
        CL_CHECK(clEnqueueBarrierWithWaitList(c.queue, 1, &write_ev, nullptr));
        CL_CHECK(clReleaseEvent(write_ev));
        enqueue_prepared_convert_no_profile(c.queue, conv);
        for (prepared_transpose & tr : transposes) {
            enqueue_prepared_transpose_no_profile(c.queue, tr);
        }
    }

    CL_CHECK(clFinish(c.queue));
    const auto t1 = std::chrono::steady_clock::now();
    return { std::chrono::duration<double, std::milli>(t1 - t0).count() };
}

static void bench_reload_chain_ab(cl_ctx & c, const std::string & type, const params & p) {
    if (type != "q4_0" && type != "q8_0") {
        throw std::runtime_error("reload_chain_ab currently supports q4_0 and q8_0");
    }
    if (p.m % 64 != 0) {
        throw std::runtime_error("reload_chain_ab requires --m multiple of 64");
    }

    bench_case bc = make_case(type, p.k, p.m, "dense");
    const uint64_t nblk = uint64_t(p.k) * uint64_t(p.m) / bc.block;
    const size_t src_bytes = nblk * bc.src_block_bytes;
    bench_alloc old_alloc = allocate_case(c, bc, src_bytes);
    bench_alloc new_alloc = allocate_case(c, bc, src_bytes);
    std::vector<uint8_t> input = make_input(src_bytes);

    prepared_convert old_conv(c.cvt, type, p.k, p.m, old_alloc);
    prepared_convert new_conv(c.cvt, type, p.k, p.m, new_alloc);
    std::vector<prepared_transpose> old_transposes = make_prepared_transposes(c, type, p.k, p.m, old_alloc, true);
    std::vector<prepared_transpose> new_transposes = make_prepared_transposes(c, type, p.k, p.m, new_alloc, true);

    for (int i = 0; i < p.warmup; ++i) {
        (void) run_reload_chain_once(c, old_conv, old_transposes, old_alloc.src, input, false);
        (void) run_reload_chain_once(c, new_conv, new_transposes, new_alloc.src, input, true);
    }

    std::vector<double> old_wall;
    std::vector<double> new_wall;
    old_wall.reserve(p.iters);
    new_wall.reserve(p.iters);
    for (int i = 0; i < p.iters; ++i) {
        old_wall.push_back(run_reload_chain_once(c, old_conv, old_transposes, old_alloc.src, input, false).wall_ms);
        new_wall.push_back(run_reload_chain_once(c, new_conv, new_transposes, new_alloc.src, input, true).wall_ms);
    }

    const stats so = calc_stats(old_wall);
    const stats sn = calc_stats(new_wall);
    const double ratio = sn.med / so.med;
    const double gib = double(src_bytes) / 1024.0 / 1024.0 / 1024.0;
    std::printf("%-7s %-15s K=%-6d M=%-6d src=%8.2f MiB repeats=%-4d "
                "old_med=%8.3f ms old_avg=%8.3f ms new_med=%8.3f ms new_avg=%8.3f ms "
                "new/old=%5.2fx old_GiB/s=%7.2f new_GiB/s=%7.2f\n",
            type_name(type), "reload_chain_ab", p.k, p.m, double(src_bytes) / 1024.0 / 1024.0, p.iters,
            so.med, so.avg, sn.med, sn.avg, ratio,
            gib / (so.med / 1000.0), gib / (sn.med / 1000.0));
}

static void bench_dense_fused_wall(cl_ctx & c, const std::string & type, const params & p) {
    if (p.m % 64 != 0) {
        throw std::runtime_error("dense_fused_wall requires --m multiple of 64");
    }
    bench_case bc = make_case(type, p.k, p.m, "dense");
    const uint64_t nblk = uint64_t(p.k) * uint64_t(p.m) / bc.block;
    const size_t src_bytes = nblk * bc.src_block_bytes;
    bench_alloc alloc = allocate_case(c, bc, src_bytes);
    std::vector<uint8_t> input = make_input(src_bytes);
    CL_CHECK(clEnqueueWriteBuffer(c.queue, alloc.src.mem, CL_TRUE, 0, input.size(), input.data(), 0, nullptr, nullptr));

    prepared_fused fused(c.cvt, type, p.k, p.m, alloc);
    for (int i = 0; i < p.warmup; ++i) {
        enqueue_prepared_fused_no_profile(c.queue, fused);
    }
    CL_CHECK(clFinish(c.queue));

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < p.iters; ++i) {
        enqueue_prepared_fused_no_profile(c.queue, fused);
    }
    CL_CHECK(clFinish(c.queue));
    const auto t1 = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double avg_ms = total_ms / double(p.iters);
    const double gib = double(src_bytes) / 1024.0 / 1024.0 / 1024.0;
    const double gelem_s = (double(p.k) * double(p.m)) / (avg_ms / 1000.0) / 1.0e9;
    std::printf("%-7s %-16s K=%-6d M=%-6d repeats=%-4d src=%8.2f MiB wall_avg=%8.3f ms "
                "wall_GiB/s=%7.2f Gelem/s=%7.2f GFLOP/s=%7.2f\n",
        type_name(type), "dense_fused_wall", p.k, p.m, p.iters, double(src_bytes) / 1024.0 / 1024.0,
        avg_ms, gib / (avg_ms / 1000.0), gelem_s, gelem_s * p.flops_per_elem);
}

static run_times run_moe(cl_ctx & c, const std::string & type, int k_dim, int m_dim, bench_alloc & a) {
    if (m_dim % 64 != 0) {
        throw std::runtime_error("moe mode requires --m multiple of 64 for local size compatibility");
    }

    const uint64_t elems = uint64_t(k_dim) * uint64_t(m_dim);
    cl_int ne00 = k_dim;
    cl_int ne01 = m_dim;
    run_times t;

    if (type == "q8_0" || type == "iq4_nl") {
        throw std::runtime_error(type + " has no set_tensor MoE trans4 path");
    }

    if (type == "q4_0" || type == "q4_1" || type == "q5_0" || type == "q5_1" || type == "mxfp4") {
        size_t gws[3] = { size_t(align_up(m_dim, 64)), size_t(k_dim / 32), 1 };
        size_t lws[3] = { 64, 2, 1 };
        if (type == "q4_0") {
            kernel kc(c.cvt, "kernel_convert_block_q4_0_trans4_ns");
            set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.d.mem);
            set_scalar(kc.k, 3, ne00); set_scalar(kc.k, 4, ne01);
            t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        } else if (type == "q4_1") {
            kernel kc(c.cvt, "kernel_convert_block_q4_1_trans4_ns");
            set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.d.mem); set_arg(kc.k, 3, a.m.mem);
            set_scalar(kc.k, 4, ne00); set_scalar(kc.k, 5, ne01);
            t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        } else if (type == "q5_0") {
            kernel kc(c.cvt, "kernel_convert_block_q5_0_trans4_ns");
            set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.qs.mem); set_arg(kc.k, 2, a.qh.mem); set_arg(kc.k, 3, a.d.mem);
            set_scalar(kc.k, 4, ne00); set_scalar(kc.k, 5, ne01);
            t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        } else if (type == "q5_1") {
            kernel kc(c.cvt, "kernel_convert_block_q5_1_trans4_ns");
            set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.qs.mem); set_arg(kc.k, 2, a.qh.mem); set_arg(kc.k, 3, a.d.mem); set_arg(kc.k, 4, a.m.mem);
            set_scalar(kc.k, 5, ne00); set_scalar(kc.k, 6, ne01);
            t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        } else {
            kernel kc(c.cvt, "kernel_convert_block_mxfp4_trans4_ns");
            set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.e.mem);
            set_scalar(kc.k, 3, ne00); set_scalar(kc.k, 4, ne01);
            t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        }
    } else if (type == "q4_k" || type == "q5_k" || type == "q6_k") {
        size_t gws[3] = { size_t(align_up(m_dim, 64)), size_t(k_dim / 256), 1 };
        size_t lws[3] = { 64, 1, 1 };
        cl_uchar mask0 = 0x0f, maskf = 0xf0;
        if (type == "q4_k") {
            kernel kc(c.cvt, "kernel_convert_block_q4_k_trans4_ns");
            set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.d.mem); set_arg(kc.k, 3, a.dm.mem); set_arg(kc.k, 4, a.s.mem);
            set_scalar(kc.k, 5, ne00); set_scalar(kc.k, 6, ne01); set_scalar(kc.k, 7, mask0); set_scalar(kc.k, 8, maskf);
            t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        } else if (type == "q5_k") {
            kernel kc(c.cvt, "kernel_convert_block_q5_k_trans4_ns");
            set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.q.mem); set_arg(kc.k, 2, a.qh.mem); set_arg(kc.k, 3, a.d.mem); set_arg(kc.k, 4, a.dm.mem); set_arg(kc.k, 5, a.s.mem);
            set_scalar(kc.k, 6, ne00); set_scalar(kc.k, 7, ne01); set_scalar(kc.k, 8, mask0); set_scalar(kc.k, 9, maskf);
            t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        } else {
            (void) elems;
            kernel kc(c.cvt, "kernel_convert_block_q6_k_trans4_ns");
            set_arg(kc.k, 0, a.src.mem); set_arg(kc.k, 1, a.ql.mem); set_arg(kc.k, 2, a.qh.mem); set_arg(kc.k, 3, a.d.mem); set_arg(kc.k, 4, a.s.mem);
            set_scalar(kc.k, 5, ne00); set_scalar(kc.k, 6, ne01); set_scalar(kc.k, 7, mask0); set_scalar(kc.k, 8, maskf);
            t.convert_ms = enqueue_kernel(c.queue, kc.k, 3, gws, lws);
        }
    } else {
        throw std::runtime_error("unsupported moe type " + type);
    }
    return t;
}

static run_times run_once(cl_ctx & c, const std::string & type, const std::string & mode, int k, int m, bench_alloc & a) {
    if (mode == "standard") return run_standard(c, type, k, m, a);
    if (mode == "dense")    return run_dense(c, type, k, m, a, true);
    if (mode == "dense_adaptive") return run_dense(c, type, k, m, a, true, true, 0);
    if (mode == "dense_tile32") return run_dense(c, type, k, m, a, true, true, 32);
    if (mode == "dense_nocopy") return run_dense(c, type, k, m, a, true, false);
    if (mode == "dense_tile32_nocopy") return run_dense(c, type, k, m, a, true, false, 32);
    if (mode == "dense_scalar") return run_dense(c, type, k, m, a, false);
    if (mode == "dense_tiled") return run_dense(c, type, k, m, a, true);
    if (mode == "dense_fused") return run_dense_fused(c, type, k, m, a);
    if (mode == "moe")      return run_moe(c, type, k, m, a);
    throw std::runtime_error("unsupported mode: " + mode);
}

static std::vector<prepared_transpose> make_prepared_transposes(cl_ctx & c, const std::string & type, int k_dim, int m_dim, bench_alloc & a, bool tiled) {
    if (m_dim % 64 != 0) {
        throw std::runtime_error("transpose mode requires --m multiple of 64 for local size compatibility");
    }

    const char * k8  = tiled ? "kernel_transpose_8_buf_tiled"  : "kernel_transpose_8_buf";
    const char * k16 = tiled ? "kernel_transpose_16_buf_tiled" : "kernel_transpose_16_buf";
    const char * k32 = tiled ? "kernel_transpose_32_buf_tiled" : "kernel_transpose_32_buf";

    std::vector<prepared_transpose> out;
    if (type == "q4_0" || type == "iq4_nl") {
        out.emplace_back(c.transpose, k16, a.q, a.tmp_q, a.q.size, k_dim / 4,  m_dim, tiled);
        out.emplace_back(c.transpose, k16, a.d, a.tmp_d, a.d.size, k_dim / 32, m_dim, tiled);
    } else if (type == "q4_1") {
        out.emplace_back(c.transpose, k16, a.q, a.tmp_q, a.q.size, k_dim / 4,  m_dim, tiled);
        out.emplace_back(c.transpose, k16, a.d, a.tmp_d, a.d.size, k_dim / 32, m_dim, tiled);
        out.emplace_back(c.transpose, k16, a.m, a.tmp_m, a.m.size, k_dim / 32, m_dim, tiled);
    } else if (type == "q8_0") {
        out.emplace_back(c.transpose, k32, a.q, a.tmp_q, a.q.size, k_dim / 4,  m_dim, tiled);
        out.emplace_back(c.transpose, k16, a.d, a.tmp_d, a.d.size, k_dim / 32, m_dim, tiled);
    } else if (type == "q4_k") {
        out.emplace_back(c.transpose, k16, a.q,  a.tmp_q,  a.q.size,  k_dim / 4,   m_dim, tiled);
        out.emplace_back(c.transpose, k16, a.d,  a.tmp_d,  a.d.size,  k_dim / 256, m_dim, tiled);
        out.emplace_back(c.transpose, k16, a.dm, a.tmp_dm, a.dm.size, k_dim / 256, m_dim, tiled);
    } else if (type == "q5_k") {
        out.emplace_back(c.transpose, k16, a.q,  a.tmp_q,  a.q.size,  k_dim / 4,   m_dim, tiled);
        out.emplace_back(c.transpose, k8,  a.qh, a.tmp_qh, a.qh.size, k_dim / 8,   m_dim, tiled);
        out.emplace_back(c.transpose, k16, a.d,  a.tmp_d,  a.d.size,  k_dim / 256, m_dim, tiled);
        out.emplace_back(c.transpose, k16, a.dm, a.tmp_dm, a.dm.size, k_dim / 256, m_dim, tiled);
    } else if (type == "q6_k") {
        out.emplace_back(c.transpose, k16, a.ql, a.tmp_ql, a.ql.size, k_dim / 4,      m_dim, tiled);
        out.emplace_back(c.transpose, k8,  a.qh, a.tmp_qh, a.qh.size, k_dim / 4,      m_dim, tiled);
        out.emplace_back(c.transpose, k16, a.s,  a.tmp_s,  a.s.size,  k_dim / 16 / 2, m_dim, tiled);
        out.emplace_back(c.transpose, k16, a.d,  a.tmp_d,  a.d.size,  k_dim / 256,    m_dim, tiled);
    } else {
        throw std::runtime_error("transpose mode is only for dense-transposed types");
    }
    return out;
}

static void bench_transpose_only(cl_ctx & c, const std::string & type, const params & p, bool tiled) {
    bench_case bc = make_case(type, p.k, p.m, "dense");
    const uint64_t nblk = uint64_t(p.k) * uint64_t(p.m) / bc.block;
    const size_t src_bytes = nblk * bc.src_block_bytes;
    bench_alloc alloc = allocate_case(c, bc, src_bytes);
    std::vector<prepared_transpose> transposes = make_prepared_transposes(c, type, p.k, p.m, alloc, tiled);

    for (int i = 0; i < p.warmup; ++i) {
        for (prepared_transpose & tr : transposes) {
            (void) enqueue_prepared_transpose(c.queue, tr);
        }
    }

    CL_CHECK(clFinish(c.queue));
    const auto wall_t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < p.iters; ++i) {
        for (prepared_transpose & tr : transposes) {
            enqueue_prepared_transpose_no_profile(c.queue, tr);
        }
    }
    CL_CHECK(clFinish(c.queue));
    const auto wall_t1 = std::chrono::steady_clock::now();

    run_times prof_sum;
    for (int i = 0; i < p.iters; ++i) {
        for (prepared_transpose & tr : transposes) {
            add(prof_sum, enqueue_prepared_transpose(c.queue, tr));
        }
    }

    const double wall_total_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
    const double prof_total_ms = prof_sum.total_ms();
    const double wall_avg_ms = wall_total_ms / double(p.iters);
    const double prof_avg_ms = prof_total_ms / double(p.iters);
    const double prof_kernel_avg_ms = prof_sum.transpose_kernel_ms / double(p.iters);
    const double prof_copy_avg_ms = prof_sum.transpose_copy_ms / double(p.iters);
    const double gib = double(src_bytes) / 1024.0 / 1024.0 / 1024.0;

    std::printf("%-7s %-8s K=%-6d M=%-6d repeats=%-4d transposes/repeat=%zu src=%8.2f MiB "
                "cpu_wall_avg=%8.3f ms profile_avg=%8.3f ms profile_kernel_avg=%8.3f ms profile_copy_avg=%8.3f ms "
                "wall_GiB/s=%7.2f profile_GiB/s=%7.2f\n",
        type_name(type), tiled ? "tr_tile" : "transpose", p.k, p.m, p.iters, transposes.size(), double(src_bytes) / 1024.0 / 1024.0,
        wall_avg_ms, prof_avg_ms, prof_kernel_avg_ms, prof_copy_avg_ms,
        gib / (wall_avg_ms / 1000.0), gib / (prof_avg_ms / 1000.0));
}

static void fill_device_buffer(cl_command_queue q, buffer & b, uint32_t seed) {
    std::vector<uint8_t> data(b.size);
    uint32_t r = seed ^ uint32_t(b.size);
    for (uint8_t & v : data) {
        r = r * 1664525u + 1013904223u;
        v = uint8_t(r >> 24);
    }
    CL_CHECK(clEnqueueWriteBuffer(q, b.mem, CL_TRUE, 0, data.size(), data.data(), 0, nullptr, nullptr));
}

static bool check_transpose_field(
        cl_ctx & c, const char * label,
        const char * scalar_kernel, const char * tiled_kernel,
        size_t bytes, int stride, int rows,
        uint32_t seed) {
    buffer scalar_src(c.context, bytes);
    buffer scalar_tmp(c.context, bytes);
    buffer tiled_src(c.context, bytes);
    buffer tiled_tmp(c.context, bytes);

    fill_device_buffer(c.queue, scalar_src, seed);
    std::vector<uint8_t> input(bytes);
    CL_CHECK(clEnqueueReadBuffer(c.queue, scalar_src.mem, CL_TRUE, 0, input.size(), input.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(c.queue, tiled_src.mem, CL_TRUE, 0, input.size(), input.data(), 0, nullptr, nullptr));

    (void) transpose_as(c, scalar_kernel, scalar_src, scalar_tmp, bytes, stride, rows);
    (void) transpose_as_tiled(c, tiled_kernel, tiled_src, tiled_tmp, bytes, stride, rows);

    std::vector<uint8_t> scalar_out(bytes);
    std::vector<uint8_t> tiled_out(bytes);
    CL_CHECK(clEnqueueReadBuffer(c.queue, scalar_src.mem, CL_TRUE, 0, scalar_out.size(), scalar_out.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(c.queue, tiled_src.mem, CL_TRUE, 0, tiled_out.size(), tiled_out.data(), 0, nullptr, nullptr));

    const bool ok = scalar_out == tiled_out;
    std::printf("  %-8s %-4s bytes=%8zu stride=%6d rows=%6d\n", label, ok ? "OK" : "FAIL", bytes, stride, rows);
    if (!ok) {
        for (size_t i = 0; i < bytes; ++i) {
            if (scalar_out[i] != tiled_out[i]) {
                std::printf("    first mismatch at byte %zu: scalar=%u tiled=%u\n",
                    i, unsigned(scalar_out[i]), unsigned(tiled_out[i]));
                break;
            }
        }
    }
    return ok;
}

static void bench_transpose_check(cl_ctx & c, const std::string & type, const params & p, int tile_size = 16) {
    bench_case bc = make_case(type, p.k, p.m, "dense");
    bool ok = true;
    uint32_t seed = 0x1234abcdU;

    auto check8 = [&](const char * label, size_t bytes, int stride, int rows) {
        if (bytes) {
            const char * tiled_kernel = tile_size == 32 ? "kernel_transpose_8_buf_tiled32" : "kernel_transpose_8_buf_tiled";
            ok = check_transpose_field(c, label, "kernel_transpose_8_buf", tiled_kernel,
                bytes, stride, rows, seed++) && ok;
        }
    };
    auto check16 = [&](const char * label, size_t bytes, int stride, int rows) {
        if (bytes) {
            const char * tiled_kernel = tile_size == 32 ? "kernel_transpose_16_buf_tiled32" : "kernel_transpose_16_buf_tiled";
            ok = check_transpose_field(c, label, "kernel_transpose_16_buf", tiled_kernel,
                bytes, stride, rows, seed++) && ok;
        }
    };
    auto check32 = [&](const char * label, size_t bytes, int stride, int rows) {
        if (bytes) {
            const char * tiled_kernel = tile_size == 32 ? "kernel_transpose_32_buf_tiled32" : "kernel_transpose_32_buf_tiled";
            ok = check_transpose_field(c, label, "kernel_transpose_32_buf", tiled_kernel,
                bytes, stride, rows, seed++) && ok;
        }
    };

    if (type == "q4_0" || type == "iq4_nl") {
        check16("q", bc.q_bytes, p.k / 4, p.m);
        check16("d", bc.d_bytes, p.k / 32, p.m);
    } else if (type == "q4_1") {
        check16("q", bc.q_bytes, p.k / 4, p.m);
        check16("d", bc.d_bytes, p.k / 32, p.m);
        check16("m", bc.m_bytes, p.k / 32, p.m);
    } else if (type == "q8_0") {
        check32("q", bc.q_bytes, p.k / 4, p.m);
        check16("d", bc.d_bytes, p.k / 32, p.m);
    } else if (type == "q4_k") {
        check16("q",  bc.q_bytes,  p.k / 4,   p.m);
        check16("d",  bc.d_bytes,  p.k / 256, p.m);
        check16("dm", bc.dm_bytes, p.k / 256, p.m);
    } else if (type == "q5_k") {
        check16("q",  bc.q_bytes,  p.k / 4,   p.m);
        check8 ("qh", bc.qh_bytes, p.k / 8,   p.m);
        check16("d",  bc.d_bytes,  p.k / 256, p.m);
        check16("dm", bc.dm_bytes, p.k / 256, p.m);
    } else if (type == "q6_k") {
        check16("ql", bc.ql_bytes, p.k / 4,      p.m);
        check8 ("qh", bc.qh_bytes, p.k / 4,      p.m);
        check16("s",  bc.s_bytes,  p.k / 16 / 2, p.m);
        check16("d",  bc.d_bytes,  p.k / 256,    p.m);
    } else {
        throw std::runtime_error("transpose_check is only for dense-transposed types");
    }

    std::printf("%-7s %-15s K=%-6d M=%-6d %s\n", type_name(type), tile_size == 32 ? "transpose_check32" : "transpose_check", p.k, p.m, ok ? "OK" : "FAIL");
    if (!ok) {
        throw std::runtime_error("transpose_check failed for " + type);
    }
}

static bool compare_buffer(cl_command_queue q, const char * label, const buffer & a, const buffer & b) {
    if (a.size != b.size) {
        std::printf("  %-8s FAIL size mismatch %zu vs %zu\n", label, a.size, b.size);
        return false;
    }

    std::vector<uint8_t> va(a.size);
    std::vector<uint8_t> vb(b.size);
    CL_CHECK(clEnqueueReadBuffer(q, a.mem, CL_TRUE, 0, va.size(), va.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(q, b.mem, CL_TRUE, 0, vb.size(), vb.data(), 0, nullptr, nullptr));

    const bool ok = va == vb;
    std::printf("  %-8s %-4s bytes=%8zu\n", label, ok ? "OK" : "FAIL", a.size);
    if (!ok) {
        for (size_t i = 0; i < va.size(); ++i) {
            if (va[i] != vb[i]) {
                std::printf("    first mismatch at byte %zu: dense=%u fused=%u\n",
                    i, unsigned(va[i]), unsigned(vb[i]));
                break;
            }
        }
    }
    return ok;
}

static void bench_dense_fused_check(cl_ctx & c, const std::string & type, const params & p) {
    if (type != "q4_0" && type != "q8_0") {
        throw std::runtime_error("dense_fused_check currently supports q4_0 and q8_0");
    }

    bench_case bc = make_case(type, p.k, p.m, "dense");
    const uint64_t nblk = uint64_t(p.k) * uint64_t(p.m) / bc.block;
    const size_t src_bytes = nblk * bc.src_block_bytes;
    bench_alloc dense = allocate_case(c, bc, src_bytes);
    bench_alloc fused = allocate_case(c, bc, src_bytes);

    std::vector<uint8_t> input = make_input(src_bytes);
    CL_CHECK(clEnqueueWriteBuffer(c.queue, dense.src.mem, CL_TRUE, 0, input.size(), input.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(c.queue, fused.src.mem, CL_TRUE, 0, input.size(), input.data(), 0, nullptr, nullptr));

    (void) run_dense(c, type, p.k, p.m, dense, true);
    (void) run_dense_fused(c, type, p.k, p.m, fused);

    bool ok = true;
    if (type == "q4_0") {
        ok = compare_buffer(c.queue, "q", dense.q, fused.q) && ok;
        ok = compare_buffer(c.queue, "d", dense.d, fused.d) && ok;
    } else if (type == "q8_0") {
        ok = compare_buffer(c.queue, "q", dense.q, fused.q) && ok;
        ok = compare_buffer(c.queue, "d", dense.d, fused.d) && ok;
    }

    std::printf("%-7s %-15s K=%-6d M=%-6d %s\n", type_name(type), "dense_fused_check", p.k, p.m, ok ? "OK" : "FAIL");
    if (!ok) {
        throw std::runtime_error("dense_fused_check failed for " + type);
    }
}

static void bench_one(cl_ctx & c, const std::string & type, const std::string & mode, const params & p) {
    if (mode == "transpose") {
        bench_transpose_only(c, type, p, false);
        return;
    }
    if (mode == "transpose_tiled") {
        bench_transpose_only(c, type, p, true);
        return;
    }
    if (mode == "transpose_check") {
        bench_transpose_check(c, type, p);
        return;
    }
    if (mode == "transpose_check32") {
        bench_transpose_check(c, type, p, 32);
        return;
    }
    if (mode == "dense_fused_wall") {
        bench_dense_fused_wall(c, type, p);
        return;
    }
    if (mode == "dense_fused_check") {
        bench_dense_fused_check(c, type, p);
        return;
    }
    if (mode == "reload_chain_ab") {
        bench_reload_chain_ab(c, type, p);
        return;
    }

    bench_case bc = make_case(type, p.k, p.m, mode);
    const uint64_t nblk = uint64_t(p.k) * uint64_t(p.m) / bc.block;
    const size_t src_bytes = nblk * bc.src_block_bytes;

    if ((mode == "standard" || mode == "dense" || mode == "dense_scalar" || mode == "dense_tiled") && ((nblk % 64) != 0) &&
        (type == "q4_0" || type == "q4_1" || type == "q8_0" || type == "q4_k" || type == "q5_k")) {
        throw std::runtime_error(type + " requires block count multiple of 64 for this benchmark local size");
    }

    bench_alloc alloc = allocate_case(c, bc, src_bytes);
    std::vector<uint8_t> input = make_input(src_bytes);
    CL_CHECK(clEnqueueWriteBuffer(c.queue, alloc.src.mem, CL_TRUE, 0, input.size(), input.data(), 0, nullptr, nullptr));

    for (int i = 0; i < p.warmup; ++i) {
        (void) run_once(c, type, mode, p.k, p.m, alloc);
    }

    std::vector<double> total;
    std::vector<double> convert;
    std::vector<double> tker;
    std::vector<double> tcopy;
    total.reserve(p.iters);
    convert.reserve(p.iters);
    tker.reserve(p.iters);
    tcopy.reserve(p.iters);

    for (int i = 0; i < p.iters; ++i) {
        const run_times r = run_once(c, type, mode, p.k, p.m, alloc);
        total.push_back(r.total_ms());
        convert.push_back(r.convert_ms);
        tker.push_back(r.transpose_kernel_ms);
        tcopy.push_back(r.transpose_copy_ms);
    }

    const stats st = calc_stats(total);
    const stats sc = calc_stats(convert);
    const stats sk = calc_stats(tker);
    const stats sp = calc_stats(tcopy);
    const double gib = double(src_bytes) / 1024.0 / 1024.0 / 1024.0;
    const double gib_s = gib / (st.med / 1000.0);
    const double gelem_s = (double(p.k) * double(p.m)) / (st.med / 1000.0) / 1.0e9;
    const double gflop_s = gelem_s * p.flops_per_elem;

    std::printf("%-7s %-8s K=%-6d M=%-6d src=%8.2f MiB total_med=%8.3f ms total_avg=%8.3f ms GiB/s=%7.2f Gelem/s=%7.2f GFLOP/s=%7.2f",
        type_name(type), mode.c_str(), p.k, p.m, double(src_bytes) / 1024.0 / 1024.0, st.med, st.avg, gib_s, gelem_s, gflop_s);
    std::printf("  convert_med=%7.3f transpose_kernel_med=%7.3f transpose_copy_med=%7.3f min=%7.3f max=%7.3f\n",
        sc.med, sk.med, sp.med, st.min, st.max);
}

int main(int argc, char ** argv) {
    try {
        const params p = parse(argc, argv);
        cl_ctx c = init_opencl(p);

        const std::vector<std::string> all_types = {
            "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl", "q4_k", "q5_k", "q6_k", "mxfp4",
        };
        const std::vector<std::string> all_modes = { "standard", "dense", "dense_adaptive", "dense_tile32", "dense_nocopy", "dense_tile32_nocopy", "dense_scalar", "dense_fused", "dense_fused_wall", "dense_fused_check", "moe", "transpose", "transpose_tiled", "transpose_check", "transpose_check32", "reload_chain_ab" };

        std::vector<std::string> types = p.type == "all" ? all_types : std::vector<std::string>{ p.type };
        std::vector<std::string> modes = p.mode == "all" ? all_modes : std::vector<std::string>{ p.mode };

        std::printf("timing uses OpenCL event profiling; buffer/image creation and host upload are excluded\n");
        for (const std::string & mode : modes) {
            for (const std::string & type : types) {
                try {
                    bench_one(c, type, mode, p);
                } catch (const std::exception & e) {
                    std::printf("%-7s %-8s skipped: %s\n", type.c_str(), mode.c_str(), e.what());
                }
            }
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
