#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <CL/cl.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
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

static constexpr size_t MiB = 1024ull * 1024ull;

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static uint64_t align_up(uint64_t x, uint64_t a) {
    return ((x + a - 1) / a) * a;
}

static std::string read_text(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

struct stats {
    double min = 0.0;
    double med = 0.0;
    double avg = 0.0;
    double max = 0.0;
};

static stats calc_stats(std::vector<double> v) {
    if (v.empty()) {
        return {};
    }
    std::sort(v.begin(), v.end());
    stats s;
    s.min = v.front();
    s.med = v[v.size() / 2];
    s.avg = std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
    s.max = v.back();
    return s;
}

struct params {
    std::string file;
    std::string kernel_dir = LLAMA_OPENCL_KERNEL_DIR;
    int platform = 0;
    int device = 0;
    int iters = 20;
    int warmup = 3;
    int load_mb = 32;
    int cpu_xform_mb = 32;
    int cpu_compute_mb = 32;
    int gpu_compute_mb = 32;
    int k = 4096;
    int m = 4096;
    int cpu_rounds = 16;
    int gpu_rounds = 256;
    int pipeline_items = 24;
    int pipeline_slots = 4;
    int prefetch_slots = 4;
    std::string pipeline_gpu_mode = "full";
    bool pipeline = false;
    bool pipeline_only = false;
};

static void usage(const char * argv0) {
    std::printf(
        "usage: %s [options]\n"
        "\n"
        "Pipeline contention microbenchmark for elastic-memory stages.\n"
        "It measures each stage alone and then starts selected stages together.\n"
        "competition = overlap_wall_ms / max(single_stage_ms...). 1.0 means ideal overlap.\n"
        "\n"
        "options:\n"
        "  --file <path>          file for O_DIRECT disk load; omit to skip disk_load pairs\n"
        "  --load-mb <int>        bytes read by disk_load per iteration, default 32\n"
        "  --cpu-xform-mb <int>   q4_0-like host transform input bytes, default 32\n"
        "  --cpu-compute-mb <int> float working-set bytes, default 32\n"
        "  --gpu-compute-mb <int> float working-set bytes, default 32\n"
        "  --k <int>              q4_0 GPU transform K dimension, default 4096\n"
        "  --m <int>              q4_0 GPU transform M dimension, default 4096\n"
        "  --cpu-rounds <int>     CPU compute loop rounds, default 16\n"
        "  --gpu-rounds <int>     GPU compute kernel rounds, default 256\n"
        "  --prefetch-slots <int> ring slots for async disk prefetch, default 4\n"
        "  --pipeline             also run disk->CPU-xform->GPU-xform->GPU-compute pipeline\n"
        "  --pipeline-only        skip pairwise contention tests and only run the pipeline\n"
        "  --pipeline-items <int> number of items through the pipeline, default 24\n"
        "  --pipeline-slots <int> ring slots, default 4; use 1 for no-overlap baseline\n"
        "  --pipeline-gpu-mode <full|kernels-only>\n"
        "                         full runs GPU write + transpose + copy-back, default\n"
        "                         kernels-only runs transpose kernels only with preloaded GPU buffers\n"
        "  --iters <int>          timed iterations, default 20\n"
        "  --warmup <int>         warmup iterations, default 3\n"
        "  --platform <int>       OpenCL platform index, default 0\n"
        "  --device <int>         OpenCL device index, default 0\n"
        "  --kernel-dir <path>    directory containing cvt.cl and transpose.cl\n",
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
        } else if (a == "--file") {
            p.file = need("--file");
        } else if (a == "--load-mb") {
            p.load_mb = std::atoi(need("--load-mb"));
        } else if (a == "--cpu-xform-mb") {
            p.cpu_xform_mb = std::atoi(need("--cpu-xform-mb"));
        } else if (a == "--cpu-compute-mb") {
            p.cpu_compute_mb = std::atoi(need("--cpu-compute-mb"));
        } else if (a == "--gpu-compute-mb") {
            p.gpu_compute_mb = std::atoi(need("--gpu-compute-mb"));
        } else if (a == "--k") {
            p.k = std::atoi(need("--k"));
        } else if (a == "--m") {
            p.m = std::atoi(need("--m"));
        } else if (a == "--cpu-rounds") {
            p.cpu_rounds = std::atoi(need("--cpu-rounds"));
        } else if (a == "--gpu-rounds") {
            p.gpu_rounds = std::atoi(need("--gpu-rounds"));
        } else if (a == "--prefetch-slots") {
            p.prefetch_slots = std::atoi(need("--prefetch-slots"));
        } else if (a == "--pipeline") {
            p.pipeline = true;
        } else if (a == "--pipeline-only") {
            p.pipeline = true;
            p.pipeline_only = true;
        } else if (a == "--pipeline-items") {
            p.pipeline_items = std::atoi(need("--pipeline-items"));
        } else if (a == "--pipeline-slots") {
            p.pipeline_slots = std::atoi(need("--pipeline-slots"));
        } else if (a == "--pipeline-gpu-mode") {
            p.pipeline_gpu_mode = need("--pipeline-gpu-mode");
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
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (p.iters <= 0 || p.warmup < 0 || p.k <= 0 || p.m <= 0 ||
        p.load_mb <= 0 || p.cpu_xform_mb <= 0 || p.cpu_compute_mb <= 0 ||
        p.gpu_compute_mb <= 0 || p.cpu_rounds <= 0 || p.gpu_rounds <= 0) {
        throw std::runtime_error("invalid non-positive benchmark parameter");
    }
    if (p.pipeline_items <= 0 || p.pipeline_slots < 1 || p.prefetch_slots < 1) {
        throw std::runtime_error("invalid pipeline item/slot/prefetch count");
    }
    if (p.pipeline_gpu_mode != "full" && p.pipeline_gpu_mode != "kernels-only") {
        throw std::runtime_error("--pipeline-gpu-mode must be full or kernels-only");
    }
    if (p.k % 32 != 0) {
        throw std::runtime_error("--k must be a multiple of 32 for q4_0");
    }
    return p;
}

static cl_program build_program(cl_context ctx, cl_device_id dev, const std::string & src, const char * name) {
    const char * csrc = src.c_str();
    const size_t len = src.size();
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

struct cl_env {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue compute_q = nullptr;
    cl_command_queue xform_q = nullptr;
    cl_program cvt = nullptr;
    cl_program transpose = nullptr;
    cl_program compute = nullptr;

    ~cl_env() {
        if (compute) clReleaseProgram(compute);
        if (transpose) clReleaseProgram(transpose);
        if (cvt) clReleaseProgram(cvt);
        if (xform_q) clReleaseCommandQueue(xform_q);
        if (compute_q) clReleaseCommandQueue(compute_q);
        if (context) clReleaseContext(context);
    }
};

static cl_command_queue make_queue(cl_context ctx, cl_device_id dev) {
    cl_int err = CL_SUCCESS;
#ifdef CL_VERSION_2_0
    const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
#else
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    CL_CHECK(err);
    return q;
}

static cl_env init_opencl(const params & p) {
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

    cl_env e;
    e.platform = platforms[p.platform];
    e.device = devices[p.device];
    cl_int err = CL_SUCCESS;
    e.context = clCreateContext(nullptr, 1, &e.device, nullptr, nullptr, &err);
    CL_CHECK(err);
    e.compute_q = make_queue(e.context, e.device);
    e.xform_q = make_queue(e.context, e.device);
    e.cvt = build_program(e.context, e.device, read_text(p.kernel_dir + "/cvt.cl"), "cvt.cl");
    e.transpose = build_program(e.context, e.device, read_text(p.kernel_dir + "/transpose.cl"), "transpose.cl");
    const char * compute_src = R"CLC(
        kernel void elastic_gpu_compute(global const float * a, global const float * b, global float * c, uint n, uint rounds) {
            uint i = get_global_id(0);
            if (i >= n) return;
            float x = a[i];
            float y = b[(i * 17u + 13u) % n];
            for (uint r = 0; r < rounds; ++r) {
                x = fma(x, 1.00010002f, y);
                y = fma(y, 0.99989998f, x);
            }
            c[i] = x + y;
        }
    )CLC";
    e.compute = build_program(e.context, e.device, compute_src, "elastic_gpu_compute");

    char dev_name[256] = {};
    char plat_name[256] = {};
    clGetPlatformInfo(e.platform, CL_PLATFORM_NAME, sizeof(plat_name), plat_name, nullptr);
    clGetDeviceInfo(e.device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    std::printf("platform=%s device=%s\n", plat_name, dev_name);
    return e;
}

struct cl_buffer {
    cl_context ctx = nullptr;
    cl_mem mem = nullptr;
    size_t size = 0;

    cl_buffer() = default;
    cl_buffer(cl_context c, size_t n, cl_mem_flags flags = CL_MEM_READ_WRITE) : ctx(c), size(n) {
        cl_int err = CL_SUCCESS;
        mem = clCreateBuffer(ctx, flags, std::max<size_t>(n, 1), nullptr, &err);
        CL_CHECK(err);
    }
    cl_buffer(const cl_buffer &) = delete;
    cl_buffer & operator=(const cl_buffer &) = delete;
    cl_buffer(cl_buffer && other) noexcept : ctx(other.ctx), mem(other.mem), size(other.size) {
        other.mem = nullptr;
    }
    cl_buffer & operator=(cl_buffer && other) noexcept {
        if (this != &other) {
            if (mem) clReleaseMemObject(mem);
            ctx = other.ctx;
            mem = other.mem;
            size = other.size;
            other.mem = nullptr;
            other.size = 0;
        }
        return *this;
    }
    ~cl_buffer() {
        if (mem) clReleaseMemObject(mem);
    }
};

struct cl_kernel_wrap {
    cl_kernel k = nullptr;
    cl_kernel_wrap() = default;
    cl_kernel_wrap(cl_program prog, const char * name) {
        cl_int err = CL_SUCCESS;
        k = clCreateKernel(prog, name, &err);
        CL_CHECK(err);
    }
    cl_kernel_wrap(const cl_kernel_wrap &) = delete;
    cl_kernel_wrap & operator=(const cl_kernel_wrap &) = delete;
    ~cl_kernel_wrap() {
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

class barrier {
public:
    explicit barrier(int count) : total(count), left(count) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        const int gen = generation;
        if (--left == 0) {
            generation++;
            left = total;
            cv.notify_all();
        } else {
            cv.wait(lock, [&] { return gen != generation; });
        }
    }
private:
    int total;
    int left;
    int generation = 0;
    std::mutex mtx;
    std::condition_variable cv;
};

struct disk_load_stage {
    int fd = -1;
    size_t file_size = 0;
    size_t bytes = 0;
    size_t align = 4096;
    void * dst = nullptr;
    std::mt19937_64 rng{1};

    disk_load_stage(const std::string & path, size_t nbytes) : bytes(nbytes) {
        fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) {
            throw std::runtime_error("failed to open O_DIRECT file: " + path + " errno=" + std::to_string(errno));
        }
        struct stat st {};
        if (fstat(fd, &st) != 0) {
            throw std::runtime_error("fstat failed");
        }
        file_size = size_t(st.st_size);
        align = std::max<size_t>(4096, size_t(st.st_blksize));
        bytes = align_up(bytes, align);
        if (file_size <= bytes) {
            throw std::runtime_error("file is smaller than requested load chunk");
        }
        if (posix_memalign(&dst, align, bytes) != 0) {
            throw std::runtime_error("posix_memalign failed for disk buffer");
        }
    }
    ~disk_load_stage() {
        if (dst) free(dst);
        if (fd >= 0) close(fd);
    }
    void operator()() {
        const size_t max_off = file_size - bytes;
        const size_t slots = max_off / align;
        const off_t off = off_t((rng() % std::max<size_t>(slots, 1)) * align);
        ssize_t got = pread(fd, dst, bytes, off);
        if (got != ssize_t(bytes)) {
            throw std::runtime_error("pread O_DIRECT short read/error");
        }
        volatile uint8_t sink = static_cast<uint8_t *>(dst)[0];
        (void) sink;
    }
    void read_into(void * out, size_t nbytes) {
        const size_t n = align_up(nbytes, align);
        if (n > bytes) {
            throw std::runtime_error("pipeline read exceeds disk_load_stage buffer size");
        }
        const size_t max_off = file_size - n;
        const size_t slots = max_off / align;
        const off_t off = off_t((rng() % std::max<size_t>(slots, 1)) * align);
        ssize_t got = pread(fd, out, n, off);
        if (got != ssize_t(n)) {
            throw std::runtime_error("pipeline pread O_DIRECT short read/error");
        }
    }
};

struct disk_async_prefetch_stage {
    struct slot {
        void * data = nullptr;
        bool full = false;
    };

    int fd = -1;
    size_t file_size = 0;
    size_t bytes = 0;
    size_t align = 4096;
    std::vector<slot> slots;
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;
    bool started = false;
    size_t fill_idx = 0;
    size_t consume_idx = 0;
    std::exception_ptr worker_error;
    std::mt19937_64 rng{3};

    disk_async_prefetch_stage(const std::string & path, size_t nbytes, int nslots) : bytes(nbytes) {
        fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) {
            throw std::runtime_error("failed to open async O_DIRECT file: " + path + " errno=" + std::to_string(errno));
        }
        struct stat st {};
        if (fstat(fd, &st) != 0) {
            throw std::runtime_error("async fstat failed");
        }
        file_size = size_t(st.st_size);
        align = std::max<size_t>(4096, size_t(st.st_blksize));
        bytes = align_up(bytes, align);
        if (file_size <= bytes) {
            throw std::runtime_error("file is smaller than requested async load chunk");
        }
        slots.resize(size_t(nslots));
        for (slot & s : slots) {
            if (posix_memalign(&s.data, align, bytes) != 0) {
                throw std::runtime_error("posix_memalign failed for async disk buffer");
            }
        }
    }

    disk_async_prefetch_stage(const disk_async_prefetch_stage &) = delete;
    disk_async_prefetch_stage & operator=(const disk_async_prefetch_stage &) = delete;

    ~disk_async_prefetch_stage() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        for (slot & s : slots) {
            if (s.data) free(s.data);
        }
        if (fd >= 0) close(fd);
    }

    void start_worker() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!started) {
            started = true;
            worker = std::thread([this] { run_worker(); });
        }
    }

    void run_worker() {
        try {
            while (true) {
                slot * s = nullptr;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] { return stop || !slots[fill_idx].full; });
                    if (stop) {
                        return;
                    }
                    s = &slots[fill_idx];
                }

                const size_t max_off = file_size - bytes;
                const size_t nslots = max_off / align;
                const off_t off = off_t((rng() % std::max<size_t>(nslots, 1)) * align);
                ssize_t got = pread(fd, s->data, bytes, off);
                if (got != ssize_t(bytes)) {
                    throw std::runtime_error("async pread O_DIRECT short read/error");
                }

                {
                    std::lock_guard<std::mutex> lock(mtx);
                    s->full = true;
                    fill_idx = (fill_idx + 1) % slots.size();
                }
                cv.notify_all();
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                worker_error = std::current_exception();
                stop = true;
            }
            cv.notify_all();
        }
    }

    void operator()() {
        start_worker();
        slot * s = nullptr;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return stop || slots[consume_idx].full; });
            if (worker_error) {
                std::rethrow_exception(worker_error);
            }
            if (stop && !slots[consume_idx].full) {
                throw std::runtime_error("async prefetch worker stopped");
            }
            s = &slots[consume_idx];
        }
        volatile uint8_t sink = static_cast<uint8_t *>(s->data)[0];
        (void) sink;
        {
            std::lock_guard<std::mutex> lock(mtx);
            s->full = false;
            consume_idx = (consume_idx + 1) % slots.size();
        }
        cv.notify_all();
    }
};

struct cpu_mem_load_stage {
    std::vector<uint8_t> src;
    std::vector<uint8_t> dst;

    explicit cpu_mem_load_stage(size_t bytes) :
        src(std::max<size_t>(bytes, 4096)),
        dst(src.size()) {
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = uint8_t(i * 19u + 7u);
        }
    }

    void operator()() {
        std::memcpy(dst.data(), src.data(), src.size());
        volatile uint8_t sink = dst[src.size() / 2];
        (void) sink;
    }
};

struct cpu_xform_stage {
    struct q4blk {
        uint16_t d;
        uint8_t qs[16];
    };
    size_t blocks = 0;
    std::vector<q4blk> src;
    std::vector<uint8_t> q;
    std::vector<uint16_t> d;

    explicit cpu_xform_stage(size_t input_bytes) {
        blocks = std::max<size_t>(1, input_bytes / sizeof(q4blk));
        src.resize(blocks);
        q.resize(blocks * 16);
        d.resize(blocks);
        for (size_t i = 0; i < blocks; ++i) {
            src[i].d = uint16_t(i * 13u);
            for (int j = 0; j < 16; ++j) {
                src[i].qs[j] = uint8_t(i + j * 7u);
            }
        }
    }
    void operator()() {
        for (size_t i = 0; i < blocks; ++i) {
            d[i] = src[i].d;
            const uint8_t * in = src[i].qs;
            uint8_t * out = q.data() + i * 16;
            for (int j = 0; j < 8; ++j) {
                const uint8_t x0 = in[2 * j + 0];
                const uint8_t x1 = in[2 * j + 1];
                out[j]     = uint8_t((x0 & 0x0f) | ((x1 & 0x0f) << 4));
                out[j + 8] = uint8_t(((x0 & 0xf0) >> 4) | (x1 & 0xf0));
            }
        }
        volatile uint8_t sink = q[blocks & 15u];
        (void) sink;
    }
    static void transform_raw(const uint8_t * raw, size_t nblocks, uint8_t * out_q, uint16_t * out_d) {
        const q4blk * src_blk = reinterpret_cast<const q4blk *>(raw);
        for (size_t i = 0; i < nblocks; ++i) {
            out_d[i] = src_blk[i].d;
            const uint8_t * in = src_blk[i].qs;
            uint8_t * out = out_q + i * 16;
            for (int j = 0; j < 8; ++j) {
                const uint8_t x0 = in[2 * j + 0];
                const uint8_t x1 = in[2 * j + 1];
                out[j]     = uint8_t((x0 & 0x0f) | ((x1 & 0x0f) << 4));
                out[j + 8] = uint8_t(((x0 & 0xf0) >> 4) | (x1 & 0xf0));
            }
        }
    }
};

struct cpu_compute_stage {
    std::vector<float> a;
    std::vector<float> b;
    int rounds = 0;

    cpu_compute_stage(size_t bytes, int r) : rounds(r) {
        const size_t n = std::max<size_t>(1024, bytes / sizeof(float));
        a.resize(n);
        b.resize(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = float(i % 251) * 0.001f + 1.0f;
            b[i] = float(i % 127) * 0.002f + 0.5f;
        }
    }
    void operator()() {
        float acc = 0.0f;
        for (int r = 0; r < rounds; ++r) {
            for (size_t i = 0; i < a.size(); ++i) {
                a[i] = std::fma(a[i], 1.0001f, b[(i + size_t(r) * 17u) % b.size()]);
                acc += a[i] * 0.000001f;
            }
        }
        volatile float sink = acc;
        (void) sink;
    }
};

struct gpu_compute_stage {
    cl_command_queue q = nullptr;
    cl_buffer a;
    cl_buffer b;
    cl_buffer c;
    cl_kernel_wrap kernel;
    uint32_t n = 0;
    uint32_t rounds = 0;

    gpu_compute_stage(cl_env & e, size_t bytes, int r) :
        q(e.compute_q),
        a(e.context, std::max<size_t>(bytes, 4096), CL_MEM_READ_WRITE),
        b(e.context, std::max<size_t>(bytes, 4096), CL_MEM_READ_WRITE),
        c(e.context, std::max<size_t>(bytes, 4096), CL_MEM_READ_WRITE),
        kernel(e.compute, "elastic_gpu_compute"),
        n(uint32_t(a.size / sizeof(float))),
        rounds(uint32_t(r)) {
        std::vector<float> host(n, 1.0f);
        CL_CHECK(clEnqueueWriteBuffer(q, a.mem, CL_TRUE, 0, n * sizeof(float), host.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueWriteBuffer(q, b.mem, CL_TRUE, 0, n * sizeof(float), host.data(), 0, nullptr, nullptr));
        set_arg(kernel.k, 0, a.mem);
        set_arg(kernel.k, 1, b.mem);
        set_arg(kernel.k, 2, c.mem);
        set_scalar(kernel.k, 3, n);
        set_scalar(kernel.k, 4, rounds);
    }
    void operator()() {
        const size_t lws = 128;
        const size_t gws = align_up(n, lws);
        CL_CHECK(clEnqueueNDRangeKernel(q, kernel.k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr));
        CL_CHECK(clFinish(q));
    }
};

struct gpu_write_stage {
    cl_command_queue q = nullptr;
    std::vector<uint8_t> host;
    cl_buffer dst;

    gpu_write_stage(cl_env & e, size_t bytes) :
        q(e.xform_q),
        host(std::max<size_t>(bytes, 4096)),
        dst(e.context, host.size(), CL_MEM_READ_WRITE) {
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = uint8_t(i * 17u + 3u);
        }
    }

    void operator()() {
        CL_CHECK(clEnqueueWriteBuffer(q, dst.mem, CL_FALSE, 0, host.size(), host.data(), 0, nullptr, nullptr));
        CL_CHECK(clFinish(q));
    }
};

struct gpu_write_two_stage {
    cl_command_queue q = nullptr;
    std::vector<uint8_t> host_q;
    std::vector<uint8_t> host_d;
    cl_buffer qbuf;
    cl_buffer dbuf;

    gpu_write_two_stage(cl_env & e, int k, int m) :
        q(e.xform_q),
        host_q(size_t(k) * size_t(m) / 32 * 16),
        host_d(size_t(k) * size_t(m) / 32 * 2),
        qbuf(e.context, host_q.size(), CL_MEM_READ_WRITE),
        dbuf(e.context, host_d.size(), CL_MEM_READ_WRITE) {
        for (size_t i = 0; i < host_q.size(); ++i) {
            host_q[i] = uint8_t(i * 17u + 3u);
        }
        for (size_t i = 0; i < host_d.size(); ++i) {
            host_d[i] = uint8_t(i * 29u + 5u);
        }
    }

    void operator()() {
        CL_CHECK(clEnqueueWriteBuffer(q, qbuf.mem, CL_FALSE, 0, host_q.size(), host_q.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueWriteBuffer(q, dbuf.mem, CL_FALSE, 0, host_d.size(), host_d.data(), 0, nullptr, nullptr));
        CL_CHECK(clFinish(q));
    }
};

struct gpu_write_raw_stage {
    cl_command_queue q = nullptr;
    std::vector<uint8_t> host;
    cl_buffer dst;

    gpu_write_raw_stage(cl_env & e, int k, int m) :
        q(e.xform_q),
        host(size_t(k) * size_t(m) / 32 * 18),
        dst(e.context, host.size(), CL_MEM_READ_WRITE) {
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = uint8_t(i * 13u + 11u);
        }
    }

    void operator()() {
        CL_CHECK(clEnqueueWriteBuffer(q, dst.mem, CL_FALSE, 0, host.size(), host.data(), 0, nullptr, nullptr));
        CL_CHECK(clFinish(q));
    }
};

struct gpu_xform_stage {
    cl_command_queue q = nullptr;
    int k_dim = 0;
    int m_dim = 0;
    size_t blocks = 0;
    std::vector<uint8_t> host;
    cl_buffer src;
    cl_buffer qbuf;
    cl_buffer dbuf;
    cl_buffer tmp_q;
    cl_buffer tmp_d;
    cl_kernel_wrap convert;
    cl_kernel_wrap tr_q;
    cl_kernel_wrap tr_d;

    gpu_xform_stage(cl_env & e, int k, int m) :
        q(e.xform_q),
        k_dim(k),
        m_dim(m),
        blocks(size_t(k) * size_t(m) / 32),
        host(blocks * 18),
        src(e.context, host.size(), CL_MEM_READ_WRITE),
        qbuf(e.context, blocks * 16, CL_MEM_READ_WRITE),
        dbuf(e.context, blocks * 2, CL_MEM_READ_WRITE),
        tmp_q(e.context, blocks * 16, CL_MEM_READ_WRITE),
        tmp_d(e.context, blocks * 2, CL_MEM_READ_WRITE),
        convert(e.cvt, "kernel_convert_block_q4_0_noshuffle"),
        tr_q(e.transpose, "kernel_transpose_16_buf_tiled"),
        tr_d(e.transpose, "kernel_transpose_16_buf_tiled") {
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = uint8_t(i * 31u + 7u);
        }
        set_arg(convert.k, 0, src.mem);
        set_arg(convert.k, 1, qbuf.mem);
        set_arg(convert.k, 2, dbuf.mem);
        set_arg(tr_q.k, 0, qbuf.mem);
        set_arg(tr_q.k, 1, tmp_q.mem);
        const int q_stride = k_dim / 4;
        set_scalar(tr_q.k, 2, q_stride);
        set_scalar(tr_q.k, 3, m_dim);
        set_arg(tr_d.k, 0, dbuf.mem);
        set_arg(tr_d.k, 1, tmp_d.mem);
        const int d_stride = k_dim / 32;
        set_scalar(tr_d.k, 2, d_stride);
        set_scalar(tr_d.k, 3, m_dim);
    }
    void operator()() {
        CL_CHECK(clEnqueueWriteBuffer(q, src.mem, CL_FALSE, 0, host.size(), host.data(), 0, nullptr, nullptr));
        const size_t conv_lws[3] = { 64, 1, 1 };
        const size_t conv_gws[3] = { align_up(blocks, conv_lws[0]), 1, 1 };
        CL_CHECK(clEnqueueNDRangeKernel(q, convert.k, 1, nullptr, conv_gws, conv_lws, 0, nullptr, nullptr));

        const size_t tr_lws[3] = { 16, 16, 1 };
        const size_t q_gws[3] = { size_t(align_up(k_dim / 4, 16)), size_t(align_up(m_dim, 16)), 1 };
        const size_t d_gws[3] = { size_t(align_up(k_dim / 32, 16)), size_t(align_up(m_dim, 16)), 1 };
        CL_CHECK(clEnqueueNDRangeKernel(q, tr_q.k, 3, nullptr, q_gws, tr_lws, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueCopyBuffer(q, tmp_q.mem, qbuf.mem, 0, 0, qbuf.size, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueNDRangeKernel(q, tr_d.k, 3, nullptr, d_gws, tr_lws, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueCopyBuffer(q, tmp_d.mem, dbuf.mem, 0, 0, dbuf.size, 0, nullptr, nullptr));
        CL_CHECK(clFinish(q));
    }
};

struct gpu_xform_kernels_stage {
    cl_command_queue q = nullptr;
    int k_dim = 0;
    int m_dim = 0;
    size_t blocks = 0;
    std::vector<uint8_t> host;
    cl_buffer src;
    cl_buffer qbuf;
    cl_buffer dbuf;
    cl_buffer tmp_q;
    cl_buffer tmp_d;
    cl_kernel_wrap convert;
    cl_kernel_wrap tr_q;
    cl_kernel_wrap tr_d;

    gpu_xform_kernels_stage(cl_env & e, int k, int m) :
        q(e.xform_q),
        k_dim(k),
        m_dim(m),
        blocks(size_t(k) * size_t(m) / 32),
        host(blocks * 18),
        src(e.context, host.size(), CL_MEM_READ_WRITE),
        qbuf(e.context, blocks * 16, CL_MEM_READ_WRITE),
        dbuf(e.context, blocks * 2, CL_MEM_READ_WRITE),
        tmp_q(e.context, blocks * 16, CL_MEM_READ_WRITE),
        tmp_d(e.context, blocks * 2, CL_MEM_READ_WRITE),
        convert(e.cvt, "kernel_convert_block_q4_0_noshuffle"),
        tr_q(e.transpose, "kernel_transpose_16_buf_tiled"),
        tr_d(e.transpose, "kernel_transpose_16_buf_tiled") {
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = uint8_t(i * 31u + 7u);
        }
        CL_CHECK(clEnqueueWriteBuffer(q, src.mem, CL_TRUE, 0, host.size(), host.data(), 0, nullptr, nullptr));
        set_arg(convert.k, 0, src.mem);
        set_arg(convert.k, 1, qbuf.mem);
        set_arg(convert.k, 2, dbuf.mem);
        set_arg(tr_q.k, 0, qbuf.mem);
        set_arg(tr_q.k, 1, tmp_q.mem);
        const int q_stride = k_dim / 4;
        set_scalar(tr_q.k, 2, q_stride);
        set_scalar(tr_q.k, 3, m_dim);
        set_arg(tr_d.k, 0, dbuf.mem);
        set_arg(tr_d.k, 1, tmp_d.mem);
        const int d_stride = k_dim / 32;
        set_scalar(tr_d.k, 2, d_stride);
        set_scalar(tr_d.k, 3, m_dim);
    }

    void operator()() {
        const size_t conv_lws[3] = { 64, 1, 1 };
        const size_t conv_gws[3] = { align_up(blocks, conv_lws[0]), 1, 1 };
        CL_CHECK(clEnqueueNDRangeKernel(q, convert.k, 1, nullptr, conv_gws, conv_lws, 0, nullptr, nullptr));

        const size_t tr_lws[3] = { 16, 16, 1 };
        const size_t q_gws[3] = { size_t(align_up(k_dim / 4, 16)), size_t(align_up(m_dim, 16)), 1 };
        const size_t d_gws[3] = { size_t(align_up(k_dim / 32, 16)), size_t(align_up(m_dim, 16)), 1 };
        CL_CHECK(clEnqueueNDRangeKernel(q, tr_q.k, 3, nullptr, q_gws, tr_lws, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueNDRangeKernel(q, tr_d.k, 3, nullptr, d_gws, tr_lws, 0, nullptr, nullptr));
        CL_CHECK(clFinish(q));
    }
};

struct gpu_convert_only_stage {
    cl_command_queue q = nullptr;
    size_t blocks = 0;
    std::vector<uint8_t> host;
    cl_buffer src;
    cl_buffer qbuf;
    cl_buffer dbuf;
    cl_kernel_wrap convert;

    gpu_convert_only_stage(cl_env & e, int k, int m) :
        q(e.xform_q),
        blocks(size_t(k) * size_t(m) / 32),
        host(blocks * 18),
        src(e.context, host.size(), CL_MEM_READ_WRITE),
        qbuf(e.context, blocks * 16, CL_MEM_READ_WRITE),
        dbuf(e.context, blocks * 2, CL_MEM_READ_WRITE),
        convert(e.cvt, "kernel_convert_block_q4_0_noshuffle") {
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = uint8_t(i * 31u + 7u);
        }
        CL_CHECK(clEnqueueWriteBuffer(q, src.mem, CL_TRUE, 0, host.size(), host.data(), 0, nullptr, nullptr));
        set_arg(convert.k, 0, src.mem);
        set_arg(convert.k, 1, qbuf.mem);
        set_arg(convert.k, 2, dbuf.mem);
    }

    void operator()() {
        const size_t conv_lws[3] = { 64, 1, 1 };
        const size_t conv_gws[3] = { align_up(blocks, conv_lws[0]), 1, 1 };
        CL_CHECK(clEnqueueNDRangeKernel(q, convert.k, 1, nullptr, conv_gws, conv_lws, 0, nullptr, nullptr));
        CL_CHECK(clFinish(q));
    }
};

struct gpu_transpose_only_stage {
    cl_command_queue q = nullptr;
    int k_dim = 0;
    int m_dim = 0;
    size_t blocks = 0;
    std::vector<uint8_t> host_q;
    std::vector<uint8_t> host_d;
    cl_buffer qbuf;
    cl_buffer dbuf;
    cl_buffer tmp_q;
    cl_buffer tmp_d;
    cl_kernel_wrap tr_q;
    cl_kernel_wrap tr_d;

    gpu_transpose_only_stage(cl_env & e, int k, int m) :
        q(e.xform_q),
        k_dim(k),
        m_dim(m),
        blocks(size_t(k) * size_t(m) / 32),
        host_q(blocks * 16),
        host_d(blocks * 2),
        qbuf(e.context, host_q.size(), CL_MEM_READ_WRITE),
        dbuf(e.context, host_d.size(), CL_MEM_READ_WRITE),
        tmp_q(e.context, host_q.size(), CL_MEM_READ_WRITE),
        tmp_d(e.context, host_d.size(), CL_MEM_READ_WRITE),
        tr_q(e.transpose, "kernel_transpose_16_buf_tiled"),
        tr_d(e.transpose, "kernel_transpose_16_buf_tiled") {
        for (size_t i = 0; i < host_q.size(); ++i) {
            host_q[i] = uint8_t(i * 17u + 3u);
        }
        for (size_t i = 0; i < host_d.size(); ++i) {
            host_d[i] = uint8_t(i * 29u + 5u);
        }
        CL_CHECK(clEnqueueWriteBuffer(q, qbuf.mem, CL_TRUE, 0, host_q.size(), host_q.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueWriteBuffer(q, dbuf.mem, CL_TRUE, 0, host_d.size(), host_d.data(), 0, nullptr, nullptr));
        set_arg(tr_q.k, 0, qbuf.mem);
        set_arg(tr_q.k, 1, tmp_q.mem);
        const int q_stride = k_dim / 4;
        set_scalar(tr_q.k, 2, q_stride);
        set_scalar(tr_q.k, 3, m_dim);
        set_arg(tr_d.k, 0, dbuf.mem);
        set_arg(tr_d.k, 1, tmp_d.mem);
        const int d_stride = k_dim / 32;
        set_scalar(tr_d.k, 2, d_stride);
        set_scalar(tr_d.k, 3, m_dim);
    }

    void operator()() {
        const size_t tr_lws[3] = { 16, 16, 1 };
        const size_t q_gws[3] = { size_t(align_up(k_dim / 4, 16)), size_t(align_up(m_dim, 16)), 1 };
        const size_t d_gws[3] = { size_t(align_up(k_dim / 32, 16)), size_t(align_up(m_dim, 16)), 1 };
        CL_CHECK(clEnqueueNDRangeKernel(q, tr_q.k, 3, nullptr, q_gws, tr_lws, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueNDRangeKernel(q, tr_d.k, 3, nullptr, d_gws, tr_lws, 0, nullptr, nullptr));
        CL_CHECK(clFinish(q));
    }
};

struct pipeline_gpu_xform_stage {
    struct slot_buffers {
        cl_buffer qbuf;
        cl_buffer dbuf;
        cl_buffer tmp_q;
        cl_buffer tmp_d;
    };

    cl_command_queue q = nullptr;
    int k_dim = 0;
    int m_dim = 0;
    size_t blocks = 0;
    size_t q_bytes = 0;
    size_t d_bytes = 0;
    bool kernels_only = false;
    std::vector<slot_buffers> bufs;
    std::vector<uint8_t> preload_q;
    std::vector<uint8_t> preload_d;
    cl_kernel_wrap tr_q;
    cl_kernel_wrap tr_d;

    pipeline_gpu_xform_stage(cl_env & e, int k, int m, int slots, bool kernels_only_) :
        q(e.xform_q),
        k_dim(k),
        m_dim(m),
        blocks(size_t(k) * size_t(m) / 32),
        q_bytes(blocks * 16),
        d_bytes(blocks * 2),
        kernels_only(kernels_only_),
        tr_q(e.transpose, "kernel_transpose_16_buf_tiled"),
        tr_d(e.transpose, "kernel_transpose_16_buf_tiled") {
        if (kernels_only) {
            preload_q.resize(q_bytes);
            preload_d.resize(d_bytes);
            for (size_t j = 0; j < preload_q.size(); ++j) {
                preload_q[j] = uint8_t(j * 17u + 3u);
            }
            for (size_t j = 0; j < preload_d.size(); ++j) {
                preload_d[j] = uint8_t(j * 29u + 5u);
            }
        }
        bufs.reserve(slots);
        for (int i = 0; i < slots; ++i) {
            slot_buffers b;
            b.qbuf = cl_buffer(e.context, q_bytes, CL_MEM_READ_WRITE);
            b.dbuf = cl_buffer(e.context, d_bytes, CL_MEM_READ_WRITE);
            b.tmp_q = cl_buffer(e.context, q_bytes, CL_MEM_READ_WRITE);
            b.tmp_d = cl_buffer(e.context, d_bytes, CL_MEM_READ_WRITE);
            if (kernels_only) {
                CL_CHECK(clEnqueueWriteBuffer(q, b.qbuf.mem, CL_FALSE, 0, q_bytes, preload_q.data(), 0, nullptr, nullptr));
                CL_CHECK(clEnqueueWriteBuffer(q, b.dbuf.mem, CL_FALSE, 0, d_bytes, preload_d.data(), 0, nullptr, nullptr));
            }
            bufs.push_back(std::move(b));
        }
        if (kernels_only) {
            CL_CHECK(clFinish(q));
        }
    }

    void run(int slot, const uint8_t * host_q, const uint16_t * host_d) {
        slot_buffers & b = bufs[size_t(slot)];
        if (!kernels_only) {
            CL_CHECK(clEnqueueWriteBuffer(q, b.qbuf.mem, CL_FALSE, 0, q_bytes, host_q, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueWriteBuffer(q, b.dbuf.mem, CL_FALSE, 0, d_bytes, host_d, 0, nullptr, nullptr));
        } else {
            (void) host_q;
            (void) host_d;
        }

        set_arg(tr_q.k, 0, b.qbuf.mem);
        set_arg(tr_q.k, 1, b.tmp_q.mem);
        const int q_stride = k_dim / 4;
        set_scalar(tr_q.k, 2, q_stride);
        set_scalar(tr_q.k, 3, m_dim);
        set_arg(tr_d.k, 0, b.dbuf.mem);
        set_arg(tr_d.k, 1, b.tmp_d.mem);
        const int d_stride = k_dim / 32;
        set_scalar(tr_d.k, 2, d_stride);
        set_scalar(tr_d.k, 3, m_dim);

        const size_t tr_lws[3] = { 16, 16, 1 };
        const size_t q_gws[3] = { size_t(align_up(k_dim / 4, 16)), size_t(align_up(m_dim, 16)), 1 };
        const size_t d_gws[3] = { size_t(align_up(k_dim / 32, 16)), size_t(align_up(m_dim, 16)), 1 };
        CL_CHECK(clEnqueueNDRangeKernel(q, tr_q.k, 3, nullptr, q_gws, tr_lws, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueNDRangeKernel(q, tr_d.k, 3, nullptr, d_gws, tr_lws, 0, nullptr, nullptr));
        if (!kernels_only) {
            CL_CHECK(clEnqueueCopyBuffer(q, b.tmp_q.mem, b.qbuf.mem, 0, 0, q_bytes, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueCopyBuffer(q, b.tmp_d.mem, b.dbuf.mem, 0, 0, d_bytes, 0, nullptr, nullptr));
        }
        CL_CHECK(clFinish(q));
    }
};

struct bench_stage {
    std::string name;
    std::function<void()> fn;
    double single_ms = 0.0;
};

static double time_one(const std::function<void()> & fn) {
    const double t0 = now_ms();
    fn();
    return now_ms() - t0;
}

static stats bench_single(const std::function<void()> & fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        samples.push_back(time_one(fn));
    }
    return calc_stats(std::move(samples));
}

static double bench_overlap_once(const std::vector<bench_stage *> & stages) {
    barrier start_barrier(int(stages.size()) + 1);
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(stages.size());
    for (bench_stage * s : stages) {
        threads.emplace_back([&start_barrier, &failures, s] {
            try {
                start_barrier.wait();
                s->fn();
            } catch (const std::exception & ex) {
                std::fprintf(stderr, "stage %s failed: %s\n", s->name.c_str(), ex.what());
                failures.fetch_add(1);
            }
        });
    }
    const double t0 = now_ms();
    start_barrier.wait();
    for (std::thread & t : threads) {
        t.join();
    }
    if (failures.load() != 0) {
        throw std::runtime_error("overlap stage failed");
    }
    return now_ms() - t0;
}

static stats bench_overlap(const std::vector<bench_stage *> & stages, int warmup, int iters) {
    if (stages.empty()) {
        return {};
    }
    const int total_iters = warmup + iters;
    barrier start_barrier(int(stages.size()) + 1);
    barrier done_barrier(int(stages.size()) + 1);
    std::atomic<int> failures{0};
    std::mutex err_mtx;
    std::exception_ptr first_error;
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    threads.reserve(stages.size());
    for (bench_stage * s : stages) {
        threads.emplace_back([&, s] {
            for (int iter = 0; iter < total_iters; ++iter) {
                start_barrier.wait();
                if (stop.load()) {
                    done_barrier.wait();
                    continue;
                }
                try {
                    s->fn();
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(err_mtx);
                        if (!first_error) {
                            first_error = std::current_exception();
                        }
                    }
                    failures.fetch_add(1);
                    stop.store(true);
                }
                done_barrier.wait();
            }
        });
    }

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < total_iters; ++i) {
        const double t0 = now_ms();
        start_barrier.wait();
        done_barrier.wait();
        const double dt = now_ms() - t0;
        if (failures.load() != 0) {
            stop.store(true);
        }
        if (i >= warmup) {
            samples.push_back(dt);
        }
    }
    for (std::thread & t : threads) {
        t.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
    return calc_stats(std::move(samples));
}

static void print_single(const bench_stage & s, const stats & st, double mb) {
    const double bw = mb / (st.med / 1000.0);
    std::printf("single %-13s med=%8.3f ms avg=%8.3f ms min=%8.3f max=%8.3f  size=%7.1f MiB rate=%8.1f MiB/s\n",
        s.name.c_str(), st.med, st.avg, st.min, st.max, mb, bw);
}

static void print_overlap(const std::vector<bench_stage *> & stages, const stats & st) {
    double ideal = 0.0;
    std::string name;
    for (size_t i = 0; i < stages.size(); ++i) {
        if (i) {
            name += "+";
        }
        name += stages[i]->name;
        ideal = std::max(ideal, stages[i]->single_ms);
    }
    const double sum = std::accumulate(stages.begin(), stages.end(), 0.0, [](double a, const bench_stage * s) {
        return a + s->single_ms;
    });
    std::printf("overlap %-43s med=%8.3f ms ideal=%8.3f ms sum=%8.3f ms competition=%5.2f speedup_vs_serial=%5.2f\n",
        name.c_str(), st.med, ideal, sum, ideal > 0.0 ? st.med / ideal : 0.0, st.med > 0.0 ? sum / st.med : 0.0);
}

enum class pipe_state {
    empty,
    loading,
    loaded,
    cpuing,
    cpu_done,
    gpuing,
    gpu_done,
    computing,
};

struct pipe_slot {
    void * raw = nullptr;
    size_t raw_capacity = 0;
    std::vector<uint8_t> q;
    std::vector<uint16_t> d;
    pipe_state state = pipe_state::empty;

    pipe_slot(size_t raw_bytes_aligned, size_t blocks, size_t align) :
        raw_capacity(raw_bytes_aligned),
        q(blocks * 16),
        d(blocks) {
        if (posix_memalign(&raw, align, raw_capacity) != 0) {
            throw std::runtime_error("posix_memalign failed for pipeline slot");
        }
    }
    pipe_slot(const pipe_slot &) = delete;
    pipe_slot & operator=(const pipe_slot &) = delete;
    ~pipe_slot() {
        if (raw) free(raw);
    }
};

struct pipe_metric {
    double wait_ms = 0.0;
    double busy_ms = 0.0;
    int items = 0;
};

static int find_slot_with_state(const std::vector<std::unique_ptr<pipe_slot>> & slots, pipe_state state) {
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i]->state == state) {
            return int(i);
        }
    }
    return -1;
}

static bool any_slot_with_state(const std::vector<std::unique_ptr<pipe_slot>> & slots, pipe_state state) {
    return find_slot_with_state(slots, state) >= 0;
}

static void run_pipeline_bench(
        const params & p,
        disk_load_stage & disk,
        cl_env & cl,
        gpu_compute_stage & gpu_compute) {
    const size_t blocks = size_t(p.k) * size_t(p.m) / 32;
    const size_t raw_bytes = blocks * sizeof(cpu_xform_stage::q4blk);
    const size_t raw_aligned = align_up(raw_bytes, disk.align);

    std::vector<std::unique_ptr<pipe_slot>> slots;
    slots.reserve(size_t(p.pipeline_slots));
    for (int i = 0; i < p.pipeline_slots; ++i) {
        slots.emplace_back(new pipe_slot(raw_aligned, blocks, disk.align));
    }
    const bool gpu_kernels_only = p.pipeline_gpu_mode == "kernels-only";
    pipeline_gpu_xform_stage gpu_xform(cl, p.k, p.m, p.pipeline_slots, gpu_kernels_only);

    std::mutex mtx;
    std::condition_variable cv;
    bool failed = false;
    std::exception_ptr error;
    pipe_metric load_m;
    pipe_metric cpu_m;
    pipe_metric gpu_m;
    pipe_metric compute_m;
    barrier start_barrier(5);

    auto fail = [&](std::exception_ptr ep) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!failed) {
            failed = true;
            error = ep;
        }
        cv.notify_all();
    };

    auto worker = [&](const char * name,
                      pipe_state want,
                      pipe_state working,
                      pipe_state done,
                      pipe_metric & metric,
                      const std::function<void(int)> & fn) {
        try {
            start_barrier.wait();
            for (int item = 0; item < p.pipeline_items; ++item) {
                const double tw0 = now_ms();
                int slot = -1;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] {
                        return failed || any_slot_with_state(slots, want);
                    });
                    if (failed) {
                        return;
                    }
                    slot = find_slot_with_state(slots, want);
                    if (slot < 0) {
                        throw std::runtime_error(std::string("missing slot for stage ") + name);
                    }
                    slots[size_t(slot)]->state = working;
                }
                metric.wait_ms += now_ms() - tw0;

                const double tb0 = now_ms();
                fn(slot);
                metric.busy_ms += now_ms() - tb0;
                metric.items++;

                {
                    std::lock_guard<std::mutex> lock(mtx);
                    slots[size_t(slot)]->state = done;
                }
                cv.notify_all();
            }
        } catch (...) {
            fail(std::current_exception());
        }
    };

    auto load_fn = [&](int slot) {
        disk.read_into(slots[size_t(slot)]->raw, raw_bytes);
    };
    auto cpu_fn = [&](int slot) {
        pipe_slot & s = *slots[size_t(slot)];
        cpu_xform_stage::transform_raw(static_cast<const uint8_t *>(s.raw), blocks, s.q.data(), s.d.data());
    };
    auto gpu_fn = [&](int slot) {
        pipe_slot & s = *slots[size_t(slot)];
        gpu_xform.run(slot, s.q.data(), s.d.data());
    };
    auto compute_fn = [&](int) {
        gpu_compute();
    };

    std::thread load_t(worker, "disk_load", pipe_state::empty, pipe_state::loading, pipe_state::loaded, std::ref(load_m), load_fn);
    std::thread cpu_t(worker, "cpu_xform", pipe_state::loaded, pipe_state::cpuing, pipe_state::cpu_done, std::ref(cpu_m), cpu_fn);
    std::thread gpu_t(worker, "gpu_xform", pipe_state::cpu_done, pipe_state::gpuing, pipe_state::gpu_done, std::ref(gpu_m), gpu_fn);
    std::thread compute_t(worker, "gpu_compute", pipe_state::gpu_done, pipe_state::computing, pipe_state::empty, std::ref(compute_m), compute_fn);

    const double t0 = now_ms();
    start_barrier.wait();
    load_t.join();
    cpu_t.join();
    gpu_t.join();
    compute_t.join();
    const double wall = now_ms() - t0;
    if (error) {
        std::rethrow_exception(error);
    }

    const double steady_items = std::max(1, p.pipeline_items - p.pipeline_slots + 1);
    const double item_ms = wall / double(p.pipeline_items);
    const double serial_ms = load_m.busy_ms + cpu_m.busy_ms + gpu_m.busy_ms + compute_m.busy_ms;
    std::printf("\n");
    std::printf("pipeline disk_load->cpu_xform->gpu_xform->gpu_compute mode=%s items=%d slots=%d raw=%.1fMiB q=%.1fMiB d=%.1fMiB\n",
        p.pipeline_gpu_mode.c_str(), p.pipeline_items, p.pipeline_slots, double(raw_bytes) / double(MiB),
        double(blocks * 16) / double(MiB), double(blocks * 2) / double(MiB));
    std::printf("pipeline wall=%8.3f ms item_avg=%8.3f ms serial_est=%8.3f ms speedup_vs_serial=%5.2f steady_items/s=%8.2f\n",
        wall, item_ms, serial_ms, wall > 0.0 ? serial_ms / wall : 0.0, double(p.pipeline_items) * 1000.0 / wall);
    auto print_metric = [](const char * name, const pipe_metric & m) {
        std::printf("  %-11s items=%4d busy=%8.3f ms wait=%8.3f ms busy/item=%8.3f ms wait/item=%8.3f ms\n",
            name, m.items, m.busy_ms, m.wait_ms,
            m.items > 0 ? m.busy_ms / double(m.items) : 0.0,
            m.items > 0 ? m.wait_ms / double(m.items) : 0.0);
    };
    print_metric("disk_load", load_m);
    print_metric("cpu_xform", cpu_m);
    print_metric("gpu_xform", gpu_m);
    print_metric("gpu_compute", compute_m);
    (void) steady_items;
}

int main(int argc, char ** argv) {
    try {
        const params p = parse(argc, argv);
        cl_env cl = init_opencl(p);

        std::unique_ptr<disk_load_stage> disk;
        std::unique_ptr<disk_async_prefetch_stage> disk_async;
        if (!p.file.empty()) {
            size_t disk_bytes = size_t(p.load_mb) * MiB;
            if (p.pipeline) {
                const size_t pipe_raw_bytes = size_t(p.k) * size_t(p.m) / 32 * sizeof(cpu_xform_stage::q4blk);
                disk_bytes = std::max(disk_bytes, pipe_raw_bytes + 4096);
            }
            disk.reset(new disk_load_stage(p.file, disk_bytes));
            disk_async.reset(new disk_async_prefetch_stage(p.file, disk_bytes, p.prefetch_slots));
        }
        cpu_mem_load_stage cpu_mem_load(size_t(p.load_mb) * MiB);
        cpu_xform_stage cpu_xform(size_t(p.cpu_xform_mb) * MiB);
        cpu_compute_stage cpu_compute(size_t(p.cpu_compute_mb) * MiB, p.cpu_rounds);
        gpu_write_stage gpu_write(cl, size_t(p.gpu_compute_mb) * MiB);
        gpu_write_raw_stage gpu_write_raw(cl, p.k, p.m);
        gpu_write_two_stage gpu_write_two(cl, p.k, p.m);
        gpu_xform_stage gpu_xform(cl, p.k, p.m);
        gpu_xform_kernels_stage gpu_xform_kernels(cl, p.k, p.m);
        gpu_convert_only_stage gpu_convert_only(cl, p.k, p.m);
        gpu_transpose_only_stage gpu_transpose_only(cl, p.k, p.m);
        gpu_compute_stage gpu_compute(cl, size_t(p.gpu_compute_mb) * MiB, p.gpu_rounds);

        std::vector<bench_stage> stages;
        if (disk) {
            stages.push_back({"disk_load", [&] { (*disk)(); }, 0.0});
            stages.push_back({"disk_async_prefetch", [&] { (*disk_async)(); }, 0.0});
        }
        stages.push_back({"cpu_mem_load", [&] { cpu_mem_load(); }, 0.0});
        stages.push_back({"cpu_xform", [&] { cpu_xform(); }, 0.0});
        stages.push_back({"gpu_write", [&] { gpu_write(); }, 0.0});
        stages.push_back({"gpu_write_raw", [&] { gpu_write_raw(); }, 0.0});
        stages.push_back({"gpu_write_two", [&] { gpu_write_two(); }, 0.0});
        stages.push_back({"gpu_xform", [&] { gpu_xform(); }, 0.0});
        stages.push_back({"gpu_xform_kernels", [&] { gpu_xform_kernels(); }, 0.0});
        stages.push_back({"gpu_convert_only", [&] { gpu_convert_only(); }, 0.0});
        stages.push_back({"gpu_transpose_only", [&] { gpu_transpose_only(); }, 0.0});
        stages.push_back({"cpu_compute", [&] { cpu_compute(); }, 0.0});
        stages.push_back({"gpu_compute", [&] { gpu_compute(); }, 0.0});

        std::printf("config iters=%d warmup=%d load=%dMiB cpu_xform=%dMiB cpu_compute=%dMiB gpu_compute=%dMiB gpu_xform=q4_0 K=%d M=%d src=%.1fMiB cpu_rounds=%d gpu_rounds=%d\n",
            p.iters, p.warmup, p.load_mb, p.cpu_xform_mb, p.cpu_compute_mb, p.gpu_compute_mb,
            p.k, p.m, double(size_t(p.k) * size_t(p.m) / 32 * 18) / double(MiB), p.cpu_rounds, p.gpu_rounds);

        for (bench_stage & s : stages) {
            const stats st = bench_single(s.fn, p.warmup, p.iters);
            s.single_ms = st.med;
            double mb = 0.0;
            if (s.name == "disk_load") mb = p.load_mb;
            if (s.name == "disk_async_prefetch") mb = p.load_mb;
            if (s.name == "cpu_mem_load") mb = p.load_mb;
            if (s.name == "cpu_xform") mb = p.cpu_xform_mb;
            if (s.name == "gpu_write") mb = p.gpu_compute_mb;
            if (s.name == "gpu_write_raw") mb = double(size_t(p.k) * size_t(p.m) / 32 * 18) / double(MiB);
            if (s.name == "gpu_write_two") mb = double(size_t(p.k) * size_t(p.m) / 32 * 18) / double(MiB);
            if (s.name == "gpu_xform") mb = double(size_t(p.k) * size_t(p.m) / 32 * 18) / double(MiB);
            if (s.name == "gpu_xform_kernels") mb = double(size_t(p.k) * size_t(p.m) / 32 * 18) / double(MiB);
            if (s.name == "gpu_convert_only") mb = double(size_t(p.k) * size_t(p.m) / 32 * 18) / double(MiB);
            if (s.name == "gpu_transpose_only") mb = double(size_t(p.k) * size_t(p.m) / 32 * 18) / double(MiB);
            if (s.name == "cpu_compute") mb = p.cpu_compute_mb;
            if (s.name == "gpu_compute") mb = p.gpu_compute_mb;
            print_single(s, st, mb);
        }

        if (!p.pipeline_only) {
        auto find_stage = [&](const char * name) -> bench_stage * {
            for (bench_stage & s : stages) {
                if (s.name == name) {
                    return &s;
                }
            }
            return nullptr;
        };
        auto run_pair = [&](const char * a, const char * b) {
            bench_stage * sa = find_stage(a);
            bench_stage * sb = find_stage(b);
            if (!sa || !sb) {
                return;
            }
            print_overlap({sa, sb}, bench_overlap({sa, sb}, p.warmup, p.iters));
        };

        std::printf("\n");
        run_pair("disk_load", "cpu_xform");
        run_pair("disk_async_prefetch", "cpu_xform");
        run_pair("cpu_mem_load", "cpu_xform");
        run_pair("disk_load", "gpu_xform");
        run_pair("disk_async_prefetch", "gpu_xform");
        run_pair("cpu_mem_load", "gpu_xform");
        run_pair("disk_load", "cpu_compute");
        run_pair("disk_async_prefetch", "cpu_compute");
        run_pair("cpu_mem_load", "cpu_compute");
        run_pair("disk_load", "gpu_compute");
        run_pair("disk_async_prefetch", "gpu_compute");
        run_pair("cpu_mem_load", "gpu_compute");
        run_pair("cpu_xform", "gpu_xform");
        run_pair("cpu_xform", "cpu_compute");
        run_pair("cpu_xform", "gpu_compute");
        run_pair("gpu_write", "cpu_compute");
        run_pair("gpu_write_raw", "cpu_compute");
        run_pair("gpu_write_two", "cpu_compute");
        run_pair("gpu_xform", "cpu_compute");
        run_pair("gpu_xform_kernels", "cpu_compute");
        run_pair("gpu_convert_only", "cpu_compute");
        run_pair("gpu_transpose_only", "cpu_compute");
        run_pair("gpu_write", "gpu_compute");
        run_pair("gpu_write_raw", "gpu_compute");
        run_pair("gpu_write_two", "gpu_compute");
        run_pair("gpu_xform", "gpu_compute");
        run_pair("gpu_xform_kernels", "gpu_compute");
        run_pair("gpu_convert_only", "gpu_compute");
        run_pair("gpu_transpose_only", "gpu_compute");
        run_pair("cpu_compute", "gpu_compute");

        bench_stage * dl = find_stage("disk_load");
        bench_stage * cx = find_stage("cpu_xform");
        bench_stage * gx = find_stage("gpu_xform");
        bench_stage * gc = find_stage("gpu_compute");
        if (dl && cx && gx && gc) {
            print_overlap({dl, cx, gx, gc}, bench_overlap({dl, cx, gx, gc}, p.warmup, p.iters));
        }
        }

        if (p.pipeline) {
            if (!disk) {
                throw std::runtime_error("--pipeline requires --file for disk_load");
            }
            run_pipeline_bench(p, *disk, cl, gpu_compute);
        }

        return 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
