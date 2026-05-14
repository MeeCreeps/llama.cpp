// tests/elastic/probe_cl_release.cpp
//
// 验证 clReleaseMemObject 是否真的把 DRAM 还给系统（对应规范 §10 的 open
// issue + §6 "验证 evict 真的释放了 DRAM" 段落）。
//
// 流程：
//   1. 记初始 VmRSS（rss0）
//   2. 分配 N 个 size_mb 大小的 cl_mem，每个都 clEnqueueWriteBuffer 强制把
//      数据写过去 + clFinish，确保 backing store 真的落地 → 记 rss_alloc
//   3. 全部 clReleaseMemObject + clFinish → 记 rss_free
//   4. 打表对比 rss_alloc - rss0 是否 ~= N*size_mb（说明 alloc 真占了 DRAM）
//      以及 rss_free - rss0 是否 ~= 0（说明 release 真把 DRAM 还了）
//
// 输出是 CSV，便于和 Android 真机上同一份程序的输出直接对比。
//
// 主要目标平台是 Android（Adreno / Mali，统一 DRAM）。桌面 OpenCL ICD
// （PoCL / Intel / NVIDIA）上的结论不能外推到 mobile 驱动 —— mobile GPU
// 驱动可能把"释放"的 buffer 留在隐藏 pool 里。
//
// 编译：scripts/elastic/build_android.sh 用 NDK 交叉编译 + adb push 到真机
// 运行。CMakeLists 用 find_package(OpenCL QUIET) 守护：本机装了 OpenCL ICD
// 也能编出来对照看（但结论以真机为准）。
//
// 用法：
//   ./probe_cl_release [--size-mb 256] [--iters 8] [--use-host-ptr]
//                      [--mode interleaved|batched]
//
// 退出码：0 = 跑完（不代表 "DRAM 真释放"，结论看输出表）；非 0 = OpenCL 失败

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// 读 /proc/self/status 里的 VmRSS（单位 kB），失败返回 -1
long read_vmrss_kb() {
    std::ifstream f("/proc/self/status");
    if (!f.is_open()) return -1;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long v = -1;
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &v);
            return v;
        }
    }
    return -1;
}

const char *cl_err_str(cl_int e) {
    switch (e) {
        case CL_SUCCESS:                       return "CL_SUCCESS";
        case CL_OUT_OF_RESOURCES:              return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:            return "CL_OUT_OF_HOST_MEMORY";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_INVALID_VALUE:                 return "CL_INVALID_VALUE";
        case CL_INVALID_BUFFER_SIZE:           return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_HOST_PTR:              return "CL_INVALID_HOST_PTR";
        case CL_INVALID_CONTEXT:               return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE:         return "CL_INVALID_COMMAND_QUEUE";
        default:                               return "(unknown)";
    }
}

#define CL_CHECK(expr) do { cl_int _e = (expr); \
    if (_e != CL_SUCCESS) { \
        std::fprintf(stderr, "OpenCL 失败 @ %s:%d: %s (%d)\n", \
            __FILE__, __LINE__, cl_err_str(_e), _e); \
        std::exit(2); \
    } } while (0)

struct probe_opts {
    size_t size_mb     = 256;
    int    iters       = 8;
    bool   use_host_ptr = false;     // CL_MEM_ALLOC_HOST_PTR vs CL_MEM_READ_ONLY
    bool   mode_batched = true;      // batched: 全部 alloc 再全部 free
                                     // interleaved: alloc/free 交替
};

probe_opts parse_args(int argc, char **argv) {
    probe_opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s 需要参数\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if      (a == "--size-mb")      o.size_mb = static_cast<size_t>(std::atoll(next("--size-mb")));
        else if (a == "--iters")        o.iters   = std::atoi(next("--iters"));
        else if (a == "--use-host-ptr") o.use_host_ptr = true;
        else if (a == "--mode") {
            std::string m = next("--mode");
            o.mode_batched = (m == "batched");
        }
        else if (a == "-h" || a == "--help") {
            std::printf("用法：%s [--size-mb N] [--iters N] [--use-host-ptr] [--mode batched|interleaved]\n", argv[0]);
            std::exit(0);
        }
        else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            std::exit(1);
        }
    }
    return o;
}

}  // namespace

int main(int argc, char **argv) {
    probe_opts opts = parse_args(argc, argv);
    const size_t size_bytes = opts.size_mb * 1024 * 1024;

    // 选第一个 platform 的第一个 device
    cl_platform_id platform;
    cl_uint n_pf = 0;
    CL_CHECK(clGetPlatformIDs(1, &platform, &n_pf));
    if (n_pf == 0) {
        std::fprintf(stderr, "找不到 OpenCL platform\n");
        return 3;
    }
    cl_device_id device;
    cl_uint n_dev = 0;
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &n_dev));
    if (n_dev == 0) {
        std::fprintf(stderr, "platform 上没有 device\n");
        return 3;
    }

    char dev_name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    std::fprintf(stderr, "device: %s\n", dev_name);

    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, nullptr, &err);
    CL_CHECK(err);

    // 准备 host 端的填充数据
    std::vector<unsigned char> pattern(size_bytes);
    for (size_t i = 0; i < size_bytes; ++i) pattern[i] = static_cast<unsigned char>(i & 0xff);

    const long rss0_kb = read_vmrss_kb();
    std::printf("phase,iter,vmrss_mb,delta_mb,n_resident\n");
    std::printf("baseline,-1,%.2f,0.00,0\n", rss0_kb / 1024.0);

    std::vector<cl_mem> bufs;
    bufs.reserve(opts.iters);

    auto rss_now_delta = [&]() -> double {
        long r = read_vmrss_kb();
        return r < 0 ? 0.0 : (r - rss0_kb) / 1024.0;
    };

    const cl_mem_flags flags =
        opts.use_host_ptr ? CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR
                          : CL_MEM_READ_ONLY;

    if (opts.mode_batched) {
        // 全部分配 + 全部写 + 全部释放
        for (int i = 0; i < opts.iters; ++i) {
            cl_mem b = clCreateBuffer(ctx, flags, size_bytes, nullptr, &err);
            CL_CHECK(err);
            CL_CHECK(clEnqueueWriteBuffer(queue, b, CL_FALSE, 0, size_bytes,
                                          pattern.data(), 0, nullptr, nullptr));
            bufs.push_back(b);
        }
        CL_CHECK(clFinish(queue));
        std::printf("alloc_all,%d,%.2f,%.2f,%d\n",
            opts.iters - 1, rss_now_delta() + rss0_kb / 1024.0,
            rss_now_delta(), opts.iters);

        for (size_t i = 0; i < bufs.size(); ++i) {
            CL_CHECK(clReleaseMemObject(bufs[i]));
        }
        CL_CHECK(clFinish(queue));
        bufs.clear();
        std::printf("free_all,%d,%.2f,%.2f,0\n",
            opts.iters - 1, rss_now_delta() + rss0_kb / 1024.0, rss_now_delta());
    } else {
        // 交替：alloc → 写 → finish → 量 → release → finish → 量
        for (int i = 0; i < opts.iters; ++i) {
            cl_mem b = clCreateBuffer(ctx, flags, size_bytes, nullptr, &err);
            CL_CHECK(err);
            CL_CHECK(clEnqueueWriteBuffer(queue, b, CL_FALSE, 0, size_bytes,
                                          pattern.data(), 0, nullptr, nullptr));
            CL_CHECK(clFinish(queue));
            std::printf("alloc,%d,%.2f,%.2f,1\n",
                i, rss_now_delta() + rss0_kb / 1024.0, rss_now_delta());

            CL_CHECK(clReleaseMemObject(b));
            CL_CHECK(clFinish(queue));
            std::printf("free,%d,%.2f,%.2f,0\n",
                i, rss_now_delta() + rss0_kb / 1024.0, rss_now_delta());
        }
    }

    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    const double final_delta = rss_now_delta();
    const double expected_alloc_mb = static_cast<double>(opts.iters * opts.size_mb);
    std::fprintf(stderr,
        "总结：iters=%d size_mb=%zu 期望 alloc 后增 ~%.0f MB；最终 delta=%.2f MB "
        "(use_host_ptr=%d mode=%s)\n",
        opts.iters, opts.size_mb, expected_alloc_mb, final_delta,
        opts.use_host_ptr ? 1 : 0, opts.mode_batched ? "batched" : "interleaved");

    return 0;
}
