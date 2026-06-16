// tests/elastic/sync_boundary_bench.cpp
//
// Mobile OpenCL synchronization boundary microbenchmark.
//
// This measures the fixed cost of synchronization patterns that show up in the
// elastic plan runtime:
//   - clFinish on a compute queue
//   - waiting for one event
//   - xfer queue -> compute queue event dependency
//   - blocking vs non-blocking host transfer
//   - CL_MEM_USE_HOST_PTR visibility
//   - fine-grain SVM flag visibility, when the device supports it
//   - GPU writes a host-visible flag, CPU polls it, then falls back if needed

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct opts {
    int    iters       = 200;
    size_t size_mb     = 16;
    int    poll_us     = 2000;
    int    poll_sleep_us = 0;
    int    sleep_us    = 0;
    bool   gpu_poll    = false;
};

const char * cl_err_str(cl_int e) {
    switch (e) {
        case CL_SUCCESS:                         return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                return "CL_DEVICE_NOT_FOUND";
        case CL_COMPILER_NOT_AVAILABLE:          return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";
        case CL_INVALID_VALUE:                   return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE:                  return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                 return "CL_INVALID_CONTEXT";
        case CL_INVALID_MEM_OBJECT:              return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_COMMAND_QUEUE:           return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_KERNEL:                  return "CL_INVALID_KERNEL";
        case CL_INVALID_KERNEL_ARGS:             return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_GROUP_SIZE:         return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_OPERATION:               return "CL_INVALID_OPERATION";
        case CL_BUILD_PROGRAM_FAILURE:           return "CL_BUILD_PROGRAM_FAILURE";
        default:                                 return "(unknown)";
    }
}

#define CL_CHECK(expr) do { \
    cl_int _err = (expr); \
    if (_err != CL_SUCCESS) { \
        std::fprintf(stderr, "OpenCL failure @ %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cl_err_str(_err), _err); \
        std::exit(2); \
    } \
} while (0)

double us_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::micro>(clock_type::now() - t0).count();
}

struct summary {
    std::string name;
    std::vector<double> samples_us;
    int failures = 0;

    explicit summary(const char * n) : name(n) {}

    void add(double us) {
        samples_us.push_back(us);
    }
};

void print_summary(const summary & s) {
    if (s.samples_us.empty()) {
        std::printf("%s,0,0,0,0,0,%d\n", s.name.c_str(), s.failures);
        return;
    }
    std::vector<double> v = s.samples_us;
    std::sort(v.begin(), v.end());
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    auto pct = [&](double p) {
        const size_t idx = std::min(v.size() - 1, static_cast<size_t>(p * (v.size() - 1)));
        return v[idx];
    };
    std::printf("%s,%zu,%.3f,%.3f,%.3f,%.3f,%d\n",
            s.name.c_str(), v.size(), sum / v.size(), pct(0.50), pct(0.95), v.back(), s.failures);
}

opts parse_args(int argc, char ** argv) {
    opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--iters") {
            o.iters = std::atoi(next("--iters"));
        } else if (a == "--size-mb") {
            o.size_mb = static_cast<size_t>(std::atoll(next("--size-mb")));
        } else if (a == "--poll-us") {
            o.poll_us = std::atoi(next("--poll-us"));
        } else if (a == "--poll-sleep-us") {
            o.poll_sleep_us = std::atoi(next("--poll-sleep-us"));
        } else if (a == "--sleep-us") {
            o.sleep_us = std::atoi(next("--sleep-us"));
        } else if (a == "--gpu-poll") {
            o.gpu_poll = true;
        } else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [--iters N] [--size-mb N] [--poll-us N] [--poll-sleep-us N] [--sleep-us N] [--gpu-poll]\n", argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            std::exit(1);
        }
    }
    if (o.iters <= 0) {
        o.iters = 1;
    }
    if (o.size_mb == 0) {
        o.size_mb = 1;
    }
    return o;
}

const char * kernel_src = R"CLC(
__kernel void write_u32(__global uint * dst, uint value, uint n) {
    uint gid = get_global_id(0);
    if (gid < n) {
        dst[gid] = value + gid;
    }
}

__kernel void copy_u32(__global const uint * src, __global uint * dst, uint n) {
    uint gid = get_global_id(0);
    if (gid < n) {
        dst[gid] = src[gid];
    }
}

__kernel void signal_flag(__global volatile int * flag, int value) {
    flag[0] = value;
}

__kernel void signal_flag_many(__global volatile int * flag, int value, uint n) {
    uint gid = get_global_id(0);
    if (gid < n) {
        flag[gid] = value + gid;
    }
}

__kernel void wait_flag_bounded(__global volatile int * flag,
                                __global int * result,
                                int expected,
                                uint max_spins) {
    uint i = 0;
    while (i < max_spins && flag[0] < expected) {
        ++i;
    }
    result[0] = flag[0] >= expected ? 1 : 0;
}

__kernel void validate_after_flag(__global volatile int * flag,
                                  __global const uint * data,
                                  __global int * result,
                                  int expected,
                                  uint n,
                                  uint max_spins) {
    uint i = 0;
    while (i < max_spins && flag[0] < expected) {
        ++i;
    }
    if (flag[0] < expected) {
        result[0] = 0;
        return;
    }
    uint ok = 1;
    for (uint j = 0; j < n; j += 1024) {
        if (data[j] != (uint)(expected + j)) {
            ok = 0;
            break;
        }
    }
    if (n > 0 && data[n - 1] != (uint)(expected + n - 1)) {
        ok = 0;
    }
    result[0] = ok ? 1 : 0;
}

__kernel void fill_data_and_flag(__global uint * data,
                                 __global volatile int * flag,
                                 int expected,
                                 uint n) {
    uint gid = get_global_id(0);
    if (gid < n) {
        data[gid] = (uint)(expected + gid);
    }
    if (gid == 0) {
        flag[0] = expected;
    }
}
)CLC";

cl_program build_program(cl_context ctx, cl_device_id dev) {
    cl_int err = CL_SUCCESS;
    const char * src = kernel_src;
    size_t len = std::strlen(kernel_src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    CL_CHECK(err);
    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
    }
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
        std::fprintf(stderr, "build log:\n%s\n", log.data());
        CL_CHECK(err);
    }
    return prog;
}

void enqueue_write_kernel(cl_command_queue q, cl_kernel k, cl_mem dst, uint32_t value,
                          uint32_t n, cl_uint wait_n = 0, const cl_event * wait = nullptr,
                          cl_event * out = nullptr) {
    CL_CHECK(clSetKernelArg(k, 0, sizeof(dst), &dst));
    CL_CHECK(clSetKernelArg(k, 1, sizeof(value), &value));
    CL_CHECK(clSetKernelArg(k, 2, sizeof(n), &n));
    size_t g = ((static_cast<size_t>(n) + 255) / 256) * 256;
    CL_CHECK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, nullptr, wait_n, wait, out));
}

} // namespace

int main(int argc, char ** argv) {
    opts o = parse_args(argc, argv);
    const size_t bytes = o.size_mb * 1024 * 1024;
    const uint32_t n_u32 = static_cast<uint32_t>(bytes / sizeof(uint32_t));

    cl_platform_id platform = nullptr;
    cl_uint n_platforms = 0;
    CL_CHECK(clGetPlatformIDs(1, &platform, &n_platforms));
    if (n_platforms == 0) {
        std::fprintf(stderr, "no OpenCL platform\n");
        return 3;
    }

    cl_device_id dev = nullptr;
    cl_uint n_devs = 0;
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &dev, &n_devs));
    if (n_devs == 0) {
        std::fprintf(stderr, "no OpenCL device\n");
        return 3;
    }

    char dev_name[256] = {};
    char dev_version[256] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(dev_version), dev_version, nullptr);

    cl_device_svm_capabilities svm = 0;
    clGetDeviceInfo(dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm), &svm, nullptr);

    std::fprintf(stderr, "device=%s version=%s svm_caps=0x%llx iters=%d size_mb=%zu poll_us=%d\n",
            dev_name, dev_version, static_cast<unsigned long long>(svm), o.iters, o.size_mb, o.poll_us);
    std::fprintf(stderr, "poll_sleep_us=%d\n", o.poll_sleep_us);
    std::fprintf(stderr, "sleep_us=%d\n", o.sleep_us);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);

    cl_command_queue compute_q = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue xfer_q = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);
    CL_CHECK(err);

    cl_program prog = build_program(ctx, dev);
    cl_kernel write_k = clCreateKernel(prog, "write_u32", &err);
    CL_CHECK(err);
    cl_kernel copy_k = clCreateKernel(prog, "copy_u32", &err);
    CL_CHECK(err);
    cl_kernel signal_k = clCreateKernel(prog, "signal_flag", &err);
    CL_CHECK(err);
    cl_kernel signal_many_k = clCreateKernel(prog, "signal_flag_many", &err);
    CL_CHECK(err);
    cl_kernel wait_flag_k = clCreateKernel(prog, "wait_flag_bounded", &err);
    CL_CHECK(err);
    cl_kernel validate_k = clCreateKernel(prog, "validate_after_flag", &err);
    CL_CHECK(err);
    cl_kernel fill_flag_k = clCreateKernel(prog, "fill_data_and_flag", &err);
    CL_CHECK(err);

    std::vector<uint32_t> host(n_u32);
    for (uint32_t i = 0; i < n_u32; ++i) {
        host[i] = i;
    }

    cl_mem a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    CL_CHECK(err);
    cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    CL_CHECK(err);

    void * flag_host_raw = nullptr;
    if (posix_memalign(&flag_host_raw, 4096, 4096) != 0) {
        std::fprintf(stderr, "posix_memalign failed\n");
        return 4;
    }
    std::memset(flag_host_raw, 0, 4096);
    volatile int * flag_host = static_cast<volatile int *>(flag_host_raw);
    cl_mem flag_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                     4096, flag_host_raw, &err);
    CL_CHECK(err);
    cl_mem result_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4096, nullptr, &err);
    CL_CHECK(err);

    cl_mem alloc_flag_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                           4096, nullptr, &err);
    CL_CHECK(err);
    int * alloc_flag_ptr = static_cast<int *>(clEnqueueMapBuffer(
            compute_q, alloc_flag_buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
            0, 4096, 0, nullptr, nullptr, &err));
    CL_CHECK(err);
    std::memset(alloc_flag_ptr, 0, 4096);

    void * shared_data_raw = nullptr;
    if (posix_memalign(&shared_data_raw, 4096, bytes) != 0) {
        std::fprintf(stderr, "posix_memalign shared data failed\n");
        return 4;
    }
    std::memset(shared_data_raw, 0, bytes);
    uint32_t * shared_data = static_cast<uint32_t *>(shared_data_raw);
    cl_mem shared_data_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                            bytes, shared_data_raw, &err);
    CL_CHECK(err);

    cl_mem alloc_host_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                           bytes, nullptr, &err);
    CL_CHECK(err);
    uint32_t * alloc_host_ptr = static_cast<uint32_t *>(clEnqueueMapBuffer(
            compute_q, alloc_host_buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
            0, bytes, 0, nullptr, nullptr, &err));
    CL_CHECK(err);
    std::memset(alloc_host_ptr, 0, bytes);

    enqueue_write_kernel(compute_q, write_k, a, 1, n_u32);
    CL_CHECK(clFinish(compute_q));

    summary s_finish{"clFinish_empty"};
    summary s_marker{"marker_event_wait"};
    summary s_event{"single_event_wait"};
    summary s_barrier{"xfer_to_compute_event_chain"};
    summary s_write_block{"blocking_write"};
    summary s_write_event{"nonblocking_write_event_wait"};
    summary s_read_block{"blocking_read"};
    summary s_use_host_ptr{"use_host_ptr_kernel_read_finish"};
    summary s_gpu_cpu_flag{"gpu_signal_cpu_poll_flag"};
    summary s_gpu_cpu_event_host_ptr{"gpu_signal_cpu_event_host_ptr"};
    summary s_gpu_cpu_sleep_host_ptr{"gpu_signal_cpu_sleep_host_ptr"};
    summary s_gpu_cpu_map_host_ptr{"gpu_signal_cpu_map_host_ptr"};
    summary s_gpu_cpu_read_host_ptr{"gpu_signal_cpu_read_host_ptr"};
    summary s_cpu_gpu_use_host_ptr_data_flag{"cpu_write_gpu_validate_use_host_ptr"};
    summary s_cpu_gpu_alloc_host_ptr_data_flag{"cpu_write_gpu_validate_alloc_host_ptr"};
    summary s_gpu_cpu_alloc_host_ptr_direct{"gpu_write_cpu_direct_alloc_host_ptr"};
    summary s_gpu_cpu_alloc_host_ptr_event_direct{"gpu_write_cpu_event_read_alloc_host_ptr"};
    summary s_gpu_cpu_alloc_host_ptr_poll_flag_direct{"gpu_write_cpu_poll_flag_read_alloc_host_ptr"};
    summary s_gpu_cpu_alloc_host_ptr_poll_alloc_flag_direct{"gpu_write_cpu_poll_alloc_flag_read_alloc_host_ptr"};
    summary s_gpu_cpu_alloc_host_ptr_remap{"gpu_write_cpu_remap_alloc_host_ptr"};
    summary s_cpu_gpu_svm_data_flag{"cpu_write_gpu_validate_svm"};
    summary s_gpu_cpu_svm_direct{"gpu_write_cpu_direct_svm"};
    summary s_cpu_gpu_flag{"cpu_signal_gpu_wait_flag"};
    summary s_svm_gpu_cpu_flag{"svm_gpu_signal_cpu_poll_flag"};
    summary s_svm_cpu_gpu_flag{"svm_cpu_signal_gpu_wait_flag"};

    std::vector<uint32_t> readback(n_u32);
    int * svm_flag = nullptr;
    int * svm_result = nullptr;
    uint32_t * svm_data = nullptr;
    const bool have_fine_svm =
        (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0;
    if (have_fine_svm) {
        cl_svm_mem_flags svm_flags = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;
        if ((svm & CL_DEVICE_SVM_ATOMICS) != 0) {
            svm_flags |= CL_MEM_SVM_ATOMICS;
        }
        svm_flag = static_cast<int *>(clSVMAlloc(ctx, svm_flags, 4096, 4096));
        svm_result = static_cast<int *>(clSVMAlloc(ctx, svm_flags, 4096, 4096));
        svm_data = static_cast<uint32_t *>(clSVMAlloc(ctx, svm_flags, bytes, 4096));
        if (svm_flag == nullptr || svm_result == nullptr || svm_data == nullptr) {
            if (svm_flag) {
                clSVMFree(ctx, svm_flag);
            }
            if (svm_result) {
                clSVMFree(ctx, svm_result);
            }
            if (svm_data) {
                clSVMFree(ctx, svm_data);
            }
            svm_flag = nullptr;
            svm_result = nullptr;
            svm_data = nullptr;
        } else {
            std::memset(svm_flag, 0, 4096);
            std::memset(svm_result, 0, 4096);
            std::memset(svm_data, 0, bytes);
        }
    }

    for (int it = 0; it < o.iters; ++it) {
        auto t0 = clock_type::now();
        CL_CHECK(clFinish(compute_q));
        s_finish.add(us_since(t0));

        cl_event marker = nullptr;
        CL_CHECK(clEnqueueMarkerWithWaitList(compute_q, 0, nullptr, &marker));
        CL_CHECK(clFlush(compute_q));
        t0 = clock_type::now();
        CL_CHECK(clWaitForEvents(1, &marker));
        s_marker.add(us_since(t0));
        CL_CHECK(clReleaseEvent(marker));

        cl_event ev = nullptr;
        enqueue_write_kernel(compute_q, write_k, a, static_cast<uint32_t>(it), n_u32, 0, nullptr, &ev);
        CL_CHECK(clFlush(compute_q));
        t0 = clock_type::now();
        CL_CHECK(clWaitForEvents(1, &ev));
        s_event.add(us_since(t0));
        CL_CHECK(clReleaseEvent(ev));

        cl_event xfer_ev = nullptr;
        CL_CHECK(clEnqueueWriteBuffer(xfer_q, a, CL_FALSE, 0, bytes, host.data(), 0, nullptr, &xfer_ev));
        CL_CHECK(clFlush(xfer_q));
        t0 = clock_type::now();
        enqueue_write_kernel(compute_q, write_k, b, static_cast<uint32_t>(it + 7), n_u32, 1, &xfer_ev, nullptr);
        CL_CHECK(clFinish(compute_q));
        s_barrier.add(us_since(t0));
        CL_CHECK(clReleaseEvent(xfer_ev));

        t0 = clock_type::now();
        CL_CHECK(clEnqueueWriteBuffer(xfer_q, a, CL_TRUE, 0, bytes, host.data(), 0, nullptr, nullptr));
        s_write_block.add(us_since(t0));

        ev = nullptr;
        t0 = clock_type::now();
        CL_CHECK(clEnqueueWriteBuffer(xfer_q, a, CL_FALSE, 0, bytes, host.data(), 0, nullptr, &ev));
        CL_CHECK(clFlush(xfer_q));
        CL_CHECK(clWaitForEvents(1, &ev));
        s_write_event.add(us_since(t0));
        CL_CHECK(clReleaseEvent(ev));

        t0 = clock_type::now();
        CL_CHECK(clEnqueueReadBuffer(xfer_q, a, CL_TRUE, 0, bytes, readback.data(), 0, nullptr, nullptr));
        s_read_block.add(us_since(t0));

        cl_mem host_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                         bytes, host.data(), &err);
        if (err == CL_SUCCESS) {
            CL_CHECK(clSetKernelArg(copy_k, 0, sizeof(host_buf), &host_buf));
            CL_CHECK(clSetKernelArg(copy_k, 1, sizeof(b), &b));
            CL_CHECK(clSetKernelArg(copy_k, 2, sizeof(n_u32), &n_u32));
            size_t g = ((static_cast<size_t>(n_u32) + 255) / 256) * 256;
            t0 = clock_type::now();
            CL_CHECK(clEnqueueNDRangeKernel(compute_q, copy_k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));
            CL_CHECK(clFinish(compute_q));
            s_use_host_ptr.add(us_since(t0));
            CL_CHECK(clReleaseMemObject(host_buf));
        } else {
            ++s_use_host_ptr.failures;
        }

        *flag_host = 0;
        const int expected = it + 1;
        CL_CHECK(clSetKernelArg(signal_k, 0, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(signal_k, 1, sizeof(expected), &expected));
        size_t one = 1;
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, signal_k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr));
        CL_CHECK(clFlush(compute_q));
        t0 = clock_type::now();
        bool seen = false;
        while (us_since(t0) < o.poll_us) {
            if (*flag_host >= expected) {
                seen = true;
                break;
            }
            if (o.poll_sleep_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(o.poll_sleep_us));
            }
        }
        if (seen) {
            s_gpu_cpu_flag.add(us_since(t0));
        } else {
            ++s_gpu_cpu_flag.failures;
            CL_CHECK(clFinish(compute_q));
        }

        *flag_host = 0;
        CL_CHECK(clSetKernelArg(signal_k, 0, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(signal_k, 1, sizeof(expected), &expected));
        cl_event flag_ev = nullptr;
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, signal_k, 1, nullptr, &one, nullptr, 0, nullptr, &flag_ev));
        CL_CHECK(clFlush(compute_q));
        t0 = clock_type::now();
        CL_CHECK(clWaitForEvents(1, &flag_ev));
        if (*flag_host >= expected) {
            s_gpu_cpu_event_host_ptr.add(us_since(t0));
        } else {
            ++s_gpu_cpu_event_host_ptr.failures;
        }
        CL_CHECK(clReleaseEvent(flag_ev));

        *flag_host = 0;
        CL_CHECK(clSetKernelArg(signal_k, 0, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(signal_k, 1, sizeof(expected), &expected));
        flag_ev = nullptr;
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, signal_k, 1, nullptr, &one, nullptr, 0, nullptr, &flag_ev));
        CL_CHECK(clFlush(compute_q));
        t0 = clock_type::now();
        CL_CHECK(clWaitForEvents(1, &flag_ev));
        if (o.sleep_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(o.sleep_us));
        }
        if (*flag_host >= expected) {
            s_gpu_cpu_sleep_host_ptr.add(us_since(t0));
        } else {
            ++s_gpu_cpu_sleep_host_ptr.failures;
        }
        CL_CHECK(clReleaseEvent(flag_ev));

        *flag_host = 0;
        CL_CHECK(clSetKernelArg(signal_k, 0, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(signal_k, 1, sizeof(expected), &expected));
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, signal_k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr));
        CL_CHECK(clFlush(compute_q));
        t0 = clock_type::now();
        void * mapped_flag = clEnqueueMapBuffer(compute_q, flag_buf, CL_TRUE, CL_MAP_READ,
                                                0, 4096, 0, nullptr, nullptr, &err);
        CL_CHECK(err);
        volatile int * mapped_i = static_cast<volatile int *>(mapped_flag);
        if (mapped_i[0] >= expected) {
            s_gpu_cpu_map_host_ptr.add(us_since(t0));
        } else {
            ++s_gpu_cpu_map_host_ptr.failures;
        }
        CL_CHECK(clEnqueueUnmapMemObject(compute_q, flag_buf, mapped_flag, 0, nullptr, nullptr));
        CL_CHECK(clFinish(compute_q));

        *flag_host = 0;
        int read_flag = 0;
        CL_CHECK(clSetKernelArg(signal_k, 0, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(signal_k, 1, sizeof(expected), &expected));
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, signal_k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr));
        t0 = clock_type::now();
        CL_CHECK(clEnqueueReadBuffer(compute_q, flag_buf, CL_TRUE, 0, sizeof(read_flag), &read_flag, 0, nullptr, nullptr));
        if (read_flag >= expected) {
            s_gpu_cpu_read_host_ptr.add(us_since(t0));
        } else {
            ++s_gpu_cpu_read_host_ptr.failures;
        }

        for (uint32_t i = 0; i < n_u32; ++i) {
            shared_data[i] = static_cast<uint32_t>(expected + i);
        }
        *flag_host = expected;
        int validate_result = 0;
        CL_CHECK(clEnqueueWriteBuffer(compute_q, result_buf, CL_TRUE, 0, sizeof(validate_result), &validate_result, 0, nullptr, nullptr));
        CL_CHECK(clSetKernelArg(validate_k, 0, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(validate_k, 1, sizeof(shared_data_buf), &shared_data_buf));
        CL_CHECK(clSetKernelArg(validate_k, 2, sizeof(result_buf), &result_buf));
        CL_CHECK(clSetKernelArg(validate_k, 3, sizeof(expected), &expected));
        CL_CHECK(clSetKernelArg(validate_k, 4, sizeof(n_u32), &n_u32));
        const uint32_t max_spins_validate = 1000000;
        CL_CHECK(clSetKernelArg(validate_k, 5, sizeof(max_spins_validate), &max_spins_validate));
        t0 = clock_type::now();
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, validate_k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(compute_q, result_buf, CL_TRUE, 0, sizeof(validate_result), &validate_result, 0, nullptr, nullptr));
        if (validate_result == 1) {
            s_cpu_gpu_use_host_ptr_data_flag.add(us_since(t0));
        } else {
            ++s_cpu_gpu_use_host_ptr_data_flag.failures;
        }

        for (uint32_t i = 0; i < n_u32; ++i) {
            alloc_host_ptr[i] = static_cast<uint32_t>(expected + i);
        }
        *flag_host = expected;
        validate_result = 0;
        CL_CHECK(clEnqueueWriteBuffer(compute_q, result_buf, CL_TRUE, 0, sizeof(validate_result), &validate_result, 0, nullptr, nullptr));
        CL_CHECK(clSetKernelArg(validate_k, 0, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(validate_k, 1, sizeof(alloc_host_buf), &alloc_host_buf));
        CL_CHECK(clSetKernelArg(validate_k, 2, sizeof(result_buf), &result_buf));
        CL_CHECK(clSetKernelArg(validate_k, 3, sizeof(expected), &expected));
        CL_CHECK(clSetKernelArg(validate_k, 4, sizeof(n_u32), &n_u32));
        CL_CHECK(clSetKernelArg(validate_k, 5, sizeof(max_spins_validate), &max_spins_validate));
        t0 = clock_type::now();
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, validate_k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(compute_q, result_buf, CL_TRUE, 0, sizeof(validate_result), &validate_result, 0, nullptr, nullptr));
        if (validate_result == 1) {
            s_cpu_gpu_alloc_host_ptr_data_flag.add(us_since(t0));
        } else {
            ++s_cpu_gpu_alloc_host_ptr_data_flag.failures;
        }

        std::memset(alloc_host_ptr, 0, bytes);
        *flag_host = 0;
        CL_CHECK(clSetKernelArg(fill_flag_k, 0, sizeof(alloc_host_buf), &alloc_host_buf));
        CL_CHECK(clSetKernelArg(fill_flag_k, 1, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(fill_flag_k, 2, sizeof(expected), &expected));
        CL_CHECK(clSetKernelArg(fill_flag_k, 3, sizeof(n_u32), &n_u32));
        size_t g_data = ((static_cast<size_t>(n_u32) + 255) / 256) * 256;
        cl_event fill_ev = nullptr;
        t0 = clock_type::now();
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, fill_flag_k, 1, nullptr, &g_data, nullptr, 0, nullptr, &fill_ev));
        CL_CHECK(clFlush(compute_q));
        CL_CHECK(clWaitForEvents(1, &fill_ev));
        if (o.sleep_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(o.sleep_us));
        }
        if (alloc_host_ptr[0] == static_cast<uint32_t>(expected) &&
            alloc_host_ptr[n_u32 - 1] == static_cast<uint32_t>(expected + n_u32 - 1)) {
            s_gpu_cpu_alloc_host_ptr_direct.add(us_since(t0));
        } else {
            ++s_gpu_cpu_alloc_host_ptr_direct.failures;
        }
        CL_CHECK(clReleaseEvent(fill_ev));

        std::memset(alloc_host_ptr, 0, bytes);
        *flag_host = 0;
        CL_CHECK(clSetKernelArg(fill_flag_k, 0, sizeof(alloc_host_buf), &alloc_host_buf));
        CL_CHECK(clSetKernelArg(fill_flag_k, 1, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(fill_flag_k, 2, sizeof(expected), &expected));
        CL_CHECK(clSetKernelArg(fill_flag_k, 3, sizeof(n_u32), &n_u32));
        fill_ev = nullptr;
        t0 = clock_type::now();
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, fill_flag_k, 1, nullptr, &g_data, nullptr, 0, nullptr, &fill_ev));
        CL_CHECK(clFlush(compute_q));
        CL_CHECK(clWaitForEvents(1, &fill_ev));
        if (alloc_host_ptr[0] == static_cast<uint32_t>(expected) &&
            alloc_host_ptr[n_u32 - 1] == static_cast<uint32_t>(expected + n_u32 - 1)) {
            s_gpu_cpu_alloc_host_ptr_event_direct.add(us_since(t0));
        } else {
            ++s_gpu_cpu_alloc_host_ptr_event_direct.failures;
        }
        CL_CHECK(clReleaseEvent(fill_ev));

        std::memset(alloc_host_ptr, 0, bytes);
        *flag_host = 0;
        CL_CHECK(clSetKernelArg(fill_flag_k, 0, sizeof(alloc_host_buf), &alloc_host_buf));
        CL_CHECK(clSetKernelArg(fill_flag_k, 1, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(fill_flag_k, 2, sizeof(expected), &expected));
        CL_CHECK(clSetKernelArg(fill_flag_k, 3, sizeof(n_u32), &n_u32));
        t0 = clock_type::now();
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, fill_flag_k, 1, nullptr, &g_data, nullptr, 0, nullptr, nullptr));
        CL_CHECK(clFlush(compute_q));
        seen = false;
        while (us_since(t0) < o.poll_us) {
            if (*flag_host >= expected) {
                seen = true;
                break;
            }
            if (o.poll_sleep_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(o.poll_sleep_us));
            }
        }
        if (seen &&
            alloc_host_ptr[0] == static_cast<uint32_t>(expected) &&
            alloc_host_ptr[n_u32 - 1] == static_cast<uint32_t>(expected + n_u32 - 1)) {
            s_gpu_cpu_alloc_host_ptr_poll_flag_direct.add(us_since(t0));
        } else {
            ++s_gpu_cpu_alloc_host_ptr_poll_flag_direct.failures;
            CL_CHECK(clFinish(compute_q));
        }

        std::memset(alloc_host_ptr, 0, bytes);
        alloc_flag_ptr[0] = 0;
        CL_CHECK(clSetKernelArg(fill_flag_k, 0, sizeof(alloc_host_buf), &alloc_host_buf));
        CL_CHECK(clSetKernelArg(fill_flag_k, 1, sizeof(alloc_flag_buf), &alloc_flag_buf));
        CL_CHECK(clSetKernelArg(fill_flag_k, 2, sizeof(expected), &expected));
        CL_CHECK(clSetKernelArg(fill_flag_k, 3, sizeof(n_u32), &n_u32));
        t0 = clock_type::now();
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, fill_flag_k, 1, nullptr, &g_data, nullptr, 0, nullptr, nullptr));
        CL_CHECK(clFlush(compute_q));
        seen = false;
        while (us_since(t0) < o.poll_us) {
            if (alloc_flag_ptr[0] >= expected) {
                seen = true;
                break;
            }
            if (o.poll_sleep_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(o.poll_sleep_us));
            }
        }
        if (seen &&
            alloc_host_ptr[0] == static_cast<uint32_t>(expected) &&
            alloc_host_ptr[n_u32 - 1] == static_cast<uint32_t>(expected + n_u32 - 1)) {
            s_gpu_cpu_alloc_host_ptr_poll_alloc_flag_direct.add(us_since(t0));
        } else {
            ++s_gpu_cpu_alloc_host_ptr_poll_alloc_flag_direct.failures;
            CL_CHECK(clFinish(compute_q));
        }

        CL_CHECK(clEnqueueUnmapMemObject(compute_q, alloc_host_buf, alloc_host_ptr, 0, nullptr, nullptr));
        CL_CHECK(clFinish(compute_q));
        *flag_host = 0;
        CL_CHECK(clSetKernelArg(fill_flag_k, 0, sizeof(alloc_host_buf), &alloc_host_buf));
        CL_CHECK(clSetKernelArg(fill_flag_k, 1, sizeof(flag_buf), &flag_buf));
        CL_CHECK(clSetKernelArg(fill_flag_k, 2, sizeof(expected), &expected));
        CL_CHECK(clSetKernelArg(fill_flag_k, 3, sizeof(n_u32), &n_u32));
        t0 = clock_type::now();
        CL_CHECK(clEnqueueNDRangeKernel(compute_q, fill_flag_k, 1, nullptr, &g_data, nullptr, 0, nullptr, nullptr));
        alloc_host_ptr = static_cast<uint32_t *>(clEnqueueMapBuffer(
                compute_q, alloc_host_buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                0, bytes, 0, nullptr, nullptr, &err));
        CL_CHECK(err);
        if (alloc_host_ptr[0] == static_cast<uint32_t>(expected) &&
            alloc_host_ptr[n_u32 - 1] == static_cast<uint32_t>(expected + n_u32 - 1)) {
            s_gpu_cpu_alloc_host_ptr_remap.add(us_since(t0));
        } else {
            ++s_gpu_cpu_alloc_host_ptr_remap.failures;
        }

        if (o.gpu_poll) {
            *flag_host = expected;
            const uint32_t max_spins = 1000000;
            CL_CHECK(clSetKernelArg(wait_flag_k, 0, sizeof(flag_buf), &flag_buf));
            CL_CHECK(clSetKernelArg(wait_flag_k, 1, sizeof(result_buf), &result_buf));
            CL_CHECK(clSetKernelArg(wait_flag_k, 2, sizeof(expected), &expected));
            CL_CHECK(clSetKernelArg(wait_flag_k, 3, sizeof(max_spins), &max_spins));
            t0 = clock_type::now();
            CL_CHECK(clEnqueueNDRangeKernel(compute_q, wait_flag_k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr));
            CL_CHECK(clFinish(compute_q));
            s_cpu_gpu_flag.add(us_since(t0));
        }

        if (svm_flag != nullptr && svm_result != nullptr) {
            *svm_flag = 0;
            CL_CHECK(clSetKernelArgSVMPointer(signal_k, 0, svm_flag));
            CL_CHECK(clSetKernelArg(signal_k, 1, sizeof(expected), &expected));
            CL_CHECK(clEnqueueNDRangeKernel(compute_q, signal_k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr));
            CL_CHECK(clFlush(compute_q));
            t0 = clock_type::now();
            bool svm_seen = false;
            while (us_since(t0) < o.poll_us) {
                if (*svm_flag >= expected) {
                    svm_seen = true;
                    break;
                }
                if (o.poll_sleep_us > 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(o.poll_sleep_us));
                }
            }
            if (svm_seen) {
                s_svm_gpu_cpu_flag.add(us_since(t0));
            } else {
                ++s_svm_gpu_cpu_flag.failures;
                CL_CHECK(clFinish(compute_q));
            }

            if (o.gpu_poll) {
                *svm_flag = expected;
                const uint32_t max_spins = 1000000;
                CL_CHECK(clSetKernelArgSVMPointer(wait_flag_k, 0, svm_flag));
                CL_CHECK(clSetKernelArgSVMPointer(wait_flag_k, 1, svm_result));
                CL_CHECK(clSetKernelArg(wait_flag_k, 2, sizeof(expected), &expected));
                CL_CHECK(clSetKernelArg(wait_flag_k, 3, sizeof(max_spins), &max_spins));
                t0 = clock_type::now();
                CL_CHECK(clEnqueueNDRangeKernel(compute_q, wait_flag_k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr));
                CL_CHECK(clFinish(compute_q));
                if (*svm_result == 1) {
                    s_svm_cpu_gpu_flag.add(us_since(t0));
                } else {
                    ++s_svm_cpu_gpu_flag.failures;
                }
            }

            for (uint32_t i = 0; i < n_u32; ++i) {
                svm_data[i] = static_cast<uint32_t>(expected + i);
            }
            *svm_flag = expected;
            *svm_result = 0;
            CL_CHECK(clSetKernelArgSVMPointer(validate_k, 0, svm_flag));
            CL_CHECK(clSetKernelArgSVMPointer(validate_k, 1, svm_data));
            CL_CHECK(clSetKernelArgSVMPointer(validate_k, 2, svm_result));
            CL_CHECK(clSetKernelArg(validate_k, 3, sizeof(expected), &expected));
            CL_CHECK(clSetKernelArg(validate_k, 4, sizeof(n_u32), &n_u32));
            const uint32_t max_spins_validate_svm = 1000000;
            CL_CHECK(clSetKernelArg(validate_k, 5, sizeof(max_spins_validate_svm), &max_spins_validate_svm));
            t0 = clock_type::now();
            CL_CHECK(clEnqueueNDRangeKernel(compute_q, validate_k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr));
            CL_CHECK(clFinish(compute_q));
            if (*svm_result == 1) {
                s_cpu_gpu_svm_data_flag.add(us_since(t0));
            } else {
                ++s_cpu_gpu_svm_data_flag.failures;
            }

            std::memset(svm_data, 0, bytes);
            *svm_flag = 0;
            CL_CHECK(clSetKernelArgSVMPointer(fill_flag_k, 0, svm_data));
            CL_CHECK(clSetKernelArgSVMPointer(fill_flag_k, 1, svm_flag));
            CL_CHECK(clSetKernelArg(fill_flag_k, 2, sizeof(expected), &expected));
            CL_CHECK(clSetKernelArg(fill_flag_k, 3, sizeof(n_u32), &n_u32));
            size_t g_svm_data = ((static_cast<size_t>(n_u32) + 255) / 256) * 256;
            t0 = clock_type::now();
            CL_CHECK(clEnqueueNDRangeKernel(compute_q, fill_flag_k, 1, nullptr, &g_svm_data, nullptr, 0, nullptr, nullptr));
            CL_CHECK(clFinish(compute_q));
            if (svm_data[0] == static_cast<uint32_t>(expected) &&
                svm_data[n_u32 - 1] == static_cast<uint32_t>(expected + n_u32 - 1)) {
                s_gpu_cpu_svm_direct.add(us_since(t0));
            } else {
                ++s_gpu_cpu_svm_direct.failures;
            }
        } else if (have_fine_svm) {
            ++s_svm_gpu_cpu_flag.failures;
            if (o.gpu_poll) {
                ++s_svm_cpu_gpu_flag.failures;
            }
            ++s_cpu_gpu_svm_data_flag.failures;
            ++s_gpu_cpu_svm_direct.failures;
        }
    }

    std::printf("test,n,mean_us,p50_us,p95_us,max_us,failures\n");
    print_summary(s_finish);
    print_summary(s_marker);
    print_summary(s_event);
    print_summary(s_barrier);
    print_summary(s_write_block);
    print_summary(s_write_event);
    print_summary(s_read_block);
    print_summary(s_use_host_ptr);
    print_summary(s_gpu_cpu_flag);
    print_summary(s_gpu_cpu_event_host_ptr);
    print_summary(s_gpu_cpu_sleep_host_ptr);
    print_summary(s_gpu_cpu_map_host_ptr);
    print_summary(s_gpu_cpu_read_host_ptr);
    print_summary(s_cpu_gpu_use_host_ptr_data_flag);
    print_summary(s_cpu_gpu_alloc_host_ptr_data_flag);
    print_summary(s_gpu_cpu_alloc_host_ptr_direct);
    print_summary(s_gpu_cpu_alloc_host_ptr_event_direct);
    print_summary(s_gpu_cpu_alloc_host_ptr_poll_flag_direct);
    print_summary(s_gpu_cpu_alloc_host_ptr_poll_alloc_flag_direct);
    print_summary(s_gpu_cpu_alloc_host_ptr_remap);
    print_summary(s_cpu_gpu_svm_data_flag);
    print_summary(s_gpu_cpu_svm_direct);
    print_summary(s_svm_gpu_cpu_flag);
    if (o.gpu_poll) {
        print_summary(s_cpu_gpu_flag);
        print_summary(s_svm_cpu_gpu_flag);
    }

    if (svm_flag) {
        clSVMFree(ctx, svm_flag);
    }
    if (svm_result) {
        clSVMFree(ctx, svm_result);
    }
    if (svm_data) {
        clSVMFree(ctx, svm_data);
    }
    if (alloc_host_ptr) {
        clEnqueueUnmapMemObject(compute_q, alloc_host_buf, alloc_host_ptr, 0, nullptr, nullptr);
        clFinish(compute_q);
    }
    clReleaseMemObject(alloc_host_buf);
    if (alloc_flag_ptr) {
        clEnqueueUnmapMemObject(compute_q, alloc_flag_buf, alloc_flag_ptr, 0, nullptr, nullptr);
        clFinish(compute_q);
    }
    clReleaseMemObject(alloc_flag_buf);
    clReleaseMemObject(shared_data_buf);
    free(shared_data_raw);
    clReleaseMemObject(result_buf);
    clReleaseMemObject(flag_buf);
    free(flag_host_raw);
    clReleaseMemObject(b);
    clReleaseMemObject(a);
    clReleaseKernel(wait_flag_k);
    clReleaseKernel(fill_flag_k);
    clReleaseKernel(validate_k);
    clReleaseKernel(signal_many_k);
    clReleaseKernel(signal_k);
    clReleaseKernel(copy_k);
    clReleaseKernel(write_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(xfer_q);
    clReleaseCommandQueue(compute_q);
    clReleaseContext(ctx);
    return 0;
}
