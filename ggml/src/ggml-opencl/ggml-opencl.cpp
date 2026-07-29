#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// suppress warnings in CL headers for GCC and Clang
#pragma GCC diagnostic ignored "-Woverlength-strings"
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
#endif

#include "ggml-opencl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml.h"

#include <CL/cl.h>

#include <inttypes.h>
#include <string.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <memory>
#include <new>
#include <charconv>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <condition_variable>
#include <deque>

#include <sys/mman.h>   // posix_madvise
#include <unistd.h>     // sysconf(_SC_PAGESIZE)

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Elastic baseline 集成：runtime/ 模块的头文件由 ggml-opencl CMakeLists.txt 的
// target_include_directories 把 runtime/ 加入搜索路径。
#include "budget_watcher.h"
#include "elastic_granularity.h"
#include "elastic_profile_writer.h"
#include "metrics_logger.h"
#include "weight_buffer_manager.h"
#include "weight_buffer_manager_opencl.h"
// O_DIRECT path: 让 elastic reload 绕过 mmap page cache. 通过
// llama_mmap_registry 反查 host_ptr → (file, file_offset).
#include "../../../src/llama-mmap.h"

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define UNUSED(x) (void)(x)

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            GGML_LOG_ERROR("ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            GGML_ASSERT(0);                                         \
        }                                                           \
    } while (0)

//------------------------------------------------------------------------------
// OpenCL
//------------------------------------------------------------------------------

bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor);

enum GPU_FAMILY {
    ADRENO,
    INTEL,
    UNKNOWN,
};

enum ADRENO_GPU_GEN {
    ADRENO_UNKNOWN,
    A7X,
    A8X,
    X1E,
};

enum ADRENO_CL_COMPILER_TYPE {
    E031,
    DX,
};

struct ggml_cl_version {
    cl_uint major = 0;
    cl_uint minor = 0;
};


struct ggml_cl_compiler_version {
    ADRENO_CL_COMPILER_TYPE type;
    int major = -1;
    int minor = -1;
    int patch = -1;

    bool same(ADRENO_CL_COMPILER_TYPE t, int x, int y, int z) const {
        return major == x && minor == y && patch == z && type == t;
    }
    bool newer_than(ADRENO_CL_COMPILER_TYPE t, int x, int y, int z) const {
        return major*10000 + minor*100 + patch > x*10000 + y*100 + z && type == t;
    }
    bool newer_than_or_same(ADRENO_CL_COMPILER_TYPE t, int x, int y, int z) const {
        return same(t, x, y, z) || newer_than(t, x, y, z);
    }
};

static size_t align_to(size_t value, size_t to_alignment) {
    GGML_ASSERT(to_alignment && "Invalid alignment (must be non-zero)");
    GGML_ASSERT((to_alignment & (to_alignment - 1)) == 0 && "to_alignment must be power-of-two");

    return ((value + to_alignment - 1) / to_alignment) * to_alignment;
}


// Parses a version string of form "XX.YY ". On an error returns ggml_cl_version with all zeroes.
static ggml_cl_version parse_cl_version(std::string_view str) {
    size_t major_str_begin = 0;
    size_t major_str_end   = str.find(".", major_str_begin);
    if (major_str_end == std::string::npos) {
        return {};
    }

    size_t minor_str_begin = major_str_end + 1;
    size_t minor_str_end   = str.find(" ", minor_str_begin);
    if (minor_str_end == std::string::npos) {
        return {};
    }

    cl_uint version_major;
    if (std::from_chars(str.data() + major_str_begin, str.data() + major_str_end, version_major).ec != std::errc{}) {
        return {};
    }

    cl_uint version_minor;
    if (std::from_chars(str.data() + minor_str_begin, str.data() + minor_str_end, version_minor).ec != std::errc{}) {
        return {};
    }
    return { version_major, version_minor };
}

// Returns OpenCL platform's version. On an error returns ggml_cl_version with all zeroes.
static ggml_cl_version get_opencl_platform_version(cl_platform_id platform) {
    size_t param_size;
    CL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &param_size));
    std::unique_ptr<char[]> param_storage(new char[param_size]);
    CL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, param_size, param_storage.get(), nullptr));

    auto              param_value    = std::string_view(param_storage.get(), param_size);
    const std::string version_prefix = "OpenCL ";  // Suffix: "XX.YY <platform-specific-info>"
    if (param_value.find(version_prefix) != 0) {
        return {};
    }
    param_value.remove_prefix(version_prefix.length());
    return parse_cl_version(param_value);
}

// Return a version to use in OpenCL C compilation. On an error returns ggml_cl_version with all zeroes.
static ggml_cl_version get_opencl_c_version(ggml_cl_version platform_version, cl_device_id device) {
    size_t param_size;

#if CL_TARGET_OPENCL_VERSION >= 300
    if (platform_version.major >= 3) {
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0, nullptr, &param_size));
        if (!param_size) {
            return {};
        }

        std::unique_ptr<cl_name_version[]> versions(new cl_name_version[param_size]);
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, param_size, versions.get(), nullptr));
        unsigned versions_count = param_size / sizeof(cl_name_version);

        cl_version version_max = 0;
        for (unsigned i = 0; i < versions_count; i++) {
            version_max = std::max<cl_version>(versions[i].version, version_max);
        }

        return { CL_VERSION_MAJOR(version_max), CL_VERSION_MINOR(version_max) };
    }
#else
    GGML_UNUSED(platform_version);
#endif  // CL_TARGET_OPENCL_VERSION >= 300

    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &param_size));
    if (!param_size) {
        return {};
    }

    std::unique_ptr<char[]> param_storage(new char[param_size]);
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, param_size, param_storage.get(), nullptr));
    auto param_value = std::string_view(param_storage.get(), param_size);

    const std::string version_prefix = "OpenCL C ";  // Suffix: "XX.YY <platform-specific-info>"
    if (param_value.find(version_prefix) != 0) {
        return {};
    }
    param_value.remove_prefix(version_prefix.length());

    return parse_cl_version(param_value);
}

static ADRENO_GPU_GEN get_adreno_gpu_gen(const char *device_name) {
    if (strstr(device_name, "730") ||
        strstr(device_name, "740") ||
        strstr(device_name, "750")) {
        return ADRENO_GPU_GEN::A7X;
    }

    if (strstr(device_name, "830")) {
        return ADRENO_GPU_GEN::A8X;
    }

    if (strstr(device_name, "X1")) {
        return ADRENO_GPU_GEN::X1E;
    }

    return ADRENO_GPU_GEN::ADRENO_UNKNOWN;
}

static ggml_cl_compiler_version get_adreno_cl_compiler_version(const char *driver_version) {
    std::string driver_ver_str(driver_version);
    ADRENO_CL_COMPILER_TYPE type = ADRENO_CL_COMPILER_TYPE::E031;
    size_t compiler_ver_pos = driver_ver_str.find("E031");
    size_t compiler_ver_len = 13;
    size_t compiler_major_offset = 5;
    size_t compiler_minor_offset = 8;
    size_t compiler_patch_offset = 11;

    if (compiler_ver_pos == std::string::npos) {
        compiler_ver_pos = driver_ver_str.find("DX");
        if (compiler_ver_pos == std::string::npos) {
            return {};
        }
        type = ADRENO_CL_COMPILER_TYPE::DX;
        compiler_ver_len = 11;
        compiler_major_offset = 3;
    }

    std::string compiler_ver_str = driver_ver_str.substr(compiler_ver_pos, compiler_ver_len);
    int major = std::atoi(compiler_ver_str.substr(compiler_major_offset, 2).c_str());
    int minor = std::atoi(compiler_ver_str.substr(compiler_minor_offset, 2).c_str());
    int patch = std::atoi(compiler_ver_str.substr(compiler_patch_offset, 2).c_str());
    return { type, major, minor, patch };
}

// Profiling
struct ProfilingInfo {
    std::string op_name;
    std::string kernel_name;

    cl_kernel kernel;
    cl_event evt;

    cl_ulong cmd_queued;
    cl_ulong cmd_submit;
    cl_ulong cmd_start;
    cl_ulong cmd_end;
    cl_ulong overhead_start;
    cl_ulong overhead_end;
    // For the times below, see spec for clGetEventProfilingInfo
    // The time kernel spent in cmd queue - SUBMIT - QUEUED
    cl_ulong cmd_queued_duration_ns;
    // The time kernel spent for submission - START - SUBMIT
    cl_ulong cmd_submit_duration_ns;
    // Kernel execution time in nanoseconds - END - START
    cl_ulong cmd_duration_ns;
    // The time for the kernel to complete - COMPLETE - END
    cl_ulong cmd_complete_duration_ns;
    // Total time to finish the kernel - COMPELTE - QUEUED
    cl_ulong cmd_total_duration_ns;
    // Global and local work sizes.
    size_t global_size[3];
    size_t local_size[3];
    // Op output size.
    size_t output_size[4];
};

static void populateProfilingInfo(
        ProfilingInfo& info, cl_event evt, cl_kernel kernel, cl_uint work_dim,
        size_t global_size[3], size_t local_size[3],
        const ggml_tensor * tensor) {
    info.op_name     = tensor->name;
    info.kernel      = kernel;
    info.evt         = evt;

    // 0 means not specified, e.g., 2D workgroup, or NULL for driver to choose
    info.local_size[0] = 0;
    info.local_size[1] = 0;
    info.local_size[2] = 0;

    info.global_size[0] = 0;
    info.global_size[1] = 0;
    info.global_size[2] = 0;

    if (local_size) {
        for (cl_uint i = 0; i < work_dim; ++i) {
            info.local_size[i] = local_size[i];
        }
    }

    for (cl_uint i = 0; i < work_dim; ++i) {
        info.global_size[i] = global_size[i];
    }

    info.output_size[0] = tensor->ne[0];
    info.output_size[1] = tensor->ne[1];
    info.output_size[2] = tensor->ne[2];
    info.output_size[3] = tensor->ne[3];
}

struct ggml_backend_opencl_context;

// backend device context
struct ggml_backend_opencl_device_context {
    cl_platform_id platform;
    std::string platform_name;

    cl_device_id   device;
    std::string    device_name;
    cl_device_type device_type;
    std::string    device_version;

    // Initialized by ggml_cl2_init().
    ggml_backend_opencl_context * backend_ctx = nullptr;

    // Initialized by ggml_backend_opencl_device_get_buffer_type()
    ggml_backend_buffer_type buffer_type;
    ggml_backend_buffer_type host_buffer_type;

    cl_context context = nullptr;
};

// backend context
struct ggml_backend_opencl_context {
    int ref_count;

    cl_device_id device;
    std::string device_name;

    std::string driver_version;

    GPU_FAMILY gpu_family;
    ADRENO_GPU_GEN adreno_gen;

    cl_int alignment;
    size_t max_alloc_size;
    size_t max_workgroup_size;
    bool fp16_support;
    bool has_vector_subgroup_broadcast;
    bool disable_fusion;
    ggml_cl_compiler_version adreno_cl_compiler_version;

    int adreno_wave_size;

    cl_bool non_uniform_workgroups;

    cl_context context;
    cl_command_queue queue;

    cl_program program_add;
    cl_program program_add_id;
    cl_program program_clamp;
    cl_program program_cpy;
    cl_program program_cvt;
    cl_program program_diag_mask_inf;
    cl_program program_gelu;
    cl_program program_gemv_noshuffle_general;
    cl_program program_gemv_noshuffle;
    cl_program program_get_rows;
    cl_program program_set_rows;
    cl_program program_glu;
    cl_program program_im2col_f16;
    cl_program program_im2col_f32;
    cl_program program_mul_mat_Ab_Bi_8x4;
    cl_program program_mul_mv_q4_0_f32;
    cl_program program_mul_mv_q4_0_f32_v;
    cl_program program_mul_mv_q4_0_f32_8x_flat;
    cl_program program_mul_mv_q4_0_f32_1d_8x_flat;
    cl_program program_mul_mv_q4_0_f32_1d_16x_flat;
    cl_program program_mul_mv_q6_K;
    cl_program program_mul_mv_q8_0_f32, program_mul_mv_q8_0_f32_flat;
    cl_program program_mul_mv_mxfp4_f32;
    cl_program program_mul_mv_mxfp4_f32_flat;
    cl_program program_mul_mv_f16_f16;
    cl_program program_mul_mv_f16_f32_1row;
    cl_program program_mul_mv_f16_f32_l4;
    cl_program program_mul_mv_f16_f32;
    cl_program program_mul_mv_f32_f32;
    cl_program program_mul;
    cl_program program_mul_mat_f16_f32_tiled;
    cl_program program_div;
    cl_program program_sub;
    cl_program program_norm;
    cl_program program_relu;
    cl_program program_rms_norm;
    cl_program program_group_norm;
    cl_program program_rope;
    cl_program program_scale;
    cl_program program_silu;
    cl_program program_sigmoid;
    cl_program program_softmax_f32;
    cl_program program_softmax_f16;
    cl_program program_softmax_4_f32;
    cl_program program_softmax_4_f16;
    cl_program program_argsort_f32_i32;
    cl_program program_sum_rows_f32;
    cl_program program_repeat;
    cl_program program_pad;
    cl_program program_tanh;
    cl_program program_upscale;
    cl_program program_concat;
    cl_program program_conv_2d_f16;
    cl_program program_conv_2d_f32;
    cl_program program_conv_2d_f16_f32;
    cl_program program_tsembd;
    cl_program program_gemv_moe_mxfp4_f32, program_gemm_moe_mxfp4_f32;
    cl_program program_mul_mv_id_q4_0_f32_8x_flat;
    cl_program program_mul_mv_id_q8_0_f32, program_mul_mv_id_q8_0_f32_flat;
    cl_program program_mul_mv_id_mxfp4_f32;
    cl_program program_mul_mv_id_mxfp4_f32_flat;
    cl_program program_mul_mm_f32_f32_l4_lm;
    cl_program program_mul_mm_f16_f32_l4_lm;
    cl_program program_mul_mm_q8_0_f32_l4_lm;

    cl_kernel kernel_add, kernel_add_row, kernel_add_f16, kernel_add_row_f16;
    cl_kernel kernel_mul, kernel_mul_row, kernel_mul_f16, kernel_mul_row_f16;
    cl_kernel kernel_div, kernel_div_row, kernel_div_f16, kernel_div_row_f16;
    cl_kernel kernel_sub, kernel_sub_row, kernel_sub_f16, kernel_sub_row_f16;
    cl_kernel kernel_add_id;
    cl_kernel kernel_scale;
    cl_kernel kernel_silu, kernel_silu_4;
    cl_kernel kernel_gelu, kernel_gelu_4;
    cl_kernel kernel_gelu_erf, kernel_gelu_erf_4;
    cl_kernel kernel_gelu_quick, kernel_gelu_quick_4;
    cl_kernel kernel_relu;
    cl_kernel kernel_sigmoid_f32, kernel_sigmoid_f16;
    cl_kernel kernel_clamp;
    cl_kernel kernel_geglu, kernel_reglu, kernel_swiglu, kernel_swiglu_oai, kernel_geglu_erf, kernel_geglu_quick,
              kernel_geglu_f16, kernel_reglu_f16, kernel_swiglu_f16, kernel_geglu_erf_f16, kernel_geglu_quick_f16;
    cl_kernel kernel_norm, kernel_norm_mul_add;
    cl_kernel kernel_rms_norm, kernel_rms_norm_mul;
    cl_kernel kernel_group_norm, kernel_group_norm_mul_add;
    cl_kernel kernel_diag_mask_inf, kernel_diag_mask_inf_8;
    cl_kernel kernel_soft_max, kernel_soft_max_4;
    cl_kernel kernel_soft_max_f16, kernel_soft_max_4_f16;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f16;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f16_q1;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f32;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f32_q1;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f32_f16;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f32_f16_q1;
    std::map<std::pair<int, int>, int>       kernels_flash_attn_bm;
    std::map<std::pair<int, int>, int>       kernels_flash_attn_bn;
    cl_kernel kernel_get_rows_f32, kernel_get_rows_f16, kernel_get_rows_q4_0;
    cl_kernel kernel_set_rows_f32_i64, kernel_set_rows_f32_i32, kernel_set_rows_f16_i64, kernel_set_rows_f16_i32;
    cl_kernel kernel_rope_norm_f32, kernel_rope_norm_f16, kernel_rope_neox_f32, kernel_rope_neox_f16;
    cl_kernel kernel_rope_multi_f32, kernel_rope_multi_f16, kernel_rope_vision_f32, kernel_rope_vision_f16;
    cl_kernel kernel_cpy_f16_f16, kernel_cpy_f16_f32, kernel_cpy_f32_f16, kernel_cpy_f32_f32;
    cl_kernel kernel_mul_mat_f32_f32;
    cl_kernel kernel_mul_mat_f16_f16;
    cl_kernel kernel_mul_mat_f16_f32_1row;
    cl_kernel kernel_mul_mat_f16_f32;
    cl_kernel kernel_mul_mat_f16_f32_l4;
    cl_kernel kernel_mul_mat_f16_f32_tiled;
    cl_kernel kernel_mul_mat_q4_0_f32, kernel_mul_mat_q4_0_f32_v;
    cl_kernel kernel_convert_block_q4_0, kernel_restore_block_q4_0;
    cl_kernel kernel_convert_block_mxfp4, kernel_convert_block_mxfp4_trans, kernel_restore_block_mxfp4, kernel_restore_block_mxfp4_trans;
    cl_kernel kernel_convert_block_q8_0, kernel_restore_block_q8_0;
    cl_kernel kernel_mul_mat_q4_0_f32_8x_flat;
    cl_kernel kernel_convert_block_q4_0_noshuffle;
    cl_kernel kernel_mul_mat_q4_0_f32_1d_8x_flat, kernel_mul_mat_q4_0_f32_1d_16x_flat;
    cl_kernel kernel_mul_mv_q6_K_f32;
    cl_kernel kernel_mul_mv_mxfp4_f32, kernel_mul_mv_mxfp4_f32_flat;
    cl_kernel kernel_mul_mv_q8_0_f32, kernel_mul_mv_q8_0_f32_flat;
    cl_kernel kernel_im2col_f32, kernel_im2col_f16;
    cl_kernel kernel_argsort_f32_i32;
    cl_kernel kernel_sum_rows_f32;
    cl_kernel kernel_repeat;
    cl_kernel kernel_pad;
    cl_kernel kernel_tanh_f32_nd;
    cl_kernel kernel_tanh_f16_nd;
    cl_kernel kernel_upscale;
    cl_kernel kernel_upscale_bilinear;
    cl_kernel kernel_concat_f32_contiguous;
    cl_kernel kernel_concat_f32_non_contiguous;
    cl_kernel kernel_conv_2d_f16;
    cl_kernel kernel_conv_2d_f32;
    cl_kernel kernel_conv_2d_f16_f32;
    cl_kernel kernel_timestep_embedding;
    cl_kernel kernel_gemv_moe_mxfp4_f32, kernel_gemm_moe_mxfp4_f32;
    cl_kernel kernel_mul_mv_id_q4_0_f32_8x_flat;
    cl_kernel kernel_mul_mv_id_q8_0_f32, kernel_mul_mv_id_q8_0_f32_flat;
    cl_kernel kernel_mul_mv_id_mxfp4_f32;
    cl_kernel kernel_mul_mv_id_mxfp4_f32_flat;
    cl_kernel kernel_mul_mm_f32_f32_l4_lm;
    cl_kernel kernel_mul_mm_f16_f32_l4_lm;
    cl_kernel kernel_mul_mm_q8_0_f32_l4_lm;

    std::vector<ProfilingInfo> profiling_info;

    void write_profiling_info() {
        FILE * fperf = fopen("cl_profiling.csv", "w");
        if (!fperf) {
            GGML_LOG_ERROR("Failed to open cl_profiling.csv\n");
            return;
        }

        // Populate profiling info
        for (ProfilingInfo & info : profiling_info) {
            cl_ulong cmd_queued;
            cl_ulong cmd_submit;
            cl_ulong cmd_start;
            cl_ulong cmd_end;
            cl_ulong cmd_complete;

            CL_CHECK(clWaitForEvents(1, &info.evt));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &cmd_queued, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &cmd_submit, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &cmd_start, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &cmd_end, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_COMPLETE, sizeof(cl_ulong), &cmd_complete, NULL));
            CL_CHECK(clReleaseEvent(info.evt));

            char kernel_name[512];
            CL_CHECK(clGetKernelInfo(info.kernel, CL_KERNEL_FUNCTION_NAME,
                sizeof(kernel_name), kernel_name, NULL));
            info.kernel_name = kernel_name;

            info.cmd_queued = cmd_queued;
            info.cmd_submit = cmd_submit;
            info.cmd_start  = cmd_start;
            info.cmd_end    = cmd_end;

            info.cmd_queued_duration_ns     = cmd_submit    - cmd_queued;
            info.cmd_submit_duration_ns     = cmd_start     - cmd_submit;
            info.cmd_duration_ns            = cmd_end       - cmd_start;
            info.cmd_complete_duration_ns   = cmd_complete  - cmd_end;
            info.cmd_total_duration_ns      = cmd_complete  - cmd_queued;
        }

        // Dump a csv
        fprintf(fperf, "op name, kernel name, exec duration (ms), global size, local size, output size\n");
        for (const ProfilingInfo & info : profiling_info) {
            fprintf(fperf, "%s,%s,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\n",
                info.op_name.c_str(), info.kernel_name.c_str(),
                info.cmd_duration_ns/1.e6f,
                info.global_size[0], info.global_size[1], info.global_size[2],
                info.local_size[0], info.local_size[1], info.local_size[2],
                info.output_size[0], info.output_size[1], info.output_size[2], info.output_size[3]);
        }
        fclose(fperf);

        // Dump a simple chrome trace
        FILE* ftrace = fopen("cl_trace.json", "w");
        if (!ftrace) {
            GGML_LOG_ERROR("Failed to open cl_trace.json\n");
            return;
        }

        fprintf(ftrace, "[\n");
        for (const ProfilingInfo & info : profiling_info) {
            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"B\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Host\"},\n",
                info.kernel_name.c_str(), info.cmd_queued/1000);
            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"E\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Host\"},\n",
                info.kernel_name.c_str(), info.cmd_submit/1000);

            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"B\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Device\"},\n",
                info.kernel_name.c_str(), info.cmd_start/1000);
            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"E\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Device\"},\n",
                info.kernel_name.c_str(), info.cmd_end/1000);
        }
        fclose(ftrace);
    }

    size_t get_kernel_workgroup_size(cl_kernel kernel) const {
        size_t workgroup_size = 0;
        size_t ret_size = 0;
        CL_CHECK(
            clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                sizeof(size_t), &workgroup_size, &ret_size));
        GGML_ASSERT(sizeof(size_t) == ret_size);
        return workgroup_size;
    }

    void enqueue_ndrange_kernel(cl_kernel kernel, cl_uint work_dim, size_t *global_work_size, size_t *local_work_size, const ggml_tensor * tensor) {
#ifdef GGML_OPENCL_PROFILING
        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, &evt));

        profiling_info.emplace_back();
        populateProfilingInfo(profiling_info.back(), evt, kernel, work_dim, global_work_size, local_work_size, tensor);
#else
        GGML_UNUSED(tensor);
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL));
#endif
    }

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // Transpose kernels
    cl_program program_transpose;

    cl_kernel kernel_transpose_32;
    cl_kernel kernel_transpose_32_16;
    cl_kernel kernel_transpose_16;
    cl_kernel kernel_transpose_16_4x1;

    cl_mem A_s_d_max;            // max scale buffer size for transpose
    cl_mem A_q_d_max;            // max weight buffer size for transpose
    cl_mem B_d_max;              // max activation buffer size for transpose

    // Gemm and Gemv related programs, kernels, etc
    cl_program program_CL_gemm;
    cl_program program_gemv_noshuffle_q4_0_f32;
    cl_program program_gemv_noshuffle_q4_0_f32_4096_1_11008;
    cl_program program_gemv_noshuffle_q4_0_f32_4096_1_4096;
    cl_program program_gemv_noshuffle_q4_0_f32_11008_1_4096;
    cl_program program_gemv_noshuffle_q4_0_f32_32000_1_4096;
    cl_program program_gemv_noshuffle_q8_0_f32;
    cl_kernel CL_mul_mat_Ab_Bi_8x4;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_dual;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_4096_1_11008;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_4096_1_4096;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_11008_1_4096;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_32000_1_4096;
    cl_kernel kernel_gemv_noshuffle_q8_0_f32;
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    void free() {
        ref_count--;
        if (ref_count == 0) {
#ifdef GGML_OPENCL_PROFILING
            write_profiling_info();
            profiling_info.clear();
#endif
        }
    }
};

// All registered devices with a default device in the front.
static std::vector<ggml_backend_device> g_ggml_backend_opencl_devices;

inline std::string read_file(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs) {
    return "";
  }
  std::string text;
  ifs.seekg(0, std::ios::end);
  text.resize(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  ifs.read(&text[0], text.size());
  return text;
}

static cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer, const std::string &compile_opts) {
    cl_program p;
    char *program_log;
    size_t program_size;
    size_t log_size;
    int err;

    program_size = strlen(program_buffer);

    p = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        GGML_LOG_ERROR("OpenCL error creating program");
        exit(1);
    }

    err = clBuildProgram(p, 0, NULL, compile_opts.c_str(), NULL, NULL);
    if(err < 0) {
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        GGML_LOG_ERROR("ggml_opencl: kernel compile error:\n\n%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return p;
}

static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx, ggml_cl_version opencl_c_version) {
    cl_int err;

    // compiler options for general kernels
    auto opencl_c_std =
        std::string("CL") + std::to_string(opencl_c_version.major) + "." + std::to_string(opencl_c_version.minor);
    std::string compile_opts = std::string("-cl-std=") + opencl_c_std +
                               " -cl-mad-enable -cl-unsafe-math-optimizations"
                               " -cl-finite-math-only -cl-fast-relaxed-math";

    GGML_LOG_INFO("ggml_opencl: loading OpenCL kernels");

    // add
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "add.cl.h"
        };
#else
        const std::string kernel_src = read_file("add.cl");
#endif
        backend_ctx->program_add =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_add         = clCreateKernel(backend_ctx->program_add, "kernel_add", &err), err));
        CL_CHECK((backend_ctx->kernel_add_row     = clCreateKernel(backend_ctx->program_add, "kernel_add_row", &err), err));
        CL_CHECK((backend_ctx->kernel_add_f16     = clCreateKernel(backend_ctx->program_add, "kernel_add_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_add_row_f16 = clCreateKernel(backend_ctx->program_add, "kernel_add_row_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // add_id
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "add_id.cl.h"
        };
#else
        const std::string kernel_src = read_file("add_id.cl");
#endif
        backend_ctx->program_add_id =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_add_id = clCreateKernel(backend_ctx->program_add_id, "kernel_add_id", &err), err));
        GGML_LOG_CONT(".");
    }

    // clamp
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "clamp.cl.h"
        };
#else
        const std::string kernel_src = read_file("clamp.cl");
#endif
        backend_ctx->program_clamp =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_clamp = clCreateKernel(backend_ctx->program_clamp, "kernel_clamp", &err), err));
        GGML_LOG_CONT(".");
    }

    // cpy
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "cpy.cl.h"
        };
#else
        const std::string kernel_src = read_file("cpy.cl");
#endif
        backend_ctx->program_cpy =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_cpy_f16_f16 = clCreateKernel(backend_ctx->program_cpy, "kernel_cpy_f16_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f16_f32 = clCreateKernel(backend_ctx->program_cpy, "kernel_cpy_f16_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f32_f16 = clCreateKernel(backend_ctx->program_cpy, "kernel_cpy_f32_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f32_f32 = clCreateKernel(backend_ctx->program_cpy, "kernel_cpy_f32_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // cvt
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "cvt.cl.h"
        };
#else
        const std::string kernel_src = read_file("cvt.cl");
#endif
        backend_ctx->program_cvt =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_convert_block_q4_0_noshuffle = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_q4_0_noshuffle", &err), err));
        CL_CHECK((backend_ctx->kernel_convert_block_q4_0  = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_q4_0", &err), err));
        CL_CHECK((backend_ctx->kernel_restore_block_q4_0  = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_q4_0", &err), err));
        CL_CHECK((backend_ctx->kernel_convert_block_mxfp4 = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_mxfp4", &err), err));
        CL_CHECK((backend_ctx->kernel_convert_block_mxfp4_trans = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_mxfp4_trans", &err), err));
        CL_CHECK((backend_ctx->kernel_restore_block_mxfp4_trans = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_mxfp4_trans", &err), err));
        CL_CHECK((backend_ctx->kernel_restore_block_mxfp4 = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_mxfp4", &err), err));
        CL_CHECK((backend_ctx->kernel_convert_block_q8_0  = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_q8_0", &err), err));
        CL_CHECK((backend_ctx->kernel_restore_block_q8_0  = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_q8_0", &err), err));
        GGML_LOG_CONT(".");
    }

    // diag_mask_inf
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "diag_mask_inf.cl.h"
        };
#else
        const std::string kernel_src = read_file("diag_mask_inf.cl");
#endif
        backend_ctx->program_diag_mask_inf =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_diag_mask_inf_8 = clCreateKernel(backend_ctx->program_diag_mask_inf, "kernel_diag_mask_inf_8", &err), err));
        CL_CHECK((backend_ctx->kernel_diag_mask_inf   = clCreateKernel(backend_ctx->program_diag_mask_inf, "kernel_diag_mask_inf", &err), err));
        GGML_LOG_CONT(".");
    }

    // gelu
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gelu.cl.h"
        };
#else
        const std::string kernel_src = read_file("gelu.cl");
#endif
        backend_ctx->program_gelu =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_gelu         = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu", &err), err));
        CL_CHECK((backend_ctx->kernel_gelu_4       = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu_4", &err), err));
        CL_CHECK((backend_ctx->kernel_gelu_erf     = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu_erf", &err), err));
        CL_CHECK((backend_ctx->kernel_gelu_erf_4   = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu_erf_4", &err), err));
        CL_CHECK((backend_ctx->kernel_gelu_quick   = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu_quick", &err), err));
        CL_CHECK((backend_ctx->kernel_gelu_quick_4 = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu_quick_4", &err), err));
        GGML_LOG_CONT(".");
    }

    // glu
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "glu.cl.h"
        };
#else
        const std::string kernel_src = read_file("glu.cl");
#endif
        backend_ctx->program_glu =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_geglu           = clCreateKernel(backend_ctx->program_glu, "kernel_geglu", &err), err));
        CL_CHECK((backend_ctx->kernel_reglu           = clCreateKernel(backend_ctx->program_glu, "kernel_reglu", &err), err));
        CL_CHECK((backend_ctx->kernel_swiglu          = clCreateKernel(backend_ctx->program_glu, "kernel_swiglu", &err), err));
        CL_CHECK((backend_ctx->kernel_swiglu_oai      = clCreateKernel(backend_ctx->program_glu, "kernel_swiglu_oai", &err), err));
        CL_CHECK((backend_ctx->kernel_geglu_erf       = clCreateKernel(backend_ctx->program_glu, "kernel_geglu_erf", &err), err));
        CL_CHECK((backend_ctx->kernel_geglu_quick     = clCreateKernel(backend_ctx->program_glu, "kernel_geglu_quick", &err), err));
        CL_CHECK((backend_ctx->kernel_geglu_f16       = clCreateKernel(backend_ctx->program_glu, "kernel_geglu_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_reglu_f16       = clCreateKernel(backend_ctx->program_glu, "kernel_reglu_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_swiglu_f16      = clCreateKernel(backend_ctx->program_glu, "kernel_swiglu_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_geglu_erf_f16   = clCreateKernel(backend_ctx->program_glu, "kernel_geglu_erf_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_geglu_quick_f16 = clCreateKernel(backend_ctx->program_glu, "kernel_geglu_quick_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // get_rows
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "get_rows.cl.h"
        };
#else
        const std::string kernel_src = read_file("get_rows.cl");
#endif
        backend_ctx->program_get_rows =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_get_rows_f32  = clCreateKernel(backend_ctx->program_get_rows, "kernel_get_rows_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_get_rows_f16  = clCreateKernel(backend_ctx->program_get_rows, "kernel_get_rows_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_get_rows_q4_0 = clCreateKernel(backend_ctx->program_get_rows, "kernel_get_rows_q4_0", &err), err));
        GGML_LOG_CONT(".");
    }

    // im2col_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "im2col_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("im2col_f32.cl");
#endif
        backend_ctx->program_im2col_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_im2col_f32 = clCreateKernel(backend_ctx->program_im2col_f32, "kernel_im2col_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // im2col_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "im2col_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("im2col_f16.cl");
#endif
        backend_ctx->program_im2col_f16 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_im2col_f16 = clCreateKernel(backend_ctx->program_im2col_f16, "kernel_im2col_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32.cl");
#endif
        backend_ctx->program_mul_mv_q4_0_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32 = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32, "kernel_mul_mat_q4_0_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_v
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_v.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_v.cl");
#endif
        backend_ctx->program_mul_mv_q4_0_f32_v =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_v = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32_v, "kernel_mul_mat_q4_0_f32_v", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_8x_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_8x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_8x_flat.cl");
#endif
        backend_ctx->program_mul_mv_q4_0_f32_8x_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_8x_flat = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32_8x_flat, "kernel_mul_mat_q4_0_f32_8x_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_1d_8x_flat
    // This kernel does not compiler on Adreno cl compiler 38.01. Skip it for
    // those compiler versions since it is anyway not used for Adreno.
    if (backend_ctx->gpu_family != ADRENO ||
        backend_ctx->adreno_cl_compiler_version.newer_than_or_same(E031, 38, 11, 0) ||
        backend_ctx->adreno_cl_compiler_version.type == DX) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_1d_8x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_1d_8x_flat.cl");
#endif
        backend_ctx->program_mul_mv_q4_0_f32_1d_8x_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_1d_8x_flat = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32_1d_8x_flat, "kernel_mul_mat_q4_0_f32_1d_8x_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_1d_16x_flat
    // This kernel does not compiler on Adreno cl compiler 38.01. Skip it for
    // those compiler versions since it is anyway not used for Adreno.
    if (backend_ctx->gpu_family != ADRENO ||
        backend_ctx->adreno_cl_compiler_version.newer_than_or_same(E031, 38, 11, 0) ||
    backend_ctx->adreno_cl_compiler_version.type == DX) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_1d_16x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_1d_16x_flat.cl");
#endif
        backend_ctx->program_mul_mv_q4_0_f32_1d_16x_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_1d_16x_flat = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32_1d_16x_flat, "kernel_mul_mat_q4_0_f32_1d_16x_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q6_k
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q6_k.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q6_k.cl");
#endif
        backend_ctx->program_mul_mv_q6_K =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_q6_K_f32 = clCreateKernel(backend_ctx->program_mul_mv_q6_K, "kernel_mul_mv_q6_K_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q8_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q8_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q8_0_f32.cl");
#endif
        backend_ctx->program_mul_mv_q8_0_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_q8_0_f32 = clCreateKernel(backend_ctx->program_mul_mv_q8_0_f32, "kernel_mul_mv_q8_0_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q8_0_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q8_0_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q8_0_f32_flat.cl");
#endif
        backend_ctx->program_mul_mv_q8_0_f32_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_q8_0_f32_flat = clCreateKernel(backend_ctx->program_mul_mv_q8_0_f32_flat, "kernel_mul_mv_q8_0_f32_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_mxfp4_f32.cl");
#endif
        backend_ctx->program_mul_mv_mxfp4_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_mxfp4_f32 = clCreateKernel(backend_ctx->program_mul_mv_mxfp4_f32, "kernel_mul_mv_mxfp4_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_mxfp4_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_mxfp4_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_mxfp4_f32_flat.cl");
#endif
        backend_ctx->program_mul_mv_mxfp4_f32_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_mxfp4_f32_flat = clCreateKernel(backend_ctx->program_mul_mv_mxfp4_f32_flat, "kernel_mul_mv_mxfp4_f32_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f16.cl");
#endif
        backend_ctx->program_mul_mv_f16_f16 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f16_f16 = clCreateKernel(backend_ctx->program_mul_mv_f16_f16, "kernel_mul_mat_f16_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f32_1row
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f32_1row.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f32_1row.cl");
#endif
        backend_ctx->program_mul_mv_f16_f32_1row =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32_1row = clCreateKernel(backend_ctx->program_mul_mv_f16_f32_1row, "kernel_mul_mat_f16_f32_1row", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f32_l4
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f32_l4.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f32_l4.cl");
#endif
        backend_ctx->program_mul_mv_f16_f32_l4 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32_l4   = clCreateKernel(backend_ctx->program_mul_mv_f16_f32_l4, "kernel_mul_mat_f16_f32_l4", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f32.cl");
#endif
        backend_ctx->program_mul_mv_f16_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32 = clCreateKernel(backend_ctx->program_mul_mv_f16_f32, "kernel_mul_mat_f16_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f32_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f32_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f32_f32.cl");
#endif
        backend_ctx->program_mul_mv_f32_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f32_f32 = clCreateKernel(backend_ctx->program_mul_mv_f32_f32, "kernel_mul_mat_f32_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mat_f16_f32_tiled
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mat_f16_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mat_f16_f32.cl");
#endif
        backend_ctx->program_mul_mat_f16_f32_tiled =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32_tiled = clCreateKernel(backend_ctx->program_mul_mat_f16_f32_tiled, "mul_mat_f16_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mm_f32_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_f32_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_f32_f32_l4_lm.cl");
#endif
        backend_ctx->program_mul_mm_f32_f32_l4_lm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mm_f32_f32_l4_lm = clCreateKernel(backend_ctx->program_mul_mm_f32_f32_l4_lm, "kernel_mul_mm_f32_f32_l4_lm", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mm_f16_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_f16_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_f16_f32_l4_lm.cl");
#endif
        backend_ctx->program_mul_mm_f16_f32_l4_lm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mm_f16_f32_l4_lm = clCreateKernel(backend_ctx->program_mul_mm_f16_f32_l4_lm, "kernel_mul_mm_f16_f32_l4_lm", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q8_0_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q8_0_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q8_0_f32_l4_lm.cl");
#endif
        backend_ctx->program_mul_mm_q8_0_f32_l4_lm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mm_q8_0_f32_l4_lm = clCreateKernel(backend_ctx->program_mul_mm_q8_0_f32_l4_lm, "kernel_mul_mm_q8_0_f32_l4_lm", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul.cl");
#endif
        backend_ctx->program_mul =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul         = clCreateKernel(backend_ctx->program_mul, "kernel_mul", &err), err));
        CL_CHECK((backend_ctx->kernel_mul_row     = clCreateKernel(backend_ctx->program_mul, "kernel_mul_row", &err), err));
        CL_CHECK((backend_ctx->kernel_mul_f16     = clCreateKernel(backend_ctx->program_mul, "kernel_mul_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_mul_row_f16 = clCreateKernel(backend_ctx->program_mul, "kernel_mul_row_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // norm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "norm.cl.h"
        };
#else
        const std::string kernel_src = read_file("norm.cl");
#endif
        backend_ctx->program_norm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_norm         = clCreateKernel(backend_ctx->program_norm, "kernel_norm", &err), err));
        CL_CHECK((backend_ctx->kernel_norm_mul_add = clCreateKernel(backend_ctx->program_norm, "kernel_norm_mul_add", &err), err));
        GGML_LOG_CONT(".");
    }

    // relu
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "relu.cl.h"
        };
#else
        const std::string kernel_src = read_file("relu.cl");
#endif
        backend_ctx->program_relu =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_relu = clCreateKernel(backend_ctx->program_relu, "kernel_relu", &err), err));
        GGML_LOG_CONT(".");
    }

    // rms_norm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "rms_norm.cl.h"
        };
#else
        const std::string kernel_src = read_file("rms_norm.cl");
#endif
        backend_ctx->program_rms_norm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_rms_norm     = clCreateKernel(backend_ctx->program_rms_norm, "kernel_rms_norm", &err), err));
        CL_CHECK((backend_ctx->kernel_rms_norm_mul = clCreateKernel(backend_ctx->program_rms_norm, "kernel_rms_norm_mul", &err), err));
        GGML_LOG_CONT(".");
    }

    // rope
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "rope.cl.h"
        };
#else
        const std::string kernel_src = read_file("rope.cl");
#endif
        backend_ctx->program_rope =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_rope_norm_f32   = clCreateKernel(backend_ctx->program_rope, "kernel_rope_norm_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_norm_f16   = clCreateKernel(backend_ctx->program_rope, "kernel_rope_norm_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_neox_f32   = clCreateKernel(backend_ctx->program_rope, "kernel_rope_neox_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_neox_f16   = clCreateKernel(backend_ctx->program_rope, "kernel_rope_neox_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_multi_f32  = clCreateKernel(backend_ctx->program_rope, "kernel_rope_multi_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_multi_f16  = clCreateKernel(backend_ctx->program_rope, "kernel_rope_multi_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_vision_f32 = clCreateKernel(backend_ctx->program_rope, "kernel_rope_vision_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_vision_f16 = clCreateKernel(backend_ctx->program_rope, "kernel_rope_vision_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // scale
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "scale.cl.h"
        };
#else
        const std::string kernel_src = read_file("scale.cl");
#endif
        backend_ctx->program_scale =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_scale = clCreateKernel(backend_ctx->program_scale, "kernel_scale", &err), err));
        GGML_LOG_CONT(".");
    }

    // silu
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "silu.cl.h"
        };
#else
        const std::string kernel_src = read_file("silu.cl");
#endif
        backend_ctx->program_silu =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_silu   = clCreateKernel(backend_ctx->program_silu, "kernel_silu", &err), err));
        CL_CHECK((backend_ctx->kernel_silu_4 = clCreateKernel(backend_ctx->program_silu, "kernel_silu_4", &err), err));
        GGML_LOG_CONT(".");
    }

    // softmax_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_f32.cl");
#endif
        backend_ctx->program_softmax_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_soft_max = clCreateKernel(backend_ctx->program_softmax_f32, "kernel_soft_max", &err), err));
        GGML_LOG_CONT(".");
    }

    // softmax_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_f16.cl");
#endif
        backend_ctx->program_softmax_f16 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_soft_max_f16 = clCreateKernel(backend_ctx->program_softmax_f16, "kernel_soft_max_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // softmax_4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_4_f32.cl");
#endif
        backend_ctx->program_softmax_4_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_soft_max_4 = clCreateKernel(backend_ctx->program_softmax_4_f32, "kernel_soft_max_4", &err), err));
        GGML_LOG_CONT(".");
    }

    // softmax_4_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_4_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_4_f16.cl");
#endif
        backend_ctx->program_softmax_4_f16 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_soft_max_4_f16 = clCreateKernel(backend_ctx->program_softmax_4_f16, "kernel_soft_max_4_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // flash_attn
    {
        #ifdef GGML_OPENCL_EMBED_KERNELS
                const std::string kernel_src_f16 {
                    #include "flash_attn_f16.cl.h"
                };
                const std::string kernel_src_f32 {
                    #include "flash_attn_f32.cl.h"
                };
                const std::string kernel_src_f32_f16 {
                    #include "flash_attn_f32_f16.cl.h"
                };
        #else
                const std::string kernel_src_f16 = read_file("flash_attn_f16.cl");
                const std::string kernel_src_f32 = read_file("flash_attn_f32.cl");
                const std::string kernel_src_f32_f16 = read_file("flash_attn_f32_f16.cl");
        #endif

        if (!kernel_src_f16.empty() && !kernel_src_f32.empty() && !kernel_src_f32_f16.empty()) {
            const struct { int dk; int dv; int bm; int bn; } fa_dims[] = {
                { 40,  40, 32, 32}, { 64,  64, 64, 64}, { 80,  80, 64, 32}, { 96,  96, 64, 32},
                {112, 112, 32, 32}, {128, 128, 32, 32}, {192, 128, 16, 16},
                {192, 192, 16, 16}, {256, 256, 16, 16},
            };

            for (size_t i = 0; i < sizeof(fa_dims)/sizeof(fa_dims[0]); ++i) {
                const int dk = fa_dims[i].dk;
                const int dv = fa_dims[i].dv;
                const int bm = fa_dims[i].bm;
                const int bn = fa_dims[i].bn;
                std::string OPTS = compile_opts +
                    " -D DK=" + std::to_string(dk) +
                    " -D DV=" + std::to_string(dv) +
                    " -D BLOCK_M=" + std::to_string(bm) +
                    " -D BLOCK_N=" + std::to_string(bn);

                cl_program prog_f16 = build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_f16.c_str(), OPTS);
                cl_kernel k_f16, k_f16_q1;
                CL_CHECK((k_f16 = clCreateKernel(prog_f16, "flash_attn_f16", &err), err));
                CL_CHECK((k_f16_q1 = clCreateKernel(prog_f16, "flash_attn_f16_q1", &err), err));
                backend_ctx->kernels_flash_attn_f16[{dk, dv}] = k_f16;
                backend_ctx->kernels_flash_attn_f16_q1[{dk, dv}] = k_f16_q1;
                CL_CHECK(clReleaseProgram(prog_f16));

                cl_program prog_f32 = build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_f32.c_str(), OPTS);
                cl_kernel k_f32, k_f32_q1;
                CL_CHECK((k_f32 = clCreateKernel(prog_f32, "flash_attn_f32", &err), err));
                CL_CHECK((k_f32_q1 = clCreateKernel(prog_f32, "flash_attn_f32_q1", &err), err));
                backend_ctx->kernels_flash_attn_f32[{dk, dv}] = k_f32;
                backend_ctx->kernels_flash_attn_f32_q1[{dk, dv}] = k_f32_q1;
                CL_CHECK(clReleaseProgram(prog_f32));

                cl_program prog_f32_f16 = build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_f32_f16.c_str(), OPTS);
                cl_kernel k_f32_f16, k_f32_f16_q1;
                CL_CHECK((k_f32_f16 = clCreateKernel(prog_f32_f16, "flash_attn_f32_f16", &err), err));
                CL_CHECK((k_f32_f16_q1 = clCreateKernel(prog_f32_f16, "flash_attn_f32_f16_q1", &err), err));
                backend_ctx->kernels_flash_attn_f32_f16[{dk, dv}] = k_f32_f16;
                backend_ctx->kernels_flash_attn_f32_f16_q1[{dk, dv}] = k_f32_f16_q1;
                CL_CHECK(clReleaseProgram(prog_f32_f16));

                backend_ctx->kernels_flash_attn_bm[{dk, dv}] = bm;
                backend_ctx->kernels_flash_attn_bn[{dk, dv}] = bn;
            }
            GGML_LOG_CONT(".");
        }
    }

    // argsort
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "argsort.cl.h"
        };
#else
        const std::string kernel_src = read_file("argsort.cl");
#endif
        backend_ctx->program_argsort_f32_i32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_argsort_f32_i32 = clCreateKernel(backend_ctx->program_argsort_f32_i32, "kernel_argsort_f32_i32", &err), err));
        GGML_LOG_CONT(".");
    }

    // div
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "div.cl.h"
        };
#else
        const std::string kernel_src = read_file("div.cl");
#endif
        std::string compile_opts = std::string("-cl-std=") + opencl_c_std +
                               " -cl-mad-enable -cl-finite-math-only ";

        backend_ctx->program_div =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_div         = clCreateKernel(backend_ctx->program_div, "kernel_div", &err), err));
        CL_CHECK((backend_ctx->kernel_div_row     = clCreateKernel(backend_ctx->program_div, "kernel_div_row", &err), err));
        CL_CHECK((backend_ctx->kernel_div_f16     = clCreateKernel(backend_ctx->program_div, "kernel_div_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_div_row_f16 = clCreateKernel(backend_ctx->program_div, "kernel_div_row_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // sub
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sub.cl.h"
        };
#else
        const std::string kernel_src = read_file("sub.cl");
#endif
        backend_ctx->program_sub =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_sub         = clCreateKernel(backend_ctx->program_sub, "kernel_sub", &err), err));
        CL_CHECK((backend_ctx->kernel_sub_row     = clCreateKernel(backend_ctx->program_sub, "kernel_sub_row", &err), err));
        CL_CHECK((backend_ctx->kernel_sub_f16     = clCreateKernel(backend_ctx->program_sub, "kernel_sub_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_sub_row_f16 = clCreateKernel(backend_ctx->program_sub, "kernel_sub_row_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // sum_rows
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sum_rows.cl.h"
        };
#else
        const std::string kernel_src = read_file("sum_rows.cl");
#endif
        backend_ctx->program_sum_rows_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_sum_rows_f32 = clCreateKernel(backend_ctx->program_sum_rows_f32, "kernel_sum_rows_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // sigmoid
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sigmoid.cl.h"
        };
#else
        const std::string kernel_src = read_file("sigmoid.cl");
#endif
        backend_ctx->program_sigmoid =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_sigmoid_f32 = clCreateKernel(backend_ctx->program_sigmoid, "kernel_sigmoid_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_sigmoid_f16 = clCreateKernel(backend_ctx->program_sigmoid, "kernel_sigmoid_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // group_norm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "group_norm.cl.h"
        };
#else
        const std::string kernel_src = read_file("group_norm.cl");
#endif
        backend_ctx->program_group_norm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_group_norm         = clCreateKernel(backend_ctx->program_group_norm, "kernel_group_norm", &err), err));
        CL_CHECK((backend_ctx->kernel_group_norm_mul_add = clCreateKernel(backend_ctx->program_group_norm, "kernel_group_norm_mul_add", &err), err));
        GGML_LOG_CONT(".");
    }

    // repeat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "repeat.cl.h"
        };
#else
        const std::string kernel_src = read_file("repeat.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_repeat =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_repeat = clCreateKernel(backend_ctx->program_repeat, "kernel_repeat", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: repeat kernel source not found or empty. Repeat operations will not be available.\n");
            backend_ctx->program_repeat = nullptr;
            backend_ctx->kernel_repeat = nullptr;
        }
    }

    // pad
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "pad.cl.h"
        };
#else
        const std::string kernel_src = read_file("pad.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_pad =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_pad = clCreateKernel(backend_ctx->program_pad, "kernel_pad", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: pad kernel source not found or empty. Pad operations will not be available.\n");
            backend_ctx->program_pad = nullptr;
            backend_ctx->kernel_pad = nullptr;
        }
    }

    // tanh
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "tanh.cl.h"
        };
#else
        const std::string kernel_src = read_file("tanh.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_tanh =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_tanh_f32_nd = clCreateKernel(backend_ctx->program_tanh, "kernel_tanh_f32_nd", &err), err));
            CL_CHECK((backend_ctx->kernel_tanh_f16_nd = clCreateKernel(backend_ctx->program_tanh, "kernel_tanh_f16_nd", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: tanh kernel source not found or empty. Tanh operation will not be available.\n");
            backend_ctx->program_tanh = nullptr;
            backend_ctx->kernel_tanh_f32_nd = nullptr;
            backend_ctx->kernel_tanh_f16_nd = nullptr;
        }
    }

    // upscale
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "upscale.cl.h"
        };
#else
        const std::string kernel_src = read_file("upscale.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_upscale =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_upscale = clCreateKernel(backend_ctx->program_upscale, "kernel_upscale", &err), err));
            if (backend_ctx->program_upscale) {
                 cl_int err_bilinear;
                 backend_ctx->kernel_upscale_bilinear = clCreateKernel(backend_ctx->program_upscale, "kernel_upscale_bilinear", &err_bilinear);
                 if (err_bilinear != CL_SUCCESS) {
                    GGML_LOG_WARN("ggml_opencl: kernel_upscale_bilinear not found in upscale.cl. Bilinear upscale will not be available. Error: %d\n", err_bilinear);
                    backend_ctx->kernel_upscale_bilinear = nullptr;
                 }
            } else {
                backend_ctx->kernel_upscale_bilinear = nullptr;
            }
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: upscale kernel source not found or empty. Upscale operations will not be available.\n");
            backend_ctx->program_upscale = nullptr;
            backend_ctx->kernel_upscale = nullptr;
            backend_ctx->kernel_upscale_bilinear = nullptr;
        }
    }

    // concat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "concat.cl.h"
        };
#else

        const std::string kernel_src = read_file("concat.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_concat =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

            CL_CHECK((backend_ctx->kernel_concat_f32_contiguous = clCreateKernel(backend_ctx->program_concat, "kernel_concat_f32_contiguous", &err), err));
            CL_CHECK((backend_ctx->kernel_concat_f32_non_contiguous = clCreateKernel(backend_ctx->program_concat, "kernel_concat_f32_non_contiguous", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: concat kernel source not found or empty. Concat operations will not be available.\n");
            backend_ctx->program_concat = nullptr;
            backend_ctx->kernel_concat_f32_contiguous = nullptr;
            backend_ctx->kernel_concat_f32_non_contiguous = nullptr;
        }
    }

    // timestep_embedding
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "tsembd.cl.h"
        };
#else

        const std::string kernel_src = read_file("tsembd.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_tsembd =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_timestep_embedding = clCreateKernel(backend_ctx->program_tsembd, "kernel_timestep_embedding", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: timestep_embedding kernel source not found or empty. This op will not be available.\n");
            backend_ctx->program_tsembd = nullptr;
            backend_ctx->kernel_timestep_embedding = nullptr;
        }
    }

    // set_rows
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "set_rows.cl.h"
        };
#else
        const std::string kernel_src = read_file("set_rows.cl");
#endif
        backend_ctx->program_set_rows =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_set_rows_f32_i64 = clCreateKernel(backend_ctx->program_set_rows, "kernel_set_rows_f32_i64", &err), err));
        CL_CHECK((backend_ctx->kernel_set_rows_f32_i32 = clCreateKernel(backend_ctx->program_set_rows, "kernel_set_rows_f32_i32", &err), err));
        CL_CHECK((backend_ctx->kernel_set_rows_f16_i64 = clCreateKernel(backend_ctx->program_set_rows, "kernel_set_rows_f16_i64", &err), err));
        CL_CHECK((backend_ctx->kernel_set_rows_f16_i32 = clCreateKernel(backend_ctx->program_set_rows, "kernel_set_rows_f16_i32", &err), err));
        GGML_LOG_CONT(".");
    }

     // conv2d
     {
        #ifdef GGML_OPENCL_EMBED_KERNELS
                const std::string kernel_src {
                    #include "conv2d.cl.h"
                };
                const std::string kernel_src_f16_f32 {
                    #include "conv2d_f16_f32.cl.h"
                };
        #else
                const std::string kernel_src = read_file("conv2d.cl");
                const std::string kernel_src_f16_f32 = read_file("conv2d_f16_f32.cl");
        #endif
                if (!kernel_src.empty()) {
                    backend_ctx->program_conv_2d_f16 =
                        build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), (std::string(compile_opts) + " -DUSE_FP16=1").c_str());
                    CL_CHECK((backend_ctx->kernel_conv_2d_f16 = clCreateKernel(backend_ctx->program_conv_2d_f16, "kernel_conv_2d", &err), err));
                    GGML_LOG_CONT(".");
                    backend_ctx->program_conv_2d_f32 =
                        build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
                    CL_CHECK((backend_ctx->kernel_conv_2d_f32 = clCreateKernel(backend_ctx->program_conv_2d_f32, "kernel_conv_2d", &err), err));
                    GGML_LOG_CONT(".");
                } else {
                    GGML_LOG_WARN("ggml_opencl: conv2d kernel source not found or empty. This op will not be available.\n");
                    backend_ctx->program_conv_2d_f16 = nullptr;
                    backend_ctx->kernel_conv_2d_f16 = nullptr;
                    backend_ctx->program_conv_2d_f32 = nullptr;
                    backend_ctx->kernel_conv_2d_f32 = nullptr;
                }
                if (!kernel_src_f16_f32.empty()) {
                    backend_ctx->program_conv_2d_f16_f32 =
                        build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_f16_f32.c_str(), compile_opts);
                    CL_CHECK((backend_ctx->kernel_conv_2d_f16_f32 = clCreateKernel(backend_ctx->program_conv_2d_f16_f32, "kernel_conv_2d", &err), err));
                    GGML_LOG_CONT(".");
                } else {
                    GGML_LOG_WARN("ggml_opencl: conv2d_f16_f32 kernel source not found or empty. This op will not be available.\n");
                    backend_ctx->program_conv_2d_f16_f32 = nullptr;
                    backend_ctx->kernel_conv_2d_f16_f32 = nullptr;
                }
    }

    // mul_mv_id_q4_0_f32_8x_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_q4_0_f32_8x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_q4_0_f32_8x_flat.cl");
#endif
        backend_ctx->program_mul_mv_id_q4_0_f32_8x_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_id_q4_0_f32_8x_flat = clCreateKernel(backend_ctx->program_mul_mv_id_q4_0_f32_8x_flat, "kernel_mul_mv_id_q4_0_f32_8x_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_q8_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_q8_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_q8_0_f32.cl");
#endif
        backend_ctx->program_mul_mv_id_q8_0_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_id_q8_0_f32 = clCreateKernel(backend_ctx->program_mul_mv_id_q8_0_f32, "kernel_mul_mv_id_q8_0_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_q8_0_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_q8_0_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_q8_0_f32_flat.cl");
#endif
        backend_ctx->program_mul_mv_id_q8_0_f32_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_id_q8_0_f32_flat = clCreateKernel(backend_ctx->program_mul_mv_id_q8_0_f32_flat, "kernel_mul_mv_id_q8_0_f32_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_mxfp4_f32.cl");
#endif
        backend_ctx->program_mul_mv_id_mxfp4_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_id_mxfp4_f32 = clCreateKernel(backend_ctx->program_mul_mv_id_mxfp4_f32, "kernel_mul_mv_id_mxfp4_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_mxfp4_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_mxfp4_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_mxfp4_f32_flat.cl");
#endif
        backend_ctx->program_mul_mv_id_mxfp4_f32_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_id_mxfp4_f32_flat = clCreateKernel(backend_ctx->program_mul_mv_id_mxfp4_f32_flat, "kernel_mul_mv_id_mxfp4_f32_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // Adreno kernels
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // transpose
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "transpose.cl.h"
        };
#else
        const std::string kernel_src = read_file("transpose.cl");
#endif
        backend_ctx->program_transpose =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_transpose_32_16 = clCreateKernel(backend_ctx->program_transpose, "kernel_transpose_32_16", &err), err));
        CL_CHECK((backend_ctx->kernel_transpose_32    = clCreateKernel(backend_ctx->program_transpose, "kernel_transpose_32", &err), err));
        CL_CHECK((backend_ctx->kernel_transpose_16    = clCreateKernel(backend_ctx->program_transpose, "kernel_transpose_16", &err), err));
        CL_CHECK((backend_ctx->kernel_transpose_16_4x1    = clCreateKernel(backend_ctx->program_transpose, "kernel_transpose_16_4x1", &err), err));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q4_0_f32
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable "
                                       " -DN_SIMDGROUP=4 "
                                       " -DSIMDGROUP_WIDTH=" +
                                       std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemv_general {
            #include "gemv_noshuffle_q4_0_f32.cl.h"
        };
#else
        const std::string kernel_src_CL_gemv_general = read_file("gemv_noshuffle_q4_0_f32.cl");
#endif

        backend_ctx->program_gemv_noshuffle_q4_0_f32 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv_general.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->kernel_gemv_noshuffle_q4_0_f32 = clCreateKernel(backend_ctx->program_gemv_noshuffle_q4_0_f32, "kernel_gemv_noshuffle_q4_0_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_gemv_noshuffle_q4_0_f32_dual = clCreateKernel(backend_ctx->program_gemv_noshuffle_q4_0_f32, "kernel_gemv_noshuffle_q4_0_f32_dual", &err), err));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q4_0_f32_spec
    {
        // Gemv 2048, 16384
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=2048 "
            " -DBLOCK_STRIDE_A=16384 "
            " -DN_SIMDGROUP=4 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemv {
            #include "gemv_noshuffle_q4_0_f32_spec.cl.h"
        };
#else
        const std::string kernel_src_CL_gemv = read_file("gemv_noshuffle_q4_0_f32_spec.cl");
#endif

        backend_ctx->program_gemv_noshuffle_q4_0_f32_4096_1_4096 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->kernel_gemv_noshuffle_q4_0_f32_4096_1_4096 = clCreateKernel(backend_ctx->program_gemv_noshuffle_q4_0_f32_4096_1_4096, "kernel_gemv_noshuffle_q4_0_f32", &err), err));
        GGML_LOG_CONT(".");

        // Gemv 2048, 16384
        CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=2048 "
            " -DBLOCK_STRIDE_A=16384 "
            " -DN_SIMDGROUP=4 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

        backend_ctx->program_gemv_noshuffle_q4_0_f32_4096_1_11008 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->kernel_gemv_noshuffle_q4_0_f32_4096_1_11008 = clCreateKernel(backend_ctx->program_gemv_noshuffle_q4_0_f32_4096_1_11008, "kernel_gemv_noshuffle_q4_0_f32", &err), err));
        GGML_LOG_CONT(".");

        // Gemv 5504, 44032
        CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=5504 "
            " -DBLOCK_STRIDE_A=44032 "
            " -DN_SIMDGROUP=4 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

        backend_ctx->program_gemv_noshuffle_q4_0_f32_11008_1_4096 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->kernel_gemv_noshuffle_q4_0_f32_11008_1_4096 = clCreateKernel(backend_ctx->program_gemv_noshuffle_q4_0_f32_11008_1_4096, "kernel_gemv_noshuffle_q4_0_f32", &err), err));
        GGML_LOG_CONT(".");

        // Gemv 16000, 128000
        CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=16000 "
            " -DBLOCK_STRIDE_A=128000 "
            " -DN_SIMDGROUP=4 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);

        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

        backend_ctx->program_gemv_noshuffle_q4_0_f32_32000_1_4096 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->kernel_gemv_noshuffle_q4_0_f32_32000_1_4096 = clCreateKernel(backend_ctx->program_gemv_noshuffle_q4_0_f32_32000_1_4096, "kernel_gemv_noshuffle_q4_0_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q8_0_f32
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable "
                                       " -DSIMDGROUP_WIDTH=" +
                                       std::to_string(backend_ctx->adreno_wave_size);

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemv_q8_0 {
            #include "gemv_noshuffle_q8_0_f32.cl.h"
        };
#else
        const std::string kernel_src_CL_gemv_q8_0 = read_file("gemv_noshuffle_q8_0_f32.cl");
#endif

        backend_ctx->program_gemv_noshuffle_q8_0_f32 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv_q8_0.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->kernel_gemv_noshuffle_q8_0_f32 = clCreateKernel(backend_ctx->program_gemv_noshuffle_q8_0_f32, "kernel_gemv_noshuffle_q8_0_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mat_Ab_Bi_8x4
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemm {
            #include "mul_mat_Ab_Bi_8x4.cl.h"
        };
#else
        const std::string kernel_src_CL_gemm = read_file("mul_mat_Ab_Bi_8x4.cl");
#endif
        backend_ctx->program_CL_gemm = build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_CL_gemm.c_str(), compile_opts);
        CL_CHECK((backend_ctx->CL_mul_mat_Ab_Bi_8x4 = clCreateKernel(backend_ctx->program_CL_gemm, "kernel_mul_mat_Ab_Bi_8x4", &err), err));
        GGML_LOG_CONT(".");
    }

    std::string CL_moe_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -cl-fast-relaxed-math";

    // gemv_moe_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_mxfp4_f32.cl");
#endif
        backend_ctx->program_gemv_moe_mxfp4_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->kernel_gemv_moe_mxfp4_f32 = clCreateKernel(backend_ctx->program_gemv_moe_mxfp4_f32, "kernel_gemv_moe_mxfp4_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_mxfp4_f32.cl");
#endif
        backend_ctx->program_gemm_moe_mxfp4_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->kernel_gemm_moe_mxfp4_f32 = clCreateKernel(backend_ctx->program_gemm_moe_mxfp4_f32, "kernel_gemm_moe_mxfp4_f32", &err), err));
        GGML_LOG_CONT(".");
    }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
    GGML_LOG_CONT("\n");
}

// XXX static ggml_backend_opencl_context * ggml_cl2_init(ggml_backend_dev_t dev) {
// XXX    static bool initialized = false;
// XXX    static ggml_backend_opencl_context *backend_ctx = nullptr;

static ggml_backend_opencl_context * ggml_cl2_init(ggml_backend_dev_t dev);

namespace /* anonymous */ {
extern struct ggml_backend_device_i ggml_backend_opencl_device_i;
}

// Look for available and suitable devices.
static std::vector<ggml_backend_device> ggml_opencl_probe_devices(ggml_backend_reg * reg) {
    std::vector<ggml_backend_device> found_devices;

#ifdef GGML_OPENCL_PROFILING
    GGML_LOG_INFO("ggml_opencl: OpenCL profiling enabled\n");
#endif

    struct cl_device;
    struct cl_platform {
        cl_platform_id id;
        unsigned number;
        char name[128];
        char vendor[128];
        struct cl_device * devices;
        unsigned n_devices;
        struct cl_device * default_device;
    };

    struct cl_device {
        struct cl_platform * platform;
        cl_device_id id;
        unsigned number;
        cl_device_type type;
        char name[128];
        char version[128];
    };

    enum { NPLAT = 16, NDEV = 16 };

    struct cl_platform platforms[NPLAT];
    unsigned n_platforms = 0;
    struct cl_device devices[NDEV];
    unsigned n_devices = 0;
    struct cl_device * default_device = NULL;
    unsigned           default_platform_number = 0;

    cl_platform_id platform_ids[NPLAT];
    if (clGetPlatformIDs(NPLAT, platform_ids, &n_platforms) != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml_opencl: plaform IDs not available.\n");
        return found_devices;
    }

    for (unsigned i = 0; i < n_platforms; i++) {
        struct cl_platform * p = &platforms[i];
        p->number = i;
        p->id = platform_ids[i];
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_NAME, sizeof(p->name), &p->name, NULL));
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_VENDOR, sizeof(p->vendor), &p->vendor, NULL));

        cl_device_id device_ids[NDEV];
        cl_int clGetDeviceIDsError = clGetDeviceIDs(p->id, CL_DEVICE_TYPE_ALL, NDEV, device_ids, &p->n_devices);
        if (clGetDeviceIDsError == CL_DEVICE_NOT_FOUND) {
            p->n_devices = 0;
        } else {
            CL_CHECK(clGetDeviceIDsError);
        }
        p->devices = p->n_devices > 0 ? &devices[n_devices] : NULL;
        p->default_device = NULL;

        for (unsigned j = 0; j < p->n_devices; j++) {
            struct cl_device * d = &devices[n_devices];
            d->number = n_devices++;
            d->id = device_ids[j];
            d->platform = p;
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_NAME, sizeof(d->name), &d->name, NULL));
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_TYPE, sizeof(d->type), &d->type, NULL));
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_VERSION, sizeof(d->version), &d->version, NULL));

            if (p->default_device == NULL && d->type == CL_DEVICE_TYPE_GPU) {
                p->default_device = d;
            }
        }

        if (default_device == NULL && p->default_device != NULL) {
            default_device          = p->default_device;
            default_platform_number = i;
        }
    }

    if (n_devices == 0) {
        GGML_LOG_ERROR("ggml_opencl: could find any OpenCL devices.\n");
        return found_devices;
    }

    char *      user_platform_string = getenv("GGML_OPENCL_PLATFORM");
    char *      user_device_string   = getenv("GGML_OPENCL_DEVICE");
    int         user_platform_number = -1;
    int         user_device_number   = -1;
    cl_device * candidate_devices    = nullptr;
    unsigned    n_candidate_devices  = 0;

    unsigned n;
    if (user_platform_string != NULL && sscanf(user_platform_string, " %u", &n) == 1 && n < n_platforms) {
        user_platform_number = (int)n;
    }
    if (user_device_string != NULL && sscanf(user_device_string, " %u", &n) == 1 && n < n_devices) {
        user_device_number = (int)n;
    }
    if (user_platform_number != -1 && user_device_number != -1) {
        cl_platform* platform = &platforms[user_platform_number];
        if ((unsigned)user_device_number >= platform->n_devices) {
            GGML_LOG_ERROR("ggml_opencl: invalid device number %d\n", user_device_number);
            exit(1);
        }
        default_device      = &platform->devices[user_device_number];
        candidate_devices   = platform->devices;
        n_candidate_devices = platform->n_devices;
    } else {
        // Choose a platform by matching a substring.
        if (user_platform_number == -1 && user_platform_string != NULL && user_platform_string[0] != 0) {
            for (unsigned i = 0; i < n_platforms; i++) {
                struct cl_platform * p = &platforms[i];
                if (strstr(p->name, user_platform_string) != NULL ||
                    strstr(p->vendor, user_platform_string) != NULL) {
                    user_platform_number = (int)i;
                    break;
                }
            }
            if (user_platform_number == -1) {
                GGML_LOG_ERROR("ggml_opencl: no platform matching '%s' was found.\n", user_platform_string);
                exit(1);
            }
        }

        int                  platform_idx = user_platform_number != -1 ? user_platform_number : default_platform_number;
        struct cl_platform * p            = &platforms[platform_idx];
        candidate_devices                 = p->devices;
        n_candidate_devices               = p->n_devices;
        default_device                    = p->default_device;
        if (n_candidate_devices == 0) {
            GGML_LOG_ERROR("ggml_opencl: selected platform '%s' does not have any devices.\n", p->name);
            exit(1);
        }

        if (user_device_number == -1 && user_device_string != NULL && user_device_string[0] != 0) {
            for (unsigned i = 0; i < n_candidate_devices; i++) {
                struct cl_device * d = &candidate_devices[i];
                if (strstr(d->name, user_device_string) != NULL) {
                    user_device_number = d->number;
                    break;
                }
            }
            if (user_device_number == -1) {
                GGML_LOG_ERROR("ggml_opencl: no device matching '%s' was found.\n", user_device_string);
                exit(1);
            }
        }
        if (user_device_number != -1) {
            candidate_devices   = &devices[user_device_number];
            n_candidate_devices = 1;
            default_device      = &candidate_devices[0];
        }

        GGML_ASSERT(n_candidate_devices > 0);

        if (default_device == NULL) {
            default_device = &candidate_devices[0];
        }
    }

    GGML_ASSERT(n_candidate_devices != 0 && candidate_devices);

    // Put the default device in front.
    for (unsigned i = 1; i < n_candidate_devices; i++) {
        if (&candidate_devices[i] == default_device) {
            std::swap(candidate_devices[0], candidate_devices[i]);
            default_device = &candidate_devices[0];
            break;
        }
    }

    GGML_LOG_INFO("ggml_opencl: selected platform: '%s'\n", default_device->platform->name);

    std::vector<cl_device_id> device_ids;
    for (auto dev = candidate_devices, dev_end = candidate_devices + n_candidate_devices; dev != dev_end; dev++) {
        device_ids.push_back(dev->id);
    }

    cl_int                err;
    cl_context            shared_context;
    cl_context_properties properties[] = { (intptr_t) CL_CONTEXT_PLATFORM, (intptr_t) default_device->platform->id, 0 };

    CL_CHECK(
        (shared_context = clCreateContext(properties, device_ids.size(), device_ids.data(), NULL, NULL, &err), err));

    for (auto dev = candidate_devices, dev_end = candidate_devices + n_candidate_devices; dev != dev_end; dev++) {
        GGML_LOG_INFO("\nggml_opencl: device: '%s (%s)'\n", dev->name, dev->version);

        auto dev_ctx = std::unique_ptr<ggml_backend_opencl_device_context>(new ggml_backend_opencl_device_context{
            /*.platform         =*/dev->platform->id,
            /*.platform_nane    =*/dev->platform->name,
            /*.device           =*/dev->id,
            /*.device_name      =*/dev->name,
            /*.device_type      =*/dev->type,
            /*.device_version   =*/dev->version,
            /*.backend_ctx      =*/nullptr,
            /*.buffer_type      =*/{},
            /*.host_buffer_type =*/{},
            /*.context          =*/shared_context,
        });

        found_devices.push_back(ggml_backend_device{
            /* .iface   = */ ggml_backend_opencl_device_i,
            /* .reg     = */ reg,
            /* .context = */ dev_ctx.get(),
        });

        if (!ggml_cl2_init(&found_devices.back())) {
            found_devices.pop_back();
            GGML_LOG_INFO("ggml_opencl: drop unsupported device.\n");
            continue;
        }

        dev_ctx.release();
    }

    if (found_devices.size()) {
        auto * dev_ctx = static_cast<ggml_backend_opencl_device_context *>(found_devices.front().context);
        GGML_LOG_INFO("ggml_opencl: default device: '%s (%s)'\n", dev_ctx->device_name.c_str(),
                      dev_ctx->device_version.c_str());

        if (dev_ctx->device_type != CL_DEVICE_TYPE_GPU) {
            GGML_LOG_WARN("ggml_opencl: warning, the default device is not a GPU: '%s'.\n",
                          dev_ctx->device_name.c_str());
        }
    }

    return found_devices;
}

// Initialize device if it is supported (returns nullptr if it is not).
static ggml_backend_opencl_context * ggml_cl2_init(ggml_backend_dev_t dev) {
    GGML_ASSERT(dev);
    GGML_ASSERT(dev->context);

    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) dev->context;
    GGML_ASSERT(dev_ctx->platform);
    GGML_ASSERT(dev_ctx->device);

    if (dev_ctx->backend_ctx) {
        return dev_ctx->backend_ctx;
    }

    auto backend_ctx        = std::make_unique<ggml_backend_opencl_context>();
    backend_ctx->device     = dev_ctx->device;
    backend_ctx->gpu_family = GPU_FAMILY::UNKNOWN;

    // ref_count get increased in ggml_backend_opencl_device_init
    // This function is also used to retrieve backend context, so we don't want
    // to increase ref_count for each call. We only want to increase ref_count
    // when the associated device is initialized
    backend_ctx->ref_count  = 0;

    if (strstr(dev_ctx->device_name.c_str(), "Adreno") ||
        strstr(dev_ctx->device_name.c_str(), "Qualcomm") ||
        strstr(dev_ctx->device_version.c_str(), "Adreno")) {
        backend_ctx->gpu_family = GPU_FAMILY::ADRENO;
        // Usually device version contains the detailed device name
        backend_ctx->adreno_gen = get_adreno_gpu_gen(dev_ctx->device_version.c_str());
        if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::ADRENO_UNKNOWN) {
            backend_ctx->adreno_gen = get_adreno_gpu_gen(dev_ctx->device_name.c_str());
        }

        // Use wave size of 64 for all Adreno GPUs.
        backend_ctx->adreno_wave_size = 64;
    } else if (strstr(dev_ctx->device_name.c_str(), "Intel")) {
        backend_ctx->gpu_family = GPU_FAMILY::INTEL;
    } else {
        GGML_LOG_ERROR("Unsupported GPU: %s\n", dev_ctx->device_name.c_str());
        backend_ctx->gpu_family = GPU_FAMILY::UNKNOWN;
        return nullptr;
    }

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (backend_ctx->gpu_family != GPU_FAMILY::ADRENO) {
        GGML_LOG_ERROR("ggml_opencl: Adreno-specific kernels should not be enabled for non-Adreno GPUs; "
            "run on an Adreno GPU or recompile with CMake option `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`\n");
        return nullptr;
    }
#endif

    // Populate backend device name
    backend_ctx->device_name = dev_ctx->device_name;

    // A local ref of cl_device_id for convenience
    cl_device_id device = backend_ctx->device;

    ggml_cl_version platform_version = get_opencl_platform_version(dev_ctx->platform);

    // Check device OpenCL version, OpenCL 2.0 or above is required
    ggml_cl_version opencl_c_version = get_opencl_c_version(platform_version, device);
    if (opencl_c_version.major < 2) {
        GGML_LOG_ERROR("ggml_opencl: OpenCL 2.0 or above is required\n");
        return nullptr;
    }

    // Check driver version
    size_t driver_version_str_size;
    clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &driver_version_str_size);
    char *driver_version = (char *)alloca(driver_version_str_size + 1);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, driver_version_str_size, driver_version, NULL);
    driver_version[driver_version_str_size] = '\0';
    GGML_LOG_INFO("ggml_opencl: OpenCL driver: %s\n", driver_version);
    backend_ctx->driver_version = driver_version;

    backend_ctx->adreno_cl_compiler_version = get_adreno_cl_compiler_version(driver_version);
    backend_ctx->has_vector_subgroup_broadcast =
        (backend_ctx->adreno_cl_compiler_version.type == E031 && backend_ctx->adreno_cl_compiler_version.major >= 47) ||
        (backend_ctx->adreno_cl_compiler_version.type == DX   && backend_ctx->adreno_cl_compiler_version.major >= 17);
    GGML_LOG_INFO("ggml_opencl: vector subgroup broadcast support: %s\n",
        backend_ctx->has_vector_subgroup_broadcast ? "true" : "false");

    size_t ext_str_size;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_str_size);
    char *ext_buffer = (char *)alloca(ext_str_size + 1);
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_str_size, ext_buffer, NULL);
    ext_buffer[ext_str_size] = '\0'; // ensure it is null terminated
    // Check if ext_buffer contains cl_khr_fp16
    backend_ctx->fp16_support = strstr(ext_buffer, "cl_khr_fp16") != NULL;
    GGML_LOG_INFO("ggml_opencl: device FP16 support: %s\n", backend_ctx->fp16_support ? "true" : "false");

    // fp16 is required
    if (!backend_ctx->fp16_support) {
        GGML_LOG_ERROR("ggml_opencl: device does not support FP16\n");
        return nullptr;
    }

    // If OpenCL 3.0 is supported, then check for cl_khr_subgroups, which becomes
    // optional in OpenCL 3.0 (cl_khr_subgroup is mandatory in OpenCL 2.x)
    if (opencl_c_version.major == 3 && strstr(ext_buffer, "cl_khr_subgroups") == NULL &&
        strstr(ext_buffer, "cl_intel_subgroups") == NULL) {
        GGML_LOG_ERROR("ggml_opencl: device does not support subgroups (cl_khr_subgroups or cl_intel_subgroups) "
            "(note that subgroups is an optional feature in OpenCL 3.0)\n");
        return nullptr;
    }

    cl_uint base_align_in_bits;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &base_align_in_bits, NULL));
    GGML_ASSERT(base_align_in_bits % 8u == 0);
    backend_ctx->alignment = base_align_in_bits / 8u;
    GGML_LOG_INFO("ggml_opencl: mem base addr align: %u\n", backend_ctx->alignment);

    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &backend_ctx->max_alloc_size, NULL);
    GGML_LOG_INFO("ggml_opencl: max mem alloc size: %zu MB\n", backend_ctx->max_alloc_size/1024/1024);

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &backend_ctx->max_workgroup_size, NULL);
    GGML_LOG_INFO("ggml_opencl: device max workgroup size: %lu\n", backend_ctx->max_workgroup_size);

    // Check SVM.
    cl_device_svm_capabilities svm_caps;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &svm_caps, 0));
    GGML_LOG_INFO("ggml_opencl: SVM coarse grain buffer support: %s\n",
        svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM fine grain buffer support: %s\n",
        svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM fine grain system support: %s\n",
        svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM atomics support: %s\n",
        svm_caps & CL_DEVICE_SVM_ATOMICS ? "true" : "false");

    if (opencl_c_version.major >= 3) {
        // Assume it is not available for 3.0, since it is optional in 3.0.
        // If compiling against 3.0, then we can query.
        backend_ctx->non_uniform_workgroups = false;
#if CL_TARGET_OPENCL_VERSION >= 300
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, sizeof(cl_bool),
                                 &backend_ctx->non_uniform_workgroups, 0));
#endif
    } else {
        GGML_ASSERT(opencl_c_version.major == 2);
        // Non-uniform workgroup sizes is mandatory feature in v2.x.
        backend_ctx->non_uniform_workgroups = true;
    }

    // Print out configurations
#ifdef GGML_OPENCL_SOA_Q
    GGML_LOG_INFO("ggml_opencl: flattening quantized weights representation as struct of arrays (GGML_OPENCL_SOA_Q)\n");
#endif // GGML_OPENCL_SOA_Q

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    GGML_LOG_INFO("ggml_opencl: using kernels optimized for Adreno (GGML_OPENCL_USE_ADRENO_KERNELS)\n");
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    cl_int err;

    // A local ref of cl_context for convenience
    cl_context context = backend_ctx->context = dev_ctx->context;

    //CL_CHECK((queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err),
    //    (err != CL_INVALID_QUEUE_PROPERTIES && err != CL_INVALID_VALUE ? err :
    //    (queue = clCreateCommandQueue(context, device, 0, &err), err)
    //)));
    cl_command_queue_properties command_queue_props = 0;
#ifdef GGML_OPENCL_PROFILING
    command_queue_props |= CL_QUEUE_PROFILING_ENABLE;
#endif
    if (const char *e = std::getenv("GGML_ELASTIC_DEVICE_TIMING"); e && *e && *e != '0') {
        command_queue_props |= CL_QUEUE_PROFILING_ENABLE;
    }
    CL_CHECK((backend_ctx->queue = clCreateCommandQueue(context, device, command_queue_props, &err), err));

    // Load kernels
    load_cl_kernels(backend_ctx.get(), opencl_c_version);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // Allocate intermediate buffers and images
    size_t required_A_q_d_bytes = 311164928;
    size_t required_A_s_d_bytes = 38895616;
    size_t required_B_d_bytes = 45088768;

    // Ensure buffer sizes do not exceed the maximum allocation size
    size_t max_A_q_d_bytes = MIN(required_A_q_d_bytes, backend_ctx->max_alloc_size);
    size_t max_A_s_d_bytes = MIN(required_A_s_d_bytes, backend_ctx->max_alloc_size);
    size_t max_B_d_bytes   = MIN(required_B_d_bytes, backend_ctx->max_alloc_size);
    if (required_A_q_d_bytes > backend_ctx->max_alloc_size) {
        GGML_LOG_WARN("ggml_opencl: A_q_d buffer size reduced from %zu to %zu due to device limitations.\n",
                      required_A_q_d_bytes, max_A_q_d_bytes);
    }
    if (required_A_s_d_bytes > backend_ctx->max_alloc_size) {
        GGML_LOG_WARN("ggml_opencl: A_s_d buffer size reduced from %zu to %zu due to device limitations.\n",
                      required_A_s_d_bytes, max_A_s_d_bytes);
    }
    if (required_B_d_bytes > backend_ctx->max_alloc_size) {
        GGML_LOG_WARN("ggml_opencl: B_d buffer size reduced from %zu to %zu due to device limitations.\n",
                      required_B_d_bytes, max_B_d_bytes);
    }

    CL_CHECK((backend_ctx->A_q_d_max = clCreateBuffer(context, 0, max_A_q_d_bytes, NULL, &err), err));
    CL_CHECK((backend_ctx->A_s_d_max = clCreateBuffer(context, 0, max_A_s_d_bytes, NULL, &err), err));
    CL_CHECK((backend_ctx->B_d_max   = clCreateBuffer(context, 0, max_B_d_bytes,   NULL, &err), err));
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    backend_ctx->disable_fusion = getenv("GGML_OPENCL_DISABLE_FUSION") != nullptr;

    dev_ctx->backend_ctx = backend_ctx.release();
    return dev_ctx->backend_ctx;
}

static void ggml_cl2_free(ggml_backend_t backend) {
    ggml_backend_opencl_context * ctx = (ggml_backend_opencl_context *) backend->context;
    ctx->free();

    // The CL context is shared by all backends, release it if all backends have been released
    bool should_release_opencl = true;
    for (auto device : g_ggml_backend_opencl_devices) {
        ggml_backend_opencl_device_context * ctx_dev = (ggml_backend_opencl_device_context *) device.context;
        if (ctx_dev->backend_ctx->ref_count > 0) {
            should_release_opencl = false;
        }
    }

    if (should_release_opencl) {
        CL_CHECK(clReleaseContext(ctx->context));
    }
}

//------------------------------------------------------------------------------
// Tensor extra management
//------------------------------------------------------------------------------
struct ggml_tensor_extra_cl {
    // The buffer object that holds the data.
    cl_mem data_device;
    // The offset into the buffer object. This is primarily for scratch buffer
    // and view operation.
    // NB: this offset no longer includes view offset (view_offs). Whenever this
    // offset is used, view_offs should be considered.
    cl_ulong offset;
    // The actual size of the cl_mem object. This is needed when returning the
    // block to the pool.
    size_t actual_size;
    // elastic baseline 用：被 WBM 管理的 block_idx；-1 = 不归 WBM 管。
    // 见 runtime/weight_buffer_manager.h 与 RUNTIME_PATCHES.md §3 H4。
    int wbm_idx;
    // elastic baseline 用：buffer_context::buffer 里对应 cl_mem 的 slot index；
    // -1 = 不在 elastic buffer 槽位中（如 view tensor / MONOLITHIC 路径）。
    // ensure_resident 重新分配 cl_mem 后用来把新 cl_mem 同步回 ctx->buffer。
    int ctx_slot;

    void reset() {
        data_device = nullptr;
        offset = 0;
        actual_size = 0;
        wbm_idx = -1;
        ctx_slot = -1;
    }
};

// Additional tensor extra structs for quantized tensors.
// These tensors are loaded from files and should not be allocated in scratch --
// they should always be allocated from the pool. Hence, they do not have an
// `offset`, which indicate their locations in the scratch buffer.
struct ggml_tensor_extra_cl_q4_0 {
    // Quantized values.
    cl_mem q = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem q_img = nullptr;
    // Elastic Cut fused GEMV creates an image view over q once per resident
    // lifetime.  It must be released with q on eviction so the image cannot
    // retain an otherwise evicted parent buffer.
    bool owns_q_img = false;
    // Scales.
    cl_mem d = nullptr;
    // Scales in image1d_buffer_t.
    cl_mem d_img = nullptr;
    // Size of quantized values.
    size_t size_q = 0;
    // Size of scales.
    size_t size_d = 0;
    // elastic baseline: 同 ggml_tensor_extra_cl::wbm_idx，对 SOA q4_0 tensor
    // 启用 evict/reload 时填。SOA 路径替换 tensor->extra 后，elastic 集成钩子
    // 通过这个 idx 反查 WBM block；-1 = 不归 WBM 管。
    int wbm_idx = -1;
    int ctx_slot = -1;
    // Parent buffer：SOA convert kernel 把 mmap 的原始字节 → {scales, quants}
    // 写入这个 cl_mem 的两段（extra->d、extra->q 是它的 sub-buffer）。evict 时
    // 释放它 + q/d sub-buffer；reload 时 alloc 新 parent 并重跑 convert。
    cl_mem parent_buffer = nullptr;

    ~ggml_tensor_extra_cl_q4_0() {
        reset();
    }

    bool reset_owned_images() {
        const bool released = owns_q_img && q_img != nullptr;
        if (released) {
            CL_CHECK(clReleaseMemObject(q_img));
        }
        q_img = nullptr;
        owns_q_img = false;
        return released;
    }

    void reset() {
        reset_owned_images();
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        // Non-owned image handles (for example SMALL_ALLOC aliases) remain
        // owned by the buffer context.
        d_img = nullptr;
        size_q = 0;
        size_d = 0;
    }
};

struct ggml_tensor_extra_cl_mxfp4 {
    // Quantized values.
    cl_mem q = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem q_img = nullptr;
    // Scales in E8M0.
    cl_mem e = nullptr;
    // Scales in image1d_buffer_t.
    cl_mem e_img = nullptr;
    // Size of quantized values.
    size_t size_q = 0;
    // Size of scales.
    size_t size_e = 0;
    // elastic baseline：与 ggml_tensor_extra_cl::wbm_idx 同义；类型不同 struct
    // layout 不同，graph_compute 读 wbm_idx 必须通过 ggml_opencl_get_wbm_idx() 帮手
    // 按 tensor->type 分派，不能直接 cast 通用 extra。
    int wbm_idx = -1;
    int ctx_slot = -1;
    cl_mem parent_buffer = nullptr;

    ~ggml_tensor_extra_cl_mxfp4() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (e != nullptr) {
            CL_CHECK(clReleaseMemObject(e));
            e = nullptr;
        }
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q_img));
            q = nullptr;
        }
        // Currently, q_img and d_img are not used. They can be image1d_buffer_t
        // that wraps around q and d to utilize image access path.
        q_img = nullptr;
        e_img = nullptr;
        size_q = 0;
        size_e = 0;
    }
};

struct ggml_tensor_extra_cl_q8_0 {
    cl_mem q = nullptr;
    cl_mem q_img = nullptr;

    cl_mem d = nullptr;
    cl_mem d_img = nullptr;

    size_t size_q = 0;
    size_t size_d = 0;
    int wbm_idx = -1;
    int ctx_slot = -1;
    cl_mem parent_buffer = nullptr;

    ~ggml_tensor_extra_cl_q8_0() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        // Currently, q_img and d_img are not used. They can be image1d_buffer_t
        // that wraps around q and d to utilize image access path.
        q_img = nullptr;
        d_img = nullptr;
        size_q = 0;
        size_d = 0;
    }
};

// Type-aware accessors：SOA 量化 tensor 用独立 extra struct，layout 与 generic
// 不同——直接 (ggml_tensor_extra_cl*) 强转读 wbm_idx / ctx_slot 会读错偏移。
// graph_compute / elastic 集成路径必须通过这两个 helper 拿值。
static inline int ggml_opencl_get_wbm_idx(const ggml_tensor *t) {
    if (!t || !t->extra) return -1;
    switch (t->type) {
        case GGML_TYPE_Q4_0:
            return ((const ggml_tensor_extra_cl_q4_0 *)t->extra)->wbm_idx;
        case GGML_TYPE_Q8_0:
            return ((const ggml_tensor_extra_cl_q8_0 *)t->extra)->wbm_idx;
        case GGML_TYPE_MXFP4:
            return ((const ggml_tensor_extra_cl_mxfp4 *)t->extra)->wbm_idx;
        default:
            return ((const ggml_tensor_extra_cl *)t->extra)->wbm_idx;
    }
}
static inline int ggml_opencl_get_ctx_slot(const ggml_tensor *t) {
    if (!t || !t->extra) return -1;
    switch (t->type) {
        case GGML_TYPE_Q4_0:
            return ((const ggml_tensor_extra_cl_q4_0 *)t->extra)->ctx_slot;
        case GGML_TYPE_Q8_0:
            return ((const ggml_tensor_extra_cl_q8_0 *)t->extra)->ctx_slot;
        case GGML_TYPE_MXFP4:
            return ((const ggml_tensor_extra_cl_mxfp4 *)t->extra)->ctx_slot;
        default:
            return ((const ggml_tensor_extra_cl *)t->extra)->ctx_slot;
    }
}

//------------------------------------------------------------------------------
// Backend API
//------------------------------------------------------------------------------

//
// backend
//
static const char * ggml_backend_opencl_name(ggml_backend_t backend) {
    return "OpenCL";

    UNUSED(backend);
}

static void ggml_backend_opencl_free(ggml_backend_t backend) {
    ggml_cl2_free(backend);
}

static void ggml_backend_opencl_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static void ggml_backend_opencl_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static bool ggml_backend_opencl_cpy_tensor_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static void ggml_backend_opencl_synchronize(ggml_backend_t backend) {
    auto * backend_ctx = static_cast<ggml_backend_opencl_context *>(backend->context);

    cl_event evt;
    CL_CHECK(clEnqueueBarrierWithWaitList(backend_ctx->queue, 0, nullptr, &evt));
    CL_CHECK(clWaitForEvents(1, &evt));
    CL_CHECK(clReleaseEvent(evt));
}

// Syncronizes the 'backend_ctx's device with others so that commands
// enqueued to it won't start until commands in the other devices have
// completed.
static void sync_with_other_backends(ggml_backend_opencl_context * backend_ctx) {
    if (g_ggml_backend_opencl_devices.size() < 2)
      return; // No other devices to synchronize with.

    std::vector<cl_event> events;
    events.reserve(g_ggml_backend_opencl_devices.size());

    for (ggml_backend_device & backend_dev : g_ggml_backend_opencl_devices) {
        auto * other_backend_ctx = ggml_cl2_init(&backend_dev);
        if (backend_ctx != other_backend_ctx) {
            cl_event ev;
            CL_CHECK(clEnqueueMarkerWithWaitList(other_backend_ctx->queue, 0, nullptr, &ev));
            CL_CHECK(clFlush(other_backend_ctx->queue));
            events.push_back(ev);
        }
    }

    CL_CHECK(clEnqueueBarrierWithWaitList(backend_ctx->queue, events.size(), events.data(), nullptr));
    for (auto ev : events) {
        CL_CHECK(clReleaseEvent(ev));
    }
}

static void sync_with_other_backends(ggml_backend_t backend) {
    auto * backend_ctx = static_cast<ggml_backend_opencl_context *>(backend->context);
    sync_with_other_backends(backend_ctx);
}

static bool ggml_opencl_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        // rms_norm only supports f32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        // if rms_norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        // rms_norm assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }
    } else if (ops.size() == 3 && ops.begin()[0] == GGML_OP_NORM && ops.begin()[1] == GGML_OP_MUL && ops.begin()[2] == GGML_OP_ADD) {
        const ggml_tensor *norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul  = cgraph->nodes[node_idx+1];
        const ggml_tensor *add  = cgraph->nodes[node_idx+2];
        const ggml_tensor *w    = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        const ggml_tensor *b    = add->src[0] == mul  ? add->src[1] : add->src[0];

        // norm fusion only supports F32
        if (norm->src[0]->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
            return false;
        }

        if (norm->src[0]->ne[0] % 4 != 0) {
            return false;
        }

        if (!ggml_is_contiguous(norm->src[0]) || !ggml_is_contiguous(w) || !ggml_is_contiguous(b)) {
            return false;
        }
    } else if (ops.size() == 3 && ops.begin()[0] == GGML_OP_GROUP_NORM && ops.begin()[1] == GGML_OP_MUL && ops.begin()[2] == GGML_OP_ADD) {
        const ggml_tensor *gn = cgraph->nodes[node_idx];
        const ggml_tensor *mul = cgraph->nodes[node_idx+1];
        const ggml_tensor *add = cgraph->nodes[node_idx+2];
        const ggml_tensor *w   = mul->src[0] == gn ? mul->src[1] : mul->src[0];
        const ggml_tensor *b   = add->src[0] == mul ? add->src[1] : add->src[0];

        if (gn->src[0]->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
            return false;
        }

        if (!ggml_is_contiguous(gn->src[0]) || !ggml_is_contiguous(w) || !ggml_is_contiguous(b)) {
            return false;
        }
    }

    return true;
}

static void ggml_opencl_op_rms_norm_fused(ggml_backend_t backend, ggml_tensor * rms_norm_tensor, ggml_tensor * mul_tensor);
static void ggml_opencl_op_norm_fused(ggml_backend_t backend, ggml_tensor * norm_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor);
static void ggml_opencl_op_group_norm_fused(ggml_backend_t backend, ggml_tensor * gn_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor);

// ============================================================
// Elastic baseline: 全局 WBM + BudgetWatcher + MetricsLogger 单例
// ============================================================
//
// 用 env 触发：
//   GGML_OPENCL_ELASTIC=1           启用 per-tensor cl_mem (见 F2)
//   GGML_ELASTIC_BUDGET_CSV=<path>  BudgetWatcher trace
//   GGML_ELASTIC_METRICS_JSONL=<path>  MetricsLogger 写盘
//   GGML_ELASTIC_KV_MB=<num>        KV cache 预算扣除（默认 128）
//   GGML_ELASTIC_MISC_MB=<num>      compute / 激活预算扣除（默认 256）
//   GGML_ELASTIC_EVICT_INTERVAL=<n> 每 n 个 op 检查一次预算（默认 8）
//
// 单例懒初：第一次有 weight tensor 走 set_tensor 时初始化 WBM；
// 第一次有 BudgetWatcher CSV 时拉起后台线程。

// F4-profile：按 tensor 名后缀 / op 类型聚合 reload IO 和 compute 时间。
// GGML_ELASTIC_PROFILE=1 触发。compute 时间会在每个 op 后 clFinish 拿真实
// GPU 时间，因此会显著拖慢，只在调研性能时使用。
struct elastic_profile_bucket {
    uint64_t n           = 0;
    double   total_ms    = 0.0;
    size_t   total_bytes = 0;
};

struct elastic_moe_q4_cache {
    cl_mem parent = nullptr;
    cl_mem route_slots = nullptr;
    cl_event route_upload_event = nullptr;
    int capacity = 0;
    int budget_capacity = 0;
    size_t raw_slice_bytes = 0;
    size_t q_slice_bytes = 0;
    size_t d_slice_bytes = 0;
    std::vector<int> expert_for_slot;
    std::vector<uint64_t> last_used;
    std::vector<uint32_t> frequency;
    std::vector<int32_t> route_slots_host;
    uint64_t accesses = 0;
};

struct elastic_moe_prefetch_task {
    int wbm_idx = -1;
    int expert_id = -1;
    std::string filename;
    size_t file_offset = 0;
    size_t data_size = 0;
    std::unique_ptr<unsigned char[]> storage;
    unsigned char * data = nullptr;
    bool done = false;
    bool ok = false;
    uint64_t io_us = 0;
};

struct elastic_q4_cut_part {
    int64_t row_start = 0;
    int64_t row_count = 0;
    size_t byte_offset = 0;
    size_t byte_size = 0;
    ggml_tensor_extra_cl_q4_0 * extra = nullptr;
    int wbm_idx = -1;
};

struct opencl_pipeline_successor {
    uint64_t signature = 0;
    std::vector<std::vector<int>> units;
};

struct opencl_pipeline_pin_state {
    size_t refs = 0;
    bool original = false;
};

static constexpr size_t ELASTIC_MOE_DIRECT_ALIGNMENT = 4096;
static constexpr size_t ELASTIC_MOE_DIRECT_PADDING = 2 * ELASTIC_MOE_DIRECT_ALIGNMENT;

static unsigned char * ggml_opencl_moe_direct_ptr(unsigned char * storage, size_t file_offset) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(storage);
    const uintptr_t aligned = (base + ELASTIC_MOE_DIRECT_ALIGNMENT - 1) &
                              ~(uintptr_t) (ELASTIC_MOE_DIRECT_ALIGNMENT - 1);
    return reinterpret_cast<unsigned char *>(aligned + file_offset % ELASTIC_MOE_DIRECT_ALIGNMENT);
}

struct ggml_opencl_elastic_state {
    elastic::weight_buffer_manager wbm;
    elastic::wbm_opencl_ctx        octx;
    elastic::budget_watcher        bw;
    elastic::metrics_logger        metrics;

    bool   wbm_inited     = false;
    bool   bw_inited      = false;
    bool   metrics_inited = false;

    elastic::granularity_config granularity;
    uint64_t eviction_group_generation = UINT64_MAX;
    std::unordered_map<const ggml_tensor *, std::vector<elastic_q4_cut_part>> q4_cut_parts;
    cl_mem cut_output_scratch = nullptr;
    size_t cut_output_scratch_bytes = 0;
    cl_command_queue cut_compute_queue = nullptr;
    uint64_t cut_dual_candidates = 0;
    uint64_t cut_dual_calls = 0;
    uint64_t cut_dual_budget_fallbacks = 0;
    uint64_t cut_dual_shape_fallbacks = 0;
    uint64_t cut_dual_queue_errors = 0;
    uint64_t cut_dual_image_creates = 0;
    uint64_t cut_dual_image_cache_hits = 0;
    uint64_t cut_dual_image_releases = 0;
    uint64_t cut_dual_image_errors = 0;
    uint64_t granularity_units = 0;
    uint64_t granularity_cut_ops = 0;
    uint64_t granularity_fallback_ops = 0;
    size_t granularity_peak_unit_bytes = 0;
    size_t granularity_cut_tensors = 0;
    size_t granularity_cut_parts = 0;
    // A logical working-unit boundary must not imply a host-blocking drain of
    // the entire OpenCL queue.  FLUSH submits queued work without waiting;
    // FINISH is retained only as an explicit diagnostic baseline.
    enum class unit_sync_mode {
        NONE,
        FLUSH,
        FINISH,
    };
    unit_sync_mode granularity_unit_sync = unit_sync_mode::FLUSH;
    uint64_t granularity_unit_boundaries = 0;
    uint64_t granularity_unit_flushes = 0;
    uint64_t granularity_unit_finishes = 0;
    uint64_t granularity_unit_sync_errors = 0;
    uint64_t unit_pipeline_current_missing_units = 0;
    uint64_t unit_pipeline_issued = 0;
    uint64_t unit_pipeline_ready = 0;
    uint64_t unit_pipeline_waits = 0;
    uint64_t unit_pipeline_wait_us = 0;
    uint64_t unit_pipeline_window_samples = 0;
    uint64_t unit_pipeline_window_units_total = 0;
    size_t unit_pipeline_window_units_max = 0;
    size_t unit_pipeline_window_bytes_total = 0;
    size_t unit_pipeline_window_bytes_max = 0;
    uint64_t unit_pipeline_oversize_windows = 0;
    // Keep the byte-bounded unit pipeline alive across backend-graph calls.
    std::unordered_map<uint64_t, opencl_pipeline_successor> unit_pipeline_successors;
    uint64_t unit_pipeline_previous_graph_signature = 0;
    bool unit_pipeline_previous_graph_valid = false;
    std::deque<opencl_pipeline_successor> unit_pipeline_cross_graphs;
    std::unordered_map<int, opencl_pipeline_pin_state> unit_pipeline_cross_pins;
    uint64_t unit_pipeline_cross_issued = 0;
    uint64_t unit_pipeline_cross_ready = 0;
    uint64_t unit_pipeline_cross_waits = 0;
    size_t unit_pipeline_cross_bytes = 0;
    uint64_t unit_pipeline_budget_samples = 0;
    uint64_t unit_pipeline_budget_violations = 0;
    uint64_t unit_pipeline_plan_protection_relaxations = 0;
    size_t unit_pipeline_plan_protection_relaxed_bytes = 0;
    size_t unit_pipeline_resident_bytes_peak = 0;
    size_t unit_pipeline_pinned_bytes_peak = 0;
    size_t unit_pipeline_over_budget_bytes_peak = 0;

    size_t kv_bytes       = static_cast<size_t>(128) * 1024 * 1024;
    size_t misc_overhead  = static_cast<size_t>(256) * 1024 * 1024;
    int    evict_check_interval = 8;

    // Baseline (spec §5)：静态 target = M_floor - kv - misc - safety，整个 run 不变。
    // bw_inited 后在 lazy_init 里算一次。bw 没启用时为 0，等价于关闭 evict。
    size_t static_target_bytes = 0;

    // Dynamic mode (GGML_ELASTIC_DYNAMIC=1): graph_compute 时实时算
    // target = B(t)*MB - kv - misc - safety + extra_target_bytes (跟 budget trace 走)。
    // extra_target_bytes 单独累加 EMBED_OUTSIDE_BUDGET 等 extras, 跟 static 路径平行
    // (static 路径把 extras 加进 static_target_bytes; dynamic 路径用 extra_target_bytes)。
    bool   dynamic_target       = false;
    size_t extra_target_bytes   = 0;

    // 流水线 prefetch：GGML_ELASTIC_PREFETCH=N 提前对后 N 个 node 的 src 发
    // async write。N=0 关闭。开启时 xfer_queue 自动建。
    int      prefetch_lookahead   = 0;
    uint64_t n_prefetch_issued    = 0;
    uint64_t n_prefetch_skipped   = 0;
    int      gpu_prefetch_lookahead = 0;
    uint64_t n_gpu_prefetch_issued  = 0;
    uint64_t n_gpu_prefetch_skipped = 0;
    bool     retain_cap_auto         = false;  // GGML_ELASTIC_CL_RETAIN_MB=auto

    uint64_t current_token       = 0;
    uint64_t n_op_dispatched     = 0;
    uint64_t n_evicts_total      = 0;
    uint64_t n_reloads_total     = 0;
    size_t   bytes_reloaded_total = 0;

    bool profile = false;
    bool profile_csv = false;
    std::map<std::string, elastic_profile_bucket> reload_buckets;   // tensor name suffix → 累计
    std::map<int,         elastic_profile_bucket> compute_buckets;  // ggml_op → 累计

    std::chrono::steady_clock::time_point t0;  // wbm 初始化时刻；用于 metrics.t_sec

    // GGML_ELASTIC_TIMING=1：不带 profile clFinish 干扰的真实时间统计。atexit 输出。
    // reload_host_us = reload_fn 调用的纯 CPU 侧 enqueue 时间（不等 GPU 完成）
    // reload_gpu_us  = GPU side convert/transpose kernel 真实运行时间（从 cl_event
    //                  CL_PROFILING_COMMAND_{START,END} 提取，绕过 host clFinish）
    bool       timing       = false;
    uint64_t   t_reload_host_us  = 0;   // sum of host-issue time for all reload_fn calls
    uint64_t   t_reload_gpu_us   = 0;   // sum of GPU profiling time on convert kernel
    uint64_t   t_compute_gpu_us  = 0;   // sum of GPU profiling time on matmul kernels
    uint64_t   n_reload_gpu_ev   = 0;   // count of GPU events sampled
    uint64_t   n_compute_gpu_ev  = 0;

    // === Runtime scheduler 集成: tensor name → wbm_idx ===
    std::mutex                              sched_mtx;
    std::unordered_map<std::string, int>    name_to_wbm;
    std::unordered_map<std::string, std::vector<int>> name_to_wbms;
    std::unordered_map<int, std::string>    wbm_to_name;
    bool                                    sched_registered = false;

    // Optional Q4_0 MoE expert-slice cache. Packed expert tensors remain the
    // model-file representation, while decode residency is managed per expert.
    // This is deliberately keyed by WBM tensor id so gate/up/down tensors keep
    // independent caches and lifetimes.
    std::mutex                                      moe_cache_mtx;
    std::unordered_map<int, elastic_moe_q4_cache>   moe_q4_cache;
    size_t                                           moe_cache_bytes = 0;
    uint64_t                                         moe_cache_clock = 0;
    uint64_t                                         moe_cache_hits = 0;
    uint64_t                                         moe_cache_misses = 0;
    size_t                                           moe_cache_disk_bytes = 0;
    uint64_t                                         moe_cache_disk_us = 0;
    uint64_t                                         moe_cache_router_read_us = 0;
    uint64_t                                         moe_cache_router_read_calls = 0;
    uint64_t                                         moe_cache_cpu_xform_us = 0;
    uint64_t                                         moe_cache_upload_us = 0;
    uint64_t                                         moe_cache_route_upload_us = 0;
    uint64_t                                         moe_cache_prepare_us = 0;
    uint64_t                                         moe_cache_prepare_calls = 0;
    uint64_t                                         moe_cache_grow_resizes = 0;
    size_t                                           moe_cache_grow_copy_bytes = 0;
    int                                              moe_cache_active_capacity = -1;
    int                                              moe_cache_plan_capacity = -1;
    int                                              moe_cache_required_high_water = 0;
    int                                              moe_cache_pending_capacity = -1;
    int                                              moe_cache_pending_hits = 0;
    uint64_t                                         moe_cache_capacity_decision_token = 0;
    uint64_t                                         moe_cache_last_capacity_change_token = 0;
    int                                              moe_cache_resize_target = -1;
    uint64_t                                         moe_cache_resize_syncs = 0;
    std::unordered_set<int>                          moe_q4_packed_indices;
    uint64_t                                         moe_packed_stage_skips = 0;
    uint64_t                                         moe_packed_parent_evictions = 0;
    size_t                                           moe_packed_parent_evicted_bytes = 0;
    FILE *                                           moe_route_trace_file = nullptr;
    uint64_t                                         moe_route_trace_seq = 0;
    std::mutex                                       moe_prefetch_mtx;
    std::condition_variable                          moe_prefetch_cv;
    std::deque<std::shared_ptr<elastic_moe_prefetch_task>> moe_prefetch_queue;
    std::unordered_map<uint64_t, std::shared_ptr<elastic_moe_prefetch_task>> moe_prefetch_tasks;
    std::vector<std::pair<size_t, std::unique_ptr<unsigned char[]>>> moe_prefetch_free_buffers;
    bool                                             moe_prefetch_workers_started = false;
    uint64_t                                         moe_prefetch_issued = 0;
    uint64_t                                         moe_prefetch_completed = 0;
    uint64_t                                         moe_prefetch_used = 0;
    uint64_t                                         moe_prefetch_skipped_hit = 0;
    uint64_t                                         moe_prefetch_skipped_cap = 0;
    uint64_t                                         moe_prefetch_wait_us = 0;
    uint64_t                                         moe_prefetch_buffer_allocs = 0;
    uint64_t                                         moe_prefetch_buffer_reuses = 0;
    std::unordered_map<const ggml_tensor *, std::pair<uint64_t, std::vector<int32_t>>> moe_router_ids;
    std::unordered_map<int, std::vector<int32_t>> moe_router_ids_by_layer;
    std::unordered_map<int, int>                  moe_router_reuse_left_by_layer;

    std::mutex                              async_xform_mtx;
    std::condition_variable                 async_xform_cv;
    std::unordered_set<int>                 async_xform_inflight;
    std::deque<int>                         async_xform_queue;
    bool                                    async_xform_worker_started = false;
    size_t                                  async_xform_max_inflight = 0;
    uint64_t                                async_xform_enqueued = 0;
    uint64_t                                async_xform_completed = 0;
    uint64_t                                async_xform_waits = 0;
    uint64_t                                async_xform_wait_us = 0;
    size_t                                  async_xform_max_pending_seen = 0;
};

static ggml_opencl_elastic_state * ggml_opencl_elastic() {
    static ggml_opencl_elastic_state s;
    return &s;
}

static int ggml_opencl_moe_prefetch_workers() {
    static const int workers = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_PREFETCH_WORKERS");
        return std::max(0, e && *e ? std::atoi(e) : 0);
    }();
    return workers;
}

static size_t ggml_opencl_moe_prefetch_max_pending() {
    static const size_t max_pending = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_PREFETCH_MAX_PENDING");
        const long long v = e && *e ? std::atoll(e) : 16;
        return (size_t) std::max<long long>(1, v);
    }();
    return max_pending;
}

static uint64_t ggml_opencl_moe_prefetch_key(int wbm_idx, int expert_id) {
    return (uint64_t) (uint32_t) wbm_idx << 32 | (uint32_t) expert_id;
}

static void ggml_opencl_start_moe_prefetch_workers(ggml_opencl_elastic_state * state) {
    const int worker_count = ggml_opencl_moe_prefetch_workers();
    if (!state || worker_count <= 0) return;
    {
        std::lock_guard<std::mutex> lock(state->moe_prefetch_mtx);
        if (state->moe_prefetch_workers_started) return;
        state->moe_prefetch_workers_started = true;
    }
    for (int i = 0; i < worker_count; ++i) {
        std::thread([state]() {
            for (;;) {
                std::shared_ptr<elastic_moe_prefetch_task> task;
                {
                    std::unique_lock<std::mutex> lock(state->moe_prefetch_mtx);
                    state->moe_prefetch_cv.wait(lock, [state]() { return !state->moe_prefetch_queue.empty(); });
                    task = state->moe_prefetch_queue.front();
                    state->moe_prefetch_queue.pop_front();
                    for (auto it = state->moe_prefetch_free_buffers.begin();
                        it != state->moe_prefetch_free_buffers.end(); ++it) {
                        if (it->first == task->data_size) {
                            task->storage = std::move(it->second);
                            state->moe_prefetch_free_buffers.erase(it);
                            state->moe_prefetch_buffer_reuses++;
                            break;
                        }
                    }
                }
                const auto t0 = std::chrono::steady_clock::now();
                if (!task->storage) {
                    task->storage.reset(new (std::nothrow) unsigned char[
                        task->data_size + ELASTIC_MOE_DIRECT_PADDING]);
                    std::lock_guard<std::mutex> lock(state->moe_prefetch_mtx);
                    state->moe_prefetch_buffer_allocs++;
                }
                task->data = task->storage ?
                    ggml_opencl_moe_direct_ptr(task->storage.get(), task->file_offset) : nullptr;
                const bool ok = task->data &&
                    llama_pread_direct(task->filename.c_str(), task->data,
                                       task->file_offset, task->data_size) == 0;
                const uint64_t io_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count();
                {
                    std::lock_guard<std::mutex> lock(state->moe_prefetch_mtx);
                    task->ok = ok;
                    task->io_us = io_us;
                    task->done = true;
                    state->moe_prefetch_completed++;
                }
                state->moe_prefetch_cv.notify_all();
            }
        }).detach();
    }
}

static void ggml_opencl_enqueue_moe_companion_prefetch(
        ggml_opencl_elastic_state * state,
        const ggml_tensor * gate,
        const std::vector<int32_t> & expert_ids,
        size_t raw_slice_bytes,
        int n_experts) {
    if (!state || !gate || ggml_opencl_moe_prefetch_workers() <= 0 ||
        std::strstr(gate->name, "ffn_gate_exps.weight") == nullptr) {
        return;
    }
    ggml_opencl_start_moe_prefetch_workers(state);
    const std::string gate_name = gate->name;
    const size_t marker = gate_name.find("ffn_gate_exps.weight");
    if (marker == std::string::npos) return;

    for (const char * replacement : {"ffn_up_exps.weight", "ffn_down_exps.weight"}) {
        std::string companion = gate_name;
        companion.replace(marker, std::strlen("ffn_gate_exps.weight"), replacement);
        int target_idx = -1;
        {
            std::lock_guard<std::mutex> lock(state->sched_mtx);
            auto it = state->name_to_wbm.find(companion);
            if (it != state->name_to_wbm.end()) target_idx = it->second;
        }
        const elastic::block_meta * target = elastic::wbm_get(&state->wbm, target_idx);
        if (target_idx < 0 || !target || !target->host_ptr ||
            target->byte_size < raw_slice_bytes * (size_t) n_experts) {
            continue;
        }
        for (int expert_id : expert_ids) {
            bool resident = false;
            {
                std::lock_guard<std::mutex> lock(state->moe_cache_mtx);
                auto cache_it = state->moe_q4_cache.find(target_idx);
                if (cache_it != state->moe_q4_cache.end()) {
                    const auto & slots = cache_it->second.expert_for_slot;
                    resident = std::find(slots.begin(), slots.end(), expert_id) != slots.end();
                }
            }
            if (resident) {
                std::lock_guard<std::mutex> lock(state->moe_prefetch_mtx);
                state->moe_prefetch_skipped_hit++;
                continue;
            }

            const unsigned char * mmap_src = (const unsigned char *) target->host_ptr +
                                             (size_t) expert_id * raw_slice_bytes;
            auto reg = llama_mmap_registry_find(mmap_src);
            if (reg.filename.empty()) continue;
            auto task = std::make_shared<elastic_moe_prefetch_task>();
            task->wbm_idx = target_idx;
            task->expert_id = expert_id;
            task->filename = reg.filename;
            task->file_offset = (const char *) mmap_src - (const char *) reg.base;
            task->data_size = raw_slice_bytes;
            const uint64_t key = ggml_opencl_moe_prefetch_key(target_idx, expert_id);
            {
                std::lock_guard<std::mutex> lock(state->moe_prefetch_mtx);
                if (state->moe_prefetch_tasks.find(key) != state->moe_prefetch_tasks.end()) continue;
                if (state->moe_prefetch_tasks.size() >= ggml_opencl_moe_prefetch_max_pending()) {
                    state->moe_prefetch_skipped_cap++;
                    continue;
                }
                state->moe_prefetch_tasks.emplace(key, task);
                state->moe_prefetch_queue.push_back(task);
                state->moe_prefetch_issued++;
            }
            state->moe_prefetch_cv.notify_one();
        }
    }
}

static std::shared_ptr<elastic_moe_prefetch_task> ggml_opencl_consume_moe_prefetch(
        ggml_opencl_elastic_state * state,
        int wbm_idx,
        int expert_id) {
    if (!state || ggml_opencl_moe_prefetch_workers() <= 0) return {};
    const uint64_t key = ggml_opencl_moe_prefetch_key(wbm_idx, expert_id);
    std::shared_ptr<elastic_moe_prefetch_task> task;
    const auto wait_t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lock(state->moe_prefetch_mtx);
        auto it = state->moe_prefetch_tasks.find(key);
        if (it == state->moe_prefetch_tasks.end()) return {};
        task = it->second;
        state->moe_prefetch_cv.wait(lock, [&task]() { return task->done; });
        state->moe_prefetch_tasks.erase(key);
        state->moe_prefetch_used++;
    }
    state->moe_prefetch_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - wait_t0).count();
    if (!task->ok || !task->storage || !task->data || task->data_size == 0) return {};
    state->moe_cache_disk_bytes += task->data_size;
    state->moe_cache_disk_us += task->io_us;
    return task;
}

static bool ggml_opencl_moe_expert_cache_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_EXPERT_CACHE");
        return e && *e && *e != '0';
    }();
    return enabled;
}

static bool ggml_opencl_is_moe_expert_weight(const ggml_tensor * tensor) {
    return tensor && tensor->type == GGML_TYPE_Q4_0 && tensor->ne[2] > 1 &&
           std::strstr(tensor->name, "_exps.weight") != nullptr;
}

struct elastic_moe_q4_cache_view {
    cl_mem parent = nullptr;
    cl_mem route_slots = nullptr;
    size_t slot_stride = 0;
    size_t q_offset = 0;
};

static bool ggml_opencl_prepare_moe_q4_cache(
        ggml_backend_opencl_context * backend_ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src2,
        cl_ulong offset2,
        const ggml_tensor * id_mask,
        cl_mem id_mask_buffer,
        cl_ulong id_mask_offset,
        cl_ulong mask_nb1,
        elastic_moe_q4_cache_view * out) {
    const auto prepare_t0 = std::chrono::steady_clock::now();
    if (!backend_ctx || !src0 || !src2 || !out || !ggml_opencl_moe_expert_cache_enabled()) {
        return false;
    }
    if (!ggml_opencl_is_moe_expert_weight(src0) || src2->ne[1] != 1 || src2->type != GGML_TYPE_I32) {
        return false;
    }

    const int wbm_idx = ggml_opencl_get_wbm_idx(src0);
    auto * state = ggml_opencl_elastic();
    const elastic::block_meta * bm = elastic::wbm_get(&state->wbm, wbm_idx);
    if (wbm_idx < 0 || !bm || !bm->host_ptr || bm->byte_size == 0) {
        return false;
    }

    const int n_selected = (int) src2->ne[0];
    const int n_experts = (int) src0->ne[2];
    if (n_selected <= 0 || n_selected > n_experts) {
        return false;
    }

    // LLAMA_MOE_DYNAMIC_ALL_EXPERTS keeps all expert IDs in src2 and masks
    // inactive routes through src3.  Materializing all IDs here defeats the
    // dynamic routing memory saving and can allocate several GiB unnecessarily.
    std::vector<uint8_t> active_routes((size_t) n_selected, 1);
    int n_active = n_selected;
    if (id_mask && id_mask_buffer && id_mask->type == GGML_TYPE_F32) {
        std::vector<float> mask_values((size_t) n_selected, 0.0f);
        cl_int mask_err = CL_SUCCESS;
        if (mask_nb1 == sizeof(float)) {
            mask_err = clEnqueueReadBuffer(backend_ctx->queue, id_mask_buffer, CL_TRUE,
                                           id_mask_offset, sizeof(float) * (size_t) n_selected,
                                           mask_values.data(), 0, nullptr, nullptr);
        } else {
            for (int i = 0; i < n_selected && mask_err == CL_SUCCESS; ++i) {
                mask_err = clEnqueueReadBuffer(backend_ctx->queue, id_mask_buffer, CL_TRUE,
                                               id_mask_offset + (cl_ulong) i * mask_nb1,
                                               sizeof(float), &mask_values[(size_t) i],
                                               0, nullptr, nullptr);
            }
        }
        if (mask_err != CL_SUCCESS) {
            GGML_LOG_ERROR("ggml_opencl moe cache: read active mask failed tensor=%s err=%d\n",
                           src0->name, mask_err);
            return false;
        }
        n_active = 0;
        for (int i = 0; i < n_selected; ++i) {
            active_routes[(size_t) i] = mask_values[(size_t) i] != 0.0f;
            n_active += active_routes[(size_t) i] ? 1 : 0;
        }
    }
    const int n_required = std::max(1, n_active);

    static const int requested_capacity = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_EXPERT_CACHE_SLOTS");
        const int v = e && *e ? std::atoi(e) : 8;
        return std::max(1, v);
    }();
    const size_t raw_slice_bytes = src0->nb[2];
    const size_t block_bytes = ggml_type_size(GGML_TYPE_Q4_0);
    const size_t qk = ggml_blck_size(GGML_TYPE_Q4_0);
    if (raw_slice_bytes == 0 || block_bytes < 2 || qk == 0 || raw_slice_bytes % block_bytes != 0) {
        return false;
    }
    const size_t n_blocks = raw_slice_bytes / block_bytes;
    const size_t q_slice_bytes = n_blocks * (qk / 2);
    const size_t d_slice_bytes = n_blocks * sizeof(ggml_fp16_t);
    int budget_capacity = std::min(n_experts, requested_capacity);
    int capacity = std::min(n_experts, std::max(n_required, budget_capacity));
    static const bool adaptive_capacity = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_EXPERT_CACHE_ADAPTIVE");
        return e && *e && *e != '0';
    }();
    if (adaptive_capacity && state->bw_inited) {
        static const int max_slots = []() {
            const char * e = std::getenv("GGML_ELASTIC_MOE_EXPERT_CACHE_MAX_SLOTS");
            return std::max(1, e && *e ? std::atoi(e) : 60);
        }();
        static const int tensor_count = []() {
            const char * e = std::getenv("GGML_ELASTIC_MOE_EXPERT_TENSOR_COUNT");
            return std::max(1, e && *e ? std::atoi(e) : 72);
        }();
        static const int slot_step = []() {
            const char * e = std::getenv("GGML_ELASTIC_MOE_EXPERT_CACHE_SLOT_STEP");
            return std::max(1, e && *e ? std::atoi(e) : 4);
        }();
        static const size_t configured_base_budget_mib = []() {
            const char * e = std::getenv("GGML_ELASTIC_MOE_EXPERT_CACHE_BASE_BUDGET_MIB");
            return e && *e ? (size_t) std::max<long long>(0, std::atoll(e)) : 0;
        }();
        size_t budget_mib = elastic::budget_watcher_get(&state->bw);
        static const size_t budget_bucket_mib = []() {
            const char * e = std::getenv("GGML_ELASTIC_BUDGET_BUCKET_MB");
            return (size_t) std::max<long long>(1, e && *e ? std::atoll(e) : 1);
        }();
        // The planner changes residency only at bucket boundaries. Using raw
        // trace samples here made tiny fluctuations repeatedly resize every
        // expert tensor even while the active plan stayed unchanged.
        budget_mib = (budget_mib / budget_bucket_mib) * budget_bucket_mib;
        const size_t floor_mib = configured_base_budget_mib > 0 ?
            configured_base_budget_mib : state->bw.m_floor_mb;
        const size_t extra_bytes = budget_mib > floor_mib ?
            (budget_mib - floor_mib) * size_t(1024 * 1024) : 0;
        const size_t global_slot_bytes = (q_slice_bytes + d_slice_bytes) * (size_t) tensor_count;
        int extra_slots = global_slot_bytes > 0 ? (int) (extra_bytes / global_slot_bytes) : 0;
        extra_slots = (extra_slots / slot_step) * slot_step;
        // The budget controls optional retained experts; active experts are a
        // correctness requirement and may temporarily raise a tensor above the
        // retained-cache tier. Keeping these quantities separate also avoids
        // sizing every layer from the first layer observed for this token.
        budget_capacity = std::min({n_experts, max_slots, requested_capacity + extra_slots});
        capacity = std::max(n_required, budget_capacity);
    }

    if (adaptive_capacity) {
        static const int grow_stable_graphs = []() {
            const char * e = std::getenv("GGML_ELASTIC_MOE_EXPERT_CACHE_GROW_STABLE_GRAPHS");
            return std::max(1, e && *e ? std::atoi(e) : 8);
        }();
        static const uint64_t resize_min_gap_graphs = []() {
            const char * e = std::getenv("GGML_ELASTIC_MOE_EXPERT_CACHE_RESIZE_MIN_GAP_GRAPHS");
            return (uint64_t) std::max(0, e && *e ? std::atoi(e) : 16);
        }();
        int desired_capacity = budget_capacity;
        std::lock_guard<std::mutex> capacity_lock(state->moe_cache_mtx);
        state->moe_cache_required_high_water = std::max(state->moe_cache_required_high_water, n_required);
        const bool plan_controlled = state->moe_cache_plan_capacity >= 0;
        if (state->moe_cache_plan_capacity >= 0) {
            desired_capacity = std::min(n_experts, state->moe_cache_plan_capacity);
        }
        if (state->moe_cache_active_capacity < 0) {
            state->moe_cache_active_capacity = desired_capacity;
            state->moe_cache_capacity_decision_token = state->current_token;
            state->moe_cache_last_capacity_change_token = state->current_token;
        } else if (state->moe_cache_capacity_decision_token != state->current_token) {
            const int old_capacity = state->moe_cache_active_capacity;
            const char * reason = nullptr;
            if (desired_capacity < old_capacity) {
                // Budget compliance takes precedence over cache hit rate.
                state->moe_cache_active_capacity = desired_capacity;
                state->moe_cache_pending_capacity = -1;
                state->moe_cache_pending_hits = 0;
                reason = "budget_shrink";
            } else if (desired_capacity > old_capacity) {
                if (plan_controlled) {
                    state->moe_cache_active_capacity = desired_capacity;
                    state->moe_cache_pending_capacity = -1;
                    state->moe_cache_pending_hits = 0;
                    reason = "plan_grow";
                } else if (state->moe_cache_pending_capacity == desired_capacity) {
                    state->moe_cache_pending_hits++;
                } else {
                    state->moe_cache_pending_capacity = desired_capacity;
                    state->moe_cache_pending_hits = 1;
                }
                const bool gap_ready = state->current_token >= state->moe_cache_last_capacity_change_token +
                                       resize_min_gap_graphs;
                if (state->moe_cache_pending_hits >= grow_stable_graphs && gap_ready) {
                    state->moe_cache_active_capacity = desired_capacity;
                    state->moe_cache_pending_capacity = -1;
                    state->moe_cache_pending_hits = 0;
                    reason = "stable_grow";
                }
            } else {
                state->moe_cache_pending_capacity = -1;
                state->moe_cache_pending_hits = 0;
            }
            if (reason) {
                state->moe_cache_last_capacity_change_token = state->current_token;
                GGML_LOG_INFO("ggml_opencl moe cache: capacity tier %d->%d budget=%zuMiB reason=%s graph=%llu\n",
                              old_capacity, state->moe_cache_active_capacity,
                              state->bw_inited ? elastic::budget_watcher_get(&state->bw) : 0,
                              reason, (unsigned long long) state->current_token);
            }
            state->moe_cache_capacity_decision_token = state->current_token;
        }
        capacity = std::max(n_required, state->moe_cache_active_capacity);
    }
    // Gate misses are on the decode critical path, while gate routing gives
    // up/down loads a short overlap window. Shift slots toward gate without
    // increasing the aggregate gate+up+down cache footprint.
    static const int gate_slot_shift = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_GATE_SLOT_SHIFT");
        return std::max(0, e && *e ? std::atoi(e) : 0);
    }();
    if (gate_slot_shift > 0) {
        const int actual_shift = std::min(gate_slot_shift, n_experts - capacity);
        if (std::strstr(src0->name, "ffn_gate_exps.weight") != nullptr) {
            capacity += actual_shift;
        } else if (std::strstr(src0->name, "ffn_up_exps.weight") != nullptr ||
                   std::strstr(src0->name, "ffn_down_exps.weight") != nullptr) {
            capacity = std::max(n_required, capacity - (actual_shift + 1) / 2);
        }
    }

    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *) src2->extra;
    if (!extra2 || !extra2->data_device) {
        return false;
    }
    std::vector<int32_t> expert_ids;
    static const bool reuse_layer_route = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_ROUTE_REUSE");
        return e && *e && *e != '0';
    }();
    int moe_layer = -1;
    if (reuse_layer_route) {
        if (std::sscanf(src0->name, "blk.%d.", &moe_layer) != 1) {
            moe_layer = -1;
        }
    }
    {
        std::lock_guard<std::mutex> lock(state->moe_cache_mtx);
        if (moe_layer >= 0) {
            auto ids_it = state->moe_router_ids_by_layer.find(moe_layer);
            auto left_it = state->moe_router_reuse_left_by_layer.find(moe_layer);
            if (ids_it != state->moe_router_ids_by_layer.end() &&
                left_it != state->moe_router_reuse_left_by_layer.end() &&
                left_it->second > 0 && ids_it->second.size() == (size_t) n_selected) {
                expert_ids = ids_it->second;
                left_it->second--;
            }
        } else {
            auto it = state->moe_router_ids.find(src2);
            if (it != state->moe_router_ids.end() && it->second.first == state->current_token) {
                expert_ids = it->second.second;
            }
        }
    }
    cl_int err = CL_SUCCESS;
    if (expert_ids.size() != (size_t) n_selected) {
        expert_ids.resize((size_t) n_selected);
        const auto router_t0 = std::chrono::steady_clock::now();
        err = clEnqueueReadBuffer(backend_ctx->queue, extra2->data_device, CL_TRUE,
                                  offset2, expert_ids.size() * sizeof(int32_t),
                                  expert_ids.data(), 0, nullptr, nullptr);
        state->moe_cache_router_read_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - router_t0).count();
        state->moe_cache_router_read_calls++;
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("ggml_opencl moe cache: read router ids failed tensor=%s err=%d\n", src0->name, err);
            return false;
        }
        std::lock_guard<std::mutex> lock(state->moe_cache_mtx);
        if (moe_layer >= 0) {
            // A decode layer consumes the same selected-expert tensor for its
            // up, gate, and down MUL_MAT_ID operations. These may be split
            // across separate backend graphs, so current_token is not a valid
            // lifetime key. The first operation reads the route and the next
            // two consume it; the following operation refreshes the entry.
            state->moe_router_ids_by_layer[moe_layer] = expert_ids;
            state->moe_router_reuse_left_by_layer[moe_layer] = 2;
        } else {
            state->moe_router_ids[src2] = {state->current_token, expert_ids};
        }
    }

    static const char * route_trace_path = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_ROUTE_TRACE_CSV");
        return e && *e ? e : nullptr;
    }();
    if (route_trace_path && std::strstr(src0->name, "ffn_gate_exps.weight") != nullptr) {
        std::lock_guard<std::mutex> lock(state->moe_cache_mtx);
        if (!state->moe_route_trace_file) {
            state->moe_route_trace_file = std::fopen(route_trace_path, "w");
            if (state->moe_route_trace_file) {
                std::fprintf(state->moe_route_trace_file,
                             "seq,graph_id,weight,budget_mib,capacity,expert_ids\n");
            } else {
                GGML_LOG_WARN("ggml_opencl moe cache: cannot open route trace %s\n", route_trace_path);
            }
        }
        if (state->moe_route_trace_file) {
            std::fprintf(state->moe_route_trace_file, "%llu,%llu,%s,%zu,%d,",
                         (unsigned long long) state->moe_route_trace_seq++,
                         (unsigned long long) state->current_token, src0->name,
                         state->bw_inited ? elastic::budget_watcher_get(&state->bw) : 0, capacity);
            for (size_t i = 0; i < expert_ids.size(); ++i) {
                std::fprintf(state->moe_route_trace_file, "%s%d", i ? ";" : "", expert_ids[i]);
            }
            std::fputc('\n', state->moe_route_trace_file);
            if ((state->moe_route_trace_seq % 64) == 0) {
                std::fflush(state->moe_route_trace_file);
            }
        }
    }
    std::vector<int32_t> active_expert_ids;
    active_expert_ids.reserve((size_t) n_active);
    for (int i = 0; i < n_selected; ++i) {
        if (active_routes[(size_t) i]) {
            active_expert_ids.push_back(expert_ids[(size_t) i]);
        }
    }
    ggml_opencl_enqueue_moe_companion_prefetch(state, src0, active_expert_ids, raw_slice_bytes, n_experts);

    std::lock_guard<std::mutex> lock(state->moe_cache_mtx);
    elastic_moe_q4_cache & cache = state->moe_q4_cache[wbm_idx];
    if (!adaptive_capacity && cache.parent) {
        // Active expert counts vary by layer and token. Keep the high-water
        // capacity instead of reallocating the cache on every downward change.
        capacity = std::max(capacity, cache.capacity);
    }
    static const bool retain_workload_high_water = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_RETAIN_WORKLOAD_HIGH_WATER");
        return e && *e && *e != '0';
    }();
    const bool budget_tier_shrank = cache.parent && budget_capacity < cache.budget_capacity;
    if (adaptive_capacity && retain_workload_high_water && cache.parent &&
        !budget_tier_shrank && capacity < cache.capacity) {
        // n_required can vary from layer to layer even while the memory budget
        // is unchanged. Keep the workload high-water mark for that budget tier
        // instead of rebuilding and repopulating the cache on every dip. A real
        // budget decrease still takes the shrink path below.
        capacity = cache.capacity;
    }
    const int aggregate_capacity = capacity;
    if (state->moe_cache_resize_target < 0) {
        state->moe_cache_resize_target = aggregate_capacity;
    }
    static const bool preserve_grow = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_PRESERVE_GROW");
        // Growing a packed expert cache does not invalidate the experts that
        // already occupy its prefix. Preserve them by default so workload or
        // budget growth pays only for newly admitted experts. The switch stays
        // available for driver diagnostics and memory-pressure experiments.
        return !e || !*e || *e != '0';
    }();
    const bool capacity_must_grow = cache.parent && capacity > cache.capacity;
    const bool capacity_can_shrink = cache.parent && capacity < cache.capacity &&
                                     cache.capacity - capacity >= 4;
    if (capacity_must_grow || capacity_can_shrink) {
        // A route-upload event does not cover kernels that subsequently read the
        // cache parent. Synchronize once when the aggregate slot tier changes so
        // every old cache buffer is idle before the batch of per-tensor resizes.
        // Without this barrier the driver can retain an entire old tier while a
        // new tier is allocated, causing a multi-GiB transient allocation spike.
        if (state->moe_cache_resize_target != aggregate_capacity) {
            const cl_int finish_err = clFinish(backend_ctx->queue);
            if (finish_err != CL_SUCCESS) {
                GGML_LOG_ERROR("ggml_opencl moe cache: resize barrier failed slots=%d->%d err=%d\n",
                               state->moe_cache_resize_target, aggregate_capacity, finish_err);
                return false;
            }
            state->moe_cache_resize_target = aggregate_capacity;
            state->moe_cache_resize_syncs++;
        }
        if (cache.route_upload_event) {
            clWaitForEvents(1, &cache.route_upload_event);
            clReleaseEvent(cache.route_upload_event);
            cache.route_upload_event = nullptr;
        }
        const size_t old_bytes = (cache.q_slice_bytes + cache.d_slice_bytes) * (size_t) cache.capacity;
        if (preserve_grow && capacity > cache.capacity) {
            const size_t new_bytes = (cache.q_slice_bytes + cache.d_slice_bytes) * (size_t) capacity;
            cl_int grow_err = CL_SUCCESS;
            cl_mem grown_parent = clCreateBuffer(backend_ctx->context, CL_MEM_READ_ONLY,
                                                  new_bytes, nullptr, &grow_err);
            cl_mem grown_route = nullptr;
            if (grow_err == CL_SUCCESS) {
                grown_route = clCreateBuffer(backend_ctx->context, CL_MEM_READ_ONLY,
                                             sizeof(int32_t) * (size_t) n_selected, nullptr, &grow_err);
            }
            if (grow_err == CL_SUCCESS) {
                grow_err = clEnqueueCopyBuffer(backend_ctx->queue, cache.parent, grown_parent,
                                               0, 0, old_bytes, 0, nullptr, nullptr);
            }
            if (grow_err == CL_SUCCESS) {
                clReleaseMemObject(cache.parent);
                clReleaseMemObject(cache.route_slots);
                cache.parent = grown_parent;
                cache.route_slots = grown_route;
                cache.capacity = capacity;
                cache.budget_capacity = std::max(cache.budget_capacity, budget_capacity);
                cache.expert_for_slot.resize((size_t) capacity, -1);
                cache.last_used.resize((size_t) capacity, 0);
                state->moe_cache_bytes += new_bytes - old_bytes;
                state->moe_cache_grow_resizes++;
                state->moe_cache_grow_copy_bytes += old_bytes;
            } else {
                if (grown_parent) clReleaseMemObject(grown_parent);
                if (grown_route) clReleaseMemObject(grown_route);
                capacity = cache.capacity;
                GGML_LOG_WARN("ggml_opencl moe cache: preserve-grow failed tensor=%s slots=%d->%d err=%d\n",
                              src0->name, cache.capacity, capacity, grow_err);
            }
        } else {
            clReleaseMemObject(cache.parent);
            clReleaseMemObject(cache.route_slots);
            cache = {};
            state->moe_cache_bytes -= std::min(state->moe_cache_bytes, old_bytes);
        }
    }
    if (!cache.parent) {
        cache.capacity = capacity;
        cache.budget_capacity = budget_capacity;
        cache.raw_slice_bytes = raw_slice_bytes;
        cache.q_slice_bytes = q_slice_bytes;
        cache.d_slice_bytes = d_slice_bytes;
        cache.expert_for_slot.assign((size_t) capacity, -1);
        cache.last_used.assign((size_t) capacity, 0);
        cache.frequency.assign((size_t) n_experts, 0);
        cache.route_slots_host.assign((size_t) n_selected, -1);
        cache.parent = clCreateBuffer(backend_ctx->context, CL_MEM_READ_ONLY,
                                      (q_slice_bytes + d_slice_bytes) * (size_t) capacity, nullptr, &err);
        if (err == CL_SUCCESS) {
            cache.route_slots = clCreateBuffer(backend_ctx->context, CL_MEM_READ_ONLY,
                                               sizeof(int32_t) * (size_t) n_selected, nullptr, &err);
        }
        if (err != CL_SUCCESS || !cache.parent || !cache.route_slots) {
            if (cache.parent) clReleaseMemObject(cache.parent);
            if (cache.route_slots) clReleaseMemObject(cache.route_slots);
            cache = {};
            GGML_LOG_ERROR("ggml_opencl moe cache: allocation failed tensor=%s capacity=%d err=%d\n",
                           src0->name, capacity, err);
            return false;
        }
        state->moe_cache_bytes += (q_slice_bytes + d_slice_bytes) * (size_t) capacity;
        GGML_LOG_INFO("ggml_opencl moe cache: tensor=%s experts=%d slots=%d slice=%.2f MiB cache=%.2f MiB\n",
                      src0->name, n_experts, capacity,
                      raw_slice_bytes / 1024.0 / 1024.0,
                      (q_slice_bytes + d_slice_bytes) * capacity / 1024.0 / 1024.0);
    }
    if (budget_capacity > cache.budget_capacity) {
        cache.budget_capacity = budget_capacity;
    }

    static thread_local std::vector<unsigned char> raw_storage;
    raw_storage.resize(raw_slice_bytes + ELASTIC_MOE_DIRECT_PADDING);
    std::vector<int32_t> route_slots((size_t) n_selected, -1);
    enum class moe_cache_policy {
        lru,
        mru,
        lfu,
    };
    static const moe_cache_policy cache_policy = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_EXPERT_CACHE_POLICY");
        if (e && std::strcmp(e, "mru") == 0) return moe_cache_policy::mru;
        if (e && std::strcmp(e, "lfu") == 0) return moe_cache_policy::lfu;
        return moe_cache_policy::lru;
    }();
    static const uint64_t lfu_decay_accesses = []() {
        const char * e = std::getenv("GGML_ELASTIC_MOE_LFU_DECAY_ACCESSES");
        const long long v = e && *e ? std::atoll(e) : 1024;
        return (uint64_t) std::max<long long>(1, v);
    }();
    for (int i = 0; i < n_selected; ++i) {
        if (!active_routes[(size_t) i]) {
            continue;
        }
        const int expert_id = expert_ids[(size_t) i];
        if (expert_id < 0 || expert_id >= n_experts) {
            GGML_LOG_ERROR("ggml_opencl moe cache: invalid expert id=%d tensor=%s\n", expert_id, src0->name);
            return false;
        }
        cache.accesses++;
        cache.frequency[(size_t) expert_id]++;
        if (cache_policy == moe_cache_policy::lfu &&
            (cache.accesses % lfu_decay_accesses) == 0) {
            for (uint32_t & f : cache.frequency) {
                f = (f + 1) / 2;
            }
        }
        int slot = -1;
        for (int s = 0; s < cache.capacity; ++s) {
            if (cache.expert_for_slot[(size_t) s] == expert_id) {
                slot = s;
                break;
            }
        }
        if (slot >= 0) {
            state->moe_cache_hits++;
        } else {
            state->moe_cache_misses++;
            for (int s = 0; s < cache.capacity; ++s) {
                if (cache.expert_for_slot[(size_t) s] < 0) {
                    slot = s;
                    break;
                }
            }
            if (slot < 0) {
                auto protected_this_token = [&](int candidate) {
                    return std::find(route_slots.begin(), route_slots.begin() + i, candidate) !=
                           route_slots.begin() + i;
                };
                uint32_t best_freq = std::numeric_limits<uint32_t>::max();
                uint64_t best_age = cache_policy == moe_cache_policy::mru
                    ? 0
                    : std::numeric_limits<uint64_t>::max();
                for (int s = 0; s < cache.capacity; ++s) {
                    if (protected_this_token(s)) continue;
                    const int resident_expert = cache.expert_for_slot[(size_t) s];
                    const uint32_t freq = resident_expert >= 0 ? cache.frequency[(size_t) resident_expert] : 0;
                    const uint64_t age = cache.last_used[(size_t) s];
                    const bool better =
                        (cache_policy == moe_cache_policy::mru && (slot < 0 || age > best_age)) ||
                        (cache_policy == moe_cache_policy::lru && age < best_age) ||
                        (cache_policy == moe_cache_policy::lfu &&
                         (freq < best_freq || (freq == best_freq && age < best_age)));
                    if (better) {
                        slot = s;
                        best_freq = freq;
                        best_age = age;
                    }
                }
            }
            if (slot < 0) {
                GGML_LOG_ERROR("ggml_opencl moe cache: no replaceable slot tensor=%s expert=%d capacity=%d\n",
                               src0->name, expert_id, cache.capacity);
                return false;
            }

            const unsigned char * mmap_src = (const unsigned char *) bm->host_ptr +
                                             (size_t) expert_id * raw_slice_bytes;
            const std::shared_ptr<elastic_moe_prefetch_task> prefetched_task =
                ggml_opencl_consume_moe_prefetch(state, wbm_idx, expert_id);
            const bool prefetched = prefetched_task && prefetched_task->data_size == raw_slice_bytes;
            bool loaded = prefetched;
            unsigned char * raw = ggml_opencl_moe_direct_ptr(raw_storage.data(), 0);
            const auto io_t0 = std::chrono::steady_clock::now();
            if (!loaded) {
                auto reg = llama_mmap_registry_find(mmap_src);
                if (!reg.filename.empty()) {
                    const size_t file_offset = (const char *) mmap_src - (const char *) reg.base;
                    raw = ggml_opencl_moe_direct_ptr(raw_storage.data(), file_offset);
                    loaded = llama_pread_direct(reg.filename.c_str(), raw, file_offset, raw_slice_bytes) == 0;
                }
            }
            const auto io_t1 = std::chrono::steady_clock::now();
            if (loaded && !prefetched) {
                state->moe_cache_disk_bytes += raw_slice_bytes;
                state->moe_cache_disk_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(io_t1 - io_t0).count();
            }
            if (!loaded) {
                std::memcpy(raw, mmap_src, raw_slice_bytes);
            }
            const auto upload_t0 = std::chrono::steady_clock::now();
            err = clEnqueueWriteBuffer(backend_ctx->queue, cache.parent, CL_TRUE,
                                       (size_t) slot * raw_slice_bytes, raw_slice_bytes,
                                       prefetched ? prefetched_task->data : raw,
                                       0, nullptr, nullptr);
            state->moe_cache_upload_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - upload_t0).count();
            if (prefetched && prefetched_task->storage) {
                std::lock_guard<std::mutex> prefetch_lock(state->moe_prefetch_mtx);
                state->moe_prefetch_free_buffers.emplace_back(
                    prefetched_task->data_size, std::move(prefetched_task->storage));
                prefetched_task->data = nullptr;
            }
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("ggml_opencl moe cache: upload failed tensor=%s expert=%d slot=%d err=%d\n",
                               src0->name, expert_id, slot, err);
                return false;
            }
            cache.expert_for_slot[(size_t) slot] = expert_id;
        }
        cache.last_used[(size_t) slot] = ++state->moe_cache_clock;
        route_slots[(size_t) i] = slot;
    }

    if (cache.route_upload_event) {
        clWaitForEvents(1, &cache.route_upload_event);
        clReleaseEvent(cache.route_upload_event);
        cache.route_upload_event = nullptr;
    }
    cache.route_slots_host = route_slots;
    const auto route_t0 = std::chrono::steady_clock::now();
    err = clEnqueueWriteBuffer(backend_ctx->queue, cache.route_slots, CL_FALSE,
                               0, cache.route_slots_host.size() * sizeof(int32_t),
                               cache.route_slots_host.data(), 0, nullptr,
                               &cache.route_upload_event);
    state->moe_cache_route_upload_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - route_t0).count();
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml_opencl moe cache: upload route slots failed tensor=%s err=%d\n", src0->name, err);
        return false;
    }
    out->parent = cache.parent;
    out->route_slots = cache.route_slots;
    out->slot_stride = q_slice_bytes + d_slice_bytes;
    out->q_offset = d_slice_bytes;
    llama_weight_runtime_mark_resident(src0->name, LLAMA_WEIGHT_RUNTIME_GPU);
    state->moe_cache_prepare_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - prepare_t0).count();
    state->moe_cache_prepare_calls++;
    return true;
}

static void opencl_profile_stage(const char *kind, const char *name, int idx,
                                 size_t bytes, double ms, int ok,
                                 const char *extra = "");

static std::string opencl_name_for_idx(ggml_opencl_elastic_state *s, int idx) {
    if (!s || idx < 0) return {};
    std::lock_guard<std::mutex> lk(s->sched_mtx);
    auto it = s->wbm_to_name.find(idx);
    return it == s->wbm_to_name.end() ? std::string{} : it->second;
}

static void opencl_wait_async_xform(ggml_opencl_elastic_state *s, int idx);

static bool ggml_opencl_moe_q4_cache_manages_idx(ggml_opencl_elastic_state * s, int idx) {
    if (!s || idx < 0 || !ggml_opencl_moe_expert_cache_enabled()) return false;
    std::lock_guard<std::mutex> lock(s->sched_mtx);
    return s->moe_q4_packed_indices.find(idx) != s->moe_q4_packed_indices.end();
}

static int ggml_opencl_release_moe_q4_packed_idx(ggml_opencl_elastic_state * s, int idx) {
    if (!ggml_opencl_moe_q4_cache_manages_idx(s, idx)) return -2;
    opencl_wait_async_xform(s, idx);
    const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
    if (!bm || !bm->resident) return 0;
    const size_t bytes = bm->byte_size;
    const int released = elastic::wbmcl_evict_batch(&s->octx, &idx, 1);
    if (released <= 0) return -3;
    const std::string name = opencl_name_for_idx(s, idx);
    if (!name.empty()) {
        llama_weight_runtime_mark_evicted(name.c_str(), LLAMA_WEIGHT_RUNTIME_GPU);
    }
    s->n_evicts_total += 1;
    s->moe_packed_parent_evictions += 1;
    s->moe_packed_parent_evicted_bytes += bytes;
    return 0;
}

static void ggml_opencl_release_all_moe_q4_packed(ggml_opencl_elastic_state * s) {
    if (!s || !ggml_opencl_moe_expert_cache_enabled()) return;
    std::vector<int> resident;
    size_t bytes = 0;
    {
        std::lock_guard<std::mutex> lock(s->sched_mtx);
        resident.reserve(s->moe_q4_packed_indices.size());
        for (int idx : s->moe_q4_packed_indices) {
            const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
            if (bm && bm->resident) {
                resident.push_back(idx);
                bytes += bm->byte_size;
            }
        }
    }
    if (resident.empty()) return;
    for (int idx : resident) {
        opencl_wait_async_xform(s, idx);
    }
    const int released = elastic::wbmcl_evict_batch(&s->octx, resident.data(), (int) resident.size());
    if (released <= 0) return;
    for (int i = 0; i < released && i < (int) resident.size(); ++i) {
        const std::string name = opencl_name_for_idx(s, resident[(size_t) i]);
        if (!name.empty()) {
            llama_weight_runtime_mark_evicted(name.c_str(), LLAMA_WEIGHT_RUNTIME_GPU);
        }
    }
    s->n_evicts_total += (uint64_t) released;
    s->moe_packed_parent_evictions += (uint64_t) released;
    s->moe_packed_parent_evicted_bytes += bytes;
    GGML_LOG_INFO("ggml_opencl moe cache: released %d packed parents (%.1f MiB) before slice-cache execution\n",
                  released, bytes / 1024.0 / 1024.0);
}

static void opencl_wait_async_xform(ggml_opencl_elastic_state *s, int idx) {
    if (!s || idx < 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(s->async_xform_mtx);
    if (s->async_xform_inflight.find(idx) != s->async_xform_inflight.end()) {
        s->async_xform_waits++;
    }
    s->async_xform_cv.wait(lk, [s, idx]() {
        return s->async_xform_inflight.find(idx) == s->async_xform_inflight.end();
    });
    const auto t1 = std::chrono::steady_clock::now();
    s->async_xform_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

static bool opencl_async_xform_contains(ggml_opencl_elastic_state *s, int idx) {
    if (!s || idx < 0) return false;
    std::lock_guard<std::mutex> lk(s->async_xform_mtx);
    return s->async_xform_inflight.find(idx) != s->async_xform_inflight.end();
}

static int opencl_sched_transform_request(const char *name, llama_weight_transform_kind kind, void * /*ud*/);

static void opencl_start_async_xform_worker(ggml_opencl_elastic_state *s) {
    if (!s) return;
    {
        std::lock_guard<std::mutex> lk(s->async_xform_mtx);
        if (s->async_xform_worker_started) return;
        s->async_xform_worker_started = true;
        const char *max_env = std::getenv("GGML_ELASTIC_ASYNC_XFORM_MAX_INFLIGHT");
        long long max_v = max_env && *max_env ? atoll(max_env) : 1;
        if (max_v < 1) max_v = 1;
        s->async_xform_max_inflight = static_cast<size_t>(max_v);
    }
    std::thread([s]() {
        for (;;) {
            int idx = -1;
            {
                std::unique_lock<std::mutex> lk(s->async_xform_mtx);
                s->async_xform_cv.wait(lk, [s]() { return !s->async_xform_queue.empty(); });
                idx = s->async_xform_queue.front();
                s->async_xform_queue.pop_front();
            }
            const std::string weight_name = opencl_name_for_idx(s, idx);
            const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
            const int rc_bg = (bm && bm->resident) ? 0 : elastic::wbmcl_transform_backend(&s->octx, idx);
            if (rc_bg == 0 && !weight_name.empty()) {
                llama_weight_runtime_mark_resident(weight_name.c_str(), LLAMA_WEIGHT_RUNTIME_GPU);
            }
            {
                std::lock_guard<std::mutex> lk(s->async_xform_mtx);
                s->async_xform_inflight.erase(idx);
                s->async_xform_completed++;
            }
            s->async_xform_cv.notify_all();
        }
    }).detach();
}

static int opencl_plan_aware_victim(const elastic::weight_buffer_manager *wbm,
                                    int exclude_idx,
                                    void *ud) {
    auto *s = static_cast<ggml_opencl_elastic_state *>(ud);
    if (!wbm || !s) return -1;
    int victim = -1;
    if (wbm->evict_mru) {
        uint64_t newest = 0;
        bool found = false;
        for (const auto &b : wbm->blocks) {
            if (!b.resident || b.is_pinned || b.block_idx == exclude_idx) continue;
            if (opencl_async_xform_contains(s, b.block_idx)) continue;
            const std::string name = opencl_name_for_idx(s, b.block_idx);
            if (!name.empty() &&
                llama_weight_runtime_desired_query(name.c_str()) == LLAMA_WEIGHT_RUNTIME_GPU) {
                continue;
            }
            if (!found || b.last_used_token > newest) {
                newest = b.last_used_token;
                victim = b.block_idx;
                found = true;
            }
        }
    } else {
        uint64_t oldest = static_cast<uint64_t>(-1);
        for (const auto &b : wbm->blocks) {
            if (!b.resident || b.is_pinned || b.block_idx == exclude_idx) continue;
            if (opencl_async_xform_contains(s, b.block_idx)) continue;
            const std::string name = opencl_name_for_idx(s, b.block_idx);
            if (!name.empty() &&
                llama_weight_runtime_desired_query(name.c_str()) == LLAMA_WEIGHT_RUNTIME_GPU) {
                continue;
            }
            if (b.last_used_token < oldest) {
                oldest = b.last_used_token;
                victim = b.block_idx;
            }
        }
    }
    return victim;
}

// === Runtime scheduler handlers (registered with llama-mmap registry) ===
static std::vector<int> opencl_sched_indices_for_name(
        ggml_opencl_elastic_state *s, const char *name) {
    if (!s || !name) return {};
    std::lock_guard<std::mutex> lk(s->sched_mtx);
    auto many = s->name_to_wbms.find(name);
    if (many != s->name_to_wbms.end() && !many->second.empty()) {
        return many->second;
    }
    auto one = s->name_to_wbm.find(name);
    return one == s->name_to_wbm.end() ? std::vector<int>{}
                                       : std::vector<int>{one->second};
}

static bool opencl_sched_residency_query(const char *name, void * /*ud*/) {
    if (!name) return false;
    auto *s = ggml_opencl_elastic();
    const auto indices = opencl_sched_indices_for_name(s, name);
    if (indices.empty()) return false;
    for (int idx : indices) {
        const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
        if (bm && bm->resident) continue;
        bool cached = false;
        if (ggml_opencl_moe_expert_cache_enabled()) {
            std::lock_guard<std::mutex> cache_lock(s->moe_cache_mtx);
            auto cache = s->moe_q4_cache.find(idx);
            cached = cache != s->moe_q4_cache.end() && cache->second.parent != nullptr;
        }
        if (!cached) return false;
    }
    return true;
}

static uint32_t opencl_sched_state_query(const char *name, void * /*ud*/) {
    if (!name) return 0;
    auto *s = ggml_opencl_elastic();
    const auto indices = opencl_sched_indices_for_name(s, name);
    if (indices.empty()) return 0;
    uint32_t flags = 0;
    bool all_disk = true;
    bool all_gpu = true;
    bool all_cpu_raw = true;
    for (int idx : indices) {
        const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
        all_disk = all_disk && bm && bm->host_ptr;
        bool gpu = bm && bm->resident;
        if (!gpu && ggml_opencl_moe_expert_cache_enabled()) {
            std::lock_guard<std::mutex> cache_lock(s->moe_cache_mtx);
            auto cache = s->moe_q4_cache.find(idx);
            gpu = cache != s->moe_q4_cache.end() && cache->second.parent;
        }
        all_gpu = all_gpu && gpu;
        bool cpu_raw = false;
        if (bm) {
            std::lock_guard<std::mutex> staging_lock(s->octx.host_staging_mtx);
            auto staged = s->octx.host_staging_by_idx.find(idx);
            cpu_raw = staged != s->octx.host_staging_by_idx.end() &&
                      staged->second.size() >= bm->byte_size;
        }
        all_cpu_raw = all_cpu_raw && cpu_raw;
    }
    if (all_disk) flags |= LLAMA_WEIGHT_STATE_DISK_AVAILABLE;
    if (all_gpu) flags |= LLAMA_WEIGHT_STATE_GPU_COMPUTE_RESIDENT;
    if (all_cpu_raw) flags |= LLAMA_WEIGHT_STATE_CPU_RAW_RESIDENT;
    return flags;
}

static void * opencl_sched_host_ptr_query(const char *name, void * /*ud*/) {
    if (!name) return nullptr;
    auto *s = ggml_opencl_elastic();
    const auto indices = opencl_sched_indices_for_name(s, name);
    if (indices.empty()) return nullptr;
    const int idx = indices.front();
    const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
    return bm ? bm->host_ptr : nullptr;
}

static int opencl_sched_movement_request(const char *name, bool evict, void * /*ud*/) {
    if (!name) return -1;
    auto *s = ggml_opencl_elastic();
    const auto indices = opencl_sched_indices_for_name(s, name);
    if (indices.empty()) return -2;
    if (evict) {
        static const bool runtime_mru_cache = []() {
            const char * e = std::getenv("LLAMA_ELASTIC_RUNTIME_MRU_CACHE");
            return e && *e && *e != '0';
        }();
        if (!runtime_mru_cache &&
            llama_weight_runtime_desired_query(name) ==
                LLAMA_WEIGHT_RUNTIME_GPU) {
            return 0;
        }
        std::vector<int> resident;
        for (int idx : indices) {
            if (ggml_opencl_moe_q4_cache_manages_idx(s, idx)) {
                ggml_opencl_release_moe_q4_packed_idx(s, idx);
                continue;
            }
            opencl_wait_async_xform(s, idx);
            const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
            if (bm && bm->resident) resident.push_back(idx);
        }
        if (!resident.empty()) {
            const int released = elastic::wbmcl_evict_batch(
                &s->octx, resident.data(), (int) resident.size());
            s->n_evicts_total += released > 0 ? released : 0;
            if (released > 0) llama_weight_runtime_mark_evicted(name, LLAMA_WEIGHT_RUNTIME_GPU);
        }
        return 0;
    }
    for (int idx : indices) {
        if (ggml_opencl_moe_q4_cache_manages_idx(s, idx) &&
            llama_weight_runtime_desired_query(name) != LLAMA_WEIGHT_RUNTIME_CPU) {
            ggml_opencl_release_moe_q4_packed_idx(s, idx);
            continue;
        }
        const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
        if (bm && bm->resident) continue;
        const int rc = elastic::wbmcl_ensure_resident(&s->octx, idx);
        if (rc != 0) return -4;
    }
    llama_weight_runtime_mark_resident(name, LLAMA_WEIGHT_RUNTIME_GPU);
    return 0;
}

static int opencl_sched_stage_request(const char *name, const char *stage, void * /*ud*/) {
    if (!name || !stage) return -1;
    auto *s = ggml_opencl_elastic();
    const auto indices = opencl_sched_indices_for_name(s, name);
    if (indices.empty()) return -2;
    const int idx = indices.front();
    if (strcmp(stage, "load_cpu") == 0 || strcmp(stage, "prepare_cpu") == 0) {
        return -2;
    }
    const bool gpu_materialization_stage =
        strcmp(stage, "load") == 0 || strcmp(stage, "load_gpu") == 0 ||
        strcmp(stage, "transfer") == 0 || strcmp(stage, "dma") == 0 ||
        strcmp(stage, "prepare") == 0 || strcmp(stage, "prepare_gpu") == 0 ||
        strcmp(stage, "materialize") == 0;
    if (gpu_materialization_stage && indices.size() == 1 &&
        ggml_opencl_moe_q4_cache_manages_idx(s, idx)) {
        // A plan may place a dynamic expert tensor on GPU for compute routing,
        // but loading its full packed parent duplicates the slice cache and can
        // add multiple GiB during a budget transition.
        s->moe_packed_stage_skips++;
        return ggml_opencl_release_moe_q4_packed_idx(s, idx);
    }
    size_t bytes = 0;
    for (int part_idx : indices) {
        const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, part_idx);
        if (bm) bytes += bm->byte_size;
    }
    const auto t0 = s->profile_csv
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
    int rc = -3;
    if (strcmp(stage, "load") == 0 || strcmp(stage, "load_gpu") == 0) {
        rc = 0;
        for (int part_idx : indices) {
            if (elastic::wbmcl_load_host_async(&s->octx, part_idx) != 0) rc = -3;
        }
        if (s->profile_csv) {
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            opencl_profile_stage("LOAD", name, idx, bytes, ms, rc == 0, "plan_stage");
        }
        return rc;
    }
    if (strcmp(stage, "transfer") == 0 || strcmp(stage, "dma") == 0) {
        rc = 0;
        for (int part_idx : indices) {
            if (elastic::wbmcl_dma_to_backend(&s->octx, part_idx) != 0) rc = -3;
        }
        if (s->profile_csv) {
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            opencl_profile_stage("TRANSFER", name, idx, bytes, ms, rc == 0, "plan_stage");
        }
        return rc;
    }
    if (strcmp(stage, "prepare") == 0 || strcmp(stage, "prepare_gpu") == 0 ||
        strcmp(stage, "materialize") == 0) {
        // Route plan-level PREPARE through the paced transform worker instead
        // of the lower-level async SOA reload queue.  The lower queue is still
        // the correctness primitive, but it can accept many reloads at once;
        // the transform worker gives the runtime a small in-flight window so
        // conversion work does not flood the device while decode kernels are
        // trying to make forward progress.
        rc = opencl_sched_transform_request(name, LLAMA_WEIGHT_TRANSFORM_GPU_CONVERT, nullptr);
        if (s->profile_csv) {
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            opencl_profile_stage("PREPARE", name, idx, bytes, ms, rc == 0, "plan_stage");
        }
        return rc;
    }
    return -3;
}

static int opencl_sched_transform_request(const char *name, llama_weight_transform_kind kind, void * /*ud*/) {
    if (!name) return -1;
    if (kind == LLAMA_WEIGHT_TRANSFORM_NONE) return 0;
    if (kind != LLAMA_WEIGHT_TRANSFORM_GPU_CONVERT) return -2;
    auto *s = ggml_opencl_elastic();
    const auto indices = opencl_sched_indices_for_name(s, name);
    if (indices.empty()) return -2;
    const int idx = indices.front();
    if (indices.size() == 1 && ggml_opencl_moe_q4_cache_manages_idx(s, idx)) {
        s->moe_packed_stage_skips++;
        return ggml_opencl_release_moe_q4_packed_idx(s, idx);
    }
    size_t bytes = 0;
    for (int part_idx : indices) {
        const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, part_idx);
        if (bm) bytes += bm->byte_size;
    }
    static const bool s_async_xform_stage = []() {
        const char *e = std::getenv("GGML_ELASTIC_ASYNC_STAGE_PREPARE");
        if (e && *e) return *e != '0';
        e = std::getenv("GGML_ELASTIC_ASYNC_XFORM_STAGE");
        return e && *e && *e != '0';
    }();
    if (s_async_xform_stage) {
        opencl_start_async_xform_worker(s);
        {
            std::lock_guard<std::mutex> lk(s->async_xform_mtx);
            static const size_t s_max_pending = []() {
                const char *e = std::getenv("GGML_ELASTIC_ASYNC_XFORM_MAX_PENDING");
                long long v = e && *e ? atoll(e) : 0;
                return v > 0 ? static_cast<size_t>(v) : static_cast<size_t>(0);
            }();
            for (int part_idx : indices) {
                const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, part_idx);
                if ((bm && bm->resident) ||
                    s->async_xform_inflight.find(part_idx) != s->async_xform_inflight.end()) {
                    continue;
                }
                if (s_max_pending > 0 &&
                    s->async_xform_inflight.size() >= s_max_pending) break;
                s->async_xform_inflight.insert(part_idx);
                s->async_xform_queue.push_back(part_idx);
                s->async_xform_enqueued++;
            }
            s->async_xform_max_pending_seen = std::max(s->async_xform_max_pending_seen,
                                                       s->async_xform_inflight.size());
        }
        s->async_xform_cv.notify_all();
        if (s->profile_csv) {
            opencl_profile_stage("XFORM", name, idx, bytes, 0.0, 1, "plan_stage_async_submit");
        }
        return 0;
    }
    const auto t0 = s->profile_csv
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
    int rc = 0;
    for (int part_idx : indices) {
        if (elastic::wbmcl_transform_backend(&s->octx, part_idx) != 0) rc = -3;
    }
    if (rc == 0) llama_weight_runtime_mark_resident(name, LLAMA_WEIGHT_RUNTIME_GPU);
    if (s->profile_csv) {
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        opencl_profile_stage("XFORM", name, idx, bytes, ms, rc == 0, "plan_stage");
    }
    return rc;
}

// Budget provider: 把 BudgetWatcher 当前预算 (MB) 暴露给 llama_context 的 runtime
// scheduler, 让 scheduler 跟 elastic evict target 用同一个内存信号源 (而非 /proc/meminfo)。
// bw 没启用时返 -1 → scheduler fallback /proc/meminfo。
static int64_t opencl_sched_budget_query(void * /*ud*/) {
    auto *s = ggml_opencl_elastic();
    if (!s->bw_inited) return -1;
    return (int64_t) elastic::budget_watcher_get(&s->bw);
}

static void opencl_sched_budget_reset(void * /*ud*/) {
    auto *s = ggml_opencl_elastic();
    if (!s->bw_inited) return;
    elastic::budget_watcher_reset_clock(&s->bw);
    GGML_LOG_INFO("ggml_opencl elastic: BudgetWatcher replay clock reset, B(t)=%zu MB\n",
                  elastic::budget_watcher_get(&s->bw));
}

static void opencl_sched_register_once() {
    auto *s = ggml_opencl_elastic();
    if (s->sched_registered) return;
    s->sched_registered = true;
    llama_weight_residency_register(opencl_sched_residency_query, nullptr);
    llama_weight_state_register    (opencl_sched_state_query,     nullptr);
    llama_weight_movement_register (opencl_sched_movement_request, nullptr);
    llama_weight_stage_register    (opencl_sched_stage_request,    nullptr);
    llama_weight_transform_register(opencl_sched_transform_request, nullptr);
    llama_weight_host_ptr_register (opencl_sched_host_ptr_query, nullptr);
    llama_working_set_register(
        [](const char * kind, llama_working_set_runtime_state * out, void *) -> bool {
            if (!kind || !out) return false;
            auto * state = ggml_opencl_elastic();
            if (std::strcmp(kind, "weights") == 0) {
                size_t target = state->dynamic_target
                    ? (([&] {
                          const size_t budget = elastic::budget_watcher_get(&state->bw) * size_t(1024 * 1024);
                          const size_t reserved = state->kv_bytes + state->misc_overhead;
                          return (budget > reserved ? budget - reserved : 0) + state->extra_target_bytes;
                      })())
                    : state->static_target_bytes;
                if (ggml_opencl_moe_expert_cache_enabled()) {
                    target = target > state->moe_cache_bytes ? target - state->moe_cache_bytes : 0;
                }
                out->active_capacity = state->wbm.n_resident;
                out->target_capacity = (int) (target / 1024 / 1024);
                out->misses = state->n_reloads_total;
                out->capacity_changes = state->n_evicts_total;
                out->resident_bytes = state->wbm.resident_bytes;
                return true;
            }
            if (std::strcmp(kind, "expert_slices") != 0) return false;
            std::lock_guard<std::mutex> lock(state->moe_cache_mtx);
            out->active_capacity = state->moe_cache_active_capacity;
            out->pending_capacity = state->moe_cache_pending_capacity;
            out->target_capacity = state->moe_cache_plan_capacity;
            out->observed_required_capacity = state->moe_cache_required_high_water;
            out->hits = state->moe_cache_hits;
            out->misses = state->moe_cache_misses;
            out->accesses = out->hits + out->misses;
            out->capacity_changes = state->moe_cache_grow_resizes;
            out->resident_bytes = state->moe_cache_bytes;
            return true;
        },
        [](const char * kind, int target_capacity, void *) -> int {
            if (!kind || std::strcmp(kind, "expert_slices") != 0) return -2;
            if (target_capacity < -1) return -1;
            auto * state = ggml_opencl_elastic();
            std::lock_guard<std::mutex> lock(state->moe_cache_mtx);
            state->moe_cache_plan_capacity = target_capacity;
            state->moe_cache_required_high_water = 0;
            state->moe_cache_pending_capacity = -1;
            state->moe_cache_pending_hits = 0;
            return 0;
        },
        nullptr);
    llama_budget_register          (opencl_sched_budget_query,    nullptr);
    llama_budget_reset_register    (opencl_sched_budget_reset,    nullptr);
}

// 提取 tensor 名后缀用于 profile 聚合：blk.<N>.<X>.weight → X
// 非 blk. 前缀的（token_embd/output/output_norm 等）直接去掉 .weight 后缀返回
static std::string ggml_opencl_tensor_suffix(const char *name) {
    if (!name) return std::string();
    std::string s(name);
    if (s.size() >= 4 && s.compare(0, 4, "blk.") == 0) {
        size_t dot2 = s.find('.', 4);
        if (dot2 != std::string::npos) s = s.substr(dot2 + 1);
    }
    if (s.size() > 7 && s.compare(s.size() - 7, 7, ".weight") == 0) {
        s = s.substr(0, s.size() - 7);
    }
    return s;
}

static void opencl_profile_stage(const char *kind, const char *name, int idx,
                                 size_t bytes, double ms, int ok,
                                 const char *extra) {
    if (!elastic::profile_enabled()) return;
    elastic::profile_record rec;
    rec.backend   = "OpenCL";
    rec.kind      = kind;
    rec.name      = name ? name : "";
    rec.weight_id = idx;
    rec.bytes     = bytes;
    rec.ms        = ms;
    rec.ok        = ok;
    rec.extra     = extra ? extra : "";
    elastic::profile_write(rec);
}

static void opencl_profile_compute(const ggml_tensor *node, int op_id,
                                   double ms, int ok) {
    if (!elastic::profile_enabled() || !node) return;
    elastic::profile_record rec;
    rec.backend = "OpenCL";
    rec.kind    = "COMPUTE";
    rec.name    = node->name;
    rec.op      = ggml_op_name(node->op);
    rec.quant   = ggml_type_name(node->type);
    rec.op_id   = op_id;
    rec.ne[0]   = node->ne[0];
    rec.ne[1]   = node->ne[1];
    rec.ne[2]   = node->ne[2];
    rec.ne[3]   = node->ne[3];
    rec.bytes   = ggml_nbytes(node);
    rec.ms      = ms;
    rec.ok      = ok;
    rec.extra   = "clFinish_profiled";
    elastic::profile_write(rec);
}

static size_t ggml_opencl_env_mb(const char *name, size_t def_mb) {
    const char *v = std::getenv(name);
    if (!v || !*v) return def_mb * 1024 * 1024;
    long long n = atoll(v);
    if (n < 0) n = 0;
    return static_cast<size_t>(n) * 1024 * 1024;
}

// 把新分配的 cl_mem 回填到对应 buffer_context 的 slot；buffer_context 完整类型
// 在文件靠后定义，所以这里仅做前向声明，实现放在 buffer_context 之后。
static void ggml_opencl_elastic_update_ctx_slot(ggml_backend_buffer_t buf,
                                                int slot, cl_mem new_buf);

static void ggml_opencl_elastic_lazy_init(cl_context cl_ctx, cl_command_queue queue) {
    auto *s = ggml_opencl_elastic();
    if (s->wbm_inited) return;

    if (elastic::wbm_init(&s->wbm, 0) != 0) {
        GGML_LOG_ERROR("ggml_opencl elastic: wbm_init 失败\n");
        return;
    }
    s->wbm.victim_fn = opencl_plan_aware_victim;
    s->wbm.victim_ud = s;
    // GGML_ELASTIC_EVICT_POLICY=mru|lru（默认 mru）。LLM decode 是 round-robin
    // 访问，cache < model 时 LRU 会 100% miss（每次 evict 的恰好是即将再用的），
    // MRU 反而能让命中率随 cache/model 比例线性提升。wbm.evict_mru 默认已 true，
    // 这里只处理 env 显式切回 lru 调试。
    if (const char *p = std::getenv("GGML_ELASTIC_EVICT_POLICY")) {
        if (std::string(p) == "lru") {
            s->wbm.evict_mru = false;
            GGML_LOG_INFO("ggml_opencl elastic: 切回 LRU 驱逐策略 (调试)\n");
        } else if (std::string(p) == "mru") {
            s->wbm.evict_mru = true;
            GGML_LOG_INFO("ggml_opencl elastic: 启用 MRU 驱逐策略\n");
        }
    }
    // 默认走单队列同步路径：在 Adreno + ggml-opencl 上 enqueue 开销 + barrier
    // 同步反而拖累整体（实测 Test 3 上 async eval 时间 2223 ms/tok vs sync
    // 1782 ms/tok，慢 25%）。GGML_ELASTIC_ASYNC_XFER=1 可显式开异步路径，
    // 留作未来调优时复用——其它芯片或更激进的 prefetch 策略下可能有收益。
    // prefetch 也需要 xfer_queue；先把 lookahead 读出来，下面建 queue 的判断要用
    int prefetch_la_env = 0;
    if (const char *pf = std::getenv("GGML_ELASTIC_PREFETCH")) {
        int v = atoi(pf);
        if (v > 0) prefetch_la_env = v;
    }
    s->prefetch_lookahead = prefetch_la_env;

    int gpu_pf_env = 0;
    if (const char *gp = std::getenv("GGML_ELASTIC_GPU_PREFETCH")) {
        int v = atoi(gp);
        if (v > 0) gpu_pf_env = v;
    }
    s->gpu_prefetch_lookahead = gpu_pf_env;

    static const bool device_timing_env = []() {
        const char *e = std::getenv("GGML_ELASTIC_DEVICE_TIMING");
        return e && *e && *e != '0';
    }();

    cl_command_queue xfer_q = nullptr;
    const bool need_xfer = prefetch_la_env > 0
        || gpu_pf_env > 0
        || ([]{ const char *a = std::getenv("GGML_ELASTIC_ASYNC_XFER"); return a && *a && *a != '0'; })()
        || ([]{ const char *a = std::getenv("GGML_ELASTIC_RELOAD_ON_XFER"); return a && *a && *a != '0'; })();
    if (need_xfer) {
        cl_int q_err = CL_SUCCESS;
        cl_device_id dev = nullptr;
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
        const cl_queue_properties xfer_props[] = {
            CL_QUEUE_PROPERTIES,
            device_timing_env ? (cl_queue_properties) CL_QUEUE_PROFILING_ENABLE : 0,
            0,
        };
        xfer_q = clCreateCommandQueueWithProperties(cl_ctx, dev,
                                                    device_timing_env ? xfer_props : nullptr,
                                                    &q_err);
        if (q_err != CL_SUCCESS) {
            GGML_LOG_ERROR("ggml_opencl elastic: 创建 xfer queue 失败 %d，退回单队列\n", q_err);
            xfer_q = nullptr;
            s->prefetch_lookahead = 0;
        } else {
            GGML_LOG_INFO("ggml_opencl elastic: 启用 xfer queue (prefetch=%d async_xfer=%s)\n",
                          prefetch_la_env,
                          std::getenv("GGML_ELASTIC_ASYNC_XFER") ? std::getenv("GGML_ELASTIC_ASYNC_XFER") : "0");
        }
    }
    if (elastic::wbmcl_init(&s->octx, &s->wbm, cl_ctx, queue, xfer_q) != 0) {
        GGML_LOG_ERROR("ggml_opencl elastic: wbmcl_init 失败\n");
        return;
    }
    // The OpenCL elastic state is process-global while the stage workers are
    // joinable. Register worker shutdown before the later timing/granularity
    // dump handlers so those reports run first and static mutexes are never
    // destroyed under a live worker at process exit.
    std::atexit([]() {
        auto * state = ggml_opencl_elastic();
        elastic::wbmcl_stop_async_workers(&state->octx);
        if (state->cut_compute_queue) {
            clFinish(state->cut_compute_queue);
            clReleaseCommandQueue(state->cut_compute_queue);
            state->cut_compute_queue = nullptr;
        }
    });
    s->octx.device_timing = device_timing_env;
    if (device_timing_env) {
        GGML_LOG_INFO("ggml_opencl elastic: GGML_ELASTIC_DEVICE_TIMING=1 (staged OpenCL event profiling)\n");
    }
    if (const char *e = std::getenv("GGML_ELASTIC_STAGE_DETAIL"); e && *e && *e != '0') {
        s->octx.stage_detail = true;
        GGML_LOG_INFO("ggml_opencl elastic: GGML_ELASTIC_STAGE_DETAIL=1 (host substage timing)\n");
    }
    // Elastic reload 默认走 O_DIRECT pread (绕 page cache, 模拟 model>RAM 真 disk
    // 成本). 只有 GGML_ELASTIC_DIRECT_IO=0 时回退 mmap source.
    // (SOA 量化路径在 reload_fn 里另有自己的 direct 逻辑.)
    static const bool direct_io_default = []() {
        const char * d = std::getenv("GGML_ELASTIC_DIRECT_IO");
        return !(d && *d == '0');
    }();
    if (direct_io_default) {
        s->octx.direct_read_fn = [s](const void *host_ptr, void *dst, size_t nbytes) -> int {
            auto t0 = std::chrono::steady_clock::now();
            s->octx.direct_read_calls++;
            auto reg = llama_mmap_registry_find(host_ptr);
            if (reg.filename.empty()) {
                s->octx.direct_read_fail++;
                return -1;
            }
            size_t file_offset = (const char *)host_ptr - (const char *)reg.base;
            int rc = llama_pread_direct(reg.filename.c_str(), dst, file_offset, nbytes);
            auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count();
            s->octx.direct_read_us += (uint64_t) dt;
            if (rc == 0) {
                s->octx.direct_read_ok++;
                s->octx.direct_read_bytes += nbytes;
            } else {
                s->octx.direct_read_fail++;
            }
            return rc;
        };
        GGML_LOG_INFO("ggml_opencl elastic: direct disk reload enabled by default (set GGML_ELASTIC_DIRECT_IO=0 to use mmap)\n");
    } else {
        GGML_LOG_INFO("ggml_opencl elastic: direct disk reload disabled (GGML_ELASTIC_DIRECT_IO=0, using mmap source)\n");
    }
    // Explicit LOAD stage host staging pool. LOAD allocates/reuses a CPU buffer,
    // Transfer consumes it, then the buffer returns to this size-based pool. This keeps
    // memory-growth replans from repeatedly malloc/free-ing staging memory.
    {
        size_t host_pool_mb = 256;
        if (const char *m = std::getenv("GGML_ELASTIC_HOST_STAGING_POOL_MB")) {
            host_pool_mb = (size_t) atoll(m);
        }
        s->octx.retain_host_staging = host_pool_mb > 0;
        s->octx.host_staging_pool_limit = host_pool_mb * 1024 * 1024;
        if (s->octx.retain_host_staging) {
            GGML_LOG_INFO("ggml_opencl elastic: host staging pool cap = %zu MB\n", host_pool_mb);
        } else {
            GGML_LOG_INFO("ggml_opencl elastic: host staging pool disabled\n");
        }
    }
    {
        const char *r = std::getenv("GGML_ELASTIC_CL_RETAIN");
        const bool retain_default = !(r && *r == '0');
        if (!retain_default) {
            GGML_LOG_INFO("ggml_opencl elastic: cl_mem retain pool disabled\n");
        } else {
            s->octx.retain_cl_mem = true;
            // GGML_ELASTIC_CL_RETAIN_MB=N 设 pool 上限（MB）；"auto" 让 backend
            // 根据 model 的 max_block_bytes 自算（首次 graph_compute 触发）；
            // 不设 = auto（默认，避免每次 reload 都 clCreateBuffer，同时限制 pool 规模）。
            if (const char *m = std::getenv("GGML_ELASTIC_CL_RETAIN_MB")) {
                if (std::string(m) == "auto") {
                    s->octx.cache_byte_limit = 0;
                    s->retain_cap_auto = true;
                    GGML_LOG_INFO("ggml_opencl elastic: cl_mem retain pool cap = AUTO (待 first graph_compute 算)\n");
                } else {
                    size_t mb = (size_t)atoll(m);
                    s->octx.cache_byte_limit = mb * 1024 * 1024;
                    s->retain_cap_auto = false;
                    GGML_LOG_INFO("ggml_opencl elastic: cl_mem retain pool cap = %zu MB\n", mb);
                }
            } else {
                s->octx.cache_byte_limit = 0;
                s->retain_cap_auto = true;
                GGML_LOG_INFO("ggml_opencl elastic: cl_mem retain pool cap = AUTO (default)\n");
            }
        }
    }
    if (false) {  // 占位避免重复 wbmcl_init 错误处理
        GGML_LOG_ERROR("dummy\n");
        return;
    }
    // GGML_ELASTIC_XFER_EXTRA=N：额外创建 N 条 xfer queue（总共 N+1 条 xfer）
    // wbmcl 在 N+1 条 queue 间 round-robin 发 write。micro-bench 实测 2 总
    // queue 数 (extra=1) 已经 ~2× 吞吐，超过 2 收益递减。N <= 4。
    // 兼容旧 env GGML_ELASTIC_DUAL_XFER=1 = N=1。
    int xfer_extra_n = 0;
    if (const char *x = std::getenv("GGML_ELASTIC_XFER_EXTRA")) {
        int v = atoi(x);
        if (v > 0) xfer_extra_n = v;
    } else if (const char *d = std::getenv("GGML_ELASTIC_DUAL_XFER"); d && *d && *d != '0') {
        xfer_extra_n = 1;
    }
    if (xfer_q && xfer_extra_n > 0) {
        if (xfer_extra_n > elastic::wbm_opencl_ctx::N_XFER_EXTRA) {
            xfer_extra_n = elastic::wbm_opencl_ctx::N_XFER_EXTRA;
        }
        cl_int q_err = CL_SUCCESS;
        cl_device_id dev = nullptr;
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
        int n_ok = 0;
        const cl_queue_properties xfer_props[] = {
            CL_QUEUE_PROPERTIES,
            device_timing_env ? (cl_queue_properties) CL_QUEUE_PROFILING_ENABLE : 0,
            0,
        };
        for (int i = 0; i < xfer_extra_n; ++i) {
            cl_command_queue q2 = clCreateCommandQueueWithProperties(cl_ctx, dev,
                                                                     device_timing_env ? xfer_props : nullptr,
                                                                     &q_err);
            if (q_err != CL_SUCCESS) {
                GGML_LOG_ERROR("ggml_opencl elastic: 第%d条额外 xfer queue 创建失败 %d\n", i + 1, q_err);
                break;
            }
            s->octx.xfer_extra[i] = q2;
            n_ok = i + 1;
        }
        s->octx.n_xfer_extra = n_ok;
        if (n_ok > 0) {
            GGML_LOG_INFO("ggml_opencl elastic: 启用 %d 条额外 xfer queue (总 %d 条)\n",
                          n_ok, n_ok + 1);
        }
    }
    s->wbm_inited = true;
    s->granularity = elastic::granularity_from_env();
    if (const char * unit_sync = std::getenv("GGML_ELASTIC_GPU_UNIT_SYNC")) {
        if (std::strcmp(unit_sync, "none") == 0) {
            s->granularity_unit_sync =
                ggml_opencl_elastic_state::unit_sync_mode::NONE;
        } else if (std::strcmp(unit_sync, "flush") == 0) {
            s->granularity_unit_sync =
                ggml_opencl_elastic_state::unit_sync_mode::FLUSH;
        } else if (std::strcmp(unit_sync, "finish") == 0) {
            s->granularity_unit_sync =
                ggml_opencl_elastic_state::unit_sync_mode::FINISH;
        } else {
            GGML_LOG_WARN(
                "ggml_opencl elastic: invalid GGML_ELASTIC_GPU_UNIT_SYNC=%s; "
                "using flush\n", unit_sync);
        }
    }
    if (s->granularity.mode == elastic::granularity_mode::MULTI_FUSED) {
        GGML_LOG_WARN(
            "ggml_opencl elastic: multi_fused is not implemented for OpenCL; "
            "falling back explicitly to multi (no fused layout/kernel)\n");
        s->granularity.mode = elastic::granularity_mode::MULTI;
    }
    if (s->granularity.explicitly_enabled) {
        GGML_LOG_INFO("ggml_opencl elastic: working-unit granularity=%s cut_parts=%d multi_tensors=%d\n",
                      elastic::granularity_mode_name(s->granularity.mode),
                      s->granularity.cut_parts, s->granularity.multi_tensors);
        const char * unit_sync_name =
            s->granularity_unit_sync ==
                    ggml_opencl_elastic_state::unit_sync_mode::FINISH
                ? "finish"
                : s->granularity_unit_sync ==
                          ggml_opencl_elastic_state::unit_sync_mode::FLUSH
                      ? "flush"
                      : "none";
        GGML_LOG_INFO(
            "ggml_opencl elastic: nonblocking unit completion mode=%s\n",
            unit_sync_name);
        std::atexit([]() {
            auto *st = ggml_opencl_elastic();
            std::fprintf(stderr,
                         "[elastic granularity] backend=opencl mode=%s units=%llu cut_ops=%llu "
                         "fallback=%llu peak_unit=%.2f MiB cut_tensors=%zu cut_parts=%zu wbm_total=%.2f MiB\n",
                         elastic::granularity_mode_name(st->granularity.mode),
                         (unsigned long long) st->granularity_units,
                         (unsigned long long) st->granularity_cut_ops,
                         (unsigned long long) st->granularity_fallback_ops,
                         st->granularity_peak_unit_bytes / 1024.0 / 1024.0,
                         st->granularity_cut_tensors, st->granularity_cut_parts,
                         elastic::wbm_total_bytes(&st->wbm) / 1024.0 / 1024.0);
            const char * unit_sync_name =
                st->granularity_unit_sync ==
                        ggml_opencl_elastic_state::unit_sync_mode::FINISH
                    ? "finish"
                    : st->granularity_unit_sync ==
                              ggml_opencl_elastic_state::unit_sync_mode::FLUSH
                          ? "flush"
                          : "none";
            std::fprintf(
                stderr,
                "[elastic granularity sync] backend=opencl mode=%s "
                "boundaries=%llu flushes=%llu finishes=%llu errors=%llu\n",
                unit_sync_name,
                (unsigned long long) st->granularity_unit_boundaries,
                (unsigned long long) st->granularity_unit_flushes,
                (unsigned long long) st->granularity_unit_finishes,
                (unsigned long long) st->granularity_unit_sync_errors);
            std::fprintf(
                stderr,
                "[elastic cut dual] enabled=%d candidates=%llu calls=%llu "
                "budget_fallbacks=%llu shape_fallbacks=%llu queue_errors=%llu "
                "image_creates=%llu image_cache_hits=%llu "
                "image_releases=%llu image_errors=%llu\n",
                []() {
                    const char * e =
                        std::getenv("GGML_ELASTIC_CUT_DUAL_COMPUTE");
                    return e && *e && *e != '0';
                }() ? 1 : 0,
                (unsigned long long) st->cut_dual_candidates,
                (unsigned long long) st->cut_dual_calls,
                (unsigned long long) st->cut_dual_budget_fallbacks,
                (unsigned long long) st->cut_dual_shape_fallbacks,
                (unsigned long long) st->cut_dual_queue_errors,
                (unsigned long long) st->cut_dual_image_creates,
                (unsigned long long) st->cut_dual_image_cache_hits,
                (unsigned long long) st->cut_dual_image_releases,
                (unsigned long long) st->cut_dual_image_errors);
        });
        static const bool unit_pipeline_enabled = []() {
            const char * e = std::getenv("GGML_ELASTIC_UNIT_PIPELINE");
            return e && *e && *e != '0';
        }();
        if (unit_pipeline_enabled) {
            GGML_LOG_INFO("ggml_opencl elastic: granularity-aware unit pipeline enabled\n");
            std::atexit([]() {
                auto * st = ggml_opencl_elastic();
                std::fprintf(stderr,
                    "[elastic unit pipeline] backend=opencl issued=%llu ready=%llu waits=%llu wait_ms=%.3f\n",
                    (unsigned long long) st->unit_pipeline_issued,
                    (unsigned long long) st->unit_pipeline_ready,
                    (unsigned long long) st->unit_pipeline_waits,
                    st->unit_pipeline_wait_us / 1000.0);
                std::fprintf(stderr,
                    "[elastic unit pipeline window] samples=%llu avg_units=%.2f max_units=%zu "
                    "avg_mib=%.2f max_mib=%.2f oversize=%llu\n",
                    (unsigned long long) st->unit_pipeline_window_samples,
                    st->unit_pipeline_window_samples
                        ? (double) st->unit_pipeline_window_units_total /
                            st->unit_pipeline_window_samples : 0.0,
                    st->unit_pipeline_window_units_max,
                    st->unit_pipeline_window_samples
                        ? (double) st->unit_pipeline_window_bytes_total /
                            st->unit_pipeline_window_samples / 1024.0 / 1024.0 : 0.0,
                    st->unit_pipeline_window_bytes_max / 1024.0 / 1024.0,
                    (unsigned long long) st->unit_pipeline_oversize_windows);
                std::fprintf(stderr,
                    "[elastic unit pipeline cross-graph] issued=%llu ready=%llu waits=%llu "
                    "staged_mib=%.2f transitions=%zu\n",
                    (unsigned long long) st->unit_pipeline_cross_issued,
                    (unsigned long long) st->unit_pipeline_cross_ready,
                    (unsigned long long) st->unit_pipeline_cross_waits,
                    st->unit_pipeline_cross_bytes / 1024.0 / 1024.0,
                    st->unit_pipeline_successors.size());
                std::fprintf(stderr,
                    "[elastic unit pipeline budget] samples=%llu violations=%llu "
                    "plan_protection_relaxations=%llu relaxed_mib=%.2f "
                    "resident_peak_mib=%.2f pinned_peak_mib=%.2f over_peak_mib=%.2f\n",
                    (unsigned long long) st->unit_pipeline_budget_samples,
                    (unsigned long long) st->unit_pipeline_budget_violations,
                    (unsigned long long)
                        st->unit_pipeline_plan_protection_relaxations,
                    st->unit_pipeline_plan_protection_relaxed_bytes /
                        1024.0 / 1024.0,
                    st->unit_pipeline_resident_bytes_peak / 1024.0 / 1024.0,
                    st->unit_pipeline_pinned_bytes_peak / 1024.0 / 1024.0,
                    st->unit_pipeline_over_budget_bytes_peak / 1024.0 / 1024.0);
            });
        }
    }
    s->profile_csv = elastic::profile_enabled();
    if (s->profile_csv) {
        GGML_LOG_INFO("ggml_opencl elastic: GGML_ELASTIC_PROFILE_CSV enabled\n");
    }
    s->t0 = std::chrono::steady_clock::now();

    s->kv_bytes      = ggml_opencl_env_mb("GGML_ELASTIC_KV_MB",   128);
    s->misc_overhead = ggml_opencl_env_mb("GGML_ELASTIC_MISC_MB", 256);
    const size_t safety_overhead = ggml_opencl_env_mb("GGML_ELASTIC_SAFETY_MB", 0);
    s->misc_overhead += safety_overhead;
    if (const char *iv = std::getenv("GGML_ELASTIC_EVICT_INTERVAL")) {
        int v = atoi(iv);
        if (v >= 1) s->evict_check_interval = v;
    }

    const char *csv = std::getenv("GGML_ELASTIC_BUDGET_CSV");
    if (csv && *csv) {
        if (elastic::budget_watcher_init(&s->bw, csv) == 0) {
            s->bw_inited = true;
            const size_t mfloor_bytes = s->bw.m_floor_mb * 1024 * 1024;
            const size_t kv_misc      = s->kv_bytes + s->misc_overhead;
            s->static_target_bytes    = mfloor_bytes > kv_misc ? mfloor_bytes - kv_misc : 0;
            // Dynamic mode: GGML_ELASTIC_DYNAMIC=1 让 target 跟着 B(t) 实时变.
            if (const char *d = std::getenv("GGML_ELASTIC_DYNAMIC"); d && *d && *d != '0') {
                s->dynamic_target = true;
            }
            GGML_LOG_INFO("ggml_opencl elastic: BudgetWatcher trace=%s 初始 B(t)=%zu MB "
                          "M_floor=%zu MB → static_target=%zu MB mode=%s (kv=%zu MB misc+safety=%zu MB)\n",
                          csv, elastic::budget_watcher_get(&s->bw), s->bw.m_floor_mb,
                          s->static_target_bytes / 1024 / 1024,
                          s->dynamic_target ? "DYNAMIC" : "static",
                          s->kv_bytes / 1024 / 1024, s->misc_overhead / 1024 / 1024);
        } else {
            GGML_LOG_ERROR("ggml_opencl elastic: BudgetWatcher 加载失败: %s\n", csv);
        }
    }

    // GGML_ELASTIC_TIMING=1：只记 reload host-issue 时间, 不像 PROFILE 那样
    // per-op clFinish 拖速度. 用于不带干扰地测 reload IO vs total wall time.
    if (const char *t = std::getenv("GGML_ELASTIC_TIMING"); t && *t && *t != '0') {
        s->timing = true;
        GGML_LOG_INFO("ggml_opencl elastic: GGML_ELASTIC_TIMING=1 (host reload 计时, 无 clFinish 干扰)\n");
        std::atexit([]() {
            auto *st = ggml_opencl_elastic();
            if (!st->timing) return;
            std::fprintf(stderr, "\n=== ggml_opencl elastic timing dump (no clFinish overhead) ===\n");
            double total_ms = st->t_reload_host_us / 1000.0;
            std::fprintf(stderr, "reload host-issue total: %.1f ms\n", total_ms);
            std::fprintf(stderr, "reload calls: %llu (avg %.2f ms/call)\n",
                         (unsigned long long)st->n_reloads_total,
                         st->n_reloads_total ? total_ms/st->n_reloads_total : 0);
            auto print_stage = [](const char *name, uint64_t calls, uint64_t ok,
                                  uint64_t us, size_t bytes) {
                const double ms = us / 1000.0;
                const double mb = bytes / 1024.0 / 1024.0;
                const double avg = calls ? ms / calls : 0.0;
                const double mbps = ms > 0 ? mb / (ms / 1000.0) : 0.0;
                std::fprintf(stderr,
                             "stage %-8s calls=%llu ok=%llu total=%.2f ms avg=%.3f ms MB=%.1f MB/s=%.1f\n",
                             name,
                             (unsigned long long) calls,
                             (unsigned long long) ok,
                             ms, avg, mb, mbps);
            };
            print_stage("load", st->octx.stage_load_calls, st->octx.stage_load_ok,
                        st->octx.stage_load_us, st->octx.stage_load_bytes);
            print_stage("transfer", st->octx.stage_transfer_calls, st->octx.stage_transfer_ok,
                        st->octx.stage_transfer_us, st->octx.stage_transfer_bytes);
            print_stage("xform", st->octx.stage_xform_calls, st->octx.stage_xform_ok,
                        st->octx.stage_xform_us, st->octx.stage_xform_bytes);
            {
                const double ms = st->octx.direct_read_us / 1000.0;
                const double mb = st->octx.direct_read_bytes / 1024.0 / 1024.0;
                const double avg = st->octx.direct_read_calls ? ms / st->octx.direct_read_calls : 0.0;
                const double mbps = ms > 0 ? mb / (ms / 1000.0) : 0.0;
                std::fprintf(stderr,
                             "direct O_DIRECT read calls=%llu ok=%llu fail=%llu total=%.2f ms avg=%.3f ms MB=%.1f MB/s=%.1f\n",
                             (unsigned long long) st->octx.direct_read_calls,
                             (unsigned long long) st->octx.direct_read_ok,
                             (unsigned long long) st->octx.direct_read_fail,
                             ms, avg, mb, mbps);
                const double async_ms = st->octx.async_load_direct_read_us / 1000.0;
                const double async_mb = st->octx.async_load_direct_read_bytes / 1024.0 / 1024.0;
                const double fg_ms = st->octx.foreground_direct_read_us / 1000.0;
                const double fg_mb = st->octx.foreground_direct_read_bytes / 1024.0 / 1024.0;
                std::fprintf(stderr,
                             "direct read split: async_load calls=%llu total=%.2f ms MB=%.1f; foreground calls=%llu total=%.2f ms MB=%.1f\n",
                             (unsigned long long) st->octx.async_load_direct_read_calls,
                             async_ms, async_mb,
                             (unsigned long long) st->octx.foreground_direct_read_calls,
                             fg_ms, fg_mb);
            }
            {
                const double create_ms = st->octx.parent_create_us / 1000.0;
                const double create_mb = st->octx.parent_create_bytes / 1024.0 / 1024.0;
                std::fprintf(stderr,
                             "pool stats: soa_hit=%llu soa_miss=%llu parent_hit=%llu parent_miss=%llu parent_create=%llu total=%.2f ms MB=%.1f\n",
                             (unsigned long long) st->octx.soa_pool_hit,
                             (unsigned long long) st->octx.soa_pool_miss,
                             (unsigned long long) st->octx.parent_pool_hit,
                             (unsigned long long) st->octx.parent_pool_miss,
                             (unsigned long long) st->octx.parent_create_calls,
                             create_ms, create_mb);
            }
            if (ggml_opencl_moe_expert_cache_enabled()) {
                const uint64_t accesses = st->moe_cache_hits + st->moe_cache_misses;
                const double hit_rate = accesses ? 100.0 * st->moe_cache_hits / accesses : 0.0;
                const double io_ms = st->moe_cache_disk_us / 1000.0;
                const double io_mb = st->moe_cache_disk_bytes / 1024.0 / 1024.0;
                std::fprintf(stderr,
                             "moe expert cache: tensors=%zu bytes=%.1fMB hits=%llu misses=%llu hit_rate=%.2f%% disk=%.1fMB/%.2fms\n",
                             st->moe_q4_cache.size(), st->moe_cache_bytes / 1024.0 / 1024.0,
                             (unsigned long long) st->moe_cache_hits,
                             (unsigned long long) st->moe_cache_misses,
                             hit_rate, io_mb, io_ms);
                std::fprintf(stderr,
                             "moe cache stages: prepare=%llu/%.2fms router=%llu/%.2fms cpu_xform=%.2fms upload=%.2fms route_upload=%.2fms\n",
                             (unsigned long long) st->moe_cache_prepare_calls,
                             st->moe_cache_prepare_us / 1000.0,
                             (unsigned long long) st->moe_cache_router_read_calls,
                             st->moe_cache_router_read_us / 1000.0,
                             st->moe_cache_cpu_xform_us / 1000.0,
                             st->moe_cache_upload_us / 1000.0,
                             st->moe_cache_route_upload_us / 1000.0);
                std::fprintf(stderr, "moe cache resize: syncs=%llu preserved_grow=%llu copied=%.1fMB\n",
                             (unsigned long long) st->moe_cache_resize_syncs,
                             (unsigned long long) st->moe_cache_grow_resizes,
                             st->moe_cache_grow_copy_bytes / 1024.0 / 1024.0);
                std::fprintf(stderr,
                             "moe packed bypass: stage_skips=%llu parent_evictions=%llu evicted=%.1fMB\n",
                             (unsigned long long) st->moe_packed_stage_skips,
                             (unsigned long long) st->moe_packed_parent_evictions,
                             st->moe_packed_parent_evicted_bytes / 1024.0 / 1024.0);
                std::fprintf(stderr,
                             "moe prefetch: issued=%llu completed=%llu used=%llu skipped_hit=%llu skipped_cap=%llu wait=%.2fms pending=%zu buffers=%zu alloc=%llu reuse=%llu\n",
                             (unsigned long long) st->moe_prefetch_issued,
                             (unsigned long long) st->moe_prefetch_completed,
                             (unsigned long long) st->moe_prefetch_used,
                             (unsigned long long) st->moe_prefetch_skipped_hit,
                             (unsigned long long) st->moe_prefetch_skipped_cap,
                             st->moe_prefetch_wait_us / 1000.0,
                             st->moe_prefetch_tasks.size(),
                             st->moe_prefetch_free_buffers.size(),
                             (unsigned long long) st->moe_prefetch_buffer_allocs,
                             (unsigned long long) st->moe_prefetch_buffer_reuses);
                if (st->moe_route_trace_file) {
                    std::fclose(st->moe_route_trace_file);
                    st->moe_route_trace_file = nullptr;
                }
            }
            {
                std::lock_guard<std::mutex> lk(st->async_xform_mtx);
                const double wait_ms = st->async_xform_wait_us / 1000.0;
                const double avg_wait = st->async_xform_waits ? wait_ms / st->async_xform_waits : 0.0;
                std::fprintf(stderr,
                             "async xform worker enqueued=%llu completed=%llu inflight=%zu queue=%zu max_pending_seen=%zu waits=%llu wait_total=%.3f ms wait_avg=%.3f ms\n",
                             (unsigned long long) st->async_xform_enqueued,
                             (unsigned long long) st->async_xform_completed,
                             st->async_xform_inflight.size(),
                             st->async_xform_queue.size(),
                             st->async_xform_max_pending_seen,
                             (unsigned long long) st->async_xform_waits,
                             wait_ms, avg_wait);
            }
            elastic::wbmcl_dump_device_timing(&st->octx, stderr);
            elastic::wbmcl_dump_stage_detail(&st->octx, stderr);
            std::fprintf(stderr, "per-suffix breakdown (host issue time):\n");
            std::fprintf(stderr, "  %-22s %8s %12s %12s %10s %10s\n",
                         "suffix", "n", "total ms", "MB total", "avg ms", "MB/s");
            for (const auto &kv : st->reload_buckets) {
                const auto &b = kv.second;
                double mb = b.total_bytes / 1024.0 / 1024.0;
                double avg = b.n ? b.total_ms / b.n : 0;
                double mbps = b.total_ms > 0 ? mb / (b.total_ms / 1000.0) : 0;
                std::fprintf(stderr, "  %-22s %8llu %12.1f %12.1f %10.2f %10.1f\n",
                             kv.first.c_str(), (unsigned long long)b.n,
                             b.total_ms, mb, avg, mbps);
            }
            std::fprintf(stderr, "===========================================================\n");
        });
    }

    if (const char *prof = std::getenv("GGML_ELASTIC_PROFILE"); prof && *prof && *prof != '0') {
        s->profile = true;
        GGML_LOG_INFO("ggml_opencl elastic: GGML_ELASTIC_PROFILE=1 启用按 tensor / op 聚合的 IO+compute 计时（每个 op 后 clFinish，性能下降，仅调研用）\n");
        std::atexit([]() {
            auto *st = ggml_opencl_elastic();
            if (!st->profile) return;
            std::fprintf(stderr, "\n=== ggml_opencl elastic profile dump ===\n");
            std::fprintf(stderr, "reload IO （按 tensor 名后缀聚合）：\n");
            std::fprintf(stderr, "  %-22s %8s %12s %12s %10s %10s\n",
                         "suffix", "n", "total ms", "MB total", "avg ms", "MB/s");
            for (const auto &kv : st->reload_buckets) {
                const auto &b = kv.second;
                double mb = b.total_bytes / 1024.0 / 1024.0;
                double avg = b.n ? b.total_ms / b.n : 0;
                double mbps = b.total_ms > 0 ? mb / (b.total_ms / 1000.0) : 0;
                std::fprintf(stderr, "  %-22s %8llu %12.1f %12.1f %10.2f %10.1f\n",
                             kv.first.c_str(), (unsigned long long)b.n,
                             b.total_ms, mb, avg, mbps);
            }
            std::fprintf(stderr, "compute （按 ggml_op 聚合，含 clFinish 同步）：\n");
            std::fprintf(stderr, "  %-22s %8s %12s %10s\n", "op", "n", "total ms", "avg ms");
            for (const auto &kv : st->compute_buckets) {
                const auto &b = kv.second;
                double avg = b.n ? b.total_ms / b.n : 0;
                std::fprintf(stderr, "  %-22s %8llu %12.1f %10.3f\n",
                             ggml_op_name((ggml_op)kv.first),
                             (unsigned long long)b.n, b.total_ms, avg);
            }
            std::fprintf(stderr, "=========================================\n");
        });
    }

    // Prefetch 计数 dump（无条件）：开了 GGML_ELASTIC_PREFETCH 才有意义
    if (s->prefetch_lookahead > 0) {
        std::atexit([]() {
            auto *st = ggml_opencl_elastic();
            std::fprintf(stderr, "[elastic prefetch] lookahead=%d issued=%llu skipped=%llu reloads_total=%llu evicts_total=%llu\n",
                         st->prefetch_lookahead,
                         (unsigned long long)st->n_prefetch_issued,
                         (unsigned long long)st->n_prefetch_skipped,
                         (unsigned long long)st->n_reloads_total,
                         (unsigned long long)st->n_evicts_total);
        });
    }

    const char *json = std::getenv("GGML_ELASTIC_METRICS_JSONL");
    if (json && *json) {
        if (elastic::metrics_logger_init(&s->metrics, json) == 0) {
            s->metrics_inited = true;
            GGML_LOG_INFO("ggml_opencl elastic: metrics → %s\n", json);
        }
    }

    GGML_LOG_INFO("ggml_opencl elastic: 单例就绪 (kv=%zu MB misc+safety=%zu MB interval=%d)\n",
                  s->kv_bytes / 1024 / 1024, s->misc_overhead / 1024 / 1024, s->evict_check_interval);
}

static ggml_status ggml_backend_opencl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    // Elastic baseline (F4-minimal)：拿到 elastic 单例（如果已 lazy_init 过）。
    auto *est = ggml_opencl_elastic();
    const bool elastic_active = est->wbm_inited;
    if (elastic_active) {
        elastic::granularity_config planned_granularity =
            elastic::granularity_from_env();
        if (planned_granularity.mode ==
            elastic::granularity_mode::MULTI_FUSED) {
            // The GPU implementation currently exposes ordinary Multi as its
            // coarsest unit; CPU-only fused layout/kernel is not mislabeled.
            planned_granularity.mode =
                elastic::granularity_mode::MULTI;
        }
        if (planned_granularity.explicitly_enabled &&
            (planned_granularity.mode != est->granularity.mode ||
             planned_granularity.cut_parts !=
                 est->granularity.cut_parts ||
             planned_granularity.multi_tensors !=
                 est->granularity.multi_tensors)) {
            GGML_LOG_INFO(
                "ggml_opencl elastic: working-unit plan switch %s -> %s "
                "cut_parts=%d multi_tensors=%d\n",
                elastic::granularity_mode_name(est->granularity.mode),
                elastic::granularity_mode_name(
                    planned_granularity.mode),
                planned_granularity.cut_parts,
                planned_granularity.multi_tensors);
            est->granularity = planned_granularity;
        }
        est->current_token += 1;
        // Replace packed Q4 MoE parents with the bounded expert-slice cache as
        // one batch. Releasing them one layer at a time would overlap the full
        // packed model with newly allocated slice caches for most of the first
        // graph and recreate the transition memory spike this path avoids.
        ggml_opencl_release_all_moe_q4_packed(est);
        // Auto cap：首次 graph_compute 时 wbm 已全部注册，按非 pinned tensor
        // 的最大 size × 2 算 pool cap（pool by size 设计下足够 1-2 个 in-flight
        // 同 size cl_mem）。
        if (est->retain_cap_auto && est->octx.cache_byte_limit == 0) {
            size_t max_unpinned = 0;
            for (const auto &b : est->wbm.blocks) {
                if (b.is_pinned) continue;
                if (b.byte_size > max_unpinned) max_unpinned = b.byte_size;
            }
            if (max_unpinned > 0) {
                // pool by size：每 size class 只需 1 个 cl_mem 就能 cycle，
                // cap = max unpinned tensor size 即可（多了反而 working_set 缩水）
                est->octx.cache_byte_limit = max_unpinned;
                GGML_LOG_INFO("ggml_opencl elastic: auto cap = max_unpinned = %zu MB\n",
                              est->octx.cache_byte_limit / 1024 / 1024);
            }
        }
    }

    // Elastic baseline (F4-evict)：ensure_resident 当前 node 的所有 WBM src
    // tensor。封成 lambda 以便 fused 分支也能给被融合进来的下一节点跑一遍。
    auto ensure_node_srcs_resident = [&](ggml_tensor *n) -> bool {
        if (!elastic_active || !n) return true;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor *src = n->src[j];
            if (!src || !src->extra) continue;
            const int src_wbm_idx = ggml_opencl_get_wbm_idx(src);
            if (src_wbm_idx < 0) continue;
            // Plan-driven staged execution: when the real decode graph reaches
            // a weight/op anchor, ask llama_context to fire the timeline events
            // attached to that anchor. The regular ensure_resident path below
            // remains the correctness fallback.
            llama_weight_anchor_request(src->name);
            const bool moe_slice_src =
                j == 0 && n->op == GGML_OP_MUL_MAT_ID && n->src[2] && n->src[2]->ne[1] == 1 &&
                ggml_opencl_moe_expert_cache_enabled() && ggml_opencl_is_moe_expert_weight(src);
            if (moe_slice_src) {
                // MUL_MAT_ID materializes only the router-selected expert slices.
                // Drop the packed parent so it cannot consume the same budget.
                const elastic::block_meta * packed = elastic::wbm_get(&est->wbm, src_wbm_idx);
                if (packed && packed->resident) {
                    const int released = elastic::wbmcl_evict_batch(&est->octx, &src_wbm_idx, 1);
                    if (released > 0) {
                        llama_weight_runtime_mark_evicted(src->name, LLAMA_WEIGHT_RUNTIME_GPU);
                        est->n_evicts_total += released;
                    }
                }
                continue;
            }
            // SOA 量化 tensor (Q4_0 / Q8_0 / MXFP4) 走自定义 evict/reload 回调，
            // 不需要也不能写 src_extra->data_device（它的字段不在同一个偏移）。
            const bool is_soa = (src->type == GGML_TYPE_Q4_0 ||
                                 src->type == GGML_TYPE_Q8_0 ||
                                 src->type == GGML_TYPE_MXFP4);
            ggml_tensor_extra_cl *src_extra_generic =
                is_soa ? nullptr : (ggml_tensor_extra_cl *) src->extra;
            const int src_ctx_slot = ggml_opencl_get_ctx_slot(src);

            const elastic::block_meta *bm =
                elastic::wbm_get(&est->wbm, src_wbm_idx);
            if (bm && !bm->resident) {
                opencl_wait_async_xform(est, src_wbm_idx);
                bm = elastic::wbm_get(&est->wbm, src_wbm_idx);
            }
            if (bm && !bm->resident) {
                // 预先 evict：保证 reload 之后 resident_bytes <= target，避免
                // "reload 一个 → resident 涨 → 下次周期 evict 之前违反预算" 的
                // 短暂超额。spec §5 第 7 条要求每个 step 都满足合规。
                // GGML_ELASTIC_NO_AUTO_EVICT=1: 关掉 backend 自己的预算驱逐, 让上层
                // (plan framework) 做唯一 residency 权威。backend 仍 reload-on-use
                // (ensure_resident), 只是不再按 budget 自动 evict —— 避免 plan 与 backend
                // 两个驱逐者打架 (plan 把某 weight 留 GPU, backend 又按 trace min 把它 evict
                // → GPU kernel 用到已释放 cl_mem 崩)。
                static const bool s_no_auto_evict_pre = []() {
                    const char *e = std::getenv("GGML_ELASTIC_NO_AUTO_EVICT");
                    return e && *e && *e != '0';
                }();
                if (est->bw_inited && !s_no_auto_evict_pre) {
                    static const bool s_pool_counts_budget_pre = []() {
                        const char *e = std::getenv("GGML_ELASTIC_CL_RETAIN_COUNTS_BUDGET");
                        return e && *e && *e != '0';
                    }();
                    size_t target = est->dynamic_target
                    ? (([&]{
                          const size_t bt    = elastic::budget_watcher_get(&est->bw) * size_t(1024 * 1024);
                          const size_t km    = est->kv_bytes + est->misc_overhead;
                          const size_t base  = bt > km ? bt - km : 0;
                          return base + est->extra_target_bytes;
                    })())
                    : est->static_target_bytes;
                    if (ggml_opencl_moe_expert_cache_enabled()) {
                        target = target > est->moe_cache_bytes ? target - est->moe_cache_bytes : 0;
                    }
                    if (s_pool_counts_budget_pre && est->octx.cache_byte_limit > 0
                        && target > est->octx.cache_byte_limit) {
                        target -= est->octx.cache_byte_limit;
                    }
                    const size_t needed  = est->wbm.resident_bytes + bm->byte_size;
                    if (needed > target) {
                        const size_t pre_target = target > bm->byte_size ?
                                                  target - bm->byte_size : 0;
                        std::vector<int> victims;
                        int n = elastic::wbm_evict_to_byte_budget(&est->wbm, pre_target,
                                                                  /*exclude=*/src_wbm_idx, &victims);
                        if (n > 0) {
                            int released = elastic::wbmcl_evict_batch(&est->octx, victims.data(), n);
                            for (int i = 0; i < released && i < n; ++i) {
                                const std::string name = opencl_name_for_idx(est, victims[i]);
                                if (!name.empty()) {
                                    llama_weight_runtime_mark_evicted(name.c_str(), LLAMA_WEIGHT_RUNTIME_GPU);
                                }
                            }
                            est->n_evicts_total += released;
                        }
                    }
                }
                const bool csv_profile = est->profile_csv;
                auto rl_t0 = (est->profile || est->timing || csv_profile) ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
                int rc = 0;
                if (is_soa) {
                    auto cit = est->octx.soa_per_idx.find(src_wbm_idx);
                    if (cit == est->octx.soa_per_idx.end() || !cit->second.reload_fn) {
                        GGML_LOG_ERROR("ggml_opencl elastic: missing SOA reload_fn idx=%d tensor=%s\n",
                                       src_wbm_idx, src->name);
                        return false;
                    }
                    rc = elastic::wbmcl_wait_soa_reload(&est->octx, src_wbm_idx);
                    bm = elastic::wbm_get(&est->wbm, src_wbm_idx);
                    if (rc == 0 && bm && bm->resident) {
                        rc = 0;
                    } else {
                        rc = elastic::wbmcl_reload_soa_sync(&est->octx, src_wbm_idx);
                    }
                } else {
                    rc = elastic::wbmcl_ensure_resident(&est->octx, src_wbm_idx);
                }
                if (rc != 0) {
                    GGML_LOG_ERROR("ggml_opencl elastic: ensure_resident 失败 idx=%d rc=%d\n",
                                   src_wbm_idx, rc);
                    return false;
                }
                {
                    const std::string name = opencl_name_for_idx(est, src_wbm_idx);
                    if (!name.empty()) {
                        llama_weight_runtime_mark_resident(name.c_str(), LLAMA_WEIGHT_RUNTIME_GPU);
                    }
                }
                if (est->profile || est->timing || csv_profile) {
                    auto rl_t1 = std::chrono::steady_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(rl_t1 - rl_t0).count();
                    if (est->profile || est->timing) {
                        auto &b = est->reload_buckets[ggml_opencl_tensor_suffix(src->name)];
                        b.n           += 1;
                        b.total_ms    += ms;
                        b.total_bytes += bm->byte_size;
                        est->t_reload_host_us += (uint64_t)(ms * 1000.0);
                    }
                    if (csv_profile) {
                        opencl_profile_stage("RELOAD_ENSURE", src->name, src_wbm_idx,
                                             bm->byte_size, ms, 1, "decode_ensure");
                    }
                }
                bm = elastic::wbm_get(&est->wbm, src_wbm_idx);
                cl_mem new_buf = bm ? static_cast<cl_mem>(bm->backend_handle) : nullptr;
                if (!is_soa && new_buf) {
                    src_extra_generic->data_device = new_buf;
                    ggml_opencl_elastic_update_ctx_slot(src->buffer, src_ctx_slot, new_buf);
                }
                est->n_reloads_total += 1;
                est->bytes_reloaded_total += bm->byte_size;
            }
            // A staged TRANSFER may materialize ordinary OpenCL tensors before
            // the foreground ensure path runs.  Keep the tensor extra/ctx slot in
            // sync even when WBM already marks the block resident.
            bm = elastic::wbm_get(&est->wbm, src_wbm_idx);
            if (!is_soa && bm && bm->resident && bm->backend_handle) {
                cl_mem resident_buf = static_cast<cl_mem>(bm->backend_handle);
                if (src_extra_generic->data_device != resident_buf) {
                    src_extra_generic->data_device = resident_buf;
                    ggml_opencl_elastic_update_ctx_slot(src->buffer, src_ctx_slot, resident_buf);
                }
            }
            elastic::wbm_touch(&est->wbm, src_wbm_idx, est->current_token);
        }
        return true;
    };

    const bool granularity_active = elastic_active &&
        est->granularity.explicitly_enabled;
    auto is_weight_matmul = [&](const ggml_tensor *n) -> bool {
        return n && n->op == GGML_OP_MUL_MAT && n->src[0] &&
               ggml_opencl_get_wbm_idx(n->src[0]) >= 0;
    };
    auto weight_bytes = [&](const ggml_tensor *n) -> size_t {
        if (!is_weight_matmul(n)) return 0;
        auto cut = est->q4_cut_parts.find(n->src[0]);
        if (cut != est->q4_cut_parts.end()) {
            size_t bytes = 0;
            for (const auto & part : cut->second) {
                bytes += part.byte_size;
            }
            return bytes;
        }
        const elastic::block_meta *bm = elastic::wbm_get(
            &est->wbm, ggml_opencl_get_wbm_idx(n->src[0]));
        return bm ? bm->byte_size : 0;
    };
    // Plan budgets are expressed in whole MiB while Q4 blocks and aligned
    // OpenCL sub-buffers are discrete.  Permit only the sub-MiB rounding tail;
    // this is deliberately far smaller than even one ordinary Tensor unit.
    constexpr size_t weight_budget_alignment_slack = 64 * 1024;
    auto force_evict_to_target = [&](size_t target) -> bool {
        if (!est->bw_inited || est->wbm.resident_bytes <= target) return true;
        // Preserve the logical plan whenever ordinary streaming blocks can
        // satisfy the physical target. Only a second, explicitly audited pass
        // may bypass the plan-aware victim selector. Pin and atomic
        // eviction-group constraints remain authoritative in both passes.
        int total_victims = 0;
        auto evict_pass = [&](bool relax_plan_protection) -> int {
            std::vector<int> victims;
            auto * saved_victim_fn = est->wbm.victim_fn;
            void * saved_victim_ud = est->wbm.victim_ud;
            if (relax_plan_protection) {
                est->wbm.victim_fn = nullptr;
                est->wbm.victim_ud = nullptr;
            }
            const int count = elastic::wbm_evict_to_byte_budget(
                &est->wbm, target, -1, &victims);
            est->wbm.victim_fn = saved_victim_fn;
            est->wbm.victim_ud = saved_victim_ud;
            if (count <= 0) return 0;
            total_victims += count;
            const int released = elastic::wbmcl_evict_batch(
                &est->octx, victims.data(), count);
            size_t relaxed_plan_bytes = 0;
            for (int v = 0; v < released && v < count; ++v) {
                const std::string name =
                    opencl_name_for_idx(est, victims[v]);
                if (relax_plan_protection && !name.empty() &&
                    llama_weight_runtime_desired_query(name.c_str()) ==
                        LLAMA_WEIGHT_RUNTIME_GPU) {
                    const elastic::block_meta * block =
                        elastic::wbm_get(&est->wbm, victims[v]);
                    if (block) {
                        relaxed_plan_bytes += block->byte_size;
                    }
                }
                if (!name.empty()) {
                    llama_weight_runtime_mark_evicted(
                        name.c_str(), LLAMA_WEIGHT_RUNTIME_GPU);
                }
            }
            if (relaxed_plan_bytes > 0) {
                est->unit_pipeline_plan_protection_relaxations++;
                est->unit_pipeline_plan_protection_relaxed_bytes +=
                    relaxed_plan_bytes;
            }
            est->n_evicts_total += released > 0 ? released : 0;
            return released;
        };

        // One plan-aware pass can select every movable streaming victim. If it
        // is insufficient, the hard budget wins, but the fallback is visible
        // to the experiment audit instead of silently changing residency.
        evict_pass(false);
        for (int attempt = 0;
             attempt < 4 && est->wbm.resident_bytes > target;
             ++attempt) {
            if (evict_pass(true) <= 0) break;
        }
        if (est->wbm.resident_bytes > target) {
            size_t pinned_bytes = 0;
            for (const auto & block : est->wbm.blocks) {
                if (block.resident && block.is_pinned) {
                    pinned_bytes += block.byte_size;
                }
            }
            std::fprintf(
                stderr,
                "[elastic mandatory reserve failed] target_mib=%.2f "
                "resident_mib=%.2f pinned_mib=%.2f victims=%d\n",
                target / 1024.0 / 1024.0,
                est->wbm.resident_bytes / 1024.0 / 1024.0,
                pinned_bytes / 1024.0 / 1024.0,
                total_victims);
            return false;
        }
        return true;
    };
    auto evict_to_current_budget = [&]() {
        if (!est->bw_inited) return;
        static const bool no_auto_evict = []() {
            const char *e = std::getenv("GGML_ELASTIC_NO_AUTO_EVICT");
            return e && *e && *e != '0';
        }();
        if (no_auto_evict) return;
        size_t target = est->dynamic_target
            ? (([&] {
                  const size_t bt = elastic::budget_watcher_get(&est->bw) * size_t(1024 * 1024);
                  const size_t km = est->kv_bytes + est->misc_overhead;
                  return (bt > km ? bt - km : 0) + est->extra_target_bytes;
              })())
            : est->static_target_bytes;
        if (ggml_opencl_moe_expert_cache_enabled()) {
            target = target > est->moe_cache_bytes ? target - est->moe_cache_bytes : 0;
        }
        static const bool pool_counts_budget = []() {
            const char *e = std::getenv("GGML_ELASTIC_CL_RETAIN_COUNTS_BUDGET");
            return e && *e && *e != '0';
        }();
        if (pool_counts_budget && est->octx.cache_byte_limit > 0 &&
            target > est->octx.cache_byte_limit) {
            target -= est->octx.cache_byte_limit;
        }
        force_evict_to_target(target);
    };
    auto sample_pipeline_budget = [&]() {
        if (!est->bw_inited) return;
        size_t target = est->dynamic_target
            ? (([&] {
                  const size_t bt =
                      elastic::budget_watcher_get(&est->bw) *
                      size_t(1024 * 1024);
                  const size_t km = est->kv_bytes + est->misc_overhead;
                  return (bt > km ? bt - km : 0) +
                      est->extra_target_bytes;
              })())
            : est->static_target_bytes;
        if (ggml_opencl_moe_expert_cache_enabled()) {
            target =
                target > est->moe_cache_bytes
                    ? target - est->moe_cache_bytes
                    : 0;
        }
        const size_t aligned_target =
            target <= SIZE_MAX - weight_budget_alignment_slack
                ? target + weight_budget_alignment_slack
                : SIZE_MAX;
        // BudgetWatcher may cross a bucket while layout or a kernel is being
        // queued.  Enforce against the same freshly sampled target immediately
        // before recording the invariant, including plan-driven NO_AUTO mode.
        if (aligned_target != SIZE_MAX) {
            force_evict_to_target(aligned_target);
        }
        size_t pinned_resident_bytes = 0;
        for (const auto & block : est->wbm.blocks) {
            if (block.resident && block.is_pinned) {
                pinned_resident_bytes += block.byte_size;
            }
        }
        est->unit_pipeline_budget_samples++;
        est->unit_pipeline_resident_bytes_peak = std::max(
            est->unit_pipeline_resident_bytes_peak,
            est->wbm.resident_bytes);
        est->unit_pipeline_pinned_bytes_peak = std::max(
            est->unit_pipeline_pinned_bytes_peak,
            pinned_resident_bytes);
        if (est->wbm.resident_bytes > aligned_target) {
            est->unit_pipeline_budget_violations++;
            est->unit_pipeline_over_budget_bytes_peak = std::max(
                est->unit_pipeline_over_budget_bytes_peak,
                est->wbm.resident_bytes - target);
            if (est->unit_pipeline_budget_violations <= 4) {
                std::fprintf(
                    stderr,
                    "[elastic unit pipeline budget debug] violation=%llu "
                    "target_mib=%.2f resident_mib=%.2f pinned_mib=%.2f\n",
                    (unsigned long long)
                        est->unit_pipeline_budget_violations,
                    target / 1024.0 / 1024.0,
                    est->wbm.resident_bytes / 1024.0 / 1024.0,
                    pinned_resident_bytes / 1024.0 / 1024.0);
                for (const auto & block : est->wbm.blocks) {
                    if (!block.resident || !block.is_pinned) continue;
                    const std::string name =
                        opencl_name_for_idx(est, block.block_idx);
                    std::fprintf(
                        stderr,
                        "[elastic unit pipeline budget debug] pinned "
                        "idx=%d mib=%.2f name=%s\n",
                        block.block_idx,
                        block.byte_size / 1024.0 / 1024.0,
                        name.empty() ? "(unknown)" : name.c_str());
                }
            }
        }
    };

    // Build the same real working units used by execution, then pipeline the
    // next unit's LOAD/PREPARE while the current unit computes.  This is kept
    // separate from the legacy node-distance prefetch: one lookahead means one
    // Multi, Tensor, or Cut unit rather than one graph node.
    static const bool unit_pipeline_enabled = []() {
        const char * e = std::getenv("GGML_ELASTIC_UNIT_PIPELINE");
        return e && *e && *e != '0';
    }();
    static const int unit_pipeline_lookahead = []() {
        const char * e = std::getenv("GGML_ELASTIC_UNIT_PIPELINE_LOOKAHEAD");
        return e && *e ? std::max(1, std::atoi(e)) : 1;
    }();
    static const size_t unit_pipeline_lookahead_bytes = []() {
        const char * e = std::getenv("GGML_ELASTIC_UNIT_PIPELINE_LOOKAHEAD_MB");
        const long long mib = e && *e ? std::atoll(e) : 0;
        return mib > 0 ? static_cast<size_t>(mib) * 1024ULL * 1024ULL : 0;
    }();
    static const size_t unit_pipeline_graph_lookahead = []() {
        const char * e = std::getenv("GGML_ELASTIC_UNIT_PIPELINE_GRAPH_LOOKAHEAD");
            const long long graphs = e && *e ? std::atoll(e) : 1;
        // Zero retains the current-graph unit pipeline while disabling
        // speculative successor-graph staging.  Fine-grained Cut units can
        // use this to avoid cross-graph budget churn.
        return static_cast<size_t>(std::max<long long>(0, graphs));
    }();
    struct opencl_mixed_weight_binding {
        bool planned = false;
        bool fine_cut = false;
        std::vector<int> part_unit_ids;
        uint32_t flags = 0;
    };
    std::unordered_map<const ggml_tensor *, opencl_mixed_weight_binding>
        mixed_weight_bindings;
    auto mixed_binding_for_weight =
        [&](const ggml_tensor * weight)
            -> const opencl_mixed_weight_binding & {
        auto cached = mixed_weight_bindings.find(weight);
        if (cached != mixed_weight_bindings.end()) {
            return cached->second;
        }
        opencl_mixed_weight_binding binding;
        if (!weight || !weight->name[0]) {
            return mixed_weight_bindings.emplace(
                weight, std::move(binding)).first->second;
        }
        const int count =
            llama_weight_unit_plan_query(weight->name, nullptr, 0);
        std::vector<llama_weight_unit_slice> slices;
        if (count > 0) {
            slices.resize(static_cast<size_t>(count));
            const int copied = llama_weight_unit_plan_query(
                weight->name, slices.data(), count);
            if (copied < count) {
                slices.resize(static_cast<size_t>(
                    std::max(0, copied)));
            }
        }
        auto cuts = est->q4_cut_parts.find(weight);
        if (!slices.empty() && cuts != est->q4_cut_parts.end()) {
            for (const auto & part : cuts->second) {
                const llama_weight_unit_slice * selected = nullptr;
                for (const auto & slice : slices) {
                    if (slice.row_start == part.row_start &&
                        slice.row_count == part.row_count) {
                        selected = &slice;
                        break;
                    }
                }
                if (!selected) {
                    for (const auto & slice : slices) {
                        if (slice.row_start == 0 &&
                            slice.row_count < 0) {
                            selected = &slice;
                            break;
                        }
                    }
                }
                if (!selected) {
                    binding.part_unit_ids.clear();
                    break;
                }
                binding.part_unit_ids.push_back(selected->unit_id);
                binding.flags |= selected->flags;
            }
        } else if (!slices.empty()) {
            const llama_weight_unit_slice * selected = nullptr;
            for (const auto & slice : slices) {
                if (slice.row_start == 0 &&
                    slice.row_count < 0) {
                    selected = &slice;
                    break;
                }
            }
            if (!selected && slices.size() == 1) {
                selected = &slices.front();
            }
            if (selected) {
                binding.part_unit_ids.push_back(selected->unit_id);
                binding.flags = selected->flags;
            }
        }
        binding.planned = !binding.part_unit_ids.empty();
        if (binding.planned && binding.part_unit_ids.size() > 1) {
            binding.fine_cut = std::any_of(
                binding.part_unit_ids.begin() + 1,
                binding.part_unit_ids.end(),
                [&](int unit_id) {
                    return unit_id != binding.part_unit_ids.front();
                });
        }
        return mixed_weight_bindings.emplace(
            weight, std::move(binding)).first->second;
    };

    std::vector<std::vector<int>> pipeline_units;
    std::vector<int64_t> pipeline_group_ids;
    std::unordered_map<int, size_t> pipeline_pos_by_wbm;
    std::unordered_map<const ggml_tensor *, size_t>
        mixed_pipeline_pos_by_weight;
    std::vector<std::vector<ggml_tensor *>>
        mixed_nodes_by_pipeline_pos;
    if (granularity_active) {
        std::vector<int> building;
        std::unordered_set<int> building_seen;
        auto append_pipeline_unit = [&](
                std::vector<int> indices,
                int64_t planned_group_id = -1) {
            std::vector<int> unique;
            std::unordered_set<int> seen;
            for (int idx : indices) {
                if (idx >= 0 && seen.insert(idx).second) unique.push_back(idx);
            }
            if (unique.empty()) return;
            const size_t pos = pipeline_units.size();
            const int fallback_idx = *std::min_element(
                unique.begin(), unique.end());
            pipeline_units.push_back(std::move(unique));
            pipeline_group_ids.push_back(
                planned_group_id >= 0
                    ? planned_group_id
                    : ((INT64_C(1) << 50) +
                       static_cast<int64_t>(
                           std::max(0, fallback_idx))));
            for (int idx : pipeline_units.back()) pipeline_pos_by_wbm[idx] = pos;
        };
        auto flush_multi = [&]() {
            if (!building.empty()) append_pipeline_unit(std::move(building));
            building.clear();
            building_seen.clear();
        };

        std::unordered_map<int, size_t> mixed_pos_by_unit_id;
        auto append_mixed_index = [&](int unit_id, int idx) {
            if (idx < 0 || unit_id < 0) return SIZE_MAX;
            auto found = mixed_pos_by_unit_id.find(unit_id);
            if (found == mixed_pos_by_unit_id.end()) {
                const size_t pos = pipeline_units.size();
                pipeline_units.push_back({});
                pipeline_group_ids.push_back(
                    static_cast<int64_t>(unit_id));
                mixed_nodes_by_pipeline_pos.resize(
                    pipeline_units.size());
                mixed_pos_by_unit_id[unit_id] = pos;
                found = mixed_pos_by_unit_id.find(unit_id);
            }
            const size_t pos = found->second;
            auto & indices = pipeline_units[pos];
            if (std::find(indices.begin(), indices.end(), idx) ==
                indices.end()) {
                indices.push_back(idx);
            }
            pipeline_pos_by_wbm[idx] = pos;
            return pos;
        };

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_tensor * node = cgraph->nodes[i];
            if (!node || node->op != GGML_OP_MUL_MAT || !node->src[0]) continue;
            auto cut = est->q4_cut_parts.find(node->src[0]);
            const auto & mixed =
                mixed_binding_for_weight(node->src[0]);
            if (mixed.planned) {
                flush_multi();
                size_t coarse_pos = SIZE_MAX;
                if (cut != est->q4_cut_parts.end()) {
                    for (size_t part = 0;
                         part < cut->second.size() &&
                         part < mixed.part_unit_ids.size(); ++part) {
                        const size_t pos = append_mixed_index(
                            mixed.part_unit_ids[part],
                            cut->second[part].wbm_idx);
                        if (!mixed.fine_cut) coarse_pos = pos;
                    }
                } else {
                    coarse_pos = append_mixed_index(
                        mixed.part_unit_ids.front(),
                        ggml_opencl_get_wbm_idx(node->src[0]));
                }
                if (!mixed.fine_cut && coarse_pos != SIZE_MAX) {
                    mixed_pipeline_pos_by_weight[node->src[0]] =
                        coarse_pos;
                    auto & nodes =
                        mixed_nodes_by_pipeline_pos[coarse_pos];
                    if (std::find(nodes.begin(), nodes.end(), node) ==
                        nodes.end()) {
                        nodes.push_back(node);
                    }
                }
                continue;
            }
            if (!unit_pipeline_enabled) continue;
            if (cut != est->q4_cut_parts.end()) {
                if (est->granularity.mode ==
                    elastic::granularity_mode::CUT) {
                    for (const auto & part : cut->second) {
                        append_pipeline_unit({part.wbm_idx});
                    }
                    continue;
                }
            }
            std::vector<int> logical_indices;
            if (cut != est->q4_cut_parts.end()) {
                for (const auto & part : cut->second) {
                    logical_indices.push_back(part.wbm_idx);
                }
            } else {
                const int idx =
                    ggml_opencl_get_wbm_idx(node->src[0]);
                if (idx >= 0) logical_indices.push_back(idx);
            }
            if (logical_indices.empty()) continue;
            if (est->granularity.mode == elastic::granularity_mode::MULTI ||
                est->granularity.mode == elastic::granularity_mode::MULTI_FUSED) {
                const int logical_key = logical_indices.front();
                if (building_seen.insert(logical_key).second) {
                    building.insert(
                        building.end(), logical_indices.begin(),
                        logical_indices.end());
                }
                if ((int) building_seen.size() >=
                    est->granularity.multi_tensors) {
                    flush_multi();
                }
            } else {
                append_pipeline_unit(std::move(logical_indices));
            }
        }
        flush_multi();
        if (mixed_nodes_by_pipeline_pos.size() <
            pipeline_units.size()) {
            mixed_nodes_by_pipeline_pos.resize(
                pipeline_units.size());
        }

        // Physical row tiles are provisioned once so an online plan can
        // split/merge without reallocating OpenCL objects.  Make residency
        // follow the current logical units: WBM evicts every non-pinned tile
        // in a Tensor/Multi unit atomically, whereas Cut's one-tile units can
        // be evicted independently.
        const uint64_t grouping_generation =
            llama_weight_unit_plan_generation();
        if (est->eviction_group_generation != grouping_generation) {
            elastic::wbm_clear_eviction_groups(&est->wbm);
            est->eviction_group_generation = grouping_generation;
        }
        for (size_t unit = 0; unit < pipeline_units.size(); ++unit) {
            const int64_t group_id =
                unit < pipeline_group_ids.size()
                    ? pipeline_group_ids[unit]
                    : ((INT64_C(1) << 50) +
                       static_cast<int64_t>(unit));
            for (int idx : pipeline_units[unit]) {
                const elastic::block_meta * block =
                    elastic::wbm_get(&est->wbm, idx);
                if (block && !block->is_pinned) {
                    elastic::wbm_set_eviction_group(
                        &est->wbm, idx, group_id);
                }
            }
        }
    }

    std::vector<bool> pipeline_issued(pipeline_units.size(), false);
    std::vector<bool> pipeline_begun(pipeline_units.size(), false);
    uint64_t pipeline_graph_signature = 1469598103934665603ULL;
    auto pipeline_signature_mix = [&](uint64_t value) {
        pipeline_graph_signature ^= value;
        pipeline_graph_signature *= 1099511628211ULL;
    };
    pipeline_signature_mix(static_cast<uint64_t>(cgraph->n_nodes));
    pipeline_signature_mix(static_cast<uint64_t>(pipeline_units.size()));
    for (const auto & unit : pipeline_units) {
        pipeline_signature_mix(static_cast<uint64_t>(unit.size()));
        for (int idx : unit) {
            pipeline_signature_mix(static_cast<uint64_t>(idx + 1));
        }
    }
    if (granularity_active && unit_pipeline_enabled && !pipeline_units.empty()) {
        if (est->unit_pipeline_previous_graph_valid) {
            opencl_pipeline_successor successor;
            successor.signature = pipeline_graph_signature;
            successor.units = pipeline_units;
            est->unit_pipeline_successors[
                est->unit_pipeline_previous_graph_signature] =
                    std::move(successor);
        }
        est->unit_pipeline_previous_graph_signature =
            pipeline_graph_signature;
        est->unit_pipeline_previous_graph_valid = true;
    }

    // A future unit remains protected until its consumer has made it resident.
    // Otherwise periodic budget enforcement can evict a just-prepared unit and
    // turn pipeline work into extra reload churn.
    std::unordered_map<int, opencl_pipeline_pin_state> pipeline_pins;
    auto acquire_pipeline_pins = [&](const std::vector<int> & indices) {
        for (int idx : indices) {
            const elastic::block_meta * bm =
                elastic::wbm_get(&est->wbm, idx);
            if (!bm) continue;
            auto & pin = pipeline_pins[idx];
            if (pin.refs == 0) pin.original = bm->is_pinned;
            pin.refs++;
            elastic::wbm_set_pinned(&est->wbm, idx, true);
        }
    };
    auto release_pipeline_pins = [&](const std::vector<int> & indices) {
        for (int idx : indices) {
            auto pin = pipeline_pins.find(idx);
            if (pin == pipeline_pins.end() || pin->second.refs == 0) continue;
            if (--pin->second.refs == 0) {
                elastic::wbm_set_pinned(
                    &est->wbm, idx, pin->second.original);
                pipeline_pins.erase(pin);
            }
        }
    };

    // Adopt the exact prefix prepared by the preceding backend graph.
    bool incoming_cross_match =
        !pipeline_units.empty() &&
        !est->unit_pipeline_cross_graphs.empty() &&
        est->unit_pipeline_cross_graphs.front().signature ==
            pipeline_graph_signature &&
        est->unit_pipeline_cross_graphs.front().units.size() <=
            pipeline_units.size();
    if (incoming_cross_match) {
        const auto & incoming = est->unit_pipeline_cross_graphs.front();
        for (size_t i = 0; i < incoming.units.size(); ++i) {
            if (incoming.units[i] != pipeline_units[i]) {
                incoming_cross_match = false;
                break;
            }
        }
    }
    if (incoming_cross_match) {
        opencl_pipeline_successor incoming =
            std::move(est->unit_pipeline_cross_graphs.front());
        est->unit_pipeline_cross_graphs.pop_front();
        std::unordered_map<int, size_t> transferred_refs;
        for (const auto & unit : incoming.units) {
            for (int idx : unit) transferred_refs[idx]++;
        }
        std::unordered_map<int, bool> local_original;
        for (const auto & transfer : transferred_refs) {
            auto cross =
                est->unit_pipeline_cross_pins.find(transfer.first);
            if (cross == est->unit_pipeline_cross_pins.end()) {
                local_original[transfer.first] = false;
                continue;
            }
            const size_t moved =
                std::min(cross->second.refs, transfer.second);
            cross->second.refs -= moved;
            if (cross->second.refs > 0) {
                // A later prefetched graph still owns the WBM pin.
                local_original[transfer.first] = true;
            } else {
                local_original[transfer.first] = cross->second.original;
                est->unit_pipeline_cross_pins.erase(cross);
            }
        }
        bool all_ready = true;
        for (size_t i = 0; i < incoming.units.size(); ++i) {
            pipeline_issued[i] = true;
            for (int idx : pipeline_units[i]) {
                const elastic::block_meta * bm =
                    elastic::wbm_get(&est->wbm, idx);
                if (bm && !bm->resident) all_ready = false;
                auto & pin = pipeline_pins[idx];
                if (pin.refs == 0) {
                    auto original = local_original.find(idx);
                    pin.original =
                        original != local_original.end()
                            ? original->second
                            : false;
                }
                pin.refs++;
                elastic::wbm_set_pinned(&est->wbm, idx, true);
            }
        }
        if (all_ready) est->unit_pipeline_cross_ready++;
        else est->unit_pipeline_cross_waits++;
    } else if (!est->unit_pipeline_cross_graphs.empty() ||
               !est->unit_pipeline_cross_pins.empty()) {
        for (const auto & pin : est->unit_pipeline_cross_pins) {
            elastic::wbm_set_pinned(
                &est->wbm, pin.first, pin.second.original);
        }
        est->unit_pipeline_cross_graphs.clear();
        est->unit_pipeline_cross_pins.clear();
    }

    auto pipeline_now_us = []() -> uint64_t {
        return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    struct pipeline_observation {
        size_t pos = 0;
        bool valid = false;
        bool missing_before = false;
        uint64_t t0_us = 0;
    };
    auto pipeline_begin = [&](int wbm_idx) {
        pipeline_observation obs;
        auto found = pipeline_pos_by_wbm.find(wbm_idx);
        if (found == pipeline_pos_by_wbm.end()) return obs;
        obs.pos = found->second;
        if (pipeline_begun[obs.pos]) return obs;
        pipeline_begun[obs.pos] = true;
        obs.valid = true;
        for (int idx : pipeline_units[obs.pos]) {
            const elastic::block_meta * bm =
                elastic::wbm_get(&est->wbm, idx);
            if (bm && !bm->resident) {
                obs.missing_before = true;
                break;
            }
        }
        if (obs.missing_before) {
            est->unit_pipeline_current_missing_units++;
        }
        if (pipeline_issued[obs.pos]) {
            obs.t0_us = pipeline_now_us();
        }
        return obs;
    };
    auto pipeline_finish_observation = [&](const pipeline_observation & obs) {
        if (!obs.valid || !pipeline_issued[obs.pos]) return;
        if (obs.missing_before) {
            est->unit_pipeline_waits++;
            est->unit_pipeline_wait_us += pipeline_now_us() - obs.t0_us;
        } else {
            est->unit_pipeline_ready++;
        }
    };
    std::deque<opencl_pipeline_successor> pending_cross_graphs;
    std::unordered_map<int, size_t> pending_cross_pin_refs;
    auto stage_pipeline_after = [&](size_t current_pos,
                                    const std::vector<int> & current_indices) {
        if (current_pos >= pipeline_units.size()) return;
        std::vector<int> future_indices;
        std::vector<size_t> selected_unit_positions;
        std::unordered_set<int> future_seen;
        size_t missing_bytes = 0;
        size_t window_bytes = 0;
        size_t selected_units = 0;
        const size_t future_end = unit_pipeline_lookahead_bytes > 0
            ? pipeline_units.size()
            : std::min(pipeline_units.size(),
                       current_pos + 1 + (size_t) unit_pipeline_lookahead);
        for (size_t future = current_pos + 1; future < future_end; ++future) {
            size_t added_window_bytes = 0;
            for (int idx : pipeline_units[future]) {
                if (future_seen.find(idx) != future_seen.end()) continue;
                const elastic::block_meta * bm = elastic::wbm_get(&est->wbm, idx);
                // The window bounds protected working-set bytes, not only
                // incremental allocation.  Counting an already-resident
                // future tensor as zero while pinning it can protect far more
                // than the advertised cap and block budget enforcement.
                if (bm) added_window_bytes += bm->byte_size;
            }
            if (unit_pipeline_lookahead_bytes > 0 &&
                window_bytes + added_window_bytes > unit_pipeline_lookahead_bytes) {
                if (selected_units == 0) est->unit_pipeline_oversize_windows++;
                break;
            }
            if (!pipeline_issued[future]) {
                pipeline_issued[future] = true;
                est->unit_pipeline_issued++;
                acquire_pipeline_pins(pipeline_units[future]);
            }
            selected_units++;
            selected_unit_positions.push_back(future);
            for (int idx : pipeline_units[future]) {
                if (!future_seen.insert(idx).second) continue;
                future_indices.push_back(idx);
                const elastic::block_meta * bm = elastic::wbm_get(&est->wbm, idx);
                if (bm) {
                    window_bytes += bm->byte_size;
                    if (!bm->resident) {
                        missing_bytes += bm->byte_size;
                    }
                }
            }
        }
        est->unit_pipeline_window_samples++;
        est->unit_pipeline_window_units_total += selected_units;
        est->unit_pipeline_window_units_max = std::max(
            est->unit_pipeline_window_units_max, selected_units);
        est->unit_pipeline_window_bytes_total += window_bytes;
        est->unit_pipeline_window_bytes_max = std::max(
            est->unit_pipeline_window_bytes_max, window_bytes);
        bool can_prepare = true;
        if (!future_indices.empty() && est->bw_inited && missing_bytes > 0) {
            size_t target = est->dynamic_target
                ? (([&] {
                      const size_t bt = elastic::budget_watcher_get(&est->bw) * size_t(1024 * 1024);
                      const size_t km = est->kv_bytes + est->misc_overhead;
                      return (bt > km ? bt - km : 0) + est->extra_target_bytes;
                  })())
                : est->static_target_bytes;
            if (ggml_opencl_moe_expert_cache_enabled()) {
                target = target > est->moe_cache_bytes ? target - est->moe_cache_bytes : 0;
            }
            const size_t reserve_target = target > missing_bytes ? target - missing_bytes : 0;
            std::vector<std::pair<int, bool>> pins;
            std::unordered_set<int> protected_seen;
            for (int idx : current_indices) protected_seen.insert(idx);
            for (int idx : future_indices) protected_seen.insert(idx);
            pins.reserve(protected_seen.size());
            for (int idx : protected_seen) {
                const elastic::block_meta * bm = elastic::wbm_get(&est->wbm, idx);
                if (!bm) continue;
                pins.push_back({idx, bm->is_pinned});
                elastic::wbm_set_pinned(&est->wbm, idx, true);
            }
            if (est->wbm.resident_bytes > reserve_target) {
                std::vector<int> victims;
                const int count = elastic::wbm_evict_to_byte_budget(
                    &est->wbm, reserve_target, -1, &victims);
                if (count > 0) {
                    const int released = elastic::wbmcl_evict_batch(
                        &est->octx, victims.data(), count);
                    for (int v = 0; v < released && v < count; ++v) {
                        const std::string name = opencl_name_for_idx(est, victims[v]);
                        if (!name.empty()) {
                            llama_weight_runtime_mark_evicted(
                                name.c_str(), LLAMA_WEIGHT_RUNTIME_GPU);
                        }
                    }
                    est->n_evicts_total += released > 0 ? released : 0;
                }
            }
            can_prepare = est->wbm.resident_bytes <= reserve_target;
            for (const auto & pin : pins) {
                elastic::wbm_set_pinned(&est->wbm, pin.first, pin.second);
            }
        }

        for (size_t future : selected_unit_positions) {
            const auto & unit = pipeline_units[future];
            elastic::wbmcl_load_host_async_unit(
                &est->octx, unit.data(), unit.size());
            if (can_prepare) {
                elastic::wbmcl_prepare_backend_unit(
                    &est->octx, unit.data(), unit.size());
            }
        }

        // Fill the remainder of the same byte window with a learned successor
        // graph prefix.  The first observation only learns the transition;
        // later decode iterations can overlap the next graph's first unit.
        if (!pending_cross_graphs.empty() ||
            !est->unit_pipeline_cross_graphs.empty()) {
            return;
        }
        std::unordered_set<int> cross_seen = future_seen;
        std::unordered_set<uint64_t> seen_graphs;
        size_t cross_missing_bytes = 0;
        size_t cross_window_bytes = 0;
        size_t cross_selected_units = 0;
        uint64_t cursor = pipeline_graph_signature;
        bool window_full = false;
        for (size_t hop = 0;
             hop < unit_pipeline_graph_lookahead && !window_full; ++hop) {
            if (!seen_graphs.insert(cursor).second) break;
            auto successor_it = est->unit_pipeline_successors.find(cursor);
            if (successor_it == est->unit_pipeline_successors.end()) break;
            const opencl_pipeline_successor & successor =
                successor_it->second;
            opencl_pipeline_successor staged_graph;
            staged_graph.signature = successor.signature;
            size_t graph_new_indices = 0;
            size_t graph_selected_units = 0;
            std::unordered_map<int, size_t> graph_pin_refs;
            for (const auto & unit : successor.units) {
                if (unit_pipeline_lookahead_bytes == 0 &&
                    selected_units + cross_selected_units +
                            graph_selected_units >=
                        static_cast<size_t>(unit_pipeline_lookahead)) {
                    window_full = true;
                    break;
                }
                size_t added_window_bytes = 0;
                for (int idx : unit) {
                    if (cross_seen.find(idx) != cross_seen.end()) continue;
                    const elastic::block_meta * bm =
                        elastic::wbm_get(&est->wbm, idx);
                    if (bm) added_window_bytes += bm->byte_size;
                }
                if (unit_pipeline_lookahead_bytes > 0 &&
                    window_bytes + cross_window_bytes +
                            added_window_bytes >
                        unit_pipeline_lookahead_bytes) {
                    window_full = true;
                    break;
                }
                staged_graph.units.push_back(unit);
                graph_selected_units++;
                for (int idx : unit) {
                    graph_pin_refs[idx]++;
                    if (!cross_seen.insert(idx).second) continue;
                    graph_new_indices++;
                    const elastic::block_meta * bm =
                        elastic::wbm_get(&est->wbm, idx);
                    if (bm) cross_window_bytes += bm->byte_size;
                    const bool already_local =
                        pipeline_pins.find(idx) != pipeline_pins.end();
                    if (bm && !bm->resident && !already_local) {
                        cross_missing_bytes += bm->byte_size;
                    }
                }
            }
            // Different backend graph signatures can form a short cycle while
            // referring to the same model weights.  Once one successor has
            // covered every WBM index in the byte window, later signatures
            // add zero bytes.  Previously those zero-byte graphs were still
            // appended and every unit was prepared again (hundreds of
            // redundant stages per token, plus pins for the whole model).
            // A cross-graph lookahead must advance the materialized working
            // set, not merely traverse a distinct graph signature.
            if (staged_graph.units.empty() || graph_new_indices == 0) break;
            for (const auto & pin_ref : graph_pin_refs) {
                pending_cross_pin_refs[pin_ref.first] +=
                    pin_ref.second;
            }
            cross_selected_units += graph_selected_units;
            pending_cross_graphs.push_back(std::move(staged_graph));
            if (window_full) break;
            cursor = successor.signature;
        }
        if (pending_cross_graphs.empty()) return;

        for (const auto & pin_ref : pending_cross_pin_refs) {
            const elastic::block_meta * bm =
                elastic::wbm_get(&est->wbm, pin_ref.first);
            if (!bm) continue;
            auto & pin = pipeline_pins[pin_ref.first];
            if (pin.refs == 0) pin.original = bm->is_pinned;
            pin.refs += pin_ref.second;
            elastic::wbm_set_pinned(&est->wbm, pin_ref.first, true);
        }

        bool can_prepare_cross = true;
        if (est->bw_inited && cross_missing_bytes > 0) {
            size_t target = est->dynamic_target
                ? (([&] {
                      const size_t bt =
                          elastic::budget_watcher_get(&est->bw) *
                          size_t(1024 * 1024);
                      const size_t km =
                          est->kv_bytes + est->misc_overhead;
                      return (bt > km ? bt - km : 0) +
                          est->extra_target_bytes;
                  })())
                : est->static_target_bytes;
            if (ggml_opencl_moe_expert_cache_enabled()) {
                target =
                    target > est->moe_cache_bytes
                        ? target - est->moe_cache_bytes
                        : 0;
            }
            const size_t total_missing =
                missing_bytes + cross_missing_bytes;
            const size_t reserve_target =
                target > total_missing ? target - total_missing : 0;
            std::vector<std::pair<int, bool>> temporary_pins;
            temporary_pins.reserve(current_indices.size());
            for (int idx : current_indices) {
                const elastic::block_meta * bm =
                    elastic::wbm_get(&est->wbm, idx);
                if (!bm) continue;
                temporary_pins.push_back({idx, bm->is_pinned});
                elastic::wbm_set_pinned(&est->wbm, idx, true);
            }
            if (est->wbm.resident_bytes > reserve_target) {
                std::vector<int> victims;
                const int count = elastic::wbm_evict_to_byte_budget(
                    &est->wbm, reserve_target, -1, &victims);
                if (count > 0) {
                    const int released = elastic::wbmcl_evict_batch(
                        &est->octx, victims.data(), count);
                    for (int v = 0; v < released && v < count; ++v) {
                        const std::string name =
                            opencl_name_for_idx(est, victims[v]);
                        if (!name.empty()) {
                            llama_weight_runtime_mark_evicted(
                                name.c_str(), LLAMA_WEIGHT_RUNTIME_GPU);
                        }
                    }
                    est->n_evicts_total += released > 0 ? released : 0;
                }
            }
            can_prepare_cross =
                est->wbm.resident_bytes <= reserve_target;
            for (const auto & pin : temporary_pins) {
                elastic::wbm_set_pinned(
                    &est->wbm, pin.first, pin.second);
            }
        }
        for (const auto & graph : pending_cross_graphs) {
            for (const auto & unit : graph.units) {
                elastic::wbmcl_load_host_async_unit(
                    &est->octx, unit.data(), unit.size());
                if (can_prepare_cross) {
                    elastic::wbmcl_prepare_backend_unit(
                        &est->octx, unit.data(), unit.size());
                }
                est->unit_pipeline_issued++;
                est->unit_pipeline_cross_issued++;
            }
        }
        est->unit_pipeline_cross_bytes += cross_missing_bytes;
    };

    auto complete_granularity_unit = [&]() -> bool {
        est->granularity_unit_boundaries++;
        cl_int err = CL_SUCCESS;
        switch (est->granularity_unit_sync) {
            case ggml_opencl_elastic_state::unit_sync_mode::NONE:
                return true;
            case ggml_opencl_elastic_state::unit_sync_mode::FLUSH:
                est->granularity_unit_flushes++;
                err = clFlush(backend_ctx->queue);
                break;
            case ggml_opencl_elastic_state::unit_sync_mode::FINISH:
                est->granularity_unit_finishes++;
                err = clFinish(backend_ctx->queue);
                break;
        }
        if (err != CL_SUCCESS) {
            est->granularity_unit_sync_errors++;
            GGML_LOG_ERROR(
                "ggml_opencl elastic: working-unit completion failed: %d\n",
                err);
            return false;
        }
        return true;
    };

    // 0=off, 1=two queues, 2=one fused two-cut GEMV dispatch.
    static const int cut_dual_compute_mode = []() {
        const char * e = std::getenv("GGML_ELASTIC_CUT_DUAL_COMPUTE");
        if (!e || !*e || *e == '0' || std::strcmp(e, "off") == 0) {
            return 0;
        }
        return std::strcmp(e, "fused") == 0 ? 2 : 1;
    }();
    auto current_weight_target = [&]() -> size_t {
        if (!est->bw_inited) return SIZE_MAX;
        size_t target = est->dynamic_target
            ? (([&] {
                  const size_t bt =
                      elastic::budget_watcher_get(&est->bw) *
                      size_t(1024 * 1024);
                  const size_t km = est->kv_bytes + est->misc_overhead;
                  return (bt > km ? bt - km : 0) +
                      est->extra_target_bytes;
              })())
            : est->static_target_bytes;
        if (ggml_opencl_moe_expert_cache_enabled()) {
            target =
                target > est->moe_cache_bytes
                    ? target - est->moe_cache_bytes
                    : 0;
        }
        return target;
    };
    auto evict_to_explicit_target = [&](size_t target) -> bool {
        return force_evict_to_target(target);
    };
    auto ensure_cut_compute_queue = [&]() -> bool {
        if (est->cut_compute_queue) return true;
        cl_int err = CL_SUCCESS;
        static const bool device_timing = []() {
            const char * e = std::getenv("GGML_ELASTIC_DEVICE_TIMING");
            return e && *e && *e != '0';
        }();
        const cl_queue_properties props[] = {
            CL_QUEUE_PROPERTIES,
            device_timing
                ? static_cast<cl_queue_properties>(
                      CL_QUEUE_PROFILING_ENABLE)
                : 0,
            0,
        };
        est->cut_compute_queue = clCreateCommandQueueWithProperties(
            backend_ctx->context, backend_ctx->device,
            device_timing ? props : nullptr, &err);
        if (err != CL_SUCCESS || !est->cut_compute_queue) {
            est->cut_compute_queue = nullptr;
            est->cut_dual_queue_errors++;
            GGML_LOG_WARN(
                "ggml_opencl elastic: Cut auxiliary compute queue "
                "unavailable (%d); using sequential Cut\n",
                err);
            return false;
        }
        GGML_LOG_INFO(
            "ggml_opencl elastic: Cut dual-compute auxiliary queue enabled\n");
        return true;
    };
    auto compute_cut_fused = [&](ggml_tensor &cut0,
                                 ggml_tensor &cut1) -> bool {
#if defined(GGML_OPENCL_USE_ADRENO_KERNELS) && defined(GGML_OPENCL_SOA_Q)
        if (!backend_ctx->kernel_gemv_noshuffle_q4_0_f32_dual ||
            backend_ctx->gpu_family != ADRENO ||
            cut0.src[0]->type != GGML_TYPE_Q4_0 ||
            cut1.src[0]->type != GGML_TYPE_Q4_0 ||
            cut0.src[1] != cut1.src[1] ||
            cut0.ne[0] != cut1.ne[0] ||
            cut0.src[0]->ne[0] != cut1.src[0]->ne[0] ||
            cut0.ne[1] * cut0.ne[2] * cut0.ne[3] != 1 ||
            cut1.ne[1] * cut1.ne[2] * cut1.ne[3] != 1) {
            return false;
        }
        auto * q0 =
            (ggml_tensor_extra_cl_q4_0 *) cut0.src[0]->extra;
        auto * q1 =
            (ggml_tensor_extra_cl_q4_0 *) cut1.src[0]->extra;
        auto * activation =
            (ggml_tensor_extra_cl *) cut0.src[1]->extra;
        auto * output =
            (ggml_tensor_extra_cl *) cut0.extra;
        if (!q0 || !q1 || !activation || !output ||
            !q0->q || !q0->d || !q1->q || !q1->d) {
            return false;
        }

        const int K = (int) cut0.src[0]->ne[0];
        const int M = (int) cut0.src[0]->ne[1];
        if (K <= 0 || M <= 0 || (K % 32) != 0 ||
            (M % 2) != 0) {
            return false;
        }
        auto q_image = [&](ggml_tensor_extra_cl_q4_0 * extra) -> cl_mem {
            if (extra->q_img) {
                est->cut_dual_image_cache_hits++;
                return extra->q_img;
            }
            cl_image_format format = {CL_R, CL_UNSIGNED_INT32};
            cl_image_desc desc{};
            desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            desc.image_width =
                (size_t) M * (size_t) K / 2 / sizeof(cl_uint);
            desc.buffer = extra->q;
            cl_int image_err = CL_SUCCESS;
            cl_mem image = clCreateImage(
                backend_ctx->context, CL_MEM_READ_ONLY, &format, &desc,
                nullptr, &image_err);
            if (image_err != CL_SUCCESS || !image) {
                est->cut_dual_image_errors++;
                return nullptr;
            }
            extra->q_img = image;
            extra->owns_q_img = true;
            est->cut_dual_image_creates++;
            return image;
        };
        cl_mem image0 = q_image(q0);
        cl_mem image1 = q_image(q1);
        if (!image0 || !image1) return false;

        const cl_ulong offset1 =
            activation->offset + cut0.src[1]->view_offs;
        const cl_ulong offsetd0 =
            output->offset + cut0.view_offs;
        const cl_ulong offsetd1 =
            output->offset + cut1.view_offs;
        cl_kernel kernel =
            backend_ctx->kernel_gemv_noshuffle_q4_0_f32_dual;
        cl_int err = CL_SUCCESS;
        cl_uint arg = 0;
        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &image0);
        if (err == CL_SUCCESS) {
            err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &q0->d);
        }
        if (err == CL_SUCCESS) {
            err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &image1);
        }
        if (err == CL_SUCCESS) {
            err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &q1->d);
        }
        if (err == CL_SUCCESS) {
            err = clSetKernelArg(
                kernel, arg++, sizeof(cl_mem),
                &activation->data_device);
        }
        if (err == CL_SUCCESS) {
            err = clSetKernelArg(
                kernel, arg++, sizeof(cl_ulong), &offset1);
        }
        if (err == CL_SUCCESS) {
            err = clSetKernelArg(
                kernel, arg++, sizeof(cl_mem), &output->data_device);
        }
        if (err == CL_SUCCESS) {
            err = clSetKernelArg(
                kernel, arg++, sizeof(cl_ulong), &offsetd0);
        }
        if (err == CL_SUCCESS) {
            err = clSetKernelArg(
                kernel, arg++, sizeof(cl_ulong), &offsetd1);
        }
        if (err == CL_SUCCESS) {
            err = clSetKernelArg(kernel, arg++, sizeof(int), &K);
        }
        if (err == CL_SUCCESS) {
            err = clSetKernelArg(kernel, arg++, sizeof(int), &M);
        }
        const size_t wavesize = backend_ctx->adreno_wave_size;
        constexpr size_t n_simgroup = 4;
        const size_t global[] = {
            (((size_t) M / 2 + wavesize - 1) / wavesize) *
                wavesize,
            n_simgroup,
            2,
        };
        const size_t local[] = {wavesize, n_simgroup, 1};
        if (err == CL_SUCCESS) {
            err = clEnqueueNDRangeKernel(
                backend_ctx->queue, kernel, 3, nullptr, global, local,
                0, nullptr, nullptr);
        }
        return err == CL_SUCCESS;
#else
        GGML_UNUSED(cut0);
        GGML_UNUSED(cut1);
        return false;
#endif
    };

    auto run_cut_matmul = [&](
            ggml_tensor *node,
            const std::vector<elastic_q4_cut_part> &parts,
            bool fine_grained_cut) -> bool {
        if (!node || !node->src[0] || node->op != GGML_OP_MUL_MAT || parts.empty()) {
            return false;
        }
        // Activation/bias inputs remain ordinary dependencies.  src0 is
        // supplied one independently resident part at a time below.
        ggml_tensor deps = *node;
        deps.src[0] = nullptr;
        if (!ensure_node_srcs_resident(&deps)) return false;

        const ggml_tensor *original_weight = node->src[0];
        auto *original_dst_extra = (ggml_tensor_extra_cl *) node->extra;
        if (!original_dst_extra || node->type != GGML_TYPE_F32) return false;
        const int64_t output_columns = node->ne[1] * node->ne[2] * node->ne[3];
        auto build_cut_node = [&](const elastic_q4_cut_part &part,
                                  ggml_tensor &weight_part,
                                  ggml_tensor &cut_node) {
            weight_part = *original_weight;
            weight_part.ne[1] = part.row_count;
            weight_part.ne[2] = 1;
            weight_part.ne[3] = 1;
            weight_part.nb[2] =
                weight_part.nb[1] * (size_t) weight_part.ne[1];
            weight_part.nb[3] = weight_part.nb[2];
            weight_part.view_src = nullptr;
            weight_part.view_offs = 0;
            weight_part.data =
                static_cast<unsigned char *>(original_weight->data) +
                part.byte_offset;
            weight_part.extra = part.extra;

            cut_node = *node;
            cut_node.src[0] = &weight_part;
            cut_node.ne[0] = part.row_count;
            cut_node.nb[1] =
                (size_t) part.row_count * sizeof(float);
            cut_node.nb[2] =
                cut_node.nb[1] * (size_t) cut_node.ne[1];
            cut_node.nb[3] =
                cut_node.nb[2] * (size_t) cut_node.ne[2];
            cut_node.view_offs = node->view_offs +
                (size_t) part.row_start * sizeof(float);
        };
        auto prepare_cut_part = [&](const elastic_q4_cut_part &part,
                                    ggml_tensor &cut_node,
                                    bool defer_pipeline_release,
                                    std::vector<size_t> *deferred_positions) -> bool {
            // Keep LOAD/PREPARE and residency independent per half even when
            // the two ready kernels are later submitted as one compute pair.
            ggml_tensor ensure_node = cut_node;
            for (int k = 1; k < GGML_MAX_SRC; ++k) {
                ensure_node.src[k] = nullptr;
            }
            const pipeline_observation pipe_obs =
                pipeline_begin(part.wbm_idx);
            const bool had_pipeline_pin =
                pipe_obs.valid && pipeline_issued[pipe_obs.pos];
            if (!ensure_node_srcs_resident(&ensure_node)) return false;
            evict_to_current_budget();
            sample_pipeline_budget();
            pipeline_finish_observation(pipe_obs);
            if (pipe_obs.valid) {
                stage_pipeline_after(pipe_obs.pos, {part.wbm_idx});
                evict_to_current_budget();
                sample_pipeline_budget();
                if (had_pipeline_pin && defer_pipeline_release &&
                    deferred_positions) {
                    deferred_positions->push_back(pipe_obs.pos);
                } else if (had_pipeline_pin) {
                    release_pipeline_pins(
                        pipeline_units[pipe_obs.pos]);
                }
            }
            return true;
        };

        // Decode GEMV writes each row range directly into a disjoint output
        // region, so two half-tensor kernels can execute concurrently. Prompt
        // GEMM still uses the shared transpose/output scratch and remains on
        // the sequential path.
        if (cut_dual_compute_mode != 0) {
            est->cut_dual_candidates++;
            const bool dual_shape =
                parts.size() == 2 && output_columns == 1;
            if (!dual_shape) {
                est->cut_dual_shape_fallbacks++;
            } else if (cut_dual_compute_mode == 2 ||
                       ensure_cut_compute_queue()) {
                std::vector<int> pair_indices;
                std::unordered_set<int> pair_seen;
                size_t missing_bytes = 0;
                for (const auto & part : parts) {
                    if (!pair_seen.insert(part.wbm_idx).second) continue;
                    const elastic::block_meta * bm =
                        elastic::wbm_get(&est->wbm, part.wbm_idx);
                    if (!bm) continue;
                    pair_indices.push_back(part.wbm_idx);
                    if (!bm->resident) missing_bytes += bm->byte_size;
                }
                acquire_pipeline_pins(pair_indices);
                auto release_pair_pins = [&]() {
                    release_pipeline_pins(pair_indices);
                    pair_indices.clear();
                };
                std::vector<size_t> deferred_pipeline_positions;
                auto release_deferred_pipeline_pins = [&]() {
                    for (size_t pos : deferred_pipeline_positions) {
                        release_pipeline_pins(pipeline_units[pos]);
                    }
                    deferred_pipeline_positions.clear();
                };

                const size_t target = current_weight_target();
                const size_t reserve_target =
                    target == SIZE_MAX
                        ? SIZE_MAX
                        : (target > missing_bytes
                               ? target - missing_bytes
                               : 0);
                bool pair_fits =
                    target == SIZE_MAX ||
                    evict_to_explicit_target(reserve_target);
                if (pair_fits) {
                    std::vector<ggml_tensor> weight_parts(2);
                    std::vector<ggml_tensor> cut_nodes(2);
                    for (size_t p = 0; p < 2; ++p) {
                        build_cut_node(
                            parts[p], weight_parts[p], cut_nodes[p]);
                        if (fine_grained_cut &&
                            !prepare_cut_part(
                                parts[p], cut_nodes[p], true,
                                &deferred_pipeline_positions)) {
                            release_deferred_pipeline_pins();
                            release_pair_pins();
                            return false;
                        }
                    }
                    evict_to_current_budget();
                    pair_fits =
                        target == SIZE_MAX ||
                        est->wbm.resident_bytes <= target;
                    sample_pipeline_budget();

                    if (pair_fits) {
                        if (cut_dual_compute_mode == 2) {
                            if (!compute_cut_fused(
                                    cut_nodes[0], cut_nodes[1])) {
                                est->cut_dual_queue_errors++;
                                release_deferred_pipeline_pins();
                                release_pair_pins();
                                return false;
                            }
                            if (fine_grained_cut &&
                                (!complete_granularity_unit() ||
                                 !complete_granularity_unit())) {
                                release_deferred_pipeline_pins();
                                release_pair_pins();
                                return false;
                            }
                            for (const auto & part : parts) {
                                elastic::wbm_touch(
                                    &est->wbm, part.wbm_idx,
                                    est->current_token);
                                if (fine_grained_cut) {
                                    est->granularity_units += 1;
                                    est->granularity_peak_unit_bytes =
                                        std::max(
                                            est->granularity_peak_unit_bytes,
                                            part.byte_size);
                                }
                            }
                            release_deferred_pipeline_pins();
                            release_pair_pins();
                            evict_to_current_budget();
                            sample_pipeline_budget();
                            est->cut_dual_calls++;
                            if (fine_grained_cut) {
                                est->granularity_cut_ops += 1;
                            }
                            return true;
                        }
                        cl_command_queue primary = backend_ctx->queue;
                        cl_command_queue auxiliary =
                            est->cut_compute_queue;
                        cl_event inputs_ready = nullptr;
                        cl_event auxiliary_done = nullptr;
                        cl_int err = clEnqueueMarkerWithWaitList(
                            primary, 0, nullptr, &inputs_ready);
                        if (err == CL_SUCCESS) err = clFlush(primary);
                        if (err == CL_SUCCESS) {
                            err = clEnqueueBarrierWithWaitList(
                                auxiliary, 1, &inputs_ready, nullptr);
                        }
                        if (err != CL_SUCCESS) {
                            if (inputs_ready) {
                                clReleaseEvent(inputs_ready);
                            }
                            est->cut_dual_queue_errors++;
                            release_deferred_pipeline_pins();
                            release_pair_pins();
                            evict_to_current_budget();
                            // No Cut kernel was submitted, so the ordinary
                            // sequential path below is still a safe fallback.
                        } else {
                            if (!ggml_cl_compute_forward(
                                    backend, &cut_nodes[0])) {
                                clReleaseEvent(inputs_ready);
                                release_deferred_pipeline_pins();
                                release_pair_pins();
                                return false;
                            }
                            backend_ctx->queue = auxiliary;
                            const bool second_ok =
                                ggml_cl_compute_forward(
                                    backend, &cut_nodes[1]);
                            backend_ctx->queue = primary;
                            if (!second_ok) {
                                clReleaseEvent(inputs_ready);
                                release_deferred_pipeline_pins();
                                release_pair_pins();
                                return false;
                            }
                            err = clEnqueueMarkerWithWaitList(
                                auxiliary, 0, nullptr, &auxiliary_done);
                            if (err == CL_SUCCESS) err = clFlush(auxiliary);
                            if (err == CL_SUCCESS) {
                                err = clEnqueueBarrierWithWaitList(
                                    primary, 1, &auxiliary_done, nullptr);
                            }
                            clReleaseEvent(inputs_ready);
                            if (auxiliary_done) {
                                clReleaseEvent(auxiliary_done);
                            }
                            if (err != CL_SUCCESS) {
                                est->cut_dual_queue_errors++;
                                release_deferred_pipeline_pins();
                                release_pair_pins();
                                return false;
                            }
                            if (fine_grained_cut &&
                                (!complete_granularity_unit() ||
                                 !complete_granularity_unit())) {
                                release_deferred_pipeline_pins();
                                release_pair_pins();
                                return false;
                            }
                            for (const auto & part : parts) {
                                elastic::wbm_touch(
                                    &est->wbm, part.wbm_idx,
                                    est->current_token);
                                if (fine_grained_cut) {
                                    est->granularity_units += 1;
                                    est->granularity_peak_unit_bytes =
                                        std::max(
                                            est->granularity_peak_unit_bytes,
                                            part.byte_size);
                                }
                            }
                            release_deferred_pipeline_pins();
                            release_pair_pins();
                            evict_to_current_budget();
                            sample_pipeline_budget();
                            est->cut_dual_calls++;
                            if (fine_grained_cut) {
                                est->granularity_cut_ops += 1;
                            }
                            return true;
                        }
                    }
                }
                if (!pair_fits) {
                    est->cut_dual_budget_fallbacks++;
                    release_deferred_pipeline_pins();
                    release_pair_pins();
                    evict_to_current_budget();
                }
            }
        }

        for (const auto &part : parts) {
            ggml_tensor weight_part{};
            ggml_tensor cut_node{};
            build_cut_node(part, weight_part, cut_node);
            if (fine_grained_cut &&
                !prepare_cut_part(
                    part, cut_node, false, nullptr)) return false;

            ggml_tensor_extra_cl scratch_extra{};
            const bool direct_output = output_columns == 1;
            if (!direct_output) {
                const size_t scratch_bytes = (size_t) part.row_count *
                    (size_t) output_columns * sizeof(float);
                if (scratch_bytes > est->cut_output_scratch_bytes) {
                    CL_CHECK(clFinish(backend_ctx->queue));
                    if (est->cut_output_scratch) {
                        CL_CHECK(clReleaseMemObject(est->cut_output_scratch));
                    }
                    cl_int err = CL_SUCCESS;
                    est->cut_output_scratch = clCreateBuffer(
                        backend_ctx->context, CL_MEM_READ_WRITE,
                        scratch_bytes, nullptr, &err);
                    if (err != CL_SUCCESS || !est->cut_output_scratch) return false;
                    est->cut_output_scratch_bytes = scratch_bytes;
                }
                scratch_extra.reset();
                scratch_extra.data_device = est->cut_output_scratch;
                scratch_extra.actual_size = est->cut_output_scratch_bytes;
                cut_node.extra = &scratch_extra;
                cut_node.view_offs = 0;
            }

            if (!ggml_cl_compute_forward(backend, &cut_node)) return false;
            if (!direct_output) {
                const size_t part_row_bytes =
                    (size_t) part.row_count * sizeof(float);
                const size_t dst_base = (size_t) original_dst_extra->offset +
                    node->view_offs + (size_t) part.row_start * sizeof(float);
                for (int64_t column = 0; column < output_columns; ++column) {
                    const size_t src_offset = (size_t) column * part_row_bytes;
                    const size_t dst_offset = dst_base +
                        (size_t) column * node->nb[1];
                    const cl_int err = clEnqueueCopyBuffer(
                        backend_ctx->queue, est->cut_output_scratch,
                        original_dst_extra->data_device,
                        src_offset, dst_offset, part_row_bytes,
                        0, nullptr, nullptr);
                    if (err != CL_SUCCESS) return false;
                }
            }
            if (fine_grained_cut &&
                !complete_granularity_unit()) return false;
            elastic::wbm_touch(&est->wbm, part.wbm_idx, est->current_token);
            evict_to_current_budget();
            if (fine_grained_cut) {
                est->granularity_units += 1;
                est->granularity_peak_unit_bytes = std::max(
                    est->granularity_peak_unit_bytes, part.byte_size);
            }
        }
        if (fine_grained_cut) est->granularity_cut_ops += 1;
        return true;
    };

    auto ensure_tiled_weight_parts =
        [&](ggml_tensor * node,
            const std::vector<elastic_q4_cut_part> & parts) -> bool {
        if (!node || !node->src[0]) return false;
        for (const auto & part : parts) {
            ggml_tensor weight_part = *node->src[0];
            weight_part.ne[1] = part.row_count;
            weight_part.ne[2] = 1;
            weight_part.ne[3] = 1;
            weight_part.nb[2] =
                weight_part.nb[1] * (size_t) weight_part.ne[1];
            weight_part.nb[3] = weight_part.nb[2];
            weight_part.view_src = nullptr;
            weight_part.view_offs = 0;
            weight_part.data =
                static_cast<unsigned char *>(node->src[0]->data) +
                part.byte_offset;
            weight_part.extra = part.extra;

            ggml_tensor ensure_node = *node;
            ensure_node.src[0] = &weight_part;
            for (int k = 1; k < GGML_MAX_SRC; ++k) {
                ensure_node.src[k] = nullptr;
            }
            if (!ensure_node_srcs_resident(&ensure_node)) return false;
        }
        return true;
    };

    int multi_weights_remaining = 0;
    size_t multi_pipeline_pos = SIZE_MAX;
    std::vector<std::pair<int, bool>> multi_unit_pins;
    auto release_multi_unit_pins = [&]() {
        for (const auto &entry : multi_unit_pins) {
            elastic::wbm_set_pinned(&est->wbm, entry.first, entry.second);
        }
        multi_unit_pins.clear();
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        auto tiled_weight =
            node && node->src[0]
                ? est->q4_cut_parts.find(node->src[0])
                : est->q4_cut_parts.end();
        const auto & mixed_binding =
            mixed_binding_for_weight(
                node && node->src[0] ? node->src[0] : nullptr);
        const bool mixed_current = mixed_binding.planned;
        const bool fine_cut_current =
            mixed_current
                ? mixed_binding.fine_cut
                : est->granularity.mode ==
                    elastic::granularity_mode::CUT;
        size_t mixed_pipeline_pos = SIZE_MAX;
        if (mixed_current && node && node->src[0]) {
            auto found =
                mixed_pipeline_pos_by_weight.find(node->src[0]);
            if (found != mixed_pipeline_pos_by_weight.end()) {
                mixed_pipeline_pos = found->second;
            }
        }
        const bool grouped_current =
            mixed_current
                ? mixed_pipeline_pos != SIZE_MAX &&
                    mixed_pipeline_pos <
                        mixed_nodes_by_pipeline_pos.size() &&
                    mixed_nodes_by_pipeline_pos[
                        mixed_pipeline_pos].size() > 1
                : est->granularity.mode ==
                        elastic::granularity_mode::MULTI ||
                    est->granularity.mode ==
                        elastic::granularity_mode::MULTI_FUSED;

        if (granularity_active &&
            fine_cut_current &&
            node && node->op == GGML_OP_MUL_MAT && node->src[0]) {
            if (tiled_weight != est->q4_cut_parts.end()) {
                sync_with_other_backends(backend);
                if (!run_cut_matmul(
                        node, tiled_weight->second, true)) {
                    return GGML_STATUS_FAILED;
                }
                continue;
            }
        }

        const bool current_weight_matmul = is_weight_matmul(node);
        pipeline_observation pipe_obs;
        std::vector<int> current_pipeline_indices;
        std::vector<int> current_execution_indices;
        std::unordered_set<int> current_execution_seen;
        bool current_execution_pin = false;
        bool current_prefetch_pin = false;
        auto acquire_current_source_pins = [&](ggml_tensor * current_node) {
            if (!elastic_active || !current_node) return;
            std::vector<int> added;
            for (int src_pos = 0; src_pos < GGML_MAX_SRC; ++src_pos) {
                ggml_tensor * src = current_node->src[src_pos];
                if (!src || !src->extra) continue;
                auto src_tiled = est->q4_cut_parts.find(src);
                if (src_tiled != est->q4_cut_parts.end()) {
                    for (const auto & part : src_tiled->second) {
                        if (part.wbm_idx >= 0 &&
                            current_execution_seen.insert(
                                part.wbm_idx).second) {
                            added.push_back(part.wbm_idx);
                        }
                    }
                    continue;
                }
                const int src_idx = ggml_opencl_get_wbm_idx(src);
                if (src_idx >= 0 &&
                    current_execution_seen.insert(src_idx).second) {
                    added.push_back(src_idx);
                }
            }
            if (!added.empty()) {
                acquire_pipeline_pins(added);
                current_execution_indices.insert(
                    current_execution_indices.end(),
                    added.begin(), added.end());
                current_execution_pin = true;
            }
        };
        if (granularity_active && unit_pipeline_enabled && current_weight_matmul) {
            int current_idx = ggml_opencl_get_wbm_idx(node->src[0]);
            if (mixed_pipeline_pos != SIZE_MAX &&
                mixed_pipeline_pos < pipeline_units.size() &&
                !pipeline_units[mixed_pipeline_pos].empty()) {
                current_idx =
                    pipeline_units[mixed_pipeline_pos].front();
            }
            if (!grouped_current || multi_weights_remaining == 0) {
                pipe_obs = pipeline_begin(current_idx);
                if (pipe_obs.valid) current_pipeline_indices = pipeline_units[pipe_obs.pos];
            }
            // pipeline_begin() only observes an existing lookahead pin; the
            // first unit in a graph (and any repeated consumer) historically
            // had no protection of its own.  During a sharp budget decrease
            // the post-ensure budget pass could therefore select the just
            // touched current Tensor unit as the MRU victim, clear its SOA
            // q/d handles, and then submit its GEMV with a null cl_mem.
            //
            // Give every non-Multi consumer one execution reference from
            // before ensure through kernel submission.  A separately issued
            // lookahead reference is consumed after the same submission.
            // Multi already protects the complete group with
            // multi_unit_pins and retires it at the group boundary.
            if (!grouped_current) {
                auto current_pos = pipeline_pos_by_wbm.find(current_idx);
                if (current_pos != pipeline_pos_by_wbm.end()) {
                    current_pipeline_indices =
                        pipeline_units[current_pos->second];
                    current_prefetch_pin =
                        pipe_obs.valid &&
                        pipeline_issued[current_pos->second];
                    for (int idx : current_pipeline_indices) {
                        if (current_execution_seen.insert(idx).second) {
                            current_execution_indices.push_back(idx);
                        }
                    }
                    if (!current_execution_indices.empty()) {
                        acquire_pipeline_pins(
                            current_execution_indices);
                        current_execution_pin = true;
                    }
                }
            }
        }
        // Every WBM source consumed by the current operator must remain valid
        // through kernel submission, not only src[0] of MUL_MAT.  In
        // particular RMS_NORM->MUL consumes a small norm weight as MUL src[1].
        // MRU budget enforcement could otherwise evict that just-ensured
        // source while reserving the following matrix unit and submit a null
        // cl_mem to the fused/non-fused elementwise kernel.
        acquire_current_source_pins(node);
        auto release_current_execution_pins = [&]() {
            if (current_execution_pin) {
                release_pipeline_pins(current_execution_indices);
                current_execution_pin = false;
            }
            if (current_prefetch_pin) {
                release_pipeline_pins(current_pipeline_indices);
                current_prefetch_pin = false;
            }
        };
        // LLAMA_ELASTIC_FORCE_PLAN_EVICT runs with NO_AUTO_EVICT so that the
        // placement plan, rather than the backend's MRU policy, chooses normal
        // victims.  A current execution unit is different: if it is missing,
        // ensure_node_srcs_resident() must materialize it before the next plan
        // callback.  Reserve those bytes explicitly while the unit is pinned,
        // otherwise resident_bytes transiently grows by one Tensor unit on
        // every miss and violates the advertised weight budget.
        //
        // Pipeline positions use the actual Cut/Tensor/Multi storage indices.
        // Fall back to the node binding only for the first, not-yet-observed
        // graph.
        if (granularity_active && current_weight_matmul &&
            !grouped_current) {
            if (current_pipeline_indices.empty()) {
                auto current_tiled =
                    est->q4_cut_parts.find(node->src[0]);
                if (current_tiled != est->q4_cut_parts.end()) {
                    for (const auto & part : current_tiled->second) {
                        current_pipeline_indices.push_back(part.wbm_idx);
                    }
                } else {
                    const int current_idx =
                        ggml_opencl_get_wbm_idx(node->src[0]);
                    if (current_idx >= 0) {
                        current_pipeline_indices.push_back(current_idx);
                    }
                }
            }
            size_t current_missing_bytes = 0;
            std::unordered_set<int> current_seen;
            for (int idx : current_execution_indices) {
                if (!current_seen.insert(idx).second) continue;
                const elastic::block_meta * bm =
                    elastic::wbm_get(&est->wbm, idx);
                if (bm && !bm->resident) {
                    current_missing_bytes += bm->byte_size;
                }
            }
            if (current_missing_bytes > 0) {
                const size_t target = current_weight_target();
                const size_t aligned_target =
                    target <= SIZE_MAX - weight_budget_alignment_slack
                        ? target + weight_budget_alignment_slack
                        : SIZE_MAX;
                const size_t reserve_target =
                    target == SIZE_MAX
                        ? SIZE_MAX
                        : (aligned_target > current_missing_bytes
                               ? aligned_target - current_missing_bytes
                               : 0);
                if (target != SIZE_MAX &&
                    !evict_to_explicit_target(reserve_target)) {
                    release_current_execution_pins();
                    return GGML_STATUS_FAILED;
                }
            }
        }
        if (granularity_active && current_weight_matmul &&
            grouped_current &&
            multi_weights_remaining == 0) {
            std::unordered_set<const ggml_tensor *> seen;
            size_t unit_bytes = 0;
            std::vector<ggml_tensor *> future_nodes;
            if (mixed_current &&
                mixed_pipeline_pos <
                    mixed_nodes_by_pipeline_pos.size()) {
                future_nodes =
                    mixed_nodes_by_pipeline_pos[mixed_pipeline_pos];
            } else {
                for (int j = i; j < cgraph->n_nodes &&
                     (int) seen.size() <
                        est->granularity.multi_tensors; ++j) {
                    ggml_tensor * future = cgraph->nodes[j];
                    if (is_weight_matmul(future) &&
                        seen.insert(future->src[0]).second) {
                        future_nodes.push_back(future);
                    }
                }
                seen.clear();
            }
            for (ggml_tensor * future : future_nodes) {
                if (!is_weight_matmul(future) ||
                    !seen.insert(future->src[0]).second) continue;
                const int future_idx = ggml_opencl_get_wbm_idx(future->src[0]);
                auto future_tiled =
                    est->q4_cut_parts.find(future->src[0]);
                std::vector<int> future_indices;
                if (future_tiled != est->q4_cut_parts.end()) {
                    for (const auto & part : future_tiled->second) {
                        future_indices.push_back(part.wbm_idx);
                    }
                } else if (future_idx >= 0) {
                    future_indices.push_back(future_idx);
                }
                for (int idx : future_indices) {
                    const elastic::block_meta *future_bm =
                        elastic::wbm_get(&est->wbm, idx);
                    if (!future_bm) continue;
                    multi_unit_pins.push_back(
                        {idx, future_bm->is_pinned});
                    elastic::wbm_set_pinned(
                        &est->wbm, idx, true);
                }
                if (!ensure_node_srcs_resident(future)) return GGML_STATUS_FAILED;
                if (future_tiled != est->q4_cut_parts.end() &&
                    !ensure_tiled_weight_parts(
                        future, future_tiled->second)) {
                    return GGML_STATUS_FAILED;
                }
                unit_bytes += weight_bytes(future);
            }
            multi_weights_remaining = (int) seen.size();
            est->granularity_units += 1;
            est->granularity_peak_unit_bytes = std::max(
                est->granularity_peak_unit_bytes, unit_bytes);
        }

        // 当前 node 的 src 先 ensure；fused 分支命中时再补本 i+1/i+2 节点的 src。
        if (!ensure_node_srcs_resident(node)) return GGML_STATUS_FAILED;
        if (tiled_weight != est->q4_cut_parts.end() &&
            !fine_cut_current &&
            !ensure_tiled_weight_parts(node, tiled_weight->second)) {
            return GGML_STATUS_FAILED;
        }
        if (granularity_active && current_weight_matmul &&
            !grouped_current) {
            const size_t target = current_weight_target();
            const size_t aligned_target =
                target <= SIZE_MAX - weight_budget_alignment_slack
                    ? target + weight_budget_alignment_slack
                    : SIZE_MAX;
            if (aligned_target != SIZE_MAX &&
                !evict_to_explicit_target(aligned_target)) {
                release_current_execution_pins();
                return GGML_STATUS_FAILED;
            }
        } else {
            evict_to_current_budget();
        }
        sample_pipeline_budget();
        pipeline_finish_observation(pipe_obs);
        if (pipe_obs.valid) {
            stage_pipeline_after(pipe_obs.pos, current_pipeline_indices);
            evict_to_current_budget();
            sample_pipeline_budget();
            if (grouped_current) {
                // The execution-unit pin was acquired while this pipeline pin
                // was already set.  Release them in reverse nesting order at
                // the Multi boundary; restoring the execution pin first and
                // immediately dropping the pipeline pin recovers the true
                // original bit instead of permanently pinning the model.
                multi_pipeline_pos = pipe_obs.pos;
            }
        }

        // NOTE: this may oversynchronize by synchronizing with
        //       backends/devices which don't compute 'cgraph's
        //       dependencies.
        sync_with_other_backends(backend);

        // Dynamic Tensor/Multi plans use the same independently resident base
        // tiles as CUT, but group their LOAD/PREPARE/retire boundaries above.
        // Compute both halves here (normally one dual GEMV dispatch) instead
        // of accidentally executing only the first tile referenced by the
        // original tensor->extra.
        if (granularity_active &&
            tiled_weight != est->q4_cut_parts.end() &&
            node && node->op == GGML_OP_MUL_MAT &&
            !fine_cut_current) {
            std::vector<std::pair<int, bool>> tensor_tile_pins;
            if (!grouped_current) {
                for (const auto & part : tiled_weight->second) {
                    const elastic::block_meta * bm =
                        elastic::wbm_get(&est->wbm, part.wbm_idx);
                    if (!bm) continue;
                    tensor_tile_pins.push_back(
                        {part.wbm_idx, bm->is_pinned});
                    elastic::wbm_set_pinned(
                        &est->wbm, part.wbm_idx, true);
                }
            }
            const bool tiled_ok =
                run_cut_matmul(node, tiled_weight->second, false);
            for (const auto & pin : tensor_tile_pins) {
                elastic::wbm_set_pinned(
                    &est->wbm, pin.first, pin.second);
            }
            if (!tiled_ok) {
                return GGML_STATUS_FAILED;
            }
            if (!grouped_current) {
                if (!complete_granularity_unit()) {
                    return GGML_STATUS_FAILED;
                }
                est->granularity_units += 1;
                est->granularity_peak_unit_bytes = std::max(
                    est->granularity_peak_unit_bytes,
                    weight_bytes(node));
            } else {
                if (multi_weights_remaining > 0) {
                    --multi_weights_remaining;
                }
                if (multi_weights_remaining == 0) {
                    if (!complete_granularity_unit()) {
                        return GGML_STATUS_FAILED;
                    }
                    release_multi_unit_pins();
                    if (multi_pipeline_pos != SIZE_MAX) {
                        release_pipeline_pins(
                            pipeline_units[multi_pipeline_pos]);
                        multi_pipeline_pos = SIZE_MAX;
                    }
                }
            }
            // All kernels consuming the current coarse Tensor unit have now
            // been queued.  It is safe for the next budget pass to retire the
            // unit; retained-pool reuse still waits on its queue marker.
            release_current_execution_pins();
            continue;
        }

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            release_current_execution_pins();
            continue;
        }

        if (!backend_ctx->disable_fusion && ggml_opencl_can_fuse(cgraph, i, { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
            acquire_current_source_pins(cgraph->nodes[i+1]);
            acquire_current_source_pins(cgraph->nodes[i+2]);
            if (!ensure_node_srcs_resident(cgraph->nodes[i+1])) {
                release_current_execution_pins();
                return GGML_STATUS_FAILED;
            }
            if (!ensure_node_srcs_resident(cgraph->nodes[i+2])) {
                release_current_execution_pins();
                return GGML_STATUS_FAILED;
            }
            ggml_opencl_op_norm_fused(backend, node, cgraph->nodes[i+1], cgraph->nodes[i+2]);
            release_current_execution_pins();
            i += 2;
            continue;
        }
        if (!backend_ctx->disable_fusion && ggml_opencl_can_fuse(cgraph, i, { GGML_OP_GROUP_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
            acquire_current_source_pins(cgraph->nodes[i+1]);
            acquire_current_source_pins(cgraph->nodes[i+2]);
            if (!ensure_node_srcs_resident(cgraph->nodes[i+1])) {
                release_current_execution_pins();
                return GGML_STATUS_FAILED;
            }
            if (!ensure_node_srcs_resident(cgraph->nodes[i+2])) {
                release_current_execution_pins();
                return GGML_STATUS_FAILED;
            }
            ggml_opencl_op_group_norm_fused(backend, node, cgraph->nodes[i+1], cgraph->nodes[i+2]);
            release_current_execution_pins();
            i += 2;
            continue;
        }
        if (!backend_ctx->disable_fusion && ggml_opencl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            acquire_current_source_pins(cgraph->nodes[i+1]);
            if (!ensure_node_srcs_resident(cgraph->nodes[i+1])) {
                release_current_execution_pins();
                return GGML_STATUS_FAILED;
            }
            ggml_opencl_op_rms_norm_fused(backend, node, cgraph->nodes[i+1]);
            release_current_execution_pins();
            i++;
            continue;
        }

        const bool csv_compute_profile = elastic_active && est->profile_csv;
        auto op_t0 = (elastic_active && (est->profile || csv_compute_profile))
                     ? std::chrono::steady_clock::now()
                     : std::chrono::steady_clock::time_point{};
        bool ok = ggml_cl_compute_forward(backend, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
        if (elastic_active && (est->profile || csv_compute_profile)) {
            // 拿真实 GPU 时间：每个 op 后 clFinish。会显著拖速度，仅 profile 模式。
            clFinish(backend_ctx->queue);
            auto op_t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(op_t1 - op_t0).count();
            if (est->profile) {
                auto &b = est->compute_buckets[(int) node->op];
                b.n        += 1;
                b.total_ms += ms;
            }
            if (csv_compute_profile) {
                opencl_profile_compute(node, i, ms, ok ? 1 : 0);
            }
        }

        if (granularity_active && current_weight_matmul) {
            if (mixed_current) {
                if (!grouped_current) {
                    if (!complete_granularity_unit()) return GGML_STATUS_FAILED;
                    est->granularity_units += 1;
                    est->granularity_peak_unit_bytes = std::max(
                        est->granularity_peak_unit_bytes, weight_bytes(node));
                } else {
                    if (multi_weights_remaining > 0) --multi_weights_remaining;
                    if (multi_weights_remaining == 0) {
                        if (!complete_granularity_unit()) return GGML_STATUS_FAILED;
                        release_multi_unit_pins();
                        if (multi_pipeline_pos != SIZE_MAX) {
                            release_pipeline_pins(
                                pipeline_units[multi_pipeline_pos]);
                            multi_pipeline_pos = SIZE_MAX;
                        }
                    }
                }
            } else {
                switch (est->granularity.mode) {
                    case elastic::granularity_mode::TENSOR:
                        if (!complete_granularity_unit()) return GGML_STATUS_FAILED;
                        est->granularity_units += 1;
                        est->granularity_peak_unit_bytes = std::max(
                            est->granularity_peak_unit_bytes, weight_bytes(node));
                        break;
                    case elastic::granularity_mode::MULTI:
                    case elastic::granularity_mode::MULTI_FUSED:
                        if (multi_weights_remaining > 0) --multi_weights_remaining;
                        if (multi_weights_remaining == 0) {
                            if (!complete_granularity_unit()) return GGML_STATUS_FAILED;
                            release_multi_unit_pins();
                            if (multi_pipeline_pos != SIZE_MAX) {
                                release_pipeline_pins(
                                    pipeline_units[multi_pipeline_pos]);
                                multi_pipeline_pos = SIZE_MAX;
                            }
                        }
                        break;
                    case elastic::granularity_mode::CUT:
                        // Non-Q4 or otherwise unsplittable model weights retain a
                        // real one-tensor fallback unit and are reported explicitly.
                        if (!complete_granularity_unit()) return GGML_STATUS_FAILED;
                        est->granularity_units += 1;
                        est->granularity_fallback_ops += 1;
                        est->granularity_peak_unit_bytes = std::max(
                            est->granularity_peak_unit_bytes, weight_bytes(node));
                        break;
                }
            }
        }

        // Keep the current unit pinned through ggml_cl_compute_forward(), not
        // merely through its LOAD/PREPARE stages.  Releasing here preserves
        // overlap while preventing a budget transition from evicting a
        // weight between ensure and kernel submission.
        release_current_execution_pins();

        // Elastic baseline (F4-event)：op 派发后插一个 marker event 到 compute
        // queue 当前位置，写到所有被本 op 用过的 WBM 权重的 last_use_event。
        // 后续 evict 只需 clWaitForEvents 这些 marker（不再 clFinish 整个 queue），
        // 旧 marker 已结束的 buffer wait 立即返回。
        // Per-op marker stamping 已废弃：wbmcl_evict_batch 改成在 evict 当下
        // 插 1 个 marker 等 queue 跑完所有 prior kernel——in-order queue 顺序
        // 保证那时被 evict 的 block 上次用它的 kernel 必然已完成。省下每个
        // op 的 ~5 个 OpenCL API call（约 18k call / token 在 1B 模型上）。

        if (elastic_active && est->bw_inited) {
            est->n_op_dispatched += 1;
            static const bool s_no_auto_evict_periodic = []() {
                const char *e = std::getenv("GGML_ELASTIC_NO_AUTO_EVICT");
                return e && *e && *e != '0';
            }();
            if (!s_no_auto_evict_periodic &&
                (int)(est->n_op_dispatched % est->evict_check_interval) == 0) {
                // Baseline 静态：target = M_floor - kv - misc - safety。
                // 若 GGML_ELASTIC_CL_RETAIN_COUNTS_BUDGET=1，把 pool cap 从 target
                // 里扣掉，让 working_set + cached_bytes ≤ target，严格合规。
                static const bool s_pool_counts_budget = []() {
                    const char *e = std::getenv("GGML_ELASTIC_CL_RETAIN_COUNTS_BUDGET");
                    return e && *e && *e != '0';
                }();
                size_t target = est->dynamic_target
                    ? (([&]{
                          const size_t bt    = elastic::budget_watcher_get(&est->bw) * size_t(1024 * 1024);
                          const size_t km    = est->kv_bytes + est->misc_overhead;
                          const size_t base  = bt > km ? bt - km : 0;
                          return base + est->extra_target_bytes;
                      })())
                    : est->static_target_bytes;
                if (ggml_opencl_moe_expert_cache_enabled()) {
                    target = target > est->moe_cache_bytes ? target - est->moe_cache_bytes : 0;
                }
                if (s_pool_counts_budget && est->octx.cache_byte_limit > 0
                    && target > est->octx.cache_byte_limit) {
                    target -= est->octx.cache_byte_limit;
                }
                if (est->wbm.resident_bytes > target) {
                    // F4-event：不再 clFinish 整个 queue，依赖 wbmcl_evict_batch
                    // 里 clWaitForEvents 在 victim 的 last_use_event 上选择性等待。
                    std::vector<int> victims;
                    int n = elastic::wbm_evict_to_byte_budget(&est->wbm, target,
                                                              /*exclude=*/-1, &victims);
                    if (n > 0) {
                        int released = elastic::wbmcl_evict_batch(&est->octx, victims.data(), n);
                        for (int i = 0; i < released && i < n; ++i) {
                            const std::string name = opencl_name_for_idx(est, victims[i]);
                            if (!name.empty()) {
                                llama_weight_runtime_mark_evicted(name.c_str(), LLAMA_WEIGHT_RUNTIME_GPU);
                            }
                        }
                        est->n_evicts_total += released;
                    }
                }
            }
        }

        // Async disk->host load lookahead: this is the runtime half of the
        // overlapping pipeline. The CP schedule can still place explicit LOAD
        // anchors, but this fallback keeps decode from doing a synchronous
        // direct-read when a future SOA weight was missed by the schedule.
        if (elastic_active && est->octx.async_stage_load) {
            static const int s_async_load_lookahead = []() {
                const char *e = std::getenv("GGML_ELASTIC_ASYNC_LOAD_LOOKAHEAD");
                return e ? std::max(0, atoi(e)) : 0;
            }();
            if (s_async_load_lookahead > 0) {
                const int max_j = std::min(i + 1 + s_async_load_lookahead, cgraph->n_nodes);
                for (int j = i + 1; j < max_j; ++j) {
                    ggml_tensor *nj = cgraph->nodes[j];
                    if (!nj) continue;
                    for (int k = 0; k < GGML_MAX_SRC; ++k) {
                        ggml_tensor *src = nj->src[k];
                        if (!src) continue;
                        if (ggml_opencl_moe_expert_cache_enabled() && ggml_opencl_is_moe_expert_weight(src)) continue;
                        const int se_wbm_idx = ggml_opencl_get_wbm_idx(src);
                        if (se_wbm_idx < 0) continue;
                        const elastic::block_meta *bm =
                            elastic::wbm_get(&est->wbm, se_wbm_idx);
                        if (!bm || bm->resident || bm->prefetch_event) continue;
                        if (!bm->host_ptr || bm->byte_size == 0) continue;
                        elastic::wbmcl_load_host_async(&est->octx, se_wbm_idx);
                    }
                }
            }
        }

        // Pipeline prefetch (F5)：CPU 侧 page-cache 预热。Adreno unified memory
        // 下 GPU async upload 没法跟 compute overlap（同一条 DRAM bus），改成对
        // 后 N 个 node 的 src host_ptr 发 POSIX_MADV_WILLNEED——kernel 后台
        // readahead，等到 ensure_resident 同步上传时 mmap 页已 warm，driver 端
        // memcpy(host→GPU buf) 拿满 ~2.87 GB/s，不再触发 page fault / disk read。
        if (elastic_active && est->prefetch_lookahead > 0) {
            static const size_t pgsz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
            const int max_j = std::min(i + 1 + est->prefetch_lookahead, cgraph->n_nodes);
            for (int j = i + 1; j < max_j; ++j) {
                ggml_tensor *nj = cgraph->nodes[j];
                if (!nj) continue;
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    ggml_tensor *src = nj->src[k];
                    if (!src) continue;
                    if (ggml_opencl_moe_expert_cache_enabled() && ggml_opencl_is_moe_expert_weight(src)) continue;
                    const int se_wbm_idx = ggml_opencl_get_wbm_idx(src);
                    if (se_wbm_idx < 0) continue;
                    const elastic::block_meta *bm =
                        elastic::wbm_get(&est->wbm, se_wbm_idx);
                    if (!bm || !bm->host_ptr || bm->byte_size == 0) continue;
                    if (bm->resident) continue;
                    // host_ptr 是 mmap_base + tensor_offset，tensor offset 通常 32B 对齐
                    // 但不一定 4K page 对齐，向下/向上 round 到 page 边界再 madvise。
                    uintptr_t addr     = reinterpret_cast<uintptr_t>(bm->host_ptr);
                    uintptr_t aligned  = addr & ~(pgsz - 1);
                    size_t    over     = addr - aligned;
                    size_t    raw_len  = bm->byte_size + over;
                    size_t    a_len    = ((raw_len + pgsz - 1) / pgsz) * pgsz;
                    int rc = posix_madvise(reinterpret_cast<void *>(aligned), a_len,
                                           POSIX_MADV_WILLNEED);
                    if (rc == 0) est->n_prefetch_issued += 1;
                    else         est->n_prefetch_skipped += 1;
                }
            }
        }

        // GPU prefetch：往 xfer_queue (可能多条) 上提前发 clEnqueueWriteBuffer。
        // 配合 GGML_ELASTIC_XFER_EXTRA 用多 queue round-robin，让 transfer 真并行。
        // ensure_resident 命中时用 barrier-on-compute_queue 等 prefetch_event。
        if (elastic_active && est->gpu_prefetch_lookahead > 0 && est->octx.xfer_queue) {
            static const size_t inflight_cap = []() {
                const char *e = std::getenv("GGML_ELASTIC_GPU_PREFETCH_INFLIGHT_MB");
                size_t mb = e ? (size_t)atoi(e) : 256;
                return mb * 1024 * 1024;
            }();
            size_t inflight = 0;
            for (const auto &b : est->wbm.blocks) {
                if (!b.resident && b.prefetch_event) inflight += b.byte_size;
            }
            const int max_j = std::min(i + 1 + est->gpu_prefetch_lookahead, cgraph->n_nodes);
            for (int j = i + 1; j < max_j; ++j) {
                ggml_tensor *nj = cgraph->nodes[j];
                if (!nj) continue;
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    ggml_tensor *src = nj->src[k];
                    if (!src) continue;
                    if (ggml_opencl_moe_expert_cache_enabled() && ggml_opencl_is_moe_expert_weight(src)) continue;
                    const int se_wbm_idx = ggml_opencl_get_wbm_idx(src);
                    if (se_wbm_idx < 0) continue;
                    const elastic::block_meta *bm =
                        elastic::wbm_get(&est->wbm, se_wbm_idx);
                    if (!bm) continue;
                    if (bm->resident || bm->prefetch_event) continue;
                    if (inflight + bm->byte_size > inflight_cap) {
                        est->n_gpu_prefetch_skipped += 1;
                        continue;
                    }
                    const bool is_soa = (src->type == GGML_TYPE_Q4_0 ||
                                         src->type == GGML_TYPE_Q8_0 ||
                                         src->type == GGML_TYPE_MXFP4);
                    if (is_soa) {
                        // SOA prefetch：直接调注册的 reload_fn 提前 enqueue 整个
                        // SOA 重建管线 (staging write + convert + transpose, 全在
                        // xfer/compute queue 上 async). reload_fn 会 mark_resident,
                        // 后续 ensure_resident 看到 resident=true 跳过. 让 reload
                        // 真正跟 compute 重叠. env GGML_ELASTIC_SOA_PREFETCH=0 关掉.
                        static const bool s_soa_pf = []() {
                            const char *e = std::getenv("GGML_ELASTIC_SOA_PREFETCH");
                            return !(e && *e == '0');
                        }();
                        if (!s_soa_pf) { est->n_gpu_prefetch_skipped += 1; continue; }
                        auto cit = est->octx.soa_per_idx.find(se_wbm_idx);
                        if (cit == est->octx.soa_per_idx.end() || !cit->second.reload_fn) {
                            est->n_gpu_prefetch_skipped += 1; continue;
                        }
                        auto pf_t0 = (est->profile || est->timing) ? std::chrono::steady_clock::now()
                                                                  : std::chrono::steady_clock::time_point{};
                        int rc = elastic::wbmcl_reload_soa_sync(&est->octx, se_wbm_idx);
                        if (est->profile || est->timing) {
                            auto pf_t1 = std::chrono::steady_clock::now();
                            double ms = std::chrono::duration<double, std::milli>(pf_t1 - pf_t0).count();
                            est->t_reload_host_us += (uint64_t)(ms * 1000.0);
                        }
                        if (rc == 0) {
                            inflight += bm->byte_size;
                            est->n_gpu_prefetch_issued += 1;
                        } else {
                            est->n_gpu_prefetch_skipped += 1;
                        }
                        continue;
                    }
                    int rc = elastic::wbmcl_prefetch_async(&est->octx, se_wbm_idx);
                    if (rc == 0) {
                        inflight += bm->byte_size;
                        est->n_gpu_prefetch_issued += 1;
                        const elastic::block_meta *bm2 =
                            elastic::wbm_get(&est->wbm, se_wbm_idx);
                        if (bm2 && bm2->backend_handle) {
                            cl_mem nb = static_cast<cl_mem>(bm2->backend_handle);
                            ggml_tensor_extra_cl *se = (ggml_tensor_extra_cl *) src->extra;
                            se->data_device = nb;
                            ggml_opencl_elastic_update_ctx_slot(src->buffer, ggml_opencl_get_ctx_slot(src), nb);
                        }
                    } else {
                        est->n_gpu_prefetch_skipped += 1;
                    }
                }
            }
        }
    }

    if (granularity_active && multi_weights_remaining > 0) {
        if (!complete_granularity_unit()) return GGML_STATUS_FAILED;
        release_multi_unit_pins();
        if (multi_pipeline_pos != SIZE_MAX) {
            release_pipeline_pins(
                pipeline_units[multi_pipeline_pos]);
            multi_pipeline_pos = SIZE_MAX;
        }
    }

    if (!pending_cross_graphs.empty()) {
        // Move exactly the successor-chain references to persistent state.
        for (const auto & transfer : pending_cross_pin_refs) {
            auto local = pipeline_pins.find(transfer.first);
            if (local == pipeline_pins.end() ||
                local->second.refs == 0) {
                continue;
            }
            auto & cross =
                est->unit_pipeline_cross_pins[transfer.first];
            if (cross.refs == 0) {
                cross.original = local->second.original;
            }
            const size_t moved =
                std::min(local->second.refs, transfer.second);
            local->second.refs -= moved;
            cross.refs += moved;
            if (local->second.refs == 0) pipeline_pins.erase(local);
            elastic::wbm_set_pinned(&est->wbm, transfer.first, true);
        }
        while (!pending_cross_graphs.empty()) {
            est->unit_pipeline_cross_graphs.push_back(
                std::move(pending_cross_graphs.front()));
            pending_cross_graphs.pop_front();
        }
    }
    for (const auto & pin : pipeline_pins) {
        elastic::wbm_set_pinned(
            &est->wbm, pin.first, pin.second.original);
    }
    pipeline_pins.clear();

    // 一次 graph_compute 收尾：把单步 metrics 落盘
    if (elastic_active && est->metrics_inited) {
        elastic::metrics_record r{};
        r.t_sec = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - est->t0).count();
        r.B_t_mb               = est->bw_inited ? elastic::budget_watcher_get(&est->bw) : 0;
        r.resident_bytes       = est->wbm.resident_bytes;
        r.n_blocks_resident    = est->wbm.n_resident;
        r.token_id             = static_cast<int>(est->current_token);
        r.layer_load_latency_ms = 0.0;
        r.decode_latency_ms    = 0.0;
        r.flash_bytes_read     = est->bytes_reloaded_total;
        elastic::metrics_logger_write(&est->metrics, r);
    }

    return GGML_STATUS_SUCCESS;
}

static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_backend_opencl_device_context * dev_ctx     = (ggml_backend_opencl_device_context *)dev->context;
    ggml_backend_opencl_context *        backend_ctx = dev_ctx->backend_ctx;

    switch (op->op) {
        case GGML_OP_NONE:
            return true;
        case GGML_OP_GET_ROWS:
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                    return true;
                case GGML_TYPE_Q4_0:
#ifdef GGML_OPENCL_SOA_Q
                    // We do not support flattened Q4_0 (and possibly other Q's)
                    return false;
#else // GGML_OPENCL_SOA_Q
                    return true;
#endif // GGML_OPENCL_SOA_Q
                default:
                    return false;
            }
        case GGML_OP_SET_ROWS:
            {
                // TODO: add support
                // ref: https://github.com/ggml-org/llama.cpp/pull/14274
#pragma message("TODO: implement BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL support (https://github.com/ggml-org/llama.cpp/pull/14661)")
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }
                switch (op->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                        return (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                    default:
                        return false;
                }
            }
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                    switch (op->type) {
                        case GGML_TYPE_F16:
                        case GGML_TYPE_F32:
                            return true;
                        default:
                            return false;
                    }
                case GGML_TYPE_F16:
                    switch (op->type) {
                        case GGML_TYPE_F16:
                        case GGML_TYPE_F32:
                            return true;
                        default:
                            return false;
                    }
                default:
                    return false;
            }
        case GGML_OP_SCALE:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_ADD:
            if (op->type == GGML_TYPE_F16) {
                const bool src0_ok = op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32;
                const bool src1_ok = op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32;
                if (src0_ok && src1_ok) {
                    return true;
                }
            }
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
            return (op->src[0]->type == op->src[1]->type) &&
                   (op->src[0]->type == op->type) &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_ADD_ID:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                   return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
                case GGML_UNARY_OP_SIGMOID:
                    return ggml_is_contiguous(op->src[0]);
                case GGML_UNARY_OP_TANH:
                   return (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
                          (op->src[0]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_CLAMP:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SOFT_MAX:
        case GGML_OP_NORM:
            return true;
        case GGML_OP_RMS_NORM:
            return op->ne[0] % 4 == 0 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_REPEAT:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32; // Assuming F32 for now, can be expanded
        case GGML_OP_PAD:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_UPSCALE: {
            ggml_scale_mode mode = (ggml_scale_mode)(ggml_get_op_params_i32(op, 0) & 0xFF);
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   (mode == GGML_SCALE_MODE_NEAREST || mode == GGML_SCALE_MODE_BILINEAR);
        }
        case GGML_OP_CONV_2D:
            return (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16) ||
                   (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
                   (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
        case GGML_OP_CONCAT:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_TIMESTEP_EMBEDDING:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_MUL_MAT:
            if (op->src[0]->type == GGML_TYPE_F16) {
                return true;
            } else if (op->src[0]->type == GGML_TYPE_F32) {
                return op->src[1]->type == GGML_TYPE_F32;
            } else if (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_MXFP4 ||
                       op->src[0]->type == GGML_TYPE_Q6_K) {
                return op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            } else if (op->src[0]->type == GGML_TYPE_Q8_0) {
                return op->src[1]->type == GGML_TYPE_F32;
            }
            return false;
        case GGML_OP_MUL_MAT_ID:
            if (op->src[0]->type == GGML_TYPE_Q4_0 ||
                op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_MXFP4) {
                if (op->src[1]->type == GGML_TYPE_F32) {
                    if (op->src[3] != nullptr && (op->src[3]->type != GGML_TYPE_F32 || !ggml_is_contiguous(op->src[3]))) {
                        return false;
                    }
                    return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
                }
            }
            return false;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return op->ne[3] == 1;
        case GGML_OP_ROPE: {
            const int mode = ((const int32_t *) op->op_params)[2];
            const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
            if (is_mrope && !is_vision) {
                if (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16) {
                    return true;
                }
                return false;
            }
            if (is_vision) {
                if (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16) {
                    return true;
                }
                return false;
            }
            return true;
        }
        case GGML_OP_IM2COL:
            return true;
        case GGML_OP_ARGSORT: {
            cl_kernel kernel = backend_ctx->kernel_argsort_f32_i32;
            int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);

            int cols = 1;
            while (cols < op->ne[0]) {
                cols *= 2;
            }

            return cols <= max_workgroup_size && op->src[0]->type == GGML_TYPE_F32;
        }
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_FLASH_ATTN_EXT:
            {
                const ggml_tensor * q = op->src[0];
                const ggml_tensor * k = op->src[1];
                const ggml_tensor * v = op->src[2];

                const int dk = q->ne[0];
                const int dv = v->ne[0];

                const struct { int dk; int dv; } supported_dims[] = {
                    { 40,  40}, { 64,  64}, { 80,  80}, { 96,  96},
                    {112, 112}, {128, 128}, {192, 128},
                    {192, 192}, {256, 256},
                };

                bool dims_supported = false;
                for (size_t i = 0; i < sizeof(supported_dims)/sizeof(supported_dims[0]); ++i) {
                    if (supported_dims[i].dk == dk && supported_dims[i].dv == dv) {
                        dims_supported = true;
                        break;
                    }
                }
                if (!dims_supported) {
                    return false;
                }

                const bool is_f32_f32 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 &&
                                        v->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
                const bool is_f16_f16 = q->type == GGML_TYPE_F16 && k->type == GGML_TYPE_F16 &&
                                        v->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16;
                const bool is_f32_f16 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
                                        v->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F32;

                return is_f32_f32 || is_f16_f16 || is_f32_f16;
            }
        default:
            return false;
    }
}

// Forward declaration - implementation appears later in the file.
static const char * ggml_backend_opencl_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type);

static ggml_guid_t ggml_backend_opencl_guid() {
    static ggml_guid guid = { 0xde, 0xe0, 0x70, 0xa2, 0x73, 0x4e, 0x4d, 0xbc, 0xb0, 0xc7, 0x4f, 0xd4, 0x6d, 0x4e, 0x90, 0xfe };
    return &guid;
}

static ggml_backend_i ggml_backend_opencl_i = {
    /* .get_name                = */ ggml_backend_opencl_name,
    /* .free                    = */ ggml_backend_opencl_free,
    /* .set_tensor_async        = */ NULL,  /* ggml_backend_opencl_set_tensor_async */
    /* .get_tensor_async        = */ NULL,  /* ggml_backend_opencl_get_tensor_async */
    /* .cpy_tensor_async        = */ NULL,  /* ggml_backend_opencl_cpy_tensor_async */
    /* .synchronize             = */ ggml_backend_opencl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_opencl_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_t ggml_backend_opencl_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_opencl_reg(), 0);
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(dev);

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_opencl_guid(),
        /* .iface   = */ ggml_backend_opencl_i,
        /* .device  = */ dev,
        /* .context = */ backend_ctx
    };

    return backend;
}

bool ggml_backend_is_opencl(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_opencl_name;
}

bool ggml_backend_opencl_get_working_set_state(
        ggml_backend_t backend,
        ggml_backend_opencl_working_set_state * out) {
    if (!out || !ggml_backend_is_opencl(backend)) return false;
    ggml_opencl_elastic_state * state = ggml_opencl_elastic();
    if (!state) return false;
    std::lock_guard<std::mutex> lock(state->moe_cache_mtx);
    out->active_capacity = state->moe_cache_active_capacity;
    out->pending_capacity = state->moe_cache_pending_capacity;
    out->target_capacity = state->moe_cache_plan_capacity;
    out->observed_required_capacity = state->moe_cache_required_high_water;
    out->hits = state->moe_cache_hits;
    out->misses = state->moe_cache_misses;
    out->accesses = out->hits + out->misses;
    out->capacity_changes = state->moe_cache_grow_resizes;
    out->resident_bytes = state->moe_cache_bytes;
    return true;
}

bool ggml_backend_opencl_set_working_set_target(ggml_backend_t backend, int target_capacity) {
    if (!ggml_backend_is_opencl(backend) || target_capacity < -1) return false;
    ggml_opencl_elastic_state * state = ggml_opencl_elastic();
    if (!state) return false;
    std::lock_guard<std::mutex> lock(state->moe_cache_mtx);
    state->moe_cache_plan_capacity = target_capacity;
    state->moe_cache_required_high_water = 0;
    state->moe_cache_pending_capacity = -1;
    state->moe_cache_pending_hits = 0;
    return true;
}

bool ggml_backend_opencl_get_granularity_state(
        ggml_backend_t backend,
        ggml_backend_opencl_granularity_state * out) {
    if (!out || !ggml_backend_is_opencl(backend)) return false;
    ggml_opencl_elastic_state * state = ggml_opencl_elastic();
    if (!state || !state->wbm_inited) return false;
    out->mode = static_cast<int>(state->granularity.mode);
    out->units = state->granularity_units;
    out->nonresident_units =
        state->unit_pipeline_current_missing_units;
    out->pipeline_issued = state->unit_pipeline_issued;
    out->pipeline_ready = state->unit_pipeline_ready;
    out->pipeline_waits = state->unit_pipeline_waits;
    out->pipeline_wait_us = state->unit_pipeline_wait_us;
    out->reloads = state->n_reloads_total;
    out->reload_bytes = state->bytes_reloaded_total;
    uint64_t soa_prepare_calls = 0;
    uint64_t soa_prepare_ok = 0;
    uint64_t soa_prepare_us = 0;
    size_t soa_prepare_bytes = 0;
    elastic::wbmcl_get_soa_prepare_state(
        &state->octx,
        &soa_prepare_calls,
        &soa_prepare_ok,
        &soa_prepare_us,
        &soa_prepare_bytes);
    (void) soa_prepare_ok;
    out->prepare_us = soa_prepare_calls > 0 ?
        soa_prepare_us : state->octx.stage_xform_us;
    out->direct_read_calls = state->octx.direct_read_calls;
    out->direct_read_us = state->octx.direct_read_us;
    out->direct_read_bytes = state->octx.direct_read_bytes;
    out->load_calls =
        state->octx.stage_load_calls +
        state->octx.foreground_direct_read_calls;
    out->load_us =
        state->octx.stage_load_us +
        state->octx.foreground_direct_read_us;
    out->load_bytes =
        state->octx.stage_load_bytes +
        state->octx.foreground_direct_read_bytes;
    out->prepare_calls = soa_prepare_calls > 0 ?
        soa_prepare_calls : state->octx.stage_xform_calls;
    out->prepare_bytes = soa_prepare_calls > 0 ?
        soa_prepare_bytes : state->octx.stage_xform_bytes;
    // Production OpenCL runs intentionally avoid per-kernel event profiling:
    // enabling it changes queue retirement and pipeline overlap.  Keep the
    // phase unavailable instead of manufacturing a residual compute time.
    out->compute_calls = 0;
    out->compute_us = 0;
    out->compute_timing_available = 0;
    out->pipeline_residency_us = 0;
    out->pipeline_unissued_us = 0;
    out->pipeline_stage_us = 0;
    out->pipeline_retire_us = 0;
    out->evict_us = 0;
    out->resident_bytes = state->wbm.resident_bytes;
    return true;
}

bool ggml_backend_opencl_synchronize_elastic_pipeline(
        ggml_backend_t backend) {
    if (!ggml_backend_is_opencl(backend)) return false;
    ggml_opencl_elastic_state * state = ggml_opencl_elastic();
    if (!state || !state->wbm_inited) return false;
    elastic::wbmcl_wait_async_idle(&state->octx);
    // The host PREPARE worker may have enqueued OpenCL work after an earlier
    // generic backend synchronization. Drain that work only at the explicit
    // phase boundary, after the host queues are idle.
    ggml_backend_opencl_synchronize(backend);
    return true;
}

//
// buffer
//
//
// elastic 模式 (env GGML_OPENCL_ELASTIC=1)：
//   只对**权重 buffer** 启用 per-tensor cl_mem；KV / 计算 / 临时 buffer 仍走
//   原 monolithic 一块大 cl_mem 的路径，行为零变化。
//
//   alloc_buffer 时拿不到 tensor 信息，无法判断这个 buffer 是给权重还是 KV。
//   所以 ε 模式下 alloc_buffer 把 buffer 标记为 PENDING（不立刻 alloc 任何
//   cl_mem），延迟到第一次 init_tensor 看 tensor name 才定：
//     - 名字以 ".weight" 结尾 → ELASTIC（per-tensor cl_mem）
//     - 否则                  → MONOLITHIC（lazy 把整块 cl_mem alloc 起来）
//
//   详见 docs/elastic/baseline_external_execution.md 与 runtime/RUNTIME_PATCHES.md §3 H2。
//
static bool ggml_opencl_elastic_enabled() {
    static int cached = -1;
    if (cached == -1) {
        const char *e = std::getenv("GGML_OPENCL_ELASTIC");
        cached = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (cached) {
            GGML_LOG_INFO("ggml_opencl: GGML_OPENCL_ELASTIC=1，权重 buffer 启用 per-tensor cl_mem 模式\n");
        }
    }
    return cached == 1;
}

// 判断 tensor 是不是权重。简单规则：名字以 ".weight" 结尾。
// LLM 权重 (blk.<N>.*.weight, token_embd.weight, output_norm.weight, output.weight)
// 都满足；KV cache 与激活的 tensor 都不以 .weight 结尾。
static bool ggml_opencl_is_weight_tensor(const char *name) {
    if (!name) return false;
    const size_t n = strlen(name);
    const size_t suf = 7;  // strlen(".weight")
    if (n < suf) return false;
    return strcmp(name + n - suf, ".weight") == 0;
}

// 见文件顶部的 ggml_opencl_elastic_state 前置定义（必须在 graph_compute 之前）。

struct ggml_backend_opencl_buffer_context {
    // 三态生命周期：alloc_buffer 进 PENDING；首个 init_tensor 决定升 ELASTIC
    // 或 MONOLITHIC。MONOLITHIC 才会真正 push 一个 cl_mem 到 buffer；ELASTIC
    // 在每次 init_tensor 时 push 一个 per-tensor cl_mem。
    enum mode_t { MODE_MONOLITHIC, MODE_PENDING, MODE_ELASTIC, MODE_HOST_MAPPED };

    // A buffer context can hold multiple cl_mem objects. This is for flattening
    // quantized weights and should be used with GGML_OPENCL_SMALL_ALLOC where
    // each tensor is allocated a separate buffer. When flattening is enabled
    // with small allocation, each tensor is backed by two cl_mem objects (for
    // quants and scales) packed into a backend_opencl_buffer.
    ggml_backend_opencl_buffer_context(cl_mem buf)
        : mode(MODE_MONOLITHIC), pending_size_hint(0), host_ptr(nullptr), host_queue(nullptr), name("OpenCL") {
        buffer.push_back(buf);
    }

    // PENDING 构造：alloc_buffer 把 size 记下来，等 init_tensor 决定升级路径。
    struct pending_tag_t {};
    ggml_backend_opencl_buffer_context(pending_tag_t, size_t size_hint)
        : mode(MODE_PENDING), pending_size_hint(size_hint), host_ptr(nullptr), host_queue(nullptr), name("OpenCL-Pending") {}

    struct host_mapped_tag_t {};
    ggml_backend_opencl_buffer_context(host_mapped_tag_t, cl_mem buf, void * ptr, cl_command_queue queue)
        : mode(MODE_HOST_MAPPED), pending_size_hint(0), host_ptr(ptr), host_queue(queue), name("OpenCL-HostMapped") {
        buffer.push_back(buf);
    }

    ~ggml_backend_opencl_buffer_context() {
        if (mode == MODE_HOST_MAPPED && host_ptr != nullptr && host_queue != nullptr && !buffer.empty() && buffer[0] != nullptr) {
            cl_int err = clEnqueueUnmapMemObject(host_queue, buffer[0], host_ptr, 0, NULL, NULL);
            if (err == CL_SUCCESS) {
                CL_CHECK(clFinish(host_queue));
            }
            host_ptr = nullptr;
        }
        for (size_t i = 0; i < buffer.size(); ++i) {
            cl_mem buf = buffer[i];
            if (!buf) continue;  // 显式 nullptr，跳过
            // ELASTIC 模式下 WBM 可能已经 release 了某些 slot 的 cl_mem。
            // 判定：如果 slot 有 WBM 关联 (wbm_idx_per_slot[i] >= 0)，且 WBM
            // 该 block 当前 backend_handle 与 buf 不一致（已 evict 或 reload 成
            // 新 cl_mem），则 buf 是死指针，跳过释放。
            if (mode == MODE_ELASTIC && i < wbm_idx_per_slot.size()) {
                int wbm_idx = wbm_idx_per_slot[i];
                if (wbm_idx >= 0) {
                    auto *st = ggml_opencl_elastic();
                    if (st->wbm_inited) {
                        const elastic::block_meta *bm = elastic::wbm_get(&st->wbm, wbm_idx);
                        if (bm && (cl_mem)bm->backend_handle != buf) {
                            continue;  // 死指针
                        }
                    }
                }
            }
            CL_CHECK(clReleaseMemObject(buf));
        }
        for (cl_mem im : img) {
            CL_CHECK(clReleaseMemObject(im));
        }

        // Delete all extras to trigger their destructors
        for (ggml_tensor_extra_cl * e : temp_tensor_extras) {
            delete e;
        }
        for (ggml_tensor_extra_cl * e : temp_tensor_extras_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_mxfp4 * e : temp_tensor_extras_mxfp4) {
            delete e;
        }
        for (ggml_tensor_extra_cl_mxfp4 * e : temp_tensor_extras_mxfp4_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q8_0 * e : temp_tensor_extras_q8_0) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q8_0 * e : temp_tensor_extras_q8_0_in_use) {
            delete e;
        }
    }

    ggml_tensor_extra_cl * ggml_opencl_alloc_temp_tensor_extra() {
        ggml_tensor_extra_cl * extra;
        if (temp_tensor_extras.empty()) {
            extra = new ggml_tensor_extra_cl();
        } else {
            extra = temp_tensor_extras.back();
            temp_tensor_extras.pop_back();
        }

        temp_tensor_extras_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q4_0 * ggml_opencl_alloc_temp_tensor_extra_q4_0() {
        ggml_tensor_extra_cl_q4_0 * extra;
        if (temp_tensor_extras_q4_0.empty()) {
            extra = new ggml_tensor_extra_cl_q4_0();
        } else {
            extra = temp_tensor_extras_q4_0.back();
            temp_tensor_extras_q4_0.pop_back();
        }

        temp_tensor_extras_q4_0_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_mxfp4 * ggml_opencl_alloc_temp_tensor_extra_mxfp4() {
        ggml_tensor_extra_cl_mxfp4 * extra;
        if (temp_tensor_extras_mxfp4.empty()) {
            extra = new ggml_tensor_extra_cl_mxfp4();
        } else {
            extra = temp_tensor_extras_mxfp4.back();
            temp_tensor_extras_mxfp4.pop_back();
        }

        temp_tensor_extras_mxfp4_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q8_0 * ggml_opencl_alloc_temp_tensor_extra_q8_0() {
        ggml_tensor_extra_cl_q8_0 * extra;
        if (temp_tensor_extras_q8_0.empty()) {
            extra = new ggml_tensor_extra_cl_q8_0();
        } else {
            extra = temp_tensor_extras_q8_0.back();
            temp_tensor_extras_q8_0.pop_back();
        }

        temp_tensor_extras_q8_0_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    void reset() {
        for (ggml_tensor_extra_cl * e : temp_tensor_extras_in_use) {
            temp_tensor_extras.push_back(e);
        }
        temp_tensor_extras_in_use.clear();

        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0_in_use) {
            temp_tensor_extras_q4_0.push_back(e);
        }
        temp_tensor_extras_q4_0_in_use.clear();

        for (ggml_tensor_extra_cl_mxfp4 * e : temp_tensor_extras_mxfp4_in_use) {
            temp_tensor_extras_mxfp4.push_back(e);
        }
        temp_tensor_extras_mxfp4_in_use.clear();

        for (ggml_tensor_extra_cl_q8_0 * e : temp_tensor_extras_q8_0_in_use) {
            temp_tensor_extras_q8_0.push_back(e);
        }
        temp_tensor_extras_q8_0_in_use.clear();
    }

    // Pools for extras. Available extras are in `temp_tensor_extras`. Extras
    // being used are in `temp_tensor_extras_in_use`. At the first run, new
    // extras get created and put in `in_use`. When the buffer is reset via
    // the `reset` callback, all extras in `in_use` get moved to available extras
    // for reuse.
    std::vector<ggml_tensor_extra_cl *> temp_tensor_extras;
    std::vector<ggml_tensor_extra_cl *> temp_tensor_extras_in_use;
    std::vector<ggml_tensor_extra_cl_q4_0 *> temp_tensor_extras_q4_0;
    std::vector<ggml_tensor_extra_cl_q4_0 *> temp_tensor_extras_q4_0_in_use;
    std::vector<ggml_tensor_extra_cl_mxfp4 *> temp_tensor_extras_mxfp4;
    std::vector<ggml_tensor_extra_cl_mxfp4 *> temp_tensor_extras_mxfp4_in_use;
    std::vector<ggml_tensor_extra_cl_q8_0 *> temp_tensor_extras_q8_0;
    std::vector<ggml_tensor_extra_cl_q8_0 *> temp_tensor_extras_q8_0_in_use;

    // The buffer_context is initially created by ggml_backend_buft_alloc_buffer
    // before any tensor is initialized (at the beginning of alloc_tensor_range).
    // Hence, there is alway a buffer object in this vector. When each tensor is
    // being initialized, this original buffer object will be released if both
    // flattening and small allocation are enabled, and additional buffer
    // objects will be created in init_tensor to represent flattened quantized
    // weights.
    std::vector<cl_mem> buffer;
    // These are image1d_buffer_t objects that wrap around the quants and scales.
    // For Q4_0 quantization, there should be two of them - one for quants and
    // one for scales. They should be populated only when flattening and small
    // allocation are enabled.
    std::vector<cl_mem> img;
    mode_t mode;              // MODE_MONOLITHIC / MODE_PENDING / MODE_ELASTIC
    size_t pending_size_hint; // alloc_buffer 时记录，PENDING → MONOLITHIC 升级时用来真正 alloc
    void * host_ptr;
    cl_command_queue host_queue;
    // elastic 模式下与 buffer 平行的 slot → WBM block_idx 映射；-1 表示该 slot 未注册到 WBM
    std::vector<int> wbm_idx_per_slot;
    std::string name;
};

// 前向声明在 elastic helpers 处。
static void ggml_opencl_elastic_update_ctx_slot(ggml_backend_buffer_t buf,
                                                int slot, cl_mem new_buf) {
    if (!buf || slot < 0) return;
    auto *bctx = (ggml_backend_opencl_buffer_context *) buf->context;
    if (bctx->mode != ggml_backend_opencl_buffer_context::MODE_ELASTIC) return;
    if (static_cast<size_t>(slot) >= bctx->buffer.size()) return;
    bctx->buffer[slot] = new_buf;
}

static void ggml_backend_opencl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_opencl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    if (ctx->mode == ggml_backend_opencl_buffer_context::MODE_HOST_MAPPED) {
        return ctx->host_ptr;
    }
    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer->buft->device);
    return (void *) (uintptr_t) backend_ctx->alignment;
}

static enum ggml_status ggml_backend_opencl_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;

    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer->buft->device);

    // PENDING 升级：第一次见到 tensor 时按 name 决定走 ELASTIC 还是 MONOLITHIC。
    // 后续 init_tensor 沿用本 buffer 已决定的模式。
    if (ctx->mode == ggml_backend_opencl_buffer_context::MODE_PENDING) {
        if (ggml_opencl_is_weight_tensor(tensor->name)) {
            ctx->mode = ggml_backend_opencl_buffer_context::MODE_ELASTIC;
            ctx->name = "OpenCL-Elastic";
            GGML_LOG_INFO("ggml_opencl elastic: buffer 升 ELASTIC（size=%.2f MiB，首 tensor='%s'）\n",
                          ctx->pending_size_hint / 1024.0 / 1024.0, tensor->name);
        } else {
            // 非权重 buffer 退回 monolithic：现在才把整块 cl_mem 真正分配出来。
            cl_int err = CL_SUCCESS;
            cl_mem mem = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE,
                                        ctx->pending_size_hint, NULL, &err);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("ggml_opencl elastic: 升 MONOLITHIC 时 clCreateBuffer %.2f MiB 失败: %d\n",
                               ctx->pending_size_hint / 1024.0 / 1024.0, err);
                return GGML_STATUS_ALLOC_FAILED;
            }
            ctx->buffer.push_back(mem);
            ctx->mode = ggml_backend_opencl_buffer_context::MODE_MONOLITHIC;
            ctx->name = "OpenCL";
            GGML_LOG_INFO("ggml_opencl elastic: buffer 升 MONOLITHIC（size=%.2f MiB，首 tensor='%s'）\n",
                          ctx->pending_size_hint / 1024.0 / 1024.0, tensor->name);
        }
    }

    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);

        ggml_tensor_extra_cl * view_extra = (ggml_tensor_extra_cl *) tensor->view_src->extra;
        GGML_ASSERT(view_extra && "view_extra is nullptr?");

        // Reuse extra of the parent tensor. The offset of this view tensor
        // becomes `extra->offset + view_offs` and needs to be calculated when
        // it is used. This changes is needed because of the change to
        // ggml_alloc.c in https://github.com/ggerganov/llama.cpp/pull/7640.
        // `buffer` passed in here will always be `tensor->buffer`. It is OK
        // to allocate extras from the same buffer context for ordinary
        // intermediate tensors. But for views into kv cache tensors, doing so
        // would mess up the extras used by kv cache.
        // Before #7640, `buffer` is for intermediate tensors, which is always
        // different from that of kv cache tensors.
        //
        // NB: now extra->offset no longer accounts for view_offs.
        // NB: this should not apply to weight tensors (for end-to-end runs, but
        //     may apply for test-backend-ops).
        // FIXME: if any unexpected results are seen, double check the offset -
        // there could be other places that need fix.
        tensor->extra = view_extra;
    } else if (ctx->mode == ggml_backend_opencl_buffer_context::MODE_ELASTIC) {
        // elastic 模式：每个 tensor 一块独立 cl_mem。完全无视 ggml-alloc 算出来
        // 的 tensor->data offset。extra->offset = 0 因为 tensor 独占整块 buffer。
        size_t nbytes = ggml_nbytes(tensor);
        if (nbytes == 0) nbytes = 1;            // clCreateBuffer 不接受 size=0
        cl_int err = CL_SUCCESS;
        cl_mem own_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE,
                                        nbytes, NULL, &err);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("ggml_opencl elastic: clCreateBuffer 失败 tensor '%s' size %zu: %d\n",
                           tensor->name, nbytes, err);
            return GGML_STATUS_ALLOC_FAILED;
        }
        const int slot = static_cast<int>(ctx->buffer.size());
        ctx->buffer.push_back(own_buf);
        ctx->wbm_idx_per_slot.push_back(-1);   // 在 set_tensor 注册时填 WBM block idx

        ggml_tensor_extra_cl * extra = ctx->ggml_opencl_alloc_temp_tensor_extra();
        extra->offset      = 0;
        extra->data_device = own_buf;
        extra->actual_size = ggml_nbytes(tensor);
        extra->ctx_slot    = slot;
        tensor->extra      = extra;
    } else {
        {
            size_t offset = (char *) tensor->data - (char *) ggml_backend_opencl_buffer_get_base(buffer);

            ggml_tensor_extra_cl * extra = ctx->ggml_opencl_alloc_temp_tensor_extra();
            extra->offset = offset;
            extra->data_device = ctx->buffer[0];
            extra->actual_size = ggml_nbytes(tensor);

            tensor->extra = extra;
        }
    }
    return GGML_STATUS_SUCCESS;
}

// The optimized gemm and gemv kernels are used for large matrices without batch.
// tensor is the quantized weights matrix.
static bool ggml_opencl_elastic_evict_during_load_enabled();
static void ggml_opencl_elastic_evict_to_initial_budget(
        elastic::weight_buffer_manager * wbm,
        elastic::wbm_opencl_ctx * octx,
        size_t target_bytes,
        int exclude_idx);

// 公共注册逻辑：SOA tensor (q4_0/q8_0/mxfp4) 注册到 WBM + 绑定 evict/reload 回调。
// 三个 SOA extra 类型有同名字段 wbm_idx / ctx_slot / parent_buffer，模板就够；
// 不同的转换 kernel / sub-buffer 字段在调用方的闭包里处理。
template <typename ExtraT>
static void ggml_opencl_elastic_register_soa(
        ggml_backend_buffer_t buffer,
        ggml_tensor *tensor,
        ExtraT *extra,
        cl_mem parent_buffer,
        int ctx_slot,
        const void *host_ptr,
        size_t nbytes,
        cl_context cl_ctx,
        cl_command_queue queue,
        std::function<int()> evict_fn,
        std::function<int()> reload_fn) {
    ggml_backend_opencl_buffer_context *bctx =
        (ggml_backend_opencl_buffer_context *) buffer->context;
    if (bctx->mode != ggml_backend_opencl_buffer_context::MODE_ELASTIC) return;
    if (!ggml_opencl_is_weight_tensor(tensor->name)) return;
    if (tensor->view_src != nullptr) return;

    ggml_opencl_elastic_lazy_init(cl_ctx, queue);
    auto *s = ggml_opencl_elastic();
    if (!s->wbm_inited) return;

    extra->parent_buffer = parent_buffer;
    extra->ctx_slot      = ctx_slot;
    int idx = elastic::wbm_add_block(&s->wbm, const_cast<void*>(host_ptr), nbytes);
    if (idx < 0) return;
    extra->wbm_idx = idx;
    elastic::wbm_mark_resident(&s->wbm, idx, static_cast<void*>(parent_buffer));
    elastic::wbm_touch(&s->wbm, idx, s->current_token);
    if (ctx_slot >= 0 && (size_t)ctx_slot < bctx->wbm_idx_per_slot.size()) {
        bctx->wbm_idx_per_slot[ctx_slot] = idx;
    }
    elastic::wbmcl_register_soa(&s->octx, idx, std::move(evict_fn), std::move(reload_fn));

    // Runtime scheduler 用的 name → idx 索引
    if (tensor && tensor->name[0]) {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        s->name_to_wbm[tensor->name] = idx;
        s->name_to_wbms[tensor->name].push_back(idx);
        s->wbm_to_name[idx] = tensor->name;
        if (tensor->type == GGML_TYPE_Q4_0 && ggml_opencl_is_moe_expert_weight(tensor)) {
            s->moe_q4_packed_indices.insert(idx);
        }
    }
    if (tensor && tensor->name[0]) {
        llama_weight_runtime_mark_resident(tensor->name, LLAMA_WEIGHT_RUNTIME_GPU);
    }
    opencl_sched_register_once();

    // Pin 策略
    static const std::string s_pin_policy = []() {
        const char *p = std::getenv("GGML_ELASTIC_PIN");
        return std::string(p ? p : "");
    }();
    if (!s_pin_policy.empty()) {
        const std::string suffix = ggml_opencl_tensor_suffix(tensor->name);
        auto contains = [&](const char *tok) {
            if (s_pin_policy == "all") return true;
            const std::string delimited = "," + s_pin_policy + ",";
            return delimited.find(
                "," + std::string(tok) + ",") != std::string::npos;
        };
        bool should_pin = false;
        if (contains("norm") && (suffix == "attn_norm" || suffix == "ffn_norm" || suffix == "output_norm")) should_pin = true;
        if (contains("k") && suffix == "attn_k") should_pin = true;
        if (contains("v") && suffix == "attn_v") should_pin = true;
        if (contains("q") && suffix == "attn_q") should_pin = true;
        if (contains("o") && suffix == "attn_output") should_pin = true;
        if (contains("token_embd") && suffix == "token_embd") should_pin = true;
        if (contains("output") && suffix == "output") should_pin = true;
        if (should_pin) {
            elastic::wbm_set_pinned(&s->wbm, idx, true);
            if (suffix == "token_embd" || suffix == "output") {
                GGML_LOG_INFO(
                    "ggml_opencl elastic: pin %s inside weight budget "
                    "(%.2f MiB)\n",
                    suffix.c_str(), nbytes / 1024.0 / 1024.0);
            }
        }
    }
    static const bool s_embed_out = []() {
        const char *e = std::getenv("GGML_ELASTIC_EMBED_OUTSIDE_BUDGET");
        return e && *e && *e != '0';
    }();
    if (s_embed_out && ggml_opencl_tensor_suffix(tensor->name) == "token_embd") {
        elastic::wbm_set_pinned(&s->wbm, idx, true);
        s->static_target_bytes += nbytes;
        s->extra_target_bytes  += nbytes;   // dynamic 路径用 (跟 static 平行累加)
    }
    static const bool s_pin_unplanned_output = []() {
        const char *e = std::getenv("GGML_ELASTIC_PIN_UNPLANNED_OUTPUT");
        return !(e && *e == "0"[0]);
    }();
    static const bool s_unplanned_output_counts_budget = []() {
        const char *e = std::getenv("GGML_ELASTIC_PIN_UNPLANNED_OUTPUT_COUNTS_BUDGET");
        return e && *e && *e != '0';
    }();
    if (s_pin_unplanned_output && ggml_opencl_tensor_suffix(tensor->name) == "output") {
        elastic::wbm_set_pinned(&s->wbm, idx, true);
        if (!s_unplanned_output_counts_budget) {
            s->static_target_bytes += nbytes;
            s->extra_target_bytes  += nbytes;
            GGML_LOG_INFO("ggml_opencl elastic: pin unplanned output (%zu MB) outside budget -> target=%zu MB\n",
                          nbytes / 1024 / 1024, s->static_target_bytes / 1024 / 1024);
        } else {
            GGML_LOG_INFO("ggml_opencl elastic: pin unplanned output (%zu MB) inside budget -> target=%zu MB\n",
                          nbytes / 1024 / 1024, s->static_target_bytes / 1024 / 1024);
        }
    }

    if (ggml_opencl_elastic_evict_during_load_enabled()) {
        ggml_opencl_elastic_evict_to_initial_budget(&s->wbm, &s->octx, s->static_target_bytes, -1);
    }
}

// 公共 reload helper：pool 命中复用 parent / 否则 alloc。
// 调用方负责后续 sub-buffer 创建 + convert kernel + mark_resident。
// SOA parent ownership diagnostics live in wbm_opencl_ctx. Keeping the
// container in runtime state avoids a static-destruction race with the async
// layout worker at process exit.

static bool ggml_opencl_elastic_evict_during_load_enabled() {
    const char * e = std::getenv("GGML_ELASTIC_EVICT_DURING_LOAD");
    return !(e && *e == '0');
}

static void ggml_opencl_elastic_evict_to_initial_budget(
        elastic::weight_buffer_manager * wbm,
        elastic::wbm_opencl_ctx * octx,
        size_t target_bytes,
        int exclude_idx) {
    if (!wbm || !octx || target_bytes == 0) return;
    if (wbm->resident_bytes <= target_bytes) return;

    std::vector<int> victims;
    const int n = elastic::wbm_evict_to_byte_budget(wbm, target_bytes, exclude_idx, &victims);
    if (n <= 0) return;
    const int released = elastic::wbmcl_evict_batch(octx, victims.data(), n);
    auto * s = ggml_opencl_elastic();
    if (s) {
        s->n_evicts_total += released;
        for (int i = 0; i < released && i < n; ++i) {
            const std::string name = opencl_name_for_idx(s, victims[i]);
            if (!name.empty()) {
                llama_weight_runtime_mark_evicted(name.c_str(), LLAMA_WEIGHT_RUNTIME_GPU);
            }
        }
    }
}

static cl_mem ggml_opencl_elastic_alloc_or_pool_parent(
        elastic::wbm_opencl_ctx *octx, cl_context cl_ctx, size_t nbytes) {
    if (octx->retain_cl_mem) {
        auto it = octx->retained_buffers_by_size.find(nbytes);
        if (it != octx->retained_buffers_by_size.end() && !it->second.empty()) {
            octx->parent_pool_hit++;
            cl_mem cached = static_cast<cl_mem>(it->second.back());
            it->second.pop_back();
            for (auto lit = octx->retain_order_sizes.rbegin();
                 lit != octx->retain_order_sizes.rend(); ++lit) {
                if (*lit == nbytes) {
                    octx->retain_order_sizes.erase(std::next(lit).base());
                    break;
                }
            }
            octx->cached_bytes -= std::min(octx->cached_bytes, nbytes);
            return cached;
        }
    }
    octx->parent_pool_miss++;
    cl_int err = CL_SUCCESS;
    auto t0 = std::chrono::steady_clock::now();
    cl_mem buf = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, nbytes, nullptr, &err);
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
    octx->parent_create_calls++;
    octx->parent_create_us += (uint64_t) dt;
    octx->parent_create_bytes += nbytes;
    if (err != CL_SUCCESS) return nullptr;
    octx->n_creates += 1;
    return buf;
}

// 公共 evict helper：parent 释放或入 pool；同时调 mark_evicted。
// 调用方负责 sub-buffer 释放 (通常调 extra->reset())。
static void ggml_opencl_elastic_evict_parent_pool_or_release(
        elastic::wbm_opencl_ctx *octx, cl_mem parent, size_t nbytes, int idx) {
    if (octx->retain_cl_mem && parent) {
        if (octx->cache_byte_limit > 0) {
            while (octx->cached_bytes + nbytes > octx->cache_byte_limit &&
                   !octx->retain_order_sizes.empty()) {
                size_t old_sz = octx->retain_order_sizes.front();
                octx->retain_order_sizes.pop_front();
                auto pit = octx->retained_buffers_by_size.find(old_sz);
                if (pit == octx->retained_buffers_by_size.end() || pit->second.empty()) continue;
                cl_mem old_buf = static_cast<cl_mem>(pit->second.back());
                pit->second.pop_back();
                clReleaseMemObject(old_buf);
                octx->n_releases += 1;
                octx->cached_bytes -= std::min(octx->cached_bytes, old_sz);
            }
        }
        octx->retained_buffers_by_size[nbytes].push_back(static_cast<void*>(parent));
        octx->retain_order_sizes.push_back(nbytes);
        octx->cached_bytes += nbytes;
    } else if (parent) {
        clReleaseMemObject(parent);
        octx->n_releases += 1;
    }
    octx->bytes_evicted_total += nbytes;
    elastic::wbm_mark_evicted(octx->wbm, idx);
}

// SOA staging buffer pool：多个 slot 轮转，避免所有 reload 被一个 staging buffer 串行化。
static cl_mem ggml_opencl_elastic_get_staging(
        elastic::wbm_opencl_ctx *octx, cl_context cl_ctx, size_t nbytes) {
    static const size_t s_slots = []() {
        const char *e = std::getenv("GGML_ELASTIC_SOA_STAGING_SLOTS");
        long v = e && *e ? std::atol(e) : 4;
        if (v < 1) v = 1;
        if (v > 16) v = 16;
        return (size_t) v;
    }();
    if (octx->soa_staging_slots.size() != s_slots) {
        octx->soa_staging_slots.assign(s_slots, nullptr);
        octx->soa_staging_slot_capacity.assign(s_slots, 0);
        octx->soa_staging_slot_last_use_ev.assign(s_slots, nullptr);
        octx->soa_staging_next_slot = 0;
        octx->soa_staging_current_slot = 0;
    }
    const size_t slot = octx->soa_staging_next_slot++ % s_slots;
    octx->soa_staging_current_slot = slot;
    if (octx->soa_staging_slot_capacity[slot] >= nbytes) {
        return octx->soa_staging_slots[slot];
    }
    if (octx->soa_staging_slot_last_use_ev[slot]) {
        clWaitForEvents(1, &octx->soa_staging_slot_last_use_ev[slot]);
        clReleaseEvent(octx->soa_staging_slot_last_use_ev[slot]);
        octx->soa_staging_slot_last_use_ev[slot] = nullptr;
    }
    if (octx->soa_staging_slots[slot]) {
        clReleaseMemObject(octx->soa_staging_slots[slot]);
        octx->n_releases += 1;
    }
    size_t new_cap = octx->soa_staging_slot_capacity[slot] ? octx->soa_staging_slot_capacity[slot] * 2 : nbytes;
    while (new_cap < nbytes) new_cap *= 2;
    cl_int err = CL_SUCCESS;
    octx->soa_staging_slots[slot] = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, new_cap, nullptr, &err);
    if (err != CL_SUCCESS) {
        octx->soa_staging_slots[slot] = nullptr;
        octx->soa_staging_slot_capacity[slot] = 0;
        return nullptr;
    }
    octx->soa_staging_slot_capacity[slot] = new_cap;
    octx->n_creates += 1;
    return octx->soa_staging_slots[slot];
}

// Q4_0 Adreno transpose helper：被 set_tensor 首装路径 & elastic reload 路径共用。
// `sync_each` = true：每个 kernel 后 clWaitForEvents（set_tensor 首装路径用，
//   要等内容写完才能 release sub-buffer）。
// `sync_each` = false：纯 enqueue，依赖 in-order queue 串行——elastic reload 用，
//   外层 ensure_resident 在 matmul 入队前会做一次 sync（每 op 之前）。
//   省掉 6 个 clWaitForEvents 是 tight budget 的主要加速。
// 输入：extra->q + extra->d 已经由 convert kernel 写入（generic SOA layout）。
// 输出：原位转置（4 image1d + 2 transpose kernel + 2 copy），extra->q/d 仍是同
// sub-buffer，但内容变成 Adreno mat-mul kernel 期待的 transpose layout。
// M = tensor->ne[1], K = tensor->ne[0]。需 K%32==0 && M%4==0。
static int ggml_opencl_run_q4_0_adreno_transpose(
        ggml_backend_opencl_context *backend_ctx,
        cl_command_queue queue,
        ggml_tensor_extra_cl_q4_0 *extra,
        int M, int K,
        bool sync_each = true,
        elastic::wbm_opencl_ctx *octx = nullptr) {
    cl_int err = CL_SUCCESS;
    cl_context context = backend_ctx->context;

    // === image / sub-buffer POOL (GGML_ELASTIC_IMG_POOL=1, 默认关) ===
    // 目的: 消除每次 reload 的 clCreateImage/clCreateSubBuffer 创建开销(实测 host-side
    // 15-37ms/call, 主导 reload 时间), 这样测出的是【layout 转换 kernel 的真实代价】而非
    // 创建代价. qT/dT 是 fixed A_q/s_d_max 的 sub-buffer(按 size 复用); q/d input image
    // 绑定轮转的 extra->q/d(按 cl_mem 复用). image width 由 M*K 对称决定 → gate/down 同
    // size 可共用. 仅 K_tile_trans 启用 pool; 依赖 RETAIN_MB 足够大使 SOA pool 不在 run 内
    // FIFO 释放 extra->q/d(否则 cached image 悬空). 默认关 → 行为与原来完全一致(零风险).
    static const bool s_img_pool = []() {
        const char *e = std::getenv("GGML_ELASTIC_IMG_POOL");
        return e && *e && *e != '0';
    }();
    static std::unordered_map<size_t, cl_mem> s_qT_sub, s_dT_sub, s_qT_img, s_dT_img;
    static std::unordered_map<cl_mem, cl_mem>  s_q_img, s_d_img;

    size_t q_size_bytes = (size_t)K * M / 8 * sizeof(float);
    bool   K_tile_trans = ((K / 32) % 4 == 0);
    size_t d_size_bytes = (size_t)M * (K / 32) * 2;
    const bool pool = s_img_pool && K_tile_trans;

    cl_image_format img_fmt_1d = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc img_desc_1d;
    cl_mem qT_d = nullptr, dT_d = nullptr;
    cl_mem q_d_image1D = nullptr, d_d_image1D = nullptr, qT_d_image1D = nullptr, dT_d_image1D = nullptr;

    auto mk_sub = [&](cl_mem parent, size_t sz) -> cl_mem {
        cl_buffer_region r; r.origin = 0; r.size = sz;
        return clCreateSubBuffer(parent, 0, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
    };
    auto mk_img = [&](cl_mem buf, cl_image_format fmt, size_t width) -> cl_mem {
        cl_image_desc d; memset(&d, 0, sizeof(d));
        d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER; d.image_width = width; d.buffer = buf;
        return clCreateImage(context, 0, &fmt, &d, NULL, &err);
    };

    // qT_d sub-buffer (A_q_d_max, region[0,q_size]) — pool 按 q_size 复用
    if (pool && s_qT_sub.count(q_size_bytes)) qT_d = s_qT_sub[q_size_bytes];
    else { qT_d = mk_sub(backend_ctx->A_q_d_max, q_size_bytes); if (err != CL_SUCCESS) return -1; if (pool) s_qT_sub[q_size_bytes] = qT_d; }
    // dT_d sub-buffer (A_s_d_max, region[0,d_size]) — pool 按 d_size 复用
    if (pool && s_dT_sub.count(d_size_bytes)) dT_d = s_dT_sub[d_size_bytes];
    else { dT_d = mk_sub(backend_ctx->A_s_d_max, d_size_bytes); if (err != CL_SUCCESS) { if (!pool) clReleaseMemObject(qT_d); return -2; } if (pool) s_dT_sub[d_size_bytes] = dT_d; }

    // q_d_image1D ← extra->q (输入, 轮转 buffer) — pool 按 cl_mem 复用
    if (pool && s_q_img.count(extra->q)) q_d_image1D = s_q_img[extra->q];
    else { q_d_image1D = mk_img(extra->q, img_fmt_1d, (size_t)M * K / 4 / 4); if (err != CL_SUCCESS) { if (!pool){clReleaseMemObject(qT_d);clReleaseMemObject(dT_d);} return -3; } if (pool) s_q_img[extra->q] = q_d_image1D; }
    // qT_d_image1D ← qT_d — pool 按 q_size 复用
    if (pool && s_qT_img.count(q_size_bytes)) qT_d_image1D = s_qT_img[q_size_bytes];
    else { qT_d_image1D = mk_img(qT_d, img_fmt_1d, (size_t)M * K / 4 / 4); if (err != CL_SUCCESS) { if (!pool){clReleaseMemObject(q_d_image1D);clReleaseMemObject(qT_d);clReleaseMemObject(dT_d);} return -4; } if (pool) s_qT_img[q_size_bytes] = qT_d_image1D; }
    // d_d_image1D ← extra->d — pool 按 cl_mem 复用
    cl_image_format fmt_d = K_tile_trans ? (cl_image_format){ CL_RGBA, CL_HALF_FLOAT } : (cl_image_format){ CL_R, CL_HALF_FLOAT };
    size_t w_d_in = K_tile_trans ? (size_t)M * K / 32 / 4 : (size_t)M * K / 32;
    if (pool && s_d_img.count(extra->d)) d_d_image1D = s_d_img[extra->d];
    else { d_d_image1D = mk_img(extra->d, fmt_d, w_d_in); if (err != CL_SUCCESS) { if (!pool){clReleaseMemObject(qT_d_image1D);clReleaseMemObject(q_d_image1D);clReleaseMemObject(qT_d);clReleaseMemObject(dT_d);} return -5; } if (pool) s_d_img[extra->d] = d_d_image1D; }
    // dT_d_image1D ← dT_d — pool 按 d_size 复用
    if (pool && s_dT_img.count(d_size_bytes)) dT_d_image1D = s_dT_img[d_size_bytes];
    else { dT_d_image1D = mk_img(dT_d, img_fmt_1d, (size_t)M * K / 32 / 4); if (err != CL_SUCCESS) { if (!pool){clReleaseMemObject(d_d_image1D);clReleaseMemObject(qT_d_image1D);clReleaseMemObject(q_d_image1D);clReleaseMemObject(qT_d);clReleaseMemObject(dT_d);} return -6; } if (pool) s_dT_img[d_size_bytes] = dT_d_image1D; }
    (void)img_desc_1d;

    cl_event evt;
    // weights transpose
    int height_q = M / 4;
    int width_q  = K / 4 / 4;
    cl_kernel kernel = backend_ctx->kernel_transpose_16;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &q_d_image1D));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &qT_d_image1D));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_q));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_q));
    size_t local_size_q[3]  = {4, 16, 1};
    size_t global_size_q[3] = {(size_t)width_q, (size_t)height_q, 1};
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size_q, local_size_q, 0, NULL, &evt));
    elastic::wbmcl_record_device_event(octx, evt, elastic::wbmcl_device_event_kind::XFORM_TRANSPOSE, q_size_bytes);
    if (sync_each) { CL_CHECK(clWaitForEvents(1, &evt)); }
    clReleaseEvent(evt);

    // scales transpose
    int height_s = M / 4;
    int width_s  = K / 32 / 4;
    kernel = backend_ctx->kernel_transpose_16;
    if (!K_tile_trans) {
        kernel  = backend_ctx->kernel_transpose_16_4x1;
        width_s = K / 32;
    }
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_d_image1D));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dT_d_image1D));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &height_s));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &width_s));
    size_t local_size_s[3]  = {4, 16, 1};
    size_t global_size_s[3] = {(size_t)width_s, (size_t)height_s, 1};
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size_s, local_size_s, 0, NULL, &evt));
    elastic::wbmcl_record_device_event(octx, evt, elastic::wbmcl_device_event_kind::XFORM_TRANSPOSE, d_size_bytes);
    if (sync_each) { CL_CHECK(clWaitForEvents(1, &evt)); }
    clReleaseEvent(evt);

    // copy transposed contents back into extra->q / extra->d
    CL_CHECK(clEnqueueCopyBuffer(queue, qT_d, extra->q, 0, 0, q_size_bytes, 0, NULL, &evt));
    elastic::wbmcl_record_device_event(octx, evt, elastic::wbmcl_device_event_kind::XFORM_COPY, q_size_bytes);
    if (sync_each) { CL_CHECK(clWaitForEvents(1, &evt)); }
    clReleaseEvent(evt);
    CL_CHECK(clEnqueueCopyBuffer(queue, dT_d, extra->d, 0, 0, d_size_bytes, 0, NULL, &evt));
    elastic::wbmcl_record_device_event(octx, evt, elastic::wbmcl_device_event_kind::XFORM_COPY, d_size_bytes);
    if (sync_each) { CL_CHECK(clWaitForEvents(1, &evt)); }
    clReleaseEvent(evt);

    // 异步路径下：release 只减引用，driver 等 queue 用完才真销毁，安全。
    // pool 模式下这些对象缓存复用, 不释放(下次 reload 直接命中, 省 clCreateImage/SubBuffer).
    if (!pool) {
        clReleaseMemObject(qT_d);
        clReleaseMemObject(dT_d);
        clReleaseMemObject(q_d_image1D);
        clReleaseMemObject(d_d_image1D);
        clReleaseMemObject(qT_d_image1D);
        clReleaseMemObject(dT_d_image1D);
    }
    return 0;
}

inline bool use_adreno_kernels(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    static const bool s_disable_adreno_kernels = []() {
        const char *e = std::getenv("GGML_OPENCL_DISABLE_ADRENO_KERNELS");
        return e && *e && *e != '0';
    }();
    if (s_disable_adreno_kernels) {
        return false;
    }

    // GGML_ELASTIC_NO_TRANSPOSE=1：elastic 模式下跳过 Adreno transpose 路径。
    // 对 1B 这种小模型 decode 是 mat-vec, transpose 不是优势 (实测 1B-Q4 generic
    // 比 Adreno transpose 还快 ~30%), 但 evict/reload 时 transpose dance 多 6
    // 个 kernel/copy enqueue × ~0.5 ms = 大头. 关掉后 reload 路径只剩 write+
    // convert (3 enqueue), 8B 也能省 ~50% reload 开销 (代价是 loose 慢 ~30%).
    static const bool s_no_transpose = []() {
        const char *e = std::getenv("GGML_ELASTIC_NO_TRANSPOSE");
        return e && *e && *e != '0';
    }();
    static const bool s_elastic_on = []() {
        const char *e = std::getenv("GGML_OPENCL_ELASTIC");
        return e && *e && *e != '0';
    }();
    if (s_elastic_on && s_no_transpose) return false;
    int64_t threshold_ne0 = 512;
    int64_t threshold_ne1 = 512;
    if (!backend_ctx->adreno_cl_compiler_version.newer_than_or_same(E031, 38, 11, 0) &&
         backend_ctx->adreno_cl_compiler_version.type != DX) {
        threshold_ne0 = 128;
        threshold_ne1 = 128;
    }
    return tensor->ne[0] >= threshold_ne0 && tensor->ne[1] >= threshold_ne1 &&
            tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

inline bool use_adreno_moe_kernels(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    GGML_UNUSED(backend_ctx);
    static const bool s_disable_adreno_kernels = []() {
        const char *e = std::getenv("GGML_OPENCL_DISABLE_ADRENO_KERNELS");
        return e && *e && *e != '0';
    }();
    if (s_disable_adreno_kernels) {
        return false;
    }

    int ne01 = tensor->ne[1];
    return ((strstr(tensor->name, "ffn") != NULL) || (strstr(tensor->name, "as") != NULL)) && (ne01 % 64 == 0);
}

static void ggml_backend_opencl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer->buft->device);

    cl_context context = backend_ctx->context;
    cl_command_queue queue = backend_ctx->queue;

#ifdef GGML_OPENCL_SOA_Q
    // We separate the quantized bits and scale from block_q4_0 by using an
    // additional kernel, where each thread handles a block. We first read the
    // original weights into a temporary buffer, then create two separate
    // buffers for quantized bits and scales, which are then populated by the
    // conversion kernel.
    if (tensor->type == GGML_TYPE_Q4_0) {
        // Tensors should have been preallocated, therefore they should
        // already have ggml_tensor_extra_cl as extra.
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // CUT is materialized here, on the real model loading path.  Each row
        // partition receives its own OpenCL parent, SOA conversion, WBM block,
        // and evict/reload callback.  The recursive call reuses the exact Q4_0
        // preparation path below; the guard prevents a part being split again.
        static thread_local bool s_registering_cut_part = false;
        const elastic::granularity_config granularity = elastic::granularity_from_env();
        auto * cut_bctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        const bool cut_candidate =
            !s_registering_cut_part &&
            (elastic::granularity_dynamic_prepare_cut() ||
             (granularity.explicitly_enabled &&
              granularity.mode == elastic::granularity_mode::CUT)) &&
            cut_bctx->mode == ggml_backend_opencl_buffer_context::MODE_ELASTIC &&
            offset == 0 && size == ggml_nbytes(tensor) && tensor->view_src == nullptr &&
            tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
            ggml_opencl_is_weight_tensor(tensor->name) &&
            ggml_opencl_tensor_suffix(tensor->name) != "token_embd";
        if (cut_candidate) {
            const auto rows = elastic::granularity_partition_rows(
                tensor->ne[1], granularity.cut_parts, 64);
            if (!rows.empty()) {
                const size_t row_bytes = tensor->nb[1];
                const int original_slot = extra_orig->ctx_slot;
                cl_mem original_parent = extra_orig->data_device;
                if (original_slot >= 0 && (size_t) original_slot < cut_bctx->buffer.size()) {
                    cut_bctx->buffer[original_slot] = nullptr;
                }
                extra_orig->reset();
                if (original_parent) CL_CHECK(clReleaseMemObject(original_parent));

                std::vector<elastic_q4_cut_part> cut_parts;
                cut_parts.reserve(rows.size());
                for (const auto & row : rows) {
                    const size_t byte_offset = (size_t) row.row_start * row_bytes;
                    const size_t part_bytes = (size_t) row.row_count * row_bytes;
                    cl_int part_err = CL_SUCCESS;
                    cl_mem part_parent = clCreateBuffer(
                        context, CL_MEM_READ_WRITE, part_bytes, nullptr, &part_err);
                    CL_CHECK(part_err);

                    const int part_slot = (int) cut_bctx->buffer.size();
                    cut_bctx->buffer.push_back(part_parent);
                    cut_bctx->wbm_idx_per_slot.push_back(-1);
                    ggml_tensor_extra_cl * part_orig =
                        cut_bctx->ggml_opencl_alloc_temp_tensor_extra();
                    part_orig->data_device = part_parent;
                    part_orig->offset = 0;
                    part_orig->actual_size = part_bytes;
                    part_orig->ctx_slot = part_slot;

                    ggml_tensor part_tensor = *tensor;
                    part_tensor.ne[1] = row.row_count;
                    part_tensor.ne[2] = 1;
                    part_tensor.ne[3] = 1;
                    part_tensor.nb[2] = part_tensor.nb[1] * (size_t) part_tensor.ne[1];
                    part_tensor.nb[3] = part_tensor.nb[2];
                    part_tensor.view_src = nullptr;
                    part_tensor.view_offs = 0;
                    part_tensor.data = const_cast<unsigned char *>(
                        static_cast<const unsigned char *>(data) + byte_offset);
                    part_tensor.extra = part_orig;

                    s_registering_cut_part = true;
                    ggml_backend_opencl_buffer_set_tensor(
                        buffer, &part_tensor, part_tensor.data, 0, part_bytes);
                    s_registering_cut_part = false;

                    auto * part_extra =
                        (ggml_tensor_extra_cl_q4_0 *) part_tensor.extra;
                    GGML_ASSERT(part_extra && part_extra->wbm_idx >= 0);
                    cut_parts.push_back({row.row_start, row.row_count,
                                         byte_offset, part_bytes,
                                         part_extra, part_extra->wbm_idx});
                }

                auto * state = ggml_opencl_elastic();
                state->q4_cut_parts[tensor] = cut_parts;
                state->granularity_cut_tensors += 1;
                state->granularity_cut_parts += cut_parts.size();
                tensor->extra = cut_parts.front().extra;
                return;
            }
        }

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q4_0 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q4_0();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(
            queue, data_device, CL_TRUE, 0,
            ggml_nbytes(tensor), data, 0, NULL, NULL));

        // We consider the specified offset arg as always, although For weights
        // the offset arg should be 0 (we do not assert this).
        //GGML_ASSERT(offset == 0);

        // We create subbuffers from the original tensor buffer for scales and
        // quants - i.e., scales and quants are aliases into the buffer obejct
        // that backs the original tensor. This is a cleaner way to adapt to the
        // new memory management.
        // In the old code, we allocate new buffers for scales and quants
        // respectively, which could still be done but would result in double
        // allocation; properly deallocating the preallocated buffer that backs
        // the tensors is tricky and would leak the backend specific information
        // into the general backend code.
        // Does this create misaligned subbuffers (alignment is 1024) in certain
        // cases ?
        cl_buffer_region region;

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

        //cl_kernel kernel = backend_ctx->kernel_convert_block_q4_0;
    #ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->kernel_convert_block_q4_0;

        // The optimized kernels need weights in natural order, so unshuffle.
        if (use_adreno_kernels(backend_ctx, tensor)) {
            kernel = backend_ctx->kernel_convert_block_q4_0_noshuffle;
        }
    #else
        cl_kernel kernel = backend_ctx->kernel_convert_block_q4_0;
    #endif // GGML_OPENCL_USE_ADRENO_KERNELS
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;

        // transpose the weights and scales
    #ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Only do transpose for large, non batched matrix
        if (use_adreno_kernels(backend_ctx, tensor)) {
            int M = tensor->ne[1];   // ne01
            int K = tensor->ne[0];   // ne00
            GGML_ASSERT(K % 32 == 0);
            GGML_ASSERT(M % 4 == 0);
            int rc = ggml_opencl_run_q4_0_adreno_transpose(backend_ctx, queue, extra, M, K);
            if (rc != 0) {
                GGML_LOG_ERROR("ggml_opencl: q4_0 Adreno transpose failed rc=%d for '%s'\n", rc, tensor->name);
            }
        }
    #endif // GGML_OPENCL_USE_ADRENO_KERNELS

        // ───── elastic SOA register (Q4_0 — 含 Adreno transpose) ─────────────
        // Adreno fast path 启用时 (use_adreno_kernels = true)，set_tensor 已经
        // 跑了 transpose 把 q/d 内容重排成 transpose layout；reload 时要重做：
        // 1. 跑 convert kernel (raw → q/d, Adreno 路径用 noshuffle 变种)
        // 2. 跑 Adreno transpose helper (q/d 就地转置)
        // 不走 Adreno (ne[0]/ne[1] < 512) 只跑 convert。
        if (auto *bctx = (ggml_backend_opencl_buffer_context *) buffer->context;
            bctx->mode == ggml_backend_opencl_buffer_context::MODE_ELASTIC &&
            offset == 0 && size == ggml_nbytes(tensor)) {
            const void *host_ptr        = data;
            size_t nbytes               = ggml_nbytes(tensor);
            const size_t cap_n_blocks   = (size_t)ggml_nelements(tensor) / ggml_blck_size(tensor->type);
            const size_t cap_size_q     = size_q;
            const size_t cap_size_d     = size_d;
            const bool need_transpose   = use_adreno_kernels(backend_ctx, tensor);
            cl_kernel cap_kernel        = need_transpose
                                          ? backend_ctx->kernel_convert_block_q4_0_noshuffle
                                          : backend_ctx->kernel_convert_block_q4_0;
            const int cap_M = tensor->ne[1];
            const int cap_K = tensor->ne[0];
            elastic::wbm_opencl_ctx *octx = &ggml_opencl_elastic()->octx;
            ggml_tensor_extra_cl_q4_0 *cap_extra = extra;
            ggml_backend_opencl_buffer_context *cap_bctx = bctx;
            ggml_backend_opencl_context *cap_backend_ctx = backend_ctx;
            cl_context cap_ctx = context;
            cl_command_queue cap_q = queue;

            auto evict_fn = [octx, cap_extra, cap_q, nbytes]() -> int {
                // Pool ownership can change concurrently with the async layout
                // worker. Protect only pool bookkeeping; do not serialize
                // eviction against the later transfer/convert work.
                std::lock_guard<std::mutex> pool_guard(octx->soa_pool_mtx);
                int idx = cap_extra->wbm_idx;
                if (cap_extra->reset_owned_images()) {
                    ggml_opencl_elastic()->cut_dual_image_releases++;
                }
                // SOA pool: parent + d + q 一起入池，不 reset extra (sub-buffer 还活着)
                if (octx->retain_cl_mem && cap_extra->parent_buffer) {
                    // SOA triples and ordinary cl_mem entries share one byte
                    // cap/FIFO, so trimming must use the unified helper.
                    elastic::wbmcl_retain_pool_trim_locked(octx, nbytes);
                    elastic::soa_pool_entry e;
                    e.parent = cap_extra->parent_buffer;
                    e.d      = cap_extra->d;
                    e.q      = cap_extra->q;
                    // A pooled parent and its q/d sub-buffers may still be
                    // referenced by commands already queued for compute.
                    // Record a completion marker before making the entry
                    // reusable; otherwise a later reload can release the old
                    // sub-buffers while clCreateImage is using them and the
                    // driver reports CL_INVALID_MEM_OBJECT.  Unsafe immediate
                    // reuse remains available only as an explicit diagnostic.
                    static const bool s_defer_reuse = []() {
                        const char *defer = std::getenv("GGML_ELASTIC_EVICT_DEFER_REUSE");
                        return !(defer && *defer == '0');
                    }();
                    if (s_defer_reuse) {
                        cl_event ready = nullptr;
                        if (clEnqueueMarkerWithWaitList(cap_q, 0, nullptr, &ready) == CL_SUCCESS && ready) {
                            e.ready_event = ready;
                        }
                    }
                    // double-evict 检测: 该 parent 已在 pool 里? (= bug: 它仍被某 resident
                    // tensor 使用却被当成可复用) → 跳过 push 避免 double-handout。
                    if (!octx->soa_pooled_parents.insert({e.parent, 1}).second) {
                        bool actually_in_pool = false;
                        auto pit = octx->soa_pool_by_size.find(nbytes);
                        if (pit != octx->soa_pool_by_size.end()) {
                            for (const auto & pooled : pit->second) {
                                if ((cl_mem) pooled.parent == (cl_mem) e.parent) {
                                    actually_in_pool = true;
                                    break;
                                }
                            }
                        }
                        if (actually_in_pool) {
                            octx->soa_double_evict++;
                            std::fprintf(stderr, "[soa-pool] DOUBLE-EVICT parent=%p idx=%d (already pooled, skip)\n",
                                         (void*)e.parent, idx);
                            // 不重复入池; 但仍需 mark_evicted + 清 extra 指针
                            cap_extra->parent_buffer = nullptr; cap_extra->d = nullptr; cap_extra->q = nullptr;
                            elastic::wbm_mark_evicted(octx->wbm, idx);
                            return 0;
                        }
                        // The tracking set can become stale if a pooled parent
                        // was released while trimming the cache and the driver
                        // later reused the same cl_mem handle value. Recover the
                        // set and keep this resident buffer eligible for reuse.
                        octx->soa_pooled_parents.erase(e.parent);
                        octx->soa_pooled_parents.insert({e.parent, 1});
                    }
                    octx->soa_pool_by_size[nbytes].push_back(e);
                    octx->retain_order_sizes.push_back(nbytes);
                    octx->cached_bytes += nbytes;
                    cap_extra->parent_buffer = nullptr;
                    cap_extra->d = nullptr;
                    cap_extra->q = nullptr;
                    octx->bytes_evicted_total += nbytes;
                    elastic::wbm_mark_evicted(octx->wbm, idx);
                    return 0;
                }
                // 不开 retain：直接 release sub-buffer + parent
                cap_extra->reset();
                ggml_opencl_elastic_evict_parent_pool_or_release(octx, cap_extra->parent_buffer, nbytes, idx);
                cap_extra->parent_buffer = nullptr;
                return 0;
            };
            auto reload_fn = [octx, cap_extra, cap_bctx, cap_backend_ctx, host_ptr, nbytes, cap_n_blocks,
                              cap_size_q, cap_size_d, cap_q, cap_ctx, cap_kernel,
                              need_transpose, cap_M, cap_K]() -> int {
                int idx = cap_extra->wbm_idx;
                cl_int err = CL_SUCCESS;
                cl_mem new_parent = nullptr;
                bool soa_hit = false;
                auto detail_now_us = []() -> uint64_t {
                    using clock = std::chrono::steady_clock;
                    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                            clock::now().time_since_epoch()).count();
                };
                auto detail_record = [&](elastic::wbmcl_stage_detail_kind kind, uint64_t t0, size_t bytes) {
                    elastic::wbmcl_record_stage_detail(octx, kind, detail_now_us() - t0, bytes);
                };
                std::lock_guard<std::mutex> staging_guard(octx->soa_staging_mtx);
                std::unique_lock<std::mutex> pool_guard(octx->soa_pool_mtx);
                // 诊断: GGML_ELASTIC_POOL_SYNC=1 → reload 前 drain compute queue,
                // 确保被复用 buffer 上任何 in-flight GPU op 已完成 (验证 use-after-evict race)。
                static const bool s_pool_sync = []() {
                    const char *e = std::getenv("GGML_ELASTIC_POOL_SYNC");
                    return e && *e && *e != '0';
                }();
                if (s_pool_sync) clFinish(cap_q);
                // 诊断: GGML_ELASTIC_NO_SOA_HIT=1 → 跳过三件套(parent+q+d)复用,
                // 只复用 parent(走 alloc_or_pool_parent)然后重建 fresh q/d sub-buffer。
                // 用来 bisect: 若此时输出正确, 则 corruption 在 q/d 三件套复用本身。
                static const bool s_no_soa_hit = []() {
                    const char *e = std::getenv("GGML_ELASTIC_NO_SOA_HIT");
                    return e && *e && *e != '0';
                }();
                // 先查 SOA pool (parent + d + q 三件套都复用，省 4 个 sub-buffer 操作)
                uint64_t detail_t0 = detail_now_us();
                bool tried_soa_pool = false;
                if (octx->retain_cl_mem && !s_no_soa_hit) {
                    tried_soa_pool = true;
                    auto it = octx->soa_pool_by_size.find(nbytes);
                    if (it != octx->soa_pool_by_size.end() && !it->second.empty()) {
                        octx->soa_pool_hit++;
                        auto e = it->second.back(); it->second.pop_back();
                        if (e.ready_event) {
                            cl_event ready = (cl_event) e.ready_event;
                            clWaitForEvents(1, &ready);
                            clReleaseEvent(ready);
                        }
                        new_parent     = (cl_mem)e.parent;
                        octx->soa_pooled_parents.erase(e.parent);   // 出池
                        // 默认只复用 parent, q/d sub-buffer 每次 reload 重建。
                        // OP12/Adreno 在 dynamic plan 切换后复用旧 sub-buffer
                        // 偶发 CL_INVALID_MEM_OBJECT; GGML_ELASTIC_POOL_PARENT_ONLY=0
                        // 可恢复完整三件套复用用于性能诊断。
                        static const bool s_parent_only = []() {
                            const char *e2 = std::getenv("GGML_ELASTIC_POOL_PARENT_ONLY");
                            return !(e2 && *e2 == '0');
                        }();
                        if (s_parent_only) {
                            if (e.q) clReleaseMemObject((cl_mem)e.q);
                            if (e.d) clReleaseMemObject((cl_mem)e.d);
                            soa_hit = false;   // 走下面 !soa_hit 重建 fresh sub-buffer
                        } else {
                            cap_extra->d   = (cl_mem)e.d;
                            cap_extra->q   = (cl_mem)e.q;
                            cap_extra->size_d = cap_size_d;
                            cap_extra->size_q = cap_size_q;
                            soa_hit = true;
                        }
                        // 从 FIFO 列表移除一个 == nbytes 的 entry
                        for (auto lit = octx->retain_order_sizes.rbegin();
                             lit != octx->retain_order_sizes.rend(); ++lit) {
                            if (*lit == nbytes) { octx->retain_order_sizes.erase(std::next(lit).base()); break; }
                        }
                        octx->cached_bytes -= std::min(octx->cached_bytes, nbytes);
                    }
                }
                if (tried_soa_pool && !new_parent) {
                    octx->soa_pool_miss++;
                }
                detail_record(elastic::wbmcl_stage_detail_kind::SOA_POOL_LOOKUP, detail_t0, nbytes);
                if (!new_parent) {
                    detail_t0 = detail_now_us();
                    new_parent = ggml_opencl_elastic_alloc_or_pool_parent(octx, cap_ctx, nbytes);
                    detail_record(elastic::wbmcl_stage_detail_kind::PARENT_ALLOC, detail_t0, nbytes);
                    if (!new_parent) return -1;
                }
                cap_extra->parent_buffer = new_parent;
                if (cap_extra->ctx_slot >= 0 && (size_t)cap_extra->ctx_slot < cap_bctx->buffer.size()) {
                    cap_bctx->buffer[cap_extra->ctx_slot] = new_parent;
                }
                pool_guard.unlock();
                // 诊断 GGML_ELASTIC_POOL_COHERE=1: 复用 parent 时 convert 前 clEnqueueFillBuffer
                // 全写一遍, 逼 driver 重置该 cl_mem 残留的 image-aliasing 状态。
                static const bool s_pool_cohere = []() {
                    const char *e = std::getenv("GGML_ELASTIC_POOL_COHERE");
                    return e && *e && *e != '0';
                }();
                if (s_pool_cohere) {
                    const cl_uint zero = 0;
                    cl_event fev = nullptr;
                    if (clEnqueueFillBuffer(cap_q, new_parent, &zero, sizeof(zero), 0, nbytes,
                                            0, nullptr, &fev) == CL_SUCCESS && fev) {
                        clWaitForEvents(1, &fev); clReleaseEvent(fev);
                    }
                }
                cl_event write_ev = nullptr;
                cl_command_queue xfer_q = octx->xfer_queue ? octx->xfer_queue : cap_q;
                static const bool s_reload_on_xfer = []() {
                    const char *e = std::getenv("GGML_ELASTIC_RELOAD_ON_XFER");
                    return e && *e && *e != '0';
                }();
                cl_command_queue kernel_q = s_reload_on_xfer ? xfer_q : cap_q;
                // 默认用 O_DIRECT pread 从 disk 真读 host_ptr 对应的 file 区域,
                // 然后 transfer 到 staging. 设置 GGML_ELASTIC_DIRECT_IO=0 才回退 mmap.
                static const bool s_direct_io = []() {
                    const char *e = std::getenv("GGML_ELASTIC_DIRECT_IO");
                    return !(e && *e == '0');
                }();
                static thread_local std::vector<char> direct_scratch;
                const void *src_for_dma = host_ptr;
                bool src_is_direct_scratch = false;
                detail_t0 = detail_now_us();
                {
                    std::lock_guard<std::mutex> staging_lock(octx->host_staging_mtx);
                    auto staged = octx->host_staging_by_idx.find(idx);
                    if (staged != octx->host_staging_by_idx.end() && staged->second.size() >= nbytes) {
                        src_for_dma = staged->second.data();
                    }
                }
                if (src_for_dma == host_ptr) {
                    elastic::wbmcl_wait_host_load(octx, idx);
                    std::lock_guard<std::mutex> staging_lock(octx->host_staging_mtx);
                    auto staged = octx->host_staging_by_idx.find(idx);
                    if (staged != octx->host_staging_by_idx.end() && staged->second.size() >= nbytes) {
                        src_for_dma = staged->second.data();
                    }
                }
                if (src_for_dma != host_ptr) {
                    // Host staging was produced by an earlier LOAD stage.
                } else if (s_direct_io) {
                    auto reg = llama_mmap_registry_find(host_ptr);
                    if (!reg.filename.empty()) {
                        size_t file_offset = (const char*)host_ptr - (const char*)reg.base;
                        if (direct_scratch.size() < nbytes) direct_scratch.resize(nbytes);
                        auto direct_t0 = std::chrono::steady_clock::now();
                        octx->direct_read_calls++;
                        int rc = llama_pread_direct(reg.filename.c_str(), direct_scratch.data(), file_offset, nbytes);
                        auto direct_dt = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - direct_t0).count();
                        octx->direct_read_us += (uint64_t) direct_dt;
                        if (rc == 0) {
                            octx->direct_read_ok++;
                            octx->direct_read_bytes += nbytes;
                            octx->foreground_direct_read_calls++;
                            octx->foreground_direct_read_us += (uint64_t) direct_dt;
                            octx->foreground_direct_read_bytes += nbytes;
                            src_for_dma = direct_scratch.data();
                            src_is_direct_scratch = true;
                            static const bool s_cache_foreground_load = []() {
                                const char *e = std::getenv("GGML_ELASTIC_CACHE_FOREGROUND_LOAD");
                                return !(e && *e == "0"[0]);
                            }();
                            if (octx->async_stage_load && s_cache_foreground_load) {
                                {
                                    std::lock_guard<std::mutex> staging_lock(octx->host_staging_mtx);
                                    auto &staged = octx->host_staging_by_idx[idx];
                                    staged.assign(direct_scratch.data(), direct_scratch.data() + nbytes);
                                }
                                {
                                    std::lock_guard<std::mutex> load_lock(octx->async_load_mtx);
                                    octx->async_load_state[idx] = 2;
                                }
                            }
                        } else {
                            octx->direct_read_fail++;
                            std::fprintf(stderr, "[direct-io] pread_direct rc=%d, fallback mmap\n", rc);
                        }
                    }
                }
                detail_record(elastic::wbmcl_stage_detail_kind::HOST_SRC, detail_t0, nbytes);

                if (!soa_hit) {
                    detail_t0 = detail_now_us();
                    cl_buffer_region region = {0, cap_size_d};
                    cap_extra->d = clCreateSubBuffer(new_parent, CL_MEM_READ_WRITE,
                                                     CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
                    if (err != CL_SUCCESS) { return -4; }
                    region = {cap_size_d, cap_size_q};
                    cap_extra->q = clCreateSubBuffer(new_parent, CL_MEM_READ_WRITE,
                                                     CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
                    if (err != CL_SUCCESS) { clReleaseMemObject(cap_extra->d); cap_extra->d = nullptr; return -5; }
                    cap_extra->size_q = cap_size_q;
                    cap_extra->size_d = cap_size_d;
                    detail_record(elastic::wbmcl_stage_detail_kind::SUBBUFFER, detail_t0, nbytes);
                }

                // Experimental split prepare path:
                // raw q4_0 AOS -> q/d SOA is done on CPU, then q/d are written
                // directly to the already-created GPU sub-buffers. This skips
                // the GPU convert kernel so the CPU part can overlap decode GPU
                // compute; upload + transpose remain GPU-side and still need
                // pacing.
                static const bool s_cpu_pretransform_q4_0 = []() {
                    const char *e = std::getenv("GGML_ELASTIC_CPU_PRETRANSFORM_Q4_0");
                    return e && *e && *e != '0';
                }();
                if (s_cpu_pretransform_q4_0) {
                    static thread_local std::vector<unsigned char> cpu_q;
                    static thread_local std::vector<unsigned char> cpu_d;
#ifndef __ARM_NEON
                    static const std::vector<uint16_t> q4_0_reorder_lut = []() {
                        std::vector<uint16_t> lut(65536);
                        for (uint32_t v = 0; v < 65536; ++v) {
                            const unsigned char x0 = (unsigned char) (v & 0xffu);
                            const unsigned char x1 = (unsigned char) ((v >> 8) & 0xffu);
                            const unsigned char lo = (unsigned char) ((x0 & 0x0f) | ((x1 & 0x0f) << 4));
                            const unsigned char hi = (unsigned char) (((x0 & 0xf0) >> 4) | (x1 & 0xf0));
                            lut[v] = (uint16_t) lo | ((uint16_t) hi << 8);
                        }
                        return lut;
                    }();
#endif
                    detail_t0 = detail_now_us();
                    cpu_q.resize(cap_size_q);
                    cpu_d.resize(cap_size_d);
                    const unsigned char *src = static_cast<const unsigned char *>(src_for_dma);
#ifndef __ARM_NEON
                    const uint16_t *lut = q4_0_reorder_lut.data();
#endif
                    for (size_t bi = 0; bi < cap_n_blocks; ++bi) {
                        const unsigned char *block = src + bi * 18;
                        cpu_d[2 * bi + 0] = block[0];
                        cpu_d[2 * bi + 1] = block[1];
                        const unsigned char *qs = block + 2;
                        unsigned char *q = cpu_q.data() + bi * 16;
#ifdef __ARM_NEON
                        const uint8x16_t x = vld1q_u8(qs);
                        const uint8x16_t lo = vandq_u8(x, vdupq_n_u8(0x0f));
                        const uint8x16_t hi = vandq_u8(x, vdupq_n_u8(0xf0));
                        const uint8x16_t lo_even_16 = vuzp1q_u8(lo, lo);
                        const uint8x16_t lo_odd_16  = vuzp2q_u8(lo, lo);
                        const uint8x16_t hi_even_16 = vuzp1q_u8(hi, hi);
                        const uint8x16_t hi_odd_16  = vuzp2q_u8(hi, hi);
                        const uint8x8_t lo_even = vget_low_u8(lo_even_16);
                        const uint8x8_t lo_odd  = vget_low_u8(lo_odd_16);
                        const uint8x8_t hi_even = vget_low_u8(hi_even_16);
                        const uint8x8_t hi_odd  = vget_low_u8(hi_odd_16);
                        const uint8x8_t q_low  = vorr_u8(lo_even, vshl_n_u8(lo_odd, 4));
                        const uint8x8_t q_high = vorr_u8(vshr_n_u8(hi_even, 4), hi_odd);
                        vst1_u8(q,     q_low);
                        vst1_u8(q + 8, q_high);
#else
                        for (int j = 0; j < 8; ++j) {
                            const uint16_t packed = lut[(uint16_t) qs[2 * j + 0] | ((uint16_t) qs[2 * j + 1] << 8)];
                            q[j]     = (unsigned char) (packed & 0xffu);
                            q[j + 8] = (unsigned char) (packed >> 8);
                        }
#endif
                    }
                    detail_record(elastic::wbmcl_stage_detail_kind::CPU_XFORM, detail_t0, nbytes);

                    cl_event write_events[2] = { nullptr, nullptr };
                    detail_t0 = detail_now_us();
                    err = clEnqueueWriteBuffer(xfer_q, cap_extra->q, CL_FALSE,
                                               0, cap_size_q, cpu_q.data(),
                                               0, nullptr, &write_events[0]);
                    if (err == CL_SUCCESS) {
                        err = clEnqueueWriteBuffer(xfer_q, cap_extra->d, CL_FALSE,
                                                   0, cap_size_d, cpu_d.data(),
                                                   0, nullptr, &write_events[1]);
                    }
                    detail_record(elastic::wbmcl_stage_detail_kind::WRITE_ENQUEUE, detail_t0, nbytes);
                    if (err != CL_SUCCESS) {
                        if (write_events[0]) clReleaseEvent(write_events[0]);
                        if (write_events[1]) clReleaseEvent(write_events[1]);
                        std::fprintf(stderr, "[soa-reload-q4_0] CPU pretransform q/d write failed idx=%d size=%zu err=%d\n",
                                     idx, nbytes, err);
                        return -8;
                    }
                    elastic::wbmcl_record_device_event(octx, write_events[0],
                                                       elastic::wbmcl_device_event_kind::TRANSFER_WRITE,
                                                       cap_size_q);
                    elastic::wbmcl_record_device_event(octx, write_events[1],
                                                       elastic::wbmcl_device_event_kind::TRANSFER_WRITE,
                                                       cap_size_d);
                    if (xfer_q != cap_q) clFlush(xfer_q);

                    if (!s_reload_on_xfer) {
                        detail_t0 = detail_now_us();
                        cl_int berr = clEnqueueBarrierWithWaitList(cap_q, 2, write_events, nullptr);
                        detail_record(elastic::wbmcl_stage_detail_kind::BARRIER_ENQUEUE, detail_t0, nbytes);
                        if (berr != CL_SUCCESS) clWaitForEvents(2, write_events);
                    }
                    clReleaseEvent(write_events[0]);
                    clReleaseEvent(write_events[1]);

                    if (need_transpose) {
                        detail_t0 = detail_now_us();
                        int rc = ggml_opencl_run_q4_0_adreno_transpose(
                            cap_backend_ctx, kernel_q, cap_extra, cap_M, cap_K, /*sync_each=*/false, octx);
                        detail_record(elastic::wbmcl_stage_detail_kind::TRANSPOSE_ENQUEUE, detail_t0, nbytes);
                        if (rc != 0) {
                            std::fprintf(stderr, "[soa-reload-q4_0] CPU pretransform adreno transpose failed idx=%d rc=%d\n",
                                         idx, rc);
                            return -9;
                        }
                    }
                    if (s_reload_on_xfer && kernel_q != cap_q) {
                        clFlush(kernel_q);
                        cl_event final_ev = nullptr;
                        if (clEnqueueMarkerWithWaitList(kernel_q, 0, nullptr, &final_ev) == CL_SUCCESS && final_ev) {
                            cl_event wait_ev = nullptr;
                            detail_t0 = detail_now_us();
                            clEnqueueBarrierWithWaitList(cap_q, 1, &final_ev, &wait_ev);
                            detail_record(elastic::wbmcl_stage_detail_kind::FINAL_WAIT_ENQUEUE, detail_t0, nbytes);
                            elastic::wbmcl_record_device_event(octx, wait_ev,
                                                               elastic::wbmcl_device_event_kind::COMPUTE_WAIT,
                                                               nbytes);
                            if (wait_ev) clReleaseEvent(wait_ev);
                            clReleaseEvent(final_ev);
                        }
                    }
                    octx->bytes_uploaded_total += nbytes;
                    detail_t0 = detail_now_us();
                    elastic::wbm_mark_resident(octx->wbm, idx, static_cast<void*>(new_parent));
                    detail_record(elastic::wbmcl_stage_detail_kind::MARK_RESIDENT, detail_t0, nbytes);
                    return 0;
                }

                detail_t0 = detail_now_us();
                cl_mem staging = ggml_opencl_elastic_get_staging(octx, cap_ctx, nbytes);
                detail_record(elastic::wbmcl_stage_detail_kind::STAGING_ALLOC, detail_t0, nbytes);
                if (!staging) { return -2; }
                // Async pipeline: staging write 走 xfer_queue (跟 compute_queue 并行)，
                // 但 staging 被复用 —— 必须 wait 前一次 reload 的 convert kernel 完成。
                // 用 octx->soa_staging_last_use_ev 串行 staging 访问。
                cl_event prev_use_ev = octx->soa_staging_slot_last_use_ev.empty() ? nullptr : octx->soa_staging_slot_last_use_ev[octx->soa_staging_current_slot];

                detail_t0 = detail_now_us();
                err = clEnqueueWriteBuffer(xfer_q, staging, CL_FALSE,
                                           0, nbytes, src_for_dma,
                                           prev_use_ev ? 1 : 0,
                                           prev_use_ev ? &prev_use_ev : nullptr,
                                           &write_ev);
                detail_record(elastic::wbmcl_stage_detail_kind::WRITE_ENQUEUE, detail_t0, nbytes);
                if (err != CL_SUCCESS) {
                    std::fprintf(stderr, "[soa-reload-q4_0] clEnqueueWriteBuffer failed idx=%d size=%zu err=%d\n",
                                 idx, nbytes, err);
                    return -3;
                }
                elastic::wbmcl_record_device_event(octx, write_ev,
                                                   elastic::wbmcl_device_event_kind::TRANSFER_WRITE,
                                                   nbytes);
                if (xfer_q != cap_q) clFlush(xfer_q);
                if (src_is_direct_scratch) {
                    clWaitForEvents(1, &write_ev);
                }
                if (prev_use_ev) clReleaseEvent(prev_use_ev);

                // GGML_ELASTIC_RELOAD_ON_XFER=1: convert + transpose 也跑在
                // xfer_queue 上, 不挤占 compute_queue. 需要的 barrier 改成
                // "compute_queue 等最终 reload event"。默认关闭: 真机
                // reload_chain_ab 显示无独立 compute 可 overlap 时通常持平或略慢。
                if (!s_reload_on_xfer) {
                    // 原路径: compute_queue 等 write_ev 完成才能开 convert kernel
                    detail_t0 = detail_now_us();
                    cl_int berr = clEnqueueBarrierWithWaitList(cap_q, 1, &write_ev, nullptr);
                    detail_record(elastic::wbmcl_stage_detail_kind::BARRIER_ENQUEUE, detail_t0, nbytes);
                    if (berr != CL_SUCCESS) clWaitForEvents(1, &write_ev);
                }
                clReleaseEvent(write_ev);
                CL_CHECK(clSetKernelArg(cap_kernel, 0, sizeof(cl_mem), &staging));
                CL_CHECK(clSetKernelArg(cap_kernel, 1, sizeof(cl_mem), &cap_extra->q));
                CL_CHECK(clSetKernelArg(cap_kernel, 2, sizeof(cl_mem), &cap_extra->d));
                size_t gws[3] = {cap_n_blocks, 1, 1};
                size_t lws[3] = {64, 1, 1};
                cl_event convert_ev = nullptr;
                detail_t0 = detail_now_us();
                err = clEnqueueNDRangeKernel(kernel_q, cap_kernel, 3, nullptr, gws, lws, 0, nullptr, &convert_ev);
                detail_record(elastic::wbmcl_stage_detail_kind::CONVERT_ENQUEUE, detail_t0, nbytes);
                if (err != CL_SUCCESS) return -6;
                elastic::wbmcl_record_device_event(octx, convert_ev,
                                                   elastic::wbmcl_device_event_kind::XFORM_CONVERT,
                                                   nbytes);
                octx->soa_staging_slot_last_use_ev[octx->soa_staging_current_slot] = convert_ev;
                // Adreno path: 转置 q/d. 也跑在 kernel_q 上.
                if (need_transpose) {
                    detail_t0 = detail_now_us();
                    int rc = ggml_opencl_run_q4_0_adreno_transpose(
                        cap_backend_ctx, kernel_q, cap_extra, cap_M, cap_K, /*sync_each=*/false, octx);
                    detail_record(elastic::wbmcl_stage_detail_kind::TRANSPOSE_ENQUEUE, detail_t0, nbytes);
                    if (rc != 0) {
                        std::fprintf(stderr, "[soa-reload-q4_0] adreno transpose 失败 idx=%d rc=%d\n", idx, rc);
                        return -7;
                    }
                }
                // 如果 kernel 跑在 xfer_q 上, compute_queue 用 barrier 等最终 event
                // (后续 matmul 入 compute_queue 时自动有依赖, 但显式 barrier 更稳)
                if (s_reload_on_xfer && kernel_q != cap_q) {
                    clFlush(kernel_q);
                    cl_event final_ev = nullptr;
                    if (clEnqueueMarkerWithWaitList(kernel_q, 0, nullptr, &final_ev) == CL_SUCCESS && final_ev) {
                        cl_event wait_ev = nullptr;
                        detail_t0 = detail_now_us();
                        clEnqueueBarrierWithWaitList(cap_q, 1, &final_ev, &wait_ev);
                        detail_record(elastic::wbmcl_stage_detail_kind::FINAL_WAIT_ENQUEUE, detail_t0, nbytes);
                        elastic::wbmcl_record_device_event(octx, wait_ev,
                                                           elastic::wbmcl_device_event_kind::COMPUTE_WAIT,
                                                           nbytes);
                        if (wait_ev) clReleaseEvent(wait_ev);
                        clReleaseEvent(final_ev);
                    }
                }
                octx->bytes_uploaded_total += nbytes;
                detail_t0 = detail_now_us();
                elastic::wbm_mark_resident(octx->wbm, idx, static_cast<void*>(new_parent));
                detail_record(elastic::wbmcl_stage_detail_kind::MARK_RESIDENT, detail_t0, nbytes);
                return 0;
            };
            ggml_opencl_elastic_register_soa(buffer, tensor, extra,
                extra_orig->data_device, extra_orig->ctx_slot,
                host_ptr, nbytes, cap_ctx, cap_q,
                std::move(evict_fn), std::move(reload_fn));
        }

        return;

    }
    if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_mxfp4 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_mxfp4();

        size_t size_e = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(char);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        GGML_ASSERT(size_e + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(
            queue, data_device, CL_TRUE, 0,
            ggml_nbytes(tensor), data, 0, NULL, NULL));

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        cl_buffer_region region;

        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_e;
        extra->e = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_e, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->kernel_convert_block_mxfp4_trans;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->e));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));
            tensor->extra = extra;

            return;
        }
#endif
        cl_kernel kernel = backend_ctx->kernel_convert_block_mxfp4;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->e));

        size_t global_work_size[3] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[3] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        // Create image for Q
        cl_image_format img_format_q = {CL_RG, CL_UNSIGNED_INT32};
        cl_image_desc img_desc_q = {
            CL_MEM_OBJECT_IMAGE1D_BUFFER,
            static_cast<size_t>(ggml_nelements(tensor)/32*2),
            0, 0, 0, 0, 0, 0, 0,
            { extra->q }
        };
        extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_q, &img_desc_q, NULL, &err);
        tensor->extra = extra;

        return;
    }
    if (tensor->type == GGML_TYPE_Q8_0) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q8_0 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q8_0();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*(ggml_blck_size(tensor->type)*sizeof(char));
        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(
            queue, data_device, CL_TRUE, 0,
            ggml_nbytes(tensor), data, 0, NULL, NULL));

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        cl_buffer_region region;

        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->kernel_convert_block_q8_0;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;

        // ───── elastic SOA register (Q8_0) ───────────────────────────────────
        if (auto *bctx = (ggml_backend_opencl_buffer_context *) buffer->context;
            bctx->mode == ggml_backend_opencl_buffer_context::MODE_ELASTIC &&
            offset == 0 && size == ggml_nbytes(tensor)) {
            const void *host_ptr        = data;
            size_t nbytes               = ggml_nbytes(tensor);
            const size_t cap_n_blocks   = (size_t)ggml_nelements(tensor) / ggml_blck_size(tensor->type);
            const size_t cap_size_q     = size_q;
            const size_t cap_size_d     = size_d;
            cl_kernel cap_kernel        = backend_ctx->kernel_convert_block_q8_0;
            elastic::wbm_opencl_ctx *octx = &ggml_opencl_elastic()->octx;
            ggml_tensor_extra_cl_q8_0 *cap_extra = extra;
            ggml_backend_opencl_buffer_context *cap_bctx = bctx;
            cl_context cap_ctx = context;
            cl_command_queue cap_q = queue;

            auto evict_fn = [octx, cap_extra, cap_q, nbytes]() -> int {
                std::lock_guard<std::mutex> pool_guard(octx->soa_pool_mtx);
                int idx = cap_extra->wbm_idx;
                if (octx->retain_cl_mem && cap_extra->parent_buffer) {
                    elastic::wbmcl_retain_pool_trim_locked(octx, nbytes);
                    elastic::soa_pool_entry e;
                    e.parent = cap_extra->parent_buffer;
                    e.d      = cap_extra->d;
                    e.q      = cap_extra->q;
                    // Q8 retained parent reuse is only safe after prior compute
                    // queue users have drained.  The xfer queue may immediately
                    // run the convert kernel after this parent leaves the pool.
                    cl_event ready = nullptr;
                    if (clEnqueueMarkerWithWaitList(cap_q, 0, nullptr, &ready) == CL_SUCCESS && ready) {
                        e.ready_event = ready;
                    }
                    if (!octx->soa_pooled_parents.insert({e.parent, 1}).second) {
                        bool actually_in_pool = false;
                        auto pit = octx->soa_pool_by_size.find(nbytes);
                        if (pit != octx->soa_pool_by_size.end()) {
                            for (const auto & pooled : pit->second) {
                                if ((cl_mem) pooled.parent == (cl_mem) e.parent) {
                                    actually_in_pool = true;
                                    break;
                                }
                            }
                        }
                        if (actually_in_pool) {
                            octx->soa_double_evict++;
                            std::fprintf(stderr, "[soa-pool-q8] DOUBLE-EVICT parent=%p idx=%d (already pooled, skip)\n",
                                         (void*)e.parent, idx);
                            cap_extra->parent_buffer = nullptr; cap_extra->d = nullptr; cap_extra->q = nullptr;
                            elastic::wbm_mark_evicted(octx->wbm, idx);
                            return 0;
                        }
                        octx->soa_pooled_parents.erase(e.parent);
                        octx->soa_pooled_parents.insert({e.parent, 1});
                    }
                    octx->soa_pool_by_size[nbytes].push_back(e);
                    octx->retain_order_sizes.push_back(nbytes);
                    octx->cached_bytes += nbytes;
                    cap_extra->parent_buffer = nullptr;
                    cap_extra->d = nullptr;
                    cap_extra->q = nullptr;
                    octx->bytes_evicted_total += nbytes;
                    elastic::wbm_mark_evicted(octx->wbm, idx);
                    return 0;
                }
                cap_extra->reset();
                ggml_opencl_elastic_evict_parent_pool_or_release(octx, cap_extra->parent_buffer, nbytes, idx);
                cap_extra->parent_buffer = nullptr;
                return 0;
            };
            auto reload_fn = [octx, cap_extra, cap_bctx, host_ptr, nbytes, cap_n_blocks,
                              cap_size_q, cap_size_d, cap_q, cap_ctx, cap_kernel]() -> int {
                int idx = cap_extra->wbm_idx;
                cl_int err = CL_SUCCESS;
                auto detail_now_us = []() -> uint64_t {
                    using clock = std::chrono::steady_clock;
                    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                            clock::now().time_since_epoch()).count();
                };
                auto detail_record = [&](elastic::wbmcl_stage_detail_kind kind, uint64_t t0, size_t bytes) {
                    elastic::wbmcl_record_stage_detail(octx, kind, detail_now_us() - t0, bytes);
                };
                std::lock_guard<std::mutex> staging_guard(octx->soa_staging_mtx);
                std::unique_lock<std::mutex> pool_guard(octx->soa_pool_mtx);
                uint64_t detail_t0 = detail_now_us();
                cl_mem new_parent = nullptr;
                bool soa_hit = false;
                bool parent_from_pool = false;
                static const bool s_pool_sync = []() {
                    const char *e = std::getenv("GGML_ELASTIC_POOL_SYNC");
                    return e && *e && *e != '0';
                }();
                if (s_pool_sync) clFinish(cap_q);
                static const bool s_no_soa_hit = []() {
                    const char *e = std::getenv("GGML_ELASTIC_NO_SOA_HIT");
                    return e && *e && *e != '0';
                }();
                bool tried_soa_pool = false;
                if (octx->retain_cl_mem && !s_no_soa_hit) {
                    tried_soa_pool = true;
                    auto it = octx->soa_pool_by_size.find(nbytes);
                    if (it != octx->soa_pool_by_size.end() && !it->second.empty()) {
                        octx->soa_pool_hit++;
                        auto e = it->second.back(); it->second.pop_back();
                        if (e.ready_event) {
                            cl_event ready = (cl_event) e.ready_event;
                            clWaitForEvents(1, &ready);
                            clReleaseEvent(ready);
                        }
                        new_parent = (cl_mem)e.parent;
                        parent_from_pool = true;
                        octx->soa_pooled_parents.erase(e.parent);
                        static const bool s_parent_only = []() {
                            const char *e2 = std::getenv("GGML_ELASTIC_POOL_PARENT_ONLY");
                            return !(e2 && *e2 == '0');
                        }();
                        if (s_parent_only) {
                            if (e.q) clReleaseMemObject((cl_mem)e.q);
                            if (e.d) clReleaseMemObject((cl_mem)e.d);
                            soa_hit = false;
                        } else {
                            cap_extra->d = (cl_mem)e.d;
                            cap_extra->q = (cl_mem)e.q;
                            cap_extra->size_d = cap_size_d;
                            cap_extra->size_q = cap_size_q;
                            soa_hit = true;
                        }
                        for (auto lit = octx->retain_order_sizes.rbegin();
                             lit != octx->retain_order_sizes.rend(); ++lit) {
                            if (*lit == nbytes) { octx->retain_order_sizes.erase(std::next(lit).base()); break; }
                        }
                        octx->cached_bytes -= std::min(octx->cached_bytes, nbytes);
                    }
                }
                if (tried_soa_pool && !new_parent) {
                    octx->soa_pool_miss++;
                }
                detail_record(elastic::wbmcl_stage_detail_kind::SOA_POOL_LOOKUP, detail_t0, nbytes);
                if (!new_parent) {
                    detail_t0 = detail_now_us();
                    new_parent = ggml_opencl_elastic_alloc_or_pool_parent(octx, cap_ctx, nbytes);
                    detail_record(elastic::wbmcl_stage_detail_kind::PARENT_ALLOC, detail_t0, nbytes);
                    if (!new_parent) return -1;
                }
                cap_extra->parent_buffer = new_parent;
                if (cap_extra->ctx_slot >= 0 && (size_t)cap_extra->ctx_slot < cap_bctx->buffer.size()) {
                    cap_bctx->buffer[cap_extra->ctx_slot] = new_parent;
                }
                pool_guard.unlock();
                detail_t0 = detail_now_us();
                cl_mem staging = ggml_opencl_elastic_get_staging(octx, cap_ctx, nbytes);
                detail_record(elastic::wbmcl_stage_detail_kind::STAGING_ALLOC, detail_t0, nbytes);
                if (!staging) { clReleaseMemObject(new_parent); return -2; }
                // Async pipeline (Q8_0 同 Q4_0): xfer_queue staging write + 串行事件链
                cl_event write_ev = nullptr;
                cl_command_queue xfer_q = octx->xfer_queue ? octx->xfer_queue : cap_q;
                cl_event prev_use_ev = octx->soa_staging_slot_last_use_ev.empty() ? nullptr : octx->soa_staging_slot_last_use_ev[octx->soa_staging_current_slot];
                const void *src_for_dma = host_ptr;
                static const bool s_direct_io = []() {
                    const char *e = std::getenv("GGML_ELASTIC_DIRECT_IO");
                    return !(e && *e == '0');
                }();
                static thread_local std::vector<char> direct_scratch;
                bool src_is_direct_scratch = false;
                detail_t0 = detail_now_us();
                elastic::wbmcl_wait_host_load(octx, idx);
                {
                    std::lock_guard<std::mutex> staging_lock(octx->host_staging_mtx);
                    auto staged = octx->host_staging_by_idx.find(idx);
                    if (staged != octx->host_staging_by_idx.end() && staged->second.size() >= nbytes) {
                        src_for_dma = staged->second.data();
                    }
                }
                if (src_for_dma == host_ptr && s_direct_io) {
                    auto reg = llama_mmap_registry_find(host_ptr);
                    octx->direct_read_calls++;
                    auto direct_t0 = std::chrono::steady_clock::now();
                    int rc = -1;
                    if (!reg.filename.empty()) {
                        size_t file_offset = (const char*)host_ptr - (const char*)reg.base;
                        if (direct_scratch.size() < nbytes) direct_scratch.resize(nbytes);
                        rc = llama_pread_direct(reg.filename.c_str(), direct_scratch.data(), file_offset, nbytes);
                        static const bool s_verify_direct = []() {
                            const char * e = std::getenv("GGML_ELASTIC_DIRECT_VERIFY");
                            return e && *e && *e != '0';
                        }();
                        if (rc == 0 && s_verify_direct &&
                            std::memcmp(direct_scratch.data(), host_ptr, nbytes) != 0) {
                            const unsigned char * a = (const unsigned char *) direct_scratch.data();
                            const unsigned char * b = (const unsigned char *) host_ptr;
                            size_t mismatch = 0;
                            while (mismatch < nbytes && a[mismatch] == b[mismatch]) {
                                mismatch++;
                            }
                            static std::atomic<int> s_q8_direct_mismatch_logs{0};
                            if (s_q8_direct_mismatch_logs.fetch_add(1, std::memory_order_relaxed) < 8) {
                                std::fprintf(stderr,
                                             "[direct-io-q8] verify mismatch idx=%d file_off=%zu rel=%zu direct=%02x mmap=%02x, fallback mmap\n",
                                             idx, file_offset, mismatch,
                                             mismatch < nbytes ? a[mismatch] : 0,
                                             mismatch < nbytes ? b[mismatch] : 0);
                            }
                            rc = -5;
                        }
                    }
                    auto direct_dt = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - direct_t0).count();
                    octx->direct_read_us += (uint64_t) direct_dt;
                    if (rc == 0) {
                        octx->direct_read_ok++;
                        octx->direct_read_bytes += nbytes;
                        octx->foreground_direct_read_calls++;
                        octx->foreground_direct_read_us += (uint64_t) direct_dt;
                        octx->foreground_direct_read_bytes += nbytes;
                        src_for_dma = direct_scratch.data();
                        src_is_direct_scratch = true;
                        static const bool s_cache_foreground_load = []() {
                            const char *e = std::getenv("GGML_ELASTIC_CACHE_FOREGROUND_LOAD");
                            return !(e && *e == '0');
                        }();
                        if (octx->async_stage_load && s_cache_foreground_load) {
                            {
                                std::lock_guard<std::mutex> staging_lock(octx->host_staging_mtx);
                                auto &staged = octx->host_staging_by_idx[idx];
                                staged.assign(direct_scratch.data(), direct_scratch.data() + nbytes);
                            }
                            {
                                std::lock_guard<std::mutex> load_lock(octx->async_load_mtx);
                                octx->async_load_state[idx] = 2;
                            }
                        }
                    } else {
                        octx->direct_read_fail++;
                        if (reg.filename.empty()) {
                            static std::atomic<int> s_q8_direct_reg_miss_logs{0};
                            if (s_q8_direct_reg_miss_logs.fetch_add(1, std::memory_order_relaxed) < 8) {
                                std::fprintf(stderr, "[direct-io-q8] mmap registry miss idx=%d size=%zu, fallback mmap\n",
                                             idx, nbytes);
                            }
                        } else {
                            std::fprintf(stderr, "[direct-io-q8] pread_direct rc=%d, fallback mmap\n", rc);
                        }
                    }
                }
                detail_record(elastic::wbmcl_stage_detail_kind::HOST_SRC, detail_t0, nbytes);
                detail_t0 = detail_now_us();
                err = clEnqueueWriteBuffer(xfer_q, staging, CL_FALSE,
                                           0, nbytes, src_for_dma,
                                           prev_use_ev ? 1 : 0,
                                           prev_use_ev ? &prev_use_ev : nullptr,
                                           &write_ev);
                detail_record(elastic::wbmcl_stage_detail_kind::WRITE_ENQUEUE, detail_t0, nbytes);
                if (err != CL_SUCCESS) {
                    std::fprintf(stderr, "[soa-reload-q8_0] clEnqueueWriteBuffer failed idx=%d size=%zu err=%d\n",
                                 idx, nbytes, err);
                    clReleaseMemObject(new_parent);
                    return -3;
                }
                elastic::wbmcl_record_device_event(octx, write_ev,
                                                   elastic::wbmcl_device_event_kind::TRANSFER_WRITE,
                                                   nbytes);
                if (xfer_q != cap_q) clFlush(xfer_q);
                if (src_is_direct_scratch) {
                    clWaitForEvents(1, &write_ev);
                }
                if (prev_use_ev) clReleaseEvent(prev_use_ev);

                detail_t0 = detail_now_us();
                if (!soa_hit) {
                    cl_buffer_region region = {0, cap_size_d};
                    cap_extra->d = clCreateSubBuffer(new_parent, CL_MEM_READ_WRITE,
                                                     CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
                    if (err != CL_SUCCESS) { clReleaseMemObject(new_parent); return -4; }
                    region = {cap_size_d, cap_size_q};
                    cap_extra->q = clCreateSubBuffer(new_parent, CL_MEM_READ_WRITE,
                                                     CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
                    if (err != CL_SUCCESS) { clReleaseMemObject(cap_extra->d); cap_extra->d = nullptr;
                                             clReleaseMemObject(new_parent); return -5; }
                    cap_extra->size_q = cap_size_q;
                    cap_extra->size_d = cap_size_d;
                }
                detail_record(elastic::wbmcl_stage_detail_kind::SUBBUFFER, detail_t0, nbytes);
                static const bool s_reload_on_xfer = []() {
                    const char *e = std::getenv("GGML_ELASTIC_RELOAD_ON_XFER");
                    return e && *e && *e != '0';
                }();
                cl_command_queue kernel_q = (s_reload_on_xfer && !parent_from_pool) ? xfer_q : cap_q;
                if (kernel_q == cap_q) {
                    detail_t0 = detail_now_us();
                    cl_int berr = clEnqueueBarrierWithWaitList(cap_q, 1, &write_ev, nullptr);
                    detail_record(elastic::wbmcl_stage_detail_kind::BARRIER_ENQUEUE, detail_t0, nbytes);
                    if (berr != CL_SUCCESS) clWaitForEvents(1, &write_ev);
                }
                clReleaseEvent(write_ev);
                CL_CHECK(clSetKernelArg(cap_kernel, 0, sizeof(cl_mem), &staging));
                CL_CHECK(clSetKernelArg(cap_kernel, 1, sizeof(cl_mem), &cap_extra->q));
                CL_CHECK(clSetKernelArg(cap_kernel, 2, sizeof(cl_mem), &cap_extra->d));
                size_t gws[3] = {cap_n_blocks, 1, 1};
                size_t lws[3] = {64, 1, 1};
                cl_event convert_ev = nullptr;
                detail_t0 = detail_now_us();
                err = clEnqueueNDRangeKernel(kernel_q, cap_kernel, 3, nullptr, gws, lws, 0, nullptr, &convert_ev);
                detail_record(elastic::wbmcl_stage_detail_kind::CONVERT_ENQUEUE, detail_t0, nbytes);
                if (err != CL_SUCCESS) return -6;
                elastic::wbmcl_record_device_event(octx, convert_ev,
                                                   elastic::wbmcl_device_event_kind::XFORM_CONVERT,
                                                   nbytes);
                octx->soa_staging_slot_last_use_ev[octx->soa_staging_current_slot] = convert_ev;
                if (s_reload_on_xfer && kernel_q != cap_q) {
                    clFlush(kernel_q);
                    cl_event final_ev = nullptr;
                    if (clEnqueueMarkerWithWaitList(kernel_q, 0, nullptr, &final_ev) == CL_SUCCESS && final_ev) {
                        cl_event wait_ev = nullptr;
                        detail_t0 = detail_now_us();
                        clEnqueueBarrierWithWaitList(cap_q, 1, &final_ev, &wait_ev);
                        detail_record(elastic::wbmcl_stage_detail_kind::FINAL_WAIT_ENQUEUE, detail_t0, nbytes);
                        elastic::wbmcl_record_device_event(octx, wait_ev,
                                                           elastic::wbmcl_device_event_kind::COMPUTE_WAIT,
                                                           nbytes);
                        if (wait_ev) clReleaseEvent(wait_ev);
                        clReleaseEvent(final_ev);
                    }
                }
                octx->bytes_uploaded_total += nbytes;
                detail_t0 = detail_now_us();
                elastic::wbm_mark_resident(octx->wbm, idx, static_cast<void*>(new_parent));
                detail_record(elastic::wbmcl_stage_detail_kind::MARK_RESIDENT, detail_t0, nbytes);
                return 0;
            };
            ggml_opencl_elastic_register_soa(buffer, tensor, extra,
                extra_orig->data_device, extra_orig->ctx_slot,
                host_ptr, nbytes, cap_ctx, cap_q,
                std::move(evict_fn), std::move(reload_fn));
        }

        return;
    }
#endif // GGML_OPENCL_SOA_Q

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    GGML_ASSERT(extra);

    CL_CHECK(clEnqueueWriteBuffer(
        queue, extra->data_device, CL_TRUE, extra->offset + offset,
        size, data, 0, NULL, NULL));

    // Elastic baseline (F4-minimal)：把权重 tensor 注册到 WBM。host_ptr 用
    // set_tensor 的 data 参数——这是 llama_model_loader 调来的 mmap 指针，
    // 在模型生命周期内稳定。后续 evict 时重新 clEnqueueWriteBuffer 即可恢复。
    // 只走 elastic 模式 + 完整覆盖写 (offset == 0 && size == ggml_nbytes) 的情况，
    // 跳过 view 与已经注册过的 tensor。
    {
        ggml_backend_opencl_buffer_context * bctx =
            (ggml_backend_opencl_buffer_context *) buffer->context;
        if (bctx->mode == ggml_backend_opencl_buffer_context::MODE_ELASTIC
            && tensor->view_src == nullptr
            && extra->wbm_idx < 0
            && offset == 0
            && size == ggml_nbytes(tensor)
            && ggml_opencl_is_weight_tensor(tensor->name)) {
            ggml_opencl_elastic_lazy_init(context, queue);
            auto *s = ggml_opencl_elastic();
            if (s->wbm_inited) {
                int idx = elastic::wbm_add_block(&s->wbm, const_cast<void*>(data), size);
                if (idx >= 0) {
                    elastic::wbm_mark_resident(&s->wbm, idx,
                        static_cast<void*>(extra->data_device));
                    elastic::wbm_touch(&s->wbm, idx, s->current_token);
                    extra->wbm_idx = idx;

                    // v8.4: 注册 name → idx 让 llama_weight_host_ptr_query 找得到.
                    // 跟另一个 register 路径 (line 4188 area) 平行 — 两处都要 register.
                    if (tensor && tensor->name[0]) {
                        std::lock_guard<std::mutex> lk(s->sched_mtx);
                        s->name_to_wbm[tensor->name] = idx;
                        s->name_to_wbms[tensor->name].push_back(idx);
                        s->wbm_to_name[idx] = tensor->name;
                    }
                    if (tensor && tensor->name[0]) {
                        llama_weight_runtime_mark_resident(tensor->name, LLAMA_WEIGHT_RUNTIME_GPU);
                    }
                    opencl_sched_register_once();
                    // 同步 slot → wbm_idx，析构时能识别 WBM 已 evict 的 dead slot
                    if (extra->ctx_slot >= 0 &&
                        static_cast<size_t>(extra->ctx_slot) < bctx->wbm_idx_per_slot.size()) {
                        bctx->wbm_idx_per_slot[extra->ctx_slot] = idx;
                    }

                    // Q5_K/Q6_K currently use the raw GGUF block layout in the
                    // OpenCL matmul path. Register a reload callback anyway so
                    // explicit elastic plans can drive them through the same
                    // LOAD -> TRANSFER -> TRANSFORM callback sequence as SOA weights.
                    // The callback does not run an AOS->SOA kernel; it reloads
                    // the raw cl_mem and refreshes the elastic buffer slot.
                    if (tensor->type == GGML_TYPE_Q5_K || tensor->type == GGML_TYPE_Q6_K) {
                        elastic::wbm_opencl_ctx * octx = &s->octx;
                        ggml_tensor_extra_cl * cap_extra = extra;
                        ggml_backend_buffer_t cap_buffer = buffer;
                        auto reload_fn = [octx, cap_extra, cap_buffer]() -> int {
                            const int ridx = cap_extra->wbm_idx;
                            int rc = elastic::wbmcl_ensure_resident(octx, ridx);
                            if (rc != 0) return rc;
                            cl_mem buf = elastic::wbmcl_get_buffer(octx, ridx);
                            if (buf) {
                                cap_extra->data_device = buf;
                                ggml_opencl_elastic_update_ctx_slot(cap_buffer, cap_extra->ctx_slot, buf);
                            }
                            return 0;
                        };
                        elastic::wbmcl_register_soa(&s->octx, idx, nullptr, std::move(reload_fn));
                    }

                    // Pin 小但常访问的 tensor：norm（4 KB 量级、每 op 必用）+ GQA
                    // 的 attn_k/v（每层 2 MB）。按 tensor 名后缀决定。
                    // GGML_ELASTIC_PIN=norm,k,v,q（默认 norm,k,v；写空串关闭；
                    // 写 all 把所有 attn 权重和 norm 都 pin 上做 A/B 对比）
                    static std::string pin_policy = []() -> std::string {
                        const char *e = std::getenv("GGML_ELASTIC_PIN");
                        return e ? std::string(e) : std::string("norm,k,v");
                    }();
                    auto contains = [&](const char *tok) {
                        if (pin_policy == "all") return true;
                        const std::string delimited =
                            "," + pin_policy + ",";
                        return delimited.find(
                            "," + std::string(tok) + ",") !=
                            std::string::npos;
                    };
                    const std::string suffix = ggml_opencl_tensor_suffix(tensor->name);
                    bool should_pin = false;
                    if (contains("norm") &&
                        (suffix == "attn_norm" || suffix == "ffn_norm" ||
                         suffix == "output_norm")) should_pin = true;
                    if (contains("k") && suffix == "attn_k") should_pin = true;
                    if (contains("v") && suffix == "attn_v") should_pin = true;
                    if (contains("q") && suffix == "attn_q") should_pin = true;
                    if (contains("o") && suffix == "attn_output") should_pin = true;
                    if (contains("token_embd") &&
                        suffix == "token_embd") should_pin = true;
                    if (contains("output") &&
                        suffix == "output") should_pin = true;
                    if (should_pin) {
                        elastic::wbm_set_pinned(&s->wbm, idx, true);
                        if (suffix == "token_embd" ||
                            suffix == "output") {
                            GGML_LOG_INFO(
                                "ggml_opencl elastic: pin %s inside "
                                "weight budget (%.2f MiB)\n",
                                suffix.c_str(),
                                size / 1024.0 / 1024.0);
                        }
                    }

                    // GGML_ELASTIC_EMBED_OUTSIDE_BUDGET=1：把 token_embd（501 MB
                    // 但只用 1 行 / token，每次 reload 要 ~240 ms）pin 起来，并
                    // 把它的字节数加到 static_target_bytes 上等于"对它的开销
                    // 不计入预算"。技术上违反 spec §5 的 M_floor 契约（实际 GPU
                    // 占用 = M_floor + embed_bytes），但因为是 read-only 的固定
                    // 量，可以理解成对 M_floor 的常数偏置。
                    static const bool s_embed_out = []() {
                        const char *e = std::getenv("GGML_ELASTIC_EMBED_OUTSIDE_BUDGET");
                        return e && *e && *e != '0';
                    }();
                    if (s_embed_out && suffix == "token_embd") {
                        elastic::wbm_set_pinned(&s->wbm, idx, true);
                        s->static_target_bytes += size;
                        s->extra_target_bytes  += size;   // dynamic 路径用
                        GGML_LOG_INFO("ggml_opencl elastic: pin token_embd (%zu MB) outside budget → target=%zu MB\n",
                                      size / 1024 / 1024, s->static_target_bytes / 1024 / 1024);
                    }
                    static const bool s_pin_unplanned_output_nonsoa = []() {
                        const char *e = std::getenv("GGML_ELASTIC_PIN_UNPLANNED_OUTPUT");
                        return !(e && *e == "0"[0]);
                    }();
                    static const bool s_unplanned_output_counts_budget_nonsoa = []() {
                        const char *e = std::getenv("GGML_ELASTIC_PIN_UNPLANNED_OUTPUT_COUNTS_BUDGET");
                        return e && *e && *e != '0';
                    }();
                    if (s_pin_unplanned_output_nonsoa && suffix == "output") {
                        elastic::wbm_set_pinned(&s->wbm, idx, true);
                        if (!s_unplanned_output_counts_budget_nonsoa) {
                            s->static_target_bytes += size;
                            s->extra_target_bytes  += size;
                            GGML_LOG_INFO("ggml_opencl elastic: pin unplanned output (%zu MB) outside budget -> target=%zu MB\n",
                                          size / 1024 / 1024, s->static_target_bytes / 1024 / 1024);
                        } else {
                            GGML_LOG_INFO("ggml_opencl elastic: pin unplanned output (%zu MB) inside budget -> target=%zu MB\n",
                                          size / 1024 / 1024, s->static_target_bytes / 1024 / 1024);
                        }
                    }

                    if (ggml_opencl_elastic_evict_during_load_enabled()) {
                        ggml_opencl_elastic_evict_to_initial_budget(&s->wbm, &s->octx, s->static_target_bytes, -1);
                    }
                }
            }
        }
    }

    GGML_UNUSED(buffer);
}

static void ggml_backend_opencl_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->extra);

    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer->buft->device);

    cl_context context = backend_ctx->context;
    cl_command_queue queue = backend_ctx->queue;

    // Make sure all previously submitted commands in other devices are finished.
    sync_with_other_backends(backend_ctx);

#ifdef GGML_OPENCL_SOA_Q
    // In end-to-end runs, get_tensor is usually used to get back the logits,
    // where we can simply do clEnqueueReadBuffer since they are f32.
    // However, in test-backend-ops, the GPU graph is copied to the CPU backend,
    // which requires reading back quantized weight tensors.
    // To properly support this, we need to restore block_q4_0 struct arrays
    // from the flattened buffers.
    if (tensor->type == GGML_TYPE_Q4_0) {
        ggml_tensor_extra_cl_q4_0 * extra = (ggml_tensor_extra_cl_q4_0 *)tensor->extra;

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->kernel_restore_block_q4_0;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    } else if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_tensor_extra_cl_mxfp4 * extra = (ggml_tensor_extra_cl_mxfp4 *)tensor->extra;

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->kernel_restore_block_mxfp4_trans;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->e));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
#endif
        cl_kernel kernel = backend_ctx->kernel_restore_block_mxfp4;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->e));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_Q8_0) {
        ggml_tensor_extra_cl_q8_0 * extra = (ggml_tensor_extra_cl_q8_0 *)tensor->extra;

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->kernel_restore_block_q8_0;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
#endif // GGML_OPENCL_SOA_Q

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;

    CL_CHECK(clEnqueueReadBuffer(
        queue, extra->data_device, CL_TRUE, extra->offset + tensor->view_offs + offset,
        size, data, 0, NULL, NULL));

    GGML_UNUSED(buffer);
}

static void ggml_backend_opencl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_dev_t dev = buffer->buft->device;
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(dev);
    cl_command_queue queue = backend_ctx->queue;

    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    // ELASTIC 是权重 buffer，set_tensor 会覆盖写完整内容，不需要 zero-init。
    // PENDING 还没决定升路径，更不需要 clear。
    if (ctx->mode != ggml_backend_opencl_buffer_context::MODE_MONOLITHIC) {
        return;
    }
    for (cl_mem buf : ctx->buffer) {
        CL_CHECK(clEnqueueFillBuffer(queue, buf, &value, sizeof(value), 0, buffer->size, 0, NULL, NULL));
    }
    CL_CHECK(clFinish(queue));
}

static void ggml_backend_opencl_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    ctx->reset();
}

static ggml_backend_buffer_i ggml_backend_opencl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_opencl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_opencl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_opencl_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_opencl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_opencl_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_opencl_buffer_clear,
    /* .reset           = */ ggml_backend_opencl_buffer_reset,
};

//
// buffer type
//

static const char * ggml_backend_opencl_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    return "OpenCL";

    GGML_UNUSED(buffer_type);
}

static ggml_backend_buffer_t ggml_backend_opencl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buffer_type, size_t size) {
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer_type->device);

    // clCreateBuffer returns -61 for size 0
    size = std::max(size, (size_t)1);

    // elastic 模式：alloc_buffer 拿不到 tensor 名，无法判断是权重还是 KV。
    // 进 PENDING 态，记录 size 但**不**调 clCreateBuffer，等首个 init_tensor
    // 看 tensor name 再决定升 ELASTIC（权重）还是 MONOLITHIC（KV / 激活）。
    if (ggml_opencl_elastic_enabled()) {
        ggml_backend_opencl_buffer_context * ctx = new ggml_backend_opencl_buffer_context(
            ggml_backend_opencl_buffer_context::pending_tag_t{}, size);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_opencl_buffer_interface, ctx, size);
    }

    cl_int err;
    cl_mem mem = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_INFO("%s: failed to allocate %.2f MiB\n", __func__, size / 1024.0 / 1024.0);
        return nullptr;
    }

    ggml_backend_opencl_buffer_context * ctx = new ggml_backend_opencl_buffer_context(mem);

    return ggml_backend_buffer_init(buffer_type, ggml_backend_opencl_buffer_interface, ctx, size);
}

static size_t ggml_backend_opencl_buffer_type_get_alignment(ggml_backend_buffer_type_t buffer_type) {
    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer_type->device);
    return backend_ctx->alignment;
}

static size_t ggml_backend_opencl_buffer_type_get_max_size(ggml_backend_buffer_type_t buffer_type) {
    static size_t max_size = -1;
    if (max_size == (size_t)-1) {
        ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer_type->device);
        max_size = backend_ctx->max_alloc_size;
    }
    return max_size;
}

static bool ggml_backend_opencl_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_opencl(backend);

    UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_opencl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_opencl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_opencl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_opencl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_opencl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ NULL,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_opencl_host_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    return "OpenCL-HostMapped";

    GGML_UNUSED(buffer_type);
}

static ggml_backend_buffer_t ggml_backend_opencl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buffer_type, size_t size) {
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer_type->device);

    size = std::max(size, (size_t)1);

    cl_int err = CL_SUCCESS;
    cl_mem mem = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_INFO("%s: failed to allocate %.2f MiB host-mapped OpenCL buffer: %d\n",
                      __func__, size / 1024.0 / 1024.0, err);
        return nullptr;
    }

    void * ptr = clEnqueueMapBuffer(
        backend_ctx->queue, mem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS || ptr == nullptr) {
        GGML_LOG_INFO("%s: failed to map %.2f MiB host-mapped OpenCL buffer: %d\n",
                      __func__, size / 1024.0 / 1024.0, err);
        CL_CHECK(clReleaseMemObject(mem));
        return nullptr;
    }

    ggml_backend_opencl_buffer_context * ctx = new ggml_backend_opencl_buffer_context(
        ggml_backend_opencl_buffer_context::host_mapped_tag_t{}, mem, ptr, backend_ctx->queue);

    return ggml_backend_buffer_init(buffer_type, ggml_backend_opencl_buffer_interface, ctx, size);
}

static bool ggml_backend_opencl_host_buffer_type_is_host(ggml_backend_buffer_type_t buffer_type) {
    return true;

    GGML_UNUSED(buffer_type);
}

static ggml_backend_buffer_type_i ggml_backend_opencl_host_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_opencl_host_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_opencl_host_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_opencl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_opencl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ NULL,
    /* .is_host          = */ ggml_backend_opencl_host_buffer_type_is_host,
};

//
// backend device
//

static const char * ggml_backend_opencl_device_get_name(ggml_backend_dev_t dev) {
    return "GPUOpenCL";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_opencl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_opencl_device_context *dev_ctx = (ggml_backend_opencl_device_context *) dev->context;
    return dev_ctx->device_name.c_str();
}

static void ggml_backend_opencl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free = 1;
    *total = 1;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_opencl_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static bool ggml_backend_opencl_host_mapped_enabled() {
    return std::getenv("GGML_OPENCL_HOST_MAPPED") != nullptr ||
           std::getenv("GGML_SCHED_RUNTIME_DISPATCH_UNIFIED_MIGRATE") != nullptr ||
           std::getenv("GGML_SCHED_RUNTIME_DISPATCH_UNIFIED_ACTIVATION") != nullptr;
}

static void ggml_backend_opencl_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_opencl_device_get_name(dev);
    props->description = ggml_backend_opencl_device_get_description(dev);
    props->type        = ggml_backend_opencl_device_get_type(dev);
    ggml_backend_opencl_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = ggml_backend_dev_caps {
        /* .async                 = */ false,
        /* .host_buffer           = */ ggml_backend_opencl_host_mapped_enabled(),
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_opencl_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(dev);
    // Getting a new reference to the backend, increase ref_count
    backend_ctx->ref_count++;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_opencl_guid(),
        /* .interface = */ ggml_backend_opencl_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_opencl_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = static_cast<ggml_backend_opencl_device_context *>(dev->context);

    dev_ctx->buffer_type = ggml_backend_buffer_type{
        /* .iface   = */ ggml_backend_opencl_buffer_type_interface,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };

    return &dev_ctx->buffer_type;
}

static ggml_backend_buffer_type_t ggml_backend_opencl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    if (!ggml_backend_opencl_host_mapped_enabled()) {
        return nullptr;
    }

    auto * dev_ctx = static_cast<ggml_backend_opencl_device_context *>(dev->context);

    dev_ctx->host_buffer_type = ggml_backend_buffer_type{
        /* .iface   = */ ggml_backend_opencl_host_buffer_type_interface,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };

    return &dev_ctx->host_buffer_type;
}

static ggml_backend_buffer_t ggml_backend_opencl_device_buffer_from_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_opencl_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return ggml_opencl_supports_op(dev, op);
}

static bool ggml_backend_opencl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Check 'dev' and 'buffer_type' are not objects belonging to this backend.
    if (dev->iface.get_name != ggml_backend_opencl_device_get_name ||
        (buft->iface.get_name != ggml_backend_opencl_buffer_type_get_name &&
         buft->iface.get_name != ggml_backend_opencl_host_buffer_type_get_name)) {
        return false;
    }

    // Check cl_context is the same. clEnqueue* commands may not use
    // buffers from another cl_context.
    ggml_backend_opencl_context * backend_ctx0 = ggml_cl2_init(dev);
    ggml_backend_opencl_context * backend_ctx1 = ggml_cl2_init(buft->device);
    return backend_ctx0->context == backend_ctx1->context;
}

namespace /* anonymous */ {
struct ggml_backend_device_i ggml_backend_opencl_device_i = {
    /* .get_name             = */ ggml_backend_opencl_device_get_name,
    /* .get_description      = */ ggml_backend_opencl_device_get_description,
    /* .get_memory           = */ ggml_backend_opencl_device_get_memory,
    /* .get_type             = */ ggml_backend_opencl_device_get_type,
    /* .get_props            = */ ggml_backend_opencl_device_get_props,
    /* .init_backend         = */ ggml_backend_opencl_device_init,
    /* .get_buffer_type      = */ ggml_backend_opencl_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_opencl_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ggml_backend_opencl_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_opencl_device_supports_op,
    /* .supports_buft        = */ ggml_backend_opencl_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};
}

// Backend registry

static const char * ggml_backend_opencl_reg_get_name(ggml_backend_reg_t reg) {
    return "OpenCL";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_opencl_reg_device_count(ggml_backend_reg_t reg) {
    return g_ggml_backend_opencl_devices.size();

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_opencl_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index < ggml_backend_opencl_reg_device_count(reg));

    return &g_ggml_backend_opencl_devices[index];

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static struct ggml_backend_reg_i ggml_backend_opencl_reg_i = {
    /* .get_name         = */ ggml_backend_opencl_reg_get_name,
    /* .device_count     = */ ggml_backend_opencl_reg_device_count,
    /* .device_get       = */ ggml_backend_opencl_reg_device_get,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_opencl_reg(void) {
    static std::mutex mutex;
    static ggml_backend_reg reg;
    static bool initialized = false;
    std::lock_guard<std::mutex> lock(mutex);

    if (initialized) {
        return &reg;
    }
    initialized = true;

    g_ggml_backend_opencl_devices = ggml_opencl_probe_devices(&reg);

    reg = ggml_backend_reg{
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_opencl_reg_i,
        /* .context     = */ NULL,
    };

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_opencl_reg)

//------------------------------------------------------------------------------
// Debugging utils
//------------------------------------------------------------------------------
#if 0
#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2,
    "wrong q4_0 block size/padding");

#include <math.h>
#ifdef __cplusplus
#include "half.hpp"
#endif

static void dump_tensor(ggml_backend_t backend, const struct ggml_tensor * tensor) {
    void * buf = malloc(ggml_nbytes(tensor));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;
#ifdef GGML_OPENCL_SOA_Q
    void * buf_q;
    void * buf_d;
#endif

    // Make sure everything is done.
    CL_CHECK(clFinish(queue));

#ifdef GGML_OPENCL_SOA_Q
    if (tensor->type == GGML_TYPE_Q4_0) {
        ggml_tensor_extra_cl_q4_0 * extra = (ggml_tensor_extra_cl_q4_0 *) tensor->extra;
        GGML_ASSERT(extra);

        size_t size_q = ggml_nelements(tensor)/QK4_0 * QK4_0/2;
        size_t size_d = ggml_nelements(tensor)/QK4_0 * sizeof(ggml_fp16_t);
        GGML_ASSERT(size_q + size_d == ggml_nbytes(tensor));
        buf_q = malloc(size_q);
        buf_d = malloc(size_d);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->q, CL_TRUE, 0, size_q, buf_q, 0, NULL, NULL));
        CL_CHECK(clEnqueueReadBuffer(queue, extra->d, CL_TRUE, 0, size_d, buf_d, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    } else if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_tensor_extra_cl_mxfp4 * extra = (ggml_tensor_extra_cl_mxfp4 *) tensor->extra;
        GGML_ASSERT(extra);

        size_t size_q = ggml_nelements(tensor)/QK_MXFP4 * QK_MXFP4/2;
        size_t size_e = ggml_nelements(tensor)/QK_MXFP4 * sizeof(char);
        GGML_ASSERT(size_q + size_e == ggml_nbytes(tensor));
        buf_q = malloc(size_q);
        buf_d = malloc(size_e);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->q, CL_TRUE, 0, size_q, buf_q, 0, NULL, NULL));
        CL_CHECK(clEnqueueReadBuffer(queue, extra->d, CL_TRUE, 0, size_e, buf_d, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    } else {
        // Read out the tensor from GPU memory.
        ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
        GGML_ASSERT(extra);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->data_device, CL_TRUE,
        extra->offset, ggml_nbytes(tensor), buf, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    }
#else
    // Read out the tensor from GPU memory.
    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    GGML_ASSERT(extra);

    CL_CHECK(clEnqueueReadBuffer(queue, extra->data_device, CL_TRUE,
        extra->offset, ggml_nbytes(tensor), buf, 0, NULL, NULL));
    CL_CHECK(clFinish(queue));
#endif // GGML_OPENCL_SOA_Q

    // Open file and dump.
    char fname[512];
    snprintf(fname, sizeof(fname), "./tensor-dumps/%s.txt", tensor->name);
    FILE * f = fopen(fname, "w");
    if (!f) {
        printf("Failed to open %s\n", fname);
        return;
    }

    if (tensor->type == GGML_TYPE_F32) {
        float * data = (float *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%f\n", data[i]);
        }
    } else if (tensor->type == GGML_TYPE_I32) {
        int * data = (int *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%d\n", data[i]);
        }
    } else if (tensor->type == GGML_TYPE_F16) {
#ifdef __cplusplus
        half_float::half * data = (half_float::half *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (std::isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%f\n", float(data[i]));
        }
#endif
    } else if (tensor->type == GGML_TYPE_Q4_0) {
#ifdef GGML_OPENCL_SOA_Q
        ggml_fp16_t * data_d = (ggml_fp16_t *)buf_d;
        unsigned char * data_q = (unsigned char *)buf_q;

        for (int i = 0; i < ggml_nelements(tensor)/QK4_0; ++i) {
            fprintf(f, "%04x, ", data_d[i]);
            for (int k = 0; k < QK4_0/2; ++k) {
                fprintf(f, "%02x, ", data_q[k]);
            }
            fprintf(f, "\n");
            data_q += QK4_0/2;
        }
        free(buf_d);
        free(buf_q);
#else
        block_q4_0 * data = (block_q4_0 *) buf;
        for (int i = 0; i < ggml_nelements(tensor)/QK4_0; ++i) {
            fprintf(f, "%04x, ", data[i].d);
            for (int k = 0; k < QK4_0/2; ++k) {
                fprintf(f, "%02x, ", data[i].qs[k]);
            }
            fprintf(f, "\n");
        }
#endif // GGML_OPENCL_SOA_Q
    }
    free(buf);
    fflush(f);
    fclose(f);
}
#else
#define dump_tensor(tensor)
#endif

//------------------------------------------------------------------------------
// Ops
//------------------------------------------------------------------------------

static bool ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
            src1->type == GGML_TYPE_F32 &&
             dst->type == GGML_TYPE_F32 &&
            (ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
}

static void ggml_cl_nop(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(backend);
    UNUSED(src0);
    UNUSED(src1);
    UNUSED(dst);
}

static void ggml_cl_get_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const int      ne00 = src0->ne[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];
    const int      ne10 = src1->ne[0];
    const cl_ulong nb10 = src1->nb[0];
    const int      ne11 = src1->ne[1];
    const int      ne12 = src1->ne[2];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    switch (src0->type) {
        case GGML_TYPE_F32:
            kernel = backend_ctx->kernel_get_rows_f32;
            break;
        case GGML_TYPE_F16:
            kernel = backend_ctx->kernel_get_rows_f16;
            break;
        case GGML_TYPE_Q4_0:
            kernel = backend_ctx->kernel_get_rows_q4_0;
            break;
        default:
            GGML_ASSERT(false && "not implemented");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb3));

    size_t global_work_size[] = {(size_t)ne10*64, (size_t)ne11, (size_t)ne12};
    size_t local_work_size[] = {64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_set_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    // ne0 = ne00
    // ne2 = ne02
    // ne3 = ne03

    const int      ne01 = src0->ne[1];
    const int      ne02 = src0->ne[2];
    const int      ne03 = src0->ne[3];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int      ne11 = src1->ne[1];
    const int      ne12 = src1->ne[2];

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];

    const int      ne0  = dst->ne[0];

    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    const int nblk0 = ne0/ggml_blck_size(dst->type);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    switch (dst->type) {
        case GGML_TYPE_F32:
            if (src1->type == GGML_TYPE_I64) {
                kernel = backend_ctx->kernel_set_rows_f32_i64;
            } else {
                kernel = backend_ctx->kernel_set_rows_f32_i32;
            }
            break;
        case GGML_TYPE_F16:
            if (src1->type == GGML_TYPE_I64) {
                kernel = backend_ctx->kernel_set_rows_f16_i64;
            } else {
                kernel = backend_ctx->kernel_set_rows_f16_i32;
            }
            break;
        default:
            GGML_ABORT("not implemented");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &nblk0));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb3));

    int nth0 = 64;
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 32;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
    }

    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    while (nth0 < nblk0 && nth0 < max_workgroup_size) {
        nth0 *= 2;
    }

    int rows_per_workgroup = 1;
    if (nth0 > nblk0) {
        rows_per_workgroup = nth0 / nblk0;
        nth0 = nblk0;
    }

    size_t global_work_size[] = {
        (size_t)(ne01 + rows_per_workgroup - 1)/rows_per_workgroup*nth0,
        (size_t)ne02*rows_per_workgroup,
        (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth0, (size_t)rows_per_workgroup, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_add(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];

    const cl_ulong nb0  = dst->nb[0];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    const bool bcast_row = ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0;

    if (bcast_row) {
        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ne11 == 1);
    }

    if (dst->type == GGML_TYPE_F32) {
        GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32);
        if (bcast_row) {
            kernel = backend_ctx->kernel_add_row;
            const int ne = ne00 / 4;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
        } else {
            kernel = backend_ctx->kernel_add;
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne03));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne13));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &ne2));
            CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &ne3));
            CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb0));
            CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb1));
            CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb2));
            CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nb3));
        }
    } else if (dst->type == GGML_TYPE_F16) {
        GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
        const int type_src0 = (src0->type == GGML_TYPE_F32);
        const int type_src1 = (src1->type == GGML_TYPE_F32);
        if (bcast_row) {
            kernel = backend_ctx->kernel_add_row_f16;
            const int ne = ne00 / 4;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &type_src0));
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),      &type_src1));
        } else {
            kernel = backend_ctx->kernel_add_f16;
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne03));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne13));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &ne2));
            CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &ne3));
            CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb0));
            CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb1));
            CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb2));
            CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nb3));
            CL_CHECK(clSetKernelArg(kernel, 30, sizeof(int),      &type_src0));
            CL_CHECK(clSetKernelArg(kernel, 31, sizeof(int),      &type_src1));
        }
    } else {
        GGML_ASSERT(false && "unsupported data types for add");
    }

    if (bcast_row) {
        int n = ggml_nelements(dst)/4;
        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        size_t * local_work_size_ptr = local_work_size;
        if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
            local_work_size_ptr = nullptr;
        }

        backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size_ptr, dst);
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_add_id(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const ggml_tensor * src2 = dst->src[2];
    GGML_ASSERT(src2);
    GGML_ASSERT(src2->extra);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(src0));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];

    const cl_ulong nb11 = src1->nb[1];

    const cl_ulong nb21 = src2->nb[1];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->kernel_add_id;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb21));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne1));

    int nth = MIN(ne00, (int) backend_ctx->get_kernel_workgroup_size(kernel));
    size_t global_work_size[] = { (size_t)ne01*nth, (size_t)ne02, 1 };
    size_t local_work_size[] = { (size_t)nth, 1, 1 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mul(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3]; UNUSED(ne13);

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3]; UNUSED(nb13);

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];

    const cl_ulong nb0  = dst->nb[0];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    bool bcast_row = false;
    cl_kernel kernel;

    if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
        GGML_ASSERT(ggml_is_contiguous(src0));

        // src1 is a row
        GGML_ASSERT(ne11 == 1);

        bcast_row = true;
        int ne = ne00 / 4;

        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_mul_row;
        } else {
            kernel = backend_ctx->kernel_mul_row_f16;
        }

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
    } else {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_mul;
        } else {
            kernel = backend_ctx->kernel_mul_f16;
        }

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb00));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne13));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &ne2));
        CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &ne3));
        CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb0));
        CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nb3));
    }

    if (bcast_row) {
        int n = ggml_nelements(dst)/4;
        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        size_t * local_work_size_ptr = local_work_size;
        if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
            local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
        }

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_div(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const int ne0  = dst->ne[0];

    const cl_ulong nb0  = dst->nb[0];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    bool bcast_row = false;
    cl_kernel kernel;

    if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
        GGML_ASSERT(ggml_is_contiguous(src0));

        // src1 is a row
        GGML_ASSERT(ne11 == 1);

        bcast_row = true;
        int ne = ne00 / 4;

        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_div_row;
        } else {
            kernel = backend_ctx->kernel_div_row_f16;
        }

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
    } else {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_div;
        } else {
            kernel = backend_ctx->kernel_div_f16;
        }

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne11));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne13));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb13));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb0));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &nb3));
    }

    if (bcast_row) {
        int n = ggml_nelements(dst)/4;
        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_sub(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const int ne0  = dst->ne[0];

    const cl_ulong nb0  = dst->nb[0];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    bool bcast_row = false;
    cl_kernel kernel;

    if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
        GGML_ASSERT(ggml_is_contiguous(src0));

        // src1 is a row
        GGML_ASSERT(ne11 == 1);

        bcast_row = true;
        int ne = ne00 / 4;

        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_sub_row;
        } else {
            kernel = backend_ctx->kernel_sub_row_f16;
        }

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
    } else {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_sub;
        } else {
            kernel = backend_ctx->kernel_sub_f16;
        }

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne11));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne13));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb13));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb0));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &nb3));
    }

    if (bcast_row) {
        int n = ggml_nelements(dst)/4;
        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_gelu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    int n = ggml_nelements(dst);

    if (n % 4 == 0) {
        kernel = backend_ctx->kernel_gelu_4;
        n /= 4;
    } else {
        kernel = backend_ctx->kernel_gelu;
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_gelu_erf(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    int n = ggml_nelements(dst);

    if (n % 4 == 0) {
        kernel = backend_ctx->kernel_gelu_erf_4;
        n /= 4;
    } else {
        kernel = backend_ctx->kernel_gelu_erf;
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_gelu_quick(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    int n = ggml_nelements(dst);

    if (n % 4 == 0) {
        kernel = backend_ctx->kernel_gelu_quick_4;
        n /= 4;
    } else {
        kernel = backend_ctx->kernel_gelu_quick;
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_silu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    int n = ggml_nelements(dst);

    if (n % 4 == 0) {
        kernel = backend_ctx->kernel_silu_4;
        n /= 4;
    } else {
        kernel = backend_ctx->kernel_silu;
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_relu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->kernel_relu;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    const int64_t n = ggml_nelements(dst);

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_sigmoid(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        kernel = backend_ctx->kernel_sigmoid_f32;
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->kernel_sigmoid_f16;
    } else {
        GGML_ASSERT(false && "Unsupported data types for sigmoid (input and output must be both f32 or f16)");
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    const int64_t n = ggml_nelements(dst);

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_clamp(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    float min;
    float max;
    memcpy(&min, ((int32_t *) dst->op_params) + 0, sizeof(float));
    memcpy(&max, ((int32_t *) dst->op_params) + 1, sizeof(float));

    cl_kernel kernel = backend_ctx->kernel_clamp;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(float),    &min));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(float),    &max));

    const int64_t n = ggml_nelements(dst);

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int ne00 = src0 ? src0->ne[0] : 0;
    const int ne01 = src0 ? src0->ne[1] : 0;
    const int ne02 = src0 ? src0->ne[2] : 0;
    const int ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    const int nth = MIN(64, ne00);

    cl_kernel kernel = backend_ctx->kernel_norm;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),    &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),  &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),    &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),  &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),       &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),       &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),       &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),       &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong),  &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong),  &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),  &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(float),     &eps));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float)*nth, NULL));

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_rms_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    //ggml_backend_opencl_device_context * dev_ctx =
    //    (ggml_backend_opencl_device_context *)backend->device->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int ne00 = src0 ? src0->ne[0] : 0;
    const int ne01 = src0 ? src0->ne[1] : 0;
    const int ne02 = src0 ? src0->ne[2] : 0;
    const int ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    GGML_ASSERT(ne00 % 4 == 0);

    const int nth = MIN(64, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    cl_kernel kernel = backend_ctx->kernel_rms_norm;

    // Note, this kernel declares local memory in kernel args and the size
    // depends on subgroup size.
    // Note, this requires OpenCL 2.1 and above
    // For now we use fixed subgroup size to simplify support for OpenCL 2.0.
    size_t sgs;
    //CL_CHECK(clGetKernelSubGroupInfo(kernel, dev_ctx->device,
    //    CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
    //    sizeof(local_work_size), local_work_size,
    //    sizeof(size_t), &sgs, NULL));
    if (backend_ctx->gpu_family == ADRENO) {
        sgs = 64;
    } else if (backend_ctx->gpu_family == INTEL) {
        sgs = 32;
    } else {
        GGML_ASSERT(false && "Unsupported GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),    &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),  &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),    &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),  &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),       &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),       &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),       &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),       &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong),  &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong),  &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),  &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(float),     &eps));
    // This is local memory - the size depends on subgroup size.
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float)*nth/sgs,  NULL));

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_opencl_op_rms_norm_fused(ggml_backend_t backend, ggml_tensor * rms_norm_tensor, ggml_tensor * mul_tensor) {
    GGML_ASSERT(mul_tensor);
    GGML_ASSERT(rms_norm_tensor);

    // src0 is the src of rms_norm, src1 is the other src of mul (one being rms_norm)
    const ggml_tensor * src0 = rms_norm_tensor->src[0];
    const ggml_tensor * src1;
    if (mul_tensor->src[0] == rms_norm_tensor) {
        src1 = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == rms_norm_tensor) {
        src1 = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false && "Invalid args for rms_norm and mul");
    }
    const ggml_tensor * dst = mul_tensor;

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    float eps;
    memcpy(&eps, rms_norm_tensor->op_params, sizeof(float));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const cl_ulong nb1 = dst->nb[1];
    const cl_ulong nb2 = dst->nb[2];
    const cl_ulong nb3 = dst->nb[3];

    GGML_ASSERT(ne00 % 4 == 0);

    size_t sgs;
    if (backend_ctx->gpu_family == ADRENO) {
        sgs = 64;
    } else if (backend_ctx->gpu_family == INTEL) {
        sgs = 32;
    } else {
        GGML_ASSERT(false && "Unsupported GPU");
    }

    cl_kernel kernel = backend_ctx->kernel_rms_norm_mul;

    int nth = sgs;
    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    while (nth < ne00 && nth < max_workgroup_size) {
        nth *= 2;
    }
    nth = MIN(nth, max_workgroup_size);
    nth = MIN(nth, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),        &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),      &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),        &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),      &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),        &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong),      &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),           &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),           &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),           &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),           &ne03));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),      &nb01));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),      &nb02));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong),      &nb03));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),           &ne10));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),           &ne11));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),           &ne12));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),           &ne13));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),      &nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),      &nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),      &nb13));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong),      &nb1));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong),      &nb2));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong),      &nb3));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(float),         &eps));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(float)*nth/sgs, NULL));

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_opencl_op_norm_fused(ggml_backend_t backend, ggml_tensor * norm_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    GGML_ASSERT(norm_tensor && mul_tensor && add_tensor);

    const ggml_tensor * src0 = norm_tensor->src[0];
    const ggml_tensor * src1 = mul_tensor->src[0] == norm_tensor ? mul_tensor->src[1] : mul_tensor->src[0];
    const ggml_tensor * src2 = add_tensor->src[0] == mul_tensor ? add_tensor->src[1] : add_tensor->src[0];
    const ggml_tensor * dst = add_tensor;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    float eps;
    memcpy(&eps, norm_tensor->op_params, sizeof(float));

    const int ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const cl_ulong nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const int ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const cl_ulong nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const int ne20 = src2->ne[0], ne21 = src2->ne[1], ne22 = src2->ne[2], ne23 = src2->ne[3];
    const cl_ulong nb21 = src2->nb[1], nb22 = src2->nb[2], nb23 = src2->nb[3];
    const cl_ulong nbd1 = dst->nb[1], nbd2 = dst->nb[2], nbd3 = dst->nb[3];

    size_t sgs;
    if (backend_ctx->gpu_family == ADRENO) sgs = 64;
    else if (backend_ctx->gpu_family == INTEL) sgs = 32;
    else GGML_ASSERT(false && "Unsupported GPU");

    cl_kernel kernel = backend_ctx->kernel_norm_mul_add;

    int nth = sgs;
    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    while (nth < ne00/4 && nth < max_workgroup_size) nth *= 2;
    nth = MIN(nth, max_workgroup_size);
    nth = MIN(nth, ne00/4);

    size_t gws[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t lws[] = {(size_t)nth, 1, 1};
    size_t num_subgroups = (nth + sgs - 1) / sgs;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), &ne00));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int), &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int), &ne03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int), &ne10));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int), &ne11));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int), &ne12));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int), &ne13));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int), &ne20));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int), &ne21));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int), &ne22));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int), &ne23));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb21));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb22));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb23));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nbd1));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(cl_ulong), &nbd2));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(cl_ulong), &nbd3));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(float), &eps));
    CL_CHECK(clSetKernelArg(kernel, 33, sizeof(cl_float2) * num_subgroups, NULL));

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
}

static void ggml_opencl_op_group_norm_fused(ggml_backend_t backend, ggml_tensor * gn_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    GGML_ASSERT(gn_tensor && mul_tensor && add_tensor);

    const ggml_tensor * src0 = gn_tensor->src[0];
    const ggml_tensor * src1 = mul_tensor->src[0] == gn_tensor ? mul_tensor->src[1] : mul_tensor->src[0];
    const ggml_tensor * src2 = add_tensor->src[0] == mul_tensor ? add_tensor->src[1] : add_tensor->src[0];
    const ggml_tensor * dst = add_tensor;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    int groups;
    float eps;
    memcpy(&groups, gn_tensor->op_params, sizeof(int));
    memcpy(&eps, (char *)gn_tensor->op_params + sizeof(int), sizeof(float));

    cl_kernel kernel = backend_ctx->kernel_group_norm_mul_add;
    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    int ne = ggml_nelements(src0);
    int group_size = ne / groups;

    size_t lws[] = { (size_t)MIN(max_workgroup_size, group_size) };
    size_t gws[] = { (size_t)groups * lws[0] };

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), &ne));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &group_size));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(float), &eps));

    backend_ctx->enqueue_ndrange_kernel(kernel, 1, gws, lws, dst);
}

static void ggml_cl_group_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int32_t n_groups   = ((const int32_t *) dst->op_params)[0];
    int32_t group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + n_groups - 1) / n_groups);
    float   eps        = ((const float *) dst->op_params)[1];

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne = ne00*ne01*ne02;

    cl_kernel kernel = backend_ctx->kernel_group_norm;

    size_t sgs = 64;
    if (backend_ctx->gpu_family == ADRENO) {
        sgs = 64;
    } else if (backend_ctx->gpu_family == INTEL) {
        sgs = 32;
    } else {
        GGML_ASSERT(false && "Unsupported GPU");
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &group_size));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(float),    &eps));

    size_t global_work_size[] = {(size_t)n_groups*sgs, 1, 1};
    size_t local_work_size[] = {(size_t)sgs, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_tanh(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0_abs = extra0->offset + src0->view_offs;
    cl_ulong offsetd_abs = extrad->offset + dst->view_offs;

    cl_kernel kernel;
    if (dst->type == GGML_TYPE_F32) {
        kernel = backend_ctx->kernel_tanh_f32_nd;
    } else if (dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->kernel_tanh_f16_nd;
    } else {
        GGML_ASSERT(false && "Unsupported type for ggml_cl_tanh");
    }
    GGML_ASSERT(kernel != nullptr);

    const int ne00 = src0->ne[0]; const int ne01 = src0->ne[1]; const int ne02 = src0->ne[2]; const int ne03 = src0->ne[3];
    const cl_ulong nb00 = src0->nb[0]; const cl_ulong nb01 = src0->nb[1]; const cl_ulong nb02 = src0->nb[2]; const cl_ulong nb03 = src0->nb[3];

    const int ne10 = dst->ne[0]; const int ne11 = dst->ne[1]; const int ne12 = dst->ne[2]; const int ne13 = dst->ne[3];
    const cl_ulong nb10 = dst->nb[0]; const cl_ulong nb11 = dst->nb[1]; const cl_ulong nb12 = dst->nb[2]; const cl_ulong nb13 = dst->nb[3];

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0_abs));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd_abs));

    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),&nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),&nb03));

    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),     &ne10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),     &ne11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),     &ne12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),     &ne13));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong),&nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),&nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),&nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),&nb13));

    size_t global_work_size[3];
    if (ne10 == 0 || ne11 == 0 || ne12 == 0 || ne13 == 0) { // Handle case of 0 elements
        return;
    }
    global_work_size[0] = (size_t)ne10;
    global_work_size[1] = (size_t)ne11;
    global_work_size[2] = (size_t)ne12;

    size_t lws0 = 16, lws1 = 4, lws2 = 1;
    if (ne10 < 16) lws0 = ne10;
    if (ne11 < 4) lws1 = ne11;
    if (ne12 < 1) lws2 = ne12 > 0 ? ne12 : 1;

    while (lws0 * lws1 * lws2 > 256 && lws0 > 1) lws0 /= 2;
    while (lws0 * lws1 * lws2 > 256 && lws1 > 1) lws1 /= 2;
    while (lws0 * lws1 * lws2 > 256 && lws2 > 1) lws2 /= 2;


    size_t local_work_size[] = {lws0, lws1, lws2};

    size_t* local_work_size_ptr = local_work_size;
    if (!backend_ctx->non_uniform_workgroups) {
        if (global_work_size[0] % local_work_size[0] != 0 ||
            global_work_size[1] % local_work_size[1] != 0 ||
            global_work_size[2] % local_work_size[2] != 0) {
            local_work_size_ptr = NULL;
        }
    }
    if (global_work_size[0] == 0 || global_work_size[1] == 0 || global_work_size[2] == 0) return;

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_repeat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1_shape_def, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(dst->type == src0->type);

    UNUSED(src1_shape_def);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    if (backend_ctx->kernel_repeat == nullptr) {
        GGML_LOG_WARN("%s: repeat kernel not available, skipping OpenCL execution.\n", __func__);
        return;
    }

    ggml_tensor_extra_cl * extra_src0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra_dst  = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra_src0->offset + src0->view_offs;
    cl_ulong off_dst  = extra_dst->offset  + dst->view_offs;

    const int src0_ne0 = src0->ne[0]; const int src0_ne1 = src0->ne[1]; const int src0_ne2 = src0->ne[2]; const int src0_ne3 = src0->ne[3];
    const cl_ulong src0_nb0 = src0->nb[0]; const cl_ulong src0_nb1 = src0->nb[1]; const cl_ulong src0_nb2 = src0->nb[2]; const cl_ulong src0_nb3 = src0->nb[3];

    const int dst_ne0 = dst->ne[0]; const int dst_ne1 = dst->ne[1]; const int dst_ne2 = dst->ne[2]; const int dst_ne3 = dst->ne[3];
    const cl_ulong dst_nb0 = dst->nb[0]; const cl_ulong dst_nb1 = dst->nb[1]; const cl_ulong dst_nb2 = dst->nb[2]; const cl_ulong dst_nb3 = dst->nb[3];

    cl_kernel kernel = backend_ctx->kernel_repeat;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra_src0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),    &extra_dst->data_device));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_ulong),  &off_src0));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &off_dst));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),       &src0_ne0));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),       &src0_ne1));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),       &src0_ne2));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),       &src0_ne3));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong),  &src0_nb0));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong),  &src0_nb1));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &src0_nb2));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &src0_nb3));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &dst_ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &dst_ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &dst_ne2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &dst_ne3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &dst_nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &dst_nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &dst_nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &dst_nb3));

    size_t gws0 = dst_ne1 > 0 ? (size_t)dst_ne1 : 1;
    size_t gws1 = dst_ne2 > 0 ? (size_t)dst_ne2 : 1;
    size_t gws2 = dst_ne3 > 0 ? (size_t)dst_ne3 : 1;

    size_t global_work_size[] = { gws0, gws1, gws2 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, NULL, dst);
}

static void ggml_cl_pad(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    if (backend_ctx->kernel_pad == nullptr) {
        GGML_LOG_WARN("%s: pad kernel not available, skipping OpenCL execution.\n", __func__);
        return;
    }

    ggml_tensor_extra_cl * extra_src0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra_dst  = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra_src0->offset + src0->view_offs;
    cl_ulong off_dst  = extra_dst->offset  + dst->view_offs;

    const int s_ne0 = src0->ne[0];
    const int s_ne1 = src0->ne[1];
    const int s_ne2 = src0->ne[2];
    const int s_ne3 = src0->ne[3];

    const int s_nb0 = src0->nb[0];
    const int s_nb1 = src0->nb[1];
    const int s_nb2 = src0->nb[2];
    const int s_nb3 = src0->nb[3];

    const int d_ne0 = dst->ne[0];
    const int d_ne1 = dst->ne[1];
    const int d_ne2 = dst->ne[2];
    const int d_ne3 = dst->ne[3];

    const int d_nb0 = dst->nb[0];
    const int d_nb1 = dst->nb[1];
    const int d_nb2 = dst->nb[2];
    const int d_nb3 = dst->nb[3];

    const int lp0 = ((const int*)(dst->op_params))[0];
    const int rp0 = ((const int*)(dst->op_params))[1];
    const int lp1 = ((const int*)(dst->op_params))[2];
    const int rp1 = ((const int*)(dst->op_params))[3];
    const int lp2 = ((const int*)(dst->op_params))[4];
    const int rp2 = ((const int*)(dst->op_params))[5];
    const int lp3 = ((const int*)(dst->op_params))[6];
    const int rp3 = ((const int*)(dst->op_params))[7];

    cl_kernel kernel = backend_ctx->kernel_pad;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),    &extra_src0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),  &off_src0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),    &extra_dst->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),  &off_dst));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),       &s_ne0));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),       &s_ne1));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),       &s_ne2));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),       &s_ne3));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong),  &s_nb0));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong),  &s_nb1));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),  &s_nb2));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),  &s_nb3));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),       &d_ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),       &d_ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),       &d_ne2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),       &d_ne3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong),  &d_nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),  &d_nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),  &d_nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),  &d_nb3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),       &lp0));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),       &rp0));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),       &lp1));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),       &rp1));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),       &lp2));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),       &rp2));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(int),       &lp3));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(int),       &rp3));

    size_t lws0 = 64;
    size_t gws0 = (( (size_t)d_ne0 + lws0 - 1 ) / lws0) * lws0;

    size_t global_work_size[] = { gws0, (size_t)d_ne1, (size_t)d_ne2*d_ne3 };
    size_t local_work_size[]  = { lws0, 1, 1 };

    size_t * local_work_size_ptr = local_work_size;
     if (d_ne0 % lws0 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_upscale(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    const int mode_flags        = (ggml_scale_mode) ggml_get_op_params_i32(dst, 0);
    const ggml_scale_mode mode  = (ggml_scale_mode) (mode_flags & 0xFF);
    cl_kernel kernel = nullptr;

    if (mode == GGML_SCALE_MODE_NEAREST) {
        kernel = backend_ctx->kernel_upscale;
        if (kernel == nullptr) {
            GGML_LOG_WARN("%s: nearest upscale kernel not available, skipping OpenCL execution.\n", __func__);
            return;
        }
    } else if (mode == GGML_SCALE_MODE_BILINEAR) {
        kernel = backend_ctx->kernel_upscale_bilinear;
        if (kernel == nullptr) {
            GGML_LOG_WARN("%s: bilinear upscale kernel not available, skipping OpenCL execution.\n", __func__);
            return;
        }
    } else {
        GGML_LOG_WARN("%s: unsupported upscale mode %d, skipping OpenCL execution.\n", __func__, mode);
        return;
    }

    ggml_tensor_extra_cl * extra_src0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra_dst  = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra_src0->offset + src0->view_offs;
    cl_ulong off_dst  = extra_dst->offset  + dst->view_offs;

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];
    const int ne3 = dst->ne[3];

    float sf0 = (float)ne0 / ne00;
    float sf1 = (float)ne1 / ne01;
    float sf2 = (float)ne2 / ne02;
    float sf3 = (float)ne3 / ne03;

    float pixel_offset = 0.5f;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra_src0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong),  &off_src0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    &extra_dst->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &off_dst));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong),  &nb00));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong),  &nb01));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong),  &nb02));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong),  &nb03));

    if (mode == GGML_SCALE_MODE_NEAREST) {
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),       &ne0));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int),       &ne1));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne2));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne3));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float),    &sf0));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(float),    &sf1));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(float),    &sf2));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(float),    &sf3));
    } else if (mode == GGML_SCALE_MODE_BILINEAR) {
        if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
            sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
            sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
            pixel_offset = 0.0f;
        }

        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),       &ne00));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int),       &ne01));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne2));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne3));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(float),    &sf0));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(float),    &sf1));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(float),    &sf2));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(float),    &sf3));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(float),    &pixel_offset));
    }


    size_t dst_total_elements = (size_t)ne0 * ne1 * ne2 * ne3;
    if (dst_total_elements == 0) {
        return;
    }
    size_t global_work_size[] = { dst_total_elements, 1, 1 };
    size_t local_work_size_pref = 256;
    size_t local_work_size[] = { MIN(local_work_size_pref, dst_total_elements), 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (dst_total_elements % local_work_size[0] != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_concat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    if (backend_ctx->kernel_concat_f32_contiguous == nullptr || backend_ctx->kernel_concat_f32_non_contiguous == nullptr) {
        GGML_LOG_WARN("%s: concat kernels not available, skipping OpenCL execution.\n", __func__);
        return;
    }

    ggml_tensor_extra_cl * extra0_cl = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1_cl = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad_cl = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra0_cl->offset + src0->view_offs;
    cl_ulong off_src1 = extra1_cl->offset + src1->view_offs;
    cl_ulong off_dst  = extrad_cl->offset + dst->view_offs;

    const int32_t dim = ((const int32_t *) dst->op_params)[0];
    GGML_ASSERT(dim >= 0 && dim <= 3);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
        if (dim == 3) {

            size_t nbytes_src0 = ggml_nbytes(src0);
            size_t nbytes_src1 = ggml_nbytes(src1);

            CL_CHECK(clEnqueueCopyBuffer(queue, extra0_cl->data_device, extrad_cl->data_device,
                                         off_src0, off_dst, nbytes_src0, 0, NULL, NULL));
            CL_CHECK(clEnqueueCopyBuffer(queue, extra1_cl->data_device, extrad_cl->data_device,
                                         off_src1, off_dst + nbytes_src0, nbytes_src1, 0, NULL, NULL));
        } else {

            cl_kernel kernel = backend_ctx->kernel_concat_f32_contiguous;
            size_t global_work_size[3];

            for (int i3 = 0; i3 < dst->ne[3]; ++i3) {
                cl_ulong current_off_src0 = off_src0 + (i3 * src0->nb[3]);
                cl_ulong current_off_src1 = off_src1 + (i3 * src1->nb[3]);
                cl_ulong current_off_dst  = off_dst  + (i3 * dst->nb[3]);

                int d_ne00 = src0->ne[0]; int d_ne01 = src0->ne[1]; int d_ne02 = src0->ne[2];
                int d_ne10 = src1->ne[0]; int d_ne11 = src1->ne[1]; int d_ne12 = src1->ne[2];
                int d_ne0  = dst->ne[0];  int d_ne1  = dst->ne[1];  int d_ne2  = dst->ne[2];

                CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra0_cl->data_device));
                CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong),  &current_off_src0));
                CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    &extra1_cl->data_device));
                CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &current_off_src1));
                CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),    &extrad_cl->data_device));
                CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong),  &current_off_dst));
                CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),       &d_ne00));
                CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),       &d_ne01));
                CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),       &d_ne02));
                CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int),       &d_ne10));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &d_ne11));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &d_ne12));
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &d_ne0));
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &d_ne1));
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &d_ne2));
                CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &dim));

                global_work_size[0] = d_ne0;
                global_work_size[1] = d_ne1;
                global_work_size[2] = d_ne2;

                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, NULL, dst);
            }
        }
    } else {
        cl_kernel kernel = backend_ctx->kernel_concat_f32_non_contiguous;

        cl_long ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
        cl_ulong nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

        cl_ulong nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

        cl_long d_ne0 = dst->ne[0], d_ne1 = dst->ne[1], d_ne2 = dst->ne[2], d_ne3 = dst->ne[3];
        cl_ulong d_nb0 = dst->nb[0], d_nb1 = dst->nb[1], d_nb2 = dst->nb[2], d_nb3 = dst->nb[3];


        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra0_cl->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong),  &off_src0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    &extra1_cl->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &off_src1));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),    &extrad_cl->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong),  &off_dst));

        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_long),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_long),      &ne01));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_long),      &ne02));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_long),      &ne03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),    &nb00));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),    &nb01));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong),    &nb02));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong),    &nb03));

        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong),    &nb10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong),    &nb11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong),    &nb12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),    &nb13));

        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_long),     &d_ne0));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_long),     &d_ne1));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_long),     &d_ne2));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_long),     &d_ne3));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong),    &d_nb0));
        CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong),    &d_nb1));
        CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong),    &d_nb2));
        CL_CHECK(clSetKernelArg(kernel, 25, sizeof(cl_ulong),    &d_nb3));
        CL_CHECK(clSetKernelArg(kernel, 26, sizeof(int),      &dim));

        size_t global_work_size_nc[] = { d_ne1 > 0 ? (size_t)d_ne1 : 1,
                                         d_ne2 > 0 ? (size_t)d_ne2 : 1,
                                         d_ne3 > 0 ? (size_t)d_ne3 : 1 };

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size_nc, NULL, dst);
    }
}

static void ggml_cl_timestep_embedding(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    if (backend_ctx->kernel_timestep_embedding == nullptr) {
        GGML_LOG_WARN("%s: timestep_embedding kernel not available, skipping OpenCL execution.\n", __func__);
        return;
    }

    ggml_tensor_extra_cl * extra_src0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra_dst  = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra_src0->offset + src0->view_offs;
    cl_ulong off_dst  = extra_dst->offset  + dst->view_offs;

    const int logical_dim = dst->op_params[0];
    const int max_period  = dst->op_params[1];
    const int dst_nb1_bytes = dst->nb[1];

    cl_kernel kernel = backend_ctx->kernel_timestep_embedding;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra_src0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong),  &off_src0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    &extra_dst->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &off_dst));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),       &dst_nb1_bytes));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),       &logical_dim));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),       &max_period));

    size_t gws0 = (size_t)(((logical_dim + 1) / 2) + 1);

    size_t gws1 = (size_t)src0->ne[0];

    size_t global_work_size[] = {gws0, gws1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, NULL, dst);
}

static void ggml_cl_flash_attn(ggml_backend_t backend, const ggml_tensor * q, const ggml_tensor * k, ggml_tensor * dst) {
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    GGML_ASSERT(q->extra);
    GGML_ASSERT(k->extra);
    GGML_ASSERT(v->extra);
    GGML_ASSERT(dst->extra);
    if (mask) {
        GGML_ASSERT(mask->extra);
    }
    if (sinks) {
        GGML_ASSERT(sinks->extra);
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    const int n_q = q->ne[1];
    const int n_kv = k->ne[1];
    const int d_head_q = q->ne[0];
    const int d_head_v = v->ne[0];
    const int n_head = q->ne[2];
    const int n_head_kv = k->ne[2];
    const int n_batch = q->ne[3];

    cl_kernel kernel = NULL;

    const bool is_f16 = q->type == GGML_TYPE_F16;
    const bool is_mixed = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16;
    const std::pair<int, int> dk_dv = {d_head_q, d_head_v};

    if (n_q == 1) {
        if (is_mixed) {
            kernel = backend_ctx->kernels_flash_attn_f32_f16_q1.at(dk_dv);
        } else if (is_f16) {
            kernel = backend_ctx->kernels_flash_attn_f16_q1.at(dk_dv);
        } else {
            kernel = backend_ctx->kernels_flash_attn_f32_q1.at(dk_dv);
        }
    } else {
        if (is_mixed) {
            kernel = backend_ctx->kernels_flash_attn_f32_f16.at(dk_dv);
        } else if (is_f16) {
            kernel = backend_ctx->kernels_flash_attn_f16.at(dk_dv);
        } else {
            kernel = backend_ctx->kernels_flash_attn_f32.at(dk_dv);
        }
    }
    GGML_ASSERT(kernel != NULL);

    ggml_tensor_extra_cl * extra_q = (ggml_tensor_extra_cl *)q->extra;
    ggml_tensor_extra_cl * extra_k = (ggml_tensor_extra_cl *)k->extra;
    ggml_tensor_extra_cl * extra_v = (ggml_tensor_extra_cl *)v->extra;
    ggml_tensor_extra_cl * extra_o = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl * extra_mask = mask ? (ggml_tensor_extra_cl *)mask->extra : NULL;
    ggml_tensor_extra_cl * extra_sinks = sinks ? (ggml_tensor_extra_cl *)sinks->extra : NULL;

    cl_ulong offset_q = extra_q->offset + q->view_offs;
    cl_ulong offset_k = extra_k->offset + k->view_offs;
    cl_ulong offset_v = extra_v->offset + v->view_offs;
    cl_ulong offset_o = extra_o->offset + dst->view_offs;
    cl_mem   mask_buffer = extra_mask ? extra_mask->data_device : NULL;
    cl_ulong offset_mask = extra_mask ? extra_mask->offset + mask->view_offs : 0;
    cl_mem   sinks_buffer = extra_sinks ? extra_sinks->data_device : NULL;
    cl_ulong offset_sinks = extra_sinks ? extra_sinks->offset + sinks->view_offs : 0;

    const cl_ulong q_nb1 = q->nb[1], q_nb2 = q->nb[2], q_nb3 = q->nb[3];
    const cl_ulong k_nb1 = k->nb[1], k_nb2 = k->nb[2], k_nb3 = k->nb[3];
    const cl_ulong v_nb1 = v->nb[1], v_nb2 = v->nb[2], v_nb3 = v->nb[3];
    const cl_ulong o_nb1 = dst->nb[1], o_nb2 = dst->nb[2], o_nb3 = dst->nb[3];
    const cl_ulong mask_nb1 = mask ? mask->nb[1] : 0;
    const cl_ulong mask_nb2 = mask ? mask->nb[2] : 0;
    const cl_ulong mask_nb3 = mask ? mask->nb[3] : 0;
    const int mask_ne2 = mask ? mask->ne[2] : 0;
    const int mask_ne3 = mask ? mask->ne[3] : 0;

    float scale, max_bias, logit_softcap;
    const float * params = (const float *)dst->op_params;
    scale         = params[0];
    max_bias      = params[1];
    logit_softcap = params[2];

    const int is_causal = (mask == NULL && n_q > 1 && n_q == n_kv);

    const int n_head_log2_val = n_head > 0 ? 1u << (int)floorf(log2f((float)n_head)) : 0;
    const float n_head_log2_f = n_head_log2_val > 0 ? (float)n_head_log2_val : 1.0f;
    const float m0 = powf(2.0f, -(max_bias) / n_head_log2_f);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2_f);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra_q->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset_q));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra_k->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset_k));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extra_v->data_device));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offset_v));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem),   &extra_o->data_device));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong), &offset_o));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(float),    &scale));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int),      &n_q));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),     &n_kv));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),     &is_causal));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),     &n_head));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &q_nb1)); CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &q_nb2)); CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &q_nb3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &k_nb1)); CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &k_nb2)); CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &k_nb3));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &v_nb1)); CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &v_nb2)); CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &v_nb3));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &o_nb1)); CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &o_nb2)); CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong), &o_nb3));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(float),    &max_bias));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(float),    &m0));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(float),    &m1));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(int),      &n_head_log2_val));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(float),    &logit_softcap));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(int),      &n_head_kv));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(cl_mem),   &mask_buffer));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(cl_ulong), &offset_mask));
    CL_CHECK(clSetKernelArg(kernel, 33, sizeof(cl_ulong), &mask_nb1));
    CL_CHECK(clSetKernelArg(kernel, 34, sizeof(cl_ulong), &mask_nb2));
    CL_CHECK(clSetKernelArg(kernel, 35, sizeof(cl_ulong), &mask_nb3));
    CL_CHECK(clSetKernelArg(kernel, 36, sizeof(int),      &mask_ne2));
    CL_CHECK(clSetKernelArg(kernel, 37, sizeof(int),      &mask_ne3));
    CL_CHECK(clSetKernelArg(kernel, 38, sizeof(cl_mem),   &sinks_buffer));
    CL_CHECK(clSetKernelArg(kernel, 39, sizeof(cl_ulong), &offset_sinks));

    if (n_q == 1) {
        const size_t wg_size = 32;
        size_t local_work_size[] = { wg_size, 1 };
        size_t global_work_size[] = { wg_size, (size_t)(n_head * n_batch) };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
    } else {
        const int block_m = backend_ctx->kernels_flash_attn_bm.at(dk_dv);
        const size_t wg_size = block_m;
        size_t local_work_size[] = { wg_size, 1 };
        size_t global_work_size[] = { (size_t)((n_q + block_m - 1) / block_m) * wg_size, (size_t)(n_head * n_batch) };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_mul_mat_f16_f32_tiled(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int M = src0->ne[1];
    const int N = src1->ne[1];
    const int K = src0->ne[0];

    cl_kernel kernel = backend_ctx->kernel_mul_mat_f16_f32_tiled;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(int),      &M));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int),      &N));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),      &K));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &offsetd));

    // Tiling parameters. These need to be tuned for optimal performance.
    // They must match the #defines in the kernel mul_mat_f16_f32.cl.
    //
    // OPWM / OPWN: Output tile size per Work-Group. A work-group computes a tile of size OPWM x OPWN.
    // TPWM / TPWN: Threads per Work-group. This is the work-group size.
    // OPTM / OPTN: Output elements per Thread. Each thread computes OPTM x OPTN elements.
    //
    // The following relationships must hold:
    //   OPWM = TPWM * OPTM
    //   OPWN = TPWN * OPTN
    //
    const int OPWM = 64;
    const int OPWN = 64;
    const int TPWM = 16;
    const int TPWN = 8;

    size_t local_work_size[2] = { TPWM, TPWN };
    size_t global_work_size[2] = {
        (size_t) ((M + OPWM - 1) / OPWM) * TPWM,
        (size_t) ((N + OPWN - 1) / OPWN) * TPWN,
    };

    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
}

static void ggml_cl_conv_2d(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const cl_uint Cout = ne03; const cl_uint Cin = ne02; const cl_uint N = ne13;
    const cl_uint KW = ne00; const cl_uint KH = ne01; const cl_uint W = ne10; const cl_uint H = ne11; const cl_uint OW = ne0; const cl_uint OH = ne1;

    const cl_uint s0 = dst->op_params[0]; const cl_uint s1 = dst->op_params[1];
    const cl_uint p0 = dst->op_params[2]; const cl_uint p1 = dst->op_params[3];
    const cl_uint d0 = dst->op_params[4]; const cl_uint d1 = dst->op_params[5];

    const cl_uint cl_nb01 = nb01/ggml_type_size(src0->type); const cl_uint cl_nb02 = nb02/ggml_type_size(src0->type); const cl_uint cl_nb03 = nb03/ggml_type_size(src0->type);
    const cl_uint cl_nb11 = nb11/ggml_type_size(src1->type); const cl_uint cl_nb12 = nb12/ggml_type_size(src1->type); const cl_uint cl_nb13 = nb13/ggml_type_size(src1->type);
    const cl_uint cl_nb1 = nb1/ggml_type_size(dst->type); const cl_uint cl_nb2 = nb2/ggml_type_size(dst->type); const cl_uint cl_nb3 = nb3/ggml_type_size(dst->type);

    const int64_t NPQ = (int64_t)N * OW * OH;

    const uint32_t BS_K = 64;
    const uint32_t BS_NPQ = 64;
    const uint32_t BS_CRS = 16;
    const uint32_t VEC_SIZE = 4;

    const uint32_t TS_K = 4;
    const uint32_t TS_NPQ = 8;

    const uint32_t WG_K = BS_K / TS_K;
    const uint32_t WG_NPQ = BS_NPQ / TS_NPQ;

    auto splitWork = [](uint32_t work_size, uint32_t block_size) { return (block_size + work_size - 1) / block_size; };
    const uint32_t NB_K = splitWork(Cout, BS_K);
    const uint32_t NB_NPQ = splitWork(NPQ, BS_NPQ);

    cl_kernel kernel;
    size_t shmem_size;

    if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        kernel = backend_ctx->kernel_conv_2d_f16;
        shmem_size = (size_t)(BS_K * BS_CRS * sizeof(cl_half) + BS_CRS * (BS_NPQ / VEC_SIZE) * sizeof(cl_half4));
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        kernel = backend_ctx->kernel_conv_2d_f32;
        shmem_size = (size_t)(BS_K * BS_CRS * sizeof(cl_float) + BS_CRS * (BS_NPQ / VEC_SIZE) * sizeof(cl_float4));
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        kernel = backend_ctx->kernel_conv_2d_f16_f32;
        shmem_size = (size_t)(BS_K * BS_CRS * sizeof(cl_half) + BS_CRS * (BS_NPQ / VEC_SIZE) * sizeof(cl_float4));
    } else {
        GGML_ASSERT(false && "Unsupported data type combination for conv2d");
    }

    cl_uint idx = 0;
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem), &extra0->data_device)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem), &extra1->data_device)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem), &extrad->data_device)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, idx++, shmem_size, NULL));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &Cout)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &Cin)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &N));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &KW)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &KH)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &W)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &H));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &OW)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &OH));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &s0)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &s1)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &p0)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &p1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &d0)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &d1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb01)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb02)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb03));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb11)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb12)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb13));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb1)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb2)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb3));

    size_t global_work_size[] = { (size_t)NB_K * WG_K, (size_t)NB_NPQ * WG_NPQ, 1 };
    size_t local_work_size[] = { (size_t)WG_K, (size_t)WG_NPQ, 1 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
}

static void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
    const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)src0->extra;
    ggml_tensor_extra_cl_mxfp4 * extra0_mxfp4 = (ggml_tensor_extra_cl_mxfp4 *)src0->extra;
    ggml_tensor_extra_cl_q8_0 * extra0_q8_0 = (ggml_tensor_extra_cl_q8_0 *)src0->extra;
#endif

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;
    const int  ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    const int  ne10 = src1 ? src1->ne[0] : 0;
    const int  ne11 = src1 ? src1->ne[1] : 0;
    const int  ne12 = src1 ? src1->ne[2] : 0;
    const int  ne13 = src1 ? src1->ne[3] : 0;

    const cl_ulong nb10 = src1 ? src1->nb[0] : 0;
    const cl_ulong nb11 = src1 ? src1->nb[1] : 0;
    const cl_ulong nb12 = src1 ? src1->nb[2] : 0;
    const cl_ulong nb13 = src1 ? src1->nb[3] : 0;

    const int  ne0 = dst ? dst->ne[0] : 0;
    const int  ne1 = dst ? dst->ne[1] : 0;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    GGML_ASSERT(ne00 == ne10);

    int nth0 = 32;
    int nth1 = 1;
    int nrows = 1;
    // The number of values produced by each subgroup
    int ndst = 4;

    cl_kernel kernel;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    cl_context context = backend_ctx->context;

    if (ne01 && ne1 && use_adreno_kernels(backend_ctx, src0)) {

    // init CL objects
    // <--------------------------------------------> //
    cl_int              status;
    cl_image_format     img_fmt_1d;
    cl_image_desc       img_desc_1d;
    cl_buffer_region    region;
    cl_mem              A_image1d = nullptr;
    cl_mem              B_image1d = nullptr;
    cl_mem              B_sub_buffer = nullptr;
    cl_mem              C_d = nullptr;
    // for B transpose
    cl_mem B_d = nullptr;
    cl_mem B_d_input_image = nullptr;
    // <--------------------------------------------> //

    // define matrix dimensions
    // <--------------------------------------------> //
    int M = ne01;
    int N = ne1;
    int K = ne00;
    int padding;
    // <--------------------------------------------> //

    // q8_0 x fp32 decode fast path. Some Adreno drivers accept the
    // IMAGE1D_BUFFER objects but produce wrong logits, so keep the safe buffer
    // kernel as the Adreno default.  The env override is left for experiments.
    static const bool s_disable_q8_image_fast_path_env = []() {
        const char * e = std::getenv("GGML_OPENCL_DISABLE_Q8_IMAGE_FAST_PATH");
        return e && *e && *e != '0';
    }();
    static const bool s_enable_q8_image_fast_path_env = []() {
        const char * e = std::getenv("GGML_OPENCL_ENABLE_Q8_IMAGE_FAST_PATH");
        return e && *e && *e != '0';
    }();
    const bool disable_q8_image_fast_path =
        s_disable_q8_image_fast_path_env ||
        (backend_ctx->gpu_family == ADRENO && !s_enable_q8_image_fast_path_env);
    if (!disable_q8_image_fast_path && src0t == GGML_TYPE_Q8_0 && src1t == GGML_TYPE_F32 && N == 1) {
        img_fmt_1d = { CL_R, CL_UNSIGNED_INT32 };
        memset(&img_desc_1d, 0, sizeof(img_desc_1d));
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = M * K / 4;
        img_desc_1d.buffer = extra0_q8_0->q;
        A_image1d = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt_1d, &img_desc_1d, NULL, &status);
        cl_int q8_image_status = status;

        if (q8_image_status == CL_SUCCESS) {
            region.origin = offset1;
            region.size = K * N * sizeof(float);
            B_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
            q8_image_status = status;
        }

        if (q8_image_status == CL_SUCCESS) {
            img_fmt_1d = { CL_RGBA, CL_FLOAT };
            memset(&img_desc_1d, 0, sizeof(img_desc_1d));
            img_desc_1d.image_width = K * N / 4;
            img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc_1d.buffer = B_sub_buffer;
            B_image1d = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt_1d, &img_desc_1d, NULL, &status);
            q8_image_status = status;
        }

        if (q8_image_status == CL_SUCCESS) {
            kernel = backend_ctx->kernel_gemv_noshuffle_q8_0_f32;

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &A_image1d));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &B_image1d));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));

            const size_t wavesize = backend_ctx->adreno_wave_size;
            const size_t n_simdgroup = 16;
            size_t local_work_size[3] = { wavesize, n_simdgroup, 1 };
            size_t global_work_size[3] = { (size_t) CEIL_DIV(M, wavesize) * wavesize, n_simdgroup, 1 };

            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

            CL_CHECK(clReleaseMemObject(A_image1d));
            CL_CHECK(clReleaseMemObject(B_sub_buffer));
            CL_CHECK(clReleaseMemObject(B_image1d));

            return;
        }

        static std::once_flag q8_image_fallback_once;
        std::call_once(q8_image_fallback_once, [&]() {
            std::fprintf(stderr,
                         "ggml_opencl: Q8_0 image fast path unavailable, falling back to buffer matvec "
                         "(status=%d tensor=%s M=%d K=%d N=%d width=%zu)\n",
                         (int) q8_image_status,
                         src0 && src0->name[0] ? src0->name : "(unnamed)",
                         M,
                         K,
                         N,
                         (size_t) M * (size_t) K / 4);
        });
        if (A_image1d) {
            CL_CHECK(clReleaseMemObject(A_image1d));
        }
        if (B_sub_buffer) {
            CL_CHECK(clReleaseMemObject(B_sub_buffer));
        }
        if (B_image1d) {
            CL_CHECK(clReleaseMemObject(B_image1d));
        }
    }

    // q4_0 x fp32
    if(src0t == GGML_TYPE_Q4_0 && src1t == GGML_TYPE_F32) {
        // TODO: remove duplicate definitions of image description + format -- move to top

        // create an image for A
        // <--------------------------------------------> //
        if (N == 1) {
            img_fmt_1d = { CL_R, CL_UNSIGNED_INT32};
        } else {
            img_fmt_1d = { CL_R, CL_FLOAT};
        }
        memset(&img_desc_1d, 0, sizeof(img_desc_1d));
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = M * K / 2 / 4;    // Divide by 4 for char -> float
        img_desc_1d.buffer = extra0_q4_0->q;
        A_image1d = clCreateImage(
            context,
            CL_MEM_READ_ONLY,
            &img_fmt_1d,
            &img_desc_1d,
            NULL,
            &status);
        if (status != CL_SUCCESS) {
            static const bool s_elastic_q4_debug = []() {
                const char * e = std::getenv("GGML_ELASTIC_Q4_IMAGE_DEBUG");
                return e && *e && *e != '0';
            }();
            if (s_elastic_q4_debug) {
                std::fprintf(stderr,
                             "[elastic-q4-image] clCreateImage failed status=%d tensor=%s q=%p d=%p parent=%p "
                             "wbm_idx=%d size_q=%zu size_d=%zu M=%d K=%d N=%d width=%zu\n",
                             (int) status,
                             src0 && src0->name[0] ? src0->name : "(unnamed)",
                             (void *) extra0_q4_0->q,
                             (void *) extra0_q4_0->d,
                             (void *) extra0_q4_0->parent_buffer,
                             extra0_q4_0->wbm_idx,
                             extra0_q4_0->size_q,
                             extra0_q4_0->size_d,
                             (int) M,
                             (int) K,
                             (int) N,
                             (size_t) img_desc_1d.image_width);
            }
        }
        CL_CHECK(status);
        // <--------------------------------------------> //


        if (N != 1) {
            // create a sub_buffer for B
            // <--------------------------------------------> //
            region.origin = offset1;
            region.size = K * N * sizeof(float);
            B_sub_buffer = clCreateSubBuffer(
                extra1->data_device,
                0,
                CL_BUFFER_CREATE_TYPE_REGION,
                &region,
                &status);
            CL_CHECK(status);
            // <--------------------------------------------> //
        }

        // transpose activation for Skyler's gemm
        if (N != 1) {
            //how many extra elements beyond multiple of 8
            int extra_elements = N % 8;

            //how much padding to add
            padding = 0;
            if (extra_elements > 0){
                padding = 8 - extra_elements;
            }

            // Specify the starting offset (in bytes)
            region.origin = 0;
            // Specify the size of the sub-buffer (divide by 2 for FP16)
            region.size = K * (N + padding) * sizeof(float)/2;
            B_d = clCreateSubBuffer(
                backend_ctx->B_d_max,
                0,
                CL_BUFFER_CREATE_TYPE_REGION,
                &region,
                &status);
            CL_CHECK(status);

            cl_image_format image_format_B_d_input = { CL_RGBA, CL_FLOAT };
            cl_image_desc image_desc_B_d_input = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(K * N / 4),
                0, 0, 0, 0, 0, 0, 0, { B_sub_buffer }
            };
            B_d_input_image = clCreateImage(
                context,
                0,
                &image_format_B_d_input,
                &image_desc_B_d_input,
                NULL,
                &status);
            CL_CHECK(status);

            cl_image_format image_format_B_d_output = { CL_RGBA, CL_HALF_FLOAT }; //(CL_HALF_FLOAT for FP16)
            cl_image_desc image_desc_B_d_output = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(K * (N + padding)/4),
                0, 0, 0, 0, 0, 0, 0, { B_d }
            };
            B_image1d = clCreateImage(
                context,
                0,
                &image_format_B_d_output,
                &image_desc_B_d_output,
                NULL,
                &status);
            CL_CHECK(status);

            int height_B = N/4;
            if (height_B == 0) {
                height_B = 1;
            }
            int width_B = K/4;
            int padded_height_B = (N + padding)/4;

            kernel = backend_ctx->kernel_transpose_32_16;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &B_d_input_image));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_image1d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

            size_t local_size_t[2] = { 1, 16 };
            //WGS tuning
            if (ne0 == 4096 && ne1 == 128 && ne10 == 4096) {
                local_size_t[0]=4;
                local_size_t[1]=8;
            } else if (ne0 == 11008 && ne1 == 128 && ne10 == 4096) {
                local_size_t[0]=2;
                local_size_t[1]=8;
            } else if(ne0 == 4096 && ne1 == 128 && ne10 == 11008) {
                local_size_t[0]=1;
                local_size_t[1]=8;
            } else if(ne0 == 32000 && ne1 == 128 && ne10 == 4096) {
                local_size_t[0]=2;
                local_size_t[1]=8;
            }

            size_t global_size_t[2] = {
                static_cast<size_t>(width_B),
                static_cast<size_t>(padded_height_B)
            };

            backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_size_t, local_size_t, dst);
        }

        // choose gemm or gemv kernel
        // <--------------------------------------------> //
        if (N == 1) {
            kernel = backend_ctx->kernel_gemv_noshuffle_q4_0_f32;
            if (M == 4096 && K == 4096) {
                kernel = backend_ctx->kernel_gemv_noshuffle_q4_0_f32_4096_1_4096;
            } else if (M == 4096 && K == 11008) {
                kernel = backend_ctx->kernel_gemv_noshuffle_q4_0_f32_4096_1_11008;
            } else if (M == 11008 && K == 4096) {
                kernel = backend_ctx->kernel_gemv_noshuffle_q4_0_f32_11008_1_4096;
            } else if (M == 32000 && K == 4096) {
                kernel = backend_ctx->kernel_gemv_noshuffle_q4_0_f32_32000_1_4096;
            }
        } else {
            kernel = backend_ctx->CL_mul_mat_Ab_Bi_8x4;
        }
        // <--------------------------------------------> //

        // set kernel args
        // <--------------------------------------------> //
        cl_uint k_arg = 0;

        if (N == 1) {
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &A_image1d));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &r3));
        } else {
            region.origin = extrad->offset; // Specify the starting offset (in bytes)
            region.size = M * N * sizeof(float); // Specify the size of the sub-buffer
            C_d = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
            CL_CHECK(status);

            int padded_N = ne1 + padding;

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0_q4_0->q)); //A_q_dextra0_q4_0->q
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra0_q4_0->d)); //A_s_d
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &B_image1d)); //B_d
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &C_d)); //C_d
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &ne01)); //M
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),    &padded_N)); //N with padding
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),    &ne00)); //K
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),    &ne1)); //N without padding
        }
        // <--------------------------------------------> //

        // choose workgroup size
        // <--------------------------------------------> //
        size_t global_work_size[3] = {
            64, static_cast<size_t>((M+63)/64), static_cast<size_t>((N+31)/32)};
        size_t local_work_size[3] = {64, 2, 4};

        global_work_size[0] = (size_t)(ceil((float)ne1/8));
        global_work_size[1] = (size_t)(ne01/4);
        global_work_size[2] = (size_t)(1);

        local_work_size[0]  = (size_t)(1); //4x32 for FP32
        local_work_size[1]  = (size_t)(128);
        local_work_size[2]  = (size_t)(1);

        //WGS tuning
        if (ne0 == 4096 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 1;
            local_work_size[1] = 128;
        } else if (ne0 == 11008 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        } else if (ne0 == 4096 && ne1 == 128 && ne10 == 11008) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        } else if (ne0 == 32000 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        }

        if (N == 1) {
            constexpr size_t q4_gemv_n_simgroup = 4;

            size_t wavesize = backend_ctx->adreno_wave_size;
            local_work_size[0] = wavesize; // localsize
            local_work_size[1] = q4_gemv_n_simgroup; // must match N_SIMDGROUP in gemv_noshuffle_q4_0_f32.cl
            local_work_size[2] = 1;

            global_work_size[0] = (((M / 2) + wavesize - 1) / wavesize) * wavesize;
            global_work_size[1] = q4_gemv_n_simgroup;
            global_work_size[2] = 1;
        }
        // <--------------------------------------------> //

        // enqueue kernel with profiling
        // <--------------------------------------------> //
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
        // <--------------------------------------------> //

        // deallocate sub buffers and images
        // <--------------------------------------------> //
        CL_CHECK(clReleaseMemObject(A_image1d));
        if (B_sub_buffer != nullptr) {
            CL_CHECK(clReleaseMemObject(B_sub_buffer));
        }
        if (B_image1d != nullptr) {
            CL_CHECK(clReleaseMemObject(B_image1d));
        }

        if (N != 1) {
            CL_CHECK(clReleaseMemObject(B_d));
            CL_CHECK(clReleaseMemObject(B_d_input_image));
            CL_CHECK(clReleaseMemObject(C_d));
        }
        // <--------------------------------------------> //

        return;
    }
    } // if (ne01 && ne1)
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    // GEMM using local memory
    // Current BK = 16, so ne00 % 16 == 0
    if (ggml_is_contiguous(src0) &&
        ggml_is_contiguous(src1) &&
        src1t == GGML_TYPE_F32 &&
        ne00 % 16 == 0 &&
        ne11 > 1) {
        switch(src0t) {
            case GGML_TYPE_F32: {
                kernel = backend_ctx->kernel_mul_mm_f32_f32_l4_lm;
                nth0 = 128; // calculated as (BM*BN)/(TM*TN)

                int batch_stride_a = ne00*ne01;
                int batch_stride_b = ne10*ne11;
                int batch_stride_d = ne0*ne1;

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
                CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
                CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
                CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
                CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
                CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
                CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
                CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

                // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
                size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
                size_t local_work_size[] = {(size_t)nth0, 1, 1};

                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                return;
            }
            case GGML_TYPE_F16: {
                kernel = backend_ctx->kernel_mul_mm_f16_f32_l4_lm;
                nth0 = 128; // calculated as (BM*BN)/(TM*TN)

                int batch_stride_a = ne00*ne01;
                int batch_stride_b = ne10*ne11;
                int batch_stride_d = ne0*ne1;

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
                CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
                CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
                CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
                CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
                CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
                CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
                CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

                // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
                size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
                size_t local_work_size[] = {(size_t)nth0, 1, 1};

                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                return;
            }
            case GGML_TYPE_Q8_0: {
                if (ne11 < 32) {
                    break;
                }
                kernel = backend_ctx->kernel_mul_mm_q8_0_f32_l4_lm;
                nth0 = 128; // calculated as (BM*BN)/(TM*TN)

                int batch_stride_a = ne00*ne01;
                int batch_stride_b = ne10*ne11;
                int batch_stride_d = ne0*ne1;

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q8_0->q));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
                CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
                CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
                CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
                CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
                CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
                CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
                CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

                // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
                size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
                size_t local_work_size[] = {(size_t)nth0, 1, 1};

                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                return;
            }
            default:
                break;
        }
    }

    if (src0t == GGML_TYPE_F16 && src1t == GGML_TYPE_F32 &&
        src0->ne[1] > 32 &&   // M > 32
        src1->ne[1] > 32 &&   // N > 32
        src0->ne[0] > 32 &&   // K > 32
        src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
        backend_ctx->kernel_mul_mat_f16_f32_tiled != NULL) {
        ggml_cl_mul_mat_f16_f32_tiled(backend, src0, src1, dst);
        return;
    }

    if (!ggml_is_transposed(src0) &&
        !ggml_is_transposed(src1) &&
        src1t == GGML_TYPE_F32 &&
        ne00%32 == 0 &&
        ne11 > 2) {
#ifdef GGML_OPENCL_SOA_Q
        // Set up kernel.
        switch(src0t) {
            case GGML_TYPE_Q4_0:
                // This should have been satisfied.
                GGML_ASSERT(ne11 == ne1);
                GGML_ASSERT(ne01 == ne0);

                if (backend_ctx->gpu_family == INTEL) {
                    nth0 = 16;
                    nth1 = 1;

                    kernel = backend_ctx->kernel_mul_mat_q4_0_f32_1d_16x_flat;
                } else if (backend_ctx->gpu_family == ADRENO) {
                    nth0 = 64;
                    nth1 = 1;

                    kernel = backend_ctx->kernel_mul_mat_q4_0_f32_1d_8x_flat;
                } else {
                    GGML_ASSERT(false && "TODO: Unknown GPU");
                }

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
                CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
                CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
                CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
                break;
            default:
                break;
        }

        // Launch kernel.
        if (src0t == GGML_TYPE_Q4_0) {
            size_t global_work_size[] = {(size_t)(ne01 + 7)/8*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
            size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

            if (backend_ctx->gpu_family == INTEL) {
                // Set global size for Intel. It uses 16x output values.
                global_work_size[0] = (size_t)(ne01 + 15)/16*nth0;
                global_work_size[1] = (size_t)ne11*nth1;
                global_work_size[2] = (size_t)ne12*ne13;
            }

            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
            return;
        }
#else // GGML_OPENCL_SOA_Q
        // TODO: add block_q4_0 variant.
#endif // GGML_OPENCL_SOA_Q
    }

    // use custom matrix x vector kernel
    switch (src0t) {
        case GGML_TYPE_F32:
            //GGML_ASSERT(ne02 == ne12);
            GGML_ASSERT(src1t == GGML_TYPE_F32);
            kernel = backend_ctx->kernel_mul_mat_f32_f32;
            nrows = 4;

            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 32;
                nth1 = 1;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
            break;
        case GGML_TYPE_F16:
            //GGML_ASSERT(ne02 == ne12);
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 32;
                nth1 = 1;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            if (src1t == GGML_TYPE_F32) {
                if (ne11 * ne12 < 4) {
                    kernel = backend_ctx->kernel_mul_mat_f16_f32_1row;
                } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
                    kernel = backend_ctx->kernel_mul_mat_f16_f32_l4;
                    nrows = ne11;
                } else {
                    kernel = backend_ctx->kernel_mul_mat_f16_f32;
                    nrows = 4;
                }
            } else {
                kernel = backend_ctx->kernel_mul_mat_f16_f16;
                nrows = 4;
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
            break;
        case GGML_TYPE_Q4_0:
            // This should have been satisfied.
            GGML_ASSERT(ne11 == ne1);
            GGML_ASSERT(ne01 == ne0);

#ifdef GGML_OPENCL_SOA_Q
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 1;

                kernel = backend_ctx->kernel_mul_mat_q4_0_f32_8x_flat;
                ndst = 8;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;

                kernel = backend_ctx->kernel_mul_mat_q4_0_f32_8x_flat;
                ndst =8;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
#else // GGML_OPENCL_SOA_Q
            if (backend_ctx->gpu_family == INTEL) {
                // Use 1D local size. Each workgroup is a SIMD group. Each SIMD
                // group produces N_DST (4 for Q4_0 kernel) values in the result.
                // The number of workgroups on dim 0 (the leading dimension) is
                // the nearest multiple of 4 that covers ne0 (equals ne01).
                nth0 = 16;
                nth1 = 1;

                kernel = backend_ctx->kernel_mul_mat_q4_0_f32;
                ndst = 4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;

                kernel = backend_ctx->kernel_mul_mat_q4_0_f32_v;
                ndst = 4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
#endif // GGML_OPENCL_SOA_Q
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0: {
#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_q8_0_f32_flat;

            // nth0 - subgroup size
            // nth1 - number of subgroups per workgroup
            // ndst - number of output values per workgroup = output per subgroup * number of subgroups
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 2;
                ndst = nth1*4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 2;
                ndst = nth1*4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q8_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
#else
            kernel = backend_ctx->kernel_mul_mv_q8_0_f32;

            // nth0 - subgroup size
            // nth1 - number of subgroups per workgroup
            // ndst - number of output values per workgroup = output per subgroup * number of subgroups
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 2;
                ndst = nth1*4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 2;
                ndst = nth1*4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
#endif // GGML_OPENCL_SOA_Q
            break;
        }
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            kernel = backend_ctx->kernel_mul_mv_q6_K_f32;

            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 2;
                nth1 = 16;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 2;
                nth1 = 64;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
            break;
        case GGML_TYPE_MXFP4: {
#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_mxfp4_f32_flat;

            cl_mem q;
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 2;
                ndst = nth1*2;

                q = extra0_mxfp4->q;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 2;
                ndst = nth1*2;

                q = extra0_mxfp4->q_img;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_mxfp4->e));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r3));
#else
            kernel = backend_ctx->kernel_mul_mv_mxfp4_f32;

            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 2;
                ndst = nth1*2;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 2;
                ndst = nth1*2;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r3));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(float)*nth0,nullptr));
#endif
            break;
        }
        default:
            GGML_ASSERT(false && "not implemented");
    }

    if (src0t == GGML_TYPE_Q4_0 || src0t == GGML_TYPE_MXFP4 ||
        src0t == GGML_TYPE_Q4_1 ||
        src0t == GGML_TYPE_Q8_0 ||
        src0t == GGML_TYPE_Q2_K) {
        // Each SIMD group produces N_DST values in the result. Assuming each
        // workgroup has N_SIMDGROUP SIMD groups, then each workgroup will
        // produce N_DST*N_SIMDGROUP values in the result. Hence, the grid size
        // (number of workgroups) will be a nearest multiple of
        // N_DST*N_SIMDGROUP to cover the size of the dimension. Below, 4 is
        // N_DST*N_SIMDGROUP (see the kernel for Q4_0 matmul).
        size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
        size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else if (src0t == GGML_TYPE_Q4_K) {
        GGML_ASSERT(false && "not implemented");
    } else if (src0t == GGML_TYPE_Q3_K) {
        GGML_ASSERT(false && "not implemented");
    } else if (src0t == GGML_TYPE_Q5_K) {
        GGML_ASSERT(false && "not implemented");
    } else if (src0t == GGML_TYPE_Q6_K) {
        size_t global_work_size[] = {(size_t)(ne01+1)/2*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
        size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else {
        int64_t ny = (ne11 + nrows - 1)/nrows;

        size_t global_work_size[] = {(size_t)ne01*nth0, (size_t)ny*nth1, (size_t)ne12*ne13};
        size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_mul_mat_id(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const ggml_tensor * src2 = dst->src[2];
    GGML_ASSERT(src2);
    GGML_ASSERT(src2->extra);
    const ggml_tensor * id_mask = dst->src[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extra_mask = id_mask ? (ggml_tensor_extra_cl *)id_mask->extra : extra2;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offset_mask = id_mask ? extra_mask->offset + id_mask->view_offs : 0;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_UNUSED(offset0);

#ifdef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)src0->extra;
    ggml_tensor_extra_cl_mxfp4 * extra0_mxfp4 = (ggml_tensor_extra_cl_mxfp4 *)src0->extra;
    ggml_tensor_extra_cl_q8_0 * extra0_q8_0 = (ggml_tensor_extra_cl_q8_0 *)src0->extra;
#endif

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const int ne20 = src2->ne[0];
    const int ne21 = src2->ne[1];

    const cl_ulong nb21 = src2->nb[1];
    const cl_ulong nb20 = src2->nb[0];

    UNUSED(nb20);

    const cl_ulong mask_nb1 = id_mask ? id_mask->nb[1] : 0;
    const cl_ulong mask_nb2 = id_mask ? id_mask->nb[2] : 0;
    const int has_id_mask = id_mask != nullptr;

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];

    const int r2 = ne12/ne02;
    const int r3 = ne13/ne03;
    const int dst_rows = ne20*ne21; // ne20 = n_used_experts, ne21 = n_rows

    GGML_ASSERT(ne00 == ne10);

    elastic_moe_q4_cache_view moe_cache_view;
    const bool use_moe_q4_cache = src0->type == GGML_TYPE_Q4_0 &&
        ggml_opencl_prepare_moe_q4_cache(backend_ctx, src0, src2, offset2,
                                         id_mask, extra_mask->data_device, offset_mask, mask_nb1,
                                         &moe_cache_view);
    cl_mem expert_slots = use_moe_q4_cache ? moe_cache_view.route_slots : extra2->data_device;
    const int has_expert_slots = use_moe_q4_cache ? 1 : 0;
    const cl_ulong expert_slot_stride = use_moe_q4_cache ? moe_cache_view.slot_stride : 0;
    const cl_ulong expert_q_offset = use_moe_q4_cache ? moe_cache_view.q_offset : 0;

    int sgs   = 32; // subgroup size
    int nsg   = 1;  // number of subgroups
    int nrows = 1;  // number of row in src1
    int ndst  = 4;  // number of values produced by each subgroup

    cl_kernel kernel;

    // subgroup mat vec
    switch (src0->type) {
        case GGML_TYPE_Q4_0: {
            kernel = backend_ctx->kernel_mul_mv_id_q4_0_f32_8x_flat;

            cl_mem q_buffer = use_moe_q4_cache ? moe_cache_view.parent : extra0_q4_0->q;
            cl_mem d_buffer = use_moe_q4_cache ? moe_cache_view.parent : extra0_q4_0->d;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 1;
                ndst = 8;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 1;
                ndst = 8;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q_buffer));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &d_buffer));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &r3));
            CL_CHECK(clSetKernelArg(kernel, 25, sizeof(cl_mem),   &extra_mask->data_device));
            CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &offset_mask));
            CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &mask_nb1));
            CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &mask_nb2));
            CL_CHECK(clSetKernelArg(kernel, 29, sizeof(int),      &has_id_mask));
            CL_CHECK(clSetKernelArg(kernel, 30, sizeof(cl_mem),   &expert_slots));
            CL_CHECK(clSetKernelArg(kernel, 31, sizeof(int),      &has_expert_slots));
            CL_CHECK(clSetKernelArg(kernel, 32, sizeof(cl_ulong), &expert_slot_stride));
            CL_CHECK(clSetKernelArg(kernel, 33, sizeof(cl_ulong), &expert_q_offset));

            break;
        }
        case GGML_TYPE_Q8_0: {
#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_id_q8_0_f32_flat;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 2;
                ndst = 4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q8_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_mem),   &extra_mask->data_device));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &offset_mask));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &mask_nb1));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong), &mask_nb2));
            CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &has_id_mask));
#else
            kernel = backend_ctx->kernel_mul_mv_id_q8_0_f32;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 2;
                ndst = 4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_mem),   &extra_mask->data_device));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &offset_mask));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &mask_nb1));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong), &mask_nb2));
            CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &has_id_mask));
#endif // GGML_OPENCL_SOA_Q
            break;
        }
        case GGML_TYPE_MXFP4: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (!id_mask && use_adreno_moe_kernels(backend_ctx, src0)) {
                cl_int status;

                size_t local_size[3] = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                cl_mem src1_sub_buffer, buf_src1_image, buf_src2;

                int tile_size = 320;
                if (ne12 == 1) { // for gemv
                    kernel = backend_ctx->kernel_gemv_moe_mxfp4_f32;

                    // create a sub_buffer for src2
                    cl_buffer_region region;
                    region.origin = offset2;
                    region.size = ne20 * ne21 * sizeof(int);
                    buf_src2 = clCreateSubBuffer(extra2->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(ne01);
                    global_size[1] = 4;
                    global_size[2] = static_cast<size_t>(ne20);
                    local_size[1] = 4;
                } else { // for gemm
                    kernel = backend_ctx->kernel_gemm_moe_mxfp4_f32;

                    // preprocess router table
                    int num_tiles_per_expert = (ne01 + tile_size - 1) / tile_size;
                    void * host_src2_reorder = malloc(ne20 * ne21 * 4 * num_tiles_per_expert * sizeof(short));
                    void * host_src2 = malloc(ne21 * nb21);
                    CL_CHECK(clEnqueueReadBuffer(backend_ctx->queue, extra2->data_device, CL_TRUE, offset2, ne21 * nb21, host_src2, 0, NULL, NULL));
                    int total_experts = nb21 / nb20;
                    int out_idx = 0;
                    for (int i_expert = 0; i_expert < ne02; i_expert++) {
                        for (int i_tile = 0; i_tile < num_tiles_per_expert; i_tile++) {
                            for (int j = 0; j < ne21; j++) {
                                for (int i = 0; i < ne20; i++) {
                                    int expert = ((int *)host_src2)[j * total_experts + i];
                                    if (i_expert == expert) {
                                        ((short *)host_src2_reorder)[out_idx] = static_cast<short>(expert);
                                        ((short *)host_src2_reorder)[out_idx + 1] = static_cast<short>(j * ne11 + (i % ne11));
                                        ((short *)host_src2_reorder)[out_idx + 2] = static_cast<short>(j * ne20 + i);
                                        ((short *)host_src2_reorder)[out_idx + 3] = static_cast<short>(i_tile);
                                        out_idx += 4;
                                    }
                                }
                            }
                        }
                    }
                    buf_src2 = clCreateBuffer(backend_ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ne20 * ne21 * 4 * num_tiles_per_expert * sizeof(short), host_src2_reorder, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(tile_size);
                    global_size[2] = static_cast<size_t>(ne20 * ne21 * num_tiles_per_expert);
                }

                // create a sub_buffer for src1
                cl_buffer_region region;
                region.origin = offset1;
                region.size = ne10 * ne11 * ne12 * sizeof(float);
                src1_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                CL_CHECK(status);

                // create image for src1
                cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne10 * ne11 * ne12 / 4), 0,0,0,0,0,0,0, {src1_sub_buffer}};
                buf_src1_image = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                CL_CHECK(status);

                // Set kernel args
                int arg_idx = 0;
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_mxfp4->q));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_mxfp4->e));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src1_image));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_ulong),  &offsetd));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                if (ne12 == 1) {
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne11));
                } else {
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &tile_size));
                }

                // launch kernel
                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                // deallocate sub buffers and images
                CL_CHECK(clReleaseMemObject(src1_sub_buffer));
                CL_CHECK(clReleaseMemObject(buf_src1_image));
                CL_CHECK(clReleaseMemObject(buf_src2));
                return;
            } // else fallback to generic kernel
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_id_mxfp4_f32_flat;

            cl_mem q;
            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 2;

                q = extra0_mxfp4->q;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 1;
                ndst = 4;

                q = extra0_mxfp4->q_img;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_mxfp4->e));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_mem),   &extra_mask->data_device));
            CL_CHECK(clSetKernelArg(kernel, 25, sizeof(cl_ulong), &offset_mask));
            CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &mask_nb1));
            CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &mask_nb2));
            CL_CHECK(clSetKernelArg(kernel, 28, sizeof(int),      &has_id_mask));
#else // GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_id_mxfp4_f32;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 2;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 2;
                ndst = 2;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_mem),   &extra_mask->data_device));
            CL_CHECK(clSetKernelArg(kernel, 25, sizeof(cl_ulong), &offset_mask));
            CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &mask_nb1));
            CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &mask_nb2));
            CL_CHECK(clSetKernelArg(kernel, 28, sizeof(int),      &has_id_mask));
            CL_CHECK(clSetKernelArg(kernel, 29, sizeof(float)*sgs,nullptr));
#endif // GGML_OPENCL_SOA_Q
            break;
        }
        default:
            GGML_ASSERT(false && "not implemented");;
    }

    int _ne1 = 1;
    int ne123 = dst_rows;

    size_t global_work_size[] = {(size_t)(ne01+ndst*nsg-1)/(ndst*nsg)*sgs, (size_t)(_ne1+nrows-1)/nrows*nsg, (size_t)ne123};
    size_t local_work_size[] = {(size_t)sgs, (size_t)nsg, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_scale(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_UNUSED(src1);

    GGML_ASSERT(ggml_is_contiguous(src0));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    float scale;
    float bias;
    memcpy(&scale, ((int32_t *) dst->op_params) + 0, sizeof(float));
    memcpy(&bias,  ((int32_t *) dst->op_params) + 1, sizeof(float));

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->kernel_scale;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(float),    &scale));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(float),    &bias));

    int n = ggml_nelements(dst)/4;

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_cpy(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);

    // GGML_OP_CPY happens between src0 and src1.
    // GGML_OP_DUP and GGML_OP_CONT happen between src0 and dst.
    UNUSED(dst);

    const int ne00 = src0 ? src0->ne[0] : 0;
    const int ne01 = src0 ? src0->ne[1] : 0;
    const int ne02 = src0 ? src0->ne[2] : 0;
    const int ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    const int ne10 = src1 ? src1->ne[0] : 0;
    const int ne11 = src1 ? src1->ne[1] : 0;
    const int ne12 = src1 ? src1->ne[2] : 0;
    const int ne13 = src1 ? src1->ne[3] : 0;

    const cl_ulong nb10 = src1 ? src1->nb[0] : 0;
    const cl_ulong nb11 = src1 ? src1->nb[1] : 0;
    const cl_ulong nb12 = src1 ? src1->nb[2] : 0;
    const cl_ulong nb13 = src1 ? src1->nb[3] : 0;

    const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
    const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;

    cl_kernel kernel;

    switch (src0t) {
        case GGML_TYPE_F32:
            switch (src1t) {
                case GGML_TYPE_F16:
                    kernel = backend_ctx->kernel_cpy_f32_f16;
                    break;
                case GGML_TYPE_F32:
                    kernel = backend_ctx->kernel_cpy_f32_f32;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented");
            }
            break;
        case GGML_TYPE_F16:
            switch (src1t) {
                case GGML_TYPE_F16:
                    kernel = backend_ctx->kernel_cpy_f16_f16;
                    break;
                case GGML_TYPE_F32:
                    kernel = backend_ctx->kernel_cpy_f16_f32;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented");
            }
            break;
        default:
            GGML_ASSERT(false && "not implemented");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne13));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));

    const int nth = MIN(64, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, src1);
}

static void ggml_cl_dup(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cl_cpy(backend, src0, dst, nullptr);
    UNUSED(src1);
}

static void ggml_cl_diag_mask_inf(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    int n_past = ((int32_t *)(dst->op_params))[0];

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    if (ne00%8 == 0) {
        kernel = backend_ctx->kernel_diag_mask_inf_8;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_past));

        size_t global_work_size[] = {(size_t)ne00*ne01*ne02/8, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else {
        kernel = backend_ctx->kernel_diag_mask_inf;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_past));

        size_t global_work_size[] = {(size_t)ne00, (size_t)ne01, (size_t)ne02};
        size_t local_work_size[] = {64, 1, 1};

        size_t * local_work_size_ptr = local_work_size;
        if (ne00 % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
            local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
        }

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
    }
}

static void ggml_cl_soft_max(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    // Softmax can now fuse KQ mask and KQ scale, which used to be two additional
    // ops before softmax. It now also fuses alibi if `max_bias > 0`. For llama,
    // alibi is not used; however, for some other models, it is used.
    // KQ_mask
    if (src1) {
        GGML_ASSERT(src1);
        GGML_ASSERT(src1->extra);
    }

    const ggml_tensor * src2 = dst->src[2];
    if (src2) {
        GGML_ASSERT(src2->extra);
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    ggml_tensor_extra_cl * extra1 = src1 ? (ggml_tensor_extra_cl *)src1->extra : nullptr;
    ggml_tensor_extra_cl * extra2 = src2 ? (ggml_tensor_extra_cl *)src2->extra : nullptr;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_ulong offset1 = extra1 ? extra1->offset + src1->view_offs : offset0;
    cl_ulong offset2 = extra2 ? extra2->offset + src2->view_offs : offset0;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_long nb01 = src0->nb[1];
    const cl_long nb02 = src0->nb[2];
    const cl_long nb03 = src0->nb[3];

    const int ne12 = src1 ? src1->ne[2] : 0;
    const int ne13 = src1 ? src1->ne[3] : 0;

    const cl_long nb11 = src1 ? src1->nb[1] : 0;
    const cl_long nb12 = src1 ? src1->nb[2] : 0;
    const cl_long nb13 = src1 ? src1->nb[3] : 0;

    const cl_long nb1 = dst->nb[1];
    const cl_long nb2 = dst->nb[2];
    const cl_long nb3 = dst->nb[3];

    float scale, max_bias;
    memcpy(&scale,    dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, dst->op_params + 1, sizeof(float));

    const int n_head      = src0->ne[2];
    const int n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // Local size must be wave size. Each workgroup is a wave, working on a row,
    // where a row corresponds to leading dimension.
    int nth = MIN(32, ne00);

    if (backend_ctx->gpu_family == INTEL) {
        // This is the same as the initial value.
        nth = MIN(32, ne00);
    }
    else if (backend_ctx->gpu_family == ADRENO) {
        nth = 64;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    cl_kernel kernel;

    if (ne00%4 == 0) {
        if (use_f16) {
            kernel = backend_ctx->kernel_soft_max_4_f16;
        } else {
            kernel = backend_ctx->kernel_soft_max_4;
        }
    } else {
        if (use_f16) {
            kernel = backend_ctx->kernel_soft_max_f16;
        } else {
            kernel = backend_ctx->kernel_soft_max;
        }
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   extra1 ? &extra1->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   extra2 ? &extra2->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne13));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(float),    &scale));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(float),    &max_bias));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(float),    &m0));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(float),    &m1));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &n_head_log2));

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_rope(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_tensor * src2 = dst->src[2];
    ggml_tensor_extra_cl * extra2 = src2 ? (ggml_tensor_extra_cl *)src2->extra : nullptr;

    cl_ulong offset2 = extra2 ? extra2->offset + src2->view_offs : offset0;

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;
    const int  ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong  nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong  nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong  nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong  nb03 = src0 ? src0->nb[3] : 0;

    const int ne10 = src1 ? src1->ne[0] : 0;
    const int ne11 = src1 ? src1->ne[1] : 0; UNUSED(ne11);
    const int ne12 = src1 ? src1->ne[2] : 0; UNUSED(ne12);
    const int ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

    const int  ne0 = dst ? dst->ne[0] : 0;
    const int  ne1 = dst ? dst->ne[1] : 0;
    const int  ne2 = dst ? dst->ne[2] : 0;
    const int  ne3 = dst ? dst->ne[3] : 0;

    const cl_ulong  nb0 = dst ? dst->nb[0] : 0;
    const cl_ulong  nb1 = dst ? dst->nb[1] : 0;
    const cl_ulong  nb2 = dst ? dst->nb[2] : 0;
    const cl_ulong  nb3 = dst ? dst->nb[3] : 0;

    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    int nth = MIN(64, ne00);

    const int n_past     = ((int *) dst->op_params)[0];
    const int n_dims     = ((int *) dst->op_params)[1];
    const int mode       = ((int *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int32_t sections[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int32_t)*4);

    const bool is_neox = mode & 2;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    const int  is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    if (is_mrope) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
    }

    cl_kernel kernel;

    if (is_neox) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_neox_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_neox_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    } else if (is_mrope && !is_vision) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_multi_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_multi_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    } else if (is_vision) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_vision_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_vision_f16;
                break;
            default:
                GGML_ASSERT(false);
        }
    } else {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_norm_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_norm_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   extra2 ? &extra2->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &nb3));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &n_past));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &n_dims));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(int),      &n_ctx_orig));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(float),    &freq_base));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(float),    &freq_scale));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(float),    &ext_factor));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(float),    &attn_factor));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(float),    &beta_fast));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(float),    &beta_slow));
    // both mrope and vision kernels have sections
    if (is_mrope || is_vision) {
        CL_CHECK(clSetKernelArg(kernel, 33, sizeof(int32_t)*4, &sections));
    }
    // only mrope has is_imrope
    if (is_mrope && !is_vision) {
        CL_CHECK(clSetKernelArg(kernel, 34, sizeof(int), &is_imrope));
    }

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_im2col(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    // src0 - filter, src1 - input
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const cl_long IC = src1->ne[is_2D ? 2 : 1];
    const cl_long IH = is_2D ? src1->ne[1] : 1;
    const cl_long IW =         src1->ne[0];

    const cl_long KH = is_2D ? src0->ne[1] : 1;
    const cl_long KW =         src0->ne[0];

    const cl_long OH = is_2D ? dst->ne[2] : 1;
    const cl_long OW =         dst->ne[1];

    // nb is byte offset, src is type float32
    const cl_ulong delta_offset = src1->nb[is_2D ? 2 : 1]/4;
    const cl_long  batch        = src1->ne[is_2D ? 3 : 2];
    const cl_ulong batch_offset = src1->nb[is_2D ? 3 : 2]/4;

    const cl_long pelements = OW*KW*KH;
    const cl_long CHW       = IC*KH*KW;

    cl_kernel kernel;

    if(dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->kernel_im2col_f16;
    } else {
        kernel = backend_ctx->kernel_im2col_f32;
    }

    CL_CHECK(clSetKernelArg(kernel,   0, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,   1, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,   2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,   3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,   4, sizeof(cl_ulong), &batch_offset));
    CL_CHECK(clSetKernelArg(kernel,   5, sizeof(cl_ulong), &delta_offset));
    CL_CHECK(clSetKernelArg(kernel,   6, sizeof(cl_long),  &IW));
    CL_CHECK(clSetKernelArg(kernel,   7, sizeof(cl_long),  &IH));
    CL_CHECK(clSetKernelArg(kernel,   8, sizeof(cl_long),  &IC));
    CL_CHECK(clSetKernelArg(kernel,   9, sizeof(cl_long),  &OW));
    CL_CHECK(clSetKernelArg(kernel,  10, sizeof(cl_long),  &OH));
    CL_CHECK(clSetKernelArg(kernel,  11, sizeof(cl_long),  &KW));
    CL_CHECK(clSetKernelArg(kernel,  12, sizeof(cl_long),  &KH));
    CL_CHECK(clSetKernelArg(kernel,  13, sizeof(cl_long),  &pelements));
    CL_CHECK(clSetKernelArg(kernel,  14, sizeof(cl_long),  &CHW));
    CL_CHECK(clSetKernelArg(kernel,  15, sizeof(int),      &s0));
    CL_CHECK(clSetKernelArg(kernel,  16, sizeof(int),      &s1));
    CL_CHECK(clSetKernelArg(kernel,  17, sizeof(int),      &p0));
    CL_CHECK(clSetKernelArg(kernel,  18, sizeof(int),      &p1));
    CL_CHECK(clSetKernelArg(kernel,  19, sizeof(int),      &d0));
    CL_CHECK(clSetKernelArg(kernel,  20, sizeof(int),      &d1));

    const int num_blocks = (pelements + 256 - 1) / 256;
    size_t global_work_size[] = {(size_t)num_blocks*256, (size_t)OH, (size_t)batch*IC};
    size_t local_work_size[] = {256, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_argsort(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_UNUSED(src1);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00  = src0->ne[0];
    const int nrows = ggml_nrows(src0);

    int ne00_padded = 1;
    while (ne00_padded < ne00) {
        ne00_padded *= 2;
    }

    int order = (enum ggml_sort_order) dst->op_params[0];

    cl_kernel kernel = backend_ctx->kernel_argsort_f32_i32;

    CL_CHECK(clSetKernelArg(kernel,   0, sizeof(cl_mem),            &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,   1, sizeof(cl_ulong),          &offset0));
    CL_CHECK(clSetKernelArg(kernel,   2, sizeof(cl_mem),            &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,   3, sizeof(cl_ulong),          &offsetd));
    CL_CHECK(clSetKernelArg(kernel,   4, sizeof(int),               &ne00));
    CL_CHECK(clSetKernelArg(kernel,   5, sizeof(int),               &ne00_padded));
    CL_CHECK(clSetKernelArg(kernel,   6, sizeof(int),               &order));
    CL_CHECK(clSetKernelArg(kernel,   7, ne00_padded*sizeof(int),   NULL));

    size_t global_work_size[] = {(size_t)ne00_padded, (size_t)nrows, (size_t)1};
    size_t local_work_size[] = {(size_t)ne00_padded, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_sum_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_UNUSED(src1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(ggml_is_contiguous(src0));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    cl_kernel kernel = backend_ctx->kernel_sum_rows_f32;

    CL_CHECK(clSetKernelArg(kernel,   0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,   1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,   2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,   3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,   4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,   5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,   6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,   7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,   8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,   9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  10, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel,  11, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel,  12, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel,  13, sizeof(cl_ulong), &nb3));

    size_t global_work_size[] = {(size_t)ne01, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_glu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(ggml_is_contiguous_1(src0));

    if (src1) {
        GGML_ASSERT(src1);
        GGML_ASSERT(src1->extra);
        GGML_ASSERT(ggml_are_same_shape(src0, src1));
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    cl_kernel kernel;
    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_GEGLU:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->kernel_geglu;
            } else {
                kernel = backend_ctx->kernel_geglu_f16;
            }
            break;
        case GGML_GLU_OP_REGLU:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->kernel_reglu;
            } else {
                kernel = backend_ctx->kernel_reglu_f16;
            }
            break;
        case GGML_GLU_OP_SWIGLU:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->kernel_swiglu;
            } else {
                kernel = backend_ctx->kernel_swiglu_f16;
            }
            break;
        case GGML_GLU_OP_SWIGLU_OAI:
            kernel = backend_ctx->kernel_swiglu_oai;
            break;
        case GGML_GLU_OP_GEGLU_ERF:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->kernel_geglu_erf;
            } else {
                kernel = backend_ctx->kernel_geglu_erf_f16;
            }
            break;
        case GGML_GLU_OP_GEGLU_QUICK:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->kernel_geglu_quick;
            } else {
                kernel = backend_ctx->kernel_geglu_quick_f16;
            }
            break;
        default:
            GGML_ABORT("Unsupported glu op");
    }

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    ggml_tensor_extra_cl * extra1 = src1 ? (ggml_tensor_extra_cl *)src1->extra : nullptr;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_ulong offset1 = extra1 ? extra1->offset + src1->view_offs : offset0;

    const int ne0       = dst->ne[0];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb11 = src1 ? src1->nb[1] : nb01;

    const cl_ulong nb1  = dst->nb[1];

    const int   swp   = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    const int ne00_off = src1 ? 0 : (swp ? ne0 : 0);
    const int ne10_off = src1 ? 0 : (swp ? 0 : ne0);

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   src1 ? &extra1->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne00_off));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10_off));

    if (ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU_OAI) {
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float), &limit));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(float), &alpha));
    }

    const size_t nrows = ggml_nrows(src0);
    size_t nth = 512;
    size_t global_work_size[] = {nrows*nth, 1, 1};
    size_t local_work_size[] = {nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

//------------------------------------------------------------------------------
// Op offloading
//------------------------------------------------------------------------------

typedef void (*ggml_cl_func_t)(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor) {
    ggml_cl_func_t func = nullptr;

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    const bool any_on_device = tensor->extra
        || (src0 != nullptr && src0->extra)
        || (src1 != nullptr && src1->extra);

    switch (tensor->op) {
        case GGML_OP_GET_ROWS:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_get_rows;
            break;
        case GGML_OP_SET_ROWS:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_set_rows;
            break;
        case GGML_OP_CPY:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_cpy;
            break;
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_dup;
            break;
        case GGML_OP_ADD:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_add;
            break;
        case GGML_OP_ADD_ID:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_add_id;
            break;
        case GGML_OP_MUL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_mul;
            break;
        case GGML_OP_DIV:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_div;
            break;
        case GGML_OP_SUB:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_sub;
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_GELU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_gelu;
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_gelu_erf;
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_gelu_quick;
                    break;
                case GGML_UNARY_OP_SILU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_silu;
                    break;
                case GGML_UNARY_OP_RELU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_relu;
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_sigmoid;
                    break;
                case GGML_UNARY_OP_TANH:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_tanh;
                    break;
                default:
                    return false;
            } break;
        case GGML_OP_GLU:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_glu;
            break;
        case GGML_OP_CLAMP:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_clamp;
            break;
        case GGML_OP_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_norm;
            break;
        case GGML_OP_RMS_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_rms_norm;
            break;
        case GGML_OP_GROUP_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_group_norm;
            break;
                case GGML_OP_REPEAT:
             if (!any_on_device) {
                return false;
            }
            func = ggml_cl_repeat;
            break;
        case GGML_OP_PAD:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_pad(backend, tensor->src[0], tensor);
            return true;
        case GGML_OP_UPSCALE:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_upscale(backend, tensor->src[0], tensor);
            return true;
        case GGML_OP_CONV_2D:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_conv_2d;
            break;
        case GGML_OP_CONCAT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_concat;
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_timestep_embedding(backend, tensor->src[0], tensor);
            return true;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_cl_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_cl_mul_mat;
            break;
        case GGML_OP_MUL_MAT_ID:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_mul_mat_id;
            break;
        case GGML_OP_SCALE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_scale;
            break;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_nop;
            break;
        case GGML_OP_DIAG_MASK_INF:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_diag_mask_inf;
            break;
        case GGML_OP_SOFT_MAX:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_soft_max;
            break;
        case GGML_OP_ROPE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_rope;
            break;
        case GGML_OP_IM2COL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_im2col;
            break;
        case GGML_OP_ARGSORT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_argsort;
            break;
        case GGML_OP_SUM_ROWS:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_sum_rows;
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_flash_attn(backend, tensor->src[0], tensor->src[1], tensor);
            return true;
        default:
            return false;
    }

    func(backend, tensor->src[0], tensor->src[1], tensor);
    return true;
}
