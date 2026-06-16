// runtime/weight_buffer_manager_opencl.h
//
// WeightBufferManager 的 OpenCL 包装层：实际把 cl_mem 申请 / 释放 / 写入；
// evict 前用 cl_event 等 in-flight kernel 结束。
//
// 设计要点：
//   - 复用 runtime/weight_buffer_manager.{h,cpp} 的纯逻辑层（LRU、字节核算）
//   - block_meta::backend_handle 在这里被强转为 cl_mem
//   - block_meta::last_use_event   在这里被强转为 cl_event
//   - 首版 prefetch == 同步 ensure_resident（避开 in-flight 写 + 跟主 queue
//     竞争的复杂度）；future work: 用 xfer_queue + event 异步化
//
// 编译：只有 find_package(OpenCL) 找到 ICD/headers 时本文件才进 elastic_runtime
// 静态库（见 runtime/CMakeLists.txt）。桌面没装 OpenCL 时整个文件不编。

#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#include <CL/cl.h>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "weight_buffer_manager.h"

namespace elastic {

// SOA-aware evict/reload 回调. set_tensor 完成 SOA-split 后注册, ensure/evict
// 命中此 idx 时改走回调路径 (默认 cl_mem create+write 重建 SOA layout 错). 详见
// wbmcl_register_soa.
struct soa_callbacks {
    std::function<int()> evict_fn;
    std::function<int()> reload_fn;
};

// SOA pool entry: parent buffer + d/q sub-buffer 一起回收避免重建子视图开销.
struct soa_pool_entry {
    void *parent = nullptr;
    void *d      = nullptr;
    void *q      = nullptr;
};

enum class wbmcl_device_event_kind {
    TRANSFER_WRITE = 0,
    XFORM_CONVERT  = 1,
    XFORM_TRANSPOSE = 2,
    XFORM_COPY      = 3,
    COMPUTE_WAIT    = 4,
};

struct wbmcl_device_event_sample {
    cl_event                  event = nullptr;
    wbmcl_device_event_kind   kind  = wbmcl_device_event_kind::TRANSFER_WRITE;
    size_t                    bytes = 0;
};

enum class wbmcl_stage_detail_kind {
    SOA_POOL_LOOKUP = 0,
    PARENT_ALLOC    = 1,
    STAGING_ALLOC   = 2,
    HOST_SRC        = 3,
    WRITE_ENQUEUE   = 4,
    SUBBUFFER       = 5,
    BARRIER_ENQUEUE = 6,
    CONVERT_ENQUEUE = 7,
    TRANSPOSE_ENQUEUE = 8,
    FINAL_WAIT_ENQUEUE = 9,
    MARK_RESIDENT   = 10,
};

struct wbmcl_stage_detail_bucket {
    uint64_t n = 0;
    uint64_t us = 0;
    size_t bytes = 0;
};

struct wbm_opencl_ctx {
    weight_buffer_manager *wbm;           // 不持有所有权
    cl_context        cl_ctx;
    cl_command_queue  compute_queue;      // 主算用
    cl_command_queue  xfer_queue;         // 异步 prefetch；空 → 用 compute_queue

    // GGML_ELASTIC_CL_RETAIN=1 触发：evict 时不调 clReleaseMemObject，把 cl_mem
    // 暂存到 retained_buffers[idx]；ensure_resident 直接 clEnqueueWriteBuffer 到
    // 已存在的 cl_mem，省 driver per-call alloc/free overhead。
    //
    // GGML_ELASTIC_CL_RETAIN_MB=N 设 pool 字节上限（默认 0 = 不限）。pool 满时
    // FIFO 释放最早入池的 cl_mem，保证 cached_bytes ≤ cache_byte_limit。
    // 这样 elastic 的 "evict 真释放" 语义部分保留，driver overhead 摊到 cache miss。
    bool                                                       retain_cl_mem;
    size_t                                                     cache_byte_limit;  // 0 = 不限
    size_t                                                     cached_bytes;      // 当前 pool 字节
    // Pool indexed by **byte size** (不是 tensor idx)：不同 idx 的 tensor 可以
    // 共享同 size 的 cl_mem。这样 total GPU = working_set + 小 pool overhead，
    // 真的合规 budget——而不是 indexed by idx 那种"每个 tensor 一份" ≈ 全 model。
    std::unordered_map<size_t, std::vector<void *>>            retained_buffers_by_size;
    std::list<size_t>                                          retain_order_sizes;  // FIFO 顺序，每个 entry = 一个 cl_mem 的 size
    // 额外的 xfer queue 池：round-robin 派发，让多个 backend transfer 真并行
    // micro-bench (probe_overlap.cpp) 实测 2 queue 能 2× 吞吐，3+ 边际递减
    static constexpr int N_XFER_EXTRA = 4;
    cl_command_queue  xfer_extra[N_XFER_EXTRA];  // 0..n_xfer_extra-1 有效
    int               n_xfer_extra;       // 实际启用条数（0..4）
    unsigned          xfer_round_robin;   // 选 queue 用的计数器

    // 内部计数 / 统计，便于 metrics_logger 取
    size_t bytes_uploaded_total;          // 历史累计上传字节
    size_t bytes_evicted_total;           // 历史累计释放字节
    int    n_creates;                     // clCreateBuffer 调用次数
    int    n_releases;                    // clReleaseMemObject 调用次数

    // SOA: per-idx 回调 + 按 size 复用 parent+d+q triple + 共享 staging buffer.
    std::unordered_map<int, soa_callbacks>             soa_per_idx;
    std::unordered_map<size_t, std::vector<soa_pool_entry>> soa_pool_by_size;
    cl_mem            soa_staging          = nullptr;
    size_t            soa_staging_capacity = 0;
    cl_event          soa_staging_last_use_ev = nullptr;

    // O_DIRECT reload (非 SOA 路径). 默认由 ggml-opencl 注入;
    // GGML_ELASTIC_DIRECT_IO=0 时关闭:
    // 给 host_ptr (mmap VA) 反查文件 + O_DIRECT pread 到 dst, 返 0 成功. runtime 层
    // 不直接依赖 libllama, 通过函数指针解耦. nullptr = 走 mmap host_ptr (默认).
    std::function<int(const void *host_ptr, void *dst, size_t nbytes)> direct_read_fn;

    // Plan-stage staging: LOAD 把 disk/mmap 内容拷到 host_staging_by_idx，
    // TRANSFER/XFORM 可复用该 host staging，避免把 disk load 和 backend transform 混在一起。
    //
    // host staging pool: transfer 完成后可把 staging buffer 归还到按 size 分组的
    // CPU pool，后续 LOAD 直接复用，避免内存预算变大时反复 malloc/free。
    bool retain_host_staging = false;
    size_t host_staging_pool_limit = 0; // 0 = 不限
    size_t host_staging_pool_bytes = 0;
    std::unordered_map<int, std::vector<char>> host_staging_by_idx;
    std::unordered_map<size_t, std::vector<std::vector<char>>> host_staging_pool_by_size;
    std::list<size_t> host_staging_pool_order;
    size_t bytes_loaded_total = 0;

    uint64_t stage_load_calls = 0;
    uint64_t stage_load_ok    = 0;
    uint64_t stage_load_us    = 0;
    size_t   stage_load_bytes = 0;
    uint64_t direct_read_calls = 0;
    uint64_t direct_read_ok    = 0;
    uint64_t direct_read_fail  = 0;
    uint64_t direct_read_us    = 0;
    size_t   direct_read_bytes = 0;

    uint64_t stage_transfer_calls = 0;
    uint64_t stage_transfer_ok    = 0;
    uint64_t stage_transfer_us    = 0;
    size_t   stage_transfer_bytes = 0;

    uint64_t stage_xform_calls = 0;
    uint64_t stage_xform_ok    = 0;
    uint64_t stage_xform_us    = 0;
    size_t   stage_xform_bytes = 0;

    uint64_t soa_pool_hit       = 0;
    uint64_t soa_pool_miss      = 0;
    uint64_t parent_pool_hit    = 0;
    uint64_t parent_pool_miss   = 0;
    uint64_t parent_create_calls = 0;
    uint64_t parent_create_us    = 0;
    size_t   parent_create_bytes = 0;

    bool device_timing = false;
    std::mutex device_timing_mtx;
    std::vector<wbmcl_device_event_sample> device_events;

    bool stage_detail = false;
    std::mutex stage_detail_mtx;
    wbmcl_stage_detail_bucket stage_detail_buckets[11];
};

void wbmcl_register_soa(wbm_opencl_ctx *octx, int idx,
                        std::function<int()> evict_fn,
                        std::function<int()> reload_fn);

// 绑定一个已存在的 WBM 和 OpenCL 上下文。不接管 cl_context / queue 的生命周期，
// 调用方仍负责销毁。xfer_queue 可传 nullptr，此时 prefetch / 同步上传走
// compute_queue。
int  wbmcl_init(wbm_opencl_ctx *octx,
                weight_buffer_manager *wbm,
                cl_context cl_ctx,
                cl_command_queue compute_queue,
                cl_command_queue xfer_queue);

void wbmcl_record_device_event(wbm_opencl_ctx *octx,
                               cl_event ev,
                               wbmcl_device_event_kind kind,
                               size_t bytes);
void wbmcl_dump_device_timing(wbm_opencl_ctx *octx, FILE *out);
void wbmcl_record_stage_detail(wbm_opencl_ctx *octx,
                               wbmcl_stage_detail_kind kind,
                               uint64_t us,
                               size_t bytes);
void wbmcl_dump_stage_detail(wbm_opencl_ctx *octx, FILE *out);

// 阻塞确保 block idx 驻留：
//   - 已驻留 → no-op，返回 0
//   - 未驻留 → clCreateBuffer + clEnqueueWriteBuffer(blocking=CL_TRUE) + mark_resident
// 失败时尽量回滚（释放新建的 cl_mem），不会把 WBM 置成半驻留状态。
int  wbmcl_ensure_resident(wbm_opencl_ctx *octx, int idx);

// 释放 block idx 的 cl_mem：
//   - 未驻留 → no-op
//   - last_use_event 非空 → clWaitForEvents 等它完成，然后 clReleaseEvent
//   - clReleaseMemObject 释放 buffer，调 wbm_mark_evicted
int  wbmcl_evict(wbm_opencl_ctx *octx, int idx);

// 异步 prefetch 占位：首版直接同步 ensure。后续做异步化时改这一个函数即可。
int  wbmcl_prefetch(wbm_opencl_ctx *octx, int idx);

// 分阶段 plan API:
//   LOAD  : disk/mmap -> host staging
//   TRANSFER : host staging -> backend buffer
//   XFORM : backend/raw -> compute layout (SOA callback 或 generic no-op)
// 旧 ensure_resident 仍是完整兼容路径。
int  wbmcl_load_host(wbm_opencl_ctx *octx, int idx);
int  wbmcl_dma_to_backend(wbm_opencl_ctx *octx, int idx);
int  wbmcl_transform_backend(wbm_opencl_ctx *octx, int idx);

// 真正的异步 prefetch：在 xfer_queue 上发非阻塞 clEnqueueWriteBuffer，把 write
// 的 cl_event 存到 block_meta::prefetch_event。block 此时 backend_handle 已分配
// 但 resident=false，留给后续 ensure_resident 看到 prefetch_event 时 wait 它。
//
// 返回值：
//   0  发出 prefetch 成功（或 block 已 resident / 已 in-flight，no-op 返回 0）
//   <0 失败（xfer_queue 没设、clCreateBuffer / clEnqueueWriteBuffer 报错等）
//
// 注意：必须有独立 xfer_queue 才有意义；compute_queue 上发 async 会跟 compute
// kernel 抢同一个 in-order queue 位置，没有 overlap 收益。
int  wbmcl_prefetch_async(wbm_opencl_ctx *octx, int idx);

// 批量驱逐：一次 clWaitForEvents 在所有 victim 的 last_use_event 上（过滤空
// 的），随后逐个 clReleaseEvent + clReleaseMemObject + wbm_mark_evicted。
// 比 N 次单独 wbmcl_evict 省 N-1 次同步往返。
// victims 通常由 wbm_evict_to_byte_budget 产出。返回成功释放的 victim 数。
int  wbmcl_evict_batch(wbm_opencl_ctx *octx,
                       const int *victims,
                       int n_victims);

// 取 idx 的 cl_mem（已驻留返回非空；未驻留返回 nullptr）
cl_mem wbmcl_get_buffer(const wbm_opencl_ctx *octx, int idx);

// 收尾：把还驻留的 block 全 evict 掉，释放未消费的 event。
void wbmcl_shutdown(wbm_opencl_ctx *octx);

}  // namespace elastic
