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

#include <functional>
#include <list>
#include <unordered_map>
#include <vector>

#include "weight_buffer_manager.h"

namespace elastic {

// SOA pool: q4_0/q8_0/mxfp4 走 SOA 重建管线 (parent+scales(d)+quants(q) 三件套).
// evict 时把这三个 cl_mem 一起塞 pool, reload 时按 nbytes 整组取出复用,
// 省掉 4 次 sub-buffer + 一次 convert kernel.
struct soa_pool_entry {
    void *parent = nullptr;
    void *d      = nullptr;   // scales
    void *q      = nullptr;   // quants
};

// SOA per-block evict/reload 回调注册. 调用方在 register_soa 时绑.
// reload_fn 由 prefetch 路径异步调; evict_fn 由 wbmcl_evict 替换走.
struct soa_callback_pair {
    std::function<int()> evict_fn;
    std::function<int()> reload_fn;
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
    // 额外的 xfer queue 池：round-robin 派发，让多个 DMA 真并行
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

    // SOA (q4_0/q8_0/mxfp4) reload 共享 staging buffer (复用, 2× growth).
    // ggml_opencl_elastic_ensure_soa_staging 维护.
    cl_mem    soa_staging              = nullptr;
    size_t    soa_staging_capacity     = 0;
    // 上一次用 staging 的 convert kernel 完成 event — 下次 reuse 前要 wait,
    // 否则新 write 会覆盖正在被 kernel 读的数据.
    cl_event  soa_staging_last_use_ev  = nullptr;

    // SOA pool: 按字节数索引 {parent,d,q} 三件套缓存. 跟 retained_buffers_by_size 配合,
    // 共享同一个 cache_byte_limit / retain_order_sizes FIFO.
    std::unordered_map<size_t, std::vector<soa_pool_entry>> soa_pool_by_size;

    // SOA per-block 回调注册表. wbmcl_register_soa 加, prefetch 路径用 reload_fn.
    std::unordered_map<int, soa_callback_pair>              soa_per_idx;

    // O_DIRECT reload (非 SOA 路径). GGML_ELASTIC_DIRECT_IO=1 时由 ggml-opencl 注入:
    // 给 host_ptr (mmap VA) 反查文件 + O_DIRECT pread 到 dst, 返 0 成功. runtime 层
    // 不直接依赖 libllama, 通过函数指针解耦. nullptr = 走 mmap host_ptr (默认).
    int (*direct_read_fn)(const void *host_ptr, void *dst, size_t nbytes) = nullptr;
};

// 注册 SOA tensor 的 evict/reload 回调. idx 是 wbm 里的 block index.
// 同一个 idx 多次注册以最后一次为准.
void wbmcl_register_soa(wbm_opencl_ctx *octx,
                        int idx,
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
