// runtime/weight_buffer_manager_cpu.h
//
// WeightBufferManager 的 CPU 后端：用 malloc/free + memcpy 模拟 evict/reload。
// 跟 OpenCL 版对偶——把所有 GPU 概念替换成 CPU 概念，让 elastic 算法
// （LRU/MRU/pin/prefetch/budget）能在纯 CPU 后端上验证 + 测试。
//
// 语义：
//   - host_ptr = mmap(GGUF) 的指针（不变）
//   - backend_handle = malloc 的 CPU buffer，独立于 mmap
//   - ggml-cpu compute 时 tensor->data 应该指向 backend_handle（reload 后）
//   - evict = free(backend_handle)（mmap 不动；下次 reload 再 malloc + memcpy）
//
// 跟 OpenCL 版的区别：
//   - 不需要 cl_event / queue / barrier（CPU memcpy 同步执行）
//   - 不需要 prefetch_event 的 in-flight 状态（memcpy 没法异步——或者用 worker
//     thread 做，本文件首版同步）
//   - 不需要 ensure_resident 内插 marker（CPU 模型不存在跨 op 的依赖问题）
//
// 用途：
//   1. 验证 elastic 算法本身（LRU/MRU/pin 等），跟 GPU 版结果对照
//   2. 无 GPU 设备上做 model > RAM 的 paging
//   3. 量化"per-tensor cl_mem 架构"的固有开销（vs CPU malloc-based）

#pragma once

#include <cstddef>

#include "weight_buffer_manager.h"

namespace elastic {

struct wbm_cpu_ctx {
    weight_buffer_manager *wbm;            // 不持有所有权

    // 统计
    size_t bytes_uploaded_total;           // 累计 memcpy 字节
    size_t bytes_evicted_total;            // 累计 free 字节
    int    n_mallocs;                      // malloc 次数
    int    n_frees;                        // free 次数

    // 可选 worker thread 池（后续做 async memcpy 用，首版不用）
    int    n_workers;                      // 0 = 同步
};

// 绑定一个已存在的 WBM。不接管所有权。n_workers=0 = 全同步实现。
int  wbmcpu_init(wbm_cpu_ctx *cctx,
                 weight_buffer_manager *wbm,
                 int n_workers = 0);

// 阻塞确保 block idx 驻留：
//   - 已驻留 → no-op，返回 0
//   - 未驻留 → malloc + memcpy(host_ptr → backend_handle) + mark_resident
// 失败时回滚 free 已分配 buffer。
int  wbmcpu_ensure_resident(wbm_cpu_ctx *cctx, int idx);

// 释放 block idx 的 backend_handle：
//   - 未驻留 → no-op
//   - free(backend_handle)，mark_evicted
int  wbmcpu_evict(wbm_cpu_ctx *cctx, int idx);

// 同 OpenCL 版语义：首版直接同步 ensure。
int  wbmcpu_prefetch(wbm_cpu_ctx *cctx, int idx);

// 批量驱逐：用 victim 列表逐个 free + mark_evicted。
int  wbmcpu_evict_batch(wbm_cpu_ctx *cctx, const int *victims, int n_victims);

// 取 idx 的 backend_handle（已驻留返回非空；未驻留返回 nullptr）
void * wbmcpu_get_buffer(const wbm_cpu_ctx *cctx, int idx);

// 收尾：把还驻留的 block 全 free 掉。
void wbmcpu_shutdown(wbm_cpu_ctx *cctx);

}  // namespace elastic
