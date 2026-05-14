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

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include "weight_buffer_manager.h"

namespace elastic {

struct wbm_opencl_ctx {
    weight_buffer_manager *wbm;           // 不持有所有权
    cl_context        cl_ctx;
    cl_command_queue  compute_queue;      // 主算用
    cl_command_queue  xfer_queue;         // 异步 prefetch；空 → 用 compute_queue

    // 内部计数 / 统计，便于 metrics_logger 取
    size_t bytes_uploaded_total;          // 历史累计上传字节
    size_t bytes_evicted_total;           // 历史累计释放字节
    int    n_creates;                     // clCreateBuffer 调用次数
    int    n_releases;                    // clReleaseMemObject 调用次数
};

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

// 取 idx 的 cl_mem（已驻留返回非空；未驻留返回 nullptr）
cl_mem wbmcl_get_buffer(const wbm_opencl_ctx *octx, int idx);

// 收尾：把还驻留的 block 全 evict 掉，释放未消费的 event。
void wbmcl_shutdown(wbm_opencl_ctx *octx);

}  // namespace elastic
