// runtime/weight_buffer_manager_opencl.cpp —— 见 weight_buffer_manager_opencl.h

#include "weight_buffer_manager_opencl.h"

#include <cassert>
#include <cstdio>

namespace elastic {

namespace {

const char *cl_err(cl_int e) {
    switch (e) {
        case CL_SUCCESS:                       return "CL_SUCCESS";
        case CL_OUT_OF_RESOURCES:              return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:            return "CL_OUT_OF_HOST_MEMORY";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_INVALID_VALUE:                 return "CL_INVALID_VALUE";
        case CL_INVALID_BUFFER_SIZE:           return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_CONTEXT:               return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE:         return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_EVENT:                 return "CL_INVALID_EVENT";
        case CL_INVALID_MEM_OBJECT:            return "CL_INVALID_MEM_OBJECT";
        default:                               return "(unknown cl_int)";
    }
}

}  // namespace

int wbmcl_init(wbm_opencl_ctx *octx,
               weight_buffer_manager *wbm,
               cl_context cl_ctx,
               cl_command_queue compute_queue,
               cl_command_queue xfer_queue) {
    if (!octx || !wbm || !cl_ctx || !compute_queue) return -1;
    octx->wbm                  = wbm;
    octx->cl_ctx               = cl_ctx;
    octx->compute_queue        = compute_queue;
    octx->xfer_queue           = xfer_queue;
    octx->bytes_uploaded_total = 0;
    octx->bytes_evicted_total  = 0;
    octx->n_creates            = 0;
    octx->n_releases           = 0;
    return 0;
}

int wbmcl_ensure_resident(wbm_opencl_ctx *octx, int idx) {
    if (!octx || !octx->wbm) return -1;
    const block_meta *meta = wbm_get(octx->wbm, idx);
    if (!meta) return -2;
    if (meta->resident) return 0;
    if (!meta->host_ptr || meta->byte_size == 0) {
        std::fprintf(stderr, "[wbmcl] block %d 未注册或 byte_size=0\n", idx);
        return -3;
    }

    cl_int err = CL_SUCCESS;
    // 走两步法（先 alloc，再 enqueueWriteBuffer），方便后续插 prefetch + event：
    cl_mem buf = clCreateBuffer(octx->cl_ctx, CL_MEM_READ_ONLY,
                                meta->byte_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[wbmcl] clCreateBuffer 失败 block %d size %zu: %s (%d)\n",
                     idx, meta->byte_size, cl_err(err), err);
        return -4;
    }
    octx->n_creates += 1;

    if (octx->xfer_queue) {
        // 异步路径：上传走 xfer queue，compute queue 插 barrier 等上传 event
        // 完成。host 端不再阻塞。配合 compute queue 自身的 in-order 语义，
        // 后续派发的 kernel 会自动等到 buffer 写完才执行。
        cl_event write_ev = nullptr;
        err = clEnqueueWriteBuffer(octx->xfer_queue, buf, CL_FALSE,
                                   0, meta->byte_size, meta->host_ptr,
                                   0, nullptr, &write_ev);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] async clEnqueueWriteBuffer 失败 block %d: %s (%d)\n",
                         idx, cl_err(err), err);
            clReleaseMemObject(buf);
            octx->n_releases += 1;
            return -5;
        }
        // flush xfer queue 让 enqueue 真的发出去（不调 flush 时驱动可能
        // 攒到下一次 clFinish/clWaitForEvents 才提交）
        clFlush(octx->xfer_queue);
        // compute queue 插 barrier，依赖 write_ev
        cl_int berr = clEnqueueBarrierWithWaitList(octx->compute_queue, 1, &write_ev, nullptr);
        if (berr != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] clEnqueueBarrierWithWaitList 失败: %s (%d)，退回 wait\n",
                         cl_err(berr), berr);
            clWaitForEvents(1, &write_ev);
        }
        clReleaseEvent(write_ev);
    } else {
        // 同步路径：单队列阻塞写
        err = clEnqueueWriteBuffer(octx->compute_queue,
                                   buf, CL_TRUE /* 阻塞 */,
                                   0, meta->byte_size, meta->host_ptr,
                                   0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] clEnqueueWriteBuffer 失败 block %d: %s (%d)\n",
                         idx, cl_err(err), err);
            clReleaseMemObject(buf);
            octx->n_releases += 1;
            return -5;
        }
    }

    octx->bytes_uploaded_total += meta->byte_size;
    wbm_mark_resident(octx->wbm, idx, static_cast<void *>(buf));
    return 0;
}

int wbmcl_evict(wbm_opencl_ctx *octx, int idx) {
    if (!octx || !octx->wbm) return -1;
    const block_meta *meta = wbm_get(octx->wbm, idx);
    if (!meta) return -2;
    if (!meta->resident) return 0;

    cl_mem buf = static_cast<cl_mem>(meta->backend_handle);
    cl_event ev = static_cast<cl_event>(meta->last_use_event);

    // 等待最近一次使用此 buffer 的 kernel 结束，避免释放正在用的 cl_mem
    if (ev) {
        cl_int werr = clWaitForEvents(1, &ev);
        if (werr != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] clWaitForEvents 失败 block %d: %s (%d) —— 继续释放\n",
                         idx, cl_err(werr), werr);
        }
        // event 是 H4 钩点 retain 的；这里释放掉
        cl_int rerr = clReleaseEvent(ev);
        if (rerr != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] clReleaseEvent 失败 block %d: %s (%d)\n",
                         idx, cl_err(rerr), rerr);
        }
    }

    cl_int err = clReleaseMemObject(buf);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[wbmcl] clReleaseMemObject 失败 block %d: %s (%d)\n",
                     idx, cl_err(err), err);
        return -6;
    }
    octx->n_releases += 1;
    octx->bytes_evicted_total += meta->byte_size;
    wbm_mark_evicted(octx->wbm, idx);
    return 0;
}

int wbmcl_prefetch(wbm_opencl_ctx *octx, int idx) {
    // 首版同步：等价于 ensure_resident。
    // TODO：xfer_queue + cl_event 异步化（block_meta 增加 write_event 字段，
    //       后续 ensure_resident 看到该字段则 clWaitForEvents 而不是再 alloc）。
    return wbmcl_ensure_resident(octx, idx);
}

int wbmcl_evict_batch(wbm_opencl_ctx *octx, const int *victims, int n_victims) {
    if (!octx || !octx->wbm || !victims || n_victims <= 0) return 0;

    // 1) 收集所有 in-flight event 一次性等
    std::vector<cl_event> events;
    events.reserve(static_cast<size_t>(n_victims));
    for (int i = 0; i < n_victims; ++i) {
        const int v = victims[i];
        const block_meta *m = wbm_get(octx->wbm, v);
        if (!m || !m->resident) continue;
        if (m->last_use_event) {
            events.push_back(static_cast<cl_event>(m->last_use_event));
        }
    }
    if (!events.empty()) {
        cl_int err = clWaitForEvents(static_cast<cl_uint>(events.size()), events.data());
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] 批量 clWaitForEvents 失败: %s (%d) —— 继续释放\n",
                         cl_err(err), err);
        }
    }

    // 2) 逐个 release event + release cl_mem + mark_evicted
    int released = 0;
    for (int i = 0; i < n_victims; ++i) {
        const int v = victims[i];
        const block_meta *m = wbm_get(octx->wbm, v);
        if (!m || !m->resident) continue;

        if (m->last_use_event) {
            cl_int rerr = clReleaseEvent(static_cast<cl_event>(m->last_use_event));
            if (rerr != CL_SUCCESS) {
                std::fprintf(stderr, "[wbmcl] clReleaseEvent 失败 block %d: %s (%d)\n",
                             v, cl_err(rerr), rerr);
            }
        }
        cl_mem buf = static_cast<cl_mem>(m->backend_handle);
        if (buf) {
            cl_int err = clReleaseMemObject(buf);
            if (err != CL_SUCCESS) {
                std::fprintf(stderr, "[wbmcl] 批量 clReleaseMemObject 失败 block %d: %s (%d)\n",
                             v, cl_err(err), err);
                continue;
            }
            octx->n_releases += 1;
            octx->bytes_evicted_total += m->byte_size;
        }
        wbm_mark_evicted(octx->wbm, v);
        ++released;
    }
    return released;
}

cl_mem wbmcl_get_buffer(const wbm_opencl_ctx *octx, int idx) {
    if (!octx || !octx->wbm) return nullptr;
    const block_meta *meta = wbm_get(octx->wbm, idx);
    if (!meta || !meta->resident) return nullptr;
    return static_cast<cl_mem>(meta->backend_handle);
}

void wbmcl_shutdown(wbm_opencl_ctx *octx) {
    if (!octx || !octx->wbm) return;
    for (auto &b : octx->wbm->blocks) {
        if (b.resident) {
            // 不走 wbmcl_evict 以免被 last_use_event 阻塞太久；强行回收
            if (b.last_use_event) {
                clReleaseEvent(static_cast<cl_event>(b.last_use_event));
                b.last_use_event = nullptr;
            }
            if (b.backend_handle) {
                clReleaseMemObject(static_cast<cl_mem>(b.backend_handle));
                octx->n_releases += 1;
                octx->bytes_evicted_total += b.byte_size;
            }
            b.backend_handle = nullptr;
            b.resident = false;
        }
    }
    octx->wbm->n_resident     = 0;
    octx->wbm->resident_bytes = 0;
    octx->wbm                 = nullptr;
    octx->cl_ctx              = nullptr;
    octx->compute_queue       = nullptr;
    octx->xfer_queue          = nullptr;
}

}  // namespace elastic
