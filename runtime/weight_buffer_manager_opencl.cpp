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
    octx->n_xfer_extra         = 0;
    octx->xfer_round_robin     = 0;
    for (int i = 0; i < wbm_opencl_ctx::N_XFER_EXTRA; ++i) octx->xfer_extra[i] = nullptr;
    octx->retain_cl_mem                = false;
    octx->cache_byte_limit             = 0;
    octx->cached_bytes                 = 0;
    octx->retained_buffers_by_size.clear();
    octx->retain_order_sizes.clear();
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

    // In-flight prefetch 命中：buffer 已 alloc + 已发 enqueueWriteBuffer，
    // 等 write event 完成后直接 mark_resident，避免重复 alloc。
    if (meta->prefetch_event && meta->backend_handle) {
        cl_event ev = static_cast<cl_event>(meta->prefetch_event);
        cl_int werr = clWaitForEvents(1, &ev);
        if (werr != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] prefetch_event clWaitForEvents 失败 block %d: %s (%d)\n",
                         idx, cl_err(werr), werr);
        }
        clReleaseEvent(ev);
        wbm_set_prefetch_event(octx->wbm, idx, nullptr);
        octx->bytes_uploaded_total += meta->byte_size;
        wbm_mark_resident(octx->wbm, idx, meta->backend_handle);
        return 0;
    }

    if (!meta->host_ptr || meta->byte_size == 0) {
        std::fprintf(stderr, "[wbmcl] block %d 未注册或 byte_size=0\n", idx);
        return -3;
    }

    cl_int err = CL_SUCCESS;
    // Retain 模式：从 size 池里拿一个同 size 的 cl_mem 复用，省 clCreateBuffer
    if (octx->retain_cl_mem) {
        auto it = octx->retained_buffers_by_size.find(meta->byte_size);
        if (it != octx->retained_buffers_by_size.end() && !it->second.empty()) {
            cl_mem cached = static_cast<cl_mem>(it->second.back());
            it->second.pop_back();
            // 从 FIFO 列表里移除一个 == 该 size 的 entry（LIFO 找最近的就行）
            for (auto lit = octx->retain_order_sizes.rbegin(); lit != octx->retain_order_sizes.rend(); ++lit) {
                if (*lit == meta->byte_size) {
                    octx->retain_order_sizes.erase(std::next(lit).base());
                    break;
                }
            }
            octx->cached_bytes -= std::min(octx->cached_bytes, meta->byte_size);
            err = clEnqueueWriteBuffer(octx->compute_queue,
                                       cached, CL_TRUE,
                                       0, meta->byte_size, meta->host_ptr,
                                       0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::fprintf(stderr, "[wbmcl retain] enqueueWriteBuffer 失败 idx=%d: %s (%d)\n",
                             idx, cl_err(err), err);
                return -5;
            }
            octx->bytes_uploaded_total += meta->byte_size;
            wbm_mark_resident(octx->wbm, idx, static_cast<void *>(cached));
            return 0;
        }
        // 没缓存的话走下面正常 create
    }

    // GGML_ELASTIC_USE_HOST_PTR=1 实验：让 OpenCL 用 mmap 指针直接做 cl_mem
    // 后备存储，省掉显式的 host→GPU memcpy。Adreno unified memory 下可能
    // 实现零拷贝；非 unified 架构（如桌面独显）driver 会自己做一次 copy，
    // 等价但多一次驱动开销。默认关。
    static const bool s_use_host_ptr = []() {
        const char *e = std::getenv("GGML_ELASTIC_USE_HOST_PTR");
        return e && *e && *e != '0';
    }();

    cl_mem buf = nullptr;
    if (s_use_host_ptr) {
        // host_ptr 由 llama_model_loader 的 mmap 而来，生命周期 = 进程 →
        // cl_mem 持续引用安全。Adreno 上 USE_HOST_PTR 触发零拷贝。
        buf = clCreateBuffer(octx->cl_ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                             meta->byte_size, meta->host_ptr, &err);
    } else {
        buf = clCreateBuffer(octx->cl_ctx, CL_MEM_READ_ONLY,
                             meta->byte_size, nullptr, &err);
    }
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[wbmcl] clCreateBuffer 失败 block %d size %zu: %s (%d)\n",
                     idx, meta->byte_size, cl_err(err), err);
        return -4;
    }
    octx->n_creates += 1;

    if (s_use_host_ptr) {
        // USE_HOST_PTR 路径：driver 已绑 host_ptr，不再需要 enqueueWriteBuffer
        octx->bytes_uploaded_total += meta->byte_size;
        wbm_mark_resident(octx->wbm, idx, static_cast<void *>(buf));
        return 0;
    }

    if (octx->xfer_queue) {
        // 多 xfer queue 池：round-robin 派发；micro-bench (probe_overlap.cpp)
        // 实测 2 queue 并发能让 DMA 吞吐 ~2× (Adreno 单 DMA 没吃满 host→GPU
        // staging 的 per-call overhead)。
        cl_command_queue use_q = octx->xfer_queue;
        if (octx->n_xfer_extra > 0) {
            unsigned slot = octx->xfer_round_robin++ % (octx->n_xfer_extra + 1);
            if (slot > 0) use_q = octx->xfer_extra[slot - 1];
        }
        cl_event write_ev = nullptr;
        err = clEnqueueWriteBuffer(use_q, buf, CL_FALSE,
                                   0, meta->byte_size, meta->host_ptr,
                                   0, nullptr, &write_ev);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] async clEnqueueWriteBuffer 失败 block %d: %s (%d)\n",
                         idx, cl_err(err), err);
            clReleaseMemObject(buf);
            octx->n_releases += 1;
            return -5;
        }
        clFlush(use_q);
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

    // in-order queue：插 1 个 marker 等 queue 跑完 prior kernel
    cl_event marker = nullptr;
    cl_int merr = clEnqueueMarkerWithWaitList(octx->compute_queue, 0, nullptr, &marker);
    if (merr == CL_SUCCESS && marker) {
        clWaitForEvents(1, &marker);
        clReleaseEvent(marker);
    } else {
        clFinish(octx->compute_queue);
    }
    // 兼容残留 last_use_event 引用
    if (meta->last_use_event) {
        clReleaseEvent(static_cast<cl_event>(meta->last_use_event));
    }

    // Retain 模式：cl_mem 按 size 入池，cap 检查
    if (octx->retain_cl_mem) {
        if (octx->cache_byte_limit > 0) {
            while (octx->cached_bytes + meta->byte_size > octx->cache_byte_limit &&
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
        octx->retained_buffers_by_size[meta->byte_size].push_back(static_cast<void *>(buf));
        octx->retain_order_sizes.push_back(meta->byte_size);
        octx->cached_bytes += meta->byte_size;
        octx->bytes_evicted_total += meta->byte_size;
        wbm_mark_evicted(octx->wbm, idx);
        return 0;
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
    return wbmcl_ensure_resident(octx, idx);
}

int wbmcl_prefetch_async(wbm_opencl_ctx *octx, int idx) {
    if (!octx || !octx->wbm) return -1;
    if (!octx->xfer_queue) return -7;  // 没独立 queue，不发 async prefetch
    const block_meta *meta = wbm_get(octx->wbm, idx);
    if (!meta) return -2;
    if (meta->resident) return 0;
    if (meta->prefetch_event) return 0;  // 已 in-flight
    if (!meta->host_ptr || meta->byte_size == 0) return -3;

    cl_int err = CL_SUCCESS;
    cl_mem buf = nullptr;
    // Retain pool 优先 (跟 ensure_resident 对齐) — 之前 prefetch 直接 clCreateBuffer
    // 不查池, 导致 prefetch 是热路径时 10000+ 次 alloc/release, Adreno driver 开销
    // 主导 (实测 5000 MB budget 3B F16 eval 6500 ms/tok vs ceiling 183).
    if (octx->retain_cl_mem) {
        auto it = octx->retained_buffers_by_size.find(meta->byte_size);
        if (it != octx->retained_buffers_by_size.end() && !it->second.empty()) {
            buf = static_cast<cl_mem>(it->second.back());
            it->second.pop_back();
            for (auto lit = octx->retain_order_sizes.rbegin(); lit != octx->retain_order_sizes.rend(); ++lit) {
                if (*lit == meta->byte_size) {
                    octx->retain_order_sizes.erase(std::next(lit).base());
                    break;
                }
            }
            octx->cached_bytes -= std::min(octx->cached_bytes, meta->byte_size);
        }
    }
    if (!buf) {
        buf = clCreateBuffer(octx->cl_ctx, CL_MEM_READ_ONLY,
                             meta->byte_size, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] async clCreateBuffer 失败 block %d size %zu: %s (%d)\n",
                         idx, meta->byte_size, cl_err(err), err);
            return -4;
        }
        octx->n_creates += 1;
    }

    // 多 xfer queue 池：round-robin 派发
    cl_command_queue use_q = octx->xfer_queue;
    if (octx->n_xfer_extra > 0) {
        unsigned slot = octx->xfer_round_robin++ % (octx->n_xfer_extra + 1);
        if (slot > 0) use_q = octx->xfer_extra[slot - 1];
    }
    cl_event write_ev = nullptr;
    err = clEnqueueWriteBuffer(use_q, buf, CL_FALSE,
                               0, meta->byte_size, meta->host_ptr,
                               0, nullptr, &write_ev);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[wbmcl] async clEnqueueWriteBuffer 失败 block %d: %s (%d)\n",
                     idx, cl_err(err), err);
        clReleaseMemObject(buf);
        octx->n_releases += 1;
        return -5;
    }
    clFlush(use_q);

    // 暂存 backend_handle + prefetch_event；不调 mark_resident，等 ensure_resident
    // 看到 prefetch_event 时 wait 完才正式标 resident（更新 n_resident /
    // resident_bytes）。这样在 prefetch 完成前 LRU 不会把它当 resident。
    octx->wbm->blocks[static_cast<size_t>(idx)].backend_handle = static_cast<void *>(buf);
    wbm_set_prefetch_event(octx->wbm, idx, static_cast<void *>(write_ev));
    return 0;
}

int wbmcl_evict_batch(wbm_opencl_ctx *octx, const int *victims, int n_victims) {
    if (!octx || !octx->wbm || !victims || n_victims <= 0) return 0;

    // in-order queue 顺序保证：在 queue 末尾插一个 marker，等它完成就等于等
    // queue 上所有 prior kernel 完成（含最后一次用 victim 的 kernel）。
    // 不需要 per-block 精细追踪——graph_compute 那边因此不再 stamp per-op
    // marker，省 ~18k OpenCL API call / token。
    cl_event marker = nullptr;
    cl_int merr = clEnqueueMarkerWithWaitList(octx->compute_queue, 0, nullptr, &marker);
    if (merr == CL_SUCCESS && marker) {
        cl_int werr = clWaitForEvents(1, &marker);
        if (werr != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] evict-marker clWaitForEvents 失败: %s (%d)\n",
                         cl_err(werr), werr);
        }
        clReleaseEvent(marker);
    } else {
        // fallback：marker 插入失败就 clFinish 全队列（保守正确）
        std::fprintf(stderr, "[wbmcl] enqueueMarker 失败: %s (%d) —— 退回 clFinish\n",
                     cl_err(merr), merr);
        clFinish(octx->compute_queue);
    }

    // 释放 cl_mem + mark_evicted
    int released = 0;
    for (int i = 0; i < n_victims; ++i) {
        const int v = victims[i];
        const block_meta *m = wbm_get(octx->wbm, v);
        if (!m || !m->resident) continue;

        // 兼容性：若早期残留 last_use_event 引用还在则 release 掉
        if (m->last_use_event) {
            clReleaseEvent(static_cast<cl_event>(m->last_use_event));
        }
        cl_mem buf = static_cast<cl_mem>(m->backend_handle);
        if (buf) {
            // Retain 模式：cl_mem 按 size 入池。pool 满时 FIFO 释放最早 size。
            if (octx->retain_cl_mem) {
                if (octx->cache_byte_limit > 0) {
                    while (octx->cached_bytes + m->byte_size > octx->cache_byte_limit &&
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
                octx->retained_buffers_by_size[m->byte_size].push_back(static_cast<void *>(buf));
                octx->retain_order_sizes.push_back(m->byte_size);
                octx->cached_bytes += m->byte_size;
                octx->bytes_evicted_total += m->byte_size;
                wbm_mark_evicted(octx->wbm, v);
                ++released;
                continue;
            }
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
    // 释放 retain 模式下暂存的所有 cl_mem
    for (auto &kv : octx->retained_buffers_by_size) {
        for (void *p : kv.second) {
            if (p) {
                clReleaseMemObject(static_cast<cl_mem>(p));
                octx->n_releases += 1;
            }
        }
    }
    octx->retained_buffers_by_size.clear();
    octx->retain_order_sizes.clear();
    octx->cached_bytes = 0;
    for (auto &b : octx->wbm->blocks) {
        // 清掉未消费的 in-flight prefetch（block 还没 resident 但已发 write）
        if (!b.resident && b.prefetch_event) {
            cl_event ev = static_cast<cl_event>(b.prefetch_event);
            clWaitForEvents(1, &ev);
            clReleaseEvent(ev);
            b.prefetch_event = nullptr;
            if (b.backend_handle) {
                clReleaseMemObject(static_cast<cl_mem>(b.backend_handle));
                octx->n_releases += 1;
                b.backend_handle = nullptr;
            }
        }
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
