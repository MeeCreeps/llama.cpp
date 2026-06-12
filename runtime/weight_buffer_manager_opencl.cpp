// runtime/weight_buffer_manager_opencl.cpp —— 见 weight_buffer_manager_opencl.h

#include "weight_buffer_manager_opencl.h"

#include <cassert>
#include <cstdio>
#include <cstring>

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

void host_staging_pool_trim(wbm_opencl_ctx *octx, size_t incoming) {
    if (!octx || octx->host_staging_pool_limit == 0) return;
    while (octx->host_staging_pool_bytes + incoming > octx->host_staging_pool_limit &&
           !octx->host_staging_pool_order.empty()) {
        size_t old_sz = octx->host_staging_pool_order.front();
        octx->host_staging_pool_order.pop_front();
        auto it = octx->host_staging_pool_by_size.find(old_sz);
        if (it == octx->host_staging_pool_by_size.end() || it->second.empty()) continue;
        it->second.pop_back();
        octx->host_staging_pool_bytes -= std::min(octx->host_staging_pool_bytes, old_sz);
    }
}

std::vector<char> host_staging_pool_take(wbm_opencl_ctx *octx, size_t nbytes) {
    if (!octx || !octx->retain_host_staging) return std::vector<char>();
    auto it = octx->host_staging_pool_by_size.find(nbytes);
    if (it == octx->host_staging_pool_by_size.end() || it->second.empty()) {
        return std::vector<char>();
    }
    std::vector<char> buf = std::move(it->second.back());
    it->second.pop_back();
    for (auto lit = octx->host_staging_pool_order.rbegin();
         lit != octx->host_staging_pool_order.rend(); ++lit) {
        if (*lit == nbytes) {
            octx->host_staging_pool_order.erase(std::next(lit).base());
            break;
        }
    }
    octx->host_staging_pool_bytes -= std::min(octx->host_staging_pool_bytes, nbytes);
    return buf;
}

void host_staging_pool_put(wbm_opencl_ctx *octx, std::vector<char> &&buf) {
    if (!octx || !octx->retain_host_staging || buf.empty()) return;
    const size_t nbytes = buf.size();
    if (octx->host_staging_pool_limit > 0 && nbytes > octx->host_staging_pool_limit) {
        return;
    }
    host_staging_pool_trim(octx, nbytes);
    if (octx->host_staging_pool_limit > 0 &&
        octx->host_staging_pool_bytes + nbytes > octx->host_staging_pool_limit) {
        return;
    }
    octx->host_staging_pool_by_size[nbytes].push_back(std::move(buf));
    octx->host_staging_pool_order.push_back(nbytes);
    octx->host_staging_pool_bytes += nbytes;
}

void release_host_staging(wbm_opencl_ctx *octx, int idx) {
    if (!octx) return;
    auto it = octx->host_staging_by_idx.find(idx);
    if (it == octx->host_staging_by_idx.end()) return;
    host_staging_pool_put(octx, std::move(it->second));
    octx->host_staging_by_idx.erase(it);
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
    octx->direct_read_fn       = nullptr;  // ggml-opencl lazy_init 按 env 注入
    octx->retain_host_staging  = false;
    octx->host_staging_pool_limit = 0;
    octx->host_staging_pool_bytes = 0;
    octx->host_staging_by_idx.clear();
    octx->host_staging_pool_by_size.clear();
    octx->host_staging_pool_order.clear();
    octx->bytes_loaded_total   = 0;
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

    // DMA 源解析: LOAD stage 已执行时优先用 host staging；否则默认 mmap host_ptr
    // (隐式 page fault 读盘)。 若注入了
    // direct_read_fn (GGML_ELASTIC_DIRECT_IO=1), 先 O_DIRECT pread 到 thread_local
    // scratch (绕 page cache, 模拟真 disk 成本), 再用 scratch 做 DMA 源。
    const void *dma_src = meta->host_ptr;
    bool use_direct = false;
    auto staged = octx->host_staging_by_idx.find(idx);
    if (staged != octx->host_staging_by_idx.end() && staged->second.size() >= meta->byte_size) {
        dma_src = staged->second.data();
        use_direct = true; // staging lifetime is owned by octx, so async DMA is safe.
    } else if (octx->direct_read_fn) {
        static thread_local std::vector<char> direct_scratch;
        if (direct_scratch.size() < meta->byte_size) direct_scratch.resize(meta->byte_size);
        if (octx->direct_read_fn(meta->host_ptr, direct_scratch.data(), meta->byte_size) == 0) {
            dma_src    = direct_scratch.data();
            use_direct = true;
        } else {
            std::fprintf(stderr, "[wbmcl] direct_read_fn 失败 idx=%d, fallback mmap\n", idx);
        }
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
                                       0, meta->byte_size, dma_src,
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
    static const bool s_use_host_ptr_env = []() {
        const char *e = std::getenv("GGML_ELASTIC_USE_HOST_PTR");
        return e && *e && *e != '0';
    }();
    // direct 模式跟 USE_HOST_PTR 互斥: direct 要把 scratch 显式 DMA 上去,
    // USE_HOST_PTR 是让 driver 直接绑 mmap 指针 (不会读 scratch)。direct 优先。
    const bool s_use_host_ptr = s_use_host_ptr_env && !use_direct;

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
        // direct 模式: dma_src 是复用的 thread_local scratch, 非阻塞写会在下次
        // ensure_resident 覆盖 scratch 前来不及读完 → 用阻塞写保证 scratch 安全。
        cl_event write_ev = nullptr;
        err = clEnqueueWriteBuffer(use_q, buf, use_direct ? CL_TRUE : CL_FALSE,
                                   0, meta->byte_size, dma_src,
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
                                   0, meta->byte_size, dma_src,
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

    // 默认不等 queue drain (MRU + 每 token sampling barrier 保证 victim 已 idle,
    // 详见 wbmcl_evict_batch)。GGML_ELASTIC_EVICT_WAIT=1 恢复保守等待。
    static const bool s_evict_wait = []() {
        const char *e = std::getenv("GGML_ELASTIC_EVICT_WAIT");
        return e && *e && *e != '0';
    }();
    if (s_evict_wait) {
        cl_event marker = nullptr;
        cl_int merr = clEnqueueMarkerWithWaitList(octx->compute_queue, 0, nullptr, &marker);
        if (merr == CL_SUCCESS && marker) {
            clWaitForEvents(1, &marker);
            clReleaseEvent(marker);
        } else {
            clFinish(octx->compute_queue);
        }
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

int wbmcl_load_host(wbm_opencl_ctx *octx, int idx) {
    if (!octx || !octx->wbm) return -1;
    const block_meta *meta = wbm_get(octx->wbm, idx);
    if (!meta) return -2;
    if (!meta->host_ptr || meta->byte_size == 0) return -3;

    auto & staging = octx->host_staging_by_idx[idx];
    if (staging.size() < meta->byte_size) {
        std::vector<char> pooled = host_staging_pool_take(octx, meta->byte_size);
        if (!pooled.empty()) {
            staging = std::move(pooled);
        } else {
            staging.resize(meta->byte_size);
        }
    }

    int rc = -1;
    if (octx->direct_read_fn) {
        rc = octx->direct_read_fn(meta->host_ptr, staging.data(), meta->byte_size);
    }
    if (rc != 0) {
        std::memcpy(staging.data(), meta->host_ptr, meta->byte_size);
    }
    octx->bytes_loaded_total += meta->byte_size;
    return 0;
}

int wbmcl_dma_to_backend(wbm_opencl_ctx *octx, int idx) {
    if (octx && octx->soa_per_idx.find(idx) != octx->soa_per_idx.end()) {
        return 0;
    }
    int rc = wbmcl_ensure_resident(octx, idx);
    if (rc == 0) {
        release_host_staging(octx, idx);
    }
    return rc;
}

int wbmcl_transform_backend(wbm_opencl_ctx *octx, int idx) {
    if (!octx || !octx->wbm) return -1;
    auto it = octx->soa_per_idx.find(idx);
    if (it != octx->soa_per_idx.end() && it->second.reload_fn) {
        const block_meta *meta = wbm_get(octx->wbm, idx);
        if (meta && meta->resident) {
            release_host_staging(octx, idx);
            return 0;
        }
        int rc = it->second.reload_fn();
        if (rc == 0) release_host_staging(octx, idx);
        return rc;
    }
    release_host_staging(octx, idx);
    return 0;
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

    // evict 前是否等 queue drain。默认 *不等*：
    //   MRU 策略下 victim = 上一个 graph 最近用过的 weight；而 decode 每个 token
    //   结尾要 sampling，读 logits 会隐式 drain 整个 graph → 下个 token 的 hook
    //   进来做 evict 时，上个 graph 的所有 kernel (含 victim 最后一次使用) 必已完成。
    //   所以插 marker + clWaitForEvents 是纯 overhead。
    //   GGML_ELASTIC_EVICT_WAIT=1 恢复保守等待 (无 sampling barrier 的场景 / 调试)。
    static const bool s_evict_wait = []() {
        const char *e = std::getenv("GGML_ELASTIC_EVICT_WAIT");
        return e && *e && *e != '0';
    }();
    if (s_evict_wait) {
        // in-order queue 顺序保证：queue 末尾插 marker，等它 = 等所有 prior kernel
        // (含最后用 victim 的 kernel) 完成。
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
            std::fprintf(stderr, "[wbmcl] enqueueMarker 失败: %s (%d) —— 退回 clFinish\n",
                         cl_err(merr), merr);
            clFinish(octx->compute_queue);
        }
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
    octx->host_staging_by_idx.clear();
    octx->host_staging_pool_by_size.clear();
    octx->host_staging_pool_order.clear();
    octx->host_staging_pool_bytes = 0;
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

void wbmcl_register_soa(wbm_opencl_ctx *octx, int idx,
                        std::function<int()> evict_fn,
                        std::function<int()> reload_fn) {
    if (!octx) return;
    octx->soa_per_idx[idx] = soa_callbacks{std::move(evict_fn), std::move(reload_fn)};
}

}  // namespace elastic
