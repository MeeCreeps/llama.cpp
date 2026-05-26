// runtime/weight_buffer_manager_cpu.cpp —— 见 weight_buffer_manager_cpu.h

#include "weight_buffer_manager_cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace elastic {

int wbmcpu_init(wbm_cpu_ctx *cctx, weight_buffer_manager *wbm, int n_workers) {
    if (!cctx || !wbm) return -1;
    cctx->wbm                  = wbm;
    cctx->bytes_uploaded_total = 0;
    cctx->bytes_evicted_total  = 0;
    cctx->n_mallocs            = 0;
    cctx->n_frees              = 0;
    cctx->n_workers            = n_workers < 0 ? 0 : n_workers;
    return 0;
}

int wbmcpu_ensure_resident(wbm_cpu_ctx *cctx, int idx) {
    if (!cctx || !cctx->wbm) return -1;
    const block_meta *meta = wbm_get(cctx->wbm, idx);
    if (!meta) return -2;
    if (meta->resident) return 0;
    if (!meta->host_ptr || meta->byte_size == 0) {
        std::fprintf(stderr, "[wbmcpu] block %d 未注册或 byte_size=0\n", idx);
        return -3;
    }

    void *buf = std::malloc(meta->byte_size);
    if (!buf) {
        std::fprintf(stderr, "[wbmcpu] malloc 失败 block %d size %zu\n",
                     idx, meta->byte_size);
        return -4;
    }
    cctx->n_mallocs += 1;

    // GGUF mmap 已经是 RAM 里，memcpy 直接走 CPU L1/L2 cache
    std::memcpy(buf, meta->host_ptr, meta->byte_size);
    cctx->bytes_uploaded_total += meta->byte_size;

    wbm_mark_resident(cctx->wbm, idx, buf);
    return 0;
}

int wbmcpu_evict(wbm_cpu_ctx *cctx, int idx) {
    if (!cctx || !cctx->wbm) return -1;
    const block_meta *meta = wbm_get(cctx->wbm, idx);
    if (!meta) return -2;
    if (!meta->resident) return 0;

    void *buf = meta->backend_handle;
    if (buf) {
        std::free(buf);
        cctx->n_frees += 1;
        cctx->bytes_evicted_total += meta->byte_size;
    }
    wbm_mark_evicted(cctx->wbm, idx);
    return 0;
}

int wbmcpu_prefetch(wbm_cpu_ctx *cctx, int idx) {
    // 首版同步实现，等价于 ensure_resident。
    // TODO: worker thread 池做 async memcpy，让 prefetch 跟主线程 compute overlap
    return wbmcpu_ensure_resident(cctx, idx);
}

int wbmcpu_evict_batch(wbm_cpu_ctx *cctx, const int *victims, int n_victims) {
    if (!cctx || !cctx->wbm || !victims || n_victims <= 0) return 0;
    int released = 0;
    for (int i = 0; i < n_victims; ++i) {
        const int v = victims[i];
        const block_meta *m = wbm_get(cctx->wbm, v);
        if (!m || !m->resident) continue;
        if (m->backend_handle) {
            std::free(m->backend_handle);
            cctx->n_frees += 1;
            cctx->bytes_evicted_total += m->byte_size;
        }
        wbm_mark_evicted(cctx->wbm, v);
        ++released;
    }
    return released;
}

void * wbmcpu_get_buffer(const wbm_cpu_ctx *cctx, int idx) {
    if (!cctx || !cctx->wbm) return nullptr;
    const block_meta *meta = wbm_get(cctx->wbm, idx);
    if (!meta || !meta->resident) return nullptr;
    return meta->backend_handle;
}

void wbmcpu_shutdown(wbm_cpu_ctx *cctx) {
    if (!cctx || !cctx->wbm) return;
    for (auto &b : cctx->wbm->blocks) {
        if (b.resident && b.backend_handle) {
            std::free(b.backend_handle);
            cctx->n_frees += 1;
            cctx->bytes_evicted_total += b.byte_size;
            b.backend_handle = nullptr;
            b.resident = false;
        }
    }
    cctx->wbm->n_resident     = 0;
    cctx->wbm->resident_bytes = 0;
    cctx->wbm                 = nullptr;
}

}  // namespace elastic
