// ggml-cpu-elastic.cpp
//
// CPU elastic backend：在 CPU compute 路径外加一层 WBM 弹性内存管理。
//
// 设计：
//   - 自己的 buffer type 分配一整块 region（跟普通 CPU buffer 一样）
//   - set_tensor 拷贝数据到 region 的 offset，同时把 mmap 源指针注册到 WBM
//   - graph_compute 前对每个 op 的 srcs 走 ensure_resident
//     - resident：no-op
//     - 已被 evict：memcpy(region+offset, mmap_src, size)
//   - 周期 evict 用 madvise(MADV_DONTNEED) 让 anonymous 页归还 OS
//   - compute 内部 delegate 到 ggml_backend_cpu_init() 拿的真 CPU backend

#include "ggml-cpu-elastic.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

// async worker / event infrastructure
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <thread>

#include "weight_buffer_manager.h"
#include "weight_buffer_manager_cpu.h"
#include "budget_watcher.h"
// O_DIRECT path (跟 GPU 共用 src/llama-mmap registry + pread_direct)
#include "../../../src/llama-mmap.h"
#include "../../../src/llama-uring.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <thread>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <unordered_map>
#include <vector>

namespace {

// ============================================================
// 单例 elastic state（与 ggml-opencl 的对偶，简化版）
// ============================================================
struct elastic_state {
    elastic::weight_buffer_manager wbm;
    elastic::budget_watcher        bw;

    bool   wbm_inited      = false;
    bool   bw_inited       = false;

    size_t kv_bytes        = 128ULL * 1024 * 1024;
    size_t misc_overhead   = 256ULL * 1024 * 1024;
    size_t static_target   = 0;       // = M_floor - kv - misc + extra_target
    size_t extra_target    = 0;       // EMBED_OUTSIDE_BUDGET 等加进来
    int    evict_interval  = 8;

    int    prefetch_lookahead = 0;    // posix_madvise lookahead

    // 统计
    uint64_t n_op_dispatched = 0;
    uint64_t n_reloads_total = 0;
    uint64_t n_evicts_total  = 0;
    size_t   bytes_reloaded_total = 0;
    size_t   bytes_evicted_total  = 0;

    // GGML_ELASTIC_PROFILE=1 时 graph_compute 累计三段时间
    bool     profile = false;
    double   profile_io_total_ms      = 0.0;  // ensure_node loop 总时间（含 memcpy reload）
    double   profile_compute_total_ms = 0.0;  // delegate compute 时间
    double   profile_overhead_total_ms = 0.0; // 其它（evict + bookkeeping）
    uint64_t profile_n_graph = 0;
};

elastic_state * get_state() {
    static elastic_state s;
    return &s;
}

// pinned 策略 / EMBED_OUTSIDE_BUDGET
std::string s_pin_policy() {
    const char *e = std::getenv("GGML_ELASTIC_PIN");
    return e ? std::string(e) : std::string("norm,k,v");
}
bool s_embed_out() {
    const char *e = std::getenv("GGML_ELASTIC_EMBED_OUTSIDE_BUDGET");
    return e && *e && *e != '0';
}

std::string tensor_suffix(const char *name) {
    if (!name) return {};
    std::string s(name);
    if (s.size() >= 4 && s.compare(0, 4, "blk.") == 0) {
        size_t d = s.find('.', 4);
        if (d != std::string::npos) s = s.substr(d + 1);
    }
    if (s.size() > 7 && s.compare(s.size() - 7, 7, ".weight") == 0) {
        s = s.substr(0, s.size() - 7);
    }
    return s;
}

// ============================================================
// Buffer context: 一整块 region + tensor → wbm_idx 映射
// ============================================================
struct elastic_buffer_ctx {
    void   *base;
    size_t  size;
    std::unordered_map<const ggml_tensor *, int> tensor_to_wbm;  // 反查
    std::vector<bool>                            block_pinned;   // wbm_idx → pinned
};

// 工具：page-align 一段范围 + madvise DONTNEED
void madvise_dontneed(void *ptr, size_t len) {
    static const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    uintptr_t addr   = (uintptr_t)ptr;
    uintptr_t aligned = (addr + pgsz - 1) & ~(pgsz - 1);  // round UP
    uintptr_t end    = (addr + len) & ~(pgsz - 1);        // round DOWN
    if (end > aligned) {
        ::madvise((void *)aligned, end - aligned, MADV_DONTNEED);
    }
}

// 同样 page-align 触发 WILLNEED
void madvise_willneed(void *ptr, size_t len) {
    static const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    uintptr_t addr   = (uintptr_t)ptr;
    uintptr_t aligned = addr & ~(pgsz - 1);
    size_t    over   = addr - aligned;
    size_t    raw    = len + over;
    size_t    a_len  = ((raw + pgsz - 1) / pgsz) * pgsz;
    ::posix_madvise((void *)aligned, a_len, POSIX_MADV_WILLNEED);
}

// ============================================================
// ensure_resident: 把 block idx 的数据 memcpy 回它的 region 槽
// 调用前要保证 wbm 已 init 且 idx 注册
// ============================================================
void ensure_block_resident(elastic_state *s, int wbm_idx, void *backend_handle) {
    if (wbm_idx < 0) return;
    const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, wbm_idx);
    if (!bm) return;
    if (bm->resident) return;
    if (!bm->host_ptr || !backend_handle) return;

    // GGML_ELASTIC_DIRECT_IO=1: 用 O_DIRECT pread 从 disk 真读 (绕 page cache),
    // 模拟 model > RAM 场景的真实 IO 成本.
    static const bool s_direct_io = []() {
        const char *e = std::getenv("GGML_ELASTIC_DIRECT_IO");
        return e && *e && *e != '0';
    }();
    bool used_direct = false;
    if (s_direct_io) {
        auto reg = llama_mmap_registry_find(bm->host_ptr);
        if (!reg.filename.empty()) {
            size_t file_offset = (const char*)bm->host_ptr - (const char*)reg.base;
            int rc = llama_pread_direct(reg.filename.c_str(), backend_handle, file_offset, bm->byte_size);
            if (rc == 0) used_direct = true;
        }
    }
    if (!used_direct) {
        // memcpy mmap → region 的 (backend_handle 已经是 region+offset 指针)
        std::memcpy(backend_handle, bm->host_ptr, bm->byte_size);
    }
    s->n_reloads_total += 1;
    s->bytes_reloaded_total += bm->byte_size;
    elastic::wbm_mark_resident(&s->wbm, wbm_idx, backend_handle);
}

// evict 列表：madvise DONTNEED + mark_evicted
void evict_blocks(elastic_state *s, elastic_buffer_ctx *bctx,
                  const std::vector<int> &victims) {
    for (int v : victims) {
        const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, v);
        if (!bm || !bm->resident) continue;
        // backend_handle 是 region+offset 指针；madvise DONTNEED 让 OS 把
        // anonymous page 回收，但 backend_handle 自身保持有效（下次 ensure
        // 时 memcpy 回来）
        // DEBUG: print first bytes BEFORE evict
        if (std::getenv("GGML_ELASTIC_DEBUG_RELOAD") && s->n_evicts_total < 10) {
            const uint8_t *hd = (const uint8_t *)bm->backend_handle;
            const uint8_t *hp = (const uint8_t *)bm->host_ptr;
            std::fprintf(stderr, "[elastic-debug evict] idx=%d size=%zu handle=%p[%02x %02x %02x %02x] host_ptr=%p[%02x %02x %02x %02x] %s\n",
                         v, bm->byte_size, bm->backend_handle,
                         hd[0],hd[1],hd[2],hd[3], bm->host_ptr,
                         hp[0],hp[1],hp[2],hp[3],
                         (std::memcmp(bm->backend_handle, bm->host_ptr, 16)==0)?"EQ":"DIFF");
        }
        // DEBUG: GGML_ELASTIC_NO_MADVISE=1 跳过实际页面回收，只 mark_evicted
        if (!std::getenv("GGML_ELASTIC_NO_MADVISE")) {
            madvise_dontneed(bm->backend_handle, bm->byte_size);
        }
        s->bytes_evicted_total += bm->byte_size;
        elastic::wbm_mark_evicted(&s->wbm, v);
        // mark_evicted 把 backend_handle 清掉了，但我们需要它保留（地址永不变）
        // 重新设上去：直接修改 blocks[]（注意是 hack——绕过 mark_evicted 的清零）
        // 更干净的做法是不调 mark_evicted，自己更新字段，避免清 backend_handle
    }
    s->n_evicts_total += victims.size();

    GGML_UNUSED(bctx);  // 暂未用到
}

// ============================================================
// Buffer interface
// ============================================================
const char * elastic_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "CPU_Elastic";
}

// O_DIRECT 需要 dst 4096 byte 对齐才能 pread 直写, 省掉 bounce buffer memcpy.
// 32 (默认 ggml 对齐) 对 SIMD 足够, 但 pread direct 必须 4096. 改成 4096
// 让 region 内每个 tensor 起始地址都 page-aligned. 内存浪费可忽略 (每 tensor
// 多 ~4 KB padding, 3B model ~280 tensors × 4 KB = 1.1 MB).
size_t elastic_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 4096;  // O_DIRECT direct-pread 需要; 也包含 SIMD 32B 对齐要求
}

bool elastic_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;  // 数据在 host 内存，compute 直接读
}

void elastic_buffer_free(ggml_backend_buffer_t buffer) {
    auto *bctx = (elastic_buffer_ctx *)buffer->context;
    if (bctx->base) std::free(bctx->base);
    delete bctx;
}

void * elastic_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto *bctx = (elastic_buffer_ctx *)buffer->context;
    return bctx->base;
}

void elastic_buffer_set_tensor(ggml_backend_buffer_t buffer,
                               ggml_tensor *tensor,
                               const void *data, size_t offset, size_t size) {
    auto *bctx = (elastic_buffer_ctx *)buffer->context;
    // DEBUG: 记录所有 set_tensor 调用
    static int dbg_count = 0;
    if (std::getenv("GGML_ELASTIC_DEBUG")) {
        if (dbg_count < 30 || (size_t)offset != 0 || (size_t)size != ggml_nbytes(tensor)) {
            std::fprintf(stderr, "[elastic-debug set_tensor] #%d buf=%p name=%s data=%p offset=%zu size=%zu nbytes=%zu view_src=%p full_overwrite=%d\n",
                         dbg_count, (void*)buffer, tensor->name, data, offset, size,
                         ggml_nbytes(tensor), (void*)tensor->view_src,
                         (offset == 0 && size == ggml_nbytes(tensor)) ? 1 : 0);
        }
        dbg_count++;
    }
    // 标准 CPU 行为：先把数据 memcpy 进 region
    std::memcpy((char *)tensor->data + offset, data, size);

    // 仅在"完整覆盖写"时注册 WBM（views / partial writes 跳过）
    auto *s = get_state();
    if (!s->wbm_inited) {
        if (elastic::wbm_init(&s->wbm, 0) != 0) {
            GGML_LOG_ERROR("elastic: wbm_init 失败\n");
            return;
        }
        s->wbm_inited = true;

        // 读 env 配置
        if (const char *kv = std::getenv("GGML_ELASTIC_KV_MB")) {
            s->kv_bytes = (size_t)std::atoll(kv) * 1024 * 1024;
        }
        if (const char *mc = std::getenv("GGML_ELASTIC_MISC_MB")) {
            s->misc_overhead = (size_t)std::atoll(mc) * 1024 * 1024;
        }
        if (const char *iv = std::getenv("GGML_ELASTIC_EVICT_INTERVAL")) {
            int v = std::atoi(iv);
            if (v >= 1) s->evict_interval = v;
        }
        if (const char *pf = std::getenv("GGML_ELASTIC_PREFETCH")) {
            int v = std::atoi(pf);
            if (v > 0) s->prefetch_lookahead = v;
        }
        // 默认 MRU. LLM decode 是 round-robin (layer 0..N..0..N), LRU 总把"马上
        // 再用的"踢出 → 命中率近 0; MRU 把"刚用的"踢出, 下次 cycle 再装回来.
        // 实测 3B F16 B=5000: LRU 2546 ms/tok, MRU 699 (3.6× 差距).
        // GGML_ELASTIC_EVICT_POLICY=lru 可显式切回 LRU 调试用.
        s->wbm.evict_mru = true;
        if (const char *policy = std::getenv("GGML_ELASTIC_EVICT_POLICY")) {
            std::string ps(policy);
            if (ps == "lru") {
                s->wbm.evict_mru = false;
                GGML_LOG_INFO("elastic: LRU policy (explicit override)\n");
            } else if (ps == "mru") {
                GGML_LOG_INFO("elastic: MRU policy (default)\n");
            }
        }
        if (const char *p = std::getenv("GGML_ELASTIC_PROFILE"); p && *p && *p != '0') {
            s->profile = true;
            std::atexit([]() {
                auto *st = get_state();
                if (!st->profile || st->profile_n_graph == 0) return;
                double io_pt   = st->profile_io_total_ms / st->profile_n_graph;
                double comp_pt = st->profile_compute_total_ms / st->profile_n_graph;
                double bytes_pt = (double)st->bytes_reloaded_total / st->profile_n_graph;
                double bw = st->profile_io_total_ms > 0 ?
                            (st->bytes_reloaded_total / 1024.0 / 1024.0) /
                            (st->profile_io_total_ms / 1000.0) : 0;
                std::fprintf(stderr,
                    "\n=== CPU_Elastic profile dump (n_graph=%llu) ===\n"
                    "  io_pt        : %.1f ms/graph (memcpy reload from mmap)\n"
                    "  compute_pt   : %.1f ms/graph (delegate to ggml-cpu)\n"
                    "  bytes_pt     : %.1f MB/graph reloaded\n"
                    "  effective_bw : %.0f MB/s\n"
                    "  reload_count : %llu (%llu bytes total)\n"
                    "  evict_count  : %llu (%llu bytes total)\n"
                    "============================================\n",
                    (unsigned long long)st->profile_n_graph,
                    io_pt, comp_pt,
                    bytes_pt / 1024 / 1024,
                    bw,
                    (unsigned long long)st->n_reloads_total,
                    (unsigned long long)st->bytes_reloaded_total,
                    (unsigned long long)st->n_evicts_total,
                    (unsigned long long)st->bytes_evicted_total);
            });
        }
        const char *csv = std::getenv("GGML_ELASTIC_BUDGET_CSV");
        if (csv && *csv) {
            if (elastic::budget_watcher_init(&s->bw, csv) == 0) {
                s->bw_inited = true;
                size_t mfloor = s->bw.m_floor_mb * 1024 * 1024;
                size_t kvm    = s->kv_bytes + s->misc_overhead;
                s->static_target = mfloor > kvm ? mfloor - kvm : 0;
                GGML_LOG_INFO("elastic: BudgetWatcher trace=%s M_floor=%zu MB -> static_target=%zu MB\n",
                              csv, s->bw.m_floor_mb, s->static_target / 1024 / 1024);
            }
        }
    }

    if (tensor->view_src == nullptr && offset == 0 && size == ggml_nbytes(tensor)) {
        // host_ptr = data (mmap 源指针；llama_model_loader 生命周期内有效)
        int idx = elastic::wbm_add_block(&s->wbm, const_cast<void *>(data), size);
        if (idx >= 0) {
            // backend_handle = tensor->data（region 内地址，永不变）
            elastic::wbm_mark_resident(&s->wbm, idx, tensor->data);
            elastic::wbm_touch(&s->wbm, idx, 0);
            bctx->tensor_to_wbm[tensor] = idx;
            while ((int)bctx->block_pinned.size() <= idx) bctx->block_pinned.push_back(false);

            // pin policy
            std::string suf = tensor_suffix(tensor->name);
            std::string pp  = s_pin_policy();
            bool should_pin = false;
            if (pp == "all") {
                should_pin = true;  // 真的所有 tensor 都 pin
            } else {
                auto contains = [&](const char *t) {
                    return pp.find(t) != std::string::npos;
                };
                if (contains("norm") && (suf == "attn_norm" || suf == "ffn_norm" || suf == "output_norm")) should_pin = true;
                if (contains("k") && suf == "attn_k") should_pin = true;
                if (contains("v") && suf == "attn_v") should_pin = true;
                if (contains("q") && suf == "attn_q") should_pin = true;
                if (contains("o") && suf == "attn_output") should_pin = true;
            }
            // 外部 schedule hook (LP solver / 自定义): callback 返回 true 则覆盖 env policy
            // 也强制 pin. callback 为空时 fall back 到 env policy. 不允许 callback 解 pin
            // (env pin 是基线), 但允许 callback 额外 pin.
            if (llama_weight_pin_query(tensor->name, -1, size)) {
                should_pin = true;
            }
            if (should_pin) {
                elastic::wbm_set_pinned(&s->wbm, idx, true);
                bctx->block_pinned[idx] = true;
            }
            // EMBED_OUTSIDE_BUDGET
            if (s_embed_out() && suf == "token_embd") {
                elastic::wbm_set_pinned(&s->wbm, idx, true);
                bctx->block_pinned[idx] = true;
                s->extra_target += size;
                s->static_target += size;  // 同步更新已算好的 target
            }
        }
    }
}

void elastic_buffer_get_tensor(ggml_backend_buffer_t buffer,
                               const ggml_tensor *tensor,
                               void *data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(data, (const char *)tensor->data + offset, size);
}

void elastic_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor,
                                  uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memset((char *)tensor->data + offset, value, size);
}

bool elastic_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                               const ggml_tensor *src, ggml_tensor *dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

void elastic_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto *bctx = (elastic_buffer_ctx *)buffer->context;
    std::memset(bctx->base, value, buffer->size);
}

const ggml_backend_buffer_i elastic_buffer_i = {
    /* .free_buffer     = */ elastic_buffer_free,
    /* .get_base        = */ elastic_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ elastic_buffer_memset_tensor,
    /* .set_tensor      = */ elastic_buffer_set_tensor,
    /* .get_tensor      = */ elastic_buffer_get_tensor,
    /* .cpy_tensor      = */ elastic_buffer_cpy_tensor,
    /* .clear           = */ elastic_buffer_clear,
    /* .reset           = */ nullptr,
};

ggml_backend_buffer_t elastic_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // Region 4096 对齐让 O_DIRECT pread 能 direct 写到 region+offset (省 bounce memcpy)
    void *data = std::aligned_alloc(4096, (size + 4095) & ~size_t(4095));
    if (!data) {
        GGML_LOG_ERROR("elastic: alloc %zu 失败\n", size);
        return nullptr;
    }
    auto *bctx = new elastic_buffer_ctx{data, size, {}, {}};
    return ggml_backend_buffer_init(buft, elastic_buffer_i, bctx, size);
}

const ggml_backend_buffer_type_i elastic_buffer_type_i = {
    /* .get_name         = */ elastic_buffer_type_get_name,
    /* .alloc_buffer     = */ elastic_buffer_type_alloc_buffer,
    /* .get_alignment    = */ elastic_buffer_type_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ nullptr,
    /* .is_host          = */ elastic_buffer_type_is_host,
};

ggml_backend_buffer_type g_elastic_buffer_type = {
    /* .iface   = */ elastic_buffer_type_i,
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

// ============================================================
// Backend (stream) interface：delegate compute 到 CPU backend
// ============================================================
struct elastic_backend_ctx {
    ggml_backend_t cpu;  // 真 CPU backend
};

const char * elastic_backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "CPU_Elastic";
}

void elastic_backend_free(ggml_backend_t backend) {
    auto *bctx = (elastic_backend_ctx *)backend->context;
    if (bctx->cpu) ggml_backend_free(bctx->cpu);
    delete bctx;
    delete backend;
}

// graph_compute: 每个 op 前 ensure_resident，然后 delegate
ggml_status elastic_backend_graph_compute(ggml_backend_t backend, ggml_cgraph *cgraph) {
    auto *bctx = (elastic_backend_ctx *)backend->context;
    auto *s    = get_state();

    if (!s->wbm_inited) {
        // 模型完全 fit RAM、没有 WBM 注册 → 直接 delegate
        return bctx->cpu->iface.graph_compute(bctx->cpu, cgraph);
    }

    // DEBUG counters (must be declared BEFORE lambdas that capture them)
    static int dbg_graph_n = 0;
    int dbg = (std::getenv("GGML_ELASTIC_DEBUG") != nullptr);
    uint64_t pre_reloads = s->n_reloads_total;
    uint64_t pre_evicts  = s->n_evicts_total;
    size_t   pre_resident = s->wbm.resident_bytes;
    int      dbg_resolved = 0, dbg_total_srcs = 0, dbg_non_resident = 0;

    // PROFILE timers
    using clk = std::chrono::steady_clock;
    clk::time_point t_total_start = clk::now();
    clk::time_point t_io_start{};
    if (s->profile) t_io_start = clk::now();

    // 工具：从 tensor 找它的 wbm_idx + buffer ctx + handle。如果 t 本身是
    // view 没注册，跟着 view_src 链找到原始 tensor（reload 时要把原始 weight
    // 的数据 restore 回原 memory 位置——view 的 data 指针落在那段 memory 里）。
    auto resolve = [](const ggml_tensor *t, int &wbm_idx, void *&handle) {
        wbm_idx = -1; handle = nullptr;
        if (!t || !t->buffer) return false;
        if (t->buffer->iface.set_tensor != elastic_buffer_set_tensor) return false;
        auto *bc = (elastic_buffer_ctx *)t->buffer->context;
        // 跟 view chain
        const ggml_tensor *root = t;
        while (root && root->view_src) root = root->view_src;
        auto it = bc->tensor_to_wbm.find(root);
        if (it == bc->tensor_to_wbm.end()) return false;
        wbm_idx = it->second;
        handle  = root->data;  // 原始 tensor 的 data 位置（reload 目标）
        return true;
    };

    auto target_bytes = [&]() {
        if (!s->bw_inited) return SIZE_MAX;  // 没 trace → 不 evict
        return s->static_target;  // baseline 静态
    };

    // 处理一个 node 的所有 srcs
    auto ensure_node = [&](ggml_tensor *n) {
        if (!n) return;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor *src = n->src[j];
            if (!src) continue;
            dbg_total_srcs++;
            int idx; void *h;
            if (!resolve(src, idx, h)) continue;
            dbg_resolved++;
            const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
            if (!bm) continue;
            if (!bm->resident) {
                dbg_non_resident++;
                // 不做 pre-evict！CPU backend 的 compute 是整图一次跑（黑盒），
                // 我们不知道每个 tensor 在 compute 内部什么时候被读。pre-evict
                // 可能把后续 op 还要用的 tensor 提前丢弃，compute 读到 zeros。
                // 容忍 resident 在 ensure 阶段涨过 target，compute 完成后再 evict。
                ensure_block_resident(s, idx, h);
            }
            elastic::wbm_touch(&s->wbm, idx, 0);  // global counter 内部递增
        }
    };

    // 1) 先扫一遍所有 node 的 srcs ensure_resident。CPU backend 的 compute 是
    //    "黑盒"——一次跑整个 graph，没法插中间 evict。所以我们必须先把所有
    //    需要的 src 都加载好，compute 期间不能动它们。
    // 2) prefetch 期间 madvise WILLNEED 给后 N 个 node 的 srcs 预读 mmap 页
    elastic_buffer_ctx *some_ctx = nullptr;
    auto pick_some_ctx = [&](ggml_tensor *node) {
        if (some_ctx || !node) return;
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            ggml_tensor *src = node->src[k];
            if (src && src->buffer &&
                src->buffer->iface.set_tensor == elastic_buffer_set_tensor) {
                some_ctx = (elastic_buffer_ctx *)src->buffer->context;
                break;
            }
        }
    };

    // GGML_ELASTIC_CHUNK_SIZE=N: 把 graph 切成 N node 一段, 段间插 worker thread
    // 跑 NEXT chunk 的 ensure_phase, 主线程跑 CURRENT chunk 的 graph_compute.
    // 真正实现 IO/compute overlap. =0 (default) 走原 monolithic 路径.
    static const int s_chunk_size = []() {
        const char *e = std::getenv("GGML_ELASTIC_CHUNK_SIZE");
        return (e && *e) ? std::atoi(e) : 0;
    }();

    if (s_chunk_size > 0) {
        // GGML_ELASTIC_URING=1: 用 io_uring 替代 std::thread worker, kernel 异步执行
        // pread, 主线程跑 compute. 无 thread spawn/join 开销, 无 DRAM 竞争 (UFS DMA
        // 走独立 hardware path). Linux/Android kernel >= 5.1.
        static const bool s_uring = []() {
            const char *e = std::getenv("GGML_ELASTIC_URING");
            return e && *e && *e != '0';
        }();

        const int n_nodes = cgraph->n_nodes;
        // Sync ensure (worker thread 或 fallback 用)
        auto ensure_chunk_sync = [&](int i0, int i1) {
            for (int i = i0; i < i1; i++) {
                ggml_tensor *n = cgraph->nodes[i];
                if (n) for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    ggml_tensor *src = n->src[k];
                    if (!src) continue;
                    int idx; void *h;
                    if (!resolve(src, idx, h)) continue;
                    const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
                    if (!bm || bm->resident) continue;
                    ensure_block_resident(s, idx, h);
                }
            }
        };

        // Async ensure via io_uring: submit aligned reads, fall back sync for unaligned.
        // 返回 submitted block 列表 (caller 在 wait_all 后 mark_resident).
        auto ensure_chunk_uring_submit = [&](int i0, int i1, std::vector<std::pair<int,size_t>> &pending) {
            for (int i = i0; i < i1; i++) {
                ggml_tensor *n = cgraph->nodes[i];
                if (!n) continue;
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    ggml_tensor *src = n->src[k];
                    if (!src) continue;
                    int idx; void *h;
                    if (!resolve(src, idx, h)) continue;
                    const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
                    if (!bm || bm->resident || !bm->host_ptr || !bm->backend_handle) continue;
                    auto reg = llama_mmap_registry_find(bm->host_ptr);
                    if (reg.filename.empty()) {
                        ensure_block_resident(s, idx, h);
                        continue;
                    }
                    size_t file_offset = (const char*)bm->host_ptr - (const char*)reg.base;
                    // 必须 4096-aligned 才能 io_uring + O_DIRECT 直写
                    const bool ok = ((uintptr_t)bm->backend_handle & 4095) == 0
                                 && (file_offset & 4095) == 0
                                 && (bm->byte_size & 4095) == 0;
                    if (ok && llama_uring::submit_pread_aligned(reg.filename.c_str(),
                                                                  bm->backend_handle, file_offset, bm->byte_size) == 0) {
                        pending.emplace_back(idx, bm->byte_size);
                    } else {
                        ensure_block_resident(s, idx, h);
                    }
                }
            }
        };
        // Wait for first `n` completions, mark those pending blocks resident.
        // Uses io_uring wait_n so future-chunk submissions stay in-flight.
        auto mark_first_n_resident = [&](std::vector<std::pair<int,size_t>> &pending, int n) {
            if (n <= 0 || pending.empty()) return;
            llama_uring::wait_n(n);
            for (int i = 0; i < n && i < (int)pending.size(); ++i) {
                int idx = pending[i].first;
                const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
                if (bm && bm->backend_handle) {
                    elastic::wbm_mark_resident(&s->wbm, idx, bm->backend_handle);
                    s->n_reloads_total += 1;
                    s->bytes_reloaded_total += pending[i].second;
                }
            }
            pending.erase(pending.begin(), pending.begin() + std::min(n, (int)pending.size()));
        };

        // GGML_ELASTIC_URING_LOOKAHEAD=K: 预提交 K 个 chunk 的 IO 让 kernel 用 UFS
        // queue depth>1 并行处理. K=1 (default) = 上面老行为. K=2-4 让 disk read
        // 跟 compute 更深 pipeline.
        static const int s_lookahead = []() {
            const char *e = std::getenv("GGML_ELASTIC_URING_LOOKAHEAD");
            return (e && *e) ? std::max(1, std::atoi(e)) : 1;
        }();

        int i0 = 0, i1 = std::min(s_chunk_size, n_nodes);
        ensure_chunk_sync(i0, i1);  // first chunk fully sync
        pick_some_ctx(cgraph->nodes[i0]);

        ggml_status st_chunk = GGML_STATUS_SUCCESS;
        // Per chunk pending list. pending[k] = chunks N+1, N+2, ..., N+lookahead.
        // Index 0 always = next chunk's pending (to wait before computing next).
        std::vector<std::vector<std::pair<int,size_t>>> pending_q;

        // Initial: pre-submit lookahead chunks 1..lookahead
        if (s_uring) {
            int sub_i = i1;
            for (int la = 0; la < s_lookahead && sub_i < n_nodes; ++la) {
                int e = std::min(sub_i + s_chunk_size, n_nodes);
                pending_q.emplace_back();
                ensure_chunk_uring_submit(sub_i, e, pending_q.back());
                sub_i = e;
            }
        }
        // Track next submission point (one past last pre-submitted chunk)
        int next_sub_i0 = i1 + s_chunk_size * (int)pending_q.size();

        while (i0 < n_nodes) {
            const int next_i0 = i1;
            const int next_i1 = std::min(i1 + s_chunk_size, n_nodes);

            if (s_uring) {
                // Submit chunk lookahead-ahead (if any left), so UFS queue stays deep
                int new_sub_end = std::min(next_sub_i0 + s_chunk_size, n_nodes);
                if (next_sub_i0 < n_nodes) {
                    pending_q.emplace_back();
                    ensure_chunk_uring_submit(next_sub_i0, new_sub_end, pending_q.back());
                    next_sub_i0 = new_sub_end;
                }

                // Compute current chunk in parallel with kernel processing pre-submitted IOs
                struct ggml_cgraph chunk = ggml_graph_view(cgraph, i0, i1);
                st_chunk = bctx->cpu->iface.graph_compute(bctx->cpu, &chunk);

                // Wait for next chunk's IO (front of pending_q) before computing it next iter
                if (!pending_q.empty()) {
                    int n_to_wait = (int)pending_q.front().size();
                    mark_first_n_resident(pending_q.front(), n_to_wait);
                    pending_q.erase(pending_q.begin());
                }
            } else {
                // worker thread path (fallback, 默认)
                std::thread worker;
                if (next_i0 < n_nodes) {
                    worker = std::thread([&ensure_chunk_sync, next_i0, next_i1]() {
                        ensure_chunk_sync(next_i0, next_i1);
                    });
                }
                struct ggml_cgraph chunk = ggml_graph_view(cgraph, i0, i1);
                st_chunk = bctx->cpu->iface.graph_compute(bctx->cpu, &chunk);
                if (worker.joinable()) worker.join();
            }
            if (st_chunk != GGML_STATUS_SUCCESS) break;

            i0 = next_i0;
            i1 = next_i1;
        }
        // Skip the monolithic path; jump to post-compute evict section using goto-ish
        // (mimicked by setting st and skipping below).
        // Run the post-compute housekeeping by falling through to evict section below.
        if (s->profile) {
            // Roughly attribute: count ensure as IO, graph_compute as compute.
            // (we don't have fine-grained per-chunk timing without more invasive code)
            s->profile_n_graph += 1;
        }
        // 4) evict to target
        if (s->bw_inited && some_ctx) {
            const size_t target = target_bytes();
            if (target != SIZE_MAX && s->wbm.resident_bytes > target) {
                std::vector<int> victims;
                int n_v = elastic::wbm_evict_to_byte_budget(&s->wbm, target, -1, &victims);
                if (n_v > 0) evict_blocks(s, some_ctx, victims);
            }
        }
        dbg_graph_n++;
        return st_chunk;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ensure_node(cgraph->nodes[i]);
        pick_some_ctx(cgraph->nodes[i]);
        // CPU prefetch (madvise WILLNEED) 给后 N 个 node 的 srcs
        if (s->prefetch_lookahead > 0) {
            int maxj = std::min(i + 1 + s->prefetch_lookahead, cgraph->n_nodes);
            for (int j = i + 1; j < maxj; ++j) {
                ggml_tensor *nj = cgraph->nodes[j];
                if (!nj) continue;
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    ggml_tensor *src = nj->src[k];
                    if (!src) continue;
                    int idx; void *h;
                    if (!resolve(src, idx, h)) continue;
                    const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
                    if (!bm || bm->resident) continue;
                    madvise_willneed(bm->host_ptr, bm->byte_size);
                }
            }
        }
    }

    if (dbg) {
        // 在 compute 前再扫一次：所有 resident block 的 byte 跟 host_ptr 是否一致
        int n_check = 0, n_mismatch = 0;
        for (const auto &b : s->wbm.blocks) {
            if (!b.resident || !b.backend_handle || !b.host_ptr) continue;
            n_check++;
            if (std::memcmp(b.backend_handle, b.host_ptr, b.byte_size) != 0) {
                n_mismatch++;
                if (n_mismatch <= 3) {
                    std::fprintf(stderr, "[elastic-debug PRE-COMPUTE MISMATCH] idx=%d size=%zu\n",
                                 b.block_idx, b.byte_size);
                }
            }
        }
        std::fprintf(stderr, "[elastic-debug graph#%d pre-compute] nodes=%d resolved=%d non_resident=%d reloads=+%llu resident_MB=%zu  pre-compute integrity: checked=%d mismatch=%d\n",
                     dbg_graph_n, cgraph->n_nodes, dbg_resolved, dbg_non_resident,
                     (unsigned long long)(s->n_reloads_total - pre_reloads),
                     s->wbm.resident_bytes / 1024 / 1024, n_check, n_mismatch);
    }
    // 3) 数据齐了 → delegate compute
    clk::time_point t_compute_start{};
    if (s->profile) {
        auto t_io_end = clk::now();
        s->profile_io_total_ms += std::chrono::duration<double, std::milli>(t_io_end - t_io_start).count();
        t_compute_start = t_io_end;
    }
    ggml_status st = bctx->cpu->iface.graph_compute(bctx->cpu, cgraph);
    if (s->profile) {
        auto t_compute_end = clk::now();
        s->profile_compute_total_ms += std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
    }

    // 4) compute 完成后 evict 到 target（为下一次 graph_compute 腾位置）。
    //    "下一个 graph 用到的 block" 会在它的 ensure 阶段 reload 回来。
    if (s->bw_inited && some_ctx) {
        const size_t target = target_bytes();
        if (target != SIZE_MAX && s->wbm.resident_bytes > target) {
            std::vector<int> victims;
            int n_v = elastic::wbm_evict_to_byte_budget(&s->wbm, target, -1, &victims);
            if (n_v > 0) evict_blocks(s, some_ctx, victims);
        }
    }
    if (dbg) {
        std::fprintf(stderr, "[elastic-debug graph#%d post] evicts=%llu(+%llu) resident_MB=%zu\n",
                     dbg_graph_n, (unsigned long long)s->n_evicts_total,
                     (unsigned long long)(s->n_evicts_total - pre_evicts),
                     s->wbm.resident_bytes / 1024 / 1024);
    }
    if (s->profile) {
        double total = std::chrono::duration<double, std::milli>(clk::now() - t_total_start).count();
        double accounted = s->profile_io_total_ms + s->profile_compute_total_ms;
        // overhead accumulates only the delta beyond io+compute for this graph
        // (post-evict + bookkeeping). Re-derive from total minus per-graph io+compute
        // (we tracked totals not per-graph, so compute delta differently)
        (void)total; (void)accounted;
        s->profile_n_graph += 1;
    }
    dbg_graph_n++;
    return st;
}

// ============================================================
// Async worker: 让 ggml-cpu compute 跑在后台线程, 主线程 graph_compute_async
// 立刻返回. 配合 event_record/wait 让 ggml-sched 在 CPU 跑期间能去 dispatch GPU.
// 开关: LLAMA_ELASTIC_CPU_ASYNC=1
//
// 设计:
//   - 全局单 worker thread + 任务 queue (per backend instance)
//   - 任务种类: COMPUTE(cgraph), EVENT_RECORD(event), EVENT_WAIT(event)
//   - 主线程: graph_compute_async push COMPUTE, 立刻返回
//   - Worker: pop 任务, 顺序处理, 保留 ggml-sched 期望的依赖序
// ============================================================
struct elastic_event_state {
    std::mutex              m;
    std::condition_variable cv;
    bool                    signaled = false;
};

struct elastic_async_task {
    enum kind_t { COMPUTE, EVENT_RECORD, EVENT_WAIT } kind;
    ggml_backend_t          backend = nullptr;
    ggml_cgraph *           cgraph  = nullptr;
    elastic_event_state *   event   = nullptr;
};

struct elastic_async_worker {
    std::thread             th;
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<elastic_async_task> queue;
    std::atomic<bool>       stop{false};
    bool                    started = false;
};

static elastic_async_worker * get_async_worker() {
    static elastic_async_worker w;
    return &w;
}

static void async_worker_loop(elastic_async_worker *w) {
    while (!w->stop.load(std::memory_order_acquire)) {
        elastic_async_task task;
        {
            std::unique_lock<std::mutex> lk(w->mtx);
            w->cv.wait(lk, [&]{ return w->stop.load(std::memory_order_acquire) || !w->queue.empty(); });
            if (w->stop.load(std::memory_order_acquire)) return;
            task = std::move(w->queue.front());
            w->queue.pop_front();
        }
        switch (task.kind) {
            case elastic_async_task::COMPUTE: {
                // Direct call to underlying CPU backend (skip elastic ensure since
                // sync ensure_phase already happened on main thread before enqueue).
                auto *bctx = (elastic_buffer_ctx *) task.backend->context;
                // Re-enter our own elastic graph_compute path (which itself ensures + computes)
                // 不直接调 bctx->cpu, 因为 ensure_phase 还要做.
                extern ggml_status elastic_backend_graph_compute(ggml_backend_t, ggml_cgraph *);
                elastic_backend_graph_compute(task.backend, task.cgraph);
                (void)bctx;
                break;
            }
            case elastic_async_task::EVENT_RECORD: {
                std::lock_guard<std::mutex> lk(task.event->m);
                task.event->signaled = true;
                task.event->cv.notify_all();
                break;
            }
            case elastic_async_task::EVENT_WAIT: {
                std::unique_lock<std::mutex> lk(task.event->m);
                task.event->cv.wait(lk, [&]{ return task.event->signaled; });
                break;
            }
        }
    }
}

static void async_worker_start() {
    auto *w = get_async_worker();
    static std::once_flag f;
    std::call_once(f, [&]{ w->th = std::thread(async_worker_loop, w); w->started = true; });
}
static void async_worker_enqueue(elastic_async_task t) {
    auto *w = get_async_worker();
    {
        std::lock_guard<std::mutex> lk(w->mtx);
        w->queue.push_back(std::move(t));
    }
    w->cv.notify_one();
}
static bool async_enabled() {
    static const bool enabled = []{
        const char *e = std::getenv("LLAMA_ELASTIC_CPU_ASYNC");
        return e && *e && *e != '0';
    }();
    return enabled;
}

// Async-aware wrappers (default fall back sync if async disabled)
ggml_status elastic_backend_graph_compute_async(ggml_backend_t backend, ggml_cgraph *cgraph) {
    if (!async_enabled()) {
        extern ggml_status elastic_backend_graph_compute(ggml_backend_t, ggml_cgraph *);
        return elastic_backend_graph_compute(backend, cgraph);
    }
    async_worker_start();
    elastic_async_task t{};
    t.kind = elastic_async_task::COMPUTE;
    t.backend = backend;
    t.cgraph  = cgraph;
    async_worker_enqueue(std::move(t));
    return GGML_STATUS_SUCCESS;  // dispatched, completion via events
}

void elastic_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    (void)backend;
    if (!async_enabled()) return;  // sync mode: event always considered ready
    async_worker_start();
    elastic_async_task t{};
    t.kind = elastic_async_task::EVENT_RECORD;
    t.event = (elastic_event_state *) event->context;
    async_worker_enqueue(std::move(t));
}

void elastic_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    (void)backend;
    if (!async_enabled()) return;
    async_worker_start();
    elastic_async_task t{};
    t.kind = elastic_async_task::EVENT_WAIT;
    t.event = (elastic_event_state *) event->context;
    async_worker_enqueue(std::move(t));
}

// Device-level event APIs
ggml_backend_event_t elastic_device_event_new(ggml_backend_dev_t dev) {
    auto *st = new elastic_event_state();
    auto *e = new ggml_backend_event{ dev, st };
    return e;
}
void elastic_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    (void)dev;
    delete (elastic_event_state *) event->context;
    delete event;
}
void elastic_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    (void)dev;
    auto *st = (elastic_event_state *) event->context;
    std::unique_lock<std::mutex> lk(st->m);
    st->cv.wait(lk, [&]{ return st->signaled; });
}

void elastic_backend_synchronize(ggml_backend_t backend) {
    (void)backend;
    if (!async_enabled()) return;
    // 等 worker queue 清空: enqueue 一个 event, wait it
    auto *st = new elastic_event_state();
    auto *e = new ggml_backend_event{ nullptr, st };
    elastic_backend_event_record(backend, e);
    {
        std::unique_lock<std::mutex> lk(st->m);
        st->cv.wait(lk, [&]{ return st->signaled; });
    }
    delete st;
    delete e;
}

const ggml_backend_i elastic_backend_i = {
    /* .get_name                = */ elastic_backend_get_name,
    /* .free                    = */ elastic_backend_free,
    // NOTE: async path 已实现 (elastic_backend_graph_compute_async + event_* +
    // worker thread) 但接到 iface 后引起 ggml-cpu ops.cpp 越界 assert — ggml-sched
    // 调 compute_async 返回后, 主线程继续准备下一 split, 共享 cgraph 内部状态
    // (tensor->data 等) 被 mutate, worker 还在读 → race. 要修需要 deep copy cgraph
    // 或加更严同步. 暂时回退 sync iface, 代码留着.
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ elastic_backend_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

ggml_guid_t elastic_backend_guid(void) {
    static ggml_guid guid = { 0xe1, 0xa5, 0x71, 0xc0, 0x57, 0x3e, 0x10, 0xa0,
                              0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0 };
    return &guid;
}

// ============================================================
// Device + reg (最小实现，让 backend registry 能找到)
// ============================================================
const char * elastic_device_get_name(ggml_backend_dev_t dev)        { GGML_UNUSED(dev); return "CPU_Elastic"; }
const char * elastic_device_get_description(ggml_backend_dev_t dev) { GGML_UNUSED(dev); return "CPU backend with WBM elastic memory management"; }
void         elastic_device_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total) {
    GGML_UNUSED(dev);
    *free = *total = 8ULL * 1024 * 1024 * 1024;  // 占位 8 GB
}
enum ggml_backend_dev_type elastic_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // ACCEL（跟 BLAS 一样）：不替换默认 CPU device，避免被 main.cpp 当作
    // 标准 CPU 调 ggml_backend_cpu_set_abort_callback 等 CPU-only 入口。
    // 选用：CLI --device CPU_Elastic 或 -ngl 1 把权重 offload 到我们。
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}
void elastic_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props *props) {
    props->name        = elastic_device_get_name(dev);
    props->description = elastic_device_get_description(dev);
    props->type        = elastic_device_get_type(dev);
    elastic_device_get_memory(dev, &props->memory_free, &props->memory_total);
    // async iface 暂未启用 (race issue, 见 backend_i 注释), caps 保持 sync.
    props->caps = { /*async*/false, /*host_buffer*/false, /*buffer_from_host_ptr*/true, /*events*/false };
}
ggml_backend_buffer_type_t elastic_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_elastic_buffer_type();
}
ggml_backend_buffer_t elastic_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void *ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev); GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}
bool elastic_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor *op) {
    GGML_UNUSED(dev); GGML_UNUSED(op);
    return true;  // 跟 CPU 一样支持所有 op
}
bool elastic_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    // 只接 elastic 自己的 buffer 和标准 CPU buffer；拒绝 CPU_REPACK / KleidiAI
    // 等 extra 类型——它们的 set_tensor 走自己的 repack 路径，我们对应不上。
    if (buft == ggml_backend_cpu_elastic_buffer_type()) return true;
    if (buft == ggml_backend_cpu_buffer_type())         return true;
    return false;
}
ggml_backend_t elastic_device_init_backend(ggml_backend_dev_t dev, const char *params) {
    GGML_UNUSED(dev); GGML_UNUSED(params);
    return ggml_backend_cpu_elastic_init();
}

const ggml_backend_device_i elastic_device_i = {
    /* .get_name             = */ elastic_device_get_name,
    /* .get_description      = */ elastic_device_get_description,
    /* .get_memory           = */ elastic_device_get_memory,
    /* .get_type             = */ elastic_device_get_type,
    /* .get_props            = */ elastic_device_get_props,
    /* .init_backend         = */ elastic_device_init_backend,
    /* .get_buffer_type      = */ elastic_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ elastic_device_buffer_from_host_ptr,
    /* .supports_op          = */ elastic_device_supports_op,
    /* .supports_buft        = */ elastic_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

const char * elastic_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "CPU_Elastic";
}
size_t elastic_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}
ggml_backend_dev_t elastic_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = {
        /* .iface   = */ elastic_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &dev;
}

void * elastic_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    GGML_UNUSED(reg);
    // 不能直接转发 set_abort_callback / get_features 等以 backend 实例为
    // 参数的入口——那些函数里 GGML_ASSERT(is_cpu(backend)) 会炸（我们不是
    // 标准 cpu）。安全做法：只转发不带 backend 实例的入口（threadpool_new
    // / threadpool_free 等）。
    if (!name) return nullptr;
    static const char *passthrough[] = {
        "ggml_threadpool_new",
        "ggml_threadpool_free",
        "ggml_threadpool_pause",
        "ggml_threadpool_resume",
        nullptr,
    };
    for (int i = 0; passthrough[i]; ++i) {
        if (std::strcmp(name, passthrough[i]) == 0) {
            return ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), name);
        }
    }
    return nullptr;
}

const ggml_backend_reg_i elastic_reg_i = {
    /* .get_name         = */ elastic_reg_get_name,
    /* .get_device_count = */ elastic_reg_get_device_count,
    /* .get_device       = */ elastic_reg_get_device,
    /* .get_proc_address = */ elastic_reg_get_proc_address,
};

}  // anon

// ============================================================
// Public API
// ============================================================
ggml_backend_buffer_type_t ggml_backend_cpu_elastic_buffer_type(void) {
    return &g_elastic_buffer_type;
}

ggml_backend_t ggml_backend_cpu_elastic_init(void) {
    auto *ctx = new elastic_backend_ctx;
    ctx->cpu  = ggml_backend_cpu_init();
    if (!ctx->cpu) { delete ctx; return nullptr; }

    auto *backend = new ggml_backend{
        /* .guid    = */ elastic_backend_guid(),
        /* .iface   = */ elastic_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_elastic_reg(), 0),
        /* .context = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_cpu_elastic(ggml_backend_t backend) {
    return backend && ggml_guid_matches(backend->guid, elastic_backend_guid());
}

ggml_backend_reg_t ggml_backend_cpu_elastic_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ elastic_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}
