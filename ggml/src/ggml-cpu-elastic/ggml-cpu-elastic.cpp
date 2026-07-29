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
#include "ggml-cpu/repack.h"
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
#include "elastic_granularity.h"
#include "elastic_profile_writer.h"
// O_DIRECT path (跟 GPU 共用 src/llama-mmap registry + pread_direct)
#include "../../../src/llama-mmap.h"
#include "../../../src/llama-uring.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <thread>
#include <string>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <list>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace {

// fwd decls
struct elastic_buffer_ctx;

struct elastic_tensor_part {
    int     wbm_idx    = -1;
    int64_t row_start  = 0;
    int64_t row_count  = 0;
    size_t  byte_offset = 0;
    size_t  byte_size   = 0;
};

struct elastic_tensor_units {
    std::vector<elastic_tensor_part> parts;
    bool row_cut = false;
};

struct elastic_cpu_tensor_slice {
    ggml_tensor * tensor = nullptr;
    int64_t row_start = 0;
    int64_t row_count = 0;
    bool repacked = false;
};

// Reusable CPU LOAD buffer whose visible pointer has the same page offset as
// the source file offset.  llama_pread_direct() can then read the aligned
// middle directly into this buffer instead of allocating/copying a full-size
// bounce buffer.  The owning allocation remains 4096-aligned and move-only.
struct cpu_staging_buffer {
    void * allocation = nullptr;
    uint8_t * ptr = nullptr;
    size_t nbytes = 0;
    size_t io_bias = 0;

    cpu_staging_buffer() = default;
    cpu_staging_buffer(const cpu_staging_buffer &) = delete;
    cpu_staging_buffer & operator=(const cpu_staging_buffer &) = delete;

    cpu_staging_buffer(cpu_staging_buffer && other) noexcept {
        *this = std::move(other);
    }

    cpu_staging_buffer & operator=(cpu_staging_buffer && other) noexcept {
        if (this == &other) return *this;
        reset();
        allocation = other.allocation;
        ptr = other.ptr;
        nbytes = other.nbytes;
        io_bias = other.io_bias;
        other.allocation = nullptr;
        other.ptr = nullptr;
        other.nbytes = 0;
        other.io_bias = 0;
        return *this;
    }

    ~cpu_staging_buffer() { reset(); }

    void reset() {
        std::free(allocation);
        allocation = nullptr;
        ptr = nullptr;
        nbytes = 0;
        io_bias = 0;
    }

    bool allocate(size_t size, size_t bias) {
        reset();
        constexpr size_t alignment = 4096;
        void * base = nullptr;
        if (bias >= alignment ||
            posix_memalign(&base, alignment, size + alignment) != 0) {
            return false;
        }
        allocation = base;
        ptr = static_cast<uint8_t *>(base) + bias;
        nbytes = size;
        io_bias = bias;
        return true;
    }

    uint8_t * data() { return ptr; }
    const uint8_t * data() const { return ptr; }
    size_t size() const { return nbytes; }
    bool empty() const { return ptr == nullptr || nbytes == 0; }
};

struct cpu_stage_unit_task {
    uint64_t id = 0;
    std::vector<int> indices;
    std::vector<std::pair<int, int>> fused_pairs;
};

struct cpu_pipeline_successor {
    uint64_t signature = 0;
    std::vector<std::vector<int>> units;
    std::vector<std::pair<int, int>> fused_pairs;
};

struct cpu_pipeline_pin_state {
    size_t refs = 0;
    bool original = false;
};

enum class cpu_aux_task_kind {
    COPY,
    REPACK,
};

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
    size_t static_target   = 0;       // = M_floor - kv - misc - safety + extra_target
    size_t extra_target    = 0;       // EMBED_OUTSIDE_BUDGET 等加进来
    bool   dynamic_target  = false;   // GGML_ELASTIC_DYNAMIC=1: 用 B(t) 实时算 target
    int    evict_interval  = 8;

    int    prefetch_lookahead = 0;    // posix_madvise lookahead

    elastic::granularity_config granularity;
    bool granularity_logged = false;
    uint64_t eviction_group_generation = UINT64_MAX;
    uint64_t granularity_units = 0;
    uint64_t granularity_cut_ops = 0;
    uint64_t granularity_fallback_ops = 0;
    size_t granularity_peak_unit_bytes = 0;
    uint64_t granularity_registered_tensors = 0;
    uint64_t granularity_cut_tensors = 0;
    uint64_t granularity_registered_parts = 0;
    uint64_t unit_pipeline_issued = 0;
    uint64_t unit_pipeline_ready = 0;
    uint64_t unit_pipeline_waits = 0;
    uint64_t unit_pipeline_wait_us = 0;
    uint64_t unit_pipeline_current_residency_us = 0;
    uint64_t unit_pipeline_current_missing_units = 0;
    uint64_t unit_pipeline_current_missing_tensors = 0;
    size_t unit_pipeline_current_missing_bytes = 0;
    uint64_t unit_pipeline_unissued_missing_units = 0;
    uint64_t unit_pipeline_unissued_missing_tensors = 0;
    size_t unit_pipeline_unissued_missing_bytes = 0;
    uint64_t unit_pipeline_unissued_missing_us = 0;
    uint64_t unit_pipeline_stage_us = 0;
    uint64_t unit_pipeline_retire_us = 0;
    uint64_t unit_pipeline_window_samples = 0;
    uint64_t unit_pipeline_window_units_total = 0;
    size_t unit_pipeline_window_bytes_total = 0;
    size_t unit_pipeline_window_bytes_max = 0;
    size_t unit_pipeline_window_units_max = 0;
    uint64_t unit_pipeline_oversize_windows = 0;
    std::unordered_map<uint64_t, cpu_pipeline_successor> unit_pipeline_successors;
    uint64_t unit_pipeline_previous_graph_signature = 0;
    bool unit_pipeline_previous_graph_valid = false;
    std::deque<cpu_pipeline_successor> unit_pipeline_cross_graphs;
    std::unordered_map<int, cpu_pipeline_pin_state> unit_pipeline_cross_pins;
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
    uint64_t async_prepare_cap_declines = 0;
    uint64_t async_prepare_cap_declined_tensors = 0;
    uint64_t fused_layout_pair_calls = 0;
    uint64_t fused_layout_pair_ok = 0;
    uint64_t fused_layout_pair_fallbacks = 0;
    uint64_t fused_layout_parallel_calls = 0;
    uint64_t fused_layout_pair_us = 0;
    size_t   fused_layout_pair_bytes = 0;
    uint64_t fused_kernel_pair_candidates = 0;
    uint64_t fused_kernel_pair_calls = 0;
    uint64_t fused_kernel_pair_fallbacks = 0;
    uint64_t fused_kernel_pair_errors = 0;
    uint64_t fused_kernel_pair_us = 0;
    uint64_t fused_scan_mul_mat = 0;
    uint64_t fused_scan_shared_activation = 0;
    uint64_t fused_scan_compatible = 0;
    uint64_t fused_scan_weight_units = 0;
    uint64_t fused_scan_safe_output = 0;

    // 统计
    uint64_t n_op_dispatched = 0;
    uint64_t n_reloads_total = 0;
    uint64_t n_evicts_total  = 0;
    size_t   bytes_reloaded_total = 0;
    size_t   bytes_evicted_total  = 0;
    uint64_t evict_wall_us = 0;
    uint64_t direct_read_calls = 0;
    uint64_t direct_read_ok    = 0;
    uint64_t direct_read_fail  = 0;
    uint64_t direct_read_us    = 0;
    size_t   direct_read_bytes = 0;
    uint64_t direct_batch_units = 0;
    uint64_t direct_batch_tensors = 0;
    uint64_t direct_batch_fallbacks = 0;
    uint64_t direct_batch_errors = 0;
    std::mutex io_stats_mtx;
    uint64_t stage_load_calls  = 0;
    uint64_t stage_load_ok     = 0;
    uint64_t stage_load_us     = 0;
    size_t   stage_load_bytes  = 0;
    uint64_t stage_xform_calls = 0;
    uint64_t stage_xform_ok    = 0;
    uint64_t stage_xform_us    = 0;
    size_t   stage_xform_bytes = 0;
    uint64_t stage_repack_calls = 0;
    uint64_t stage_repack_ok    = 0;
    uint64_t stage_repack_us    = 0;
    size_t   stage_repack_bytes = 0;
    uint64_t stage_materialize_calls = 0;
    uint64_t stage_materialize_us    = 0;
    size_t   stage_materialize_bytes = 0;
    uint64_t initial_repack_calls = 0;
    uint64_t initial_repack_us    = 0;
    size_t   initial_repack_bytes = 0;

    // GGML_ELASTIC_PROFILE=1 / GGML_ELASTIC_TIMING=1 时 graph_compute 累计三段时间
    bool     profile = false;
    bool     profile_csv = false;
    double   profile_io_total_ms      = 0.0;  // ensure_node loop 总时间（含 memcpy reload）
    double   profile_compute_total_ms = 0.0;  // delegate compute 时间
    uint64_t profile_compute_calls = 0;
    double   profile_overhead_total_ms = 0.0; // 其它（evict + bookkeeping）
    uint64_t profile_n_graph = 0;

    // === Runtime scheduler integration ===
    // name → wbm_idx for residency query / movement request from llama_context.
    // bctx_by_idx allows movement request to find the backend handle on demand.
    std::mutex                              sched_mtx;
    std::unordered_map<std::string, int>    name_to_wbm;
    std::unordered_map<std::string, std::vector<int>> name_to_wbms;
    std::unordered_map<int, std::string>    wbm_to_name;
    std::unordered_map<int, elastic_buffer_ctx *> bctx_by_idx;
    std::unordered_map<int, void *>         backend_handle_by_idx;
    std::unordered_map<int, elastic_cpu_tensor_slice> tensor_slice_by_idx;
    std::unordered_map<int, cpu_staging_buffer>   staged_raw_by_idx;
    // Reusable LOAD-stage buffers.  Keep active buffers separate from the
    // size-class pool so PREPARE can return a consumed buffer without a
    // malloc/free (and zero-fill) on every working unit.
    bool                                    retain_host_staging = false;
    size_t                                  host_staging_pool_limit = 0;
    size_t                                  host_staging_pool_bytes = 0;
    std::unordered_map<size_t, std::vector<cpu_staging_buffer>> host_staging_pool_by_size;
    std::list<size_t>                       host_staging_pool_order;
    uint64_t                                host_staging_pool_hit = 0;
    uint64_t                                host_staging_pool_miss = 0;
    uint64_t                                host_staging_direct_compatible = 0;
    cpu_staging_buffer                      fused_compute_workspace;
    uint64_t                                fused_workspace_allocations = 0;
    uint64_t                                fused_workspace_reuses = 0;
    bool                                    sched_registered = false;

    // CPU stage pipeline state.  State values: 1=in-flight, 2=ok, <0=error.
    bool                                    async_load_worker_started = false;
    bool                                    async_load_shutdown = false;
    std::thread                             async_load_worker;
    std::condition_variable                 async_load_cv;
    std::deque<cpu_stage_unit_task>         async_load_queue;
    std::unordered_map<int, int>            async_load_state;
    uint64_t                                async_stage_next_unit_id = 1;
    uint64_t                                async_load_units_enqueued = 0;
    uint64_t                                async_load_units_completed = 0;
    uint64_t                                async_load_enqueued = 0;
    uint64_t                                async_load_completed = 0;
    uint64_t                                async_load_waits = 0;
    uint64_t                                async_load_wait_us = 0;
    // Persistent second O_DIRECT lane for a two-tensor Multi LOAD.  It reuses
    // the ordinary staging pool and avoids per-unit thread creation.
    bool                                    multi_io_worker_started = false;
    bool                                    multi_io_worker_shutdown = false;
    bool                                    multi_io_task_pending = false;
    bool                                    multi_io_task_done = true;
    int                                     multi_io_task_idx = -1;
    int                                     multi_io_task_rc = 0;
    std::thread                             multi_io_worker;
    std::mutex                              multi_io_mtx;
    std::condition_variable                 multi_io_cv;
    uint64_t                                multi_io_parallel_units = 0;
    uint64_t                                multi_io_parallel_tensors = 0;

    bool                                    async_prepare_worker_started = false;
    bool                                    async_prepare_shutdown = false;
    std::thread                             async_prepare_worker;
    std::condition_variable                 async_prepare_cv;
    std::deque<cpu_stage_unit_task>         async_prepare_queue;
    std::unordered_map<int, int>            async_prepare_state;
    uint64_t                                async_prepare_units_enqueued = 0;
    uint64_t                                async_prepare_units_completed = 0;
    uint64_t                                async_prepare_enqueued = 0;
    uint64_t                                async_prepare_completed = 0;
    uint64_t                                async_prepare_waits = 0;
    uint64_t                                async_prepare_wait_us = 0;

    // Persistent second PREPARE lane. It is reused for pooled staging copies
    // and for the second tensor of a fused layout pair; no per-unit thread is
    // created.
    bool                                    copy_worker_started = false;
    bool                                    copy_worker_shutdown = false;
    bool                                    copy_task_pending = false;
    // No task is outstanding initially.  This lets the first PREPARE submit
    // immediately; subsequent submissions wait for the previous task.
    bool                                    copy_task_done = true;
    std::thread                             copy_worker;
    std::mutex                              copy_mtx;
    std::condition_variable                 copy_cv;
    void *                                  copy_dst = nullptr;
    const void *                            copy_src = nullptr;
    size_t                                  copy_bytes = 0;
    cpu_aux_task_kind                       copy_task_kind =
                                                cpu_aux_task_kind::COPY;
    ggml_tensor *                           copy_repack_tensor = nullptr;
    int                                     copy_task_rc = 0;
    uint64_t                                copy_parallel_calls = 0;
    uint64_t                                copy_parallel_bytes = 0;
};

elastic_state * get_state() {
    static elastic_state s;
    return &s;
}

struct cpu_fused_mul_mat_pair_call {
    ggml_tensor * first = nullptr;
    ggml_tensor * second = nullptr;
    void * workspace = nullptr;
    size_t workspace_size = 0;
    ggml_backend_cpu_repack_pair_sync sync;
    std::atomic<int> error{0};
};

static void cpu_fused_mul_mat_pair_callback(
        ggml_tensor * dst, int ith, int nth, void * userdata) {
    auto * call = static_cast<cpu_fused_mul_mat_pair_call *>(userdata);
    if (!call) return;
    const int rc = ggml_backend_cpu_repack_mul_mat_pair_compute_thread(
        call->first, call->second, call->workspace, call->workspace_size,
        ith, nth, &call->sync);
    if (rc != 0) call->error.store(rc, std::memory_order_release);
    GGML_UNUSED(dst);
}

static int lookup_wbm_idx_locked(elastic_state *s, const char *name) {
    if (!s || !name || !*name) return -1;
    auto it = s->name_to_wbm.find(name);
    if (it != s->name_to_wbm.end()) return it->second;
    std::string cpu_wrapped = std::string("CPU_Elastic#") + name + "#0";
    it = s->name_to_wbm.find(cpu_wrapped);
    if (it != s->name_to_wbm.end()) return it->second;
    return -1;
}

static std::vector<int> lookup_wbm_indices_locked(elastic_state *s, const char *name) {
    if (!s || !name || !*name) return {};
    auto many = s->name_to_wbms.find(name);
    if (many != s->name_to_wbms.end()) return many->second;
    const int one = lookup_wbm_idx_locked(s, name);
    return one >= 0 ? std::vector<int>{one} : std::vector<int>{};
}

// === Static handlers exposed to llama-mmap registry (called by llama_context) ===
// 单例 state 假设: 一个进程内仅有一个 elastic-cpu backend.
bool elastic_sched_residency_query(const char *name, void * /*ud*/) {
    auto *s = get_state();
    std::lock_guard<std::mutex> lk(s->sched_mtx);
    const std::vector<int> indices = lookup_wbm_indices_locked(s, name);
    if (indices.empty()) return false;
    return std::all_of(indices.begin(), indices.end(), [s](int idx) {
        const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
        return bm && bm->resident;
    });
}

uint32_t elastic_sched_state_query(const char *name, void * /*ud*/) {
    auto *s = get_state();
    std::lock_guard<std::mutex> lk(s->sched_mtx);
    const std::vector<int> indices = lookup_wbm_indices_locked(s, name);
    if (indices.empty()) return 0;
    uint32_t flags = 0;
    bool disk = true;
    bool compute = true;
    bool raw = true;
    for (int idx : indices) {
        const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
        disk = disk && bm && bm->host_ptr;
        compute = compute && bm && bm->resident;
        raw = raw && s->staged_raw_by_idx.find(idx) != s->staged_raw_by_idx.end();
    }
    if (disk) flags |= LLAMA_WEIGHT_STATE_DISK_AVAILABLE;
    if (compute) flags |= LLAMA_WEIGHT_STATE_CPU_COMPUTE_RESIDENT;
    if (raw) {
        flags |= LLAMA_WEIGHT_STATE_CPU_RAW_RESIDENT;
    }
    return flags;
}

// 接到 ensure_block_resident / evict_blocks 的同步路径. 在 graph_compute
// 流之外被调用 (e.g. llama_decode 的 scheduler hook), 必须线程安全.
// 这里直接调 sync 版本 — backend_handle 是 region+offset 永久指针, 安全.
extern void ensure_block_resident(elastic_state *s, int wbm_idx, void *backend_handle);
extern void evict_blocks(elastic_state *s, elastic_buffer_ctx *bctx,
                         const std::vector<int> &victims);
void cpu_profile_record(const char *kind, const char *name, int idx,
                        size_t bytes, double ms, int ok,
                        const char *extra);

int read_block_direct_or_mmap(elastic_state *s, const elastic::block_meta *bm, void *dst) {
    if (!s || !bm || !bm->host_ptr || !dst) return -1;

    static const bool s_direct_io = []() {
        const char *e = std::getenv("GGML_ELASTIC_DIRECT_IO");
        return !(e && *e == '0');
    }();

    bool used_direct = false;
    if (s_direct_io) {
        auto reg = llama_mmap_registry_find(bm->host_ptr);
        if (!reg.filename.empty()) {
            size_t file_offset = (const char *) bm->host_ptr - (const char *) reg.base;
            auto t0 = std::chrono::steady_clock::now();
            int rc = llama_pread_direct(reg.filename.c_str(), dst, file_offset, bm->byte_size);
            auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count();
            {
                std::lock_guard<std::mutex> stats(s->io_stats_mtx);
                s->direct_read_calls++;
                s->direct_read_us += (uint64_t) dt;
                if (rc == 0) {
                    s->direct_read_ok++;
                    s->direct_read_bytes += bm->byte_size;
                } else {
                    s->direct_read_fail++;
                }
            }
            used_direct = rc == 0;
        } else {
            std::lock_guard<std::mutex> stats(s->io_stats_mtx);
            s->direct_read_fail++;
        }
    }
    if (!used_direct) {
        std::memcpy(dst, bm->host_ptr, bm->byte_size);
    }
    return 0;
}

static bool cpu_async_stage_load_enabled() {
    static const bool enabled = []() {
        const char *e = std::getenv("GGML_ELASTIC_ASYNC_STAGE_LOAD");
        return e && *e && *e != '0';
    }();
    return enabled;
}

static bool cpu_async_stage_prepare_enabled() {
    static const bool enabled = []() {
        const char *e = std::getenv("GGML_ELASTIC_ASYNC_STAGE_PREPARE");
        return e && *e && *e != '0';
    }();
    return enabled;
}

static bool cpu_sync_stage_load_enabled() {
    static const bool enabled = []() {
        const char *e = std::getenv("GGML_ELASTIC_SYNC_STAGE_LOAD");
        return e && *e && *e != '0';
    }();
    return enabled;
}

static bool cpu_sync_stage_workers_enabled() {
    static const bool enabled = []() {
        const char *e = std::getenv("GGML_ELASTIC_SYNC_STAGE_WORKERS");
        return e && *e && *e != '0';
    }();
    return enabled;
}

static size_t cpu_async_prepare_max_pending() {
    static const size_t max_pending = []() {
        const char *e = std::getenv("GGML_ELASTIC_ASYNC_PREPARE_MAX_PENDING");
        long long v = e && *e ? atoll(e) : 0;
        if (v < 0) v = 0;
        return static_cast<size_t>(v);
    }();
    return max_pending;
}

static uint64_t cpu_now_us() {
    using clock = std::chrono::steady_clock;
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now().time_since_epoch()).count();
}

static void * cpu_fixed_handle_for_idx(elastic_state *s, int idx) {
    if (!s) return nullptr;
    std::lock_guard<std::mutex> lk(s->sched_mtx);
    auto hit = s->backend_handle_by_idx.find(idx);
    return hit == s->backend_handle_by_idx.end() ? nullptr : hit->second;
}

static std::string cpu_name_for_idx_copy(elastic_state *s, int idx) {
    if (!s) return {};
    std::lock_guard<std::mutex> lk(s->sched_mtx);
    auto it = s->wbm_to_name.find(idx);
    return it == s->wbm_to_name.end() ? std::string{} : it->second;
}

// Callers hold sched_mtx.  The policy mirrors the OpenCL host-staging pool:
// exact-size reuse plus a FIFO byte cap.  Exact sizes are intentional because
// vector::size() is also the validity check in the stage interface.
static void cpu_staging_pool_trim_locked(elastic_state *s, size_t incoming) {
    if (!s || s->host_staging_pool_limit == 0) return;
    while (s->host_staging_pool_bytes + incoming > s->host_staging_pool_limit &&
           !s->host_staging_pool_order.empty()) {
        const size_t old_size = s->host_staging_pool_order.front();
        s->host_staging_pool_order.pop_front();
        auto it = s->host_staging_pool_by_size.find(old_size);
        if (it == s->host_staging_pool_by_size.end() || it->second.empty()) continue;
        it->second.pop_back();
        s->host_staging_pool_bytes -= std::min(s->host_staging_pool_bytes, old_size);
        if (it->second.empty()) s->host_staging_pool_by_size.erase(it);
    }
}

static cpu_staging_buffer cpu_staging_pool_take_locked(
        elastic_state *s, size_t nbytes, size_t io_bias) {
    if (!s || !s->retain_host_staging) return {};
    auto it = s->host_staging_pool_by_size.find(nbytes);
    if (it == s->host_staging_pool_by_size.end() || it->second.empty()) {
        s->host_staging_pool_miss++;
        return {};
    }
    auto matching = std::find_if(it->second.rbegin(), it->second.rend(),
        [io_bias](const cpu_staging_buffer & buffer) {
            return buffer.io_bias == io_bias;
        });
    if (matching == it->second.rend()) {
        s->host_staging_pool_miss++;
        return {};
    }
    cpu_staging_buffer buffer = std::move(*matching);
    it->second.erase(std::next(matching).base());
    if (it->second.empty()) s->host_staging_pool_by_size.erase(it);
    for (auto order = s->host_staging_pool_order.rbegin();
         order != s->host_staging_pool_order.rend(); ++order) {
        if (*order == nbytes) {
            s->host_staging_pool_order.erase(std::next(order).base());
            break;
        }
    }
    s->host_staging_pool_bytes -= std::min(s->host_staging_pool_bytes, nbytes);
    s->host_staging_pool_hit++;
    return buffer;
}

static void cpu_staging_pool_put_locked(elastic_state *s, cpu_staging_buffer &&buffer) {
    if (!s || !s->retain_host_staging || buffer.empty()) return;
    const size_t nbytes = buffer.size();
    if (s->host_staging_pool_limit > 0 && nbytes > s->host_staging_pool_limit) return;
    cpu_staging_pool_trim_locked(s, nbytes);
    if (s->host_staging_pool_limit > 0 &&
        s->host_staging_pool_bytes + nbytes > s->host_staging_pool_limit) return;
    s->host_staging_pool_by_size[nbytes].push_back(std::move(buffer));
    s->host_staging_pool_order.push_back(nbytes);
    s->host_staging_pool_bytes += nbytes;
}

static size_t cpu_staging_io_bias(const elastic::block_meta * bm) {
    if (!bm || !bm->host_ptr) return 0;
    constexpr size_t alignment = 4096;
    const auto reg = llama_mmap_registry_find(bm->host_ptr);
    if (reg.filename.empty()) return 0;
    const size_t file_offset = static_cast<const char *>(bm->host_ptr) -
                               static_cast<const char *>(reg.base);
    return file_offset & (alignment - 1);
}

static void cpu_copy_staging_to_final(elastic_state *s, void * dst,
                                      const void * src, size_t nbytes);

static int elastic_cpu_stage_load_sync(elastic_state *s, const char *name, int idx) {
    if (!s || !name) return -1;
    const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
    if (!bm) return -3;
    if (bm->resident) return 0;
    const size_t io_bias = cpu_staging_io_bias(bm);
    cpu_staging_buffer staging;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        if (s->staged_raw_by_idx.find(idx) != s->staged_raw_by_idx.end()) return 0;
        staging = cpu_staging_pool_take_locked(s, bm->byte_size, io_bias);
    }
    auto t0 = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> stats(s->io_stats_mtx);
        s->stage_load_calls++;
    }
    if (staging.empty() && !staging.allocate(bm->byte_size, io_bias)) return -6;
    if ((reinterpret_cast<uintptr_t>(staging.data()) & 4095U) == io_bias) {
        std::lock_guard<std::mutex> stats(s->io_stats_mtx);
        s->host_staging_direct_compatible++;
    }
    int rc = read_block_direct_or_mmap(s, bm, staging.data());
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
    {
        std::lock_guard<std::mutex> stats(s->io_stats_mtx);
        s->stage_load_us += (uint64_t) dt;
    }
    if (rc != 0) {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        cpu_staging_pool_put_locked(s, std::move(staging));
        if (s->profile_csv) {
            cpu_profile_record("LOAD", name, idx, bm->byte_size, dt / 1000.0, 0, "plan_stage");
        }
        return rc;
    }
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        if (s->staged_raw_by_idx.find(idx) == s->staged_raw_by_idx.end()) {
            s->staged_raw_by_idx.emplace(idx, std::move(staging));
        } else {
            // A concurrent synchronous request won the race.  Keep one valid
            // copy and return the duplicate allocation to the pool.
            cpu_staging_pool_put_locked(s, std::move(staging));
        }
    }
    {
        std::lock_guard<std::mutex> stats(s->io_stats_mtx);
        s->stage_load_ok++;
        s->stage_load_bytes += bm->byte_size;
    }
    if (s->profile_csv) {
        cpu_profile_record("LOAD", name, idx, bm->byte_size, dt / 1000.0, 1, "plan_stage");
    }
    return 0;
}

// LOAD one real working unit. Metadata is temporary, but every byte buffer is
// obtained from the same reusable staging pool as the tensor path. A Multi
// unit uses one independent-offset batch submission when possible; PREPARE
// deliberately remains per tensor because Multi does not fuse layouts.
static int elastic_cpu_stage_load_unit_sync(
        elastic_state * s, const std::vector<int> & indices) {
    if (!s || indices.empty()) return 0;
    if (indices.size() == 1) {
        const int idx = indices.front();
        const std::string name = cpu_name_for_idx_copy(s, idx);
        return elastic_cpu_stage_load_sync(s, name.c_str(), idx);
    }

    struct staged_entry {
        int idx = -1;
        const elastic::block_meta * block = nullptr;
        std::string name;
        cpu_staging_buffer staging;
        llama_mmap_registry_entry registry;
        size_t file_offset = 0;
    };
    std::vector<staged_entry> entries;
    entries.reserve(indices.size());

    for (int idx : indices) {
        const elastic::block_meta * block = elastic::wbm_get(&s->wbm, idx);
        if (!block || block->resident) continue;
        const size_t io_bias = cpu_staging_io_bias(block);
        cpu_staging_buffer staging;
        {
            std::lock_guard<std::mutex> lk(s->sched_mtx);
            if (s->staged_raw_by_idx.find(idx) != s->staged_raw_by_idx.end()) continue;
            staging = cpu_staging_pool_take_locked(s, block->byte_size, io_bias);
        }
        if (staging.empty() && !staging.allocate(block->byte_size, io_bias)) {
            return -6;
        }
        if ((reinterpret_cast<uintptr_t>(staging.data()) & 4095U) == io_bias) {
            s->host_staging_direct_compatible++;
        }
        staged_entry entry;
        entry.idx = idx;
        entry.block = block;
        entry.name = cpu_name_for_idx_copy(s, idx);
        entry.staging = std::move(staging);
        entry.registry = llama_mmap_registry_find(block->host_ptr);
        if (!entry.registry.filename.empty()) {
            entry.file_offset = static_cast<const char *>(block->host_ptr) -
                                static_cast<const char *>(entry.registry.base);
        }
        entries.push_back(std::move(entry));
    }
    if (entries.empty()) return 0;

    const auto start = std::chrono::steady_clock::now();
    s->stage_load_calls += entries.size();
    bool all_direct = true;
    for (const staged_entry & entry : entries) {
        all_direct = all_direct && !entry.registry.filename.empty();
    }

    int rc = 0;
    if (all_direct && entries.size() > 1) {
        std::vector<llama_pread_request> requests;
        requests.reserve(entries.size());
        size_t total_bytes = 0;
        for (staged_entry & entry : entries) {
            requests.push_back({entry.registry.filename.c_str(), entry.staging.data(),
                                entry.file_offset, entry.block->byte_size});
            total_bytes += entry.block->byte_size;
        }
        s->direct_read_calls++;
        s->direct_batch_units++;
        s->direct_batch_tensors += entries.size();
        rc = llama_pread_direct_batch(requests.data(), requests.size());
        if (rc >= 0) {
            s->direct_read_ok++;
            s->direct_read_bytes += total_bytes;
            if (rc > 0) s->direct_batch_fallbacks++;
            rc = 0;
        } else {
            s->direct_read_fail++;
            s->direct_batch_errors++;
        }
    } else {
        for (staged_entry & entry : entries) {
            const int one_rc = read_block_direct_or_mmap(
                s, entry.block, entry.staging.data());
            if (one_rc != 0 && rc == 0) rc = one_rc;
        }
    }

    // A failed direct batch is allowed to fall back to the already-mapped
    // model only for correctness. The explicit error counter makes such a run
    // ineligible for the strict batch-I/O experiment.
    if (rc != 0) {
        for (staged_entry & entry : entries) {
            std::memcpy(entry.staging.data(), entry.block->host_ptr,
                        entry.block->byte_size);
        }
        rc = 0;
    }

    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    s->direct_read_us += all_direct && entries.size() > 1
        ? static_cast<uint64_t>(elapsed_us) : 0;
    s->stage_load_us += static_cast<uint64_t>(elapsed_us);

    for (staged_entry & entry : entries) {
        {
            std::lock_guard<std::mutex> lk(s->sched_mtx);
            if (s->staged_raw_by_idx.find(entry.idx) == s->staged_raw_by_idx.end()) {
                s->staged_raw_by_idx.emplace(entry.idx, std::move(entry.staging));
            } else {
                cpu_staging_pool_put_locked(s, std::move(entry.staging));
            }
        }
        s->stage_load_ok++;
        s->stage_load_bytes += entry.block->byte_size;
        if (s->profile_csv) {
            cpu_profile_record("LOAD", entry.name.c_str(), entry.idx,
                               entry.block->byte_size,
                               elapsed_us / 1000.0 / entries.size(), 1,
                               all_direct ? "unit_batch" : "unit_fallback");
        }
    }
    return 0;
}

int elastic_sched_movement_request(const char *name, bool evict, void * /*ud*/) {
    if (!name) return -1;
    auto *s = get_state();
    std::vector<int> indices;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        indices = lookup_wbm_indices_locked(s, name);
    }
    if (indices.empty()) return -2;
    if (evict) {
        if (llama_weight_runtime_desired_query(name) ==
                LLAMA_WEIGHT_RUNTIME_CPU) {
            return 0;
        }
        std::unordered_map<elastic_buffer_ctx *, std::vector<int>> by_buffer;
        {
            std::lock_guard<std::mutex> lk(s->sched_mtx);
            for (int idx : indices) {
                const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
                if (!bm || !bm->resident) continue;
                auto bit = s->bctx_by_idx.find(idx);
                if (bit == s->bctx_by_idx.end() || !bit->second) return -4;
                by_buffer[bit->second].push_back(idx);
            }
        }
        for (auto & entry : by_buffer) evict_blocks(s, entry.first, entry.second);
        llama_weight_runtime_mark_evicted(name, LLAMA_WEIGHT_RUNTIME_CPU);
        return 0;
    }
    for (int idx : indices) {
        const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
        if (!bm) return -3;
        if (bm->resident) continue;
        void *fixed_handle = cpu_fixed_handle_for_idx(s, idx);
        void *target = bm->backend_handle ? bm->backend_handle : fixed_handle;
        if (!target) return -5;
        ensure_block_resident(s, idx, target);
    }
    llama_weight_runtime_mark_resident(name, LLAMA_WEIGHT_RUNTIME_CPU);
    return 0;
}

static int elastic_cpu_prepare_resident(elastic_state *s, const char *name, int idx, void *fixed_handle) {
    if (!s || !name) return -1;
    const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
    if (!bm) return -3;
    if (bm->resident) return 0;
    void *target = bm->backend_handle ? bm->backend_handle : fixed_handle;
    if (!target) return -4;

    s->stage_xform_calls++;
    auto t0 = std::chrono::steady_clock::now();
    cpu_staging_buffer staging;
    elastic_cpu_tensor_slice slice;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        auto sit = s->tensor_slice_by_idx.find(idx);
        if (sit != s->tensor_slice_by_idx.end()) {
            slice = sit->second;
        }
        auto it = s->staged_raw_by_idx.find(idx);
        if (it != s->staged_raw_by_idx.end()) {
            if (it->second.size() != bm->byte_size) return -5;
            staging = std::move(it->second);
            s->staged_raw_by_idx.erase(it);
        }
    }
    // A repack cannot safely use the final destination as its raw read buffer:
    // the interleaver reads several source rows while overwriting the packed
    // row group.  Ensure that even a synchronous pipeline miss goes through a
    // pooled raw buffer.
    if (slice.repacked && staging.empty()) {
        const size_t io_bias = cpu_staging_io_bias(bm);
        {
            std::lock_guard<std::mutex> lk(s->sched_mtx);
            staging = cpu_staging_pool_take_locked(s, bm->byte_size, io_bias);
        }
        if (staging.empty() && !staging.allocate(bm->byte_size, io_bias)) return -6;
        const int rc = read_block_direct_or_mmap(s, bm, staging.data());
        if (rc != 0) return rc;
    }

    int prepare_rc = 0;
    const uint64_t prepare_t0 = cpu_now_us();
    if (slice.repacked) {
        ggml_tensor packed = *slice.tensor;
        if (slice.row_start != 0 || slice.row_count != slice.tensor->ne[1]) {
            packed.ne[1] = slice.row_count;
            packed.ne[2] = 1;
            packed.ne[3] = 1;
            packed.nb[2] = packed.nb[1] * static_cast<size_t>(packed.ne[1]);
            packed.nb[3] = packed.nb[2];
        }
        packed.data = target;
        packed.view_src = nullptr;
        packed.view_offs = 0;
        packed.extra = nullptr;
        if (!ggml_backend_cpu_repack_tensor_init(&packed) ||
            ggml_nbytes(&packed) != bm->byte_size) {
            prepare_rc = -7;
        } else {
            prepare_rc = ggml_backend_cpu_repack_tensor(
                &packed, staging.data(), bm->byte_size);
        }
        const uint64_t repack_us = cpu_now_us() - prepare_t0;
        s->stage_repack_calls++;
        s->stage_repack_us += repack_us;
        s->stage_repack_bytes += bm->byte_size;
        if (prepare_rc == 0) s->stage_repack_ok++;
    } else if (!staging.empty()) {
        cpu_copy_staging_to_final(s, target, staging.data(), bm->byte_size);
        s->stage_materialize_calls++;
        s->stage_materialize_us += cpu_now_us() - prepare_t0;
        s->stage_materialize_bytes += bm->byte_size;
    } else {
        int rc = read_block_direct_or_mmap(s, bm, target);
        if (rc != 0) return rc;
        s->stage_materialize_calls++;
        s->stage_materialize_us += cpu_now_us() - prepare_t0;
        s->stage_materialize_bytes += bm->byte_size;
    }
    if (!staging.empty()) {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        cpu_staging_pool_put_locked(s, std::move(staging));
    }
    if (prepare_rc != 0) return prepare_rc;
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
    s->stage_xform_us += (uint64_t) dt;
    s->stage_xform_ok++;
    s->stage_xform_bytes += bm->byte_size;
    s->n_reloads_total += 1;
    s->bytes_reloaded_total += bm->byte_size;
    elastic::wbm_mark_resident(&s->wbm, idx, target);
    llama_weight_runtime_mark_resident(name, LLAMA_WEIGHT_RUNTIME_CPU);
    if (s->profile_csv) {
        cpu_profile_record("PREPARE", name, idx, bm->byte_size, dt / 1000.0, 1, "plan_stage");
    }
    return 0;
}

static int cpu_repack_pair_with_aux_lane(
        elastic_state * s,
        ggml_tensor * first, const void * first_data, size_t first_size,
        ggml_tensor * second, const void * second_data, size_t second_size);

static int elastic_cpu_prepare_resident_pair(
        elastic_state * s, int first_idx, int second_idx) {
    if (!s || first_idx < 0 || second_idx < 0 || first_idx == second_idx) return -1;
    const elastic::block_meta * first_bm = elastic::wbm_get(&s->wbm, first_idx);
    const elastic::block_meta * second_bm = elastic::wbm_get(&s->wbm, second_idx);
    if (!first_bm || !second_bm || first_bm->resident || second_bm->resident) return -2;

    elastic_cpu_tensor_slice first_slice;
    elastic_cpu_tensor_slice second_slice;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        auto first_it = s->tensor_slice_by_idx.find(first_idx);
        auto second_it = s->tensor_slice_by_idx.find(second_idx);
        if (first_it == s->tensor_slice_by_idx.end() ||
            second_it == s->tensor_slice_by_idx.end()) {
            return -3;
        }
        first_slice = first_it->second;
        second_slice = second_it->second;
    }
    if (!first_slice.tensor) return -41;
    if (!second_slice.tensor) return -42;
    if (!first_slice.repacked) return -43;
    if (!second_slice.repacked) return -44;

    auto make_packed = [](const elastic_cpu_tensor_slice & slice, void * target) {
        ggml_tensor packed = *slice.tensor;
        if (slice.row_start != 0 || slice.row_count != slice.tensor->ne[1]) {
            packed.ne[1] = slice.row_count;
            packed.ne[2] = 1;
            packed.ne[3] = 1;
            packed.nb[2] = packed.nb[1] * static_cast<size_t>(packed.ne[1]);
            packed.nb[3] = packed.nb[2];
        }
        packed.data = target;
        packed.view_src = nullptr;
        packed.view_offs = 0;
        packed.extra = nullptr;
        ggml_backend_cpu_repack_tensor_init(&packed);
        return packed;
    };

    void * first_target = first_bm->backend_handle
        ? first_bm->backend_handle : cpu_fixed_handle_for_idx(s, first_idx);
    void * second_target = second_bm->backend_handle
        ? second_bm->backend_handle : cpu_fixed_handle_for_idx(s, second_idx);
    if (!first_target || !second_target) return -5;
    ggml_tensor first_packed = make_packed(first_slice, first_target);
    ggml_tensor second_packed = make_packed(second_slice, second_target);
    if (ggml_nbytes(&first_packed) != first_bm->byte_size) return -61;
    if (ggml_nbytes(&second_packed) != second_bm->byte_size) return -62;
    if (!ggml_backend_cpu_repack_tensor_pair_compatible(
            &first_packed, &second_packed)) {
        static std::atomic<int> incompatible_logs{0};
        if (incompatible_logs.fetch_add(1, std::memory_order_relaxed) < 8) {
            GGML_LOG_WARN(
                "elastic: layout pair incompatible first=%s type=%s ne=[%lld,%lld] extra=%p "
                "second=%s type=%s ne=[%lld,%lld] extra=%p\n",
                first_slice.tensor->name, ggml_type_name(first_packed.type),
                static_cast<long long>(first_packed.ne[0]),
                static_cast<long long>(first_packed.ne[1]), first_packed.extra,
                second_slice.tensor->name, ggml_type_name(second_packed.type),
                static_cast<long long>(second_packed.ne[0]),
                static_cast<long long>(second_packed.ne[1]), second_packed.extra);
        }
        return -63;
    }

    cpu_staging_buffer first_staging;
    cpu_staging_buffer second_staging;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        auto take_staged = [&](int idx, size_t expected, cpu_staging_buffer & staging) -> bool {
            auto it = s->staged_raw_by_idx.find(idx);
            if (it == s->staged_raw_by_idx.end()) return true;
            if (it->second.size() != expected) return false;
            staging = std::move(it->second);
            s->staged_raw_by_idx.erase(it);
            return true;
        };
        if (!take_staged(first_idx, first_bm->byte_size, first_staging) ||
            !take_staged(second_idx, second_bm->byte_size, second_staging)) {
            return -7;
        }
    }

    auto return_staging = [&]() {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        if (!first_staging.empty()) {
            cpu_staging_pool_put_locked(s, std::move(first_staging));
        }
        if (!second_staging.empty()) {
            cpu_staging_pool_put_locked(s, std::move(second_staging));
        }
    };
    auto fill_staging = [&](const elastic::block_meta * bm,
                            cpu_staging_buffer & staging) -> int {
        if (!staging.empty()) return 0;
        const size_t io_bias = cpu_staging_io_bias(bm);
        {
            std::lock_guard<std::mutex> lk(s->sched_mtx);
            staging = cpu_staging_pool_take_locked(s, bm->byte_size, io_bias);
        }
        if (staging.empty() && !staging.allocate(bm->byte_size, io_bias)) return -8;
        return read_block_direct_or_mmap(s, bm, staging.data());
    };
    int rc = fill_staging(first_bm, first_staging);
    if (rc == 0) rc = fill_staging(second_bm, second_staging);
    if (rc != 0) {
        return_staging();
        return rc;
    }

    s->fused_layout_pair_calls++;
    s->stage_xform_calls++;
    const uint64_t transform_t0 = cpu_now_us();
    rc = cpu_repack_pair_with_aux_lane(
        s,
        &first_packed, first_staging.data(), first_bm->byte_size,
        &second_packed, second_staging.data(), second_bm->byte_size);
    const uint64_t transform_us = cpu_now_us() - transform_t0;
    const size_t pair_bytes = first_bm->byte_size + second_bm->byte_size;
    s->fused_layout_pair_us += transform_us;
    s->fused_layout_pair_bytes += pair_bytes;
    s->stage_xform_us += transform_us;
    s->stage_xform_bytes += pair_bytes;
    s->stage_repack_calls++;
    s->stage_repack_us += transform_us;
    s->stage_repack_bytes += pair_bytes;
    return_staging();
    if (rc != 0) return rc;

    s->fused_layout_pair_ok++;
    s->stage_xform_ok++;
    s->stage_repack_ok++;
    s->n_reloads_total += 2;
    s->bytes_reloaded_total += pair_bytes;
    elastic::wbm_mark_resident(&s->wbm, first_idx, first_target);
    elastic::wbm_mark_resident(&s->wbm, second_idx, second_target);
    const std::string first_name = cpu_name_for_idx_copy(s, first_idx);
    const std::string second_name = cpu_name_for_idx_copy(s, second_idx);
    llama_weight_runtime_mark_resident(first_name.c_str(), LLAMA_WEIGHT_RUNTIME_CPU);
    llama_weight_runtime_mark_resident(second_name.c_str(), LLAMA_WEIGHT_RUNTIME_CPU);
    if (s->profile_csv) {
        cpu_profile_record("PREPARE_FUSED", first_name.c_str(), first_idx,
                           first_bm->byte_size, transform_us / 2000.0, 1, "pair_repack");
        cpu_profile_record("PREPARE_FUSED", second_name.c_str(), second_idx,
                           second_bm->byte_size, transform_us / 2000.0, 1, "pair_repack");
    }
    return 0;
}

static void cpu_stop_async_workers(elastic_state *s);

static void cpu_register_async_shutdown() {
    static std::once_flag once;
    std::call_once(once, []() {
        std::atexit([]() { cpu_stop_async_workers(get_state()); });
    });
}

static void cpu_pin_pipeline_worker(const char * env_name, const char * worker_name) {
#if defined(__linux__) || defined(__ANDROID__)
    const char * value = std::getenv(env_name);
    if (!value || !*value) return;
    const int cpu = std::atoi(value);
    if (cpu < 0 || cpu >= CPU_SETSIZE) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    const int rc = sched_setaffinity(0, sizeof(set), &set);
    GGML_LOG_INFO("elastic: %s worker affinity cpu=%d rc=%d\n", worker_name, cpu, rc);
#else
    GGML_UNUSED(env_name);
    GGML_UNUSED(worker_name);
#endif
}

static void cpu_start_copy_worker(elastic_state *s) {
    if (!s) return;
    {
        std::lock_guard<std::mutex> lk(s->copy_mtx);
        if (s->copy_worker_started) return;
        s->copy_worker_started = true;
    }
    cpu_register_async_shutdown();
    s->copy_worker = std::thread([s]() {
        cpu_pin_pipeline_worker("GGML_ELASTIC_PIPELINE_COPY_CPU", "COPY");
        for (;;) {
            cpu_aux_task_kind kind = cpu_aux_task_kind::COPY;
            void * dst = nullptr;
            const void * src = nullptr;
            size_t bytes = 0;
            ggml_tensor * repack_tensor = nullptr;
            {
                std::unique_lock<std::mutex> lk(s->copy_mtx);
                s->copy_cv.wait(lk, [s]() {
                    return s->copy_worker_shutdown || s->copy_task_pending;
                });
                if (s->copy_worker_shutdown && !s->copy_task_pending) break;
                kind = s->copy_task_kind;
                dst = s->copy_dst;
                src = s->copy_src;
                bytes = s->copy_bytes;
                repack_tensor = s->copy_repack_tensor;
                s->copy_task_pending = false;
            }
            int rc = 0;
            if (kind == cpu_aux_task_kind::REPACK) {
                rc = ggml_backend_cpu_repack_tensor(
                    repack_tensor, src, bytes);
            } else {
                std::memcpy(dst, src, bytes);
            }
            {
                std::lock_guard<std::mutex> lk(s->copy_mtx);
                s->copy_task_rc = rc;
                s->copy_task_done = true;
            }
            s->copy_cv.notify_all();
        }
    });
}

static bool cpu_parallel_prepare_lane_enabled() {
    static const bool enabled = []() {
        const char * value =
            std::getenv("GGML_ELASTIC_CPU_STAGE_COPY_PARALLEL");
        return value && *value && *value != '0';
    }();
    return enabled;
}

static size_t cpu_parallel_prepare_lane_min_bytes() {
    static const size_t value = []() {
        const char * env =
            std::getenv("GGML_ELASTIC_CPU_STAGE_COPY_MIN_KB");
        const long long kib = env && *env ? std::atoll(env) : 1024;
        return static_cast<size_t>(
            std::max<long long>(0, kib)) * 1024ULL;
    }();
    return value;
}

static void cpu_copy_staging_to_final(elastic_state *s, void * dst,
                                      const void * src, size_t nbytes) {
    if (!cpu_parallel_prepare_lane_enabled() || !s ||
        nbytes < cpu_parallel_prepare_lane_min_bytes()) {
        std::memcpy(dst, src, nbytes);
        return;
    }

    cpu_start_copy_worker(s);
    // Cache-line-aligned split; both halves are independent and the source
    // staging buffer remains live until the helper completion is observed.
    const size_t first_bytes = (nbytes / 2) & ~static_cast<size_t>(63);
    if (first_bytes == 0 || first_bytes == nbytes) {
        std::memcpy(dst, src, nbytes);
        return;
    }
    {
        std::unique_lock<std::mutex> lk(s->copy_mtx);
        s->copy_cv.wait(lk, [s]() { return !s->copy_task_pending && s->copy_task_done; });
        s->copy_task_done = false;
        s->copy_task_kind = cpu_aux_task_kind::COPY;
        s->copy_dst = dst;
        s->copy_src = src;
        s->copy_bytes = first_bytes;
        s->copy_repack_tensor = nullptr;
        s->copy_task_rc = 0;
        s->copy_task_pending = true;
    }
    s->copy_cv.notify_one();
    std::memcpy(static_cast<char *>(dst) + first_bytes,
                static_cast<const char *>(src) + first_bytes,
                nbytes - first_bytes);
    {
        std::unique_lock<std::mutex> lk(s->copy_mtx);
        s->copy_cv.wait(lk, [s]() { return s->copy_task_done; });
    }
    s->copy_parallel_calls++;
    s->copy_parallel_bytes += nbytes;
}

static int cpu_repack_pair_with_aux_lane(
        elastic_state * s,
        ggml_tensor * first, const void * first_data, size_t first_size,
        ggml_tensor * second, const void * second_data, size_t second_size) {
    if (!s || !cpu_parallel_prepare_lane_enabled() ||
        first_size + second_size <
            cpu_parallel_prepare_lane_min_bytes()) {
        return ggml_backend_cpu_repack_tensor_pair(
            first, first_data, first_size,
            second, second_data, second_size);
    }

    cpu_start_copy_worker(s);
    {
        std::unique_lock<std::mutex> lk(s->copy_mtx);
        s->copy_cv.wait(lk, [s]() {
            return !s->copy_task_pending && s->copy_task_done;
        });
        s->copy_task_done = false;
        s->copy_task_kind = cpu_aux_task_kind::REPACK;
        s->copy_dst = nullptr;
        s->copy_src = second_data;
        s->copy_bytes = second_size;
        s->copy_repack_tensor = second;
        s->copy_task_rc = 0;
        s->copy_task_pending = true;
    }
    s->copy_cv.notify_one();
    const int first_rc = ggml_backend_cpu_repack_tensor(
        first, first_data, first_size);
    int second_rc = 0;
    {
        std::unique_lock<std::mutex> lk(s->copy_mtx);
        s->copy_cv.wait(lk, [s]() { return s->copy_task_done; });
        second_rc = s->copy_task_rc;
    }
    s->fused_layout_parallel_calls++;
    return first_rc != 0 ? first_rc : second_rc;
}

static bool cpu_parallel_multi_io_enabled() {
    static const bool enabled = []() {
        const char * value =
            std::getenv("GGML_ELASTIC_CPU_PARALLEL_MULTI_IO");
        // Two concurrent O_DIRECT streams on current Android UFS devices
        // reduce aggregate bandwidth and turn a coarse unit into a
        // head-of-line barrier. The work-conserving default streams members
        // sequentially through the existing pooled LOAD worker, publishing
        // each completion immediately to PREPARE. Keep the experimental
        // second reader as an explicit opt-in.
        return value && *value && *value != '0';
    }();
    return enabled;
}

static void cpu_start_multi_io_worker(elastic_state * s) {
    if (!s) return;
    {
        std::lock_guard<std::mutex> lk(s->multi_io_mtx);
        if (s->multi_io_worker_started) return;
        s->multi_io_worker_started = true;
    }
    cpu_register_async_shutdown();
    s->multi_io_worker = std::thread([s]() {
        cpu_pin_pipeline_worker(
            "GGML_ELASTIC_PIPELINE_LOAD_AUX_CPU", "LOAD_AUX");
        for (;;) {
            int idx = -1;
            {
                std::unique_lock<std::mutex> lk(s->multi_io_mtx);
                s->multi_io_cv.wait(lk, [s]() {
                    return s->multi_io_worker_shutdown ||
                           s->multi_io_task_pending;
                });
                if (s->multi_io_worker_shutdown &&
                    !s->multi_io_task_pending) {
                    break;
                }
                idx = s->multi_io_task_idx;
                s->multi_io_task_pending = false;
            }
            const std::string name = cpu_name_for_idx_copy(s, idx);
            const int rc =
                elastic_cpu_stage_load_sync(s, name.c_str(), idx);
            {
                std::lock_guard<std::mutex> lk(s->sched_mtx);
                s->async_load_state[idx] = rc == 0 ? 2 : rc;
                s->async_load_completed++;
            }
            s->async_load_cv.notify_all();
            {
                std::lock_guard<std::mutex> lk(s->multi_io_mtx);
                s->multi_io_task_rc = rc;
                s->multi_io_task_done = true;
            }
            s->multi_io_cv.notify_all();
        }
    });
}

static void cpu_start_async_load_worker(elastic_state *s) {
    if (!s) return;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        if (s->async_load_worker_started) return;
        s->async_load_worker_started = true;
    }
    cpu_register_async_shutdown();
    s->async_load_worker = std::thread([s]() {
        cpu_pin_pipeline_worker("GGML_ELASTIC_PIPELINE_LOAD_CPU", "LOAD");
        for (;;) {
            cpu_stage_unit_task task;
            {
                std::unique_lock<std::mutex> lk(s->sched_mtx);
                s->async_load_cv.wait(lk, [s]() {
                    return s->async_load_shutdown || !s->async_load_queue.empty();
                });
                if (s->async_load_shutdown && s->async_load_queue.empty()) break;
                task = std::move(s->async_load_queue.front());
                s->async_load_queue.pop_front();
            }
            // When Android policy blocks io_uring, do not turn a Multi unit
            // into a serial head-of-line barrier. Publish each completed LOAD
            // immediately so PREPARE can consume tensor i while this worker
            // reads tensor i+1. Compute still waits for the complete Multi
            // unit, so its working-unit semantics remain atomic.
            if (task.indices.size() > 1 && !llama_pread_direct_batch_available()) {
                s->direct_batch_units++;
                s->direct_batch_tensors += task.indices.size();
                s->direct_batch_fallbacks++;
                size_t first_serial = 0;
                if (cpu_parallel_multi_io_enabled() &&
                    task.indices.size() >= 2) {
                    cpu_start_multi_io_worker(s);
                    {
                        std::unique_lock<std::mutex> lk(s->multi_io_mtx);
                        s->multi_io_cv.wait(lk, [s]() {
                            return !s->multi_io_task_pending &&
                                   s->multi_io_task_done;
                        });
                        s->multi_io_task_done = false;
                        s->multi_io_task_idx = task.indices[1];
                        s->multi_io_task_rc = 0;
                        s->multi_io_task_pending = true;
                    }
                    s->multi_io_cv.notify_one();

                    const int idx = task.indices[0];
                    const std::string name = cpu_name_for_idx_copy(s, idx);
                    const int rc = elastic_cpu_stage_load_sync(s, name.c_str(), idx);
                    {
                        std::lock_guard<std::mutex> lk(s->sched_mtx);
                        s->async_load_state[idx] = rc == 0 ? 2 : rc;
                        s->async_load_completed++;
                    }
                    s->async_load_cv.notify_all();
                    {
                        std::unique_lock<std::mutex> lk(s->multi_io_mtx);
                        s->multi_io_cv.wait(
                            lk, [s]() { return s->multi_io_task_done; });
                    }
                    s->multi_io_parallel_units++;
                    s->multi_io_parallel_tensors += 2;
                    first_serial = 2;
                }
                for (size_t member = first_serial;
                     member < task.indices.size(); ++member) {
                    const int idx = task.indices[member];
                    const std::string name =
                        cpu_name_for_idx_copy(s, idx);
                    const int rc =
                        elastic_cpu_stage_load_sync(
                            s, name.c_str(), idx);
                    {
                        std::lock_guard<std::mutex> lk(s->sched_mtx);
                        s->async_load_state[idx] = rc == 0 ? 2 : rc;
                        s->async_load_completed++;
                    }
                    s->async_load_cv.notify_all();
                }
            } else {
                const int rc = elastic_cpu_stage_load_unit_sync(s, task.indices);
                {
                    std::lock_guard<std::mutex> lk(s->sched_mtx);
                    for (int idx : task.indices) {
                        s->async_load_state[idx] = rc == 0 ? 2 : rc;
                        s->async_load_completed++;
                    }
                }
                s->async_load_cv.notify_all();
            }
            {
                std::lock_guard<std::mutex> lk(s->sched_mtx);
                s->async_load_units_completed++;
            }
        }
    });
}

static int cpu_enqueue_async_load_unit(
        elastic_state * s, const std::vector<int> & indices) {
    if (!s) return -1;
    cpu_stage_unit_task task;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        task.id = s->async_stage_next_unit_id++;
        task.indices.reserve(indices.size());
        for (int idx : indices) {
            if (idx < 0) continue;
            const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
            if (!bm || bm->resident) continue;
            if (s->staged_raw_by_idx.find(idx) != s->staged_raw_by_idx.end()) continue;
            auto preparing = s->async_prepare_state.find(idx);
            if (preparing != s->async_prepare_state.end() && preparing->second == 1) {
                // PREPARE may already have moved the staging buffer out of the
                // active map and be copying it. Do not enqueue a duplicate.
                continue;
            }
            auto loading = s->async_load_state.find(idx);
            if (loading != s->async_load_state.end() && loading->second == 1) continue;
            s->async_load_state[idx] = 1;
            task.indices.push_back(idx);
        }
        if (task.indices.empty()) return 0;
        s->async_load_enqueued += task.indices.size();
        s->async_load_units_enqueued++;
        s->async_load_queue.push_back(std::move(task));
    }
    cpu_start_async_load_worker(s);
    s->async_load_cv.notify_one();
    return 0;
}

static int cpu_enqueue_async_load(elastic_state * s, int idx) {
    return cpu_enqueue_async_load_unit(s, std::vector<int>{idx});
}

static int cpu_wait_async_load(elastic_state *s, int idx) {
    if (!s || idx < 0) return -1;
    const uint64_t t0 = cpu_now_us();
    int state = 0;
    {
        std::unique_lock<std::mutex> lk(s->sched_mtx);
        auto it = s->async_load_state.find(idx);
        if (it == s->async_load_state.end()) return 0;
        if (it->second == 1) {
            s->async_load_waits++;
            s->async_load_cv.wait(lk, [s, idx]() {
                auto cur = s->async_load_state.find(idx);
                return cur == s->async_load_state.end() || cur->second != 1;
            });
        }
        auto done = s->async_load_state.find(idx);
        state = done == s->async_load_state.end() ? 0 : done->second;
    }
    s->async_load_wait_us += cpu_now_us() - t0;
    return state >= 0 ? 0 : state;
}

static void cpu_start_async_prepare_worker(elastic_state *s) {
    if (!s) return;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        if (s->async_prepare_worker_started) return;
        s->async_prepare_worker_started = true;
    }
    cpu_register_async_shutdown();
    s->async_prepare_worker = std::thread([s]() {
        cpu_pin_pipeline_worker("GGML_ELASTIC_PIPELINE_PREPARE_CPU", "PREPARE");
        for (;;) {
            cpu_stage_unit_task task;
            {
                std::unique_lock<std::mutex> lk(s->sched_mtx);
                s->async_prepare_cv.wait(lk, [s]() {
                    return s->async_prepare_shutdown || !s->async_prepare_queue.empty();
                });
                if (s->async_prepare_shutdown && s->async_prepare_queue.empty()) break;
                task = std::move(s->async_prepare_queue.front());
                s->async_prepare_queue.pop_front();
            }
            int task_rc = 0;
            std::vector<int> load_rc(task.indices.size(), 0);
            std::vector<bool> load_waited(task.indices.size(), false);
            auto wait_load = [&](size_t index) {
                if (!load_waited[index]) {
                    load_rc[index] = cpu_wait_async_load(
                        s, task.indices[index]);
                    load_waited[index] = true;
                }
                return load_rc[index];
            };
            std::vector<bool> pair_prepared(
                task.indices.size(), false);
            for (const auto & pair : task.fused_pairs) {
                const auto first = std::find(
                    task.indices.begin(), task.indices.end(), pair.first);
                const auto second = std::find(
                    task.indices.begin(), task.indices.end(), pair.second);
                if (first == task.indices.end() ||
                    second == task.indices.end()) {
                    continue;
                }
                const size_t first_pos = static_cast<size_t>(
                    first - task.indices.begin());
                const size_t second_pos = static_cast<size_t>(
                    second - task.indices.begin());
                if (pair_prepared[first_pos] ||
                    pair_prepared[second_pos] ||
                    wait_load(first_pos) != 0 ||
                    wait_load(second_pos) != 0) {
                    continue;
                }
                const int pair_rc = elastic_cpu_prepare_resident_pair(
                    s, pair.first, pair.second);
                if (pair_rc == 0) {
                    pair_prepared[first_pos] = true;
                    pair_prepared[second_pos] = true;
                    for (int idx : {pair.first, pair.second}) {
                        std::lock_guard<std::mutex> lk(s->sched_mtx);
                        s->async_prepare_state[idx] = 2;
                        s->async_prepare_completed++;
                    }
                    s->async_prepare_cv.notify_all();
                } else {
                    s->fused_layout_pair_fallbacks++;
                    static std::atomic<int> pair_fallback_logs{0};
                    const int log_index =
                        pair_fallback_logs.fetch_add(1, std::memory_order_relaxed);
                    if (log_index < 8) {
                        const std::string first_name =
                            cpu_name_for_idx_copy(s, pair.first);
                        const std::string second_name =
                            cpu_name_for_idx_copy(s, pair.second);
                        GGML_LOG_WARN(
                            "elastic: multi_fused layout pair fallback rc=%d first=%d:%s second=%d:%s\n",
                            pair_rc, pair.first, first_name.c_str(),
                            pair.second, second_name.c_str());
                    }
                }
            }
            for (size_t i = 0; i < task.indices.size(); ++i) {
                if (pair_prepared[i]) continue;
                const int idx = task.indices[i];
                // Do not wait for every member of a coarse unit before
                // starting PREPARE. With the streaming fallback, LOAD(i+1)
                // now overlaps layout preparation of i while COMPUTE still
                // observes the atomic unit boundary.
                int rc = wait_load(i);
                if (rc == 0) {
                    const std::string name = cpu_name_for_idx_copy(s, idx);
                    void * fixed_handle = cpu_fixed_handle_for_idx(s, idx);
                    rc = elastic_cpu_prepare_resident(s, name.c_str(), idx, fixed_handle);
                }
                if (rc != 0 && task_rc == 0) task_rc = rc;
                {
                    std::lock_guard<std::mutex> lk(s->sched_mtx);
                    s->async_prepare_state[idx] = rc == 0 ? 2 : rc;
                    s->async_prepare_completed++;
                }
                s->async_prepare_cv.notify_all();
            }
            {
                std::lock_guard<std::mutex> lk(s->sched_mtx);
                GGML_UNUSED(task_rc);
                s->async_prepare_units_completed++;
            }
            s->async_prepare_cv.notify_all();
        }
    });
}

static int cpu_enqueue_async_prepare_unit(
        elastic_state * s, const std::vector<int> & indices,
        bool fused_pair = false,
        const std::vector<std::pair<int, int>> * fused_pairs = nullptr) {
    if (!s) return -1;
    cpu_stage_unit_task task;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        task.id = s->async_stage_next_unit_id++;
        task.indices.reserve(indices.size());
        size_t pending = 0;
        const size_t max_pending = cpu_async_prepare_max_pending();
        if (max_pending > 0) {
            for (const auto & kv : s->async_prepare_state) {
                if (kv.second == 1) pending++;
            }
        }
        for (int idx : indices) {
            if (idx < 0) continue;
            const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
            if (!bm || bm->resident) continue;
            auto preparing = s->async_prepare_state.find(idx);
            if (preparing != s->async_prepare_state.end() && preparing->second == 1) continue;
            task.indices.push_back(idx);
        }
        // Preserve unit atomicity: never enqueue only the first half of a
        // Multi unit because a block-count cap was reached.
        if (max_pending > 0 && pending > 0 &&
            pending + task.indices.size() > max_pending) {
            s->async_prepare_cap_declines++;
            s->async_prepare_cap_declined_tensors += task.indices.size();
            return 0;
        }
        if (task.indices.empty()) return 0;
        // Pair preparation is valid only when both explicitly selected
        // compatible tiles survived residency/in-flight filtering.
        std::unordered_set<int> selected(
            task.indices.begin(), task.indices.end());
        if (fused_pairs) {
            for (const auto & pair : *fused_pairs) {
                if (selected.count(pair.first) > 0 &&
                    selected.count(pair.second) > 0) {
                    task.fused_pairs.push_back(pair);
                }
            }
        } else if (fused_pair && task.indices.size() == 2) {
            task.fused_pairs.emplace_back(
                task.indices[0], task.indices[1]);
        }
        for (int idx : task.indices) s->async_prepare_state[idx] = 1;
        s->async_prepare_enqueued += task.indices.size();
        s->async_prepare_units_enqueued++;
        s->async_prepare_queue.push_back(std::move(task));
    }
    cpu_start_async_prepare_worker(s);
    s->async_prepare_cv.notify_one();
    return 0;
}

static int cpu_enqueue_async_prepare(elastic_state * s, int idx) {
    return cpu_enqueue_async_prepare_unit(s, std::vector<int>{idx});
}

static int cpu_wait_async_prepare(elastic_state *s, int idx) {
    if (!s || idx < 0) return -1;
    const uint64_t t0 = cpu_now_us();
    int state = 0;
    {
        std::unique_lock<std::mutex> lk(s->sched_mtx);
        auto it = s->async_prepare_state.find(idx);
        if (it == s->async_prepare_state.end()) return 0;
        if (it->second == 1) {
            s->async_prepare_waits++;
            s->async_prepare_cv.wait(lk, [s, idx]() {
                auto cur = s->async_prepare_state.find(idx);
                return cur == s->async_prepare_state.end() || cur->second != 1;
            });
        }
        auto done = s->async_prepare_state.find(idx);
        state = done == s->async_prepare_state.end() ? 0 : done->second;
    }
    s->async_prepare_wait_us += cpu_now_us() - t0;
    return state >= 0 ? 0 : state;
}

static void cpu_stop_async_workers(elastic_state *s) {
    if (!s) return;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        s->async_load_shutdown = true;
        s->async_prepare_shutdown = true;
    }
    s->async_load_cv.notify_all();
    s->async_prepare_cv.notify_all();
    if (s->async_load_worker.joinable()) s->async_load_worker.join();
    if (s->async_prepare_worker.joinable()) s->async_prepare_worker.join();
    {
        std::lock_guard<std::mutex> lk(s->multi_io_mtx);
        s->multi_io_worker_shutdown = true;
    }
    s->multi_io_cv.notify_all();
    if (s->multi_io_worker.joinable()) s->multi_io_worker.join();
    {
        std::lock_guard<std::mutex> lk(s->copy_mtx);
        s->copy_worker_shutdown = true;
    }
    s->copy_cv.notify_all();
    if (s->copy_worker.joinable()) s->copy_worker.join();
}

static void cpu_wait_async_workers_idle(elastic_state *s) {
    if (!s) return;
    {
        std::unique_lock<std::mutex> lk(s->sched_mtx);
        s->async_load_cv.wait(lk, [s]() {
            if (!s->async_load_queue.empty()) return false;
            return std::none_of(
                s->async_load_state.begin(),
                s->async_load_state.end(),
                [](const auto &entry) { return entry.second == 1; });
        });
    }
    {
        std::unique_lock<std::mutex> lk(s->sched_mtx);
        s->async_prepare_cv.wait(lk, [s]() {
            if (!s->async_prepare_queue.empty()) return false;
            return std::none_of(
                s->async_prepare_state.begin(),
                s->async_prepare_state.end(),
                [](const auto &entry) { return entry.second == 1; });
        });
    }
}

int elastic_sched_stage_request(const char *name, const char *stage, void * /*ud*/) {
    if (!name || !stage) return -1;
    auto *s = get_state();
    std::vector<int> indices;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        indices = lookup_wbm_indices_locked(s, name);
    }
    if (indices.empty()) return -2;
    if (std::strcmp(stage, "load_gpu") == 0) {
        return -2;
    }
    if (std::strcmp(stage, "load") == 0 || std::strcmp(stage, "load_cpu") == 0) {
        for (int idx : indices) {
            const int rc = cpu_async_stage_load_enabled()
                ? cpu_enqueue_async_load(s, idx)
                : elastic_cpu_stage_load_sync(s, name, idx);
            if (rc != 0) return rc;
        }
        return 0;
    }
    if (std::strcmp(stage, "transfer") == 0 || std::strcmp(stage, "dma") == 0) {
        return 0;
    }
    if (std::strcmp(stage, "prepare_gpu") == 0) {
        return -2;
    }
    if (std::strcmp(stage, "prepare") == 0 || std::strcmp(stage, "prepare_cpu") == 0 ||
        std::strcmp(stage, "materialize") == 0) {
        for (int idx : indices) {
            const int rc = cpu_async_stage_prepare_enabled()
                ? cpu_enqueue_async_prepare(s, idx)
                : elastic_cpu_prepare_resident(s, name, idx, cpu_fixed_handle_for_idx(s, idx));
            if (rc != 0) return rc;
        }
        return 0;
    }
    return -3;
}

int elastic_sched_transform_request(const char *name, llama_weight_transform_kind kind, void * /*ud*/) {
    if (!name) return -1;
    if (kind != LLAMA_WEIGHT_TRANSFORM_CPU_REPACK) return -2;
    auto *s = get_state();
    std::vector<int> indices;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        indices = lookup_wbm_indices_locked(s, name);
    }
    if (indices.empty()) return -2;
    for (int idx : indices) {
        const int rc = elastic_cpu_prepare_resident(
            s, name, idx, cpu_fixed_handle_for_idx(s, idx));
        if (rc != 0) return rc;
    }
    return 0;
}

void * elastic_sched_host_ptr_query(const char *name, void * /*ud*/) {
    if (!name) return nullptr;
    auto *s = get_state();
    int idx = -1;
    {
        std::lock_guard<std::mutex> lk(s->sched_mtx);
        idx = lookup_wbm_idx_locked(s, name);
        if (idx < 0) return nullptr;
    }
    const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, idx);
    return bm ? bm->host_ptr : nullptr;
}

// Budget provider: BudgetWatcher 当前预算 (MB) → llama_context scheduler 同源信号。
// bw 没启用返 -1 → scheduler fallback /proc/meminfo。
int64_t elastic_sched_budget_query(void * /*ud*/) {
    auto *s = get_state();
    if (!s->bw_inited) return -1;
    return (int64_t) elastic::budget_watcher_get(&s->bw);
}

void elastic_sched_budget_reset(void * /*ud*/) {
    auto *s = get_state();
    if (!s->bw_inited) return;
    elastic::budget_watcher_reset_clock(&s->bw);
    GGML_LOG_INFO("elastic: BudgetWatcher replay clock reset, B(t)=%zu MB\n",
                  elastic::budget_watcher_get(&s->bw));
}

void elastic_sched_register_once() {
    auto *s = get_state();
    if (s->sched_registered) return;
    s->sched_registered = true;
    llama_weight_residency_register(elastic_sched_residency_query, nullptr);
    llama_weight_state_register    (elastic_sched_state_query,     nullptr);
    llama_weight_movement_register (elastic_sched_movement_request, nullptr);
    llama_weight_stage_register    (elastic_sched_stage_request,    nullptr);
    llama_weight_transform_register(elastic_sched_transform_request, nullptr);
    llama_weight_host_ptr_register (elastic_sched_host_ptr_query, nullptr);
    llama_budget_register          (elastic_sched_budget_query,    nullptr);
    llama_budget_reset_register    (elastic_sched_budget_reset,    nullptr);
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

const char * name_for_idx(elastic_state *s, int idx) {
    if (!s) return "";
    auto it = s->wbm_to_name.find(idx);
    if (it == s->wbm_to_name.end()) return "";
    return it->second.c_str();
}

void cpu_profile_record(const char *kind, const char *name, int idx,
                        size_t bytes, double ms, int ok,
                        const char *extra = "") {
    if (!elastic::profile_enabled()) return;
    elastic::profile_record rec;
    rec.backend   = "CPU_Elastic";
    rec.kind      = kind;
    rec.name      = name ? name : "";
    rec.weight_id = idx;
    rec.bytes     = bytes;
    rec.ms        = ms;
    rec.ok        = ok;
    rec.extra     = extra ? extra : "";
    elastic::profile_write(rec);
}

void cpu_profile_compute_graph(const ggml_cgraph *cgraph, double ms, int ok) {
    if (!elastic::profile_enabled()) return;
    elastic::profile_record rec;
    rec.backend = "CPU_Elastic";
    rec.kind    = "COMPUTE_GRAPH";
    rec.name    = "ggml_cgraph";
    rec.op      = "GRAPH";
    rec.op_id   = cgraph ? cgraph->n_nodes : -1;
    rec.ms      = ms;
    rec.ok      = ok;
    rec.extra   = "delegate_cpu_graph";
    elastic::profile_write(rec);
}

void cpu_profile_compute_node(const ggml_tensor *node, int op_id, double ms, int ok) {
    if (!elastic::profile_enabled() || !node) return;
    elastic::profile_record rec;
    rec.backend = "CPU_Elastic";
    rec.kind    = "COMPUTE";
    rec.name    = node->name;
    rec.op      = ggml_op_name(node->op);
    rec.op_id   = op_id;
    rec.ne[0]   = node->ne[0];
    rec.ne[1]   = node->ne[1];
    rec.ne[2]   = node->ne[2];
    rec.ne[3]   = node->ne[3];
    rec.bytes   = ggml_nbytes(node);
    rec.ms      = ms;
    rec.ok      = ok;
    rec.extra   = "chunk_size_1";
    elastic::profile_write(rec);
}

// ============================================================
// Buffer context: 一整块 region + tensor → wbm_idx 映射
// ============================================================
struct elastic_buffer_ctx {
    void   *base;
    size_t  size;
    std::unordered_map<const ggml_tensor *, int> tensor_to_wbm;  // 反查
    std::unordered_map<const ggml_tensor *, elastic_tensor_units> tensor_to_units;
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

    // All reload paths must materialize the same compute layout as the initial
    // tensor load.  In particular, a repacked tensor cannot be restored by
    // copying raw GGUF bytes directly into its interleaved destination.
    const char * name = name_for_idx(s, wbm_idx);
    const int rc = elastic_cpu_prepare_resident(s, name, wbm_idx, backend_handle);
    if (rc != 0) {
        GGML_LOG_ERROR("elastic: CPU prepare failed for %s idx=%d rc=%d\n",
                       name, wbm_idx, rc);
    }
}

static void ensure_block_resident_pipeline(elastic_state *s, int wbm_idx, void *backend_handle) {
    if (!s || wbm_idx < 0) return;
    const elastic::block_meta *bm = elastic::wbm_get(&s->wbm, wbm_idx);
    if (!bm || bm->resident) return;

    if (cpu_sync_stage_workers_enabled()) {
        // Execute through the exact same LOAD/PREPARE workers and affinities
        // as pipeline-on, but wait immediately: no future-unit or
        // compute-stage overlap.
        cpu_enqueue_async_load(s, wbm_idx);
        cpu_enqueue_async_prepare(s, wbm_idx);
        cpu_wait_async_prepare(s, wbm_idx);
        bm = elastic::wbm_get(&s->wbm, wbm_idx);
        if (!bm || bm->resident) return;
    }

    cpu_wait_async_prepare(s, wbm_idx);
    bm = elastic::wbm_get(&s->wbm, wbm_idx);
    if (!bm || bm->resident) return;

    cpu_wait_async_load(s, wbm_idx);
    bm = elastic::wbm_get(&s->wbm, wbm_idx);
    if (!bm || bm->resident) return;

    const char *name = name_for_idx(s, wbm_idx);
    if (cpu_sync_stage_load_enabled()) {
        // Same LOAD -> PREPARE implementation used by the asynchronous
        // pipeline, executed inline.  This is the controlled no-overlap
        // baseline and also the pooled fallback for an unstaged current unit.
        const int load_rc = elastic_cpu_stage_load_sync(s, name, wbm_idx);
        if (load_rc == 0 &&
            elastic_cpu_prepare_resident(s, name, wbm_idx, backend_handle) == 0) {
            return;
        }
    }
    int rc = elastic_cpu_prepare_resident(s, name, wbm_idx, backend_handle);
    if (rc == 0) return;
    ensure_block_resident(s, wbm_idx, backend_handle);
}

// evict 列表：madvise DONTNEED + mark_evicted
void evict_blocks(elastic_state *s, elastic_buffer_ctx *bctx,
                  const std::vector<int> &victims) {
    const uint64_t evict_t0 = cpu_now_us();
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
        llama_weight_runtime_mark_evicted(name_for_idx(s, v), LLAMA_WEIGHT_RUNTIME_CPU);
        // mark_evicted 把 backend_handle 清掉了，但我们需要它保留（地址永不变）
        // 重新设上去：直接修改 blocks[]（注意是 hack——绕过 mark_evicted 的清零）
        // 更干净的做法是不调 mark_evicted，自己更新字段，避免清 backend_handle
    }
    s->n_evicts_total += victims.size();
    s->evict_wall_us += cpu_now_us() - evict_t0;

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

enum ggml_status elastic_buffer_init_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    // Attach the same native CPU tensor trait used by CPU_REPACK.  Unsupported
    // shapes/types keep a null trait and use the ordinary CPU byte layout.
    ggml_backend_cpu_repack_tensor_init(tensor);
    return GGML_STATUS_SUCCESS;
}

void elastic_buffer_set_tensor(ggml_backend_buffer_t buffer,
                               ggml_tensor *tensor,
                               const void *data, size_t offset, size_t size) {
    auto *bctx = (elastic_buffer_ctx *)buffer->context;
    auto *s = get_state();
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
    const bool full_overwrite = tensor->view_src == nullptr && offset == 0 &&
        size == ggml_nbytes(tensor);
    if (full_overwrite && elastic::profile_enabled()) {
        elastic::profile_record rec;
        rec.backend = "CPU_Elastic";
        rec.kind = "TENSOR_META";
        rec.name = tensor->name;
        rec.quant = ggml_type_name(tensor->type);
        rec.ne[0] = tensor->ne[0];
        rec.ne[1] = tensor->ne[1];
        rec.ne[2] = tensor->ne[2];
        rec.ne[3] = tensor->ne[3];
        rec.bytes = ggml_nbytes(tensor);
        rec.ok = 1;
        const std::string dims =
            "n_dims=" + std::to_string(ggml_n_dims(tensor));
        rec.extra = dims.c_str();
        elastic::profile_write(rec);
    }
    const bool native_repack = full_overwrite &&
        ggml_backend_cpu_repack_tensor_compatible(tensor);
    if (native_repack) {
        const uint64_t repack_t0 = cpu_now_us();
        const int rc = ggml_backend_cpu_repack_tensor(tensor, data, size);
        GGML_ASSERT(rc == 0);
        s->initial_repack_calls++;
        s->initial_repack_us += cpu_now_us() - repack_t0;
        s->initial_repack_bytes += size;
    } else {
        std::memcpy((char *)tensor->data + offset, data, size);
    }

    // 仅在"完整覆盖写"时注册 WBM（views / partial writes 跳过）
    if (!s->wbm_inited) {
        if (elastic::wbm_init(&s->wbm, 0) != 0) {
            GGML_LOG_ERROR("elastic: wbm_init 失败\n");
            return;
        }
        s->wbm_inited = true;
        s->granularity = elastic::granularity_from_env();
        s->profile_csv = elastic::profile_enabled();
        if (s->profile_csv) {
            GGML_LOG_INFO("elastic: GGML_ELASTIC_PROFILE_CSV enabled\n");
        }
        if (s->granularity.explicitly_enabled) {
            GGML_LOG_INFO("elastic: working-unit granularity=%s multi_tensors=%d cut_parts=%d\n",
                          elastic::granularity_mode_name(s->granularity.mode),
                          s->granularity.multi_tensors, s->granularity.cut_parts);
            static const bool unit_pipeline_enabled = []() {
                const char * e = std::getenv("GGML_ELASTIC_UNIT_PIPELINE");
                return e && *e && *e != '0';
            }();
            if (unit_pipeline_enabled) {
                GGML_LOG_INFO("elastic: granularity-aware unit pipeline enabled\n");
                std::atexit([]() {
                    auto * st = get_state();
                    std::fprintf(stderr,
                        "[elastic unit pipeline] backend=cpu issued=%llu ready=%llu waits=%llu wait_ms=%.3f\n",
                        (unsigned long long) st->unit_pipeline_issued,
                        (unsigned long long) st->unit_pipeline_ready,
                        (unsigned long long) st->unit_pipeline_waits,
                        st->unit_pipeline_wait_us / 1000.0);
                    std::fprintf(stderr,
                        "[elastic unit pipeline residency] total_ms=%.3f missing_units=%llu missing_tensors=%llu missing_mib=%.2f unissued_units=%llu unissued_tensors=%llu unissued_mib=%.2f unissued_ms=%.3f prepare_cap_declines=%llu prepare_cap_tensors=%llu\n",
                        st->unit_pipeline_current_residency_us / 1000.0,
                        (unsigned long long) st->unit_pipeline_current_missing_units,
                        (unsigned long long) st->unit_pipeline_current_missing_tensors,
                        st->unit_pipeline_current_missing_bytes / 1024.0 / 1024.0,
                        (unsigned long long) st->unit_pipeline_unissued_missing_units,
                        (unsigned long long) st->unit_pipeline_unissued_missing_tensors,
                        st->unit_pipeline_unissued_missing_bytes / 1024.0 / 1024.0,
                        st->unit_pipeline_unissued_missing_us / 1000.0,
                        (unsigned long long) st->async_prepare_cap_declines,
                        (unsigned long long) st->async_prepare_cap_declined_tensors);
                    std::fprintf(stderr,
                        "[elastic unit pipeline timing] stage_ms=%.3f retire_ms=%.3f evict_ms=%.3f\n",
                        st->unit_pipeline_stage_us / 1000.0,
                        st->unit_pipeline_retire_us / 1000.0,
                        st->evict_wall_us / 1000.0);
                    std::fprintf(stderr,
                        "[elastic unit pipeline window] samples=%llu avg_units=%.2f max_units=%zu avg_mib=%.2f max_mib=%.2f oversize=%llu\n",
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
                        "[elastic unit pipeline cross-graph] issued=%llu ready=%llu waits=%llu staged_mib=%.2f transitions=%zu\n",
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

        // 读 env 配置
        if (const char *kv = std::getenv("GGML_ELASTIC_KV_MB")) {
            s->kv_bytes = (size_t)std::atoll(kv) * 1024 * 1024;
        }
        if (const char *mc = std::getenv("GGML_ELASTIC_MISC_MB")) {
            s->misc_overhead = (size_t)std::atoll(mc) * 1024 * 1024;
        }
        if (const char *sc = std::getenv("GGML_ELASTIC_SAFETY_MB")) {
            s->misc_overhead += (size_t)std::atoll(sc) * 1024 * 1024;
        }
        if (const char *iv = std::getenv("GGML_ELASTIC_EVICT_INTERVAL")) {
            int v = std::atoi(iv);
            if (v >= 1) s->evict_interval = v;
        }
        if (const char *pf = std::getenv("GGML_ELASTIC_PREFETCH")) {
            int v = std::atoi(pf);
            if (v > 0) s->prefetch_lookahead = v;
        }
        {
            size_t host_pool_mb = 256;
            if (const char * pool = std::getenv("GGML_ELASTIC_HOST_STAGING_POOL_MB")) {
                host_pool_mb = static_cast<size_t>(std::max<long long>(0, std::atoll(pool)));
            }
            s->retain_host_staging = host_pool_mb > 0;
            s->host_staging_pool_limit = host_pool_mb * 1024ULL * 1024ULL;
            GGML_LOG_INFO("elastic: CPU host staging pool %s (cap=%zu MiB)\n",
                          s->retain_host_staging ? "enabled" : "disabled", host_pool_mb);
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
        const bool profile_env = []() {
            const char *p = std::getenv("GGML_ELASTIC_PROFILE");
            const char *t = std::getenv("GGML_ELASTIC_TIMING");
            return (p && *p && *p != '0') || (t && *t && *t != '0');
        }();
        if (profile_env) {
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
                    "  io_pt        : %.1f ms/graph (direct disk reload / fallback mmap)\n"
                    "  compute_pt   : %.1f ms/graph (delegate to ggml-cpu; total=%.2f ms calls=%llu avg=%.4f ms)\n"
                    "  bytes_pt     : %.1f MB/graph reloaded\n"
                    "  effective_bw : %.0f MB/s\n"
                    "  reload_count : %llu (%llu bytes total)\n"
                    "  direct_read  : calls=%llu ok=%llu fail=%llu total=%.2f ms MB=%.1f MB/s=%.1f\n"
                    "  direct_batch : units=%llu tensors=%llu fallbacks=%llu errors=%llu\n"
                    "  stage_load   : calls=%llu ok=%llu total=%.2f ms MB=%.1f\n"
                    "  stage_xform  : calls=%llu ok=%llu total=%.2f ms MB=%.1f\n"
                    "  staging_pool : hit=%llu miss=%llu retained=%.1f MiB cap=%.1f MiB direct_compatible=%llu\n"
                    "  stage_copy   : parallel_calls=%llu parallel_MB=%.1f\n"
                    "  async_load   : units=%llu/%llu tensors=%llu/%llu waits=%llu wait_total=%.2f ms\n"
                    "  async_prepare: units=%llu/%llu tensors=%llu/%llu waits=%llu wait_total=%.2f ms\n"
                    "  granularity  : mode=%s units=%llu cut_ops=%llu fallback_ops=%llu peak_unit=%.2f MiB tensors=%llu cut_tensors=%llu parts=%llu wbm_total=%.2f MiB\n"
                    "  evict_count  : %llu (%llu bytes total) wall=%.2f ms\n"
                    "============================================\n",
                    (unsigned long long)st->profile_n_graph,
                    io_pt, comp_pt,
                    st->profile_compute_total_ms,
                    (unsigned long long) st->profile_compute_calls,
                    st->profile_compute_calls
                        ? st->profile_compute_total_ms / st->profile_compute_calls : 0.0,
                    bytes_pt / 1024 / 1024,
                    bw,
                    (unsigned long long)st->n_reloads_total,
                    (unsigned long long)st->bytes_reloaded_total,
                    (unsigned long long)st->direct_read_calls,
                    (unsigned long long)st->direct_read_ok,
                    (unsigned long long)st->direct_read_fail,
                    st->direct_read_us / 1000.0,
                    st->direct_read_bytes / 1024.0 / 1024.0,
                    st->direct_read_us ? (st->direct_read_bytes / 1024.0 / 1024.0) / (st->direct_read_us / 1000000.0) : 0.0,
                    (unsigned long long) st->direct_batch_units,
                    (unsigned long long) st->direct_batch_tensors,
                    (unsigned long long) st->direct_batch_fallbacks,
                    (unsigned long long) st->direct_batch_errors,
                    (unsigned long long)st->stage_load_calls,
                    (unsigned long long)st->stage_load_ok,
                    st->stage_load_us / 1000.0,
                    st->stage_load_bytes / 1024.0 / 1024.0,
                    (unsigned long long)st->stage_xform_calls,
                    (unsigned long long)st->stage_xform_ok,
                    st->stage_xform_us / 1000.0,
                    st->stage_xform_bytes / 1024.0 / 1024.0,
                    (unsigned long long)st->host_staging_pool_hit,
                    (unsigned long long)st->host_staging_pool_miss,
                    st->host_staging_pool_bytes / 1024.0 / 1024.0,
                    st->host_staging_pool_limit / 1024.0 / 1024.0,
                    (unsigned long long)st->host_staging_direct_compatible,
                    (unsigned long long)st->copy_parallel_calls,
                    st->copy_parallel_bytes / 1024.0 / 1024.0,
                    (unsigned long long)st->async_load_units_enqueued,
                    (unsigned long long)st->async_load_units_completed,
                    (unsigned long long)st->async_load_enqueued,
                    (unsigned long long)st->async_load_completed,
                    (unsigned long long)st->async_load_waits,
                    st->async_load_wait_us / 1000.0,
                    (unsigned long long)st->async_prepare_units_enqueued,
                    (unsigned long long)st->async_prepare_units_completed,
                    (unsigned long long)st->async_prepare_enqueued,
                    (unsigned long long)st->async_prepare_completed,
                    (unsigned long long)st->async_prepare_waits,
                    st->async_prepare_wait_us / 1000.0,
                    elastic::granularity_mode_name(st->granularity.mode),
                    (unsigned long long)st->granularity_units,
                    (unsigned long long)st->granularity_cut_ops,
                    (unsigned long long)st->granularity_fallback_ops,
                    st->granularity_peak_unit_bytes / 1024.0 / 1024.0,
                    (unsigned long long)st->granularity_registered_tensors,
                    (unsigned long long)st->granularity_cut_tensors,
                    (unsigned long long)st->granularity_registered_parts,
                    elastic::wbm_total_bytes(&st->wbm) / 1024.0 / 1024.0,
                    (unsigned long long)st->n_evicts_total,
                    (unsigned long long)st->bytes_evicted_total,
                    st->evict_wall_us / 1000.0);
                std::fprintf(stderr,
                    "[elastic cpu layout] initial_repack_calls=%llu initial_repack_ms=%.2f initial_repack_mib=%.1f "
                    "repack_calls=%llu repack_ok=%llu repack_ms=%.2f repack_mib=%.1f "
                    "materialize_calls=%llu materialize_ms=%.2f materialize_mib=%.1f\n",
                    (unsigned long long) st->initial_repack_calls,
                    st->initial_repack_us / 1000.0,
                    st->initial_repack_bytes / 1024.0 / 1024.0,
                    (unsigned long long) st->stage_repack_calls,
                    (unsigned long long) st->stage_repack_ok,
                    st->stage_repack_us / 1000.0,
                    st->stage_repack_bytes / 1024.0 / 1024.0,
                    (unsigned long long) st->stage_materialize_calls,
                    st->stage_materialize_us / 1000.0,
                    st->stage_materialize_bytes / 1024.0 / 1024.0);
                std::fprintf(stderr,
                    "[elastic cpu multi_fused] layout_pairs=%llu ok=%llu fallback=%llu parallel=%llu ms=%.2f mib=%.1f "
                    "kernel_candidates=%llu calls=%llu fallback=%llu errors=%llu ms=%.2f "
                    "workspace_alloc=%llu reuse=%llu bytes=%zu "
                    "scan_mul_mat=%llu shared_activation=%llu compatible=%llu "
                    "weight_units=%llu safe_output=%llu\n",
                    (unsigned long long) st->fused_layout_pair_calls,
                    (unsigned long long) st->fused_layout_pair_ok,
                    (unsigned long long) st->fused_layout_pair_fallbacks,
                    (unsigned long long) st->fused_layout_parallel_calls,
                    st->fused_layout_pair_us / 1000.0,
                    st->fused_layout_pair_bytes / 1024.0 / 1024.0,
                    (unsigned long long) st->fused_kernel_pair_candidates,
                    (unsigned long long) st->fused_kernel_pair_calls,
                    (unsigned long long) st->fused_kernel_pair_fallbacks,
                    (unsigned long long) st->fused_kernel_pair_errors,
                    st->fused_kernel_pair_us / 1000.0,
                    (unsigned long long) st->fused_workspace_allocations,
                    (unsigned long long) st->fused_workspace_reuses,
                    st->fused_compute_workspace.size(),
                    (unsigned long long) st->fused_scan_mul_mat,
                    (unsigned long long) st->fused_scan_shared_activation,
                    (unsigned long long) st->fused_scan_compatible,
                    (unsigned long long) st->fused_scan_weight_units,
                    (unsigned long long) st->fused_scan_safe_output);
                std::fprintf(stderr,
                    "[elastic cpu multi io] parallel_units=%llu parallel_tensors=%llu\n",
                    (unsigned long long) st->multi_io_parallel_units,
                    (unsigned long long) st->multi_io_parallel_tensors);
            });
        }
        const char *csv = std::getenv("GGML_ELASTIC_BUDGET_CSV");
        if (csv && *csv) {
            if (elastic::budget_watcher_init(&s->bw, csv) == 0) {
                s->bw_inited = true;
                size_t mfloor = s->bw.m_floor_mb * 1024 * 1024;
                size_t kvm    = s->kv_bytes + s->misc_overhead;
                s->static_target = mfloor > kvm ? mfloor - kvm : 0;
                if (const char *d = std::getenv("GGML_ELASTIC_DYNAMIC"); d && *d && *d != '0') {
                    s->dynamic_target = true;
                }
                GGML_LOG_INFO("elastic: BudgetWatcher trace=%s M_floor=%zu MB -> static_target=%zu MB "
                              "mode=%s (kv=%zu MB misc+safety=%zu MB)\n",
                              csv, s->bw.m_floor_mb, s->static_target / 1024 / 1024,
                              s->dynamic_target ? "DYNAMIC" : "static",
                              s->kv_bytes / 1024 / 1024,
                              s->misc_overhead / 1024 / 1024);
            }
        }
    }

    if (tensor->view_src == nullptr && offset == 0 && size == ggml_nbytes(tensor)) {
        elastic_tensor_units units;
        const bool wants_cut =
            elastic::granularity_dynamic_prepare_cut() ||
            (s->granularity.explicitly_enabled &&
             s->granularity.mode == elastic::granularity_mode::CUT);
        std::vector<elastic::row_partition> row_parts;
        if (wants_cut && ggml_n_dims(tensor) == 2 && ggml_is_contiguous(tensor) &&
            tensor->ne[0] > 0 && tensor->ne[1] > 0) {
            const int64_t page_rows = static_cast<int64_t>(4096 / std::gcd<size_t>(tensor->nb[1], 4096));
            const int64_t repack_rows = static_cast<int64_t>(
                ggml_backend_cpu_repack_row_alignment(tensor));
            const int64_t cut_alignment = std::lcm(page_rows, repack_rows);
            row_parts = elastic::granularity_partition_rows(
                tensor->ne[1], s->granularity.cut_parts, cut_alignment);
        }
        if (!row_parts.empty()) {
            units.row_cut = true;
            for (const auto & part : row_parts) {
                const size_t part_offset = static_cast<size_t>(part.row_start) * tensor->nb[1];
                const size_t part_size = static_cast<size_t>(part.row_count) * tensor->nb[1];
                units.parts.push_back({-1, part.row_start, part.row_count, part_offset, part_size});
            }
        } else {
            units.parts.push_back({-1, 0, tensor->ne[1], 0, size});
        }

        std::vector<int> registered_indices;
        for (auto & part : units.parts) {
            const void * part_host = static_cast<const char *>(data) + part.byte_offset;
            void * part_handle = static_cast<char *>(tensor->data) + part.byte_offset;
            const int idx = elastic::wbm_add_block(
                &s->wbm, const_cast<void *>(part_host), part.byte_size);
            if (idx < 0) continue;
            part.wbm_idx = idx;
            registered_indices.push_back(idx);
            elastic::wbm_mark_resident(&s->wbm, idx, part_handle);
            elastic::wbm_touch(&s->wbm, idx, 0);
            while ((int)bctx->block_pinned.size() <= idx) bctx->block_pinned.push_back(false);
            {
                std::lock_guard<std::mutex> lk(s->sched_mtx);
                s->wbm_to_name[idx] = tensor->name;
                s->bctx_by_idx[idx] = bctx;
                s->backend_handle_by_idx[idx] = part_handle;
                s->tensor_slice_by_idx[idx] = {
                    tensor, part.row_start, part.row_count, native_repack
                };
            }
        }
        if (!registered_indices.empty()) {
            s->granularity_registered_tensors++;
            s->granularity_registered_parts += registered_indices.size();
            if (units.row_cut) s->granularity_cut_tensors++;
            bctx->tensor_to_wbm[tensor] = registered_indices.front();
            bctx->tensor_to_units[tensor] = units;
            {
                std::lock_guard<std::mutex> lk(s->sched_mtx);
                s->name_to_wbm[tensor->name] = registered_indices.front();
                s->name_to_wbms[tensor->name] = registered_indices;
            }
            elastic_sched_register_once();

            // pin policy
            std::string suf = tensor_suffix(tensor->name);
            std::string pp  = s_pin_policy();
            bool should_pin = false;
            if (pp == "all") {
                should_pin = true;  // 真的所有 tensor 都 pin
            } else {
                auto contains = [&](const char *t) {
                    const std::string delimited = "," + pp + ",";
                    return delimited.find(
                        "," + std::string(t) + ",") !=
                        std::string::npos;
                };
                if (contains("norm") && (suf == "attn_norm" || suf == "ffn_norm" || suf == "output_norm")) should_pin = true;
                if (contains("k") && suf == "attn_k") should_pin = true;
                if (contains("v") && suf == "attn_v") should_pin = true;
                if (contains("q") && suf == "attn_q") should_pin = true;
                if (contains("o") && suf == "attn_output") should_pin = true;
                // Plan-controlled granularity experiments can keep the large
                // shared embedding/output tensor resident in every mode.  It
                // still counts against the ordinary weight budget; unlike
                // GGML_ELASTIC_EMBED_OUTSIDE_BUDGET this does not enlarge the
                // target.  In CUT mode all physical parts inherit the same
                // logical-tensor pin decision.
                if (contains("token_embd") && suf == "token_embd") should_pin = true;
                if (contains("output") && suf == "output") should_pin = true;
            }
            // 外部 schedule hook (LP solver / 自定义): callback 返回 true 则覆盖 env policy
            // 也强制 pin. callback 为空时 fall back 到 env policy. 不允许 callback 解 pin
            // (env pin 是基线), 但允许 callback 额外 pin.
            if (llama_weight_pin_query(tensor->name, -1, size)) {
                should_pin = true;
            }
            if (should_pin) {
                for (int idx : registered_indices) {
                    elastic::wbm_set_pinned(&s->wbm, idx, true);
                    bctx->block_pinned[idx] = true;
                }
                if (suf == "token_embd" || suf == "output") {
                    GGML_LOG_INFO(
                        "elastic: pin %s inside weight budget "
                        "(%.2f MiB, %zu physical unit%s)\n",
                        suf.c_str(),
                        size / 1024.0 / 1024.0,
                        registered_indices.size(),
                        registered_indices.size() == 1 ? "" : "s");
                }
            }
            // EMBED_OUTSIDE_BUDGET
            if (s_embed_out() && suf == "token_embd") {
                for (int idx : registered_indices) {
                    elastic::wbm_set_pinned(&s->wbm, idx, true);
                    bctx->block_pinned[idx] = true;
                }
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
    /* .init_tensor     = */ elastic_buffer_init_tensor,
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
    auto *bctx = new elastic_buffer_ctx{data, size, {}, {}, {}};
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

    // Plan-driven granularity changes are published through the process
    // environment after llama synchronizes the previous graph. Snapshot the
    // new grouping only at graph entry; physical CUT tiles were provisioned
    // during model load when GGML_ELASTIC_GRANULARITY_DYNAMIC=1.
    const elastic::granularity_config planned_granularity =
        elastic::granularity_from_env();
    if (planned_granularity.explicitly_enabled &&
        (planned_granularity.mode != s->granularity.mode ||
         planned_granularity.cut_parts != s->granularity.cut_parts ||
         planned_granularity.multi_tensors !=
             s->granularity.multi_tensors)) {
        GGML_LOG_INFO(
            "elastic: working-unit plan switch %s -> %s "
            "cut_parts=%d multi_tensors=%d\n",
            elastic::granularity_mode_name(s->granularity.mode),
            elastic::granularity_mode_name(planned_granularity.mode),
            planned_granularity.cut_parts,
            planned_granularity.multi_tensors);
        s->granularity = planned_granularity;
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
    const bool csv_profile = s->profile_csv;
    if (s->profile || csv_profile) t_io_start = clk::now();

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

    auto resolve_units = [](const ggml_tensor *t,
                            const ggml_tensor *&root,
                            elastic_buffer_ctx *&bc,
                            const elastic_tensor_units *&units) {
        root = nullptr;
        bc = nullptr;
        units = nullptr;
        if (!t || !t->buffer || t->buffer->iface.set_tensor != elastic_buffer_set_tensor) return false;
        bc = static_cast<elastic_buffer_ctx *>(t->buffer->context);
        root = t;
        while (root && root->view_src) root = root->view_src;
        auto it = bc->tensor_to_units.find(root);
        if (it == bc->tensor_to_units.end()) return false;
        units = &it->second;
        return true;
    };

    // A mixed plan binds each pre-provisioned physical row tile to a logical
    // unit id. The cache is graph-local: apply_exec_plan synchronizes before
    // publishing a new generation, so no binding can change mid-graph.
    std::unordered_map<std::string, std::vector<llama_weight_unit_slice>>
        mixed_slice_cache;
    auto mixed_slices_for = [&](const ggml_tensor * root)
            -> const std::vector<llama_weight_unit_slice> & {
        static const std::vector<llama_weight_unit_slice> empty;
        if (!root || !root->name[0]) return empty;
        auto found = mixed_slice_cache.find(root->name);
        if (found != mixed_slice_cache.end()) return found->second;
        const int count =
            llama_weight_unit_plan_query(root->name, nullptr, 0);
        std::vector<llama_weight_unit_slice> slices;
        if (count > 0) {
            slices.resize(static_cast<size_t>(count));
            const int copied = llama_weight_unit_plan_query(
                root->name, slices.data(), count);
            if (copied < count) {
                slices.resize(static_cast<size_t>(std::max(0, copied)));
            }
        }
        return mixed_slice_cache.emplace(root->name, std::move(slices))
            .first->second;
    };
    auto mixed_binding_for_part = [&](
            const ggml_tensor * root,
            const elastic_tensor_part & part,
            llama_weight_unit_slice & out) {
        const auto & slices = mixed_slices_for(root);
        // Prefer an exact base-tile match. A whole-tensor entry is a useful
        // compatibility shorthand and binds all physical tiles to one unit.
        for (const auto & slice : slices) {
            if (slice.row_start == part.row_start &&
                slice.row_count == part.row_count) {
                out = slice;
                return true;
            }
        }
        for (const auto & slice : slices) {
            if (slice.row_start == 0 && slice.row_count < 0) {
                out = slice;
                return true;
            }
        }
        return false;
    };

    auto target_bytes = [&]() -> size_t {
        if (!s->bw_inited) return SIZE_MAX;  // 没 trace → 不 evict
        if (s->dynamic_target) {
            const size_t bt   = elastic::budget_watcher_get(&s->bw) * size_t(1024 * 1024);
            const size_t km   = s->kv_bytes + s->misc_overhead;
            const size_t base = bt > km ? bt - km : 0;
            return base + s->extra_target;
        }
        return s->static_target;  // baseline 静态
    };

    // Explicit granularity mode uses real graph submissions. Each unit is
    // made resident, computed, and retired before the next unit. CUT replaces
    // an eligible 2-D MUL_MAT with independent output-row slices.
    if (s->granularity.explicitly_enabled) {
        // Filled after the ordinary Multi runtime units have been built.
        // Keeping this graph-local map separate from unit construction lets
        // Multi and Multi_Fused use exactly the same pipeline boundaries.
        std::unordered_map<int, int> fused_layout_partner;

        auto buffer_for_idx = [s](int idx) -> elastic_buffer_ctx * {
            std::lock_guard<std::mutex> lk(s->sched_mtx);
            auto it = s->bctx_by_idx.find(idx);
            return it == s->bctx_by_idx.end() ? nullptr : it->second;
        };

        static const bool protect_plan_resident = []() {
            const char * e = std::getenv("LLAMA_ELASTIC_FORCE_PLAN_EVICT");
            return e && *e && *e != '0';
        }();
        std::vector<int> plan_resident_indices;
        std::unordered_set<int> configured_pinned_indices;
        {
            // Buffer metadata records the model's persistent pin policy.
            // Keep it separate from transient unit/cross-graph pins so hard
            // budget enforcement may cancel speculative lookahead without
            // ever evicting token_embd/output or another true pinned weight.
            std::lock_guard<std::mutex> lk(s->sched_mtx);
            for (const auto & entry : s->bctx_by_idx) {
                const int idx = entry.first;
                const elastic_buffer_ctx * bctx = entry.second;
                if (!bctx || idx < 0 ||
                    static_cast<size_t>(idx) >=
                        bctx->block_pinned.size()) {
                    continue;
                }
                if (bctx->block_pinned[static_cast<size_t>(idx)]) {
                    configured_pinned_indices.insert(idx);
                }
            }
        }
        if (protect_plan_resident) {
            plan_resident_indices.reserve(s->wbm.blocks.size());
            for (const elastic::block_meta & block : s->wbm.blocks) {
                const char * name = name_for_idx(s, block.block_idx);
                if (name && *name &&
                    llama_weight_runtime_desired_query(name) ==
                        LLAMA_WEIGHT_RUNTIME_CPU) {
                    plan_resident_indices.push_back(block.block_idx);
                }
            }
        }

        auto evict_to = [&](size_t target, const std::vector<int> & protected_indices) {
            if (target == SIZE_MAX) return;
            const std::unordered_set<int> hard_protected(
                protected_indices.begin(), protected_indices.end());
            std::unordered_set<int> in_flight;
            {
                std::lock_guard<std::mutex> lk(s->sched_mtx);
                for (const auto & entry : s->async_load_state) {
                    if (entry.second == 1) {
                        in_flight.insert(entry.first);
                    }
                }
                for (const auto & entry : s->async_prepare_state) {
                    if (entry.second == 1) {
                        in_flight.insert(entry.first);
                    }
                }
            }
            size_t unprotected_pending_bytes = 0;
            for (int idx : in_flight) {
                if (hard_protected.count(idx) != 0) continue;
                const elastic::block_meta * block =
                    elastic::wbm_get(&s->wbm, idx);
                if (block && !block->resident) {
                    unprotected_pending_bytes += block->byte_size;
                }
            }
            // Callers already subtract the missing bytes of their protected
            // lookahead window. Reserve additionally for older asynchronous
            // work that is still queued but no longer belongs to that window;
            // otherwise it can complete after eviction and transiently push
            // residency above the hard budget.
            const size_t eviction_target =
                target > unprotected_pending_bytes
                    ? target - unprotected_pending_bytes
                    : 0;
            if (s->wbm.resident_bytes <= eviction_target) return;
            std::unordered_map<int, bool> old_pin;
            auto protect = [&](int idx) {
                const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                if (!bm) return;
                if (!old_pin.emplace(idx, bm->is_pinned).second) return;
                elastic::wbm_set_pinned(&s->wbm, idx, true);
            };
            for (int idx : protected_indices) {
                protect(idx);
            }
            if (protect_plan_resident) {
                for (int idx : plan_resident_indices) {
                    protect(idx);
                }
            }
            std::vector<int> victims;
            elastic::wbm_evict_to_byte_budget(
                &s->wbm, eviction_target, -1, &victims);
            if (!victims.empty()) {
                evict_blocks(
                    s, buffer_for_idx(victims.front()), victims);
            }

            // The residency plan is a preference, while the live memory
            // budget is a hard constraint. First preserve every planned
            // resident and evict ordinary streaming blocks. If that does not
            // free enough space (for example, a budget drop occurs within a
            // long token), relax only movable plan weights and keep the
            // current working unit plus genuinely pinned weights protected.
            if (protect_plan_resident &&
                s->wbm.resident_bytes > eviction_target) {
                auto relax_and_evict = [&](
                        bool allow_in_flight, size_t pass_target) {
                    size_t relaxed_bytes = 0;
                    for (const elastic::block_meta & block : s->wbm.blocks) {
                        const int idx = block.block_idx;
                        if (hard_protected.count(idx) != 0) continue;
                        if (!allow_in_flight &&
                            in_flight.count(idx) != 0) {
                            continue;
                        }
                        if (!block.resident || !block.is_pinned ||
                            configured_pinned_indices.count(idx) != 0) {
                            continue;
                        }
                        // Preserve a transient pipeline pin across the
                        // fallback even if this index was not part of the
                        // logical plan's protected set.
                        old_pin.emplace(idx, true);
                        elastic::wbm_set_pinned(&s->wbm, idx, false);
                        relaxed_bytes += block.byte_size;
                    }
                    if (relaxed_bytes == 0) return;
                    std::vector<int> fallback_victims;
                    elastic::wbm_evict_to_byte_budget(
                        &s->wbm, pass_target, -1, &fallback_victims);
                    if (!fallback_victims.empty()) {
                        s->unit_pipeline_plan_protection_relaxations++;
                        for (int idx : fallback_victims) {
                            const elastic::block_meta * bm =
                                elastic::wbm_get(&s->wbm, idx);
                            if (bm) {
                                s->unit_pipeline_plan_protection_relaxed_bytes +=
                                    bm->byte_size;
                            }
                        }
                        evict_blocks(
                            s, buffer_for_idx(fallback_victims.front()),
                            fallback_victims);
                    }
                };
                relax_and_evict(false, eviction_target);

                // A cross-graph PREPARE can finish after the ordinary
                // eviction pass and add its bytes back to WBM. Do not cancel
                // or evict a buffer while the worker is writing it. This
                // slow-path is entered only when no immediately evictable
                // block can satisfy the hard budget: drain the outstanding
                // speculative units, then make one final eviction pass.
                if (s->wbm.resident_bytes > eviction_target &&
                    !in_flight.empty()) {
                    for (int idx : in_flight) {
                        cpu_wait_async_prepare(s, idx);
                        cpu_wait_async_load(s, idx);
                    }
                    if (s->wbm.resident_bytes > target) {
                        relax_and_evict(true, target);
                    }
                }
            }
            for (const auto & entry : old_pin) {
                elastic::wbm_set_pinned(&s->wbm, entry.first, entry.second);
            }
        };

        auto sample_pipeline_budget = [&](size_t target) {
            if (target == SIZE_MAX) return;
            constexpr size_t alignment_slack = 64ULL * 1024;
            const size_t aligned_target =
                target <= SIZE_MAX - alignment_slack
                    ? target + alignment_slack
                    : SIZE_MAX;
            size_t pinned_resident_bytes = 0;
            for (const elastic::block_meta & block : s->wbm.blocks) {
                if (block.resident && block.is_pinned) {
                    pinned_resident_bytes += block.byte_size;
                }
            }
            s->unit_pipeline_budget_samples++;
            s->unit_pipeline_resident_bytes_peak = std::max(
                s->unit_pipeline_resident_bytes_peak,
                s->wbm.resident_bytes);
            s->unit_pipeline_pinned_bytes_peak = std::max(
                s->unit_pipeline_pinned_bytes_peak,
                pinned_resident_bytes);
            if (s->wbm.resident_bytes > aligned_target) {
                s->unit_pipeline_budget_violations++;
                s->unit_pipeline_over_budget_bytes_peak = std::max(
                    s->unit_pipeline_over_budget_bytes_peak,
                    s->wbm.resident_bytes - target);
                if (s->unit_pipeline_budget_violations <= 4) {
                    std::fprintf(
                        stderr,
                        "[elastic unit pipeline budget debug] violation=%llu "
                        "target_mib=%.2f resident_mib=%.2f pinned_mib=%.2f\n",
                        (unsigned long long)
                            s->unit_pipeline_budget_violations,
                        target / 1024.0 / 1024.0,
                        s->wbm.resident_bytes / 1024.0 / 1024.0,
                        pinned_resident_bytes / 1024.0 / 1024.0);
                }
            }
        };

        auto make_unit_resident = [&](const std::vector<int> & indices) {
            size_t missing = 0;
            size_t unit_bytes = 0;
            for (int idx : indices) {
                const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                if (!bm) continue;
                unit_bytes += bm->byte_size;
                if (!bm->resident) missing += bm->byte_size;
            }
            s->granularity_peak_unit_bytes = std::max(s->granularity_peak_unit_bytes, unit_bytes);
            const size_t target = target_bytes();
            if (target != SIZE_MAX && missing > 0) {
                const size_t before_load_target = target > missing ? target - missing : 0;
                evict_to(before_load_target, indices);
            }
            std::vector<std::pair<int, int>> layout_pairs;
            std::unordered_set<int> unit_indices(
                indices.begin(), indices.end());
            for (int idx : indices) {
                auto partner = fused_layout_partner.find(idx);
                if (partner == fused_layout_partner.end() ||
                    idx >= partner->second ||
                    unit_indices.count(partner->second) == 0) {
                    continue;
                }
                layout_pairs.emplace_back(idx, partner->second);
            }
            if (!layout_pairs.empty() && missing > 0) {
                // Preserve pair atomicity even for the first/unissued unit:
                // route it through the same pooled PREPARE worker so the two
                // corresponding tile repacks can share transformation calls.
                cpu_enqueue_async_prepare_unit(
                    s, indices, false, &layout_pairs);
                for (int idx : indices) cpu_wait_async_prepare(s, idx);
            }
            for (int idx : indices) {
                const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                if (!bm || bm->resident) continue;
                ensure_block_resident_pipeline(s, idx, cpu_fixed_handle_for_idx(s, idx));
            }
        };

        auto retire_unit = [&](const std::vector<int> & indices) {
            for (int idx : indices) elastic::wbm_touch(&s->wbm, idx, 0);
            const size_t target = target_bytes();
            evict_to(target, {});
            sample_pipeline_budget(target);
            s->granularity_units++;
        };

        auto collect_indices = [&](int begin, int end) {
            std::vector<int> indices;
            std::unordered_set<int> seen;
            for (int i = begin; i < end; ++i) {
                ggml_tensor * node = cgraph->nodes[i];
                if (!node) continue;
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    const ggml_tensor * root = nullptr;
                    elastic_buffer_ctx * bc = nullptr;
                    const elastic_tensor_units * units = nullptr;
                    if (!resolve_units(node->src[j], root, bc, units)) continue;
                    GGML_UNUSED(root);
                    GGML_UNUSED(bc);
                    for (const auto & part : units->parts) {
                        if (part.wbm_idx >= 0 && seen.insert(part.wbm_idx).second) {
                            indices.push_back(part.wbm_idx);
                        }
                    }
                }
            }
            return indices;
        };

        auto compute_graph_range = [&](int begin, int end) -> ggml_status {
            if (begin >= end) return GGML_STATUS_SUCCESS;
            const std::vector<int> indices = collect_indices(begin, end);
            make_unit_resident(indices);
            ggml_cgraph view = ggml_graph_view(cgraph, begin, end);
            const auto compute_start = clk::now();
            const ggml_status status = bctx->cpu->iface.graph_compute(bctx->cpu, &view);
            const double ms = std::chrono::duration<double, std::milli>(clk::now() - compute_start).count();
            if (csv_profile) cpu_profile_compute_graph(&view, ms, status == GGML_STATUS_SUCCESS ? 1 : 0);
            if (s->profile) {
                s->profile_compute_total_ms += ms;
                s->profile_compute_calls++;
            }
            retire_unit(indices);
            return status;
        };

        auto cut_units_for_node = [&](ggml_tensor * node,
                                      const elastic_tensor_units *& units) -> bool {
            units = nullptr;
            if (!node || node->op != GGML_OP_MUL_MAT ||
                !node->src[0] || node->src[0]->view_src ||
                ggml_n_dims(node->src[0]) != 2) return false;
            const ggml_tensor * root = nullptr;
            elastic_buffer_ctx * bc = nullptr;
            if (!resolve_units(node->src[0], root, bc, units)) return false;
            GGML_UNUSED(bc);
            if (root != node->src[0] || !units || !units->row_cut ||
                units->parts.size() <= 1) {
                return false;
            }
            const auto & planned = mixed_slices_for(root);
            if (planned.empty()) {
                return s->granularity.mode ==
                    elastic::granularity_mode::CUT;
            }
            int first_unit = -1;
            for (const auto & part : units->parts) {
                llama_weight_unit_slice binding;
                if (!mixed_binding_for_part(root, part, binding)) {
                    return false;
                }
                if (first_unit < 0) {
                    first_unit = binding.unit_id;
                } else if (binding.unit_id != first_unit) {
                    return true;
                }
            }
            return false;
        };

        auto compute_cut_part = [&](int node_index,
                                    const elastic_tensor_part & part,
                                    bool retire_after_compute) -> ggml_status {
            ggml_tensor * node = cgraph->nodes[node_index];
            if (!node || part.wbm_idx < 0) return GGML_STATUS_FAILED;
            const std::vector<int> one{part.wbm_idx};
            make_unit_resident(one);

            ggml_tensor src_part = *node->src[0];
            src_part.ne[1] = part.row_count;
            src_part.ne[2] = 1;
            src_part.ne[3] = 1;
            src_part.nb[2] = src_part.nb[1] * static_cast<size_t>(part.row_count);
            src_part.nb[3] = src_part.nb[2];
            src_part.data = cpu_fixed_handle_for_idx(s, part.wbm_idx);
            src_part.view_src = nullptr;
            src_part.view_offs = 0;

            ggml_tensor dst_part = *node;
            dst_part.ne[0] = part.row_count;
            dst_part.data = static_cast<char *>(node->data) +
                            static_cast<size_t>(part.row_start) * node->nb[0];
            dst_part.src[0] = &src_part;
            dst_part.view_src = nullptr;
            dst_part.view_offs = 0;

            ggml_tensor * part_node = &dst_part;
            ggml_cgraph part_graph = ggml_graph_view(cgraph, node_index, node_index + 1);
            part_graph.nodes = &part_node;
            const auto compute_start = clk::now();
            const ggml_status status = bctx->cpu->iface.graph_compute(bctx->cpu, &part_graph);
            const double ms = std::chrono::duration<double, std::milli>(clk::now() - compute_start).count();
            if (csv_profile) cpu_profile_compute_node(&dst_part, node_index, ms,
                                                      status == GGML_STATUS_SUCCESS ? 1 : 0);
            if (s->profile) {
                s->profile_compute_total_ms += ms;
                s->profile_compute_calls++;
            }
            if (retire_after_compute) retire_unit(one);
            return status;
        };

        auto compute_cut_node = [&](int node_index, const elastic_tensor_units & units) -> ggml_status {
            for (const auto & part : units.parts) {
                const ggml_status status = compute_cut_part(node_index, part, true);
                if (status != GGML_STATUS_SUCCESS) return status;
            }
            s->granularity_cut_ops++;
            return GGML_STATUS_SUCCESS;
        };

        const bool multi_unit =
            s->granularity.mode == elastic::granularity_mode::MULTI ||
            s->granularity.mode == elastic::granularity_mode::MULTI_FUSED;
        const int tensors_per_unit = multi_unit ? s->granularity.multi_tensors : 1;

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
            // Zero keeps the LOAD -> PREPARE -> COMPUTE pipeline within the
            // current graph, but disables speculative staging of successor
            // graphs.  This is useful for fine-grained units whose successor
            // window can otherwise churn the weight budget.
            return static_cast<size_t>(std::max<long long>(0, graphs));
        }();

        if (unit_pipeline_enabled) {
            struct cpu_runtime_unit {
                enum class kind { GRAPH_RANGE, CUT_PART } type = kind::GRAPH_RANGE;
                int begin = 0;
                int end = 0;
                int node_index = -1;
                int plan_unit_id = -1;
                uint32_t plan_flags = 0;
                elastic_tensor_part cut_part;
                bool last_cut_part = false;
                std::vector<int> indices;
                bool pipeline_issued = false;
            };

            std::vector<cpu_runtime_unit> runtime_units;
            auto append_range = [&](int begin, int end,
                                    int plan_unit_id = -1,
                                    uint32_t plan_flags = 0) {
                if (begin >= end) return;
                cpu_runtime_unit unit;
                unit.type = cpu_runtime_unit::kind::GRAPH_RANGE;
                unit.begin = begin;
                unit.end = end;
                unit.plan_unit_id = plan_unit_id;
                unit.plan_flags = plan_flags;
                unit.indices = collect_indices(begin, end);
                runtime_units.push_back(std::move(unit));
            };

            auto append_plain_segment = [&](int segment_begin, int segment_end) {
                int range_begin = segment_begin;
                int weights = 0;
                std::unordered_set<const ggml_tensor *> roots;
                for (int i = segment_begin; i < segment_end; ++i) {
                    ggml_tensor * node = cgraph->nodes[i];
                    if (node) {
                        for (int j = 0; j < GGML_MAX_SRC; ++j) {
                            const ggml_tensor * root = nullptr;
                            elastic_buffer_ctx * bc = nullptr;
                            const elastic_tensor_units * units = nullptr;
                            if (!resolve_units(node->src[j], root, bc, units)) continue;
                            GGML_UNUSED(bc);
                            GGML_UNUSED(units);
                            if (roots.insert(root).second) weights++;
                        }
                    }
                    if (weights >= tensors_per_unit) {
                        append_range(range_begin, i + 1);
                        range_begin = i + 1;
                        weights = 0;
                        roots.clear();
                    }
                }
                append_range(range_begin, segment_end);
            };

            bool graph_has_mixed_plan = false;
            for (int i = 0; i < cgraph->n_nodes && !graph_has_mixed_plan; ++i) {
                ggml_tensor * node = cgraph->nodes[i];
                if (!node) continue;
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    const ggml_tensor * root = nullptr;
                    elastic_buffer_ctx * bc = nullptr;
                    const elastic_tensor_units * units = nullptr;
                    if (!resolve_units(node->src[j], root, bc, units)) continue;
                    GGML_UNUSED(bc);
                    GGML_UNUSED(units);
                    if (!mixed_slices_for(root).empty()) {
                        graph_has_mixed_plan = true;
                        break;
                    }
                }
            }

            if (graph_has_mixed_plan) {
                int range_begin = 0;
                int current_unit_id = -1;
                uint32_t current_flags = 0;
                int fallback_unit_id = -2;
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    const elastic_tensor_units * cut_units = nullptr;
                    if (cut_units_for_node(cgraph->nodes[i], cut_units)) {
                        append_range(
                            range_begin, i, current_unit_id, current_flags);
                        for (size_t p = 0; p < cut_units->parts.size(); ++p) {
                            cpu_runtime_unit unit;
                            unit.type = cpu_runtime_unit::kind::CUT_PART;
                            unit.node_index = i;
                            unit.cut_part = cut_units->parts[p];
                            unit.last_cut_part =
                                p + 1 == cut_units->parts.size();
                            const ggml_tensor * root =
                                cgraph->nodes[i]->src[0];
                            llama_weight_unit_slice binding;
                            if (mixed_binding_for_part(
                                    root, unit.cut_part, binding)) {
                                unit.plan_unit_id = binding.unit_id;
                                unit.plan_flags = binding.flags;
                            }
                            if (unit.cut_part.wbm_idx >= 0) {
                                unit.indices.push_back(
                                    unit.cut_part.wbm_idx);
                            }
                            runtime_units.push_back(std::move(unit));
                        }
                        range_begin = i + 1;
                        current_unit_id = -1;
                        current_flags = 0;
                        continue;
                    }

                    int node_unit_id = -1;
                    uint32_t node_flags = 0;
                    ggml_tensor * node = cgraph->nodes[i];
                    if (node) {
                        for (int j = 0; j < GGML_MAX_SRC; ++j) {
                            const ggml_tensor * root = nullptr;
                            elastic_buffer_ctx * bc = nullptr;
                            const elastic_tensor_units * units = nullptr;
                            if (!resolve_units(
                                    node->src[j], root, bc, units)) {
                                continue;
                            }
                            GGML_UNUSED(bc);
                            if (mixed_slices_for(root).empty()) continue;
                            int root_unit_id = -1;
                            uint32_t root_flags = 0;
                            bool one_unit = true;
                            for (const auto & part : units->parts) {
                                llama_weight_unit_slice binding;
                                if (!mixed_binding_for_part(
                                        root, part, binding)) {
                                    one_unit = false;
                                    break;
                                }
                                if (root_unit_id < 0) {
                                    root_unit_id = binding.unit_id;
                                    root_flags = binding.flags;
                                } else if (
                                    root_unit_id != binding.unit_id) {
                                    one_unit = false;
                                    break;
                                } else {
                                    root_flags |= binding.flags;
                                }
                            }
                            if (one_unit && root_unit_id >= 0) {
                                node_unit_id = root_unit_id;
                                node_flags |= root_flags;
                                break;
                            }
                        }
                    }

                    if (node_unit_id < 0 && node) {
                        // Unplanned elastic weights retain tensor semantics.
                        // Give each encountered root a local synthetic id so
                        // they cannot accidentally merge across a gap.
                        for (int j = 0; j < GGML_MAX_SRC; ++j) {
                            const ggml_tensor * root = nullptr;
                            elastic_buffer_ctx * bc = nullptr;
                            const elastic_tensor_units * units = nullptr;
                            if (resolve_units(
                                    node->src[j], root, bc, units)) {
                                GGML_UNUSED(root);
                                GGML_UNUSED(bc);
                                GGML_UNUSED(units);
                                node_unit_id = fallback_unit_id--;
                                break;
                            }
                        }
                    }
                    // Nodes without an elastic weight remain in the preceding
                    // range. A new weight id closes that range immediately
                    // before the new operator, preserving graph dependencies.
                    if (node_unit_id >= 0 || node_unit_id <= -2) {
                        if (current_unit_id == -1) {
                            current_unit_id = node_unit_id;
                            current_flags = node_flags;
                        } else if (node_unit_id != current_unit_id) {
                            append_range(
                                range_begin, i,
                                current_unit_id, current_flags);
                            range_begin = i;
                            current_unit_id = node_unit_id;
                            current_flags = node_flags;
                        } else {
                            current_flags |= node_flags;
                        }
                    }
                }
                append_range(
                    range_begin, cgraph->n_nodes,
                    current_unit_id, current_flags);
            } else if (
                s->granularity.mode ==
                elastic::granularity_mode::MULTI_FUSED) {
                append_plain_segment(0, cgraph->n_nodes);
            } else {
                int build_range_begin = 0;
                int build_weights = 0;
                std::unordered_set<const ggml_tensor *> build_roots;
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    const elastic_tensor_units * cut_units = nullptr;
                    if (cut_units_for_node(cgraph->nodes[i], cut_units)) {
                        append_range(build_range_begin, i);
                        for (size_t p = 0; p < cut_units->parts.size(); ++p) {
                            cpu_runtime_unit unit;
                            unit.type = cpu_runtime_unit::kind::CUT_PART;
                            unit.node_index = i;
                            unit.cut_part = cut_units->parts[p];
                            unit.last_cut_part = p + 1 == cut_units->parts.size();
                            if (unit.cut_part.wbm_idx >= 0) {
                                unit.indices.push_back(unit.cut_part.wbm_idx);
                            }
                            runtime_units.push_back(std::move(unit));
                        }
                        build_range_begin = i + 1;
                        build_weights = 0;
                        build_roots.clear();
                        continue;
                    }

                    ggml_tensor * node = cgraph->nodes[i];
                    if (node) {
                        for (int j = 0; j < GGML_MAX_SRC; ++j) {
                            const ggml_tensor * root = nullptr;
                            elastic_buffer_ctx * bc = nullptr;
                            const elastic_tensor_units * units = nullptr;
                            if (!resolve_units(node->src[j], root, bc, units)) continue;
                            GGML_UNUSED(bc);
                            GGML_UNUSED(units);
                            if (build_roots.insert(root).second) build_weights++;
                        }
                    }
                    if (build_weights >= tensors_per_unit) {
                        append_range(build_range_begin, i + 1);
                        build_range_begin = i + 1;
                        build_weights = 0;
                        build_roots.clear();
                    }
                }
                append_range(build_range_begin, cgraph->n_nodes);
            }

            // The pre-provisioned representation always uses Cut-capable row
            // tiles when dynamic granularity is enabled.  Restore the logical
            // residency semantics here: tiles in a Tensor/Multi unit share an
            // eviction group, while distinct Cut parts remain independent.
            // This prevents coarse modes from silently receiving Cut's
            // half-tensor budget-fitting benefit.
            const uint64_t grouping_generation =
                llama_weight_unit_plan_generation();
            if (s->eviction_group_generation != grouping_generation) {
                elastic::wbm_clear_eviction_groups(&s->wbm);
                s->eviction_group_generation = grouping_generation;
            }
            // Prompt/decode may use different backend graphs under the same
            // plan generation. Idempotently register every graph's units;
            // otherwise only the first prompt graph receives coarse
            // Tensor/Multi residency semantics.
            for (const cpu_runtime_unit & unit : runtime_units) {
                const int fallback_idx = unit.indices.empty()
                    ? -1
                    : *std::min_element(
                        unit.indices.begin(), unit.indices.end());
                const int64_t group_id =
                    unit.plan_unit_id >= 0
                        ? static_cast<int64_t>(unit.plan_unit_id)
                        : ((INT64_C(1) << 50) +
                           static_cast<int64_t>(
                               std::max(0, fallback_idx)));
                for (int idx : unit.indices) {
                    const elastic::block_meta * block =
                        elastic::wbm_get(&s->wbm, idx);
                    // Permanently pinned metadata (norms, embedding, etc.)
                    // must not make the otherwise evictable unit immortal.
                    if (block && !block->is_pinned) {
                        elastic::wbm_set_eviction_group(
                            &s->wbm, idx, group_id);
                    }
                }
            }

            struct cpu_fused_pair_candidate {
                int first_node = -1;
                int second_node = -1;
                int first_idx = -1;
                int second_idx = -1;
                std::vector<int> first_indices;
                std::vector<int> second_indices;
                size_t first_unit = SIZE_MAX;
                size_t second_unit = SIZE_MAX;
            };
            std::vector<cpu_fused_pair_candidate> fused_pair_candidates;
            std::unordered_map<int, size_t> fused_candidate_by_first_node;

            const bool global_fused =
                s->granularity.mode ==
                elastic::granularity_mode::MULTI_FUSED;
            const bool planned_fused = std::any_of(
                runtime_units.begin(), runtime_units.end(),
                [](const cpu_runtime_unit & unit) {
                    return (unit.plan_flags &
                            (LLAMA_WEIGHT_UNIT_FUSE_LAYOUT |
                             LLAMA_WEIGHT_UNIT_FUSE_COMPUTE)) != 0;
                });
            const bool fused_graph_has_missing_weight = std::any_of(
                runtime_units.begin(), runtime_units.end(),
                [&](const cpu_runtime_unit & unit) {
                    return std::any_of(
                        unit.indices.begin(), unit.indices.end(),
                        [&](int idx) {
                            const elastic::block_meta * block =
                                elastic::wbm_get(&s->wbm, idx);
                            return block && !block->resident;
                        });
                });
            const bool fused_graph_has_batched_activation = std::any_of(
                cgraph->nodes, cgraph->nodes + cgraph->n_nodes,
                [](const ggml_tensor * node) {
                    return node && node->op == GGML_OP_MUL_MAT &&
                           node->src[1] && node->src[1]->ne[1] > 1;
                });
            // A fully resident single-token graph has neither layout work to
            // pair nor a compute shape that benefits from fusion. Avoid even
            // the candidate hash scan so multi_fused has the same decode fast
            // path as optimized Multi in that regime.
            if ((global_fused || planned_fused) &&
                (fused_graph_has_missing_weight ||
                 fused_graph_has_batched_activation)) {
                std::vector<size_t> node_to_unit(
                    static_cast<size_t>(cgraph->n_nodes), SIZE_MAX);
                for (size_t u = 0; u < runtime_units.size(); ++u) {
                    const cpu_runtime_unit & unit = runtime_units[u];
                    if (unit.type != cpu_runtime_unit::kind::GRAPH_RANGE) continue;
                    for (int node_index = unit.begin; node_index < unit.end; ++node_index) {
                        node_to_unit[static_cast<size_t>(node_index)] = u;
                    }
                }

                auto weight_indices_for_node =
                    [&](int node_index) -> std::vector<int> {
                    std::vector<int> result;
                    if (node_index < 0 ||
                        node_index >= cgraph->n_nodes) return result;
                    const ggml_tensor * node = cgraph->nodes[node_index];
                    if (!node || !node->src[0]) return result;
                    const ggml_tensor * root = nullptr;
                    elastic_buffer_ctx * bc = nullptr;
                    const elastic_tensor_units * units = nullptr;
                    if (!resolve_units(node->src[0], root, bc, units) ||
                        root != node->src[0] || !units) {
                        return result;
                    }
                    GGML_UNUSED(bc);
                    for (const auto & part : units->parts) {
                        if (part.wbm_idx >= 0) {
                            result.push_back(part.wbm_idx);
                        }
                    }
                    return result;
                };

                auto tensor_storage_overlaps = [](const ggml_tensor * a,
                                                  const ggml_tensor * b) {
                    if (!a || !b || !a->data || !b->data) return false;
                    const uintptr_t a_begin = reinterpret_cast<uintptr_t>(a->data);
                    const uintptr_t b_begin = reinterpret_cast<uintptr_t>(b->data);
                    const uintptr_t a_end = a_begin + ggml_nbytes(a);
                    const uintptr_t b_end = b_begin + ggml_nbytes(b);
                    return a_begin < b_end && b_begin < a_end;
                };

                // Pair consecutive compatible projections of the same
                // activation (for example Q/K). They may straddle ordinary
                // Multi unit boundaries because norm/scale weights also count
                // toward those boundaries. Do not change either boundary.
                std::unordered_map<const ggml_tensor *, int> pending_by_activation;
                for (int second = 0; second < cgraph->n_nodes; ++second) {
                    ggml_tensor * second_node = cgraph->nodes[second];
                    if (!second_node || second_node->op != GGML_OP_MUL_MAT ||
                        !second_node->src[1]) {
                        continue;
                    }
                    s->fused_scan_mul_mat++;
                    const ggml_tensor * activation = second_node->src[1];
                    auto pending = pending_by_activation.find(activation);
                    if (pending == pending_by_activation.end()) {
                        pending_by_activation.emplace(activation, second);
                        continue;
                    }
                    s->fused_scan_shared_activation++;
                    const int first = pending->second;
                    ggml_tensor * first_node = cgraph->nodes[first];
                    if (!ggml_backend_cpu_repack_mul_mat_pair_compatible(
                            first_node, second_node)) {
                        pending->second = second;
                        continue;
                    }
                    s->fused_scan_compatible++;

                    const std::vector<int> first_indices =
                        weight_indices_for_node(first);
                    const std::vector<int> second_indices =
                        weight_indices_for_node(second);
                    const size_t first_unit =
                        node_to_unit[static_cast<size_t>(first)];
                    const size_t second_unit =
                        node_to_unit[static_cast<size_t>(second)];
                    const bool same_planned_unit =
                        first_unit != SIZE_MAX &&
                        first_unit == second_unit;
                    if (!first_indices.empty() &&
                        first_indices.size() == second_indices.size() &&
                        first_unit != SIZE_MAX && second_unit != SIZE_MAX) {
                        s->fused_scan_weight_units++;
                    }
                    const uint32_t unit_flags =
                        same_planned_unit
                            ? runtime_units[first_unit].plan_flags
                            : 0;
                    const bool allow_layout =
                        global_fused ||
                        (unit_flags &
                         LLAMA_WEIGHT_UNIT_FUSE_LAYOUT) != 0;
                    const bool allow_compute =
                        global_fused ||
                        (unit_flags &
                         LLAMA_WEIGHT_UNIT_FUSE_COMPUTE) != 0;
                    bool safe_early_output =
                        !first_indices.empty() &&
                        first_indices.size() == second_indices.size() &&
                        first_indices.front() !=
                            second_indices.front() &&
                        first_unit != SIZE_MAX && second_unit != SIZE_MAX;
                    // The fused kernel produces the second projection at the
                    // first node. Reject a pair if the scheduler has reused
                    // the second output storage for an intervening node.
                    for (int middle = first + 1;
                         safe_early_output && middle < second; ++middle) {
                        if (tensor_storage_overlaps(
                                second_node, cgraph->nodes[middle])) {
                            safe_early_output = false;
                        }
                    }
                    if (safe_early_output) {
                        s->fused_scan_safe_output++;
                    }
                    if (!safe_early_output ||
                        (!allow_layout && !allow_compute)) {
                        pending->second = second;
                        continue;
                    }

                    if (allow_layout) {
                        for (size_t part = 0;
                             part < first_indices.size(); ++part) {
                            fused_layout_partner[
                                first_indices[part]] =
                                second_indices[part];
                            fused_layout_partner[
                                second_indices[part]] =
                                first_indices[part];
                        }
                    }
                    // For single-token decode, sharing activation
                    // quantization saves only one very small Q8 row while a
                    // CUSTOM node adds scheduling and synchronization cost.
                    // Keep the fused layout path, but fuse compute only for a
                    // multi-row activation (prompt/prefill), where that saved
                    // conversion is large enough to amortize the wrapper.
                    const bool compute_shape_profitable =
                        first_node->src[1] &&
                        first_node->src[1]->ne[1] > 1;
                    if (allow_compute && compute_shape_profitable) {
                        cpu_fused_pair_candidate candidate;
                        candidate.first_node = first;
                        candidate.second_node = second;
                        candidate.first_idx =
                            first_indices.front();
                        candidate.second_idx =
                            second_indices.front();
                        candidate.first_indices = first_indices;
                        candidate.second_indices = second_indices;
                        candidate.first_unit = first_unit;
                        candidate.second_unit = second_unit;
                        const size_t candidate_index =
                            fused_pair_candidates.size();
                        fused_pair_candidates.push_back(
                            std::move(candidate));
                        fused_candidate_by_first_node[first] =
                            candidate_index;
                        s->fused_kernel_pair_candidates++;
                    }
                    pending_by_activation.erase(pending);
                }
            }

            // llama's backend scheduler may invoke graph_compute many times
            // for one decode graph.  A unit pipeline that stops at each
            // backend-graph boundary penalizes a coarse first unit: it has no
            // predecessor inside the current call that can hide its LOAD and
            // PREPARE.  Learn the stable graph-to-graph transition online and
            // carry the next graph's prefix across that boundary.  The first
            // observation only records the transition; subsequent decode
            // iterations use exactly the same byte-bounded unit window.
            uint64_t graph_signature = 1469598103934665603ULL;
            auto signature_mix = [&](uint64_t value) {
                graph_signature ^= value;
                graph_signature *= 1099511628211ULL;
            };
            signature_mix(static_cast<uint64_t>(cgraph->n_nodes));
            signature_mix(static_cast<uint64_t>(runtime_units.size()));
            for (const cpu_runtime_unit & unit : runtime_units) {
                signature_mix(static_cast<uint64_t>(unit.type == cpu_runtime_unit::kind::CUT_PART));
                signature_mix(static_cast<uint64_t>(unit.begin + 1));
                signature_mix(static_cast<uint64_t>(unit.end + 1));
                signature_mix(static_cast<uint64_t>(unit.node_index + 2));
                if (unit.type == cpu_runtime_unit::kind::CUT_PART) {
                    signature_mix(static_cast<uint64_t>(unit.cut_part.row_start));
                    signature_mix(static_cast<uint64_t>(unit.cut_part.row_count));
                }
                signature_mix(static_cast<uint64_t>(unit.indices.size()));
                for (int idx : unit.indices) signature_mix(static_cast<uint64_t>(idx + 1));
            }
            if (s->unit_pipeline_previous_graph_valid && !runtime_units.empty()) {
                cpu_pipeline_successor successor;
                successor.signature = graph_signature;
                successor.units.reserve(runtime_units.size());
                for (const cpu_runtime_unit & unit : runtime_units) {
                    successor.units.push_back(unit.indices);
                }
                if (!fused_layout_partner.empty()) {
                    successor.fused_pairs.reserve(
                        fused_layout_partner.size() / 2);
                    for (const auto & pair : fused_layout_partner) {
                        if (pair.first < pair.second) {
                            successor.fused_pairs.emplace_back(
                                pair.first, pair.second);
                        }
                    }
                }
                s->unit_pipeline_successors[s->unit_pipeline_previous_graph_signature] =
                    std::move(successor);
            }
            s->unit_pipeline_previous_graph_signature = graph_signature;
            s->unit_pipeline_previous_graph_valid = true;

            // A future unit must remain protected after its asynchronous load
            // finishes.  Temporary protection only during pre-eviction lets
            // retire_unit() immediately evict the just-prefetched weight,
            // turning the apparent pipeline into extra reload churn.
            std::unordered_map<int, cpu_pipeline_pin_state> pipeline_pins;
            auto acquire_unit_pins = [&](const std::vector<int> & indices) {
                for (int idx : indices) {
                    const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                    if (!bm) continue;
                    auto & pin = pipeline_pins[idx];
                    if (pin.refs == 0) pin.original = bm->is_pinned;
                    pin.refs++;
                    elastic::wbm_set_pinned(&s->wbm, idx, true);
                }
            };
            auto release_unit_pins = [&](const std::vector<int> & indices) {
                for (int idx : indices) {
                    auto it = pipeline_pins.find(idx);
                    if (it == pipeline_pins.end() || it->second.refs == 0) continue;
                    if (--it->second.refs == 0) {
                        elastic::wbm_set_pinned(&s->wbm, idx, it->second.original);
                        pipeline_pins.erase(it);
                    }
                }
            };

            // Transfer a prefix prefetched by the preceding backend graph to
            // this call's ordinary per-unit pin accounting.  Keeping the
            // original pin bit prevents the carry from changing the model's
            // configured residency policy.
            bool incoming_cross_match =
                !s->unit_pipeline_cross_graphs.empty() &&
                s->unit_pipeline_cross_graphs.front().signature == graph_signature &&
                s->unit_pipeline_cross_graphs.front().units.size() <= runtime_units.size();
            if (incoming_cross_match) {
                const auto & incoming = s->unit_pipeline_cross_graphs.front();
                for (size_t i = 0; i < incoming.units.size(); ++i) {
                    if (incoming.units[i] != runtime_units[i].indices) {
                        incoming_cross_match = false;
                        break;
                    }
                }
            }
            if (incoming_cross_match) {
                const cpu_pipeline_successor incoming =
                    std::move(s->unit_pipeline_cross_graphs.front());
                s->unit_pipeline_cross_graphs.pop_front();
                std::unordered_map<int, size_t> transferred_refs;
                for (const std::vector<int> & indices : incoming.units) {
                    for (int idx : indices) transferred_refs[idx]++;
                }
                std::unordered_map<int, bool> local_original;
                for (const auto & transfer : transferred_refs) {
                    auto cross = s->unit_pipeline_cross_pins.find(transfer.first);
                    if (cross == s->unit_pipeline_cross_pins.end()) {
                        local_original[transfer.first] = false;
                        continue;
                    }
                    const size_t moved = std::min(cross->second.refs, transfer.second);
                    cross->second.refs -= moved;
                    if (cross->second.refs > 0) {
                        // A later prefetched graph still owns this pin.
                        local_original[transfer.first] = true;
                    } else {
                        local_original[transfer.first] = cross->second.original;
                        s->unit_pipeline_cross_pins.erase(cross);
                    }
                }
                bool all_ready = true;
                for (size_t i = 0; i < incoming.units.size(); ++i) {
                    cpu_runtime_unit & unit = runtime_units[i];
                    unit.pipeline_issued = true;
                    for (int idx : unit.indices) {
                        const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                        if (bm && !bm->resident) all_ready = false;
                        auto & pin = pipeline_pins[idx];
                        if (pin.refs == 0) {
                            auto original = local_original.find(idx);
                            pin.original = original != local_original.end()
                                ? original->second : false;
                        }
                        pin.refs++;
                        elastic::wbm_set_pinned(&s->wbm, idx, true);
                    }
                }
                if (all_ready) s->unit_pipeline_cross_ready++;
                else s->unit_pipeline_cross_waits++;
            } else {
                for (const auto & entry : s->unit_pipeline_cross_pins) {
                    elastic::wbm_set_pinned(&s->wbm, entry.first, entry.second.original);
                }
                s->unit_pipeline_cross_graphs.clear();
                s->unit_pipeline_cross_pins.clear();
            }

            auto stage_future_units = [&](size_t current) -> size_t {
                std::vector<int> protected_indices = runtime_units[current].indices;
                std::unordered_set<int> window_indices;
                std::vector<cpu_runtime_unit *> staged_units;
                size_t missing_bytes = 0;
                size_t window_bytes = 0;
                const size_t unit_limit = unit_pipeline_lookahead_bytes > 0
                    ? runtime_units.size()
                    : std::min(runtime_units.size(), current + 1 +
                        static_cast<size_t>(unit_pipeline_lookahead));
                for (size_t future = current + 1; future < unit_limit; ++future) {
                    cpu_runtime_unit & unit = runtime_units[future];
                    size_t added_window_bytes = 0;
                    for (int idx : unit.indices) {
                        if (window_indices.find(idx) != window_indices.end()) continue;
                        const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                        // The lookahead cap bounds additional pipeline working
                        // memory, not weights that are already resident.  Counting
                        // a resident/pinned embedding here creates a false
                        // head-of-line barrier (especially for a coarse Multi
                        // unit) even though staging that unit costs zero bytes.
                        if (bm && !bm->resident) added_window_bytes += bm->byte_size;
                    }
                    if (unit_pipeline_lookahead_bytes > 0 &&
                        window_bytes + added_window_bytes > unit_pipeline_lookahead_bytes) {
                        if (staged_units.empty()) s->unit_pipeline_oversize_windows++;
                        break;
                    }
                    if (!unit.pipeline_issued) {
                        unit.pipeline_issued = true;
                        s->unit_pipeline_issued++;
                        acquire_unit_pins(unit.indices);
                    }
                    staged_units.push_back(&unit);
                    // Always rescan issued units inside the active window.
                    // A bounded worker may have declined one block on an
                    // earlier visit; enqueue helpers deduplicate resident and
                    // in-flight blocks, while skipped blocks get another
                    // chance instead of falling through to synchronous I/O.
                    for (int idx : unit.indices) {
                        if (!window_indices.insert(idx).second) continue;
                        protected_indices.push_back(idx);
                        const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                        if (!bm) continue;
                        if (!bm->resident) {
                            window_bytes += bm->byte_size;
                            missing_bytes += bm->byte_size;
                        }
                    }
                }
                s->unit_pipeline_window_samples++;
                s->unit_pipeline_window_units_total += staged_units.size();
                s->unit_pipeline_window_bytes_total += window_bytes;
                s->unit_pipeline_window_units_max = std::max(
                    s->unit_pipeline_window_units_max, staged_units.size());
                s->unit_pipeline_window_bytes_max = std::max(
                    s->unit_pipeline_window_bytes_max, window_bytes);
                const size_t target = target_bytes();
                if (target != SIZE_MAX && missing_bytes > 0) {
                    const size_t reserve_target = target > missing_bytes ? target - missing_bytes : 0;
                    evict_to(reserve_target, protected_indices);
                }
                if (!fused_layout_partner.empty()) {
                    // Coalesce only pairs already selected by the ordinary
                    // byte-bounded lookahead. This changes neither the window
                    // nor residency pressure; enqueue deduplication makes the
                    // following per-unit submissions harmless.
                    for (const auto & pair : fused_layout_partner) {
                        if (pair.first >= pair.second ||
                            window_indices.find(pair.first) ==
                                window_indices.end() ||
                            window_indices.find(pair.second) ==
                                window_indices.end()) {
                            continue;
                        }
                        const elastic::block_meta * first_bm =
                            elastic::wbm_get(&s->wbm, pair.first);
                        const elastic::block_meta * second_bm =
                            elastic::wbm_get(&s->wbm, pair.second);
                        if (!first_bm || !second_bm ||
                            first_bm->resident || second_bm->resident) {
                            continue;
                        }
                        const std::vector<int> pair_indices{
                            pair.first, pair.second};
                        if (cpu_async_stage_load_enabled()) {
                            cpu_enqueue_async_load_unit(
                                s, pair_indices);
                        }
                        cpu_enqueue_async_prepare_unit(
                            s, pair_indices, true);
                    }
                }
                for (cpu_runtime_unit * unit : staged_units) {
                    // CPU_Elastic already owns a stable final destination for
                    // every weight.  When the separate LOAD stage is disabled,
                    // PREPARE reads directly into that destination and avoids
                    // an extra zero-filled host allocation plus staging memcpy.
                    // This is the normal unit-pipeline path; explicit staged
                    // plan experiments can still enable ASYNC_STAGE_LOAD.
                    if (cpu_async_stage_load_enabled()) {
                        cpu_enqueue_async_load_unit(s, unit->indices);
                    }
                    cpu_enqueue_async_prepare_unit(s, unit->indices);
                }
                return missing_bytes;
            };

            std::deque<cpu_pipeline_successor> pending_cross_graphs;
            std::unordered_map<int, size_t> pending_cross_pin_refs;
            auto stage_cross_graph_prefix = [&](size_t current, size_t reserved_missing_bytes) {
                if (!pending_cross_graphs.empty() ||
                    !s->unit_pipeline_cross_graphs.empty()) return;
                std::unordered_set<int> seen;
                std::unordered_set<uint64_t> seen_graphs;
                std::vector<int> protected_indices = runtime_units[current].indices;
                size_t missing_bytes = 0;
                size_t selected_units = 0;
                uint64_t cursor = graph_signature;
                bool window_full = false;
                // Bound graph metadata/pins as well as bytes. A short rolling
                // chain avoids pinning a large fraction of the model merely
                // because most of those future weights are already resident.
                for (size_t hop = 0;
                     hop < unit_pipeline_graph_lookahead && !window_full; ++hop) {
                    if (!seen_graphs.insert(cursor).second) break;
                    auto successor_it = s->unit_pipeline_successors.find(cursor);
                    if (successor_it == s->unit_pipeline_successors.end()) break;
                    const cpu_pipeline_successor & successor = successor_it->second;
                    cpu_pipeline_successor staged_graph;
                    staged_graph.signature = successor.signature;
                    std::unordered_set<int> staged_graph_indices;
                    for (const std::vector<int> & indices : successor.units) {
                        if (unit_pipeline_lookahead_bytes == 0 &&
                            selected_units >= static_cast<size_t>(unit_pipeline_lookahead)) {
                            window_full = true;
                            break;
                        }
                        size_t added_missing_bytes = 0;
                        for (int idx : indices) {
                            if (seen.find(idx) != seen.end()) continue;
                            const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                            const bool already_local = pipeline_pins.find(idx) != pipeline_pins.end();
                            if (bm && !bm->resident && !already_local) {
                                added_missing_bytes += bm->byte_size;
                            }
                        }
                        if (unit_pipeline_lookahead_bytes > 0 &&
                            reserved_missing_bytes + missing_bytes + added_missing_bytes >
                                unit_pipeline_lookahead_bytes) {
                            window_full = true;
                            break;
                        }
                        staged_graph.units.push_back(indices);
                        selected_units++;
                        for (int idx : indices) {
                            staged_graph_indices.insert(idx);
                            pending_cross_pin_refs[idx]++;
                            if (!seen.insert(idx).second) continue;
                            protected_indices.push_back(idx);
                            const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                            const bool already_local = pipeline_pins.find(idx) != pipeline_pins.end();
                            if (bm && !bm->resident && !already_local) {
                                missing_bytes += bm->byte_size;
                            }
                        }
                    }
                    for (const auto & pair : successor.fused_pairs) {
                        if (staged_graph_indices.find(pair.first) !=
                                staged_graph_indices.end() &&
                            staged_graph_indices.find(pair.second) !=
                                staged_graph_indices.end()) {
                            staged_graph.fused_pairs.push_back(pair);
                        }
                    }
                    if (!staged_graph.units.empty()) {
                        pending_cross_graphs.push_back(std::move(staged_graph));
                    }
                    if (window_full) break;
                    cursor = successor.signature;
                }
                if (pending_cross_graphs.empty()) return;
                // Keep the entire byte-bounded cross-graph chain in the same
                // reference-counted pin set as local future units.
                for (const auto & entry : pending_cross_pin_refs) {
                    const int idx = entry.first;
                    const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                    if (!bm) continue;
                    auto & pin = pipeline_pins[idx];
                    if (pin.refs == 0) pin.original = bm->is_pinned;
                    pin.refs += entry.second;
                    elastic::wbm_set_pinned(&s->wbm, idx, true);
                }
                const size_t target = target_bytes();
                if (target != SIZE_MAX && missing_bytes > 0) {
                    const size_t total_missing = reserved_missing_bytes + missing_bytes;
                    const size_t reserve_target = target > total_missing ? target - total_missing : 0;
                    evict_to(reserve_target, protected_indices);
                }
                for (const cpu_pipeline_successor & graph : pending_cross_graphs) {
                    for (const auto & fused_pair : graph.fused_pairs) {
                        const elastic::block_meta * first_bm =
                            elastic::wbm_get(&s->wbm, fused_pair.first);
                        const elastic::block_meta * second_bm =
                            elastic::wbm_get(&s->wbm, fused_pair.second);
                        if (!first_bm || !second_bm ||
                            first_bm->resident || second_bm->resident) {
                            continue;
                        }
                        const std::vector<int> pair{
                            fused_pair.first, fused_pair.second};
                        if (cpu_async_stage_load_enabled()) {
                            cpu_enqueue_async_load_unit(s, pair);
                        }
                        cpu_enqueue_async_prepare_unit(s, pair, true);
                    }
                    for (const std::vector<int> & indices : graph.units) {
                        if (cpu_async_stage_load_enabled()) {
                            cpu_enqueue_async_load_unit(s, indices);
                        }
                        cpu_enqueue_async_prepare_unit(s, indices);
                        s->unit_pipeline_issued++;
                        s->unit_pipeline_cross_issued++;
                    }
                }
                s->unit_pipeline_cross_bytes += missing_bytes;
            };

            ggml_status pipeline_status = GGML_STATUS_SUCCESS;
            std::unordered_set<int> fused_second_nodes_completed;
            for (size_t u = 0; u < runtime_units.size(); ++u) {
                cpu_runtime_unit & unit = runtime_units[u];
                size_t current_missing_tensors = 0;
                size_t current_missing_bytes = 0;
                for (int idx : unit.indices) {
                    const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                    if (!bm || bm->resident) continue;
                    current_missing_tensors++;
                    current_missing_bytes += bm->byte_size;
                }
                if (current_missing_tensors > 0) {
                    s->unit_pipeline_current_missing_units++;
                    s->unit_pipeline_current_missing_tensors += current_missing_tensors;
                    s->unit_pipeline_current_missing_bytes += current_missing_bytes;
                }
                const bool was_pipeline_issued = unit.pipeline_issued;
                if (!unit.pipeline_issued) acquire_unit_pins(unit.indices);
                const uint64_t residency_t0 = cpu_now_us();
                if (unit.pipeline_issued) {
                    bool ready = true;
                    for (int idx : unit.indices) {
                        const elastic::block_meta * bm = elastic::wbm_get(&s->wbm, idx);
                        if (bm && !bm->resident) {
                            ready = false;
                            break;
                        }
                    }
                    if (ready) {
                        s->unit_pipeline_ready++;
                    } else {
                        const uint64_t wait_t0 = cpu_now_us();
                        s->unit_pipeline_waits++;
                        make_unit_resident(unit.indices);
                        s->unit_pipeline_wait_us += cpu_now_us() - wait_t0;
                    }
                } else {
                    make_unit_resident(unit.indices);
                }
                const uint64_t residency_us = cpu_now_us() - residency_t0;
                s->unit_pipeline_current_residency_us += residency_us;
                if (!was_pipeline_issued && current_missing_tensors > 0) {
                    s->unit_pipeline_unissued_missing_units++;
                    s->unit_pipeline_unissued_missing_tensors += current_missing_tensors;
                    s->unit_pipeline_unissued_missing_bytes += current_missing_bytes;
                    s->unit_pipeline_unissued_missing_us += residency_us;
                }

                const uint64_t stage_t0 = cpu_now_us();
                const size_t local_missing_bytes = stage_future_units(u);
                stage_cross_graph_prefix(u, local_missing_bytes);
                s->unit_pipeline_stage_us += cpu_now_us() - stage_t0;

                if (unit.type == cpu_runtime_unit::kind::GRAPH_RANGE) {
                    ggml_cgraph view = ggml_graph_view(cgraph, unit.begin, unit.end);
                    std::vector<ggml_tensor *> fused_nodes;
                    ggml_tensor fused_node;
                    cpu_fused_mul_mat_pair_call fused_call;
                    const cpu_fused_pair_candidate * fused_candidate = nullptr;
                    if (!fused_pair_candidates.empty()) {
                        for (int node_index = unit.begin;
                             node_index < unit.end; ++node_index) {
                            auto found = fused_candidate_by_first_node.find(node_index);
                            if (found == fused_candidate_by_first_node.end()) continue;
                            const cpu_fused_pair_candidate & candidate =
                                fused_pair_candidates[found->second];
                            const bool all_first_resident = std::all_of(
                                candidate.first_indices.begin(),
                                candidate.first_indices.end(),
                                [&](int idx) {
                                    const elastic::block_meta * block =
                                        elastic::wbm_get(&s->wbm, idx);
                                    return block && block->resident;
                                });
                            const bool all_second_resident = std::all_of(
                                candidate.second_indices.begin(),
                                candidate.second_indices.end(),
                                [&](int idx) {
                                    const elastic::block_meta * block =
                                        elastic::wbm_get(&s->wbm, idx);
                                    return block && block->resident;
                                });
                            if (all_first_resident &&
                                all_second_resident) {
                                fused_candidate = &candidate;
                            } else {
                                // Opportunistic only: never wait for or widen
                                // the pipeline window to make fusion happen.
                                s->fused_kernel_pair_fallbacks++;
                            }
                            break;
                        }
                    }
                    const bool use_fused_kernel = fused_candidate != nullptr;
                    if (use_fused_kernel) {
                        fused_call.first =
                            cgraph->nodes[fused_candidate->first_node];
                        fused_call.second =
                            cgraph->nodes[fused_candidate->second_node];
                        fused_call.workspace_size =
                            ggml_backend_cpu_repack_mul_mat_pair_work_size(
                                fused_call.first, fused_call.second);
                        if (fused_call.workspace_size == 0) {
                            pipeline_status = GGML_STATUS_FAILED;
                        } else {
                            if (s->fused_compute_workspace.size() < fused_call.workspace_size) {
                                if (!s->fused_compute_workspace.allocate(
                                        fused_call.workspace_size, 0)) {
                                    pipeline_status = GGML_STATUS_ALLOC_FAILED;
                                } else {
                                    s->fused_workspace_allocations++;
                                }
                            } else {
                                s->fused_workspace_reuses++;
                            }
                        }
                        if (pipeline_status == GGML_STATUS_SUCCESS) {
                            fused_call.workspace = s->fused_compute_workspace.data();
                            fused_node = *fused_call.first;
                            fused_node.op = GGML_OP_CUSTOM;
                            ggml_custom_op_params params = {
                                cpu_fused_mul_mat_pair_callback,
                                GGML_N_TASKS_MAX,
                                &fused_call,
                            };
                            std::memset(fused_node.op_params, 0, sizeof(fused_node.op_params));
                            std::memcpy(fused_node.op_params, &params, sizeof(params));
                            fused_nodes.reserve(static_cast<size_t>(unit.end - unit.begin));
                            for (int node_index = unit.begin; node_index < unit.end; ++node_index) {
                                if (node_index == fused_candidate->first_node) {
                                    fused_nodes.push_back(&fused_node);
                                } else if (node_index != fused_candidate->second_node &&
                                           fused_second_nodes_completed.find(node_index) ==
                                               fused_second_nodes_completed.end()) {
                                    fused_nodes.push_back(cgraph->nodes[node_index]);
                                }
                            }
                            view.nodes = fused_nodes.data();
                            view.n_nodes = static_cast<int>(fused_nodes.size());
                        }
                    } else if (!fused_second_nodes_completed.empty()) {
                        fused_nodes.reserve(static_cast<size_t>(unit.end - unit.begin));
                        for (int node_index = unit.begin; node_index < unit.end; ++node_index) {
                            if (fused_second_nodes_completed.find(node_index) ==
                                fused_second_nodes_completed.end()) {
                                fused_nodes.push_back(cgraph->nodes[node_index]);
                            }
                        }
                        view.nodes = fused_nodes.data();
                        view.n_nodes = static_cast<int>(fused_nodes.size());
                    }
                    const auto compute_start = clk::now();
                    if (pipeline_status == GGML_STATUS_SUCCESS) {
                        pipeline_status = bctx->cpu->iface.graph_compute(bctx->cpu, &view);
                    }
                    const double ms = std::chrono::duration<double, std::milli>(
                        clk::now() - compute_start).count();
                    if (use_fused_kernel) {
                        s->fused_kernel_pair_calls++;
                        s->fused_kernel_pair_us += static_cast<uint64_t>(ms * 1000.0);
                        if (fused_call.error.load(std::memory_order_acquire) != 0) {
                            s->fused_kernel_pair_errors++;
                            pipeline_status = GGML_STATUS_FAILED;
                        } else if (pipeline_status == GGML_STATUS_SUCCESS) {
                            fused_second_nodes_completed.insert(
                                fused_candidate->second_node);
                        }
                    }
                    if (csv_profile) cpu_profile_compute_graph(
                        &view, ms, pipeline_status == GGML_STATUS_SUCCESS ? 1 : 0);
                    if (s->profile) {
                        s->profile_compute_total_ms += ms;
                        s->profile_compute_calls++;
                    }
                } else {
                    pipeline_status = compute_cut_part(
                        unit.node_index, unit.cut_part, false);
                    if (pipeline_status == GGML_STATUS_SUCCESS && unit.last_cut_part) {
                        s->granularity_cut_ops++;
                    }
                }
                release_unit_pins(unit.indices);
                if (u + 1 == runtime_units.size() && !pending_cross_graphs.empty()) {
                    // Drop only the local bookkeeping reference and keep the
                    // WBM pin set while the chain moves across graph calls.
                    for (const auto & transfer : pending_cross_pin_refs) {
                        auto pin = pipeline_pins.find(transfer.first);
                        if (pin == pipeline_pins.end() || pin->second.refs == 0) continue;
                        auto & cross = s->unit_pipeline_cross_pins[transfer.first];
                        if (cross.refs == 0) cross.original = pin->second.original;
                        const size_t moved = std::min(pin->second.refs, transfer.second);
                        pin->second.refs -= moved;
                        cross.refs += moved;
                        if (pin->second.refs == 0) pipeline_pins.erase(pin);
                        elastic::wbm_set_pinned(&s->wbm, transfer.first, true);
                    }
                    while (!pending_cross_graphs.empty()) {
                        s->unit_pipeline_cross_graphs.push_back(
                            std::move(pending_cross_graphs.front()));
                        pending_cross_graphs.pop_front();
                    }
                }
                const uint64_t retire_t0 = cpu_now_us();
                retire_unit(unit.indices);
                s->unit_pipeline_retire_us += cpu_now_us() - retire_t0;
                if (pipeline_status != GGML_STATUS_SUCCESS) break;
            }
            if (s->profile) s->profile_n_graph += 1;
            dbg_graph_n++;
            return pipeline_status;
        }

        int range_begin = 0;
        int weights_in_range = 0;
        std::unordered_set<const ggml_tensor *> roots_in_range;
        ggml_status status = GGML_STATUS_SUCCESS;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const elastic_tensor_units * cut_units = nullptr;
            if (cut_units_for_node(cgraph->nodes[i], cut_units)) {
                status = compute_graph_range(range_begin, i);
                if (status != GGML_STATUS_SUCCESS) break;
                status = compute_cut_node(i, *cut_units);
                if (status != GGML_STATUS_SUCCESS) break;
                range_begin = i + 1;
                weights_in_range = 0;
                roots_in_range.clear();
                continue;
            }

            ggml_tensor * node = cgraph->nodes[i];
            if (node) {
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    const ggml_tensor * root = nullptr;
                    elastic_buffer_ctx * bc = nullptr;
                    const elastic_tensor_units * units = nullptr;
                    if (!resolve_units(node->src[j], root, bc, units)) continue;
                    GGML_UNUSED(bc);
                    GGML_UNUSED(units);
                    if (roots_in_range.insert(root).second) weights_in_range++;
                }
            }
            if (weights_in_range >= tensors_per_unit) {
                status = compute_graph_range(range_begin, i + 1);
                if (status != GGML_STATUS_SUCCESS) break;
                range_begin = i + 1;
                weights_in_range = 0;
                roots_in_range.clear();
            }
        }
        if (status == GGML_STATUS_SUCCESS) status = compute_graph_range(range_begin, cgraph->n_nodes);
        if (s->profile) s->profile_n_graph += 1;
        dbg_graph_n++;
        return status;
    }

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
                ensure_block_resident_pipeline(s, idx, h);
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
                    ensure_block_resident_pipeline(s, idx, h);
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
                        ensure_block_resident_pipeline(s, idx, h);
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
                        ensure_block_resident_pipeline(s, idx, h);
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
                    llama_weight_runtime_mark_resident(name_for_idx(s, idx), LLAMA_WEIGHT_RUNTIME_CPU);
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
        double chunk_compute_total_ms = 0.0;
        uint64_t chunk_compute_calls = 0;
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
                const auto t_chunk_start = clk::now();
                st_chunk = bctx->cpu->iface.graph_compute(bctx->cpu, &chunk);
                const auto t_chunk_end = clk::now();
                const double chunk_ms = std::chrono::duration<double, std::milli>(t_chunk_end - t_chunk_start).count();
                chunk_compute_total_ms += chunk_ms;
                chunk_compute_calls++;
                if (csv_profile && s_chunk_size == 1) {
                    cpu_profile_compute_node(cgraph->nodes[i0], i0, chunk_ms, st_chunk == GGML_STATUS_SUCCESS ? 1 : 0);
                }

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
                const auto t_chunk_start = clk::now();
                st_chunk = bctx->cpu->iface.graph_compute(bctx->cpu, &chunk);
                const auto t_chunk_end = clk::now();
                const double chunk_ms = std::chrono::duration<double, std::milli>(t_chunk_end - t_chunk_start).count();
                chunk_compute_total_ms += chunk_ms;
                chunk_compute_calls++;
                if (csv_profile && s_chunk_size == 1) {
                    cpu_profile_compute_node(cgraph->nodes[i0], i0, chunk_ms, st_chunk == GGML_STATUS_SUCCESS ? 1 : 0);
                }
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
            s->profile_compute_total_ms += chunk_compute_total_ms;
            s->profile_compute_calls += chunk_compute_calls;
        }
        if (csv_profile) {
            cpu_profile_compute_graph(cgraph, chunk_compute_total_ms, st_chunk == GGML_STATUS_SUCCESS ? 1 : 0);
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
    if (s->profile || csv_profile) {
        auto t_io_end = clk::now();
        if (s->profile) {
            s->profile_io_total_ms += std::chrono::duration<double, std::milli>(t_io_end - t_io_start).count();
        }
        t_compute_start = t_io_end;
    }
    ggml_status st = bctx->cpu->iface.graph_compute(bctx->cpu, cgraph);
    double compute_ms = 0.0;
    if (s->profile) {
        auto t_compute_end = clk::now();
        compute_ms = std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
        s->profile_compute_total_ms += compute_ms;
        s->profile_compute_calls++;
    } else if (csv_profile) {
        auto t_compute_end = clk::now();
        compute_ms = std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
    }
    if (csv_profile) {
        cpu_profile_compute_graph(cgraph, compute_ms, st == GGML_STATUS_SUCCESS ? 1 : 0);
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

// CPU_Elastic delegates every compute unit to a real CPU backend.  Forward
// runtime controls to that delegate so fine-grained Cut units reuse llama's
// persistent threadpool (including its thread count and affinity) instead of
// constructing a disposable default threadpool for every sub-graph.
void elastic_backend_set_n_threads(ggml_backend_t backend, int n_threads) {
    auto * bctx = static_cast<elastic_backend_ctx *>(backend->context);
    ggml_backend_cpu_set_n_threads(bctx->cpu, n_threads);
}

void elastic_backend_set_threadpool(ggml_backend_t backend, ggml_threadpool_t threadpool) {
    auto * bctx = static_cast<elastic_backend_ctx *>(backend->context);
    ggml_backend_cpu_set_threadpool(bctx->cpu, threadpool);
}

void elastic_backend_set_abort_callback(
        ggml_backend_t backend,
        ggml_abort_callback abort_callback,
        void * abort_callback_data) {
    auto * bctx = static_cast<elastic_backend_ctx *>(backend->context);
    ggml_backend_cpu_set_abort_callback(
        bctx->cpu, abort_callback, abort_callback_data);
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
    // Backend-instance controls need wrappers because CPU_Elastic itself is
    // not a standard CPU backend.  The wrappers above apply them to the real
    // CPU delegate held by elastic_backend_ctx.
    if (!name) return nullptr;
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fn = elastic_backend_set_n_threads;
        return reinterpret_cast<void *>(fn);
    }
    if (std::strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        ggml_backend_set_abort_callback_t fn = elastic_backend_set_abort_callback;
        return reinterpret_cast<void *>(fn);
    }
    if (std::strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return reinterpret_cast<void *>(elastic_backend_set_threadpool);
    }
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

bool ggml_backend_cpu_elastic_get_granularity_state(
        ggml_backend_t backend,
        ggml_backend_elastic_granularity_state * out) {
    if (!out || !ggml_backend_is_cpu_elastic(backend)) return false;
    const elastic_state * state = get_state();
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
    out->prepare_us = state->stage_xform_us;
    out->direct_read_calls = state->direct_read_calls;
    out->direct_read_us = state->direct_read_us;
    out->direct_read_bytes = state->direct_read_bytes;
    out->load_calls = state->stage_load_calls;
    out->load_us = state->stage_load_us;
    out->load_bytes = state->stage_load_bytes;
    out->prepare_calls = state->stage_xform_calls;
    out->prepare_bytes = state->stage_xform_bytes;
    out->compute_calls = state->profile_compute_calls;
    out->compute_us = static_cast<uint64_t>(
        std::max(0.0, state->profile_compute_total_ms) * 1000.0 + 0.5);
    out->compute_timing_available = state->profile ? 1 : 0;
    out->pipeline_residency_us =
        state->unit_pipeline_current_residency_us;
    out->pipeline_unissued_us =
        state->unit_pipeline_unissued_missing_us;
    out->pipeline_stage_us = state->unit_pipeline_stage_us;
    out->pipeline_retire_us = state->unit_pipeline_retire_us;
    out->evict_us = state->evict_wall_us;
    out->resident_bytes = state->wbm.resident_bytes;
    return true;
}

bool ggml_backend_cpu_elastic_synchronize_pipeline(ggml_backend_t backend) {
    if (!ggml_backend_is_cpu_elastic(backend)) return false;
    elastic_state * state = get_state();
    if (!state || !state->wbm_inited) return false;
    cpu_wait_async_workers_idle(state);
    return true;
}

ggml_backend_reg_t ggml_backend_cpu_elastic_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ elastic_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}
