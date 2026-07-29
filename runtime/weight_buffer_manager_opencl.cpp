// runtime/weight_buffer_manager_opencl.cpp —— 见 weight_buffer_manager_opencl.h

#include "weight_buffer_manager_opencl.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace elastic {

namespace {

uint64_t now_us() {
    using clock = std::chrono::steady_clock;
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now().time_since_epoch()).count();
}

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

const char *device_event_kind_name(wbmcl_device_event_kind kind) {
    switch (kind) {
        case wbmcl_device_event_kind::TRANSFER_WRITE:  return "transfer_write";
        case wbmcl_device_event_kind::XFORM_CONVERT:   return "xform_convert";
        case wbmcl_device_event_kind::XFORM_TRANSPOSE: return "xform_transpose";
        case wbmcl_device_event_kind::XFORM_COPY:      return "xform_copy";
        case wbmcl_device_event_kind::COMPUTE_WAIT:    return "compute_wait";
    }
    return "unknown";
}

const char *stage_detail_kind_name(wbmcl_stage_detail_kind kind) {
    switch (kind) {
        case wbmcl_stage_detail_kind::SOA_POOL_LOOKUP:    return "soa_pool_lookup";
        case wbmcl_stage_detail_kind::PARENT_ALLOC:       return "parent_alloc";
        case wbmcl_stage_detail_kind::STAGING_ALLOC:      return "staging_alloc";
        case wbmcl_stage_detail_kind::HOST_SRC:           return "host_src";
        case wbmcl_stage_detail_kind::WRITE_ENQUEUE:      return "write_enqueue";
        case wbmcl_stage_detail_kind::SUBBUFFER:          return "subbuffer";
        case wbmcl_stage_detail_kind::BARRIER_ENQUEUE:    return "barrier_enqueue";
        case wbmcl_stage_detail_kind::CONVERT_ENQUEUE:    return "convert_enqueue";
        case wbmcl_stage_detail_kind::TRANSPOSE_ENQUEUE:  return "transpose_enqueue";
        case wbmcl_stage_detail_kind::FINAL_WAIT_ENQUEUE: return "final_wait_enqueue";
        case wbmcl_stage_detail_kind::MARK_RESIDENT:      return "mark_resident";
        case wbmcl_stage_detail_kind::CPU_XFORM:          return "cpu_xform";
    }
    return "unknown";
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


bool async_load_on_evict_enabled() {
    static const bool enabled = []() {
        const char *e = std::getenv("GGML_ELASTIC_ASYNC_LOAD_ON_EVICT");
        return e && *e && *e != '0';
    }();
    return enabled;
}

bool async_stage_load_legacy_unbounded_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("GGML_ELASTIC_ASYNC_LOAD_UNSAFE");
        return e && *e && *e != '0';
    }();
    return enabled;
}

bool async_stage_load_bounded_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("GGML_ELASTIC_ASYNC_LOAD_BOUNDED");
        return e && *e && *e != '0';
    }();
    return enabled;
}


bool release_stage_after_xform_enabled() {
    static const bool enabled = []() {
        const char *e = std::getenv("GGML_ELASTIC_RELEASE_STAGE_AFTER_XFORM");
        if (e && *e) return *e != '0';
        return true;
    }();
    return enabled;
}

bool soa_reload_on_transfer_enabled() {
    static const bool enabled = []() {
        const char *e = std::getenv("GGML_ELASTIC_SOA_RELOAD_ON_TRANSFER");
        return e && *e && *e != '0';
    }();
    return enabled;
}

bool soa_async_reload_on_transfer_enabled() {
    static const bool enabled = []() {
        const char *e = std::getenv("GGML_ELASTIC_SOA_ASYNC_RELOAD_ON_TRANSFER");
        return e && *e && *e != '0';
    }();
    return enabled;
}

bool async_stage_prepare_enabled() {
    static const bool enabled = []() {
        const char *e = std::getenv("GGML_ELASTIC_ASYNC_STAGE_PREPARE");
        if (e && *e) return *e != '0';
        e = std::getenv("GGML_ELASTIC_SOA_ASYNC_RELOAD_ON_TRANSFER");
        return e && *e && *e != '0';
    }();
    return enabled;
}

size_t async_stage_prepare_max_pending() {
    static const size_t max_pending = []() {
        const char *e = std::getenv("GGML_ELASTIC_ASYNC_PREPARE_MAX_PENDING");
        long long v = e && *e ? atoll(e) : 0;
        if (v < 0) v = 0;
        return static_cast<size_t>(v);
    }();
    return max_pending;
}

size_t async_stage_load_max_pending_bytes() {
    static const size_t max_pending_bytes = []() {
        const char * e = std::getenv("GGML_ELASTIC_ASYNC_LOAD_MAX_PENDING_MB");
        long long mib = e && *e ? atoll(e) : 128;
        if (mib < 0) mib = 0;
        return static_cast<size_t>(mib) * 1024ull * 1024ull;
    }();
    return max_pending_bytes;
}

size_t async_stage_load_max_pending_count() {
    static const size_t max_pending_count = []() {
        const char * e = std::getenv("GGML_ELASTIC_ASYNC_LOAD_MAX_PENDING_COUNT");
        long long v = e && *e ? atoll(e) : 2;
        if (v < 0) v = 0;
        return static_cast<size_t>(v);
    }();
    return max_pending_count;
}

size_t async_stage_load_max_staged_bytes() {
    static const size_t max_staged_bytes = []() -> size_t {
        const char * e = std::getenv("GGML_ELASTIC_ASYNC_LOAD_MAX_STAGED_MB");
        if (e && *e) {
            long long mib = atoll(e);
            if (mib < 0) mib = 0;
            return static_cast<size_t>(mib) * 1024ull * 1024ull;
        }
        return async_stage_load_max_pending_bytes();
    }();
    return max_staged_bytes;
}

size_t host_staging_active_bytes_locked(wbm_opencl_ctx * octx) {
    if (!octx) return 0;
    size_t bytes = 0;
    for (const auto & kv : octx->host_staging_by_idx) {
        bytes += kv.second.size();
    }
    return bytes;
}

void release_host_staging(wbm_opencl_ctx *octx, int idx) {
    if (!octx) return;
    bool released = false;
    {
        std::lock_guard<std::mutex> lock(octx->host_staging_mtx);
        auto it = octx->host_staging_by_idx.find(idx);
        if (it == octx->host_staging_by_idx.end()) return;
        host_staging_pool_put(octx, std::move(it->second));
        octx->host_staging_by_idx.erase(it);
        released = true;
    }
    if (octx->async_stage_load) {
        std::lock_guard<std::mutex> lock(octx->async_load_mtx);
        octx->async_load_state.erase(idx);
        octx->async_load_bytes_by_idx.erase(idx);
    }
    if (released) octx->async_load_cv.notify_all();
}

cl_mem retain_pool_take_buffer(wbm_opencl_ctx *octx, size_t nbytes) {
    if (!octx || !octx->retain_cl_mem) return nullptr;
    std::lock_guard<std::mutex> lock(octx->soa_pool_mtx);
    auto it = octx->retained_buffers_by_size.find(nbytes);
    if (it == octx->retained_buffers_by_size.end() || it->second.empty()) {
        return nullptr;
    }
    cl_mem cached = static_cast<cl_mem>(it->second.back());
    it->second.pop_back();
    for (auto lit = octx->retain_order_sizes.rbegin();
         lit != octx->retain_order_sizes.rend(); ++lit) {
        if (*lit == nbytes) {
            octx->retain_order_sizes.erase(std::next(lit).base());
            break;
        }
    }
    octx->cached_bytes -= std::min(octx->cached_bytes, nbytes);
    return cached;
}

void retain_pool_put_buffer(
        wbm_opencl_ctx *octx, cl_mem buffer, size_t nbytes) {
    if (!octx || !buffer || !octx->retain_cl_mem) return;
    std::lock_guard<std::mutex> lock(octx->soa_pool_mtx);
    wbmcl_retain_pool_trim_locked(octx, nbytes);
    octx->retained_buffers_by_size[nbytes].push_back(
        static_cast<void *>(buffer));
    octx->retain_order_sizes.push_back(nbytes);
    octx->cached_bytes += nbytes;
}

}  // namespace

void wbmcl_retain_pool_trim_locked(
        wbm_opencl_ctx *octx, size_t incoming_bytes) {
    if (!octx || octx->cache_byte_limit == 0) return;
    while (octx->cached_bytes + incoming_bytes > octx->cache_byte_limit &&
           !octx->retain_order_sizes.empty()) {
        const size_t old_sz = octx->retain_order_sizes.front();
        octx->retain_order_sizes.pop_front();

        auto soa = octx->soa_pool_by_size.find(old_sz);
        if (soa != octx->soa_pool_by_size.end() && !soa->second.empty()) {
            soa_pool_entry entry = soa->second.back();
            soa->second.pop_back();
            if (entry.ready_event) {
                cl_event ready = static_cast<cl_event>(entry.ready_event);
                clWaitForEvents(1, &ready);
                clReleaseEvent(ready);
            }
            if (entry.q) {
                clReleaseMemObject(static_cast<cl_mem>(entry.q));
                octx->n_releases += 1;
            }
            if (entry.d) {
                clReleaseMemObject(static_cast<cl_mem>(entry.d));
                octx->n_releases += 1;
            }
            if (entry.parent) {
                octx->soa_pooled_parents.erase(entry.parent);
                clReleaseMemObject(static_cast<cl_mem>(entry.parent));
                octx->n_releases += 1;
            }
            octx->cached_bytes -= std::min(octx->cached_bytes, old_sz);
            continue;
        }

        auto ordinary = octx->retained_buffers_by_size.find(old_sz);
        if (ordinary == octx->retained_buffers_by_size.end() ||
            ordinary->second.empty()) {
            continue;
        }
        cl_mem old_buffer =
            static_cast<cl_mem>(ordinary->second.back());
        ordinary->second.pop_back();
        clReleaseMemObject(old_buffer);
        octx->n_releases += 1;
        octx->cached_bytes -= std::min(octx->cached_bytes, old_sz);
    }
}

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
    octx->soa_staging = nullptr;
    octx->soa_staging_capacity = 0;
    octx->soa_staging_last_use_ev = nullptr;
    octx->soa_pooled_parents.clear();
    octx->soa_double_evict = 0;
    octx->soa_staging_slots.clear();
    octx->soa_staging_slot_capacity.clear();
    octx->soa_staging_slot_last_use_ev.clear();
    octx->soa_staging_next_slot = 0;
    octx->soa_staging_current_slot = 0;
    octx->retain_host_staging  = false;
    octx->host_staging_pool_limit = 0;
    octx->host_staging_pool_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(octx->host_staging_mtx);
        octx->host_staging_by_idx.clear();
        octx->host_staging_pool_by_size.clear();
        octx->host_staging_pool_order.clear();
    }
    octx->bytes_loaded_total   = 0;
    octx->async_stage_load = []() {
        const char * e = std::getenv("GGML_ELASTIC_ASYNC_STAGE_LOAD");
        return e && *e && *e != '0';
    }();
    if (octx->async_stage_load &&
        !async_stage_load_bounded_enabled() &&
        !async_stage_load_legacy_unbounded_enabled()) {
        std::fprintf(stderr,
                     "ggml_opencl elastic: async stage LOAD disabled by fail-safe "
                     "(set GGML_ELASTIC_ASYNC_LOAD_BOUNDED=1 for bounded experimental path)\n");
        octx->async_stage_load = false;
    } else if (octx->async_stage_load && async_stage_load_bounded_enabled() &&
               !async_stage_load_legacy_unbounded_enabled()) {
        std::fprintf(stderr,
                     "ggml_opencl elastic: async stage LOAD bounded "
                     "(pending_cap=%zuMiB pending_count=%zu staged_cap=%zuMiB; "
                     "set GGML_ELASTIC_ASYNC_LOAD_UNSAFE=1 for legacy unbounded path)\n",
                     async_stage_load_max_pending_bytes() / 1024 / 1024,
                     async_stage_load_max_pending_count(),
                     async_stage_load_max_staged_bytes() / 1024 / 1024);
    }
    octx->async_load_shutdown = false;
    octx->async_load_queue.clear();
    octx->async_load_state.clear();
    octx->async_load_bytes_by_idx.clear();
    octx->async_load_enqueued = 0;
    octx->async_load_completed = 0;
    octx->async_load_units_enqueued = 0;
    octx->async_load_units_completed = 0;
    octx->async_load_waits = 0;
    octx->async_load_wait_us = 0;
    octx->async_load_pending_bytes = 0;
    octx->async_load_max_pending_bytes_seen = 0;
    octx->async_load_max_pending_count_seen = 0;
    octx->async_load_throttle_waits = 0;
    octx->async_load_throttle_wait_us = 0;
    octx->async_load_throttle_skipped = 0;
    octx->async_soa_reload_worker_started = false;
    octx->async_soa_reload_shutdown = false;
    octx->async_soa_reload_queue.clear();
    octx->async_soa_reload_state.clear();
    octx->async_soa_reload_enqueued = 0;
    octx->async_soa_reload_completed = 0;
    octx->async_soa_reload_units_enqueued = 0;
    octx->async_soa_reload_units_completed = 0;
    octx->async_soa_reload_waits = 0;
    octx->async_soa_reload_wait_us = 0;
    octx->async_load_direct_read_calls = 0;
    octx->async_load_direct_read_us = 0;
    octx->async_load_direct_read_bytes = 0;
    octx->foreground_direct_read_calls = 0;
    octx->foreground_direct_read_us = 0;
    octx->foreground_direct_read_bytes = 0;
    octx->stage_load_calls = 0;
    octx->stage_load_ok = 0;
    octx->stage_load_us = 0;
    octx->stage_load_bytes = 0;
    octx->stage_transfer_calls = 0;
    octx->stage_transfer_ok = 0;
    octx->stage_transfer_us = 0;
    octx->stage_transfer_bytes = 0;
    octx->stage_xform_calls = 0;
    octx->stage_xform_ok = 0;
    octx->stage_xform_us = 0;
    octx->stage_xform_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(octx->soa_prepare_stats_mtx);
        octx->soa_prepare_calls = 0;
        octx->soa_prepare_ok = 0;
        octx->soa_prepare_us = 0;
        octx->soa_prepare_bytes = 0;
    }
    octx->soa_pool_hit = 0;
    octx->soa_pool_miss = 0;
    octx->parent_pool_hit = 0;
    octx->parent_pool_miss = 0;
    octx->parent_create_calls = 0;
    octx->parent_create_us = 0;
    octx->parent_create_bytes = 0;
    octx->device_timing = false;
    octx->device_events.clear();
    octx->stage_detail = false;
    for (auto & b : octx->stage_detail_buckets) b = {};

    if (octx->async_stage_load) {
        octx->async_load_worker = std::thread([octx]() {
            for (;;) {
                std::vector<int> task;
                {
                    std::unique_lock<std::mutex> lock(octx->async_load_mtx);
                    octx->async_load_cv.wait(lock, [octx]() {
                        return octx->async_load_shutdown || !octx->async_load_queue.empty();
                    });
                    if (octx->async_load_shutdown && octx->async_load_queue.empty()) break;
                    task = std::move(octx->async_load_queue.front());
                    octx->async_load_queue.pop_front();
                }
                for (int idx : task) {
                    const int rc = wbmcl_load_host(octx, idx);
                    {
                        std::lock_guard<std::mutex> lock(octx->async_load_mtx);
                        auto bit = octx->async_load_bytes_by_idx.find(idx);
                        if (bit != octx->async_load_bytes_by_idx.end()) {
                            octx->async_load_pending_bytes -=
                                std::min(octx->async_load_pending_bytes, bit->second);
                            octx->async_load_bytes_by_idx.erase(bit);
                        }
                        octx->async_load_state[idx] = rc == 0 ? 2 : rc;
                        octx->async_load_completed++;
                    }
                    // Publish each tensor as soon as it is available so the
                    // PREPARE worker may overlap tensor i with LOAD of i+1.
                    octx->async_load_cv.notify_all();
                }
                {
                    std::lock_guard<std::mutex> lock(octx->async_load_mtx);
                    octx->async_load_units_completed++;
                }
                octx->async_load_cv.notify_all();
            }
        });
    }
    return 0;
}

void wbmcl_record_device_event(wbm_opencl_ctx *octx,
                               cl_event ev,
                               wbmcl_device_event_kind kind,
                               size_t bytes) {
    if (!octx || !octx->device_timing || !ev) return;
    if (clRetainEvent(ev) != CL_SUCCESS) return;
    std::lock_guard<std::mutex> lock(octx->device_timing_mtx);
    octx->device_events.push_back({ev, kind, bytes});
}

void wbmcl_dump_device_timing(wbm_opencl_ctx *octx, FILE *out) {
    if (!octx || !octx->device_timing || !out) return;
    std::vector<wbmcl_device_event_sample> events;
    {
        std::lock_guard<std::mutex> lock(octx->device_timing_mtx);
        events.swap(octx->device_events);
    }

    struct bucket {
        uint64_t n = 0;
        uint64_t ok = 0;
        uint64_t ns = 0;
        size_t bytes = 0;
    };
    bucket buckets[5];

    for (auto & sample : events) {
        cl_event ev = sample.event;
        const int idx = (int) sample.kind;
        if (idx < 0 || idx >= 5) {
            clReleaseEvent(ev);
            continue;
        }
        buckets[idx].n++;
        buckets[idx].bytes += sample.bytes;

        cl_int werr = clWaitForEvents(1, &ev);
        if (werr == CL_SUCCESS) {
            cl_ulong start = 0;
            cl_ulong end = 0;
            cl_int serr = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
            cl_int eerr = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
            // Some mobile OpenCL drivers return bogus profiling timestamps for
            // barrier/marker commands. Stage events in this path should be well
            // below one second; discard outliers instead of polluting the dump.
            const uint64_t dur = (end >= start) ? (uint64_t) (end - start) : UINT64_MAX;
            if (serr == CL_SUCCESS && eerr == CL_SUCCESS && dur < 1000ull * 1000ull * 1000ull) {
                buckets[idx].ok++;
                buckets[idx].ns += dur;
            }
        }
        clReleaseEvent(ev);
    }

    std::fprintf(out, "\n=== wbmcl device timing dump (event profiling) ===\n");
    for (int i = 0; i < 5; ++i) {
        const auto & b = buckets[i];
        const double ms = b.ns / 1000000.0;
        const double mb = b.bytes / 1024.0 / 1024.0;
        const double avg = b.ok ? ms / b.ok : 0.0;
        const double mbps = ms > 0 ? mb / (ms / 1000.0) : 0.0;
        std::fprintf(out,
                     "device %-15s events=%llu ok=%llu total=%.3f ms avg=%.3f ms MB=%.1f MB/s=%.1f\n",
                     device_event_kind_name((wbmcl_device_event_kind) i),
                     (unsigned long long) b.n,
                     (unsigned long long) b.ok,
                     ms, avg, mb, mbps);
    }
    std::fprintf(out, "==================================================\n");
}

void wbmcl_record_stage_detail(wbm_opencl_ctx *octx,
                               wbmcl_stage_detail_kind kind,
                               uint64_t us,
                               size_t bytes) {
    if (!octx || !octx->stage_detail) return;
    const int idx = (int) kind;
    if (idx < 0 || idx >= 12) return;
    std::lock_guard<std::mutex> lock(octx->stage_detail_mtx);
    auto & b = octx->stage_detail_buckets[idx];
    b.n++;
    b.us += us;
    b.bytes += bytes;
}

void wbmcl_dump_stage_detail(wbm_opencl_ctx *octx, FILE *out) {
    if (!octx || !octx->stage_detail || !out) return;
    wbmcl_stage_detail_bucket buckets[12];
    {
        std::lock_guard<std::mutex> lock(octx->stage_detail_mtx);
        for (int i = 0; i < 12; ++i) buckets[i] = octx->stage_detail_buckets[i];
    }
    std::fprintf(out, "\n=== wbmcl stage detail timing dump (host substage) ===\n");
    for (int i = 0; i < 12; ++i) {
        const auto & b = buckets[i];
        const double ms = b.us / 1000.0;
        const double mb = b.bytes / 1024.0 / 1024.0;
        const double avg = b.n ? ms / b.n : 0.0;
        std::fprintf(out, "detail %-18s calls=%llu total=%.3f ms avg=%.3f ms MB=%.1f\n",
                     stage_detail_kind_name((wbmcl_stage_detail_kind) i),
                     (unsigned long long) b.n, ms, avg, mb);
    }
    if (octx->async_stage_load) {
        const double wait_ms = octx->async_load_wait_us / 1000.0;
        const double avg_wait = octx->async_load_waits ? wait_ms / octx->async_load_waits : 0.0;
        const double throttle_ms = octx->async_load_throttle_wait_us / 1000.0;
        const double throttle_avg = octx->async_load_throttle_waits ? throttle_ms / octx->async_load_throttle_waits : 0.0;
        std::fprintf(out,
                     "async_load   : units=%llu/%llu tensors=%llu/%llu waits=%llu wait_total=%.3f ms wait_avg=%.3f ms "
                     "throttle_waits=%llu throttle_total=%.3f ms throttle_avg=%.3f ms throttle_skipped=%llu "
                     "max_pending=%zu MB max_pending_count=%zu\n",
                     (unsigned long long) octx->async_load_units_enqueued,
                     (unsigned long long) octx->async_load_units_completed,
                     (unsigned long long) octx->async_load_enqueued,
                     (unsigned long long) octx->async_load_completed,
                     (unsigned long long) octx->async_load_waits,
                     wait_ms, avg_wait,
                     (unsigned long long) octx->async_load_throttle_waits,
                     throttle_ms, throttle_avg,
                     (unsigned long long) octx->async_load_throttle_skipped,
                     octx->async_load_max_pending_bytes_seen / 1024 / 1024,
                     octx->async_load_max_pending_count_seen);
    }
    if (async_stage_prepare_enabled() || octx->async_soa_reload_worker_started ||
        octx->async_soa_reload_enqueued > 0 || octx->async_soa_reload_completed > 0) {
        const double wait_ms = octx->async_soa_reload_wait_us / 1000.0;
        const double avg_wait = octx->async_soa_reload_waits ? wait_ms / octx->async_soa_reload_waits : 0.0;
        std::fprintf(out,
                     "async_prepare: units=%llu/%llu tensors=%llu/%llu waits=%llu "
                     "wait_total=%.3f ms wait_avg=%.3f ms\n",
                     (unsigned long long) octx->async_soa_reload_units_enqueued,
                     (unsigned long long) octx->async_soa_reload_units_completed,
                     (unsigned long long) octx->async_soa_reload_enqueued,
                     (unsigned long long) octx->async_soa_reload_completed,
                     (unsigned long long) octx->async_soa_reload_waits,
                     wait_ms, avg_wait);
    }
    std::fprintf(out, "======================================================\n");
}

void wbmcl_get_soa_prepare_state(wbm_opencl_ctx *octx,
                                 uint64_t *calls,
                                 uint64_t *ok,
                                 uint64_t *us,
                                 size_t *bytes) {
    if (calls) *calls = 0;
    if (ok) *ok = 0;
    if (us) *us = 0;
    if (bytes) *bytes = 0;
    if (!octx) return;
    std::lock_guard<std::mutex> lock(octx->soa_prepare_stats_mtx);
    if (calls) *calls = octx->soa_prepare_calls;
    if (ok) *ok = octx->soa_prepare_ok;
    if (us) *us = octx->soa_prepare_us;
    if (bytes) *bytes = octx->soa_prepare_bytes;
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
        if (!octx->async_stage_load || release_stage_after_xform_enabled()) release_host_staging(octx, idx);
        return 0;
    }

    if (!meta->host_ptr || meta->byte_size == 0) {
        std::fprintf(stderr, "[wbmcl] block %d 未注册或 byte_size=0\n", idx);
        return -3;
    }

    // If a plan LOAD stage is in flight for this weight, wait here only because
    // compute is about to need the tensor. This is the consumer-side dependency
    // that turns anchor LOAD into real producer/consumer overlap.
    wbmcl_wait_host_load(octx, idx);

    // Transfer 源解析: LOAD stage 已执行时优先用 host staging；否则默认 mmap host_ptr
    // (隐式 page fault 读盘)。 若注入了
    // direct_read_fn (GGML_ELASTIC_DIRECT_IO=1), 先 O_DIRECT pread 到 thread_local
    // scratch (绕 page cache, 模拟真 disk 成本), 再用 scratch 做 transfer 源。
    const void *dma_src = meta->host_ptr;
    bool use_direct_scratch = false;
    {
        std::lock_guard<std::mutex> lock(octx->host_staging_mtx);
        auto staged = octx->host_staging_by_idx.find(idx);
        if (staged != octx->host_staging_by_idx.end() && staged->second.size() >= meta->byte_size) {
            dma_src = staged->second.data();
        }
    }
    if (dma_src != meta->host_ptr) {
        // using staged buffer from async/scheduled LOAD
    } else if (octx->direct_read_fn) {
        static thread_local std::vector<char> direct_scratch;
        if (direct_scratch.size() < meta->byte_size) direct_scratch.resize(meta->byte_size);
        const uint64_t direct_t0 = now_us();
        if (octx->direct_read_fn(meta->host_ptr, direct_scratch.data(), meta->byte_size) == 0) {
            octx->foreground_direct_read_calls++;
            octx->foreground_direct_read_us += now_us() - direct_t0;
            octx->foreground_direct_read_bytes += meta->byte_size;
            dma_src            = direct_scratch.data();
            use_direct_scratch = true;
        } else {
            std::fprintf(stderr, "[wbmcl] direct_read_fn 失败 idx=%d, fallback mmap\n", idx);
        }
    }

    cl_int err = CL_SUCCESS;
    // Retain 模式：从 size 池里拿一个同 size 的 cl_mem 复用，省 clCreateBuffer
    if (cl_mem cached = retain_pool_take_buffer(octx, meta->byte_size)) {
            if (octx->xfer_queue) {
                cl_command_queue use_q = octx->xfer_queue;
                if (octx->n_xfer_extra > 0) {
                    unsigned slot = octx->xfer_round_robin++ % (octx->n_xfer_extra + 1);
                    if (slot > 0) use_q = octx->xfer_extra[slot - 1];
                }
                cl_event write_ev = nullptr;
                err = clEnqueueWriteBuffer(use_q, cached, use_direct_scratch ? CL_TRUE : CL_FALSE,
                                           0, meta->byte_size, dma_src,
                                           0, nullptr, &write_ev);
                if (err == CL_SUCCESS) {
                    wbmcl_record_device_event(octx, write_ev,
                                              wbmcl_device_event_kind::TRANSFER_WRITE,
                                              meta->byte_size);
                    clFlush(use_q);
                    cl_event wait_ev = nullptr;
                    cl_int berr = clEnqueueBarrierWithWaitList(octx->compute_queue, 1, &write_ev, &wait_ev);
                    if (berr != CL_SUCCESS) {
                        std::fprintf(stderr, "[wbmcl retain] clEnqueueBarrierWithWaitList 失败: %s (%d)，退回 wait\n",
                                     cl_err(berr), berr);
                        clWaitForEvents(1, &write_ev);
                    } else {
                        wbmcl_record_device_event(octx, wait_ev,
                                                  wbmcl_device_event_kind::COMPUTE_WAIT,
                                                  meta->byte_size);
                    }
                    if (wait_ev) clReleaseEvent(wait_ev);
                    clReleaseEvent(write_ev);
                }
            } else {
                err = clEnqueueWriteBuffer(octx->compute_queue,
                                           cached, use_direct_scratch ? CL_TRUE : CL_FALSE,
                                           0, meta->byte_size, dma_src,
                                           0, nullptr, nullptr);
            }
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

    // GGML_ELASTIC_USE_HOST_PTR=1 实验：让 OpenCL 用 mmap 指针直接做 cl_mem
    // 后备存储，省掉显式的 host→GPU memcpy。Adreno unified memory 下可能
    // 实现零拷贝；非 unified 架构（如桌面独显）driver 会自己做一次 copy，
    // 等价但多一次驱动开销。默认关。
    static const bool s_use_host_ptr_env = []() {
        const char *e = std::getenv("GGML_ELASTIC_USE_HOST_PTR");
        return e && *e && *e != '0';
    }();
    // direct 模式跟 USE_HOST_PTR 互斥: direct scratch 要显式 transfer 上去,
    // USE_HOST_PTR 是让 driver 直接绑 mmap 指针 (不会读 scratch)。direct 优先。
    const bool s_use_host_ptr = s_use_host_ptr_env && !use_direct_scratch;

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
        // 实测 2 queue 并发能让 transfer 吞吐 ~2× (Adreno 单队列没吃满 host→GPU
        // staging 的 per-call overhead)。
        cl_command_queue use_q = octx->xfer_queue;
        if (octx->n_xfer_extra > 0) {
            unsigned slot = octx->xfer_round_robin++ % (octx->n_xfer_extra + 1);
            if (slot > 0) use_q = octx->xfer_extra[slot - 1];
        }
        // direct scratch 模式: dma_src 是复用的 thread_local scratch, 非阻塞写会在下次
        // ensure_resident 覆盖 scratch 前来不及读完 → 用阻塞写保证 scratch 安全。
        cl_event write_ev = nullptr;
        err = clEnqueueWriteBuffer(use_q, buf, use_direct_scratch ? CL_TRUE : CL_FALSE,
                                   0, meta->byte_size, dma_src,
                                   0, nullptr, &write_ev);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] async clEnqueueWriteBuffer 失败 block %d: %s (%d)\n",
                         idx, cl_err(err), err);
            clReleaseMemObject(buf);
            octx->n_releases += 1;
            return -5;
        }
        wbmcl_record_device_event(octx, write_ev,
                                  wbmcl_device_event_kind::TRANSFER_WRITE,
                                  meta->byte_size);
        clFlush(use_q);
        // compute queue 插 barrier，依赖 write_ev
        cl_event wait_ev = nullptr;
        cl_int berr = clEnqueueBarrierWithWaitList(octx->compute_queue, 1, &write_ev, &wait_ev);
        if (berr != CL_SUCCESS) {
            std::fprintf(stderr, "[wbmcl] clEnqueueBarrierWithWaitList 失败: %s (%d)，退回 wait\n",
                         cl_err(berr), berr);
            clWaitForEvents(1, &write_ev);
        } else {
            wbmcl_record_device_event(octx, wait_ev,
                                      wbmcl_device_event_kind::COMPUTE_WAIT,
                                      meta->byte_size);
        }
        if (wait_ev) clReleaseEvent(wait_ev);
        clReleaseEvent(write_ev);
    } else {
        // 同步路径：单队列阻塞写
        err = clEnqueueWriteBuffer(octx->compute_queue,
                                   buf, use_direct_scratch ? CL_TRUE : CL_FALSE,
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

    // 保守默认等待 queue drain。OP12/Adreno 750 在 dynamic plan 切换后若立即
    // release/reuse cl_mem，driver 线程可能在旧 kernel 完成前触碰失效对象而崩溃。
    // GGML_ELASTIC_EVICT_WAIT=0 可恢复激进路径用于单独做性能诊断。
    static const bool s_evict_wait = []() {
        const char *e = std::getenv("GGML_ELASTIC_EVICT_WAIT");
        return !(e && *e && *e == '0');
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

    auto soa = octx->soa_per_idx.find(idx);
    if (soa != octx->soa_per_idx.end() && soa->second.evict_fn) {
        const int rc = soa->second.evict_fn();
        if (rc != 0) {
            std::fprintf(stderr, "[wbmcl] SOA evict_fn 失败 block %d rc=%d\n", idx, rc);
            return rc;
        }
        if (!octx->async_stage_load) {
            release_host_staging(octx, idx);
        } else if (async_load_on_evict_enabled()) {
            wbmcl_load_host_async(octx, idx);
        }
        return 0;
    }

    // Retain 模式：cl_mem 按 size 入池，cap 检查
    if (octx->retain_cl_mem) {
        retain_pool_put_buffer(octx, buf, meta->byte_size);
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
    if (octx->async_stage_load && async_load_on_evict_enabled()) wbmcl_load_host_async(octx, idx);
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
    const uint64_t t0 = now_us();
    octx->stage_load_calls++;

    std::lock_guard<std::mutex> lock(octx->host_staging_mtx);
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
        const uint64_t direct_t0 = now_us();
        rc = octx->direct_read_fn(meta->host_ptr, staging.data(), meta->byte_size);
        octx->async_load_direct_read_calls++;
        octx->async_load_direct_read_us += now_us() - direct_t0;
        if (rc == 0) octx->async_load_direct_read_bytes += meta->byte_size;
    }
    if (rc != 0) {
        std::memcpy(staging.data(), meta->host_ptr, meta->byte_size);
    }
    octx->bytes_loaded_total += meta->byte_size;
    octx->stage_load_ok++;
    octx->stage_load_bytes += meta->byte_size;
    octx->stage_load_us += now_us() - t0;
    return 0;
}

int wbmcl_load_host_async_unit(wbm_opencl_ctx *octx,
                               const int *indices, size_t n_indices) {
    if (!octx || !octx->wbm) return -1;
    if (!indices || n_indices == 0) return 0;
    if (!octx->async_stage_load) {
        for (size_t i = 0; i < n_indices; ++i) {
            const int rc = wbmcl_load_host(octx, indices[i]);
            if (rc != 0) return rc;
        }
        return 0;
    }

    std::vector<int> task;
    size_t task_bytes = 0;
    {
        std::lock_guard<std::mutex> staging_lock(octx->host_staging_mtx);
        task.reserve(n_indices);
        for (size_t i = 0; i < n_indices; ++i) {
            const int idx = indices[i];
            if (idx < 0 || std::find(task.begin(), task.end(), idx) != task.end()) continue;
            const block_meta * meta = wbm_get(octx->wbm, idx);
            if (!meta) return -2;
            if (meta->resident) continue;
            if (!meta->host_ptr || meta->byte_size == 0) return -3;
            auto hit = octx->host_staging_by_idx.find(idx);
            if (hit != octx->host_staging_by_idx.end() &&
                hit->second.size() >= meta->byte_size) {
                continue;
            }
            task.push_back(idx);
            task_bytes += meta->byte_size;
        }
    }
    if (task.empty()) return 0;

    size_t staged_bytes = 0;
    const bool legacy_unbounded = async_stage_load_legacy_unbounded_enabled();
    const size_t max_staged_bytes = legacy_unbounded ? 0 : async_stage_load_max_staged_bytes();
    if (max_staged_bytes > 0) {
        std::lock_guard<std::mutex> staging_lock(octx->host_staging_mtx);
        staged_bytes = host_staging_active_bytes_locked(octx);
    }
    {
        std::unique_lock<std::mutex> lock(octx->async_load_mtx);
        // Drop members already covered by an earlier unit.  Never split a
        // remaining Multi task merely to satisfy a pending-count cap.
        task.erase(std::remove_if(task.begin(), task.end(), [&](int idx) {
            auto it = octx->async_load_state.find(idx);
            return it != octx->async_load_state.end() && it->second == 1;
        }), task.end());
        if (task.empty()) return 0;
        task_bytes = 0;
        for (int idx : task) {
            const block_meta * meta = wbm_get(octx->wbm, idx);
            if (meta) task_bytes += meta->byte_size;
        }
        if (max_staged_bytes > 0 &&
            staged_bytes + octx->async_load_pending_bytes + task_bytes > max_staged_bytes &&
            (staged_bytes + octx->async_load_pending_bytes > 0 || task_bytes > max_staged_bytes)) {
            octx->async_load_throttle_skipped += task.size();
            return 0;
        }
        const size_t max_pending_bytes = legacy_unbounded ? 0 : async_stage_load_max_pending_bytes();
        const size_t max_pending_count = legacy_unbounded ? 0 : async_stage_load_max_pending_count();
        if ((max_pending_bytes > 0 || max_pending_count > 0) && task_bytes > 0) {
            const uint64_t wait_t0 = now_us();
            bool waited = false;
            octx->async_load_cv.wait(lock, [&]() {
                if (octx->async_load_shutdown) return true;
                const size_t pending_count = octx->async_load_bytes_by_idx.size();
                const bool bytes_ok =
                    max_pending_bytes == 0 ||
                    octx->async_load_pending_bytes + task_bytes <= max_pending_bytes ||
                    octx->async_load_pending_bytes == 0;
                const bool count_ok =
                    max_pending_count == 0 ||
                    pending_count + task.size() <= max_pending_count ||
                    pending_count == 0;
                const bool ok = bytes_ok && count_ok;
                if (!ok) waited = true;
                return ok;
            });
            if (waited) {
                octx->async_load_throttle_waits++;
                octx->async_load_throttle_wait_us += now_us() - wait_t0;
            }
        }
        if (octx->async_load_shutdown) return -10;
        for (int idx : task) {
            const block_meta * meta = wbm_get(octx->wbm, idx);
            if (!meta) continue;
            octx->async_load_state[idx] = 1;
            octx->async_load_bytes_by_idx[idx] = meta->byte_size;
            octx->async_load_pending_bytes += meta->byte_size;
        }
        octx->async_load_max_pending_bytes_seen =
            std::max(octx->async_load_max_pending_bytes_seen, octx->async_load_pending_bytes);
        octx->async_load_max_pending_count_seen =
            std::max(octx->async_load_max_pending_count_seen, octx->async_load_bytes_by_idx.size());
        octx->async_load_enqueued += task.size();
        octx->async_load_units_enqueued++;
        octx->async_load_queue.push_back(std::move(task));
    }
    octx->async_load_cv.notify_one();
    return 0;
}

int wbmcl_load_host_async(wbm_opencl_ctx *octx, int idx) {
    return wbmcl_load_host_async_unit(octx, &idx, 1);
}

int wbmcl_wait_host_load(wbm_opencl_ctx *octx, int idx) {
    if (!octx || !octx->async_stage_load) return 0;
    {
        std::lock_guard<std::mutex> staging_lock(octx->host_staging_mtx);
        const block_meta *meta = wbm_get(octx->wbm, idx);
        auto hit = octx->host_staging_by_idx.find(idx);
        if (meta && hit != octx->host_staging_by_idx.end() && hit->second.size() >= meta->byte_size) {
            return 0;
        }
    }
    const uint64_t t0 = now_us();
    int state = 0;
    {
        std::unique_lock<std::mutex> lock(octx->async_load_mtx);
        auto it = octx->async_load_state.find(idx);
        if (it == octx->async_load_state.end()) return 0;
        if (it->second != 1) return it->second >= 0 ? 0 : it->second;
        octx->async_load_waits++;
        octx->async_load_cv.wait(lock, [&]() {
            auto cur = octx->async_load_state.find(idx);
            return cur == octx->async_load_state.end() || cur->second != 1;
        });
        auto done = octx->async_load_state.find(idx);
        state = done == octx->async_load_state.end() ? 0 : done->second;
    }
    octx->async_load_wait_us += now_us() - t0;
    return state >= 0 ? 0 : state;
}

static uint64_t wbmcl_host_source_us(wbm_opencl_ctx *octx) {
    if (!octx || !octx->stage_detail) return 0;
    std::lock_guard<std::mutex> lock(octx->stage_detail_mtx);
    return octx->stage_detail_buckets[
        static_cast<int>(wbmcl_stage_detail_kind::HOST_SRC)].us;
}

static int wbmcl_run_soa_prepare(
        wbm_opencl_ctx *octx,
        int idx,
        const std::function<int()> &fn) {
    if (!octx || !octx->wbm || idx < 0 || !fn) return -1;
    const block_meta *meta = wbm_get(octx->wbm, idx);
    if (!meta) return -2;

    const uint64_t host_src_before = wbmcl_host_source_us(octx);
    const uint64_t t0 = now_us();
    const int rc = fn();
    const uint64_t wall_us = now_us() - t0;
    const uint64_t host_src_after = wbmcl_host_source_us(octx);
    const uint64_t host_src_us =
        host_src_after >= host_src_before ? host_src_after - host_src_before : 0;
    const uint64_t prepare_us =
        wall_us > host_src_us ? wall_us - host_src_us : 0;

    {
        std::lock_guard<std::mutex> lock(octx->soa_prepare_stats_mtx);
        octx->soa_prepare_calls++;
        octx->soa_prepare_us += prepare_us;
        if (rc == 0) {
            octx->soa_prepare_ok++;
            octx->soa_prepare_bytes += meta->byte_size;
        }
    }
    return rc;
}

int wbmcl_reload_soa_sync(wbm_opencl_ctx *octx, int idx) {
    if (!octx) return -1;
    auto it = octx->soa_per_idx.find(idx);
    if (it == octx->soa_per_idx.end() || !it->second.reload_fn) return -2;
    return wbmcl_run_soa_prepare(octx, idx, it->second.reload_fn);
}

int wbmcl_transfer_soa_sync(wbm_opencl_ctx *octx, int idx) {
    if (!octx) return -1;
    auto it = octx->soa_per_idx.find(idx);
    if (it == octx->soa_per_idx.end() || !it->second.transfer_fn) return -2;
    return wbmcl_run_soa_prepare(octx, idx, it->second.transfer_fn);
}

int wbmcl_xform_soa_sync(wbm_opencl_ctx *octx, int idx) {
    if (!octx) return -1;
    auto it = octx->soa_per_idx.find(idx);
    if (it == octx->soa_per_idx.end() || !it->second.xform_fn) return -2;
    return wbmcl_run_soa_prepare(octx, idx, it->second.xform_fn);
}

static void wbmcl_start_soa_reload_worker(wbm_opencl_ctx *octx) {
    if (!octx) return;
    {
        std::lock_guard<std::mutex> lock(octx->async_soa_reload_mtx);
        if (octx->async_soa_reload_worker_started) return;
        octx->async_soa_reload_worker_started = true;
    }
    octx->async_soa_reload_worker = std::thread([octx]() {
        for (;;) {
            std::vector<int> task;
            {
                std::unique_lock<std::mutex> lock(octx->async_soa_reload_mtx);
                octx->async_soa_reload_cv.wait(lock, [octx]() {
                    return octx->async_soa_reload_shutdown ||
                           !octx->async_soa_reload_queue.empty();
                });
                if (octx->async_soa_reload_shutdown &&
                    octx->async_soa_reload_queue.empty()) break;
                task = std::move(octx->async_soa_reload_queue.front());
                octx->async_soa_reload_queue.pop_front();
            }
            for (int idx : task) {
                int rc = -2;
                auto it = octx->soa_per_idx.find(idx);
                if (it != octx->soa_per_idx.end() && it->second.reload_fn) {
                    const block_meta *meta = wbm_get(octx->wbm, idx);
                    rc = (meta && meta->resident) ? 0 :
                        wbmcl_reload_soa_sync(octx, idx);
                }
                {
                    std::lock_guard<std::mutex> lock(octx->async_soa_reload_mtx);
                    octx->async_soa_reload_state[idx] = rc == 0 ? 2 : rc;
                    octx->async_soa_reload_completed++;
                }
                // PREPARE for tensor i may be consumed while the same logical
                // unit continues materializing tensor i+1.
                octx->async_soa_reload_cv.notify_all();
            }
            {
                std::lock_guard<std::mutex> lock(octx->async_soa_reload_mtx);
                octx->async_soa_reload_units_completed++;
            }
            octx->async_soa_reload_cv.notify_all();
        }
    });
}

int wbmcl_soa_reload_async_unit(wbm_opencl_ctx *octx,
                                const int *indices, size_t n_indices) {
    if (!octx || !octx->wbm) return -1;
    if (!indices || n_indices == 0) return 0;
    std::vector<int> task;
    task.reserve(n_indices);
    for (size_t i = 0; i < n_indices; ++i) {
        const int idx = indices[i];
        if (idx < 0 || std::find(task.begin(), task.end(), idx) != task.end()) continue;
        const block_meta *meta = wbm_get(octx->wbm, idx);
        if (!meta) return -2;
        if (meta->resident) continue;
        auto it = octx->soa_per_idx.find(idx);
        if (it == octx->soa_per_idx.end() || !it->second.reload_fn) return -3;
        task.push_back(idx);
    }
    if (task.empty()) return 0;
    wbmcl_start_soa_reload_worker(octx);
    {
        std::lock_guard<std::mutex> lock(octx->async_soa_reload_mtx);
        task.erase(std::remove_if(task.begin(), task.end(), [&](int idx) {
            auto cur = octx->async_soa_reload_state.find(idx);
            return cur != octx->async_soa_reload_state.end() && cur->second == 1;
        }), task.end());
        if (task.empty()) return 0;
        const size_t max_pending = async_stage_prepare_max_pending();
        if (max_pending > 0) {
            size_t pending = 0;
            for (const auto & kv : octx->async_soa_reload_state) {
                if (kv.second == 1) pending++;
            }
            // Preserve unit atomicity: defer the entire unit instead of
            // enqueuing only the first member of a Multi task.
            if (pending > 0 && pending + task.size() > max_pending) return 0;
        }
        for (int idx : task) octx->async_soa_reload_state[idx] = 1;
        octx->async_soa_reload_enqueued += task.size();
        octx->async_soa_reload_units_enqueued++;
        octx->async_soa_reload_queue.push_back(std::move(task));
    }
    octx->async_soa_reload_cv.notify_one();
    return 0;
}

int wbmcl_soa_reload_async(wbm_opencl_ctx *octx, int idx) {
    return wbmcl_soa_reload_async_unit(octx, &idx, 1);
}

int wbmcl_wait_soa_reload(wbm_opencl_ctx *octx, int idx) {
    if (!octx || idx < 0) return -1;
    const uint64_t t0 = now_us();
    int state = 0;
    {
        std::unique_lock<std::mutex> lock(octx->async_soa_reload_mtx);
        auto it = octx->async_soa_reload_state.find(idx);
        if (it == octx->async_soa_reload_state.end()) return 0;
        if (it->second == 1) {
            octx->async_soa_reload_waits++;
            octx->async_soa_reload_cv.wait(lock, [&]() {
                auto cur = octx->async_soa_reload_state.find(idx);
                return cur == octx->async_soa_reload_state.end() || cur->second != 1;
            });
        }
        auto done = octx->async_soa_reload_state.find(idx);
        state = done == octx->async_soa_reload_state.end() ? 0 : done->second;
    }
    octx->async_soa_reload_wait_us += now_us() - t0;
    return state >= 0 ? 0 : state;
}

int wbmcl_dma_to_backend(wbm_opencl_ctx *octx, int idx) {
    if (!octx || !octx->wbm) return -1;
    const block_meta *meta = wbm_get(octx->wbm, idx);
    const size_t nbytes = meta ? meta->byte_size : 0;
    const uint64_t t0 = now_us();
    octx->stage_transfer_calls++;

    int rc = 0;
    if (!meta) {
        rc = -2;
    } else if (meta->resident || meta->prefetch_event) {
        rc = 0;
    } else if (auto soa = octx->soa_per_idx.find(idx);
               soa != octx->soa_per_idx.end() && (soa->second.transfer_fn || soa->second.reload_fn)) {
        // SOA tensors need backend materialization (write + convert/transpose)
        // before compute.  In the staged pipeline experiment, let TRANSFER
        // trigger that callback at the transfer anchor.  Without the env flag,
        // preserve the older behavior where XFORM/foreground ensure performs it.
        if (async_stage_prepare_enabled()) {
            rc = wbmcl_soa_reload_async(octx, idx);
        } else if (soa->second.transfer_fn) {
            rc = wbmcl_transfer_soa_sync(octx, idx);
        } else {
            rc = soa_reload_on_transfer_enabled() ?
                wbmcl_reload_soa_sync(octx, idx) : 0;
        }
    } else {
        const int wait_rc = wbmcl_wait_host_load(octx, idx);
        if (wait_rc != 0) {
            rc = wait_rc;
        }
        std::vector<char> * staged_ptr = nullptr;
        if (rc == 0) {
            std::lock_guard<std::mutex> lock(octx->host_staging_mtx);
            auto staged = octx->host_staging_by_idx.find(idx);
            if (staged != octx->host_staging_by_idx.end() && staged->second.size() >= meta->byte_size) {
                staged_ptr = &staged->second;
            }
        }
        if (rc == 0 && staged_ptr == nullptr) {
            // No preceding LOAD stage reached this tensor. Fall back to the
            // correctness path, which may perform direct disk read + transfer
            // synchronously.
            rc = wbmcl_ensure_resident(octx, idx);
            if (rc == 0 && (!octx->async_stage_load || release_stage_after_xform_enabled())) release_host_staging(octx, idx);
        } else {
            cl_int err = CL_SUCCESS;
            cl_mem buf = retain_pool_take_buffer(octx, meta->byte_size);
            if (!buf) {
                buf = clCreateBuffer(octx->cl_ctx, CL_MEM_READ_ONLY, meta->byte_size, nullptr, &err);
                if (err != CL_SUCCESS) {
                    std::fprintf(stderr, "[wbmcl stage transfer] clCreateBuffer failed block %d size %zu: %s (%d)\n",
                                 idx, meta->byte_size, cl_err(err), err);
                    rc = -4;
                } else {
                    octx->n_creates += 1;
                }
            }
            if (rc == 0) {
                if (octx->xfer_queue) {
                    cl_command_queue use_q = octx->xfer_queue;
                    if (octx->n_xfer_extra > 0) {
                        unsigned slot = octx->xfer_round_robin++ % (octx->n_xfer_extra + 1);
                        if (slot > 0) use_q = octx->xfer_extra[slot - 1];
                    }
                    cl_event write_ev = nullptr;
                    err = clEnqueueWriteBuffer(use_q, buf, CL_FALSE,
                                               0, meta->byte_size, staged_ptr->data(),
                                               0, nullptr, &write_ev);
                    if (err != CL_SUCCESS) {
                        std::fprintf(stderr, "[wbmcl stage transfer] async clEnqueueWriteBuffer failed block %d: %s (%d)\n",
                                     idx, cl_err(err), err);
                        clReleaseMemObject(buf);
                        octx->n_releases += 1;
                        rc = -5;
                    } else {
                        wbmcl_record_device_event(octx, write_ev,
                                                  wbmcl_device_event_kind::TRANSFER_WRITE,
                                                  meta->byte_size);
                        clFlush(use_q);
                        octx->wbm->blocks[static_cast<size_t>(idx)].backend_handle = static_cast<void *>(buf);
                        wbm_set_prefetch_event(octx->wbm, idx, static_cast<void *>(write_ev));
                    }
                } else {
                    err = clEnqueueWriteBuffer(octx->compute_queue, buf, CL_TRUE,
                                               0, meta->byte_size, staged_ptr->data(),
                                               0, nullptr, nullptr);
                    if (err != CL_SUCCESS) {
                        std::fprintf(stderr, "[wbmcl stage transfer] clEnqueueWriteBuffer failed block %d: %s (%d)\n",
                                     idx, cl_err(err), err);
                        clReleaseMemObject(buf);
                        octx->n_releases += 1;
                        rc = -5;
                    } else {
                        octx->bytes_uploaded_total += meta->byte_size;
                        wbm_mark_resident(octx->wbm, idx, static_cast<void *>(buf));
                        if (!octx->async_stage_load || release_stage_after_xform_enabled()) release_host_staging(octx, idx);
                    }
                }
            }
        }
    }

    if (rc == 0) {
        octx->stage_transfer_ok++;
        octx->stage_transfer_bytes += nbytes;
    }
    octx->stage_transfer_us += now_us() - t0;
    return rc;
}

int wbmcl_transform_backend(wbm_opencl_ctx *octx, int idx) {
    if (!octx || !octx->wbm) return -1;
    const block_meta *meta0 = wbm_get(octx->wbm, idx);
    const size_t nbytes = meta0 ? meta0->byte_size : 0;
    const uint64_t t0 = now_us();
    octx->stage_xform_calls++;
    int rc = 0;
    auto it = octx->soa_per_idx.find(idx);
    if (it != octx->soa_per_idx.end() && (it->second.xform_fn || it->second.reload_fn)) {
        const block_meta *meta = wbm_get(octx->wbm, idx);
        if (meta && meta->resident) {
            if (!octx->async_stage_load || release_stage_after_xform_enabled()) release_host_staging(octx, idx);
            rc = 0;
        } else if (async_stage_prepare_enabled() && wbmcl_wait_soa_reload(octx, idx) == 0 &&
                   (meta = wbm_get(octx->wbm, idx)) && meta->resident) {
            if (release_stage_after_xform_enabled()) release_host_staging(octx, idx);
            rc = 0;
        } else if (it->second.xform_fn) {
            rc = wbmcl_xform_soa_sync(octx, idx);
            if (rc == 0 && (!octx->async_stage_load || release_stage_after_xform_enabled())) release_host_staging(octx, idx);
        } else {
            rc = wbmcl_reload_soa_sync(octx, idx);
            if (rc == 0 && (!octx->async_stage_load || release_stage_after_xform_enabled())) release_host_staging(octx, idx);
        }
    } else {
        if (!octx->async_stage_load || release_stage_after_xform_enabled()) release_host_staging(octx, idx);
        rc = 0;
    }
    if (rc == 0) {
        octx->stage_xform_ok++;
        octx->stage_xform_bytes += nbytes;
    }
    octx->stage_xform_us += now_us() - t0;
    return rc;
}

int wbmcl_prepare_backend(wbm_opencl_ctx *octx, int idx) {
    if (!octx || !octx->wbm) return -1;
    const block_meta *meta = wbm_get(octx->wbm, idx);
    if (!meta) return -2;
    if (meta->resident) return 0;

    auto soa = octx->soa_per_idx.find(idx);
    if (soa != octx->soa_per_idx.end() && (soa->second.xform_fn || soa->second.transfer_fn || soa->second.reload_fn)) {
        // Treat OpenCL SOA materialization as one prepare stage.  Internally it
        // may perform host write, convert and transpose; planner/runtime should
        // not need to schedule those as separate high-level stages.
        if (async_stage_prepare_enabled()) {
            return wbmcl_soa_reload_async(octx, idx);
        }
        if (soa->second.xform_fn) {
            return wbmcl_xform_soa_sync(octx, idx);
        }
        if (soa->second.transfer_fn) {
            return wbmcl_transfer_soa_sync(octx, idx);
        }
        return wbmcl_reload_soa_sync(octx, idx);
    }

    int rc = wbmcl_dma_to_backend(octx, idx);
    if (rc != 0) return rc;
    return wbmcl_transform_backend(octx, idx);
}

int wbmcl_prepare_backend_unit(wbm_opencl_ctx *octx,
                               const int *indices, size_t n_indices) {
    if (!octx || !octx->wbm) return -1;
    if (!indices || n_indices == 0) return 0;
    if (!async_stage_prepare_enabled()) {
        for (size_t i = 0; i < n_indices; ++i) {
            const int rc = wbmcl_prepare_backend(octx, indices[i]);
            if (rc != 0) return rc;
        }
        return 0;
    }

    std::vector<int> soa_indices;
    soa_indices.reserve(n_indices);
    for (size_t i = 0; i < n_indices; ++i) {
        const int idx = indices[i];
        const block_meta * meta = wbm_get(octx->wbm, idx);
        if (!meta) return -2;
        if (meta->resident) continue;
        auto soa = octx->soa_per_idx.find(idx);
        if (soa != octx->soa_per_idx.end() &&
            (soa->second.xform_fn || soa->second.transfer_fn || soa->second.reload_fn)) {
            soa_indices.push_back(idx);
            continue;
        }
        // Non-SOA tensors retain the existing correctness path.  Model
        // weight units in the OpenCL Q4 path are SOA, so this fallback is not
        // used for the measured granularity tasks.
        const int rc = wbmcl_prepare_backend(octx, idx);
        if (rc != 0) return rc;
    }
    return soa_indices.empty()
        ? 0
        : wbmcl_soa_reload_async_unit(
              octx, soa_indices.data(), soa_indices.size());
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
    buf = retain_pool_take_buffer(octx, meta->byte_size);
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
    wbmcl_record_device_event(octx, write_ev,
                              wbmcl_device_event_kind::TRANSFER_WRITE,
                              meta->byte_size);
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

    // evict 前是否等 queue drain。默认等待以避免 Adreno 在动态切换后释放/复用
    // 仍被 driver 内部线程引用的 cl_mem。GGML_ELASTIC_EVICT_WAIT=0 可恢复激进路径。
    static const bool s_evict_wait = []() {
        const char *e = std::getenv("GGML_ELASTIC_EVICT_WAIT");
        return !(e && *e && *e == '0');
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
        auto soa = octx->soa_per_idx.find(v);
        if (soa != octx->soa_per_idx.end() && soa->second.evict_fn) {
            const int rc = soa->second.evict_fn();
            if (rc != 0) {
                std::fprintf(stderr, "[wbmcl] 批量 SOA evict_fn 失败 block %d rc=%d\n", v, rc);
                continue;
            }
            if (!octx->async_stage_load) {
                release_host_staging(octx, v);
            } else if (async_load_on_evict_enabled()) {
                wbmcl_load_host_async(octx, v);
            }
            ++released;
            continue;
        }
        cl_mem buf = static_cast<cl_mem>(m->backend_handle);
        if (buf) {
            // Retain 模式：cl_mem 按 size 入池。pool 满时 FIFO 释放最早 size。
            if (octx->retain_cl_mem) {
                retain_pool_put_buffer(octx, buf, m->byte_size);
                octx->bytes_evicted_total += m->byte_size;
                wbm_mark_evicted(octx->wbm, v);
                if (octx->async_stage_load && async_load_on_evict_enabled()) wbmcl_load_host_async(octx, v);
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

void wbmcl_wait_async_idle(wbm_opencl_ctx *octx) {
    if (!octx) return;
    {
        std::unique_lock<std::mutex> lock(octx->async_load_mtx);
        octx->async_load_cv.wait(lock, [octx]() {
            if (!octx->async_load_queue.empty()) return false;
            return std::none_of(
                octx->async_load_state.begin(),
                octx->async_load_state.end(),
                [](const auto &entry) { return entry.second == 1; });
        });
    }
    {
        std::unique_lock<std::mutex> lock(octx->async_soa_reload_mtx);
        octx->async_soa_reload_cv.wait(lock, [octx]() {
            if (!octx->async_soa_reload_queue.empty()) return false;
            return std::none_of(
                octx->async_soa_reload_state.begin(),
                octx->async_soa_reload_state.end(),
                [](const auto &entry) { return entry.second == 1; });
        });
    }
}

void wbmcl_stop_async_workers(wbm_opencl_ctx *octx) {
    if (!octx) return;

    // Join based on std::thread lifetime itself. This remains safe and
    // idempotent even if feature flags have already changed during teardown.
    {
        std::lock_guard<std::mutex> lock(octx->async_load_mtx);
        octx->async_load_shutdown = true;
    }
    octx->async_load_cv.notify_all();
    if (octx->async_load_worker.joinable()) {
        octx->async_load_worker.join();
    }
    octx->async_stage_load = false;

    {
        std::lock_guard<std::mutex> lock(octx->async_soa_reload_mtx);
        octx->async_soa_reload_shutdown = true;
    }
    octx->async_soa_reload_cv.notify_all();
    if (octx->async_soa_reload_worker.joinable()) {
        octx->async_soa_reload_worker.join();
    }
    octx->async_soa_reload_worker_started = false;
}

void wbmcl_shutdown(wbm_opencl_ctx *octx) {
    if (!octx || !octx->wbm) return;
    wbmcl_stop_async_workers(octx);
    if (octx->soa_staging_last_use_ev) {
        clWaitForEvents(1, &octx->soa_staging_last_use_ev);
        clReleaseEvent(octx->soa_staging_last_use_ev);
        octx->soa_staging_last_use_ev = nullptr;
    }
    if (octx->soa_staging) {
        clReleaseMemObject(octx->soa_staging);
        octx->n_releases += 1;
        octx->soa_staging = nullptr;
        octx->soa_staging_capacity = 0;
    }
    for (cl_event ev : octx->soa_staging_slot_last_use_ev) {
        if (ev) {
            clWaitForEvents(1, &ev);
            clReleaseEvent(ev);
        }
    }
    for (cl_mem mem : octx->soa_staging_slots) {
        if (mem) {
            clReleaseMemObject(mem);
            octx->n_releases += 1;
        }
    }
    octx->soa_staging_slots.clear();
    octx->soa_staging_slot_capacity.clear();
    octx->soa_staging_slot_last_use_ev.clear();
    octx->soa_staging_next_slot = 0;
    octx->soa_staging_current_slot = 0;

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
    {
        std::lock_guard<std::mutex> lock(octx->host_staging_mtx);
        octx->host_staging_by_idx.clear();
        octx->host_staging_pool_by_size.clear();
        octx->host_staging_pool_order.clear();
    }
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
    wbmcl_register_soa_split(octx, idx, std::move(evict_fn), std::move(reload_fn), nullptr, nullptr);
}

void wbmcl_register_soa_split(wbm_opencl_ctx *octx, int idx,
                              std::function<int()> evict_fn,
                              std::function<int()> reload_fn,
                              std::function<int()> transfer_fn,
                              std::function<int()> xform_fn) {
    if (!octx) return;
    octx->soa_per_idx[idx] = soa_callbacks{
        std::move(evict_fn),
        std::move(reload_fn),
        std::move(transfer_fn),
        std::move(xform_fn),
    };
}

}  // namespace elastic
