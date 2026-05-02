#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>

// AdapterPool: size-aware (byte-budget) LRU cache of llama_adapter_lora handles.
//
// M3 scope (spec section 4 M3):
//   - Constraint is total resident bytes, not count. The byte cost of an
//     adapter is approximated as the size of its on-disk GGUF file (see
//     known-issues.md I-2 for the rationale; this is the cheap and
//     reproducible option).
//   - The cache is split into two intrusive lists:
//       lru_free_  — entries with ref_count == 0; eviction-eligible,
//                    ordered front=MRU / back=LRU
//       pinned_    — entries with ref_count > 0; never evicted, no order
//     This is the I-2 fix: the M1 sketch's "splice in-use entry to MRU"
//     trick contaminated the LRU order. Two lists keep ref_count and
//     LRU semantics decoupled.
//   - acquire / release move entries between the two lists at ref-count
//     boundaries (0->1 to pinned_, 1->0 to lru_free_ MRU end).
//   - evict_until walks lru_free_ from the back; if even after draining
//     all free entries we still don't have room, the load proceeds and
//     the cache temporarily exceeds max_bytes (caller is warned via stderr).
//
// Single-threaded by design: no mutex. Main loop is the sole caller.
class multilora_adapter_pool {
public:
    struct stats_t {
        size_t hits           = 0;
        size_t misses         = 0;
        size_t evicts         = 0;
        size_t resident_count = 0;
        size_t resident_bytes = 0;
        // Highest current_bytes_ ever observed (useful for spotting the
        // "budget exceeded because everything was pinned" pathology).
        size_t peak_bytes     = 0;
    };

    // max_bytes == 0 -> unbounded (no eviction); behaves like a pure
    // load-on-demand pool. Use SIZE_MAX or any large value for the same
    // effect — 0 is the explicit sentinel main.cpp passes when the user
    // omits --max-adapter-mem-mb.
    multilora_adapter_pool(struct llama_model * model,
                           std::string          adapter_dir,
                           size_t               max_bytes);

    ~multilora_adapter_pool();

    multilora_adapter_pool(const multilora_adapter_pool &)             = delete;
    multilora_adapter_pool & operator=(const multilora_adapter_pool &) = delete;

    struct acquire_result {
        struct llama_adapter_lora * handle    = nullptr;
        bool                        cache_hit = false;
        double                      load_ms   = 0.0;  // 0 on hit
        size_t                      bytes     = 0;    // adapter's accounted bytes
    };

    acquire_result acquire(struct llama_context * ctx, const std::string & id);

    void release(const std::string & id);

    void shutdown();

    stats_t stats() const { return stats_; }
    double  hit_rate() const {
        const size_t total = stats_.hits + stats_.misses;
        return total == 0 ? 0.0 : double(stats_.hits) / double(total);
    }

private:
    struct entry {
        std::string                      id;
        struct llama_adapter_lora *      adapter   = nullptr;
        size_t                           bytes     = 0;
        int                              ref_count = 0;
        bool                             in_pinned = false;
        std::list<std::string>::iterator list_it;
    };

    void   evict_until(size_t bytes_needed);
    void   detach_from_current_list(entry & e);
    void   attach_to_pinned(entry & e);
    void   attach_to_lru_free(entry & e);

    std::string path_for(const std::string & id) const;
    static size_t file_bytes(const std::string & path);

    struct llama_model *                    model_;
    std::string                             adapter_dir_;
    size_t                                  max_bytes_;       // 0 = unbounded

    std::unordered_map<std::string, entry>  resident_;
    std::list<std::string>                  lru_free_;        // ref==0
    std::list<std::string>                  pinned_;          // ref>0

    size_t                                  current_bytes_ = 0;
    stats_t                                 stats_;
};
