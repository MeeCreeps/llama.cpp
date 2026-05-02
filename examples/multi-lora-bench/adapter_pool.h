#pragma once

#include "llama.h"

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>

// AdapterPool: count-based LRU cache of llama_adapter_lora handles.
//
// M1 scope per docs/multi-lora/IMPLEMENTATION_GUIDE.md section 4 M1:
//   - At most max_resident adapters live in the cache simultaneously.
//   - acquire(ctx, id) loads the adapter on miss, sets it as the only active
//     LoRA on the context, bumps the ref-count, marks it MRU, and returns
//     the handle. The set-on-context step uses llama_set_adapters_lora,
//     which replaces (does not stack with) any currently active LoRA.
//   - release(id) decrements the ref-count; eviction is deferred to the
//     next miss. Adapters with ref_count > 0 are pinned and skipped during
//     eviction so they cannot be freed while a slot is mid-decode.
//
// M3 will replace max_resident (count) with max_bytes (byte budget); the
// public surface here keeps the door open for that without forcing it now.
//
// Single-threaded by design: no mutex. The main loop is the sole caller
// per CLAUDE.md "single-thread main loop is the design".
class multilora_adapter_pool {
public:
    struct stats_t {
        size_t hits     = 0;
        size_t misses   = 0;
        size_t evicts   = 0;
        size_t resident = 0;
    };

    multilora_adapter_pool(struct llama_model * model,
                           std::string          adapter_dir,
                           size_t               max_resident);

    ~multilora_adapter_pool();

    multilora_adapter_pool(const multilora_adapter_pool &)             = delete;
    multilora_adapter_pool & operator=(const multilora_adapter_pool &) = delete;

    // Result of a single acquire. Caller uses cache_hit + load_ms for metrics.
    struct acquire_result {
        struct llama_adapter_lora * handle   = nullptr;
        bool                        cache_hit = false;
        double                      load_ms   = 0.0;  // 0 on hit
    };

    // Resolve, load if needed, set as the sole active LoRA on ctx, return handle.
    // Throws std::runtime_error on failure (file missing, init failed, OOM).
    acquire_result acquire(struct llama_context * ctx, const std::string & id);

    // Decrement ref-count; entry stays resident until evicted by LRU.
    void release(const std::string & id);

    // Free every cached adapter and forget all state. Call before model_free.
    void shutdown();

    stats_t stats() const { return stats_; }

private:
    struct entry {
        std::string                 id;
        struct llama_adapter_lora * adapter   = nullptr;
        int                         ref_count = 0;
        std::list<std::string>::iterator lru_it;  // position in lru_order_
    };

    // Free one adapter that is not currently referenced; returns true if
    // anything was evicted. LRU end is the eviction candidate; we walk
    // backwards to find the first ref_count == 0 entry.
    bool evict_one_lru();

    std::string path_for(const std::string & id) const;

    struct llama_model *                    model_;
    std::string                             adapter_dir_;
    size_t                                  max_resident_;

    std::unordered_map<std::string, entry> resident_;
    std::list<std::string>                  lru_order_;  // front = MRU, back = LRU

    stats_t                                 stats_;
};
