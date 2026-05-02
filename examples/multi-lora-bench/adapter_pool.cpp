#include "adapter_pool.h"

#include <chrono>
#include <stdexcept>
#include <utility>

multilora_adapter_pool::multilora_adapter_pool(struct llama_model * model,
                                               std::string          adapter_dir,
                                               size_t               max_resident)
    : model_(model), adapter_dir_(std::move(adapter_dir)), max_resident_(max_resident) {
    if (max_resident_ == 0) {
        throw std::invalid_argument("multilora_adapter_pool: max_resident must be >= 1");
    }
}

multilora_adapter_pool::~multilora_adapter_pool() {
    shutdown();
}

void multilora_adapter_pool::shutdown() {
    for (auto & kv : resident_) {
        if (kv.second.adapter) {
            llama_adapter_lora_free(kv.second.adapter);
            kv.second.adapter = nullptr;
        }
    }
    resident_.clear();
    lru_order_.clear();
}

std::string multilora_adapter_pool::path_for(const std::string & id) const {
    // Convention: <adapter_dir>/<adapter_id>.gguf. If the id already contains
    // a slash or ends in .gguf, treat it as a literal path so callers can
    // pass either an id or a fully-qualified path.
    const bool has_slash = id.find('/') != std::string::npos;
    const bool has_gguf  = id.size() >= 5 && id.compare(id.size() - 5, 5, ".gguf") == 0;
    if (has_slash || has_gguf) {
        return id;
    }
    std::string p = adapter_dir_;
    if (!p.empty() && p.back() != '/') {
        p += '/';
    }
    p += id;
    p += ".gguf";
    return p;
}

bool multilora_adapter_pool::evict_one_lru() {
    // Walk LRU back-to-front; skip pinned (ref_count > 0) entries. If every
    // resident adapter is currently in use, give up — the caller (acquire)
    // will then exceed the cap rather than block, and stats_.evicts records
    // 0 for this attempt.
    for (auto it = lru_order_.rbegin(); it != lru_order_.rend(); ++it) {
        auto entry_it = resident_.find(*it);
        if (entry_it == resident_.end()) {
            continue;  // stale entry; skip defensively
        }
        if (entry_it->second.ref_count > 0) {
            continue;
        }
        if (entry_it->second.adapter) {
            llama_adapter_lora_free(entry_it->second.adapter);
        }
        // Convert reverse iterator to forward and erase.
        auto fwd = std::next(it).base();
        lru_order_.erase(fwd);
        resident_.erase(entry_it);
        ++stats_.evicts;
        return true;
    }
    return false;
}

multilora_adapter_pool::acquire_result
multilora_adapter_pool::acquire(struct llama_context * ctx, const std::string & id) {
    using clock = std::chrono::steady_clock;
    acquire_result result;

    auto it = resident_.find(id);
    if (it != resident_.end()) {
        // Hit: move to MRU, bump ref, set on context.
        ++stats_.hits;
        lru_order_.erase(it->second.lru_it);
        lru_order_.push_front(id);
        it->second.lru_it = lru_order_.begin();
        ++it->second.ref_count;
        result.handle    = it->second.adapter;
        result.cache_hit = true;
    } else {
        // Miss: ensure room, then load.
        ++stats_.misses;
        while (resident_.size() >= max_resident_) {
            if (!evict_one_lru()) {
                break;  // every entry pinned; we'll temporarily exceed the cap
            }
        }

        const auto load_t0 = clock::now();
        struct llama_adapter_lora * h = llama_adapter_lora_init(model_, path_for(id).c_str());
        const auto load_t1 = clock::now();
        if (!h) {
            throw std::runtime_error("multilora_adapter_pool::acquire: llama_adapter_lora_init failed for id='" + id + "' path='" + path_for(id) + "'");
        }
        result.load_ms = std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();

        lru_order_.push_front(id);
        entry e;
        e.id        = id;
        e.adapter   = h;
        e.ref_count = 1;
        e.lru_it    = lru_order_.begin();
        auto inserted = resident_.emplace(id, std::move(e));
        result.handle    = inserted.first->second.adapter;
        result.cache_hit = false;
    }

    // Set this single adapter as the active set on the context. The new API
    // is replace-semantics: passing 1 entry clears any previously-active set.
    struct llama_adapter_lora * adapters[1] = { result.handle };
    float                       scales[1]   = { 1.0f };
    int rc = llama_set_adapters_lora(ctx, adapters, 1, scales);
    if (rc != 0) {
        // Roll back the ref bump so release(id) is balanced. The adapter
        // stays loaded — the bind-to-ctx step is what failed.
        auto it2 = resident_.find(id);
        if (it2 != resident_.end() && it2->second.ref_count > 0) {
            --it2->second.ref_count;
        }
        throw std::runtime_error("multilora_adapter_pool::acquire: llama_set_adapters_lora returned " + std::to_string(rc));
    }

    stats_.resident = resident_.size();
    return result;
}

void multilora_adapter_pool::release(const std::string & id) {
    auto it = resident_.find(id);
    if (it == resident_.end()) {
        return;  // unknown id; silent — release should be tolerant
    }
    if (it->second.ref_count > 0) {
        --it->second.ref_count;
    }
}
