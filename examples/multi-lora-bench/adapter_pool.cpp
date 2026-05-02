#include "adapter_pool.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <sys/stat.h>
#include <utility>

multilora_adapter_pool::multilora_adapter_pool(struct llama_model * model,
                                               std::string          adapter_dir,
                                               size_t               max_bytes)
    : model_(model), adapter_dir_(std::move(adapter_dir)), max_bytes_(max_bytes) {}

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
    lru_free_.clear();
    pinned_.clear();
    current_bytes_ = 0;
}

std::string multilora_adapter_pool::path_for(const std::string & id) const {
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

size_t multilora_adapter_pool::file_bytes(const std::string & path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        return 0;
    }
    return static_cast<size_t>(st.st_size);
}

// list/ref movement primitives: maintain the invariant
//     in_pinned == (ref_count > 0)
// for every resident entry, and list_it points to the entry in whichever
// list it currently lives.
void multilora_adapter_pool::detach_from_current_list(entry & e) {
    if (e.in_pinned) {
        pinned_.erase(e.list_it);
    } else {
        lru_free_.erase(e.list_it);
    }
}

void multilora_adapter_pool::attach_to_pinned(entry & e) {
    pinned_.push_front(e.id);
    e.list_it   = pinned_.begin();
    e.in_pinned = true;
}

void multilora_adapter_pool::attach_to_lru_free(entry & e) {
    // MRU end is front; new arrivals at front, evictions from back.
    lru_free_.push_front(e.id);
    e.list_it   = lru_free_.begin();
    e.in_pinned = false;
}

void multilora_adapter_pool::evict_until(size_t bytes_needed) {
    if (max_bytes_ == 0) {
        return;  // unbounded
    }
    while (current_bytes_ + bytes_needed > max_bytes_ && !lru_free_.empty()) {
        const std::string victim_id = lru_free_.back();
        auto it = resident_.find(victim_id);
        if (it == resident_.end()) {
            // defensive: stale entry; drop it from the list and move on
            lru_free_.pop_back();
            continue;
        }
        if (it->second.adapter) {
            llama_adapter_lora_free(it->second.adapter);
        }
        current_bytes_ -= it->second.bytes;
        lru_free_.pop_back();
        resident_.erase(it);
        ++stats_.evicts;
    }
    // If we still don't fit (everything left is pinned), give up cleanly.
    // The caller will load anyway; cache will exceed max_bytes_ until
    // some pin releases. peak_bytes records the high-water mark.
    if (max_bytes_ != 0 && current_bytes_ + bytes_needed > max_bytes_) {
        std::fprintf(stderr,
            "adapter_pool: cannot fit %zu more bytes within budget %zu (current=%zu, "
            "all remaining entries are pinned); proceeding over-budget\n",
            bytes_needed, max_bytes_, current_bytes_);
    }
}

multilora_adapter_pool::acquire_result
multilora_adapter_pool::acquire(struct llama_context * ctx, const std::string & id) {
    using clock = std::chrono::steady_clock;
    acquire_result result;

    auto it = resident_.find(id);
    if (it != resident_.end()) {
        ++stats_.hits;
        entry & e = it->second;
        if (e.ref_count == 0) {
            // Was in lru_free_; promote to pinned now that someone uses it.
            detach_from_current_list(e);
            attach_to_pinned(e);
        }
        ++e.ref_count;
        result.handle    = e.adapter;
        result.cache_hit = true;
        result.bytes     = e.bytes;
    } else {
        ++stats_.misses;
        const std::string path  = path_for(id);
        const size_t      bytes = file_bytes(path);
        if (bytes == 0) {
            throw std::runtime_error("multilora_adapter_pool::acquire: stat failed for '" + path + "'");
        }
        evict_until(bytes);

        const auto load_t0 = clock::now();
        struct llama_adapter_lora * h = llama_adapter_lora_init(model_, path.c_str());
        const auto load_t1 = clock::now();
        if (!h) {
            throw std::runtime_error("multilora_adapter_pool::acquire: llama_adapter_lora_init failed for id='" + id + "' path='" + path + "'");
        }
        result.load_ms = std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();
        result.bytes   = bytes;

        entry e;
        e.id        = id;
        e.adapter   = h;
        e.bytes     = bytes;
        e.ref_count = 1;
        auto inserted = resident_.emplace(id, std::move(e));
        attach_to_pinned(inserted.first->second);
        current_bytes_ += bytes;
        if (current_bytes_ > stats_.peak_bytes) {
            stats_.peak_bytes = current_bytes_;
        }
        result.handle    = inserted.first->second.adapter;
        result.cache_hit = false;
    }

    // Bind as the sole active LoRA on the context (replace-semantics).
    struct llama_adapter_lora * adapters[1] = { result.handle };
    float                       scales[1]   = { 1.0f };
    int rc = llama_set_adapters_lora(ctx, adapters, 1, scales);
    if (rc != 0) {
        // Roll back the ref bump so release(id) is balanced.
        auto it2 = resident_.find(id);
        if (it2 != resident_.end()) {
            entry & e = it2->second;
            if (e.ref_count > 0) {
                --e.ref_count;
                if (e.ref_count == 0) {
                    detach_from_current_list(e);
                    attach_to_lru_free(e);
                }
            }
        }
        throw std::runtime_error("multilora_adapter_pool::acquire: llama_set_adapters_lora returned " + std::to_string(rc));
    }

    stats_.resident_count = resident_.size();
    stats_.resident_bytes = current_bytes_;
    return result;
}

void multilora_adapter_pool::release(const std::string & id) {
    auto it = resident_.find(id);
    if (it == resident_.end()) {
        return;
    }
    entry & e = it->second;
    if (e.ref_count > 0) {
        --e.ref_count;
    }
    if (e.ref_count == 0 && e.in_pinned) {
        detach_from_current_list(e);
        attach_to_lru_free(e);
    }
}
