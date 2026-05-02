#include "scheduler.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <utility>

multilora_scheduler::multilora_scheduler(struct llama_context *                ctx,
                                         const struct llama_vocab *            vocab,
                                         struct llama_sampler *                smpl,
                                         multilora_adapter_pool *              pool,
                                         multilora_metrics *                   metrics,
                                         size_t                                max_slots,
                                         int32_t                               n_batch,
                                         std::chrono::steady_clock::time_point t0)
    : ctx_(ctx), vocab_(vocab), smpl_(smpl), pool_(pool), metrics_(metrics),
      max_slots_(max_slots), n_batch_(n_batch), t0_(t0) {
    if (max_slots_ == 0) {
        throw std::invalid_argument("multilora_scheduler: max_slots must be >= 1");
    }
    if (n_batch_ <= 0) {
        throw std::invalid_argument("multilora_scheduler: n_batch must be >= 1");
    }
    // Allocate a reusable batch; n_seq_max=1 because every token belongs to
    // exactly one slot (each slot has its own seq_id and we never alias).
    batch_ = llama_batch_init(n_batch_, /*embd=*/0, /*n_seq_max=*/1);

    // Pre-fill the seq_id free list with [0, max_slots).
    free_seq_ids_.reserve(max_slots_);
    for (size_t i = 0; i < max_slots_; ++i) {
        free_seq_ids_.push_back(static_cast<llama_seq_id>(i));
    }
}

multilora_scheduler::~multilora_scheduler() {
    llama_batch_free(batch_);
}

llama_seq_id multilora_scheduler::alloc_seq_id() {
    if (free_seq_ids_.empty()) {
        throw std::runtime_error("multilora_scheduler: out of seq_ids (admit beyond max_slots)");
    }
    llama_seq_id id = free_seq_ids_.back();
    free_seq_ids_.pop_back();
    return id;
}

void multilora_scheduler::free_seq_id(llama_seq_id id) {
    free_seq_ids_.push_back(id);
}

void multilora_scheduler::admit(multilora_request req) {
    if (active_.size() >= max_slots_) {
        // Caller contract violation; surface clearly rather than silently dropping.
        throw std::runtime_error("multilora_scheduler::admit: exceeds max_slots");
    }
    auto s = std::make_unique<multilora_slot>();
    s->req     = std::move(req);
    s->seq_id  = alloc_seq_id();
    active_.push_back(std::move(s));
}

void multilora_scheduler::erase_active(multilora_slot * s) {
    auto it = std::find_if(active_.begin(), active_.end(),
        [&](const std::unique_ptr<multilora_slot> & up) { return up.get() == s; });
    if (it == active_.end()) {
        return;
    }
    free_seq_id(s->seq_id);
    active_.erase(it);
}

void multilora_scheduler::clear_batch() {
    batch_.n_tokens = 0;
}

double multilora_scheduler::seconds_since_t0() const {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_).count();
}

void multilora_scheduler::batch_add(llama_token tok, llama_pos pos,
                                    llama_seq_id seq, bool with_logits) {
    const int32_t i = batch_.n_tokens;
    batch_.token   [i] = tok;
    batch_.pos     [i] = pos;
    batch_.n_seq_id[i] = 1;
    batch_.seq_id  [i][0] = seq;
    batch_.logits  [i] = with_logits ? 1 : 0;
    batch_.n_tokens = i + 1;
}

std::vector<multilora_slot *> multilora_scheduler::pick_largest_adapter_group() {
    // Bucket by adapter while remembering each adapter's first-appearance
    // index in active_ (admit order). The chosen group is the largest
    // bucket; ties broken by smallest first-appearance index ("oldest
    // slot first" per spec section 5.2). We must NOT rely on unordered_map
    // iteration order — without an explicit tie-break the schedule is
    // non-deterministic and same-trace runs diverge.
    std::unordered_map<std::string, std::vector<multilora_slot *>> groups;
    std::unordered_map<std::string, size_t>                         first_idx;
    for (size_t i = 0; i < active_.size(); ++i) {
        auto & up  = active_[i];
        auto & vec = groups[up->req.adapter_id];
        if (vec.empty()) {
            first_idx[up->req.adapter_id] = i;
        }
        vec.push_back(up.get());
    }

    const std::string * best_id    = nullptr;
    size_t              best_size  = 0;
    size_t              best_first = static_cast<size_t>(-1);
    for (auto & kv : groups) {
        const size_t sz = kv.second.size();
        const size_t fi = first_idx[kv.first];
        if (sz > best_size || (sz == best_size && fi < best_first)) {
            best_id    = &kv.first;
            best_size  = sz;
            best_first = fi;
        }
    }
    if (!best_id) {
        return {};
    }
    return groups[*best_id];
}

void multilora_scheduler::finalize_and_record(multilora_slot * s, double now_s) {
    multilora_request_metric m;
    m.id               = s->req.id;
    m.adapter_id       = s->req.adapter_id;
    m.arrival_time     = s->req.arrival_time;
    m.first_token_time = s->first_token_recorded ? s->first_token_time : now_s;
    m.finish_time      = now_s;
    m.n_input_tokens   = s->req.input_tokens.size();
    m.n_output_tokens  = s->output.size();
    m.cache_hit        = s->acquire_cache_hit;
    m.acquire_ms       = s->acquire_ms;
    metrics_->record(std::move(m));

    // Optional: dump detokenized output text for byte-level cross-config
    // diff (see docs/multi-lora/IMPLEMENTATION_GUIDE.md section 9
    // verification policy).
    if (!output_dir_.empty()) {
        std::string text;
        text.reserve(8 * s->output.size());
        char buf[256];
        for (llama_token tok : s->output) {
            const int n = llama_token_to_piece(vocab_, tok, buf, sizeof(buf),
                                               /*lstrip=*/0, /*special=*/false);
            if (n > 0) {
                text.append(buf, static_cast<size_t>(n));
            }
        }
        const std::string path = output_dir_ + "/" + s->req.id + ".txt";
        std::FILE * f = std::fopen(path.c_str(), "wb");
        if (f) {
            std::fwrite(text.data(), 1, text.size(), f);
            std::fclose(f);
        } else {
            std::fprintf(stderr, "scheduler: failed to open '%s' for output dump\n",
                         path.c_str());
        }
    }

    // Free this slot's KV so the next admit on the same seq_id starts clean.
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_seq_rm(mem, s->seq_id, -1, -1);
}

bool multilora_scheduler::step() {
    if (active_.empty()) {
        return false;
    }

    auto group = pick_largest_adapter_group();
    if (group.empty()) {
        return !active_.empty();
    }

    // Bind the group's adapter on the context (replace-semantics).
    multilora_adapter_pool::acquire_result acq;
    try {
        acq = pool_->acquire(ctx_, group.front()->req.adapter_id);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "scheduler: pool.acquire('%s') failed: %s; dropping group\n",
                     group.front()->req.adapter_id.c_str(), e.what());
        const double now_s = seconds_since_t0();
        // Drop every slot in the group rather than spin-loop forever.
        for (auto * s : group) {
            finalize_and_record(s, now_s);
            erase_active(s);
        }
        return !active_.empty();
    }

    // Stamp first-acquire metric on each slot that hasn't seen one yet.
    for (auto * s : group) {
        if (!s->first_acquire_recorded) {
            s->acquire_cache_hit      = acq.cache_hit;
            s->acquire_ms             = acq.load_ms;
            s->first_acquire_recorded = true;
        }
    }

    // Build the batch. For each slot in the group:
    //   * still-prefilling slot -> feed its remaining prompt tokens, last
    //     with logits=1 so we can sample the first decode token afterwards
    //   * decoding slot -> feed its last-sampled token with logits=1
    // Slots whose contribution would overflow n_batch are skipped this
    // step (sample_idx stays at -1).
    //
    // sample_idx is the BATCH POSITION (index into batch_.logits[]) of the
    // logits-bearing token, not a running count of logits-bearing tokens.
    // llama_sampler_sample(smpl, ctx, idx) requires batch.logits[idx] == 1
    // and aborts otherwise. (See docs/multi-lora/api-versions.md.)
    clear_batch();
    for (auto * s : group) {
        s->sample_idx = -1;

        if (multilora_slot_in_prefill(*s)) {
            const int32_t prompt_len = static_cast<int32_t>(s->req.input_tokens.size());
            const int32_t start      = static_cast<int32_t>(s->prefill_done);
            const int32_t to_feed    = prompt_len - start;
            if (to_feed <= 0) {
                continue;  // defensive — should not happen
            }
            if (batch_.n_tokens + to_feed > n_batch_) {
                // Doesn't fit this step. M2 doesn't chunk prefill across
                // steps; either pick this slot first next round (admit
                // order keeps the same group together) or trip an error.
                if (batch_.n_tokens == 0) {
                    std::fprintf(stderr,
                        "scheduler: request '%s' has %d-token prompt > n_batch=%d; aborting request\n",
                        s->req.id.c_str(), to_feed, n_batch_);
                    finalize_and_record(s, seconds_since_t0());
                    erase_active(s);
                    pool_->release(group.front()->req.adapter_id);
                    return !active_.empty();
                }
                continue;
            }
            for (int32_t k = 0; k < to_feed; ++k) {
                const bool is_last = (k == to_feed - 1);
                batch_add(s->req.input_tokens[start + k],
                          /*pos=*/static_cast<llama_pos>(start + k),
                          s->seq_id, is_last);
                if (is_last) {
                    s->sample_idx = batch_.n_tokens - 1;  // batch position
                }
            }
        } else {
            // Decode step: feed the last-sampled token, expect new logits.
            if (s->output.empty()) {
                continue;  // invariant violation guard
            }
            const llama_token last = s->output.back();
            const llama_pos   pos  = static_cast<llama_pos>(
                s->req.input_tokens.size() + s->output.size() - 1);
            if (batch_.n_tokens + 1 > n_batch_) {
                continue;  // try next step
            }
            batch_add(last, pos, s->seq_id, /*with_logits=*/true);
            s->sample_idx = batch_.n_tokens - 1;  // batch position
        }
    }

    if (batch_.n_tokens == 0) {
        // Nothing fit at all (shouldn't happen given size guards above).
        pool_->release(group.front()->req.adapter_id);
        return !active_.empty();
    }

    const int32_t rc = llama_decode(ctx_, batch_);
    if (rc != 0) {
        std::fprintf(stderr,
            "scheduler: llama_decode failed (rc=%d, n_tokens=%d, group=%zu); aborting group\n",
            rc, batch_.n_tokens, group.size());
        const double now_s = seconds_since_t0();
        for (auto * s : group) {
            finalize_and_record(s, now_s);
            erase_active(s);
        }
        pool_->release(group.front()->req.adapter_id);
        return !active_.empty();
    }

    // Read the wall clock AFTER llama_decode returns. This is the
    // earliest moment the first decoded token is observable, which is
    // what we want stamped as first_token_time / finish_time (matches
    // M1 semantics — see spec section 1.1.2).
    const double sample_now = seconds_since_t0();

    // Sample one new token per slot whose sample_idx >= 0.
    for (auto * s : group) {
        if (s->sample_idx < 0) {
            continue;
        }

        const bool was_in_prefill = multilora_slot_in_prefill(*s);
        if (was_in_prefill) {
            // We just finished prefill in this step.
            s->prefill_done = s->req.input_tokens.size();
        }

        const llama_token tok = llama_sampler_sample(smpl_, ctx_, s->sample_idx);

        if (!s->first_token_recorded) {
            s->first_token_time     = sample_now;
            s->first_token_recorded = true;
        }

        if (llama_vocab_is_eog(vocab_, tok)) {
            finalize_and_record(s, sample_now);
            erase_active(s);
            continue;
        }
        s->output.push_back(tok);
        if (static_cast<int>(s->output.size()) >= s->req.max_output) {
            finalize_and_record(s, sample_now);
            erase_active(s);
        }
    }

    pool_->release(group.front()->req.adapter_id);
    return !active_.empty();
}
