#pragma once

#include "adapter_pool.h"
#include "metrics.h"
#include "slot.h"

#include "llama.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// Scheduler: drives the M2 main-loop step. Owns the active slot list, the
// llama_batch scratch buffer, and a small free-list of seq_ids. Single-
// threaded; the spec section 1.1 architecture forbids threads/futures.
//
// Algorithm (per step):
//   1. Group active slots by adapter_id, pick the largest group
//      (tie-break: oldest admit wins).
//   2. Acquire that adapter on the context (replace-semantics via
//      llama_set_adapters_lora; AdapterPool handles cache + LRU).
//   3. Build a llama_batch: each slot in the picked group contributes
//      either its remaining prefill prompt tokens (last with logits=1)
//      or its last sampled token (logits=1) for the decode step.
//   4. llama_decode the batch.
//   5. For each slot in the group, sample its next token using its
//      recorded sample_idx, advance prefill_done or append to output,
//      finalize on EOG / max_output, release adapter ref.
//
// Slots whose tokens did not fit in this step's batch (sample_idx == -1)
// are simply skipped; the next step picks them up. M2 deliberately does
// not enforce starvation guarantees.
class multilora_scheduler {
public:
    multilora_scheduler(struct llama_context *                          ctx,
                        const struct llama_vocab *                      vocab,
                        struct llama_sampler *                          smpl,
                        multilora_adapter_pool *                        pool,
                        multilora_metrics *                             metrics,
                        size_t                                          max_slots,
                        int32_t                                         n_batch,
                        std::chrono::steady_clock::time_point           t0);

    // If non-empty, every finalized request's detokenized output is written
    // to <dir>/<req_id>.txt. Used for byte-level content-equality checks
    // across configurations (see docs/multi-lora/IMPLEMENTATION_GUIDE.md
    // section 9 verification policy). The directory must already exist.
    void set_output_dir(std::string dir) { output_dir_ = std::move(dir); }

    ~multilora_scheduler();

    multilora_scheduler(const multilora_scheduler &)             = delete;
    multilora_scheduler & operator=(const multilora_scheduler &) = delete;

    // Admit a newly-arrived request. Caller is responsible for honoring
    // the max_slots cap (admit() does NOT block or reject; main.cpp
    // checks active_count() < max_slots before calling).
    void admit(multilora_request req);

    // Run one decode step. Returns true if there is still work pending.
    // The scheduler reads the clock relative to its own t0 to stamp
    // first_token_time / finish_time at the actual sample point (post
    // llama_decode), matching M1 semantics.
    bool step();

    size_t active_count() const { return active_.size(); }

    // Pool reference for end-of-run stats reporting in main.
    multilora_adapter_pool * pool() const { return pool_; }

private:
    // Returns active slots for the largest current adapter group; stable
    // ordering by admit-order to keep the prefill / decode position
    // deterministic across steps.
    std::vector<multilora_slot *> pick_largest_adapter_group();

    void clear_batch();
    void batch_add(llama_token tok, llama_pos pos, llama_seq_id seq, bool with_logits);

    double seconds_since_t0() const;
    void finalize_and_record(multilora_slot * s, double now_s);
    void erase_active(multilora_slot * s);

    llama_seq_id alloc_seq_id();
    void         free_seq_id(llama_seq_id);

    struct llama_context *           ctx_;
    const struct llama_vocab *       vocab_;
    struct llama_sampler *           smpl_;
    multilora_adapter_pool *         pool_;
    multilora_metrics *              metrics_;
    size_t                           max_slots_;
    int32_t                          n_batch_;

    std::chrono::steady_clock::time_point        t0_;
    std::string                                  output_dir_;

    std::vector<std::unique_ptr<multilora_slot>> active_;
    std::vector<llama_seq_id>                    free_seq_ids_;  // LIFO

    llama_batch                                  batch_;
};
