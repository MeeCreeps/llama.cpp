#pragma once

#include "llama.h"

#include <cstddef>
#include <string>
#include <vector>

// One pre-tokenized trace request (loaded from JSON in main.cpp). Lives for
// the duration of its slot once admitted; Scheduler holds it by value inside
// the slot.
struct multilora_request {
    std::string              id;
    double                   arrival_time = 0.0;
    std::string              adapter_id;
    std::vector<llama_token> input_tokens;
    int                      max_output = 128;
};

// Active in-flight request inside Scheduler. Single-threaded by design;
// Scheduler is the sole writer of these fields.
struct multilora_slot {
    multilora_request        req;
    llama_seq_id             seq_id           = 0;     // unique while slot is active
    std::vector<llama_token> output;                    // sampled tokens, in order
    size_t                   prefill_done     = 0;      // # prompt tokens already fed
    bool                     first_token_recorded = false;
    double                   first_token_time = 0.0;    // seconds since experiment t0

    // Set by Scheduler::step each iteration. Index into the output-logits
    // buffer at which this slot's "next" sample lives, i.e. the j passed to
    // llama_sampler_sample / llama_get_logits_ith. -1 means "did not fit
    // into this step's batch; try again next step". (No starvation
    // mitigation in M2 per spec section 4 M2; see known-issues.md I-1.)
    int32_t                  sample_idx       = -1;

    // Captured from AdapterPool::acquire on the FIRST step that picks this
    // slot's adapter. Replays M1's per-request metric semantics: cold/hot
    // is a property of the request, not of a single batch step.
    bool                     first_acquire_recorded = false;
    bool                     acquire_cache_hit      = false;
    double                   acquire_ms             = 0.0;
};

inline bool multilora_slot_in_prefill(const multilora_slot & s) {
    return s.prefill_done < s.req.input_tokens.size();
}
