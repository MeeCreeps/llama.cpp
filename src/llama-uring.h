// Minimal io_uring wrapper for async O_DIRECT reads (Linux/Android).
// Used by cpu-elastic to submit a batch of pread() upfront, let kernel do them
// in parallel, then wait when needed. Bypasses worker-thread scheduling +
// liburing dependency.
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__linux__) || defined(__ANDROID__)

namespace llama_uring {

struct pread_request {
    const char * filename = nullptr;
    void * dst = nullptr;
    size_t file_offset = 0;
    size_t len = 0;
};

// init/shutdown global ring (lazy init on first submit)
bool init(unsigned entries = 256);
void shutdown();

// Submit a pread request. dst MUST be 4096-byte aligned, file_offset MUST be
// 4096-byte aligned, len MUST be 4096-multiple. Returns 0 on success, <0 error.
// Caller can submit many before any wait. Then poll/wait.
int submit_pread_aligned(const char *filename, void *dst, size_t file_offset, size_t len);

// Submit an independent-offset batch with one io_uring_enter() and wait for
// the whole batch. Every request has the same alignment requirements as
// submit_pread_aligned(). This synchronous wrapper is intended to run on a
// LOAD worker while CPU compute proceeds on another thread. Returns 0 only
// when every request completed with the requested byte count.
int pread_aligned_batch(const pread_request * requests, size_t count);

// Wait for ALL submitted-but-not-yet-completed requests to finish.
// Returns # completed.
int wait_all();

// Wait until at least N completions have arrived since last drain. Lets caller
// submit more requests after this returns while N+ are still pending in kernel.
// Used for multi-chunk lookahead pipelining.
int wait_n(int n_to_wait);

// Current in-flight count (submitted but not completed).
int inflight();

// Stats
struct stats {
    uint64_t n_submitted;
    uint64_t n_completed;
};
stats get_stats();

} // namespace
#endif
