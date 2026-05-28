// Minimal io_uring wrapper for async O_DIRECT reads (Linux/Android).
// Used by cpu-elastic to submit a batch of pread() upfront, let kernel do them
// in parallel, then wait when needed. Bypasses worker-thread scheduling +
// liburing dependency.
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__linux__) || defined(__ANDROID__)

namespace llama_uring {

// init/shutdown global ring (lazy init on first submit)
bool init(unsigned entries = 256);
void shutdown();

// Submit a pread request. dst MUST be 4096-byte aligned, file_offset MUST be
// 4096-byte aligned, len MUST be 4096-multiple. Returns 0 on success, <0 error.
// Caller can submit many before any wait. Then poll/wait.
int submit_pread_aligned(const char *filename, void *dst, size_t file_offset, size_t len);

// Wait for ALL submitted-but-not-yet-completed requests to finish.
// Returns # completed.
int wait_all();

// Stats
struct stats {
    uint64_t n_submitted;
    uint64_t n_completed;
};
stats get_stats();

} // namespace
#endif
