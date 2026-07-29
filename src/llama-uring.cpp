// Minimal io_uring wrapper — raw syscalls (no liburing).
#include "llama-uring.h"

#if defined(__linux__) || defined(__ANDROID__)

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <atomic>
#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

// Raw syscall wrappers (NDK doesn't ship glibc io_uring helpers).
static int sys_io_uring_setup(unsigned entries, struct io_uring_params *p) {
    return (int)syscall(__NR_io_uring_setup, entries, p);
}
static int sys_io_uring_enter(int ring_fd, unsigned to_submit, unsigned min_complete, unsigned flags, void *arg) {
    return (int)syscall(__NR_io_uring_enter, ring_fd, to_submit, min_complete, flags, arg, 0);
}

struct ring_state {
    int ring_fd = -1;
    unsigned sq_entries = 0;
    unsigned cq_entries = 0;
    // SQ
    void *sq_ptr = nullptr;
    size_t sq_size = 0;
    unsigned *sq_head = nullptr;
    unsigned *sq_tail = nullptr;
    unsigned *sq_mask = nullptr;
    unsigned *sq_array = nullptr;
    struct io_uring_sqe *sqes = nullptr;
    size_t sqes_size = 0;
    // CQ
    void *cq_ptr = nullptr;
    size_t cq_size = 0;
    unsigned *cq_head = nullptr;
    unsigned *cq_tail = nullptr;
    unsigned *cq_mask = nullptr;
    struct io_uring_cqe *cqes = nullptr;

    std::mutex mtx;
    uint64_t n_submitted = 0;
    uint64_t n_completed = 0;
    uint64_t n_inflight  = 0;

    // Per-filename fd cache (O_DIRECT). Lazy open on first submit.
    std::unordered_map<std::string, int> fd_cache;
};

ring_state g_ring;

int get_or_open_fd(const std::string &filename) {
    auto it = g_ring.fd_cache.find(filename);
    if (it != g_ring.fd_cache.end()) return it->second;
    int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fprintf(stderr, "[uring] open O_DIRECT failed %s: %s\n", filename.c_str(), strerror(errno));
        return -1;
    }
    g_ring.fd_cache[filename] = fd;
    return fd;
}

} // namespace

namespace llama_uring {

bool init(unsigned entries) {
    std::lock_guard<std::mutex> lk(g_ring.mtx);
    if (g_ring.ring_fd >= 0) return true;

    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    int fd = sys_io_uring_setup(entries, &p);
    if (fd < 0) {
        fprintf(stderr, "[uring] io_uring_setup failed: %s (need kernel >= 5.1)\n", strerror(errno));
        return false;
    }
    g_ring.ring_fd = fd;
    g_ring.sq_entries = p.sq_entries;
    g_ring.cq_entries = p.cq_entries;

    g_ring.sq_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    g_ring.cq_size = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);

    // If kernel has IORING_FEAT_SINGLE_MMAP, SQ and CQ in same mmap.
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
        if (g_ring.cq_size > g_ring.sq_size) g_ring.sq_size = g_ring.cq_size;
        g_ring.cq_size = g_ring.sq_size;
    }

    g_ring.sq_ptr = mmap(0, g_ring.sq_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (g_ring.sq_ptr == MAP_FAILED) {
        fprintf(stderr, "[uring] mmap SQ ring failed: %s\n", strerror(errno));
        close(fd); g_ring.ring_fd = -1; return false;
    }
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
        g_ring.cq_ptr = g_ring.sq_ptr;
    } else {
        g_ring.cq_ptr = mmap(0, g_ring.cq_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (g_ring.cq_ptr == MAP_FAILED) {
            fprintf(stderr, "[uring] mmap CQ ring failed: %s\n", strerror(errno));
            munmap(g_ring.sq_ptr, g_ring.sq_size); close(fd); g_ring.ring_fd = -1; return false;
        }
    }
    g_ring.sqes_size = p.sq_entries * sizeof(struct io_uring_sqe);
    g_ring.sqes = (struct io_uring_sqe *)mmap(0, g_ring.sqes_size, PROT_READ|PROT_WRITE,
                                              MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_SQES);
    if (g_ring.sqes == MAP_FAILED) {
        fprintf(stderr, "[uring] mmap SQES failed: %s\n", strerror(errno));
        return false;
    }

    // Set up SQ pointers (offsets within sq_ptr)
    char *sb = (char*)g_ring.sq_ptr;
    g_ring.sq_head = (unsigned *)(sb + p.sq_off.head);
    g_ring.sq_tail = (unsigned *)(sb + p.sq_off.tail);
    g_ring.sq_mask = (unsigned *)(sb + p.sq_off.ring_mask);
    g_ring.sq_array = (unsigned *)(sb + p.sq_off.array);

    char *cb = (char*)g_ring.cq_ptr;
    g_ring.cq_head = (unsigned *)(cb + p.cq_off.head);
    g_ring.cq_tail = (unsigned *)(cb + p.cq_off.tail);
    g_ring.cq_mask = (unsigned *)(cb + p.cq_off.ring_mask);
    g_ring.cqes    = (struct io_uring_cqe *)(cb + p.cq_off.cqes);

    fprintf(stderr, "[uring] init OK: sq_entries=%u cq_entries=%u\n", g_ring.sq_entries, g_ring.cq_entries);
    return true;
}

void shutdown() {
    std::lock_guard<std::mutex> lk(g_ring.mtx);
    if (g_ring.ring_fd < 0) return;
    if (g_ring.sqes && g_ring.sqes_size) munmap(g_ring.sqes, g_ring.sqes_size);
    if (g_ring.sq_ptr && g_ring.sq_ptr != g_ring.cq_ptr && g_ring.sq_size) munmap(g_ring.sq_ptr, g_ring.sq_size);
    if (g_ring.cq_ptr && g_ring.cq_size) munmap(g_ring.cq_ptr, g_ring.cq_size);
    close(g_ring.ring_fd);
    g_ring.ring_fd = -1;
    for (auto &kv : g_ring.fd_cache) close(kv.second);
    g_ring.fd_cache.clear();
}

int submit_pread_aligned(const char *filename, void *dst, size_t file_offset, size_t len) {
    if (g_ring.ring_fd < 0 && !init()) return -1;
    if (!filename || !dst || len == 0) return -2;
    if (((uintptr_t)dst & 4095) || (file_offset & 4095) || (len & 4095)) {
        fprintf(stderr, "[uring] alignment fail: dst=%p off=%zu len=%zu\n", dst, file_offset, len);
        return -3;
    }
    std::lock_guard<std::mutex> lk(g_ring.mtx);
    int fd = get_or_open_fd(filename);
    if (fd < 0) return -4;

    unsigned tail = __atomic_load_n(g_ring.sq_tail, __ATOMIC_ACQUIRE);
    unsigned head = __atomic_load_n(g_ring.sq_head, __ATOMIC_ACQUIRE);
    if (tail - head >= g_ring.sq_entries) {
        // SQ full → submit + wait some, then retry
        sys_io_uring_enter(g_ring.ring_fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr);
        return -5;
    }
    unsigned idx = tail & *g_ring.sq_mask;
    struct io_uring_sqe *sqe = &g_ring.sqes[idx];
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ;
    sqe->fd     = fd;
    sqe->off    = file_offset;
    sqe->addr   = (__u64)(uintptr_t)dst;
    sqe->len    = (__u32)len;
    sqe->user_data = (__u64)(uintptr_t)dst;

    g_ring.sq_array[idx] = idx;
    __atomic_store_n(g_ring.sq_tail, tail + 1, __ATOMIC_RELEASE);
    g_ring.n_submitted += 1;
    g_ring.n_inflight  += 1;

    // submit (non-blocking, kernel processes async)
    int r = sys_io_uring_enter(g_ring.ring_fd, 1, 0, 0, nullptr);
    if (r < 0) {
        fprintf(stderr, "[uring] enter submit failed: %s\n", strerror(errno));
        return -6;
    }
    return 0;
}

int pread_aligned_batch(const pread_request * requests, size_t count) {
    if (!requests || count == 0) return -2;
    if (g_ring.ring_fd < 0 && !init(static_cast<unsigned>(std::max<size_t>(256, count)))) {
        return -1;
    }

    std::lock_guard<std::mutex> lk(g_ring.mtx);
    if (g_ring.n_inflight != 0 || count > g_ring.sq_entries) return -3;

    const unsigned tail = __atomic_load_n(g_ring.sq_tail, __ATOMIC_ACQUIRE);
    const unsigned head = __atomic_load_n(g_ring.sq_head, __ATOMIC_ACQUIRE);
    if (tail - head + count > g_ring.sq_entries) return -4;

    for (size_t i = 0; i < count; ++i) {
        const pread_request & request = requests[i];
        if (!request.filename || !request.dst || request.len == 0 ||
            (reinterpret_cast<uintptr_t>(request.dst) & 4095U) ||
            (request.file_offset & 4095U) || (request.len & 4095U)) {
            return -5;
        }
        const int fd = get_or_open_fd(request.filename);
        if (fd < 0) return -6;

        const unsigned sq_index = (tail + static_cast<unsigned>(i)) & *g_ring.sq_mask;
        struct io_uring_sqe * sqe = &g_ring.sqes[sq_index];
        memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ;
        sqe->fd = fd;
        sqe->off = request.file_offset;
        sqe->addr = (__u64)(uintptr_t) request.dst;
        sqe->len = (__u32) request.len;
        // Zero is reserved so a malformed completion cannot alias request 0.
        sqe->user_data = static_cast<__u64>(i + 1);
        g_ring.sq_array[sq_index] = sq_index;
    }

    __atomic_store_n(g_ring.sq_tail, tail + static_cast<unsigned>(count), __ATOMIC_RELEASE);
    const int submitted = sys_io_uring_enter(
        g_ring.ring_fd, static_cast<unsigned>(count), 0, 0, nullptr);
    if (submitted != static_cast<int>(count)) {
        if (submitted < 0) {
            // The kernel consumed no SQEs. Make the batch retry/fallback path
            // reusable instead of leaving an artificial full SQ behind.
            __atomic_store_n(g_ring.sq_tail, tail, __ATOMIC_RELEASE);
            fprintf(stderr, "[uring] batch submit failed: %s\n", strerror(errno));
        } else {
            fprintf(stderr, "[uring] short batch submit: %d / %zu\n", submitted, count);
        }
        return -7;
    }
    g_ring.n_submitted += count;
    g_ring.n_inflight += count;

    size_t completed = 0;
    bool ok = true;
    while (completed < count) {
        const int wait_rc = sys_io_uring_enter(
            g_ring.ring_fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr);
        if (wait_rc < 0) {
            fprintf(stderr, "[uring] batch wait failed: %s\n", strerror(errno));
            ok = false;
            break;
        }
        unsigned cq_head = __atomic_load_n(g_ring.cq_head, __ATOMIC_ACQUIRE);
        const unsigned cq_tail = __atomic_load_n(g_ring.cq_tail, __ATOMIC_ACQUIRE);
        while (cq_head != cq_tail && completed < count) {
            const struct io_uring_cqe * cqe = &g_ring.cqes[cq_head & *g_ring.cq_mask];
            const size_t request_index = cqe->user_data > 0
                ? static_cast<size_t>(cqe->user_data - 1) : count;
            if (request_index >= count ||
                cqe->res != static_cast<int>(requests[request_index].len)) {
                fprintf(stderr,
                        "[uring] batch completion mismatch: request=%zu res=%d expected=%zu\n",
                        request_index, cqe->res,
                        request_index < count ? requests[request_index].len : 0);
                ok = false;
            }
            ++cq_head;
            ++completed;
            --g_ring.n_inflight;
            ++g_ring.n_completed;
        }
        __atomic_store_n(g_ring.cq_head, cq_head, __ATOMIC_RELEASE);
    }
    return ok && completed == count ? 0 : -8;
}

int wait_all() {
    std::lock_guard<std::mutex> lk(g_ring.mtx);
    if (g_ring.ring_fd < 0) return 0;
    int n_completed_this = 0;
    while (g_ring.n_inflight > 0) {
        // wait at least 1 completion
        sys_io_uring_enter(g_ring.ring_fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr);
        unsigned chead = __atomic_load_n(g_ring.cq_head, __ATOMIC_ACQUIRE);
        unsigned ctail = __atomic_load_n(g_ring.cq_tail, __ATOMIC_ACQUIRE);
        while (chead != ctail) {
            struct io_uring_cqe *cqe = &g_ring.cqes[chead & *g_ring.cq_mask];
            if (cqe->res < 0) {
                fprintf(stderr, "[uring] CQE err: res=%d\n", cqe->res);
            }
            chead++;
            n_completed_this++;
            g_ring.n_inflight--;
            g_ring.n_completed++;
        }
        __atomic_store_n(g_ring.cq_head, chead, __ATOMIC_RELEASE);
    }
    return n_completed_this;
}

stats get_stats() {
    std::lock_guard<std::mutex> lk(g_ring.mtx);
    return { g_ring.n_submitted, g_ring.n_completed };
}

int wait_n(int n_to_wait) {
    std::lock_guard<std::mutex> lk(g_ring.mtx);
    if (g_ring.ring_fd < 0) return 0;
    int drained = 0;
    while (drained < n_to_wait && g_ring.n_inflight > 0) {
        sys_io_uring_enter(g_ring.ring_fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr);
        unsigned chead = __atomic_load_n(g_ring.cq_head, __ATOMIC_ACQUIRE);
        unsigned ctail = __atomic_load_n(g_ring.cq_tail, __ATOMIC_ACQUIRE);
        while (chead != ctail && drained < n_to_wait) {
            struct io_uring_cqe *cqe = &g_ring.cqes[chead & *g_ring.cq_mask];
            if (cqe->res < 0) fprintf(stderr, "[uring] CQE err: res=%d\n", cqe->res);
            chead++;
            drained++;
            g_ring.n_inflight--;
            g_ring.n_completed++;
        }
        __atomic_store_n(g_ring.cq_head, chead, __ATOMIC_RELEASE);
    }
    return drained;
}

int inflight() {
    std::lock_guard<std::mutex> lk(g_ring.mtx);
    return (int)g_ring.n_inflight;
}

} // namespace llama_uring
#endif
