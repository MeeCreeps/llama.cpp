// io_bench.cpp — 测 mmap (page cache) vs O_DIRECT (真 disk) 读 GGUF 文件的带宽
// 用法: ./io_bench <gguf_file> <read_chunk_MB>
//   - mmap mode: 每次随机偏移 read_chunk_MB
//   - direct mode: 同上但用 O_DIRECT
//   - 第一次跑 mmap = cold cache, 测真 disk
//   - 第二次跑 mmap = warm cache, 测 RAM
//   - direct mode 每次都是 disk

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <file> <chunk_MB> [n_iter]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    size_t chunk = atoi(argv[2]) * 1024UL * 1024;
    int n_iter = argc >= 4 ? atoi(argv[3]) : 8;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t fsize = st.st_size;
    size_t blk_size = st.st_blksize;
    if (blk_size < 4096 || (blk_size & (blk_size - 1))) blk_size = 4096;
    printf("file=%s size=%.1f MB blk_size=%zu chunk=%.1f MB n_iter=%d\n",
           path, fsize / 1024.0 / 1024.0, blk_size, chunk / 1024.0 / 1024.0, n_iter);

    // === mmap mode (warm via second pass) ===
    void *mmp = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmp == MAP_FAILED) { perror("mmap"); return 1; }
    char *dst = (char *)aligned_alloc(blk_size, chunk + blk_size);

    // mmap pass: random offset reads (use volatile sink to prevent optimization)
    {
        volatile char sink = 0;
        double t0 = now_sec();
        for (int i = 0; i < n_iter; i++) {
            size_t off = ((size_t)rand() % (fsize - chunk)) & ~(size_t)(blk_size - 1);
            memcpy(dst, (char *)mmp + off, chunk);
            sink ^= dst[0] ^ dst[chunk - 1];
        }
        double t1 = now_sec();
        double total_mb = (double)chunk * n_iter / 1024.0 / 1024.0;
        printf("[mmap]   %.1f MB total %.3f s -> %.1f MB/s (sink=%d)\n",
               total_mb, t1 - t0, total_mb / (t1 - t0), sink);
    }

    // drop cache for this file
    posix_fadvise(fd, 0, fsize, POSIX_FADV_DONTNEED);

    // mmap second pass after fadvise drop
    {
        volatile char sink = 0;
        double t0 = now_sec();
        for (int i = 0; i < n_iter; i++) {
            size_t off = ((size_t)rand() % (fsize - chunk)) & ~(size_t)(blk_size - 1);
            memcpy(dst, (char *)mmp + off, chunk);
            sink ^= dst[0] ^ dst[chunk - 1];
        }
        double t1 = now_sec();
        double total_mb = (double)chunk * n_iter / 1024.0 / 1024.0;
        printf("[mmap-after-fadvise] %.1f MB %.3f s -> %.1f MB/s (sink=%d)\n",
               total_mb, t1 - t0, total_mb / (t1 - t0), sink);
    }

    munmap(mmp, fsize);

    // === O_DIRECT mode ===
    int dfd = open(path, O_RDONLY | O_DIRECT);
    if (dfd < 0) {
        fprintf(stderr, "O_DIRECT open failed: %s\n", strerror(errno));
        return 0;  // partial result OK
    }

    {
        volatile char sink = 0;
        double t0 = now_sec();
        for (int i = 0; i < n_iter; i++) {
            size_t off = ((size_t)rand() % (fsize - chunk)) & ~(size_t)(blk_size - 1);
            ssize_t rd = pread(dfd, dst, chunk, off);
            if (rd < 0) { perror("pread direct"); break; }
            sink ^= dst[0] ^ dst[rd - 1];
        }
        double t1 = now_sec();
        double total_mb = (double)chunk * n_iter / 1024.0 / 1024.0;
        printf("[O_DIRECT] %.1f MB %.3f s -> %.1f MB/s (sink=%d)\n",
               total_mb, t1 - t0, total_mb / (t1 - t0), sink);
    }

    close(dfd);
    free(dst);
    close(fd);
    return 0;
}
