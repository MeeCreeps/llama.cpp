#include "llama-mmap.h"

#include "llama-impl.h"

#include "ggml.h"

#include <cstring>
#include <climits>
#include <stdexcept>
#include <cerrno>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <utility>

#if !defined(_WIN32)
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <sys/ioctl.h>
    #include <stdlib.h>      // posix_memalign
#endif

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
            #include <fcntl.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// TODO: consider moving to llama-impl.h if needed in more places
#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

// llama_file

struct llama_file::impl {
#if defined(_WIN32)
    HANDLE fp_win32;
    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                    NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = format("Win32 error code: %lx", error_code);
        } else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }

        return ret;
    }

    impl(const char * fname, const char * mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    void read_raw(void * ptr, size_t len) const {
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64*1024*1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        }
    }

    uint32_t read_u32() const {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void * ptr, size_t len) const {
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64*1024*1024);
            DWORD chunk_written = 0;
            BOOL result = WriteFile(fp_win32, reinterpret_cast<char const*>(ptr) + bytes_written, chunk_size, &chunk_written, NULL);
            if (!result) {
                throw std::runtime_error(format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~impl() {
        if (fp) {
            std::fclose(fp);
        }
    }
#else
    impl(const char * fname, const char * mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
// TODO: this ifdef is never true?
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        if (ret == -1) {
            throw std::runtime_error(format("ftell error: %s", strerror(errno)));
        }

        return (size_t) ret;
    }

    void seek(size_t offset, int whence) const {
// TODO: this ifdef is never true?
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        if (ret != 0) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    // O_DIRECT bypass page cache.
    // Linux/Android only. 处理:
    //   - lazy 开 fd (一次)
    //   - 探测 device block size 作 alignment
    //   - 用对齐 bounce buffer 处理 unaligned offset/size/dst
    int pread_direct_impl(void * dst, size_t file_offset, size_t len) const {
#if defined(__linux__) || defined(__ANDROID__)
        if (direct_fd < 0) {
            int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
            if (fd < 0) {
                // 某些 fs (e.g. tmpfs, f2fs 配置不允许) 不支持 O_DIRECT
                fprintf(stderr, "[pread_direct] open O_DIRECT failed for %s: %s, fallback no-direct\n",
                        filename.c_str(), strerror(errno));
                fd = open(filename.c_str(), O_RDONLY);
                if (fd < 0) return -1;
            }
            direct_fd = fd;
            // block size detect: 探测 fs block size, fallback 4096
            struct stat st;
            if (fstat(fd, &st) == 0) {
                align = (size_t) st.st_blksize;
                if (align == 0 || (align & (align - 1)) != 0) align = 4096;
            } else {
                align = 4096;
            }
        }
        size_t a = align;
        size_t off_aligned   = file_offset & ~(a - 1);
        size_t head_skip     = file_offset - off_aligned;
        size_t total_aligned = ((head_skip + len + a - 1) / a) * a;
        // bounce buffer (aligned) 容量
        if (bounce_cap < total_aligned) {
            if (bounce_buf) free(bounce_buf);
            bounce_buf = nullptr;
            if (posix_memalign(&bounce_buf, a, total_aligned) != 0) {
                bounce_cap = 0;
                return -2;
            }
            bounce_cap = total_aligned;
        }
        ssize_t rd = pread(direct_fd, bounce_buf, total_aligned, off_aligned);
        if (rd < 0) {
            fprintf(stderr, "[pread_direct] pread fail off=%zu len=%zu: %s\n",
                    file_offset, len, strerror(errno));
            return -3;
        }
        // 拷贝有效区到 dst
        memcpy(dst, (char *)bounce_buf + head_skip, len);
        return 0;
#else
        (void)dst; (void)file_offset; (void)len;
        return -1;
#endif
    }

    ~impl() {
        if (fp) {
            std::fclose(fp);
        }
        if (direct_fd >= 0) {
            close(direct_fd);
            direct_fd = -1;
        }
        if (bounce_buf) {
            free(bounce_buf);
            bounce_buf = nullptr;
            bounce_cap = 0;
        }
    }
#endif

    FILE * fp;
    size_t size;

#if !defined(_WIN32)
    // O_DIRECT fd (lazy-opened), 跟 fp 同文件但独立, 用于绕 page cache 的 pread.
    mutable int direct_fd       = -1;
    mutable std::string         filename;   // 记下来方便 lazy open
    mutable size_t              align       = 0;   // pread alignment requirement (block size)
    mutable void *              bounce_buf  = nullptr;
    mutable size_t              bounce_cap  = 0;
#endif
};

llama_file::llama_file(const char * fname, const char * mode) : pimpl(std::make_unique<impl>(fname, mode)) {
#if !defined(_WIN32)
    pimpl->filename = fname;
#endif
}
llama_file::~llama_file() = default;

size_t llama_file::tell() const { return pimpl->tell(); }
size_t llama_file::size() const { return pimpl->size; }

int llama_file::file_id() const {
#ifdef _WIN32
    return _fileno(pimpl->fp);
#else
#if defined(fileno)
    return fileno(pimpl->fp);
#else
    return ::fileno(pimpl->fp);
#endif
#endif
}

void llama_file::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void llama_file::read_raw(void * ptr, size_t len) const { pimpl->read_raw(ptr, len); }

int llama_file::pread_direct(void * dst, size_t file_offset, size_t len) const {
#if defined(_WIN32)
    (void)dst; (void)file_offset; (void)len;
    return -1;
#else
    return pimpl->pread_direct_impl(dst, file_offset, len);
#endif
}

uint32_t llama_file::read_u32() const { return pimpl->read_u32(); }

void llama_file::write_raw(const void * ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void llama_file::write_u32(uint32_t val) const { pimpl->write_u32(val); }

// llama_mmap

struct llama_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
    std::vector<std::pair<size_t, size_t>> mapped_fragments;

    impl(struct llama_file * file, size_t prefetch, bool numa) {
        size = file->size();
        int fd = file->file_id();
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            if (posix_madvise(addr, std::min(file->size(), prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }

        mapped_fragments.emplace_back(0, file->size());
    }

    static void align_range(size_t * first, size_t * last, size_t page_size) {
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void * next_page_start = (uint8_t *) addr + first;

        if (munmap(next_page_start, len)) {
            LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto & frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
            } else {
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~impl() {
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }
#elif defined(_WIN32)
    impl(struct llama_file * file, size_t prefetch, bool numa) {
        GGML_UNUSED(numa);

        size = file->size();

        HANDLE hFile = (HANDLE) _get_osfhandle(file->file_id());

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(error).c_str()));
        }

        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            BOOL (WINAPI *pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            pPrefetchVirtualMemory = (decltype(pPrefetchVirtualMemory))(void *) GetProcAddress(hKernel32, "PrefetchVirtualMemory");

            if (pPrefetchVirtualMemory) {
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T) std::min(size, prefetch);
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    LLAMA_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
#else
            LLAMA_LOG_DEBUG("skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#endif
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    ~impl() {
        if (!UnmapViewOfFile(addr)) {
            LLAMA_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    impl(struct llama_file * file, size_t prefetch, bool numa) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);
        GGML_UNUSED(numa);

        throw std::runtime_error("mmap not supported");
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }
#endif

    void * addr;
    size_t size;
};

// === Global mmap registry ===
#include <mutex>
namespace {
std::mutex g_mmap_reg_mtx;
std::vector<llama_mmap_registry_entry> g_mmap_registry;
}

static void mmap_registry_add(void *base, size_t size, const std::string &filename) {
    std::lock_guard<std::mutex> lk(g_mmap_reg_mtx);
    g_mmap_registry.push_back({base, size, filename});
}
static void mmap_registry_remove(void *base) {
    std::lock_guard<std::mutex> lk(g_mmap_reg_mtx);
    g_mmap_registry.erase(
        std::remove_if(g_mmap_registry.begin(), g_mmap_registry.end(),
                       [base](const auto &e) { return e.base == base; }),
        g_mmap_registry.end());
}
llama_mmap_registry_entry llama_mmap_registry_find(const void *host_ptr) {
    std::lock_guard<std::mutex> lk(g_mmap_reg_mtx);
    for (const auto &e : g_mmap_registry) {
        if (host_ptr >= e.base && host_ptr < (char*)e.base + e.size) return e;
    }
    return {};
}

// 独立 O_DIRECT pread: backend 不依赖 llama_file (它在 loader destruct 时释放).
// 内部按 filename 缓存 fd + bounce buffer.
struct direct_io_handle {
    int    fd      = -1;
    size_t blk     = 4096;
    void * bounce  = nullptr;
    size_t bcap    = 0;
};
static std::mutex g_direct_mtx;
static std::unordered_map<std::string, direct_io_handle> g_direct_handles;

static int llama_pread_full_direct(int fd, void * dst, size_t len, size_t file_offset, const char * tag) {
    char * out = static_cast<char *>(dst);
    size_t done = 0;
    while (done < len) {
        ssize_t rd = pread(fd, out + done, len - done, file_offset + done);
        if (rd < 0) {
            fprintf(stderr, "[llama_pread_direct %s] pread fail: %s\n", tag, strerror(errno));
            return -3;
        }
        if (rd == 0) {
            fprintf(stderr, "[llama_pread_direct %s] short read: got %zu / %zu at offset=%zu\n",
                    tag, done, len, file_offset);
            return -4;
        }
        done += (size_t) rd;
    }
    return 0;
}

int llama_pread_direct(const char *filename, void *dst, size_t file_offset, size_t len) {
#if defined(__linux__) || defined(__ANDROID__)
    std::lock_guard<std::mutex> lk(g_direct_mtx);
    auto &h = g_direct_handles[filename];
    if (h.fd < 0) {
        int fd = open(filename, O_RDONLY | O_DIRECT);
        if (fd < 0) {
            fprintf(stderr, "[llama_pread_direct] open O_DIRECT failed %s: %s\n", filename, strerror(errno));
            return -1;
        }
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_blksize >= 4096 && (st.st_blksize & (st.st_blksize-1)) == 0) {
            h.blk = (size_t)st.st_blksize;
        }
        h.fd = fd;
    }
    const size_t blk = h.blk;

    // Fast path: dst, file_offset, len 都 blk 对齐 → pread 直接到 dst, 零 memcpy.
    const bool dst_aligned    = (reinterpret_cast<uintptr_t>(dst) % blk) == 0;
    const bool offset_aligned = (file_offset % blk) == 0;
    const bool len_aligned    = (len % blk) == 0;
    if (dst_aligned && offset_aligned && len_aligned && len > 0) {
        return llama_pread_full_direct(h.fd, dst, len, file_offset, "fastpath");
    }

    // Split path: 头尾用 bounce 处理对齐, 中间 (如果对齐) 直接 pread 到 dst.
    // 省掉 99% 的 bounce→dst memcpy (对于 50+ MB tensor 而言, 头尾 < 8 KB).
    const size_t head_off_aligned = file_offset & ~(blk - 1);
    const size_t head_skip        = file_offset - head_off_aligned;
    const size_t head_block_size  = blk - head_skip;  // 第一个 block 内有效字节
    const size_t tail_end         = file_offset + len;
    const size_t tail_end_aligned = (tail_end + blk - 1) & ~(blk - 1);

    // 如果 len 很小整个落在 head block 内, 直接 bounce
    if (len <= head_block_size) {
        size_t need = ((head_skip + len + blk - 1) / blk) * blk;
        if (h.bcap < need) {
            if (h.bounce) free(h.bounce);
            h.bounce = nullptr;
            if (posix_memalign(&h.bounce, blk, need) != 0) { h.bcap = 0; return -2; }
            h.bcap = need;
        }
        int rc = llama_pread_full_direct(h.fd, h.bounce, need, head_off_aligned, "head-only");
        if (rc != 0) return rc;
        memcpy(dst, (char*)h.bounce + head_skip, len);
        return 0;
    }

    char *dst_c = (char *)dst;

    // 1) Head bounce: 读对齐起始 1 block, 拷贝有效部分到 dst[0..head_block_size)
    {
        if (h.bcap < blk) {
            if (h.bounce) free(h.bounce);
            h.bounce = nullptr;
            if (posix_memalign(&h.bounce, blk, blk) != 0) { h.bcap = 0; return -2; }
            h.bcap = blk;
        }
        int rc = llama_pread_full_direct(h.fd, h.bounce, blk, head_off_aligned, "head");
        if (rc != 0) return rc;
        memcpy(dst_c, (char*)h.bounce + head_skip, head_block_size);
    }

    // 2) Middle direct: 从 file_offset+head_block_size 开始, 必然 blk-aligned.
    //    长度尽量取 blk 倍数, 写到 dst+head_block_size (检查是否 blk-aligned).
    const size_t mid_file_off = file_offset + head_block_size;       // blk-aligned
    char *mid_dst             = dst_c + head_block_size;
    const size_t remaining    = len - head_block_size;
    const size_t mid_blocks   = remaining / blk;                     // 完整 block 数
    const size_t mid_size     = mid_blocks * blk;
    const size_t tail_size    = remaining - mid_size;

    if (mid_size > 0) {
        if ((reinterpret_cast<uintptr_t>(mid_dst) % blk) == 0) {
            // mid_dst blk-aligned → direct pread, 零 memcpy
            int rc = llama_pread_full_direct(h.fd, mid_dst, mid_size, mid_file_off, "mid");
            if (rc != 0) return rc;
        } else {
            // 罕见: dst 不 blk-aligned, 中段也 bounce. 跟旧路径等效.
            if (h.bcap < mid_size) {
                if (h.bounce) free(h.bounce);
                h.bounce = nullptr;
                if (posix_memalign(&h.bounce, blk, mid_size) != 0) { h.bcap = 0; return -2; }
                h.bcap = mid_size;
            }
            int rc = llama_pread_full_direct(h.fd, h.bounce, mid_size, mid_file_off, "mid-bounce");
            if (rc != 0) return rc;
            memcpy(mid_dst, h.bounce, mid_size);
        }
    }

    // 3) Tail bounce: 读最后 1 block, 拷贝前 tail_size 字节
    if (tail_size > 0) {
        const size_t tail_file_off = mid_file_off + mid_size;        // blk-aligned
        if (h.bcap < blk) {
            if (h.bounce) free(h.bounce);
            h.bounce = nullptr;
            if (posix_memalign(&h.bounce, blk, blk) != 0) { h.bcap = 0; return -2; }
            h.bcap = blk;
        }
        int rc = llama_pread_full_direct(h.fd, h.bounce, blk, tail_file_off, "tail");
        if (rc != 0) return rc;
        memcpy(mid_dst + mid_size, h.bounce, tail_size);
    }

    (void)tail_end_aligned;
    return 0;
#else
    (void)filename; (void)dst; (void)file_offset; (void)len;
    return -1;
#endif
}

static std::string mmap_file_path(struct llama_file *f) {
    // 用 file_id() 反查 /proc/self/fd/<id> 的 symlink 拿绝对路径
    if (!f) return "";
    int fd = f->file_id();
    char link[64];
    snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    char buf[4096];
    ssize_t n = readlink(link, buf, sizeof(buf)-1);
    if (n <= 0) return "";
    buf[n] = 0;
    return std::string(buf);
}

// === Weight pin schedule registry (cross-translation-unit hook) ===
// elastic backends 在 pin 决策时调 llama_weight_pin_query. 默认空 = 让 elastic
// 按 env 的 PIN policy 决定. 设置 callback → callback 优先.
namespace {
std::mutex             g_weight_pin_mtx;
llama_weight_pin_fn_t  g_weight_pin_fn = nullptr;
void *                 g_weight_pin_ud = nullptr;
}
void llama_weight_pin_register(llama_weight_pin_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_weight_pin_mtx);
    g_weight_pin_fn = fn;
    g_weight_pin_ud = user_data;
}
bool llama_weight_pin_query(const char * name, int layer, size_t byte_size) {
    llama_weight_pin_fn_t fn;
    void * ud;
    {
        std::lock_guard<std::mutex> lk(g_weight_pin_mtx);
        fn = g_weight_pin_fn;
        ud = g_weight_pin_ud;
    }
    if (!fn) return false;
    return fn(name, layer, byte_size, ud);
}

// === Weight residency probe + movement request registry ===
// 支持多 provider (cpu-elastic + opencl-elastic 共存) — 任一返回 true / 接到请求即停.
namespace {
std::mutex                                                       g_weight_res_mtx;
std::vector<std::pair<llama_weight_residency_fn_t, void *>>      g_weight_res_providers;
std::vector<std::pair<llama_weight_state_fn_t,     void *>>      g_weight_state_providers;
std::vector<std::pair<llama_weight_movement_fn_t,  void *>>      g_weight_mov_providers;
std::vector<std::pair<llama_weight_stage_fn_t,     void *>>      g_weight_stage_providers;
std::vector<std::pair<llama_weight_anchor_fn_t,    void *>>      g_weight_anchor_providers;
std::vector<std::pair<llama_weight_transform_fn_t, void *>>      g_weight_transform_providers;
struct llama_weight_runtime_record {
    llama_weight_runtime_location desired = LLAMA_WEIGHT_RUNTIME_UNKNOWN;
    bool cpu_compute_resident = false;
    bool gpu_compute_resident = false;
};
std::unordered_map<std::string, llama_weight_runtime_record>      g_weight_runtime_state;
// Budget provider: 单 slot (预算是全局值)。
std::mutex                                                       g_budget_mtx;
llama_budget_fn_t                                                g_budget_fn = nullptr;
void *                                                           g_budget_ud = nullptr;
llama_budget_reset_fn_t                                          g_budget_reset_fn = nullptr;
void *                                                           g_budget_reset_ud = nullptr;
}
void llama_budget_register(llama_budget_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_budget_mtx);
    g_budget_fn = fn;
    g_budget_ud = user_data;
}
void llama_budget_reset_register(llama_budget_reset_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_budget_mtx);
    g_budget_reset_fn = fn;
    g_budget_reset_ud = user_data;
}
int64_t llama_budget_query(void) {
    llama_budget_fn_t fn;
    void * ud;
    {
        std::lock_guard<std::mutex> lk(g_budget_mtx);
        fn = g_budget_fn;
        ud = g_budget_ud;
    }
    return fn ? fn(ud) : -1;
}
void llama_budget_reset_clock(void) {
    llama_budget_reset_fn_t fn;
    void * ud;
    {
        std::lock_guard<std::mutex> lk(g_budget_mtx);
        fn = g_budget_reset_fn;
        ud = g_budget_reset_ud;
    }
    if (fn) fn(ud);
}

// === Budget target hook (scheduler 的 "留多少" 维度) ===
namespace {
std::mutex                g_budget_target_mtx;
llama_budget_target_fn_t  g_budget_target_fn = nullptr;
void *                    g_budget_target_ud = nullptr;
}
void llama_budget_target_register(llama_budget_target_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_budget_target_mtx);
    g_budget_target_fn = fn;
    g_budget_target_ud = user_data;
}
bool llama_budget_target_active(void) {
    std::lock_guard<std::mutex> lk(g_budget_target_mtx);
    return g_budget_target_fn != nullptr;
}
size_t llama_budget_target_query(int64_t b_t_mb, int64_t m_floor_mb, size_t kv_bytes, size_t misc_bytes) {
    llama_budget_target_fn_t fn;
    void * ud;
    {
        std::lock_guard<std::mutex> lk(g_budget_target_mtx);
        fn = g_budget_target_fn;
        ud = g_budget_target_ud;
    }
    return fn ? fn(b_t_mb, m_floor_mb, kv_bytes, misc_bytes, ud) : SIZE_MAX;
}

// === Victim selector hook (scheduler 的 "踢哪个" 维度) ===
namespace {
std::mutex         g_victim_mtx;
llama_victim_fn_t  g_victim_fn = nullptr;
void *             g_victim_ud = nullptr;
}
void llama_victim_register(llama_victim_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_victim_mtx);
    g_victim_fn = fn;
    g_victim_ud = user_data;
}
llama_victim_fn_t llama_victim_query(void ** out_user_data) {
    std::lock_guard<std::mutex> lk(g_victim_mtx);
    if (out_user_data) *out_user_data = g_victim_ud;
    return g_victim_fn;
}
void llama_weight_residency_register(llama_weight_residency_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    g_weight_res_providers.emplace_back(fn, user_data);
}
void llama_weight_state_register(llama_weight_state_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    g_weight_state_providers.emplace_back(fn, user_data);
}
void llama_weight_movement_register(llama_weight_movement_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    g_weight_mov_providers.emplace_back(fn, user_data);
}
void llama_weight_stage_register(llama_weight_stage_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    g_weight_stage_providers.emplace_back(fn, user_data);
}
void llama_weight_anchor_register(llama_weight_anchor_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    g_weight_anchor_providers.emplace_back(fn, user_data);
}
void llama_weight_transform_register(llama_weight_transform_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    g_weight_transform_providers.emplace_back(fn, user_data);
}
bool llama_weight_residency_query(const char * name) {
    std::vector<std::pair<llama_weight_residency_fn_t, void *>> snap;
    {
        std::lock_guard<std::mutex> lk(g_weight_res_mtx);
        snap = g_weight_res_providers;
    }
    for (auto & p : snap) {
        if (p.first && p.first(name, p.second)) return true;
    }
    return false;
}
uint32_t llama_weight_state_query(const char * name) {
    std::vector<std::pair<llama_weight_state_fn_t, void *>> snap;
    {
        std::lock_guard<std::mutex> lk(g_weight_res_mtx);
        snap = g_weight_state_providers;
    }
    uint32_t flags = 0;
    for (auto & p : snap) {
        if (p.first) flags |= p.first(name, p.second);
    }
    flags |= llama_weight_runtime_state_query(name);
    return flags;
}

static uint32_t llama_weight_runtime_flags_locked(const llama_weight_runtime_record & r) {
    uint32_t flags = 0;
    if (r.cpu_compute_resident) flags |= LLAMA_WEIGHT_STATE_CPU_COMPUTE_RESIDENT;
    if (r.gpu_compute_resident) flags |= LLAMA_WEIGHT_STATE_GPU_COMPUTE_RESIDENT;
    return flags;
}

void llama_weight_runtime_mark_desired(const char * name, llama_weight_runtime_location loc) {
    if (!name || !*name) return;
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    g_weight_runtime_state[name].desired = loc;
}

llama_weight_runtime_location llama_weight_runtime_desired_query(const char * name) {
    if (!name || !*name) return LLAMA_WEIGHT_RUNTIME_UNKNOWN;
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    auto it = g_weight_runtime_state.find(name);
    if (it == g_weight_runtime_state.end()) return LLAMA_WEIGHT_RUNTIME_UNKNOWN;
    return it->second.desired;
}

void llama_weight_runtime_mark_resident(const char * name, llama_weight_runtime_location loc) {
    if (!name || !*name) return;
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    auto & r = g_weight_runtime_state[name];
    if (loc == LLAMA_WEIGHT_RUNTIME_CPU) {
        r.cpu_compute_resident = true;
        r.gpu_compute_resident = false;
    } else if (loc == LLAMA_WEIGHT_RUNTIME_GPU) {
        r.gpu_compute_resident = true;
        r.cpu_compute_resident = false;
    } else if (loc == LLAMA_WEIGHT_RUNTIME_DISK) {
        r.cpu_compute_resident = false;
        r.gpu_compute_resident = false;
    }
}

void llama_weight_runtime_mark_evicted(const char * name, llama_weight_runtime_location loc) {
    if (!name || !*name) return;
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    auto & r = g_weight_runtime_state[name];
    if (loc == LLAMA_WEIGHT_RUNTIME_CPU) {
        r.cpu_compute_resident = false;
    } else if (loc == LLAMA_WEIGHT_RUNTIME_GPU) {
        r.gpu_compute_resident = false;
    } else if (loc == LLAMA_WEIGHT_RUNTIME_DISK) {
        r.cpu_compute_resident = false;
        r.gpu_compute_resident = false;
    }
}

uint32_t llama_weight_runtime_state_query(const char * name) {
    if (!name || !*name) return 0;
    std::lock_guard<std::mutex> lk(g_weight_res_mtx);
    auto it = g_weight_runtime_state.find(name);
    if (it == g_weight_runtime_state.end()) return 0;
    return llama_weight_runtime_flags_locked(it->second);
}
int llama_weight_movement_request(const char * name, bool evict) {
    std::vector<std::pair<llama_weight_movement_fn_t, void *>> snap;
    {
        std::lock_guard<std::mutex> lk(g_weight_res_mtx);
        snap = g_weight_mov_providers;
    }
    if (evict) {
        bool handled = false;
        bool any_success = false;
        int first_err = -2;
        for (auto & p : snap) {
            if (!p.first) continue;
            int rc = p.first(name, true, p.second);
            if (rc == -2) continue;
            handled = true;
            if (rc == 0) any_success = true;
            if (rc != 0 && first_err == -2) first_err = rc;
        }
        if (!handled) return -2;
        return any_success ? 0 : first_err;
    }
    for (auto & p : snap) {
        if (!p.first) continue;
        int rc = p.first(name, evict, p.second);
        if (rc != -2) return rc;
    }
    return -2;
}
int llama_weight_stage_request(const char * name, const char * stage) {
    std::vector<std::pair<llama_weight_stage_fn_t, void *>> snap;
    {
        std::lock_guard<std::mutex> lk(g_weight_res_mtx);
        snap = g_weight_stage_providers;
    }
    for (auto & p : snap) {
        if (!p.first) continue;
        int rc = p.first(name, stage, p.second);
        if (rc != -2) return rc;
    }
    return -2;
}
int llama_weight_anchor_request(const char * anchor_name) {
    std::vector<std::pair<llama_weight_anchor_fn_t, void *>> snap;
    {
        std::lock_guard<std::mutex> lk(g_weight_res_mtx);
        snap = g_weight_anchor_providers;
    }
    for (auto & p : snap) {
        if (!p.first) continue;
        int rc = p.first(anchor_name, p.second);
        if (rc != -2) return rc;
    }
    return -2;
}
int llama_weight_transform_request(const char * name, llama_weight_transform_kind kind) {
    if (kind == LLAMA_WEIGHT_TRANSFORM_NONE) return 0;
    std::vector<std::pair<llama_weight_transform_fn_t, void *>> snap;
    {
        std::lock_guard<std::mutex> lk(g_weight_res_mtx);
        snap = g_weight_transform_providers;
    }
    for (auto & p : snap) {
        if (!p.first) continue;
        int rc = p.first(name, kind, p.second);
        if (rc != -2) return rc;
    }
    if (kind == LLAMA_WEIGHT_TRANSFORM_CPU_REPACK) return 0;
    return -2;
}

// === Weight host (mmap) pointer registry ===
namespace {
std::mutex                 g_weight_hostptr_mtx;
llama_weight_host_ptr_fn_t g_weight_hostptr_fn = nullptr;
void *                     g_weight_hostptr_ud = nullptr;
}
void llama_weight_host_ptr_register(llama_weight_host_ptr_fn_t fn, void * user_data) {
    std::lock_guard<std::mutex> lk(g_weight_hostptr_mtx);
    g_weight_hostptr_fn = fn;
    g_weight_hostptr_ud = user_data;
}
void * llama_weight_host_ptr_query(const char * name) {
    llama_weight_host_ptr_fn_t fn;
    void * ud;
    {
        std::lock_guard<std::mutex> lk(g_weight_hostptr_mtx);
        fn = g_weight_hostptr_fn;
        ud = g_weight_hostptr_ud;
    }
    if (!fn) return nullptr;
    return fn(name, ud);
}

llama_mmap::llama_mmap(struct llama_file * file, size_t prefetch, bool numa) : pimpl(std::make_unique<impl>(file, prefetch, numa)) {
    mmap_registry_add(pimpl->addr, pimpl->size, mmap_file_path(file));
}
llama_mmap::~llama_mmap() {
    if (pimpl) mmap_registry_remove(pimpl->addr);
}

size_t llama_mmap::size() const { return pimpl->size; }
void * llama_mmap::addr() const { return pimpl->addr; }

void llama_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mmap::SUPPORTED  = true;
#else
const bool llama_mmap::SUPPORTED  = false;
#endif

// llama_mlock

struct llama_mlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

#ifdef __APPLE__
#define MLOCK_SUGGESTION \
        "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
        "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
        "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX)
        // visionOS/tvOS dont't support RLIMIT_MEMLOCK
        // Skip resource limit checks on visionOS/tvOS
        suggest = false;
#else
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
            suggest = false;
        }
#endif

        LLAMA_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            LLAMA_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                LLAMA_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                LLAMA_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            size_t increment = len + 1048576;
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                LLAMA_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            LLAMA_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) const {
        LLAMA_LOG_WARN("warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void * addr, size_t len) {}
#endif

    impl() : addr(NULL), size(0), failed_already(false) {}

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0);
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

    void * addr;
    size_t size;

    bool failed_already;
};

llama_mlock::llama_mlock() : pimpl(std::make_unique<impl>()) {}
llama_mlock::~llama_mlock() = default;

void llama_mlock::init(void * ptr) { pimpl->init(ptr); }
void llama_mlock::grow_to(size_t target_size) { pimpl->grow_to(target_size); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mlock::SUPPORTED = true;
#else
const bool llama_mlock::SUPPORTED = false;
#endif

size_t llama_path_max() {
    return PATH_MAX;
}
