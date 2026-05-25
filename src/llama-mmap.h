#pragma once

#include <cstdint>
#include <memory>
#include <vector>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len) const;
    uint32_t read_u32() const;

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    // O_DIRECT read：绕过 page cache，每次真打 disk (UFS controller). 用于 elastic
    // benchmark "model > RAM" 场景模拟. 内部用 pread + O_DIRECT fd, 处理对齐
    // (buffer / offset / size 不必 4KB 对齐, 函数内部用对齐 bounce buffer 补齐).
    // 仅 Linux/Android, 其它平台 fallback 走普通 pread.
    // 注意: 跟 read_raw/seek 用的是独立 fd, 不影响 mmap path.
    // 返回 0 = 成功, <0 = 错误.
    int pread_direct(void * dst, size_t file_offset, size_t len) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();
