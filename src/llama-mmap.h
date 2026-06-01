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

// Global mmap registry — 让 backend 能反查 "这个 host_ptr 在哪个 file 的哪个 offset",
// 用于 elastic 把 reload 从 mmap memcpy 切到 O_DIRECT pread.
// llama_mmap ctor 自动注册; dtor 自动反注册.
// thread-safe (内部 mutex). lookup 返 empty filename 表示未找到.
// 注意: 存 filename 字符串而非 llama_file* — file 对象在 loader 析构时释放,
// 但 mmap 还在 model 里活着 → 不能拿 file*. backend 自己再 open 一个 fd.
#include <string>
struct llama_mmap_registry_entry {
    void *      base = nullptr;     // mmap 起始虚拟地址
    size_t      size = 0;
    std::string filename;           // GGUF 文件路径
};
llama_mmap_registry_entry llama_mmap_registry_find(const void * host_ptr);

// 独立 O_DIRECT pread helper (不依赖 llama_file 对象, 自己 open/cache fd)
// 路径为 filename 的文件: lazy 开 O_DIRECT fd, 对齐 bounce buffer, pread.
// 返 0 成功, <0 失败.
int llama_pread_direct(const char * filename, void * dst, size_t file_offset, size_t len);

// === Weight pin schedule hook (shared with llama-context API) ===
// elastic backends call llama_weight_pin_query during pin decision. Returns true
// → force pin. Default = false (let elastic budget decide). LP solver / 自定义 schedule
// 通过 llama_set_weight_pin (公开 API in llama.h) 注册 callback. 见
// llama-context.cpp::llama_set_weight_pin.
#include <cstddef>
typedef bool (*llama_weight_pin_fn_t)(const char * name, int layer, size_t byte_size, void * user_data);
void llama_weight_pin_register(llama_weight_pin_fn_t fn, void * user_data);
bool llama_weight_pin_query   (const char * name, int layer, size_t byte_size);

// === Weight residency probe + movement request (shared across elastic backends) ===
// elastic backends call _register on init, expose runtime state to user schedulers.
// Public API (llama_weight_is_resident / _request_prefetch / _request_evict) lives
// in llama.h and delegates here.
typedef bool (*llama_weight_residency_fn_t)(const char * name, void * user_data);
// evict=false → prefetch, evict=true → evict. Return 0 on enqueue OK, <0 fail.
typedef int  (*llama_weight_movement_fn_t)(const char * name, bool evict, void * user_data);

void llama_weight_residency_register(llama_weight_residency_fn_t fn, void * user_data);
void llama_weight_movement_register (llama_weight_movement_fn_t  fn, void * user_data);

// Public query/action used by llama_context.cpp.
bool llama_weight_residency_query(const char * name);
int  llama_weight_movement_request(const char * name, bool evict);

// === Weight host (mmap) pointer query ===
// elastic backends register: 给 name → host_ptr 查询 (mmap 区指针).
// 用于跨 backend dispatch: weight 被 evict 时, 不走 cl_mem 而走 mmap 直接读.
typedef void * (*llama_weight_host_ptr_fn_t)(const char * name, void * user_data);
void llama_weight_host_ptr_register(llama_weight_host_ptr_fn_t fn, void * user_data);
void * llama_weight_host_ptr_query (const char * name);

// === Budget provider (统一 scheduler 的内存信号源) ===
// elastic backend 若启用了 BudgetWatcher (GGML_ELASTIC_BUDGET_CSV), 注册一个返回
// "当前预算 MB" 的函数。 llama_context 的 runtime scheduler 优先用它 (跟 elastic
// evict target 同源, 可重现); 没注册时 fallback /proc/meminfo MemAvailable。
// 单 slot (预算是全局值, 不 chain)。 返回 < 0 表示该 provider 当前无有效预算。
typedef int64_t (*llama_budget_fn_t)(void * user_data);
void    llama_budget_register(llama_budget_fn_t fn, void * user_data);
int64_t llama_budget_query(void);   // 返回当前预算 MB; 无 provider 或无效返 -1
