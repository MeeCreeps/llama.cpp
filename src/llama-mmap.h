#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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

struct llama_pread_request {
    const char * filename = nullptr;
    void * dst = nullptr;
    size_t file_offset = 0;
    size_t len = 0;
};

// Read independent file offsets as one unit-level I/O submission when
// io_uring is available. Destinations may have the same page bias as their
// file offsets; aligned middles are submitted together and small edges reuse
// the direct-I/O bounce pool. Returns 0 for a real batch, 1 for the
// correctness-preserving sequential fallback, and <0 on failure.
int llama_pread_direct_batch(const llama_pread_request * requests, size_t count);

// Probe whether independent-offset direct reads can use a real kernel batch.
// Android devices commonly deny io_uring_setup to unprivileged apps. Callers
// use this to select a streaming per-tensor fallback instead of serializing a
// whole Multi unit before exposing any completed LOAD to PREPARE.
bool llama_pread_direct_batch_available();

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
typedef uint32_t (*llama_weight_state_fn_t)(const char * name, void * user_data);

enum llama_weight_state_flags_internal {
    LLAMA_WEIGHT_STATE_DISK_AVAILABLE       = 1u << 0,
    LLAMA_WEIGHT_STATE_CPU_RAW_RESIDENT     = 1u << 1,
    LLAMA_WEIGHT_STATE_CPU_COMPUTE_RESIDENT = 1u << 2,
    LLAMA_WEIGHT_STATE_GPU_RAW_RESIDENT     = 1u << 3,
    LLAMA_WEIGHT_STATE_GPU_COMPUTE_RESIDENT = 1u << 4,
};
// evict=false → prefetch, evict=true → evict. Return 0 on enqueue OK, <0 fail.
typedef int  (*llama_weight_movement_fn_t)(const char * name, bool evict, void * user_data);
// 分阶段 movement: load = disk/mmap->host, transfer = host->backend.
typedef int  (*llama_weight_stage_fn_t)(const char * name, const char * stage, void * user_data);
// Anchor-stage trigger: backend calls this when a weight/op anchor is reached
// during real graph execution. Return 0 if handled, -2 if not handled.
typedef int  (*llama_weight_anchor_fn_t)(const char * anchor_name, void * user_data);

// Explicit layout transform hook. This is separate from stage strings so runtime plans can
// request a concrete transform implementation instead of treating transform as generic movement.
enum llama_weight_transform_kind {
    LLAMA_WEIGHT_TRANSFORM_NONE        = 0,
    LLAMA_WEIGHT_TRANSFORM_CPU_REPACK  = 1,
    LLAMA_WEIGHT_TRANSFORM_GPU_CONVERT = 2,
};
typedef int  (*llama_weight_transform_fn_t)(const char * name, llama_weight_transform_kind kind, void * user_data);

struct llama_working_set_runtime_state {
    int      active_capacity = -1;
    int      pending_capacity = -1;
    int      target_capacity = -1;
    int      observed_required_capacity = 0;
    uint64_t accesses = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t capacity_changes = 0;
    size_t   resident_bytes = 0;
};
typedef bool (*llama_working_set_query_fn_t)(
        const char * kind, llama_working_set_runtime_state * state, void * user_data);
typedef int (*llama_working_set_target_fn_t)(
        const char * kind, int target_capacity, void * user_data);

void llama_weight_residency_register(llama_weight_residency_fn_t fn, void * user_data);
void llama_weight_state_register    (llama_weight_state_fn_t     fn, void * user_data);
void llama_weight_movement_register (llama_weight_movement_fn_t  fn, void * user_data);
void llama_weight_stage_register    (llama_weight_stage_fn_t     fn, void * user_data);
void llama_weight_anchor_register   (llama_weight_anchor_fn_t    fn, void * user_data);
void llama_weight_transform_register(llama_weight_transform_fn_t fn, void * user_data);
void llama_working_set_register(
        llama_working_set_query_fn_t query_fn,
        llama_working_set_target_fn_t target_fn,
        void * user_data);

// Public query/action used by llama_context.cpp.
bool llama_weight_residency_query(const char * name);
uint32_t llama_weight_state_query(const char * name);
int  llama_weight_movement_request(const char * name, bool evict);
int  llama_weight_stage_request   (const char * name, const char * stage);
int  llama_weight_anchor_request  (const char * anchor_name);
int  llama_weight_transform_request(const char * name, llama_weight_transform_kind kind);
bool llama_working_set_query(const char * kind, llama_working_set_runtime_state * state);
int  llama_working_set_set_target(const char * kind, int target_capacity);

// Mixed Super-Tensor partition published by the active ExecPlan. Backends
// query row tiles by logical weight name at graph entry. unit_id may be shared
// by tiles from different tensors (Multi), shared by all tiles of one tensor
// (Tensor), or differ per tile (Cut).
struct llama_weight_unit_slice {
    int64_t row_start = 0;
    int64_t row_count = -1;
    int     unit_id = -1;
    uint32_t flags = 0; // bit0=fuse_layout, bit1=fuse_compute
};
enum llama_weight_unit_flags : uint32_t {
    LLAMA_WEIGHT_UNIT_FUSE_LAYOUT  = 1u << 0,
    LLAMA_WEIGHT_UNIT_FUSE_COMPUTE = 1u << 1,
};
void llama_weight_unit_plan_clear();
void llama_weight_unit_plan_add(
        const char * name,
        const llama_weight_unit_slice & slice);
// Atomically replace the complete mixed frontier under one lock. Returns the
// number of logical weights whose slice list changed; identical replacement
// leaves the generation unchanged.
int llama_weight_unit_plan_replace(
        const std::vector<std::pair<
            std::string, llama_weight_unit_slice>> & entries);
int llama_weight_unit_plan_query(
        const char * name,
        llama_weight_unit_slice * out,
        int capacity);
uint64_t llama_weight_unit_plan_generation();

// Canonical elastic runtime placement map. Backends still expose their own WBM
// probes, but plan apply / staged movement update this shared map so online
// planners can dump a stable CPU/GPU/DISK state instead of reconstructing it
// only from backend-local query side effects.
enum llama_weight_runtime_location {
    LLAMA_WEIGHT_RUNTIME_UNKNOWN = 0,
    LLAMA_WEIGHT_RUNTIME_DISK    = 1,
    LLAMA_WEIGHT_RUNTIME_CPU     = 2,
    LLAMA_WEIGHT_RUNTIME_GPU     = 3,
};
void llama_weight_runtime_mark_desired (const char * name, llama_weight_runtime_location loc);
void llama_weight_runtime_mark_resident(const char * name, llama_weight_runtime_location loc);
void llama_weight_runtime_mark_evicted (const char * name, llama_weight_runtime_location loc);
llama_weight_runtime_location llama_weight_runtime_desired_query(const char * name);
uint32_t llama_weight_runtime_state_query(const char * name);

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
typedef void    (*llama_budget_reset_fn_t)(void * user_data);
void    llama_budget_register(llama_budget_fn_t fn, void * user_data);
void    llama_budget_reset_register(llama_budget_reset_fn_t fn, void * user_data);
int64_t llama_budget_query(void);   // 返回当前预算 MB; 无 provider 或无效返 -1
extern "C" void llama_budget_reset_clock(void); // reset trace replay origin when provider supports it

// === Budget target hook (统一 scheduler 的 "留多少" 维度) ===
// scheduler 决定"这一刻 GPU 上留多少字节 weight"。 给当前预算 b_t_mb + trace 最低
// m_floor_mb + kv/misc 扣除, 返回 target 字节。 elastic backend 算驱逐目标时调它:
//   static builtin  → 忽略 b_t_mb, 用 m_floor_mb (恒定保守);
//   dynamic builtin → 用 b_t_mb (随 trace 变)。
// llama_set_scheduler_v2 注册; 没注册时 backend 用自己的 static/dynamic 旧逻辑。
// 单 slot。 SIZE_MAX 哨兵 = 没注册。 runtime 库通过这个解耦, 不依赖 llama.h。
typedef size_t (*llama_budget_target_fn_t)(int64_t b_t_mb, int64_t m_floor_mb,
                                           size_t kv_bytes, size_t misc_bytes, void * user_data);
void   llama_budget_target_register(llama_budget_target_fn_t fn, void * user_data);
bool   llama_budget_target_active(void);   // 有 scheduler 注册了吗
size_t llama_budget_target_query(int64_t b_t_mb, int64_t m_floor_mb, size_t kv_bytes, size_t misc_bytes);

// === Victim selector hook (统一 scheduler 的 "踢哪个" 维度) ===
// scheduler 决定驱逐时踢哪个 block。 签名跟 weight_buffer_manager::victim_fn 一致,
// 但用 void* 避免 llama-mmap 依赖 runtime header。 backend 在 init 时把注册的指针
// 灌进 wbm.victim_fn。 单 slot。 nullptr = 没注册 (用内置 MRU/LRU)。
typedef int (*llama_victim_fn_t)(const void * wbm, int exclude_idx, void * user_data);
void              llama_victim_register(llama_victim_fn_t fn, void * user_data);
llama_victim_fn_t llama_victim_query(void ** out_user_data);  // 返回注册的 fn (没有返 nullptr)
