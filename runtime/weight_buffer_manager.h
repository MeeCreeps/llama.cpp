// runtime/weight_buffer_manager.h
//
// WeightBufferManager —— 按 transformer block 粒度管理权重在 GPU buffer 与
// mmap GGUF 文件之间的搬运。本文件只放**纯逻辑**：block 元数据表、LRU 选取、
// 字节核算、max_resident 计算。所有跟 OpenCL 实际打交道的部分
// （clCreateBuffer / clReleaseMemObject / cl_event 等）放在
// weight_buffer_manager_opencl.{h,cpp}（下一步加入），用 void* 不透明句柄
// 经由本头文件传递。
//
// 这样设计的好处：
//   1. 纯逻辑可以在没有 OpenCL 的 Linux 桌面上编译 + 单测
//   2. backend 切换（比如要不要 fallback 到 Vulkan）时只换包装层
//   3. WBM 算法本身（LRU、预算计算）的正确性独立验证
//
// 与 docs/elastic/baseline_external_execution.md §3 的 weight_buffer_manager
// 接口对齐；与 runtime/RUNTIME_PATCHES.md 的 H4（B1' 方案）配套使用。

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace elastic {

// 每个 transformer block 的元数据
struct block_meta {
    int      block_idx;          // 0 .. n_blocks-1
    void    *host_ptr;           // mmap_base + tensor_offset，host 端起点
    size_t   byte_size;          // 这个 block 全部权重字节数（在 GGUF 里）
    bool     resident;           // true = backend_handle 有效，权重已在 GPU 上
    uint64_t last_used_token;    // LRU 用的"最近使用 token 序号"
    void    *backend_handle;     // 不透明 cl_mem 句柄；nullptr = 未分配
    void    *last_use_event;     // 不透明 cl_event；evict 前需 wait 此 event
};

struct weight_buffer_manager {
    std::vector<block_meta> blocks;
    uint64_t current_token;
    int      n_resident;
    size_t   resident_bytes;
};

// 初始化：分配 n_blocks 个空槽位（block_idx = 0..n_blocks-1）。
// register_block 必须随后调用 n_blocks 次填充各槽。成功返回 0。
int  wbm_init(weight_buffer_manager *wbm, int n_blocks);

// 注册 block 元数据。host_ptr 与 byte_size 来自 GGUF tensor 索引；
// host_ptr = mapping->addr() + first_tensor_offset_of_block。
// register 阶段 resident=false、backend_handle=nullptr。
int  wbm_register_block(weight_buffer_manager *wbm,
                        int    idx,
                        void  *host_ptr,
                        size_t byte_size);

// 标记 block 已被搬到 GPU（OpenCL 包装层 clCreateBuffer + write 完成后调）。
// 更新 resident=true、resident_bytes += byte_size、n_resident += 1。
// 重复调用同一 idx 会断言失败。
void wbm_mark_resident(weight_buffer_manager *wbm,
                       int   idx,
                       void *backend_handle);

// 标记 block 已 evict。更新 resident=false、resident_bytes -=、清 handle。
// 对未驻留的 block 调用是 no-op。
void wbm_mark_evicted(weight_buffer_manager *wbm, int idx);

// 标记 block 在 token N 被使用，更新 LRU。token 必须单调不减。
void wbm_touch(weight_buffer_manager *wbm, int idx, uint64_t token);

// 记下某 block 上最近一次 GPU kernel 的 event。OpenCL 包装层在 evict 前会
// 拿这个 event 去 clWaitForEvents 等它结束才释放 cl_mem。
void wbm_set_last_use_event(weight_buffer_manager *wbm, int idx, void *event);

// LRU 选受害者：从已驻留 block 里挑 last_used_token 最小、且不是
// exclude_idx 的。返回 block_idx；没合适候选时返回 -1。
int  wbm_pick_lru_victim(const weight_buffer_manager *wbm, int exclude_idx);

// 按预算算出最大允许驻留 block 数：
//     floor((B - kv - misc) / max_block_bytes)
// 用最大 block 字节数做保守估算，保证选出的数字下塞进去一定不超预算。
// 返回值 ∈ [0, n_blocks]。budget_mb*MB <= kv+misc 时返回 0。
int  wbm_max_resident_blocks(const weight_buffer_manager *wbm,
                             size_t budget_mb,
                             size_t kv_bytes,
                             size_t misc_overhead);

int    wbm_resident_count(const weight_buffer_manager *wbm);
size_t wbm_resident_bytes(const weight_buffer_manager *wbm);
size_t wbm_total_bytes(const weight_buffer_manager *wbm);     // 全 resident 时的总字节
size_t wbm_max_block_bytes(const weight_buffer_manager *wbm); // 单 block 最大字节
const block_meta *wbm_get(const weight_buffer_manager *wbm, int idx);

void wbm_shutdown(weight_buffer_manager *wbm);

}  // namespace elastic
