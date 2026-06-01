// runtime/weight_buffer_manager.cpp —— 见 weight_buffer_manager.h

#include "weight_buffer_manager.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>

namespace elastic {

int wbm_init(weight_buffer_manager *wbm, int n_blocks) {
    if (!wbm || n_blocks < 0) return -1;
    wbm->blocks.clear();
    wbm->blocks.resize(static_cast<size_t>(n_blocks));
    for (int i = 0; i < n_blocks; ++i) {
        wbm->blocks[i].block_idx       = i;
        wbm->blocks[i].host_ptr        = nullptr;
        wbm->blocks[i].byte_size       = 0;
        wbm->blocks[i].resident        = false;
        wbm->blocks[i].is_pinned       = false;
        wbm->blocks[i].last_used_token = 0;
        wbm->blocks[i].backend_handle  = nullptr;
        wbm->blocks[i].last_use_event  = nullptr;
        wbm->blocks[i].prefetch_event  = nullptr;
    }
    wbm->current_token  = 0;
    wbm->n_resident     = 0;
    wbm->resident_bytes = 0;
    return 0;
}

int wbm_add_block(weight_buffer_manager *wbm, void *host_ptr, size_t byte_size) {
    if (!wbm || !host_ptr || byte_size == 0) return -1;
    int idx = static_cast<int>(wbm->blocks.size());
    wbm->blocks.emplace_back();
    block_meta &b = wbm->blocks.back();
    b.block_idx       = idx;
    b.host_ptr        = host_ptr;
    b.byte_size       = byte_size;
    b.resident        = false;
    b.is_pinned       = false;
    b.last_used_token = 0;
    b.backend_handle  = nullptr;
    b.last_use_event  = nullptr;
    b.prefetch_event  = nullptr;
    return idx;
}

int wbm_register_block(weight_buffer_manager *wbm,
                       int idx, void *host_ptr, size_t byte_size) {
    if (!wbm) return -1;
    if (idx < 0 || static_cast<size_t>(idx) >= wbm->blocks.size()) return -2;
    if (!host_ptr || byte_size == 0) return -3;
    block_meta &b = wbm->blocks[idx];
    b.host_ptr  = host_ptr;
    b.byte_size = byte_size;
    b.resident  = false;
    b.backend_handle = nullptr;
    b.last_use_event = nullptr;
    b.prefetch_event = nullptr;
    b.last_used_token = 0;
    return 0;
}

void wbm_mark_resident(weight_buffer_manager *wbm, int idx, void *backend_handle) {
    assert(wbm);
    assert(idx >= 0 && static_cast<size_t>(idx) < wbm->blocks.size());
    block_meta &b = wbm->blocks[idx];
    assert(!b.resident && "重复 mark_resident 同一 block 是 bug");
    b.resident        = true;
    b.backend_handle  = backend_handle;
    wbm->n_resident  += 1;
    wbm->resident_bytes += b.byte_size;
}

void wbm_mark_evicted(weight_buffer_manager *wbm, int idx) {
    if (!wbm) return;
    if (idx < 0 || static_cast<size_t>(idx) >= wbm->blocks.size()) return;
    block_meta &b = wbm->blocks[idx];
    if (!b.resident) return;
    b.resident       = false;
    b.backend_handle = nullptr;
    b.last_use_event = nullptr;
    b.prefetch_event = nullptr;
    wbm->n_resident -= 1;
    assert(wbm->resident_bytes >= b.byte_size);
    wbm->resident_bytes -= b.byte_size;
}

void wbm_touch(weight_buffer_manager *wbm, int idx, uint64_t /*token_unused*/) {
    if (!wbm) return;
    if (idx < 0 || static_cast<size_t>(idx) >= wbm->blocks.size()) return;
    // 用一个全局递增计数器记 access 顺序，比 per-graph_compute 的 token 计数器
    // 精细：能区分同一 forward pass 内不同 weight 的访问先后，让 MRU 真的能
    // 选到"刚用过"的 victim（而不是被同 token 的 ties 卡到 lowest block_idx）。
    wbm->current_token += 1;
    wbm->blocks[idx].last_used_token = wbm->current_token;
}

void wbm_set_last_use_event(weight_buffer_manager *wbm, int idx, void *event) {
    if (!wbm) return;
    if (idx < 0 || static_cast<size_t>(idx) >= wbm->blocks.size()) return;
    wbm->blocks[idx].last_use_event = event;
}

void wbm_set_prefetch_event(weight_buffer_manager *wbm, int idx, void *event) {
    if (!wbm) return;
    if (idx < 0 || static_cast<size_t>(idx) >= wbm->blocks.size()) return;
    wbm->blocks[idx].prefetch_event = event;
}

void wbm_set_pinned(weight_buffer_manager *wbm, int idx, bool pinned) {
    if (!wbm) return;
    if (idx < 0 || static_cast<size_t>(idx) >= wbm->blocks.size()) return;
    wbm->blocks[idx].is_pinned = pinned;
}

int wbm_pick_lru_victim(const weight_buffer_manager *wbm, int exclude_idx) {
    if (!wbm) return -1;
    int victim = -1;
    if (wbm->victim_fn) {
        victim = wbm->victim_fn(wbm, exclude_idx, wbm->victim_ud);
        if (victim >= 0) {
            const block_meta &bv = wbm->blocks[victim];
            if (!bv.resident || bv.is_pinned || victim == exclude_idx) victim = -1;
        }
        return victim;
    }
    if (wbm->evict_mru) {
        // MRU：选 last_used_token 最大（刚被访问过的）
        uint64_t newest = 0;
        bool found = false;
        for (const auto &b : wbm->blocks) {
            if (!b.resident)            continue;
            if (b.is_pinned)            continue;
            if (b.block_idx == exclude_idx) continue;
            if (!found || b.last_used_token > newest) {
                newest = b.last_used_token;
                victim = b.block_idx;
                found = true;
            }
        }
    } else {
        // LRU：选 last_used_token 最小
        uint64_t oldest = static_cast<uint64_t>(-1);  // UINT64_MAX
        for (const auto &b : wbm->blocks) {
            if (!b.resident)            continue;
            if (b.is_pinned)            continue;
            if (b.block_idx == exclude_idx) continue;
            if (b.last_used_token < oldest) {
                oldest = b.last_used_token;
                victim = b.block_idx;
            }
        }
    }
    return victim;
}

int wbm_evict_to_byte_budget(weight_buffer_manager *wbm,
                             size_t target_bytes,
                             int exclude_idx,
                             std::vector<int> *out_victims) {
    if (!wbm || !out_victims) return 0;
    // 纯挑选：不修改 resident 状态、不释放 backend handle，
    // 只把 victim 序号追加到 out_victims。这样 OpenCL 包装层能拿到完整
    // (cl_mem, cl_event) 表做批量 clWaitForEvents + clReleaseMemObject，
    // 然后才回头逐个 mark_evicted。
    //
    // 模拟"如果驱逐已选 victim 后字节数会是多少"：用本地 simulated_bytes
    // 跟踪，pick 算法继续基于真实 last_used_token / pinned / exclude。
    size_t simulated_bytes = wbm->resident_bytes;
    std::vector<bool> picked(wbm->blocks.size(), false);
    int n_picked = 0;
    // 本轮选中的 victim 临时置 resident=false, 防 victim_fn / 内置策略重复选中
    // (它们都以 !resident 为过滤条件); 函数末尾恢复, 真正的 mark_evicted 由调用方做。
    std::vector<int> hidden;
    while (simulated_bytes > target_bytes) {
        int victim = -1;
        if (wbm->victim_fn) {
            // 注入的 pick_victim hook (桥接到用户 scheduler)。 看 wbm 当前 resident
            // 候选 (已 picked 的本轮被置 resident=false, 不会被选)。
            victim = wbm->victim_fn(wbm, exclude_idx, wbm->victim_ud);
            // 防御: hook 返回非候选 (pinned/exclude/非 resident) 时丢弃, 退出
            if (victim >= 0) {
                const block_meta &bv = wbm->blocks[victim];
                if (!bv.resident || bv.is_pinned || victim == exclude_idx) victim = -1;
            }
        } else if (wbm->evict_mru) {
            uint64_t newest = 0;
            bool found = false;
            for (const auto &b : wbm->blocks) {
                if (!b.resident)               continue;
                if (b.is_pinned)               continue;
                if (b.block_idx == exclude_idx) continue;
                if (!found || b.last_used_token > newest) {
                    newest = b.last_used_token;
                    victim = b.block_idx;
                    found = true;
                }
            }
        } else {
            uint64_t oldest = static_cast<uint64_t>(-1);
            for (const auto &b : wbm->blocks) {
                if (!b.resident)               continue;
                if (b.is_pinned)               continue;
                if (b.block_idx == exclude_idx) continue;
                if (b.last_used_token < oldest) {
                    oldest = b.last_used_token;
                    victim = b.block_idx;
                }
            }
        }
        if (victim < 0) break;
        picked[victim] = true;
        wbm->blocks[victim].resident = false;   // 临时隐藏 (末尾恢复)
        hidden.push_back(victim);
        simulated_bytes -= wbm->blocks[victim].byte_size;
        out_victims->push_back(victim);
        ++n_picked;
    }
    for (int v : hidden) wbm->blocks[v].resident = true;  // 恢复; mark_evicted 由调用方做
    return n_picked;
}

size_t wbm_max_block_bytes(const weight_buffer_manager *wbm) {
    if (!wbm) return 0;
    size_t mx = 0;
    for (const auto &b : wbm->blocks) {
        if (b.byte_size > mx) mx = b.byte_size;
    }
    return mx;
}

size_t wbm_total_bytes(const weight_buffer_manager *wbm) {
    if (!wbm) return 0;
    size_t s = 0;
    for (const auto &b : wbm->blocks) s += b.byte_size;
    return s;
}

int wbm_max_resident_blocks(const weight_buffer_manager *wbm,
                            size_t budget_mb, size_t kv_bytes, size_t misc) {
    if (!wbm || wbm->blocks.empty()) return 0;
    const size_t budget_bytes = budget_mb * static_cast<size_t>(1024 * 1024);
    if (budget_bytes <= kv_bytes + misc) return 0;
    const size_t avail = budget_bytes - kv_bytes - misc;
    const size_t mxb = wbm_max_block_bytes(wbm);
    if (mxb == 0) return 0;
    size_t n = avail / mxb;
    if (n > wbm->blocks.size()) n = wbm->blocks.size();
    return static_cast<int>(n);
}

int wbm_resident_count(const weight_buffer_manager *wbm) {
    return wbm ? wbm->n_resident : 0;
}

size_t wbm_resident_bytes(const weight_buffer_manager *wbm) {
    return wbm ? wbm->resident_bytes : 0;
}

const block_meta *wbm_get(const weight_buffer_manager *wbm, int idx) {
    if (!wbm) return nullptr;
    if (idx < 0 || static_cast<size_t>(idx) >= wbm->blocks.size()) return nullptr;
    return &wbm->blocks[idx];
}

void wbm_shutdown(weight_buffer_manager *wbm) {
    if (!wbm) return;
    wbm->blocks.clear();
    wbm->current_token = 0;
    wbm->n_resident = 0;
    wbm->resident_bytes = 0;
}

}  // namespace elastic
