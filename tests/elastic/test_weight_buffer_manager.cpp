// tests/elastic/test_weight_buffer_manager.cpp
//
// 单测 WBM 的纯逻辑：注册、mark_resident、mark_evicted、touch、LRU 选取、
// max_resident_blocks 预算计算。不涉及任何 OpenCL。
// 与 test_budget_watcher.cpp 风格一致：手写 main + assert。

#include "weight_buffer_manager.h"

#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using elastic::block_meta;
using elastic::weight_buffer_manager;
using elastic::wbm_get;
using elastic::wbm_init;
using elastic::wbm_mark_evicted;
using elastic::wbm_mark_resident;
using elastic::wbm_max_block_bytes;
using elastic::wbm_max_resident_blocks;
using elastic::wbm_pick_lru_victim;
using elastic::wbm_register_block;
using elastic::wbm_resident_bytes;
using elastic::wbm_resident_count;
using elastic::wbm_shutdown;
using elastic::wbm_total_bytes;
using elastic::wbm_touch;
using elastic::wbm_set_last_use_event;
using elastic::wbm_set_pinned;
using elastic::wbm_evict_to_byte_budget;

namespace {

// 假装 host 端的 mmap 起点；只用作不为 null 的占位指针
char fake_host_pool[4096];

void check_eq_int(int got, int want, const char *msg) {
    if (got != want) {
        std::fprintf(stderr, "断言失败 [%s]: got=%d want=%d\n", msg, got, want);
        std::abort();
    }
}
void check_eq_sz(size_t got, size_t want, const char *msg) {
    if (got != want) {
        std::fprintf(stderr, "断言失败 [%s]: got=%zu want=%zu\n", msg, got, want);
        std::abort();
    }
}

void register_uniform(weight_buffer_manager *wbm, int n, size_t each_bytes) {
    for (int i = 0; i < n; ++i) {
        int rc = wbm_register_block(wbm, i, &fake_host_pool[i], each_bytes);
        if (rc != 0) {
            std::fprintf(stderr, "register_uniform: wbm_register_block(i=%d, host=%p, bytes=%zu) → rc=%d\n",
                         i, (void*)&fake_host_pool[i], each_bytes, rc);
            std::abort();
        }
    }
}

// —— 用例 1：init + register 元数据正确 ——
void test_init_and_register() {
    weight_buffer_manager wbm{};
    int rc_init = wbm_init(&wbm, 4);
    if (rc_init != 0) { std::fprintf(stderr, "wbm_init failed: %d\n", rc_init); std::abort(); }
    register_uniform(&wbm, 4, 50 * 1024 * 1024);

    check_eq_int(wbm_resident_count(&wbm), 0, "初始 n_resident");
    check_eq_sz(wbm_resident_bytes(&wbm), 0, "初始 resident_bytes");
    check_eq_sz(wbm_total_bytes(&wbm), 4 * 50ull * 1024 * 1024, "total bytes");
    check_eq_sz(wbm_max_block_bytes(&wbm), 50ull * 1024 * 1024, "max block bytes");

    for (int i = 0; i < 4; ++i) {
        auto *b = wbm_get(&wbm, i);
        assert(b);
        check_eq_int(b->block_idx, i, "block_idx");
        assert(b->host_ptr != nullptr);
        assert(b->resident == false);
        assert(b->backend_handle == nullptr);
    }

    // 越界 register 应失败
    [[maybe_unused]] int rc = wbm_register_block(&wbm, 100, &fake_host_pool[0], 1);
    assert(rc != 0);
    // 0 字节 register 应失败
    rc = wbm_register_block(&wbm, 0, &fake_host_pool[0], 0);
    assert(rc != 0);

    wbm_shutdown(&wbm);
    std::printf("[OK] test_init_and_register\n");
}

// —— 用例 2：mark_resident / mark_evicted 维护字节核算 ——
void test_resident_accounting() {
    weight_buffer_manager wbm{};
    wbm_init(&wbm, 4);
    register_uniform(&wbm, 4, 10);

    void *fake_handle1 = reinterpret_cast<void *>(0x1001);
    void *fake_handle2 = reinterpret_cast<void *>(0x1002);

    wbm_mark_resident(&wbm, 0, fake_handle1);
    check_eq_int(wbm_resident_count(&wbm), 1, "after mark 0");
    check_eq_sz(wbm_resident_bytes(&wbm), 10, "bytes after mark 0");
    assert(wbm_get(&wbm, 0)->resident);
    assert(wbm_get(&wbm, 0)->backend_handle == fake_handle1);

    wbm_mark_resident(&wbm, 2, fake_handle2);
    check_eq_int(wbm_resident_count(&wbm), 2, "after mark 2");
    check_eq_sz(wbm_resident_bytes(&wbm), 20, "bytes after mark 2");

    wbm_mark_evicted(&wbm, 0);
    check_eq_int(wbm_resident_count(&wbm), 1, "after evict 0");
    check_eq_sz(wbm_resident_bytes(&wbm), 10, "bytes after evict 0");
    assert(!wbm_get(&wbm, 0)->resident);
    assert(wbm_get(&wbm, 0)->backend_handle == nullptr);

    // 重复 evict 是 no-op
    wbm_mark_evicted(&wbm, 0);
    check_eq_int(wbm_resident_count(&wbm), 1, "再次 evict 无副作用");

    wbm_shutdown(&wbm);
    std::printf("[OK] test_resident_accounting\n");
}

// —— 用例 3：LRU pick 选最旧、跳过 exclude_idx、跳过未驻留 ——
void test_lru_pick() {
    weight_buffer_manager wbm{};
    wbm_init(&wbm, 4);
    register_uniform(&wbm, 4, 10);

    // 全部 mark resident
    for (int i = 0; i < 4; ++i) {
        wbm_mark_resident(&wbm, i, reinterpret_cast<void *>(0x100 + i));
    }
    // touch 顺序：3=1, 1=2, 0=3, 2=4  → 最旧是 3
    wbm_touch(&wbm, 3, 1);
    wbm_touch(&wbm, 1, 2);
    wbm_touch(&wbm, 0, 3);
    wbm_touch(&wbm, 2, 4);

    check_eq_int(wbm_pick_lru_victim(&wbm, -1), 3, "最旧应是 3");
    check_eq_int(wbm_pick_lru_victim(&wbm,  3), 1, "排除 3 后最旧是 1");
    check_eq_int(wbm_pick_lru_victim(&wbm,  1), 3, "排除 1 后最旧仍是 3");

    // 驱逐 3 后，pick 不应再选 3
    wbm_mark_evicted(&wbm, 3);
    check_eq_int(wbm_pick_lru_victim(&wbm, -1), 1, "evict 3 后最旧是 1");

    // 全部驱逐后无候选
    wbm_mark_evicted(&wbm, 0);
    wbm_mark_evicted(&wbm, 1);
    wbm_mark_evicted(&wbm, 2);
    check_eq_int(wbm_pick_lru_victim(&wbm, -1), -1, "全空时返回 -1");

    wbm_shutdown(&wbm);
    std::printf("[OK] test_lru_pick\n");
}

// —— 用例 4：LRU pick 在 exclude 是唯一驻留时返回 -1 ——
void test_lru_pick_only_excluded() {
    weight_buffer_manager wbm{};
    wbm_init(&wbm, 3);
    register_uniform(&wbm, 3, 10);
    wbm_mark_resident(&wbm, 1, reinterpret_cast<void *>(0xAAAA));
    wbm_touch(&wbm, 1, 1);

    check_eq_int(wbm_pick_lru_victim(&wbm, 1), -1, "排除唯一驻留 → -1");
    check_eq_int(wbm_pick_lru_victim(&wbm, 0), 1, "排除非驻留 → 选 1");

    wbm_shutdown(&wbm);
    std::printf("[OK] test_lru_pick_only_excluded\n");
}

// —— 用例 5：max_resident_blocks 预算计算 ——
void test_max_resident_blocks() {
    weight_buffer_manager wbm{};
    wbm_init(&wbm, 32);
    // 32 个 block，每个 50 MB → total 1600 MB
    register_uniform(&wbm, 32, 50ull * 1024 * 1024);

    // 预算非常宽：4000 MB，KV 200 MB，misc 100 MB → avail = 3700 MB → 74 block
    // 但 n_blocks = 32，应钳到 32
    check_eq_int(wbm_max_resident_blocks(&wbm, 4000, 200ull * 1024 * 1024, 100ull * 1024 * 1024),
                 32, "宽松预算");

    // 预算刚好够 4 个 block + KV + misc：
    //   need = 4*50 + 200 + 100 = 500 MB
    check_eq_int(wbm_max_resident_blocks(&wbm, 500, 200ull * 1024 * 1024, 100ull * 1024 * 1024),
                 4, "正好 4 个 block");

    // 预算只够 KV + misc，没空间装 block
    check_eq_int(wbm_max_resident_blocks(&wbm, 300, 200ull * 1024 * 1024, 100ull * 1024 * 1024),
                 0, "刚好 KV+misc");

    // 预算比 KV+misc 还小
    check_eq_int(wbm_max_resident_blocks(&wbm, 100, 200ull * 1024 * 1024, 100ull * 1024 * 1024),
                 0, "不够 KV");

    wbm_shutdown(&wbm);
    std::printf("[OK] test_max_resident_blocks\n");
}

// —— 用例 6：异构 block size 时按最大 block 保守估算 ——
void test_max_resident_blocks_heterogeneous() {
    weight_buffer_manager wbm{};
    wbm_init(&wbm, 4);
    wbm_register_block(&wbm, 0, &fake_host_pool[0], 10ull * 1024 * 1024);
    wbm_register_block(&wbm, 1, &fake_host_pool[1], 20ull * 1024 * 1024);
    wbm_register_block(&wbm, 2, &fake_host_pool[2], 50ull * 1024 * 1024);   // 最大
    wbm_register_block(&wbm, 3, &fake_host_pool[3], 30ull * 1024 * 1024);

    check_eq_sz(wbm_max_block_bytes(&wbm), 50ull * 1024 * 1024, "max bytes");

    // 预算 = 200 MB，KV=misc=0，按 max=50 MB 算 → 4 个
    check_eq_int(wbm_max_resident_blocks(&wbm, 200, 0, 0), 4, "异构@200MB");

    // 预算 = 120 MB → 120/50 = 2
    check_eq_int(wbm_max_resident_blocks(&wbm, 120, 0, 0), 2, "异构@120MB");

    // 预算 = 49 MB → 0（保守）
    check_eq_int(wbm_max_resident_blocks(&wbm, 49, 0, 0), 0, "异构@49MB");

    wbm_shutdown(&wbm);
    std::printf("[OK] test_max_resident_blocks_heterogeneous\n");
}

// —— 用例 7：set_last_use_event 与 evict 时的 event 清理 ——
void test_last_use_event() {
    weight_buffer_manager wbm{};
    wbm_init(&wbm, 2);
    register_uniform(&wbm, 2, 10);
    wbm_mark_resident(&wbm, 0, reinterpret_cast<void *>(0xDEAD));

    void *fake_event = reinterpret_cast<void *>(0xBEEF);
    wbm_set_last_use_event(&wbm, 0, fake_event);
    assert(wbm_get(&wbm, 0)->last_use_event == fake_event);

    wbm_mark_evicted(&wbm, 0);
    assert(wbm_get(&wbm, 0)->last_use_event == nullptr);

    wbm_shutdown(&wbm);
    std::printf("[OK] test_last_use_event\n");
}

// —— 用例 8：模拟一段流式调度，校验合规 ——
//
// 配置：n=8 个 block，每个 50 MB；max_resident = 3。
// 模拟 forward 顺序访问 block 0..7，每访问一个就 ensure；若 resident > 3 就
// LRU evict（排除当前 block）。最终：resident 集合应是 {5,6,7}（最近 3 个）。
void test_streaming_simulation() {
    weight_buffer_manager wbm{};
    const int n = 8;
    wbm_init(&wbm, n);
    register_uniform(&wbm, n, 50ull * 1024 * 1024);

    const int max_resident = 3;
    uint64_t token = 0;
    for (int il = 0; il < n; ++il) {
        ++token;
        // ensure：如果不驻留就 mark_resident（模拟 OpenCL 包装层完成搬运后调用）
        if (!wbm_get(&wbm, il)->resident) {
            wbm_mark_resident(&wbm, il, reinterpret_cast<void *>(
                static_cast<uintptr_t>(0x1000 + il)));
        }
        wbm_touch(&wbm, il, token);
        // LRU evict 到预算内
        while (wbm_resident_count(&wbm) > max_resident) {
            int v = wbm_pick_lru_victim(&wbm, il);
            if (v < 0) break;
            wbm_mark_evicted(&wbm, v);
        }
        // 不变式：每一步合规
        assert(wbm_resident_count(&wbm) <= max_resident);
    }

    // 最终应驻留最近 3 个：5,6,7
    assert(!wbm_get(&wbm, 0)->resident);
    assert(!wbm_get(&wbm, 4)->resident);
    assert( wbm_get(&wbm, 5)->resident);
    assert( wbm_get(&wbm, 6)->resident);
    assert( wbm_get(&wbm, 7)->resident);
    check_eq_int(wbm_resident_count(&wbm), 3, "末态驻留数");

    wbm_shutdown(&wbm);
    std::printf("[OK] test_streaming_simulation\n");
}

// —— 用例 9：is_pinned 让 LRU 跳过该 block ——
void test_pinned_skipped_by_lru() {
    weight_buffer_manager wbm{};
    wbm_init(&wbm, 4);
    register_uniform(&wbm, 4, 10);
    for (int i = 0; i < 4; ++i) {
        wbm_mark_resident(&wbm, i, reinterpret_cast<void *>(0x100 + i));
        wbm_touch(&wbm, i, static_cast<uint64_t>(i + 1));
    }
    // pin block 0（最旧）。LRU 应跳过它，选 block 1。
    wbm_set_pinned(&wbm, 0, true);
    check_eq_int(wbm_pick_lru_victim(&wbm, -1), 1, "pinned 跳过最旧");

    // 全 pin → 返 -1
    wbm_set_pinned(&wbm, 1, true);
    wbm_set_pinned(&wbm, 2, true);
    wbm_set_pinned(&wbm, 3, true);
    check_eq_int(wbm_pick_lru_victim(&wbm, -1), -1, "全 pin 时 -1");

    // unpin block 2 → 应选 2
    wbm_set_pinned(&wbm, 2, false);
    check_eq_int(wbm_pick_lru_victim(&wbm, -1), 2, "unpin 后选 2");

    wbm_shutdown(&wbm);
    std::printf("[OK] test_pinned_skipped_by_lru\n");
}

// —— 用例 10：wbm_evict_to_byte_budget 批量挑 victim ——
void test_evict_to_byte_budget_picks() {
    weight_buffer_manager wbm{};
    wbm_init(&wbm, 5);
    // 异构 size：10, 20, 30, 40, 50（共 150）
    wbm_register_block(&wbm, 0, &fake_host_pool[0], 10);
    wbm_register_block(&wbm, 1, &fake_host_pool[1], 20);
    wbm_register_block(&wbm, 2, &fake_host_pool[2], 30);
    wbm_register_block(&wbm, 3, &fake_host_pool[3], 40);
    wbm_register_block(&wbm, 4, &fake_host_pool[4], 50);
    for (int i = 0; i < 5; ++i) {
        wbm_mark_resident(&wbm, i, reinterpret_cast<void *>(0x200 + i));
        wbm_touch(&wbm, i, static_cast<uint64_t>(i + 1));  // 0 最旧
    }
    check_eq_sz(wbm_resident_bytes(&wbm), 150, "init bytes");

    // 目标 80 bytes：从最旧的 0 开始挑：10 (剩 140) → 20 (剩 120) → 30 (剩 90)
    // → 40 (剩 50) → 满足 ≤ 80。共选 4 个，剩 block 4 (50 bytes)
    std::vector<int> victims;
    int n = wbm_evict_to_byte_budget(&wbm, 80, -1, &victims);
    check_eq_int(n, 4, "选中 4 个");
    check_eq_int(static_cast<int>(victims.size()), 4, "victims 大小 4");
    check_eq_int(victims[0], 0, "首选最旧 0");
    check_eq_int(victims[1], 1, "次选 1");
    check_eq_int(victims[2], 2, "再选 2");
    check_eq_int(victims[3], 3, "再选 3");

    // 注意：evict_to_byte_budget 不修改 resident 状态（OpenCL 包装层负责）
    check_eq_sz(wbm_resident_bytes(&wbm), 150, "pick 不动 resident");
    check_eq_int(wbm_resident_count(&wbm), 5, "pick 不动 count");

    wbm_shutdown(&wbm);
    std::printf("[OK] test_evict_to_byte_budget_picks\n");
}

// —— 用例 11：byte budget 配合 pin 和 exclude ——
void test_evict_to_byte_budget_pin_exclude() {
    weight_buffer_manager wbm{};
    wbm_init(&wbm, 4);
    // 全 30 bytes
    register_uniform(&wbm, 4, 30);
    for (int i = 0; i < 4; ++i) {
        wbm_mark_resident(&wbm, i, reinterpret_cast<void *>(0x300 + i));
        wbm_touch(&wbm, i, static_cast<uint64_t>(i + 1));
    }
    check_eq_sz(wbm_resident_bytes(&wbm), 120, "init bytes");

    // pin block 0（最旧）、exclude block 1。目标 60 bytes → 只能选 2、3。
    wbm_set_pinned(&wbm, 0, true);
    std::vector<int> victims;
    int n = wbm_evict_to_byte_budget(&wbm, 60, /*exclude=*/1, &victims);
    check_eq_int(n, 2, "只选到 2 个");
    check_eq_int(victims[0], 2, "首选 2");
    check_eq_int(victims[1], 3, "次选 3");

    // 把目标压更小：30。可选剩 60 bytes，pin/exclude 阻断进一步驱逐。
    victims.clear();
    n = wbm_evict_to_byte_budget(&wbm, 30, /*exclude=*/1, &victims);
    check_eq_int(n, 2, "目标 30：能挑的还是只有 2 个");
    // resident_bytes 仍 120（pick 不动状态）。函数返回但目标未达。

    wbm_shutdown(&wbm);
    std::printf("[OK] test_evict_to_byte_budget_pin_exclude\n");
}

// —— 用例 12：byte budget 已满足时返回 0、不动 victims ——
void test_evict_to_byte_budget_already_satisfied() {
    weight_buffer_manager wbm{};
    wbm_init(&wbm, 3);
    register_uniform(&wbm, 3, 10);
    wbm_mark_resident(&wbm, 0, reinterpret_cast<void *>(0x400));
    wbm_mark_resident(&wbm, 1, reinterpret_cast<void *>(0x401));
    // resident_bytes = 20

    std::vector<int> victims;
    victims.push_back(99);  // 预填的"脏"内容
    int n = wbm_evict_to_byte_budget(&wbm, 100, -1, &victims);
    check_eq_int(n, 0, "预算已满足");
    check_eq_int(static_cast<int>(victims.size()), 1, "out_victims 不清空（保留 99）");
    check_eq_int(victims[0], 99, "预填内容保留");

    wbm_shutdown(&wbm);
    std::printf("[OK] test_evict_to_byte_budget_already_satisfied\n");
}

}  // namespace

int main() {
    test_init_and_register();
    test_resident_accounting();
    test_lru_pick();
    test_lru_pick_only_excluded();
    test_max_resident_blocks();
    test_max_resident_blocks_heterogeneous();
    test_last_use_event();
    test_streaming_simulation();
    test_pinned_skipped_by_lru();
    test_evict_to_byte_budget_picks();
    test_evict_to_byte_budget_pin_exclude();
    test_evict_to_byte_budget_already_satisfied();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
