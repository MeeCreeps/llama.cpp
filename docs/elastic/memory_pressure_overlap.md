# Memory Pressure Overlap — O_DIRECT + io_uring 在内存压力下反而最优

## 背景

之前测试 (model 6.4 GB < phone RAM 12 GB) 表明:
- mmap 路径完胜 O_DIRECT (page cache 命中, 无真 disk IO)
- io_uring 收益微小 (没有 stage 1 可以 hide)

但 mobile 真实场景下 memory 是动态的 — 其它 app 抢内存时 model pages
被 evict, 必须真 disk re-read. 这时 page cache 假设不成立.

## 模拟设定

`mem_eater2 5000` 持续触摸 5 GB 防止 kernel 反向回收 → MemAvailable 6 GB,
Cached 1 GB. 3B F16 model 6.4 GB 装不全 cache → mmap path 触发真 page fault.

Profile baseline (mmap, no chunk):
- effective_bw 1207 MB/s (无压力时 3758)
- io_pt 33 ms/graph (无压力时 10.6)
- eval 2695 ms/tok (无压力时 1108)

## 对比

3B F16 B=3500, n=10 tokens, 5GB eater pressure:

| Config | r1 | r2 | min |
|--------|---:|---:|----:|
| A baseline (no chunk, mmap) | 2897 | 3793 | 2897 |
| B chunk256 + std::thread worker (mmap) | 5089 | 5555 | 5089 |
| C chunk256 + io_uring (mmap) | 5130 | 5177 | 5130 |
| **D chunk256 + io_uring + O_DIRECT** | **1913** | **1887** | **1887** |

D 比 A 快 35%, 比 B/C 快 2.7×.

## 分析

### 为什么 mmap 路径在 pressure 下变慢
- Kernel page cache 不断被 eater 抢走
- ensure_phase 访问 host_ptr → page fault → 同步 disk read → 主线程 block
- madvise WILLNEED hint 也没用 (kernel 在 pressure 下不读 ahead)
- chunked + worker (B) / uring (C) 进一步推高 memcpy 量 → 跟 compute 抢 DRAM → 反慢

### 为什么 O_DIRECT + io_uring (D) 大胜
- O_DIRECT 绕过 page cache → 不受 eater 影响, disk 速度可预测 (UFS ~1.2 GB/s 随机)
- io_uring 异步: pread 在 kernel 处理时主线程跑 compute
- chunked: 上一个 chunk compute 时, 下一个 chunk 的 pread 在 UFS 队列
- 真正实现 stage 1 (UFS DMA) 跟 stage 2+3 (CPU 内 memcpy+compute) overlap

### 结论 — 场景决定最优配置

| 场景 | 最优 |
|------|------|
| Model 装得下 RAM, 无 pressure | mmap + 无 chunked (baseline) ~1100 ms/tok |
| Model 装得下, 有 chunked overlap 收益 | mmap + chunk=256 + worker ~870 ms/tok |
| **Model 不全装下 / 有 memory pressure** | **O_DIRECT + io_uring + chunk=256** ~1887 ms/tok |
| Model >> RAM (forced disk-bound) | O_DIRECT + io_uring (3B F16 没装到这个) |

mobile inference 实战中 memory 状态是动态的, 应该根据当前 pressure 切配置:
- 检测 MemAvailable, 高时用 mmap baseline, 低时切 O_DIRECT+uring chunked
- 或者干脆默认 O_DIRECT+uring chunked, 接受 mmap-hit 场景 1700 vs 1100 的微弱亏损

## 实现细节

`GGML_ELASTIC_DIRECT_IO=1 GGML_ELASTIC_URING=1 GGML_ELASTIC_CHUNK_SIZE=256` 一起开.
chunked path 已经在 dynamic-phase1 cherry-pick 到这分支.
io_uring 实现见 `src/llama-uring.cpp`.
