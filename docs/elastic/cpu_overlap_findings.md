# CPU Elastic IO Overlap — 真 overlap 是 kernel readahead, 不是 worker thread

## 问题

3B F16 sweep 显示 CPU 在 sub-budget 下 IO 时间明显多于 GPU:

| B (MB) | CPU IO (total - ceiling) | GPU IO |
|--------|-------------------------:|-------:|
| 5000   | 213                      | 291    |
| 3500   | 989                      | 822    |
| 2500   | 1292                     | 1095   |
| 1500   | 2475                     | 1727   |

GPU 用 `xfer_queue` 让 cl_mem write 跟 compute 真 overlap. CPU 没有对应机制.

## 实验 1: 加后台 prefetch worker thread (失败)

设计: 主线程 evict 完把 just-evicted block enqueue 给 worker, worker 在后台
`pread_direct` 写到 region+offset. 主线程下一个 graph 的 ensure phase 看到
resident=true 跳过. 加 steal 语义 (queue 没开工 main 抢过来同步做).

实测 3B F16 B=5000:
- worker OFF: 4190 ms/tok
- worker ON (sync wait): 11984 ms/tok (**worker 帮倒忙 2.9×**)
- worker ON (steal): 6302 ms/tok (1.5×)

### 失败根因

1. **ggml-cpu compute 是黑盒** — `bctx->cpu->iface.graph_compute(...)` 一次性跑
   完整 graph. ensure_phase 必须在 compute 前把所有 src 都加载好. 无法把
   "compute 跟 prefetch" 在一个 graph 内 overlap.

2. **inter-graph 窗口太短** — 主线程 graph_compute 返回后, 到下次调用 graph_compute
   之间只有 sampling/KV-update 等 host 工作, 大概 10-50 ms. Worker 要 pread
   500 MB (170 ms IO) 来不及.

3. **worker 跟 main 抢资源** — 一旦下次 graph 的 ensure_phase 开始 sync pread,
   worker 还在跑, 两条 pread 抢 disk queue + DRAM 带宽. F16 GEMM 本来就 memory-bound,
   被 worker 抢 → compute 慢.

要做真 intra-graph overlap, 得把 ggml graph_compute 拆成 chunk (per-layer 之类),
chunk 之间插 ensure + prefetch. 是个大改, 跨出 elastic backend 范围.

## 实验 2: mmap + kernel readahead (成功)

`GGML_ELASTIC_PREFETCH=64` 让 ensure_phase 在处理 node `i` 时同时对 node
`i+1..i+64` 的 srcs 调 `posix_madvise(POSIX_MADV_WILLNEED)`. 内核 page-cache
异步 readahead 那些页, 当 compute 真访问时已经 cache 命中.

但 `GGML_ELASTIC_DIRECT_IO=1` 完全绕过 page cache → madvise 失效. O_DIRECT 让
内核异步预读机制全部废掉.

实测 3B F16 sweep (CPU elastic, prefetch_lookahead=64):

| B (MB) | mmap+willneed | O_DIRECT | speedup |
|--------|--------------:|---------:|--------:|
| 7000 (ceiling) | 525 | 542 | 1.03× |
| 5000 (92%)     | 3071 | 4802 | **1.56×** |
| 3500 (62%)     | 1824 | 3250 | **1.78×** |
| 2500 (45%)     | 1635 | 2953 | **1.81×** |
| 1500 (27%)     | 1622 | 2920 | **1.80×** |

**mmap + madvise 是真 overlap**: kernel 在主线程 compute 期间用 read-ahead
threads 把 page 加载到 cache, 不抢主线程 CPU/DRAM 带宽 (kernel I/O thread
独立). 主线程的下一个 memcpy 直接命中 cache → 接近无 IO 等待.

## 实验 3: CPU 默认 LRU 是另一个隐藏 bug

sweep mmap+willneed 后发现 B=5000 (2546 ms/tok) 反而比 B=3500 (3568) 慢, 跟
GPU 上之前看到的 LRU bug 同款症状. 这个分支上 CPU elastic 默认仍是 LRU
(MRU-default 修复在 stream-loader 分支没合过来).

实测 3B F16 (LRU vs explicit MRU):

| B (MB) | LRU 默认 | MRU explicit | speedup |
|--------|--------:|--------------:|--------:|
| 5000   |   2546  |     **699**   | 3.6×    |
| 3500   |   3568  |    **1137**   | 3.1×    |
| 2500   |   1431  |     1393      | ~       |
| 1500   |   1640  |     1580      | ~       |

LRU 在 high-budget 灾难: round-robin 访问下 LRU 评出来的 victim = 最早访问的
layer 0 们 = 下一次马上要用. 命中率近 0. MRU 评出 layer 27 = 下一次 ~28 layer
后才再用, 命中率高.

CPU MRU 默认 fix 已应用到 ggml-cpu-elastic.cpp.

## 最终数据 (CPU MRU + mmap + willneed=64 vs GPU MRU + prefetch pool):

| B (MB) | CPU eval | GPU eval | 谁更快 |
|--------|---------:|---------:|-------:|
| 7000 (ceiling, 100%) | 511 |  178 | GPU 2.9× |
| 5000 (92%)           | 750 |  469 | GPU 1.6× |
| 3500 (62%)           | 1502| 1000 | GPU 1.5× |
| 2500 (45%)           | 1296| 1273 | ~       |
| 1500 (27%)           | 1566| 1905 | **CPU 1.2× faster** |

GPU 优势随 budget 收紧而递减, B=1500 (~75% miss) 时 CPU 反超 — 此时都是
disk-bound, GPU 还要付每次 reload 的 cl_mem write 开销 (即使有 retain pool),
CPU 直接 memcpy 区域内.

## 结论

- 想要真 CPU IO/compute overlap → **用 mmap, 不要 O_DIRECT**, prefetch
  lookahead 调到 32-64
- O_DIRECT 的意义是模拟 "model > RAM, page cache 撑不下" 场景; 在 model 装得
  下时纯粹拖速度
- 业务路径 (model fits in RAM) 应该走 mmap; 测试路径 (强制 disk-bound) 才用
  O_DIRECT
- userspace worker thread 不能补救 O_DIRECT 的丢失 overlap, 因为 ggml-cpu
  compute 不能 yield CPU 给 worker
