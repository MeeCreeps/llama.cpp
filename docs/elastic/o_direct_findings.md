# O_DIRECT + Page Cache 影响实测 (feature/elastic-o-direct)

Phone OnePlus 12 / Snapdragon 8 Gen 3 / UFS 4.0. 2026-05-25.

## 背景

之前所有 elastic 测试都隐式跑在 page cache 命中条件下（mmap 的 file pages 全在
RAM）。要回答"如果 model > RAM, elastic 速度怎样"必须绕开 page cache 真打 disk.

## 工具

1. `src/llama-mmap.cpp`: 加 `llama_file::pread_direct()` API (O_DIRECT lazy fd
   + 对齐 bounce buffer + 自动 fs block size 探测).
2. `tests/elastic/io_bench.cpp`: 比较 mmap warm / mmap+fadvise / O_DIRECT 3 种
   读取模式的带宽.
3. `tests/elastic/elastic_io_bench.cpp`: 模拟 elastic reload 模式 (每 token 按
   tensor sizes 分散 pread N 次), 测真 disk IO 时间.

## 实测带宽 (io_bench)

| 模式 | 1B file (663 MB) | 8B file (4.6 GB) |
|---|---:|---:|
| mmap warm (page cache 全命中) | **7112 MB/s** | 2450 MB/s |
| mmap + posix_fadvise(DONTNEED) | 5247 MB/s | 2858 MB/s |
| **O_DIRECT (真 UFS read)** | 3569 MB/s | **3312 MB/s** |

**关键发现**:
- 小 file (<RAM/4): mmap warm 7 GB/s 占优 (纯 RAM speed)
- 大 file (>RAM/2): mmap 被 cache thrash 拖慢, **O_DIRECT 反而比 mmap 快** (8B 上 3.3 vs 2.5 GB/s)
- UFS 4.0 稳定读约 3.3 GB/s, 跟厂商标称 ~3 GB/s 一致

## Elastic reload 模式实测 (elastic_io_bench)

每 token 模拟 evict N MB, 分成 Llama 一层的 7 个 tensor (4/1/1/4/16/16/16 MB)
分别 pread 读取:

| 场景 | mmap warm | mmap+fadvise/tok | **O_DIRECT** |
|---|---:|---:|---:|
| 1B 60% (234 MB/tok) |  9.2 ms (25 GB/s) | 43 ms | **92 ms** (2.5 GB/s) |
| 8B 80% (800 MB/tok) | 43 ms (18 GB/s) | 314 ms | **326 ms** (2.5 GB/s) |

## vs 实测 CPU/GPU elastic

| 场景 | 实际 elastic io | O_DIRECT 真 disk 预测 | 差距 |
|---|---:|---:|---|
| CPU 1B 60% | 55 ms | 92 ms | 比 disk 快 40% (page cache 部分命中) |
| **CPU 8B 80%** | **774 ms** | **326 ms** | **比 O_DIRECT 慢 2.4×** ⚠️ |
| GPU 8B 80% (reload + kernel) | 467 ms | ~330 ms + GPU kernel | 接近 |

**重大发现**: CPU elastic 在 8B 80% 实测 **比直接 O_DIRECT 读 disk 还慢 2.4×** !

意思是 CPU elastic 当前实现的 evict loop / bookkeeping / page fault overhead
加起来 (~450 ms/tok) 大于真正的 disk IO 工作 (326 ms/tok). 用 O_DIRECT 重写
elastic 数据通路理论上能砍掉这 450 ms 开销.

## drop cache 实测无效

试 `posix_fadvise(POSIX_FADV_DONTNEED)` 在跑前 drop file cache, 再 run elastic:

| 场景 | 热 cache | 冷 cache (drop_run) | 差异 |
|---|---:|---:|---:|
| CPU 1B 60% | 116 ms | 115 ms | < 1% |
| CPU 8B 80% | 1022 ms | 1005 ms | < 2% |
| GPU 8B 80% | 658 ms | 681 ms | < 4% |

**原因**: model load 阶段 llama-cli 必须读整个 file (用 mmap 主动 touch 或 read),
**自动重新 warm 整个 cache**. 再之后 inference 阶段全是 cache 命中.

要真实 cold cache 测试只能:
1. 后台持续 drop cache (另一进程, 噪声大)
2. **真 O_DIRECT 改 reload 路径** ← 干净方案

## GPU SOA reload 接入 O_DIRECT (commit 9de8f1c21)

实测 1B-Q4 GPU mmap vs O_DIRECT (GPU ceiling 21 ms/tok):

| budget | GPU mmap | GPU O_DIRECT | mmap io | direct io | slowdown |
|---|---:|---:|---:|---:|---:|
| 95% (1013) | 23 | 28 | 2 | 7 | **+22%** |
| 70% (848) | 67 | 107 | 46 | 86 | **+59%** |
| 60% (781) | 142 | 196 | 121 | 175 | **+38%** |
| 50% (715) | 180 | 293 | 159 | 273 | **+63%** |
| 40% (649) | 267 | 382 | 246 | 361 | **+43%** |
| 30% (582) | 335 | 475 | 314 | 454 | **+42%** |

**关键结论**:
- DIRECT_IO 全档慢 22-63% — 走真 UFS 4.0 (~3.3 GB/s) 而不是 page cache (7 GB/s)
- mmap io 带宽 1.5-3 GB/s (部分 cache + 部分 fault 混合)
- DIRECT_IO io 带宽 ~2-3 GB/s (接近 UFS 标称)
- 差距比 io_bench 预测小 (io_bench mmap warm 7 vs O_DIRECT 3.6, 差 2×) 因为 elastic
  框架自带的 per-op WBM check + kernel dispatch overhead 摊薄了 IO 路径影响

**意义**: 这个 +42-63% 就是 "model > RAM, page cache 帮不了" 真实场景的成本.
当前 GPU elastic 数据在 cache hit 假设下偏乐观 ~40-60%.

## CPU + GPU 都接 O_DIRECT 后 1B-Q4 完整对比

CPU elastic (cherry-pick from feature/elastic-cpu-phase1) + GPU SOA reload 都接
`pread_direct`. `GGML_ELASTIC_DIRECT_IO=1` 同时控制两边. 冷机起每跑前 sleep 20-25s.

| budget | CPU O_DIRECT | GPU O_DIRECT | 谁赢 |
|---|---:|---:|---|
| 95% | 52.5 | 24.2 | GPU 2.17× |
| 80% | 52.6 | 28.1 | GPU 1.87× |
| 70% | 292.9 | 110.2 | GPU 2.66× |
| 60% | 293.9 | 197.3 | GPU 1.49× |
| 50% | 293.3 | 295.0 | tied |
| 40% | 292.4 | 382.2 | CPU 1.31× |
| 30% | 293.9 | 471.8 | CPU 1.61× |

**Crossover at 50% budget**:
- ≥ 60%: GPU 赢 (xfer_queue + SOA prefetch overlap reload IO)
- ≤ 40%: CPU 赢 (CPU 固定 overhead 比 GPU 线性 IO 工作量小)

**CPU plateau 现象**: 70%-30% 全 293 ms 一致, 说明 CPU elastic 现在的瓶颈是
framework 开销 (evict loop + ensure 逻辑), 不是 IO 量本身. mmap path 也是 plateau
116 ms — 加 O_DIRECT 后净 IO 影响 = 293-116 = ~177 ms (跟 budget 也无关).

**对比 mmap baseline** (1B 70%/60%):
- CPU mmap: 115 ms → O_DIRECT: 293 (+154%, 真打 UFS vs page cache RAM 速度)
- GPU mmap:  84 ms → O_DIRECT: 110 (+32%, pipeline + overhead 摊薄)

GPU O_DIRECT 性能损失小是因为:
1. xfer_queue 上 pread 跟 compute_queue 上 matmul 并行
2. SOA prefetch lookahead 32 op 提前发 reload
3. CPU 单线程严格串行, reload 全计入 critical path

## CPU async prefetch (未做, 留 phase 2)

CPU 当前是严格单线程: pread → matmul → pread → matmul 串行.
要让 CPU 也 overlap, 需要加 worker thread + lookahead:
```
worker thread: 跑 pread_direct 把未来 N op 的 weight 读到 region
main thread: matmul 用 condvar 等 'tensor.ready' flag
```

工作量约 150-200 行. 预期 CPU 70% 从 293 ms 降到 100-150 ms (跟 GPU 接近).
但 CPU framework 自带 116 ms overhead 仍在 → 下限 ~150 ms.

## 实现细节

- `llama_mmap_registry` (filename + base + size): 反查 host_ptr -> file path
  - 存 filename **字符串** 不是 llama_file* (loader 析构后 file* 失效)
  - 用 `/proc/self/fd/N` symlink 拿绝对路径
- `llama_pread_direct(filename, dst, off, len)`: 独立 helper, lazy O_DIRECT fd
  cache, 对齐 bounce buffer (跟 ioctl FIONREAD 探测 fs block size)
- ggml-opencl SOA reload: env GGML_ELASTIC_DIRECT_IO=1 启用
