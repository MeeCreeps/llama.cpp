# Elastic Inference 完整 Pipelining 总结

mobile LLM inference 三大维度的 pipelining 优化探索全记录:
1. **Weight loading** (disk → backend buffer): elastic backend WBM + io_uring + chunked
2. **Op backend assignment** (CPU vs GPU): op-sched runtime hook in graph_get_cb
3. **Data transfer** (CPU↔GPU activation): ggml-sched 自动

## 三层 pipeline 现状

### Layer 1: Disk → Backend Buffer (weight load)

| 机制 | 实现 | 实测收益 |
|------|------|---------|
| mmap + madvise WILLNEED | kernel async readahead 隐式 stage 1 overlap | mmap path baseline |
| O_DIRECT + split pread | head/tail bounce, middle direct 写 region (省 99% memcpy) | bw 1017→1265 MB/s (+24%) |
| Worker thread chunked | 主线程 compute chunk N, worker memcpy ensure chunk N+1 | mmap path 17% (B=3500) |
| io_uring chunked | 主线程 compute, kernel 异步 pread (无线程开销) | O_DIRECT 35% (5GB pressure) |
| Multi-chunk lookahead | 预提交 K chunks, kernel UFS QD>1 并行 | 无压力反慢, 压力下未充分测试 |

### Layer 2: Op Backend Assignment

`LLAMA_OP_SCHED=<strategy>` (`src/llama-context.cpp::graph_get_cb`):
- per-decode 重新决定每个 MUL_MAT 跑哪个 backend
- 5 strategy: all-cpu / all-gpu / alternate / memory / external
- 验证: 4 strategy 输出 30-token 完全一致
- alternate 模式: graph splits = 185 (vs baseline 2), 184 个 CPU↔GPU 切换

### Layer 3: Data Transfer

ggml-sched 自动在 backend 边界插 copy ops. 验证:
- alternate 模式 token 输出正确
- 跟普通 ggml-sched 完全兼容
- 不需要额外代码

## 3-stage Pipeline 在 mobile (Adreno+UFS) 上的硬件限制

probe_overlap.cpp 实测 Adreno 750 上 stage 2 (host→GPU DMA) 跟 stage 3 (GPU compute)
**0% overlap** — driver 强制 serialize. xfer_queue + compute_queue 两条 queue 在
软件层并行, 硬件层物理串行 (共享 memory controller).

唯一真 overlap 路径:
- Stage 1 (disk → host, UFS DMA, 独立 hardware) ∥ Stage 2+3 (GPU 串行)

PowerInfer-2 等论文实现的真 3-stage 需要:
- NPU + GPU (各自独立 memory path) — 本项目 Hexagon NPU 不可用
- Neuron-cluster 子 tensor 切分 — 要改 ggml-cpu matmul kernel
- Sparsity-aware prefetch — 需要 sparsity-refit 模型 (TurboSparse-Llama 等)

我们在 ggml-cpu/ggml-opencl 框架内能挤的 overlap 上限基本到了:
- mmap CPU + chunked worker: 17% 收益
- O_DIRECT + io_uring + chunked: pressure 下 35% 收益

## Dynamic Budget Pipeline (终极愿景)

终极目标: B(t) 变化 → schedule (weight load + op assignment) 跟随变化.

进展:
- ✅ B(t) infrastructure: budget_watcher 后台 thread + atomic get
- ✅ Weight load 响应 budget: elastic backend evict_to_byte_budget
- ✅ Op assignment infra: op-sched hook
- ✅ "memory" strategy 读 /proc/meminfo 跟随系统压力
- 🟡 budget_watcher → op-sched 联动 (未做, infra 都在了, 几十行代码)

实测 dynamic budget (无 op-sched):
- 无 pressure: dynamic 比 static-safe 快 3.4× (能用闲置 RAM)
- 5GB pressure: 持平 (OS 已经把 RAM 限死, dynamic 优势消散)
- 3GB pressure: dynamic 仅快 4%

## 全 tag 路径

| Tag | 内容 |
|-----|------|
| `elastic-3b-f16-sweep` | Static elastic baseline (CPU/GPU MRU fixes) |
| `elastic-dynamic-full` | Dynamic + 完整 GPU+CPU elastic |
| `elastic-dynamic-uring` | io_uring chunked overlap |
| `elastic-mmap-dynamic-3x` | Dynamic vs Static 时变 3.4× |
| `elastic-pressure-odirect-uring-win` | 压力下 O_DIRECT+uring win |
| `elastic-op-sched-mvp` | Per-op runtime scheduler 验证 |

最新 HEAD `feature/elastic-op-sched`: 77ca0c39b (io_uring lookahead).

## 下一步 (留给未来)

1. budget_watcher → op-sched 联动: per-decode 读 B(t), 在多个 memory band 之间
   切换 op assignment 策略
2. 实测 NPU 路径 (如果 Hexagon SDK 可用)
3. Sparsity refit + neuron-cluster 切分 (PowerInfer-2 套路)
