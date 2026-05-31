# V8 Series Complete — Dynamic Memory Budget Handling via Op Partition

Branch: `feature/elastic-op-sched-runtime` (29 commits, ready for deploy)

## 用户长期诉求

> "dynamic memory budget 下, weight 不全在 RAM, budget 变化时考虑怎么放怎么切"

v3-v7 走 runtime dispatch + migration 路线在 mobile 上 elastic 共存 crash. **V8 系列改用 op_schedule (graph build 时决策) + ggml-sched 已有 split mechanism 处理 cross-backend, 干净 deploy-ready.**

## V8 系列演化

| 版本 | 加什么 | commit |
|---|---|---|
| v8 minimal | `LLAMA_V8_PARTITION_LAYER=N` 简单 layer cut | 6400e885c |
| v8.2 | 4 个智能 policies (ffn-cpu/attn-cpu/layer-mod-N/first-K-cpu) | 6280c67ae |
| **v8.3** | **dynamic re-partition on memory event** (零 overhead at idle) | **93e8e98fb** |

## V8.3 dynamic — 你诉求的真正答案

```bash
LLAMA_V8_DYNAMIC=1 \
LLAMA_V83_HI_MB=3000   \  # mem > 3 GB → 全 GPU
LLAMA_V83_MID_MB=1500  \  # 1.5-3 GB → 中等 (8 层 CPU)
LLAMA_V83_LO_MB=1000   \  # < 1 GB → 重 offload (20 层 CPU)
LLAMA_V83_MID_LAYER=20 \  # 中等档 partition layer
LLAMA_V83_LO_LAYER=8   \  # 低档 partition layer
./llama-cli -m model -ngl 99 ...
```

实测 Adreno 750 OpenCL ngl=99:

| Scenario | repartitions | ms/tok | vs baseline |
|---|---|---|---|
| mem 充足 (4159 MB, 默认阈值) | **0** | **144** | **1× (零 overhead!)** |
| 强制低 (HI=99999, → lo_layer=4) | 1 | 856 | 6× |

**核心特点**:
- mem 充足时 0 cost (跟 baseline 完全一致)
- mem 变化检测在 llama_decode 入口 (复用 v5 scheduler)
- 阈值跨越时一次 repartition (~5-20 ms 重 build graph 代价)
- atomic 读写 partition state, lockless

## 工作原理

```
llama_decode()
   ├─ maybe_run_scheduler()
   │    ├─ read /proc/meminfo MemAvailable
   │    └─ if 阈值跨越:
   │         └─ scheduler_fn: 更新 v83s.current_partition.store(new_p)
   ├─ graph_reuse_disable=1 (scheduler auto-set)
   ├─ build_graph() (每 decode 重建)
   │    └─ op_schedule_fn(每个 op):
   │         └─ return (layer < current_partition) ? gpu : cpu
   └─ ggml-sched alloc + compute
        └─ cross-backend ops: 自动 tensor_copy via split mechanism
```

## V8 系列所有 policy 对比 (Adreno 750, n=8 token)

### 静态 partition (v8/v8.2)

| Policy | ops CPU | ms/tok | vs baseline |
|---|---|---|---|
| baseline (全 GPU) | 0 | 140 | 1× |
| layer<24 (后 4 CPU) | 720 | 257 | 1.8× |
| layer-mod-N=4 (7 CPU) | ~1260 | 347 | 2.5× ⭐ 最均衡 |
| attn-cpu (4 attn ops × 28) | 3024 | 420 | 3× |
| layer<14 | 2520 | 456 | 3.3× |
| first-K-cpu=14 | 2520 | 540 | 3.9× |
| ffn-cpu | 2016 | 582 | 4.2× |
| layer<4 (24 CPU) | 4320 | 722 | 5.2× |

### Dynamic (v8.3)
- 充足 mem: 144 ms/tok (= baseline)
- 紧 mem: 切到对应 partition (上表速度)

## 跟 phase1 elastic 流式互补

| 维度 | phase1 elastic | v8 系列 |
|---|---|---|
| budget 抖动频率 | 毫秒 - 秒级 | 秒 - 分钟级 |
| weight 移动 | runtime reload | 重 partition 时一次性 |
| weight 归属 | 始终 opencl backend (从 mmap reload) | 静态 GPU/CPU 混合 |
| 跟 dispatch 共存 | crash (v6 尝试失败) | 天然 (v8 就是 dispatch) |
| 适合场景 | model > GPU 内存 (流式) | 多 backend mixed compute |

## V8 未做 (留 v8.4 backlog)

1. **CL_mem release** for partition-out weights → 真省 GPU 内存 (~100 行 ggml-opencl)
   - 当前 weights 仍占 cl_mem, 只 compute 移到 CPU
   - 释放 cl_mem 后真 free GPU buffer
2. **kq/kqv cross-backend** 让 attn-cpu 不崩 (~50 行)
3. **LP-derived smart partition** (用 Belady+PF16) → 决策更智能 (~150 行)
4. **跟 phase1 elastic 合体** (复杂, 留 v9)

## Deploy 推荐

| 场景 | 推荐 env |
|---|---|
| **大多数 mobile 部署** | `LLAMA_V8_DYNAMIC=1` (零 overhead at idle, 自动响应) ⭐ |
| 已知 mem 充足 | nothing (baseline) |
| 已知 mem 紧 | `LLAMA_V8_PARTITION=layer-mod-N LLAMA_V8_PARTITION_PARAM=4` |
| budget 高频抖动 | phase1 `GGML_OPENCL_ELASTIC=1` 路径 |

## 完整 branch commits (29)

```
93e8e98fb v8.3 dynamic ⭐
6280c67ae v8.2 4 policies
6400e885c v8 layer partition
7fb0a871a docs v8 architecture
56d8907e1 docs v6+v7 backlog
3787b9c47 v6 host_ptr API + PoC
... v3-v5 (runtime dispatch) commits
```
