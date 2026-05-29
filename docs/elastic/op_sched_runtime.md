# Op-level Runtime Scheduler — 验证完成

llama.cpp 上游只支持 layer-level offload (`-ngl N`). 本 MVP 加 **per-op 运行时
动态分配**, 每次 decode 重新决定每个 op 跑哪个 backend, 跨 backend 数据传输
由 ggml-sched 自动 handle.

## 使用

```bash
LLAMA_OP_SCHED=<strategy> [LLAMA_OP_SCHED_DEBUG=1] ./llama-cli -ngl 99 ...
```

| Strategy | 含义 |
|----------|------|
| (空) | 默认 ggml-sched 自动 (跟原 llama.cpp 行为一致) |
| `all-cpu` | 强制所有 MUL_MAT 到 CPU backend |
| `all-gpu` | 强制所有 MUL_MAT 到第一个非 CPU backend |
| `alternate` | **per-OP** 轮换: graph 内每个 MUL_MAT 交替 CPU↔GPU |
| `memory` | /proc/meminfo MemAvailable ≥ thresh 时选 GPU, 否则 CPU |
| `external` | 每次 decode 读 `LLAMA_OP_SCHED_BACKEND` env (`cpu`/`gpu`) |

`LLAMA_OP_SCHED_DEBUG=1` 打印 per-node 分配 + post-compute splits 数.

## 实现位置

`src/llama-context.cpp::graph_get_cb()`. Hook 在每个 node 被加入 cgraph 时调用,
通过 `ggml_backend_sched_set_tensor_backend(sched, node, backend)` 覆盖默认决策.

## 验证 — 3B F16, -ngl 99, 30 token decode

### Correctness (4 strategies 输出完全一致)
```
baseline:  "The story begins: \"The year was 1922, and the world was a very 
            different place. The war was over, and the world was slowly rebuilding. But"
all-cpu:   same
all-gpu:   same  
alternate: same
```

### Backend splits (proof ops 真在跨 backend 执行)
```
[op-sched] post-compute graph splits = 185  (alternate)
[op-sched] post-compute graph splits = 2    (baseline / all-gpu)
```
alternate 模式触发 185 个 backend 切换 = ggml-sched 自动插入 184 个 CPU↔GPU
copy ops, 数据 transfer 在 backend 间正常流转.

### Perf (3 runs min/avg, ms/tok)

| Strategy | min | avg | vs baseline |
|----------|----:|----:|:------------|
| baseline | 176 | 177 | — |
| all-cpu | 879 | 1046 | **5.9× slower** |
| all-gpu | 176 | 178 | ≈ baseline (默认就走 GPU) |
| alternate | 589 | 727 | **4.1× slower** |

### 分析

- baseline ≈ all-gpu: -ngl 99 默认所有 MUL_MAT 走 GPU
- all-cpu 慢 5.9×: F16 GEMM CPU compute 慢 (memory bandwidth bound), 加上 weight
  跨 backend 拷贝 (model 在 GPU buffer, CPU 算需要拉过来)
- alternate 慢 4.1×: 一半 ops CPU (慢) + 185 个 copy ops 在 ggml-sched 自动插入
  之处, transfer 成本明显

## 跟 elastic budget 的协同 (留待后续)

当前 op-sched 只决定 compute 跑哪儿. Elastic budget 决定 weight 哪些 resident.
两个机制正交但可以协同:
- 内存紧张: budget 缩 weights → 部分 weight 不在 GPU → 那些 op 应该 schedule 到 CPU
- 内存空闲: budget 放 weights → GPU 有 capacity → schedule 大 ops 到 GPU

要做的: 让 op-sched 读 `budget_watcher_get()` + per-tensor residence 状态, 输出
基于这些信号的 schedule. 这是下一步集成工作.

## 设计要点

- **只移动 compute, 不移动 weight 存储**: 跨 backend 时 ggml-sched 自动插 copy ops
  搬运 activation. Weight 留在 model loader 分配的 buffer.
- **每个 decode 重新决定**: counter / strategy 检查在 build_graph 阶段, build_graph
  per decode 跑一次 (除非 graph reuse).
- **跟现有 ggml-sched 完全兼容**: 通过 set_tensor_backend 公开 API, 不破坏现有
  decision logic, 只是给出更优先的提示.
