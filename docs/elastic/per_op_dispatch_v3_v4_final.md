# Per-Op Runtime Dispatch — v3 + v4 Final (Mobile)

Branch: `feature/elastic-op-sched-runtime` (17+ commits on top of `feature/elastic-op-sched`)

## 演化

| 版本 | 行为 | mobile 实测 |
|---|---|---|
| v1 (`e387208f9`) | per-op hook + grouping + cache | 100% silent fallback (cl_mem buft 不容兼 CPU) |
| v2 (safety check) | 安全 fallback 避免 corrupt | 0 override 真生效 |
| **v3** (`6b5718c2e`) | input + output migration via `ggml_backend_tensor_copy` | 真切了, 但每 decode alloc/free → 9× slower |
| **v4** (`6c7d6ff4c`) | + temp buffer pool 跨 decode 复用 | 6.3× slower (减 31%) |

## API (无变化)

```c
// 注册 hook
llama_set_op_runtime_dispatch(ctx, fn, ud);

// 启用 cross-backend migration (mobile UMA 必需)
GGML_SCHED_RUNTIME_DISPATCH_MIGRATE=1

// 调试
GGML_SCHED_RUNTIME_DISPATCH_DEBUG=1
```

## v3 实现细节

1. **预扫描** 每个 op 调 hook 拿 target_backend
2. **兼容检查** target 不支持 src buft → 标记为 migrate (v3) 或 silent fallback (v2)
3. **Input migration** (compute 前):
   - 对每个不兼容 src: `alloc temp on target` + `ggml_backend_tensor_copy(src, tmp)` + swap `op->src[i]`
4. **Output redirect** (compute 前):
   - op 自己是 output tensor, 它的 buffer 不兼容 target → save `op->data` + `op->buffer`, redirect to target's host buffer
5. **Compute** `ggml_backend_graph_compute_async(target, op)` + sync
6. **Output writeback**: 把 host temp data 写回原 backend's storage via `ggml_backend_tensor_set`
7. **Restore** op->src[i], op->data, op->buffer

## v4 cache 实现

```cpp
struct ggml_backend_sched {
    ...
    struct migration_buf_entry {
        int backend_id;
        size_t size;             // power-of-2 bucket
        ggml_backend_buffer_t buf;
        bool in_use;
    };
    std::vector<migration_buf_entry> migration_pool;
};

// Get buffer:
//   round need_bytes up to power-of-2
//   找 pool 里 !in_use + same backend + size>=bucket 的 entry, 返回
//   没有则 alloc 新的, 入 pool

// End-of-split: 标 in_use=false (不 free, 留给后续复用)
// End-of-sched: free all
```

## 实测对比 (Adreno 750 OpenCL, ngl=99, n=8 token, single run)

| Config | overrides | ms/tok | vs baseline |
|---|---|---|---|
| baseline (all GPU) | 0 | ~140 | 1× |
| v3 attn-cpu | 168 | 749 | 5.4× slower |
| **v4 attn-cpu** | 168 | **591** | 4.2× (减 21%) |
| v3 alternate | 886 | 1277 | 9.1× slower |
| **v4 alternate** | 886 | **882** | 6.3× (减 31%) |
| v3 ffn-cpu | 84 (大权重) | 2208 | 15.7× slower |

## 慢的成本来源分析

每 migration on Adreno UMA (单 mul_mat with 6-48 MB weight):
- input copy (3 srcs avg 20 MB): ~10-15 ms via clEnqueueReadBuffer
- CPU F16 GEMM: ~5-50 ms depending on weight size (CPU 比 Adreno 慢 5-10×)
- output writeback: ~3 ms via clEnqueueWriteBuffer
- per-op dispatch overhead (grouping + sync): ~1 ms
- **total: ~20-70 ms per migrated mul_mat**

**CPU GEMM 时间占大头**, 不是 dispatch overhead 主导. v4 cache 主要省掉 alloc/free
overhead (~3-5 ms/migration). 剩下慢的部分需要更深的改造.

## 何时用 / 何时不用

✅ **用**:
- GPU 内存紧 (OOM 边缘): 临时把几个非关键 op 推 CPU
- 实验/验证 routing 策略: 不用改 graph topology, 写 hook 即可试
- 混合精度兜底: GPU 上某 op 卡 bug, 临时切 CPU 验证正确性

❌ **不用**:
- 生产推理: 100+ overrides/tok = 6-9× throughput 损失
- GPU 慢于 CPU 的少见情况 (Adreno F16 一般比 CPU 快)
- Realtime / 低延迟场景

## v5 优化方向 (留 backlog, 工作量大)

1. **Async cpy_tensor_async** — 所有 backend 现在都 NULL, 需扩展 ggml-opencl
   用 xfer_queue + cl_event 实现真异步 (~150 行)
2. **Adreno UMA 零拷贝** — clEnqueueMapBuffer/Unmap 替代 Read/WriteBuffer,
   省 ~3-10 ms/transfer (~80 行)
3. **Migration + compute pipelining** — 把 group B 的 input migration 跟
   group A 的 compute overlap (~200 行)
4. **Disk IO 重叠** — 让 elastic backend 的 io_uring lookahead 知道 dispatch
   决策, 不再 prefetch 错 backend (~100 行)
5. **Output skip 优化** — 若 output 立即被另一个 backend 读, skip writeback,
   直接给那个 backend 用 (~150 行)

v5 总估 700 行, 3-4 天工作。预期可把 6-9× slowdown 减到 3-5× (mobile 上 CPU 真
GEMM 比 GPU 慢的成本是物理上限).

## 文件

- `ggml/include/ggml-backend.h` — `ggml_backend_sched_set_runtime_dispatch` API
- `ggml/src/ggml-backend.cpp` — compute_splits per-op dispatch + v3 migration + v4 pool
- `ggml/src/ggml-opencl/CMakeLists.txt` — Android `--unresolved-symbols=ignore-all` link fix
- `include/llama.h` — `llama_set_op_runtime_dispatch` API
- `src/llama-context.{h,cpp}` — trampoline + ctx 字段
- `tools/main/main.cpp` — 5 builtin demo policies (alternate/layer-half/ffn-cpu/attn-cpu/memory-driven)

## Commits 索引

```
6c7d6ff4c perf(ggml-sched): v4 migration temp buffer pool (跨 decode 复用)
6b5718c2e feat(ggml-sched): v3 完整版 — cross-backend per-op dispatch (input + output migration)
d84b4e143 feat(ggml-sched): v3 input migration for per-op runtime dispatch (WIP)
ce0de9c17 fix(build): Android ggml-opencl link 加 --unresolved-symbols=ignore-all
aa3a8d2de docs(elastic): per-op dispatch overlap v3 设计
c808a0a99 perf(ggml-sched): defer sync to backend transitions
a69390219 perf(llama-context): op_runtime_dispatch 不强制 graph_reuse_disable
6b20818bb docs(elastic): per-op runtime dispatch benchmark
8994aca0f fix(demo): runtime dispatch policies 正确解析真实 op naming
4578e3f30 feat(ggml-sched): GGML_SCHED_RUNTIME_DISPATCH_DEBUG diagnostic
873ac1b0b feat(demo): 5 个 builtin runtime dispatch policy
7b56ac4ce feat(ggml-sched): input buffer-type compatibility check
c5e8ded43 docs(elastic): per-op runtime dispatch — API + design
e952bb702 perf(ggml-sched): cache runtime dispatch hook result
a0df7a00e perf(ggml-sched): group consecutive same-target ops
e387208f9 feat(ggml-sched): true per-op runtime backend dispatch hook
```
