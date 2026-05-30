# Per-Op Dispatch Overlap — v3 设计 (未实现)

当前 per-op dispatch 没用上已有 async pipeline (stage 1 io_uring / stage 2+3
xfer_queue / cross-backend cpy_tensor_async). 每个 backend 切换都阻塞 sync.

v3 目标: 4 路 producer-consumer overlap:

```
时间 →
┌──────────────────────────────────────────────────────────────┐
│ disk → host  (io_uring) │ layer N+3 weight │ layer N+4 ...   │
├──────────────────────────────────────────────────────────────┤
│ host → cl_mem (xfer_q)  │ layer N+2 newly arrived weights    │
├──────────────────────────────────────────────────────────────┤
│ activation cpy_async    │ op N output (CPU→GPU)              │
├──────────────────────────────────────────────────────────────┤
│ compute (graph_compute) │ op N on its target backend         │
└──────────────────────────────────────────────────────────────┘
```

## 当前各 stage 已有 async 基础设施

| Stage | 机制 | 在哪 |
|---|---|---|
| Stage 1 (disk → host) | io_uring + chunked lookahead | ggml-cpu-elastic, 已成熟 |
| Stage 2+3 (host → device) | xfer_queue (cl_command_queue per-backend) | ggml-opencl, 已用 |
| Cross-backend activation | `cpy_tensor_async` | ggml-backend iface, split 边界用 |
| Compute | `graph_compute_async` | 所有 backend |

## 现状 — 完全没串起来

当前 `ggml_backend_sched_compute_splits` 的 per-op dispatch 路径:

```cpp
for each group:
    sync(prev_backend)             // 阻塞等
    compute_async(exec_backend, gv) // 串行启动
    prev_backend = exec_backend
```

✗ 没用 cpy_tensor_async
✗ 没触发 elastic backend 的 lookahead prefetch
✗ 每个 backend 切换都串行等

## v3 需要做的 3 件事

### A. 把 dispatch 决策传给 elastic backend 的 lookahead

elastic backend 内部 ensure_phase 提前 K 步预 load weight, 假设这 K 步都在自己 backend
跑. 当 dispatch 把其中一些 op 路由到 OTHER backend 时, 这些 op 的 weight 不该在
"自己" 这里 prefetch, 应该 prefetch 到 OTHER backend.

实现: 加 API `ggml_backend_sched_get_op_target_backend(sched, op_idx)`, elastic
backend 在 ensure_phase 用它判断是否真要 prefetch.

### B. 加 op-level input migration with cpy_tensor_async

dispatch 把 op N 路由到 backend B, 但 op N 的 inputs 在 backend A 内存. 需要:

```cpp
for each input of op N that's on different backend:
    dst = allocate_temp_on(B)
    backend_dst->cpy_tensor_async(A, B, input, dst)  // async, returns immediately
    op->src[i] = dst  // 临时改 ptr
compute_async(B, op)
// 之后 restore op->src[i]
```

难点: 
- temp allocation: 需要预留 scratch space, 或动态 alloc 复杂
- src[i] 改 ptr: 安全性 (compute 并发时不能改其它 ref)
- 跟 ggml-sched 现有 tensor_copy 机制不冲突

可能借用 ggml-sched 的 `tensor_id_copy(id, backend_id, copy_id)` infrastructure
(line 736), 已有支持多 backend copies.

### C. Defer sync until真依赖

当前每个 cross-backend group 都 sync 上一个. 实际只在下一个 group 真读上一个 output 时需要.
依赖分析:

```cpp
deps[group_g][source_g] = whether group g reads any output of source_g
```

只在 deps[next][cur] = true 时 sync cur.

LLM forward 图大部分 op 是顺序依赖 (Qcur 之后 Kcur 等), 但有些是并行的 (Q/K/V proj
都读 attn_norm 输出). 并行 op 间不需要 sync 即可 launch 多个 backend.

## 难度估计

| 任务 | 行数 | 风险 |
|---|---|---|
| A: dispatch 决策传给 elastic | ~100 | 低, 加 API + 内部检查 |
| B: op-level input migration | ~300 | 中, 涉及 alloc + src ptr 改 |
| C: dep 分析 + 选择性 sync | ~150 | 低, scan graph 一次 |
| 总 | ~550 | 中 |

完整 v3 估计 2-3 天工作.

## 实测预期收益

mobile (CPU + Adreno OpenCL) 上, 100 overrides/token 场景:
- 当前: +100 ms dispatch overhead + 真 compute loss
- v3: dispatch overhead → ~30-50 ms (减 50-70%, 来自 sync 减少 + activation 重叠)

若 disk IO 也重叠到 dispatch (内存压力 forced 写出 weight 时):
- 当前: dispatch 跟 disk IO 串行 → ~200 ms 累加
- v3: 重叠 → 单个 stage 限制 max ~100 ms

## v1 已落地, v3 留 backlog

- v1 (本分支 c5e8ded43): per-op dispatch 基本能用, ~1 ms/override overhead, 元
  scheduler 实验充足
- v2 placeholder (撤销): 没找到不动 ggml-sched 大动的简单 win
- **v3 (本文)**: 真要 4 路 overlap, 需要 ~500 行 + 2-3 天

实际部署优先级:
1. ✅ 用 v1 在 cpu-elastic + cpu 上做实验, 验证 routing 策略思路
2. ⏳ 接 GPU build 实测真 mobile 速度差 (CPU vs GPU 真差距决定 dispatch 实用性)
3. ⏳ v3 投资值不值, 要先看 step 2 的真实 GPU↔CPU compute 速度差
