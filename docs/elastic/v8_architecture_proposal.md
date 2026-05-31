# v8 架构提案 — 静态分区 + 重 partition (代替 elastic 流式)

## 问题

v6 整合 elastic 流式 + dispatch 失败. 原因: elastic backend 的 reload-on-demand 假设
op 在它 backend 跑, dispatch 把 op 路由走打乱了这个假设. cl_mem 生命周期管理冲突.

## v8 新思路: 静态分区, 不流式

放弃 "weight 跑动" 模型, 改成 "weight 静态归属":

```
每个 weight 在任一时刻有明确归属 backend:
  - GPU (cl_mem 已 alloc + write)
  - CPU (mmap 区, 永不动)

ops 编译时按 weight 归属决定 backend (op_schedule callback).

budget 变化时 (e.g., 别的 app 抢内存):
  - 触发 re-partition 决策 (LP / Belady+PF16 heuristic)
  - 把 weight X 从 GPU 移到 CPU: clReleaseMemObject(X 的 cl_mem)
  - 把 weight Y 从 CPU 移到 GPU: alloc cl_mem + clEnqueueWriteBuffer(Y 的 mmap)
  - 重 build graph (graph_reuse_disable 已支持) 用新归属
```

## 跟现有 phase1 elastic 区别

| 维度 | phase1 elastic (流式) | v8 静态分区 |
|---|---|---|
| weight 移动 | 每个 op 用前 reload (ensure_phase) | 仅 budget 事件时 re-partition |
| reload 时机 | runtime 频繁 | 较少 (~每分钟级) |
| op 跑哪 | 都在 opencl backend (从 mmap reload) | weight 归属决定 (GPU/CPU 混合) |
| 复杂度 | 每 op 一个 ensure check | partition 决策 + 一次性数据搬运 |
| 跟 dispatch 整合 | 难 (cl_mem 生命周期冲突) | 自然 (weight 归属 = op backend) |

## 实现路径 (~500 行)

### 1. Weight partition manager (~150 行)
```cpp
struct weight_placement {
    bool on_gpu;
    cl_mem cl_handle;        // 若 on_gpu
    void *host_ptr;          // 总有 (mmap 区)
};
unordered_map<string, weight_placement> g_partition;

void repartition(int new_budget_mb):
    decisions = run_lp_or_heuristic(weights, new_budget_mb);
    for (w, decision : decisions) {
        if (decision.on_gpu && !g_partition[w].on_gpu) {
            alloc cl_mem + clEnqueueWriteBuffer(host_ptr)
            g_partition[w].cl_handle = cl_mem
            g_partition[w].on_gpu = true
        } else if (!decision.on_gpu && g_partition[w].on_gpu) {
            clReleaseMemObject(g_partition[w].cl_handle)
            g_partition[w].on_gpu = false
        }
    }
```

### 2. Op routing based on partition (~100 行)
```cpp
int op_schedule_fn(node, name, layer, ud):
    if (op is mul_mat):
        weight_name = src[0]->name
        if (g_partition[weight_name].on_gpu): return gpu_id
        else:                                  return cpu_id
    return -1
```

### 3. Budget watcher trigger (~50 行)
- 复用现有 phase1 budget_watcher
- 加 callback: budget 变化 ≥ N MB 时调 repartition()

### 4. 跟 dispatch 配合 (~100 行)
- op_schedule_fn 在 graph build 决定 backend (替 v3-v5 的 runtime dispatch)
- v3 input migration 不需要 (weight 跟 op 已在同 backend)
- 跨 backend activation 流由 ggml-sched 自动处理 (split 间 tensor copy)

### 5. LP / heuristic 决策接入 (~100 行)
- 复用 project_lp_oracle_findings 的 Belady+PF16 heuristic
- 输入: 当前 budget, weights list (size + 访问模式)
- 输出: per-weight on_gpu 决策

## 预期效果

- **稳态** (budget 不变): 0 runtime overhead, 跟正常 op_schedule 一样
- **budget 变化**: re-partition 一次性付 N MB / B 的搬运代价 (e.g., 500 MB / 3 GB/s = 167 ms)
- **vs 现 phase1 elastic 流式**:
  - 优势: dispatch 无冲突, 稳定可 deploy
  - 劣势: budget 抖动时多次 re-partition 比 elastic 慢; 但实际 mobile budget 抖动频率秒级而非毫秒级, 可接受

## 何时该投资 v8

- ✅ 长时间运行场景 (chat app), budget 偶尔变化
- ✅ LP/heuristic 实际 deploy (v8 跟 LP 设计天然吻合)
- ❌ budget 高频抖动 (phase1 elastic 流式还是更适合)
- ❌ 模型刚好装得下 GPU (用 smart-pressure 就够)

## 跟现有 commits 关系

v8 复用:
- v0 `llama_set_op_schedule` API (graph build 时决策)
- v6 `llama_weight_get_host_ptr` API (partition 知道 mmap 位置)
- ggml-sched 现有 split + tensor_copy 机制 (cross-backend activation)

v8 不需要:
- v3-v5 runtime dispatch + migration (因为决策在 graph build 时做完)

所以 v8 是 SIMPLER 的实现路径, 不需要继续叠 v7 复杂度.

## 工作量

~500 行新代码 + ~200 行测试, 2-3 天.

更详细的 LP 接入还可参考 `project_lp_oracle_findings.md` 的 Belady+PF16 heuristic.
