# Elastic + Per-Op Dispatch 整合 (v6 / v7 状态)

## 用户诉求

> "dynamic memory budget, weight 不全在 RAM. budget 变化时考虑怎么放, 怎么切"

需要把 3 件事串起来:
1. **动态 budget** (已有 phase1 `GGML_ELASTIC_BUDGET_CSV` budget watcher)
2. **WBM eviction 决策** (已有 phase1 MRU/Belady, LP 实验给出 heuristic Belady+PF16)
3. **per-op dispatch** (v3-v5 已实现)

整合: budget 收紧 → elastic evict weight X → ops 使用 X 时路由到 CPU (从 mmap 读, 避免 reload IO)

## v6 已实现的基础设施

### API (include/llama.h)
```c
void * llama_weight_get_host_ptr(struct llama_context * ctx, const char * name);
```
返回 weight 在 mmap 区的 host pointer (elastic 的 WBM 提供), 或 NULL.

### Provider (跨 TU registry, llama-mmap.h)
```c
typedef void * (*llama_weight_host_ptr_fn_t)(const char * name, void * ud);
void llama_weight_host_ptr_register(fn, ud);
void * llama_weight_host_ptr_query(name);
```
ggml-opencl-elastic 跟 ggml-cpu-elastic 启动时注册 provider, 返回 `wbm.blocks[idx].host_ptr`.

### Migration fallback (ggml-backend.cpp)
v3 input migration 之前: 调 `llama_weight_host_ptr_query(src->name)`
- 拿到 host_ptr → `memcpy(temp_buf, host_ptr, nbytes)` (绕过 cl_mem)
- 拿不到 → fall back to `ggml_backend_tensor_copy` (走 clEnqueueRead)

### Demo policy `elastic-aware`
```c
if op is weight-mulmat (Q/K/V/attn_out/ffn_*) {
    if !llama_weight_is_resident(weight) {
        route to CPU;  // 用 host_ptr fallback 从 mmap 拿 weight
    }
}
```

## v6 实测结果

✅ **non-elastic mode** (smart-pressure / alternate 等): 全 v3-v5 policies 正常
⚠️ **elastic mode + migration**: crash at `clEnqueueReadBuffer error -38 (CL_INVALID_MEM_OBJECT)`

## 为什么 v6 crash?

elastic backend 在 evict 时 `clReleaseMemObject` 释放 cl_mem. 但 tensor->extra->data_device
仍指向已释放的 cl_mem.

我们的 migration 用 host_ptr fallback **正确处理了 weight tensor** (host_ptr 可用).
但 activations / non-weight tensors 也走 migration 路径:
- `inputs_compatible(activation, cpu_backend)` = false (cl_mem 不 host)
- migration 尝试拿 host_ptr → null (activation 不是 weight, 不在 WBM)
- fall back to `ggml_backend_tensor_copy` → opencl get_tensor → `clEnqueueReadBuffer` on `extra->data_device`
- 如果该 activation 之前在某个 op 用过, 而 elastic 因 budget 重排释放了它的 cl_mem → 崩

更深层: elastic backend 的 reload/evict 时机假设 op 在它自己 backend 跑. 我们 dispatch
把 op 路由走后, elastic 不知道, 仍可能 evict 后续 op 需要的 weight 而不重新 reload.

## v7 真要 deploy 还需 (~200-300 行 elastic backend 改动)

1. **elastic backend dispatch-awareness**: ensure_phase 检查 op 是否被 dispatch
   走 (查 sched 的 op_backend[] 状态), 跳过那些 op 的 weight reload
2. **cl_mem 生命周期保护**: 被 dispatch 路由的 op 的 src activations 不能 evict
   (引用计数)
3. **dispatch 决策 + elastic eviction joint LP**: 把 LP 实验 (project_lp_oracle_findings)
   的 weight placement 决策跟我们 dispatch 决策一起算

## 推荐 deploy 路径

### 立即可用 (无 elastic)
```bash
LLAMA_TEST_OP_RUNTIME_DISPATCH=smart-pressure \
GGML_SCHED_RUNTIME_DISPATCH_MIGRATE=1 \
./llama-cli -m model.gguf -ngl 99 ...
```
- 99% 时间 0 overhead (跟 baseline 一致)
- mem 紧时温和切几层 ffn 给 GPU 喘气
- 适合 model 装得下 GPU 内存的场景

### 现有 elastic (无 dispatch, 已 phase1)
```bash
GGML_OPENCL_ELASTIC=1 GGML_ELASTIC_BUDGET_CSV=trace.csv \
GGML_ELASTIC_EVICT_POLICY=mru GGML_ELASTIC_PREFETCH=32 \
GGML_ELASTIC_PIN=norm,k,v,q GGML_ELASTIC_EMBED_OUTSIDE_BUDGET=1 \
./llama-cli -m model.gguf -ngl 99 ...
```
- model > GPU 可用内存的场景
- weights 流式 reload (test3_tight 1096 ms/tok)
- 已 deploy 验证

### v7 deploy (elastic + dispatch, 留 backlog)
等 elastic backend 改造完成后:
```bash
GGML_OPENCL_ELASTIC=1 GGML_ELASTIC_BUDGET_CSV=trace.csv \
GGML_SCHED_RUNTIME_DISPATCH_MIGRATE=1 \
LLAMA_TEST_OP_RUNTIME_DISPATCH=elastic-aware \
./llama-cli -m model.gguf -ngl 99 ...
```

## Commits (新加 v6)

- `3787b9c47` feat(elastic): v6 host_ptr API + elastic-aware dispatch policy (PoC)
