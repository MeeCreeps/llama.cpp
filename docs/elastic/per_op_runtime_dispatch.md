# Per-Op Runtime Backend Dispatch (Feature B)

分支: `feature/elastic-op-sched-runtime`

## 用途

每个 op 即将 compute 之前调用 user hook, hook 根据当前 runtime 状态返回 target
backend_id, 覆盖 ggml-sched split 时预分配的 backend。

跟现有 `llama_set_op_schedule` 的区别:

| API | 何时调 | 决策粒度 | 决策依据 |
|---|---|---|---|
| `llama_set_op_schedule` | graph build (1 次) | 整 graph | 编译时已知信息 (op type, layer, name) |
| `llama_set_op_runtime_dispatch` | **每个 op compute 前** | per op | 含 runtime 信息 (上一 op 时间, queue 深度, mem) |

注册后自动 `graph_reuse_disable=1` (每 decode 重 build graph)。

## API

### Public (include/llama.h)

```c
typedef int (*llama_op_runtime_dispatch_fn)(
    const struct ggml_tensor * op,
    int                        default_backend_id,
    int                        n_backends,
    void *                     user_data);
// 返回 -1 → 用 split 默认; 0..n_backends-1 → 强制 target.

LLAMA_API void llama_set_op_runtime_dispatch(
    struct llama_context * ctx,
    llama_op_runtime_dispatch_fn fn,
    void *               user_data);
```

### Lower-level (ggml/include/ggml-backend.h)

```c
typedef int (*ggml_backend_sched_runtime_dispatch_fn)(
    const struct ggml_tensor *op, int default_backend_id,
    int n_backends, void *user_data);

GGML_API void ggml_backend_sched_set_runtime_dispatch(
    ggml_backend_sched_t sched,
    ggml_backend_sched_runtime_dispatch_fn fn,
    void *user_data);
```

## 实现细节

`ggml-backend.cpp::ggml_backend_sched_compute_splits` 加新分支:

1. 当 `callback_runtime_dispatch` 设置时, 走 per-op 路径
2. **预扫描** (一次): 每个 op 调 hook 1 次, 检查 `ggml_backend_supports_op`,
   cache 决策到 `op_backend[]` 数组
3. **Grouping**: 贪心扫描连续 same-target ops, 合成 sub-graph 一次
   `ggml_backend_graph_compute_async`
4. **Cross-backend sync**: 仅 target ≠ split_backend 时 `ggml_backend_synchronize`

跟 `callback_eval` 路径并存, 互斥 (runtime_dispatch 优先)。

## Demo (tools/main)

`LLAMA_TEST_OP_RUNTIME_DISPATCH=1`:

```c
// Policy: 每偶数个 mul_mat 强制路由到 CPU
llama_set_op_runtime_dispatch(ctx, [](const ggml_tensor *op,
                                       int default_backend_id, int n_backends,
                                       void *ud) -> int {
    auto *s = (dispatch_state *)ud;
    if (op && op->op == GGML_OP_MUL_MAT) {
        s->n_mulmat++;
        return (s->n_mulmat % 2 == 0) ? s->cpu_id : -1;
    }
    return -1;
}, &ds);
```

实测 3B F16 cpu-elastic build, n=8 token decode:
- baseline (无 hook): 283 ms/tok
- per-op dispatch (本 feature): 398 ms/tok (~40% 慢)
- override: 1012 mul_mat 强制路由到 CPU (vs 1012 默认), 输出正确

## 性能特点

- **Per-op iteration overhead**: ~30-50% 慢 vs 整 split 一次 compute
- **Cross-backend sync**: 每段 target 切换 1 次 sync. cpu-elastic ↔ cpu 共享
  host memory, sync 几乎免费; 真 GPU↔CPU 会更慢
- **Grouping**: 连续 same-target ops 合成一段, 减启动开销
- **Hook 调用**: 每个 op 1 次 (cached)

## 限制 (v1)

1. **跨内存空间 backend (GPU↔CPU) 强切尚未自动 migration input**: 假设 inputs
   已在原 split_backend 内存里 (ggml-sched split 时已 copy), target backend
   能否直接读取依赖 backend memory model. cpu-elastic ↔ cpu 共享 host memory 可
   直接读; 真 CUDA↔CPU 暂无 input 自动 copy, 需补 `tensor_copy + tensor_copy_async`
   pre-op 逻辑.
2. **每 decode 重 build graph**: 强制 graph_reuse_disable=1, 每 token 增加 ~5-20ms
   build overhead. 真生产部署不建议常开.
3. **不影响 weight 驻留**: 改 op backend 不会自动搬 weight. 跟
   `llama_weight_request_prefetch/evict` 配合用.

## 应用模式

### 实验 routing 策略
试验"哪些 op 放 CPU 哪些放 GPU 最好", 不用重 build / 改 graph_get_cb 逻辑。

### 接 cost-model / runtime profiler
hook 内查 backend queue depth / mem available, 动态平衡负载。

### 元 LP / RL 调度
hook 调外部决策模块 (轻量 NN / 表查找), 每 op 输出 backend。

## 文件改动

- `ggml/include/ggml-backend.h` — 新 typedef + setter
- `ggml/src/ggml-backend.cpp` — sched struct 加字段 + compute_splits 新分支 + setter 实现
- `include/llama.h` — public typedef + setter
- `src/llama-context.{h,cpp}` — 字段 + trampoline + setter
- `tools/main/main.cpp` — `LLAMA_TEST_OP_RUNTIME_DISPATCH=1` demo

## Commits

- `e387208f9` feat(ggml-sched): true per-op runtime backend dispatch hook
- `a0df7a00e` perf(ggml-sched): group consecutive same-target ops
- `e952bb702` perf(ggml-sched): cache runtime dispatch hook result
