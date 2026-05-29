# Op-level Runtime Scheduler (MVP)

llama.cpp 上游只支持 layer-level offload (`-ngl N`), 整个 layer 落到 CPU 或 GPU,
启动后不变. 本 MVP 加 **per-op 运行时动态分配**: 每个 decode 用不同 schedule.

## 使用

```bash
LLAMA_OP_SCHED=<strategy> ./llama-cli ...
```

Strategy:
- 空 (默认): 用 ggml-sched 自动决定 (跟原 llama.cpp 行为一致)
- `all-cpu`: 强制所有 MUL_MAT 到 CPU backend
- `all-gpu`: 强制所有 MUL_MAT 到第一个非 CPU backend
- `alternate`: per-decode 轮换, 偶数次 decode 到 CPU 奇数次到 GPU
- `memory`: 读 /proc/meminfo MemAvailable, ≥`LLAMA_OP_SCHED_GPU_THRESH_MB` (默认 8000) 时
            选 GPU, 否则 CPU
- `external`: 每次 decode 读 `LLAMA_OP_SCHED_BACKEND` env (= "cpu" 或 "gpu") 决定

## 实现

`src/llama-context.cpp::graph_get_cb()`. 每次 build_graph 调用时:
1. counter 递增 (per decode)
2. 重读 strategy env
3. lambda 内对每个 MUL_MAT 节点, 按 strategy 决定 backend
4. 调 `ggml_backend_sched_set_tensor_backend` 应用

## 设计要点

- **只移动 compute, 不移动 weight 存储**: weights 留在 model loader 分配的 buffer
  (CPU 或 GPU buffer). compute backend 跟 weight backend 不一致时, ggml-sched 自动
  插 copy op 把 weight/activation 复制过去
- **每个 decode 重新决定**: counter+strategy 检查在 build_graph 阶段, 而 build_graph
  per decode 跑一次, 所以 schedule 可以 token-level 动态变化
- **跟现有 elastic budget 兼容**: weight 仍按 elastic budget evict/reload; op-sched
  只决定 compute 跑哪儿. 两个机制可以协同 — 比如低 budget 时把 ops 推到 CPU
  (避免 GPU 跟 RAM 抢)

## Sanity check

3B F16, CPU-only build (build-android-cpuelastic, GPU backend 暂时因 SOA 头不
匹配跑不起来):

```
baseline:           "The story begins: The year was"
LLAMA_OP_SCHED=all-cpu:     same output
LLAMA_OP_SCHED=alternate:   same output  (CPU only, alternate 等效 all-cpu)
```

Op-sched 框架不破坏 token 输出. 没 GPU backend 时三种 strategy 等效, 但 hook 已经
工作 (counter 增, set_tensor_backend 调用; 后续接 GPU 即生效).

## 未做

- GPU backend SOA 头不匹配, 本分支没修. 要 GPU demo 需先解决 (cherry-pick 4cd1aea43
  到这分支 + 解决跟现有 ggml-opencl.cpp 的 conflict)
- 实际 budget-aware 策略 (跟 budget_watcher 集成, 用 B(t) 而非 /proc/meminfo) 未实现
- Cost-aware 策略 (考虑每 op 的预期 latency) 未实现
