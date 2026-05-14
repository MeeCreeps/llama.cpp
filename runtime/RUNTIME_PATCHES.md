# RUNTIME_PATCHES.md

记录 Phase 2 为接入 WeightBufferManager（WBM）对 llama.cpp 主干 + ggml-opencl
后端打的所有钩子 / 补丁。**每改一处源码同步更新本文件**：写明位置、为什么、
能不能干净退出。合并回上游或换 llama.cpp 版本时这是最重要的参考。

> 行号锚定 git 提交：`f117be185`（master HEAD on 2026-05-14）。
> 升级 llama.cpp 后行号会漂移，但函数名和概念位置应保持稳定。

---

## 1. 钩点全景

| # | 文件 | 函数 / 位置 | 干啥 | 状态 |
|---|---|---|---|---|
| H1 | `src/llama-model.cpp` | `llama_model::load_tensors` (≈L2203) | 不在这里搬权重；记录每个 layer 的 tensor name → (host_ptr, byte_size) 表给 WBM。**不调** ggml-opencl 的 alloc。 | TODO |
| H2 | `ggml/src/ggml-opencl/ggml-opencl.cpp` | `ggml_backend_opencl_buffer_type_alloc_buffer` (≈L3939) | 对**权重 buffer 类型**改成 lazy：不立刻 `clCreateBuffer`，返回一个空壳 `cl_mem` slot，等 WBM 真要用时再建。激活 buffer / KV / 中间 tensor 走原路径。 | TODO |
| H3 | `ggml/src/ggml-opencl/ggml-opencl.cpp` | `ggml_backend_opencl_buffer_set_tensor` (≈L3342) | 上传时调 `wbm_ensure_resident(block_idx_from_tensor)`。 | TODO |
| H4 | `src/models/<arch>.cpp` 各 `llm_build_*` | 逐 layer 的 graph 构造（如 `llm_build_glm4::llm_build_glm4`） | 在引用 `model.layers[il].*` 之前调 `wbm_ensure_resident(il)`；完事 LRU + `wbm_evict` 释放过期 layer。 | TODO |
| H5 | `examples/elastic-cli/elastic-cli.cpp` | 主 decode 循环 | 拉 BudgetWatcher、按 step 写 MetricsLogger、每 step 校验 `resident_bytes ≤ B(t)`、违反 abort。 | TODO |

H1 仅做"建索引"。真正的物质性补丁是 H2 + H4（前者把权重 cl_mem 改成
lazy，后者保证 graph build 时该 layer 已驻留）。H3 是过渡补丁，可能在
设计稳定后被 H4 取代。

---

## 2. 关键基础设施位置（仅引用，不改）

### 2.1 GGUF tensor offset → host 指针

`src/llama-model-loader.h:28-44` 的 `llama_tensor_weight` 已经算好：

```cpp
const int tensor_idx = gguf_find_tensor(gguf_ctx, ggml_get_name(tensor));
offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
```

mmap 路径在 `src/llama-model-loader.cpp:899` 的 `load_data_for`：
```cpp
cur->data = (uint8_t *)mapping->addr() + w.offs;
```

WBM 直接复用 `mapping->addr() + w.offs` 作为 host 端 byte 起点。

### 2.2 ggml-opencl 后端结构

`ggml/src/ggml-opencl/ggml-opencl.cpp` 关键接口：

- `ggml_backend_opencl_buffer_type_interface`（≈L3977）：buffer **type** 的 vtable，关键方法 `alloc_buffer = ggml_backend_opencl_buffer_type_alloc_buffer` (L3939)
- `ggml_backend_opencl_buffer_interface`（L3917）：单个 buffer 实例的 vtable，包含 `set_tensor`、`get_tensor`、`free_buffer`
- `ggml_backend_opencl_buffer_context`（L3111）：内部存 `std::vector<cl_mem> buffer`

H2 在 alloc 阶段拿到 `size`，但拿不到 tensor name；需要在 H1 阶段建好 size→tensor 映射，或者在 `set_tensor` 时回填 block 归属。这是一个待定的设计点（见 §4）。

### 2.3 Decode 逐层迭代位置

`src/llama-model.cpp:2274` 的 `for (int il = 0; il < n_layer; ++il)` 是 **load_tensors 的**设备分配循环，不是 forward 迭代。

forward 的逐 layer 在每个架构的 `llm_build_*` 类里（例如
`src/models/glm4.cpp:23`），通常长这样：
```cpp
for (int il = 0; il < n_layer; ++il) {
    auto * Wq = model.layers[il].wq;
    // ... attention + ffn graph nodes ...
}
```

H4 就插在这个循环的最前。

---

## 3. 钩点详细设计

### H1：`load_tensors` 建 block 索引

**位置**：`src/llama-model.cpp` 的 `llama_model::load_tensors` 末尾，所有
tensor 已经从 GGUF 解出后。

**改动量**：约 30 行新代码 + 调一个 WBM 新 API。

**逻辑**：
```cpp
// 伪代码
WeightBufferManager *wbm = get_or_create_wbm(this);
for (int il = 0; il < n_layer; ++il) {
    block_handle &h = wbm->blocks[il];
    h.host_ptr = mapping->addr() + first_tensor_offset_of_block(il);
    h.byte_size = sum_of_tensor_bytes_of_block(il);
    h.dev_buf = nullptr;
    h.resident = false;
}
```

"哪些 tensor 属于 block i" 用 name prefix `blk.<i>.*` 判定（GGUF 命名约定）。

**能不能干净退出**：能。WBM 没 init 时这段代码全是 no-op。

### H2：buffer_type_alloc_buffer 改 lazy

**位置**：`ggml_backend_opencl_buffer_type_alloc_buffer` (L3939)。

**问题**：这函数现在长这样（要求复核源码）—— 拿 size，立刻
`clCreateBuffer(ctx, ..., size, ...)`，返回 `ggml_backend_buffer_t` 包了
`std::vector<cl_mem> buffer`。

**计划**：加一种 buffer_type 变体 `weight_lazy`：alloc 时不调 `clCreateBuffer`，
只记 size，等 set_tensor 时由 WBM 决定真分配。激活 / KV / 临时 tensor 走原路径。

**改动量**：新增 ~80 行（新 buffer_type 实例 + 新 vtable）+ 在 backend 选择
buffer_type 处分流（weight 用 lazy，其它用原）。**这里需要查上游 ggml 怎么
区分"这是 weight 还是 activation"** —— 可能要看 `ggml_backend_alloc_ctx_tensors_from_buft`
或 `model.cpu_buft_list` 之类，§4 列为 open question。

**能不能干净退出**：基本能。把新 buffer_type 注释掉、走回原 alloc_buffer 即恢复上游行为。

### H3：set_tensor 时触发 ensure_resident

**位置**：`ggml_backend_opencl_buffer_set_tensor` (L3342)。

**逻辑**：根据 tensor->name 反查 block_idx，调 `wbm_ensure_resident(idx)`。
然后让原有的 clEnqueueWriteBuffer 路径继续 —— 但 dst cl_mem 现在来自 WBM
而不是 vector。

**风险**：set_tensor 也会被 KV / 激活路径调到，需要白名单只对"权重 tensor"
插钩，否则会破坏 KV 上传。判定方式：检查 buffer_type 是不是 weight_lazy。

### H4：架构 graph build 加 ensure/evict

**位置**：`src/models/<arch>.cpp` 的 `llm_build_*` 类构造里逐层循环开头。
每个支持的 arch（llama、glm4、qwen 等）都要改一遍 —— **首版只覆盖
Llama-3.2 系列**，其它架构作为后续 follow-up。

**逻辑**：
```cpp
for (int il = 0; il < n_layer; ++il) {
    wbm_ensure_resident(wbm, il);
    // ... 原有 attention + ffn graph 节点 ...

    size_t B_t = budget_watcher_get(bw);
    int max_resident = wbm_max_resident_blocks(wbm, B_t, kv_bytes, misc);
    while (wbm_resident_count(wbm) > max_resident) {
        int v = wbm_pick_lru_victim(wbm, /*exclude=*/il);
        if (v < 0) break;
        wbm_evict(wbm, v);
    }
}
```

**问题**：llama.cpp 的 graph 是先**构建**整个 ggml_cgraph，再一次性
**执行**，不是逐 layer 同步执行。`wbm_ensure_resident` 放在 graph 构建期意
义不大 —— 真正的 clEnqueueNDRangeKernel 在 `ggml_backend_graph_compute` 里
触发。**这是 Phase 2 设计的核心难点**，详见 §4 open Q3。

### H5：elastic-cli 主入口

**位置**：新文件 `examples/elastic-cli/elastic-cli.cpp`。

参考 `examples/main/main.cpp` / `tools/llama-cli/` 的解析模式，扩展两个 flag：
- `--budget-csv <path>`：BudgetWatcher 输入
- `--metrics-jsonl <path>`：MetricsLogger 输出

每个 decode step 后写一条 metrics。

---

## 4. Open Questions

**Q1：H2 怎么区分 weight buffer_type 和 activation/KV buffer_type？**

ggml 的 buffer_type 概念是按"内存域"分的（CPU / OpenCL / CUDA），不是按
"权重 vs 激活"。要做 lazy weight，需要在 backend init 时新增一个独立 type
（如 `ggml-opencl-weight`），然后让 llama_model_loader 在分配权重时挑这个
type、激活仍走原 type。**待定**：要不要 patch llama.cpp 的 buft 选择，还
是用环境变量绕开？倾向前者，但要先读懂上游 buft list 构造路径。

**Q2：lazy buffer 怎么和 ggml_backend_graph_compute 协作？**

执行时 ggml runtime 不知道 cl_mem 是 NULL 还是已分配。可能的方案：
1. 在 H4 的 graph build 之后、compute 之前，扫一遍要用的 tensor，调
   wbm_ensure_resident 把这一轮所有需要的 layer 都先驻留好（无法做到逐 layer
   流式，但保证不崩）
2. 自定义 ggml backend 的 compute 路径，在每个 op 执行前注入 ensure
   （工作量大，侵入性强）

**首版选 (1)**：在 decode 入口处一次性 ensure 当前 max_resident 个 layer，
随着 token 推进调整。流式行为以"一个 forward 内不变"为粒度 —— 比规范 §2
描述的"每个 block 内 ensure → compute → evict"粗，但能跑通，性能验证后
再升级。**先和你确认这个降级是否可接受**（详见本文档结尾的 ⚠️）。

**Q3：mmap unmap_fragment 要不要用？**

`src/llama-mmap.h:45` 有 `unmap_fragment(first, last)`。规范 §10 提到：CPU
侧 mmap 镜像即使权重在 GPU 上也算 DRAM。Evict 一个 block 时，除了
`clReleaseMemObject` 还要 `madvise(MADV_DONTNEED)` 它的 mmap 范围，否则
内核 page cache 仍会占着。**待定**：首版先只动 cl_mem，看 VmRSS 是否
"够低"；不够再加 madvise。

**Q4：KV cache 在哪个 buffer_type 上？**

规范 §6 说 KV 始终留在 GPU。需要确认 ggml-opencl 后端给 KV 分配的是
哪个 buffer_type，确保 H2 的 lazy 改造**不影响 KV 路径**。代码点 TBD。

---

## 5. 测试与回退

每个钩点应该有一个独立开关，可在编译期或 runtime 关掉，让代码回到原
llama.cpp 行为。**回退基线 = 跑通 examples/main，没有 WBM 任何介入**。
回退方式：

- H1：WBM 没 init → load_tensors 内的索引建立代码会跳过
- H2：`#ifdef ELASTIC_LAZY_WEIGHT_BUFFER` 包裹新 buffer_type 注册
- H3：白名单为空时 set_tensor 行为不变
- H4：每个 arch 文件里用环境变量 `LLAMA_ELASTIC=1` 守护，默认关
- H5：elastic-cli 是独立 example，不影响 main / llama-cli

---

## 6. 改动清单（每改一处 commit 后回来打勾）

- [ ] H1：`src/llama-model.cpp`
- [ ] H2：`ggml/src/ggml-opencl/ggml-opencl.cpp`（buffer_type 拆分）
- [ ] H3：`ggml/src/ggml-opencl/ggml-opencl.cpp`（set_tensor 钩子）
- [ ] H4：`src/models/llama.cpp`（首版只 Llama 架构）
- [ ] H5：`examples/elastic-cli/`（新文件）
- [ ] Q1-Q4 设计点决议后回填本文档

---

> ⚠️ **降级提示**：首版 H4 不做"逐 block ensure/evict"而是"每个 forward 入口处一次性 ensure，forward 内不换"。流式粒度从 block 退到 forward，比规范 §2 描述粗。正式跑 Test 1 前需要你点头。
