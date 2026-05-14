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
| H2 | `ggml/src/ggml-opencl/ggml-opencl.cpp` | `buffer_context` + `init_tensor` (≈L3279-3320) | **ε 方案 per-tensor**：每个 weight tensor 独立 cl_mem。`buffer_context` 改成 `vector<cl_mem>`（按 init 顺序，一一对应 tensor）。alloc_buffer 推迟实际分配；init_tensor 调用 `clCreateBuffer(ggml_nbytes(tensor))` 给 tensor 自己的 cl_mem，`extra->data_device = own_cl_mem`，`extra->offset = 0`。GGML_OPENCL_ELASTIC env var 守护，关闭走老 monolithic 路径。 | TODO |
| H2.5 | `src/llama-model.cpp` + `ggml/src/ggml-rpc/ggml-rpc.cpp` | 调 `get_base` 的位置 | ε 后没有单一 base 概念。get_base 在 elastic 模式下返回第一个 cl_mem (vector[0]) + 注释说明语义变了；mlock 改成对每个 tensor cl_mem 独立 init（或在 elastic 模式下跳过 mlock，因为我们正要换出去）；ggml-rpc 不支持 elastic 模式（assert）。 | TODO |
| H4 | `ggml/src/ggml-opencl/ggml-opencl.cpp` | `ggml_backend_opencl_graph_compute` | **ε 方案 per-tensor**：遍历 cgraph 节点，每个 src tensor 是 WBM 注册过的 weight → `wbmcl_ensure_resident(tensor_id)` + `wbm_touch`；dispatch op 拿到 event 写回 `last_use_event`。节流：每 EVICT_CHECK_INTERVAL 个 op 调一次 `wbm_evict_to_byte_budget` + `wbmcl_evict_batch`，**字节预算 + 批量 flush** —— B(t) 跌时一次性挑 N 个 LRU victim flush 出去，比 N 次单独 evict 省 N-1 次同步。 | TODO |
| H4.5 | `runtime/weight_buffer_manager.{h,cpp}` + `_opencl.{h,cpp}` | WBM API 扩展 | 新增 `wbm_evict_to_byte_budget(budget_bytes, exclude_idx)` 按字节贪心 LRU；新增 `wbmcl_evict_batch(victims[])` 一次 `clWaitForEvents` + 批量 release；新增 `block_meta::is_pinned` 标志让小 tensor (RMSNorm 等) 永驻、跳过 LRU。 | TODO |
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

### H2：buffer_context 改 per-tensor cl_mem (ε 方案，已选定)

**位置**：`struct ggml_backend_opencl_buffer_context` (L3111) + 
`ggml_backend_opencl_buffer_type_alloc_buffer` (L3939) + 
`ggml_backend_opencl_buffer_init_tensor` (L3279)。

**核心改动**：保留 buffer_type 对外接口不变（kernel dispatch 全部不动），
重构 `buffer_context` 的内部存储成 per-tensor cl_mem：

```cpp
// 改造后（elastic 模式下）
struct ggml_backend_opencl_buffer_context {
    // 旧：std::vector<cl_mem> buffer;   // 一个大 cl_mem
    // 新：
    bool elastic;                                  // env GGML_OPENCL_ELASTIC=1 触发
    std::vector<cl_mem> per_tensor_buffer;         // 每个 tensor 一个 cl_mem (elastic 时)
    std::vector<cl_mem> buffer;                    // 兼容老路径 (非 elastic 时)
    cl_context cl_ctx;
    size_t alignment;
    // 推迟分配：alloc_buffer 时只记 size_total，init_tensor 时按 tensor 拆
    bool deferred_alloc_done;
    size_t pending_total_size;
};
```

**关键流程**（elastic 模式）：

1. **alloc_buffer(size)**：不立刻 `clCreateBuffer`，只把 size 记到
   `pending_total_size` 当 sanity check。
2. **init_tensor(buffer, tensor)**：直接 `clCreateBuffer(ggml_nbytes(tensor))`
   给 tensor 自己的 cl_mem；`extra->data_device = own_cl_mem`，
   `extra->offset = 0`（tensor 独占整个 buffer）。
   - 完全 stateless，每个 tensor 独立处理，**不需要扫 tensor list、不需要
     name parsing、不需要 layer 划分**。
   - WBM 注册：如果 tensor name 匹配 `blk.<N>.*` 或 `*.weight` 模式 →
     调 `wbm_register_block(tensor_id, host_ptr_from_mmap, byte_size)`。
   - 小权重 (RMSNorm 等 < pin_threshold) 标记 `is_pinned = true`，永驻 GPU。
3. **buffer_context 析构**：释放 `per_tensor_buffer` 里所有 cl_mem（已经被
   WBM 释放掉的会是 nullptr，跳过）。

**view tensor 处理**：[L3284-3306](ggml/src/ggml-opencl/ggml-opencl.cpp#L3284) 
现状是 view 复用 parent extra。ε 下：view 直接挂到 parent 的 cl_mem 上
（parent 的 own_cl_mem），跨"层"概念无意义，**γ 方案下要做的 layer 守护
不再需要**。

**改动量**：
- buffer_context 改造：~50 行
- init_tensor per-tensor 分配：~40 行
- env var GGML_OPENCL_ELASTIC 守护 + 兼容老路径：~30 行
- 总：~120 行（比 γ 省 ~90 行）

**能不能干净退出**：能。`GGML_OPENCL_ELASTIC=0`（默认）走老 monolithic
路径，elastic 关闭时 buffer_context 行为零变化。

**注意 clCreateBuffer 调用次数**：Llama-3B 32 layer × ~8 tensor/layer = ~256
tensors。每个 forward 最坏情况 ~256 次 create + 256 次 release。Adreno
上每次估计 0.1-1ms = 50-500ms overhead。缓解：
1. `is_pinned` 让 RMSNorm 等小 tensor 永驻
2. LRU 自然倾向保留最近用过的小 tensor
3. probe_cl_release `--count-mode` 真机量 clCreate/Release 实际延迟

### H3：set_tensor 时触发 ensure_resident

**位置**：`ggml_backend_opencl_buffer_set_tensor` (L3342)。

**逻辑**：根据 tensor->name 反查 block_idx，调 `wbm_ensure_resident(idx)`。
然后让原有的 clEnqueueWriteBuffer 路径继续 —— 但 dst cl_mem 现在来自 WBM
而不是 vector。

**风险**：set_tensor 也会被 KV / 激活路径调到，需要白名单只对"权重 tensor"
插钩，否则会破坏 KV 上传。判定方式：检查 buffer_type 是不是 weight_lazy。

### H4：ggml-opencl 后端的 graph_compute 入口按 tensor 名分组（B1' 方案）

**位置**：`ggml/src/ggml-opencl/ggml-opencl.cpp` 里注册到
`ggml_backend_i::graph_compute` 的回调（具体函数名待 patch 时锁定 —— 这层
入口接收 `ggml_cgraph *`，遍历 nodes 并 enqueue OpenCL kernel）。

**关键洞察**：llama.cpp 给每个权重 tensor 起的名字是稳定约定 `blk.<N>.*`
（`blk.7.attn_q.weight`、`blk.7.ffn_up.weight` ...）。op 节点的 `src[]` 里
任一权重 tensor 的名字前缀就能反查 layer 序号。**不需要给 ggml 节点新加
"layer 属性"字段，也不需要 fork ggml 核心调度器。**

**改造逻辑**（伪代码）：
```cpp
int last_layer = -1;
for (int i = 0; i < graph->n_nodes; ++i) {
    ggml_tensor *node = graph->nodes[i];
    int layer = layer_id_from_node_srcs(node);   // "blk.7.*" → 7；非 layer 节点 → -1

    if (layer >= 0 && layer != last_layer) {
        wbm_ensure_resident(wbm, layer);

        // LRU 释放
        size_t B_t = budget_watcher_get(bw);
        int max_resident = wbm_max_resident_blocks(wbm, B_t, kv_bytes, misc);
        while (wbm_resident_count(wbm) > max_resident) {
            int v = wbm_pick_lru_victim(wbm, /*exclude=*/layer);
            if (v < 0) break;
            // Event-sync：wait 受害者上最近一次 kernel event，再 release
            wbm_evict(wbm, v);
        }

        // headroom 够就 prefetch 下一段
        if (max_resident > wbm_resident_count(wbm) && layer + 1 < n_layer) {
            wbm_prefetch(wbm, layer + 1);
        }
        last_layer = layer;
    }

    cl_event ev = dispatch_op(node);   // 原有 enqueue 路径，但要拿 event
    // 给 node->src[i] 里的每个权重 cl_mem 记一笔 last_use_event = ev
    for (auto *s : node->src) {
        int l = layer_id_from_tensor(s);
        if (l >= 0) wbm_mark_in_use(wbm, l, ev, current_token);
    }
}
```

**前提与坑**：

1. **节点拓扑序天然按 layer 顺序**：llama.cpp 的 `llm_build_llama` 是
   `for (il = 0..n_layer-1)` 一层一层往 cgraph 塞节点，topo 排序后同 layer
   节点保持连续且 layer 0 在前、layer N-1 在后。这是稳定的实现事实但不是
   ggml 抽象保证。**首次 patch 时要 assert 验证**：扫一遍节点序列，确认
   "layer 序号单调不减"，不满足直接 abort 给清晰错误。

2. **OpenCL 异步性 & event 同步**：`clEnqueueNDRangeKernel` 返回时 kernel
   还在 queue 里没真跑。如果 layer 7 的 kernel 还没 run 完就
   `clReleaseMemObject` 它的 Wq，行为 UB。**用 event 跟踪**：每个 op
   dispatch 拿一个 `cl_event`，记到它 src 里所有 layer 权重的
   `last_use_event` 字段；evict 前 `clWaitForEvents(1, &last_use_event)` 只
   等"用到这个被驱逐 buffer 的"最后一个 event。比每个 layer 边界
   `clFinish` 精确，不丢 dispatch/compute 流水。

3. **非 layer tensor**：`token_embd.weight`、`output_norm.weight`、
   `output.weight` 不在 `blk.<N>.*` 命名下。这些**永远常驻**，alloc 走原
   路径，不归 WBM 管。layer_id_from_tensor 对它们返回 -1。

4. **KV / 激活也走 src[]**：判定权重 tensor 用前缀 `blk.<N>.` + 后缀 `.weight`
   双重过滤，避免误把 KV 或临时 tensor 当 layer 权重。

**改动量**：~250 行；位置集中在 ggml-opencl 后端一个文件里，**一处覆盖所
有架构**（llama / qwen / glm 都共享后端）。

**能不能干净退出**：能。整个 layer-aware 逻辑用 `#ifdef GGML_OPENCL_ELASTIC`
或 runtime env `LLAMA_ELASTIC=1` 守护，关掉就回原 enqueue 路径。

### H5：elastic-cli 主入口

**位置**：新文件 `examples/elastic-cli/elastic-cli.cpp`。

参考 `examples/main/main.cpp` / `tools/llama-cli/` 的解析模式，扩展两个 flag：
- `--budget-csv <path>`：BudgetWatcher 输入
- `--metrics-jsonl <path>`：MetricsLogger 输出

每个 decode step 后写一条 metrics。

---

## 4. Open Questions

**Q1：H2 怎么区分 weight buffer_type 和 activation/KV buffer_type？** ✅ 已决议

**结论：γ 方案下不区分 buffer_type，按 tensor name 内部分流。**

γ 改造把 per-layer 拆分塞进 `buffer_context` 内部，对外接口不变。无论
权重 / 激活 / KV 都进同一个 buffer_type；init_tensor 时按 name 解 layer：
- `blk.<N>.*.weight` → layer N 的 cl_mem，参与 LRU
- `token_embd.weight` / `output.weight` / `output_norm.weight` → shared 桶，常驻
- 激活 / KV（无 `blk.` 或 `.weight` 后缀）→ 也归 shared 桶，常驻

KV 已是常驻的（规范 §6 + §8 明确 out of scope），自然不参与 LRU。

**Q2：lazy buffer 怎么和 ggml_backend_graph_compute 协作？** ✅ 已决议

**结论：B1' —— 在 ggml-opencl 后端的 `graph_compute` 回调里按 tensor 名分组。**

核心洞察：llama.cpp 给权重 tensor 起的名字是稳定约定 `blk.<N>.*`，op 节点
的 `src[]` 里任一权重 tensor 名前缀就能反查 layer 序号，不需要给 ggml 核
心新加 `elastic_layer_id` 字段。详见 §3 H4 改写后的伪代码。

排除的备选：
- 节点级 fork ggml 核心：要改 `struct ggml_tensor` + 通用调度器 + 每个
  arch 的 `llm_build_*`。工程量大、跟上游 rebase 难，且会污染 CUDA/Metal
  等不需要 elastic 的 backend。
- forward 粒度（"装不下整个 forward 就拆 sub-graph"）：流式表达力弱，
  装不下时的 sub-graph 切分本身比 B1' 还麻烦。

OpenCL 异步同步：用 `cl_event` 跟踪每个 op，evict 前 `clWaitForEvents` 只
等用到被驱逐 buffer 的最后一个 event；不在 layer 边界 `clFinish` 整个
queue，保留 dispatch/compute 流水。

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

> Phase 2 设计基线：B1' + event 跟踪。逐 layer 粒度，贴合规范 §2；
> patch 集中在 ggml-opencl 后端一处。
