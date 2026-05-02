# Mobile Multi-LoRA Serving on llama.cpp + OpenCL — Implementation Guide

> 这份文档是给 Claude Code 用来逐步实现一个**端侧多 LoRA serving 系统**的工程 spec。底座是 llama.cpp，重点后端是 OpenCL（针对 Adreno GPU on Snapdragon 手机）。

---

## 0. 目标与范围

### 0.1 What we are building

一个跑在 Android 手机上的 **multi-LoRA LLM serving runtime**：
- 一个共享的 base model（Llama-3.2-3B Q4_0）
- N 个 LoRA adapter（典型 N=10–100）
- 输入：一个预生成的 trace JSON（100–1000 条 request，含 arrival_time / prompt / adapter_id）
- 主循环：按到达时间推进，按 adapter 分组 batch
- 输出：CSV metric 文件（TTFT、TPOT、throughput、peak memory、能耗）
- 支持 adapter 热切换、batching、cache 管理

### 0.2 What we are NOT building (scope cuts)

明确不做的事情，避免 scope creep：

- ❌ Routing（adapter 由请求显式指定，不做 prompt → adapter 分类器）
- ❌ Multi-modal（vision / audio 都不做，纯文本）
- ❌ NPU backend（OpenCL only；Hexagon QNN 留 future work）
- ❌ Heterogeneous rank batching（第一版假设池内所有 adapter 同 rank=16）
- ❌ Compositional / multi-LoRA 同时激活
- ❌ 加密 / TEE / 多租户隔离
- ❌ 分布式 / 跨设备

### 0.3 Success criteria（最低）

- 在 Snapdragon 8 Gen 3+ 手机上跑通 Llama-3.2-3B Q4_0 + ≥10 个 r=16 LoRA adapter
- Open-loop driver 能持续 10 分钟，不 crash、不 OOM
- TTFT、TPOT、能耗的 measurement pipeline 可重现
- 至少在两个 metric 上击败 vanilla llama.cpp（多次 swap 场景）

### 0.4 Stretch goals（如果时间允许）

- LRU + size-aware adapter cache
- Group LoRA inference kernel（OpenCL custom）
- Cross-adapter KV reuse（参考 MobiLoRA）
- 异构 rank 支持

---

## 1. 设计概览

### 1.1 架构图

> **架构原则**：单二进制、单线程、纯 in-memory simulation。**没有 HTTP / 没有 driver thread / 没有 future**。
> Trace 是一个内存 list of `Request`，主循环按到达时间推进 + 按 adapter 分组 batch，结束 dump CSV。
> 这正是 vLLM `benchmark_throughput.py` 的研究风格——一个 main loop 干完所有事。

```
┌──────────────────────────────────────────────────────────────┐
│  Single binary: multi-lora-bench (跑在 Android 上)            │
│  单线程 main loop（也可以多线程，但默认单线程更可控）          │
│                                                                │
│  启动时：                                                      │
│   1. 加载 base model                                          │
│   2. 解析 workload trace → std::vector<Request>（100 条）     │
│   3. 初始化 AdapterPool + Metrics                             │
│                                                                │
│  Main loop 反复做这 4 件事直到 workload 跑完：                 │
│                                                                │
│   ┌────────────────────────────────────────────────┐          │
│   │ ① 把"已到达"的 request 加进 active slot 列表    │          │
│   │   （根据 elapsed time 与 req.arrival_time 比较）│          │
│   └─────────────────┬──────────────────────────────┘          │
│                     ▼                                          │
│   ┌────────────────────────────────────────────────┐          │
│   │ ② 在 active 中按 adapter_id 分组，取最大组      │          │
│   └─────────────────┬──────────────────────────────┘          │
│                     ▼                                          │
│   ┌────────────────────────────────────────────────┐          │
│   │ ③ AdapterPool.acquire(adapter)                 │          │
│   │   - cache hit: 直接 set                         │          │
│   │   - cache miss: load + LRU evict + set          │          │
│   └─────────────────┬──────────────────────────────┘          │
│                     ▼                                          │
│   ┌────────────────────────────────────────────────┐          │
│   │ ④ llama_decode(batch) 跑一步                    │          │
│   │   - 每个 slot 出一个 token                      │          │
│   │   - 第一个 token: 记 first_token_time           │          │
│   │   - EOS / max_output: 记 finish_time, 移出 slot │          │
│   └────────────────────────────────────────────────┘          │
│                                                                │
│  跑完：metrics.dump_csv("out.csv")                             │
└──────────────────────────────────────────────────────────────┘

执行流（adb shell 上）：
  1. adb push base + adapters + workload.json 到 /data/local/tmp/
  2. adb shell ./multi-lora-bench --workload workload.json --out /sdcard/m.csv
  3. adb pull /sdcard/m.csv → PC 上 python plot.py 出图
```

### 1.1.1 Request / Slot 数据结构

```cpp
// 输入：从 trace JSON 反序列化得到
struct Request {
  std::string id;
  double arrival_time;            // 实验开始后第几秒到达
  std::string adapter_id;
  std::vector<llama_token> prompt; // 已 tokenize
  size_t max_output;
};

// 运行时：active 状态的请求
struct Slot {
  Request req;
  llama_seq_id seq_id;            // llama.cpp 的 sequence id
  std::vector<llama_token> output;
  size_t prefill_done = 0;        // 已喂给 llama 多少 prompt token
  bool first_token_recorded = false;
  double first_token_time;
  // 记录用：metric record 在 finalize 时写
};
```

### 1.1.2 主循环骨架（伪代码）

```cpp
int main(int argc, char** argv) {
  // 启动初始化
  auto cfg = parse_args(argc, argv);
  llama_model* model = llama_load_model_from_file(cfg.model_path, ...);
  llama_context* ctx = llama_new_context_with_model(model, ...);
  AdapterPool pool(model, cfg.adapter_dir, cfg.max_adapter_mb);
  Metrics metrics;

  std::vector<Request> workload = load_trace(cfg.workload_path);
  std::sort(workload.begin(), workload.end(),
            [](auto& a, auto& b){ return a.arrival_time < b.arrival_time; });

  std::vector<Slot> active;
  size_t next_idx = 0;
  auto t0 = std::chrono::steady_clock::now();

  while (next_idx < workload.size() || !active.empty()) {
    double now = elapsed_seconds(t0);

    // ① 把已到达的请求转成 slot
    while (next_idx < workload.size()
           && workload[next_idx].arrival_time <= now) {
      active.push_back(make_slot(workload[next_idx], next_idx));
      next_idx++;
    }

    // 没有 active：sleep 到下一个到达
    if (active.empty()) {
      auto next_t = workload[next_idx].arrival_time;
      sleep_until(t0 + seconds(next_t));
      continue;
    }

    // ② 按 adapter 分组，取最大组
    auto batch = pick_largest_adapter_group(active);

    // ③ 切 adapter
    pool.acquire(ctx, batch[0]->req.adapter_id);

    // ④ 跑一步 decode（混合 prefill + decode token）
    llama_batch llb = build_llama_batch(batch);
    llama_decode(ctx, llb);

    for (auto* slot : batch) {
      if (still_in_prefill(slot)) {
        slot->prefill_done += llb.n_tokens_for_slot(slot);
        continue;
      }
      llama_token tok = sample_token(ctx, slot->seq_id);
      if (!slot->first_token_recorded) {
        slot->first_token_time = elapsed_seconds(t0);
        slot->first_token_recorded = true;
      }
      slot->output.push_back(tok);
      if (tok == llama_token_eos(model)
          || slot->output.size() >= slot->req.max_output) {
        finalize_and_record(slot, t0, metrics);
        remove_slot(active, slot);
      }
    }

    pool.release(batch[0]->req.adapter_id);
  }

  metrics.dump_csv(cfg.out_path);
  return 0;
}
```

整个程序就这么个 loop——纯 in-process，无线程，无网络，无 future。


### 1.2 核心机制选择

| 机制 | 我们的选择 | 理由 |
|------|----------|------|
| Routing | **Bypass**（请求直接带 adapter_id）| 不在主战场，避免与 AgileCore 撞 |
| LoRA execution | **Group by adapter**（同 adapter 一组 batch）| 实现简单；EdgeLoRA 同思路 |
| Adapter cache | **LRU 固定 slot pool** | 简单、可解释、对端侧足够 |
| Adapter loading | **mmap from UFS** | 利用 OS page cache，零 copy |
| Multi-adapter forward | **Slot-based** | 每个 slot 独立 KV，绑定一个 adapter |
| Continuous batching | **Same-adapter only** | 不混 adapter 减少 kernel 复杂度 |
| OpenCL kernel | **沿用 llama.cpp 现有 kernel** + 自定义 LoRA op | 最小改动 |

### 1.3 与现有工作的关系

| 论文 | 我们的关系 |
|------|----------|
| **EdgeLoRA** | 思路最接近（group LoRA + LRU pool + slot state machine）。我们 OpenCL 后端而非 GGML CPU/CUDA |
| **AgileCore** | 不竞争（无 router、无 NPU split、无 retention-density） |
| **MobiLoRA** | 不竞争第一版（不做 KV reuse） |
| **vanilla llama.cpp** | 主 baseline |
| **S-LoRA / Punica** | 思路参考，无法直接移植（CUDA） |

---

## 2. 项目脚手架

### 2.1 仓库结构

主仓库 fork 自 `https://github.com/ggerganov/llama.cpp`，分支命名 `mobile-multi-lora`。

新增目录：

```
llama.cpp/
├── examples/
│   └── multi-lora-bench/         # 新建：单二进制
│       ├── main.cpp              # entry + main loop（整个程序就一个文件）
│       ├── workload.cpp/h        # Request struct + load_trace() JSON 反序列化
│       ├── adapter_pool.cpp/h    # adapter cache + load
│       ├── slot.cpp/h            # Slot struct + 工具函数
│       ├── scheduler.cpp/h       # M2 引入：把 main loop 的 batch 逻辑抽出来
│       ├── metrics.cpp/h         # 内存 vector + dump CSV
│       └── CMakeLists.txt
├── src/
│   ├── llama-adapter.cpp         # 必要时小补丁
│   └── ...
├── ggml/src/ggml-opencl/         # 不修改（第一版）
├── tools/
│   └── workload-gen/             # PC 离线工具：生成 trace + 后处理
│       ├── gen_workload.py       # Poisson + Pareto 生成 trace JSON
│       ├── prompts/              # ShareGPT 等 prompt 池
│       └── plot_results.py       # 读 CSV 出图
└── docs/
    └── multi-lora-design.md      # 设计决策记录
```

**关键变化**：
- 没有 server / driver 单独文件——整个程序是一个 `main.cpp` 主循环
- 没有 HTTP / 网络 / 异步层
- Python 只在 PC 端做 *离线 workload 生成* 和 *后处理*；不参与 runtime

### 2.2 工具链

**开发机**（macOS / Linux）：
- Android NDK r26+ (`https://developer.android.com/ndk/downloads`)
- CMake 3.22+
- Python 3.11+ (driver)
- adb tools

**目标设备**（必须，按优先级）：
1. Snapdragon 8 Gen 3 / Elite 手机（Xiaomi 14、Redmi K70 Pro、OnePlus 12 等），Adreno GPU 必须支持 OpenCL 3.0
2. Pixel 8 / 9（Tensor G3/G4，OpenCL 在 Mali 上跑，备用）

### 2.3 编译命令模板

```bash
# Android cross-compile, OpenCL backend
mkdir build-android && cd build-android
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-31 \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_TARGET_VERSION=300 \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON
cmake --build . -j --target multi-lora-bench
```

PC 上离线生成 workload trace：
```bash
cd tools/workload-gen
python gen_workload.py \
  --duration 600 \
  --rate 0.5 \
  --pool-size 30 \
  --pareto-alpha 1.5 \
  --prompts prompts/sharegpt_400.json \
  --out workloads/baseline.json
```

部署到手机 + 跑实验：
```bash
adb push build-android/bin/multi-lora-bench /data/local/tmp/
adb push models/llama-3.2-3b-Q4_0.gguf /data/local/tmp/
adb push lora_adapters/ /data/local/tmp/lora_adapters/
adb push workloads/baseline.json /data/local/tmp/

adb shell '/data/local/tmp/multi-lora-bench \
  --model /data/local/tmp/llama-3.2-3b-Q4_0.gguf \
  --adapter-dir /data/local/tmp/lora_adapters/ \
  --workload /data/local/tmp/baseline.json \
  --max-adapter-mem-mb 2048 \
  --n-slots 8 \
  --out /sdcard/baseline_results.csv'

adb pull /sdcard/baseline_results.csv ./results/

cd tools/workload-gen
python plot_results.py ../results/baseline_results.csv
```

---

## 3. llama.cpp 现状与改造点

### 3.1 LoRA 现状

> ⚠️ 本节里出现的 C API 符号都已迁移。**实际签名以 `docs/multi-lora/api-versions.md`
> 为准**——那份文件 pin 在具体 commit 上、给出真实参数表，跟下面的概念性
> 描述配合看。本节后续 pseudocode（§1.1.2 / §5.1 / §5.3 等）也是概念草图，不是
> 可直接 copy-paste 的代码。

llama.cpp 当前支持的 LoRA 操作（概念层面）：
- **加载 adapter**：`llama_adapter_lora_init(model, path)` 返回 `llama_adapter_lora *` 句柄
- **设置 / 清空 adapter set**：`llama_set_adapters_lora(ctx, adapters[], n_adapters, scales[])`
  *单次调用替换整个 active adapter set*；`n_adapters=0` 即清空。**不再是 `set` / `clear`
  两个独立 API**；旧版本里"设置后会与已有 adapter 累加"的隐式语义没了。
- **多 adapter additive blending**：通过传多个 `(adapter, scale)` 一次性激活
- **释放**：`llama_adapter_lora_free(adapter)`

`llama_set_adapters_lora` 内部有快速路径："如果 ctx 里当前激活的 adapter set 跟新传入
的相同就不动"——所以 hot-cache 命中时反复传同一个 handle 是廉价操作，AdapterPool
不必自己缓存 "currently set" 状态。

**所以单 context 内多 adapter 共存是支持的——但仍是 *additive blending*，不是
*per-token / per-slot dispatch*。M2 group-by-adapter 调度策略由此而来。**

### 3.2 缺失的能力

| 缺失 | 影响 | 改造方案 |
|------|------|---------|
| 单 forward 内多 adapter 动态切换 | 不支持 batch 内 per-slot 不同 adapter | 同 adapter 才进同一 batch（group by adapter） |
| Adapter pool / 缓存抽象 | 每次 set/clear 都要重建 tensor | 在 llama_context 之上加 pool 层 |
| LRU eviction | 加载不限量，OOM | 自己写 pool manager |
| Load 性能 | `llama_lora_adapter_init` 同步 + 全权重读 | 第一版用现有 API；后续可 mmap + lazy |
| Per-request scaling | global scale | 每个 slot 独立 scale |

### 3.3 改造策略

**最小改动原则**：
- 第一版**不改 ggml** —— 完全用 llama.cpp 现有 LoRA API
- **不改 OpenCL kernel** —— LoRA 在 cpu 张量上做，因为 adapter 占比小
- 主要工作在 `examples/multi-lora-bench/` 新增代码 + 极少 `llama-adapter.cpp` patch

### 3.4 不改的边界

第一版**禁止**做的修改：
- 不要碰 `ggml.c`
- 不要重写 OpenCL kernel
- 不要改 batch token 的内存布局
- 不要碰 KV cache 实现

如果某个需求必须改这些，停下来重新评估 scope。

---

## 4. 实现路线图（5 个里程碑）

每个里程碑都有：**目标 / 涉及文件 / 接口 / 验收标准**。

### M0：单 LoRA 在 OpenCL on Android 上跑通

**目标**：验证 toolchain 工作，建立基线性能测点。

**涉及文件**：
- 仅 `CMakeLists.txt` 配置改动，无代码改动

**任务**：
1. Fork `llama.cpp` 到 `mobile-multi-lora` 分支
2. 配置 Android NDK 编译，OpenCL 后端打开
3. 在 Snapdragon 手机上跑 `llama-cli` + 单个 LoRA adapter（用 `--lora` 参数）
4. 记录基线性能（TTFT、tok/s、内存峰值）
5. 写一份 `docs/multi-lora/M0.md` 记录硬件 + 软件 + 数据点

**验收**：
- `llama-cli --model llama-3.2-3b-Q4_0.gguf --lora <adapter>.gguf -p "hello"` 在 Adreno GPU 上正确生成
- TTFT < 2s，tok/s ≥ 10（基线参考）
- `docs/multi-lora/M0.md` 完成

#### Implementation Notes (M0a, 2026-05-02)

执行时拆成 M0a (host x86_64 sanity) + M0b (Android Adreno OpenCL)。M0a 已通过；M0b
依赖物理设备授权，未启动。完整记录见 `docs/multi-lora/M0.md`，要点：

1. **`-DLLAMA_BUILD_SERVER=OFF` 会破坏 `llama-cli` 构建**（§2.3 build 模板里的
   flag）。`tools/cli/CMakeLists.txt` 链接 `server-context`。M1 起手前要么删掉这个
   flag，要么解耦 cli 与 server-context。
2. **`llama-cli` 在当前 master 上是 interactive-only**——非交互补全应当走
   `llama-completion`。本节 acceptance 命令 `llama-cli ... -p "hello"` 在新版本
   会进入 chat 模式而不是单 prompt 跑完。M1 起手时改 spec 与 API 名字一并 grep 验证
   (见 `known-issues.md` I-3)。
3. **base model HF id 是 gated**：`meta-llama/Llama-3.2-3B-Instruct` 返 401，
   `convert_lora_to_gguf.py` 改用 `unsloth/Llama-3.2-3B-Instruct` 镜像（同架构）。
   GGUF 权重从 `bartowski/Llama-3.2-3B-Instruct-GGUF` 拉 Q4_0 文件。
4. **默认 `--ctx-size 0` 会把上下文撑到模型最大值**（Llama-3.2 是 131K），host
   RSS 立即到 18 GB。M1 起 bench 二进制必须显式 pin ctx（默认 4096，按需上调）。

---

### M1：多 adapter 加载，串行切换（in-process）

**目标**：能加载 N 个 adapter 到内存，按预定 trace 串行切换执行。**全在同一进程，无网络层**。

**涉及文件**：
- 新建 `examples/multi-lora-bench/adapter_pool.cpp/h`
- 新建 `examples/multi-lora-bench/main.cpp`
- 新建 `examples/multi-lora-bench/metrics.cpp/h`
- 新建 `examples/multi-lora-bench/CMakeLists.txt`
- 新建 `tools/workload-gen/gen_workload.py`

**接口设计**：

```cpp
// adapter_pool.h
class AdapterPool {
public:
  AdapterPool(llama_model* model,
              const std::string& adapter_dir,
              size_t max_resident);

  // 确保 adapter 在内存中并 set 到 ctx；返回 handle
  llama_lora_adapter* acquire(llama_context* ctx, const std::string& adapter_id);

  // 释放引用
  void release(const std::string& adapter_id);

  void shutdown();

private:
  std::unordered_map<std::string, AdapterEntry> resident_;
  std::list<std::string> lru_;
  size_t max_resident_;
  llama_model* model_;
  std::string adapter_dir_;
  std::mutex mu_;
};
```

```cpp
// metrics.h
struct RequestMetric {
  std::string id;
  std::string adapter_id;
  double arrival_time;       // since experiment start, seconds
  double first_token_time;   // since experiment start
  double finish_time;
  size_t n_input_tokens;
  size_t n_output_tokens;
  double TTFT() const  { return first_token_time - arrival_time; }
  double TPOT() const  { return (finish_time - first_token_time)
                                / std::max<size_t>(n_output_tokens - 1, 1); }
};

class Metrics {
public:
  void record(RequestMetric m);     // 线程安全
  void dump_csv(const std::string& path);
private:
  std::vector<RequestMetric> records_;
  std::mutex mu_;
};
```

```cpp
// main.cpp 主流程（M1 版，无 batching）
int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  // 1. 加载 base
  llama_model* model = llama_load_model_from_file(args.model_path, ...);
  llama_context* ctx = llama_new_context_with_model(model, ...);

  // 2. 初始化 pool 与 metrics
  AdapterPool pool(model, args.adapter_dir, args.max_resident);
  Metrics metrics;

  // 3. 加载 workload trace（pre-generated JSON）
  auto requests = load_workload(args.workload_path);

  // 4. 串行 driver loop（M1 单线程；M2 升级成 scheduler）
  auto t_start = now();
  for (const auto& req : requests) {
    sleep_until(t_start + req.arrival_time);
    double arrival = elapsed(t_start);

    auto* adapter = pool.acquire(ctx, req.adapter_id);
    RequestMetric m{req.id, req.adapter_id, arrival, 0, 0, req.input_tokens.size(), 0};

    auto tokens = req.input_tokens;
    llama_decode(ctx, build_batch(tokens));            // prefill

    bool first = true;
    for (size_t i = 0; i < req.max_output; i++) {
      llama_token tok = sample_token(ctx);
      if (first) { m.first_token_time = elapsed(t_start); first = false; }
      m.n_output_tokens++;
      if (tok == EOS) break;
      llama_decode(ctx, build_batch({tok}));           // decode
    }
    m.finish_time = elapsed(t_start);
    metrics.record(m);
    pool.release(req.adapter_id);
  }

  metrics.dump_csv(args.out_path);
  return 0;
}
```

**Workload trace JSON 格式**（PC 上 `gen_workload.py` 生成）：

```json
[
  {
    "id": "req_00001",
    "arrival_time": 1.234,
    "adapter_id": "alpaca-medical",
    "prompt": "What are common symptoms of...",
    "input_tokens": [128000, 3923, 527, ...],
    "max_output": 256
  },
  ...
]
```

**任务**：
1. PC 端：写 `gen_workload.py`，参数 (duration, rate, pool_size, pareto_alpha)，输出 JSON
2. 手机端：实现 `AdapterPool`（基础 LRU，固定 max_resident=10）
3. 手机端：实现 `Metrics`，CSV 输出
4. 手机端：`main.cpp` 串行 trace replay
5. 实测：跑一个 60s 短 trace 验证正确性

**验收**：
- `multi-lora-bench --workload <60s_trace.json> --out m.csv` 跑完不 crash
- CSV 文件包含 TTFT、TPOT 等 metric
- 串行切换 100 次 adapter 不 crash
- 输出对得上：每个 adapter 输出风格不同（人工抽检 5 条）
- Cold vs hot acquire 时间能从 metric 拆出

#### Implementation Notes (M1, 2026-05-02)

完整记录见 `docs/multi-lora/M1.md` + `docs/multi-lora/api-versions.md`。要点：

1. **API 重命名全部走 `api-versions.md`**——本节及上游 §1.1.2 / §5.1 / §5.2 / §5.3
   伪代码里写的 `llama_lora_adapter_set/_clear/_init/_free`、`llama_load_model_from_file`、
   `llama_token_eos(model)` 等等已经全部不存在；M1 实际代码用 `llama_set_adapters_lora`
   (replace-semantics)、`llama_model_load_from_file`、`llama_vocab_is_eog(vocab,tok)` 等。
2. **`AdapterPool` / `Metrics` 不要加 mutex**——CLAUDE.md 单线程 main loop。spec §5.1
   伪代码里的 `std::mutex mu_;` 在实际实现里删掉了。
3. **CMake 里 common 库的 target 名字是 `llama-common`，不是 `common`**——例子工程
   要写 `target_link_libraries(... PRIVATE llama llama-common)`，否则 `<nlohmann/json.hpp>`
   找不到。
4. **CSV schema 加了 `acquire_ms` 一列**：spec §4.M4 的 schema
   `id,adapter_id,arrival,first_token,finish,n_input_tokens,n_output_tokens,cache_hit`
   不足以从 metric 拆出 cold/hot acquire 时间（§4.M1 acceptance 第五条要求的）。
   M1 实现增加 `acquire_ms` 列（hit 时为 0，miss 时为 LoRA load + bind 总耗时）。
   M4 的 plot/summarize 脚本届时同步更新 schema 期望。
5. **Workload generator 的 `apply_chat_template(tokenize=True)` 返回 `BatchEncoding`
   dict，不是 token list**（transformers 5.x）。`gen_workload.py` 显式传
   `return_dict=False`。
6. **adapter_id → 文件名约定** `<adapter_dir>/<id>.gguf`，但若 id 含 `/` 或以 `.gguf`
   结尾则视为字面路径透传。文件 layout 上对外推荐 symlink 到短 id（设备端：
   `models/lora/reasoning.gguf -> reasoning-r16-f16.gguf`）。

---

### M2：同 adapter 内的 batching（Group LoRA）

**目标**：M1 是 *单请求串行*；M2 把同一 adapter 的多个 in-flight 请求合到一个 batch，提升吞吐。仍然 single-thread main loop，**不引入 future / 不引入线程**。

**涉及文件**：
- 新建 `examples/multi-lora-bench/scheduler.cpp/h`
- 新建 `examples/multi-lora-bench/slot.cpp/h`
- 修改 `main.cpp`：把 §1.1.2 的主循环 ②③④ 抽到 `Scheduler` 里

**接口设计（无 future、无线程）**：

```cpp
// slot.h
struct Slot {
  Request req;                       // 来自 trace
  llama_seq_id seq_id;               // llama.cpp seq id
  std::vector<llama_token> output;
  size_t prefill_done = 0;           // prefill 已喂的 token 数
  bool first_token_recorded = false;
  double first_token_time = 0;
};

bool is_in_prefill(const Slot& s);
bool is_finished(const Slot& s, llama_token last);
```

```cpp
// scheduler.h
class Scheduler {
public:
  Scheduler(llama_context* ctx, AdapterPool* pool, Metrics* m,
            size_t max_concurrent_slots);

  // 把新到达的 request 注册成 active slot
  void admit(const Request& req);

  // 推进一步：选最大 adapter 组 → set adapter → llama_decode → 收 token
  // 返回是否还有 active slot
  bool step(double now_seconds);

  size_t active_count() const;

private:
  std::vector<Slot*> pick_largest_adapter_group();

  std::vector<std::unique_ptr<Slot>> active_;
  llama_context* ctx_;
  AdapterPool* pool_;
  Metrics* metrics_;
  size_t max_slots_;
};
```

```cpp
// main.cpp（M2 版本，主循环更短了）
int main(int argc, char** argv) {
  auto cfg = parse_args(argc, argv);
  auto* model = llama_load_model_from_file(cfg.model_path, ...);
  auto* ctx = llama_new_context_with_model(model, ...);
  AdapterPool pool(model, cfg.adapter_dir, cfg.max_adapter_mb);
  Metrics metrics;
  Scheduler sched(ctx, &pool, &metrics, /*max_slots=*/cfg.n_slots);

  auto workload = load_trace(cfg.workload_path);
  std::sort(workload.begin(), workload.end(), by_arrival);

  size_t next_idx = 0;
  auto t0 = steady_clock::now();

  while (next_idx < workload.size() || sched.active_count() > 0) {
    double now = elapsed(t0);

    // 把已到达的 req 加进 scheduler（受 max_slots 限制；超过则等）
    while (next_idx < workload.size()
           && workload[next_idx].arrival_time <= now
           && sched.active_count() < cfg.n_slots) {
      sched.admit(workload[next_idx++]);
    }

    if (sched.active_count() == 0) {
      sleep_until(t0 + seconds(workload[next_idx].arrival_time));
      continue;
    }

    sched.step(now);
  }

  metrics.dump_csv(cfg.out_path);
}
```

**调度策略 `pick_largest_adapter_group`**：

```
1. 把所有 active slot 按 adapter_id 分桶
2. 选 size 最大的那个桶（tie-breaker: oldest slot first）
3. 返回该桶里的所有 slot
4. （第一版不限 batch size 上限；后续可加 cap，比如 8 token/step）
```

**Pick batch 时的 starvation 处理（先简化版）**：
- 第一版**不解决 starvation**——如果某 adapter 永远是少数，就让它等
- 验收时如果发现有 adapter 卡 > 30s，说明需要 priority age；记录但不在 M2 修

**任务**：
1. 抽 `Slot` struct 与工具函数（is_finished、build_llama_batch）
2. 实现 `Scheduler::admit / step / pick_largest_adapter_group`
3. 改 `main.cpp` 让 main loop 调用 sched，主循环只剩 admit + step
4. 单元测试：mock llama_context，构造 4 个 slot 跨 2 adapter，测 batch 形成

**验收**：
- 同 adapter 并发 8 个请求：吞吐 ≥ M1 的 3×（不要求 4×，因为 prefill 还是序列化）
  ⚠ **hardware-conditional**：Adreno 750 OpenCL 上不达标（实测 ≈ 1.0×）；
  TTFT p99 改善 4.4×，作为 partial proxy。详见 `docs/multi-lora/M2.md`。
- 4 adapter × 2 请求并发：吞吐相比单请求 ≥ 1.5×（adapter 切换 amortize）
  ⚠ 同上 hardware-conditional；adapter swap 数减半（5→3）但 wall 没动。
- 没有 slot 卡死：任何 active slot 在 30s 内必定推进 ≥ 1 token ✓
- 100% 请求正确响应（output 内容跟 M1 单请求模式一致，token-by-token diff）✓
  （以 `n_output_tokens` + EOG 位置作为 greedy 等价证明）

#### Implementation Notes (M2, 2026-05-02)

完整记录见 `docs/multi-lora/M2.md`，要点：

1. **`llama_sampler_sample(smpl, ctx, idx)` 的 idx 是 batch 位置而非 logits 计数**——
   不是 j-th 输出，而是 `batch.logits[idx]==1` 那一格。第一版我读漏了这个，跑起来直接
   `GGML_ASSERT(logits != nullptr)` 抛。修复：build batch 时记 `slot.sample_idx =
   batch.n_tokens - 1` 在写入 logits-bearing token 那一刻。`api-versions.md` 已校正。
2. **`pick_largest_adapter_group` 的平局必须显式按 admit 顺序**——
   `unordered_map` 迭代顺序不确定，没有显式 tie-break 时同一份 trace 跑两次产出不同
   schedule（哪怕 greedy 采样）。fix: 走 `active_` 顺序记 `first_idx`，平局取最小。
3. **`first_token_time` / `finish_time` 在 `llama_decode` 返回之后再读时钟**——
   step 入口拿的 `now_seconds` 是 *batch 开始* 时间，不是 *first token 出来* 时间。
   M1 的"sample_token 之后立刻打时间戳"语义要求 post-decode 读。Scheduler 现在持有
   `t0_` 自己 `seconds_since_t0()`。
4. **同 binary，`--n-slots=1` 即 M1-equivalent serial baseline**——M2 binary 替换
   M1 binary（同 CLI，多了 `--n-slots`）。"M1 vs M2" 对照实际是 `--n-slots=1`
   vs `--n-slots>1`。
5. **吞吐目标（≥3×）在 Adreno OpenCL 后端达不到**：单 token decode 是带宽 bound，
   batch=8 step ≈ 8 × batch=1 step。spec §6.2 已经标注过 "r=16 LoRA matmul Adreno
   利用率低"，这个限制延伸到 base decode batching 层面。fix 在 M5（自定义 OpenCL
   kernel），不在 scheduler。

---

### M3：Size-aware LRU cache（**为 paper experiment 服务，不是为了解决 OOM**）

**Framing**：典型 mobile workload（10–50 个 r=16 adapter，总 ≤ 1 GB）在 16 GB 手机上**全部驻留 LPDDR 完全装得下**，LRU 永远不触发。M3 的目的是给实验**人为加内存约束**，让 cache hit rate / evict count / TTFT 这些 metric 有意义、能画出 scaling curve。

具体使用场景：
- **Scaling experiment**：x 轴是 `--max-adapter-mem-mb`，y 轴是 hit rate / TTFT，画出"内存预算 vs 性能"曲线
- **Baseline 对比**：把 budget 设成 1 个 adapter 大小（强制 cold），跟 vanilla llama.cpp swap 模式对比
- **Stretch 实验**：未来跑 1000 adapter 池或 r=128 大 rank 时，budget 必须收紧才不爆

**涉及文件**：
- 修改 `adapter_pool.cpp/h`：M1 版本只有 max_count，M3 改成 max_bytes
- 加 `--max-adapter-mem-mb` CLI 参数

**接口扩展**：

```cpp
class AdapterPool {
public:
  AdapterPool(llama_model* model,
              const std::string& adapter_dir,
              size_t max_bytes);                // ← M3 改成 byte budget

  llama_lora_adapter* acquire(llama_context* ctx, const std::string& id);
  void release(const std::string& id);

  // 新增统计接口（实验输出用）
  size_t resident_bytes() const;
  size_t resident_count() const;
  size_t hit_count() const;
  size_t miss_count() const;
  size_t evict_count() const;
  double hit_rate() const { return double(hit_count()) / (hit_count() + miss_count()); }

private:
  void evict_until(size_t bytes_needed);
};
```

**LRU 实现要点**：
- 用 `std::list<std::string>` 维护 LRU 顺序（双向链表，O(1) 移动）
- 用 `std::unordered_map<std::string, list::iterator>` 配合 O(1) 定位
- Evict 时调用 `llama_lora_adapter_free()` 释放底层 tensor
- 引用计数：只 evict `ref_count == 0` 的 adapter（防止正在用的被踢）
- Size 来源：第一次 load 后用 `llama_lora_adapter_n_params() * sizeof(half)` 估算

**任务**：
1. 把 M1 的 max_count 替换成 max_bytes
2. 实现 `evict_until(bytes_needed)`
3. 加四个 counter（hit/miss/evict/resident_bytes）
4. metric CSV 末尾写入 pool 总统计行
5. 单元测试（host 上跑，mock `llama_lora_adapter`）

**验收**：
- **Scaling 实验**：跑同一份 trace，扫 `--max-adapter-mem-mb ∈ {18, 100, 500, 2000, ∞}`，输出 5 份 CSV，summary 显示 hit rate 单调上升、TTFT p99 单调下降——这是 paper figure 的雏形
- **行为正确性**：连续访问 A → B → C 各 1 次，再访问 A（budget 容 2 个），B 应被 evict
- **不会 evict 正在用的**：构造一个边界 case（budget = 1 个 adapter），同时有 2 个 active slot 用同一 adapter，第 3 个 adapter 来时不 evict 当前 in-use 的


---

### M4：Workload 生成器 + 实验 harness（PC 侧）

**目标**：PC 上的 trace 生成 + CSV 后处理 + 跨 scenario 实验脚本。手机端的 bench binary 已经在 M3 完整，M4 是把它包装成可重复的实验流。

**涉及文件**（全部在 PC 上，不是手机上）：
- `tools/workload-gen/gen_workload.py` —— 生成 trace JSON
- `tools/workload-gen/plot_results.py` —— 读 CSV 出图
- `tools/workload-gen/summarize.py` —— 输出 metric summary（mean/p50/p99/throughput）
- `tools/workload-gen/configs/*.yaml` —— scenario 定义
- `tools/workload-gen/scripts/run_all.sh` —— 一键跑所有 scenario

**`gen_workload.py` 接口**：

```python
# tools/workload-gen/gen_workload.py
import argparse, json, random
import numpy as np

def pareto_sample(items, alpha):
    """Pareto distribution: hot items more likely"""
    weights = np.array([1.0 / (i + 1) ** alpha for i in range(len(items))])
    weights /= weights.sum()
    return np.random.choice(items, p=weights)

def generate(cfg):
    rng = np.random.default_rng(cfg['seed'])
    prompts = json.load(open(cfg['prompt_file']))
    requests = []
    t = 0.0
    idx = 0
    while t < cfg['duration_s']:
        t += rng.exponential(1.0 / cfg['rate_lambda'])
        adapter = pareto_sample(cfg['adapter_pool'], cfg['pareto_alpha'])
        prompt = random.choice(prompts)
        requests.append({
            'id': f'req_{idx:05d}',
            'arrival_time': t,
            'adapter_id': adapter,
            'prompt': prompt['text'],
            'input_tokens': prompt['tokens'],   # 预先 tokenize 好
            'max_output': cfg['max_output'],
        })
        idx += 1
    return requests

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    reqs = generate(cfg)
    json.dump(reqs, open(args.out, 'w'), indent=2)
    print(f'Wrote {len(reqs)} requests to {args.out}')
```

**Scenario YAML 模板**（`configs/baseline.yaml`）：

```yaml
seed: 42
duration_s: 600
rate_lambda: 0.5          # 0.5 req/s
adapter_pool:
  - alpaca-medical
  - alpaca-coding
  - tinyllama-chat
  # ... 50 个 adapter id（与 /data/local/tmp/lora_adapters/ 文件名对应）
pareto_alpha: 1.2
prompt_file: ./prompts/sharegpt_400.json
max_output: 256
```

**`plot_results.py` 接口**：

```python
# tools/workload-gen/plot_results.py
# 读手机产出的 CSV，画 4 张图：
#   1. TTFT CDF（按 adapter 颜色分）
#   2. Throughput timeline（每 10s 桶）
#   3. Adapter access pattern (访问次数 vs adapter)
#   4. Per-request TTFT vs arrival_time 散点
```

**CSV schema（手机 metric 输出）**：
```
id,adapter_id,arrival,first_token,finish,n_input_tokens,n_output_tokens,cache_hit
req_00001,alpaca-medical,1.234,1.523,8.412,387,256,0
...
```

**`summarize.py`**：

```python
# tools/workload-gen/summarize.py
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
df['TTFT'] = df['first_token'] - df['arrival']
df['TPOT'] = (df['finish'] - df['first_token']) / (df['n_output_tokens'] - 1).clip(lower=1)
print(f"requests:        {len(df)}")
print(f"TTFT mean:       {df.TTFT.mean()*1000:.1f} ms")
print(f"TTFT p50:        {df.TTFT.quantile(0.5)*1000:.1f} ms")
print(f"TTFT p99:        {df.TTFT.quantile(0.99)*1000:.1f} ms")
print(f"TPOT mean:       {df.TPOT.mean()*1000:.1f} ms/tok")
print(f"throughput req/s:  {len(df)/df.finish.max():.2f}")
print(f"throughput tok/s:  {df.n_output_tokens.sum()/df.finish.max():.2f}")
print(f"cache hit rate:   {df.cache_hit.mean()*100:.1f}%")
```

**任务**：
1. 准备 prompt pool：从 ShareGPT 采样 400 条，预先 tokenize 存成 JSON
2. 准备 adapter pool：从 HF Hub 拉 ≥10 个 r=16 LoRA（兼容 Llama-3.2-3B），转成 GGUF 格式
3. 实现 `gen_workload.py` + scenario YAML
4. 实现 `summarize.py` + `plot_results.py`
5. 写 `scripts/run_all.sh`：自动 push trace → 跑 bench → pull CSV → summarize
6. 跑 3 个 scenario：
   - S1：rate=0.1, pool=10, α=2.5（idle 突发）
   - S2：rate=0.5, pool=30, α=1.5（典型）
   - S3：rate=1.0, pool=50, α=1.2（高负载）

**`run_all.sh` 模板**：

```bash
#!/bin/bash
set -e
DEVICE=/data/local/tmp
HOST_OUT=./results

for scenario in idle_burst typical high_load; do
  echo "=== $scenario ==="
  python gen_workload.py --config configs/$scenario.yaml \
    --out workloads/$scenario.json
  adb push workloads/$scenario.json $DEVICE/
  adb shell "$DEVICE/multi-lora-bench \
    --model $DEVICE/llama-3.2-3b-Q4_0.gguf \
    --adapter-dir $DEVICE/lora_adapters/ \
    --workload $DEVICE/$scenario.json \
    --max-adapter-mem-mb 2048 --n-slots 8 \
    --out /sdcard/$scenario.csv"
  adb pull /sdcard/$scenario.csv $HOST_OUT/
  python summarize.py $HOST_OUT/$scenario.csv > $HOST_OUT/$scenario.summary.txt
  python plot_results.py $HOST_OUT/$scenario.csv \
    --out-dir $HOST_OUT/figures/$scenario/
done
```

**能耗采集**（可选 stretch）：

```bash
# 实验之前 reset
adb shell dumpsys batterystats --reset
adb shell setprop debug.power.profiler 1

# 跑实验（同上）

# 实验之后
adb shell dumpsys batterystats > $HOST_OUT/$scenario.batstats.txt
adb shell cat /sys/class/power_supply/battery/uevent > $HOST_OUT/$scenario.battery.txt
# 用 battery_historian 工具解析
```

更准的方法（需要硬件改装）：
- Snapdragon Profiler GUI（高通官方，但要 root）
- Monsoon Power Monitor（外接，断电池供电）

**验收**：
- 3 个 scenario 一条 `./run_all.sh` 全部跑完不挂
- 每个 scenario 出一份 summary + 4 张图
- TTFT p99、throughput、cache hit rate 三个数字与 M3 验收时手测一致
- 跨 scenario 对比表（手工汇总）：cache hit 高的 scenario TTFT 更低，吞吐更高


---

## 5. 核心组件详细设计

### 5.1 Adapter Pool（M1–M3）

**职责**：管理 N 个 adapter 在内存中的驻留 / 加载 / 驱逐。

**数据结构**：

```cpp
struct AdapterEntry {
  std::string id;
  llama_lora_adapter* adapter;     // llama.cpp handle
  size_t bytes;                    // adapter 占用字节
  std::atomic<int> ref_count;      // 多少 slot 正在用
  std::list<std::string>::iterator lru_pos;
};

class AdapterPool {
  std::unordered_map<std::string, AdapterEntry> resident_;
  std::list<std::string> lru_;     // front = MRU, back = LRU
  std::mutex mu_;                  // 保护 resident_ + lru_
  size_t max_bytes_;
  size_t current_bytes_ = 0;

  // metrics
  std::atomic<size_t> hits_{0}, misses_{0}, evicts_{0};
};
```

**关键操作**：

```cpp
// 上锁 acquire
llama_lora_adapter* acquire(ctx, id):
  lock(mu_);
  if (resident_.has(id)) {
    hits_++;
    move_to_front(id);
    entry = &resident_[id];
    entry->ref_count++;
    set_adapter_on_ctx(ctx, entry->adapter);
    return entry->adapter;
  }
  misses_++;
  // need to load
  size_t need = estimate_size(id);
  evict_until(need);   // 可能释放别的 adapter
  auto* new_adapter = llama_lora_adapter_init(model_, path_for(id).c_str());
  size_t actual = adapter_size(new_adapter);
  resident_[id] = AdapterEntry{id, new_adapter, actual, 1, ...};
  lru_.push_front(id);
  resident_[id].lru_pos = lru_.begin();
  current_bytes_ += actual;
  set_adapter_on_ctx(ctx, new_adapter);
  return new_adapter;

release(id):
  lock(mu_);
  resident_[id].ref_count--;
  // 不立即 evict，保留在 LRU 后面

evict_until(bytes_needed):
  // 已锁
  while (current_bytes_ + bytes_needed > max_bytes_ && !lru_.empty()) {
    auto victim_id = lru_.back();
    auto& victim = resident_[victim_id];
    if (victim.ref_count > 0) {
      // 跳过正在用的，找前面的
      // 实际实现：用辅助 list 把 ref==0 的单独排
      lru_.splice(lru_.begin(), lru_, --lru_.end());
      continue;
    }
    llama_lora_adapter_free(victim.adapter);
    current_bytes_ -= victim.bytes;
    evicts_++;
    resident_.erase(victim_id);
    lru_.pop_back();
  }
  if (current_bytes_ + bytes_needed > max_bytes_) {
    throw std::runtime_error("OOM: cannot fit even after evict");
  }
```

**注意**：
- 第一版用粗粒度 mutex；并发性能不优但正确
- 性能优化（细粒度锁 / RCU）留 future
- ref_count 是 atomic 但 LRU 操作仍需 lock

### 5.2 Slot State Machine（M2）

**状态**：

```
IDLE → PROMPT → GENERATING → FINISHED → IDLE
  ↑                                       │
  └───────────────────────────────────────┘
```

**字段**（无 future / promise，main loop 同进程读 metric）：

```cpp
struct Slot {
  Request req;                      // 来自 trace 的整条请求
  llama_seq_id seq_id;              // llama.cpp 内部 sequence ID
  std::vector<llama_token> output;
  size_t prefill_done = 0;          // 已 fed prompt token 数
  bool first_token_recorded = false;
  double first_token_time = 0;      // 实验时间秒
  // 生成完成时由 main loop 写 metric record，slot 移除
};
```

注意：M2 没有 IDLE/PROMPT/GENERATING/FINISHED 这种状态机——slot 一旦 admit 就是 active，prefill 与 decode 用 `prefill_done` 区分。状态机过于复杂，单线程 main loop 不需要。

**调度逻辑**（在 `Scheduler::step` 里，每次跑一步 decode）：

```cpp
bool Scheduler::step(double now_seconds) {
  if (active_.empty()) return false;

  // 1. 选最大 adapter 组
  auto batch = pick_largest_adapter_group();

  // 2. 切 adapter
  pool_->acquire(ctx_, batch[0]->req.adapter_id);

  // 3. 构建 llama_batch（混合 prefill 与 decode token）
  llama_batch llb = build_llama_batch(batch);

  // 4. 一次 forward
  llama_decode(ctx_, llb);

  // 5. 处理每个 slot 的输出
  for (auto* slot : batch) {
    if (still_in_prefill(slot, llb)) {
      slot->prefill_done += tokens_for_slot(llb, slot->seq_id);
      continue;
    }
    auto tok = sample_token(ctx_, slot->seq_id);
    if (!slot->first_token_recorded) {
      slot->first_token_time = now_seconds;
      slot->first_token_recorded = true;
    }
    slot->output.push_back(tok);
    if (tok == EOS || slot->output.size() >= slot->req.max_output) {
      finalize_and_record(slot, now_seconds);
      remove_slot(active_, slot);
    }
  }

  pool_->release(batch[0]->req.adapter_id);
  return !active_.empty();
}
```

**pick_largest_adapter_group**：

```cpp
std::vector<Slot*> Scheduler::pick_largest_adapter_group() {
  std::unordered_map<std::string, std::vector<Slot*>> groups;
  for (auto& uptr : active_) {
    groups[uptr->req.adapter_id].push_back(uptr.get());
  }
  auto best = std::max_element(groups.begin(), groups.end(),
    [](const auto& a, const auto& b) { return a.second.size() < b.second.size(); });
  return best->second;
}
```

**finalize_and_record**：

```cpp
void Scheduler::finalize_and_record(Slot* slot, double now_s) {
  RequestMetric m;
  m.id = slot->req.id;
  m.adapter_id = slot->req.adapter_id;
  m.arrival_time = slot->req.arrival_time;
  m.first_token_time = slot->first_token_time;
  m.finish_time = now_s;
  m.n_input_tokens = slot->req.prompt.size();
  m.n_output_tokens = slot->output.size();
  metrics_->record(m);
}
```

**待优化**：
- Starvation 防止（少数 adapter 永远等不到）—— priority age 机制，第一版先不做
- Batch size 上限（避免单个 adapter 占住）—— 加个 `max_batch_size` 配置项
- pick 策略 tie-breaker（多个 adapter 组同 size）—— 第一版按 oldest first

### 5.3 Group LoRA Inference（M2 集成 llama.cpp 的 LoRA API）

第一版**不写自定义 OpenCL kernel**——直接用 llama.cpp 现有 multi-adapter API（实际
签名见 `docs/multi-lora/api-versions.md`）：

```cpp
// 切 active adapter set —— 一次调用替换整组，不是先 clear 再 set
struct llama_adapter_lora * adapters[1] = { handle_for_target_id };
float scales[1] = { 1.0f };
llama_set_adapters_lora(ctx, adapters, 1, scales);

// 然后 llama_decode 正常跑
llama_decode(ctx, batch);

// 清空（adapter 设回 base-only）：传 nullptr / 0
llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
```

`llama_set_adapters_lora` 自带 fast-path："如果 ctx 里当前 active set 跟传入参数完全
一致就不动"，所以反复传同一 handle 不会做多余工作。

**性能假设**：因为 batch 里所有 slot 用同一 adapter，llama.cpp 现有逻辑就能正确处理——LoRA 计算对整个 batch 生效。

**验证**：跑 batch_size=1 vs batch_size=8 同 adapter，throughput 应该 ≥ 4×（GPU 并行起作用）。

### 5.4 OpenCL backend 注意事项

llama.cpp 的 GGML OpenCL 后端 (`ggml/src/ggml-opencl/`) 由 Qualcomm 维护，针对 Adreno。**第一版不要碰它**。

但有几个事项需要确认：
1. 编译时 `GGML_OPENCL=ON`，运行时 `LLAMA_ARG_NUMA=0`（OpenCL 与 NUMA 不兼容）
2. KV cache 的 layout 在 OpenCL 是 `f16`，确保 LoRA 也是 `f16`（不要 f32 混用导致 kernel 路径失败）
3. Adreno 上 OpenCL kernel 编译有 cache（`~/.cache/`），首次启动慢
4. 内存分配走 `clCreateBuffer(CL_MEM_ALLOC_HOST_PTR)`，是 unified memory pool，不需要 explicit copy

### 5.5 Tokenizer 与 prompt cache

llama.cpp 的 tokenizer 是 base model 自带的——adapter 不会改 tokenizer。第一版直接用。

### 5.6 KV cache

`llama_context` 内置 KV cache，slot-based 时每个 slot 是 *独立 sequence*。当前 API
通过一个 `llama_memory_t` 句柄操作（旧 `llama_kv_cache_*` 全部迁到 `llama_memory_*`，
详见 `api-versions.md`）：

```cpp
slot[0].seq_id = 0;
slot[1].seq_id = 1;
// 清掉某个 seq 的 KV：
llama_memory_t mem = llama_get_memory(ctx);
llama_memory_seq_rm(mem, /*seq_id=*/0, /*p0=*/-1, /*p1=*/-1);
```

是 llama.cpp 现有功能，不需要新写。

---

## 6. OpenCL 后端要点（不修改，仅理解）

### 6.1 现状

- 路径：`ggml/src/ggml-opencl/`
- 维护方：Qualcomm 工程师（注释里能看到）
- 主要 op：matmul / dequant / RoPE / attention / softmax 等
- 数据格式：`f16` 主，`q4_0` / `q4_K` / `q5_K` / `q6_K` 量化后通过 dequant kernel（第一版只用 `q4_0`）
- 不支持：动态 shape（kernel 全是 specialized for fixed dims）

### 6.2 LoRA 在 OpenCL 上发生什么

llama.cpp 的 LoRA 是通过 *额外的 GGML node* 实现：
- 原 op：`y = x @ W`
- 加 LoRA：`y = x @ W + scale * (x @ A) @ B`
- 这两个额外 matmul 也要在 OpenCL 上跑

**问题**：`x @ A` 的输出形状是 `(seq, r)`，r=16 在 GPU 上太瘦，kernel 利用率低。

**第一版的取舍**：接受这个低效——优化 LoRA kernel 是 stretch goal。

### 6.3 如果要写 custom LoRA OpenCL kernel（M5+ stretch）

参考 Punica 的 SGMV 思路：
- 输入：`x[batch, seq, hidden]`，多个 adapter 的 `(A_i, B_i)`，每个 token 的 `adapter_id[batch, seq]`
- 输出：`y[batch, seq, hidden] = x @ W + scale * gather_then_apply_lora(x, A, B, adapter_id)`

OpenCL kernel 大致结构：
```c
__kernel void multi_lora_forward(
  __global const half* x,         // [B, S, H]
  __global const half* W,         // base weight [H, H_out]
  __global const half* A_pool,    // [N_adapter, R, H]
  __global const half* B_pool,    // [N_adapter, H_out, R]
  __global const int* adapter_id, // [B*S]
  __global half* y,
  int B, int S, int H, int H_out, int R)
{
  int row = get_global_id(0);
  int col = get_global_id(1);
  // ... 略
}
```

这个 kernel 留 M5+ 实现，第一版用 llama.cpp 现有的多 LoRA additive blending（性能较差但够用）。

---

## 7. 测试 / 验证策略

### 7.1 单元测试（host 上跑）

```
tests/
├── adapter_pool_test.cpp     # mock llama_lora_adapter，测 LRU 行为
├── scheduler_test.cpp        # mock llama_context，测 group-by-adapter
└── slot_state_test.cpp       # 状态机 transition
```

每个里程碑结束都要补对应单元测试。

### 7.2 集成测试（手机 / 模拟器上）

```
integration_tests/
├── correctness/
│   ├── single_adapter.sh     # 单 adapter 输出对得上 vanilla llama-cli
│   └── adapter_isolation.sh  # 不同 adapter 输出确实不同
├── stress/
│   ├── 100_swaps.sh          # 100 次 adapter 切换不挂
│   └── 1hour_sustain.sh      # 1 小时压测
└── perf/
    ├── ttft_baseline.sh      # 单 req TTFT
    └── batch_throughput.sh   # 8 并发 throughput
```

### 7.3 端到端验证

每个 milestone 验收时跑 `run_milestone_M{i}.sh`，输出 `results/M{i}.csv` 和 `results/M{i}.png`。结果不达标不进入下一 milestone。

---

## 8. 评估协议（最终 paper 跑的 scenario）

### 8.1 平台

| Device | SoC | RAM | 注意 |
|--------|-----|-----|------|
| Redmi K70 Pro / Xiaomi 14 | SD 8 Gen 3 | 12–16 GB | 主要测试机 |
| Pixel 8 / 9 | Tensor G3/G4 | 8–12 GB | 跨 vendor 验证 |
| (备选) Jetson Orin Nano | NVIDIA NPU | 8 GB | 不算手机但 reproducible |

### 8.2 模型 + adapter

- **Base**：Llama-3.2-3B **Q4_0** (≈ 2 GB) —— 第一版选 Q4_0 而非 Q4_K_M，理由：OpenCL dequant kernel 路径更老更稳定，调试 / 性能基线更可重复；Q4_K_M 留 stretch goal
- **Adapter pool**（50 个，r=16）：
  - 从 HuggingFace Hub 取（搜 `lora` tag + 兼容 Llama-3.2-3B）
  - 涵盖：medical / coding / writing / math / chat / translation / summarization
  - 第一版假设全部 r=16；rank 异构留 future

### 8.3 Workload

| Scenario | rate | pool | Pareto α | 时长 |
|----------|------|------|----------|------|
| S1 idle-burst | 0.1 req/s | 10 | 2.5 | 600s |
| S2 typical | 0.5 req/s | 30 | 1.5 | 600s |
| S3 high load | 1.0 req/s | 50 | 1.2 | 600s |

Prompt 来自 ShareGPT（400 条采样）。

### 8.4 Baseline

| Baseline | 怎么跑 |
|----------|-------|
| llama.cpp `llama-server` | 单 adapter，每次切换重启 |
| llama.cpp 多 adapter set 模式 | 全部 adapter 加载，全 scale=0 除当前 |
| Ours (M3 + group LoRA) | 完整版 |
| (可选) EdgeLoRA artifact | 如果他们放出代码 |

### 8.5 Metric

| Category | Metric | 怎么算 |
|----------|--------|-------|
| **Latency** | TTFT mean / p50 / p99 | metric collection 自动 |
| | TPOT mean | 同上 |
| | E2E latency CDF | 画图 |
| **Throughput** | req/s | 完成请求数 / 总时长 |
| | tok/s | 总 token 数 / 总时长 |
| **Memory** | peak resident set | `cat /proc/$PID/status \| grep VmHWM` |
| | adapter cache hit rate | pool internal counter |
| **Energy** | mJ/token | batterystats 或 Snapdragon Profiler |
| **Quality** | output sample inspection | 每 adapter 抽 5 条人工看 |

---

## 9. 代码质量

### 9.1 Coding standard

- 遵循 llama.cpp 现有风格（snake_case 函数，`llama_*` 命名空间）
- 新增组件用 `multilora_*` 前缀（避免冲突）
- C++17 即可（不用 C++20，与 llama.cpp 一致）
- `clang-format` 用 llama.cpp 自带配置

### 9.2 错误处理

- `llama_*` API 失败一律抛 `std::runtime_error`，main 捕获后写 ERROR log，进程 exit code = 1
- I/O 错误（adapter 文件不存在）：日志 + 把该 request 标记 `failed=1` 写入 CSV，继续跑（不阻塞整个 trace）
- OOM 在 pool 层捕获：dump 当前 pool 状态到日志，进程 exit code = 2
- Trace JSON 解析错误：直接 abort（fail fast，避免 partial run）

### 9.3 日志

- 用 `fprintf(stderr, ...)` 配合 log level（DEBUG/INFO/WARN/ERROR）
- 关键事件：adapter load/evict、batch 形成、slot transition
- 性能事件：每个 metric 记录独立一行 JSON 方便后处理

### 9.4 Git workflow

- 每个 milestone 一个 PR
- PR 必须包含：代码 + 单元测试 + 集成测试 + `docs/multi-lora/M{i}.md` 写本 milestone 的 design notes 和 measurement
- Commit message 用 conventional commits（`feat:` / `fix:` / `test:`）

---

## 10. 常见坑（避免重复踩）

1. **llama.cpp 的 LoRA 加载是同步且会拷贝权重**——大 adapter 加载要 100ms+。第一版接受，后续优化。
2. **OpenCL kernel cache 第一次启动慢**（~30s），不要误认为挂了。
3. **Android `/data/local/tmp/` 没有持久化保证**，重启可能清空——加载脚本要重 push。
4. **GGUF adapter 必须用 *相同 base model* 训练**——直接用别的 base 训的 LoRA 加载会输出乱码。
5. **`llama_set_adapters_lora` 是 *replace* 语义**，**不是** 旧 API 那种 set/clear-then-set
   的累加语义——一次传整个 active set，scale 数组一一对应。N 个 adapter 同时激活就一次
   传 N 个 handle。详见 `docs/multi-lora/api-versions.md` LoRA 一节。
6. **Slot 内 KV cache 用错 `seq_id`** = output 互相污染。每个 slot 必须有独立 seq_id。
7. **多线程 + llama.cpp** 不安全——所有 `llama_*` 调用串行化（用 scheduler 主线程）。
8. **TTFT 必须在 *第一个 sample 出来的瞬间* 记录**——不要等整个请求完成再算，否则 metric 失真。在 `Scheduler::step` 里 `sample_token` 之后立刻打时间戳。
9. **Android 测能耗时把屏幕关掉 + 飞行模式 + 拔 USB**（USB 充电会污染电流读数）——`adb shell` 用 over-WiFi adb 替代。
10. **不要在 release 构建里留 DEBUG log**——log 本身能吃 5–10% 性能；用 `-DNDEBUG` + 编译期开关。
11. **Trace JSON 里 `arrival_time` 必须从 0 开始递增**——`gen_workload.py` 输出前 sort 一下；main loop 假设有序。
12. **预先 tokenize prompt 存进 trace JSON**——避免 main loop 里 tokenizer 开销污染 TTFT。`gen_workload.py` 跑的时候用 `llama_tokenize` Python binding 预处理。

---

## 11. 后续 milestone 提示（不在第一版 scope）

写完 M0–M4 之后可以考虑：

- **M5：自定义 OpenCL multi-LoRA kernel**（参考 Punica SGMV 思路，提升 batch 内并行）
- **M6：异构 rank 支持**（adapter pool 同 rank 分组，scheduler 跨组调度）
- **M7：Cross-adapter KV reuse**（参考 MobiLoRA delta encoding）
- **M8：Energy / thermal-aware scheduling**（这是 paper 的真正卖点）
- **M9：NPU backend (Hexagon QNN)**（与 OpenCL 共存，phase-aware split）

每个 M5+ 都是 *一篇 paper 一个 contribution* 的体量，不要在第一版里塞。

---

## 12. 给 Claude Code 的工作指引

### 12.1 工作模式
- 一次只做一个 milestone
- 完成后必须运行验收命令并把输出粘到 PR description
- 写完代码先跑单元测试再跑集成测试
- 不要跨 milestone 改文件（避免 PR 混乱）

### 12.2 决策权限
- 函数签名 / 数据结构：**必须遵循本文档**，不能擅自改
- 内部实现：**自由发挥**
- 新增依赖：**必须先问**（避免引入大型库）
- 修改 ggml.c / ggml-opencl：**严格禁止**（除非和我确认过）

### 12.3 提问触发条件

遇到以下情况立刻停下来问：
- llama.cpp 的某个内部 API 跟文档描述不符
- 某 milestone 的验收标准达不到
- 想引入新依赖
- 发现 spec 里有矛盾
- 想改本文档

### 12.4 文档更新

每完成一个 milestone：
- 在该 milestone 末尾加一段"Implementation Notes"记录踩到的坑
- 如果 spec 错了，提 PR 改文档（带原因）

---

*最后更新：2026-05-01。第一版 spec，覆盖 M0–M4。M5+ 留待后续。*
