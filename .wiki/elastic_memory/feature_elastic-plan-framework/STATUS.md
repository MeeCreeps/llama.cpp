# STATUS — elastic plan framework 实现总览(给 review)

> 分支:`feature/elastic-plan-framework`(从 `feature/elastic-op-sched-runtime` 切出)
> 日期:2026-06-03
> 一句话:**用户定义的核心 DoD(给一个 plan 能执行 + 内存变化换 plan)已实现并在桌面 Llama-3.2-3B 端到端验证通过。**

## 完成度

| Milestone | 内容 | 状态 | 验证 |
|---|---|---|---|
| M0 | 切分支 + plan_ir 骨架 + CMake | ✅ | 编译 |
| M1 | Plan IR + JSON 加载(native + make_plan 格式) | ✅ | `test_plan_ir` 加载真实 197w/197op/305ev plan |
| M2 | PlanExecutor apply(plan) 差量 reconcile | ✅ | `test_plan_executor`(fresh/replan/runtime/anchor) |
| M4-prov | PlanProvider table/callback | ✅ | `test_plan_provider`(29 bands + 指针稳定) |
| **M3 (DoD#1)** | **`llama_elastic_apply_plan` 喂 plan 执行** | ✅ | **E2E:apply→0 + decode 8 token 正确** |
| **M4 (DoD#2)** | **online loop 内存变化换 plan** | ✅ | **E2E:enable(callback)+ 切 plan + decode 继续** |
| M5 | D2b runtime dispatch 接线 + 迁移意图 | ✅(桌面) | hook 安装验证;decode 不回归 |
| M5-迁移 | 真 CPU↔GPU 迁移 + layout 转换 | ⏳ 设备侧 | 需 Android + GPU |
| M6 | overlap 编排(timeline→异步 prefetch) | ⏳ 设备侧 | 需 opencl WBM |
| M7 | trace 端到端 plan-driven vs static 实测 | ⏳ 设备侧 | 需真机 |

## 提交(3 个,master..HEAD)

```
M5    e4051a0  D2b runtime dispatch + 迁移意图接线
M3/M4 f474261  llama-context 接线 + C API, E2E 跑通
M0-M2 4d12c50  Plan IR + Executor + Provider 纯逻辑核心
```
对应文档:`CHANGES_01_runtime_core.md` / `CHANGES_02_llama_integration.md` / `CHANGES_03_runtime_dispatch.md`。

## 架构(三层)

```
B(t) ─► PlanProvider.get(B) ─► ExecPlan(IR) ─► PlanExecutor.apply() ─► WBM/op_schedule/dispatch
       (table 查档/callback)   (纯数据)         (差量 reconcile)        (真实后端)
```

- **runtime/plan_ir.{h,cpp}** — ExecPlan = WeightPlan[](weight 在哪+xform)+ OpPlan[](谁算+dispatch+migrate)+ PlanEvent[](搬运时间线+engine 归属)。native JSON + 加载 make_plan.py 格式。
- **runtime/plan_executor.{h,cpp}** — apply(plan):residency 差量 reconcile(D1)+ static routing(D2b)+ migration 意图(D2b)+ anchor 索引(D3)。ExecSinks 回调解耦后端。
- **runtime/plan_provider.{h,cpp}** — get(budget):table(≤B 最大档 + 同档同指针)/ callback。
- **src/llama-context.{h,cpp}** — `apply_exec_plan`(sinks 桥接 WBM/op_schedule/runtime_dispatch)+ `maybe_apply_plan`(online loop, 挂 decode 入口, 防抖)+ C API。
- **include/llama.h** — `llama_plan` opaque + load/apply/enable/provider/get C API。

## 怎么跑(桌面验证)

```bash
# 1. 纯逻辑单测
cmake -S runtime -B runtime/build && cmake --build runtime/build -j
cd runtime/build && ./tests_elastic/test_plan_ir ../../runtime/plan/plans/plan_4144MiB.json
                    ./tests_elastic/test_plan_executor
                    ./tests_elastic/test_plan_provider ../../runtime/plan/plans

# 2. 库编译(plan TUs 进 libllama)
cmake -S . -B build-native -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=OFF -DGGML_OPENCL=OFF -DLLAMA_CURL=OFF
cmake --build build-native --target llama -j

# 3. E2E(标准 llama arch 模型;in-tree 的 3B 即可,plans 就是给它标的)
g++ -std=c++17 -O2 -I include -I ggml/include tests/elastic/test_plan_e2e.cpp \
    -L build-native/bin -lllama -lggml -lggml-base -Wl,-rpath,build-native/bin -o build-native/test_plan_e2e
./build-native/test_plan_e2e models/llama-3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    runtime/plan/plans/plan_4144MiB.json
# 已验证可用模型:in-tree Q4_K_M 3B、F16 3B(a.gguf)。dllm/sdar 自定义 arch 不行。
```

## 用 API(三种用法)

```c
// (1) DoD#1 — 直接喂一个 plan
llama_plan * p = llama_plan_load_json("plans/plan_4144MiB.json");
llama_elastic_apply_plan(ctx, p);            // decode 即按 plan 执行
// ... llama_decode ...
llama_plan_free(p);

// (2) DoD#2 — table provider,内存变化自动换
llama_elastic_enable(ctx, "table", "runtime/plan/plans");   // 之后每 decode 自动 snap 档
// ... llama_decode 循环,maybe_apply_plan 自动跑 ...

// (3) DoD#2 — callback provider,自己决定每个预算用哪个 plan
llama_elastic_set_plan_provider(ctx, my_fn, my_ud);
llama_elastic_enable(ctx, "callback", NULL);
```

## 设计要点 / 实测对齐

- **overlap = per-engine busy 的 max**(§5.5):timeline 事件带 `engine`,GPU convert 标 ENG_GPU
  (∥CPU compute/disk,不∥GPU matmul);CPU repack 标 ENG_CPU。
- **迁移**:GPU 算但 weight 不在 GPU → `migrate=true` + `xform=GPU_CONVERT`;loader 自动派生。
- **防抖**:online loop 用「指针相同 或 budget_mib 相同」判断同档,避免预算抖 1MB 重 apply。
- **STATIC 0 开销**:无 RUNTIME op 的 plan 不装 runtime dispatch hook,不付 per-op 单 op 代价。

## 已知问题 / 限制

- `test_weight_buffer_manager` ctest 失败 = **pre-existing**(继承自 `7dec93972` WBM pick_victim hook,MRU 排序),与本框架无关,我没动 WBM 文件。
- **桌面无 GPU**:residency 搬运(86 prefetch)落全局 registry no-op,GPU 路由 fallback CPU。
  真实 residency + 跨后端 routing/迁移/overlap **要 Android + ggml-opencl-elastic** 才生效。
- M5 的真迁移仍 env 门控(`GGML_SCHED_RUNTIME_DISPATCH_MIGRATE`);逐 op 按 `elastic_migrate`
  自动开 = 设备侧收尾(改 ggml-backend.cpp,留真机一起调,避免盲改)。

## 建议的下一步(设备侧,按优先级)

1. **M5 收尾**:让 ggml-backend per-op migration 读 `elastic_migrate`(取代 env),真机验证
   CPU↔GPU 迁移 + convert。
2. **M6 overlap**:executor 的 timeline anchor → 接 opencl WBM 异步 prefetch,按 §5.5 per-engine 编排。
3. **M7 实测**:`scripts/elastic/` trace 驱动 B(t),plan-driven(table provider)vs static baseline 出指标。
4. **make_plan.py 升级**:dump `timeline` + `xform` 进 native schema(目前 loader 从 schedule 派生,够用)。
