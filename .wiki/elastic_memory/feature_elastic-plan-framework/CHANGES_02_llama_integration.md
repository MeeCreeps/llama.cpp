# CHANGES 02 — llama-context 接线 + E2E (M3 DoD#1 + M4 DoD#2)

> 分支:`feature/elastic-plan-framework`
> 日期:2026-06-03
> 范围:把 Plan 框架接进 llama-context,加 C API,**桌面 Llama-3.2-3B 端到端跑通**。
> **两个 DoD 均已验证。**

## 做了什么

把 CHANGES_01 的纯逻辑核心(Plan IR / Executor / Provider)接进 libllama,实现用户定义的两个 DoD:
1. **给一个 plan 就能执行**(`llama_elastic_apply_plan`)
2. **内存变化每次给一个 plan 就能切**(`llama_elastic_enable` + provider + online loop)

### C API(include/llama.h 新增)

```c
struct llama_plan;  // opaque, 包装 elastic::ExecPlan
llama_plan * llama_plan_load_json(const char * path);     // 自动识别 native / make_plan 格式
void         llama_plan_free(llama_plan *);
int          llama_plan_save_json(const llama_plan *, const char * path);
int64_t      llama_plan_budget_mib / int llama_plan_n_weights / n_ops(const llama_plan *);

// DoD#1
int  llama_elastic_apply_plan(llama_context *, const llama_plan *);
// DoD#2
int  llama_elastic_enable(llama_context *, const char * kind/*"table"/"callback"*/, const char * plans_dir);
void llama_elastic_disable(llama_context *);
void llama_elastic_set_plan_provider(llama_context *, llama_plan_provider_fn, void * ud);
const llama_plan * llama_elastic_get_plan(llama_context *, int64_t budget_mib);
```

### llama-context 改动

- **src/llama-context.h**:新增 elastic 成员(executor / provider / route 表 / cpu_gpu id / provider fn / view 缓存)+ 公开方法(`apply_exec_plan` / `elastic_enable` / `elastic_disable` / `elastic_set_provider_fn` / `elastic_get_plan_view`)+ 内部 helper(`elastic_install_op_schedule` / `maybe_apply_plan` / `elastic_budget_mib`)。前向声明 `elastic::{ExecPlan,PlanExecutor,PlanProvider}`(避免 json 进 header)。
- **src/llama-context.cpp**:
  - `apply_exec_plan(plan)`:懒建 PlanExecutor + **ExecSinks 桥接到真实后端**:
    - `set_resident(wid,want)` → `llama_weight_movement_request(name, evict=!want)`(全局 registry)
    - `is_resident(wid)` → `llama_weight_residency_query(name)`
    - `set_op_backend(op_id,be)` → 填 `elastic_route[weight_name] = backend_id`
    - `set_op_migrate` → M5 占位(记录意图)
  - `elastic_install_op_schedule()`:装 op_schedule_fn,MUL_MAT 按 `node->src[0]` 的 weight 名查 `elastic_route`(与 tools/main 的 plan 路由同机制)。
  - `maybe_apply_plan()`:**online loop**,挂在 `decode()` 入口(紧跟 `maybe_run_scheduler`)。每 decode 前 sample 预算 → provider.get → 档变才 apply + `graph_invalidate`。**防抖**:指针相同 *或* `budget_mib` 相同 → 跳过重 apply。
  - C API wrappers + `struct llama_plan` 定义 + `llama_plan_load_json` 自动格式识别。
  - 析构函数释放 `elastic_plan_view_cache`。
- **src/CMakeLists.txt**:把 `runtime/plan_{ir,executor,provider}.cpp` 编进 libllama + 加 `../runtime` `../vendor` include。

## E2E 验证(桌面 CPU,Llama-3.2-3B)

`tests/elastic/test_plan_e2e.cpp`(手动编译,需 model + plan;非 ctest):

```bash
g++ -std=c++17 -O2 -I include -I ggml/include tests/elastic/test_plan_e2e.cpp \
    -L build-native/bin -lllama -lggml -lggml-base -Wl,-rpath,build-native/bin \
    -o build-native/test_plan_e2e
./build-native/test_plan_e2e <Llama-3.2-3B.gguf> runtime/plan/plans/plan_4144MiB.json
```

**结果(ALL PASS):**
```
plan: budget=4144MiB weights=197 ops=197
apply_exec_plan: applied plan budget=4144MiB weights=197 ops=197 prefetch=86 evict=0 route=197 migrate=0 events=305
decoded 8 continuation tokens (rc=0)               ← DoD#1: 喂 plan → decode 正确
elastic_enable: elastic enabled (provider=callback)
maybe_apply_plan: budget=11499MiB → switched to plan(budget_mib=4144)   ← DoD#2: online 换 plan
provider callback invoked 1 times
test_plan_e2e: ALL PASS
```

- **DoD#1 ✓**:`llama_elastic_apply_plan` 返回 0,plan 的 197 ops 路由 + 86 GPU-resident 意图全下发,decode 续 8 token 正确。
- **DoD#2 ✓**:`llama_elastic_enable("callback")` + online loop 每 decode 触发,预算变化切 plan,decode 继续不崩;防抖生效(同档不重 apply)。

## 说明 / 限制

- **桌面无 GPU/elastic backend**:`set_resident` 的 86 个 prefetch 落到全局 registry(无 OpenCL 时 no-op);op routing 因只有 CPU backend,GPU 路由 fallback CPU。**真实 residency 搬运 + 跨后端 routing 要在 Android + ggml-opencl-elastic 上才生效**(M5/M6 设备侧)。桌面 E2E 验证的是「**apply/换 plan 不崩 + decode 正确 + 框架接线通**」。
- `set_op_migrate` 目前是占位(M5 接 `ggml_backend_sched_set_runtime_dispatch` + migration pool)。
- `dllm` / `sdar` 等自定义 arch 模型 stock llama.cpp 不识别;E2E 用标准 llama arch 的 Llama-3.2-3B(`a.gguf`)。

## 编译验证

- `libllama.so` 全量编过(plan TUs 进库)。
- 独立 `elastic_runtime` 单测仍全过(CHANGES_01)。

## 下一步

- **M5**:`set_op_migrate` / runtime dispatch sink 接 ggml-backend per-op migration(设备侧 D2b 真迁移 + layout 转换)。
- **M6**:overlap 编排(timeline anchor → 异步 prefetch),接 opencl WBM。
