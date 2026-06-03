# CHANGES 01 — runtime core (M0–M2 + provider)

> 分支:`feature/elastic-plan-framework`(从 `feature/elastic-op-sched-runtime` 切出)
> 日期:2026-06-03
> 范围:Plan IR + PlanExecutor + PlanProvider 三个纯逻辑模块 + 单测。**全部桌面编译通过 + 测试通过。**
> 对应 IMPLEMENTATION.md 的 M0 / M1 / M2 / M4-provider。

## 做了什么

实现了 elastic 执行框架的**纯逻辑核心三件套**(不依赖 llama/ggml,桌面 `cmake -S runtime` 独立可编可测)。这是 DoD 的地基:Plan 当输入、Executor 当主角。

### 新增文件

| 文件 | 作用 |
|---|---|
| `runtime/plan_ir.{h,cpp}` | **Plan IR**:`ExecPlan` = WeightPlan[](weight 在哪+变换)+ OpPlan[](谁算+迁移)+ PlanEvent[](搬运时间线+引擎归属)。native JSON 序列化 + **加载现有 make_plan.py 格式** plan_*.json。 |
| `runtime/plan_executor.{h,cpp}` | **PlanExecutor**:`apply(plan)` = residency reconcile(D1)+ static routing(D2b)+ migration 下发(D2b)+ timeline anchor 索引(D3)。通过 `ExecSinks`(std::function 回调)落到真实后端,测试喂 mock。 |
| `runtime/plan_provider.{h,cpp}` | **PlanProvider**:`get(budget)→ExecPlan`。`table`(查 index.json 选 ≤B 最大档 + 懒加载缓存 + **同档同指针**)+ `callback`(调用方自定义 budget→plan)。 |
| `tests/elastic/test_plan_ir.cpp` | native round-trip + 加载真实 plan_4144MiB.json + 校验 op/weight/timeline 合法性。 |
| `tests/elastic/test_plan_executor.cpp` | fresh apply / **replan 差量 reconcile** / runtime dispatch / anchor 索引。 |
| `tests/elastic/test_plan_provider.cpp` | table 选档 + 指针稳定 + callback 缓存。 |

### 改动文件

- `runtime/CMakeLists.txt`:加 3 个新 TU + vendor/nlohmann json include(header-only)。
- `tests/elastic/CMakeLists.txt`:加 3 个新 test target,传 `runtime/plan/plans` 给 table/loader 测试。

## 关键设计落点(与 IMPLEMENTATION.md 对齐)

1. **Plan IR 是 backend-无关纯数据**:枚举 `Location{GPU,CPU,DISK}` / `Backend{CPU,GPU,NPU}` / `Engine{CPU,GPU,DISK,DMA}` / `Xform{NONE,CPU_REPACK,GPU_CONVERT}` / `Dispatch{STATIC,RUNTIME}`。无任何 cl_mem/指针。
2. **make_plan loader 真能吃现有 plan**:把 `routes`(per-tensor backend)→ OpPlan、`resident_in_memory`(常驻集合)→ WeightPlan.location、`schedule`(per-step disk_in/dma_to_gpu/evict_out)→ timeline(带 engine 归属:disk_in→ENG_DISK、dma_to_gpu→ENG_DMA+伴随 ENG_GPU 的 XFORM convert、evict_out→EVICT)。
   - 实测加载 `plan_4144MiB.json`:**197 weights / 197 ops / 305 timeline events / 86 resident-GPU**,per_token=1492ms,bottleneck=disk。
   - 自动派生:GPU 算但 weight 不在 GPU 的 op → `migrate=true` + `migrate_xform=GPU_CONVERT`;weight.xform 按其 op compute backend(GPU→GPU_CONVERT,CPU→NONE 走 generic 退路)。
3. **Executor 差量 reconcile**:`apply` 只对「当前 GPU 驻留 ≠ plan 期望」的 weight 发 set_resident,plan 切换时正确算出 evict/prefetch 差集(单测验证:{0,1,2,3}→{2,3,4,5} = evict 2 + prefetch 2 + already 2)。
4. **engine 归属进 timeline**:`PlanEvent.engine` 决定该段进哪条引擎 busy(overlap = per-engine busy 的 max,§5.5);GPU convert 标 ENG_GPU。
5. **Provider 同档同指针**:在线主循环靠指针比较判断换没换 plan(table 按文件缓存,callback 按 budget 缓存)。

## 测试结果

```
test_plan_ir ......... ALL PASS   (loaded 197w/197op/305ev real plan)
test_plan_executor ... ALL PASS
test_plan_provider ... ALL PASS   (29 bands)
```

## 已知问题(非本次引入)

- `test_weight_buffer_manager` 在 ctest 里失败(`最旧应是 3 got=2 want=3`,MRU victim 排序)。**这是分支继承的 pre-existing 失败**,引入自 `7dec93972`(WBM pick_victim hook),与本框架无关 —— 我没动 `weight_buffer_manager.{h,cpp}` 及其测试。建议单独排查。

## 下一步

- **M3(DoD#1)**:接线 llama-context,`llama_elastic_apply_plan(ctx, plan)`,executor sinks 桥接真实 WBM + op_schedule,桌面 cpu-elastic 按固定 plan 跑通。
- **M4(DoD#2)**:online loop + provider 接线,trace 驱动 B(t) 换 plan。
