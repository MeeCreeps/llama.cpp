# CHANGES 03 — D2b runtime dispatch + 迁移意图接线 (M5)

> 分支:`feature/elastic-plan-framework`
> 日期:2026-06-03
> 范围:plan 驱动的 per-op **runtime dispatch** 接线 + 跨后端迁移意图记录。
> 编译通过 + 桌面验证 hook 安装路径;**实际 CPU↔GPU 迁移是设备侧(需 GPU)**。

## 做了什么

把 plan 里 `dispatch == RUNTIME` 的 op 接到已有的 per-op runtime dispatch 通路
(`ggml_backend_sched_set_runtime_dispatch`),让这些 op 在 **compute 即将开始前**按 plan
决定 backend,并记录跨后端迁移意图(供设备侧 migration pool 做 CPU↔GPU + layout 转换)。

### 改动(src/llama-context.{h,cpp})

- 新增成员:
  - `elastic_runtime_route`(weight 名 → backend_id,RUNTIME op 用)
  - `elastic_migrate`(weight 名 → (migrate_from_backend, xform),迁移意图)
- `set_op_migrate` sink:不再是 no-op,**记录迁移意图**进 `elastic_migrate`。
- `apply_exec_plan`:apply 后收集 RUNTIME op 的 backend 路由;调 `elastic_install_runtime_dispatch`。
- 新增 `elastic_install_runtime_dispatch()`:
  - 有 RUNTIME op → 装一个 op_runtime_dispatch hook(按 `op->src[0]` weight 名查
    `elastic_runtime_route`,返回 backend;-1 = split 默认)。复用已有 `set_op_runtime_dispatch`
    → `ggml_backend_sched_set_runtime_dispatch` 通路。
  - 无 RUNTIME op → 卸掉 hook(STATIC plan **0 额外开销**,不付 per-op 单 op 执行代价)。

### 两条 routing 通路的分工(D2b 两面)

| | 何时 | 通路 | plan 字段 |
|---|---|---|---|
| **static** | graph-build | `op_schedule_fn` → `elastic_route` | `dispatch=STATIC` |
| **runtime** | 每个 op compute 前 | `op_runtime_dispatch` hook → `elastic_runtime_route` | `dispatch=RUNTIME` |

STATIC 复用 graph(便宜);RUNTIME 可按现场状态改 + 触发迁移(贵,plan 只标少量 op)。

## 验证(桌面 Llama-3.2-3B)

构造一个把 6 个 CPU op 标成 RUNTIME 的 native plan,apply + decode:
```
elastic_install_runtime_dispatch: installed runtime dispatch for 6 op(s), 0 migrate
apply_exec_plan: applied plan budget=4144MiB weights=197 ops=197 route=191 ...   (197-6=191 static)
decoded 8 continuation tokens (rc=0)
test_plan_e2e: ALL PASS
```
- RUNTIME hook **正确安装**(6 op),static route 相应减到 191。
- decode 经过 runtime dispatch hook **仍产出正确 token**(桌面 CPU-only:hook 返回 cpu_id,无迁移)。
- 单测 `test_plan_executor` 的 `runtime_dispatch(op_id, default)` 逻辑早已覆盖(返回 plan backend / default 透传)。

## 设备侧待完成(M5 的真迁移 + M6)

- **真 CPU↔GPU 迁移**:`elastic_migrate` 已记录意图;实际迁移由 ggml-backend 的
  migration pool(`GGML_SCHED_RUNTIME_DISPATCH_MIGRATE` 通路)在 target backend ≠ split
  backend 时执行(含 OpenCL set_tensor 触发的 q4_0→SOA+transpose)。桌面无 GPU → 验证不了,
  需 Android + ggml-opencl-elastic。当前迁移**仍受 env 门控**;让它逐 op 按 `elastic_migrate`
  自动开是 M5 的设备侧收尾(改 ggml-backend.cpp,风险较高,留待真机一起调)。
- **M6 overlap 编排**:timeline 的 `anchor_op_id` → 异步 prefetch 挂到 op compute 窗口,
  接 opencl WBM。per-engine busy 模型见 IMPLEMENTATION.md §5.5。设备侧。

## 安全性

- 改动**纯加法**:无 RUNTIME op 的 plan(现有 make_plan 产出全 STATIC)走原路径,
  E2E 回归确认无影响(decode 正确,online loop 正常)。
