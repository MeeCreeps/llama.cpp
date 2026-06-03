# CHANGES 04 — 真机验证 (OnePlus 12 / Adreno 750) + 3 个集成修复

> 分支:`feature/elastic-plan-framework`
> 日期:2026-06-03
> 设备:OnePlus 12 (CPH2583) / Snapdragon 8 Gen 3 (SM8650) / Adreno 750, 16GB UMA
> **两个 DoD 都在真机 GPU + ggml-opencl-elastic 上跑通。**

## 结果总览

| | 配置 | 结果 |
|---|---|---|
| **DoD#1** apply 一个 plan | F16 3B + `GGML_OPENCL_ELASTIC=1` + `LLAMA_ELASTIC_APPLY=plan_4144.json` | ✅ apply rc=0,**真 GPU residency evict 111**,输出正确:"The capital of France is Paris. The capital of Italy is Rome..." |
| **DoD#2** online 换 plan | F16 3B + budget trace + `LLAMA_ELASTIC_DIR=plans/` | ✅ **5 个 band 切换**(6448→5488→3952→5680→7408 MiB),无崩,输出正确,末档还触发 migrate=16 |

桌面与真机的关键差异:桌面无 GPU,residency sink 是 no-op(evict=0);**真机 `is_resident` 探针返回真实 GPU 状态,executor 算出真实差量**(DoD#1 evict 111;DoD#2 各档 evict 63/22/32...)。

## 新增的设备测试入口(tools/main/main.cpp)

```
LLAMA_ELASTIC_APPLY=<plan.json>  → llama_plan_load_json + llama_elastic_apply_plan (DoD#1)
LLAMA_ELASTIC_DIR=<plans_dir>    → llama_elastic_enable("table", dir)            (DoD#2)
```
与旧的 `LLAMA_PLAN_DIR` 手写 demo 互不影响(不同 env)。

## 3 个真机集成修复(从崩到跑通的过程)

真机上 DoD#2 一开始 `clSetKernelArg(... &extra0->data_device) error -38`(CL_INVALID_MEM_OBJECT)崩 ——
GPU kernel 用到已释放的 weight cl_mem。逐个定位 + 修:

### 修复 1:apply 时强制重建 graph(src/llama-context.cpp)
`apply_exec_plan` 原来只 `graph_invalidate()`,没置 `graph_reuse_disable`。routing 一变,
复用的 cached graph 仍引用旧 split / 已 evict 的 cl_mem。**与 `set_op_schedule` 同策略**:
apply 时 `graph_reuse_disable=true`(除非 `LLAMA_KEEP_GRAPH_REUSE`)。

### 修复 2:`GGML_ELASTIC_NO_AUTO_EVICT` —— 让 plan 做唯一 residency 权威(ggml-opencl.cpp)
根因之一是**两个驱逐者打架**:plan framework 按 band 决定 residency,而 backend 自己又在
graph_compute 里按 `static_target`(= trace 最小值)`wbm_evict_to_byte_budget` 自动 evict。
backend 把 plan 想留 GPU 的 weight evict 掉 → GPU kernel 崩。新增 env 关掉 backend 的预算驱逐
(保留 reload-on-use),plan 成为唯一权威。两处 evict 站点(预 evict + 周期 evict)都加门控。

### 修复 3:plan 只主动 EVICT,reload 交给 backend(src/llama-context.cpp)— **关键**
即使 plan 独占权威仍崩。根因:**外部 prefetch 只换了 WBM 的 cl_mem,没同步 `tensor->extra->data_device`**
(kernel 真正读的指针)。backend 自己的 `ensure_resident`(graph_compute 里 reload-on-use)
**会正确同步 tensor->extra**,外部 `movement_request(prefetch)` 不会。
→ 改成:**plan 的 set_resident sink 只做 evict;want=true 不主动 prefetch,靠 backend
ensure_resident 在用到时按需 reload**(它会 alloc 新 cl_mem 并重跑 convert + 同步 extra)。
诊断用 `LLAMA_ELASTIC_ACTIVE_PREFETCH=1` 可强制回到主动 prefetch(会复现崩)。

**集成范式(重要结论)**:
> **plan 控制「踢谁」(budget 决策),backend 控制「怎么搬回来」(它知道怎么同步 cl_mem ↔ tensor->extra)。**
> 二者职责分离后,DoD#2 在真机 5 次 band 切换全程稳定。

## 复现命令

```bash
# build (NDK)
ANDROID_NDK=/home/hz85760/android-ndk-r28b bash scripts/elastic/build_llama_android.sh Release
adb push build-android-llama/bin/llama-cli /data/local/tmp/elastic/llama-cli-planfw

# DoD#1
adb shell 'cd /data/local/tmp/elastic && LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
  GGML_OPENCL_ELASTIC=1 LLAMA_ELASTIC_APPLY=plans/plan_4144MiB.json \
  ./llama-cli-planfw -m Llama-3.2-3B-Instruct-f16.gguf -p "The capital of France is" -n 12 --temp 0 -ngl 99 -no-cnv'

# DoD#2 (online, budget trace 驱动换 band)
adb shell 'cd /data/local/tmp/elastic && LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
  GGML_OPENCL_ELASTIC=1 GGML_ELASTIC_NO_AUTO_EVICT=1 GGML_ELASTIC_BUDGET_CSV=trace_dod2.csv \
  LLAMA_ELASTIC_DIR=plans/plans \
  ./llama-cli-planfw -m Llama-3.2-3B-Instruct-f16.gguf -p "The capital of France is" -n 14 --temp 0 -ngl 99 -no-cnv'
```

## 改动文件

- `tools/main/main.cpp` — LLAMA_ELASTIC_APPLY / LLAMA_ELASTIC_DIR 测试入口
- `src/llama-context.cpp` — 修复1(graph_reuse_disable)+ 修复3(evict-only sink + ACTIVE_PREFETCH 诊断)
- `ggml/src/ggml-opencl/ggml-opencl.cpp` — 修复2(GGML_ELASTIC_NO_AUTO_EVICT 门控两处 auto-evict)

## 性能备注(非优化目标)

F16 3B 流式:DoD#2 eval ~2.2s/token(各 band 平均,含 reload)。F16 per-token 流式本就慢
(见 project_q4_elastic_findings);本期目标是**框架正确性 + 真机跑通**,非速度。Q4 模型 +
plans_q4 会快很多,后续实测(M7)用。

## 待深化(可选)

- **M5 真迁移**:末档已出现 migrate=16(GPU-streamed weights);runtime dispatch + 迁移在真机
  的端到端正确性还需专门验证(本次 DoD#2 主要验 residency + routing)。
- **M6 overlap**:timeline anchor → 异步 prefetch 编排(per-engine busy,§5.5),设备侧。
- **prefetch 同步根治**:让外部 prefetch 也能同步 tensor->extra(则可主动 prefetch,
  reload 时机更可控);当前 evict-only + backend-reload 已足够正确。
