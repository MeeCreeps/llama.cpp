# 统一 Scheduler 抽象 — 设计提案

> 状态:**设计提案,未实现 —— 目的是搭框架,不是优化**。 目标分支 `feature/elastic-op-sched`。

## 设计意图 (先读这个)

**scheduler = "当内存预算变化时,决定哪些 weight evict、哪些保留 (以及 op 放哪个 backend、怎么 prefetch) 的统一决策者"。**

核心是把"内存变化 → weight 去留 (+ op 路由 + prefetch) 的决策"抽象成一个**可替换的 scheduler 接口**:

- **MRU 是这个接口的最基础 baseline 实现** —— 内存紧了就按 round-robin 踢最近用的,够用但不聪明。
- **之后插自定义 scheduler**:比如把离线 LP plan (见 `runtime/lp/`) 算出的最优调度查表喂进来,或任何启发式策略。
- **现在这一步只搭骨架** —— 定义接口、把 MRU 收编成第一个 baseline、留好扩展点。 不追求性能,追求"以后换 scheduler 不用改 backend"。

所以下面所有内容的目的不是"修补现状缺陷",而是**为可扩展性搭一个干净的 scheduler 框架,让 MRU / LP plan / 未来策略都只是它的不同实现**。

---

## 1. 动机

当前 elastic 调度的决策分散在 5 个互不相关的接口里,各管一摊:

| 决策 | 现状接口 | 形态 |
|---|---|---|
| op 放哪个 backend | `llama_set_op_schedule` | ✅ callback |
| 哪些 weight 留 (pin 永驻) | `llama_set_weight_pin` | ✅ callback, 默认不 pin |
| 踢哪个 weight (指名) | `llama_weight_request_evict(name)` / `_prefetch(name)` | ✅ 能指名踢/取任意 weight (含 pin 的) |
| 超预算时自动选 victim | `evict_mru` (bool, MRU/LRU 二选一) | ⚠️ **自动路径的排序算法写死**, 默认 MRU |
| 留多少 (预算阈值) | static / `GGML_ELASTIC_DYNAMIC` | env 开关 |
| prefetch 什么 | `GGML_ELASTIC_PREFETCH` (自动) + `llama_weight_request_prefetch` (手动) | env + 分散 API |
| 内存变化时做什么 | `llama_set_scheduler` (通知 callback) | callback, 但只"通知"不结构化 |

注意 residency 控制其实**已经相当完整**,有两条驱逐路径:
- **主动路径**:`request_evict(name)` / `request_prefetch(name)` 能在 scheduler 里**指名搬运任意 weight** (代码里直接 `wbmcl_evict_batch(idx)`,连 pin 的都能踢) → "踢哪个" **已可完全自定义**;
- **自动路径**:graph_compute 超预算时 WBM 自己在非 pin 候选里选 victim → 这一步的**排序算法**写死 MRU/LRU 二选一,没暴露。

所以"踢哪个"不是不能自定义 —— 用 `request_evict` 已经能。 真正的缺口是**自动兜底路径的 victim 排序没法插拔**:要自定义它,用户只能在 `on_step` 里手动枚举一堆 `request_evict`,繁琐且时机不对 (on_step 每 decode 一次,驱逐却发生在 graph_compute 内)。

**问题**:决策分散在 5-6 个 API,而且 weight residency 一半在 `weight_pin`、一半在 `request_*`、自动 victim 又写死;没有"一个对象 = 一个完整策略"的抽象,也没法优雅地自定义自动驱逐顺序。

---

## 2. 核心模型:scheduler 决定 4 类决策

把 scheduler 定义成一个**统一决策者**,负责四个维度:

```
scheduler {
   1. budget_target(B_t)   → 这一刻 GPU 上留多少字节 weight    (核心: 留多少 / budget 策略)
   2. pick_victim(cands)   → 要腾空间时, 踢哪个 weight (怎么移动)  (residency: 踢谁)
   3. route_op(op)         → 这个 op 在哪个 backend 算          (空间: CPU/GPU, 可选)
   4. on_step(state)       → 内存变化时预取/驱逐/重路由什么      (时间: 响应 + 预取, 可选)
}
```

**关键区分:`static` 和 `dynamic` 是两种不同 scheduler,区别在维度 1 (budget 策略);MRU 是维度 2 (怎么移动 weight) 的底层方式,被它们共用。**

- **`static` scheduler**:`budget_target` **恒定 = min(B(t)) - kv - misc**,完全不随 trace 变 —— 永远按整条 trace 的最低预算保守配置。 用 MRU 把驻留维持在这个固定上限内。 最保守的 baseline,不响应内存变化。
- **`dynamic` scheduler**:`budget_target` **= 当前 B(t) - kv - misc**,实时跟随内存变化 —— 内存宽裕时多留 (少 reload),紧时多踢。 **同样用 MRU 移动 weight**,只是 target 是动的。
- **`mru`/`lru`**:严格说是维度 2 (victim 选择) 的策略名,不是完整 scheduler。 最朴素的整体 baseline = `static` budget + MRU 移动 + 全 GPU + 无预取。

不同 scheduler = 这四个维度的不同组合:

| scheduler | budget (留多少) | 移动方式 (踢谁) | route_op (放哪) | on_step (预取/响应) |
|---|---|---|---|---|
| **`static`** (最保守 baseline) | **恒定 min(B(t))** | MRU | 全 GPU | 无 |
| **`dynamic`** | **随 B(t) 实时** | MRU | 全 GPU | 可选 lookahead 预取 |
| `static-partition` | 恒定 min | MRU | 前 K 层 GPU,其余 CPU | 无 |
| `smart-pressure` | 随 B(t) | MRU | 内存紧时 ffn 切 CPU | 看压力预取/驱逐 |
| `lp-plan` (未来) | LP 算 | LP 算的淘汰顺序 | LP 算的 op 分配 | LP 算的预取时刻 |

这样:
- **"static = budget 不变" 和 "dynamic = budget 随 trace 变" 都是一等 scheduler**,只是维度 1 不同;
- **MRU 是它们复用的移动机制**,未来 `lp-plan` 可以连移动顺序都自己定 (不用 MRU);
- 现在代码里的 `GGML_ELASTIC_DYNAMIC` env (static vs dynamic) 本质就是在选维度 1 —— 框架化后它变成"选哪个 scheduler"。

---

## 3. 接口设计

### 3.1 scheduler 对象 (一组 hook + user_data)

```c
// 候选 weight 信息 (pick_victim 看这个排序)
struct llama_block_info {
    const char * name;          // tensor 名
    int          layer;         // 层号 (-1 = 非分层)
    size_t       byte_size;
    uint64_t     last_used_step; // 上次使用的 decode step (LRU/MRU 用)
    bool         is_pinned;
};

// scheduler 一步里可以 emit 的动作 (on_step 往里塞)
struct llama_sched_actions;     // opaque, 通过下面 API 追加
void llama_sched_action_prefetch(llama_sched_actions*, const char* name);
void llama_sched_action_evict   (llama_sched_actions*, const char* name);
void llama_sched_action_repartition(llama_sched_actions*, int new_gpu_layers); // 触发重 build graph

// 统一 scheduler
struct llama_scheduler {
    // (1) 留多少: 给定当前预算 B(t) MB, 返回 GPU 上 weight 字节上限 (target)。
    //     static = 忽略 b_t_mb, 恒返 min(B(t)) 算出的固定值;
    //     dynamic = 用当前 b_t_mb 实时算。 NULL = 用内置 static。
    size_t (*budget_target)(int64_t b_t_mb, size_t kv_bytes, size_t misc_bytes, void* ud);

    // (2) 怎么移动 (踢谁): 要驱逐到 target 时调, 从候选里选一个踢, 返回下标 (-1=不踢)。
    //     反复调直到 resident_bytes <= target。 MRU/LRU 是内置实现, 各 scheduler 共用。
    int  (*pick_victim)(const struct llama_block_info* cands, int n_cands,
                        size_t need_free_bytes, void* ud);

    // (3) op 放哪: graph build 时每个 op 调一次; 返回 backend_id (-1 = 用默认)。可选。
    int  (*route_op)(const struct ggml_tensor* op, int default_backend, int n_backends, void* ud);

    // (4) 内存变化时: 每 decode (或预算变化超阈值) 调; 产出预取/驱逐/重路由动作。可选。
    void (*on_step)(const struct llama_runtime_state* st,
                    struct llama_sched_actions* out, void* ud);

    void* user_data;
};

void llama_set_scheduler_v2(struct llama_context*, const struct llama_scheduler*);

// builtin: 不想自己写就用现成的
const struct llama_scheduler* llama_scheduler_builtin(const char* name); // "static"/"dynamic"/"smart-pressure"/...
```

### 3.2 每个 hook 可以单独留空 (留空 = 内置默认)

- `budget_target == NULL` → 内置 **static** (恒定 min(B(t)))
- `pick_victim == NULL` → 内置 **MRU**
- `route_op == NULL` → 所有 op 默认 backend (= 全 GPU)
- `on_step == NULL` → 不预取、不主动驱逐

所以两个 baseline 就是不同 hook 的组合:
- **`static`** = 四个 hook 全空 (budget 默认 static + MRU + 全 GPU + 无预取) —— 最保守。
- **`dynamic`** = 只把 `budget_target` 换成"用当前 B(t) 实时算",其余留空 (仍 MRU 移动)。

用户要做自定义 scheduler (如 LP plan),就实现需要的 hook,不碰 backend。

---

## 4. 四个决策点的数据流 (何时、在哪调)

```
模型 load
  └─ (graph build, 每次重建)
        每个 op → route_op(op)            ── 决定 op_backend, 交 ggml-sched split
                                              [hook 3, 在 llama-context graph 构建路径]

llama_decode(token)
  ├─ on_step(state, actions)              ── 内存变化时产出动作
  │     执行 actions: prefetch/evict 指定 weight, repartition→标记重 build
  │                                          [hook 4, 在 maybe_run_scheduler]
  └─ graph_compute (每个 backend)
        target = budget_target(B(t), kv, misc)   ── 这一刻留多少 (static 恒定 / dynamic 跟 B(t))
                                              [hook 1, 替换现 static_target/dynamic 分支]
        resident_bytes > target ?
          └─ 反复 pick_victim(cands, ...) ── 选 victim 踢到 target 内 (怎么移动)
                                              [hook 2, 在 WBM evict_to_byte_budget]
        每个 op 的 src 不在 → ensure_resident (自动 reload)
```

- **hook 1 (budget_target)**:graph_compute 算驱逐目标时,替换现在 `static_target_bytes` vs `GGML_ELASTIC_DYNAMIC` 的二分支 —— static/dynamic scheduler 的区别就落在这。
- **hook 2 (pick_victim)**:超预算时调,频率 = 每次驱逐。 是当前写死 MRU 的地方,改成调 hook (各 scheduler 共用 MRU 默认)。
- **hook 3 (route_op)**:graph 构建时,频率 = 每次重建 graph。 静态决策。 (本分支只 graph-build 时;per-op runtime 版在 op-sched-runtime。)
- **hook 4 (on_step)**:每 decode 入口 (现 `maybe_run_scheduler`),内存信号来自统一的 BudgetWatcher (见 `feature/elastic-op-sched` 的 budget provider)。

---

## 5. 跟现有 API 的迁移 / 兼容

统一抽象**不删旧 API**,而是把它们变成统一 scheduler 的内部实现 / 薄封装:

| 旧 | 新 | 关系 |
|---|---|---|
| `GGML_ELASTIC_DYNAMIC` (static vs dynamic) | `scheduler.budget_target` (static/dynamic builtin) | **env 二分支 → 选哪个 scheduler**;static/dynamic 各成一个 builtin |
| `evict_mru` bool | `scheduler.pick_victim` 的内置 mru/lru 实现 | victim 排序: 写死二选一 → hook (默认 mru),各 scheduler 共用 |
| `llama_set_op_schedule(fn)` | `scheduler.route_op` | 旧 API = 只设 route_op 的便捷封装 |
| `llama_set_weight_pin(fn)` | `pick_victim` 候选里的 `is_pinned` 标志 | pin 的 weight 不进 victim 候选; 旧 callback 仍可用 |
| `llama_set_scheduler(fn)` (通知式) | `scheduler.on_step` (动作式) | 升级:从"通知"变"返回动作" |
| `request_prefetch/evict(name)` | `on_step` 里 emit action | 仍可手动调,也能从 on_step 统一产出 |

**向后兼容**:`llama_scheduler_builtin("static")` 等价于现在不开 `GGML_ELASTIC_DYNAMIC` 的默认行为 (budget 恒定 min + MRU + 全 GPU + 无预取);`"dynamic"` 等价于开了 `GGML_ELASTIC_DYNAMIC`。 老用户不动代码,行为不变。

---

## 6. 落地分步

1. **scheduler 对象骨架 + `budget_target` + `pick_victim` 两个 hook (核心)**:定义 `llama_scheduler` struct;把现在的 `static_target` vs `GGML_ELASTIC_DYNAMIC` 二分支重构成 `budget_target` hook 的 `static`/`dynamic` 两个内置实现;把写死的 MRU/LRU 重构成 `pick_victim` 内置实现。 这一步就能让 `static` / `dynamic` 成为两个一等 scheduler。
2. **builtin 注册 + 选择入口**:`llama_scheduler_builtin("static"/"dynamic"/...)` + `llama_set_scheduler_v2`;`GGML_ELASTIC_DYNAMIC` env 改成"选 builtin"的薄封装。
3. **`route_op` 归位**:把现有 `llama_set_op_schedule` 接进 scheduler 的 `route_op` 维度。
4. **`on_step` + actions**:把 `maybe_run_scheduler` 的通知 callback 升级成产出 `llama_sched_actions`,运行时执行 (prefetch/evict/repartition)。
5. **更多 builtin + 自定义**:`static-partition` / `smart-pressure` 收编现有 demo policy;`lp-plan` 接 `runtime/lp/` 离线 plan。
6. **旧 API 改薄封装**,保持兼容。

---

## 7. 关键设计点与风险

- **pick_victim 调用频率**:驱逐时反复调,每次给候选数组。 候选可能上百,排序逻辑要轻 (或一次返回完整 victim 列表而非逐个,减少调用次数 —— 可改成 `pick_victims(cands, n, need_bytes, out_indices)` 批量版)。
- **跨 backend 一致性**:opencl 和 cpu-elastic 两条 evict 路径都要改成调 hook,行为一致。
- **graph-build (route_op) vs runtime**:本分支 route_op 是 graph 构建时静态决策;真正的 per-op runtime dispatch 在下游 `op-sched-runtime`。 统一 scheduler 先做静态版,runtime 版作为 `route_op` 的另一种落地后续接。
- **repartition action 要重建 graph**:`on_step` emit repartition 后,得标记 graph_reuse_disable 让下次 decode 重 build (用新 route_op 结果)。 已有 graph_reuse_disable 机制可复用。
- **内存信号源**:`on_step` 的 `state.mem_avail_mb` 已统一走 BudgetWatcher (见 budget provider 改动),scheduler 看到的预算跟 evict target 同源、可重现。
- **LP plan 作为 scheduler**:`lp-plan` builtin 可以把 `runtime/lp/` 算出的离线 plan 加载进来,budget_target/pick_victim/route_op/on_step 全按 plan 查表 —— 这正是 Schedule API 当初的设计目标 (接 LP solver)。

---

## 8. 一句话总结

把"调度"抽象成**一个 scheduler 对象,决定 {留多少 weight, 踢哪个 (怎么移动), op 放哪, 怎么预取}**:

- **`static`** (budget 恒定锁 min(B(t)) + MRU 移动) 和 **`dynamic`** (budget 随 B(t) 实时变 + MRU 移动) 是两个最基础 baseline scheduler,区别只在"留多少"那一维;
- **MRU 是它们共用的"怎么移动 weight"机制**,不是独立 scheduler;
- 现在搭的是**框架/骨架** —— 把 static/dynamic 收编成 builtin、留好 hook,**之后能无痛插自定义 scheduler (离线 LP plan 等),不碰 backend**。

要省事用 builtin (`static`/`dynamic`),要自定义就实现需要的 hook,要最优就挂 LP plan。
