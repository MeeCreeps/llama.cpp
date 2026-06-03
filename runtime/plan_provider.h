// runtime/plan_provider.h
//
// PlanProvider —— 「给定当前预算 B → 返回该用的 ExecPlan」(见 IMPLEMENTATION.md §4)。
// Provider 是策略,Plan 是结果。本期两种:
//   * table    : 从 plans_dir 的 index.json 按 budget 选最近档(≤B 的最大档),
//                懒加载 make_plan.py 格式的 plan_*.json 并缓存。
//   * callback : 调用方注册 fn(budget)→ExecPlan,完全自定义「内存变化喂哪个 plan」。
//
// 关键契约:**同一预算档返回同一指针**。在线主循环靠指针比较判断「要不要换 plan」,
// 档不变则 0 开销;档变才触发 executor.apply + graph 重建(DoD#2)。

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "plan_ir.h"

namespace elastic {

class PlanProvider {
public:
    virtual ~PlanProvider() = default;

    // 返回该预算下应使用的 plan(provider 拥有其生命周期)。失败返回 nullptr。
    // 同档 → 同指针。
    virtual const ExecPlan * get(int64_t budget_mib,
                                 size_t kv_bytes = 0, size_t misc_bytes = 0) = 0;

    // 可用档数(table = index 条目数;callback = 已缓存数)
    virtual int n_bands() const = 0;

    // ── 工厂 ──
    // table:plans_dir 下需有 index.json(make_plan.py 产出)。
    static std::unique_ptr<PlanProvider> create_table(const std::string & plans_dir,
                                                       std::string * err = nullptr);
    // callback:fn 把 budget 填进 out(返回 true 成功)。provider 按 budget 缓存以保证指针稳定。
    static std::unique_ptr<PlanProvider> create_callback(
            std::function<bool(int64_t budget_mib, ExecPlan & out)> fn);

    // 通用入口(C-ABI 桥接用):kind = "table" / "callback"(callback 需随后 set)。
    static std::unique_ptr<PlanProvider> create(const std::string & kind,
                                                const std::string & plans_dir,
                                                std::string * err = nullptr);
};

}  // namespace elastic
