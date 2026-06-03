// runtime/plan_provider.cpp —— PlanProvider 实现(table / callback)。

#include "plan_provider.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace elastic {

// ───────────────────────── table provider ─────────────────────────

namespace {

struct Band {
    int64_t     budget_mib;
    std::string file;       // 相对 plans_dir 的文件名
};

class TableProvider : public PlanProvider {
public:
    TableProvider(std::string dir, std::vector<Band> bands)
        : dir_(std::move(dir)), bands_(std::move(bands)) {
        std::sort(bands_.begin(), bands_.end(),
                  [](const Band & a, const Band & b) { return a.budget_mib < b.budget_mib; });
    }

    const ExecPlan * get(int64_t budget_mib, size_t kv_bytes, size_t misc_bytes) override {
        if (bands_.empty()) return nullptr;

        // 选 ≤ budget 的最大档(保守:不能用超过预算的 plan)。比最小档还小 → 用最小档。
        const Band * pick = &bands_.front();
        for (const auto & b : bands_) {
            if (b.budget_mib <= budget_mib) pick = &b;
            else break;
        }

        // 按文件缓存:同档 → 同 ExecPlan 指针(在线循环靠这个判断换没换 plan)。
        auto it = cache_.find(pick->file);
        if (it == cache_.end()) {
            auto plan = std::make_unique<ExecPlan>();
            std::string err;
            std::string path = dir_ + "/" + pick->file;
            if (!plan_from_make_plan_file(path, *plan, &err)) {
                last_err_ = err;
                return nullptr;
            }
            // 补上 kv/misc(plan_*.json 不含;由调用方现场给)
            plan->kv_bytes   = kv_bytes;
            plan->misc_bytes = misc_bytes;
            it = cache_.emplace(pick->file, std::move(plan)).first;
        }
        return it->second.get();
    }

    int n_bands() const override { return (int) bands_.size(); }

    std::string last_err_;

private:
    std::string                                     dir_;
    std::vector<Band>                               bands_;
    std::map<std::string, std::unique_ptr<ExecPlan>> cache_;
};

}  // namespace

std::unique_ptr<PlanProvider> PlanProvider::create_table(const std::string & plans_dir,
                                                         std::string * err) {
    std::string idx_path = plans_dir + "/index.json";
    std::ifstream f(idx_path);
    if (!f) {
        if (err) *err = "open failed: " + idx_path;
        return nullptr;
    }
    json j;
    try {
        std::stringstream ss;
        ss << f.rdbuf();
        j = json::parse(ss.str());
    } catch (const std::exception & e) {
        if (err) *err = std::string("index.json parse: ") + e.what();
        return nullptr;
    }

    std::vector<Band> bands;
    try {
        for (const auto & e : j.value("index", json::array())) {
            Band b;
            b.budget_mib = (int64_t) llround(e.value("budget_mib", 0.0));
            b.file       = e.value("file", std::string());
            if (!b.file.empty()) bands.push_back(std::move(b));
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("index decode: ") + e.what();
        return nullptr;
    }
    if (bands.empty()) {
        if (err) *err = "index.json has no usable bands";
        return nullptr;
    }
    return std::make_unique<TableProvider>(plans_dir, std::move(bands));
}

// ───────────────────────── callback provider ─────────────────────────

namespace {

class CallbackProvider : public PlanProvider {
public:
    explicit CallbackProvider(std::function<bool(int64_t, ExecPlan &)> fn)
        : fn_(std::move(fn)) {}

    const ExecPlan * get(int64_t budget_mib, size_t kv_bytes, size_t misc_bytes) override {
        if (!fn_) return nullptr;
        // 按 budget 缓存以保证指针稳定(同 budget → 同 plan 指针)。
        auto it = cache_.find(budget_mib);
        if (it == cache_.end()) {
            auto plan = std::make_unique<ExecPlan>();
            if (!fn_(budget_mib, *plan)) return nullptr;
            plan->kv_bytes   = kv_bytes;
            plan->misc_bytes = misc_bytes;
            it = cache_.emplace(budget_mib, std::move(plan)).first;
        }
        return it->second.get();
    }

    int n_bands() const override { return (int) cache_.size(); }

private:
    std::function<bool(int64_t, ExecPlan &)>         fn_;
    std::map<int64_t, std::unique_ptr<ExecPlan>>     cache_;
};

}  // namespace

std::unique_ptr<PlanProvider> PlanProvider::create_callback(
        std::function<bool(int64_t, ExecPlan &)> fn) {
    return std::make_unique<CallbackProvider>(std::move(fn));
}

std::unique_ptr<PlanProvider> PlanProvider::create(const std::string & kind,
                                                   const std::string & plans_dir,
                                                   std::string * err) {
    if (kind == "table") return create_table(plans_dir, err);
    if (kind == "callback") {
        // callback 需调用方随后注入 fn;这里先给一个永远失败的占位。
        return create_callback(nullptr);
    }
    if (err) *err = "unknown provider kind: " + kind;
    return nullptr;
}

}  // namespace elastic
