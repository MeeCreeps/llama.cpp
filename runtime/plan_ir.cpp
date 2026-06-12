// runtime/plan_ir.cpp —— Plan IR 实现 + JSON 序列化/加载。
//
// 用 vendor/nlohmann/json.hpp(header-only)。native schema 用字符串枚举,可读。
// make_plan loader 把现有 plan_*.json 的 routes/placement/schedule/resident_in_memory
// 映射成 ExecPlan(见 IMPLEMENTATION.md §3.3)。

#include "plan_ir.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace elastic {

// ───────────────────────── 枚举 ⇄ 字符串 ─────────────────────────

const char * to_string(Location l) {
    switch (l) {
        case Location::GPU:  return "gpu";
        case Location::CPU:  return "cpu";
        case Location::DISK: return "disk";
    }
    return "?";
}
const char * to_string(Backend b) {
    switch (b) {
        case Backend::CPU: return "cpu";
        case Backend::GPU: return "gpu";
        case Backend::NPU: return "npu";
    }
    return "?";
}
const char * to_string(Engine e) {
    switch (e) {
        case Engine::CPU:  return "cpu";
        case Engine::GPU:  return "gpu";
        case Engine::DISK: return "disk";
        case Engine::DMA:  return "dma";
    }
    return "?";
}
const char * to_string(Xform x) {
    switch (x) {
        case Xform::NONE:        return "none";
        case Xform::CPU_REPACK:  return "cpu_repack";
        case Xform::GPU_CONVERT: return "gpu_convert";
    }
    return "?";
}
const char * to_string(Dispatch d) {
    switch (d) {
        case Dispatch::STATIC:  return "static";
        case Dispatch::RUNTIME: return "runtime";
    }
    return "?";
}
const char * to_string(EvKind k) {
    switch (k) {
        case EvKind::LOAD:     return "load";
        case EvKind::PREFETCH: return "prefetch";
        case EvKind::EVICT:    return "evict";
        case EvKind::DMA:      return "dma";
        case EvKind::XFORM:    return "xform";
    }
    return "?";
}

Backend backend_of_location(Location l) {
    switch (l) {
        case Location::GPU:  return Backend::GPU;
        case Location::CPU:  return Backend::CPU;
        case Location::DISK: return Backend::CPU;  // reload 后落 CPU RAM
    }
    return Backend::CPU;
}

namespace {

Location location_from_string(const std::string & s) {
    if (s == "gpu")  return Location::GPU;
    if (s == "cpu")  return Location::CPU;
    return Location::DISK;
}
Backend backend_from_string(const std::string & s) {
    if (s == "gpu") return Backend::GPU;
    if (s == "npu") return Backend::NPU;
    return Backend::CPU;
}
Engine engine_from_string(const std::string & s) {
    if (s == "gpu")  return Engine::GPU;
    if (s == "disk") return Engine::DISK;
    if (s == "dma")  return Engine::DMA;
    return Engine::CPU;
}
Xform xform_from_string(const std::string & s) {
    if (s == "cpu_repack")  return Xform::CPU_REPACK;
    if (s == "gpu_convert") return Xform::GPU_CONVERT;
    return Xform::NONE;
}
Dispatch dispatch_from_string(const std::string & s) {
    return s == "runtime" ? Dispatch::RUNTIME : Dispatch::STATIC;
}
EvKind evkind_from_string(const std::string & s) {
    if (s == "load")  return EvKind::LOAD;
    if (s == "evict") return EvKind::EVICT;
    if (s == "dma")   return EvKind::DMA;
    if (s == "xform") return EvKind::XFORM;
    return EvKind::PREFETCH;
}

// 从 "blk.5.attn_q.weight" 解析层号;非分层返回 -1。
int parse_layer(const std::string & name) {
    auto p = name.find("blk.");
    if (p == std::string::npos) return -1;
    p += 4;
    int v = 0;
    bool any = false;
    while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
        v = v * 10 + (name[p] - '0');
        any = true;
        ++p;
    }
    return any ? v : -1;
}

}  // namespace

// ───────────────────────── 查询辅助 ─────────────────────────

const WeightPlan * ExecPlan::weight_by_id(int id) const {
    if (id < 0 || id >= (int) weights.size()) return nullptr;
    return &weights[id];
}
const WeightPlan * ExecPlan::weight_by_name(const std::string & name) const {
    for (const auto & w : weights) if (w.name == name) return &w;
    return nullptr;
}
int ExecPlan::weight_id_of(const std::string & name) const {
    for (const auto & w : weights) if (w.name == name) return w.weight_id;
    return -1;
}
const OpPlan * ExecPlan::op_by_id(int id) const {
    if (id < 0 || id >= (int) ops.size()) return nullptr;
    return &ops[id];
}
std::vector<int> ExecPlan::resident_gpu_ids() const {
    std::vector<int> out;
    for (const auto & w : weights) if (w.location == Location::GPU) out.push_back(w.weight_id);
    return out;
}

// ───────────────────────── native schema 序列化 ─────────────────────────

bool plan_to_json_string(const ExecPlan & plan, std::string & out) {
    json j;
    j["schema_version"]    = plan.schema_version;
    j["budget_mib"]        = plan.budget_mib;
    j["kv_bytes"]          = plan.kv_bytes;
    j["misc_bytes"]        = plan.misc_bytes;
    j["pred_per_token_ms"] = plan.pred_per_token_ms;
    j["bottleneck"]        = plan.bottleneck;

    json jw = json::array();
    for (const auto & w : plan.weights) {
        jw.push_back({
            {"weight_id", w.weight_id},
            {"name",      w.name},
            {"layer",     w.layer},
            {"byte_size", w.byte_size},
            {"location",  to_string(w.location)},
            {"pinned",    w.pinned},
            {"xform",     to_string(w.xform)},
        });
    }
    j["weights"] = std::move(jw);

    json jo = json::array();
    for (const auto & o : plan.ops) {
        jo.push_back({
            {"op_id",           o.op_id},
            {"name",            o.name},
            {"layer",           o.layer},
            {"compute_backend", to_string(o.compute_backend)},
            {"weight_id",       o.weight_id},
            {"dispatch",        to_string(o.dispatch)},
            {"migrate",         o.migrate},
            {"migrate_from",    to_string(o.migrate_from)},
            {"migrate_xform",   to_string(o.migrate_xform)},
        });
    }
    j["ops"] = std::move(jo);

    json jt = json::array();
    for (const auto & e : plan.timeline) {
        jt.push_back({
            {"kind",          to_string(e.kind)},
            {"weight_id",     e.weight_id},
            {"from_loc",      to_string(e.from_loc)},
            {"to_loc",        to_string(e.to_loc)},
            {"engine",        to_string(e.engine)},
            {"anchor_op_id",  e.anchor_op_id},
            {"overlap_group", e.overlap_group},
        });
    }
    j["timeline"] = std::move(jt);

    out = j.dump(2);
    return true;
}

bool plan_to_json_file(const ExecPlan & plan, const std::string & path) {
    std::string s;
    if (!plan_to_json_string(plan, s)) return false;
    std::ofstream f(path);
    if (!f) return false;
    f << s;
    return f.good();
}

bool plan_from_json_string(const std::string & s, ExecPlan & out, std::string * err) {
    json j;
    try {
        j = json::parse(s);
    } catch (const std::exception & e) {
        if (err) *err = std::string("json parse: ") + e.what();
        return false;
    }
    try {
        out = ExecPlan{};
        out.schema_version    = j.value("schema_version", 1);
        out.budget_mib        = j.value("budget_mib", (int64_t) 0);
        out.kv_bytes          = j.value("kv_bytes", (size_t) 0);
        out.misc_bytes        = j.value("misc_bytes", (size_t) 0);
        out.pred_per_token_ms = j.value("pred_per_token_ms", 0.0);
        out.bottleneck        = j.value("bottleneck", std::string());

        for (const auto & jw : j.value("weights", json::array())) {
            WeightPlan w;
            w.weight_id = jw.value("weight_id", -1);
            w.name      = jw.value("name", std::string());
            w.layer     = jw.value("layer", -1);
            w.byte_size = jw.value("byte_size", (size_t) 0);
            w.location  = location_from_string(jw.value("location", std::string("disk")));
            w.pinned    = jw.value("pinned", false);
            w.xform     = xform_from_string(jw.value("xform", std::string("none")));
            out.weights.push_back(std::move(w));
        }
        for (const auto & jo : j.value("ops", json::array())) {
            OpPlan o;
            o.op_id           = jo.value("op_id", -1);
            o.name            = jo.value("name", std::string());
            o.layer           = jo.value("layer", -1);
            o.compute_backend = backend_from_string(jo.value("compute_backend", std::string("gpu")));
            o.weight_id       = jo.value("weight_id", -1);
            o.dispatch        = dispatch_from_string(jo.value("dispatch", std::string("static")));
            o.migrate         = jo.value("migrate", false);
            o.migrate_from    = backend_from_string(jo.value("migrate_from", std::string("cpu")));
            o.migrate_xform   = xform_from_string(jo.value("migrate_xform", std::string("none")));
            out.ops.push_back(std::move(o));
        }
        for (const auto & je : j.value("timeline", json::array())) {
            PlanEvent e;
            e.kind          = evkind_from_string(je.value("kind", std::string("prefetch")));
            e.weight_id     = je.value("weight_id", -1);
            e.from_loc      = location_from_string(je.value("from_loc", std::string("disk")));
            e.to_loc        = location_from_string(je.value("to_loc", std::string("cpu")));
            e.engine        = engine_from_string(je.value("engine", std::string("disk")));
            e.anchor_op_id  = je.value("anchor_op_id", -1);
            e.overlap_group = je.value("overlap_group", -1);
            out.timeline.push_back(std::move(e));
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("plan decode: ") + e.what();
        return false;
    }
    return true;
}

bool plan_from_json_file(const std::string & path, ExecPlan & out, std::string * err) {
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "open failed: " + path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return plan_from_json_string(ss.str(), out, err);
}

// ───────────────────────── make_plan.py 格式加载 ─────────────────────────
//
// 现有 plan_XXXXMiB.json:
//   routes:             { "blk.0.attn_q.weight": "cpu"|"gpu", ... }      → OpPlan
//   resident_in_memory: [ {name, backend:"gpu"|"cpu"}, ... ]            → WeightPlan.location
//   schedule:           [ {step, compute:{weight,backend,start_ms,end_ms},
//                          disk_in:[{weight,ds,de}], dma_to_gpu:[...],
//                          evict_out:[{weight,at}]}, ... ]              → timeline
//   per_token_ms / bottleneck / budget_mib                              → 指标

bool plan_from_make_plan_file(const std::string & path, ExecPlan & out, std::string * err) {
    json j;
    {
        std::ifstream f(path);
        if (!f) {
            if (err) *err = "open failed: " + path;
            return false;
        }
        try {
            std::stringstream ss;
            ss << f.rdbuf();
            j = json::parse(ss.str());
        } catch (const std::exception & e) {
            if (err) *err = std::string("json parse: ") + e.what();
            return false;
        }
    }

    try {
        out = ExecPlan{};
        out.schema_version    = 1;
        out.budget_mib        = (int64_t) llround(j.value("budget_mib", 0.0));
        out.pred_per_token_ms = j.value("per_token_ms", 0.0);
        out.bottleneck        = j.value("bottleneck", std::string());

        // 1) 先用 resident_in_memory 决定每个 weight 的 location(gpu/cpu);
        //    其余出现在 routes 里、但不常驻的 → DISK(流式)。
        std::unordered_map<std::string, Location> loc_of;
        for (const auto & r : j.value("resident_in_memory", json::array())) {
            std::string name = r.value("name", std::string());
            std::string be   = r.value("backend", std::string("gpu"));
            if (name.empty()) continue;
            loc_of[name] = (be == "gpu") ? Location::GPU : Location::CPU;
        }

        // 2) 建 weight 表:union(routes keys, resident names),去重并分配 weight_id。
        std::unordered_map<std::string, int> wid;  // name → weight_id
        auto ensure_weight = [&](const std::string & name) -> int {
            auto it = wid.find(name);
            if (it != wid.end()) return it->second;
            WeightPlan w;
            w.weight_id = (int) out.weights.size();
            w.name      = name;
            w.layer     = parse_layer(name);
            auto lit = loc_of.find(name);
            w.location  = (lit != loc_of.end()) ? lit->second : Location::DISK;
            w.pinned    = (w.location == Location::GPU);  // 常驻 GPU 视为 pinned(此 plan 不动)
            out.weights.push_back(std::move(w));
            wid[name] = out.weights.back().weight_id;
            return out.weights.back().weight_id;
        };

        const json & routes = j.value("routes", json::object());
        // routes 是 object;按插入序不稳定,但 schedule 给了真实 use-order。先把 routes 全建出来。
        for (auto it = routes.begin(); it != routes.end(); ++it) {
            ensure_weight(it.key());
        }
        for (const auto & r : j.value("resident_in_memory", json::array())) {
            std::string name = r.value("name", std::string());
            if (!name.empty()) ensure_weight(name);
        }

        // 3) OpPlan:每条 route = 一个消费该 weight 的 op。compute_backend = route 值。
        //    op 顺序按 schedule 的 step 顺序(真实 decode use-order),routes 里没在
        //    schedule 出现的(理论上不该有)追加在后面。
        std::unordered_map<std::string, int> opid;  // weight name → op_id
        auto ensure_op = [&](const std::string & name, Backend be) -> int {
            auto it = opid.find(name);
            if (it != opid.end()) return it->second;
            OpPlan o;
            o.op_id           = (int) out.ops.size();
            o.name            = name;
            o.layer           = parse_layer(name);
            o.compute_backend = be;
            o.weight_id       = ensure_weight(name);
            o.dispatch        = Dispatch::STATIC;
            // 迁移:op 在 GPU 算但 weight 当前不在 GPU → 需 CPU→GPU 迁移(含 convert)。
            const WeightPlan & w = out.weights[o.weight_id];
            Backend wbe = backend_of_location(w.location);
            if (be != wbe) {
                o.migrate       = true;
                o.migrate_from  = wbe;
                o.migrate_xform = (be == Backend::GPU) ? Xform::GPU_CONVERT
                                : (be == Backend::CPU) ? Xform::CPU_REPACK : Xform::NONE;
            }
            out.ops.push_back(std::move(o));
            opid[name] = out.ops.back().op_id;
            return out.ops.back().op_id;
        };

        auto route_backend = [&](const std::string & name) -> Backend {
            auto it = routes.find(name);
            if (it != routes.end()) return backend_from_string(it.value().get<std::string>());
            // 没 route 信息:跟 location 一致(常驻 GPU→GPU,否则 CPU)
            int id = ensure_weight(name);
            return backend_of_location(out.weights[id].location);
        };

        const json & sched = j.value("schedule", json::array());
        for (const auto & st : sched) {
            const auto & cj = st.value("compute", json::object());
            std::string cw  = cj.value("weight", std::string());
            if (cw.empty()) continue;
            Backend cbe = backend_from_string(cj.value("backend", std::string("gpu")));
            ensure_op(cw, cbe);
        }
        // routes 里没进 schedule 的兜底
        for (auto it = routes.begin(); it != routes.end(); ++it) {
            ensure_op(it.key(), route_backend(it.key()));
        }

        // 4) 派生每个 weight 的 xform:取它给其 op 的 compute backend 用时要的变换。
        for (auto & w : out.weights) {
            auto it = opid.find(w.name);
            Backend cbe = (it != opid.end()) ? out.ops[it->second].compute_backend
                                             : backend_of_location(w.location);
            if (cbe == Backend::GPU)      w.xform = Xform::GPU_CONVERT;
            else if (cbe == Backend::CPU) w.xform = Xform::NONE;  // 有 generic 退路,默认不 repack
        }

        // 5) timeline:把 schedule 的 disk_in / dma_to_gpu / evict_out 映射成 PlanEvent,
        //    anchor = 该 step 正在 compute 的 op。
        for (const auto & st : sched) {
            const auto & cj = st.value("compute", json::object());
            std::string cw  = cj.value("weight", std::string());
            int anchor = -1;
            if (!cw.empty()) {
                auto it = opid.find(cw);
                if (it != opid.end()) anchor = it->second;
            }
            for (const auto & d : st.value("disk_in", json::array())) {
                std::string nm = d.value("weight", std::string());
                if (nm.empty()) continue;
                PlanEvent e;
                e.kind         = EvKind::LOAD;
                e.weight_id    = ensure_weight(nm);
                e.from_loc     = Location::DISK;
                e.to_loc       = Location::CPU;
                e.engine       = Engine::DISK;
                e.anchor_op_id = anchor;
                out.timeline.push_back(e);
            }
            for (const auto & d : st.value("dma_to_gpu", json::array())) {
                std::string nm = d.value("weight", std::string());
                if (nm.empty()) continue;
                int wi = ensure_weight(nm);
                PlanEvent dma;
                dma.kind         = EvKind::DMA;
                dma.weight_id    = wi;
                dma.from_loc     = Location::CPU;
                dma.to_loc       = Location::GPU;
                dma.engine       = Engine::DMA;
                dma.anchor_op_id = anchor;
                out.timeline.push_back(dma);
                // GPU 落地伴随 convert+transpose(跑在 GPU 引擎)
                PlanEvent xf;
                xf.kind         = EvKind::XFORM;
                xf.weight_id    = wi;
                xf.from_loc     = Location::GPU;
                xf.to_loc       = Location::GPU;
                xf.engine       = Engine::GPU;
                xf.anchor_op_id = anchor;
                out.timeline.push_back(xf);
            }
            for (const auto & d : st.value("evict_out", json::array())) {
                std::string nm = d.value("weight", std::string());
                if (nm.empty()) continue;
                PlanEvent e;
                e.kind         = EvKind::EVICT;
                e.weight_id    = ensure_weight(nm);
                e.from_loc     = Location::GPU;
                e.to_loc       = Location::DISK;
                e.engine       = Engine::GPU;
                e.anchor_op_id = anchor;
                out.timeline.push_back(e);
            }
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("make_plan decode: ") + e.what();
        return false;
    }
    return true;
}

}  // namespace elastic
