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
#include <unordered_set>

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
        case Engine::CPU:      return "cpu";
        case Engine::GPU:      return "gpu";
        case Engine::DISK:     return "disk";
        case Engine::TRANSFER: return "transfer";
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
        case EvKind::TRANSFER: return "transfer";
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

struct NameAnchorHash {
    size_t operator()(const std::pair<std::string, int> & p) const {
        return std::hash<std::string>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
    }
};

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
    if (s == "transfer" || s == "dma") return Engine::TRANSFER;
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
    if (s == "transfer" || s == "dma") return EvKind::TRANSFER;
    if (s == "xform" || s == "prepare" || s == "materialize") return EvKind::XFORM;
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
    if (plan.working_unit.enabled) {
        json jwu = {
            {"enabled",        true},
            {"mode",           plan.working_unit.mode},
            {"cut_parts",      plan.working_unit.cut_parts},
            {"multi_tensors",  plan.working_unit.multi_tensors},
            {"policy",         plan.working_unit.policy},
            {"state_aware",    plan.working_unit.state_aware},
            {"predicted_ms",   plan.working_unit.predicted_ms},
            {"switch_cost_ms", plan.working_unit.switch_cost_ms},
        };
        if (!plan.working_unit.units.empty()) {
            jwu["units"] = json::array();
            for (const auto & unit : plan.working_unit.units) {
                json ju = {
                    {"unit_id", unit.unit_id},
                    {"fuse_layout", unit.fuse_layout},
                    {"fuse_compute", unit.fuse_compute},
                    {"tiles", json::array()},
                };
                for (const auto & tile : unit.tiles) {
                    ju["tiles"].push_back({
                        {"weight_id", tile.weight_id},
                        {"weight_name", tile.weight_name},
                        {"row_start", tile.row_start},
                        {"row_count", tile.row_count},
                        {"byte_offset", tile.byte_offset},
                        {"byte_size", tile.byte_size},
                    });
                }
                jwu["units"].push_back(std::move(ju));
            }
        }
        j["working_unit"] = std::move(jwu);
    }
    if (!plan.working_sets.empty()) {
        json jws = json::array();
        for (const auto & ws : plan.working_sets) {
            jws.push_back({
                {"name",            ws.name},
                {"kind",            ws.kind},
                {"target_capacity", ws.target_capacity},
                {"budget_capacity", ws.budget_capacity},
                {"min_capacity",    ws.min_capacity},
                {"max_capacity",    ws.max_capacity},
                {"policy",          ws.policy},
                {"state_aware",     ws.state_aware},
                {"coupled_to_core", ws.coupled_to_core},
            });
        }
        j["working_sets"] = std::move(jws);
    }
    if (!plan.schedule_kind.empty() || !plan.schedule_events.empty()) {
        json js;
        js["kind"]         = plan.schedule_kind;
        js["status"]       = plan.schedule_status;
        js["objective_ms"] = plan.schedule_objective_ms;
        json je = json::array();
        for (const auto & e : plan.schedule_events) {
            je.push_back({
                {"weight_id",   e.weight_id},
                {"anchor_op_id", e.anchor_op_id},
                {"weight_name", e.weight_name},
                {"choice",      e.choice},
                {"kind",        e.kind},
                {"engine",      e.engine},
                {"start_ms",    e.start_ms},
                {"end_ms",      e.end_ms},
                {"duration_ms", e.duration_ms},
            });
        }
        js["events"] = std::move(je);
        j["schedule"] = std::move(js);
    }

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
        if (j.contains("working_unit") && j["working_unit"].is_object()) {
            const auto & wu = j["working_unit"];
            out.working_unit.enabled =
                wu.value("enabled", true);
            out.working_unit.mode =
                wu.value("mode", std::string("tensor"));
            out.working_unit.cut_parts =
                std::max(2, wu.value("cut_parts", 2));
            out.working_unit.multi_tensors =
                std::max(2, wu.value("multi_tensors", 2));
            out.working_unit.policy =
                wu.value("policy", std::string());
            out.working_unit.state_aware =
                wu.value("state_aware", false);
            out.working_unit.predicted_ms =
                wu.value("predicted_ms", 0.0);
            out.working_unit.switch_cost_ms =
                wu.value("switch_cost_ms", 0.0);
            for (const auto & ju :
                 wu.value("units", json::array())) {
                SuperTensorUnitPlan unit;
                unit.unit_id = ju.value(
                    "unit_id",
                    (int) out.working_unit.units.size());
                unit.fuse_layout =
                    ju.value("fuse_layout", false);
                unit.fuse_compute =
                    ju.value("fuse_compute", false);
                for (const auto & jt :
                     ju.value("tiles", json::array())) {
                    WorkingUnitTilePlan tile;
                    tile.weight_id =
                        jt.value("weight_id", -1);
                    tile.weight_name =
                        jt.value("weight_name", std::string());
                    tile.row_start =
                        jt.value("row_start", (int64_t) 0);
                    tile.row_count =
                        jt.value("row_count", (int64_t) -1);
                    tile.byte_offset =
                        jt.value("byte_offset", (size_t) 0);
                    tile.byte_size =
                        jt.value("byte_size", (size_t) 0);
                    unit.tiles.push_back(std::move(tile));
                }
                if (!unit.tiles.empty()) {
                    out.working_unit.units.push_back(
                        std::move(unit));
                }
            }
        }
        for (const auto & jws : j.value("working_sets", json::array())) {
            WorkingSetPlan ws;
            ws.name            = jws.value("name", std::string());
            ws.kind            = jws.value("kind", std::string());
            ws.target_capacity = jws.value("target_capacity", -1);
            ws.budget_capacity = jws.value("budget_capacity", ws.target_capacity);
            ws.min_capacity    = jws.value("min_capacity", 0);
            ws.max_capacity    = jws.value("max_capacity", -1);
            ws.policy          = jws.value("policy", std::string());
            ws.state_aware     = jws.value("state_aware", false);
            ws.coupled_to_core = jws.value("coupled_to_core", false);
            out.working_sets.push_back(std::move(ws));
        }
        if (j.contains("schedule") && j["schedule"].is_object()) {
            const auto & js = j["schedule"];
            out.schedule_kind         = js.value("kind", std::string());
            out.schedule_status       = (js.contains("status") and js["status"].is_string())
                                      ? js["status"].get<std::string>() : std::string();
            out.schedule_objective_ms = (js.contains("objective_ms") and not js["objective_ms"].is_null())
                                      ? js["objective_ms"].get<double>() : 0.0;
            for (const auto & je : js.value("events", json::array())) {
                ScheduleEvent e;
                e.weight_id   = je.value("weight_id", -1);
                e.anchor_op_id = je.value("anchor_op_id", -1);
                e.weight_name = je.value("weight_name", std::string());
                e.choice      = je.value("choice", std::string());
                e.kind        = je.value("kind", std::string());
                e.engine      = je.value("engine", std::string());
                e.start_ms    = je.value("start_ms", 0.0);
                e.end_ms      = je.value("end_ms", 0.0);
                e.duration_ms = je.value("duration_ms", 0.0);
                out.schedule_events.push_back(std::move(e));
            }
        }

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
        if (j.contains("working_unit") && j["working_unit"].is_object()) {
            const auto & wu = j["working_unit"];
            out.working_unit.enabled =
                wu.value("enabled", true);
            out.working_unit.mode =
                wu.value("mode", std::string("tensor"));
            out.working_unit.cut_parts =
                std::max(2, wu.value("cut_parts", 2));
            out.working_unit.multi_tensors =
                std::max(2, wu.value("multi_tensors", 2));
            out.working_unit.policy =
                wu.value("policy", std::string());
            out.working_unit.state_aware =
                wu.value("state_aware", false);
            out.working_unit.predicted_ms =
                wu.value("predicted_ms", 0.0);
            out.working_unit.switch_cost_ms =
                wu.value("switch_cost_ms", 0.0);
            for (const auto & ju :
                 wu.value("units", json::array())) {
                SuperTensorUnitPlan unit;
                unit.unit_id = ju.value(
                    "unit_id",
                    (int) out.working_unit.units.size());
                unit.fuse_layout =
                    ju.value("fuse_layout", false);
                unit.fuse_compute =
                    ju.value("fuse_compute", false);
                for (const auto & jt :
                     ju.value("tiles", json::array())) {
                    WorkingUnitTilePlan tile;
                    tile.weight_id =
                        jt.value("weight_id", -1);
                    tile.weight_name =
                        jt.value("weight_name", std::string());
                    tile.row_start =
                        jt.value("row_start", (int64_t) 0);
                    tile.row_count =
                        jt.value("row_count", (int64_t) -1);
                    tile.byte_offset =
                        jt.value("byte_offset", (size_t) 0);
                    tile.byte_size =
                        jt.value("byte_size", (size_t) 0);
                    unit.tiles.push_back(std::move(tile));
                }
                if (!unit.tiles.empty()) {
                    out.working_unit.units.push_back(
                        std::move(unit));
                }
            }
        }

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
            else if (cbe == Backend::CPU) w.xform = Xform::CPU_REPACK;
        }

        // 5) timeline:把 schedule 的 disk_in / dma_to_gpu / evict_out 映射成 PlanEvent,
        //    anchor = 该 step 正在 compute 的 op。
        std::unordered_set<std::pair<std::string, int>, NameAnchorHash> explicit_dma;
        auto stage_anchor_for = [&](const std::string & weight_name,
                                    const std::string & current_compute,
                                    int current_anchor) -> int {
            if (route_backend(weight_name) != Backend::GPU) return current_anchor;
            if (!current_compute.empty() && route_backend(current_compute) == Backend::GPU) {
                return current_anchor;
            }
            auto it = opid.find(weight_name);
            return it != opid.end() ? it->second : current_anchor;
        };
        for (const auto & st : sched) {
            const auto & cj = st.value("compute", json::object());
            std::string cw  = cj.value("weight", std::string());
            int anchor = -1;
            if (!cw.empty()) {
                auto it = opid.find(cw);
                if (it != opid.end()) anchor = it->second;
            }
            for (const auto & d : st.value("dma_to_gpu", json::array())) {
                std::string nm = d.value("weight", std::string());
                if (!nm.empty()) explicit_dma.insert({nm, anchor});
            }
        }
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
                const int wi = ensure_weight(nm);
                const int event_anchor = stage_anchor_for(nm, cw, anchor);
                PlanEvent e;
                e.kind         = EvKind::LOAD;
                e.weight_id    = wi;
                e.from_loc     = Location::DISK;
                e.to_loc       = Location::CPU;
                e.engine       = Engine::DISK;
                e.anchor_op_id = event_anchor;
                out.timeline.push_back(e);
                if (route_backend(nm) == Backend::CPU) {
                    PlanEvent xf;
                    xf.kind         = EvKind::XFORM;
                    xf.weight_id    = wi;
                    xf.from_loc     = Location::CPU;
                    xf.to_loc       = Location::CPU;
                    xf.engine       = Engine::CPU;
                    xf.anchor_op_id = event_anchor;
                    out.timeline.push_back(xf);
                } else if (route_backend(nm) == Backend::GPU &&
                           explicit_dma.find({nm, anchor}) == explicit_dma.end()) {
                    PlanEvent transfer;
                    transfer.kind         = EvKind::TRANSFER;
                    transfer.weight_id    = wi;
                    transfer.from_loc     = Location::CPU;
                    transfer.to_loc       = Location::GPU;
                    transfer.engine       = Engine::TRANSFER;
                    transfer.anchor_op_id = event_anchor;
                    out.timeline.push_back(transfer);

                    PlanEvent xf;
                    xf.kind         = EvKind::XFORM;
                    xf.weight_id    = wi;
                    xf.from_loc     = Location::GPU;
                    xf.to_loc       = Location::GPU;
                    xf.engine       = Engine::GPU;
                    xf.anchor_op_id = event_anchor;
                    out.timeline.push_back(xf);
                }
            }
            for (const auto & d : st.value("dma_to_gpu", json::array())) {
                std::string nm = d.value("weight", std::string());
                if (nm.empty()) continue;
                int wi = ensure_weight(nm);
                const int event_anchor = stage_anchor_for(nm, cw, anchor);
                PlanEvent transfer;
                transfer.kind         = EvKind::TRANSFER;
                transfer.weight_id    = wi;
                transfer.from_loc     = Location::CPU;
                transfer.to_loc       = Location::GPU;
                transfer.engine       = Engine::TRANSFER;
                transfer.anchor_op_id = event_anchor;
                out.timeline.push_back(transfer);
                // GPU 落地伴随 convert+transpose(跑在 GPU 引擎)
                PlanEvent xf;
                xf.kind         = EvKind::XFORM;
                xf.weight_id    = wi;
                xf.from_loc     = Location::GPU;
                xf.to_loc       = Location::GPU;
                xf.engine       = Engine::GPU;
                xf.anchor_op_id = event_anchor;
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
