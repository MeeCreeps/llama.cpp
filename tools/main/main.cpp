#include "arg.h"
#include "common.h"
#include "http.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <cerrno>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting  = false;
static bool need_insert_eot = false;

static std::string shell_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static bool mkdir_p_local(const std::string & path) {
    if (path.empty()) {
        return false;
    }

    std::string cur;
    cur.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        const char c = path[i];
        cur.push_back(c);
        if (c != '/' && i + 1 != path.size()) {
            continue;
        }
        while (i + 1 < path.size() && path[i + 1] == '/') {
            ++i;
        }
        if (cur.empty() || cur == "/") {
            continue;
        }
#if defined(_WIN32)
        if (!CreateDirectoryA(cur.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
            return false;
        }
#else
        if (mkdir(cur.c_str(), 0777) != 0 && errno != EEXIST) {
            return false;
        }
#endif
    }
    return true;
}

struct elastic_online_solver_state {
    struct item_base {
        int id = -1;
        std::string name;
        std::string backend;
        std::string quant;
        int layer = -1;
        size_t bytes = 0;
        double resident_value_gpu = 0.0;
        double resident_value_cpu = 0.0;
    };

    llama_context * ctx = nullptr;
    nlohmann::json  model_meta;
    std::vector<std::string> weight_names;
    std::unordered_map<std::string, nlohmann::json> weight_by_name;
    std::vector<item_base> item_bases;
    nlohmann::json stage_costs;
    nlohmann::json op_costs;
    std::string mode;
    std::string python;
    std::string solver;
    std::string remote_url;
    std::string model_meta_path;
    std::string cost_dir;
    std::string work_dir;
    int kv_mib = 128;
    int misc_mib = 256;
    int safety_mib = 64;
    int time_limit_ms = 20;
    bool allow_cpu_fallback = false;
    double transition_weight = 1.0;
    uint64_t calls = 0;
    uint64_t failures = 0;
    double remote_wall_ms_total = 0.0;
    double remote_server_ms_total = 0.0;
    std::vector<llama_plan *> plans;
};

static double elastic_cost_lookup(const nlohmann::json & records, const std::string & backend,
                                  const std::string & kind, const std::string & name,
                                  size_t bytes, double fallback) {
    const auto & rs = records.value("records", nlohmann::json::array());
    const nlohmann::json * best_size = nullptr;
    for (const auto & r : rs) {
        if (r.value("backend", std::string()) != backend) continue;
        if (!kind.empty() && r.value("kind", std::string()) != kind) continue;
        if (r.value("name", std::string()) == name) return r.value("median_ms", fallback);
        if ((size_t) r.value("bytes", 0) == bytes && best_size == nullptr) best_size = &r;
    }
    return best_size ? best_size->value("median_ms", fallback) : fallback;
}

static bool elastic_state_has(uint32_t flags, const std::string & be);

static std::vector<std::string> elastic_compute_profile_names(const std::string & name) {
    std::vector<std::string> out;
    out.push_back(name);
    if (name == "output.weight") {
        out.push_back("result_output");
        return out;
    }

    const std::string prefix = "blk.";
    const std::string suffix = ".weight";
    if (name.rfind(prefix, 0) != 0 || name.size() <= prefix.size() + suffix.size()) {
        return out;
    }
    if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return out;
    }

    const size_t layer_begin = prefix.size();
    const size_t layer_end = name.find('.', layer_begin);
    if (layer_end == std::string::npos) {
        return out;
    }
    const std::string layer = name.substr(layer_begin, layer_end - layer_begin);
    const std::string field = name.substr(layer_end + 1, name.size() - layer_end - 1 - suffix.size());

    if (field == "attn_q") out.push_back("Qcur-" + layer);
    else if (field == "attn_k") out.push_back("Kcur-" + layer);
    else if (field == "attn_v") out.push_back("Vcur-" + layer);
    else if (field == "attn_output") out.push_back("attn_out-" + layer);
    else if (field == "ffn_gate") out.push_back("ffn_gate-" + layer);
    else if (field == "ffn_up") out.push_back("ffn_up-" + layer);
    else if (field == "ffn_down") out.push_back("ffn_out-" + layer);
    return out;
}

static bool elastic_has_measured_compute(elastic_online_solver_state * s, const std::string & backend) {
    const auto & rs = s->op_costs.value("records", nlohmann::json::array());
    for (const auto & r : rs) {
        if (r.value("backend", std::string()) == backend && r.value("kind", std::string()) == "COMPUTE") {
            return true;
        }
    }
    return false;
}

static bool elastic_has_measured_stage_kind(elastic_online_solver_state * s,
                                            const std::string & backend, const std::string & kind) {
    const auto & rs = s->stage_costs.value("records", nlohmann::json::array());
    for (const auto & r : rs) {
        if (r.value("backend", std::string()) == backend && r.value("kind", std::string()) == kind) {
            return true;
        }
    }
    return false;
}

static double elastic_stage_cost(elastic_online_solver_state * s, const std::string & backend,
                                 const std::string & kind, const std::string & name, size_t bytes) {
    const double mb = bytes / 1024.0 / 1024.0;
    double fb = 0.0;
    if (kind == "LOAD" || kind == "RELOAD_ENSURE") fb = 0.15 + mb / 1800.0 * 1000.0;
    else if (kind == "TRANSFER") fb = 0.05 + mb / 6000.0 * 1000.0;
    else if (kind == "XFORM") fb = 0.10 + mb / 8000.0 * 1000.0;
    return elastic_cost_lookup(s->stage_costs, backend, kind, name, bytes, fb);
}

static double elastic_gpu_reload_cost(elastic_online_solver_state * s, const std::string & name, size_t bytes) {
    if (elastic_has_measured_stage_kind(s, "OpenCL", "RELOAD_ENSURE")) {
        return elastic_stage_cost(s, "OpenCL", "RELOAD_ENSURE", name, bytes);
    }
    return elastic_stage_cost(s, "OpenCL", "LOAD", name, bytes) +
           elastic_stage_cost(s, "OpenCL", "TRANSFER", name, bytes) +
           elastic_stage_cost(s, "OpenCL", "XFORM", name, bytes);
}

static double elastic_compute_cost(elastic_online_solver_state * s, const std::string & backend,
                                   const std::string & name, size_t bytes) {
    if (backend == "CPU_Elastic" && !elastic_has_measured_compute(s, "CPU_Elastic")) {
        return std::numeric_limits<double>::infinity();
    }
    const auto candidates = elastic_compute_profile_names(name);
    const auto & rs = s->op_costs.value("records", nlohmann::json::array());
    const nlohmann::json * exact = nullptr;
    const nlohmann::json * best_mul_mat = nullptr;
    for (const auto & r : rs) {
        if (r.value("backend", std::string()) != backend) continue;
        if (r.value("kind", std::string()) != "COMPUTE") continue;
        const std::string rname = r.value("name", std::string());
        if (std::find(candidates.begin(), candidates.end(), rname) == candidates.end()) continue;
        if (r.value("op", std::string()) == "MUL_MAT") {
            if (best_mul_mat == nullptr || r.value("samples", 0) > best_mul_mat->value("samples", 0)) {
                best_mul_mat = &r;
            }
            continue;
        }
        if (exact == nullptr) exact = &r;
    }
    if (best_mul_mat != nullptr) return best_mul_mat->value("median_ms", 0.0);
    if (exact != nullptr) return exact->value("median_ms", 0.0);

    const double mb = bytes / 1024.0 / 1024.0;
    const double fb = backend == "OpenCL" ? 0.03 + mb * 0.020 : 0.05 + mb * 0.045;
    return elastic_cost_lookup(s->op_costs, backend, "COMPUTE", name, bytes, fb);
}

static double elastic_backend_path_cost(elastic_online_solver_state * s, const std::string & backend,
                                        const std::string & name, size_t bytes, uint32_t flags = 0) {
    if (backend == "GPU") {
        double cost = elastic_compute_cost(s, "OpenCL", name, bytes);
        if (!elastic_state_has(flags, "GPU")) {
            cost += elastic_gpu_reload_cost(s, name, bytes);
        }
        return cost;
    }

    double cost = elastic_compute_cost(s, "CPU_Elastic", name, bytes);
    if (!elastic_state_has(flags, "CPU")) {
        cost += elastic_stage_cost(s, "CPU_Elastic", "LOAD", name, bytes);
        cost += elastic_stage_cost(s, "CPU_Elastic", "XFORM", name, bytes);
    }
    return cost;
}

static bool elastic_state_has(uint32_t flags, const std::string & be) {
    if (be == "GPU") return (flags & LLAMA_ELASTIC_WEIGHT_GPU_COMPUTE_RESIDENT) != 0;
    return (flags & LLAMA_ELASTIC_WEIGHT_CPU_COMPUTE_RESIDENT) != 0;
}

static bool elastic_state_any_resident(uint32_t flags) {
    return (flags & (LLAMA_ELASTIC_WEIGHT_CPU_RAW_RESIDENT |
                     LLAMA_ELASTIC_WEIGHT_CPU_COMPUTE_RESIDENT |
                     LLAMA_ELASTIC_WEIGHT_GPU_RAW_RESIDENT |
                     LLAMA_ELASTIC_WEIGHT_GPU_COMPUTE_RESIDENT)) != 0;
}

static std::string elastic_backend_from_state(uint32_t flags, const std::string & fallback) {
    const bool gpu = (flags & (LLAMA_ELASTIC_WEIGHT_GPU_RAW_RESIDENT |
                               LLAMA_ELASTIC_WEIGHT_GPU_COMPUTE_RESIDENT)) != 0;
    const bool cpu = (flags & (LLAMA_ELASTIC_WEIGHT_CPU_RAW_RESIDENT |
                               LLAMA_ELASTIC_WEIGHT_CPU_COMPUTE_RESIDENT)) != 0;
    if (gpu && !cpu) return "GPU";
    if (cpu && !gpu) return "CPU";
    return fallback;
}

static std::string elastic_backend_for_item(uint32_t flags, const std::string & fallback, const std::string & quant) {
    if (quant.empty()) {
        return elastic_backend_from_state(flags, fallback);
    }
    return fallback;
}


static std::string elastic_xform_for(const std::string & be, const std::string & quant) {
    if (be == "GPU") {
        return (quant == "Q4_0" || quant == "Q8_0" || quant == "MXFP4" || quant.empty()) ? "gpu_convert" : "none";
    }
    return "cpu_repack";
}

static bool elastic_online_generate_native(elastic_online_solver_state * s, int64_t budget_mib, const std::string & plan_path) {
    struct item {
        int id;
        std::string name;
        std::string backend;
        std::string quant;
        int layer;
        size_t bytes;
        uint32_t flags;
        double value;
        bool manageable;
    };
    std::vector<item> items;
    items.reserve(s->item_bases.size());
    const bool mru_mode = s->mode == "mru" || s->mode == "native-mru";
    for (const auto & b : s->item_bases) {
        item it;
        it.id = b.id;
        it.name = b.name;
        it.layer = b.layer;
        it.bytes = b.bytes;
        it.quant = b.quant;
        llama_elastic_weight_state st{};
        llama_weight_get_state(s->ctx, it.name.c_str(), &st);
        it.flags = st.flags;
        // Items in model_meta are elastic model weights and can be evicted and
        // reloaded from the model file. Do not let a missing disk_available bit
        // in a transient state dump pin online plans to an old resident set.
        it.manageable = true;
        it.backend = elastic_backend_for_item(it.flags, b.backend, it.quant);
        const double movement_value = it.backend == "GPU" ? b.resident_value_gpu : b.resident_value_cpu;
        if (elastic_state_has(it.flags, it.backend)) {
            // Finite churn bonus only. An infinite bonus makes online keep cheap
            // stale tensors and can be worse than the offline table.
            it.value = movement_value + std::min(movement_value * 0.25, 2.0);
        } else {
            it.value = movement_value;
        }
        if (!it.name.empty()) items.push_back(std::move(it));
    }

    const int64_t usable_mib = budget_mib - s->kv_mib - s->misc_mib - s->safety_mib;
    const size_t budget_bytes = usable_mib > 0 ? (size_t) usable_mib * 1024 * 1024 : 0;
    if (mru_mode) {
        // MRU eviction baseline: after a completed decode step, later op ids are
        // treated as the most recently used weights, so pressure evicts them first.
        std::sort(items.begin(), items.end(), [](const item & a, const item & b) {
            if (a.id != b.id) return a.id < b.id;
            return a.name < b.name;
        });
    } else {
        std::sort(items.begin(), items.end(), [](const item & a, const item & b) {
            const double da = a.value / std::max<size_t>(a.bytes, 1);
            const double db = b.value / std::max<size_t>(b.bytes, 1);
            if (da != db) return da > db;
            return a.value > b.value;
        });
    }
    std::unordered_set<int> keep;
    size_t used = 0;
    for (const auto & it : items) {
        if (!it.manageable) {
            keep.insert(it.id);
            used += it.bytes;
        }
    }
    for (const auto & it : items) {
        if (!it.manageable) continue;
        if (used + it.bytes <= budget_bytes) {
            keep.insert(it.id);
            used += it.bytes;
        }
    }
    std::sort(items.begin(), items.end(), [](const item & a, const item & b) { return a.id < b.id; });

    nlohmann::json plan;
    plan["schema_version"] = 1;
    plan["budget_mib"] = budget_mib;
    plan["kv_bytes"] = (size_t) s->kv_mib * 1024 * 1024;
    plan["misc_bytes"] = (size_t) s->misc_mib * 1024 * 1024;
    plan["weights"] = nlohmann::json::array();
    plan["ops"] = nlohmann::json::array();
    plan["timeline"] = nlohmann::json::array();
    plan["pred_per_token_ms"] = 0.0;
    plan["bottleneck"] = mru_mode ? "native_mru" : "native_greedy";

    for (const auto & it : items) {
        const bool resident = keep.find(it.id) != keep.end();
        const std::string loc = resident ? (it.backend == "GPU" ? "gpu" : "cpu") : "disk";
        plan["weights"].push_back({
            {"weight_id", it.id},
            {"name", it.name},
            {"layer", it.layer},
            {"byte_size", it.bytes},
            {"location", loc},
            {"pinned", false},
            {"xform", resident ? elastic_xform_for(it.backend, it.quant) : "none"},
        });
        plan["ops"].push_back({
            {"op_id", it.id},
            {"name", it.name},
            {"layer", it.layer},
            {"compute_backend", it.backend == "GPU" ? "gpu" : "cpu"},
            {"weight_id", it.id},
            {"dispatch", "static"},
            {"migrate", false},
            {"migrate_from", "cpu"},
            {"migrate_xform", "none"},
        });
        if (!resident) {
            const int anchor_id = it.id > 1 ? it.id - 1 : it.id;
            if (elastic_state_any_resident(it.flags)) {
                plan["timeline"].push_back({
                    {"kind", "evict"},
                    {"weight_id", it.id},
                    {"from_loc", (it.flags & LLAMA_ELASTIC_WEIGHT_GPU_COMPUTE_RESIDENT) ? "gpu" : "cpu"},
                    {"to_loc", "disk"},
                    {"engine", "cpu"},
                    {"anchor_op_id", anchor_id},
                    {"overlap_group", -1},
                });
            }
            plan["timeline"].push_back({
                {"kind", "load"}, {"weight_id", it.id}, {"from_loc", "disk"}, {"to_loc", "cpu"},
                {"engine", "disk"}, {"anchor_op_id", anchor_id}, {"overlap_group", -1},
            });
            if (it.backend == "GPU") {
                plan["timeline"].push_back({
                    {"kind", "transfer"}, {"weight_id", it.id}, {"from_loc", "cpu"}, {"to_loc", "gpu"},
                    {"engine", "transfer"}, {"anchor_op_id", anchor_id}, {"overlap_group", -1},
                });
                plan["timeline"].push_back({
                    {"kind", "xform"}, {"weight_id", it.id}, {"from_loc", "gpu"}, {"to_loc", "gpu"},
                    {"engine", "gpu"}, {"anchor_op_id", anchor_id}, {"overlap_group", -1},
                });
            } else {
                plan["timeline"].push_back({
                    {"kind", "xform"}, {"weight_id", it.id}, {"from_loc", "cpu"}, {"to_loc", "cpu"},
                    {"engine", "cpu"}, {"anchor_op_id", anchor_id}, {"overlap_group", -1},
                });
            }
        }
    }

    std::ofstream f(plan_path);
    if (!f) return false;
    f << plan.dump(2) << "\n";
    return f.good();
}

static void elastic_online_state_dump(elastic_online_solver_state * s, const std::string & path) {
    nlohmann::json out;
    out["weights"] = nlohmann::json::array();
    for (const auto & name : s->weight_names) {
        llama_elastic_weight_state st{};
        int rc = llama_weight_get_state(s->ctx, name.c_str(), &st);
        nlohmann::json flags = nlohmann::json::array();
        if (rc == 0) {
            if (st.flags & LLAMA_ELASTIC_WEIGHT_DISK_AVAILABLE)       flags.push_back("disk_available");
            if (st.flags & LLAMA_ELASTIC_WEIGHT_CPU_RAW_RESIDENT)     flags.push_back("cpu_raw_resident");
            if (st.flags & LLAMA_ELASTIC_WEIGHT_CPU_COMPUTE_RESIDENT) flags.push_back("cpu_compute_resident");
            if (st.flags & LLAMA_ELASTIC_WEIGHT_GPU_RAW_RESIDENT)     flags.push_back("gpu_raw_resident");
            if (st.flags & LLAMA_ELASTIC_WEIGHT_GPU_COMPUTE_RESIDENT) flags.push_back("gpu_compute_resident");
        }
        out["weights"].push_back({
            {"name", name},
            {"flags", flags},
        });
    }
    std::ofstream f(path);
    f << out.dump(2) << "\n";
}

static bool elastic_online_generate_remote(elastic_online_solver_state * s, int64_t budget_mib,
                                           const std::string & state_path, const std::string & plan_path,
                                           double * remote_wall_ms, double * server_ms) {
    if (!s || s->remote_url.empty()) return false;

    nlohmann::json req;
    req["budget_mib"] = budget_mib;
    req["kv_mib"] = s->kv_mib;
    req["misc_mib"] = s->misc_mib;
    req["safety_mib"] = s->safety_mib;
    req["time_limit_ms"] = s->time_limit_ms;
    req["allow_cpu_fallback"] = s->allow_cpu_fallback;
    req["transition_weight"] = s->transition_weight;
    {
        std::ifstream sf(state_path);
        if (!sf) return false;
        sf >> req["state"];
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
        auto [cli, parts] = common_http_client(s->remote_url);
        cli.set_connection_timeout(5, 0);
        cli.set_write_timeout(5, 0);
        cli.set_read_timeout(60, 0);
        auto res = cli.Post(parts.path, req.dump(), "application/json");
        const auto t1 = std::chrono::steady_clock::now();
        if (remote_wall_ms) {
            *remote_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        if (!res || res->status < 200 || res->status >= 300) {
            LOG_ERR("[elastic-online] remote solver http failed status=%d url=%s\n",
                    res ? res->status : -1, common_http_show_masked_url(parts).c_str());
            return false;
        }
        nlohmann::json out = nlohmann::json::parse(res->body);
        if (server_ms) *server_ms = out.value("solve_ms", 0.0);
        nlohmann::json plan = out.contains("plan") ? out["plan"] : out;
        std::ofstream pf(plan_path);
        if (!pf) return false;
        pf << plan.dump(2) << "\n";
        return pf.good();
    } catch (const std::exception & e) {
        LOG_ERR("[elastic-online] remote solver exception url=%s err=%s\n",
                s->remote_url.c_str(), e.what());
        return false;
    }
}

static const llama_plan * elastic_online_plan_provider(int64_t budget_mib, void * user_data) {
    auto * s = static_cast<elastic_online_solver_state *>(user_data);
    if (!s || !s->ctx) return nullptr;
    s->calls++;

    std::string prefix = s->work_dir + "/online_" + std::to_string((long long) budget_mib) + "_" + std::to_string((unsigned long long) s->calls);
    std::string state_path = prefix + "_state.json";
    std::string plan_path  = prefix + "_plan.json";

    const auto t0 = std::chrono::steady_clock::now();
    if (s->mode == "external") {
        elastic_online_state_dump(s, state_path);
        std::ostringstream cmd;
        cmd << shell_quote(s->python) << " " << shell_quote(s->solver)
            << " --model-meta " << shell_quote(s->model_meta_path)
            << " --cost-dir " << shell_quote(s->cost_dir)
            << " --state " << shell_quote(state_path)
            << " --budget-mib " << budget_mib
            << " --kv-mib " << s->kv_mib
            << " --misc-mib " << s->misc_mib
            << " --safety-mib " << s->safety_mib
            << " --time-limit-ms " << s->time_limit_ms
            << " --out " << shell_quote(plan_path)
            << " >/dev/null";
        int rc = std::system(cmd.str().c_str());
        if (rc != 0) {
            s->failures++;
            LOG_ERR("[elastic-online] external solver failed rc=%d budget=%lld cmd=%s\n",
                    rc, (long long) budget_mib, cmd.str().c_str());
            return nullptr;
        }
    } else if (s->mode == "remote") {
        elastic_online_state_dump(s, state_path);
        double remote_wall_ms = 0.0;
        double server_ms = 0.0;
        if (!elastic_online_generate_remote(s, budget_mib, state_path, plan_path, &remote_wall_ms, &server_ms)) {
            s->failures++;
            LOG_ERR("[elastic-online] remote solver failed budget=%lld url=%s\n",
                    (long long) budget_mib, s->remote_url.c_str());
            return nullptr;
        }
        s->remote_wall_ms_total += remote_wall_ms;
        s->remote_server_ms_total += server_ms;
        LOG_INF("[elastic-online] remote budget=%lld remote_wall_ms=%.3f server_solve_ms=%.3f\n",
                (long long) budget_mib, remote_wall_ms, server_ms);
    } else {
        if (!elastic_online_generate_native(s, budget_mib, plan_path)) {
            s->failures++;
            LOG_ERR("[elastic-online] native-greedy generator failed budget=%lld out=%s\n",
                    (long long) budget_mib, plan_path.c_str());
            return nullptr;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    llama_plan * plan = llama_plan_load_json(plan_path.c_str());
    if (!plan) {
        s->failures++;
        LOG_ERR("[elastic-online] failed to load generated plan %s\n", plan_path.c_str());
        return nullptr;
    }
    s->plans.push_back(plan);
    const double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG_INF("[elastic-online] budget=%lld generated %s mode=%s gen_ms=%.3f weights=%d ops=%d calls=%llu failures=%llu\n",
            (long long) budget_mib, plan_path.c_str(),
            s->mode.c_str(), gen_ms,
            llama_plan_n_weights(plan), llama_plan_n_ops(plan),
            (unsigned long long) s->calls,
            (unsigned long long) s->failures);
    return plan;
}

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting  = true;
            need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);

            // make sure all logs are flushed
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    common_params params;
    g_params = &params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    std::vector<common_chat_msg> chat_msgs;

    // Demo pin callback registered BEFORE model load so it's queried during weight setup.
    // op-schedule callback registered AFTER load (needs ctx) — done below.
    if (const char *e = std::getenv("LLAMA_TEST_SCHEDULE"); e && *e && *e != '0') {
        extern void llama_weight_pin_register(bool (*)(const char *, int, size_t, void *), void *);
        struct test_pin_state { uint64_t n_calls = 0; uint64_t n_pinned = 0; };
        static test_pin_state ps;
        llama_weight_pin_register([](const char *name, int /*layer*/, size_t byte_size, void *ud) -> bool {
            auto *p = (test_pin_state *)ud;
            p->n_calls++;
            const bool pin = byte_size < 10 * 1024 * 1024;
            if (pin) p->n_pinned++;
            return pin;
        }, &ps);
        std::atexit([]() {
            LOG_INF("[test-sched-pin] pin_calls=%llu pinned=%llu\n",
                    (unsigned long long)ps.n_calls, (unsigned long long)ps.n_pinned);
        });
    }

    // load the model and apply lora adapter, if any
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(params);

    model = llama_init.model.get();
    ctx = llama_init.context.get();

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    // ===== NEW: elastic plan framework (feature/elastic-plan-framework) =====
    // 用统一的 Plan IR → Execute 框架(llama_elastic_*),区别于下面 LLAMA_PLAN_DIR 的旧手写 demo。
    //   LLAMA_ELASTIC_APPLY=<plan.json>  : 加载一个 plan, llama_elastic_apply_plan (DoD#1)
    //   LLAMA_ELASTIC_DIR=<plans_dir>    : llama_elastic_enable("table",dir), 内存变化 online 换 plan (DoD#2)
    // 这两个 env 与 LLAMA_PLAN_DIR 互不影响(不同 env);设了新的就走新框架。
    if (const char * ap = std::getenv("LLAMA_ELASTIC_APPLY")) {
        llama_plan * plan = llama_plan_load_json(ap);
        if (plan) {
            int rc = llama_elastic_apply_plan(ctx, plan);
            LOG_INF("[elastic-fw] apply_plan(%s) rc=%d budget=%lldMiB weights=%d ops=%d\n",
                    ap, rc, (long long) llama_plan_budget_mib(plan),
                    llama_plan_n_weights(plan), llama_plan_n_ops(plan));
            // plan 须在推理期间存活;CLI 一次性进程,泄漏到退出即可。
        } else {
            LOG_ERR("[elastic-fw] failed to load plan %s\n", ap);
        }
    }
    if (const char * ed = std::getenv("LLAMA_ELASTIC_DIR")) {
        int rc = llama_elastic_enable(ctx, "table", ed);
        LOG_INF("[elastic-fw] enable(table, %s) rc=%d\n", ed, rc);
    }
    if (const char * eo = std::getenv("LLAMA_ELASTIC_ONLINE"); eo && *eo && *eo != '0') {
        static elastic_online_solver_state online;
        online.ctx = ctx;
        online.mode = std::getenv("LLAMA_ELASTIC_ONLINE_MODE") ? std::getenv("LLAMA_ELASTIC_ONLINE_MODE") : "native-greedy";
        online.python = std::getenv("LLAMA_ELASTIC_ONLINE_PYTHON") ? std::getenv("LLAMA_ELASTIC_ONLINE_PYTHON") : "python3";
        online.solver = std::getenv("LLAMA_ELASTIC_ONLINE_SOLVER") ? std::getenv("LLAMA_ELASTIC_ONLINE_SOLVER") : "runtime/plan/dynamic_budget_solver.py";
        online.remote_url = std::getenv("LLAMA_ELASTIC_ONLINE_REMOTE_URL") ? std::getenv("LLAMA_ELASTIC_ONLINE_REMOTE_URL") : "";
        online.model_meta_path = std::getenv("LLAMA_ELASTIC_MODEL_META") ? std::getenv("LLAMA_ELASTIC_MODEL_META") : "";
        online.cost_dir = std::getenv("LLAMA_ELASTIC_COST_DIR") ? std::getenv("LLAMA_ELASTIC_COST_DIR") : "";
        online.work_dir = std::getenv("LLAMA_ELASTIC_ONLINE_WORK_DIR") ? std::getenv("LLAMA_ELASTIC_ONLINE_WORK_DIR") : "/data/local/tmp/elastic/online";
        if (const char * e = std::getenv("LLAMA_ELASTIC_ONLINE_KV_MB")) online.kv_mib = std::atoi(e);
        if (const char * e = std::getenv("LLAMA_ELASTIC_ONLINE_MISC_MB")) online.misc_mib = std::atoi(e);
        if (const char * e = std::getenv("LLAMA_ELASTIC_ONLINE_SAFETY_MB")) online.safety_mib = std::atoi(e);
        if (const char * e = std::getenv("LLAMA_ELASTIC_ONLINE_TIME_LIMIT_MS")) online.time_limit_ms = std::atoi(e);
        if (const char * e = std::getenv("LLAMA_ELASTIC_ALLOW_CPU_FALLBACK")) online.allow_cpu_fallback = std::atoi(e) != 0;
        if (const char * e = std::getenv("LLAMA_ELASTIC_TRANSITION_WEIGHT")) online.transition_weight = std::atof(e);

        if (online.model_meta_path.empty() || online.cost_dir.empty()) {
            LOG_ERR("[elastic-online] LLAMA_ELASTIC_MODEL_META and LLAMA_ELASTIC_COST_DIR are required\n");
        } else {
            std::ifstream mf(online.model_meta_path);
            if (!mf) {
                LOG_ERR("[elastic-online] failed to open model meta %s\n", online.model_meta_path.c_str());
            } else {
                mf >> online.model_meta;
                if (std::ifstream sf(online.cost_dir + "/stage_costs.json"); sf) sf >> online.stage_costs;
                if (std::ifstream of(online.cost_dir + "/op_costs.json"); of) of >> online.op_costs;
                for (const auto & w : online.model_meta.value("weights", nlohmann::json::array())) {
                    std::string name = w.value("name", std::string());
                    if (!name.empty()) {
                        online.weight_by_name[name] = w;
                        online.weight_names.push_back(std::move(name));
                    }
                }
                online.item_bases.clear();
                online.item_bases.reserve(online.weight_names.size());
                for (const auto & w : online.model_meta.value("weights", nlohmann::json::array())) {
                    elastic_online_solver_state::item_base b;
                    b.id = w.value("weight_id", (int) online.item_bases.size());
                    b.name = w.value("name", std::string());
                    if (b.name.empty()) continue;
                    b.layer = w.value("layer", -1);
                    b.bytes = (size_t) w.value("byte_size", 0);
                    b.quant = w.value("quant", std::string());
                    const double gpu = elastic_backend_path_cost(&online, "GPU", b.name, b.bytes);
                    const double cpu = elastic_backend_path_cost(&online, "CPU", b.name, b.bytes);
                    b.backend = gpu <= cpu ? "GPU" : "CPU";
                    b.resident_value_gpu =
                        elastic_gpu_reload_cost(&online, b.name, b.bytes);
                    b.resident_value_cpu =
                        elastic_stage_cost(&online, "CPU_Elastic", "LOAD", b.name, b.bytes) +
                        elastic_stage_cost(&online, "CPU_Elastic", "XFORM", b.name, b.bytes);
                    online.item_bases.push_back(std::move(b));
                }
                if (!mkdir_p_local(online.work_dir)) {
                    LOG_ERR("[elastic-online] failed to create work_dir %s\n", online.work_dir.c_str());
                }

#if defined(_WIN32)
                _putenv_s("GGML_ELASTIC_CALLBACK_NOCACHE", "1");
#else
                setenv("GGML_ELASTIC_CALLBACK_NOCACHE", "1", 1);
#endif
                int rc = llama_elastic_enable(ctx, "callback", nullptr);
                llama_elastic_set_plan_provider(ctx, elastic_online_plan_provider, &online);
                LOG_INF("[elastic-online] enable(callback) rc=%d mode=%s weights=%zu solver=%s cost_dir=%s work_dir=%s\n",
                        rc, online.mode.c_str(), online.item_bases.size(), online.solver.c_str(),
                        online.cost_dir.c_str(), online.work_dir.c_str());
                std::atexit([]() {
                    for (llama_plan * p : online.plans) llama_plan_free(p);
                    LOG_INF("[elastic-online] calls=%llu failures=%llu remote_wall_ms=%.3f remote_server_ms=%.3f\n",
                            (unsigned long long) online.calls,
                            (unsigned long long) online.failures,
                            online.remote_wall_ms_total,
                            online.remote_server_ms_total);
                });
            }
        }
    }

    // ===== v8 demo: layer-partition via op_schedule (graph build time) =====
    // 比 op_runtime_dispatch 简单 — 走 ggml-sched 已有的 split mechanism 处理
    // cross-backend weight + activation, 不需要 v3-v6 的 runtime migration code.
    //
    // LLAMA_V8_PARTITION_LAYER=N: layer < N 的 ops → GPU, layer ≥ N → CPU
    //   N=0: 全 CPU (= -ngl 0 但 weights 在 GPU buffer cl_mem 占用没释放)
    //   N=28: 全 GPU (默认)
    //   N=14: 前 14 层 GPU, 后 14 层 CPU (动态 budget 紧时 PoC)
    //
    // 决策也用 LP study (project_lp_oracle_findings): Belady+PF16 给出的"哪些 op
    // 该在哪 backend"的 oracle. 这里只演示 layer-cut, 真接 LP 时把 layer < N 改成
    // partition[op_name] 查表.
    // ===== Plan scheduler: 加载离线 plan, 每次 budget 变化查表执行 =====
    // LLAMA_PLAN_DIR=<dir> 指向 make_plan.py 生成的 plans/ (index.json + plan_*.json)。
    // scheduler 每次内存 budget 变化时按 mem_avail_mb snap 到 band 切 current plan;
    //   - op_schedule: mul_mat 按 src[0] 的 weight 名查 plan.routes -> backend
    //   - weight_pin : 按 weight 名查 plan.resident_in_memory (常驻名单)
    //   - 切 band 时 evict 不再常驻的 / prefetch 新常驻的
    if (const char *pdir = std::getenv("LLAMA_PLAN_DIR")) {
        using json = nlohmann::json;
        struct PlanBand {
            double budget_mib = 0;
            int    ngl = 32;                                // partial-offload: 前 ngl 层 GPU, 其余 CPU
            std::unordered_map<std::string, int> routes;   // weight 名 -> backend_id
            std::unordered_set<std::string>      pins;      // 常驻 weight 名
        };
        struct PlanState {
            std::vector<PlanBand> bands;                    // 按 budget 升序
            std::atomic<int>      current{0};
            int cpu_id = 0, gpu_id = -1;
            bool force_gpu = false;
            uint64_t n_fire = 0, n_switch = 0, n_op_hit = 0;
            // ===== online residency-aware 模式 (LLAMA_ONLINE) =====
            // offline 是 stateless 查表: budget 每次跌都按新 band 把溢出层路由 CPU
            //   → 低 budget 期 token 跑 CPU(慢) + 回升时可能 reconvert。residency-blind。
            // online 知道权重物理上还在 UMA(16GB,模型 4.6GB,瞬时 budget 抖动并没把
            //   权重踢出),所以维持 high-water 常驻 ngl: budget 涨才升 GPU 层(必然
            //   convert 一次);瞬时跌不动(sticky);只有【持续】跌 debounce 次才真降。
            //   → 全程 GPU,避免 offline 的 CPU-fallback + reconvert churn。
            bool online = false;
            std::atomic<int> res_ngl{0};   // 当前常驻 GPU 层数 (high-water)
            int debounce = 4;              // 连续跌多少次才真降 (吸收瞬时震荡)
            int low_run = 0;               // 连续"目标<常驻"计数 (scheduler 单线程, 不需原子)
        };
        static PlanState ps;
        ps.cpu_id = llama_n_backends(ctx) - 1;
        ps.gpu_id = (llama_n_backends(ctx) > 1) ? 0 : -1;
        ps.force_gpu = std::getenv("LLAMA_PLAN_FORCE_GPU") != nullptr;  // 诊断: 全 GPU 路由
        ps.online    = std::getenv("LLAMA_ONLINE") != nullptr;
        if (const char *d = std::getenv("LLAMA_ONLINE_DEBOUNCE")) ps.debounce = std::atoi(d);
        std::string dir = pdir;
        try {
            std::ifstream idxf(dir + "/index.json");
            json idx; idxf >> idx;
            for (auto &it : idx["index"]) {
                std::ifstream pf(dir + "/" + it["file"].get<std::string>());
                if (!pf) continue;
                json pj; pf >> pj;
                PlanBand b;
                b.budget_mib = pj["budget_mib"].get<double>();
                b.ngl = pj.value("ngl", 32);   // partial-offload GPU 层数 (make_plan_q4 budget_to_ngl)
                for (auto &kv : pj["routes"].items())
                    b.routes[kv.key()] = (kv.value().get<std::string>() == "gpu") ? ps.gpu_id : ps.cpu_id;
                for (auto &p : pj["resident_in_memory"])
                    b.pins.insert(p["name"].get<std::string>());
                ps.bands.push_back(std::move(b));
            }
            std::sort(ps.bands.begin(), ps.bands.end(),
                      [](const PlanBand &a, const PlanBand &b) { return a.budget_mib < b.budget_mib; });
            LOG_INF("[plan] loaded %zu bands from %s (cpu=%d gpu=%d)\n",
                    ps.bands.size(), dir.c_str(), ps.cpu_id, ps.gpu_id);
        } catch (const std::exception &e) {
            LOG_ERR("[plan] load failed: %s\n", e.what());
        }
        if (!ps.bands.empty()) {
            // route_op: 默认【不】注册 —— per-op 路由跟 elastic GPU 流式共存会 cl_mem
            // 冲突崩溃 (clSetKernelArg -38, CL_INVALID_MEM_OBJECT). 选项1: 只用 pin/budget,
            // op 全留 GPU. LLAMA_PLAN_ROUTE=1 才开 per-op 路由(实验, 需配 V8 host_ptr fix).
            // LLAMA_PLAN_CONTIG=1 (推荐): 连续层分区路由 —— 当前 band 的前 ngl 层 → GPU,
            // 其余层 → CPU(读 mmap 免 convert). 只 1 个 cross-backend 边界(避免 per-op
            // 散路由的切换风暴, 每切换 ~38ms). 配 LLAMA_KEEP_GRAPH_REUSE=1 复用 graph,
            // band 切换时 scheduler 调 llama_graph_invalidate 重建一次. 实测 @3500 503ms
            // vs dynamic 984ms (no-retain, 输出正确).
            if (std::getenv("LLAMA_PLAN_CONTIG")) {
            llama_set_op_schedule(ctx, [](const struct ggml_tensor * /*node*/, const char * /*name*/,
                                          int layer, void *ud) -> int {
                auto *s = (PlanState *)ud;
                if (layer < 0) return -1;                               // embd/head/output → 默认 GPU
                if (s->force_gpu) { s->n_op_hit++; return s->gpu_id; }
                s->n_op_hit++;
                // online: 用 high-water res_ngl(sticky); offline: 用当前 band 的 ngl(stateless)
                const int cur_ngl = s->online ? s->res_ngl.load()
                                              : s->bands[s->current.load()].ngl;
                return (layer < cur_ngl) ? s->gpu_id : s->cpu_id;      // 连续: 前 ngl 层 GPU
            }, &ps);
            } else if (std::getenv("LLAMA_PLAN_ROUTE")) {
            // (旧)per-op 散路由: 按 weight 名查 plan.routes. 切换风暴慢 4015ms, 仅诊断.
            llama_set_op_schedule(ctx, [](const struct ggml_tensor *node, const char * /*name*/,
                                          int /*layer*/, void *ud) -> int {
                auto *s = (PlanState *)ud;
                if (!node || node->op != GGML_OP_MUL_MAT) return -1;
                if (s->force_gpu) { s->n_op_hit++; return s->gpu_id; }   // 诊断: 全部 GPU
                if (!node->src[0]) return -1;
                const char *w = node->src[0]->name;
                if (!w || !*w) return -1;
                const PlanBand &b = s->bands[s->current.load()];
                auto it = b.routes.find(w);
                if (it == b.routes.end()) return -1;
                s->n_op_hit++;
                return it->second;
            }, &ps);
            }
            // weight_pin: pin 住 GPU 常驻的 ngl 层 → elastic 不会因 budget 抖动把它们
            // evict 重转(churn)。这是"静态 partial offload"的关键: GPU 层 convert 一次、
            // 常驻; 溢出层走 CPU 读 mmap。CONTIG/online 按 layer<ngl pin(GPU 集);
            // 否则回退 plan 名单。(LLAMA_PLAN_NOPIN=1 关掉做诊断, 会退化成 churn)
            if (!std::getenv("LLAMA_PLAN_NOPIN")) {
            const bool contig = std::getenv("LLAMA_PLAN_CONTIG") != nullptr;
            if (contig || ps.online) {
            llama_set_weight_pin(ctx, [](const char * /*name*/, int layer, size_t /*sz*/,
                                         void *ud) -> bool {
                auto *s = (PlanState *)ud;
                if (layer < 0) return true;                         // embd/head/norm 常驻 GPU
                const int ngl = s->online ? s->res_ngl.load()
                                          : s->bands[s->current.load()].ngl;
                return layer < ngl;                                 // GPU 路由的层 → pin 住
            }, &ps);
            } else {
            llama_set_weight_pin(ctx, [](const char *name, int /*layer*/, size_t /*sz*/,
                                         void *ud) -> bool {
                auto *s = (PlanState *)ud;
                return name && s->bands[s->current.load()].pins.count(name) > 0;
            }, &ps);
            }
            }
            // scheduler: 内存变化触发 -> snap band -> 切 plan -> evict/prefetch 差异
            llama_set_scheduler(ctx, [](struct llama_context *c,
                                        const struct llama_runtime_state *st, void *ud) {
                auto *s = (PlanState *)ud;
                s->n_fire++;
                int idx = 0;   // 最大 budget <= 当前内存 的 band(保证驻留 <= budget)
                for (int i = 0; i < (int)s->bands.size(); ++i)
                    if (s->bands[i].budget_mib <= (double)st->mem_avail_mb) idx = i;
                // ===== online residency-aware: sticky high-water ngl + debounce =====
                if (s->online) {
                    // residency-aware target: 直接按真实 footprint 算能放多少层
                    // (offline 的 band.ngl 过度保留 headroom → 多路 CPU → 慢)。
                    // 模型物理上能放进设备 RAM(16GB vs 模型 4.6GB), 故高 budget 时
                    // 可全 GPU(32 层), 瞬时跌靠 high-water + debounce 骑过去, 不重路 CPU。
                    static const int   ONL_NLAYER = []{ const char*e=getenv("LLAMA_ONLINE_NLAYER"); return e?atoi(e):32; }();
                    static const double ONL_BASE  = []{ const char*e=getenv("LLAMA_ONLINE_BASE");   return e?atof(e):300.0; }();
                    static const double ONL_PER   = []{ const char*e=getenv("LLAMA_ONLINE_PER");    return e?atof(e):117.0; }();
                    int fit = (int)(((double)st->mem_avail_mb - ONL_BASE) / ONL_PER);
                    if (fit < 0) fit = 0;
                    if (fit > ONL_NLAYER) fit = ONL_NLAYER;
                    const int target = fit;
                    const int cur    = s->res_ngl.load();
                    bool changed = false;
                    if (target > cur) {                 // budget 涨 → 升 GPU 层(convert 一次)
                        s->res_ngl.store(target); s->low_run = 0; changed = true;
                    } else if (target < cur) {          // budget 跌 → 先不动, 攒 debounce
                        if (++s->low_run >= s->debounce) {
                            s->res_ngl.store(target); s->low_run = 0; changed = true;
                        }
                    } else { s->low_run = 0; }
                    s->current.store(idx);
                    if (changed) { s->n_switch++; llama_graph_invalidate(c); }
                    LOG_INF("[online] fire#%llu step=%llu mem=%lld MB target_ngl=%d res_ngl=%d%s\n",
                            (unsigned long long)s->n_fire, (unsigned long long)st->decode_step,
                            (long long)st->mem_avail_mb, target, s->res_ngl.load(),
                            changed ? " (MIGRATE)" : "");
                    return;
                }
                int prev = s->current.load();
                if (idx != prev) {
                    s->n_switch++;
                    s->current.store(idx);
                    // band 变 → routing(ngl)变 → 强制重建 graph 一次(配 LLAMA_KEEP_GRAPH_REUSE
                    // 平时复用; 不调则 cached graph 保持旧 band 的路由 → 错). 见 llama_graph_invalidate.
                    llama_graph_invalidate(c);
                    // 主动迁移默认【关】—— 一次 prefetch 整个 pin set(上百个 weight)会
                    // 跟 elastic budget/pool 冲突崩. 让 elastic 按 budget+weight_pin 自然 reload.
                    // LLAMA_PLAN_MOVE=1 才开(需限量/分批, 见选项3).
                    if (std::getenv("LLAMA_PLAN_MOVE")) {
                        const auto &nb = s->bands[idx];
                        const auto &ob = s->bands[prev];
                        for (const auto &w : ob.pins) if (!nb.pins.count(w)) llama_weight_request_evict(c, w.c_str());
                        for (const auto &w : nb.pins) if (!ob.pins.count(w)) llama_weight_request_prefetch(c, w.c_str());
                    }
                    s->current.store(idx);
                }
                LOG_INF("[plan] fire#%llu step=%llu mem=%lld MB delta=%+lld -> band %.0f MiB%s\n",
                        (unsigned long long)s->n_fire, (unsigned long long)st->decode_step,
                        (long long)st->mem_avail_mb, (long long)st->mem_delta_mb,
                        s->bands[idx].budget_mib, idx != prev ? " (SWITCH)" : "");
            }, &ps);
            std::atexit([]() {
                LOG_INF("[plan] fires=%llu switches=%llu op_routed=%llu final_band=%.0f MiB\n",
                        (unsigned long long)ps.n_fire, (unsigned long long)ps.n_switch,
                        (unsigned long long)ps.n_op_hit,
                        ps.bands.empty() ? 0.0 : ps.bands[ps.current.load()].budget_mib);
            });
        }
    }

    if (const char *pv8 = std::getenv("LLAMA_V8_PARTITION_LAYER")) {
        struct v8_state {
            int partition_layer = 0;
            int cpu_id = 0;
            int gpu_id = -1;
            uint64_t n_ops_cpu = 0;
            uint64_t n_ops_gpu = 0;
        };
        static v8_state v8s;
        v8s.partition_layer = std::atoi(pv8);
        v8s.cpu_id = llama_n_backends(ctx) - 1;
        v8s.gpu_id = (llama_n_backends(ctx) > 1) ? 0 : -1;
        LOG_INF("[v8-partition] layer < %d → GPU, ≥ → CPU (n_backends=%d cpu=%d gpu=%d)\n",
                v8s.partition_layer, llama_n_backends(ctx), v8s.cpu_id, v8s.gpu_id);

        // v9: 主动 evict partition-out 层的 weight cl_mem (释放 GPU 内存).
        // 这些 weight 由 v8.4 host_ptr fallback 从 mmap 读, 不需要 cl_mem.
        // 给 elastic backend 让出 budget headroom 给真正 GPU 上跑的 layer.
        // env LLAMA_V9_EVICT_OUT=1 启用 (默认关, 因为需要 elastic mode 才有 evict API).
        if (std::getenv("LLAMA_V9_EVICT_OUT")) {
            uint64_t n_evicted = 0;
            const char *tensors[] = {"attn_q", "attn_k", "attn_v", "attn_output",
                                     "attn_norm", "ffn_norm",
                                     "ffn_gate", "ffn_up", "ffn_down"};
            for (int L = v8s.partition_layer; L < llama_model_n_layer(model); L++) {
                for (auto t : tensors) {
                    char nm[64]; std::snprintf(nm, sizeof(nm), "blk.%d.%s.weight", L, t);
                    if (llama_weight_request_evict(ctx, nm) == 0) {
                        n_evicted++;
                    }
                }
            }
            LOG_INF("[v9-evict] partition-out (layer ≥ %d) weights evicted: %llu\n",
                    v8s.partition_layer, (unsigned long long)n_evicted);
        }

        llama_set_op_schedule(ctx, [](const struct ggml_tensor */*node*/, const char */*name*/,
                                      int layer, void *ud) -> int {
            auto *s = (v8_state *)ud;
            if (layer < 0) return -1;  // 非 layer op (output / embed), 默认
            int target = (layer < s->partition_layer) ? s->gpu_id : s->cpu_id;
            if (target == s->gpu_id) s->n_ops_gpu++;
            else                     s->n_ops_cpu++;
            return target;
        }, &v8s);

        std::atexit([]() {
            LOG_INF("[v8-partition] ops routed: GPU=%llu  CPU=%llu\n",
                    (unsigned long long)v8s.n_ops_gpu,
                    (unsigned long long)v8s.n_ops_cpu);
        });
    }

    // ===== v8.2: smarter partition policies (基于 LP study + phase1 PIN= insight) =====
    // LLAMA_V8_PARTITION=<policy>:
    //   ffn-cpu       — 所有 ffn ops → CPU, attn ops → GPU
    //                   (用 phase1 "PIN=norm,k,v,q" 思路: 小 weights 留 GPU, 大 ffn 移 CPU)
    //   attn-cpu      — 反过来: ffn 留 GPU, attn 移 CPU
    //   layer-mod-N   — 每 N 层 1 个 CPU (稀疏分布), e.g., N=4 → layer 3, 7, 11, ... 在 CPU
    //   first-K-cpu   — 前 K 层 → CPU (early-layer offload, 余下 GPU)
    if (const char *pv82 = std::getenv("LLAMA_V8_PARTITION")) {
        struct v82_state {
            std::string policy;
            int param = 4;
            int cpu_id = 0;
            int gpu_id = -1;
            uint64_t n_cpu = 0, n_gpu = 0;
        };
        static v82_state v82s;
        v82s.policy = pv82;
        if (const char *p = std::getenv("LLAMA_V8_PARTITION_PARAM")) v82s.param = std::atoi(p);
        v82s.cpu_id = llama_n_backends(ctx) - 1;
        v82s.gpu_id = (llama_n_backends(ctx) > 1) ? 0 : -1;
        LOG_INF("[v8.2-partition] policy=%s param=%d cpu=%d gpu=%d\n",
                v82s.policy.c_str(), v82s.param, v82s.cpu_id, v82s.gpu_id);

        llama_set_op_schedule(ctx, [](const struct ggml_tensor */*node*/, const char *name,
                                       int layer, void *ud) -> int {
            auto *s = (v82_state *)ud;
            if (layer < 0 || !name) return -1;
            const std::string &pol = s->policy;
            int target = -1;
            bool is_ffn = (strncmp(name, "ffn_", 4) == 0);
            // 只路由 attn 中带 weight 的 (Q/K/V proj + output proj), 不路由 kq/kqv
            // (activation×activation, 没 weight, 跨 backend 自动 copy 在 ggml-sched
            // split 里 OK 但实际有 case 崩, v8.3 再调).
            bool is_attn = (strncmp(name, "Qcur", 4) == 0 || strncmp(name, "Kcur", 4) == 0
                            || strncmp(name, "Vcur", 4) == 0 || strncmp(name, "attn_out", 8) == 0);
            if (pol == "ffn-cpu") {
                if (is_ffn) target = s->cpu_id;
            } else if (pol == "attn-cpu") {
                if (is_attn) target = s->cpu_id;
            } else if (pol == "layer-mod-N") {
                if (s->param > 0 && (layer % s->param) == (s->param - 1)) target = s->cpu_id;
            } else if (pol == "first-K-cpu") {
                if (layer < s->param) target = s->cpu_id;
            }
            if (target == s->cpu_id) s->n_cpu++;
            else if (target != -1)   s->n_gpu++;
            return target;
        }, &v82s);

        std::atexit([]() {
            LOG_INF("[v8.2-partition] policy=%s ops: CPU=%llu  GPU=%llu\n",
                    v82s.policy.c_str(),
                    (unsigned long long)v82s.n_cpu,
                    (unsigned long long)v82s.n_gpu);
        });
    }

    // ===== v8.3: dynamic re-partition on memory event =====
    // Scheduler 看 MemAvailable 变化 → 算新 partition layer → 触发下次 decode 重 build.
    // 三档:
    //   mem > HI_MB (默认 3000): baseline (全 GPU)
    //   mem 中间: layer<MID (默认 20, 8 层 CPU)
    //   mem < LO_MB (默认 1000): 重 offload, layer<LO_LAYER (默认 8)
    // env: LLAMA_V83_HI_MB / _MID_MB / _LO_MB / _MID_LAYER / _LO_LAYER
    if (const char *e = std::getenv("LLAMA_V8_DYNAMIC"); e && *e && *e != '0') {
        struct v83_state {
            int hi_mb = 3000, mid_mb = 1500, lo_mb = 1000;
            int mid_layer = 20, lo_layer = 8;
            int n_layers = 28;
            int cpu_id = 0, gpu_id = -1;
            std::atomic<int> current_partition{28};  // 28 = 全 GPU
            uint64_t n_repartitions = 0;
        };
        static v83_state v83s;
        if (const char *p = std::getenv("LLAMA_V83_HI_MB"))    v83s.hi_mb    = std::atoi(p);
        if (const char *p = std::getenv("LLAMA_V83_MID_MB"))   v83s.mid_mb   = std::atoi(p);
        if (const char *p = std::getenv("LLAMA_V83_LO_MB"))    v83s.lo_mb    = std::atoi(p);
        if (const char *p = std::getenv("LLAMA_V83_MID_LAYER")) v83s.mid_layer = std::atoi(p);
        if (const char *p = std::getenv("LLAMA_V83_LO_LAYER"))  v83s.lo_layer  = std::atoi(p);
        v83s.n_layers = llama_model_n_layer(model);
        v83s.cpu_id = llama_n_backends(ctx) - 1;
        v83s.gpu_id = (llama_n_backends(ctx) > 1) ? 0 : -1;
        LOG_INF("[v8.3-dynamic] init: layers=%d backends(cpu=%d gpu=%d)\n",
                v83s.n_layers, v83s.cpu_id, v83s.gpu_id);
        LOG_INF("[v8.3-dynamic] thresholds: HI=%d MID=%d LO=%d MB | MID_L=%d LO_L=%d\n",
                v83s.hi_mb, v83s.mid_mb, v83s.lo_mb, v83s.mid_layer, v83s.lo_layer);

        // Op_schedule reads current_partition
        llama_set_op_schedule(ctx, [](const struct ggml_tensor */*node*/, const char */*name*/,
                                       int layer, void *ud) -> int {
            auto *s = (v83_state *)ud;
            if (layer < 0) return -1;
            int p = s->current_partition.load();
            return (layer < p) ? s->gpu_id : s->cpu_id;
        }, &v83s);

        // Scheduler reacts to memory changes
        llama_set_memory_watch_threshold(ctx, 100);  // 100 MB delta
        llama_set_scheduler(ctx, [](struct llama_context *c,
                                      const struct llama_runtime_state *st,
                                      void *ud) {
            auto *s = (v83_state *)ud;
            int64_t mem = st->mem_avail_mb;
            int new_p;
            if (mem >= s->hi_mb) new_p = s->n_layers;       // 全 GPU
            else if (mem >= s->lo_mb) new_p = s->mid_layer; // 中等 offload
            else new_p = s->lo_layer;                       // 重 offload
            int old_p = s->current_partition.load();
            if (new_p != old_p) {
                s->current_partition.store(new_p);
                s->n_repartitions++;
                LOG_INF("[v8.3-dynamic] step=%llu mem=%lld MB → repartition layer<%d (was <%d)\n",
                        (unsigned long long)st->decode_step, (long long)mem, new_p, old_p);
            }
            (void)c;
        }, &v83s);

        std::atexit([]() {
            LOG_INF("[v8.3-dynamic] total repartitions=%llu (final layer<%d)\n",
                    (unsigned long long)v83s.n_repartitions,
                    v83s.current_partition.load());
        });
    }

    // ===== Demo: schedule callback (LLAMA_TEST_SCHEDULE=1) =====
    // 演示 op-schedule + weight-pin callback API. 真实 LP solver 接入时用类似 pattern.
    if (const char *e = std::getenv("LLAMA_TEST_SCHEDULE"); e && *e && *e != '0') {
        struct test_sched_state {
            int n_backends = 0;
            int cpu_id = 0;     // 假定最后一个 backend = CPU (llama convention)
            int gpu_id = 0;     // 假定第 0 个 = GPU (有 -ngl 时)
            uint64_t n_op_calls = 0;
            uint64_t n_pin_calls = 0;
            uint64_t n_pinned = 0;
            uint64_t n_to_cpu = 0;
            uint64_t n_to_gpu = 0;
        };
        static test_sched_state ts;
        ts.n_backends = llama_n_backends(ctx);
        ts.cpu_id = ts.n_backends - 1;        // llama 约定 CPU 在最后
        ts.gpu_id = (ts.n_backends > 1) ? 0 : -1;  // 第一个非 CPU
        LOG_INF("[test-sched] registered. n_backends=%d cpu_id=%d gpu_id=%d\n",
                ts.n_backends, ts.cpu_id, ts.gpu_id);
        for (int i = 0; i < ts.n_backends; i++) {
            LOG_INF("[test-sched] backend[%d] = %s\n", i, llama_backend_name(ctx, i));
        }

        // Op-schedule: attn ops 到 GPU, ffn ops 到 CPU (示意)
        llama_set_op_schedule(ctx, [](const struct ggml_tensor *node, const char *name,
                                       int layer, void *ud) -> int {
            auto *s = (test_sched_state *)ud;
            s->n_op_calls++;
            if (s->gpu_id < 0) return -1;
            // 简单规则: name 含 "attn" → GPU, 含 "ffn" → CPU, 其它默认
            if (name && strstr(name, "attn")) { s->n_to_gpu++; return s->gpu_id; }
            if (name && strstr(name, "ffn"))  { s->n_to_cpu++; return s->cpu_id; }
            return -1;
        }, &ts);

        // Weight-pin: 大 weight (>=10 MB) 不 pin, 小的 pin (避免 reload)
        llama_set_weight_pin(ctx, [](const char *name, int layer, size_t byte_size,
                                       void *ud) -> bool {
            auto *s = (test_sched_state *)ud;
            s->n_pin_calls++;
            const bool pin = byte_size < 10 * 1024 * 1024;  // < 10 MB → pin
            if (pin) s->n_pinned++;
            return pin;
        }, &ts);

        // 退出时打印统计
        std::atexit([]() {
            LOG_INF("[test-sched] stats: op_calls=%llu (cpu=%llu gpu=%llu) pin_calls=%llu pinned=%llu\n",
                    (unsigned long long)ts.n_op_calls,
                    (unsigned long long)ts.n_to_cpu,
                    (unsigned long long)ts.n_to_gpu,
                    (unsigned long long)ts.n_pin_calls,
                    (unsigned long long)ts.n_pinned);
        });
    }

    // ===== Demo: runtime scheduler (LLAMA_TEST_RUNTIME_SCHED=1) =====
    // 每次 decode 之前 sample MemAvailable. 变化 >= LLAMA_RUNTIME_SCHED_THRESH_MB (默认 100)
    // 触发回调. 回调演示:
    //   - mem 下降时, 调 llama_weight_request_evict 把名字含 "ffn_up" 的 weight 主动 evict
    //   - mem 上升时, 调 llama_weight_request_prefetch 把当前 layer 的 attn weight 预 load
    //   - 同时切 op_schedule_fn 改路由 (mem 紧时 ffn→CPU 让 GPU 缓口气)
    if (const char *e = std::getenv("LLAMA_TEST_RUNTIME_SCHED"); e && *e && *e != '0') {
        int thresh = 100;
        if (const char *t = std::getenv("LLAMA_RUNTIME_SCHED_THRESH_MB")) thresh = std::atoi(t);
        llama_set_memory_watch_threshold(ctx, thresh);
        struct rt_state {
            uint64_t n_fires = 0;
            uint64_t n_evict = 0;
            uint64_t n_pref  = 0;
            // op-backend routing state (mutated by scheduler):
            int      cpu_id = -1;
            int      gpu_id = -1;
            int      ffn_target = -1;   // 当前 ffn op 目标 backend, -1 = default
            uint64_t n_ffn_to_cpu = 0;
            uint64_t n_ffn_to_gpu = 0;
            uint64_t n_op_calls  = 0;
        };
        static rt_state rs;
        rs.cpu_id = llama_n_backends(ctx) - 1;
        rs.gpu_id = (llama_n_backends(ctx) > 1) ? 0 : -1;

        // op_schedule_fn 在每次 graph build 时跑 (graph_reuse_disable=1 由
        // llama_set_scheduler 自动设). 它读 rs.ffn_target 决定 ffn 上哪个 backend.
        llama_set_op_schedule(ctx, [](const struct ggml_tensor */*node*/, const char *name,
                                       int /*layer*/, void *ud) -> int {
            auto *s = (rt_state *)ud;
            s->n_op_calls++;
            if (!name) return -1;
            if (strstr(name, "ffn")) {
                if (s->ffn_target >= 0) {
                    if (s->ffn_target == s->cpu_id) s->n_ffn_to_cpu++;
                    else if (s->ffn_target == s->gpu_id) s->n_ffn_to_gpu++;
                    return s->ffn_target;
                }
            }
            return -1;
        }, &rs);

        llama_set_scheduler(ctx, [](struct llama_context *c,
                                     const struct llama_runtime_state *st,
                                     void *ud) {
            auto *s = (rt_state *)ud;
            s->n_fires++;

            // Policy:
            //   mem_avail < 4096 MB (紧): ffn → CPU 卸载 GPU 压力, evict ffn_up
            //   mem_avail > 8192 MB (松): ffn → GPU 抢回去, 预 prefetch attn
            //   中间区: 保持
            //   LLAMA_RUNTIME_SCHED_LO_MB / _HI_MB 覆盖阈值
            int lo = 4096, hi = 8192;
            if (const char *e = std::getenv("LLAMA_RUNTIME_SCHED_LO_MB")) lo = std::atoi(e);
            if (const char *e = std::getenv("LLAMA_RUNTIME_SCHED_HI_MB")) hi = std::atoi(e);
            int new_target = s->ffn_target;
            if (st->mem_avail_mb < lo) {
                new_target = s->cpu_id;
            } else if (st->mem_avail_mb > hi || st->decode_step == 1) {
                new_target = s->gpu_id;
            }
            bool routing_changed = (new_target != s->ffn_target);
            s->ffn_target = new_target;

            LOG_INF("[runtime-sched] fire #%llu  step=%llu  mem=%lld MB  delta=%+lld MB  ffn→%s%s\n",
                    (unsigned long long)s->n_fires,
                    (unsigned long long)st->decode_step,
                    (long long)st->mem_avail_mb, (long long)st->mem_delta_mb,
                    (s->ffn_target < 0 ? "default" :
                     (s->ffn_target == s->cpu_id ? "CPU" : "GPU")),
                    routing_changed ? " (CHANGED)" : "");

            // Weight movement (仅在 LLAMA_RUNTIME_SCHED_WEIGHT_MOVE=1 时主动 evict/prefetch.
            // 默认不动 weight, 因为粗 demo policy 会破坏 correctness — 真接 LP solver
            // 时 weight 选择基于精确 use-序列, 安全.)
            const bool wm_on = std::getenv("LLAMA_RUNTIME_SCHED_WEIGHT_MOVE") != nullptr;
            if (wm_on) {
                const char *patterns_ev[] = { "ffn_up.weight" };
                const char *patterns_pf[] = { "attn_q.weight", "attn_k.weight" };
                if (st->mem_delta_mb < -50) {
                    for (int l = 0; l < 4; l++) {
                        for (auto p : patterns_ev) {
                            char nm[64]; std::snprintf(nm, sizeof(nm), "blk.%d.%s", l, p);
                            if (llama_weight_is_resident(c, nm)) {
                                if (llama_weight_request_evict(c, nm) == 0) s->n_evict++;
                            }
                        }
                    }
                } else if (st->mem_delta_mb > +50) {
                    for (int l = 0; l < 4; l++) {
                        for (auto p : patterns_pf) {
                            char nm[64]; std::snprintf(nm, sizeof(nm), "blk.%d.%s", l, p);
                            if (!llama_weight_is_resident(c, nm)) {
                                if (llama_weight_request_prefetch(c, nm) == 0) s->n_pref++;
                            }
                        }
                    }
                }
            }
        }, &rs);

        std::atexit([]() {
            LOG_INF("[runtime-sched] fires=%llu evicts=%llu prefetches=%llu  ffn_routing(cpu=%llu gpu=%llu)  total_op_calls=%llu\n",
                    (unsigned long long)rs.n_fires,
                    (unsigned long long)rs.n_evict,
                    (unsigned long long)rs.n_pref,
                    (unsigned long long)rs.n_ffn_to_cpu,
                    (unsigned long long)rs.n_ffn_to_gpu,
                    (unsigned long long)rs.n_op_calls);
        });
    }

    // ===== Demo: TRUE per-op runtime dispatch (LLAMA_TEST_OP_RUNTIME_DISPATCH=<policy>) =====
    // 真 runtime per-op 决策: 每个 op 即将 compute 前 hook 触发, 可基于当前 state
    // (上一 op 时间, op 计数器, 当前 op 类型) 即时选 backend.
    //
    // policy 选项 (env value):
    //   "alternate" / "1"     : 偶数 mul_mat → CPU (default for backward compat)
    //   "layer-half"          : layer < N/2 → GPU, ≥ N/2 → CPU (适合 GPU 内存紧时)
    //   "ffn-cpu"             : 所有 ffn ops → CPU
    //   "attn-cpu"            : 所有 attn ops → CPU
    //   "memory-driven"       : 看当前 MemAvailable, < LO_MB 时所有 mul_mat → CPU
    if (const char *e = std::getenv("LLAMA_TEST_OP_RUNTIME_DISPATCH"); e && *e && *e != '0') {
        struct dispatch_state {
            uint64_t n_calls       = 0;
            uint64_t n_overrides   = 0;
            uint64_t n_mulmat      = 0;
            uint64_t n_mulmat_cpu  = 0;
            uint64_t n_mulmat_gpu  = 0;
            int cpu_id = 0;
            int gpu_id = -1;
            std::string policy;
            int mem_lo = 4096;
            int n_layers = 28;  // 用 llama_model_n_layer 实际取
            struct llama_context *ctx_ref = nullptr;  // 给 elastic-aware 用
        };
        static dispatch_state ds;
        ds.cpu_id = llama_n_backends(ctx) - 1;
        ds.gpu_id = (llama_n_backends(ctx) > 1) ? 0 : -1;
        ds.policy = std::string(e);
        ds.ctx_ref = ctx;
        if (const char *p = std::getenv("LLAMA_OP_DISPATCH_LO_MB")) ds.mem_lo = std::atoi(p);
        ds.n_layers = llama_model_n_layer(model);
        LOG_INF("[op-runtime-dispatch] policy=%s cpu_id=%d gpu_id=%d n_layers=%d\n",
                ds.policy.c_str(), ds.cpu_id, ds.gpu_id, ds.n_layers);

        llama_set_op_runtime_dispatch(ctx, [](const struct ggml_tensor *op,
                                                int default_backend_id, int n_backends,
                                                void *ud) -> int {
            auto *s = (dispatch_state *)ud;
            s->n_calls++;
            (void)n_backends;
            if (!op) return -1;
            // dump first N op names for debugging
            static int dump_count = 0;
            if (std::getenv("LLAMA_OP_DISPATCH_DUMP_NAMES") && dump_count < 30 && op->op == GGML_OP_MUL_MAT) {
                fprintf(stderr, "[op-name-dump] op=%s name='%s'\n",
                        ggml_op_name(op->op), op->name);
                dump_count++;
            }

            int target = -1;
            const std::string &pol = s->policy;
            const char *name = op->name;

            // op naming convention (decode graph): "Qcur-N", "Kcur-N", "Vcur-N",
            // "kq-N", "kqv-N", "attn_out-N", "ffn_gate-N", "ffn_up-N", "ffn_out-N"
            // 解析 layer: 找最后一个 '-' 之后的数字
            auto parse_layer = [](const char *nm) -> int {
                if (!nm) return -1;
                const char *dash = strrchr(nm, '-');
                if (!dash) return -1;
                int l = -1;
                if (sscanf(dash + 1, "%d", &l) == 1) return l;
                return -1;
            };
            auto is_attn = [](const char *nm) -> bool {
                if (!nm) return false;
                return strncmp(nm, "Qcur", 4) == 0 || strncmp(nm, "Kcur", 4) == 0
                    || strncmp(nm, "Vcur", 4) == 0 || strncmp(nm, "kq", 2) == 0
                    || strncmp(nm, "kqv", 3) == 0 || strstr(nm, "attn_out") != nullptr;
            };
            auto is_ffn = [](const char *nm) -> bool {
                if (!nm) return false;
                return strncmp(nm, "ffn_", 4) == 0;
            };

            if (pol == "alternate" || pol == "1") {
                if (op->op == GGML_OP_MUL_MAT) {
                    s->n_mulmat++;
                    target = (s->n_mulmat % 2 == 0) ? s->cpu_id : -1;
                }
            } else if (pol == "layer-half") {
                if (op->op == GGML_OP_MUL_MAT) {
                    s->n_mulmat++;
                    int layer = parse_layer(name);
                    if (layer >= 0) {
                        target = (layer < s->n_layers / 2) ? -1 : s->cpu_id;
                    }
                }
            } else if (pol == "ffn-cpu") {
                if (op->op == GGML_OP_MUL_MAT && is_ffn(name)) {
                    s->n_mulmat++;
                    target = s->cpu_id;
                }
            } else if (pol == "attn-cpu") {
                if (op->op == GGML_OP_MUL_MAT && is_attn(name)) {
                    s->n_mulmat++;
                    target = s->cpu_id;
                }
            } else if (pol == "memory-driven") {
                if (op->op == GGML_OP_MUL_MAT) {
                    s->n_mulmat++;
                    int64_t avail = llama_runtime_mem_avail_mb();
                    target = (avail < s->mem_lo) ? s->cpu_id : -1;
                }
            } else if (pol == "elastic-aware") {
                // v6 动态 budget: 看 weight 在 GPU 是否常驻, 非常驻 → CPU.
                // ggml-backend migration 用 llama_weight_host_ptr_query 拿 mmap 源
                // ptr 直接 memcpy (绕过 cl_mem release 问题).
                //
                // 跳过 list: result_output / token_embd (特殊处理 op, 跨 backend 难)
                //          + 只切 layer-* 的 mul_mat
                static uint64_t n_resident_kept = 0, n_evicted_routed = 0;
                if (op->op == GGML_OP_MUL_MAT && op->src[0] && name) {
                    s->n_mulmat++;
                    // 只考虑 weight×activation 的 mul_mat (跳 activation×activation 的
                    // kq/kqv — src[0] 不是 weight, host_ptr 拿不到). 也跳 result_output.
                    bool is_weight_mulmat =
                        (strncmp(name, "Qcur-", 5) == 0 || strncmp(name, "Kcur-", 5) == 0
                         || strncmp(name, "Vcur-", 5) == 0 || strncmp(name, "attn_out", 8) == 0
                         || strncmp(name, "ffn_", 4) == 0);
                    if (is_weight_mulmat) {
                        const char *w_name = op->src[0]->name;
                        if (w_name && w_name[0]) {
                            bool resident = llama_weight_is_resident(s->ctx_ref, w_name);
                            if (resident) {
                                n_resident_kept++;
                            } else {
                                n_evicted_routed++;
                                target = s->cpu_id;
                                if (std::getenv("LLAMA_OP_DISPATCH_ELASTIC_DEBUG") && n_evicted_routed < 20) {
                                    LOG_INF("[elastic-aware] %s evicted → CPU\n", w_name);
                                }
                            }
                        }
                    }
                }
                (void)n_resident_kept;
            } else if (pol == "smart-pressure") {
                // v5 smart policy: 只在 GPU 真有压力时挑 ffn 切 CPU.
                // 优先 ffn (mul_mat 中最大 weight, 切 1 个省 GPU 内存最多),
                // 只切前 K 个 layer 的 ffn (避免雪崩).
                // env: LLAMA_OP_DISPATCH_PRESSURE_MB (default 1500),
                //     LLAMA_OP_DISPATCH_MAX_FFN_LAYERS (default 4).
                static int pressure_lo = -1;
                static int max_ffn_layers = -1;
                if (pressure_lo < 0) {
                    pressure_lo = 1500;
                    if (const char *e = std::getenv("LLAMA_OP_DISPATCH_PRESSURE_MB")) pressure_lo = std::atoi(e);
                    if (const char *e = std::getenv("LLAMA_OP_DISPATCH_MAX_FFN_LAYERS")) max_ffn_layers = std::atoi(e); else max_ffn_layers = 4;
                }
                if (op->op == GGML_OP_MUL_MAT && is_ffn(name)) {
                    s->n_mulmat++;
                    int64_t avail = llama_runtime_mem_avail_mb();
                    int layer = parse_layer(name);
                    // 只切前 max_ffn_layers 层的 ffn (避免全切)
                    if (avail < pressure_lo && layer >= 0 && layer < max_ffn_layers) {
                        target = s->cpu_id;
                    }
                }
            }

            if (target == s->cpu_id) {
                s->n_mulmat_cpu++;
                if (target != default_backend_id) s->n_overrides++;
                return target;
            }
            if (op->op == GGML_OP_MUL_MAT) s->n_mulmat_gpu++;
            return -1;
        }, &ds);

        std::atexit([]() {
            LOG_INF("[op-runtime-dispatch] policy=%s calls=%llu overrides=%llu  mul_mat(total=%llu cpu=%llu gpu=%llu)\n",
                    ds.policy.c_str(),
                    (unsigned long long)ds.n_calls,
                    (unsigned long long)ds.n_overrides,
                    (unsigned long long)ds.n_mulmat,
                    (unsigned long long)ds.n_mulmat_cpu,
                    (unsigned long long)ds.n_mulmat_gpu);
        });
    }

    auto * mem = llama_get_memory(ctx);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto chat_templates = common_chat_templates_init(model, params.chat_template);

    LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        LOG_ERR("%s: no CPU backend found\n", __func__);
        return 1;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    set_process_priority(params.cpuparams.priority);

    struct ggml_threadpool * threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return 1;
        }

        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return 1;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // auto enable conversation mode if chat template is available
    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        } else {
            params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }

    // in case user force-activate conversation mode (via -cnv) without proper chat template, we show a warning
    if (params.conversation_mode && !has_chat_template) {
        LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
    }

    // print chat template example in conversation mode
    if (params.conversation_mode) {
        if (params.enable_chat_template) {
            if (!params.prompt.empty() && params.system_prompt.empty()) {
                LOG_WRN("*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?\n");
            }

            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(chat_templates.get(), params.use_jinja, params.default_template_kwargs).c_str());
        } else {
            LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG_INF("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG_INF("%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(path_session)) {
            LOG_INF("%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_ERR("%s: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            LOG_INF("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;

    bool waiting_for_first_input = false;
    auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
        common_chat_msg new_msg;
        new_msg.role = role;
        new_msg.content = content;
        auto formatted = common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
        chat_msgs.push_back(new_msg);
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };

    std::string prompt;
    {
        if (params.conversation_mode && params.enable_chat_template) {
            if (!params.system_prompt.empty()) {
                // format the system prompt (will use template default if empty)
                chat_add_and_format("system", params.system_prompt);
            }

            if (!params.prompt.empty()) {
                // format and append the user prompt
                chat_add_and_format("user", params.prompt);
            } else {
                waiting_for_first_input = true;
            }

            if (!params.system_prompt.empty() || !params.prompt.empty()) {
                common_chat_templates_inputs inputs;
                inputs.use_jinja = g_params->use_jinja;
                inputs.messages = chat_msgs;
                inputs.add_generation_prompt = !params.prompt.empty();

                prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            }
        } else {
            // otherwise use the prompt as is
            prompt = params.prompt;
        }

        if (params.interactive_first || !prompt.empty() || session_tokens.empty()) {
            LOG_DBG("tokenize the prompt\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            LOG_DBG("use session tokens\n");
            embd_inp = session_tokens;
        }

        LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
    }

    // Should not run without any tokens
    if (!waiting_for_first_input && embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_vocab_bos(vocab));
            LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
        } else {
            LOG_ERR("input is empty\n");
            return -1;
        }
    }

    // Tokenize negative prompt
    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_INF("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_INF("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_WRN("%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_INF("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        if (!llama_memory_seq_rm(mem, -1, n_matching_session_tokens, -1)) {
            LOG_INF("%s: unable to resuse common prefix\n", __func__);
            n_matching_session_tokens = 0;
            llama_memory_seq_rm(mem, -1, -1, -1);
        }
    }

    LOG_DBG("recalculate the cached logits (check): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
         embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )\n", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    if (params.conversation_mode) {
        if (params.single_turn && !params.prompt.empty()) {
            params.interactive = false;
            params.interactive_first = false;
        } else {
            params.interactive_first = true;
        }
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_INF("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            LOG_INF("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }

    // ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (params.interactive) {
        LOG_INF("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_INF("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG_INF("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_INF("Input prefix: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG_INF("Input suffix: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    smpl = common_sampler_init(model, sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        return 1;
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl));
    LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl).c_str());

    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
      //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
      //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_INF("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_INF("\n");

    if (params.interactive) {
        const char * control_message;
        if (params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_INF(       "%s", control_message);
        if (params.conversation_mode && params.enable_chat_template && params.system_prompt.empty()) {
            LOG_INF(   " - Not using system message. To change it, set a different value via -sys PROMPT\n");
        }
        LOG_INF("\n");

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool display              = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;
    std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode

    // the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(console::prompt);
    display = params.display_prompt;

    std::vector<llama_token> embd;
    const double elastic_bench_seconds = []() {
        const char * e = std::getenv("LLAMA_ELASTIC_BENCH_SECONDS");
        return (e && *e) ? std::atof(e) : 0.0;
    }();
    std::chrono::steady_clock::time_point elastic_bench_t0{};
    bool elastic_bench_started = false;
    bool elastic_bench_time_done = false;
    int  elastic_bench_generated = 0;
    if (elastic_bench_seconds > 0.0) {
        LOG_INF("[elastic-bench] duration limit enabled: %.3f seconds\n", elastic_bench_seconds);
    }

    // single-token antiprompts
    std::vector<llama_token> antiprompt_token;

    for (const std::string & antiprompt : params.antiprompt) {
        auto ids = ::common_tokenize(ctx, antiprompt, false, true);
        if (ids.size() == 1) {
            antiprompt_token.push_back(ids[0]);
        }
    }

    if (llama_model_has_encoder(model)) {
        int enc_input_size = embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(console::error);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(console::reset);
            }

            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                if (n_past + (int) embd.size() >= n_ctx) {
                    if (!params.ctx_shift){
                        LOG_WRN("\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                        break;
                    }

                    if (params.n_predict == -2) {
                        LOG_WRN("\n\n%s: context full and n_predict == %d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left/2;

                    LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                            n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_memory_seq_rm (mem, 0, params.n_keep            , params.n_keep + n_discard);
                    llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    LOG_DBG("after swap: n_past = %d\n", n_past);

                    LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

                    LOG_DBG("clear session path\n");
                    path_session.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG_DBG("\n");
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_memory_seq_add(mem, 0, ga_i,                n_past,              ib*bd);
                    llama_memory_seq_div(mem, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_memory_seq_add(mem, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    LOG_ERR("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                LOG_DBG("n_past = %d\n", n_past);
                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG_DBG("saved session to %s\n", path_session.c_str());
            }

            llama_token id = common_sampler_sample(smpl, ctx, -1);
            if (elastic_bench_seconds > 0.0 && llama_vocab_is_eog(vocab, id)) {
                auto cont = common_tokenize(ctx, "\n", false, false);
                if (!cont.empty() && !llama_vocab_is_eog(vocab, cont.front())) {
                    id = cont.front();
                } else {
                    id = llama_vocab_bos(vocab);
                }
                LOG_DBG("[elastic-bench] replaced EOG with continuation token %d\n", id);
            }

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);

            // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            if (params.conversation_mode && !waiting_for_first_input && !llama_vocab_is_eog(vocab, id)) {
                assistant_ss << common_token_to_piece(ctx, id, false);
            }

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;
            ++elastic_bench_generated;
            if (elastic_bench_seconds > 0.0 && elastic_bench_generated > 0) {
                if (!elastic_bench_started) {
                    elastic_bench_started = true;
                    elastic_bench_t0 = std::chrono::steady_clock::now();
                }
                const auto now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(now - elastic_bench_t0).count();
                if (elapsed >= elastic_bench_seconds) {
                    elastic_bench_time_done = true;
                }
            }

            LOG_DBG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                // Console/Stream Output
                LOG("%s", token_str.c_str());

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(id);
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(console::reset);
            display = true;
        }
        if (elastic_bench_time_done) {
            LOG_INF("\n[elastic-bench] reached duration %.3f s after %d generated tokens\n",
                    elastic_bench_seconds, elastic_bench_generated);
            break;
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                // avoid calling common_sampler_last() if last_output is empty
                if (!last_output.empty()) {
                    llama_token last_token = common_sampler_last(smpl);
                    for (auto token : antiprompt_token) {
                        if (token == last_token) {
                            if (params.interactive) {
                                is_interacting = true;
                            }
                            is_antiprompt = true;
                            break;
                        }
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of generation tokens in interactive mode
            if (!waiting_for_first_input && llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
                LOG_DBG("found an EOG token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (params.enable_chat_template) {
                        chat_add_and_format("assistant", assistant_ss.str());
                    }
                    is_interacting = true;
                    LOG("\n");
                }
            }

            if (params.conversation_mode && !waiting_for_first_input) {
                if (!prompt.empty()) {
                    prompt.clear();
                    is_interacting = false;
                }
            }

            if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
                LOG_DBG("waiting for user input\n");

                if (params.conversation_mode) {
                    LOG("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG_DBG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_vocab_bos(vocab));
                }

                std::string buffer;
                if (!params.input_prefix.empty() && !params.conversation_mode) {
                    LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    LOG("%s", params.input_prefix.c_str());
                }

                // color user input only
                console::set_display(console::user_input);
                display = params.display_prompt;

                std::string line;
                bool another_line = true;
                do {
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);

                // done taking input, reset color
                console::set_display(console::reset);
                display = true;

                if (buffer.empty()) { // Ctrl+D on empty line exits
                    LOG("EOF by user\n");
                    break;
                }

                if (buffer.back() == '\n') {
                    // Implement #587:
                    // If the user wants the text to end in a newline,
                    // this should be accomplished by explicitly adding a newline by using \ followed by return,
                    // then returning control by pressing return again.
                    buffer.pop_back();
                }

                if (buffer.empty()) { // Enter key on empty line lets the user pass control back
                    LOG_DBG("empty line, passing control back\n");
                } else { // Add tokens to embd only if the input buffer is non-empty
                    // append input suffix if any
                    if (!params.input_suffix.empty() && !params.conversation_mode) {
                        LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        LOG("%s", params.input_suffix.c_str());
                    }

                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    bool format_chat = params.conversation_mode && params.enable_chat_template;
                    std::string user_inp = format_chat
                        ? chat_add_and_format("user", std::move(buffer))
                        : std::move(buffer);
                    // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    // if user stop generation mid-way, we must add EOT to finish model's last response
                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_vocab_eot(vocab);
                        embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    if (params.verbose_prompt) {
                        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size() - original_size);
                    }

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        const std::string token_str = common_token_to_piece(ctx, token);
                        output_tokens.push_back(token);
                        output_ss << token_str;

                        if (params.verbose_prompt) {
                            LOG_INF("%6d -> '%s'\n", token, token_str.c_str());
                        }
                    }

                    // reset assistant message
                    assistant_ss.str("");

                    n_remain -= line_inp.size();
                    LOG_DBG("n_remain: %d\n", n_remain);
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0 || waiting_for_first_input) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;

                if (waiting_for_first_input && params.single_turn) {
                    params.interactive = false;
                    params.interactive_first = false;
                }
                waiting_for_first_input = false;
            }
        }

        // end of generation
        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    LOG("\n\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);

    llama_backend_free();

    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);

    return 0;
}
