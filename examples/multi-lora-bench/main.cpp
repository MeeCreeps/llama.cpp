// multi-lora-bench: M2 — same-adapter batching scheduler.
//
// M1 was strict serial: at most one in-flight request, no batching across
// requests. M2 keeps the single-thread main loop but lets up to --n-slots
// requests be active simultaneously, batches same-adapter requests through
// one llama_decode call, and amortises adapter-swap overhead across the
// group.
//
// Out of scope at M2: cross-adapter mixed batches (spec section 1.2 makes
// this an explicit non-goal — same-adapter only), size-aware cache budget
// (M3), starvation handling (see docs/multi-lora/known-issues.md I-1 —
// observed but not fixed in M2).
//
// Single-threaded by design (CLAUDE.md). All `llama_*` calls happen on the
// main thread.

#include "adapter_pool.h"
#include "metrics.h"
#include "scheduler.h"
#include "slot.h"

#include "llama.h"
#include "ggml.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using nlohmann::json;
using clock_t_ = std::chrono::steady_clock;

namespace {

struct args_t {
    std::string model_path;
    std::string adapter_dir;
    std::string workload_path;
    std::string out_path;
    std::string output_dir;     // optional: per-request detokenized text

    int  n_ctx              = 4096;
    int  n_batch            = 512;
    int  n_gpu_layers       = 99;
    int  max_adapter_mem_mb = 0;   // 0 = unbounded; M3 byte budget
    int  n_slots            = 4;
    int  seed               = 42;
    bool verbose            = true;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL -a ADAPTER_DIR -w WORKLOAD.json -o OUT.csv [opts]\n"
        "\n"
        "M2 multi-lora bench: same-adapter batching scheduler.\n"
        "\n"
        "  -m, --model FNAME            base model GGUF (required)\n"
        "  -a, --adapter-dir DIR        directory holding <id>.gguf adapters (required)\n"
        "  -w, --workload FNAME         pre-tokenized trace JSON (required)\n"
        "  -o, --out FNAME              metrics CSV output path (required)\n"
        "      --output-dir DIR         optional: dump <req_id>.txt with detokenized\n"
        "                               output for each finished request (dir must exist)\n"
        "      --max-adapter-mem-mb N   adapter cache byte budget in MiB (default: 0 = unbounded)\n"
        "      --n-slots N              max in-flight requests / batch slots (default: 4)\n"
        "  -c, --n-ctx N                context size (default: 4096)\n"
        "  -b, --n-batch N              max tokens per llama_decode (default: 512)\n"
        "  -ngl, --n-gpu-layers N       layers offloaded to GPU (default: 99)\n"
        "  -s, --seed N                 RNG seed; greedy sampler so mostly cosmetic (default: 42)\n"
        "      --quiet                  suppress per-request progress\n"
        "  -h, --help                   show this and exit\n",
        argv0);
}

bool parse_args(int argc, char ** argv, args_t & a) {
    auto need = [&](int i) {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "error: %s requires an argument\n", argv[i]);
            return false;
        }
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "-h" || s == "--help") { print_usage(argv[0]); std::exit(0); }
        else if (s == "-m" || s == "--model")          { if (!need(i)) return false; a.model_path    = argv[++i]; }
        else if (s == "-a" || s == "--adapter-dir")    { if (!need(i)) return false; a.adapter_dir   = argv[++i]; }
        else if (s == "-w" || s == "--workload")       { if (!need(i)) return false; a.workload_path = argv[++i]; }
        else if (s == "-o" || s == "--out")            { if (!need(i)) return false; a.out_path      = argv[++i]; }
        else if (s == "--output-dir")                  { if (!need(i)) return false; a.output_dir    = argv[++i]; }
        else if (s == "--max-adapter-mem-mb")          { if (!need(i)) return false; a.max_adapter_mem_mb = std::stoi(argv[++i]); }
        else if (s == "--n-slots")                     { if (!need(i)) return false; a.n_slots       = std::stoi(argv[++i]); }
        else if (s == "-c" || s == "--n-ctx")          { if (!need(i)) return false; a.n_ctx         = std::stoi(argv[++i]); }
        else if (s == "-b" || s == "--n-batch")        { if (!need(i)) return false; a.n_batch       = std::stoi(argv[++i]); }
        else if (s == "-ngl" || s == "--n-gpu-layers") { if (!need(i)) return false; a.n_gpu_layers  = std::stoi(argv[++i]); }
        else if (s == "-s" || s == "--seed")           { if (!need(i)) return false; a.seed          = std::stoi(argv[++i]); }
        else if (s == "--quiet")                       { a.verbose = false; }
        else {
            std::fprintf(stderr, "error: unknown arg '%s'\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }
    }
    if (a.model_path.empty() || a.adapter_dir.empty()
        || a.workload_path.empty() || a.out_path.empty()) {
        std::fprintf(stderr, "error: --model, --adapter-dir, --workload, --out are all required\n");
        print_usage(argv[0]);
        return false;
    }
    return true;
}

std::vector<multilora_request> load_trace(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("load_trace: failed to open '" + path + "'");
    }
    json j;
    f >> j;
    if (!j.is_array()) {
        throw std::runtime_error("load_trace: top-level JSON must be an array of requests");
    }
    std::vector<multilora_request> out;
    out.reserve(j.size());
    for (const auto & rj : j) {
        multilora_request r;
        r.id           = rj.at("id").get<std::string>();
        r.arrival_time = rj.at("arrival_time").get<double>();
        r.adapter_id   = rj.at("adapter_id").get<std::string>();
        r.max_output   = rj.value("max_output", 128);
        const auto & toks = rj.at("input_tokens");
        if (!toks.is_array()) {
            throw std::runtime_error("load_trace: input_tokens of '" + r.id + "' is not an array");
        }
        r.input_tokens.reserve(toks.size());
        for (const auto & t : toks) {
            r.input_tokens.push_back(t.get<llama_token>());
        }
        out.push_back(std::move(r));
    }
    std::sort(out.begin(), out.end(),
        [](const multilora_request & a, const multilora_request & b) {
            return a.arrival_time < b.arrival_time;
        });
    return out;
}

double seconds_since(clock_t_::time_point t0) {
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

void sleep_until_arrival(clock_t_::time_point t0, double arrival_s) {
    const auto target = t0 + std::chrono::duration_cast<clock_t_::duration>(
        std::chrono::duration<double>(arrival_s));
    if (target > clock_t_::now()) {
        std::this_thread::sleep_until(target);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    args_t args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    if (args.verbose) {
        std::fprintf(stderr, "[bench] model=%s adapter_dir=%s workload=%s out=%s\n",
                     args.model_path.c_str(), args.adapter_dir.c_str(),
                     args.workload_path.c_str(), args.out_path.c_str());
        std::fprintf(stderr,
            "[bench] n_ctx=%d n_batch=%d ngl=%d max_adapter_mem_mb=%d n_slots=%d seed=%d\n",
            args.n_ctx, args.n_batch, args.n_gpu_layers,
            args.max_adapter_mem_mb, args.n_slots, args.seed);
    }

    ggml_backend_load_all();
    llama_backend_init();

    std::vector<multilora_request> requests;
    try {
        requests = load_trace(args.workload_path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        llama_backend_free();
        return 2;
    }
    std::fprintf(stderr, "[bench] loaded %zu requests\n", requests.size());

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;

    struct llama_model * model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "fatal: llama_model_load_from_file failed\n");
        llama_backend_free();
        return 3;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    auto cparams = llama_context_default_params();
    cparams.n_ctx     = static_cast<uint32_t>(args.n_ctx);
    cparams.n_batch   = static_cast<uint32_t>(args.n_batch);
    cparams.n_seq_max = static_cast<uint32_t>(args.n_slots);
    // Per llama.h: with n_seq_max > 1 and sequences not sharing a large
    // prefix (true for unrelated requests), kv_unified=false performs better.
    cparams.kv_unified = false;
    cparams.no_perf    = false;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "fatal: llama_init_from_model failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const size_t max_bytes = static_cast<size_t>(args.max_adapter_mem_mb) * 1024ull * 1024ull;
    multilora_adapter_pool pool(model, args.adapter_dir, max_bytes);
    multilora_metrics      metrics;

    int    exit_code = 0;
    const  auto t0   = clock_t_::now();
    size_t next_idx  = 0;
    size_t completed_at_last_print = 0;

    multilora_scheduler    sched(ctx, vocab, smpl, &pool, &metrics,
                                 static_cast<size_t>(args.n_slots),
                                 static_cast<int32_t>(args.n_batch),
                                 t0);
    if (!args.output_dir.empty()) {
        sched.set_output_dir(args.output_dir);
    }

    while (next_idx < requests.size() || sched.active_count() > 0) {
        const double now = seconds_since(t0);

        // Admit anything that has arrived, up to the slot cap.
        while (next_idx < requests.size()
               && requests[next_idx].arrival_time <= now
               && sched.active_count() < static_cast<size_t>(args.n_slots)) {
            sched.admit(requests[next_idx]);
            ++next_idx;
        }

        if (sched.active_count() == 0) {
            // No work in flight; sleep until next arrival.
            if (next_idx < requests.size()) {
                sleep_until_arrival(t0, requests[next_idx].arrival_time);
                continue;
            }
            break;  // exhausted
        }

        sched.step();

        if (args.verbose && metrics.size() != completed_at_last_print) {
            // Print one line per completed request as they finish.
            for (size_t i = completed_at_last_print; i < metrics.size(); ++i) {
                const auto & m = metrics.records()[i];
                std::fprintf(stderr,
                    "[%4zu/%zu] %-20s adapter=%-12s %s ttft=%6.3fs e2e=%6.3fs out=%4zu/%-4d acq=%5.1fms\n",
                    i + 1, requests.size(),
                    m.id.c_str(), m.adapter_id.c_str(),
                    m.cache_hit ? "HIT " : "MISS",
                    m.first_token_time - m.arrival_time,
                    m.finish_time - m.arrival_time,
                    m.n_output_tokens, /*max_output unknown here*/ 0,
                    m.acquire_ms);
            }
            completed_at_last_print = metrics.size();
        }
    }

    if (!metrics.dump_csv(args.out_path)) {
        std::fprintf(stderr, "error: failed to write CSV to '%s'\n", args.out_path.c_str());
        exit_code = (exit_code == 0) ? 6 : exit_code;
    } else {
        const auto s = pool.stats();
        const double hr = pool.hit_rate();
        std::fprintf(stderr,
            "[bench] done. %zu metrics -> %s. cache: hit=%zu miss=%zu evict=%zu "
            "resident=%zu (%zu MiB peak %zu MiB) hit_rate=%.3f budget=%zu MiB\n",
            metrics.size(), args.out_path.c_str(),
            s.hits, s.misses, s.evicts, s.resident_count,
            s.resident_bytes / (1024 * 1024),
            s.peak_bytes / (1024 * 1024),
            hr,
            max_bytes / (1024 * 1024));

        // JSON sidecar at <csv>.summary.json — single-line pool + run summary
        // for plot scripts (M4) and post-hoc analysis. Keys mirror stats_t
        // plus a few derived values; keep schema flat.
        const std::string summary_path = args.out_path + ".summary.json";
        std::FILE * sf = std::fopen(summary_path.c_str(), "w");
        if (sf) {
            std::fprintf(sf,
                "{\"hits\":%zu,\"misses\":%zu,\"evicts\":%zu,"
                "\"resident_count\":%zu,\"resident_bytes\":%zu,"
                "\"peak_bytes\":%zu,\"hit_rate\":%.6f,"
                "\"budget_bytes\":%zu,\"n_slots\":%d,\"n_ctx\":%d,"
                "\"n_requests\":%zu}\n",
                s.hits, s.misses, s.evicts,
                s.resident_count, s.resident_bytes,
                s.peak_bytes, hr,
                max_bytes, args.n_slots, args.n_ctx,
                metrics.size());
            std::fclose(sf);
        } else {
            std::fprintf(stderr, "warning: failed to write summary to %s\n", summary_path.c_str());
        }
    }

    pool.shutdown();
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
