// multi-lora-bench: M1 milestone — serial trace replay with adapter pool.
//
// Reads a pre-tokenized JSON workload trace, replays it on a single
// llama_context one request at a time, swapping LoRA adapters via
// AdapterPool, and writes per-request CSV metrics.
//
// Out of scope at M1: batching across requests (M2), size-aware cache
// budget (M3), workload generator (M4 PC-side tool, separate file).
//
// Single-threaded by design (CLAUDE.md): no std::thread, no future, no
// async. The whole program is one main loop.
//
// CLI flags exposed for M1 are listed in print_usage(). Spec section 4 M1
// for behavior; api-versions.md for the actual llama_* signatures used.

#include "adapter_pool.h"
#include "metrics.h"

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

    int    n_ctx        = 4096;
    int    n_batch      = 512;
    int    n_gpu_layers = 99;
    int    max_resident = 10;
    int    seed         = 42;
    bool   verbose      = true;
};

struct request_t {
    std::string              id;
    double                   arrival_time = 0.0;
    std::string              adapter_id;
    std::vector<llama_token> input_tokens;
    int                      max_output = 128;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL -a ADAPTER_DIR -w WORKLOAD.json -o OUT.csv [opts]\n"
        "\n"
        "M1 multi-lora bench: serial trace replay with adapter LRU pool.\n"
        "\n"
        "  -m, --model FNAME            base model GGUF (required)\n"
        "  -a, --adapter-dir DIR        directory holding <id>.gguf adapters (required)\n"
        "  -w, --workload FNAME         pre-tokenized trace JSON (required)\n"
        "  -o, --out FNAME              metrics CSV output path (required)\n"
        "      --max-resident N         adapters cached simultaneously (default: 10)\n"
        "  -c, --n-ctx N                context size (default: 4096)\n"
        "  -b, --n-batch N              max tokens per llama_decode (default: 512)\n"
        "  -ngl, --n-gpu-layers N       layers offloaded to GPU (default: 99)\n"
        "  -s, --seed N                 RNG seed; greedy sampler so mostly cosmetic (default: 42)\n"
        "      --quiet                  suppress per-request progress\n"
        "  -h, --help                   show this and exit\n",
        argv0);
}

// Tiny manual arg parser. We deliberately do NOT use common's gpt_params /
// arg.cpp because that pulls in chat template + sampler infrastructure we
// don't need here, and its flag set conflicts with our trace-driven model.
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
        else if (s == "--max-resident")                { if (!need(i)) return false; a.max_resident  = std::stoi(argv[++i]); }
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

std::vector<request_t> load_trace(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("load_trace: failed to open '" + path + "'");
    }
    json j;
    f >> j;
    if (!j.is_array()) {
        throw std::runtime_error("load_trace: top-level JSON must be an array of requests");
    }
    std::vector<request_t> out;
    out.reserve(j.size());
    for (const auto & rj : j) {
        request_t r;
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
    std::sort(out.begin(), out.end(), [](const request_t & a, const request_t & b) {
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

// Run a single request end-to-end on the given context. Mutates `m` in
// place. Returns false on a fatal llama_decode error so the caller can
// decide whether to abort the trace or skip the request.
bool run_request(struct llama_context * ctx,
                 struct llama_sampler * smpl,
                 const struct llama_vocab * vocab,
                 const request_t & req,
                 clock_t_::time_point t0,
                 multilora_request_metric & m) {
    m.id              = req.id;
    m.adapter_id      = req.adapter_id;
    m.arrival_time    = req.arrival_time;
    m.n_input_tokens  = req.input_tokens.size();
    m.n_output_tokens = 0;

    // Reset sampler state between requests (chain-internal counters etc.).
    llama_sampler_reset(smpl);

    // Drop seq_id=0 KV from the prior request so positions reset.
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);

    // ---- prefill ----
    // We mutate a local copy of the token buffer because llama_batch_get_one
    // returns a non-owning view onto the array.
    std::vector<llama_token> prompt = req.input_tokens;
    if (prompt.empty()) {
        std::fprintf(stderr, "warn: request '%s' has zero input tokens; skipping\n",
                     req.id.c_str());
        m.first_token_time = seconds_since(t0);
        m.finish_time      = m.first_token_time;
        return true;
    }

    llama_batch batch = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "error: llama_decode (prefill) failed for '%s'\n", req.id.c_str());
        return false;
    }

    // ---- decode loop ----
    llama_token new_tok = 0;
    bool first_token_recorded = false;
    while ((int) m.n_output_tokens < req.max_output) {
        new_tok = llama_sampler_sample(smpl, ctx, -1);
        if (!first_token_recorded) {
            m.first_token_time   = seconds_since(t0);
            first_token_recorded = true;
        }
        if (llama_vocab_is_eog(vocab, new_tok)) {
            break;
        }
        ++m.n_output_tokens;
        batch = llama_batch_get_one(&new_tok, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: llama_decode (step) failed for '%s' at out=%zu\n",
                         req.id.c_str(), m.n_output_tokens);
            return false;
        }
    }
    if (!first_token_recorded) {
        // max_output == 0: still record TTFT as the prefill-finish point
        m.first_token_time = seconds_since(t0);
    }
    m.finish_time = seconds_since(t0);
    return true;
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
        std::fprintf(stderr, "[bench] n_ctx=%d n_batch=%d ngl=%d max_resident=%d seed=%d\n",
                     args.n_ctx, args.n_batch, args.n_gpu_layers,
                     args.max_resident, args.seed);
    }

    // ---- backends ----
    ggml_backend_load_all();
    llama_backend_init();

    // ---- load trace BEFORE model so JSON failures fail fast ----
    std::vector<request_t> requests;
    try {
        requests = load_trace(args.workload_path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        llama_backend_free();
        return 2;
    }
    std::fprintf(stderr, "[bench] loaded %zu requests\n", requests.size());

    // ---- model + context ----
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
    cparams.n_ctx   = (uint32_t) args.n_ctx;
    cparams.n_batch = (uint32_t) args.n_batch;
    cparams.no_perf = false;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "fatal: llama_init_from_model failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    // ---- sampler chain (greedy = deterministic baseline) ----
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // ---- adapter pool + metrics ----
    multilora_adapter_pool pool(model, args.adapter_dir, (size_t) args.max_resident);
    multilora_metrics      metrics;

    // ---- main loop ----
    int  exit_code   = 0;
    auto t0          = clock_t_::now();

    for (size_t idx = 0; idx < requests.size(); ++idx) {
        const auto & req = requests[idx];
        sleep_until_arrival(t0, req.arrival_time);

        // Acquire (cold or hot).
        multilora_adapter_pool::acquire_result acq;
        try {
            acq = pool.acquire(ctx, req.adapter_id);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "error: pool.acquire('%s') failed: %s; skipping request '%s'\n",
                         req.adapter_id.c_str(), e.what(), req.id.c_str());
            continue;
        }

        multilora_request_metric m;
        m.cache_hit   = acq.cache_hit;
        m.acquire_ms  = acq.load_ms;

        const bool ok = run_request(ctx, smpl, vocab, req, t0, m);
        pool.release(req.adapter_id);

        if (!ok) {
            exit_code = 5;
            // Still record the partial metric; downstream tooling can filter.
            metrics.record(std::move(m));
            break;
        }

        if (args.verbose) {
            std::fprintf(stderr,
                "[%4zu/%zu] %-20s adapter=%-12s %s ttft=%6.3fs e2e=%6.3fs out=%4zu/%-4d acq=%5.1fms\n",
                idx + 1, requests.size(),
                req.id.c_str(), req.adapter_id.c_str(),
                m.cache_hit ? "HIT " : "MISS",
                m.first_token_time - m.arrival_time,
                m.finish_time - m.arrival_time,
                m.n_output_tokens, req.max_output,
                m.acquire_ms);
        }
        metrics.record(std::move(m));
    }

    if (!metrics.dump_csv(args.out_path)) {
        std::fprintf(stderr, "error: failed to write CSV to '%s'\n", args.out_path.c_str());
        exit_code = (exit_code == 0) ? 6 : exit_code;
    } else {
        const auto s = pool.stats();
        std::fprintf(stderr,
            "[bench] done. %zu metrics written to %s. cache: hit=%zu miss=%zu evict=%zu resident=%zu\n",
            metrics.size(), args.out_path.c_str(),
            s.hits, s.misses, s.evicts, s.resident);
    }

    // ---- cleanup ----
    pool.shutdown();
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
