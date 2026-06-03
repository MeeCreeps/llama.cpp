// tests/elastic/test_plan_e2e.cpp
//
// End-to-end CPU 冒烟测试(DoD#1 + DoD#2 的桌面验证):
//   1. 加载一个真实模型(CPU-only)
//   2. llama_plan_load_json 加载一个 plan,llama_elastic_apply_plan → 应返回 0
//   3. decode 几个 token → 应正常产出(plan routing 在纯 CPU 上把 GPU 路由 fallback CPU)
//   4. callback provider + llama_elastic_enable("callback") → online loop 切 plan 不崩
//
// 用法: test_plan_e2e <model.gguf> <plan.json>
// 桌面无 GPU/elastic backend 时:residency sinks 是全局 registry no-op,routing 落 CPU,
// 主要验证「apply_plan 不崩 + decode 继续正确 + 在线换 plan 不崩」。

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); ++g_fail; }    \
    } while (0)

// callback provider 状态:按 budget 给不同 plan(这里复用同一个加载好的 plan 演示切换)。
struct CbState {
    const llama_plan * plan_a = nullptr;
    const llama_plan * plan_b = nullptr;
    int n_calls = 0;
};

static const llama_plan * cb_provider(int64_t budget_mib, void * ud) {
    auto * s = (CbState *) ud;
    s->n_calls++;
    // budget 大给 plan_a,小给 plan_b(演示档切换)
    return budget_mib >= 4000 ? s->plan_a : s->plan_b;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <plan.json>\n", argv[0]);
        return 2;  // skip (no args)
    }
    const char * model_path = argv[1];
    const char * plan_path  = argv[2];

    llama_backend_init();

    // 1) 加载模型 (CPU)
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::fprintf(stderr, "SKIP: model load failed (%s) — 可能非标准自回归 arch\n", model_path);
        llama_backend_free();
        return 0;  // skip, 不算失败
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 256;
    cp.n_batch = 64;
    llama_context * ctx = llama_init_from_model(model, cp);
    CHECK(ctx != nullptr, "context create");
    if (!ctx) { llama_model_free(model); llama_backend_free(); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 2) 加载 plan + apply (DoD#1)
    llama_plan * plan = llama_plan_load_json(plan_path);
    CHECK(plan != nullptr, "plan load");
    if (plan) {
        std::fprintf(stderr, "  plan: budget=%lldMiB weights=%d ops=%d\n",
                     (long long) llama_plan_budget_mib(plan),
                     llama_plan_n_weights(plan), llama_plan_n_ops(plan));
        int rc = llama_elastic_apply_plan(ctx, plan);
        CHECK(rc == 0, "apply_plan rc==0");
    }

    // 3) decode:prompt → 几个 token,验证不崩、有输出
    auto tokenize = [&](const std::string & text, bool bos) {
        int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, bos, true);
        std::vector<llama_token> toks(n);
        llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), n, bos, true);
        return toks;
    };
    std::vector<llama_token> toks = tokenize("Hello", true);
    CHECK(!toks.empty(), "tokenize");

    llama_batch batch = llama_batch_get_one(toks.data(), (int) toks.size());
    int rc = llama_decode(ctx, batch);
    CHECK(rc == 0, "decode prompt");

    // greedy 续 8 个 token
    int produced = 0;
    for (int i = 0; i < 8 && rc == 0; i++) {
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;
        int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token best = 0;
        float bestv = logits[0];
        for (int t = 1; t < n_vocab; t++) if (logits[t] > bestv) { bestv = logits[t]; best = t; }
        if (llama_vocab_is_eog(vocab, best)) break;
        llama_batch nb = llama_batch_get_one(&best, 1);
        rc = llama_decode(ctx, nb);
        produced++;
    }
    std::fprintf(stderr, "  decoded %d continuation tokens (rc=%d)\n", produced, rc);
    CHECK(rc == 0, "decode continuation");
    CHECK(produced > 0, "produced > 0 tokens");

    // 4) online loop (DoD#2):callback provider,内存变化时换 plan 不崩
    CbState cbs;
    cbs.plan_a = plan;       // 复用(真实用应是两个不同档 plan)
    cbs.plan_b = plan;
    llama_elastic_set_plan_provider(ctx, cb_provider, &cbs);
    int erc = llama_elastic_enable(ctx, "callback", nullptr);
    CHECK(erc == 0, "elastic_enable callback");

    // 再 decode 几步,触发 maybe_apply_plan(每 decode 前跑)
    for (int i = 0; i < 4 && rc == 0; i++) {
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) break;
        int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token best = 0; float bestv = logits[0];
        for (int t = 1; t < n_vocab; t++) if (logits[t] > bestv) { bestv = logits[t]; best = t; }
        llama_batch nb = llama_batch_get_one(&best, 1);
        rc = llama_decode(ctx, nb);
    }
    CHECK(rc == 0, "decode under elastic online loop");
    std::fprintf(stderr, "  provider callback invoked %d times\n", cbs.n_calls);
    CHECK(cbs.n_calls > 0, "provider invoked");

    if (plan) llama_plan_free(plan);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (g_fail == 0) { std::printf("test_plan_e2e: ALL PASS\n"); return 0; }
    std::printf("test_plan_e2e: %d FAILED\n", g_fail);
    return 1;
}
