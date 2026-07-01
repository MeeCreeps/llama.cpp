#pragma once

#include "llama.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// elastic plan framework (前向声明,完整类型在 llama-context.cpp include)
namespace elastic {
    struct ExecPlan;
    class  PlanExecutor;
    class  PlanProvider;
}

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_op_schedule(llama_op_schedule_fn fn, void * user_data);
    void set_weight_pin (llama_weight_pin_fn  fn, void * user_data);
    void set_scheduler  (llama_scheduler_fn   fn, void * user_data);
    void set_mem_watch_threshold(int mb);
    // 强制下次 decode 重建 graph(routing 变, e.g. plan band 切换). 配 LLAMA_KEEP_GRAPH_REUSE.
    void graph_invalidate();
    void set_op_runtime_dispatch(llama_op_runtime_dispatch_fn fn, void * user_data);
    int  n_backends() const;
    const char * backend_name(int i) const;

    // Fires the runtime scheduler if MemAvailable changed by ≥ threshold since last tick.
    // Called pre-decode. Updates last_mem_avail_mb + decode_step.
    void maybe_run_scheduler();

    // ── Elastic plan framework 公开入口(C API 经由这些方法)──
    int  apply_exec_plan(const elastic::ExecPlan * plan);    // DoD#1:apply 一个 plan
    int  elastic_enable(const char * provider_kind, const char * plans_dir);  // DoD#2
    void elastic_disable();
    void elastic_set_provider_fn(llama_plan_provider_fn fn, void * user_data);
    const struct llama_plan * elastic_get_plan_view(int64_t budget_mib);

    void set_adapter_lora(
            llama_adapter_lora * adapter,
            float scale);

    bool rm_adapter_lora(
            llama_adapter_lora * adapter);

    void clear_adapter_lora();

    bool apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    //
    // graph
    //

public:
    uint32_t graph_max_nodes() const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams       cparams;
    llama_adapter_cvec  cvec;
    llama_adapter_loras loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // Runtime op-level scheduler. Called from graph_get_cb per node, can override
    // backend assignment. LLAMA_OP_SCHED env enables (string strategy name).
    // Counter incremented per build_graph call (== per decode batch).
    uint64_t    op_sched_counter = 0;
    std::string op_sched_strategy;  // empty = disabled

    // Schedule callbacks (set via llama_set_op_schedule / llama_set_weight_pin).
    // 优先级高于 env strategies — fn 非空且返回 valid id 时覆盖.
    llama_op_schedule_fn op_schedule_fn  = nullptr;
    void *               op_schedule_ud  = nullptr;
    llama_weight_pin_fn  weight_pin_fn   = nullptr;
    void *               weight_pin_ud   = nullptr;

    // Runtime scheduler — fired pre-decode when MemAvailable changes by ≥ threshold.
    llama_scheduler_fn  scheduler_fn        = nullptr;
    void *              scheduler_ud        = nullptr;
    int                 mem_watch_threshold = 100;   // MB
    int64_t             last_mem_avail_mb   = -1;    // -1 = never sampled
    uint64_t            decode_step         = 0;

    // True per-op runtime dispatch hook — called by ggml-sched compute_splits
    // before each op compute. Plumbed via ggml_backend_sched_set_runtime_dispatch.
    llama_op_runtime_dispatch_fn op_runtime_dispatch_fn = nullptr;
    void *                       op_runtime_dispatch_ud = nullptr;

    // ── Elastic plan framework (Plan IR → Execute) ──
    // executor 把当前 plan 翻译成 residency/routing 动作;route 表喂给 op_schedule。
    std::unique_ptr<elastic::PlanExecutor> elastic_executor;
    std::unique_ptr<elastic::PlanProvider> elastic_provider;   // dynamic 模式
    const elastic::ExecPlan * elastic_plan         = nullptr;  // 当前已 apply(不拥有)
    const elastic::ExecPlan * elastic_last_applied = nullptr;  // online loop 指针比较用
    const elastic::ExecPlan * elastic_last_provider_plan = nullptr; // provider returned this plan for this budget
    uint64_t elastic_last_plan_signature = 0;                  // execution-equivalence debounce
    int64_t  elastic_last_provider_budget_mib = -1;            // raw/effective budget for provider-result cache
    int64_t  elastic_pending_budget_mib  = -1;                 // budget switch hysteresis candidate
    int      elastic_pending_budget_hits = 0;                  // consecutive decode ticks at candidate budget
    int64_t  elastic_effective_budget_mib = -1;                // optional slew-limited budget for online planning
    uint64_t elastic_last_switch_decode_step = 0;              // last real apply tick
    std::unordered_map<std::string, int> elastic_route;        // weight 名 → backend_id (STATIC routing)
    std::unordered_map<std::string, int> elastic_runtime_route;// weight 名 → backend_id (RUNTIME dispatch, M5)
    std::unordered_map<std::string, int> elastic_anchor_op;    // anchor weight/op 名 → op_id
    std::unordered_map<const ggml_tensor *, int> elastic_graph_op_index; // current graph tensor* -> op index
    std::unordered_set<int>              elastic_anchor_fired; // 当前 plan 已触发过的 anchor op
    uint64_t elastic_anchor_requests        = 0;               // backend 到达 weight anchor 的通知次数
    uint64_t elastic_anchor_hits            = 0;               // 命中当前 plan anchor 的次数
    uint64_t elastic_anchor_duplicates      = 0;               // 同一 anchor 重复通知次数
    uint64_t elastic_anchor_events_fired    = 0;               // 实际下发的 timeline events 数
    uint64_t elastic_anchor_load_events     = 0;
    uint64_t elastic_anchor_transfer_events = 0;
    uint64_t elastic_anchor_xform_events    = 0;
    uint64_t elastic_anchor_stage_failures  = 0;               // provider 返回非 0 的 stage/xform 请求数
    // True runtime MRU cache baseline. This is separate from plan-time resident
    // selection: every pre-op access updates recency; a miss loads the weight,
    // then pressure evicts the most recently used resident weight other than
    // the current op's input.
    bool     elastic_mru_cache_enabled      = false;
    const elastic::ExecPlan * elastic_mru_cache_plan = nullptr;
    uint64_t elastic_mru_cache_clock        = 0;
    uint64_t elastic_mru_cache_accesses     = 0;
    uint64_t elastic_mru_cache_hits         = 0;
    uint64_t elastic_mru_cache_misses       = 0;
    uint64_t elastic_mru_cache_load_failures = 0;
    uint64_t elastic_mru_cache_evictions    = 0;
    uint64_t elastic_mru_cache_evict_failures = 0;
    size_t   elastic_mru_cache_resident_bytes = 0;
    size_t   elastic_mru_cache_last_budget_bytes = 0;
    bool     elastic_mru_cache_logged_budget = false;
    std::unordered_map<int, uint64_t> elastic_mru_cache_last_use;
    std::unordered_set<int>           elastic_mru_cache_resident;
    std::unordered_set<int>           elastic_mru_cache_pending_evict;
    // weight 名 → (migrate_from_backend, xform) — 跨后端迁移意图 (M5, 设备侧用)
    std::unordered_map<std::string, std::pair<int,int>> elastic_migrate;
    bool    elastic_enabled = false;                           // dynamic online loop 开关
    int     elastic_cpu_id  = -1;                              // CPU backend index
    int     elastic_gpu_id  = -1;                              // GPU backend index(-1=无)
    llama_plan_provider_fn elastic_provider_fn = nullptr;      // callback provider
    void *  elastic_provider_ud = nullptr;
    // llama_elastic_get_plan 的只读句柄缓存(按 budget;析构时 llama_plan_free)
    std::map<int64_t, struct llama_plan *> elastic_plan_view_cache;

    // 内部 helpers(apply_exec_plan 在 public 区声明)
    void elastic_install_op_schedule();                        // 安装读 elastic_route 的 op_schedule_fn
    void elastic_install_runtime_dispatch();                   // M5:装 per-op runtime dispatch hook
    void elastic_fire_anchor_op(int op_id, const char * reason);// 按 graph op index 触发 staged pipeline events
    bool elastic_mru_cache_pre_op(const struct ggml_tensor * op);
    bool elastic_mru_cache_access_weight(const char * raw_name);
    void elastic_mru_cache_reset();
    void elastic_mru_cache_flush_pending_evict();
    size_t elastic_mru_cache_budget_bytes() const;
    void maybe_apply_plan();                                   // online loop:档变换 plan
    int64_t elastic_budget_mib() const;                        // 当前预算(BudgetWatcher/meminfo)

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
