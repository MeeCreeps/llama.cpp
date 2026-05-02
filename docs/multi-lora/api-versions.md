# llama.cpp public API: actual symbols vs spec

Pinned to repo `HEAD = 73a2f509c` (branch `m1-pool`, fork of upstream master
@ `1a03cf47f`). `include/llama.h` last touched at `cfe9838d2`.

`IMPLEMENTATION_GUIDE.md` was drafted against an older API. This file is the
delta. M1+ code uses the **actual** signatures below — spec pseudocode is
illustrative only.

## Model load / free / context

| Spec wrote | Actually exists |
|---|---|
| `llama_load_model_from_file(path, ...)` | `llama_model_load_from_file(const char * path, struct llama_model_params params)` (also `..._ptr`, `..._from_splits`) |
| `llama_free_model(model)` (implicit) | `llama_model_free(struct llama_model *)` |
| `llama_new_context_with_model(model, ...)` | `llama_init_from_model(struct llama_model *, struct llama_context_params)` |
| — | `llama_free(struct llama_context *)` (context destructor) |
| — | `llama_model_default_params()` / `llama_context_default_params()` for parameter structs |

## Vocab + tokens

`llama_token_eos(model)` and friends are gone. The vocab is now a separate
opaque obtained from the model, and token-id getters live on it.

| Spec wrote | Actually exists |
|---|---|
| `llama_token_eos(model)` | `llama_vocab_eos(const struct llama_vocab *)`; also `llama_vocab_bos / eot / sep / nl / pad / mask / fim_*` |
| (no equivalent) | `llama_vocab_is_eog(vocab, tok)` — true for any end-of-generation token (recommended termination check; covers EOS + EOT + alora invocation tokens) |
| (no equivalent) | `llama_model_get_vocab(model)` — get the vocab handle |
| `llama_tokenize(model, text, ...)` | `llama_tokenize(const struct llama_vocab *, const char * text, int32_t text_len, llama_token * out, int32_t n_tokens_max, bool add_special, bool parse_special)` |
| — | `llama_token_to_piece(vocab, tok, buf, len, lstrip, special)` for detokenize-one |

**M1 implication**: cache `const llama_vocab * vocab = llama_model_get_vocab(model)`
once after model load. Termination check is `llama_vocab_is_eog(vocab, tok)`,
*not* `tok == llama_vocab_eos(vocab)` — the latter misses EOT and other
generation-stop tokens (Llama-3 instruct uses `<|eot_id|>`).

## LoRA adapter

This is the big rename. Spec described per-adapter `set/clear/scale` calls;
the new world is **one batched call** that replaces the entire active
adapter set.

| Spec wrote | Actually exists |
|---|---|
| `llama_lora_adapter_init(model, path)` returns `llama_lora_adapter *` | `llama_adapter_lora_init(struct llama_model *, const char * path_lora)` returns `struct llama_adapter_lora *` |
| `llama_lora_adapter_set(ctx, adapter, scale)` | gone — fold into `llama_set_adapters_lora` |
| `llama_lora_adapter_clear(ctx)` | gone — call `llama_set_adapters_lora(ctx, nullptr, 0, nullptr)` |
| `llama_lora_adapter_free(adapter)` | `llama_adapter_lora_free(struct llama_adapter_lora *)` |
| (no equivalent) | `int32_t llama_set_adapters_lora(struct llama_context *, struct llama_adapter_lora ** adapters, size_t n_adapters, float * scales)` — returns 0 on success |
| (no equivalent) | `llama_adapter_meta_count / _key_by_index / _val_str_by_index / _val_str` — read GGUF metadata KV pairs from adapter |
| `llama_lora_adapter_n_params()` (M3 §5.1 size-estimate hint) | **does not exist** — see §M3 caveat below |

`llama_set_adapters_lora` has a built-in optimisation: "Will only modify if
the adapters currently in context are different." So a hot-cache `acquire`
that re-sets the same adapter handle is cheap — `AdapterPool` doesn't need
to track "currently set" itself, but tracking still helps avoid mode-switch
overhead between same-id different-scale cases.

**M3 size-aware caveat**: there is no public API to ask an adapter how big
it is in bytes. Three options for `AdapterEntry::bytes`:
1. `stat()` the GGUF file on disk — close to in-memory cost for f16 LoRAs
2. Iterate `llama_adapter_meta_count`, parse tensor shapes from GGUF metadata
3. Track `RSS_after_load - RSS_before_load` per adapter (noisy, single-process safe)
   Pick option 1 for M3 first cut; revisit if it under-counts.

## KV cache / memory

The whole `llama_kv_cache_*` namespace was renamed to `llama_memory_*` and
now operates on a `llama_memory_t` handle obtained from the context.

| Spec wrote | Actually exists |
|---|---|
| `llama_kv_cache_seq_rm(ctx, seq, p0, p1)` | `llama_memory_seq_rm(llama_memory_t, llama_seq_id, llama_pos p0, llama_pos p1)` |
| (no equivalent) | `llama_memory_t llama_get_memory(const struct llama_context *)` — get the handle |
| — | `llama_memory_clear(mem, bool data)` — full reset |
| — | `llama_memory_seq_cp / _keep / _add / _div / _pos_min / _pos_max` |

**M1 implication**: M1 is single slot, no manual seq_id juggling. KV-cache
manipulation only matters in M2+ when we drop a finished slot's KV.

## Decode + batch

Mostly unchanged, with one nudge:

| API | Notes |
|---|---|
| `llama_decode(ctx, batch)` returns `int32_t` | 0 = success, 1 = no KV slot (resize batch / ctx), 2 = aborted, <0 = fatal |
| `llama_batch_get_one(tokens, n_tokens)` | header marks **"avoid using"** — kept as transition helper |
| `llama_batch_init(n_tokens, embd, n_seq_max)` / `llama_batch_free(batch)` | the real API; required for M2 multi-slot batches with explicit `seq_id` arrays |

**`llama_batch` struct fields** (M2+ fills these by hand):

```c
typedef struct llama_batch {
    int32_t          n_tokens;      // current fill level (mutate as we add)
    llama_token   *  token;         // [n_tokens]
    float         *  embd;          // unused unless feeding embeddings
    llama_pos     *  pos;           // [n_tokens]
    int32_t       *  n_seq_id;      // [n_tokens], how many seqs each token belongs to
    llama_seq_id ** seq_id;         // [n_tokens][n_seq_id[i]]
    int8_t        *  logits;        // [n_tokens], 1 = produce logits at this slot
} llama_batch;
```

For M2 each token belongs to exactly one slot, so `n_seq_id[i] = 1` and
`seq_id[i][0] = slot.seq_id`. `llama_batch_init(n_tokens=cap, embd=0,
n_seq_max=1)` is enough — that `n_seq_max=1` is the per-token max, not the
context's.

**`llama_sampler_sample(smpl, ctx, idx)` index semantics**: `idx` is the
BATCH POSITION of the token whose logits we want to sample from — i.e.
the index into `batch.logits[]`, not a running count of logits-bearing
tokens. `llama_get_logits_ith(ctx, idx)` aborts (`get_logits_ith: invalid
logits id N, reason: batch.logits[N] != true`) if the corresponding
`batch.logits[idx]` was 0 at decode time. Negative idx is allowed: -1
means "last logit-bearing token in the batch", but mixing positive and
negative across slots in the same step is brittle.

So building a batch:

```cpp
for (slot in batch_slots):
    feed-tokens-for-this-slot()   // last token has logits=1
    slot.sample_idx = batch.n_tokens - 1   // batch position of that token
// after llama_decode:
for slot in batch_slots: tok = llama_sampler_sample(smpl, ctx, slot.sample_idx);
```

The earlier draft of this doc said `idx` was "the j-th output index" —
that was wrong; the GGML_ASSERT in llama-sampler.cpp:850 is on
`batch.logits[idx] == true`. M2 scheduler.cpp records batch position.

**M1 implication**: M1's serial replay can use `llama_batch_get_one` for
prefill chunks (prompt tokens, all `seq_id=0`) and single-token decode steps —
simplest possible. M2 *must* migrate to `llama_batch_init` because
`llama_batch_get_one` doesn't accept a `seq_id` array. Document this as a
M1→M2 hand-off boundary so we don't accidentally lock the design into the
helper.

## Context multi-sequence (M2)

`llama_context_params` exposes two knobs that matter for batching:

| field | what it does | what M2 wants |
|---|---|---|
| `n_seq_max` | max distinct `llama_seq_id` the context tracks | set to `max_slots` (e.g. 8) |
| `kv_unified` | unified KV buffer across sequences | header recommends **`false` when n_seq_max > 1** if sequences don't share a large prefix; set false for M2 |

Defaults from `llama_context_default_params()` are `n_seq_max=1`. Forgetting
to bump it before `llama_init_from_model` will cause `llama_decode` to fail
with "could not find a KV slot" the first time we feed a token with
`seq_id != 0`.

`llama_n_seq_max(ctx)` returns the live value — useful sanity-check.

## Sampler

New namespace; "chain" pattern.

```cpp
auto sparams = llama_sampler_chain_default_params();
auto * smpl  = llama_sampler_chain_init(sparams);
llama_sampler_chain_add(smpl, llama_sampler_init_greedy());   // temp=0 path
// for each step:
llama_token tok = llama_sampler_sample(smpl, ctx, /*idx=*/-1); // -1 = last logits
llama_sampler_accept(smpl, tok);                                // feed back
// at end:
llama_sampler_free(smpl);  // frees chain + all chained samplers
```

For deterministic M1 baseline use `init_greedy`. For M2+ multi-slot we'll
need *one sampler chain per slot* (samplers carry per-stream state); spec
§1.1.1 already implies this via `Slot::seq_id`. The header has an
**experimental** `llama_set_sampler(ctx, seq_id, smpl)` that attaches a
sampler to a specific seq inside the context — worth investigating in M2,
not M1.

## Threading / safety

`llama_*` calls are **not** thread-safe across the same context. Spec §10
pitfall #7 already says this. Nothing has changed.

## What this means for M1 code

Concrete M1 surface (single slot, no batching, count-based LRU):

```cpp
auto mparams = llama_model_default_params();
auto * model = llama_model_load_from_file(args.model_path.c_str(), mparams);
auto * vocab = llama_model_get_vocab(model);

auto cparams = llama_context_default_params();
cparams.n_ctx     = args.n_ctx;
cparams.n_batch   = args.n_batch;
auto * ctx = llama_init_from_model(model, cparams);

auto * sparams = llama_sampler_chain_default_params();
auto * smpl    = llama_sampler_chain_init(sparams);
llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

// adapter cache
struct AdapterEntry { struct llama_adapter_lora * h; /*...*/ };
// on acquire(id): if not resident, llama_adapter_lora_init(model, path);
//   then llama_set_adapters_lora(ctx, &h, 1, &scale);
// on release: ref_count-- (still resident in cache); evict by LRU when over count cap.

// per request:
//   prefill: build llama_batch via llama_batch_get_one, llama_decode
//   decode loop: sample via llama_sampler_sample, accept, single-token llama_decode,
//                terminate on llama_vocab_is_eog or max_output
//   on finish: llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);  // clear KV
```

That's the minimum API surface M1 actually exercises. Everything else
(M2 batching with multiple seq_id, M3 size-aware byte budget, M5 custom
LoRA kernel) layers on top.
