# Milestone 13: stage event firing and immediate-stage baseline

Date: 2026-06-16
Device: `172.20.115.151:5555` (OnePlus 15 / Adreno 840)
Workspace on device: `/data/local/tmp/hyzheng/elastic`

## Goal

Close the gap where online plans contained timeline events but the runtime reported:

```text
fired=0 load=0 transfer=0 xform=0
```

This meant the online provider/apply path worked, but stage load/transfer/xform timings were not actually being collected.

## Fixes

### 1. Anchor selection for early weights

The native-greedy generator used `max(0, weight_id - 1)` as the event anchor. For `token_embd.weight` (`weight_id=1`), this anchored events to `rope_freqs.weight` (`op_id=0`), which was not requested by the OpenCL graph in the measured run.

The generator now uses:

```cpp
const int anchor_id = it.id > 1 ? it.id - 1 : it.id;
```

This lets early managed weights self-anchor instead of attaching to an anchor that may never fire.

### 2. Runtime capability guard

Some names in `model_meta.json` are logical plan names but are not exposed by a backend stage provider. Generating disk/stage events for those names produced `rc=-2` failures.

The native-greedy generator now marks an item as stage-manageable only if runtime state includes `LLAMA_ELASTIC_WEIGHT_DISK_AVAILABLE`. Non-manageable items are forced into the keep set and do not produce disk timeline events.

### 3. Narrow CPU backend override

Some special unquantized weights, especially `token_embd.weight`, are resident in CPU_Elastic even though the cost fallback could select GPU. The generator now allows quant-empty items to use the current runtime residency backend.

This avoids requesting GPU transforms for `token_embd.weight` when the available stage path is CPU_Elastic.

## Commands

Deferred anchor validation with trace:

```bash
adb -s 172.20.115.151:5555 shell 'cd /data/local/tmp/hyzheng/elastic && timeout 150 sh -c '"'"'LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 GGML_OPENCL_DEVICE=0 GGML_OPENCL_DISABLE_HOST_PTR=1 GGML_OPENCL_SVM_DISABLE=1 GGML_ELASTIC_BUDGET_CSV=budget_trace_decode_low_plateau.csv GGML_ELASTIC_PROFILE_CSV=run_online_native_lowplateau_n64_manageable_guard_trace_profile.csv LLAMA_ELASTIC_ANCHOR_TRACE=1 LLAMA_ELASTIC_ONLINE=1 LLAMA_ELASTIC_ONLINE_MODE=native-greedy LLAMA_ELASTIC_MODEL_META=model_meta.json LLAMA_ELASTIC_COST_DIR=profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 LLAMA_ELASTIC_ONLINE_WORK_DIR=online_work_native_lowplateau_manageable_guard_trace ./llama-cli -m Llama-3.2-3B-Instruct-q4_0.gguf -p "Summarize dynamic elastic memory planning for mobile LLM inference." -n 64 -ngl 999 --ctx-size 512 --batch-size 64 --ubatch-size 64 --no-warmup --temp 0.0 -no-cnv'"'"' 2>&1 | tee run_online_native_lowplateau_n64_manageable_guard_trace.log; echo EXIT:$?'
```

Immediate-stage baseline:

```bash
adb -s 172.20.115.151:5555 shell 'cd /data/local/tmp/hyzheng/elastic && timeout 150 sh -c '"'"'LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 GGML_OPENCL_DEVICE=0 GGML_OPENCL_DISABLE_HOST_PTR=1 GGML_OPENCL_SVM_DISABLE=1 GGML_ELASTIC_BUDGET_CSV=budget_trace_decode_low_plateau.csv GGML_ELASTIC_PROFILE_CSV=run_online_native_lowplateau_n64_immediate_stage_profile.csv LLAMA_ELASTIC_DEFER_STAGE=0 LLAMA_ELASTIC_ONLINE=1 LLAMA_ELASTIC_ONLINE_MODE=native-greedy LLAMA_ELASTIC_MODEL_META=model_meta.json LLAMA_ELASTIC_COST_DIR=profiles/android-opencl/Llama-3.2-3B-Instruct-q4_0 LLAMA_ELASTIC_ONLINE_WORK_DIR=online_work_native_lowplateau_immediate_stage ./llama-cli -m Llama-3.2-3B-Instruct-q4_0.gguf -p "Summarize dynamic elastic memory planning for mobile LLM inference." -n 64 -ngl 999 --ctx-size 512 --batch-size 64 --ubatch-size 64 --no-warmup --temp 0.0 -no-cnv'"'"' 2>&1 | tee run_online_native_lowplateau_n64_immediate_stage.log; echo EXIT:$?'
```

## Results

### Deferred anchor validation

- Exit: `0`
- Plan: `timeline=3`, `load=1`, `transfer=0`, `xform=1`
- Stage failures: `0`
- Anchor summary: `requests=12608 hits=110 duplicate=6930 fired=0`

The failure count is fixed, but this particular decode graph still does not request the `token_embd.weight` anchor during the measured decode section. Deferred overlap remains a separate follow-up.

### Immediate-stage baseline

- Exit: `0`
- Plan: `timeline=3`, `load=1`, `transfer=0`, `xform=1`, `stage_defer=0`
- Online calls/failures: `1 / 0`
- `gen_ms`: 1.746 ms
- `provider_get_ms`: 3.653 ms
- `apply_ms`: 363.526 ms
- Prompt eval: 26.87 ms/token
- Eval: 49.59 ms/token
- Total: 7960.51 ms / 77 tokens

Profile CSV stage rows:

| backend | kind | weight | ok | ms | MB |
| --- | --- | --- | --- | ---: | ---: |
| CPU_Elastic | LOAD | token_embd.weight | 1 | 203.966 | 308.2 |
| CPU_Elastic | XFORM | token_embd.weight | 1 | 135.688 | 308.2 |

No `TRANSFER` event is generated for the CPU_Elastic path.

## Current interpretation

- Online native-greedy now has a working immediate-stage baseline with real stage load/xform timings.
- Deferred stage overlap no longer fails, but the low-plateau trace does not fire the `token_embd.weight` anchor during decode. It needs either a better earlier anchor policy or a graph-level anchor hook that can fire embedding events before the decode-only graph skips that source.
- Direct disk-read timing is represented by the CPU_Elastic stage `LOAD` row in this run; the separate `DIRECT_READ` kind is not emitted for the plan-stage path.
