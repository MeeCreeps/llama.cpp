# M5 phase 2 — Profile current LoRA path on Adreno 750 OpenCL

Quantifies "how bad is the LoRA tax" before writing any kernels. Numbers
de-risk the design choices in `M5-design.md`.

## Setup

- Device: OnePlus 12 (Snapdragon 8 Gen 3, Adreno 750, OpenCL 3.0)
- Build: `build-android-prof/` configured with `-DGGML_OPENCL_PROFILING=ON`,
  release otherwise. Pushed `bin/{llama-completion,multi-lora-bench}-prof`
  + `lib_prof/lib*.so` to `/data/local/tmp/hyzheng/multi-lora/`.
- Workload: `llama-completion` invocation, identical settings except
  `--lora`, prompt 33 tokens (1 prefill forward) + 16 decode tokens
  = 17 total forward passes per run.
  ```
  ./bin/llama-completion-prof \
      -m models/llama-3.2-3b-instruct-q4_0.gguf \
      [--lora models/lora/reasoning.gguf] \
      -p '...' -n 16 --temp 0 --seed 42 --no-warmup \
      -c 4096 -ngl 99
  ```
- Profiling artefact: `cl_profiling.csv` (per-kernel-launch timings,
  one row per `clEnqueueNDRangeKernel` call) written to CWD on exit.
  Pulled to `results/M5/{no_lora,with_lora}/`.

## Aggregate

| | total launches | per forward | total kernel ms | per forward ms |
|---|---:|---:|---:|---:|
| no_lora  |  8 370 |   492 | 1 723.9 | 101.4 |
| with_lora| 21 698 | 1 276 | 2 179.4 | 128.2 |
| **delta**| **+13 328** | **+784** | **+455.5** | **+26.8** |

LoRA flips on:
- **+784 kernel launches per forward** (matches the spec arithmetic:
  28 layers × 7 LoRA-target projections × 4 GGML ops added = 784)
- **+26.8 ms of GPU time per forward**
- ~26 % over baseline per-forward GPU time

## What kernels carry the LoRA tax

Three kernel programs appear **only** when LoRA is active:

| kernel | launches/forward | ms/forward | µs / launch |
|---|---:|---:|---:|
| `kernel_mul_mat_f16_f32_1row` | 369 | 11.8 | 31.9 |
| `kernel_mul_mm_f16_f32_l4_lm` |  23 | 17.9 | 788.8 |
| `kernel_scale_f32_4`          | 196 |  0.58 | 3.0 |

Plus two existing kernels see more launches:

| kernel | no_lora → with_lora launches | extra ms | new µs / launch |
|---|---:|---:|---:|
| `kernel_add_row` | 898 → 4 037 | +7.8 | ~2.6 |
| `kernel_add`     |  54 →   247 | +7.1 | ~36.6 |

Mapping to the 4-node LoRA chain (`mul_mat A → mul_mat B → scale → add`)
inserted by `src/llama-graph.cpp:969 build_lora_mm`:

| chain step | matched OpenCL kernel | per-forward cost |
|---|---|---|
| `tmp_a = A @ x` (small thin output `(seq, R=16)`) | `mul_mat_f16_f32_1row` | 12 ms, 369 launches |
| `tmp_b = B @ tmp_a` (back to `(seq, H_out)`) | `mul_mm_f16_f32_l4_lm` | 18 ms, 23 launches |
| `scale * tmp_b` | `scale_f32_4` | 0.6 ms, 196 launches |
| `res = base + scaled_tmp_b` | `add` / `add_row` | ~1 ms, ~190+185 launches |
| **subtotal** | | **~32 ms / 1 050 launches per forward** |

(Why the per-forward kernel launch count for the matmul row sums to
369 + 23 = 392 instead of 196 — the `mul_mat_f16_f32_1row` kernel is
dispatched per-row of activations on Adreno's matmul path, not per
matmul. The 23-launch `mul_mm_f16_f32_l4_lm` path is the pre-existing
"long matmul" version. The split between them depends on input shape
heuristics in `ggml_cl_mul_mat_*` selection; we treat both as "the LoRA
matmul work" together.)

## Implications for Approach A

The design doc (`M5-design.md`) targets fusing the two matmuls into
one kernel. From the numbers:

- The **two matmul kernels combined** = 30 ms/forward of GPU time.
- A naive fused kernel that just merges the launches without any local
  memory or arithmetic-intensity gains buys back ≤ launch overhead =
  ~5 µs × 392 launches = 2 ms/forward. **Not enough** for the 5 % gate
  at 27 % LoRA tax of 128 ms/forward.
- The **real win** has to come from improved arithmetic intensity:
  keep the `(seq, R=16)` intermediate in registers / workgroup local
  memory, never write it to DRAM. For a fully decoded forward this
  is `(1, 16) = 32 B` per projection × 196 projections = ~6 KiB
  saved per forward direction, so ~12 KiB DRAM round-trip avoided —
  bandwidth savings alone are negligible at decode.
- The actual matmul `mul_mat_f16_f32_1row` runs at **31.9 µs / launch
  for tiny (seq=1, H=3072) → (1, R=16)** matmul. The Adreno launch
  overhead floor is ~5–10 µs per launch in production. So `mul_mat A`
  is **~3-5× over launch floor** in cost — meaning launches dominate.
  Fusing 2 matmuls into 1 launch each cuts this in half: target ~6
  ms/forward instead of 12 (plus the 18 ms `mul_mm` reduces somewhat
  too because it can do its work without waiting on the round-trip).

**Conservative Approach A perf model:**
- Current LoRA matmul work: ~30 ms/forward
- Fused: ~15 ms/forward (50 % cut, plausible based on launch-dominance)
- Saved per forward: ~15 ms → **~12 % of total per-forward time**
- On the 41-req `m3-scaling.json` trace (181 s wall): could save
  ~21 s → **~12 % wall reduction** → ✅ comfortably above the 5 % gate.

**Optimistic case** (kernel ends up matching `mul_mm` quality + saves
DRAM completely): could approach 20 % wall reduction. Still under the
3× M2 throughput target spec wanted, which is unattainable without
backend-wide rework (Approach B + Adreno-specific tunings).

## Implications for Approach B

Approach B fuses everything (base matmul + LoRA delta) into one kernel.
The base matmul (`gemv_noshuffle*` family for q4_0) takes ~520 ms /
17 forwards = ~30 ms/forward — comparable in magnitude to the LoRA
chain (32 ms/forward).

If we could fuse base + LoRA into one kernel that runs at the same
speed as base alone, we save the entire 32 ms LoRA tax → ~25 % wall
reduction. That is the M2 throughput unlock.

But this requires modifying the heavily-tuned `gemv_noshuffle_q*_K_f32`
kernels, each of which has Adreno-specific image1d_t paths and
subgroup-broadcast tricks. Risk of regressing the no-LoRA path is
significant.

**Recommendation stays as in M5-design.md**: do A first (low-risk,
~12 % wall reduction expected). If the result is ~10–12 %, that's a
good-enough M5 milestone landing — leave B for an M5-paper follow-up.

## Artefacts

- `results/M5/no_lora/cl_profiling.csv` — 8 370 rows, baseline
- `results/M5/no_lora/cl_trace.json`    — Chrome-trace of same run
- `results/M5/with_lora/cl_profiling.csv` — 21 698 rows, with reasoning LoRA
- `results/M5/with_lora/cl_trace.json`

To re-aggregate later:
```python
import pandas as pd
df = pd.read_csv("results/M5/with_lora/cl_profiling.csv", skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"op name":"op","kernel name":"kernel","exec duration (ms)":"ms"})
df.groupby("kernel").agg(n=("ms","count"), ms=("ms","sum")).sort_values("ms", ascending=False).head(15)
```
