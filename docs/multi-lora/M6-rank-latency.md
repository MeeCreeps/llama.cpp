# M6 follow-up — Per-token decode latency vs LoRA rank

A controlled latency study to inform the **fixed-rank vs adaptive-rank**
design decision: when a serving system can choose what rank a request
runs at, what is the latency curve over `R ∈ {8, 16, 32, 64, 128}`?

## Setup

- Device: OnePlus 12 (Snapdragon 8 Gen 3, Adreno 750, OpenCL 3.0)
- Base: Llama-3.2-3B-Instruct Q4_0, ngl=99, ctx=4096
- Bench: `multi-lora-bench-m5` with `--n-slots=1` (one in-flight at a
  time, no batching → measurements are pure per-request decode cost)
- Trace per rank: 5 identical-prompt requests
  (`"What are the main causes of climate change? Explain in 2-3
  sentences."`, 52 prompt tokens, `max_output=32`).
- Measurement: drop the first request (cold model + cold OpenCL
  kernel cache); take the **median** of the 4 remaining hot decodes.
  Latency reported as ms / output token.
- Two fusion modes per rank: `GGML_OPENCL_DISABLE_FUSION=1` (control)
  vs default (M5 iter-3 fused kernel).
- Adapter pool from M6: same 5 r-values, all targeting all 7
  projections, base = Llama-3.2-3B-Instruct.

## Numbers

| rank | fusion off (ms/tok) | fusion on (ms/tok) | speedup | tax over no-LoRA |
|---:|---:|---:|---:|---:|
| (no LoRA, M0b reference) | — | **79.3** | — | 0 |
| 8   | 95.68 | **82.26** | 1.16× | +2.9 ms (+3.7 %) |
| 16  | 96.89 | **82.91** | 1.17× | +3.6 ms (+4.5 %) |
| 32  | 98.53 | **84.90** | 1.16× | +5.6 ms (+7.0 %) |
| 64  | 99.03 | **89.63** | 1.10× | +10.3 ms (+13.0 %) |
| 128 | 96.97 |   97.88   | 0.99× | +18.6 ms (+23.4 %) |

Plot: `results/M6.5/latency_vs_rank.png`.

## Observations

1. **Without fusion, latency is rank-flat**: 96–99 ms/tok for every
   rank tested. The LoRA tax is ~17–20 ms/tok regardless of `R`.
   That makes sense — the un-fused chain pays 4 fixed kernel-launch
   overheads per LoRA-target projection, which dominate over the
   actual compute that depends on `R`.

2. **With M5 fusion, latency scales gently with rank up to the
   `LORA_R_MAX = 64` ceiling**: r=8→r=32 each cost ~3–5 ms/tok over
   no-LoRA, r=64 jumps to ~10 ms/tok over no-LoRA, r=128 drops back
   to the un-fused tax (because pattern matcher refuses to fire above
   the ceiling).

3. **Fusion benefit scales inversely with rank**:
   - r ≤ 32: ~16 % speedup vs un-fused
   - r = 64: ~10 % speedup
   - r = 128: 0 % (fusion falls back to the un-fused chain)

4. **The fixed-vs-adaptive question:** *adaptive rank only beats
   fixed rank if you can use ranks below 16 or above 64*. Within
   8 ≤ R ≤ 32 the fused-kernel latency curve is essentially flat
   (82–85 ms/tok); switching rank inside that band buys you almost
   nothing on latency. Outside the band there is real headroom:
   - **down to r=8**: a tiny 0.7 ms/tok over r=16 — not worth the
     fine-tune work
   - **up to r=128**: 15+ ms/tok penalty (fusion falls off the cliff)

   So the practical adaptive-rank knob on this hardware is "stay at
   or below 64 to keep the fusion path; bigger is much more
   expensive".

## What this means for adaptive-rank policy design

Three concrete policies become testable from the table:

- **Latency-budget-N policy**: given budget `B` ms/token, accept any
  rank where the fused-kernel curve is below `B`. From the table:
  - `B = 85` → max rank 32
  - `B = 90` → max rank 64
  - `B = 95` → max rank 64 (r=128 doesn't qualify; un-fused r=128 is
    97 ms but that loses the fusion benefit)
- **"Highest rank that fits the fusion ceiling"**: pick `R = 64`
  unconditionally — pays only +10 ms/tok over no-LoRA, gives the
  most LoRA capacity at fusion-on speed. Works well as a default.
- **"Quality-vs-latency mixed pool"**: deploy a single adapter at
  multiple ranks (r=16 + r=64), route easy requests to r=16 and
  hard ones to r=64. Saves ~7 ms/tok on every easy request,
  amortised across the workload. Requires a routing classifier
  (out of scope for M0–M6 per spec §0.2).

## Caveats

- All five r-values use **different fine-tunes** (different HF repos)
  because there is no widely-available "same task at multiple
  ranks" Llama-3.2-3B LoRA family. Output content varies; the
  **latency** measurement is unaffected (matmul cost is determined
  by tensor shapes, not by adapter weights).
- We did not measure **quality** as a function of rank. Spec §0.3
  treats quality only via "human spot check 5 outputs" — not a
  latency study question.
- All measurements at `--n-slots=1`. Multi-slot batched cases
  (M2 case) compound rank cost differently because B-side GEMM
  shape changes; that is a separate study.
- `LORA_R_MAX = 64` is **trivially raisable to 128** (16 more bytes
  of workgroup-local memory) — leaving the cliff at r=128 in place
  is the kernel author's deliberate caution, not a hardware limit.
  If a workload genuinely needs r=128 with fusion, lift the cap and
  re-measure.

## Adaptive-rank policy — actual wall comparison

The per-rank latency curve above tells you what one rank costs. The
real adaptive-rank question is: **if half my requests can use a low
rank (cheap) and half need a high rank (expensive), how much wall do
I save vs running everything at the high rank?**

Setup: 8 prompts, all `arrival_time=0`, `--n-slots=4`. The first 4
("easy" — `What is 2+2?`, `Capital of France?`, …) and the last 4
("hard" — `Explain quantum entanglement…`, `Why is the sky blue?
detailed reasoning.`, …) are deterministic — no router, the trace
itself encodes the policy.

Five policies on the same 8 prompts:

| policy | rank choice per req | wall (s) | tok/s | saved vs uniform-high |
|---|---|---:|---:|---:|
| fix_r8        | all r=8                       | 23.74 | 8.00 | — |
| fix_r64       | all r=64                      | 27.47 | 6.92 | (baseline for r=64) |
| **adaptive**  | r=8 × 4 + r=64 × 4            | **25.85** | 7.35 | **+5.9 % vs fix_r64** |
| fix_r128      | all r=128 (fusion declines)   | 42.90 | 4.31 | (baseline for r=128) |
| **adaptive_wide** | r=8 × 4 + r=128 × 4       | **36.96** | 5.14 | **+13.8 % vs fix_r128** |

Output token counts are identical across all five policies (190 total),
so the wall comparison is clean — no output-length confound.

### Reading the table

- **adaptive vs fix_r64 = 5.9 % wall saved.** Modest, because the
  per-token gap between r=8 and r=64 within the fusion path is only
  ~10 % (curve is flat in `8 ≤ R ≤ 32`, gentle slope to `R = 64`).
  Half the requests at r=8 saves ~10 % on those, halved across the
  whole batch ≈ 5 % wall.
- **adaptive_wide vs fix_r128 = 13.8 % wall saved.** Big jump because
  r=128 falls out of the fused path and pays the full ~18 ms LoRA
  tax — gap r=8 vs r=128 is ~16 ms / token. Half the requests
  shifted off the cliff buy back substantial wall.

### When does adaptive rank actually help?

**Roughly: only when the cheap-to-expensive rank gap is ≥ 10 % per
token.** That's the case for:

- **r=8 vs r=128** (one side fused, the other not) — 14 % wall save
- **r=8 vs r=64** (both fused but at different cost) — 6 % save
- **r=8 vs r=16** — basically zero save (the fused-kernel curve is
  flat below R = 32)

So the actually-useful "adaptive" knob on this hardware is
**"escape the fusion ceiling"**: keep most requests at r ≤ 64 to
benefit from M5 fusion, only push the requests that genuinely
need higher capacity to r=128+ ranks.

The other adaptive direction — varying rank in the 8…32 band based
on workload — is **not worth the routing infrastructure**, the wall
delta is below noise.

### Caveats on this comparison

- 8-prompt traces; not paper-quality sample size. The `fix_r128`
  measurement is especially noisy because most of its 42.9 s is the
  4 r=128 requests serial-batching at +18 ms/tok, which is sensitive
  to OS scheduling.
- All five policies use the SAME prompts. If "easy" reqs were
  shorter and "hard" reqs were longer, the adaptive savings would
  differ accordingly. We didn't model that here.
- We did NOT measure output **quality** between r=8 and r=64 — the
  premise is that the workload owner already knows which ranks
  they need for which requests. The "adaptive policy" is the
  classifier; we just measured the latency consequence.
- Routing classifier itself (deciding easy-vs-hard at admit time)
  is spec §0.2 scope-cut; the trace pre-encodes the policy
  decision.

## Files

- `workloads/m65-{r8,reasoning,r32,r64,r128}.json` — per-rank latency traces
- `workloads/m65b-{fix_r8,fix_r64,fix_r128,adaptive,adaptive_wide}.json`
  — adaptive-policy traces
- `results/M6.5/m65/{r8,reasoning,r32,r64,r128}_{off,on}.csv` — bench output
- `results/M6.5b/m65b/*.csv` — adaptive-policy bench output
- `results/M6.5/latency_vs_rank.png` — the plot
