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

## Files

- `workloads/m65-{r8,reasoning,r32,r64,r128}.json` — per-rank traces
- `results/M6.5/m65/{r8,reasoning,r32,r64,r128}_{off,on}.csv` — bench output
- `results/M6.5/latency_vs_rank.png` — the plot
