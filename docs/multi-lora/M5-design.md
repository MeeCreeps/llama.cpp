# M5 Design — Custom OpenCL fused LoRA-delta kernel

**Status**: phase 1 design doc. **No code written yet.** Stop and ask the user
to review before phase 2 (kernel implementation).

## Why M5

M2 acceptance §"throughput ≥ 3× M1" and M3 acceptance §"TTFT p99
monotonically decreasing with budget" both **failed on Adreno OpenCL** —
our M2.md / M3.md flag them as hardware-conditional. Both root-cause to
the same thing: at this model size on this GPU, decode is **memory-
bandwidth-bound** rather than compute-bound, and the LoRA delta path
specifically suffers from a thin intermediate tensor `(seq, R=16)` that
gets written to and read from DRAM every layer-projection.

Custom kernel work is the unlock that spec §6.3 / §11 explicitly calls
out as M5+ stretch.

## Where LoRA actually lives in llama.cpp today

Two layers, easy to confuse:

1. **`src/llama-adapter.cpp`** — loads the GGUF adapter file into
   `ggml_tensor` pairs `{a, b}` keyed by base weight name. Pure I/O, no
   math. This is what `llama_adapter_lora_init` calls.
2. **`src/llama-graph.cpp:969 build_lora_mm`** — runs every time the
   graph is built (i.e. every `llama_decode`). For each base
   `ggml_mul_mat(W, x)` projection in the model, it asks the active
   adapter set "do you have an `(A, B)` pair for this `W`?" and if yes,
   inserts the delta:

   ```cpp
   ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);            // node[i+0] base
   for (lora in *loras) {
       ggml_tensor * tmp_a = ggml_mul_mat(ctx0, lw->a, cur);  // node[i+1] (seq, R)
       ggml_tensor * tmp_b = ggml_mul_mat(ctx0, lw->b, tmp_a);// node[i+2] (seq, H_out)
       tmp_b   = ggml_scale(ctx0, tmp_b, scale);              // node[i+3]
       res     = ggml_add  (ctx0, res, tmp_b);                // node[i+4]
   }
   ```

   For Llama-3.2-3B with 28 layers × 7 LoRA-target projections (q, k, v,
   o, gate, up, down) and 1 active LoRA, this expands the graph by
   **28 × 7 × 4 = 784 extra GGML nodes per forward pass**.

3. **`ggml/src/ggml-opencl/`** — the OpenCL backend. Contains zero
   strings matching `lora` (verified by grep). It just executes the
   GGML ops it is given. The LoRA-vs-base distinction is invisible at
   this layer.

That third point is the design opportunity: the OpenCL backend
**already has fusion-pattern matching** at `ggml-opencl.cpp:3936
ggml_opencl_can_fuse(...)`. Existing patterns:

```c
{ GGML_OP_NORM,       GGML_OP_MUL, GGML_OP_ADD }
{ GGML_OP_GROUP_NORM, GGML_OP_MUL, GGML_OP_ADD }
{ GGML_OP_RMS_NORM,   GGML_OP_MUL              }
```

Each pattern dispatches a hand-written single-kernel implementation
(`ggml_opencl_op_norm_fused`, etc.) instead of running the ops one by
one. We add a fourth pattern for LoRA delta.

This means **M5 can be implemented entirely under `ggml/src/ggml-
opencl/`** without touching `ggml.c`, the GGML core op table, or
`src/llama-graph.cpp`. CLAUDE.md still treats `ggml/src/ggml-opencl/*`
as a Hard Don't, but the user has explicitly approved crossing that
boundary for M5; the rest of the GGML core stays unchanged.

## What the kernel does

Inputs (per LoRA-target projection per layer):

| | shape | dtype | source |
|---|---|---|---|
| `x` | `(B*S, H_in)` | f16 / f32 | layer activation (already on GPU) |
| `A` | `(R, H_in)`   | f16       | LoRA weight A (47 MiB total per adapter ≈ 1.7 KiB per layer-projection at R=16) |
| `B` | `(H_out, R)`  | f16       | LoRA weight B |
| `scale` | scalar    | f32       | adapter alpha / R |
| `base` | `(B*S, H_out)` | f16   | the result of `x @ W_base` from a *prior* GGML node |

Output:

| | shape | dtype |
|---|---|---|
| `y` | `(B*S, H_out)` | f16 |

Math: `y[b, h_out] = base[b, h_out] + scale * Σ_r B[h_out, r] * Σ_h A[r, h] * x[b, h]`

Two ways to schedule, in order of ambition:

### Approach A — fuse the two LoRA matmuls only (recommended start)

Match pattern `{ GGML_OP_MUL_MAT (with R=16-class output), GGML_OP_MUL_MAT (followed by) }`
where node[i+1] consumes node[i] as src[1], and the intermediate output
shape is small (e.g. `≤ 64`). One fused kernel computes
`tmp_b = B @ (A @ x)` keeping the `(B*S, R)` intermediate in **work-
group local memory** (Adreno 750 has ≥ 32 KiB local mem per workgroup,
huge headroom — even seq=512 × R=16 × 2 bytes = 16 KiB).

The subsequent `scale` and `add` stay as their own ops; the existing
add fuses cheaply with read-modify-write because the dst tensor is
already on GPU.

Net saved: 1 kernel launch + 1 DRAM write+read of the thin
`(seq, R=16)` intermediate per LoRA-target projection.

For Llama-3.2-3B that is 28 × 7 = 196 saved DRAM round-trips per
forward pass. Each round trip moves `seq × 16 × 2` bytes per
direction, so 196 × seq × 64 B/forward. For seq=1 (single decode):
12.5 KiB/forward, negligible. For seq=64 (batched prefill or
n_slots=8 batched decode): 800 KiB/forward — already significant
but not huge.

The bigger win is **kernel launch amortisation**: 196 fewer launches
× ~50 µs Adreno launch overhead ≈ 10 ms/forward saved. At 10 tok/s
decode (100 ms/token), this is ~10 % wall reduction directly.

### Approach B — fuse all four ops including the base matmul

Match pattern `{ GGML_OP_MUL_MAT (base), GGML_OP_MUL_MAT (A), GGML_OP_MUL_MAT (B), GGML_OP_SCALE, GGML_OP_ADD }`
where the final `add` consumes the base `mul_mat` and the scaled `B @
(A @ x)` chain. One kernel computes `y = x @ W + scale * (x @ A) @ B`
in one launch.

Wins on top of Approach A:
- saves the base matmul kernel launch + the scale + the add (3 more
  launches per projection)
- can keep `base = x @ W` partial sums in local memory and fold the
  LoRA delta into the same accumulator before writing once

But this **modifies the base matmul kernel path**, which is the most
heavily Adreno-tuned code in the backend (`ggml_cl_mul_mat_q4_0_*`
variants, ~600 lines each). Risk of regressing the no-LoRA path is
real.

Approach B is M5-paper. Approach A is M5-acceptable.

### Pattern-match constraint check

For our specific use case (M2 same-adapter batching, M3 byte-budget
LRU), there is at most **one** active LoRA per `llama_decode`. So the
inner `for (lora in *loras)` loop in `build_lora_mm` runs at most once,
producing a stable `mul_mat A → mul_mat B → scale → add` 4-tuple per
projection. The pattern matcher just needs to:

1. node[i] is `GGML_OP_MUL_MAT` and `node[i]->src[0]` is the LoRA-A
   tensor (we identify by checking the second dim is small, e.g.
   `≤ 64`, OR by tensor name prefix `*.lora_a` if names are preserved
   through the graph)
2. node[i+1] is `GGML_OP_MUL_MAT` and `node[i+1]->src[1] == node[i]`
3. Approach A stops here. Approach B additionally requires:
   node[i+2] is `GGML_OP_SCALE` consuming node[i+1], node[i+3] is
   `GGML_OP_ADD` whose other src is some earlier `GGML_OP_MUL_MAT`
   (not necessarily adjacent — the base matmul).

Tensor-name-based detection is the cleanest signal because
`llama-adapter.cpp:373-374` does `ggml_set_name(tensor_a, w.a->name)`
where `w.a->name` ends in `.lora_a`. Names propagate through GGML
ops (with op-specific suffixes), so the matcher can assert
`strstr(node[i]->src[0]->name, ".lora_a") != NULL`. Robust and
explicit.

## OpenCL kernel sketch (Approach A)

`ggml/src/ggml-opencl/kernels/lora_delta_f16.cl`:

```c
// Computes tmp_b[b, h_out] = sum_{r in 0..R-1} B[h_out, r] * sum_{h in 0..H_in-1} A[r, h] * x[b, h]
// Caller stages on host:
//   - x : f16 [B*S, H_in]   (layer input, already on GPU)
//   - A : f16 [R, H_in]     (small, fits in local mem)
//   - B : f16 [H_out, R]    (also fits if R is small)
//   - tmp_b : f16 [B*S, H_out]   (output)
// Workgroup layout: one workgroup per (B*S) row × tile of H_out.
// Local mem holds the (B*S_tile, R) intermediate.
//
// R is fixed at compile time via GGML_OPENCL_LORA_R macro; spec'd at
// 16 for our adapters. R=8/32/64 specialisations can be added later.

#define LORA_R 16

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void lora_delta_f16(
    __global const half * x,        // [N, H_in]
    __global const half * A,        // [R, H_in]
    __global const half * B,        // [H_out, R]
    __global       half * tmp_b,    // [N, H_out]
    const int N,                    // B * S (rows of x)
    const int H_in,
    const int H_out)
{
    const int row    = get_global_id(0);   // 0..N-1
    const int h_out  = get_global_id(1);   // 0..H_out-1
    if (row >= N || h_out >= H_out) return;

    // Step 1: each work-item computes one entry of (x @ A) for its row, all R columns
    // Stored in private memory (registers).
    half intermediate[LORA_R];
    #pragma unroll
    for (int r = 0; r < LORA_R; ++r) {
        half acc = 0;
        for (int h = 0; h < H_in; ++h) {
            acc += x[row * H_in + h] * A[r * H_in + h];
        }
        intermediate[r] = acc;
    }

    // Step 2: dot intermediate (size R) with B[h_out, :] (size R)
    half acc = 0;
    #pragma unroll
    for (int r = 0; r < LORA_R; ++r) {
        acc += B[h_out * LORA_R + r] * intermediate[r];
    }
    tmp_b[row * H_out + h_out] = acc;
}
```

This is the **simplest** correct version. R=16 means 16 half registers
of intermediate per work-item — trivial. The hot loop is the inner
`for (h = 0..H_in)` which we want vectorised.

Adreno-specific tunings to layer in iteratively (each is its own
benchmark step):
- `vload8(...)` / `dot()` on f16x4 — Adreno's vector ALU loves this
- subgroup-broadcast for `B[h_out, :]` — share across the 64-lane
  subgroup so each lane computes its own `h_out`
- Use **image1d_t** for A and B — Adreno's existing matmul kernels
  use texture path; same trick applies
- Specialise on `H_in == 3072` and `H_out == 3072` (Llama-3.2-3B
  hidden) so the loop bounds become compile-time constants

Goal: match or beat the existing per-op kernel chain on a
`(seq=8, H_in=3072, H_out=3072, R=16)` micro-benchmark before
integrating.

## Plumbing sketch

Files to add / edit (all under `ggml/src/ggml-opencl/`):

```
ggml/src/ggml-opencl/
├── ggml-opencl.cpp                      ~25 lines edit
│     - extend ggml_opencl_can_fuse() pattern set
│     - add ggml_opencl_op_lora_delta_fused()
│     - register the new kernel in load step (kernel_lora_delta_f16 etc.)
│     - dispatch hook in ggml_backend_opencl_graph_compute
│
└── kernels/
    └── lora_delta_f16.cl                NEW, ~80 lines
```

CMakeLists in `ggml/src/ggml-opencl/CMakeLists.txt` already discovers
`*.cl` files in `kernels/` automatically (verified by inspection of
existing kernel additions).

No changes to:
- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-cpu/`, `ggml/src/ggml-cuda/`, etc.
- `src/llama-graph.cpp` (build_lora_mm stays as is — it produces the
  same 4 nodes, the OpenCL backend just sees them as fusable)
- `src/llama-adapter.cpp`
- `examples/multi-lora-bench/`

Means: the new kernel only kicks in on OpenCL backend and only when
fusion is enabled (env var `GGML_OPENCL_DISABLE_FUSION=1` already
exists for A/B comparison).

## Verification plan

Per spec §7.4 byte-equality is mandatory.

### Correctness gate

Use the existing M2/M3 deterministic-arrival traces:
- `workloads/m2-mix3.json` (8 reqs, 3 adapters, all `arrival_time=0`)
- Sweep configurations:
  - **baseline**: `GGML_OPENCL_DISABLE_FUSION=1` (turns off all fusion,
    falls back to per-op kernels) ⇒ identical to current M3 binary
  - **fused-A**: default (LoRA fusion enabled, Approach A)
- Diff `--output-dir` per request, must be 8/8 MATCH

If divergence: kernel is buggy (most likely fp16 rounding order
mismatch). Iterate until 8/8.

### Performance gate

Same trace, same hardware (OnePlus 12, ngl=99), measure:

| metric | current M3 | M5 fused-A target | M5 fused-B target |
|---|---|---|---|
| TTFT mean (single decode) | (M3 baseline) | ≥ 5 % improvement | ≥ 15 % improvement |
| TPOT mean (batched n_slots=8) | (M3 baseline) | ≥ 5 % improvement | ≥ 20 % improvement |
| total wall on m3-scaling 41-req trace | 181 s (M3) | ≤ 172 s | ≤ 150 s |
| GGML node count emitted | (M3 baseline) | down by 196 | down by ~590 |

Approach A's ≥ 5 % improvement is the **lower bar** — the kernel
launch saving alone should hit it. If we don't, the fusion isn't
actually firing (run with `GGML_OPENCL_DISABLE_FUSION=1` and check
graph dump).

The unlock target — beating M2 acceptance ("3× throughput") — is
realistic only with Approach B + Adreno-specific tunings (subgroup
broadcast, image1d_t for A/B). Approach A alone won't deliver 3×.

### Diagnostic steps before committing kernel work

The right next step (before writing kernels) is **profile the current
LoRA path in isolation** to confirm where the time actually goes. The
expectation is "kernel launches dominate at small batch, intermediate
DRAM dominates at large batch", but we should measure rather than
assume. Suggested probes:

1. Run M3 sweep budget=0 with `GGML_OPENCL_PROFILING=ON` (CMake build
   flag) and dump per-op timings.
2. Compare `multi-lora-bench` with `--lora` set vs without on a
   non-LoRA baseline: gives a clean "LoRA tax" number (we already
   measured this in M0b — vanilla 12.61 tok/s vs LoRA 10.24 tok/s on
   batch=1, ~19 % tax).
3. Decode at increasing `--n-slots`: does LoRA tax stay flat (compute
   bound) or scale up (launch / DRAM bound)?

This is **2-3 hours** of measurement work. The numbers tell us
whether to proceed with Approach A, jump to Approach B, or rethink
the kernel design entirely (e.g. block-sparse base + LoRA fusion if
intermediate isn't the bottleneck).

## Estimated work breakdown

| phase | hours | output |
|---|---|---|
| 1 (this doc) | 2 | M5-design.md, no code |
| 2 profile current path | 2-3 | M5-profile.md, baseline numbers |
| 3 kernel skeleton (Approach A naive) | 4-6 | lora_delta_f16.cl + can_fuse hook |
| 4 byte-equality on m2-mix3 | 1-2 | passes 8/8 |
| 5 perf iteration on Adreno | 8-16 | tuned kernel, hits ≥ 5 % gate |
| 6 quantized A/B variants | 4-8 | f16/q4_0 / mixed types |
| 7 Approach B (optional) | 16-30 | fused base+delta kernel |
| 8 docs (M5.md + spec patch + commit) | 1-2 | landed |

Total Approach A floor: ~22 hours focused work. Realistic 3-5 days at
1-shift cadence given Adreno debugging cycles.

## Risks and stop conditions

- **GPU hangs on Adreno require phone reset.** First wrong index, first
  out-of-bound write, first division-by-zero in a kernel → device locks
  up. Save work after every successful build/run.
- **fp16 accumulation drifts**: small numerical differences in the
  inner loop reduction order can produce different argmax tokens at
  the sampling boundary. Spec §7.4 demands byte-equality; if we can
  only achieve token-equality up to N tokens then divergence, that's a
  hard-fail unless we widen the spec (which I would NOT do without
  user sign-off).
- **fusion doesn't fire**: pattern matcher's identification of which
  nodes are the LoRA chain is fragile to GGML graph-builder changes
  upstream. Adding tensor-name signature + a unit test that exercises
  `build_lora_mm` end-to-end mitigates.
- **Approach A delivers <5 % improvement**: would mean the kernel
  launches aren't the bottleneck, and we'd need Approach B (heavy
  matmul fusion) or look elsewhere (memory layout, SVM coarse vs fine
  grain). At that point: stop, update this doc, decide.

## What I'm asking for

Phase 1 done = this document.

**Stop here for review.** Phase 2+ writes OpenCL code under the
`ggml/src/ggml-opencl/*` Hard Don't. Three calls to make:

1. **Approach A or B?** Recommend **A** to start (smaller risk,
   smaller patch, clear correctness gate, M5-paper extension is
   straightforward).
2. **Profile first or skip to coding?** Recommend **profile first**
   (the 2-3 hour investment de-risks the whole milestone).
3. **Acceptance threshold**: I propose ≥ 5 % wall improvement on
   `m3-scaling.json` plus 100 % byte-equality with fusion-disabled
   baseline. That's the smallest gate that says "the kernel is doing
   something useful and not wrong". OK?

Once those three are answered, phase 2 starts on this branch
(`m5-kernel`) with no further design changes.
