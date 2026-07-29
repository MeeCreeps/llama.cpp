# Memory-elastic operator benchmark

This benchmark characterizes the stateful trade-off between Q4 weight
materialization and steady-state GEMV on one OpenCL device. It reports:

- wall-clock `set_tensor` materialization time;
- synchronized per-operation GEMV latency;
- batched GEMV latency with one final synchronization;
- raw and backend allocation sizes;
- a deterministic-output checksum.

The default path uses the Adreno transposed SOA layout. Set
`GGML_OPENCL_DISABLE_ADRENO_KERNELS=1` to measure the generic, non-transposed
SOA path with the same binary.

Example:

```sh
./llama-memory-elastic-op-bench \
  --k 4096 --m 14336 --warmup 20 --iters 100 --materialize-iters 1

GGML_OPENCL_DISABLE_ADRENO_KERNELS=1 \
./llama-memory-elastic-op-bench \
  --k 4096 --m 14336 --warmup 20 --iters 100 --materialize-iters 1
```

The current SOA backend replaces tensor metadata on the first `set_tensor`, so
one process intentionally measures one independent materialization sample.
Repeat the process externally to collect transition samples.

## Layout/compute coupling benchmark

`llama-memory-layout-compute-bench` is a mechanism prototype that controls the
layout, transformation algorithm, compute mapping, and fusion boundary in
self-contained Q4_0 OpenCL kernels. It includes:

- raw row-major AOS direct GEMV, with scalar-row and 16x16 tiled mappings;
- row-major and K-block-major split layouts;
- naive and local-memory-tiled AOS-to-block transformations;
- naive promotion+first-GEMV fusion;
- 16x16 tiled promotion+first-GEMV fusion that persists the resulting layout.
- FP32-scale Q4 and expanded-Q4 layouts with different persistent footprints,
  enabling a multi-layout memory-transition-compute frontier.

The benchmark checks every output against the raw-AOS path and separately
checks that a layout written by a fused kernel can be consumed by a later
standalone GEMV. It also reports horizon totals for direct execution,
separate promotion, and fused promotion.

```sh
./llama-memory-layout-compute-bench \
  --k 4096 --m 14336 --warmup 5 --iters 15
```

### Connected row-granularity pipeline

`--granularity-only` compares one full Q4_0 matrix with compact M-row
tiles while keeping the mathematical work fixed. Each tile uses one reusable
staging slot and executes the connected path

```text
O_DIRECT pread -> OpenCL buffer write -> tile-local layout transform -> Q4 GEMV
```

The default in-order path enqueues all three OpenCL commands and waits only at
the tile boundary, immediately before the compact buffers are reused.
`--granularity-stage-sync` is a diagnostic control that waits after every
stage. Allocation and correctness readback are outside the timed region.

```sh
./llama-memory-layout-compute-bench \
  --k 4096 --m 14336 --warmup 10 --iters 50 \
  --granularity-only --max-row-parts 32 \
  --load-file ./q4-k4096-m14336.bin --prepare-load-file
```

`--prepare-load-file` deliberately creates or overwrites the named file with
the deterministic benchmark matrix before reopening it with `O_DIRECT`.
Subsequent runs can omit that flag. `--reverse-row-parts` reverses the sweep
order to control for temperature and DVFS drift.

Reported peak sizes are application-controlled live allocations, not physical
DRAM measurements. On a unified-memory phone, `host_peak_mib` is the aligned
I/O staging allocation and `cl_buffer_peak_mib` is the sum of live OpenCL
buffers; `controlled_peak_mib` reports their sum. Fixed runtime, driver, and
OpenCL program allocations are not included.

This executable deliberately uses transparent research kernels rather than
claiming parity with the production Adreno kernels. Its purpose is to isolate
cross-layer mechanisms and attribution; production-kernel measurements remain
in `llama-memory-elastic-op-bench`.

## Dynamic trace cost replay

`replay-layout-trace.py` places measured promotion and GEMV costs into real
memory traces. It emits macro summaries, high-budget-window outcomes, and
budget-up/down event logs.

```sh
python3 replay-layout-trace.py \
  --trace ../../trace/traces/trace_06_user_204.csv \
  --out-dir /tmp/layout-trace-replay \
  --duration-sec 180 --execution-interval-ms 200
```

The replay is explicitly a measured-cost characterization. A real runtime run
is still required to validate pressure-driven layout invalidation and overlap.

The OpenCL benchmark can execute a generated pattern directly on the phone.
Budget-up physically allocates/materializes the block layout and budget-down
releases it:

```sh
./llama-memory-layout-compute-bench --k 4096 --m 14336 \
  --trace-pattern trace06_p50.csv \
  --trace-policy greedy --trace-transform tiled
```
