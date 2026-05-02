# Multi-LoRA Serving on llama.cpp + Adreno OpenCL — Quickstart

A 5-minute walkthrough of the multi-LoRA serving runtime built on top of
llama.cpp. End-to-end on a Snapdragon 8 Gen 3 phone (Adreno 750 GPU,
OpenCL 3.0): **3.09× faster than naive per-request `llama-completion`
invocation** on a 5-request mixed-adapter trace.

This folder bundles the demo data so you can replay the comparison
without re-running everything from scratch.

```
docs/multi-lora/quickstart/
├── README.md          ← you are here
├── reproduce.sh       ← one-shot script that re-runs the demo end-to-end
└── data/
    ├── demo-5req.json         ← 5-request trace (3 adapters mixed)
    ├── runtimes.txt           ← captured wall times + per-request notes
    ├── ours.csv               ← multi-lora-bench per-request metrics
    ├── ours.csv.summary.json  ← cache + run summary
    ├── ours_dump/             ← detokenized output, one file per request
    └── vanilla/               ← per-process llama-completion outputs
```

---

## What is this?

A single binary, `multi-lora-bench`, that:

1. Loads one base model (Llama-3.2-3B-Instruct Q4_0).
2. Reads a JSON trace of requests, each tagged with an `adapter_id`.
3. Holds an LRU cache of LoRA adapters (size-aware byte budget).
4. Routes each request to its adapter, batches same-adapter requests
   into one `llama_decode`, dumps per-request CSV metrics + optional
   detokenized output.

Single-thread, single-process. No HTTP, no IPC, no daemon — by design.
Runtime expects one workload, runs through it, exits. See
`../IMPLEMENTATION_GUIDE.md` for the full spec; this document is a tour.

---

## Setup (one-time, on the phone)

Prerequisites on the host:

- Linux x86_64 + Android NDK r26+ + CMake 3.22+ + adb
- `/home/myid/hz85760/env/env.sh` already configures NDK / CMake paths
  on this machine (`source` it before any cross-compile).
- Python venv at `/home/myid/hz85760/env/llama-py` with
  `transformers`, `torch`, `gguf`, `pandas`, `matplotlib`, `pyyaml`.
- A Snapdragon 8 Gen 3+ device with `adb` over USB.

Build the cross-compiled binary:

```bash
source /home/myid/hz85760/env/env.sh

cmake -S . -B build-android \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
    -DGGML_OPENCL=ON -DGGML_OPENCL_TARGET_VERSION=300

cmake --build build-android --target multi-lora-bench -j 8
```

Push assets to the phone (one-time, ~2 GB so use a cable):

```bash
DEV=5ae7a43d                     # or whatever `adb devices` shows
DEV_ROOT=/data/local/tmp/hyzheng/multi-lora

# directories
adb -s $DEV shell "mkdir -p $DEV_ROOT/{bin,lib,models/lora,workloads,results/m4}"

# binaries
adb -s $DEV push build-android/bin/multi-lora-bench  $DEV_ROOT/bin/
adb -s $DEV push build-android/bin/llama-completion  $DEV_ROOT/bin/

# OpenMP runtime (Android doesn't ship libomp)
LIBOMP=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so
adb -s $DEV push $LIBOMP $DEV_ROOT/lib/

# base model + 3 r=16 LoRA adapters (47 MB each)
adb -s $DEV push models/llama-3.2-3b-instruct-q4_0.gguf  $DEV_ROOT/models/
adb -s $DEV push models/lora/{reasoning,hebrew,summary}-r16-f16.gguf $DEV_ROOT/models/lora/

# convenience symlinks so adapter_id can stay short
adb -s $DEV shell "cd $DEV_ROOT/models/lora && \
    for a in reasoning hebrew summary; do \
        [ -e \$a.gguf ] || ln -s \$a-r16-f16.gguf \$a.gguf; done"
```

That gives you the layout the rest of this guide expects:

```
/data/local/tmp/hyzheng/multi-lora/
├── bin/{multi-lora-bench, llama-completion}
├── lib/libomp.so
├── models/
│   ├── llama-3.2-3b-instruct-q4_0.gguf
│   └── lora/{reasoning,hebrew,summary}.gguf
├── workloads/    (filled by the demo / run_all.sh)
└── results/      (filled at runtime)
```

`libOpenCL.so` is the Adreno vendor driver at `/vendor/lib64/libOpenCL.so`
— already on every Snapdragon device, no push.

---

## How a request goes through the system

```
   workload.json  ─────────►  multi-lora-bench (one process, one thread)
   [
     {
       "id": "req_00000",
       "arrival_time": 0.0,
       "adapter_id": "reasoning",
       "input_tokens": [128000, 128006, ...],   ← chat-templated + tokenized offline
       "max_output": 24
     },
     ...
   ]

                                  │
              ┌───────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ main loop                                                   │
   │   while (still_pending || sched.active_count() > 0) {       │
   │     for r in arrived requests, while slots open:            │
   │         sched.admit(r)            ◄── assigns a seq_id      │
   │     sched.step()                                            │
   │   }                                                         │
   └──────────────┬──────────────────────────────────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ scheduler.step()                                            │
   │   1. group active slots by adapter_id, pick largest group   │
   │      (tie-break: oldest admit)                              │
   │   2. AdapterPool::acquire(group's adapter)                  │
   │      ┌─ hit:  llama_set_adapters_lora(handle), 0 ms         │
   │      └─ miss: llama_adapter_lora_init (read 47 MB LoRA      │
   │                from UFS / page cache, alloc OpenCL buffer,  │
   │                ~80 ms warm), then bind                      │
   │   3. build llama_batch (one llama_batch_init, reused):      │
   │      - prefill slot:  feed remaining prompt tokens          │
   │                       (last with logits=1)                  │
   │      - decode  slot:  feed last sampled token (logits=1)    │
   │      record slot.sample_idx = batch.n_tokens - 1            │
   │   4. llama_decode(ctx, batch)                               │
   │   5. for each slot: tok = sample(smpl, ctx, sample_idx);    │
   │      append, check EOG / max_output → finalize / continue   │
   └─────────────────────────────────────────────────────────────┘

   on finalize: per-request CSV row, optional detokenized text dump,
                llama_memory_seq_rm to free the slot's KV.
```

The unit of caching is a **whole adapter file** (392 LoRA tensors). Each
adapter is 47 MB f16 GGUF on disk; `--max-adapter-mem-mb` controls how
many fit at once. `--n-slots N` controls the max concurrent in-flight
requests (== max concurrent `llama_seq_id`).

---

## The demo: vanilla per-process vs ours

5 requests across 3 adapters (4 reasoning, 1 hebrew via Pareto skew,
plus 1 summary inserted by hand for diversity), all `arrival_time=0`,
`max_output=24`.

To reproduce end-to-end, with assets in place:

```bash
# from repo root
./docs/multi-lora/quickstart/reproduce.sh
```

Or do it manually (this matches what `reproduce.sh` runs):

```bash
DEV=5ae7a43d
DEV_ROOT=/data/local/tmp/hyzheng/multi-lora

# push the trace + the vanilla loop script
adb -s $DEV push docs/multi-lora/quickstart/data/demo-5req.json   $DEV_ROOT/workloads/
adb -s $DEV push tools/workload-gen/scripts/demo_vanilla.sh       $DEV_ROOT/
adb -s $DEV shell "chmod +x $DEV_ROOT/demo_vanilla.sh"

# vanilla loop: 5 separate llama-completion processes
adb -s $DEV shell "$DEV_ROOT/demo_vanilla.sh"
# -> "VANILLA total ms: 105000" (approx)

# ours: one multi-lora-bench process
adb -s $DEV shell "
  mkdir -p $DEV_ROOT/results/demo/ours_dump &&
  cd $DEV_ROOT && export LD_LIBRARY_PATH=$DEV_ROOT/lib:\$LD_LIBRARY_PATH &&
  ./bin/multi-lora-bench \
      -m models/llama-3.2-3b-instruct-q4_0.gguf \
      -a models/lora \
      -w workloads/demo-5req.json \
      -o results/demo/ours.csv \
      --output-dir results/demo/ours_dump \
      --max-adapter-mem-mb 0 --n-slots 4 -c 4096 -ngl 99 --quiet"
# -> "OURS total ms: 34000" (approx)
```

### Captured numbers (this run)

| | wall time | per-request avg |
|---|---|---|
| **vanilla** (5 × `llama-completion`) | **105.0 s** | ~21.0 s |
| **ours** (multi-lora-bench, n_slots=4) | **34.0 s** | ~6.8 s |

→ **3.09× speedup**.

### Where the savings come from

Three sources, in order of magnitude:

1. **Model load amortisation**. Vanilla mmaps the 1.7 GB Q4_0 GGUF and
   inits the OpenCL backend once per request — ~3–4 s × 5 = 15–20 s of
   pure startup overhead. Ours pays it once.

2. **Same-adapter batching**. Two requests use the `reasoning` adapter
   and arrive at the same time. The scheduler bundles them into one
   `llama_decode` per step, halving decode wall for that pair. Visible
   in `ours.csv`: `req_00000` and `req_00003` have identical
   `first_token` (1.89 s) and `finish` (6.56 s) timestamps because they
   were processed in the same batch.

3. **Adapter cache**. The third reasoning request (`req_00004`) is a
   pure cache hit — `cache_hit=1, acquire_ms=0` in the CSV. Vanilla
   would have re-read the 47 MB LoRA file from page cache (~80 ms).

The cache-hit savings are small here because the trace is short and
the budget is unbounded. Where they matter is the M3 size-aware sweep
(see `../M3.md` for the curve as `--max-adapter-mem-mb` shrinks).

### Output equality

```
req_00000 (reasoning) MATCH
req_00001 (hebrew)    DIFFER  (different wording, same theme)
req_00002 (summary)   MATCH
req_00003 (reasoning) DIFFER
req_00004 (reasoning) DIFFER  (same answer, different prelude)
```

2 / 5 byte-equal vs vanilla. The mismatches are **floating-point
non-determinism in the OpenCL kernels under different batch shapes**:
vanilla runs at `n_batch=2048, n_seq_max=1, kv_unified=true` while ours
runs at `n_batch=512, n_seq_max=4, kv_unified=false`. Different ubatch
shapes feed the matmul kernels in different reduction orders; on
borderline argmax decisions the resulting top token can flip.

This is **not a scheduler bug**. The internal-consistency check —
running multi-lora-bench against itself with `--n-slots=1` vs
`--n-slots=8` on a deterministic-arrival trace — is **byte-identical
24 / 24** across the M2 and M3 milestones (see `../M2.md`, `../M3.md`).
The verification policy in `../IMPLEMENTATION_GUIDE.md` §7.4 covers
this distinction.

---

## Anatomy of `multi-lora-bench`

```
examples/multi-lora-bench/
├── main.cpp        ← arg parse, trace load, scheduler driver,
│                     CSV + summary.json + optional output_dir dump
├── slot.h          ← Request + Slot structs (header-only)
├── scheduler.h/cpp ← admit / step / pick_largest_adapter_group;
│                     llama_batch builder; sample_idx accounting
├── adapter_pool.h/cpp
│                   ← byte-budget LRU; pinned/free split lists
└── metrics.h/cpp   ← RequestMetric struct + CSV writer
```

CLI (most-used flags):

```
-m  --model                base GGUF (required)
-a  --adapter-dir          directory holding <id>.gguf adapters (required)
-w  --workload             pre-tokenized trace JSON (required)
-o  --out                  metrics CSV output path (required)
    --output-dir DIR       optional: dump <req_id>.txt per request
    --max-adapter-mem-mb N adapter cache byte budget (0 = unbounded)
    --n-slots N            max concurrent requests / batch slots (default 4)
-c  --n-ctx N              context size (default 4096)
-b  --n-batch N            tokens per llama_decode max (default 512)
-ngl N                     layers offloaded to GPU (default 99)
```

Full help: `./bin/multi-lora-bench --help`.

---

## Workload generation (host side)

`tools/workload-gen/gen_workload.py` produces the JSON trace
multi-lora-bench reads. Three pre-canned scenarios live in
`tools/workload-gen/configs/{idle_burst, typical, high_load}.yaml`,
each defining arrival rate, Pareto skew, and adapter pool. Quick run:

```bash
PY=/home/myid/hz85760/env/llama-py/bin/python
$PY tools/workload-gen/gen_workload.py \
    --config tools/workload-gen/configs/typical.yaml
# -> wrote 32 requests to workloads/m4-typical.json
```

The full M4 harness is in `tools/workload-gen/scripts/run_all.sh`:
loops every YAML in `configs/`, runs the bench on the device, pulls
the CSV + JSON sidecar, prints a summary, generates 4 plots
(TTFT CDF / throughput timeline / adapter access bar /
TTFT-vs-arrival scatter). See `../M4.md` for layout and example
output.

---

## What this implementation is and isn't

In:

- ✓ Multi-LoRA loading + per-request routing
- ✓ Size-aware LRU cache (`--max-adapter-mem-mb`)
- ✓ Same-adapter batching (`--n-slots`)
- ✓ Per-request CSV metrics + run-level JSON summary
- ✓ Optional detokenized output dump for byte-equality regression
- ✓ Three reproducible scenarios + plot harness

Out of scope by design (CLAUDE.md / spec §0.2):

- ✗ HTTP / gRPC / network (single binary, single process)
- ✗ Cross-adapter mixed batches (same-adapter only)
- ✗ Dynamic adapter routing / classifier
- ✗ Streaming output
- ✗ NPU / Vulkan / Metal backends (OpenCL only)
- ✗ Multi-threaded main loop

For the things that *aren't* in scope but might be next:

- Custom OpenCL multi-LoRA kernel (see spec §6.3 / §11). Adreno's
  current kernel scales batched decode linearly with batch size, which
  is what muted the M2 throughput curve and the M3 TTFT-vs-budget
  curve. A custom SGMV-style kernel is the unlock.
- Real 50-adapter pool (M4 currently uses 3 distinct r=16 LoRAs).
- 600 s scenarios for paper-quality TTFT p99 / p99.9 stabilisation.
