# Project: Mobile Multi-LoRA Serving Bench (on llama.cpp + OpenCL)

This is a research codebase forked from llama.cpp to build an end-to-end
multi-LoRA serving system on Android phones (Snapdragon Adreno GPU via OpenCL).

## Single Source of Truth

`docs/multi-lora/IMPLEMENTATION_GUIDE.md` is the spec. **Always read it before making changes.**
Specifically:
- §0   for goals + scope cuts
- §3.4 for files you must NOT modify
- §4   for milestones (M0..M4) and their acceptance criteria
- §5   for component design
- §10  for known pitfalls
- §12  for your working mode (decision rights, when to stop and ask)

When the spec contradicts reality, **fix the spec first**, then the code.

## Milestone Discipline

- Only work on the milestone the user explicitly names
- Do not implement features from a future milestone
- Each milestone goes on its own git branch (e.g. `m1-pool`, `m2-scheduler`)
- A milestone is "done" only when its acceptance criteria pass

## Hard Don'ts

These are forbidden without explicit user confirmation:
- Modifying `ggml.c`, `ggml/src/ggml-opencl/*`, or any file under `ggml/`
- Changing the GGML compute graph or batch token layout
- Modifying KV cache implementation in `src/llama-context.cpp` or
  `src/llama-kv-cache*`
- Adding new third-party dependencies (PRs that vendor a new lib are nyet)
- Switching backend (sticking to OpenCL only; no NPU / Metal / Vulkan)
- Adding HTTP / TCP / gRPC interfaces (single binary, in-process only)
- Multi-threading the main loop (single-thread main loop is the design)

## Architecture Reminders

The whole bench is **one binary, one thread**:

```
load trace JSON → main loop (admit + step) → dump CSV
```

No futures, no promises, no async, no sockets. If you find yourself
reaching for `std::future` or `std::thread`, **stop and re-read §1.1**.

## Stop and Ask If

- A spec section seems wrong or contradicts another section
- An llama.cpp internal API doesn't match the spec's description
- Acceptance criteria for the current milestone seem unreachable
- A change you want to make crosses a milestone boundary
- You're tempted to add a new file not listed in the spec

Don't guess. Stopping costs minutes; the wrong design costs days.

## After Each Milestone

1. Write `docs/multi-lora/M{i}.md` with implementation notes (~150–300 words)
2. Add "Implementation Notes" subsection to that milestone in
   `docs/multi-lora/IMPLEMENTATION_GUIDE.md` recording any spec deviations
3. Verify acceptance criteria, paste output into the PR description
4. Open PR with title `feat(m{i}): <one-line summary>`
5. Stop. Wait for user review before starting M{i+1}.

## Code Style

- Follow llama.cpp conventions (snake_case functions, `llama_*` namespace)
- New components use `multilora_*` prefix to avoid collisions
- C++17, no C++20
- `clang-format` with the existing repo config

## Environment

- Target device: Android phone with Snapdragon 8 Gen 3+ (Adreno GPU)
- OpenCL 3.0 backend (`-DGGML_OPENCL=ON -DGGML_OPENCL_TARGET_VERSION=300`)
- Base model: Llama-3.2-3B Q4_0 GGUF (~2 GB)
- Adapters: r=16 LoRA, GGUF format
- Build: cross-compile from macOS/Linux via Android NDK r26+

## When in Doubt

Read `docs/multi-lora/IMPLEMENTATION_GUIDE.md` again. If the answer isn't there, ask.
