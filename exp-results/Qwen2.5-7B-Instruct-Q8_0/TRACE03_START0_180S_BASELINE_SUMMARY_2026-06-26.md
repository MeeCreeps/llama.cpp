# Qwen2.5-7B-Instruct-Q8_0 Trace03 Start0 180s Baseline Summary

Generated: 2026-06-26

## Setup

- device: `3C15AU002CL00000`
- remote dir: `/data/local/tmp/hyzheng/elastic`
- model: `Qwen2.5-7B-Instruct-Q8_0.gguf`
- trace: `trace_03_user_116_cap7300.csv`
- source window: start `0s`, span `180s`
- window stats: min/mean/max `5954.0 / 6769.7 / 7300.0 MiB`
- buckets: min `5888 MiB`, max `7424 MiB`
- placement family: `gpu,disk_gpu`
- timed decode: `180s`

## Headline Baselines

This table uses the original all-baseline run for `static-min`, `static-max`,
`offline`, and `diff-tree-ideal`, the latest fixed `online` run with remote CP,
transition horizon amortization, Q8 stable pool path, and `-b/-ub 32`, plus the
strict MRU retest that uses the original `kv=512` budget accounting after
enabling O_DIRECT source reads in the Q8 SOA reload path.

`mru runtime-cache` is included because it does perform real runtime misses,
evictions, and layout conversion. Its boundary is still different from the
planner baselines: it has no planner-visible movement timeline
(`planned evict/load/xfer/xform=0/0/0/0`), so interpret it as a runtime-cache
MRU baseline, not as an offline/online planner with explicit load/xfer
scheduling. After the Q8 direct-read fix it does report O_DIRECT disk reads.

| method | source artifact | status | raw ms/token | exec ms/token | eval runs | apply count | planned evict/load/xfer/xform | direct read MB | online calls/failures | GPU after C |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| static-min | `trace03_cap7300_start0_180s_gpu_disk_all` | ok | 2026.42 | 2026.42 | 89 | 0 | 0/74/74/74 | 177049.7 | n/a | 53.1 |
| static-max | `trace03_cap7300_start0_180s_gpu_disk_all` | ok | 348.51 | 348.51 | 516 | 0 | 0/22/22/22 | 431.4 | n/a | 51.5 |
| offline | `trace03_cap7300_start0_180s_gpu_disk_all` | ok | 645.15 | 679.28 | 266 | 9 | 0/602/602/602 | 10535.2 | n/a | 50.8 |
| diff-tree-ideal | `trace03_cap7300_start0_180s_gpu_disk_all` | ok | 439.39 | 439.24 | 413 | 3 | 0/85/40/85 | 2891.3 | 3/0 | 56.2 |
| online remote CP, fastest fixed | `trace03_cap7300_start0_180s_online_timeline_kv256_guard80_tl50_b32` | ok | 479.74 | 437.67 | 349 | 7 | 0/154/154/154 | 182990.4 | 7/0 | 52.7 |
| mru runtime-cache, Q8 direct-read kv512 | `trace03_cap7300_start0_180s_mru_q8direct_kv512_noretain_b32` | ok | 1085.56 | 1086.46 | 168 | 10 | 0/0/0/0 | 202249.8 | 10/0 | 44.2 |

## Online Fix Provenance

| online variant | artifact | status | raw ms/token | exec ms/token | eval runs | apply count | planned evict/load/xfer/xform | direct read MB | note |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| old all-baseline online | `trace03_cap7300_start0_180s_gpu_disk_all` | ok | 829.80 | 783.77 | 207 | 5 | 0/359/359/359 | 3207.4 | old cost/profile path |
| reprofile, retain500, no horizon amortization | `trace03_cap7300_start0_180s_online_reprofile_q8hybrid_retain500_remote_diskgpu` | ok | 1503.71 | 1498.75 | 120 | 1 | 0/53/53/53 | 2006.3 | over-penalized one-time transition, repeatedly selected equivalent plan |
| reprofile, horizon8, `-b/-ub 1` | `trace03_cap7300_start0_180s_online_reprofile_q8hybrid_retain500_remote_diskgpu_horizon8` | ok | 710.14 | 697.99 | 254 | 8 | 0/213/213/213 | 3250.2 | correct planner behavior but not batch-comparable to diff-tree |
| reprofile, horizon8, `-b/-ub 32` | `trace03_cap7300_start0_180s_online_reprofile_q8hybrid_retain500_remote_diskgpu_horizon8_b32` | ok | 479.89 | 472.37 | 375 | 7 | 0/188/188/188 | 3250.2 | previous fixed online result |
| timeline, `kv=256`, guard80, tl50, `-b/-ub 32` | `trace03_cap7300_start0_180s_online_timeline_kv256_guard80_tl50_b32` | ok | 479.74 | 437.67 | 349 | 7 | 0/154/154/154 | 182990.4 | fastest headline online result |

## MRU Runtime Detail

| method | artifact | status | raw ms/token | true-MRU accesses/hits/misses/evictions | runtime xform | parent alloc | write enqueue | direct read |
|---|---|---:|---:|---|---|---|---|---|
| mru runtime-cache, old all-baseline row | `trace03_cap7300_start0_180s_gpu_disk_all` | ok | 699.76 | `50568/43336/7232/9380` | `7232 calls, 100175.28 ms` | `32990 ms` | n/a | `0 calls, 0.0 MB` |
| mru runtime-cache, pre-Q8-direct retest | `trace03_cap7300_start0_180s_mru_retest_reprofile_retain500_b32` | ok | 376.03 | `95256/88028/7228/11045` | `7228 calls, 65716.40 ms` | `1785 calls, 26646.47 ms` | `7228 calls, 35549.11 ms` | `0 calls, 0.0 MB` |
| mru runtime-cache, Q8 direct-read fix | `trace03_cap7300_start0_180s_mru_retest_q8direct_retain500_b32` | ok | 587.68 | `60564/56083/4481/6823` | `4481 calls, 135868.69 ms` | `995 calls, 6691.51 ms` | `4481 calls, 44837.25 ms` | `4481 calls, 83746.88 ms, 222676.9 MB` |
| mru runtime-cache, Q8 direct-read fix r2 | `trace03_cap7300_start0_180s_mru_retest_q8direct_retain500_b32_r2` | ok | 580.63 | `61152/56643/4509/6875` | `4509 calls, 135438.10 ms` | `1005 calls, 6648.81 ms` | `4509 calls, 44075.59 ms` | `4509 calls, 84117.94 ms, 224413.5 MB` |
| mru runtime-cache, Q8 direct-read no-retain, kv128 | `trace03_cap7300_start0_180s_mru_q8direct_noretain_b32` | ok | 694.08 | `51156/47305/3851/5821` | `3851 calls, 133159.32 ms` | `3851 calls, 20865.61 ms` | `3851 calls, 40890.64 ms` | `3851 calls, 70972.55 ms, 190480.0 MB` |
| mru runtime-cache, Q8 direct-read no-retain, kv512 | `trace03_cap7300_start0_180s_mru_q8direct_kv512_noretain_b32` | ok | 1085.56 | `32928/28213/4715/6115` | `4715 calls, 138550.09 ms` | `4715 calls, 24078.45 ms` | `4715 calls, 38836.41 ms` | `4715 calls, 75174.51 ms, 202249.8 MB` |
| mru runtime-cache, CPU compute | `trace03_cap7300_start0_180s_mru_cpu_b32` | ok | 2342.03 | `15288/12618/2670/1762` | `7542 calls, 107955.31 ms` | n/a | n/a | `7542 calls, 107944.80 ms, 151005.3 MB` |

## Key Interpretation

- The fixed online CP result is now close to `diff-tree-ideal`: `479.74` vs `439.39 ms/token`.
- The earlier `710.14 ms/token` online number was not batch-comparable; it used `-b/-ub 1`.
- The earlier `1503.71 ms/token` online number was a cost-model bug: one-time transition cost was not amortized over a horizon, so online stayed near the initial low-residency state.
- Remaining gap versus `diff-tree-ideal` is mostly movement/runtime overhead: the fixed online run applies 7 plans and schedules 154 load/xfer/xform movements, while diff-tree applies 3 plans and schedules 85 load / 40 transfer / 85 xform movements.
- The initial MRU retest was too fast because Q8 SOA reload did not have the Q4-style O_DIRECT source-read branch; it uploaded from `host_ptr`/mmap and therefore reported `direct_read=0`.
- After adding Q8 direct-read source support, MRU reports real disk reload work. The comparable strict headline MRU number uses the original baseline budget accounting (`kv=512`, `misc+safety=730`): `1085.56 ms/token`, with `1203.9 MB/token` of O_DIRECT reload/transform. It still has no planner-visible load/xfer timeline because MRU is a runtime pre-op cache policy, not an ExecPlan movement schedule.
- The retained-pool MRU retests (`587.68`, `580.63 ms/token`) are useful diagnostics, but they are not used in the headline baseline table because they are a different runtime/cache-memory configuration from the strict `kv=512` baseline.
- CPU-compute MRU is much slower (`2342.03 ms/token`) even though it also performs O_DIRECT reloads (`151005.3 MB`). This confirms the sub-600 ms/token GPU MRU result is not a CPU fallback artifact; it comes from GPU compute plus runtime MRU cache behavior.

## Source Files

- final merged baseline CSV: `window_180s_baselines/baseline_results.csv`
- exact 180s trace window CSV: `window_180s_baselines/trace03_start0_180s.csv`
- exact 180s trace window plot: `window_180s_baselines/trace03_start0_180s.svg`
- static/offline/diff-tree raw results: `window_180s_baselines/raw_results/static_offline_diff_tree_results.csv`
- fastest fixed online raw results: `window_180s_baselines/raw_results/online_fastest_results.csv`
- strict MRU kv512 raw results: `window_180s_baselines/raw_results/mru_kv512_results.csv`
