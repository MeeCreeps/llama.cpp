# OP12 v21 granularity experiment manifest

- Branch: `feature/elastic-working-unit-granularity`
- Source commit base: `777d2802c50dc6da2143c21146e5b5121fd846d0`
- Device: OnePlus 12, adb serial `5ae7a43d`
- Model: `Meta-Llama-3-8B-Instruct.Q4_0.gguf`
- Android binary SHA-256:
  `09af13e72384cf61c0ac3ce342a0f284267b8d25394e5c1bca3ef6ce2fef936e`
- Model GGUF SHA-256:
  `2b4675c2208f09ad8762d8cf1b6a4a26bf65e6f0641aba324ec65143c0b4ad9f`
- Model metadata SHA-256:
  `55c9f06709a30bd3b8bb07ff8dc0e3826ad1ceb88a55ab0db3e76742524304cc`
- Historical raw absolute-budget 600-second trace SHA-256:
  `ac69401a9089b7371921dcdfd8040c12909eb87eac25fa90986af6ccf2cc30ec`
- Final CPU and GPU consolidations are written to separate `cpu/` and `gpu/`
  subdirectories under the final artifact root. A later backend can never
  overwrite the first backend's CSV, Markdown summary, or JSON audit.
  CPU consolidation runs immediately after the CPU Diff-before row, before
  any GPU smoke/calibration/trace work, so a later GPU failure cannot delay or
  hide an already complete CPU result.
- Normalized host replay-input SHA-256:
  `95ebfa2c8ca60d9f1adac25a1856a40269011b7eb8c8bc75533df4cb6e1306f2`
- The production CPU/GPU wrappers pin both files and fail unless their 602
  ordered `(time, budget)` points are exactly equal. The normalization changes
  only the CSV header/number formatting. Window generation writes the runtime
  `t_sec,mem_available_mb` schema and, for this 0--600 s 1x window, reproduces
  the historical raw file byte-for-byte (`ac6940…30ec`). Held-out
  pilot/boundary wrappers independently pin their own derived input hashes.
- Trace replay: original `0--600 s`, `1x`; no time compression or
  min/max remapping.
- Each result records both the common source-window hash and the effective
  runtime-trace hash. MRU/Offline/Online/Diff-before/Diff-now must replay the
  generated dynamic window byte-for-byte. Static-Min/Static-Max must instead
  replay 600-second constant traces at the source window's bucketed
  minimum/maximum; the final audit verifies both constants explicitly. Every
  pushed runtime trace is hashed on OP12 before a row starts, resume requires
  the current source/runtime hashes, and consolidation rejects any device
  trace hash that differs from its archived host file. Rows separately retain
  the historical raw source-file hash above, the normalized host-input hash,
  and the generated runtime-trace hash,
  so schema conversion cannot be mistaken for budget remapping. The fixed 30%--90%
  runner applies the same device-side hash contract to every constant-budget
  CSV before calibration can consume it.

## Fairness contract

- All granularity modes use the same physical weight target at each fixed
  ratio, the same `token_embd`/`output` policy, and the same 128 MiB future
  byte window.
- A fixed-budget point is valid only when all measured granularities produce
  the same complete decode-token sequence and token count, not merely the same
  short prefix.
- Fixed-budget ratios use the exact physical WBM total
  `4154.9765625 + 281.8125 = 4436.7890625 MiB`, rather than the earlier
  rounded 4437.8 MiB constant.
- `token_embd` must contribute `281.8125 MiB` and `output` must contribute
  `410.9765625 MiB` to the ordinary weight budget in every mode. A missing,
  partial, or outside-budget pin invalidates the run.
- The 4154.9765625 MiB planner metadata already contains `output` but omits
  `token_embd`. The planner therefore subtracts the 281.8125 MiB embedding
  reserve once, while both CPU and OpenCL WBM targets subtract only the common
  base safety reserve; WBM resident-byte accounting charges the pinned
  embedding itself. This prevents GPU from double-counting the same pin.
- CPU and GPU are evaluated independently. No heterogeneous placement is
  enabled.
- Fixed calibration and dynamic planning provision the same Cut-capable
  physical base tiles for every logical mode. Tensor and Multi retain both
  tiles of each logical tensor atomically; only Cut may schedule/retain a tile
  independently. Every fixed and dynamic row must report a nonzero physical
  cut-capable tensor count and exactly two parts per such tensor, and the
  fixed summary rejects any per-budget cross-mode mismatch. Thus the cost
  model is calibrated on the physical representation it later controls. The
  calibration consumer independently rejects missing/mixed tiling, a WBM
  total change, or different physical targets across modes at the same
  budget, so a hand-assembled CSV cannot bypass the producer-side gate.
- Every GPU logical mode shares the same allocation-free two-tile physical
  Q4 representation and submits its two row halves through one fused
  dual-half GEMV; Cut differs only in its logical pipeline frontier. Q4 image
  views are created once per resident tile lifetime and reused across calls.
  WBM eviction releases each owned image before the underlying q sub-buffer,
  so this optimization cannot retain hidden device memory. Fused-call,
  creation, cache-hit, and error counters are required by both fixed and
  dynamic runtime audits for Multi, Tensor, and Cut.
- The GPU cost model mirrors that dependency exactly: a Cut tensor contributes
  two independent LOAD/PREPARE tiles followed by one fused compute only after
  both tiles are ready. It must not model two independently executable
  half-GEMVs, which would invent intra-tensor compute/prepare overlap and
  over-reward Cut. CPU records `independent_halves`; GPU records
  `fused_pair`. Formal profiles must explicitly encode both this dependency
  and the selected Multi implementation; missing fields are rejected rather
  than silently replaced with defaults. Final consolidation rejects a
  backend/profile mismatch. Any decode that cannot retain both halves and
  falls back to sequential Cut compute invalidates the fixed or dynamic GPU
  row, because its execution dependency no longer matches the calibrated
  fused-pair profile.
- LOAD, layout preparation, and compute use the real working-unit pipeline.
  Pipeline budget violations, external inference overlap, token mismatch,
  incomplete trace replay, or failed worker affinity invalidate a run.
- In a granularity run, placement reconciliation alone controls one-time
  persistent promotions (materialized lazily by backend ensure). Recurring
  DISK LOAD/PREPARE is owned exclusively by the backend's physical-unit
  pipeline. Complete-weight plan-stage anchors are
  deferred and filtered out; otherwise every Online/Diff plan apply would
  eagerly stage whole tensors before staging the same Cut/Tensor/Multi units,
  both double-counting switch work and collapsing Cut's finer overlap. Each
  row must report `stage_defer=1` for every full plan apply and zero fired
  plan LOAD/TRANSFER/PREPARE anchors.
- Formal dynamic runs disable the model-load replay reset, reset B(t) exactly
  once at the first generated-token boundary, and forbid later Online/Diff
  plan transitions from rewinding the trace. The device log records that
  decode-boundary reset; a timed row is invalid unless it occurs once at
  generated token zero. Its token CSV must also have monotonic decode
  timestamps spanning the requested 600-second interval within one measured
  decode-latency tolerance, so a nominal duration exit cannot hide a replay
  that started late or stopped early.
- The watcher replay epoch is an atomic steady-clock timestamp. A background
  sample publishes its interpolated budget only if the epoch is unchanged,
  preventing both a C++ data race and a stale pre-reset budget from
  overwriting B(0) after the first-decode boundary.
- Measurement isolation identifies the actual process executable rather than
  counting shell-command strings. The sole `llama-cli` executable appearing
  after an empty clean-start gate is the measured process; any concurrent
  executable is rejected even when it uses the same binary and model paths.
  This avoids both treating adb's `sh -c` wrapper as a second inference and
  accidentally exempting a competing identical experiment. The check fails
  closed unless it obtains at least one sample and observes the measured
  `llama-cli`; short fixed-budget runs are sampled at most one second after
  launch rather than inheriting the slower cooldown polling interval. Any
  failed monitor poll also invalidates the row and prevents resume,
  calibration, or final consolidation from accepting it. Measurement-only
  ADB probes use a five-second timeout, and the runner requires their monitor
  thread to exit before advancing, preventing a hung probe from overlapping
  the next benchmark.
- CPU Multi and Multi-fused use identical logical unit boundaries. The fused
  implementation reuses the persistent CPU1 helper as a second PREPARE lane;
  it does not create a thread or allocate a staging buffer per unit.
- The fixed CPU sweep measures both Multi implementations. One implementation
  is selected globally using paired fixed-budget measurements; it is never
  selected independently at each budget. The initial 30% fusion smoke gates
  only runtime/token/pin contracts; it does not require either implementation
  to win a noisy one-round latency comparison.
- The selected coarse implementation is recorded in every dynamic row. If the
  fixed sweep selects `multi_fused`, any mixed frontier that contains Multi
  must produce and execute real fused MUL_MAT pairs with zero fused-kernel
  errors; otherwise the row is invalid. Final consolidation requires all
  seven methods and the phase-only ablation to carry the same coarse
  implementation.
- Although the requested fixed sweep uses one measured round, its mode order
  rotates across the seven budget points (with the same thermal/start gates)
  so no granularity is systematically always first or last.
- The 16 MiB streaming reserve is derived from the measured 15.75 MiB
  maximum in-flight unit and is charged inside the unchanged hard budget.
  Residency policy is selected on a held-out four-node pilot; any
  plan-protection relaxation invalidates a candidate. The selector also
  requires all candidates to share one policy-excluded runtime-core hash,
  verified binary/model/metadata/cost/profile/table/trace lineage, identical
  two-tile physical representation and WBM total, and a common generated-token
  prefix. Their full runtime hashes must be distinct, proving the intended
  residency policy is the screened difference. It then selects the minimum
  measured `ms/token`; no expected winner or ordering is encoded. Both the
  selected policy and the common measured-input reserve are persisted in
  `selection.json`; the formal 600-second run reads both values rather than
  reintroducing an untracked reserve constant.
  Mixed-boundary cost is calibrated on a separate four-node sequence before
  the original ten-minute trace is run.
- Mixed-boundary repeatability groups use a SHA-256 of the ordered
  `(granularity kind, weight ids)` frontier. Matching aggregate Multi/Tensor/Cut
  counts are not treated as an identical plan, because they may select
  different weights and have different pipeline behavior.
- Boundary calibration does not hard-code “Cut low/Multi high.” The fixed
  sweep records the actually fastest mode at every ratio after globally
  selecting the Multi implementation. The 3712/5504 MiB boundary endpoints
  inherit the measured 60%/90% winners and compare coverage over flexible
  logical weights rather than biased raw unit counts or mandatory Tensor-only
  weights. If those fixed endpoints have the same winner, the chain stops
  instead of manufacturing a crossover.
- Baselines are not delayed, throttled, or otherwise weakened to manufacture
  an ordering.
- Fixed and dynamic runners acquire the same nonblocking host lock keyed by
  OP12 serial before touching the device. This closes the race in which two
  processes both pass an empty clean-start gate before either launches;
  runtime process monitoring remains the independent device-side check.
- Because the dynamic runs intentionally reuse the 4.6 GiB GGUF already on
  OP12, each CPU/GPU runner hashes the remote model and `llama-cli` before
  accepting or resuming any row. A same-name stale model or binary aborts the
  matrix instead of producing incomparable data. The pushed model-metadata
  JSON and every placement-cost file are also checked byte-for-byte against
  their host inputs; final consolidation requires all seven methods to share
  the same verified hashes.
- The fixed 30%--90% sweeps enforce the same lineage contract. Their resume
  logic skips a cell only when its recorded model and binary hashes match the
  current remote inputs and its fixed-run configuration hash matches every
  execution, budget, granularity, pipeline, affinity, and validity-gate
  setting plus the host runner content. Calibration and final fixed-budget
  summaries reject mixed or missing binary/model/config lineage.
- CPU dynamic runs require a minimum observed scaling limit of 1.4 GHz, mean
  active-core frequency of 1.55 GHz, and median per-sample active-core floor
  of 1.5 GHz. These thresholds reject clear throttling while retaining the
  normal OP12 fixed-performance operating range measured in prior valid runs.
  The check fails closed: if any requested cpufreq signal is unavailable, the
  row is invalid rather than silently accepted.
- Dynamic end-to-end execution latency uses an independent critical-path
  wall timer that starts before the runtime scheduler/provider/plan-apply
  hooks and ends at the backend synchronization that makes each token's
  logits consumable. The older component sum (`eval + provider wait + apply`)
  is retained only as a containment cross-check. A row is invalid unless the
  wall-timer run count equals the standard eval run count and its interval
  contains all three measured components. Full asynchronous remote-solver
  wall time is never added separately because its nonblocking portion may
  overlap compute. Single-token latency in the token CSV uses this same
  critical-path boundary, so fixed-budget calibration, trace-coverage
  tolerance, and final ranking do not silently use different timers.
- I/O, LOAD, layout preparation, CPU compute, pipeline wait/residency, and
  eviction breakdowns use cumulative-backend counter deltas. The origin is
  captured only after the prompt graph has synchronized and immediately
  before B(t) is reset at generated token zero; the endpoint is captured after
  final decode synchronization. At both boundaries, a measurement-only
  quiescence gate waits for the persistent host LOAD/PREPARE queues to become
  idle and then drains backend commands they enqueued. The gate is never
  called per graph or token, so it cannot remove measured pipeline overlap;
  it prevents prompt work and an unfinished final cross-graph prefetch from
  racing the cumulative snapshots. A formal row is invalid unless this
  decode-only phase window occurs exactly once and its run count equals both
  the eval and independent decode-wall counts. GPU compute remains N/A in
  production runs because enabling OpenCL event profiling would perturb queue
  retirement and the pipeline overlap being measured.
- The granularity cost model uses the full decode-only LOAD service as its
  first pipeline stage, followed by PREPARE and COMPUTE. LOAD includes direct
  reads plus host staging/pool service. GPU foreground direct reads are also
  charged to LOAD. GPU SOA PREPARE is measured around the real asynchronous
  materialization callback (buffer/pool management, backend write, layout
  conversion/transposition, and necessary queue/event work) with its
  HOST_SRC dependency interval subtracted, so waiting for an async LOAD or
  falling back to a direct read is not counted twice. Pure direct-read
  bytes/time are retained as a diagnostic subcomponent and are not
  substituted for LOAD when fitting the planner. LOAD/PREPARE launch cost is
  fitted against decode-only
  non-resident logical working units, while COMPUTE launch cost is fitted
  against all logical working units; physical tensor/cut calls remain
  breakdown diagnostics and cannot silently redefine the planner's unit.
- Android fixed-performance mode is explicitly reset off, enabled only around
  one measured method, and restored off in a `finally` cleanup. Restore
  failure invalidates the row, and the matrix runner also performs a
  last-resort cleanup before exit, so one method cannot leak its power-mode
  state into the next cooldown or another user's experiment.
- Online, Diff-before, and Diff-now each receive a fresh remote-solver process
  and a method-specific solver log. The runner requires at least one cold
  solve, one exact run id, and the canonical `online-global-mixed` or
  `diff-tree-mixed` ExecPlan working-unit policy. This prevents an earlier
  method's Python/JSON/plan caches from warming a later method. The process is
  started only after the phone passes its clean-start gate, so a long thermal
  or battery wait cannot leave a supposedly fresh solver stale before use.
  The server process default is also selected from the current measured
  method, rather than from the full matrix method list; the per-request policy
  remains an independent method-specific contract.
- Diff-tree solver events record the untruncated edit count. The runner
  rejects a row if any cold solve omits this count or exceeds the configured
  local split/merge edit bound; the final audit reports the observed maximum.
- Within one callback, Diff-tree caches the immutable LOAD/PREPARE/COMPUTE
  cost of every distinct decision across all IFF-tree prefixes. This changes
  neither the selected frontier nor the objective: a host preflight over the
  existing OP12 3712/5504 MiB frontiers reduced the 128-edge planning path
  from roughly 170--260 ms to 79--95 ms with identical predicted latency,
  edit count, and target distance. These host timings are implementation
  diagnostics only; formal latency still comes from OP12.
- A local edit prefix is installed atomically at one graph boundary, so its
  cost model charges the fixed switch overhead once and each uniquely changed
  weight once. Intermediate Cut-to-Tensor-to-Multi tree edges are not
  double-charged.
- Runtime frontier publication compares and updates only changed weight
  entries under one lock. Identical frontiers keep the previous generation;
  changed frontiers report the exact number of changed logical weights in the
  device log and final audit. This replaces the former
  clear-plus-one-lock-per-tile path and avoids exposing a transient empty
  frontier. When the placement/dispatch/event core is unchanged, a
  frontier-only fast path also skips weight reconciliation and graph
  invalidation. Unit IDs are derived from weight IDs rather than frontier
  position, so a local edit does not renumber untouched suffixes; host C++ and
  planner tests verify this delta contract.
- The granularity switch coefficient is not a hand-tuned constant and is not
  fitted on the held-out ten-minute trace. Each held-in mixed-boundary
  candidate records `frontier_publish_ms_total/max`: wall time spent building,
  validating, and atomically publishing only its working-unit frontier after
  scheduler quiescence. The first publication is an initial installation, not
  a split/merge. The parser identifies real transitions from consecutive exact
  `frontier_sig` changes, and the selected candidate's maximum
  changed-frontier publication time is installed as the fixed per-change
  switch coefficient in both the residual and phase-only profiles. All formal
  mixed methods must record the same positive coefficient and source string.
  Final consolidation also requires the count of changed-signature
  publications to equal the exact frontier state-change count. A change that
  preserves the same Multi/Tensor/Cut unit counts but selects different
  weights is therefore still measured, while a repeated identical frontier is
  not charged. Transition changed-weight totals likewise exclude the initial
  full-frontier installation; all-publication totals remain in the JSON audit
  as a separate startup-inclusive diagnostic.
- Diff-before versus Diff-now attribution uses a transition-path hash formed
  after collapsing consecutive duplicate exact frontier signatures. Different
  callback/reapply counts cannot masquerade as different split/merge
  decisions; final consolidation requires the two ablations to execute
  genuinely different ordered frontier paths.
- Frontier publication deliberately excludes residency reconciliation,
  backend placement apply, and solver time. Those costs are already measured
  on the end-to-end critical path and charging them again as a granularity
  switch would double-count work common to all placement methods. The final
  audit therefore reports predicted frontier switch cost, measured frontier
  publication time, complete `apply_exec_plan` time, blocking provider time,
  remote-wall time, and server-solve time separately. This prevents either a
  small modeled `switch_ms` or a large complete-plan `apply_ms` from being
  misrepresented as the isolated cost of split/merge.
- A rejected runtime plan is never recorded as applied: a nonzero
  `apply_exec_plan` result retains the previous safe frontier and emits a
  fatal audit marker that invalidates the device row.
- Every row records the granularity-profile path, content SHA-256, profile
  kind, configured policy, effective per-method policy, and placement source.
  Effective policies are explicitly audited as Offline=`offline`,
  Online=`online`, and both Diff variants=`diff-tree`. Final consolidation requires
  Offline, Online, and Diff-now to share one `pipeline-residual` profile,
  while Diff-before must use a distinct `phase-only` profile. Offline derives
  placement from its precomputed table. Online, Diff-before, and Diff-now use
  the same stateful CP placement solver; the device state and transition
  objective are therefore real parts of the dynamic-planner comparison
  instead of silently reducing Online to Offline placement. All four still
  use the per-budget offline mixed frontier as the common steady-state
  granularity target: Online may reach it globally and Diff may approach it
  through bounded IFF edits. The complete target table is content-addressed:
  Offline, Online, and Diff-now must share the exact same table SHA-256.
  Diff-before records its independently generated phase-only ablation table
  rather than pretending it is byte-identical.
  Before inference, every remote table file is checked against the local
  relative path, byte count, and SHA-256. Static-Min/Static-Max likewise
  record and validate their fixed-Tensor table. Thus the
  Diff-before/Diff-now attribution and each device input are proved from
  profile/table content rather than inferred from artifact names.
- Each fresh remote solver event records its effective placement source.
  Formal Online/Diff rows are rejected unless every cold solve reports
  `stateful-runtime-cp`; Offline is independently required to report the
  effective `offline-table` source. Controlled fixed-budget and boundary
  experiments continue to hold placement constant, because those experiments
  isolate granularity rather than compare complete dynamic planners.
- The runtime emits an exact working-unit signature covering unit IDs, fusion
  flags, weight/tile identities, row/byte ranges, and physical tiling. The
  runner hashes the ordered signature sequence and separately hashes the
  human-readable `mixed/Multi/Tensor/Cut/tile` counts. Final consolidation
  rejects a Diff-before/Diff-now pair whose exact split/merge decision traces
  are identical: changing only predicted-cost annotations is not evidence
  that the new granularity cost model affected execution.
- A mixed-method row must apply at least one frontier that simultaneously
  contains two or more of Multi/Tensor/Cut. Merely observing different
  homogeneous modes at different trace points is not accepted as evidence of
  mixed-granularity planning.
- Mixed-boundary calibration is a consumer-side audited experiment, not just
  a search over rows marked `ok`: every candidate and its fixed-Tensor
  reference must share the binary, model, metadata, placement costs, runtime
  configuration, source/runtime trace, token prefix, and exact physical WBM
  representation, while passing timed-run, backend-compute, pipeline-budget,
  pin, frontier, and isolation contracts.
- Final consolidation preserves any runner-level invalid status when it
  reparses a raw log. Its audit records 600-second duration exit, measurement
  isolation, start/end thermals and battery, CPU-frequency measurements and
  thresholds, budget violations, plan-protection relaxations, and maximum
  Multi/Tensor/Cut frontier populations for every method. It also rejects the
  seven-method matrix unless every row reports the same physical cut-capable
  tensor count, two-tile part count, and WBM total. CPU/GPU wrappers pass the
  explicit expected total `4436.7890625 MiB`; each formal row and the held-out
  residency pilot must match it within log precision. This catches a shared
  omission that cross-method equality alone would miss.
- A matrix runner itself exits nonzero unless every requested trace/method
  has exactly one valid row. It does not defer an invalid thermal, token,
  isolation, or runtime row until final consolidation; the persistent
  autochain immediately restarts and resumes only the failed work.
- Dynamic rows also carry one shared runtime-configuration SHA over planner,
  pipeline, budget, affinity, backend, and validity settings plus the host
  runner content. Resume rejects a stale hash, and final consolidation
  requires all seven methods—including the separate Diff-before artifact—to
  share it. Profile bytes are excluded from this hash because their deliberate
  Diff-before/Diff-now difference is audited by the independent profile/table
  lineage contract above. A second runtime-core SHA omits only the residency
  policy; it is equal in the formal matrix and is used by the held-out pilot
  to prove every other runtime setting, including the 16 MiB reserve, is
  unchanged while that one policy varies.
- If a resumed rerun replaces the sole divergent token trace and restores a
  common deterministic prefix, the runner clears only the derived
  `token_mismatch` status. It never clears unrelated thermal, budget,
  isolation, affinity, or runtime failures.

## Result lineage

The host-only mixed-frontier sanity check is recorded in
`OP12_V21_HOST_PLANNER_STRUCTURAL_AUDIT.md`. It demonstrates planner structure
and caught validation defects, but it is explicitly not device latency
evidence.

The partial GPU v19 fixed sweep is diagnostic-only and carries its own
`INVALID.md`: its requested `output` pin was ignored, so repeated allocation
and reload of the 410.98 MiB output tensor contaminated all rows. v21 starts
from fresh output directories and cannot resume those rows.
