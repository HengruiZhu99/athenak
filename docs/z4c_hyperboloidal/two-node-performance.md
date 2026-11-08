# Native N12 two-node performance parity

The active objective is performance parity with the fastest validated applicable
previous implementation on two Aurora nodes / 24 GPU-tile ranks. The request in
`/Users/hz0693/.codex/attachments/290805e0-7e5d-427b-9ea2-932e424784a8/pasted-text-1.txt`
supersedes the earlier full-fresh-3551-before-timing restriction. No further long
N12 evolution or eight-node campaign is part of this goal. The earlier evidence
and scientific gates remain intact; `requirements.json` describes that previous
architecture/long-trajectory effort, while `two-node-performance-requirements.json`
tracks this objective.

## Starting evidence

The native trajectory is admitted through cycle 2486, t/M=3.501098714602395.
Complete fields, native chi records, 9,944 stage records, eight diagnostic times
and independent review pass. See `evidence/native-n12-common-07-science.json` and
`evidence/native-n12-common-07-independent-review.json`. The ordinary checkpoint
SHA256 is `dd4b86ab42c10827f14bd339930cb4d2ffaf9fb5e2b43953d1e5d7924a9fc1ec`.
The original native executable remains
`f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c`.

Observed native validation steps take about 5.2--5.4 seconds on 24 ranks. The
historical N12 run takes about 0.326 seconds on 48 ranks. These are unmatched
observations, not the acceptance comparison. Existing N24 timings likewise
cannot establish the fastest N12 reference.

Retained previous implementations v3, v6 and v8 have verified binaries and
numerical evidence. v3/v6 have favorable single N24 screening measurements;
v8 has repeated and poisoned-workspace qualification. Screen their applicable
N12 versions under the same protocol before selecting the reference. Retain an
immutable previous-implementation control to qualify the measurement wrapper
and application of those optimizations to N12.

## Initial bounded experiment

- Use the admitted native cycle-2486 checkpoint unchanged. Construct a private
  legacy restart from its complete canonical fields and eight chi-history values,
  preserving/checking the historical topology and physical parameters. Verify
  both initial states, rather than assuming equivalent restart files.
- Use 10 warmup steps and three windows of 20 complete RK4 steps, ending at cycle
  2556. Both paths retain dt=0.00046944203735617445, float64, KO=.1, chi floor=1e-10,
  constant shift-speed policy and the same grid/physical equations.
- Compare original native, timed native and timed native with the stage observer.
  Reuse the qualified timing object and compose the observer main only after
  verifying source/object/header compatibility. Compare complete final 31+8
  state, meaningful restart metadata and ordinary diagnostic outputs.
- Time maximum-rank elapsed full cycles with detailed profiling disabled. Keep
  startup, final diagnostics/output and warmup separate. Preserve raw windows.
  Verify no scheduled output falls inside the bounded timed interval.
- Match the legacy controls and contenders to the same initial physical state,
  cycle count, compiler/FP policy, binding and timing boundaries. Increase samples
  if repeat variation prevents a clear comparison.
- Prefer a common allocation for direct comparisons where practical. A native
  observer-overhead diagnostic can proceed independently if the legacy wrapper
  is not ready; it cannot establish legacy/native parity by itself.

No GPU benchmark has yet passed this protocol. Existing measurements must not be
relabeled as a matched baseline.

## Investigation and acceptance

Static PVC binary metadata is preserved in
`evidence/two-node-kernel-selected-resources.json`, with verified raw-artifact
locations/hashes in `evidence/two-node-kernel-transfer-verification.json`.
It compares the unchanged native binary with archived v8; it is not a runtime
profile and does not select v8 as the N12 reference.

| Compiled kernel | Private allocation per work-item (bytes) | Reported spill bytes | Spill/fill allocation (bytes) |
|---|---:|---:|---:|
| Native bulk CalcRHS | 22,928 | absent | absent |
| v8 bulk PointKernel | 24,480 | absent | absent |
| Native normal solve, nd_item variant | 34,024 | 8,960 | 16,384 |
| v8 normal case evaluation | 25,216 | absent | absent |
| Native surface RHS, nd_item variant | 26,440 | 2,944 | 4,096 |
| v8 surface RHS | 26,440 | absent | absent |

The native normal solve's single_task variant also reports spill storage.
Actual reduction variants dispatched, achieved occupancy, scratch traffic and
elapsed time remain unmeasured. An absent spill field does not establish zero
spilling. Native bulk private storage is slightly smaller than v8's, so the
larger source kernel alone does not establish register pressure as the cause
of its slowdown. Surface batching is a profiling lead, not an accepted fix.

Inspect bulk kernel organization, geometry reuse, surface arithmetic,
synchronization, communication/data layout and imbalance one at a time. Reorder
only using measured evidence. Preserve the native Driver, task lists, block
ownership, tree, equal weights, restart/output and SMR/AMR infrastructure.
Follow Cartesian Z4c's useful kernel and boundary patterns. No replacement
driver, scheduler or globally replicated evolution state is permitted.

Numerical controls use existing gates, chosen before observing candidate results.
Require bitwise equivalence where arithmetic order is preserved; otherwise report
all differences against the existing justified tolerances. Validate complete
states/records, compatibility, constraints, horizons and restart continuity as
applicable. Changing dependencies/storage requires poison checks; changes affecting
refinement require relevant SMR/AMR checks. Preserve all required validity checks.

Commit an optimization separately only after repeated complete-step improvement
and scientific acceptance. Measurement infrastructure and carried-over evidence
receive separately labeled commits. Revert or reject slower/inconclusive candidates
while retaining their evidence. Do not count RK stages as complete updates.

All allocations are serialized and limited to two nodes. Check the live queue
before submission; leave the three older held jobs and production untouched.
Build with `make -j64`. Keep large binaries/checkpoints remotely where practical
and check local free space before transfers. The previously staged surface-SMR
24/96 campaign is not automatically authorized by this new plan; use only relevant
bounded two-node correctness checks when a proposed change requires them.

## Experiment ledger

| Experiment | Numerical result | Complete-step timing | Decision |
|---|---|---|---|
| Matched native/observer/previous baseline | Pending | Pending | No performance claim |

Completion requires the committed native implementation to match or beat the
fastest accepted previous N12 reference median on the same two nodes, with repeat
variation small enough to support that conclusion, and all applicable architecture
and scientific requirements verified. A partial speedup is progress, not completion.
