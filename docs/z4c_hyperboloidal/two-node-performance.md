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

The native controls passed in two-node job 9051913 (terminal F/0). All three
final 24,704,784-scalar padded 31+8 payloads, complete parameter maps and ordinary
history/horizon outputs agree bitwise. The zero-step capture also agrees
bitwise with the source. All 280 observed stages have the exact prescribed
clocks and pass compatibility checks. Independent review confirms these results.
See `evidence/native-two-node-timing-validation.json` and
`evidence/native-two-node-timing-independent-review.json`.

| Native case | Three 20-step windows (s/complete RK4 step) | Median |
|---|---|---:|
| Timed normal | 4.572643, 4.847819, 4.572417 | 4.572643 |
| Timed observer | 5.480417, 5.422913, 5.391933 | 5.422913 |

Observed median overhead is 0.850270 seconds/step (18.59%). Normal windows have
a 6.02% spread, so smaller candidate improvements will need more repeated
measurements. Startup/final diagnostics and I/O are separately 4.207/6.200
seconds for timed-normal and 3.603/6.315 seconds for timed-observer. These are
native instrumentation controls; no matched legacy comparison or solver
optimization has yet passed. Max-rank windows do not identify which rank was
slowest because this timer does not serialize local rank times.

A report-only extension now gathers those already-recorded local samples after
all measured intervals and finalization timing have ended. It writes every rank's
duration and the IDs attaining each existing MPI_MAX result. The existing two-rank,
eight-step CPU control passes complete-state/output equivalence and the original
negative controls; see `evidence/native-window-rank-export-cpu.json`. It has not
been built or qualified on the GPU yet and does not change the timings above.
The rank attaining the measured maximum is not by itself a causal critical-path
diagnosis.

The user's subsequent priority clarification makes a measured bottleneck and a
correctness-qualified complete-step speedup the next milestone. Finish and
validate the active native timing controls, then run the already prepared native
profile at the next serialized two-node allocation. Legacy diagnostic/reference
completion proceeds independently and does not block that profile or native
candidate screening. The approximately 4.57-second observation has now passed
the native controls above. Use measured kernel durations,
launch sizes and waits to select one change: surface case batching with separate
elimination, field-parallel KO using existing RHS storage, or a specifically
identified synchronization fix. Workgroup tuning stays separate. Do not add
another measurement framework without a concrete question existing tools cannot
answer. Final parity still requires the matched fastest validated predecessor.

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
The profile below confirms the reduction launch shape; achieved occupancy and
scratch traffic remain unmeasured. An absent spill field does not establish zero
spilling. Native bulk private storage is slightly smaller than v8's, so the
larger source kernel alone does not establish register pressure as the cause
of its slowdown. Surface batching is a profiling lead, not an accepted fix.

The bounded profile in job 9051936 completed both solver cases with exit 0.
The batch ended F/2 because its audit misclassified `/etc/ld.so.cache` as a loaded
library. A second output-location assumption required correction too. Offline
reanalysis preserves that original failure and verifies exact complete 31+8
payload and ordinary diagnostic equality. No solver rerun was needed.
See `evidence/native-n12-selected-profile-findings.json`; raw traces remain remote.

| Representative rank | First-to-last bulk RHS span | GPU interval union | Activity coverage |
|---|---:|---:|---:|
| 8 (many bulk points) | 17.287 s | 1.272 s | 7.36% |
| 23 (outer surface owner) | 17.183 s | 2.047 s | 11.91% |

These spans are not complete-step timing windows or globally aligned clocks.
Activity coverage is not achieved occupancy. On rank 23, normal compatibility
launches one group of 256 work-items and takes approximately 19 ms/call; surface
RHS also launches one group of 256 and takes approximately 13 ms/call. Bulk launches
use 512 or 1024 work-items/group. Most elapsed time on both sampled ranks lies
outside recorded GPU work and SYCL/Level Zero calls, often between transfers.
MPI calls were not traced, so this does not yet identify the responsible MPI
operation or prove that either sampled rank determines global elapsed time.

The all-rank APS follow-up now passes scientific and actual-interception checks.
Job 9051976 stopped in the external process monitor at startup; that failure is
preserved. Retry 9051996 completed both solver cases with exact full state and
diagnostic agreement. Its F/2 batch status records an output-format audit failure;
offline reanalysis passes and preserves the original report. See
`evidence/native-n12-aps-findings.json`.

Across the 24 ranks, mean whole-process MPI residence is 22.25 s out of about 28.70 s.
`MPI_Testall` accounts for 316.38 summed rank-seconds (13.18 s/rank) and
1,431,038,674 calls. `MPI_Irecv` averages 1.59 s/rank, reaching 7.22 s on rank 19;
`MPI_Isend` averages 0.089 s/rank. `MPI_Allreduce` averages 5.70 s/rank, but its
1,133 calls/rank include setup and final diagnostics/FastFlow. Its 16-byte bin
contains five calls/rank, consistent with initial plus four post-cycle timestep
reductions. These are whole-process observations, not a sum of phase maxima or
a demonstrated causal critical path. Disabled imbalance reporting means zero is
unmeasured, not proof of balance.

The selected single candidate replaces `MappedExchange`'s two `MPI_Testall`
completion scans with individual `MPI_Test` calls, following native Bvals.
Every request is polled; unpacking and buffer reuse still require all requests
to complete. Earlier retirement of completed requests is the hypothesis, not
an established explanation or speedup. Only one runtime `.cpp` file changes;
task order, arithmetic, storage layout and ownership remain unchanged. Existing
local MPI/poison tests and a delayed-second-peer check pass. Next is unprofiled
control/candidate/candidate/control screening with complete-state comparisons.
Surface batching and field-parallel KO remain untested candidates.

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
| Native timing and observer controls | Complete payload/diagnostics bitwise; independent review passes | 4.572643 s normal; 5.422913 s observed | Qualified measurement; no solver speedup |
| Selected-rank native profile | Complete payload/diagnostics bitwise; offline audit correction preserves original F/2 | GPU activity covers 7.36%/11.91% of sampled RHS spans | Diagnose transfer-adjacent host gaps next; no solver change |
| All-rank APS MPI profile | Complete payload/diagnostics/runtime identity pass; original F/2 retained | Whole-process mean 13.18 s/rank in Testall | Select one completion-policy experiment; no throughput claim |
| Individual MPI completion checks | Local MPI/poison/mixed-peer tests pass; GPU pending | Pending | Uncommitted candidate; retain only after complete-step improvement |
| Matched previous-implementation reference | Builds ready; GPU qualification pending | Pending | Does not block native screening |

Completion requires the committed native implementation to match or beat the
fastest accepted previous N12 reference median on the same two nodes, with repeat
variation small enough to support that conclusion, and all applicable architecture
and scientific requirements verified. A partial speedup is progress, not completion.
