# Native N12 two-node performance parity

The original objective was performance parity with the fastest validated applicable
previous implementation on two Aurora nodes / 24 GPU-tile ranks. The request in
`/Users/hz0693/.codex/attachments/290805e0-7e5d-427b-9ea2-932e424784a8/pasted-text-1.txt`
supersedes the earlier full-fresh-3551-before-timing restriction. No further long
N12 evolution or eight-node campaign is part of this goal. The earlier evidence
and scientific gates remain intact; `requirements.json` describes that previous
architecture/long-trajectory effort, while `two-node-performance-requirements.json`
tracks this objective.

The user cleared that goal after the radial experiment and subsequently requested
a bounded comparison of boundary MeshBlock weights2 and3 against equal weights,
subsequently permitting a higher weight where justified. Weight4 was also tested.
That request explicitly relaxes the former equal-weight constraint for these
isolated tests; it does not resume the broader parity campaign.

## Current retained implementation

The user resumed on 2026-11-08, requesting full KO qualification first, then
one validated legacy optimization at a time. Correctness and repeated
complete-step improvement must pass before each optimization is committed and
pushed. Production remains untouched.

The retained MPI GPU cache setting reduced complete-step time by 7.89x in its
controlled comparison. Field-parallel bulk KO is now also qualified and retained:
its six-window median is **0.549904 s**, versus **0.600897 s** for its unchanged
native control, an **8.49% reduction**. Every candidate window beats every control
window. The exact screened source passed the additional two-node qualification
in job9052266, including independent observer, poison, AMR and restart reviews.
See `evidence/native-field-ko-admission.json`.

The KO kernel uses existing RHS storage and preserves the per-field operator,
transport/KO/Lambda-to-D addition order, subset halo dependencies and validity
checks. Driver, TaskLists, MeshBlocks, tree, equal weights, local surfaces,
restart/output and refinement machinery remain unchanged. No new persistent
scratch allocation or workgroup tuning is part of this change.

The main bulk RHS SYCL workgroup size of128 is now also qualified and retained.
Its repeated unprofiled comparison gives **0.551614 → 0.380092 s/step**, a
**31.09% reduction** over accepted KO; all six candidate windows beat all six
controls. Full qualification in job9052329 passes complete-state/chi, observer,
poison, refinement, restart and diagnostic checks. Only this launch policy
changes; arithmetic and task dependencies remain intact. See
`evidence/native-bulk-group128-admission.json`.

Field-parallel endpoint radial derivatives are now also qualified and retained:
**0.379937 → 0.300340 s/step**, a **20.95% reduction**, with all six candidate
windows faster than all six controls. The already-collected job9052427 passed
independent observer/restart, poison, AMR and private-cache SMR reviews. The
cache fixture explicitly checks NaN overwrite, stale finite values and refresh
after plan rebuilding. See `evidence/native-radial-fields-admission.json`.

Parity remains unfinished. Leading legacy v3/v6 references measured about
0.1364 s/step in their earlier matched allocation; a final same-allocation
comparison would still be required. The post-radial profile records rank23 bulk
RHS0.575609s, normal compatibility0.302462s and surface RHS0.200214s across four
steps; radial preparation is only0.004238s. The surface-dependent bulk batch fell
from0.542856s to0.218005s. These are representative-rank GPU durations, not an
additive global elapsed-time budget or demonstrated critical path.

The user-requested weight experiment changes only boundary block cost metadata
in private restart copies, using the admitted radial numerical objects and the
native load balancer. Geometry, normal correction, surface Solve/Evaluate and
workgroup policies remain fixed. No production checkpoint or source is modified.

## Independent normal compatibility cases

The user subsequently authorized systematic solver/cache experiments, one factor
at a time, committing only correctness-qualified complete-step improvements.
The first is now accepted: four independent normal-compatibility evaluations run
in a separate batched kernel, followed by the unchanged3x3 elimination. The
case kernel retains the measured256-work-item group size. A private per-plan
buffer stores defects and validity flags, about18KiB per144-point surface face.
Diagnostic plans remain isolated from module RHS storage; native rebuilding
resizes the buffer. No geometry reuse or workgroup tuning is combined here.

Job9052593 (F/0) gives six-window medians **0.299643 -> 0.241150 s/complete RK4
step**, a **19.52% time reduction**. All six candidate windows beat all six
control windows; all four complete31+8 endpoints and diagnostics are bitwise
exact. The same frozen candidate passed six qualification cases in9052603
(F/0): observer280 stages, poisoned state/history and first RHS, coherent AMR
snapshots/refine/coarsen, prefix2/resume8, and the refined cache fixture. The
958-file collected record verifies876 scientific files bitwise against the
admitted radial reference, plus complete checkpoint/parameter comparisons.
Existing restart diagnostic gates and unused-header exclusions remain unchanged.
Local eight-rank surface/zero-face/NaN-output checks and real history-output
isolation also pass. See `evidence/native-normal-cases-admission.json`.

Next: separately test stage-local surface geometry reuse, then the remaining
measured cache/reuse candidates. The user-authorized task continues; the former
formal parity goal remains cleared. This result does not establish parity with
the earlier~0.1364s legacy reference or scaling beyond two nodes.

## Boundary weight experiment

The clean two-node/24-rank allocation9052547 completed F/0 in10:02. Eight
unprofiled continuations used the order1A,2A,3A,4A,4B,3B,2B,1B. Each starts
from the same cycle2486 physical state, warms up10 complete RK4 steps, then
measures three20-step windows using maximum-rank elapsed time. Startup,
final diagnostics/output and the three stage-observer runs are excluded.

| Outer-block weight | Six-window median, s/step | Time reduction vs1 | Repeat A/B medians |
|---:|---:|---:|---:|
| 1 | 0.301176 | baseline | 0.301048 / 0.301298 |
| 2 | 0.299043 | 0.71% | 0.299116 / 0.299035 |
| 3 | 0.299823 | 0.45% | 0.299781 / 0.299939 |
| 4 | 0.298797 | 0.79% | 0.298791 / 0.298803 |

Every weighted window is faster than every equal-weight control window. The
improvements repeat in both halves of the sequence, but remain small. Weight4
is only0.082% faster than weight2, comparable to the control's0.083% A-to-B drift;
this does not establish a general optimum. No permanent initialization/AMR
weighting policy is introduced by this fixed-topology restart experiment.

The actual startup ownership matches the native contiguous, reverse-GID greedy
load balancer's predicted maps. Six ranks still own one outer block each, with
144 surface points per owner; the global work remains106 blocks,229392 bulk
points and864 surface points. Changing weights moves bulk work between ranks,
without dividing or accelerating an individual surface block.

| Weight | Outer-block owner ranks | Their block counts, in owner order | Their total bulk points | Largest bulk count on an ordinary rank |
|---:|---|---|---:|---:|
| 1 | 11,14,16,19,21,23 | 4,5,5,5,5,5 | 65772 | 12663 |
| 2 | 11,13,16,18,21,23 | 5,4,5,4,4,4 | 60588 | 12663 |
| 3 | 10,13,15,18,21,23 | 3,5,3,3,5,3 | 50544 | 15525 |
| 4 | 9,12,15,18,21,23 | 2,2,2,2,4,3 | 41580 | 17010 |

Weight4 removes36.78% of aggregate bulk points from surface-owning ranks, while
increasing the largest ordinary-rank bulk count by34.33%. Native greedy
partitioning and indivisible blocks make this response nonmonotonic. The offline
scan of weights5,6,8,10 gave less favorable boundary-owner layouts than4, so
these were not allocated GPU tests. Bulk point counts quantify ownership, not
measured kernel cost. The per-rank complete-window clocks include synchronization
and waits; their0.885--8.448 microsecond spread cannot identify the causal critical
rank. See the complete96-row comparison in
[`evidence/native-boundary-weight-workload-by-rank.csv`](evidence/native-boundary-weight-workload-by-rank.csv)
and the independent
[`timing review`](evidence/native-boundary-weight-timing-review.json).

Redistribution alone gives less than1% here. Combined with the retained
representative-rank profile, this favors testing surface-kernel organization
next if optimization is resumed: batch the four normal-compatibility cases,
then perform their small elimination in a separate kernel, holding workgroup
tuning fixed. No such experiment is part of this bounded request.

The original attempt9052526 stopped at a test-only timer guard requiring unit
costs, before weight2 evolution. Its failure and all partial artifacts are
preserved. The retry changes only two timing/observer Driver objects to verify
the explicit boundary weight; all numerical objects and timing boundaries are
unchanged. All14 retry solver cases and original before/after guards passed.
Full collection contains222 hash-verified files. Independent reviews confirm
bitwise equality of all24,704,784 padded31+8 state scalars in every timed,
zero-step and observer endpoint, against the appropriate same-phase reference.
Each of the three observers retains280 complete stage records, with maximum
compatibility defect9.11115e-14. Constraint and horizon diagnostics pass the
existing redistribution gates, with no tolerance changes. Cost vectors are
checked explicitly; only weight4 changes the output writer's rank0 block count
from4 to5. The separately bounded FastFlow seed and previously established
unused-header exclusion remain explicitly recorded. The scientific review is
[`evidence/native-boundary-weight-science-review.json`](evidence/native-boundary-weight-science-review.json).
These qualify the fixed-topology configuration experiment. No refinement,
storage or solver dependency changed; no general AMR weighting claim is made.
The final decision and evidence hashes are in
[`evidence/native-boundary-weight-admission.json`](evidence/native-boundary-weight-admission.json).

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
negative controls; see `evidence/native-window-rank-export-cpu.json`. The subsequent
two-node ABBA experiment also passes the exact GPU state/output and rank-array
checks. Across its 12 windows, maximum-minus-minimum rank durations differ by only
3.308--136.205 microseconds over roughly 91--101 seconds. Native per-cycle timestep
reductions synchronize completion. These endpoint rank IDs do not identify the
causal critical rank.

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

The rejected single candidate replaced `MappedExchange`'s two `MPI_Testall`
completion scans with individual `MPI_Test` calls, following native Bvals.
Every request is polled; unpacking and buffer reuse still require all requests
to complete. Earlier retirement of completed requests is the hypothesis, not
an established explanation or speedup. Only one runtime `.cpp` file changes;
task order, arithmetic, storage layout and ownership remain unchanged. Existing
local MPI/poison tests and a delayed-second-peer check pass. The two-node
unprofiled ABBA screen in job 9052023 (F/0) rejects this candidate: both candidate
run medians exceed both control run medians. Every endpoint passes exact full
31+8, parameter and ordinary history/horizon comparisons. Since performance did
not improve, further observer/AMR qualification is unnecessary and no optimization
is retained. The experiment patches and raw evidence are preserved; see
`evidence/native-mapped-poll-abba-review.json`.

| Run, in execution order | Three 20-step windows (s/complete RK4 step) | Median |
|---|---|---:|
| Control C0 | 4.551556, 4.577729, 4.553905 | 4.553905 |
| Individual polling K0 | 4.602801, 5.040799, 4.915901 | 4.915901 |
| Individual polling K1 | 4.886818, 4.691250, 4.663744 | 4.691250 |
| Control C1 | 4.673306, 4.578566, 4.540198 | 4.578566 |

The six-window medians are 4.565817 seconds for control and 4.789034 for the
candidate (4.89% slower). The rejection rests on the absent repeated improvement,
not the pooled statistic alone. Per-stage observer and AMR tests were not run for
this rejected candidate.

The accepted optimization increases MPI GPU IPC cache capacity, using the unchanged
native executable and identical physics/task order; see
`evidence/native-mpi-ipc-cache-hypothesis.json`. Exact installed-MPI source
and initialization code default to specialized, limited caching with 16 entries
per device. Existing CPU donor-plan metadata projected onto the verified 24-rank
ownership gives 18--20 same-node bulk send allocations on eight ranks, before
surface plans. Every projected bulk send allocation exceeds the allocation-size
IPC threshold. LRU overflow closes the oldest mapping. The sampled trace contains
repeated opens/closes, but their fast API calls do not explain the preceding gaps
and do not prove eviction. Prior environment captures omitted these cache knobs,
so inherited overrides are unknown. The controlled experiment explicitly sets
the common cache mode, clears aliases, and varies only the limit from 16 to 128.
The successful retry below demonstrates capacity sensitivity; individual cache
evictions and a causal critical rank remain unmeasured.
The first capacity screen (job 9052050) produced a successful 16-entry control
median of 4.885799 seconds/step and two successful 128-entry medians of 0.599515
and 0.599614 seconds/step, approximately 8.15 times faster. Each successful case
passes complete endpoint and ordinary diagnostic equality. The final 16-entry
control reached cycle 2556 and wrote matching fields and timing data, but rank 13
then reported SIGSEGV and the batch ended F/143. There is no stack trace; its
precise failing function and cause are unknown. That failed control is not an
admitted performance sample, and no cache optimization is retained yet.

All 81 raw files and the original missing post-run guards are preserved. Separate
offline frozen-source checks pass without converting the failed batch to success;
see `evidence/native-ipc-cache-abba-failure-review.json`. The successful clean
retry does not explain the earlier rank-13 failure.

The unchanged retry, job 9052075, completed F/0. All four ABBA runs and five
qualification runs exited 0; before/after source, executable, object and launcher
guards pass. All 1,035 collected files were size/hash verified. The same native
executable is used in all timing cases, from the admitted cycle-2486 checkpoint,
with 10 warmup steps and three unprofiled 20-step maximum-rank windows per run.

| Run, in execution order | Cache entries | Three windows (s/complete RK4 step) | Median |
|---|---:|---|---:|
| Control C0 | 16 | 4.639723, 4.529840, 4.960999 | 4.639723 |
| Candidate K0 | 128 | 0.594748, 0.594281, 0.594322 | 0.594322 |
| Candidate K1 | 128 | 0.594144, 0.594420, 0.594493 | 0.594420 |
| Control C1 | 16 | 4.740049, 4.843840, 4.612780 | 4.740049 |

Six-window medians are **4.689886 versus 0.594371 seconds/step: 7.8905x faster**.
Every candidate window beats every control window. Candidate startup costs are
3.790/3.542 s and finalization, including diagnostics/output, costs 5.505/5.451 s;
these are outside the complete-step windows. Control startup spans 3.458--4.752 s
and finalization 6.071--6.264 s. Maximum-rank IDs vary; tiny endpoint rank spreads
reflect synchronization and do not identify a causal critical rank.

Independent existing readers verify complete 24,704,784-scalar padded 31+8 state,
initialized metadata, parameters and ordinary history/horizon diagnostics exactly
for all four runs. Further qualification passes:

- 280 observer stage records are exact, with maximum compatibility defect
  9.11115e-14; complete endpoint and diagnostics are exact.
- Poisoned-workspace checks cover 624 dumps, geometry, first RHS, all 11 state
  phases and 32 stage records, exactly matching the clean reference.
- AMR evolution covers refinement and coarsening, 106→127→127→106→106 leaves,
  48,665,448 state/history scalars and 16 stages, all exact to its coherent
  qualified reference.
- Fresh prefix and resumed evolution preserve full state and joined 32 stage
  records exactly. Existing restart diagnostic tolerances pass, retaining the
  extra prefix-final horizon search and its expected output/seed differences.

See `evidence/native-ipc-cache-abba-review.json`,
`evidence/native-ipc-cache-observer-review.json` and
`evidence/native-ipc-cache-poison-amr-review.json` for admission details.
At cache128, the observer run's three-window median is 0.613874 s/step,
0.019503 s (3.28%) above the unobserved pooled median. This is descriptive,
not another randomized overhead experiment; observer timings are excluded from
throughput acceptance. The earlier matched observer controls remain valid for
their original slow configuration.

The retained configuration is
`tst/z4c_hyperboloidal/aurora/native-gpu-ipc-cache.env`. Source it before `mpiexec`
for this validated Aurora N12 layout. It explicitly selects specialized, limited
caching with 128 entries and clears six aliases. Its exported variables match the
measured candidate exactly. No solver C++ changes, task reordering, ownership
changes, altered arithmetic or reduced validity checks are required. Production
launches remain untouched. Capacity128 is qualified here on two nodes; this is
not a universal capacity recommendation or an eight-node scaling result.

The matched native/legacy comparison, job 9052116, completed F/0 and passes
scientific qualification for the rebuilt original control and v3/v6/v8. All use
the same two allocated nodes, 24 tile ranks, explicit cache128, equivalent
cycle-2486 states, compiler/FP policy, timestep and physics. Native runs bracket
the legacy suite; all timing windows are unprofiled complete RK4 steps.

| Implementation | Three windows (s/step) | Median |
|---|---|---:|
| Native before | 0.599136, 0.598923, 0.599410 | 0.599136 |
| Legacy original control | 0.326684, 0.325343, 0.325245 | 0.325343 |
| Legacy v3 | 0.136397, 0.136240, 0.136658 | 0.136397 |
| Legacy v6 | 0.137866, 0.136526, 0.136795 | 0.136795 |
| Legacy v8 | 0.143026, 0.143832, 0.143693 | 0.143693 |
| Native after | 0.599376, 0.599432, 0.599456 | 0.599432 |

The six-window native median is **0.599393 s**, 4.3945 times v3's observed median.
Native bracket medians differ by only 0.0494%. **Parity is not achieved.**
The v3/v6 median difference is only 0.292%, within their observed window variation;
retain both as final-reference contenders instead of claiming a unique fastest
version. The current gap is much larger than this uncertainty.

Each legacy endpoint checks all 7,137,936 field values and 1,842,048 chi records.
Optimized variants are bitwise equal to the rebuilt original control. Against
native, maximum bulk/surface field differences are 1.795e-13/4.258e-13, at most
0.03594 of the existing field gate. Two bulk chi minima differ by 2.776e-17;
event counts, first-floor times and corrections are exact. Compatibility,
constraints and horizon gates pass; areal radius and mass are exact. All 169
harmonic coefficients are covered and finite. Their maximum 3.818e-9 difference
is reported as diagnostic-only under the predeclared protocol, not a new relaxed
coefficient tolerance. Both actual native brackets match all padded 31+8 state
and ordinary outputs exactly. The immutable original campaign executable also
passes its initial/final fields and 70-step clock/CFL checks; its SP1 format does
not supply seeded chi-history equivalence.

Two reader failures are preserved. The first required the exact archived timer
header for an older scientific report; current native timings use their own
independently reviewed provenance. The second exposed a schema mismatch: the
original producer with `extended=false` fills its required r>.94 exterior sums,
while an optional duplicate fixed-cut array remains zero. The corrected reader
checks the populated exterior sums and requires that inactive array to be zero.
All numerical gates are unchanged; 28 existing/focused offline controls pass.
No solver rerun or numerical change was needed. See
`evidence/legacy-native-two-node-matched-review.json`, which binds both full
reviews, the successful collection and preserved failures. Startup, accepted
output and final diagnostics are separately recorded there; no setup speed claim
is inferred across the different drivers.

The cache128 profile, job 9052139, completed F/0. Its normal/profiled four-step
continuations pass complete state, diagnostics, actual runtime/library identity
and cache-environment checks. The new normal endpoint also matches the earlier
normal2490 exactly. The existing profiler and extractor are reused, including
ranks 8/23 and unchanged trace options. Actual text nanosecond summaries agree
with Chrome microsecond durations and launch counts.

| Sampled rank | RHS span | GPU interval union | Pure bulk RHS kernels | Surface solve + RHS |
|---|---:|---:|---:|---:|
| 8, bulk-heavy | 2.702677 s | 1.279110 s (47.3%) | 1.232348 s, 48 launches | Empty reductions, microseconds |
| 23, surface owner | 2.570627 s | 2.036681 s (79.2%) | 1.463776 s, 64 launches | 0.302860 + 0.200479 s |

Each span is local to one rank, from its first to last bulk RHS kernel; it is
neither a complete-step timing window nor a globally aligned critical path.
GPU interval coverage is not achieved occupancy. Bulk validity reductions are
excluded from the main-kernel totals. On rank23, 16 normal solves and 16 surface
RHS evaluations each launch one 256-work-item group, taking up to 19.06/12.62 ms.
Bulk shapes remain 1×1024 (up to 43.69 ms), 8×1024 (25.41 ms), and 1×512
(12.85 ms). On rank8, bulk shapes are 11×1024, 1×1024 and 2×1024,
with maxima about26 ms. Counts and shapes describe executed launches, not
occupancy or useful-thread utilization. See
`evidence/native-n12-profile-cache128-findings.json`.

The large transfer-adjacent gaps have substantially contracted from the earlier
~17-second sampled spans. Bulk RHS now dominates recorded GPU work on both
representative ranks. This selects the next **single experiment: field-parallel
KO using existing RHS storage**. The per-field operator, its line summation,
transport/KO/Lambda-to-D addition order, subset dependencies and validity checks
remain unchanged. No workgroup tuning is combined with this experiment. The
candidate has passed existing two-rank full-RHS CPU oracles at KO=.1 and KO=0.
Its two-node ABBA screen, job9052165, completed F/0. Both control and candidate
use cache128, identical saved-state continuations and the same complete-cycle
timing driver, with no profiler or stage observer inside the measured runs.

| Run | Three windows (s/complete RK4 step) | Median |
|---|---|---:|
| Control C0 | 0.600802, 0.600835, 0.600995 | 0.600835 |
| Field-parallel KO K0 | 0.550150, 0.549933, 0.549831 | 0.549933 |
| Field-parallel KO K1 | 0.549575, 0.549876, 0.550191 | 0.549876 |
| Control C1 | 0.600959, 0.600457, 0.600999 | 0.600959 |

The six-window medians are 0.600897 and 0.549904 s/step, an 8.49% time reduction.
Every candidate window beats every control window. Complete padded31+8 states,
parameters and ordinary diagnostics match exactly in all four cases. This is a
successful timing screen. At the user's earlier pause, the candidate was archived
and the runtime restored; that historical status remains in
`evidence/native-field-ko-abba-review.json`. After resumption, the identical source
passed full applicable GPU qualification in job9052266 (F/0):

- Observer70: all280 stage rows, complete24,704,784-scalar padded31+8 state,
  parameters and ordinary constraint/horizon diagnostics match exactly;
  maximum compatibility defect9.11115e-14.
- Poison8:98,779,824 state/history scalars,7,137,936 first-RHS values,
  690,768 geometry values and32 stage rows are bitwise exact to the clean reference.
- Coherent AMR4:48,665,448 owned31+8 scalars acrossfive snapshots,16 stages,
  topology/ownership and both refinement/coarsening transitions match exactly.
- Prefix2/resume8: complete accepted states and joined32 stages match continuous
  evolution exactly; the established restart diagnostic phase gates pass,
  retaining the extra prefix-final horizon search.

All980 collected files were verified. The independent scientific reviews are
`evidence/native-field-ko-observer-restart-review.json` and
`evidence/native-field-ko-poison-amr-review.json`; the combined acceptance record is
`evidence/native-field-ko-admission.json`. Qualification timings are excluded from
performance acceptance. The field-parallel KO source is now retained. No wider
scaling or rank-changing restart claim is made. Surface batching remains untested.

The subsequent main bulk workgroup128 experiment, job9052307 (F/0), compares
against the accepted KO implementation with identical cache128 settings:

| Run | Three windows (s/complete RK4 step) | Median |
|---|---|---:|
| Control C0 | 0.552354, 0.551167, 0.551552 | 0.551552 |
| Bulk128 K0 | 0.379914, 0.379943, 0.380023 | 0.379943 |
| Bulk128 K1 | 0.380238, 0.380206, 0.380160 | 0.380206 |
| Control C1 | 0.551316, 0.551853, 0.551677 | 0.551677 |

The six-window medians are0.551614 and0.380092 s/step. All four complete
24,704,784-scalar padded31+8 endpoints, parameters and ordinary diagnostics are
bitwise equal. The same source passed all five qualification cases in job9052329
(F/0): observer280 stages; poison98,779,824 state/history scalars and complete
first RHS; AMR48,665,448 scalars acrossfive snapshots and two regrids; and
prefix2/resume8 continuity. The independent reviews retain the existing restart
diagnostic phase gates and confirm bitwise agreement with accepted KO. All989
qualification artifacts were verified. No numerical tolerance changed.

That allocation also ran the existing normal/profiled four-step protocol. Both
complete endpoints and diagnostics match, and the normal endpoint matches the
retained cycle2490 reference exactly. Main and KO kernels are distinguished by
their actual argument signatures, not lambda numbering. The trace confirms
main workgroup128 and unchanged KO workgroup1024. See
`evidence/native-bulk-group128-profile-review.json`.

| Sampled rank | Main RHS GPU time | KO GPU time | Normal solve + surface RHS |
|---|---:|---:|---:|
| 8 | 0.371550 s, 48 launches | 0.005049 s, 48 launches | No nonempty surface kernels |
| 23 | 0.901014 s, 64 launches | 0.003889 s, 64 launches | 0.302570 + 0.199277 s, 16 each |

These are recorded kernel durations over four steps, excluding initialization
and final diagnostic surface calls, not complete-step timings or a global
critical path. On rank23, the5×128 main launch contributes0.542856 s across16
calls, with a maximum35.38 ms/call. Its place after normal solve/surface RHS in
all16 stages identifies the surface-dependent bulk subset. The other three
bulk subsets together take0.358159 s. Source dependencies predict576 points in
that subset, including432 radial reconstruction points and288 normal-lift
points; group shapes corroborate but do not count active work-items themselves.

This selected the field-batching experiment: batch the existing endpoint radial
derivative loops as in validated legacy `ScriRadialDevice::Prepare`. The slow
kernel also includes restoration, normal correction and the complete point RHS,
so the radial contribution remains a hypothesis to test with complete-step
timing. Its subsequent repeated comparison measured0.379937 versus0.300340s/step.
The exact source preserves each field's term order and refreshes stage-local
storage before the existing surface-dependent batch. All four complete timing
endpoints match the admitted31+8 state and diagnostics exactly. Independent
qualification covers280 observer stages,98,779,824 poison state/history scalars,
48,665,448 AMR scalars acrossfive snapshots/two regrids,32 joined restart stages,
and3,942,432 exact private-cache components in the refined fixture. The latter
exposes101,088 stale values and checks NaN overwrite and Rebuild refresh. Existing
operator bounds and restart diagnostic gates remain unchanged. Full evidence is
linked by `evidence/native-radial-fields-admission.json`.

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
| Individual MPI completion checks | All four full endpoints/diagnostics exact; local mixed-peer tests pass | Control 4.565817 s; candidate 4.789034 s (six-window medians) | Rejected; no repeated improvement; original runtime restored |
| MPI GPU IPC cache capacity, first screen | Three successful endpoints exact; last control failed after output | Successful medians 4.885799 vs 0.599515/0.599614 s | Failed full ABBA preserved; not admitted |
| MPI GPU IPC cache capacity, clean retry | Four exact full endpoints; observer, poison, AMR and restart pass | 4.689886 → 0.594371 s; all six candidate windows faster | Accepted native improvement, 7.89x; parity pending |
| Matched previous-implementation reference | All four references pass full31+8/constraints/horizon gates; actual native brackets exact | Native 0.599393 s; v3/v6 0.136397/0.136795 s | Parity not reached; leading references about4.39x faster |
| Field-parallel bulk KO | Four complete endpoints exact; CPU KO.1/0 oracles, GPU observer, poison, AMR and restart pass | Control 0.600897 s; candidate 0.549904 s; all six candidate windows faster | Accepted, 8.49% lower time; native infrastructure preserved |
| Main bulk SYCL workgroup128 | Four complete endpoints exact; observer, poison, AMR, restart and profile-reference equivalence pass | Control 0.551614 s; candidate 0.380092 s; all six candidate windows faster | Accepted, 31.09% lower time; arithmetic and dependencies unchanged |
| Post-KO/bulk128 selected-rank profile | Normal/profiled/reference complete31+8 and diagnostics exact | Rank23 surface-dependent bulk0.542856 s across16 launches; KO only0.003889 s | Select field-batched radial derivatives next; no occupancy or global critical-rank claim |
| Field-parallel endpoint radial derivatives | Full timing endpoints exact; observer, poison, AMR, restart, refined private-cache stale/poison/Rebuild and profile equivalence pass | Control0.379937 s; candidate0.300340 s; all six candidate windows faster | Accepted,20.95% lower complete-step time; surface and geometry arithmetic unchanged |
| Post-radial selected-rank profile | Normal/profiled/reference full31+8 and diagnostics exact | Rank23 main0.575609 s, normalSolve0.302462 s, surfaceRHS0.200214 s acrossfour steps | Surface organization remains costly; current user instead requests isolated weight2/3 comparison |
| Boundary weights2/3/4 through native restart/load balance | All14 cases pass; full31+8 states bitwise,840 observer stages and existing constraint/horizon gates pass | Weight1/2/3/4 medians0.301176/0.299043/0.299823/0.298797 s | Qualified isolated configuration gains0.71/0.45/0.79%; no permanent weighting policy; original goal remains cleared |

Completion requires the committed native implementation to match or beat the
fastest accepted previous N12 reference median on the same two nodes, with repeat
variation small enough to support that conclusion, and all applicable architecture
and scientific requirements verified. A partial speedup is progress, not completion.
