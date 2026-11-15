# Experimental hybrid overlap

Status: v4 passes the measured two-node performance criterion and the focused
correctness checks. **Late-time scientific benefit and matched-resolution
acceptance remain unestablished.** The selected N12 mask-plus-agreement run
failed during evolution after 18.750454 M0 history output, while archived legacy
reached 30 M0. N16 continues toward its <=30 M0 limit. Legacy remains the
production default.

## Selected scientific comparison

Per the user's updated scope, only N12 and N16 mask-plus-agreement are evolved
to the fixed limits 21300 and 28400 cycles (29.997346 M0). The completed
November 9 N12 and November 10 N16 legacy campaigns supply the baseline.
Redundant fresh legacy and mask-only evolutions were stopped; their saved
checkpoints and short controls are retained. No old campaign is resumed.

N12 job 9077378 terminated with scheduler exit 1 and
`Invalid hyperboloidal metric chart`. The exception is raised by algebraic
metric reconstruction when the leading metric component or its 2x2 principal
minor fails the positivity test, including invalid values. This is an evolution
failure, not a horizon acceptance failure. Its last complete checkpoint is
18.001224 M0 (cycle 12782); independent reduction found all stored fields finite
and unchanged cumulative chi-event counts there. The later failed RK stage is
not stored, so the exact failing point and condition cannot be recovered from
that checkpoint alone. The selected configuration has not met the N12 30 M0
requirement. No restart with bypassed checks or changed parameters is requested.

The source audit found that the added scri treatments are disabled for these
inputs, preserving the existing active equations. Archived and freshly captured
legacy initial checkpoint payloads are bitwise identical at both resolutions.
At 3 M0, N16 remains bitwise identical across all stored fields and persistent
chi records, despite the archived run initially using four nodes. N12 used an
earlier diagnostic executable for 0–10 M0 and is not bitwise identical at 3 M0.
Its matched exterior H/M/Theta/Z histories through 3 M0 differ by at most
3.97e-10 relative. This is an early compatibility check, not a bound on future
roundoff growth. Historical N12 scri and puncture histories start at 10 M0.

`legacy-reuse/archived-legacy-receipts.tar.gz` in the validation directory
preserves the original history bytes, input files, source manifests, executable
hashes, and checkpoint metadata. SHA256:
`1d3f291bbd30ff471c3d29958817ee3909960b99fbf2aa6c942305b3ebdf38be`.
The old terminal steps reached 30.000163 and 30.000515 M0; their recorded times
are preserved and no new evolution exceeds 30 M0. Native read-only checkpoint
diagnostics at approximately 26/28/30 M0 provide common fixed-region and
overlap-observer measurements without changing the saved states.

The report accepts `--legacy-archive` pointing to the extracted receipt tree.
In this mode completion requires the two archived legacy cases and two new
agreement cases; partial mask-only controls do not block completion. A completed
run matrix alone does not establish scientific acceptance.

The selected evolutions use bounded two-node segments with same-case checkpoint
hash/identity checks and one queued successor per running case. Long capacity
continuations that remained resource-queued for over 90 minutes were replaced by
the previously successful debug/debug-scaling protocol. This scheduling change
preserves the frozen v4 executable, numerical inputs, and final cycle limits.

The early constraint-growth audit found a concrete change in KO near the cut.
All compared inputs use `ko=.1`, applied at every bulk RK stage, including the
Lambda contribution to D. Cartesian KO lines crossing Rc are omitted. A
geometric reconstruction on nominal Cartesian nodes gives mean retained
checkerboard damping of 45.4% (N12) and 56.1% (N16) in r=.24..25, versus legacy;
inside r=.23 it is unchanged. These are reconstructed responses, not measured
nonlinear RHS budgets. The common coefficient does not imply equal boundary
dissipation.

Native read-only onset diagnostics localize 92–98% of the positive excess H/M
squared integrals in the audited r>=.18 region to r=.20..32 at the matched
N12 8/10M0 and N16 12M0 checkpoints. Existing N16 controls through 7 M0 reproduce
the early excess with agreement disabled. At 7 M0 in r=.25..32, mask-only/legacy
H and M RMS ratios are 4.5412 and4.5568; agreement/legacy ratios are 4.5247 and4.5433.
This implicates changes shared by the masked modes, while KO closure versus
interpolation and later-time causality remain unresolved. No additional mask-only
evolution was performed. The nine-checkpoint control receipt archive SHA256 is
`7ca1d5b7373b5ef8855eb711dec65a1e294fa1d93555570f8a27fa1f8bfa56b4`.

## Current performance qualification

Four fresh 947-step windows (1.000264 M0 each), in legacy/agreement/agreement/
legacy order on the same two-node allocation, include initialization, evolution,
required diagnostics, and normal-cadence restart output. There is no timestep
reduction, resolution/order change, or weakened damping: Rc=0.25,
Rs=0.18, sampling band 0.18..0.195, kappa=1 (kappa M0=1/3).

| Mode | Windows | Mean window-median step (s) | Mean interval including diagnostics/final output (s) | Wall/M0 mean ± sample SD (s) |
| --- | ---: | ---: | ---: | ---: |
| Captured legacy | 2 | 0.705210 | 0.714114 | 776.090 ± 10.489 |
| Agreement v4 | 2 | 0.696858 | 0.755123 | 818.223 ± 3.458 |

The mean wall/M0 ratio is **1.05429**, below 1.15. The temporal-pair ratios are
1.04743 and 1.06128; the all-window envelope is 1.04119..1.06764 (an observed
range, not a confidence interval). Startup to native cycle zero was
94.104..104.687 seconds for legacy and 96.481..109.554 for agreement. Final-step
plus output intervals were 33.517..41.199 and 70.132..84.153 seconds, respectively;
these intervals include a PDE step and are not instrumented pure I/O times.
The complete earlier 300-step protocol also passed all three temporal pairs,
with pooled mean ratio 1.01945 but much larger startup/output variation.

The qualified run is 9076481. `aurora-v4/paired-long/` contains its checked
receipts and reviewed plot/table in the validation directory. Archive SHA256:
`f9d38cb29dd51c269ad3b908acd63648719e482843b6422be5d5444fe3444ae9`.
One-M0 accepted payloads in timing and fresh science runs are bitwise equal for
both legacy and agreement. This timing qualification applies to the frozen v4
configuration; it does not establish late-time scientific usefulness.

## Frozen working-tree baseline

The baseline is the complete modified working tree at
`8d9a34ec37bc05461e7a31f88722fbcbc3d16560`, not a checkout of that commit.
The local validation directory is
`/Users/hz0693/.codex/worktrees/hyperboloidal-binary-integration/hybrid-overlap-validation`.
`baseline-20261111/manifest.json` hashes 3,803 captured files, including relevant
untracked sources, Kokkos, inputs, build settings, and local executables.
`working-tree.tar.gz` SHA256:
`09919d91aa3d3f2acb5f892e024221e5d829369fd93dfc8ee5f653c2c7fd7ace`.
The independently extracted source was rebuilt for disabled-mode comparison.

Aurora validation is isolated under
`/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111`.
Its baseline build archive SHA256 is
`1f3e7bd434df5402af6583d47fb5a8af61bf8b73d73c7042715918875d8d1aeb`.
Previous production campaigns are read-only inputs; they are not resumed.

## Inputs

All controls below belong to `<z4c_hyperboloidal>`.

| Control | Meaning |
| --- | --- |
| `overlap_mode = legacy` | Default; original work lists, donors, and restart identity. No agreement plan or exchange. |
| `overlap_mode = masked` | Spherical Cartesian cut and evolved-only fringe donors; no agreement damping. |
| `overlap_mode = agreement` | Mask plus paired agreement operator on all 26 independent fields. |
| `cartesian_cutoff` | Required for masked modes; finite Rc greater than the atlas spherical inner radius Rs. |
| `agreement_kappa` | Required positive finite rate in native inverse-time units for agreement mode. |
| `agreement_inner` | Sampling-band inner radius, default Rs. |
| `agreement_outer` | Sampling-band outer radius, inner < outer <= Rc. Default: midpoint between Rs and min(Rc, Cartesian half-width). Set explicitly for matched comparisons. |

These are separate from the existing scri penalty and atlas evolved-overlap
storage controls. No production input has been changed. Geometry is checked
using actual sixth-order interpolation footprints, radius-two mixed differences,
and radius-four directional dependencies, including angular corners. Invalid
coverage is rejected without reducing interpolation order. Required donor points
must evolve; recursive interpolation through fringe values is forbidden.

## Activity and integration

`ActivityPolicy` owns the geometry predicate and the evolved/fringe/inactive
classification. Evolved points remain within the existing native atlas storage
bounds; Cartesian nodes additionally satisfy r < Rc and spherical nodes r >= Rs.
A fringe point is in the stencil dependency union of at least one evolved node.
All other stored nodes are inactive. Compact native work lists drive initialization,
reference/residual caches, RHS, reconstruction, RK updates, and restart packing.
Halo rows fill only the dependency union. Rectangular arrays and native ownership
remain unchanged.

Cartesian KO lines are retained only when their five-point physical footprint
stays inside the activity cut. Shell KO lines require evolved nodes. AMR transfer
staging obtains every required corner from evolved donors, even where the stored
nominal Cartesian field is inactive. Transfers and history folds use compact
new-target indices; newly active points are populated before plans are rebuilt.
When coarsening removes every evolved center in a thin Cartesian active sliver,
its chi ledger moves once to the nearest surviving Cartesian node through the
existing sparse history gather. Native min/max, slope, second-derivative, and binary chi criteria exclude inactive
points. The existing native `NeedsAcceptedHalos` route refreshes fringe values for derivative criteria.
Horizon and puncture samplers share the same donor eligibility policy. Scri
geometry and its surface evolution are unchanged.

Diagnostic partition taper ends at min(Rc, Cartesian half-width); its support
therefore stays within nominal Cartesian coverage. Existing fixed outer-shell
regions are retained. Restart block slots remain fixed: unused slots are zero,
and only evolved points are packed/unpacked. Mask and agreement parameters are
part of the mathematical identity. No checkpoint migration is implemented.
A fresh binary initialization may select active nodes from a full original export;
all file identities, coordinates, field conventions, and algebraic checks remain
mandatory. This is not a restart migration.

## Paired operator

Let q be the independent reference-subtracted residual. Chi uses the configured
regular residual weight, and the auxiliary coordinates are Lambda and D-Lambda.
All components are in the common Cartesian tensor basis. Fields are measured in
their stored native units with unit component scaling. This defines a numerical
quadratic agreement energy, not a physical Z4c energy. Dependent metric/curvature
components and dormant J fields are excluded.

The samples are the union of nominal evolved Cartesian and spherical nodes in
`(agreement_inner, agreement_outer)`. Each quadrature contributes half its positive
coordinate volume. At a node of one grid its interpolation is an identity selection;
the other grid uses the full existing interpolation order. This avoids relying
solely on a spherical sampling grid that can under-resolve Cartesian oscillations.
The taper is `64 t^3 (1-t)^3`, where t is the normalized sampling-band radius.
H is the positive coordinate volume at every evolved recipient, including overlap
nodes; diagnostic partition weights are never used for H.

With A = [Pc, -Ps], the implemented correction is
`qdot = -kappa H^-1 A^T W A q`. Forward and transpose use the same coefficients.
Thus the penalty alone has `dE/dt = -kappa (Aq)^T W (Aq)`,
`E = q^T H q / 2`. Chi corrections are multiplied by the residual weight on return
to stored RHS units, and `Ddot += (D-Lambda)dot + Lambdadot`.
The existing algebraic reconstruction follows every RK update.

Each rank evaluates the rows needed by its owned transpose recipients using a
cached sparse donor gather. Precomputed recipient adjacency avoids atomic scatter
and a second sequential correction exchange. Geometry/ownership changes rebuild
the plan. Stage data and all working arrays stay on device. Receive posting and
packing join the native stage transport, and the correction task waits for all
ordinary RHS batches before releasing the RK update. No per-stage global
collective or host field copy was added.

A cached conservative rate bound is
`kappa max_i H_i^-1 sum_s W_s |A_si| sum_j |A_sj|`.
The plan rejects `dt * bound > 2.5`, within classical RK4's negative-real interval.
The bound is recomputed after remeshing. This is a penalty-only timestep test;
it does not establish stability of the coupled nonlinear equations. The existing
fixed-step policy is retained rather than silently changing dt.

## Diagnostics and validation

Masked modes add an AthenaK `.overlap.hst` stream with six field-group disagreement
norms, penalty dE/dt, sampling volume, evolved/fringe counts, and sparse send/receive
donor counts. Mask-only evaluates the same disagreement at history cadence, with zero penalty; it has no stage agreement communication or kernels. Legacy allocates neither plan. Kokkos labels distinguish
discrepancy, transpose, and mapped exchange kernels for profiling.

Initial evidence in the local validation directory:

- `disabled-regression.json`: initial and one-step MPI restart payloads bitwise
  identical to the captured working-tree baseline (197,642,734 bytes each).
- `ctest.log`: 12/12 registered tests passed.
- `agreement-smoke-mpi/results.json`: two-rank distributed penalty energy identity,
  constant reproduction, grid-scale response, smooth interpolation error, and four
  native RK stages with poisoned non-evolved storage.
- Refined test geometry correctly rejects its original dt when the conservative
  penalty bound is exceeded. The separate AMR-transfer fixture uses dt=1e-6;
  this smaller test step is not a performance result.

The tests extend the existing stage-smoke, task-graph, and AMR-lifecycle harnesses.
The qualified Aurora performance result is reported above. The selected late-time
comparison remains required before a scientific assessment.

## Current validation geometry and run protocol

Actual head-on coverage passes at both N12 and N16 with Rs=0.18, Rc=0.25,
agreement band (0.18,0.195), and kappa=1 (kappa*M0=1/3). Rc=0.23 passed
N16 but failed N12 six-point support near angular corners, so it is not the
cross-resolution comparison geometry. The original fixed timesteps pass the
penalty bound; no timestep reduction is used in the paired benchmark.

For this selected Rc=0.25 geometry, the diagnostic taper still ends at the
Cartesian half-width 0.2. Every previously positive-weight nominal point remains
active, so the physical-domain quadrature is mathematically unchanged. All three
eight-step GPU admissions report exactly the same volume, 4.167650360781702.
Native checkpoint work lists contain 3,215,584 evolved bulk points for legacy
and 2,972,024 for either masked mode, a 7.574% reduction. The maximum/mean rank
point count changes from 1.21546 to 1.14892. These counts do not substitute for
the profiled rank-dependent cost of each point or penalty row.

All paired measurements retain initialization and standard output costs, with
source, executable, input, initial-data, module, binding, and cache-setting
receipts. Selected-rank profile timings are attribution evidence and are kept
separate from throughput measurements.

The short single-hole matrix uses N12 (12 steps) and N16 (16 steps) at matched
physical time and fixed kappa*M0 for legacy, mask-only, and agreement. It is an
admission check, not late-time stability or convergence evidence. Scientific
head-on inputs are prepared for matched cycle limits 21300 (N12) and 28400
(N16), ending at 29.9973461871 M0. This avoids the existing fixed-step driver's
last-step overshoot beyond the user-specified 30 M0 ceiling.

`history_output.cpp` also has a diagnostic-only checkpoint mode:
`history_output saved.rst checkpoint Rc agreement_inner agreement_outer`.
It reads native restart identity normally, takes zero evolution steps, writes
fixed-band raw constraint integrals and a common overlap observable through
native HistoryOutput, and checks accepted fields/chi records bitwise unchanged.
It does not migrate checkpoint policies. The fixed bands are
0.18, 0.20, 0.25, 0.32, 0.40, 0.55, 0.80, 0.94, 1.00; a 1e-12 boundary tolerance
assigns nodes on a band boundary consistently. `hybrid_checkpoint_audit.py`
independently checks their Theta/volume sums from fixed checkpoint slots and
extracts shell-corner Theta in the fixed angular region |u|,|v| >= 0.75.
Legacy and agreement checkpoint checks passed at 3e-12 relative tolerance.

Additional local receipts include `agreement-validation.json`,
`agreement-restart-equivalence.json`, `masked-criteria/results.json`,
`agreement-amr-v3-valid/results.json`, and `checkpoint-history/*-bands-v2/`.
The AMR test now poisons every non-evolved curved-cut point and includes the
vanishing-active-sliver ledger-routing case. The v3 changes after the frozen
Aurora v2 bundle concern AMR history routing and initialization receipts;
v2 benchmark results must not be labeled as a v3 executable measurement.

### First Aurora pilot (candidate v2)

The completed 300-step pair at matched dt and physical interval measured:

| Mode | End-to-end seconds | Median step seconds | Mean post-warmup interval seconds |
| --- | ---: | ---: | ---: |
| Captured legacy | 328.596 | 0.575726 | 0.931377 |
| Agreement v2 | 431.819 | 0.956048 | 1.250786 |

End-to-end wall/M0 ratio: **1.31413**, above the 1.15 requirement. Median-step
ratio: 1.66060. Means include history-bearing intervals and final output; they
are not pure-kernel times. The remaining v2 windows were stopped after this
pair to profile and optimize; this is a failed pilot, not the final repeated
acceptance measurement. Raw logs, measurements, and `early-stop.json` remain
under `runs/paired-9076258` and the local `aurora-pilot-v2` receipt directory.

The six local short single-hole cases now pass with finite histories and zero
cumulative chi-floor, nonpositive, or nonfinite events. Their exterior H/M/Theta/Z
N12-to-N16 norm ratios lie between 2.217 and 2.365 and are nearly unchanged across
all three schemes. The interval is only 0.0168999 M0; these results do not replace
the required late-time scientific comparisons. See `single-hole-short/assessment.json`
and `constraints-short.png` in the isolated validation directory.

### Profile-directed candidate v4

The separate unitrace job 9076298 completed all three eight-step modes. Each
accepted checkpoint payload was bitwise identical to its unprofiled counterpart
(2,483,495,040 bytes). Selected ranks were 0, 8, 22, and 23. This is attribution
evidence, not throughput: host waits overlap device work. On rank 22 the old
agreement discrepancy and transpose kernels used respectively 0.108774 and
0.220496 GPU seconds per RK cycle, versus 0.298022 for the bulk RHS.

Candidate v4 assigns one GPU work item to each row/field and recipient/field,
coalescing field reads and eliminating the private 26-component correction array.
Every scalar accumulation retains its entry order; the actual interpolation,
transpose, weights, chi conversion, D-Lambda conversion, kappa, and dt are unchanged.
Local serial and two-rank MPI accepted bulk/scri states are bitwise identical to
the prior kernels. Energy identity, inactive poisoning, and all 12 registered
tests pass. Aurora source qualification verified 2,211 source/input files.
Its executable SHA256 is
`d4f1faa63a41b68b9b6ebaf4d44ccb1df1f015ac98952fb6a7707b4e8191c833`.

Job 9076396 completed the short repeated paired protocol with this frozen candidate.
Each short GPU admission reproduced the v2 checkpoint payload bitwise before
sustained windows began. Plan rebuild
receipts also count the hypothetical single-owner gather/return traffic, allowing
the implemented one-phase duplicated-row strategy to be compared with a two-phase
owner-compute strategy without changing stage behavior.

All three v4 GPU admissions passed bitwise comparison. Startup and forced final
output dominate short-window end-to-end measurements; the longer, normal-cadence
windows reported above provide the performance qualification.

The N16 plan has 151,792 unique sample rows and 302,208 distributed evaluations
(duplication factor 1.99093). The chosen sparse gather transfers 71,828,432 bytes
globally per RK stage in one phase. A one-owner row strategy would gather and
return 65,240,032 bytes each, in two sequential phases. These alternative-volume
counts use unique remote donors per owner and aggregated returns; they are not
measured timings of an alternate implementation. Evolved points total 2,972,024;
the most loaded rank by point count has 1.14892 times the mean. Actual GPU
attribution, rather than point count alone, determines the critical rank.
`dt * rate_bound = 0.222568771`, with no timestep reduction. The maximum local
plan rebuild time in this admission was 5.814 seconds.

The launcher starts in the segment directory because native puncture tracker
files open before AthenaK applies `-d`. Agreement continuations may resume only
their own isolated checkpoints after verifying checkpoint hashes and native
identity. Profile/diagnostic job 9076440 is separate from throughput measurements.

The raw unitrace `dsize` increments for agreement sends are half the native FP64
payload sizes inferred from exact sparse-plan counts. They are retained as raw
profiler metadata and are not labeled wire bytes or automatically rescaled.
Both qualified builds have `SINGLE_PRECISION_ENABLED=0`; communication-volume
comparisons use the native donor counts and eight-byte Real. Traced Allreduce
call counts are identical across the three modes in each selected-rank window;
the added stage operator introduces no global collective.

The completed v4 profile preserved all three accepted checkpoint payloads
bitwise. On rank 22, discrepancy evaluation fell from 108.774 to 9.845 GPU
milliseconds per RK cycle (11.05x), and transpose accumulation from 220.496 to
66.064 ms (3.34x). These are selected-rank instrumented costs, not end-to-end
speedups. Native GPU checkpoint diagnostics also passed all three modes,
preserved accepted fields/chi records bitwise, left original checkpoint files
unchanged, and matched independent Theta/volume integrals within 3e-12 relative.
Receipts and reviewed plots are under `aurora-v4/` in the validation directory.

The complete v4 short-window protocol has three baseline and three agreement
300-step windows, plus one mask-only window:

| Mode | Mean window-median step (s) | Mean interval including diagnostics/final output (s) | Wall/M0 mean ± sample SD (s) |
| --- | ---: | ---: | ---: |
| Captured legacy | 0.697009 | 0.882227 | 1088.302 ± 164.490 |
| Mask only (one window) | 0.528282 | 0.816991 | 957.157; variability not estimated |
| Agreement v4 | 0.708163 | 0.932824 | 1109.470 ± 137.085 |

Pooled mean wall/M0 ratio is **1.01945**. The three temporally paired ratios are
1.06103, 0.92743, and 1.08663. Startup/output variation is substantial, so the
cadence-aligned longer protocol supplies the qualification above. Neither these
timings nor the successful operator tests establish late-time scientific benefit.

### Reproducing focused checks and reports

From the worktree root, with the existing independent candidate build:

```sh
validation=../hybrid-overlap-validation
ctest --test-dir "$validation/candidate-build" --output-on-failure
python3 tst/z4c_hyperboloidal/stage_smoke.py \
  --build "$validation/candidate-build" --output "$validation/new-agreement-check" \
  --input "$validation/agreement-regular-smoke.athinput" --ranks 2
python3 tst/z4c_hyperboloidal/aurora/hybrid_performance_summary.py RUN_DIRECTORY
python3 tst/z4c_hyperboloidal/aurora/hybrid_performance_plot.py PERFORMANCE_SUMMARY_JSON --output PLOT_DIRECTORY
python3 tst/z4c_hyperboloidal/aurora/hybrid_plan_summary.py RUN_LOG --output plan.json
python3 tst/z4c_hyperboloidal/aurora/hybrid_profile_summary.py PROFILE_DIRECTORY
python3 tst/z4c_hyperboloidal/aurora/hybrid_profile_plot.py PROFILE_SUMMARY_JSON --output PLOT_DIRECTORY
python3 tst/z4c_hyperboloidal/hybrid_science_report.py SCIENCE_DIRECTORY \
  --legacy-archive ARCHIVED_LEGACY_RECEIPTS --output REPORT_DIRECTORY
```

Each report hashes its inputs or records the run provenance. The science report
labels partial data explicitly, retains full and horizon-exterior histories,
tracks scri amplitudes and horizon acceptance policy, and reports two-resolution
norm ratios without interpreting them as a measured convergence order. The
checkpoint reducer additionally checks every stored evolved/scri slot for finite
values and sums all cumulative chi-event records, separately from nominal-cell
spatial statistics.

Early N16 agreement histories are finite through 3.00079 M0. Independent checkpoint
reductions at that time found 69,344 cumulative chi-floor events, 69,128
nonpositive events, and zero nonfinite events. All recorded events are at
nominal points; the maximum compact radius with a historical floor remains
0.101565. This location statistic does not by itself establish containment
inside a horizon. Matched late-time scientific assessment remains pending.

`hybrid_science_report.py` now also plots fixed-radius Theta, shell-corner
Theta in r=0.32..0.40 with |u|,|v|>=0.75, identical-coordinate samples, cumulative
chi events, scri diagnostics, trajectories, and diagnostic surface behavior.
Native checkpoint audits additionally supply fixed-band H/M/Z plots. The
historical horizon-summary `hrms` column is a mean square; the report takes
its square root to display expansion RMS. Mass-only surface acceptance is
reported explicitly. Partial science snapshots are stored as immutable tar
archives with hashes computed from the exact archived bytes.

`hybrid-checkpoint-two.pbs` and the science mode of `hybrid_checkpoint_batch.py`
accept an explicit JSON list of at most twelve isolated science or selected
archived legacy checkpoints.
They reuse the already-qualified native GPU history harness without evolution,
verify source checkpoint hashes before/after, and run the independent reduction.
They do not evolve or change the previous production campaign. The archived
legacy 26/28/30 M0 batch 9076992 completed all six checks: accepted states were
bitwise preserved, source checkpoint hashes were unchanged, and independent
Theta/volume sums matched native diagnostics. Its 119-file receipt archive is
`legacy-reuse/archived-legacy-native-audits.tar.gz`, SHA256
`8b8e54bc839873688f7a329e4306bbf826e8073273b6e5cbf25de74b88c3a8fe`.
All six checkpoints have zero cumulative nonfinite chi events; the floor and
nonpositive counts are unchanged from the early legacy checkpoints.
The final agreement diagnostic selector rejects partial segments before selecting
its six checkpoints. The first held audit 9077179 was canceled when its queued
capacity dependencies were replaced; it will be submitted against the actual
final segments.
Unrelated queued and held jobs remain untouched.

The independent native overlap observer reproduces production mask-only and
agreement group norms/sample volume to 3e-12 relative on the qualified GPU
checkpoints (`aurora-v4/overlap-observer-check.json`). The science reducer now
records that observer and an exact-time production comparison where available,
allowing legacy overlap disagreement to appear in the same scientific report.
At the first one-M0 endpoint, legacy and agreement share exactly 47,072
floor/nonpositive events, zero nonfinite events, and the same maximum historical
floor radius. Their scri history files are bitwise identical. Fixed-band Theta
changes are mixed: agreement/legacy RMS is 1.379 in r=0.25..0.32 and 0.572 in
r=0.32..0.40. These early observations are not late-time scientific acceptance.

The one-M0 mask-only checkpoint has the same chi-event counts and essentially
the same outer-shell Theta changes as agreement (ratio 0.572081 to legacy in
r=0.32..0.40). The early outer-shell reduction is therefore a masking effect,
not evidence of an additional penalty benefit. The penalty changes the inner
overlap norms; its late-time contribution remains to be assessed.

Segmented science histories use native output scheduling. A forced-final dump
advances the saved output counter, so the next nominal history sample can be
skipped on restart (observed 4.1014 M0 final row followed by 4.5007 M0).
Analysis uses the actual recorded times and does not assume a perfectly uniform
history cadence. This does not affect the fresh, matched performance windows.
The common-observer check archives the exact runtime history bytes used in each
comparison, avoiding hashes of a different version of a growing file.

For focused review, `review/current-feature.patch` in the validation directory
contains only changes relative to the captured modified working tree, excluding
pre-existing user changes. `build-review-patch.py` verifies application in a
disposable baseline subset and checks all reconstructed file hashes. This patch
is regenerated as documentation/analysis tools evolve; immutable qualified
solver archives and executable hashes remain the v4 provenance above.

## Unmasked full-overlap N12 experiment (2026-11-14)

The user authorized one new configuration, N12 only, through 12 M0, reusing the
previous N12 experiments for comparison. This is separate from the failed masked
configuration above; it does not establish the original 30 M0 acceptance goal.

`overlap_mode = agreement_full` leaves the activity policy at `legacy`, preserving
Cartesian evolution, KO, and ordinary halo donor selection. Only the agreement
sampler explicitly forces spherical donors for its spherical interpolation.
Use `agreement_kappa = 1` and `agreement_taper_width = 0.015`. This mode does not
use `cartesian_cutoff`, `agreement_inner`, or `agreement_outer`.

Samples cover both representations' evolved nodes in the usable overlap. A C2
quintic window turns on above Rs over the specified taper width. Six separable
C2 windows taper to a conservative Cartesian envelope, using the smallest
exterior evolved-node extension on each face. This avoids weight jumps when
selecting a donor at a refinement boundary. Outer extension tails can have zero
sampling weight while still receiving transpose corrections as evolved donors.
For the actual N12 grid the envelope is approximately +/-0.20694444 in x and
+/-0.21388889 in y and z, with 364368 samples extending to r=0.35766518.

The broad operator needs volume-balanced sampling weights. For interpolation
row A, coordinate volumes H, sample volume V, and taper T, set

    W = min(V*T/2, T / sum_i(A_i*A_i/H_i)).

The same nonnegative W is used in the discrepancy energy and both transpose
contributions. Each rank-one contribution's nonzero H-energy eigenvalue is
bounded by kappa*T; the total operator still uses its computed rate bound.
The naive uncapped broad weights failed preflight at the original timestep
(rate bound 472872.375); no science evolution used that version. The balanced
N12 bound is 99.52929908, with 41704 capped samples. The original native timestep
0.00046944203735617445 is retained. Cycle 8520 stops at 11.9989384748 M0.

Validation includes constant reproduction, random and grid-scale mismatch,
independent adjoint/representation checks, bitwise serial/MPI accepted states,
bitwise disabled-mode regression, restart/repartition equivalence and policy
rejection, and a complete native RK4 cycle on the 730-block N12 geometry. The
large-grid adjoint audit uses extended-precision diagnostic sums and a tolerance
scaled to the longest sequential transpose sum (27853 terms); the largest
observed absolute residual is approximately 1.46e-11. No solver arithmetic was
changed to pass this audit. Initial penalty-only weighted-RMS e-folding times
are about 1.97 M0 for a uniform inter-grid offset, 1.90 M0 for a smooth offset,
and 4.21 M0 for a Cartesian checkerboard. These are instantaneous probes, not
nonlinear stability guarantees.

The full and narrow agreement diagnostics use different regions and weights;
their raw RMS values must not be directly interpreted as an improvement. The
frozen executable's history registration is still gated on `activity.Masked()`,
so this unmasked experiment does not write a runtime `overlap.hst`. This output
omission does not disable the agreement stage task: its plan is constructed for
`agreement.full_overlap`, and every RK stage evaluates and applies that plan.
The read-only checkpoint observer evaluates an identical full-overlap diagnostic
on unmasked legacy and new states, without evolving or changing accepted data.
The original narrow observer remains available for comparing all three schemes.

Local receipts and progress are in
`../hybrid-overlap-validation/full-overlap-N12-12M-20261114/`.
Remote science data are isolated under
`hybrid-overlap-20261111/science/N12-full-overlap-20261114/`.

This scoped experiment completed successfully in jobs 9089250 and 9089261,
both exit 0. The final accepted state is cycle 8520 at 11.99893847482544 M0;
all recorded timesteps equal the original 0.00046944203735617445. No other new
science configuration or resolution was evolved. Read-only audit 9089344 also
exited 0: new and archived legacy states at cycle 7101 (10.000523721799558 M0),
plus the new final state, passed native diagnostics and independent Theta/volume
reduction. Accepted fields/chi records remained bitwise unchanged during the
audits, and the original checkpoint hashes were unchanged. Final cumulative
chi floor/nonpositive/nonfinite counts are 23664/23664/0, matching the audited
legacy 10 M0 state and the new 10 M0 state.

At the last common recorded history cycle, 11.7510730791013 M0, the new exterior
H/M/Theta/Z RMS ratios to legacy are 0.99997047/0.99996823/0.99997653/0.99999054.
The old masked-plus-agreement ratios are 1.11755932/1.22930161/1.02949357/1.02131186.
Time-series ratios use exact common recorded cycles, avoiding interpolation
artifacts from forced-final outputs. At the matched 10 M0 checkpoint, in
r=0.25..0.32, new H/M RMS ratios to legacy are 0.78194531/0.61666106; the old
masked ratios are 35.04167667/46.36328445. All four reported constraint norms
are lower than legacy in each of the four audited bands from r=0.18 to 0.40.
Farther out the changes are small and mixed.

With the identical full-overlap observer applied to new and legacy states at
10 M0, disagreement ratios for chi/metric/curvature/Lambda/gauge/D-minus-Lambda
are 0.42061137/0.70086570/0.84393855/0.84976134/0.95226041/0.56415229.
Thus this configuration reduces measured mismatch and local constraints at this
resolution and time. It does not establish long-time stability: the old masked
N12 run failed only after 18.75 M0. The test also does not establish convergence
or satisfy the original 30 M0/N16/performance acceptance goal.

The final 104-file receipt archive is `snapshot-20261114T182248Z.tar.gz`, SHA256
`58b38c26bbdd29773fd764d372de4cc9247a00aba001469eecea36a0a4698d41`.
The final checkpoint SHA256 is
`c2b47c493f8b12a6b416308c7b5ce89b9d1ba1f87752c89c7415624f8cbe201a`.
Plots, numeric assessments, and verification receipts are in the local
experiment's `report/` directory. The original goal remains paused; this
separately authorized N12-to-12-M0 experiment is complete.

## N12 full-overlap performance tuning (2026-11-14)

This separately authorized optimization uses the completed, unmasked N12
full-overlap run as its baseline. It does not benchmark against legacy or
establish N16 performance. The original production executables remain frozen.

Only `OverlapAgreementPlan::AddRHS` changes: each target's 26 independent fields
occupies 32 work slots, with unused slots returning immediately. On Aurora PVC
this avoids mixing adjacent targets with very different adjacency lengths in
one SIMD group. Every field keeps its original sequential accumulation order,
including the D/Lambda conversion. The forward interpolation, numerical
operator, sample set, weights, kappa, timestep, KO, halos, MPI exchanges, mesh
ownership and task graph are unchanged. No new arrays or infrastructure are
introduced. An initial prototype also padded forward interpolation; profiling
showed that made the forward kernel slower, so that part was removed.

On the busiest profiled rank, the transpose kernel takes 0.076186 rather than
0.109911 seconds per RK4 cycle (30.7% less), while forward interpolation remains
approximately 0.02285 seconds. Instrumented traces are attribution measurements,
not throughput measurements. The six alternating 300-step windows in job
9090197 show mean step time 0.511487 -> 0.477711 seconds and typical step time
0.499883 -> 0.466630 seconds. Their total process times do not improve: an
optimized window's final step and output took 53 seconds versus 22--25 seconds
in the baseline windows. This failed short-window end-to-end gate is retained in the
receipts; longer paired windows measure the total runtime separately.

The longer test, job 9090371 (two nodes, 24 ranks, debug-scaling), uses four
711-step windows in baseline/optimized/optimized/baseline order, all starting
from the same 12-M0 checkpoint. Each evolves 1.0013198657 M0. Native timestep,
physical output cadence, precision and affinity are matched; both executables
print cycle timing every step. Means below average the two windows per case.
Interior step statistics omit the first 20 cycles and final forced output,
while total process time includes MPI launch, startup and all output.

| Measurement | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| Typical step, seconds | 0.501489 | 0.467911 | 6.70% lower |
| Mean interior step, seconds | 0.535703 | 0.492120 | 8.14% lower |
| Total process seconds per evolved M0 | 456.019 | 488.641 | 7.15% higher |

Both pair orders reproduce the typical-step reduction (6.73% and 6.66%). Total
window times are 472.683/429.455/549.117/440.559 seconds. The second optimized
window's final step/output takes 169.076 seconds, versus 41.146--42.257 seconds
in the other three. Startup stays within 32.4--33.5 seconds and plan building
within 23.36--23.45 seconds. This locates the outlier in final output, but the
span includes evolution/diagnostics and is not a dedicated filesystem timer.
All four complete output checkpoint files are byte-identical, and all 24 ranks'
plan counts and numerical parameters match. The kernel/typical-step improvement
is repeatable; a whole-job runtime improvement is **not established**. Both
short and long end-to-end gates remain false. Two repeats are not a confidence
interval, and these data do not establish the original legacy-relative target.

All 12 local CTests pass. CPU baseline/candidate and restart/repartition checks
are bitwise equal across 24,704,784 stored scalars. GPU qualification includes
fresh and late-26-M0 eight-step comparisons, alternating timing windows and
profiled runs; all nine accepted-payload comparisons in job 9090197 pass
bitwise. Job 9090313 continues the original 26-M0 checkpoint to cycle 21301,
29.998754513173402 M0. Its entire 1,361,093,760-byte accepted payload, all 16
common constraint-history rows, and even the complete final checkpoint file
are identical to the archived original N12 result. This is a 26--30-M0
continuation check, not a new optimized 0--30-M0 evolution. Original checkpoint
hashes are unchanged.

The four longer-test accepted payloads also reproduce the archived original
13-M0 checkpoint bitwise, and their four common constraint-history rows match
all 12 columns exactly. `previous-results-audit.json` records these independent
checks against the prior science run, rather than only comparing optimized and
baseline benchmark executables with each other.

The baseline executable SHA256 is
`0b0173704b67b4e3b868c6ccc8a15adc7cab480a1e61065b517384bc6c222a09`;
the optimized executable is
`8c7aaaca9b7c488233fb8569aaa11d4fff622e96948dc6a8b41552acbc693ffc`.
The reproduced final checkpoint SHA256 is
`184979b130d7f79eb3b6a566070785967b13b4f232a35d8f19068df8450517c2`.
Build receipts record identical effective precise-FP/O2/SYCL/PVC flags; the
original compile command merely repeats `-ffp-contract=off` once more.

Local scripts, focused patches, captured baseline sources, per-file hashed
receipts and plots are in
`../hybrid-overlap-validation/full-overlap-performance-20261114/` relative to the
repository root. Remote data use the corresponding directory beneath
`/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111/`.
`source-audit.json` checks the captured baseline rather than Git HEAD, preserving
the substantial pre-existing modified and untracked implementation.

The final 86-file timing/reproduction receipt archive is
`receipts-20261115T013108Z.tar.gz`, SHA256
`9bf63768d2c80c49970f65c90ab008b2a54aac482d250a90f8f3d180133e02d0`.
Short qualification/profiling receipts are `receipts-20261115T010421Z.tar.gz`,
SHA256 `10c06291d49009ed6ff66976799899bc79280144d7e5894cca51cb31cf6dce17`.
Both archive and individual file hashes were verified after retrieval.
`report/performance.png` shows individual timing windows, and
`report/constraint-reproduction.png` overlays the identical late constraints.
`performance-change.patch` contains only this optimization and its documentation,
relative to the captured working tree; `transpose-only.patch` isolates the kernel.

### N12 performance follow-up: 20% legacy target (2026-11-15)

The retained follow-up changes only `overlap_agreement.cpp`. It reuses completed
Lambda corrections in D, stably groups recipients by adjacency length, pads the
discrepancy/scratch array to 32 field slots, and uses a four-iteration unroll
hint. A lookup-only volume hash cache reduces plan-building work. Every scalar
sum retains its original entry order; row and bound construction remain
ordered. Native task scheduling, communication, restart I/O, activity, halos,
KO, kappa=1, timestep, and output cadence are unchanged. No new donor-conversion
kernel or reordered accumulator is retained. The final source audit checks
876 captured files: this C++ file and this appended README section are the
only changes relative to the captured working tree. The focused patch is
verified by applying it to the capture and reconstructing both files exactly.

Job 9091365 compared the selected implementation and legacy on two Aurora
nodes, 24 ranks, with the qualified GPU binding and MPI cache settings. Four
opposite-order windows (legacy/full/full/legacy), using the frozen original
legacy executable and the selected full-overlap executable,
each evolved 1065 steps from cycle 8521 to 9586. Both modes wrote checkpoints
at cycles 9231 and 9586. The archived output counters were explicitly restored
after preparing the one-step full-overlap checkpoint; physical restart identity
was unchanged. These are approximately 12--13.5-M0 restart windows, including
startup, the regular 13-M0 checkpoint, and the forced final 13.5-M0 checkpoint.
They do not measure the total cost of a fresh optimized 0--30-M0 evolution;
startup amortization and the extra forced final output can affect the ratio.

| Window | Median step (s) | Mean logged step (s) | Total wall (s) | Wall s/M0 |
|---|---:|---:|---:|---:|
| timing-0-legacy | 0.357583 | 0.396980 | 462.703 | 308.496 |
| timing-1-candidate | 0.433011 | 0.462018 | 543.422 | 362.313 |
| timing-2-candidate | 0.433007 | 0.464727 | 545.960 | 364.005 |
| timing-3-legacy | 0.379832 | 0.399339 | 457.197 | 304.825 |

The combined total-wall ratio is 1.184239; the two
opposite-order pair ratios are 1.174450 and 1.194147.
Both pairs meet the 1.20 target in this allocation.
Total wall includes startup and outputs; mean logged step includes any
output stalls inside its sampled interval. These two repeats are not a
confidence interval or a guarantee across allocations. Legacy itself
ran at a different speed from the earlier experiment, so a change in
the cross-job ratio cannot all be attributed to this code change.
The median-step pair ratios are 1.210940 and
1.139999. Legacy median step time changed from
0.357583 to 0.379832 seconds while both
candidate medians were about 0.433011 seconds.
The small margin below the total-wall target and this variability
do not establish robust performance across allocations or a full
optimized 0--30-M0 production run.


The preceding sum-reuse implementation measured a full/legacy ratio of 1.299325
in job 9090714. Matched short pilots for that change reduce typical-step time by
about 5.6% against the prior transpose implementation. The host volume cache
reduces maximum plan-building time from about 23.4 to 16.4 seconds. These are
separate paired experiments, not a single combined speedup estimate. N16 has
not been performance-qualified with the selected executable and continues with
its prior frozen science executable.

All 3 early/repeat payload comparisons pass bitwise.
The full-overlap 13-M0 and legacy 13.5-M0 payloads also match their
archived references bitwise, and all four windows reproduce their six
common 12-column constraint-history rows exactly. The 24-rank agreement
plan counts, rates, and parameters are unchanged in both full-overlap
windows. Source science checkpoints remain unchanged.


Job 9091104 independently continued both the simpler sum-reuse version and the
selected cache version from the original 26-M0 checkpoint to cycle 21301,
29.998754513173402 M0. Both entire final checkpoint files are byte-identical to
the original: SHA256
`184979b130d7f79eb3b6a566070785967b13b4f232a35d8f19068df8450517c2`.
All 1,361,093,760 accepted-payload bytes and all 16 common constraint-history
rows match exactly. This is a 26--30-M0 continuation check, not a new optimized
0--30-M0 evolution. Original checkpoint hashes are unchanged. Local
baseline/candidate and restart/repartition checks are bitwise equal across
24,704,784 stored scalars. All six independent energy oracle cases, the
RK-stage check, and all 12 CTests pass; the final 203 compiled CPU objects match
the validated object manifest.

Selected-rank profiling is attribution only, not throughput. On rank 22 the
selected agreement discrepancy and transpose/application kernels account for
0.02286 and 0.04772 GPU seconds per step, versus 0.00614 and 0.00087 on rank 0.
Rank 22 has 184,577 local discrepancy rows and 40,053,209 interpolation entries,
while only 14,876 rows are locally owned. Across all 24 ranks, 364,368 unique samples produce 721,817 local
discrepancy rows (1.98 times as many) and 156,634,289 interpolation entries.
Remaining work includes duplicated interpolation and uneven agreement work
across ranks; these counts do not imply a corresponding total-runtime saving. Sharing that work or
rebalancing it is a larger next step; four profiled ranks do not establish an
exhaustive critical path. GPU durations and MPI waits overlap and must not be
added. The profiled fresh accepted payload is also bitwise identical.

Job 9091219 completed all four timing windows with raw total-wall overhead
16.33%, but its second legacy repeat failed strict bitwise reproduction
(maximum column-normalized field error 4.7892e-7; constraint-history relative
error 8.1655e-11). Its first legacy window and both selected full-overlap
windows match the archived histories exactly; the full-overlap 13-M0 payload
is exact, and the full-overlap repeats are identical. This failed comparison
is not used to certify the performance target. Its cause remains unresolved.
Job 9091365 uses the original frozen legacy executable to control this
uncertainty; that comparison does not resolve the intermittent disabled-mode
repeat failure in job 9091219. General legacy-mode regression qualification
therefore remains open; the measured comparison qualifies the full-overlap
path against the frozen original legacy binary. The verified 260-file archive for the failed
comparison and selected late continuation is `receipts-20261115T080906Z.tar.gz`,
SHA256 `bf3b84dadfbfad9dea38593597cd5644ba3ebf8766027ed4f0bae1a2abe5f5ae`.

Several other trials are excluded. Batched restart I/O produced a legacy regression
and an invalid bulk RHS (job 9090542); its cause was not established, and
`restart.cpp` was restored exactly. That trial also had unequal checkpoint
schedules, so its full/legacy timing is invalid. The donor-conversion plus
eight-way-unroll variant measured 26.8% overhead in short matched windows but
failed the 26--30-M0 check near 27.6 M0 (job 9091045). Its 27-M0 payload and all
constraint samples through 27.5 M0 were exact before the abrupt failure; the
cause remains unresolved. Its short-window speed is not an accepted result.
Neither failure affected the frozen N16 executable or original science files.

Four-accumulator and workgroup-size trials also remain excluded. An indexing
error in the first padded prototype was caught by the independent energy and
field checks. Its corrected version passed numerical checks but gave no
compute gain in either pair order (job 9091038). Workgroups of 128 and 256 threads passed all four fresh/late field checks
and five pilot repeat comparisons. Their paired mean and median gains were
only 0.5--0.64%, below the preset 2% retention threshold (job 9091219).
The selected implementation keeps the existing workgroup settings.

The selected executable SHA256 is
`de633cda33dd17fc312214a0d7e2253512bf4ea0f2ba19911b71fe2fb94a36e3`.
Evidence, scripts, captured baseline, final focused patch, source audit, and
plots are in
`../hybrid-overlap-validation/full-overlap-perf20-20261114/` relative to the
repository root. The remote directory has the same name beneath
`/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111/`.
Use `performance-change.patch` for the retained change; earlier candidate
patches are experimental artifacts and include rejected changes.
The verified 61-file late-reproduction/profile archive is
`receipts-20261115T070405Z.tar.gz`, SHA256
`502d726be74c2f5bc33ba04a01a027bd1de662d93170b622a73ce9f328acd84d`.
The final benchmark and selected late-reproduction archive contains
342 verified files: `receipts-20261115T084744Z.tar.gz`,
SHA256 `85b0f2c0c1954455a49f36c36ac197b8bc244801992bc15d53c6a53f260a59f3`.
Plots are `report/performance.png` and
`report/constraint-reproduction.png` (PDF copies included).
