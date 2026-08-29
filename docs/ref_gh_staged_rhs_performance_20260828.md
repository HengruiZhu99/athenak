# Ref-GH staged RHS performance qualification

## Scope and exact base

This branch starts from the accepted compact analytic radial-q checkpoint
`a09caf707f88d9fb6ca71f9abf62c9302fde3bac` on
`codex/ref-gh-analytic-device-view-performance-20260828`.  The parent worktree
was clean and local `HEAD`, its configured upstream, and the GitHub branch all
matched that commit before this separate worktree and branch were created.

Branch under development:
`codex/ref-gh-staged-rhs-performance-20260828`.

The controlling inherited baseline is the retained ordinary-RangePolicy
production path:

- 12 static plus 8 stage analytic radial-q Reals per ghosted cell;
- no generic provider/workspace/evolution/diagnostic allocation in analytic
  mode;
- `T_RefGH,RHS/T_Z4c,RHS = 10.524508196915962` from Aurora job 8789684;
- a handoff estimate of about 503 spilled Reals per work item, superseded below
  by the fresh exact-current-build compiler report.

The rejected team-per-cell and non-inlined-team variants are not part of this
work.  Equations, q dependence, gauge driver, gamma0/gamma2, Phi ordering,
finite differences, RK4, and dissipation are frozen.

## Phase 1 local baseline

Fresh local provenance:

- source: `a09caf707f88d9fb6ca71f9abf62c9302fde3bac`;
- Kokkos: `6739bc623081648af9e752b616d9671527922cbf`, version 4.7.2;
- compiler: GCC/G++ 13.3.0;
- build: Release, Kokkos Serial, MPI/OpenMP/SYCL disabled;
- oracle executable SHA-256:
  `0915275de6cca898c51f8a163010ef819f688d607bc0d6eaa20f5231be2c9415`.

Two independent SymPy 1.14.0 regenerations took 321.99 and 323.00 seconds and
were byte-identical to one another and to the committed generated headers:

- geometry: `6b4b3976f8cfc62924aabaf6cb960cd79a09677e8a7dd59441b5d3a4f90184e7`;
- gauge: `f6ec5bb54d10c490f65f78d9d9a9e7df07c7f080eef4d6edd3fff55d45053908`;
- source: `b5dff1d44bd1e0ed53070fdd5fa4b30d0c4c176a13c2d0924dd6b0ad51012f21`;
- generator: `2c19efa4affd02c0d97ebbe26ea51a3f6bbb23467f3cb3c3e07c134320e26cc3`.

The fresh default-on source-unit executable passed without changed tolerances:

- original coefficient oracle: 216 samples, `8.88178e-15`;
- expanded coefficient oracle: 2160 samples, `1.48837e-13`;
- generated geometry: 2376 samples, `2.33147e-15`;
- moving gauge/mixed jet: 2160 samples, maximum reported motion error
  `1.24829e-14`, including finite `dtTheta`;
- compact boundary projection: 2160 samples, metric `4.56474e-14`, gauge
  `9.21858e-16`, subtracted gauge `7.11991e-16`;
- all-61 RHS: 4320 compatible and standard comparisons, `2.84217e-14`;
- generic production-cache oracle: `1.63758e-14`.

Fresh compatible and standard one-cycle runs completed for both analytic and
generic-cache backends.  Analytic allocation was exactly 1,327,104 static bytes
plus 884,736 stage bytes and zero generic bytes.  Compatible evolved rows agree
at conditioned Linf `4.32986979603811051e-15`.  Initial common-ADM shell
diagnostics retain the already documented conditioning; the two reference
Ricci columns are backend-specific and excluded from analytic/generic history
comparisons.  The all-61 source-unit oracle is the hard compatible/standard
pointwise gate.

Compact logs and histories are under
`artifacts/ref_gh_staged_rhs_performance_20260828/baseline_local/`; restart
files remain only in the temporary run directory and are not committed.

## Phase 1 Aurora baseline

A fresh remote source clone is staged at:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260828_baseline_a09caf_phase1`

An initial `qsub` command printed a per-user queued-job-limit diagnostic, but
inspection found that it had nevertheless created job 8790322.  No other job
or directory was touched.  Both retained jobs ran in Aurora's `debug` queue,
charged to `CompactBinaryMerger`:

- job 8790322, node `x4610c7s6b0n0`, passed the bounded dynamic-q one/eight-rank
  full-output evolved gate;
- all eight ranks mapped to distinct PVC tiles (`0.0` through `3.1`);
- one/eight-rank conditioned Linf difference was
  `3.88980825583101983e-14`, below the unchanged `5e-12` tolerance;
- executable SHA-256 was
  `5d56f36afa384694fa55704affd08e478c637b9e6425113df79fbc25fd9a166f`;
- job 8790348, node `x4610c7s2b0n0`, completed the matched warmed 64-cubed
  dynamic/static Ref-GH and Z4c measurement.

The fresh compiler report is authoritative for this exact build.  For finite
difference orders 4 and 3 the primary analytic scalar-source/Pi kernel uses
SIMD16, 256 registers and reports 1,279 spilled Reals per work item; order 2
reports 1,282.  This supersedes the inherited approximate 503-Real estimate,
which was not from this exact retained build.  The generic gauge-driver kernel
also reports 386 spilled Reals, but it is not active in the matched benchmark.

The warmed, unchanged-input aggregate is:

| Metric | Value |
|---|---:|
| dynamic-q Ref-GH complete stage | 0.08696205 s |
| static Ref-GH complete stage | 0.08636561 s |
| Z4c complete stage | 0.009406503 s |
| dynamic/static complete-stage ratio | 1.006906 |
| Ref-GH/Z4c complete-stage ratio | 9.244886 |
| Ref-GH/Z4c RHS ratio, no dissipation | 10.902828 |
| Ref-GH/Z4c RHS ratio, including dissipation | 10.422812 |
| q-control fraction of Ref-GH stage | 0.6073% |
| analytic-reference fraction of Ref-GH stage | 0.0475% |
| main RHS fraction of Ref-GH stage | 88.8541% |
| physical-boundary fraction of Ref-GH stage | 7.7391% |

Thus all compact-controller/reference overhead targets pass, while the required
RHS ratio of at most 2 does not.  Raw throughput-only ratios in the job's
`performance_summary.tsv` do not subtract the 20-cycle warmup and are not used
for the controlling comparison.  The deterministic analyzer and exact compiler
extractor are committed under `scripts/ref_gh/`; their JSON/TSV results and all
compact logs are under
`artifacts/ref_gh_staged_rhs_performance_20260828/baseline_aurora/`.

## Qualification and claim boundary

Current state:

- exact base frozen: yes;
- deterministic generation: fresh pass;
- local coefficient/geometry/mixed-gauge/boundary/all-61 gates: fresh pass;
- local analytic allocation: fresh pass;
- fresh Aurora one/eight-tile baseline: passed;
- fresh Aurora matched benchmark: complete, target failed at 10.902828 RHS;
- exact primary RHS compiler pressure: 256 registers, 1,279--1,282 spilled
  Reals per work item;
- staged hot-reference or physical scratch implementation: not started;
- Ref-GH performance target: not established.

No convergence, trumpet stability, production readiness, or performance
improvement is claimed from this local checkpoint.

## Phase 2/3 hot-reference discriminator

Candidate 1 added exactly 141 analytic hot-reference Reals per ghosted cell:
24 antisymmetric spin coefficients, 96 symmetry-packed spin derivatives, and
21 bivector-symmetric Riemann components.  The generated direct fill consumes
only the accepted 12-static/8-stage radial representation.  Analytic mode
allocated 1,327,104 static bytes, 884,736 stage bytes, and 15,593,472 hot bytes
for the local test mesh, with zero generic-cache bytes.  The generic 1171-Real
path remained an oracle only.

Fresh local gates passed at `ba563692e7e0c541eae35c94e35b33ee29a1336e`:

- deterministic SymPy regeneration, twice, byte-identical;
- 216 and 2160 coefficient samples;
- 2376 generated-geometry samples;
- 2160 moving mixed-jet gauge/dtTheta samples;
- 2160 compact-boundary samples;
- 4320 compatible/standard all-61 comparisons, maximum `2.84217e-14`;
- analytic/generic one-cycle histories, conditioned Linf below `3e-13`.

An additional exploratory comparison of individual generated hot coefficients
showed near-puncture absolute differences up to about `3.3e-6` for very large
coefficients.  It was not a controlling oracle and was removed rather than
weakening the accepted binary64-conditioned end-to-end gates.  The consumed
covariant contractions retain their generated-source and all-61 oracle.  This
is a documented limitation of individual-coefficient interpretation, not a
claim that each stored coefficient is independently pointwise qualified near
the puncture.

Aurora exposed two pre-existing portability pressure points while qualifying
the candidate.  Jobs 8790472, 8790490, 8790496, and 8790507 preserve the exact
failures.  Equation-preserving corrections replaced the custom q reducer with
one compact nine-moment atomic device accumulation plus one MPI collective and
computed the analytic boundary metric one symmetric component at a time.
After fresh local all-61 and evolved checks, exact commit
`e02e8ced53a66ae45de5615ae8943081c217f8ac` passed job 8790518:

- one/eight full-output dynamic-q evolved cycle: pass;
- all eight ranks on distinct PVC tiles: pass;
- finite native GH/reduction/curl diagnostics from the initial row: pass;
- one/eight conditioned Linf: `1.36191095905485950e-14 < 5e-12`.

Matched warmed job 8790530 then rejected the production loop-source candidate:

| Metric | Baseline | Hot loop candidate |
|---|---:|---:|
| complete Ref-GH/Z4c stage | 9.244886 | 14.956614 |
| RHS ratio, no dissipation | 10.902828 | 17.269546 |
| RHS ratio, with dissipation | 10.422812 | 16.401222 |
| main RHS seconds/stage | 0.0772693 | 0.121589 |
| primary-kernel spill Reals | 1279 | 895 |

The q controller (`0.0634%` of stage), reference-stage update (`0.0300%`), and
dynamic/static ratio (`1.07595`) meet their targets.  However, lower compiler-
reported spill did not offset the loop contraction and hot-cache traffic: the
primary source kernel increased from `0.072283` to `0.116564` seconds/call.
The production loop-source discriminator is rejected.  This triggers the
flat-RangePolicy staged physical/source/Pi phases; it does not establish the
performance target.

## Phases 4--8: staged covariant candidate

Candidate 2 implemented the specified ordinary flat-kernel decomposition with
exactly 32 physical, 64 packed covariant-preparation, and 10 scalar-source
Reals per active cell, for 106 transient Reals per cell.  The persistent
analytic footprint was 12 static plus 8 stage plus 141 hot-reference Reals per
ghosted cell; generic-cache allocation remained zero.  Compatible and standard
Phi ordering remained supported and unchanged.

At source commit `9f64fcf6113ac9efd31b2957d1677d8203f5ebca`, all local
gates passed without tolerance changes.  The coefficient, expanded radial,
geometry, moving mixed-jet/dtTheta, compact-boundary, and all-61 errors were,
respectively, `8.88178e-15`, `1.48837e-13`, `2.33147e-15`,
`1.24829e-14`, `4.56474e-14`, and `2.84217e-14`.  Compatible and
standard one-cycle analytic/generic histories each agreed at conditioned Linf
`2.9721190786258234e-13 < 5e-12`.

Job 8790555 passed the one/eight-rank full-output PVC gate on eight distinct
tiles.  Its one/eight conditioned Linf was
`1.56518125145110397e-14 < 5e-12`.  Job 8790573 then supplied the
controlling warmed benchmark and synchronized kernel profile:

| Metric | Candidate 2 |
|---|---:|
| complete Ref-GH stage | 0.10803409375 s |
| complete-stage ratio to Z4c | 11.4388156 |
| main RHS without dissipation | 0.0909013228 s |
| RHS ratio to Z4c | 12.7850728 |
| scalar source | 0.0584220886 s, 54.08% of stage |
| physical geometry and gauge | 0.0277257232 s, 25.66% of stage |
| packed covariant preparation | 0.0008895847 s, 0.82% of stage |
| Pi principal update | 0.0025744840 s, 2.38% of stage |
| q control | 0.0836% of stage |
| analytic reference update | 7.26% of stage |
| dynamic/static complete stage | 1.0886525 |

The component-parallel FD4 covariant source reported only 7 spilled Reals,
while the preparation kernel reported 903.  The low-spill source nevertheless
dominated runtime.  This is direct evidence that the retained covariant loop
contractions are primarily arithmetic/traffic expensive after componentization,
not evidence that all Ref-GH kernels are spill free.  No hardware HBM-byte
counter was collected; explicit allocation sizes and synchronized kernel times
are recorded, so no stronger HBM-bandwidth claim is made.

Candidate 2 passed all correctness and PVC gates but was slower than the frozen
baseline and missed the 2x RHS target by a wide margin.  It is rejected as the
production implementation.  Its profile is the measured bottleneck evidence
that authorized the single Phase-13 coordinate-source discriminator.

## Phase 13: coordinate-source discriminator

Candidate 3 replaced only the scalar-wave-source representation with the
already-qualified standard-coordinate GH source followed by the analytic frame
transformation.  It used 32 physical, 84 coordinate/transform, and 10 partial-
source Reals per active cell, totaling 126 and remaining below the hard
128-Real cap.  The all-61 gate continued to compare the complete compatible and
standard RHS against the independent generic cache.

At `627b0d447c39c67e364dda36ca2fcbc1e83e30f8`, the same deterministic
local gates passed.  The all-61 error was `2.84217e-14`; the new pointwise
operation-conditioned coordinate/covariant source error was `4.83589e-16`;
and both evolved-ordering comparisons were
`2.9721190786258234e-13 < 5e-12`.  A fresh Debug Kokkos Serial build with
view bounds checking also completed the full four-stage one-cycle workload,
including every physical-boundary and diagnostic fence.

The required Aurora gate failed reproducibly:

- job 8790590, node `x4504c1s3b0n0`, exit 134;
- source-identical job 8790608, node `x4514c1s0b0n0`, exit 134.

In both cases initial full output passed.  The first evolved CalcRHS, RK update,
restriction, communication, and prolongation fences passed, after which the
PVC runtime reported a level-2 `NotPresent` write page fault before the evolved
physical-boundary fence completed.  The eight-rank evolved workload was never
reached.  The source-identical retry on a different node makes this a
reproducible candidate failure, while the host bounds check only establishes
that no host-visible Kokkos view bound violation was found.

The coordinate frame-transform compiler kernel reported 10 spilled Reals for
FD4 and 28 for FD3.  Other four-index lambda warnings cannot be assigned to a
named source kernel from the compiler text alone and are recorded as
unresolved.  Under the controlling rule, no benchmark or profiler followed the
failed PVC gate.  Candidate 3 is rejected.

## Final decision and retained production tree

The structural candidates are preserved in branch history and compact
artifacts, but the final production `src/` tree is byte-for-byte identical to
the fully qualified starting tree at
`a09caf707f88d9fb6ca71f9abf62c9302fde3bac`.  A fresh local rebuild after
restoration passed the unchanged source-unit suite, including the 4320-sample
all-61 oracle at `2.84217e-14`.  The already preserved exact-base Aurora jobs
8790322 and 8790348 remain the production qualification and benchmark evidence;
the rejected variants do not supersede them.

Final classification:

`REF-GH PERFORMANCE TARGET NOT MET — RESIDUAL COST CHARACTERIZED`

The best fully qualified production result remains:

- Ref-GH/Z4c complete-stage ratio: `9.244886`;
- Ref-GH/Z4c RHS ratio without dissipation: `10.902828`;
- main RHS: 88.85% of the Ref-GH stage;
- baseline primary-kernel compiler warning: 1,279--1,282 spilled Reals.

The staged covariant experiment reduced its scalar-source spill to 7 Reals but
remained slower because that kernel alone consumed 54.08% of the complete
stage, with physical geometry/gauge consuming another 25.66%.  The final
coordinate alternative could not be performance-qualified because its evolved
PVC gate failed reproducibly.  Therefore neither the 2x RHS target nor the
approximately 2.5x stage target is established, and no long single-hole,
closed-loop recovery, or later scientific campaign is authorized by this
performance checkpoint.
