# Reference-covariant FO-GH source repair

## Current diagnosis

The original stationary trumpet residual was a lower-order coordinate-source
cancellation problem, compounded by an interpolation provider that returned
independently interpolated value, first-derivative, and second-derivative
tables.  The repair uses one quintic Hermite polynomial per interval for the
regular primitives `alpha(y)`, `R(y)`, and `q(y)=beta^r/r`, with `y=ln(r/M)`.
The physical-looking `psi^2=R/r` and Cartesian shift derivatives are rebuilt
analytically from that one jet.

The production lower-order source is now the reference-frame covariant
connection-difference form: `Q=bar-nabla g`, `Delta=Gamma[g]-bar-Gamma`, the
frame Riemann term, quadratic Q and Delta sectors, GH damping, and the scalar
frame-connection correction.  The legacy coordinate source remains selectable
only as `ref_gh/source=coordinate_oracle`; `covariant` is the default.

## Hard algebra evidence

`high_precision_covariant_trumpet_oracle.json` evaluates the direct implicit
n=2 trumpet at 80 decimal digits, without the binary64 table, at all prescribed
radii from `1/8` through `1/128`.  It finds exact-arithmetic `Q=Delta=0` and
maximum scalar covariant-source and frame-Ricci residuals of `9.68e-73` and
`4.84e-73`.  In contrast, the legacy coordinate source intermediate grows from
`3.33e4` to `4.22e9` over the same radii while its arbitrary-precision residual
vanishes.  This establishes correct continuum algebra but severe coordinate
conditioning.

The independent random-state oracle covers 1000 flat Lorentzian samples and
64 curved diagonal/off-diagonal/shift/generic references.  Its largest
frame-vs-coordinate source mismatch is `1.11e-15`.

## Stationary t=0 result

The clean-build three-resolution covariant ladder is recorded in
`stationary_covariant_t0.tsv`.  The regular-state RHS is
`4.57e-12`, `1.58e-11`, and `9.29e-12` for `dx=1/16,1/24,1/32`: it is below the
`1e-10` target and has no inward resolution reversal.  The frame Ricci remains
bounded below `7e-10`; the coordinate Ricci grows inward and is retained only
as the predicted conditioning diagnostic.  All final field and native
constraint norms are zero for the exact constant regular state.

This establishes only the algebra and stationary-initial-data gates.  Flat
regressions now also pass: exact Minkowski remains exactly zero, linear-wave
L1 orders are 3.913 and 3.946, and robust-Minkowski perturbations decay at
8, 16, and 32 cells.  See `flat_regression_covariant.tsv`.

## Short multi-GPU stationary and restart gates

On Perlmutter allocation 57256611, the CUDA/MPI build ran one rank on each of
four distinct A100 GPUs.  The exact stationary trumpet stayed finite through
`t=0.1` and `t=1.0` at `dx=1/16,1/24,1/32`.  At `t=1`, the maximum regular
field errors were `2.52e-12`, `5.37e-12`, and `5.74e-12`; native constraint
Linf values were `6.60e-12`, `1.48e-11`, and `2.56e-11`.  The bad-state flag
remained zero in every history.  These are bounded, roundoff-scale stationary
results, not a long-time or puncture-stability claim.

The first restart attempt exposed a concrete plumbing defect: the Ref-GH
50-field state was absent from restart-file serialization and deserialization.
Commit `e162917c` repairs both paths.  A fresh four-rank checkpoint at
`t=0.1023608699` now restarts to `t=0.25` with the same history values as the
uninterrupted run, to the printed precision.  Pre-repair checkpoints contain
no Ref-GH state and are invalid; they were not used as evidence after the
repair.

Compact Perlmutter histories and rank-to-GPU evidence are under
`docs/fo_gh_artifacts/reference_covariant_repair_20260818/perlmutter/`.  Raw
restart files (about 205 MB each) are intentionally excluded.

## Paused stationary `t=20` ladder (not a gate pass)

The subsequent four-A100 campaign used the same CUDA-aware-MPI build and was
intentionally stopped at the user's request.  The `dx=1/16` and `dx=1/24`
stationary cases completed `t=20` with bad-state flag zero and effective CFL
0.05; their final square-root-of-integrated GH/reduction norms were
`4.19e-12`/`3.27e-10` and `8.68e-12`/`6.03e-10`, respectively.  The `dx=1/32`
case was cleanly cancelled after checkpoint `.00002`; its final written history
row is at `t=4.00184777`, also with bad-state flag zero and CFL 0.05, but has
GH/reduction norms `8.29e-11`/`1.81e-10`.

Thus all observed runs remained finite, but these common norms do not show a
resolution-improving trend.  The fine case has not reached `t=20`, so the
three-resolution stationary gate has **not** passed and no conclusion about
long-time puncture stability or formulation success is justified.  This is an
observation requiring source/formulation review, not a diagnosed cause: no
parameter, threshold, clipping, floor, or source term was changed to improve
it.  The compact histories, an exact table of the final rows, and reviewer
questions are in the artifact directory.

The time-dependent wormhole-to-trumpet reference remains out of scope until
the stationary gate is resolved.

## Curvature-conditioning diagnosis and correction

The pause review isolated the non-improving stationary histories to a
table-faithful binary64 reference-curvature error, not to nonzero `Q`,
`Delta`, damping, or a source-sector ambiguity.  At the closest off-grid cells
of the 64/96/128 ladder, the old generic coordinate-2-jet/Cartan construction
produced frame-Ricci Linf values `6.54e-11`, `2.54e-10`, and `6.88e-10`.
For the exact regular state its curvature sector was the complete source
residual; the other four source sectors vanish.

The stationary reference is exactly Schwarzschild, so its Riemann tensor in
the Eulerian Cartesian orthonormal frame is known analytically.  Commit
`b5594291` supplies that full vacuum Weyl tensor from the interpolated areal
radius, with electric part
`E_IJ = M/R^3 (delta_IJ - 3 n_I n_J)` and zero magnetic part.  This is neither
background-source subtraction nor projection of an approximate curvature:
the full analytic Riemann remains available to the covariant source for
perturbed states.  The generic Cartan spin and spin-derivative construction is
retained for the scalar-frame correction.

The new table-faithful audit records a roundoff-scale analytic Ricci trace and
source at all three closest cells (source Linf at most `5.68e-16`), while
preserving the independently measured raw-curvature discrepancy.  A clean
serial C++ 64^3 initialization also gives RHS Linf `1.17e-16` and frame-Ricci
Linf `3.33e-16`; its coordinate-Ricci diagnostic remains `3.76e-10`, as
expected from the deliberately cancellation-prone coordinate calculation.
The 96^3/128^3 executable ladder and all post-change flat/long-time gates are
still required; this correction does not retroactively pass them.

## Post-correction flat regressions

The correction is confined to the stationary-trumpet reference Riemann
provider, but the required clean release reruns also preserve exact Minkowski
(zero error), fourth-order linear-wave convergence (L1 orders `3.9134` and
`3.9456`), and robust-Minkowski decay through `t=0.2` at 8, 16, and 32 cells.
The exact rows and commands are summarized in
`flat_regression_analytic_vacuum.tsv`.  These flat results pass, but they do
not replace the new long-time stationary ladder.

## Post-curvature executable stationary t=0 ladder

The pending executable 96^3/128^3 checks are now complete on the exact source
state containing the analytic stationary vacuum Riemann.  The compact table
`stationary_analytic_vacuum_t0.tsv` records clean `64^3`, `96^3`, and `128^3`
initializations.  Their regular-state RHS Linf values are respectively
`1.17e-16`, `1.59e-16`, and `1.46e-16`; all final regular-field and native
constraint Linf values are zero.  Frame-Ricci Linf stays in
`[3.33e-16, 5.00e-16]`, with no inward growth.  Coordinate-Ricci Linf grows
from `3.76e-10` to `5.33e-08`, exactly as expected of the retained
coordinate-conditioning diagnostic rather than of the frame-native source.

The 64^3 and 96^3 rows use a clean Release/Kokkos-Serial executable; the 128^3
row uses a separately built Release/Kokkos-OpenMP executable with 32 host
threads.  Exact executable hashes, commands, and complete terminal records
are retained beside the table in `stationary_analytic_vacuum_t0_provenance.md`.
This completes the post-curvature **t=0** stationary gate only.  It does not
qualify t=0.1, t=1, t=20, t=100, or the time-dependent transition after the
source/curvature correction.

## Source-sector history diagnostics

Commit `9c2e3579` makes the stationary source decomposition visible in the
standard Ref-GH history rather than relying on a separate post-processing
oracle.  In addition to the six native constraint reductions, the history now
records Linf values for `Q`, `Delta`, frame Ricci, coordinate Ricci, and the
five covariant-source sectors: curvature, `Q Q`, `Delta Delta`, damping, and
the scalar-frame correction.  The native stationary acceptance check continues
to use only the six native constraints; these source diagnostics cannot turn a
conditioning indicator into a reported GH-constraint failure.

The compact clean-release `16^3` one-cycle smoke history is retained as
`fo_gh_artifacts/reference_covariant_repair_20260818/stationary_history_diagnostics_smoke_n16.hst`.
At initialization it records zero `Q` and `Delta`, frame-Ricci Linf
`3.33e-16`, coordinate-Ricci Linf `1.43e-11`, curvature-sector Linf
`6.66e-16`, and zero remaining exact-state source sectors.  The short smoke
only validates the wiring and the expected exact-state decomposition.  It is
not a stationary-convergence or long-time-stability result.

## Delta-upper construction repair and replacement local evidence

A subsequent compiled-source audit found an additional generic-state defect:
the source formed `Delta_lower[A][B][C]` and immediately raised its first
index. That contraction requires `Delta_lower[D][B][C]` for every `D`, so it
could read rows not yet initialized by the enclosing `A` loop. The repair is
two passes: fill all lower-index connection differences, then form all raised
ones. It changes no equation or threshold.

The new device-side unit test uses 1,000 deterministic nonzero-Delta flat
Lorentzian metric jets and compares the production covariant source with the
retained coordinate partial-wave oracle and transform. It failed before the
repair at `1.14718e-4` (and at `2.25991e-5` even with zero shift); the repaired
OpenMP executable gives `6.93889e-17`. The corresponding exact-Minkowski,
fourth-order linear-wave, and robust-Minkowski reruns are all recorded in
`fo_gh_artifacts/reference_covariant_repair_20260818/delta_upper_source_repair/`.
The wave L1 orders are 3.9134 and 3.9456, and robust perturbations decay at
all three tested resolutions through `t=0.2`.

The initial compiled test was deliberately extended beyond flat space after a
review identified that the existing generic nonflat oracle was Python-only.
The repaired C++ regression now supplies 128 deterministic nonflat references
with nonzero shift, spin connection, and curvature, together with nonzero
regular physical perturbations. It reconstructs the coordinate metric jet,
runs the retained C++ coordinate oracle and scalar transform, and compares it
directly to the production covariant source. The maximum error is
`3.33e-16`. This closes the compiled nonflat source-oracle evidence gap; the
compact terminal record and executable hash are in `delta_upper_source_repair/`.

The post-oracle principal check retains a rank-50 trumpet basis with zero
imaginary characteristic part, unit basis condition to printed precision, and
zero trumpet symmetrizer residual. The generic reference-frame audit retains a
positive minimum symmetrizer eigenvalue `0.1251`; the retained JSON records
also show standard-source agreement `2.78e-17` and independent nonflat
covariant-source agreement `1.11e-15`. These are algebra/principal checks,
not stationary-evolution evidence.

## Aurora frozen stationary t=1 execution blocker

The current corrected source commit
`9d55f7b411171aaf2d7e0dc6c3d9be2bfd7ffe0a` was rebuilt in an isolated
Aurora SYCL/PVC directory and launched in PBS debug jobs 8766897 and 8766926
on distinct nodes. Each one-rank mapping selected one Intel Data Center GPU
Max 1550 tile; application startup completed with stationary initial RHS Linf
`8.55241e-17` and an identical clean t=0 Ref-GH history. At the first task-list
execution status (`cycle=0`, `time=0`), each Level Zero runtime raised a GPU
`NotPresent` write fault and aborted the rank (exit status 134). No
positive-time history row was produced.

Accordingly, the corrected Aurora three-resolution `t=1` stationary gate was
**not assessed**: only the 64-cell case started in each retry; the 96/128-cell
cases and all `t=20` cases were deliberately not launched. This is not
numerical evidence for a formulation defect, constraint growth, or lack of
stationary stability; it is a reproducible first-step device/runtime execution
blocker. No solver, source, parameter, or threshold was changed in response.
The complete compact
provenance, scheduler records, mapping, histories, and exact failure log are
in `fo_gh_artifacts/reference_covariant_repair_20260818/aurora_stationary_frozen/`.

The earlier post-curvature stationary `t=0` rows are superseded for source
provenance, even though the exact state has `Q=Delta=0` and therefore did not
excite this bug. The corrected OpenMP 64/96/128 replacement ladder has RHS
Linf `1.17e-16`, `1.59e-16`, and `1.46e-16` and frame-Ricci Linf
`3.33e-16`, `4.44e-16`, and `5.00e-16`, with zero-time final field and native
constraint norms. This is still a `t=0` gate only. No existing stationary
evolution or puncture conclusion is requalified by this correction.

### Frozen-campaign phase localization

Frozen diagnostic job 8768490 independently reproduced the failure on Aurora
node `x4311c0s0b0n0`, one MPI rank, one PVC tile (`ZE_AFFINITY_MASK=0.0`). It
used a one-block physical `[-2,2]^3` `64^3` case, excluding an internal-block
or SMR-interface precondition. The initial RHS and all initial RHS phase fences
completed at roundoff. On the first evolved stage, receive initialization,
`CopyU`, and RHS zeroing completed; the next primary-RHS kernel produced the
same Level-Zero `NotPresent` write fault (exit 134). This is an execution
localization observation, not a root-cause or formulation diagnosis. The
frozen campaign made no solver/source change in response and stopped before
any positive-time sample. See
`fo_gh_artifacts/reference_covariant_repair_20260818/aurora_stationary_frozen/attempt4_*`.

### Authorized equation-preserving portability attempt

The later authorized portability investigation changed device execution and
diagnostic plumbing without changing the equations or numerical algorithm.
Reference-geometry aggregate returns were replaced with caller-owned outputs,
the large RHS was split into smaller kernels, Psi-only consumers use a compact
kinematics object, and history MPI reductions use valid separately batched
sum/max buffers.  A condition-number diagnostic was cached outside history,
and the final history implementation uses mature-AthenaK-style combined
built-in scalar reducers rather than a custom array maximum.

These changes compile and complete the focused CPU cycle.  The final native
and common-ADM CPU histories are byte-identical to the pre-refactor histories.
They do **not** fix the Aurora gate: PBS job 8769672 completes all history
reductions and fences, then the next evolved `CalcRHS zero` write triggers the
same Level Zero `NotPresent` page fault and exits 134.  Consequently the PVC
bug remains open and the three-resolution `t=1` ladder was not launched.  The
full evidence and remote-debugging handoff are in
`fo_gh_artifacts/reference_covariant_repair_20260818/aurora_portability_20260820/`
and `ref_gh_aurora_portability_handoff_prompt.md`.
