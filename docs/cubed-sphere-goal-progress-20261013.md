# Active hyperboloidal validation goal

## Corrected GPU-tile scaling and continuation to t=1 — 2026-10-23

The requested 1/2/3/6/12-rank tile benchmark passed; twelve tiles were fastest
for the fourteen-block benchmark. The unchanged seven-block saved pair separately
selected seven tiles, passed bitwise restart comparisons against one rank, and
both trajectories reached t=1 at dt=1e-4. All runs used at most one Aurora debug
node at a time. The original solver, checkpoints, references and failed criteria
are preserved; Goal 2 remains inactive.

The observed interface signal is bounded and oscillatory through t=1. Global
connection RMS remains higher than at t=.2, while final H/M/Z/Theta RMS is lower;
this is not complete error saturation, convergence, or long-time stability proof.
See [Goal 1's current record](goals/01-minkowski-foundations.md), the
[scaling review](validation/goal1-minkowski-20261018/native-mpi-tiles-v33/result-review.json),
and the [final continuation review](validation/goal1-minkowski-20261018/native-mpi-continuation-v34/result-review.json).
The dated records below remain as historical evidence.

## Completed saved-pair continuation to t=.2 — 2026-10-22

Aurora job **8954658** finished F/0 on one node and one physical GPU in debug
under CompactBinaryMerger (reported walltime 21m04s). Both saved v28 trajectories
reached .2 at dt=1e-4, with only endpoint landing reductions, finite diagnostics
and positive lapse, chi and metric eigenvalues. All earlier evidence and eight
new checkpoint/snapshot segments are preserved; Goal 2 remains inactive.

The normalized tangential Lambda maxima peak at the sampled t=.125 endpoints
(.020701 coarse / .006939 fine), then decrease at .15, .175 and .2, ending
.015977 / .005572. The peak turns over rather than sustaining acceleration.
However, whole-domain Lambda RMS rises about 20% / 36% from .1 to .2 and the
wider-interface RMS also rises. Both growth rates slow in the final interval;
Z RMS still grows after .125. This supports transient peak weakening and
spreading, not complete error saturation or a long-stability proof. The coarse
.01 screen failure remains. Convergence and accuracy stay deferred; Goal 1
is incomplete. No further evolution was launched.

See the [v29 result](validation/goal1-minkowski-20261018/native-interface-continuation-v29/README.md), [detailed evidence](validation/goal1-minkowski-20261018/native-interface-continuation-v29/result-review.json),
and [final checkpoint provenance](validation/goal1-minkowski-20261018/native-interface-continuation-v29/execution-20261022.json).
The v28 findings below are retained as earlier evidence.

## Current Goal 1 priority — 2026-10-22

The user authorized the bounded interface refinement experiment with dt=1e-4,
prioritizing stability and deferring convergence/accuracy acceptance. Corrected
GPU baseline and active-state/restart checks are complete; full stored-ghost
CPU/GPU failures remain recorded. The allocated ghost audit supports a narrower
mapped-only diagnostic admission. All substantial work remains on Aurora, one
node and one GPU at a time in debug under CompactBinaryMerger.

See the [v28 protocol](validation/goal1-minkowski-20261018/native-interface-stability-v28/protocol.json),
[admission and retained limitations](validation/goal1-minkowski-20261018/native-interface-stability-v28/mapped-diagnostic-admission.json),
and [live execution record](validation/goal1-minkowski-20261018/native-interface-stability-v28/execution-20261022.json).
The first coarse segment reached .05; a postprocessing check incorrectly expected
a mass-profile file in this mass=0 case. Its output and failure are preserved,
and the repaired continuation reuses the checkpoint. No main-root replacement,
accuracy acceptance, Goal 1 completion or Goal 2 activation follows from this work.

Current CFL follow-up: [v14 compactification audit](validation/goal1-minkowski-20261018/native-timestep-audit-v14/README.md) verifies that conformal
fields and the inverse patch map already include coordinate-speed changes. The
private candidate makes gauge-flag precedence explicit and records individual runtime
limits; a 20-step implementation fork is bitwise equal. Keep cap 1e-4 pending
separate discrete/source stability experiments.

**Completed bounded result:** both grids reached t=.1 on Aurora at dt=1e-4,
with finite diagnostics and positive lapse, chi and metric eigenvalues. At .1,
normalized tangential Lambda maxima are .017692 (coarse) and .006804 (fine);
H RMS is 2.2306e-4 and 5.6186e-5. The coarse .01 screen failure is retained.
The localized maximum still grows with time. A longer bounded stability window
is the next priority; no convergence/accuracy acceptance, long-stability proof,
Goal 1 completion or Goal 2 activation is claimed. See the
[result and preserved checkpoint provenance](validation/goal1-minkowski-20261018/native-interface-stability-v28/result-review.json).

## Previous Goal 1 roadmap — 2026-10-20

**Priority revision:** hold long-duration continuation after the active bounded
segment; optimize performance and diagnose timestep limits first. The latest user
revision moves all substantial CPU/GPU computation, analysis, builds and benchmarks
to Aurora to avoid local battery drain. Use one-node debug allocations under
CompactBinaryMerger; establish one-GPU performance before node scaling. Local
activity is limited to lightweight coordination and document edits.
[Plan v30](validation/goal1-minkowski-20261018/native-performance-v9/README.md)
targets demonstrated CPU/GPU cost within 3× a comparable Cartesian evolution.
Keep timestep forks separate from implementation controls and retain all failures.
This supersedes the earlier execution priority and local-only restriction below.

[Reconciled v5](validation/goal1-minkowski-20261018/native-literature-operators-v5/README.md) and Plan v26 prioritize long stable native 3D stationary/gauge evolutions from
the existing .004 checkpoints, through the .6 and 1.67 reference crossings and
beyond 2. Accuracy/convergence refinement and acceptance matrices are deferred
unless needed for instability diagnosis. All failures remain; survival is not
success. Actual-scri compatibility and full characteristics remain required.
Goal 2 stays inactive. Plan v26’s local-only resource policy is historical and
superseded by the current Aurora policy above.

The [local continuation](validation/goal1-minkowski-20261018/native-literature-stability-v6/README.md)
has reached t=.05 for both cases with strictly compatible restarts. Short screens
pass, but gauge constraint growth and interface maxima remain under observation.
The reference crossing times have not been reached.

Neither stable substantial angular Minkowski evolution nor a resolved
wormhole-to-trumpet transition is accepted. The goal remains active.
This is a working progress record; the earlier frozen evidence capsules
are unchanged. These experiments include uncommitted changes beyond
`24616eb8d52060414f234192548abbfc7758efb0`.

The quantitative [acceptance gates](validation/cubed-sphere-balanced-20261013/goal-acceptance-20261013.json)
cover native baseline survival, global constraint convergence, substantial
angular content, and late-time horizon/trumpet diagnostics. The late-time
targets are at least `t=6` for the angular pulse (extend the interval if it
has not settled) and `50M` for the puncture, including a final `10M`
stationarity interval. These are prospective criteria, not completed results.

The existing baseline must not count as the substantial angular target.
An independent [initial angular-content calculation](validation/cubed-sphere-balanced-20261013/pulse-angular-content/review.json)
finds only **0.0826676%** of its squared lapse perturbation in the angular
part, versus the required 50%. The spherical mean of its angular multiplier
is 1 and its variance is `.04*r^2/3 + .09*r^4/15`. Radial Gauss–Legendre
quadrature with 64, 128 and 256 nodes agrees to roundoff, using the consumed
width `.35` and coordinate-volume weight. This quantifies the distinction
between the archived-failure baseline and the later angular validation.

## Matched-collar baseline in progress

The original amplitude `.2`, wave-map gauge, `kappa1=10`, curvature radius
`.5`, layer `[.05,.95]`, dissipation `.1`, and CFL controls are retained.
The six shell patches use B16/A12/R48 and the radial map

`F(s) = (1-exp(-k*s))/(1-exp(-k)), k=2.8592697712828787`.

This gives minimum active Omega `0.0021701388888882883` on Aurora, matching
the archived N24 value to roundoff. The smooth outer face is at `r=1`;
there is no evolved exact-scri cell or derived characteristic scri closure.
The radial map changes the discrete operator and cell distribution, so the
matched collar does not make this an identical discretization to the old
cut-sphere case.

![Actual graded domain](validation/cubed-sphere-balanced-20261013/graded-domain/computational-domain.png)

Both original-timestep cases now reach **exactly `t=2`**, completing in
allocation **8925007**. Independent collection through allocation 8925179
confirms finite diagnostics, unchanged Minkowski reference payloads and exact
parent-end/child-start diagnostics throughout both original and half-CFL
chains. Both original cases cross the archived N24 failure time
`0.7927734375` and N32 time `1.5837303636`. This matches the N24 minimum Omega;
the N32 marker remains a time threshold, not an identical-collar N32 replay.
The continuation manager records each checkpoint,
submission and outcome in
`build-cubed-sphere/research-20261013-balanced/pulse-chain.json`.

The coordinator then transfers to `pulse-matrix-chain.json`. The last old
allocation is confirmed finished before this handoff; its scientific result
and checkpoints are preserved. The new frozen run directory continues those
two original cases and starts two independent half-CFL controls, with CFL
`.05` and pole factor `.015`. The four cases share one allocated node on
separate GPU tiles. The executable SHA256 remains
`ad413bf2f8c2c64500525ec8e117bc23d75e6d35b21263956c57c3295d6bba38`, and all
376 numerical source/input entries match the original frozen baseline.
The independent [effective-command audit](validation/cubed-sphere-balanced-20261013/pulse-history/provenance-review.json)
checks 66 collected case segments through allocation 8925179. It verifies
all 376 numerical source entries, the executable and input-file hashes, and
the actual last-wins command overrides. The four families differ only in
the declared outer degree and full/half CFL factors, apart from runtime
checkpoint paths, wall limits and cycle guards.
The cycle guard is raised to 100,000 to allow the half-step cases to reach
`t=2`; equations, original physical parameters and reference are unchanged.

CPU fresh/resumed admission passed for all four case configurations, including
exact parent-end/child-start diagnostics and reference equality. Its evidence
is in [pulse-matrix admission](validation/cubed-sphere-balanced-20261013/pulse-matrix-admission/matrix-CPU-review.json).
The half-CFL GPU controls have independently collected endpoints
`t=1.1979014283018954` and `1.1965635996364814` as of allocation 8925260.
They have crossed the N24 time; the N32 and final `t=2` temporal comparisons
remain pending. All four preserve their Minkowski reference payloads. The independent
[matrix-chain recount](validation/cubed-sphere-balanced-20261013/aurora/pulse-matrix/chain-review.json)
confirms exact restart diagnostics for the half-CFL cases as well as the
original cases, including their transfer between coordinators.
The prospective temporal screen requires relative H/M/Z norm differences
below 1% at exact recorded `t=2` endpoints; larger sensitivity calls for further
timestep control. This is not a spatial-convergence or stability certificate.

At interpolated `t=1`, full/half-CFL shell M differences are about `.00432%`
for both closures; Z differences are `.00229%` (degree 2) and `.00868%`
(degree 4). These exceed their interpolation indicators but are much smaller
than the approximately 8.4% boundary-degree effect in Z. H differences are
below the interpolation indicators. This supports a spatial-closure source
for the observed Z sensitivity at this time, without isolating the old
Cartesian failure's cause or replacing the exact endpoint screen.

The [history comparison](validation/cubed-sphere-balanced-20261013/pulse-history/control-review.json)
uses explicitly labeled scalar interpolation between recorded intermediate
samples. Its linear/cubic differences are heuristic indicators, not rigorous
error bounds, and its intermediate values cannot replace the exact endpoint
comparison. At the old N24 time the shell Z boundary-degree difference is
approximately .20%, while the bulk difference is below the interpolation
indicator. By `t=1`, the shell Z difference is approximately 8.4%, versus
`.014%` for H and `.073%` for M; the bulk differences remain below the
interpolation indicators. Thus boundary sensitivity is becoming visible in Z
even while both runs survive. These are single-grid diagnostic observations
only. At the old N24 time, full/half-CFL shell M and Z differences are about
`.0018%` and `.0032%` respectively; these exceed the interpolation indicators
but remain small. They do not replace the planned exact `t=2` comparison.

The [exact endpoint regional recount](validation/cubed-sphere-balanced-20261013/pulse-history/region-review.json)
now shows a substantial outer-collar effect:

| Region at `t=2` | Degree | H RMS | conformal M RMS | conformal Z RMS |
| --- | ---: | ---: | ---: | ---: |
| bulk | 2 | 9.62519e-5 | 1.31311e-5 | 3.65786e-7 |
| bulk | 4 | 9.62519e-5 | 1.31311e-5 | 3.65786e-7 |
| shell | 2 | 2.60156e-4 | 7.64121e-4 | 1.13454e-4 |
| shell | 4 | 2.56520e-4 | 6.18037e-4 | 4.76169e-5 |
| `r>.9` collar | 2 | 2.82039e-4 | 1.29387e-3 | 2.09342e-4 |
| `r>.9` collar | 4 | 2.69164e-4 | 9.76676e-4 | 7.83107e-5 |

The collar accounts for 95.8% / 76.1% of the shell's squared Z norm for
degree 2 / 4. Its endpoint Z difference is 62.6% relative to the larger norm.
Between the first recorded sample at or after `t=1.5` and `t=2`, collar Z
increases by 67.0% for degree 2 and decreases by 17.5% for degree 4. The
actual first times are retained in the recount; no exact `t=1.5` sample is
invented. Bulk and inner-shell norms decrease over this interval. The global
Hamiltonian maximum lies in the collar near `(-.820070,-.341696,.205017)`.
This is evidence of boundary-closure sensitivity, not proof that the old
Cartesian failure was entirely a boundary fault. The map, resolution,
derivative operator, interfaces and dissipation directions also changed.

A [direct checkpoint-field comparison](validation/cubed-sphere-balanced-20261013/pulse-history/endpoint-field-comparison.json)
at exact `t=2` finds maximum absolute differences `0.8268583` in Lambda-x
and `0.3804179` in A-xx, both at the outermost active radial cell (index 47
of 48). At the Lambda-x maximum the values are `-.2978384` and `-1.1246967`
for degrees 2 and 4. The largest bulk difference across all 25 components
is `1.73e-12`; for shell cells at `r<.5`, it is `4.92e-12`.
The comparison excludes ghosts, verifies checkpoint hashes and equal grid,
physics and time, and records every field's maximum and active-cell indices.
These large pointwise collar differences prevent interpreting the smaller
degree-4 Z norm as proof of a more accurate solution. Subsequent refinement
must examine evolved fields and their derivatives as well as constraint RMS.

![Matched-collar pulse histories through t=2](validation/cubed-sphere-balanced-20261013/pulse-history/pulse-history.png)

![Outer-collar constraint sensitivity](validation/cubed-sphere-balanced-20261013/pulse-history/pulse-collar.png)

Research checkpoint admission passed bitwise full-versus-split state equality
on CPU and GPU and a corrupted-payload rejection test. Continuations bind the
executable hash, numerical reference fingerprint, grid and physical parameters.
The independent [chain collector](../tst/cubed_sphere/collect_pulse_chain.py)
recounts diagnostic partitions and verifies identical diagnostic values at
restart boundaries. Production hyperboloidal restart and AMR remain outside
this research adapter's scope.

The adapter retains the old native solver's subtraction of the analytic
stationary Minkowski geometric RHS residual. It does not subtract a
manufactured disturbed solution or black-hole RHS. A new
[raw reference residual audit](validation/cubed-sphere-balanced-20261013/reference-rhs-audit-corrected/review.json)
samples the layer and every outermost cell of the pulse and finest puncture
reference grids: maximum component magnitudes are `2.42e-12` and `4.55e-13`.
The corrected diagnostic uses the actual puncture core half-width
`.25245594360687235`; the earlier `.25` sampling attempt is retained separately.
These are sampled residuals, not global bounds or evidence about error growth.
This host's `long double` has the same 53-bit mantissa as `double`, so no
higher-precision confirmation is claimed. The final physical validation must
demonstrate both targets with the un-subtracted native geometric RHS, following
paired short and stationary-drift controls; the frozen baseline is unchanged.
That future change requires a new source/binary and checkpoint binding. No
future success will rest on assuming that a small static residual cannot grow.

For every region `S`, the reported RMS is
`sqrt(sum_cell(V_cell * squared_constraint) / sum_cell(V_cell))`, where
`V_cell = dq1*dq2*dq3 / abs(det(dq/dx))`. The conformal M and Z contractions
use `bar_gamma^{ij} = chi * tilde_gamma^{ij}`; Z is reconstructed from
`Z_i = tilde_gamma_ij * (Lambda^j - contracted_Gamma^j) / 2`.
Hamiltonian maxima are maxima of `abs(H)` at sampled active cells.
There is no extra physical-volume or Omega weighting. Region subsets overlap;
only the radial bins within each pack, or disjoint bulk and shell, are summed
to reconstruct a parent norm. Initial analytic-jet constraints do not test
the accuracy of later finite-difference derivatives.

## Balanced puncture evidence and remaining collar error

At `t=.005`, B24/A18/R123, B32/A24/R164 and B40/A30/R205 give fixed-layer
(`.375<=r<.75`) coordinate-volume RMS values:

| Grid | H | conformal M | conformal Z |
| --- | ---: | ---: | ---: |
| 24/18/123 | .002980852 | .001197634 | .001007903 |
| 32/24/164 | .001484408 | .000593255 | .000389451 |
| 40/30/205 | .000755944 | .000306068 | .000174619 |

Finest-pair constraint-error slopes are approximately 3.02, 2.97 and 3.59.
The short fine-grid timestep pair at `t=.0025` changes these norms by less
than `.007%`. These are constraint-error slopes, not solution-convergence
orders. The longer `t=.02` family decreases the layer errors with substantially
lower slopes. Initial constraint values use analytic jets and do not establish
finite-difference accuracy.

The global conclusion is limited by the `r>.9` collar: its Z values
`.00159225`, `.00198983`, `.00181379` do not decrease monotonically.
The minimum active Omega also decreases under this refinement. A fixed-collar
comparison and an approach-to-scri comparison must be distinguished explicitly.
All completed balanced cases preserve both the independent Minkowski reference
and the separate immutable black-hole initial reconstruction profile.

Completed single-grid collar controls at B32/A24/R164 and `t=.005` show:

| Outer closure / dissipation | Collar H | Collar M | Collar Z |
| --- | ---: | ---: | ---: |
| cubic / .1 | .000459054 | .01415974 | .001989829 |
| quintic / .1 | .000231679 | .00797627 | .001482781 |
| cubic / .05 | .000458339 | .01418781 | .001994340 |
| cubic / 0 | .000457907 | .01421675 | .001999044 |

Quintic ghosts reduce collar Z by 25.5% and M by 43.7%, leaving the fixed-layer
norms identical at this time. Removing dissipation increases collar Z by only
.46%. These controls show sensitivity to the outer closure and do not support
claiming that removing dissipation cures the collar error. A three-grid quintic
control is the next puncture experiment after the baseline pulse comparison.

The [fine-grid recount](validation/cubed-sphere-balanced-20261013/fine-review/review.json)
and [collar control recount](validation/cubed-sphere-balanced-20261013/collar-review/control-differences.json)
retain the regional values and qualifications. No unconverged collar is removed
from the global acceptance criteria, and no change of norm weighting is used
to claim improvement.

An independent NumPy reconstruction of Z from the unchanged cubic and quintic
checkpoints agrees with the native collar RMS to relative errors `2.24e-14`
and `1.29e-13`. It independently differentiates the constant-reference outer
metric, constructs the inverse map Jacobian and contracted connection, and
recounts the same coordinate-volume norm. The first three radial cell layers
account for 99.77% and 99.79% of the squared collar norm respectively. Comparable
errors occur in face interiors and seam bands; a seam-only explanation is
insufficient. The [recount source](../tst/cubed_sphere/recount_collar_checkpoint.py)
is restricted to the outer CMC region where the initial twice-conformal metric
is exactly identity. It does not validate other regions or repair the closure.

The independent initial-data [connection-rate diagnostic](validation/cubed-sphere-balanced-20261013/initial-connection/review.json)
also passes 84 cases: mass zero and `.5`, 14 radii and three directions, through
`Omega~1e-5`. Differentiating the metric evolution and its contracted connection
agrees with the initial Lambda RHS within `1.29e-10*max(1,|Gamma_dot|,|Lambda_RHS|)`.
This tests an initial continuum consistency identity. It does not clear the full
equations, their gauge evolution or the discrete near-scri operator.

The literature distinguishes staggered extrapolation from exact-scri
regularization. Vañó-Viñuales and Husa describe both methods and impose
regularity relations and limiting RHS equations for their exact-scri method
in [sections 6.1–6.2](https://arxiv.org/pdf/1705.06298). Those spherical formulas
cannot be applied directly to our angular 3D system without a derivation.
Our inference is that ghost-degree convergence is a useful diagnostic, while
it cannot by itself certify characteristic constraint control at scri.

## Integrity and execution policy

A local CFL performance candidate now reconstructs only the field values
needed for characteristic speeds, retaining deviation-plus-background
rounding order and full native-RHS validation before stage updates. It has
passed exact CPU timestep/state-bit comparisons for 100 steps each of
graded pulse, stationary Minkowski and native puncture. Timings were
139.0/132.3, 132.5/126.9 and 132.1/127.6 seconds before/after, including
startup and diagnostics; these single pairs do not establish GPU speedup.
All five CPU regressions pass in 151.69 seconds; GPU admission remains
pending. The [CPU evidence](validation/cubed-sphere-balanced-20261013/cfl-performance/comparison.json)
retains both binaries by hash, all six run receipts, exact timestep histories
and before/after source snapshots. Only `conformal_probe.cpp` differs in the
376-entry numerical source index.
The previous CPU binary and source are preserved there. All Aurora pulse and
prepared puncture binaries/snapshots remain frozen; this local source change
must not be silently introduced into their checkpoint chains.

All 1,373 indexed files in the four earlier cubed-sphere evidence capsules
were rehashed unchanged on entering goal mode. The latest five CPU regressions
passed in 229.51 seconds after the checkpoint implementation. Source, input
and binary hashes are bound separately for each frozen experiment; earlier
puncture refinements preceded the graded-map and checkpoint additions.

The earlier graded-map GPU admission (8924683) is now also retained locally
and [independently recounted](validation/cubed-sphere-balanced-20261013/aurora/graded-admission/independent-review.json).
Its stationary and pulse cases reach `.001`; the step-limited wormhole smoke
reaches only `6.46612e-5`. Reference hashes remain unchanged and regional
constraint partitions agree. These are admission checks, not physical-target
validation or additional AMR evidence.

All task allocations use `CompactBinaryMerger`, `capacity`, one node and
30 minutes, with one task allocation at a time. Solver segments are bounded
to five minutes and continued from validated checkpoints. Other agents' jobs
are not managed by this task. Ordinary Z4c GPU and Cartesian bulk-AMR support
must continue to pass its regressions; the hyperboloidal adapter still requires
a uniform bulk and one rank.

### Interior-only radial closure investigation (user steering, 2026-10-13)

The frozen pulse chain remains authoritative and unchanged. Job 8925287 was
verified running in capacity, CompactBinaryMerger, one node, 30 minutes; local
manager 48043 owns continuation. No additional allocation was submitted.

The exact-rational audit `tst/cubed_sphere/audit_outer_radial_operators.py`
records eliminated NGHOST=3 coefficients for ghost polynomial degrees 2–5,
three outer radial layers, D1, D2 and both advection signs in
`outer-closure/operator-audit.json` within this validation capsule. At the
last active angular-interior node, degrees 2/3/4 give the familiar backward
interpolating D1 with orders 2/3/4, but D2 orders only 1/2/3. Degree 5 gives
fourth-order D2, while D1 and advection remain fourth order and are not the
six-node fifth-order interpolating derivative. Thus eliminating ghosts alone
is not a new mathematical boundary treatment. Mixed radial/angular stencils
factor only where angular ghost interpolation is nodal identity; seams and
corners still require direct plan-level checks. Anchored floating-point ghost
sums need not match collapsed sums bitwise. The probe's radial dissipation
already uses only active third-difference lines; no mapped energy estimate
has been established.

Prospective admission and convergence criteria are recorded in
`outer-closure/acceptance.json`. No alternative evolution operator has yet
been admitted. Actual coupled characteristic analysis, including live
perturbations, is outstanding; the existing lapse/light CFL estimate is not
a complete characteristic audit. No all-outgoing assumption is justified.

The next characteristic admission screen now extracts a 22-variable
pseudo-differential radial symbol directly from the implemented RHS, then
restricts to the 20-dimensional trace-free tangent space in the outer CMC
identity metric. Eighteen synthetic reference/lapse/shift cases pass two
coefficient-extraction amplitudes, eigenpair residual and subspace checks.
At r=.99 the reference physical incoming speeds are -.0001 (pulse) and
-.00005 (puncture reference); +20% lapse gives -.39612/-.19806. Thus even
finite-Omega reference cells do not justify a pure-outflow assumption.
`outer-closure/characteristic-method.md` documents the reduction and limits;
raw symbols, compilation command, binary/source hashes and spectra are retained.
Evolved states and gauge/constraint mode classification remain outstanding.
No evolution closure or reference was changed by this diagnostic.

Evolved characteristic follow-up: direct hash-verified checkpoint scans found
incoming physical radial modes in every outer CMC collar and last-layer cell
of both t=2 pulse closures and both t=.005 puncture collar controls. Pulse
last-layer incoming magnitudes are roughly 0.46–2.05e-6; puncture values are
.00156–.00199. Coupled symbols extracted at 20 extremal cells agree with the
physical-speed extrema and pass step-size and algebraic-subspace checks using
live inverse metrics. This is evidence against pure-outflow treatment, not a
causal attribution of constraint errors. Sources, samples, spectra and hashes
are under `outer-closure`; gauge/constraint classification remains pending.
Job 8925314 was independently verified queued in capacity with the authorized
one-node/30-minute policy; the original continuation manager owns this job.

Principal mode classification now covers 38 synthetic/evolved outer-collar
states. Leading H/M/Theta/Z symbols have rank eight with an invariant
constraint-free kernel. Its transverse trace-free quotient has four physical
modes, and the remaining invariant gauge subspace has eight modes. Each
sample has four incoming constraint, four incoming gauge and two incoming
physical modes, with corresponding outgoing branches; speeds agree with
light branches to the prechecked absolute tolerance 1e-8. Raw spectra,
subspace residuals and hashes are retained under
`outer-closure/constraint-symbols`. This is a principal-symbol result at
finite positive Omega, not a nonlinear boundary or scri proof. It motivates
separately controlled stencil and incoming-data experiments; no native
solver equation or frozen baseline has been changed. Job 8925314 was
verified running in the authorized capacity allocation.

Interior-only candidate implementation has begun in
`src/mesh/cubed_sphere/interior_radial.hpp`: six active radial nodes define
fifth-order D1 and fourth-order D2 at the last three cell centers. Exact
rational generation checks polynomial moments through degree five. A compiled
host test exercises the actual header, including radial/angular tensor-product
mixed derivatives. Finest-pair smooth-function orders are 4.968 (D1), 3.969
(D2), 4.080 (mixed). Coefficients, tests, raw accuracy CSV and hashes are
retained under `outer-closure`. This is a non-SBP diagnostic candidate; neither
an energy estimate nor a complete boundary prescription is claimed. It is
not yet wired into evolution. D1 differs from collapsed degree-5 ghosts;
D2 agrees at the outermost node, which is a useful equivalence control.
Mapped jet/advection integration, dissipation compatibility, seam/corner
checks, GPU admission and three-resolution native comparisons remain pending.
The frozen Aurora binary/reference/inputs remain untouched.

The six-active-node closure is now opt-in through
`cubed_sphere/conformal_interior_radial=false` (default). On the final three
shell layers it reconstructs radial first/second derivatives and radial-angular
mixed derivatives directly from active radial samples, then applies the full
inverse-map Jacobian/Hessian transformation and adds immutable analytic jets.
The same reconstruction supplies native RHS and constraint diagnostics. Radial
advection uses this D1; angular sign-biased advection and the existing active-line
KO remain. This is explicitly a non-SBP stencil experiment, not a completed
constraint-preserving boundary condition; angular patch interpolation remains.

Closure identity is encoded in the research restart signature as outer-degree
plus 100 for the opt-in mode, rejecting accidental closure switching on restart.
The CPU build succeeded. Four-step stationary, pulse and native puncture runs
passed with finite diagnostic rows and identical before/after Minkowski hashes;
a default-closure pulse control passed too. Compact evidence is under
`outer-closure/integration`. These tiny runs establish integration only, not
stability, convergence, GPU correctness or an incoming-data prescription.
The frozen GPU chain and completed old evidence remain untouched.
The default-closure four-step pulse was also replayed with the preserved
pre-integration binary: checkpoint contents are exactly identical after
excluding the executable tag and checksum. The original half-CFL chain has
independently collected endpoints 1.3177308436983906/1.3152628901207417 from
8925314, with unchanged references. Manager-owned next job is 8925337.

Further interior-closure admission: compiled mapped analytic tests cover all
six faces, three outer radial depths and angular interior/edge/corner locations.
Both linear and stretched maps attain approximately fourth-order Cartesian
gradient and Hessian accuracy (finest orders 3.995–3.999). Analytic angular
ghosts are used in this test; it is not an interpolation convergence result.
Same-mode four-plus-four-step restart matches a fresh eight-step checkpoint
bitwise, and old-closure to interior-closure restart is rejected as intended.

A separate native four-step pulse control changed outer ghost extrapolation
from degree 2 to degree 5 while holding the interior-only mode fixed. Every
active bulk and shell field, including seams/corners, is exactly identical.
This confirms radial outer-ghost independence for this test, without claiming
independence from angular interpolation. All evidence is retained under
`outer-closure/integration`. Job 8925337 was verified queued, capacity/one
node/30 minutes, with the original manager alive; no competing job submitted.

GPU preparation (no allocation submitted): isolated candidate source is at
`/lus/flare/projects/CompactBinaryMerger/hzhu/cubed-sphere-balanced-20261013/interior-radial/source`.
All 448 packaged source/input/test files were hash-verified remotely before
starting an independent Release SYCL/PVC build (`build-sycl`, four compiler
workers, preserved flags and oneAPI 2025.3.1). This does not modify any frozen
phase binary. Build is ongoing via local exec session 28080; observe this
handle and remote build log before starting any replacement. The candidate
binary still requires a completed build/hash and GPU execution admission.

Six matched CPU golden cases (stationary/pulse/puncture, each old/interior
closure) completed four steps with finite regional diagnostics and unchanged
reference hashes. Golden raw checkpoints remain under
`research-20261013-balanced/outer-closure/cpu-golden`; compact evidence and
predeclared CPU/GPU tolerances are in `outer-closure/gpu-admission`.
The staged `run_gpu_admission.pbs` explicitly selects phase pulse, with
per-case puncture overrides, capacity/CompactBinaryMerger/one node/30 minutes.
It must NOT be submitted until the frozen baseline chain finishes. Candidate
`athena` needs to be copied from the completed isolated build before that run.
The generic PBS file also present is not the dedicated admission entry point.

Local five-test regression run is ongoing in exec session 3790, log
`research-20261013-balanced/outer-closure/ctest.log`; map and bridge have passed,
regular Z4c was running at last check. Frozen chain job 8925337 was independently
verified running in capacity. No duplicate allocation was submitted.

The integrated candidate passed all five local CTests (150.49 s): map,
conformal bridge, regular Z4c, scalar transport and failure capture. These
preserve the tested ordinary solver paths; they do not establish hyperboloidal
AMR support. The GPU build remains active in session 28080 and had reached
34% at the last remote log inspection. Do not duplicate that build or submit
its admission allocation while the frozen pulse manager remains active.

`tst/cubed_sphere/compare_closure_admission.py` now checks matched source hashes,
parameters, physical endpoint, checkpoint payload hashes, every active field,
and every regional H/M/Z norm against the predeclared CPU/GPU tolerances.
The CPU self-comparison passes; a separate synthetic +1 perturbation to the
final outer-collar Z is rejected, with the failed check recorded. Neither
checker exercise is GPU evidence. Compact reports are under
`outer-closure/gpu-admission`; native CPU golden checkpoints are preserved.

A decisive SBP limitation is now checked exactly in
`tst/cubed_sphere/check_interior_radial_sbp.py`. For last node i and node
j=i-5, candidate D_ij=-1/5 while D_ji=D_jj=0. With positive diagonal H and
local disjoint endpoint traces, the j,j SBP equation forces right trace t_j=0;
the i,j equation then requires H_i*(-1/5)=0, a contradiction. Thus this
candidate cannot be upgraded to a conventional diagonal-norm SBP operator by
choosing weights alone. Dense norms or redesigned boundary rows are not ruled
out. It remains a diagnostic stencil control, not an energy-stable boundary
claim; a compatible SBP/SAT construction requires a separate operator design.
The exact scope and proof are in `outer-closure/sbp-obstruction.json`.

Frozen pulse job 8925337 has been collected and verified: half-degree2 is at
1.3771654086481 and half-degree4 at 1.3739759701364811; the reference hashes
remain unchanged. The original degree2/4 cases remain complete at t=2.
Manager-owned continuation is 8925370. Pulse plots/control tables are being
refreshed from the collected chain (local session 83713). Isolated GPU build
session 28080 is still active; no candidate GPU allocation has been submitted.

A separate cell-centered SBP construction has now been explored without changing
the integrated candidate or isolated GPU build. Six boundary rows, quadratic
boundary traces, a positive diagonal norm and centered fourth-order interior
D1 satisfy the SBP identity numerically. Fully compatible D2=D1 squared and
H-weighted active-line KO satisfy their energy identities; homogeneous scalar
inflow SATs are non-increasing in energy for both advection signs. This basic
D2 is only first-order accurate at the boundary, so it is not a preferred final
closure or accepted native implementation. Coefficients, residuals, references
and limitations are in `outer-closure/cell-centered-sbp-design.md` and its JSON.
Patch coupling, mapped coefficients and actual Z4c boundary data remain open.

A higher-order SBP algebra prototype now has sixth-order interior/third-order
boundary D1 and second-order compatible D2, confirmed by 40–320 cell smooth
refinement; positive norms and energy identities pass. It is not integrated.
Its face trace would map to scri on the current shell, so importing its scalar
SAT into the finite-Omega native equations would be unjustified. Boundary
placement at finite Omega versus separately derived scri regularity must be
settled before integration. This limitation is documented alongside the
coefficients, accuracy table and energy checks. Frozen source/build untouched.

Finite-Omega SBP placement now has an algebra prototype: nodal traces at the
unchanged first/last active centers, positive rebuilt norm, compatible derivative
identities, and measured D1/D2 boundary convergence 2.984/1.991. It avoids any
Omega=0 evaluation but changes the energy quadrature domain (N-1 spacings),
which is documented separately from unchanged scientific regional norms.
The last-node constant-logical-radius surface has normal J[0]/|J[0]| rather
than x/r. Direct live-field scans with that actual surface normal still find
incoming physical modes throughout the tested pulse/puncture collars. Native
SBP integration, interface coupling and incoming boundary-data derivation remain
outstanding; no frozen binary or equation was changed by this exploration.

The actual constant-logical-radius normal now has a coupled characteristic
classification on ten selected evolved pulse/puncture cells. A further check
on all 48 synthetic/live radial/normal samples verifies the two-branch minimal
polynomial and builds incoming/outgoing projectors without arbitrary numerical
eigenvectors. Their positive frozen-coefficient symmetrizer has flux asymmetry
below 1.22e-15 and condition number up to 4.86e6. This is not a scri-uniform or
nonlinear stability proof. Matrices, diagnostic binary/source hashes and
reports are under `outer-closure/logical-symbols` and `projector-review`.
Isolated SYCL build session 28080 remains live; pulse job 8925370 was verified
running with the authorized capacity/one-node/30-minute allocation. No duplicate
job or replacement build was started.

Isolated SYCL/PVC candidate build completed successfully (session 28080 exit 0).
Binary SHA256 is
`4ce9bea8a1fd27e565ec831bc5d0da45ae40a142e6dd38ce58659a49980b754d`.
It was copied to the dedicated `interior-radial/athena` admission path and its
hash rechecked. Configure/build logs and hash are retained in
`outer-closure/gpu-admission`. No GPU execution/admission claim yet, and no
allocation submitted. Frozen binary ad413... remains unchanged in its chain.

`compare_closure_field_norms.py` now measures within-grid coordinate-volume
componentwise closure differences, checking volume/counts against native region
diagnostics. Bulk block ordering comes from hash-verified immutable reference
coordinates rather than an assumed single block. This enables comparing
closure-difference norms across balanced resolutions without cross-grid
interpolation; changed near-scri sampling still requires explicit reporting.
Four-step pulse and puncture examples pass the geometric cross-check, with
max-component collar L2 differences 3.81e-12 and 1.10e-7 respectively. These
are admission examples, not evidence of refinement convergence.

Collected job 8925370 advances half-degree2 to 1.4372197614370044 and
half-degree4 to 1.4328719398998675, retaining immutable hashes and exact restart
checks. Original full-CFL cases remain at t=2. Manager-owned next job 8925389
is queued according to its log; recheck scheduler state before acting.

The next non-SBP interior-only balanced matrix is staged, NOT submitted, at
`.../cubed-sphere-balanced-20261013/interior-refinement`. It contains
B24/A18/R123, B32/A24/R164, B40/A30/R205 at t=.005 and dt=5e-5, a fine halfstep
control, and medium ghost3/ghost5 controls to check old/new binary consistency.
The candidate binary hash is 4ce9...754d; remote grids hash
`4bb33388086936df7a4deb3572dd5315e93d888d1d7fe7ae681f7b60a60bb9c8` was verified.
The source symlink points to the preserved isolated candidate source. Submission
is gated on frozen pulse completion, candidate GPU admission, and the prepared
quintic baseline. Quantitative requirements and physical matching are recorded
in `outer-closure/refinement-plan.json`; no completed evidence was replaced.

The dedicated future GPU admission PBS now runs six short cases followed by
same-allocation restart/ghost controls through `run_closure_followup.py`.
Bounded worker timeouts are 540s plus 360s inside one capacity/one-node/30-minute
allocation. The CPU rehearsal of this exact follow-up passed: four-plus-four
steps match fresh eight steps bitwise, and changing outer ghost degree 3 to 5
leaves all active fields bitwise equal. `review_closure_followup.py` checks both
conditions and payload hashes. Evidence is retained under
`outer-closure/gpu-admission/cpu-followup`; this is still CPU evidence only.
No GPU job was submitted. Current manager-owned pulse job 8925389 was verified
queued in capacity at the start of this turn.

Cross-closure immutable-data audit now goes beyond within-run before/after
hashes. `check_reference_closure_identity.py` confirms bitwise identity of
Minkowski field values and all preceding per-cell geometry/reference-jet/RHS/
quadrature records for old versus interior-only closure in the matched CPU
pulse and puncture cases. The full reference cache includes two trailing
numerical closure-control integers per Cell; these explain the differing
whole-file hashes across closure modes. Only those integers are separated for
this cross-closure comparison; full before/after hashes remain required for
every evolution. ABI/shape/metadata checks constrain the parser to the candidate
layout. Evidence is in `outer-closure/gpu-admission/*reference-identity.json`.

The separate black-hole initial reconstruction profile has identical whole-file
hashes before/after and across both puncture closures. It remains distinct from
the Minkowski reference. No state/reference change was needed for this audit.
Job 8925389 remains queued in capacity, independently verified, and manager
48043 is alive. No replacement or duplicate job was submitted.

Further seam audit (`audit_outer_seam_elimination.py`, evidence
`outer-closure/seam-elimination.json`) independently reproduces the Map and
active donor-plan algebra, without modifying or instrumenting the frozen
native binary. Pure radial D1/D2 and both advection signs agree with the
collapsed one-sided operator to roundoff on all six faces, three boundary
layers, and both linear/graded maps. Mixed radial/angular operators at patch
edges and corners do not generally equal a radial elimination composed with
active-radius seam interpolation: the full cross-patch operator is needed.
For example, the degree-five maximum relative coefficient L1 difference is
0.0806 on the linear map and 0.0907 on the graded map, despite face-interior
agreement. This coefficient discrepancy is not a physical solution error.
For a smooth exponential sampled on three linear shell grids, mixed-operator
differences decrease for degrees 2–4; degree-five differences become unresolved
against the conservative roundoff bound, so no degree-five order is claimed.
These are donor-model checks, not direct native plan dumps, mapped Hessian
validation, constraint convergence, or evidence that seams cause the failure.
The planned evolved-state comparison must include seam/corner localization.

A prospective phase-three pure quadrupolar lapse profile is checked in
`angular-target-design/profile.json`: maximum relative amplitude 0.2, zero
angular mean, angular energy fraction 1, and alpha/reference >=0.8. It is smooth
at the origin and vanishes as (1-r^2)^4 near scri. No native initialization or
current input was changed and no phase-three evolution is claimed.

Job 8925389 completed its pulse segment; half-degree2 reached
1.4969204780128658 and half-degree4 reached 1.4918338172235457, according to
the manager's checkpoint summary. Both remain below the later archived
Cartesian failure time. The existing manager submitted successor 8925424;
no independent job submission was made. Full-CFL cases remain at t=2.

Closure-field diagnostics now retain componentwise maxima locations and add
complete outer-collar face/edge/corner partitions. Fixed logical angular bands
(abs(u) or abs(v)>0.8) are distinguished from two-cell stencil bands, whose
physical width shrinks on refinement. Both partitions preserve the entire
r>0.9 collar; their counts, volumes and squared component integrals are checked
to sum to the existing collar totals. Global and native fixed-region norms
remain unchanged and checked against native quadrature records.
Replayed the four-step CPU pulse and puncture comparisons with verified
checkpoint/reference hashes. Puncture maximum-component L2 differences on the
fixed-band face interior, edge and corner subsets are 1.15e-7, 1.50e-7 and
1.97e-7 respectively. These compare numerical closures after only four steps;
they are not H/M/Z constraints, convergence, or evidence of a failure cause.
Outputs: `outer-closure/gpu-admission/*four-step-seam-difference.json`.

At this continuation's live check, manager PID 48043 was alive and job
8925424 was queued in capacity under CompactBinaryMerger. No new independent
allocation was submitted. The preceding goal turn made progress through the
seam algebra audit and collection of completed job 8925389; the current turn
adds localization diagnostics while the frozen continuation waits normally.

Refreshed full histories, outer-collar plots, scalar temporal review, regional
review and provenance through collected job 8925389. Provenance audit verified
84 case segments and 376 numerical source entries. The regenerated collar
plot includes half-CFL controls through t~1.49; those curves closely follow
their corresponding full-CFL curves while the two ghost degrees separate in
Z. This visual agreement is not the final exact-t=2 temporal-error gate and
cannot establish spatial convergence. Current job 8925424 was confirmed
running; the preceding turn is classified as a verified wait.

Completed pulse job 8925424 (PBS F, 5m16s used) advances the half-CFL
controls to t=1.5566616198987624 (degree2) and 1.5517635221985191 (degree4).
The full-CFL cases remain at t=2. Both new checkpoint summaries retain the
frozen binary and immutable reference hashes. The serial manager submitted
8925450 and reported it queued; no competing allocation was submitted.
At stage 26/40, recent increments leave sufficient stage headroom to reach
2, without modifying the numerical or scheduler controls. Neither new
half-CFL checkpoint has yet crossed the later Cartesian failure threshold.

Milestone: collected and validated job 8925450. Half-degree2 reached
1.6156489577524973; half-degree4 reached 1.6109333452292853. Both have now
crossed both archived Cartesian failure times with the original frozen
parameters and minimum Omega. Collector checks passed for reference invariance,
finite diagnostics and exact parent/child restart agreement. Full-CFL cases
remain at t=2. Successor 8925482 was submitted by the existing manager.

Updated histories and regional plots through this segment. At the later
Cartesian threshold t=1.5837303635638276, local cubic interpolation of shell
scalar diagnostics gives ghost-degree Z sensitivity 38.8439% at full CFL and
38.8867% at half CFL. Full-versus-half CFL Z sensitivity is 0.0718311% for
degree2 and 0.00174178% for degree4. The relative linear/cubic indicators are
below 9.5e-7 for these shell comparisons (heuristic, not rigorous error bounds).
Bulk tiny closure/time differences are unresolved relative to their larger
interpolation indicators and must not be advertised as quantitative temporal
accuracy. Captured the threshold evidence and input hashes in
`pulse-history/later-cartesian-threshold-review.json`.

This establishes survival past the archived times and closure sensitivity on
one mapped grid; it does not prove convergence, identify the native failure's
cause, or exonerate the equations. Exact t=2 half-CFL endpoints remain required
before the prepared GPU admission and three-resolution puncture matrices.

Collected and validated completed job 8925482. Half-degree2 reached
1.675022164774727; half-degree4 reached 1.669922010486353. Full-CFL cases
remain at t=2. Reference invariance, finite diagnostics, scheduler policy and
parent/child restart checks passed. Manager-owned successor 8925513 is queued
according to its submission log; recheck live state before further action.
No independent allocation or numerical parameter change was made.

Collected and validated job 8925513: half-degree2 t=1.7340839545385445,
half-degree4 t=1.729603387439731; immutable reference, finite metrics and
restart continuity passed. Full-CFL cases remain at t=2. Manager-owned job
8925559 was submitted serially and logged queued. No independent allocation
or numerical change was introduced; exact t=2 temporal controls remain pending.

Collected and validated job 8925559: half-degree2 t=1.7944541303562276,
half-degree4 t=1.789507402141771, full-CFL cases still t=2. Reference invariance,
finite diagnostics, scheduler policy and exact restart continuity passed.
Existing manager submitted successor 8925601 and logged it queued. No duplicate
allocation or altered numerical parameters; final exact t=2 controls remain
pending before isolated GPU admission.

Collected and validated job 8925601: half-degree2 t=1.8550478554074812,
half-degree4 t=1.8486210581594944. Reference invariance, finite diagnostics,
scheduler policy and exact restart continuity passed; full-CFL cases remain
at t=2. Existing manager submitted 8925630 and logged it queued. No independent
allocation or parameter change. Exact t=2 half-CFL comparison remains pending.

Collected and validated job 8925630: half-degree2 t=1.9146362447992438,
half-degree4 t=1.9072173110496184. Reference invariance, finite diagnostics,
scheduler policy and exact restart continuity passed. Full-CFL cases remain
at t=2. Manager-owned successor 8925651 was submitted serially and logged
queued. Exact t=2 half-step comparison remains pending; no duplicate allocation
or changed parameters.

Collected and validated job 8925651: half-degree2 t=1.9749919389996389,
half-degree4 t=1.9673264733853821. Reference invariance, finite diagnostics,
scheduler policy and exact restart continuity passed. Full-CFL cases remain
at t=2. Existing manager submitted successor 8925691 and logged it queued.
The remaining interval fits within one recent-rate bounded segment; this is
an estimate, not endpoint completion. No duplicate allocation or changed
parameters. Exact t=2 comparison still gates subsequent closure allocations.

Frozen matched-collar baseline comparison completed. Job 8925691 ended with
all four cases at exact t=2; manager PID 48043 exited without further submission.
Collected segment checks passed for native diagnostic finiteness, immutable
reference hashes, restart continuity and scheduler policy. Final provenance
review verified 102 case segments and 376 numerical source entries.
Downloaded both final half-CFL checkpoint payloads and verified receipt hashes.

At exact t=2, outer-collar relative ghost-degree differences (half CFL) are
H=4.5590%, M=24.4739%, Z=62.6876%. Full/half temporal differences in the same
collar are H=0.001401%, M=0.045547%, Z=0.230662% for degree2, and H=0.007616%,
M=0.009386%, Z=0.025080% for degree4. These are differences of regional scalar
norms on one grid, not norms of constraint differences or spatial error.
Maximum active-field component differences are 0.828811 between half-CFL
closures, compared with 0.000696603 (degree2) and 0.00502803 (degree4) for
full/half timestep pairs. Maxima are in the outer collar; componentwise fields
have differing physical dimensions, so these maxima are diagnostics only.
All comparisons use exact endpoints. Full/half signatures and payload hashes
are checked by `compare_pulse_checkpoints.py`.

Both full and half controls cross both archived Cartesian failure times and
reach 2 with the frozen parameters/minimum Omega. Thus the old early failure
is not reproduced by this mapped-shell baseline. Topology, interpolation and
boundary treatment changed together; the evidence does not uniquely attribute
the Cartesian failure, exonerate the native equations, establish spatial
convergence, or demonstrate late settling. The mostly spherical pulse does not
meet the genuinely angular target. Conclusions, comparisons and evidence hashes
are retained in `pulse-history/baseline-conclusion.json`. Proceed to preserved
candidate GPU admission, then the quintic and interior-only balanced puncture
matrices; physical acceptance remains incomplete.

Submitted preserved isolated GPU admission as 8925704 after verifying manager
48043 had exited and qstat showed no queued/running allocation from this task.
Remote binary hash remains 4ce9...754d. Explicit submission: capacity,
CompactBinaryMerger, select=1, walltime=00:30:00. Six four-step matched cases
plus same-allocation restart/ghost controls; no puncture refinement matrix
submitted concurrently. Submission hashes/identity are in
`outer-closure/gpu-admission/submission.json`. Other agents' jobs untouched.

While GPU admission 8925704 remains queued, tightened the single-matrix review
to report the predeclared finest-pair RMS order >0.5 gate explicitly. Reanalysis
of preserved linear-fine data fails both monotonicity and order gates; its
temporal budget remains unavailable at matched spatial time. The report now
explicitly leaves full constraint acceptance false: regional M/Z maxima and
locations, measured roundoff floor, and alternate-closure refinement are not
provided by these CSVs. Evidence and reviewer snapshot retained in
`outer-closure/reviewer-gates`. No numerical source or frozen binary changed.

Extended independent outer-CMC checkpoint Z recount to the six-active-node
radial closure, with polynomial moment checks and explicit stencil-radius
checks on the reported collar. The CPU four-step interior case agrees with
native collar RMS to 2.44e-10 relative; preserved t=.005 quintic is also
rechecked. Four-step ghost control narrowly fails the unchanged 1e-9 relative
recount threshold (1.171e-9 relative, ~1.55e-18 absolute difference); recorded
as a failure, not rounded into a pass. This adds Z maxima/locations only in
the outer CMC collar, not full-domain M/H maxima. No evolution source changes.
Evidence: `outer-closure/Z-recount-admission.json`. GPU admission 8925704
remains queued; no second allocation submitted.

GPU admission 8925704 completed F/0. All six stationary/pulse/puncture
CPU/GPU comparisons pass predeclared field and native H/M/Z tolerances.
Same-device 4+4 versus fresh 8 steps is bitwise exact; degree3 versus degree5
interior-only active states match bitwise. All nine receipts report invariant
references. Hash-checked checkpoint payloads retained locally; compact evidence
and scheduler proof in `outer-closure/gpu-admission`. These are short admission
checks, not physical or convergence validation.

After confirming no task allocation remained, submitted frozen quintic-ghost
puncture refinement job 8925728: capacity, CompactBinaryMerger, one node,
30 minutes. Three balanced grids plus fine halfstep to t=.005, unchanged old
binary ad413...ba38. Metadata in `outer-closure/quintic-refinement/submission.json`.
No auto successor or concurrent interior-refinement job; other jobs untouched.

Frozen quintic matrix 8925728 completed F/0, all four cases at exact t=.005.
Collected inputs, receipts, CSVs and logs; independently rehashed remote
checkpoint and before/after Minkowski reference payloads against receipts.
Outer-collar Z RMS = .0013987750, .0014827807, .0010213277: coarse-to-medium
rises despite fine-pair p=1.67074. All other reported RMS trends decrease,
but the collar Z criterion fails. Fine halfstep/spatial norm-difference
ratio is .00013004 for collar Z; maximum across reported quantities is
.00089402 (layer M). Thus timestep error does not explain this nonmonotonicity.
Min Omega changes with balanced resolution; survival and finest-pair slope
are not convergence acceptance. Evidence/plot: `outer-closure/quintic-refinement`.

After checking terminal state and no live task allocation, submitted interior
refinement job 8925749: capacity, CompactBinaryMerger, one node, 30 minutes.
Candidate binary 4ce9...754d unchanged. Three balanced grids, fine halfstep,
and medium ghost3/5 binary controls at t=.005. No automatic successor and
no concurrent task allocation. Submission is in `outer-closure/interior-refinement`.

Interior matrix 8925749 ended F/1 before evolution: provenance launcher
combined resolved source root with unresolved __file__ sibling paths through
a source symlink. No case ran. Preserved job log/scheduler under
`outer-closure/interior-refinement/failed-8925749`. Corrected only the PBS
invocation to resolve source_root with readlink -f before calling Python;
path consistency and shell syntax pass, binary/grid hashes unchanged.
After confirming no live task allocation, retry 8925755 submitted to capacity,
CompactBinaryMerger, one node, 30 minutes. Original submission/launcher retained;
retry metadata and corrected launcher separately recorded. Quintic raw-checkpoint
transfer session 58247 remains active; do not duplicate it.

Independent quintic Z extrema: coarse .0062425997, medium .0063242717,
fine-halfstep .0053070434, all outermost radial layer. Coarse-to-medium
maximum grows as well as RMS; angular peak moves from near a corner to a
face interior. Medium new payload is bitwise identical to the archived
quintic checkpoint, allowing verified reuse before transfer completes.
Fine fullstep remains pending; halfstep is not substituted into spatial
acceptance. Evidence: `outer-closure/quintic-refinement/Z-extrema-progress.json`.

Quintic checkpoint transfer session 58247 completed; all four local payload
hashes/sizes match receipts. Independent outer-CMC Z recount agrees with
native collar RMS to <=2.33e-13 relative for all four endpoints. Fullstep
Z maxima coarse/medium/fine: .0062425997/.0063242717/.0053066848; coarse
to medium still grows. Fine halfstep .0053070434, with temporal/spatial
maximum-difference ratio recorded in `quintic-refinement/Z-extrema-conclusion.json`.
All peaks are at radial depth zero, but angular locations change. No excluded
collar cells or causality claim. Interior retry 8925755 now running.

Interior matrix 8925755 completed F/0; all six cases reach exact t=.005
with receipt reference invariance. Collected CSV/input/log/receipt evidence.
Interior-only collar RMS Z: .0015285306/.0016545045/.0011933065; M:
.0188415178/.0070453655/.0083951553. Collar Z coarse-to-medium, M medium-to-
fine, and global/shell M fail monotonicity. Collar H maximum also rises
medium-to-fine. Temporal screen passes; this is not a convergent repair.
Medium ghost5 old/new binary native norm comparison passes declared tolerance;
active field comparison remains pending. Raw checkpoint transfer session81989
and remote payload rehash session36960 active; no new allocation submitted.
Evidence: `outer-closure/interior-refinement/preliminary-conclusion.json`.

All six interior-matrix remote checkpoint and Minkowski before/after payload
hashes now independently verified against receipts. Raw transfer session81989
continues. Added hash-bound compact bulk coordinate/index extraction for
pointwise comparisons without downloading multi-GB reference jets. CPU
extracted records match original bytes, and full-reference versus compact
field comparison outputs are byte-identical. Analysis code only. Remote
quintic coarse/medium/fine coordinate extraction session2892 active; no
Aurora allocation submitted. Evidence `outer-closure/geometry-extraction-admission`.

Quintic bulk coordinate extracts for all three balanced resolutions completed
and downloaded; hashes match manifests bound to original reference/checkpoint
payloads. Coarse pointwise quintic/interior field comparison completed with
native quadrature count/volume cross-checks: active bulk differences exactly
zero; largest collar coordinate L2 component differences ~8.50836e-4 in
Lambda components 14-16. Coarse interior Z maximum .0074463187 versus
quintic .0062425997; native collar RMS recount agrees to 4.22e-15 relative.
Full resolution comparison pending remaining transfer session81989. No new
allocation. Evidence `interior-refinement/coarse-field-difference.json` and
`interior-refinement/coarse-Z-recount.json`.

Fine fullstep interior checkpoint downloaded and hash/recount checked.
Native collar Z agrees to 1.72e-13 relative, maximum .0070097843 in outermost
radial layer near face center. Fine pointwise comparison versus quintic
completed: collar Lambda_x L2 closure difference .0010553422 exceeds coarse
.0008508362, contradicting decrease across both refinement pairs. Active bulk
fields are still identical. Middle-grid and field timestep/binary controls
pending transfer; no causality claim or new allocation. Evidence
`interior-refinement/coarse-fine-field-observation.json`.

Added signature-enforced fixed-timestep field comparison (only signature
index15 halves; physical endpoint/grid unchanged). Fine collar Lambda_x L2
full/half differences: quintic 9.65612e-8, interior 1.01107e-7, versus
closure difference .0010553422. Ratios <9.59e-5, so this closure discrepancy
is not explained by the measured timestep field difference. This is not
the per-quantity temporal/spatial gate. New compact interior-fine geometry
was independently extracted/hash-bound. Raw transfer session81989 remains
active for medium and binary controls. No new allocation.

Medium cubic old/candidate GPU binary control completed from hash-checked
checkpoints: all 25 active components in bulk and shell are bitwise identical
at exact t=.005 with matching physical/checkpoint signatures. Native regional
H/M/Z comparisons also pass declared tolerance. Ghost5 binary control and
medium interior remain in active transfer session81989. No allocation
submitted. Evidence `interior-refinement/medium-ghost3-binary-control.json`.

Medium quintic old/candidate GPU binary field control completed. All active
components are bitwise identical at t=.005; the existing native H/M/Z norm
control also passed. Both cubic and quintic baseline binary controls now
rule out an unrelated change in these matched baseline evolutions. Medium
interior checkpoint still in transfer session81989. No new allocation.
Evidence `interior-refinement/medium-ghost5-binary-field-difference.json`.

Transfer session81989 completed. All six local interior-matrix checkpoint
payloads verified. Medium pointwise comparison and Z recount complete:
collar Lambda_x closure L2 differences coarse/medium/fine = .0008508362/
.0009547436/.0010553422, with negative observed slopes -.4005/-.4489.
Interior Z maxima .0074463187/.0076901412/.0070097843 also fail monotonicity.
Both baseline binary controls bitwise identical, and fine field timestep
effects much smaller than closure differences. Full results/plot in
`interior-refinement/three-grid-closure-review.json` and `closure-comparison.png`.
This interior-only non-SBP operator is rejected as a convergent repair.
No active transfers or allocations remain. Next diagnostic proposed: separate
finite physical outer radius from immutable reference scri radius, then
fixed-finite-Omega refinement to isolate near-scri sampling from closure.
Adapter currently couples map_.R to reference radius; must derive/test
decoupling before submission. This control must not replace full targets.

Predeclared finite-radius diagnostic in `outer-closure/finite-radius/acceptance.json`.
Adapter decouples physical map outer radius <=1 from fixed unit-scri Minkowski
reference; pulse taper, gauge validation and radial bins retain reference radius.
Native equations unchanged. Preserved old CPU binary 95dc...f8fb and source.
New CPU build: all six R=1 four-step controls bitwise identical to old active
fields, reference payloads and diagnostic CSVs. Six R=.98 short stationary,
pulse and puncture cases pass finite/reference invariance checks; shared bulk
reference records identical across R=1 and .98. Domain descriptor records
both radii. Restart/GPU admission and true fixed-radius refinement still
pending. No Aurora allocation submitted.

Finite-radius followup exposed a real diagnostic-adapter bug before GPU
admission: KO line membership still used r<1, allowing extrapolated ghosts
inside r<1 but outside physical R=.98. First attempt restart was bitwise
exact but degree independence failed by 7.63e-13. Preserved v1 source and
failure record, kept exact criterion. Replaced only KO domain membership
with physical outer radius squared. New CPU binary ee24814a10cc75f3dd02c00a2e1bf1a0a1a7ca79bf1f041bcc4db0cd4886eaa8.
All six unit-radius baselines remain bitwise identical in active fields,
reference payloads and CSVs. Finite-radius v2 4+4/fresh8 restart is bitwise
exact and ghost3/5 active fields now exactly equal. No native PDE/reference
change. Evidence under `outer-closure/finite-radius`, especially
`followup-v1-failure.json`, `unit-v2-compatibility.json`,
`cpu-followup-v2-review.json`. No Aurora allocation; GPU build/admission next.

Finite-radius incompatible-domain restart negative control passed: .98
checkpoint rejected at R=1 without modifying original payload. Isolated
remote GPU source at `cubed-sphere-balanced-20261013/finite-radius-v2`;
450 source files hash-verified. Configure first failed for omitted
config.hpp.in, now included with manifest and retained failure log.
Build retry session24847 active, no allocation submitted. Prepared six-case
finite-radius admission and same-allocation restart/ghost controls; pending
binary build before submission. Source/archive/script hashes retained in
`finite-radius/gpu-build-preparation.json`. Archive sync session48811 pending.

Finite-radius GPU build session24847 completed successfully: binary
223aabe4f0840e28ac01b27c7d9b66c5f2ca187b728ce7c8688789c3a83cb99b.
All 450 remote sources rehashed, build/config logs retained. Archive sync
and previous observation sessions completed. No task allocation existed
before submission. Admission job8925830 submitted capacity/CompactBinaryMerger/
one node/30 minutes; currently queued, only live task allocation. Remote
root finite-radius-v2, submission record gpu-admission-submission.json.
Prepared locally two four-case fixed-R=.98 refinement matrices and one
sequential PBS, NOT submitted; admission must pass first.

Finite-radius independent diagnostic readers now verify unit reference scri
from the domain descriptor and accept .9<R<=1. Native logical-normal spectra
at five four-step CPU puncture extrema retain incoming physical, gauge and
constraint modes; all 4392 collar cells have incoming physical branch.
No pure-outflow claim. Independent Z recount at four steps fails unchanged
1e-9 relative tolerance by 4.63e-9 (absolute 9.37e-19, native Z=2.02e-10);
retained failed admission, must validate at meaningful t=.005 amplitudes.
Unit-radius medium recount regression still passes. No native equation change.

GPU finite-radius admission job8925830 completed F/0. All nine remote
checkpoint hashes verified; downloaded checkpoint archive190479049bytes.
CPU/GPU six-case field and H/M/Z comparison passes predeclared tolerances.
Followup 4+4 versus fresh8 whole-checkpoint bitwise exact; interior ghost3/5
active fields exactly equal in bulk and shell. All references invariant.
Evidence gpu-admission-comparison.json, gpu-followup-review.json, compact
gpu-admission/. Full downloaded checkpoints remain in research directory.
Fixed-radius refinement submission initiated after F/0 and no-live-task check;
remote source rehash and qsub observation session32863. Never repeat qsub
without checking refinement-submission.json locally/remotely.

Submission session32863 completed: only live task allocation is8925848,
fixed-radius ghost5 and interior matrices sequentially in capacity, one node,
30 minutes, CompactBinaryMerger. All450source files and binary/scripts/grids
rehash before qsub. No active local transfer/build/submission sessions remain.
Next: observe this job; collect both summaries and checkpoint evidence;
apply existing strict three-grid/time/regional and closure-difference gates.
Goal stays active and incomplete. Do not submit a duplicate or successor
until this job is final and its results have been audited.

Goal continuation: previous turn classified progress (completed GPU admission,
submitted fixed-radius matrix). Job8925848 revalidated live. Added reusable
paired three-grid closure review; regression against preserved unit-radius
reports reproduces failed closure-difference screen (138 nondecreasing nonzero
component/region entries, supplementary overlapping regions included).
Exact-zero entries are distinguished; no nonzero roundoff exemption invented.
No physical acceptance asserted; regional M maxima remain a known gap.

Job8925848 still live; quintic fixed-R=.98 four-case matrix complete.
Compact diagnostics collected, reviewed, plot visually inspected. All required
regional/global H/M/Z RMS decrease, finest orders>.5, time budget<10%.
Collar Z=8.07610e-7/3.97171e-7/2.38804e-7. This passes only a necessary
short-time diagnostic screen, not original scri-boundary convergence or either
physical target. Interior matrix pending. Ghost5 checkpoint/verified bulk
geometry transfer active local session13730; do not restart it. Evidence
finite-radius/refinement/ghost5-review and ghost5-preliminary-conclusion.json.

Job8925848 F/0; no active task allocation. Both fixed-radius closure matrices
complete t=.005 and pass necessary RMS monotonic/order/time screens, including
all fixed regions and collar. Matched source/binary/physical/grid parameters
and exact same within-pair minOmega verified; reference and BH reconstruction
hash invariance recorded. Interior collar Z=3.33560e-7/1.71929e-7/1.36909e-7.
Compact evidence and reviews under finite-radius/refinement. Still awaiting
pointwise closure comparison, independent Z maxima and field timestep checks.
Transfers remain live: ghost5 session13730; interior session39265 (includes
remote checkpoint verification plus fine bulk geometry extraction). Preserve
these sessions; no successor allocation until paired analysis is audited.

While checkpoint transfers13730/39265 remain live, added separate native
cubed_sphere_conformal_maxima.csv with regional M/Z supremum magnitudes and
physical locations, using the same existing native constraint samples. No RHS,
reference, closure or old CSV changes. Frozen GPU sources/binaries untouched.
CPU six-case stationary/pulse/puncture x ghost/interior admission passes:
active fields and all existing diagnostic CSVs bitwise equal, reference
payload hashes identical; all312 new regional maxima rows finite and >=RMS.
CPU binary31dd3c86cf72e2891d70206c3f99b16018967665fcaf6f61c0e067534bfecd21.
Evidence outer-closure/native-maxima/acceptance.json and cpu-admission.json.
No GPU admission/build for this diagnostic yet; pending paired finite-radius
checkpoint analysis before choosing next allocation. CPU session83180 ended.

Ghost5 transfer13730 completed (archive870526643bytes). All four local
checkpoints hash verified. Independent finite-radius collar Z recounts pass
unchanged relative-native-RMS tolerance. Z maximum refinement and fine field
timestep comparison saved under finite-radius/refinement/ghost5-*. Interior
transfer39265 still live; no allocation. Do not restart completed ghost5
transfer or duplicate interior transfer.

Interior transfer39265 completed (869971686bytes); no live transfer/build/job
handles remain. All4 local payloads verified; all independent collar Z recounts
pass unchanged native RMS tolerance. Three matched closure field comparisons
and both fine timestep comparisons complete. Every nonzero component/region
closure difference decreases at R=.98 (versus failure at R=1). Collar Lambda_x
L2=1.77056e-6/1.16289e-6/7.50504e-7, orders1.46/1.96; time/closure5.8e-5.
BUT interior collar Z maxima1.18485e-6/4.62187e-7/4.65732e-7 are nonmonotonic.
Medium max at angular seam, fine face center; both radial depth2, and change
resolved against halfstep. Thus interior candidate still not accepted.
Paired conclusion/plot retained finite-radius/refinement/paired-conclusion.json
and paired-refinement.png/pdf, visually inspected. Supports investigating
nonuniform near-scri error, does not exonerate native PDE or prove BC cause.
Next: use admitted separate native-maxima diagnostics for complete regional
maxima and a controlled approach-to-scri study, preserving R=1 failed baseline;
consistent incoming characteristic BC/SBP remains to be derived. No next
allocation has been submitted or prepared by this turn.

After completed paired audit, predeclared next native-maxima GPU admission
and quintic fixed-R=.99 balanced/time-control study in native-maxima/
next-phase-acceptance.json. Diagnostic only: R=1 and physical targets remain.
Prepared isolated remote native-maxima-v1/source from preserved finite-radius
source; all450files hash verified, only conformal_probe.cpp changes (maxima CSV).
Source preparation session51953 ended. GPU build session64147 active, precise
O2 SYCL PVC+SERIAL; no allocation. Admission grids/PBS prepared locally at
research/outer-closure/native-maxima-gpu but not uploaded/submitted yet.
Wait on same build session; inspect logs if needed. Full local CPU six-case
bitwise admission is native-maxima/cpu-admission.json. Need GPU max-value
comparison in addition to existing fields/RMS/restart/ghost-independence checks.

Native-maxima build64147 revalidated live; initial configure reached Kokkos
SYCL checks. Added compare_native_maxima.py with predeclared absolute+relative
peak tolerance, exact regional coverage/count checks, in-domain positions,
finite maxima and RMS<=maximum. Self comparison and corrupted-M-maximum
negative control pass. These are checker controls, not GPU admission.
Prepared admission grids/PBS uploaded to native-maxima-v1, NOT submitted.
Need completed build+hashes+no-live-allocation check before qsub; then existing
field/RMS/restart/ghost controls AND new maxima CPU/GPU check. CPU golden at
research/outer-closure/native-maxima-cpu. No other live task processes except
build64147; completed transfer sessions must not be restarted.

Native-maxima GPU build64147 completed successfully. Binary
55f71c50f98e5e88ae916340482fd0070e9a7c414d818f4c8c640c5006e53a10.
All450source files plus input/script hashes verified; build/config logs saved
in native-maxima/. No-live-task allocation check passed before submission.
Admission job8925932 submitted capacity/CompactBinaryMerger/one node/30min;
only task allocation. Build/verification/submission sessions all completed.
Next audit six base + three followup cases; compare with CPU golden native-
maxima-cpu using compare_closure_admission.py, compare_native_maxima.py and
review_closure_followup.py. Preserve original inputs/reference hash receipts.
R=.99 refinement remains gated on admission; not submitted.

Admission8925932 revalidated queued. R=.99 four-case quintic balanced/time
inputs and one capacity PBS prepared locally (not submitted/uploaded):
research/outer-closure/native-maxima-gpu/r099-grids.json and run_r099.pbs.
Hashes in native-maxima/r099-preparation.json. Must await admission field,
RMS,maxima,restart,ghost checks before this successor. No duplicate job.

Native-maxima admission8925932 F/0. Download/remote hash review active
session91528; do not restart it. Result root native-maxima-v1/results/
8925932.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov and .followup.
CPU golden research/outer-closure/native-maxima-cpu. Need three validators:
compare_closure_admission.py, compare_native_maxima.py, review_closure_followup.py.
R=.99 input/PBS now uploaded but NOT submitted; no live task allocation.
Do not advance before admission results are checked.

Admission transfer91528 completed (190503868bytes). All three validators pass:
CPU/GPU fields/RMS, native M/Z maxima, and exact restart/ghost independence.
Compact admission evidence124files retained native-maxima/admission/manifest.
After no-live-allocation, priorF/0 and all450source/binary/input/PBS hash checks,
R=.99 quintic three balanced+finehalf matrix submitted as8925944. Capacity,
CompactBinaryMerger,one node,30min; only task allocation. Submission98149 ended.
No build/transfer sessions remain. Next collect job8925944 evidence, verify
native RMS AND regional M/Z maxima convergence/location and timestep effects;
compare with R=.98 and R=1 preserved diagnostics, not a substitute physical goal.
Need add analysis of new maxima CSV to fixed-region review (global=max of bulk/
shell all; layer=radial6..11; other fixed subsets unchanged). Criteria predeclared
native-maxima/next-phase-acceptance.json. Do not duplicate job8925944.

R099 job8925944 revalidated queued. Added review_native_maxima_refinement.py
for native M/Z maxima across global, bulk, shell, fixed interior/interface/
collar and existing layer-bin aggregate; saves maxima locations, slopes and
time/spatial differences. Added explicit reached_time_target check after a
negative control exposed that outcome=complete can mean step-only completion.
Negative control now rejects four-step admission; no runs changed.
Explicit diagnostic limitation: old "layer" aggregate is radial6..11, i.e.
.375<=r<.75 for unit scri, not whole transition .35<=r<.75. Global/shell
include the omitted strip, but exact-transition regional convergence needs
separate diagnostic; do not call current layer-bin screen full-layer proof.
Reviewer positive full-matrix application pending job8925944 evidence.

R099 job8925944 F/0, all4 endpoints t=.005. Compact results/review/native
maxima under native-maxima/r099. All required RMS screens and all reported
regional M/Z maximum monotonic/time checks pass. Collar finest Mmax order.981,
Zmax order.752 (slower than finiteR=.98 Zmax order2.34). All4 remote checkpoint
hashes verified and references/BH initial reconstructions invariant. No local
payload download started; no live job/build/transfer handles remain. New
conclusion.json states limited finite-radius interpretation. RMS plot legend
now says layer bins6–11; reviewer explicitly records exact-transition gap.
Next decisive work: diagnose approach-to-scri with controls (e.g. R=.995 matrix,
plus existing-R=.98/R=1 binary controls), and/or source-term/closure isolation;
predeclare before submission. Do not repeat completed R099 job.

User redirected priority to longer Rb=.98 finite-radius stability for puncture
and pure angular pulse. Scheduler audited: no task allocation; unrelated
cz_lowres_long8925878 and held telegraph jobs left untouched. Existing .98
checkpoints use residual-subtracted RHS; preserved all frozen sources/results.
New local native path removes analytic Minkowski RHS residual subtraction,
raises tlim ceiling1000/maxcycles1e7, implements normalized pure quadrupolar
relative lapse A=.2,width.35, adds exact fixed_transition diagnostic. CK02
prevents mixing old RHS policy checkpoints; angular flag encoded in signature.
Predeclared duration/failure/stationarity criteria finite-radius-long/acceptance.
First CPU launch failed before evolution for unregistered input option; fixed
input template, preserved failed run. Three four-step native CPU cases pass;
all three 4+4/fresh8 checkpoints bitwise equal, immutable reference/BH initial
hashes invariant. Evidence finite-radius-long/cpu-admission-review.json.
No active local sessions/jobs after27915 ended. No new GPU build/submission.
Remaining prerequisites: horizon/mass/lapse/radial geometry diagnostics, metric
SPD failure checks, validated bounded chain orchestration and GPU admission.
New source binary hash recorded in review; do not resume old CK01 with CK02.
Original near-scri convergence and full physical goal remain unresolved.

User explicitly requested completing diagnostics/GPU admission and starting both
long Rb=.98 runs. Scheduler re-audited: unrelated cz_lowres_long8926712 queued,
held telegraph jobs; left untouched, no task allocation. Added probe_surfaces.hpp
and native CSV surface area/Hawking mass/outgoing expansion/lapse/proper radial
factor/accumulated distance, spherical horizon candidate with angular residual
and doubled-quadrature error. Bulk-only cubic diagnostic interpolation uses
active nodes; initialM=.5 throat r=.250004, areal~.999946, mass~.499981,
quadrature area error7e-7. CPU admission passed. SPD principal-minor checks
added to stage/projection/diagnostics: stop only, no clipping/floors.
Added regional gauge RMS for settling, exact transition already present.
CPU split restarts all3 cases exact; final gauge-output addition leaves fields
bitwise unchanged. CK02 native unsubtracted evolution retained. CPU golden
research/long-final-cpu; restart evidence long-surfaces-restart-cpu.
Isolated remote long-native-surfaces-v1 all451files verified, GPU build8749
ACTIVE; no allocation. Admission3cases +6fresh/resumed followups prepared and
uploaded run_admission.pbs, admission-grids.json, prepare_followup.py. State
finite-radius-long/gpu-build-state.json. Need poll same build; then hashes,
no-task-allocation gate, qsub capacity single node30min. No duplicate build.
Remaining: GPU surface/gauge comparison and exact CK02 split restart; bounded
stage runner/monitor with failure criteria; start paired long stages after
admission. Spherical horizon candidate is NOT yet a validated general apparent
horizon; require residual/refinement and do not claim trumpet prematurely.

GPU build8749 completed, binary1813952328dcfa9a00cdabc8c325c6b6c43f6eb5f65c99b2696b0bebdd028036.
All451source files and prepared admission scripts checked; no task allocation
before qsub. Native GPU admission8926769 submitted capacity/account/node/time
correct, only task allocation. Build/submission observation sessions ended.
Added run_finite_radius_long_stage.py external staged runner: CK02 payload/header/
shape/source/binary/reference-fingerprint checks; full regional H/M/Z RMS and
maxima thresholds, equal .02-time-window growth monitor, horizon-candidate
loss/mass drift monitor, gauge diagnostics. CPU checkpoint/monitor admission
and injected-failure negative control pass. Stage script/PBS uploaded and hashed
finite-radius-long/stage-preparation.json, NOT submitted before admission.
First paired stage targets .02, B24/A18 punctureR123/pulseR72, quinticRb=.98,
CFL.1/pole.03/no fixed_dt,300s wall per branch. All later signatures immutable.
Admission audit still needed: base3 CPU/GPU fields/RMS/maxima + compare_long_
diagnostics.py; all3 fresh8/resumed4+4 whole checkpoints exact and ref invariance.
Admission CPUgold long-final-cpu. GPU result paths results/8926769... and.followup.
No result transfer started yet. Poll job8926769, do not duplicate it.

User requested timed sleeping to conserve tokens. Admission8926769 revalidated
Q. Created thread heartbeat continue-hyperboloidal-long-runs, every5minutes,
ACTIVE, quiet on unchanged state. It is authorized to validate admission and
launch/continue both prepared long stages without confirmation, maxonecapacity
allocation. Timer record finite-radius-long/monitor-timer.json. No new jobs.
Do not create duplicate monitors; use this timer for waits instead of repeated
status-only turns. Overall physical/convergence requirements remain unchanged.

2026-10-14 13:14UTC: admission8926769 F/0 in capacity. GPU field/RMS,
native maxima, surface/horizon/gauge admission PASS. Surface radius identity
comparator corrected from literal string equality to tight numeric tolerance
(1ULP discrepancy); diagnostic value tolerances unchanged. All3 GPU fresh8
vs resumed4+4 whole CK02 SHA exact. Remote451source hashes and24 reference/
BH-initial payload hashes verified. Evidence gpu-*-admission.json and
gpu-remote-integrity.json. Transfers83601/15459 and audit53755 all terminal.
Prepared stage script/binary hashes reverified; no task allocation before qsub.
First paired long stage8928317 submitted13:13:40UTC capacity,1node,30min,
CompactBinaryMerger; Q verified. Remote long-stage01-submission.json and local
stage01-submission.json authoritative. No duplicate jobs. User now permits
debug-scaling/additional nodes when needed; present paired adapter stage stays
1node capacity. Timer already hourly; updated prompt to new stage handle.
No long-time stability/convergence claims yet. Validate all monitors/checkpoints
before successor, preserve native equations and immutable Minkowski reference.

2026-10-14 16:35UTC: stage01 job8928317 F/0, both branches t=.02. All monitors passed; independent remote451source/checkpoint/reference/initial hashes passed, evidence stage01-integrity-audit.json. Audit first script comparison failed tuple-vs-JSON-list metadata equality only, corrected normalization and preserved log. No physics change. Puncture shell Hmax .50460885 at r~.423 outside candidate r=.251256 (expansion max9.33e-5); stage01-localization.json. All active cells monitored, no horizon exclusion. Stage02 job8928904 submitted after terminal/no-other-task-allocation gate, capacity1node30min, previous stage01 CK02, target .1 both. Local/remote authoritative submission records preserved; timer updated. No long-time stability or accuracy claim.

Stage02 job8928904 F/1: puncture completes t=.1 but monitors FAIL (shell Hmax1.65975, shell H_RMS.18129 M_RMS.28350 Z_RMS.01136; absolute/growth/doubling failures). Pulse wall-checkpoint t=.06805936226 passes monitors. Do not submit blind continuation of failed puncture. User requests constraint slice and now permits debug queue for future jobs. Isolated diagnostic-only CPU binary in research/constraint-slice (original source and frozen GPU binary unchanged) adds per-active-cell native H/M/Z output, intended checkpoint measurement with zero evolution. Transfer62506 stage02 active; compile49018 terminal. Timer updated failure gate.

Constraint slice delivered: constraint-equatorial-stage02.png/pdf, puncture t=.1 only. Native H/M/Z read-only CPU checkpoint reconstruction, all puncture regional RMS/max constraints agree GPU endpoints within admission tolerances. Original GPU reference fingerprint differs CPU (signature/tag match); isolated diagnostic reader enforces tlim=time/requested_steps=cycle, retains FNV/header checks, returns immediately after measure. Production guard/source/GPU binary unchanged. Plot averages two adjacent equatorial cell-plane magnitudes, no horizon exclusion; cyan interface/green candidate. Pulse seven regional discrepancies unresolved, plot withheld (constraint-slice-validation.json retains failures). All transfer/build/extraction/plot handles terminal. User authorizes debug queue for future jobs; failed puncture stage02 must be diagnosed before further physical continuation.

2026-10-14 17:03UTC: no task allocations verified, debug queue1node30min permitted. Predeclared puncture-timestep-diagnosis/criteria.json: reproduce native failure fresh to.1 and half both CFL/poleCFL on identical B24/A18/R123 grid,Rb.98; compare all regional norms/max to archived resumed stage02 and growth fromstage01. No equation/reference/closure changes. All451remote sources/frozenbinary and scripts reverified. Diagnosis job8928969 submitted debug1node30min, authoritative local/remote puncture-time-submission.json. Review script review_puncture_timestep_failure.py prepared, not yet applied. No blind puncture successor; pulse checkpoint preserved. Timer updated to diagnosis handle.

2026-10-14 18:07UTC: diagnosis8928969 F/0, fresh native puncture failure reproduces at.1. All regional timestep norm/max differences <.2813%, all restart/milestone differences <.000235%; review.json plus timestep-control.png. Remote source/binary/ref/initial/checkpoint integrity verified. No timestep/restart explanation for failure. Predeclared balanced-spatial diagnosis uses coarse16/12/82 andfine32/24/164 versus existingmiddle24/18/123(no duplicate). Same quinticRb.98/reference1/native equations; common.1 target,300s wall percase, unequal endpoints cannot claim convergence. Independently validated passing pulse stage02 CK02/reference fingerprint/hash/376source hashes and monitor history before continuation toward.1. Mixed job8929122 submitted debug1node30min, no other task allocation; authoritative spatial-pulse-submission.json / local puncture-spatial-diagnosis/submission.json. Timer updated. Keep failed puncture baseline; no blind physical continuation.

2026-10-14 19:06UTC: job8929122 F/0; coarse puncture t=.1, fine checkpoint .0523838191, pulse t=.1. No common-time three-grid convergence claim yet. Pulse per-branch monitor now FAILS (pulse-monitor.json): outermost Mmax3.878e-5 vs baseline3.458e-7, collarHmax6.606e-6, multiple equal-window doublings. Finite fields/no metric blowup; conservative growth gate is distinct from physical instability. No blind pulse or puncture successor. All parent source/binary/checkpoint/reference/initial hashes independently verified remote spatial-pulse-integrity.json(local integrity.json). Fine completion controlled-diagnosis job8929327 submitted debug1node30min after no-task-allocation gate; validated parent CK02 SHA19bb59db.../fingerprint6920210541828223261. Predeclared unchanged spatialcriteria; finish.1 and apply review_puncture_spatial_failure.py across coarse/middle/fine and halfdtmiddle. Review script py_compile passed; timer updated. Diagnose pulse timestep/spatial/background controls before physical continuation.

2026-10-14 20:05UTC: fine completion8929327 F/0, common t=.1. Source/binary/CK02/fingerprint/reference/BHinitial integrity passed. Three-grid review: shell H_RMS .356774/.181293/.100582,M .484579/.283503/.172518,Z .0203786/.0113639/.00626693, improve; six fixed-region quantities nonmonotonic (bulk transition M_RMS,bulk/all/interface/transition Mmax,inner-shellHmax/Mmax). Full convergence criterion FAILS. Temporal differences <10% spatial differences all fixed regions. All sampling/minOmega/cell volumes in review.json; figure balanced-constraints-t01.png/pdf, script plot_puncture_failure_convergence.py. No physical puncture successor. Predeclared pulse-growth-diagnosis/criteria.json coarse/fine/halfdt/stationary controls at common.1 using unchanged native source/reference and original growthgate. New external wrapper only, py_compile passed, max1200s bounded native wall perbranch under30minallocation (checkpoints may require continuation). All451source hashes/frozenbinary/scripts reverified, no task allocation before qsub. Job8929531 submitted debug1node30min, authoritative pulse-growth-submission.json/localsubmission.json, timer updated. No physical stability or trumpet claim.

2026-10-14 21:02UTC: pulse-control8929531 F/1 BEFORE EVOLUTION: external runner requested wall1200 but frozen native cap360. Orchestration error acknowledged; no scientific failure/refmutation result inferred. Failed matrix preserved research/pulse-growth-diagnosis-failed and originalcriteria. v2 wall300/timeout420/overall540, physical criteria/native source/binary unchanged. Added --check-only preflight: positive300passes/negative1200rejects, v2/preflight.json, py_compilepasses. All451remote sources/binary/input/scripts reverified; no taskallocation before qsub. Corrected job8929698 submitted debug1node30min; authoritative pulse-growth-v2-submission.json/localv2submission.json; timer updated. Diagnostic checkpoints may need multiple continuations to common.1. No stability/convergence claim altered. All failed-transfer/admission/submission sessions terminal.

2026-10-14 22:06UTC:8929698 F/0,coarse pulse reaches.1; fine .011282454,halfdt .015658973,stationary .033002446 checkpoint finite/refimmutable. No common-time3-grid claim. Coarse/middle shellH/M/Z RMS drops4.4–4.7 at.1; stationarynativeH/M~1e-12 through.033 (preliminary, no RHSsubtraction). v2endpoint-status.json. All451sources/binary/CK02/refpayloads verified remote pulse-growth-v2-integrity.json(localv2integrity). External continue_pulse_diagnostic_segments.py py_compile/preflight actualparents passes, up to3 validated300s segments perbranch in same30minallocation; nativecap unchanged360. Completedcoarse notrerun. Job8929890 submitteddebug1node30min after no-task-allocationgate, rootpulse-diag-cont01-submission/localcontinuation01submission authoritative. Eachsegment immutablefingerprint/refhash and CK02 SHA checked, native restart verifiesFNV; stop on failure. Timerupdated; native source/reference/physicalcriteria unchanged. Preserve persegment histories whencollecting; no unequal-time convergence/no physicalfailedbranchcontinuation.

2026-10-14 23:10UTC:8929890 F/0,9 validated diagnostic segments. Finepulse.0438054254,halfdt.0646122593 incomplete,stationary reaches.1. BackgroundH/MRMS~1e-12, maxfielddrift1.54e-11 noRHSsubtraction; source/CK02/all9segmentinputs/ref fingerprints verified, latestrefpayloads rehashed, continuation01/integrity.json. Plot stationary-vs-angular-t01.png/pdf via plot_pulse_background_control.py inspected. No3-grid/timeconvergence yet. Completedcoarse/stationary reused. Onlyfine/halfdt successor job8930039 submitteddebug1node30min afterpreflight/source/binary/script/no-taskallocationgate, continuation02/submission.json authoritative. Same max3native300s segments, unchangedsciencecriteria/nativeequations/reference. Timerupdated; preservepersegmentpaths/histories fornextaudit. Alltransfer/audit/submission handles terminal. No physicalstabilityacceptance.

2026-10-15 00:07UTC:8930039 F/0,6segments. Halfstep reaches.1; every regional norm/max differs<.035565% vs archivedmiddlepulse (includes restartmilestonetruncation), timestep-review.json. Finecheckpoint .0781358375 remains. Normdefinitions regional-norm-definitions.json: mappedcompactcoordinate-volume weights,notphysicalpropervolume; M/Z conformalnorms notOmega-suppressed, no horizon exclusion. Allsegment source/input/CK02/fingerprints/latestreference payloads verified, continuation02/integrity.json. Onlyfine successor8930135 submitteddebug1node30min afterpreflight/no-task-allocationgate; continuation03/submission.json authoritative; nativeequations/reference/scientificcriteria unchanged. Common.1 spatial/time/background reviewscript review_pulse_spatial_controls.py prepared andpy_compilepassed, fine notyetcomplete so no3-gridconvergenceclaim. Reusecompletedcoarse/middle/halfdt/stationary. Timerupdated; alltransfer/audit/submission sessions terminal.

2026-10-15 01:06UTC:8930135 F/0 finepulse completes.1,2segments/source/input/CK02/fingerprint/latestref integrity passed. Pulsecommon.1 spatial-review: ALL reported fixed AND radialH/M/Z RMS/max decrease16/24/32, alltemporalerrors<10%spatialdifferences,backgroundnegligible. Shellfinest RMSorders3.94/3.90/3.91; originalfinegrowthgateSTILLFAILS (fine-original-monitor.json), no longstability/near-scri acceptance. Plots balanced-constraints-t01.png/pdf via plot_pulse_control_convergence.py, conclusions-t01.json explicitlylistsunsupportedclaims. Puncture remains6nonmonotonicfixedquantities; predeclaredpuncture-localization-controls freshbalanced40,bulk-only32,shell-only24 atcommon.1 vsreusedbaseline24 andbalanced32. Nativeequations/reference/quinticboundary unchanged; newexternalrun_native_diagnostic_segments.py fresh/resume logicpy_compile/preflightpasses,max3validated300ssegments. All451sources/binary/scripts verified,no taskallocationbeforeqsub. Job8930229 submitteddebug1node30min; authoritativepuncture-localization-submission/localcontrols/submission. Timerupdated; incompletecontrolsneedvalidatedcontinuation. Preserveall4balancedgridresults/normregions, no cherry-picking originalfailures; no physicalstabilityclaim.

2026-10-15 01:58UTC:8930229 F/0; balanced40checkpoint.0666262145(3segments),bulk-only32.1(1),shell-only24.1(2). All6segmentinputs/source/CK02/fingerprints/reference/BHinitial unchanged; latestpayloads rehashed, controls/integrity.json. Matched.1 independentrefinement: shellonlywholeshellH/M/Z RMSreductions44.05/39.07/44.85%,bulkonly<.2%; bulkonlyinnerHmaxreduces72.43% butbulkinterfaceMmaxworsens25.21%,shellonlyinnerMmaxworsens31.59%. Coupledresolution/reconstruction sensitivity; NOTsoleboundaryfault. Allregionalvalues/peakcoords/counts/guide inlocalization-review.json, plot independent-refinement-t01.png/pdf via plot_puncture_localization.py. Onlybalanced40 successor8930320 submitteddebug1node30min afterpreflight/no-taskallocationgate; continuation01/submission.json authoritative. Reusecompletedcontrols, samephysics/nativeequations/reference/quinticclosure. Finishcommon.1 before4grid/last3review, preserveoriginalN16nonmonotonicfailuredata. Timerupdated; alltransfer/audit/submission handles terminal. No longphysicalacceptanceclaim.

2026-10-15 03:05UTC:8930320 F/0,balanced40 finishes common.1 in two segments. Independent source/binary/input/CK02/reference/BHinitial verification passes, continuation01/integrity.json; finalCK SHA71f9aa9c82f582a164a82659ae081103425ffad1005e56c6cdef75a8d8aa231d. N24/32/40 resolves previous six interface/transition nonmonotonic quantities, but FOUR outer-collar H/M RMS/max now nonmonotonic; Z decreases. CollarH_RMS4.73899e-5/1.71885e-5/2.27451e-5,M_RMS7.16503e-4/3.09528e-4/4.42721e-4. Full convergence FAILS; time error<10% spatial differences. All4grids retained in four-grid-constraints-t01.json/png/pdf via plot_puncture_four_grid.py, no horizon exclusion. Evolved N40 .1 all5400 last-radial cells have incoming physical logical-normal speeds[-.005071,-.004947],outgoing[1.8178,1.8191]. Four selected evolved-state principal-symbol decompositions also have incoming physical2/gauge4/constraint4 modes, invariant-subspace leaks<5.3e-14; these are sampled classification, not all-cell gauge/constraint proof. Evidence boundary-speeds-*.json and boundary-symbol-logical-review.json. First symbol invocation wrong sample suffix failed before analysis; corrected existing emitted filename, no scientific changes. Interior-only derivative closure is NOT a complete characteristic BC or SBP scheme; quintic ghostD2 already equivalent six-node one-sidedD2. Predeclared puncture-collar-controls isolates radial-only32 B32/A24/R205,angular-only32 B32/A30/R164,interior32 andinterior40 at matched .1,Rb.98,closure boolean only. Existing frozen compiled closure, native source/binary unchanged. Two closure resolutions are sensitivity only, thirdgrid/timestep controls still required. All451sources/binary/preflight/scripts verified,no taskallocation beforeqsub. Newauthoritative8930422 submitteddebug1node30min,CompactBinaryMerger; Q verified, controls/submission.json andscheduler.json. Max3 validated300ssegments, stop invariant/native failure, resume only validated incomplete endpoints. Do not advance failed physical runs blindly or relax original pulsegrowthgate. Longstability/trumpet/originalnear-scri requirements unmet. Hourlytimer updated; no duplicate jobs.

2026-10-15 04:04UTC:8930422 F/0,11 validated segments. Radial-only32/angular-only32/interior32 complete common.1; interior40 checkpoint.0668920362 only. Independent remote451source/binary/input/CK02/header/referencefingerprint and latestref/BHinitial rehash PASS, puncture-collar-controls/integrity.json. First independent audit hit tuple-vs-JSON-list metadata comparison; normalized serialization, physical/checkpoint tolerances unchanged. Native restart verified intermediate CK02 FNV; incomplete latest will undergo native-reader check on successor. Atcommon.1 radial-onlyR205 increases collarH/M RMS32.3%/43.0%,angular-onlyA30 changes RMS<.01%,max<1.3%; radial-angular-review.json. Radial-only reproduces balancedN40 collar behavior, angular/bulk changes small there, but radial refinement changes sampling/minOmega and evolved solution: not unique boundary causation or PDE exoneration. Interior32 closure worsens collarH/M/Z RMS83.9%/78.4%/30.0%,H/Mmax~101%/104%; closure32-preliminary.json,collar-sensitivity-t01.png/pdf via plot_puncture_collar_controls.py. No evidence of improvement from current interior-only closure. Preserve originals/allregions/horizoninterior. Additional criteria predeclared continuation01: resume onlyinterior40; freshinterior24 completes3closuregrids andradial-half-step/interior32-half-step tests temporal contamination. Completed prior controls notduplicated. All451sources/binary/scripts/preflight/no-taskallocation gate passes. Newauthoritative8930665 submitteddebug1node30min,CompactBinaryMerger; submission/scheduler saved. Max3validated300ssegments,420s processcap,1700s wrapper; bounded incompletecontrols require validatedsuccessors. review_puncture_closures.py prepared/py_compilepasses: allfixed/radialnorms/max andsampling/minOmega,closure differencesmustdecrease24/32/40,temporal<10%spatial; notyetapplied pendingcommon.1. Nativeequations/referenceunchanged,interioroperatornotSBP/notcompleteincomingBC. Longphysicalrunsremainheld at failedmonitors; no stability/trumpet/originalnear-scri acceptance. Hourlytimer updated,no duplicatejobs.

2026-10-15 04:58UTC:8930665 F/0,nine validated segments; interior40/interior24 reach.1,halfdt radial.0528083011/interior32.0789242196 incomplete. Source451/binary/input/CK02/header/fingerprint/latestMinkowski/BHinitial payload hashes independently verified, continuation01/integrity.json. Allthree interiorclosure resolutions now common.1: collarH_RMS4.57048e-5/3.16032e-5/4.12830e-5,M_RMS7.40385e-4/5.52193e-4/8.10344e-4 atN24/32/40; H/M remainnonmonotonic. Every collar closure-difference sequence fails monotonic decrease overthreegrids, although some differencesdecrease32->40. Zconstraints decrease butabsoluteZclosure differencesnonmonotonic. Preliminary spatial only, timestepcontrols pending: closure-spatial-preliminary.json/closure-spatial-t01.png/pdf, plot_puncture_closure_spatial.py. No evidence for accuracyimprovement orcompleteincomingBC frominteriorstencil. Preserveallfailednorms andallregions. Onlytwo incompletehalfdt controls successor8930791 submitteddebug1node30min,CompactBinaryMerger after parentintegrity/preflight/source/binary/no-taskallocationgate; continuation02/submission.json authoritative. Max3validated300ssegments,checkpointcontinuationonly,scientificcriteriaunchanged. Reusecompletedinterior24/32/40 andallquinticcases; fullreview_puncture_closures.py waitscommon.1, includesobservedorders/counts/minOmega/timeerrorvsdifferences. Alltransfer/audit/submissionhandles terminal; hourlytimerupdated. Overallphysicalgoal remainsincomplete; no longphysicalsuccessor untildiagnosissupports it.

Pulse growth-trigger interpretation, predeclared before new propagation phase: a constraint-free continuum pulse can generate discretization error C_h obeying dC_h/dt=L(t)C_h+h^p S(t), initially near zero. A ratio to an early near-roundoff baseline need not bound the later forced solution. The observed .1 three-grid order supports a truncation-error contribution, but an unstable mode seeded at orderh^p can also converge at any short fixed time; thus no harmlessness/stability inference is justified. A bounded .1->.2 diagnostic compares all regional growth factors, observed orders and h^-p-scaled errors at common times against stationary control, while preserving original100x/doubling failures. New10x-over-.1 concern andabsolute1 stop are diagnostic limits, not replacements for the original physical acceptance criteria. Timestep andboundary-sensitivity controls at.2 remain necessary before accepted longcontinuation. Criteria in finite-radius-long/pulse-propagation-diagnosis/criteria.json were saved/hashed before submission; allfour parents are existing .1 checkpoints, no completedcontrols rerun.

2026-10-15 05:56UTC:8930791 F/0,four native segments finish bothhalfstepcontrolscommon.1. Independent451source/binary/input/CK02/header/fingerprint/latestMinkowski/BHinitial hashes pass continuation02/integrity.json. Closurefullreview reproduces spatialfailure:19fixed-region closure-difference sequencesfaildecrease,4fixedcollarH/Mconstraintsequencesnonmonotonic. Collar temporalcontrols small(<0.84% ofcomparison scales),thusnotexplanation. Conservativeallfixedtime-vs-closuregateFAILS forverytinyclosureeffectselsewhere;theseunresolvedeffectsarenotclaimedmeaningful. closure-review.json/conclusions-t01.json/review-command.sh/closure-review-t01.png preserveallvalues,orders,sampling,minOmega,failureflags. No adoptionofinteriorclosure;notSBP/notincomingcharacteristicBC. Newboundedpulsepropagationdiagnosis8930878 submitteddebug1node30min,CompactBinaryMerger,targetcommon.2 forcoarse16/middle24/fine32 andstationarymiddle24,allfromexisting.1CK02. Allfourparentcheckpoint/source/binary/reference/initialpayloadhashes reverified,pulse-propagation-diagnosis/parent-integrity.json. Predeclaredcriteria saved/hashed,newexternalrun_native_pulse_propagation_segments.py onlychangesdiagnostic targetto.2,requiresrestart,addsabsoluteallregionalH/M/Z<=1stopbeforesuccessor;nativeequations/source/binaryunchanged. Actualparentpreflightpasses,no taskallocationbeforeqsub;submission.json authoritative. Max3validated300ssegments,420sprocesscap,1700swrapper;incompletecontrolsneedvalidatedcontinuation. OriginalpulsegrowthgatesremainFAILED andmustbereappliedwithhistory;diagnostic doesnotestablishacceptedlongstability. Requireallfixed/radialconstraintsdecreaseatcommon.2,reportgrowthfactors/orders/h^-p-scalederrors;>10x finestfixednorm/maxgrowthover.1 triggersdiagnosisbeforefurtherdiagnosticstage;stationaryallnorms/max<=1e-8 andmaxfielddrift<=1e-8. Theseareadditionaldiagnosticlimits,notrelaxedphysicalacceptance. Timestepandboundarysensitivityat.2remainunmeasured/required. Hourlytimerupdated;alltransfer/audit/submissionhandles terminal. Physicaltargetsandoriginalnear-scriconvergenceunmet.

2026-10-15 06:52UTC:8930878 F/0,tenvalidatedsegments; coarsepulse/stationaryreach.2,middlecheckpoint.1958772326,fine.1347286294. Independent451source/binary/input/CK02/header/fingerprint/latestreference/initial hashesPASS,pulse-propagation-diagnosis/integrity.json. No unequal-time spatialclaim. Prepared/executed review_pulse_propagation.py viareview-manifest.json:common_target_reachedfalse,spatialacceptancenull,no finestfixed10x concerns(maxendpointgrowth2.22118),stationaryallconstraintsmax6.084e-10/maxfielddrift2.055e-11withinpredeclared1e-8limits. Allregions/sampling/minOmega/peaks inpartial-review.json. Reappliedunchangedoriginalmonitorwithprecedinghistory viareplay_native_monitors.py andmonitor-replay-manifest.json:coarse/middle/fineFAIL39/29/19historicalflags,stationaryPASS;original-monitors.json preservesflags,notacceptedlongcontinuation. Onlyincompletemiddle/fine successor8930970 submitteddebug1node30min,CompactBinaryMerger afterparentintegrity/partialcriteria/no-taskallocation/source/binary/preflightgates;continuation01/submission.json authoritative. Reusecompletedcoarse/stationary. Samefrozennativeequations/reference/quinticRb.98,externalmax3validated300ssegments perbranch/420sprocesscap/1700swrapper. Onnextfinalextendbothreviewmanifests withnewsegmentpaths,histories retained,compareonlycommon.2. Common.2time/boundarysensitivityunmeasured andrequired;originalgrowthflagsremainfailed. Hourlytimerupdated;no duplicatejobs;alltransfer/audit/submissionhandles terminal. Overallphysicaltargetsandoriginalnear-scriconvergenceunmet.

2026-10-15 07:47UTC:8930970 F/0,fournativesegments;middlepulsecompletes.2,finecheckpoint.1677637726. Independent451source/binary/input/CK02/header/fingerprint/latestreference/initialhashesPASS,propagationcontinuation01/integrity.json. Additionalcollectedactualinput.athinput SHA andlaunch/receiptcommandidentitiescheckedforall14propagationsegments, collected-input-integrity.json. Common.2notyetavailable,nospatialclaim. Updatedcontinuation01/review-manifest.json/review.json:no finestfixed10xconcerns,maxpeakgrowth4.08239(collarZRMS),stationarylimitsPASS. Revieweralsorecordssecantlogrates/effectivetimepowers;thesearedescriptiveregionalnorms,notmodeeigenvalues,forcing/transportcanalterrates. Replayedunchangedoriginalmonitorswithallhistory:coarse/middle/fineFAIL39/30/27historicalflags,stationaryPASS,continuation01/original-monitors.json. Onlyfine successor8931054submitteddebug1node30min,CompactBinaryMerger,Qverified,afterparentintegrity/partialcriteriapass/preflight/source/binary/no-taskallocationgate. continuation02/submission.json authoritative. FineCK SHA c5b331425f31a65b1046852e8962b8539b86e84b319910529ce4af05a6783510. Reuseallcompletedcoarse/middle/stationary. Sameexternalmax3validated300ssegments,420sprocesscap,1700swrapper;nativeequations/source/binary/referenceunchanged. Nextfinalextendcontinuation01review/monitor manifestswithnewfinepaths;onlycommon.2comparison. OriginalgrowthflagsremainFAILED;time/boundarysensitivity.2required;nostabilityclaim. Hourlytimerupdated;no duplicatejobs,alltransfer/audit/submissionhandles terminal.

2026-10-15 08:47UTC:8931054 F/0,threefinesegmentsfinishcommon.2. Independent451source/binary/input/CK02/header/fingerprint/latestreference/initialhashesPASS,propagationcontinuation02/integrity.json;actualcopiedinputs/launchcommandsverified collected-input-integrity.json. ALLfixedANDradialH/M/Z RMS/max decrease16/24/32 at.2,whole-shellfinestordersH3.8197/M3.8012/Z3.9156,collarH4.09/M3.08/Z3.89. Finegreatestfixedgrowth6.55419(collarZRMS)over.1,below10diagnosticlimit;stationarymaxconstraint6.084e-10/maxfielddrift2.055e-11passes1e-8limits. OriginalmonitorsSTILLFAIL39/30/30historicalflags,stationaryPASS. Allvalues/orders/sampling/minOmega/growthincontinuation02/review.json,original-monitors.json;conclusions-t02.json explicitunsupportedclaims. Plot balanced-constraints-t02.png/pdf via plot_pulse_propagation_convergence.py inspected. Nophysicalstabilityclaim,nofailuretimescrossed. Newpredeclaredpulse-t02-controls:matchedinteriorradialclosures16/24/32fresh(nativeCKsignaturepreventsquinticrestart),andquinticmiddlehalfCFLresumedfromexisting.1halfstepparent. SameRb.98/minOmegaeachpair/20percentquad/nativeequations/source/binary/immutableMinkowski. InteriorisnotSBP/notcompleteincomingBC;sensitivityonly,noadoption. Requireallregionalconstraintsandclosuredifferencesdecreaseacross3grids;baselinehalftime<10%bothspatialdifferences;alternativeown temporal/background controlsneededifproposedadoption. ParenthalfstepCK/ref/source451/binaryreverified,pulse-t02-controls/parent-integrity.json;fresh/resumedexternalwrapper target.2/max3validated300ssegments/420sprocesscap/1700swrapper/absoluteallregional1stop. Allscript/inputhashes/preflight/no-taskallocationgatepasses. Newauthoritative8931108submitteddebug1node30min,CompactBinaryMerger,Qverified,pulse-t02-controls/submission.json. No completedquinticcontrolrerun. Futurepointwisestateclosurecomparisonusingcompare_closure_field_norms.py distinguishnormofdifferencefromdifferenceofconstraintnorms. PreparedpostprocesssupportCFLvariation13/14,adaptivecyclecountsallowedatmatchedphysicaltime,nonbinaryreferencefingerprint/tagmatch;py_compileonly,notyetappliedto.2data. Requirescheckpointsandbulkgeometryextract,doesnotreplace nativeH/M/Z norms. Hourlytimerupdated;alltransfer/audit/submissionhandles terminal. Overallgoaloriginalnear-scri/longpulse/trumpetunmet.

2026-10-15 09:43UTC:8931108 F/0,elevenvalidatedsegments. Interior-coarsecomplete.2;interior-middle.0955669593/interior-fine.0340510983/quintic-half-step.1485569363 incomplete. Independent451source/binary/actualinput/launchcommand/CK02/header/fingerprint/latestreference/initialhashesPASS pulse-t02-controls/integrity.json. Originalmonitorreplaywithallrelevanthalfstephistory preserves41/17/0/22flags respectively;fineearly.034passingnotastabilityclaim. No unequal-timecomparisons. Onlythreeincompletecases successor8931168submitteddebug1node30min,CompactBinaryMerger,Qverified;continuation01/submission.json authoritative. Newexternalv2wrapper increases max3->4validated300ssegments only;lastallocationused15m48s,expected4~21min,1200sstartdeadline/420sprocesscap/1700souterwrapper retained. Nativeequations/source/binary/solverparametersunchanged;max4preflightpasses,allscript/grids/criteriahashes/source/no-taskallocationgatespass. v1wrapperandevidencepreserved. Scientificcriteriaunchanged,plusexplicitpointwisestatecomponentclosure-difference consistencyreporting incontinuation01criteria:constraintnormdifferencesarenotnormofpointwisedifferences. compare_closure_field_norms.py preparedCFL/physicaltime/referenceguardsandfullbulk/shelltransitionmasks,nativevolumechecks retained;py_compileonly,actual.2fielddata-validationpending. Coarsepaired.2canbeusedforread-onlyfieldcomparison/extractionwhilefinecompletes,butdoesnotestablish3gridconsistency. Allcompletedquinticandinteriorcoarsereused;no duplicates. Baseline.2allregionalnormsconvergebutoriginal39/30/30growthflagsfail,longstabilityandboundaryconsistencyunproved. Hourlytimerupdated;alltransfer/audit/submissionhandles terminal.

2026-10-15 10:46UTC:8931168 F/0,twelvevalidatedsegments,wall21m04swithin30min. Interior-middle andquintic-half-step reach.2;onlyinterior-fine remains.0795110407. Source451/binary/actualinput/launchcommand/CK02/header/fingerprint/latestreference/initialhashesPASS continuation01/integrity.json. Originalmonitorreplay41/37/4/30flags preserved;no physicalacceptance. Baseline.2timestepreview COMPLETE:allfixedANDradialnorm/max temporal differences<10%bothspatialdifferences,maxrelative.0661787%,backgroundnegligible. review_pulse_spatial_controls.py now--time option(default.1preserved),timestep-review.json. Onlyincompletefine successor8931221submitteddebug1node30min,CompactBinaryMerger,max4validated300ssegments/420sprocesscap/1200sstartdeadline/1700swrapper;continuation02/submission.json authoritative,Rverified,noother taskallocation beforeqsub. Allcompletedcoarse/middle/half/baselinesreused. Read-onlyGPUcheckpointfields downloaded/sha-verifiedcoarse/middlepairs+half. Referencebulkgeometry extractedhash-verified withCK02-compatibleextractor,nativequadraturecounts/volumescrosschecked. compare_closure_field_norms.py actual.2data-validationpasses3comparisons(field-coarse-t02.json,field-middle-t02.json,field-timestep-t02.json),includingfulltransition/seampartitions. LargestcollarcomponentRMS closure differences5.71554e-7(N16),3.59117e-7(N24);halfstep2.34552e-10. Onlytwo closuregrids,not3gridconsistency;bulkclosureeffects~1e-12 smallerthan timestepdifferences~1e-10 so unresolved. Pointwisestate normsnotconstraintnorms/invariantfieldnorms. Actualangularpulse.2characteristics NOWcheckedusingmiddleGPUCK:all1944lastradialcellsincomingphysical[-.000436875,-.000428234],outgoing[3.89021,3.94731];five selectedevolvedstatesreference-wave-map principalphysical/gauge/constraintincomingbranches confirmed,subspaceleak<=1.29e-14. Evidence pulse-boundary-speeds-t02.json,pulse-boundary-symbol-t02-review.json,pulse-symbol-provenance-t02.json;symbolbinary/currentfrozenheaderhashesverified. Speedcollar r>.95(endoflayer),explicitthreshold recorded;nativeconstraintcollar remainsr>.9. DoesnotjustifyalloutgoingBCorstatewiseconsistency. Minkowski reference/nativeequations unchanged. Hourlytimerupdated,no duplicates;alltransfer/audit/submissionhandles terminal.

2026-10-15 11:44UTC:8931221 F/0,fourvalidatedfinecontrolsegmentsadvance to.1256510136. Independent451source/binary/actualinput/launchcommand/CK02/header/fingerprint/latestreference/initialhashesPASS pulse-t02-controls/continuation02/integrity.json. Fullhistoricaloriginalmonitorreplay flags41/37/25/30 preserved,continuation02/original-monitors.json. Allcoarse/middle/half/baselinesremaincompleted.2,reused;no unequal-timecomparison. Onlyfine successor8931289 submitteddebug1node30min,CompactBinaryMerger,afterparentintegrity/source/binary/grids/scripts/preflight/no-taskallocationgates,continuation03/submission.json authoritative;stateRverified. ParentCK SHA be8581e7053195f8036a8f5c6278a7503d25bd1e0f670b2347339c272484b0c0. Max4validated300ssegments,420sprocesscap/1200sstartdeadline/1700swrapper unchanged. Nativeequations/source/binary/referenceunchanged,nofloors/clipping/resets/RHSsubtraction. Three-gridclosure.2comparisonpendingfineendpoint,originalgrowthgatesunresolved,nolongphysicalacceptance. Hourlytimerupdated;alltransfer/audit/submissionhandles terminal.

2026-10-15 12:35UTC:8931289 F/0,fourvalidatedfinecontrolsegmentsadvance to.1712603515. Independent451source/binary/actualinput/launchcommand/CK02/header/fingerprint/latestreference/initialhashesPASS pulse-t02-controls/continuation03/integrity.json. Originalmonitorreplay41/37/34/30flagsretained,continuation03/original-monitors.json. Onlyfine successor8931339submitteddebug1node30min,CompactBinaryMerger,afterparentintegrity/source/binary/scripts/grids/preflight/no-taskallocationgates;continuation04/submission.json authoritative,stateQverified. ParentCK SHA faee251033806989ca1d0a967649fb8234d884b458b6b96001577dbed44d8839. Max4validated300ssegments/420sprocesscap/1200sstartdeadline/1700swrapper unchanged,allcompletedcontrolsreused. Jointreview_pulse_closure_controls.py prepared,closure-review-manifest.json points3matchedpairs and3fielddifferencefiles/timestepreview;partialexecution correctlyreturnscommonfalse andnullcomparisons(no unequal-timeclaim). Atfine.2endpointupdatefineinteriorpath,collectverifiedCK02andfinebaselinegeometry,compute field-fine-t02.json thenexecutejointreview. Keepsallnativeconstraintnorms/radialregions/pointwisestatecomponents/roundoffclassifications/failures;normdifferencesnotnormofpointwise differences,alternative own temporal/backgroundcontrolsremainunmeasured. No nativephysical/source/referencechanges,nostabilityacceptance. Hourlytimerupdated;alltransfer/audit/submissionhandles terminal.

2026-10-15 13:33UTC check:8931339 F/0; three fine-control segments finish common.2. Independent451source/binary/actualinput/launchcommand/CK02/header/fingerprint/latest immutable reference/initial hashes PASS continuation04/integrity.json. Original monitor flags41/37/37/30 retained. Full matched closure-review.json: every fixed AND radial H/M/Z RMS/max decreases16/24/32 within both closures, but112 constraint-norm-difference sequences(42fixed) and86 resolved component/region/norm sequences fail monotonic decrease. Largest collar component RMS differences5.71554e-7/3.59117e-7/2.62847e-7 decrease while connection maxima plateau. Some fine increments/tiny bulk effects are time-unresolved; alternative own time/background controls pending. Fine checkpoint/reference bulk geometry downloaded/hash verified, field-fine-t02.json completes native GPU pointwise comparison. closure-comparison-t02.png/pdf inspected, conclusions-t02.json preserves unsupported claims. No adoption/long stability acceptance.

New CPU-only frozen principal SBP-SAT test predeclared in boundary-principal-sbp/criteria.json, implemented check_native_principal_sbp_sat.py: nine actual evolved pulse/puncture symbols pass positive norm/symmetrizer, canonical shifted-wave reconstruction and characteristic penalty energy identity at24/48nodes (max relative residual1.162e-15; canonical9.621e-15). Removing outer incoming penalty yields a positive energy direction, not proof of a native unstable eigenmode. Native tracefree matrix[[A,B],[C,D]] roots beta+-c; Pi=(A-beta I)q_n+Bp reduces frozen principal system to q_t=beta q_n+Pi,Pi_t=c^2q_nn+beta Pi_n. Physical S=-M and branch-projector symmetrizer W give SBP-SAT energy derivative -(tLtL^T+tRtR^T) tensor W|S|. This is ONLY first-order frozen principal algebra; compatible native second/mixed derivatives, variable coefficients, advection/KO, seams and nonlinear/constraint boundary data remain to derive/test. Initial small-matrix BLAS emitted spurious RuntimeWarnings despite finite results; first-pass-with-blas-warnings.json preserved. Recomputed with einsum and strict NumPy exceptions, no warning/error and identical canonical errors. Frozen native source/binary untouched.

Prepared closure-background-radial-phase criteria before submission: alternative fresh middle half-step and stationary controls to.2; puncture B32/A24 radial160/168 to.1 alongside existing164/205(no reruns). Report changing minOmega/counts/volumes and layer-join phases; this scan is sensitivity, not balanced convergence or unique boundary/PDE causation. External generic wrapper permits target.1/.2,max4validated300s segments,420s process cap/1200s startdeadline/1700s wrapper. Pulse diagnostic absolute1 retained; puncture diagnostic-only cap3 allows reproducing known failed Mmax1.371445 at.1, never weakens original physical absolute1/growth acceptance, which must be replayed unchanged. No evolution modification/exclusion. Overall goal and near-scri convergence remain unmet.

2026-10-15 14:03UTC: new authoritative closure-background-radial-phase job8931498 submitted debug1node30min CompactBinaryMerger after terminal-parent integrity/451source/binary/script/grid/preflight/no-other-task-allocation gates. submission.json and scheduler.json authoritative; fresh four controls only, completed controls reused. Original physical acceptance unchanged. Hourly heartbeat updated; validate all segment/source/input/CK02/reference/initial hashes before diagnostic successors.

2026-10-15 14:28UTC check:8931498 F/0,wall20m42s;12segments all451source/binary/actualinput/launchcommand/CK02/header/fingerprint/latest immutable reference/initial payload hashes independently verified,closure-background-radial-phase/integrity.json. Puncture radial160/168 bothcomplete.1; originalmonitor47failures each retained, no physical acceptance. Radial-review.json: nearby +/-2.44% radial changes produce at most7.089% collarH/M and0.06885% bulk-interior constraint changes, below predeclared25% strong-sensitivityguide. R205 stillworsens compared164; local scan doesnot supportstronglocalphaseexplanation oruniqueboundary/PDEcausation. Allregionalvalues/sampling/minOmega/peaks retained,radial-scan-t01.png/pdf inspected. Alternativehalfstep only.0641825519,stationary.1273344359: cannot comparecommon.2 yet; partial-background.json records boundedbackground result. Originalpulsecontrolsremaincomplete/reused. Compact collection initially hit local olderPython tarfilter keyword incompatibility; correctedbyvalidating each memberisregular/relative/no-parent-path beforeextracting, no evidencechange. Onlytwoincompletecontrols successor8931563 submitted14:30UTC debug1node30min CompactBinaryMerger, afterparentintegrity/451source/binary/grids/scripts/preflight/no-other-task-allocationgates;continuation01/submission.json authoritative,Qverified. No puncturereruns orfailedphysicalsuccessor. Sameexternalmax4validated300ssegments/420sprocesscap/1200sstartdeadline/1700swrapper. Nextaudit/collect/latestcheckpoint-onlycontinuation untilcommon.2. Nativeequations/source/binary/reference unchanged; originalphysicalcriteria/failedflags preserved. Hourlyheartbeatupdated.

2026-10-15 15:27UTC check:8931563 F/0,wall20m37s;sevenvalidatedsegments. All451source/binary/actualinput/launch/CK02/header/fingerprint/latestimmutable reference/initial hashes PASS closure-background-radial-phase/continuation01/integrity.json. Alternative stationary reaches.2 andfullseven-segmenthistory passesbackgroundlimits(maxconstraint7.75865e-10,fielddrift1.76157e-11),background-review.json. Alternative halfstep reaches.12704886158,originalgrowthmonitor nowFAILS24historicalflags;preserved original-monitors.json. This is bounded timestep diagnosis, not accepted physicalcontinuation;currentnativeabsolute1diagnosticcap/finite/referencechecks pass. Onlyhalfstep successor8931723 submitted15:28UTC debug1node30min CompactBinaryMerger afterparentintegrity/source451/binary/scripts/input/preflight/no-task-allocationgates;continuation02/submission.json authoritative. Completedstationary/puncture/allbaselinecontrols reused, no duplicates. Nativeequations/source/binary/reference unchanged. Preparedreview_alternative_pulse_controls.py withallfixed/radialH/M/Z time-vs-spatialandtime-vs-closureeffects atcommon.2;partialexecution correctlyreturnscommonfalse/nullacceptance. FirstexecutionassumedBHinitialreceiptkeyonMinkowskipulse;correctedoptionalBHinitialcheckwithmass0guard,notphysicschange. Requiresupdatehalfpathwhencomplete. Pointwisealternativeown-timeGPUfieldcomparison stillneeded forresolvedstateclosureeffectclaims. Sameexternalmax4validated300ssegments/420sprocesscap/1200sstartdeadline/1700swrapper;preserveoriginalflags. Overallnear-scri/longpulse/trumpetgoalsunmet. Hourlyheartbeatupdated.

2026-10-15 16:31UTC check:8931723 F/0,wall20m34s;fourvalidatedhalfstepsegments reach.19296068457104823. Independent451source/binary/actualinput/launchcommand/CK02/header/fingerprint/latestimmutable reference/initial hashes PASS closure-background-radial-phase/continuation02/integrity.json. Alloriginalgrowthflags retained,replaynow37historicalfailures,absolute1diagnosticcapandfinite/referencechecks pass. Time-controlreviewstillcorrectlycommonfalse/nullacceptance. Onlyremaininghalfstep successor8931902submitted16:33UTC debug1node30min CompactBinaryMerger afterparentintegrity/source451/binary/scripts/input/preflight/no-task-allocationgate;continuation03/submission.json authoritative. ValidatedparentCK SHAf80891c74f6a59a0e70371c86cf2e185ee79676b8f2e51d6b42e5dc02204374f. Reuseallcompletedstationary/puncture/pulsecontrols,no duplicates. Nativeequations/source/binary/reference unchanged;originalphysicalmonitorfailures notrelaxed. Sameexternalmax4validated300ssegments/420sprocesscap/1200sstartdeadline/1700swrapper. Aftercommon.2 updatealternative-review-manifest halfpath,apply allregional time-vs-spatial/time-vs-closure review andnativeGPUpointwisetimecomparison before resolved-effectclaims. Backgroundalternativealreadypassesfullhistory.2. Hourlyheartbeatupdated;longpulse/trumpet/originalnear-scriconvergence remainunmet.

2026-10-15 17:31UTC check:8931902 F/0,wall2m23s;singlehalfstepsegment completescommon.2. Source451/binary/actualinput/launch/CK02/header/fingerprint/latestimmutable reference/initial hashes PASS closure-background-radial-phase/continuation03/integrity.json. OriginalhalfstepmonitorFAILS38historicalflags;allpreserved. Alternative own temporal normdifferences<10%bothspatialdifferences inALLfixed/radialregions,maxrelative.0661752%;157normclosureeffectcomparisons remainunresolvedrelativetime. NativeGPUmatchedhalfstepCK downloaded/hashvalidated;middleimmutableGPUreferencebulkgeometry extracted/validated. field-alternative-timestep-t02.json:largestcollarcomponentRMS timechange2.33672e-10. Usingbothclosuresowntimecontrols,482component-region-normclosureeffects resolvedat10%,268unresolved(field-effect-resolution.json). No claimingallnormdifferencefailuresarephysicalorresolved. Stationaryalternativefullhistorypasses.2. Completecontrolchain, no newAuroraallocation orduplicatephysicalruns. Frozenquinticbaseline retained, no nativeequation/reference/source/binarychange orclosureadoption.

New CPU compatible SECOND-order principal SBP-SAT model predeclared boundary-second-order-sbp/criteria.json,implemented check_native_second_order_sbp_sat.py. Native frozenblocks satisfy A^2+BC=2beta A+(c^2-beta^2)I and AB+BD=2beta B atallnineactualstates(maxresidual2.883e-17). With v=q_t, second-order wave q_tt=2beta q_tx+(c^2-beta^2)q_xx, energy||v||H^2+(c^2-beta^2)||D1q||H^2. CompatibleD2=D1^2 and incomingSATright -(c+beta)Hinv tR[v+(c-beta)tRD1q],left -(c-beta)Hinv tL[v-(c+beta)tLD1q] giveenergyderivative -2c(sumfacev^2). Frozen nativepSATwouldbeB^-1timesvelocitySAT;notimplemented. Nine24/48node tests PASS energyidentity(maxrelative1.583e-14),RK4spectralradius<=1 atdeclaredsmallCFL,oneconstantneutralmode;removingouterincomingpenaltypositiveenergy direction. Polynomial0/1/2exacttoroundoff,butdegree3boundaryD2error.7742->.3871showsfirst-orderboundaryaccuracy. Thisorderlossandunprovedmixed/seam/variablecoefficient/KO/nonlinearconstraintBC preventadoption;neitherCPUenergyproof norRK4testisnativeGPUvalidation. Next derive/testcompatiblehigheraccuracy/mixed/advection/dissipationandfrozenwavepropagation before nativeintegration/newGPUadmission. Overallgoal/near-scriconvergence/longpulse/trumpetremainunmet. HourlyheartbeatupdatedtoCPUboundaryphase,allcompletedcontrols reused.

2026-10-15 18:32UTC localCPUboundaryphase: audited completedchains, noAuroraallocation orreruns submitted. Reusedarchived6/3cell-centeredSBP coefficients(cell-centered-sbp-sixth-exploration.json),notnewposthocoptimization. Newpredeclaredboundary-sixth-order-sbp/criteria.json. Generalizedfrozen second-order model toarchivedwiderrows/6thinterior;allsampled9nativeevolvedstates24/48node principalenergy/RK4 gates PASS(maxrelativeenergy3.629e-14). Original4/2scriptreconstructedandhashmatched archivedreview, savedboundary-second-order-sbp/original-script.py beforepreservingchangedhelper. Nativeproduction/source/binary/reference untouched.

Source-freeoutgoingGaussian wave withanalyticincoming facedata(noRHSforcing/subtraction),same9frozenstates/checkpointsoutgoingdistance.35/.7/1.2 on32/64/128:allstatecoarsesettlingcriterionFAILS(maxcoarseenergy~2.2%initial vs1e-4gate),onepunctureDqmaxsequencealsononmonotonic. Preservedpropagation-review.json/original-propagation-script.py; firstdevelopmentexecutionhadPythonfunction/listnameshadowingafterfirstcase, correctedandreranwithoutcriterionchange. Finer128/192/256predeclared separately(finer-criteria.json),samepulse/duration/gates,noKOadded:ALL9statesPASSallerror/order/time/settlingchecks,minfinestweightedRMSorder3.2178,maxfinerenergyresidual2.14449e-5. Original32/64failuresremaininplots/evidence. HalfstepcontrolsincludeerrornormdifferencesANDpointwise q/v/Dq differences. Constantq isenergyneutral,soqerrorsretained;energydecayaloneisnotfieldsettlement. principalmodelpropagation-convergence.png/pdf inspected,source/input/evidencehashindexsaved.

ConstantrectangulartensorSBP mixedderivativescommute4.06e-14relative,polynomialx2y3error1.59e-11allcells;scalaradvectionincomingSATnegativeenergytoroundoff;compatiblevelocity-onlysixth-difference dissipationnegativeenergy5.19e-17normalized. check_sbp_tensor_dissipation.py/tensor-dissipation-review.json. ThisisNOTnativecurvilinearseam/mixed/advection/KOvalidation. Candidatechangesradialinterioraswellasboundaryandcomposeswiderstencils, cannotclaimboundary-only. native-integration-gaps.md documentschainrule/variablecoefficient/seam/intermediateexchange,nonlinearincomingphysical/gauge/constraintdata andmass-consistentpunctureBCobstacles. InparticularhomogeneousincomingMinkowskideviationsmustnotimplicitlyremoveBHmass orsubstituteBHreference. No nativeadoption,longphysicalsuccessor,actualnativeRK/KOprooforPDEexonerationfromCPUmodels. Nextderivenonlinearincomingboundarydataandcompatiblemapping/seamimplementationin-isolation beforeCPU-native/GPUadmission. Originalgoal/near-scriconvergence/longpulse/trumpetunmet;hourlyheartbeatupdated.

2026-10-15 19:33UTC localnativeboundary-dataaudit: noAurorajob/physicalsuccessor orcompleted-controlrerun. ReadactualConformalRHS/ReferenceWaveMap/LayerWormhole andprimaryZ4c/GHboundarysources(sourceslinkedinnonlinear-boundary-data/derivation-and-next-tests.md),noassumptiontheirliteraturetheoremsapplytoadapter. Predeclared analyticcompatibility,angular-gaugevariant,thenoff-constraintidentityprobes separatelybeforeexecution;olderCSV/scripts/builds preserved. NewstandaloneC++native-headerdiagnosticusesunchangednative equations,exactjets,separateimmutableMinkowskiref/M=.5initialdata. Initialstrict451localindexgatefound3historicallymodifieddiagnosticPythonscripts(check_checkpoint_light_speeds.py,recount_collar_checkpoint.py,review_balanced_hyperboloidal.py),notnativeheaders/inputs;448indexfilesmatch,allnativeheaders/inputsmatchfrozenindex. Recordedexplicitlyinbuild.json;didnotresetdiagnosticsorpretendall451localfilesmatch. ProductionCPU/GPUbinaryunchanged.

48constraint-satisfying samples(curvature1,radii.9/.95/.98/.99,axis+2obliquedirections,Minkowski/wormholewithandwithoutanalyticangularlapse/shiftjets) PASSnativeH/M/ZandTheta/Zfirst-timecompatibility,maxresidual5e-14. 24intentionalchi-jetH/Mviolations testnontrivialrestrictednonlinearidentities:Theta_dot=alpha H/(2Omega),Zdot_i=alpha M_i/Omega atTheta=Z=0,maxerrors3.55e-14/1.75e-14,Hprobe.11685/Mprobe.001933. AnalyticGamma_dotcheckedindependentlyby2centralmetric-jetvariations,error2.43e-14withGamma_dotmax.18875;notatrivialstationarycheck. review.json/analytic-compatibility.csv/build/source/binary/hashindexpreserved. All9evolvedsymbolTheta/Zincomingandoutgoingvaluerowsrank4,leaving6incomingfreedoms;ranknotCPBC/energyproof. Naivehomogeneouschi-deviationSommerfeldresidual1.911872at.98onmassivedatawithH/M/Zzeroandchi_dot~2.8e-14:explicitboundary-data incompatibility,NOTproofexistingghostmasslossornativePDEexoneration.

Documentednextnativeboundarywork:fullsubsidiaryabsorbingTheta/Zoperatorincludingtangent/damping/Omega/state-dependentterms,weakSATnotfieldresets;then4gauge+2radiativeincomingconditionswithmass-consistentcurvature/tetraddiagnostics;compatiblemappedSBP/seamsbeforeCPU-native/GPUadmission. Noall-fieldhomogeneousMinkowskideviationconditionadopted,noBH-reference substitution. Originalphysicaltargetsandnear-scriconvergenceunmet;hourlyheartbeatupdated.

2026-10-15 20:31UTC nativeconstrainttransportphase: auditedpreviousreports;noAuroraallocation/rerun/physicalsuccessor. Predeclared native-constraint-transport/criteria.json;newdiagnostic-onlynative_constraint_transport.hpp independentlycontractsfullTheta/ZtimeequationswithH/M,advection,damping,Omega,metricandshiftJacobiantermsretained. NewC++audit96analyticstates includesprevious72and24nonzeroTheta/Z,spatiallyvaryingunitdeterminantanisotropicmetric/exacttracefreeA/dA,analyticLambda/dLambda. FirstvectorformulaFAILEDoff-manifold(error2.94) dueourfactor2conversionfromLambda toZ;failedheader/criteria/build/outputs preserved failed-vector-contraction/. Correctedfinal.5contractionbeforepredeclaredreexecution,no nativeequation/tolerancechange. CorrectedformulasPASS96:maxThetaerror5.773e-14,Zerror2.310e-14,independentGamma_timevariationerror2.871e-12. ShifttermsfollowactualnativecontractedGammaequation,notrememberedcovariantformula.

Fulloutgoing-nullvector vout=-beta+alpha sqrt(chi)ginv*n/sqrt(ginv(n,n)) retains tangential transport; coordinatecomponentconstraintSommerfeldresidualR=(Theta_t+vout*dTheta,Zdot+vout*dZ)vanishes<=6.61e-14on48constraint-satisfyingMinkowski/massive/angular-gaugedata. Scalar(c-beta_n)n substitutionomitsup to1.66029e-4 onanisotropic/shiftprobes. Thisisaconcretemass-compatibleconstraintresidual,NOTclosedH/M/Theta/Zsubsidiarywaveproof,absorbingSATenergy/reflectionvalidation,orfull10conditionBC. Remaining6gauge/physicalincomingdata/mapping/seams unvalidated. AllCSV/source/compiler/binary/failure/evidencehashesindexed,review.json/derivation.md/conclusions.json. Localnativeintegrityagain448indexfilesincludingallnativeheaders/inputsmatch;3historicallymodifieddiagnosticPythonscripts explicitlyrecorded. ProductionCPU/GPUbinary/reference/nativeequations untouched. Nextderive/testsubsidiaryclosureandweakabsorbingSATreflection/injectionbeforeadoption,notTheta/Lambdaresets;preserveCoulombmass andordinaryGPU/bulkAMR. Originalnear-scri/longangularpulse/trumpetgoalsremainunmet;hourlyheartbeatupdated.

2026-10-15 21:34UTC constraintincomingliftingphase: auditedlatestnativeconstrainttransportandfrozenprincipalrecords. Predeclared constraint-incoming-lift/criteria.json;newcheck_constraint_incoming_lift.py testsK=J(M+cplusI)withJTheta/Zvaluerows. All9actualevolvedsymbolsPASS:incomingrank4,outgoingannihilationmax5.20e-16. Adding4lapse/shiftderivativeand2live-metricTTderivativesensors givesfullincomingrank10;fourunitandtwomixedconstraintresidualtargets canceled<=5.58e-15,outgoingchange<=2.12e-15,selectedfreesensorchange<=2.38e-16. Raw/equilibratedconditionmax79.68/4.15,unitresidualliftgain44.06coordinateunits;notuniformOmega0bound. Thisisderivative-levellifting,notTheta/Zresetsorfullsixnonlineargauge/radiationdata. DocumentedfrozenincomingboundaryODEroutewithoutdivisionbysmallincomingSpeed;liveJ/Pin/time/tangenttermsandsecond-orderq_ncompatibilityremainrequiredforweakSAT. NoSATadopted/no nativePDEexoneration.

Separatelypredeclaredreflection-criteria.json beforeanalyticobliqueconstraint-wave test. All9samplednormal-speedpairs,angles0/15/30/45/60/75PASSshifteddispersion/rootgroupspeed/zero-normal-reflection/zero-shift tan(theta/2)^2/energyfluxchecks. At45degpulsefieldreflection1.87e-5..1.48e-4butreflectednormalwavenumberstretch1395..11062;normalderivativeamplituderatio~.2071. Puncturefieldreflection.000467..002989,stretch69.3..443.8,similar.2071derivativeratio. Smallfieldreflectiondoesnotboundsmallradialderivative;normal-onlyprincipalsettlingisinsufficientforangularboundaryassessment. Thisisascalarprincipalmodelwithrescaledtangentmetric,NOTactualnativeconstraintreflectionmeasurement/collarcausation/instabilityclaim. DoesnotprovefirstorderSommerfeldcannotbeacceptable. oblique-reflection.png/pdf inspected,derivation-and-limits.md/review/reflection-review/conclusions/source/input/hashrecords saved. Schedulerread-onlyauditconfirmsnoactiveallocationunderthistaskremote root;nojobs submittedorcompletedcontrolsduplicated,otheragentsuntouched. Nativeequations/source/reference/productionbinariesunchanged. Nextlive-coefficientweakSATandreduction-compatibility/subsidiaryclosure,obliqueconstraintinjection/reflectioncontrols,remaining6physical/gaugedata/mappedseamsbeforeCPU-native/GPUadmission. Overallnear-scri/longangularpulse/trumpetgoalsunmet;hourlyheartbeatupdated.

2026-10-15 22:39UTC live-coefficientweakSATprototypephase: auditedincominglift/nativeconstraintrecords;noAuroraallocation/physicalsuccessor/reruns. Predeclaredlive-coefficient-sbp-sat/criteria.json;newisolatedscalarcanonicalprototypewithspatial/time-varyingnative-speed-scaledbranches,positivea=c²-beta². UseddecoupledgeneralizedSBPfacefluxextensionQN/P tokeepvariablecoefficientfaceproductscorrect;weakD2a and(b-Da)Dq retaintime/volume-workterms. q_ncomputedDq,preconstructedp=v-betaDq hasp_t=v_t-beta_tDq-betaDq_t,noindependentevolutionofderivatives. CompleteenergyidentityPASSall9speedpairs24/48nodes3times(maxrelative2.212e-15);naive2betaDnegativecontrolfailsidentityup to.00211051. Nomonotonicenergyclaim:volumeworkGronwallrateup to4.808. SeparatepredeclaredauxiliarymodelboundaryODEg_t=-.4gwithweakincomingSATsourcepassesextendedenergyidentity/nonpositiveboundaryquadratic(maxeig7.78e-16). ThisisNOTnativeconstraintboundaryODEorreflectiontest. Livecharactertimechainrulevalidatedby2directionalvariationswithin6.01e-8;omittedcoefficienttermschangevalues.03379/.07892,notnativebugclaim.

Separatepredeclaredconsistency-criteria.json:testq=exp(x),v=sin(2x),exactanalyticincomingfacedata,nointeriorforcing/subtraction;allboundarycellsincluded. v_rhs/p_rhsweightedRMS/maxerrorsdecrease64/128/256forall9speedpairsand3times,minfinestorder2.079. review/consistency-review/source/input/criteria/original-script+review/hashindex/derivation-and-limits.mdsaved. Actualnativeheaders/inputsagainmatchfrozenindex,448indexfilesmatch,3historicaldiagnosticdifferencesexplicit. NativePDE/reference/productionCPU-GPUbinariesunchanged. Remainingvariable-coefficientmodelpropagation/reflection/injection,fullnativeH/Msubsidiary/matrixsourcecoupling/6physicalgaugedata/mappedseams/actualKO/CPU-native+GPUadmissionbeforelongphysicalsuccessor. Overallnear-scri/longangularpulse/trumpetgoalsunmet;hourlyheartbeatupdated.


2026-10-15 23:50UTC (23:42UTC heartbeat): completed 24 bounded local scalar live-coefficient weakSAT evolutions (two preselected native speed representatives; outgoing and synthetic injection; N96/128/192 plus384/512 references and halfdt128). All source/payload receipts independently rehashed. Every energy/Gronwall/auxiliaryODE gate passes; largest relative energy-balance defect1.90e-10. All four case accuracy gates FAIL; primary outgoing final energy fractions~.0113/.0055/.00151 exceed1e-4 settling criterion. N384/512 outgoing residuals decrease below1e-4 but reference-control failures prevent promoting them into an accuracy claim. All timestep gates pass, so this test does not support timestep error as explanation. Injection was deliberately q=v=0 with g0=.01: initial-boundary corner is incompatible and cannot establish smooth-order convergence. Initial-speed penetration estimates at final injection time are .00838 cells(pulse) and1.338 cells(puncture) even at N512; changing coefficients mean these estimates are not exact characteristic integrals. Numerical-reference grids are unresolved, not analytic truth. Preserve failed criteria/results, do not weaken gates or infer native instability/causation. E/live-coefficient-sbp-sat/evolution/{criteria,review,conclusions,hashes}.json and evolution-failures.png/pdf saved, plot inspected. Script tst/cubed_sphere/evolve_live_coefficient_sbp_sat.py; all24 runs terminal under L/live-sbp-evolution, no live local process. Production equations/binary/reference untouched; no new Aurora allocation. Next isolate smooth corner-compatible incoming characteristic transport with analytic characteristic solution and mapped SBP/energy checks; predeclare mapping/CFL cost before full coupled mapped-wave time tests. A very fine near-boundary grid also constrains the fast outgoing explicit CFL, so do not assume mapping is computationally cheap. Full native subsidiary/gauge/radiation/mapped seam consistency remains missing. Physical pulse remains t.2, puncture t.1 failed acceptance, no long stability or trumpet demonstrated. Overall goal incomplete.


2026-10-16 00:46UTC: incoming-mapped-transport predeclared smooth compatible scalar transport with exact characteristic oracle, fixed exponential map and archived6/3SBP. Completed16 original evolutions(uniform/mapped two speed representatives,64/128/256+half128) and8 separately predeclared uniform boundary-time controls. All mapped energy gates PASS but both mapped full accuracy gates FAIL (4nonmonotonic/10order failures each); derivative errors are not controlled by L2 energy. Original uniform CFL-only controls undersampled boundary pulse; dt<=tau/200 controls repair energy but spatial accuracy remains failed, all preserved. Mapped N256 incoming scale resolves73pulse/102puncture cells; coupled fast-outgoing explicit-CFL estimates179,472,326pulse/1,124,158puncture steps over1.2tau, not production timings or a proposed cheap map. Independent derivative decomposition separates analytic-data spatial error from amplified evolved-state error. Additional predeclared exact-data derivative check256/512/1024 FAILS4late pulse quantities(minmaxorder.753,RMS1.431); no acceptance or reinterpretation of original failures. All24new evolution payload/source identities verified, exact inversion/SBP checked, native index448match plus3historical diagnostic differences; allnativeheaders/inputs unchanged. Initial dense-BLAS review warnings archived, strictFP/einsum recompute clean. E/incoming-mapped-transport contains criteria/reviews/derivation/conclusions/hashindex/mapped-incoming-errors.png/pdf, plot inspected. No live local process/no newAurora job; no duplicate completed native controls. Next derive/test compatible scalar derivative-energy/H1 estimate with variable coefficients, discrete commutators and g_t/aR boundary data, before using L2 stability to justify derivative-sensitive constraints. Full native subsidiary/gauge/radiation/seam/GPU admission and original physical targets remain incomplete.


2026-10-16 01:44UTC: predeclared incoming-derivative-energy/criteria.json and graph-criteria.json before read-only CPU tests, reusing completed mapped scalar snapshots (no evolution rerun). Full r=Dw derivative identity retains commutator/SAT-source C and boundary zB=-g_t/aR: all18snapshot identitiesPASS(maxrelativeenergy4.43e-12/RHS2.05e-12); omissionnegativecontrol defect.932. Proposed mapped H1 log-rate screenFAILS strongly, not eigenmode-instability proof. Separate homogeneous scalar graph norm Gg=H+tau²LTHL with live Gg_t satisfies exact identity and energybound on all12grids/states; maxidentity2.51e-16/Rayleigh5.68e-11/generalizedresidual1.95e-9,condition1.33e7reported. Mapped graph/H1 equivalence FAILS(min~1.4e-4,max~7.4e3 at128, worsening), so graph bound doesnot uniformlycontrolphysicalderivatives. Homogeneous proofdoesnotboundtime-dependentinjectionwithoutadditionalboundary-source derivativework. Uniformpulse H1 excessnearzero makesratioindeterminate,notmeaningfulgrowth. NumPybool JSONserializationfailure correctedwithoutformula/gatechange. E/incoming-derivative-energy containsreviews/criteria/derivation/conclusions/hashindex/derivative-graph-energy.png/pdf;plotinspected. Nativeindex448match+3historicaldiagnostics;allnativeheaders/inputsunchanged,productionbinary/referenceuntouched;no new/liveAuroraallocation. NextreturntonativeH/M/Theta/Zsubsidiaryclosure,tangential/finiteOmega/lowerorderterms,reductioncompatibilityandremaining6gauge/radiationdata. ScalarL2/graphstabilitydoesnotadmitnativeboundaryorlongsuccessor. Originalphysicalpulse t.2/puncture t.1failedacceptance andnear-scriconvergence goalsremainincomplete.


2026-10-16 02:44UTC: native-subsidiary-principal predeclared explicit8component frozen principal systemX=(H,Mi,partial_nTheta,partial_nZi), derived from native constraint/RHS jets. All9actualevolvedsymbols PASS C M=B C(max2.283e-15), independentSVDquotient1.029e-13, wavepolynomial2.395e-17, speederror2.443e-15, symmetricflux4.463e-15, symmetrizercondition<=4981reported. alpha/chi/beta fromindependentarchivedGPU samples,notfitted. Fourincoming/fouroutgoingconstraintbranches, transverseWcoupling omissionmismatch.01120. Separatepredeclared boundaryrow testPASS R C=J(M+cout I),outgoingannihilation/incomingrank4. Explicitfullnormalcoefficientsstored;principal-onlyoriginalscript/reviewretained. InitialNumPypinv internalmatmul floatingpointerror correctedwithstrictSVD/einsum,unchangedformulas/gates. E/native-subsidiary-principal containscriteria/boundarycriteria/review/conclusions/derivation/hashindex/nativeintegrity. Nativeindex448match+3historicaldiagnostics;allnativeheaders/inputsunchanged,no newAurorajob,noduplicatecontrols. FullfiniteOmega/lowerorder nonlinearH/Mclosure STILLMISSING. Documented exactfunctionalvariationroute forH/M time derivatives requiresconsistentnativeRHS spatialjets(secondmetric/chi,firstA/K); nextbuildindependentanalyticjet/timevariationoracle,thenreduce toclosedsubsidiarysystem inclOmega/damping/tangent/state terms. Constraint-satisfying-onlyprobes insufficient;nonzerosource testsneeded. PrincipalrowsarenotcompleteBC/SBPpenalty;remaining6gauge/radiationdata,mappingseams/reduction/GPUadmissionbeforephysicalsuccessor. Longpulse/trumpet/originalnear-scrigoalsincomplete.


2026-10-16 03:51UTC: native-hm-time-oracle predeclared criteria and separate functional-criteria before tests. New diagnostic C++ AD scalar differentiates unchanged native RHS spatial jets exactly for analytic polynomial/exponential fields, then a separate timeAD direction differentiates H/M alongfullRHSjets. All48nonzero-constraint probes(4radii,3directions,2variants,kappa1=0/5) PASS centraltimevariationeps1e-5/5e-6/2.5e-6 maxnormalizederrors1.93703e-7/4.84255e-8/1.21063e-8 (gate5e-7). FullTheta/Z identitiesPASS8.70e-16/7.86e-16; omittingspatialRHSjetsnegativecontrol1.27925. Separately hand-assembled fullconstraint-functional variationHdot/MdotPASS2.35e-16/4.11e-16, retaininginversemetric/A/curvature/Omega/Kandmomentumconnection/chigradientterms. CurvaturevariationstillusesnativegeometryAD,notindependentcurvatureproof. Omega agreesimmutableouterMinkref7.63e-17. Exported48fullstate/RHSjet283-entryJSONLplusallH/Mcomponents/timevalues;build/run/source/binary/payloadhashes saved. Originaloracle andprefunctionalresultsretained. InitialC++templatededuction/mixedautocompileerrors corrected, noformula/gate/nativechange. E/native-hm-time-oracle reviews/criteria/conclusions/derivation/hashindex/nativeintegrity;nativeindex448match+3historicaldiagnostics,allnativeheaders/inputsunchanged,productionbinary/reference/checkpointsuntouched. Fullfunctionalvariation is NOTCLOSEDsubsidiaryformula. NextalgebraicallyreducetoclosedH/M/Theta/ZwithfiniteOmega/damping/tangential/livetransformationterms, testagainstoracleandadditionalanalyticfamilies. Remaining6gauge/radiationdata/compatibleSAT/mappedseams/KO/nativeCPU/GPUadmissionbeforephysicalsuccessor. No new/liveAurorajob;allcompletedcontrolsreused;longphysicalgoalsandnear-scriconvergenceincomplete.


2026-10-16 04:57UTC: independently derived physicalADM sourceEij and CLOSED nonlinearH/Mcandidate, retainingfiniteOmega/Zgradient/damping/Theta/shiftterms; predeclared native-subsidiary-closed-candidate criteria. Original48oracleprobesPASSsource2.82e-11/H9.50e-16/M5.26e-16. Added192predeclaredprobes(newkappa2.3,rotatednondiagonalmetric/Afamily,differentpolynomials,exactCMCMinkowski), reusingcompletedbaseline. AllH/Mclosurevaluespass<=2.66e-13/4.74e-12, butextendedORIGINALvalidationFAILS16gates:8coarsetimevariation+8Minkowskisourcesubtractionat.99(max1.62e-9>1e-9). Preservedinitial-review andfullreview. Smaller eps2.5e-6/1.25e-6/6.25e-7control reducestimeerror1.01e-6->6.32e-8;8sourcefailuresremain. Nominal longdouble NOTwiderprecisiononARM(8bytesboth);explicitlyrecorded,noprecisionclaim. Separatepredeclared60digitDecimal exactMinkADMsourcecancellationPASS<=2e-53,floatindependentformula2.33e-10;supportsdiagnosticcancellation,butdoesnotreplaceoriginalfailednativegate/noproductionfix. E/native-subsidiary-closed-candidate hasallcriteria/source/builds/payloads/reviews/derivation/conclusions/hashindex/closed-candidate-controls.png/pdf,plotinspected. PrimaryADMidentitysourcearxivgr-qc0103032eq3.1-3.6crosschecked;sourceEderivedfromnative,notfit. Nativeindex448match+3historicaldiagnostics,allnativeheaders/inputsunchanged;no productionequation/binary/reference/checkpointchange/noAurorajob. Closedcandidateprovisional;nextconstraint-satisfyingmassivewormhole/angular-gaugecontrolswithconsistenthigherRHSjets,resolveprecision/sourceoraclelimitationswithoutgatechanges,thensubsidiarywave/weakSAT/live reductioncompatibilityandsixCoulomb-preservinggauge/physicaldata. No nativeboundaryadmission/physicalsuccessor;longgoalsandnear-scriconvergencestillincomplete.


2026-10-16 05:57UTC: predeclared native-subsidiary-massive-controls60outeranalyticprobes. IndependentAD/Radial2/CartesianreconstructionmatchesnativeLayerWormhole/referencejets3.55e-15, integrabilitythroughthirdorder1.42e-14. Mass.5base/angular-gauge/nonzeroThetaZ andmass0angularvariants;staticMinknotduplicated. Constraint-satisfyingcasesH/M/Theta/Z<=2.66e-15;closedH/McandidatePASS3.58e-13/4.92e-12,timevariation5.28e-8,Theta/Z4.50e-14/2.01e-14. FouroriginalsourcesubtractiongatesFAILat.99(max2.44e-9>1e-9);fullvalidationnotadmitted,failures/payloads/reviews/plotretained. ProductionMinkref/equations/binaries/checkpointsunchanged. Localnativeindex448match+3historicaldiagnostics. Toresolveprecisionlimitation,submittedCPU-onlydiagnostic8934689(debug1node30minCompactBinaryMerger),Qverified. RemoteROOT/cpu-subsidiary-precision-v1; authoritativecpu-precision/submission.json/scheduler.json. Auditbeforeqsub foundnoactive taskallocation andverifiedall451remote sourcefiles/GPUbinary181395.../diagnosticmanifest;localprecisionmassive syntaxpreflightPASS. RealprivateSERIALKokkosbuild,longdoubleprobe mustdemonstrate>53mantissabitsbeforeprecisionclaim, then192synthetic+60massivecontrols,originalgatesretained. Individualtimeouts/1700wrapper;productionGPU/source/ref/checkpointsuntouched;no physicalsuccessor. Neverduplicate8934689;onfinalauditscheduler/source/binary/input/output/payloadprecisionreceipts,scientificCSVandcommandsbeforeconclusions. IfQ/Rwaitnext hourlywake. NextafterverifiedprecisioncontrolderiveTheta/Zsubsidiarywave/weakSATwithlive/tangential/reductioncompatibilityandremaining6Coulomb-preservinggauge/physicaldata. Longpulse/trumpet/originalnear-scriconvergencegoalsincomplete.


2026-10-16 06:49UTC: CPUprecision8934689 F/1,25sec, failed synthetic diagnostic compilation before ANY scientific cases due missingKokkos mdspaninclude. Precisionformat executable64mantissabits/16bytes confirmswiderCPUtype,notclosurevalidation. Archivedscheduler-final/failure-review/failed-jobcompact/fullcommands/logs. Fixedexternalrunnerincludeonly, native/mathinputs/gatesunchanged. Bothactualsynthetic/massivediagnostic sources PASSAuroraicpxsyntaxpreflightwithrealserialKokkosandmdspan;remote-preflight.json. Freshscheduler/source/GPUbinary/manifestauditPASS451files,noactivetaskallocation. Submittedreplacement8934851(debug1node30minCompactBinaryMerger),Qverified,remoteROOT/cpu-subsidiary-precision-v2/results/<fullPBSID>; authoritativecpu-precision/v2/submission.json,scheduler.json,manifest. Neverduplicatehandle. Same192synthetic+60massivecontrols andoriginalscientificgates,privateCPUbuild/binaries/no physicalbranchadvance. Originalfailedparentretained;CPUprocesssuccessalonewillnotadmitclosure. Nextfinalauditscheduler/source/binary/precision/outputpayload/commandsandscientificgatesbeforeconclusions;Q/Rawaitnexthourlywake. ProductionGPU/reference/checkpoints/equationsunchanged,originalphysicalgoalsincomplete.


2026-10-16 07:54UTC: CPUprecision8934851F/0,57sec,real64bitmantissa/16bytes. Independentremoteaudit verifiesallcompletedfile/binary/source/manifests,451frozennativefiles/GPUbinary/serialKokkoslibs/CMakecache;localcollectedpayloadhashesmatch. ALL252scientificprobesPASSoriginalgates:192synthetic maxsource1.79e-12/H2.00e-13/M3.33e-12/time6.32e-8;60massivemaxsource4.70e-12/H2.05e-13/M3.43e-12/time5.28e-8. Originaldoublefailuresandfailed8934689preserved;architecture/compileralsodiffer,cancellation-limitedoraclesupported,notuniquecausedproof/productionfix. cpu-precision/v2/scientific-review.json/conclusions/hashindex/precision-source-comparison.png/pdfsaved,plotinspected. No newallocation,no physicalsuccessor.

Separatepredeclarednative-physical-constraint-transport exactLIVE-index/volumeidentitiesPASSall300analyticprobes(48original+192extended+60massive):physicalTheta2.98e-14/Zup2.26e-17/C6.02e-16;omittingmetricinverse_t inraisingnegativecontrol4.95e-3. Zphysical^i=Omega²chi(Lambda-Gamma)^i/2,C_i=partial_i ln(Omega sqrtchi),D0=dt-beta.partial. DerivedD0Theta=N[H/2+partialZup-2C.Zup-dTheta],D0Zup=N[gammaInv(M+dTheta)-kappa1Zup],D0C=partial(NK-divbeta)/3+dbeta.C. Allnativecovector shifttermsandlivemetricvariationretained;nativeconstraintnormsunchanged,noOmega-amplitudesuppressionclaim. Diagnosticbuild/source/CSV/payloadreviewsstored;nativeindex448match+3historicaldiagnostics,allnativeheaders/inputsmatch,productionequations/reference/binaries/checkpointsuntouched. Documentedclosedsecondtimechain includingmetric/gauge/derivativecommutators asNEXTUNTESTEDcandidate;needindependenttimeoracle/fullwaveenergy/incomingweakSAT/live reductioncompatibilityplus6Coulomb-preservinggauge/physicaldata/mappedseams/KO/nativeGPUadmission. RaisedversuscovectorSommerfeldchangeslowerorderdata;thisrewritealoneisnotBCselection. No longpulse/trumpet/near-scriconvergencesuccess.


2026-10-16 09:00UTC: native-subsidiary-second-time predeclared600state/gaugeprobes(300constructors,twoEXTERNALspatiallyconstantgaugevelocitychoices). Allsecond-timeclosedTheta/ZupchainsPASSmax4.015e-11/4.619e-14 versus1e-9; independentcentralnativevelocityvariationsatthreepredeclaredstepsPASSmax2.171e-7 versus5e-7. Omittedcommutator/liveinversemetriccontrolmax.0194. RawZuporacleusesnativeLambdaRHSandanalyticcontractedGamma_time,retainingmetricvariation;closedH/MvaluesindependentoforacleRHSderivatives. Finalgeneratedsources,preFDsources/results,buildcommands,binaries/payloadhashes/review/derivationpreservedinE/native-subsidiary-second-time. Externalalpha_t/beta_tchoicesARENOTnativeReferenceWaveMapcoupling. Localindex448match+3historicaldiagnostics,allnativeheaders/inputsunchanged. NoAuroraallocation/no completedcontrolrerun/no productionequation/reference/binary/checkpointchange. NextpredeclaretestnativeReferenceWaveMapgaugevelocitiesandconsistentjets,explicitLaplacian/lower-sourcechain,thencompatibleweakSAT/reflection/injectionwithsixCoulomb-preservingincominggauge/physicaldata,mappingseams/KO/nativeadmission. No boundary-stabilityclaim/physicalsuccessor; pulse.2,puncture.1failedacceptance,long/near-scrigoalsincomplete.


2026-10-16 09:51UTC: native-rwm-second-time:300analyticprobes nowuseunchangedReferenceWaveMapgaugevelocitiesandconsistentfirst/secondspatialjets. Allclosedsecond-timeTheta/ZupchainsPASS4.014e-11/4.564e-14;centralnativevelocityvariationsPASS1.421e-7. Independentouterreference/state/connectionmatches4.45e-16. InitialcoarsespatialgradientcontrolFAILandineffectiveomissiontestpreserved;smallstepgradientPASSbutvaluesecond-differenceHessianFAILfromcancellation,preservedseparately. PredeclaredcentralvariationoffirstADgaugegradientschecksallmixedHessiancomponentsPASSmax1.539e-6(gate5e-6),gradient6.281e-8. Originalcontrolfailuresremain;notAD-freeHessianproof. Correctedomissioncomparesraworaclesandshowsall-gaugevelocitysensitivity.3764;spatial-velocitytermscancelasexpected. FormalprincipalcoefficientbookkeepingPASSlightconeLaplaciancancellation,notlower-sourceenergy/SAT. E/native-rwm-second-timecriteria/reviews/derivation/sources/commands/failures/hashessaved. Nativeindex448match+3historicaldiagnostics/allnativeheaders-inputsunchanged,noAuroraallocation/physicalsuccessor. NEXTexplicitnonlinearlowersources/reduction-compatibleincomingweakSAT,remaining6Coulomb-preservinggauge/radiationdata,mappingseams/KO/nativeCPU-GPUadmission. Physicalpulse.2/puncture.1andfailedacceptanceunchanged;overallgoalunfinished.


2026-10-16 10:49UTC: native-subsidiary-wave-sources predeclared300nativeReferenceWaveMapcoupledanalyticprobes. Explicitnonlinearcoordinatewaveprincipal/lower-sources derived independently,notdefinedbysubtractingnativeoracle. AllTheta/ZupwavegatesPASS4.014e-11/4.619e-14;covariantEdiv-gradtraceEcurvature/roughLaplacianidentityPASS1.712e-14. Negativeomissiongroupchanges.1801. RetainedfiniteOmega/lapsegradients/curvature/damping/shiftJacobian/liveinversemetric/volumecommutators. Sourceshaveconstraintvalues/firstderivatives,noH/Mgradientsorsecondconstraintderivativesafteranalyticcancellation;backgroundmetricsecondjetsremain. DocumentedU,V,gradUalgebraicH/MreconstructionbutNOTyetcoefficient/timevalidated. E/native-subsidiary-wave-sourcescriteria/review/conclusions/derivation/source/commands/binary-payloadhashes saved. Nativeindex448match+3historicaldiagnostics,allnativeheaders/inputsmatch;productionequations/reference/binaries/checkpointsunchanged,noAuroraallocation/completedcontrolrerun. NEXTindependentreducedsourceoperatorvalidation,compatiblecoupledweakSAT/energy/obliquereflection-injection/nativepenaltyliftand6Coulomb-preservinggauge/physicalconditions,mappedseams/KO/nativeadmission. No nativephysicalsuccessororlongstability;pulset.2/puncturet.1failedphysicalacceptance/near-scrigoalsincomplete.


2026-10-16 11:49UTC: native-reduced-subsidiary-source300backgrounds PASSindependentH/Mreconstruction1.166e-15,reducednativewave9.440e-13,zeroexact,20basis/3mixturematrixaction1.224e-13,secondconstraintHessianinvarianceexact. Exported4x20sourcematricespairedwithnativeanalyticjets;sourceexpressionreusespreviouscheckedderivation,notnewindependentderivation. SeparatepredeclaredfrozenfacevelocitySATalgebraPASS6000samples(identity2.221e-16,drivenbound5.552e-17)onmeasurednegativebeta.normal;obliqueanglesarefluxsamplesNOTreflection. Incomingbranchnegativeonall300. Veryconservativesource-onlyenergybounds1.23e4..7.85e7excludevolume-work,notstability/instabilityproof. FullcoupledSBP/timeevolution/reductioncommutators/volumeenergy/nativepenaltyliftSTILLMISSING. E/native-reduced-subsidiary-sourcecriteria/reviews/matrices/derivation/hashes/commands preserved. Native448indexmatch+3historicaldiagnostics/allheaders-inputsunchanged,noAuroraallocation/physicalsuccessor. NEXTisolatedcoupledSBPsource/reduction/energy/stiffness/obliqueinjection-reflection3grid-timecontrols,remaining6Coulomb-preservinggauge/physicaldata/mappedseams/KO/nativeadmission. Pulse.2/puncture.1andfailedphysicalcriteriaunchanged;overallgoalincomplete.


2026-10-16 12:54UTC: coupled-subsidiary-sbp isolated4componentprototype:actualfrozenlower-source matrices+syntheticsmoothnative-scaledprincipalcoefficients,NOTnativebackgroundsolution.36discrete6/3SBPenergyidentitychecksPASS8.662e-16includingcoefficient/shiftvolume-work,Uconstantmodeenergyandcoupledsourcework;pointwisesourceblocksPASS2.773e-16,Q=D U. All300normalFourierlocalsymbolsk0/1/8/32source-stiffnesscontrol:waveCFLRK4accuracyFAILmax1.158;source-awaredt5.892e-6..1.867e-4PASS8.324e-13. MaxfrozenlocalRe(lambda)206.565,notnativeinstabilityproof/productiontimestepblame/fullsemidiscretespectralbound. Highwavenumbers/nonnormalgrowthnotadmitted. E/coupled-subsidiary-sbpcriteria/review/conclusions/derivation/command/source-inputhashessaved. Native448indexmatch+3historicaldiagnostics,allheaders-inputsunchanged,noAuroraallocation/physicalsuccessor. NEXTboundedmatrix-freecoupledtimeevolutionwithsource-aware/highkcontrol,fullenergyand3grid/timeaccuracybeforeobliquetangentialinjection-reflection/nativepenaltyliftand6physicalgaugedata/mappedseams/KO/admission. Physicalpulse.2/puncture.1failedacceptanceandlong/near-scrigoalsincomplete.


2026-10-16 14:00UTC: coupled-subsidiary-sbp/evolution18boundedCPUrunsterminal,.005/.01/.02,N32/48/64+128/160references+half48. Fullinstantaneous/integratedenergyPASS9.981e-16/1.419e-8;highk33symbolRK4accuracyPASS8.331e-13;timecontrolsPASS. OriginalaccuracyFAILreferenceunresolved/minorder.2649/nonmonotonicmaxima. Selectionauditfoundsynthetic/extendedidenticalcoefficientsandmassivefirstmembermass0;only2uniquecases,NOBHtimeevolutionadmitted. Preserveerror/duplicates,countonce,no rerun. SeparatepredeclaredcomparisonconditioningbyknownanalyticinitialfieldalsoFAILS;initialinterpbiasmax2.829%earlyerror,notsoleexplanation. Plotinspected/ticksfixed,allcriteria/failures/arrays/receipts/hashessaved. Maxsurrogateenergyamp28225.7notnativeinstabilityproof. NEXTpredeclaresource-offmatchedgrid/time/pulsecontrolsfortwouniquecasesandinitialderivativesamplingdiagnosis,beforeaddingreferences;metadata/hashselectionrequired. NoAuroraallocation/physicalsuccessor,native448indexmatch+3historicaldiagnostics/allheaders-inputsunchanged. Userrequestedscalarbenchmarkinvestigation/prompt,providedbutnotyetexecutiondirection;activegoalpreserved. Nativepulse.2/puncture.1failedacceptance/originalgoalincomplete.


2026-10-18 Goal1 native source and endpoint chart audit: private typed actual
native AD/reference connection/Fgradient checks pass. Full nonlinear reduced
point source460states agrees5.47e-13; independent unforced native chain138states
agrees8.29e-13. No native trajectories repeated. Bounded old Pi rejects a regular
stationary same-RWM-gauge exterior harmonic-time Minkowski family: Pi_alpha~1/Omega
while physical fields/constraints/nullness regular. Actual native22 exact-scri
regular-plus-first-jet pole quotient passes6analytic fixtures2.14e-14; invalid
poles rejected and unused higherjets invariant. Full endpoint characteristic/time
compatibility and stable numerical closures remain open. Proposed source-inclusive
velocity chart has finite lapse witness; no full factored inverse/source yet.
Evidence: docs/validation/goal1-minkowski-20261018/native-reduced-source-v1/README.md
and scri-momentum-chart-v1/README.md. Goal1/results/roadmap/Goal2 updated; Goal2
inactive, acceptance protocolv0 draft, no matrix/native reduced run/remote work.


2026-10-18 Goal1 velocity candidate: full W=q_t(native)-beta*g map/inverse now
implemented using native regular/pole parts; no divergent Pi+F subtraction in the
numerical map.460 finite point maps pass;8 independent80-digit first/second/mixed
jet checks max6.02e-15. Six reused exact-scri stationary family fixtures recover
native11 momentum values/first derivatives8.54e-16 using compatible second-jet
beta-source quotient; nonzero and second-tangential controls rejected. Full nonlinear
finite-Omega source460states7.17e-13, preserved source change2.31e-11, independent
unforced native chain138states4.85e-12. Offcurl omission .00755062 retained. No
new trajectory. Nonstationary local exterior harmonic-time endpoint protocol/design
predeclared, not executed; full endpoint dynamics/characteristics/time compatibility,
stable true-face coupled method and scientific acceptance remain open. Evidence:
docs/validation/goal1-minkowski-20261018/velocity-momentum-v1/README.md. Production,
reference, earlier candidate sources/payloads unchanged. Local only, no Goal2 activation.


2026-10-18 explicit user strategy correction: abandon optional50-field spatial
reduction/added wave momenta/independent gradients; preserve code/evidence as
rejected isolated research. Active native Z4c remains first-time/second-space
with native A/P/Theta/Lambda. RWM gauge/reference immutable. Native inventory14
categories and frozen rejected source registry saved under Goal1/native-second-order-v1.
No symmetric first-order energy prerequisite for bounded native coupled pilots.
Native-only exact-scri nonstationary family12fixtures: actual22RHS1.43e-14,
13native RHS firstgrad1.07e-14, full22poleTime4.62e-13; omitted metricgradienttime
fails .04277/Ztime .008008. No reduction include or extra evolved field. Direct
full native22 zero-time actionB16/A16/R24: firstRMS .004318, second .2832, mixed
.06750, commutator .4134, RHS .2737; cube-shellRHS1.0327. Rawstat8.89e-14.
Nextcompare/repair native direct-second/mixed ghost-atlas operators and trueface
finite incoming treatment, then bounded coupled pilots. Exact characteristic/
stable endpoint and scientific gates remain open. No new native trajectory,
remote work, matrix or Goal2 activation. All previous failures retained.


2026-10-19 Goal1 restricted native spherical exact-scri prototype: eight native
components call full22 Cartesian tensor source/RWM; direct second-order method,
active r1, no added momenta/gradients. Clock64/96 and stationary32 through.02,
10roots15segments/five strict restarts. Fine field/first/Hessian RMS
1.93e-5/3.80e-4/5.54e-4; raw constraint RMS refinement ratios .217–.275.
CK01 allregion timestep fraction .60086 fails; nativeoffset CK02 .09725 at.002
and.00067252 at.02 pass, without general temporal-order admission.70 actual
full limiting-bulk symbols pass, regular endpoint DAE characteristics open.
Local native40/43 weakbranch smooth-DAE rate error growsN^(3/43), a failed
generalclosure criterion. No unjustified native/auxiliary boundedness, gauge
replacement or incoming physical radiation condition. Evidence:
docs/validation/goal1-minkowski-20261018/native-radial-scri-v1/README.md.
Preservation audit rechecks2139 prior pins,401 root numerical sources and
production/reference/rejected research. General/mapped3D exactscri, original
scientific gates and matrix remain open. Goal1 active; Goal2 inactive.


2026-10-19 Goal1 weighted endpoint research: nativeOmegaA/OmegaLambda exactfinite
chains18fixtures passactualtensorRWM; no added fields. Hybridnative8linearized
weakcoupled refinement fails evenboundedN48extension. Actualnativeindicialaction
findsmovingordinaryrowdefect~.002; globalrank9fitteddirectD1/D2 restorescoupled
kernel to1.7e-11..3.9e-10(twodeepprecisionfailuresretained). FineweakfieldLinf
3.77e-6→1.34e-8, order2.324; RMS/derivative/modelgatesremainfailed/open.
Fifteenlinearizedroots/nineteensegments/fourstrictrestarts, compactTheta/Zto.01;
commonprearrivalt.003field/epsilon.0434N32fails/.006635N48passes; bothfitted
weakcoefficientindicatorsfail. Positivecoupledmode62unresolved; no generic
Cauchy40/43selection, generalnonlinear/mapped3Dclosure oractualregularfull
characteristicadmission. IndependentoutgoinggaugeADMoraclespass60digits; gauge
unchanged. Evidence:docs/validation/goal1-minkowski-20261018/native-weighted-endpoint-v1/README.md.
Preservation2263priorpins,401baseline/rejectedresearch/referenceunchanged.
Goal1active;Goal2inactive;scientificmatrixnotrun.

## Goal 1 native constraint/boundary phase, 2026-10-19

Latest native boundary work: [source-complete constraint/boundary diagnostics](validation/goal1-minkowski-20261018/native-constraint-boundary-v1/README.md)
retains the actual native second-order system and RWM control. Finite constraint
identities keep all mapped/source commutators; characteristic grading and an
explicit native/normalized power conversion match the independent full22 data.
Eighteen unique coupled native8 linearized exact-gauge pilots reach .001 with
allstage positivity and quantitative native field/derivative/constraint readbacks.
Slow-gauge two-grid errors improve; fast-gauge whole timestep orders are3.03–3.06.
However, refinement retains growing modes3.00→10.60, exact-scri regional Hessian
RMS .0354–.0356 exceeds .01, and temporal contamination .114–.115 exceeds .1.
Six necessary endpoint rate equations alone do not cure coupled growth; rejected
Schur/SAT/Robin candidates and every original failure remain preserved. This is
radial linearized evidence, not full nonlinear/mapped3D endpoint admission.
[Plan v18](validation/goal1-minkowski-20261018/native-constraint-boundary-v1/native-next-plan-v18.json)
prioritizes whole native boundary/endpoint time consistency and stable coupled
closures. Production/reference/rejected code and compatible checkpoints remain
unchanged. Goal1 active, Goal2 inactive; scientific acceptance remains unfrozen.


## Goal1 literature-reproduction priority and entry, 2026-10-19

Latest priority: [full nonlinear 3D literature reproduction](validation/goal1-minkowski-20261018/native-literature-reproduction-v1/README.md)
uses an isolated published-gauge/native-Z4c candidate on the actual Cartesian bulk
and cubed-sphere shell, with no enforced spherical symmetry. The source-linked
comparison covers all six requested areas, and every published vacuum row passes
independent off-constraint checks within2.01e-13. Actual native17 characteristics
are verified on54 evolved 3D jet/covector fixtures; the full regular endpoint
system is still open. Five roots/seven successful segments include stationary
and small gauge evolution through.004 plus a higher-grid one-cycle check.
Stationary raw drift stays7.31e-16; resolved reproduction is not yet established:
interface Hessian RMS .01482 and curvature symmetry .01786 exceed .01, while
sampling uncertainty and genuine half/quarter controls remain unresolved.
An independent artificial-inner-boundary gauge mode145.368864 is distinguished
from the constraint-bearing legacy62 mode. Current staggered runs are much shorter
than crossings .6/5/3; exact scri and the original scientific gates remain open.
The RWM control, production/reference and rejected research are unchanged.
[Plan v19](validation/goal1-minkowski-20261018/native-literature-reproduction-v1/native-next-plan-v19.json)
prioritizes this reproduction over fractional endpoint refinements. Goal1 active,
Goal2 inactive; no reproduction/scientific acceptance matrix has run.


## Literature reproduction: resolution and genuine timestep controls

Latest [literature resolution and timestep pilots](validation/goal1-minkowski-20261018/native-literature-resolution-v2/README.md)
retain the published native equations and full nonlinear 3D bulk/shell evolution.
B48/A32/R96 initial interface Hessian RMS 0.00492443 and maximum 0.0240072 meet
the retained entry budgets. The existing B36 checkpoint reaches common time
0.004: raw H RMS falls to 3.04549e-5 and curvature symmetry maximum to 0.00456690.
Genuine full-step 0.0002/0.0001/0.00005 controls reach 0.004, but RMS time orders
3.438/3.465/1.955 fail 3.5. Matching-coordinate temporal sensitivity is below
0.01 of the spatial proxy; sampling uncertainty still fails 0.1 (maximum 0.7693).
The bounded B48 continuation ends at 0.001627781387, retaining its checkpoint.
Legacy62 matrix attribution separates strong source/interior/inner-lift sensitivity
from small endpoint-row sensitivity without changing the operator. A separate
high-precision audit identifies the literal Gaussian's nonzero, underflowing scri
tail; compatible exact-endpoint initial data remain a distinct case.
[Plan v20](validation/goal1-minkowski-20261018/native-literature-resolution-v2/native-next-plan-v20.json) addresses
continuation, local CPU cost, sampling/time resolution, crossings and actual scri.
Eight roots/fourteen segments are retained. No matrix or Goal 2 activation; Goal 1 active.


## Native literature CPU and longer coupled pilots

Latest [native CPU and longer literature pilots](validation/goal1-minkowski-20261018/native-literature-cpu-v3/README.md)
verify OpenMP execution and checkpoint transfers bitwise through all native data,
with 3.1–6.5 measured wall-time speedup in the tested cases. All four full nonlinear
3D grids reach 0.02 by compatible continuations; eight physical roots/21 segments,
plus four backend verification forks. Finest raw H/M RMS are 2.06e-4/1.04e-4 and
curvature symmetry maximum 0.00333. Interface first/Hessian self-difference RMS
0.02231/0.02019 and maxima 0.08200/0.07740 still fail; sampling indicators still
fail, and the previous timestep-order failure is retained. Actual native17
characteristics identify TT/gauge/constraint sectors on 54 evolved 3D fixtures,
with independent actual constraint-symbol closure. An independent linear gauge
reference matches all native RHS rows on 38 rotated/origin points; finite-time
reference and nonlinear amplitude controls remain open. [Plan v21](validation/goal1-minkowski-20261018/native-literature-cpu-v3/native-next-plan-v21.json)
prioritizes native commutator/interface diagnosis and reference verification.
No matrix, crossing-duration success, actual-scri completion or Goal 2 activation.


## Native literature initial constraint forcing and mapped commutators

Latest [initial native forcing and mapped commutator audit](validation/goal1-minkowski-20261018/native-literature-forcing-v4/README.md)
reuses the four saved 3D initial states bitwise, advances no trajectory, and changes
no coupled equation/gauge/operator. All22 native source rows pass81 arbitrary
nonspherical lapse-jet fixtures. Actual constraint-slope formulas agree with AD
at every active point within1.81e-15. The initial native momentum slope is dominated
by the measured mapped/direct Hessian third-derivative commutator; including the
product defects reconstructs actual M/H slopes within7.51e-11/8.53e-11.
Finest raw M slope RMS .023399 (.15893 at the shell cube-interface) remains nonzero;
halo-only repair is insufficient. This is an initial operator mechanism, not a
later nonlinear-growth or stability proof. Failed diagnostic storage/input/compile
attempts and all prior coupled derivative/time/sampling failures remain.
[Plan v22](validation/goal1-minkowski-20261018/native-literature-forcing-v4/native-next-plan-v22.json) calls for
controlled isolated native-operator comparisons, stage/projection and genuine
nonlinear/time/reference tests before crossing-duration and actual-scri gates.
Eight physical roots/21 segments remain unchanged; no scientific matrix or Goal2 activation.

