# Goal 1: Minkowski foundations, coupled perturbations, and exact scri

2026-10-23: execution shifts to the explicitly authorized out-of-sequence
[puncture pilot](../validation/puncture-pilot-20261023/README.md), in a separate
candidate. Goal 1 remains incomplete and its complete source/evidence history,
failed criteria and unresolved exact-scri requirements are preserved. Goal 2
remains inactive; pilot survival cannot complete Goal 1 or full Goal 3.

## Corrected MPI tile scaling and saved-pair continuation — 2026-10-23

The user authorized MPI scaling on one Aurora debug node, then selection of a
configuration for longer stability evolution toward t=1. The current hardware
limit is one node, with up to twelve distinct GPU tiles; this supersedes the
earlier single-GPU limit below. Keep dt=1e-4, preserve all evidence, and leave
Goal 2 inactive. Convergence and accuracy acceptance remain deferred.

Job **8957214** completed successfully in 16m19s. With the unchanged v32 solver
and fourteen fixed blocks, two forty-step repeats at 1, 2, 3, 6 and 12 ranks
passed all 23 comparisons: fields, jets and checkpoint payloads were bitwise
identical. Tile identities, one visible tile per rank, half-GPU compute-unit
counts, and GPU kernel execution were verified against whole-GPU controls.
Twelve tiles were fastest: 0.179646 s/step coarse and 0.354182 s/step fine,
giving 2.491x and 2.806x speedups over one tile. These are steady-step timings;
periodic diagnostics increase the measured evolution averages to 0.205050 and
0.428711 s/step. See the [v33 evidence](../validation/goal1-minkowski-20261018/native-mpi-tiles-v33/result-review.json).

The saved t=.2 pair has seven blocks. Job **8957254** completed its separate
qualification and continuation to t=.25 in 19m40s. Seven ranks on distinct tiles
were fastest for both original grids (0.123051/0.300820 s per steady step).
Checkpoint block payloads were redistributed without interpolation after exact
physics/reference checks; three resumed steps on each grid matched the preserved
v24 one-rank solver bitwise. All twenty qualification comparisons passed.
See the [qualification record](../validation/goal1-minkowski-20261018/native-mpi-continuation-v34/qualification-review.json).

At t=.25, both grids remain finite and positive. Normalized tangential Lambda
maxima decline to .0140005/.00482817; whole-domain RMS changes by -0.43%/+2.36%
since .2, and Z RMS still grows by 3.99%/4.98%. Coarse H maximum grows despite
falling H RMS. This supports further bounded continuation, not a claim that all
errors have saturated. The coarse .01 screen failure is retained. See the
[milestone review](../validation/goal1-minkowski-20261018/native-mpi-continuation-v34/milestone-t250-review.json).
Both saved trajectories subsequently reached **t=1** at constant dt=1e-4,
using seven ranks on distinct tiles. Jobs **8957499** (48m18s) and **8957581**
(14m47s) completed sequentially on one debug node at a time. Solver invocations
for the two t=.2-to-1 branches totaled approximately 65 node-minutes, excluding
separate benchmark/control work, passive analysis and queue waits.

All eighteen continuation segments passed finite/positive-state checks, exact
dt checks, reference-hash preservation, and diagnostic consistency checks.
The final normalized tangential Lambda maxima are .00647263/.00182451; global
Lambda RMS is 23%/29% above t=.2, while final H/M/Z/Theta RMS is lower. Interface
signals are bounded and oscillatory in this window, not uniformly saturated.
Dense native diagnostics reveal late conformal-Z transients on both grids that
subside by t=1; the coarse peak lies between saved snapshot times. Some
pointwise constraint maxima remain higher than at t=.2. These results do not
establish accuracy, convergence, asymptotic stability, or completion of Goal 1.
Goal 2 remains inactive.

See the [final review and checkpoint provenance](../validation/goal1-minkowski-20261018/native-mpi-continuation-v34/result-review.json),
[stability plot](../validation/goal1-minkowski-20261018/native-mpi-continuation-v34/stability-timeseries.png),
[conformal/physical Z comparison](../validation/goal1-minkowski-20261018/native-mpi-continuation-v34/stability-timeseries-Z-norms.png),
and [execution record](../validation/goal1-minkowski-20261018/native-mpi-continuation-v34/execution-status.json).
An administrative cycle-limit rejection before evolution is preserved separately;
the corrected driver raises the run-length ceiling to the requested cycle 10003
without changing the solver, timestep or saved state. Short implementation
comparisons do not establish long-time stability. Earlier scientific results
and failed criteria remain intact.

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

See the [v29 result](../validation/goal1-minkowski-20261018/native-interface-continuation-v29/README.md), [detailed evidence](../validation/goal1-minkowski-20261018/native-interface-continuation-v29/result-review.json),
and [final checkpoint provenance](../validation/goal1-minkowski-20261018/native-interface-continuation-v29/execution-20261022.json).
The v28 findings below are retained as earlier evidence.

Latest execution priority and corrected GPU results are recorded in the
2026-10-22 bounded-interface-stability section below. The following CFL and
performance overview records the earlier diagnostic context.

Current CFL follow-up: [v14 compactification audit](../validation/goal1-minkowski-20261018/native-timestep-audit-v14/README.md) verifies that conformal
fields and the inverse patch map already include coordinate-speed changes. The
private candidate makes gauge-flag precedence explicit and records individual runtime
limits; a 20-step implementation fork is bitwise equal. Keep cap 1e-4 pending
separate discrete/source stability experiments.

Earlier performance/timestep evidence: the qualified
[v22 CPU comparison](../validation/goal1-minkowski-20261018/native-performance-stage-timing-v22/README.md)
measures mapped cost at 1.59× Cartesian excluding diagnostic blocks and 1.47× end
to end, with bitwise implementation checks. The
[v21 timestep forks](../validation/goal1-minkowski-20261018/native-timestep-forks-v21/README.md)
compare caps .0001 and .000125 through gauge t=.075 and stationary t=.0999.
These are short diagnostic windows, not long-stability success or timestep
promotion. The one-GPU pilot completed but
[failed numerical compatibility](../validation/goal1-minkowski-20261018/native-gpu-comparison-v23/README.md):
coordinate-rounding differences changed corner donor ownership. The isolated
[v24 correction](../validation/goal1-minkowski-20261018/native-coordinate-rounding-v24/README.md)
passes a 20-step CPU bitwise check; GPU verification remains outstanding.
Main continuation remains held and Goal 2 inactive.

The [v25 gauge timestep extensions](../validation/goal1-minkowski-20261018/native-timestep-extensions-v25/README.md)
reach t=.1 at both caps but fail the retained interface connection-symmetry
maximum screen (.01769 versus .01). The two maxima agree within 3.82e-7 relative,
pointing toward spatial/interface effects in this window rather than strong
timestep sensitivity. Preserve this failure; hold further gauge extensions for
diagnosis and retain the main cap of 1e-4.

Status: active in goal mode as of 2026-10-18. Inventory and local diagnostic
pilots completed; native acceptance gates remain open. No acceptance matrix run.
Combines former milestones 1–3. Follow [shared rules](README.md).

## Current priority: bounded interface stability — 2026-10-22

The user authorized the recommended interface refinement experiment after the
corrected GPU checks, keeping dt=1e-4 and prioritizing stability. Convergence and
accuracy acceptance are deferred. Run only on Aurora, one node and one physical
GPU at a time in debug under CompactBinaryMerger. Preserve all prior evidence.
Goal 2 remains inactive; this diagnostic does not complete Goal 1.

The corrected baseline checks passed (job 8954368). The 20-step active-state
comparison and exact GPU restart passed (8954399), but the original full-ghost
CPU/GPU screen remains failed. A separate allocated audit (8954452) finds every
potentially read mapped ghost component below the unchanged tolerance; all 221
mapped violations lie outside the evolution stencil footprint. Cartesian ghost
violations remain unresolved. This admits only the bounded mapped diagnostic,
not full checkpoint interchange or a correctness-qualified Cartesian benchmark.
See the [recorded admission](../validation/goal1-minkowski-20261018/native-interface-stability-v28/mapped-diagnostic-admission.json).

[Experiment v28](../validation/goal1-minkowski-20261018/native-interface-stability-v28/protocol.json)
compares B36/A24/Nr72 and B48/A32/Nr96 through .05, .075 and .1. The coarse
case continues the verified GPU checkpoint chain; fine data are initialized
analytically. Track positivity, constraints and interface localization, preserving
the historical .01 tangential-connection screen without treating it alone as an
instability stop. [Execution record](../validation/goal1-minkowski-20261018/native-interface-stability-v28/execution-20261022.json).

**Completed bounded result:** both grids reached t=.1 on Aurora at dt=1e-4,
with finite diagnostics and positive lapse, chi and metric eigenvalues. At .1,
normalized tangential Lambda maxima are .017692 (coarse) and .006804 (fine);
H RMS is 2.2306e-4 and 5.6186e-5. The coarse .01 screen failure is retained.
The localized maximum still grows with time. A longer bounded stability window
is the next priority; no convergence/accuracy acceptance, long-stability proof,
Goal 1 completion or Goal 2 activation is claimed. See the
[result and preserved checkpoint provenance](../validation/goal1-minkowski-20261018/native-interface-stability-v28/result-review.json).

## Previous priority: performance and timestep diagnosis — 2026-10-20

The following priority text is historical; the dated revision above supersedes
its execution order while retaining its evidence and resource constraints.


The user superseded long-duration continuation with performance optimization and
independent timestep diagnosis. Let the active bounded segment finish safely;
hold subsequent long runs while profiling and optimizing. Preserve every completed
trajectory, checkpoint, failed criterion, production baseline and control.

Measure evolution-only and end-to-end costs, including RHS/derivatives/transforms,
ghost interpolation, dissipation, projection/checks, diagnostics/output and setup.
Demonstrate CPU and GPU throughput against a comparable Cartesian evolution;
≤3× Cartesian cost is a target, not an assumed result. Match full nonlinear native
3D Z4c, all tensor components, FD6/RK4/precision, point count, diagnostic work and
hardware where possible; label inequivalent implementations as references.
Temporary derived buffers and caching of fixed geometry are allowed. No spherical
enforcement, removed equation terms, evolved wave momenta or independent
configuration gradients. Preserve required diagnostics and justify any rescheduling.

Use separately recorded bounded checkpoint forks to diagnose CFL, mapped spacing,
conformal source/pole, dissipation and cap limits, then test larger timesteps with
constraints, drift, positivity, symmetry, interface and endpoint measurements.
Select a defensible timestep with a stated safety margin; a few surviving steps
are insufficient. Keep timestep experiments separate from implementation checks.

Resource revision, 2026-10-20: the user stopped substantial local computation
because of battery drain. Run CPU evolutions, analysis, builds and benchmarks,
as well as GPU work, on **Aurora**. Local use is limited to lightweight coordination
and document edits. Use one node at a time in the **debug** queue under
**CompactBinaryMerger**, retaining bounded jobs and one-GPU verification before
node scaling. Run substantial computation in an allocation, respecting scheduler
and login-node rules. Preserve completed local checkpoints and evidence. This
supersedes both the original local-only restriction and the later local-CPU policy;
archived resource policies remain historical. Goal 2 remains inactive.

Follow [performance evidence](../validation/goal1-minkowski-20261018/native-performance-v9/README.md).
Goal 2 stays inactive. Goal 1 remains unfinished: regular compatible actual-scri
equations, full characteristic verification and no freely prescribed incoming
physical radiation are still required. Performance gains and survival are not
scientific success. Accuracy/convergence acceptance matrices remain deferred
unless needed to diagnose instability or validate an implementation change.

## Immediate priority: literature reproduction in native 3D

User revision on 2026-10-19: reproduce the successful published hyperboloidal
setup before further extending the fractional fitted endpoint candidate. Use the
full nonlinear native tensor solver on the actual Cartesian bulk and cubed-sphere
shell; do not enforce spherical symmetry or substitute radial/linear evolution.
A separately versioned published-gauge candidate is explicitly authorized, with
RWM retained as an unchanged control. Compare the actually evolved published
formulation, variable maps, gauge/damping, background, endpoint and numerical
methods term by term. Keep the published CMC background distinct from later
restoration of the layer. Staggered-scri reproduction is diagnostic; exact scri
and every original Goal1 completion gate remain required. A reproduction milestone
does not activate Goal2. See [comparison and candidate evidence](../validation/goal1-minkowski-20261018/native-literature-reproduction-v1/README.md)
and [historical Plan v22](../validation/goal1-minkowski-20261018/native-literature-forcing-v4/native-next-plan-v22.json);
the stability-first priority above supersedes its execution order.

## Active implementation decision: native second-order Z4c

The user explicitly abandoned the optional first-order spatial reduction. Evolve
the native Z4c system, first order in time and second order in space, retaining
native curvature A, trace P=K_phys-2Theta, Theta, Lambda, metric, lapse and shift.
Do not add wave momenta or independently evolved configuration gradients, including
an equivalent reduction under another name. First-order/pseudodifferential forms
are allowed for characteristic analysis only. Regular changes of native variables
need explicit finite-radius/scri analysis and preserve the underlying equations.

Keep the reference-wave-map gauge as control. The divergent auxiliary canonical
momentum is a rejected-chart defect, not evidence against native Z4c or the gauge.
Do not impose auxiliary boundedness, reference-valued leading fields or unjustified
falloffs at scri. Preserve the reduction code/evidence as isolated rejected research.
The [native evidence inventory](../validation/goal1-minkowski-20261018/native-second-order-v1/native-evidence-inventory-v1.json)
separates directly reusable native results from reduction-specific proofs.

Repair native first/second/mixed operators, true-face interfaces, dissipation and
finite/exact boundaries directly, retaining mapped coefficient/commutator effects.
A symmetric first-order energy proof is not a prerequisite for bounded quantitative
native coupled pilots. It also cannot replace native stability/accuracy evidence.
All original scientific gates, exact-scri/no incoming physical radiation rule,
immutable production/reference requirements remain unchanged; follow the current resource policy above.

## Objective

Establish a verified, stable and convergent local-CPU hyperboloidal Z4c foundation
on the Cartesian bulk plus six cubed-sphere shell. Demonstrate geometry and
operator correctness, stationary Minkowski preservation, and small coupled
perturbations. Exact scri is a required completion gate: finite-radius success
alone is insufficient. Reuse existing derivations and evidence before writing
new mathematics or repeating runs.

## User decisions and remaining refinement

Confirmed on 2026-10-18:

- Diagnose at finite radius first, then include exact scri within this goal.
- Run small local CPU pilots; proceed to a local three-resolution acceptance
  matrix only after operator checks pass.
- Preserve the production baseline; no Aurora jobs are part of this activation.
- Prioritize boundary/operator repairs while retaining the existing
  reference-wave-map gauge as the control. Introduce a separately versioned gauge
  candidate only if characteristic or scri-compatibility tests require it. This
  original restriction is superseded for the explicitly authorized literature
  reproduction candidate above; the RWM control remains unchanged.

Execution refinements to settle autonomously using pilots:

- Use at least two measured crossing times for small-perturbation acceptance and
  a ten-crossing-time coarse stationary/late-growth screen. Determine travel
  times from the actual coefficients and packet support. Do not infer a crossing
  time from the tangent characteristic at scri or from an unused slow branch.
- Draft accuracy budgets are given below. Lock them with the pilot protocol;
  they are proposals, not claims that previous runs passed them.

## Execution

### 1A. Inventory, geometry, and spatial operators

1. Create a source/evidence index: mark existing results as reusable, incompatible,
   failed, or untested. Identify the smallest missing test in each category.
2. Verify the physical and reference metric distinction, map Jacobian/Hessian,
   derivative contractions, angular seams/corners, and Cartesian-shell transfer.
   Field components remain global Cartesian; do not transform Lambda as a tensor
   connection under a patch change. Test nonconstant analytic fields, not only
   stationary constants. Include first, second, and mixed derivatives.
3. Check Omega/height jets against an independent high-precision pullback near
   transition endpoints and scri. Audit whether scalar smooth-switch cancellation
   affects any Z4c code; do not assume that it does. Verify the continuous
   compactification and its differentiability before modifying the sampling.
4. Construct/test compatible radial SBP derivatives and penalties as needed.
   Include variable coefficients, advection, mixed derivatives, interpolation and
   dissipation in the checks that claim coupled stability. A radial energy
   identity alone does not certify the interfaces or complete operator.
   Hold the control gauge fixed in these comparisons. Preserve its original
   operator as a negative/control case so improvements can be attributed to the
   numerical change. A failed gauge admissibility check triggers a separate
   formulation candidate, not an unrecorded simultaneous change.
5. Extract the full characteristic system from the actual candidate RHS. Test
   oblique covectors and evolved admissible states; distinguish finite-radius
   incoming modes from the exact-scri limit. Test gauge degeneracies and the
   conditioning of the characteristic basis.

### 1B. Stationary Minkowski and exact endpoint closure

1. Start from the exact Minkowski reference in an isolated native candidate.
   Compare raw continuum and discrete RHS residuals independently, including any
   existing reference-roundoff correction. No new subtraction may hide a defect.
2. At finite outer cuts, use derived incoming characteristic treatment and vary
   the cut while keeping the reference fixed. Compare only after the boundary
   can causally influence the measured diagnostic. Move the cut without silently
   changing the physical map/reference or sampled comparison region.
3. Derive and implement regular exact-scri equations for the selected variables
   and gauge. Check preservation of nullness, pole-numerator conditions and their
   required spatial/time compatibility. Reuse existing residue/higher-jet audits;
   cancelling the value-only leading pole is not sufficient.
4. Evolve the reference through the declared stationary screen. Report metric
   drift, raw H/M/Theta/Z, algebraic constraints, endpoint compatibility and
   characteristic signs. Compare finite-cut and exact-scri behavior separately.

### 1C. Small perturbations and coupled evolution

1. Use smooth small angular lapse/shift pulses on constraint-satisfying spatial
   Minkowski data. Lapse/shift changes do not themselves require solving the
   vacuum spatial constraints. Evolve all native metric/gauge/Z4c fields.
2. Use intentional constraint perturbations to test the checked subsidiary
   predictions and boundary injection/reflection. Their correct answer is not
   zero constraint amplitude; compare with an independently resolved solution.
3. Exercise both tensor polarizations in frozen-principal and linear operator
   propagation tests, including oblique interface crossings. Do not call these
   nonlinear constraint-satisfying radiative initial data. A complete linearized
   native tensor propagation test requires a documented compatible data/oracle
   construction; if missing, derive it or report the coverage gap explicitly.
4. Compare small-amplitude native runs with a source-consistent independent
   linearized solution where available. An exact coordinate-map solution is an
   oracle only if its gauge/source equations match, its slices remain spacelike,
   and its required jets have independent checks. Manufactured forcing tests are
   separate consistency tests, not unforced evolution acceptance.
5. Cross the Cartesian-shell and angular interfaces with resolved packets. Measure
   native field/derivative errors and constraints at exact scri as well as in
   the interior. A stationary-reference pass cannot replace these tests.

## Draft quantitative gates

Freeze a machine-readable protocol after pilots and before acceptance runs.
Proposed defaults:

- Analytic geometry/jet and matrix-identity relative scaled errors <= 1e-10 at
  well-conditioned points; use absolute tests for exact-zero targets and explicit
  condition-aware bounds elsewhere. Record precision and negative controls.
- Three-resolution field and derivative errors decrease in L2 and Linf at common
  times; report measured order against the selected operator's predicted order.
  Proposed minimum resolved fine-pair order is 1.5, subject to pre-run adjustment
  justified by endpoint/interface consistency. A coarse non-asymptotic failure
  permits one declared refinement extension, reusing completed runs.
- Independent linear-solution field error <= 1% in L2 and <= 5% in Linf relative
  to the declared initial perturbation scale. Apply the same declared derivative
  normalization; do not normalize by a waveform crossing zero.
- On constraint-satisfying perturbations, dimensionless constraint L2 <= 1% of
  initial perturbation amplitude, with length scales explicitly included for H,
  M and Z. Require refinement and bounded resolved growth. Intentional constraint
  injection uses oracle-error budgets instead of a zero-constraint test.
- Stationary screen: proposed maximum scaled field drift <= 1e-9 and dimensionless
  constraint norms <= 1e-8. Establish all normalizations before running. A failure
  is retained even if later diagnosis shows that a different budget is warranted.
- Half/quarter timestep contamination and independent-reference uncertainty each
  below 10% of the finest spatial error. Assess source stiffness, not only wave CFL.
- No incoming physical characteristic at exact scri; outgoing/tangent gauge and
  constraint branches for the chosen pure-outflow formulation. Endpoint identities
  must converge to their derived limits. Floating-point sign tolerances must be
  tied to coefficient uncertainty, with raw signed values preserved.
- No resolution-worsening interface/outer growth; compare reflection with a
  resolved reference. Set a numerical reflection budget before packet tests.

If full tensor propagation or endpoint closure is missing, keep that subgate
open; do not silently narrow this goal and mark all former milestones complete.

## Deliverables and completion

Save operator/geometry comparisons, characteristic maps, stationary histories,
three-grid field/derivative/constraint plots, spatial error maps, radial profiles,
interface reflection tests, exact-scri time series, timestep controls and hashes.
Separate mathematical identities, isolated diagnostic evolution, and native
coupled evolution results. Record precisely which initial-data sectors are covered.

Completion requires all locked gates and the handoff below. If a formulation
obstruction appears, document its minimal reproducible counterexample and propose
a bounded repair; do not launch larger grids indefinitely.

## Required final task: refine Goal 2

Update [Goal 2](02-angular-minkowski.md) with the selected equations/gauge,
endpoint and interface closures, source/binary hashes, compatible initial-data
family, measured crossing times, practical grids, timestep restrictions and
accepted error budgets. Identify which Goal 1 controls must accompany the large
pulse. Preserve rejected candidates. Write a concise recommendation explaining
why Goal 2 is ready or which specific gate prevents it. Update this file and the
index with evidence links; do not activate Goal 2 automatically.

## Results

Latest [initial native forcing and mapped commutator audit](../validation/goal1-minkowski-20261018/native-literature-forcing-v4/README.md)
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
[Plan v22](../validation/goal1-minkowski-20261018/native-literature-forcing-v4/native-next-plan-v22.json) calls for
controlled isolated native-operator comparisons, stage/projection and genuine
nonlinear/time/reference tests before crossing-duration and actual-scri gates.
Eight physical roots/21 segments remain unchanged; no scientific matrix or Goal2 activation.

Latest [native CPU and longer literature pilots](../validation/goal1-minkowski-20261018/native-literature-cpu-v3/README.md)
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
reference and nonlinear amplitude controls remain open. [Plan v21](../validation/goal1-minkowski-20261018/native-literature-cpu-v3/native-next-plan-v21.json)
prioritizes native commutator/interface diagnosis and reference verification.
No matrix, crossing-duration success, actual-scri completion or Goal 2 activation.

Latest [literature resolution and timestep pilots](../validation/goal1-minkowski-20261018/native-literature-resolution-v2/README.md)
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
[Plan v20](../validation/goal1-minkowski-20261018/native-literature-resolution-v2/native-next-plan-v20.json) addresses
continuation, local CPU cost, sampling/time resolution, crossings and actual scri.
Eight roots/fourteen segments are retained. No matrix or Goal 2 activation; Goal 1 active.

Latest priority: [full nonlinear 3D literature reproduction](../validation/goal1-minkowski-20261018/native-literature-reproduction-v1/README.md)
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
[Plan v19](../validation/goal1-minkowski-20261018/native-literature-reproduction-v1/native-next-plan-v19.json)
prioritizes this reproduction over fractional endpoint refinements. Goal1 active,
Goal2 inactive; no reproduction/scientific acceptance matrix has run.

Latest native boundary work: [source-complete constraint/boundary diagnostics](../validation/goal1-minkowski-20261018/native-constraint-boundary-v1/README.md)
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
[Plan v18](../validation/goal1-minkowski-20261018/native-constraint-boundary-v1/native-next-plan-v18.json)
prioritizes whole native boundary/endpoint time consistency and stable coupled
closures. Production/reference/rejected code and compatible checkpoints remain
unchanged. Goal1 active, Goal2 inactive; scientific acceptance remains unfrozen.

Latest weighted endpoint work: [native weighted operators and coupled linearized pilots](../validation/goal1-minkowski-20261018/native-weighted-endpoint-v1/README.md)
verify native Omega*A/Omega*Lambda chains against the actual tensor/RWM source,
retaining eight native spherical fields and no added momenta/gradients. A hybrid
fractional-fit/ordinary stencil fails coupled refinement; the one bounded extension
also fails. Actual source action identifies a moving-transition defect in the
native indicial cancellation. Global direct-D1/D2 function fitting reduces that
defect from~.002 to near roundoff and improves native linearized weak-profile
field Linf3.77e-6→1.34e-8, with fine Linf order2.324. RMS/derivative orders,
model uncertainty and two fine absolute precision criteria remain failed/open.
Fifteen linearized roots/nineteen segments include four strict restarts and compact
Theta/Z propagation through.01. Common-time pre-arrival field leakage fails on32
cells and passes on48; weak-coefficient indicators fail on both and generic
40/43 Cauchy-class selection is unresolved. A positive coupled mode~62 persists;
no general stability, nonlinear/mapped3D endpoint or full regular-characteristic
admission follows. Independent outgoing ADM gauge controls pass native source
and constraints; RWM is unchanged. All old failures/baselines are preserved.
[Plan v17](../validation/goal1-minkowski-20261018/native-weighted-endpoint-v1/native-next-plan-v17.json)
prioritizes coupled boundary/conditioning/causal and native regularity repairs,
then general nonlinear/mapped3D tests. Goal1 remains active, Goal2 inactive;
scientific acceptance protocol is still draft and its matrix has not run.

Latest numerical exact-scri work: [restricted native spherical prototype](../validation/goal1-minkowski-20261018/native-radial-scri-v1/README.md)
evolves eight independent native components using the full Cartesian tensor
Z4c/RWM source and an actual r=1 node, with no added momentum/gradient fields.
Smooth-clock64/96 and stationary32 reach.02 through five strict restarts;
ten roots/fifteen segments preserve all earlier evidence. Fine normalized RMS
field/first/Hessian errors1.93e-5/3.80e-4/5.54e-4 decrease under refinement;
raw constraints decrease too. Native-offset arithmetic retains the unsubtracted
RHS and improves the failed.60086 regional timestep fraction to.09725 at.002
and.00067252 at.02, without resolving every temporal order. Seventy full native
limiting-bulk symbols pass at numerical endpoint states; the regular endpoint
characteristic system is still unadmitted. Crucially, the smooth endpoint DAE
fails the local native40/43 weak-branch action, with error growing as N^(3/43).
No bounded-A/Lambda assumption, auxiliary bound or gauge-failure inference is
made. This prototype is not mapped3D/general exact-scri closure or scientific
acceptance. N32 entry, CK01 timestep and weak-class failures remain preserved.
[Plan v16](../validation/goal1-minkowski-20261018/native-radial-scri-v1/native-next-plan-v16.json)
prioritizes the native regularity/error class and compatible singular-operator
closure before general mapped3D admission. Goal1 active; Goal2 inactive; scientific
protocol remains draft and the three-resolution matrix has not run.

Latest native boundary repair: [compatible finite incoming history](../validation/goal1-minkowski-20261018/native-incoming-history-v1/README.md)
uses actual unforced native F/DF data for the unchanged original angular pulse.
No extra evolved fields, gauge or primitive/mapped-operator change. Native
source/time/basis checks pass, and the disabled4step state/reference control is
bitwise identical. Full coupled short physical M RMS now falls4.79194e-7 to
9.98246e-8 under refinement; the fine outer error is about228x smaller than with
homogeneous incoming data. Eight roots/nine segments include dt/history controls,
119 finite symbols, active stationary4 and strict CK07 continuation to.002.
Spatial sampling and tiny-difference time-order limitations remain; a time-Taylor
boundary history is not a general long-time or exact-scri closure. Original
failures are preserved. Full regular endpoint characteristics/equations/stability,
independent constraint prediction, causal/long/oblique propagation and scientific
matrix gates remain open. Goal1 active; Goal2 inactive; acceptance still draft.

Latest native coupled work: [angular and intentional-constraint entry pilots](../validation/goal1-minkowski-20261018/native-entry-pulses-v1/README.md)
now cover both original missing sectors on two finite-radius grids, with fine
half/quarter timestep and half-amplitude controls. Nine short quantitative
screens and231 actual finite symbol samples pass. Ten unique roots/eleven
segments include a compatible native constraint checkpoint continuation to.002.
No additional evolved fields or operator/RHS/gauge change; the constraint adapter
changes initial native Theta/P/Lambda only. Production/reference remain immutable.
The spatial interpolation screen fails, and angular physical M increases globally
on refinement. A bounded no-SAT control localizes the dominant outer error to
finite SAT/initial incoming gauge-data mismatch (outer M reduced1040x, core
unchanged); no untreated boundary is adopted. These failures, unresolved tiny
constraint timestep orders and earlier failures remain. Full regular exact-scri
closure/characteristics, causal/long-time/propagation and scientific matrix gates
are open. Goal1 active; Goal2 inactive; acceptance protocol remains draft.

Latest source-inclusive endpoint analysis: [native normal indicial system](../validation/goal1-minkowski-20261018/native-normal-indicial-v1/README.md)
retains the actual RWM/damping terms and native second spatial order. Correct
mixed-order normal constraints have four tangent/four outgoing branches. A
subsidiary root alone fails to lift to a native single-power mode; that failure
and the first shifted attempt remain failed. A coupled radial formal branch
with configuration power40/43 passes six subleading equations and independent
finite-source residual orders5.97-5.99. It has weak native curvature/connection
power-3/43, with constraints appearing two levels later. This is not a global
solution, instability, gauge failure or selection result for regular data.
The stationary formal control fails at fourth order. No falloff, numerical
source/gauge change, extra field or PDE trajectory is adopted. Regular-data/
constraint-error propagation, full actual regular endpoint characteristics and
stable mapped closure remain open, along with original scientific gates.
Goal1 active; Goal2 inactive; acceptance protocol still draft.

Latest endpoint work: [analytic native compatibility identities](../validation/goal1-minkowski-20261018/native-endpoint-identities-v1/README.md)
explain the corrected tau/Z rank2 (earlier narrative rank4 was wrong; raw data
unchanged). Six explicit conditions reproduce all scalar/gauge pole-time rows;
conditional native constraint/nullness falloffs are derived, not imposed. An
actual singular-branch control prevents extending them indiscriminately.
Old197 fixtures fail the next time layer in48/72 cases. The202-row extension
passes48 geometric/clock/compensated-shear fixtures below3.95e-13, retaining
native second spatial order. Equivalent RWM source algebra is verified, but19
deep finite binary64 raw criteria fail and remain failed; no numerical adoption.
A regular leading metric flow is derived without fixing its value or radiation.
No new evolved field, trajectory, gauge or numerical candidate change. Full
invariant endpoint/regular characteristics/stable closure and all scientific
acceptance gates remain open. Goal1 active; Goal2 inactive; matrix not run.

Latest exact-scri analysis: a [native197-row jet compatibility system](../validation/goal1-minkowski-20261018/native-endpoint-completion-v1/README.md)
retains the original fixed-first and six-normal failures. Allowing two geometric
normal metric traces gives24 compatible general/clock fixtures; the full24
first-jet map has rank6 and the105 higher-jet map rank49. Both tensor shear
amplitudes are preserved in24 additional compensated fixtures, with residuals
below3.95e-13; uncompensated shear failures remain failed. Sampled actual-source
row identities link native tau/Z to time compatibility, without imposing a new
falloff. Independent12 clock targets and72 higher-jet invariance checks pass.
No evolved field, gauge/numerical candidate change or new trajectory. This is
necessary local consistency, not invariant regular endpoint evolution, actual
endpoint characteristic admission or stable closure. Those and all original
scientific gates remain open. Goal1 active; Goal2 inactive; matrix not run.

Latest finite-radius numerical work: a [native total-RHS algebraic tangent candidate](../validation/goal1-minkowski-20261018/native-algebraic-tangent-v1/README.md)
changes only metric/A rates after all native operator/advection/KO/SAT terms;
no added fields, physical constraint reset or gauge change.33 point checks and
2,784 reused true-face controls pass, and the disabled control is byte-identical.
Short stationary drift is2.26e-14, with unprojected discrete RHS1.09e-11 reported.
Full/half/quarter native clock pilots reach.02 through compatible checkpoints;
field/derivative/constraint screens pass. Temporal contamination is<5.73e-6 of
spatial error, but RMS orders2.0069/1.7251/1.7831 **fail** the declared2.5 gate.
This is partial improvement, not third-order or stability admission. Old failures,
production/reference and earlier clock/TT checkpoints remain unchanged. General
exact-scri equations/characteristics/closure and all original scientific gates
remain open. Goal1 stays active; Goal2 inactive; acceptance matrix not run.

Latest endpoint work: the [full native limiting bulk symbol](../validation/goal1-minkowski-20261018/native-scri-symbol-v1/README.md)
passes182 state/covector cases, with an independent physical finite-map extraction
agreeing below1.10e-14 (constraints5.63e-13). At all26 outward normals, the
admissible20 system has10 tangent/10 outgoing modes, including all constraint,
gauge and physical sectors. The raw determinant/A-trace2 sector retains a Jordan
block; raw22 strong hyperbolicity and the regular endpoint DAE are not claimed.
No extra PDE fields or incoming physical scri radiation are introduced.

The [native geometric trace/Theta source candidate](../validation/goal1-minkowski-20261018/native-trace-chart-v1/README.md)
replaces two existing native fields, Q=(P+2Theta-3w)/Omega and tau=Theta/Omega;
all other native fields and RWM equations remain. It is source analysis only,
with no numerical adoption. Full22 finite mapped-back RHS checks agree below
1.37e-13;5 full clock and2 linear TT endpoint source controls agree below1.59e-14.
Combined numerator compatibility is below5.81e-14. No separate nullness O(Omega²),
reference-leading condition, zero-tau assumption or auxiliary momentum bound is
imposed. However two deep finite-Omega raw precision criteria fail, with errors
3.05e-7/.0017635 at Omega≈2e-4/2e-6. The failures remain failed and motivate regular
numerical assembly and compatibility/error control, not a gauge-failure inference.
General invariant endpoint hierarchy, actual regular-boundary characteristics,
stable numerical closure and scientific acceptance remain open. Goal1 active;
Goal2 inactive. The existing numerical tensor/clock checkpoints are unchanged.

Latest native progress: both unchanged full coupled tensor trajectories now
reach t=.005 by compatible checkpoint continuation, with nine-region field,
first/Hessian, raw/normalized constraint and positivity checks passing. Final
plus/cross field RMS errors are .00016745/.00015443 and first RMS
.00149184/.00121651. Cross full/half/quarter amplitude controls also pass the
short model-contamination screen. No profile/normalization/gauge/source change,
new full-trajectory root or scientific matrix was used. This is bounded early
propagation, not complete packet/interface/causal-boundary acceptance. See
[native continuation](../validation/goal1-minkowski-20261018/native-tensor-propagation-v1/README.md).

[General nonlinear leading-scri analysis](../validation/goal1-minkowski-20261018/native-scri-leading-v1/README.md)
now solves the actual native22 pole conditions on96 nonreference positive
metric fixtures, retaining finite native A/P/Theta/Lambda and permitting general
leading spatial metrics. Actual pole residuals are below1.50e-14 and14 prior
clock/TT fixtures are reused. A manual Lambda subtraction error was rejected
and retained. All96 arbitrary higher-jet completions fail further compatibility;
leading pole cancellation alone remains insufficient and is not a boundary reset.

[Linear TT endpoint hierarchy](../validation/goal1-minkowski-20261018/native-tensor-endpoint-v1/README.md)
now passes both existing oblique tensor fixtures: actual22 RHS and13 first RHS
gradients agree with independent80digit time targets below3.79e-16; all22 pole-
time/nullness/Z conditions pass below3.34e-15. Omitting metric-gradient time
terms gives defect5.4086. This is amplitude differentiation of the actual native
system, not additional evolved variables or nonlinear data admission. General
nonlinear time compatibility, actual full endpoint characteristics and stable
native numerical exact-scri closure remain open. Goal1 active; Goal2 inactive.

Current coupled tensor evidence: the [native TT pilots](../validation/goal1-minkowski-20261018/native-tensor-pilot-v1/README.md)
evolve all22 native Z4c fields without forcing, added state or gauge change.
Both B12/A8/R16 and B18/A12/R24 entry cases fail the original L=.15
field/derivative/constraint budgets; those failures remain failed. The actual
uniform FD4/native22 action control also shows coarse underresolution and
approaches fourth order on fine halvings. This does not admit mapped-interface
stability. Independent continuum quadrature measures physical configuration
gradient length .01292881647, agreeing across quadratures to3.05e-13 relatively.

A separately pre-recorded physical-length protocol keeps epsilon and field
amplitude unchanged. Both B36/A16/R64 four-step pilots pass its nine-region
entry budgets through about t=.00106121; plus/cross first RMS errors are
.001419/.001153 and configuration-Hessian RMS .001559/.001481. The original
seed-width criteria are also reported on the new grid and remain failed.
Short half/quarter timestep and half-amplitude controls use a common time,
with CK04 continuation after a wall-time stop, without replaying the trajectory.
Their temporal and model contamination screens both pass; see
[quantitative control review](../validation/goal1-minkowski-20261018/native-tensor-pilot-v1/short-controls-review-v4.json).
These are short pilots, not tensor propagation, long-time stability or a
three-resolution acceptance matrix. Exact-scri characteristics and regular
compatible stable endpoint equations, original angular/intentional constraints,
oblique propagation, causal boundaries and long-duration controls remain open.
Goal1 remains active and Goal2 inactive.

Current native evidence: the [native tensor oracle/source phase](../validation/goal1-minkowski-20261018/native-tensor-source-v1/README.md)
now supplies both polarizations' native curvature, trace, connection, consumed
spatial jets and time targets. Independent Cartesian physical pullback/ADM
checks agree within3.75e-79 at80 digits; actual native22 source checks agree
within1.44e-14, including linear exact-scri quotients and origin controls. The
C++ runtime oracle agrees within5.69e-13. These are point/source checks, with
zero added evolved fields; native tensor propagation and translated-centre
oblique seam crossings remain open.

A six-case finite-Omega approach retains two raw1e-10 precision exceedances:
unfactored native source error reaches6.67e-9 at Omega≈2e-8, while error*Omega
is about1e-16 and the compatible known-face quotient agrees within2.22e-16.
This is cancellation sensitivity, not a gauge or auxiliary-momentum obstruction.
Regular native endpoint assembly, full actual characteristics/time compatibility
and stable numerical exact scri remain required.

The [late native clock continuation](../validation/goal1-minkowski-20261018/native-clock-late-v1/README.md)
reuses the saved.3 trajectory and unchanged candidate through t=.4 in ten
continued segments. All nine-region short-pilot field/first/Hessian and raw/
normalized constraint budgets pass at each stop. Final largest regional RMS
values are .0002787/.002559/.003581, and normalized constraint RMS .001449.
No new trajectory root, source change, forcing or repeated trajectory is used.
Causal finite-cut comparisons, full tails, longer timestep controls and resolved
late growth remain open. Earlier spatial refinement, measured interface phase
and complete oracle-uncertainty checks remain reusable within their scope.

The [native next plan](../validation/goal1-minkowski-20261018/native-endpoint-identities-v1/native-next-plan-v12.json)
keeps the native second-order method and RWM control. The original angular and
intentional-constraint cases, full native tensor propagation, general numerical
exact scri and all three-resolution scientific acceptance gates remain intact.
Native NGHOST3 means FD4; historical metadata corrections and all failures are
preserved. Production, the separate reference and rejected research stay
immutable. Acceptance remains unfrozen; Goal1 active, Goal2 inactive.

The active method is native Z4c, first order in time and second order in space.
The user rejected optional spatial reduction and equivalent added-field methods;
wave momenta/independently evolved gradients are excluded from the implementation
plan. Native curvature/trace/connection/constraint variables are retained. Rejected
research and its original evidence remain isolated and unchanged. Geometry,
reference jets, actual native symbols, operator/commutator findings and compatible
native checkpoints are reused within their verified scope. A symmetric first-order
energy proof is not a prerequisite for native pilots. Auxiliary momentum divergence
imposes no scri boundedness, reference-valued leading fields or unjustified falloff.

The older phase descriptions below are historical. References to continuing a
50-field/velocity reduction are superseded by the explicit decision above; numerical
failures and the limited reusable native evidence remain preserved.

Historical phase (superseded): the [full velocity momentum candidate](../validation/goal1-minkowski-20261018/velocity-momentum-v1/README.md)
uses W=q_t(native)-beta*g=Pi+F with a factored physical trace/Theta inverse. All460
finite-radius map checks pass, and eight independent80-digit first/second/mixed
checks agree within `6.02e-15`. On the six regular exact-scri stationary fixtures,
all native11 momentum values/first derivatives reconstruct within `8.54e-16`;
nonzero and second-tangential source-pole controls are rejected. This admits the
earlier singular-Pi witness with the gauge and reference unchanged.

The full nonlinear finite-Omega velocity-reduced source passes460 actual native
source/action comparisons (`7.17e-13`) and138 independent unforced native chains
(`4.85e-12`). The old full-source change of variables passes `2.31e-11`. No new
trajectory ran; the earlier curl-omission and scientific failures remain failed.
A regular chart on these stationary fixtures does not prove endpoint dynamics.
The [next endpoint protocol](../validation/goal1-minkowski-20261018/velocity-momentum-v1/endpoint-dynamics-protocol-v1.json)
predeclares a nonstationary same-gauge exterior family and actual mixed native
endpoint time chain; it is unexecuted. Full endpoint dynamics/characteristics/time
compatibility, stable true-face closures, coupled pilots and scientific acceptance
remain open. Goal1 stays active; Goal2 remains inactive.

Historical phase (superseded): the [complete nonlinear reduced point source](../validation/goal1-minkowski-20261018/native-reduced-source-v1/README.md)
now passes460 compatible native/synthetic states (`5.47e-13`) and a separate138-state
unforced actual native spatial/time chain (`8.29e-13`). Private AD maps, physical
reference connection gradients and source first derivatives are independently
checked. These are finite-Omega point equations; no native reduced trajectory ran.

The [exact-scri chart witness](../validation/goal1-minkowski-20261018/scri-momentum-chart-v1/README.md)
rejects bounded old Pi as a general endpoint chart: a regular, stationary same-gauge
exterior Minkowski family has Pi_alpha proportional to1/Omega. Physical fields,
vacuum constraints and scri nullness remain regular. The actual22 native endpoint
point equations, evaluated with compatible first-jet pole quotients, are stationary
to `2.14e-14` on six fixtures. Incompatible poles are rejected, with no floor or
reset. This is neither a general gauge obstruction nor a stable endpoint closure.
A source-inclusive velocity momentum has a bounded lapse component; its full
factored inverse/source and actual endpoint characteristics/time compatibility are
next. Keep the RWM gauge and all prior failed criteria. True-face coupled transport,
endpoint numerical closure, scientific derivative budgets and the frozen acceptance
matrix remain open. Goal2 is inactive.

The earlier phase reports below describe their scope when executed; the latest
source checks supersede their statements that the finite-Omega P source is missing.
They do not supersede any failed scientific criterion.

Historical phase (superseded): the [holonomic metric chart](../validation/goal1-minkowski-20261018/holonomic-chart-v1/README.md)
passes independent80-digit value/first/second/time maps (worst scaled error
`9.07e-16`) and stationary/reference fixtures through the exact-scri reference.
The old saved pulse's differential det-one consistency fails: second-jet
reconstruction `7.21e-7`, determinant second derivative `2.01e-6`. Those raw
jets are not reset or admitted as new reduced data. The
[exact nonlinear momentum/configuration source map](../validation/goal1-minkowski-20261018/holonomic-momentum-v1/README.md)
passes460 actual/synthetic point states and24 fixed-q source-invariance pairs.
It uses unchanged native ConformalRHS/RWM rows; no frozen-wave or manufactured
source substitutes. Native momentum inversion and live tracefree A pass;
configuration data remain unchanged. Full native P-source/Frechet derivatives,
evolved reduction constraints, true-face boundary and exact-scri source equations
still need implementation/admission. Scientific acceptance has not started.


Historical phase (superseded): [full actual frozen-native20 graph energy](../validation/goal1-minkowski-20261018/full-native-graph-v1/README.md)
checks1,600 compatible states and32 independent all-grid actions. Exact mixed
native action and exterior/geometric/commutator identities pass; the commuting
shortcut fails. Positive commutator work persists for interface grid data,
while smooth-profile work decreases. This is not proof of instability or a
uniform stability estimate. A separate [first-order numerical reduction](../validation/goal1-minkowski-20261018/first-order-reduction-v1/README.md)
keeps the RWM gauge and adds independently evolved gradients. Its corrected
500-jet native-action comparison, full50 characteristic symmetry and explicit
frozen graph energy bound pass on the five actual evolved states. The original
wrong-sign comparison failure is retained; completed graph checks were reused.
No nonlinear source implementation, native reduced evolution or regular
exact-scri equations exist yet. Next admit a holonomic metric chart, exact native
source/Frechet transformations and reduction constraints, then true-face boundary
and live/source/time stability. Scientific derivative budgets remain unmet.


Latest repair: [physical true-face interface v1](../validation/goal1-minkowski-20261018/physical-interface-v1/README.md)
now restores actual native bulk-to-shell derivative dependency. It couples the
cube and all six shell seams with positive cell-centered norms, mapped cofactor
splits and norm-adjoint face traces. Actual native22/16 derivative-slot tests,
byte-identical disabled control and immutable-reference checks pass at
B16/A16/R24. Three zero-time mapped operator grids show decreasing
first/second/mixed/curl errors (minimum resolved fine order1.767), with an explicit
nonzero geometric volume term. This is **not** complete coupled stability or an
acceptance matrix. Large Hessian/curl errors, full native20 mixed energy,
variable/source/projection/time stability and a compatible true-face outer SAT
remain open. The old v4 last-center penalty cannot be transferred silently.
Exact scri and all scientific gates remain mandatory; Goal 2 is inactive.


Latest local phase: [native finite-radius SAT v4](../validation/goal1-minkowski-20261018/full-boundary-v4/README.md)
passes786 full native-jet fixtures and13,824 live boundary point certificates,
with affine work and short stationary/control/checkpoint checks recorded. This
is not complete-method admission: the [actual donor dependency probe](../validation/goal1-minkowski-20261018/native-donor-operator-v1/README.md)
finds **zero bulk-to-shell jet transmission** in nodal-SBP candidates v1–v4.
All1,968 bulk-sourced shell donor targets are unused radial ghosts; reverse
coupling exists. Stop longer/refined native trajectories until this structural
defect is repaired. Production/reference remain immutable.

Archived true-face cell-centered SBP coefficients restore transmission and frozen
energy control in bounded native20 models. Original central/dual accuracy failures
remain recorded, including the one central refinement extension. A uniform
control identifies insufficient resolution of the old steep packet. A new,
explicitly versioned [compact C15 profile](../validation/goal1-minkowski-20261018/packet-profile-v2/README.md)
resolves that control. Its dual-adjoint interface still fails five derivative-Linf
criteria. Its [central interface](../validation/goal1-minkowski-20261018/central-interface-v2/README.md)
passes all60 convergence criteria (minimum derivative fine orders3.539/3.533)
and field amplitude budgets, but derivative amplitudes remain too large
(RMS .09939/Linf .2524 versus draft .01/.05). No further extension is scheduled.
These are isolated principal models, not nonlinear native evolution. Full mapped
interface implementation, exact-scri endpoint and scientific acceptance remain
open. Goal 2 stays inactive.

2026-10-18: activated locally. The [evidence index](../validation/goal1-minkowski-20261018/source-evidence-index.json)
classifies existing checks as reusable, incompatible, failed or untested and
names the smallest missing tests. Of 451 frozen-index files, 448 match; three
historical diagnostic scripts differ. All numerical sources and inputs match
that index. All 401 numerical source/input files recorded at activation, the
production binary and the immutable reference helpers remain unchanged; see
[preservation audit](../validation/goal1-minkowski-20261018/baseline-preservation.json).
Research snapshots, diagnostic binaries and native pilot outputs are isolated in
`build-goal1-minkowski-20261018`. No remote jobs/controllers were started.

| Gate | Evidence and current limit |
| --- | --- |
| Analytic geometry and reference jets | [90-digit independent pullback/map audit](../validation/goal1-minkowski-20261018/geometry-review.json): 6,184 comparisons, worst scaled error `1.16e-14`; omitted-Hessian control fails clearly. No Z4c smooth-switch repair is justified by this check. Actual donor-plan and coupled energy gates remain open. |
| Actual characteristic system and algebraic extension | [142-sample pilot](../validation/goal1-minkowski-20261018/characteristic-pilot-review.json): 30 oblique covectors on saved evolved states and 112 synthetic approach-to-scri cases pass physical/gauge/constraint sector checks on the constrained20 manifold. [Raw22 audit](../validation/goal1-minkowski-20261018/algebraic-characteristics/README.md) identifies the two algebraic modes' advective Jordan quotient and three additional frozen B slots; projected frozen-principal SSPRK3 checks pass on30 evolved cases. Full nonlinear projection/ghost/source stability remains open. Symmetrizer condition reaches `4.57e10`; no uniform endpoint chart. No gauge change required by these diagnostics. |
| Raw stationary RHS and endpoint jets | Sampled unsubtracted continuum stationary RHS max `1.18e-13`. Actual interior and gauge assemblers reject Omega=0. The [RWM jet diagnostic](../validation/goal1-minkowski-20261018/scri-jet-interpretation.md) checks removable analytic limits and time compatibility, not a numerical endpoint closure. Quadratic-null falloff is not automatically invariant; this alone is not a general smooth-gauge obstruction. |
| Short native stationary preservation | [Local native pilot](../validation/goal1-minkowski-20261018/native-pilot-review.json), B8/A8/R48, finite cut `.98`, `t=.002`: max field drift `8.84e-15`, max reported constraint `1.93e-12`, no RHS subtraction, reference payloads invariant. This passes only the short pilot budgets; ten-crossing stationary/growth screen remains open. |
| Small coupled perturbations and timestep control | Lapse/shift amplitude `.001`, same native control, ordinary steps `5e-5`, `2.5e-5`, `1.25e-5` reach common `.002`. Continued matching checkpoints. All-cell field and first-derivative time differences include interfaces/corners; derivative differences do not decrease monotonically at these small levels. No resolved spatial/oracle error exists for the 10% contamination gate. |
| Tensor propagation coverage | [TT Hertz construction](../validation/goal1-minkowski-20261018/tensor-oracle-construction.md) supplies a source-consistent linear physical radiative family, including translated centers for oblique crossings. Both polarizations pass independent 75-digit Cartesian TT/wave/projector identities (`1.53e-74`). Transformed native jets, native point action and actual propagation remain untested. |
| Radial dissipation repair | [SBP v1](../validation/goal1-minkowski-20261018/sbp-v1/README.md) component KO derivative-energy failure remains. [Canonical KO v2](../validation/goal1-minkowski-20261018/canonical-ko-v2/README.md) repairs the radial dissipative stage with zero q penalty and an actual native inverse lift; a separately retained untapered version fails the `1.5` consistency gate. Quadratically tapered v2 passes nodes24/48/96 principal consistency (minimum fine order `3.0921`),138 native point-action checks, and60 live RK-stage radial energy identities (worst defect `1.18e-14`). Short stationary entry and bitwise restart pass; no full-method energy or scientific accuracy adoption. |
| Full-gradient dissipation repair | [Full canonical v3](../validation/goal1-minkowski-20261018/full-canonical-v3/README.md) retains a new v2 full3D energy failure: compatible tangential variations yield positive work, maximum normalized rate `3.85e-5`. Actual mixed native wave identities pass (`3.46e-13`). V3 full Pi and all-direction positive line kernels pass1,800 frozen energy witnesses,138 native point checks,156 new line cases (minimum fine penalty order `4.4975`) and72 actual bulk/shell stage identities (worst defect `2.55e-14`). Control active fields equal v2 bitwise; stationary4 and continued pulse8 entry pass. No full source/advection/interface or scientific acceptance claim. |
| Incoming hierarchy, interfaces and exact scri | Open. [Finite-radius rank/reflection pilot](../validation/goal1-minkowski-20261018/incoming-hierarchy/README.md): four Theta/Z plus six incoming lapse/shift/TT conditions have rank10 on142 symbols. Spatial-metric alternative rank9; momentum-only lift rank7. Reflective conditions have positive canonical flux directions for some covectors. The later v4 full/conormal SAT passes local native certificates, not a full discrete energy estimate. Actual donor probe finds zero incoming bulk-to-shell dependence in nodal-SBP candidates; native interface repair and exact-scri equations/closures remain missing. No free incoming physical scri data are proposed. |
| Frozen outer-port penalty | [Isolated second-order port check](../validation/goal1-minkowski-20261018/outer-port-v1/README.md) supplies a zero-q-penalty reflective SAT in the KO v2 energy on five reused evolved normals and16 unique synthetic radial states. Retains the original `.5513` omitted-volume identity failure; corrected24/48/96 mapped-volume commutator accounting passes. This freezes normal wave coefficients and excludes actual varying J/tangential/source/interface terms; no native boundary or endpoint admission. |
| Full native-jet boundary compatibility and tangential port | [786 point fixtures](../validation/goal1-minkowski-20261018/full-boundary-point-v1/README.md) include all384 outer native nodes in stationary/pulse cases, with actual saved ghosts, SBP/Hessian and reference jets. Full Θ/Z value hierarchy and both TT/free conditions pass; omitted reference/tangential/trace terms are nonzero. Point correction is not a time update; actual Θ/Z time residuals remain. [Full frozen40-variable derivative port](../validation/goal1-minkowski-20261018/tangential-port-v1/README.md), with full Pi, conormal derivatives and actual tangential constraint rows, is dissipative on five evolved normals. Lower sources, discrete reduction/trace commutators and native/interface/endpoint closure remain open. |

The [pilot protocols and results](../validation/goal1-minkowski-20261018/README.md)
preserve an initial pre-evolution radial-ghost rejection and a timestep-ceiling
misinterpretation. `fixed_dt` is a ceiling; actual steps must be audited. The
compatible `.0015` checkpoint was continued rather than repeated. Measured
coarse CPU cost is about `.78` seconds per step; this is not a fine-grid estimate.

[Reference travel times](../validation/goal1-minkowski-20261018/reference-crossing-times.json)
give origin-to-scri outgoing travel `.7457643839` for the control reference.
These are coefficient integrals, not measured selected-packet crossings. The
finite cut `.98` takes 30 time units to send an incoming CMC characteristic to
`.95`; the short pilots cannot assess that distant boundary effect.

[Acceptance protocol v0](../validation/goal1-minkowski-20261018/acceptance-protocol-v0.json)
records normalizations, proposed budgets, provisional grids and launch gates.
It is explicitly **not frozen for acceptance**: selected numerical closures,
exact-scri equations, packet/oracle coverage, reflection budget and practical
matrix durations still need admission. Freeze v1 before any acceptance matrix.
Do not spend larger-grid trajectories while those formulation/operator gates
remain open. Goal 1 remains unfinished. [Goal 2](02-angular-minkowski.md) is
updated with these limitations and has not been activated.

### Revision history

- 2026-10-18: Activated Goal 1 with the confirmed local-only scope. Completed
  evidence/source inventory, independent geometry, oblique characteristics,
  RWM jet and physical tensor-oracle diagnostics, and bounded coupled CPU
  timestep/cost pilots. Retained all failures and open native endpoint/interface
  gates; no production change or gauge replacement.
- 2026-10-18: Native radial 6/3 SBP/compatible-D2 and mapped norm-compatible
  KO implemented in a separate source/build. Actual-header smooth derivative
  tests and native bitwise control/reference entry checks pass. The declared
  tensor derivative-energy test fails for all-field KO (stretched 24/48-node
  logarithmic rates `89.43/418.79`); witnesses and setup failures retained.
  No adoption, acceptance matrix or remote work.
- 2026-10-18: Implemented isolated canonical radial dissipation v2, preserving
  both component-energy and untapered-consistency failures. Tapered consistency,
  actual native point action, live radial-stage energy and restart entry checks
  pass; no complete-method adoption. Classified raw algebraic characteristic
  modes and finite incoming hierarchy, retaining rank/flux obstructions. Updated
  Goal 2 as not ready; exact-scri/interface and scientific acceptance remain open.
- 2026-10-18: Full native-jet point hierarchy passes but exposes time-compatibility
  limits. A new compatible full3D energy witness rejects normal-only v2 outside
  its radial scope. Implemented full-gradient v3 dissipation in all bulk/shell
  directions; actual stage/control/entry checks pass. Full frozen tangential
  boundary port passes with conormal derivatives. No native SAT, gauge replacement,
  acceptance matrix or successor activation; earlier failures retained.

- 2026-10-18: Added isolated native finite-cut SAT v4 and actual donor dependency
  evidence. Missing bulk-to-shell transmission rejects the nodal-SBP research
  family as a complete method. Bounded archived true-face interface models restore
  energy/transmission; retained accuracy failures motivate independently checked
  compact C15 data. Central C15 convergence passes but derivative amplitude
  budgets remain open. No complete adoption, native acceptance or Goal 2 activation.

- 2026-10-18: Implemented isolated mapped true-face physical derivative kernel,
  with explicit geometric SBP volume defect and compatible physical Hessian
  compositions. Actual native22/16-slot entry restores bulk-to-shell dependency;
  disabled control is byte-identical and references invariant. Zero-time mapped
  refinement decreases first/second/mixed/curl errors. No coupled evolution,
  full stability, true-face outer SAT or endpoint adoption; large errors retained.

- 2026-10-18: Full frozen-native20 graph action exposes derivative-commutator
  work absent from scalar SBP. Retained commuting-shortcut failure, without an
  unsupported instability claim. Separate same-gauge50-field derivative reduction
  passes native principal equivalence, full characteristic symmetry and frozen
  energy bounds after a retained/corrected comparison-sign failure. Native live
  chart/source, reduction accuracy and exact endpoint gates remain open.

- 2026-10-18: Holonomic metric chart passes80-digit derivative/time oracle and
  stationary/reference jet maps, including exact-scri reference values. Saved
  pulse differential det-one consistency fails and remains recorded. Exact
  nonlinear native q-source/momentum inverse passes460 point states/24 source
  invariance controls; live P-source/derivative/reduction and endpoint admission
  remain open. No native reduced evolution, scientific acceptance or Goal 2 launch.

- 2026-10-18: Actual nonlinear reduced source and independent unforced native chain
  pass; private AD/reference/source derivatives admitted at finite Omega. Regular
  same-gauge stationary exterior family rejects bounded old canonical scri chart.
  Exact native22 endpoint point quotient passes analytic fixtures; full endpoint
  characteristics/time compatibility and stable numerical closure remain open.
  Source-inclusive velocity candidate is proposed; no gauge change, acceptance
  matrix, new native trajectory or Goal2 activation.

- 2026-10-18: Full factored velocity map/native inverse and80-digit derivatives
  pass; exact-scri stationary value/first-jet inverse admits prior divergent-Pi
  family. Complete nonlinear finite-Omega velocity source and independent native
  chain pass. Nonstationary endpoint dynamics predeclared, still untested. No gauge
  replacement, new trajectory, scientific matrix or Goal2 activation.

- 2026-10-18: Explicit user correction: abandon optional first-order spatial
  reduction; preserve it as rejected research. Native second-order Z4c with
  native A/P/Theta/Lambda and RWM gauge is the active method. Reuse native geometry,
  symbols, operators, boundary findings and compatible checkpoints; no added wave
  momenta/gradients, no equivalent reduction, no energy-proof prerequisite for
  quantitative native pilots. Goal2 remains inactive.

- 2026-10-18: Native-only nonstationary exact-scri22 RHS/13 RHS-gradient/pole-time
  checks pass; omitted metric-gradient time control fails. Direct native coupled
  initial action exposes large composed-Hessian/interface error despite stationary
  preservation. Repair native second/mixed discretization and finite boundaries
  next, then quantitative native pilots. No reduction fields, matrix or Goal2 launch.

- 2026-10-18: Same-point native ghost-atlas direct D2/mixed action reduces native22
  RHS RMS from .273677 to .000120094 and Hessian RMS from .283169 to .000139020.
  Select direct native stencils for bounded finite-boundary pilots; temporary
  commutator diagnostics add no state. Spatial action excludes KO/advection/SAT
  and is not stability or propagation acceptance. Goal1 active; Goal2 inactive.

- 2026-10-18: Direct native ghost/D2 path selected, true-face stateless native RHS
  boundary implemented in an isolated CPU candidate. Native786 row/lift and112
  live-symbol checks pass; stationary/small coupled pilots reach .02 with measured
  field/derivative/physical-constraint/timestep/outer-interface diagnostics. Degree2
  controls, setup failures and nonzero initial incoming tail retained. Degree5
  improves initial Hessians; short screens pass within scope. Causal boundaries,
  independent propagation targets, general exact-scri numerical/characteristic
  gates and frozen scientific matrix remain open. Goal2 remains inactive.

- 2026-10-18: Added global regular same-gauge radial clock oracle and actual native
  coupled exact-error pilots. Independent origin/transition/exact-scri native
  source/time21 and all-coordinate readbacks pass. Two-grid spatial errors
  decrease; coarse regional Hessian excess and approximatelyfirst-order actual
  timestep trend retained. Native NGHOST3 is FD4, correcting historical metadata
  without changing results. Original angular/constraint/tensor, causal boundary,
  general numerical exact-scri and frozen acceptance gates remain open. Goal2 inactive.

- 2026-10-18: Compatible fine native clock checkpoint reaches t=.3; all short
  field/first/Hessian/raw and normalized constraint budgets pass with nine-region
  locations. Measured interface waveform phase error <=1.77e-5 on23 saved times;
  all-node independent fine readback and reference-volume uncertainty pass. Five
  unique trajectories/33 successful segments; one pre-evolution control-limit
  setup failure retained. Production/reference/rejected research hashes unchanged.
  Full-packet tails, causal boundary, longer-time controls, angular/constraint/
  complete native tensor and general numerical exact-scri gates remain open.
  Acceptance unfrozen; Goal1 active and Goal2 inactive.

- 2026-10-18: Verified both native TT curvature/trace/connection jets and actual22
  source, including linear exact-scri quotient, with independent80digit Cartesian
  ADM checks and a C++ runtime oracle. Preserved two deep finite-Omega precision
  exceedances; regular native assembly is needed, with no gauge-failure inference.
  Continued the existing native clock trajectory through.4, ten compatible segments
  and complete regional quantitative screens. No native tensor evolution, causal
  boundary or general numerical exact-scri/scientific acceptance yet. Goal2 inactive.

- 2026-10-18: First actual native coupled TT pilots for both polarizations.
  Original two-grid derivative/constraint failures preserved; full native uniform
  action identifies profile underresolution. Continuum bandwidth measured before
  separately versioned B36/A16/R64 entry pilots; both pass short v3 screens.
  Short common-time temporal/model controls recorded and measured, with compatible
  checkpoint continuation. Full propagation, original angular/constraint, causal
  boundary and general numerical exact-scri/scientific gates remain open. No added
  state, gauge replacement or Goal2 activation.

- 2026-10-18: Continued both unchanged native TT checkpoints through.005 in four
  compatible segments; completed cross amplitude controls and later finite live
  symbols/positivity. General nonlinear leading pole manifold96+14 reused points
  verified; original Lambda grouping failure retained and all96 incomplete higher
  completions rejected. Both linear TT native22/first13/time endpoint fixtures now
  pass independently checked targets. No full nonlinear endpoint/characteristic/
  numerical closure admission, scientific matrix, gauge change or Goal2 activation.

- 2026-10-18: Full actual limiting bulk principal system182 cases includes raw22,
  native20 and all constraint/gauge/physical sectors;182 independent actual finite
  extractions agree. Raw algebraic Jordan sector remains explicit. Two-native-field
  geometric trace/Theta source candidate23 points passes finite chain/known-face
  identities, but two deep finite precision criteria fail and are retained. No
  numerical adoption, new evolved field, gauge replacement, matrix or Goal2 launch.

- 2026-10-18: Native total metric/A RHS tangent isolated candidate adds no fields,
  keeps RWM, and passes33 point/2784 reused face checks and short native stationary/
  clock accuracy controls. Timestep-order gate fails at RMS2.0069/1.7251/1.7831;
  temporal contamination remains<5.73e-6 of spatial error. All failures preserved.
  Five roots/eight checkpoint segments; no full matched trajectory replay, matrix,
  exact-scri numerical admission, remote work or Goal2 activation.

- 2026-10-18: Native197-row exact-scri necessary jet system: fixed-first and
  six-normal failures retained; geometric normal metric traces admit24 complete
  general/clock points.24 compensated tensor-shear points pass below3.95e-13,
  preserving amplitudes; pure-shear failures retained.24 sampled tau/Z row
  certificates and72 higher-jet invariance checks pass. No evolved fields,
  numerical/gauge change, new trajectory, invariant endpoint/stability admission,
  scientific matrix or Goal2 activation.

- 2026-10-18: Analytic native scalar/gauge scri identities correct narrative
  constraint rank4 to2 (raw data unchanged), identify six lower conditions and
  conditional branch gaps. Old197 next-time tangency48/72 fails;202-row layer48
  fixtures passes<3.95e-13. Native singular-branch control and19 deep finite RWM
  arithmetic failures retained. Live leading metric flow keeps normal extensionB.
  No evolved fields, PDE/gauge/numerical change, general invariant endpoint/
  characteristic/stability admission, scientific matrix or Goal2 activation.

- 2026-10-18: Native full-source normal/constraint indicial analysis and mixed-order
  characteristics verified. Failed direct/first-shifted lifts retained; nonzero-time
  radial formal40/43 branch passes six orders and finite-source checks. Stationary
  fourth-order control fails. Regular data/error propagation class and exact-scri
  numerical closure remain open; no evolved reduction/gauge/numerical change, new
  trajectory, acceptance matrix or Goal2 activation.

- 2026-10-18: Full native angular and intentional-constraint finite-cut entry plus
  dt/amplitude controls:10roots11segments,231 live finite symbols, quantitative
  short response/initial-data/reference/positivity screens pass. Spatial sampling
  and angular global-M nonreduction failures retained; noSAT diagnostic identifies
  early finite-boundary mismatch, not a replacement closure. Exact-scri/scientific
  gates remain open, original Gaussian target retained and Goal2 inactive.

- 2026-10-18: Actual native finite boundary history repairs short original-angular
  constraint refinement without changing pulse, RWM or bulk/mapped equations.
  Native time/source/basis and coupled dt/history/stationary/restart checks pass;
 119finite symbols,8roots9segments. Old failures and remaining sampling/time
  limitations retained. Local Taylor data are not a long-time/large-pulse/scri
  closure. General exact-scri/scientific gates open; Goal2 inactive.

- 2026-10-19: Added restricted native spherical exact-scri evolution and actual
  endpoint-state bulk symbols. Smooth-clock budgets/refinement pass through.02;
  regional timestep arithmetic failure repaired within native offsets. Weak-branch
  closure criterion fails, so general/mapped3D exact scri and all original scientific
  gates remain open. No extra evolved fields, gauge change or Goal2 activation.

- 2026-10-19: Native weighted chain and coupled linearized endpoint operators
  added; moving hybrid indicial defect identified and globally fitted D1/D2
  improves weak-profile fields. Full nonlinear/mapped3D regularity/stability,
  derivative/reference/causal criteria and scientific matrix remain open.
  Failed hybrid, precision and pre-arrival criteria retained; Goal2 inactive.

- 2026-10-19: Actual native finite constraint identity/characteristics and correct
  finite-domain gauge orientation verified;18 unique native8 exact-gauge pilots
  add raw/normalized field/derivative/constraint and allstage positivity evidence.
  Source-complete boundary and necessary endpoint rate equations retain coupled
  growth; regional scri Hessian/temporal budgets fail. Native second-order/RWM
  remains active, no extra fields/reduction, scientific matrix or Goal2 activation.

- 2026-10-19: User reprioritizes full nonlinear3D literature reproduction and
  explicitly authorizes a separately versioned published-gauge candidate, retaining
  RWM control. Actual C48 vacuum rows, native characteristic projectors and
  full bulk/shell entry pilots added; five roots/seven segments, two restarts.
  Retained spatial/symmetry/sampling/time/duration failures and unresolved KO sign.
  Staggered entry is not exact-scri or reproduction success. Goal1 remains active,
  original gates preserved, Goal2 inactive and matrices not started.

- 2026-10-19: Literature B48 initial operator entry passes; B36 checkpoint reaches
  common .004 with improved native constraints/symmetry. Genuine half/quarter
  controls complete but RMS order criterion fails; sampling failure retained.
  Legacy62 matrix attribution and literal-Gaussian endpoint compatibility audit
  added. Eight roots/fourteen segments retained; Plan v20, Goal1 active, Goal2
  inactive, no matrix or remote work.

- 2026-10-19: Verified native literature OpenMP backend/identity transfers bitwise;
  four full nonlinear3D grids reach .02 via compatible checkpoints. Actual17
  TT/gauge/constraint sectors and38-point independent linear-source reference
  verified. Interface derivative, sampling, time-order, crossing/late and exact
  endpoint gates remain open. Plan v21, Goal1 active, Goal2 inactive, no matrix.

- 2026-10-19: Initial actual fullnative3D nonlinear source/constraint rates verified
  on allfour saved grids. Actual mapped third-derivative/product identities explain
  initial Mforcing; maxresidual7.51e-11, no timeupdate or equationchange. Failed
  diagnostic attempts retained. Plan v22: isolated operator comparisons then native
  coupled/stage/time/reference and allcrossing/exactscri gates. Goal2 inactive.

- 2026-10-20: Reconciled/sealed v5 operator and timestep evidence; stability-first
  Plan v26 resumes existing .004 checkpoints locally. All failed criteria and
  actual-scri/characteristic gates retained; Goal 2 inactive.
