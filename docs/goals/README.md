# Hyperboloidal Z4c benchmark goals

2026-10-23 puncture-pilot authorization: the user explicitly authorized an
exploratory single Schwarzschild puncture pilot before the original Goal 2
prerequisite, with stability first and dt=1e-4 fixed. See the
[pilot record](../validation/puncture-pilot-20261023/README.md) and
[complete objective](../validation/puncture-pilot-20261023/objective.md).
Goal 2 remains inactive; Goal 1 is incomplete; this does not grant full Goal 3
acceptance. All existing failures and unresolved exact-scri requirements remain.
The active execution task is this isolated pilot; earlier priority text below
is retained as history and evidence.

Current execution update, 2026-10-23: the user authorized one-node Aurora MPI
scaling across GPU tiles, superseding the earlier single-GPU limit. The
corrected 1/2/3/6/12-tile benchmark passed; twelve tiles were fastest for its
fourteen-block layout. The original seven-block pair passed unchanged-mesh MPI
restart checks, selected seven tiles, and both trajectories reached t=1 with
finite, positive states. Interface diagnostics remain bounded and oscillatory
over this window; complete saturation and convergence are not established.
See [Goal 1's current record](01-minkowski-foundations.md), the
[continuation evidence](../validation/goal1-minkowski-20261018/native-mpi-continuation-v34/result-review.json), and the
[scaling evidence](../validation/goal1-minkowski-20261018/native-mpi-tiles-v33/result-review.json).
Keep dt=1e-4, one debug node at a time under CompactBinaryMerger, all evidence
preserved, and Goal 2 inactive. Stability remains the priority; no convergence,
accuracy, or long-time stability success is inferred from the MPI tests.

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

Planning revision: 2026-10-22. These are sequential, reusable goal-mode prompts.
Saving them does not launch an evolution, submit a job, or activate all goals.
Goal 1 is active in goal mode. Goals 2–4 are not active.

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

## Sequence

| Goal | Earlier milestones | Completion target | Status |
| --- | --- | --- | --- |
| [1: Minkowski foundations and exact scri](01-minkowski-foundations.md) | 1–3 | Verified geometry/operators, stationary preservation, and small coupled perturbations through exact scri | Active: v29 pair reached .2 at dt=1e-4 with positivity; peaks decline after .125 but broader RMS grows; convergence/accuracy deferred; old failures and actual-scri requirements retained |
| [2: Nonlinear angular Minkowski pulse](02-angular-minkowski.md) | 4 | Stable, convergent substantial angular gauge disturbance | Not ready; updated with Goal 1 pilot evidence and specific missing gates; not activated |
| [3: Schwarzschild and trumpet evolution](03-schwarzschild.md) | 6 | Stationary preservation and resolved wormhole-to-trumpet evolution | Provisional; refined by Goal 2 |
| [4: Kerr, a/M = 0.5](04-kerr-spin05.md) | 7 | Valid spinning geometry, stationary/relaxation tests, and convergent 3D evolution | Provisional; refined by Goal 3 |

Earlier milestone 5, an independent weak gravitational-radiation benchmark, is
deferred. A Cauchy linear-wave generator exists, and constraint-satisfying
Minkowski geometry with perturbed lapse/shift is available. Neither establishes
an admitted localized radiative initial-data construction on the complete
hyperboloidal domain. A centred linear TT native jet/source oracle is now verified. Tensor
principal/operator propagation tests stay in Goal 1, with their scope explicit. Gravitational ringdown is conditional on suitable data
and extraction admission; scalar QNMs cannot validate gravitational evolution.

## How to use these files in goal mode

Activate only one benchmark goal at a time, using its Objective and Execution
instructions. Do not activate future goals or
create extra chats/automations automatically. Goal completion requires its
evidence and successor-plan revision, not simply reaching the requested time.
If scientific gates remain failed, retain the unfinished status and report the
smallest unresolved question. A diagnostic negative result can close a subtask
without passing the benchmark.

Each goal must update its own results section, this index, and the next goal.
Keep a dated revision history and the original failed criteria. Successor edits
must name what changed, which evidence supports it, and what remains uncertain.
Do not silently weaken acceptance criteria or launch the successor.

## Shared rules

- Current priority: reproduce the published vacuum Minkowski/gauge setup with the
  full nonlinear3D native Cartesian-bulk/cubed-sphere solver. A separately versioned
  published gauge is authorized; RWM remains the unchanged control. Term-by-term
  comparison and distinct staggered/exact-scri gates precede reproduction acceptance.
  Reproduction is an intermediate milestone; Goal2 remains inactive.

- Active Goal1 method: native Z4c, first order in time and second order in space.
  Retain native curvature, trace, connection and constraint variables. No added
  wave momenta or independently evolved configuration gradients. Preserve rejected
  spatial-reduction research; characteristic reductions are analysis only. Repair
  and test the actual mapped native discretization directly; a symmetric first-order
  energy proof is not a prerequisite for bounded quantitative native pilots.

- Preserve the frozen production Z4c baseline and all existing evidence. Develop
  candidates in isolated research copies/builds; record the diff and source hashes.
- Keep the separate Minkowski reference immutable, including in black-hole runs.
  Fix coordinate maps and Omega within each run. The physical metric and gauge
  evolve in these Z4c goals; the scalar experiment's frozen physical metric rule
  does not apply to full Einstein evolution.
- Evolve the actual coupled Z4c system in native acceptance runs. Isolated scalar,
  principal-symbol, or subsidiary models are diagnostics, not substitutes.
- Reuse completed, source-compatible evidence. Index runs by equations, source,
  binary, compiler, initial data, map, grid, timestep, boundary, and target time.
  Continue matching checkpoints instead of duplicating trajectories.
- Run all substantial computation on Aurora under the resource policy above.
  Keep local activity to lightweight coordination and document edits. Current
  priority is the bounded interface stability experiment above; defer acceptance matrices
  unless needed to diagnose instability or validate implementation changes.
- Use CPU evidence before short GPU comparisons. GPU agreement establishes
  implementation agreement, not independent continuum correctness.
- At exact scri, supply no freely prescribed incoming physical radiation. Derive
  the regular endpoint equations, check the full live characteristic spectrum,
  and use a stable endpoint discretization. Gauge branches must be outgoing or
  tangent for the intended pure-outflow formulation. Finite-radius cuts generally
  have incoming branches and require separately justified boundary treatment.
- No Omega floors, arbitrary constraint resets, or black-hole RHS subtraction to
  manufacture success. Label any manufactured forcing or reference-roundoff
  subtraction explicitly and test the unsubtracted residual independently.

## Shared evidence and acceptance protocol

Before acceptance runs, save a protocol fixing cases, resolutions, times,
normalizations, norms, thresholds, characteristic checks, and numerical method.
Pilot data may determine practical grids and duration; freeze the protocol before
the acceptance matrix. Later amendments get a new version and preserve failures.

Use at least three resolutions with independently documented bulk, angular, and
radial spacings. Compute errors at common physical coordinates/times against an
independent solution where possible; distinguish self-differences from actual
errors and constraint norms. Use the correct volume/surface weights and report
whole-domain plus core/interface/transition/outer/scri regions. No undocumented
removal of troublesome cells. For unequal refinement ratios, solve for the order
using the actual spacings rather than applying a dyadic formula.

Measure fields, first derivatives, H/M/Theta/Z and native algebraic/differential
constraint consistency. Record raw physical constraints as well as
dimensionless versions; Omega-weighted smallness alone is insufficient. Exact
scri constraints use their derived finite limits, not physical ADM divisions by
zero. Monitor lapse, metric positivity, scri nullness, gauge compatibility, and
the full actual characteristic speeds, including interfaces and inner cuts.

Use half/quarter timestep controls, with source stiffness included. Bound
reference/interpolation uncertainty below 10% of the finest reported error and
temporal contamination below 10% of spatial error. These are proposed defaults
until the individual protocol is locked. State expected interior and boundary
orders for the selected operator. A non-asymptotic coarse pair triggers a bounded
refinement/diagnosis, not an immediate claim of a formulation defect. Passing
requires resolved error reduction and quantitative amplitude budgets, not order
alone. Do not claim asymptotic convergence from an unresolved plateau.

Save machine-readable summaries, spatial error maps, radial profiles, interface
slices, scri histories, convergence plots, checkpoints, and source/binary/data
hashes. Record negative controls and failures alongside successes.

## Existing work to reuse

- [Conformal equations, variables, gauge and characteristics](../hyperboloidal-layer.md)
- [Historical spherical trumpet results](../hyperboloidal.md)
- [Boundary transport counterexample](../hyperboloidal-gauge-boundary-audit.md)
- [Scri compatibility hierarchy](../hyperboloidal-scri-linear-hierarchy-audit.md)
- [Mapped-domain and AMR implementation](../cubed-sphere-extension.md)
- [Latest coupled constraint/boundary journal](../cubed-sphere-goal-progress-20261013.md)
- [Scalar fixes, admissions and limitations](../test-scalar-frozen.md)

## Revision history

- 2026-10-18: Initial four-goal draft. User selected exact scri as a required
  Goal 1 gate and local CPU pilots followed by the acceptance matrix. Milestone 5
  deferred; future goals must be refined by their predecessors.
- 2026-10-18: User confirmed boundary/operator repairs first, retaining the
  existing reference-wave-map gauge as the control. Gauge changes require
  characteristic or scri-compatibility evidence and a separately versioned case.
- 2026-10-18: Goal 1 activated locally. [Inventory and pilots](../validation/goal1-minkowski-20261018/README.md)
  reuse the frozen native evidence, add high-precision geometry/characteristics
  and tensor-oracle diagnostics, and complete a short coupled CPU timestep pilot.
  Exact-scri equations and full native compatible operators/interfaces remain
  open. Existing failed growth/closure criteria remain failed. Goal 2 updated
  but not activated; no acceptance matrix or remote work started.
- 2026-10-18: Native radial SBP v1 tested against the frozen executable; control
  fields match bitwise and reference/stationary entry passes. All-field radial
  KO fails its declared derivative-energy criterion.
  [Failure and bounded next repair](../validation/goal1-minkowski-20261018/sbp-v1/README.md)
  retained; v1 not adopted. Goal 2 remains unready and inactive.

- 2026-10-18: Native finite-cut SAT v4 passes local entry/port checks. Actual
  donor dependency rejects nodal-SBP v1–v4 for missing bulk-to-shell transmission.
  Native refinement stopped. Archived true-face operators restore frozen energy
  and transmission; original failed criteria retained. Central resolved-C15 model
  passes convergence but derivative amplitude budgets remain open. No native
  interface adoption, exact-scri closure, acceptance matrix or Goal 2 activation.

- 2026-10-18: New isolated true-face mapped physical operator restores native
  cube-to-shell dependency and passes field-slot/control/reference entry checks.
  Three zero-time operator grids converge; explicit geometric/curl defects and
  large Hessian errors remain. No complete coupled method, true-face outer
  penalty, exact-scri closure, acceptance run or Goal 2 activation.

- 2026-10-18: Full native20 mapped energy identities pass with explicit
  commutators; commuting shortcut rejected. Separately tested50-field derivative
  reduction supplies a frozen-system bound with RWM gauge unchanged. Corrected
  comparison-sign failure retained; no live nonlinear/native source adoption,
  exact-scri closure, scientific matrix or Goal 2 activation.

- 2026-10-18: Independently verified nonlinear holonomic chart and exact native
  configuration/momentum source map. Retained saved pulse differential
  determinant-consistency failure. Full native P-source/evolution, derivative
  reduction accuracy, boundaries and exact-scri equations remain open. Goal 2 inactive.

- 2026-10-18: Full nonlinear reduced finite-Omega point source/native chain verified.
  Actual22 exact-scri point limits admit a regular stationary same-gauge exterior
  family that rejects bounded old canonical Pi. Next derive a factored regular
  velocity chart and full endpoint characteristics/time compatibility, then stable
  true-face transport and local pilots. No gauge replacement, scientific matrix
  or Goal2 activation; failures retained.

- 2026-10-18: Same-gauge full velocity variable map and nonlinear source verified
  against actual native equations/independent80-digit jets; stationary scri inverse
  passes without bounded-Pi restriction. Next execute nonstationary endpoint native
  dynamics and time compatibility, then full actual characteristic and stable
  true-face transport gates. No new trajectory/matrix/Goal2 activation.

- 2026-10-18: User explicitly rejects optional first-order spatial reduction and
  equivalent added-field formulations. Active path is native second-order Z4c,
  preserving native A/P/Theta/Lambda and the control gauge. Inventory separates
  direct native evidence from rejected reduction proofs. Auxiliary momentum
  divergence imposes no native scri restriction. Old energy-before-pilot planning
  gate is superseded; quantitative native coupled tests remain mandatory. Goal2 inactive.

- 2026-10-18: Direct native22 nonstationary endpoint/time compatibility passes
  analytic fixtures; compiled native dependencies exclude rejected reduction.
  Coarse full native RHS action localizes large composed-Hessian error at
  cube-shell interface. Next repair native direct second/mixed and true-face
  boundary operators, then bounded coupled pilots; no first-order energy gate,
  new auxiliary state, acceptance matrix or Goal2 activation.

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

- 2026-10-18: Source-inclusive native normal indicial system and correctly graded
  constraint characteristics verified. Six-order local radial branch requires
  regular-data/error-class analysis; failed lifts and stationary recursion remain.
  No arbitrary falloff/gauge change/numerical adoption or new trajectory. Original
  coupled scientific and exact-scri closure/matrix gates remain open; Goal2 inactive.

- 2026-10-18: Actual native angular and intentional-constraint short pilots,231
  live symbols and strict checkpoint continuation added. Retained interpolation
  and angular global-M failures; bounded noSAT control localizes outer mismatch
  without boundary adoption. Original Gaussian target, exact-scri and full
  scientific/causal/long-time gates retained. Goal1 active; Goal2 inactive.

- 2026-10-18: Finite boundary data from native F/DF restore short angular physical-M
  refinement; pulse and RWM unchanged. Disabled control bitwise,119live symbols,
  stationary and strict restart pass. Local time-Taylor scope only, earlier failures
  retained; full exact-scri/long-time/sampling/scientific gates open. Goal2 inactive.

- 2026-10-19: Restricted native spherical exact-scri pilot through.02 added with
  measured field/derivative/constraint errors, native-offset dt controls, five
  checkpoint continuations and70 full limiting-bulk symbols at numerical endpoints.
  Smooth DAE fails native weak-branch action; no general closure or scientific
  matrix admission. Native second spatial order/RWM/baselines preserved. Goal2 inactive.

- 2026-10-19: Weighted nativeA/Lambda and full-source linearized endpoint work
  identify and repair a moving-stencil indicial defect. No added evolved fields.
  Weak-field improvement does not clear coupled positive mode, causal/conditioning,
  general nonlinear/regular-characteristic or original scientific gates. All failures
  and immutable baselines retained. Goal2 remains inactive.

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
