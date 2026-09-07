# Clean intrinsic reduction implementation status

2026-09-07. Work in progress. Only one-step legacy controls have evolved; no new-formulation qualification claim.

The remote source was fetched at `62945657b4f2828a481abb5e7708e0e3e06dbd8d`,
exactly the independently verified SHA. The requested new branch
`codex/pcgh-clean-reduction-transfer-20260907` was created there, then fast-forwarded
to the originating checkout's committed descendant `249929fd` after inspecting
the intervening commit inventory and production diff. This retains the direct
lapse-gradient helper, tracker and CUDA fixes, and negative qualification evidence.
The originating checkout has dirty analysis and extensive untracked raw evidence;
none of those files was modified or merged. The parent main checkout is untouched.

Separate detached legacy control: `../athenak-pcgh-legacy-control-20260907`,
commit `b81b44d658f3b81584e94ce79b92656c112ff908`. It is an evidence-only checkout.
The mathematical reference is detached at
`7ef9c61c0c2bd12a46b28f334a53a1aabcd1842e` in
`../pcgh-candidate-reference-20260907`; its files are not changed by tests.
The candidate suite executes in `../pcgh-clean-reduction-tests-20260907/candidate-suite`
using a separate virtual environment. All eight original checks returned zero.
This is reproduction of candidate algebra/point jets, not independent compiled
AthenaK verification, a puncture theorem, or an evolution result.

The direct-lapse report is recovered at
`qualification-runs-20260905/direct-lapse-gradient/REPORT.md`. Preserve its failure
at 5.167923M versus 5.187818M, unavailable first-invalid intermediate state,
and scalar-max/stale-ghost limitations. The R16 20M ladder has anti-aligned field
differences and does not pass the binary gate. Existing transfer brackets do not
establish a causal projection-feedback explanation of late growth.

Code inspection confirms GH projection every stage, reduction projection only
on the final stage, followed by restriction/exchange/BC/prolongation. Both ordinary
and postprojection transfers require explicit coverage in the new operator work.
The inherited direct lapse helper returns **twice** the gradient: intrinsic l
must account for that convention and cannot reinterpret legacy L.

Della access works using the preserved multiplexed ControlPath. Initial occupancy:
A100-PCIE-40GB, 40960 MiB total, 3496 MiB occupied, 0% sampled utilization; no
user Slurm jobs. This does not imply the occupied memory is available. Scheduler
gputest reports a 15-day limit and heterogeneous GPU node types. Resource use is recorded separately for the completed small CUDA controls. Inspect
occupancy and exact requested node resources again before each new allocation. User authorization permits direct vis1 testing or up to
eight 80GB A100s through gputest; request only resources needed by each staged test.

## Legacy equivalence checkpoint

The inherited direct-lapse patch also changed the optional global L projection
in legacy mode. Therefore merely selecting `reduction_system=legacy` does not
reproduce the collision source. Added explicit
`lapse_projection_target=collision_factorized` for legacy controls; the default
remains `direct_product`. Unknown targets and collision targets in advective
mode fail at input validation. No bulk equation or projection timing changed.

Two independently built full CPU executables use identical test adapters against
the collision source and current source. Seven smooth non-diagonal off-constraint
seeds, all active cells and all 55 fields, FD2/4/6, and 2D/3D anisotropic fixtures
have exactly matching saved RHS, algebraic/GH projection and auxiliary projection
values with the collision target. These are sampled nonlinear equivalence checks,
not a proof for arbitrary jets. The direct-product negative control differs by
0.0010--0.0049 in the projection fixtures, confirming the target distinction.
Both invalid-input tests pass. CPU raw outputs/builds are preserved outside Git
under `../pcgh-clean-reduction-tests-20260907`; compact results and complete source
manifests are in the dated evidence directory.

Independent CUDA builds and controls have completed in
`/scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-legacy`.
Six zero-step fixtures match to normalized error at most 5.53e-17; six actual
RK3 one-step fixtures match to at most 3.62e-17 (frozen tolerance 2e-12).
The CPU one-step counterparts match exactly. The old-source CUDA one-step
executable includes only the view-capture repair required to avoid host-this
access; its patch, complete source manifest, executable hashes, inputs and raw
output hashes are retained. The exact unpatched CUDA executable is preserved.
The CUDA build and test controller exit files both contain zero. Raw CSVs remain
outside Git; compact records are in `cuda-legacy-001` in the evidence directory.
These comparisons do not exercise MPI, AMR, restart, or a physical solution.

An opt-in `state_budget` writer now records before/after/signed increments for
all fine-array state components, including ghost cells, at existing operation
brackets and before post-RK validation. CPU 2D/3D independent-index fixtures,
no-op records and truncated-record rejection pass. The writer explicitly makes
no ghost-validity assertion. It does not yet record the coarse buffer, separate
GH from algebraic correction, or compute independent constraint increments;
these limitations prevent calling it a complete causal budget.

A proposed shifted derivative-ghost stencil was screened in a periodic TT toy
subsystem. Floating-point eigenvalues flagged tiny positive roots near neutral
modes, including the uniform control. Exact rational FD6 spatial matrices for
two blocks of 8 or 16 cells have distinct real nonpositive eigenvalues and a
constant kernel. This resolves those no-KO toy flags without changing a numerical
tolerance. It does not establish a uniform energy bound, KO/RK stability,
multidimensional/AMR stability, or a complete transfer repair. Both raw flags and
exact polynomials are preserved; no production halo operator had changed at that checkpoint.

Next: complete the archived transfer discriminator and implement full coherent
transfer with valid stencil support and signed operation diagnostics, then the
intrinsic map and complete kernel oracles. Gates 1--5 remain unpassed.

## Actual mesh transfer checkpoint

Implemented opt-in `coherent_transfer=residual_shifted` with a second residual
exchange on a separate communicator and shifted derivatives contained in valid
primary halos. Both ordinary and post-projection paths are connected. Default
transfer, legacy bulk equations and projection policies are unchanged. This
initial option accepts only periodic fixed topology with legacy fields.

FD2/4/6 constant-residual checks now pass on actual 2D/3D uniform and 7/15-leaf
static meshes, serial and two-rank MPI with bounds checking. Every primary and
active auxiliary stays bitwise unchanged within the correction; maximum ghost
residual error is 1.20e-13 versus the frozen 2e-12 bound. The one-step task-path
smoke check sees operations 11 at stages 0/1/2/3 and 12 only at stage 3. Default
legacy CPU zero-step and one-step comparisons still match the collision control.

The initial obsolete SMR input syntax, inherited 2D prolongation crashes, and
inherited FD4 parser rejection are preserved as failures. Corrected dimensional
prolongation and actual-nghost dispatch resolve those tested cases. The early
failing executables were overwritten during incremental development; their
recorded hashes, inputs, logs and inherited failing code remain available, but
no archived early binary is claimed. Final tested source manifests and build
configurations are retained. Historical 2D restriction still averages cells.

Next: CUDA validation, varying-residual/curl convergence and complete coupled
operator analysis, then boundary/regridding/restart completion. Constant-residual
and single-step checks do not pass Gate 1 or authorize physical promotion.

The isolated CUDA/MPI transfer build has started on Della at
`/scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-transfer-001`,
source snapshot `40e0bc1fc6e8f5dd0c474ed7d5d127060ea44937`. All 3965 snapshot
file hashes were independently verified before compilation. Controller PID
385334 was confirmed live at 6%; tool session 64564 observes it. Configuration
succeeded. Inspect that controller and `build.exit` before taking any restart
action. This build controller launches no GPU tests; new CUDA transfer checks
remain NOT_RUN until actual results are collected.

## Variable residual and curl convergence checkpoint

Added a smooth off-reduction operator fixture with analytic curls. Fixed physical
block boundaries and 7/15-leaf refinement maps are retained while cells per block
increase 8 -> 16 -> 32. All 33 residual and all 33 curl components meet the frozen
thresholds over three exchanges. 2D refined ghost residual rates are 1.997/1.999;
3D rates are 4.977/4.988. Active curl rates reflect one derivative of interpolation
error: approximately first order in 2D and fourth order in 3D. Uniform curls
approach sixth order; uniform residual ghosts are at roundoff, so their apparent
negative rates are not meaningful convergence estimates.

These measurements improve on the ordinary transfer in this controlled fixture,
with the actual primaries fixed. Ghost residual comparisons use the declared
shifted target, not an independent physical solution. Active curl errors use the
analytic seeded curl, retaining the legacy factorized-lapse product-rule error.
The 2D averaging restriction remains a substantive order limitation, not a claim
of FD6 interface accuracy. The aggregate first pass and the subsequent complete
component pass are both retained, with binary/source hashes and per-run durations.
A plot is `varying-transfer-002/convergence.png` in the dated evidence directory.

No evolution stability, puncture regularity, full Gate 1 or intrinsic-system
qualification is implied. Next substantive work is physical boundary completion
and coupled transfer/RK/KO behavior. The independently launched CUDA/MPI build
of snapshot 40e0bc1f remains separate from these newer CPU-only diagnostics.

The necessary TT model now includes reconstruction at KO auxiliary consumers.
This exposes K_Q*h in the discrete reduction equation, absent from the earlier
KO screen. The FD6 smooth-mode defect approaches fifth order. Raw near-neutral
spectral flags are preserved; the quotient by analytically invariant neutral
h/v modes has no growing-mode/RK flags in the tested matrices. This does not
bound nonnormal growth or qualify the full Einstein/interface evolution.

Della test controller PID 479693 (tool session 78343) is confirmed live and waits
on build PID 385334. It checks the build exit, source hashes and >=4 GiB free GPU
memory before each group. Only successful single-rank uniform/refined groups
permit two-rank groups. No CUDA transfer result exists yet at this checkpoint.

## Physical boundary operator checkpoint

The opt-in legacy residual transfer now admits periodic, outflow and reflecting
faces on fixed meshes. It completes physical boundary values after ordinary
prolongation (operation 13), then transfers residuals with their tensor parity,
completes residual physical corners after prolongation, and reconstructs auxiliary
ghosts. Operation 13 may change ghost primaries; operations 11/12 preserve all
primaries and all active entries. No bulk equation or projection policy changed.
Reflection reconstruction folds coordinates to the interior mirror and applies
the auxiliary tensor parity. Centered target arithmetic matches the existing
projection; shifted targets use differences to reduce cancellation.

The first physical-boundary matrix failed because prolongation left physical
corners stale. Both pre-fix and post-fix results remain archived. The corrected
operator passes all FD2/4/6, 2D/3D default outflow and compatible mixed reflection/
outflow fixtures on uniform and refined meshes. The latter seeds all 33 residual
components with independently defined parity-compatible multiaffine polynomials.
Its serial refined maximum residual error is 2.24e-13 and parity error 3.47e-18.
Two-rank MPI with bounds checking passes the same compatible fixtures and default
outflow refinement. Periodic uniform/refined serial regression also passes.

Two negative controls remain FAIL at their frozen 2e-12 tolerance. Nonzero
constant odd residuals have discontinuous reflected extensions; their refined
interpolation errors cannot be interpreted as smooth exact-preservation errors.
The separate compatible fixture addresses that ambiguity without deleting the
failure. Cubic outflow extrapolation in 3D amplifies encoded residual roundoff
above tolerance (up to 6.16e-11); it is not qualified. Default extrapolation order
2 and order 3 passed the tested constant residual checks. No tolerance was raised.

The original CUDA transfer build failed: NVCC requires the member enclosing an
extended lambda to be public. The access-only correction is rebuilding in the
same owned remote directory, with original failure logs and source manifest
preserved. New controllers are build PID 605504 and test PID 625235; the latter
waits for success and checks GPU occupancy. This snapshot is 40e0bc1f plus the
recorded access patch, not this newer boundary implementation. GPU transfer tests
remain NOT_RUN until actual results are collected. Full Gate 1 and all physical
gates remain unpassed; dynamic regrid/restart, complete causal budgets and coupled
amplification still require work. The intrinsic 50-field kernel is not yet present.

The actual one-step physical-boundary task check initially crashed in 2D even
with coherent transfer disabled. The inherited Sommerfeld helper differentiated
the inactive third direction with no halo. An explicit inactive-axis guard now
makes all four outflow/mixed 2D/3D tests pass, including a Kokkos bounds-enabled
build. The 3D final active CSVs are byte-identical before/after this guard.
Operations 11, 12, 13 occur at the frozen stages and satisfy exact signed
increment and active/primary invariance checks. These are arbitrary non-solution
one-step fixtures, not physical boundary convergence or stability evidence.

CUDA access-only build completed successfully. Single-rank periodic uniform and
refined FD2/4/6 2D/3D tests pass. The first attempted two-rank run is invalidated:
Anaconda MPICH mpiexec launched singleton OpenMPI copies. The harness now checks
the reported runtime rank count and exact rank-file set, accepts an explicit
launcher, and rejects this negative control. Local MPI logs confirm true two-rank
execution. The corrected remote rerun uses CMake's recorded OpenMPI launcher.

The corrected CUDA/OpenMPI two-rank uniform/refined reruns are now PASS with
verified runtime ranks and rank files. All 24 accepted single/two-rank cases
preserve fixed entries exactly; their maximum residual error is 1.96e-13.
This validates only the older periodic access-fix snapshot. Detailed durations
and hashes are in cuda-transfer-accessfix-001/verified-summary.json and per-run
results. The rejected launcher attempt remains FAIL in the test ledger.

## Intrinsic geometry checkpoint

The new intrinsic_geometry.hpp and intrinsic_state_map.hpp implement the
explicit 50-field chart, metric/inverse, curvature, independent Q=J*S, chart
Jacobian/Hessian and finite-radius 50/55 state conversions. Legacy evolution and
restart parsing are unchanged; intrinsic_clean_v1 is a named implementation
foundation, not an enabled evolution mode. Arbitrary off-algebraic legacy states
are rejected rather than projected. Old L=2*l is tested explicitly.

Exact symbolic checks prove both triangular inverse identities, the metric
inverse, determinant, curvature/gradient traces, and the five-component tangent
inverse. CPU and CUDA independently pass 100 nontrivial valid and six invalid
map cases. Worst normalized errors are 8.14e-16 and 6.36e-16, against the frozen
2e-12 tolerance. A separate byte-identical-input CPU/CUDA comparison has maximum
error 4.11e-16. CUDA default device and selected A100 are recorded by Kokkos.
The isolated CUDA map controller completed with exit 0.

This is geometry and conversion evidence, not the actual complete RHS oracle
or characteristic qualification. Next implementation work is the complete
configuration sources and their true chain-rule derivatives, followed by all
curvature/GH/auxiliary rows and independent point-jet comparison. No new bulk
evolution may be promoted past unfinished operator and oracle gates.

## Complete intrinsic point kernel checkpoint

All 50 continuum point-RHS rows are now implemented in intrinsic_rhs.hpp, using
complete configuration-source directional differentiation. CPU source/jet and
full matrix comparisons pass, as does the independent physical GH primary-row
oracle with nonzero C/Z. This kernel is not integrated into mesh evolution.

CUDA remains FAIL: the original combined diagnostic returns NaNs. Memory
checking reports zero errors; additional diagnostic stores change the result
to agreement with CPU. A smaller probe also shows a finite but discrepant K
row. The cause is unresolved, so no CUDA or evolution promotion follows. All
outputs, source/build manifests and probes are retained; all controllers are
terminal. The prioritized next action is to isolate this diagnostic-layout
dependence before grid integration. See analysis/pc_gh_clean_reduction/INTRINSIC_RHS.md.

## CUDA point-kernel repair checkpoint

The optimized CUDA failure is resolved in the new geometry construction.
BaseGeometry computes only the quantities consumed by the RHS, with Q evaluated
by an exactly equivalent direct tangent formula. It retains complete true Jet
derivatives. CPU and CUDA original/instrumented/probe/-O3/permuted-input checks
pass, including all 50 rows and eight full matrices. The independent physical
GH oracle, map regression and memory checker also pass. Kernel stack allocation
fell from 23904 to 9088 bytes. The prior construction's failures remain recorded;
no specific compiler bug is claimed proved.

The complete point kernel is now CPU/CUDA checked on the stated sample domain.
It remains separate from mesh evolution; characteristic conditioning, subsidiary
identities, full Fourier/transient and discrete operator/RK/KO checks, restart and
50-field transfer integration remain unfinished. Physical qualification has not
started. The next action is full-symbol conditioning/coincidence and subsidiary
checks using the compiled kernel before enabling an evolution mode.


## Full compiled symbol checkpoint

CPU and A100 checks pass all 52 sampled full-symbol cases, including the three
admitted speed coincidences. The CPU/CUDA matrix difference is 4.864e-16.
Complete eigenspaces and spectral projector algebra pass the frozen numerical
criteria. Projector norms remain bounded along each admitted crossing ladder;
they grow toward the excluded alpha=2,w=0.5 endpoint, where both nonzero
eigenspaces have dimension 9 rather than multiplicity 10, as predicted.
That negative-control PASS records detection of a defect, not an admitted state.
See analysis/pc_gh_clean_reduction/INTRINSIC_SYMBOL.md for scope and reproduction.

The local pseudoinverse warning was isolated and the equivalent SVD image
projector passes warnings-as-errors replay without altering raw matrices or
thresholds. No physical or full operator gate is promoted. Next is an independent
compiled subsidiary-law check with nonzero curvature, GH, reductions and curls,
then full Fourier/transient/discrete checks before mesh integration.


## Nonlinear subsidiary checkpoint

Exact arbitrary-function reduction, curl, Cartan and raw metric curl identities
pass. Independent spatial differentiation of the compiled CPU/A100 RHS on 12
nonlinear cubic fields also passes for all 30 reduction and 30 curl components.
It includes nonzero curvature, C/Z, true shift gradients and variable
lapse-scaled damping. Fourth-order residual decay reaches 1.424e-9 for reductions
and 7.913e-9 for curls; backend output disagreement is 5.471e-16. Negative controls
are discriminating. See analysis/pc_gh_clean_reduction/INTRINSIC_SUBSIDIARY.md.

This establishes sampled compiled continuum identities alongside the exact
formal law, not a mesh or evolution result. No physical gate is promoted.
Next is the full Minkowski Fourier operator with sources, neutral/Jordan and
transient behavior, then coupled discrete RK/KO checks and mesh integration.

## Complete Minkowski Fourier checkpoint

CPU/A100 full source and three principal matrices match the exact oracle.
The arbitrary-k polynomial, reduction closure and rotational covariance pass
exactly. Finite-time exponentials retain substantial amplification despite
nonpositive spectral real parts: the largest sampled full-state 2-norm is
72.5847 at t=20 in the undamped reduction/GH control, independently confirmed
by a 60-digit exponential. Homogeneous neutral Jordan chains are explicit.
At the sampled nonzero-frequency neutral roots, exact nullities show no chains.
See analysis/pc_gh_clean_reduction/INTRINSIC_FOURIER.md for normalization and scope.

The local SciPy expm warning is retained as an analysis failure; unchanged CPU
outputs pass Linux replay, and the A100 batch passes there. No warning or failed
physical state was hidden. No evolution gate is promoted. Next is coupled
RK3/KO operator analysis, then intrinsic mesh/transfer/restart integration.

## Intrinsic finite-difference consumer checkpoint

A reusable intrinsic 50-field Dx/KO RHS consumer is now implemented and checked
on CPU/A100 for FD2/4/6, 2D/3D and KO off/on. All 60 full-symbol cases pass;
12000 compiled points per backend give maximum symbol error 3.259e-11 and
backend disagreement 6.662e-16. Coupled uniform 50-field RK3 matrix powers have
minimum temporal order 2.9721, sampled radius <=1 and discrete reduction closure
to roundoff. See analysis/pc_gh_clean_reduction/INTRINSIC_DISCRETE.md for limits.

The routine is not enabled in mesh evolution. Uniform Fourier checks do not
cover halo arrays, nonconforming interfaces or variable backgrounds. Next is
explicit 50-field mesh/task/restart integration and coherent transfer testing.
No smooth-evolution, single-puncture or binary gate is promoted.

## Restart identity and continuation checkpoint

New legacy restarts carry an explicit layout name/version/count. Saved metadata
is captured before input/CLI overrides; incompatible or relabeled headers fail
before payload reading. Untagged files now require an explicit collision-layout
declaration after provenance validation. The intrinsic 50-field identity is
reserved but its mesh mode remains disabled.

A real serial continuation test found a preexisting 1.298e-7 mismatch caused by
an extra restart initialization projection (signed active Q change 1.339e-3).
Skipping that extra reset only on restart makes tagged and declared-untagged
continuations bitwise identical to uninterrupted evolution. Fresh runs are
unchanged. All 19 serial controls pass; failures and raw evidence are retained.
See analysis/pc_gh_clean_reduction/RESTART_LAYOUT.md for compatibility changes.

MPI/CUDA, multilevel and tracker restart coverage remains open. The next work
is actual 50-field mesh/storage/task integration with protected restart identity
and coherent intrinsic transfer. No physical gate is promoted.

## Layout consumer preparation checkpoint

Storage allocation, boundary receive counts, RK registers, restart payload sizes,
load-balance packing counts and evolved output selection now obtain their field
count from the allocated PcGh layout. Restart metadata also uses that immutable
object identity. Legacy equation indices remain unchanged; intrinsic mesh mode
is still rejected before allocation. Its task dispatch, diagnostic definitions,
initial data and transfer reconstruction remain to be implemented.

All 19 serial restart controls pass with bitwise continuation. Six FD2/4/6,
2D/3D one-step legacy comparisons against the collision-source executable are
bitwise equal. Evidence and exact build/source/binary/input hashes are in
qualification-runs-20260907/pcgh-clean-reduction/layout-consumers-001.
The changed MPI packing paths are not runtime-tested in this checkpoint.
No evolution gate is promoted. Next: uniform periodic intrinsic allocation and
task dispatch, with explicit rejection of unsupported numerical/diagnostic paths.

## Intrinsic mesh evolution checkpoint

The previous disabled-mode checkpoints are superseded: `intrinsic_clean` now
runs the real 50-field uniform-periodic RK3 task path, with protected layout I/O,
lapse-scaled default damping and an explicit constant-rate control. Both
projections are off. Unsupported legacy options, nonperiodic/refined grids,
legacy constraint/history output and unrelated initial data fail explicitly.

All 12 FD2/4/6, 2D/3D, two-rate-law one-step oracle cases pass to 1.111e-16,
including final ghost cells. Intrinsic restart continuation is bitwise equal;
13 unsupported/domain/output controls pass. Legacy one-step and restart
regressions remain bitwise equal. Health bounds are recorded with independent
initial eigenvalue/condition checks. See
analysis/pc_gh_clean_reduction/INTRINSIC_MESH.md for exact scope and limitations.

No physical gate is promoted. Integrated MPI/CUDA and multiple-block tests,
independent H/M/reduction/curl diagnostics, coherent intrinsic refinement and
physical boundaries still precede puncture qualification.

## Intrinsic multiple-block and snapshot diagnostic checkpoint

CPU serial and two-rank MPI now pass FD2/4/6 in 2D/3D with 4/8 blocks at fixed
global resolution, including all 50 ghost-cell values over three steps.
Maximum single-block disagreement is 1.681e-18; same-rank and two-to-one-rank
restart continuations are bitwise identical. Rank counts are verified.

An independent global-periodic snapshot analysis now measures physical H/M
from primaries alone, all 30 reductions/curls and raw Q-curl, with full-volume
component norms and signed maximum locations. Separate H and M convergence
on an analytic curved conformal/shear fixture recovers the expected FD orders.
This is not yet the production diagnostic task or a physical evolution gate.

The isolated integrated CUDA build and its queued test controller are still
active; no CUDA result is claimed. See
analysis/pc_gh_clean_reduction/INTRINSIC_DECOMPOSITION.md and the controller
status evidence before continuing the existing processes. Physical diagnostics
in the task graph, intrinsic refinement/physical boundaries and puncture/binary
qualification remain unfinished.

The asymmetric per-field oblique restart fixture strengthens permutation coverage
and found a repeated-restart metadata rejection. Accepting the reader-generated
false tracker marker repairs it while true tracker state remains rejected.
All six asymmetric two-rank comparisons, including ghosts and subsequent
rank-changing restarts, are now bitwise equal to serial. The original failure is
preserved. The running CUDA snapshot predates this parser-only correction and
needs its follow-up check before any latest-code GPU claim.


## CUDA and injection evidence checkpoint

The pending CUDA statements above are historical: both controllers have now
completed successfully, including the parser correction. Six nonlinear mesh
oracle cases pass at 1.111e-16; seeded serial/two-rank decomposition and repeated
rank-changing restart are bitwise equal. CPU/CUDA all-field/all-ghost comparisons
pass at 3.331e-16. Default UCX memcheck remains a recorded failure from CUDA
context API calls during MPI initialization; single-rank ob1 + pt2pt passes with
zero errors on the same evolution binary/input/restart.

The independent periodic FD/KO assembly around compiled PointRHS measures nonzero
reduction/curl injection converging at the expected FD2/4/6 rates, including the
nonlinear lapse tangent and variable-lambda curl term. It is a smooth snapshot
check, not a production stage budget or an Einstein evolution qualification.
See `analysis/pc_gh_clean_reduction/INTRINSIC_CUDA_INJECTION.md` and evidence
`intrinsic-mesh-cuda-001/`, `intrinsic-injection-001/`. No production equation or
operator changed in this checkpoint. Production diagnostics and signed stage
budgets are the next implementation priority; intrinsic refinement and all
physical convergence/puncture/binary promotion gates remain open.

## Actual RK and exchange stage instrumentation

The opt-in intrinsic stage dump now records actual float64 states, valid active
RHS and RK accumulator, with explicit stage and ghost-validity metadata. Six
serial multi-block FD2/4/6 2D/3D fixtures pass at all 18 RK stages; dump on/off
is bitwise neutral, RK reconstruction agrees to 1.110e-16, and active periodic
transfer increments and synchronized ghost errors are zero. Duplicate-write
controls preserve earlier stage data. All 19 legacy restart controls pass.

Signed physical/reduction/curl increments and actual-RHS semidiscrete defects
are assembled offline on reconstructed periodic global arrays, with component
norms and extrema. This is not yet an in-process physical history diagnostic.
See `analysis/pc_gh_clean_reduction/INTRINSIC_STAGE_BUDGET.md`. Stage-dump MPI/CUDA
validation and production diagnostic integration remain next; refinement/core
and physical evolution/puncture/binary gates remain open.

The two-rank extension of actual stage instrumentation now passes: all 54
operation payloads agree bitwise with serial, including active RHS and RK
registers. Twenty-four malformed rank/stage grouping controls are rejected.
See `intrinsic-stage-mpi-001/`. CUDA controller PID 1860046 is confirmed live
building the previous production commit, with serial/MPI tests queued inside
the same controller; its result is not yet available. Reader/test changes in
this checkpoint do not change production equations or operators.

## Primary physical stencil and completed CUDA stages

A primary-only physical H/M diagnostic kernel now passes 612 analytic compiled
CPU points across two oblique fixtures, including nonzero conformal Ricci,
FD2/4/6 and 2D/3D. The direct second/mixed derivative construction fits the
existing valid ghost reach. It remains to be wired into synchronized mesh tasks
and rank-reduced histories; its CUDA test is not run. See
`analysis/pc_gh_clean_reduction/INTRINSIC_PHYSICAL_STENCIL.md`.

The preceding CUDA stage controller is now terminal with exit zero. All six
serial and six two-rank suites pass, including dump neutrality, RK reconstruction,
valid ghost checks and duplicate-write controls. The verified binary is
`e10099aa4138bed435f26095fa4966f2aebe1825efbb1fd12c876f3af0c94c0a`;
all 337 source manifest entries match the preceding production source.
Evidence: `intrinsic-stage-cuda-001/`. Earlier pending statements are historical.
No new evolution equation is enabled by the physical diagnostic header.

## In-process physical and reduction histories

The intrinsic synchronized constraint task now computes and globally aggregates
all 89 physical/GH/reduction/curl components when `intrinsic_diagnostics=true`.
The independent primary physical stencil operates on valid materialized geometry
halos. CSV observations retain actual volume, component norms, signed maxima,
block/level/local indices and physical coordinates with explicit cycle/stage.
Exclusive file creation preserves prior epochs; diagnostic cadence is explicit.

Serial and two-rank FD2/4/6 2D/3D initial/final checks agree with the independent
89-component oracle to 1.142e-13. Diagnostic on/off is bitwise neutral. Cadence,
restart history/state continuity and collision controls pass, as do all 19 legacy
restart controls and the historical offline diagnostic regression. See
`analysis/pc_gh_clean_reduction/INTRINSIC_DIAGNOSTIC_TASK.md` and evidence
`intrinsic-diagnostic-task-001/`. CUDA validation of this new task is not run.
Only the full uniform-periodic region is implemented here; excised regions,
intrinsic interfaces and physical convergence/puncture/binary gates remain open.

## Nonlinear smooth convergence and CUDA diagnostics

The CUDA component-diagnostic controller is complete: serial/two-rank oracle
and cadence/restart/collision controls pass; maximum component discrepancy is
1.144e-13. All 338 production-source entries match the tested source.

The 2D off-constraint smooth PDE fixture shows RK3 temporal convergence near
order three. Uniform FD2/4/6 spatial group norms pass their order/alignment and
independent temporal-error controls. A component audit, however, finds poor
rho alignment at KO=0.3 (0.751 for FD6), so that ladder does not support per-field
Richardson qualification. A matched FD6 KO=0 control improves every component's
alignment above 0.99997 and order above 5.9659. Both the negative original arm
and positive control are retained. No physical Einstein-data convergence or
intrinsic-interface/puncture/binary qualification is claimed. See
`analysis/pc_gh_clean_reduction/INTRINSIC_SMOOTH_CONVERGENCE.md` for exact scope,
reproduction commands, timings and the component plot.
