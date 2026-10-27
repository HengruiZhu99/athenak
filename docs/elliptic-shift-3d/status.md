# Current continuation checkpoint, 2026-10-27

**The first scientific milestone remains incomplete.** The full variable-metric
vector solver converges on the tested global composite CPU domains. Actual
AthenaK Driver/RK tasks now dispatch the shared tensor evolution and shift
solver in a deliberately restricted experimental configuration. The selected
wormhole evolution still fails the unchanged scientific accuracy gates.

## Implemented and checked

- The exact metric-built connection, its full variable-metric Frechet operator,
  independent evolved Lambda, future-null/fixed-angle shift data and prescribed
  scri lapse clock are retained. No CFC, differential projection, field floor,
  puncture beta pin or physical-RHS background subtraction is used.
- The host hierarchy supports odd logical sizes, complete Cartesian sibling
  coarsening, global face/edge/corner and coarse/fine halos at every level,
  volume-aware restriction and global active prolongation. Twenty-five actual
  scalar/curved-vector cycle qualifications pass, including seven shell layers,
  mixed Cartesian AMR and shell subdivisions where the coarse budget allows.
  The 1024-cell dense coarse guard remains; angular shell topology coarsening
  is still missing. This is selectively reused scalar MG infrastructure and
  new mapped/coupled code, not Alan's four-potential CFC solver.
- Global radial-line smoothing changes only the scalar preconditioner. Original
  physical/conformal RMS and point residuals are recomputed independently of
  the preconditioner. Tight-tolerance and timestep controls separate solver and
  temporal errors from the unresolved spatial/interface errors.
- Exponent-independent positive log fields and local polynomial-plus-log
  derivatives retain the original physical values. Puncture position, map,
  storage, rank, conditioning and mesh lifecycle guards are explicit. An
  optional value-only Cartesian coarse/fine halo cache uses the same fixed
  function space, with bounded donor enlargement and VALUE amplification.
  It changes no active value, signed tensor, beta or evolution equation.
- Fixed odd-radial enrichment was rejected as a general replacement after
  fractional angular/noise counterexamples. A separate affine-plus-remainder
  beta prototype preserves constant/linear translation modes without power
  division; its augmented global solve is not implemented. Neither prototype
  establishes generic wormhole-to-trumpet or moving-puncture safety.

## Actual native Task/RK/restart validation

The experimental `z4c/composite_elliptic_experimental=true` path uses actual
AthenaK task callbacks and live Driver RK coefficients, including all 22 free
3D tensor/gauge fields and the algebraic full-vector solve at each stage.
[The native architecture and exact commands](native-composite-integration.md)
record its restrictions and failed attempts. A mechanical extraction shares
the same tensor engine with the research
pilot; the native path never invokes the pilot's classical-RK Step method.
The canonical host core/shell/scri state has a verified physical native-core
mirror and separate physical ADM gauge conversion. Independent Lambda and
nonzero telegraph flux evolve; there is no symmetry enforcement.

The native two-shell Minkowski test reaches t=.02 in 40 RK4 cycles (42.23s):
H/M RMS 2.02e-15/4.80e-15, original shift RMS maximum 1.65e-13,
scri pole 8.97e-14, ghost clock 0 and null/tangential residuals below 7e-16.
Small radial and nonspherical lapse pulses both complete with nonzero flux
and tensor response. Their actual native restart continuations are bit-identical
in full volume, scri, flux, RK registers and warm shift. Independent native
binary ADM decoding agrees at the output precision. Corruption, mirror edit,
changed RK and changed pulse configuration are rejected. Evidence:
`evidence/native-composite-pulse-controls-20261027` and
`evidence/native-composite-task-qualified-20261027`.

This is Serial/double/vacuum with one static cubic native Cartesian block,
ng=3/order=4, and dedicated Minkowski initialization. Shells are engine-owned
host blocks, not native MeshBlocks or device/rank-distributed storage. Native
wormhole initialization, native AMR/regridding, matter, tracking, MPI/GPU and
midstage restart remain unsupported and guarded. Native restart is paired
with a complete composite sidecar; this does not claim general shell output
or native distributed multigrid support. Full-grid characteristic, scri penalty
and flux-relaxation timestep selection is now active. A small-CFL control
selects4.861111e-6 from a requested ceiling .0005 and advances with that step;
all34 default output CSVs remain bit-identical after this update. A .98,1
thin-shell attempt fails the initial shift solve and is retained as unsupported
by that tight-tolerance control, not counted as a runtime CFL pass.

## Selected wormhole: actual coupled failures retained

The following full 3D classical-RK4 research pilots solve the full vector
shift each stage and evolve actual Omega=0 with the M=1/3 isotropic wormhole,
Minkowski reference and verified spherical taper width .24. All reach only
t=.0002 and restart bitwise; none establishes collapse/front convergence.

| Configuration | Cells | H RMS | M RMS | Maximum H | Tensor scri pole |
|---|---:|---:|---:|---:|---:|
| N8, 3 shells, log derivatives | 13312 | .51056 | .14016 | — | 3.93e-6 |
| N8, 7 buffered shells, radial lines | 25600 | .50906 | .02839 | 1.57085 | 1.46e-6 |
| N6, 7 shells, 3 Cartesian refinement levels | 47088 | .02981 | .03286 | .40613 | 1.44e-5 |
| Same, value-only log halos within rho<.12 | 47088 | .02968 | .03286 | .19555 | 1.44e-5 |

Every case **fails** the unchanged 1e-7 pole target and remaining scientific
criteria. Every active cell remains in the norms. The last run costs 127.55s,
including cache rebuild/restart, and has original vector RMS maximum 2.73e-11
and point maximum 1.25e-9. The halo cache uses 9,616 plans for all 19,248 selected
receivers; no exclusion or fallback hides a rejected fit. Its reduced maximum
moves to the next Cartesian coarse/fine interface at r=.16949, outside .12.
The expanded support and scri component localization are separate followups.

Step halving in the seven-shell N8 pilot changes fields/shift by 3.10e-11;
tightening original vector RMS from 1e-10 to 1e-12 changes them by 3.06e-13.
Thus converged elliptic tolerance does not fix these discretization defects.
Earlier 180s timeout and auxiliary-norm stopping failures remain recorded.
Current research checkpoint v13 binds the halo choice; older versions retain
old defaults. Earlier .08-taper controls are explicitly not the selected .24
benchmark. The integrated suite now passes 31/31 in 13.46s, including the new value-only
halo and affine-remainder contracts (`evidence/integrated-continuation-20261027`).
The actual native task controls are separately executed and recorded above.

Frozen true 1D collapse snapshots at t=1/5/10 have nine instantaneous full 3D
solves with original RMS 5.2–5.9e-11. Shift RMS differences are about .006 with
FD sources and .001 with independent numerical snapshot jets. They are not 3D
evolution to those times. The recorded t=10 flux feature near r=4e-7 is far
below present 3D resolution; AF and outer trumpet-like powers coexist. A budget
estimate for eight cells across that radius requires about 599k active cells
and 20 nested levels, with over 800MB for state/Krylov arrays before halos.
This is a geometry/memory estimate, not a request or a scalability result.

Remaining work includes all-region radial accuracy and scri consistency,
applicable constraint/scalar/collapse families, a validated changing-regularity
puncture endpoint, and general native shell ownership/AMR/MPI/device paths.
No mathematical incompatibility or new permission blocker has been established.
Original 1D jobs/checkouts have never been modified or signaled by this task.

---

## Historical checkpoints (superseded by the status above)

# Implementation and validation status, 2026-10-27

**Current wormhole continuation:** commit `62170e4bf` implements the selected
M=1/3 isotropic-wormhole source and three positive coefficient-chart controls.
[The derivation and equivalence audit](wormhole-chart.md) explain why these
factors retain the Minkowski reference connection, gauge sources and complete
physical RHS. Selected lapse agreement with independent 1D is 5.53e-15
relative; full tensor/gauge chart equivalence is 1.94e-12 relative. The
[final matrix](evidence/wormhole-final-20261027/source-revision.json) contains
three related N=6/8/12 families through t=.0002: **18 setup/initializer passes,
27 retained scientific failures, no timeouts**. All original vector solves
converge, all three nonzero wormhole checkpoints restart bitwise, and actual
ghost face boundary data hold at roundoff. The coupled tensor poles do not
meet the unchanged 1e-7 gate. The best smooth-chart N12 case has H/M RMS
.0483254/.133361 and pole2.61657e-5. A halved timestep or tightened elliptic
tolerance changes stored fields/shift by only about2.1e-12. These tests identify
unresolved spatial/interface/scri errors; they do not establish an accepted
wormhole or permit a late-time stability claim. The compact blend is rejected
because its unresolved collar produces very large constraint errors.

The independent [spherical reference](evidence/wormhole-radial-reference-20261027/manifest.json)
has been extended through N2049, with explicit N1025/N2049 differences. The
[comparison](evidence/wormhole-final-20261027/comparison.json) retains raw
angular scatter, every cell, full-domain norms, puncture/transition/scri norms,
time/tolerance controls and all failed plots. No general 3D Misner-Sharp mass
is used. Compressed data can be restored with `rehydrate_evidence.py`.

The native 25-field pack adapter now passes independent roundtrip/rate/ghost
tests on twenty permuted bulk/shell blocks: roundtrip2.22e-16,
physical-rate finite-difference error1.15e-11, untouched ghost/block error0.
It uses actual `Z4c` enums and Kokkos mirrors, but is **not connected to native
tasks** and does not supply native shell allocation, MPI, AMR lifecycle or
restart. The latest focused CTest suite has10/10 passes. All coupled results
remain from the Serial CPU research harness, with Kokkos-inline physical
kernels; the inherited AthenaK MG transfer stencils are adapted in the host
preconditioner, not a port of the native MG task driver or CFC physics.

The spherical checkout independently advanced to `98fcd04e1` during this work.
[A read-only audit](evidence/spherical-tip-audit-20261027/receipt.json) records
new late puncture failures and experimental floor/attenuation controls. None
were imported or called successful. Historical paused jobs and the new active
spherical workers89528/89529 were untouched. The isolated branch remains based
on the inspected starting HEAD8bef4966e. There is no mathematical impossibility
or permission denial established here: native integration and complete AF
angular endpoint treatment remain engineering/formulation work, and the
measured coupled failures are the gate against extending these pilots.

The following sections preserve earlier implementation/evidence snapshots;
their outstanding-item descriptions are historical where superseded above.

**Continuation update:** the full tensor finite limiting source and evolved scri
surface are now implemented and independently checked. See
[tensor-scri-closure.md](tensor-scri-closure.md) and
[the continuation matrix](evidence/tensor-scri-20261027/manifest.json).
All15 continuation cases pass their stated component/boundary checks, including
the nonzero-pulse boundary-state restart and a t=.02 radial extension. The old
nine active-only extrapolation failures remain preserved; an initial-time and
degree-six manufactured audit shows why that extrapolation is a different
quantity from the actual discrete face boundary residual. Raw tensor poles and
raw clock-rate corner defects remain separately monitored. The historical
sections below describe the first frozen backend; they are not overwritten or
retrospectively marked accepted. Native integration, selected wormhole/chart,
scalar and late-time scientific coverage remain work in progress.

**Partial implementation; the requested first scientific milestone is not
complete.** There is a working global composite-grid CPU elliptic backend and
short genuine 3D coupled tensor evolution. Production AthenaK task integration,
a complete regular scri closure, the selected M=1/3 wormhole lapse/puncture
formulation, scalar propagation and late-time validation remain outstanding.

The implementation and frozen numerical evidence are committed as
`487f750d8b8503b4944ee75d0b3b1efdcee1b72a` on
`codex/hyperboloidal-elliptic-shift-3d-multigrid`, based on the actual spherical
HEAD `8bef4966e253e92d85761f46d8885e7cc80dc2ed`. The checkout is
`/Users/hz0693/Documents/Codex/2026-10-26/task/hyperboloidal-3d`.
Subsequent reporting work audits surface interpolation and fixes figure labels;
it does not change the validated solver or evolution kernels.

## Implemented scope

| Layer | Implemented behavior | Production limitation |
| --- | --- | --- |
| Physics | Exact Frechet derivative of the metric-built connection; every variable-metric term; independent evolved Lambda; direct future-scri/fixed-angle vector data | Kokkos-inline kernels are available, but have not been wired into AthenaK's native evolution task graph |
| Geometry | Cartesian leaf boxes, cube-to-sphere transition, arbitrary additional sphere layers, subdivided faces, common N-cubed shape and two ghosts | New host descriptor; production six-block extension is unchanged |
| Connectivity | Active-only interpolation through seams, edges, corners, Cartesian/shell and mixed Cartesian refinement interfaces | Host reference plans; native rank-packed/device halo engine not integrated |
| Solver | Fourth-order original vector operator, flexible GMRES, second-order global mapped scalar MG preconditioner, global coarse solve | Adapts AthenaK transfer stencils; does not subclass/reuse the Cartesian MG task driver or import CFC physics |
| Coupled stages | Original free `ConformalRHS`, Appendix-C Theta option, live elliptic shift every RK4 stage, reference-vector telegrapher, no symmetry averaging or constraint projection | Separate C++ CPU research harness; regular-center symmetric-restoring lapse control, not selected wormhole source |
| Scri | Exact physical face r=1/Omega=0; prescribed lapse clock and three vector shift conditions; independently measured face pole/clock/null/tangential defects | Geometric active-only outflow extrapolation is experimental; no complete tensor corner/characteristic regularity closure |
| Restart/regrid | All bulk/shell/new flux fields, operator/gauge controls and warm start; corruption/source mismatch rejection; rebuilt hierarchies; bitwise continuation | Research format only; production native restart guards remain |
| MPI | Owned Krylov vectors, real local one/two-rank communication and reductions | Global preconditioner replicated on each rank; correctness oracle, not scalable production MPI |

The inherited production extension still rejects hyperboloidal patches,
nontrivial MPI, restart and telegrapher configurations. No guard was removed to
manufacture apparent support. The native `src/athena` executable was not the
runner for the new numerical results.

## Verified evidence

The frozen [manifest](evidence/validated-20261027/manifest.json) records every
command, exit code, source and executable hash, elapsed time and complete raw
result. All profile rows are retained losslessly as gzip CSV. Its 27 cases have
**18 zero exits and 9 explicitly retained scientific boundary-consistency
failures**. None timed out. The final [CTest log](evidence/ctest-final.log) has
5/5 passing registered component/regression tests. Original source files and
jobs were preserved; see [preservation](evidence/preservation.json) and the
[process receipt](evidence/preserved-processes-after.txt).

The exact operator audit gives connection-variation error 5.69e-12, flat Lamé
error 5.55e-17, tensor metric-RHS mismatch 5.55e-17 and independent nonflat
radial residual error 2.65e-14. Scri sign and both angular conditions are checked
on a nondiagonal metric; the vector telegrapher agrees with the radial formulas.

For the genuinely non-conformally-flat manufactured metric, Cotton components
have maximum .51480255087; halving the independent Cotton derivative step
changes that value by 8.84e-11. The solution RMS errors at N=8/12/16 are
1.21169e-4, 2.40630e-5, 7.62792e-6, giving orders **3.9868 and 3.9935**.
Original vector residual RMS is below 1e-9. Tightening the solve tolerance to
1e-11 changes the N12 solution error only at about 2.2e-13; a second-order
**final operator** instead has error 6.01488e-4. Tolerance cannot repair that
discretization error. See [convergence/cost](evidence/validated-20261027/elliptic-convergence.png).

The mixed-refinement case has 63 blocks, 32,256 cells, two shell layers and 2x2
face subdivisions. It converges in 23 iterations to original RMS residual
7.60e-10, with a separately reported point maximum 3.37e-8. Its one/two-layer
and interface errors remain distinct. The regrid test refines **one of eight**
Cartesian leaves touching the shell, rather than uniformly refining the whole
central box. Refine/coarsen transfer error for the independent polynomial is
8.22e-15. All block shapes remain equal. Shell AMR is not implemented.

Local MPI one/two-rank solves both take 27 iterations, with original residuals
4.01950e-10/4.01952e-10 and solution RMS differences about 2.2e-17. Costs are
.78/.82 seconds on this small run; each rank performs 58 allgathers and 643
allreductions. There is **no scalability claim**. The sandbox initially rejected
OpenMPI's local socket bind before execution; the approved local-only retry
completed. No pending denied action remains.

## Coupled results and failed gates

The pilot duration was fixed at **t=.002**, amplitude 1e-6, at N=6/8/12. It uses
up to 22,464 cells for coupled evolution, one CPU worker and bounded per-case
timeouts. Three Minkowski cases, including two shells, preserve equilibrium at
roundoff; the finest two-shell maximum drift is 6.12e-15. A nonzero radial-pulse
checkpoint also reproduces the uninterrupted evolved state and shift warm start
bitwise after hierarchy reconstruction.

At N12 the radial pulse's full-domain H/M RMS are 1.41e-11/1.80e-10. Its
pointwise independent-1D lapse comparison RMS improves from 1.21e-11 to
4.33e-12 to 8.60e-13 across the three grids. All evolved geometry fields remain
free, and the [matched-radius comparison](evidence/validated-20261027/radial-3d-comparison.png)
retains angular scatter. Surface extraction can itself create substantial
coarse angular scatter. The separate
[interpolation audit](evidence/validated-20261027/surface-interpolation-audit.json)
projects independent 1D Cartesian tensors through exactly the same extraction;
it reproduces the native extraction to 1.11e-16, retains both raw and controlled
errors, and never alters evolution data or full-domain norms.

Halving/quartering dt at the middle grid changes the complete stored fields and
shift by at most 6.47e-14/2.07e-13. Tightening the elliptic tolerance from 1e-11
to 1e-13 changes them by at most 1.29e-13. The second-order spatial control
changes them by 2.17e-9. These distinguish solver, time and spatial effects, but
the tiny duration/amplitude leaves temporal differences near solver/roundoff
effects: **resolved RK4 temporal order is not established**.

The original clock/null reference targets are 1e-13/1e-12. The radial N12
active-only face consistency defects are 3.89e-11/4.00e-11 and still **fail**.
For the nonspherical two-shell pulse, N6 and N8 also fail; N12 reaches
7.43e-14/7.83e-14, with raw face pole 1.98e-12, H RMS 7.68e-12 and positive
lapse/chi. This individual short success does not admit the whole family or
validate late fronts. The [boundary figure](evidence/validated-20261027/coupled-boundary-consistency.png)
shows every failed resolution against the unchanged targets. Prescribed face
data and independently extrapolated active-field consistency are separate
quantities; neither is hidden by a potential residual or a norm exclusion.

## Remaining blockers and required work

1. **Regular scri evolution closure:** the normal/two tangential elliptic data
   are derived and implemented, but a full general 3D tensor characteristic and
   pole-limit closure is not. Coarse radial consistency failures are reproducible
   and need resolution/closure work before extending duration or claiming the
   first milestone. Tiny positive-field survival is insufficient.
2. **Required nonlinear wormhole:** the selected multiplicative/physical-log/
   collapsed-inner lapse and actual AF puncture coefficient chart have not been
   extended with full 3D angular/tensor singular powers. Regular-center parity,
   a finite inner cutoff or a substituted CMC trumpet would change the requested
   benchmark, so none was used. This blocks the M=1/3 wormhole evolution.
3. **Native integration:** AthenaK pack/task-stage wiring, production shell
   allocation, native distributed/device multigrid communication, regrid and
   restart integration remain engineering work. The host and replicated-MPI
   reference backends are not presented as those implementations.
4. **Scientific coverage:** dedicated scalar and constraint-pulse propagation,
   long-duration fronts, matched wormhole mass/lapse comparisons and completed
   temporal/interface controls are still required. No generic 3D Misner-Sharp
   mass or long-time convergence claim is made. Moving/binary punctures remain
   explicitly later milestones.

No remote allocation or expensive GPU run was requested or started. There is no
remaining permission denial; these are numerical/formulation and implementation
gaps, not an instruction to weaken acceptance thresholds.
