# Implementation and validation status, 2026-10-27

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
