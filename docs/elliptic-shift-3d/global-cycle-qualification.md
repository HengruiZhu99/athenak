# Optional global topology hierarchy: actual cycle qualification

`CompositePoisson(mesh, radial_stencils, radial_lines, topology)` now accepts an
optional fourth Boolean, `topology=false`. The default classic hierarchy retains
its original logical 2:1 transfers, radial-line flag recursion, fine operator,
and 1,024-cell dense coarse cap. The new hierarchy is still opt-in; no stage,
pilot, CMake, native task dispatch or user input flag was changed here.

When enabled, the constructor validates the complete
`CompositeCoarseningPlan`, creates a fresh global `CompositeDifference` at
**each** descriptor level, and rebuilds its active-only face/edge/corner ghost
plans there. It constructs `CompositeLevelTransfer` between adjacent levels and
carries accumulated physical mapped-cell measures into the next restriction.
The measure is the scalar Poisson preconditioner's mapped geometric volume;
the outer vector solver retains its full metric-dependent norm and operator.
`Cycle()` applies the unchanged fine/global operator, restricts its defect,
recursively solves the coarse correction, prolongs to all fine active cells,
and applies the selected existing smoother. Homogeneous correction boundary
data continue to enter the global ghost operator. No interface values are held
fixed during the outer solve.

The new constructor branch uses ceiling-halving down to N2, including N3→2,
then complete finest Cartesian sibling merges. The terminal dense matrix is
constructed only after the plan passes the existing cap. Shell descriptors and
refinement-zero Cartesian roots remain unchanged. A sufficiently large shell
or root floor is still explicitly unsupported, rather than allocated as an
unbounded dense matrix. The transfer-only proof and negative donor controls are
recorded in [amr-topology-coarsening.md](amr-topology-coarsening.md).

## What was qualified

The new standalone source `tst/elliptic_shift_3d/test_topology_cycles.cpp` runs
the existing flexible GMRES with the actual installed `CompositePoisson::Cycle`
as a right preconditioner. It tests:

* The unchanged second-order global scalar Poisson operator, checked against
  direct fine-grid stencil evaluation on a separate nonpolynomial field.
* A direct functional implementation of the fourth-order full vector operator,
  using the same `MetricJet`, `Geometry` and `ApplyEllipticShift` tensor kernels
  as `CompositeShiftStage::Apply`. Geometry is frozen from the existing
  non-conformally-flat `manufactured::State(...,true)` fixture; all metric,
  determinant, curvature, mixed derivative and vector coupling terms are kept.
  This is not a claim that `CompositeShiftStage` has a new topology argument.
* Direct inhomogeneous Cartesian vector boundary data `beta=-x` at the unit
  outer sphere. The scalar fields vanish there. The vector boundary is lifted
  through the original ghost operator before the homogeneous correction solve;
  no potential boundary conditions are substituted.

Most cases use **discrete manufactured right-hand sides** to isolate linear
inversion of the final global operator from spatial truncation. They use a
nontrivial nonspherical exact field, start the solution at zero, and evaluate
the original residual again after GMRES. A separate curved-metric case uses the
analytic continuum source from `EllipticShiftSource`, and reports its spatial
error. Its N6 physical solution error is 0.001433 despite a residual below
7e-11; it is a coarse spatial control, not a physically converged evolution.
Every case also records the continuum source/stencil discrepancy separately.

## Reproduction and frozen evidence

```sh
python3 tst/elliptic_shift_3d/run_topology_cycles.py \
  --build build-shift-3d \
  --output docs/elliptic-shift-3d/evidence/topology-cycles-reproduction \
  --cases scalar_n6_s7 vector_n8_s7 scalar_n8_mixed vector_n8_mixed \
          scalar_n8_subdivided vector_n8_subdivided
```

All 25 completed case commands and exact results are in
`evidence/topology-cycles-qualified-20261027/manifest.json`. Each process writes
its own stdout/stderr; the manifest records the source HEAD, frozen headers,
compiler/configuration, library and binary hashes, full commands, durations,
and live source/library stability. The binary is compiled from the **frozen
archive**, with a distinct output path. The existing pilot binary and CMake
targets were not rebuilt. Both source and libraries remained unchanged for the
final qualification. Build took 2.35 seconds; the 25 serial processes totalled
45.73 seconds, with a maximum of 6.99 seconds per process. Each has a 180-second
limit and a 50,000-active-cell guard. The largest tested solver grid has 28,672
active cells. These are bounded local host tests; timings are not isolated
performance measurements or scalability evidence.

The main matrix has five meshes per resolution: one, two, and seven shell
layers; two-shell mixed Cartesian AMR; and two shells with four blocks on each
face. Every base mesh has eight Cartesian roots. Mixed AMR refines one corner
root and then its corner child, directly adjoining the Cartesian/shell
interface. This produces 34 leaves with three Cartesian refinement levels.
The seven shell radii are `.45,.55,.60,.65,.70,.75,1`. Shell AMR is not introduced.

All twenty matrix cases use the existing radial-line smoother and meet the
unchanged acceptance limits: physical global RMS <=1e-9 and all-cell maximum
<=1e-7. The Krylov inner absolute tolerance is 1e-10. No cells are excluded.
Bulk, shell, interface/edge/corner collars, outer-boundary collars, and refined
cells have separate recorded RMS/max diagnostics. The largest all-cell residual
over all 25 cases is 1.541e-8; maximum global RMS is 9.50e-11. The scalar fine
operator comparison is exactly zero in every case.

| Operator / resolution | Mesh cases | GMRES iterations | Largest residual RMS | Largest all-cell residual |
|---|---:|---:|---:|---:|
| Scalar N6 | 5 | 11–19 | 7.97e-11 | 7.95e-9 |
| Scalar N8 | 5 | 11–22 | 9.33e-11 | 5.29e-9 |
| Curved vector, fourth order, N6 | 5 | 18–30 | 9.44e-11 | 8.62e-9 |
| Curved vector, fourth order, N8 | 5 | 19–34 | 8.70e-11 | 1.55e-8 |

Representative actual hierarchies and costs:

| Case | Active cells by level | Iterations | Setup / solve seconds |
|---|---|---:|---:|
| Seven shells, vector N6 | 10800→1350→400 | 22 | 1.35 / 1.85 |
| Seven shells, vector N8 | 25600→3200→400 | 24 | 2.77 / 4.13 |
| Mixed AMR, vector N6 | 7344→918→272→216→160 | 30 | 0.43 / 1.76 |
| Mixed AMR, vector N8 | 17408→2176→272→216→160 | 34 | 0.68 / 3.86 |
| Subdivided faces, vector N8 | 28672→3584→448 | 19 | 1.61 / 3.27 |

The remaining five controls pass: classic hierarchy with topology disabled;
curved vector with analytic continuum source; flat-metric vector reduction; and
scalar/vector N6 two-shell point-smoother controls. The point smoother needs 42
scalar and 75 vector iterations, compared with 13 and 20 using radial lines on
that mesh. This measures a small local test only and does not establish that
point smoothing is adequate on all thin-shell meshes.

## Preserved failure and independent tolerance control

The initial scalar N8 mixed-AMR solve at inner tolerance 1e-9 converged in
global RMS (6.58e-10) but missed the fixed maximum criterion (1.264e-7 versus
1e-7) at refined interface cells. The original result is retained in
`evidence/topology-cycles-matrix-20261027`, with exit status one. A source hash
changed during that run because a class comment was updated; the binary itself
was compiled from its already-frozen archive, whose exact bytes are retained.
The manifest records that live-source difference instead of claiming stability.

Tightening only the Krylov inner tolerance to 1e-10, while retaining acceptance
RMS <=1e-9 and maximum <=1e-7, gives scalar RMS 2.95e-11 and maximum 5.29e-9 in
22 iterations. The corresponding curved fourth-order vector solve reaches RMS
8.70e-11 and maximum 1.55e-8 in 34 iterations. These independent controls are in
`evidence/topology-cycles-mixed-inner-control-20261027`. No gauge, source,
boundary, spatial discretization, or acceptance tolerance changed. Earlier
initial N6/one-shell, N6/two-shell vector and N8/seven-shell scalar results are
retained in `evidence/topology-cycles-initial-20261027`.

## Remaining qualification and integration

The optional scalar preconditioner hierarchy is now qualified for these global
scalar and full vector elliptic tests. It has not been exposed through a stage
or pilot input, exercised in an actual coupled AMR/regrid evolution, or used in
native AthenaK task/shell/restart wiring. After regrid, coarsening, restart, or a
changed map, rebuild every descriptor, transfer, coarse ghost plan, and line
factor; this component does not update cached objects in place. The full shift
must be solved again before the next tensor RHS. Root-stage epoch and endpoint
state requirements remain in `native-wiring-audit.md` and
`native_composite_stage.hpp`.

The 4,096-Cartesian-leaf descriptor/transfer test was completed separately; an
actual solver cycle at that topology depth is not claimed. No MPI, device,
moving-puncture, or binary test was run here. The final vector discretization's
spatial convergence, scri tensor evolution closure, wormhole radial reproduction
and coupled scientific acceptance remain separate gates. No new extrapolation,
physical background subtraction, spherical projection, or reference-gauge
change is introduced by this hierarchy.
