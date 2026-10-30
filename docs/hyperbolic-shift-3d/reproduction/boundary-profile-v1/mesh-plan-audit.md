# Mesh-plan audit, current fixed-grid runner

Audit basis: source `555ad41facd84c90a303f2f80f794ed3b917efaa` and the
verified V21/V22 execution comparisons. This records implementation scope,
not a claim that the persistent performance goal is complete.

| Operation | Persistent data | Per-stage work | Lifecycle status |
|---|---|---|---|
| Core/shell ghost fill | `HyperbolicCompositeExchange` constructor uploads active IDs, halo row offsets, donor indices/weights, scri interpolation plans and positive reference factors | Bulk copy/interpolation; current metric positivity and null projection | Constructed once; must be replaced when mesh/ownership changes |
| Mapped bulk derivatives | `HyperbolicCompositeDevice` uploads Cartesian Jacobians/Hessians, active indices, volumes and KO terms | Fixed centered/biased logical differences, mapped contractions, state-dependent transport direction | Constructed once; device operator currently has shape checks, not a mesh epoch check |
| Direct radial derivatives | `ScriRadialDevice` uploads derivative donor indices and first/second derivative weights | Applies the same ordered weighted sums to current volume/surface fields | Constructed once with owning operator |
| Scri jets and radial numerator samples | `HyperbolicScriDevice` and `HyperbolicSampledNormalDevice` upload maps, stencils, normal derivative weights and donor IDs | Bulk jet sampling, nonlinear geometric equations and compatibility reconstruction | Constructed once; separate surface/volume ownership is honored |
| Angular dissipation | `HyperbolicAngularDissipationDevice` uploads the fixed angular plan | Current residuals and their dissipation; cached only when the entire stage state and timestep are unchanged | Whole-state reuse invalidated by gather/upload/rollback |
| Sparse exchange | `SparseStageExchange` owns request counts/offsets, pack/unpack IDs and device buffers | Pack field values, GPU-aware MPI, unpack values | Collective topology/geometry/partition epoch rejection and rebuild tested |
| Horizon sampling | Horizon points depend on the current trial surface | Owner search and jets at each current sample | Deliberately not cached as immutable mesh locations |

`Evolution` constructs these objects once before stepping. No `Owner`/`Inverse`
search or interpolation-polynomial construction occurs in the device `Fill`
applications. Fixed finite-difference coefficients and integer offsets are
simple kernel expressions, not donor searches. Some analytic background
expressions (radius, reference lapse, Omega) remain evaluated in kernels; only
the stage/floor reference lapse and positive dissipation references have dedicated
persistent arrays. Whether additional arrays help must be measured against their
memory traffic and GPU register use. State-dependent metric factors, transport
upwinding, lapse/shift coefficients and moving-normal closure must stay current.

## What the lifecycle tests prove, and what remains

The sparse plan rejects a stale or inconsistent epoch collectively, invalidates
an old plan before attempting rebuild, and supports rebuild with changed mesh
size, donor set and ownership cuts. Poisoned-workspace tests verify that only
requested/owned entries are consumed. Restart tests reconstruct all device plans
from the checkpoint geometry descriptor and chosen ownership.

However, the current runner has a fixed mesh. Other device operators do not yet
share an epoch-checked owning bundle: mutating a host mesh underneath an existing
operator is unsupported. A future refinement/coarsening or geometry-change event
must rebuild **all** dependent operators and exchange plans as one transaction,
then migrate/prolong/restrict state and compensation arrays and rebuild diagnostic
ownership. Merely calling `SparseStageExchange::Rebuild` is insufficient. Dynamic
AMR field transfer, coherent bundle invalidation/rebuild, and trajectory tests
across such events remain outstanding; the current epoch unit test does not prove
AMR evolution support.

## Timing evidence and next decision

Verified sparse exchange plus earlier boundary reuse improved the per-step
100-step puncture comparison by 1.92x against frozen uncached V11. A separate
same-node test of rank-average measured cuts did not reduce slowest-rank RHS
cost and was 1.3% slower per step; those cuts remain experimental. Do not treat
MPI waiting as intrinsic per-cell work or multiply cross-job speedups.

V23 adds boundary subphase and all-error-synchronization timers to distinguish
actual boundary application from waiting. The verified job 8987400 profile identifies sampled-normal numerator RHS
(11–12 s per 100 steps) as the dominant boundary cost; see the adjacent README.
Per-stage sparse communication is already small in the V21/V22 profiles, so an
asynchronous exchange is not automatically worthwhile. Any overlap experiment
must preserve producer/consumer and collective ordering, drain pending requests
on failure, and compare complete checkpoints and failure captures.

## Rebuild integration requirements found during V24 qualification

The optimized continuous Cartesian core has an additional restriction beyond
cache invalidation: `CompositeDifference::BuildCartesianLookup` explicitly
rejects mixed Cartesian spacings. Therefore, a local leaf-refinement event cannot
currently rebuild this particular discretization successfully. The older mapped
core component fixtures support coarse/fine layouts, but those fixtures do not
establish mixed-level support in the continuous-core evolution runner. A safe
cache owner must propagate this rejection to every rank and prevent use of a
partially rebuilt mesh; it must not quietly switch Cartesian discretizations.

A coordinated device-plan owner must include the two halo applications, bulk
operator and its radial plan, surface operator and global projector, sampled
normal operator and its radial plan, angular operator, sparse exchange,
reference-lapse array, positive-dissipation references, and ownership-dependent
counts/displacements and diagnostic sampling ownership. The host difference,
surface and radial plans must also be constructed together: the surface/radial
objects reference their host grid, so moving or mutating the host descriptor
underneath them is not a safe rebuild mechanism.

The implementation must construct a replacement bundle at a synchronization
point, synchronize local-construction failures before entering any new MPI
collectives, then construct the communication plan. It may publish the bundle
only after all ranks agree and current fields/compensation have the new layout.
An unsuccessful rebuild must leave the proposed generation unusable on every
rank. Stage-dependent projected boundary values and derivative samples must be
invalidated even when the new state has the same array shape. Reference geometry
and ownership must come from the same generation as stencil plans.

Required verification includes same-shape geometry and ownership changes, mesh
size changes, rebuild failure on one rank, and fresh-versus-rebuilt full operator
comparisons. For the current continuous-core discretization, unsupported
mixed-level refinement must be explicitly rejected. A complete AMR evolution
would additionally need a supported coarse/fine derivative policy, field and
compensation transfer, and trajectory validation; a plan-bundle test alone does
not establish those features. This section describes remaining implementation
requirements, not completed functionality.
