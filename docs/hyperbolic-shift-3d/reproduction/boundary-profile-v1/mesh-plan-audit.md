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
actual boundary application from waiting. Those results are still pending.
Per-stage sparse communication is already small in the V21/V22 profiles, so an
asynchronous exchange is not automatically worthwhile. Any overlap experiment
must preserve producer/consumer and collective ordering, drain pending requests
on failure, and compare complete checkpoints and failure captures.
