# Hyperboloidal field AMR integration contract

Status: required, not implemented. Chart-tree refinement tests do not establish
field AMR. Dynamic adaptive runs remain rejected until the lifecycle below passes.

## Native lifecycle and storage

`MeshRefinement::AdaptiveMeshRefinement` remains the driver. Its existing tree
update, 2:1 closure, equal-weight load balance, logical-location mappings and
topology sequence are authoritative. In `RedistAndRefineMeshBlocks`, transfer
must capture the old state before local array reordering and before the old
`MeshBlock`/coordinate objects are destroyed. It must install the new local state
after the new pack/maps exist. `InitBoundaryValuesAndPrimitives` then rebuilds
plans and populates ghosts, followed by the module timestep calculation.

Derivative-based refinement criteria need accepted-state halos before
`CheckForRefinement`: the last RK update changes owned fields after the final
stage's halo fill. Refresh these boundaries at the accepted-step boundary when
the selected criterion needs them. After reallocation, rebind the criteria's
shallow `rdata` views; rebinding is not currently part of the generic native
regrid path. Criteria must also account for the module origin `ng+overlap`,
where existing cell-centered criteria assume origin `ng`.

The standard cell-centered AMR calls currently omit this module. Adding it to
`ncc_tosend` alone is incorrect: their fixed nominal bounds and storage origin
do not cover the independently evolved overlap or the separate scri surface.
Native migration transport and cell-centered operators should be extended where
their indexing and interpolation contracts match. Mapped donor transport is
available for transfers that require chart-dependent nodal stencils. Neither
route may replicate global field state or introduce another evolution driver.

The old research `cartesian_amr.hpp` already delegates nominal core transfer to
native registered cell-centered arrays, supplying four raw-field halos for the
native prolongation. Its `amr_evolution.cpp::TransferExtra` copies fixed shells
and unchanged overlap samples, then uses six-point raw-field interpolation for
new overlap samples. That is useful numerical reference for core transfer; it
does not establish shell or scri refinement. Do not import its global snapshot
gathers or separate `Evolution` reconstruction into the native driver. The old
compensation register belongs to its more general formulation path; the pinned
N12 spherical RK branch clears compensation and needs no persistent RK history
across an accepted-step regrid.

All state/work arrays currently have the actual local block count, not the AMR
capacity. A topology change must resize bulk, coarse, surface, RK, reference,
source and record arrays; rebuild evolved-point and block/face mappings; and
invalidate all four RHS partitions and all donor/KO/surface plans. Plan rebuilding
must occur after `MarkMeshUpdated`, including on ranks with no scri patches.

## State that must survive

- Unchanged leaves: copy all 31 evolved bulk fields, including overlap, and
  their eight ChiRecord members exactly, regardless of rank/GID changes.
- Refined/coarsened leaves: transfer the 26 independent fields at the actual
  mapped sample coordinates and reconstruct dependent algebraic components.
  Initialize dormant slots 28–30 to their policy value zero on new leaves;
  unchanged-leaf copies retain all 31 slots. Accepted-state guards inspect all
  slots, so dormant fields must never remain uninitialized.
  These are point values, not finite-volume averages. If reference-subtracted
  interpolation is used, it must use the existing chi normalization and restore
  the target reference. Such a change from the old raw-field transfer needs an
  explicit numerical error budget. An analytic initial state must not replace
  evolved data.
- Outer leaves: transfer the separately evolved scri state in two angular
  dimensions. Only outer radial children own a scri patch. Copy unchanged
  patches exactly, including records; apply the existing surface projection
  after changed-resolution transfer.
- Chi records: define and test history semantics before enabling refinement.
  Counts and accumulated correction must not be multiplied by refinement or
  silently discarded by derefinement. Minima, maxima and first-event time must
  survive. A conservative history redistribution is distinct from interpolating
  physical field values; polynomial interpolation of event counts is invalid.
- Difference ghosts, RHS values and stage accumulators are not accepted state.
  Rebuild them at the accepted-step boundary. No AMR operation may split an RK4
  cycle.

The nodal outer layer inherits radial offset zero in every child. Only the
original layer's inner endpoint has a half-volume weight; an interior radial
child start has full weight. The refined-surface test checks this independently
with the analytic spherical Jacobian, since KO vanishes in that region.

## Required validation before admission

1. Exercise the actual native AMR lifecycle at two and eight local MPI ranks,
   with changed ownership, refine and derefine operations, zero-face ranks, and
   refinement touching both angular seams and the core/shell overlap.
2. Compare complete state/records for unchanged leaves bit for bit. Poison all
   unused storage before exchange; required ghosts must be filled and unused
   padding must not mask absent dependencies.
3. Establish interpolation gates before running them: constants and supported
   polynomial degrees reproduce within an a priori coefficient-norm roundoff
   bound; nonpolynomial smooth data show the expected convergence order on
   refinement/coarsening. Test the first shell's radial refinement separately.
4. Exercise scri refinement/coarsening with four angular children, internal
   radial siblings, and chart seams. Check surface projection, radial jets and
   compatibility, including the existing compatibility bound of 1e-7.
5. Seed nonzero values in every record member. Verify exact count totals and
   history extrema through refinement, migration and derefinement; compare
   distributed and single-rank transfers.
6. Run coupled evolution across multiple actual regrids and a restart on a
   different rank decomposition. Check accepted-state validity, record
   accounting, plan freshness and timestep admission on every new mesh. Include
   a derivative-based refinement criterion to expose stale accepted halos or
   a shallow refinement view left bound to the previous allocation.
7. Retain the fixed N12 trajectory gates and Cartesian AMR/restart regression.
   Passing a transfer-only manufactured test does not prove coupled stability.
