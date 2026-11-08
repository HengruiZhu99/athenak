# Hyperboloidal field AMR integration contract

Status: implemented and admitted after local CPU MPI validation. Actual native
refine/coarsen lifecycle and coupled evolution pass at two and eight ranks;
ordinary adaptive input and a repartitioned restart also pass. See
`evidence/native-amr-validation-index.json` for scope and provenance. These
tests do not establish GPU behavior or long-time AMR accuracy.

## Implemented transfer and history semantics

`RemeshState` captures transfer results before the native mesh mutates. It owns
only local fields and global leaf metadata. For nominal Cartesian core points,
temporary arrays with the ordinary `ng` origin are registered with native
cell-centered restriction, migration and prolongation. Their full tensor halos
are populated from old raw evolved data using `MappedExchange`; the evolution
halo's axis and mixed-derivative stencil alone is insufficient for these tensor
operators. These arrays are scratch space: their shell results are discarded.

Every unchanged leaf copies all 31 fields and all eight ledger members directly,
including evolved overlap. A changed overlap or shell point first selects an
old member of its native replacement family (the old parent on refinement, or
old siblings on coarsening). A nominal containing leaf takes precedence over
overlap; otherwise the largest logical margin wins, followed by old GID. Its
six-point tensor stencil is centered where possible and clamped one-sided to
the old evolved bounds. The covered logical cell extent permits within-cell
endpoint extrapolation, as required when fine centers extend beyond coarse
centers. Only coarse overlap outside the entire old fine-family extent uses
the existing physical-space mapped selector. Thus independently evolved first
shell values retain their own lineage wherever that old family supplies data.
Changed scri points similarly use their old outer angular family. These are
raw-field transfers; no analytic initial state replaces evolved data.

Changed points retain the 26 independent fields, reconstruct slots 6 and 12,
and set dormant slots 28–30 to zero. Unchanged values bypass that reconstruction
inside the transfer. Normal native boundary initialization runs after the mesh
sequence advances and rebuilds all stage plans. The transfer does not add a
floor event or silently repair an interpolated state. Nonfinite state or
nonpositive bulk chi/physical lapse is rejected collectively.

The read-only predecessor's `tst/hyperbolic_shift_3d/amr_evolution.cpp`,
`Validate()` at lines 126–149 and `Install()` beginning at line 152, also rejects
nonpositive interpolated chi/lapse and invalid metric geometry before upload.
Its `TransferExtra()` at lines 360–438 performs raw interpolation without a
remesh floor; `coupled_mpi.cpp::Upload()` at line 816 copies/materializes and
exchanges that state. Thus this positivity rejection preserves the old AMR
admission policy. Positive values below the configured floor are not clamped
by remeshing; the ordinary subsequent RK candidate applies the evolution
floor policy. Raw interpolation near the puncture can still be rejected, which
requires evolved-data validation and is not silently converted into a floor
event. The older external historical-event ledger is distinct from the native
packet redistribution defined here.

Each old ChiRecord is an indivisible inherited-history packet. It is routed
exactly once to a replacement-family point. The candidate index is nearest in
the common chart coordinates, clamped to the new evolved bounds; exact
half-index ties choose the lower index. Across candidate leaves, squared
distances normalized by the old leaf's chart spans choose the target, with
new-GID/k/j/i ties. Scri uses the same rule in the two angular coordinates and
only outer radial children. Old overlap which no longer exists is clamped to
the nearest surviving family sample rather than discarded.

Packets are folded in old-GID/k/j/i order independently of MPI ownership.
Counts and accumulated correction are summed; minima, accepted minima and
first-event time take minima; maximum correction takes the maximum. Counts
are never polynomially interpolated, fractionally split, or duplicated.
Floating correction sums retain ordinary floating-point summation error;
integer event counts remain exact within the representable integer range.
Points receiving no packet get explicit default ChiRecords. The resulting
ledger describes inherited history, not an interpolation of the new physical
field or an assertion that every new point experienced every old event.

The coordinated build and lifecycle/interpolation tests, rather than this
implementation description, determine admission. No AMR correctness or coupled
stability result is claimed merely by adding these paths.

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
shallow `rdata` views; the hyperboloidal post-regrid hook performs this rebinding.
Criteria must also account for the module origin `ng+overlap`,
where existing cell-centered criteria assume origin `ng`.

Adding this module to standard cell-centered `ncc_tosend` alone is incorrect:
the native operators' fixed nominal bounds and storage origin
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

All persistent module state/work arrays have the actual local block count, not the AMR
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
