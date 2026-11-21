# Native hyperboloidal refinement criteria

The native `amr_criterion*` blocks accept `hyperboloidal_` followed by the exact
stored field name from `Fields::names`: for example `hyperboloidal_chi`,
`hyperboloidal_P+3` or `hyperboloidal_beta_x+x`. Offsets are part of the quantity
being tested. Thresholds for an offset field are not thresholds for its physical
unshifted value.

Min/max, slope and second-derivative criteria scan the nominal cells of each
native MeshBlock. They account for the module's `ng+overlap` storage origin;
independently evolved overlap cells do not add another criterion domain. Before
a scheduled derivative criterion or user callback, the original boundary
initializer refreshes the accepted state. A topology change also refreshes the
old accepted state before transfer, if the criterion did not already do so.
After installing new storage, the native lifecycle marks the mesh updated,
rebinds the criterion views and rebuilds all communication plans before evolution.

The slope indicator is
`sqrt(sum_axis (q[+1]-q[-1])^2)/(2*abs(q))`. The second-derivative indicator is
`abs(sum_axis (q[+1]-2*q+q[-1]))/abs(q)`. These are normalized computational-index
differences, consistent with the native criterion's purpose; they are not
physical-coordinate derivatives. For a zero denominator, the indicator is zero
when the numerator is zero and `FLT_MAX` otherwise. This absolute denominator
and explicit zero policy applies only to hyperboloidal fields, which include
signed offsets. Existing Cartesian criteria retain their formulas.

On an outer block, the last nominal radial sample's positive radial neighbor is
the owned, current `s0` surface value. The bulk array has no evolved endpoint or
valid exterior ghost there. Angular and all other radial neighbors use refreshed
bulk values.

`amr_criteria_registry.py` tests nominal-origin indexing with misleading padding,
rebinding after replacing the bulk allocation, positive/negative/zero derivative
behavior, the surface neighbor with poisoned exterior padding, and collective
registration shape/schema validation and buffer reuse. The test does not evolve
or perform a regrid. Coupled AMR admission requires the separate lifecycle and
transfer gates in `amr-transfer-contract.md`.

## Binary Cartesian chi refinement

For a hyperboloidal binary, set `mesh_refinement/refinement = adaptive` and
`amr_criterion0/method = user`. The callback tests the minimum physical chi on
each eligible Cartesian block inside the regular core. A block requests
refinement below `z4c_amr/chi_min` (default 0.2) and coarsening above 1.25
times that threshold. Set `z4c_amr/max_ref_lev` relative to the Cartesian
root. Sibling and neighbor checks still apply to coarsening.

Set `z4c_amr/horizon_padding_cells` to a nonnegative value to protect both
horizons at the finest allowed level. The code centers the last accepted
FastFlow radius bound on each current puncture position and adds the specified
number of finest Cartesian cells. Before the first accepted search it uses the
configured initial radius. Restart files retain the accepted radius bounds.
The default value of -1 disables this option.

With `z4c_amr/allow_coarsening = true`, the first scheduled check can trim the
seed grid. Subsequent coarsening requires a check whose coarse cycle is a
multiple of `z4c_amr/coarsen_interval` (default 1). Use a multiple of
`mesh_refinement/ncycle_check` for a predictable cadence. Refinement requests
are evaluated at every scheduled check.
