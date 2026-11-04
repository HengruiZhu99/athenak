# Static puncture coverage without refining the entire core

The new optional `smr_half_widths=h_x,h_y,h_z` input selects an origin-centered rectangular **refinement target**. It replaces `smr_radius`; specifying both is rejected. The values are positive, finite, strictly inside the shell overlap radius, and require nonzero SMR levels. The actual refined region follows the existing octree and 2:1 balance, so it is not generally identical to the requested box. The full input is appended to the checkpoint formulation identity; old spherical-policy identities and checkpoints retain their previous interpretation.

`CartesianAMR::RefineBox` only generates intersecting-block flags. AthenaK's existing `Regrid`, octree balance, redistribution and registered point-value transfers construct the mesh. There is no new transfer algorithm, moving refinement criterion, or change to the spherical-shell interface. Analytic initial data are installed **after** this initial topology construction, as for existing SMR.

For the configured core half-width0.4, four roots per axis and three refinement levels, target half-widths `(0.075,0.025,0.025)` produce400 Cartesian leaves. The finest nominal region is `[-0.1,0.1] × [-0.05,0.05] × [-0.05,0.05]`. The original spherical target has232 leaves and finest region `[-0.05,0.05]^3`. The tube expands coverage in both boost-axis directions at unchanged finest spacing. It is not a resolution refinement of the puncture itself. At N16 its nominal core-cell count is1.7241 times the original, or1.4941 times total nominal core-plus-radial18 cells. These are cell counts, not measured memory or runtime ratios.

The candidate keeps the stationary radial18 geometry, conformal factor, gauge, original scri treatment and floors. Its native boost is0.1. The reference chart displacement `k v (10/3)=0.03372287847` is used only to size the static region. The free initial inner shift is zero, so live gauge motion is measured rather than prescribed by this displacement. No candidate Aurora evolution is submitted until its local checks and the stationary controls are assessed. The ongoing stationary campaign remains bound to its old source and binary.

## Measured coverage

For tracker position `x_p`, the new read-only diagnostic computes the largest coordinate-ball radius entirely inside nominal finest Cartesian volumes:

`R_fine = max(0, min(distance to core boundary, distance to every coarser Cartesian leaf))`.

Distance to a leaf uses Euclidean point-to-axis-aligned-box distance. If the point lies in a coarse leaf or outside the core, `R_fine=0`. Appended evolved overlap and ghost cells do not increase this radius. `h_fine` is the maximum nominal finest cell width. Histories record:

- `finest_level`, `finest_spacing`, `finest_support_radius`, and the configured stencil buffer `buffer_cells*h_fine`;
- `puncture_finest_margin = R_fine - buffer`;
- `candidate_finest_margin = R_fine - buffer - candidate_max_radius`, only for a complete finite positive-radius candidate, otherwise JSON null;
- `core_margin`, which can be negative outside the core.

A nonnegative candidate margin is a conservative enclosing-ball coverage check for a surface centered on the tracker. It does not establish adequate resolution, horizon acceptance, physical excision, or accuracy. Original finder flags and missing masses are preserved. No automatic regridding, timestep/floor change or failure replay is introduced.

## Qualification

A separate local candidate binary preserves the frozen production executable. Tests check one/two-rank topology, distances at the origin, both reference drift directions, near edges, in coarse cells and outside the core; invalid box inputs; unchanged old spherical-policy checkpoint; short exact-boost evolution MPI parity; uninterrupted versus cross-rank restart; and rejection of a different topology identity. See `smr-tube-checks/` for results. The initial N8 test was correctly rejected by the existing transition-width guard and is retained under `smr-tube-checks-rejected-n8-resolution-guard/`; the guard was not weakened. Checks use admitted N12 instead. This is fixed SMR qualification, not dynamic AMR or long-time boosted convergence.
