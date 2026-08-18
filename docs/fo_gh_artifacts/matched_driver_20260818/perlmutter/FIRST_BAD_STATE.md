# Coarse old-control first-bad-state replay

This is a diagnostic replay of the inherited formulation, not a run of a new
matched-driver production system.

- Parent source: `e0d8c653d30d41a676467c23e02f4969f7629156`
- Allocation/node: `57213683`, `nid001048`
- Runtime: four MPI ranks on four distinct A100 UUIDs
- Grid: 232 MeshBlocks, physical levels 0--3, `dx_min=1/16M`
- Last periodic history record: `t=3.400608334264673M`
- First invalid state: cycle 1657, `t=3.4274776869830532M`
- Failure cells: four symmetry-related cells at coordinate magnitudes
  `0.21875M`, radius `0.378886114155692M`, physical level 3
- Nearest finest-grid boundary: cube face at coordinate magnitude `2M`,
  coordinate distance `1.78125M`
- Representative lapse: `alpha=-0.14805374010543476`
- Representative conformal metric: determinant `1.2097452854436379`,
  eigenvalues `0.166853485115589`, `2.683040014724675`,
  `2.702287212989337`
- Largest reported state/RHS: `Atzz=2.7193503378429919e4`, last-stage
  `rhs(Atzz)=2.9642533896420613e7`
- Normal gauge residual: `-3.0429968689637848e4`

The event is localized to the near-puncture interior, not an SMR interface or
the outer boundary. This is localization, not a causal attribution.

The fail-closed hook skipped pointwise constraint evaluation once the lapse was
invalid. It also did not record global minima or numerical `d_i a_j`; those
requested quantities are unavailable and are not inferred. The `.hst` files
retain the last valid integrated diagnostic sample. See `coarse_first_bad.log`
for all four reports and component values.
