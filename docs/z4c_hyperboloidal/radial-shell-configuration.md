# Radial shell regions and independent subdivisions

`mesh_atlas/shell_radii` specifies strictly increasing physical region ends.
The optional `shell_radial_blocks` specifies a positive integer count for each
region. Counts default to one. Each region is subdivided uniformly in radius;
region widths and counts may differ, and counts need not be powers of two.
Angular resolution and per-block cell counts remain unchanged.

For the N12 binary mesh, subdivide the penultimate region twice and the final
region four times with:

```ini
<mesh_atlas>
type = cubed_sphere
spherical_inner_radius = .18
shell_radii = .24,.32,.40,.48,.56,.64,.72,.80,.88,.94,1
shell_radial_blocks = 1,1,1,1,1,1,1,1,1,2,4
evolved_overlap = 3
outer_endpoint_grid = true
```

This adds 24 blocks across the six spherical faces. The radial spacing in
the last region changes from `.06/12 = .005` to `.06/(4*12) = .00125`,
without multiplying the angular cell count. Additional static refinement
still uses `coordinate_system = radius_angles` and physical radial bounds;
it acts on the subdivided native root blocks.

## Region conventions

Subdividing a region differs from replacing it with several entries in
`shell_radii`. The region keeps its identity, its first-region overlap rules,
and its outer-region node placement. With `outer_endpoint_grid = true`,
**all radial blocks of the final region** use the inherited nodal grid;
only the outermost leaf owns scri. Internal radial starts have full quadrature
weight. The half weight remains at the physical inner edge of the nodal region.
Evolved radial overlap is attached to the original first-region boundary,
not each new internal block boundary. Subdividing that region reduces the
physical width represented by a fixed number of overlap cells.

Omitting counts, or explicitly supplying all ones, preserves the previous
atlas identity and coordinates. Nontrivial counts are recorded in the atlas
v3 identity as `radial_blocks=...`; restart geometry checks therefore reject a
different subdivision. The native block tree, mapped donors, and surface
operators remain authoritative; no second topology is introduced.

## Initial data, timestep, and validation limits

Binary `.hbi` files bind both GIDs and sampling coordinates. A changed mesh
needs an initial-data export at its actual coordinates; changing an input
file alone does not make an existing archive reusable. Likewise, an evolved
checkpoint cannot be restarted onto a different atlas without an explicit
transfer procedure.

Radial subdivision increases derivative and boundary-closure scales. The
hyperboloidal formulation uses the requested fixed `time/dt`; changing
`cfl_number` does not automatically make a subdivided mesh stable. A
characteristic CFL estimate is not a stability bound for the complete scri
closure, penalties, and source terms. Retain the bulk equations and diagnose
these contributions before selecting a production timestep.

The November 17 binary attempt with manually inserted shell boundaries passed
16-step startup/restart equivalence but aborted after the last tracker output
at cycle 34 (`t/M0 = .0478831`). Its reported characteristic CFL was about
0.918 at step 16. The exception combined invalid restored state/RHS and
normal compatibility exceeding `1e-7`; it did not identify which triggered.
The exception now reports both the invalid count and compatibility value,
without changing the acceptance threshold. These configuration changes do
not by themselves establish that this evolution failure is fixed.

Regression commands (from the repository root):

```sh
cmake -S . -B build -DAthena_ENABLE_MPI=ON -DAthena_ENABLE_HYPERBOLOIDAL_TESTS=ON
cmake --build build --target athena hyperboloidal_topology_tests -j 8
ctest --test-dir build -R '^hyperboloidal_topology$' --output-on-failure
python3 tst/z4c_hyperboloidal/radial_blocks.py --build build --output build/radial-check --ranks 2 --inner-blocks 3
```

The MPI regression checks degree-zero through degree-five radial polynomials
on both sides of unequal region/block interfaces, distributed scri values,
gradients and Hessians, unchanged angular surface counts, and malformed input
rejection. These are geometry and operator-consistency checks, not a long
binary evolution. Legacy checkpoint scripts that assume one root block per
region now explicitly reject subdivided atlases; native scri archives retain
the actual maps and derivative rows.

Validation on November 17, 2026: the full native executable built, topology
tests passed, and the two-rank MPI test passed with 148 blocks, 756 radial
polynomial checks, 864 scri points, and all bulk halo dependencies constructed.
The existing two-rank `surface_smr.py` regression also passed: 4,228,848
operator comparisons, zero failures, and bitwise cache refresh after poisoning
and rebuilding. Existing input and new radial subdivision paths are both
covered; no long-time binary stability claim is made.
