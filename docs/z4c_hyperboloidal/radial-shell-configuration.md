# Radial shell regions and independent subdivisions

For six spherical patches surrounding a Cartesian core, specify the two radii
and the number of equal-width base shells. Each patch defaults to one block
in each angular direction. `angular_blocks_2` and `angular_blocks_3` set the
per-face tiling. Each block takes its cell counts from `<meshblock>`.

```ini
<mesh_atlas>
type = cubed_sphere
rmin = .18
rmax = 1
nshell = 10
evolved_overlap = 3

<refined_shell0>
rmin = .918
rmax = 1
level = 2
```

Each base shell whose midpoint lies in `[rmin, rmax)` receives at least the
specified level on all six faces. Overlapping requests use the highest level.
The code grades neighboring base shells to differ by at most one level. In
this example the last two base shells contain two and four radial blocks,
respectively, and the first eight contain one each. Angular resolution stays
fixed. The 2:1 spacing guarantee applies to a uniform base grid.

For the Cartesian core, use `<mesh>` for the domain and root cell counts,
`<meshblock>` for the cells per leaf, and `<mesh_refinement>` with
`refinement = static` plus `<refined_regionN>` boxes with `x1min` through
`x3max` and `level`. For adaptive refinement use `refinement = adaptive`,
`num_levels`, `max_nmb_per_rank`, and the usual `<amr_criterionN>`.
For isotropic refinement of a single shell patch use `<refined_regionN>`
with `chart_id = 1..6` and `coordinate_system = radius_angles`.

Older binary exports can retain their exact nonuniform base edges using
`shell_radii` in place of `nshell` and `rmax`. The archived N12 binary
example uses this form and a single outer refinement request. Legacy
`shell_radial_blocks` lists are also accepted in place of radial requests.
Select a prepared binary export with `problem/binary_file = data/block-0.hbi`.
The importer obtains the identity from its header and checks each block's
identity, GID, point counts, and coordinates.

## Time integration and load balance

With `time/integrator = mrgark4`, the default coarse interval is
`cfl_number * min(dx1, dx2, dx3)` on the Cartesian root grid. An explicit
`time/dt` overrides it. Single-rate RK4 uses the finest initial Cartesian spacing for
its inferred step. Cartesian time levels follow octree depth. On the
shells, radial subdivisions and octree depth set a minimum level. The
initial characteristic speeds, agreement and scri groups, and donor
dependencies can require shorter local steps. Level L takes 2^L substeps.

A fresh run needs no `<multirate>` block. After initializing the fields,
the solver estimates work from the actual stage schedule, occupied points,
mapped halo stencils, overlap agreement and outer surface work. It scores
possible rank maps using batch size, donor traffic and dependencies before
the first interval. METIS contributes candidates when available. The solver
also considers the initial distribution, and later reviews placement against
measured batch costs. Checkpoints record the resolved choices. Shared
checkpoints can recompute automatic placement when the MPI rank count changes.

Stage receives are preposted. Complete local batches coalesce messages and
evaluate the fused RHS after their donors arrive. The former
`prepost_exchanges`, `coalesce_stage_packets`, `wait_for_donors`, and
`fused_rhs` experiment switches can be removed from older inputs. Scheme A
is the sole scri evolution treatment. Existing Scheme A checkpoints use the
same formulation identity and scri output layout.

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
multirate startup checks the characteristic CFL, agreement rate and time-level
compatibility. Long-time stability of a changed mesh or base interval requires
separate evidence.

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
