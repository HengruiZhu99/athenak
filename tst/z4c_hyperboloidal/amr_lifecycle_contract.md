# Native field AMR lifecycle checks

These gates precede the first run of `amr_lifecycle.cpp`. The test calls the real
`MeshRefinement::AdaptiveMeshRefinement` with a user refinement criterion. It
refines a Cartesian corner, the first +X shell, and the outer +X shell, then
coarsens their descendants. Native 2:1 closure, ownership, transport, storage
replacement, plan rebuilding and timestep admission all participate. The
initial central SMR region remains. Expected leaf counts are 106, 127, 106;
expected scri patch counts are 6, 9, 6.

The fixture starts in static mode and changes to dynamic mode only when
constructing the test Driver. This deliberately tests the pending AMR path
without admitting unvalidated adaptive evolution through ordinary inputs.
This transfer-only test does not establish coupled evolution or AMR convergence.

The original reference fields are retained except three independent B fields.
B_x is constant, B_y is a chart-coordinate cubic with chart-dependent offsets,
and B_z is a globally smooth Cartesian polynomial. No exact reference field is
substituted by the transfer. B_y distinguishes old-family lineage from the
different values on overlapping charts. All evolved fields must remain finite.

At every phase, every B_x value must reproduce its constant to 512 machine
epsilons. Changed shell nominal B_y values, and first-shell overlap values on
refinement, must reproduce their polynomial. Cartesian B_y checks use the
middle half of the changed parent, where native transfer stencils do not need
cross-chart halo values. The polynomial bound is 256 eps * 32768 * .02:
six-point Lagrange rows within half a sample beyond either endpoint have
one-dimensional absolute coefficient sum below 32, giving a tensor bound 32^3;
.02 bounds the manufactured samples. This is a roundoff allowance for this
specific manufactured field, not a general evolving-state tolerance.

Every old ChiRecord member is seeded with nonzero binary-exact values. Global
additive totals and extrema must survive exactly, separately for bulk and scri.
New points may have explicit default records, but no old event may be lost or
duplicated. The test writes complete 31-field/eight-record snapshots keyed by
logical leaf identity and local evolved index. The runner compares every
unchanged leaf byte for byte, including migrated leaves, and requires at least
one unchanged leaf to migrate on each regrid at both two and eight ranks.
Two/eight-rank canonical snapshots must agree byte for byte.

Unused old bulk storage, residuals, RHS and stage accumulators are poisoned
before each regrid. New module storage is poisoned by the runtime reset. Required
state must therefore be written explicitly. Each regrid must increment the
native topology sequence, rebuild usable plans, retain the requested timestep,
and satisfy surface compatibility <= 1e-7. The eight-rank fixture must include
ranks with zero owned scri patches.
