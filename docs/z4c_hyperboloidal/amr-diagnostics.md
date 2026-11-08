# Accepted-state diagnostics across actual native AMR

The optional `--diagnostics` mode of the existing AMR lifecycle harness primes
the module's diagnostic cache before real refinement and coarsening. It uses
`MeshRefinement::AdaptiveMeshRefinement`, including native ownership migration,
field/history transfer, storage replacement and plan rebuilding. The fixture
refines a Cartesian corner, the first +X shell and the outer +X shell, then
coarsens them: leaf counts are 106, 127, 106 and scri patch counts are 6, 9, 6.

The same test executable passed on two and eight MPI ranks. At the three phases:

| Check | Initial | Refined | Coarsened |
|---|---:|---:|---:|
| Cached diagnostic refresh count | 1 | 2 | 3 |
| Constraint/status/weight scalars compared exactly | 5,276,016 | 6,376,428 | 5,276,016 |
| Owned field/history scalars preserved exactly | 8,979,984 | 10,862,748 | 8,979,984 |
| RK/RHS/status/gap/CFL sentinel values preserved | 174,032,736 | 208,542,372 | 174,032,736 |

All 23 named constraint, primitive, validity and weight values at every evolved
bulk point match a freshly constructed diagnostic instance bitwise. Diagnostic
array extents track the new point lists. A second read at each phase is a cache
hit with unchanged results. Both cached and fresh refreshes leave all 31 owned
fields and eight ChiRecord members, state version, and last actual-stage
compatibility/clock observations unchanged.

The original AMR transfer gates remain active. Complete owned-state snapshots
agree byte for byte across two/eight ranks; 103 unchanged leaves retain all data
through each regrid. Seven of these leaves migrate at two ranks and 30 at eight
ranks. The eight-rank fixture includes three ranks with no owned scri patches
at every phase.

Runtime sources and linked objects remained unchanged throughout these runs.
The executable SHA256 is
`0bdb1f73c351767d24057473f18e4e53197654843680bce970b9014e82aaec43`.
Detailed evidence is `evidence/native-amr-diagnostics-validation.json`; commands
and predeclared assertions are in
`tst/z4c_hyperboloidal/amr_lifecycle_contract.md`.

This gate establishes cache invalidation, geometry/transport plan rebuilding,
and read-only behavior after actual remeshing. Constraint equations are checked
against the predecessor by a separate oracle. It does not claim a FastFlow
search on a remeshed state or supply performance measurements.
