# Accepted-state composite transfer

`composite_regrid.hpp::RebuildCompositeAtAcceptedState` prepares a new host
tensor engine while leaving the old engine and its fields untouched. It is a
building block for the native transaction documented in
[the owner lifecycle audit](native-dynamic-owner-lifecycle.md), **not an enabled
native dynamic-AMR path**. Native dynamic regridding remains guarded.

The new descriptor must retain logical shape/ghosts, Cartesian half-width,
all shell maps, physics/chart controls and puncture location. It must advance
the geometry revision, satisfy complete nonoverlapping Cartesian coverage and
2:1 face/edge/corner balance, and stay within50000 active cells. Constructor
failure, unsupported transfer rows and bad input leave the source unchanged.
Hierarchies, derivative/halo plans, surface samplers and coefficient caches are
rebuilt. Previous geometry/shift certificates are not transferred.

Each unchanged block is identified by its complete map/level descriptor and
copied bitwise, independent of block ordering. This includes all shell volume
fields and actual scri values. Every changed Cartesian cell receives all22
stored tensor/gauge/flux fields and a separate three-component shift warm start.
Anchored-difference interpolation preserves constants exactly. It does not pin
beta or apply puncture parity, and does not evolve beta as an RK variable.
The caller must recompute the shift against the transferred accepted geometry
before certifying it; an old last-stage warm start is not such a certificate.

For the currently supported puncture chart, positive-storage2 logarithms are
interpolated using all56 Cartesian degree≤5 monomials plus log(rho), with no
prescribed, measured or evolved exponent. Fields0 and18 use independent values
on the same geometric plan. All scalar-log source values have the same global
chart on Cartesian and shell patches. Consequently the default catalog includes
both patches, rather than extrapolating exclusively from Cartesian interior
points near a box corner. The actual value functional must satisfy L1≤8,
the unchanged rank/condition/reproduction checks, and≤256 donors. Rejections
and selected support counts are retained. Where a nearest catalog lacks enough
Cartesian planes, an explicitly counted geometry-only plane support is tried
with the same full basis and guards; there is no degree/field fallback.

Default radius0 applies this scalar-log value chart at every changed Cartesian
cell. A finite radius is a diagnostic control. Restricting it to radius.1 or.24
leaves maximum log transfer error.03804 on this coarse test: a wide one-sided
polynomial stencil can sample the central logarithm even when its query is
outside the chosen ball. This is why query-radius filtering alone was rejected.
Moving the puncture/coordinate chart is unsupported here; transport and native
stage integration must be supplied separately. The function accepts accepted
field payloads only, and does not by itself transfer native RK register history.

## Bounded tests and retained failures

The test refines the Cartesian octant adjoining the outer interface, coarsens
it again, and checks every field/cell. It also evolves free3D Minkowski data and
a nonzero angular lapse pulse before, between and after those transactions.
The strengthened nonzero-shell control completes in17.30s under its60s bound:

| Check | Result |
|---|---:|
| All25-field Cartesian polynomial maximum | 2.60209e-17 |
| All-cell stored-log maximum, powers4/2 and2/1.09 | 1.77636e-14 |
| All-cell positive physical-field maximum | 6.38378e-16 |
| Combined positive-field Penrose-volume RMS | 7.06936e-19 |
| Positive value rows across both power cases | 3456 |
| Rows needing256 nearest donors | 2 |
| Maximum value functional L1 | 7.43789 |
| Evolved Minkowski deviation after transactions | 2.40763e-16 |
| Angular-pulse flux / metric response maxima | 2.47923e-7 / 1.07426e-11 |
| Pulse original-vector RMS maximum | 9.5002e-12 |

The combined RMS sums both positive fields and both manufactured power cases
using each case's exact Penrose-volume weights; it is a transfer test, not an
evolution error estimate. Every active cell is included. Shell/scri values and
constant translational beta are bitwise preserved. No wormhole collapse,
time-dependent puncture position or generic nonspherical regularity qualification
follows from these results.

Earlier failures remain in the `accepted-regrid-*20261027` evidence directories:
the first175-donor cap, radius.24 rank failure, all-core Cartesian-only corner
amplification failure, and the unsuccessful Gaussian-weight widening diagnostic.
The latter was reverted: the production `EnrichedLogValue` API and default
arithmetic are unchanged. Actual global shell donors resolve the demonstrated
corner failure without relaxing L1≤8. Complete logs and final source receipts
are under `evidence/accepted-regrid-build-20261027`.

```sh
cmake -S . -B build-shift-3d
cmake --build build-shift-3d --target elliptic_shift_3d_composite_regrid_tests -j2
ctest --test-dir build-shift-3d -R accepted_regrid -V
```
