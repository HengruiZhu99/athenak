# IrisK XCTS initial data import

Configure AthenaK with `-DPROBLEM=z4c_irisk_xcts` and run
`inputs/z4c/collapse_wave_irisk_xcts.athinput`. The problem generator reads the
versioned `IRISK_ADM_SPEC1` file produced by IrisK's `collapse_wave_xcts`
problem generator.

Build IrisK target `iris_athenak_interpolator` before configuring this problem.
Set AthenaK's `IRISK_ROOT` and, if necessary,
`IRISK_INTERPOLATOR_LIBRARY` CMake cache paths. The integration follows the
TwoPunctures model: AthenaK supplies each local meshblock's x/y/z cell-center
arrays to the linked IrisK library.

The export retains IrisK's analytic block maps and converged LGL nodal fields.
IrisK reconciles the two weak-DG traces at every conforming face node before
writing; shared edge and corner nodes are averaged through the same equivalence
classes. The library can also write a reconciled copy of an existing version-1
file without modifying its input.
The library performs map inversion and tensor-product spectral interpolation at
every AthenaK active and outer-ghost cell center. The AthenaK root grid and
static-refinement structure are arbitrary and need not be known at export time.
Import fails closed when any requested point lies outside IrisK's outer sphere.

The loader fills ADM `gamma_ij`, `K_ij`, `psi^4`, lapse, and shift on all cells,
then follows the established Z4c problem-generator sequence:

1. `ADMToZ4c<ng>` for the configured stencil width;
2. `Z4cToADM` as a consistency round trip;
3. `ADMConstraints<ng>` and optional RMS threshold/report output.

The supported finite-difference/ghost combinations are:

| `z4c.spatial_order` | Required `mesh.nghost` on SMR |
|---:|---:|
| 2 | 2 |
| 4 | 4 |
| 6 | 4 |

Although the fourth-order stencil needs only three points algebraically,
AthenaK requires an even ghost count when SMR/AMR is enabled.

Set `problem.xy_plane_output` or `problem.xz_plane_output` to write exact
coordinate slices after ADM-to-Z4c conversion. The files contain `psi`,
`alpha`, physical `|beta|`, `H`, `|M|`, `C`, `Z`, refinement level, and block
provenance. An eight-point normal interpolant prevents the slice coordinate
from changing when MeshBlock resolution changes.

For convergence with a fixed physical SMR hierarchy, keep the root/refinement
regions unchanged and increase both `mesh.nx*` and `meshblock.nx*` together.
This holds the 15-block topology in the example fixed while increasing the
cells per MeshBlock.

Unlike puncture initial data, the loader does not apply
`GaugePreCollapsedLapse`: the elliptically solved XCTS lapse and shift are part
of the constrained data and are retained.

For collapse evolutions without puncture trackers, FastFlow can follow the
global active-cell lapse minimum by setting
`fastflow.use_minimum_lapse_center_N = true` for horizon `N`. This option is
mutually exclusive with `use_puncture_N` and
`use_puncture_massweighted_center_N`.

The IrisK pgen already supplies all data needed here. The separate experimental
native CTS/multigrid implementation in AthenaK is not used by this import path.
