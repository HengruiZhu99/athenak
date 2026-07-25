# IrisK XCTS initial data import

Configure AthenaK with `-DPROBLEM=z4c_irisk_xcts`. The fixed-SMR import check is
`inputs/z4c/collapse_wave_irisk_xcts.athinput`; the subsequent scale-invariant
critical-collapse evolution is
`inputs/z4c/collapse_wave_irisk_xcts_critical.athinput`. Both read the versioned
`IRISK_ADM_SPEC1` file produced by IrisK's `collapse_wave_xcts` problem
generator.

Build IrisK target `iris_athenak_interpolator` before configuring this problem.
Set AthenaK's `IRISK_ROOT` and, if necessary,
`IRISK_INTERPOLATOR_LIBRARY` CMake cache paths. The integration follows the
TwoPunctures model: AthenaK supplies each local meshblock's x/y/z cell-center
arrays to the linked IrisK library.

For sibling `bbhk` and `athenak` checkouts, run the complete example as:

```sh
# From the IrisK (bbhk) repository root:
cmake --build --preset serial-clang \
  --target iris_app iris_athenak_interpolator -j2
env OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 \
  ./build/serial-clang/src/iris \
  inputs/xcts/collapse_wave_xcts_export.athinput \
  --manifest ../collapse-wave-data/collapse_wave_xcts_A1_N6.manifest.json

# From the sibling AthenaK repository root:
cmake -S . -B build_irisk_xcts -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=z4c_irisk_xcts
cmake --build build_irisk_xcts -j2
env OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 \
  ./build_irisk_xcts/src/athena \
  -i inputs/z4c/collapse_wave_irisk_xcts_critical.athinput \
  -d ../collapse-wave-data/athenak-critical-A1
```

The two example inputs meet at
`../collapse-wave-data/collapse_wave_xcts_A1_N6.adm_spectral`. IrisK resolves
its relative `export_path` against its launch directory. AthenaK resolves
`problem.irisk_adm_spectral_file` against its own launch directory before
applying `-d`. Both programs print the absolute normalized path they actually
wrote or opened. For non-sibling checkouts, override both input keys with the
same absolute path.

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

The critical-collapse input then evolves those fields with no 1+log or
slow-start lapse contribution. Its telegraph lapse uses
`max|K|/telegraph_tau` for relaxation and
`telegraph_kappa/telegraph_tau` for gradient forcing. Shift damping and the
optional Z4c `kappa1` damping use the same global curvature scale. The example
sets constraint damping to zero and uses the dimensionless two-cell `dchi`
indicator for AMR. AthenaK requires a finite level budget; the example reserves
physical levels 0 through 11.

Z4c history includes the MPI-global `max_abs_K`. The critical-collapse example
also opts into `history_kretschmann=true`, which adds the MPI-global
`maxKretsch` column using the four-ghost, sixth-order vacuum-curvature
diagnostic. This full-volume reduction is disabled by default and requires
`spatial_order=6`.

Dimensionless constraint monitors can be formed as
`sqrt(H-norm2/Volume)/max_abs_K^2`,
`sqrt(M-norm2/Volume)/max_abs_K^2`, and
`sqrt(Theta-norm2/Volume)/max_abs_K`, guarding the initial `max_abs_K=0` case.

For collapse evolutions without puncture trackers, FastFlow can follow the
global active-cell lapse minimum by setting
`fastflow.use_minimum_lapse_center_N = true` for horizon `N`. This option is
mutually exclusive with `use_puncture_N` and
`use_puncture_massweighted_center_N`.

FastFlow opens its summary, verbose, grid, and harmonic output lazily on the
first accepted search, after AthenaK has entered the directory selected by
`-d`. All horizon artifacts therefore share the same run directory as the
history, restart, and slice outputs.

The critical-collapse example enables two accepted-step stopping conditions.
A confirmed FastFlow surface terminates the collapse side immediately after
scheduled output. The dispersive side is checked every eight cycles after
`t=20`: both MPI-global `maxKretsch` and `max_abs_K` must fall below five
percent of their run-wide peaks, while the maxima over the second half of a
16-sample window must be no more than half those over the first half.
Constraint and curvature diagnostics must remain finite. These conservative
conditions survive restart through persisted peak values; the full observation
window is deliberately rebuilt after restart. Cases that satisfy neither
condition continue to the ordinary time or MeshBlock limit.

The IrisK pgen already supplies all data needed here. The separate experimental
native CTS/multigrid implementation in AthenaK is not used by this import path.
