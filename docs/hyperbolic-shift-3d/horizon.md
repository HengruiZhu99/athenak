# Physical fast-flow horizons on hyperboloidal data

The existing `FastFlow` expansion, angular harmonics, flow update, surface area,
and spin kernels can now consume an external geometry sampler. This removes
their dependence on `MeshBlockPack` interpolation when used with the composite
Cartesian/shell research evolution. The ordinary constructor and ADM sampling
remain available. Both paths now integrate every Gauss–Legendre angle; two
exclusive-end `RangePolicy` bounds previously omitted the last point.

For the evolved twice-conformal variables, the physical tensors are

\[
\gamma_{ij}=\frac{\widetilde\gamma_{ij}}{\chi\Omega^2},\qquad
K_{ij}=\frac{\widetilde A_{ij}}{\chi\Omega}
       +\frac13\gamma_{ij}(P+2\Theta),
\]

\[
\partial_a\gamma_{ij}=\frac{1}{\chi\Omega^2}
\left[\partial_a\widetilde\gamma_{ij}
-\widetilde\gamma_{ij}\left(\frac{\partial_a\chi}{\chi}
+2\frac{\partial_a\Omega}{\Omega}\right)\right].
\]

Here `P` is the full physical `K-2Theta`, not its deviation from the CMC
reference. `PhysicalHorizonGeometry` in
`src/z4c/hyperboloidal/horizon_geometry.hpp` implements this conversion. It
does not use the lapse, the shift, or the normal derivative of Omega. It
rejects nonpositive Omega/chi, invalid physical metrics, and nonfinite jets;
there is no floor or evaluation at scri. The reconstruction applies to finite
interior apparent horizons on hyperboloidal slices. At Omega=1 with zero Omega
gradient it reduces to ordinary Z4c ADM reconstruction.

## Sampler integration

```cpp
FastFlow finder(nullptr, &pin, 0,
  [&](const Real x[3], z4c::HorizonGeometry<Real> &physical) {
    if (!owns_finest_active_patch_point(x)) return false;
    physical = z4c::hyperboloidal::PhysicalHorizonGeometry(jet_at(x), omega_at(x));
    return true;
  });
finder.Find(step, time);
const auto result = finder.GetResult();
```

The callback runs on the host. It supplies the physical tensors and first
metric derivatives in a common Cartesian basis; interpolate the evolved
fields and their Cartesian first derivatives before reconstruction. Return
`false` only when this rank does not own the point. Return `true` with
`physical.valid=false` for an owned invalid point. Avoid rank-local exceptions
inside the callback. All ranks must call `Find` in the same order. FastFlow
collectively checks that exactly one rank owns every quadrature point and that
all owned jets are valid; a missing point or overlapping ownership fails the
search. It retains its existing MPI sums of integrals and harmonic updates.

The sampler must resolve active patches, shell seams, coarse/fine overlap and
AMR ownership itself. It must not return duplicate interpolation contributions.
Stage the required grid data once per horizon search on a GPU backend, rather
than copying the full grid for each sample. No `MetricDerivatives` call is
needed in sampler mode. The flow and surface-integration kernels remain on the
Kokkos execution space.

Configure `fastflow/start_time_0`, `stop_time_0`, `initial_radius_0`, and
`center_x_0`, `center_y_0`, `center_z_0`. The public `center` and `initial_radius`
also allow a caller to supply a moving search center without a MeshBlockPack
puncture tracker. Use `1 <= lmax < ntheta`. Sampler mode additionally requires
positive `expansion_rms_tol_0` (default `1e-8`) as well as `mass_tol_0`:
mass stagnation alone cannot declare a horizon. The residual is the physical
outgoing-expansion RMS, so set its dimensional tolerance consistently with the
simulation mass scale. The existing mesh mode retains its mass-only default.

`GetResult()` exposes the found flag, complete-sampling flag, iteration count,
physical area, areal radius, irreducible and Christodoulou masses, coordinate
mean/minimum radius, and physical expansion RMS/mean. Failed sampler searches
clear the previous horizon properties to NaN. The historical summary-file
column named `hrms` stores the mean square; `GetResult().expansion_rms` takes
its square root explicitly. A failed finder is an unavailable measurement,
not evidence by itself that no horizon exists.
`last_expansion_rms` and `last_mean_radius` report the most recently sampled
trial surface even if convergence fails; these are not horizon measurements.

Initialize `global_variable::my_rank` and `global_variable::nranks` after
`MPI_Init` and before construction. The communicator is `MPI_COMM_WORLD`.
The constructor opens `<job/basename>.horizon_summary_0.txt` on `mpi_root`
(default 0). Give each simulation its own basename, such as its output prefix
plus `.fastflow`. Calling `Write()` also writes `.horizon_shape_0.txt` on
successful searches. Optional verbose/grid/harmonic outputs default to false.

## Tests and scope

`tst/hyperbolic_shift_3d/test_horizon.cpp` uses the real FastFlow kernels:

- An exact CMC Schwarzschild slice with `K=-3`, `M=1`, `C=16`, compactified
  as `R=r/(1-r)`. Its radial physical metric is `gamma_RR=1/F` with
  `F=1-2M/R+(-R+C/R^2)^2`, and its extrinsic-curvature eigenvalues are
  `K^R_R=-1-2C/R^3`, `K^theta_theta=-1+C/R^3`. The future horizon has `R=2M`.
  Direct physical and hyperboloidal samplers recover the same area and mass.
- The same black hole displaced from the search center, requiring angular
  harmonics rather than only the spherical mode.
- An isotropic time-symmetric wormhole with `M=1/3`, nonconstant Omega, and
  exact coordinate radius `r=M/2` and areal radius `R=2M`.
- A Minkowski hyperboloid with no horizon; an intentionally permissive mass
  tolerance and tiny flow step demonstrate that the expansion residual
  prevents false acceptance of a stalled search.
- Omega=1 reduction, spatial Omega-gradient and `2Theta` reconstruction,
  invalid/scri geometry, missing quadrature points, and duplicate MPI ownership
  (when run with at least two ranks).

Build target `hyperbolic_shift_3d_horizon`; CTest names are
`hyperbolic_shift_3d_horizon` and, with MPI enabled,
`hyperbolic_shift_3d_horizon_mpi`. The reusable `hyperboloidal_fastflow` library
is declared in `tst/hyperbolic_shift_3d/horizon.cmake`.

The serial and two-rank MPI analytic tests passed locally. The CMC Schwarzschild
physical and hyperboloidal samplers each converged in 281 iterations with
expansion RMS `3.398e-10`; the displaced case converged in 277 iterations with
RMS `3.488e-10`. The isotropic wormhole converged in 22 iterations with RMS
`8.414e-10`. Raw results are in `reproduction/horizon/analytic-*.txt`.

## Actual composite puncture initial data

The optional `hyperbolic_shift_3d_horizon_grid` program reads a previous native
`initial.txt` without evolving or projecting it. It constructs the same
Cartesian/shell grid, fills interpolation halos once, and supplies
`SampleJets(..., use_halos=true)` to the physical FastFlow sampler. For example:

```sh
hyperbolic_shift_3d_horizon_grid 16 \
  output/hyperbolic-core-shell/puncture-v3/n16/initial.txt \
  /tmp/horizon-n16 1e-3 6 12 500 .07 .1
```

The trailing arguments are physical expansion tolerance, `lmax`, `ntheta`,
maximum iterations, initial radius, and flow coefficient. The probe refuses
to overwrite an existing `.result.txt`. This diagnostic replicates the input
on each rank and partitions angular samples by direction; the evolution
backend must use its actual block ownership instead.

On the preserved `puncture-v3` initial data with `M=1/3`, core half-width
`0.125`, `lmax=6`, `ntheta=12`, initial radius `0.07`, flow coefficient `0.1`
and mass tolerance `1e-10`, the observed initial-data results are:

| Core cells per side | Physical expansion tolerance | Iterations | Measured expansion RMS | Horizon mass | Mean coordinate radius |
| --- | --- | --- | --- | --- | --- |
| 8 | `5e-3` | 67 | `2.92203e-3` | `0.3333771840` | `0.0623481563` |
| 12 | `2e-3` | 62 | `1.14032e-3` | `0.3333392496` | `0.0628618513` |
| 16 | `1e-3` | 61 | `4.70786e-4` | `0.3333353261` | `0.0629189382` |

The reference coordinate horizon is approximately `0.06292449`; the exact
areal radius is `2/3`. Relative horizon-mass errors are respectively
`1.316e-4`, `1.775e-5`, and `5.978e-6`. These are initial-data finder tests,
not evolution-convergence results.

Requiring RMS `1e-8` correctly leaves all three searches unconverged after 500
iterations, with residuals `2.92221e-3`, `1.13744e-3`, and `4.62548e-4`.
At n=16, doubling the angular representation to `lmax=12`, `ntheta=24` reduces
the residual only to `3.25925e-4`. Thus interpolated coarse-grid geometry limits
acceptance well above the analytic-test tolerance. The table's tolerances
permit coarse initial monitoring while retaining a measured expansion check;
they are not a claim of roundoff-accurate horizons and must be reassessed as
the evolved geometry changes. Raw successful and failed searches, and SHA256
identities of their inputs, are retained in `reproduction/horizon/`.

The MPI runner records failed horizon searches separately from volume failures.
It continues full-domain diagnostics when a horizon is unresolved, with null
exterior norms and null found-horizon quantities; the last trial radius and
expansion remain explicit diagnostic fields. Every report timestamps its
constraint snapshot and states whether it matches the checkpoint time. The
exterior comparison currently uses a coordinate sphere at the fast-flow mean
radius and is intended for the spherical-data campaign, not general distorted
surfaces. Fast-flow summaries and harmonic shape coefficients are written under
the run's `.fastflow` prefix.

An optional `.checkpoint.horizon-seed.txt` contains a previous measured radius
as a restart search hint, bound to the checkpoint time and step count. It does
not change the evolution state; absent or invalid hints fall back to radius
`0.07`. Copy and bind this small file alongside the checkpoint when reusing it.
The report includes the actual seed and whether it was loaded. An n=8 initial
snapshot restarted from its hint needed 16 iterations instead of 67; its radius
changed slightly within the deliberately finite coarse-grid tolerance.

Local two-rank main-path checks, saved in `reproduction/horizon/mpi-main/`, covered
one Minkowski step (`H_L2=1.081302e-15`), a successful puncture search, a strict
unresolved search, seed restart, collective walltime checkpointing, and a
deliberately rejected oversized timestep. A root-only invalid output directory
also produced a coordinated nonzero exit without hanging. The rejected
timestep report marks its checkpoint unadmitted and unmeasured diagnostics
null. These checks used the host-scri evolution class before the subsequent
full-device scri rewrite; they establish the diagnostics and control-flow
behavior, not GPU parity or long-time stability of the later implementation.

The subsequent full-device bulk/scri class was separately rebuilt and checked
locally with Kokkos Serial. All four checkpoint arrays (volume, volume RK
compensation, scri, scri RK compensation) were bitwise identical to the earlier
CPU two-step pulse output, between 1/2/12 MPI ranks, and between four direct
steps and two steps followed by a two-step restart. The checkpoint header
remained 72 bytes. The scri component test passed background and nonspherical
perturbed geometry, partitioned evaluation, and dependent-auxiliary input/order
independence. A one-step n=8 puncture run retained its measured horizon
(`M_AH=0.333376822694`, expansion RMS `2.93124e-3` with tolerance `5e-3`) and
null errors below `9e-16`. A one-step Minkowski run had `H_L2=1.081302e-15`
and maximum state deviation `1.148e-16`. These short Serial-backend checks are
saved in `reproduction/mpi-full-device-local-v2/`; actual GPU parity and
long-time stability require the Aurora campaign.

For local postprocessing, copy the admitted binary checkpoint and its small
`.report.json`, worker-status and source-manifest evidence files from Aurora.
The independent `tst/hyperbolic_shift_3d/checkpoint_export.py` validates the
72-byte header, fixed composite layout, admission, payload length and all four
arrays' finiteness. It can verify the worker artifact/source hashes, export a
fresh directory compatible with `compare_coupled.py`, and run the existing
local diagnostic executable without evolving anything:

```sh
python3 tst/hyperbolic_shift_3d/checkpoint_export.py RUN.checkpoint LOCAL_EXPORT \
  --expected-time 0.0002 --reanalyze /path/to/hyperbolic_shift_3d_reanalyze \
  --drop-intermediates
```

Add `--worker-status worker-status.json --case NAME --source-manifest sources.json`
when the bound execution evidence is available. The exporter records its own
hash separately from the frozen numerical source. `--drop-intermediates`
removes only the exported state/compensation CSVs after successful reanalysis;
it preserves the original checkpoint and the locally generated diagnostics,
rays and metadata. Large CSVs need not occupy Aurora's home storage. Export
completion is distinguished from source campaign completion through
`source_status` and `source_target_reached`, including walltime checkpoints.
The local audit confirms bitwise equality of all four exported arrays and
reanalyzed diagnostics/rays with the known CPU output, plus rejection of
malformed/nonfinite/unadmitted inputs; its compact JSON is retained alongside
the other local checks.

In these comparisons, `M_r` is a coordinate radial momentum component, whereas
`M_norm` and MPI `M_L2` use the Penrose-metric invariant norm. The physical norm
is `Omega*M_norm`, as used by `M_physical_L2`. Even for spherical data,
`M_norm=sqrt(chi/g_rr)*abs(M_r)`; these quantities must not be interchanged when
comparing the 1D and 3D constraint errors. All current L2 measures use normalized
compactified Cartesian coordinate volume.

These are analytic geometry and finder tests. They do not establish stability
of the coupled evolution, interpolation convergence near the puncture, AMR
transfer correctness, or accuracy after a horizon crosses a patch seam.
FastFlow still represents a star-shaped surface about a chosen center, and
uses coordinate rotational vectors for its spin estimate. Hyperboloidal
checkpoint/tracker integration is a caller responsibility.
