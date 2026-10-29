# Thin-overlap prototype: preliminary evidence

Experimental branch `codex/hyperbolic-thin-overlap`, based on
`e48a8c43f` of `codex/hyperbolic-core-shell`. The V11 default remains a control.
See [design and derivation](DESIGN.md) and the [user specification](request.txt).

Implemented: separate Cartesian extension count; an explicit shifted
six-point donor fallback; paired weighted-adjoint penalties on common value
and normal-derivative traces; batched cached device application; two-interface
diagnostics; an isolated penalty timestep bound; and configuration-bound V12
restart. The volume remains cell centered, avoiding a new puncture node at
r=0. Spherical shells remain spherical. Actual VC MeshBoundaryValues classes
are inspected infrastructure, not wholesale dependencies of this prototype.

## Preliminary results

| Check | Result |
|---|---|
| Centered donors, n12, one/two layers | Rejected: missing donor coverage |
| Centered donors, n16, one layer | Rejected: missing donor coverage |
| Centered donors, n16, two/three layers | Construct successfully |
| Explicit shifted donors, one layer, n12/16/24 | Construct successfully, no outside-hull extrapolation |
| Positive-weight penalty energy identity | Relative discrepancy <2e-15 in tested random data |
| Constant trace mismatch | <1.2e-14 |
| CPU/device physical penalty RHS | Maximum difference 5.42e-19 |
| Partitioned trace norms | Equal to whole-domain norms in the fixture |
| Minkowski, 1/2/3 layers, kappa=0/1 | Four steps preserve state to 3.94e-15, no floor interventions |
| Three layers, zero penalty, shifted donors off | V11 checkpoint payload bitwise identical |
| Nonspherical short evolution | One/two ranks and restarted checkpoint bitwise identical |
| Restart with different penalty strength | Rejected |

At n16 the one-layer configuration has 143,632 evolved points, compared with
159,200 for the control: 9.78% fewer. Shell angular and radial extensions are
unchanged. This is not elimination of geometric overlap, nor an accuracy-
matched speedup. The initial serial CPU measurements show additional penalty
work; no GPU speedup is established.

![Preliminary checks](preliminary.png)

The smooth-data plot measures interpolation consistency only. These short
Minkowski/oblique tests are not interface-crossing, long-time stability,
puncture, or convergence qualification. The `3M` result in the neighboring
V11 report belongs to the preserved control, not this variant. No horizon
acceptance tolerance has changed.

Reproducible local preflight:

```sh
cmake -S . -B build-thin-overlap -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_MPI=ON -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON
cmake --build build-thin-overlap --target \
  hyperbolic_shift_3d_paired_trace hyperbolic_shift_3d_paired_trace_device \
  hyperbolic_shift_3d_mpi_thin_trace hyperbolic_shift_3d_mpi_continuous_positive_fast -j2
build-thin-overlap/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_paired_trace 16 1 1
build-thin-overlap/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_paired_trace_device
python3 tst/hyperbolic_shift_3d/test_thin_trace_mpi.py \
  --binary build-thin-overlap/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_thin_trace \
  --control build-thin-overlap/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_continuous_positive_fast \
  --mpiexec /path/to/mpiexec --output /fresh/output/directory
```

Runtime variables, only for the experimental target:
`HYPERBOLIC_TRACE_LAYERS` (1..3, default 1),
`HYPERBOLIC_TRACE_SHIFTED` (0/1, default 1),
`HYPERBOLIC_TRACE_KAPPA` (nonnegative, default 0), and
`HYPERBOLIC_TRACE_NORMAL_RATIO` (nonnegative, default .25).
The extra `oblique` initial condition is a compact off-center lapse pulse;
its continuous evolution is not an exact analytic solution.

## Remaining gates

1. Longer oblique and spherical interface probes, reflection/error diagnostics
   and independent timestep refinement. Preserve failed cases.
2. Merge the separately validated sparse-exchange work; include trace donors
   in its persistent dependency union and repeat poisoned dependency tests.
3. Fresh frozen SYCL+MPI source/build and a 12-tile, single-debug-node preflight,
   sharing the existing campaign allocation/storage guards.
4. Isotropic puncture through at least 3M, with matching 1D norm conventions
   and accepted horizons only. The n16 thin initial data have been generated,
   with provenance retained, but no successful duration test is claimed.
5. Comparable-accuracy performance and convergence comparison; AMR evolution
   remains a separate unresolved infrastructure limitation.

Recommendation pending these gates: retain V11 as the control. One-layer
coupling is a viable bounded experiment only with an explicit donor policy;
its construction and dissipative added term alone do not justify adoption.
