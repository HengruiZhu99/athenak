# Hyperboloidal solver

The current Cartesian-AMR driver is **`hyperbolic_shift_3d_amr_evolution`**.
It constructs the final halo plan directly at startup, restart, and regrid.
No performance-selection macro or alternate executable is needed.

```sh
cmake -S . -B build-hyperbolic-mpi -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_MPI=ON -DAthena_SINGLE_PRECISION=OFF \
  -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON -DKokkos_ENABLE_SERIAL=ON
cmake --build build-hyperbolic-mpi -j4 --target hyperbolic_shift_3d_amr_evolution
mpiexec -n 2 build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_amr_evolution \
  12 wave 6 .00025 output-prefix - dynamic
```

Use a fresh output prefix. The command is a short correctness fixture, not a
production evolution. The AMR1 scientific configuration, timestep, native rank
limits, checkpoint format, diagnostics, and known dynamic boosted rejection are
unchanged. See the [Cartesian AMR report](reproduction/cartesian-amr-v1/README.md)
for those contracts and the [promotion record](reproduction/default-cleanup-v1/README.md)
for current validation.

## Code map

* [`amr_evolution.cpp`](../../tst/hyperbolic_shift_3d/amr_evolution.cpp) owns native
  regrids, AMR1 snapshots/restart, scientific diagnostics, and failure capture.
* [`coupled_mpi.cpp`](../../tst/hyperbolic_shift_3d/coupled_mpi.cpp) implements the
  shared `Evolution` machinery: state, compensated RK4, sparse exchange, boundary
  operations, horizon sampling, and the fixed-mesh CLI. The AMR driver reuses it.
* [`hyperbolic_mesh_plans.hpp`](../../src/z4c/hyperboloidal/hyperbolic_mesh_plans.hpp)
  stages and publishes coordinated plan generations for the planned driver.
  It also selects final interpolation before constructing its host model.
* [`amr_performance.hpp`](../../tst/hyperbolic_shift_3d/amr_performance.hpp) writes
  observational timings and workload/allocation counts, without extra fences.

These lifecycle files use the local
[formatting configuration](../../tst/hyperbolic_shift_3d/.clang-format), with
expanded control flow and unchanged include ordering. Scientific variants with
different equations or checkpoint identities remain separate; they are not
interchangeable performance alternatives.

## Historical performance experiments

The [performance/scaling report](reproduction/performance-scaling-v1/README.md)
records measurements from commit `71765458a` and earlier frozen variants. Its
names and commands describe that experiment, not today's target list.
The former `hyperbolic_shift_3d_amr_setup` behavior is now the default AMR driver.
The duplicate target, direct-setup switch, and unsuccessful diagnostic-transfer
implementation/target have been removed. The radial-mask experiment was already
removed. Historical source and raw evidence remain available in Git and the
frozen artifact archives; replay them at their recorded revision. The replay
helper rejects current source trees that no longer contain the experimental
switches, rather than silently building a mislabeled variant.

The earlier [method and admission documentation](method-and-admission-history.md)
is preserved separately, including formulation equations, scientific qualifications,
and links to the original resolution and boosted-gauge campaigns.
