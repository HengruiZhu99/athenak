# Composite 3D elliptic shift: research backend

The exact variable-metric shift and a global composite-grid CPU solver are
implemented and tested. **The full requested production/scientific milestone
is incomplete.** Start with [status and blockers](status.md), the
[equation/boundary derivation](derivation.md), and the
[spherical-to-3D test map](spherical-test-mapping.md).

Source revisions and selective reuse are recorded in
[source-revisions.json](source-revisions.json). Alan Lam's CFC scalar potentials
were inspected as infrastructure references; no CFC physics or isolated Robin,
Sommerfeld or multipole condition was substituted for the hyperboloidal gauge.

## Reproduce on local CPU

Use this isolated worktree and its exact Kokkos gitlink. If initializing a fresh
clone, initialize Kokkos **in that new clone**, not a running research checkout.

```sh
cmake -S . -B build-shift-3d -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_NATIVE=OFF \
  -DAthena_ENABLE_ELLIPTIC_SHIFT_3D_TESTS=ON \
  -DAthena_ENABLE_CUBED_SPHERE_TESTS=ON
cmake --build build-shift-3d --target elliptic_shift_3d_operator_tests \
  elliptic_shift_3d_composite_tests elliptic_shift_3d_coupled_pilot \
  elliptic_shift_3d_lifecycle_tests cubed_sphere_map_tests -j2
ctest --test-dir build-shift-3d \
  -R 'elliptic_shift_3d|cubed_sphere_map$' --output-on-failure
```

Optional bounded **local** MPI correctness oracle:

```sh
cmake -S . -B build-shift-3d-mpi -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_MPI=ON -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_NATIVE=OFF -DAthena_ENABLE_ELLIPTIC_SHIFT_3D_TESTS=ON
cmake --build build-shift-3d-mpi --target elliptic_shift_3d_mpi_tests -j2
```

The complete matrix uses Python3, NumPy and Matplotlib, one test worker, at most
two local MPI ranks and a 180-second per-case timeout. Choose an unused output
directory; it deliberately refuses to overwrite evidence.

```sh
python3 tst/elliptic_shift_3d/validate.py --build build-shift-3d \
  --mpi-build build-shift-3d-mpi --output build-shift-3d/reproduced-evidence
python3 tst/elliptic_shift_3d/analyze_surfaces.py build-shift-3d/reproduced-evidence
python3 tst/elliptic_shift_3d/plot_validation.py build-shift-3d/reproduced-evidence
```

Omit `--mpi-build` if local MPI is unavailable. The nine retained scientific
boundary failures are expected outcomes in the current matrix; they are not
relabelled passing checks. The frozen numerical implementation is commit
`487f750d8b8503b4944ee75d0b3b1efdcee1b72a`; its full source/executable hashes
and outcomes are in [the manifest](evidence/validated-20261027/manifest.json).
Later plotting/audit changes do not require rerunning physics and are recorded
separately in the report provenance.

Double-precision CPU execution is the validated configuration. The host mapped
backend, replicated MPI oracle and research checkpoint are separate from native
AthenaK pack/task/GPU/restart integration. Existing production guards remain.
