# Composite 3D elliptic shift: current implementation and evidence

**Takeover in progress:** [current spherical alignment and interface evidence](takeover-alignment-20261027.md).
The [latest affine and endpoint diagnosis](compatible-affine-and-endpoint-20261027.md)
identifies a dominant high radial mode and tests an enriched shift space.
The [combined-grid and physical wormhole gauge tests](compatible-wormhole-gauge-20261027.md)
pass the bulk solve but expose a large scri-rate failure. The candidates remain
unadopted; stable 3D collapse is unverified. The preceding
[compatible derivative prototype](compatible-gauge-prototype-20261027.md) retains its component evidence.
The [dated handoff](TAKEOVER-20261027.md) preserves the previous paused state.

**Residual qualification:** all prior "original vector residual" results refer to
the expanded continuum-equation discretization. Exact freezing of the actual
discrete metric-built connection is not established and fails the new accepted-state
audit. See [discrete gauge qualification](discrete-gauge-qualification.md).

The exact variable-metric shift and a global composite-grid CPU solver are
implemented and tested. A restricted experimental native Driver/RK/restart
path now passes Minkowski and small-pulse controls on engine-owned shells. **The full requested production/scientific milestone
is incomplete.** Start with [status and blockers](status.md), the
[equation/boundary derivation](derivation.md), and the
[spherical-to-3D test map](spherical-test-mapping.md).

Source revisions and selective reuse are recorded in
[source-revisions.json](source-revisions.json). Alan Lam's CFC scalar potentials
were inspected as infrastructure references; no CFC physics or isolated Robin,
Sommerfeld or multipole condition was substituted for the hyperboloidal gauge.

The sections below retain their historical commands and outcomes; consult
[the current status](status.md) and
[native integration qualification](native-composite-integration.md) before interpreting an old failure or integration
statement. Actual native controls are reproduced with:

```sh
cmake --build build-shift-3d --target athena -j2
python3 tst/elliptic_shift_3d/run_native_composite_smoke.py \
  --extended-controls --output build-shift-3d/native-reproduced
```

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
  elliptic_shift_3d_tensor_scri_tests elliptic_shift_3d_wormhole_chart_tests \
  elliptic_shift_3d_chart_equivalence_tests elliptic_shift_3d_native_fields_tests \
  elliptic_shift_3d_radial_kernel elliptic_shift_3d_lifecycle_tests cubed_sphere_map_tests -j2
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

## Tensor scri and selected wormhole continuation

The tensor scri limit/surface continuation is described in
[tensor-scri-closure.md](tensor-scri-closure.md). The selected M=1/3 wormhole
lapse, positive charts, rejected controls and explicit proof that the
Minkowski-reference equations are unchanged are in [wormhole-chart.md](wormhole-chart.md).
The latest focused suite has10/10 passing component/regression tests, while
**all27 wormhole scientific cases in the final three-chart matrix fail**.
They are retained with unchanged targets and independently matched 1D profiles;
passing component tests does not override those failures.

For a bounded rerun of one complete chart family (one Serial worker,
N=6/8/12, t=.0002, at most34,560 cells, 180 seconds per case):

```sh
python3 tst/elliptic_shift_3d/validate_wormhole.py \
  --binary build-shift-3d/elliptic_shift_3d_coupled_pilot \
  --output build-shift-3d/wormhole-reproduced --isotropic-reference 2
```

Chart0 is the generic positive-power chart, chart1 the rejected compact
initial-factor blend, and chart2 the exploratory smooth AF factorization.
Every variant evaluates the same full physical tensor/gauge RHS. No lapse
floor, attenuation candidate, Omega floor or regular-center parity is used.
The independent spherical reference uses the existing equations with centered,
zero-KO discretization controls, explicitly distinguished from historical
upwind/KO long runs.

Frozen CSV/initializer files are losslessly compressed; restore them before
replaying the recorded comparison commands:

```sh
python3 tst/elliptic_shift_3d/rehydrate_evidence.py docs/elliptic-shift-3d/evidence
python3 tst/elliptic_shift_3d/compare_wormhole.py \
  --matrix docs/elliptic-shift-3d/evidence/wormhole-final-20261027 \
  --reference docs/elliptic-shift-3d/evidence/wormhole-radial-reference-20261027
```

The first command verifies hashes and refuses to overwrite different data.
Use `--check-only` for a read-only archive check. The final matrix's kernel
revision is62170e4bf. The native field adapter is tested on actual enums and
Kokkos pack views; production stage scheduling, arbitrary shell allocation,
MPI/AMR lifecycle and native restart remain unconnected.
