# Physical-trace layer validation

Date: 2026-10-09, America/New_York. Implementation:
`27c19d20696ea6dd4704032c51dfd026218f64f2`, branch
`z4c_hyperboloidal_layer`. The preserved `z4c_hyperboloidal` branch remains at
`ca77b353a60a939c66227e42b02318c5cd32be9d`.

The new lapse removes the identified frozen growing pole, the symmetric
continuation removes cube-symmetry mismatch, and the native layer now accepts
derived Schwarzschild wormhole data with a Minkowski reference. **Finite-pulse
evolution stability, nonlinear scri closure and wormhole-to-trumpet evolution
remain unverified.** The equations and their limitations are in
[the stabilization document](hyperboloidal-layer-stabilization.md).

The [JSON receipt](validation/hyperboloidal-stability-20261009.json) retains
inputs, commands, history rows, executable hashes, constraint budgets, source
limitations and log hashes. The
[source manifest](validation/hyperboloidal-stability-source-20261009.json)
identifies all 543 tracked source/test/build-description files captured for the
clean implementation build. Generated executables and full field dumps remain
in ignored local output directories.

## Build identity and checks

Local Apple Clang 21, CMake 4.4.3, arm64, Kokkos Serial and double precision;
MPI and OpenMP disabled. Kokkos is clean at
`6739bc623081648af9e752b616d9671527922cbf`. Debug uses AddressSanitizer,
UndefinedBehaviorSanitizer and Kokkos bounds checks. Source was clean before
and after the implementation-commit builds; the compiled source manifest did
not change during those builds. No Aurora build or PBS job was run in this stage.

| Clean executable | SHA256 |
|---|---|
| Release `build-layer-release/src/athena` | `5ba555211db81985274f8ebe789869b8f6300f638ac8dfd7ba7736b541160788` |
| Debug `build-layer-debug/src/athena` | `7a1a85f1f2062b31d31eadfb11cfe6b49b69f04bb3bce0aa71a4c437d76ba090` |

| Check | Result | Seconds |
|---|---|---:|
| Full Release CTests | 23 passed | 109.60 |
| Final native/Cauchy Release regressions | 113 passed | 34.39 |
| Focused Debug CTests | 12 passed | 103.84 |
| Clean implementation Release gates | 6 passed | 2.00 |
| Clean implementation Debug gates | 6 passed | 29.18 |
| Final Debug native rejection cases | 12 passed | 2.87 |

The full Release CTests ran after the collapsed-lapse logarithm fix, before the
final adapter-only legacy-curvature rejection. Its tested kernel sources changed
only by a reference comment afterward. The final native Release set covers the
curvature rejection and all runtime changes; its executable is byte-identical
to the clean implementation build. Clean-commit gates cover physical-gauge
reference/poles, the actual 20-field principal matrix, wormhole geometry/symbolics,
and the constraint-tangent audit in both configurations.

A selected 22-test Debug native run passed in 631.99 seconds before the later
logarithm and curvature-admission fixes. The M=.5 wormhole case passed again
after the logarithm fix in 50.76 seconds; the final 12 rejection cases passed
with the curvature restriction. These are documented development checks, not
a claim that all final native/Cauchy tests were rerun in Debug. The earlier
full Debug limitations remain in the original validation receipt.

New/edited standalone C++ headers and tests pass project cpplint; Python passes
pycodestyle with the project 90-column limit. The preexisting whole-file lint
issues in `z4c_newdt.cpp` were not treated as new failures. `git diff --check`
passes. Peer review prompted the positive-collapsed-lapse log test through
`1e-300` and rejection of nonunit curvature in the legacy fixed-speed CMC path.

## Reproduce the checked paths

Use the Debug/Release configuration commands in
[the original validation document](hyperboloidal-layer-validation.md).
The added targets are enabled by the same hyperboloidal test/analysis options.
Set `ROOT` to this checkout and `PYTHON` to an interpreter with numpy, sympy,
mpmath and pytest.

```sh
ctest --test-dir "$ROOT/build-layer-release" --output-on-failure
ctest --test-dir "$ROOT/build-layer-debug" --output-on-failure \
  -R 'layer|physical_gauge|symmetric_ghost'

ATHENA_OVERHAUL_EXE="$ROOT/build-layer-release/src/athena" \
ATHENA_HYP_PATCH_EXE="$ROOT/build-layer-release/hyperboloidal_cartesian_tests" \
"$PYTHON" -m pytest -q \
  "$ROOT/tst/test_suite/z4c/test_hyperboloidal_layer_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_hyperboloidal_native_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_hyperboloidal_wormhole_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_z4c_conversion_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_z4c_restart_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_z4c_overhaul_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_adm_gauge_storage_cpu.py"

"$PYTHON" "$ROOT/tst/hyperboloidal/run_constraint_tangent.py" \
  "$ROOT/build-layer-release/hyperboloidal_layer_constraint_tangent" \
  "$ROOT/build-layer-research/fresh-tangent"
"$PYTHON" "$ROOT/tst/hyperboloidal/run_layer_validation.py" \
  "$ROOT/build-layer-release/src/athena" \
  "$ROOT/build-layer-research/fresh-geometry" --suite geometry-scan
```

The fresh directories must not already contain the harness's immutable
executable. Harness launch-workspace snapshots are explicitly distinguished
from proven executable build source. Some earlier dirty development evolutions
lack a complete source snapshot; their base commit, binary hash and actual
inputs are retained without inventing an exact dirty source identity.

The inputs `hyperboloidal_layer_physical.athinput` and
`hyperboloidal_layer_wormhole.athinput` are short two-step diagnostics. Their
small time limit is intentional and supplies no long-evolution acceptance.

## Finite-pulse and reference experiments

All times below are coordinate time, S=1; no black-hole mass normalization
applies to these Minkowski runs. Pulse amplitudes are lapse .1 and shift .02,
with the native nonspherical modulation and width .5. Preferred source is off
for the physical-P lapse. Fields stay positive in these runs, but the constraints
do not meet acceptance.

| Reference / continuation | a | End time | H | M | Z |
|---|---:|---:|---:|---:|---:|
| Layer .35–.75, cubic | 1 | .2 | 2.1175 | .9672 | .05393 |
| Pure CMC, cubic | 1 | .2 | .006414 | .03590 | .008659 |
| Layer .35–.75, symmetric quadratic | .5 | .5 | .9831 | .7381 | .1068 |
| Layer .2–.8, symmetric quadratic | .5 | .5 | .15577 | .23862 | .09341 |
| Broad stationary reference, N24 | .5 | .5 | 6.48e-14 | 7.21e-14 | 2.10e-15 |
| Broad angular pulse, N36 | .5 | .2 | .03140 | .02518 | .005031 |

The corresponding N24 broad history, interpolated to t=.2, is
`(H,M,Z)=(.05505,.05653,.02234)`. Two resolutions improve the error, but do not
establish an asymptotic order or multiple-crossing stability. Reducing the
actual N24 timestep from `.00057216` to `.00042773` changes t=.5 H from
`.155767` to `.155697`. The global CFL cancels in this adapter; the receipt
records the actual timestep rather than calling that ratio a halving.

The instantaneous constraint-tangent sweep contains 31 successful probes.
The analytic continuum sampled maximum `|Hdot|` is `8.42e-7`. Broad native
RMS Hdot at N24/36/48/64/72 is `.7971,.3702,.2163,.1098,.08118`, with final
observed orders 2.36 and 2.56. At N72, 99.85% of squared Hdot lies in the
transition. Cubic and quadratic continuation have exactly the same bulk peak.
This identifies a spatial defect independently of the time integrator and
outer ghost choice. N64/72 are instantaneous probes, not long evolutions.
The attempted matrix-free eigenvalue solve returned no converged eigenpairs.

## Single-hole evidence

The native M=.1,.2,.5 tests recover the Schwarzschild mass from physical ADM
fields and keep the reference metric/connection equal to Minkowski. Initial
Hamiltonian norms are `1.04e-14`, `7.95e-15`, `4.65e-15`; momentum norms are
about `1.4–1.6e-15`, and Theta is zero. Each completes two actual steps to
`t=.0005703125`. For M=.5 this is only `.001140625 M`; final H/M/Z are
`.0002351/.0003326/.00009218`, with positive lapse and chi. Geometry and gauge
change, and restart agrees with uninterrupted evolution.

No long single-hole run, trumpet profile, mass convergence or binary result is
claimed. The remaining black-hole criterion is the inner wormhole-to-trumpet
transition with a Minkowski hyperboloidal reference over multiple crossing times.
