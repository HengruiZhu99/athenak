# Native spherical CMC/Dirac solver

`<spherical_cmc>` selects a native vacuum 1+1D constrained Einstein module.
It supports regular-center Minkowski geometry and excised Schwarzschild CMC
trumpets, with evolving radial coordinate distortions, an elliptic Dirac shift,
and exact algebraic scri fixing. See [the derivation](spherical_cmc.pdf) and
[editable LaTeX](spherical_cmc.tex).

This is a constrained spherical research solver, **not free Z4c evolution**.
Mass and the trumpet integration constant are fixed. Vacuum spherical symmetry
has no gravitational-wave degrees of freedom. The implementation does not prove
well-posedness of the proposed 3D elliptic-hyperbolic formulation.

## Build and run

Initialize Kokkos if necessary, then build the ordinary AthenaK executable:

```sh
git submodule update --init --recursive
cmake -S . -B build-cmc -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_MPI=OFF -DKokkos_ENABLE_SERIAL=ON \
  -DAthena_ENABLE_SPHERICAL_CMC_TESTS=ON
cmake --build build-cmc -j4
mkdir -p run-cmc
cd run-cmc
../build-cmc/src/athena -i ../inputs/spherical_cmc/minkowski.athinput
```

Use `inputs/spherical_cmc/schwarzschild.athinput` for the black-hole example.
Both inputs initialize `a=exp(A*r^2*(1-r^2)^2)` with `A=0.05`. The pulse is a
coordinate distortion, not injected gravitational radiation.

| Parameter | Meaning |
|---|---|
| `spherical_cmc/metric_pulse` | Initial distortion amplitude, finite and within `[-0.5,0.5]` |
| `spherical_cmc/dirac_damping` | Nonnegative coefficient in `d_t V=-eta V` |
| `spherical_cmc/mass` | `0` for a regular center, positive for an excised critical trumpet |
| `mesh/x1min` | `0` for regular center; positive and physically inside the horizon for black holes |
| `mesh/x1max` | Exactly `1`, the fixed scri location |
| `mesh/nx1`, `meshblock/nx1` | Equal radial cell counts, at least 8 |
| `spherical_cmc/state_version` | Restart layout version, currently `1` |

Set `metric_pulse=0` for an undistorted reference, or `dirac_damping=0` for a
stationary distorted geometry. The black-hole example uses 128 cells, `mass=0.5` and
`x1min=0.05`. Repeat at 64/128/256 cells before interpreting accuracy. Different masses need an appropriate excision radius; the module
rejects an inner boundary outside the horizon or with incoming physical modes.

Supported configuration: one rank, one uniform 1D MeshBlock, double precision,
RK3, `nghost>=2`, fixed `K=-3` and `Omega=(1-r^2)/2`, no matter/other physics.
The radial elliptic solve executes on the host; GPU performance has not been
validated. The examples' `outflow` mesh flags are placeholders for the native
mesh setup: the module itself supplies the mathematical endpoint conditions.

## Native integration

`MeshBlockPack::pspherical_cmc` owns the physics state. Each RK stage evaluates

```text
CopyState -> SolveGauge -> RKUpdate -> SolveGauge -> NewTimeStep
```

The module reuses native input parsing, cell coordinates, Kokkos state arrays,
RK coefficients, task lists, CFL selection, field/history output, and restart.
It also reuses the existing hyperboloidal spherical tensor, physical constraint,
conformal RHS, mass-diagnostic, and factored CMC-trumpet kernels.

The new radial solve uses a fixed-width band matrix with O(N) storage/work;
it does not instantiate a 3D multigrid hierarchy for a single radial block.
Fourth-order finite differences and sixth-order interval quadrature avoid the
boundary errors encountered with lower-order closures. The continuum system
remains second order in space.

The two exact-boundary metric DOFs occupy the nearest ghost slots. They are
preserved in native restarts, but they **are not values at ghost cell centers**.
Consequently field output with `ghost_zones=true` is rejected.

## Outputs and restart

Use `variable=spherical_cmc` for the 13 native fields documented in the PDF.
History output has suffix `.spherical_cmc.hst`; `null_scri` is the exact-endpoint
null residual. `scmc_Krhs` and `scmc_Arhs` come from the full tensor equations.
The latter is not an error by itself during black-hole gauge evolution.
The radial C++ test compares it with the tangent of the constraint projection.

The mass diagnostic near scri suffers cancellation. The history column
`mass_rle08_max` deliberately uses only cells with `r<=0.8`; the unrestricted
field remains available. This is not yet a regular Bondi-mass extractor.

Restart uses AthenaK's usual interface:

```sh
../build-cmc/src/athena -r rst/cmc.00000.rst time/tlim=2
```

Use the actual generated checkpoint filename. Both shared-file and
`single_file_per_rank` layouts are tested. Dependent fields are recomputed after
restart; retaining the same mass/gauge parameters is required for a physical
continuation of the same problem.

## Verification

Dependencies for the Python tests are NumPy, pytest, and SymPy; the optional
binary-reader test also uses h5py. From the root:

```sh
python3 tst/spherical_cmc/check_equations.py
ctest --test-dir build-cmc -R spherical_cmc --output-on-failure
ATHENA_CMC_EXE="$PWD/build-cmc/src/athena" python3 -m pytest -q \
  tst/test_suite/spherical_cmc/test_native_cpu.py
```

The native suite has 18 tests: reference preservation, Dirac relaxation and
constraint refinement, stationary and evolving black-hole tests, four bitwise
restart-equivalence cases, native binary/table field equivalence, and nine
invalid-configuration checks. The C++ radial
test also verifies the reconstructed black-hole extrinsic-curvature evolution
against the full tensor RHS. The SymPy script verifies the continuum identities.

Representative CPU results (double precision, amplitude 0.05, eta=1, time=2):

| Family | N | max physical Hamiltonian residual | max full-tensor K RHS |
|---|---:|---:|---:|
| Regular center | 32 | 2.03e-7 | 9.61e-7 |
| Regular center | 64 | 1.31e-8 | 1.51e-7 |
| Regular center | 128 | 9.93e-10 | 2.07e-8 |
| Schwarzschild M=0.5 | 64 | 1.01e-3 | 7.48e-6 |
| Schwarzschild M=0.5 | 128 | 1.85e-4 | 8.67e-7 |
| Schwarzschild M=0.5 | 256 | 3.77e-5 | 9.29e-8 |

Longer N=64 runs through t=10 stayed finite and retained zero scri null residual.
The black-hole Dirac residual then reached about 1.8e-3, illustrating the
remaining coarse-grid error near excision rather than exact discrete exponential
damping. The regular-center Dirac residual was 2.7e-6.

The scri null residual was exactly zero. These are measured convergence results,
not a uniform-order theorem: evolving black-hole constraints converge more
slowly than smooth static projections, and the regular-center Hamiltonian
reaches a floating-point differentiation floor at higher N.

The radial kernel passed AddressSanitizer and UndefinedBehaviorSanitizer:

```sh
cmake -S . -B build-cmc-sanitize -DAthena_ENABLE_MPI=OFF \
  -DKokkos_ENABLE_SERIAL=ON -DAthena_ENABLE_SPHERICAL_CMC_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer -O1'
cmake --build build-cmc-sanitize --target spherical_cmc_tests -j4
ASAN_OPTIONS=detect_leaks=0 build-cmc-sanitize/spherical_cmc_tests
```

LeakSanitizer is disabled because this execution environment does not support
its process inspection. Memory-access and undefined-behavior instrumentation
remain enabled. The full instrumented AthenaK executable also passed 17 bounded
native tests (the long black-hole refinement test was run in the release build):

```sh
cmake --build build-cmc-sanitize --target athena -j4
ASAN_OPTIONS=detect_leaks=0 \
ATHENA_CMC_EXE="$PWD/build-cmc-sanitize/src/athena" python3 -m pytest -q \
  tst/test_suite/spherical_cmc/test_native_cpu.py \
  -k 'not black_hole_gauge_evolution'
```

For changes to shared output/restart hooks, also run the existing suites:

```sh
ATHENA_OVERHAUL_EXE="$PWD/build-cmc/src/athena" python3 -m pytest -q \
  tst/test_suite/z4c/test_z4c_overhaul_cpu.py \
  tst/test_suite/z4c/test_z4c_conversion_cpu.py \
  tst/test_suite/z4c/test_z4c_restart_cpu.py
```

All 46 existing Z4c tests passed, including the binary-conversion tests with
h5py installed.

Enable `Athena_ENABLE_HYPERBOLOIDAL_TESTS=ON` to build the broader CTest suite.
All 12 CTests passed, including the new radial test.

Compile the documentation with:

```sh
mkdir -p build-docs
pdflatex -halt-on-error -output-directory=build-docs docs/spherical_cmc.tex
pdflatex -halt-on-error -output-directory=build-docs docs/spherical_cmc.tex
```
