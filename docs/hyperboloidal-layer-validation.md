# Layer validation and negative results

Date: 2026-10-08, America/New_York. This records local arm64 CPU evidence only.
The exact source/build/run receipts are in
[validation/hyperboloidal-layer-20261008.json](validation/hyperboloidal-layer-20261008.json).
The full equations and unresolved closure are in
[hyperboloidal-layer.md](hyperboloidal-layer.md).

**Acceptance outcome: reference geometry and principal-symbol gates pass;
nonlinear regularity, live convergence and evolution stability gates fail.**
No single-hole data, binary, GPU, MPI/AMR evolution, Bondi mass or waveform claim
is supported. All time values below use S=a=1 and the coordinate time whose
normalization is fixed by t=T-h(R); no black-hole mass scale is present.

## Build and test commands

Kokkos is clean at `6739bc623081648af9e752b616d9671527922cbf`. Compiler is Apple
Clang 21.0.0, target arm64-apple-darwin25.6.0; CMake 4.4.3. Native vacuum pgen is
from `built_in_pgens`. Both builds explicitly disable MPI and OpenMP and use
Kokkos Serial. Debug enables Kokkos bounds checks and AddressSanitizer/UBSan.

Use a Python environment with numpy, sympy, mpmath and pytest. The environment
used here has Python 3.9.6, numpy 2.0.2, sympy 1.14.0, mpmath 1.3.0 and pytest
8.4.2. Set `PYTHON` to that interpreter and `ROOT` to this checkout.

```sh
cmake -S "$ROOT" -B "$ROOT/build-layer-release" \
  -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON \
  -DAthena_ENABLE_MPI=OFF -DAthena_ENABLE_OPENMP=OFF \
  -DAthena_ENABLE_HYPERBOLOIDAL_TESTS=ON \
  -DAthena_ENABLE_HYPERBOLOIDAL_ANALYSIS=ON -DPython3_EXECUTABLE="$PYTHON"
cmake --build "$ROOT/build-layer-release" -j8
ctest --test-dir "$ROOT/build-layer-release" --output-on-failure

cmake -S "$ROOT" -B "$ROOT/build-layer-debug" \
  -DCMAKE_BUILD_TYPE=Debug -DKokkos_ENABLE_SERIAL=ON \
  -DAthena_ENABLE_MPI=OFF -DAthena_ENABLE_OPENMP=OFF \
  -DAthena_ENABLE_HYPERBOLOIDAL_TESTS=ON \
  -DAthena_ENABLE_HYPERBOLOIDAL_ANALYSIS=ON -DPython3_EXECUTABLE="$PYTHON" \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
  '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined'
cmake --build "$ROOT/build-layer-debug" -j8
ctest --test-dir "$ROOT/build-layer-debug" --output-on-failure \
  -E 'spherical_ghosts|conformal_wave|cartesian_patch|hawking_mass'
ctest --test-dir "$ROOT/build-layer-debug" -R layer --output-on-failure

ATHENA_OVERHAUL_EXE="$ROOT/build-layer-release/src/athena" \
ATHENA_HYP_PATCH_EXE="$ROOT/build-layer-release/hyperboloidal_cartesian_tests" \
"$PYTHON" -m pytest -q \
  "$ROOT/tst/test_suite/z4c/test_hyperboloidal_layer_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_hyperboloidal_native_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_z4c_conversion_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_z4c_restart_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_z4c_overhaul_cpu.py" \
  "$ROOT/tst/test_suite/z4c/test_adm_gauge_storage_cpu.py"
```

Repeat the pytest command with both executable paths changed to Debug for the
sanitizer regressions. The legacy fifth-degree N=36 donor/evolution test has a
120-second subprocess budget. Its sanitizer trial exceeded that budget; the
resource-only rerun uses
`docs/validation/rerun-layer-debug-timeout.py` from the checkout root, with the
same Debug executable environment variables, to allow 600 seconds while retaining
every test assertion. The receipt distinguishes the original timeout and rerun.
These explicit paths prevent accidental testing of a
production or stale executable. The sweep script makes an immutable executable
copy in a fresh output directory and records its hash and commands:

```sh
"$PYTHON" "$ROOT/tst/hyperboloidal/run_layer_validation.py" \
  "$ROOT/build-layer-release/src/athena" "$ROOT/build-layer-validation/fresh-quick"
"$PYTHON" "$ROOT/tst/hyperboloidal/run_layer_validation.py" \
  "$ROOT/build-layer-release/src/athena" "$ROOT/build-layer-validation/fresh-control" \
  --suite control
"$PYTHON" "$ROOT/tst/hyperboloidal/run_layer_validation.py" \
  "$ROOT/build-layer-release/src/athena" "$ROOT/build-layer-validation/fresh-long" \
  --suite long
"$PYTHON" "$ROOT/tst/hyperboloidal/run_layer_validation.py" \
  "$ROOT/build-layer-release/src/athena" "$ROOT/build-layer-validation/fresh-reference" \
  --suite reference-long
```

## What passed

The original baseline Release build and all 11 original hyperboloidal CTests
passed before integration (108.62 seconds). With the new analytical gates, all
17 Release CTests passed. The final six layer gates were rerun in Release and
Debug/ASan/UBSan after formatting and the residue/boundary audits were added.
The selected 13 Debug CTests passed (197.75 seconds); the four expensive legacy
spatial/evolution tests listed in the command above were excluded from that
Debug run, not from the full Release run.

The clean implementation commit `d21fb74ce6ad1308d77aad3be6f39da78eca93a7`
was checked again with all six analytical/kernel gates in Release (1.52 seconds)
and Debug/ASan/UBSan (4.82 seconds), an immutable native control, and the complete
100-test Release regression set (31.13 seconds).

The Release native/Cauchy regression set passes **100 tests**, including the
11 new layer tests, the 37 original native hyperboloidal tests, and 52 Cauchy
conversion/restart/overhaul/separate-gauge tests. New native tests cover
true nonflat transition initial data, stationary reference, physical ADM map,
radial/angular live lapse and shift, restart, invalid radii/coefficients, and
rejection of CMC trumpet data on the layer.

The original 100-test Debug/ASan/UBSan development run finished with **96 passed,
4 failed in 2004.62 seconds**. Failures were the legacy fifth-degree N=36
120-second timeout, two refinement-cap tests exposing a pre-existing
`Mesh::PrintMeshDiagnostics` heap-buffer-overflow, and the 15-step separate-gauge
AMR outflow test's 90-second timeout. The fifth-degree rerun passed unchanged
assertions in 274.97 seconds with a 600-second subprocess allowance.

The mesh diagnostic allocated `max_level` entries but read the inclusive highest
physical level. The identical defect exists in the starting commit. Commit
`49e3ee3d17c51767abbd1b53492113869ee269be` allocates and initializes
`max_level-root_level+1` entries. Both affected refinement-cap tests then pass
under ASan/UBSan (53.22 seconds), and the complete Release regression set passes
again (100 tests in 31.11 seconds). This fixes a startup diagnostic, with no
change to the evolution equations or AMR policy.

The 15-step AMR outflow sanitizer test was not completed with a larger budget;
its original run timed out after the first refinement cycle. Its complete
Release test passes. It remains a Debug resource/coverage limitation rather than
a sanitizer pass or an established numerical failure. Layer AMR execution is
explicitly rejected. Thus this work does **not** claim a clean single-run
100-test sanitizer suite. The original failures and successful rerun logs are
preserved alongside the negative layer experiments.

The compiled principal-symbol gate checks 180 cases: lapse .2/1/3, chi .4/1/2,
10 radii including exact harmonic coefficients, and aligned/oblique sheared
frames. Maximum matrix error is 3.553e-15, left-eigenfield error 8.882e-16, and
normalized basis condition number 11.53 over this sample. Exact symbolic checks
cover generic q0 and the canceled limiting ratios; no uniform puncture bound is
claimed. The old endpoint's geometric multiplicity three is reproduced.

The independent 100-digit cutoff oracle checks 13 positions including both
endpoints and derivatives through order three, with maximum scaled error
2.17e-15. Analytic reference jets satisfy physical constraints and the continuum
stationary tensor RHS to 2.33e-12 in the sampled transition points. A live,
nonflat collar source identity has maximum error 5.62e-16. Interior assembly at
exact scri is explicitly rejected.

The **raw centered AthenaK stencil** stationary residual, before analytic
reference restoration, is:

| Spacing h | Maximum geometric/gauge RHS | Order |
|---|---:|---:|
| .02 | .107017 | — |
| .01 | .00626457 | 4.09448 |
| .005 | .000384231 | 4.02717 |
| .0025 | .0000238977 | 4.00703 |

These manufactured interior stencils do not use an exterior spherical ghost.
The native restored N=24 reference at t=.05 has H/M/Z RMS
7.43e-14 / 4.22e-14 / 4.10e-15 and Theta RMS 1.04e-15. It is a short fixed-point
check; its longer evolution fails below.

## Live refinement fails

All runs below use cubic ghosts, kappa1 input 5, restoring rates 0→1.5 and 0→1,
KO .1, pole coefficient .04, and the preferred source. Small angular pulse is
lapse .001 and shift .0002; finite radial pulse is lapse .1 and shift .02.
Outputs at t=.01 are synchronized by the native final timestep.

| Pulse / N | H RMS | M RMS | Z RMS | Outcome |
|---|---:|---:|---:|---|
| Small angular / 24 | 2.43218e-4 | 3.19759e-4 | 3.74545e-5 | Reaches .01 |
| Small angular / 36 | 1.43713e-2 | 4.63209e-2 | 1.58312e-3 | Reaches .01 |
| Small angular / 48 | 1.88265e-3 | 8.43115e-3 | 2.03184e-4 | Reaches .01 |
| Finite radial / 24 | 2.38122e-2 | 3.26063e-2 | 3.95620e-3 | Reaches .01 |
| Finite radial / 36 | — | — | — | Fails at mesh time .0081340 |
| Finite radial / 48 | .126920 | .367011 | .0113785 | Reaches .01 |

This is neither monotone global constraint convergence nor a converged solution
sequence. The failed N=36 row's initial saved norms must not be presented as
final evolved norms. Its failing cell is
(-.0291667,-.6125,-.7875), Omega=.00191840, alpha=209.871, chi=-.0130585,
determinant~1, P=-1.79607 and Theta=.421349. Negative chi causes the physical-ADM
rejection; it is not hidden by a floor.

Boundary and bulk budgets are measured separately over all active cells. For
small angular N=24/36/48, the fractions of squared error at r>.9 are

| N | H fraction | M fraction | Z fraction | Interior r<.35 H/M/Z RMS |
|---|---:|---:|---:|---|
| 24 | .00225 | .34561 | .96666 | 3.179e-4 / 1.939e-4 / 6.101e-6 |
| 36 | .99157 | .99477 | .99995 | 3.450e-4 / 1.184e-4 / 2.086e-6 |
| 48 | .98450 | .99933 | .99975 | 2.239e-4 / 7.636e-5 / 1.478e-6 |

The finer-grid large errors are overwhelmingly outer errors, but the interior
Hamiltonian norm is not yet an asymptotic convergence sequence either. Cell
counts, all radial bins and global accounting are retained in the JSON receipt;
no outer cells are omitted to manufacture convergence.

## Width, damping, timestep and boundary sensitivity

At N=24 and t=.01 with the finite *angular* pulse (.1/.02):

| Change | H RMS | M RMS | Z RMS |
|---|---:|---:|---:|
| Baseline | .02431853 | .03196567 | .00374224 |
| Width .6: r0=.2,r1=.8 | .00821327 | .02046595 | .00385302 |
| Width .2: r0=.5,r1=.7 | .07882735 | .08778309 | .00406997 |
| Constraint/restoring damping zero | .02429166 | .03090475 | .02076076 |
| Pole timestep coefficient .02 | .02431852 | .03196573 | .00374223 |
| Ghost degree two | .02432735 | .02655583 | .00117736 |
| Ghost degree four | .02459985 | .05949577 | .01063000 |
| Preferred source disabled | .02433543 | .03196786 | .00374486 |

A narrower smooth layer increases geometric error; higher ghost degree increases
some outer errors; halving the pole timestep does not materially repair this
case. These are sensitivity checks, not a convergence or stability argument.

The cubic N=24 finite angular run requested to t=2 fails at mesh time
.02708984375, cycle 95. The failing state has
(x,y,z)=(.56875,.48125,-.65625), Omega=.00712890625, alpha=576.020,
chi=-.00329868, determinant~1, P=-1.70794 and Theta=.637537. Its last saved
history at .026234375 has H=.272433, M=.769985, Z=.0387347, Theta=.0109638,
null deviation=5.42798 and pole deviation=8.16786.

The native null diagnostic is the interior-shell difference
`|C-Omega^2*N_hat|`. The pole diagnostic is the maximum geometric kernel pole
numerator difference from the reference, including the trace-free curvature
sector. Neither is a closed live evolution for `N`, `Sij` or `W_Omega`, nor an
evaluation at exact scri. Analytic shear checks cover the reference and the
specified counterexample; no compatible live shear limit has been established.

Even the **unperturbed** N=24 reference requested to t=2 fails at mesh time
.19096762207, cycle 670. Its failing lapse is -7.33916e-5, chi=493.461 and
P=-26.7647 at (-.74375,-.65625,.04375), Omega=.00712890625. The last saved history
at .19019921875 has H=92.2235, M=424.030 and Z=12.3787. This invalidates an
interpretation of the short well-balanced reference check as long-term stability.

For the default reference, numerical integration of 1/cplus gives layer r=.35
through scri crossing time .9733339 and origin-to-scri time 1.3233339.
The peak radial reference metric is ~15.838. Neither long trial completes one
layer crossing. Several-crossing acceptance, single-hole evolution, and binary
smoke tests are therefore not reached.

## Independent continuum and boundary failures

The exact and compiled trace/Theta checks give
`Omega*dtQ→2*deltaQ` and `dtTheta_phys→-2*deltaQ` for the smooth null finite-Q
counterexample specified in the formulation document. The full finite-Q/shear/
Z4 compatible manifold is not supplied by that construction.

The extracted lower-order pole block has a positive eigenvalue
2.5717094873/Omega for the default damping. Its exact polynomial proves a positive
root for every nonnegative kappa1 (kappa2=0). This is an independent local
continuum growth obstruction outside the regular asymptotic manifold, not a
claim that the entire coupled PDE spectrum was solved. Finite restoring rates
cannot remove that leading pole. The actual Cartesian failed evolution is
additional discrete evidence, not a replacement for the analysis.

The original ghosts pass strict interior donor, no-recursion, mixed/upwind
coverage and polynomial reproduction checks, but the new cube weight audit at
N=24 reports:

| Degree | Max weight L1 norm | Max reflection mismatch L1 | Max xy-permutation mismatch L1 |
|---|---:|---:|---:|
| 2 | 10.3226 | 4.29900 | 4.29900 |
| 3 | 25.2123 | 6.65190 | 6.65190 |
| 4 | 64.4446 | 22.7282 | 22.7282 |

These are differences of transported donor-weight maps, not field errors.
They expose support/tie-selection anisotropy and amplification. The audit reports
these defects rather than asserting a nonexistent energy bound. No stability
proof or characteristic scri closure is claimed for this mask.

## Provenance, failed setup trials and scope

Large binaries and dumps remain under ignored `build-layer-*` directories;
only compact receipts and logs are committed. The first quick sweep used the
pre-diagnostic executable for the reference/small pulses and the added-failure-
diagnostic executable for subsequent cases. Both hashes are recorded per case.
The earlier root-level batch hash alone does not identify all cases. Final
control and long trials use immutable executable copies and their exact hashes.
The two diagnostic variants have the same evolution equations; the later one
reports the first invalid state.

The long Debug pytest development suite started before the final diagnostic
line wrapping and relink; subprocesses use the executable available at launch.
It is recorded as development sanitizer evidence rather than a single immutable
binary run. The clean-commit gate/control reruns above have unambiguous source
and executable identities. Exact intermediate source snapshots for the early
dirty-tree sweeps were not retained; their executable hashes and commands are
preserved, and the final implementation file hashes are recorded separately.

A first N=16 setup on [-1.05,1.05]^3 was rejected because the cell-centered patch
did not contain the sphere's complete halo. Enlarging it to [-1.1,1.1]^3 was
then rejected for missing interior normal-ray rectangles at degree three.
Those are boundary admission failures, not evolution failures. The long trial
was moved to N=24 with valid donor coverage; neither rejection was bypassed.

Aurora provenance scripts and mpi_tag_v3 compatibility patches were inspected
read-only. Their pinned AthenaK baseline e795b5f0674badfcc8743c13bc71c8873ce5047f
is different from this feature branch; the Kokkos commit matches. No Aurora
baseline was checked out in this tree, no production files were changed, and no
Aurora build or PBS job was run. The nonlinear/stability gates fail before the
requested GPU/MPI/binary extension stage, so an Aurora evolution would not yet
validate the intended formulation. There is no Aurora executable hash or ldd
claim for this work.
