# Native initial-data audit (t = 0 only)

These results are separate from the evolved SG1 N16/N24/N32 histories. No time stepping occurred. The actual MPI executable initialized native volume coordinates and scri limits and wrote admitted NI1 checkpoints. The matching native diagnostic loaded those checkpoints and used the production halo/scri closure and volume constraint kernels. All three exterior norms use r > 0.063, outside the analytic initial horizon r = 0.0629244910.

| N | H RMS | Momentum magnitude RMS | Radial momentum RMS |
|---|---:|---:|---:|
| 16 | 0.02313650 | 0.04893855 | 0.04920565 |
| 24 | 0.01256860 | 0.01339935 | 0.01349526 |
| 32 | 0.00711020 | 0.00490590 | 0.00493861 |

Hamiltonian effective orders are 1.505 and 1.980; momentum magnitude orders are 3.195 and 3.493. Analytic/AD exterior residuals using the same native constraint formula are below 2.6e-14, and changing lapse/shift leaves H and M exactly unchanged. Thus the slow initial Hamiltonian trend is a discretization issue, not a continuum constraint violation or a shift-initialization error.

## Localization and independent radial control

`n*-radial.json` contains cumulative weighted squared errors at physical cutoffs. Nearly all initial exterior error lies in r=.45–.75, with peaks near r=.66–.69. This is the initial slicing transition, not the outer restoration collar. The latter's late-time errors still need their own investigation.

`test_initial_radial_refinement.cpp` uses actual middle-shell radial cell centers and FD4 radial derivatives; angular tensor derivatives are analytic. It has no patch exchange. Its r=.45–.75 H norms agree with the full production transition-band norms to 1.1e-7 relative, and M/Mr to 1.1e-5. This strongly identifies radial finite differencing as the dominant initial error in that band; it does not rule out smaller seam or boundary defects elsewhere.

Bounded radial refinement to N256 gives Hamiltonian order 3.95 for N192→256. Momentum orders are nonmonotone (3.59 for the last pair); no single asymptotic momentum order is claimed. These are radial operator tests, not 3D evolutions at those resolutions.

The fixed1024/fixed2048 controls evaluate the FD residual at integration points independent of h, using native fields at each shifted stencil point, with no interpolation of a solution. Changing the integration grid changes norms by at most 1e-6 relative. Coarse production radial centers undersample this continuous residual: H norms are lower by 64%, 43%, and 23% at N16/24/32. The dense quadrature is a diagnostic of the truncation-error function, not a replacement value for the production discrete norm. Both are retained.

`radial-profile-n*.csv` gives dense signed H/Mr and momentum-magnitude profiles and `radial-fixed2048.csv` includes maxima and radii for this radial control. Full 3D maximum-error locations and explicit exact-halo/projection/cutoff-volume controls remain outstanding; the radial control must not be labeled as their completion.

## Reproduction

Run from the repository root. A fresh local serial Kokkos/MPI build preserves existing frozen builds:

```sh
cmake -S . -B build-native-shared-transition -DCMAKE_BUILD_TYPE=Release -DAthena_ENABLE_MPI=ON -DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON
cmake --build build-native-shared-transition --target hyperbolic_shift_3d_mpi_native_control hyperbolic_shift_3d_checkpoint_constraints_native_control hyperbolic_shift_3d_native_wormhole hyperbolic_shift_3d_initial_continuum hyperbolic_shift_3d_initial_radial_refinement -j8
```

For each N=16/24/32 use dt=.0005/(N/16), a fresh output prefix, and target coordinate time zero:

```sh
mpiexec -n 1 build-native-shared-transition/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_native_control 16 output/native-shared-transition/initial-n16/puncture puncture 0 .0005 - - 600 .05 .001 12 24 2
python3 tst/hyperbolic_shift_3d/run_native_initial_diagnostics.py
```

The collector requires all three runs and refuses existing diagnostic output. Per-resolution commands/checkpoint SHA256 and lengths are in `n*-provenance.json`; small native reports/histories are preserved here. Large checkpoints stay in ignored output storage.

The independent pointwise and continuum tests are:

```sh
python3 tst/hyperbolic_shift_3d/validate_native_wormhole.py --executable build-native-shared-transition/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_native_wormhole --output docs/hyperbolic-shift-3d/reproduction/native-shared-transition-v1/checks/initializer
python3 tst/hyperbolic_shift_3d/validate_initial_continuum.py --executable build-native-shared-transition/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_initial_continuum --output docs/hyperbolic-shift-3d/reproduction/native-shared-transition-v1/checks/initializer
```

The latter script uses independent analytic Python jets only for validation. Radial tests take optional integration-point count (zero uses production centers) and optional profile-output prefix. Run counts 0,1024,2048, preserving `radial-refinement.csv`, `radial-fixed1024.csv`, `radial-fixed2048.csv`; use `radial-profile` prefix for the 2048 run. Then:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/native-shared-transition-v1/checks/initial-grid/analyze.py
```

No local audit result authorizes a claim of long-time convergence. Full scri consistency and controlled N16 evolutions remain required.
