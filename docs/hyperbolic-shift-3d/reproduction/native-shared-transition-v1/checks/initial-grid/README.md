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

`radial-profile-n*.csv` gives dense signed H/Mr and momentum-magnitude profiles and `radial-fixed2048.csv` includes maxima and radii for this radial control. The subsequent explicit 3D controls below now report maximum-error locations, exact-interior-halo/projection effects and cutoff-volume sensitivity separately; the radial control alone is not their evidence.

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

## Explicit patch, projection, and cutoff controls

`test_initial_patch_audit.cpp` now measures the actual SG1 composite operator with native initial data at N16/N24/N32. Its production norms reproduce the separate actual MPI checkpoint diagnostic to the printed precision. It records full 3D maximum values/Cartesian coordinates/block IDs, radial bins, and core/overlap/seam/initial-transition/scri-row/outer-collar regions. Regions overlap: their squared fractions must not be summed. The seam band is two angular cells and changes physical width with resolution.

The exact-interior-halo control replaces halo values at r<1 with direct native values at those halo coordinates. It retains the outer ghost lift and the direct scri radial plans' own cross-face donor interpolation, so it must not be described as removing every boundary approximation. All scri/halo closure steps run before diagnostics, including dependent D, even though D does not enter H/M.

| N | Relative H norm change with exact interior halos | Relative M norm change | Largest projection field change |
|---|---:|---:|---:|
| 16 | +9.98e-8 | -6.31e-7 | 1.33e-15 |
| 24 | +1.31e-8 | -2.56e-8 | 1.33e-15 |
| 32 | +3.50e-9 | -3.72e-9 | 1.78e-15 |

The norm of the **difference field**, a distinct quantity from the change of a norm, is also preserved. For H it is1.30e-6/2.50e-7/7.31e-8; for momentum magnitude5.75e-6/1.19e-7/1.12e-8. Projection rebuilds all derivatives from projected values; its constraint differences remain at roundoff. Native scri constraint-bearing fields remain exactly unchanged by closure/projection. These initial-data results do not predict the size of evolved nonspherical interpolation errors.

More than99.995% of exterior H squared norm and99.997% of M squared norm is in r=.45–.75 at every tested resolution. Full 3D Hamiltonian maxima occur at radii .665625/.65625/.6609375, with absolute values .1081043/.0876852/.0433895. Momentum maxima are .2408191/.0802525/.0390931, at .684375/.65625/.6609375. Exact Cartesian locations are in `patch-summary.json` and raw reports.

Cutoffs .062/.063/.064 independently test cell inclusion; .062 is a sensitivity control slightly inside the analytic horizon, not the claimed exterior norm. The reported exterior cutoff remains .063. Its measured coordinate-volume excess over exact4*pi*(1-cutoff^3)/3 is .04389%/.02189%/.01207%. Normalizing by exact volume changes RMS norms by at most .022%. Moving the cutoff across that interval changes the original norm by less than1.5e-5 relative. Neither effect explains the observed low initial H order. This complements the independent dense radial norm quadrature already reported above, which exposes much larger peak undersampling.

`patch-n*.json` are raw controls, `patch-summary.json` retains exact maxima, differences and cutoff values, and `patch-controls.png` shows localization and exchange differences. Empty radial bins and exact zero residuals are omitted from logarithmic plots; they are never interpolated. Build `hyperbolic_shift_3d_initial_patch_audit`, run `N fresh_output.json` for16,24,32, then `python3 .../checks/initial-grid/analyze_patches.py`. `patch-binding.json` records source/binary/flag and raw artifact hashes. The audit changes diagnostics only and has no checkpoint/evolution output.
