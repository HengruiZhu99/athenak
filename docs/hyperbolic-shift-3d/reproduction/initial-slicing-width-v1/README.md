# Initial slicing width/profile study, 2026-11-02

This is a local t=0 test, not an evolution or a change to the outer restoring collar. Original initial data already approach fourth-order Hamiltonian accuracy in the bounded fine radial control (N192→256: 3.95); the N16–N32 initial slice is underresolved. The original slicing switch is C-infinity. Its narrow higher-derivative structure, rather than insufficient formal differentiability, motivates this test.

The strongest tested balanced candidate uses the C6 polynomial switch P(z)=I_z(7,7), P'(z)=12012 z^6(1-z)^6, over areal radius R=4M–12M, with exact plateaus outside. It changes the Schwarzschild slicing and compact radial map, not mass or the constraints being solved. Original A initialization and collar remain the defaults. No evolved checkpoint is interpolated.

Full production 3D initial operators, including patch exchange and scri closure, at fixed exterior cutoff r>.063 give:

| Profile | H orders 16→24 / 24→32 | M orders 16→24 / 24→32 |
|---|---:|---:|
| Original C-infinity, 4–8M | 1.505 / 1.980 | 3.195 / 3.493 |
| C-infinity, 4–12M | 2.804 / 3.851 | 4.444 / 3.602 |
| C6 polynomial, 4–8M | 3.591 / 3.697 | 3.966 / 4.013 |
| C6 polynomial, 4–12M | 3.441 / 3.715 | 3.837 / 4.036 |

The last candidate continues at N32→48 with H order3.836 and M order3.973. Its N32 H/M RMS are .0012197651/.0012707135, 5.83/3.86 times smaller than original initial data. Widening alone or changing shape alone can increase N16 errors even when effective orders improve; both amplitudes and orders are retained. No claim that N16 is fully asymptotic is made.

Independent analytic jets give exterior continuum H/M residuals below3.2e-14. Native fields agree with the independent Python reference within7.2e-12; tightening native quadrature from2e-13 to2e-14 changes dense radial norms by less than6.4e-7 relative. Independent norm-grid1024→2048 changes them by less than8.7e-6. Gauge-only lapse/shift/normal changes leave constraints exactly unchanged. Default native fields are byte-identical to the prior initializer output.

The initial horizon changes from .06292449 to .05102371, both inside the fixed .063 cutoff. The polynomial4–12M compact transition is .297388–.780536. Exact-interior-halo substitution at N16 changes its H/M norms by+3.26%/+.65%; unlike the original middle-shell transition, this broader profile crosses a radial interface. Algebraic projection changes the norms only at roundoff. These controls retain boundary closures and are not a claim of eliminating all interface errors. Raw regions, maxima, radial bins, and cutoff sensitivity are in each patch JSON.

Dense radial controls use a fixed .3<r<.95 norm region and actual shell spacings, with analytic angular derivatives and no patch exchange. That region omits the beginning of the widest transitions; these norms are not interchangeable with full 3D exterior norms. The independent shape/width screening covers bump4–8/10/12/16/24M and polynomial4–8/12/16M. The full 3D four-profile comparison avoids attributing a joint profile/width change to width alone.

## Reproduce

Configure the separate build with Release, MPI ON, and Athena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS ON. Build targets hyperbolic_shift_3d_initial_smoothing, hyperbolic_shift_3d_initial_patch_audit, hyperbolic_shift_3d_native_wormhole, and hyperbolic_shift_3d_initial_continuum. Exact commands are in configure/build logs and *commands.json. Commands refuse existing output files. `run_radial.py` runs the width sweep, `controls.py` validates quadrature/continuum/default preservation, and `analyze.py` reproduces initial-convergence.png and summary.json. Baseline raw patches remain in ../native-shared-transition-v1/checks/initial-grid.

This establishes an earlier approach to the initial-data convergence regime. It does not establish evolved convergence or identify the outer restoring collar as the cause of late-time errors. The subsequent A wider-collar experiment deliberately retains A's original initial slice to isolate evolution gauge effects.
