# N16 approximate continuous mode versus final-only native RK3

The original C0 spatial-norm candidate direction amplifies under the exact cached final-only RK3 tangent map at the nominal pole timestep and its halves/quarters. The direction is **not an established eigenvector of that finite-step map**: intermediate algebraic-normal feedback changes it by more than its original projected-generator residual. This experiment does not certify any generator or timestep-map eigenvalue, a globally rightmost mode, a physical energy instability, nonlinear growth, or a continuum instability.

The read-only trace-control report review found no correction: `inner-trace-native/rejected-controls-report.md` agrees with the frozen trace-family v2 tables, direct combined-alpha replacement, 3872-entry attribution, and stated limitations. No earlier source, array, or receipt was modified.

## Pinned input and map

This is the original N16/span2.2 Cartesian sphere, 1640 active points, 32800 free20 coordinates, 36080 raw22 coordinates, a=.5, S=1, geometry transition (.05,.95), kappa1 input10, C0/kappa2=0, physical-P lapse, private spatial-norm shift, symmetric quadratic ray ghosts, and native KO=.1. The free20 order is point-major in original k/j/i active-cell traversal. The minimum Omega is .0026953124999994555 and h=.1375. The nominal **pole** timestep is `.03*Omega_min = 8.085937499998367e-5`; this experiment does not launch a new native evolution or infer a measured run timestep.

The original cached raw22 matrix A, pointwise lift L, and restriction P are reused without recomputation for the main action:

\[
B_{dt}=P\,[I+dt A+\tfrac12dt^2 A^2+\tfrac16dt^3 A^3]L,
\qquad J=P A L.
\]

The no-dyngr native SSPRK3 lifecycle permits raw22 algebraic-normal components at intermediate stages and projects only at the final stage. The projected-continuous generator J instead restricts every infinitesimal RHS. Its RK3 polynomial is generally different from Bdt. The raw22 cache retains its original fourth-order centered local finite-difference Jacobian approximation (local epsilon1e-4); “exact cached map” refers to applying the stated polynomial of that cache, not an exact symbolic derivative of the nonlinear code.

The literature agent exported four complex, unit-Euclidean free20 directions from independent reduced-history constructions. Their phase-aligned distances are at most1.667e-7. Each retains the explicit converged-approximate/pseudospectral label. The primary candidate has

\[
\lambda=1.916338996568537+6.790995420746044i,
\qquad \|Jv-\lambda v\|_2/\|v\|_2=5.886773215552094e-7.
\]

The other three actual cached-generator residuals range3.769e-7 to5.867e-7. No residual is divided by the large matrix norm. Small residuals of this nonnormal operator do not give an eigenvalue error bound. Candidate vectors are pinned to `fc88b20d4953f5088aed97d04dce41ad0af039fc794a401a21cceb53280e1eee`; their metadata is pinned to `e411042b90498bb4f08eadcdf9214095028c049df8a0c15b46b5dfd4d3f2d6e1`.

## Primary candidate result

Define `mu = v* Bdt v/(v*v)`. The effective growth and frequency below are `Re(log(mu))/dt` and `Im(log(mu))/dt`, using the principal logarithm. They are directional Rayleigh measurements, **not accepted modal eigenvalues**. The residual is `||Bdt v-mu v||2/||v||2` in state units. The code evaluates the increment and `log1p(mu-1)` to avoid subtractive loss near the identity.

| Timestep factor | dt | abs(mu) | Effective growth | Effective frequency | State residual | Residual/dt |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.085937500e-5 | 1.000155997158 | 1.929089750 | 6.805612911 | 3.573526334e-6 | .04419433534 |
| 1/2 | 4.042968750e-5 | 1.000077755005 | 1.923140827 | 6.798646085 | 9.658107913e-7 | .02388865339 |
| 1/4 | 2.021484375e-5 | 1.000038810150 | 1.919846503 | 6.794906286 | 2.508068700e-7 | .01240706449 |

All four candidates give growth within3.04e-7 and frequency within4.01e-7 of the primary candidate at each timestep. Their residual norms agree to3.2e-12 or better. These are correlated approximations to the same direction, not four independent discoveries of eigenvalues.

For the projected20 RK3 polynomial, the primary candidate's residual/dt remains approximately5.887e-7 at all three steps and its effective growth/frequency agrees with the continuous lambda to about6.1e-11. In contrast, `||Bdt v-R3(dt J)v||2/||v||2` is3.902675514e-6,1.050768156e-6,2.723641490e-7. The normal-stage feedback coefficient norms are

\[
\|(I-LP)ALv\|_2=.6466042208,
\quad\|(PA^2L-J^2)v\|_2=1381.167927,
\quad\|(PA^3L-J^3)v\|_2=8587761.979.
\]

Thus the difference starts at dt squared (with a substantial cubic contribution at the nominal step), and its per-time effect decreases toward zero at fixed grid as dt decreases. This rules out equating the original projected-continuous approximate pair with an exact finite-step mode. It does **not** determine the spectrum of Bdt. The one-step Euclidean norm amplification is positive for this direction, but that coordinate norm is not a physical energy or a proved symmetrizer norm.

## Independent actual one-step derivative and source identity

The original byte-pinned full22 executable was invoked with a fresh export prefix. Its actual nonlinear `NativeStep` projects the initial perturbation, evaluates three actual CartesianPatch RHS stages, and projects only the final state. Real and imaginary parts were differentiated separately using centered maximum-free-component perturbation sizes1e-3,1e-4,1e-5. All three timestep factors were tested; this is one-step differentiation, with no long evolution.

At epsilon1e-4, the combined complex state L2 discrepancy from the cached final-only map is1.798139451e-11,1.663017055e-11,1.661942492e-11. The corresponding directional growth estimates differ from the cached results by at most4.71e-9. The larger/smaller perturbation sizes expose truncation/cancellation: state errors across the full sweep range1.66e-11 to4.08e-10. The finite-step mode residual3.57e-6 to2.51e-7 is therefore resolved by this direct native check. This test alone does not resolve the continuous candidate's much smaller generator residual or furnish a certified matrix-error bound.

The newly exported raw22 CSR data/indices/indptr and L/P bytes are exactly identical to the original exports. The original executable is `bc4f4c62e4fa3e286da4c19f11fa79ba73a8ed3658f4fa42646d7dca7bf5a943`. Its original compiler command, dependencies, archives, source snapshots, and projection lifecycle are retained and reverified by the freeze script; public implementation identity remains27c19d20696ea6dd4704032c51dfd026218f64f2 plus the original explicit spatial-norm overlay. No new C++ build or physics/source change was made.

The main driver and native one-step audit completed in4.4573s, excluding input hashing/matrix loading before its internal timer. Results include all four candidates, complex scalars, norms, actual FD sweeps, cache identity checks, runtime versions, commands, and input/source hashes. Large raw matrices and output vectors remain local with metadata hashes in the frozen bundle. Reproduce from the repository root using:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=build-layer-research/boundary/python-deps python3 build-layer-research/boundary/full-tensor-mode-finite-step-20261009/check_mode.py
```

Use a fresh output directory/prefix when rerunning; do not overwrite this frozen result or the exported candidate evidence.
