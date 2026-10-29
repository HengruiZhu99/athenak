# Bounded Minkowski linear-operator audit

The frozen-volume **V2 surface subsystem has positive approximate eigenmodes** that satisfy the stated original-Jacobian residual check over a measured finite-difference amplitude plateau. The V3 surface calculation and both full coupled calculations remain **inconclusive** at the imposed Krylov limits. This audit does not identify the cause of the nonlinear duration failure or establish full-system, continuum, or finite-timestep stability.

All calculations were local, at n6 (plus preliminary surface-only n8 checks), with no nonlinear evolution or production source modification. Each Arnoldi worker was bounded to at most 390 seconds internally and 410 seconds externally; the longest completed case took 162.9 seconds. The largest Krylov basis used about 455 MB. No scheduler jobs were submitted. Libraries, large Krylov data and full coupled mode vectors are omitted; the small final surface mode vectors are retained for independent replay.

The final 512-vector cases give:

| Operator | Scheme | Leading candidate Re(lambda) | Fresh relative Jv residual | Assessment |
|---|---|---:|---:|---|
| Frozen-volume surface | V2, face-local angular jets | 7.55694659 | 4.63e-8 | Passes on amplitude plateau |
| Frozen-volume surface | V3, cross-face angular jets | 6.97783615 | 3.76e-5 | Unconverged |
| Full volume + surface | V2 | 27.33545694 | 4.87e-2 | Unconverged |
| Full volume + surface | V3 | 27.44929161 | 3.44e-3 | Unconverged |

Rates use coordinate time. The unconverged entries are **Ritz candidates, not measured eigenvalues**. In particular they must not be substituted for the fitted n16 duration growth rates. The tiny imaginary part of the leading V2 surface candidate is not interpreted as a resolved oscillation. A second V2 surface candidate near 7.44348141 passes the same selected-amplitude gate.

The adapter in `tst/hyperbolic_shift_3d/scri_linear_probe.cpp` calls the existing native equations. At every evaluation it restores all 31 native fields, embeds the perturbation, applies `ProjectAlgebraicNativeState`, reconstructs null lapse/shift values, and evaluates the existing surface rates and dependent D. The 19 independent surface coordinates are chi, five Frobenius-orthonormal tracefree metric components, five tracefree A components, P, Theta, Lambda[3], and B[3]. Lambda is independent; the weak regularity conditions are not imposed. Surface alpha, beta, D and J are dependent or fixed and are not independent perturbations.

For the frozen-volume operator, `BoundaryRates` reads an immutable analytic Minkowski active-volume array. It does not fill ghosts or evaluate the volume RHS. For the coupled operator, each volume point adds alpha, beta[3] and D[3], giving 26 coordinates. The existing `EvaluateRHS` supplies null closure, dependent D, actual ghost interpolation, mapped bulk derivatives, upwind transport, volume KO and surface feedback. Adaptive J remains zero with adaptive=false. Both variants use the same coefficients, radial nodes and penalty-CFL argument; only the angular-jet constructor flag changes between V2 and V3.

Precisely, the differentiated function is `T0 * RHS(P(U0 + E*z))`: E embeds the independent coordinates, P is the existing pointwise algebraic retraction, and T0 extracts the Minkowski tangent coordinates. The RHS includes the relevant native boundary reconstruction. T0 removes metric/A trace components at g=I,A=0. This is the stationary tangent generator associated with final algebraic projection, **not** the exact finite-dt accepted RK map. The background projected RHS maximum is approximately 1.1–1.2e-12. A variable tangent projection would have an additional term proportional to this nonzero roundoff residual; it is not represented as exactly zero.

`scri_linear_audit.py` uses central differences with a fixed maximum coordinate perturbation, two-pass orthogonalization and fresh original-Jv checks on selected modes. The predeclared gate is relative residual below 1e-7 and relative Jv change below 1e-4 when epsilon is divided by five. Negation/scaling checks are construction identities of this central-difference normalization; the independent additivity and epsilon checks provide substantive linearity evidence. The optional RK4 polynomial filters Krylov vectors only; it takes no nonlinear time steps, and filtered candidates are tested against the original J. The 120-vector filtered experiment did not resolve the coupled spectrum. All preliminary unconverged results are retained in `cases/`, with the corresponding earlier adapter versions in `preserved-adapters/`.

The final V2 surface mode was independently replayed at seven amplitudes with non-BLAS reductions. Its relative residual is 2.65e-8 at epsilon=1e-4, 2.70e-8 at 5e-5, 4.63e-8 at 1e-5 and 8.15e-8 at 5e-6. It rises to 1.82e-7 at 2e-6 and 4.00e-7 at 1e-6. Thus the result is supported by a finite amplitude plateau; it is not uniformly converged as epsilon tends to zero. The V3 residual stays about 3.76e-5 throughout the sweep. A separate direct-ctypes review using `math.fsum` confirms these results and verifies bitwise state reset after interleaved calls; see `independent-review.json`.

Norms and localization use **unweighted point-coordinate squared amplitudes**, with a Frobenius-orthonormal tensor basis, not physical area/volume energy. The leading resolved V2 surface mode has 97.9516% of its squared norm in A, 2.00347% in Lambda and 0.044964% in the metric. Its two-cell corner bands contain 99.9884% of the squared norm while occupying 44.4444% of rows. Its maximum lies on the -z face at x=(0.539163866, 0.539163866, -0.646996639). Full coupled candidates have shell/face/field localization in their reports, but their nonconvergence precludes interpreting them as established modes. V2's frozen-volume mode is not a demonstrated V3 or full-coupled instability.

NumPy 2.0.2 emitted divide/overflow/invalid warnings from `matmul`. A pure-NumPy random-matrix reproduction, before loading any native library, produces the same warnings while all outputs are finite. Its matrix/vector and Gram products agree bitwise with `einsum(optimize=False)` in this environment. The independent replay avoids BLAS for norms and Rayleigh contractions. Warnings remain in the captured execution records; they were not silently removed. `cases/surface-n6-final/independent-replay.json` records this check. Source/header hashes, actual commands, compiler output, shared-library identities, reports and timing records are retained.

Reproduce from the repository root after configuring the existing local Kokkos Serial build (fresh output paths required):

```sh
python3 tst/hyperbolic_shift_3d/scri_linear_audit.py \
  --output output/scri-surface-replay-new --operator surface --n 6 \
  --vectors 512 --epsilon 1e-5 --budget 390
python3 tst/hyperbolic_shift_3d/replay_scri_linear_modes.py \
  --campaign output/scri-surface-replay-new \
  --output output/scri-surface-replay-new/independent-replay.json
python3 tst/hyperbolic_shift_3d/scri_linear_audit.py \
  --output output/scri-coupled-replay-new --operator coupled --n 6 \
  --vectors 512 --epsilon 1e-5 --budget 390
```

A positive frozen-volume eigenvalue diagnoses that subsystem. It does not prove the coupled eigenvalue has the same sign, and a failure to find positive modes cannot clear the coupled scheme or exclude nonnormal transient growth. Further numerical changes require their own controlled validation.
