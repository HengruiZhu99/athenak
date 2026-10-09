# Actual C0 spatial-norm local angular gate

The scoped local bridge and total-J angular gates pass. The actual continuum full-tensor C0 RHS with the frozen physical-P/spatial-norm gauge preserves the tested J=0,1,2 subspaces to floating-point accuracy at finite Omega. This is an angular coefficient-action result, not a radial discretization, boundary prescription, evolution or stability result.

The root release is `0c4d415d79e6bb50acfe6f0d6262c4bcda263a537b98221282931cede87c742c`; the independently reviewed immutable basis index is `414f241e986e0c46b7d94820fb060b704166d973284854c53d2e8c64ee6c489e`. The held preparation/admission/receipt bytes were copied into this fresh tree and remain unchanged at their original paths. The runtime implementation remains production commit `27c19d20696ea6dd4704032c51dfd026218f64f2`. Launch documentation HEADs are recorded separately in the build receipts.

## Binding and local gates

Parameters are S=1,a=.5, geometry (.05,.95), gauge (.45,.85), physical_trace_lapse=true, preferred_source=false, xi=2, eta=6, C=(1-1/1.5)/.5, kappa_input=10 and kappa2=0. No C1, Q, lapse/trace repair or damping profile is included. The generic dual Reference/Gauge functions are extracted exactly from the pinned frozen full20 source with form=0/norm=true. Geometry calls the production `ConformalRHS` and `AssembleInterior` on full analytic jets; gauge beta pole/Omega is added exactly once. The fixed constant-10 analytic Minkowski residual subtraction matches the public Cartesian patch and has zero variation.

The actual private native wrapper dispatches spatial-norm feedback only for double. Its dual fallback is therefore unsuitable as the dual binding. The gate compares the exact generic-dual binding against directional finite differences of the actual double wrapper at 200 full22 response cases (20 free columns, reference and finite algebraically constrained SPD states, five radii) and five amplitudes. The omitted-feedback negative control has a nonzero derivative discrepancy of 31.57894.

| Local check | Observed maximum |
| --- | ---: |
| Reference source-subtracted RHS, 14 radii | 7.11e-15 |
| Reference physical constraints | 3.08e-14 |
| Double/dual response, best amplitude per case | 1.20e-10 scaled |
| Double/dual response, every tested amplitude | 3.87e-8 scaled |
| Raw22 input / output algebraic normals | 9.76e-17 / 1.56e-15 scaled |
| Metric first/second and A first tangent identities | 1.90e-15 scaled |
| Native point-projector formula derivative | 1.94e-11 scaled |
| Independent native coordinate-zz chart identity | 1.12e-16 scaled |
| Actual reference coefficient-map jet FD, finest step | 7.14e-9 scaled |
| Independent solid-harmonic Laplacian identity | 2.25e-15 scaled |
| Core TT hxy=z²,Axy=z RHS/constraints oracle | exactly 0 error |
| Manufactured pure-gauge initial constraints | exactly 0 |
| Coefficient-aware pure-gauge Cdot, finest step | 2.15e-7 scaled |

The physical metric lift differentiates every supplied analytic reference coefficient through second order. The A lift includes the nonzero Aref-up:delta-g term and all consumed first derivatives. It sets neither Lambda=Gamma nor a differential constraint. The point-projector adapter reproduces the public per-point algebraic formula; this test invokes neither mesh ghosts nor a finite-RK step. Its output-normal formula is scoped to this stationary reference, whose source-subtracted geometric RHS is zero.

The Cdot check spatially differentiates the complete actual tangent RHS, including background coefficients, before applying the physical constraint derivative. It is not a frozen-symbol QL comparison. Steps are .001,.0005,.00025. At r=.98 the largest absolute H residuals for lapse/shift seeds fall from .00862/.01021 to 3.34e-5/3.96e-5, approximately fourth order, while their scaled finest residuals are below 2.15e-7. Other components/points reach cancellation floors; no uniform fourth-order claim is made after that floor. All eight diagnostics use physical H, signed Mcov, signed Zcov and physical Theta without Omega rescaling.

Release and ASan/UBSan Debug pass every declared local gate. Their full numerical JSON differs in 48 entries; the largest absolute/scaled difference is 4.32e-10 in a finite-difference Cdot cancellation residual. No byte-equality claim is made. Their runs take .49 s and 6.71 s; builds take 1.40 s and .89 s. Executables are `8e9ae32418150c4350cde7d8beca78f014aea74e1c83e046427feef157371562` and `9925dff649ba5795b1d0b087734215dc1e78d1d617ded537b591ac27f3f61bbe`. Build receipts capture compiler/flags, 1055/1057 dependency hashes and four link-archive hashes; production dependencies were checked against runtime27. An initial bookkeeping freeze incorrectly required byte equality; that failed assertion and the prior report/script are preserved, and the correction required no scientific rerun.

A separate independent flat-core Cartesian formula oracle includes all full22 RHS rows and all eight physical diagnostics. It tests 1,404 cases per build: three core points including the origin, J0/J1/J2, all nonnegative m with both real/imaginary parts where applicable, every channel and three radial-envelope jet actions. Negative m is supplied by the frozen conjugacy relation, not separately queried; the raw historical boolean `includes_origin_and_all_m` refers to that representation. RHS scaled error is 1.67e-16 and physical8 constraint error 1.29e-16 in Release and Debug. The flat oracle explicitly includes alpha_t=-3P, beta_t=3Lambda/8 and kappa10; it is separate from fitting the actual kernel's output. This supplementary executable includes the unchanged bridge source and does not replace or rerun the angular batch.

## Angular coefficient action

There are 13 positive radii from .025 through the original N16 closest shell, r=sqrt(.9973046875), with minimum Omega=.0026953125. J0/J1/J2 contain 8/16/20 independent amplitudes. For each amplitude the independently supplied local envelope jets are (W,W_rho,W_rhorho)=(1,0,0),(0,1,0),(0,0,1), rho=r². Thus J2 has 60 input actions, and each of B0/B1/B2 is a 20x20 matrix. The saved coefficients represent Wdot=B0 W+B1 W_rho+B2 W_rhorho, in the immutable channel layout.

Twelve oblique fit directions are stacked in raw native22 Cartesian state space and solved with column-scaled SVD. Eight unused directions, independent m=1 for J1 and m=1,2 for J2, and four paired directions under a fixed proper rotation validate the same coefficients. The all-m polynomial data already includes its CG phase; it is not applied twice. A separate m0 evaluation from all-m data agrees with the primary generated header.

| Angular check | Worst scaled error |
| --- | ---: |
| Fit | 1.4061e-13 |
| Unused angles | 2.0295e-13 |
| Independent m | 1.8664e-13 |
| Rotated input and RHS | 2.1470e-13 |
| Primary m0 vs all-m evaluator | 5.8292e-16 |
| Raw22 input / output normals | 2.52e-16 / 1.03e-13 |

All 39 fits have full declared rank. The maximum column-scaled condition is 3.43164. The maximum unscaled condition is 3.16742e6, reflecting the declared solid-harmonic r^L column magnitudes; both unscaled and scaled values and all scaling factors are retained. No singular values were dropped and no normal equations were used. The 96,720-row actual-kernel batch takes 2.697 s. Raw actions and query/executable/source hashes remain available locally; the compact freeze retains their metadata and the small B0/B1/B2 coefficient archive.

The first NumPy2 analysis emitted floating-status RuntimeWarnings in ordinary matmul despite finite outputs. Its exact script, finite report and coefficient archive are preserved under `history/first-analysis-matmul-warnings`; the warning kinds and source lines are recorded there. The final analysis uses explicit einsum contractions, treats RuntimeWarnings as errors and has empty stderr. It reuses exactly the same saved kernel batch and gives coefficient matrices identical to the first finite calculation. The warning analysis is not the accepted final receipt. No scientific tolerance was changed in response to a failure.

## Limits and reproduction

At the origin, solid powers make the value-fit matrix rank-deficient. The origin is validated through Cartesian polynomial and full-kernel core oracles; no positive-radius fit is silently interpreted as an origin equation. No radial representation, origin/scri closure, finite-RK map, eigenvalue, propagator or native evolution was produced. All PDE evaluation is strictly at positive Omega. Generic Theta data at scri and a complete constraint-preserving characteristic closure remain unaddressed.

The actual continuum equations with spherical coefficients pass this tested angular closure. Cartesian independent Dxx, mixed DxDx, Lx and KO remain anisotropic finite-h operators and are absent from this continuum angular calculation. A later radial control changes the bulk discretization too and cannot uniquely attribute the observed Cartesian mode to the primitive ghost extension.

Reproduce the fresh-tree stages with `python3 prepare.py`, `python3 build.py release`, `python3 build.py debug`, then `python3 check_local.py release` and `python3 check_local.py debug`. Run `run_angular.py` with the recorded NumPy/SciPy environment and OPENBLAS_NUM_THREADS=1 only after both local gates pass; its source uses no eigensolve or evolution. `run_core_oracle.py` runs the separate supplementary oracle. Existing attempt directories are retained; the saved angular batch is reused when its pins match. Exact commands, source versions, input pins, compiler dependencies and prior warning analysis are in the receipts.
