# Native directional replay of early V4 growth

The actual n16 V4 trajectory has a persistent growing direction on which the native tangent RHS projects at approximately **73 per unit coordinate time**. Its degree-7 angular filter acts only weakly on that direction. A controlled, same-node **surface term-action comparison** identifies the normal derivative of the complete pole numerator as a useful next discretization control. This study does not implement a replacement closure, compute a converged eigenmode, run an evolution, or establish stability.

The three fresh epsilon2 captures at `.25/.30/.35` are independently source/binary/worker-bound by the [acquisition verification](../aurora/linear-window-dissipative-v2/acquisition/README.md). Their native checkpoint flags are true because each requested capture completed; they are **scientifically unadmitted**. The [early-direction figure](../early-direction-figure-v1/README.md) shows their angular-filter response and the opposite-sign nearby radial amplitudes. The previous [saved-state response audit](../scri-collar-response-v1/README.md) explains why the unresolved n6 Ritz vector and terminal nonlinear states were insufficient to choose a collar.

## Measured trajectory directions

| Capture time | Volume maximum deviation | Surface maximum deviation | Surface `||Bv||M/||v||M` | Isolated surface `γ₂` | Surface `l<=6` fraction |
|---|---:|---:|---:|---:|---:|
| .25 | 1.4450e-7 | 8.0767e-7 | .173106 | .958897 | .498339 |
| .30 | 5.3594e-6 | 3.0006e-5 | .141389 | .639705 | .523574 |
| .35 | 2.0670e-4 | 1.1499e-3 | .120799 | .466959 | .544655 |

Here `γ₂=-Re(v*M Q₂v)/(v*Mv)` is the angular filter's isolated energy quotient with positive reference-area weights, not a growth-rate shift. Angular degree refers to each Cartesian/Frobenius component. The pattern becomes less responsive to the filter through this interval, while remaining more angularly structured than a simple `l<=2` perturbation. The hypothetical same-B collar quotients also decrease, `.963865/.630596/.453505`; extending radial support would not remove the filter's weak action on these directions.

The unweighted full-coordinate correlations are `.998238` for `.25→.30`, `.998974` for `.30→.35`, and `.994587` for `.25→.35`. Best-fit amplitude growth rates are `72.8684/72.9842/72.8999`. The full-interval shape residual is **10.39%**, so this is not an exactly fixed eigenvector. Separate coordinate-volume and reference-area weighting give end-to-end correlations `.994549/.995045` and growth `72.8565/72.8880`; the alignment is not an equal-point-weight artifact.

The tangent coordinates use five Frobenius-orthonormal tracefree components for each metric/A tensor, 19 independent surface coordinates, and 26 volume coordinates. Raw traces discarded by that extraction have relative norm `1.51e-9/2.01e-8/7.58e-7`. Reapplying the **native** algebraic retraction recovers the checkpoints' independent fields with relative errors `1.55e-9/4.62e-11/4.66e-12`, respectively. Thus the raw nonlinear trace at `.35` is largely the algebraic retraction's expected curvature, rather than a missing independent coordinate. These comparisons exclude dependent surface lapse/shift/D and fixed J from the independent-state norm.

## Full native directional action

The thin new adapter evaluates

`T0 RHS(P(U0 + E z))`,

where `U0` is Minkowski, `E` embeds the existing 19/26 coordinates, `P` is the native standard algebraic retraction, and `T0` extracts the tangent coordinates at Minkowski. It then differentiates centrally along the **measured checkpoint direction**. This is the stationary tangent generator, not the accepted finite-dt RK map and not a Jacobian at the nonlinear checkpoint. Every call resets all 31 volume/surface fields, reconstructs actual null data, evaluates native dependent D and ghost exchange, and calls the complete native bulk RHS. The production equations and closure are unchanged. The adapter matches n16, cross-face angular jets, epsilon2, degree7, and `dt=.0005`; the old n6–8 probe is preserved.

For the `.30` direction at maximum coordinate amplitude `1e-5`:

| Action | Whole-direction projection | Surface projection | Whole relative shape residual |
|---|---:|---:|---:|
| Actual V4 tangent RHS | 72.96352870 | 72.96472155 | .0140232 |
| Epsilon0 on the same input | 73.54496325 | 73.57462132 | .0260734 |
| V4 minus epsilon0 | −.58143455 | −.60989977 | .977877 |

The epsilon0 column is a counterfactual **action** on the V4 direction, not a V3 trajectory or an eigenvalue shift. This full-action comparison includes the changed dependent-D/ghost feedback; its volume directional projection difference is only `+.00015038`. Across amplitudes `1e-4,3e-5,1e-5,3e-6`, the actual action changes by at most `2.28e-7` relative to the `1e-5` result. The directional projection changes by about `2e-8`. No large eigensolve was performed.

## Native surface decomposition

The diagnostic export assembles six contributions using the same native sample jets, `ConformalRHS`, `HyperbolicDamping`, `TensorPoleNormalJet`, `TensorScriLimit`, `ScriRegularityTraces`, angular plan and gauge flux. It checks their sum against unmodified `BoundaryRates` before finite differencing, then checks the differentiated sum and the full coupled RHS's surface slice. At `.30`, amplitude `1e-5`, the surface projections are:

| Contribution | Projected rate |
|---|---:|
| Regular R, including native damping preparation | −53.44747827 |
| `−AD_normal N` | +110.98052634 |
| Weak `4N/h` | +7.95352750 |
| Regularity `−8C/h` | +8.08804574 |
| Angular Q, including B source | −.60989977 |
| Remaining B flux | approximately 0 |
| Sum | **+72.96472155** |

The regularity contribution is positive **along this coupled direction**; its formal minus sign does not imply it is an isolated `−u` damping operator. The normal contribution dominates the positive projection. This surface decomposition excludes induced D and bulk feedback and does not identify eigenvalue shifts from individual terms.

The maximum native sum error is `2.60e-18`; differentiated sum relative errors are about `2e-16`; the decomposed full-surface action equals the full coupled generator's surface slice exactly in the recorded check. All four perturbation amplitudes agree to the stated precision.

## Same-node complete-numerator factorization control

The second export holds the radial layout, all penalties, angular scheme and Q fixed. For each actual surface row `(block,j,k)`, it samples the **complete post-damping geometric numerator N** at active radial indices `n−5…n−1` and the actual endpoint. The five active points use the exact native bulk first-jet derivative policy, actual Cartesian coordinates, and `Omega(...,false)`; the endpoint uses the native surface jet and `Omega(...,true)`. It applies the six-node endpoint derivative with offsets `[-4.5,-3.5,-2.5,-1.5,-.5,0]h`. No interior field is null-projected or overwritten; Lambda remains independent. Derivative weights use the same polynomial-Jet machinery as the surface construction.

| Surface action at `.30`, amplitude `1e-5` | Unweighted projection | Reference-area projection |
|---|---:|---:|
| Native `−AD_normal N` | +110.98052634 | +110.94813148 |
| `−D_endpoint(N_samples)` | +10.99274017 | +10.96770219 |
| Difference | −99.98778618 | −99.98042929 |
| Native full surface action | +72.96472155 | +72.94615936 |
| Other surface terms held fixed, sampled-N term substituted algebraically | **−27.02306463** | **−27.03426993** |

The last row is an **algebraic surface term-action sum**, not an implemented counterfactual coupled RHS. Its negative projection is not a stability proof. The normal replacement has not been propagated through qdot/speed_t/dependent D, refreshed ghosts, or bulk shift/auxiliary evolution. Those operations must be rebuilt in that order for any actual coupled candidate.

Both volume and manually stored dependent surface D were perturbed while fixing all geometric inputs. Sampled numerators and both normal actions remained exactly equal; every preliminary padded field used by `Z4cJet` remained exactly unchanged. This validates D-independence for this diagnostic, not permission to skip final D reclosure in an evolution. Interior Minkowski N is nonzero: the retained checks verify `N_chi=r²−1` and `N_P=N_Theta=1.5(1−r⁴)`, not zero numerators. Their maximum error is `8.88e-16`; the sampled and AD normal background actions agree to `5.96e-13`. Degree-zero-through-five endpoint-weight moments agree to `4.26e-14`; the endpoint diagonal is `1126/315`. The sampled directional action changes by at most `7.49e-7` over the amplitude sweep, and the native AD component matches the six-term export exactly.

This is evidence to study a **controlled normal-factorization discretization**, not to adopt a collar or silently replace the closure. Before a coupled implementation, test manufactured radial/angular jet consistency including seams, complete dependent-D/null reclosure, and any resulting CFL restriction. Exact smooth N receives a fifth-order endpoint derivative here, but fourth-order errors in the interior first jets can conservatively become third order after another differentiation unless their coefficients cancel suitably. The Minkowski polynomial check alone cannot establish general fourth-order consistency. A repaired short trajectory would still require Minkowski duration, puncture reproduction and convergence against the 1D baseline.

## Source binding and reproduction

The implementation consists only of new files `tst/hyperbolic_shift_3d/scri_directional_probe.cpp` and `scri_directional_replay.py`. Separate local oracle builds v1–v4 have their own source/library hashes and build commands. No shared build, remote source, checkpoint or scheduler state was changed. Binaries and full checkpoints are omitted. Retained source snapshots preserve each earlier oracle; the current wrapper reproduces the canonical v4 comparison and action results. The historical wrapper with SHA `c181bb6a…`, used by the first comparison/action and the separate figure, was recovered byte-for-byte from its preserved predecessor plus the recorded patch; the reconstructed bytes match the already-recorded hash exactly. This archival reconstruction was not a rerun.

Canonical evidence is in `v4/`, with the original validated six-term decomposition in `v3/`. All native build-source headers are unchanged from frozen production commit `d98e2a99…`. The n6 epsilon0 adapter action matches the old coupled oracle bitwise. Interleaved epsilon0/2 calls, repeated calls and recovery after a rejected NaN input reproduce exact arrays. Minkowski background maxima remain `1.1572e-12`. The two meaningful n16 workers took about 20 seconds for full Jv and 11 seconds for the numerator-factorization checks; every worker was bounded below 390 seconds and local computational concurrency never exceeded two.

Build a fresh local Serial oracle and run, from the repository root:

```sh
python3 tst/hyperbolic_shift_3d/scri_directional_replay.py build --output output/directional-repeat
python3 tst/hyperbolic_shift_3d/scri_directional_replay.py replay \
  --library output/directional-repeat/probe.dylib \
  --checkpoint output/hyperbolic-core-shell/aurora/linear-window-dissipative-v2/output/snapshot-030.checkpoint \
  --output output/directional-action-repeat
python3 tst/hyperbolic_shift_3d/scri_directional_replay.py factorization \
  --library output/directional-repeat/probe.dylib \
  --checkpoint output/hyperbolic-core-shell/aurora/linear-window-dissipative-v2/output/snapshot-030.checkpoint \
  --output output/directional-factorization-repeat
```

Use `.so` on Linux. Exact historical reproduction requires the native headers from frozen commit `d98e2a99…` and the corresponding retained adapter/wrapper, checked against each `frozen-native-headers.json`; rebuilding against a changed working tree is a new source binding. `compare` accepts three repeated `--checkpoint` arguments and optional `--library` for native retraction/volume-weight checks; `decompose` returns the six-term surface action. These commands require retained raw local checkpoints and their original worker status. All raw reads verify checkpoint hashes, sizes, configuration and all four finite arrays. Acquisition admission and whole-campaign verification remain the separate verifier's responsibility.
