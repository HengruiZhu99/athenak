# Remaining V5 growth: coupled curvature and regularity penalties

The actual n16 V5 direction from the fresh t=.6 Minkowski checkpoint has a full coupled tangent-generator projection of **+42.5680**, with relative shape residual **.0142471**. Surface and volume projections agree to .003. This is consistent with the measured late trajectory growth, and is substantially different from applying V5 to the old V4 direction, whose projection was negative. It remains an approximate growing direction, not a converged eigenpair or a continuum result.

| Part of observed V5 direction | V5 projection | Relative shape residual |
|---|---:|---:|
| Whole state | 42.5680450 | .0142471 |
| Volume | 42.5704821 | .0140312 |
| Surface | 42.5677636 | .0142718 |

The volume norm is localized in the outermost shell. Surface A accounts for **89.3561%** of the whole unweighted squared coordinate norm, volume A **9.69984%**; together the trace-free curvature components account for **99.0560%**. This is not a physical energy norm or an assertion of spherical symmetry.

The six native surface contributions, projected onto this same direction, are:

| Term | Contribution to surface growth |
|---|---:|
| Regular RHS | −23.814425 |
| Sampled normal numerator derivative | −50.836220 |
| Weak numerator term, 4 N/h | +58.870923 |
| Regularity trace term, −8 C/h | +59.406806 |
| Angular dissipation | −1.059321 |
| B flux | approximately zero |
| Sum | **+42.567764** |

The term sum matches the complete V5 surface action to **4.11e-12 relative error**. The sampled normal derivative therefore opposes growth in this direction. The two weak regularity terms together supply about +118.28, overwhelming the opposing terms. This is a directional decomposition, not a proof that removing or reversing those terms makes the coupled scheme stable. In particular, C_A contains radial metric derivatives as well as A, so a nominal negative multiple of C_A need not oppose A in the discrete coupled mode. Increasing the existing penalty strengths is not justified by this result.

The next scientific change should address the discrete A/metric regularity coupling, with a characteristic or discrete-energy analysis and a complete coupled check. It must retain independent Lambda/Theta and outgoing fields, reconstruct final dependent D and keep numerical null closure. A negative projection on this one direction would again be insufficient for admission; fresh Minkowski and puncture duration tests remain necessary.

## Method and checks

The [duration checkpoint](../aurora/stability-sampled-normal-minkowski-v1/README.md) is scientifically rejected. It was read only, never resumed. A separate V5 decoder validates its 104-byte header, exact normal tuple, epsilon2, all four finite arrays and worker-bound bytes. The recorded worker hash matches the independently verified portable duration record. The source and plan match the reviewed duration identity. The discarded metric/A trace norm is 7.89e-6 relative to the full embedded perturbation.

The existing candidate adapter evaluates `T0 RHS(P(U0+E z))` at Minkowski with complete boundary reconstruction, final dependent D, ghost refresh and bulk RHS. This is not the finite-step RK map or a Jacobian about the nonlinear t=.6 state. The observed direction is rescaled to maximum tangent-coordinate amplitudes 1e-4, 1e-5 and 1e-6. Central-difference V5 actions change by at most **1.08e-6 relative** to the 1e-5 action. At all six signed inputs, the default AD full array is bitwise identical to the preserved V4 oracle. Surface/ghost null residuals stay below **1.23e-15**, dependent-D compatibility below **1.30e-21**.

The independent frozen adapter supplies the six original boundary terms and the sampled native numerator derivative. Replacing only its normal term is checked against the full candidate surface action, rather than assuming additivity. No evolution equations or penalty coefficients were changed. The V3/V4 decoder and earlier evidence are untouched.

One local process completed the calculation in **32.91 seconds** under a 350-second internal / 385-second external bound. No Aurora allocation or larger three-dimensional file was generated. The fresh diagnostic library's 43 captured source files all match the immutable V12 scientific commit **9d72318c14a2099b5aea118cca62e9fbd8773350**; [source-verification.json](source-verification.json) records that check. The diagnostic driver is separately bound by its executed hash in [action.json](action.json). The binary, source snapshot and complete checkpoint remain in local `output/`; small build, source, execution and numerical records are retained here.

Reproduce with fresh output directories:

```sh
python3 tst/hyperbolic_shift_3d/scri_candidate_replay.py build \
  --output output/v5-direction-build-new
python3 tst/hyperbolic_shift_3d/scri_v5_direction.py \
  --library output/v5-direction-build-new/candidate.dylib \
  --frozen-library output/hyperbolic-core-shell/scri-directional-replay-v4/probe.dylib \
  --checkpoint output/hyperbolic-core-shell/aurora/stability-sampled-normal-minkowski-v1/output/minkowski.checkpoint \
  --output output/v5-direction-action-new
```

Use `.so` on Linux. The frozen oracle library must match its adjacent source manifest; its original build/source evidence is preserved under `scri-directional-replay-v1/v4`. Rebuilding the candidate on another source revision creates a new source identity and must not silently inherit this result.
