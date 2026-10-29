# Saved-state angular-filter response

The present evidence does **not** justify implementing an outer-shell collar of the existing degree-7 filter. The leading unresolved full n6 V3 Ritz vector is concentrated on scri but nearly invisible to that filter. Terminal nonlinear n16 states have strong filter response, so they cannot be used to identify the early growing direction. No equations, production source, remote files, or evolutions were changed in this study.

For each Cartesian component, let `B` be the existing polynomial residual and `Q₂ = -2 n M⁻¹ Bᵀ M B`, where `M` contains fixed positive exact reference solid angles. The reported damping number is `γ₂ = -Re(v* M Q₂v)/(v* Mv) = 2n ||Bv||²_M/||v||²_M`. It is an **isolated additive-filter energy quotient**, not a coupled eigenvalue shift. Fractions in angular degree `l<=L` are weighted least-squares projections onto restricted Cartesian polynomials spanning those scalar spherical harmonics; tensor/vector components remain in the global Cartesian basis.

| Saved input | Surface share¹ | `||Bv||M/||v||M` on surface | Surface `γ₂` | Surface `l<=2` / `l<=6` fractions | Proposed collar `γ₂`² |
|---|---:|---:|---:|---:|---:|
| Full n6 V3, leading **unconverged** vector | 95.3945% | 0.0195711 | 0.00459634 | 87.6683% / 99.8925% | 0.00450526 |
| V3 n16, terminal accepted `t=.5205` | 94.8570% | 0.755553 | 18.2675 | 2.2767% / 7.8915% | 9.60944 |
| V4 n16, terminal accepted `t=.5225` | 95.8216% | 0.813023 | 21.1522 | 1.5527% / 5.5588% | 11.3146 |

¹ Unweighted point-coordinate squared norm, matching the original mode audit. This mixes surface and volume point counts and is not physical energy or a mesh-independent participation measure. The tangent vector uses orthonormal tracefree tensor coordinates; terminal nonlinear states retain all six Cartesian tensor components with Frobenius weights. Each row is internally consistent, but these different resolutions/states are not a convergence sequence.

² The collar quotient uses its own coordinate-volume norm, includes unchanged lapse/shift and `D-Lambda`, and is **not** added to the surface quotient. The hypothetical taper is zero through `r=.75`, rises smoothly to one at `.95`, and stays one to scri. These are compactified coordinate radii: a finite physical areal-width collar cannot extend to scri at infinite physical radius.

The n6 leading candidate has original-J relative residual **0.00343712**, far above the `1e-7` gate. Its Ritz value about 27.45 is not a measured eigenvalue and must not be compared as such with the n16 duration growth rate. Its first four saved unconverged candidates have surface `γ₂` of 0.00460, 0.0523, 0.332 and 1.218. This spread is retained in `response.json`; selecting only the leading vector does not establish the full operator's least-damped or fastest-growing direction.

For the leading candidate, 4.6051% of the total unweighted norm is in the outer shell, with 3.0161% and 1.5801% in its last two radial planes; only 0.0003252% lies in the other volume blocks. The last two planes have `||Bv||/||v|| = .01809/.02150`. Thus the same angular filter in that region remains weak on this particular vector: extending its radial support does not remove its approximate angular nullspace. Its surface A tensor alone has essentially the same weak response. This supports testing a radial surface/interior mismatch mechanism, without identifying the unstable operator term.

Both terminal checkpoints are worker-hash-bound, rolled back to the last accepted state after stage failure, and marked **unadmitted**. Their last accepted times differ and their fields are nonlinear. More than 99.9% of their already-small volume coordinate norm lies in the outermost few planes. Their high-frequency content can be generated or selected during nonlinear growth; it does not contradict the smooth n6 candidate or establish the pattern during the exponential interval. The surface A tensor dominates their angular response as well. The current degree-7 operator does not preserve positivity of signed angular interpolation, so none of these energy measurements clears the off-node geometry rejection.

The full duration comparison, owned separately, measures V3/V4 growth rates 72.43551866/72.19500132 from the same five `t=.15..35` samples, a reduction of only 0.332%. Null closure remains near roundoff while both fail on nonpositive interpolated scri geometry. Those trajectory facts motivate this study but are not derived from its isolated quotients.

## Smallest discriminating follow-up

Two fresh checkpoints per scheme at `.25` and `.35`, using the same frozen binaries, n16, `dt=.0005`, and diagnostic schedule, suffice to compare endpoint normalized shape correlation, best-fit amplitude gain and shape residual, component/radial participation, and `B/Q₂` response. A third `.30` sample tests persistence inside the interval and distinguishes changing mixtures from a single aligned direction. Existing maxima at `.25` are about `1.4–1.8e-7`, rising to `2.1–2.6e-4` at `.35`; this avoids terminal nonlinearity. Root chose three V4-only fresh captures as the next bounded diagnostic. A V3 comparison can follow if needed; no capture was launched by this study.

Use the Minkowski-subtracted state in the existing tangent coordinates, retain the separate surface/volume norms, and compare both signs and weighted best-fit amplification. Two or three checkpoints cannot certify an eigenmode. A native directional replay should evaluate the **full coupled** tangent RHS, including the actual null/dependent-D closure, ghost interpolation, bulk terms and final algebraic tangent projection. It must match n16, epsilon2 and `dt=.0005`, and check a central-difference amplitude plateau. The saved n6–8 epsilon0 probe is not a matching oracle. Compare `Re(v*WJv)/(v*Wv)` and `||Jv-a v||W/||Jv||W` with the trajectory direction, with explicit weights; an epsilon0 replay on that same vector is a counterfactual operator-action comparison, not a new trajectory. No large eigensolve is warranted yet.

## Conditional collar design constraints

If early data later support it, an opt-in collar could reuse the same global unit-angle incidence plan on every outer-shell radial plane, with a nonnegative fixed smooth radial taper. At each plane `Q_r=-epsilon n w(r) M⁻¹ Bᵀ M B`; multiplying the positive angular mass by the exact radial cell-volume factor cancels out of `Q_r` and gives a nonpositive summed coordinate-volume energy contribution. There is no radial differencing in this filter. Apply it to the same 21 independent stored channels and add its `Q Lambda` to the volume D rate, leaving `D-Lambda` unchanged under the added operator. This does not require a separate D projection. At scri, geometric source terms still precede the differentiated null/dependent-D reconstruction; appending a D source afterward is not equivalent.

Degree-7 moment annihilation with uniformly bounded incidence/support/area ratios gives the conservative smooth order `Q=O(h^7)`. A fixed-width smooth taper keeps its radial derivatives bounded as n increases; it must not be replaced by a fixed-number-of-cells step. Surface-only forcing can be amplified by the existing radial ghost lift by `O(h^-2)`; the degree-7 margin then permits `O(h^5)` locally, without proving mixed-jet or full-system stability. Spherical Cartesian vectors/tensors are preserved only to truncation order, not exactly. An identical plan with `0<=w<=1` needs no larger **isolated** angular spectral bound than the surface operator, but the coupled RK stability limit remains unproved.

The outer layer uses the same angular nodes at every radial plane, so direct global donors avoid owner-face extrapolation. The current globally gathered stage state already supplies their inputs. A distributed residual-then-transpose implementation would require owned 21-channel plane residuals and a separate gather/exchange; at n16 a full outer layer is 6*n³*21 doubles (about 4.13 MB), versus 6*n²*21 (0.258 MB) for scri. Radial-fast active ownership cuts angular planes across ranks, so a plane/row index map is required; repeated full-plane evaluation on every rank is inappropriate. These are design constraints only, with no new communication or numerical implementation adopted.

## Reproduction and validation

`analyze.py --output NEW_DIRECTORY` reads retained local mode/checkpoint inputs under `output/`, verifies the two checkpoint hashes and sizes against their original worker records, and calls only native geometry accessors. It does not evaluate a new RHS or evolve anything. Large mode vectors, checkpoints and native libraries remain local and are deliberately omitted here. Their SHA-256 identities, original relative paths and the native-library identity are in `response.json`. The first four mode reports are from the separately preserved [linear audit](../scri-linear-audit-v1/README.md).

Run from the repository with NumPy installed:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/scri-collar-response-v1/analyze.py \
  --output output/hyperbolic-core-shell/scri-collar-response-repeat
```

The n6 surface coordinates match native rows exactly and outer-plane coordinates to `1.11e-16`. Two independent local runs with NumPy 2.0.2 and 2.3.5 completed in about three seconds each. The saved comparisons differ by at most `7.11e-15`; all outputs are finite. NumPy 2.0.2's previously reproduced spurious BLAS floating-point warnings are retained in its log, while 2.3.5 emitted no warnings. Energy identities agree to roundoff. The Python SVD incidence construction is the admitted reference for the native QR plan; this is a saved-state operator analysis, not a fresh native GPU parity test.
