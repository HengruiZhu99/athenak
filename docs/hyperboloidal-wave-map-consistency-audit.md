# Physical reference wave-map gauge: principal and nonlinear consistency

The private physical-Minkowski reference wave-map gauge passes the constrained 20-field principal check and a finite-amplitude, nonradial, exactly flat manufactured RHS check. A separately bounded radial coordinate family also passes its complete finite-Omega point-action gate. These results establish local consistency. They do not establish native evolution stability, lower-order constraint control, an exact-scri closure or black-hole acceptance.

The branch is `z4c_hyperboloidal_layer`. Production `src/` and root CMake remain byte-identical to implementation `27c19d20696ea6dd4704032c51dfd026218f64f2`; the experiments launched from `a0f8fc8464665db3104fdbdaf142661259a6a399`. The gauge helper remains SHA256 `56d61c56bf37bf33591bec74c029c686c7d39f7abbe62dd090becda8e7176e28`. Its equations, physical-P convention, complete reference derivatives and conditional scri identities are in the [preceding local audit](hyperboloidal-reference-wave-map-local-audit.md).

The acceptance target remains a substantial angular gauge disturbance on Minkowski, followed by a single black hole surviving the inner wormhole-to-trumpet transition with the Minkowski hyperboloidal reference throughout. The present globally harmonic core is a diagnostic candidate; it does not supply the required puncture gauge.

## Bounded physical-inertial coordinate witnesses

The earlier broad compactified-coordinate controls remain failed, including their `.995` cases. This new family prescribes bounded physical-inertial displacement and velocity envelopes, `Xi^T=tau(rho)` and `Xi^I=x^I zeta(rho)`, with independent velocities `v,w` and `rho=r^2`. The stationary inverse reference embedding gives

```
xi^t = tau - (b/alphaHat) r zeta,
xi^i = Omega^2 x^i zeta/L.
```

The reference retains its exact Cauchy core. In the outer branch these coefficients and the required input jets are smooth. This input boundedness does not prove bounded gauge RHS or preservation of a scri asymptotic class. An existing C0 pure-velocity witness, for example, has a nonzero gauge pole; its limiting `Omega*Falpha=-32` and `Omega*Fbeta=12*x` agree with saved rows to `3.59e-13`.

This witness checkpoint uses the prior physical-P/spatial-norm gauge and C0 geometry. Its successful lift and geometry checks are separate from the new wave-map gauge tests below.

All 629 prescribed geometry/source/FD cases and 420 normal/constraint/binding cases pass in Release and ASan/UBSan Debug. The geometry maximum is `3.0013e-11` against `5e-10`; the final FD maximum is `7.9702e-8` against `2e-7`. Scientific output fields agree between builds.

An added direct embedding readback initially failed at `1.1209e-8`, isolated to a third-order temporal cancellation. That failure is retained. A fresh experiment tests algebraically equivalent factored inverse-map identities, using the actual returned jets and unchanged thresholds. The three identity maxima are `5.91e-16`, `1.865e-14` and `1.16e-14`. Deliberately corrupting an actual returned third jet by `1e-6` fails all 629 cases, with minimum residual `7.583e-8`. This arithmetic repair neither accepts the old direct computation nor changes the scientific point-action results. Source-only intermediate attempts are also preserved.

The frozen bounded-family index is `b503fccd27d951aad0503cc0bb80e0153d4172e7488b748395e4ed680c67a0ee`. A general angular coordinate-lift gate remains separate and unexecuted; this radial result does not admit it.

## Actual constrained principal system

The actual C0 tensor equations and both wave-map gauge poles supply 792 constrained 20-field principal matrices. The fixed grid includes lapse `.2,1,3`, chi `.4,1,2`, identity and oblique SPD frames, height parameters `.5,.75,1,2`, and 11 radii spanning the core, gauge transition endpoints and outer collar through `.98`. Algebraic determinant and trace constraints are eliminated before the characteristic chart. Derivative grading extracts the principal part; these matrices are not finite-frequency lower-order generators.

In the normalized scalar, two vector and two tensor blocks, the expected harmonic matrix `M` obeys exact rational identities

```
M^2 = I, trace(M) = 0,
Pi+ = (I+M)/2, Pi- = (I-M)/2,
rank(Pi+) = rank(Pi-) = 10,
H = Pi+^T Pi+ + Pi-^T Pi- = (I+M^T M)/2,
H M = M^T H,
x^T H x = (|x|^2+|M x|^2)/2 >= |x|^2/2.
```

The projectors give a complete characteristic decomposition at the coincident harmonic speeds. Both builds and an independent saved-data implementation agree with the expected matrix within `4.0412e-13`; actual `M^2-I` is at most `4.0867e-13`, and `HM-M^T H` at most `3.2219e-13`. Lift/output algebraic-normal residuals are at most `2.6277e-16` scaled. The original matrix threshold is `2e-12`.

The exact proof concerns the analytic expected normalized matrix. Floating agreement with the kernel is a distinct consistency check. It assumes positive frozen lapse and spatial metric and supplies no uniform estimate through a puncture or exact scri. It also gives no common three-dimensional energy estimate, lower-order growth classification or boundary stability result.

The first extraction attempt failed its unchanged matrix gate at `2.5131e-12`. A fresh attempt subtracts identical frozen-base regular and pole parts before division by Omega, rather than subtracting assembled RHS values. The source diff, failed outputs and receipt remain preserved. The accepted receipt is `cb7ad0f3edcf8da2cf760813f820ca5b44e89347c209c5f28b6175a66782dfd0`; the frozen index is `569ebb305b36a41fdb091330fce7da8e5be7a50f658588ca3892a51d68274ced`.

## Linear core coordinate rates

Seventeen polynomial witnesses at four core radii and three orientations give 204 linear coordinate cases. They retain the actual physical-P, full-Z and kappa-input-10 conventions, without projection. The predicted coordinate equations are

```
tau_t=v, zeta_t=w,
v_t=6 tau'+4 rho tau'',
w_t=10 zeta'+4 rho zeta''.
```

The actual 22-component rates agree within `1.1103e-15`, physical initial constraints within `6.6614e-16`, and coordinate accelerations within `3.5528e-15`. The origin scalar limit uses complete input jets and an analytic identity; it is not an independently differentiated source-jet query.

An independent sparse Cartesian polynomial implementation proves all 204 targets and their linear constraints/normals exactly. Actual core output retains aggregate maxima only. Independent source/count/binding review confirms those saved aggregates; it cannot recompute unavailable per-case actual residuals. This is a linear tangent result, distinct from the finite-amplitude test below. The independent principal/core review index is `1c80add1b8ca8ab4abf8cb218d1e56c3c30f2bff7eac11b34442ec7e7e280492`.

## Finite-amplitude nonradial flat-space oracle

An independent arbitrary-precision oracle constructs exactly flat metrics through an inverse physical-inertial coordinate shear, `Y(X)=X+epsilon*e_z*phi(X)`. The regular spherical wave is

```
phi = sigma [F(T-R)-F(T+R)]/R,
F(s)=exp(-((s-u0)/sigma)^2), sigma=.35, u0=-.5.
```

A division-free center expression supplies the origin. Implicit map jets through order three produce physical metric jets through order two, all consumed stored-field jets and independent exact time rates. There are 12 fixed points, including the origin, layer tails and off-axis outer points through `.98`, at epsilon `0,.025,.05,.1`, evaluated at 80 and 110 digits: 96 precision cases, 48 distinct inputs.

All 25,152 oracle checks pass. The largest analytic residual is `4.0266e-72`; 62,004 saved precision comparisons agree to `5.4335e-71` against `1e-55`. The sampled inverse Jacobian determinant is at least `.954809`; physical ADM lapse is at least `.992995`, and sampled metrics are SPD. The separate analytic bound `D>=1-2 epsilon/sigma>=3/7` controls map inversion for this prescribed family. Saved inverse-map/metric reconstruction and direct Ricci checks also pass. None of those oracle identities invokes the actual Z4c source.

The initial oracle failed because a radial-vector composition adapter received `1/Omega` rather than its required radial magnitude `r/Omega`. Its correction changes only that adapter and retains the original failure. A separate saved-reader hash-guard failure is also retained. Points, amplitudes, precision and thresholds stayed fixed. The frozen oracle index is `eb51be9c1cc698e542099169aba1bb64e8a72dd524a4cab0bd1cfb68e27b5d84`.

## Actual nonlinear 22-row RHS gate

A separately reviewed binder submits all 48 oracle inputs to the actual raw C0 geometric RHS and the unchanged private wave-map gauge, assembling each pole once. It uses physical `P=K-2Theta`, runtime kappa input 10 and `ConformalRHS` damping argument `10/alpha`, kappa2 zero, and no algebraic projection or numerical reference-RHS subtraction. Unavailable P/Lambda/Theta second derivatives are NaN sentinels and remain unconsumed. Native Omega jets and scaled reference connection/source are checked independently.

| Actual check | Maximum | Fixed threshold |
|---|---:|---:|
| all 22 exact time rates, scaled | `2.4159e-13` | `5e-9` |
| scaled reference connection | `4.2244e-16` | `5e-9` |
| Omega times conformal source | `2.4981e-15` | `5e-9` |
| physical H/M/Z/Theta, absolute | `1.3781e-14` | `5e-9` |
| input determinant/trace normals | `3.3307e-16` | `5e-11` |
| output tangent normals, scaled | `4.1237e-14` | `5e-11` |
| native/submitted Omega jets, absolute | `8.8818e-16` | `2e-10` |

Every prescribed case passes. Release and ASan/UBSan Debug outputs are byte-identical; all five commands exit zero with empty stderr. There are 381 pinned inputs and 1,049/1,051 compiler-dependency entries. The accepted receipt is `518dcec198bf9dde402c70d0b485531162522845388a4e5d3580c39648acdf2a`; executable SHA256 values are Release `4100ece941d7258f5d51a097af3a6a48eb1c2aa470c903377c110cc9d936f82e` and Debug `72accd8ab7879330b496ada93990b12379e7ab790e61151c1a480979e86dd3dd`.

Independent review pins parser/analyzer/runner sources before reading the actual outputs, then recomputes every saved residual with 100-digit Decimal arithmetic. It also reconstructs determinant and trace-rate normals by an independent adjugate formula, checks all case ordering and rehashes inputs, dependencies and executables. All checks pass. The actual-RHS frozen index is `c80d11d90f749e0c76a23af68a2409bb91775f0f2a5ba2cf2dc95199a6f8674f`; the independent review index is `669e57d9457736094cbb33a705bb62d22505c91086ba85636cefa5436d485fa0`.

## Reproduction and remaining work

The [compact archive](validation/hyperboloidal-wave-map-consistency-experiments-20261009/README.md) contains 515 copied files plus its catalog, totaling 11,560,993 bytes. Catalog SHA256 is `734f3d83f7f971c3bdd3c345e8a269254c29f0ee9eeb6aa59303937ec1927294`. It preserves small source, plans, commands, logs, receipts and failures byte-for-byte. Executables, all NPZ/NPY/JSONL files and files larger than 1 MiB are represented by size/hash/origin metadata; 32 payloads are omitted. Saved-artifact verification checks all copied hashes, 135 finite JSON files and 1,476 external originals. Compact checks do not replay omitted large arithmetic, rebuild binaries or establish scientific claims beyond the retained evidence. Original frozen capsules remain unchanged, including the empty stdout snapshot captured during the actual-RHS freeze; completed working logs are separate addenda with an explicit timing note.

Next is the actual private native angular evolution control, with reference preflights, matched C0 controls, several resolutions and a genuine half-timestep control. The native path already subtracts the analytic Minkowski geometric floating-point residual; that distinction from the raw 48-point gate must be tested in the reference preflight. The unchanged spherical component extrapolation enforces no characteristic or constraint compatibility. Production equations have not changed, so already passing production regressions were not repeated for this checkpoint.

Exact-scri regularity and preservation of null/shear/Z4/Theta falloffs remain unresolved. The old positive finite C0 modes remain negative-control evidence and are not spectra of the new candidate. No finite-frequency generator, propagator or native evolution is certified by this stage. The later black-hole stage still requires a consistent physical initial foliation and an independently justified inner gauge that survives wormhole-to-trumpet adjustment while retaining the Minkowski reference; no black-hole fixed-point subtraction is permitted.
