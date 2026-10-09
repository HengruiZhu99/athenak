# Coupled conformal wave boundary gate (scratch only)

The current quadratic symmetric ray closure passes this two-field test with native upwind advection. The admitted N16 full spectrum has no positive eigenvalues; N20/N24 converged sparse modes and exact nonspherical pulse propagation agree with decay. A consistent local quadratic extension improves crossing errors but leaves a slower late tail. This narrows the earlier centered scalar-transport counterexample. It establishes no Z4c, characteristic-boundary or SBP stability theorem.

## Fixed geometry and equations

This uses the actual analytic `LayerReference`, S=1, a=.5, r0=.05, r1=.95, on admitted three-dimensional Cartesian spherical grids. The stored Z4c metric/chi jets are converted by native `PenroseMetric` and `Geometry` to the regular spatial metric gamma. Analytic lapse, shift and their derivatives remain fixed. With Pi = n_bar(phi), K_bar = -div(n_bar), and V = R_bar/6, the conformally covariant scalar equation is

```text
phi_t = beta^i phi_i + alpha Pi,
Pi_t  = beta^i Pi_i + alpha gamma^ij phi_ij
       +(gamma^ij alpha_j - alpha Gamma^i) phi_i
       +alpha K_bar Pi - alpha V phi.
```

Its characteristic coordinate speeds are `-beta_rad +/- alpha*sqrt(gamma^rr)`, exactly the reference outgoing/ingoing speeds. They run from +1/-1 in the Cauchy core to +4/0 at scri. The small-grid active nodes include incoming speed magnitudes down to 1.82e-6 (N16/span2.2); no incoming branch is deleted or prescribed to zero.

The curvature is assembled without dividing by Omega:

```text
K_ij = (Lie_beta gamma)_ij/(2 alpha),
K_bar = div_gamma(beta)/alpha,
R_bar = R3 + K_ij K^ij + K_bar^2
        +2 beta^i partial_i K_bar/alpha -2 Delta_gamma(alpha)/alpha.
```

K_bar and its gradient use analytic metric/shift jets through second derivatives. Stationary K agrees with the reference within 2.67e-15. The pure-CMC identity is `R_bar = 6*S/(a^3*alpha^3)-6/(a*S*alpha)`; the outer checks agree within 6.22e-15. An independent conformal-transformation curvature oracle, restricted to Omega>=.1 for its separate division-based identity, agrees within 2.09e-12. A continuous exact-dipole finite-difference oracle has maximum Pi-equation error 2.99e-9 at FD spacing .0004; smaller spacings become roundoff limited, reaching 4.34e-8 at .0001. These are oracle residuals, not nonzero physical sources.

## Actual discrete operator and alternative

Both fields use native `Lx<3>` advection, `Dx<3>`, `Dxx<3>`, `Dxy<3>`, and `InteriorKOSixth` with coefficient .1. The full anisotropic metric and mixed Hessian enter; this is not a spherical or radial substitute. Each raw stencil coefficient is extracted with a delta probe, and every required ghost is expanded through its exact donor weights. Independent actual field-wise `FillSphericalGhosts` and native stencil actions agree with the matrix within 7.28e-12. All coefficients and support remain finite.

The ray case uses unmodified `PlanSymmetricSphericalGhosts(grid,3,2)`. The comparison changes only ghost weights to a weighted Cartesian total-degree-two moving least-squares fit. Donors are strictly active interior nodes, selected in complete integer-distance shells (all ties included), with at least 80 and at most 216 donors. The fit uses centered/scaled monomials, explicit pivot and quadratic-moment admission, and weights `1/(1+integer_distance^2)^2`. It is local polynomial continuation, not an incoming-characteristic condition. There is no recursion, floor, SPD repair, nearest-donor substitution or silent order reduction.

All ten constant/linear/quadratic ghost moments agree within 1.96e-12 (ray: 7.62e-14). Scalar constants are not stationary solutions when V is nonzero: the matrix preserves the exact analytic constant-field reaction/coupling, with maximum residual 1.28e-11. The zero scalar field has exactly zero RHS on the stationary analytic background. All strictly interior coupled rows are identical between closures, with exact zero matrix difference. The two cubic-symmetry generators (axis reflection and axis interchange) commute with the operator to relative max-entry error <=7.44e-16.

Common refinement grids use span 2.2 at N16/20/24 (1640/3112/5520 active points). The exact production N24 geometry uses span 2.1 and 6152 active points. Allocation, spacing and first cell center follow native `N+6`, `h=span/N`, `first=-(N+5)h/2`, with strict r<1 admission. N12/span2.2 and N12/N16 span2.1 are explicitly rejected by the original planner; no admission rule is weakened.

## Spectral and pulse evidence

| Grid | Ray largest real part found | Local fit largest real part found | Scope |
|---|---:|---:|---|
| N16, span2.2 | -.2406371174 | -.1656644776 | All 3280 dense eigenvalues |
| N20, span2.2 | -.2240690444 | -.1580469677 | Converged ARPACK LR8 |
| N24, span2.2 | -.1943208436 | -.1263073647 | Converged ARPACK LR8 |
| N24, span2.1 | -.1946682410 | -.1344388956 | Converged ARPACK LR8 |

The full N16 real Schur decompositions independently reproduce the largest real parts, with relative Frobenius backward residual 2.64e-14 (ray), 2.86e-14 (fit), and normalized orthogonality residual <=3.76e-14. The full spectra contain zero positive eigenvalues above 1e-8. All retained sparse pairs have absolute L2 residual <=7.32e-10. Sparse LR8 is not a full-spectrum proof at N20/N24; the large-grid claims are limited to found modes and propagation.

The exact pulse is the physical Cartesian directional derivative (axis .3,.4,sqrt(.75)) of the smooth Minkowski spherical wave `[F(T-R)-F(T+R)]/R`, divided by Omega, where `F(s)=.001 exp(-((s+.5)/.35)^2)`. Both retarded and advanced pieces are retained, including their cancellation needed for core regularity. The physical radius is R=r/Omega and the regular retarded coordinate is `U=t-I(r)`, `I'=1/outgoing`; `V=U+2R`. Height integration uses deterministic midpoint-Simpson panels (8192) and cubic Hermite interpolation. Its outward crossing time is .745764383923427; t6 spans 8.04544 outward crossings.

Every retained run reaches t6 with RK4 and decaying fields. Nominal step is `.1*h/max_outgoing`. Same-span refinement errors at identical output times are:

| Closure | N | RMS(phi,Pi), t=.2 | RMS(phi,Pi), t=.5 | RMS(phi,Pi), t=1 |
|---|---:|---:|---:|---:|
| Ray | 16 | .001428562 | .000390353 | .000262692 |
| Ray | 20 | .000954673 | .000287295 | .000151618 |
| Ray | 24 | .000389718 | .000149437 | .000025376 |
| Local fit | 16 | .000523692 | .000129619 | .000090431 |
| Local fit | 20 | .000284042 | .000110698 | .000055945 |
| Local fit | 24 | .000115787 | .000069573 | .000029115 |

These show pulse-crossing convergence, not an established asymptotic order. The grid/sphere intersections vary with refinement. Late t6 RMS tails are not monotone N16->N20 (ray 4.54e-6->4.82e-6; fit 3.71e-6->3.78e-6). On production N24/span2.1 the final RMS is 9.80e-7 ray versus 2.23e-6 fit: the fit worsens the late tail despite reducing the peak crossing Linf error from .01685 to .001055 and initial exact-pulse Pi-dot RMS from .06064 to .003085. Its maximum ghost-weight L1 is about 34.6 versus 10.3 for ray, so it is not uniformly less amplifying.

Exact-time N24/span2.1 step halving changes the state by at most 6.38e-8 (ray) or 8.62e-10 (fit), far below the spatial pulse error. At t=.2 the differences are 1.15e-10/4.12e-11. KO-off ray controls at N16 and production N24 still have negative converged LR modes (-.23985/-.19443), remain bounded and decay to t6. Thus this gate's decay does not depend on interior KO=.1.

## Energy interpretation and limits

The stationary Killing energy used here is

```text
E = integral sqrt(gamma) [alpha*(Pi^2 + gamma^ij phi_i phi_j + V phi^2)/2
                         + beta^i Pi phi_i] d^3x.
```

For stationary coefficients the continuum energy balance is outgoing boundary flux. At scri, where the outer spatial metric is Cartesian, the flux is `alpha^2*(Pi-partial_r phi)^2`, hence nonnegative. V is nonnegative at every sampled active point and on the 2049-point analytic radial audit (R_bar range 0..26.44967); the energy density is nonnegative where alpha^2>|beta|_gamma^2=alpha^2-Omega^2. It degenerates at null scri. This does not prove V positivity for every possible layer parameter set.

Numerical energy uses each closure's actual centered gradient matrix and `h^3 sqrt(gamma)` point quadrature. The plotted exact energy uses analytic pulse gradients at those same points, so quadrature/gradient errors must be considered. No discrete monotonic-energy theorem is asserted: sampled numerical Killing energy has small temporary increases, up to .315% above its initial value among the retained cases. The separate positive normal norm `.5 integral sqrt(gamma)*(Pi^2+|Dphi|^2+phi^2)` is an observed norm and is not claimed to contract. Wave amplitudes also have no general scalar maximum principle. Conclusions use convergence to an explicit exact solution, finite spectra and long propagation, not raw Euclidean norm growth alone.

![Coupled wave pulse errors and energy](coupled-wave.png)

This scalar conformal wave has no gauge pole, evolved geometry, differential Z constraint or full 20-field coupling. The finite admitted-grid result cannot establish an SBP estimate, a stable characteristic closure at exact scri, uniform stability as h->0, arbitrary-data transient bounds or native Z4c pulse stability. The local fit is a diagnostic comparison, not a production remedy. The useful conclusion is narrower: current ray continuation plus native Lx/second/mixed derivatives need not be unstable solely because the first-order centered transport isolate was unstable. Further boundary proposals should be tested on the actual coupled Z4c constraint/gauge branches and should seek a characteristic/SBP or boundary-fitted estimate.

## Provenance and reproduction

`summary.json` freezes twelve long-run receipts, exact inputs, actual matrices, continuum/Schur/support checks and 158 non-system compilation dependencies. `full-compilation-provenance.json` additionally hashes all 1083 compiler-reported dependencies, including system headers. All production source headers consumed here match clean runtime source 27c19d20 byte-for-byte. The launch/preservation HEAD is documentation commit f615acf4. The Apple Clang 21.0.0 compiler, -O3/-DNDEBUG/-std=c++17/-arch arm64 flags, includes and four static Kokkos library hashes are recorded exactly.

The exporter source SHA256 is `10010537c26949ef8b6d4df3810d2bfeca4f410f36a05d32ca742c44862137d5`; immutable exporter SHA256 is `5b9dd39a1d5cec144bb1d4a32932e4af611c3ab30c85c59db3b76b7582a33088`. The saved header/source copies, native matrices, gradient matrices, points, eigenvalues/eigenvectors, exact-time state arrays, histories, logs and commands remain in this ignored directory. The compact-profile initial pilot is exploratory only and excluded from the authoritative twelve-case summary.

```sh
# From repository root; existing scratch SciPy/NumPy environment, one BLAS thread.
OPENBLAS_NUM_THREADS=1 PYTHONPATH=build-layer-research/boundary/python-deps \
 python3 build-layer-research/boundary/coupled-wave-isolate/run_wave.py \
 --n 24 --span 2.1 --closure ray --spectra
OPENBLAS_NUM_THREADS=1 PYTHONPATH=build-layer-research/boundary/python-deps \
 python3 build-layer-research/boundary/coupled-wave-isolate/fixed_times.py \
 --n 24 --span 2.1 --closure mls --dt-factor .05
```

No tracked file, CMake target, production runtime, matter option, continuation default or existing scalar-isolate receipt was changed.
