# Held analytic fixed-ray derivative gate

This is a source-only preparation. There has been no import, syntax check,
numerical evaluation, CAS, kernel query, compilation or scientific run of these
new files. The inverse map remains held. Production and all prior frozen
sources are untouched.

The completed scalar-values receipt is b054132e54cbe1e63b217f2cb1d416ea87e955f5ef1dfa2c945936e127730493.
Its 282 checks passed, including independent coarea/ray values and both u and
phi=u/Omega. That result admits no derivative or target-time claim. The
analytic derivative dependency is the independent fixed-Lorentz-ray pencil,
index c09efb27f523da0a101918bc29fa2dbdc07f61cf2696357613e33456633dc02e.
The local values_context.py is copied byte-for-byte from accepted source
89c96ee729b8cc741757740534566c9557dd6fdce2e66213c2fe8ff1c7e8ffe7.
Its main is not called. It supplies the previously tested independent layer
height/radius/root/coarea formulas and multiprecision quadrature only.

## Object and derivatives

The four scalar fields are u^A=delta Y^A on physical Minkowski spacetime,
signature (-,+,+,+), with zero initial field on T=H(|y|) and the exact native
angular-pulse normal datum. The fixed geometry is S=1, a=.5,
geometry transition .05-.95, native lapse/shift amplitudes .2/.1 and width .35.
The pulse has (1-r^2)^4 falloff and is not compactly supported below scri.
All derivatives are ordinary inertial Cartesian (T,Xx,Xy,Xz) derivatives.
The output contains four values, sixteen gradients and forty Hessian entries
per event, including TT, Tx, Ty, Tz, xx, xy, xz, yy, yz and zz for every field.
The event coordinates in the recipe are compact reference coordinates and
tau_reference; they are not native target-time points after the inverse map.

For each event we fix a Lorentz matrix B, parameterize k=B(1,omega), and solve
T-lambda*k0-H(X-lambda*kspace)=0. B is constant throughout differentiation:
the event-dependent choice of a convenient quadrature frame is never itself
differentiated. A second, independently fixed matrix is B times a z boost of
rapidity .2. The output components remain in the same inertial Cartesian
frame; they are not Lorentz-transformed between comparisons.

With y=X-lambda*kspace, K=k0-H_i*k^i, and D=h*K, the source is G=s*Omega.
The complete integrand is I=lambda*G/D. Analytic source pullbacks use

    lambda_a = h*(delta_a0-H_i*delta_ai)/D
    y_a^i = delta_ai-k^i*lambda_a
    lambda_ab = -h*H_ij*y_a^i*y_b^j/D
    y_ab^i = -k^i*lambda_ab.

Ordinary quotient/product rules then give all derivatives of I. The source
jets G_i,G_ij,D_i,D_ij include every coefficient derivative along the initial
graph. The graph third derivatives are explicitly supplied and compared to
the independent K_ab expression in the pencil; no higher-derivative table is
invented. First/second differentiated root identities are tested separately.

## Stable source algebra and center

The smooth cutoff's first three derivatives are analytic logistic formulas
in the log domain, using both tails and w*(1-w) from the smaller exponential.
Omega''' supplies L'' for the second source derivatives. The physical radius
inverse has r_q=Omega^2/L and r_qq=r_q*d_r(r_q). Radial tensors are converted
to Cartesian tensors only outside the exact Cauchy core.

In the core, H is constant and Omega=h=L=1, b=0. The source is evaluated as
the smooth Cartesian rational pulse (-delta alpha/alpha,-delta beta/alpha),
including at y=0. No radial normal, 1/q or logarithmic source derivative is
formed there. This branch is exact for every point q<=.05.

Outside the core the denominator is formed as

    D = k0*Omega^2/(h+b)
        + b*|kspace-k0*nu|^2/(2*k0).

The future-null identity for each floating multiprecision k is explicitly
gated; positivity is not substituted for a rounding estimate. In the exact
outer branch the pulse is formed as (2a)^4*Omega^4*exp(-r^2/.35^2), and G is
evaluated directly from the factored normal datum. It is not obtained by
forming Pi=s/Omega^2 and multiplying by a canceling Omega^3. No division by
s, G or a logarithm of data occurs, so zero-data controls remain exact.

For a separate algebra check, the unfactored F=G/h, K=D/h quotient produces
the same jets. Direct h*k0-b*nu.kspace is a comparison operand, never the
accepted denominator. A second-jet algebra records ordinary derivatives;
it uses explicit elementary chain rules and no differentiation library.

## Fixed recipe

The exact recipe, runtime path/hash and all 87 mpmath Python sources are
pinned in derivative-recipe.json. Precisions are 80 and 110 decimal digits.
The angular levels are 16x16, 32x32 and 64x64, with independently refined
64x32 and 32x64 levels. Height orders are 24,64,128,128,128 respectively.
The final level is 64x64, compared separately to 32x32,64x32,32x64. Coarea
value cross-binding uses the preserved independent 128x128 values formula.
There is no automatic refinement, retry or tolerance adjustment. An
inconclusive or failed 64-level gate must be retained before any new plan.

Five fixed native-data events cover center, Cauchy core, nonflat transition,
exact outer short-time rays and a zero-data outer control. This first
derivative gate deliberately does not include the long-time failure-position
queries: those values passed previously, but their differentiated angular
quadrature would need a separately costed and reviewed scope. Seven fixed
initial points cover origin, core, transition, the exact matching surfaces
and near scri. Four physical scalar outputs are always retained.

Two constant Lorentz frames are required at every native event and initial
point. Closed-form controls additionally use laboratory and an independent
oblique boost of rapidity .35 along (1,2,-1). The closed-form controls are:

* Flat graph, constant four-component normal velocity: u=s*T.
* Flat graph, affine four-component normal velocity: u=T*(c+d.X).
* Pure CMC graph H=sqrt(R^2+a^2), harmonic l=0,1,2 polynomials:
  u=P_l(X)*(1-a^(2l+2)/(T^2-R^2)^(l+1)), with s=2(l+1)*P_l/a.

The analytic control expressions are differentiated with the elementary
ordinary-jet algebra, independently of the ray/root/quotient construction.
Every gradient and all ten Hessians are compared, including exact zeros.
Their value, gradient and Hessian errors are separately recorded. Final
wave trace -u_TT+u_xx+u_yy+u_zz is also gated. Coarse-level traces are retained
as convergence information and are not demanded to meet the final threshold.

The exact lambda=0 one-sided initial limit has a separate graph/wave oracle.
Writing c=Omega/h and A=s/c, the oracle is

    u=0, u_T=A, u_i=-H_i*A,
    u_TT=(-2 H_i*d_i A-A*Delta H)/c^2,
    u_Ti=d_i A-H_i*u_TT,
    u_ij=-H_i*d_j A-H_j*d_i A-A*H_ij+H_i*H_j*u_TT.

This uses only graph-chart data derivatives and the wave equation. It is not
a finite-difference extrapolation. Its signs and orders were independently
pencil-checked by the literature agent before source preparation. No finite
time-derivative, inverse-time solve or event-dependent boost derivative is
substituted for this initial limit.

The fixed scaled comparison is max |a-b|/max(1,|a|,|b|), applied entrywise
before taking maxima. Final angular/boost/control/initial/coarea comparisons
are 1e-10; precision comparisons are 1e-30; the final wave trace is 1e-10.
Local analytic/factoring/null/root-derivative identities have threshold
1e-50. Absolute ray-root residual is 1e-40; bisection width is 1e-55 and
maximum iterations512. Positivity, finiteness and exact zero-pulse conditions
are mandatory. Both u and phi values are checked; only u gradients/Hessians
are claimed. Differentiated phi or Jacobians of the inverse map are not in
this gate.

## Execution and provenance hold

No execution is released by this plan. A later root authorization must pin
exactly flat_ivp_derivatives.py, analytic_jets.py, values_context.py, PLAN.md
and derivative-recipe.json, and declare the sole fresh output path. The
script verifies the completed values receipt, dependencies, resolved Python
executable and mpmath inventory before the batch, and rehashes them after.
The standardlib outer single-use launcher must capture argv/env, source
before/after, full stdout/stderr, exit status and early failures. It must use
-B, bytecode disabled and BLAS/VECLIB thread counts1. Every row is saved
appenditively as completed; a failed attempt is never relabeled a pass.

The operations are finite-event integral evaluations, not a PDE solver or a
native run. Wave trace alone proves neither all derivatives correct nor
global regularity. Passing this gate would not prove target-native-time
coverage, inverse-map existence, injectivity, absence of caustics, spacelike
native slices, lower-order Z4 stability or later wormhole-to-trumpet survival.
Minkowski remains the reference for that later goal; no inner BH blend is
implemented or selected here.
