# Read-only Minkowski failure localization

This analyzes frozen unprojected SYCL `stability-v1` data only. No evolution,
restart, admission, source change or modification of the input files occurred.
The native V1 checkpoint is explicitly **unadmitted**. Its rollback state is at
coordinate t=0.4895, step 979; the failed-stage file has no independent clock and
is identified only as an intermediate state of the following failed step.
Input sizes/hashes and all numerical summaries are in `localization.json`.

The report's state/constraint diagnostics are stale at t=0.45, as explicitly
indicated by `diagnostics_current=false`. They must not be identified as the
rollback state's diagnostics. Its cumulative pole/clock/null maxima can include
later attempted stages. The plots distinguish the history from the separately
measured rollback fields.

## Direct observations

* Growth is seeded near roundoff and becomes exponential. Least-squares slopes
  of log magnitude over t=0.15--0.40 are 74.49 (state deviation), 75.57 (momentum
  L2), 75.58 (Z4 L2), 77.63 (pole maximum) per coordinate time. Momentum and Z4
  grow with the fields: this cannot be classified as harmless pure gauge motion.
* At rollback, core and the first two shell layers have determinant errors at
  most 5.6e-16 and tr_g(A) at most 6.6e-16. Their Lambda vector maxima are
  3.85e-13, 4.30e-12 and 1.43e-10. The outer shell has Lambda maximum 72.73,
  determinant error 0.59448 and tr_g(A) maximum 30.912.
* Scri tensor data have determinant error 0.39798 and trace maximum 27.801.
  More than 99.99% of their solid-angle-weighted squared magnitude lies in the
  outer two angular cell strips; 96.46% of determinant and 86.12% of trace
  squared magnitude is in the strip intersections at patch corners. Those
  corner intersections occupy only 2.97% of the sampled solid angle.
* The scri radial Lambda monopole accounts for only 0.67% of its squared angular
  magnitude. Radial Lambda RMS is 3.177, versus tangential Lambda vector RMS
  5.138. The failure has strongly nonspherical angular content absent in 1D.
* Nodal metrics remain positive even in the failed RK stage: minimum metric
  eigenvalue 0.38162 in the volume and 0.56665 on scri. Therefore the first
  rejection can arise in angular interpolation without negative nodal metrics.
  This is consistent with the recorded `nonpositive interpolated scri geometry`
  error; it does not imply that the already-corrupted accepted state is usable.

These are late, nonlinear localization observations, not a determination of the
first growing eigenmode. No early-time full field snapshots were available in
the supplied failure campaign. Only a single resolution/timestep was analyzed.

## Comparison with the spherical boundary

The 1D chart algebraically reconstructs angular g/A and consequently cannot
represent determinant or A-trace drift. The 3D continuum equations preserve
those constraints if initially zero, but finite differences need not. Final
algebraic repair is therefore a meaningful controlled variant; success would
not by itself prove which boundary mechanism triggered the unprojected failure.

In the spherical, algebraically constrained linearization about Minkowski, the
3D regularity traces reduce to the donor traces:

```text
C_chi = delta chi - delta g_rr
C_Arr = delta A_rr - (delta g_rr)'/6 + 2(delta chi)'/3
C_P   = delta P;  C_Theta = delta Theta
C_Lambda = delta Lambda - delta contracted_Gamma.
```

For the isolated, derivative-free scalar penalty block, write
v=delta(chi*g^nn), p=delta P, T=delta Theta. At scri alpha=1, K=-3,
w=n_bar(Omega)=-1, and exact null elimination gives delta w=-v/2. Then

```text
S_chi   = v + (2/3)p + (4/3)T
S_P     = -3v - 2p + 3.96T
S_Theta = -3v - 2p - 8.04T

h d_t [v,p,T]^T |_penalty =
    [ -4,   8/3,  16/3 ]
    [-12,   -16, 15.84 ] [v,p,T]^T
    [-12,    -8,-40.16 ]
```

Its h-scaled eigenvalues are -8, -23.33657 and -28.82343. At outer h=0.015625,
their RK4 arguments for dt=0.0005 have magnitudes at most 0.92235, within the
negative-real stability interval. The isolated A-trace penalty is also damping,
-16*tau/h. This rules out a simple anti-damping sign in these *local* blocks;
it is not an eigenvalue analysis of the full spatial/gauge/interface operator.

The important additional 3D boundary mechanism is angular differentiation.
`ScriSurface::Build` clamps its six-point angular stencils within each face:
`first=clamp(j-2,0,n-6)` (and similarly k). Its face-edge/corner jets consequently
use one-sided angular polynomials, with no cross-face angular donor exchange or
characteristic/SAT coupling for the evolved scri rows. Gathering the surface
array does not supply that missing interface operator. Volume halos do exchange
across seams. The observed corner localization makes this a leading suspect,
but it is not yet a causal proof.

The donor also differentiates complete pole values numerically at its endpoint,
whereas the tensor closure uses analytic directional differentiation of the pole
with reconstructed jets. Both have the same continuum limit on regular radial
data; their semidiscrete endpoint operators differ and should be compared if a
radial mode is found. The present failure is overwhelmingly nonspherical.

## Targeted next diagnostic, requiring coordination before implementation

1. Record separate volume/scri per-field maxima with their block, radial index,
   angular indices and coordinates at early times, including det-1, tr_g(A),
   each pole/regularity residual and radial/tangential Lambda/D. Save a bounded
   early snapshot while deviations are 1e-8--1e-5, before nonlinear distortion.
2. Linearize the **materialized** scri operator about exact Minkowski with fixed
   Minkowski volume donors, using central directional differences at several
   perturbation sizes. Reapply alpha/beta/J and dependent-D elimination for each
   RHS evaluation; only independent boundary variables belong in the Jacobian.
   Matrix-free Arnoldi can distinguish a positive-real corner mode from simple
   forcing/non-normal transients without evolving a black hole.
3. Compare that operator with/without the controlled algebraic tangent-space
   restriction. Compare radial l=0 perturbations with tangential and corner
   perturbations, keeping the same equations and coefficients. A radial-only
   reduction is a diagnostic control, not a proposed production replacement.
4. If the angular corner mode persists, test a genuine cross-face surface
   derivative/interface closure. Altering pole penalties or adding filtering
   without identifying the mode would introduce another formulation choice.

No proposed diagnostic above was run as an evolution in this audit. Numerical
precision of the enforced null identity remains a necessary algebraic check,
but does not control the growing geometric, momentum or Z4 modes.
