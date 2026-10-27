# Angular AF endpoint audit

This is a leading-order audit of the current tensor equations and selected
spherical lapse/flux sources, not an implemented or validated angular endpoint.
No field reset, regular-center parity, conformal-flatness assumption, lapse
floor, attenuation, or change of reference connection is proposed here.

## Scope and assumptions

Initially the puncture is at the fixed Cartesian origin, where
`Omega=(1-r²)/2`, `Omega(0)=1/2`, and its gradient vanishes. Write `n=x/r` and
assume, at finite time,

    chi = r^4 C(n,t) + o(r^4),       alpha = r^2 A(n,t) + o(r^2),
    gtilde_ij = G_ij(t) + r G1_ij(n,t) + o(r),
    Atilde_ij = r^2 T_ij(n,t) + o(r^2),
    Khat = r^2 k(n,t) + o(r^2),      Theta = r^2 q(n,t) + o(r^2),
    Lambda^i = Lambda0^i(n,t) + o(1),
    beta^i = M^i_j(t) x^j + o(r),   B^i = r b^i(n,t) + o(r).

`G` is an arbitrary spatially constant positive-definite matrix with the
formulation's determinant convention; it is not set to the identity. The
components of every tensor/vector are Cartesian. Angular derivatives are on
the unit Euclidean sphere of *directions*, not a switch to spherical tensor
components. These assumptions permit angular coefficient limits; they do not
say those coefficient functions are smooth at a regular physical center.

This audit assumes the indicated derivatives/limits exist uniformly enough to
take leading orders. Higher radial terms or logarithms must be checked in an
implementation; the leading ansatz alone is not an existence theorem. In
particular, `G1`, and hence the metric-built connection limit, is additional
information beyond `G`.

Arbitrary positive `C(n)` and `G` do not automatically define an asymptotically
Euclidean end. A useful AF subclass obtained from a linear change of isotropic
coordinates is

    C(n) = Cstar (n^i G_ij n^j)^2.

Such a relation can be diagnosed for the initial AF data. It must not be
silently imposed on independently evolved off-constraint fields. More general
AF coordinate choices require their own angular geometry/compatibility audit.

## Cartesian angular differentiation

Let `P_ij=delta_ij-n_i n_j`, `D_i=P_i^j partial/partial n^j`, and let `H^S_ij f`
denote the embedded covariant Hessian of a scalar on the unit sphere. For a
homogeneous Cartesian scalar component `F=r^p f(n)`,

    partial_i F = r^(p-1) [p n_i f + D_i f],
    partial_ij F = r^(p-2) [p(p-1)n_i n_j f + p P_ij f
                     +(p-1)(n_i D_j f+n_j D_i f) + H^S_ij f].

The same formula applies component by component to tensors expressed in the
common Cartesian basis. The normal dependence of Cartesian component
functions must still be differentiated. Omitting it loses, for example, the
angular derivatives of `T_ij=t(n_i n_j-delta_ij/3)`.

Define the known leading shift quantities

    a(n)=n_i M^i_j n^j,    v^i(n)=M^i_j n^j-a(n)n^i,    d=tr(M).

`v` is tangent to the direction sphere. It is the angular transport velocity
of the endpoint coefficients, not a physical patch vector that can be mixed
with the Cartesian tensor components.

## Leading stationary-origin rates

Power-counting the complete `ConformalRHS` regular and pole terms, then taking
the limit, gives

    C_t = (4a-2d/3) C + v^i D_i C,
    G_ij,t = G_kj M^k_i + G_ik M^k_j - 2d G_ij/3,
    T_ij,t = (2a-2d/3)T_ij + v^k D_k T_ij
                +T_kj M^k_i+T_ik M^k_j,
    k_t = 2a k + v^i D_i k,
    q_t = 2a q + v^i D_i q.

The lapse equation in `WormholeLapseRate` has
`xi(0)=3 nu`, multiplicative restoring and the collapsed-inner source. Its
entire nonadvective scalar source is `O(r^4)`, so

    A_t = 2a A + v^i D_i A.

For a more general *constant* limiting restoring coefficient, before selecting
`xi(0)=3nu`, the lapse rate would additionally contain
`(-6nu+2xi(0)) A`. The reference-background choice instead of collapsed-inner
would also add a nonhomogeneous term. Neither of these alternative equations
is being substituted for the selected gauge.

The metric-rate trace is zero with respect to `G`, so an initially constant
determinant remains so at this order without projection. The displayed AF
subclass is transported consistently: its scalar multiplier obeys
`Cstar_t=(2d/3)Cstar` when `G` follows the displayed metric rate.

### Finite physical damping does not survive these leading orders

For the selected constant, finite `kappa1`, `Omega(0)=1/2` does **not** supply a
singular denominator at this AF end. The trace/Theta damping numerators are
proportional to `alpha*Theta=O(r^4)`. After dividing their physical equations by
the `r²` coefficient scale, their contributions are `O(r²)`. The Lambda damping
is `-alpha*kappa1*(Lambda-V)/Omega=O(r²)` for finite `Lambda0-V0`, and therefore
also vanishes in the order-zero Lambda limit. This conclusion must not be
reused for an `alpha`-dependent damping prescription.

The existing spherical `reference-lapse` damping option is a distinct case:
its effective coefficient contains the inverse collapsed lapse and its leading
rates include

    k_t += kappa1*(1-kappa2)*q,
    q_t -= kappa1*(2+kappa2)*q,
    (Lambda0-V0)_t -= kappa1*(Lambda0-V0)

at the centered endpoint. Those terms are correctly present in the corresponding
conditional branch of `puncture_chart.py`; they are absent for the selected
constant-damping configuration.

## Independent Lambda and metric-built connection

The limit `V0^i(n)` depends on the first spatial metric coefficient `G1`, not
only `G`. Let the *independent* connection difference be

    E0^i = Lambda0^i - V0^i.

Subtracting the Frechet variation of the metric-built connection from the
native Lambda equation cancels the second derivatives of beta. At order zero,
the remaining curvature/momentum and finite-damping sources vanish under the
stated powers, giving

    E0^i,t = v^j D_j E0^i + (2d/3) E0^i.

The exact elliptic gauge supplies `V_t=-eta V`. For a stationary endpoint with
the assumed limit this gives the equivalent independent-Lambda update

    Lambda0^i,t = -eta V0^i + v^j D_j(Lambda0^i-V0^i)
                              +(2d/3)(Lambda0^i-V0^i).

No step sets `E0=0`. Storing an independently evolved `E0` would be an equivalent
variable choice only if the metric connection is reconstructed consistently
and the complete conversion/time derivative is retained; using it as a
constraint projection would change the task.

It is not generally safe to discard beta's subleading Hessian and evolve
Lambda0 as if beta were exactly affine. A direct Lambda endpoint expression
also contains the leading mapped vector-Laplacian of beta's next radial term.
The cancellation above uses the original converged elliptic equation. An
implementation must verify the original vector residual and its limiting
identity independently, or explicitly provide that next shift jet. It must
also supply `G1`/`V0`, rather than manufacture them from `G` alone.

## Reference-vector telegrapher

The AF-regular taper equals one in a neighborhood of this centered puncture.
For Cartesian flux components `B^i=r b^i(n)`,

    partial_i B^i = n_i b^i + D_i b^i,
    partial_i(alpha-Aref) = r [2A n_i + D_i A - n_i] + o(r).

Consequently the leading vector-flux coefficient obeys

    b^i_t = a b^i + v^j D_j b^i
             +[kappa*(2A n_i + D_i A - n_i)-b^i]/tau.

The lapse correction `chi*div(B)` is `O(r^4)` and does not enter the leading
`A_t`, but is required at the next radial order. For radial `b^i=b_rad n^i`,
`D_i b^i=2 b_rad`, so `div(B)=3 b_rad`. The spherical `B_r+2B/r` follows; it does
not define the missing tangential angular terms of a 3D endpoint.

The current Cartesian flux *value* tends to zero. A complete angular atlas
needs its finite coefficient or equivalent first normal jet; storing only
`B(0)=0` loses the data required for this transport/relaxation equation.

## Spherical reduction and a cheap native falsifier

For `M=b1 I`, `a=b1`, `v=0`, `d=3b1`, `G=I`, `V0=0`, the formulas reduce to

    (log C)_t=2b1,   (log A)_t=2b1,
    T_ij,t=2b1 T_ij, k_t=2b1 k, q_t=2b1 q, Lambda0,t=2b1 Lambda0.

The generic chart has `chi_ref~16r^4`, `alpha_ref~8r²`, `W~16r²`; these rates
are exactly the selected centered endpoint rates in `PunctureChart.rhs`.
The lapse's `-6nu+2xi(0)` cancels because `xi(0)=3nu`. For a radial flux the
rate is `b_rad,t=b1*b_rad+[kappa*(2A-1)-b_rad]/tau`, agreeing with
`telegraph.python_rhs`.

A read-only numerical check used the current compiled native spherical tensor
kernel `build-shift-3d/libelliptic_shift_3d_radial_kernel.dylib` through
`kernel.Kernel.parts(...,total_chi=chi)`. Its independent analytic inputs were

    chi=16r^4, alpha=8r², Arr=.2r², Khat=.3r², Theta=.1r²,
    Lambda_r=.15, beta=-.4r,

with analytic first/second derivatives, `kappa2=-.99`, Appendix-C Theta, and
`kappa1=4` versus `400`. The expected scaled geometry rates
`[chi_t/chi, Arr_t/r², Khat_t/r², Theta_t/r², Lambda_r,t]` are
`[-.8,-.16,-.24,-.08,-.12]`.

| kappa1 | r | Maximum scaled geometry error | Selected log-lapse rate |
|---:|---:|---:|---:|
|4|1e-2|6.82259e-2|-.8138390552|
|4|1e-3|6.82470e-4|-.8001383999|
|4|1e-4|6.82503e-6|-.8000013840|
|400|1e-2|1.14565e-1|-.8138390552|
|400|1e-3|1.14560e-3|-.8001383999|
|400|1e-4|1.14560e-5|-.8000013840|

The expected log-lapse rate is `-.8`. Both damping choices approach the same
endpoint at the predicted `O(r²)` rate. This falsifies adding a finite
constant-kappa damping term to the leading centered coefficient equations.
It is not a test of angular differentiation, an atlas, or changing regularity.

## Moving-center compatibility is a separate problem

For a center `X(t)`, let `y=x-X(t)` and beta=`beta0+M y+...`. Because the native
transport sign is `+beta^i partial_i`, differentiating at fixed y shows that

    Xdot = -beta0

cancels the translational `O(1/r)` coefficient transport. This is the opposite
sign from a conventional material-flow convention. A stationary fixed-center
chart implicitly relies on `beta0=0` by symmetry; the *general elliptic solver*
must not impose that condition at every puncture.

The existing generic chart computes moving-reference time derivatives, but
this alone does **not** make the current selected gauge compatible with a
translated AF puncture. `WormholeLapseRate` uses global `z=|x|²`. At a translated
puncture `X!=0`, sending alpha and Khat to zero leaves the nonadvective lapse
source

    [z_X * (1-(1-z_X)^4)] / Omega_X.

It is nonzero for `0<|X|<1`; neither the `O(r²)` advection after center transport
nor the `O(r^4)` flux term cancels it. Thus an `alpha~|x-X|²` translated endpoint
is not preserved by this source as written. In addition, away from the origin
the fixed compactification has a nonzero gradient, adding leading Omega
transport factors; `V_t=-eta V` imposes further compatibility on the moving
metric-connection limit. A translating/multiple-puncture formulation needs an
explicitly derived extension. It must not be claimed from the reference-jet
chain-rule test or introduced silently while comparing the stationary gauge.

This does not obstruct the specified centered spherical-comparison benchmark.
It also does not authorize changing the selected current gauge. A useful
immediate diagnostic is to independently evaluate/fit beta(0) from the bulk
solution in radial and nonspherical pilots. The current angular perturbation
contains an odd `+.3z` contribution, so it can drive a translating component.
A separately recorded inversion-even quadrupolar test can exercise true 3D
terms while retaining a stationary puncture by continuum symmetry. The old
dipole case must remain recorded. This is a falsifiable test-data distinction,
not a beta=0 boundary condition or an explanation yet proved for the dominant
transition residual.

### Measured beta(0) in the frozen N8 pilots

The follow-up [fit evidence](evidence/af-beta0-20261027.json) uses the frozen
final chart2 radial and angular CSV files at t=.0002. It fits Cartesian
constant+linear+quadratic beta on all central points inside five radii
.06/.08/.10/.14/.18 (32/56/136/360/840 points), with both unweighted and physical
volume weights. Coordinates are scaled by the fit radius; the quadratic
design has rank10 and unweighted condition numbers7.22–8.81. The native beta
is reconstructed as `delta_beta-x`, without changing any evolution data.

| Diagnostic | Radial | Angular with odd z term |
|---|---:|---:|
|Maximum unweighted quadratic-fit beta0 norm|6.691e-14|3.1466e-10|
|Range of fitted z component across radii|9.43e-18|1.576e-13|
|Inversion-even beta vector RMS inside .18|6.726e-14|3.076e-10|
|Inversion-odd lapse RMS inside .18|6.61e-18|4.102e-9|
|Pilot duration times maximum fitted beta0|1.338e-17|6.293e-14|
|`4*t*|beta0|/r_min` finite-grid log-chi scale|1.98e-15|9.30e-12|

All840 central points inside .18 have their inversion partner. Fitting only
the inversion-even beta data with a constant+quadratic polynomial independently
reproduces the full fit's beta0 within3.8e-18. Both the odd lapse input and its
small even-beta response are resolved above the radial numerical asymmetry.
The translation scale is nevertheless much too small to explain the current
large transition-region errors. It is a stationary-chart compatibility check,
not the dominant mechanism identified by these pilots, nor a continuum bound
as r tends to zero. The JSON retains fit residuals, weights, matrices, radii,
condition numbers, input hashes and the independent parity check. All stored
values were explicitly checked finite; the independent parity calculation
also avoids transient BLAS floating-point-status warnings seen in the original
residual matrix multiplication.

## What an implementation still needs

1. An explicit direction-sphere atlas, Cartesian-component convention,
   edge/corner communication, angular derivatives and convergence controls.
2. Consistent endpoint estimates for `M`, `G1/V0`, all independent coefficients,
   and the leading flux jet, with the original vector residual checked.
3. A one-sided/matched radial coupling between that atlas and Cartesian cells;
   treating angular coefficient limits as regular-center values is insufficient.
4. Full-region diagnostics: asymptotic powers, beta0, scaled constraints,
   positivity and angular scatter, including every positive-radius cell.
5. Controls for the nonuniform late wormhole-to-trumpet limit. A finite-time
   AF end can retain `chi~r^4` while an exterior annulus approaches `chi~r²`;
   a surviving shrinking unresolved core does not demonstrate a regular
   puncture or justify replacing the endpoint powers.

The rates above are implementable leading data, not a complete boundary theorem
or a proof that a particular interpolation between the atlas and a Cartesian
grid is stable. They should first be checked with prescribed homogeneous
angular manufactured coefficients and an independent radial reduction before
being used to extend the failed nonlinear pilots.
