# Wormhole map and inner asymptotic evolution

Status: isotropic initial data and the leading inner equations are independently
verified. The first positive-grid outflow-only evolutions stop when their inner
gauge characteristic becomes incoming. They do not establish trumpet settling.
An actual-inner-infinity coefficient chart is the next implementation step.

## Initial slice and isotropic compactification

In Schwarzschild F=1-2M/R, prescribe the physical normal radius derivative
H=n(R)=s(R)R, where s is infinitely flat, zero below4M and one above8M.
The embedded slice has

    h_R=H/[F sqrt(F+H^2)], gamma_RR=1/(F+H^2),
    K_radial=-H_R, K_angular=-H/R, K=-H_R-2H/R.

The exterior is genuinely hyperboloidal. The isotropic interior remains the
time-symmetric two-ended wormhole, R=rho+M+M^2/(4rho); the second end is not
given the outer height-function transition.

For the outer branch set y=1/R and

    J(y)=integral_0^y ds/sqrt(s^2-2Ms^3+s_switch(1/s)^2),
    r=exp[-J(y)], r0=exp[-J(1/(4M))].

Matching to the isotropic interior gives r=k rho, with
`k=r0/rho0`, `rho0=(4M-M+sqrt(4M(4M-2M)))/2`. Thus g=1 everywhere,
`chi=[r/(Omega R)]^2`, and `A_rr=(2/3)(H/R-H_R)/Omega`.
Interior physical lapse is exactly `(1+M/(2rho))^-2`. It smoothly joins the
stationary exterior physical lapse sqrt(F+R^2). The evolution reference is
still Minkowski. No Schwarzschild RHS is subtracted.

Near scri, cancellation is avoided analytically. In the CMC exterior,
J=asinh(y)+delta, where

    delta=integral_0^y 2Ms^3/[sqrt(D_M)sqrt(D_0)(sqrt(D_M)+sqrt(D_0))] ds,
    D_M=1+s^2-2Ms^3, D_0=1+s^2.

Then `chi=[cosh(delta)+sqrt(1+y^2)sinh(delta)/y]^-2` is evaluated with
log1p/expm1. The lapse difference uses
`alpha^2-Aref^2=-r^2(chi-1)/chi-2M Omega^3 sqrt(chi)/r`.
These retain the physical mass jets rather than losing them in0/0 arithmetic.

The512-point test gives H/M below2.3e-13/3.9e-14, interior mass error1.24e-11,
and exact-exterior geometric RHS below1.1e-15. Changing coordinate quadrature
96-to128 changes data by9.4e-15. The quadrature discretization of the elliptic
shift has a separate transition-stencil error; it is not subtracted.

## Why the first inner treatment stops

For bounded conformal metric shape, the homogeneous metric-rate solution
Q=c/r^3 is inadmissible at either inner end, so Q=0 remains the elliptic shape
gauge. Its remaining beta/r constant is fixed by outer nullness. Initially
beta=b1 r+..., b1 approximately.0397623 for M=.1.

At a wormhole end alpha=O(r^2), chi=O(r^4). Physical light speeds are
`-beta +/- alpha sqrt(chi)` and lapse gauge speeds are
`-beta +/- sqrt(F_lapse chi)`, with `F_lapse=alpha^2+4nu Omega^2 alpha`.
The gauge cone is wider. N256 initially has an incoming gauge characteristic;
N512/1024/2048 initially have inner outflow, but the maximum becomes positive
at approximately t=.2042,.2132,.2156. The runs stop at that event. Positivity
and scri compatibility still hold. Their finite-cutoff outflow stencils do
not provide incoming data, so continuing them unchanged would be unjustified.

At the actual r=0 asymptotic end all these speeds vanish. A characteristic
dr/dt proportional r cannot arrive from r=0 in finite coordinate time. The
appropriate alternative is evolving the asymptotic coefficients at that
degenerate endpoint, not imposing regular-center parity at a finite cutoff.

## Independent free coefficient limits

Use the asymptotically flat inner phase

    chi=r^4 X, alpha=r^2 A, A_rr=r^2 B,
    Khat=r^2 k, Theta=r^2 tau, Lambda=L,
    beta=b1 r+O(r^5).

The six coefficients are independent. Odd radial powers in X and A are
allowed; the wormhole mass produces them. These are not origin parity data.
Physical Z norm vanishes as r^2 L, although the coordinate connection error L
may have a nonzero finite limit.

The symmetric restoring source `-xi(alpha+Aref)(alpha-Aref)/Omega` has a
nonzero constant limit at alpha=0 and breaks this lapse phase. The explicitly
labeled multiplicative variant `-2xi alpha(alpha-Aref)/Omega` differs only by
`-xi(alpha-Aref)^2/Omega`, preserves the linearized Minkowski gauge, and has
the required vanishing puncture lapse. With this variant, the full native
tensor equations give

    X_t=2b1 X,
    A_t=(2b1-6nu+2xi)A+2,
    B_t=2b1 B, k_t=2b1 k, tau_t=2b1 tau, L_t=2b1 L.

An independent native-tensor limit test retains all six nonzero coefficients
and converges to these equations at order2 (`test_puncture_limits.py`). No
Hamiltonian, momentum, Theta or connection coefficient is reset.

For the initial wormhole, X0=64/(M^4 k_map^2), X1/X0=-8/(M k_map), and
the inner ADM mass is `-X1/X0^(3/2)`. The PDE gives X0_t=2b1 X0 and
X1_t=3b1 X1, conserving this mass analytically. A discrete diagnostic must
still show convergence; it must not claim mass conservation merely because
an auxiliary ODE was chosen to enforce it.

For finite time the continuum inner end remains asymptotically flat. Trumpet
settling is the limiting behavior outside an increasingly unresolved inner
wormhole core. Demonstration requires converged limiting areal radius,
proper-distance behavior, lapse collapse, horizon/invariant mass and settling
profiles, starting from the verified genuine wormhole above.
