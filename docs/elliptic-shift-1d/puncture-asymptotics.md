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

## Implemented positive coefficient chart and its failed controls

`puncture_chart.py` now evolves an actual r=0 coefficient node and positive
physical points through actual scri. Define

    w=r^2/(r^2+Omega^4), gap=1-w,
    chi_ref=r^4/(r^4+Omega^4), alpha_ref=Aref*w,
    chi=chi_ref exp(U_chi), alpha=alpha_ref exp(U_alpha),
    A_rr=w Bstored, Khat=w(kstored-3), Theta=w taustored.

Lambda remains an independent unweighted coordinate field. Analytic reference
derivatives and numerical coefficient derivatives are combined by the chain
rule. Near scri, reference deviations and expm1 avoid losing the mass jets.
Near the inner end, total chi/alpha/Khat are supplied separately rather than
recovered by subtracting order-one reference deviations. Equivalent inner
R/S rearrangements cancel the flat-end reference constants analytically.
No field rate, constraint, or Schwarzschild numerical RHS is subtracted.

The zero node evolves the six coefficient ODEs above. Its metric/lapse values
are zero because it represents spatial infinity; positivity applies to the
physical r>0 nodes. The inner ADM mass is differentiated from U_chi and is
a diagnostic, not an enforced conserved quantity. Short aligned restarts,
including reconstructed physical chi/alpha, are bitwise identical. Native
off-constraint tensor rates agree to5.8e-14 with H/M/Theta/Z all nonzero.

Optional puncture dissipation is -epsilon B^T W B/(64h), where B is the
forward third difference on nodes1..N and W is a nonnegative taper vanishing
before scri. Column0 is excluded, leaving its exact coefficient ODE unchanged.
This negative semidefinite operator uses no ghosts or parity. Its boundary
truncation error still requires measured convergence; its sign is not a
stability proof for the coupled system.

These changes remove the finite-cutoff incoming-characteristic assumption,
but do not yet give an admitted wormhole evolution. At M=.1, epsilon=.5,
N129/257/513 survive t=1 with nonconvergent H/M errors. A variable damping
control gives final H RMS .924/1.162/1.874 and M RMS2.676/3.536/5.325
(`evidence/wormhole-reference-damping-v1`). At M=1/3 the corresponding
three-resolution controls reach t=6, but final H RMS
.01241/.007176/.01393 and M RMS .09671/.07351/.08653 do not converge.
Every scri-pole peak exceeds its declared gate. These are failed controls,
not wormhole-to-trumpet results (`evidence/wormhole-Mthird-convergence-v1`).

## Explicit optional equation changes

The reference-lapse damping control uses kappa_eff=kappa1*Aref/alpha but
evaluates every damping product without dividing by the collapsed lapse.
With D=-kappa1*(alpha-Aref), add

    Delta S_Khat=D(1-kappa2)Theta,
    Delta S_Theta=-D(2+kappa2)Theta,
    Delta S_Lambda=-D Lambda.

This is quadratic about Minkowski and vanishes on the constraint surface.
The scalar angular-Ricci/potential expression receives the same change.
The inner k/tau/L ODEs gain kappa1(1-kappa2)tau,
-kappa1(2+kappa2)tau and -kappa1 L, respectively. Independent pointwise
native-kappa and limiting tests pass. Scalar and wormhole evolution controls
do not improve sufficiently, so constant damping remains the default.

A Python-only rational lapse prototype replaces F=alpha^2+4nu Omega^2 alpha by

    F=alpha^2+4nu Omega^2 alpha*(Aref+delta)/(alpha+delta), delta>0.

The exact source difference is
+4nu Omega^2 alpha*(alpha-Aref)*(Khat+3)/(alpha+delta).
It preserves the Minkowski solution and its linearized equations, harmonic
scri, and F=O(alpha) at the puncture. Delta is a gauge parameter, not a floor
on any physical field or Omega. The inner lapse ODE replaces nu by
nu*(.5+delta)/delta. The off-constraint tensor replacement and inner limit
are independently checked. The rational family reduces the local
Bona-Masso steepening indicator for alpha well above delta, but this is not
evidence that the observed failures are gauge shocks. Both tested scale
matrices contain failed/large-error evolutions; none is adopted as a cure.

## SBP and incoming-gauge boundary controls

The optional radial SBP6/3 operator copies the repository's scalar-operator
coefficients into a portable module. Its diagonal H norm obeys
H D+D^T H=diag(-1,0,...,1). D2=D*D consequently obeys
H D2+D^T H D=E D. Both identities, polynomial differentiation, the scalar
SAT energy relation, and H-negative KO are checked independently. D2 is
only second order at the boundary. These algebra checks neither establish
the full Z4c energy estimate nor meet the nonlinear endpoint-convergence gate.
The M=.1, KO2 AF-chart control has final H RMS
.2557/.06274/.1200 at N129/257/513 and does not converge
(`evidence/wormhole-SBP-ko2-v1`). Increasing the original FD KO to2 or8 also
does not give admitted H/M convergence (`evidence/wormhole-strong-dissipation-v1`).

A distinct finite-cutoff prototype supplies incoming lapse-gauge data rather
than pretending that mode is outflow. Write A=alpha_r, a=F/Omega,
b=Omega*chi, c=sqrt(F*chi). The principal pair is
A_t=beta A_r-a Khat_r and Khat_t=beta Khat_r-b A_r. The incoming left
gauge combination is Khat+Omega*sqrt(chi/F) A. Its full principal gauge
eigenvector, in [dchi,A_rr,Khat,Theta,Lambda,dalpha], is
[0,2/(3Omega),1,0,0,c/(Omega chi)]. It has zero principal differential
constraints. Consequently the weak Robin Khat-rate penalty

    P=-Omega chi/H0 * max(c-beta,0)/c * (alpha_r-p alpha/r)

is lifted simultaneously into A_rr_t by2P/(3Omega). Theta, Lambda and
Hamiltonian/momentum errors are not reset. For homogeneous Neumann data,
its frozen left energy flux reduces to
-beta/2 [A-sqrt(a/b)Khat]^2 for0<=beta<c, and it switches off continuously
at pure gauge outflow. Robin p>0 adds lower-order boundary-energy terms;
no complete nonlinear estimate is claimed.

This treatment is allowed only while every physical/constraint characteristic
is outflow. N257/513/1025, p1/p2 runs all halt near
t=.2154/.2161/.2163 because a physical characteristic becomes incoming.
The actual AF shift slope is negative during this phase. Supplying gauge
data alone cannot justify physical incoming data from the other AF end.
These are preserved boundary-validity failures, not accepted excisions.
A pilot using physical-field boundary KO fails too; its low-order closure
contaminates the initial r^4 metric factor. The no-KO matrix isolates that
effect and still reaches the same physical-characteristic obstruction.

Another optional lapse-source control replaces the additive inner r^2
driving by r^2 alpha/Aref through a (2Omega)^4 taper. Precisely, add
Delta S_alpha=(2Omega)^4 r^2 (alpha-Aref)/Aref. It leaves the scri limits
unchanged but changes the interior linearized lapse source; it must not be
advertised as preserving the whole linear Minkowski model. Its AF A ODE
loses the additive +2. All Einstein equations and F stay unchanged. The
native off-constraint and limiting audits pass, but the first matrix still
develops large errors/failures. It is a diagnostic prototype, not an adoption.

New parameter matrices freeze their entire numerical source and both supplied
binaries once before any case starts. This prevents live edits during a long
matrix from changing the method between its resolution cases. Each individual
case retains its own source binding and checkpoint ancestry as before.

## Separate already-trumpet CMC control

`trumpet_data.py` constructs a stationary Schwarzschild CMC slice with K=-3,
H=n(R)=R-C/R^2 and f=1-2M/R+H^2. A double root at R0 obeys

    C=R0^2 sqrt(R0^2+1/3),
    M=R0^3+2R0/3-C,
    gamma=sqrt(2+9R0^2).

The second relation is rationalized for the root solve. The isotropic map is
r=exp[-J(y)], y=1/R, J=integral_0^y ds/sqrt(D(s)), with
D=1+y^2-2(M+C)y^3+C^2 y^6. Evaluate its exact double-root factorization

    D=(y0-y)^2 P(y), y0=1/R0,
    P=R0^2+2R0^3 y+3C^2 y^2/R0^2+2C^2 y^3/R0+C^2 y^4.

All P coefficients are positive. Subtract the logarithmic singularity
analytically in J, and invert in v=-log(1-y/y0), where J_v=1/sqrt(P).
This avoids resolving a near-double root by subtracting nearly equal
polynomial values. Near scri, subtract asinh(y) analytically; its regular
integrand numerator is2(M+C)y^3-C^2y^6. The log1p/expm1 metric construction
then retains its mass jets. The exterior conformal metric depends on M+C;
the independent tracefree curvature cancels the C contribution in the
invariant mass, which is M.

The fields are g=1, chi=[r/(Omega R)]^2, A_rr=-2C/(Omega R^3),
Khat=-3, Theta=Lambda=0, alpha=Omega sqrt(f), beta=-r H/R.
Analytic metric/curvature rates vanish without subtracting any numerical RHS.
Near r=0, R-R0 is proportional to r^gamma, chi~4r^2/R0^2,
physical lapse~r^gamma, and proper-distance density~R0/r. The limiting
areal radius is finite with logarithmically divergent proper distance.
The inner shift is positive and proportional to r, dominating the vanishing
physical and gauge light speeds asymptotically. The horizon R=2M has zero
outward physical light speed. No regular-center parity or AF r^4 chart is used.

For M=.1 and1/3, the native initial constraints are below6.5e-12,
geometric RHS below4.5e-13, and bulk mass error below7.3e-12.
Coordinate quadrature96-to128 changes the fields by at most1.3e-14.
The measured inner lapse powers agree with gamma. These are initial-geometry
audits only. The main evolution still uses the live Minkowski-reference lapse,
which need not make this CMC slice gauge-stationary. A separate six-case
three-resolution matrix tests that relaxation. This already-trumpet control
must never be labeled as a wormhole-to-trumpet experiment or as an identical
published-coordinate reproduction without the remaining parameter audit.
