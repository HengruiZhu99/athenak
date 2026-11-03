# Boosted isotropic puncture on the latest hyperboloidal method

Branch `codex/boosted-hyperboloidal-puncture`, based on the latest tested method
`codex/restored-six-shells-resolution` at4aa289795. The latter was committed and
pushed with an explicit latest-method designation before this experiment began.
The ongoing unboosted N24 job9034705 is preserved; do not duplicate or cancel it.
This directory is experimental until its checks and evolutions are complete.

User scope: rest mass M=1/3, asymptotic boost v=+0.1 along x, fresh nonspherical
initial data, puncture tracker and fast-flow finder; target min(estimated Cartesian
exit time,10M). N16 first; N20/N24 only after successful N16. Preserve the six-shell
broad0.6–0.9 restoration, coefficient4, KO8/original scri and C6 base hyperboloidal
slicing. Spatial/time resolution family remains dt=.008/N with matched penalty
clipping, pending local stability checks of the changed initial geometry. No
additional zero-restoration or unboosted runs. No GPU smoke gates are requested.
The inherited10GB resource cap and immutable source/build/launcher/provenance,
healthy verified same-formulation continuation and no failed-state replay rules apply.

## Exact embedding, rather than adding a shift to spherical data

Start with Schwarzschild in rest-frame isotropic coordinates (T,Y^a):

    ds² = -a(ρ)² dT² + b(ρ)² δ_ab dY^a dY^b,
    ρ=|Y|, ψ=1+M/(2ρ), a=(1-M/(2ρ))/ψ, b=ψ²,
    R=ρψ², f=1-2M/R=a².

Use the signed a for smooth wormhole extrinsic curvature through the throat;
it is not the positive evolution lapse. Positive precollapsed evolution lapse is
an independent gauge choice. The boost parameter refers to a Lorentz transformation
of an asymptotically Minkowski, outgoing time coordinate, as detailed below.
For context see the exact isotropic boosted-hole construction in Shibata, Okawa
and Yamamoto, [arXiv:0810.4735](https://arxiv.org/abs/0810.4735), and the care needed
when composing boosts with slicing transformations in Slinker, Evans and Hannam,
[arXiv:1806.08364](https://arxiv.org/abs/1806.08364). The present wormhole/hyperboloidal
embedding is derived here; it is not their trumpet initial data.

The unboosted hyperboloidal slice has T=H(ρ), H=0 on the inner isotropic branch,

    S(R)=I_z(7,7), z=(R-4M)/(8M), with exact0/1 plateaus,
    A=sqrt(f+R²S²), dH/dR=RS/(f A).

The compact radial map is exactly the base method's map, normalized by r(scri)=1:

    ln r = - integral_R^infinity dR' / [R' sqrt(f+R'² S²)],
    r=kρ on the inner branch, k=0.306142283363079.

An inner boost requires a tilted rest-frame slice, not just changed lapse/shift.
Set

    F(Y)=H(ρ)-v Y^x B(ρ),
    B=1-I_z(7,7), z=(ρ-ρ0/4)/(3ρ0/4), ρ0=0.9714045207910318,

with exact plateaus. Thus the boost tilt turns off before the hyperboloidal
height starts at R=4M. This separation is essential: using B=1-S(R) directly
was checked and gives a timelike patch (Q≈-0.18 for v=.1) in the slicing
transition. That rejected construction is not an admissible initializer.
The selected B is part of this new initial slice and must be recorded; a boost
breaks spherical symmetry but retains continuum axial symmetry about x.

For F_a=partial_a F, F_ab=partial_a partial_b F, define

    Q=1-(a²/b²) δ^ab F_a F_b > 0,
    γ_ab=b²δ_ab-a²F_aF_b,
    K_ab= -[ a F_ab + a_a F_b + a_b F_a
              -a Γ^c_ab F_c -(a²/b²)(a_c F_c)F_aF_b ] / sqrt(Q),
    Γ^c_ab=(δ^c_a b_b+δ^c_b b_a-δ_ab b^c)/b.

These expressions follow by restricting the exact four-metric and its unit
normal to T=F. The sign convention is K_ij=-L_n γ_ij/2, so the flat unit
hyperboloid has K_ij=-γ_ij. There is no constraint solve, superposed field
approximation, or Bowen–York momentum insertion. Constraints follow from the
vacuum Gauss–Codazzi equations, but numerical sampling/derivatives still require
independent checks. At the throat the expression above avoids division by a.

## Smooth asymptotic boost and spatial map

A raw Lorentz boost of Schwarzschild static time carries its radial logarithm
into the compact spatial map. Instead remove that outgoing logarithm smoothly.
Let R*=R+2M ln(R/(2M)-1) on the outer branch and

    g(R)=1/[A(A+RS)], J(R)=integral_R^infinity g(R')dR',
    C=-R*(4M)-J(4M), H=R*+C+J,
    σ(ρ)=ρ+S(R)(R−ρ), Z^a=σ Y^a/ρ,
    L(ρ)=S(R)[R*(R)-σ+C], U=T-L(ρ).

Set L=0 on the entire inner isotropic branch. On the selected slice,

    U_s(Y)=J+(1-S)(R*+C)+Sσ-vY^x B,

with U_s=-vY^x B inside. Beyond R=12M, σ=R and U_s=R+J=sqrt(R²+1)+O(M/R⁴),
without a logarithm or constant offset. Now boost

    T_lab=Γ(U+vZ^x), X^x=Γ(Z^x+vU), X^y=Z^y, X^z=Z^z,
    Γ=1/sqrt(1-v²).

Near the puncture B=1 and L=H=0: on the initial slice T_lab=0,
Y^x=Γ X^x, Y^y=X^y, Y^z=X^z. This is the standard +v isotropic Lorentz
boost locally, with the hole worldline X^x=v T_lab. At infinity the same boost
defines the asymptotic frame; the spherical hyperboloidal cut has a regular
Lorentz-transformed representation. Do not confuse a mere constant shift or
localized coordinate drift with this induced metric/extrinsic-curvature data.

Use the base radial map for the lab hybrid radius:
X^i=σ(ρ_lab(r))x^i/r. Recover all three Y components with a Newton solve
of X(Y), using its analytic Jacobian. The hybrid radius is isotropic inside
and areal outside; at zero boost its intermediate coordinate change cancels
exactly. Check the positive Jacobian and Q at every initialized grid point.
Pull back γ_ab and K_ab with dY^a/dx^i. Then, with Ω=(1-r²)/2,

    χ = det(Ω²γ)^(-1/3), gtilde_ij=χΩ²γ_ij,
    K=γ^ij K_ij, Atilde_ij=χΩ(K_ij-γ_ij K/3),
    Lambda^i=-partial_j gtilde^ij (det gtilde=1), Theta=0.

The connection must be initialized from these nonspherical metric derivatives;
leaving it zero would introduce Z-constraint error. The actual scri values and
connection limits must be checked separately before evolution; a flat-looking
limit of the metric does not imply a zero connection limit. Evolution gauge
fields and compatible dependent-D/null closure must be distinguished from the
embedding lapse/shift.

## Exit estimate and planned evolution gate

In the inner map, the initial kinematic estimate is dx_p/dt≈k v=0.03061422834.
The Cartesian cube has half width0.18, giving estimated t_exit/M≈17.63886.
The more conservative inner shell support radius0.16 gives15.67898M. Both exceed
10M, so the proposed target is10M (coordinate time10/3). This is an estimate,
not a prediction under the evolving live gauge. The native puncture tracker
must integrate dx_p^i/dt=-beta^i(x_p), report its position and distance from the
core edge, and stop safely if the supported Cartesian region is approached.
Fast-flow must follow the tracked center, retain acceptance flags and use the
invariant horizon mass M (not boosted total energy ΓM) as its reference.

Before submitting N16: verify spacelikeness and invertibility throughout the
actual mesh; zero-boost recovery; analytic/independent constraint residuals and
sampled-grid convergence; finite scri limits and null/regularity compatibility;
tracker interpolation/restart identity and moving horizon center. Preserve a
separate checkpoint identity and frozen provenance. N16 must complete the
specified target with admitted finite state before N20/N24 are enabled. Numerical
failure is terminal and must be localized, not replayed with changed parameters.

## Native implementation and bounded checks

`native_boosted_wormhole.hpp` evaluates this construction directly in C++.
First spatial derivatives propagate through the analytic map Jacobians; Lambda
is obtained by automatic differentiation, not a mesh finite difference.
An independent Python embedding agrees in primary geometric fields to5.3e-14;
its fourth-order finite-difference connection agrees within3.7e-10. Zero boost
recovers a conformally flat metric and zero Lambda within2.2e-13. Independent
physical vacuum H/M finite-difference residuals decrease approximately sixteenfold
on halving a local step, until cancellation/roundoff limits H. These checks are
not production-grid convergence or long-time stability evidence.

At scri, the hybrid areal chart gives gtilde−I=O(Ω³), χ−1=O(Ω³),
Lambda=O(Ω²), K=−3 and Atilde=0. Sampled approach to scri is preserved in
embedding-probe.json; the surface uses the exact baseline limits and retained
null/dependent-D closure. The physical horizon geometry is boosted and is not
initialized by changing shift alone.

Gauge data: retain the base positive precollapsed lapse and radial shift, plus
a localized compact-coordinate pullback of −v e_x multiplied by B for the
initial shift. Initial D=0; geometric Lambda is initialized independently.
The evolution still determines the subsequent coordinate velocity. The existing
positive-chi floor patch r<.03 remains origin-centered to retain the baseline
floor rule; finite nonpositive-chi rescue remains global. The tracker does not
silently translate that floor or change the equations. Its effect as the puncture
moves must be visible in the recorded floor cells and pre-floor counts.

BX1 has a208-byte header: the baseline168-byte configuration followed by five
float64 values (v, puncture x/y/z, support limit.16). All four RK stages integrate
x_p'=-beta(x_p) using six-node tensor interpolation of filled Cartesian halos.
Tracker state rolls back on a rejected step and is part of the admitted checkpoint.
Fast-flow is centered on x_p at each diagnostic; the origin-centered exterior
sphere uses |x_p|+max sampled horizon radius, with candidate/acceptance flags
retained. This differs from the old centered mean-radius cut and must not be
silently compared as an identical integration domain.

All N16 active cells initialized with positive slice Q and positive map Jacobian;
no field-table input is used. A two-step SERIAL puncture run completed, with an
accepted horizon at both diagnostics, initial mass0.3333345527, null residuals
below1.12e-15, pole2.13e-8 and clock3.86e-7. No chi/lapse floor intervention
occurred. One-rank/two-rank and uninterrupted/restarted two-step checkpoint
payloads and tracker positions are byte-identical. At coordinate t=.001 the
tracker is x=3.05225606e-5; transverse motion is below1e-20. These are bounded
CPU/communication tests, not GPU or long-time stability results.

Current status: preparing separately frozen Aurora build and N16 evolution.
No boosted evolution has been submitted; no claim of a completed boosted test.
