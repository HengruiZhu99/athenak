# Nonlinear spherical hybrid: derivation and implementation audit

Status: nonlinear vacuum and coupled scalar regular-center kernels implemented;
actual-scri finite-difference closure is an experimental candidate, not yet admitted.
Scalar weak dispersal is a pilot; puncture evolution remains unimplemented.
The supplied linear theorem is not evidence of nonlinear stability.

## Variables and exact spherical restriction

Use physical metric
`gamma_phys = (Omega^2 chi)^(-1) diag(g, r^2 b, r^2 b)`,
`b = g^(-1/2)`, `Omega=(1-r^2)/2`, reference lapse `A=(1+r^2)/2`,
reference shift `B=-r`, reference physical trace `K*=-3` (S=a=1).
Evolved fields are chi, g, A_rr, p=K_phys-2T, T=Theta_phys, Lambda^r,
and alpha. Beta is an elliptic stage variable. Transverse curvature is
`A_theta_theta/r^2 = -A_rr b/(2g)`. Eliminating b and transverse A enforces only
determinant and trace algebraic constraints. Lambda is evolved independently;
`Z_r = g (Lambda^r - V^r)/2` is a diagnostic, never reset.

The existing `SphericalJet` constructs Cartesian tensors on the radial axis
using n_i=x_i/r and its first and second angular derivatives. Thus scalar
transverse Hessians are f'/r, radial vectors have transverse first derivatives
f/r, and tensor angular derivatives include g-b. Passing these exact jets to
`ConformalRHS` is an exact spherical restriction of the tensor equations, not
replacement of Cartesian derivatives with radial ones. New code does not use
the historical staggered driver or its background RHS subtraction.

The interior equations are the complete regular/pole expressions in
`src/z4c/hyperboloidal/conformal_rhs.hpp`, with p and T in physical normalization.
Each field obeys `U_t=R_U+S_U/Omega`, without an Omega floor. The adapter
`tst/elliptic_shift_1d/spherical_api.cpp` preserves all these expressions.
Independent full Appendix C term comparison is still required; a successful
spherical restriction of the repository does not prove the repository correct.

## Nonlinear elliptic shift, derived from the metric

For the flat spherical reference connection and det(g_tilde)=det(g_flat),

    V^r = g' / g^2 + (2/r)(sqrt(g) - 1/g).
    Q := g_t = beta g' + (4g/3)(beta' - beta/r) - 2 alpha A_rr.
    V_t = Q'/g^2 + [-2g'/g^3 + 1/(r sqrt(g)) + 2/(rg^2)] Q.

These follow by contracting the Christoffel difference and differentiating
that expression in coordinate time. The gauge is V_t=-eta_D V, eta_D constant
and nonnegative. It refers to V, not Lambda; no connection error is projected.
Define

    P = -2g'/g + (g^(3/2)+2)/r,
    C = 4g/3, D = g' - 4g/(3r).

The actual nonlinear second-order elliptic equation is

    C beta'' + (C'+D+PC) beta' + (D'+PD) beta
      = 2(alpha A_rr)' + 2P alpha A_rr - eta_D g^2 V.

The principal coefficient is positive when g>0. At g=1 it reduces to the
radial vector Lame operator `(4/3)(beta''+2beta'/r-2beta/r^2)`.
This derivation includes all nonlinear coefficient and 1/r terms.

For a regular center, g=1+O(r^2), A_rr=O(r^2), alpha=alpha0+O(r^2),
V=O(r), beta=O(r). The integrating factors are

    F(r) = integral_0^r (g^(3/2)-1)/s ds,
    mu = r^3 exp(F) / g^2,
    Q = -eta_D / mu * integral_0^r s^3 exp(F) V ds,
    H = (3/4) g^(-1/4) (Q + 2 alpha A_rr) / r,
    beta(r) = r g^(-3/4) [g(S)^(3/4) beta(S)/S - integral_r^S H ds].

The homogeneous Q solution is c/mu~c/r^3 and violates center regularity;
therefore c=0. The homogeneous beta solution is c r g^(-3/4); its coefficient
is fixed by the outer Dirichlet datum. This proves uniqueness in the stated
regular-center class, and gives a direct O(N) quadrature solve at every RK
stage. There is no iterative tolerance; independently change quadrature order
and compare with a second-order boundary-value discretization to separate
elliptic error from evolution truncation error. A zero reported expanded
residual alone is insufficient because analytic beta derivatives satisfy the
ODE algebraically; tests must also differentiate the sampled beta independently.

In spherical symmetry eta_D=0 implies Q=0: the conformal metric shape stays at
its initial value through the metric equation. This is a consequence of this
gauge and regularity, not a metric projection or a CMC constraint. Chi, p,
A_rr, T, and Lambda remain free. In particular K_phys=p+2T need not be constant.

## Lapse and actual scri

The original control retains the reference-reduced lapse from
`UnfactoredReferenceGauge` with zero additional restoring coefficient,
one-plus-log interior coefficient,
`F_lapse=alpha^2+4 nu Omega^2 alpha`, nu>=0, and p-p* as driver. Thus

    R_alpha = beta (alpha'-A') + (beta-B) A',
    S_alpha = -F_lapse (p-p*)
              - [A(beta-B)+B(alpha-A)+(alpha-A)(beta-B)] Omega'.

This is a nonlinear reference prescription, not linear formulas with substituted
live coefficients. Its linearization is the supplied reference lapse. Clock
condition alpha(S)=A(S)=1 is imposed only on lapse. Scri null-normal shift is
`beta(S)=-alpha(S) sqrt(chi(S)/g(S))`. Its linearization is
`delta beta_n=-delta alpha + A(delta g_nn-delta chi)/2`, as required.

The current candidate adds `-xi (alpha+A)(alpha-A)` to `S_alpha`, xi=1.
This changes the lapse gauge and leaves Einstein/Z4 bulk sources and the lapse
principal coefficient unchanged. It is authorized by the user's equation/gauge
revision, but is not identical to the original linear-note prescription. For
`chi=1+Omega C`, `alpha=A+Omega L`, g=1, the raw boundary rates are
`chi_t=C0-2L0` and `alpha_t=C0-2xi L0`. Thus xi=1 aligns these two necessary
gauge compatibility rates. This Taylor identity alone does not prove closure.
Zero-restoring controls lose scri compatibility in resolved pulse evolutions.

## Canceled g=1 implementation and regular-center limitations

For initial g=1, metric-connection freezing implies g_t=0 exactly. The separate
`g1_rhs.py` backend analytically cancels common reference and trace-free terms;
it does not subtract a numerical equilibrium RHS. Its independent, nonlinear,
off-constraint native-tensor comparison has maximum error 7.6e-14 over 24 cases,
including both Theta sources and lapse restoring coefficients 0 and 1.
Curvature, Khat, Theta and Lambda are independent evolved fields. This backend
currently cannot initialize general g(r) or positive puncture chi/lapse variables.

Regular-center parity is even for chi, g, A_rr, Khat, Theta, alpha and odd for
Lambda. Isotropy additionally requires g-1,A_rr=O(r^2), Lambda=O(r). Current
finite differences use reflected data, with a candidate KO6 filter on A_rr/r^2,
Lambda/r and the scalar deviations. The filter has fixed taper
`max(1-(r/.4)^2,0)^6` and coefficient epsilon/(64h). It vanishes near scri and
does not reset differential constraints. Epsilon=.1 admits sixth-order bulk
convergence at t=1 but a fine-grid central instability by t=3. Increasing the
coefficient and reducing derivative order are controlled experiments, not
accepted defaults. An invertible origin-variable chain-rule discretization
failed earlier and is rejected.

The optional actual-origin grid uses analytic limits f'/r=f''(0), Lambda/r=
Lambda'(0), A_rr/r=0. Its scalar wave and geometric coefficient checks pass,
but its N1025 lapse-pulse evolution fails positivity at t=2.256494. It is not a
stability repair. The actual-scri endpoint is present in both origin-grid variants.

## Coupled scalar equation actually implemented

Use `f=R phi/r=phi/(Omega sqrt(chi))` and `P=n_bar(R phi)/r`, both even at a
regular center. Here R=r/(Omega sqrt(chi)), n_bar is the normal for the conformal
spacetime metric, K=Khat+2Theta, and `k=K/3-Omega A_rr/2`. Then

    phi_r=sqrt(chi)[Omega f_r+(-r+Omega chi_r/(2chi))f],
    n_bar(phi)=sqrt(chi)[Omega P+k f].

The matter additions to the regular RHS are

    R_A -= (16pi/3) alpha chi phi_r^2,
    R_Khat += 8pi alpha Omega n_bar(phi)^2,
    R_Theta -= 4pi alpha Omega [n_bar(phi)^2+chi phi_r^2],
    R_Lambda += 16pi alpha n_bar(phi) phi_r.

Hamiltonian and momentum diagnostics include respectively
`-8pi Omega^2[n_bar(phi)^2+chi phi_r^2]` and
`+8pi Omega n_bar(phi) phi_r`. The exact radius-weighted equation is
`box_bar2(R phi)-V(R phi)=0`, with
`V=2m/(Omega^2 R^3)-Ric4_angular/Omega^2`. On the free-Z4c equations,

    Ric4_angular = -Omega^2 chi Lambda/r +Omega^2 chi_r Lambda/3
                   -(2/3)Omega r chi Lambda +kappa1(1+kappa2)Theta.

The repository Theta variant adds `2 w Theta`, w=beta r/alpha. Retaining these
off-constraint terms matters: dropping them silently changes the scalar wave
equation. Eight independent comparisons with the native geometric evolution
give angular Ricci error below 1.8e-15. A separately evaluated analytic Minkowski
wave gives RHS convergence orders 5.95 and 5.99.

With `alpha Kbar_rad=-1+alpha A_rr+U/Omega`,
`U=alpha(K+3)/3-(alpha-A)-r(beta+r)`, the evolved scalar equations are

    f_t=beta f_r+(beta/r)f+alpha P,
    P_t=beta P_r+(beta/r+alpha Kbar_rad)P
        +alpha chi(f_rr+2f_r/r)+(.5alpha chi_r+chi alpha_r)f_r
        +[(.5alpha chi_r+chi alpha_r)/r-alpha V]f.

The actual-scri limits use U/Omega=-U_r and V=0 for finite mass and compatible
constraint weights. The unfactored potential numerator and its first derivative
are measured separately. No Omega floor is used. At a regular origin, the flat
limit is `f_t=-f+P/2`, `P_t=-2P+3f_2+f_0` for f=f_0+f_2 r^2+... .

Constraint-solved scalar initial data use psi=1+Omega^3 w, chi=psi^-4,
A_rr=Theta=Lambda=0, K=-3 and n_bar(phi)=0. The nonlinear regular Hamiltonian
equation is

    Omega w''+(5Omega'+2Omega/r)w'-15w
      -7.5Omega^2w^2-7.5Omega^5w^3-3.75Omega^8w^4-.75Omega^11w^5
      +pi(1+Omega^3w)phi_r^2/Omega^2=0.

Evenness at the origin and w'(1)=-3w(1) close the solve. Bondi initial mass is
8w(1). For amplitude .058, independent ADM residuals and volume-integrated
scalar energy show approximately sixth-order convergence. Newton tolerance
1e-11 encounters a double-precision residual floor about1.6e-11 at N512;
the explicit candidate tolerance1e-10 remains below measured discretization
error. Tightening/precision isolation is still required. This solve is only at
initialization; evolution never projects Hamiltonian or momentum errors.

At amplitude .01, N256 reaches t=3 with H/M RMS about3.1e-12/3.8e-12. The
initial mass .003132366825394506 minus RK4-integrated coordinate-time scalar
flux .003132355606598225 is1.1219e-8. The full Bondi-time normalization,
three-resolution flux study, tails and collapse remain unaccepted.

For each geometric/Z4 field, combine its entire pole numerator before division.
If S_U(S)=0 and sufficient differentiability holds, the nonlinear limiting
identity is `U_t(S)=R_U(S)+S_U'(S)/Omega'(S)=R_U(S)-S_U'(S)`.
This identity does not by itself prove propagation of the nonlinear corner
compatibility hierarchy. The prototype records S_U(S), S_alpha(S), nullness,
clock error, and weighted constraints; it does not overwrite geometric boundary
values to force cancellation. A finite-difference implementation may fail those
gates and must retain the failure. No nonlinear well-posedness is claimed.

## Scalar-Theta source audit

At C_Z4c=0 the repository has

    T_t = beta T' + alpha H/(2 Omega) + Omega alpha chi D_tilde_i Z^i
          - alpha[3w + kappa1(2+kappa2)]T/Omega,
    w = n_bar(Omega) = -beta Omega'/alpha.

Appendix C (48j) of https://arxiv.org/html/1412.3827 has no 3w term. The supplied
candidate explicitly adds `+3 alpha w T/Omega`. The adapter changes only the
Theta pole numerator. The repository control retains its original equation.
All other off-constraint tensor sources are unchanged. Candidate values are
kappa1=4, kappa2=-.99; published regular-spacetime control is kappa1=1.5,kappa2=0.
Switching this source changes the off-constraint formulation. The linear note's
rejected finite curvature constraint additions are not implemented.

## Massless scalar derivation (not yet implemented)

Let phi be the physical scalar and v=n_bar(phi)=(phi_t-beta phi')/alpha.
Physical normal momentum is Omega v. Then

    rho_phys = Omega^2 [v^2 + (chi/g) phi'^2]/2,
    j_r_phys = -Omega v phi',
    S_ij_phys = phi_i phi_j + gamma_bar_ij [v^2-|D_bar phi|^2]/2.

Using `K_bar=(p+2T-3w)/Omega` in physical box(phi)=0 gives

    phi_t = beta phi' + alpha v,
    v_t = beta v' + alpha [D_bar^2 phi + (p+2T-w)v/Omega
                            - 2 D_bar Omega . D_bar phi/Omega]
                    + D_bar alpha . D_bar phi.

Here `D_bar^2 phi = (chi/g)[phi''+(2/r+b'/b-g'/(2g)-chi'/(2chi))phi']`.
Physical Einstein matter terms, expressed in the adapter's variables, are

    delta p_t = +8 pi alpha Omega v^2,
    delta T_t = -4 pi alpha Omega [v^2+(chi/g)phi'^2],
    delta A_rr_t = -(16 pi/3) alpha chi phi'^2,
    delta Lambda_t = +16 pi alpha v phi'/g,
    delta H = -8 pi Omega^2 [v^2+(chi/g)phi'^2],
    delta M_r = +8 pi Omega v phi'.

These agree in scaling with (48d,e,f,j) and (49a,b) after converting the paper's
coordinate-time Pi to alpha v+beta phi'. They require an independent stress
projection test before use. Rescaled extraction field is f=phi/Omega; using
phi directly at scri requires limits and is not an implemented scalar closure.
The conformal wave identity is
`box_bar(f)-R_bar f/6+(4pi/3)(|D_bar phi|^2-v^2) f=0` on Einstein solutions.
R_bar and the endpoint equation must be derived without assuming the Einstein
constraints vanish in the free-Z4c evolution. This remains open.

## Wormhole data and puncture end (not yet implemented)

Do not reuse regular-center parity for a puncture. In the isotropic interior,
`R=rho (1+M/(2rho))^2`, gamma_phys=psi^4 delta, K_ij=0, and prescribed initial
physical lapse is `alpha_phys=psi^-2`; conformal lapse is Omega alpha_phys.
As rho->0, R~M^2/(4rho), proper distance integral psi^2 d rho diverges as 1/rho:
this is a second asymptotically flat wormhole end, not a trumpet (finite limiting
R and logarithmically divergent proper distance). Beta and Z4 conditions at
this end must be derived using puncture expansions, not center reflection.

A legitimate hyperboloidal exterior can be glued as a slice in exact
Schwarzschild spacetime, rather than gluing metric components. In either
exterior use Schwarzschild time T=t+h(R), F_S=1-2M/R. The induced metric is
`gamma_RR=1/F_S-F_S h'^2`; choose h'=0 in the isotropic interior exterior
region and asymptotic `h'~1/F_S` with positive induced metric in the outer
exterior. A smooth height transition begins strictly outside R=2M, where static
coordinates are regular, and leaves the Einstein-Rosen throat untouched.
Compute the slice normal and K_ij from this embedding. An isotropic rho(R)
map can then be integrated in the transitioned exterior and compactified.
Since these are induced data in exact vacuum Schwarzschild, constraints hold
analytically; verify discretely and by invariant mass. The initial lapse is a
freely chosen gauge function on that slice: smoothly interpolate physical
psi^-2 to A/Omega in the exterior using the same regular transition region.
This is a construction recipe, not yet constraint-verified initial data or an
executed wormhole-to-trumpet evolution. In particular the elliptic inner-end
uniqueness argument must be re-derived before applying the center solver.

## Implemented wormhole embedding and precision audit

`wormhole_data.py` now implements the preceding construction for a=S=1, M=.1.
Set rho=r/(1-r), R=rho+M+M^2/(4rho), F=1-2M/R. On the outer exterior only,
let s(R) be an infinitely smooth transition from zero at R0=4M to one at R1=8M.
The second asymptotically flat end is explicitly excluded from this transition.
Define `P=s (R/a)/sqrt(F+(R/a)^2)`, `h_R=P/F`. Then

    gamma_RR=(1-P^2)/F,
    H=n_phys(R)=P sqrt(F/(1-P^2)),
    K_R^R=-dH/dR, K_theta^theta=-H/R,
    K_phys=-dH/dR-2H/R.

The whole neighborhood of the Einstein-Rosen throat remains on the ordinary
isotropic time-symmetric slice; the height transition is strictly outside the
horizon, avoiding Schwarzschild static-coordinate singularities. In the outer
region H=R/a, hence K_phys=-3/a and the physical trace-free curvature vanishes.

With bar_rr=Omega^2 gamma_rr, bar_tt=Omega^2 R^2/r^2, set
`chi=(bar_rr bar_tt^2)^(-1/3)`, `g=chi bar_rr`,
`A_rr=g(K_R^R-K_phys/3)/Omega`, and `Lambda=V(g)` initially. The latter sets the
initial connection constraint correctly; Lambda is not subsequently slaved.
Initial conformal lapse is `(1-s)Omega psi^(-2)+s A`. The physical lapse is
therefore exactly psi^(-2) in the interior and A/Omega in the outer hyperboloidal
region. The hyperboloidal Minkowski reference is unchanged.

Actual-scri evaluation uses the regular rational functions Y=Omega R and
W=Omega^2 R', with `bar_rr=W^2/(Omega^2 F+Y^2/a^2)` in the outer region.
This avoids infinity times zero at r=1. Jets are computed by second-order
forward automatic differentiation. Kr derivatives use H'' and R'', sufficient
for A_rr' and p'; their unused second derivatives are not claimed exact.

The tests cover 405 points, the actual scri pole numerators, invariant mass,
and explicit puncture samples. First testing found loss of accuracy from
reconstructing small chi as 1+(chi-1). A separate positive-chi adapter fixes
that cancellation for these initial-data checks. Generic Misner-Sharp mass
extraction still loses precision near scri by subtracting large R^2 terms;
its maximum error here is about 2.61e-10. The evolved vacuum prototype still
uses chi deviations and has not been admitted for puncture evolution.
The initial-data test uses beta=-r only for gauge-independent diagnostics;
it is not evidence of solving the puncture elliptic gauge or evolving the data.

## Endpoint discretization experiments and disclosed penalties

The raw l'Hopital candidate fails through a discrete endpoint mode whose
N24 growth rate is about232.44. Over99.999% of its nodal energy is at r>.9,
mostly Lambda and A_rr. The minimal uncoupled model
`T_t=-c T/Omega` with endpoint `T_t=c D_r T` already has a positive endpoint
rate c D_r(S,S), about232.60 for c=4.04. This mode does not belong to the
compatible weighted continuum phase. It explains why a naive raw nodal
extension does not follow from the supplied linear graph-phase theorem.

The quotient reconstruction candidate extrapolates the *combined* interior
S/Omega to scri, consistent with the regular quotient trace. It does not change
or reset fields and records the independently computed S(S). It reduces but
does not remove the instability; N64 pulse positivity fails near t=1.012.

Two optional weak boundary penalties are now explicit numerical experiments:

    U_t(S) <- U_t(S) + sigma S_U(S)/h - tau C_U(S)/h,
    C = [chi-g, 0, A_rr-g'/6+2chi'/3, p+3, T, Lambda-V, 0, 0].

Clock and elliptic shift conditions are applied separately. These trace
relations follow from the nonlinear pole cancellations, alpha(S)=1 and nullness:
S_chi=0 gives K_phys=3w; S_p=0 then gives T=0; S_alpha=0 gives beta=-1,
chi=g and p=-3. The trace-free curvature residue gives A_rr=g'/6-2chi'/3.
The compatible phase requires Z=0 at scri. The penalties vanish identically
on such regular continuum solutions and never reset a field or an interior
Hamiltonian, momentum, Theta or connection error. They are additional *boundary
numerical terms*, not part of the continuum bulk equations. Their effect on
constraint propagation and convergence must be tested, not presumed acceptable.

Parameter screens preserve every result. sigma=4,tau=8 removes the observed
O(1/h) endpoint growth of the sampled Jacobian. A remaining positive complex
mode has real part about1.32,1.27,1.22 at N16,32,64; this finite growth rate
is not a continuum stability proof. Unperturbed N64 Minkowski remains within
5e-13 through t=3, but the 1e-3 lapse pulse has order-unity drift and violates
pole/constraint thresholds by t=6. These penalties are not an accepted repair.
No claim of completing the main hybrid formulation follows from these screens.
