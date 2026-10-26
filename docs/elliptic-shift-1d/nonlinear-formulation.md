# Nonlinear spherical hybrid: derivation and implementation audit

Status: nonlinear vacuum interior kernel and regular-center elliptic gauge implemented;
actual-scri finite-difference closure is an experimental candidate, not yet admitted.
Matter and puncture formulae below are derivations awaiting independent tests.
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

Retain the reference-reduced lapse from `UnfactoredReferenceGauge` with zero
additional restoring coefficient, one-plus-log interior coefficient,
`F_lapse=alpha^2+4 nu Omega^2 alpha`, nu>=0, and p-p* as driver. Thus

    R_alpha = beta (alpha'-A') + (beta-B) A',
    S_alpha = -F_lapse (p-p*)
              - [A(beta-B)+B(alpha-A)+(alpha-A)(beta-B)] Omega'.

This is a nonlinear reference prescription, not linear formulas with substituted
live coefficients. Its linearization is the supplied reference lapse. Clock
condition alpha(S)=A(S)=1 is imposed only on lapse. Scri null-normal shift is
`beta(S)=-alpha(S) sqrt(chi(S)/g(S))`. Its linearization is
`delta beta_n=-delta alpha + A(delta g_nn-delta chi)/2`, as required.

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
