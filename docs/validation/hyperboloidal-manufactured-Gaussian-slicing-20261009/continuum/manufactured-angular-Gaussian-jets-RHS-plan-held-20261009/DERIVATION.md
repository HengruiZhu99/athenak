# Third embedding jets and consumed Z4c data for the pure-time Gaussian wave map

This is a source/pencil-only proposal. No implementation, numerical/CAS work,
new inverse solve, kernel query or evolution is included. The same Minkowski
hyperboloidal reference is retained. The proposed positive sampled family is
sigma=7/20, epsilon=3/4, with epsilon=0 as reference. The completed value screen
is only finite sampled admissibility context. Its sigma=1/2, epsilon=3/4 negative
future-D events remain negative controls; no ADM square root is allowed there.
The manufactured map solves the full physical-reference wave-map gauge on the
Einstein sector. It does not solve the new inner modified-BM helper in general.

## Reference and exact inverse jets

Use native y=(t,x1,x2,x3), fixed Omega(x), Q=x/Omega, R=|Q| and
Yhat0=t+H(R). The physical inertial inverse is X=(T,Q), determined by

    T+epsilon F(T,Q)=t+H(R),
    F=Q1 Q2 C(T,R), f(T)=sigma^4 exp(-T^2/(2sigma^2)).

All derivatives below are ordinary derivatives, with time first in the index
order. Let e_a^I=Q^I_,a, E_a^A=(T_a,Q^I_,a), and
J=1+epsilon F_T. Define B(y)=t+H(R(y)). Then

    T_a=(B_a-epsilon F_I Q^I_,a)/J,
    T_ab=(B_ab-epsilon F_AB E_a^A E_b^B
                    -epsilon F_I Q^I_,ab)/J,
    T_abc=(B_abc-epsilon F_ABC E_a^A E_b^B E_c^C
       -epsilon F_AB [E_ab^A E_c^B+E_ac^A E_b^B+E_bc^A E_a^B]
       -epsilon F_I Q^I_,abc)/J,
    E_ab^A=(T_ab,Q^I_,ab).

The last formula excludes the unknown T_abc from the explicit E terms; its
coefficient has already been moved into J. Q and B have no time dependence
apart from B_t=1. This is the full compact Cartesian composition of the pinned
(t,Q) implicit formulas, including second/third spatial embedding terms.
Neither radial-only differentiation nor holding the inverse T fixed while
forming angular jets is permissible.

In the transition, Omega through three ordinary Cartesian derivatives is
required for Q through three. H through three is required for B. With the
actual reference L=Omega-r Omega_r, b=2rw at S1,a=.5 and
alpha_hat=sqrt(Omega^2+b^2),

    d/dR=(Omega^2/L)d/dr,
    H_R=b/alpha_hat,
    H_RR=(Omega^2/L)d_r(b/alpha_hat),
    H_RRR=(Omega^2/L)d_r[(Omega^2/L)d_r(b/alpha_hat)].

The scalar height value retains the accepted panel integral and H(0)=0. Its
jets use these analytic derivatives, not derivatives of a quadrature routine.
The exact Cauchy branch r<=.05 and exact outer branch r>=.95 are separate
formulas. Smooth-cutoff derivatives at both endpoints must be exactly zero;
near-endpoint complementary tails must be retained even when the cutoff value
has rounded to an endpoint. A LayerPoint interface exposing only second jets
cannot be used to invent missing third embedding/reference data.

## Bounded outer implicit jets

For r>=.95 set z=Omega/r, n=x/r, p=n1*n2, s=t+C_H,

    u=s+z c, v=u+2/z,
    Phi=f2(u)-f2(v)+3z[f1(u)+f1(v)]+3z^2[f0(u)-f0(v)],
    G(c,y)=c+epsilon p Phi-a^2/[sqrt(1+a^2z^2)+1]=0.

Here G_c=J. The unknown is the bounded c=c_ret, not T-R or (u-s)/z computed
by subtraction. For an independent variable vector y and augmented tangent
E_a=(e_a,c_a), complete implicit differentiation gives

    c_a=-G_a/G_c,
    c_ab=-G_AB E_a^A E_b^B/G_c,
    c_abc=-[G_ABC E_a^A E_b^B E_c^C
       +G_cA(E_a^A c_bc+E_b^A c_ac+E_c^A c_ab)]/G_c.

All partial derivatives on the right hold c independent. These formulas retain
every mixed t/r/angular derivative and advanced tail. They are equivalent to
the preceding T composition, but avoid forming large cancelling T/R values as
the primary metric algorithm. Gaussian advanced terms are analytically flat
at z=0; at finite positive z they must be evaluated in a scaled form when
needed, not dropped because an unscaled binary64 seed has underflowed.

Use the pinned compact profiles Aplus/Aminus, then

    h=1/sqrt(1+a^2z^2), eta=a^2/[sqrt(...)(sqrt(...)+1)],
    J=1+epsilon p z(Aminus+z Aplus)/2,
    wr=h+epsilon p z(Aminus-z Aplus)/2,
    km=eta+epsilon p Aplus, jp=1+h+epsilon p z Aminus,
    tangent=n2 e1+n1 e2-2p n,
    Delta=km*jp-epsilon^2 z^2 Phi^2 |tangent|^2.

Thus D=z^2 Delta, D/Omega^2=Delta/r^2 and J-wr=z^2 km exactly. Form
Delta directly, retaining its real subtraction and rejecting Delta<=0.

The primary conformal fields through total spacetime order two are

    bargamma=I-nn+L^2 km jp/(r^2 J^2) nn
      +epsilon L z wr Phi/(r J^2)(n tangent+tangent n)
      -epsilon^2 z^4 Phi^2/J^2 tangent tangent,
    alpha=r/sqrt(Delta),
    beta=-r^2 wr/(L Delta)n+epsilon Omega Phi/Delta tangent,
    det(bargamma)=L^2 Delta/(J^2 r^2),
    chi=[J^2 r^2/(L^2 Delta)]^(1/3), gtilde=chi*bargamma.

Differentiate these identities as complete Cartesian spacetime jets. Their
values already include the implicit T/c dependence. Three inverse/embedding
jets are sufficient for this consumed schema; third jets of the ADM fields
are not requested. Genuine small Delta/J conditioning is not removed by the
factorization and must not be hidden by a floor.

## Curvature, connection and all twenty-two independent rates

Let the lapse above be the Penrose lapse and beta contravariant in native
Cartesian coordinates. Compute the conformal extrinsic curvature directly:

    barK_ij=-(dt bargamma_ij-Lie_beta bargamma_ij)/(2alpha),
    omega_n=-beta^i Omega_i/alpha,
    barK=bargammaInv^{ij}barK_ij,
    Kphys=Omega*barK+3omega_n,
    Theta_phys=0,
    P=Kphys-2Theta_phys=Kphys,
    Atilde=chi*(barK_ij-bargamma_ij*barK/3),
    Lambda^i=gtilde^{jk}Gamma[gtilde]^i_jk.

The full tensor Lie derivative includes both shift-gradient terms. These
formulas provide P/A/Lambda through total order one from the metric/lapse/
shift order-two jets. There is no construction of large physical K components
followed by a trace subtraction in the primary compact branch. For an
independent finite-radius check, the physical graph in (t,Q) has

    w=grad H-epsilon grad F, D=J^2-|w|^2,
    gamma_Q=I-ww/J^2, alpha_phys=1/sqrt(D), beta_Q=-w/D,
    K_Q,ij=-J*T_ij/sqrt(D).

Transform that graph tensor under Q(x) and compare against the compact
curvature. This graph check is secondary near scri because of cancellation.
The primary compact curvature has no forced division by Omega.

Independent time rates are the time derivatives of the constructed fields,
never copied from the RHS under test. Explicitly, chi_t=-(chi/3)tr(bargamma^-1
bargamma_t), gtilde_t=chi_t*bargamma+chi*bargamma_t, P_t=Omega*barK_t+3omega_n,t,
Theta_t=0; A_t follows the full product and trace derivative; Lambda_t follows
the differentiated contracted connection, including inverse-metric derivatives.
Alpha_t and beta_t are coefficients of the complete implicit field jets.
Every metric/curvature symmetry and mixed-derivative permutation must be
retained. Det(gtilde)=1 and tr_gtilde(A)=0 are identities, not projections.

The ordinary raw22 order is

    chi,gxx,gxy,gxz,gyy,gyz,gzz,P,
    Axx,Axy,Axz,Ayy,Ayz,Azz,Lambda_x,Lambda_y,Lambda_z,
    Theta,alpha,beta_x,beta_y,beta_z.

Alpha/beta/chi/gtilde/Omega export complete spatial jets through order two;
P/A/Lambda/Theta export through one. All22 rates export separately. Unavailable
P/Lambda/Theta second jets must remain absent/NaN sentinels in the binder,
not fabricated zeros. The second A jet has no native consumed slot.

## Exact origin

Use F=Q1Q2 C(T,R) as a Cartesian polynomial times a smooth function of R^2.
The regular integral is C=-1/8 integral_(−1)^1(1-v^2)^2 f5(T+Rv)dv.
At Q=0: F and first derivatives vanish, F_ij=−2 f5(T)E_ij/15,
F_Tij=−2 f6(T)E_ij/15, and pure third spatial derivatives vanish.
Consequently J=D=1 and the metric/lapse values are reference-like there, but
K_ij=epsilon F_ij generally does not vanish. An origin implementation that
sets all perturbation jets to zero is wrong. The polynomial/integral origin
representation, not a radial floor or a direction-dependent n, supplies these
jets and their implicit-time composition.
