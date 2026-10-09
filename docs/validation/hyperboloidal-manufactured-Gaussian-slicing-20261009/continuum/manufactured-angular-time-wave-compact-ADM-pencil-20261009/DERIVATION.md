# Compact ADM factorization for the angular pure-time wave map

This is an independent pencil/source-only follow-up to the frozen general-f
manufactured-wave derivation. No implementation, numerical/CAS evaluation,
inverse solve, compiler, kernel query or evolution is included. The reference
Minkowski geometry, compact map and physical-reference wave-map gauge remain
the same. No amplitude is admitted here. The separately pinned Gaussian plan
is checked only as symbolic context below.

The goal is to remove cancellation forced by powers of Omega near future scri.
It does not remove genuine ill-conditioning when a physical timelike margin
approaches zero, nor provide a universal floating-point error bound.

## Exact outer variables and bounded retarded-time solve

Use the exact outer CMC branch and r>0, Omega>0:

    Omega=(S^2-r^2)/(2aS), R=r/Omega, z=1/R=Omega/r,
    L=Omega-r Omega_r=(S^2+r^2)/(2aS)=alpha_hat,
    h=H_R=R/sqrt(R^2+a^2)=1/sqrt(1+a^2 z^2),
    H(R)=sqrt(R^2+a^2)+C_H.

Here h is the physical height derivative, not the reference conformal lapse.
Let n=X/R=x/r, p=n1*n2, and define the Cartesian tangent vector

    t_i=n2 delta_i1+n1 delta_i2-2p n_i,
    n.t=0, tau=t.t=n1^2+n2^2-4p^2.

The production Q symbol continues to mean (P-3 omega_n)/Omega; it is not used
for a new profile or light-cone combination in this note.

Set s=t_native+C_H, u=T-R, v=T+R=u+2/z. Introduce three exact profiles by
their displayed formulas, without division by p (which can vanish):

    Phi=f''(u)-f''(v)
          +3z[f'(u)+f'(v)]+3z^2[f(u)-f(v)],

    Aplus=-f''(u)-6z f'(u)-9z^2 f(u)
          -2 f'''(v)/z+7 f''(v)-12z f'(v)+9z^2 f(v),

    Aminus=2 f'''(u)+7z f''(u)+12z^2 f'(u)+9z^3 f(u)
          -z f''(v)+6z^2 f'(v)-9z^3 f(v).

These give, exactly at finite positive z,

    F=p z Phi,
    F_T+F_R=p z^2 Aplus,
    F_T-F_R=p z Aminus,
    grad_X F=n F_R+z^2 Phi t.

All advanced terms are retained. Aplus contains the normalized advanced tail
f'''(u+2/z)/z explicitly; it must not be set to zero merely because a rounded
unscaled seed value underflows. Bounded/smooth future limits require the decay
assumptions on these tails and their needed derivatives. A generic smooth f
without such decay need not give a smooth future end.

The exact inverse equation becomes

    u+epsilon p z Phi=t_native+C_H
             +a^2 z/[sqrt(1+a^2 z^2)+1].

No subtraction sqrt(R^2+a^2)-R or T-R is required to evaluate its arguments.
For a fully bounded unknown, solve for c_ret by

    u=s+z c_ret,
    c_ret+epsilon p Phi(s+z c_ret,s+z c_ret+2/z,z)
          =a^2/[sqrt(1+a^2 z^2)+1].

The derivative of the left side with respect to c_ret is exactly
J=1+epsilon F_T, since partial_u Phi differentiates u and v together.
Thus the previously established monotone time inverse applies to this
algebraically equivalent one-dimensional solve. c_ret is solved directly,
not recovered by subtracting nearly equal u and s and dividing by z.

On the future outgoing end with the stated advanced decay,

    c_ret -> a^2/2-epsilon p f''(s).

This is a bounded-root representation, not a numerical bracket or solver gate.
The exact global physical-time bracket from the separate Gaussian plan is a
different bound and remains available before selecting any numerical solve.

## Factoring the timelike margin

Define the reference small factor without subtracting 1-h:

    eta=(1-h)/z^2
       =a^2/[sqrt(1+a^2 z^2)(sqrt(1+a^2 z^2)+1)].

An equivalent expression using compact r is

    eta=4a^2 S^2 r^2/[(S^2+r^2)(S+r)^2].

Write w=grad H-epsilon grad F=w_r n+w_T and retain

    J=1+(epsilon p/2)z(Aminus+z Aplus),
    w_r=h+(epsilon p/2)z(Aminus-z Aplus),
    w_T=-epsilon z^2 Phi t,
    kminus=eta+epsilon p Aplus,
    jplus=1+h+epsilon p z Aminus.

The two crucial exact identities are

    J-w_r=z^2 kminus,  J+w_r=jplus.

Compute kminus directly rather than subtracting J-w_r. Then

    D=J^2-|w|^2=z^2 Delta,
    Delta=kminus*jplus-epsilon^2 z^2 Phi^2 tau,
    D/Omega^2=Delta/r^2.

The angular subtraction in Delta is a real physical timelike condition. It
cannot be clamped or assumed positive. J>0 and Delta>0 are the finite-event
admissibility requirements. At epsilon=0, eta*(1+h)=a^2 h^2, so

    Delta_ref=a^2 h^2,  D_ref/Omega^2=1/alpha_hat^2.

For the full future angular sphere, the leading minimum is

    Delta -> a^2-2epsilon p f''(s),
    min_n Delta -> a^2-|epsilon f''(s)|.

Positive leading limits still require uniform weighted-remainder control and
finite-annulus/core positivity. They do not establish those gates by themselves.

In an eventual arithmetic implementation tau may be formed from the explicit
sum t_i^2. The identity tau=n1^2+n2^2-4p^2 is useful analytically but subtracts
equal terms at true angular stationary points. This note admits no particular
binary64 rounding strategy or floor.

## Compact conformal ADM values

Let Pi_ij=delta_ij-n_i n_j. The fixed spatial embedding has radial eigenvalue
L/Omega^2 and tangential eigenvalue 1/Omega. Applying it to
gamma_Q=I-w w^T/J^2 yields the exact compact Cartesian expression

    bar_gamma = Pi
      +[L^2*kminus*jplus/(r^2*J^2)] n n
      +[epsilon L z w_r Phi/(r J^2)](n t+t n)
      -[epsilon^2 z^4 Phi^2/J^2] t t.

Every displayed coefficient is bounded on an admissible smooth future end.
In particular the radial metric coefficient uses the factored J^2-w_r^2,
not a difference of nearly equal terms multiplied by Omega^-2.

The lapse, contravariant compact Cartesian shift and determinant are

    alpha_bar=r/sqrt(Delta),
    beta_x= -[r^2 w_r/(L Delta)] n
                   +[epsilon Omega Phi/Delta] t,
    det(bar_gamma)=L^2 Delta/(J^2 r^2),
    chi=[J^2 r^2/(L^2 Delta)]^(1/3),
    gtilde=chi bar_gamma.

These formulas require J>0, Delta>0, r>0, L>0 and positive root conventions.
The determinant is evaluated from its analytic factorization rather than a
subtraction-prone cofactor expansion; an independent determinant check is a
later numerical gate. Reference limits give alpha_hat=L and chi=1 throughout
the exact outer branch. They do not imply chi=1 in the geometric transition.

At scri the leading angular lapse and radial shift are

    alpha_bar -> S/sqrt(margin),
    beta_rad -> -aS/margin,
    margin=a^2-2epsilon p f''(s),

because L_scri=S/a. The tangential shift vanishes in that limit. These are
values of a manufactured pure-gauge flat spacetime, not a BH or pulse prediction.

## Finite extrinsic data without a stretched physical trace subtraction

The physical metric can be reconstructed at any finite Omega if needed, but
near scri it is preferable to obtain conformal extrinsic curvature directly:

    bar_K_ij=-(partial_t bar_gamma_ij-Lie_beta bar_gamma_ij)/(2 alpha_bar),
    omega_n=-beta^i Omega_i/alpha_bar,
    bar_K=bar_gammaInv^{ij} bar_K_ij,
    P=Omega bar_K+3 omega_n, Theta_physical=0,
    Atilde_ij=chi(bar_K_ij-bar_gamma_ij bar_K/3),
    Lambda^i=contracted Gamma[gtilde]^i.

Here Lie_beta bar_gamma has its full tensor derivative terms. The conformal
identity bar_K_ij=Omega Kphysical_ij-(omega_n/Omega)bar_gamma_ij proves these
equal the physical graph expressions and the production P=Kphysical storage
on this Einstein sector. Atilde=Omega chi Kphysical_TF is therefore retained
without first forming large physical K components and subtracting their trace.
This avoids cancellation forced by Omega, but genuine trace-free cancellations
and small-lapse conditioning still need an arithmetic gate. Lambda is obtained
from metric jets and is not independently assigned or assumed to vanish.

The physical ADM constraints and Z=0 follow analytically from the exact flat
pullback. For a future executable oracle, they and the complete physical-RWM
source/time rows must nevertheless be independently checked from consumed jets.
The manufactured family generally does not solve the new coupled-inner helper.

## Complete third-jet pathway, transition and origin

Use ordinary multivariate jets, not nested finite differences. At outer finite
z, treat the c_ret equation as an analytic implicit scalar equation. Its
first/second/third derivatives follow the standard implicit chain rule with
denominator J, retaining every mixed t/r/angular derivative and every advanced
tail derivative. Equivalently use the pinned general-f T_a,T_ab,T_abc formulas;
they must agree after compact composition. Compose z=Omega/r, p=n1*n2 and
t=n2 e1+n1 e2-2p n with their full Cartesian derivatives. Radial-only jets are
insufficient for this angular family.

For the Z4c consumed schema, construct alpha_bar, beta, chi and gtilde through
total spacetime order2 from the factored expressions. Then bar_K/P/A/Lambda
through order1 and all22 independent time derivatives use those order2 jets.
The underlying inverse/embedding third jets supply exactly the required
derivative information. No P/A/Lambda second jets or missing fourth map jet
may be invented. If one instead asks for third jets of the ADM fields
themselves, additional map/profile derivatives are needed; that is not the
current consumed schema.

Generic LayerPoint2 reference jets are not enough: L derivatives through2
need Omega through3, and the height composition needs H through3. Exact outer
CMC formulas provide these without extending an incomplete interface. In the
transition, use the complete fixed reference jets and its actual height,
not the outer formula with a fitted constant. A later numerical interface must
pin the backend and all derivatives it consumes before any query.

At the center z and n are invalid coordinates. Retain the pinned regular
Cartesian representation

    F=X1 X2 C(T,R),
    C=-(1/8) integral_-1^1 (1-q^2)^2 f^(5)(T+Rq)dq,
    C=-2f5/15-R^2 f7/105-R^4 f9/3780+... .

The even radial series is a smooth function of R^2, so Cartesian polynomial
jets and Leibniz give complete origin data. The exact Cauchy reference height
is constant there. The scalar inverse is T+epsilon F=t+H_core and uses the
same implicit3 recurrences. At X=0, J=D=1, gamma=I, alpha_physical=1, beta=0,
but K_12=epsilon*(-2f5/15), so the live core is generally not constant geometry.
The regular integral needs f through8 for full F jets through3. No radius or
Omega floor is introduced, and matching a future origin/transition/outer
evaluation strategy is itself a later source/numerical gate.

## Exact fixed-physical-event angular minimum

Independently writing F=R^2 p C, s_ang=n1^2+n2^2 gives

    |grad F|^2=R^2 C^2 s_ang+4R^3 p^2 C C_R+R^4 p^2 C_R^2,
    A=C_T+h(C_R+2C/R), B=C_T^2-C_R^2-4C C_R/R,
    D=d0+2epsilon R^2 A p
          +epsilon^2(R^4 B p^2-R^2 C^2 s_ang).

At fixed physical T,R the s_ang coefficient is nonpositive. For every
p in[-1/2,1/2], s_ang=1 is attainable, so minimizing the resulting quadratic
over its endpoints and any interior vertex gives the exact sphere minimum.
Use the vertex only if its quadratic coefficient is positive and nonzero;
epsilon=0 and other degenerate cases are handled directly. At R=0 use origin
Cartesian data. T varies with p at fixed native t, so applying one angle's
inverse event to all angles is invalid. An independently enclosing physical
T interval can make this reduction a sufficient bound. No such finite-domain
minimization is performed or admitted here.

## Additive Gaussian-plan cross-check

The separately pinned source-only plan uses f= sigma^4 exp(-T^2/(2sigma^2)),
sigma=7/20 or1/2 and epsilon=0,1/4,1/2,3/4; no amplitude is admitted.
Direct Hermite differentiation gives M4=3 and M2=sigma^2, while the separate
maxima of |x|^3 exp(-x^2/2) and3|x|exp(-x^2/2) give the stated coarse
M3<=4sigma. Thus J>=1-4epsilon/pi>=1-3/pi>0, and the global scalar inverse
bracket width16epsilon sigma/(3pi), match the plan. Its largest profile's
leading future margin is bounded below by a^2-epsilon sigma^2=1/16 for a=1/2,sigma=1/2,
epsilon=3/4; the extremal leading lapse ratio is2 at s=0. This checks only
the plan's exact symbolic bounds and proposed profiles. Finite-annulus/core
timelike estimates, weighted outer remainder bounds, consumed jets, source
identity, arithmetic robustness and native/boundary diagnostics remain held.

## Conditional diagnostic-ghost continuation

No PDE is evolved outside the compact domain by this note. If a later analytic
diagnostic requires ghost samples across Omega=0, the preceding compact
variables can provide a continuation for a two-sided rapidly decaying seed,
such as the separately proposed Gaussian. Continue signed z=Omega/r, use

    H=1/z+C_H+a^2 z/[sqrt(1+a^2 z^2)+1],
    u=s+z c_ret, v=u+2/z,

and the same factored profiles. Gaussian advanced tails are flat as z tends
to zero from either side; their apparently singular powers therefore have
smooth vanishing limits. The resulting compact alpha_bar=r/sqrt(Delta)
continues with the positive branch. Naively computing Omega/sqrt(D) would
instead acquire sign(Omega), because sqrt(D)=|z|sqrt(Delta). The positive
compact continuation is an analytic diagnostic definition beyond the physical
domain, not a claim about positive physical future lapse at negative Omega.

This signed height is the continuation of the future outer branch; at negative
signed R it is not the original positive-radius sqrt(R^2+a^2)+C_H expression.
The Cartesian X=(1/z)n and radial formula for F are continued consistently;
the scalar G and C are even in signed R, so this is algebraically compatible
with the quadrupolar factor. A generic f with only future advanced decay does
not automatically permit this two-sided continuation. A later ghost recipe
must state its branch, profile decay, normals, support and arithmetic gates.
No implementation, ghost interpolation/closure or boundary admission is made.
