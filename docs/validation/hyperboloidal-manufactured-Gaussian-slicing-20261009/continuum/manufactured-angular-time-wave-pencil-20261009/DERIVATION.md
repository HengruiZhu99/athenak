# Manufactured angular pure-time physical wave map

This is a pencil derivation and source-context inventory only. It contains no
numerical/CAS evaluation, implementation, compiler, kernel query, inverse solve
or evolution. The seed f and amplitude epsilon remain symbolic. This is an
alternative manufactured diagnostic for the full physical-reference wave-map
gauge, not the actual native lapse/shift pulse and not the coupled inner-gauge
candidate. The same analytic Minkowski hyperboloidal reference is retained.

## Exact scalar and regular origin

Use physical inertial coordinates (T,X), R=|X|, and distinguish Cartesian
components X_1 and X_2. Define

    G(T,R) = [f(T-R)-f(T+R)]/R,
    F(T,X) = partial_1 partial_2 G = X_1 X_2 C(T,R),
    Y^0 = T+epsilon F(T,X),  Y^I = X^I.

For R>0 the exact radial factor is

    C = [f''(T-R)-f''(T+R)]/R^3
        +3[f'(T-R)+f'(T+R)]/R^4
        +3[f(T-R)-f(T+R)]/R^5.

The numerator R G is a one-dimensional wave. The radial three-dimensional
wave operator and Cartesian differentiation therefore give Box_M G=Box_M F=0.
The negative sign between the advanced and retarded terms makes G and F smooth
at the origin. No radius floor or finite-difference differentiation is needed.
Two exact integral identities supply independent regular representations:

    G = -(1/(2 pi)) integral_S2 f'(T+n.X) dOmega_n,
    F = -(1/(2 pi)) integral_S2 n_1 n_2 f'''(T+n.X) dOmega_n,
    C = -(1/8) integral_-1^1 (1-z^2)^2 f^(5)(T+R z) dz.

The last identity follows by two integrations by parts, with vanishing endpoint
weights. Its differentiated form is

    partial_T^m partial_R^n C
       = -(1/8) integral_-1^1 (1-z^2)^2 z^n f^(5+m+n)(T+R z) dz.

Thus smooth f supplies all origin/collar jets; derivatives through total order
three use f through order eight in this representation. The origin series is

    C = -2 f^(5)(T)/15 - R^2 f^(7)(T)/105
        -R^4 f^(9)(T)/3780 + ...,
    F = -2 X_1 X_2 f^(5)(T)/15 + O(R^4).

In particular F, F_T, F_i, F_TT and F_Ti vanish at X=0, while

    F_ij = -(2/15) f^(5)(T) E_ij,
    F_Tij = -(2/15) f^(6)(T) E_ij,
    E_ij = delta_i1 delta_j2 + delta_i2 delta_j1.

Pure third spatial derivatives vanish there. These statements retain a genuine
quadrupolar Hessian even though the scalar and its first derivatives vanish.
For R>0, write P=X_1 X_2, n_i=X_i/R. Complete spatial jets follow by Leibniz
from P_i=delta_i1 X_2+delta_i2 X_1, P_ij=E_ij, P_ijk=0 and

    C_i = C_R n_i,
    C_ij = C_RR n_i n_j + (C_R/R)(delta_ij-n_i n_j),
    C_ijk = C_RRR n_i n_j n_k
      +(C_RR/R-C_R/R^2)
       (delta_ij n_k+delta_ik n_j+delta_jk n_i-3 n_i n_j n_k).

Mixed time jets replace C by the corresponding T derivative. Their exact
radial wave identity is C_TT=C_RR+6 C_R/R, including its regular origin limit.
F has pure scalar l=2 angular dependence in these inertial coordinates. This is
coordinate-wave freedom, not gravitational radiative data; nonlinear inverse
and ADM fields need not have a finite angular harmonic expansion.

## One-dimensional inverse versus timelike native slices

The forward Jacobian determinant is

    J = 1+epsilon F_T.

At fixed X, the physical target time Y^0=T+epsilon F is a one-dimensional map.
If J is strictly positive globally and F is bounded, it is strictly increasing
and onto, hence has a unique global inverse at that X. In particular,
integral_S2 |n_1 n_2| dOmega_n=8/3 gives the sufficient bounds

    |F| <= 4 ||f'''||_infinity/(3 pi),
    |F_T|, |grad_X F| <= 4 ||f''''||_infinity/(3 pi).

Consequently |epsilon| 4 ||f''''||_infinity/(3 pi)<1 is a sufficient uniform
J>0 condition. A second available bound, for any fixed R0>0 and
M_j=sup_T |f^(j)(T)|, is

    sup |F_T| <= max(R0^2 M_6/15,
                    M_3/R0+3 M_2/R0^2+3 M_1/R0^3).

The first term follows from the regular C integral and |X_1 X_2|<=R^2/2;
the second bounds the full advanced/retarded formula. These are sufficient
conditions, not a chosen amplitude or a necessary inverse criterion.

Let H(R) be the complete fixed Minkowski reference height, including its Cauchy
core and layered transition. Native time is

    t = Y^0-H(|Y^I|) = T+epsilon F(T,X)-H(R).

It is not physical target time Y^0, nor reference time T-H(R). Put

    w_i = H_i-epsilon F_i,
    D = J^2-|w|^2
      = 1-H_R^2 +2 epsilon(F_T+H_R F_R)
        +epsilon^2(F_T^2-|grad F|^2).

The native slices have future timelike normal precisely when J>0 and D>0.
A global monotone Y^0 inverse does not imply D>0. On a finite spatial ball,
let v_max=sup |H_R|<1 and use bounds M_T,M_X for |F_T|,|grad F| there.
The stronger sufficient condition

    |epsilon|(M_T+M_X) < 1-v_max

implies J>|w| and D>0. It becomes weak near infinity, where the reference
timelike margin tends to zero. It is not a uniform scri estimate.

## Future outgoing end and its missing global estimate

In the exact outer CMC branch,

    H(R)=sqrt(R^2+a^2)+C_H
        =R+C_H+a^2/(2R)+O(R^-3).

Assume the advanced f(T+R) and its needed derivatives decay sufficiently rapidly
on the future outgoing end with u=T-R bounded. Then

    F = n_1 n_2 [f''(u)/R+3 f'(u)/R^2+3 f(u)/R^3]
        + decaying advanced terms,
    F_T+F_R = n_1 n_2 [-f''(u)/R^2-6 f'(u)/R^3-9 f(u)/R^4].

The leading R^-2 terms cancel in F_T^2-|grad F|^2. At a fixed native time t,

    u_infinity=t+C_H,
    u=u_infinity+[a^2/2-epsilon n_1 n_2 f''(u_infinity)]/R+O(R^-2),
    R^2 D -> a^2-2 epsilon n_1 n_2 f''(u_infinity).

For the full sphere the minimum leading margin is
a^2-|epsilon f''(u_infinity)|. Therefore
|epsilon| sup |f''|<a^2 is a uniform positive leading future-end criterion.
If Omega R->S, the conformal lapse tends to
S/sqrt(a^2-2 epsilon n_1 n_2 f''(u_infinity)).

This does not prove D>0 at finite R, for all native times, or throughout all
Minkowski. In particular, near past incoming null infinity a decaying seed can
still have F_T+H_R F_R=O(R^-1), which exceeds the reference O(R^-2) margin.
A later admissibility proof must specify a bounded future native-time interval,
bound its weighted outer remainder, and independently cover the finite annulus
and the core. Positivity of the displayed leading coefficient alone is
insufficient. If epsilon is dimensionless and Y^0 has length, f has dimension
length^4; the leading combination a^2-2 epsilon n_1 n_2 f'' is dimensionally
consistent.

For a possible later seed f(T)=A exp[-(T-t0)^2/(2 sigma^2)], the exact identity

    G = 2 A exp[-((T-t0)^2+R^2)/(2 sigma^2)]
            sinh((T-t0) R/sigma^2)/R

has a positive sinh sign and admits an analytic sinhc origin representation.
This note chooses neither that profile nor an amplitude.

## Exact implicit jets and physical-P ADM conversion

Use Q^i=Y^i=X^i and solve T+epsilon F(T,Q)=t+H(|Q|). Let a,b,c range over
(t,Q^1,Q^2,Q^3), with H_a,H_ab,H_abc zero whenever a time index is present,
and E_a^A=(T_a,delta_a^i). All F jets below are evaluated at this same implicit
event. Differentiating the exact relation gives

    T_a = (delta_a0+H_a-epsilon F_i delta_a^i)/J,
    T_ab = [H_ab-epsilon F_AB E_a^A E_b^B]/J,
    T_abc = [H_abc-epsilon F_ABC E_a^A E_b^B E_c^C
              -epsilon F_0A(E_a^A T_bc+E_b^A T_ac+E_c^A T_ab)]/J.

These are analytic implicit derivatives, not nested FD. They require the full
reference H through order three. In the transition, with compact r and
L=Omega-r Omega_r, alphahat^2=Omega^2+b^2,

    partial_R=(Omega^2/L) partial_r,
    H_R=b/alphahat,
    H_RR=(Omega^2/L) partial_r(b/alphahat),
    H_RRR=(Omega^2/L) partial_r[(Omega^2/L) partial_r(b/alphahat)].

The exact core H is constant and has vanishing Cartesian derivatives. Generic
LayerPoint second jets cannot stand in for missing third height/embedding jets.

The physical ADM fields in (t,Q), in the convention K_ij=-Lie_n gamma_ij/2,
are

    gamma_Q,ij=delta_ij-w_i w_j/J^2,
    alpha_physical=1/sqrt(D),  beta_Q^i=-w_i/D,
    K_Q,ij=-T_ij/sqrt(1-|grad_Q T|^2)=-J T_ij/sqrt(D).

Under the fixed compactification Q^I=x^I/Omega(x),

    bar_gamma_ij=Omega^2 Q_i^I Q_j^J gamma_Q,IJ,
    alpha=Omega/sqrt(D),
    beta_x^i=(partial Q/partial x)^-1 i_I beta_Q^I,
    chi=(det bar_gamma)^(-1/3),  gtilde=chi bar_gamma,
    Atilde_ij=Omega chi (Kphysical_ij-gamma_physical_ij Kphysical/3),
    P=Kphysical, Theta=0, Lambda=contracted Gamma[gtilde].

Kphysical_ij in the Atilde line has first been transformed to compact spatial
coordinates. Thus Z_i=0 and the physical vacuum ADM constraints hold exactly.
The consumed alpha/beta/chi/gtilde second spatial jets, A/P/Lambda first jets,
and all twenty-two time derivatives require inverse T and compact embedding
jets through order three. At the exact origin J=D=1 and first metric values are
flat, but T_ij=-epsilon F_ij and K_ij=epsilon F_ij need not vanish. The exact
Cauchy core of the reference does not make the manufactured solution's entire
core geometry constant.

Since all four Y^A are physical harmonic scalars, this flat pullback obeys the
full physical-reference RWM condition in the native chart on the Einstein
sector. Its linear active coordinate vector in the Y chart is -F partial_T;
the finite inverse map, not the forward map, determines the pullback sign.
The new coupled inner helper adds independent P/Lambda/relative-chi terms and
does not generally preserve this manufactured solution. No inner blend,
stationary BH, moving-puncture transition, exact-scri closure or native
stability is asserted here.
