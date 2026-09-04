# Puncture-conformal first-order generalized harmonic derivation

## 2026-09-04 correction: FO-GH equivalence is not an off-constraint theorem

The [complete gamma2 audit](pc_gh_gamma2_audit.md) supersedes the principal-symbol
and symmetrizer transfer claims below wherever they rely only on equality on the
reduction manifold. The regular production `rho` row differs from the exact
`gamma1=-1` FO-GH pullback by `-beta^i Ralpha_i/(2w)`, already at `gamma2=0`.
An independent zero-step production-kernel oracle reproduces this discrepancy.
The moving-puncture source substitution also makes the stated forward FO-GH map
independent of `Cperp`; that map cannot establish invertibility for this gauge.
The earlier constrained identities and recorded numerical observations remain
evidence within their stated scope. They do not prove equality of the full
off-constraint principal matrices. No dynamical gamma2 implementation is qualified.

## Status and scope

This is a from-scratch derivation from

\[
C^\mu=H^\mu+\Gamma^\mu,
\qquad
R_{\mu\nu}-\nabla_{(\mu}C_{\nu)}
=-\kappa\left[n_{(\mu}C_{\nu)}-
\frac12g_{\mu\nu}n^\rho C_\rho\right].
\]

No FO-GH/Ref-GH source, generated equation set, reference geometry, reference frame,
controller, or old gauge-subtraction implementation is used. The target equations in
the controlling specification are regression targets, not derivation premises.

Classifications have the following meanings:

- `PROVED`: an algebraic or differential identity is established under the stated
  nonsingularity assumptions.
- `PROVED ON r>0`: the proof additionally uses positive `A` and/or `chi`, which is the
  punctured-domain theorem.
- `CONDITIONAL`: an identity is proved only when named constraints or gauge hypotheses
  hold.
- `NOT ESTABLISHED`: the derivation or independent covariant comparison is incomplete.
- `FAILED`: a proposed identity has a nonzero residual or violates an audit condition.

No production equation may be implemented while classified `NOT ESTABLISHED`, `FAILED`,
or subject to an unmet `CONDITIONAL` hypothesis.

## 2026-09-02 puncture-regular 55-field representation (superseding production map)

The older `chi,A,X,Y,pi,Lambda` representation derived below is retained as the
independent positive-field oracle.  Production now applies the invertible point map
on `w>0,rho>0`

\[
\begin{gathered}
w=\sqrt\chi,\quad \rho=\alpha/w,\quad p_i=\partial_iw,\quad
L_i=2\partial_i\alpha,\\
C_\perp=\pi+K,\qquad Z^i=\widetilde\Gamma^i(Q)-\widetilde\Lambda^i,
\end{gathered}
\]

so the 55 variables are

\[
\{w,\widetilde\gamma_{ij},K,\widetilde A_{ij},Z^i,C_\perp,
\rho,\beta^i,p_i,Q_{kij},L_i,B_i{}^j\}.
\]

The inverse identities used by the symbolic equivalence proof are

\[
\chi=w^2,\quad \alpha=\rho w,\quad A=\rho^2w^2,\quad X_i=2wp_i,
\quad Y_i=\alpha L_i,\quad \pi=C_\perp-K,
\quad\widetilde\Lambda^i=\widetilde\Gamma^i(Q)-Z^i.
\]

No `partial_i rho` variable is evolved.  On the stationary 1+log trumpet its fitted
power is `-0.908702735`, whereas the evolved powers are
`w~r`, `rho~r^0.091297265`, `p~r^0`, and `L~r^0.091297265`.

For the direct moving-puncture gauge, exact differentiation gives (`PROVED`)

\[
\begin{aligned}
D_0w&={w\over3}(\alpha K-B),\\
D_0\rho&=\rho\left[-2K-{\alpha K-B\over3}\right],\\
\partial_tp_i&=\beta^k\partial_kp_i+B_i{}^kp_k
 +{1\over3}\{p_i(\alpha K-B)
 +w[(L_i/2)K+\alpha\partial_iK-\partial_iB]\},\\
\partial_tL_i&=\beta^k\partial_kL_i+B_i{}^kL_k
-2KL_i-4\alpha\partial_iK.
\end{aligned}
\]

The denominator-free reduction constraints are

\[
R_i^w=p_i-\partial_iw,\qquad
R_i^\alpha=L_i-2(w\partial_i\rho+\rho p_i),
\]

along with `R^Q`, `R^B`, and the curls of `p,Q,L,B`.  Substitution into the old
positive-field expressions yields

\[
\begin{aligned}
H={}&{2\over3}K^2-\widetilde A_{ij}\widetilde A^{ij}
 +w^2{\cal R}+4w\widetilde D^ip_i-6p_ip^i,\\
E_{ij}={}&{w^2\over2}\widetilde D_iL_j,\\
m_i={}&\widetilde D_j\widetilde A^j{}_i-{2\over3}\partial_iK,\\
\alpha\widetilde M_i={}&\rho w m_i-3\rho\widetilde A^j{}_ip_j,\\
S_{ij}={}&\rho w^3{\cal R}_{ij}+\rho w^2\widetilde D_ip_j
-{w^2\over2}\widetilde D_iL_j-{w\over2}(L_ip_j+L_jp_i),\\
T_{ij}={}&-w(Z_ip_j+Z_jp_i)-{w^2\over2}Z^kQ_{kij}.
\end{aligned}
\]

The corresponding primary equations implemented by the production kernel are

\[
\begin{aligned}
D_0\widetilde\gamma_{ij}={}&-2\alpha\widetilde A_{ij}
 +2\widetilde\gamma_{k(i}B_{j)}{}^k-{2\over3}\widetilde\gamma_{ij}B,\\
D_0K={}&\alpha\widetilde A_{ij}\widetilde A^{ij}+{\alpha\over3}K^2
-E^i{}_i+{w\over2}p^iL_i+\alpha(H-KC_\perp)
+\alpha w p_iZ^i-{3\over2}\kappa\alpha C_\perp,\\
D_0\widetilde A_{ij}={}&S_{ij}^{TF}+2\widetilde A_{k(i}B_{j)}{}^k
-{2\over3}\widetilde A_{ij}B-2\alpha\widetilde A_{ik}\widetilde A^k{}_j
+\alpha(K-C_\perp)\widetilde A_{ij}+\alpha T_{ij}^{TF},\\
D_0C_\perp={}&\alpha(H-KC_\perp)
+w^2(\rho p_i+L_i/2)Z^i-2\kappa\alpha C_\perp,\\
D_0Z^i={}&-2\widetilde\gamma^{ij}
(\rho wm_j-3\rho\widetilde A^k{}_jp_k)
-\rho w\widetilde D^iC_\perp+{C_\perp\over2}\widetilde\gamma^{ij}L_j\\
&-Z^kB_k{}^i+{2\over3}Z^iB
-\left({2\over3}\rho wK+\kappa\rho w\right)Z^i.
\end{aligned}
\]

For both moving-puncture variants,

\[
D_0\beta^i=\widetilde\Gamma^i(Q)-Z^i-\eta\beta^i+S(z)M^i,
\quad z=\rho w^3,
\quad M^i=\rho w^3\widetilde\gamma^{ij}(\rho p_j-L_j/2).
\]

`S=0` is the direct gauge.  The switched gauge uses the existing smoothstep.  The
standard-order `B_i{}^j` equation is the complete spatial derivative of this
expression.  In particular,

\[
\partial_i z=w^2(L_i/2+2\rho p_i),
\]

and `partial_i(SM^j)=S partial_i M^j+S'(partial_i z)M^j`; after using
`R_i^alpha=0`, `z partial_i M^j` is represented without a division by `w`.
The `Q` equation is likewise the exact standard-order derivative of the
`gtilde` configuration equation.

`verify_puncture_regular_55.py` proves these substitutions, the `H/E/S/T/m`
identities, both shift derivatives including the `S'` product, and the wormhole and
trumpet exponents.  The regular tangent-map determinant is
`-32 rho^4 w^9`; hence for positive `w,rho` the principal matrix is similar to the old
one.  Its exact characteristic polynomial is

\[
-{1\over3}v^{30}(v-1)^2(v+1)^2(3v^2-4)
(2\rho w^3-v^2)(\rho^2w^4-v^2)^6.
\]

The known direct-gauge defective surfaces become
`rho^2 w^4=4/3`, `rho w=2`, and `rho w^3=2/3`; there is no additional finite,
positive-`rho` degeneracy.  These results are `PROVED ON w>0,rho>0`.

## Conventions

Use signature \((-+++ )\),

\[
n_\mu=(-\alpha,0,0,0),\qquad
n^\mu=\alpha^{-1}(1,-\beta^i),
\]

and

\[
K_{ij}=-\frac12\mathcal L_n\gamma_{ij}.
\]

Write

\[
D_0=\partial_t-\beta^k\partial_k,
\qquad B_i{}^j=\partial_i\beta^j,
\qquad B=B_k{}^k.
\]

Parentheses carry weight one half. Spatial indices on conformal objects are raised and
lowered with \(\tilde\gamma_{ij}\). Define the source projections

\[
H_\perp=n_\mu H^\mu,
\qquad
H_\parallel^i=H^i+\beta^iH^0,
\]

and their regularized forms

\[
h_\perp=\alpha H_\perp,
\qquad h^i=A H_\parallel^i.
\]

The analogous GH-constraint projections are

\[
C_\perp=n_\mu C^\mu,
\qquad C_\parallel^i=C^i+\beta^iC^0.
\]

These signs are fixed below by direct evaluation of the contracted Christoffel symbol.

## Contracted Christoffels and gauge variables

From

\[
\Gamma^\mu=-\frac1{\sqrt{-g}}\partial_\nu
\left(\sqrt{-g}\,g^{\mu\nu}\right),
\qquad \sqrt{-g}=\alpha\sqrt\gamma,
\]

the time component gives

\[
n_\mu\Gamma^\mu
=K+\frac{D_0\alpha}{\alpha^2}.
\]

Define

\[
\pi=\frac{D_0\alpha}{\alpha^2}+H_\perp.
\]

Then, without using an evolution equation,

\[
C_\perp=H_\perp+n_\mu\Gamma^\mu=\pi+K.
\]

This establishes the temporal GH constraint and sign convention (`PROVED`). It also
implies

\[
D_0\alpha=\alpha^2\pi-\alpha h_\perp,
\qquad
D_0A=2A(\alpha\pi-h_\perp),
\]

where \(A=\alpha^2\) (`PROVED ON r>0`).

The spatial coordinate projection satisfies

\[
\Gamma^i+\beta^i\Gamma^0
= {}^{(3)}\Gamma^i-D^i\ln\alpha-\frac{D_0\beta^i}{\alpha^2}.
\]

For

\[
\gamma_{ij}=\chi^{-1}\tilde\gamma_{ij},
\qquad \det\tilde\gamma=1,
\]

direct use of the divergence form of the contracted three-Christoffel gives

\[
{}^{(3)}\Gamma^i
=\chi\tilde\Gamma^i+\frac12\tilde\gamma^{ij}X_j,
\qquad X_i=\partial_i\chi.
\]

Since \(D^i\ln\alpha=\chi\tilde\gamma^{ij}Y_j/(2A)\), define

\[
Z^i=\tilde\Gamma^i-\tilde\Lambda^i,
\qquad C_\parallel^i=\chi Z^i.
\]

Solving the spatial constraint definition for the shift velocity yields

\[
D_0\beta^i
=h^i+A\chi\tilde\Lambda^i
+\frac12\tilde\gamma^{ij}(AX_j-\chi Y_j).
\]

Thus the shift configuration equation and the precise spatial GH-constraint scaling are
`PROVED ON r>0`.

## Conformal configuration equations

The ADM kinematic identity in coordinate components is

\[
D_0\gamma_{ij}
=-2\alpha K_{ij}+\gamma_{ik}B_j{}^k+\gamma_{jk}B_i{}^k.
\]

Its determinant gives

\[
D_0\ln\sqrt\gamma=-\alpha K+B.
\]

With \(\chi=(\det\gamma)^{-1/3}\),

\[
D_0\chi=\frac23\chi(\alpha K-B).
\]

Decompose

\[
K_{ij}=\chi^{-1}\left(\tilde A_{ij}
+\frac13\tilde\gamma_{ij}K\right).
\]

Differentiating \(\tilde\gamma_{ij}=\chi\gamma_{ij}\) then gives

\[
D_0\tilde\gamma_{ij}
=-2\alpha\tilde A_{ij}
+2\tilde\gamma_{k(i}B_{j)}{}^k
-\frac23\tilde\gamma_{ij}B.
\]

The chi and conformal-metric configuration equations are `PROVED`.

## First-order conformal geometry

Define

\[
Q_{kij}=\partial_k\tilde\gamma_{ij},
\qquad
\tilde\Gamma^i{}_{jk}
=\frac12\tilde\gamma^{i\ell}
(Q_{j\ell k}+Q_{k\ell j}-Q_{\ell jk}),
\]

and \(\tilde\Gamma^i=\tilde\gamma^{jk}\tilde\Gamma^i{}_{jk}\).
For an exactly unimodular metric and a consistent Q field, rearranging the coordinate
definition of the Ricci tensor gives

\[
\begin{aligned}
\tilde R_{ij}={}&
-\frac12\tilde\gamma^{k\ell}\partial_kQ_{\ell ij}
+\tilde\gamma^{k\ell}\left[
\tilde\Gamma^m{}_{k\ell}\tilde\Gamma_{(ij)m}
+2\tilde\Gamma^m{}_{k(i}\tilde\Gamma_{j)m\ell}
+\tilde\Gamma^m{}_{ik}\tilde\Gamma_{mj\ell}\right]\\
&+\tilde\gamma_{k(i}\partial_{j)}\tilde\Gamma^k,
\end{aligned}
\]

where \(\tilde\Gamma_{ijk}=\tilde\gamma_{i\ell}
\tilde\Gamma^\ell{}_{jk}\) lowers the first index. Replacing the final contracted
Christoffel by the independent \(\tilde\Lambda^i\) defines \({\cal R}_{ij}\). Therefore

\[
{\cal R}_{ij}=\tilde R_{ij}
-\tilde\gamma_{k(i}\partial_{j)}Z^k.
\]

The rearrangement is `PROVED` under \(\det\tilde\gamma=1\),
\(Q=\partial\tilde\gamma\), and nonsingular \(\tilde\gamma\). The exact component
regression in `verify_conformal_ricci.py` independently checks a non-diagonal,
coordinate-dependent unimodular metric. Equality \({\cal R}_{ij}=\tilde R_{ij}\) is
`CONDITIONAL` on \(Z^i=0\) (or the corresponding symmetrized derivative vanishing).

## Puncture-regular conformal identities

For \(r>0\), assume \(A>0\), \(\chi>0\), and define

\[
W_i=\frac{X_i}{\sqrt\chi},\qquad
L_i=\frac{Y_i}{\alpha},\qquad
r_-=\frac\chi\alpha,\qquad
r_+=\frac\alpha{\sqrt\chi}.
\]

No conclusion here establishes a uniformly conditioned extension through
\(A=\chi=0\).

### Lapse Hessian

Because \(Y_i=\partial_iA=2\alpha\partial_i\alpha\), direct differentiation gives

\[
\chi\tilde D_i\tilde D_j\alpha
=\frac{r_-}{2}\left({\cal Y}_{ij}-\frac12L_iL_j\right)
\equiv {\cal A}_{ij},
\]

where \({\cal Y}_{ij}=\partial_iY_j-
\tilde\Gamma^k{}_{ij}Y_k\). This is `PROVED ON r>0` by exact simplification.

The physical/conformal connection difference is

\[
{}^{(3)}\Gamma^k{}_{ij}-\tilde\Gamma^k{}_{ij}
=-\frac1{2\chi}\left(\delta^k_iX_j+\delta^k_jX_i
-\tilde\gamma_{ij}X^k\right).
\]

Consequently,

\[
\chi D_iD_j\alpha
={\cal A}_{ij}+\frac14(L_iX_j+L_jX_i)
-\frac14\tilde\gamma_{ij}L_kX^k.
\]

This is `PROVED ON r>0`.

### Scalar curvature, Hamiltonian, and momentum

The connection difference gives

\[
{}^{(3)}R
=\chi\tilde R+2\tilde D^iX_i-\frac52W_iW^i.
\]

Using
\(K_{ij}K^{ij}=\tilde A_{ij}\tilde A^{ij}+K^2/3\), the ADM Hamiltonian is

\[
{\cal H}_{ADM}
=\frac23K^2-\tilde A_{ij}\tilde A^{ij}
+\chi\tilde R+2\tilde D^iX_i-\frac52W_iW^i.
\]

Replacing \(\tilde R\) by the reduced GH scalar \({\cal R}\) defines the requested
\({\cal H}\). The regularization is `PROVED ON r>0`; equality to the ADM Hamiltonian is
`CONDITIONAL` on the spatial GH constraint.

For the momentum constraint,

\[
D_j\tilde A^j{}_i
=\tilde D_j\tilde A^j{}_i
-\frac{3}{2\chi}\tilde A^j{}_iX_j.
\]

Multiplication by \(\sqrt\chi\) gives

\[
\widehat{\cal M}_i
=\sqrt\chi\left(\tilde D_j\tilde A^j{}_i
-\frac23\partial_iK\right)
-\frac32\tilde A^j{}_iW_j.
\]

This is `PROVED ON r>0`.

### Trace-free curvature/lapse tensor

The conformal Ricci transformation and lapse-Hessian identity imply that the trace-free
part of \(\alpha\chi R_{ij}-\chi D_iD_j\alpha\) equals the trace-free part of

\[
\begin{aligned}
{\cal S}_{ij}={}&
\alpha\chi\tilde R_{ij}
+\frac\alpha2{\cal X}_{ij}
-\frac\alpha4W_iW_j
-{\cal A}_{ij}
-\frac14(L_iX_j+L_jX_i),
\end{aligned}
\]

where \({\cal X}_{ij}=\partial_iX_j-
\tilde\Gamma^k{}_{ij}X_k\). Terms omitted from this representative are proportional to
\(\tilde\gamma_{ij}\) and vanish under TF projection. Replacing \(\tilde R_{ij}\) by
\({\cal R}_{ij}\) gives the reduced-GH tensor in the target system. The regular TF
identity is `PROVED ON r>0`; equality to the physical-Ricci expression is `CONDITIONAL`
on the spatial GH constraint.

## Covariant reduced-equation projections

The spatial projection of the starting covariant equation uses

\[
\gamma_i{}^\mu\gamma_j{}^\nu\nabla_\mu C_\nu
=D_iC_j+K_{ij}C_\perp,
\qquad C_i=\tilde\gamma_{ij}Z^j,
\]

and gives

\[
{}^{(4)}R_{ij}
=D_{(i}C_{j)}+K_{ij}C_\perp
+\frac\kappa2\gamma_{ij}C_\perp.
\]

The normal-normal projection is

\[
{}^{(4)}R_{nn}
-n(C_\perp)+C_iD^i\ln\alpha
=\frac\kappa2C_\perp.
\]

These signs follow directly from \(K_{ij}=-\gamma_i{}^\mu\gamma_j{}^\nu
\nabla_\mu n_\nu\) and \(C_\perp=n^\mu C_\mu\).

### pi

Using

\[
{}^{(4)}R_{nn}
=\frac1\alpha(D_0K+D^iD_i\alpha)-K_{ij}K^{ij}
\]

and \(D_0C_\perp=D_0\pi+D_0K\), the normal projection yields

\[
D_0\pi
=D^iD_i\alpha-\alpha K_{ij}K^{ij}
+\alpha C_iD^i\ln\alpha
-\frac12\kappa\alpha C_\perp.
\]

Substitution of the regular identities gives

\[
D_0\pi
=-\alpha\tilde A^2-\frac13\alpha K^2
+{\cal A}-\frac14X_iL^i
+\frac\chi2Z^iL_i
-\frac12\kappa\alpha C_\perp.
\]

The supplied pi target is therefore `PROVED ON r>0`.

### K and a failed supplied regression target

The trace of the spatial projection gives

\[
D_0K
=-D^iD_i\alpha+\alpha({}^{(3)}R+K^2)
-\alpha D_iC^i-\alpha KC_\perp
-\frac32\kappa\alpha C_\perp.
\]

Here

\[
D_iC^i=\chi\tilde D_iZ^i-\frac12X_iZ^i
\]

and the reduced Hamiltonian defined with \({\cal R}\) obeys

\[
{\cal H}={\cal H}_{ADM}-\chi\tilde D_iZ^i.
\]

The divergence in \({\cal H}_{ADM}\) therefore cancels the divergence in
\(D_iC^i\). The covariant equation becomes

\[
\boxed{
\begin{aligned}
D_0K={}&
\alpha\tilde A^2+\frac13\alpha K^2
-{\cal A}+\frac14X_iL^i\\
&+\alpha\left[{\cal H}-KC_\perp+\frac12X_iZ^i\right]
-\frac32\kappa\alpha C_\perp.
\end{aligned}}
\]

The supplied regression target additionally contains
\(-\alpha\chi\tilde D_iZ^i\). It therefore counts the spatial GH divergence twice.
For \(\alpha=\chi=1\), flat conformal metric, vanishing curvature/extrinsic fields,
\(C_\perp=0\), and nonzero \(\partial_iZ^i\), the covariant equation gives
\(D_0K=-\partial_iZ^i\) while the supplied target gives
\(-2\partial_iZ^i\). This exact counterexample is encoded in
`verify_primary_projections.py`.

Classification: supplied K target `FAILED`; boxed corrected K equation `PROVED ON r>0`.

### Atilde and a failed supplied nonlinear Z term

The trace-free spatial projection, conformal rescaling, and ADM kinematics give the
usual shift/extrinsic pieces plus

\[
\left[\alpha\chi({}^{(3)}R_{ij}-D_{(i}C_{j)})
-\chi D_iD_j\alpha\right]^{TF}.
\]

Using the Brown operator exactly as defined above, all derivatives of \(Z^i\) cancel.
The remaining regular algebraic term is

\[
\left[-Z_{(i}X_{j)}
-\frac\chi2 Z^kQ_{kij}\right]^{TF}.
\]

Here and below \(Z_i=\tilde\gamma_{ij}Z^j\); the first term is not a contraction
with the coordinate component array \(Z^i\).

Thus the covariant result is

\[
\boxed{
\begin{aligned}
D_0\tilde A_{ij}={}&
[ {\cal S}_{ij}]^{TF}
+2\tilde A_{k(i}B_{j)}{}^k
-\frac23\tilde A_{ij}B\\
&-2\alpha\tilde A_{ik}\tilde A^k{}_j
+\alpha K\tilde A_{ij}
-\alpha C_\perp\tilde A_{ij}\\
&+\alpha\left[-Z_{(i}X_{j)}
-\frac\chi2Z^kQ_{kij}\right]^{TF}.
\end{aligned}}
\]

The supplied target instead uses \(-\chi Z_k\tilde\Gamma^k{}_{ij}\). In an
orthonormal conformal frame its TF difference from the covariant projection is a
nonzero polynomial in arbitrary trace-free \(Q_{kij}\) and \(Z^i\); the exact residual
is checked in `verify_primary_projections.py`.

Classification: supplied Atilde nonlinear Z term `FAILED`; boxed corrected Atilde
equation `PROVED ON r>0` under the algebraic and reduction hypotheses of the Brown
operator. The non-diagonal exact-rational point jet in
`verify_4d_component_oracle.py` independently verifies this index lowering and the
complete corrected Atilde equation directly against the covariant four-tensor equation.

### Lambda and a missing lapse-acceleration term

The mixed projection uses the Codazzi identity

\[
{}^{(4)}R_{ni}=D_iK-D_jK^j{}_i.
\]

Expanding the two projected derivatives of \(C_\mu\) gives

\[
\mathcal L_n C_i
=-2{\cal M}_i-2K_i{}^jC_j-D_iC_\perp
+C_\perp D_i\ln\alpha-\kappa C_i,
\]

where \({\cal M}_i=D_jK^j{}_i-D_iK\) is the physical momentum constraint. Since

\[
\alpha\mathcal L_nC_i=D_0C_i-B_i{}^jC_j,
\qquad C_i=\tilde\gamma_{ij}Z^j,
\]

the constraint propagation equation is

\[
\begin{aligned}
D_0Z^i={}&-2\alpha\tilde{\cal M}^i
-\alpha\tilde D^iC_\perp
+\alpha C_\perp\tilde D^i\ln\alpha\\
&-Z^kB_k{}^i+\frac23Z^iB
-\left(\frac23\alpha K+\kappa\alpha\right)Z^i,
\end{aligned}
\]

where \(\tilde{\cal M}^i\) denotes the momentum residual with its index raised by the
conformal metric.

Independent differentiation of \(\tilde\Gamma^i\) from the conformal-metric
configuration equation gives the same first line as the supplied Lambda target, plus
\(-2\alpha\tilde{\cal M}^i\), with \(\tilde\Gamma^i\) in its shift-algebra terms.
Using \(Z^i=\tilde\Gamma^i-\tilde\Lambda^i\) and the mixed-projection propagation
equation therefore gives

\[
\boxed{
\begin{aligned}
D_0\tilde\Lambda^i={}&
\tilde\gamma^{k\ell}\partial_kB_\ell{}^i
+\frac13\tilde\gamma^{ij}\partial_jB
-\tilde\Lambda^kB_k{}^i
+\frac23\tilde\Lambda^iB\\
&-\tilde A^{ik}L_k
+2\alpha\tilde A^{k\ell}\tilde\Gamma^i{}_{k\ell}
-3r_+\tilde A^{ik}W_k
-\frac43\alpha\tilde D^iK\\
&+\alpha\tilde D^iC_\perp
-\frac\chi2 C_\perp L^i
+\left(\frac23\alpha K+\kappa\alpha\right)Z^i.
\end{aligned}}
\]

The supplied target omits

\[
-\alpha C_\perp D^i\ln\alpha
=-\frac\chi2C_\perp L^i.
\]

That term is nonzero for simultaneous lapse gradient and temporal GH-constraint
violation. Its exact mixed-projection residual is recorded by
`verify_primary_projections.py`.

Classification: supplied Lambda target `FAILED`; boxed corrected Lambda equation
`PROVED ON r>0`. The exact 4D point-jet oracle independently verifies the complete
corrected K, Atilde, pi, and Lambda equations while satisfying all ten components of
the starting covariant reduced equation.

## Standard first-order gradient equations

For each configuration field \(q\in\{\chi,A,\beta^j,
\tilde\gamma_{ab}\}\), write

\[
\partial_tq=\beta^kG_k^{(q)}+F^{(q)}.
\]

Direct differentiation gives

\[
\partial_tG_i^{(q)}
=\beta^k\partial_iG_k^{(q)}+B_i{}^kG_k^{(q)}+\partial_iF^{(q)}.
\]

The production standard ordering is

\[
\boxed{
\partial_tG_i^{(q)}
=\beta^k\partial_kG_i^{(q)}+B_i{}^kG_k^{(q)}+\partial_i^{(1)}F^{(q)}}.
\]

The compatible and standard forms differ by

\[
\beta^k(\partial_iG_k^{(q)}-\partial_kG_i^{(q)})
=\beta^kK^{(q)}_{ik}.
\]

Thus standard ordering corresponds to the stated \(\gamma_1=-1\) baseline relative to
direct differentiation. No \(\gamma_2\) term is introduced.

The required algebraic source gradients are as follows. First,

\[
\begin{aligned}
\partial_iF^{(\chi)}
=\frac23\{&X_i(\alpha K-B)\\
&+\chi[\tfrac12L_iK+\alpha\partial_iK-\partial_iB]\},
\end{aligned}
\]

and

\[
\begin{aligned}
\partial_iF^{(A)}
=2Y_i(\alpha\pi-h_\perp)
+2A[\tfrac12L_i\pi+\alpha\partial_i\pi-\partial_i h_\perp].
\end{aligned}
\]

For the shift, define \(V_\ell=AX_\ell-\chi Y_\ell\). Since

\[
\partial_i\tilde\gamma^{j\ell}
=-\tilde\gamma^{ja}\tilde\gamma^{\ell b}Q_{iab},
\]

one obtains

\[
\begin{aligned}
\partial_iF^{(\beta^j)}={}&
\partial_i h^j+(Y_i\chi+AX_i)\tilde\Lambda^j
+A\chi\partial_i\tilde\Lambda^j\\
&-\frac12\tilde\gamma^{ja}\tilde\gamma^{\ell b}Q_{iab}V_\ell\\
&+\frac12\tilde\gamma^{j\ell}
[Y_iX_\ell+A\partial_iX_\ell-X_iY_\ell-\chi\partial_iY_\ell].
\end{aligned}
\]

Finally,

\[
\begin{aligned}
\partial_\ell F^{(\tilde\gamma_{ij})}={}&
-L_\ell\tilde A_{ij}-2\alpha\partial_\ell\tilde A_{ij}\\
&+Q_{\ell ki}B_j{}^k+\tilde\gamma_{ki}\partial_\ell B_j{}^k\\
&+Q_{\ell kj}B_i{}^k+\tilde\gamma_{kj}\partial_\ell B_i{}^k\\
&-\frac23[Q_{\ell ij}B+\tilde\gamma_{ij}\partial_\ell B].
\end{aligned}
\]

Every derivative on the right is a first derivative of a stored first-order or primary
field. Gauge A must provide \(\partial_i h_\perp\) and \(\partial_i h^j\) by analytic
differentiation of its prescribed-coordinate/metric-only source; those derivatives may
depend on X, Y, Q, and B but not on K, Atilde, Lambda, pi, or derivative-field feedback.

`verify_gradient_rhs.py` evaluates the product rules with an independent exact dual
number implementation, including the inverse-metric derivative and curl-ordering
difference. Classification: standard gradient equations `PROVED ON r>0`, conditional on
a differentiable Gauge A source satisfying the stated dependency restriction.

## Direct moving-puncture gauge and its characteristic defects

To match the default AthenaK `project/bbh` Z4c coordinate conditions on the physical
constraint surface, define the PC-GH gauge `z4c_mp` by

\[
 D_0 A=-4AK,
 \qquad
 D_0\beta^i=\tilde\Lambda^i-\eta\beta^i,
 \qquad D_0=\partial_t-\beta^k\partial_k .
\]

Here \(K\), rather than a GH replacement for the Z4c \(\hat K=K-2\Theta\), is used.
In the regular GH-source representation this is exactly

\[
 h_\perp=\alpha\pi+2K,
 \qquad
 h^i=(1-A\chi)\tilde\Lambda^i
 -\frac12\tilde\gamma^{ij}(AX_j-\chi Y_j)-\eta\beta^i.
\]

For constant \(\eta\), differentiating these equations in the STANDARD ordering gives

\[
 \partial_tY_i=\beta^k\partial_kY_i+B_i{}^kY_k
 -4KY_i-4A\partial_iK,
\]

\[
 \partial_tB_i{}^j=\beta^k\partial_kB_i{}^j+B_i{}^kB_k{}^j
 +\partial_i\tilde\Lambda^j-\eta B_i{}^j.
\]

`verify_z4c_mp_gauge.py` proves the cancellations in both the primary and differentiated
source forms exactly. This is a gauge replacement only: the corrected \(K\),
\(\tilde A_{ij}\), \(\pi\), and \(\tilde\Lambda^i\) Einstein equations are unchanged.

The principal-symbol calculation is performed after pulling the 55-field symbol back
to the 50-dimensional tangent space of the five conformal algebraic constraints. At
any point with positive-definite \(\tilde\gamma_{ij}\), a conformal-orthonormal frame
and a unit wave covector reduce the calculation to the exact matrix in
`analyze_z4c_mp_principal.py`. With \(v\) measured relative to the shift, its
characteristic polynomial is

\[
 -\frac13v^{30}(v-1)^2(v+1)^2(3v^2-4)
 (2\alpha\chi-v^2)(\alpha^2\chi-v^2)^6.
\]

Thus the nonzero speeds are

\[
 v=\pm1\;(2),\quad
 v=\pm\sqrt{4/3}\;(1),\quad
 v=\pm\sqrt{2\alpha\chi}\;(1),\quad
 v=\pm\alpha\sqrt\chi\;(6),
\]

with 30 zero-speed fields. The exact eigenspace ranks identify three defective
surfaces:

| surface | algebraic/geometric multiplicity of the coincident positive-speed root |
|---|---:|
| \(\alpha^2\chi=4/3\) | `7/6` |
| \(\alpha=2\), generically | `7/5` |
| \(\alpha\chi=2/3\) | `2/1` |

The coincidences \(\alpha^2\chi=1\) and \(2\alpha\chi=1\) are semisimple. Therefore
the direct gauge is strongly hyperbolic only away from the three defective surfaces;
it is not a spectral/SAT candidate over a domain crossing one of them. The unit-mass
initial wormhole has \(\alpha\chi=\psi^{-6}\) and crosses the third surface at

\[
 r/M=\frac{1}{2[(3/2)^{1/6}-1]}=7.151725902759133.
\]

For the prescribed modified shift, set \(z=\alpha\chi\) and

\[
 D_0\beta^i=\tilde\Lambda^i-\eta\beta^i+S(z)M^i,
 \qquad
 M^i=\frac12\tilde\gamma^{ij}(AX_j-\chi Y_j).
\]

Production uses the cubic smoothstep

\[
 S(z)=\begin{cases}
 0,&z\le z_0,\\
 3s^2-2s^3,&z_0<z<z_1,\quad s=(z-z_0)/(z_1-z_0),\\
 1,&z\ge z_1,
 \end{cases}
 \qquad (z_0,z_1)=(0.1,0.5).
\]

The STANDARD \(B\) equation includes the complete product rule
\(\partial_i(SM^j)=S'\partial_i z\,M^j+S\partial_iM^j\). The modified longitudinal
speed is

\[
 v_L^2=\frac{4-S\alpha^2\chi}{3}.
\]

Exact ranks show that the \(S=1\) lapse/longitudinal and light/longitudinal
coincidences are semisimple. If \(0<\alpha\le1\), \(0<\alpha\chi\le1\), and the switch
has reached one below \(\alpha\chi=4/7\), no coincidence occurs inside the transition:
the lapse condition would require \(S\alpha z=4-6z\), impossible for \(z<4/7\), and
the physical-light condition would require \((S+3)\alpha z=4\), also impossible there.
This proves conditional strong hyperbolicity on that stated domain. It is not a global
theorem, and the numerical campaign below does not establish that every evolved point
remains inside the stated bounds.

## Gauge A0: prescribed stationary 1+log trumpet

The construction here uses only the stationary Schwarzschild equations in
[Bruegmann, arXiv:0904.4418](https://arxiv.org/abs/0904.4418), independently mapped
to the PC-GH variables. Set the mass to one while constructing the dimensionless
table. With areal radius \(R\), inverse areal radius \(S=1/R\), and isotropic radius
\(r\), the regular stationary branch is defined implicitly by

\[
 \alpha^2-1+2S-Ce^\alpha S^4=0,
 \qquad
 C=\frac12R_c^3e^{-\alpha_c},
\]

\[
 \alpha_c=\sqrt{10}-3,
 \qquad
 R_c=\frac{\sqrt{10}+3}{4},
 \qquad
 r_c=0.30345204271479997.
\]

The isotropic-radius equation is nonsingular when written in \(x=\ln r\):

\[
 \frac{dS}{dx}=-S\alpha(S),
 \qquad S(x_c)=R_c^{-1}.
\]

Define the positive physical shift norm and its areal derivative by

\[
 \beta=\sqrt{\alpha^2-1+2/R},
 \qquad
 k=\frac{d\beta}{dR},
 \qquad
 \ell=\frac{\beta}{R}.
\]

In isotropic Cartesian coordinates, with \(n^i=x^i/r\), the complete target state is

\[
\begin{aligned}
A_T&=\alpha^2,&
\chi_T&=(r/R)^2,&
\tilde\gamma_{Tij}&=\delta_{ij},\\
\beta_T^i&=r\ell n^i,&
K_T&=k+2\ell,&
\tilde A_{Tij}&=(k-\ell)(n_i n_j-\delta_{ij}/3),\\
\tilde\Lambda_T^i&=0,&
\pi_T&=-K_T,&
Q_{T kij}&=0.
\end{aligned}
\]

The remaining reduction fields are coordinate derivatives of these primaries. In
particular, if \(\beta_T^r=r\ell\),

\[
 B_{Ti}{}^j=\frac{\beta_T^r}{r}\delta_i{}^j
 +\left(\partial_r\beta_T^r-\frac{\beta_T^r}{r}\right)n_i n^j.
\]

Gauge A0 is prescribed as a fixed function of \(r/M\), not recomputed from the evolved
state. Stationarity and the configuration equations give

\[
 h_{\perp,T}=-\alpha K+\frac{\beta_T^r}{2A}\partial_r A
 =(2-\alpha)K,
\]

where the second equality uses stationary advective 1+log slicing, and

\[
 h_T^i=\left[-\beta_T^r\partial_r\beta_T^r
 -\frac12(A_T\partial_r\chi_T-\chi_T\partial_rA_T)\right]n^i.
\]

The source derivatives used by the standard-order gradient equations are derivatives
of these prescribed radial functions. Thus Gauge A0 depends on coordinates only and
satisfies the metric-only GH principal-part restriction (`PROVED`). On this exact
stationary vacuum target, the source is \(H_T=-\Gamma_T\), so \(C_T=0\); the reduced
covariant equations and every PC-GH RHS vanish (`PROVED`, before table interpolation).

Let \(R_0\) be the smaller positive root of
\(R_0^4-2R_0^3+C=0\) and
\(\gamma=(2-R_0)/(6-4R_0)\). The inner behavior is

\[
 \chi_T=O(r^2),\qquad A_T=O(r^{2/\gamma}),\qquad
 2/\gamma\simeq2.18259.
\]

Consequently all 55 stored target fields are bounded, while the exact theorem remains
on \(r>0\). The direction-dependent bounded limit of the radial trace-free tensor is
not asserted to be smooth at the compactified point. The generated table spans only a
declared positive-radius interval; production interpolation must fail closed outside it
and must never evaluate the regularity ratios at \(A=\chi=0\).

`generate_gauge_a0_table.py` independently integrates the isotropic ODE, constructs the
regular target and source scalars, checks the implicit equation and both source
definitions, and measures the inner powers. Binary64 table generation is numerically
audited. Production interpolation, residual ladders, and the multi-precision
cancellation audit are recorded separately in `docs/pc_gh_qualification_log.md`.

## Gauge A1: bounded feedback linearization

Consider the bounded metric-only feedback proposed after Gauge A0,

\[
h_\perp=h_{\perp,T}+\mu_L\chi\frac{A-A_T}{A+A_T},\qquad
h^i=h_T^i-\mu_S\chi(\beta^i-\beta_T^i).
\]

At the stationary target its first variation is

\[
\delta h_\perp=\frac{\mu_L\chi_T}{2A_T}\,\delta A,
\qquad
\delta h^i=-\mu_S\chi_T\,\delta\beta^i.
\]

Therefore the only new lower-order terms in the configuration/gradient system are

\[
\begin{aligned}
\delta(\partial_t A)&\supset-\mu_L\chi_T\delta A,\\
\delta(\partial_tY_i)&\supset
 -\mu_L(X_{Ti}\delta A+\chi_T\delta Y_i),\\
\delta(\partial_t\beta^j)&\supset-\mu_S\chi_T\delta\beta^j,\\
\delta(\partial_tB_i{}^j)&\supset
 -\mu_S(X_{Ti}\delta\beta^j+\chi_T\delta B_i{}^j).
\end{aligned}
\]

This linearization does not act on the (Q) sector. That omission is decisive for the
observed Gauge-A0 frozen mode. On the radial axis write the stationary shift gradient as

\[
B_i{}^j=\operatorname{diag}(u,t,t).
\]

For a spatially constant, tangential-derivative-index, tangential trace-free (Q)
perturbation, the standard-order (Q) equation closes on an algebraically projected
subspace and gives

\[
\partial_t\delta Q=\lambda_Q\delta Q,
\qquad
\lambda_Q=3t-\frac23(u+2t)=\frac{5t-2u}{3}.
\]

At (r=M), the production target has
(u=-0.02183002615/M) and (t=0.15424545392/M), hence

\[
\lambda_Q=0.27162910729/M>0.
\]

This subspace is unchanged by either (mu_L) or (mu_S), explaining the positive
floor in the direct low-rank Gauge-A1 scans. Classification: the bounded Gauge A1 above
is `PROVED` not to clear the full frozen-operator gate for any feedback coefficients.
It must not be implemented as a purported cure for this mode.

### Pointwise Q rate versus the reduction-constraint rate

The eigenvalue above belongs to the complete *pointwise-frozen* first-order state
operator, in which `Q` may be perturbed independently of the conformal metric.  At
`k=0` this is off the reduction manifold and must not be mislabeled as a physical
Fourier mode.  The corresponding reduction constraint is

\[
R^{\gamma}_{\ell ij}=Q_{\ell ij}-\partial_\ell\tilde\gamma_{ij}.
\]

For the standard gradient extension, linearizing the nonadvective metric source in a
tangential trace-free reduction perturbation gives

\[
\delta(\partial_\ell^{(1)}F_{ij})
=\left[2t-\frac23(u+2t)\right]R^{\gamma}_{\ell ij}
=\frac23(t-u)R^{\gamma}_{\ell ij}.
\]

The full pointwise `Q` row has one additional
`B_ell{}^ell R^gamma_ell=t R^gamma_ell` term, producing `(5t-2u)/3`.  In the actual
constraint equation that term cancels the same contribution from
`partial_ell(beta^k R^gamma_k)`.  Therefore the true local zero-wave-number
reduction-constraint rate is

\[
\lambda_R=\frac23(t-u)=0.11738365338/M>0
\qquad (r=M),
\]

not `0.27162910729/M`.  This refinement weakens the quantitative growth estimate but
does not remove the sign failure.  Gauge A1 still cannot change it because the
conformal-metric source contains neither `h_perp` nor `h^i`.  The identities are
checked exactly by `verify_reduction_constraint_growth.py`.  Classification:
positive Gauge-A0 tangential trace-free Q reduction rate `PROVED` for the stated local
radial background; no claim is made here about a global eigenmode.

## Explicit map to standard first-order GH

Fix the standard FO-GH variables and sign convention by

\[
\psi_{\mu\nu}=g_{\mu\nu},\qquad
\Phi_{k\mu\nu}=\partial_k g_{\mu\nu},\qquad
\Pi_{\mu\nu}=-n^\rho\partial_\rho g_{\mu\nu}
=-\frac1\alpha D_0g_{\mu\nu}.
\]

On \(A>0\), \(\chi>0\), use the positive root \(\alpha=\sqrt A\), and set

\[
\gamma_{ij}=\chi^{-1}\tilde\gamma_{ij},\qquad
G_{kij}:=\partial_k\gamma_{ij}
=\chi^{-1}Q_{kij}-\chi^{-2}X_k\tilde\gamma_{ij}.
\]

The spacetime metric and its spatial derivative are reconstructed explicitly as

\[
\begin{aligned}
g_{ij}&=\gamma_{ij},&
g_{0i}&=\gamma_{ij}\beta^j,&
g_{00}&=-A+\gamma_{ij}\beta^i\beta^j,\\
\Phi_{kij}&=G_{kij},&
\Phi_{k0i}&=G_{kij}\beta^j+\gamma_{ij}B_k{}^j,\\
\Phi_{k00}&=-Y_k+G_{kij}\beta^i\beta^j
+2\gamma_{ij}\beta^iB_k{}^j.
\end{aligned}
\]

Define

\[
\begin{aligned}
v^i:=D_0\beta^i={}&h^i+A\chi\tilde\Lambda^i
+\frac12\tilde\gamma^{ij}(AX_j-\chi Y_j),\\
D_0A={}&2A(\alpha\pi-h_\perp),\\
D_0\gamma_{ij}={}&-2\alpha K_{ij}
+\gamma_{ik}B_j{}^k+\gamma_{jk}B_i{}^k,
\end{aligned}
\]

with

\[
K_{ij}=\chi^{-1}\left(\tilde A_{ij}
+\frac13\tilde\gamma_{ij}K\right).
\]

The remaining components of \(D_0g_{\mu\nu}\), and hence all of \(\Pi\), are

\[
\begin{aligned}
D_0g_{ij}&=D_0\gamma_{ij},\\
D_0g_{0i}&=(D_0\gamma_{ij})\beta^j+\gamma_{ij}v^j,\\
D_0g_{00}&=-D_0A+(D_0\gamma_{ij})\beta^i\beta^j
+2\gamma_{ij}\beta^iv^j.
\end{aligned}
\]

This forward map uses no derivatives beyond the stored first-order variables.

Conversely, recover the ADM configuration from \(\psi\) by

\[
\gamma_{ij}=\psi_{ij},\qquad
\beta^i=\gamma^{ij}\psi_{0j},\qquad
A=\gamma_{ij}\beta^i\beta^j-\psi_{00},
\]

then \(\chi=(\det\gamma)^{-1/3}\) and
\(\tilde\gamma_{ij}=\chi\gamma_{ij}\). From \(\Phi\),

\[
\begin{aligned}
B_k{}^i&=\gamma^{ij}(\Phi_{k0j}-\Phi_{kj\ell}\beta^\ell),\\
Y_k&=-\Phi_{k00}+\Phi_{kij}\beta^i\beta^j
+2\gamma_{ij}\beta^iB_k{}^j,\\
X_k&=-\frac\chi3\gamma^{ij}\Phi_{kij},\\
Q_{kij}&=\chi\Phi_{kij}+X_k\gamma_{ij}.
\end{aligned}
\]

Set \(D_0g_{\mu\nu}=-\alpha\Pi_{\mu\nu}\). The inverse velocity map is

\[
\begin{aligned}
D_0\beta^i&=\gamma^{ij}(D_0g_{0j}-D_0\gamma_{jk}\beta^k),\\
D_0A&=-D_0g_{00}+(D_0\gamma_{ij})\beta^i\beta^j
+2\gamma_{ij}\beta^iD_0\beta^j,\\
K_{ij}&=-\frac1{2\alpha}\left(D_0\gamma_{ij}
-\gamma_{ik}B_j{}^k-\gamma_{jk}B_i{}^k\right),\\
K&=\gamma^{ij}K_{ij},\qquad
\tilde A_{ij}=\chi\left(K_{ij}-\frac13\gamma_{ij}K\right),\\
\pi&=\frac1\alpha\left(\frac{D_0A}{2A}+h_\perp\right),\\
\tilde\Lambda^i&=\frac1{A\chi}\left[D_0\beta^i-h^i
-\frac12\tilde\gamma^{ij}(AX_j-\chi Y_j)\right].
\end{aligned}
\]

The two displayed maps are mutual inverses on the conformal algebraic-constraint
manifold. `verify_fo_gh_map.py` checks the complete round trip with exact rational
arithmetic for a non-diagonal state, including
\(\det\tilde\gamma=1\),
\(\tilde\gamma^{ij}Q_{kij}=0\), and
\(\tilde\gamma^{ij}\tilde A_{ij}=0\).

### Equation equivalence and reduction ordering

The primary equations above are tensor projections of the same covariant reduced GH
equation used to obtain standard FO-GH. The configuration equations are the inverse
metric map, and the gradient equations are its first-order reduction. On the algebraic
and first-order reduction constraint manifold, substitution of the explicit map
therefore gives the standard FO-GH equations exactly, without assuming that the GH
constraint itself vanishes.

Off the first-order reduction constraint manifold, replacing
\(\beta^k\partial_iG_k\) by \(\beta^k\partial_kG_i\) is precisely the
\(\gamma_1=-1\) standard ordering. No \(\gamma_2\) reduction-constraint damping term
is present, so this baseline corresponds to \(\gamma_2=0\). Classification:
nonlinear equation equivalence to standard FO-GH `PROVED ON r>0`, for prescribed or
metric-only differentiable \(H_\mu\) and on the stated algebraic/reduction constraint
manifold.

### Principal symbol and symmetrizer

Freeze a background satisfying \(A>0\), \(\chi>0\), and positive-definite
\(\tilde\gamma_{ij}\). For a unit spatial covector \(s_i\), the standard
\((\gamma_1,\gamma_2)=(-1,0)\) FO-GH characteristic fields are

\[
\begin{array}{c|c}
\text{field} & \text{coordinate speed}\\ \hline
\delta\psi_{\mu\nu} & 0\\
(\delta_i{}^k-s_i s^k)\delta\Phi_{k\mu\nu} & -\beta^is_i\\
\delta\Pi_{\mu\nu}\pm s^k\delta\Phi_{k\mu\nu}
&-\beta^is_i\pm\alpha.
\end{array}
\]

The linearization of the explicit map above sends these fields bijectively to the
tangent space of the PC-GH algebraic constraints, so this is also a complete
diagonalization of the PC-GH principal symbol.

For an explicit positive energy, let
\(m^{\mu\nu}=g^{\mu\nu}+2n^\mu n^\nu\), which is positive definite, and let
\(M\) be its induced positive inner product on symmetric two-tensors. Then

\[
E_{FO}=c\,M(\delta\psi,\delta\psi)
+M(\delta\Pi,\delta\Pi)
+\gamma^{ij}M(\delta\Phi_i,\delta\Phi_j),\qquad c>0,
\]

symmetrizes the frozen FO-GH principal matrices. If \(J\) is the Jacobian of the
forward PC-GH-to-FO-GH map, the pulled-back form

\[
E_{PC}(\delta u)=E_{FO}(J\delta u)
\]

is positive definite and symmetrizes the PC-GH principal symbol because \(J\) is
invertible on the constrained tangent space. This proves symmetric hyperbolicity for
every fixed \(r>0\). It does not prove a uniformly positive puncture-limit
symmetrizer: factors of \(A^{-1}\) and \(\chi^{-1}\) in the map may make the pulled-back
energy singular as \(r\to0\).

## Consistent algebraic projection

Let

\[
s=(\det\tilde\gamma)^{-1/3},
\qquad \tilde\gamma'_{ij}=s\tilde\gamma_{ij}.
\]

Jacobi's determinant identity gives

\[
\partial_k\ln s=-\frac13\tilde\gamma^{ab}Q_{kab}.
\]

Therefore the derivative of the projected metric is

\[
Q'_{kij}=s\left(Q_{kij}
-\frac13\tilde\gamma_{ij}\tilde\gamma^{ab}Q_{kab}\right).
\]

It obeys \((\tilde\gamma')^{ij}Q'_{kij}=0\) identically. This is the consistent Q
projection (`PROVED` for nonsingular \(\tilde\gamma\)); it is not an independently
guessed trace subtraction. `verify_q_projection.py` checks both product-rule consistency
and the projected trace exactly.

The metric and Q projections above do not by themselves specify the numerically best
simultaneous projection of \(\tilde A_{ij}\), nor the norm in which correction size
should be monitored. The ordinary trace removal after metric rescaling is algebraically
valid, but its coupling to the full semi-discrete energy remains `NOT ESTABLISHED`.

## Evolution-equation audit status

| Sector | Status | Evidence or missing work |
|---|---|---|
| `D0 chi`, `D0 gtilde` | `PROVED` | ADM kinematics and determinant differentiation above. |
| `D0 A`, `D0 beta` | `PROVED ON r>0` | Direct contracted-Christoffel projections above. |
| regular Hessian, Hamiltonian, scaled momentum, S-TF | `PROVED ON r>0` | Written derivation plus exact SymPy checks. |
| Brown first-order conformal Ricci | `PROVED` under algebraic/reduction hypotheses | Coordinate Ricci rearrangement plus exact non-diagonal component regression. |
| K equation | corrected equation `PROVED ON r>0`; supplied target `FAILED` | Covariant spatial trace and exact flat counterexample above. |
| Atilde equation | corrected equation `PROVED ON r>0`; supplied nonlinear Z term `FAILED` | Covariant trace-free spatial projection and exact arbitrary-Q residual above. |
| pi equation | `PROVED ON r>0` | Normal-normal covariant projection and exact symbolic comparison. |
| Lambda equation | corrected equation `PROVED ON r>0`; supplied target `FAILED` | Mixed covariant projection and exact lapse-acceleration residual above. |
| X/Y/Q/B standard-order equations | `PROVED ON r>0`, Gauge-A conditional | Exact product-rule and curl-ordering audit above. |
| equivalence to standard FO-GH for r>0 | `PROVED ON r>0`, constraint-manifold and gauge conditional | Explicit mutually inverse variable map and common covariant origin above. |
| symmetric hyperbolicity for r>0 | `PROVED ON r>0` for the established metric-only/prescribed Gauge-A class | Complete pulled-back FO-GH characteristic basis and positive symmetrizer above; no uniform puncture-limit claim. The direct moving-puncture gauge changes this symbol. |
| direct `z4c_mp` gauge | equations `PROVED`; global strong hyperbolicity `FAILED` | Exact STANDARD Y/B identities; defective surfaces at \(\alpha^2\chi=4/3\), \(\alpha=2\), and \(\alpha\chi=2/3\). |
| switched `z4c_mp_hyperbolic` gauge | equations `PROVED`; strong hyperbolicity `PROVED` only on the stated conditional domain | Exact modified symbol and eigenspace ranks; no spectral/SAT promotion or global-domain claim. |
| Gauge A0 continuum target | `PROVED`; pointwise production gate passed, frozen stability failed | Stationary construction above and bounded numerical evidence in the qualification log. |
| bounded Gauge A1 feedback | linearization `PROVED`; frozen clearance `FAILED` | The invariant tangential trace-free Q subspace has positive rate `(5t-2u)/3` and is untouched by `mu_L,mu_S`. |
| Gauge B scaled driver | `NOT ESTABLISHED` | Deferred until Gauge A qualification and a new driver derivation. |

The baseline production primary and gradient RHS are authorized on \(r>0\), subject to
the metric-only/prescribed Gauge-A dependency restriction and the exact corrected
equations above. No puncture-limit regularity or Gauge-A0 qualification is authorized.

## Reproduction

Run the current exact symbolic gates with:

```bash
python3 -m venv /tmp/pc-gh-sympy
/tmp/pc-gh-sympy/bin/python -m pip install -r analysis/pc_gh_symbolic/requirements.txt
/tmp/pc-gh-sympy/bin/python analysis/pc_gh_symbolic/run_all.py
```

The scripts use exact SymPy simplification or exact rational component comparisons. They
do not use floating-point tolerances.
