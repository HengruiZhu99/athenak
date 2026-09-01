# Puncture-conformal first-order generalized harmonic derivation

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
| K equation | `NOT ESTABLISHED` | Spatial/normal projections of the covariant reduced equation and damping signs must be completed. |
| Atilde equation | `NOT ESTABLISHED` | Trace-free spatial projection, Z terms, and damping contributions require an independent component comparison. |
| pi equation | `NOT ESTABLISHED` | Normal-normal projection and source-definition differentiation are incomplete. |
| Lambda equation | `NOT ESTABLISHED` | Time derivative of the spatial GH constraint and standard ordering are incomplete. |
| X/Y/Q/B standard-order equations | `NOT ESTABLISHED` | Algebraic expansion of each `partial_i^(1) F` and curl-ordering comparison is incomplete. |
| equivalence to standard FO-GH for r>0 | `NOT ESTABLISHED` | Requires an explicit invertible variable map and equation comparison. |
| symmetric hyperbolicity for r>0 | `NOT ESTABLISHED` | Principal symbol and positive symmetrizer have not been constructed. |
| Gauge A0 target and table | `NOT ESTABLISHED` | Stationary trumpet construction follows only after the baseline formulation audit. |
| Gauge B scaled driver | `NOT ESTABLISHED` | Deferred until Gauge A qualification and a new driver derivation. |

No production primary or gradient RHS is authorized by this document yet.

## Reproduction

Run the current exact symbolic gates with:

```bash
python3 -m venv /tmp/pc-gh-sympy
/tmp/pc-gh-sympy/bin/python -m pip install -r analysis/pc_gh_symbolic/requirements.txt
/tmp/pc-gh-sympy/bin/python analysis/pc_gh_symbolic/run_all.py
```

The scripts use exact SymPy simplification or exact rational component comparisons. They
do not use floating-point tolerances.
