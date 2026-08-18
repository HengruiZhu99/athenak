# Exact-pullback FO-GH formulation investigation

## Status

**FORMULATION NOT ESTABLISHED.**  The requested regular projections fail the
individual-intermediate puncture-regularity gate when they are applied exactly
to the Lindblom--Szilagyi improved gauge driver.  This is an analytic stop
condition, not a numerical parameter choice.  No replacement production
continuum system has been installed, and the previous 63-field system remains
available unchanged as the control.

The investigation branch was created from parent
`3a9ba3bb5997e3d3071fed875b2fb0a1672303a8`.  Its scope is vacuum FO-GH only.

## Conventions and parent system

The parent state has 58 independent fields,

\[
 Z=\{\psi_{ab},\Pi_{ab},\Phi_{iab},H_a,Z_a\},\qquad
 \Pi_{ab}=-n^c\partial_c\psi_{ab},\qquad
 \Phi_{iab}=\partial_i\psi_{ab},
\]

with 10, 10, 30, 4, and 4 components, respectively.  Here the signature is
`(-,+,+,+)`, \(n^a=(1,-\beta^i)/\alpha\), and
\(Z_a=\theta_a+\eta_H H_a\).  The equations and signs are those of
Lindblom et al., [arXiv:gr-qc/0512093](https://arxiv.org/abs/gr-qc/0512093),
Eqs. (35)--(37), and Lindblom--Szilagyi,
[arXiv:0904.4873](https://arxiv.org/abs/0904.4873), Eqs. (B1)--(B3), (9), and
(11). Brown, [arXiv:1109.1707](https://arxiv.org/abs/1109.1707), is used only
as the 3+1/conformal algebra oracle.
The stationary trumpet scalings are checked against Hannam et al.,
[arXiv:0804.0628](https://arxiv.org/abs/0804.0628), in addition to the powers
specified by the controlling instructions.

Writing the lapse and shift as \(N=\alpha\) and \(N^i=\beta^i\), the parent
metric equations are

\[
\begin{split}
\partial_t\psi_{ab}-(1+\gamma_1)\beta^k\partial_k\psi_{ab}
  &=-\alpha\Pi_{ab}-\gamma_1\beta^i\Phi_{iab},\\
\partial_t\Pi_{ab}-\beta^k\partial_k\Pi_{ab}
 +\alpha\gamma^{ki}\partial_k\Phi_{iab}
 -\gamma_1\gamma_2\beta^k\partial_k\psi_{ab}
 +2\alpha\partial_{(a}H_{b)}
  &=S^{\Pi}_{ab},\\
\partial_t\Phi_{iab}-\beta^k\partial_k\Phi_{iab}
 +\alpha\partial_i\Pi_{ab}-\alpha\gamma_2\partial_i\psi_{ab}
  &=\tfrac12\alpha n^c n^d\Phi_{icd}\Pi_{ab}
    +\alpha\gamma^{jk}n^c\Phi_{ijc}\Phi_{kab}
    -\alpha\gamma_2\Phi_{iab}.
\end{split}
\]

The complete algebraic source used for the parent \(\Pi\) equation is

\[
\begin{split}
S^{\Pi}_{ab}={}&-\tfrac12\alpha n^c n^d\Pi_{cd}\Pi_{ab}
-\alpha n^c\Pi_{ci}\gamma^{ij}\Phi_{jab}
+2\alpha\psi^{cd}\gamma^{ij}\Phi_{ica}\Phi_{jdb}\\
&-\alpha\psi^{cd}\psi^{ef}\Gamma_{ace}\Gamma_{bdf}
+\alpha\gamma_0[2\delta^c{}_{(a}n_{b)}-\psi_{ab}n^c]
  (H_c+\psi^{ef}\Gamma_{cef})\\
&+2\alpha\Gamma^c{}_{ab}H_c
-\gamma_1\gamma_2\beta^i\Phi_{iab}.
\end{split}
\]

The initial parent parameters are \(\gamma_1=-1\) and configurable positive
\(\gamma_0,\gamma_2\).  No Brown-style independent constraint additions are
part of this parent system.

The improved driver is

\[
D_0H_a=Z_a-(\mu_H+\eta_H)H_a+\mu_HF_a,
\qquad
\partial_t Z_a=-\eta_H\mu_H(H_a-F_a),
\]

where \(D_0=\partial_t-\beta^i\partial_i\).  These equations follow exactly
from Eqs. (9) and (11) after defining \(Z_a=\theta_a+\eta_HH_a\).

The parent system has the characteristic fields and symmetrizer of
Lindblom--Szilagyi Appendix B.  In particular its speeds are
\(-(1+\gamma_1)n_i\beta^i\), \(-n_i\beta^i\pm\alpha\),
\(-n_i\beta^i\), and zero for the metric, physical, transverse/driver, and
\(Z\)-type families.  This known parent theorem is not claimed for a failed
regular pullback.

## Proposed independent regular chart

The requested redundant production storage would still contain 63 fields,
but the local chart for a hyperbolicity proof has exactly 58:

| field | independent components |
|---|---:|
| unit-determinant \(\tilde\gamma_{ij}\) | 5 |
| \(\chi,A,\beta^i\) | 5 |
| trace-free \(\tilde A_{ij}\) | 5 |
| \(K,\Lambda^i,\pi\) | 5 |
| tangent \(Q_{kij}\) | 15 |
| \(X_i,Y_i,B_i{}^j\) | 15 |
| \(h_A,z_A\) | 8 |
| total | 58 |

The five stored algebraic constraints would be
\(\det\tilde\gamma-1=0\),
\(\tilde\gamma^{ij}\tilde A_{ij}=0\), and
\(\tilde\gamma^{ij}Q_{kij}=0\) for each \(k\).

At finite \(A>0\), \(\chi>0\), the metric part of the proposed map is

\[
\alpha=\sqrt A,\quad
\gamma_{ij}=\tilde\gamma_{ij}/\chi,\quad
K_{ij}=(\tilde A_{ij}+\tilde\gamma_{ij}K/3)/\chi,
\]

\[
\psi_{00}=-A+\gamma_{ij}\beta^i\beta^j,\quad
\psi_{0i}=\gamma_{ij}\beta^j,\quad
\psi_{ij}=\gamma_{ij}.
\]

The first spatial derivatives are obtained algebraically from
\(Q=\partial\tilde\gamma\), \(X=\partial\chi\),
\(Y=\partial A\), and \(B=\partial\beta\).  Brown's velocity map gives
\(\rho^i=\chi\Lambda^i+\tilde\gamma^{ij}X_j/2\), and the corresponding
coordinate-time metric derivatives determine \(\Pi_{ab}\).  This is an
away-from-puncture oracle map only; production may not reconstruct its
divergent physical components.

## Regular moving-puncture target

The target requested in the controlling specification is algebraically
consistent:

\[
f_\perp=\sqrt A\,\pi+2K,
\]

\[
f^i=(\nu-A\chi)\Lambda^i
-\frac A2\tilde\gamma^{ij}X_j
+\frac\chi2\tilde\gamma^{ij}Y_j
-\eta_\beta\beta^i,
\qquad \nu=3/4.
\]

With

\[
D_0A=2A(\sqrt A\,\pi-h_\perp),
\]

and

\[
D_0\beta^i=h^i+A\chi\Lambda^i
+\frac A2\tilde\gamma^{ij}X_j
-\frac\chi2\tilde\gamma^{ij}Y_j,
\]

setting \(h=f\) gives exactly \(D_0A=-4AK\) and
\(D_0\beta^i=\nu\Lambda^i-\eta_\beta\beta^i\).  Random dense checks agree
to a worst relative error of \(4.21\times10^{-16}\).

## Exact driver pullback

For a spacetime covector, Brown's projections imply

\[
h_\perp=\alpha H_\perp=H_0-\beta^iH_i,
\qquad
h^i=A\chi\tilde\gamma^{ij}H_j.
\]

Thus the exact inverse at finite positive \(A,\chi\) is

\[
H_i=\frac{\tilde\gamma_{ij}h^j}{A\chi},
\qquad
H_0=h_\perp+\beta^iH_i,
\]

and likewise for \(Z_a\).  Differentiating this map, rather than copying the
unweighted relaxation equations, gives

\[
\boxed{
D_0h_\perp=z_\perp-(\mu_H+\eta_H)h_\perp+\mu_Hf_\perp
-(D_0\beta^i)\frac{\tilde\gamma_{ij}h^j}{A\chi}}
\]

and

\[
\boxed{
D_0h^i=z^i-(\mu_H+\eta_H)h^i+\mu_Hf^i
+\left[D_0\ln(A\chi)\delta^i{}_j
-\tilde\gamma^{ik}D_0\tilde\gamma_{kj}\right]h^j.}
\]

The exact zero-speed equation transforms to

\[
\boxed{
\partial_tz_\perp
=-(\partial_t\beta^i)\frac{\tilde\gamma_{ij}z^j}{A\chi}
-\eta_H\mu_H(h_\perp-f_\perp)}
\]

and

\[
\boxed{
\partial_tz^i=
\left[\partial_t\ln(A\chi)\delta^i{}_j
-\tilde\gamma^{ik}\partial_t\tilde\gamma_{kj}\right]z^j
-\eta_H\mu_H(h^i-f^i).}
\]

An independent dense 4-by-4 weight-matrix oracle verifies these four component
equations over 256 random nonsingular states with worst relative mismatch
\(1.21\times10^{-15}\).

## Puncture regularity stop

For a stationary Cartesian trumpet, use

| quantity | leading power |
|---|---:|
| \(\chi\) | 2 |
| \(A=\alpha^2\) | \(2p\), \(p\simeq1.091\) |
| \(X_i\) | 1 |
| \(Y_i\) | \(2p-1\) |
| \(\beta^i\), \(h^i=f^i\) | 1 |
| \(B_i{}^j\) | 0 |
| \(D_0\beta^i=-\beta^j\partial_j\beta^i\) | 1 |

The normal-projection connection term has power

\[
(D_0\beta)\,h/(A\chi)=O(r^{1+1-(2p+2)})=O(r^{-2p})
=O(r^{-2.182}).
\]

It is an individual exact-pullback term, not a removable implementation
artifact.  In a stationary regular solution, rearranging the boxed
\(D_0h_\perp\) equation requires \(z_\perp=O(r^{-2p})\) to cancel it unless
another supposedly regular RHS quantity is allowed to diverge.  Either choice
violates the hard requirements that every production intermediate and every
production evolved variable remain finite.  At the nearest cell this grows as
\(\Delta x^{-2p}\), so keeping the puncture off-grid does not cure the
resolution limit.

The automated audit evaluates the sequence \(r=2^{-n}\) in binary64 and with a
100-digit `decimal` reference.  The divergent magnitude reaches
\(1.64\times10^{41}\) by \(n=64\); the double/reference relative difference
remains about \(2.7\times10^{-15}\), showing that this is genuine asymptotic
growth rather than a low-precision false positive.

This triggers the controlling stop condition before an exact regular Einstein
RHS, symmetrizer, frozen-spectrum scan, finite-difference system, or puncture
ablation can be promoted.  The known parent symmetrizer does not rescue a
singular requested variable map.  No theorem at the puncture, no new-system
hyperbolicity claim, and no new puncture-stability claim is made.

## Reproduction

```sh
PYTHONPATH=tst/test_suite python3 \
  tst/test_suite/fo_gh/exact_driver_pullback_audit.py \
  --samples 256 --maximum-n 64
```

Exit status 2 is the expected fail-closed result.  The same checks can be
called from `test_exact_driver_pullback_audit.py`; the local system Python did
not provide the `pytest` package, so the three test functions were also invoked
directly and passed.
