# Matched-weight exact-pullback FO-GH investigation

## Status and scope

This branch investigates a vacuum, puncture-regular pullback of the standard
58-field symmetric-hyperbolic first-order GH system plus the improved
Lindblom--Szilagyi gauge driver. The analytic matched-driver gate V0 passes.
That result removes the *previous gauge-weight mismatch*, but it does not yet
establish the exact Einstein pullback, hyperbolicity at the puncture, or
puncture stability.

No fluid coupling, Kerr--Schild data, horizon finding, excision, floors,
clipping, or puncture resets are in scope.

## Parent driver and sign check

Lindblom--Szilagyi, [arXiv:0904.4873](https://arxiv.org/abs/0904.4873),
Eqs. (9) and (11), use

\[
 \partial_t\theta_a+\eta_H\theta_a=-\eta_H\beta^i\partial_iH_a,
 \qquad
 D_0H_a=-\mu_H(H_a-F_a)+\theta_a,
\]

where \(D_0=\partial_t-\beta^i\partial_i\). Defining
\(Z_a=\theta_a+\eta_HH_a\) gives, without a sign convention change,

\[
 D_0H_a=Z_a-(\mu_H+\eta_H)H_a+\mu_HF_a,
 \qquad
 \partial_tZ_a=-\eta_H\mu_H(H_a-F_a).
\]

This was checked directly against the published equations rather than inferred
from the earlier production relaxation driver.

## Independent reproduction of the old obstruction

The old projection used

\[
 h_\perp=H_0-\beta^iH_i,
 \qquad h^i=A\chi\tilde\gamma^{ij}H_j.
\]

Direct block inversion gives

\[
 H_i=\frac{\tilde\gamma_{ij}h^j}{A\chi},
 \qquad H_0=h_\perp+\beta^iH_i.
\]

Differentiating the first row along \(D_0\) produces

\[
 D_0h_\perp\supset
 -(D_0\beta^i)\frac{\tilde\gamma_{ij}h^j}{A\chi}.
\]

With \(A=O(r^{2p})\), \(\chi=O(r^2)\),
\(D_0\beta=O(r)\), and \(h^i=O(r)\), its power is
\(2-(2p+2)=-2p=-2.182\). A new NumPy block-matrix oracle, independent of the
previous audit module, reproduced the inverse to relative error
\(1.03\times10^{-15}\).

## Matched gauge map

Let \(w=A\chi\), and apply it to all four covector components:

\[
 \widehat h_\perp=w(H_0-\beta^iH_i),\qquad
 \widehat h_i=wH_i,
\]

with the identical definitions for \(\widehat z\). In matrix form,

\[
 W=wT,\qquad
 T=\begin{pmatrix}1&-\beta^j\\0&\delta_i{}^j\end{pmatrix}.
\]

The exact finite-radius inverse is

\[
 H_i=\widehat h_i/w,
 \qquad
 H_0=(\widehat h_\perp+\beta^i\widehat h_i)/w.
\]

The normalized map \(\overline W=T\) and inverse
\(T^{-1}=\left(\begin{smallmatrix}1&\beta^j\\0&I\end{smallmatrix}\right)\)
contain no puncture weight. For 1000 random samples with each
\(\beta^i\in[-2,2]\), the largest measured two-norm condition number was
12.066 and the inverse residual was zero in binary64.

## Exact matched-driver pullback

Define

\[
 c_H=\mu_H+\eta_H,\quad
 \ell_0=D_0\ln w,\quad
 \ell_t=\partial_t\ln w.
\]

From
\((D W)W^{-1}=(D\ln w)I+(D T)T^{-1}\), the exact component equations are

\[
 \boxed{D_0\widehat h_\perp=(\ell_0-c_H)\widehat h_\perp
 +\widehat z_\perp+\mu_H\widehat f_\perp
 -(D_0\beta^i)\widehat h_i},
\]

\[
 \boxed{D_0\widehat h_i=(\ell_0-c_H)\widehat h_i
 +\widehat z_i+\mu_H\widehat f_i},
\]

\[
 \boxed{\partial_t\widehat z_\perp=\ell_t\widehat z_\perp
 -(\partial_t\beta^i)\widehat z_i
 -\eta_H\mu_H(\widehat h_\perp-\widehat f_\perp)},
\]

\[
 \boxed{\partial_t\widehat z_i=\ell_t\widehat z_i
 -\eta_H\mu_H(\widehat h_i-\widehat f_i)}.
\]

Thus the old division by \(A\chi\) is absent from the shift-basis mixing
term. An explicit component implementation and a separately differentiated
dense \(4\times4\) matrix implementation agreed over 1000 random nonsingular
states with worst relative error \(2.06\times10^{-15}\).

## The \(A,Y_i\) lapse representation and target

Use

\[
 A=\alpha^2,\qquad Y_i=\partial_iA,qquad \alpha=\sqrt A.
\]

The matched target is

\[
 \widehat f_\perp=A\chi(\sqrt A\,\pi+2K),
\]

\[
 \widehat f_i=(\nu-A\chi)\tilde\gamma_{ij}\Lambda^j
 -\frac A2X_i+\frac\chi2Y_i
 -\eta_\beta\tilde\gamma_{ij}\beta^j,qquad \nu=3/4.
\]

The primary gauge kinematics in these variables are

\[
 D_0A=2A\sqrt A\,\pi-\frac{2\widehat h_\perp}{\chi},
\]

\[
 D_0\beta^i=\tilde\gamma^{ij}\widehat h_j+A\chi\Lambda^i
 +\frac A2\tilde\gamma^{ij}X_j
 -\frac\chi2\tilde\gamma^{ij}Y_j.
\]

Substitution of \(\widehat h=\widehat f\) gives exactly

\[
 D_0A=-4AK,qquad
 D_0\beta^i=\nu\Lambda^i-\eta_\beta\beta^i.
\]

The random-state target oracle's worst relative error was
\(3.10\times10^{-15}\).

The logarithmic rates must be evaluated as regular contractions:

\[
 \ell_0=2\sqrt A\,\pi-2\frac{\widehat h_\perp}{A\chi}
 +\frac23(\sqrt A K-B_k{}^k),
\]

\[
 \ell_t=\ell_0+\frac{\beta^iY_i}{A}
 +\frac{\beta^iX_i}{\chi}.
\]

Production must form the contracted ratios shown here, never the singular
vectors \(Y_i/A\) or \(X_i/\chi\) as fields or diagnostics.

## Stationary-trumpet power audit

For \(p=1.091\), the matched-driver leading powers are:

| quantity | power of \(r\) |
|---|---:|
| \(A\), \(\chi\), \(w\) | \(2p\), 2, \(2p+2=4.182\) |
| \(Y_i\), \(\partial_jY_i\) | \(2p-1=1.182\), \(2p-2=0.182\) |
| \(\widehat h_\perp\), \(\widehat f_\perp\) | 4.182 |
| \(\widehat h_i\), \(\widehat f_i\) | 1 |
| \((D_0\beta^i)\widehat h_i\) | 2 |
| stationary-balance \(\widehat z_\perp\) | 2 |
| stationary-balance \(\widehat z_i\) | 1 |
| \(\widehat h_\perp/w\) | 0 |
| \(\beta^iY_i/A\), \(\beta^iX_i/\chi\) | 0 |

The scanner enumerates every individual term in all four driver equations and
the logarithmic-rate and target intermediates. The minimum power is zero; no
negative production power occurs in this driver sector. Its scan covers
\(r=2^{-n}M\), \(n=1,\ldots,64\), comparing binary64 with 110-digit Decimal
arithmetic. The largest relative representation difference was
\(1.71\times10^{-14}\).

## Independent 58-dimensional Einstein chart

The audit uses five free components of \(\tilde\gamma\), solving
\(\det\tilde\gamma=1\) for the sixth. It uses five free components of each of
\(\tilde A\) and the three \(Q_k\), solving their traces with
\(\tilde\gamma^{ij}\). Together with the remaining unconstrained fields this
is an explicit 58-dimensional chart, not a 58-to-63 Jacobian.

At finite positive \(A,\chi\), its map to the standard parent begins with

\[
 \gamma_{ij}=\tilde\gamma_{ij}/\chi,\qquad
 K_{ij}=(\tilde A_{ij}+\tilde\gamma_{ij}K/3)/\chi,
\]

\[
 \psi_{00}=-A+\gamma_{ij}\beta^i\beta^j,\quad
 \psi_{0i}=\gamma_{ij}\beta^j,\quad
 \psi_{ij}=\gamma_{ij}.
\]

The spatial derivatives are

\[
 \partial_k\gamma_{ij}=Q_{kij}/\chi
 -X_k\tilde\gamma_{ij}/\chi^2,
\]

\[
 \Phi_{kij}=\partial_k\gamma_{ij},\quad
 \Phi_{k0i}=\partial_k\gamma_{ij}\beta^j+\gamma_{ij}B_k{}^j,
\]

\[
 \Phi_{k00}=-Y_k+\partial_k\gamma_{ij}\beta^i\beta^j
 +2\gamma_{ij}\beta^iB_k{}^j.
\]

Using the gauge kinematics above gives \(D_0\gamma\), \(D_0\beta\), and
\(D_0\alpha\), hence \(\Pi_{ab}=-D_0\psi_{ab}/\sqrt A\). The matched inverse
supplies \(H_a,Z_a\). A separately coded parent-to-regular inverse recovers
\(A,\chi,\beta,Q,X,Y,B,\tilde A,K,\Lambda,\pi,\widehat h,\widehat z\).

Across 1000 random finite-radius states, the regular and parent round trips
have worst relative errors \(2.43\times10^{-15}\) and
\(2.82\times10^{-16}\). The five algebraic constraints remain below
\(4.45\times10^{-16}\). Independently differenced forward and inverse tangent
maps compose to the identity within \(1.65\times10^{-9}\).

This establishes the finite-radius algebraic map. The complete pulled-back
lower-order Einstein source was not generated after the conditioning stop
below, so an exact production Einstein RHS is not claimed.

## Finite-radius symmetrizer check

The parent principal matrices were assembled directly from
Lindblom--Szilagyi Eqs. (B1)--(B3), (9), and (11), with
\(\gamma_1=-1\), and the quadratic symmetrizer from Eq. (B16). For the
finite-radius Jacobian \(J\), the audit forms

\[
 A_V(n)=J^{-1}A_Z(n)J,\qquad H_V=J^TH_ZJ.
\]

Minkowski and eight weak random states have positive \(H_V\), real
characteristic speeds to roundoff, and normalized symmetry residuals below
\(2.5\times10^{-15}\). Thus the expected finite-radius inherited theorem is
reproduced.

## Puncture-conditioning stop

Finite-radius diagonalizability is not uniform on the exact wormhole data.
The audit constructs the published characteristic map (B6)--(B10) for a
physical x-directed unit covector, transforms its right eigenvectors with
\(J^{-1}\), normalizes every distinct-speed wave column, and replaces the
38-dimensional repeated zero-speed family and each ten-dimensional repeated
wave family by orthonormal bases for the same three subspaces. Consequently the
result is not an artifact of eigenvector amplitudes or arbitrary bases inside
repeated eigenspaces.

| wormhole radius | \(\kappa(R)\), Jacobian step \(10^{-5}\) | step \(3\times10^{-6}\) |
|---:|---:|---:|
| 0.5 | \(3.569\times10^3\) | \(3.569\times10^3\) |
| 0.25 | \(2.014\times10^5\) | \(2.014\times10^5\) |
| 0.125 | \(3.314\times10^7\) | \(3.314\times10^7\) |
| 0.0625 | \(1.183\times10^{10}\) | \(1.183\times10^{10}\) |

The log--log slope is approximately \(-7.23\). The mechanism is visible
directly in the characteristic inverse. A parent \(u^{\hat3}_a=H_a\) mode
requires a longitudinal \(\Phi\) companion. On a conformally flat wormhole,
mapping that mode to the regular chart gives gradient components of order one
but matched gauge components of order \(w=A\chi\). As \(w\to0\), the
zero-speed driver subspace and the distinct \(\pm\alpha\) wave subspaces become
nearly tangent. Mixing inside the zero-speed eigenspace cannot remove this
angle collapse.

Diagonal equilibration of \(H_V\) does not cure it: its condition number grows
from \(1.01\times10^4\) at \(r=0.5\) to \(2.98\times10^5\) already at
\(r=0.25\), while raw \(\kappa(H_V)\) reaches \(1.36\times10^{16}\).
Removing the collapse would require a transformation singular in \(1/w\), not
a bounded puncture-regular normalization of the requested production chart.

This meets the controlling stop condition that eigenvector/symmetrizer
conditioning not become uncontrollably singular with refinement. Therefore:

**FORMULATION NOT ESTABLISHED.**

The matched weight fixes the old driver-intermediate divergence, but the
combined exact GH characteristic structure is not uniformly strongly
hyperbolic in the requested puncture variables. No production Einstein RHS,
ablation, stationary-trumpet evolution, or puncture ladder is promoted.

## Reproduction

```sh
PYTHONPATH=tst/test_suite python3 \
  tst/test_suite/fo_gh/matched_driver_pullback_audit.py \
  --samples 1000 --maximum-n 64

PYTHONPATH=tst/test_suite python3 \
  tst/test_suite/fo_gh/matched_einstein_map_audit.py \
  --samples 1000 --tangent-samples 8

# Expected exit status 2 at the explicit conditioning stop.
PYTHONPATH=tst/test_suite python3 \
  tst/test_suite/fo_gh/matched_hyperbolicity_audit.py \
  --random-states 8 --directions 16
```
