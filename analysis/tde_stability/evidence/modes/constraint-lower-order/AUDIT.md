# Residual Z4c lower-order constraint audit

The next justified bounded experiment is **lapse-adjusted constraint damping in the current Z4c equations**, applied consistently to Khat, Theta and Gamma. It is a constraint addition that vanishes on the physical constraint surface, not interior field zeroing. A full CCZ4 source restoration is a separate, coupled formulation change and is not automatically stabilizing on this trumpet. No source or job was changed by this audit.

## Definitions and code convention

Let `gtilde` have unit determinant, `chi=psi^-4`, `K=Khat+2Theta`, and

\[
Q^i=\Gamma^i_{\rm evolved}-\Gamma^i_{\rm metric}
=2\tilde\gamma^{ij} Z_j,\qquad Z^i=\chi Q^i/2.
\]

Here the index on `Z^i` is raised with the physical metric, while Q uses the conformal inverse. Algebraic determinant/trace constraints are assumed enforced. Define the coordinate source product `sigma=alpha*kappa1`; current code uses

\[
\dot{\hat K}|_\kappa=(1-\kappa_2)\sigma\Theta,\quad
\dot\Theta|_\kappa=-(2+\kappa_2)\sigma\Theta,\quad
\dot\Gamma^i|_\kappa=-2\sigma Q^i.
\]

Consequently `dot K|kappa=-3(1+kappa2)sigma Theta`. These terms appear both in `BuildStandardPointwiseRHS` and the direct residual branch of `src/z4c/z4c_calcrhs.cpp`; forensic term recomputations have their own corresponding expressions. A profile change must modify all consistently, including full and background contributions.

There is a genuine printed convention discrepancy. [Z4c, arXiv:1111.2177](https://arxiv.org/pdf/1111.2177), Eq. 11 defines Q as above. Its Eq. 5 gives `dot Z_i|kappa=-alpha*kappa1*Z_i`, which transforms to `dot Q=-alpha*kappa1*Q`. Its Eq. 17 instead prints `-2alpha*kappa1*Q`, which matches this repository. [CCZ4, arXiv:1307.7391](https://arxiv.org/pdf/1307.7391), Eq. 6 and definition 11 give `-alpha*kappa1*Q`. This algebra identifies a factor-of-two inconsistency among the printed Z4c equations; it does not establish which publication expression was intended or that the repository's choice caused the measured instability. Keep the current Gamma coefficient in the first damping-profile experiment.

## A bounded lapse-adjusted source

Alic, Kastaun and Rezzolla propose replacing kappa1 by kappa1/alpha in Sec. III F, Eq. 27, motivated by loss of coordinate damping near collapsed lapse. They test it in CCZ4 black-hole evolutions. That is primary-source motivation, not a proof for the current residual Z4c discretization. Their modification introduces slicing dependence into damping; it does not make the damping four-dimensionally covariant.

Implementing the bounded product directly as **`sigma=kappa0`** avoids a numerical division by alpha. At positive alpha this is exactly that prescription. It remains finite as alpha tends to zero, and makes no change to the geometric or gauge principal part. Keep the existing lapse-validity checks; a nonpositive lapse is still a failure, not something this damping should repair.

Use the same source product in all three equations above, preserving kappa2 and the current Gamma factor. In residual mode subtract the background source with the same operator. On `Theta=Q=0` every new contribution vanishes, so the physical constraint surface and genuine matter sources are unchanged. For a background with discrete nonzero Q, preserve full-minus-background cancellation rather than assuming Q_bg is numerically zero.

This differs from `kappa0/alpha_bg`, whose full-state product is `kappa0*alpha_full/alpha_bg` and can become large for finite lapse residuals near the puncture. A smooth bounded denominator such as `sqrt(alpha_bg^2+epsilon^2)` would define another profile, but adds an unnecessary parameter for the first test. The direct constant product is simpler and bounded.

Suggested matched controls are the original uniform kappa1=0.1/M, then coordinate products sigma=0.1/M and 0.3/M, with every other physical/numerical option fixed. Both are safely nonstiff at dt=0.075M: the largest pure-damping eigenvalue has `|lambda|dt=0.015` or `0.045` for kappa2=0. Preserve or add the appropriate source timestep limit if much larger rates are later used. Validate exact-zero cancellation, analytical source differences, finite initial matter response, and diagnostic consistency before the 300M pulse. Extend only if the complete constraints and ghost diagnostics improve. A previous uniform kappa1=0.5/M failure does not duplicate this test because its coordinate product is still lapse-suppressed.

## Continuum linearized constraint subsystem on the present trumpet

This section derives constraint propagation from the repository's continuum equations; it is not a discrete eigenanalysis. All coefficients below are evaluated on the stationary `R0=M=1` background:

\[
R=r+1,\quad\alpha=r/R,\quad\chi=\alpha^2,\quad
\tilde\gamma_{ij}=\delta_{ij},\quad\beta^i=x^i/R^2,
\]
\[
K=1/R^2,\qquad K^i{}_j=(\delta^i{}_j-2n^in_j)/R^2.
\]

Let `H=R_phys+K^2-K_ij K^ij`, `M_i=D_j(K^j_i-delta^j_i K)`, and `D0=partial_t-beta^j partial_j` act on coordinate components. The Ricci tensor actually used by Z4c is

\[
R^{\rm code}_{ij}=R^{\rm phys}_{ij}+C_{ij},\qquad
C_{ij}=\tilde\gamma_{k(i}\partial_{j)}Q^k.
\]

On the conformally flat background, the linearized first two constraint equations are

\[
D_0\Theta=\frac\alpha2(H+\chi\partial_iQ^i)-(2+\kappa_2)\sigma\Theta,
\]
\[
D_0Q^i=2\alpha\delta^{ij}(M_j+\partial_j\Theta)-2\sigma Q^i.
\]

The second equation follows by subtracting the time derivative of the metric-defined conformal Gamma from evolved Gamma. The current code's nonadvective shift-gradient terms use metric Gamma, so they cancel in this difference. In covariant Z_i variables there are apparent shear/shift-gradient terms, but on this stationary, spatially constant conformal background they cancel back to the equation above.

For a compact expression of the remaining equations define the linear modification of the ADM K_ij equation

\[
T_{ij}=\alpha C_{ij}-(1+\kappa_2)\sigma\Theta\gamma_{ij}.
\]

Then

\[
D_0H=-2\alpha D_iM^i-4M^i\partial_i\alpha+2\alpha KH
+2(K\gamma^{ij}-K^{ij})T_{ij},
\]
\[
D_0M_i=M_j\partial_i\beta^j+\alpha KM_i
-\frac\alpha2\partial_iH-H\partial_i\alpha
+D_jT^j{}_i-D_iT^j{}_j.
\]

These variable-coefficient equations form the continuum linearized constraint subsystem under the stated algebraic constraints. The source part of the Hamiltonian addition is `-4(1+kappa2)sigma*K*Theta`; momentum includes `2(1+kappa2)partial_i(sigma*Theta)`. Thus changing a spatial damping profile changes both direct damping and its propagation couplings. Derivative terms and background coefficients must remain present in any spectral or radial model claiming actual mode growth.

Several scales explain why a flat frozen principal symbol is insufficient. `alpha*K=r/(1+r)^3` has maximum 4/27 per M. Hamiltonian propagation contains `+2alpha*K*H`, whose maximum is 8/27 per M. In the evolved-state Theta equation, holding Khat/A/geometry fixed gives an explicit `+(4/3)alpha*K*Theta` contribution, maximum 16/81≈0.19753 per M. Constant sigma=0.1 gives Theta damping 0.2/M, just larger than that one diagonal term everywhere; sigma=0.3 is stronger. Neither inequality bounds the full coupled system.

At the nearest dx=0.25M cell radius 0.216506M, alpha=0.177974 and K=0.675727/M. Uniform kappa1=0.1 gives Theta damping only 0.035595/M, while constant sigma=0.1 gives 0.2/M. At the nearest dx=0.125M radius 0.108253M, the uniform value falls to 0.019536/M. This makes a lapse-scaled control useful when comparing resolutions, without proving that lapse suppression caused the observed refinement trend.

For illustration only, dropping momentum and all gradients produces the **nonclosed** H–Theta block

\[
\begin{pmatrix}2\alpha K&-4\sigma K\\\alpha/2&-2\sigma\end{pmatrix},\qquad
\lambda_+=\alpha K-\sigma+\sqrt{(\alpha K)^2+\sigma^2}>0.
\]

This is not a valid global mode of the actual trumpet: H drives momentum through the lapse gradient, momentum feeds H, and advection/boundaries matter. It demonstrates only that suppressing the explicit Theta diagonal does not establish stability. [lower-order-scales.png](lower-order-scales.png) labels that toy separately; its curves are not predictions of measured growth rates.

## Minimal coupled CCZ4 source restoration

The difference between physical covariant Z derivatives and the already absorbed Ricci addition is

\[
E_{ij}=D_iZ_j+D_jZ_i-C_{ij}
=\frac12Q^k\partial_k\tilde\gamma_{ij}
+\frac{\tilde\gamma_{ik}Q^k\partial_j\chi+
\tilde\gamma_{jk}Q^k\partial_i\chi-
\tilde\gamma_{ij}Q^k\partial_k\chi}{2\chi}.
\]

With unit conformal determinant, `tr_gamma(E)=-Q^i partial_i chi/2`. This E contains no derivative of Q, so around a constraint-satisfying background it changes lower-order terms only. The independent script `audit.py` verifies the tensor identity against physical Christoffel derivatives for a nonconstant determinant-one metric, nonconstant chi, and nonconstant Q at three points to maximum error `3.9e-62` at 60 digits. It is not an AthenaK implementation test.

To match the geometric terms of fully covariant CCZ4 (`kappa3=1`) while retaining Khat as the evolved trace variable, the additions to the current equations are

\[
\Delta\dot{\hat K}=2Z^i\partial_i\alpha,
\]
\[
\Delta\dot\Theta=\frac\alpha2\operatorname{tr}_\gamma E
-\alpha K\Theta-Z^i\partial_i\alpha,
\]
\[
\Delta\dot{\tilde A}_{ij}=\alpha\chi E_{ij}^{\rm TF}
-2\alpha\Theta\tilde A_{ij},
\]
\[
\Delta\dot\Gamma^i=-2\tilde\gamma^{ij}\Theta\partial_j\alpha
-\frac23\alpha KQ^i+
\frac23Q^i\partial_j\beta^j-Q^j\partial_j\beta^i.
\]

The `-2alpha*K*Theta` contribution to full K cancels the corresponding term in twice Theta when forming Khat; it must **not** be added again to Khat. Conformal metric/chi equations are unchanged. For exact agreement with printed CCZ4 damping, Gamma damping must additionally change from `-2sigma Q` to `-sigma Q`; that is a separate parameter-convention change. Keeping the current factor while restoring the geometric sources is a clearly labeled hybrid, not the quoted CCZ4 equations. Every correction vanishes on `Theta=Q=0`, and each must be included in background subtraction and RHS audits before any evolution trial.

On this specific trumpet, `tr E=-Q·grad(chi)/2` makes the total new Theta-to-Q coupling `-alpha^2 Q·grad(alpha)`, in addition to `-alpha*K*Theta`. The covariant Gamma lower-order terms combine to a radial Q coefficient `+K-sigma` and a tangential coefficient `+(1-r)/(1+r)^3-sigma`, including CCZ4 damping. Near the dx=0.25M core, K≈0.676/M, so that radial coefficient is positive for sigma=0.1 or 0.3. These are coordinate-component local terms, not full eigenvalues. They show why adding only the attractive negative Theta term would misrepresent the formulation and why a full restoration is not guaranteed to help.

Recommendation: first measure the matched bounded sigma controls without changing the current formulation. If a coupled CCZ4 prototype is later attempted, implement and independently verify the entire set above, its damping convention, constraint-surface preservation, and source subtraction. Do not infer a cure from a local sign or from finite target completion.
