# Regular advective reduction extension (2026-09-04)

Status: **implemented research candidate; CUDA evolution qualification pending**.
This follows the user's explicit choice to retain the regular variables and moving
puncture gauge while allowing a separately derived PC-GH extension. It supersedes
neither the failed literal FO-GH pullback audit nor the saved numerical evidence.
It is not the Lindblom gamma1=-1 system. The distinctions and counterexamples in
[the gamma2 audit](pc_gh_gamma2_audit.md) remain valid.

## Complete specification relative to the existing equations

Use the full regular equations and H/E/S/T/m definitions in
[the production derivation](pc_gh_derivation.md#2026-09-02-puncture-regular-55-field-representation-superseding-production-map),
with every change listed below. This additive specification fixes every one of the
55 evolution rows; omitted rows receive exactly zero increment. Use
`D0 = partial_t - beta^k partial_k`, `alpha=rho*w`, `chi=w^2`, and denote the conformal
metric by g. All tensor contractions below use g, whose determinant is one.
Define the *true* reduction residuals

\[
r_i=p_i-\partial_iw,\quad q_{iab}=Q_{iab}-\partial_i g_{ab},\quad
\ell_i=L_i-2\partial_i(\rho w),\quad b_i{}^a=B_i{}^a-\partial_i\beta^a.
\]

The existing diagnostic is \(a_i=R^\alpha_i=\ell_i-2\rho r_i\).
The new input is `reduction_system=advective`, `reduction_rate=lambda >= 0`.
The defaults remain `legacy` and zero; a nonzero rate in legacy mode is rejected.
The rate is constant in the present implementation, has dimension inverse length,
and is independent of GH damping `kappa`. No physical length is inferred from the
mesh spacing. In units with c=G=1, configuration variables are dimensionless,
first derivatives/K/Atilde/Z/Cperp have dimension inverse length, and each new RHS
term has the dimension of its evolved variable divided by length.

**Configuration rows:** replace stored-gradient advection by true advection:

\[
\Delta w_t=-\beta^i r_i,\qquad
\Delta g_{ab,t}=-\beta^i q_{iab},\qquad
\Delta\beta^a_t=-\beta^i b_i{}^a,\qquad \Delta\rho_t=0.
\]

In particular `D0 alpha=-2 alpha K` still holds exactly for the moving puncture
lapse, including off the reduction manifold. The switched moving shift remains

\[
D_0\beta^a=F_\beta^a=\Gamma^a(Q)-Z^a-\eta\beta^a+S(z)V^a,
\quad z=\alpha w^2,\quad
V^a=g^{ac}(\alpha^2 w p_c-\tfrac12\alpha w^2 L_c).
\]

S is the existing smoothstep. For the new switched mode its transition must finish
by z=1/2. The parser enforces this. The proof below concerns this gauge, not an
unrestricted direct/harmonic/A0 gauge theorem.

**First derivative rows:**

\[
\begin{split}
\Delta p_{i,t}&=-\lambda r_i,\\
\Delta L_{i,t}&=-\lambda\ell_i,\\
\Delta B_i{}^a{}_{,t}&=-\lambda b_i{}^a,\\
\Delta Q_{iab,t}&=-\lambda q_{iab}
 -\frac{2\alpha}{3}g_{ab}\widetilde A^{cd}q_{icd}.
\end{split}
\]

The last term follows from differentiating `tr_g Atilde=0` intrinsically when the
metric and Q jets are independent. Its trace cancels the mismatch
`2 alpha Atilde^{cd} q_icd` in the differentiated algebraic constraint. It is
necessary on general Atilde backgrounds and vanishes on the reduction manifold.

**Curvature/GH rows:** add to the old S tensor before taking its trace-free part

\[
\Delta S_{ab}=\tfrac12\rho w^2(\partial_b p_a-\partial_a p_b)
-\tfrac14w^2(\partial_b L_a-\partial_a L_b).
\]

Thus S uses symmetrized p and L Hessians. Although this expression is antisymmetric,
adding it to the old unsymmetrized Hessian produces a symmetric total S. Also add

\[
\Delta Z^a_t=g^{ac}\sum_d(\partial_d B_c{}^d-\partial_c B_d{}^d).
\]

These are curl completions, essential for a tensorial and complete principal
system away from the curl manifold. There are no further K, Cperp, Atilde, or Z
increments. In particular there are no gamma2-like coupled curvature *damping*
sources in this candidate. A preliminary coupled prototype was tested during the
derivation; it is not the implemented option.

Every new coefficient is polynomial in w/rho/alpha and the bounded evolved fields,
using only the inverse of the positive conformal metric. There is no division by
w, rho, alpha, or chi in the new production terms. The code evaluates
`ell=a+2*rho*r` with actual finite differences, avoiding a discrete product-rule
assumption. Finite differences need not obey the continuum product rule; its
truncation error is included in the numerical qualification, not projected away.

## Exact nonlinear reduction and curl subsidiary system

A compact component formula specifies *all* subsidiary equations, including the
couplings of the moving gauge. Let

\[
x^A=(w,g_{ab},\alpha,\beta^a),\quad G_i^A=(p_i,Q_{iab},L_i/2,B_i{}^a),
\quad E_i^A=G_i^A-\partial_i x^A.
\]

The nonadvective configuration sources are

\[
F_w=w(\alpha K-B)/3,\quad F_\alpha=-2\alpha K,\quad
F_{g,ab}=-2\alpha\widetilde A_{ab}+g_{ka}B_b{}^k+g_{kb}B_a{}^k-2g_{ab}B/3,
\]

and F_beta above. In derivatives `F^A_,x^D`, hold all G/K/Atilde/Z/Cperp variables
fixed. This prescription is important: derivative-variable dependencies cancel
between the configuration derivative and stored gradient equations. Define

\[
N_{i,g_{ab}}=-2\alpha g_{ab}\widetilde A^{cd}q_{icd}/3,
\quad N_i^A=0\text{ in the other rows}.
\]

Then direct differentiation gives the complete nonlinear system

\[
\boxed{\partial_t E_i^A=\beta^k\partial_k E_i^A
 +(\partial_i\beta^k)E_k^A+b_i{}^kG_k^A
 +F^A{}_{,x^D}E_i^D-\lambda E_i^A+N_i^A.}
\]

All coefficients can be evaluated using the displayed polynomial F without any
field division. This is a closed homogeneous first order system in the reductions
on a given evolving background; it includes nonlinear products of reductions.
As explicit checks in the code's regular diagnostic variables,

\[
\begin{split}
\partial_t r_i={}&\beta^k\partial_k r_i+B_i{}^k r_k+b_i{}^k\partial_kw
 +(2\alpha K-B)r_i/3+(wK/6)a_i-\lambda r_i,\\
\partial_t a_i={}&\beta^k\partial_k a_i+B_i{}^k a_k+2w b_i{}^k\partial_k\rho
 -(2+\alpha/3)K a_i-(2\rho\alpha K/3)r_i-\lambda a_i.
\end{split}
\]

Let `omega^A=dG^A=dE^A`, with
`omega_ij=partial_i G_j-partial_j G_i`, and
`J_i^A=b_i^k G_k^A+F^A_,x^D E_i^D+N_i^A`. Exterior differentiation gives

\[
\boxed{\partial_t\omega^A=\mathcal L_\beta\omega^A+dJ^A
 -\lambda\omega^A-d\lambda\wedge E^A.}
\]

The L curl is twice the alpha component. This formula includes spatially varying
lambda for a future taper, although the current input is constant. It explicitly
shows the curl source from any future smooth relaxation mask. No hard reset or
mask is included in the present candidate.

The reduction characteristics have coordinate speed `-beta.n`. On shifted flat
space with constant coefficients they decay as `exp(-lambda*t)`; curls do too.
On varying backgrounds the source matrix also stretches and mixes errors. A
positive lambda is not a theorem of monotonically decreasing constraint norms on
an arbitrary spacetime: the symmetric part of the remaining source and the
transport energy terms must be bounded/dominated. The characteristic degeneracy
of physical/lapse modes at the puncture does not force lambda to vanish.

### Nonflat isotropic initial-slice check

On the requested wormhole initial slice, `rho=1`, `alpha=w`, `L=2p`,
`g=identity`, and `beta=B=K=Atilde=Q=0`. The frozen linear reduction source
system, for each derivative index i, is exactly

\[
\dot a_i=-\lambda a_i,\quad
\dot b_i{}^k=-(\lambda+\eta)b_i{}^k+\tfrac12 S w^2p^k a_i,\quad
\dot r_i=-\lambda r_i+p_k b_i{}^k,\quad
\dot q_{iab}=-\lambda q_{iab}.
\]

Its ten independent eigenvalues per derivative index are -lambda (seven) and
-lambda-eta (three). Repeated source eigenvalues can produce polynomial transients
multiplying `exp(-lambda*t)`; this is not necessarily monotone decay in every
component. The exact isotropic profile `w=[r/(r+M/2)]^2` gives `p=O(r)`,
`w^2 p=O(r^5)`, and `|p| <= 16/(27M)`. Thus the complete source coefficients are
bounded at the puncture on this initial slice. `wormhole_subsidiary.py` checks the
shift differentiation, characteristic polynomial, and limits independently.
This is stronger than a flat test but remains a frozen initial-slice result; the
wormhole is not stationary in moving puncture gauge.

## Einstein consistency and GH propagation

All changes are homogeneous in reduction/curl residuals. On their invariant zero
manifold the existing first-principles four-dimensional Ricci oracle therefore
continues to test exactly the same equations. No use of the Einstein constraints
is required for this equality. An additional compiled polynomial-jet oracle has
verified all 55 old/new RHS rows are identical, including nonzero Cperp/Z/K, with
an exactly differentiable affine w/rho/beta and quadratic alpha.

For signature (-+++), the covariant reduced equation on this manifold is

\[
R_{ab}-\nabla_{(a}{\cal C}_{b)}+
\kappa[n_{(a}{\cal C}_{b)}-\tfrac12g_{ab}n^c{\cal C}_c]=0.
\]

Trace reversing and applying the contracted Bianchi identity, rather than
assuming an independent constraint equation, yields

\[
\boxed{\square {\cal C}_b+R_b{}^a{\cal C}_a
 -2\nabla^a[\kappa n_{(a}{\cal C}_{b)}]=0.}
\]

Consequently data satisfying the GH and Einstein initial constraints solve vacuum
Einstein evolution as long as the reduced initial-value problem is well posed.
For flat space and constant positive kappa this reduces to

\[
{\cal C}_{i,tt}-\Delta {\cal C}_i+\kappa{\cal C}_{i,t}=0,\quad
{\cal C}_{0,tt}-\Delta {\cal C}_0+2\kappa{\cal C}_{0,t}
 -\kappa\partial_i{\cal C}_i=0.
\]

Homogeneous zero-frequency GH modes need not decay; kappa is not a proof that every
possible GH violation decays strictly. On a variable background curvature and
coefficient gradients also enter.

Off the reduction manifold define the exact residual of this covariant equation
as `J_ab`. Bianchi then gives precisely the same boxed wave operator with forcing
`-2 nabla^a(J_ab - g_ab J/2)`. Reconstructing the metric jets from the evolved
configuration and using the above subsidiary system makes J homogeneous in E,
curls, and their derivatives on r>0. Thus the combined zero surface is invariant;
reduction error can source GH error. A fully expanded nonlinear component formula
for this off-reduction J has not been independently checked. The independent full
production Fourier test does check this coupling in the shifted Minkowski
linearization: all 30 independent reductions satisfy the exact operator identity
`C(k)(J_source+i*k*P)=(-lambda+i*k*beta_x)C(k)`, and the entire 50-dimensional
algebraic tangent spectrum has no eigenvalue with positive real part for any real
wave number. In the comoving spectral variable z its exact polynomial is
`(z+lambda)^30 (z^2+2k^2)(z^2+k^2)^5(z^2+z+k^2)^3(z^2+2z+k^2)`
for kappa=1 and eta=0. This was independently recovered from the compiled source
Jacobian and the exact reduction-manifold embedding. Zero-frequency neutral
modes need not be strictly damped or free of polynomial gauge transients.
This is a flat linear damping result, not a nonlinear black-hole estimate.

## Principal symbol, characteristic fields, and domain

Use the 50-dimensional tangent of `det(g)=1`, `tr_g Atilde=0`, `tr_g Q_i=0`.
For a unit covector in the conformal metric the nonadvective speeds squared are

\[
c_j^2=\{1,\ 2\rho w^3,\ \rho^2w^4,\ (4-S\rho^2w^4)/3\}.
\]

The exact characteristic polynomial is

\[
v^{30}(v^2-1)^2(v^2-2\rho w^3)(v^2-\rho^2w^4)^6
[v^2-(4-S\rho^2w^4)/3].
\]

The system is written `u_t=P(n) partial_n u`, so coordinate characteristic speeds
are `-beta.n` and `-beta.n +/- c_j`. A general covector multiplies c_j by its
conformal norm. The proof uses the production tensor ordering, not just the old
legacy matrix. In an orthonormal frame the general symbol is

\[
P=P_0+D=(I-T)P_0(I+T),\quad T^2=0,\quad P_0T=D,\quad TP_0=DT=0.
\]

D contains the rate and general algebraically allowed Atilde/Q tangent
contributions. A polynomial group inverse constructs
`T=h(P0)D`, with
`q(v)=product_j(v^2-c_j^2)`, `h(v)=[1-q(v)/q(0)]/v`.
Its only physical-domain denominator is
`q(0)=2 rho^3 w^7 (4-S rho^2 w^4)/3`; speed coincidences introduce no shear poles.
The symbolic script verifies all identities with symbolic Atilde/Q, w/rho/S/rate.
Constant shift adds a multiple of the identity. Tensor covariance extends the
orthonormal result to SPD conformal metrics and arbitrary directions; twelve
compiled affine-jet probes check axes, an oblique direction, non-diagonal metrics,
nonzero shift, and nonzero curvature/gradient backgrounds.

For distinct positive speeds, write

\[
\Pi_0=\prod_j\frac{P_0^2-c_j^2I}{-c_j^2},\qquad
\Pi_j=\frac{P_0^2}{c_j^2}\prod_{l\ne j}
\frac{P_0^2-c_l^2I}{c_j^2-c_l^2}.
\]

Characteristic fields are the independent components of `Pi0(I+T)u` and
`(I +/- P0/c_j) Pi_j(I+T)u/2`. At coincidences combine the coincident subspaces.
Exact ranks check semisimplicity at z=1/2, z=4/(6+alpha), and alpha^2 chi=1.
Crucially, an additional symbolic cancellation test shows that the continued
S=1 projectors have denominator factors only `alpha`, `chi`, and `alpha-2`:
there are no poles at these speed crossings. This supplies bounded local
characteristic projectors, beyond merely checking ranks at isolated parameters.
In the switch transition the speeds are distinct in the domain below.

The resulting strong-hyperbolicity domain is

\[
g>0,\quad w>0,\quad\rho>0,\quad0<\alpha<2,\quad
0<\alpha^2\chi<4,\quad0<z_0<z_1\le1/2.
\]

This is local on compact subsets of the stated state domain. The shear/basis can
become unbounded as w or rho tends to zero. **No uniform symmetrizer or well-posedness
theorem at the compactified puncture point is claimed.** Finiteness of production
coefficients is a separate established property. Resolution-dependent puncture
powers, exterior convergence, and comparison with Z4c are still required.

## Implementation and evidence

The source option is backward compatible. Explicit time steps additionally obey
`dtnew <= 1/lambda` before the mesh multiplies by CFL. This is a conservative
source-step bound, not an IMEX integrator or a nonlinear stability guarantee.
Primary qualification inputs disable reduction projection and keep fixed KO/CFL.

`reduction_monitor=true` records every pre-RHS and post-stage reduction/curl
maximum, with cycle, step-start time, dt, RK stage, operation, phase, location,
refinement level, block ID, and MPI rank. No puncture/exterior mask enters these
maxima. Additional rows record alpha and alpha^2 chi maxima and their locations
to detect departure from the proved domain. Nonfinite values cause a fatal diagnostic. Transfer operation IDs are:
0 restriction, 1 prolongation, 2 algebraic projection, 3 reduction projection,
4/5/6 post-projection restriction/exchange/prolongation, 7 ordinary exchange,
8 physical boundary fill. Negative IDs are snapshots: -1 pre-RHS, -2 post-stage,
-3 before adaptive regrid and -4 after regrid plus initialized ghosts.

Fixed hierarchy operations are individually bracketed. Dynamic regridding
currently has a net before/after bracket, not separate interior redistribution,
refinement, and derefinement samples with invalid intermediate mesh metadata.
No face-integrated constraint flux diagnostic has yet been added. Stage times
are reported as the step origin plus the stage label, not a fabricated physical
RK intermediate time. Intermediate ghost-fill samples can contain unsynchronized
ghosts; compare their labelled before/after differences and the synchronized
pre-RHS/post-stage samples separately.

Local evidence in `qualification-runs-20260904/regular-extension/` includes:

- `symbol-general-background.log`: exact similarity and intrinsic trace proof.
- `hyperbolicity-extended.log` and `projector-limits.log`: spectrum, coincidence
  ranks, and absence of spectral-projector poles at the accepted crossings.
- `subsidiary-final.log`: nonlinear reductions, curls/rate gradients, intrinsic Q.
- `production-symbol-background.log`: 12 compiled principal matrices, maximum
  discrepancy <=4.45e-16; `production-invariance.log`: exactly zero RHS difference
  on the independent reduction-manifold jet.
- `production-final-zero-step/fourier-exact.log`: exact reduction operator identity,
  all 30 independent reductions, and an exact all-real-k stable spectral polynomial;
  sampled eigenvalues provide an additional floating-point cross-check.
- `wormhole-subsidiary-final.log`: negative frozen reduction-source eigenvalues
  and bounded coefficients on the nonflat isotropic initial slice.
- `full-symbolic.log`: the existing symbolic/first-principles regression suite.

These are AppleClang/Kokkos Serial zero-step oracles and symbolic calculations.
They are **not CUDA evolution results**. Remote access reached
`hz0693@della-vis1.princeton.edu` but was denied authentication; no Della job has
been launched for this candidate. None of the puncture/AMR/binary gates is passed
by these derivations. The saved hard-Q correction and old 73.7999M metric-positivity
failure remain relevant controls. No result here attributes that failure to
projection or claims a successful puncture scheme.

The CUDA fixture collection is generated by `make_inputs.py`; use the latest
`inputs-v2` collection in the evidence directory. An exploratory one-dimensional
SMR initialization fixture exited with SIGSEGV before evolution; its raw logs
are retained under `pulse-initial-oracles/Q-amr`. The AMR qualification fixture
was changed to a three-dimensional compact pulse using the puncture transfer
path. This is a fixture limitation, not a successful AMR evolution or a diagnosed
continuum instability.
