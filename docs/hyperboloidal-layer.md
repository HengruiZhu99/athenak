# Cauchy interior and hyperboloidal layer: implemented research prototype

Date: 2026-10-08 (America/New_York). Starting source:
`ca77b353a60a939c66227e42b02318c5cd32be9d`, `z4c_hyperboloidal`.
Implementation branch: `z4c_hyperboloidal_layer`.

**The reference geometry, gauge blend, harmonic source projection and native
Cartesian interior evolution are implemented. The nonlinear scri closure and
stability gates FAIL. This is not a regular evolution system at exact scri.**
The tests explicitly expose an unclosed Theta/trace limit, a positive
lower-order pole eigenvalue, and nonsymmetric embedded-boundary weights.
A finite-duration successful run is not an acceptance test for stability.
See [validation results](hyperboloidal-layer-validation.md).

This describes the first implementation (`d21fb74c` and its validation receipt).
The subsequent opt-in physical-trace lapse, symmetric ghost plans and derived
wormhole initial data are documented in
[the ongoing stabilization work](hyperboloidal-layer-stabilization.md).
The initial negative results remain reproducible; the follow-up has not yet
passed the nonlinear evolution acceptance gates.

## Storage, conventions and unchanged geometric equations

Signature is (-,+,+,+), `K_ij=-Lie_n(gamma_ij)/2`. Penrose compactification is
`gbar=Omega^2 gphysical`. `gtilde=chi*gammabar` has determinant one.
All tensor components and derivative indices are Cartesian. The flat Cartesian
reference connection is retained; there is no change of connection convention.

AthenaK's actual stored fields in this path are

| Storage | Meaning |
|---|---|
| `chi` | conformal spatial factor, not `Omega` |
| `g_dd` | `gtilde_ij`, determinant one |
| `vA_dd` | `chi*(Kbar_ij-Kbar*gammabar_ij/3)` |
| `vKhat` | `P=Kphysical-2*Thetaphysical` |
| `vTheta` | `Thetaphysical=Omega*Thetabar` |
| `vGam_u` | `Lambda^i=Gamma(gtilde)^i+2*ztilde^i` |
| `alpha`, `beta_u` | conformal lapse and shift |
| `vB_d` | unused and zero in the hyperboloidal adapter |

The physical-trace evolution in `conformal_rhs.hpp` is preserved, including
`C_Z4c=0`, `-3*omega_n*Thetaphysical/Omega`, and physical constraint damping.
No spherical stabilization formula has replaced a tensor equation. The kernel
argument is `kappa1=hyperboloidal_kappa1/alpha`; `kappa2=0` in the native adapter.
The unmodified geometric kernel supplies `R + S/Omega` for each evolved field.
There is no Omega floor, evolution outside scri, or assembly at `Omega=0`.

The physical ADM map inside the domain remains

```
alpha_phys = alpha/Omega
psi4_phys = 1/(Omega^2 chi)
gamma_physij = psi4_phys*gtildeij
K_physij = psi4_phys*(Omega*Atildeij + gtildeij*(P+2*Theta_phys)/3).
```

## One height function and its complete consumed reference jets

`layer_reference.hpp` evaluates an independent monotone smooth cutoff `w` on
`[r0,r1]`, with exact constant branches outside it. For `s=(r-r0)/(r1-r0)` its
logistic argument is `g=-1/s+1/(1-s)`. Evaluate the smaller exponential
`e=exp(-abs(g))`. The cutoff derivative prefactor is `e/(1+e)^2`, rather than
`w*(1-w)`, which would lose accuracy when `w` rounds to one. Endpoint branches
return all derivatives zero; an underflowed exponential returns zero derivatives
before forming products with large inverse powers. There is no numerical
endpoint differentiation. The high-precision compiled-code oracle checks
values and the first three derivatives.

For finite `S,a>0`, `0<r0<r1<S`, require `a>=S/2`. Then

```
Omega_out = (S-r)*(S+r)/(2*a*S)
Omega = (1-w)+w*Omega_out
b = r*w/a
L = Omega-r*Omega'
A = hypot(Omega,b)
R = r/Omega;   t = T-h(R);   h_R=b/A.
```

The sufficient positivity proof is
`L=(1-w)+w*(S^2+r^2)/(2*a*S)+r*w'*(1-Omega_out)>0`.
`Omega>0` for `r<S`; `A>0` through scri. Parameters and every active grid's
reference metric/positive factors are checked. This family does not provide
strong-field initial data merely by changing these parameters.

Write `d=L^2/A^2` and `n_i=x_i/r`. The exactly Cauchy origin branch avoids every
radial division and returns `Omega=alpha=chi=1`, `gtilde=I`, all other fields zero.
In the transition,

```
alpha_hat=A;    beta_hat^r=-b*A/L
gammabar_hat = I+(d-1)*n*n
kR=-b'/L;      kT=-b/(r*L);      Kbar_hat=kR+2*kT
P_hat=-(3*w+r*Omega*w'/L)/a
chi_hat=(A/L)^(2/3)
gtilde_hat = chi_hat*[I+(d-1)*n*n]
Atilde_hat = chi_hat*(kT-Kbar_hat/3)*I
             + chi_hat*[d*(kR-Kbar_hat/3)-(kT-Kbar_hat/3)]*n*n.
```

The physical trace expression is algebraically factored from
`Omega*Kbar_hat+3*b*Omega'/L`. Outside `r1` the code uses the exact old CMC branch:
`gammabar=I`, `chi=1`, `Atilde=Lambda=0`, `beta=-x/a`, `P=-3/a`.
Reference-only continuation beyond `S` is used for deviation ghosts.

For the connection, let `u=1/chi_hat`, `v=(1/d-1)/chi_hat`. Then

```
Lambda_hat^i = -[(u+v)' + 2*v/r]*n^i.
```

It is generally nonzero in the transition. The implementation obtains this
connection and its first derivatives from the analytic metric jets with the
existing `Geometry` contraction. Scalar, metric and shift derivatives through
second order, curvature and physical-trace derivatives through first order,
and connection derivatives through first order are supplied. These are exactly
the derivatives consumed by the tensor RHS. Unused second derivatives of
curvature, physical trace and connection are not represented as complete jets.

The radial projector derivatives used by `CartesianRadialJet` are

```
partial_j n_i = (delta_ij-n_i*n_j)/r
partial_k partial_j n_i =
  (-delta_ij*n_k-delta_ik*n_j-delta_jk*n_i+3*n_i*n_j*n_k)/r^2.
```

`L'=-r*Omega''`, `L''=-Omega''-r*Omega'''`. Thus third analytic cutoff derivatives
are essential even though the geometric kernel is second order in space.
Independent Cartesian finite differences check these jets in aligned and
oblique directions, both layer endpoints, and a small-Omega sequence.

## Implemented gauge and physical-trace assembly

`layer_gauge.hpp` defines
`u=log(alpha/alpha_hat)`, `v=log(chi/chi_hat)`, `D0=dt-beta.d`.
They are gauge definitions; the existing positive lapse and chi remain stored.
The independent gauge cutoff `W` is zero inside `gauge_r0`, one outside
`gauge_r1<S`. With `e=1-W`, `0<q0<1`,

```
f=1+2*e/alpha
alpha2f=alpha^2+2*e*alpha
mu=e*3*q0/4+W; epsilon_alpha=W; epsilon_chi=W/2
q=q0+(1-q0)*W
nu=nu_inner+(nu_outer-nu_inner)*W
eta=eta_inner+(eta_outer-eta_inner)*W.
```

All rates are finite, nonnegative and nondecreasing outward. The default
`q0=1/2` gives modified Bona–Massó slicing, `f=1+2/alpha`, in the interior.
Exact standard 1+log is not an option. The shift coefficient is also different
from the standard constant-coefficient moving-puncture driver.

Let `da=alpha-alpha_hat`, `db=beta-beta_hat`. Without reconstructing the full
`Q=(P-3*omega_n)/Omega`, form its deviation numerator

```
J = (P-P_hat)+3*Omega_i*(db^i-beta_hat^i*da/alpha_hat)/alpha
  = Omega*(Q-Q_hat),       Q_hat=Kbar_hat.

R_alpha = beta^i*alpha_i-alpha*beta^i*partial_i(log(alpha_hat))-alpha*nu*u
S_alpha = -alpha2f*J.
```

The stored-lapse RHS is `R_alpha+S_alpha/Omega` **only at Omega>0**. `log1p`
handles the small lapse ratio deviation. This is cancellation-aware interior
assembly, not a nonlinear scri regularization. The remaining quotient is
explicitly rejected at scri and audited below.

The baseline stored-shift equation is exactly

```
D0 beta^i = alpha^2*chi*[mu*(Lambda^i-Lambda_hat^i)
  + epsilon_chi*gtilde^{ij}*partial_j v
  - epsilon_alpha*gtilde^{ij}*partial_j u]
  - beta_hat^j*partial_j beta_hat^i - eta*db^i.
```

Centered analytic-plus-deviation jets supply these terms. The existing AthenaK
upwind correction replaces `beta.d` of each stored deviation; reference
advection and reference derivatives remain analytic. Fourth-order geometry and
mixed derivatives retain their original stencils. KO dissipation acts only
where its whole six-point axis neighborhood is strictly interior.

The physical-trace RHS has not been transformed into a new Q evolution. The
analytic Minkowski reference's floating-point geometric RHS is subtracted as in
the old adapter; no black-hole or live evolution RHS is subtracted. Raw stencil
stationarity converges at fourth order; reference restoration gives stationarity
near roundoff. These are different checks.

## Preferred-conformal source projection

In the harmonic collar, define spatial `ztilde^i=(Lambda^i-Gamma(gtilde)^i)/2`.
The spacetime Z4 components are

```
Zbar^0 = Theta_phys/(alpha*Omega)
Zbar^i = chi*ztilde^i-beta^i*Theta_phys/(alpha*Omega).
```

With the metric evolution's ADM identities,

```
F0=Gamma4^0+2*Zbar^0 = -D0(alpha)/alpha^3-Q/alpha
Fi+beta^i*F0 = chi*Lambda^i + (1/2)*gtilde^{ij}*chi_j
               -chi*gtilde^{ij}*partial_j(log(alpha))-D0(beta^i)/alpha^2.
```

At `W=1` the implemented lapse gives the algebraic, quotient-free source

```
F0=(beta.d(log(alpha_hat))+nu*u)/alpha^2-Q_hat/alpha
Fbase^i=chi*Lambda_hat^i
   +chi*gtilde^{ij}*[(1/2)*partial_j(log(chi_hat))-partial_j(log(alpha_hat))]
   +(beta_hat.d(beta_hat^i)+eta*db^i)/alpha^2-beta^i*F0.
```

Let

```
W_hat = 2*Omega'^2/L^2 + Omega*[Omega''/L^2
          +2*Omega'/(r*L^2)-Omega'*L'/L^3]
H4=(chi*gtilde^{ij}-beta^i*beta^j/alpha^2)*Omega_ij
V^i=delta^{ij}*Omega_j/(delta^{kl}*Omega_k*Omega_l)
Delta=H4-Omega*W_hat-Omega_i*Fbase^i.
```

The shift receives the algebraic correction `-W*alpha^2*V^i*Delta`.
This smoothly extends the source correction into the transition; the harmonic
source interpretation is exact only in the collar. It introduces no principal
couplings, preserves the reference, and enforces
`Omega_i*Fi=H4-Omega*W_hat` there. Require `gauge_r0>layer_r0` when projection is
on, so it never divides by a vanishing Cauchy-region gradient. The denominator
is Euclidean, never the null four-metric gradient norm.

Consequently

```
Box_bar(Omega) = Omega*W_hat + 2*Zbar^i*Omega_i
 = Omega*W_hat+chi*(Lambda-Gamma_tilde).dOmega
                  +2*omega_n*Theta_phys/Omega.
```

The last line matters off constraint: preferred source projection alone does
not enforce a regular Box limit for arbitrary Theta and spatial Z4 fields.
The live metric/lapse/shift source identity is checked independently of
reference stationarity. Its error is below 6e-16 in the tested collar states.

## Principal-symbol gate: complete, with explicit canceled eigenfields

Eliminate the determinant and trace-free algebraic constraints. Freeze a smooth
positive lapse/chi and a positive-definite spatial metric at Omega>0. Use a
spatial orthonormal frame and normalize by the frozen lapse and conformal
factors. The scalar reduced state is
`(d,c,h,k,t,A,L,B)`: lapse/chi logarithmic directional derivatives, longitudinal
metric derivative, `delta P/Omega`, `delta Theta_phys/Omega`, longitudinal
trace-free curvature, connection and shift derivative. Their precise frozen
normalization is implemented in `kernel_symbol.cpp`.

The actual extracted matrix is

```
M = [ 0     0     0     -f      0     0     0       0
      0     0     0      2/3    4/3   0     0      -2/3
      0     0     0      0      0    -2     0       4/3
     -1     0     0      0      0     0     0       0
      0     1     0      0      0     0     1/2     0
     -2/3   1/3  -1/2    0      0     0     2/3     0
      0     0     0     -4/3   -2/3   0     0       4/3
     -ea    ec    0      0      0     0     mu      0 ]
```

Its polynomial is `(lambda^2-f)*(lambda^2-1)^2*(lambda^2-q)`,
`q=(4*mu-2*ec)/3`. For each sign `sigma=+/-1`, the light and lapse left fields are

```
U_light1 = -sigma*c-sigma*h/2-2*k/3-4*t/3+A
U_light2 = 2*c+2*sigma*t+L
U_lapse = k-sigma*d/sqrt(f).
```

Set `F=2/alpha`, `lambda=sigma*sqrt(q)`. The analytically canceled shift field is

```
U_shift = B-lambda*d/(F+1-q0)+lambda*c/[2*(1-q0)]
  +(q0-W*F)*k/(F+1-q0)+q0*t/[2*(1-q0)]
  +lambda*(1-3*q0/4)*L/(1-q0).
```

The scalar left determinant is `-24*sqrt(q/f)`, exactly `-24` at `W=1`.
The dangerous uncanceled ratios contain `f-q=(1-W)*(F+1-q0)` and
`q-1=-(1-W)*(1-q0)`. They are never evaluated as numerical 0/0.
For each transverse block `(h_sA,A_sA,L_A,B_A)`,

```
V = [0 -2 0 1; -1/2 0 1/2 0; 0 0 0 1; 0 0 mu 0]
U_light = -sigma*h_sA/2+A_sA+sigma*L_A/2
U_shift = B_A+sigma*sqrt(mu)*L_A.
```

Its determinant is `-2*sqrt(mu)` in the implemented row ordering and its
polynomial is `(lambda^2-1)*(lambda^2-mu)`. Each of two tensor polarizations has
matrix `[0 -2; -1/2 0]` and fields `A_T-sigma*h_T/2`. Counts are 8 scalar,
2x4 vector and 2x2 tensor = 20. `f,q,mu>0`, `q0<1`, `F+1-q0>0` give a complete
basis for this family. No uniform conditioning through alpha=0, extreme metric
anisotropy, q0=0 or q0=1 is claimed.

For covector `n_i`, coordinate speeds are
`-beta.n +/- alpha*sqrt(gammabar^{ij}n_i n_j)*{sqrt(f),sqrt(q),sqrt(mu),1}`.
The new native timestep uses the maximum live axis speed
`abs(beta^i)+sqrt(alpha2f*chi*gtilde^{ii})` and retains the separate interior
pole timestep bound. This is conservative timestep control, not stability.
The cancellation-free reference radial speeds are

```
cplus=A*(A+b)/L
cminus=-A*Omega^2/[L*(A+b)].
```

At scri they are `2*S/a` and zero. The old simple Gamma endpoint
`f=1,mu=3/4,ea=ec=0` has only three eigenvectors at each multiplicity-four scalar
light root. The new endpoint is complete. The compiled gate checks nonunit
lapse/chi, oblique frames, sheared determinant-one positive metrics, and exact
harmonic coefficients against the tensor kernel, not just the proposed gauge.
This proof is in weighted interior variables; it does not regularize their
nonlinear equations or establish masked finite-difference stability.

## Exact nonlinear closure obstruction

Compatible variables must close all residues simultaneously. The regular
reference identities are

```
1-h_R^2=Omega^2/A^2;     gbar4^{rr}=Omega^2/L^2
N_hat=(|D Omega|^2-omega_n^2)/Omega^2=Omega'^2/L^2
S_hat,rr=Omega''/L+Omega'^2/A^2
S_hat,tangent=Omega*Omega'/(r*L^2)
Box_hat(Omega)/Omega=W_hat.
```

They supply reference limits only. No evolution equations for live
`N=(|D Omega|^2-omega_n^2)/Omega^2`,
`Sij=(DiDjOmega+omega_n*Kbarij)/Omega` have been closed by these definitions.
There is no assumption that A, metric or connection deviations are O(Omega).

The exact trace time transformation is

```
dt P = Omega*dt Q-3*Omega_i*dt beta^i/alpha-3*omega_n*dt alpha/alpha.
```

Its mass matrix determinant is Omega. Write the unchanged kernel as
`dtP=RP+SP/Omega`, `dtalpha=Ra+Sa/Omega`, `dtbeta=Rb`. Then

```
dtQ = E2/Omega^2+E1/Omega
E2=SP+3*omega_n*Sa/alpha
E1=RP+3*Omega_i*Rb^i/alpha+3*omega_n*Ra/alpha.
```

These combined numerators need a compatible expansion/constraint closure.
Independently requiring finite Q, N or the gauge spectrum is insufficient.

For `Theta_phys=Omega*tau`, `P=Omega*Q+3*omega_n`, the existing Theta equation is

```
dt Theta_phys = alpha*[2*D^2Omega+2*omega_n*Q-kappa1*(2+kappa2)*tau]
 + Omega*{alpha*[(Rbar-|A|^2+2*chi*div(ztilde))/2
                  -3*N+(Q+2*tau)^2/3]+beta.d(tau)}.
```

Here the `beta.d(Omega*tau)` term has been expanded; it cancels one normal-Theta
term. Thus preserving `Theta_phys=Omega*tau` requires the first bracket vanish
at scri, with a compatible limiting evolution for tau. Setting tau zero by fiat
would not close the other sectors and is not implemented.

Concrete check: take outer flat CMC reference spatial/gauge fields, Theta=0,
replace `P` by `P_hat+Omega*deltaQ` and its gradients consistently. The gauge
source correction is unchanged for this perturbation. This data is smooth and
null but violates the full asymptotic shear/constraint compatibility. Direct
symbolic and compiled-kernel evaluation give

```
Omega*dtQ -> 2*deltaQ;       dtTheta_phys -> -2*deltaQ.
```

For deltaQ=.01 the compiled sequence down to Omega~=1e-8 approaches .02 and -.02.
This explicitly disproves the interpretation of this interior gauge as a closed
regular scri formulation on arbitrary smooth finite-Q states.

There is a separate lower-order result. At `S=a=1`, exact reference scri,
`kappa2=0`, freeze all perturbation spatial derivatives and extract the leading
pole Jacobian in `(delta alpha,delta chi,delta P,delta Theta_phys)`:

```
Jpole = [ 3   0  -1    0
         -2   0   2/3  4/3
          0  -3  -2    kappa1-4
          0  -3  -2   -1-2*kappa1 ].
```

This is a closed block of the leading pole: the remaining shift, determinant-one
metric, trace-free A and connection have no pole forcing from this subspace.
Its characteristic polynomial is
`lambda*(lambda^3+2*kappa1*lambda^2-9*lambda-12*kappa1)`.
For kappa1=5 its positive root is **2.57170948731154**. For every kappa1>0 there
is a positive root between sqrt(6) and 3; at kappa1=0 it is 3. Finite restoring
rates cannot change this leading 1/Omega coefficient. The C++ finite-difference
Jacobian and exact symbolic polynomial agree. This is an off-manifold local
lower-order growth obstruction, distinct from a global PDE or numerical spectrum
proof. It explains why a complete principal basis is inadequate as acceptance
and why simply increasing this damping coefficient is not a justified repair.

## Scri null compatibility and Cartesian boundary status

With outward spatial unit covector `s_i`, the live scri condition is
`beta^i*s_i=-alpha`. At W=1 its normal characteristic speeds become 0 and 2alpha.
The reference satisfies it; arbitrary evolved gauge values do not.
For fixed Omega define `C=chi*gtilde^{ij}Omega_i Omega_j-(beta.dOmega)^2/alpha^2`.
Its exact preservation condition is

```
dtC = [dtchi*gtilde^{ij}-chi*gtilde^{ik}*dtgtilde_kl*gtilde^{lj}]*Omega_i*Omega_j
      -2*(beta.dOmega)*(dtbeta.dOmega)/alpha^2
      +2*(beta.dOmega)^2*dtalpha/alpha^3.
```

A regular closure must make this O(Omega^2) and enforce compatible shear/Z4/
Hamiltonian limits. The present interior equations and extrapolation have not
established that property. No independent lapse/shift Dirichlet condition is
imposed at a characteristic scri surface.

The original normal-ray ghost policy is retained for an auditable comparison.
Every donor and transverse rectangle is strictly interior, with no recursive
use of ghosts. Its plan covers mixed derivatives of radius 2 and upwind axis
stencils of radius 3. The interior-only KO policy needs no exterior donor.
Existing polynomial, poisoned-exterior, coverage, transport and derivative
convergence tests pass. New weight audits nevertheless expose large amplification
and non-invariance under reflection/axis permutation. These defects are reported
as failed acceptance gates, not hidden behind polynomial consistency.

A boundary-fitted outer shell with Cartesian tensor components would remove
Cartesian cut-cell interpolation and permit a radial SBP/SAT analysis. The
existing separate radial driver is useful infrastructure, but would need this
same layer/reference and a proven nonlinear closure. An outer patch cannot
remove the positive continuum pole block by discretization alone. We therefore
do not replace the working CMC boundary with an unvalidated alternative here.
Closing regular null/shear and Z4 residues comes before such a patch's nonlinear
stability claim.

## Runtime restrictions, inputs and next gates

`inputs/z4c/hyperboloidal_layer.athinput` opts into this research path with
`hyperboloidal_layer=true`. The layer's geometry and gauge weights are independent.
Defaults r0=.35/r1=.75, gauge_r0=.45/gauge_r1=.85 are vacuum audit choices,
not binary recommendations. The geometric transition spans only about 4.6
coarse N=24 cells and the reference radial metric reaches roughly 15.8; it is
not a gentle resolved strong-field layer. Width/damping/resolution controls are
saved by `run_layer_validation.py`; reference-only manufactured refinement
uses much smaller spacing than these inexpensive live grids.

The actual native path rejects single precision, MPI builds, non-Serial execution, matter,
AMR/multilevel, multiple blocks, unsupported ghost/stencil/integrator choices,
floors, trackers and extraction. Layer trumpet data are rejected before use:
old CMC black-hole data are not transformed to the new height function.
No black-hole fixed-point subtraction exists. GPU, MPI/AMR and binary tests
have not passed the required preceding gates and are not enabled.

Next mathematical work is a coupled closure/evolution for the residues and
constraint falloffs above, preserving the physical-trace stabilization or
justifying its replacement. It must control the extracted off-manifold pole
block and prove live scri null/shear compatibility. Next discrete work must
address the measured ghost anisotropy/amplification and demonstrate bulk and
boundary convergence. Neither an Omega floor nor an unexamined vanishing gauge
factor is an acceptable substitute.

## Relation to primary literature

Height-function references and nonlinear spherical hyperboloidal layers are
published in [Vañó-Viñuales and Valente](https://arxiv.org/html/2408.08952v2).
Our coefficient blend and the particular Cartesian prototype are derived here.
The physical-trace stabilization and the distinction between C_Z4c=0 and full
non-principal Z4 terms are described by
[Vañó-Viñuales, Husa and Hilditch](https://arxiv.org/html/1412.3827v2), especially
sections 7.2–7.3 and Appendix B. We retain the inspected tensor convention;
their spherical stabilization is not silently substituted.
[Hilditch et al.](https://arxiv.org/pdf/1111.2177) give the harmonic shift structure
and a discrete analysis for their particular discretization, not our sphere mask.
[Bona and Palenzuela](https://arxiv.org/pdf/gr-qc/0401019) analyze coupled gauge
characteristics. [Zenginoğlu](https://arxiv.org/pdf/0808.0810) motivates preferred
conformal gauge under smooth conformal-extension assumptions. Our source identity
implements one part of that condition and does not establish those assumptions.

## Subsequent private gauge work

The [physical reference wave-map local audit](hyperboloidal-reference-wave-map-local-audit.md)
records a private alternative gauge, complete higher reference derivatives,
independent high-precision readbacks, and the retained failed coordinate controls.
Its local source/dual checks pass; production equations and the unresolved
regularity/boundary claims above are unchanged. Its global harmonic core is a
diagnostic choice, with no moving-puncture blend selected.

The subsequent [wave-map consistency audit](hyperboloidal-wave-map-consistency-audit.md)
records passing actual constrained20 principal, linear-core and finite-amplitude
nonradial exact-flat RHS gates. These local results do not establish native
evolution stability or resolve the regularity and boundary limitations above.

The [native preflight audit](hyperboloidal-reference-wave-map-native-audit.md)
records the compiled array seam and eight passing stationary-reference/short
angular-pulse controls with independent binary64 field checks. In the subsequent
fixed t2 matrix, the N16 wave-map control fails its positive-state guard near
t=1.367. The subsequent [failed-pulse audit](hyperboloidal-reference-wave-map-failure-audit.md)
records failures of matched C0 N16 and wave-map N24, all 167 saved partial
observations, and a separate conditional stationary black-hole source analysis.
The other original controls continue; longer acceptance remains open.

The later acceptance target is a substantial angular gauge disturbance on
Minkowski followed by a single black hole surviving the inner
wormhole-to-trumpet transition. The Minkowski hyperboloidal reference must be
retained throughout; consistent physical black-hole initial foliation and gauge
adjustment must be derived without a black-hole fixed-point subtraction.
