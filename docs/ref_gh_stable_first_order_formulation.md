# Stable first-order reference-GH puncture formulation

## Scope and final decision

This is a theory/static-code audit of `HengruiZhu99/athenak`, branch
`codex/ref-gh-relative-damped-single-hole-20260830`, starting from commit
`0182bc2e00d115e6e1e9d2eba996b1b2f155b308`.  No new AthenaK run is claimed.

The central requirement is **not** relaxed: the production formulation must be able to start from ordinary Bowen--York wormhole puncture data in isotropic-like coordinates and dynamically approach the moving-puncture/trumpet gauge without excision.

The formulation I recommend is:

> **Dynamical-reference covariant-residual STANDARD GH (DRCR-GH):** keep the STANDARD `gamma1=-1` first-order GH principal system, but evolve the metric deviation from a prescribed reference and its background-covariant first derivatives.  In each puncture core the prescribed reference itself follows a regular wormhole-to-trumpet Schwarzschild gauge trajectory.  Use pure wave-map gauge throughout the core and strong reference-transition buffer; activate the existing algebraic relative-damped correction only outside that buffer through a `C^infinity` exact plateau.  Keep finite `gamma0,gamma2`.  Abandon a time-dependent singular exponent `q(t)`, but **do not** abandon dynamical wormhole-to-trumpet evolution.

The key distinction is:

- **rejected:** change the puncture power at finite time by writing a factor `r^{-q(t)}`;
- **retained:** let the coefficient of the wormhole branch recede to zero while the reference shift transports the compactified second end inward, so every fixed resolved `r>0` becomes trumpet-like although the exact puncture retains the wormhole branch at every finite coordinate time.

This is precisely the nonuniform limit needed for the usual moving-puncture topology change.

---

# 1. What is already mathematically correct

## 1.1 STANDARD `Phi` ordering

The current helper `src/ref_gh/phi_ordering.hpp` defines the anholonomic-frame curl constraint

```math
C_{IJAB}=E_I\Phi_{JAB}-E_J\Phi_{IAB}-c^K{}_{IJ}\Phi_{KAB}
```

and changes the compatible equation according to

```math
(\partial_t\Phi_{IAB})_{\rm standard}
=(\partial_t\Phi_{IAB})_{\rm compatible}-\beta^J C_{IJAB}.
```

This is the correct STANDARD ordering for raw frame derivatives.  The commutator term is required because `[E_I,E_J]=c^K{}_{IJ}E_K`.

**Classification: PROVED CORRECT.**

## 1.2 `gamma2` reduction damping

For `gamma1=-1`, the code adds the standard reduction-constraint terms

```math
\delta(\partial_t\Pi_{AB})=-\gamma_2\beta^i C_{iAB},
\qquad
\delta(\partial_t\Phi_{IAB})=
\alpha\gamma_2\bar e_I{}^i C_{iAB}.
```

**Classification: PROVED CORRECT.**

## 1.3 Principal symbol, characteristics, and symmetrizer

For one symmetric spacetime component and a unit spatial covector `s_I`,

```math
G^{IJ}s_Is_J=1,
```

the standard characteristic fields are

```math
u^0=\Psi,
\qquad
u^\pm=\Pi\pm s^I\Phi_I-\gamma_2\Psi,
\qquad
u_I^\perp=\Phi_I-s_I s^J\Phi_J,
```

with speeds

```math
0,
\qquad -\beta^s+\alpha,
\qquad -\beta^s-\alpha,
\qquad -\beta^s.
```

For the one-direction derivative matrix

```math
B=
\begin{pmatrix}
0&0&0\\
-\gamma_2\beta&\beta&-\alpha\\
\alpha\gamma_2&-\alpha&\beta
\end{pmatrix},
```

the matrix

```math
S=
\begin{pmatrix}
\Lambda^2&-\gamma_2&0\\
-\gamma_2&1&0\\
0&0&1
\end{pmatrix}
```

satisfies exactly

```math
SB=(SB)^T.
```

This was checked symbolically.  Positivity follows from

```math
P^2-2\gamma_2hP+\Lambda^2h^2
=(P-\gamma_2h)^2+(\Lambda^2-\gamma_2^2)h^2,
```

so `Lambda^2>gamma2^2` is sufficient.  For all ten tensor components use a positive tensor-component metric and the physical positive spatial matrix `G^{IJ}`.

**Classification: PROVED CORRECT.**

The time dependence of a prescribed reference does not alter this conclusion: it changes `S_t` and lower-order coefficients in the energy estimate, but not the principal symbol.

---

# 2. Why the fixed-reference growth is compatible with symmetric hyperbolicity

For

```math
\partial_t u+A^i(x,t)\partial_i u=B(x,t)u,
\qquad SA^i=(SA^i)^T,
```

the energy estimate contains

```math
\frac{dE}{dt}
\lesssim
\int u^T
\left[
S B+B^T S+\partial_t S+\partial_i(SA^i)
\right]u\,d^3x
+\text{boundary flux}.
```

Therefore symmetric hyperbolicity constrains the principal part but does **not** imply that the lower-order matrix is dissipative.

The observed fixed-reference mode can therefore be genuine.  In the present raw-frame variables the potentially dangerous lower-order families are:

1. spatial-frame gradients and frame commutators in the `Phi` system;
2. spin-connection terms used to reconstruct background-covariant derivatives;
3. reference curvature and spin-derivative terms in the nonlinear source;
4. derivatives of the algebraic gauge with respect to the relative metric;
5. gradients of the finite-radius reference and gauge windows;
6. `S_t` and reference-frame-motion terms when the reference evolves.

The existing observation that a fixed reference can grow GH, reduction, and curl constraints together is therefore **not** evidence against the STANDARD principal symbol.

It does mean that a new reference trajectory must be qualified by the **full lower-order energy matrix**, not only by characteristic speeds.

---

# 3. Use deviation variables and background-covariant first derivatives

Let `bar g_ab` be the prescribed reference metric with orthonormal frame `bar e_A^a` and coframe `bar theta^A_a`.  Define

```math
\Psi_{AB}=\bar e_A{}^a\bar e_B{}^b g_{ab},
\qquad
h_{AB}=\Psi_{AB}-\eta_{AB}.
```

The current code evolves raw frame derivatives and then reconstructs the background-covariant derivative by subtracting spin-connection terms.  For production punctures it is cleaner to make the covariant derivative fundamental:

```math
P_{AB}:=-n^a\bar\nabla_a h_{AB},
\qquad
Q_{IAB}:=\bar e_I{}^a\bar\nabla_a h_{AB}.
```

In frame components,

```math
Q_{IAB}
=E_I h_{AB}
-\bar\omega^C{}_{AI}h_{CB}
-\bar\omega^C{}_{BI}h_{AC}.
```

Since `bar nabla eta=0`, the exact matched state is

```math
\boxed{h_{AB}=0,\qquad P_{AB}=0,\qquad Q_{IAB}=0.}
```

The physical/reference connection difference is then formed directly:

```math
\Delta_{ABC}
=\frac12\left(Q_{BAC}+Q_{CAB}-Q_{ABC}\right),
```

with the first index raised by the physical relative inverse metric when needed.

This removes the numerically unattractive operation

```text
raw derivative of Psi  -  spin connection times Psi
```

from the definition of the main nonlinear connection-difference field.

### Important limitation

This variable change is **not claimed to eliminate a genuine continuum lower-order instability by itself**.  Away from the puncture it is an invertible local rewrite of the same equations.  Its purposes are:

- make the exact matched state zero;
- eliminate large exact-state subtractions;
- expose the geometric lower-order matrices cleanly;
- make the puncture regularity gate a statement about finite residual fields and finite frame-native coefficients.

The actual lower-order stability still has to be established for the chosen dynamical reference and gauge buffer.

---

# 4. First-order equations in the residual variables

At principal level the STANDARD system is

```math
\partial_t h_{AB}
\simeq-\alpha P_{AB}+\beta^I Q_{IAB},
```

```math
\partial_t P_{AB}
\simeq
\beta^I E_I P_{AB}
-\alpha G^{IJ}E_IQ_{JAB}
-\gamma_2\beta^I\mathcal C_{IAB},
```

```math
\partial_t Q_{IAB}
\simeq
\beta^J E_JQ_{IAB}
-\alpha E_IP_{AB}
+\alpha\gamma_2\mathcal C_{IAB}.
```

The full equations are obtained by writing the reduced Einstein equation with the background derivative `bar nabla` and the connection difference `Delta`.  A convenient exact identity is

```math
R_{ab}
=\bar R_{ab}
+\bar\nabla_c\Delta^c{}_{ab}
-\bar\nabla_b\Delta^c{}_{ac}
+\Delta^c{}_{cd}\Delta^d{}_{ab}
-\Delta^c{}_{bd}\Delta^d{}_{ac}.
```

The reduced equation is

```math
R_{ab}-\nabla_{(a}C_{b)}
+\frac{\gamma_0}{2}
\left(2n_{(a}C_{b)}-g_{ab}n^cC_c\right)=0.
```

When the reference is vacuum and the gauge increment vanishes at match, every lower-order source is at least linear in `(h,P,Q)`, so the prescribed single-hole transition is an exact zero-residual solution.

The positive symmetrizer can be written directly for `(h,P,Q)`:

```math
\mathcal E=
 m^{AC}m^{BD}
\left[
\Lambda^2h_{AB}h_{CD}
+P_{AB}P_{CD}
-2\gamma_2h_{AB}P_{CD}
+G^{IJ}Q_{IAB}Q_{JCD}
\right],
```

where `m^{AB}` is any smooth positive-definite metric on tensor components (for example the usual positive metric constructed from the physical normal).  If `Lambda^2>gamma2^2` and `G^{IJ}>0`, this is positive definite and symmetrizes the STANDARD principal matrices.

**Symmetric hyperbolicity of the proposed main system: PROVED.**

---

# 5. Subsidiary constraints in a non-coordinate reference frame

Define the background-covariant reduction constraint

```math
\mathcal C_{IAB}
:=\bar\nabla_I h_{AB}-Q_{IAB}.
```

The corresponding integrability constraint is

```math
\begin{aligned}
\mathcal C_{IJAB}:={}&
\bar\nabla_IQ_{JAB}-\bar\nabla_JQ_{IAB}
-c^K{}_{IJ}Q_{KAB}\\
&+\bar R^C{}_{A IJ}h_{CB}
+\bar R^C{}_{B IJ}h_{AC}.
\end{aligned}
```

The curvature terms are required by the commutator of background covariant derivatives.  With this definition, `C_IJAB=0` is an identity whenever `Q_IAB=bar nabla_I h_AB`.

The STANDARD `Q` equation is most safely derived as

```text
covariant compatible derivative of the h equation
+ gamma2 reduction damping
- beta^J times the covariant curl constraint.
```

This guarantees that reference-frame-motion and curvature pieces are assembled analytically into the constraint identities rather than appearing as unmatched product-rule terms.

At principal level,

```math
\partial_t\mathcal C_{IAB}
\simeq
-\alpha\gamma_2\mathcal C_{IAB}
+\beta^J\mathcal C_{IJAB},
```

with all reference connection, curvature, lapse/shift-gradient, and gauge-gradient terms lower order.

For the GH constraint use the wave-map base source

```math
B_a=-g_{ab}g^{cd}\bar\Gamma^b{}_{cd},
\qquad H_a=B_a+J_a.
```

The regular identity is

```math
\boxed{C_A=J_A+\Delta_A.}
```

Production should evaluate this residual identity directly.  It should not separately reconstruct a large coordinate `H_a` and a large coordinate `Gamma_a` and then subtract them.

From the contracted Bianchi identity, the exact nonlinear GH subsidiary equation for the damping convention used here is

```math
\boxed{
\nabla^b\nabla_b C_a
+R_a{}^b C_b
-2\gamma_0\nabla^b\!\left(n_{(a}C_{b)}\right)=0.
}
```

Thus the continuum GH constraint system is homogeneous.  Any reference dependence enters through coefficients, not through an inhomogeneous source, provided the equations are assembled consistently.

---

# 6. The relative-damped gauge is retained, but moved out of the strong transition region

Use

```math
\boxed{
H_a=B_a+\bar\theta^A{}_a J_A,
\qquad
J_A=W D_A,
}
```

with

```math
D_A=\mu_L L_RN_A^R-\frac{\mu_S}{a_R}V_A^R,
```

where the relative ADM-like quantities are algebraic functions of

```math
\Psi_{AB}=\eta_{AB}+h_{AB}.
```

At exact match,

```math
a_R=1,
\qquad L_R=0,
\qquad V_A^R=0,
```

so `D_A=J_A=0` exactly.

The flat-background first variation is

```math
D_0=-\frac{\mu_L}{2}(h_{00}+h_{ii}),
\qquad
D_i=-\mu_S h_{0i}.
```

For pure-gauge plane waves, the transverse modes obey

```math
s^2+\mu_S s+k^2=0,
```

and the scalar determinant factorizes exactly as

```math
\boxed{
(s^2+k^2)
\left[s^2+(\mu_L+\mu_S)s+k^2+\mu_L\mu_S\right].
}
```

This factorization was checked symbolically.  Positive `mu_L,mu_S` therefore produce no growing frozen matched-state gauge root; one scalar harmonic wave remains undamped.

The gauge is algebraic in the state and prescribed reference, so it does not change the GH principal symbol.

### Recommended window placement

Do **not** turn on relative damping inside the spatial region where the wormhole-to-trumpet reference has its strongest frame/connection gradients.  Use

```math
W=0
```

through the puncture core **and** a surrounding transition buffer.  Activate `W` only after the reference lower-order coefficient norms have become moderate.  The width/placement is to be chosen by a static energy/Jacobian gate, not by hiding an unstable mode with damping tuning.

The old live `Hhat/theta/Upsilon` 1+log/Gamma-driver path should remain disabled in puncture production.  Standard moving-puncture gauge may be used to **generate the prescribed reference trajectory offline**; it should not be reintroduced as a live coupled GH driver.

---

# 7. Why `q(t)` is the wrong way to make the reference dynamical

The present controlled exponent contains schematically

```math
L_q(r,t)\propto r^{-q(t)}.
```

Therefore

```math
\boxed{
\partial_t\ln L_q=-\dot q\ln r.
}
```

Higher derivatives generate

```math
\ddot q\ln r,
\qquad
\dot q^2(\ln r)^2,
\qquad
\frac{\dot q}{r}.
```

These are unbounded coefficient functions at the puncture whenever the exponent is changing.  This is not cured by a smoother time trajectory for `q` and must not be left to cancellation in floating point.

**Decision: abandon dynamic exponent tracking, not dynamic gauge transition.**

---

# 8. Regular dynamical Bowen--York wormhole -> trumpet reference

## 8.1 Initial and final powers

For ordinary Bowen--York wormhole data,

```math
\psi_{\rm BY}
=\frac{M}{2r}+u_0+O(r),
```

where the Bowen--York correction is finite.  The isotropic spatial coframe scale

```math
L:=\psi^2
```

therefore has

```math
L_{\rm W}
=\frac{A_0}{r^2}+\frac{B_0(\Omega)}{r}+O(1),
\qquad
A_0=\frac{M^2}{4}.
```

For a stationary trumpet with finite limiting areal radius `R_0`,

```math
L_{\rm T}=\frac{R_0}{r}+O(1).
```

The powers differ, but they do **not** have to be changed by a finite-time exponent parameter.

## 8.2 Receding wormhole coefficient

Use a reference whose local spatial coframe has the finite-time asymptotic form

```math
\boxed{
\bar L(t,r)
=\frac{A(t)}{r^2}
+\frac{B(t,r)}{r}
+O(1),
\qquad A(t)>0\quad\text{for every finite }t,
}
```

with

```math
A(t)\longrightarrow0
\qquad (t\longrightarrow\infty).
```

The crossover radius is

```math
r_c(t)\sim \frac{A(t)}{B(t,0)}.
```

Hence:

- for every finite `t`, the exact `r->0` asymptotic remains wormhole-like;
- `r_c(t)` moves inward;
- for every fixed `r>0`, eventually `r >> r_c(t)` and the `1/r` trumpet term dominates;
- at finite numerical resolution, the entire resolved domain becomes trumpet-like once `r_c` falls below the innermost resolved radius.

This is the desired nonuniform wormhole-to-trumpet limit.

## 8.3 Necessary leading shift transport law

Let the reference shift near the puncture be

```math
\bar\beta^i=b(t)x^i+O(r^3),
```

and let the finite-time wormhole lapse retain the usual collapsed behavior

```math
\bar\alpha=a(t)r^2+O(r^3).
```

Using

```math
\bar\gamma_{ij}\sim A(t)^2 r^{-4}\delta_{ij},
```

the leading orthonormal extrinsic curvature is

```math
\bar K_{IJ}
= -\frac{1}{\bar\alpha}
\left(\frac{\dot A}{A}+b\right)\delta_{IJ}
+\text{subleading terms}.
```

Therefore finiteness requires the exact leading cancellation

```math
\boxed{\dot A+bA=0.}
```

This is **not** a cancellation of separately divergent numerical quantities.  It is a transport equation for the reference asymptotic coefficient itself.  A regular reference provider should solve/enforce this relation analytically in its asymptotic representation.

If `b>0`,

```math
A(t)=A(0)\exp\!\left[-\int_0^t b(t')dt'\right],
```

so the wormhole coefficient decays without ever changing the puncture power at finite time.

Higher terms in `B`, the lapse, and the shift series must satisfy analogous bounded-frame conditions.  They should be obtained from a consistent reference geometry rather than by independent interpolation.

## 8.4 Preferred reference: an exact Schwarzschild gauge-transition trajectory

For the single-hole core, the cleanest construction is a spherically symmetric Schwarzschild spacetime written in a prescribed time-dependent coordinate map that:

1. starts from the desired isotropic wormhole slice and initial lapse convention (for example the usual pre-collapsed lapse) with zero initial shift;
2. follows a regular moving-puncture-type coordinate evolution;
3. approaches the stationary 1+log trumpet;
4. obeys the coefficient-recession asymptotics above.

This reference can be generated offline by a one-dimensional high-accuracy gauge evolution or directly as a coordinate map on exact Schwarzschild.  Because it is a diffeomorphic representation of Schwarzschild at every finite time,

```math
\bar R_{ab}=0.
```

At exact match,

```math
h=P=Q=0,
\qquad
J_A=0,
\qquad
C_A=0,
```

for the **entire dynamical transition**.  Thus the desired large coordinate change is represented as a zero-residual solution instead of as a large live gauge forcing.

This is the strongest available single-hole regression oracle.

An arbitrary smooth interpolation between wormhole and trumpet metric profiles is inferior: even if smooth, it is generally non-vacuum, produces extra curvature forcing, and can create large finite-radius lower-order matrices.

---

# 9. Reference coefficients must be regular before numerical evaluation

A finite geometric coefficient is unacceptable if production obtains it by subtracting two separately divergent coordinate quantities.

The dynamical reference provider should return frame-native quantities such as

```math
\bar\omega^A{}_{BC},
\qquad
\bar R^A{}_{BCD},
\qquad
\bar\nabla_D\bar\omega^A{}_{BC},
```

directly from a regular asymptotic representation / coordinate map / orthonormal ADM data.

Required qualification bounds are of the form

```math
|\bar\omega|\le C_1/M,
\qquad
|\bar R|\le C_2/M^2,
```

with corresponding finite bounds for every derivative entering the lower-order equations.

For the physical relative fields require

```math
G^{IJ}=O(1),
\qquad
(G^{-1})_{IJ}=O(1),
```

and for the gauge

```math
J_A=O(1),
\qquad
\frac{\partial J_A}{\partial h_{BC}}=O(M^{-1})
```

in the puncture limit.

These are **formulation gates**, not runtime floors.

---

# 10. `C^infinity` finite-radius plateau

For the outer relative-gauge window and any finite-radius reference stitching, replace the present quintic exact plateau by a `C^infinity` exact plateau.

Define

```math
\rho(x)=
\begin{cases}
0,&x\le0,\\
\exp(-1/x),&x>0,
\end{cases}
```

and

```math
S(x)=\frac{\rho(x)}{\rho(x)+\rho(1-x)}.
```

Then `S=0` for `x<=0`, `S=1` for `x>=1`, and all derivatives vanish at both endpoints.

For `0<x<1`, define

```math
L(x)=-\frac1x+\frac1{1-x},
\qquad
S=(1+e^{-L})^{-1}.
```

Then

```math
S'=S(1-S)L',
\qquad
L'=\frac1{x^2}+\frac1{(1-x)^2},
```

and

```math
S''=S(1-S)\left[(1-2S)(L')^2+L''\right],
\qquad
L''=-\frac2{x^3}+\frac2{(1-x)^3}.
```

For physical transition width `Delta`, derivatives scale as `Delta^{-1}` and `Delta^{-2}`.  Therefore `C^infinity` smoothness alone is insufficient: do not let the physical width collapse with resolution or with the shrinking wormhole core.

---

# 11. GH-consistent Bowen--York first-order initial data

The direct fixed-reference experiment identified the correct mathematical issue: preserving a complete coordinate first jet while changing the gauge source generally violates `C_a=0`.

The correct construction preserves the physical Cauchy data `(gamma_ij,K_ij)` and uses the four GH constraints to determine the four gauge components of the metric velocity.

## 11.1 Six physical components

Choose initial lapse `alpha` and shift `beta^i`, then form

```math
g_{00}=-\alpha^2+\gamma_{ij}\beta^i\beta^j,
\qquad
g_{0i}=\gamma_{ij}\beta^j,
\qquad
g_{ij}=\gamma_{ij}.
```

The ADM identity

```math
\boxed{
\partial_t\gamma_{ij}
=-2\alpha K_{ij}+\mathcal L_\beta\gamma_{ij}
}
```

fixes six independent components of the metric time derivative.

## 11.2 Four gauge components

The remaining four combinations may be represented by `(partial_t alpha, partial_t beta^i)`.  The exact 3+1 GH identities are

```math
(\partial_t-\beta^i\partial_i)\alpha
=-\alpha^2\left(K+n^aH_a\right),
```

and

```math
\partial_t\beta^i
=\alpha^2\gamma^{ij}
\left[
H_j-\partial_j\ln\alpha+{}^{(3)}\Gamma_j
\right],
```

where

```math
{}^{(3)}\Gamma_j
:=\gamma^{kl}\,{}^{(3)}\Gamma_{jkl}.
```

These identities prove that the four GH conditions determine the four gauge velocities whenever the spatial metric is positive definite and `alpha>0`.

However, near a puncture the production implementation should **not** evaluate a singular coordinate wave-map source `B_a` just to use these formulas.  Instead solve the equivalent four equations

```math
\boxed{C_A=J_A+\Delta_A=0}
```

directly in frame-native residual variables for the four remaining combinations of `P_AB`.  Together with the six combinations fixed by `K_ij`, this gives all ten independent components of `P_AB`.

This is the preferred implementation because `J_A` and `Delta_A` are regular residual quantities.

---

# 12. Static lower-order stability qualification

Symmetric hyperbolicity is proved, but the existing evidence shows that this is not enough.  For each frozen reference state construct the main-system energy-growth matrix

```math
\boxed{
\mathcal K
=S^{-1/2}
\left[
SB+B^TS+\partial_tS+\partial_i(SA^i)
\right]
S^{-1/2}.
}
```

Its largest eigenvalue bounds local energy growth in the chosen symmetrizer norm.

Do the same for the full subsidiary constraint state

```math
(C_A,\mathcal C_{IAB},\mathcal C_{IJAB}).
```

A production reference trajectory must pass at least the following static gates:

- all principal eigenvalues real and characteristic basis complete;
- symmetrizer positive with controlled condition number;
- all frame-native lower-order coefficients bounded;
- no large positive localized eigenvalue of the subsidiary energy-growth matrix in the reference-transition region;
- the relative-gauge Jacobian does not reintroduce the old fast inner coupled mode;
- the exact Schwarzschild transition gives zero residual analytically.

If a candidate trajectory fails the lower-order gate, change the **reference trajectory / spatial transition geometry / gauge buffer**, not the damping parameters merely to suppress the symptom.

---

# 13. Why this is genuinely a dynamical moving-puncture formulation

The reference is not fixed to a trumpet from `t=0`.

At `t=0` it has the same wormhole singular class as ordinary Bowen--York data.  Its shift initially may be zero.  As the prescribed moving-puncture reference shift develops, the wormhole coefficient `A(t)` is transported according to the regularity law `dot A + b A = 0`; the crossover radius shrinks; the resolved geometry becomes trumpet-like; and the limiting reference approaches the stationary 1+log trumpet.

The live GH system only evolves **deviations** from this gauge trajectory.  Thus the failed strategy

```text
live GH gauge driver tries to manufacture 1+log/Gamma-driver motion
```

is replaced by

```text
prescribed regular singular gauge trajectory
+ symmetric-hyperbolic GH evolution of deviations
+ algebraic relative damping outside the strong transition region.
```

This is the central recommendation.

---

# 14. Spinning punctures and generic binaries

For Bowen--York momentum or spin, the leading wormhole conformal factor remains `M/(2r)` plus a finite correction.  The leading singular spatial coframe power is therefore still `r^{-2}`.  The spherical single-hole reference can carry the universal singular gauge sector while momentum, spin, tidal fields, and nonspherical corrections remain finite residuals.

For a binary:

1. give each puncture its own local wormhole-to-trumpet reference core and local transition history;
2. move the centers with a smooth control map;
3. stitch to an outer binary frame only at finite radius using `C^infinity` windows of fixed physical width;
4. keep both local singular cores after common-horizon formation;
5. transition only the smooth outer reference toward a remnant-centered frame.

No abrupt two-puncture -> one-puncture singular transformation is required.

The binary extension is **plausible but unproved** and requires later numerical qualification.

---

# 15. Concrete AthenaK changes

## Retain

- STANDARD ordering and `gamma1=-1`;
- finite nonnegative `gamma2` reduction damping;
- finite positive `gamma0` GH damping;
- the background-covariant connection-difference architecture in `covariant_gh_source.hpp`;
- the algebraic relative-damped gauge idea;
- reference-provider abstraction and analytic/oracle backends.

## Deprecate for puncture production

- dynamic `reference_q_controlled` exponent tracking;
- `q_relaxed_controller` as a singular-exponent controller;
- the live `Hhat/theta/Upsilon` moving-puncture driver in the puncture path;
- any source assembled from separately divergent physical/reference pieces;
- `C^2` exact-plateau windows in high-order production paths.

## Change

1. Replace stored `Psi` by `h=Psi-eta` (same ten slots).
2. Replace raw `Pi/Phi` by background-covariant `P/Q` (same 10+30 slots).
3. Rewrite the nonlinear source directly in `(h,P,Q,Delta)`.
4. Evaluate `C_A=J_A+Delta_A` directly.
5. Derive the STANDARD `Q` equation by the covariant compatibility identity plus the covariant curl constraint.
6. Add a `wormhole_trumpet_transition` reference provider.
7. Make that provider expose a consistent time-dependent two-jet **and** regular frame-native spin/curvature data.
8. Encode the finite-time wormhole coefficient `A(t)` and its leading shift transport relation analytically rather than through `q(t)`.
9. Keep `W=0` through the strong reference-transition buffer; use a `C^infinity` outer plateau.
10. Add a frame-native GH-consistent initial-data solve for the four gauge components of `P_AB`.
11. Add frozen-coefficient main/subsidiary energy-matrix diagnostics as static qualification tests.

No production evolution equation should be changed before these identities are independently reproduced by unit/oracle tests.

---

# 16. Minimal later numerical test sequence

No result is claimed here; these are future gates.

1. **Symbolic/algebra gate:** coordinate STANDARD GH vs covariant-residual equations on smooth manufactured references.
2. **Zero-residual dynamical-reference gate:** nonspinning Schwarzschild, with the full wormhole-to-trumpet reference trajectory and `h=P=Q=0`; RHS must vanish to reference-table/roundoff accuracy throughout the transition.
3. **Static lower-order gate:** scan the main and subsidiary energy-growth matrices along that trajectory, especially the shrinking crossover region.
4. **Perturbation gate:** add small constraint perturbations and verify the predicted frozen-coefficient behavior.
5. **Bowen--York Schwarzschild gate:** use the GH-consistent initial-data construction and evolve the full transition.
6. **Resolution gate:** demonstrate convergence of GH, reduction, and curl constraints without retuning gauge/damping parameters.
7. **Boosted and spinning Bowen--York punctures.**
8. **Two Bowen--York punctures through inspiral and common-horizon formation while retaining two local cores.**

---

# 17. Answers to the ten requested questions

1. **Is the current STANDARD Ref-GH continuum system mathematically correct?**  The STANDARD principal part, curl correction, characteristics, and `gamma2` structure are correct.  The background-covariant source architecture is geometrically correct in form.  I do not claim a line-by-line formal proof of every handwritten contraction.
2. **Can a fixed spatially varying reference generate reduction/curl growth without violating symmetric hyperbolicity?**  Yes.  The lower-order energy matrix can have a positive symmetric part.
3. **Is the current relative-damped gauge fundamentally sound?**  Its matched state and flat frozen-coefficient sign structure are sound.  Retain it, but keep it out of the strong puncture/reference-transition buffer and use covariant residual derivatives.
4. **Is dynamic `q(t)` viable?**  No as a production singular-exponent controller; `qdot log r` is unavoidable.
5. **Cleanest wormhole -> trumpet representation?**  A dynamical reference with a receding wormhole coefficient `A(t)`, not a changing power.  Prefer an exact Schwarzschild moving-puncture gauge trajectory.
6. **What gauge should be used?**  Pure wave-map to that trajectory in the core/transition buffer, plus algebraic relative damping farther out.  Do not use the old live moving-puncture GH driver in the core.
7. **Positive symmetrizer?**  Yes; the STANDARD GH symmetrizer applies directly to `(h,P,Q)` for `Lambda^2>gamma2^2` and `G^{IJ}>0`.
8. **Are puncture coefficients controlled?**  They can be, provided the reference satisfies the frame-native bounds and the amplitude/shift asymptotic transport laws.  This remains a required static qualification, not an assumed fact for an arbitrary interpolated reference.
9. **How are GH-consistent Bowen--York data constructed?**  Preserve `(gamma,K)`, use those data for six metric-velocity combinations, and solve the four regular equations `C_A=J_A+Delta_A=0` for the remaining four components of `P_AB`.
10. **Can it extend to spin and BBHs?**  Plausibly: the universal local singular powers are unchanged by finite Bowen--York momentum/spin, and two persistent local cores avoid a merger-time singular remap.  Numerical proof remains future work.

---

# 18. Claim status

## PROVED

- STANDARD `gamma1=-1` principal symbol and characteristic speeds.
- Positive symmetrizer for `Lambda^2>gamma2^2`.
- STANDARD non-coordinate curl correction and the audited `gamma2` signs.
- The exact GH subsidiary wave equation quoted above for the chosen damping convention.
- The regular wave-map identity `C_A=J_A+Delta_A`.
- The flat matched-state relative-gauge factorization.
- The logarithmic obstruction for a changing exponent `q(t)`.
- The leading wormhole-coefficient/shift regularity condition `dot A+bA=0`.
- Four GH constraints determine the four gauge metric-velocity components once `(gamma,K)` are fixed.

## SUPPORTED BY EXISTING NUMERICAL EVIDENCE

- the old live moving-puncture GH driver is structurally unsafe in the current puncture realization;
- reference motion amplifies the instability;
- fixed spatial reference structure can support a common GH/reduction/curl lower-order growth mode;
- the present finite-radius transition region is the relevant place to inspect lower-order matrices.

## PLAUSIBLE BUT UNPROVED

- the exact Schwarzschild gauge-transition reference has sufficiently mild frame-native lower-order matrices for robust production evolution;
- covariant residual variables materially improve finite-precision conditioning of the puncture source;
- the buffered relative gauge robustly controls spinning/boosted deviations;
- two local dynamical cores remain benign through binary merger.

## REQUIRES FUTURE NUMERICAL TESTING

- all production robustness claims beyond the principal/symbolic results;
- the reference-table/map representation and interpolation accuracy;
- the physical width/location of the outer relative-gauge plateau;
- boosted, spinning, and binary qualification.

---

# 19. Reproducible symbolic identities

Two exact algebra checks used above are simple to reproduce in SymPy.

For the STANDARD one-direction principal block,

```python
B = Matrix([[0,0,0],
            [-g2*beta,beta,-alpha],
            [alpha*g2,-alpha,beta]])
S = Matrix([[Lam**2,-g2,0],
            [-g2,1,0],
            [0,0,1]])
simplify(S*B - (S*B).T)  # identically zero
```

For the scalar relative-gauge plane-wave block,

```python
A = s**2 + k**2
M = Matrix([[A + muL*s, I*muL*k],
            [I*muS*k, A + muS*s]])
factor(M.det())
```

gives exactly

```math
(s^2+k^2)
\left[s^2+(\mu_L+\mu_S)s+k^2+\mu_L\mu_S\right].
```

The wormhole-coefficient result follows directly from the leading isotropic metric

```math
\gamma_{ij}=A(t)^2r^{-4}\delta_{ij},
\qquad
\beta^i=b(t)x^i,
```

for which

```math
\partial_t\gamma_{ij}=2\frac{\dot A}{A}\gamma_{ij},
\qquad
(\mathcal L_\beta\gamma)_{ij}=-2b\gamma_{ij}.
```

Thus

```math
K_{IJ}\sim
-\bar\alpha^{-1}
\left(\frac{\dot A}{A}+b\right)\delta_{IJ},
```

and bounded finite-time wormhole lapse `bar alpha ~ r^2` requires `dot A+bA=0` at leading order.

## References

- L. Lindblom, M. A. Scheel, L. E. Kidder, R. Owen, O. Rinne, *A New Generalized Harmonic Evolution System*, arXiv:gr-qc/0512093.
- L. Lindblom, B. Szilagyi, *An Improved Gauge Driver for the Generalized Harmonic Einstein System*, arXiv:0904.4873.
- M. A. Scheel et al., *Solving Einstein's Equations With Dual Coordinate Frames*, arXiv:gr-qc/0607056, Phys. Rev. D 74, 104006.
- B. Szilagyi, L. Lindblom, M. A. Scheel, *Simulations of Binary Black Hole Mergers Using Spectral Methods*, arXiv:0909.3557.
- M. Hannam et al., *Geometry and Regularity of Moving Punctures*, arXiv:gr-qc/0606099.
- M. Hannam et al., *Where do moving punctures go?*, arXiv:gr-qc/0612097.
- M. Hannam et al., *Wormholes and trumpets: Schwarzschild spacetime for the moving-puncture generation*, arXiv:0804.0628.
- M. Hannam, S. Husa, N. O Murchadha, *Bowen--York trumpet data and black-hole simulations*, arXiv:0908.1063.
