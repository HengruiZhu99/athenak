# Stable first-order reference-GH puncture formulation

## Scope and decision

This note audits the reference-frame first-order generalized-harmonic (Ref-GH) system on branch `codex/ref-gh-relative-damped-single-hole-20260830` at commit `0182bc2e00d115e6e1e9d2eba996b1b2f155b308`.  It is a theory/static-code audit only.  No new AthenaK evolution result is asserted here.

The central requirement is retained in full: the formulation must start from ordinary Bowen--York wormhole puncture data in isotropic-like coordinates and dynamically relax toward a moving-puncture/trumpet gauge, without excision.

The recommended formulation is:

> **STANDARD `gamma1=-1` first-order GH, rewritten in reference-deviation variables and background-covariant first derivatives, with an algebraic wave-map + relative-damped gauge and a genuine dynamical wormhole-to-trumpet reference trajectory.**

The important change from the current design is that the dynamical reference is **not** represented by a time-dependent puncture exponent `q(t)`.  A finite-time change of a factor `r^{-q(t)}` necessarily generates `qdot log r` and related singular terms.  Instead, the exact puncture remains on the wormhole branch at every finite coordinate time while a transition region moves inward and the solution approaches a trumpet at every fixed `r>0`.  This is the continuum behavior that should be represented by the reference.  The cleanest implementation is a one-dimensional Schwarzschild gauge-transition reference generated from the same isotropic wormhole initial slice and approaching the stationary trumpet.

This is not a proposal to start with trumpet initial data.  Bowen--York wormhole initial data remain the intended initial data.

---

## 1. Static audit of the current continuum system

### 1.1 STANDARD ordering

The source file `src/ref_gh/phi_ordering.hpp` implements

\[
C_{IJAB}=E_I\Phi_{JAB}-E_J\Phi_{IAB}-c^K{}_{IJ}\Phi_{KAB}
\]

and changes the compatible `Phi` equation to the STANDARD equation by

\[
(\partial_t\Phi_{IAB})_{\rm std}
=(\partial_t\Phi_{IAB})_{\rm comp}-\beta^J C_{IJAB}.
\]

This is the correct anholonomic-frame form of the standard GH ordering.  The sign agrees with the identity

\[
E_I\Phi_J=E_J\Phi_I+c^K{}_{IJ}\Phi_K+C_{IJ}.
\]

**Classification: PROVED CORRECT.**

### 1.2 `gamma2` reduction damping

`src/ref_gh/gamma2_damping.hpp` adds, in coordinate notation,

\[
\delta(\partial_t\Pi_{AB})=-\gamma_2\beta^i C_{iAB},\qquad
\delta(\partial_t\Phi_{IAB})=\alpha\gamma_2\bar e_I{}^i C_{iAB},
\]

with `gamma1=-1`.  This is the standard first-order GH reduction-constraint damping structure.

**Classification: PROVED CORRECT.**

### 1.3 Principal symbol and characteristic fields

For one symmetric spacetime component, the principal variables are `(Psi,Pi,Phi_I)`.  With a unit spatial covector `s_I`, normalized using the physical spatial inverse metric in the reference frame,

\[
G^{IJ}s_I s_J=1,
\]

the characteristic fields are

\[
u^0=\Psi,
\qquad
u^\pm=\Pi\pm s^I\Phi_I-\gamma_2\Psi,
\qquad
u_I^\perp=\Phi_I-s_I s^J\Phi_J,
\]

with speeds

\[
0,\qquad -\beta^s\pm\alpha,\qquad -\beta^s.
\]

This is exactly the map implemented in `src/ref_gh/ref_gh_characteristics.hpp`.

For any positive-definite tensor-component metric `m^{AB}` and any constant `Lambda` satisfying

\[
\Lambda^2>\gamma_2^2,
\]

a symmetrizer density is

\[
\mathcal E=
 m^{AC}m^{BD}
 \left[
 \Lambda^2\Psi_{AB}\Psi_{CD}
 +\Pi_{AB}\Pi_{CD}
 -2\gamma_2\Psi_{AB}\Pi_{CD}
 +G^{IJ}\Phi_{IAB}\Phi_{JCD}
 \right].
\]

The first three terms complete the square,

\[
(\Pi-\gamma_2\Psi)^2+(\Lambda^2-\gamma_2^2)\Psi^2,
\]

so the energy is positive definite whenever `G^{IJ}` is positive definite.  Direct symbolic multiplication of the one-component principal matrices verifies `S A^s=(S A^s)^T`.

**Classification: PROVED CORRECT for the STANDARD principal part on every `r>0` where the relative metric is Lorentzian.**

### 1.4 Background-covariant lower-order source

`src/ref_gh/covariant_gh_source.hpp` constructs the reference-covariant derivative

\[
Q_{CAB}=P_{CAB}
-\bar\omega^D{}_{AC}\Psi_{DB}
-\bar\omega^D{}_{BC}\Psi_{AD},
\]

then the connection difference

\[
\Delta_{ABC}
=\frac12\left(Q_{BAC}+Q_{CAB}-Q_{ABC}\right),
\]

and includes the reference curvature, quadratic `Q`, quadratic `Delta`, GH damping, and frame-product terms needed to reproduce the coordinate reduced Einstein equation.

This is the correct geometric structure for a background-covariant GH system.  The current implementation, however, evolves the raw frame derivative `Phi` and reconstructs `Q` pointwise.  Consequently the reference spin, frame commutator, and product-rule corrections enter the evolution and subsidiary systems as explicit lower-order matrices.

**Classification: LIKELY CORRECT as a continuum rewrite away from the puncture, but not a satisfactory production variable choice for the strongly varying puncture reference.**

---

## 2. Why fixed-reference growth does not contradict symmetric hyperbolicity

For a linearized first-order system

\[
\partial_t u+A^i(x)\partial_i u=B(x)u,
\qquad S A^i=(S A^i)^T,
\]

symmetric hyperbolicity controls the principal part but does not require the lower-order operator to be dissipative.  The energy estimate contains

\[
\frac{dE}{dt}
\le
\int u^T\left[SB+B^T S+\partial_i(SA^i)\right]u\,d^3x
+\text{boundary flux}.
\]

A spatially varying but time-independent reference can therefore produce exponential growth through the symmetric part of its lower-order matrix without violating symmetric hyperbolicity.

In the present raw-`Phi` variables the relevant coefficient families are:

1. spatial-frame derivatives and structure coefficients in the STANDARD `Phi` equation;
2. spin-connection terms converting raw derivatives to `Q`;
3. reference-curvature and spin-derivative terms in the scalar source;
4. derivatives of the algebraic relative gauge with respect to `Psi`, multiplied by reduction errors when `dPsi` is reconstructed from `Pi/Phi`;
5. spatial gradients of any finite-radius reference blend/window.

The observed common growth of GH, reduction, and curl constraints in a fixed-reference transition annulus is therefore mathematically compatible with a symmetric-hyperbolic principal symbol.

The correct cure is not to change characteristic speeds.  It is to choose variables in which the geometric lower-order connection terms are assembled analytically and the exact matched state is zero, so that large reference terms are never subtracted numerically.

---

## 3. The variable change that should be made

Let `bar g_ab` be the prescribed reference metric and `bar e_A^a` an orthonormal reference frame.  Define

\[
\Psi_{AB}=\bar e_A{}^a\bar e_B{}^b g_{ab},
\qquad
h_{AB}=\Psi_{AB}-\eta_{AB}.
\]

Instead of evolving raw frame derivatives, evolve reference-covariant derivatives of the deviation:

\[
P_{AB}:=-n^a\bar\nabla_a h_{AB},
\qquad
Q_{IAB}:=\bar e_I{}^a\bar\nabla_a h_{AB}.
\]

Because `bar nabla eta=0`, these are also the reference-covariant derivatives of `Psi`.  In frame components,

\[
Q_{IAB}
=E_I h_{AB}
-\bar\omega^C{}_{AI}h_{CB}
-\bar\omega^C{}_{BI}h_{AC}.
\]

The crucial feature is that the spin connection multiplies `h`, not `eta+h` with a later cancellation.  The exact matched state is simply

\[
\boxed{h_{AB}=0,\quad P_{AB}=0,\quad Q_{IAB}=0.}
\]

The connection difference becomes directly

\[
\Delta_{ABC}
=\frac12\left(Q_{BAC}+Q_{CAB}-Q_{ABC}\right),
\]

so no raw-derivative/spin subtraction is required to form it.

### Why this is a small change

For every `r>0`, `(Psi,Pi,Phi)` and `(h,P,Q)` are related by a pointwise triangular affine transformation whose derivative-variable diagonal blocks are identities.  Therefore the principal matrices are the STANDARD GH principal matrices.  Only lower-order terms change.

The new variables are not justified by claiming that the old singular transformation is uniformly bounded at `r=0`.  Instead the new PDE is written natively in `(h,P,Q)`, and puncture compatibility is imposed as an explicit coefficient-regularity condition on the reference provider.

---

## 4. First-order evolution system

Use the reduced Einstein equation

\[
R_{ab}-\nabla_{(a}C_{b)}
+\frac{\gamma_0}{2}
\left(2n_{(a}C_{b)}-g_{ab}n^cC_c\right)=0,
\]

with the gauge constraint defined below.  Reduce the background-covariant wave equation using `(h,P,Q)` and STANDARD ordering.

The principal part is

\[
\partial_t h_{AB}
\simeq-\alpha P_{AB}+\beta^I Q_{IAB},
\]

\[
\partial_t P_{AB}
\simeq
\beta^I E_I P_{AB}
-\alpha G^{IJ}E_IQ_{JAB}
-\gamma_2\beta^I\mathcal C_{IAB},
\]

\[
\partial_t Q_{IAB}
\simeq
\beta^J E_JQ_{IAB}
-\alpha E_IP_{AB}
+\alpha\gamma_2\mathcal C_{IAB}.
\]

All reference-frame motion, spin connection, derivatives of lapse/shift, curvature, and algebraic gauge derivatives are lower order and must be assembled in reference-covariant form.  The complete lower-order scalar source is most cleanly expressed using `Q` and `Delta`; the current `CovariantGhScalarWaveSource` already contains the necessary geometric sectors.  In the new variables the separate raw-`P` to covariant-`Q` `frame_correction` should disappear: `Q` is fundamental rather than reconstructed.

The characteristic fields and symmetrizer are unchanged after replacing `Psi,Pi,Phi` by `h,P,Q`.

---

## 5. Constraint system in the covariant variables

Define the reduction constraint

\[
\mathcal C_{IAB}
:=\bar\nabla_I h_{AB}-Q_{IAB}.
\]

For an anholonomic reference frame, define the covariant curl/integrability constraint

\[
\begin{aligned}
\mathcal C_{IJAB}:={}&
\bar\nabla_IQ_{JAB}-\bar\nabla_JQ_{IAB}
-c^K{}_{IJ}Q_{KAB}\\
&+\bar R^C{}_{A IJ}h_{CB}
+\bar R^C{}_{B IJ}h_{AC}.
\end{aligned}
\]

The sign is chosen so that `C_IJAB=0` follows identically from `Q_IAB=bar nabla_I h_AB` and the anholonomic commutator identity.

The principal part of the subsidiary reduction system is

\[
\partial_t\mathcal C_{IAB}
\simeq
-\alpha\gamma_2\mathcal C_{IAB}
+\beta^J\mathcal C_{JIAB},
\]

while the curl constraint is transported with the shift and coupled algebraically to `C_I` and reference curvature.  The remaining terms contain only bounded reference connection/curvature coefficients if the reference provider satisfies the regularity conditions in Sec. 8.

The GH constraint is especially simple for the wave-map gauge.  With

\[
B_a=-g_{ab}g^{cd}\bar\Gamma^b{}_{cd}
\]

and `H_a=B_a+J_a`, one has

\[
C_a=H_a+\Gamma_a
=J_a+g_{ab}g^{cd}\Delta^b{}_{cd}.
\]

In reference-frame components,

\[
\boxed{C_A=J_A+\Delta_A.}
\]

Thus the GH constraint is constructed entirely from regular residual quantities; no separately divergent `H_A` and `Gamma_A` should be formed near the puncture.

The Bianchi identity gives the usual homogeneous damped wave propagation for `C_a`, modulo the first-order reduction constraints.  Positive `gamma0` damps short-wavelength GH-constraint violations; positive finite `gamma2` damps the reduction constraint.  No `1/alpha` rescaling of `gamma2` is recommended.

---

## 6. Gauge: retain the relative-damped idea, but let the reference carry the puncture transition

Use

\[
\boxed{
H_a=B_a+\bar\theta^A{}_a J_A,
\qquad
J_A=W D_A,
}
\]

with

\[
D_A=\mu_L L_R N_A^R-\frac{\mu_S}{a_R}V_A^R,
\]

where `a_R`, `L_R`, `N_A^R`, and `V_A^R` are constructed algebraically from the relative metric `Psi=eta+h`.

The window has an exact puncture plateau,

\[
W=0\quad\text{for}\quad r\le r_0,
\]

so the puncture core uses pure wave-map gauge to the dynamical reference.  The relative damping acts only in a finite-radius buffer/exterior where all relative ADM quantities are nonsingular.

### Matched state

At `h=0`,

\[
a_R=1,\qquad L_R=0,\qquad V_A^R=0,
\]

so

\[
D_A=J_A=0.
\]

### Flat-background linearization

Writing `Psi=eta+h`, the first variation is

\[
D_0=-\frac{\mu_L}{2}(h_{00}+h_{ii}),
\qquad
D_i=-\mu_S h_{0i}.
\]

For a pure gauge perturbation `h_ab=2 partial_(a xi_b)`, the transverse gauge modes obey

\[
s^2+\mu_S s+k^2=0.
\]

The scalar determinant factorizes exactly as

\[
\boxed{
(s^2+k^2)
\left[s^2+(\mu_L+\mu_S)s+k^2+\mu_L\mu_S\right].
}
\]

This identity was independently checked symbolically.  For `mu_L>0` and `mu_S>0`, the damped factor has no root with positive real part.  One scalar harmonic wave remains undamped, which is not an instability.

Because `J_A` is algebraic in `h` and the reference, its derivative in the reduced Einstein source is algebraic in the first-order variables `(P,Q)`.  Therefore this gauge does not alter the principal symbol or the symmetric-hyperbolic proof.

**Decision:** retain this gauge architecture; do not reactivate the old live `Hhat/theta/Upsilon` moving-puncture driver in the puncture production path.

---

## 7. Why a direct dynamic puncture exponent must be abandoned

The current `q` construction contains the factor

\[
L_q(r,t)\propto r^{-q(t)}.
\]

Exactly,

\[
\partial_t\ln L_q=-\dot q\ln r.
\]

A second derivative produces

\[
\partial_t^2\ln L_q
=-\ddot q\ln r,
\]

and derivatives of `L_q` itself additionally generate `dot q^2 (ln r)^2` and mixed `dot q/r` terms.  No choice of a smooth scalar trajectory `q(t)` removes these terms unless `dot q=0` while the puncture power is changing, which is impossible.

This is a mathematical obstruction to a finite-time global exponent change in the present reference ansatz.  It is not a parameter-tuning problem.

**Classification: INCONSISTENT as a production puncture-control mechanism.  Deprecate `reference_q_controlled` for dynamical exponent tracking.**

---

## 8. The required Bowen--York wormhole -> trumpet reference trajectory

### 8.1 Puncture asymptotics

For a standard wormhole puncture,

\[
\psi_{\rm W}=1+\frac{M}{2r}+u,
\qquad u=O(1),
\]

so the isotropic spatial scale `L=psi^2` behaves as

\[
L_{\rm W}\sim r^{-2}.
\]

With a precollapsed lapse `alpha=psi^{-2}`,

\[
\alpha_{\rm W}\sim r^2,
\qquad \beta^i_{\rm W}=0.
\]

A stationary moving-puncture trumpet has finite limiting areal radius `R_0`, hence

\[
L_{\rm T}=\frac{R}{r}\sim r^{-1},
\]

and its lapse collapses as a positive power of `r`; the shift is linear in Cartesian radius at leading order.

The exact point `r=0` therefore changes singularity class only in the infinite-time limiting geometry.  A bounded formulation should represent the transition as an inward-moving/shrinking wormhole core, not as a finite-time switch of the leading power at `r=0`.

### 8.2 Preferred construction: a genuine Schwarzschild gauge-transition reference

Construct once, in spherical symmetry, a reference spacetime `bar g_ab(t,r)` that is the Schwarzschild solution written in coordinates satisfying:

1. at `t=0`: the chosen isotropic wormhole/Bowen--York-like lapse and zero shift;
2. during the transition: a regular moving-puncture-type coordinate evolution;
3. as `t -> infinity`: the stationary 1+log trumpet profiles used by the existing trumpet table.

This can be generated as a high-accuracy one-dimensional coordinate/gauge evolution on an exact Schwarzschild spacetime.  The production reference should tabulate regular frame-native quantities or a coordinate map from which they can be differentiated consistently.

The essential properties are

\[
\bar R_{ab}=0,
\]

and, in the reference orthonormal frame,

\[
|\bar\omega^A{}_{BC}|<C/M,
\qquad
|\bar R^A{}_{BCD}|<C/M^2,
\]

throughout the transition on `r>0`, with finite one-sided puncture limits for all coefficients actually used by the residual equations.

When the physical solution equals this reference,

\[
h=P=Q=0,
\qquad J=0,
\qquad C_A=0,
\]

and the exact single-hole dynamical transition is a zero-residual solution.  This is the strongest possible regression oracle.

### 8.3 Why an arbitrary smooth interpolation is second best

A reference obtained by interpolating wormhole and trumpet metric profiles is generally not vacuum.  Smoothness prevents ill-posedness, but it creates a nonzero reference Ricci tensor and a finite forcing of the residual equations.  Such a construction may be useful as a fallback, but the vacuum gauge-transition reference is preferable because it makes the desired single-hole path an exact solution rather than an approximate attractor.

### 8.4 Reference coefficients must be provided in regular form

Do not compute a finite spin connection by subtracting separately divergent coordinate-frame terms such as

\[
\partial e + \bar\Gamma e
\]

near the puncture.  The reference provider should return regular frame-native connection coefficients directly (equivalently from regular ADM quantities such as the orthonormal extrinsic curvature, spatial acceleration, and triad rotation).  Reference curvature should likewise be evaluated in a frame-native regular representation.

This rule is as important as abandoning `q(t)`: a mathematically finite answer is not acceptable if production evaluates it as the difference of two divergent floating-point quantities.

---

## 9. Smooth finite-radius windows

For finite-radius gauge/reference stitching, use an exact `C^infinity` plateau rather than the present quintic `C^2` smoothstep.

Define

\[
\rho(x)=\begin{cases}
0,&x\le0,\\
\exp(-1/x),&x>0,
\end{cases}
\]

and

\[
S(x)=\frac{\rho(x)}{\rho(x)+\rho(1-x)}.
\]

Then `S=0` for `x<=0`, `S=1` for `x>=1`, and every derivative vanishes at both endpoints.  For `0<x<1`, let

\[
L(x)=-\frac1x+\frac1{1-x},
\qquad
S=\frac1{1+e^{-L}}.
\]

The first two derivatives are

\[
S'=S(1-S)L',
\qquad
L'=\frac1{x^2}+\frac1{(1-x)^2},
\]

\[
S''=S(1-S)\left[(1-2S)(L')^2+L''\right],
\qquad
L''=-\frac2{x^3}+\frac2{(1-x)^3}.
\]

For a physical window of width `Delta`, derivatives scale as `1/Delta` and `1/Delta^2`.  `C^infinity` smoothness alone is not enough: `Delta` must not shrink with grid spacing or with the puncture-core radius.  The reference qualification step should evaluate the actual lower-order energy matrix and reject windows whose symmetric part is too large.

---

## 10. GH-consistent Bowen--York initial data

Given physical Cauchy data `(gamma_ij,K_ij)`, a chosen initial lapse/shift, a reference, and an algebraic gauge `H_a(x,t,g)`, construct first-order data as follows.

### Step 1: metric and spatial first jet

Build

\[
g_{00}=-\alpha^2+\gamma_{ij}\beta^i\beta^j,
\quad
g_{0i}=\gamma_{ij}\beta^j,
\quad
g_{ij}=\gamma_{ij},
\]

and their spatial derivatives.  Project to

\[
h_{AB}=\bar e_A{}^a\bar e_B{}^b g_{ab}-\eta_{AB}.
\]

Then compute

\[
Q_{IAB}=\bar e_I{}^a\bar\nabla_a h_{AB}.
\]

### Step 2: use `K_ij` for the six physical time-derivative combinations

The ADM identity

\[
\partial_t\gamma_{ij}
=-2\alpha K_{ij}+\mathcal L_\beta\gamma_{ij}
\]

fixes the six `ij` components of the metric time derivative.

### Step 3: solve the four GH gauge equations

Use as unknowns

\[
x^A=(\partial_t\alpha,\partial_t\beta^1,
      \partial_t\beta^2,\partial_t\beta^3).
\]

For fixed `(gamma,K,alpha,beta)` and spatial derivatives, the ten components of `partial_t g_ab` are affine in these four unknowns.  Hence

\[
\Gamma_a=\Gamma_a^{(0)}+M_{aA}x^A.
\]

Because the recommended `H_a` is algebraic in `g` and the prescribed reference, it is already known at the initial point.  Solve the `4 x 4` system

\[
\boxed{M_{aA}x^A=-H_a-\Gamma_a^{(0)}}
\]

so that

\[
C_a=H_a+\Gamma_a=0.
\]

This determines the remaining four time-derivative combinations without changing the physical Cauchy data `(gamma,K)`.

### Step 4: construct all ten `P_AB`

With the complete coordinate first jet now known, compute

\[
P_{AB}=-n^a\bar\nabla_a h_{AB}.
\]

The result supplies exactly 10 independent symmetric components: six were fixed physically by `K_ij`, four by the GH gauge constraints.

For standard Bowen--York data, the reference should capture only the universal singular factors.  The regular Bowen--York correction `u`, linear momentum, and spin remain finite residual data rather than being absorbed into a singular reference.

---

## 11. Symmetric-hyperbolic proof for the proposed system

The change from raw first derivatives to reference-covariant first derivatives changes only lower-order terms.  The principal matrices are therefore the STANDARD `gamma1=-1` GH matrices.

For the state

\[
U=(h_{AB},P_{AB},Q_{IAB}),
\]

take

\[
\mathcal E[U]
=m^{AC}m^{BD}
\left[
\Lambda^2h_{AB}h_{CD}
+P_{AB}P_{CD}
-2\gamma_2h_{AB}P_{CD}
+G^{IJ}Q_{IAB}Q_{JCD}
\right],
\]

with `m^{AB}>0`, `G^{IJ}>0`, and `Lambda^2>gamma2^2`.  Then `S>0` and direct block multiplication gives

\[
SA^i=(SA^i)^T.
\]

The algebraic relative gauge and every reference-curvature/connection term are lower order.  They do not alter this proof.

**Classification: PROVED, conditional only on the relative metric remaining Lorentzian and the spatial block remaining positive definite.**

Symmetric hyperbolicity does not prove numerical decay.  The lower-order reference and gauge matrices still require a static local energy/Jacobian qualification before evolution tests.

---

## 12. Puncture regularity conditions

The production reference is acceptable only if the coefficients that appear in the residual equations, not merely the reconstructed physical metric, obey finite puncture limits.  Require at minimum:

\[
G^{IJ}=O(1),\quad (G^{-1})_{IJ}=O(1),
\]

\[
\bar\omega^A{}_{BC}=O(M^{-1}),
\quad
\bar R^A{}_{BCD}=O(M^{-2}),
\]

\[
J_A=O(1),\quad
\frac{\partial J_A}{\partial h_{BC}}=O(M^{-1}),
\]

and bounded corresponding first derivatives needed by the lower-order source.

The dynamic reference must be supplied in a representation where these bounds hold **before** numerical evaluation.  A term that is finite only after subtracting two quantities that diverge as `r -> 0` fails this gate.

---

## 13. Spinning punctures and BBHs

The leading Bowen--York wormhole powers are unchanged by finite linear momentum or spin, so the same local regularization architecture is plausible for boosted/spinning punctures.  The nonspherical corrections live in `h,P,Q`.

For binaries, use two local puncture reference cores, each with its own wormhole-to-trumpet transition history and moving center.  Stitch them only at finite radius with `C^infinity` windows whose physical width stays finite.  The outer reference may contain translation/rotation/inspiral control maps.

Do **not** merge the two singular references when a common horizon forms.  Keep both local puncture cores hidden inside the common horizon and transition only the smooth outer reference toward a remnant-centered frame.  The PDE and local puncture regularization remain unchanged through merger.

This BBH extension is mathematically plausible but not proved stable.

---

## 14. Concrete AthenaK recommendations

### Retain

- `STANDARD` `Phi` ordering as the principal-system model.
- `gamma1=-1` and finite nonnegative `gamma2` reduction damping.
- the background-covariant connection-difference construction in `covariant_gh_source.hpp`.
- the relative-damped gauge idea and its exact open puncture core.
- the analytic/generic reference-provider abstraction and diagnostic separation.

### Deprecate/remove from puncture production

- dynamical `reference_q_controlled` exponent tracking;
- the live `Hhat/theta/Upsilon` 1+log/Gamma-driver puncture path;
- any reference/gauge source assembled by subtracting independently divergent physical and reference quantities;
- `C^2` exact-plateau blends in high-order production paths.

### Change

1. Replace evolved `Psi` by `h=Psi-eta` (storage count unchanged).
2. Replace raw `Pi/Phi` by background-covariant `P/Q`.
3. Re-express the scalar source directly in `h,P,Q,Delta`, eliminating the raw-frame `frame_correction` cancellation path.
4. Construct `C_A=J_A+Delta_A` directly.
5. Provide a dynamic single-hole wormhole-to-trumpet reference trajectory, preferably a vacuum Schwarzschild gauge-transition table/map.
6. Return regular frame-native spin/curvature data from the reference provider.
7. Replace plateau smoothsteps by an exact `C^infinity` bump and impose gradient bounds.
8. Add a GH-consistent initial-data constructor that solves `C_a=0` for the four gauge time derivatives.

### Suggested source-file map

- `src/ref_gh/ref_gh_state.hpp`: keep the 10+10+30 Einstein-field layout but reinterpret/name it as `h`, `P`, `Q`; do not add the old 11 gauge-driver fields to the relative-damped production state.
- `src/ref_gh/ref_gh_calcrhs.cpp`: derive the STANDARD `Q` equation directly from the background-covariant first-order reduction; do not evolve raw `Phi` and then reconstruct `Q`.
- `src/ref_gh/covariant_gh_source.hpp`: make `Q` an input and delete the raw-derivative-to-`Q` conversion and its separate `frame_correction` production sector after oracle equivalence is established.
- `src/ref_gh/relative_damped_gauge.hpp`: retain the algebraic relative gauge, but feed it the covariant first jet and replace the quintic window by the `C^infinity` plateau.
- `src/ref_gh/reference_trumpet_q_controlled.hpp` and `q_relaxed_controller.hpp`: keep only as diagnostic/history code or remove from the puncture production path.
- `src/ref_gh/gauge_driver.hpp` and `physical_gauge_target.hpp`: retain for non-puncture/oracle experiments if desired, but do not use them as the live puncture gauge controller.
- reference providers: add a `wormhole_trumpet_transition` provider that returns a consistent time-dependent reference two-jet plus regular frame-native spin/curvature data.
- initial-data path: add a reusable `SolveGhGaugeTimeDerivatives` helper implementing the `4 x 4` solve in Sec. 10 before projection to `P`.

### New static qualification helpers

Before any production evolution, add pointwise/frozen-coefficient audits for:

- symmetrizer positivity;
- `S A^i` symmetry;
- reference spin/curvature puncture bounds;
- lower-order main-system energy matrix;
- lower-order subsidiary constraint matrix;
- relative-gauge Jacobian;
- exact zero-residual preservation on the dynamic Schwarzschild reference.

---

## 15. Minimal later numerical test sequence

No result is claimed here; these are future gates.

1. **Algebra/unit gate:** manufactured smooth reference; compare raw-coordinate and new covariant-variable RHS to roundoff.
2. **Dynamic-reference zero-state gate:** Schwarzschild wormhole-to-trumpet reference with `h=P=Q=0`; verify the RHS is zero to reference-table accuracy for the entire transition.
3. **Fixed-reference perturbation gate:** reproduce the formerly unstable finite-radius reference and show that reduction/curl perturbations no longer exhibit the same positive frozen-coefficient mode.
4. **Single Bowen--York Schwarzschild gate:** initialize wormhole data with the GH-consistent `4 x 4` solve and evolve through the full gauge transition.
5. **Resolution gate:** verify convergence of GH, reduction, and curl constraints without changing damping/window parameters.
6. **Boosted/spinning single puncture.**
7. **Two punctures through common-horizon formation with both local cores retained.**

---

## 16. Final answers to the ten questions

1. **Underlying current STANDARD continuum system:** correct in principal part and geometrically consistent away from the puncture; the current raw-frame lower-order realization is not the formulation I recommend for production punctures.
2. **Fixed-reference reduction/curl growth:** yes, lower-order spatial-frame/gauge matrices can grow constraints without violating symmetric hyperbolicity.
3. **Current relative-damped gauge:** fundamentally sound as an algebraic relative gauge; retain it after changing the derivative variables and window regularity.
4. **Time-dependent singular exponent `q(t)`:** abandon it.  `qdot log r` is unavoidable.
5. **Clean wormhole->trumpet representation:** a genuine dynamical single-hole gauge-transition reference with a wormhole core persisting at the exact puncture for every finite time and an asymptotic trumpet at fixed `r>0`.
6. **Gauge:** wave-map to that dynamical reference plus the existing relative-damped algebraic correction outside an exact puncture core.  Do not use the old live moving-puncture GH driver in the core.
7. **Positive-definite symmetrizer:** yes; the STANDARD GH symmetrizer above applies unchanged to `(h,P,Q)`.
8. **Puncture coefficients:** controlled if the reference provider supplies bounded frame-native connection/curvature coefficients and never forms finite quantities by divergent subtraction.  This is an explicit qualification gate.
9. **GH-consistent initial data:** preserve `(gamma,K)`, solve `C_a=0` for the four lapse/shift time derivatives, then construct all ten `P_AB`.
10. **Spin/BBH extension:** plausible with two persistent local cores and only smooth finite-radius outer-frame changes; it requires future numerical validation.

---

## 17. Claim status

### PROVED

- STANDARD `gamma1=-1` principal symbol and characteristic speeds.
- Positive-definite symmetrizer for `Lambda^2>gamma2^2`.
- `gamma2` reduction-damping sign used by the code.
- direct exponent tracking produces `qdot log r`.
- the relative-damped flat matched-state determinant factorization quoted above.
- algebraic gauges of `h` and prescribed reference fields do not change the first-order principal symbol.

### SUPPORTED BY EXISTING NUMERICAL EVIDENCE

- the old live moving-puncture GH driver is structurally unsafe in the present puncture implementation;
- reference motion amplifies the instability;
- a fixed spatially varying reference can support a lower-order reduction/curl growth mode;
- finite-radius transition regions are the important localization to analyze.

### PLAUSIBLE BUT UNPROVED

- evolving background-covariant first derivatives will remove the observed fixed-reference lower-order mode;
- a dynamic Schwarzschild transition reference plus relative damping will robustly attract generic Bowen--York single-hole data;
- the two-core architecture will remain robust through binary merger.

### REQUIRES FUTURE NUMERICAL TESTING

- all stability/robustness claims beyond the principal-symbol and local symbolic results;
- the exact choice of transition-table resolution and outer gauge-window width;
- boosted, spinning, and BBH production qualification.

---

## 18. Reproducible symbolic identities

The two algebraic identities used above can be checked with a few exact symbolic operations.  For the one-dimensional STANDARD principal derivative matrix

\[
B=\begin{pmatrix}
0&0&0\\
-\gamma_2\beta&\beta&-\alpha\\
\alpha\gamma_2&-\alpha&\beta
\end{pmatrix},
\qquad
S=\begin{pmatrix}
\Lambda^2&-\gamma_2&0\\
-\gamma_2&1&0\\
0&0&1
\end{pmatrix},
\]

exact symbolic multiplication gives `S B - (S B)^T = 0`.  Positivity is the completed-square condition `Lambda^2>gamma2^2`.

For the scalar relative-gauge plane-wave matrix

\[
M=\begin{pmatrix}
A+\mu_L s&i\mu_L k\\
i\mu_S k&A+\mu_S s
\end{pmatrix},
\qquad A=s^2+k^2,
\]

exact factorization gives

\[
\det M=(s^2+k^2)
\left[s^2+(\mu_L+\mu_S)s+k^2+\mu_L\mu_S\right].
\]

These are algebraic identities, not numerical fits.

## References

- L. Lindblom, M. A. Scheel, L. E. Kidder, R. Owen, O. Rinne, *A New Generalized Harmonic Evolution System*, arXiv:gr-qc/0512093.
- L. Lindblom, B. Szilagyi, *An Improved Gauge Driver for the Generalized Harmonic Einstein System*, arXiv:0904.4873.
- M. A. Scheel et al., *Solving Einstein's Equations With Dual Coordinate Frames*, arXiv:gr-qc/0607056, Phys. Rev. D 74, 104006.
- M. Hannam et al., *Geometry and Regularity of Moving Punctures*, arXiv:gr-qc/0606099.
- M. Hannam et al., *Where do moving punctures go?*, arXiv:gr-qc/0612097.
- M. Hannam, S. Husa, N. O Murchadha, *Bowen--York trumpet data and black-hole simulations*, arXiv:0908.1063.
- T. Dietrich, B. Bruegmann, *Solving the Hamiltonian constraint for 1+log trumpets*, arXiv:1309.3087.
