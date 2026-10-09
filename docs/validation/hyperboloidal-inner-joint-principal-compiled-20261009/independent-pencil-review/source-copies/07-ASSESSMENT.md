# Inner modified BM / physical reference wave-map feasibility

This is a source-only mathematical preparation, dated 2026-10-09. It makes no implementation, kernel-query, evolution, adoption or black-hole acceptance claim. The later user requirement is a single black hole that survives a wormhole-to-trumpet inner transition while the hyperboloidal reference remains Minkowski. A stationary end is a useful additional compatibility question; it is not an extra acceptance requirement imposed here.

There are two separate findings. A reference-preserving row blend can formally recover the complete, previously audited constrained-20 principal family at every finite positive lapse. An inner-only blend cannot repair a conditional stationary outer obstruction of the *unchanged* physical Minkowski wave-map condition: its spherical, Killing-aligned Schwarzschild branch lacks the required mass logarithm in the height. Retaining the Minkowski reference does not require retaining that exact outer gauge condition for the later BH problem.

## Conventions and the two inner regions

Write the physical four-metric as `g=Omega^-2 barg`, with signature `-+++` and `Kij=-(1/2) Lie_n gammaij`. The stored lapse `alpha` belongs to `barg`; the physical lapse is `alpha/Omega`. The shift is unchanged by this conformal rescaling. The evolved trace is `P=Kphysical-2Thetaphysical`, and the production conformal-trace notation remains `Q=(P-3 omega_n)/Omega`, `omega_n=-beta^i Omega_i/alpha`. `Lambda^i=GammaTilde^i+2 gtildeInv^{ij} Z_j`. Do not drop this Z contribution when comparing gauge sources off the Einstein sector.

`W=W_gauge(r)` is a fixed smooth cutoff, and `c=1-W`. It is distinct from the geometric compactification cutoff. For the currently studied reference these radii are respectively `.45--.85` and `.05--.95`, with `S=1,a=.5`. Only `r<=.05` is the exact geometric Cauchy plateau `Omega=alpha_hat=chi_hat=1`, `beta_hat=P_hat=Lambda_hat=0`, with all reference derivatives zero. The larger region `W=0` is the inner *gauge* plateau; most of it is geometrically hyperboloidal. A textbook physical BM or isotropic-trumpet formula cannot silently be applied throughout that region.

Throughout the support `c>0` at these parameters, `Omega>=.2775`, since the geometric blend lies between `1` and `1-r^2`. Thus the following inner additions do not add a new scri pole. Exact `c=0` must be branched before evaluating any inner quotient at `Omega=0`. This support observation gives no uniform puncture estimate.

All coordinates, `a,S,M,R,r,h` have length units; `Omega,alpha,beta,chi,W,f,mu,q` are dimensionless; `P,K,Lambda,eta,nu` have inverse-length units. The constant `q0=.5` below is a driver parameter, distinct from the endpoint boost `q_end` in the earlier trumpet calibration.

## A formal blend that preserves the Minkowski fixed point

Let `G_R` be the full coupled lapse and shift rows of the physical reference wave-map gauge, using the fixed Minkowski reference connection. On the Einstein sector it imposes

```text
g^{bc}(Gamma[g]^a_bc-Gamma[ghat]^a_bc)=0.
```

Off that sector the proposed source includes `+2 Zphysical^a`. Define `D0=partial_t-beta^i partial_i` and `Hhat^a=barg^{bc} Gamma[ghat]^a_bc`. The independently audited finite-Omega rows are

```text
D0 alpha = -(alpha^2 P+alpha beta^i Omega_i)/Omega-alpha^3 Hhat^0,
D0 beta^i = alpha^2 [chi Lambda^i
  +gtildeInv^{ij}(chi_j/2-chi alpha_j/alpha)
  +2 chi gtildeInv^{ij} Omega_j/Omega-Hhat^i-beta^i Hhat^0].
```

The frozen helper evaluates these with reference deviations and both lapse and shift pole parts; its assembled rows contain no division by the live lapse. Its stationary Minkowski identity is analytic reference algebra, not a numerical reference RHS counterterm. Source IDs S01--S03 identify this exact convention and helper.

One possible *definition to test later*, rather than an admitted implementation, is

```text
G_blend = c G_I + W G_R,

G_I alpha = beta^i alpha_i-beta_hat^i alpha_hat_i
            -alpha(alpha+2)(P-P_hat)/Omega
            -alpha nu_I log(alpha/alpha_hat),
G_I beta^i = beta^j partial_j beta^i
            -beta_hat^j partial_j beta_hat^i
            +alpha^2 chi (3q0/4)(Lambda^i-Lambda_hat^i)
            -eta_I(beta^i-beta_hat^i).
```

Here `P_hat=Kphysical_hat` because the reference has zero Theta, and `nu_I,eta_I` are prescribed nonnegative rates. The intended lapse-collapse calibration uses `nu_I=0`. Both `G_I` and `G_R` vanish on the same analytic Minkowski reference, so this formal blend has that fixed point everywhere, including the nonflat transition. No BH field is inserted into the reference source.

In the exact geometric core it reduces to the actual modified-BM/weak-driver form

```text
alpha_t = beta^i alpha_i-alpha(alpha+2)P,
beta_t^i = beta^j beta^i_j+alpha^2 chi(3q0/4)Lambda^i-eta_I beta^i,
```

when `nu_I=0`. On Einstein data `P=Kphysical`. For `W=1` it is exactly the full physical-reference wave-map gauge, with harmonic principal lapse and shift. It is not the earlier physical-P/spatial-norm outer source, nor the conformal-Q preferred-source candidate. Replacing only the lapse coefficient or retaining a separately damped old shift would define a different system.

The fixed reference-advection terms, `P_hat` and `Lambda_hat` terms in `G_I`, and the full live-metric contraction of the reference connection in `G_R` are necessary to the stated reference identity. These are lower-order source terms in the pseudodifferential principal grading. The inner modification of lapse/shift principal coefficients itself is **not** merely lower order. When differentiating this variable-coefficient system, all derivatives of `W`, reference coefficients and any prescribed rates must be retained.

For a future implementation, the lapse-trace coefficient is directly `alpha(alpha+2c)`. The harmonic lapse-gradient contribution to the shift is directly `-W alpha chi gtildeInv^{ij} alpha_j`. No live-lapse division is needed for these assembled terms. A direct branch/blend must avoid subtracting large near-puncture background terms; the exact core has zero reference gradients. If a nonzero logarithmic restoration is retained, positive-lapse log differences must be evaluated without rounding `alpha-alpha_hat` to `-alpha_hat`. These observations are cancellation requirements, not positivity or puncture regularity proofs.

## Principal structure: conditional on the stated complete blend

Freezing a positive lapse, positive chi and SPD conformal metric, the blend above has exactly the previously derived coefficient family

```text
f  = 1+2c/alpha,
mu = c(3q0/4)+W,
epsilon_alpha=W, epsilon_chi=W/2,
q  = q0+(1-q0)W = (4mu-2epsilon_chi)/3.
```

These follow by blending the *complete* coupled rows: common shift advection is retained, and the reference terms are algebraic sources. They are the source-inspected family in S04--S06. The normalized scalar eight-block has light speeds `+/-1` twice, lapse speeds `+/-sqrt(f)`, and longitudinal-shift speeds `+/-sqrt(q)`; each transverse four-block has `+/-1,+/-sqrt(mu)`; each tensor two-block has `+/-1`. This accounts for all 20 algebraically constrained modes. Coordinate propagation speeds have the corresponding `-beta^n +/- alpha sqrt(chi gtildeInv^{nn})` factors.

The existing cancelled left basis has scalar determinant `-24 sqrt(q/f)`. For `alpha>0`, `0<q0<1`, `0<=W<=1`, it stays complete, including the coincident harmonic endpoint `W=1`. This is an algebraic inheritance of an already-audited family, not a fresh actual-symbol test of a new helper. Such an implementation still needs its own full 20-field, oblique-direction symbol extraction and exact nonlinear reference/source checks.

At the limiting puncture `alpha->0` and `chi->0`, the normalization is singular and the determinant is not bounded away from zero. The normalized lapse speed diverges; the coordinate lapse speed scales as `sqrt(2alpha chi gtildeInv^{nn})` in the core and can instead tend to zero on a trumpet. Neither observation establishes uniform strong hyperbolicity or well-posedness at that limiting point. A constant-coefficient Gamma driver would change `mu` and the full coupled principal structure and cannot be adopted using these formulas unchanged. The old harmonic-endpoint scalar Jordan failure when coupling terms are omitted remains relevant context.

## What the existing inner trumpet calibration actually says

S07 is an equation-based conditional calibration of the exact-core gauge; it does not supply initial BH data or prove formation. Assuming stationary spherical Schwarzschild, isotropic Cartesian spatial metric and a regular radial power-law or polyhomogeneous expansion with controlled differentiated remainders,

```text
alpha ~ a0 r^p, chi ~ r^2/R0^2,
beta^r ~ v r, gtilde=I, Lambda=0,
p v=2K0, p>0, v>0.
```

The actual shift driver weakens as `alpha^2 chi ~ r^(2p+2)`. More generally, if a regular conformal-metric expansion gives `Lambda=O(r^-1)` up to logarithms, its driver is `o(r)` for `p>0`; bounded metric components alone do not imply this derivative bound. The leading radial shift balance is therefore `v(v-eta_I)r`, so `eta_I=v>0` is necessary for this class of stationary endpoint. The present default `eta_I=0` fails that conditional leading balance.

Necessity is not sufficiency. Even the exactly isotropic modified-BM branch with `Lambda=0` requires a nonconstant `eta_required(R)=partial_r beta^r`; setting its endpoint limit `eta_I=v` leaves a nonzero next `O(r^(1+p))` residual. Other radial coordinates may develop nonflat conformal metric and a nonzero geometric Lambda. Assigning Lambda independently on an unchanged isotropic metric would introduce Z and is not an Einstein-sector solution.

The earlier selected `f=1+2/alpha`, zero-offset areal branch obeys `q_end`'s parent relation `q=C(alpha+2)/R^2`. Its equation-based endpoint has `R0/M=1.3195497562`, `p=1.0607696620`, `M v=.5442012448`, `M K0=.2886360852`. These are context for the exact-core conditional calculation, not targets for the blended hyperboloidal BH. S07 preserves the paper's inconsistent printed `1.3955M` separately from its equations. Ohme et al. use the opposite `Kij=+(1/2) Lie_n gammaij` convention; their hybrid slicing and offset result cannot be imported without that sign translation. Their slicing-only stationary construction does not validate this weak shift driver. [Ohme et al., Eqs.24--31](https://arxiv.org/pdf/0905.0450v2).

For `M=.5`, that conditional endpoint gives `eta_I=1.0884024895`. The existing production parameter validator requires restoring rates to increase outward and would reject it against the default outer shift rate `1`. A new row-blend definition would need its own parameter contract; silently bypassing that validator is not part of this note. With the current exact geometric core radius `.05` and span `2.1`, N24/N36 contain no exact-core cell centers and N48 has only eight. Existing finite-pulse Minkowski tests cannot resolve this trumpet asymptotic question.

## BH height data need their own mass-corrected outer asymptotics

Use `R` for Schwarzschild **areal** radius in this and the next section, not the physical isotropic radius of the existing wormhole constructor. Let

```text
F=1-2M/R,
ds^2=-F dT^2+F^-1 dR^2+R^2 dOmega_sphere^2,
T=t+h_BH(R).
```

The induced radial metric is `gamma_RR=F^-1-F h_BH'^2`. A nondegenerate smooth future hyperboloidal compactified spatial end requires `gamma_RR=a_B^2/R^2+O(R^-3)`, for some `a_B>0`. Consequently

```text
h_BH' = F^-1 sqrt(1-a_B^2 F/R^2+O(R^-3))
      = F^-1-a_B^2/(2R^2)+O(R^-3),
h_BH  = R+2M log(R/ell)+C+O(R^-1).
```

The arbitrary length `ell` changes only the additive height constant. The leading `2M/R` term in the derivative is required independently of the eventual inner slicing. The future-null outer shift is inward for this height convention, whereas the future-BH trumpet core shift is outward; a matching solution must accommodate a sign change. A bare Minkowski height has derivative `1+O(R^-2)` and instead gives `gamma_RR=4M/R+O(R^-2)`, leaving a Penrose radial-metric pole under ordinary `r/Omega` compactification.

S08--S09 record prior, separate constraint-satisfying **initial-data** constructions with mass-corrected heights. In physical isotropic radius `Riso`, `psi=1+M/(2Riso)`, `N=(1-M/(2Riso))/(1+M/(2Riso))`, the static outer height relation `h'_iso=psi^2 v/N`, with `v` approaching the reference Minkowski boost, supplies the required mass logarithm. Its use must be switched off smoothly through the throat neighborhood; the signed static lapse cannot supply the desired everywhere-positive initial lapse. The detached `M=.5` construction uses an independent BH height cutoff around `.30--.95` while retaining the Minkowski reference and compactification `.05--.95`. Its ADM fields need not share the reference conformal metric. Those data pass instantaneous geometry/constraint audits only.

The native older constructor ties its height cutoff to the reference and requires `M<2 r_geom0`; therefore it cannot represent `M=.5` with the wide reference. An independent BH cutoff and its own throat inequalities are required. Neither a frozen initial constraint proof nor the source-only blend above proves survival of the subsequent wormhole-to-trumpet transition.

## Conditional stationary obstruction of the unchanged outer wave-map gauge

The following is a derivation specific to a stationary, spherical, Killing-aligned, smooth future end. Assume `M>0`, `partial_t=partial_T` is the normalized Schwarzschild Killing field, the reference inertial spatial scalars are `Yhat^I=f(R)n^I`, and the reference inertial time is

```text
Yhat^0=t+h_hat(f)=T+psi(R),
psi=h_hat(f)-h_BH,
h_hat(f)=sqrt(f^2+a^2)+constant
```

in the exact outer reference collar. On Einstein data, the unchanged physical Minkowski wave-map condition is exactly `Box_g Yhat^A=0`. This follows by applying the scalar chain rule to the reference inertial embedding: `Box_g Yhat^A=-(partial_a Yhat^A) g^{bc}(Gamma[g]^a_bc-Gamma[ghat]^a_bc)`. It is not an additional assumption about the conformal source.

The stationary scalar wave operator gives

```text
R^2 Box_g(f n^I) = [(R^2 F f')'-2f] n^I,
R^2 Box_g(T+psi) = (R^2 F psi')'.
```

The complete spatial l=1 radial solution on `R>2M` is

```text
f(R)=c1(R-M)+c2[(R-M)log(1-2M/R)+2M].
```

To verify the second branch, its derivative is `log F+2M(R-M)/(R(R-2M))`, so `R^2F f2'=R(R-2M)log F+2M(R-M)` and differentiating gives `2f2`. At infinity `f2=-2M^3/(3R^2)+O(R^-3)`; there is no `log R` branch there. The branches are independent, since their Wronskian is nonzero for `M>0,R>2M`. The usual outer radial normalization has `c1=1`; any positive constant scaling gives the same missing-mass-term conclusion once the null time normalization is matched.

The temporal equation yields `psi'=D/(R^2F)` with constant `D` of length squared. Equivalently `psi=(D/(2M))log F+constant`, so its infinity expansion is `constant-D/R+O(R^-2)`, also without `log R`. It follows that

```text
h_BH'=h_hat'(f) f'-D/(R^2F)=1+O(R^-2)
```

for `c1=1`. This contradicts the required `1+2M/R+O(R^-2)` future hyperboloidal derivative for `M>0`. A constant rescaling of the Killing time and of `f` can alter the leading constant, but cannot generate the missing inverse-radius coefficient.

For clarity, under `f=R-M+O(R^-2)=r/Omega`, `dR/dr=L/Omega^2[1+O(R^-3)]`, `L=Omega-r Omega'`. The forced stationary branch has

```text
bargamma_rr=Omega^2 gamma_RR (dR/dr)^2
           =4M L^2/(r Omega)+O(1).
```

Thus it is incompatible with the assumed finite nondegenerate Penrose spatial metric. Choosing the decaying spatial solution alone would lose the asymptotic radial coordinate and is not a hyperboloidal compactification of this end. The argument concerns the outer end, not regularity of Schwarzschild `T` at the horizon. The limit `M=0` is exceptional and restores the reference Minkowski branch.

This is **not** a finite-time blowup theorem, a native-instability diagnosis, or a proof that time-dependent BH coordinates cannot remain regular. It also does not exclude a different asymptotic frame, a polyhomogeneous conformal formulation with a different admissible domain, or a changed outer gauge. Inner blending leaves the unchanged outer equations intact and so cannot fix this stationary incompatibility. The current Minkowski source/finite-amplitude gates retain their meaning.

The distinction is consistent with the literature without invoking its stability theorem: Hintz--Vasy explicitly choose a mass-dependent Schwarzschild background near infinity rather than the Minkowski background, and define the background-connection gauge in Eq.3.1. Their continuum construction and function spaces differ from this Z4c layer and do not prove the above discrete or puncture problem. [Hintz--Vasy, introduction and Eq.3.1](https://arxiv.org/pdf/1711.00195v2).

## What an outer change would have to accomplish

The Minkowski reference may remain fixed while the **gauge source** becomes a derived, live-field-dependent extension. Such a source must vanish exactly at the Minkowski reference, supply the mass-dependent outer asymptotic term, and preserve the desired derivative-order principal matrix. A fixed nonzero `M` forcing added to the Minkowski run would not meet the first requirement. Subtracting a complete BH RHS would not be a gauge-source derivation and is excluded.

One diagnostic of the needed source follows by retaining the smooth growing spatial harmonic `f=R-M` and the regular mass-corrected height. Then `psi=h_hat(f)-h_BH=-2M log(R/ell)+O(1)`, and

```text
Box_g Yhat^0=-2M/R^2+O(R^-3),   Box_g Yhat^I=0.
```

If the physical gauge is generalized to `Cphys^a=S^a`, where `Cphys^a=g^{bc}(Gamma[g]^a_bc-Gamma[ghat]^a_bc)+2Zphysical^a`, the Einstein-sector identity is `Box_g Yhat^A=-(partial_a Yhat^A)S^a`. With the spatial scalars still harmonic this particular stationary branch would therefore require `S^r=0`, `S^t=+2M/R^2+O(R^-3)`. Both signs follow from the scalar wave identity. This is a *necessary leading asymptotic target for that chosen branch*, not a completed or proposed general source. It has inverse-length dimensions and disappears with mass.

In the conformal gauge rows the same modification is `Fbar -> Fbar+S/Omega^2`; since `R~r/Omega`, the displayed temporal target is finite of order `2M/r^2`. Its lapse/shift additions are `-alpha^3 S^0/Omega^2` and `-alpha^2(S^i+beta^i S^0)/Omega^2`. Therefore a derived algebraic live-field correction can change the relevant boundary rates while leaving the finite-Omega derivative principal part unchanged. Its metric dependence, nonlinear Jacobian, off-constraint poles and full null/trace/shear hierarchy still require derivation and actual checks.

An illustrative **leading-asymptotic surrogate**, not an admitted candidate, makes the available algebraic freedom explicit. In the exact spherical outer collar let `f=r/Omega` and `s=(det_2 bargamma_AB)^(1/4)`, where the determinant is taken on an orthonormal Euclidean tangent plane. The Minkowski reference there has `s_hat=1`; on the chosen spatial-harmonic Schwarzschild branch `s=R/f=1+M/f`. The value-only addition

```text
delta Fbar^0=2(s-1)/(r Omega),   delta Fbar^i=0
```

vanishes exactly at that Minkowski reference and gives `Sphys^t=Omega^2 delta Fbar^0=2M/f^2=2M/R^2+O(R^-3)` on that branch. It therefore supplies the necessary leading mass reaction with an off-reference *simple* Omega pole, not a fixed BH source. A smooth support would have to lie wholly in the collar where the stated reference identity holds. The corresponding assembled lapse/shift additions have no live-lapse division. Their coefficients depend only on metric values, so they do not change the finite-Omega derivative principal matrix.

This surrogate proves neither a full stationary solution nor preservation of any scri manifold. In nonspherical, nonstationary data `s` is a local angular area factor, not a justified mass measurement, and its off-reference pole may introduce unacceptable modes. This note supplies no physical mass extraction from arbitrary radiative fields. A quasi-local mass proxy is also not automatically a valid gauge source; derivative-dependent proxies can alter the principal equations. Higher-order asymptotic and actual source/pole gates would be mandatory before treating any such expression as a candidate.

A different possibility is a preferred-conformal projection. At positive `Omega`, with `v_i=Omega_i` and `|v|_E^2>0`, a spatial source replacement

```text
delta_box=barg^{ij} Omega_ij-Fbar^i Omega_i-B_ref,
Fbar_new^i=Fbar^i+Omega_i delta_box/|grad Omega|_E^2,
Fbar_new^0=Fbar^0,
```

sets `barg^{ij} Omega_ij-Fbar_new^i Omega_i=B_ref`. This equals `Box_barg Omega` on the Einstein/gauge-constraint sector. Off it, the gauge relation `GammaBar^a+2Zbar^a=Fbar^a` leaves the additional term `+2Zbar^a Omega_a`; the displayed algebraic projection does not remove that term. Choosing the analytic reference `B_ref=Box_barghat Omega` preserves Minkowski, and its boundary value is zero for the current reference. It is algebraic in the live metric and fixed reference jets, so the finite-Omega harmonic derivative principal part is unchanged. Support must avoid `grad Omega=0` and all derivative/Jacobian effects must be retained. Adding the full live Z term into the projection would require a new principal analysis, since spatial Z contains metric derivatives and Lambda. Zenginoglu's Proposition 2 explains the preferred conformal boundary condition in a general wave reduction, conditional on a smooth conformal completion; it does not prove this particular extension or its Z4c hierarchy. [Zenginoglu, sections 2--3](https://arxiv.org/pdf/0808.0810v1).

This single scalar projection does **not** by itself establish a stationary, smooth mass end. It changes spatial harmonic conditions while leaving the temporal source unchanged; both must be checked jointly. Allowing `f=R+2M log R+...` could repair the temporal height balance, for example, but introduces `Omega log Omega` behavior in ordinary compactified angular metric coefficients. That is outside the smooth Taylor-jet domain assumed above unless an explicitly justified polyhomogeneous formulation is adopted. Neither projection nor such a coordinate change is selected here.

## Necessary later gates

A future concrete inner blend requires its own algebraic reference/source identity, full constrained-20 actual symbol including oblique normals and all coincident speeds, tiny-lapse cancellation tests, and differentiated variable-coefficient checks. An outer live-field extension additionally requires complete 4D source conventions, mass-corrected stationary asymptotics, both gauge boundary-value rates and null/trace/shear jet tangencies, with no stronger falloffs imposed without a preserved-domain derivation. Finite-Omega principal completeness does not certify nonlinear pole closure.

The initial BH construction must independently retain the correct mass, constraints, positive lapse, regular throat transition and suitable consumed jets while continuing to use Minkowski reference sources. Subsequent acceptance requires the actual three-dimensional angular finite-pulse Minkowski gate first, then the authorized wormhole-to-trumpet evolution with resolved inner scales, angular perturbations, invariant/mass/horizon diagnostics, bounded finite conformal fields and constraints, and numerical refinement. Existing spherical analytical calibration and instantaneous data checks cannot replace either evolution requirement. No such new run is performed or admitted by this note.

Source IDs, exact local hashes, literature versions/sections, current HEAD and the source-only scope are recorded in `source-inventory.json` and `receipt.json`. No executable scientific source is part of this package.
