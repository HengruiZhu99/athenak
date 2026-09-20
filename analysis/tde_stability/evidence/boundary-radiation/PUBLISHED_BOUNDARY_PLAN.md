# Published Z4c boundary comparison and the next discriminating test

This is a derivation/design audit, not a stability result for a new AthenaK boundary. No AthenaK source or production job was changed by this audit. The actual v2 radiation runs still fail; the simpler direct allocation below also retains an unstable oblique mode in the independent frozen finite-difference model.

## Primary-source distinction

[Hilditch et al., 1212.2901](https://arxiv.org/pdf/1212.2901), Eqs. 17–25, assigns the gauge boundary equations to Khat/Gamma, the constraint equations to Theta and the normal components of A, and radiation control to tangential tracefree A. Its implementation updates the metric fields with their volume equations. The normal is physical-metric unit length; projections are applied after differentiation. Its special longitudinal shift condition assumes initially vanishing normal shift, which our stationary trumpet does not satisfy. This paper does not justify copying that condition unchanged into the adapted gauge.

[Hilditch et al., 1609.06925](https://arxiv.org/pdf/1609.06925), Section II D, describes a related formulation, not our exact Z4c: Eq. 34 omits the Theta-gradient term in Gamma evolution. The corresponding constraint speed and A boundary equations therefore differ. Its low-order implementation is not covered by the paper's high-order boundary-stability proof, and the discussion explicitly retains the nonlinear constraint terms while simplifying some gauge terms. We must not transfer either the proof or the modified-formulation equations to the current solver.

## A precise comparison in the frozen conformally flat limit

Use a constant conformal metric equal to the identity, arbitrary constant positive alpha/chi, and a constant shift beta. Let n be a conformal-unit outward normal and A a tangential index. These formulas retain the current volume convention with Gamma damping `-2*sigma*Q`, where `sigma=alpha*kappa1`. Define

```
Q_i = Gamma_i - partial_j h_ij
c = alpha*sqrt(chi)
F_i = Q_i,t + (c*n^j - beta^j)*partial_j Q_i
F_Theta = Theta_t + (c*n^j-beta^j)*partial_j Theta
```

All time derivatives on the right are the unmodified volume derivatives. There is no finite-radius falloff term in this frozen planar comparison. Algebraically subtracting the volume A equation from the constraint A equations of 1212.2901 gives

```
Delta Theta_t = -F_Theta
Delta A_nn,t  = -sqrt(chi)*F_n
                + alpha*chi/3 * partial_i Q_i
                - sqrt(chi)*sigma*Q_n
Delta A_nA,t  = -sqrt(chi)/2 * (F_A + sigma*Q_A)
```

The normal and tangential equations are different. In particular, there is no common isotropic `-F/2` rule and no factor `beta_n+c` multiplying these direct corrections. The extra damping terms follow from the published A equations together with the actual `-2*sigma*Q` volume damping; silently dropping them changes the boundary equation. The divergence in the scalar correction is equally important.

Two independent derivations agree: direct flat-space subtraction of the published A equations, and rescaling physical orthonormal coordinates/time, `x_phys=x/sqrt(chi)` and `tau=alpha*t`. This establishes only the frozen normalization, not a nonlinear curved-background identity.

The independent gauge rates can still be imposed after prescribing Theta. In the current scalar characteristic notation `lp`, `rate` is the immutable volume incoming rate:

```
Delta Khat = -rate_lapse/lp[lapse,Khat]
Delta Theta = -F_Theta
Delta Gamma_n = (-rate_shift
                 -lp[shift,Khat]*Delta Khat
                 -lp[shift,Theta]*Delta Theta) / lp[shift,Gamma_n]
```

Then apply the direct A correction. Transverse Gamma gauge corrections and the two TT rates remain independent. This is a different assignment from solving all four scalar incoming characteristic rates together, because the old coupled solve lets its Gamma correction change Theta even when `F_Theta=0`.

Changing A's RHS does not algebraically set Q_t to a prescribed value in the same RK stage: Q_t uses Gamma_t and derivatives of the metric RHS. Instead, these are differential boundary equations whose smooth compatibility with the volume equations supplies a boundary condition on the constraints. An implementation must not report a same-stage `F=0` guarantee that it does not enforce.

## Independent finite-difference test of that assignment

The `mode='direct'` branch in `fd-symbol/fourier_boundary.py` implements exactly the frozen equations above. I reviewed its normal signs, scalar/vector coefficients, damping product, and gauge completion after Theta; no translation error was found. It retains the actual sixth-order normal/upwind volume operators, polynomial ghosts, KO dissipation, fourth-order inner metric-connection derivative and second-order outer transport used in the v2 comparison.

At the sampled boundary-face background, `alpha=0.65317697`, `chi=0.42664016`, `beta=(0.22553665,-0.01503578,-0.01503578)`, `dx=0.25`, the cubic-ghost mode with tangential phase `k_y*dx=pi/2` has growth about **0.741/M** with direct allocation, compared with **1.213/M** for the original radiation p-map. The branch survives. This is a negative test; neither direct allocation nor its literal frozen published normalization is a demonstrated cure.

The matrix investigation also isolates nonzero normal shift as a major contributor: removing the shift nearly removes this particular fast branch, while retaining only its normal component reproduces it. That is evidence for a shifted multidimensional boundary-closure defect, not proof that the physical trumpet gauge is intrinsically unstable. It also explains why a spherical spectral pilot or a flat zero-shift face test was insufficient validation.

## Complete frozen gauge and radiation design

Merely retaining tangential terms in the old incoming-rate solve is not the complete published gauge/radiation construction. A more discriminating replacement can be derived directly from outgoing gauge waves, including the Theta coupling in our actual Gamma equation. The following is an independent constant-coefficient derivation, subsequently checked by both other audit agents.

Set `G=2` for the current G2 shift driver, and

```
D0 = partial_t - beta^j partial_j
vL^2 = 2*alpha*chi
vST^2 = G
vSL^2 = 4*G/3
c^2 = alpha^2*chi.
```

At principal order the gauge/constraint volume system gives

```
D0^2 Khat = vL^2 * Laplacian(Khat)
D0^2 Theta = c^2 * Laplacian(Theta)
D0^2 div(beta) = vSL^2 * Laplacian(div(beta))
                - alpha*vSL^2 * Laplacian(Khat)
                - alpha*vSL^2/2 * Laplacian(Theta).
```

Consequently `W=div(beta)+a*Khat+b*Theta` is a longitudinal gauge wave with speed vSL, where

```
a = alpha*vSL^2/(vL^2-vSL^2)
b = alpha*vSL^2/(2*(c^2-vSL^2)).
```

Impose `(D0+vSL*Dn)W=0`, solve for `Dnn beta_n`, and substitute in the unmodified normal Gamma PDE. The resulting direct boundary equation is

```
Gamma_n,t = beta^j*Dj Gamma_n - vSL*div(Gamma)
            + Laplacian_T(beta_n) - Dn div_T(beta_T)
            - 4*alpha/[3*(vL^2-vSL^2)]
                * [vSL*D0 Khat + vL^2*Dn Khat]
            - 2*alpha/[3*(c^2-vSL^2)]
                * [vSL*D0 Theta + c^2*Dn Theta].
```

The last line is required by our Theta-coupled Gamma equation; it is absent from the modified formulation in 1609.06925. Substitute the already chosen boundary Khat and Theta rates for their D0 derivatives. No assumption `beta_n=0` entered this derivation; beta is constant and D0 commutes with spatial derivatives. Exclude coincident gauge/constraint speeds before using these rational formulas.

For the transverse gauge, impose outgoing propagation of `curl_nA(beta)=Dn beta_A-DA beta_n`. Eliminating `Dnn beta_A` gives

```
Gamma_A,t = beta^j*Dj Gamma_A
            - vST*(Dn Gamma_A-DA Gamma_n)
            + Laplacian_T(beta_A)
            + (4/3)*DA Dn beta_n + (1/3)*DA div_T(beta_T)
            - (2*alpha/3)*DA(2*Khat+Theta).
```

These are full frozen multidimensional gauge relations, not zero incoming characteristic time rates. The independent lapse and constraint choices used with them are

```
Khat_t = beta^j*Dj Khat - vL*Dn Khat - chi*Laplacian_T(alpha)
Theta_t = beta^j*Dj Theta - c*Dn Theta.
```

The displayed lapse tangential coefficient corresponds to the 1212.2901 choice. The alternative squared-outgoing lapse operator gives `-chi/2*Laplacian_T(alpha)`, matching the later paper's low-order lapse choice; compare this explicitly rather than silently mixing the two.

Finally the frozen TT projection of the direct radiation equation is

```
A_AB^TT,t = beta^j*Dj A_AB^TT
             - c*[Dn A_AB - D_(A A_B)n]^TT
             - chi*[DA DB alpha]^TT.
```

This differs from freezing the two first-order TT incoming time rates. Together the displayed gauge, direct physical Theta/A, and TT equations are the complete next frozen boundary matrix to test. They do not assert full nonlinear stability.

### Driver and constraint damping are separate choices

The decoupled wave identities above are principal identities. If the actual `D0 beta=G*Gamma-eta*beta` is retained and the chosen gauge boundary target remains literally outgoing W/curl, eliminating the normal second derivatives adds

```
Gamma_n,t: +(4*eta/(3*vSL))*div(beta)
Gamma_A,t: +(eta/vST)*(Dn beta_A-DA beta_n).
```

The original `-2*sigma*Q` Gamma source also remains when this is a substitution in the volume PDE. Omitting the eta additions instead chooses a damped gauge boundary target. Either experiment must state that choice. Constraint damping, lapse-residual damping, and KO contributions cannot be silently counted as part of an undamped principal-wave proof. Localization is testing the full frozen matrix with explicitly distinguished principal/source cases.

The first complete principal gauge/TT comparison is also negative: the sampled cubic face still gives growth about1.371/M, and matching both helper derivatives to the volume D6 gives about1.363/M. Removing the lower-order damping does not remove the branch. The matrix's direct formulas, normal orientation, vector terms and TT projection have been independently reviewed. One source-completion qualification was found: replacing the principal gauge part while preserving KO/upwind/source terms in Khat is different from substituting that final Khat rate into the longitudinal Gamma equation. Likewise a D6 Theta rate is not the final D2 direct-transport rate. The additional `paper_actual_d0=True` comparison substitutes the actually selected Khat and Theta rates and still gives growth **1.37496872/M** with D4/D2 and **1.36315446/M** with D6/D6 (`fd-symbol/paper-actual-d0.json`). Thus correcting that completion does not cure the sampled branch. The initial principal comparison must remain separately labeled; none of these tests establishes a complete nonlinear published implementation.

## Nonlinear implementation limits and validation

One intermediate gauge discriminator compares each old incoming `C_t=0` condition against

```
C_t(boundary) = C_t(volume) - lambda_in * partial_n C.
```

This retains the volume tangential and lower-order rate, but is not equivalent to the complete outgoing-W/curl and TT construction above. A failed intermediate comparison does not exhaust that concrete alternative. All comparisons use the actual background-adapted G2 eigenstructure; its gauge advection distinction becomes important again with variable coefficients and nonzero perturbations.

For a later nonlinear implementation, a defensible route is to construct a complete boundary operator on the actual full fields and the same operator on the stationary reference, then subtract. Constraint quantities, the physical-metric connection, all normal/tangential derivatives and the selected damping product must use consistent definitions. For the adapted gauge, derive its boundary equations from its actual residual lapse/shift equations; do not copy the normal-shift assumption in Eq. 20 of 1212.2901. Preserve metric evolution and the independent gauge/radiation boundary count. Do not differentiate a projected tensor while treating its variable frame as constant unless that is the explicitly chosen principal approximation.

Simply extending the frozen corrections with local alpha/chi and a reference divergence is a homogeneous constraint addition and can preserve zero by construction, but it is **not** the complete nonlinear published CPBC. Missing background/frame derivatives, use of evolved versus metric-defined Ricci, the finite-radius transport weight, and a physical versus conformal constraint covector all require explicit derivation. The change from Eq. 22/24 to Eq. 41/42 of the two papers is a warning against dropping apparently small terms.

Required tests before promotion: exact zero through RK and MPI; manufactured nonlinear geometry and genuine constraint/gauge perturbations; identity checks using the actual metric-defined Gamma functional; shifted oblique full-spectrum and transient tests with refinement; then bounded 3D vacuum controls. Current positive growth fails that gate. No spin, atmosphere or star validation follows from this report.

The continuum half-space discriminator was subsequently performed for the original v2 combination of physical constraint radiation and zero-rate gauge/TT data. An ordered-Schur stable-subspace calculation finds a positive root at1.23071163668/M for ky=2*pi with the selected damping. The reconstructed decaying profile satisfies the original twenty-field equations to7.84e-14 relative error and the unscaled boundary equations to1.66e-16; Theta and Q are around1e-14. Undamped roots scale linearly with ky. Thus this particular fast prototype defect already exists in the frozen continuum boundary problem; it is not solely a polynomial-ghost or D2 closure effect. See `continuum-halfspace/PEER_REVIEW.md` and its validated data.

The next step is to derive a complete gauge/radiation boundary operator that also treats constraint-satisfying oblique perturbations consistently, then test its continuum half-space boundary symbol before another discrete implementation. The current metric characteristic TT condition is not automatically a physical Weyl-radiation condition for an oblique gauge disturbance. The complete frozen published-derived gauge/TT equations above are an explicit candidate to assess at the continuum level; their negative finite-difference tests remain valid and must not be erased. A compatible discrete closure should follow the continuum result, with an explicit normal-mode or energy argument where available.

The new fast prototype mode, of order1/M, is distinct from the original long-run residual growth of order0.02–0.04/M. Curing the prototype's shifted oblique mode would only remove a newly introduced problem. It would not by itself validate the old production run, solve its slow mode, or pass the vacuum gate for spinning or matter evolution.
