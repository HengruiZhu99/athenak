# Boundary allocation and gauge/tensor audit

Read-only derivation, 2026-09-20. No v3 C++ option has been implemented: the frozen oblique matrix test of the proposed direct allocation retains a rapid positive branch (reported γ≈.741/M, compared with v2≈1.213/M). This is not sufficient evidence to add another source path.

Primary reference: [Hilditch et al., Compact binary evolutions with the Z4c formulation, arXiv1212.2901](https://arxiv.org/pdf/1212.2901), equations5 and17–25. The following flat-coefficient reductions are independently checked against the actual original Z4c volume equations and the current characteristic rows. They are not a curved nonlinear well-posedness derivation.

Use α=χ=1, β=0, fixed Cartesian outward n, tangential indices A/B, Q=Γ_evolved−Γ_metric, and the actual original damping Q_t=2divA−4DKhat/3−2DΘ/3−2κQ. Write FQ=Q_t+DnQ (the optional areal decay model is a separate extension).

## Literal published constraint allocation

Eq21 prescribes Θ_t=−DnΘ. Subtracting the actual volume A equations from Eqs22/24 gives

    δA_nn = −FQ_n + (1/3)divQ − κQ_n
    δA_nA = −(1/2)(FQ_A + κQ_A).

In particular, Eq24's κ/2 inside its bracket is not a typo caused by assuming the wrong volume damping: paperEq5 also has−2ακQ. Its net boundary correction differs from naive−FQ_A/2 by−κQ_A/2. Scalar allocation has an additional divergence term and a different coefficient from vector allocation.

For frozen conformal-flat coefficients, transform using x_phys=x/√χ, τ=αt, A_orth=Ã, and Q_phys=√χQ_coord. This gives the normalization check

    δÃ_nn = −√χFQ_n + (αχ/3)div_coordQ − σ√χQ_n
    δÃ_nA = −(√χ/2)(FQ_A + σQ_A),   σ=ακ.

This frozen scaling does not supply the missing variable-coefficient, moving-frame or background-coupling terms needed for a literal nonlinear implementation. Using the existing curved physical FΘ/FQ model with these formulas would be a new experimental extension.

Prescribing δΘ first would require a different gauge solve from the old coupled four-row solve: δK=r_lapse/lp00; δΓ_n=(r_shift−lp10δK−lp11δΘ)/lp13. Direct A allocation is independent. Diagnostics must measure these targets, rather than the old incoming constraint characteristic targets.

## Tangential-principal gauge experiment

For lapse driver α_t=−L Khat, the current incoming row is w=−√L Khat+Dnα. Its bulk tangential principal rate is T=√L Δ_Tα. Imposing w_t=T gives

    Khat_t = −√L DnKhat − Δ_Tα,

which matches the principal part of paperEq17. Existing zero_rate instead omits the tangential Laplacian. A combined tangential-gauge/physical-constraint experiment is therefore meaningfully different from the already failed all-tangential test.

For transverse shift β_A,t=GΓ_A, the current incoming gauge row is w_A=√GΓ_A+Dnβ_A. The tangential-principal target gives

    Γ_A,t = −√G DnΓ_A + Δ_Tβ_A
             + (1/3)D_A Dnβ_n + (1/3)D_A D_Bβ_B
             − (2/3)D_A(2Khat+Θ).

This is **not** the complete principal part of published Eq23, which additionally has +√G D_AΓ_n and increases the mixed derivative coefficient from1/3 to4/3. Calling the combined experiment the published gauge condition would be inaccurate.

## Tangential-principal TT experiment

Using the current row−2A_AB^TF+Dn h_AB^TF, the tangential-principal target yields

    A_AB,t^TF = −Dn A_AB^TF − (1/2)Δ_T h_AB^TF
                + [D_(AΓ_B)+(1/2)D_A D_Bχ−D_A D_Bα]^TF.

Published Eq25 instead reduces to

    A_AB,t^TF = −Dn A_AB^TF
                + [D_(A A_B)n−D_A D_Bα]^TF.

The former retains the volume transverse-curvature terms; the latter controls incoming gravitational radiation. They are different boundary models. The measured v2 fast direction has very small TT polarization, so the scalar/vector gauge–constraint coupling is more directly implicated, but that observation is not a proof that the TT closure is harmless.

## Nonzero-normal-shift completion from the actual gauge principal PDE

This independently agrees with the gauge_sponge derivation. Freeze the background coefficients, set D0=∂t−β·D (including nonzero β_n), and define

    c²=α²χ, vL²=2αχ, vSL²=4G/3, vST²=G.

The actual Γ equation retains the gradient of Θ. With shift driver B_t=β·DB+GΓ at principal order,

    D0²(divB) = vSL² Δ(divB) − αvSL² ΔKhat − (α/2)vSL² ΔΘ.

The lapse and constraint wave equations are D0²Khat=vL²ΔKhat and D0²Θ=c²ΔΘ. Therefore

    W = divB + aKhat + bΘ
    a = αvSL²/(vL²−vSL²)
    b = αvSL²/[2(c²−vSL²)]

satisfies D0²W=vSL²ΔW for noncoincident gauge/light cones. This is a principal-order identity; it does not assume β_n=0. Applying (D0+vSLDn)W=0, solving for DnnB_n and substituting into the Γ_n PDE gives

    Γ_n,t = β·DΓ_n − vSL divΓ + Δ_TB_n − Dn div_TB_T
      −4α/[3(vL²−vSL²)](vSL D0Khat + vL² DnKhat)
      −2α/[3(c²−vSL²)](vSL D0Θ + c² DnΘ).

D0Khat and D0Θ must use the prescribed boundary evolution, not an inconsistent earlier volume value. The Θ terms cannot be omitted by citing a formulation whose Γ equation removed the Θ gradient.

The transverse completion follows directly from outgoing curlB:

    (D0+vSTDn)(DnB_A−D_AB_n)=0.

This yields the full mixed-derivative/Γ_n terms in Eq23 quoted above, with vST and the explicit β·DΓ_A advection. It explains precisely why tangential-principal correction of the old componentwise gauge row is not the same condition.

### Lower-order source awareness

Actual β evolution also contains−ηB. A literal outgoing W/curl target with this term retained adds

    Γ_n,t: +(4η/(3vSL))divB
    Γ_A,t: +(η/vST)(DnB_A−D_AB_n).

If only the normal second derivative of B is eliminated from the actual Γ volume PDE, its original additive damping−2σQ must also remain in the Γ boundary equation. Dropping those terms instead chooses a different damped gauge boundary model. With nonzero κ/η the W equation is forced/damped, not the homogeneous wave equation stated for the principal part. This source-aware completion is mathematically defined but still needs the full20-field oblique matrix test and then real3D discrete validation.
