# Independent read-only review of the total-J basis

No mathematical correction is required in the reviewed basis or conversion. This review inspected existing source and saved mathematical test receipts only. It did not compile a scientific kernel, regenerate the basis, run a mathematical/scientific test, or edit the external basis tree. Exact reviewed identities are pinned in `receipt.json`.

The public API matches the held contract: `EvaluateBasis<T>(J,spin,L,x,WJet<T>{W,W_rho,W_rhorho})` returns scalar/vector/full-row-major symmetric STF tensor Cartesian value, gradient and Hessian. The generated C++ API is real m=0. All 69 all-m records are retained separately with exact polynomial coefficients and real/imaginary coefficients, permitting independent m=1 and J=2,m=2 validation after a later release.

The source builds regular solid harmonics from derivative-Legendre polynomials. Their residual radial powers are asserted even and nonnegative and become powers of rho=x.x. Negative m uses the Condon--Shortley conjugacy relation. Constant Cartesian spin vectors and CG-coupled STF spin tensors give the declared spin representations; the phase i^(L+s-J) is consistent with real m=0 and the stored conjugacy convention. The checker independently applies differential total-rotation generators, rather than accepting CG labels alone. Its saved report records 69 checks each for normalization, J2, Jz, conjugacy, parity and homogeneity, plus 35 STF checks.

The orbital triangle rule gives J0 scalar L0, vector L1, tensor L2. Four scalar families plus two copies of each vector/tensor family yield 8 independent amplitudes. J1 has scalar L1, vector L0/1/2 and tensor L1/2/3, yielding 16; J2 yields 20. The metric trace tensor I/sqrt(3) and full Frobenius STF normalization are consistent. Lambda may be treated as spin1 under the constant Cartesian orthogonal rotations used here; this does not assert that it is a vector under arbitrary coordinate changes.

Multiplying each homogeneous solid polynomial by smooth W_L(rho) supplies a smooth Cartesian origin representation. `MonomialDerivative` avoids negative exponents and returns zero when a derivative exceeds a monomial degree. The evaluator has no radial division, square-root radial derivative or singular origin branch. Its Hessian correctly uses W_i=2x_i W_rho and W_ij=2delta_ij W_rho+4x_i x_j W_rhorho, including both cross-product derivative terms. The saved standalone tests include the origin and W=1,rho,rho^2,1+rho+rho^2. Positive-radius angular-fit ranks must still not be extended to r=0, where higher-L value columns vanish.

The reference conversion is also correct. Its cyclic cofactor construction gives the transpose-cofactor matrix inverse and differentiates the determinant/inverse through full jets. With g=chi*bar-gamma,

    delta chi = -(chi/3) * bar-gamma^-1:H,
    delta g = chi*H + bar-gamma*delta chi

gives g^-1:delta g=0. Products, scalar inverse and matrix inverse retain the supplied first and second coefficient derivatives. `ConvertA` raises both reference A indices with the conformal inverse and returns

    delta A = T + (g/3)*(Aref^up:delta g - g^-1:T),

so g^-1:delta A=Aref^up:delta g. Omitting the final background-A coupling would be incorrect. The conversion test independently differentiates symbolic expressions for a nonconstant anisotropic SPD background and nonzero reference A; its saved residual is about 9.30e-16 and its nonzero trace coupling check passes. The actual future consumer must supply valid reference jets: A derivatives through first order, metric/chi derivatives through second order. Unavailable higher A derivatives cannot be interpreted as physical zeros merely because the generic structure has a Hessian slot.

Integration notes, without requesting a basis edit:

- Read `channel_layouts` as the authoritative amplitude order. Its scalar order is alpha, metric_trace, P, Theta_phys. The future bridge needs explicit adapters to the distinct frozen local-helper and native raw22/free20 orders.
- The conversion helper is mathematical and does not enforce SPD, reference consistency or finite-Omega admission. Those checks remain hard gates in the actual-kernel bridge.
- The generated m=0 coefficients are rounded doubles. Saved exact all-m coefficients and independent-angle/m checks are required for the later numerical closure gate; a finite set of basis-only ranks is insufficient.
- No actual C0/spatial-norm kernel, angular operator closure, radial discretization, origin equation, scri boundary treatment, propagator or stability result is admitted by this review.

The held recipe and its double-only native-wrapper negative control remain applicable. This review recommends freezing the reviewed mathematical basis; only root may release the subsequent narrowly scoped local coefficient prototype.
