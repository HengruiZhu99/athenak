This is a source-only derivation for a future Einstein-sector pure-coordinate gauge oracle on the same stationary layer reference and C0 physical-P/spatial-norm gauge. It admits no compile, radial matrix, eigenanalysis, propagation, boundary prescription or native change. Every formula below uses an active perturbation delta g4 = Lie_xi g4 and the code's convention Kij = -Lie_n gammaij/2.

1. Fixed external conformal factor and ADM lift

Write the physical stationary ADM fields as a = alpha/Omega, gamma = gtilde/(Omega^2 chi), beta = beta_ref and Kij = Aij/(Omega chi) + gammaij K/3. Here the stored trace is P = K - 2 Theta, the reference has Theta = Z = 0, and its K is the physical trace. The symbol a in this document is the physical lapse, not the curvature-radius input a_input=.5. All background fields below are evaluated on the unchanged wide reference S=1, a_input=.5, geometry cutoffs .05/.95.

Let xi = T partial_t + X^i partial_i and define sigma = Tdot - beta^i partial_i T and psi = X^i partial_i Omega/Omega. Since Omega is fixed externally,

    delta barg4 = Lie_xi barg4 - 2 psi barg4.

The physical and Penrose ADM variations are therefore

    delta alpha = alpha sigma + X^i partial_i alpha - alpha psi,
    delta beta^i = Xdot^i + [X,beta]^i + beta^i sigma
                   - alpha^2 bargamma^{ij} partial_j T,
    h_phys_ij = Lie_X gamma_ij + beta_phys_i partial_j T
                                      + beta_phys_j partial_i T,
    hbar_ij = Lie_X bargamma_ij - 2 psi bargamma_ij
                         + betabar_i partial_j T + betabar_j partial_i T.

Here [X,beta]^i = X^j partial_j beta^i - beta^j partial_j X^i, beta_phys_i = gamma_ij beta^j, betabar_i = bargamma_ij beta^j, and hbar = Omega^2 h_phys. Tdot and Xdot are independent coordinate-velocity fields; they must not be replaced with beta-advection derivatives.

For the curvature, set N=a T and Y=X+beta T. The vacuum normal-deformation formula gives

    k_ij = Lie_Y K_ij - D_i D_j N
               + N (R_ij + K K_ij - 2 K_ik K^k_j).

Using stationarity of the reference physical ADM equations gives the equivalent expression

    k_ij = Lie_X K_ij + K_ki beta^k partial_j T
                         + K_kj beta^k partial_i T
               - a D_i D_j T - (partial_i a)(partial_j T)
                               - (partial_j a)(partial_i T).

All D and R here belong to the physical spatial metric. The simplified expression is a stationary-reference identity, not a general live-background formula. A future gate must compare both expressions, including the reference stationarity residual, instead of assuming the identity from a numerically stationary point.

The scalar trace variation includes the inverse-metric term:

    k = delta K = gamma^{ij} k_ij - K_phys^{ij} h_phys_ij.

No coordinate velocity enters h_phys or k_ij except through the slice data T,X; this is consistent with the Minkowski check below.

2. Complete stored-variable lift

For fixed Omega, set

    delta Theta = 0,   delta P = k,
    delta chi = -chi bargamma^{ij} hbar_ij/3,
    delta gtilde_ij = chi hbar_ij + gtilde_ij delta chi/chi,
    delta Aij = Omega chi [k_ij - (K/3) h_phys_ij - gamma_ij k/3]
                                                   + (delta chi/chi) Aref_ij.

These formulas imply gtilde^{ij} delta gtilde_ij=0 and

    gtilde^{ij} delta Aij - Aref^{ij} delta gtilde_ij = 0,

where Aref is raised twice using gtilde^{-1}. The stored A is not the unscaled physical tracefree curvature. This conversion is exactly the inverse of ToPhysicalADM in athenak_bridge.hpp.

Use the Einstein connection lift delta Lambda^i = delta(gtilde^{jk} Gamma^i_jk). For a determinant-one reference and its determinant-tangent variation, an equivalent Cartesian formula is

    delta Lambda^i = partial_j(gtilde^{ia} delta gtilde_ab gtilde^{bj}).

The general contracted-connection variation and this determinant identity must be evaluated independently in a future binding gate. Lambda is not supplied as an independent coordinate field, and no differential-constraint projection is performed after the lift. It follows that delta Z_i=0 exactly.

The physical Hamiltonian and momentum constraints vanish under this complete diffeomorphism lift because the reference satisfies the vacuum Einstein equations. With Theta=Z=0 the linear constraint vector is identically [H,M_cov xyz,Z_cov xyz,Theta_phys]=0. This statement concerns the analytic complete lift; pointwise numerical residuals and reference roundoff still need measurement. It is not a claim about the finite-dimensional Riesz projection, any Cartesian ghost extension, or arbitrary independently interpolated coordinate components.

At a stationary reference the geometric time derivative of the lift is the same geometric lift evaluated at (Tdot,Xdot). The lapse and shift time derivatives additionally contain Tddot,Xddot. Thus a future point oracle can compare the complete raw22 actual geometry tangent to the coordinate-kinematic tangent without projecting output normals away.

3. Acceleration from the unchanged actual gauge rows

Let F_alpha and F_beta be the exact linearized actual gauge rows evaluated on the complete coordinate lift. Differentiating the ADM gauge variations at fixed stationary reference yields

    Tddot = F_alpha/alpha + beta^i partial_i Tdot
                               - Xdot^i partial_i log(a),
    Xddot^i = F_beta^i - [Xdot,beta]^i
                       - beta^i(Tddot-beta^j partial_j Tdot)
                       + alpha^2 bargamma^{ij} partial_j Tdot.

This solves for coordinate accelerations; no lapse or shift row is imposed by hand after solving it. The direct generic-dual GenericGauge(norm=true) baseline is authoritative. Its input flags remain physical_trace_lapse=true, preferred_source=false, xi=1/a_input=2, with C0 k2=0 and runtime alpha*kappa1=10. The double-only native injection wrapper must not be used as a dual implementation.

For a separately derived Jacobian attribution check, let W be the gauge cutoff .45/.85, e=1-W, f2=alpha^2+2 e alpha, mu=3e/8+W, nu=1.5W, eta_shift=W, ea=W, ec=W/2. The stationary-reference lapse row is

    F_alpha = beta.grad(delta alpha) + delta beta.grad(alpha) - nu delta alpha
        - [f2 delta P + 2 W xi alpha delta alpha
             + W (alpha delta beta + beta delta alpha).grad(Omega)]/Omega.

The original regular shift row is

    F_beta^i = beta.grad(delta beta^i) + delta beta.grad(beta^i)
             + mu alpha^2 chi delta Lambda^i - eta_shift delta beta^i
             + alpha^2 chi gtilde^{ij}
                    [ec partial_j(delta chi/chi) - ea partial_j(delta alpha/alpha)],

and the additional spatial-norm pole is

    - eta_norm W [delta beta^i + C n^i delta G/Ghat]/Omega,
    eta_norm=6, C=2/3, n=-grad(Omega)/|grad(Omega)|,
    Ghat=chi gtilde^{ij} Omega_i Omega_j,
    delta G=[delta chi gtilde^{ij}
             -chi gtilde^{ia} delta gtilde_ab gtilde^{bj}] Omega_i Omega_j.

The pole beta contribution must be assembled exactly once. Coefficients multiplying zero reference deviations contribute no additional live coefficient-variation term in this linearization. Outside the core Ghat is checked positive on W support; no division by Ghat is evaluated in the exact W=0 branch. These formulas are source-only attribution formulas until compared with the frozen actual generic-dual helper and independent double directional differences.

4. Exact flat-core J0 envelope oracle

Use Cartesian coordinate fields T=tau(rho), X^i=x^i zeta(rho), Tdot=v(rho), Xdot^i=x^i w(rho), rho=x.x. These are regular solid-polynomial envelopes over the entire core, with four independent fields tau,zeta,v,w. A later adapter to the frozen J0 CG normalization must be explicit; no extra i phase or hidden spherical-harmonic factor is assumed here.

In the exact core Omega=alpha=chi=1, beta=K=A=0, gamma=gtilde=I. Define Sij=xi xj-rho deltaij/3. The lift becomes

    delta alpha=v,      delta beta^i=xi(w-2 tau_rho),
    hbar_ij=2 zeta deltaij+4 zeta_rho xi xj,
    delta chi=-2 zeta-(4/3)rho zeta_rho,
    delta gtilde_ij=4 zeta_rho Sij,
    delta P=-6 tau_rho-4rho tau_rhorho,
    delta Aij=-4 tau_rhorho Sij,
    delta Lambda^i=(8/3)xi(5 zeta_rho+2rho zeta_rhorho),
    delta Theta=0.

The core physical-P lapse and shift rows are F_alpha=-3 delta P and F_beta=3 delta Lambda/8. Therefore the four independent envelope equations are

    tau_t=v,             zeta_t=w,
    v_t=18 tau_rho+12rho tau_rhorho,
    w_t=2 v_rho+5 zeta_rho+2rho zeta_rhorho.

There are no r or rho denominators and no imposed cross-L conditions. Core kappa10 damping vanishes on this exact Einstein sector: Theta=0 and Lambda-Gamma=0. The exact Cartesian identities H=partial_i partial_j hbar_ij-Delta tr(hbar)=0 and M_i=partial_j k_ji-partial_i k=0 provide an independent flat-core oracle, including at the origin. Constant tau is a static time-translation coordinate mode; constant zeta is a spatial dilation coordinate mode with a nonzero metric trace variation. Neither is silently removed.

5. Derivative requirements and unresolved source scope

The existing reference adapter supplies only the consumed second configuration jets and first A/P jets; its Aref second derivatives and higher metric derivatives are not populated. They cannot be read as zero. The complete coordinate lift needed by the actual point kernel requires reference alpha/beta/bargamma/Omega through third Cartesian derivative and physical Kij through second derivative (equivalently stored A and P through second), plus coordinate T and X through third derivative and coordinate velocities Tdot/Xdot through second derivative. A complete kinematic output-constraint oracle also needs velocities through third derivative.

For the production radial geometry, the third derivative of L=Omega-r Omega_r requires Omega through fourth radial derivative. Consequently a new analytic radial jet backend must include the fourth cutoff/Omega derivative, not differentiate a Radial2 object with missing higher entries. Exact core/outer branches and the origin must be handled analytically; any Taylor/AD evaluation must preserve those branches and finite tail behavior. A CPU arbitrary-precision independent oracle can validate the new reference jets, but no such implementation or execution is admitted by this source-only preparation.

Differentiating the full actual momentum RHS twice to obtain an independent exact C_ref[F_actual] would demand still higher input/reference jets. This preparation does not claim that task solved. The first future source gate instead compares raw actual geometry tangent values with the analytically constraint-free coordinate-kinematic tangent at held-out points; complete input constraints, both A/metric normal identities, gauge attribution and independently convergent directional differences are separate gates. Any subsequent actual source-jet/constraint-rate readback needs its own stated jet order or labeled convergent finite differences. It must not reuse analytic kinematic output jets as if they independently differentiated the actual kernel.

All statements are strictly finite Omega. Generic X need not preserve a bounded fixed-Omega field at scri because psi contains 1/Omega. No Xradial=O(Omega) falloff, coordinate boundary condition, finite-rb SAT, CPBC, physical energy estimate or uniform stability conclusion is chosen here.
