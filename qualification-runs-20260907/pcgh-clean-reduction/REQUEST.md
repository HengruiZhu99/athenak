# Goal-mode task: implement and qualify clean-reduction puncture-conformal GH in AthenaK

## 1. Objective and definition of completion

Develop a reference-free, unexcised, puncture-conformal Einstein evolution scheme with direct moving-puncture gauge and substantially better controlled spatial reduction/curl constraints than the existing PC-GH implementation. Use the existing collision-capable PC-GH lineage and its numerical infrastructure; do not start another unrelated reference-tetrad formulation.

My priorities, in order, are:

1. Correct Einstein evolution, puncture-compatible coefficients, and convergent physical fields and constraints.
2. A transparent nonlinear reduction/curl subsystem, controlled discrete error injection, and strong hyperbolicity on the stated positive-field domain.
3. Direct advective 1+log lapse and the explicitly analyzed Gamma-like shift, with no excision controller or prescribed singular reference frame.
4. High-order finite-difference/spectral-element compatibility outside a regularity-adapted puncture core.

Symmetric hyperbolicity is desirable but NOT a prerequisite. Do not spend the project trying to remove the known common-symmetrizer obstruction before performing useful numerical tests. Conversely, real characteristic speeds alone are not a strong-hyperbolicity proof.

This is an IMPLEMENTATION AND QUALIFICATION task, not another literature survey. Start with a read-only provenance audit, then implement in an isolated branch, run staged checks, and continue through mechanism-driven corrections. A failed gate blocks downstream promotion, not further diagnosis or affordable corrective work. Do not stop merely because the first candidate fails. Do not claim success merely because a run survives.

A successful endpoint is a documented, reproducible single-puncture qualification followed by the matched head-on binary with controlled post-merger errors and convergence. If resources or a demonstrated mathematical/numerical obstruction prevent that endpoint, leave a tested implementation, exact negative evidence, commits, and the smallest remaining blocker. Never invent missing evidence or report planned tests as executed.

## 2. Repository, starting branch, isolated workspaces, and provenance

Implementation repository:

    https://github.com/HengruiZhu99/athenak

START development from:

    origin/codex/pc-gh-gamma2-20260904

Last independently verified remote HEAD:

    62945657b4f2828a481abb5e7708e0e3e06dbd8d

Create a NEW work branch:

    codex/pcgh-clean-reduction-transfer-20260907

Do not develop on main, overwrite the source branch, reset another checkout, or modify an existing running calculation. Read AGENTS.md and applicable repository instructions first. Use an isolated clone or worktree and separate build/run directories. Fetch the source branch, record its actual remote SHA, and inspect any changes since the verified SHA before choosing and recording the exact base. Do not blindly revert legitimate newer work. If the requested new branch already exists, inspect it; use a documented unique suffix rather than resetting it.

Representative setup, adapting paths to the actual environment:

    git fetch origin codex/pc-gh-gamma2-20260904
    git rev-parse FETCH_HEAD
    git worktree add -b codex/pcgh-clean-reduction-transfer-20260907 \
        ../athenak-pcgh-clean-reduction FETCH_HEAD

A git worktree shares refs with its parent repository. Coordinate fetches and never move unrelated branches. Avoid executing old launch scripts until they have been inspected for hard-coded paths, device use, restart selection, and overwrite behavior.

Keep a separate READ-ONLY legacy-control checkout at the exact collision source:

    b81b44d658f3b81584e94ce79b92656c112ff908

Its immediate lineage is:

    codex/pc-gh-localization-horizon-20260902
    ff0de21305435f753715ba3a1aefba7537666e88
      -> 3cfbdb5e81dcaad0176e4e7b29011eed9e7af58c
      -> b81b44d658f3b81584e94ce79b92656c112ff908

Do NOT use codex/pc-gh-from-scratch-20260901 as the new implementation base. The older and newer 55-field layouts have different meanings; equal array lengths do not imply restart compatibility.

Mathematical candidate repository, READ-ONLY reference:

    https://github.com/HengruiZhu99/generalized_harmonic_with_puncture
    branch: research/intrinsic-pcgh-derivation-20260906
    pinned commit: 7ef9c61c0c2bd12a46b28f334a53a1aabcd1842e
    directory: candidate_20260906/

Read candidate.tex, every included section, analysis/, results/, and README.md. Run its existing analysis/run_all.py suite in an isolated environment. Use the complete pinned nonlinear point-jet oracle and characteristic analysis, not only the earlier prose summaries. Treat the derivation as an auditable candidate, not an unquestionable authority. Resolve discrepancies by first-principles calculation and independent tests, documenting corrections explicitly.

The local originating checkout, if accessible, is:

    /Users/hz0693/research/athenak-pcgh-localization-20260902

Use it only as an evidence source. Inspect its branch, dirty state, and manifests before copying anything. In particular, investigate these newer source identities if locally available:

    baseline source: 1e6b0612aed52509d1f255627ba75211a82497bd
    pre-patch HEAD: 80602ffc
    direct-lapse implementation: 5a9230e3
    frozen plan/provenance: d32130a2

These did not resolve through the previously available remote API. A cluster checkout may contain baseline-plus-patch source despite reporting an older HEAD. Record actual source manifests, diffs, executable hashes, compiler/backend, CMake options, input hashes, restart hashes, and mesh maps. Never identify the executable solely by a bare Git SHA.

Preserve later engineering fixes, including the CUDA host-this capture repair and restart tracker serialization. Verify legacy-mode equivalence against the collision-source oracle before calling a run an unchanged baseline.

## 3. Existing evidence you must read and preserve

Read, where present:

    docs/pc_gh_derivation.md
    docs/pc_gh_regular_extension.md
    docs/pc_gh_gamma2_audit.md
    docs/pc_gh_qualification_log.md
    docs/pc_gh_hybrid_projection.md
    analysis/pc_gh_regular_extension/
    analysis/pc_gh_bbh/
    qualification-runs-20260902/della-r128-t100-comparison/
    qualification-runs-20260902/amr-transfer-discriminator/
    qualification-runs-20260904/regular-extension/
    qualification-runs-20260905/hybrid/REPORT.md
    qualification-runs-20260905/r16-smr128/REPORT.md
    qualification-runs-20260905/r16-smr128/refinement-evidence/

Also read the supplied report, “Direct lapse-gradient qualification — completed with failure,” including its PLAN.md, REPRODUCE.md, decisions.json, comparison.json, source manifests, and available raw diagnostic inventories. Discover its actual location; do not invent a path.

Originating large evidence locations include:

    /scratch/gpfs/FPRETORI/hz0693/pcgh-z4c-gpu-r128/
    /scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317
    /scratch/gpfs/FPRETORI/hz0693/pcgh-r16-smr128-20260905-1640
    /scratch/gpfs/FPRETORI/hz0693/pcgh-regular-extension-20260904-64c8f90b

Report missing artifacts explicitly. A README listing raw data does not establish that those files are committed or available. Reuse compact derived products when sufficient; do not repeatedly parse multi-gigabyte monitors unnecessarily.

Established evidence to preserve:

- The legacy head-on source was b81b44d6. Its exact long input is inputs/z4c/twopuncture/bbh_headon_pcgh_cuda_r128_t100.athinput, NOT the smaller-domain bbh_headon_pcgh.athinput.
- That run used boundaries +/-128M, finest spacing M/16, FD6/RK3/CFL 0.2/KO 0.3, completed moving gauge, kappa=0, global GH projection, and global auxiliary projection. It produced a merger feature but developed sustained growth around 40M and failed at total coordinate time 73.79991M. It did not survive 70M after merger, and contaminated late data are not physical ringdown.
- Inspect the code to confirm timing: GH projection resets C_perp and Z every stage, while auxiliary projection is restricted to the final RK stage and followed by another exchange/prolongation. These are not the proposed localized-core/free-GH policy.
- In the saved one-step discriminator, final curl_Q was 4.82647e-6 with the Q reset and 5.89637e-5 with only Q projection disabled. Projection was early-time corrective. Interface injection is measured; a causal late projection-feedback instability was not established by that control.
- The later implemented advective extension has true primary advection and damping, but its subsidiary equations still contain J and dJ. It is NOT the intrinsic clean-reduction candidate below.
- Flat transport/damping and smooth-wave tests passed; coarse puncture screens did not establish inner convergence. Do not spend the project merely repeating flat positive results.
- The direct-lapse patch changed the common target to 2D(rho*w) and passed isolated tests, but failed the identical fine-core test at 5.167923M versus 5.187818M for baseline. Its convergence and binary gates were not run. The actual R16 rate was (1+15P)/M with radii (0.125,0.5)M, both projection switches OFF, and otherwise matched numerical settings.
- At common t=5M, corrected full-domain curl RMS was 0.04067021 versus 0.0375294 for baseline. Near termination, corrected curl_Q and curl_B maxima were approximately 4.38e6 and 2.95e6. This is not evidence of stabilization.
- Existing operation brackets include differences of maxima and sometimes stale ghosts. They are not a signed same-cell causal budget.

## 4. Scope, permissions, and experimental discipline

You may edit, build, test, commit, and push the NEW implementation branch. Begin read-only, but do not remain read-only after provenance is established. Keep the formulation repository unchanged unless a separate derivative branch is genuinely needed and explicitly documented.

Use only compute accounts, allocations, devices, and execution routes authorized for this environment. Check resource occupancy and scheduler/site rules. An idle login-node GPU or a historically permitted direct driver is not continuing permission. Use the scheduler where required. Never cancel unrelated jobs, modify their files, kill another agent's processes, or bypass an allocation limit. Give each new run a unique directory and ownership manifest.

Before remote evolution, record resource limits and planned tests. Use affordable algebraic/unit/one-step checks first. Do not launch a broad parameter sweep, duplicate expensive campaigns, or advance to a binary after a prerequisite failure. Preserve early-stop checkpoints and the first invalid state before any projection hides it. Terminate only your own run using documented health criteria.

Maintain short progress updates, an append-only execution log, and checkpoint commits. A failed candidate is an acceptable recorded result, not a reason to lower a threshold, discard full-domain data, change several parameters silently, or claim that surviving longer is success.

## 5. Mathematical target: one reference-free intrinsic conformal system

The long-term target is ONE continuum formulation, with an optional regularity-adapted numerical core. Do not join unrelated BSSN and Lindblom systems. Do not add a reference spacetime, singular-exponent controller, excision surface, or compulsory elliptic projection.

The following equations identify the intended candidate. The pinned full derivation fixes any remaining conventions; verify them independently before implementation.

### 5.1 Variables and geometry

Use G=c=1, signature (-,+,+,+), K_ij^phys = -(1/2) L_n gamma_ij, and

    D0 = partial_t - beta^i partial_i,
    w = sqrt(chi), alpha = rho*w,
    gamma_ij = w^(-2) g_ij,
    K_ij^phys = w^(-2) [A_ij + g_ij*K/3].

All conformal contractions use g and g^{-1}. Configuration fields are dimensionless; spatial auxiliaries, K, A, C, and Z have units M^{-1}; damping rates have units M^{-1}.

The on-reduction covariant reference is the vacuum GH-reduced equation

    R_mu_nu - nabla_(mu C_nu)
      + kappa[n_(mu C_nu) - (1/2)psi_mu_nu n^sigma C_sigma] = 0,
    C_mu = H_mu + Gamma_mu, n^mu=alpha^(-1)(1,-beta^i).

Here psi is the spacetime metric, gamma the physical spatial metric, and g the conformal spatial metric. The candidate uses C=n^mu C_mu and Z^i=g^{ij}C_j. The selected derivative-dependent gauge is part of the candidate; do not import the prescribed-source Lindblom theorem.

Use five coordinates s=(a,c,b,d,e) and

    T(s) = [[exp(a), 0, 0],
            [b, exp(c), 0],
            [d, e, exp(-a-c)]],
    g = T T^T,
    A = T Ahat T^T,
    Ahat = Ahat^T, tr(Ahat)=0.

Store only five components of Ahat. This guarantees det(g)=1 and tr(g^{-1}A)=0 for finite chart variables; it does not guarantee bounded conditioning.

Define

    J_ij,A = partial g_ij / partial s^A,
    Q_kij = J_ij,A S_k^A.

Then g^{ij}Q_kij=0 for arbitrary independent S. The 50-component state is

    U = (w, rho, s^A, beta^i, K, Ahat[5], Z^i, C;
         p_i, l_i, S_i^A, B_i^j).

There are 20 primary/curvature/GH variables and 30 auxiliary components. Here C=C_perp and l_i is HALF the old stored L_i. Do not confuse Z^i with every Z4 convention or infer its meaning from its name.

True derivatives of algebraic composites are essential:

    partial_k g_ij = J_ij,A partial_k s^A,
    partial_k Q_lij = J_ij,A partial_k S_l^A
                     + J_ij,AB (partial_k s^B) S_l^A,
    partial_k g^{-1} = -g^{-1}(partial_k g)g^{-1},
    partial_k A = (partial_k T)Ahat T^T
                + T(partial_k Ahat)T^T + T Ahat(partial_k T)^T.

Do NOT replace true partial g by Q, partial beta by B, or partial alpha by l inside complete-source differentiation. Connections defined algebraically from Q below are intentional; confusing these two roles changes the off-constraint PDE.

### 5.2 Complete geometric operators

Parentheses symmetrize with weight 1/2. Let theta=B_i^i and define

    Gamma_abc = (Q_bac + Q_cab - Q_abc)/2,
    Gamma^a_bc = g^{ad} Gamma_dbc,
    Gamma^a(Q) = g^{bc} Gamma^a_bc,
    Lambda^a = Gamma^a(Q) - Z^a.

The specified modified Ricci operator is

    Rstar_ij = -(1/2) g^{kl} partial_k Q_lij
              + g_{k(i} partial_{j)} Lambda^k
              + g^{kl} [Gamma^m_kl Gamma_(ij)m
                         + Gamma^m_ki Gamma_jml
                         + Gamma^m_kj Gamma_iml
                         + Gamma^m_ik Gamma_mjl].

Use true derivatives of Lambda and Q under the rules above. Define

    P_ij = partial_i p_j - Gamma^k_ij p_k,
    N_ij = partial_i l_j - Gamma^k_ij l_k,
    m_i = partial_j(g^{jk} A_ki) + Gamma^j_jk A^k_i
          - Gamma^k_ji A^j_k - (2/3)partial_i K,
    M_i_alpha = alpha*m_i - 3*rho*A^j_i*p_j,
    Hstar = (2/3)K^2 - A_ij A^{ij}
            + w^2*g^{ij}Rstar_ij + 4w*g^{ij}P_ij - 6p_i p^i,
    Stensor_ij = alpha*w^2*Rstar_ij + alpha*w*P_(ij)
                - w^2*N_(ij) - w(l_i p_j + l_j p_i),
    Ttensor_ij = -w(Z_i p_j + Z_j p_i) - (w^2/2) Z^k Q_kij,
    curlB_i = sum_d(partial_d B_i^d - partial_i B_d^d),
    Z_i = g_ij Z^j,
    TF(X)_ij = X_ij - g_ij*g^{kl}X_kl/3.

The symmetrized Hessians and curlB completion are part of the candidate. Do not omit them because they vanish for exact gradients. Hstar is not an independent physical Hamiltonian diagnostic when Z or reductions are nonzero.

For independent physical diagnostics use the actual primary metric/extrinsic-curvature jets:

    H_phys = R[gamma] + K^2 - K_ij^phys K_phys^{ij},
    M_i_phys = D_j K_phys^j_i - D_i K.

Assemble these conformally to avoid unnecessary singular physical intermediates. A contraction of the evolution's Rstar is not an independent Ricci oracle. Preserve and label alpha-weighted momentum separately from unweighted momentum.

### 5.3 Direct puncture gauge and primary configuration equations

Keep the candidate's specified integrated Gamma-like driver; do not silently substitute a two-field driver or change its coefficient from one to 3/4.

    Fw = w*(alpha*K-theta)/3,
    D0 w = Fw,
    D0 rho = rho*[-2K-(alpha*K-theta)/3],
    D0 alpha = -2 alpha K,
    Fbeta^i = Lambda^i - eta*beta^i + sigma(z)*V^i,
    V^i = g^{ij}(alpha^2*w*p_j - alpha*w^2*l_j),
    z = alpha*w^2,
    D0 beta^i = Fbeta^i.

For the new candidate use the pinned C-infinity plateau with (z0,z1)=(0.1,0.5): sigma=0 below z0, sigma=1 above z1, and inside

    u=(z-z0)/(z1-z0),
    sigma=exp(-1/u)/[exp(-1/u)+exp(-1/(1-u))].

Evaluate it and its derivatives robustly. Preserve the legacy switch exactly in legacy controls. If switch definitions differ, compare physical oracles using matched gauge functions; do not claim zero on-manifold difference across different gauges.

Let B as a matrix have row i and column j, B_i^j. Set

    Fg = -2 alpha A + B g + g B^T - (2/3)theta*g,
    W = T^{-1} Fg T^{-T},
    Ltri = strictLower(W) + diag(W)/2,
    D0 T = T Ltri.

The five coordinate sources are

    Fa = Ltri_11,
    Fc = Ltri_22,
    Fb = b*Ltri_11 + exp(c)*Ltri_21,
    Fd = d*Ltri_11 + e*Ltri_21 + exp(-a-c)*Ltri_31,
    Fe = e*Ltri_22 + exp(-a-c)*Ltri_32,
    D0 s^A = Fs^A.

### 5.4 Curvature and GH equations

Use the exact direct-gauge candidate, NOT the different source-target Lindblom/core construction with alpha*C terms in the lapse.

    D0 K = alpha*A_ij A^{ij} + alpha*K^2/3
           - w^2*g^{ij}N_ij + w*p^i*l_i
           + alpha*(Hstar-K*C) + alpha*w*p_i*Z^i
           - (3/2)kappa*alpha*C,

    D0 C = alpha*(Hstar-K*C) + w^2*(rho*p_i+l_i)*Z^i
           - 2*kappa*alpha*C,

    FA = TF(Stensor) + B A + A B^T - (2/3)theta*A
         - 2*alpha*A*g^{-1}*A + alpha*(K-C)*A
         + alpha*TF(Ttensor),

    D0 Ahat = T^{-1} FA T^{-T} - Ltri*Ahat - Ahat*Ltri^T,

    D0 Z^i = -2*g^{ij}M_j_alpha - alpha*g^{ij}partial_j C
             + C*g^{ij}l_j - Z^j B_j^i + (2/3)theta*Z^i
             - [(2/3)alpha*K+kappa*alpha]Z^i
             + g^{ij}curlB_j.

On the full reduction surface these must agree with an independently implemented physical 3+1 GH projection, with nonzero C and Z allowed. Off that surface these equations DEFINE the candidate; they are not asserted to equal every other FO-GH extension.

### 5.5 The essential new auxiliary equations

For ten configuration potentials and their derivative variables, define

    x^A=(w,alpha,s[5],beta[3]),
    G_i^A=(p_i,l_i,S_i[5],B_i[3]),
    F^A=(Fw,-2alpha*K,Fs[5],Fbeta[3]),
    E_i^A=G_i^A-partial_i x^A,
    Omega_ij^A=partial_i G_j^A-partial_j G_i^A.

Implement ALL auxiliary rows as

    partial_t G_i^A = beta^j partial_j G_i^A
                     + (partial_i beta^j)G_j^A
                     + partial_i F^A - lambda*E_i^A.

In particular,

    partial_t l_i = beta^j partial_j l_i + (partial_i beta^j)l_j
                   - 2K partial_i alpha - 2alpha partial_i K
                   - lambda(l_i-partial_i alpha).

It is NOT the old row -2K*l_i plus a different relaxation target. Differentiate the entire composite F, retaining all chain-rule factors and the true primary derivatives. Use analytic/automatic differentiation or generated algebra where it improves correctness; validate it independently and keep generated code reproducible.

The exact continuum target is

    (partial_t - Lie_beta)E^A = -lambda E^A,
    (partial_t - Lie_beta)Omega^A = -lambda Omega^A - d(lambda) wedge E^A.

No J or dJ source mixing is allowed in the fundamental intrinsic reductions. In the Cartan form of the undamped auxiliary RHS,

    G_t = d(F + i_beta G) + i_beta dG,

the i_beta dG term MUST remain. Dropping it changes the free ordering and can reintroduce the old compatible-ordering principal defect.

For physical tensor diagnostics also retain

    partial_i Q_j - partial_j Q_i
      = J_A Omega_ij^(s,A)
        + J_AB[(partial_i s^B)E_j^(s,A)
               -(partial_j s^B)E_i^(s,A)].

Do not claim the raw Q curl is an uncoupled fundamental curl away from the reduction surface. Curl-free alone is weaker than reduction-free.

### 5.6 Damping and puncture limits

Treat kappa (GH damping), eta (shift damping), and lambda (reduction relaxation) as separate physical parameters. Do not infer units from a label, grid spacing, timestep, or projection fraction.

For the new puncture candidate, prefer the explicitly labeled lapse-scaled option

    lambda=alpha*gamma_R, gamma_R>=0 finite with units M^{-1},

rather than making a nonzero coordinate-time rate arbitrarily strong in the core. For initial NEW free-candidate controls, proposed defaults are M*gamma_R=1, M*kappa=1, and M*eta=2. These are starting test settings, not a demonstrated puncture optimum. Legacy and reproduction arms retain their exact historical values. Keep constant-lambda modes for exact algebra, reproduction, and controlled comparisons. Freeze actual values before campaigns; changing a rate law or coefficient is an experimental arm, not a hidden implementation fix.

With variable lambda, retain d(lambda) wedge E in the subsidiary analysis. Lapse-scaled damping weakens in the core and does not by itself solve the puncture problem. Verify the frozen/full principal and source calculations for the selected law.

Known obstruction to preserve, not conceal:

    P_TT(h,A,q) = [[0,0,0],
                  [0,0,-alpha*w^2/2],
                  [lambda,-2alpha,0]].

Constant nonzero lambda gives a defective nilpotent limit as alpha,w -> 0. Even without that lifted obstruction, the unchanged tensor energy has relative weight H_qq=(w^2/4)H_AA. These are not proofs of the campaign's root cause, and failure of a uniform unweighted energy does not alone prove finite-grid instability.

The isolated rescaling Phi_T=w*q/2 and Pi_T=A-gamma_R*h/2 with lambda=alpha*gamma_R balances the tensor block, but does NOT authorize weighting all Q components: the unsuppressed Gamma(Q) shift can then introduce 1/w. Keep that as a separately analyzed fallback, not part of the first combined implementation.

## 6. Numerical design: separate interface repair from formulation replacement

### 6.1 Legacy transfer-only control comes first

Before changing the bulk equations, implement and test a coherent auxiliary transfer option. Retain the collision control's gauge, rates, global projection policy, stage timing, and numerical operators in this arm.

For a linear primary/auxiliary pair, a useful commuting-residual target is

    u_f=P0 u_c,
    G_f=D_f u_f + PE(G_c-D_c u_c),

so that G_f-D_f u_f=PE E_c. The reverse operation needs its own derived rule. Include same-level exchange, coarse/fine faces, edges, corners, normal and tangential auxiliaries, restriction, prolongation, and the POST-PROJECTION exchange path.

For composite alpha=rho*w and the intrinsic chart, derive the corresponding reconstruction from the ACTUAL transferred primaries. Do not transplant the linear formula while independently interpolating alpha, rho, w, g, and chart gradients inconsistently.

This identity alone is not a stable interface scheme. Establish the full discrete operator's behavior, including norm/adjoint consistency where available. Exact curl preservation is desirable, not mandatory; controlled, convergent injection is acceptable. Do not diagnose instability solely from a nonzero transfer increment.

Account for halo support. FD6 has three-point-radius first derivatives; reconstructing derivative ghosts and then differentiating them can require a six-cell primary reach. Four primary ghosts do not support a naive version. Implement scratch halos or a genuinely compatible interpolation/stencil construction; do not merely increase nghost where dispatch routines only support existing cases. Trace the actual communication implementation.

### 6.2 Preserve the intended semidiscrete reduction law

For a chosen reconstruction R_h(U), instrument

    e_h=G-R_h(U),
    I_h=Gdot-R_h'(U)Udot-T_h e_h+lambda e_h.

I_h is the mismatch from the intended discrete transport-relaxation law. Measure it separately from the continuum source terms. Do not call it a defect oracle if unsynchronized ghosts are used.

The lapse is a required regression:

    R_h^alpha=D_h(rho*w),
    R_h^alpha'(U)Udot=D_h(rho*wdot+w*rhodot).

This generally differs from separately discretizing D_h(beta.D_h(alpha)-2alpha*K). Keep the direct shared lapse-gradient helper, multiplying rho*w AT EACH stencil point; do not duplicate FD coefficients.

Spatial analytic chain-rule discretization and differentiating a materialized composite source need not agree. Specify which is used, measure their truncation-level difference, and test the resulting high-frequency operator. Do not silently replace the continuum candidate by an unrelated second-order numerical scheme.

Include KO/filter contributions and RK stages. Linear gradient relations can be preserved by a tangent RK update; nonlinear relations such as D_h(rho*w) generally have stage defects. Demonstrate temporal convergence instead of claiming exact preservation.

### 6.3 Optional core reconstruction is a separate experiment

Only after the transfer and clean free system are individually characterized, test a bounded regularity-adapted core. Before each core RHS, use valid same-stage primary ghosts and reconstruct

    p=Dw, l=D(rho*w), S=Ds, B=Dbeta.

Hold w,rho,s,beta,K,Ahat,C,Z fixed during the auxiliary correction. Do not project C,Z to zero as part of this operation. Intrinsic Q is J(s)S; do not apply an independent Q-trace reset afterward.

A full stagewise core reconstruction is a locally second-order-in-space numerical closure, not the free 50-field system's theorem. Analyze its effective stencil, RK behavior, and interface with the exterior. Do not evaluate singular parent terms and hope discrete projection cancels them.

For a tapered map G^+=G-P E, account for

    Omega^+=(1-P)Omega-dP wedge E.

With a moving/field-dependent P, account for its time/stage dependence and changing active region. Separate a fixed per-stage projection fraction from a physical damping rate. Do not pretend the same fraction at different dt represents the same finite-rate operator.

Choose and document the core by resolved regularity/field criteria or recovered physical bounds, not by fitting a stationary exponent or enlarging it until the crash is hidden. Keep the projection transition distinct from a refinement interface where feasible. Maintain a no-core-projection control.

## 7. Mandatory mathematical and compiled-oracle tests

Do these before expensive evolution, and extend them whenever equations change:

1. Exact intrinsic determinant, inverse, curvature-trace, and gradient-trace identities for arbitrary non-diagonal geometry and independent gradients. Verify both directions of the finite-radius state map.
2. Independent nonlinear physical 3+1 GH/Ricci/Hessian/Codazzi oracle on smooth nontrivial jets with reductions satisfied but C,Z nonzero. Use the same gauge function in both oracles. Do not compare a function against itself or require equality to a different off-constraint formulation.
3. Exact auxiliary subtraction for arbitrary smooth fields, including variable lambda, nonzero shift gradients, all ten families, and the physical Q-curl reconstruction identity. Show explicitly that the unwanted J,dJ terms are absent.
4. Compile a point-jet/RHS oracle for the ACTUAL AthenaK candidate kernel on CPU and CUDA. Compare every row and the full 50x50 principal matrices against the independent candidate oracle. Test off-reduction jets, oblique normals, non-diagonal g, nonzero K/A/C/Z/B, switch transitions and plateaus, and multiple positive alpha,w values.
5. Verify the complete characteristic structure, including eigenvector completeness and bounded projectors at admitted coincidences. For conformal-normal magnitude nu=sqrt(g^{ij}n_i n_j), expected coordinate speeds are

       -beta.n                                   multiplicity 30,
       -beta.n +/- nu*alpha*w                    6 per sign,
       -beta.n +/- nu                           2 per sign,
       -beta.n +/- nu*w*sqrt(2alpha)             1 per sign,
       -beta.n +/- nu*sqrt((4-sigma*alpha^2*w^2)/3) 1 per sign.

   The theorem's domain is SPD g, w>0, rho>0, 0<alpha<2, alpha^2*w^2<4, with the prescribed switch completed by z=1/2. Test the full lifted symbol, not only the 20-field wave block. Detect approach to excluded domains during evolution. Do not claim a uniform puncture theorem from cellwise eigenvalues.
6. Reproduce the complete Minkowski Fourier polynomial including sources:

       (s+lambda)^30 (s^2+k^2)^2 (s^2+2k^2)
       * (s^2+eta*s+k^2)^3
       * (s^2+kappa*s+k^2)^3
       * (s^2+2*kappa*s+k^2).

   Check arbitrary real k algebraically where possible. Nonpositive eigenvalue real parts do not exclude neutral Jordan transients or nonnormal amplification. Test finite-time amplification as well as roots.
7. Reproduce the known common-symmetrizer obstruction as a limitation of this candidate, NOT a new failure gate. Preserve the exact certificate; do not waste the project searching the same fixed equations for an impossible common energy.
8. Examine all source coefficients and characteristic conditioning along wormhole and candidate trumpet-like sequences. No 1/w,1/rho,1/alpha should reappear in the fundamental kernel. Bounded chart variables do not prove bounded true derivatives. Do not claim an arbitrary integrated shift admits a particular stationary trumpet without solving that gauge condition.

Use exact rational/SymPy identities where feasible and scale-aware floating tolerances for compiled comparisons. Record tolerances before inspecting results. Use long-double/high-precision controls to distinguish conditioning from a coding discrepancy. Do not relax a failed tolerance without explaining and proving the original test was inappropriate.

## 8. Ordered experimental gates and minimal comparison matrix

Write PLAN.md and machine-readable gates before new evolutions. Record expected orders from the selected space/time/interface operators, not from wishful FD6 labels. A failed promotion gate stops larger/new-physics tests; affordable diagnosis may continue.

### Gate 0 — Provenance and unchanged legacy behavior

Recover exact inputs, restart layouts, scalar definitions, normalization, projection masks, and hashes. Verify legacy behavior against the old source on arbitrary jets and one-step controls. Reproduce the archived transfer discriminator where affordable. Backend/compiler differences require documented numerical equivalence rather than an unsupported bitwise claim.

Do not promote old scalar-max operation brackets to causal vector data. Preserve old data and add new diagnostics.

### Gate 1 — Operators and interfaces without black-hole evolution

Test FD2/4/6 where supported, 2D/3D, anisotropic spacing, periodic/same-level and nonconforming meshes, all directions and all auxiliary families. Include smooth determinant-one non-diagonal metrics, nonlinear rho*w, polynomial and trigonometric fields, and discontinuity-free interface crossings. Verify masks at zero/full/taper and overlapping cores.

Compare existing transfer and repaired transfer at fixed primary data. Measure reduction residual, intrinsic and raw curls, same-cell operation increments, projection idempotence where claimed, and expected truncation scaling. Check transfer faces, edges, corners, ghost layers, and repeated exchanges. Add CPU/CUDA and serial/MPI checks.

### Gate 2 — Smooth and forced-constraint evolution

Run exact Minkowski, shifted/gauge waves, and smooth curved manufactured solutions. Independently seed longitudinal reduction errors and transverse curl errors in p,l,S,B. Include nonzero K, A, lapse gradients, and shift gradients; flat tests alone cannot exercise the removed source mixing.

Use manufactured solutions or a derived stationary background when needed; do not freeze an arbitrary non-solution and call its perturbation growth a physical instability. Compare continuum forcing/subsidiary predictions to the evolved constraints.

Use three spatial resolutions with temporal errors independently suppressed and a fixed-grid timestep ladder. Determine separate spatial, temporal, and interface orders. Compare temporal rates to the chosen integrator, not the older RK4 campaign when running RK3. Test finite injection and repeated interface injection, both with and without damping. Require a bounded, resolution-improving forced error level rather than identically zero curl.

Check the complete discrete amplification/RK operator, including KO and relaxation. SSPRK3 has R(z)=1+z+z^2/2+z^3/6, but separate scalar CFL bounds do not prove coupled nonnormal/interface stability. Any new cleaning speed must enter the timestep estimate.

### Gate 3 — Nontrivial single puncture, controlled physical layout

Start with unboosted M=1 data and perturb physical, GH, reduction, and curl sectors separately. Use fixed physical refinement interfaces and outer boundaries at three resolutions, plus a temporal ladder. The M/8,M/10,M/12 outer-128M campaign can be reproduced, but add geometrically spaced levels if needed for an unambiguous asymptotic fit.

Compare on common physical regions and fixed-radius annuli; separately quantify the shrinking puncture layer using justified regularity scaling. A direction-dependent gradient limit does not promise pointwise convergence at r=0. Conversely, exponential growth at fixed physical radius, loss of positivity, or nonconvergent physical fields cannot be excused as puncture regularity.

Require a real dynamical relaxation/perturbed test, not only survival of an exact fixed point. Reach at least the recovered 20M single-hole gate before promotion, subject to the stronger convergence and health criteria below.

### Gate 4 — Original fine-core stress, not a substitute easy problem

Recover the exact saved input and logical mesh map. Required reference geometry:

    M=1; domain [-8M,8M]^3; outflow boundaries;
    cell-centered root 16^3; MeshBlocks 8^3; four ghost cells;
    eight 2:1 static refinements; finest h=M/256;
    finest box [-M/16,M/16]^3;
    456 leaves: 56 at each physical level 1--7, 64 at level 8;
    FD6, RK3, CFL 0.2, KO 0.3; original dt ceiling 0.0125M.

Use the recovered input, not a reconstruction from this description alone. The existing R16 control has (1+15P)/M and radii (M/8,M/2), both projections OFF. The new candidate deliberately differs in formulation/rate policy; identify those differences rather than calling its full input byte-identical.

Preserve the old 6M health gate, including the known 3--5M growth window. Passing 6M means no unexplained runaway, not simply delaying the fatal check. Use two timesteps at fixed spatial operators.

If necessary, move the finest interface outward at fixed h_min, then vary h at fixed physical interfaces. Change outer boundaries only in a separate comparison preserving the inner hierarchy. Do not compare the large-core coarse runs to the small-core fine runs as a resolution ladder. Do not let this stress test replace the fixed-layout convergence study.

### Gate 5 — Moving puncture and matched head-on binary

Only after preceding promotion gates pass, evolve a moving puncture through the hierarchy, checking transfer/reflection/parity behavior and restart continuity. Distinguish gauge translation tests from physically boosted constraint-satisfying initial data; label each accurately.

Then reproduce the exact documented head-on setup and matched Z4c control, retaining domain, resolution hierarchy, initial data, extraction radii 8/12/24/32/48/56M, and gauge conventions. Initial target t=100M; extend only under the resource plan and only if needed to cover a declared, healthy post-merger interval. t=100M is a screen, not sufficient convergence evidence.

For qualification use three resolutions at fixed physical layout where feasible. Assess trajectories, waveform amplitude/phase, finite-radius consistency, parity-forbidden mode leakage, Hamiltonian/momentum constraints, C/Z, all reductions/curls, and metric conditioning. Compare waveforms at common physical/retarded times; contaminated outer-radius signals are not validation. Fix waveform error budgets before looking at new curves, and justify them against resolution/extraction errors.

Z4c is a physical-reference control here, not a matched first-order BSSN/CCZ4 reduction benchmark. Do not claim numerical superiority over those first-order formulations without matched implementations and tests. A theoretical comparison of subsidiary laws must be labeled as such; building another complete solver is outside the initial scope.

Do not reclassify a restart of a contaminated legacy checkpoint as new-scheme qualification. The first qualified candidate binary starts from validated initial data.

### Required controlled arms (run adaptively, not a Cartesian sweep)

A. Verified legacy collision equations, original transfers, original projection policies.
B. Same as A, coherent transfer only.
C. Same numerical fixtures, fully specified clean intrinsic formulation; no auxiliary projection for its free-system tests.
D. Same as C, localized stage-consistent auxiliary core closure only.

Within the transition from legacy to free candidate, separate the GH-reset policy: hold it fixed for strictly numerical A/B diagnostics, then remove it and use explicit GH evolution/damping in a distinct controlled test. An intermediate globally GH-projected candidate is a diagnostic arm, not the final free-evolution claim. Do not change GH reset, auxiliary prescription, transfer, gauge completion, damping law, and projection mask all in one comparison.

## 9. Diagnostics and nonnegotiable success criteria

For every serious run retain:

- Full-domain AND excised GH, physical Hamiltonian, physical momentum, individual reductions, and individual intrinsic/raw curls. Keep historical diagnostics and append new definitions; never redefine an old column silently. C=Z=0 after a reset is not evidence that physical constraints are small.
- Coordinate-volume L1/L2/RMS and maxima, with each region's actual volume and mask definition. Use common physical regions across domain changes; large-domain RMS dilution is not improvement. Respect different dimensions of reduction and curl norms.
- Primary positivity and conditioning at synchronized states AND stages used by RHS: min/max w,rho,alpha, min metric eigenvalue, condition number, determinant, trace residuals, max chart/curvature/derivative fields, alpha and alpha^2 chi domain margins. Preserve first-bad-state location, level, distance to interface, and sufficient stencil data.
- Leading signed tensor components, not only combined norms. Record locations without interpolating maxima locations in time. Track component onset and local time series before deterioration.
- Same-cell vector corrections and signed contributions before/after RK, algebraic enforcement, auxiliary projection, GH projection, ordinary transfer and post-projection transfer. Label ghost validity; do not evaluate physical conclusions on unsynchronized states. Include norm-of-difference as well as difference-of-norms.
- Projection correction norms BEFORE and AFTER correction, and their dt/h scaling. Tiny constructed residuals alone do not qualify a projected method.
- Refinement topology changes, patch/block identities, mask crossing events, tracker restart metadata, output precision, and restart rollback epochs. Do not extract fine convergence from float32 outputs below their floor.
- Full-domain localized growth envelopes and their resolution dependence. A bounded shrinking-layer error requires justified scaling; a persistent positive-growth envelope with increasing physical errors fails regardless of survival.

For equal refinement ratio r and field differences d1=U_h-U_(h/r), d2=U_(h/r)-U_(h/r^2), require an asymptotic regime in which

    ||d1||/||d2|| -> r^p,
    <d1,d2>/(||d1|| ||d2||) -> 1.

Use the appropriate unequal-spacing relation when grids are not geometrically spaced. Compare independent primary fields on common points with controlled interpolation. Poorly aligned/anti-aligned differences invalidate a Richardson claim even when scalar constraints decrease.

All pass thresholds, growth windows, expected orders, and justified regularity exceptions belong in frozen PLAN.md/gates.json before a run. Do not suppress an unstable cell, clip determinants, increase floors, force exponents, or expand a diagnostic excision mask to pass.

## 10. Deliverables, commits, and continuation behavior

Maintain these or equivalently clear paths:

    docs/pc_gh_clean_reduction_plan.md
    docs/pc_gh_clean_reduction_equations.tex
    docs/pc_gh_clean_reduction_status.md
    analysis/pc_gh_clean_reduction/
    qualification-runs-20260907/pcgh-clean-reduction/

Include branch/commit inventory, source/input/build manifests, proof/counterexample scripts, generated-code provenance, compact CSV/JSON results, health/convergence/operation-budget plots, exact commands, and a machine-readable test ledger with PASS/FAIL/BLOCKED/NOT_RUN.

Keep legacy mode reproducible. Give the intrinsic mode an explicit name and state-layout version. Make incompatible restarts fail clearly; supply a tested conversion only where mathematically defined. Do not reinterpret an old 55-field restart as the new 50-field state. Include CPU/CUDA, serial/MPI, restart and halo regression tests.

Commit in coherent stages: provenance/tests; transfer diagnostics and repair; intrinsic map; complete new RHS and oracle checks; damping policy; optional core; qualification evidence. Inspect diffs for accidental production or unrelated changes, secrets, binary dumps, and huge archives. Store large checkpoints outside Git with hashes and paths. Preserve failures and their first invalid states under the available storage policy.

Push only the new implementation branch; do not merge or force-push. Verify the remote ref equals the intended commit before reporting “pushed.” If push fails, report the exact failure and leave an importable patch/bundle; do not claim remote success.

The final report must distinguish:

1. Which equations and numerical operators changed, and which did not.
2. What was proved, what was independently checked, and what was only inferred.
3. Whether interface-only repair or clean bulk reduction gave a measured advantage.
4. The effect of GH resets and localized auxiliary reconstruction separately.
5. Which gates actually passed, all failures, resource/provenance blockers, and exact run durations.
6. Full-domain physical/field convergence and whether the binary result is qualified or merely survived.
7. The final branch and verified commit SHA, reproduction commands, and one prioritized remaining action.

Continue toward the goal with hypothesis-driven changes and affordable falsification tests. Do not wander into a new reference-gauge program, bolt on GLM fields without joint analysis, or spend the run budget chasing every possible formulation. GLM curl waves are a later fallback only if the clean subsystem and coherent transfers leave a demonstrated transport limitation.

The central scientific question is: can we retain the existing successful puncture/collision infrastructure while removing identifiable auxiliary-error feedback and controlling repeated interface injection? Every proposed change and qualification claim should answer that question.
