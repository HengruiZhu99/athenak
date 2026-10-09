# Held finite-radius actual total-J control

This is a concrete proposed experiment, awaiting root review. No scientific code has been compiled, no actual radial matrix has been generated, and no spectrum or evolution has been run in this tree. All frozen inputs are pinned in `source-pins.json`.

The first experiment will use the unchanged C0 physical-P/spatial-norm equations, S=1, a=0.5, geometric transition [0.05,0.95], gauge transition [0.45,0.85], kappa_input=10, kappa2=0, xi=2, and the exact stationary Minkowski reference. It will retain P storage/evolution and the full existing reference subtraction. No new Q/C1/profile/feedback, primitive boundary floor, ghost extension or hidden inner boundary is introduced. The finite outer radii rb=0.98 and 0.995 define two separate artificial-boundary problems, rather than approximations asserted to be the exact-scri problem.

## State, regular trial space and explicit reduction

For each J=0,1,2, use the frozen scalar/vector/tensor solid-CG channels, with arbitrary independent envelopes W_a(rho), rho=r^2, over the whole ball. The amplitude counts are 8,16,20. At N=8,12,16, every envelope is a polynomial of degree at most N-1 represented on the same N Gauss-Jacobi(0,1/2) nodes mapped to [0,rb^2]. The exact Cartesian jet evaluator, complete physical metric-to-chi/g lift, independent-A lift including A_ref^{ij} delta g_ij, and Lambda independence remain unchanged. Near and at the origin use the frozen polynomial core blocks; do not infer amplitudes from Cartesian values divided by r^L.

The evolved trial coefficients are X=(configuration envelopes,momentum envelopes), with 4/8/10 amplitudes in each half. No independently evolved q is added. In the outward Penrose orthonormal frame, define ten reference-normalized configurations

```
U = (delta alpha/alpha_hat, delta chi/chi_hat,
     five STF components of delta gtilde/chi_hat,
     three components of delta beta/alpha_hat).
q = D_s U.
```

This means differentiating the complete normalized fields, including reference denominators, radial frame factors and solid harmonics. It is deliberately D_s(delta alpha/alpha_hat), rather than (D_s delta alpha)/alpha_hat. The difference is a specified lower-order term; it is not silently dropped. All normalizations are time-independent reference fields. D_s is the derivative along the outward unit normal for the reference Penrose spatial metric. Frame derivatives along a ray and all angular terms are retained.

The ten momenta/connections are

```
V = (delta P/Omega, delta Theta_phys/Omega,
     five components of TF_ref(delta Atilde)/chi_hat,
     three components chi_hat * delta Lambda in that frame).
```

TF_ref removes the metric-induced trace of the full delta A; its reconstruction still includes the required A_ref^{ij} delta g_ij contribution. Thus the exact map V_t also includes the configuration RHS contribution to that subtraction. P and Theta are not renamed or evolved after an Omega rescaling. This is solely a fixed-reference linear reduction. Reorder y=(q,V) into the exact `kernel_symbol.cpp` ordering before applying any principal matrix or symmetrizer.

The actual reference has Penrose metric diag((L/A_lapse)^2,1,1) in the radial/tangential frame, where A_lapse=alpha_hat, L=Omega-r Omega', and b are the pinned `LayerPoint` fields. Therefore

```
dVol = r^2 (L/A_lapse) dr dOmega,
D_s = (A_lapse/L) partial_r,
beta_n = -b,    beta_r = -b A_lapse/L,
coordinate principal = beta_r I + (A_lapse^2/L) A_symbol.
```

The lapse A_lapse and normalized principal matrix A_symbol are distinct quantities. The sphere has dSigma=rb^2 dOmega. These exact measures and factors are used in energy, derivative, bulk IBP and boundary trace work; no generic unweighted endpoint norm is substituted.

Screen axes may be chosen for evaluation, but their O(2) rotation must leave the quadratic and trace work unchanged. Check this explicitly; no axis-dependent energy is accepted. At the origin the normal frame is not evaluated. The complete Cartesian/core formula and the vanishing full-ball flux limit handle that endpoint.

## A continuous full-W normal symmetrizer

Let L(r) be the byte-pinned canceled left basis in `check_kernel_symbol.py`, evaluated at the actual reference alpha_hat(r) and gauge weight W(r), with q0=1/2. It gives H_C=L^T L, positive and symmetrizing the full-W normalized normal matrix A(r). It generally does not equal H_D=I+A_1^T A_1 at W=1.

Choose zeta(r)=SmoothCutoff(r,0.85,0.90) and

```
H(r) = (1-zeta) H_C(r) + zeta H_D.
```

The blend is confined to W=1, where both terms symmetrize the same harmonic A_1. H equals H_C below 0.85 and H_D above 0.90, with no jump at 0.85 and no use of H_D in W<1. Preserve zeta', alpha_hat', reference/frame and volume derivatives in the bulk production. Record pointwise symmetry, positive eigenvalues and condition numbers. This supplies a radial normal-block energy choice, not an all-direction three-dimensional symmetrizer theorem.

## Coupled dense energy and the chosen bulk projection

Let Phi_j be the complete regular physical trial field associated with one envelope coefficient. For the reference Penrose volume measure dV, assemble

```
E_X,ij = integral [ y_i^T H(r) y_j + epsilon/S^2 U_i^T U_j ] dV,
epsilon=1.
```

The positive U mass retains every derivative-null configuration, including the core constant scalar/vector/STF fields allowed by J. It is not made tiny to hide conditioning. The primary run uses epsilon=1; epsilon=1/4 and 4 are limited later sensitivity controls at N=12 if the primary algebraic gates pass.

Retain exact dense polynomial masses M_L=(1/2)integral rho^(L+1/2) li lj as a separate reference check. The coupled normalized E_X uses positive overintegration of its nonpolynomial reference weights and cross terms. It is neither those diagonal Gauss weights nor a sum of uncoupled M_L alone. Use modal Jacobi-(0,L+1/2) congruences for conditioning/solves while retaining common nodal envelopes. Report raw, diagonal-scaled and modal coupled conditions, Cholesky pivots and solve residuals. No configuration nullspace is removed.

The main proposed bulk discretization is **energy Galerkin**, not strong collocation mislabeled SBP:

```
K_ij = <Phi_i, L_actual Phi_j>_E,
J_bulk = E_X^{-1} K.
```

L_actual is the unchanged full linearized C0 tensor/gauge point action, with its complete reference lift and coefficient/angular terms. It will be evaluated at overintegration points through the frozen dual bridge; complete angular reconstruction is rechecked at held-out points. This changes the radial representation and projection relative to the Cartesian operator, which is part of the explicitly named control.

The K integral need not differentiate a second-order momentum RHS. With z_i=(H y_i)_q, v_i=(H y_i)_V, the derivative contribution can be evaluated independently as

```
integral z_i^T D_s U_t,j dV
 = integral_boundary z_i^T U_t,j dSigma
   - integral [D_s z_i + (div s) z_i]^T U_t,j dV.
```

The remaining terms are integral v_i^T V_t,j plus epsilon U_i^T U_t,j/S^2. With the explicit measure above, the differentiated radial flux density is partial_r(r^2 z_i), and div(s)=(2/r) A_lapse/L; the equivalent coefficient-aware divergence form is retained before grouping. All test/reference derivatives here are at most second order and already have analytic jets. The origin term must vanish from the regular core formula, not from deleting an endpoint row. Group the radial volume derivative before evaluation so an apparent 1/r factor is not used to prescribe data.

An independent strong readback computes q_t=D_s U_t from the first-order configuration/gauge rows, using their coefficient-aware spatial derivative and second jets; it does not require differentiating the second-order momentum rows. A private generic CPU dual extension may be needed for these configuration rows. It must match actual double/dual directional finite differences, include all reference derivatives and pass the negative double-only-wrapper control. No approximate third-reference-jet or frozen-coefficient shortcut is permitted.

## Independent bulk and boundary work identities

At each point let K_n=beta_hat_n I+alpha_hat A(r), so the normalized reduced system has principal part K_n D_s y. Construct the source-matched remainder R_actual=y_t-K_n D_s y, retaining angular, coefficient/frame, damping and lower-order terms. This is a point action, not a frozen primitive Fourier generator.

Compute volume production independently from this remainder, the reference divergence of H K_n s, and the positive U-mass work. The declared matrix identity is

```
K+K^T = F_boundary + G_volume,
F_boundary = B^T H_b K_n,b B.
```

G_volume must be assembled from the independently differentiated coefficient-aware point formulas and volume quadrature. Do not define it as the leftover K+K^T-F and then call that a bulk certificate. Compare weak and strong K readbacks, this IBP identity, exact constant-core polynomial action, and quadrature refinement. Record the modal generalized maximum production rate of G_volume relative to E_X, including positive values. Finite N positivity or a finite numerical rate is not a uniform degree, J, rb or continuum estimate.

At rb, W=1 and the Penrose metric is the exact CMC collar. Let B stack the full y_b trace over angular quadrature with each block weighted by rb*sqrt(w_angle), so B^T H_b K_n,b B includes exactly rb^2 integral dOmega y_b^T H_b K_n,b y_b. The derivative trace is evaluated from the regular polynomial, not from a nearest interior node. Use the exact harmonic projectors P+=(I+A_1)/2, P-=(I-A_1)/2 and H_b=H_D. In the RHS sign convention,

```
k_in=beta_hat_n+alpha_hat>0,
k_out=beta_hat_n-alpha_hat<0.
J_SAT = -E_X^{-1} B^T H_b k_in P+ B.
```

Angular sums and the exact surface Jacobian are included in B. The adjoint uses the same E_X and B as the trace identity; no inverse scalar quadrature weight or momentum-only approximation is substituted. This source generally changes both configurations and momenta. Its energy work is exactly -k_in y_b^T H_b P+ y_b, so the principal flux plus penalty is negative semidefinite. For nonzero manufactured data use P+ B X-g_in and retain its data work.

The frozen four-constraint/four-coordinate/two-screen-TT principal sectors are diagnostic classification only. Their left rows are not H-orthonormal and are not raw physical H/M/Theta/Z boundary data. Verify their restriction/ranks after the total-J reduction rather than assuming twenty independent incoming amplitudes in each J. The planned homogeneous incoming residual is a finite-boundary principal control; it is not CPBC, physical radiation data, an exact-scri compatibility hierarchy or a proof of constraint preservation.

## Concrete stage order and proposed numerical gates

After root review, first form N=8 operators at rb=0.98 for J=0,1,2, then N=12,16 and the separate rb=0.995 sequence only after source/mass/trace gates pass. The largest matrix is 320 by 320. Use base radial overintegration Q=4N+32 and independent 2Q; compare angular Gauss-Legendre/Fourier rules 12x24 and 16x32 with full measure. The actual kernel need only supply angular coefficient actions at well-conditioned fit/held-out angles; energy integrals use complete reconstructed Cartesian fields. No millions-of-point native evolution is needed for assembly.

Proposed gates, to be finalized before scientific work:

- Source/reference, full22 tangent normal and actual binding gates: reuse the frozen tolerances, with newly consumed configuration-spatial derivative directional checks at 2e-7 scaled and demonstrated fourth-order FD convergence where above roundoff.
- Exact core action and polynomial masses: exact symbolic identities; compiled residual <=5e-11 scaled.
- Raw/model-modal mass congruence, solve, SAT work and trace algebra: <=2e-9 scaled, retaining absolute residuals; coupled modal condition <=1e12 or stop and report conditioning failure.
- Weak/strong K and independently assembled bulk/trace identity: <=2e-8 scaled after quadrature refinement; mass/action/production/trace changes between Q and 2Q <=2e-8. If unresolved, preserve failure and refine quadrature once under a new receipt, rather than tune a boundary coefficient.
- Complete reference stationarity and manufactured Cartesian/core/regular-envelope actions, all-J/m angular closure, raw input/output algebraic normals and screen-axis rotation invariance.
- Manufactured Einstein-compatible lapse/shift initial data and constraint-bearing shell data: compare actual coefficient-aware physical H/M/Z/Theta and their continuum rates, then record the distinct semidiscrete constraint defect under N/refinement. Do not silently project constraints or enforce falloff to obtain equality. Source functional and finite-dimensional projection errors must be labeled separately.
- Manufactured finite-boundary consistency uses complete trace data g_in=P+ B X_exact and actual-source forcing, including coefficient/angular terms. Reference zero deviation and boundary residual zero alone do not establish arbitrary-solution consistency.

Record endpoint Omega and source scales separately: at rb=0.98, Omega=0.0396 and 10/Omega=252.5253; at rb=0.995, Omega=0.009975 and 10/Omega=1002.5063. Keep finite-boundary incoming speeds -0.0004/-0.000025 explicit. The boundary traces are not bounded by the energy norm uniformly in degree; report their scaled operator norms and growth rather than assert a degree-uniform boundary theorem.

Passing these gates would produce a source-matched finite-dimensional experimental control. Any full dense eigenanalysis or guarded pulse propagation would be a separately released next stage, with residuals, integrator/semigroup accuracy and fixed seed amplitudes. No such computation is authorized by this held recipe.

The J<=2 control omits J>=3, including possible J=3 content of a spin-one/orbital-two vector gauge pulse. Finite nonlinear pulses can mix J. Absence of growth here cannot exonerate the original full Cartesian pulse, identify a unique ghost cause, prove whole-system continuum stability or certify the exact-scri limit.
