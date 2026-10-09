# Final held source, volume and manufactured gates

This addendum finalizes the three items requested by root after reviewing `RECIPE.md`. It also makes the kernel-component adapter explicit. The original recipe and preparation receipt remain unchanged. This is a source-only preparation: no radial matrix, spectrum or evolution is admitted here. The numerical tolerances below are final before the first derivative/mass/operator run; a failed gate must be preserved rather than relaxed in place.

## Exact component adapter

Let e^a_i be a reference Penrose orthonormal coframe, with a=(n,T,U), and e_a^i its inverse. Covariant tensors are transformed with e_a^i e_b^j and contravariant vectors with e^a_i. For the spherical reference, e_n^i=(alpha_hat/L)n^i and e^n_i=(L/alpha_hat)n_i. The tangent metric tensor and independent tangent A tensor in the chart are

```
h_ab = e_a^i e_b^j delta gtilde_ij/chi_hat,
S_ab = e_a^i e_b^j [delta Atilde_ij
        - gtilde_ref,ij (Aref^kl delta gtilde_kl)/3]/chi_hat.
```

Both are tracefree. The subtraction in S is part of the time-independent linear map, so S_t contains the same subtraction using delta gtilde_t. It is not legal to discard that configuration contribution to V_t.

For a tracefree tensor T, define the five-component adapter

```
chart(T)=(T_nn,T_nT,T_nU,(T_TT-T_UU)/2,T_TU).
```

These five components are not a Frobenius-orthonormal STF basis. No additional square-root or spin-CG factor is applied after the physical Cartesian lift. The inverse reconstructs T_TT=-T_nn/2+T_plus and T_UU=-T_nn/2-T_plus, with symmetric off-diagonal entries. An independent Cartesian-to-frame-to-chart-to-Cartesian round trip and screen rotations by 0.37 and 0.81 radians gate this adapter.

Construct U from delta alpha/alpha_hat, delta chi/chi_hat, chart(h), and frame(delta beta)/alpha_hat, then differentiate the complete reference-normalized field: q=D_s U. The principal chart is the exact frozen ordering

```
y=(q_alpha,q_chi,q_hnn,P/Omega,Theta_phys/Omega,S_nn,lambda_n,q_beta_n,
   q_hnT,S_nT,lambda_T,q_beta_T,q_hnU,S_nU,lambda_U,q_beta_U,
   q_hplus,S_plus,q_hcross,S_cross),
lambda_a=chi_hat e^a_i delta Lambda^i.
```

The adapter is a component transformation, not merely a permutation of CG amplitudes. The full normalized first-order normal action, including this adapter and the chosen q=D_s U lower-order reduction, is checked against K_n=beta_n I+alpha_hat A_symbol. Algebraic output normals are checked in raw22 before applying this map.

## Explicit independently assembled symmetric volume production

Write c=alpha_hat/L, dV=r^2/c dr dOmega, D_s=c partial_r, and dSigma=rb^2 dOmega. The radial normal vector has div(s)=2c/r. Let

```
K_n=beta_n I+alpha_hat A_symbol,
Q=H K_n,                 Q=Q^T,
Gamma=D_s Q+(div s) Q.
```

For each complete trial field Phi_j, obtain U_j, y_j and its unchanged actual full linearized point source U_t,j,y_t,j. Form the pointwise remainder

```
R_j = y_t,j - K_n D_s y_j.
```

This uses the independent strong configuration derivative q_t=D_s U_t and the analytic first derivative of V; it does not use an assembled radial matrix. Angular derivatives, frame/reference derivatives, live-gauge lower-order terms, damping and the A-trace subtraction remain in R_j. No angular term is frozen or omitted. Define the volume matrix directly by

```
G_volume,ij = integral [
    y_i^T H R_j + R_i^T H y_j
    - y_i^T Gamma y_j
    + epsilon/S^2 (U_i^T U_t,j + U_t,i^T U_j)
  ] dV.
```

The integrand is manifestly symmetric under i<->j after Q=Q^T. Gamma is differentiated from the analytic reference/cutoff formulas, including zeta' in H=(1-zeta)H_C+zeta H_D and all alpha_hat/L factors. In equivalent grouped form, Gamma=(c/r^2) partial_r(r^2 Q). The regular-core polynomial evaluation and grouped radial density define the origin limit; no angular frame or 1/r endpoint formula is evaluated at r=0.

Independently assemble K_ij=<Phi_i,L_actual Phi_j>_E by the weak q_t integration specified in the recipe, and independently read back the strong integral. The required identity is

```
K+K^T = B^T H_b K_n,b B + G_volume,
```

where B contains rb sqrt(w_angle). Neither R nor Gamma nor G_volume may be inferred from K+K^T or a radial-operator residual. No coefficient of the SAT is tuned to make this identity pass. The separate incoming SAT is exactly -E^-1 B^T H_b k_in P+ B. Its work must include the configuration part of the Riesz lift.

This is a finite-dimensional radial normal energy identity. Angular derivative work is present in R and in its independently integrated bilinear form. A finite generalized upper rate for G_volume relative to E is recorded even when positive; it is not called a continuum, all-direction, all-degree or exact-scri bound.

## Fixed manufactured fields and norms

All initial fields below are complete physical Cartesian fields through the frozen all-m solid-CG basis and reference lift. Every channel and every nonnegative m is tested in its nonzero real and imaginary phases; negative m is checked by the exact frozen conjugation relation. No cross-L envelope restriction is imposed.

1. Zero deviation at all quadrature/trace points tests the exact stationary reference and the existing source subtraction.
2. Polynomial channel columns use W=1,rho,rho^2,rho^3 independently. In each J, also use the fixed mixed column sum with coefficient (-1)^a/(a+1) for channel a and W_a=1+rho/3-rho^2/5+rho^3/7. These are exact members of every proposed N>=8 trial space. They check the core, origin, physical lift, weak/strong source action, all22 normals and angular reconstruction.
3. Einstein-compatible gauge witnesses set only lapse or only one allowed shift CG channel nonzero, with W_g=exp(-8rho); all geometric/P/Theta/A/Lambda variations are exactly zero. Initial physical H/M/Z/Theta must vanish. Their continuum constraint rate must converge to zero. The full field is evaluated analytically at nearby Cartesian points; no discrete projection of this nonpolynomial W is used in that pointwise rate gate.
4. Constraint-bearing shell columns use W_s=exp(-((rho-0.49)/0.16)^2) in each P,Theta,Lambda,metric-trace,metric-STF and independent-A channel, plus the fixed alternating channel sum above. Gauge variations are zero. Their physical constraints and actual continuum constraint rates are compared with the frozen coefficient-aware reference subsidiary equations; there is no assumption that their rate vanishes or is damped pointwise.
5. Finite-boundary consistency uses X_exact(t)=exp(t) X_0 for each polynomial column and the fixed polynomial mixed field. The **pointwise** physical source is f_exact=Phi(X_0)-L_actual Phi(X_0) at t=0; it is formed from the actual kernel before radial assembly. Incoming data are g_in=P+ B X_0. The independently assembled forcing load and inhomogeneous SAT must give X_t=X_0 after adding J_bulk X_0 and J_SAT X_0. Thus this check contains nonzero configuration/momentum fields, actual-source forcing, and the complete derivative/momentum trace. It is not a manufactured time-evolution run or a check defined by subtracting the assembled matrix.

Use the exact energy norm ||X||_E=sqrt(X^T E X) for projected fields, with epsilon=1, and separately retain the actual source pointwise raw22 norm and its absolute maximum. Matrix residuals use ||left-right||_F/max(1,||left||_F,||right||_F), with maximum absolute entries also saved. Pointwise vector comparisons use the corresponding Euclidean norm divided by max(1,||expected||_2), and absolute component errors are retained. Symmetric bilinear identity checks also save the skew residual before symmetrization; no post-hoc averaging hides a failed source identity.

Constraint records remain in actual order (H_physical,M_cov,physical[3],Z_cov,physical[3],Theta_physical) with no hidden Omega rescaling. Report each component RMS/peak, and separate Penrose covector group RMS using bar-gamma^{-1} and physical covector group RMS using Omega^2 bar-gamma^{-1}, integrated with the declared Penrose dV and divided by its volume. This distinguishes two named norms without changing the diagnostic fields. No group norm is called a physical conserved energy. For each rate comparator also record its absolute RMS/peak and error/max(1,||actual rate||,||comparison rate||); a small constraint seed is never used as an unstable denominator.

The primary configurations/momenta are not Euclidean-normalized separately across N. The same fixed physical manufactured amplitudes and the same rb are used across degrees. Nonpolynomial pointwise checks and finite-dimensional projection errors are separately labeled.

## Constraint rates without invented higher reference jets

The pointwise actual tangent source and initial constraint q(x)=D C_ref(x)[Phi(x)] use the byte-pinned actual dual kernel and analytic reference/lift jets only through the orders they consume (at most second for metric/scalars, first for A). Values of these complete functions at neighboring Cartesian points are the inputs to the independent derivative readbacks below.

At core points r=0,0.025,0.049, the reference is exactly flat. The independent frozen core Cartesian formula and flat physical constraint formula provide exact polynomial rates for W=1,rho,rho^2,rho^3, including at the origin. This is an analytic polynomial comparison, with all higher reference derivatives zero; it does not recover envelopes by dividing by r^L.

At transition/collar radii r=0.15,0.30,0.50,0.70,0.90,0.96,0.975 and additionally r=0.99 for rb=0.995, use directions (1,0,0), (1,2,3)/sqrt(14), (2,-3,1)/sqrt(14). There are two independent sampled functions:

* F(x)=L_actual Phi(x), a full raw22 tangent RHS. Form its spatial jet with centered fourth-order Cartesian differences, and apply the actual dual linearized physical constraint functional D C_ref[F] at the center. This is the readback of Cdot.
* q(x)=D C_ref[Phi(x)] in the physical eight-field ordering. Form its first/second spatial derivatives with a separately evaluated centered fourth-order Cartesian difference jet and apply the frozen coefficient-aware reference-only Subsidiary(p,10,qjet). This comparator includes full coefficient gradients. It is not a primitive frozen Fourier approximation.

Diagonal second differences use the standard five-point fourth-order stencil; mixed derivatives compose the fourth-order first differences. For each center let h0=min(0.002,(rb-r)/4) and use h=h0,h0/2,h0/4,h0/8,h0/16. The furthest mixed sample is 2sqrt(2)h away, so every sample remains strictly inside rb. All initial fields and reference coefficients are freshly evaluated at those points. This convergent finite difference recovers higher spatial variation of the **consumed analytic-jet functions**; it is explicitly a numerical rate readback, not a claim that third/fourth reference jets were supplied analytically. No third-reference-jet approximation is inserted into the actual kernel or radial assembly.

For each rate vector preserve all five h values, interlevel increments, observed orders where the increment exceeds 1e-10 times max(1,rate norms), and Richardson extrapolation. Fourth-order evidence means at least one pair of successive non-floor increments reduces by a factor >=8; near-zero or roundoff-dominated cases instead need the absolute final rate/comparator gate below and are explicitly marked as order unclassified. A discrepancy cannot be dismissed as roundoff while exceeding the final tolerance. A failure to resolve a nonzero rate/defect must stop that gate and retain its data. The core exact polynomial rate is a separate analytic oracle; it does not replace transition/collar sampling.

The frozen subsidiary helper and its original dependency context are source-pinned separately. It is linearized about this stationary analytic Einstein reference with C0 damping and must not be used as a generic nonlinear live closure. The original H/M/Z diagnostics remain unchanged. The projected radial constraint defect C(J_bulk X)-C(L_actual Phi) is then recorded separately from the continuum pointwise rate test; it is not required to vanish by construction. Report C(J_SAT X) separately for homogeneous incoming data, since this principal SAT is not CPBC and can generate constraints. For the forced manufactured consistency test, use the exact g_in so its initial SAT residual vanishes. The shared actual-source forcing checks consistency of assembly; the independent core oracle, directional derivative checks and coefficient-aware subsidiary comparison supply separate checks of the underlying continuum action.

## Final thresholds and stage order

All gates preserve absolute errors and scaled errors. The 2e-7 and 2e-8 levels from the held recipe are final:

* Actual double/dual binding, reference subtraction, raw22 normals, exact-core actions and angular closure reuse the byte-pinned local gate tolerances. New analytic configuration-source spatial derivatives versus independent fourth-order differences must be <=2e-7 scaled with the convergence/floor evidence just specified. The deliberate double-only-wrapper negative control must remain detectably different; otherwise the binding gate fails.
* Transition/collar pure-gauge initial constraints <=5e-11 scaled; final Cdot and its extrapolation <=2e-7 scaled/absolute for the exact-zero expected rate. Shell actual-versus-subsidiary rates <=2e-7 scaled, with the last interlevel increment <=2e-7 scaled and the same convergence/floor evidence. Unresolved cases are failures, not accepted equality. Core polynomial source/constraint rates <=5e-11 scaled against their independent exact oracle.
* Component adapter round trip, pointwise principal symmetry, normal action, screen rotation and all22 trace reconstruction <=5e-11 scaled; every sampled H and E is positive under the stated Cholesky/eigenvalue checks.
* Dense mass/modal congruence, linear solves, independent manufactured forcing, SAT work and restricted boundary trace algebra <=2e-9 scaled. Coupled modal energy condition <=1e12; otherwise stop as a conditioning failure. Actual incoming restricted ranks and constraint/gauge/TT split are checked, not inferred from full20 counts.
* Weak-versus-strong K, independently assembled G_volume identity, and Q-versus-2Q changes of E/K/G_volume/trace <=2e-8 scaled. Preserve a failing base receipt before the one allowed explicitly named quadrature refinement. No boundary coefficient or penalty factor adjustment is authorized by a failed identity.

Assembly starts only after the reviewed source-derivative/mass/operator release: N8/rb0.98/J0,1,2, then the declared N12/N16 and rb0.995 controls only after the first gates pass. Quadrature, epsilon, source and method choices remain those in RECIPE.md. This release, if granted, still excludes eigenvalues and propagation, J>=3, nonlinear pulses, CPBC and a scri-stability claim.
