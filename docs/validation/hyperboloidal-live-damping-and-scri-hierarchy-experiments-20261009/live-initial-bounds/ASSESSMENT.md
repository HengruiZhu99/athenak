# Mathematical assessment: live isotropic damping cancellation

No new actual-kernel gate, native build/evolution, or production edit was performed in this assessment. This stage checks source-derived algebra, a highprecision strict-bound counterexample, and continuous initial-data inequalities. It is separate from the prior fixed-Omega profile and its native tests.

With stationary prescribed Omega and w=-beta^i Omega_i/alpha, take

```
kappa_eff=2 beta^i Omega_i+kappa_input Omega,
kappa2=kappa_eff/kappa_input-1.
```

The C0 isotropic physical-Theta source coefficient becomes
`sigma0=(-2alpha*w-kappa_eff)/Omega=-kappa_input` exactly for arbitrary live fields. Thus that contribution to Kij_t is `-kappa_input Theta gammaij`, and its ADM momentum propagation contribution is `+2kappa_input partial_i Theta`, without a coefficient-gradient Theta/Omega^2 term. Other C0 geometric/Z couplings remain; this is not a complete subsidiary energy or regularity closure.

Relative to C0 kappa2=0, let m=kappa_input*kappa2. The exact nonlinear RHS changes are only

```
Delta P_t=Delta Theta_t=-m Theta/Omega.
Delta H_t=-4K m Theta/Omega,
Delta Mi_t=2 partial_i(m Theta/Omega), Delta Zi_t=0.
```

They vanish on the Einstein sector and preserve the exact Minkowski reference. Because the added RHS contains live beta values and prescribed Omega gradients, no new principal derivative is introduced. On an Einstein reference Theta=0, the variation of live kappa2 times backgroundTheta is zero: the linear generator equals that for the prescribed background kappa2hat(r). At finite off-constraint Theta this equivalence no longer holds; value-only beta cross-couplings must be included in actual full20 gates.

For the exact outer reference, general S,a gives `kappa_eff_hat=2S/a^2+(kappa_input-4/a)Omega`. This equals the previous core-matched fixed-Omega profile only when a=S/2, including the current S1,a.5 experiment. It cannot be transferred unchanged across curvature radii.

The admissibility inequalities 0<kappa_eff<=kappa_input are not invariant or guaranteed for arbitrary live fields. The uncut formula fails the upper inequality already for the actual initial .02 angular shift pulse (width.5). A negative-x witness at r=.105795598913780886 gives

```
kappa_eff-10=4.20049952085689458157e-9,
kappa2=4.20049952085689458157e-10>0.
```

The negative-x violation ends at r=.109338809008013085. This is a highprecision mathematical violation, not floating error or a reason to introduce a clamp. It removes the strict all-frequency frozen flat guarantee; it is not automatically a finite-Omega actual-kernel instability. Near the C-infinity geometric cutoff r0=.05, the upper admissible inward radial shift margin tends to zero as `kappa_input*(r-r0)^2/(2*.9)`, whereas this pulse has a nonzero inward radial component.

A distinct candidate can use prescribed V(r)=SmoothCutoff(r,.15,.3) (or zero below.2), with kappa2_blend=V kappa2_live. Then

```
m=V[2beta^j Omega_j+kappa_input(Omega-1)],
m_i=V_i[2beta^j Omega_j+kappa_input(Omega-1)]
    +V[2partial_i beta^j Omega_j+2beta^j Omega_ij+kappa_input Omega_i].
sigma_blend=(1-V)sigma_C0_base-V*kappa_input.
```

The full spatial derivative is essential. Residual sigma gradients are supported in r<.3, where the fixed a.5 compactification satisfies Omega>=1-r^2>=.91; hence inverseOmega^2<=1.20759 there. Above.3 the cancellation is exact for arbitrary live fields. This is a lower-order tensor adjustment, not an automatically covariant or uniformly stable system.

The 50-digit directed mpmath.iv check covers 9000 radial subintervals, handles exponential tails analytically, and uses the exact outer CMC branch. It verifies 0<kappa_eff_hat<=10 on the reference and provides an all-angle bound for the stated initial pulse:

- r>=.15: maximum perturbation/reference upper-margin ratio <=.355188.
- r>=.2: ratio <=.155320.
- The initial effective coefficient has a positive conservative lower bound7.9999987974.

The angular bound is derived from the actual initializer: `|n dot vector|<=sqrt((1+.15r^2)^2+(.2r)^2+(.05r^2)^2)`. Any 0<=V<=1 identically zero below.15 therefore preserves the initial bounds by convexity with the unchanged coefficient10. These are reference/initial-data inequalities, not evolution preservation. Future gates must retain live coefficient derivatives and finite-Theta cross-couplings, and must not impose stronger Theta falloff, a clamp or an Omega floor.
