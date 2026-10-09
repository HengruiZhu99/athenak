# Joint leading preferred-conformal outer source: necessary terms, not a candidate admission

This is an additive pencil derivation following the conformal-reference feasibility assessment. There is no implementation, CAS, numerical evaluation, kernel query, spectrum or evolution. The Minkowski reference remains fixed and physical-P storage/evolution remain unchanged. The question is what leading temporal AND spatial source terms could allow the conditional stationary massive end that the unmodified physical and conformal reference wave-map conditions obstruct.

The construction below is only a leading-asymptotic source surrogate on a specified stationary spherical branch. It does not measure mass in generic angular/radiative data, establish a stationary solution, close the off-constraint hierarchy, or authorize use in the later wormhole-to-trumpet problem.

## 1. A preferred Box condition fixes the radial offset

Take physical Schwarzschild areal R, F=1-2M/R, and a future hyperboloidal height with hBH'=F^-1-aB^2/(2R^2)+O(R^-3). Let the fixed reference inertial radius be f=r/Omega=R+d+O(R^-1). The prescribed outer CMC compactification is

```text
Omega(f)=S/[sqrt(f^2+a^2)+a].
```

Set p=a+d. Then Omega=S/R-Sp/R^2+O(R^-3). For any stationary scalar Omega(R), with barg=Omega^2 g,

```text
Box_barg Omega=Omega^-2{(R^2 F Omega_R)'/R^2
                                  +2F Omega_R^2/Omega}.
```

The height does not enter this scalar identity, since Omega is stationary. Expanding directly and then expressing R^-1 in Omega gives

```text
Box_barg Omega=2Omega/S^2-(2p+6M)Omega^2/S^3+O(Omega^3).
```

The same formula at the Minkowski reference is

```text
Bhat=Box_barghat Omega=2Omega/S^2-2a Omega^2/S^3+O(Omega^3).
```

Matching the preferred source Bhat through the displayed next coefficient requires d=-3M. The leading boundary statement Box Omega=0 alone does NOT fix d. This is a particular next-order prescribed-source choice, not the definition of preferred conformal gauge in all settings. No conclusion about the higher radial coefficients follows from this two-term calculation.

## 2. Temporal forcing needed on that same branch

In target reference inertial coordinates use Y0=T+psi(R), YI=f(R)nI. For the smooth mass-corrected height and f=R-3M+O(R^-1),

```text
psi=hhat(f)-hBH,  psi'=-2M/R+O(R^-2).
```

The physical-reference connection residual Cphysical=g^{bc}(Gamma[g]-Gamma[ghat]) on Einstein data satisfies Cphysical^A=-Box_g Y^A. Its leading inertial components are

```text
Cphysical^Y0=+2M/R^2+O(R^-3),
Cphysical^f=-4M/R^2+O(R^-3).
```

The second term follows from (R^2F f')'-2f=4M+O(R^-1). Converting to the COMPACT time t=Y0-hhat(f), and to the compact radial coordinate r(f), gives

```text
Cphysical^t=+6M/R^2+O(R^-3),
Cphysical^r=-4M Omega^2/(L R^2)+O(Omega^2 R^-3),
L=Omega-r Omega'.
```

Thus modifying the physical-reference GH source Fp requires, on this branch,

```text
Delta Fphysical_reference^0=+6M/S^2+O(Omega),
Delta Fphysical_reference^r=-4M Omega^2/(L S^2)+O(Omega^3).
```

The latter is the spatial modification associated with the simultaneous preferred Box condition; ignoring it would retain the old radial harmonic offset f=R-M. The earlier temporal +2M/S^2 result was for that DIFFERENT unprojected spatial-harmonic branch. These two coefficients must not be mixed.

For the CONFORMAL-reference source Fc the stationary temporal wave-map equation contains the extra term 4F psi'f' k, k=dlogOmega/df. With psi'=-2M/R and k=-1/R+O(R^-2),

```text
Box_g Y0+4F psi'f' k=+6M/R^2+O(R^-3),
Hconformal^0=-6M/S^2+O(Omega).
```

Its compact temporal correction therefore has the opposite leading sign,

```text
Delta Fconformal_reference^0=-6M/S^2+O(Omega).
```

The conformal spatial equation already selects d=-3M at this leading order. Any residual spatial projection needed for exact Bhat begins at higher order on the stated smooth stationary branch; no exact all-orders vanishing is asserted. The two sources are different off-reference gauges, related by U/Omega in the feasibility assessment, so their opposite temporal coefficients are consistent.

## 3. A reference-preserving way to display those coefficients

In an outer collar let s_area be the fourth root of the determinant of the Penrose SPATIAL metric restricted to an orthonormal Euclidean tangent plane. At the exact outer Minkowski reference s_area=1. On the stationary spherical branch above,

```text
s_area=R/f=1+3M/f+O(f^-2)=1+3M Omega/S+O(Omega^2).
```

Consequently the value-only expressions

```text
Delta Fp^0=+2(s_area-1)/(r Omega),
Delta Fc^0=-2(s_area-1)/(r Omega)
```

supply their respective required leading temporal coefficients and vanish exactly on the Minkowski reference. This only exhibits the leading algebraic freedom. In a general state s_area is an angular area factor and can change under a pure radial coordinate deformation even when physical mass is zero; it is NOT a justified mass estimator. The source is bounded only in the stated class s_area-1=O(Omega). Generic finite off-constraint metric deviations give a simple pole. The expression therefore needs its own complete pole/jet/Jacobian analysis before being treated as a viable gauge source. No falloff is enforced here to make it pass.

For either base source Fb=Fp or Fc, a spatial source projection in a collar with |dOmega|_E^2>0 is algebraically

```text
E=barg^{ij}Omega_ij-Fb^i Omega_i-Bhat,
Delta Fb^i=Omega_i E/|dOmega|_E^2.
```

Together with the displayed TEMPORAL correction, this yields

```text
Box_barg Omega=Bhat+2Zbar^a Omega_a.
```

It sets the chosen preferred source on the Einstein sector. The Z term is retained off constraints. Removing it would be another source modification involving spatial Z and potentially derivative principal changes, so that is not done. The projection source uses only live metric values and fixed reference/Omega jets; the temporal area source also uses only live metric values. They therefore retain the finite-positive-Omega harmonic derivative principal part, while their off-reference lower-order pole block may be very different. No actual principal/pole stability result is inferred.

Both corrections need smooth outer support avoiding r=0 and gradOmega=0; the exact reference support numerator vanishes. This support requirement cannot be replaced by a denominator floor. Because Omega is stationary, changing F0 does not directly change the scalar Box source, but it changes BOTH alpha and beta gauge rows through

```text
Delta alpha_t=-alpha^3 Delta F0,
Delta beta_t^i=-alpha^2(Delta Fi+beta^i Delta F0).
```

Those paired changes are necessary. Keeping the old temporal row and projecting only the spatial source is not this joint construction.

## 4. What remains unclosed

At most these corrections remove the displayed leading stationary mass-log contradiction on the chosen spherical branch. The preferred Box expansion fixes the radial mass offset; the paired temporal correction supplies the missing height response. It remains to derive a complete nonlinear source from an invariant or explicitly justified gauge prescription, solve higher stationary coefficients, characterize the null/shear/trace/Z/Theta domain and its time tangencies, and demonstrate retention of angular physical data. A reference-preserving area proxy alone accomplishes none of those tasks.

The physical-P lapse stabilization of the earlier driver is not retained simply by retaining P storage. A complete gauge alternative needs its own off-constraint damping mechanism and pole/finite-frequency/constraint gates. The existing negative Q/null-feedback and physical-reference finite pulses cannot be repaired by a sign or coefficient change without those gates. No new candidate, blend, boundary condition or native evolution is admitted by this note, and no BH inner choice is made.
