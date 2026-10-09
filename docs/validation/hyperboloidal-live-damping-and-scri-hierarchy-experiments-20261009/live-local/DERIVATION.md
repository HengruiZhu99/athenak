# C0 live damping with a prescribed inner turn-on

This is a finite-positive-Omega candidate on the unchanged Cartesian C0 Z4c system. The physical-P lapse, spatial-norm shift feedback, principal equations, kappa_input=10, reference and storage remain unchanged. There is no C1 addition, covector repair, new double-pole term, Omega floor, clamp or imposed Theta falloff.

Let b=beta^i Omega_i, w=-b/alpha, Theta denote the stored physical Theta, K=P+2Theta the physical ADM trace, V(r)=SmoothCutoff(r,.15,.3), and kappa=kappa_input. Replace only the existing kappa2 argument by

```
kappa2 = V [2 b/kappa + Omega - 1],
m = kappa kappa2 = V [2 b + kappa(Omega-1)].
```

It is a live value coefficient: beta is the live shift, Omega and its jets are the prescribed compactification. The evolution uses no derivative of beta in this coefficient. The shared helper returns exactly zero for r<=.15. For r>=.3,

```
kappa_effective = kappa(1+kappa2) = 2b+kappa Omega,
sigma_C0 = [-2alpha w-kappa_effective]/Omega = -kappa.
```

For all V the exact numerator identity is

```
-2alpha w-kappa_effective
 = (1-V)(-2alpha w-kappa)-V kappa Omega.
```

Thus the isotropic Theta coefficient gradient in the physical Kij equation disappears beyond .3 for arbitrary live fields. This cancels one subsidiary coupling; it does not provide closure or an energy estimate for the remaining C0 system. Within the turn-on, all coefficient derivatives must remain:

```
m_i = V_i [2 beta^j Omega_j+kappa(Omega-1)]
    + V [2 (partial_i beta^j)Omega_j+2 beta^j Omega_ij+kappa Omega_i],
V_i = V'(r)n_i.
```

The actual production normalization is kappa1=kappa_input/alpha. Comparing the existing C0 equations at fixed live state gives

```
Delta P = Delta Theta = -m Theta/Omega,
Delta K = -3m Theta/Omega,
Delta Kij = -(m Theta/Omega) gammaij,
Delta H = -4K m Theta/Omega,
Delta Mi = 2 partial_i(m Theta/Omega)
         = 2m partial_i Theta/Omega
           +2m_i Theta/Omega-2m Omega_i Theta/Omega^2,
Delta Zi = 0.
```

Here H=R+K^2-KijK^ij and Mi=DjK^j_i-DiK; these are variations of their evolution, not changes to the constraints' definitions. The additions vanish on the Einstein sector. Their spatial derivatives remain homogeneous in constraints. Since the coefficient is value-only, the original complete principal system is unchanged. On an Einstein reference Theta=0, its first variation equals a prescribed hat-kappa2(r); finite nonzero Theta adds the beta columns

```
Delta L[P,beta_j] = Delta L[Theta,beta_j]
                 = -2V Theta Omega_j/Omega,
Delta L[P,Theta] = Delta L[Theta,Theta] = -kappa kappa2/Omega.
```

For the exact outer CMC reference, alpha=S/a-Omega and alpha w=-S/a^2+2Omega/a, hence

```
hat-kappa_effective = 2S/a^2 + (kappa-4/a)Omega.
```

It agrees with the earlier fixed profile only for a=S/2. All four a=.5,.75,1,2 are re-evaluated here rather than transferring that earlier spectrum.

The separate immutable math assessment (index SHA7f253e0ff6053543514831c3efb5b55cc1910aa52fdca875e6f9e9df3d40bef3) contains an explicit high-precision positive-kappa2 witness for the uncut live coefficient and interval bounds for the actual a=.5 initial angular pulse after V(.15,.3). These are initial-data bounds only. No nonlinear evolution preservation of 0<kappa_effective<=kappa_input follows. The all-a local kernel checks here do not extend those initial-pulse bounds to other a.

For comparison only, the flat constant-parameter subsidiary matrix in Gundlach et al., primary https://arxiv.org/pdf/gr-qc/0504114 Eq19, has rho=kappa2 and polynomial

```
(s^2+kappa(2+rho)s+wavenumber^2)(s^2+kappa s+wavenumber^2)
 - kappa^2 rho wavenumber^2.
```

Its all-nonzero-frequency Hurwitz interval is -1<rho<=0, not the broader prose condition rho>-1: positive rho gives a negative constant coefficient at sufficiently small nonzero frequency. The independent exact mapping/Routh index is SHA3a2b38c659820243b4d850afeff0d819e4c946f01e25cf22ae6d8fb5ababd5d6. This constant inertial result is not a variable-coefficient hyperboloidal/C0 theorem and is not a clamp or admission condition automatically preserved by evolution.

The eight-constraint chain is linearized about the stationary Einstein reference with all coefficient derivatives, using actual full20 dual derivatives and five refined coefficient-FD steps. Coefficient-aware local generators are still not the global constraint operator: spatial variation, boundaries, variable-coefficient energy/volume/flux and numerical product identities remain necessary. Bounded positive inner k0 roots already occur in the matched C0 baseline and are retained. No uniform puncture/scri statement, nonlinear Bianchi closure or black-hole integration is established.
