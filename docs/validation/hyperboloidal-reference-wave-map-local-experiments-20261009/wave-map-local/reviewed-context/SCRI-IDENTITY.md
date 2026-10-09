# Additive radial conformal identity; not an exact-scri closure

This separate derivation supplements the byte-preserved wave-map proposal. It does not change that reviewed source or admit a source/operator run. All formulas use the same stationary Minkowski target, fixed radial Omega, L=Omega-r Omega', and finite Omega>0; r lies away from the exact Cauchy core when denominators are used.

Let rho4=n_i n_j barg4^{ij} and T4=(delta_ij-n_i n_j)barg4^{ij}, with the inverse FOUR-metric, not inverse spatial metric. The stationary physical target connection has

```
n_k GammaHatPhysical^k_ij
 = (L'/L-2 Omega'/Omega) n_i n_j
   - (Omega'/L)(delta_ij-n_i n_j),
GammaHatPhysical^a_0b = 0.
```

These follow directly from the reference inertial spatial map x/Omega: its radial Jacobian eigenvalue is R'=L/Omega² and its tangential eigenvalue is R/r=1/Omega. Contracting the physical reference wave-map gauge gives the exact identity

```
boxBar Omega
 = rho4 [Omega''-Omega' L'/L+4 Omega'^2/Omega]
   + T4 Omega Omega'/(r L) + 2 Omega_i Zbar^i.
```

The spatial source is Fbar^i=barg4^{jk}GammaHatPhysical^i_jk-2 barg4^{ij}Omega_j/Omega; the last Z term must be retained off constraints. On Einstein solutions Z=0, a bounded T4 and rho4=O(Omega²) imply boxBar Omega=O(Omega). Thus this conditional asymptotic class is in preferred conformal gauge. The reference has rho4=Omega²/L² and T4=2; inserting them recovers exactly Omega*What from the original layer reference formula. For a general live metric the identity does not enforce boxBar Omega=Omega*What at finite Omega, and it does not prove preservation of rho4=O(Omega²).

The null residue N=(|D Omega|²-omega_n²)/Omega² is rho4 Omega'^2/Omega². It is finite under that same condition. No evolution closure for N, the shear residue, Theta or the spatial Z variables follows. In particular Zbar^i=gammaBarInv^{ij}Z_j-beta^i ThetaPhysical/(Omega alpha), so arbitrary off-constraint storage data can violate the last-term falloff. Do not impose a stronger Theta falloff solely to remove it.

For the proposed finite-amplitude flat oracle, let Y(X)=X+epsilon a phi(X), with phi=sigma(F(T-R)-F(T+R))/R and smooth localized F. Along future hyperboloidal infinity phi=O(1/R) and its inertial first derivatives are O(1/R). Since r depends only on the reference spatial Y, its physical gradient is

```
partial_A r = (Omega²/L) [n_A+epsilon(n.a) partial_A phi],
```

where n_A=(0,n_I) and a is spatial. Hence rho4=Omega²/L² [1+O(Omega)] for an invertible smooth outgoing oracle. The Einstein-sector gauge-wave family therefore has the required null-residue order. This is an asymptotic derivation for that family, not a numerical test or a theorem for the full Z4c system.

The later black-hole target needs a separate mass-dependent PHYSICAL initial foliation while the REFERENCE remains Minkowski. For example, inserting the Minkowski height slope h_R=1-a²/(2R²)+O(R^-4) into Schwarzschild coordinates with f=1-2M/R gives gammaPhysical_RR=f^-1-f h_R²=4M/R+O(R^-2). Under the same radial compactification, gammaBar_rr then grows like 4M L²/(r Omega). Thus directly pasting that physical height into Schwarzschild is not a regular compactified initial slice. A suitable asymptotic physical height normally includes the mass-dependent outgoing-time behavior; its derivation and constraints must be handled without changing the requested Minkowski reference or manufacturing a black-hole fixed point. This observation does not choose the inner wormhole/trumpet initial data or an evolution gauge.
