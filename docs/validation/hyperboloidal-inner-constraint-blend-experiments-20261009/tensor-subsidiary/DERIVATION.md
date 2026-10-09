# Repaired C1 reference subsidiary and prescribed bulk blend

All formulas use the physical metric gamma=Omega^-2*gtilde/chi, physical lapse
A=alpha/Omega, physical K=P+2Theta and physical covector Z_i. Runtime
kappa1=kappa_input/alpha; kappa2=0. The separate tensor identity gate validates
the mechanical C1 additions plus the connection repair
DeltaLambda^i=-2Ztilde^j*d_j beta^i. The repaired system's physical equations are

```
Kij_t = ADMij + S1ij,
S1ij = A(DiZj+DjZi-2Theta Kij)-A*kappa1*Theta*gammaij,
Theta_t = beta.dTheta + A(H/2+divphys Z-K Theta)
          -Zphys^i*d_i A-2A*kappa1*Theta,
Zi_t = Lie_beta Zi + A(Mi+d_iTheta-2Ki^j Zj-kappa1 Zi)-Theta*d_i A.
```

These give the eight-constraint equations by the physical ADM Bianchi identities:

```
H_t = beta.dH+2AKH-2A divphys M-4Mphys^i*d_i A
      +2(K gamma^ij-K^ij)Sij,
Mi_t = Lie_beta Mi+AKMi-(A/2)*d_i H-H*d_i A
       +Dphys^j Sij-d_i(gamma^jk Sjk).
```

The numerical audit linearizes on the stationary constraint-satisfying analytic
reference. Background constraints vanish, so coefficient variations multiplying
background constraints drop out; coefficient spatial gradients remain. The
source used here is S1 itself, not the C0 isotropic-B source. Both Kij and its
spatial derivatives are reconstructed from the actual reference jets. The
derivative of A*kappa1=kappa_input/Omega is included explicitly.

The full20 actual kernel uses physical-P norm gauge, with complete algebraic
metric/A jet reconstruction. Its exact dual tangent provides L(x,k), while the
actual physical constraint kernel provides Q(x,k). Fourth-order differences
differentiate only smooth matrix coefficients; Fourier phase jets are analytic.
The correct constraint rate differentiates L, including its first/second spatial
coefficient derivatives. The subsidiary prediction likewise differentiates Q.
The naive frozen QL operation is retained as a negative diagnostic. Static gauge
columns of Q are exactly zero; coefficient-aware Cdot converges to zero for them.
The result is a reference tangent identity, not a full nonlinear chain-rule proof.

## Prescribed bulk tensor blend

Define c(r)=1-W_gauge(r), using the existing smooth .45--.85 cutoff. Apply c to
every mechanical C1 addition and to the connection repair. The new shared
`bulk_c1_additions.hpp` does exactly that, preserving explicit regular/simple/
double-pole parts. This is a prescribed lower-order tensor blend, not a claim
that the blended off-constraint system is fully covariant.

Write S0 for the independently derived C0 physical source. Then
S=(1-c)S0+cS1, and the Theta/Z equations blend by the same value c. In the
momentum equation, the extra coefficient-gradient term is

```
Delta Mi_t|gradc = gamma^ja (d_a c)(S1ij-S0ij)
                  -(d_i c) gamma^ja(S1ja-S0ja),
S1ij-S0ij = -2ATheta Kij-A*kappa1*Theta*gammaij-gammaij*B0/3.
```

There is no gradient-c term in H_t or the direct Theta/Z rates. These terms are
homogeneous in constraints and vanish on the Einstein sector. Omitting them
fails the actual20 reference-chain identity at the gauge transition.

For S1,a=.5, geometry .05--.95, Omega=1-W_geo*r^2, and c is supported in r<.85.
Thus Omega>=.2775 there and 1/Omega^2<=12.985958932. For the other checked radii
a=.75/1/2, the general bound is min(1,(1-.85^2)/(2a)). The analytic cutoff is
flat at its endpoints: all ordinary derivatives vanish by exponential
dominance. The compiled gate separately checks value and first three jets.
For r>=.85 the helper returns zero before evaluating the C1 geometry: its
outer kernel and all coefficient jets equal C0 exactly, even off constraint.
No new outer pole is introduced. No Omega floor or live Theta weight is used.

The actual360 symbol gate binds the same prescribed radius to the copied
versioned extractor. It changes no highest derivative term. The copy differs
from the versioned extractor only by that radius binding; its equality is
checked before compilation. Its temporary positive-Omega symbol representation
is not a native double-pole assembly convention.
