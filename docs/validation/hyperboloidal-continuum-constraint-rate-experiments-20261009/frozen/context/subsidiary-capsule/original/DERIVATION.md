# Exploratory continuum constraint propagation (C_Z4c=0)

This audit uses the actual tensor ConformalRHS at stationary analytic
LayerReference, S=1,a=.5,wide .05-.95,kappa_input10. No production edits or
C_Z4c=1 implementation. Gauge RHS choices cannot enter the instantaneous
H/M/Theta/Z chain rule, because these constraint functions do not depend on
lapse/shift. The geometric RHS does depend on their spatial jets.

Write K=P+2Theta, Z^i_tilde=gtilde_inverse^ij Z_j, w=-beta.Omega/alpha,
and kappa_input=alpha*kappa1 (the native runtime convention). Actual exact
constraint equations derived directly from the lambda-minus-connection and
Theta kernels are

```
Theta_t = beta^i d_i Theta + alpha H/(2 Omega)
          + alpha Omega chi Dtilde_i Z^i_tilde
          - [3 alpha w + (2+kappa2)kappa_input]Theta/Omega,
Z_i,t = beta^j d_j Z_i + Z_j d_i beta^j
        + gtilde_ij Z^k_tilde d_k beta^j
        - 2 alpha A_ij Z^j_tilde
        - [(2/3)alpha K + kappa_input]Z_i/Omega
        + alpha (M_i+d_i Theta)/Omega.
```

The extra gtilde_ij Z^k d_k beta^j is intentional: the actual C0 lambda
kernel transports the connection term with -Gamma^j d_j beta^i, rather
than -Lambda^j d_j beta^i. The `s.lambda -= gradPTheta + 4alpha*Aup*dOmega`
source subtracts BOTH terms; its curvature/Omega-gradient sign is negative.

For a full physical ADM subsidiary formulation, let gamma=Omega^-2*gtilde/chi,
lapse A=alpha/Omega and physical K_ij=Atilde_ij/(Omega chi)+gamma_ij K/3.
The actual C0 K_ij RHS differs from vacuum ADM by the constraint addition

```
S_ij = A (Dphys_i Z_j + Dphys_j Z_i) + gamma_ij B/3,
B = alpha Omega Z^i_tilde d_i chi + 2 alpha chi Z^i_tilde d_i Omega
    - [6 alpha w + 3(1+kappa2)kappa_input]Theta/Omega.
```

This follows because the actual trace rate differs from ADM by
2alpha Omega chi div_tilde Z - [6alpha w+3(1+kappa2)kappa_input]Theta/Omega,
while its tracefree rate adds A(Dphys_iZ_j+Dphys_jZ_i)^TF. The identity
Dphys_i Z^i_phys = Omega^2 chi div_tilde Z
                 - Omega^2 Z^i_tilde d_i chi/2
                 - Omega chi Z^i_tilde d_i Omega
reconciles the traces. As a cross-check, the independently derived C1
additions cancel the two spatial-Z terms of B and supply -2A Theta K_ij;
this does not authorize their implementation or address their double pole.

The physical subsidiary equations independently checked below are the ADM
Bianchi identities with these explicit constraint additions:

```
H_t = beta.dH + 2 A K H - 2 A Dphys_i M^i - 4 M^i d_i A
      + 2(K gamma^ij-K^ij)S_ij,
M_i,t = beta^j d_j M_i + M_j d_i beta^j + A K M_i
        - (A/2)d_i H - H d_i A + Dphys^j S_ij - d_i tr_gamma S.
```

At stationary constraint-satisfying reference, linearizing these equations
has no coefficient-variation times background constraints. Their coefficients
and their gradients must still be retained when differentiating S_ij.
The independent full20 dual-chain-rule comparison now passes for H/M as well
as Theta/Z: 700 samples for each of kappa_input=5,10, including seven radii,
ten frequencies through256, radial/oblique phases and five h refinements.
At h=.000125 the maximum matrix-relative error is3.61e-9. The method is a
compiled numerical algebraic identity audit, not a formal nonlinear proof.

## Measurement

`constraint_tangent.cpp` uses exact first dual derivatives of the actual
geometric kernel and of EvolvedConstraints. Twenty independent fields obey
det=1/tracefree algebraic reconstruction including their jets. It computes
complex plane-wave coefficient matrices L(x,k), Q(x,k). Only the smooth
coefficient L is differentiated with fourth-order centered formulas; the
phase is differentiated analytically:

```
d_i (L exp(ik.n.x)) = (d_i L + ik n_i L)exp(ik.n.x),
d_ij = (d_ij L + ik[n_i d_j L+n_j d_i L]-k^2 n_i n_j L)exp(ik.n.x).
```

The correct constraint tangent D(C)[L] includes those coefficient gradients.
The comparison `frozen` replaces them by zero and therefore computes the
naive pointwise Q L. In this variable-coefficient system those two operations
do not commute. Pure lapse/shift columns have Q=0 identically at every x;
their Cdot therefore gives an unambiguous gauge-tangency test. General
nonzero primitive eigenvector constraint residues alone do not identify an
eigenmode of the physical subsidiary system. No global spectrum, energy
estimate, numerical Bianchi identity or nonlinear regularity closure follows
from this local measurement.

## Damping and energy scope

For the native kappa_input10 case, all140 directly derived eight-constraint
generator samples through k256 have negative roots. At kappa_input5, the
outer scalar branch has positive local roots. The extended local high-k
scan remains a finite frozen-coefficient measurement. In particular, neither
these roots nor the primitive full20 roots settle the variable-coefficient
initial-boundary-value problem or a physical constraint energy.

There is a concrete obstruction to a simple uniform monotone native-RMS
argument from kappa10 damping alone. The isotropic S term contributes
-(2/3)d_i(B_Theta*Theta) to M_i,t. On outer S1,a=.5 CMC, Omega=1-r² and
alpha*w=-4r², so at kappa_input10

```
B_Theta=-(30-24r²)/Omega,
M_r,t contains (20-16r²)*d_r Theta/Omega +8r*Theta/Omega².
```

The ordinary unweighted M²+Theta² damping is only O(1/Omega). Its cross term
M_r*Theta/Omega² cannot be uniformly absorbed by those terms as Omega->0;
a direct Young bound asks for Theta²/Omega³ control. A stronger/adapted
energy, cross-variable cancellation or a justified Hardy/regularity condition
would be needed. No arbitrary live Theta falloff is imposed, and the finite-Q
counterexample means such an invariant regularity condition is not established.
This is an obstruction to this simple estimate, not a proof that no energy
estimate exists. On a strict interior region Omega>=delta>0, the smooth
hyperbolic coefficients permit local bounds with constants depending on delta;
this does not give a uniform scri estimate or boundary flux sign.

The separately measured native global C_h L_h gauge tangent is nonzero, whereas
these coefficient-aware continuum equations preserve C=0 for pure gauge
perturbations. That is a numerical Bianchi/product-rule defect to quantify with
the actual final-stage projector and global boundary operator, not a continuum
physical source inferred from primitive frozen eigenvector residues.
