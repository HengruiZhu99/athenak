# Principal constraint-wave energy and limits of the damping argument

At a frozen physical orthonormal point, write A=alpha/Omega, H and M_i for
physical ADM constraints, and Theta,Z_i for the physical Z4 constraints.
After retaining only principal derivative terms, the derived C0 equations are

```
Theta_t=A(H/2+div Z),      Z_i,t=A(M_i+d_i Theta),
H_t=-2A div M,            M_i,t=-A d_i H/2+A(lap Z_i-d_i div Z).
```

Introduce U=H+2 div Z and V_i=M_i+d_i Theta. Then, at principal order,

```
Theta_t=A U/2,     U_t=2A lap Theta,
Z_i,t=A V_i,      V_i,t=A lap Z_i.
```

Thus the principal subsystem is four wave equations with the physical light
cones. Its constant-coefficient energy is proportional to

```
(U/2)^2 + |V|^2 + |grad Theta|^2 + |grad Z|^2.
```

This explicitly shows why native H/M/Z Cartesian RMS is not the constraint
wave energy: derivatives and cross terms are part of that energy. It also
shows a complete principal wave reduction on every strict interior positive
metric/lapse region, without imposing a live field weight or changing the
runtime variables. The physical/conformal coefficients restore the same
coordinate light speed alpha*sqrt(chi*gtilde_inverse^nn).

Lower-order C0 terms remain important. The Theta and Z transport/damping
identities are in DERIVATION.md. In particular the isotropic curvature source
B_Theta Theta produces +8r Theta/Omega² in radial M_t at S1,a=.5,kappa10.
A naive positive combination of M² and Theta² cannot absorb its cross term
uniformly with only the O(1/Omega) damping: a direct Young estimate needs
Theta²/Omega³ control. The principal energy's V=M+gradTheta and U=H+2divZ
cross terms may reorganize parts of this coupling, so this observation is
not a no-energy theorem. A full estimate must include all coefficient
gradients, commutators and boundary fluxes in the chosen measure.

Physical spatial volume itself behaves as Omega^-3 near scri. A physical
energy needs suitable asymptotic integrability; a conformal-volume energy
changes coefficient-gradient terms. Neither the finite-Q condition nor the
current equations prove an invariant Theta falloff adequate for a Hardy
argument. No such falloff is imposed here. The local negative kappa10 roots
therefore do not certify a uniform scri energy estimate, global continuum
stability, or the native boundary/FD/RK discretization. The independently
measured nonzero native C_h L_h gauge tangent is evidence of a numerical
Bianchi/product-rule defect that is absent from the continuum equations.
