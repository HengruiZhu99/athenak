# Energy and asymptotic scope for the damped C1 constraints

The repaired physical equations are the covariant Z4 equations at finite Omega.
Their Bianchi consequence, with kappa2=0 and variable kappa1=kappa_input/alpha,
has the covector-wave form

```
box_phys Znu + Rnu^mu Zmu
 -nabla_mu[kappa1(n^mu Znu+nnu Z^mu)] = 0.
```

Derivatives of kappa1 and the hyperboloidal normal must remain inside this
divergence. Replacing the damping by a constant-coefficient inertial friction
equation is not justified. On the Minkowski physical reference Rmunu=0, but
normal/extrinsic-curvature/acceleration and coefficient gradients remain.

Principal wave variables U=H+2divphys Z and V=M+gradphys Theta give the same
four-wave principal subsystem. A positive nonzero-frequency principal energy
contains U^2/4+|V|^2+|grad Theta|^2+|grad Z|^2 in a physical orthonormal frame.
It is not a native unweighted H/M/Z RMS, nor is principal positivity a complete
lower-order or boundary energy estimate.

In the exact outer CMC collar (S arbitrary), physical Kij=-gammaij/a and
alpha+Omega=S/a. The isotropic Theta source in S1 is sigma*Theta*gammaij with
sigma=(2alpha/a-kappa_input)/Omega. Its contribution to Mi_t is exactly

```
-2sigma*d_iTheta
 + [2r/(aS)]*(kappa_input-2S/a^2)*Theta*n_i/Omega^2.
```

For S1,a=.5,kappa10, this is the same +8r*Theta*n_i/Omega^2 coefficient-gradient
term found in the C0 audit. Thus C1 does not remove the obstruction to the
simple uniform unweighted M^2+Theta^2 damping estimate. A Young inequality
using only O(1/Omega) diagonal damping asks for stronger weighted control;
an adapted wave energy, cross terms, justified Hardy regularity and boundary
flux remain to be derived. At the special kappa_input=2S/a^2 this particular
coefficient-gradient term vanishes, but the Lambda<-Theta double pole remains.
This does not establish an energy bound at that parameter.

Natural physical components do explain part of the coordinate amplification:
|Z|phys^2=Omega^2*chi*gtilde_inverse^ij ZiZj, and likewise for M. In the exact
outer CMC reference chi=1,gtilde=I, so physical orthonormal M/Z components are
Omega times their coordinate covectors. The raw C1 Lambda transient can grow
as1/Omega while this physical spatial norm stays bounded. The primitive
analysis weighting Omega*Atilde also corresponds to physical orthonormal
trace-free extrinsic-curvature components. This is an interpretation of units,
not a proof that the previously chosen full primitive weighted norm is a
symmetrizer, nor an invariant falloff theorem.

An exact undamped Minkowski covector wave can have finite nonzero physical
Theta and coordinate Zr~1/Omega. Other polarizations or exact gradients have
different powers. Those kappa0 examples forbid assuming that stronger Theta
falloff follows from bounded physical components. Actual positive damping and
the nonconstant hyperboloidal normal can alter asymptotic amplitudes and
polarizations. No undamped falloff is transferred to kappa10 here. An asymptotic
indicial calculation, if pursued, supplies only necessary leading compatibility,
not evolution preservation of a nonlinear scri manifold.

The hyperboloidal normal degenerates relative to inertial Killing time near
scri. Any proposed positive global energy must state its multiplier, internal
component norm, volume and boundary flux, and establish uniform equivalence
rather than infer it from local orthonormal positivity. This audit establishes
no uniform global/scri energy estimate. The bulk blend adds bounded compactly
supported grad-c terms; its outer C0 regularity and energy questions remain.
