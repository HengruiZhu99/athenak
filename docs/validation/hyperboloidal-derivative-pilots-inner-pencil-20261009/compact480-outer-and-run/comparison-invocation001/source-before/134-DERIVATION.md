# Exact-flat physical-reference wave-map pulse IVP

Independent pencil/source-only check, 2026-10-09. No CAS, numerical, kernel,
operator, scalar-solver or native calls. This concerns the **physical** Minkowski
reference wave-map gauge used by the current candidate, on exact Einstein data.
It is not the distinct conformal-reference wave-map proposal, not an admitted
new scalar control, and not the previously tested manufactured dipole.

## Physical hypersurface, target and initial lapse/shift

Use physical inertial coordinates X^A with eta of signature -+++. Fix the
reference embedding of the initial native hypersurface Sigma:

```
Xhat^0=h_height(R), Xhat^I=x^I/Omega, R=r/Omega,
E_i^A=partial_i Xhat^A,
```

and its future physical unit normal n^A. The reference height is the complete
retained layered height, not a CMC formula substituted through the transition.
Its exact outer branch is sqrt(R^2+a^2)+constant and its core height is zero.
The live spatial metric and Kij initially equal the reference. Thus Sigma,
E_i and n are fixed by the same exact-flat embedding; the prescribed lapse and
shift specify the initial time-coordinate map rather than different physical
initial geometry. All P/A/connection/Theta conventions remain unchanged.

Write h=alphahat_bar for the **reference conformal lapse**, distinct from
h_height. Live conformal lapse is alpha>0, physical lapse alpha/Omega, and
shift beta. The reference identity is

```
delta_0^A=(h/Omega)n^A+betahat^i E_i^A.
```

Seek four physical scalar functions Y^A(X) satisfying `Box_eta Y^A=0`, with
`Y^A=X^A+deltaY^A`. On Sigma impose deltaY=0. The native target is the reference
inertial embedding `Yhat^A(t,x)=(t+h_height(r/Omega),x^I/Omega)`.
Require `Y(X(t,x))=Yhat(t,x)`. Initially,

```
partial_t X^A=(alpha/Omega)n^A+beta^i E_i^A,
partial_t Yhat^A=delta_0^A.
```

Since deltaY vanishes on the full initial hypersurface, its tangential
derivatives vanish there. Define `s^A=n^B partial_B deltaY^A`. The chain rule
then gives the exact nonlinear initial normal data

```
s^A=Omega[(1/alpha-1/h)delta_0^A
          -(beta^i/alpha-betahat^i/h)E_i^A]
   =(h/alpha-1)n^A-(Omega/alpha)(beta^i-betahat^i)E_i^A.
```

The signs match the parent formula. In the inertial core with constant alpha
and beta, they give `Y0=T/alpha`, `Yi=Xi-beta^i T/alpha`; inversion yields
`T=alpha t`, `Xi=x^i+beta^i t`, whose pulled-back metric has exactly that positive
lapse and shift. The inverse-map active linear coordinate generator is
`xi^A=-deltaY^A`, not +deltaY.

## Initial map determinant: a useful exact identity

On Sigma, `partial_B deltaY^A=-s^A n_B`, because n is unit timelike and all
tangential derivatives are zero. Therefore the forward map Jacobian is

```
J^A_B=delta^A_B-s^A n_B,
detJ=1-n_A s^A=h/alpha.
```

The shift term drops out since n_A E_i^A=0. Positive alpha gives positive
initial determinant; it is a local initial invertibility fact, not a bound on
the future map. For the prescribed positive-A native pulse, the angular lapse
factor obeys `1+.2x+.3yz>=.65` on r<1 at S=1, since |x|<=1 and |yz|<=1/2.
With nonnegative radial envelope, alpha>=h>0 initially. No conclusion about
later Jacobians follows from this elementary initial bound.

## Four conformal scalar equations on the fixed reference background

Each deltaY^A is a scalar component in a fixed physical inertial target frame.
Let `barghat=Omega^2 eta` in the reference hyperboloidal coordinates. The
four-dimensional conformal scalar identity, with R[eta]=0, is

```
(Box_barghat-Rbarhat/6)(deltaY^A/Omega)
     =Omega^-3 Box_eta deltaY^A.
```

Consequently `phi^A=deltaY^A/Omega` satisfies four independent linear equations
on the fixed background,

```
(Box_barghat-Rbarhat/6)phi^A=0,
phi^A|Sigma=0,
partial_tau_ref phi^A|Sigma=h*s^A/Omega^2.
```

The last identity uses `bar n=n/Omega` and
`partial_tau_ref=h bar n+betahat^i E_i`; the tangential derivative of phi is
zero on Sigma. The derivative of Omega in the quotient contributes nothing
initially because deltaY=0. A sign reversal in the scalar curvature term would
be a different conformal wave equation.

The equations are linear, but their initial velocities depend rationally and
nonlinearly on the finite lapse perturbation through 1/alpha. Reconstructing
the native metric is nonlinear as well. This is a finite-amplitude exact-flat
Einstein-sector construction, not a linearized gauge-amplitude claim.

## Outer orders and angular completeness

For the complete reference, let b be its radial boost and L=Omega-rOmega_r.
The physical normal and radial embedding derivatives have the useful forms

```
n^0=h/Omega, n^I=(b/Omega)nu^I,
partial_i R=(L/Omega^2)nu_i,
E_i^0=(bL/(h Omega^2))nu_i,
E_i^I=Omega^-1(delta_i^I-nu_i nu^I)
       +(L/Omega^2)nu_i nu^I,
```

where nu is the Euclidean unit radial direction; the regular core limits must
be taken in Cartesian form rather than dividing by r at the origin. With
deltaalpha=alpha-h and deltabeta=beta-betahat, the exact normal components are

```
s^0=-(h deltaalpha+(bL/h)deltabeta_radial)/(alpha Omega),
s^I=-(b deltaalpha+L deltabeta_radial)nu^I/(alpha Omega)
     -deltabeta_tangent^I/alpha.
```

For generic deltaalpha,deltabeta=O(Omega^m), radial/time s are O(Omega^(m-1))
and conformal initial velocities are O(Omega^(m-3)); purely tangential terms
can be one order smaller. Cancellation of the leading radial combination
must be demonstrated, not assumed.

The native envelope is `(1-r^2)^4 exp(-r^2/.35^2)` at S=1. In the exact outer
branch `1-r^2=2aOmega`, so the prescribed lapse and shift perturbations are
O(Omega^4). Hence generic inertial normal data are O(Omega^3), and initial
conformal scalar velocities are O(Omega), finite and vanishing at scri. These
are initial orders only. Radiation later reaching scri need not preserve a
homogeneous zero scalar boundary value; no artificial Dirichlet condition is
justified by this initial vanishing.

The finite native pulse's angular numerator is polynomial, but division by
`h+deltaalpha(r,angles)` generically produces infinitely many spherical
harmonics. An arbitrary fixed finite-l truncation is not the exact IVP. It
requires its own controlled angular convergence/truncation evidence. Four
spherically symmetric scalars, a single dipole, or the existing manufactured
scalar isolate cannot replace this angular-rational data. Origin smoothness
and the complete reference transition coefficients remain necessary.

## Reconstruction and the distinct target time

Given the four physical wave solutions, recover physical X for each desired
native point by solving the full spacetime equation

```
Y^A(X)=Yhat^A(t_native,x_native).
```

The physical metric is then eta pulled back by this inverse map. Reference
wave-solver time is `tau_ref=X0-h_height(|XI|)`, whereas native target time is
`t_native=Y0-h_height(|YI|)`. They generally differ at finite amplitude. A scalar
snapshot at tau_ref=2 on the reference spatial grid is not a metric snapshot
at native t=2. Full inversion and evaluation/interpolation at the resulting
physical events are required; time synchronization cannot be supplied by
renaming a scalar solver's time coordinate.

Local nonzero det(partial Y/partial X) is necessary for inversion. It is not
sufficient for global injectivity, suitable domain coverage, or a spacelike
native foliation. Also monitor the target-time gradient

```
t_native(X)=Y0(X)-h_height(|YI(X)|),
eta^{AB}(partial_A t_native)(partial_B t_native)<0
```

with the future orientation retained. The reconstructed physical lapse is
`[-eta^{-1}(dt_native,dt_native)]^-1/2` on that branch; the conformal lapse is
Omega(target)*physical_lapse. A coordinate caustic and a failed target-time
foliation are distinct potential obstructions, even while physical spacetime
is exactly flat. Hyperboloidal domain-of-dependence coverage and causal scri
treatment must also be justified independently.

If a converged reconstruction remains regular through the same native events,
the exact geometry is flat and H/M/Z/Theta are zero there. That could provide
a useful Einstein-sector continuum comparator for a native constraint failure.
If the exact map or target foliation fails first, that is instead a possible
continuum coordinate limitation. Neither outcome is established here. Scalar
energy control alone does not bound the inverse-map Jacobian, and finite scalar
resolution alone does not establish exact geometric constraints.

Off-constraint C0 Z4c evolution, native discretization/ghost closure, stage
projection and lower-order subsidiary behavior are outside this exact-Einstein
construction. No implementation, scalar propagation or native acceptance is
authorized. Later BH survival still requires an independently justified inner
wormhole-to-trumpet gauge and mass-consistent outer foliation with the Minkowski
reference retained; this exact-flat pulse IVP does not supply them.
