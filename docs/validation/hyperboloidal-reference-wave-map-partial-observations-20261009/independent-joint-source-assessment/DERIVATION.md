# Joint preferred-Box and stationary mass-log source necessities

Independent pencil/source-only analysis, 2026-10-09. No CAS, numerical,
kernel/native or operator calls. This follows the conditional stationary note
S01. It identifies necessary leading source terms on a smooth, spherical,
stationary exact-Einstein Schwarzschild end. It supplies neither a full
stationary solution nor an off-constraint closure or candidate admission.

Keep the same Minkowski reference, external Omega, normalized Killing time,
signature -+++ and reference inertial coordinates as S01:

```
F=1-2M/R, Y^0=T+psi(R), Y^I=f(R)n^I,
psi=h_hat(f)-h_BH,
Omega(f)=S/(sqrt(f^2+a^2)+a),
h_hat(f)=sqrt(f^2+a^2)+constant.
```

The live compact radius remains defined by f=r/Omega(r); f(R) is free to differ
from areal R. A smooth future height requires
`h_BH=R+2M log(R/ell)+constant+O(R^-1)`. Consequently, for a smooth radial map
`f=R+d+O(R^-1)`,

```
psi'=-2M/R+O(R^-2).
```

## Preferred BoxOmega fixes a different radial offset

Let `B_*=Box_hatbar Omega` evaluated at the same target point f, and impose
`Box_bar Omega=B_*` on this Einstein sector. The four-dimensional scalar
conformal transformation gives, for u(R)=Omega(f(R)),

```
Box_bar Omega=Omega^-2 [ (R^2 F u')'/R^2+2F u'^2/Omega ].
```

The formula is independent of height: Omega is stationary and radial in the
physical Schwarzschild geometry. The exact radial matching equation is

```
(F f'^2-1)(Omega_ff+2Omega_f^2/Omega)
 +(F'f'+Ff''+2Ff'/R-2/f)Omega_f=0.
```

For positive leading f/R=lambda, the first preferred-Box coefficient forces
lambda=1. Write `f=R+d+O(R^-1)` and p=a+d. With controlled differentiated
remainders, `Omega=S/R-Sp/R^2+O(R^-3)` gives

```
Box_bar Omega=2Omega/S^2-(2p+6M)Omega^2/S^3+O(Omega^3),
B_*            =2Omega/S^2-2a Omega^2/S^3+O(Omega^3).
```

Matching through Omega^2 necessarily requires

```
d=-3M,  f=R-3M+O(R^-1).
```

This is a necessary expansion of the preferred radial ODE, not proof of its
full smooth solution. Higher coefficients and possible resonances still need
analysis. In particular, imposing f=R exactly leaves a nonzero Box difference
`-6M Omega^2/S^3+...`; changing the height alone cannot repair it. The offset
also differs from -M for the unprojected physical harmonic spatial equation.

## Temporal source must change too

Distinguish physical connection difference
`Cphys^a=g^{bc}(Gamma[g]-Gamma[ghat])^a_bc` from conformal connection difference
`Hbar^a=barg^{bc}(Gamma[barg]-Gamma[barghat])^a_bc`, with Z=0 here.
Both are vectors. For the desired mass-log height and d=-3M,

```
Cphys^{Y0}=-Box_g Y^0=+2M/R^2+O(R^-3),
Cphys^f   =-Box_g f   =-4M/R^2+O(R^-3).
```

Transforming to compact t=Y0-h_hat(f) gives

```
Cphys^t=Cphys^{Y0}-h_hat,f Cphys^f
        =+6M/R^2+O(R^-3),
Cphys^t/Omega^2 -> +6M/S^2.
```

For the conformal-reference equation define the physical-coordinate wave
residual, with k=d(logOmega)/df,

```
E^A=Box_g Y^A+(4g^{AB}-s eta^{AB})partial_B logOmega,
s=g^{AB}eta_AB,
Hbar^A=-E^A/Omega^2.
```

Since `psi'=-2M/R+...`, the temporal row is

```
E^{Y0}=(R^2Fpsi')'/R^2+4Fpsi'f'k
       =(-2M+8M)/R^2+O(R^-3)=6M/R^2+... .
```

The spatial leading residual is `(6M+2d)/R^2`, which vanishes for the preferred
d=-3M branch. Thus `Hbar^f=O(R^-1)` and the compact-time transformation gives

```
Hbar^t -> -6M/S^2.
```

These opposite temporal signs are relative to two different unmodified base
gauges. In the standard source convention

```
barg^{bc}Gamma[barg]^a_bc+2Zbar^a=Fbar_base^a+deltaFbar^a,
```

the necessary compact temporal additions are

```
physical-reference base:   deltaFbar^t -> +6M/S^2,
conformal-reference base:  deltaFbar^t -> -6M/S^2.
```

They are not changes to P storage or to the principal metric derivative terms.
The difference is independently consistent with S01's exact identity
`Cphys/Omega^2=Hbar+U/Omega`: on the preferred radial branch,
`U^t/Omega ->12M/S^2`. Omitting this full live/reference trace term would give
the wrong sign or coefficient.

For a general constant offset d before preferred matching, the analogous
compact leading values are `Cphys^t/Omega^2 ->-2d/S^2` and
`Hbar^t ->+2d/S^2`. Substituting d=-3M yields the results above. These relations
make explicit why the offset and the temporal source cannot be chosen
independently when demanding both stationarity and preferred BoxOmega.

## Minimal spherical source roles, not an admitted prescription

With time-independent Omega, a purely spatial projection cannot alter the
source contraction in the temporal row directly. For any base source define

```
D= barg^{ab} partial_a partial_b Omega
   -Fbar_base^a Omega_a-B_*.
```

On an outer support with nonzero prescribed spatial gradOmega, the algebraic
spatial correction

```
deltaFbar^i=(delta^{ij}Omega_j/|gradOmega|_Euclidean^2) D,
deltaFbar^0=0
```

sets `Box_bar Omega=B_*` on Einstein data. For the conformal-reference base,
`D=barg^{ab} nablaHatBar_a nablaHatBar_b Omega-B_*`. All reference coefficient
derivatives in that Hessian must be retained. This is a metric/reference value
source; no live derivative was inserted in this pencil expression.

The independent temporal correction above is still necessary. On the smooth
preferred stationary branch the conformal-reference spatial correction is
`deltaFbar^r=O(Omega^3)` (since dr/df=Omega^2/L and Hbar^f=O(Omega)), so its
induced Y0 component is only O(Omega). It cannot cancel the nonzero required
compact temporal limit. Changing only the preferred spatial projection while
keeping the old temporal source therefore does not solve this joint outer
stationary problem.

A useful way to display the required leading temporal coefficient without a
fixed BH reference is the *branch-conditional* angular-area surrogate

```
s_area=R/f=Omega R/r=1+3M/f+O(f^-2),
2(s_area-1)/(r Omega) ->6M/S^2.
```

The plus sign matches the physical-reference temporal addition, and the minus
sign the conformal-reference temporal addition. In an orthonormal Euclidean
tangent screen, s_area can be represented as the fourth root of the determinant
of the Penrose spatial metric restricted to that screen. At the exact Minkowski
reference it is one. This demonstrates the algebraic possibility of a source
with a retained Minkowski fixed point; it does not define a radiative mass
measure or select a nonlinear gauge. A smooth outer-only support must be
branched before evaluating 1/(r Omega). Finite values here rely on the stated
stationary falloff; generic live data need not satisfy it.

## Off-constraint and adoption limits

With the actual `+2Zbar` gauge convention, the metric-only spatial projection
above instead gives

```
Box_bar Omega=B_*+2Zbar^a Omega_a
```

off the Einstein sector. Exact preferred Box for arbitrary live Z would require
an additional explicit Z-dependent term. Such a change can alter the stored
Lambda/derivative principal coupling and cannot inherit a complete20 statement
without a new derivation. No Theta, shear, null-jet or Z falloff is imposed here
to hide that issue.

An algebraic live-metric source leaves the finite positive-lapse principal
derivative coefficients formally unchanged, but that alone proves neither
bounded lower-order sources nor a preserved scri manifold. The temporal
surrogate contains an off-branch pole, and the preferred radial projection may
have its own support/parameter/constraint restrictions. Full nonlinear source
identities, regular gauge storage, all coupled principal modes and boundary/
constraint propagation remain separate requirements.

There is therefore no no-go statement for every corrected reference gauge:
the leading mass-log and preferred-Box conditions can be made mutually
consistent at the level of necessary temporal/radial coefficients. There is
also no full stationary existence or preserved-regularity result. Inner-only
changes cannot supply the required outer correction. Retain the Minkowski
reference and the eventual wormhole-to-trumpet BH goal without any BH
fixed-point RHS subtraction; this note authorizes no implementation or run.
