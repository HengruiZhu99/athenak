# Conformal Minkowski reference: conditional stationary Schwarzschild end

This is an independent pencil/source-only derivation, 2026-10-09. No CAS,
numerical differentiation, kernel/native query, operator or propagation was
used. The conclusion is conditional on stationary, spherical, Killing-aligned,
exact-Einstein data with a smooth nondegenerate future conformal end. It is not
a finite-time instability claim or an obstruction to every source modification.
The later user target remains a wormhole-to-trumpet black hole with the
Minkowski hyperboloidal reference retained throughout.

## Definitions and the complete conformal transformation

Use signature -+++, physical metrics `g=Omega^-2 barg` and
`ghat=Omega^-2 barghat`, the same prescribed scalar Omega for both, and
`Zphysical^a=Omega^2 Zbar^a`. Define

```
Hphys^a = g^{bc}(Gamma[g]-Gamma[ghat])^a_bc+2 Zphysical^a,
Hbar^a  = barg^{bc}(Gamma[barg]-Gamma[barghat])^a_bc+2 Zbar^a,
s       = barg^{bc} barghat_bc.
```

Contracting the full connection transformation gives

```
Hphys^a/Omega^2 = Hbar^a
  +(4 barg^{ad}-s barghat^{ad}) Omega_d/Omega.
```

The two Kronecker-delta terms cancel in the connection difference. Contracting
the remaining metric term produces the dimension four and the trace s.
Replacing it by `-2(bargInv-barghatInv)dOmega/Omega` is incorrect in general.
The existing physical-reference helper/source convention is S01--S02. This
note examines the distinct condition `Hbar=0`; no implementation is selected.

In the reference inertial coordinates `Y^A`, the reference physical metric is
eta and `barghat=Omega(f)^2 eta`, where `f=|Y^I|`. On the exact-Einstein sector
Z=0, the conformal-reference condition is equivalently

```
Box_g Y^A+(4 g^{AB}-s eta^{AB}) partial_B logOmega(f)=0,
s=g^{AB} eta_AB.
```

Here g^{AB} are physical inverse-metric components in Y coordinates. This
equivalence fixes the sign: `Box_g Y^A=-g^{BC}Gamma[g]^A_BC`.

## Stationary spherical variables, without fixing the live radial map

Use Schwarzschild areal radius R, mass M>0 and normalized Killing time T:

```
F=1-2M/R,
g=-F dT^2+F^-1 dR^2+R^2 dOmega_sphere^2,
T=t+h_BH(R),
Y^0=t+h_hat(f(R))=T+psi(R),
Y^I=f(R)n^I,
psi=h_hat(f)-h_BH.
```

The live radial map f is not assumed equal to R or to the physical-reference
harmonic radius. Assume f>0, f'>0 near infinity. In the exact outer Minkowski
CMC branch of the retained reference,

```
f=r/Omega(r), Omega(r)=(S^2-r^2)/(2aS), a,S>0,
Omega(f)=S/(sqrt(f^2+a^2)+a),
h_hat(f)=sqrt(f^2+a^2)+constant,
k(f)=d(logOmega)/df=-1/f+a/[f sqrt(f^2+a^2)].
```

In particular `k=-1/f+a/f^2+O(f^-4)`. No transition-layer height integral enters
these exact outer equations. R is areal radius, not the isotropic wormhole
radius used in some earlier initial-data constructors.

The required inverse components are

```
g^{00}=-F^-1+F psi'^2,
g^{0I}=F psi' f' n^I,
g^{IJ}=F f'^2 n^I n^J+(f^2/R^2)(delta^{IJ}-n^I n^J),
s=F^-1-F psi'^2+F f'^2+2f^2/R^2.
```

Primes on f and psi mean d/dR; k is differentiated with respect to f. The
angular l=1 term in `Box_g(f n^I)` is essential and retains the non-scalar
spatial coordinate nature of the problem.

## Exact temporal first integral

The temporal row becomes

```
(R^2 F psi')'/R^2+4F psi'f' k=0,
therefore R^2 F psi' Omega(f)^4=D,
```

where D is a constant of length squared. If `f/R->lambda>0`, then
`Omega~S/(lambda R)`. Any bounded psi' forces D=0 by taking R to infinity.
Conversely D!=0 would require `psi'~D lambda^4 R^2/S^4` and hence an unbounded
height derivative. With f' bounded, that violates an asymptotically spacelike
future end. Thus in the stated branch

```
psi'=0,  h_BH=h_hat(f)+constant.
```

This is stronger than the earlier physical-reference temporal flux
`R^2F psi'=constant`. It is still a stationary statement, not a restriction
proven to propagate for arbitrary finite-time data.

## Necessary spatial balance, including logarithmic maps

Before imposing D=0, the radial spatial row is

```
(R^2F f')'-2f
 +R^2 k [3F f'^2-F^-1+F psi'^2-2f^2/R^2]=0.
```

After D=0 the psi'^2 term vanishes. The leading R term for `f~lambda R`
requires `lambda^2=1`; positivity selects lambda=1. The same leading choice
follows from `h_BH=h_hat(f)` and the future hyperboloidal height derivative.

Permit even a logarithmic radial map as a stronger test than smooth compact
coefficients would normally allow:

```
f=R+c log(R/R_*)+d+O(log(R)/R),
f'=1+c/R+O(log(R)/R^2),
```

with correspondingly differentiated remainders. Here R_* is a fixed positive
length and c,d have length units. The two parts of the spatial equation have
the respective order-one/logarithmic balances

```
(R^2F f')'-2f
 =c-2M-2c log(R/R_*)-2d+o(1),

R^2 k [3F f'^2-F^-1-2f^2/R^2]
 =-6c+8M+4c log(R/R_*)+4d+o(1).
```

The subleading a/f^2 in k contributes only o(1), since the bracket's constant
term vanishes at lambda=1. Summing gives

```
2c log(R/R_*)-5c+6M+2d=0 at leading order,
therefore c=0, d=-3M.
```

These are necessary coefficients of a hypothetical stationary spatial branch,
not a construction or existence proof for its full ODE solution. The -3M
offset is distinct from the -M offset of the physical-reference harmonic
spatial equation. Neither offset supplies a logarithmic height term.

## Smooth future-end requirement and the obstruction

The physical induced radial metric is

```
gamma_RR=F^-1-F h_BH'^2.
```

For a nondegenerate smooth future hyperboloidal end, write
`gamma_RR=a_B^2/R^2+O(R^-3)`, with a_B>0. The future branch then necessarily has

```
h_BH'=F^-1-a_B^2/(2R^2)+O(R^-3),
h_BH=R+2M log(R/ell)+constant+O(R^-1).
```

This is the usual mass logarithm derived directly from the Schwarzschild null
balance, independently of any inner trumpet choice. Since the temporal row
requires `h_BH=h_hat(f)+constant`, such an end would require c=2M in the
logarithmic f ansatz. The spatial row instead requires c=0. For M>0 they are
incompatible.

Already within a smooth prescribed-Omega radial representation, set
`rho(r)=Omega(r)R(r)` and suppose rho is smooth with rho(S)>0. Then
`f=r/Omega` has an expansion `f=lambda R+d+O(R^-1)` with no mass logarithm.
The temporal condition therefore excludes the desired height directly. The
stronger logarithmic-map calculation shows that merely allowing a log in f
does not resolve the full pair of conformal-reference gauge equations.

On the formal c=0 branch, `h_BH'=1+O(R^-2)`, so
`gamma_RR=4M/R+O(R^-2)`. Under `R=r/Omega+O(1)` the Penrose radial coefficient
has a 1/Omega pole. This is a failed smooth-end condition, not proof that the
stationary ODE has no asymptotically null, weaker-completion solution.

The reference M=0, f=R, h_BH=h_hat(R) is an exact branch. A positive constant
Killing-time rescaling does not restore the missing mass logarithm: if
`T=kappa t+h_BH`, temporal flux still forces constant psi, smooth future leading
balance gives `kappa lambda=1`, and the analogous spatial expansion requires
c=0 (and d=-3lambda M). No arbitrary time normalization is used to evade the
obstruction.

## Scope and later alternatives

The assumptions are exact Einstein Z=0, stationarity, spherical symmetry,
Killing-aligned time, prescribed same reference/Omega map, a monotone regular
radial mapping with controlled asymptotic differentiated remainders, and a
smooth nondegenerate future spacelike conformal end. Removing one of these
assumptions requires a new analysis. No conclusion about finite-time Minkowski
pulse behavior, nonspherical evolution, regularity of every possible gauge,
off-constraint closure, or BH formation follows.

The same Minkowski reference can be retained while changing live-field
lower-order/asymptotic source terms. An inner-only blend cannot change these
unchanged outer stationary equations. A future outer correction must match
both temporal and radial rows, preserve the Minkowski fixed point, and undergo
its own complete constrained20/principal, source, null/shear/Z4/Theta and
boundary checks. Simply choosing conformal rather than physical reference
connections does not establish a stationary smooth massive end.

All equations in this note are independent algebraic deductions from the
stated metrics and source conventions. S01 points to the earlier primary-source
context (background-connection wave maps and conformal wave sources); no
published theorem is invoked for this conditional Schwarzschild obstruction.
No source correction or production/BH adoption is proposed here.
