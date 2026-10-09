# Outer inertial-vector damping: mathematical feasibility only

This fresh scratch study changes no evolution equation. It distinguishes a
finite-positive-Omega covariant option from an admissible explicit Cartesian
hyperboloidal implementation. There is no native/global/energy/scri/BH gate.

## Primary source and inspected scope

[Gundlach et al., gr-qc/0504114v2](https://arxiv.org/pdf/gr-qc/0504114v2),
14 July 2005, PDF pages 2--3, equations (2), (3), (13), (16), (17), permits a
nonzero timelike damping vector. Its normal-vector specialization supplies the
familiar 3+1 damping. The inspected hyperboloidal construction in
[1412.3827](https://arxiv.org/pdf/1412.3827), section 2.1, instead uses the
physical normal. Neither inspected source supplies a regular Killing-vector
implementation in this project's variables. This is not an exhaustive
literature exclusion.

The formulas below are independently derived projections. Signature is -+++,
Kij=-Lie_n gammaij/2, and Theta=-n^a Z_a. Let d^a denote the damping vector;
using d avoids confusing the coordinate t with a vector label. The unmodified
covariant Ricci equation has damping -kappa S_ab, where

```
S_ab = d_a Z_b+d_b Z_a-(1+rho)g_ab(d.Z),
D_ab = d_a Z_b+d_b Z_a+rho g_ab(d.Z)  [Einstein trace reversal].
```

## Full 4D projection

Write d=B n+V, n.V=0, B=-n.d>0, J=V^i Z_i. The letter B is the relative
boost, not the gauge cutoff W. The exact contractions are

```
d.Z=-B Theta+J,
S_ij=V_i Z_j+V_j Z_i+(1+rho)gamma_ij(B Theta-J),
D_nn=(2+rho)B Theta-rho J,
D_ni=-B Z_i-V_i Theta.
```

The resulting physical damping terms are

```
Kij_t|damp = -A kappa S_ij,
Theta_t|damp = -A kappa[(2+rho)B Theta-rho J],
Zi_t|damp = -A kappa[B Zi+Vi Theta],
K_t|damp = -A kappa[3(1+rho)B Theta-(1+3rho)J].
```

Here A=alpha/Omega is physical lapse. With P=K-2Theta, the evolved trace source
is +A kappa[(1-rho)B Theta+(1+rho)J]. The source-free geometric equations are
not asserted to be C1: C0's separately documented noncovariant subsidiary terms
remain if only damping is changed. Thus the exact covariant wave equation below
cannot be transferred to the current C0 implementation by this swap alone.

`check_projection.py` constructs a general symmetric spatial metric and the
entire 4D ADM metric, lowers d with that live metric, and verifies these
contractions symbolically. `normal_kernel.cpp` checks B=1,V=0 against actual
ConformalRHS damping differences for 1616 off-constraint jets, four curvatures
and four rho values. Release and ASan/UBSan output bytes agree; maximum
normalized discrepancy is 3.7863931589111304e-15. Chi/gtilde/A damping
differences are exactly zero in this normal specialization.

## Current variables and Killing choice

The project uses gamma_ij=gtilde_ij/(chi Omega^2),
Atilde_ij=Omega chi(Kij-gamma_ij K/3),
Ztilde^i=(Lambda^i-Gamma^i)/2=gtilde^ij Zj.
There is no direct chi/gtilde damping source. In the current variables,

```
Atilde_ij,t|damp = -alpha chi kappa
 [Vi Zj+Vj Zi-(2/3)gammaij J],
Lambda^i_t|damp = -2 alpha kappa/Omega
 [B Ztilde^i+gtilde^ij Vj Theta].
```

For stationary height coordinates T=t+h(R), R=r/Omega, partial_T=partial_t.
On the exact Minkowski reference it is a parallel future unit vector. Its
decomposition is B=A=alpha/Omega and V^i=beta^i. Consequently,

```
P_t|damp = kappa alpha/Omega
 [(1-rho)alpha Theta/Omega+(1+rho)beta.Z],
Theta_t|damp = -kappa alpha/Omega
 [(2+rho)alpha Theta/Omega-rho beta.Z],
Atilde_ij,t|damp = -kappa alpha/Omega^2
 [gtilde_ik beta^k Zj+gtilde_jk beta^k Zi-(2/3)gtildeij beta.Z],
Lambda^i_t|damp = -2 kappa alpha^2 Ztilde^i/Omega^2
                  -2 kappa alpha beta^i Theta/(chi Omega^3).
```

With the existing kappa=kinput/alpha, the final two coefficients reduce to
-2 kinput alpha Ztilde/Omega^2 and -2 kinput beta Theta/(chi Omega^3).
Thus this is not merely the old normal damping with a different scalar rate.
It introduces shear damping/source terms and a triple-pole raw Lambda--Theta
coupling. No extra spacetime-Z variable is algebraically necessary at positive
Omega: coordinate Z_t=-A Theta+beta^i Zi reconstructs it. These are explicit
algebraic sources, so they do not change the time mass matrix or finite-Omega
principal symbol. This statement is a derivative-order inspection, not a new
complete-basis or full20 admission test.

All dimensional factors agree: Omega,alpha,chi,beta,B are dimensionless in
length coordinates, kappa has units 1/length, and P,Theta,Z,Lambda,Atilde have
units 1/length. Their time sources therefore have units 1/length^2.

## Stiffness and raw weights

On the CMC outer reference, chi=1,gtilde=I,

```
r=sqrt(S^2-2aS Omega), alpha=S/a-Omega, beta_n=-r/a,
alpha^2-beta_n^2=Omega^2.
```

At rho=0 with current kappa normalization, the pure damping value block has
Theta eigenvalue -2 kinput alpha/Omega^2 and three Z eigenvalues
-kinput alpha/Omega^2. Its raw radial Theta-to-Z eigenvector ratio is
V_n/A=beta_n/(alpha Omega): weighting Z by Omega controls that ratio but
does not remove the genuine Omega^-2 eigenvalue scale. Coupling through
geometric derivative terms may change full-system modes; this block is not a
full20 eigenvalue or nonlinear-instability proof.

The 100-digit oracle checks 36 S,a,kinput,Omega combinations, including
Omega=1e-40. It confirms

```
lim Omega^2 (Theta<-Theta) = -2 kinput S/a,
lim Omega^2 (Lambda_n<-Lambda_n) = -kinput S/a,
lim Omega^3 (Lambda_n<-Theta) = +2 kinput S/a,
lim Omega^2 (Atilde_nn<-Lambda_n) = +2 kinput S/(3a),
lim Omega^2 (P<-Theta) = +kinput S/a.
```

For S1,a.5,kinput10, applying only the existing illustrative .03 Omega pole
cap to this damping block gives RK3 arguments about -83.56,-312.16,-368.41
on N24/36/48 span2.1 active grids. These are far outside the scalar negative
real RK3 interval. The exact values/amplifications are in projection.json.
This is a block/timestep mismatch demonstration, not an actual candidate
native timestep or propagator calculation. Any explicit attempt needs an
Omega^2 source-stiffness limit or a carefully gated implicit/exact source
treatment, plus compatible weighted fields and continuum boundary analysis.

Arbitrary finite physical Theta and finite stored fields were allowed in the
prior gates. Finiteness of the new raw Lambda source alone would demand
additional cancellations/weights (generically Theta=O(Omega^3) if all other
raw terms stay bounded), which have not been derived for damped solutions.
There is no permission here to impose that falloff. Natural physical spatial
covector weights and boosted normal/inertial components are not uniformly
equivalent in the raw conformal norm as Omega tends to zero.

## Timelikeness is an independent live-data gate

For a live metric,

```
g_phys(partial_t,partial_t)
  =[-alpha^2+(gtildeij/chi)beta^i beta^j]/Omega^2.
```

Positive lapse and spatial SPD do not make this vector timelike. On the
outer reference spatial metric and reference shift, choose
alpha_live^2=beta_ref,n^2-Omega^2 near scri. Then alpha stays positive,
alpha0=alpha_ref0, beta0=beta_ref0, and the vector has norm +1. The usual
leading null relation is still satisfied; a null-residual value condition
cannot decide this second-order-in-Omega timelike sign. This is a gauge-data
counterexample, not a constraint-satisfying solution or invariant-manifold
counterexample. Normalizing partial_t on live data would divide by the square
root of a quantity whose reference value is O(Omega^2), and is undefined on
this spacelike example.

A prescribed convex blend d=(1-v)n+v partial_t with 0<=v<=1 is future
timelike only where both vectors are. On exact Minkowski this holds; gradients
of v and n enter the subsidiary system in the transition. If v=1 in the
outer collar, the live timelike gate above remains. For a stationary black
hole partial_t also changes causal character inside the horizon, making a
normal interior choice necessary but not sufficient for global matching.

## Physical wave-energy claim and possible redesigns

For constant physical kappa, rho=0 and parallel inertial d on exact Minkowski,
the independently translated subsidiary equations are

```
Zi_TT-Delta Zi+kappa Zi_T=0,
Z0_TT-Delta Z0+2kappa Z0_T-kappa div Z=0.
```

Spatial components have the scalar damped-wave energy. The time component is
forced by div Z: its component energy derivative contains
+kappa integral(Z0_T div Z), so a diagonal sum of component energies is not
automatically monotone. The printed equation (19) gives the nonzero-frequency
rho=0 factors (s^2+2kappa s+k^2)(s^2+kappa s+k^2), with transverse copies of
the latter. This supports the familiar linear mode statement but is not a
coercive nonlinear hyperboloidal energy estimate. Furthermore, the project's
kappa=kinput/alpha(r) is spatially varying even on the reference: divergence
of kappa D_ab retains its coefficient gradient. Replacing n by a parallel
vector removes derivatives of n in the outer reference but not that gradient.

Scaling d by Omega would reduce one source-pole order but make the physical
damping vector vanish at scri and introduce gradients of Omega. Its Lambda
Theta source would still be double-pole in these variables. It loses the
constant parallel-vector premise, so it is a different, untested proposal.
A weighted/dual-frame constraint variable and implicit source reformulation
could be researched separately; neither follows from this feasibility check.

Recommendation: do not compile this as the next explicit native candidate.
The finite-Omega covariant construction is legitimate, but the direct swap
does not meet the current timelikeness, raw regularity, source stiffness or
C0-to-covariant subsidiary/energy gates. Minkowski reference and Einstein
sector fixed points are retained because all these damping sources vanish
when Z=Theta=0; no BH/reference fixed-source subtraction is involved.
