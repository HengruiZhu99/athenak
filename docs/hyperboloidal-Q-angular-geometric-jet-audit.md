# Angular gauge jets and Einstein geometric freedom

The factored Q/null source with sigma=3 preserves the first quadratic-null
time condition for every compatible gauge second jet at fixed reference
spatial geometry. A larger genuine Einstein family, obtained by spatially
pulling back the same Minkowski foliation while holding prescribed Omega
fixed, fails that condition. No constant sigma cancels both families. This
is a local smooth Taylor time-corner obstruction, not evidence of finite-Omega
amplitude blowup or a global continuum unstable mode.

These are actual linear Cartesian full20 C0 calculations with physical-P
geometry and stored physical Theta, the frozen factored Q/null gauge,
kappa_input=10, S=1 and a=.5,.75,1,2. The outer reference is

```
Omega=(1-r^2)/(2a), h=alpha_ref=(1+r^2)/(2a), beta_ref=-x/a.
```

The source satisfies, on the initial Einstein subsets,
`Box(Omega)=Box_ref(Omega)+sigma*deltaNraw/Omega`, where
`Nraw=|D Omega|^2-omega_n^2`. The required quadratic-null behavior is
`deltaNraw=Omega^2*N2+O(Omega^3)`. Here Qnum=P-3*omega_n; Qnum0 denotes its boundary numerator
coefficient,
not the boundary value of the quotient Q. Neither a sigma3 native evolution nor a
new production gauge is admitted by these audits. Earlier sigma5 global
negative screens remain unchanged.

## Complete gauge-only second jets

With spatial geometry, P and Theta initially reference, decompose lapse
and shift perturbations as

```
delta alpha=h*f,
delta beta=beta_ref*f+b_n*n+b_T,  n.b_T=0.
```

Common rescaling and tangential shift give deltaNraw=0 identically.
Quadratic-null compatibility requires b_n=Omega^2*b2+O(Omega^3). The
compatible gauge second-jet space has dimension31: ten scalar f jets,
twenty tangential-shift jets and one leading normal coefficient. The
overcomplete 70-field actual input map has rank31. An independent exact
9x40 constraint map has rank9 and annihilates it, so the image equals the
entire compatible gauge second-jet kernel at the two sampled orientations
and all four a. This completeness applies to gauge jets at reference
geometry; it does not cover gravitational or arbitrary geometric data.

For arbitrary angular amplitudes the initial actual kernel agrees with
the independently derived transport identity

```
deltaNraw_t=T*partial_r(deltaNraw)+C*deltaNraw,
T=-(r^4+6r^2+1)/(4*a*r),
C=[r^6+(16*sigma-29)*r^4+15*r^2-3]/[4*a*r^2*(r^2-1)].
```

Angular tangential divergences cancel between geometric trace evolution
and the preferred Box source. For sigma3 the weighted quantity
`nN=deltaNraw/Omega^2` has the finite reaction

```
nN_t=T*partial_r(nN)-(r^2+3)*(3*r^2-1)*nN/(4*a*r^2).
```

The reaction tends -2/a at scri. Equivalently,
`N1_t=2*(3-sigma)*N2/a^2`; sigma3 cancels this first null time jet.
This is an initial tangent subsystem on an outer collar, with no origin
extension of the displayed 1/r coefficients and no full energy estimate.

The actual geometric first RHS matches complete independent ADM fields
to 7.56e-12; the raw transport residual is 2.36e-10. Feeding their analytic
Cartesian derivatives back to the actual kernel gives next R0 pole
maximum 5.57e-9. Across 1120 controls/13440 kernel points, normalized N1
errors grow from 1.28e-7 to 6.06e-7 and 1.95e-6 as the extrapolation scale
shrinks .001/.0005/.00025. These retained cancellation errors are not a
finite-difference convergence order. Release/ASan-UBSan Debug outputs
are byte identical.

## Spatial pullbacks enlarge the Einstein subset

Let xi be a time-independent infinitesimal spatial diffeomorphism. Pull
back the stationary physical Minkowski foliation, including lapse and
shift, while keeping the prescribed Omega and reference gauge source
fixed. In the outer collar this gives

```
fOmega=(xi.dOmega)/Omega,
delta bar_gamma_ij=partial_i xi_j+partial_j xi_i-2*fOmega*delta_ij,
delta chi=-2*div(xi)/3+2*fOmega,
delta gtilde_ij=partial_i xi_j+partial_j xi_i-2*div(xi)*delta_ij/3,
delta Lambda_i=Delta(xi_i)+partial_i div(xi)/3,
delta P=delta Theta=delta Atilde=0,
delta alpha=xi.dh-h*fOmega,
delta beta=(x.grad(xi)-xi)/a.
```

These are exact linear Einstein/Z4 initial data: H/M/Z/Theta vanish
identically, rather than just at leading order. A smooth outer cutoff can
extend the local generator into the full reference. The physical geometric
first RHS is stationary and is checked in the actual kernel.

The independent stationary four-dimensional wave identity, with
`w=xi.dOmega`, is

```
deltaBox_stat=xi.d(Box_ref)-Box_ref(w)
              +2*fOmega*Box_ref-2*grad(fOmega).grad(Omega),
deltaNraw=xi.dNraw_ref-2*grad(w).grad(Omega)+2*fOmega*Nraw_ref,
deltaNraw_t=(2*r^2/a^2)*(deltaBox_stat-sigma*deltaNraw/Omega).
```

For xi=Omega*X(x)*e_j put Y=n_j*X(n). Then

```
N2=-2*Y/a,
deltaBox_stat,1=(Delta_S Y-4*Y)/a,
N1_t=(2/a^3)*(Delta_S Y+(2*sigma-4)*Y).
```

The radial xi=Omega*x is the sum of three generated columns and has Y=1.
It requires sigma2, with `N1_t=4*(sigma-2)/a^3`. At a=.5 the actual sigma3
values are 32.00000003/31.99999938/32.00000303 at the three extrapolation
scales; sigma5 gives 96.00000002/95.99999914/96.00000393. The earlier
gauge-only `delta beta=Omega^2*n` instead requires sigma3. Thus one
constant sigma cannot make the smooth quadratic-null ideal containing
both subsets tangent to this actual gauge.

All 120 analytic fields/1920 controls/23040 points initially satisfy the
full measured R0/N0/N1/Qnum0/Theta1/shear conditions. Constraints and R0
are at most 1.87e-14/1.70e-14; N0/N1/Qnum0 are below 2.59e-15. The actual
stationary geometric first RHS error is 5.11e-11 and next R0 maximum 2.14e-8.
The initial four-dimensional normalized Box error is 2.01e-7, with sampled
tracefree Hessian/Omega and scalar-curvature maxima 31.9941/72.
Normalized angular N1-map errors grow 5.74e-7/2.57e-6/8.09e-6 as Omega
shrinks; no numerical zero or convergence order is claimed. All 374 source
inputs stay unchanged and six authoritative commands pass, with identical
Release/ASan-UBSan outputs. Expanded-polynomial cancellation pilots,
temporary compiler warnings and later unreceipted exploration are retained
separately from that authoritative run.

## The conformal-frame distinction

The m1 generator vanishes on scri, but its normal derivative changes the
conformal scale there. At scri `fOmega=-Y/a`; tangential derivatives of xi
vanish, so the induced two-metric changes by
`delta q_AB=-2*fOmega*q_AB=2*Y*q_AB/a`. Fixing that two-metric would
exclude its nonzero-Y conformal-scale directions, including the radial
witness; tangential Y=0 combinations remain. Such a restriction is additional to
Einstein constraints and smoothness, and needs its own evolution,
characteristic and radiative-data justification. It is not equivalent to
requiring every metric deviation to vanish as Omega. No fixed-frame
boundary condition is imposed by this audit.

Failure of this smooth-in-time Taylor ideal does not prove order-one
constraint amplitudes, a vacuum curvature singularity, failure of every
finite-Omega sigma3 prototype or impossibility of another source. Singular
relaxation may produce a temporal boundary layer. The unresolved task is a
derived invariant compatibility hierarchy that retains the required
physical and gauge data, followed by discrete and finite-pulse validation.
Stable finite Minkowski disturbances and the later wormhole-to-trumpet
transition with a Minkowski hyperboloidal reference remain unvalidated.

The frozen angular gate and independent completeness review have indices
`8c569fdf6faf0fa9afff76bebfcd8ea888c5b31734b6fb90b092188f5aab3a65`
and `32f9738ae32e062ea8c5cf75e1160b2a6e1d50bd62b24d8888c54d776449ff37`.
The spatial-pullback gate and independent review have indices
`de31bb51e55c159100a58d14acaae0f00231a7d6128bc6c3771d535b75ff5662`
and `8d5d3d364612b389eee2cdf1d6572097eb82a9f3c3e0e01eb2c1a2374d6c6598`.
Production remains implementation `27c19d20696ea6dd4704032c51dfd026218f64f2`.

The byte-preserved [compact archive](validation/hyperboloidal-Q-angular-geometric-jet-experiments-20261009-v2/README.md)
contains 112 cataloged blobs totaling 6,325,154 bytes, including 32 finite
JSON files. Catalog SHA256 is
`90c2e6e9e9dad18c2a5ce0109b927554e31dd10657d9e1593964d9001e21dd1d`.
Seven individual payloads above 1 MiB, binaries and arrays remain metadata-only.
A root README preparation path error stopped the first collection; its exact
collector, observed-error receipt and partial output are preserved outside the
tracked destination. This fresh v2 archive includes that collector and receipt
without rerunning or modifying scientific inputs.
