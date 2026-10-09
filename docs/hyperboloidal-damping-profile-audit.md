# Constraint-damping profile audit

This report records a rejected candidate, not an accepted stable evolution. It retains
the Minkowski hyperboloidal reference, physical-P lapse/storage, spatial-norm
gauge and C_Z4c=0 geometric equations. Only the existing kappa2 parameter is
prescribed spatially; kappa_input=alpha*kappa1 remains10.

In the exact outer CMC collar,

```
Omega=(S²-r²)/(2aS), alpha=S/a-Omega, alpha*w=-r²/(a²S).
Kcrit=2S/a²,
kappa2=((Kcrit-kappa_input)/kappa_input)*(1-Omega),
kappa_eff=kappa_input*(1+kappa2)
         =Kcrit+(kappa_input-Kcrit)*Omega.
```

For S1,a=.5 this is kappa2=-.2(1-Omega), kappa_eff=8+2Omega. At Omega=1,
kappa2 and all its reference spatial jets vanish in the exact Cauchy interior.
The live spatial-Z damping coefficient remains kappa_input10.

The C0 isotropic physical stress coefficient multiplying Theta*gamma_ij is

```
sigma0=(-2alpha*w-kappa_eff)/Omega
      =-4/a-(kappa_input-Kcrit).
```

On the analytic Minkowski reference it is constant in the exact outer collar,
so its linearized contribution
-2*d_i(sigma0*Theta) has no coefficient-gradient Theta/Omega² term. The repaired
C1 coefficient would similarly be sigma1=-2/a-(kappa_input-Kcrit), but this
candidate adds no C1 terms or C1 connection double pole. The changing damping
coefficient's derivatives must still enter the physical subsidiary equations.
This cancellation alone establishes no energy estimate or stable evolution.

The mapping rho=kappa2 follows the physical equations in
[Gundlach et al., arXiv:gr-qc/0504114v2](https://arxiv.org/pdf/gr-qc/0504114v2).
Independent algebra applied to its printed Eq19 gives the longitudinal quartic

```
p(s)=(s²+kappa*(2+rho)*s+omega²)*(s²+kappa*s+omega²)
     -kappa²*rho*omega².
```

Its full nonzero-frequency constant-coefficient flat Hurwitz range is
-1<rho<=0. For rho>0 and omega²<kappa²rho, p(0)<0 gives a positive real root,
despite the paper's broader prose below Eq23. The proposed profile lies in
[-.2,0]. For the displayed general family the same flat interval requires
kappa_input>=2S/a²>0 on0<=Omega<=1. That frozen inertial calculation does not establish damping for
our variable-coefficient hyperboloidal C0 system. The independent symbolic
receipt preserves the equation/prose distinction and a positive-root example.

The completed full20, principal, coefficient-aware subsidiary, native and
global checks below do not support adoption. No Omega floor, stronger Theta falloff, source
replacement or black-hole fixed point is introduced. Finite-Q/nonlinear scri
closure and the later wormhole-to-trumpet transition remain open requirements.

## Completed local gates

All12 compile/run/check commands pass, with378 recorded inputs unchanged.
The4004-row nonlinear Release/ASan tensor check gives exact Einstein-sector
addition zero, no new double pole and relative parts identity error1.42e-16.
The360 principal cases preserve the complete basis. On1000 coefficient-aware
full20 constraint-chain rows, finest relative discrepancy is7.454e-9; omitting
d_kappa2 gives .00800463. All200 sampled local eight-constraint generators have
negative real roots, with maximum-1.13138363. These are reference-tangent/local
results, distinct from a global stability estimate.

The760 actual finite-Omega primitive matrices retain positive frozen roots
(maximum reference real part3.14600118 with profile versus3.14511812 baseline).
They are not classified as physical subsidiary branches. All96 original sampled scalar RK3 checks at N24/36/48 pole.03 steps pass
for nonpositive-real roots. The original v1/v2 reports incorrectly label
pi*N/2.2 as native Nyquist:2.2 is the global tangent grid span, whereas native
runs use2.1. A separate v3 supplement checks12 actual full20 matrices at
pi*N/2.1, radial and oblique, at independently reconstructed native minimum
Omega values. All nonpositive-real roots pass scalar RK3, with zero excess.
This corrects the label and adds the missing samples; it does not change the
helper, original numerical receipt, native as-built source or evolution. The leading analytic pole matrices at four a
values are rationally reconstructed with maximum floating discrepancy3.553e-15.
Exact rational nullity and Hurwitz checks give five semisimple zeros and negative
nonzero pole roots. This is not a nonlinear Omega=0 assembly or PDE energy proof.

The v2 immutable local report clarifies that reconstruction scope and separates
launch80b83eab from freeze2392ccd1. Its source/numerics/receipt match the preserved
v1 snapshot. Raw small-Omega reference RHS cancellation up to1.08385e-10 remains
recorded; native reference roundoff subtraction is tested independently below.

## Native reference and finite pulse

Six objects are privately rebuilt in9.548 seconds, with369 original sources,
four norm overlays,268 repository dependencies and182 original link objects plus
four Kokkos libraries verified unchanged. The private Cartesian source changes
only its helper include and four kappa2 arguments: live/reference evolution and
live/reference pole diagnostics. There are no C1 additions or gauge changes.
The actual compiled implementation remains27c19d20; build/launch HEAD is2392ccd1.

The reference t=.05 finishes in17.5626 seconds. Three binary64 snapshots have
maximum state drift1.32352966e-14 and final H/M/Z
8.46887e-14/2.72075e-14/1.25324e-15. Initial fields, coordinates and masks are
bitwise equal to the baseline. The finite angular pulse t=.02 finishes in7.4527
seconds:

| Constraint | Profile value | Ratio to same-grid C0 |
| --- | ---: | ---: |
| H | .003079145029 | .989877706 |
| M | .004895142603 | 1.000223480 |
| Z | .001194804433 | .997330588 |

All25 active evolved binary64 fields are finite, lapse/chi positive and metric
SPD; det/trace errors are below1e-12 and every BIN field is its exact binary32
cast. Only output cadence differs between pulse inputs; the endpoint is exactly
t=.02. This integrity check does not accept finite-pulse stability.

The initial launcher assumed dictionary file-index entries; this local gate
uses string hashes. It stopped at a guard before launching an evolution.
The failed launcher and observation are preserved. The corrected checker accepts
both formats, and all gate/build/launch hashes are then verified. No scientific
receipt, as-built source or numerical gate was rewritten.

## Global screen and outcome

The full22 actual native RHS and final-only RK3 tangent agree within7.25e-10
and1.84e-10. Its sparse difference from C0 contains exactly the3280 local
P/Theta<-Theta entries prescribed by -kappa_input*kappa2/Omega; every other
matrix entry and the initial gauge-pulse action are exactly unchanged. Short
projected-continuous propagation at .025/.05 agrees with independent canonical
Taylor action within1.151e-14.

The t2 Arnoldi histories are exploratory, with local truncation checks and no
long independent canonical comparison. They are not finite-RK native runs:

| Gauge | Profile H/M/Z | Ratios to continuous C0 |
| --- | --- | --- |
| Production physical-P | 1.672653 / 1.225100 / .268220 | .97221 / 1.00605 / .98405 |
| Spatial norm | 2.130484 / 1.193051 / .360541 | .97164 / .97759 / .97520 |

The configuration-H1/momentum-L2 component amplification remains47.3266 and
35.6915, versus47.0983 and35.7290. This norm is not an invariant tensor energy.
Shell H is2–3% lower but M/Z is1–3% higher. These marginal mixed changes do not
justify a long native/canonical run or production adoption. The cancellation
of one reference constraint-gradient term is insufficient for stabilization.


## Provenance and later black-hole requirement

The private native executable SHA256 is
`b7e82aea627cd10b89958c993c1429b0322df60bcb62dd1cd7ce46b2a5341b84`,
with build receipt
`062ba8b3103f90233354134ab61a99892eb5cd8a7b61de5cf0d396dc61129cfe`.
That build pins the preserved v2 gate. The later v3 supplement and independent
review correct only the frequency attribution and add actual-native samples;
there is no retrospective rewrite or recompile of the recorded executable.
Production remains implementation27c19d20.

The separate [inner trumpet calibration](hyperboloidal-inner-trumpet-calibration.md)
retains the Minkowski hyperboloidal reference. It derives a necessary inner
shift-rate limit for the selected isotropic stationary solution and shows why
that condition alone does not provide a full stationary driver. Neither this
damping profile nor that analytical calibration demonstrates formation through
the wormhole-to-trumpet transition.

The [immutable archive](validation/hyperboloidal-damping-and-inner-calibration-experiments-20261009/README.md)
contains 279 cataloged files (7,452,954 bytes), with catalog SHA256
`cb9ad3593c2d008a9fb9681a3f15b8f2e51ae404ed13132df3f35723a50fd6a0`.
It includes the exact profile source, local and independent gates, corrected
frequency supplement, native build/launch/audits, global screens and inner
calibration. Large arrays, binaries and objects are represented by hashes and
metadata. Use its verifier read-only; do not rerun collectors or scientific
scripts in frozen paths.
